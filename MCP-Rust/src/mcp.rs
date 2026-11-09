//! MCP server: JSON-RPC 2.0 over stdio or a minimal HTTP listener.
//!
//! Exposes the stylometric pipeline as MCP tools. Fully self-contained: prompts,
//! tunables, spelling rules, and word lists are embedded from `resources/`.

use crate::apply::{apply_fingerprint, ApplyOptions, PersonOverride};
use crate::fingerprint::{build_fingerprint, Document, FingerprintOptions};
use crate::llm::LlmConfig;
use crate::scoring::{
    calibrated_style_match_probability, compute_fingerprint_similarity, compute_style_compliance,
};
use crate::store;
use crate::text;
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

/// MCP protocol revision implemented by this server.
pub const PROTOCOL_VERSION: &str = "2024-11-05";
/// Server name reported during `initialize`.
pub const SERVER_NAME: &str = "stylometric-mcp";

/// Resolve the fingerprint store directory (`STYLOMETRIC_MCP_STORE_DIR` wins).
pub fn resolve_store_dir() -> PathBuf {
    match std::env::var("STYLOMETRIC_MCP_STORE_DIR") {
        Ok(value) if !value.trim().is_empty() => {
            let path = PathBuf::from(value);
            store::store_dir(Some(path.as_path()))
        }
        _ => store::store_dir(None),
    }
}

fn arg_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn arg_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
}

/// Collect documents from `documents[]` or a single `text` string.
fn documents_from(args: &Value) -> Result<Vec<Document>, String> {
    if let Some(items) = args.get("documents").and_then(Value::as_array) {
        let mut docs = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let text = item
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("documents[{index}].text is required"))?;
            let title = item.get("title").and_then(Value::as_str).unwrap_or("");
            docs.push(Document {
                title: title.to_string(),
                text: text.to_string(),
            });
        }
        if !docs.is_empty() {
            return Ok(docs);
        }
    }
    let text = arg_str(args, "text").ok_or("provide either `documents` or `text`")?;
    let title = arg_str(args, "title").unwrap_or_else(|| "corpus".to_string());
    Ok(vec![Document { title, text }])
}

/// Load a fingerprint supplied inline or referenced by GUID.
fn fingerprint_from(args: &Value) -> Result<Value, String> {
    if let Some(value) = args.get("fingerprint") {
        if value.is_object() {
            return Ok(value.clone());
        }
    }
    let guid = arg_str(args, "guid").ok_or("provide either `fingerprint` or `guid`")?;
    store::load(&resolve_store_dir(), &guid)
}

/// Map a person override string to its enum value.
fn person_from(value: Option<String>) -> Option<PersonOverride> {
    match value.as_deref() {
        Some("first") | Some("1st") | Some("1st-person") => Some(PersonOverride::First),
        Some("second") | Some("2nd") | Some("2nd-person") => Some(PersonOverride::Second),
        Some("third") | Some("3rd") | Some("3rd-person") => Some(PersonOverride::Third),
        _ => None,
    }
}

/// Tool catalogue returned by `tools/list`.
pub fn tool_definitions() -> Value {
    json!({"tools": [
        {
            "name": "fingerprint",
            "description": "Build an explicit, interpretable JSON style fingerprint from a corpus.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
                            "required": ["text"]
                        }
                    },
                    "text": {"type": "string"},
                    "title": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "author_name": {"type": "string"},
                    "fiction": {"type": "boolean"},
                    "excerpt_words": {"type": "integer", "minimum": 200},
                    "store": {"type": "boolean"},
                    "allow_scaffold": {"type": "boolean"}
                }
            }
        },
        {
            "name": "apply",
            "description": "Rewrite Markdown to match a fingerprint while preserving meaning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string"},
                    "fingerprint": {"type": "object"},
                    "guid": {"type": "string"},
                    "fiction": {"type": "boolean"},
                    "person": {"type": "string", "enum": ["first", "second", "third"]},
                    "author_name": {"type": "string"},
                    "max_retries": {"type": "integer", "minimum": 0}
                },
                "required": ["markdown"]
            }
        },
        {
            "name": "rate",
            "description": "Score how well a text matches a fingerprint (compliance plus calibrated probability).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "fingerprint": {"type": "object"},
                    "guid": {"type": "string"}
                },
                "required": ["text"]
            }
        },
        {
            "name": "similarity",
            "description": "Compare two fingerprints using interpretable component metrics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fingerprint_a": {"type": "object"},
                    "fingerprint_b": {"type": "object"},
                    "guid_a": {"type": "string"},
                    "guid_b": {"type": "string"}
                }
            }
        },
        {
            "name": "get_fingerprint",
            "description": "Fetch a stored fingerprint by GUID.",
            "inputSchema": {
                "type": "object",
                "properties": {"guid": {"type": "string"}},
                "required": ["guid"]
            }
        }
    ]})
}

/// Execute one MCP tool call.
pub fn call_tool(name: &str, args: &Value, config: &LlmConfig) -> Result<Value, String> {
    match name {
        "fingerprint" => {
            let documents = documents_from(args)?;
            let options = FingerprintOptions {
                profile_id: arg_str(args, "profile_id").unwrap_or_else(|| "profile".to_string()),
                author_name: arg_str(args, "author_name").unwrap_or_else(|| "author".to_string()),
                fiction: arg_bool(args, "fiction"),
                excerpt_words: args
                    .get("excerpt_words")
                    .and_then(Value::as_u64)
                    .unwrap_or(1200) as usize,
                allow_scaffold: arg_bool(args, "allow_scaffold").unwrap_or(false),
            };
            let fingerprint = build_fingerprint(&documents, &options, config)?;
            let guid = if arg_bool(args, "store").unwrap_or(false) {
                let dir = resolve_store_dir();
                let meta = json!({
                    "profile_id": options.profile_id,
                    "author_name": options.author_name,
                    "document_count": documents.len()
                });
                Some(store::save(&dir, &fingerprint, &meta)?)
            } else {
                None
            };
            let mut result = json!({"document_count": documents.len()});
            result["fingerprint"] = fingerprint;
            if let Some(guid) = guid {
                result["guid"] = json!(guid);
            }
            Ok(result)
        }
        "apply" => {
            let markdown = arg_str(args, "markdown").ok_or("`markdown` is required")?;
            let fingerprint = fingerprint_from(args)?;
            let options = ApplyOptions {
                fiction: arg_bool(args, "fiction"),
                person: person_from(arg_str(args, "person")),
                author_name: arg_str(args, "author_name"),
                max_retries: args.get("max_retries").and_then(Value::as_u64).map(|v| v as u32),
            };
            apply_fingerprint(&fingerprint, &markdown, config, &options)
        }
        "rate" => {
            let input = arg_str(args, "text").ok_or("`text` is required")?;
            let fingerprint = fingerprint_from(args)?;
            let compliance = compute_style_compliance(&fingerprint, &input);
            let score = compliance.get("score").and_then(Value::as_f64).unwrap_or(0.0);
            let probability = calibrated_style_match_probability(
                score,
                text::estimate_tokens(&input) as u64,
                0.75,
                0.12,
                250,
            );
            Ok(json!({
                "style_match_probability": probability,
                "compliance": compliance
            }))
        }
        "similarity" => {
            let a = match args.get("fingerprint_a") {
                Some(value) => value.clone(),
                None => {
                    let guid = arg_str(args, "guid_a").ok_or("provide `fingerprint_a` or `guid_a`")?;
                    store::load(&resolve_store_dir(), &guid)?
                }
            };
            let b = match args.get("fingerprint_b") {
                Some(value) => value.clone(),
                None => {
                    let guid = arg_str(args, "guid_b").ok_or("provide `fingerprint_b` or `guid_b`")?;
                    store::load(&resolve_store_dir(), &guid)?
                }
            };
            compute_fingerprint_similarity(&a, &b, None)
        }
        "get_fingerprint" => {
            let guid = arg_str(args, "guid").ok_or("`guid` is required")?;
            store::load(&resolve_store_dir(), &guid)
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

fn error_response(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Handle one JSON-RPC request. Returns `None` for notifications.
pub fn handle_request(request: &Value, config: &LlmConfig) -> Option<Value> {
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let id = request.get("id").cloned();
    if id.is_none() || method.starts_with("notifications/") {
        return None;
    }
    let id = id.unwrap();
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));

    let outcome: Result<Value, (i64, String)> = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"tools": {"listChanged": false}}
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tool_definitions()),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            match call_tool(name, &arguments, config) {
                Ok(value) => Ok(json!({
                    "content": [{"type": "text", "text": serde_json::to_string_pretty(&value).unwrap_or_default()}],
                    "structuredContent": value,
                    "isError": false
                })),
                Err(message) => Ok(json!({
                    "content": [{"type": "text", "text": message}],
                    "isError": true
                })),
            }
        }
        other => Err((-32601, format!("method not found: {other}"))),
    };

    Some(match outcome {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => error_response(&id, code, &message),
    })
}

/// Serve newline-delimited JSON-RPC over stdin/stdout until EOF.
pub fn serve_stdio(config: &LlmConfig) -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("stdin error: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(trimmed) {
            Ok(request) => handle_request(&request, config),
            Err(e) => Some(error_response(
                &Value::Null,
                -32700,
                &format!("parse error: {e}"),
            )),
        };
        if let Some(response) = response {
            writeln!(stdout, "{response}").map_err(|e| format!("stdout error: {e}"))?;
            stdout.flush().map_err(|e| format!("stdout error: {e}"))?;
        }
    }
    Ok(())
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|pos| pos + 4)
}

fn header_value(head: &str, name: &str) -> Option<String> {
    for line in head.lines() {
        if let Some((key, value)) = line.split_once(':') {
            if key.trim().eq_ignore_ascii_case(name) {
                return Some(value.trim().to_string());
            }
        }
    }
    None
}

fn handle_http_connection(stream: &mut TcpStream, config: &LlmConfig) -> Result<(), String> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_header_end(&buffer) {
            break pos;
        }
        let read = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if read == 0 {
            return Ok(());
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > 8 * 1024 * 1024 {
            return Err("request too large".to_string());
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let length: usize = header_value(&head, "Content-Length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    while buffer.len() < header_end + length {
        let read = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let end = (header_end + length).min(buffer.len());
    let body = &buffer[header_end..end];

    let payload = match serde_json::from_slice::<Value>(body) {
        Ok(request) => match handle_request(&request, config) {
            Some(response) => response.to_string(),
            None => "{}".to_string(),
        },
        Err(e) => error_response(&Value::Null, -32700, &format!("parse error: {e}")).to_string(),
    };
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream
        .write_all(reply.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Serve JSON-RPC over a minimal HTTP/1.1 listener (one request per connection).
pub fn serve_http(bind: &str, config: &LlmConfig) -> Result<(), String> {
    let listener = TcpListener::bind(bind).map_err(|e| format!("cannot bind {bind}: {e}"))?;
    println!("{SERVER_NAME} listening on http://{bind}");
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_http_connection(&mut stream, config) {
                    eprintln!("request error: {e}");
                }
            }
            Err(e) => eprintln!("connection error: {e}"),
        }
    }
    Ok(())
}
