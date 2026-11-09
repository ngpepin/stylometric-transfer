//! `stylometric-mcp` — stylometric profiling and author-conditioned style
//! transfer, exposed over MCP.
//!
//! This crate is a self-contained Rust implementation: prompts, tunables,
//! spelling rules, and word lists are embedded from `MCP-Rust/resources/`, and
//! nothing is read from the Python implementation at runtime.

mod apply;
mod filters;
mod fingerprint;
mod headings;
mod llm;
mod mcp;
mod measurements;
mod postprocess;
mod resources;
mod scoring;
mod spelling;
mod store;
mod text;

use std::path::PathBuf;

const USAGE: &str = "\
stylometric-mcp — stylometric profiling and author-conditioned style transfer

USAGE:
  stylometric-mcp serve [--config PATH]        MCP server over stdio (default)
  stylometric-mcp http [--bind ADDR] [--config PATH]
  stylometric-mcp tools                        Print the MCP tool catalogue
  stylometric-mcp fingerprint --input FILE [--title T] [--profile-id ID]
                             [--author-name N] [--fiction|--non-fiction]
                             [--out FILE] [--store] [--scaffold]
  stylometric-mcp apply --fingerprint FILE --input FILE [--out FILE]
                        [--1st-person|--2nd-person|--3rd-person]
                        [--author-name N]
  stylometric-mcp --help | --version

CONFIG:
  --config PATH selects an LLM connection file. When omitted, a local
  ./config.llm.json is used if present, otherwise the config compiled into
  the binary (a copy of config.llm.json with an empty api_key) applies.
  All style data (prompts, tunables, spelling rules, word lists) is embedded
  in the binary; nothing is read from the Python tree at runtime.
";

fn next_value(args: &[String], index: &mut usize) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| "missing value for flag".to_string())
}

fn flag_present(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

/// Load LLM connection config (explicit `--config`, else `./config.llm.json`).
fn load_config(args: &[String]) -> Result<llm::LlmConfig, String> {
    let mut path: Option<PathBuf> = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--config" || args[index] == "-c" {
            path = Some(PathBuf::from(next_value(args, &mut index)?));
        }
        index += 1;
    }
    match path {
        Some(path) => llm::LlmConfig::load(&path),
        None => {
            // A local `./config.llm.json` still wins (so operators can drop a
            // key next to the binary), otherwise fall back to the config that
            // is compiled into the binary itself.
            let local = PathBuf::from("config.llm.json");
            if local.exists() {
                llm::LlmConfig::load(&local)
            } else {
                Ok(llm::LlmConfig::embedded_default())
            }
        }
    }
}

fn read_input(args: &[String]) -> Result<(String, Option<PathBuf>), String> {
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" | "-i" => input = Some(PathBuf::from(next_value(args, &mut index)?)),
            "--out" | "-o" => out = Some(PathBuf::from(next_value(args, &mut index)?)),
            _ => {}
        }
        index += 1;
    }
    let path = input.ok_or("--input is required")?;
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok((content, out))
}

fn write_output(out: Option<PathBuf>, content: &str) -> Result<(), String> {
    match out {
        Some(path) => std::fs::write(&path, content)
            .map_err(|e| format!("cannot write {}: {e}", path.display())),
        None => {
            println!("{content}");
            Ok(())
        }
    }
}

fn run_fingerprint(args: &[String], config: &llm::LlmConfig) -> Result<(), String> {
    let (body, out) = read_input(args)?;
    let mut title = "corpus".to_string();
    let mut profile_id = "profile".to_string();
    let mut author_name = "author".to_string();
    let mut should_store = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--title" => title = next_value(args, &mut index)?,
            "--profile-id" => profile_id = next_value(args, &mut index)?,
            "--author-name" => author_name = next_value(args, &mut index)?,
            "--store" => should_store = true,
            _ => {}
        }
        index += 1;
    }
    let fiction = if flag_present(args, "--fiction") {
        Some(true)
    } else if flag_present(args, "--non-fiction") {
        Some(false)
    } else {
        None
    };
    let documents = vec![fingerprint::Document {
        title,
        text: body,
    }];
    let options = fingerprint::FingerprintOptions {
        profile_id,
        author_name,
        fiction,
        excerpt_words: 1200,
        allow_scaffold: flag_present(args, "--scaffold"),
    };
    let value = fingerprint::build_fingerprint(&documents, &options, config)?;
    if should_store {
        let dir = mcp::resolve_store_dir();
        let meta = serde_json::json!({
            "profile_id": options.profile_id,
            "author_name": options.author_name
        });
        let guid = store::save(&dir, &value, &meta)?;
        eprintln!("stored fingerprint as {guid}");
    }
    write_output(
        out,
        &serde_json::to_string_pretty(&value).unwrap_or_default(),
    )
}

fn run_apply(args: &[String], config: &llm::LlmConfig) -> Result<(), String> {
    let (markdown, out) = read_input(args)?;
    let mut fingerprint_path: Option<PathBuf> = None;
    let mut person: Option<apply::PersonOverride> = None;
    let mut author_name: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--fingerprint" | "-f" => {
                let mut path = next_value(args, &mut index)?;
                if !path.ends_with(".json") {
                    path.push_str(".json");
                }
                fingerprint_path = Some(PathBuf::from(path));
            }
            "--1st-person" => person = Some(apply::PersonOverride::First),
            "--2nd-person" => person = Some(apply::PersonOverride::Second),
            "--3rd-person" => person = Some(apply::PersonOverride::Third),
            "--author-name" => author_name = Some(next_value(args, &mut index)?),
            _ => {}
        }
        index += 1;
    }
    let path = fingerprint_path.ok_or("--fingerprint is required")?;
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let fingerprint: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid fingerprint JSON: {e}"))?;
    let options = apply::ApplyOptions {
        fiction: None,
        person,
        author_name,
        max_retries: None,
    };
    let result = apply::apply_fingerprint(&fingerprint, &markdown, config, &options)?;
    let rendered = result
        .get("final_markdown")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    write_output(out, &rendered)?;
    if let Some(score) = result
        .get("compliance")
        .and_then(|value| value.get("score"))
        .and_then(|value| value.as_f64())
    {
        eprintln!("style compliance: {score:.3}");
    }
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("missing command (try --help)".to_string());
    }
    match args[0].as_str() {
        "--help" | "-h" | "help" => {
            print!("{USAGE}");
            return Ok(());
        }
        "--version" | "-V" => {
            println!("{} {}", mcp::SERVER_NAME, env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {}
    }
    let config = load_config(&args[1..])?;
    match args[0].as_str() {
        "serve" | "stdio" => mcp::serve_stdio(&config),
        "http" => {
            let mut bind = "127.0.0.1:8765".to_string();
            let mut index = 1;
            while index < args.len() {
                if args[index] == "--bind" {
                    bind = next_value(args, &mut index)?;
                }
                index += 1;
            }
            mcp::serve_http(&bind, &config)
        }
        "tools" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&mcp::tool_definitions()).unwrap_or_default()
            );
            Ok(())
        }
        "fingerprint" => run_fingerprint(&args[1..], &config),
        "apply" => run_apply(&args[1..], &config),
        other => Err(format!("unknown command: {other}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(message) = run(&args) {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}
