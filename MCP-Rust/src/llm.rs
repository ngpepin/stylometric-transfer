//! OpenAI-compatible chat client with retry/backoff.

use serde_json::{json, Value};
use std::time::Duration;

/// Minimal OpenAI-compatible configuration container.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u64,
    pub temperature: f64,
    pub timeout_seconds: u64,
    pub max_prompt_tokens: u64,
    pub max_retries: u32,
    pub backoff_base_seconds: f64,
    pub backoff_max_seconds: f64,
    pub extra_headers: Vec<(String, String)>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "http://localhost:4045/v1".to_string(),
            model: "gpt-4o-mini".to_string(),
            max_tokens: 6000,
            temperature: 0.2,
            timeout_seconds: 300,
            max_prompt_tokens: 6000,
            max_retries: 6,
            backoff_base_seconds: 2.0,
            backoff_max_seconds: 20.0,
            extra_headers: Vec::new(),
        }
    }
}
impl LlmConfig {
    /// Load configuration from a JSON file (same shape as `config.llm.json`).
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
        Self::parse(&raw).map_err(|e| format!("invalid config JSON {}: {e}", path.display()))
    }

    /// Parse configuration from a JSON string (same shape as `config.llm.json`).
    pub fn parse(raw: &str) -> Result<Self, String> {
        let data: Value = serde_json::from_str(raw).map_err(|e| format!("invalid config JSON: {e}"))?;
        let mut cfg = LlmConfig::default();
        if let Some(v) = data.get("api_key").and_then(Value::as_str) {
            cfg.api_key = v.to_string();
        }
        if let Some(v) = data.get("base_url").and_then(Value::as_str) {
            cfg.base_url = v.trim_end_matches('/').to_string();
        }
        if let Some(v) = data.get("model").and_then(Value::as_str) {
            cfg.model = v.to_string();
        }
        if let Some(v) = data.get("max_tokens").and_then(Value::as_u64) {
            cfg.max_tokens = v;
            cfg.max_prompt_tokens = v;
        }
        if let Some(v) = data.get("max_prompt_tokens").and_then(Value::as_u64) {
            cfg.max_prompt_tokens = v;
        }
        if let Some(v) = data.get("temperature").and_then(Value::as_f64) {
            cfg.temperature = v;
        }
        if let Some(v) = data.get("timeout_seconds").and_then(Value::as_u64) {
            cfg.timeout_seconds = v;
        }
        if let Some(v) = data.get("max_retries").and_then(Value::as_u64) {
            cfg.max_retries = v as u32;
        }
        if let Some(v) = data.get("backoff_base_seconds").and_then(Value::as_f64) {
            cfg.backoff_base_seconds = v;
        }
        if let Some(v) = data.get("backoff_max_seconds").and_then(Value::as_f64) {
            cfg.backoff_max_seconds = v;
        }
        if let Some(map) = data.get("extra_headers").and_then(Value::as_object) {
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    cfg.extra_headers.push((k.clone(), s.to_string()));
                }
            }
        }
        Ok(cfg)
    }

    /// Embedded default configuration: a copy of the Python `config.llm.json`
    /// with the API key left blank, compiled into the binary.
    pub fn embedded_default() -> Self {
        Self::parse(crate::resources::LLM_CONFIG_JSON)
            .expect("embedded resources/config.llm.json is valid JSON")
    }

    /// Temperature scaled by a perplexity-profile multiplier, clamped to sane bounds.
    pub fn scaled_temperature(&self, multiplier: f64) -> f64 {
        (self.temperature * multiplier).clamp(0.0, 2.0)
    }
}

/// A single chat message.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    /// Construct an `assistant` role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"role": self.role, "content": self.content})
    }
}

/// Successful completion: content plus optional usage payload.
#[derive(Debug, Clone)]
pub struct Completion {
    pub content: String,
    pub usage: Option<Value>,
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect()
}
/// POST to an OpenAI-compatible `/chat/completions` endpoint with retries.
pub fn chat_completions(
    cfg: &LlmConfig,
    messages: &[Message],
    temperature: Option<f64>,
) -> Result<Completion, String> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let payload = json!({
        "model": cfg.model,
        "messages": messages.iter().map(|m| m.to_json()).collect::<Vec<_>>(),
        "max_tokens": cfg.max_tokens,
        "temperature": temperature.unwrap_or(cfg.temperature),
    });
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(cfg.timeout_seconds)))
        .http_status_as_error(false)
        .build()
        .into();

    let mut last_err = String::from("no attempts made");
    for attempt in 0..=cfg.max_retries {
        let mut req = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", cfg.api_key));
        for (k, v) in &cfg.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }
        match req.send_json(&payload) {
            Ok(mut resp) => {
                let status = resp.status().as_u16();
                let body = resp
                    .body_mut()
                    .read_to_string()
                    .unwrap_or_else(|e| format!("read error: {e}"));
                if status == 429 || (500..=599).contains(&status) {
                    last_err = format!("LLM call failed ({status}): {}", truncate(&body, 500));
                } else if status >= 400 {
                    return Err(format!("LLM call failed ({status}): {}", truncate(&body, 1000)));
                } else {
                    let data: Value = serde_json::from_str(&body)
                        .map_err(|e| format!("invalid LLM JSON response: {e}"))?;
                    let content = data
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            format!("unexpected LLM response shape: {}", truncate(&body, 500))
                        })?;
                    return Ok(Completion {
                        content: content.to_string(),
                        usage: data.get("usage").cloned(),
                    });
                }
            }
            Err(e) => last_err = format!("LLM transport error: {e}"),
        }
        if attempt < cfg.max_retries {
            let backoff = (cfg.backoff_base_seconds * 2f64.powi(attempt as i32))
                .min(cfg.backoff_max_seconds);
            let jitter = rand::random::<f64>() * backoff * 0.2;
            std::thread::sleep(Duration::from_millis(((backoff + jitter) * 1000.0) as u64));
        }
    }
    Err(format!(
        "LLM call failed after {} attempts: {last_err}",
        cfg.max_retries + 1
    ))
}


/// Parse model output into JSON, tolerating code fences and surrounding prose.
///
/// Attempt order matches the Python strategy: strict parse first, then a fenced
/// block, then the outermost `{...}` span.
pub fn parse_json_lenient(content: &str) -> Option<Value> {
    let trimmed = content.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }

    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        let after = after.strip_prefix("json").unwrap_or(after);
        if let Some(end) = after.find("```") {
            if let Ok(value) = serde_json::from_str::<Value>(after[..end].trim()) {
                return Some(value);
            }
        }
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<Value>(&trimmed[start..=end]).ok()
}
