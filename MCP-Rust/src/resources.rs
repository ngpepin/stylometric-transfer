//! Self-contained embedded resources.
//!
//! Every configuration artifact this server needs is compiled into the binary
//! from `MCP-Rust/resources/`. Nothing is read from the Python implementation at
//! runtime: the data files live in this crate and are embedded with
//! `include_str!`, so the binary is fully standalone.

use once_cell::sync::Lazy;
use serde_json::{json, Value};

/// Bundled default LLM connection config (`resources/config.llm.json`).
///
/// A copy of the Python `config.llm.json` with `api_key` left blank, so the
/// binary needs no external config file to start. A real key comes from
/// `--config PATH` or a local `./config.llm.json`.
pub const LLM_CONFIG_JSON: &str = include_str!("../resources/config.llm.json");

/// Bundled prompt templates (`resources/prompts.json`).
pub const PROMPTS_JSON: &str = include_str!("../resources/prompts.json");
/// Bundled runtime tunables (`resources/config.tunables.json`).
pub const TUNABLES_JSON: &str = include_str!("../resources/config.tunables.json");
/// Bundled locale spelling rules (`resources/config.local_spelling_rules.json`).
pub const SPELLING_RULES_JSON: &str = include_str!("../resources/config.local_spelling_rules.json");
/// Bundled global avoid list, one entry per line (`resources/config.avoid.txt`).
pub const AVOID_TXT: &str = include_str!("../resources/config.avoid.txt");
/// Bundled common-word list, one entry per line (`resources/config.common_words.txt`).
#[allow(dead_code)]
pub const COMMON_WORDS_TXT: &str = include_str!("../resources/config.common_words.txt");
/// Bundled entity blacklist, one entry per line (`resources/config.entity_blacklist.txt`).
#[allow(dead_code)]
pub const ENTITY_BLACKLIST_TXT: &str = include_str!("../resources/config.entity_blacklist.txt");

/// Parsed prompts, loaded once.
pub static PROMPTS: Lazy<Value> =
    Lazy::new(|| serde_json::from_str(PROMPTS_JSON).expect("embedded prompts.json is valid JSON"));

/// Parsed tunables, loaded once.
pub static TUNABLES: Lazy<Value> = Lazy::new(|| {
    serde_json::from_str(TUNABLES_JSON).expect("embedded config.tunables.json is valid JSON")
});

/// Parsed locale spelling rules, loaded once.
pub static SPELLING_RULES: Lazy<Value> = Lazy::new(|| {
    serde_json::from_str(SPELLING_RULES_JSON)
        .expect("embedded config.local_spelling_rules.json is valid JSON")
});

/// Return a prompt group (`fingerprint`, `apply`, `repair_json`, ...) by name.
pub fn prompt_group(name: &str) -> Option<&'static Value> {
    PROMPTS.get(name)
}

/// Return a single prompt template field, e.g. `prompt("apply", "system")`.
pub fn prompt(group: &str, field: &str) -> Option<&'static str> {
    PROMPTS.get(group)?.get(field)?.as_str()
}

/// Return the embedded tunables document.
pub fn tunables() -> &'static Value {
    &TUNABLES
}

/// Return a nested tunables value by key path.
pub fn tunable(path: &[&str]) -> Option<&'static Value> {
    let mut cur = &*TUNABLES;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

/// Read a numeric tunable with a fallback.
pub fn tunable_f64(path: &[&str], default: f64) -> f64 {
    tunable(path).and_then(Value::as_f64).unwrap_or(default)
}

/// Read an integer tunable with a fallback.
pub fn tunable_u64(path: &[&str], default: u64) -> u64 {
    tunable(path).and_then(Value::as_u64).unwrap_or(default)
}

/// Split a one-entry-per-line list into trimmed non-empty lines.
fn lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

/// Embedded global avoid list (literal entries, no normalization).
pub fn avoid_entries() -> Vec<String> {
    lines(AVOID_TXT)
}

/// Embedded common-word list (`word` or `word <zipf>`).
///
/// Bundled for parity with the Python resources and reserved for the
/// rare-word / lexical-avoidance signal port; not yet consumed by
/// `measurements.rs`.
#[allow(dead_code)]
pub fn common_words() -> Vec<String> {
    lines(COMMON_WORDS_TXT)
}

/// Embedded entity blacklist entries.
///
/// Bundled for parity with the Python resources and reserved for
/// proper-name suppression during phrase validation; not yet consumed.
#[allow(dead_code)]
pub fn entity_blacklist() -> Vec<String> {
    lines(ENTITY_BLACKLIST_TXT)
}

/// Embedded avoid list as a JSON array (handy for prompt injection).
pub fn avoid_json() -> Value {
    json!(avoid_entries())
}
