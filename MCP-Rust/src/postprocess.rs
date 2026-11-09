//! Deterministic post-processing: mandatory humanizer rules that must not
//! depend on the model (em dashes, quote normalization, heading casing).

use crate::filters;
use crate::headings;
use crate::spelling::{enforce_local_spelling, SpellingRules};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;

/// Mandatory humanizer settings resolved from `config.tunables.json`.
#[derive(Debug, Clone)]
pub struct HumanizerMandatory {
    pub avoid_em_dashes: bool,
    pub emoji_policy: String,
    pub normalize_double_quotes: bool,
    pub normalize_single_quotes: bool,
    pub heading_case_normalization: String,
    pub heading_case_by_level: BTreeMap<usize, String>,
    pub preserve_proper_name_case: bool,
    pub sanitize_heading_qualifiers: bool,
    pub heading_qualifier_allowlist: Vec<String>,
    pub force_local_spelling_rules: String,
    pub force_local_spelling_llm: String,
}

impl Default for HumanizerMandatory {
    fn default() -> Self {
        Self {
            avoid_em_dashes: true,
            emoji_policy: "remove".to_string(),
            normalize_double_quotes: true,
            normalize_single_quotes: true,
            heading_case_normalization: "automatic".to_string(),
            heading_case_by_level: BTreeMap::new(),
            preserve_proper_name_case: true,
            sanitize_heading_qualifiers: false,
            heading_qualifier_allowlist: Vec::new(),
            force_local_spelling_rules: "none".to_string(),
            force_local_spelling_llm: "none".to_string(),
        }
    }
}

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z][A-Za-z']*").unwrap());
static ATX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s{0,3}#{1,8})\s+(.*)$").unwrap());

pub const EM_DASH: char = '\u{2014}';

/// Replace em dashes with a comma-safe alternative, preserving code fences.
pub fn enforce_no_em_dashes(text: &str) -> (String, usize) {
    let mut changes = 0usize;
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            out.push_str(line);
            continue;
        }
        let mut replaced = String::with_capacity(line.len());
        for ch in line.chars() {
            if ch == EM_DASH {
                changes += 1;
                replaced.push_str(", ");
            } else {
                replaced.push(ch);
            }
        }
        out.push_str(&replaced);
    }
    (out, changes)
}

/// Convert curly double quotes to straight quotes.
pub fn enforce_straight_double_quotes(text: &str) -> (String, usize) {
    let mut changes = 0usize;
    let out: String = text
        .chars()
        .map(|c| match c {
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => {
                changes += 1;
                '"'
            }
            other => other,
        })
        .collect();
    (out, changes)
}

/// Convert curly single quotes/apostrophes to straight equivalents.
pub fn enforce_straight_single_quotes(text: &str) -> (String, usize) {
    let mut changes = 0usize;
    let out: String = text
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => {
                changes += 1;
                '\''
            }
            other => other,
        })
        .collect();
    (out, changes)
}

/// Remove/replace emoji according to policy (`remove` or `replace`).
pub fn enforce_emoji_policy(text: &str, policy: &str) -> (String, usize) {
    if policy == "none" || policy.is_empty() {
        return (text.to_string(), 0);
    }
    let mut changes = 0usize;
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let is_emoji = matches!(ch as u32, 0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F);
        if is_emoji {
            changes += 1;
            if policy == "replace" {
                out.push(' ');
            }
        } else {
            out.push(ch);
        }
    }
    (out, changes)
}
/// Deterministic spelling enforcement that leaves protected regions verbatim.
///
/// Mirrors `enforce_local_spelling_guarded`: base64 images, non-voice blocks
/// (blockquotes, references, footnotes), inline citations, and optionally
/// multi-word quotes are masked, the spelling pass runs on the remainder, and
/// placeholders are restored in reverse order.
pub fn enforce_local_spelling_guarded(
    text: &str,
    locale: &str,
    rules: &SpellingRules,
    preserve_multiword_quotes: bool,
) -> (String, usize) {
    if text.is_empty() || locale == "none" || rules.is_empty() {
        return (text.to_string(), 0);
    }
    let (mut guarded, base64_map) = filters::strip_base64_images(text);
    let (next, frozen_map) = filters::mask_non_voice_blocks(&guarded);
    guarded = next;
    let (next, quote_map) = if preserve_multiword_quotes {
        filters::mask_quoted_passages(&guarded)
    } else {
        (guarded, BTreeMap::new())
    };
    guarded = next;
    let (next, citation_map) = filters::mask_inline_citations(&guarded);
    guarded = next;
    let (mut spelled, replacements) = enforce_local_spelling(&guarded, locale, rules);
    for map in [&citation_map, &quote_map, &frozen_map, &base64_map] {
        spelled = filters::restore_placeholders(&spelled, map);
    }
    (spelled, replacements)
}
fn normalize_locale(value: &Value) -> String {
    let mode = value.as_str().unwrap_or("").trim().to_lowercase();
    match mode.as_str() {
        "none" | "canadian" | "british" | "australian" | "us" => mode,
        _ => "none".to_string(),
    }
}

fn parse_by_level(raw: Option<&Value>) -> BTreeMap<usize, String> {
    let mut out = BTreeMap::new();
    let defaults: [(usize, &str); 8] = [
        (1, "title-case"),
        (2, "sentence-case"),
        (3, "identical"),
        (4, "automatic"),
        (5, "caps"),
        (6, "lower"),
        (7, "automatic"),
        (8, "automatic"),
    ];
    let allowed = [
        "automatic",
        "identical",
        "unchanged",
        "title-case",
        "sentence-case",
        "caps",
        "lower",
    ];
    let mut configured: BTreeMap<usize, String> = BTreeMap::new();
    if let Some(map) = raw.and_then(Value::as_object) {
        for (key, value) in map {
            let k = key.trim().to_lowercase();
            let k = k.strip_prefix('h').unwrap_or(&k).to_string();
            let level: usize = match k.parse() {
                Ok(n) if (1..=8).contains(&n) => n,
                _ => continue,
            };
            let mut v = value.as_str().unwrap_or("").trim().to_lowercase();
            v = match v.as_str() {
                "upper" | "uppercase" => "caps".to_string(),
                "title" => "title-case".to_string(),
                "sentence" => "sentence-case".to_string(),
                other => other.to_string(),
            };
            if allowed.contains(&v.as_str()) {
                configured.insert(level, v);
            }
        }
    }
    for (level, default) in defaults {
        out.insert(
            level,
            configured
                .get(&level)
                .cloned()
                .unwrap_or_else(|| default.to_string()),
        );
    }
    out
}

/// Resolve mandatory humanizer settings from a tunables document.
pub fn load_mandatory(tunables: &Value) -> HumanizerMandatory {
    let mandatory = tunables.get("humanizer_mandatory");
    let mandatory = match mandatory {
        Some(v) if v.is_object() => v,
        _ => return HumanizerMandatory::default(),
    };
    let mut conf = HumanizerMandatory::default();
    conf.avoid_em_dashes = mandatory
        .get("avoid_em_dashes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    conf.emoji_policy = mandatory
        .get("emoji_policy")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .to_string();
    conf.normalize_double_quotes = mandatory
        .get("normalize_double_quotes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    conf.normalize_single_quotes = mandatory
        .get("normalize_single_quotes")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut mode = mandatory
        .get("heading_case_normalization")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if mode.is_empty() {
        mode = match mandatory.get("allow_heading_case_changes").and_then(Value::as_bool) {
            Some(true) => "automatic".to_string(),
            Some(false) => "identical".to_string(),
            None => String::new(),
        };
    }
    if !matches!(mode.as_str(), "automatic" | "identical" | "by-level") {
        mode = "by-level".to_string();
    }
    conf.heading_case_normalization = mode;
    conf.heading_case_by_level = parse_by_level(mandatory.get("heading_case_by_level"));
    conf.preserve_proper_name_case = mandatory
        .get("preserve_proper_name_case")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    match mandatory.get("sanitize_heading_qualifiers") {
        Some(Value::Object(obj)) => {
            conf.sanitize_heading_qualifiers = obj
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            conf.heading_qualifier_allowlist = obj
                .get("allowlist")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
        }
        Some(other) => {
            conf.sanitize_heading_qualifiers = other.as_bool().unwrap_or(false);
        }
        None => {}
    }
    let legacy = mandatory
        .get("force_local_spelling")
        .map(normalize_locale)
        .unwrap_or_else(|| "none".to_string());
    conf.force_local_spelling_llm = match mandatory.get("force_local_spelling_LLM") {
        Some(v) => normalize_locale(v),
        None => legacy.clone(),
    };
    conf.force_local_spelling_rules = match mandatory.get("force_local_spelling_rules") {
        Some(v) => normalize_locale(v),
        None => legacy,
    };
    conf
}
/// One deterministic post-processing edit, recorded for the deviations report.
#[derive(Debug, Clone)]
pub struct PostprocessChange {
    pub rule_or_field: String,
    pub reason: String,
    pub count: usize,
}

/// Result of the deterministic humanizer pass.
#[derive(Debug, Clone)]
pub struct PostprocessResult {
    pub text: String,
    pub changes: Vec<PostprocessChange>,
}

fn compile_allowlist(patterns: &[String]) -> Vec<Regex> {
    patterns
        .iter()
        .filter(|p| !p.trim().is_empty())
        .filter_map(|p| Regex::new(&format!("(?i){}", p)).ok())
        .collect()
}

/// Apply the mandatory humanizer rules in the same order as
/// `apply_fingerprint.py`, returning the rewritten text plus change records.
pub fn apply_mandatory_humanizer(
    source_markdown: &str,
    rewritten_markdown: &str,
    conf: &HumanizerMandatory,
    spelling_rules: &SpellingRules,
    preserve_multiword_quotes: bool,
) -> PostprocessResult {
    let mut text = rewritten_markdown.to_string();
    let mut changes: Vec<PostprocessChange> = Vec::new();

    if conf.avoid_em_dashes {
        let (next, removed) = enforce_no_em_dashes(&text);
        text = next;
        if removed > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "punctuation.em_dashes".to_string(),
                reason: "Em dashes removed to satisfy humanizer_mandatory constraint.".to_string(),
                count: removed,
            });
        }
    }

    if !conf.emoji_policy.is_empty() && conf.emoji_policy != "none" {
        let (next, removed) = enforce_emoji_policy(&text, &conf.emoji_policy);
        text = next;
        if removed > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "emoji_policy".to_string(),
                reason: format!("Applied emoji policy '{}'.", conf.emoji_policy),
                count: removed,
            });
        }
    }

    if conf.normalize_double_quotes {
        let (next, replaced) = enforce_straight_double_quotes(&text);
        text = next;
        if replaced > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "punctuation.double_quotes".to_string(),
                reason:
                    "Curly double quotes normalized to straight quotes (humanizer_mandatory)."
                        .to_string(),
                count: replaced,
            });
        }
    }

    if conf.normalize_single_quotes {
        let (next, replaced) = enforce_straight_single_quotes(&text);
        text = next;
        if replaced > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "punctuation.single_quotes".to_string(),
                reason:
                    "Curly single quotes normalized to straight quotes (humanizer_mandatory)."
                        .to_string(),
                count: replaced,
            });
        }
    }

    if conf.sanitize_heading_qualifiers {
        let allowlist = compile_allowlist(&conf.heading_qualifier_allowlist);
        let (next, sanitized) = headings::enforce_heading_qualifiers(&text, &allowlist);
        text = next;
        if sanitized > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "structure.heading_qualifiers".to_string(),
                reason: "Heading qualifiers removed when safe (humanizer_mandatory).".to_string(),
                count: sanitized,
            });
        }
    }

    if conf.force_local_spelling_rules != "none" {
        let (next, replacements) = enforce_local_spelling_guarded(
            &text,
            &conf.force_local_spelling_rules,
            spelling_rules,
            preserve_multiword_quotes,
        );
        text = next;
        if replacements > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "orthography.local_spelling".to_string(),
                reason: format!(
                    "Enforced {} spelling via deterministic rules regardless of fingerprint.",
                    conf.force_local_spelling_rules
                ),
                count: replacements,
            });
        }
    }

    if headings::heading_case_is_deterministic(
        &conf.heading_case_normalization,
        &conf.heading_case_by_level,
    ) {
        let (next, edits) = headings::enforce_heading_case_normalization_from_source(
            source_markdown,
            &text,
            &conf.heading_case_normalization,
            &conf.heading_case_by_level,
            conf.preserve_proper_name_case,
        );
        text = next;
        if edits > 0 {
            changes.push(PostprocessChange {
                rule_or_field: "structure.heading_case".to_string(),
                reason: format!(
                    "Applied heading case normalization mode '{}' (humanizer_mandatory).",
                    conf.heading_case_normalization
                ),
                count: edits,
            });
        }
    }

    PostprocessResult { text, changes }
}
