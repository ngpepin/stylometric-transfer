//! Rewrite orchestration: mask -> measure -> LLM -> repair -> post-process -> score.
//!
//! Ports the rewrite loop of `apply_fingerprint.py` without depending on the
//! Python tree. All prompts come from the embedded `resources/prompts.json`,
//! and every deterministic rule reads the embedded tunables/spelling data.

use crate::filters;
use crate::llm::{chat_completions, parse_json_lenient, LlmConfig, Message};
use crate::measurements::compute_measurements;
use crate::postprocess::{apply_mandatory_humanizer, load_mandatory};
use crate::resources;
use crate::scoring::{calibrated_style_match_probability, compute_style_compliance};
use crate::spelling::SpellingRules;
use crate::text;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Narrative voice overrides (`--1st-person` / `--2nd-person` / `--3rd-person`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonOverride {
    First,
    Second,
    Third,
}

impl PersonOverride {
    /// Human-readable label used in prompts and reports.
    pub fn label(self) -> &'static str {
        match self {
            PersonOverride::First => "first-person",
            PersonOverride::Second => "second-person",
            PersonOverride::Third => "third-person",
        }
    }
}

/// Options controlling a rewrite run.
#[derive(Debug, Clone, Default)]
pub struct ApplyOptions {
    /// Force fiction / non-fiction handling; `None` auto-detects.
    pub fiction: Option<bool>,
    /// Force narrative voice regardless of the fingerprint.
    pub person: Option<PersonOverride>,
    /// Author name used with `Third` person prompts.
    pub author_name: Option<String>,
    /// Extra style-loop retries after the first attempt.
    pub max_retries: Option<u32>,
}

/// Collected placeholder mappings from every masking pass.
type Masks = BTreeMap<String, String>;

/// Apply every masking pass and merge the produced placeholder tables.
fn mask_all(markdown: &str) -> (String, Masks) {
    let mut masks: Masks = BTreeMap::new();
    let mut current = markdown.to_string();

    let (stripped, images) = filters::strip_base64_images(&current);
    masks.extend(images);
    current = stripped;

    let (stripped, blocks) = filters::mask_non_voice_blocks(&current);
    masks.extend(blocks);
    current = stripped;

    let (stripped, citations) = filters::mask_inline_citations(&current);
    masks.extend(citations);
    current = stripped;

    let (stripped, quotes) = filters::mask_quoted_passages(&current);
    masks.extend(quotes);
    current = stripped;

    (current, masks)
}

/// Build the rewrite prompt for one chunk.
fn build_apply_prompt(
    fingerprint: &Value,
    measurements: &Value,
    chunk: &str,
    options: &ApplyOptions,
) -> Vec<Message> {
    let system = resources::prompt("apply", "system")
        .unwrap_or("You rewrite text to match an explicit style fingerprint while preserving meaning.");
    let template = resources::prompt("apply", "user").unwrap_or("");
    let fingerprint_text =
        serde_json::to_string_pretty(fingerprint).unwrap_or_else(|_| "{}".to_string());
    let measurements_text =
        serde_json::to_string_pretty(measurements).unwrap_or_else(|_| "{}".to_string());

    let mut voice = String::new();
    if let Some(person) = options.person {
        voice.push_str(&format!("\nVOICE_OVERRIDE: {}", person.label()));
        if person == PersonOverride::Third {
            if let Some(name) = &options.author_name {
                voice.push_str(&format!("\nAUTHOR_NAME: {name}"));
            }
        }
    }

    let user = format!(
        "{template}\n\nFINGERPRINT:\n{fingerprint}\n\nINPUT_MEASUREMENTS:\n{measurements}\n{voice}\n\nTEXT_TO_REWRITE:\n{chunk}",
        template = template,
        fingerprint = fingerprint_text,
        measurements = measurements_text,
        voice = voice,
        chunk = chunk,
    );
    vec![Message::system(system), Message::user(user)]
}

/// Extract `final_markdown` from a parsed rewrite response.
fn extract_markdown(value: &Value, fallback: &str) -> String {
    value
        .get("final_markdown")
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

/// Split text into chunks that stay under the configured token budget.
fn chunk_text(text: &str, max_tokens: usize) -> Vec<String> {
    if text::estimate_tokens(text) <= max_tokens {
        return vec![text.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for paragraph in text::split_paragraphs(text) {
        let candidate = if current.is_empty() {
            paragraph.clone()
        } else {
            format!("{current}\n\n{paragraph}")
        };
        if text::estimate_tokens(&candidate) > max_tokens && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
            current = paragraph;
        } else {
            current = candidate;
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

/// Rewrite `markdown` to match `fingerprint`.
///
/// Returns the rewritten Markdown plus a report containing deviations, the
/// deterministic post-processing changes, compliance score, and calibrated
/// style-match probability.
pub fn apply_fingerprint(
    fingerprint: &Value,
    markdown: &str,
    config: &LlmConfig,
    options: &ApplyOptions,
) -> Result<Value, String> {
    if !fingerprint.is_object() {
        return Err("fingerprint must be a JSON object".to_string());
    }

    let (masked, masks) = mask_all(markdown);
    let max_tokens = resources::tunable_u64(&[ "chunking", "max_input_tokens"], 5750) as usize;
    let chunks = chunk_text(&masked, max_tokens.max(512));

    let retries = options
        .max_retries
        .unwrap_or_else(|| resources::tunable_u64(&[ "style_retry", "max_retries"], 1) as u32);

    let mut rewritten = String::new();
    let mut deviations: Vec<Value> = Vec::new();
    let mut last_score = 1.0_f64;

    for chunk in &chunks {
        let measurements = compute_measurements(chunk);
        let mut attempt = 0u32;
        let mut best_text = chunk.clone();
        let mut best_score = -1.0_f64;
        loop {
            let messages = build_apply_prompt(fingerprint, &measurements, chunk, options);
            let completion = chat_completions(config, &messages, None)?;
            let candidate = match parse_json_lenient(&completion.content) {
                Some(value) => {
                    if let Some(items) = value.get("deviations").and_then(Value::as_array) {
                        deviations.extend(items.iter().cloned());
                    }
                    extract_markdown(&value, chunk)
                }
                None => completion.content.clone(),
            };
            let score = compute_style_compliance(fingerprint, &candidate)
                .get("score")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if score > best_score {
                best_score = score;
                best_text = candidate;
            }
            attempt += 1;
            if score >= resources::tunable_f64(&[ "style_retry", "threshold"], 0.59)
                || attempt > retries
            {
                break;
            }
        }
        last_score = best_score;
        rewritten.push_str(&best_text);
        rewritten.push_str("\n\n");
    }

    let restored = filters::restore_placeholders(rewritten.trim_end(), &masks);

    let mandatory = load_mandatory(resources::tunables());
    let spelling_rules = SpellingRules::from_json(&resources::SPELLING_RULES);
    let preserve_quotes = !filters::detect_fiction(
        &restored,
        resources::tunable_u64(&["fiction_detection", "quote_span_min"], 6) as usize,
        resources::tunable_f64(&["fiction_detection", "quoted_ratio_min"], 0.03),
        resources::tunable_f64(&["fiction_detection", "quote_para_ratio_min"], 0.2),
        resources::tunable_f64(&["fiction_detection", "quoted_ratio_force"], 0.08),
    );
    let processed = apply_mandatory_humanizer(
        markdown,
        &restored,
        &mandatory,
        &spelling_rules,
        preserve_quotes,
    );

    let compliance = compute_style_compliance(fingerprint, &processed.text);
    let probability = calibrated_style_match_probability(
        last_score,
        text::estimate_tokens(&processed.text) as u64,
        0.75,
        0.12,
        250,
    );
    let changes: Vec<Value> = processed
        .changes
        .iter()
        .map(|c| json!({"rule_or_field": c.rule_or_field, "reason": c.reason, "count": c.count}))
        .collect();

    Ok(json!({
        "final_markdown": processed.text,
        "deviations": deviations,
        "postprocess_changes": changes,
        "compliance": compliance,
        "style_match_probability": probability,
        "chunks": chunks.len()
    }))
}
