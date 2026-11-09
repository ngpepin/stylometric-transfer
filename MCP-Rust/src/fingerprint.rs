//! Fingerprint construction: corpus -> local measurements -> LLM synthesis.
//!
//! Ports the synthesis half of `fingerprint_style.py` with no dependency on the
//! Python tree. Prompts come from the embedded `resources/prompts.json`.

use crate::filters::{self, QuoteMode};
use crate::llm::{chat_completions, parse_json_lenient, LlmConfig, Message};
use crate::measurements::compute_measurements;
use crate::resources;
use crate::text;
use serde_json::{json, Map, Value};

/// Options controlling a fingerprint run.
#[derive(Debug, Clone)]
pub struct FingerprintOptions {
    pub profile_id: String,
    pub author_name: String,
    /// Force fiction / non-fiction handling; `None` auto-detects.
    pub fiction: Option<bool>,
    /// Approximate word budget for representative excerpts.
    pub excerpt_words: usize,
    /// Fall back to a deterministic scaffold when the LLM fails or returns
    /// unusable JSON. Defaults to `false` so failures surface to the caller.
    pub allow_scaffold: bool,
}

impl Default for FingerprintOptions {
    fn default() -> Self {
        Self {
            profile_id: "profile".to_string(),
            author_name: "author".to_string(),
            fiction: None,
            excerpt_words: 1200,
            allow_scaffold: false,
        }
    }
}

/// A single corpus document.
#[derive(Debug, Clone)]
pub struct Document {
    pub title: String,
    pub text: String,
}

/// Take a deterministic, evenly-spaced sample of sentences.
///
/// Sampling across the whole document avoids over-weighting the opening.
fn representative_excerpt(text: &str, max_words: usize) -> String {
    let sentences = text::split_sentences(text);
    if sentences.is_empty() || max_words == 0 {
        return String::new();
    }
    let stride = (sentences.len() / 24).max(1);
    let mut out: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut idx = 0usize;
    while idx < sentences.len() && used < max_words {
        let sentence = sentences[idx].trim();
        let count = text::word_count(sentence);
        if count > 0 {
            out.push(sentence.to_string());
            used += count;
        }
        idx += stride;
    }
    out.join(" ")
}

/// Decide fiction vs non-fiction from tunables thresholds and the corpus.
fn is_fiction(corpus: &str, forced: Option<bool>) -> bool {
    if let Some(value) = forced {
        return value;
    }
    let span = resources::tunable_u64(&["fiction_detection", "quote_span_min"], 6) as usize;
    let ratio = resources::tunable_f64(&["fiction_detection", "quoted_ratio_min"], 0.03);
    let para_ratio = resources::tunable_f64(&["fiction_detection", "quote_para_ratio_min"], 0.2);
    let force_ratio = resources::tunable_f64(&["fiction_detection", "quoted_ratio_force"], 0.08);
    filters::detect_fiction(corpus, span, ratio, para_ratio, force_ratio)
}

/// Compute the corpus-level measurement bundle (author voice only).
pub fn corpus_measurements(documents: &[Document], fiction: bool) -> Value {
    let mut joined = String::new();
    for doc in documents {
        joined.push_str(&doc.text);
        joined.push_str("\n\n");
    }
    let mode = if fiction {
        QuoteMode::Fiction
    } else {
        QuoteMode::NonFiction
    };
    let voice = filters::filter_author_voice(&joined, mode);
    compute_measurements(&voice)
}

/// Build the synthesis prompt from the embedded template.
fn build_fingerprint_prompt(
    options: &FingerprintOptions,
    documents: &[Document],
    measurements: &Value,
    excerpts: &str,
    fiction: bool,
) -> Vec<Message> {
    let system = resources::prompt("fingerprint", "system")
        .unwrap_or("You synthesize explicit, interpretable style fingerprints as JSON.");
    let schema_hint = resources::prompt_group("fingerprint")
        .and_then(|g| g.get("schema_hint").cloned())
        .unwrap_or_else(|| json!({}));
    let schema_text = serde_json::to_string_pretty(&schema_hint).unwrap_or_default();
    let user_template = resources::prompt("fingerprint", "user").unwrap_or("");

    let titles: Vec<Value> = documents
        .iter()
        .map(|d| json!({"title": d.title, "words_est": text::word_count(&d.text)}))
        .collect();
    let measurements_text =
        serde_json::to_string_pretty(measurements).unwrap_or_else(|_| "{}".to_string());

    let user = format!(
        "{template}\n\nPROFILE_ID: {profile}\nAUTHOR_NAME: {author}\nFICTION: {fiction}\n\nDOCUMENTS:\n{docs}\n\nMEASUREMENTS (verbatim, do not alter):\n{measurements}\n\nGLOBAL_AVOID:\n{avoid}\n\nEXCERPTS:\n{excerpts}\n\nSCHEMA_HINT:\n{schema}",
        template = user_template,
        profile = options.profile_id,
        author = options.author_name,
        fiction = if fiction { "true" } else { "false" },
        docs = serde_json::to_string_pretty(&titles).unwrap_or_default(),
        measurements = measurements_text,
        avoid = serde_json::to_string(&resources::avoid_json()).unwrap_or_default(),
        excerpts = excerpts,
        schema = schema_text,
    );
    vec![Message::system(system), Message::user(user)]
}

/// Deterministic fingerprint used when the LLM is unavailable or unusable.
fn scaffold_fingerprint(
    options: &FingerprintOptions,
    documents: &[Document],
    measurements: &Value,
) -> Value {
    let words: usize = documents.iter().map(|d| text::word_count(&d.text)).sum();
    json!({
        "schema_version": "1.0.0",
        "profile_id": options.profile_id,
        "metadata": {
            "author_name": options.author_name,
            "generator": "stylometric-mcp (deterministic scaffold)",
            "corpus": {
                "document_count": documents.len(),
                "size": {"words_est": words},
                "documents": documents
                    .iter()
                    .map(|d| json!({"title": d.title, "words_est": text::word_count(&d.text)}))
                    .collect::<Vec<_>>()
            }
        },
        "measurements": measurements,
        "targets": {},
        "lexicon": {
            "preferred_words": [],
            "preferred_phrases": [],
            "avoid_words": resources::avoid_entries()
        },
        "templates": {},
        "controls": {
            "do_not_imitate_living_author": true,
            "copyright_sensitive": true,
            "priority_order": ["meaning", "style", "fluency"]
        },
        "validators": {"weights": {}},
        "derived_instructions": {}
    })
}

/// Build a style fingerprint for `documents`.
///
/// Falls back to the deterministic scaffold when the LLM fails, so callers
/// always receive a valid, inspectable JSON fingerprint.
pub fn build_fingerprint(
    documents: &[Document],
    options: &FingerprintOptions,
    config: &LlmConfig,
) -> Result<Value, String> {
    if documents.is_empty() {
        return Err("no documents supplied".to_string());
    }
    let corpus: String = documents
        .iter()
        .map(|d| d.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let fiction = is_fiction(&corpus, options.fiction);
    let measurements = corpus_measurements(documents, fiction);

    let per_doc = options.excerpt_words.max(200) / documents.len().max(1);
    let mut parts: Vec<String> = Vec::new();
    for doc in documents {
        let excerpt = representative_excerpt(&doc.text, per_doc);
        if !excerpt.is_empty() {
            parts.push(format!("### {}\n{}", doc.title, excerpt));
        }
    }
    let excerpts = parts.join("\n\n");
    let messages = build_fingerprint_prompt(options, documents, &measurements, &excerpts, fiction);

    let completion = match chat_completions(config, &messages, None) {
        Ok(completion) => completion,
        Err(error) if options.allow_scaffold => {
            eprintln!("warning: LLM call failed ({error}); using deterministic scaffold");
            return Ok(scaffold_fingerprint(options, documents, &measurements));
        }
        Err(error) => return Err(error),
    };
    {
        match parse_json_lenient(&completion.content) {
            Some(mut value) if value.is_object() => {
                if let Some(map) = value.as_object_mut() {
                    map.insert("measurements".to_string(), measurements.clone());
                    map.entry("schema_version")
                        .or_insert_with(|| json!("1.0.0"));
                    map.entry("profile_id")
                        .or_insert_with(|| json!(options.profile_id));
                    let metadata = map
                        .entry("metadata")
                        .or_insert_with(|| json!({}));
                    if let Some(meta) = metadata.as_object_mut() {
                        meta.entry("author_name")
                            .or_insert_with(|| json!(options.author_name));
                        meta.entry("extraction").or_insert_with(|| {
                            json!({"tunables_snapshot": resources::tunables().clone()})
                        });
                    }
                }
                Ok(value)
            }
            _ if options.allow_scaffold => {
                Ok(scaffold_fingerprint(options, documents, &measurements))
            }
            _ => Err(
                "LLM returned output that could not be parsed as a fingerprint JSON object"
                    .to_string(),
            ),
        }
    }
}

/// Collect a fingerprint's preferred/avoided vocabulary for prompts.
pub fn lexicon_summary(fingerprint: &Value) -> Value {
    let empty = Map::new();
    let lexicon = fingerprint
        .get("lexicon")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let pick = |key: &str| -> Value { lexicon.get(key).cloned().unwrap_or_else(|| json!([])) };
    json!({
        "preferred_words": pick("preferred_words"),
        "preferred_phrases": pick("preferred_phrases"),
        "avoid_words": pick("avoid_words"),
        "avoid_words_soft": pick("avoid_words_soft")
    })
}
