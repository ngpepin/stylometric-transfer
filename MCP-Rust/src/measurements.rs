//! Interpretable stylometric measurements.
//!
//! Ports `compute_measurements` (input text) and the corpus variant from
//! `fingerprint_style.py`. Everything here is deterministic and explainable:
//! histogram distributions, per-1000-word rates, and token-marker counts.

use crate::text;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};

static APPOSITIVE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r",\s+(?:a|an|the|which|who|that)\b").unwrap());
static ROMAN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[ivxlcdm]+$").unwrap());

/// Sentence-length histogram bins (inclusive lower, optional upper).
pub const SENTENCE_BINS: [(usize, Option<usize>); 5] =
    [(0, Some(9)), (10, Some(17)), (18, Some(25)), (26, Some(40)), (41, None)];
/// Paragraph-length histogram bins (sentences per paragraph).
pub const PARAGRAPH_BINS: [(usize, Option<usize>); 5] =
    [(1, Some(1)), (2, Some(3)), (4, Some(5)), (6, Some(8)), (9, None)];

const SENTENCE_BIN_LABELS: [&str; 5] = ["<10", "10-17", "18-25", "26-40", ">40"];
const PARAGRAPH_BIN_LABELS: [&str; 5] = ["1", "2-3", "4-5", "6-8", ">8"];

fn stop_words() -> BTreeSet<&'static str> {
    [
        "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in", "on", "for",
        "with", "as", "is", "are", "was", "were", "be", "been", "it", "that", "this",
        "these", "those", "i", "you", "he", "she", "they", "we", "me", "him", "her",
        "them", "my", "your", "our", "their", "at", "by",
    ]
    .into_iter()
    .collect()
}

/// Heuristic US vs Canadian/British spelling detection (shared with profiling).
pub fn detect_english_spelling_variant(text: &str) -> Value {
    const PAIRS: [(&str, &str); 16] = [
        ("color", "colour"),
        ("favor", "favour"),
        ("honor", "honour"),
        ("labor", "labour"),
        ("neighbor", "neighbour"),
        ("center", "centre"),
        ("theater", "theatre"),
        ("fiber", "fibre"),
        ("liter", "litre"),
        ("meter", "metre"),
        ("check", "cheque"),
        ("defense", "defence"),
        ("offense", "offence"),
        ("license", "licence"),
        ("practice", "practise"),
        ("program", "programme"),
    ];
    let lowered = text.to_lowercase();
    let tokens: Vec<&str> = text::words(&lowered);
    let mut us_hits = 0usize;
    let mut ca_hits = 0usize;
    let mut us_examples: Vec<String> = Vec::new();
    let mut ca_examples: Vec<String> = Vec::new();
    for (us, ca) in PAIRS {
        let u = tokens.iter().filter(|&t| *t == us).count();
        let c = tokens.iter().filter(|&t| *t == ca).count();
        if u > 0 {
            us_hits += u;
            if !us_examples.iter().any(|e| e == us) {
                us_examples.push(us.to_string());
            }
        }
        if c > 0 {
            ca_hits += c;
            if !ca_examples.iter().any(|e| e == ca) {
                ca_examples.push(ca.to_string());
            }
        }
    }
    let total = us_hits + ca_hits;
    if total < 3 {
        return json!({
            "language": "en",
            "variant": "unknown",
            "confidence": "low",
            "us_hits": us_hits,
            "canadian_hits": ca_hits,
            "examples": {"us": us_examples, "canadian": ca_examples},
            "note": "Insufficient evidence to determine spelling variant."
        });
    }
    let variant = if us_hits > ca_hits {
        "us"
    } else if ca_hits > us_hits {
        "canadian"
    } else {
        "unknown"
    };
    let confidence = if total >= 8 && (us_hits as i64 - ca_hits as i64).abs() >= 4 {
        "high"
    } else {
        "medium"
    };
    us_examples.truncate(5);
    ca_examples.truncate(5);
    json!({
        "language": "en",
        "variant": variant,
        "confidence": confidence,
        "us_hits": us_hits,
        "canadian_hits": ca_hits,
        "examples": {"us": us_examples, "canadian": ca_examples},
        "note": "Heuristic based on common US vs Canadian spellings."
    })
}
/// Classic stylometry function-word list (identical to the Python source).
pub const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "if", "then", "because", "so", "while", "although",
    "though", "of", "in", "on", "for", "with", "as", "by", "from", "to", "into", "over", "under",
    "between", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "i", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them", "my", "our",
    "your", "their", "not", "no", "nor", "very", "also", "even", "only", "just", "rather",
    "however", "therefore",
];

pub const HEDGE_TERMS: &[&str] = &[
    "may", "might", "perhaps", "likely", "possibly", "seems", "appears", "suggests", "tends",
];
pub const BOOSTER_TERMS: &[&str] =
    &["clearly", "obviously", "certainly", "undoubtedly", "indeed", "surely"];
pub const DIRECTIVE_TERMS: &[&str] = &["must", "should", "need", "needs", "ought", "required"];
pub const FIRST_PERSON: &[&str] = &["i", "me", "my", "mine", "we", "us", "our", "ours"];
pub const SECOND_PERSON: &[&str] = &["you", "your", "yours"];
pub const THIRD_PERSON: &[&str] = &[
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
];

pub const TRANSITION_TERMS: &[&str] = &[
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "nevertheless",
    "nonetheless",
    "for example",
    "for instance",
    "in short",
    "in sum",
    "in practice",
    "in effect",
    "first",
    "second",
    "third",
    "finally",
    "overall",
];

pub const SPECULATIVE_TERMS: &[&str] = &[
    "may", "might", "perhaps", "possibly", "could", "seems", "appears", "suggests", "tends",
];
pub const PROBABILISTIC_TERMS: &[&str] =
    &["likely", "unlikely", "probable", "probably", "odds", "chance"];
pub const ASSERTIVE_TERMS: &[&str] =
    &["clearly", "certainly", "undoubtedly", "indeed", "surely"];

pub const SUBORDINATOR_TERMS: &[&str] = &[
    "because", "although", "though", "while", "if", "when", "since", "unless", "whereas",
    "after", "before", "once", "until",
];

pub const CLAIM_MARKERS: &[&str] = &[
    "we argue",
    "we contend",
    "we propose",
    "this suggests",
    "this shows",
    "this indicates",
    "therefore",
    "thus",
    "hence",
    "overall",
    "in sum",
];
pub const EVIDENCE_MARKERS: &[&str] = &[
    "for example",
    "for instance",
    "according to",
    "data show",
    "evidence",
    "study",
    "report",
    "survey",
    "as shown",
];
pub const COUNTERPOINT_MARKERS: &[&str] = &[
    "however",
    "yet",
    "but",
    "on the other hand",
    "nevertheless",
    "nonetheless",
];
pub const CONCESSION_MARKERS: &[&str] =
    &["although", "though", "even though", "while", "granted", "admittedly"];
pub const SYNTHESIS_MARKERS: &[&str] = &[
    "overall",
    "in sum",
    "in short",
    "on balance",
    "taken together",
    "in conclusion",
];

pub const INTENSIFIER_TERMS: &[&str] =
    &["very", "really", "extremely", "highly", "incredibly", "quite", "so"];
pub const EMOTIONAL_ADJECTIVE_TERMS: &[&str] = &[
    "happy", "sad", "angry", "afraid", "anxious", "excited", "terrible", "wonderful", "awful",
    "lovely",
];
pub const INFORMAL_SLANG_TERMS: &[&str] = &[
    "cool", "awesome", "yeah", "ok", "okay", "stuff", "gonna", "wanna", "kinda", "sorta",
];

/// Count tokens belonging to `terms` (both sides already lowercased).
pub fn count_terms(tokens: &[String], terms: &[&str]) -> usize {
    if terms.is_empty() {
        return 0;
    }
    let set: BTreeSet<&str> = terms.iter().copied().collect();
    tokens.iter().filter(|t| set.contains(t.as_str())).count()
}

/// True when a lowercased sentence contains any marker substring.
fn sentence_has_marker(sentence: &str, markers: &[&str]) -> bool {
    markers.iter().any(|m| sentence.contains(m))
}

/// Rate helper matching Python's `approx_rate_per_1000_words`.
fn rate(count: usize, total_words: usize) -> f64 {
    text::rate_per_1000w(count, total_words)
}

/// Truncate a distribution of (phrase, count) pairs to a maximum length.
fn top_pairs(counts: &HashMap<String, usize>, limit: usize) -> Vec<(String, usize)> {
    let mut rows: Vec<(String, usize)> = counts
        .iter()
        .filter(|(_, c)| **c > 0)
        .map(|(k, c)| (k.clone(), *c))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.truncate(limit);
    rows
}

/// Emit `[{phrase, count}, ...]` for the top rows.
fn pairs_json(rows: &[(String, usize)]) -> Value {
    Value::Array(
        rows.iter()
            .map(|(p, c)| json!({"phrase": p, "count": c}))
            .collect(),
    )
}
/// Build `{key: rate_per_1000w}` for a term set.
fn rates_for(tokens: &[String], total_words: usize, terms: &[&str]) -> Value {
    let mut map = Map::new();
    for term in terms {
        let count = tokens.iter().filter(|t| t.as_str() == *term).count();
        map.insert((*term).to_string(), json!(rate(count, total_words)));
    }
    Value::Object(map)
}

/// Count occurrences of each key inside `text` (used for punctuation counts).
fn punctuation_counts(text: &str) -> Vec<(&'static str, usize)> {
    vec![
        ("commas", text.matches(',').count()),
        ("semicolons", text.matches(';').count()),
        ("colons", text.matches(':').count()),
        ("exclamations", text.matches('!').count()),
        ("questions", text.matches('?').count()),
        ("parentheses_open", text.matches('(').count()),
        ("parentheses_close", text.matches(')').count()),
        ("em_dashes", text.matches('\u{2014}').count()),
        ("en_dashes", text.matches('\u{2013}').count()),
        ("hyphen_dashes", text.matches(" - ").count()),
        ("ellipses_unicode", text.matches('\u{2026}').count()),
        ("ellipses_three_dots", text.matches("...").count()),
        ("quotes_double", text.matches('"').count()),
        ("quotes_single", text.matches('\'').count()),
    ]
}

/// Build per-1000-word rates from a `(name, count)` list.
fn punctuation_rates(counts: &[(&'static str, usize)], total_words: usize) -> Value {
    let mut map = Map::new();
    for (name, count) in counts {
        map.insert((*name).to_string(), json!(rate(*count, total_words)));
    }
    Value::Object(map)
}

/// Compute the interpretable measurement bundle for a single text.
///
/// This is the apply-side profile (`compute_measurements` in
/// `apply_fingerprint.py`): the shape is stable because the LLM prompt and the
/// style-compliance scorer both consume it.
pub fn compute_measurements(text: &str) -> Value {
    let tokens_owned = text::lowered_words(text);
    let total_words = tokens_owned.len();
    let sentences = text::split_sentences(text);
    let sent_lens: Vec<usize> = sentences
        .iter()
        .map(|s| text::word_count(s))
        .collect();
    let sent_hist = text::histogram(&sent_lens, &SENTENCE_BINS);

    let paragraphs = text::split_paragraphs(text);
    let para_lens: Vec<usize> = paragraphs
        .iter()
        .map(|p| text::split_sentences(p).len())
        .collect();
    let para_hist = text::histogram(&para_lens, &PARAGRAPH_BINS);
    let one_sentence = para_lens.iter().filter(|n| **n == 1).count();
    let one_sentence_rate = if para_lens.is_empty() {
        0.0
    } else {
        one_sentence as f64 / para_lens.len() as f64
    };

    let punct_counts = punctuation_counts(text);
    let punct_total: std::collections::HashMap<&str, usize> =
        punct_counts.iter().copied().collect();
    let commas = *punct_total.get("commas").unwrap_or(&0);

    let contraction_count = text::contraction_hits(text);
    let contraction_rate = if total_words == 0 {
        0.0
    } else {
        contraction_count as f64 / total_words as f64
    };
    let lowered = text.to_lowercase();
    let and_total = lowered.matches(" and ").count();
    let comma_and = lowered.matches(", and ").count();
    let oxford_signal = if and_total == 0 {
        0.0
    } else {
        comma_and as f64 / and_total as f64
    };

// Function-word profile (classic stylometry).
    let mut fw_counts: HashMap<&str, usize> = HashMap::new();
    for fw in FUNCTION_WORDS {
        fw_counts.insert(*fw, 0);
    }
    for token in &tokens_owned {
        if let Some(slot) = fw_counts.get_mut(token.as_str()) {
            *slot += 1;
        }
    }
    let mut fw_rates = Map::new();
    for fw in FUNCTION_WORDS {
        let count = *fw_counts.get(*fw).unwrap_or(&0);
        fw_rates.insert((*fw).to_string(), json!(rate(count, total_words)));
    }
    let mut fw_rows: Vec<(String, usize)> = FUNCTION_WORDS
        .iter()
        .map(|fw| ((*fw).to_string(), *fw_counts.get(*fw).unwrap_or(&0)))
        .collect();
    fw_rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    fw_rows.truncate(20);
    let fw_top: Vec<Value> = fw_rows
        .iter()
        .map(|(w, c)| json!({"word": w, "count": c}))
        .collect();

    // Stance signals (hedging / boosting / person).
    let hedge_hits = count_terms(&tokens_owned, HEDGE_TERMS);
    let booster_hits = count_terms(&tokens_owned, BOOSTER_TERMS);
    let directive_hits = count_terms(&tokens_owned, DIRECTIVE_TERMS);
    let first_hits = count_terms(&tokens_owned, FIRST_PERSON);
    let second_hits = count_terms(&tokens_owned, SECOND_PERSON);
    let third_hits = count_terms(&tokens_owned, THIRD_PERSON);
    // Sentence openers and discourse-marker positions.
    let stop = stop_words();
    let mut sent_openers: HashMap<String, usize> = HashMap::new();
    let mut transition_hits: HashMap<String, usize> = HashMap::new();
    let mut transition_start_hits = 0usize;
    let mut transition_mid_hits = 0usize;
    let transition_set: BTreeSet<&str> = TRANSITION_TERMS.iter().copied().collect();
    for sentence in &sentences {
        let ws = text::lowered_words(sentence);
        if ws.len() >= 2 {
            let opener = if ws.len() >= 3 {
                ws[..3].join(" ")
            } else {
                ws[..2].join(" ")
            };
            let stop_hits = opener
                .split_whitespace()
                .filter(|x| stop.contains(*x))
                .count();
            if stop_hits <= 1 {
                *sent_openers.entry(opener).or_insert(0) += 1;
            }
        }
        if ws.is_empty() {
            continue;
        }
        let start_two = ws[..ws.len().min(2)].join(" ");
        let start_three = if ws.len() >= 3 {
            ws[..3].join(" ")
        } else {
            String::new()
        };
        for candidate in [start_three.as_str(), start_two.as_str(), ws[0].as_str()] {
            if !candidate.is_empty() && transition_set.contains(candidate) {
                *transition_hits.entry(candidate.to_string()).or_insert(0) += 1;
                break;
            }
        }
        let start_candidate = if ws.len() >= 3 {
            ws[..3].join(" ")
        } else if ws.len() >= 2 {
            ws[..2].join(" ")
        } else {
            ws[0].clone()
        };
        if transition_set.contains(start_candidate.as_str()) {
            transition_start_hits += 1;
        } else {
            let rest = ws[1..].join(" ");
            if TRANSITION_TERMS.iter().any(|t| rest.contains(t)) {
                transition_mid_hits += 1;
            }
        }
    }

    // Rhetorical moves.
    let lowered_sentences: Vec<String> = sentences.iter().map(|s| s.to_lowercase()).collect();
    let claim_hits = lowered_sentences
        .iter()
        .filter(|s| sentence_has_marker(s, CLAIM_MARKERS))
        .count();
    let evidence_hits = lowered_sentences
        .iter()
        .filter(|s| sentence_has_marker(s, EVIDENCE_MARKERS))
        .count();
    let counter_hits = lowered_sentences
        .iter()
        .filter(|s| sentence_has_marker(s, COUNTERPOINT_MARKERS))
        .count();
    let concession_hits = lowered_sentences
        .iter()
        .filter(|s| sentence_has_marker(s, CONCESSION_MARKERS))
        .count();
    let synthesis_hits = lowered_sentences
        .iter()
        .filter(|s| sentence_has_marker(s, SYNTHESIS_MARKERS))
        .count();

    // Paragraph cadence.
    let mut opening_lens: Vec<usize> = Vec::new();
    let mut closing_lens: Vec<usize> = Vec::new();
    for paragraph in &paragraphs {
        let sents = text::split_sentences(paragraph);
        if sents.is_empty() {
            continue;
        }
        opening_lens.push(text::word_count(&sents[0]));
        closing_lens.push(text::word_count(sents.last().unwrap()));
    }

    // Epistemic and syntax texture.
    let speculative_hits = count_terms(&tokens_owned, SPECULATIVE_TERMS);
    let probabilistic_hits = count_terms(&tokens_owned, PROBABILISTIC_TERMS);
    let assertive_hits = count_terms(&tokens_owned, ASSERTIVE_TERMS);
    let subordinator_hits = count_terms(&tokens_owned, SUBORDINATOR_TERMS);
    let parenthetical_hits = text.matches('(').count() + text.matches(')').count();
    let appositive_hits = APPOSITIVE_RE.find_iter(&lowered).count();

    // Lexical avoidance categories.
    let mut avoidance_rates = Map::new();
    for (name, terms) in [
        ("intensifiers", INTENSIFIER_TERMS),
        ("emotional_adjectives", EMOTIONAL_ADJECTIVE_TERMS),
        ("informal_slang", INFORMAL_SLANG_TERMS),
    ] {
        let hits = count_terms(&tokens_owned, terms);
        avoidance_rates.insert(name.to_string(), json!(rate(hits, total_words)));
    }

    // Self-echo repetition (bigram/trigram recurrence).
    let ngram_stop: BTreeSet<&str> = [
        "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in", "on", "for",
        "with", "as",
    ]
    .into_iter()
    .collect();
    let bigrams = ngrams(&tokens_owned, 2, &ngram_stop);
    let trigrams = ngrams(&tokens_owned, 3, &ngram_stop);
    let bigram_repeat = repeat_rate(&bigrams, 3);
    let trigram_repeat = repeat_rate(&trigrams, 3);


    json!({
        "totals": {
            "total_words_est": total_words,
            "total_sentences_est": sentences.len(),
            "total_paragraphs_est": paragraphs.len()
        },
        "sentence": {
            "length_words": {
                "histogram_bins": SENTENCE_BIN_LABELS,
                "histogram_p": sent_hist
            }
        },
        "paragraph": {
            "length_sentences_histogram_bins": PARAGRAPH_BIN_LABELS,
            "length_sentences_histogram_p": para_hist,
            "one_sentence_paragraph_rate": one_sentence_rate
        },
        "punctuation": {
            "counts": Value::Object(
                punct_counts
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), json!(v)))
                    .collect()
            ),
            "rates_per_1000w": punctuation_rates(&punct_counts, total_words)
        },
        "orthography_signals": {
            "contractions_rate": contraction_rate,
            "oxford_comma_signal": oxford_signal,
            "spelling_variant": detect_english_spelling_variant(text)
        },
        "function_words": {
            "rates_per_1000w": fw_rates,
            "top": fw_top
        },
        "stance_signals": {
            "hedge_rate": rate(hedge_hits, total_words),
            "booster_rate": rate(booster_hits, total_words),
            "directive_rate": rate(directive_hits, total_words),
            "first_person_rate": rate(first_hits, total_words),
            "second_person_rate": rate(second_hits, total_words),
            "third_person_rate": rate(third_hits, total_words)
        },
        "templates_signals": {
            "sentence_openers_top": pairs_json(&top_pairs(&sent_openers, 20)),
            "transition_openers_top": pairs_json(&top_pairs(&transition_hits, 15)),
            "transition_marker_positions": {
                "start_rate_per_1000w": rate(transition_start_hits, total_words),
                "mid_rate_per_1000w": rate(transition_mid_hits, total_words)
            }
        },
        "rhetoric_moves": {
            "claim_rate": rate(claim_hits, total_words),
            "evidence_rate": rate(evidence_hits, total_words),
            "counterpoint_rate": rate(counter_hits, total_words),
            "concession_rate": rate(concession_hits, total_words),
            "synthesis_rate": rate(synthesis_hits, total_words),
            "claim_evidence_ratio": claim_hits as f64 / (evidence_hits.max(1)) as f64
        },
        "paragraph_cadence": {
            "opening_sentence_length_mean": text::safe_mean(&opening_lens),
            "opening_sentence_length_stdev": text::safe_stdev(&opening_lens),
            "closing_sentence_length_mean": text::safe_mean(&closing_lens),
            "closing_sentence_length_stdev": text::safe_stdev(&closing_lens)
        },
        "epistemic_profile": {
            "speculative_rate": rate(speculative_hits, total_words),
            "probabilistic_rate": rate(probabilistic_hits, total_words),
            "assertive_rate": rate(assertive_hits, total_words),
            "directive_rate": rate(directive_hits, total_words)
        },
        "syntax_texture": {
            "subordinate_clause_rate": rate(subordinator_hits, total_words),
            "parenthetical_rate": rate(parenthetical_hits, total_words),
            "appositive_rate": rate(appositive_hits, total_words)
        },
        "lexical_avoidance": {
            "category_rates_per_1000w": avoidance_rates
        },
        "repetition": {
            "bigram_repeat_rate": bigram_repeat,
            "trigram_repeat_rate": trigram_repeat,
            "min_repeat_count": 3
        },
        "comma_density_per_100w": if total_words == 0 {
            0.0
        } else {
            (commas as f64 / total_words as f64) * 100.0
        }
    })
}
/// Generate n-grams, skipping windows that are mostly stop words.
fn ngrams(tokens: &[String], n: usize, stop: &BTreeSet<&str>) -> Vec<String> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..=(tokens.len() - n) {
        let chunk = &tokens[i..i + n];
        let stop_hits = chunk.iter().filter(|t| stop.contains(t.as_str())).count();
        if stop_hits >= n.saturating_sub(1) {
            continue;
        }
        out.push(chunk.join(" "));
    }
    out
}

/// Fraction of n-grams that recur at least `min_count` times.
fn repeat_rate(ngrams: &[String], min_count: usize) -> f64 {
    if ngrams.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for gram in ngrams {
        *counts.entry(gram.as_str()).or_insert(0) += 1;
    }
    let repeated: usize = counts.values().filter(|c| **c >= min_count).sum();
    repeated as f64 / ngrams.len() as f64
}
