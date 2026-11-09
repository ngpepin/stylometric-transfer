//! Interpretable text primitives shared by profiling and rewriting.
//!
//! These mirror `utils.py` plus the sentence/paragraph helpers so that a
//! fingerprint produced by the Rust server is measurement-compatible with one
//! produced by the Python implementation.

use once_cell::sync::Lazy;
use regex::Regex;

static WORD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?").unwrap());
static PARA_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n\s*\n+").unwrap());
static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static CONTRACTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b").unwrap());
static APPOSITIVE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r",\s+(?:a|an|the|which|who|that)\b").unwrap());

/// Tokenize text into word-like units (borrowed slices; no per-token allocation).
pub fn words(text: &str) -> Vec<&str> {
    WORD_RE.find_iter(text).map(|m| m.as_str()).collect()
}

/// Number of word-like tokens in `text`.
pub fn word_count(text: &str) -> usize {
    WORD_RE.find_iter(text).count()
}

/// Lowercased tokens.
pub fn lowered_words(text: &str) -> Vec<String> {
    WORD_RE
        .find_iter(text)
        .map(|m| m.as_str().to_ascii_lowercase())
        .collect()
}

/// Collapse every whitespace run (including newlines) to a single space.
pub fn collapse_ws(text: &str) -> String {
    WS_RE.replace_all(text, " ").trim().to_string()
}

/// Count contraction occurrences with the heuristic used by the profiler.
pub fn contraction_hits(text: &str) -> usize {
    CONTRACTION_RE.find_iter(text).count()
}

/// Count appositive-ish `, a|an|the|which|who|that` occurrences.
pub fn appositive_hits(text: &str) -> usize {
    APPOSITIVE_RE.find_iter(&text.to_lowercase()).count()
}

/// Split text into sentences using the same heuristic as the Python version:
/// break after `.`/`!`/`?` when followed by whitespace and an uppercase letter,
/// digit, quote, or opening bracket.
pub fn split_sentences(text: &str) -> Vec<String> {
    let normalized = collapse_ws(text);
    if normalized.is_empty() {
        return Vec::new();
    }
    let bytes = normalized.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'.' || c == b'!' || c == b'?' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j > i + 1 && j < bytes.len() {
                let next = bytes[j];
                let boundary = next.is_ascii_uppercase()
                    || next.is_ascii_digit()
                    || next == b'"'
                    || next == b'\''
                    || next == b'('
                    || next == b'[';
                if boundary {
                    let piece = normalized[start..j].trim();
                    if !piece.is_empty() {
                        out.push(piece.to_string());
                    }
                    start = j;
                    i = j;
                    continue;
                }
            }
        }
        i += 1;
    }
    let tail = normalized[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Split text into paragraphs separated by blank lines.
pub fn split_paragraphs(text: &str) -> Vec<String> {
    PARA_SPLIT_RE
        .split(text.trim())
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Build a normalized histogram from integer values and inclusive bins.
pub fn histogram(values: &[usize], bins: &[(usize, Option<usize>)]) -> Vec<f64> {
    if bins.is_empty() {
        return Vec::new();
    }
    let mut counts = vec![0usize; bins.len()];
    for &v in values {
        for (i, (lo, hi)) in bins.iter().enumerate() {
            match hi {
                None => {
                    if v >= *lo {
                        counts[i] += 1;
                        break;
                    }
                }
                Some(h) => {
                    if v >= *lo && v <= *h {
                        counts[i] += 1;
                        break;
                    }
                }
            }
        }
    }
    let total: usize = counts.iter().sum();
    if total == 0 {
        return vec![0.0; counts.len()];
    }
    counts.iter().map(|c| *c as f64 / total as f64).collect()
}

/// Convert a count into a per-1000-word rate.
pub fn rate_per_1000w(count: usize, total_words: usize) -> f64 {
    if total_words == 0 {
        return 0.0;
    }
    (count as f64 / total_words as f64) * 1000.0
}

/// Mean with a zero fallback.
pub fn safe_mean(xs: &[usize]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<usize>() as f64 / xs.len() as f64
}

/// Population standard deviation with a zero fallback.
pub fn safe_stdev(xs: &[usize]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mean = safe_mean(xs);
    let var = xs
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / xs.len() as f64;
    var.sqrt()
}

/// Clamp into the `[0, 1]` range.
pub fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Rough token estimate used for prompt budgeting (~4 characters per token).
pub fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, (text.chars().count() + 3) / 4)
}

/// Token estimate for a chat message list (per-message overhead included).
pub fn estimate_tokens_for_messages(messages: &[(String, String)]) -> usize {
    let mut total = 2usize;
    for (_, content) in messages {
        total += estimate_tokens(content) + 4;
    }
    total
}
