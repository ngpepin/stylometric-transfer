//! Heading sanitization and case normalization (deterministic humanizer rules).
//!
//! Ported from `apply_fingerprint.py`:
//! `strip_heading_qualifier`, `enforce_heading_qualifiers`,
//! `apply_heading_case_style`, `transfer_heading_casing`, and
//! `enforce_heading_case_normalization_from_source`.

use crate::filters::heading_at;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};

static WORD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z][A-Za-z']*").unwrap());
static ATX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(\s{0,3}#{1,8})\s+(.*)$").unwrap());
static ATX_TEXT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s{0,3}#{1,8}\s+(.+?)\s*$").unwrap());
static LINK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap());
static MARKUP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[`*_~]").unwrap());
static DIGIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d").unwrap());
static PAREN_QUAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?s)(.+?)\s*\(([^()]+)\)\s*$").unwrap());
static COMMA_QUAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?s)(.+?),\s+([^,]+)$").unwrap());
static LEADING_WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s*)").unwrap());

pub const HEADING_QUALIFIER_PREFIXES: &[&str] = &[
    "why", "how", "what", "when", "where", "defined", "definition", "quick", "brief",
    "overview", "context", "rationale", "motivation", "intuition", "notes", "note",
    "summary", "takeaway", "key", "guide", "guidance", "in practice", "in theory",
    "in brief", "in short", "quick fix", "quick fixes", "example", "examples",
    "case study", "case studies",
];
pub const HEADING_QUALIFIER_MAX_WORDS: usize = 10;
pub const HEADING_QUALIFIER_MIN_WORDS: usize = 2;

/// Count words in a heading after stripping lightweight markup.
pub fn heading_word_count(text: &str) -> usize {
    let cleaned = LINK_RE.replace_all(text, "$1").into_owned();
    let cleaned = MARKUP_RE.replace_all(&cleaned, "").into_owned();
    crate::text::word_count(&cleaned)
}

/// Detect trailing qualifier text in headings.
pub fn looks_like_heading_qualifier(text: &str) -> bool {
    let stripped = text.trim();
    if stripped.is_empty() {
        return false;
    }
    if DIGIT_RE.is_match(stripped) {
        return false;
    }
    let wc = crate::text::word_count(stripped);
    if wc == 0 || wc > HEADING_QUALIFIER_MAX_WORDS {
        return false;
    }
    let lower = stripped.to_lowercase();
    if HEADING_QUALIFIER_PREFIXES
        .iter()
        .any(|p| lower.starts_with(p))
    {
        return true;
    }
    stripped.chars().next().map(|c| c.is_lowercase()).unwrap_or(false)
}

/// Remove a trailing parenthetical/comma qualifier when it looks editorial.
pub fn strip_heading_qualifier(heading_text: &str) -> String {
    if heading_text.trim().is_empty() {
        return heading_text.to_string();
    }
    let base_text = heading_text.trim();
    if let Some(caps) = PAREN_QUAL_RE.captures(base_text) {
        let base = caps[1].trim();
        let qual = caps[2].trim();
        if looks_like_heading_qualifier(qual) && heading_word_count(base) >= HEADING_QUALIFIER_MIN_WORDS {
            return base.to_string();
        }
    }
    if let Some(caps) = COMMA_QUAL_RE.captures(base_text) {
        let base = caps[1].trim();
        let qual = caps[2].trim();
        if looks_like_heading_qualifier(qual) && heading_word_count(base) >= HEADING_QUALIFIER_MIN_WORDS {
            return base.to_string();
        }
    }
    heading_text.to_string()
}
/// Remove trailing qualifiers from Markdown headings (code fences preserved).
pub fn enforce_heading_qualifiers(text: &str, allowlist: &[Regex]) -> (String, usize) {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.is_empty() {
        return (text.to_string(), 0);
    }
    let mut in_code = false;
    let mut fence = String::new();
    let mut out: Vec<String> = Vec::new();
    let mut changes = 0usize;
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let stripped = line.trim();
        if stripped.starts_with("```") || stripped.starts_with("~~~") {
            if !in_code {
                in_code = true;
                fence = stripped.chars().take(3).collect();
            } else if stripped.starts_with(fence.as_str()) {
                in_code = false;
                fence.clear();
            }
            out.push(line.to_string());
            i += 1;
            continue;
        }
        if in_code {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        let heading = heading_at(&lines, i);
        let heading = match heading {
            None => {
                out.push(line.to_string());
                i += 1;
                continue;
            }
            Some(h) => h,
        };
        let skip = allowlist.iter().any(|r| r.is_match(&heading.text));
        let sanitized = if skip {
            heading.text.clone()
        } else {
            strip_heading_qualifier(&heading.text)
        };
        if sanitized != heading.text {
            if heading.span == 1 {
                match ATX_RE.captures(line) {
                    Some(caps) => out.push(format!("{} {}", &caps[1], sanitized)),
                    None => out.push(line.to_string()),
                }
            } else {
                out.push(sanitized);
                for j in i + 1..i + heading.span {
                    out.push(lines[j].to_string());
                }
            }
            changes += 1;
        } else {
            out.push(line.to_string());
            for j in i + 1..i + heading.span {
                out.push(lines[j].to_string());
            }
        }
        i += heading.span;
    }
    (out.join("\n"), changes)
}

fn is_alpha_upper(c: char) -> bool {
    !c.is_alphabetic() || c.is_uppercase()
}

fn is_upper_word(w: &str) -> bool {
    w.chars().all(is_alpha_upper) && w.chars().any(|c| c.is_alphabetic())
}

fn is_lower_word(w: &str) -> bool {
    w.chars().all(|c| !c.is_alphabetic() || c.is_lowercase())
        && w.chars().any(|c| c.is_alphabetic())
}

fn is_cap_word(w: &str) -> bool {
    let mut chars = w.chars();
    match chars.next() {
        Some(first) if first.is_uppercase() => chars.all(|c| !c.is_alphabetic() || c.is_lowercase()),
        _ => false,
    }
}

fn capitalize(w: &str) -> String {
    let mut chars = w.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}

fn word_case_pattern(word: &str) -> &'static str {
    if is_upper_word(word) {
        "upper"
    } else if is_lower_word(word) {
        "lower"
    } else if is_cap_word(word) {
        "capitalized"
    } else {
        "mixed"
    }
}

fn apply_word_case_pattern(word: &str, pattern: &str) -> String {
    match pattern {
        "upper" => word.to_uppercase(),
        "lower" => word.to_lowercase(),
        "capitalized" => capitalize(word),
        _ => word.to_string(),
    }
}

/// Apply an explicit case style to a heading (with optional source casing transfer).
pub fn apply_heading_case_style(
    heading_text: &str,
    mode: &str,
    source_heading: Option<&str>,
    preserve_proper_name_case: bool,
) -> String {
    let mode = mode.trim().to_lowercase();
    let mode = match mode.as_str() {
        "upper" | "uppercase" => "caps",
        "title" => "title-case",
        "sentence" => "sentence-case",
        other => other,
    };
    if mode == "automatic" {
        return heading_text.to_string();
    }
    if mode == "identical" || mode == "unchanged" {
        return match source_heading {
            Some(src) => transfer_heading_casing(src, heading_text, preserve_proper_name_case),
            None => heading_text.to_string(),
        };
    }

    let transformed = match mode {
        "caps" => heading_text.to_uppercase(),
        "lower" => heading_text.to_lowercase(),
        "sentence-case" => {
            let lowered = heading_text.to_lowercase();
            match WORD_RE.find(&lowered) {
                Some(m) => {
                    let start = m.start();
                    let end = m.end();
                    let mut out = String::with_capacity(lowered.len());
                    out.push_str(&lowered[..start]);
                    out.push_str(&lowered[start..end].to_uppercase());
                    out.push_str(&lowered[end..]);
                    out
                }
                None => lowered,
            }
        }
        "title-case" => title_case(heading_text),
        _ => heading_text.to_string(),
    };

    if preserve_proper_name_case {
        if let Some(src) = source_heading {
            return preserve_proper_name_case_fn(src, &transformed);
        }
    }
    transformed
}

const TITLE_SMALL_WORDS: &[&str] = &[
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "nor", "of", "on",
    "or", "per", "the", "to", "vs", "via", "with",
];

fn title_case(text: &str) -> String {
    let matches: Vec<&str> = WORD_RE.find_iter(text).map(|m| m.as_str()).collect();
    let total = matches.len();
    let mut idx = 0usize;
    WORD_RE
        .replace_all(text, |caps: &regex::Captures| {
            idx += 1;
            let word = &caps[0];
            let lower = word.to_lowercase();
            if is_upper_word(word) && word.chars().count() > 1 {
                return word.to_string();
            }
            if idx > 1 && idx < total && TITLE_SMALL_WORDS.contains(&lower.as_str()) {
                return lower;
            }
            capitalize(word)
        })
        .into_owned()
}

/// Transfer the source heading's per-token casing onto the rewritten heading.
pub fn transfer_heading_casing(
    source_heading: &str,
    rewritten_heading: &str,
    preserve_names: bool,
) -> String {
    let src_matches: Vec<&str> = WORD_RE.find_iter(source_heading).map(|m| m.as_str()).collect();
    let dst_matches: Vec<&str> = WORD_RE.find_iter(rewritten_heading).map(|m| m.as_str()).collect();
    if dst_matches.is_empty() {
        return rewritten_heading.to_string();
    }
    if src_matches.is_empty() {
        return rewritten_heading.to_string();
    }

    let patterns: Vec<&'static str> = src_matches.iter().map(|w| word_case_pattern(w)).collect();
    let mut idx = 0usize;
    let same_len = src_matches.len() == dst_matches.len();
    let out = WORD_RE
        .replace_all(rewritten_heading, |caps: &regex::Captures| {
            let word = &caps[0];
            let pattern = if idx < patterns.len() {
                patterns[idx]
            } else if same_len {
                "mixed"
            } else {
                "keep"
            };
            idx += 1;
            if pattern == "keep" {
                word.to_string()
            } else {
                apply_word_case_pattern(word, pattern)
            }
        })
        .into_owned();

    if preserve_names {
        preserve_proper_name_case_fn(source_heading, &out)
    } else {
        out
    }
}

const NAME_STOPWORDS: &[&str] = &[
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "near", "nor",
    "of", "on", "or", "per", "the", "to", "vs", "via", "with", "without", "over", "under",
    "has", "have", "had", "is", "are", "was", "were", "be", "being", "been",
];

const NAME_RUN_FOLLOWERS: &[&str] = &[
    "has", "have", "had", "is", "are", "was", "were", "be", "being", "been", "does", "do",
    "did", "can", "could", "will", "would", "should", "may", "might", "must",
];

fn is_title_token(word: &str) -> bool {
    is_cap_word(word)
}

/// Indices of tokens in a heading that look like proper names.
fn detect_proper_name_indices(text: &str) -> BTreeSet<usize> {
    let matches: Vec<&str> = WORD_RE.find_iter(text).map(|m| m.as_str()).collect();
    let mut indices = BTreeSet::new();
    for (idx, w) in matches.iter().enumerate() {
        let lower = w.to_lowercase();
        if NAME_STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        if is_upper_word(w) && w.chars().count() > 1 {
            indices.insert(idx);
            continue;
        }
        if word_case_pattern(w) == "mixed" {
            indices.insert(idx);
            continue;
        }
        if is_title_token(w) && idx == 0 {
            let mut run_end = idx;
            while run_end + 1 < matches.len() {
                let nxt = matches[run_end + 1];
                let nxt_lower = nxt.to_lowercase();
                if !is_title_token(nxt) || NAME_STOPWORDS.contains(&nxt_lower.as_str()) {
                    break;
                }
                run_end += 1;
            }
            if run_end > idx {
                let follower = matches
                    .get(run_end + 1)
                    .map(|w| w.to_lowercase())
                    .unwrap_or_default();
                if NAME_RUN_FOLLOWERS.contains(&follower.as_str()) {
                    for j in idx..=run_end {
                        indices.insert(j);
                    }
                }
            }
        }
    }
    indices
}

/// Restore source proper-name casing on identical lexical tokens.
fn preserve_proper_name_case_fn(source_heading: &str, transformed: &str) -> String {
    let name_indices = detect_proper_name_indices(source_heading);
    if name_indices.is_empty() {
        return transformed.to_string();
    }
    let mut out = transformed.to_string();
    for idx in name_indices {
        let src_matches: Vec<&str> = WORD_RE.find_iter(source_heading).map(|m| m.as_str()).collect();
        if idx >= src_matches.len() {
            break;
        }
        let src_word = src_matches[idx];
        let dst_matches: Vec<(usize, usize, String)> = WORD_RE
            .find_iter(&out)
            .map(|m| (m.start(), m.end(), m.as_str().to_string()))
            .collect();
        if idx >= dst_matches.len() {
            break;
        }
        let (start, end, dst_word) = &dst_matches[idx];
        if src_word.to_lowercase() != dst_word.to_lowercase() {
            continue;
        }
        out = format!("{}{}{}", &out[..*start], src_word, &out[*end..]);
    }
    out
}

/// Resolve the heading-case mode for a specific heading level.
pub fn resolve_heading_case_mode_for_level(
    global_mode: &str,
    by_level: &BTreeMap<usize, String>,
    level: usize,
) -> String {
    let mode = global_mode.trim().to_lowercase();
    if mode == "by-level" {
        return by_level
            .get(&level)
            .map(|v| v.trim().to_lowercase())
            .unwrap_or_else(|| "automatic".to_string());
    }
    mode
}

/// True when any level needs a deterministic (non-automatic) case transform.
pub fn heading_case_is_deterministic(
    global_mode: &str,
    by_level: &BTreeMap<usize, String>,
) -> bool {
    let mode = global_mode.trim().to_lowercase();
    if mode == "identical" {
        return true;
    }
    if mode == "by-level" {
        return (1..=8).any(|lvl| {
            resolve_heading_case_mode_for_level(global_mode, by_level, lvl) != "automatic"
        });
    }
    mode != "automatic"
}

/// Enforce heading case policy by aligning source and rewritten headings.
pub fn enforce_heading_case_normalization_from_source(
    source_markdown: &str,
    rewritten_markdown: &str,
    global_mode: &str,
    by_level: &BTreeMap<usize, String>,
    preserve_proper_name_case: bool,
) -> (String, usize) {
    let src_lines: Vec<&str> = source_markdown.split('\n').collect();
    let mut out_lines: Vec<String> = rewritten_markdown.split('\n').map(|s| s.to_string()).collect();

    let mut src_headings: Vec<(usize, String)> = Vec::new();
    let mut i = 0usize;
    while i < src_lines.len() {
        if let Some(h) = heading_at(&src_lines, i) {
            src_headings.push((h.level, h.text));
            i += h.span;
            continue;
        }
        i += 1;
    }

    let mut out_headings: Vec<(usize, usize, String, usize)> = Vec::new();
    {
        let refs: Vec<&str> = out_lines.iter().map(|s| s.as_str()).collect();
        let mut j = 0usize;
        while j < refs.len() {
            if let Some(h) = heading_at(&refs, j) {
                out_headings.push((j, h.level, h.text, h.span));
                j += h.span;
                continue;
            }
            j += 1;
        }
    }

    let mut edits = 0usize;
    let limit = src_headings.len().min(out_headings.len());
    for idx in 0..limit {
        let (src_level, src_text) = &src_headings[idx];
        let (out_idx, out_level, out_text, _span) = &out_headings[idx];
        if src_level != out_level {
            continue;
        }
        let mode = resolve_heading_case_mode_for_level(global_mode, by_level, *out_level);
        let recased = apply_heading_case_style(
            out_text,
            &mode,
            Some(src_text.as_str()),
            preserve_proper_name_case,
        );
        if &recased == out_text {
            continue;
        }
        let line = &out_lines[*out_idx];
        if let Some(caps) = ATX_TEXT_RE.captures(line) {
            let m = caps.get(1).unwrap();
            let mut new_line = String::new();
            new_line.push_str(&line[..m.start()]);
            new_line.push_str(&recased);
            new_line.push_str(&line[m.end()..]);
            out_lines[*out_idx] = new_line;
            edits += 1;
            continue;
        }
        let prefix = LEADING_WS_RE
            .captures(line)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        out_lines[*out_idx] = format!("{prefix}{recased}");
        edits += 1;
    }
    (out_lines.join("\n"), edits)
}



