//! Non-author-voice filtering, masking, and Markdown structure helpers.
//!
//! Ported from the deterministic parts of `fingerprint_style.py` and
//! `apply_fingerprint.py` so measurements are computed on author voice only and
//! frozen regions survive rewriting verbatim.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::BTreeMap;

static BASE64_IMAGE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+").unwrap()
});
static BASE64_PLACEHOLDER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[\[BASE64_IMAGE(?:_\d+)?\]\]").unwrap());
static BLOCKQUOTE_HTML_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<blockquote[^>]*>.*?</blockquote>").unwrap());
static BLOCKQUOTE_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*>").unwrap());
static FOOTNOTE_DEF_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*\[\^[^\]]+\]:").unwrap());
static INLINE_FOOTNOTE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[\^[^\]]+\]").unwrap());
static INLINE_NUMERIC_CITE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\[(?:\d+|[IVX]+)(?:\s*[-\u{2013},;]\s*(?:\d+|[IVX]+))*\]").unwrap()
});
static PAREN_GROUP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\([^()]{1,80}\)").unwrap());
static HTML_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<[A-Za-z][^>]*>.*?</[A-Za-z][^>]*>").unwrap());
static HTML_COMMENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<!--.*?-->").unwrap());
static HTML_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[A-Za-z/][^>]*>").unwrap());
static HTML_ENTITY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"&[A-Za-z0-9#]+;").unwrap());
static INLINE_CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(``[^`\n]+``|`[^`\n]+`)").unwrap());
static DISPLAY_MATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)\\\[.*?\\\]").unwrap());
static INLINE_MATH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\\\(.*?\\\)").unwrap());
static DOLLAR_DISPLAY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\$\$.*?\$\$").unwrap());
static DOLLAR_INLINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\$[^$]*\$").unwrap());
static LATEX_ENV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)\\begin\{[^}]+\}.*?\\end\{[^}]+\}").unwrap()
});
static LATEX_CMD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\\[A-Za-z]+(?:\{[^}]*\})?").unwrap());
static MULTI_NEWLINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static DOUBLE_SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").unwrap());
static LINE_PAD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r" *\n *").unwrap());
static ATX_HEADING_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s{0,3}(#{1,8})\s+(.+?)\s*$").unwrap());
static SETEXT_H1_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*=+\s*$").unwrap());
static SETEXT_H2_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*-+\s*$").unwrap());
static QUOTE_SPAN_RE: Lazy<Regex> = Lazy::new(|| Regex::new("(?s)\"([^\"]+)\"").unwrap());
static QUOTE_SPAN_CURLY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)\u{201c}([^\u{201d}]+)\u{201d}").unwrap());
static CITATION_YEAR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(19|20)\d{2}[a-z]?\b").unwrap());
static ET_AL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bet\s+al\.?\b").unwrap());
static SEE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(cf\.|see|see also)\b").unwrap());
static AUTHOR_YEAR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([A-Z][A-Za-z'\u{2019}.-]+)\s+(19|20)\d{2}[a-z]?\b").unwrap());

pub const REFERENCE_HEADINGS: &[&str] = &[
    "references",
    "bibliography",
    "works cited",
    "citations",
    "sources",
    "endnotes",
    "footnotes",
    "notes",
];

/// Quote-handling mode for the current run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteMode {
    Fiction,
    NonFiction,
}

/// Detected heading: level, text, and number of consumed lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heading {
    pub level: usize,
    pub text: String,
    pub span: usize,
}

/// Allocate the next placeholder for a frozen block.
fn take_placeholder(
    counter: &mut usize,
    mapping: &mut BTreeMap<String, String>,
    block: &str,
) -> String {
    let placeholder = format!("[[FROZEN_BLOCK_{}]]", *counter);
    *counter += 1;
    mapping.insert(placeholder.clone(), block.to_string());
    placeholder
}

/// Replace base64 image payloads with stable placeholders.
pub fn strip_base64_images(text: &str) -> (String, BTreeMap<String, String>) {
    let mut map = BTreeMap::new();
    let mut counter = 0usize;
    let replaced = BASE64_IMAGE_RE
        .replace_all(text, |caps: &regex::Captures| {
            let placeholder = format!("[[BASE64_IMAGE_{}]]", counter);
            counter += 1;
            map.insert(placeholder.clone(), caps[0].to_string());
            placeholder
        })
        .into_owned();
    (replaced, map)
}

/// Normalize line endings and common typographic artifacts.
pub fn normalize_text(s: &str) -> String {
    let mut out = s.replace("\r\n", "\n").replace('\r', "\n");
    out = out.replace('\u{fb01}', "fi").replace('\u{fb02}', "fl");
    out = out.replace('\u{2019}', "'").replace('\u{2018}', "'");
    out = out.replace('\u{201c}', "\"").replace('\u{201d}', "\"");
    out
}

/// Normalize a heading for loose matching (case/punctuation insensitive).
pub fn normalize_heading_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let stripped: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Return the heading starting at `idx`, if any.
pub fn heading_at(lines: &[&str], idx: usize) -> Option<Heading> {
    let line = *lines.get(idx)?;
    if let Some(caps) = ATX_HEADING_RE.captures(line) {
        return Some(Heading {
            level: caps[1].len(),
            text: caps[2].trim().to_string(),
            span: 1,
        });
    }
    if let Some(underline) = lines.get(idx + 1) {
        if SETEXT_H1_RE.is_match(underline) {
            return Some(Heading {
                level: 1,
                text: line.trim().to_string(),
                span: 2,
            });
        }
        if SETEXT_H2_RE.is_match(underline) {
            return Some(Heading {
                level: 2,
                text: line.trim().to_string(),
                span: 2,
            });
        }
    }
    None
}

/// True when the heading names a reference-like section.
pub fn is_reference_heading(text: &str) -> bool {
    let normalized = normalize_heading_text(text);
    REFERENCE_HEADINGS.contains(&normalized.as_str())
}
/// Line ranges `[start, end)` for reference-like sections.
pub fn find_reference_sections(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut sections = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let heading = match heading_at(lines, i) {
            Some(h) => h,
            None => {
                i += 1;
                continue;
            }
        };
        if !is_reference_heading(&heading.text) {
            i += heading.span;
            continue;
        }
        let start = i;
        let mut j = i + heading.span;
        let mut end = lines.len();
        while j < lines.len() {
            if let Some(next) = heading_at(lines, j) {
                if next.level <= heading.level {
                    end = j;
                    break;
                }
            }
            j += 1;
        }
        sections.push((start, end));
        i = end;
    }
    sections
}

/// Remove blockquotes, reference sections, and footnote definitions.
pub fn strip_non_voice_sections(text: &str) -> String {
    let stripped = BLOCKQUOTE_HTML_RE.replace_all(text, "\n").into_owned();
    let lines: Vec<&str> = stripped.lines().collect();
    let refs = find_reference_sections(&lines);
    let mut ref_iter = refs.into_iter().peekable();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if let Some(&(start, end)) = ref_iter.peek() {
            if i == start {
                i = end;
                ref_iter.next();
                continue;
            }
        }
        let line = lines[i];
        if BLOCKQUOTE_LINE_RE.is_match(line) {
            i += 1;
            continue;
        }
        if FOOTNOTE_DEF_RE.is_match(line) {
            i += 1;
            while i < lines.len()
                && (lines[i].starts_with("    ") || lines[i].starts_with('\t'))
            {
                i += 1;
            }
            continue;
        }
        out.push(line);
        i += 1;
    }
    MULTI_NEWLINE_RE
        .replace_all(&out.join("\n"), "\n\n")
        .trim()
        .to_string()
}

/// Remove fenced code blocks entirely.
pub fn strip_fenced_code_blocks(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut in_code = false;
    let mut fence = String::new();
    for line in text.lines() {
        let stripped = line.trim();
        let is_fence = stripped.starts_with("```") || stripped.starts_with("~~~");
        if is_fence {
            if !in_code {
                in_code = true;
                fence = stripped.chars().take(3).collect();
            } else if stripped.starts_with(fence.as_str()) {
                in_code = false;
                fence.clear();
            }
            continue;
        }
        if !in_code {
            out.push(line);
        }
    }
    out.join("\n")
}

/// Remove inline code spans.
pub fn strip_inline_code(text: &str) -> String {
    INLINE_CODE_RE.replace_all(text, "").into_owned()
}

/// Remove HTML blocks, comments, tags, and entities.
pub fn strip_html(text: &str) -> String {
    let mut out = HTML_BLOCK_RE.replace_all(text, "\n").into_owned();
    out = HTML_COMMENT_RE.replace_all(&out, "").into_owned();
    out = HTML_TAG_RE.replace_all(&out, "").into_owned();
    HTML_ENTITY_RE.replace_all(&out, "").into_owned()
}

/// Remove LaTeX inline/display math and bare commands.
pub fn strip_latex_math(text: &str) -> String {
    let mut out = DISPLAY_MATH_RE.replace_all(text, "\n").into_owned();
    out = INLINE_MATH_RE.replace_all(&out, "").into_owned();
    out = DOLLAR_DISPLAY_RE.replace_all(&out, "\n").into_owned();
    out = DOLLAR_INLINE_RE.replace_all(&out, "").into_owned();
    out = LATEX_ENV_RE.replace_all(&out, "\n").into_owned();
    LATEX_CMD_RE.replace_all(&out, "").into_owned()
}
/// Character spans of quoted passages with their inner text.
pub fn quote_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut spans = Vec::new();
    for m in QUOTE_SPAN_RE.find_iter(text) {
        let s = m.as_str();
        spans.push((m.start(), m.end(), s[1..s.len() - 1].to_string()));
    }
    for m in QUOTE_SPAN_CURLY_RE.find_iter(text) {
        spans.push((m.start(), m.end(), m.as_str().to_string()));
    }
    spans.sort_by_key(|s| s.0);
    spans
}

/// A quote counts as multi-word when it contains two or more tokens.
pub fn is_multiword_quote(inner: &str) -> bool {
    crate::text::word_count(inner) > 1
}

/// True when a parenthetical group looks like a citation.
pub fn is_parenthetical_citation(inner: &str) -> bool {
    if !CITATION_YEAR_RE.is_match(inner) {
        return false;
    }
    if ET_AL_RE.is_match(inner) || SEE_RE.is_match(inner) {
        return true;
    }
    if inner.contains(',') || inner.contains(';') {
        return true;
    }
    if let Some(caps) = AUTHOR_YEAR_RE.captures(inner) {
        let word = caps[1].to_lowercase();
        let months = [
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
        ];
        if !months.contains(&word.as_str()) {
            return true;
        }
    }
    false
}
/// Remove inline citation markers while keeping surrounding prose.
pub fn strip_inline_citations(text: &str) -> String {
    let mut out = INLINE_FOOTNOTE_RE.replace_all(text, "").into_owned();
    out = INLINE_NUMERIC_CITE_RE.replace_all(&out, "").into_owned();
    out = PAREN_GROUP_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let group = caps[0].to_string();
            let inner = &group[1..group.len() - 1];
            if is_parenthetical_citation(inner) {
                String::new()
            } else {
                group
            }
        })
        .into_owned();
    out = DOUBLE_SPACE_RE.replace_all(&out, " ").into_owned();
    out = LINE_PAD_RE.replace_all(&out, "\n").into_owned();
    out.trim().to_string()
}

/// Remove multi-word quoted passages (non-fiction author-voice filtering).
pub fn strip_quoted_passages(text: &str) -> String {
    let spans = quote_spans(text);
    if spans.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for (start, end, inner) in spans {
        if start < last {
            continue;
        }
        out.push_str(&text[last..start]);
        if !is_multiword_quote(&inner) {
            out.push_str(&text[start..end]);
        }
        last = end;
    }
    out.push_str(&text[last..]);
    out
}

/// Detect fiction from quote density heuristics.
pub fn detect_fiction(
    text: &str,
    quote_span_min: usize,
    quoted_ratio_min: f64,
    quote_para_ratio_min: f64,
    quoted_ratio_force: f64,
) -> bool {
    let total_words = crate::text::word_count(text).max(1);
    let mut span_count = 0usize;
    let mut quoted_words = 0usize;
    for (_, _, inner) in quote_spans(text) {
        if is_multiword_quote(&inner) {
            span_count += 1;
            quoted_words += crate::text::word_count(&inner);
        }
    }
    let mut quote_para = 0usize;
    let mut total_para = 0usize;
    for para in crate::text::split_paragraphs(text) {
        total_para += 1;
        let trimmed = para.trim_start();
        if trimmed.starts_with('"') || trimmed.starts_with('\u{201c}') {
            quote_para += 1;
        }
    }
    let quoted_ratio = quoted_words as f64 / total_words as f64;
    let quote_para_ratio = quote_para as f64 / total_para.max(1) as f64;
    (span_count >= quote_span_min && quoted_ratio >= quoted_ratio_min)
        || (quote_para_ratio >= quote_para_ratio_min && quoted_ratio >= quoted_ratio_min)
        || quoted_ratio >= quoted_ratio_force
}
/// Full author-voice filtering pipeline used before measuring text.
pub fn filter_author_voice(text: &str, mode: QuoteMode) -> String {
    let mut out = strip_fenced_code_blocks(text);
    out = strip_inline_code(&out);
    out = strip_non_voice_sections(&out);
    if mode == QuoteMode::NonFiction {
        out = strip_quoted_passages(&out);
    }
    out = strip_latex_math(&out);
    out = strip_html(&out);
    out = BASE64_PLACEHOLDER_RE.replace_all(&out, "").into_owned();
    out = strip_inline_citations(&out);
    MULTI_NEWLINE_RE
        .replace_all(&out, "\n\n")
        .trim()
        .to_string()
}

/// Replace non-voice blocks with placeholders so they survive verbatim.
pub fn mask_non_voice_blocks(markdown: &str) -> (String, BTreeMap<String, String>) {
    let mut mapping: BTreeMap<String, String> = BTreeMap::new();
    let mut counter = 0usize;

    let mut text = BLOCKQUOTE_HTML_RE
        .replace_all(markdown, |caps: &regex::Captures| {
            take_placeholder(&mut counter, &mut mapping, &caps[0])
        })
        .into_owned();

    let lines: Vec<&str> = text.lines().collect();
    let refs = find_reference_sections(&lines);
    let mut ref_iter = refs.into_iter().peekable();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0usize;
    while i < lines.len() {
        if let Some(&(start, end)) = ref_iter.peek() {
            if i == start {
                let block = lines[start..end].join("\n");
                out.push(take_placeholder(&mut counter, &mut mapping, &block));
                i = end;
                ref_iter.next();
                continue;
            }
        }
        let line = lines[i];
        if BLOCKQUOTE_LINE_RE.is_match(line) {
            let start = i;
            i += 1;
            while i < lines.len() && BLOCKQUOTE_LINE_RE.is_match(lines[i]) {
                i += 1;
            }
            let block = lines[start..i].join("\n");
            out.push(take_placeholder(&mut counter, &mut mapping, &block));
            continue;
        }
        if FOOTNOTE_DEF_RE.is_match(line) {
            let start = i;
            i += 1;
            while i < lines.len()
                && (lines[i].starts_with("    ") || lines[i].starts_with('\t'))
            {
                i += 1;
            }
            let block = lines[start..i].join("\n");
            out.push(take_placeholder(&mut counter, &mut mapping, &block));
            continue;
        }
        out.push(line.to_string());
        i += 1;
    }
    text = out.join("\n");
    (text, mapping)
}
/// Replace inline citations with placeholders.
pub fn mask_inline_citations(text: &str) -> (String, BTreeMap<String, String>) {
    let mut mapping: BTreeMap<String, String> = BTreeMap::new();
    let mut counter = 0usize;
    let mut make = |matched: &str,
                    counter: &mut usize,
                    mapping: &mut BTreeMap<String, String>| {
        let placeholder = format!("[[CITATION_{}]]", *counter);
        *counter += 1;
        mapping.insert(placeholder.clone(), matched.to_string());
        placeholder
    };
    let mut out = INLINE_FOOTNOTE_RE
        .replace_all(text, |caps: &regex::Captures| {
            make(&caps[0], &mut counter, &mut mapping)
        })
        .into_owned();
    out = INLINE_NUMERIC_CITE_RE
        .replace_all(&out, |caps: &regex::Captures| {
            make(&caps[0], &mut counter, &mut mapping)
        })
        .into_owned();
    out = PAREN_GROUP_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let group = caps[0].to_string();
            let inner = &group[1..group.len() - 1];
            if is_parenthetical_citation(inner) {
                make(&group, &mut counter, &mut mapping)
            } else {
                group
            }
        })
        .into_owned();
    (out, mapping)
}

/// Replace multi-word quotes with placeholders (non-fiction preservation).
pub fn mask_quoted_passages(text: &str) -> (String, BTreeMap<String, String>) {
    let spans = quote_spans(text);
    if spans.is_empty() {
        return (text.to_string(), BTreeMap::new());
    }
    let mut mapping: BTreeMap<String, String> = BTreeMap::new();
    let mut counter = 0usize;
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for (start, end, inner) in spans {
        if start < last {
            continue;
        }
        out.push_str(&text[last..start]);
        if is_multiword_quote(&inner) {
            let placeholder = format!("[[QUOTE_{}]]", counter);
            counter += 1;
            mapping.insert(placeholder.clone(), text[start..end].to_string());
            out.push_str(&placeholder);
        } else {
            out.push_str(&text[start..end]);
        }
        last = end;
    }
    out.push_str(&text[last..]);
    (out, mapping)
}

/// Restore placeholders produced by any masking helper.
pub fn restore_placeholders(text: &str, mapping: &BTreeMap<String, String>) -> String {
    if mapping.is_empty() {
        return text.to_string();
    }
    let mut out = text.to_string();
    for (placeholder, original) in mapping {
        if out.contains(placeholder.as_str()) {
            out = out.replace(placeholder.as_str(), original);
        }
    }
    out
}

