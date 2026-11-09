//! Deterministic locale spelling enforcement.
//!
//! Faithful port of `enforce_local_spelling`, `enforce_local_spelling_guarded`,
//! `build_local_spelling_map`, and the noun/verb disambiguation helpers from
//! `apply_fingerprint.py`.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z][A-Za-z']+").unwrap());
static INLINE_CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(``[^`\n]+``|`[^`\n]+`)").unwrap());

/// Locales that distinguish noun/verb forms (practice vs practise).
pub const NOUN_VERB_LOCALES: &[&str] = &["canadian", "british", "australian"];

fn is_upper_word(w: &str) -> bool {
    w.chars().any(|c| c.is_alphabetic())
        && w.chars().all(|c| !c.is_alphabetic() || c.is_uppercase())
}

fn is_lower_word(w: &str) -> bool {
    w.chars().any(|c| c.is_alphabetic())
        && w.chars().all(|c| !c.is_alphabetic() || c.is_lowercase())
}

fn is_cap_word(w: &str) -> bool {
    let mut it = w.chars();
    match it.next() {
        Some(f) if f.is_uppercase() => {
            it.all(|c| !c.is_alphabetic() || c.is_lowercase())
        }
        _ => false,
    }
}

fn capitalize(w: &str) -> String {
    let mut it = w.chars();
    match it.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &it.as_str().to_lowercase(),
        None => String::new(),
    }
}

fn apply_case(template: &str, replacement: &str) -> String {
    if is_upper_word(template) {
        replacement.to_uppercase()
    } else if is_cap_word(template) {
        capitalize(replacement)
    } else if is_lower_word(template) {
        replacement.to_lowercase()
    } else {
        replacement.to_string()
    }
}

fn is_mixed_case(w: &str) -> bool {
    if is_lower_word(w) || is_upper_word(w) {
        false
    } else {
        !is_cap_word(w)
    }
}

fn spelling_forms(base: &str, suffix: &str) -> (String, Option<String>) {
    let regular = format!("{base}{suffix}");
    let mut dropped = None;
    if !suffix.is_empty() && base.ends_with('e') && base.chars().count() > 1 {
        let first = suffix.chars().next().unwrap().to_ascii_lowercase();
        if "aeiouy".contains(first) {
            let mut trimmed = base.to_string();
            trimmed.pop();
            dropped = Some(format!("{trimmed}{suffix}"));
        }
    }
    (regular, dropped)
}

fn step_back(text: &str, mut i: usize, n: usize) -> usize {
    for _ in 0..n {
        if i == 0 {
            break;
        }
        i -= 1;
        while i > 0 && !text.is_char_boundary(i) {
            i -= 1;
        }
    }
    i
}

fn step_fwd(text: &str, mut i: usize, n: usize) -> usize {
    for _ in 0..n {
        if i >= text.len() {
            break;
        }
        i += 1;
        while i < text.len() && !text.is_char_boundary(i) {
            i += 1;
        }
    }
    i
}

fn window_around(text: &str, start: usize, end: usize, pad: usize) -> &str {
    &text[step_back(text, start, pad)..step_fwd(text, end, pad)]
}

fn prev_non_space(text: &str, idx: usize) -> Option<char> {
    let mut i = idx;
    while i > 0 {
        i = step_back(text, i, 1);
        let c = text[i..].chars().next()?;
        if !c.is_whitespace() {
            return Some(c);
        }
    }
    None
}

fn is_sentence_start(text: &str, idx: usize) -> bool {
    match prev_non_space(text, idx) {
        None => true,
        Some(c) => matches!(c, '.' | '!' | '?' | '\n'),
    }
}

fn looks_like_path(text: &str, start: usize, end: usize) -> bool {
    let window = window_around(text, start, end, 6);
    if window.contains("://")
        || window.starts_with("./")
        || window.starts_with("../")
        || window.starts_with("~/")
    {
        return true;
    }
    if window.contains('/') || window.contains('\\') {
        return true;
    }
    if prev_non_space(text, start) == Some('.') {
        let back = &text[step_back(text, start, 16)..start];
        if back.contains('/') || back.contains('\\') {
            return true;
        }
    }
    let back: Vec<char> = text[step_back(text, start, 3)..start].chars().collect();
    if back.len() >= 2 && back[back.len() - 2].is_alphabetic() && back[back.len() - 1] == ':' {
        return true;
    }
    false
}

fn split_possessive(word: &str) -> Option<(String, String)> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    if n >= 2 {
        let tail: String = chars[n - 2..].iter().collect();
        if tail == "'s" || tail == "\u{2019}s" {
            return Some((chars[..n - 2].iter().collect(), tail));
        }
        if n > 2 && (tail == "s'" || tail == "s\u{2019}") {
            return Some((chars[..n - 2].iter().collect(), tail));
        }
    }
    None
}

/// A single context-sensitive spelling rule (noun/verb or domain disambiguation).
#[derive(Debug, Clone, Default)]
pub struct ContextRule {
    pub variants: Vec<String>,
    pub target: String,
    pub apply_any: Vec<String>,
    pub avoid_any: Vec<String>,
    pub block_any: Vec<String>,
    pub window: usize,
}

/// Parsed locale spelling rules (direct, suffix, context, and double-l).
#[derive(Debug, Clone, Default)]
pub struct SpellingRules {
    pub direct_variants: Vec<(String, Value)>,
    pub suffix_variants: Vec<Value>,
    pub context_entries: Vec<Value>,
    pub double_l_bases: Vec<String>,
    pub double_l_suffixes: Vec<String>,
}

fn lower_list(value: Option<&Value>) -> Vec<String> {
    match value.and_then(Value::as_array) {
        Some(items) => items
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.to_lowercase()),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .collect(),
        None => Vec::new(),
    }
}

fn any_list(entry: &Value, key: &str) -> Vec<String> {
    lower_list(entry.get(key).and_then(|v| v.get("any")))
}

impl SpellingRules {
    /// Build rules from an already-parsed rules document.
    ///
    /// Rules always come from the embedded `resources/config.local_spelling_rules.json`
    /// document; there is deliberately no file-reading constructor, because the
    /// server must not depend on external configuration at runtime.
    pub fn from_json(root: &Value) -> Self {
        let conf = root.get("rules").unwrap_or(root);
        let mut out = Self::default();
        if let Some(items) = conf.get("direct_variants").and_then(Value::as_array) {
            for entry in items {
                if let Some(variants) = entry.get("variants").and_then(Value::as_object) {
                    for value in variants.values() {
                        if let Some(s) = value.as_str() {
                            out.direct_variants
                                .push((s.to_lowercase(), entry.clone()));
                        }
                    }
                }
            }
        }
        if let Some(items) = conf.get("suffix_variants").and_then(Value::as_array) {
            out.suffix_variants = items.clone();
        }
        if let Some(items) = conf.get("context_variants").and_then(Value::as_array) {
            out.context_entries = items.clone();
        }
        if let Some(double_l) = conf.get("double_l_inflection") {
            out.double_l_bases = lower_list(double_l.get("bases"));
            out.double_l_suffixes = lower_list(double_l.get("suffixes"));
        }
        out
    }

    /// True when no rule data was loaded.
    pub fn is_empty(&self) -> bool {
        self.direct_variants.is_empty()
            && self.suffix_variants.is_empty()
            && self.context_entries.is_empty()
            && self.double_l_bases.is_empty()
    }
}

/// Context-sensitive rules applicable to `locale`.
pub fn context_rules_for(rules: &SpellingRules, locale: &str) -> Vec<ContextRule> {
    let mut out = Vec::new();
    for entry in &rules.context_entries {
        let variants = match entry.get("variants").and_then(Value::as_object) {
            Some(v) => v,
            None => continue,
        };
        let target = match variants.get(locale).and_then(Value::as_str) {
            Some(t) => t.to_string(),
            None => continue,
        };
        let names: Vec<String> = variants
            .values()
            .filter_map(Value::as_str)
            .map(|s| s.to_lowercase())
            .collect();
        out.push(ContextRule {
            variants: names,
            target,
            apply_any: any_list(entry, "apply_if"),
            avoid_any: any_list(entry, "avoid_if"),
            block_any: any_list(entry, "block_if"),
            window: entry.get("window").and_then(Value::as_u64).unwrap_or(6) as usize,
        });
    }
    out
}

/// Direct and suffix variant maps for `locale` (source spelling -> target spelling).
pub fn build_local_spelling_map(
    rules: &SpellingRules,
    locale: &str,
) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut direct: HashMap<String, String> = HashMap::new();
    let mut suffix: HashMap<String, String> = HashMap::new();
    if rules.is_empty() {
        return (direct, suffix);
    }
    for entry in &rules.direct_variants {
        let variants = match entry.1.get("variants").and_then(Value::as_object) {
            Some(v) => v,
            None => continue,
        };
        let target = match variants.get(locale).and_then(Value::as_str) {
            Some(t) => t.to_string(),
            None => continue,
        };
        for variant in variants.values() {
            if let Some(s) = variant.as_str() {
                direct.insert(s.to_lowercase(), target.clone());
            }
        }
    }
    for entry in &rules.suffix_variants {
        let variants = match entry.get("variants").and_then(Value::as_object) {
            Some(v) => v,
            None => continue,
        };
        let target_base = match variants.get(locale).and_then(Value::as_str) {
            Some(t) => t.to_string(),
            None => continue,
        };
        let suffixes = match entry.get("suffixes").and_then(Value::as_array) {
            Some(s) => s,
            None => continue,
        };
        for variant_base in variants.values() {
            let base = match variant_base.as_str() {
                Some(s) => s,
                None => continue,
            };
            for suffix_value in suffixes {
                let suf = match suffix_value.as_str() {
                    Some(s) => s,
                    None => continue,
                };
                let (src_regular, src_drop) = spelling_forms(base, suf);
                let (dst_regular, dst_drop) = spelling_forms(&target_base, suf);
                suffix.insert(src_regular.to_lowercase(), dst_regular.clone());
                if let Some(drop) = src_drop {
                    suffix.insert(
                        drop.to_lowercase(),
                        dst_drop.unwrap_or_else(|| dst_regular.clone()),
                    );
                }
            }
        }
    }
    (direct, suffix)
}

/// Apply the same US-baseline mapping used by the Python `normalize_text_spelling`.
pub fn normalize_text_spelling(token: &str, rules: &SpellingRules) -> String {
    let (direct, suffix) = build_local_spelling_map(rules, "us");
    if direct.is_empty() && suffix.is_empty() {
        return token.to_string();
    }
    let lower = token.to_lowercase();
    let (base, poss) = match split_possessive(token) {
        Some((b, p)) => (b, Some(p)),
        None => (token.to_string(), None),
    };
    let base_lower = base.to_lowercase();
    let target = direct
        .get(&base_lower)
        .or_else(|| suffix.get(&base_lower))
        .cloned();
    match target {
        Some(t) => {
            let mut out = apply_case(&base, &t);
            if let Some(p) = poss {
                out.push_str(&p);
            }
            out
        }
        None => {
            let _ = lower;
            token.to_string()
        }
    }
}

fn probable_verb(prev: &str, next: &str) -> bool {
    let p = prev.to_lowercase();
    let n = next.to_lowercase();
    const AUX: &[&str] = &[
        "to", "can", "could", "may", "might", "must", "shall", "should", "will", "would", "do",
        "does", "did", "don't", "doesn't", "didn't", "cannot", "can't", "won't", "shalln't",
        "shouldn't", "wouldn't", "couldn't",
    ];
    AUX.contains(&p.as_str()) || matches!(n.as_str(), "to" | "me" | "him" | "her" | "them" | "it")
}

fn probable_noun(prev: &str, next: &str) -> bool {
    let p = prev.to_lowercase();
    const DET: &[&str] = &[
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our",
        "their", "its",
    ];
    if DET.contains(&p.as_str()) || p.ends_with("'s") {
        return true;
    }
    matches!(next.to_lowercase().as_str(), "of" | "for" | "in" | "on")
}

fn normalize_practise_license(word: &str, prev: &str, next: &str) -> (String, bool) {
    let lower = word.to_lowercase();
    if lower == "practice" || lower == "practise" {
        if probable_verb(prev, next) {
            return (apply_case(word, "practise"), lower != "practise");
        }
        if probable_noun(prev, next) {
            return (apply_case(word, "practice"), lower != "practice");
        }
        return (word.to_string(), false);
    }
    if lower == "license" || lower == "licence" {
        if probable_verb(prev, next) {
            return (apply_case(word, "license"), lower != "license");
        }
        if probable_noun(prev, next) {
            return (apply_case(word, "licence"), lower != "licence");
        }
        return (word.to_string(), false);
    }
    (word.to_string(), false)
}

fn context_has(tokens: &[String], idx: usize, keywords: &[String], window: usize) -> bool {
    if keywords.is_empty() {
        return false;
    }
    let start = idx.saturating_sub(window);
    let end = (idx + window + 1).min(tokens.len());
    let set: BTreeSet<&str> = tokens[start..end].iter().map(|s| s.as_str()).collect();
    keywords.iter().any(|k| set.contains(k.as_str()))
}

/// Decide whether a token must be left alone (placeholders, identifiers, paths).
fn should_skip_spelling(
    text: &str,
    start: usize,
    end: usize,
    word: &str,
    direct_map: &HashMap<String, String>,
    suffix_map: &HashMap<String, String>,
    context_tokens: &BTreeSet<String>,
) -> bool {
    let window = window_around(text, start, end, 3);
    if window.contains("__") || (is_upper_word(word) && word.chars().count() > 2) {
        return true;
    }
    if window.contains('/') || window.contains('\\') || window.contains('@') {
        return true;
    }
    let before = &text[step_back(text, start, 8)..start];
    if before.to_lowercase().contains("http") || looks_like_path(text, start, end) {
        return true;
    }
    if is_mixed_case(word) {
        return true;
    }
    if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        && !is_sentence_start(text, start)
    {
        let lower = word.to_lowercase();
        if !direct_map.contains_key(&lower)
            && !suffix_map.contains_key(&lower)
            && !context_tokens.contains(&lower)
        {
            return true;
        }
    }
    let prev_ch = if start > 0 {
        text[..start].chars().next_back().unwrap_or(' ')
    } else {
        ' '
    };
    let next_ch = text[end..].chars().next().unwrap_or(' ');
    prev_ch.is_alphanumeric()
        || prev_ch == '_'
        || next_ch.is_alphanumeric()
        || next_ch == '_'
}

/// Noun/verb disambiguation for `practice`/`practise` and `license`/`licence`.
fn noun_verb_override(
    word: &str,
    prev: &str,
    next: &str,
    locale: &str,
    noun_verb_locale: bool,
) -> Option<String> {
    if noun_verb_locale {
        let (updated, changed) = normalize_practise_license(word, prev, next);
        return if changed { Some(updated) } else { None };
    }
    if locale == "us" {
        let lower = word.to_lowercase();
        let target = match lower.as_str() {
            "licence" | "licences" => Some(lower.replacen("ce", "se", 1)),
            "practise" | "practises" | "practised" | "practising" => {
                Some(lower.replacen("se", "ce", 1))
            }
            _ => None,
        };
        if let Some(t) = target {
            return Some(apply_case(word, &t));
        }
    }
    None
}

/// Apply context, direct/suffix, then double-l rules to a single token.
#[allow(clippy::too_many_arguments)]
fn spelling_replacement(
    word: &str,
    idx: usize,
    tokens: &[String],
    locale: &str,
    noun_verb_locale: bool,
    direct_map: &HashMap<String, String>,
    suffix_map: &HashMap<String, String>,
    context_rules: &[ContextRule],
    rules: &SpellingRules,
) -> Option<String> {
    let (base, poss) = match split_possessive(word) {
        Some((b, p)) => (b, Some(p)),
        None => (word.to_string(), None),
    };
    let lower = base.to_lowercase();
    let with_possessive = |target: &str| {
        let mut out = apply_case(&base, target);
        if let Some(p) = &poss {
            out.push_str(p);
        }
        out
    };
    for rule in context_rules {
        if !rule.variants.contains(&lower) {
            continue;
        }
        if !rule.block_any.is_empty() && context_has(tokens, idx, &rule.block_any, rule.window) {
            return None;
        }
        if !rule.apply_any.is_empty() && !context_has(tokens, idx, &rule.apply_any, rule.window) {
            continue;
        }
        if !rule.avoid_any.is_empty() && context_has(tokens, idx, &rule.avoid_any, rule.window) {
            return None;
        }
        if lower != rule.target.to_lowercase() {
            return Some(with_possessive(&rule.target));
        }
        return None;
    }
    if let Some(target) = direct_map.get(&lower).or_else(|| suffix_map.get(&lower)) {
        return Some(with_possessive(target));
    }
    for base_word in &rules.double_l_bases {
        for suffix in &rules.double_l_suffixes {
            let us_form = format!("{base_word}{suffix}");
            let mut stem = base_word.clone();
            stem.pop();
            let double_form = format!("{stem}ll{suffix}");
            if lower == us_form && noun_verb_locale {
                return Some(apply_case(word, &double_form));
            }
            if lower == double_form && locale == "us" {
                return Some(apply_case(word, &us_form));
            }
        }
    }
    None
}

/// Apply locale spelling rules to plain text, returning the new text and the
/// number of replacements (port of `enforce_local_spelling`).
pub fn enforce_local_spelling(text: &str, locale: &str, rules: &SpellingRules) -> (String, usize) {
    if text.is_empty() || locale == "none" || rules.is_empty() {
        return (text.to_string(), 0);
    }
    let (direct_map, suffix_map) = build_local_spelling_map(rules, locale);
    let context_rules = context_rules_for(rules, locale);
    let mut context_tokens: BTreeSet<String> = BTreeSet::new();
    for rule in &context_rules {
        for v in &rule.variants {
            context_tokens.insert(v.clone());
        }
    }
    let matches: Vec<(usize, usize, String)> = WORD_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect();
    if matches.is_empty() {
        return (text.to_string(), 0);
    }
    let tokens: Vec<String> = matches.iter().map(|(_, _, w)| w.to_lowercase()).collect();
    let noun_verb_locale = NOUN_VERB_LOCALES.contains(&locale);
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut replacements = 0usize;
    for (idx, (start, end, word)) in matches.iter().enumerate() {
        out.push_str(&text[last..*start]);
        last = *end;
        if should_skip_spelling(text, *start, *end, word, &direct_map, &suffix_map, &context_tokens) {
            out.push_str(word);
            continue;
        }
        let prev_word = if idx > 0 { tokens[idx - 1].as_str() } else { "" };
        let next_word = tokens.get(idx + 1).map(|s| s.as_str()).unwrap_or("");
        let updated = noun_verb_override(word, prev_word, next_word, locale, noun_verb_locale)
            .or_else(|| {
                spelling_replacement(
                    word,
                    idx,
                    &tokens,
                    locale,
                    noun_verb_locale,
                    &direct_map,
                    &suffix_map,
                    &context_rules,
                    rules,
                )
            });
        match updated {
            Some(replacement) => {
                replacements += 1;
                out.push_str(&replacement);
            }
            None => out.push_str(word),
        }
    }
    out.push_str(&text[last..]);
    (out, replacements)
}

