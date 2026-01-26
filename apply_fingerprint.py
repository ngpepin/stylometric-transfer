#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
"""
apply_fingerprint.py

Rewrite a Markdown file to match a style fingerprint JSON.

It:
1) Loads fingerprint JSON
2) Computes measurements of the input markdown (so the LLM can see deltas)
3) Calls an OpenAI-compatible LLM to rewrite the markdown
4) Writes rewritten markdown to an output file (default: <input>.styled.md)

Usage:
  python apply_fingerprint.py -f fingerprint.json -i draft.md
  python apply_fingerprint.py -f fingerprint.json -i draft.md -o draft.styled.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import collections
import os
import datetime
import time
import random
from pathlib import Path
import copy
from typing import Any, Dict, List

import requests


# ---- same lightweight stats as fingerprint script ----
# Script overview:
# - Measure the input Markdown (lightweight, explainable stats)
# - Build a prompt using an externalized template (prompts.json)
# - Call an OpenAI-compatible LLM to rewrite while preserving meaning
# - Score style compliance and optionally retry with delta feedback
# - Handle oversized prompts by chunking the Markdown
# - Strip base64 images before prompting, then reinsert after rewriting
# - Preserve non-voice blocks (blockquotes, references, footnotes, citations) verbatim

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
BASE64_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\\s]+", re.IGNORECASE)
BASE64_PLACEHOLDER_RE = re.compile(r"\[\[BASE64_IMAGE_\d+\]\]")
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
HUMANIZER_GUIDELINES_FILENAME = "general-guidelines.md"
HUMANIZER_CACHE_FILENAME = "humanizer_rules.cache.json"
TUNABLES_FILENAME = "config.tunables.json"
AVOID_LIST_FILENAME = "config.avoid.txt"
EM_DASH_CHAR = "—"
ANSI_RED = "\x1b[31m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RESET = "\x1b[0m"
DEFAULT_TUNABLES = {
    "humanizer_conflicts": {
        "em_dash_keep_rate": 0.5,
        "hedge_keep_rate": 1.0,
        "first_person_keep_rate": 0.5,
        "contractions_avoid_threshold": 2.0,
        "contractions_use_threshold": 0.5,
        "heading_title_case_keep_rate": 0.6,
        "boldface_keep_per_1000w": 3.0,
        "inline_header_list_keep_rate": 0.2
    },
    "sanity_checks": {
        "line_count_warn_pct": 10.0,
        "word_count_warn_pct": 10.0,
        "paragraph_count_warn_pct": 10.0
    }
}

REFERENCE_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "citations",
    "sources",
    "endnotes",
    "footnotes",
    "notes"
}
ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
SETEXT_H1_RE = re.compile(r"^\s*=+\s*$")
SETEXT_H2_RE = re.compile(r"^\s*-+\s*$")
BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>")
FOOTNOTE_DEF_RE = re.compile(r"^\s*\[\^[^\]]+\]:")

INLINE_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
INLINE_NUMERIC_CITE_RE = re.compile(r"\[(?:\d+|[IVX]+)(?:\s*[-–,;]\s*(?:\d+|[IVX]+))*\]")
PAREN_GROUP_RE = re.compile(r"\([^()]{1,80}\)")
FROZEN_BLOCK_RE = re.compile(r"\[\[FROZEN_BLOCK_\d+\]\]")
CITATION_PLACEHOLDER_RE = re.compile(r"\[\[CITATION_\d+\]\]")
INLINE_CODE_PLACEHOLDER_RE = re.compile(r"\[\[INLINE_CODE_\d+\]\]")
HTML_PLACEHOLDER_RE = re.compile(r"\[\[HTML_BLOCK_\d+\]\]")
INLINE_MATH_PLACEHOLDER_RE = re.compile(r"\[\[INLINE_MATH_\d+\]\]")
DISPLAY_MATH_PLACEHOLDER_RE = re.compile(r"\[\[DISPLAY_MATH_\d+\]\]")
HTML_ENTITY_PLACEHOLDER_RE = re.compile(r"\[\[HTML_ENTITY_\d+\]\]")

SECTION_HEADING_RE = re.compile(r"^###\s+(\d+\\.)?\s*(.+)$")
WORDS_TO_WATCH_RE = re.compile(r"^\*\*Words to watch:\*\*\s*(.+)$")
PROBLEM_RE = re.compile(r"^\*\*Problem:\*\*\s*(.+)$")


def load_prompts() -> Dict[str, Any]:
    # Load externalized prompt templates located alongside this script.
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.json not found at {PROMPTS_PATH}")
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def colorize(text: str, color: str, stream: Any) -> str:
    if os.getenv("NO_COLOR") or os.getenv("CLICOLOR") == "0":
        return text
    return f"{color}{text}{ANSI_RESET}"


def print_warn(msg: str) -> None:
    print(colorize(msg, ANSI_YELLOW, sys.stderr), file=sys.stderr)


def print_error(msg: str) -> None:
    print(colorize(msg, ANSI_RED, sys.stderr), file=sys.stderr)


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_tunables(path: Path | None = None) -> Dict[str, Any]:
    # Load optional tunables JSON.
    if path and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return deep_merge_dict(DEFAULT_TUNABLES, data)
        except Exception:
            return dict(DEFAULT_TUNABLES)
    # Fallback search.
    cwd_path = Path.cwd() / TUNABLES_FILENAME
    script_path = Path(__file__).resolve().parent / TUNABLES_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    if not path:
        return dict(DEFAULT_TUNABLES)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return deep_merge_dict(DEFAULT_TUNABLES, data)
    except Exception:
        pass
    return dict(DEFAULT_TUNABLES)


def parse_avoid_list(text: str) -> List[str]:
    items: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


def load_avoid_list(path: Path | None = None) -> List[str]:
    # Load optional avoid-word list from CWD or script directory.
    if path and path.exists():
        try:
            return parse_avoid_list(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    cwd_path = Path.cwd() / AVOID_LIST_FILENAME
    script_path = Path(__file__).resolve().parent / AVOID_LIST_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    if not path:
        return []
    try:
        return parse_avoid_list(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def merge_avoid_list_into_fingerprint(
    fingerprint: Dict[str, Any],
    avoid_list: List[str]
) -> Dict[str, Any]:
    if not avoid_list or not isinstance(fingerprint, dict):
        return fingerprint
    lexicon = fingerprint.get("lexicon")
    if not isinstance(lexicon, dict):
        lexicon = {}
        fingerprint["lexicon"] = lexicon

    existing = lexicon.get("avoid_words")
    merged: List[str] = []
    seen = set()

    def add_item(item: Any) -> None:
        if not isinstance(item, str):
            return
        if item not in seen:
            seen.add(item)
            merged.append(item)

    if isinstance(existing, list):
        for item in existing:
            add_item(item)
    elif isinstance(existing, str):
        add_item(existing)

    for item in avoid_list:
        add_item(item)

    if merged:
        lexicon["avoid_words"] = merged
    return fingerprint


def should_forbid_em_dashes(fingerprint: Dict[str, Any], avoid_list: List[str]) -> bool:
    # Forbid if targets explicitly set em-dash rate to zero or avoid list includes em-dash tokens.
    if not isinstance(fingerprint, dict):
        return False
    targets = fingerprint.get("targets", {}) if isinstance(fingerprint, dict) else {}
    punctuation = targets.get("punctuation", {}) if isinstance(targets, dict) else {}
    em_target = punctuation.get("em_dashes_per_1000w", {}) if isinstance(punctuation, dict) else {}
    target_range = em_target.get("target") if isinstance(em_target, dict) else None
    if isinstance(target_range, list) and len(target_range) >= 2:
        try:
            if float(target_range[1]) <= 0.0:
                return True
        except (TypeError, ValueError):
            pass

    lexicon = fingerprint.get("lexicon", {}) if isinstance(fingerprint, dict) else {}
    avoid_words = [w.lower() for w in lexicon.get("avoid_words", []) if isinstance(w, str)]
    avoid_phrases = [w.lower() for w in lexicon.get("avoid_phrases", []) if isinstance(w, str)]
    avoid_list_lower = [w.lower() for w in avoid_list if isinstance(w, str)]
    avoid_tokens = set(avoid_words + avoid_phrases + avoid_list_lower)
    if EM_DASH_CHAR in avoid_tokens or "em dash" in avoid_tokens or "em-dash" in avoid_tokens:
        return True
    return False


def enforce_no_em_dashes(text: str) -> tuple[str, int]:
    # Replace em dashes with a spaced hyphen to preserve readability without em-dash glyphs.
    if EM_DASH_CHAR not in text:
        return text, 0
    count = text.count(EM_DASH_CHAR)
    text = re.sub(r"\s*—\s*", " - ", text)
    return text, count

def resolve_general_guidelines_path() -> Path | None:
    # Resolve optional humanizer guidelines from CWD or script directory.
    cwd_path = Path.cwd() / HUMANIZER_GUIDELINES_FILENAME
    script_path = Path(__file__).resolve().parent / HUMANIZER_GUIDELINES_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    return path


def load_general_guidelines() -> tuple[str | None, Path | None]:
    path = resolve_general_guidelines_path()
    if not path:
        return None, None
    return path.read_text(encoding="utf-8"), path

def get_prompt_value(prompts: Dict[str, Any], *path: str) -> Any:
    # Traverse a nested dict safely and fail fast if a key is missing.
    cur: Any = prompts
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing prompts key: {'.'.join(path)}")
        cur = cur[key]
    return cur

def words(text: str) -> List[str]:
    # Tokenize into simple word-like units for lightweight stats.
    return WORD_RE.findall(text)

def split_sentences(text: str) -> List[str]:
    # Naive sentence splitter to keep stats interpretable and dependency-free.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]

def split_paragraphs(text: str) -> List[str]:
    # Paragraphs separated by blank lines.
    paras = PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]

def histogram(values: List[int], bins: List[tuple[int, int | None]]) -> List[float]:
    # Convert a list of values into a probability histogram over bins.
    if not values:
        return [0.0] * len(bins)
    counts = [0] * len(bins)
    for v in values:
        for i, (lo, hi) in enumerate(bins):
            if v >= lo and (hi is None or v <= hi):
                counts[i] += 1
                break
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(bins)


def strip_base64_images(text: str) -> tuple[str, Dict[str, str]]:
    # Replace base64 images with placeholders to avoid prompt token blowups.
    mapping: Dict[str, str] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"[[BASE64_IMAGE_{counter}]]"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    stripped = BASE64_IMAGE_RE.sub(repl, text)
    return stripped, mapping


def restore_base64_images(text: str, mapping: Dict[str, str], placeholders: List[str] | None = None) -> str:
    # Reinsert the original base64 image strings where placeholders appear.
    if not mapping:
        return text
    if not placeholders:
        placeholders = list(mapping.keys())
    for placeholder in placeholders:
        if placeholder in text:
            text = text.replace(placeholder, mapping[placeholder])
    return text


def find_base64_placeholders(text: str) -> List[str]:
    # Helper to detect placeholders in rewritten output.
    return re.findall(r"\[\[BASE64_IMAGE_\d+\]\]", text)


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def get_heading_at(lines: List[str], idx: int) -> tuple[int, str, int] | None:
    # Return (level, heading_text, span_lines) if a heading starts at idx.
    line = lines[idx]
    m = ATX_HEADING_RE.match(line)
    if m:
        return (len(m.group(1)), m.group(2).strip(), 1)
    if idx + 1 < len(lines):
        underline = lines[idx + 1]
        if SETEXT_H1_RE.match(underline):
            return (1, line.strip(), 2)
        if SETEXT_H2_RE.match(underline):
            return (2, line.strip(), 2)
    return None


def is_reference_heading(text: str) -> bool:
    return normalize_heading_text(text) in REFERENCE_HEADINGS


def find_reference_sections(lines: List[str]) -> List[tuple[int, int]]:
    # Return list of (start_idx, end_idx) line ranges for reference-like sections.
    sections: List[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        h = get_heading_at(lines, i)
        if not h:
            i += 1
            continue
        level, heading_text, span = h
        if is_reference_heading(heading_text):
            start = i
            i += span
            end = len(lines)
            j = i
            while j < len(lines):
                next_h = get_heading_at(lines, j)
                if next_h and next_h[0] <= level:
                    end = j
                    break
                j += 1
            sections.append((start, end))
            i = end
            continue
        i += span
    return sections


def strip_non_voice_sections(text: str) -> str:
    # Remove blockquotes, reference sections, and footnote definitions.
    text = re.sub(r"(?is)<blockquote[^>]*>.*?</blockquote>", "\n", text)
    lines = text.splitlines()
    ref_sections = find_reference_sections(lines)
    ref_iter = iter(ref_sections)
    current_ref = next(ref_iter, None)
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        if current_ref and i == current_ref[0]:
            i = current_ref[1]
            current_ref = next(ref_iter, None)
            continue
        line = lines[i]
        if BLOCKQUOTE_LINE_RE.match(line):
            i += 1
            continue
        if FOOTNOTE_DEF_RE.match(line):
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                i += 1
            continue
        out_lines.append(line)
        i += 1
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_fenced_code_blocks(text: str) -> str:
    # Remove fenced code blocks (``` or ~~~) entirely.
    lines = text.splitlines()
    out: List[str] = []
    in_code = False
    fence = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_code:
                in_code = True
                fence = stripped[:3]
            else:
                if stripped.startswith(fence):
                    in_code = False
                    fence = ""
            continue
        if not in_code:
            out.append(line)
    return "\n".join(out)


def strip_inline_code(text: str) -> str:
    # Remove inline code spans delimited by backticks.
    return re.sub(r"(``[^`]+``|`[^`]+`)", "", text)


def strip_html(text: str) -> str:
    # Remove HTML tags and block elements to exclude HTML from profiling.
    text = re.sub(r"(?is)<[A-Za-z][^>]*>.*?</[A-Za-z][^>]*>", "\n", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = re.sub(r"<[A-Za-z/][^>]*>", "", text)
    return text


def strip_latex_math(text: str) -> str:
    # Remove LaTeX-style inline/display math.
    text = re.sub(r"(?s)\\\[.*?\\\]", "\n", text)
    text = re.sub(r"(?s)\\\(.*?\\\)", "", text)
    text = re.sub(r"(?s)\$\$.*?\$\$", "\n", text)
    text = re.sub(r"(?s)\$[^$]*\$", "", text)
    text = re.sub(r"(?s)\\begin\\{[^}]+\\}.*?\\end\\{[^}]+\\}", "\n", text)
    # Remove bare LaTeX command sequences outside delimiters.
    text = re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", "", text)
    return text


def strip_html_entities(text: str) -> str:
    # Remove HTML entities (e.g., &nbsp;).
    return re.sub(r"&[A-Za-z0-9#]+;", "", text)


def is_parenthetical_citation(inner: str) -> bool:
    if not re.search(r"\b(19|20)\d{2}[a-z]?\b", inner):
        return False
    if re.search(r"\bet\s+al\.?\b", inner, re.IGNORECASE):
        return True
    if re.search(r"\b(cf\.|see|see also)\b", inner, re.IGNORECASE):
        return True
    if "," in inner or ";" in inner:
        return True
    m = re.search(r"\b([A-Z][A-Za-z'’.-]+)\s+(19|20)\d{2}[a-z]?\b", inner)
    if m:
        month = m.group(1).lower()
        if month not in {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        }:
            return True
    return False


def strip_inline_citations(text: str) -> str:
    # Remove inline citation markers while keeping surrounding prose.
    text = INLINE_FOOTNOTE_RE.sub("", text)
    text = INLINE_NUMERIC_CITE_RE.sub("", text)

    def repl(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        return "" if is_parenthetical_citation(inner) else match.group(0)

    text = PAREN_GROUP_RE.sub(repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def filter_author_voice_text(text: str) -> str:
    # Remove non-author voice segments and inline citations for measurements.
    text = strip_fenced_code_blocks(text)
    text = strip_inline_code(text)
    text = strip_non_voice_sections(text)
    text = strip_latex_math(text)
    text = strip_html(text)
    text = strip_html_entities(text)
    text = BASE64_PLACEHOLDER_RE.sub("", text)
    text = strip_inline_citations(text)
    text = FROZEN_BLOCK_RE.sub("", text)
    text = CITATION_PLACEHOLDER_RE.sub("", text)
    text = INLINE_CODE_PLACEHOLDER_RE.sub("", text)
    text = INLINE_MATH_PLACEHOLDER_RE.sub("", text)
    text = DISPLAY_MATH_PLACEHOLDER_RE.sub("", text)
    text = HTML_ENTITY_PLACEHOLDER_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def mask_non_voice_blocks(markdown: str) -> tuple[str, Dict[str, str]]:
    # Replace non-voice blocks with placeholders to preserve them verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    def make_placeholder(block: str) -> str:
        nonlocal counter
        placeholder = f"[[FROZEN_BLOCK_{counter}]]"
        mapping[placeholder] = block
        counter += 1
        return placeholder

    def repl_html(match: re.Match[str]) -> str:
        return make_placeholder(match.group(0))

    text = re.sub(r"(?is)<blockquote[^>]*>.*?</blockquote>", repl_html, markdown)
    lines = text.splitlines()
    out_lines: List[str] = []
    ref_sections = find_reference_sections(lines)
    ref_iter = iter(ref_sections)
    current_ref = next(ref_iter, None)
    i = 0
    while i < len(lines):
        if current_ref and i == current_ref[0]:
            block = "\n".join(lines[current_ref[0]:current_ref[1]])
            out_lines.append(make_placeholder(block))
            i = current_ref[1]
            current_ref = next(ref_iter, None)
            continue
        line = lines[i]
        if BLOCKQUOTE_LINE_RE.match(line):
            start = i
            i += 1
            while i < len(lines) and BLOCKQUOTE_LINE_RE.match(lines[i]):
                i += 1
            block = "\n".join(lines[start:i])
            out_lines.append(make_placeholder(block))
            continue
        if FOOTNOTE_DEF_RE.match(line):
            start = i
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                i += 1
            block = "\n".join(lines[start:i])
            out_lines.append(make_placeholder(block))
            continue
        out_lines.append(line)
        i += 1
    return ("\n".join(out_lines), mapping)


def mask_inline_citations(text: str) -> tuple[str, Dict[str, str]]:
    # Replace inline citations with placeholders to preserve them verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    def make_placeholder(match_text: str) -> str:
        nonlocal counter
        placeholder = f"[[CITATION_{counter}]]"
        mapping[placeholder] = match_text
        counter += 1
        return placeholder

    def repl_simple(match: re.Match[str]) -> str:
        return make_placeholder(match.group(0))

    text = INLINE_FOOTNOTE_RE.sub(repl_simple, text)
    text = INLINE_NUMERIC_CITE_RE.sub(repl_simple, text)

    def repl_paren(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        return make_placeholder(match.group(0)) if is_parenthetical_citation(inner) else match.group(0)

    text = PAREN_GROUP_RE.sub(repl_paren, text)
    return text, mapping


def restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    for placeholder, original in mapping.items():
        if placeholder in text:
            text = text.replace(placeholder, original)
    return text


def find_placeholders(text: str, pattern: re.Pattern[str]) -> List[str]:
    return pattern.findall(text)


def normalize_heading(text: str) -> str:
    # Normalize heading text for loose matching (case/punct insensitive).
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_heading_blocks(markdown: str) -> List[Dict[str, Any]]:
    # Extract heading-based section blocks for completeness checks.
    lines = markdown.splitlines()
    headings: List[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        m = ATX_HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append((idx, level, title))

    blocks: List[Dict[str, Any]] = []
    for i, (start, _level, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        key = normalize_heading(title)
        if key and block:
            blocks.append({
                "title": title,
                "key": key,
                "block": block,
                "start_line": start,
                "end_line": end,
                "level": _level
            })
    return blocks


def extract_heading_keys(markdown: str) -> set[str]:
    keys: set[str] = set()
    for line in markdown.splitlines():
        m = ATX_HEADING_RE.match(line)
        if not m:
            continue
        key = normalize_heading(m.group(2))
        if key:
            keys.add(key)
    return keys


def section_signature(block: str, max_lines: int = 6) -> set[str]:
    # Build a content signature from the heading + first few non-empty lines.
    lines = block.splitlines()
    tokens: List[str] = []
    used_lines = 0
    for line in lines:
        if not line.strip():
            continue
        used_lines += 1
        for tok in words(line.lower()):
            if len(tok) >= 4 and tok.isalpha():
                tokens.append(tok)
        if used_lines >= max_lines:
            break
    return set(tokens)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def parse_humanizer_guidelines(text: str) -> List[Dict[str, Any]]:
    # Parse general-guidelines.md into structured rules.
    rules: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    in_voice_section = False
    current_category: str | None = None
    in_frontmatter = False
    frontmatter_done = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not frontmatter_done and line == "---":
            in_frontmatter = not in_frontmatter
            if not in_frontmatter:
                frontmatter_done = True
            continue
        if in_frontmatter:
            continue
        if line.startswith("## ") and not line.startswith("### "):
            current_category = line.replace("##", "").strip()
            continue
        if line == "### How to add voice:":
            in_voice_section = True
            continue
        if line.startswith("### ") and line != "### How to add voice:":
            in_voice_section = False
        m = SECTION_HEADING_RE.match(line)
        if m:
            if current:
                rules.append(current)
            title = m.group(2).strip()
            current = {
                "title": title,
                "problem": None,
                "words_to_watch": [],
                "source": "section",
                "category": current_category
            }
            continue
        if in_voice_section and line.startswith("**") and "**" in line[2:]:
            title = line.strip("*").strip().rstrip(".")
            rules.append({
                "title": title,
                "problem": "Voice/flow guidance",
                "words_to_watch": [],
                "source": "voice",
                "category": current_category
            })
            continue
        if in_voice_section and line.startswith("- "):
            rules.append({
                "title": line.lstrip("-").strip(),
                "problem": "Voice/flow guidance",
                "words_to_watch": [],
                "source": "voice",
                "category": current_category
            })
            continue
        if current:
            m = WORDS_TO_WATCH_RE.match(line)
            if m:
                words = [w.strip() for w in m.group(1).split(",") if w.strip()]
                current["words_to_watch"] = words
                continue
            m = PROBLEM_RE.match(line)
            if m:
                current["problem"] = m.group(1).strip()
                continue

    if current:
        rules.append(current)
    return rules


def normalize_humanizer_rules(rules: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        title = rule.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        problem = rule.get("problem")
        problem = problem if isinstance(problem, str) else None
        words_raw = rule.get("words_to_watch", [])
        if isinstance(words_raw, str):
            words = [w.strip() for w in words_raw.split(",") if w.strip()]
        elif isinstance(words_raw, list):
            words = [w.strip() for w in words_raw if isinstance(w, str) and w.strip()]
        else:
            words = []
        normalized.append({
            "title": title.strip(),
            "problem": problem,
            "words_to_watch": words,
            "source": source,
            "category": rule.get("category")
        })
    return normalized


def build_humanizer_parse_prompt(prompts: Dict[str, Any], raw_guidelines: str) -> List[Dict[str, str]]:
    system = get_prompt_value(prompts, "humanizer_parse", "system")
    user_template = get_prompt_value(prompts, "humanizer_parse", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.humanizer_parse.user must be an object")
    user = copy.deepcopy(user_template)
    user["raw_guidelines"] = raw_guidelines
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def parse_humanizer_guidelines_llm(
    cfg: LLMConfig,
    prompts: Dict[str, Any],
    raw_guidelines: str
) -> List[Dict[str, Any]]:
    messages = build_humanizer_parse_prompt(prompts, raw_guidelines)
    raw = chat_completions(cfg, messages)
    try:
        out_obj = parse_json_strict(raw)
    except Exception:
        out_obj = repair_json_with_llm(cfg, raw, prompts)

    rules_obj: Any = out_obj.get("rules") if isinstance(out_obj, dict) else out_obj
    if not isinstance(rules_obj, list):
        return []
    return normalize_humanizer_rules(rules_obj, "llm")


def load_humanizer_rules_cache(cache_path: Path) -> Dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_humanizer_rules_cache(
    cache_path: Path,
    rules: List[Dict[str, Any]],
    parser: str,
    source_path: Path | None
) -> None:
    payload = {
        "rules": rules,
        "parser": parser,
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_path": str(source_path) if source_path else None,
        "source_mtime": source_path.stat().st_mtime if source_path and source_path.exists() else None
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_markdown_style(text: str) -> Dict[str, Any]:
    # Estimate heading case, boldface density, and inline-header list usage.
    headings_total = 0
    headings_title_case = 0
    bold_count = 0
    word_count = 0
    list_total = 0
    inline_header_list = 0

    blocks = split_markdown_blocks(text)
    for block in blocks:
        if is_code_block(block):
            continue
        lines = block.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                headings_total += 1
                heading_text = stripped.lstrip("#").strip()
                wlist = [re.sub(r"[^A-Za-z]", "", w) for w in heading_text.split()]
                wlist = [w for w in wlist if w]
                if wlist:
                    cap = sum(1 for w in wlist if w[0].isupper())
                    if cap / max(1, len(wlist)) >= 0.6:
                        headings_title_case += 1
            if re.match(r"^\\s*[-*+]\\s+", line) or re.match(r"^\\s*\\d+\\.\\s+", line):
                list_total += 1
                if re.match(r"^\\s*[-*+]\\s+\\*\\*[^*]+:\\*\\*", line) or re.match(r"^\\s*\\d+\\.\\s+\\*\\*[^*]+:\\*\\*", line):
                    inline_header_list += 1
            bold_count += len(re.findall(r"\\*\\*[^*]+\\*\\*", line))
            word_count += len(words(line))

    return {
        "heading_title_case_rate": (headings_title_case / max(1, headings_total)) if headings_total else 0.0,
        "boldface_per_1000w": (bold_count / max(1, word_count)) * 1000.0,
        "inline_header_list_rate": (inline_header_list / max(1, list_total)) if list_total else 0.0
    }


def filter_humanizer_rules(
    rules: List[Dict[str, Any]],
    fingerprint: Dict[str, Any],
    input_style: Dict[str, Any] | None = None,
    tunables: Dict[str, Any] | None = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Drop rules that deterministically conflict with the fingerprint signals.
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    tunables = tunables or DEFAULT_TUNABLES
    conf = tunables.get("humanizer_conflicts", {}) if isinstance(tunables, dict) else {}

    meas = fingerprint.get("measurements", {}) if isinstance(fingerprint, dict) else {}
    punct = meas.get("punctuation", {}).get("rates_per_1000w", {}) if isinstance(meas, dict) else {}
    stance = meas.get("stance_signals", {}) if isinstance(meas, dict) else {}
    ortho = meas.get("orthography_signals", {}) if isinstance(meas, dict) else {}
    templates_signals = meas.get("templates_signals", {}) if isinstance(meas, dict) else {}
    common_phrases_validation = meas.get("common_phrases_validation", {}) if isinstance(meas, dict) else {}

    em_dash_rate = float(punct.get("em_dashes", 0.0)) if isinstance(punct, dict) else 0.0
    contractions_rate = float(ortho.get("contractions_rate", 0.0)) if isinstance(ortho, dict) else 0.0
    hedge_rate = float(stance.get("hedge_rate", 0.0)) if isinstance(stance, dict) else 0.0
    first_person_rate = float(stance.get("first_person_rate", 0.0)) if isinstance(stance, dict) else 0.0

    em_dash_keep_rate = float(conf.get("em_dash_keep_rate", 0.5))
    hedge_keep_rate = float(conf.get("hedge_keep_rate", 1.0))
    first_person_keep_rate = float(conf.get("first_person_keep_rate", 0.5))
    contractions_avoid_threshold = float(conf.get("contractions_avoid_threshold", 2.0))
    contractions_use_threshold = float(conf.get("contractions_use_threshold", 0.5))
    heading_title_case_keep_rate = float(conf.get("heading_title_case_keep_rate", 0.6))
    boldface_keep_per_1000w = float(conf.get("boldface_keep_per_1000w", 3.0))
    inline_header_list_keep_rate = float(conf.get("inline_header_list_keep_rate", 0.2))

    lexicon = fingerprint.get("lexicon", {}) if isinstance(fingerprint, dict) else {}
    preferred_words = set(w.lower() for w in lexicon.get("preferred_words", []) if isinstance(w, str))
    preferred_phrases = set(p.lower() for p in lexicon.get("preferred_phrases", []) if isinstance(p, str))
    avoid_words = set(w.lower() for w in lexicon.get("avoid_words", []) if isinstance(w, str))
    synonym_prefs = lexicon.get("synonym_preferences", {})
    if isinstance(synonym_prefs, dict):
        synonym_keys = set(k.lower() for k in synonym_prefs.keys() if isinstance(k, str))
    elif isinstance(synonym_prefs, list):
        synonym_keys = set(str(k).lower() for k in synonym_prefs)
    else:
        synonym_keys = set()

    transition_top = set(
        (item.get("phrase", "") or "").lower()
        for item in (templates_signals.get("transition_openers_top", []) or [])
        if isinstance(item, dict)
    )
    opener_top = set(
        (item.get("phrase", "") or "").lower()
        for item in (templates_signals.get("sentence_openers_top", []) or [])
        if isinstance(item, dict)
    )

    validated_phrases = set()
    validated = common_phrases_validation.get("validated", {}) if isinstance(common_phrases_validation, dict) else {}
    for item in (validated.get("bigrams_top", []) or []) + (validated.get("trigrams_top", []) or []):
        if isinstance(item, dict) and isinstance(item.get("phrase"), str):
            validated_phrases.add(item["phrase"].lower())

    preferred_set = preferred_words | preferred_phrases | synonym_keys | transition_top | opener_top | validated_phrases

    persona = fingerprint.get("targets", {}).get("persona", {}) if isinstance(fingerprint, dict) else {}
    pronouns = persona.get("pronoun_preferences", {}) if isinstance(persona, dict) else {}
    avoid_sets = pronouns.get("avoid_sets", []) if isinstance(pronouns, dict) else []
    avoid_first_person = isinstance(avoid_sets, list) and any("i" in s.lower() for s in avoid_sets if isinstance(s, str))

    em_dash_forbidden = False
    targets = fingerprint.get("targets", {}) if isinstance(fingerprint, dict) else {}
    punctuation_targets = targets.get("punctuation", {}) if isinstance(targets, dict) else {}
    em_target = punctuation_targets.get("em_dashes_per_1000w", {}) if isinstance(punctuation_targets, dict) else {}
    target_range = em_target.get("target") if isinstance(em_target, dict) else None
    if isinstance(target_range, list) and len(target_range) >= 2:
        try:
            em_dash_forbidden = float(target_range[1]) <= 0.0
        except (TypeError, ValueError):
            em_dash_forbidden = False
    if not em_dash_forbidden and avoid_words:
        if EM_DASH_CHAR in avoid_words or "em dash" in avoid_words or "em-dash" in avoid_words:
            em_dash_forbidden = True

    def collect_style_context() -> str:
        parts: List[str] = []
        for path in ("notes",):
            val = lexicon.get(path)
            if isinstance(val, str):
                parts.append(val)
        templates = fingerprint.get("templates", {}) if isinstance(fingerprint, dict) else {}
        for key in ("syntactic_patterns", "paragraph_moves", "rhetorical_moves"):
            vals = templates.get(key, [])
            if isinstance(vals, list):
                parts.extend(v for v in vals if isinstance(v, str))
        controls = fingerprint.get("controls", {}) if isinstance(fingerprint, dict) else {}
        rewrite_policy = controls.get("rewrite_policy")
        if isinstance(rewrite_policy, str):
            parts.append(rewrite_policy)
        return " ".join(parts).lower()

    style_context = collect_style_context()
    formal_style = any(tag in style_context for tag in ("formal", "academic", "technical", "scholarly"))

    def normalize_word(token: str) -> str:
        token = re.sub(r"[\"“”'’()\\[\\]{}]", "", token.lower()).strip()
        return re.sub(r"\\s+", " ", token)

    def expand_words(words: List[str]) -> set[str]:
        out: set[str] = set()
        for w in words:
            if not w:
                continue
            parts = re.split(r"/|\\bor\\b", w)
            for part in parts:
                part = normalize_word(part)
                if part:
                    out.add(part)
        return out

    for rule in rules:
        title = str(rule.get("title", "")).lower()
        words = " ".join(rule.get("words_to_watch", [])).lower()
        drop_reason = None
        tokens = expand_words(rule.get("words_to_watch", []))

        if "em dash" in title or "em dash" in words:
            if em_dash_forbidden:
                drop_reason = "Em dashes forbidden by fingerprint/avoid list."
            elif em_dash_rate >= em_dash_keep_rate:
                drop_reason = "Author uses em dashes frequently."
        if "hedging" in title or "hedging" in words:
            if hedge_rate >= hedge_keep_rate:
                drop_reason = "Author uses hedging at a high rate."
        if "use \"i\"" in title or "first person" in title:
            if avoid_first_person or first_person_rate < first_person_keep_rate:
                drop_reason = "Author voice avoids first person."
        if "contraction" in title or "contraction" in words:
            if "avoid" in title and contractions_rate >= contractions_avoid_threshold:
                drop_reason = "Author uses contractions frequently."
            if "use" in title and contractions_rate < contractions_use_threshold:
                drop_reason = "Author rarely uses contractions."
        if tokens and avoid_words:
            if any(t in avoid_words for t in tokens):
                drop_reason = "Rule conflicts with global avoid words."
        if tokens and preferred_set:
            if any(t in preferred_set for t in tokens):
                drop_reason = "Rule conflicts with preferred lexicon/phrases in fingerprint."
        if formal_style and any(k in title for k in ("add voice", "have opinions", "humor", "edge", "mess", "feelings")):
            drop_reason = "Formal/academic voice discourages subjective embellishment."
        if input_style:
            title_case_rate = float(input_style.get("heading_title_case_rate", 0.0))
            bold_rate = float(input_style.get("boldface_per_1000w", 0.0))
            inline_header_rate = float(input_style.get("inline_header_list_rate", 0.0))
            if "title case" in title and title_case_rate >= heading_title_case_keep_rate:
                drop_reason = "Input headings use Title Case."
            if "boldface" in title and bold_rate >= boldface_keep_per_1000w:
                drop_reason = "Input uses boldface frequently."
            if "inline-header" in title or "inline header" in title:
                if inline_header_rate >= inline_header_list_keep_rate:
                    drop_reason = "Input uses inline-header lists frequently."

        if drop_reason:
            dropped.append({**rule, "drop_reason": drop_reason})
        else:
            kept.append(rule)

    return kept, dropped


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token.
    return max(1, (len(text) + 3) // 4)


def estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    # Add a small per-message overhead to approximate chat tokenization.
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # per-message overhead
    return total + 2


def split_markdown_blocks(markdown: str) -> List[str]:
    # Split Markdown into text and code blocks while preserving fences.
    blocks: List[str] = []
    buf: List[str] = []
    in_code = False
    for line in markdown.splitlines():
        fence = line.strip()
        if fence.startswith("```") or fence.startswith("~~~"):
            if in_code:
                buf.append(line)
                blocks.append("\n".join(buf).strip("\n"))
                buf = []
                in_code = False
            else:
                if buf:
                    blocks.append("\n".join(buf).strip("\n"))
                    buf = []
                in_code = True
                buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        if not line.strip():
            if buf:
                blocks.append("\n".join(buf).strip("\n"))
                buf = []
            continue

        buf.append(line)

    if buf:
        blocks.append("\n".join(buf).strip("\n"))
    return blocks


def is_code_block(block: str) -> bool:
    # Identify fenced code blocks so we can chunk them safely.
    lines = block.splitlines()
    if not lines:
        return False
    first = lines[0].strip()
    last = lines[-1].strip()
    if (first.startswith("```") or first.startswith("~~~")) and (last.startswith("```") or last.startswith("~~~")):
        return True
    return False


def mask_inline_code(text: str) -> tuple[str, Dict[str, str]]:
    # Replace inline code spans with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"[[INLINE_CODE_{counter}]]"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    # Handle single and double backticks without crossing lines.
    pattern = re.compile(r"(``[^`\n]+``|`[^`\n]+`)")
    stripped = pattern.sub(repl, text)
    return stripped, mapping


def find_inline_code_placeholders(text: str) -> List[str]:
    return re.findall(r"\[\[INLINE_CODE_\d+\]\]", text)


def mask_html(text: str) -> tuple[str, Dict[str, str]]:
    # Replace HTML blocks and tags with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    def make_placeholder(block: str) -> str:
        nonlocal counter
        placeholder = f"[[HTML_BLOCK_{counter}]]"
        mapping[placeholder] = block
        counter += 1
        return placeholder

    def repl_block(match: re.Match[str]) -> str:
        return make_placeholder(match.group(0))

    # Mask block elements first.
    text = re.sub(r"(?is)<(script|style|table|pre|code|svg|math|div|section|article|header|footer|nav|aside)[^>]*>.*?</\\1>", repl_block, text)
    # Mask HTML comments.
    text = re.sub(r"(?is)<!--.*?-->", repl_block, text)
    # Mask any remaining tags (avoid matching inequalities like <10).
    text = re.sub(r"(?is)<[A-Za-z/][^>]*>", repl_block, text)
    return text, mapping


def mask_math_notation(text: str) -> tuple[str, Dict[str, str]]:
    # Replace LaTeX-style math with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter_inline = 0
    counter_display = 0

    def repl_display(match: re.Match[str]) -> str:
        nonlocal counter_display
        placeholder = f"[[DISPLAY_MATH_{counter_display}]]"
        mapping[placeholder] = match.group(0)
        counter_display += 1
        return placeholder

    def repl_inline(match: re.Match[str]) -> str:
        nonlocal counter_inline
        placeholder = f"[[INLINE_MATH_{counter_inline}]]"
        mapping[placeholder] = match.group(0)
        counter_inline += 1
        return placeholder

    text = re.sub(r"(?s)\\begin\\{[^}]+\\}.*?\\end\\{[^}]+\\}", repl_display, text)
    text = re.sub(r"(?s)\\$\\$.*?\\$\\$", repl_display, text)
    text = re.sub(r"(?s)\\\[.*?\\\]", repl_display, text)
    text = re.sub(r"(?s)\\\(.*?\\\)", repl_inline, text)
    text = re.sub(r"(?s)\\$(?:\\\\\\$|[^$])+\\$", repl_inline, text)
    return text, mapping


def mask_html_entities(text: str) -> tuple[str, Dict[str, str]]:
    # Replace HTML entities with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"[[HTML_ENTITY_{counter}]]"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    stripped = re.sub(r"&[A-Za-z0-9#]+;", repl, text)
    return stripped, mapping


def split_oversize_block(block: str, build_messages_fn, max_prompt_tokens: int) -> List[str]:
    # Recursively split blocks that exceed token limits, respecting code fences.
    if estimate_tokens_for_messages(build_messages_fn(block)) <= max_prompt_tokens:
        return [block]

    if is_code_block(block):
        # Preserve fences while splitting code content line-by-line.
        lines = block.splitlines()
        if len(lines) <= 2:
            return [block]
        opener = lines[0]
        fence = opener.strip()[:3]
        content = lines[1:-1] if lines[-1].strip().startswith(fence) else lines[1:]
        chunks: List[str] = []
        current: List[str] = []
        for line in content:
            current.append(line)
            candidate = "\n".join([opener] + current + [fence])
            if estimate_tokens_for_messages(build_messages_fn(candidate)) > max_prompt_tokens and len(current) > 1:
                current.pop()
                chunks.append("\n".join([opener] + current + [fence]))
                current = [line]
        if current:
            chunks.append("\n".join([opener] + current + [fence]))
        return chunks

    mid = len(block) // 2
    # Prefer splitting on a newline; fall back to a hard split if needed.
    split_idx = block.rfind("\n", 0, mid)
    if split_idx == -1:
        split_idx = block.find("\n", mid)
    if split_idx == -1:
        split_idx = mid
    left = block[:split_idx].strip()
    right = block[split_idx:].strip()
    chunks: List[str] = []
    if left:
        chunks.extend(split_oversize_block(left, build_messages_fn, max_prompt_tokens))
    if right:
        chunks.extend(split_oversize_block(right, build_messages_fn, max_prompt_tokens))
    return chunks


def chunk_markdown(markdown: str, build_messages_fn, max_prompt_tokens: int) -> List[str]:
    # Greedily group blocks into chunks that fit the prompt budget.
    blocks = split_markdown_blocks(markdown)
    chunks: List[str] = []
    current: List[str] = []

    def join_blocks(parts: List[str]) -> str:
        return "\n\n".join(p for p in parts if p.strip())

    for block in blocks:
        if not block.strip():
            continue
        for sub_block in split_oversize_block(block, build_messages_fn, max_prompt_tokens):
            if not current:
                current = [sub_block]
                continue
            candidate = join_blocks(current + [sub_block])
            if estimate_tokens_for_messages(build_messages_fn(candidate)) <= max_prompt_tokens:
                current.append(sub_block)
            else:
                chunks.append(join_blocks(current))
                current = [sub_block]

    if current:
        chunks.append(join_blocks(current))
    return [c for c in chunks if c.strip()]

def approx_rate_per_1000_words(count: int, total_words: int) -> float:
    # Normalize counts for cross-document comparability.
    if total_words <= 0:
        return 0.0
    return (count / total_words) * 1000.0

def detect_english_spelling_variant(text: str) -> Dict[str, Any]:
    # Heuristic: count common US vs Canadian/British spellings.
    pairs = [
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
    ]
    counts_us = 0
    counts_ca = 0
    examples_us: List[str] = []
    examples_ca: List[str] = []
    lowered = text.lower()
    for us, ca in pairs:
        us_hits = len(re.findall(rf"\b{re.escape(us)}\b", lowered))
        ca_hits = len(re.findall(rf"\b{re.escape(ca)}\b", lowered))
        if us_hits:
            counts_us += us_hits
            if us not in examples_us:
                examples_us.append(us)
        if ca_hits:
            counts_ca += ca_hits
            if ca not in examples_ca:
                examples_ca.append(ca)

    total = counts_us + counts_ca
    if total < 3:
        return {
            "language": "en",
            "variant": "unknown",
            "confidence": "low",
            "us_hits": counts_us,
            "canadian_hits": counts_ca,
            "examples": {"us": examples_us[:5], "canadian": examples_ca[:5]},
            "note": "Insufficient evidence to determine spelling variant."
        }

    if counts_us > counts_ca:
        variant = "us"
    elif counts_ca > counts_us:
        variant = "canadian"
    else:
        variant = "unknown"

    confidence = "medium"
    if total >= 8 and abs(counts_us - counts_ca) >= 4:
        confidence = "high"

    return {
        "language": "en",
        "variant": variant,
        "confidence": confidence,
        "us_hits": counts_us,
        "canadian_hits": counts_ca,
        "examples": {"us": examples_us[:5], "canadian": examples_ca[:5]},
        "note": "Heuristic based on common US vs Canadian spellings."
    }

def compute_measurements(text: str) -> Dict[str, Any]:
    # Gather lightweight, explainable measurements for the input chunk.
    w = words(text)
    total_words = len(w)

    # Sentence length distribution (words per sentence).
    sent_lens = [len(words(s)) for s in split_sentences(text)]
    sent_bins = [(0,9),(10,17),(18,25),(26,40),(41,None)]
    sent_hist = histogram(sent_lens, sent_bins)

    # Paragraph length distribution (sentences per paragraph).
    paras = split_paragraphs(text)
    para_lens = [len(split_sentences(p)) for p in paras]
    para_bins = [(1,1),(2,3),(4,5),(6,8),(9,None)]
    para_hist = histogram(para_lens, para_bins)
    one_sentence_rate = sum(1 for n in para_lens if n == 1) / max(1, len(para_lens))

    # Simple punctuation counts.
    punct = {
        "commas": text.count(","),
        "semicolons": text.count(";"),
        "colons": text.count(":"),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "parentheses_open": text.count("("),
        "parentheses_close": text.count(")"),
        "em_dashes": text.count("—"),
        "en_dashes": text.count("–"),
        "ellipses_unicode": text.count("…"),
        "ellipses_three_dots": text.count("...")
    }

    # Contractions rate (rough)
    contraction_hits = len(re.findall(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b", text))
    contraction_rate = contraction_hits / max(1, total_words)

    # Oxford comma heuristic (rough): ", and"/", or" vs " and"/" or"
    and_total = text.lower().count(" and ")
    comma_and = text.lower().count(", and ")
    oxford_signal = comma_and / max(1, and_total)

    toks = [t.lower() for t in w]
    function_words = [
        "the","a","an","and","or","but","if","then","because","so","while","although","though",
        "of","in","on","for","with","as","by","from","to","into","over","under","between",
        "is","are","was","were","be","been","being","it","this","that","these","those",
        "i","we","you","he","she","they","me","us","him","her","them","my","our","your","their",
        "not","no","nor","very","also","even","only","just","rather","however","therefore"
    ]
    fw_counter = collections.Counter(toks)
    fw_counts = {fw: fw_counter.get(fw, 0) for fw in function_words}
    fw_rates = {fw: approx_rate_per_1000_words(cnt, total_words) for fw, cnt in fw_counts.items()}
    fw_top = sorted(
        [{"word": fw, "count": fw_counts.get(fw, 0)} for fw in function_words],
        key=lambda x: x["count"],
        reverse=True
    )[:20]

    hedge_terms = {"may","might","perhaps","likely","possibly","seems","appears","suggests","tends"}
    booster_terms = {"clearly","obviously","certainly","undoubtedly","indeed","surely"}
    directive_terms = {"must","should","need","needs","ought","required"}
    first_person = {"i","me","my","mine","we","us","our","ours"}
    second_person = {"you","your","yours"}
    third_person = {"he","him","his","she","her","hers","they","them","their","theirs"}

    hedge_hits = sum(1 for t in toks if t in hedge_terms)
    booster_hits = sum(1 for t in toks if t in booster_terms)
    directive_hits = sum(1 for t in toks if t in directive_terms)
    first_hits = sum(1 for t in toks if t in first_person)
    second_hits = sum(1 for t in toks if t in second_person)
    third_hits = sum(1 for t in toks if t in third_person)

    sent_openers = collections.Counter()
    transition_terms = {
        "however","therefore","moreover","furthermore","nevertheless","nonetheless",
        "for example","for instance","in short","in sum","in practice","in effect",
        "first","second","third","finally","overall"
    }
    transition_hits = collections.Counter()
    for s in split_sentences(text):
        ws = [t.lower() for t in words(s)]
        if len(ws) >= 2:
            opener = " ".join(ws[:3]) if len(ws) >= 3 else " ".join(ws[:2])
            if sum(1 for x in opener.split() if x in {"the","a","an","and","or","but","if","then","to","of","in","on","for","with","as"}) <= 1:
                sent_openers[opener] += 1
        if ws:
            start_two = " ".join(ws[:2])
            start_three = " ".join(ws[:3]) if len(ws) >= 3 else ""
            for cand in (start_three, start_two, ws[0]):
                if cand and cand in transition_terms:
                    transition_hits[cand] += 1
                    break

    return {
        "totals": {
            "total_words_est": total_words,
            "total_sentences_est": len(sent_lens),
            "total_paragraphs_est": len(paras)
        },
        "sentence": {
            "length_words": {
                "histogram_bins": ["<10", "10-17", "18-25", "26-40", ">40"],
                "histogram_p": sent_hist
            }
        },
        "paragraph": {
            "length_sentences_histogram_bins": ["1", "2-3", "4-5", "6-8", ">8"],
            "length_sentences_histogram_p": para_hist,
            "one_sentence_paragraph_rate": one_sentence_rate
        },
        "punctuation": {
            "counts": punct,
            "rates_per_1000w": {k: approx_rate_per_1000_words(v, total_words) for k, v in punct.items()}
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
            "hedge_rate": approx_rate_per_1000_words(hedge_hits, total_words),
            "booster_rate": approx_rate_per_1000_words(booster_hits, total_words),
            "directive_rate": approx_rate_per_1000_words(directive_hits, total_words),
            "first_person_rate": approx_rate_per_1000_words(first_hits, total_words),
            "second_person_rate": approx_rate_per_1000_words(second_hits, total_words),
            "third_person_rate": approx_rate_per_1000_words(third_hits, total_words)
        },
        "templates_signals": {
            "sentence_openers_top": [{"phrase": p, "count": c} for p, c in sent_openers.most_common(20)],
            "transition_openers_top": [{"phrase": p, "count": c} for p, c in transition_hits.most_common(15)]
        }
    }


def l1_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b))


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(b), 1.0)
    return abs(a - b) / denom


def compute_style_compliance(
    fingerprint: Dict[str, Any],
    output_text: str
) -> Dict[str, Any]:
    # Compare output measurements to fingerprint measurements and return score + deltas.
    fp_meas = fingerprint.get("measurements", {}) if isinstance(fingerprint, dict) else {}
    out_meas = compute_measurements(output_text)
    deltas: List[Dict[str, Any]] = []

    score_parts: List[float] = []

    # Sentence length histogram
    fp_sent = fp_meas.get("sentence", {}).get("length_words", {}).get("histogram_p")
    out_sent = out_meas.get("sentence", {}).get("length_words", {}).get("histogram_p")
    if isinstance(fp_sent, list) and isinstance(out_sent, list) and len(fp_sent) == len(out_sent):
        diff = l1_distance(fp_sent, out_sent) / 2.0
        score_parts.append(max(0.0, 1.0 - min(1.0, diff)))
        if diff > 0.15:
            deltas.append({"metric": "sentence_length_histogram", "diff": diff})

    # Paragraph length histogram
    fp_para = fp_meas.get("paragraph", {}).get("length_sentences_histogram_p")
    out_para = out_meas.get("paragraph", {}).get("length_sentences_histogram_p")
    if isinstance(fp_para, list) and isinstance(out_para, list) and len(fp_para) == len(out_para):
        diff = l1_distance(fp_para, out_para) / 2.0
        score_parts.append(max(0.0, 1.0 - min(1.0, diff)))
        if diff > 0.15:
            deltas.append({"metric": "paragraph_length_histogram", "diff": diff})

    # One-sentence paragraph rate
    fp_one = fp_meas.get("paragraph", {}).get("one_sentence_paragraph_rate")
    out_one = out_meas.get("paragraph", {}).get("one_sentence_paragraph_rate")
    if isinstance(fp_one, (int, float)) and isinstance(out_one, (int, float)):
        diff = abs(fp_one - out_one)
        score_parts.append(max(0.0, 1.0 - min(1.0, diff)))
        if diff > 0.1:
            deltas.append({"metric": "one_sentence_paragraph_rate", "diff": diff})

    # Punctuation rates
    fp_punct = fp_meas.get("punctuation", {}).get("rates_per_1000w", {})
    out_punct = out_meas.get("punctuation", {}).get("rates_per_1000w", {})
    if isinstance(fp_punct, dict) and isinstance(out_punct, dict) and fp_punct:
        diffs = []
        for k, target in fp_punct.items():
            if k in out_punct and isinstance(target, (int, float)):
                diff = relative_diff(out_punct.get(k, 0.0), float(target))
                diffs.append(diff)
                if diff > 0.5:
                    deltas.append({"metric": f"punctuation.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            score_parts.append(max(0.0, 1.0 - min(1.0, avg)))

    # Contractions / Oxford comma signals
    fp_ortho = fp_meas.get("orthography_signals", {})
    out_ortho = out_meas.get("orthography_signals", {})
    for key in ("contractions_rate", "oxford_comma_signal"):
        if isinstance(fp_ortho.get(key), (int, float)) and isinstance(out_ortho.get(key), (int, float)):
            diff = relative_diff(float(out_ortho[key]), float(fp_ortho[key]))
            score_parts.append(max(0.0, 1.0 - min(1.0, diff)))
            if diff > 0.5:
                deltas.append({"metric": f"orthography.{key}", "diff": diff})

    # Stance signals (optional)
    fp_stance = fp_meas.get("stance_signals", {})
    out_stance = out_meas.get("stance_signals", {})
    if isinstance(fp_stance, dict) and isinstance(out_stance, dict) and fp_stance:
        diffs = []
        for k, target in fp_stance.items():
            if k in out_stance and isinstance(target, (int, float)):
                diff = relative_diff(float(out_stance.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"stance.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            score_parts.append(max(0.0, 1.0 - min(1.0, avg)))

    score = sum(score_parts) / len(score_parts) if score_parts else 1.0
    return {
        "score": score,
        "deltas": deltas,
        "output_measurements": out_meas
    }


# ---- OpenAI-compatible client ----

class LLMConfig:
    # Minimal OpenAI-compatible configuration container.
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: int,
        extra_headers: Dict[str, str],
        max_prompt_tokens: int,
        max_retries: int,
        backoff_base_seconds: float,
        backoff_max_seconds: float
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.extra_headers = extra_headers
        self.max_prompt_tokens = max_prompt_tokens
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds

def load_config(path: Path) -> LLMConfig:
    # Load the API config JSON and apply defaults.
    data = json.loads(path.read_text(encoding="utf-8"))
    max_tokens = int(data.get("max_tokens", 6000))
    return LLMConfig(
        api_key=data["api_key"],
        base_url=data["base_url"],
        model=data["model"],
        max_tokens=max_tokens,
        temperature=float(data.get("temperature", 0.2)),
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_headers=dict(data.get("extra_headers", {})),
        max_prompt_tokens=int(data.get("max_prompt_tokens", max_tokens)),
        max_retries=int(data.get("max_retries", 2)),
        backoff_base_seconds=float(data.get("backoff_base_seconds", 1.0)),
        backoff_max_seconds=float(data.get("backoff_max_seconds", 8.0))
    )

def chat_completions(cfg: LLMConfig, messages: List[Dict[str, str]]) -> str:
    # POST to a /v1/chat/completions-compatible endpoint.
    url = f"{cfg.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        **cfg.extra_headers
    }
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature
    }
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_seconds)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:2000]}")
            if r.status_code >= 400:
                raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:2000]}")
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, RuntimeError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
            jitter = random.uniform(0, backoff * 0.2)
            sleep_s = backoff + jitter
            print_warn(f"LLM request failed (attempt {attempt + 1}/{cfg.max_retries + 1}); retrying in {sleep_s:.1f}s.")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {cfg.max_retries + 1} attempts: {last_err}")

def parse_json_strict(s: str) -> Dict[str, Any]:
    # Strip code fences if present and parse strictly as JSON.
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)

def repair_json_with_llm(cfg: LLMConfig, bad_output: str, prompts: Dict[str, Any]) -> Dict[str, Any]:
    # Ask the LLM to repair malformed JSON while preserving content.
    system = get_prompt_value(prompts, "repair_json", "system")
    task = get_prompt_value(prompts, "repair_json", "task")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({
            "task": task,
            "bad_output": bad_output[:20000]
        })}
    ]
    fixed = chat_completions(cfg, messages)
    return parse_json_strict(fixed)


# ---- Prompting: apply fingerprint ----

def build_apply_prompt(
    fingerprint: Dict[str, Any],
    input_md: str,
    input_meas: Dict[str, Any],
    cfg: LLMConfig,
    prompts: Dict[str, Any],
    style_feedback: Dict[str, Any] | None = None,
    humanizer_rules: List[Dict[str, Any]] | None = None
) -> List[Dict[str, str]]:
    # Fill the apply prompt template with runtime data.
    system = get_prompt_value(prompts, "apply", "system")
    user_template = get_prompt_value(prompts, "apply", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.apply.user must be an object")
    user = copy.deepcopy(user_template)
    user["style_fingerprint_json"] = fingerprint
    user["input_measurements"] = input_meas
    user["input_markdown"] = input_md
    if style_feedback:
        user["style_feedback"] = style_feedback
    if humanizer_rules:
        user["humanizer_guidelines"] = humanizer_rules

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.llm.json (default: ./config.llm.json if present; else next to script)"
    )
    ap.add_argument(
        "-f",
        "--fingerprint",
        required=True,
        type=Path,
        help="Fingerprint JSON from fingerprint_style.py (adds .json if no extension)"
    )
    ap.add_argument("-i", "--in", dest="inp", required=True, type=Path, help="Input markdown file to rewrite")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output markdown path (default: <input>.styled.md)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable progress logging")
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Maximum prompt tokens before chunking (default: config max_prompt_tokens)"
    )
    ap.add_argument(
        "--no-humanizer-guidelines",
        action="store_true",
        help="Disable applying general-guidelines.md humanizer rules"
    )
    ap.add_argument(
        "--no-humanizer-llm-parse",
        action="store_true",
        help="Disable LLM-based parsing of humanizer guidelines (fallback to regex parser)"
    )
    ap.add_argument(
        "--tunables",
        type=Path,
        default=None,
        help="Path to config.tunables.json (default: ./config.tunables.json if present; else next to script)"
    )
    ap.add_argument(
        "--no-style-retry",
        action="store_true",
        help="Disable the style compliance retry pass"
    )
    ap.add_argument(
        "--style-retry-threshold",
        type=float,
        default=0.75,
        help="Retry threshold for style compliance score (default: 0.75)"
    )
    ap.add_argument(
        "--max-style-retries",
        type=int,
        default=1,
        help="Maximum number of style retry passes (default: 1)"
    )
    args = ap.parse_args()

    if args.fingerprint.suffix == "":
        args.fingerprint = args.fingerprint.with_suffix(".json")

    if args.config is None:
        # Resolve config: prefer current working directory, then script directory.
        cwd_cfg = Path.cwd() / "config.llm.json"
        script_cfg = Path(__file__).resolve().parent / "config.llm.json"
        args.config = cwd_cfg if cwd_cfg.exists() else script_cfg

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    vprint(f"Using config: {args.config}")

    cfg = load_config(args.config)
    prompts = load_prompts()
    if args.tunables is None:
        cwd_tunables = Path.cwd() / TUNABLES_FILENAME
        script_tunables = Path(__file__).resolve().parent / TUNABLES_FILENAME
        args.tunables = cwd_tunables if cwd_tunables.exists() else script_tunables
    tunables = load_tunables(args.tunables if args.tunables.exists() else None)
    if args.max_prompt_tokens is not None:
        # Allow CLI override for chunking threshold.
        cfg.max_prompt_tokens = args.max_prompt_tokens

    humanizer_rules: List[Dict[str, Any]] = []
    humanizer_debug: Dict[str, Any] | None = None
    input_style_signals: Dict[str, Any] | None = None

    if not args.fingerprint.exists():
        # Fall back to the script directory if fingerprint isn't in CWD.
        script_fp = Path(__file__).resolve().parent / args.fingerprint.name
        if script_fp.exists():
            args.fingerprint = script_fp
        else:
            print_error(f"Fingerprint not found: {args.fingerprint}")
            return 2
    if not args.inp.exists():
        print_error(f"Input markdown not found: {args.inp}")
        return 2

    vprint("Loading fingerprint and input...")
    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    avoid_list = load_avoid_list()
    if avoid_list:
        fingerprint = merge_avoid_list_into_fingerprint(fingerprint, avoid_list)
    forbid_em_dashes = should_forbid_em_dashes(fingerprint, avoid_list)
    input_md = args.inp.read_text(encoding="utf-8")
    original_input_md = input_md
    # Strip base64 images to keep prompts within token limits.
    input_md, base64_map = strip_base64_images(input_md)
    if base64_map:
        vprint(f"Stripped {len(base64_map)} base64 image embed(s) from prompt.")
    # Mask HTML, math, entities, and inline code spans so they are preserved verbatim.
    input_md, html_map = mask_html(input_md)
    input_md, math_map = mask_math_notation(input_md)
    input_md, entity_map = mask_html_entities(input_md)
    input_md, inline_code_map = mask_inline_code(input_md)
    if not args.no_humanizer_guidelines:
        raw_guidelines, guidelines_path = load_general_guidelines()
        if raw_guidelines:
            if forbid_em_dashes:
                print("Hard constraint active: em dashes are forbidden.")
            parsed_rules: List[Dict[str, Any]] = []
            parser_used = "regex"
            cache_path = Path(__file__).resolve().parent / HUMANIZER_CACHE_FILENAME
            cache = load_humanizer_rules_cache(cache_path)
            cache_used = False
            if cache and guidelines_path and cache.get("source_mtime") is not None:
                try:
                    cache_mtime = float(cache.get("source_mtime"))
                    src_mtime = guidelines_path.stat().st_mtime
                    if cache_mtime >= src_mtime and isinstance(cache.get("rules"), list):
                        parsed_rules = normalize_humanizer_rules(cache.get("rules", []), "cache")
                        parser_used = str(cache.get("parser", "cache"))
                        cache_used = True
                        vprint(f"Loaded cached humanizer rules from {cache_path.name}.")
                except Exception:
                    cache_used = False

            if not cache_used:
                if not args.no_humanizer_llm_parse:
                    try:
                        print("Parsing humanizer guidelines via LLM...")
                        parsed_rules = parse_humanizer_guidelines_llm(cfg, prompts, raw_guidelines)
                        if parsed_rules:
                            parser_used = "llm"
                    except Exception:
                        parsed_rules = []
                if not parsed_rules:
                    vprint("LLM parsing returned no rules; falling back to regex parser.")
                    parsed_rules = normalize_humanizer_rules(parse_humanizer_guidelines(raw_guidelines), "regex")
                    parser_used = "regex"
                try:
                    write_humanizer_rules_cache(cache_path, parsed_rules, parser_used, guidelines_path)
                    vprint(f"Wrote humanizer rules cache to {cache_path.name}.")
                except Exception:
                    pass
            input_style = analyze_markdown_style(input_md)
            input_style_signals = input_style
            humanizer_rules, dropped_rules = filter_humanizer_rules(parsed_rules, fingerprint, input_style, tunables)
            humanizer_debug = {
                "rule_or_field": "humanizer_guidelines",
                "parser": parser_used,
                "kept": humanizer_rules,
                "dropped": dropped_rules
            }
            if dropped_rules:
                drop_labels: List[str] = []
                for rule in dropped_rules:
                    if not isinstance(rule, dict):
                        continue
                    title = rule.get("title")
                    if not title:
                        continue
                    reason = rule.get("drop_reason")
                    if isinstance(reason, str) and reason.strip():
                        drop_labels.append(f"{title} — {reason}")
                    else:
                        drop_labels.append(str(title))
                preview = ", ".join(drop_labels[:10])
                suffix = "..." if len(drop_labels) > 10 else ""
                print(f"Dropped {len(drop_labels)} humanizer rule(s): {preview}{suffix}")
            if args.verbose:
                print(f"Humanizer rules loaded: {len(humanizer_rules)} kept, {len(dropped_rules)} dropped")
    # Mask non-voice blocks and inline citations so they are preserved verbatim.
    input_md, frozen_blocks = mask_non_voice_blocks(input_md)
    input_md, citation_map = mask_inline_citations(input_md)
    section_blocks = extract_heading_blocks(input_md)
    section_blocks_restored: List[Dict[str, Any]] = []
    for block in section_blocks:
        restored = block["block"]
        restored = restore_placeholders(restored, html_map)
        restored = restore_placeholders(restored, math_map)
        restored = restore_placeholders(restored, entity_map)
        restored = restore_placeholders(restored, inline_code_map)
        restored = restore_placeholders(restored, frozen_blocks)
        restored = restore_placeholders(restored, citation_map)
        restored = restore_base64_images(restored, base64_map, find_base64_placeholders(restored))
        section_blocks_restored.append({
            **block,
            "block": restored,
            "signature": section_signature(restored)
        })

    all_deviations: List[Any] = []
    outputs: List[str] = []

    def build_messages_for_chunk(md_chunk: str, style_feedback: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
        # Build prompts per chunk using local measurements.
        input_meas = compute_measurements(filter_author_voice_text(md_chunk))
        return build_apply_prompt(
            fingerprint,
            md_chunk,
            input_meas,
            cfg,
            prompts,
            style_feedback,
            humanizer_rules
        )

    def rewrite_chunk(md_chunk: str, chunk_index: int | None = None, chunk_total: int | None = None) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        # Rewrite a chunk with optional style retry.
        attempts = 0
        style_feedback: Dict[str, Any] | None = None
        last_out: Dict[str, Any] = {}
        while True:
            messages = build_messages_for_chunk(md_chunk, style_feedback)
            raw = chat_completions(cfg, messages)
            try:
                out_obj = parse_json_strict(raw)
            except Exception:
                vprint("Invalid JSON returned; attempting repair...")
                out_obj = repair_json_with_llm(cfg, raw, prompts)

            final_md = out_obj.get("final_markdown")
            if not isinstance(final_md, str) or not final_md.strip():
                print_error("LLM did not return final_markdown.")
                print(raw)
                raise RuntimeError("LLM did not return final_markdown")

            if forbid_em_dashes:
                final_md, removed = enforce_no_em_dashes(final_md)
                if removed:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "punctuation.em_dashes",
                        "reason": "Em dashes removed to satisfy hard constraint.",
                        "count": removed
                    })

            compliance = compute_style_compliance(fingerprint, filter_author_voice_text(final_md))
            if not args.no_style_retry and attempts < args.max_style_retries and compliance["score"] < args.style_retry_threshold:
                style_feedback = {
                    "score": compliance["score"],
                    "deltas": compliance.get("deltas", [])
                }
                attempts += 1
                continue

            last_out = out_obj
            return final_md, out_obj, compliance

    initial_messages = build_messages_for_chunk(input_md)
    initial_tokens = estimate_tokens_for_messages(initial_messages)
    if initial_tokens <= cfg.max_prompt_tokens:
        vprint("Calling LLM to apply fingerprint...")
        try:
            final_md, out_obj, compliance = rewrite_chunk(input_md)
        except RuntimeError:
            return 3
        # Ensure any frozen blocks and citation placeholders survive.
        missing_html = [p for p in find_placeholders(input_md, HTML_PLACEHOLDER_RE) if p not in final_md]
        if missing_html:
            for p in missing_html:
                all_deviations.append({
                    "rule_or_field": "html",
                    "reason": "HTML placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_html)

        missing_math = [p for p in find_placeholders(input_md, INLINE_MATH_PLACEHOLDER_RE) if p not in final_md]
        missing_math += [p for p in find_placeholders(input_md, DISPLAY_MATH_PLACEHOLDER_RE) if p not in final_md]
        if missing_math:
            for p in missing_math:
                all_deviations.append({
                    "rule_or_field": "math",
                    "reason": "Math placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_math)

        missing_entities = [p for p in find_placeholders(input_md, HTML_ENTITY_PLACEHOLDER_RE) if p not in final_md]
        if missing_entities:
            for p in missing_entities:
                all_deviations.append({
                    "rule_or_field": "html_entity",
                    "reason": "HTML entity placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_entities)

        missing_inline = [p for p in find_placeholders(input_md, INLINE_CODE_PLACEHOLDER_RE) if p not in final_md]
        if missing_inline:
            for p in missing_inline:
                all_deviations.append({
                    "rule_or_field": "inline_code",
                    "reason": "Inline code placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_inline)

        missing_frozen = [p for p in find_placeholders(input_md, FROZEN_BLOCK_RE) if p not in final_md]
        if missing_frozen:
            for p in missing_frozen:
                all_deviations.append({
                    "rule_or_field": "frozen_block",
                    "reason": "Non-voice block placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_frozen)

        missing_cites = [p for p in find_placeholders(input_md, CITATION_PLACEHOLDER_RE) if p not in final_md]
        if missing_cites:
            for p in missing_cites:
                all_deviations.append({
                    "rule_or_field": "citation",
                    "reason": "Citation placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_cites)

        # Ensure any base64 placeholders survive and reinsert originals.
        missing = [p for p in find_base64_placeholders(input_md) if p not in final_md]
        if missing:
            for p in missing:
                all_deviations.append({
                    "rule_or_field": "base64_image",
                    "reason": "Image placeholder missing from output; re-embedded at end of document.",
                    "placeholder": p
                })
            footer = "\n".join(f"![]({base64_map[p]})" for p in missing if p in base64_map)
            final_md = final_md.rstrip() + "\n\n" + footer
        final_md = restore_placeholders(final_md, html_map)
        final_md = restore_placeholders(final_md, math_map)
        final_md = restore_placeholders(final_md, entity_map)
        final_md = restore_placeholders(final_md, inline_code_map)
        final_md = restore_placeholders(final_md, frozen_blocks)
        final_md = restore_placeholders(final_md, citation_map)
        final_md = restore_base64_images(final_md, base64_map, find_base64_placeholders(final_md))
        outputs.append(final_md)
        all_deviations.extend(out_obj.get("deviations", []) or [])
        all_deviations.append({
            "rule_or_field": "style_compliance",
            "score": compliance.get("score"),
            "deltas": compliance.get("deltas", [])
        })
    else:
        vprint(f"Prompt too large ({initial_tokens} tokens); chunking input...")
        chunks = chunk_markdown(input_md, build_messages_for_chunk, cfg.max_prompt_tokens)
        vprint(f"Chunked into {len(chunks)} parts.")
        for idx, chunk in enumerate(chunks, start=1):
            vprint(f"Rewriting chunk {idx}/{len(chunks)}...")
            try:
                final_md, out_obj, compliance = rewrite_chunk(chunk, idx, len(chunks))
            except RuntimeError:
                return 3
            # Ensure any frozen blocks and citation placeholders survive.
            missing_html = [p for p in find_placeholders(chunk, HTML_PLACEHOLDER_RE) if p not in final_md]
            if missing_html:
                for p in missing_html:
                    all_deviations.append({
                        "rule_or_field": "html",
                        "reason": "HTML placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_html)

            missing_math = [p for p in find_placeholders(chunk, INLINE_MATH_PLACEHOLDER_RE) if p not in final_md]
            missing_math += [p for p in find_placeholders(chunk, DISPLAY_MATH_PLACEHOLDER_RE) if p not in final_md]
            if missing_math:
                for p in missing_math:
                    all_deviations.append({
                        "rule_or_field": "math",
                        "reason": "Math placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_math)

            missing_entities = [p for p in find_placeholders(chunk, HTML_ENTITY_PLACEHOLDER_RE) if p not in final_md]
            if missing_entities:
                for p in missing_entities:
                    all_deviations.append({
                        "rule_or_field": "html_entity",
                        "reason": "HTML entity placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_entities)

            missing_inline = [p for p in find_placeholders(chunk, INLINE_CODE_PLACEHOLDER_RE) if p not in final_md]
            if missing_inline:
                for p in missing_inline:
                    all_deviations.append({
                        "rule_or_field": "inline_code",
                        "reason": "Inline code placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_inline)

            missing_frozen = [p for p in find_placeholders(chunk, FROZEN_BLOCK_RE) if p not in final_md]
            if missing_frozen:
                for p in missing_frozen:
                    all_deviations.append({
                        "rule_or_field": "frozen_block",
                        "reason": "Non-voice block placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_frozen)

            missing_cites = [p for p in find_placeholders(chunk, CITATION_PLACEHOLDER_RE) if p not in final_md]
            if missing_cites:
                for p in missing_cites:
                    all_deviations.append({
                        "rule_or_field": "citation",
                        "reason": "Citation placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                final_md = final_md.rstrip() + "\n\n" + "\n\n".join(missing_cites)
            # Reinsert any missing base64 placeholders at the end of the chunk.
            missing = [p for p in find_base64_placeholders(chunk) if p not in final_md]
            if missing:
                for p in missing:
                    all_deviations.append({
                        "rule_or_field": "base64_image",
                        "reason": "Image placeholder missing from output; re-embedded at end of chunk.",
                        "placeholder": p,
                        "chunk_index": idx,
                        "chunk_total": len(chunks)
                    })
                footer = "\n".join(f"![]({base64_map[p]})" for p in missing if p in base64_map)
                final_md = final_md.rstrip() + "\n\n" + footer
            final_md = restore_placeholders(final_md, html_map)
            final_md = restore_placeholders(final_md, math_map)
            final_md = restore_placeholders(final_md, entity_map)
            final_md = restore_placeholders(final_md, inline_code_map)
            final_md = restore_placeholders(final_md, frozen_blocks)
            final_md = restore_placeholders(final_md, citation_map)
            final_md = restore_base64_images(final_md, base64_map, find_base64_placeholders(final_md))
            outputs.append(final_md)
            deviations = out_obj.get("deviations", []) or []
            for d in deviations:
                if isinstance(d, dict):
                    d = dict(d)
                    d.setdefault("chunk_index", idx)
                    d.setdefault("chunk_total", len(chunks))
                    all_deviations.append(d)
                else:
                    all_deviations.append({"chunk_index": idx, "chunk_total": len(chunks), "detail": d})
            all_deviations.append({
                "rule_or_field": "style_compliance",
                "score": compliance.get("score"),
                "deltas": compliance.get("deltas", []),
                "chunk_index": idx,
                "chunk_total": len(chunks)
            })

    # Stitch chunks back together, preserving the original order.
    final_md = "\n\n".join(s.strip() for s in outputs if s.strip()).strip()
    # Ensure all input sections are present; restore missing sections verbatim.
    output_blocks = extract_heading_blocks(final_md)
    output_blocks_with_sig: List[Dict[str, Any]] = []
    for block in output_blocks:
        output_blocks_with_sig.append({
            **block,
            "signature": section_signature(block["block"])
        })
    output_keys = {b["key"] for b in output_blocks_with_sig}

    used_output_idx: set[int] = set()
    matched_start_by_input: List[int | None] = []
    similarity_threshold = 0.55
    min_overlap = 6

    for block in section_blocks_restored:
        if block["key"] in output_keys:
            out_idx = next(
                (i for i, b in enumerate(output_blocks_with_sig)
                 if b["key"] == block["key"] and b.get("level") == block.get("level")),
                None
            )
            if out_idx is not None:
                used_output_idx.add(out_idx)
                matched_start_by_input.append(output_blocks_with_sig[out_idx]["start_line"])
                continue
        # Fallback: content similarity matching
        best_idx = None
        best_score = 0.0
        for i, out_block in enumerate(output_blocks_with_sig):
            if i in used_output_idx:
                continue
            if out_block.get("level") != block.get("level"):
                continue
            score = jaccard_similarity(block.get("signature", set()), out_block.get("signature", set()))
            if score > best_score:
                best_score = score
                best_idx = i
        overlap = 0
        if best_idx is not None:
            overlap = len(block.get("signature", set()) & output_blocks_with_sig[best_idx].get("signature", set()))
        if best_idx is not None and best_score >= similarity_threshold and overlap >= min_overlap:
            used_output_idx.add(best_idx)
            matched_start_by_input.append(output_blocks_with_sig[best_idx]["start_line"])
        else:
            matched_start_by_input.append(None)

    missing_sections = [b for idx, b in enumerate(section_blocks_restored) if matched_start_by_input[idx] is None]
    if missing_sections:
        titles = ", ".join(b["title"] for b in missing_sections[:10])
        suffix = "..." if len(missing_sections) > 10 else ""
        print_warn(f"Restoring {len(missing_sections)} missing section(s) in original order: {titles}{suffix}")
        lines = final_md.splitlines()
        offset = 0
        for idx, block in enumerate(section_blocks_restored):
            if matched_start_by_input[idx] is not None:
                continue
            # Find next matched section to anchor insertion.
            insertion_line = None
            for j in range(idx + 1, len(section_blocks_restored)):
                if matched_start_by_input[j] is not None:
                    insertion_line = matched_start_by_input[j]
                    break
            if insertion_line is None:
                insertion_line = len(lines)
            insertion_line += offset
            block_lines = block["block"].strip().splitlines()
            if insertion_line < 0:
                insertion_line = 0
            lines[insertion_line:insertion_line] = [""] + block_lines + [""]
            offset += len(block_lines) + 2
            all_deviations.append({
                "rule_or_field": "missing_section",
                "heading": block["title"],
                "reason": "Section missing from LLM output; restored at original position."
            })
        final_md = "\n".join(lines).strip()
    line_count_in = len(original_input_md.splitlines())
    line_count_out = len(final_md.splitlines())
    line_change_pct = ((line_count_out - line_count_in) / max(1, line_count_in)) * 100.0
    print(f"Line count change: {line_count_in} -> {line_count_out} ({line_change_pct:+.1f}%).")

    word_count_in = len(words(original_input_md))
    word_count_out = len(words(final_md))
    word_change_pct = ((word_count_out - word_count_in) / max(1, word_count_in)) * 100.0
    print(f"Word count change: {word_count_in} -> {word_count_out} ({word_change_pct:+.1f}%).")

    para_count_in = len(split_paragraphs(original_input_md))
    para_count_out = len(split_paragraphs(final_md))
    para_change_pct = ((para_count_out - para_count_in) / max(1, para_count_in)) * 100.0
    print(f"Paragraph count change: {para_count_in} -> {para_count_out} ({para_change_pct:+.1f}%).")

    sanity = tunables.get("sanity_checks", {}) if isinstance(tunables, dict) else {}
    line_warn = float(sanity.get("line_count_warn_pct", 10.0))
    word_warn = float(sanity.get("word_count_warn_pct", 10.0))
    para_warn = float(sanity.get("paragraph_count_warn_pct", 10.0))

    if abs(line_change_pct) >= line_warn:
        print_warn(
            f"Warning: line count changed by {line_change_pct:+.1f}% (threshold {line_warn:.1f}%). "
            "Review output for potential missing or expanded content."
        )
    if abs(word_change_pct) >= word_warn:
        print_warn(
            f"Warning: word count changed by {word_change_pct:+.1f}% (threshold {word_warn:.1f}%). "
            "Review output for potential missing or expanded content."
        )
    if abs(para_change_pct) >= para_warn:
        print_warn(
            f"Warning: paragraph count changed by {para_change_pct:+.1f}% (threshold {para_warn:.1f}%). "
            "Review output for potential missing or expanded content."
        )

    all_deviations.append({
        "rule_or_field": "line_count_change",
        "input_lines": line_count_in,
        "output_lines": line_count_out,
        "percent_change": line_change_pct
    })
    all_deviations.append({
        "rule_or_field": "word_count_change",
        "input_words": word_count_in,
        "output_words": word_count_out,
        "percent_change": word_change_pct
    })
    all_deviations.append({
        "rule_or_field": "paragraph_count_change",
        "input_paragraphs": para_count_in,
        "output_paragraphs": para_count_out,
        "percent_change": para_change_pct
    })

    out_path = args.out or args.inp.with_suffix(args.inp.suffix + ".styled.md")
    vprint(f"Writing output: {out_path}")
    out_path.write_text(final_md, encoding="utf-8")
    print(f"Wrote rewritten markdown to: {out_path}")

    # Optionally also write deviations report
    if humanizer_debug:
        all_deviations.append(humanizer_debug)
    if input_style_signals:
        all_deviations.append({
            "rule_or_field": "input_style_signals",
            "signals": input_style_signals
        })
    if all_deviations:
        rep = out_path.with_suffix(out_path.suffix + ".deviations.json")
        rep.write_text(json.dumps(all_deviations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote deviations report to: {rep}")
    vprint("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
