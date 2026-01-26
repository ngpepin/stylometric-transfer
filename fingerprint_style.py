#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
"""
fingerprint_style.py

Build a style fingerprint JSON from an author's writing corpus packaged as:
- .zip
- .tar
- .tar.gz / .tgz
- .tar.bz2
- .tar.xz

It:
1) Extracts archive to a temp directory
2) Reads text-like files (txt, md, rst, html, docx)
3) Computes statistical measurements to guide the LLM
4) Calls an OpenAI-compatible LLM endpoint to produce a fingerprint JSON
5) Validates/repairs JSON if needed and writes the output file

Usage:
  python fingerprint_style.py -a corpus.zip -o fingerprint.json

Notes:
  If --profile-id or --author-name are omitted, both default to the output filename without the .json extension.
"""

from __future__ import annotations

import argparse
import copy
import collections
import dataclasses
import io
import json
import os
import re
import shutil
import statistics
import sys
import tarfile
import tempfile
import textwrap
import time
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from docx import Document  # python-docx
except Exception:
    Document = None  # optional

# Script overview:
# - Extract a corpus archive into a temp directory
# - Read text-like files and normalize content
# - Filter blockquotes, references, footnotes, and inline citations from style analysis
# - Normalize common OCR artifacts (ligatures, hyphenation)
# - Compute lightweight, interpretable stylometric measurements
# - Select representative excerpts
# - Build a prompt from prompts.json and call an OpenAI-compatible LLM
# - Optionally validate common phrases with a separate LLM pass
# - Repair malformed JSON if needed and write the final fingerprint

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".rtf",
    ".html", ".htm", ".tex",
    ".csv"  # sometimes writing lives in CSV; we read as text
}
DOCX_EXTS = {".docx"}

BASE64_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\\s]+", re.IGNORECASE)
BASE64_PLACEHOLDER_RE = re.compile(r"\[\[BASE64_IMAGE(?:_\d+)?\]\]")
HTML_TAG_RE = re.compile(r"<[A-Za-z/][^>]*>")
HTML_ENTITY_RE = re.compile(r"&[A-Za-z0-9#]+;")
DEFAULT_MAX_FILES = 2000
DEFAULT_MAX_BYTES_PER_FILE = 2_000_000  # 2 MB per file
DEFAULT_MAX_TOTAL_CHARS_FOR_LLM = 180_000  # excerpt cap; we send stats + representative excerpts
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
LEXICON_HINTS_FILENAME = "lexicon_hints.json"
AVOID_LIST_FILENAME = "config.avoid.txt"

def load_prompts() -> Dict[str, Any]:
    # Load externalized prompt templates located alongside this script.
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.json not found at {PROMPTS_PATH}")
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

def load_optional_lexicon_hints() -> Optional[Dict[str, Any]]:
    # Load optional lexicon hints from CWD or script directory.
    cwd_path = Path.cwd() / LEXICON_HINTS_FILENAME
    script_path = Path(__file__).resolve().parent / LEXICON_HINTS_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def parse_avoid_list(text: str) -> List[str]:
    items: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


def load_avoid_list() -> List[str]:
    # Load optional avoid-word list from CWD or script directory.
    cwd_path = Path.cwd() / AVOID_LIST_FILENAME
    script_path = Path(__file__).resolve().parent / AVOID_LIST_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    if not path:
        return []
    try:
        return parse_avoid_list(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def merge_avoid_list_into_hints(
    lexicon_hints: Optional[Dict[str, Any]],
    avoid_list: List[str]
) -> Optional[Dict[str, Any]]:
    if not avoid_list:
        return lexicon_hints
    hints = dict(lexicon_hints) if isinstance(lexicon_hints, dict) else {}
    existing = hints.get("avoid_words")
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
        hints["avoid_words"] = merged
    return hints

def get_prompt_value(prompts: Dict[str, Any], *path: str) -> Any:
    # Traverse a nested dict safely and fail fast if a key is missing.
    cur: Any = prompts
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing prompts key: {'.'.join(path)}")
        cur = cur[key]
    return cur


# ----------------------------
# Helpers: archive extraction
# ----------------------------

def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    # Support zip and common tar variants; raise on unknown formats.
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return

    # tar variants
    if any(name.endswith(s) for s in [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"]):
        mode = "r:*"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(dest_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


# ----------------------------
# Helpers: reading documents
# ----------------------------

def read_docx(path: Path) -> str:
    # Use python-docx if installed; otherwise fail with a clear message.
    if Document is None:
        raise RuntimeError("python-docx not available; cannot read .docx")
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def read_text_file(path: Path, max_bytes: int) -> str:
    # Read raw bytes, cap size, and decode with a few common encodings.
    # Try utf-8 first, fallback latin-1
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    # last resort: replace errors
    return raw.decode("utf-8", errors="replace")


FRONTMATTER_RE = re.compile(r"(?s)\A---\s*\n.*?\n---\s*\n")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

OCR_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st"
}

def normalize_text(s: str) -> str:
    # Strip frontmatter and normalize line breaks for consistent measurement.
    # Remove common YAML frontmatter (typical in markdown/blog)
    s = re.sub(FRONTMATTER_RE, "", s)
    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Fix common OCR ligatures.
    for src, repl in OCR_LIGATURES.items():
        if src in s:
            s = s.replace(src, repl)
    # Join hyphenated line breaks (OCR/line wrap artifacts).
    s = re.sub(HYPHEN_LINEBREAK_RE, r"\1\2", s)
    # Collapse too many blank lines a bit (keep structure)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def strip_base64_images(text: str) -> str:
    # Replace embedded base64 images with a placeholder to reduce token load.
    return BASE64_IMAGE_RE.sub("[[BASE64_IMAGE]]", text)


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


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def get_heading_at(lines: List[str], idx: int) -> Optional[Tuple[int, str, int]]:
    # Return (level, heading_text, span_lines) if a heading starts at idx.
    line = lines[idx]
    m = ATX_HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        return (level, m.group(2).strip(), 1)
    if idx + 1 < len(lines):
        underline = lines[idx + 1]
        if SETEXT_H1_RE.match(underline):
            return (1, line.strip(), 2)
        if SETEXT_H2_RE.match(underline):
            return (2, line.strip(), 2)
    return None


def is_reference_heading(text: str) -> bool:
    return normalize_heading_text(text) in REFERENCE_HEADINGS


def find_reference_sections(lines: List[str]) -> List[Tuple[int, int]]:
    # Return list of (start_idx, end_idx) line ranges for reference-like sections.
    sections: List[Tuple[int, int]] = []
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
    return re.sub(r"(``[^`\n]+``|`[^`\n]+`)", "", text)


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


def strip_html(text: str) -> str:
    # Remove HTML tags and block elements to exclude HTML from profiling.
    text = re.sub(r"(?is)<[A-Za-z][^>]*>.*?</[A-Za-z][^>]*>", "\n", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = HTML_TAG_RE.sub("", text)
    return text


def strip_html_entities(text: str) -> str:
    # Remove HTML entities (e.g., &nbsp;).
    return HTML_ENTITY_RE.sub("", text)


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
    # Remove non-author voice segments and inline citations for measurements/excerpts.
    text = strip_fenced_code_blocks(text)
    text = strip_non_voice_sections(text)
    text = strip_inline_code(text)
    text = strip_latex_math(text)
    text = strip_html(text)
    text = strip_html_entities(text)
    text = BASE64_PLACEHOLDER_RE.sub("", text)
    text = strip_inline_citations(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def iter_corpus_texts(root: Path, max_files: int, max_bytes_per_file: int) -> List[Tuple[str, str]]:
    """
    Returns list of (relative_path, text).
    """
    items: List[Tuple[str, str]] = []
    count = 0
    # Walk the extracted tree and collect readable text-like files.
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if count >= max_files:
            break
        ext = p.suffix.lower()
        try:
            if ext in TEXT_EXTS:
                txt = normalize_text(read_text_file(p, max_bytes_per_file))
            elif ext in DOCX_EXTS:
                txt = normalize_text(read_docx(p))
            else:
                continue
        except Exception:
            continue

        txt = strip_base64_images(txt)
        txt = filter_author_voice_text(txt)

        # Skip tiny/empty
        if len(txt) < 200:
            continue

        items.append((str(p.relative_to(root)), txt))
        count += 1
    return items


def extract_title(text: str) -> Optional[str]:
    # Try to detect a title from HTML or Markdown headings.
    # Try HTML <title>...</title>
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if title:
            return title

    # Try HTML <h1>...</h1>
    m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", text)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1))
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title

    lines = text.splitlines()
    # Try Markdown ATX heading
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading
        break

    # Try Markdown setext heading
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        if not line:
            continue
        underline = lines[i + 1].strip()
        if underline and all(ch == "=" for ch in underline):
            return line
        if underline and all(ch == "-" for ch in underline):
            return line
        break

    return None


def build_corpus_documents(files_and_texts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    # Build per-document metadata records for the fingerprint.
    documents: List[Dict[str, Any]] = []
    for rel_path, text in files_and_texts:
        words_list = words(text)
        sentences_list = split_sentences(text)
        paragraphs_list = split_paragraphs(text)
        documents.append({
            "path": rel_path,
            "name": Path(rel_path).stem,
            "title": extract_title(text),
            "description": None,
            "language": None,
            "locale": None,
            "genres": [],
            "time_range": {"start": None, "end": None},
            "size": {
                "words_est": len(words_list),
                "sentences_est": len(sentences_list),
                "paragraphs_est": len(paragraphs_list),
                "chars": len(text)
            }
        })
    return documents


# ----------------------------
# Measurement: token-ish stats
# ----------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

def words(text: str) -> List[str]:
    # Lightweight tokenizer for measurement only (not linguistically perfect).
    return WORD_RE.findall(text)

def split_sentences(text: str) -> List[str]:
    # naive sentence splitter; good enough for style stats
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    # If splitter fails (single long sentence), return as one
    return [s.strip() for s in sents if s.strip()]

def split_paragraphs(text: str) -> List[str]:
    # Paragraphs separated by blank lines.
    paras = PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def histogram(values: List[int], bins: List[Tuple[int, Optional[int]]]) -> List[float]:
    """
    bins: list of (lo, hi) inclusive, hi=None for open-ended.
    Returns proportions summing to 1.0 (or all zeros if empty).
    """
    # Convert values into normalized proportions over the requested bins.
    if not values:
        return [0.0] * len(bins)
    counts = [0] * len(bins)
    for v in values:
        for i, (lo, hi) in enumerate(bins):
            if v >= lo and (hi is None or v <= hi):
                counts[i] += 1
                break
    total = sum(counts)
    if total == 0:
        return [0.0] * len(bins)
    return [c / total for c in counts]


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token.
    return max(1, (len(text) + 3) // 4)


def estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    # Add a small per-message overhead to approximate chat tokenization.
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4
    return total + 2

def approx_rate_per_1000_words(count: int, total_words: int) -> float:
    # Normalize counts so they are comparable across corpora.
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

def compute_measurements(texts: List[str]) -> Dict[str, Any]:
    # Compute corpus-wide measurements for stylistic grounding.
    combined = "\n\n".join(texts)
    w = words(combined)
    total_words = len(w)

    # Sentence stats
    all_sent_lens: List[int] = []
    all_sents: List[str] = []
    for t in texts:
        sents = split_sentences(t)
        all_sents.extend(sents)
        for s in sents:
            all_sent_lens.append(len(words(s)))

    # Paragraph stats
    para_lens: List[int] = []
    one_sentence_paras = 0
    total_paras = 0
    for t in texts:
        paras = split_paragraphs(t)
        for p in paras:
            total_paras += 1
            sents = split_sentences(p)
            n = len(sents)
            para_lens.append(n)
            if n == 1:
                one_sentence_paras += 1

    # Punctuation counts
    punct = {
        "commas": combined.count(","),
        "semicolons": combined.count(";"),
        "colons": combined.count(":"),
        "exclamations": combined.count("!"),
        "questions": combined.count("?"),
        "parentheses_open": combined.count("("),
        "parentheses_close": combined.count(")"),
        "em_dashes": combined.count("—"),
        "en_dashes": combined.count("–"),
        "hyphen_dashes": combined.count(" - "),
        "ellipses_unicode": combined.count("…"),
        "ellipses_three_dots": combined.count("..."),
        "quotes_double": combined.count('"'),
        "quotes_single": combined.count("'"),
    }

    # Contractions rate (rough)
    contraction_hits = len(re.findall(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b", combined))
    contraction_rate = contraction_hits / max(1, total_words)

    # Oxford comma heuristic (rough): ", and"/", or" vs " and"/" or" in lists is complex.
    # We’ll just provide a signal: fraction of occurrences of ", and " among " and " contexts
    and_total = combined.lower().count(" and ")
    comma_and = combined.lower().count(", and ")
    oxford_signal = comma_and / max(1, and_total)

    # Frequent phrases (bigrams/trigrams) excluding stop-ish tokens
    stop = set([
        "the","a","an","and","or","but","if","then","to","of","in","on","for","with","as",
        "is","are","was","were","be","been","it","that","this","these","those","i","you",
        "he","she","they","we","me","him","her","them","my","your","our","their","at","by"
    ])

    toks = [t.lower() for t in w]
    token_counts = collections.Counter(toks)

    # Rare-word signals: low-frequency tokens that the author rarely uses.
    def is_candidate_rare(token: str) -> bool:
        if token in stop:
            return False
        if len(token) < 4:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        return token.isalpha()

    max_count = max(2, int(total_words * 0.0001))
    max_count = min(5, max_count)
    rare_candidates = [
        (token, count) for token, count in token_counts.items()
        if count <= max_count and is_candidate_rare(token)
    ]
    rare_candidates.sort(key=lambda x: (x[1], x[0]))
    rare_words = [
        {
            "word": token,
            "count": count,
            "rate_per_1000w": approx_rate_per_1000_words(count, total_words)
        }
        for token, count in rare_candidates[:40]
    ]
    def ngrams(n: int) -> Iterable[str]:
        for i in range(0, len(toks) - n + 1):
            chunk = toks[i:i+n]
            # skip ngrams that are mostly stopwords
            if sum(1 for x in chunk if x in stop) >= n - 1:
                continue
            yield " ".join(chunk)

    big = collections.Counter(ngrams(2)).most_common(30)
    tri = collections.Counter(ngrams(3)).most_common(30)

    # Function word profile (classic stylometry signal).
    function_words = [
        "the","a","an","and","or","but","if","then","because","so","while","although","though",
        "of","in","on","for","with","as","by","from","to","into","over","under","between",
        "is","are","was","were","be","been","being","it","this","that","these","those",
        "i","we","you","he","she","they","me","us","him","her","them","my","our","your","their",
        "not","no","nor","very","also","even","only","just","rather","however","therefore"
    ]
    fw_counts = token_counts
    fw_rates = {fw: approx_rate_per_1000_words(fw_counts.get(fw, 0), total_words) for fw in function_words}
    fw_top = sorted(
        [{"word": fw, "count": fw_counts.get(fw, 0)} for fw in function_words],
        key=lambda x: x["count"],
        reverse=True
    )[:20]

    # Stance/persona signals (interpretable markers).
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

    # Sentence openers / rhetorical templates (top sentence starts).
    sent_openers = collections.Counter()
    transition_terms = {
        "however","therefore","moreover","furthermore","nevertheless","nonetheless",
        "for example","for instance","in short","in sum","in practice","in effect",
        "first","second","third","finally","overall"
    }
    transition_hits = collections.Counter()
    for s in all_sents:
        ws = [t.lower() for t in words(s)]
        if len(ws) >= 2:
            opener = " ".join(ws[:3]) if len(ws) >= 3 else " ".join(ws[:2])
            if sum(1 for x in opener.split() if x in stop) <= 1:
                sent_openers[opener] += 1
        if ws:
            start_two = " ".join(ws[:2])
            start_three = " ".join(ws[:3]) if len(ws) >= 3 else ""
            for cand in (start_three, start_two, ws[0]):
                if cand and cand in transition_terms:
                    transition_hits[cand] += 1
                    break

    sent_bins = [(0,9),(10,17),(18,25),(26,40),(41,None)]
    sent_hist = histogram(all_sent_lens, sent_bins)

    para_bins = [(1,1),(2,3),(4,5),(6,8),(9,None)]
    para_hist = histogram(para_lens, para_bins)

    def safe_mean(xs: List[int]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def safe_stdev(xs: List[int]) -> float:
        return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0

    measurements = {
        "totals": {
            "documents_used": len(texts),
            "total_words_est": total_words,
            "total_sentences_est": len(all_sents),
            "total_paragraphs_est": total_paras
        },
        "sentence": {
            "length_words": {
                "mean": safe_mean(all_sent_lens),
                "stdev": safe_stdev(all_sent_lens),
                "histogram_bins": ["<10", "10-17", "18-25", "26-40", ">40"],
                "histogram_p": sent_hist
            }
        },
        "paragraph": {
            "length_sentences_histogram_bins": ["1", "2-3", "4-5", "6-8", ">8"],
            "length_sentences_histogram_p": para_hist,
            "one_sentence_paragraph_rate": one_sentence_paras / max(1, total_paras)
        },
        "punctuation": {
            "counts": punct,
            "rates_per_1000w": {k: approx_rate_per_1000_words(v, total_words) for k, v in punct.items()},
            "comma_density_per_100w": (punct["commas"] / max(1, total_words)) * 100.0
        },
        "orthography_signals": {
            "contractions_rate": contraction_rate,
            "oxford_comma_signal": oxford_signal,
            "spelling_variant": detect_english_spelling_variant(combined)
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
        },
        "lexical_signals": {
            "rare_words": rare_words,
            "rare_word_max_count": max_count,
            "rare_word_min_length": 4
        },
        "common_phrases": {
            "bigrams_top": [{"phrase": p, "count": c} for p, c in big],
            "trigrams_top": [{"phrase": p, "count": c} for p, c in tri]
        }
    }
    return measurements


def pick_representative_excerpts(files_and_texts: List[Tuple[str, str]], max_total_chars: int) -> List[Dict[str, str]]:
    """
    Pick excerpts from multiple files to show the LLM real style.
    We keep this bounded; the stats do most of the work.
    """
    excerpts: List[Dict[str, str]] = []
    used = 0

    def score_paragraph_voice(p: str) -> float:
        # Heuristic: prefer clean, narrative paragraphs with fewer artifacts.
        ws = words(p)
        if not ws:
            return 0.0
        score = 1.0
        wc = len(ws)
        if wc < 60:
            score *= 0.3
        elif wc > 280:
            score *= 0.6
        if re.match(r"^\s*[-*+]\s+", p) or re.match(r"^\s*\d+\.\s+", p):
            score *= 0.5
        upper_ratio = sum(1 for ch in p if ch.isupper()) / max(1, len(p))
        if upper_ratio > 0.2:
            score *= 0.7
        digit_ratio = sum(1 for ch in p if ch.isdigit()) / max(1, len(p))
        if digit_ratio > 0.08:
            score *= 0.7
        if re.search(r"\b(ibid|supra|infra|cf|op\.|cit|pp?)\b", p.lower()):
            score *= 0.4
        if re.search(r"\[[0-9,\s]+\]", p):
            score *= 0.5
        return score

    candidates: List[Tuple[float, str, str]] = []
    for rel, txt in files_and_texts:
        for para in split_paragraphs(txt):
            if len(para) < 200:
                continue
            candidates.append((score_paragraph_voice(para), rel, para))

    if not candidates:
        return excerpts

    # Seed with top paragraph per document for variety.
    by_doc: Dict[str, List[Tuple[float, str]]] = {}
    for score, rel, para in candidates:
        by_doc.setdefault(rel, []).append((score, para))
    for rel in by_doc:
        by_doc[rel].sort(key=lambda x: x[0], reverse=True)
    for rel, paras in by_doc.items():
        if used >= max_total_chars:
            break
        score, para = paras[0]
        take = min(len(para), max_total_chars - used)
        snippet = para[:take]
        excerpts.append({"path": rel, "excerpt": snippet})
        used += len(snippet)

    # Fill remaining budget with highest-scoring paragraphs overall.
    candidates.sort(key=lambda x: x[0], reverse=True)
    for score, rel, para in candidates:
        if used >= max_total_chars:
            break
        if any(e["excerpt"] == para for e in excerpts):
            continue
        take = min(len(para), max_total_chars - used)
        snippet = para[:take]
        excerpts.append({"path": rel, "excerpt": snippet})
        used += len(snippet)

    return excerpts


# ----------------------------
# OpenAI-compatible client
# ----------------------------

@dataclasses.dataclass
class LLMConfig:
    # Minimal OpenAI-compatible configuration container.
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 6000
    temperature: float = 0.2
    timeout_seconds: int = 120
    extra_headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    max_prompt_tokens: int = 100000
    max_retries: int = 2
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 8.0

def load_config(path: Path) -> LLMConfig:
    # Load API configuration and apply defaults.
    data = json.loads(path.read_text(encoding="utf-8"))
    max_tokens = int(data.get("max_tokens", 6000))
    return LLMConfig(
        api_key=data["api_key"],
        base_url=data["base_url"].rstrip("/"),
        model=data["model"],
        max_tokens=max_tokens,
        temperature=float(data.get("temperature", 0.2)),
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_headers=dict(data.get("extra_headers", {})),
        max_prompt_tokens=int(data.get("max_prompt_tokens", max_tokens)),
        max_retries=int(data.get("max_retries", 2)),
        backoff_base_seconds=float(data.get("backoff_base_seconds", 1.0)),
        backoff_max_seconds=float(data.get("backoff_max_seconds", 8.0)),
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
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:2000]}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, RuntimeError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
            jitter = random.uniform(0, backoff * 0.2)
            sleep_s = backoff + jitter
            print(f"LLM request failed (attempt {attempt + 1}/{cfg.max_retries + 1}); retrying in {sleep_s:.1f}s.", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {cfg.max_retries + 1} attempts: {last_err}")


# ----------------------------
# Prompting
# ----------------------------

def fingerprint_schema_template(prompts: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_prompt_value(prompts, "fingerprint", "schema_hint")
    if not isinstance(schema, dict):
        raise TypeError("prompts.fingerprint.schema_hint must be an object")
    return copy.deepcopy(schema)

def build_fingerprint_prompt(
    measurements: Dict[str, Any],
    excerpts: List[Dict[str, str]],
    cfg: LLMConfig,
    prompts: Dict[str, Any],
    lexicon_hints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    # Fill the fingerprint prompt template with runtime data.
    schema = fingerprint_schema_template(prompts)

    system = get_prompt_value(prompts, "fingerprint", "system")
    user_template = get_prompt_value(prompts, "fingerprint", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.fingerprint.user must be an object")
    user = copy.deepcopy(user_template)
    user["schema_hint"] = schema
    user["measurements"] = measurements
    user["excerpts"] = excerpts
    if lexicon_hints:
        user["lexicon_hints"] = lexicon_hints

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def slim_fingerprint_for_merge(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(fingerprint)
    slim.pop("measurements", None)
    return slim


def build_merge_prompt(
    fingerprint_a: Dict[str, Any],
    fingerprint_b: Dict[str, Any],
    measurements: Dict[str, Any],
    cfg: LLMConfig,
    prompts: Dict[str, Any]
) -> List[Dict[str, str]]:
    # Merge two partial fingerprints using the LLM with a strict JSON-only prompt.
    schema = fingerprint_schema_template(prompts)
    system = get_prompt_value(prompts, "merge", "system")
    user_template = get_prompt_value(prompts, "merge", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.merge.user must be an object")
    user = copy.deepcopy(user_template)
    user["schema_hint"] = schema
    user["measurements"] = measurements
    user["fingerprint_a"] = fingerprint_a
    user["fingerprint_b"] = fingerprint_b
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def build_phrase_validation_prompt(
    phrases: List[Dict[str, Any]],
    prompts: Dict[str, Any]
) -> List[Dict[str, str]]:
    # Build a prompt to validate common phrases (OCR/citation noise filtering).
    system = get_prompt_value(prompts, "validate_phrases", "system")
    user_template = get_prompt_value(prompts, "validate_phrases", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.validate_phrases.user must be an object")
    user = copy.deepcopy(user_template)
    user["phrases"] = phrases
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def validate_common_phrases(
    cfg: LLMConfig,
    phrases: List[Dict[str, Any]],
    prompts: Dict[str, Any]
) -> Dict[str, Any]:
    # Ask the LLM to flag OCR/citation noise in common phrases.
    messages = build_phrase_validation_prompt(phrases, prompts)
    raw = chat_completions(cfg, messages)
    try:
        return parse_json_strict(raw)
    except Exception:
        return repair_json_with_llm(cfg, raw, prompts)


def chunk_excerpts(
    excerpts: List[Dict[str, str]],
    measurements: Dict[str, Any],
    cfg: LLMConfig,
    max_prompt_tokens: int,
    prompts: Dict[str, Any],
    lexicon_hints: Optional[Dict[str, Any]] = None
) -> List[List[Dict[str, str]]]:
    # Split excerpts into prompt-sized batches if the prompt is too large.
    base_messages = build_fingerprint_prompt(measurements, [], cfg, prompts, lexicon_hints)
    base_tokens = estimate_tokens_for_messages(base_messages)
    if base_tokens >= max_prompt_tokens:
        return [excerpts[:1]] if excerpts else [[]]
    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_tokens = base_tokens

    for ex in excerpts:
        ex_tokens = estimate_tokens(json.dumps(ex, ensure_ascii=False))
        if current and (current_tokens + ex_tokens) > max_prompt_tokens:
            batches.append(current)
            current = [ex]
            current_tokens = base_tokens + ex_tokens
        else:
            current.append(ex)
            current_tokens += ex_tokens

    if current:
        batches.append(current)
    return batches


def parse_json_strict(s: str) -> Dict[str, Any]:
    # Strip code fences if present and parse strictly as JSON.
    s = s.strip()
    # Some models wrap in ```json ... ```; strip that if present.
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


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.llm.json (default: ./config.llm.json if present; else next to script)"
    )
    ap.add_argument("-a", "--archive", required=True, type=Path, help="Path to .zip/.tar* corpus archive")
    ap.add_argument(
        "-o",
        "--out",
        required=True,
        type=Path,
        help="Output fingerprint JSON path (adds .json if no extension)"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable progress logging")
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Maximum prompt tokens before chunking (default: config max_prompt_tokens)"
    )
    ap.add_argument(
        "--profile-id",
        default=None,
        help="Profile ID to set in output JSON (default: output filename without .json)"
    )
    ap.add_argument(
        "--author-name",
        default=None,
        help="Author name (metadata only; default: output filename without .json)"
    )
    ap.add_argument(
        "--no-phrase-validation",
        action="store_true",
        help="Disable the LLM pass that validates common phrases (OCR/citation noise filtering)"
    )
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--max-bytes-per-file", type=int, default=DEFAULT_MAX_BYTES_PER_FILE)
    ap.add_argument("--excerpt-char-budget", type=int, default=DEFAULT_MAX_TOTAL_CHARS_FOR_LLM)
    args = ap.parse_args()

    if args.out.suffix == "":
        args.out = args.out.with_suffix(".json")

    if args.config is None:
        # Resolve config: prefer current working directory, then script directory.
        cwd_cfg = Path.cwd() / "config.llm.json"
        script_cfg = Path(__file__).resolve().parent / "config.llm.json"
        args.config = cwd_cfg if cwd_cfg.exists() else script_cfg

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    vprint(f"Using config: {args.config}")
    vprint(f"Output path: {args.out}")

    cfg = load_config(args.config)
    prompts = load_prompts()
    lexicon_hints = load_optional_lexicon_hints()
    avoid_list = load_avoid_list()
    if avoid_list:
        lexicon_hints = merge_avoid_list_into_hints(lexicon_hints, avoid_list)
    if args.max_prompt_tokens is not None:
        # Allow CLI override for chunking threshold.
        cfg.max_prompt_tokens = args.max_prompt_tokens

    if not args.archive.exists():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 2

    vprint(f"Extracting archive: {args.archive}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        extract_archive(args.archive, tmp)

        vprint("Reading corpus files...")
        files_and_texts = iter_corpus_texts(tmp, max_files=args.max_files, max_bytes_per_file=args.max_bytes_per_file)
        if not files_and_texts:
            print("No readable corpus files found (txt/md/docx/html/etc).", file=sys.stderr)
            return 3

        corpus_documents = build_corpus_documents(files_and_texts)
        vprint(f"Found {len(files_and_texts)} files; computing measurements...")
        texts = [t for _, t in files_and_texts]
        measurements = compute_measurements(texts)
        if not args.no_phrase_validation:
            vprint("Validating common phrases with LLM...")
            common = measurements.get("common_phrases", {})
            candidates: List[Dict[str, Any]] = []
            for item in common.get("bigrams_top", []) or []:
                candidates.append({
                    "phrase": item.get("phrase", ""),
                    "count": item.get("count", 0),
                    "ngram": 2
                })
            for item in common.get("trigrams_top", []) or []:
                candidates.append({
                    "phrase": item.get("phrase", ""),
                    "count": item.get("count", 0),
                    "ngram": 3
                })
            if candidates:
                validation = validate_common_phrases(cfg, candidates, prompts)
                decisions = validation.get("decisions", []) or []
                decision_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
                for d in decisions:
                    phrase = d.get("phrase")
                    ngram = d.get("ngram")
                    if isinstance(phrase, str) and isinstance(ngram, int):
                        decision_map[(phrase, ngram)] = d

                validated_bi: List[Dict[str, Any]] = []
                validated_tri: List[Dict[str, Any]] = []
                dropped: List[Dict[str, Any]] = []

                for item in common.get("bigrams_top", []) or []:
                    phrase = item.get("phrase", "")
                    decision = decision_map.get((phrase, 2))
                    if decision and decision.get("decision") == "drop":
                        dropped.append({
                            "phrase": phrase,
                            "count": item.get("count", 0),
                            "ngram": 2,
                            "reason": decision.get("reason", "")
                        })
                    else:
                        validated_bi.append(item)

                for item in common.get("trigrams_top", []) or []:
                    phrase = item.get("phrase", "")
                    decision = decision_map.get((phrase, 3))
                    if decision and decision.get("decision") == "drop":
                        dropped.append({
                            "phrase": phrase,
                            "count": item.get("count", 0),
                            "ngram": 3,
                            "reason": decision.get("reason", "")
                        })
                    else:
                        validated_tri.append(item)

                measurements["common_phrases_validation"] = {
                    "validated": {
                        "bigrams_top": validated_bi,
                        "trigrams_top": validated_tri
                    },
                    "dropped": dropped,
                    "notes": validation.get("notes", [])
                }
            else:
                measurements["common_phrases_validation"] = {
                    "validated": {"bigrams_top": [], "trigrams_top": []},
                    "dropped": [],
                    "notes": ["No common phrases to validate."]
                }
        vprint("Selecting representative excerpts...")
        excerpts = pick_representative_excerpts(files_and_texts, max_total_chars=args.excerpt_char_budget)

        vprint("Calling LLM to synthesize fingerprint...")
        messages = build_fingerprint_prompt(measurements, excerpts, cfg, prompts, lexicon_hints)
        prompt_tokens = estimate_tokens_for_messages(messages)
        if prompt_tokens <= cfg.max_prompt_tokens:
            raw = chat_completions(cfg, messages)
            try:
                fingerprint = parse_json_strict(raw)
            except Exception:
                vprint("Invalid JSON returned; attempting repair...")
                fingerprint = repair_json_with_llm(cfg, raw, prompts)
        else:
            vprint(f"Prompt too large ({prompt_tokens} tokens); chunking excerpts...")
            batches = chunk_excerpts(excerpts, measurements, cfg, cfg.max_prompt_tokens, prompts, lexicon_hints)
            vprint(f"Chunked into {len(batches)} excerpt batches.")
            partials: List[Dict[str, Any]] = []
            for idx, batch in enumerate(batches, start=1):
                vprint(f"Synthesizing partial fingerprint {idx}/{len(batches)}...")
                batch_messages = build_fingerprint_prompt(measurements, batch, cfg, prompts, lexicon_hints)
                raw = chat_completions(cfg, batch_messages)
                try:
                    partial = parse_json_strict(raw)
                except Exception:
                    vprint("Invalid JSON returned; attempting repair...")
                    partial = repair_json_with_llm(cfg, raw, prompts)
                partials.append(partial)

            fingerprint = partials[0]
            for idx, partial in enumerate(partials[1:], start=2):
                vprint(f"Merging partial fingerprint {idx}/{len(partials)}...")
                merge_messages = build_merge_prompt(
                    slim_fingerprint_for_merge(fingerprint),
                    slim_fingerprint_for_merge(partial),
                    measurements,
                    cfg,
                    prompts
                )
                raw = chat_completions(cfg, merge_messages)
                try:
                    fingerprint = parse_json_strict(raw)
                except Exception:
                    vprint("Invalid JSON returned; attempting repair...")
                    fingerprint = repair_json_with_llm(cfg, raw, prompts)

        # Ensure essential fields
        fingerprint.setdefault("schema_version", "1.0.0")
        inferred_id = args.out.stem
        profile_id = args.profile_id or inferred_id
        author_name = args.author_name or inferred_id
        fingerprint["profile_id"] = fingerprint.get("profile_id") or profile_id
        fingerprint.setdefault("metadata", {})
        fingerprint["metadata"].setdefault("author", {"name": author_name, "is_self": True})
        fingerprint["metadata"].setdefault("extraction", {})
        # put model and date if missing
        fingerprint["metadata"].setdefault("extraction", {})
        if isinstance(fingerprint["metadata"].get("extraction"), dict):
            fingerprint["metadata"]["extraction"].setdefault("model", cfg.model)
            fingerprint["metadata"]["extraction"].setdefault("methods", ["hybrid"])
            fingerprint["metadata"]["extraction"].setdefault("confidence", "medium")

        corpus = fingerprint["metadata"].setdefault("corpus", {})
        corpus_defaults = {}
        for key in ("description", "language", "locale", "genres", "time_range"):
            if key in corpus:
                corpus_defaults[key] = corpus.get(key)

        for doc in corpus_documents:
            for key, value in corpus_defaults.items():
                if value is not None:
                    doc.setdefault(key, value)

        for key in corpus_defaults.keys():
            corpus.pop(key, None)

        corpus["document_count"] = len(corpus_documents)
        corpus["documents"] = corpus_documents

        corpus_size = corpus.get("size")
        if not isinstance(corpus_size, dict):
            corpus_size = {}
        total_words = measurements.get("totals", {}).get("total_words_est")
        if isinstance(total_words, int):
            corpus_size.setdefault("words_est", total_words)
        corpus_size.pop("documents", None)
        corpus["size"] = corpus_size

        # Always embed measurements (verbatim local measurements)
        fingerprint["measurements"] = measurements

        vprint("Writing fingerprint JSON...")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
        vprint("Done.")
        print(f"Wrote fingerprint JSON to: {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
