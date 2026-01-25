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
from pathlib import Path
import copy
from typing import Any, Dict, List

import requests


# ---- same lightweight stats as fingerprint script ----
# Script overview:
# - Measure the input Markdown (lightweight, explainable stats)
# - Build a prompt using an externalized template (prompts.json)
# - Call an OpenAI-compatible LLM to rewrite while preserving meaning
# - Handle oversized prompts by chunking the Markdown
# - Strip base64 images before prompting, then reinsert after rewriting
# - Preserve non-voice blocks (blockquotes, references, footnotes, citations) verbatim

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
BASE64_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\\s]+", re.IGNORECASE)
BASE64_PLACEHOLDER_RE = re.compile(r"\[\[BASE64_IMAGE_\d+\]\]")
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"

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

def load_prompts() -> Dict[str, Any]:
    # Load externalized prompt templates located alongside this script.
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.json not found at {PROMPTS_PATH}")
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

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
    text = strip_non_voice_sections(text)
    text = BASE64_PLACEHOLDER_RE.sub("", text)
    text = strip_inline_citations(text)
    text = FROZEN_BLOCK_RE.sub("", text)
    text = CITATION_PLACEHOLDER_RE.sub("", text)
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
            "spelling_variant": detect_english_spelling_variant(text)
        }
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
        max_prompt_tokens: int
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.extra_headers = extra_headers
        self.max_prompt_tokens = max_prompt_tokens

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
        max_prompt_tokens=int(data.get("max_prompt_tokens", max_tokens))
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
    r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_seconds)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:2000]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

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
    prompts: Dict[str, Any]
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
    if args.max_prompt_tokens is not None:
        # Allow CLI override for chunking threshold.
        cfg.max_prompt_tokens = args.max_prompt_tokens

    if not args.fingerprint.exists():
        # Fall back to the script directory if fingerprint isn't in CWD.
        script_fp = Path(__file__).resolve().parent / args.fingerprint.name
        if script_fp.exists():
            args.fingerprint = script_fp
        else:
            print(f"Fingerprint not found: {args.fingerprint}", file=sys.stderr)
            return 2
    if not args.inp.exists():
        print(f"Input markdown not found: {args.inp}", file=sys.stderr)
        return 2

    vprint("Loading fingerprint and input...")
    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    input_md = args.inp.read_text(encoding="utf-8")
    # Strip base64 images to keep prompts within token limits.
    input_md, base64_map = strip_base64_images(input_md)
    if base64_map:
        vprint(f"Stripped {len(base64_map)} base64 image embed(s) from prompt.")
    # Mask non-voice blocks and inline citations so they are preserved verbatim.
    input_md, frozen_blocks = mask_non_voice_blocks(input_md)
    input_md, citation_map = mask_inline_citations(input_md)

    all_deviations: List[Any] = []
    outputs: List[str] = []

    def build_messages_for_chunk(md_chunk: str) -> List[Dict[str, str]]:
        # Build prompts per chunk using local measurements.
        input_meas = compute_measurements(filter_author_voice_text(md_chunk))
        return build_apply_prompt(fingerprint, md_chunk, input_meas, cfg, prompts)

    initial_messages = build_messages_for_chunk(input_md)
    initial_tokens = estimate_tokens_for_messages(initial_messages)
    if initial_tokens <= cfg.max_prompt_tokens:
        vprint("Calling LLM to apply fingerprint...")
        raw = chat_completions(cfg, initial_messages)

        try:
            out_obj = parse_json_strict(raw)
        except Exception:
            vprint("Invalid JSON returned; attempting repair...")
            out_obj = repair_json_with_llm(cfg, raw, prompts)

        final_md = out_obj.get("final_markdown")
        if not isinstance(final_md, str) or not final_md.strip():
            print("LLM did not return final_markdown.", file=sys.stderr)
            print(raw)
            return 3
        # Ensure any frozen blocks and citation placeholders survive.
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
        final_md = restore_placeholders(final_md, frozen_blocks)
        final_md = restore_placeholders(final_md, citation_map)
        final_md = restore_base64_images(final_md, base64_map, find_base64_placeholders(final_md))
        outputs.append(final_md)
        all_deviations.extend(out_obj.get("deviations", []) or [])
    else:
        vprint(f"Prompt too large ({initial_tokens} tokens); chunking input...")
        chunks = chunk_markdown(input_md, build_messages_for_chunk, cfg.max_prompt_tokens)
        vprint(f"Chunked into {len(chunks)} parts.")
        for idx, chunk in enumerate(chunks, start=1):
            vprint(f"Rewriting chunk {idx}/{len(chunks)}...")
            messages = build_messages_for_chunk(chunk)
            raw = chat_completions(cfg, messages)
            try:
                out_obj = parse_json_strict(raw)
            except Exception:
                vprint("Invalid JSON returned; attempting repair...")
                out_obj = repair_json_with_llm(cfg, raw, prompts)

            final_md = out_obj.get("final_markdown")
            if not isinstance(final_md, str) or not final_md.strip():
                print("LLM did not return final_markdown.", file=sys.stderr)
                print(raw)
                return 3
            # Ensure any frozen blocks and citation placeholders survive.
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

    # Stitch chunks back together, preserving the original order.
    final_md = "\n\n".join(s.strip() for s in outputs if s.strip()).strip()
    out_path = args.out or args.inp.with_suffix(args.inp.suffix + ".styled.md")
    vprint(f"Writing output: {out_path}")
    out_path.write_text(final_md, encoding="utf-8")
    print(f"Wrote rewritten markdown to: {out_path}")

    # Optionally also write deviations report
    if all_deviations:
        rep = out_path.with_suffix(out_path.suffix + ".deviations.json")
        rep.write_text(json.dumps(all_deviations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote deviations report to: {rep}")
    vprint("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
