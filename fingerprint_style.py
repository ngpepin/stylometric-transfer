#!/usr/bin/env python3
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
  python fingerprint_style.py --config config.llm.json --archive corpus.zip --out fingerprint.json
"""

from __future__ import annotations

import argparse
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
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from docx import Document  # python-docx
except Exception:
    Document = None  # optional


TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".rtf",
    ".html", ".htm", ".tex",
    ".csv"  # sometimes writing lives in CSV; we read as text
}
DOCX_EXTS = {".docx"}

DEFAULT_MAX_FILES = 2000
DEFAULT_MAX_BYTES_PER_FILE = 2_000_000  # 2 MB per file
DEFAULT_MAX_TOTAL_CHARS_FOR_LLM = 180_000  # excerpt cap; we send stats + representative excerpts


# ----------------------------
# Helpers: archive extraction
# ----------------------------

def extract_archive(archive_path: Path, dest_dir: Path) -> None:
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
    if Document is None:
        raise RuntimeError("python-docx not available; cannot read .docx")
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def read_text_file(path: Path, max_bytes: int) -> str:
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

def normalize_text(s: str) -> str:
    # Remove common YAML frontmatter (typical in markdown/blog)
    s = re.sub(FRONTMATTER_RE, "", s)
    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse too many blank lines a bit (keep structure)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def iter_corpus_texts(root: Path, max_files: int, max_bytes_per_file: int) -> List[Tuple[str, str]]:
    """
    Returns list of (relative_path, text).
    """
    items: List[Tuple[str, str]] = []
    count = 0
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

        # Skip tiny/empty
        if len(txt) < 200:
            continue

        items.append((str(p.relative_to(root)), txt))
        count += 1
    return items


# ----------------------------
# Measurement: token-ish stats
# ----------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

def words(text: str) -> List[str]:
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
    paras = PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def histogram(values: List[int], bins: List[Tuple[int, Optional[int]]]) -> List[float]:
    """
    bins: list of (lo, hi) inclusive, hi=None for open-ended.
    Returns proportions summing to 1.0 (or all zeros if empty).
    """
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

def approx_rate_per_1000_words(count: int, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    return (count / total_words) * 1000.0

def compute_measurements(texts: List[str]) -> Dict[str, Any]:
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
    def ngrams(n: int) -> Iterable[str]:
        for i in range(0, len(toks) - n + 1):
            chunk = toks[i:i+n]
            # skip ngrams that are mostly stopwords
            if sum(1 for x in chunk if x in stop) >= n - 1:
                continue
            yield " ".join(chunk)

    big = collections.Counter(ngrams(2)).most_common(30)
    tri = collections.Counter(ngrams(3)).most_common(30)

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
            "oxford_comma_signal": oxford_signal
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

    # Prefer mid-sized pieces and variety
    sorted_items = sorted(files_and_texts, key=lambda x: abs(len(x[1]) - 6000))
    for rel, txt in sorted_items:
        if used >= max_total_chars:
            break
        snippet = txt[:6000]
        if len(snippet) < 800:
            continue
        take = min(len(snippet), max_total_chars - used)
        snippet = snippet[:take]
        excerpts.append({"path": rel, "excerpt": snippet})
        used += len(snippet)

    return excerpts


# ----------------------------
# OpenAI-compatible client
# ----------------------------

@dataclasses.dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 6000
    temperature: float = 0.2
    timeout_seconds: int = 120
    extra_headers: Dict[str, str] = dataclasses.field(default_factory=dict)

def load_config(path: Path) -> LLMConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LLMConfig(
        api_key=data["api_key"],
        base_url=data["base_url"].rstrip("/"),
        model=data["model"],
        max_tokens=int(data.get("max_tokens", 6000)),
        temperature=float(data.get("temperature", 0.2)),
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_headers=dict(data.get("extra_headers", {})),
    )

def chat_completions(cfg: LLMConfig, messages: List[Dict[str, str]]) -> str:
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
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:2000]}")


# ----------------------------
# Prompting
# ----------------------------

def fingerprint_schema_template() -> Dict[str, Any]:
    """
    A pragmatic schema to keep output consistent. You can expand it.
    """
    return {
        "schema_version": "1.0.0",
        "profile_id": "string",
        "metadata": {
            "author": {"name": "string", "is_self": True},
            "corpus": {
                "description": "string",
                "language": "en",
                "locale": "en-CA",
                "genres": ["essay", "memoir", "technical", "fiction", "email", "notes", "other"],
                "time_range": {"start": "YYYY-MM-DD|null", "end": "YYYY-MM-DD|null"},
                "size": {"documents": "int", "words_est": "int", "pages_est": "int|null"},
                "sampling": {"method": "full|stratified|random|manual", "notes": "string"}
            },
            "extraction": {
                "model": "string",
                "date": "YYYY-MM-DD",
                "methods": ["llm_summary", "statistical_counts", "hybrid"],
                "confidence": "low|medium|high",
                "limitations": ["string"]
            }
        },
        "measurements": {"note": "include the computed measurement bundle here verbatim"},
        "targets": {"note": "style constraints/targets (orthography/punctuation/sentence/paragraph/lexical/semantics/rhetoric/persona)"},
        "lexicon": {"note": "preferred/avoid words/phrases and synonym preferences"},
        "templates": {"note": "syntactic patterns, paragraph moves, rhetorical moves"},
        "controls": {"note": "priority_order, strictness, rewrite_policy"},
        "validators": {"note": "scoring weights and checks"},
        "derived_instructions": {
            "system_style": "string",
            "rewrite_prompt": "string",
            "generation_prompt": "string"
        }
    }

def build_fingerprint_prompt(measurements: Dict[str, Any], excerpts: List[Dict[str, str]], cfg: LLMConfig) -> List[Dict[str, str]]:
    schema = fingerprint_schema_template()

    system = (
        "You are a style profiler and JSON generator.\n"
        "You MUST output valid JSON only (no markdown, no extra commentary).\n"
        "Do not include trailing commas. Use double quotes for all strings.\n"
        "If uncertain, use null and record a limitation.\n"
        "Prefer distributions over single averages.\n"
        "Avoid inventing stylistic claims not supported by provided measurements/excerpts.\n"
    )

    user = {
        "task": "Construct a comprehensive STYLE FINGERPRINT JSON for the author from the provided measurements and excerpts.",
        "output_requirements": [
            "Output MUST be valid JSON only.",
            "Must include keys: schema_version, profile_id, metadata, measurements, targets, lexicon, templates, controls, validators, derived_instructions.",
            "metadata.extraction.model should be set to the model name provided.",
            "Embed the provided measurements verbatim under measurements.",
            "Use controlled vocabulary values where possible (low|medium|high, rare|sometimes|often, etc.).",
            "Include numeric targets/ranges based on measurements (reasonable ranges reflecting variability).",
            "Include derived_instructions.system_style as concise bullet rules, and derived_instructions.rewrite_prompt / generation_prompt as templates."
        ],
        "schema_hint": schema,
        "measurements": measurements,
        "excerpts": excerpts
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def parse_json_strict(s: str) -> Dict[str, Any]:
    s = s.strip()
    # Some models wrap in ```json ... ```; strip that if present.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)

def repair_json_with_llm(cfg: LLMConfig, bad_output: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a JSON repair tool. Output valid JSON only."},
        {"role": "user", "content": json.dumps({
            "task": "Repair the following to be valid JSON. Preserve all fields and content as much as possible.",
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
    ap.add_argument("--config", required=True, type=Path, help="Path to config.llm.json")
    ap.add_argument("--archive", required=True, type=Path, help="Path to .zip/.tar* corpus archive")
    ap.add_argument("--out", required=True, type=Path, help="Output fingerprint JSON path")
    ap.add_argument("--profile-id", default="author_style_v1", help="Profile ID to set in output JSON")
    ap.add_argument("--author-name", default="(self)", help="Author name (metadata only)")
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--max-bytes-per-file", type=int, default=DEFAULT_MAX_BYTES_PER_FILE)
    ap.add_argument("--excerpt-char-budget", type=int, default=DEFAULT_MAX_TOTAL_CHARS_FOR_LLM)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not args.archive.exists():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        extract_archive(args.archive, tmp)

        files_and_texts = iter_corpus_texts(tmp, max_files=args.max_files, max_bytes_per_file=args.max_bytes_per_file)
        if not files_and_texts:
            print("No readable corpus files found (txt/md/docx/html/etc).", file=sys.stderr)
            return 3

        texts = [t for _, t in files_and_texts]
        measurements = compute_measurements(texts)
        excerpts = pick_representative_excerpts(files_and_texts, max_total_chars=args.excerpt_char_budget)

        messages = build_fingerprint_prompt(measurements, excerpts, cfg)
        raw = chat_completions(cfg, messages)

        try:
            fingerprint = parse_json_strict(raw)
        except Exception:
            fingerprint = repair_json_with_llm(cfg, raw)

        # Ensure essential fields
        fingerprint.setdefault("schema_version", "1.0.0")
        fingerprint["profile_id"] = fingerprint.get("profile_id") or args.profile_id
        fingerprint.setdefault("metadata", {})
        fingerprint["metadata"].setdefault("author", {"name": args.author_name, "is_self": True})
        fingerprint["metadata"].setdefault("extraction", {})
        # put model and date if missing
        fingerprint["metadata"].setdefault("extraction", {})
        if isinstance(fingerprint["metadata"].get("extraction"), dict):
            fingerprint["metadata"]["extraction"].setdefault("model", cfg.model)
            fingerprint["metadata"]["extraction"].setdefault("methods", ["hybrid"])
            fingerprint["metadata"]["extraction"].setdefault("confidence", "medium")

        # Always embed measurements (verbatim local measurements)
        fingerprint["measurements"] = measurements

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote fingerprint JSON to: {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
