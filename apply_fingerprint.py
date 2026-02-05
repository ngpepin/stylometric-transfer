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
import difflib
import re
import sys
import collections
import os
import datetime
import time
import random
import statistics
import math
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
LICENSE_FILENAME = "LICENSE.md"
HUMANIZER_GUIDELINES_FILENAME = "general-guidelines.md"
HUMANIZER_CACHE_FILENAME = "humanizer_rules.cache.json"
TUNABLES_FILENAME = "config.tunables.json"
AVOID_LIST_FILENAME = "config.avoid.txt"
EMOJI_SUBSTITUTIONS_FILENAME = "emoji-substitutions.json"
EM_DASH_CHAR = "—"
EMOJI_RE = re.compile(
    r"(?:[\U0001F1E6-\U0001F1FF]{2}|[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF]|\uFE0F)"
)
EMOJI_REMOVED_MARKER = "[[EMOJI_REMOVED]]"
TERMINAL_PUNCT_RE = re.compile(r"[.!?…]$")
CLOSER_CHARS = "\"'”’»)]}"
EMOJI_SUBSTITUTIONS: list[tuple[str, str]] | None = None
ANSI_RED = "\x1b[31m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RESET = "\x1b[0m"
QUOTE_MODE: str | None = None
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
    "humanizer_mandatory": {
        "avoid_em_dashes": False,
        "emoji_policy": "remove"
    },
    "section_restore": {
        "enabled": True,
        "max_restore_sections": 20,
        "heading_similarity_threshold": 0.75,
        "signature_similarity_threshold": 0.6,
        "signature_min_overlap": 6
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
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
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
QUOTE_SPAN_RE = re.compile(r"\"([^\"]+)\"", re.S)
QUOTE_SPAN_CURLY_RE = re.compile(r"“([^”]+)”", re.S)
QUOTE_PLACEHOLDER_RE = re.compile(r"\[\[QUOTE_\d+\]\]")

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


def load_emoji_substitutions(script_dir: Path) -> list[tuple[str, str]]:
    path = script_dir / EMOJI_SUBSTITUTIONS_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print_warn(f"Warning: failed to load {EMOJI_SUBSTITUTIONS_FILENAME}: {exc}")
        return []
    if not isinstance(data, dict):
        print_warn(f"Warning: {EMOJI_SUBSTITUTIONS_FILENAME} must be a JSON object.")
        return []
    out: list[tuple[str, str]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        replacement = None
        if isinstance(value, dict):
            replacement = value.get("replacement", "")
            if replacement is None:
                replacement = ""
        else:
            replacement = value if value is not None else ""
        if not isinstance(replacement, str):
            continue
        out.append((key, replacement))
    return out


def get_emoji_substitutions() -> list[tuple[str, str]]:
    global EMOJI_SUBSTITUTIONS
    if EMOJI_SUBSTITUTIONS is None:
        EMOJI_SUBSTITUTIONS = load_emoji_substitutions(Path(__file__).resolve().parent)
    return EMOJI_SUBSTITUTIONS


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


def normalize_rewrite_policy(text: str, conf: Dict[str, Any] | None = None) -> str:
    if not isinstance(text, str):
        return text
    policy = re.sub(r"\s+", " ", text.strip())
    if not policy:
        return policy
    conf = conf or {}
    verbs = conf.get("directive_verbs")
    if not isinstance(verbs, list) or not verbs:
        verbs = [
            "preserve", "avoid", "maintain", "ensure", "keep", "favor", "use",
            "prefer", "minimize", "maximize", "do not", "don't"
        ]
    verb_pattern = "|".join(re.escape(v) for v in verbs if isinstance(v, str) and v.strip())
    if not verb_pattern:
        verb_pattern = "preserve|avoid|maintain|ensure|keep|favor|use|prefer|minimize|maximize|do not|don't"
    clauses: List[str] = []
    for chunk in re.split(r"[.;:]+", policy):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(
            rf"(?i)(?=\b(?:{verb_pattern})\b)",
            chunk
        )
        for part in parts:
            part = part.strip()
            if part:
                clauses.append(part)

    stopwords_val = conf.get("stopwords")
    if isinstance(stopwords_val, list) and stopwords_val:
        stopwords = {str(w) for w in stopwords_val if isinstance(w, (str, int, float))}
    else:
        stopwords = {
            "the", "and", "of", "to", "a", "an", "in", "on", "for", "with", "or", "but",
            "as", "by", "from", "into", "at", "that", "this", "these", "those", "be", "is",
            "are", "was", "were", "been", "being"
        }
    try:
        threshold = float(conf.get("jaccard_threshold", 0.7))
    except (TypeError, ValueError):
        threshold = 0.7
    threshold = max(0.0, min(1.0, threshold))
    dedupe_on_subset = bool(conf.get("dedupe_on_subset", True))
    prefer_more_specific = bool(conf.get("prefer_more_specific", True))

    def norm_tokens(s: str) -> List[str]:
        s = s.lower()
        s = re.sub(r"[^\w\s'-]", " ", s)
        tokens = [t for t in s.split() if t and t not in stopwords]
        return tokens

    deduped: List[str] = []
    seen: List[set[str]] = []
    for clause in clauses:
        tokens = set(norm_tokens(clause))
        if not tokens:
            continue
        is_dup = False
        replace_idx: int | None = None
        for idx, prior in enumerate(seen):
            inter = tokens & prior
            if dedupe_on_subset and inter and len(inter) == len(tokens):
                is_dup = True
                break
            if dedupe_on_subset and inter and len(inter) == len(prior) and prefer_more_specific:
                replace_idx = idx
                is_dup = False
                break
            overlap = len(inter) / max(1, len(tokens | prior))
            if overlap >= threshold:
                if prefer_more_specific and len(tokens) > len(prior):
                    replace_idx = idx
                    is_dup = False
                else:
                    is_dup = True
                break
        if is_dup:
            continue
        if replace_idx is not None:
            seen[replace_idx] = tokens
            deduped[replace_idx] = clause
        else:
            seen.append(tokens)
            deduped.append(clause)

    if not deduped:
        return policy

    # Optional higher-level compaction: merge repeated preserve/avoid directives into a
    # smaller set of clauses. This keeps the policy interpretable while reducing noise.
    compress_directives = bool(conf.get("compress_directives", True))
    if compress_directives and len(clauses) > 1:
        def split_directive(clause: str) -> tuple[str | None, str]:
            c = clause.strip()
            m = re.match(
                r"(?i)^(do not|don't|preserve|avoid|maintain|ensure|keep|favor|use|prefer|minimize|maximize)\b",
                c
            )
            if not m:
                return None, c
            verb = m.group(1).lower()
            rest = c[m.end():].strip(" :-\t")
            return verb, rest

        preserve_rests: List[str] = []
        avoid_rests: List[str] = []
        other_clauses: List[str] = []

        # Use the pre-deduped clauses to avoid losing unique aspects (e.g., "structure")
        # when two preserve clauses have high lexical overlap.
        for clause in clauses:
            verb, rest = split_directive(clause)
            if verb in ("preserve", "maintain", "ensure", "keep") and rest:
                preserve_rests.append(rest)
            elif verb in ("avoid", "do not", "don't") and rest:
                avoid_rests.append(rest)

        for clause in deduped:
            verb, rest = split_directive(clause)
            if verb in ("preserve", "maintain", "ensure", "keep", "avoid", "do not", "don't") and rest:
                continue
            other_clauses.append(clause)

        def score_phrase(s: str) -> int:
            return len([t for t in norm_tokens(s) if t])

        # Prefer phrases that describe the intended aspect without dragging in other aspects.
        aspect_penalties: Dict[str, List[str]] = {
            "details": ["structure", "rhythm"],
            "structure": ["detail", "details", "rhythm"],
            "rhythm": ["detail", "details", "structure"],
        }

        def aspect_score(aspect: str, phrase: str) -> float:
            base = float(score_phrase(phrase))
            pl = phrase.lower()
            penalty = 0.0
            for kw in aspect_penalties.get(aspect, []):
                if kw in pl:
                    penalty += 2.0
            return base - penalty

        def pick_best(existing: str | None, candidate: str, aspect: str | None = None) -> str:
            if not existing:
                return candidate
            # Prefer a slightly more informative phrase, but don't balloon.
            if len(candidate) > 140:
                return existing
            if aspect:
                if aspect_score(aspect, candidate) > aspect_score(aspect, existing):
                    return candidate
            elif score_phrase(candidate) > score_phrase(existing):
                return candidate
            return existing

        preserve_aspects: Dict[str, str] = {}
        for rest in preserve_rests:
            rl = rest.lower()
            if "tone" in rl:
                preserve_aspects["tone"] = pick_best(preserve_aspects.get("tone"), rest, "tone")
            if "detail" in rl:
                preserve_aspects["details"] = pick_best(preserve_aspects.get("details"), rest, "details")
            if "accuracy" in rl:
                preserve_aspects["accuracy"] = pick_best(preserve_aspects.get("accuracy"), rest, "accuracy")
            if "clarity" in rl:
                preserve_aspects["clarity"] = pick_best(preserve_aspects.get("clarity"), rest, "clarity")
            if "realism" in rl:
                preserve_aspects["realism"] = pick_best(preserve_aspects.get("realism"), rest, "realism")
            if "structure" in rl:
                preserve_aspects["structure"] = pick_best(preserve_aspects.get("structure"), rest, "structure")
            if "rhythm" in rl:
                preserve_aspects["rhythm"] = pick_best(preserve_aspects.get("rhythm"), rest, "rhythm")

        def extract_segment(phrase: str, keyword: str) -> str:
            # For some aspects, keep only the segment that contains the keyword to avoid
            # leaking other aspects into the same phrase (e.g., "technical detail and narrative rhythm").
            pl = phrase.lower()
            keyword_hits = [keyword]
            if keyword == "details":
                keyword_hits = ["detail", "details"]
            if not any(k in pl for k in keyword_hits):
                return phrase
            if keyword in ("structure", "rhythm") or (keyword == "details" and ("structure" in pl or "rhythm" in pl)):
                parts = re.split(r"(?i)\band\b", phrase)
                for p in parts:
                    if any(k in p.lower() for k in keyword_hits):
                        return re.sub(r"\s+", " ", p.strip(" ,;"))
            return re.sub(r"\s+", " ", phrase.strip(" ,;"))

        preserve_phrases: List[str] = []
        for key in ("tone", "details", "accuracy", "clarity", "realism"):
            if key in preserve_aspects:
                preserve_phrases.append(extract_segment(preserve_aspects[key], key))

        narrative_bits: List[str] = []
        if "structure" in preserve_aspects:
            narrative_bits.append(extract_segment(preserve_aspects["structure"], "structure"))
        if "rhythm" in preserve_aspects:
            narrative_bits.append(extract_segment(preserve_aspects["rhythm"], "rhythm"))
        # Collapse "narrative structure" + "narrative rhythm" -> "narrative structure and rhythm"
        if len(narrative_bits) == 2:
            a, b = narrative_bits
            if a.lower().startswith("narrative ") and b.lower().startswith("narrative "):
                a_tail = a[len("narrative "):].strip()
                b_tail = b[len("narrative "):].strip()
                narrative_bits = [f"narrative {a_tail} and {b_tail}"]
        preserve_phrases.extend(narrative_bits)

        preserve_clause = ""
        if preserve_phrases:
            if len(preserve_phrases) == 1:
                preserve_clause = f"Preserve {preserve_phrases[0]}"
            elif len(preserve_phrases) == 2:
                preserve_clause = f"Preserve {preserve_phrases[0]} and {preserve_phrases[1]}"
            else:
                preserve_clause = "Preserve " + "; ".join(preserve_phrases[:-1]) + f"; {preserve_phrases[-1]}"

        avoid_items: List[str] = []
        dialogue_qual = False
        for rest in avoid_rests:
            rl = rest.lower()
            if "dialogue" in rl:
                dialogue_qual = True
            base = rest
            q_match = re.search(r"(?i)\b(?:except|unless)\b.*$", base)
            if q_match:
                base = base[:q_match.start()].strip(" ,;")
                if "dialogue" in q_match.group(0).lower():
                    dialogue_qual = True
            # Only split into list items when it looks like an actual list, and avoid
            # breaking verb phrases like "introducing informal ...".
            looks_like_list = ("," in base) or bool(re.search(r"(?i)\b(?:and|or)\b", base))
            starts_with_verb = bool(re.match(r"(?i)^(introduc|adding|add|insert|introduce|introducing)\b", base.strip()))
            if looks_like_list and not starts_with_verb:
                tmp = re.sub(r"(?i)\b(?:and|or)\b", ",", base)
                parts = [re.sub(r"\s+", " ", p.strip(" ,;")) for p in tmp.split(",")]
                avoid_items.extend([p for p in parts if p])
            else:
                base_clean = re.sub(r"\s+", " ", base.strip(" ,;"))
                if base_clean:
                    avoid_items.append(base_clean)

        # Dedupe avoid items (case-insensitive).
        seen_items = set()
        avoid_deduped: List[str] = []
        for item in avoid_items:
            key = item.lower()
            if key in seen_items:
                continue
            seen_items.add(key)
            avoid_deduped.append(item)

        # Drop verb-phrased avoids like "introducing informal or emotional language" when their
        # semantic content is already covered by other avoid items.
        ignore_tokens = {"introduce", "introducing", "adding", "add", "insert", "language"}
        token_sets = [set(norm_tokens(i)) - ignore_tokens for i in avoid_deduped]
        filtered_avoid: List[str] = []
        for i, item in enumerate(avoid_deduped):
            tl = item.lower().strip()
            starts_like_verb = tl.startswith(("introduc", "add", "insert"))
            if starts_like_verb and token_sets[i]:
                other = set().union(*(token_sets[j] for j in range(len(token_sets)) if j != i))
                if token_sets[i].issubset(other):
                    continue
            filtered_avoid.append(item)
        avoid_deduped = filtered_avoid

        avoid_clause = ""
        if avoid_deduped:
            avoid_clause = "Avoid " + ", ".join(avoid_deduped[:-1]) + (f", and {avoid_deduped[-1]}" if len(avoid_deduped) > 1 else avoid_deduped[0])
            if dialogue_qual:
                avoid_clause += " (unless in dialogue)"

        merged: List[str] = []
        if preserve_clause:
            merged.append(preserve_clause)
        if avoid_clause:
            merged.append(avoid_clause)
        merged.extend(other_clauses)
        deduped = merged or deduped

    cleaned = "; ".join(deduped)
    if cleaned and cleaned[-1] not in ".;:":
        cleaned += "."
    return cleaned


def normalize_priority_order(value: Any, conf: Dict[str, Any] | None = None) -> List[str]:
    conf = conf or {}
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [v.strip() for v in value.split(",")]
    else:
        raw_items = []
    token_pattern = conf.get("token_pattern", r"^[A-Za-z][A-Za-z0-9_\\-]*$")
    try:
        token_re = re.compile(str(token_pattern))
    except re.error:
        token_re = re.compile(r"^[A-Za-z][A-Za-z0-9_\\-]*$")
    dedupe_ci = bool(conf.get("dedupe_case_insensitive", True))
    exclude_tokens = conf.get("exclude_tokens")
    if isinstance(exclude_tokens, list):
        exclude = {str(item).lower() for item in exclude_tokens if isinstance(item, (str, int, float))}
    else:
        exclude = set()
    exclude.update({"lexical", "syntactic", "rhetorical"})
    items: List[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if not token:
            continue
        if not token_re.fullmatch(token):
            continue
        if token.lower() in exclude:
            continue
        items.append(token)
    deduped: List[str] = []
    seen = set()
    for item in items:
        key = item.lower() if dedupe_ci else item
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _baseline_quantile(stats: Dict[str, Any], q: float) -> float | None:
    if not isinstance(stats, dict):
        return None
    points = []
    for key, pct in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)):
        val = stats.get(key)
        if isinstance(val, (int, float)):
            points.append((pct, float(val)))
    if not points:
        mean = stats.get("mean")
        return float(mean) if isinstance(mean, (int, float)) else None
    points.sort(key=lambda x: x[0])
    if q <= points[0][0]:
        return points[0][1]
    if q >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= q <= x1:
            if x1 == x0:
                return y0
            t = (q - x0) / (x1 - x0)
            return y0 * (1.0 - t) + y1 * t
    return points[-1][1]


def _clamp_range(low: float, high: float, lo: float | None = None, hi: float | None = None) -> tuple[float, float]:
    if lo is not None:
        low = max(lo, low)
        high = max(lo, high)
    if hi is not None:
        low = min(hi, low)
        high = min(hi, high)
    if high < low:
        low, high = high, low
    return low, high


def build_controller_overlay(
    fingerprint: Dict[str, Any],
    tunables: Dict[str, Any] | None,
    chunk_index: int | None,
    chunk_text: str
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not isinstance(fingerprint, dict):
        return None, None
    baseline = fingerprint.get("measurements", {}).get("humanization_baseline")
    if not isinstance(baseline, dict) or not baseline.get("enabled", False):
        return None, None
    conf = {}
    if isinstance(tunables, dict):
        conf = tunables.get("humanization_controller", {}) if isinstance(tunables.get("humanization_controller", {}), dict) else {}
    if not conf or not conf.get("enabled", False):
        return None, None

    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return None, None

    allowed = conf.get("allowed_metrics")
    if not isinstance(allowed, list) or not allowed:
        allowed = [
            "sentence_length_mean",
            "sentence_length_stdev",
            "one_sentence_paragraph_rate",
            "comma_density_per_100w",
            "punctuation_semicolons_per_1000w",
            "punctuation_colons_per_1000w",
            "punctuation_em_dashes_per_1000w"
        ]

    quantiles = conf.get("quantiles")
    if not isinstance(quantiles, list) or not quantiles:
        quantiles = [0.25, 0.5, 0.75]
    quantiles = [float(q) for q in quantiles if isinstance(q, (int, float)) and 0 <= float(q) <= 1]
    if not quantiles:
        quantiles = [0.5]

    range_pct = float(conf.get("range_pct", 0.15))
    min_width = float(conf.get("min_width", 0.05))
    max_width = float(conf.get("max_width", 6.0))

    seed = int(conf.get("seed", 0))
    if chunk_index is not None:
        seed += int(chunk_index)
    else:
        seed += abs(hash(chunk_text)) % 100000
    rng = random.Random(seed)

    overlay_metrics: Dict[str, Any] = {}

    for name in allowed:
        stats = metrics.get(name)
        if not isinstance(stats, dict):
            continue
        q = rng.choice(quantiles)
        value = _baseline_quantile(stats, q)
        if value is None:
            continue
        width = max(min_width, abs(value) * range_pct)
        if max_width > 0:
            width = min(width, max_width)
        low = float(value - width)
        high = float(value + width)
        # Clamp ratios to [0,1]
        if name in ("one_sentence_paragraph_rate",):
            low, high = _clamp_range(low, high, 0.0, 1.0)
        overlay_metrics[name] = {
            "value": float(value),
            "target": [low, high],
            "quantile": q
        }

    if not overlay_metrics:
        return None, None

    # Apply to a fingerprint copy so the LLM sees chunk-specific targets.
    fp_copy = copy.deepcopy(fingerprint)
    targets = fp_copy.setdefault("targets", {})
    if not isinstance(targets, dict):
        targets = {}
        fp_copy["targets"] = targets

    def set_target(path: List[str], target: List[float]) -> None:
        node = targets
        for key in path[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        leaf = node.setdefault(path[-1], {})
        if isinstance(leaf, dict):
            leaf["target"] = target

    for name, spec in overlay_metrics.items():
        target = spec.get("target")
        if not isinstance(target, list) or len(target) != 2:
            continue
        if name == "sentence_length_mean":
            set_target(["sentence", "length_words_mean"], target)
        elif name == "sentence_length_stdev":
            set_target(["sentence", "length_words_stdev"], target)
        elif name == "one_sentence_paragraph_rate":
            set_target(["paragraph", "one_sentence_paragraph_rate"], target)
        elif name == "comma_density_per_100w":
            set_target(["punctuation", "comma_density_per_100w"], target)
        elif name == "punctuation_semicolons_per_1000w":
            set_target(["punctuation", "semicolons_per_1000w"], target)
        elif name == "punctuation_colons_per_1000w":
            set_target(["punctuation", "colons_per_1000w"], target)
        elif name == "punctuation_em_dashes_per_1000w":
            set_target(["punctuation", "em_dashes_per_1000w"], target)

    overlay = {
        "source": "humanization_baseline",
        "metrics": overlay_metrics
    }
    return fp_copy, overlay


def compute_overlay_observations(text: str) -> Dict[str, float]:
    sents = split_sentences(text)
    sent_lens = [len(words(s)) for s in sents] if sents else []
    sent_mean = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
    sent_stdev = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    paras = split_paragraphs(text)
    para_lens = [len(split_sentences(p)) for p in paras] if paras else []
    one_sentence_rate = sum(1 for n in para_lens if n == 1) / max(1, len(para_lens)) if para_lens else 0.0
    word_list = words(text)
    total_words = max(1, len(word_list))
    punct_counts = {
        "commas": text.count(","),
        "semicolons": text.count(";"),
        "colons": text.count(":"),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "em_dashes": text.count("—")
    }
    comma_density_per_100w = punct_counts["commas"] / total_words * 100.0
    per_1000w = {k: v / total_words * 1000.0 for k, v in punct_counts.items()}
    return {
        "sentence_length_mean": float(sent_mean),
        "sentence_length_stdev": float(sent_stdev),
        "one_sentence_paragraph_rate": float(one_sentence_rate),
        "comma_density_per_100w": float(comma_density_per_100w),
        "punctuation_commas_per_1000w": float(per_1000w["commas"]),
        "punctuation_semicolons_per_1000w": float(per_1000w["semicolons"]),
        "punctuation_colons_per_1000w": float(per_1000w["colons"]),
        "punctuation_exclamations_per_1000w": float(per_1000w["exclamations"]),
        "punctuation_questions_per_1000w": float(per_1000w["questions"]),
        "punctuation_em_dashes_per_1000w": float(per_1000w["em_dashes"])
    }


def build_overlay_feedback(
    overlay: Dict[str, Any],
    output_text: str,
    conf: Dict[str, Any] | None
) -> Dict[str, Any] | None:
    if not overlay or not isinstance(overlay, dict):
        return None
    metrics = overlay.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    conf = conf or {}
    tolerance = float(conf.get("feedback_tolerance", 0.35))
    observations = compute_overlay_observations(output_text)
    deltas: List[Dict[str, Any]] = []
    for name, spec in metrics.items():
        if not isinstance(spec, dict):
            continue
        target = spec.get("target")
        if not isinstance(target, list) or len(target) != 2:
            continue
        actual = observations.get(name)
        if not isinstance(actual, (int, float)):
            continue
        low, high = float(target[0]), float(target[1])
        if low <= actual <= high:
            continue
        span = max(1e-6, high - low)
        if actual < low and (low - actual) / span < tolerance:
            continue
        if actual > high and (actual - high) / span < tolerance:
            continue
        direction = "increase" if actual < low else "decrease"
        deltas.append({
            "metric": name,
            "target": [low, high],
            "actual": float(actual),
            "direction": direction
        })
    if not deltas:
        return None
    return {
        "notes": "Chunk-level targets vary by design to match within-author variability.",
        "deltas": deltas
    }


def compute_variance_aware_factor(
    fingerprint: Dict[str, Any],
    tunables: Dict[str, Any] | None
) -> float:
    if not isinstance(tunables, dict):
        return 1.0
    chunk_conf = tunables.get("chunking", {}) if isinstance(tunables.get("chunking", {}), dict) else {}
    var_conf = chunk_conf.get("variance_aware", {}) if isinstance(chunk_conf.get("variance_aware", {}), dict) else {}
    if not var_conf or not var_conf.get("enabled", False):
        return 1.0
    baseline = fingerprint.get("measurements", {}).get("humanization_baseline")
    if not isinstance(baseline, dict) or not baseline.get("enabled", False):
        return 1.0
    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict):
        return 1.0
    stdev_ref = float(var_conf.get("sentence_stdev_ref", 18.0))
    burst_ref = float(var_conf.get("paragraph_burst_ref", 0.7))
    min_factor = float(var_conf.get("min_factor", 0.6))
    max_factor = float(var_conf.get("max_factor", 1.0))
    min_factor = max(0.1, min_factor)
    max_factor = max(min_factor, max_factor)

    sent_stdev = metrics.get("sentence_length_stdev", {})
    para_burst = metrics.get("paragraph_burstiness", {})
    sent_val = sent_stdev.get("mean") if isinstance(sent_stdev, dict) else None
    para_val = para_burst.get("mean") if isinstance(para_burst, dict) else None
    scores: List[float] = []
    if isinstance(sent_val, (int, float)) and stdev_ref > 0:
        scores.append(min(2.0, float(sent_val) / stdev_ref))
    if isinstance(para_val, (int, float)) and burst_ref > 0:
        scores.append(min(2.0, float(para_val) / burst_ref))
    if not scores:
        return 1.0
    score = sum(scores) / len(scores)
    if score <= 1.0:
        return max_factor
    # Map score (1..2) to factor (max_factor..min_factor).
    t = min(1.0, score - 1.0)
    return max(min_factor, max_factor - (max_factor - min_factor) * t)


def should_forbid_em_dashes(tunables: Dict[str, Any] | None) -> bool:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    return bool(mandatory.get("avoid_em_dashes", False))


def enforce_no_em_dashes(text: str) -> tuple[str, int]:
    # Replace em dashes with a spaced hyphen to preserve readability without em-dash glyphs.
    if EM_DASH_CHAR not in text:
        return text, 0
    count = text.count(EM_DASH_CHAR)
    text = re.sub(r"\s*—\s*", " - ", text)
    return text, count


def is_heading_or_list_line(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith(">"):
        return True
    if re.match(r"([-*+]|\\d+[.)])\\s+", stripped):
        return True
    return False


def apply_removed_emoji_punctuation(text: str) -> str:
    marker = EMOJI_REMOVED_MARKER
    if marker not in text:
        return text
    parts = re.split(r"(\n\s*\n+)", text)
    updated: list[str] = []
    for part in parts:
        if not part or part.isspace():
            updated.append(part)
            continue
        if part.startswith("\n"):
            updated.append(part)
            continue
        if marker not in part:
            updated.append(part)
            continue
        first_line = part.lstrip().splitlines()[0] if part.strip() else ""
        if is_heading_or_list_line(first_line):
            updated.append(part.replace(marker, ""))
            continue
        trimmed = part.rstrip()
        trailing_ws = part[len(trimmed):]
        idx = len(trimmed)
        while idx > 0 and trimmed[idx - 1] in CLOSER_CHARS:
            idx -= 1
        closers = trimmed[idx:]
        base = trimmed[:idx]
        removed_at_end = False
        while base.endswith(marker):
            removed_at_end = True
            base = base[: -len(marker)]
        if removed_at_end and base.strip():
            if not TERMINAL_PUNCT_RE.search(base):
                base = base + "."
            rebuilt = base + closers + trailing_ws
        else:
            rebuilt = trimmed + trailing_ws
        updated.append(rebuilt.replace(marker, ""))
    return "".join(updated)


def enforce_emoji_policy(text: str, policy: str) -> tuple[str, int, int]:
    # Remove or replace emoji glyphs with conventional monochrome symbols.
    removed = 0
    replaced = 0
    if policy not in ("remove", "replace", "none"):
        return text, removed, replaced
    if policy == "none":
        return text, removed, replaced
    substitutions = get_emoji_substitutions()

    if policy == "replace" and substitutions:
        for emoji, replacement in substitutions:
            if not emoji:
                continue
            count = text.count(emoji)
            if count == 0:
                continue
            if replacement:
                replaced += count
                text = text.replace(emoji, replacement)
                continue
            else:
                removed += count
                text = text.replace(emoji, EMOJI_REMOVED_MARKER)

    if policy == "remove":
        removed += len(EMOJI_RE.findall(text))
        text = EMOJI_RE.sub(EMOJI_REMOVED_MARKER, text)
    else:
        leftover = len(EMOJI_RE.findall(text))
        if leftover:
            removed += leftover
            text = EMOJI_RE.sub(EMOJI_REMOVED_MARKER, text)

    text = apply_removed_emoji_punctuation(text)
    return text, removed, replaced


def apply_humanizer_variance(
    text: str,
    seed: int,
    max_ops_per_1000w: float,
    allowed_ops: List[str]
) -> tuple[str, List[Dict[str, Any]]]:
    # Apply small, bounded stochastic edits to reduce AI-typical uniformity.
    ops_applied: List[Dict[str, Any]] = []
    if not allowed_ops or max_ops_per_1000w <= 0:
        return text, ops_applied
    total_words = len(words(text))
    max_ops = max(0, int((total_words / 1000.0) * max_ops_per_1000w))
    if max_ops == 0:
        return text, ops_applied
    rng = random.Random(seed)

    # Small, meaning-preserving substitutions.
    transition_variants = {
        "however": ["yet", "nevertheless", "nonetheless"],
        "therefore": ["thus", "hence"],
        "moreover": ["furthermore"],
        "for example": ["for instance"],
        "in sum": ["overall", "in short"]
    }
    filler_terms = {"very", "really", "quite", "rather"}

    def replace_transition(text_in: str, budget: int) -> tuple[str, int]:
        if budget <= 0:
            return text_in, 0
        count = 0
        for term, variants in transition_variants.items():
            if count >= budget:
                break
            pattern = re.compile(rf"\\b{re.escape(term)}\\b", re.IGNORECASE)
            matches = list(pattern.finditer(text_in))
            rng.shuffle(matches)
            for m in matches[: max(0, budget - count)]:
                replacement = rng.choice(variants)
                text_in = text_in[:m.start()] + replacement + text_in[m.end():]
                count += 1
        return text_in, count

    def drop_fillers(text_in: str, budget: int) -> tuple[str, int]:
        if budget <= 0:
            return text_in, 0
        count = 0
        pattern = re.compile(rf"\\b({'|'.join(sorted(filler_terms))})\\b", re.IGNORECASE)
        matches = list(pattern.finditer(text_in))
        rng.shuffle(matches)
        for m in matches[: max(0, budget - count)]:
            text_in = text_in[:m.start()] + "" + text_in[m.end():]
            count += 1
        # Clean up extra spaces
        text_in = re.sub(r"\\s{2,}", " ", text_in)
        return text_in, count

    remaining = max_ops
    if "swap_transition" in allowed_ops and remaining > 0:
        text, applied = replace_transition(text, remaining)
        if applied:
            ops_applied.append({"op": "swap_transition", "count": applied})
        remaining -= applied
    if "drop_filler" in allowed_ops and remaining > 0:
        text, applied = drop_fillers(text, remaining)
        if applied:
            ops_applied.append({"op": "drop_filler", "count": applied})
        remaining -= applied

    return text, ops_applied

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


def resolve_license_path() -> Path | None:
    # Resolve LICENSE.md from CWD or script directory.
    cwd_path = Path.cwd() / LICENSE_FILENAME
    script_path = Path(__file__).resolve().parent / LICENSE_FILENAME
    if cwd_path.exists():
        return cwd_path
    if script_path.exists():
        return script_path
    return None


def render_markdown(text: str) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except Exception:
        print(text)
        return
    console = Console()
    console.print(Markdown(text))


def print_license_and_exit() -> int:
    path = resolve_license_path()
    if not path:
        print_error(f"License file not found: {LICENSE_FILENAME}")
        return 2
    render_markdown(path.read_text(encoding="utf-8"))
    return 0

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


META_SUMMARY_PATTERNS = [
    "as an ai",
    "as a language model",
    "i have rewritten",
    "i rewrote",
    "i have rewritten",
    "i have re-written",
    "i have summarized",
    "i summarised",
    "this summary",
    "this chunk",
    "the chunk",
    "the prompt",
    "the task",
    "rewrite",
    "rewritten",
    "re-write",
    "output",
    "final_markdown",
    "instructions",
    "followed",
    "preserving meaning",
    "narrative voice",
]


def summary_is_meta(text: str) -> bool:
    lowered = text.lower()
    return any(pat in lowered for pat in META_SUMMARY_PATTERNS)


def normalize_summary(summary: str, max_words: int | None) -> str:
    if not isinstance(summary, str):
        return ""
    summary = " ".join(summary.split()).strip()
    if not summary:
        return ""
    # Deterministic cleanup: avoid "this passage/section/..." phrasing for prior context.
    context_nouns = (
        "passage", "section", "chunk", "text", "document", "paper", "article",
        "report", "excerpt", "chapter", "part", "segment", "portion", "appendix",
        "material", "content", "discussion", "analysis", "overview", "summary"
    )
    for noun in context_nouns:
        summary = re.sub(
            rf"\bthis\s+{noun}\b",
            f"the previous {noun}",
            summary,
            flags=re.IGNORECASE
        )
        summary = re.sub(
            rf"\bthe\s+{noun}\b",
            f"the previous {noun}",
            summary,
            flags=re.IGNORECASE
        )
    summary = re.sub(r"\bin\s+this\s+summary\b", "in the previous summary", summary, flags=re.IGNORECASE)
    # Prefer past tense when referring to prior content.
    verb_map = {
        "introduces": "introduced",
        "outlines": "outlined",
        "discusses": "discussed",
        "explains": "explained",
        "details": "detailed",
        "presents": "presented",
        "describes": "described",
        "summarizes": "summarized",
        "summarises": "summarised",
        "argues": "argued",
        "emphasizes": "emphasized",
        "emphasises": "emphasised",
        "highlights": "highlighted",
        "notes": "noted",
        "states": "stated",
        "claims": "claimed",
        "defines": "defined",
        "frames": "framed",
        "motivates": "motivated",
        "proposes": "proposed",
        "recommends": "recommended",
        "compares": "compared",
        "contrasts": "contrasted",
        "evaluates": "evaluated",
        "assesses": "assessed",
        "examines": "examined",
        "explores": "explored",
        "covers": "covered",
        "lists": "listed",
        "catalogs": "catalogued",
        "catalogues": "catalogued",
        "illustrates": "illustrated",
        "demonstrates": "demonstrated",
        "shows": "showed",
        "mentions": "mentioned",
        "addresses": "addressed",
        "focuses": "focused",
        "considers": "considered",
        "identifies": "identified",
        "summons": "summoned",
        "calls out": "called out",
        "calls": "called"
    }
    verb_pattern = "|".join(re.escape(k) for k in verb_map.keys())
    summary = re.sub(
        rf"\b(the previous (?:{'|'.join(context_nouns)}))\s+({verb_pattern})\b",
        lambda m: f"{m.group(1)} {verb_map.get(m.group(2).lower(), m.group(2))}",
        summary,
        flags=re.IGNORECASE
    )
    summary = re.sub(r"\bis discussed\b", "was discussed", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare discussed\b", "were discussed", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis outlined\b", "was outlined", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare outlined\b", "were outlined", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis described\b", "was described", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare described\b", "were described", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis presented\b", "was presented", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare presented\b", "were presented", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis detailed\b", "was detailed", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare detailed\b", "were detailed", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis explained\b", "was explained", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare explained\b", "were explained", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis compared\b", "was compared", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare compared\b", "were compared", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis contrasted\b", "was contrasted", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare contrasted\b", "were contrasted", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bis evaluated\b", "was evaluated", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bare evaluated\b", "were evaluated", summary, flags=re.IGNORECASE)
    if isinstance(max_words, int) and max_words > 0:
        tokens = summary.split()
        if len(tokens) > max_words:
            summary = " ".join(tokens[:max_words])
    return summary


def build_fallback_summary(text: str, max_words: int | None) -> str:
    cleaned = filter_author_voice_text(text)
    tokens = words(cleaned)
    if not tokens:
        tokens = words(text)
    if not tokens:
        return ""
    if isinstance(max_words, int) and max_words > 0:
        tokens = tokens[:max_words]
    return " ".join(tokens)


def build_semantic_fallback_summary(text: str, max_words: int | None) -> str:
    cleaned = filter_author_voice_text(text)
    sents = [s.strip() for s in split_sentences(cleaned) if s.strip()]
    if not sents:
        return build_fallback_summary(text, max_words)
    stopwords = {
        "the", "and", "or", "but", "as", "if", "when", "than", "then",
        "a", "an", "of", "to", "in", "on", "for", "with", "at", "by",
        "from", "into", "over", "under", "between", "about", "after",
        "before", "during", "through", "without", "within", "is", "are",
        "was", "were", "be", "been", "being", "it", "this", "that",
        "these", "those", "he", "she", "they", "we", "you", "i", "me",
        "my", "our", "your", "their", "his", "her", "its", "not", "no",
        "so", "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "may", "might", "must", "should"
    }
    freq: Dict[str, int] = {}
    for tok in words(cleaned.lower()):
        if len(tok) < 4 or tok in stopwords:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    if not freq:
        return build_fallback_summary(text, max_words)
    scored: List[tuple[int, int, str]] = []
    for idx, sent in enumerate(sents):
        score = 0
        for tok in words(sent.lower()):
            if tok in freq:
                score += freq[tok]
        scored.append((score, idx, sent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:2]
    top_sorted = [s for _score, _idx, s in sorted(top, key=lambda x: x[1])]
    summary = " ".join(top_sorted)
    summary = normalize_summary(summary, max_words)
    if not summary:
        return build_fallback_summary(text, max_words)
    return summary


def _quote_spans(text: str) -> List[tuple[int, int, str]]:
    spans: List[tuple[int, int, str]] = []
    for match in QUOTE_SPAN_RE.finditer(text):
        spans.append((match.start(), match.end(), match.group(1)))
    for match in QUOTE_SPAN_CURLY_RE.finditer(text):
        spans.append((match.start(), match.end(), match.group(1)))
    spans.sort(key=lambda s: s[0])
    return spans


def is_multiword_quote(inner: str) -> bool:
    return len(words(inner)) >= 2


def detect_fiction_from_text(
    text: str,
    quote_span_min: int,
    quoted_ratio_min: float,
    quote_para_ratio_min: float,
    quoted_ratio_force: float
) -> bool:
    total_words = len(words(text))
    quoted_words = 0
    quote_spans = 0
    quote_para = 0
    total_para = 0
    for _start, _end, inner in _quote_spans(text):
        if is_multiword_quote(inner):
            quote_spans += 1
            quoted_words += len(words(inner))
    for para in split_paragraphs(text):
        if not para.strip():
            continue
        total_para += 1
        if para.lstrip().startswith(("\"", "“")):
            quote_para += 1
    quoted_ratio = quoted_words / max(1, total_words)
    quote_para_ratio = quote_para / max(1, total_para)
    if quote_spans >= quote_span_min and quoted_ratio >= quoted_ratio_min:
        return True
    if quote_para_ratio >= quote_para_ratio_min and quoted_ratio >= quoted_ratio_min:
        return True
    if quoted_ratio >= quoted_ratio_force:
        return True
    return False

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


def strip_quoted_passages(text: str) -> str:
    # Remove multi-word quoted passages (used for non-fiction measurements).
    spans = _quote_spans(text)
    if not spans:
        return text
    out: List[str] = []
    last = 0
    for start, end, inner in spans:
        if start < last:
            continue
        out.append(text[last:start])
        if not is_multiword_quote(inner):
            out.append(text[start:end])
        last = end
    out.append(text[last:])
    return "".join(out)


def filter_author_voice_text(text: str) -> str:
    # Remove non-author voice segments and inline citations for measurements.
    text = strip_fenced_code_blocks(text)
    text = strip_inline_code(text)
    text = strip_non_voice_sections(text)
    if QUOTE_MODE == "non-fiction":
        text = strip_quoted_passages(text)
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


def mask_quoted_passages(text: str) -> tuple[str, Dict[str, str]]:
    # Replace multi-word quoted passages with placeholders to preserve them verbatim.
    mapping: Dict[str, str] = {}
    counter = 0
    spans = _quote_spans(text)
    if not spans:
        return text, mapping
    out: List[str] = []
    last = 0
    for start, end, inner in spans:
        if start < last:
            continue
        out.append(text[last:start])
        if is_multiword_quote(inner):
            placeholder = f"[[QUOTE_{counter}]]"
            mapping[placeholder] = text[start:end]
            counter += 1
            out.append(placeholder)
        else:
            out.append(text[start:end])
        last = end
    out.append(text[last:])
    return "".join(out), mapping


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
    lowered = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", lowered)
    lowered = lowered.replace(EMOJI_REMOVED_MARKER.lower(), " ")
    lowered = EMOJI_RE.sub(" ", lowered)
    lowered = lowered.replace("`", " ")
    lowered = re.sub(r"[*_~]", " ", lowered)
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


def heading_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


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
    raw, _ = chat_completions(cfg, messages)
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
    emoji_policy = None
    if isinstance(tunables, dict):
        mandatory = tunables.get("humanizer_mandatory", {})
        if isinstance(mandatory, dict):
            em_dash_forbidden = bool(mandatory.get("avoid_em_dashes", False))
            emoji_policy = mandatory.get("emoji_policy")

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
                drop_reason = "Em dashes forbidden by humanizer_mandatory."
            elif em_dash_rate >= em_dash_keep_rate:
                drop_reason = "Author uses em dashes frequently."
        if ("emoji" in title or "emoji" in words) and emoji_policy in ("remove", "replace"):
            drop_reason = "Emoji handling is deterministic via humanizer_mandatory."
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


def estimate_tokens_for_text(text: str) -> int:
    # Rough token estimate for plain text.
    return estimate_tokens(text)


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


def split_words_preserve(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def split_sentences_for_chunking(text: str) -> List[str]:
    lines = text.splitlines()
    sentences: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if LIST_LINE_RE.match(line):
            if buffer:
                sentences.extend(split_sentences(" ".join(buffer)))
                buffer = []
            sentences.append(line.strip())
        else:
            if line.strip():
                buffer.append(line.strip())
    if buffer:
        sentences.extend(split_sentences(" ".join(buffer)))
    return [s for s in sentences if s.strip()]


def split_block_units(block: str, split_on: str, max_chars: int, max_input_tokens: int) -> tuple[List[str], str]:
    if is_code_block(block):
        if estimate_tokens_for_text(block) <= max_input_tokens:
            return [block], "\n\n"
        return split_oversize_block(block, estimate_tokens_for_text, max_input_tokens), "\n\n"

    if LIST_LINE_RE.search(block):
        units = [ln.strip() for ln in block.splitlines() if ln.strip()]
        refined: List[str] = []
        for unit in units:
            if len(unit) > max_chars:
                refined.extend(split_words_preserve(unit))
            else:
                refined.append(unit)
        return refined, "\n"

    mode = split_on if split_on in ("paragraph", "sentence", "word") else "sentence"
    if mode == "paragraph":
        if len(block) <= max_chars:
            return [block], "\n\n"
        # Fallback to sentence mode for oversized paragraphs.
        mode = "sentence"

    if mode == "sentence":
        sentences = split_sentences_for_chunking(block)
        units: List[str] = []
        for sent in sentences:
            if len(sent) > max_chars:
                units.extend(split_words_preserve(sent))
            else:
                units.append(sent)
        return units, " "

    # Word-level splitting (last resort)
    return split_words_preserve(block), " "


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


def split_oversize_block(
    block: str,
    estimate_tokens_fn,
    max_input_tokens: int,
    depth: int = 0,
    max_depth: int = 50
) -> List[str]:
    # Recursively split blocks that exceed token limits, respecting code fences.
    if estimate_tokens_fn(block) <= max_input_tokens:
        return [block]
    if depth >= max_depth:
        # Failsafe: avoid infinite recursion; return the block as-is.
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
            if estimate_tokens_fn(candidate) > max_input_tokens and len(current) > 1:
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
    left = block[:split_idx].rstrip()
    right = block[split_idx:].lstrip()
    if not left or not right:
        split_idx = max(1, mid)
        left = block[:split_idx].rstrip()
        right = block[split_idx:].lstrip()
    if left == block or right == block:
        split_idx = max(1, mid)
        left = block[:split_idx].rstrip()
        right = block[split_idx:].lstrip()
    if not left and not right:
        return [block]
    chunks: List[str] = []
    if left:
        chunks.extend(split_oversize_block(left, estimate_tokens_fn, max_input_tokens, depth + 1, max_depth))
    if right:
        chunks.extend(split_oversize_block(right, estimate_tokens_fn, max_input_tokens, depth + 1, max_depth))
    return chunks


def enforce_min_chunks(markdown: str, chunks: List[str], min_chunks: int) -> List[str]:
    if min_chunks <= 1:
        return chunks
    work = [c for c in chunks if isinstance(c, str) and c.strip()]
    if len(work) >= min_chunks:
        return work
    if not work:
        work = [markdown]
    # Split the largest chunk until we reach the minimum or can no longer split.
    while len(work) < min_chunks:
        idx = max(range(len(work)), key=lambda i: len(work[i]))
        block = work.pop(idx)
        if not block.strip():
            work.insert(idx, block)
            break
        max_tokens = max(1, estimate_tokens_for_text(block) // 2)
        parts = split_oversize_block(block, estimate_tokens_for_text, max_tokens)
        if len(parts) <= 1:
            mid = len(block) // 2
            split_idx = block.rfind("\n", 0, mid)
            if split_idx == -1:
                split_idx = block.find("\n", mid)
            if split_idx == -1:
                split_idx = mid
            left = block[:split_idx].rstrip()
            right = block[split_idx:].lstrip()
            parts = [p for p in (left, right) if p]
        if len(parts) <= 1:
            work.insert(idx, block)
            break
        for part in reversed(parts):
            if part.strip():
                work.insert(idx, part)
    return [c for c in work if c.strip()]


def chunk_markdown(
    markdown: str,
    build_messages_fn,
    max_prompt_tokens: int,
    max_input_tokens_override: int | None = None,
    split_on: str = "sentence"
) -> List[str]:
    # Chunk by paragraph with a fixed character budget derived from prompt overhead.
    # build_messages_fn signature is expected to be (md_chunk, style_feedback, for_estimate).
    base_messages = build_messages_fn("", None, True)
    base_tokens = estimate_tokens_for_messages(base_messages)
    max_input_tokens = max(400, max_prompt_tokens - base_tokens)
    if isinstance(max_input_tokens_override, int) and max_input_tokens_override > 0:
        max_input_tokens = max(200, min(max_input_tokens, max_input_tokens_override))
    max_chars = max_input_tokens * 4

    blocks = split_markdown_blocks(markdown)
    chunks: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        if not block.strip():
            continue
        units, joiner = split_block_units(block.strip(), split_on, max_chars, max_input_tokens)
        if not units:
            continue
        for idx, unit in enumerate(units):
            sep = ""
            if current:
                sep = "\n\n" if idx == 0 else joiner
            candidate = current + sep + unit if current else unit
            if len(candidate) > max_chars and current:
                flush()
                candidate = unit
            if len(candidate) > max_chars and not current:
                # Failsafe: force-split oversized unit at word level.
                fallback_units = split_words_preserve(unit)
                for word in fallback_units:
                    sep2 = "" if not current else " "
                    cand2 = current + sep2 + word if current else word
                    if len(cand2) > max_chars and current:
                        flush()
                        cand2 = word
                    current = cand2
                continue
            current = candidate
    flush()
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
    transition_start_hits = 0
    transition_mid_hits = 0
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
        # Discourse marker position (start vs mid-sentence)
        if ws:
            start_candidate = None
            if len(ws) >= 3:
                start_candidate = " ".join(ws[:3])
            elif len(ws) >= 2:
                start_candidate = " ".join(ws[:2])
            else:
                start_candidate = ws[0]
            if start_candidate in transition_terms:
                transition_start_hits += 1
            else:
                for term in transition_terms:
                    if term in " ".join(ws[1:]):
                        transition_mid_hits += 1
                        break

    # Rhetorical move signals (simple, interpretable heuristics).
    rhetoric_markers = {
        "claim": [
            "we argue", "we contend", "we propose", "this suggests", "this shows",
            "this indicates", "therefore", "thus", "hence", "overall", "in sum"
        ],
        "evidence": [
            "for example", "for instance", "according to", "data show", "evidence",
            "study", "report", "survey", "as shown"
        ],
        "counterpoint": [
            "however", "yet", "but", "on the other hand", "nevertheless", "nonetheless"
        ],
        "concession": [
            "although", "though", "even though", "while", "granted", "admittedly"
        ],
        "synthesis": [
            "overall", "in sum", "in short", "on balance", "taken together", "in conclusion"
        ]
    }

    def sentence_has_marker(sentence: str, markers: List[str]) -> bool:
        s = sentence.lower()
        return any(m in s for m in markers)

    all_sents = split_sentences(text)
    claim_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["claim"]))
    evidence_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["evidence"]))
    counter_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["counterpoint"]))
    concession_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["concession"]))
    synthesis_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["synthesis"]))

    # Paragraph cadence profile.
    opening_lens: List[int] = []
    closing_lens: List[int] = []
    for p in paras:
        sents = split_sentences(p)
        if not sents:
            continue
        opening_lens.append(len(words(sents[0])))
        closing_lens.append(len(words(sents[-1])))

    def safe_mean(xs: List[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def safe_stdev(xs: List[int]) -> float:
        if len(xs) < 2:
            return 0.0
        mean = safe_mean(xs)
        return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5

    # Epistemic stance bands (simple token markers).
    speculative_terms = {"may","might","perhaps","possibly","could","seems","appears","suggests","tends"}
    probabilistic_terms = {"likely","unlikely","probable","probably","odds","chance"}
    assertive_terms = {"clearly","certainly","undoubtedly","indeed","surely"}
    directive_terms = {"must","should","need","needs","ought","required"}

    speculative_hits = sum(1 for t in toks if t in speculative_terms)
    probabilistic_hits = sum(1 for t in toks if t in probabilistic_terms)
    assertive_hits = sum(1 for t in toks if t in assertive_terms)
    directive_hits = sum(1 for t in toks if t in directive_terms)

    # Syntax texture (lightweight approximations).
    subordinator_terms = {
        "because","although","though","while","if","when","since","unless","whereas","after","before","once","until"
    }
    subordinator_hits = sum(1 for t in toks if t in subordinator_terms)
    parenthetical_hits = text.count("(") + text.count(")")
    appositive_hits = len(re.findall(r",\s+(?:a|an|the|which|who|that)\b", text.lower()))

    # Lexical avoidance categories (rarely-used / stylistic no-go zones).
    avoidance_categories = {
        "intensifiers": {"very","really","extremely","highly","incredibly","quite","so"},
        "emotional_adjectives": {"happy","sad","angry","afraid","anxious","excited","terrible","wonderful","awful","lovely"},
        "informal_slang": {"cool","awesome","yeah","ok","okay","stuff","gonna","wanna","kinda","sorta"}
    }
    avoidance_rates = {
        name: approx_rate_per_1000_words(sum(1 for t in toks if t in terms), total_words)
        for name, terms in avoidance_categories.items()
    }

    # Self-echo repetition rates (bigrams/trigrams reused above a threshold).
    def repeat_rate(ngram_list: List[str], min_count: int = 3) -> float:
        if not ngram_list:
            return 0.0
        counts = collections.Counter(ngram_list)
        repeat_tokens = sum(c for _, c in counts.items() if c >= min_count)
        return repeat_tokens / max(1, len(ngram_list))

    def ngrams_all(n: int) -> List[str]:
        toks_local = [t.lower() for t in w]
        out: List[str] = []
        for i in range(0, len(toks_local) - n + 1):
            chunk = toks_local[i:i+n]
            if sum(1 for x in chunk if x in {"the","a","an","and","or","but","if","then","to","of","in","on","for","with","as"}) >= n - 1:
                continue
            out.append(" ".join(chunk))
        return out

    bigrams_all = ngrams_all(2)
    trigrams_all = ngrams_all(3)

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
            "transition_openers_top": [{"phrase": p, "count": c} for p, c in transition_hits.most_common(15)],
            "transition_marker_positions": {
                "start_rate_per_1000w": approx_rate_per_1000_words(transition_start_hits, total_words),
                "mid_rate_per_1000w": approx_rate_per_1000_words(transition_mid_hits, total_words)
            }
        },
        "rhetoric_moves": {
            "claim_rate": approx_rate_per_1000_words(claim_hits, total_words),
            "evidence_rate": approx_rate_per_1000_words(evidence_hits, total_words),
            "counterpoint_rate": approx_rate_per_1000_words(counter_hits, total_words),
            "concession_rate": approx_rate_per_1000_words(concession_hits, total_words),
            "synthesis_rate": approx_rate_per_1000_words(synthesis_hits, total_words),
            "claim_evidence_ratio": claim_hits / max(1, evidence_hits)
        },
        "paragraph_cadence": {
            "opening_sentence_length_mean": safe_mean(opening_lens),
            "opening_sentence_length_stdev": safe_stdev(opening_lens),
            "closing_sentence_length_mean": safe_mean(closing_lens),
            "closing_sentence_length_stdev": safe_stdev(closing_lens)
        },
        "epistemic_profile": {
            "speculative_rate": approx_rate_per_1000_words(speculative_hits, total_words),
            "probabilistic_rate": approx_rate_per_1000_words(probabilistic_hits, total_words),
            "assertive_rate": approx_rate_per_1000_words(assertive_hits, total_words),
            "directive_rate": approx_rate_per_1000_words(directive_hits, total_words)
        },
        "syntax_texture": {
            "subordinate_clause_rate": approx_rate_per_1000_words(subordinator_hits, total_words),
            "parenthetical_rate": approx_rate_per_1000_words(parenthetical_hits, total_words),
            "appositive_rate": approx_rate_per_1000_words(appositive_hits, total_words)
        },
        "lexical_avoidance": {
            "category_rates_per_1000w": avoidance_rates
        },
        "repetition": {
            "bigram_repeat_rate": repeat_rate(bigrams_all),
            "trigram_repeat_rate": repeat_rate(trigrams_all),
            "min_repeat_count": 3
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
    validators = fingerprint.get("validators", {}) if isinstance(fingerprint, dict) else {}
    weights = validators.get("weights", {}) if isinstance(validators, dict) else {}
    out_meas = compute_measurements(output_text)
    deltas: List[Dict[str, Any]] = []

    section_scores: Dict[str, List[float]] = {}

    def add_score(section: str, value: float) -> None:
        section_scores.setdefault(section, []).append(value)

    # Sentence length histogram
    fp_sent = fp_meas.get("sentence", {}).get("length_words", {}).get("histogram_p")
    out_sent = out_meas.get("sentence", {}).get("length_words", {}).get("histogram_p")
    if isinstance(fp_sent, list) and isinstance(out_sent, list) and len(fp_sent) == len(out_sent):
        diff = l1_distance(fp_sent, out_sent) / 2.0
        add_score("sentence", max(0.0, 1.0 - min(1.0, diff)))
        if diff > 0.15:
            deltas.append({"metric": "sentence_length_histogram", "diff": diff})

    # Paragraph length histogram
    fp_para = fp_meas.get("paragraph", {}).get("length_sentences_histogram_p")
    out_para = out_meas.get("paragraph", {}).get("length_sentences_histogram_p")
    if isinstance(fp_para, list) and isinstance(out_para, list) and len(fp_para) == len(out_para):
        diff = l1_distance(fp_para, out_para) / 2.0
        add_score("paragraph", max(0.0, 1.0 - min(1.0, diff)))
        if diff > 0.15:
            deltas.append({"metric": "paragraph_length_histogram", "diff": diff})

    # One-sentence paragraph rate
    fp_one = fp_meas.get("paragraph", {}).get("one_sentence_paragraph_rate")
    out_one = out_meas.get("paragraph", {}).get("one_sentence_paragraph_rate")
    if isinstance(fp_one, (int, float)) and isinstance(out_one, (int, float)):
        diff = abs(fp_one - out_one)
        add_score("paragraph", max(0.0, 1.0 - min(1.0, diff)))
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
            add_score("punctuation", max(0.0, 1.0 - min(1.0, avg)))

    # Contractions / Oxford comma signals
    fp_ortho = fp_meas.get("orthography_signals", {})
    out_ortho = out_meas.get("orthography_signals", {})
    for key in ("contractions_rate", "oxford_comma_signal"):
        if isinstance(fp_ortho.get(key), (int, float)) and isinstance(out_ortho.get(key), (int, float)):
            diff = relative_diff(float(out_ortho[key]), float(fp_ortho[key]))
            add_score("orthography", max(0.0, 1.0 - min(1.0, diff)))
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
            add_score("stance", max(0.0, 1.0 - min(1.0, avg)))

    # Rhetoric moves
    fp_rhet = fp_meas.get("rhetoric_moves", {})
    out_rhet = out_meas.get("rhetoric_moves", {})
    if isinstance(fp_rhet, dict) and isinstance(out_rhet, dict) and fp_rhet:
        diffs = []
        for k, target in fp_rhet.items():
            if k in out_rhet and isinstance(target, (int, float)):
                diff = relative_diff(float(out_rhet.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"rhetoric_moves.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("rhetoric_moves", max(0.0, 1.0 - min(1.0, avg)))

    # Epistemic profile
    fp_epi = fp_meas.get("epistemic_profile", {})
    out_epi = out_meas.get("epistemic_profile", {})
    if isinstance(fp_epi, dict) and isinstance(out_epi, dict) and fp_epi:
        diffs = []
        for k, target in fp_epi.items():
            if k in out_epi and isinstance(target, (int, float)):
                diff = relative_diff(float(out_epi.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"epistemic_profile.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("epistemic_profile", max(0.0, 1.0 - min(1.0, avg)))

    # Paragraph cadence
    fp_cad = fp_meas.get("paragraph_cadence", {})
    out_cad = out_meas.get("paragraph_cadence", {})
    if isinstance(fp_cad, dict) and isinstance(out_cad, dict) and fp_cad:
        diffs = []
        for k, target in fp_cad.items():
            if k in out_cad and isinstance(target, (int, float)):
                diff = relative_diff(float(out_cad.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"paragraph_cadence.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("paragraph_cadence", max(0.0, 1.0 - min(1.0, avg)))

    # Syntax texture
    fp_tex = fp_meas.get("syntax_texture", {})
    out_tex = out_meas.get("syntax_texture", {})
    if isinstance(fp_tex, dict) and isinstance(out_tex, dict) and fp_tex:
        diffs = []
        for k, target in fp_tex.items():
            if k in out_tex and isinstance(target, (int, float)):
                diff = relative_diff(float(out_tex.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"syntax_texture.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("syntax_texture", max(0.0, 1.0 - min(1.0, avg)))

    # Lexical avoidance category rates
    fp_avoid = fp_meas.get("lexical_avoidance", {}).get("category_rates_per_1000w", {})
    out_avoid = out_meas.get("lexical_avoidance", {}).get("category_rates_per_1000w", {})
    if isinstance(fp_avoid, dict) and isinstance(out_avoid, dict) and fp_avoid:
        diffs = []
        for k, target in fp_avoid.items():
            if k in out_avoid and isinstance(target, (int, float)):
                diff = relative_diff(float(out_avoid.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"lexical_avoidance.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("lexical_avoidance", max(0.0, 1.0 - min(1.0, avg)))

    # Discourse marker position rates
    fp_disc = fp_meas.get("templates_signals", {}).get("transition_marker_positions", {})
    out_disc = out_meas.get("templates_signals", {}).get("transition_marker_positions", {})
    if isinstance(fp_disc, dict) and isinstance(out_disc, dict) and fp_disc:
        diffs = []
        for k, target in fp_disc.items():
            if k in out_disc and isinstance(target, (int, float)):
                diff = relative_diff(float(out_disc.get(k, 0.0)), float(target))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"discourse_markers.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("discourse_markers", max(0.0, 1.0 - min(1.0, avg)))

    # Self-echo repetition rates
    fp_rep = fp_meas.get("repetition", {})
    out_rep = out_meas.get("repetition", {})
    if isinstance(fp_rep, dict) and isinstance(out_rep, dict) and fp_rep:
        diffs = []
        for k in ("bigram_repeat_rate", "trigram_repeat_rate"):
            if k in out_rep and isinstance(fp_rep.get(k), (int, float)):
                diff = relative_diff(float(out_rep.get(k, 0.0)), float(fp_rep.get(k, 0.0)))
                diffs.append(diff)
                if diff > 0.6:
                    deltas.append({"metric": f"repetition.{k}", "diff": diff})
        if diffs:
            avg = sum(diffs) / len(diffs)
            add_score("repetition", max(0.0, 1.0 - min(1.0, avg)))

    # Aggregate section scores with optional weighting.
    section_avgs: Dict[str, float] = {}
    for section, vals in section_scores.items():
        if vals:
            section_avgs[section] = sum(vals) / len(vals)

    if isinstance(weights, dict) and section_avgs:
        total_weight = 0.0
        weighted_sum = 0.0
        for section, score_val in section_avgs.items():
            w = weights.get(section, 1.0)
            if isinstance(w, (int, float)):
                w = float(w)
            else:
                w = 1.0
            if w <= 0:
                continue
            weighted_sum += score_val * w
            total_weight += w
        score = weighted_sum / total_weight if total_weight > 0 else 1.0
    else:
        score = sum(section_avgs.values()) / len(section_avgs) if section_avgs else 1.0
    return {
        "score": score,
        "deltas": deltas,
        "output_measurements": out_meas
    }


def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def _js_divergence(p: List[float], q: List[float]) -> float:
    if not p or not q or len(p) != len(q):
        return 0.0
    eps = 1e-12
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log((pi + eps) / (mi + eps), 2) for pi, mi in zip(p, m))
    kl_qm = sum(qi * math.log((qi + eps) / (mi + eps), 2) for qi, mi in zip(q, m))
    return (kl_pm + kl_qm) / 2.0


def compute_humanization_metrics(text: str, fingerprint: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Heuristic, research-inspired quantitative signals (lexical diversity, repetition, burstiness).
    text = filter_author_voice_text(text)
    meas = compute_measurements(text)

    tokens = [w.lower() for w in words(text)]
    total_words = len(tokens)
    unique_words = len(set(tokens))
    ttr = (unique_words / total_words) if total_words else 0.0
    if total_words > 1 and unique_words > 0:
        herdan_c = math.log(unique_words) / math.log(total_words)
        guiraud_r = unique_words / math.sqrt(total_words)
        maas_ttr = (math.log(total_words) - math.log(unique_words)) / (math.log(total_words) ** 2)
    else:
        herdan_c = 0.0
        guiraud_r = 0.0
        maas_ttr = 0.0

    freq = collections.Counter(tokens)
    m1 = total_words
    m2 = sum(v * v for v in freq.values())
    if m1 > 0:
        yules_k = (10000.0 * (m2 - m1) / (m1 * m1)) if m1 > 0 else 0.0
        simpson_d = sum(v * (v - 1) for v in freq.values()) / (m1 * (m1 - 1)) if m1 > 1 else 0.0
    else:
        yules_k = 0.0
        simpson_d = 0.0

    sent_lens = [len(words(s)) for s in split_sentences(text) if words(s)]
    sent_mean = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
    sent_stdev = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    sent_burstiness = (sent_stdev / sent_mean) if sent_mean > 0 else 0.0

    paras = []
    for block in split_markdown_blocks(text):
        if is_code_block(block):
            continue
        if not block.strip():
            continue
        paras.append(block)
    para_lens = [len(split_sentences(p)) for p in paras] if paras else []
    para_mean = (sum(para_lens) / len(para_lens)) if para_lens else 0.0
    para_stdev = statistics.pstdev(para_lens) if len(para_lens) > 1 else 0.0
    para_burstiness = (para_stdev / para_mean) if para_mean > 0 else 0.0

    repetition = meas.get("repetition", {}) if isinstance(meas, dict) else {}
    bigram_repeat = float(repetition.get("bigram_repeat_rate", 0.0)) if isinstance(repetition, dict) else 0.0
    trigram_repeat = float(repetition.get("trigram_repeat_rate", 0.0)) if isinstance(repetition, dict) else 0.0
    repeat_rate = (bigram_repeat + trigram_repeat) / 2.0

    punct = meas.get("punctuation", {}).get("rates_per_1000w", {}) if isinstance(meas, dict) else {}
    punctuation_variety = sum(1 for v in punct.values() if isinstance(v, (int, float)) and v > 0)
    punct_counts = {k: int(v * max(1, total_words) / 1000.0) for k, v in punct.items() if isinstance(v, (int, float))}
    punctuation_entropy = _entropy(punct_counts)

    function_entropy = 0.0
    function_kl = None
    out_func = meas.get("function_words", {}).get("rates_per_1000w", {}) if isinstance(meas, dict) else {}
    if isinstance(out_func, dict) and out_func:
        out_counts = {k: max(0, int(v * max(1, total_words) / 1000.0)) for k, v in out_func.items() if isinstance(v, (int, float))}
        function_entropy = _entropy(out_counts)
        if isinstance(fingerprint, dict):
            fp_func = fingerprint.get("measurements", {}).get("function_words", {}).get("rates_per_1000w", {})
            if isinstance(fp_func, dict) and fp_func:
                keys = sorted(set(out_func.keys()) | set(fp_func.keys()))
                p = []
                q = []
                for k in keys:
                    p.append(max(0.0, float(out_func.get(k, 0.0))) + 1e-9)
                    q.append(max(0.0, float(fp_func.get(k, 0.0))) + 1e-9)
                psum = sum(p)
                qsum = sum(q)
                p = [x / psum for x in p]
                q = [x / qsum for x in q]
                function_kl = sum(pi * math.log(pi / qi, 2) for pi, qi in zip(p, q))

    sentence_js = None
    if isinstance(fingerprint, dict):
        fp_hist = fingerprint.get("measurements", {}).get("sentence", {}).get("length_words", {}).get("histogram_p")
        out_hist = meas.get("sentence", {}).get("length_words", {}).get("histogram_p") if isinstance(meas, dict) else None
        if isinstance(fp_hist, list) and isinstance(out_hist, list) and len(fp_hist) == len(out_hist):
            sentence_js = _js_divergence(fp_hist, out_hist)

    letters_only = re.sub(r"[^a-zA-Z]+", "", text.lower())
    n = 3
    trigram_counts: Dict[str, int] = {}
    if len(letters_only) >= n:
        trigram_counts = collections.Counter(letters_only[i:i+n] for i in range(len(letters_only) - n + 1))
    char_trigram_entropy = _entropy(trigram_counts)

    word_lens = [len(w) for w in tokens if w]
    avg_word_len = (sum(word_lens) / len(word_lens)) if word_lens else 0.0

    scores = {
        "lexical_diversity": min(1.0, ttr / 0.5) if ttr > 0 else 0.0,
        "herdan_c": min(1.0, herdan_c / 0.9) if herdan_c > 0 else 0.0,
        "guiraud_r": min(1.0, guiraud_r / 15.0) if guiraud_r > 0 else 0.0,
        "maas_ttr_inverse": max(0.0, 1.0 - min(1.0, maas_ttr / 0.15)) if maas_ttr > 0 else 0.0,
        "yules_k_inverse": max(0.0, 1.0 - min(1.0, yules_k / 100.0)) if yules_k > 0 else 0.0,
        "simpson_d_inverse": max(0.0, 1.0 - min(1.0, simpson_d / 0.2)) if simpson_d > 0 else 0.0,
        "repetition_inverse": max(0.0, 1.0 - min(1.0, repeat_rate / 0.2)),
        "sentence_burstiness": min(1.0, sent_burstiness / 1.0) if sent_burstiness > 0 else 0.0,
        "paragraph_burstiness": min(1.0, para_burstiness / 1.0) if para_burstiness > 0 else 0.0,
        "punctuation_variety": min(1.0, punctuation_variety / 8.0) if punctuation_variety > 0 else 0.0,
        "punctuation_entropy": min(1.0, punctuation_entropy / 3.0) if punctuation_entropy > 0 else 0.0,
        "function_word_entropy": min(1.0, function_entropy / 4.0) if function_entropy > 0 else 0.0,
        "function_word_kl_inverse": max(0.0, 1.0 - min(1.0, (function_kl or 0.0) / 0.5)) if function_kl is not None else 0.0,
        "sentence_length_js_inverse": max(0.0, 1.0 - min(1.0, (sentence_js or 0.0) / 0.3)) if sentence_js is not None else 0.0,
        "char_trigram_entropy": min(1.0, char_trigram_entropy / 6.0) if char_trigram_entropy > 0 else 0.0,
        "avg_word_length": max(0.0, 1.0 - min(1.0, abs(avg_word_len - 5.0) / 3.0)) if avg_word_len > 0 else 0.0
    }

    return {
        "token_count": total_words,
        "type_token_ratio": ttr,
        "herdan_c": herdan_c,
        "guiraud_r": guiraud_r,
        "maas_ttr": maas_ttr,
        "yules_k": yules_k,
        "simpson_d": simpson_d,
        "sentence_length_mean": sent_mean,
        "sentence_length_stdev": sent_stdev,
        "sentence_burstiness": sent_burstiness,
        "paragraph_length_mean": para_mean,
        "paragraph_length_stdev": para_stdev,
        "paragraph_burstiness": para_burstiness,
        "bigram_repeat_rate": bigram_repeat,
        "trigram_repeat_rate": trigram_repeat,
        "punctuation_variety": punctuation_variety,
        "punctuation_entropy": punctuation_entropy,
        "function_word_entropy": function_entropy,
        "function_word_kl": function_kl,
        "sentence_length_js": sentence_js,
        "char_trigram_entropy": char_trigram_entropy,
        "avg_word_length": avg_word_len,
        "scores": scores,
        "notes": [
            "Scores are heuristic and intended for comparative inspection, not absolute judgment."
        ]
    }


def compute_humanization_aggregate(
    scores: Dict[str, float],
    weights: Dict[str, float] | None = None
) -> Dict[str, Any]:
    if not scores:
        return {"aggregate_score_100": 0.0, "weights": {}, "weighted_scores": {}}
    weights = weights or {}
    weighted_scores: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for key, val in scores.items():
        w = weights.get(key, 1.0)
        if not isinstance(w, (int, float)):
            w = 1.0
        if w <= 0:
            continue
        weighted_scores[key] = val
        weighted_sum += float(val) * float(w)
        total_weight += float(w)
    aggregate = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    return {
        "aggregate_score_100": round(aggregate * 100.0, 2),
        "weights": weights,
        "weighted_scores": weighted_scores
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
        timeout_seconds=int(data.get("timeout_seconds", 300)),
        extra_headers=dict(data.get("extra_headers", {})),
        max_prompt_tokens=int(data.get("max_prompt_tokens", max_tokens)),
        max_retries=int(data.get("max_retries", 6)),
        backoff_base_seconds=float(data.get("backoff_base_seconds", 2.0)),
        backoff_max_seconds=float(data.get("backoff_max_seconds", 20.0))
    )

def chat_completions(cfg: LLMConfig, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any] | None]:
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
            content = data["choices"][0]["message"]["content"]
            if attempt > 0:
                print_warn(f"LLM request succeeded after {attempt} retry(ies).")
            return content, data.get("usage")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, RuntimeError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
            jitter = random.uniform(0, backoff * 0.2)
            sleep_s = backoff + jitter
            print_warn(
                "LLM request failed "
                f"(attempt {attempt + 1}/{cfg.max_retries + 1}); "
                f"retrying in {sleep_s:.1f}s. Error: {exc}"
            )
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
    fixed, _ = chat_completions(cfg, messages)
    return parse_json_strict(fixed)


# ---- Prompting: apply fingerprint ----

def build_apply_prompt(
    fingerprint: Dict[str, Any],
    input_md: str,
    input_meas: Dict[str, Any],
    cfg: LLMConfig,
    prompts: Dict[str, Any],
    style_feedback: Dict[str, Any] | None = None,
    humanizer_rules: List[Dict[str, Any]] | None = None,
    controller_overlay: Dict[str, Any] | None = None,
    voice_override: str | None = None,
    chunk_summary: Dict[str, Any] | None = None
) -> List[Dict[str, str]]:
    # Fill the apply prompt template with runtime data.
    system = get_prompt_value(prompts, "apply", "system")
    user_template = get_prompt_value(prompts, "apply", "user")
    if not isinstance(user_template, dict):
        raise TypeError("prompts.apply.user must be an object")
    user = copy.deepcopy(user_template)
    fp_payload = fingerprint
    if isinstance(fingerprint, dict):
        fp_payload = copy.deepcopy(fingerprint)
        meta = fp_payload.get("metadata")
        if isinstance(meta, dict):
            author = meta.get("author")
            if isinstance(author, dict):
                meta = {"author": author}
            else:
                meta = {}
            fp_payload["metadata"] = meta if meta else None
            if fp_payload.get("metadata") is None:
                fp_payload.pop("metadata", None)
        # Strip controller/baseline blocks that are useful for local logic but not for the LLM.
        meas = fp_payload.get("measurements")
        if isinstance(meas, dict):
            meas.pop("humanization_baseline", None)
    user["style_fingerprint_json"] = fp_payload
    user["input_measurements"] = input_meas
    user["input_markdown"] = input_md
    if style_feedback:
        user["style_feedback"] = style_feedback
    if humanizer_rules:
        user["humanizer_guidelines"] = humanizer_rules
    if controller_overlay:
        user["controller_overlay"] = controller_overlay
    if voice_override:
        override_rules = {
            "first": "Force narrative voice to first-person (I/we). Avoid second- and third-person pronouns unless they appear in preserved quotations or frozen blocks.",
            "second": "Force narrative voice to second-person (you/your). Avoid first- and third-person pronouns unless they appear in preserved quotations or frozen blocks.",
            "third": "Force narrative voice to third-person (he/she/they). Avoid first- and second-person pronouns unless they appear in preserved quotations or frozen blocks."
        }
        rule = override_rules.get(voice_override)
        if isinstance(user.get("rules"), list):
            if rule:
                user["rules"].append(rule)
        elif rule:
            user["rules"] = [rule]
        user["voice_override"] = voice_override
    if chunk_summary:
        user["chunk_summary"] = chunk_summary
        if isinstance(user.get("output_format"), dict):
            user["output_format"]["chunk_summary"] = "string"
        if isinstance(user.get("rules"), list):
            summary_words = chunk_summary.get("summary_words")
            summary_clause = f"~{summary_words} words" if summary_words else "a short (approx. 25-word) summary"
            user["rules"].append(
                "Return chunk_summary as plain text (no Markdown). It is ONLY for "
                "context continuity between chunks and must NOT appear in final_markdown. "
                f"Keep it to {summary_clause}. Synthesize the prior summary with the current "
                "chunk in your own words (do not copy previous_summary verbatim). "
                "Describe the content (events, topics, claims) rather than describing the task. "
                "Treat previous_summary as referring to the previous passage; do not refer to it as "
                "\"this passage\" or \"this section\" unless the current chunk is explicitly the subject. "
                "Do not mention rewriting, the prompt, or the model. "
                "Do not introduce new facts."
            )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def apply_pronoun_override(fingerprint: Dict[str, Any], mode: str | None) -> None:
    if not mode:
        return
    if not isinstance(fingerprint, dict):
        return
    targets = fingerprint.setdefault("targets", {})
    if not isinstance(targets, dict):
        targets = {}
        fingerprint["targets"] = targets
    persona = targets.setdefault("persona", {})
    if not isinstance(persona, dict):
        persona = {}
        targets["persona"] = persona

    pronoun_prefs: Dict[str, Any] = {}
    if mode == "first":
        pronoun_prefs = {
            "default_set": "I/me/my",
            "allowed_sets": ["I/me/my", "we/us/our"],
            "avoid_sets": ["you/your", "they/them", "he/him", "she/her"],
            "strictness": "hard",
            "notes": "Override: force first-person voice."
        }
    elif mode == "second":
        pronoun_prefs = {
            "default_set": "you/your",
            "allowed_sets": ["you/your"],
            "avoid_sets": ["I/me/my", "we/us/our", "they/them", "he/him", "she/her"],
            "strictness": "hard",
            "notes": "Override: force second-person voice."
        }
    elif mode == "third":
        pronoun_prefs = {
            "default_set": "they/them",
            "allowed_sets": ["they/them", "he/him", "she/her"],
            "avoid_sets": ["I/me/my", "we/us/our", "you/your"],
            "strictness": "hard",
            "notes": "Override: force third-person voice."
        }

    if pronoun_prefs:
        persona["pronoun_preferences"] = pronoun_prefs


def evaluate_pronoun_override(text: str, mode: str) -> Dict[str, Any]:
    tokens = [t.lower() for t in words(text)]
    first_person = {"i","me","my","mine","we","us","our","ours"}
    second_person = {"you","your","yours"}
    third_person = {"he","him","his","she","her","hers","they","them","their","theirs"}

    def count_hits(word_set: set[str]) -> int:
        return sum(1 for t in tokens if t in word_set)

    allowed: set[str]
    avoid_sets: Dict[str, set[str]]
    mode_label = mode
    if mode == "first":
        allowed = first_person
        avoid_sets = {"second_person": second_person, "third_person": third_person}
    elif mode == "second":
        allowed = second_person
        avoid_sets = {"first_person": first_person, "third_person": third_person}
    else:
        mode_label = "third"
        allowed = third_person
        avoid_sets = {"first_person": first_person, "second_person": second_person}

    allowed_count = count_hits(allowed)
    violations: Dict[str, int] = {}
    for label, words_set in avoid_sets.items():
        count = count_hits(words_set)
        if count:
            violations[label] = count

    return {
        "mode": mode_label,
        "allowed_count": allowed_count,
        "violations": violations
    }


def main() -> int:
    if "--license" in sys.argv:
        return print_license_and_exit()
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
    pronoun_group = ap.add_mutually_exclusive_group()
    pronoun_group.add_argument(
        "--1st-person",
        dest="force_person",
        action="store_const",
        const="first",
        help="Override the fingerprint to force first-person voice."
    )
    pronoun_group.add_argument(
        "--2nd-person",
        dest="force_person",
        action="store_const",
        const="second",
        help="Override the fingerprint to force second-person voice."
    )
    pronoun_group.add_argument(
        "--3rd-person",
        dest="force_person",
        action="store_const",
        const="third",
        help="Override the fingerprint to force third-person voice."
    )
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fiction",
        action="store_true",
        help="Treat input as fiction (quoted passages may be rewritten)."
    )
    mode_group.add_argument(
        "--non-fiction",
        dest="non_fiction",
        action="store_true",
        help="Treat input as non-fiction (multi-word quotes preserved)."
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Compute and report humanization metrics for both input and output"
    )
    ap.add_argument(
        "--license",
        action="store_true",
        help="Print LICENSE.md and exit"
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

    # print("Use --license to view licensing terms")
    print(f"Applying fingerprint {args.fingerprint.name} to {args.inp}")

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
    if isinstance(tunables, dict):
        style_retry = tunables.get("style_retry", {})
        if isinstance(style_retry, dict):
            if args.no_style_retry is False:
                enabled = style_retry.get("enabled")
                if isinstance(enabled, bool) and not enabled:
                    args.no_style_retry = True
            if args.style_retry_threshold == 0.75:
                threshold = style_retry.get("threshold")
                if isinstance(threshold, (int, float)):
                    args.style_retry_threshold = float(threshold)
            if args.max_style_retries == 1:
                max_retries = style_retry.get("max_retries")
                if isinstance(max_retries, int):
                    args.max_style_retries = max(0, max_retries)

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
    controls = fingerprint.get("controls")
    if isinstance(controls, dict) and isinstance(controls.get("rewrite_policy"), str):
        controls_norm = tunables.get("controls_normalization", {}) if isinstance(tunables, dict) else {}
        rewrite_conf = controls_norm.get("rewrite_policy", {}) if isinstance(controls_norm, dict) else {}
        controls["rewrite_policy"] = normalize_rewrite_policy(controls["rewrite_policy"], rewrite_conf)
    if isinstance(controls, dict) and controls.get("priority_order") is not None:
        controls_norm = tunables.get("controls_normalization", {}) if isinstance(tunables, dict) else {}
        priority_conf = controls_norm.get("priority_order", {}) if isinstance(controls_norm, dict) else {}
        controls["priority_order"] = normalize_priority_order(controls.get("priority_order"), priority_conf)
    apply_pronoun_override(fingerprint, args.force_person)
    forbid_em_dashes = should_forbid_em_dashes(tunables)
    emoji_policy = None
    if isinstance(tunables, dict):
        mandatory = tunables.get("humanizer_mandatory", {})
        if isinstance(mandatory, dict):
            emoji_policy = mandatory.get("emoji_policy")
    input_md = args.inp.read_text(encoding="utf-8")
    original_input_md = input_md
    fiction_conf = {}
    if isinstance(tunables, dict):
        fiction_conf = tunables.get("fiction_detection", {}) or {}
    quote_span_min = int(fiction_conf.get("quote_span_min", 6))
    quoted_ratio_min = float(fiction_conf.get("quoted_ratio_min", 0.03))
    quote_para_ratio_min = float(fiction_conf.get("quote_para_ratio_min", 0.2))
    quoted_ratio_force = float(fiction_conf.get("quoted_ratio_force", 0.08))
    if args.fiction:
        fiction_mode = True
        print("Assuming fiction: quoted passages may be rewritten.")
    elif args.non_fiction:
        fiction_mode = False
        print("Assuming non-fiction: multi-word quotations will be preserved.")
    else:
        fiction_mode = detect_fiction_from_text(
            original_input_md,
            quote_span_min=quote_span_min,
            quoted_ratio_min=quoted_ratio_min,
            quote_para_ratio_min=quote_para_ratio_min,
            quoted_ratio_force=quoted_ratio_force
        )
        if fiction_mode:
            print("Detected fiction: quoted passages may be rewritten.")
        else:
            print("Detected non-fiction: multi-word quotations will be preserved.")
    global QUOTE_MODE
    QUOTE_MODE = "fiction" if fiction_mode else "non-fiction"
    # Strip base64 images to keep prompts within token limits.
    input_md, base64_map = strip_base64_images(input_md)
    if base64_map:
        vprint(f"Stripped {len(base64_map)} base64 image embed(s) from prompt.")
    # Mask HTML, math, entities, and inline code spans so they are preserved verbatim.
    input_md, html_map = mask_html(input_md)
    input_md, math_map = mask_math_notation(input_md)
    input_md, entity_map = mask_html_entities(input_md)
    input_md, inline_code_map = mask_inline_code(input_md)
    quote_map: Dict[str, str] = {}
    if not fiction_mode:
        input_md, quote_map = mask_quoted_passages(input_md)
    if not args.no_humanizer_guidelines:
        raw_guidelines, guidelines_path = load_general_guidelines()
        if raw_guidelines:
            if forbid_em_dashes:
                vprint("Hard constraint active: em dashes are forbidden (humanizer_mandatory).")
            if emoji_policy and str(emoji_policy) != "none":
                vprint(f"Hard constraint active: emoji policy = {emoji_policy}.")
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
                        clean_reason = reason.strip().rstrip(".")
                        drop_labels.append(f"{title} — {clean_reason}")
                    else:
                        drop_labels.append(str(title))
                preview = "; ".join(drop_labels[:10])
                suffix = "..." if len(drop_labels) > 10 else ""
                vprint(f"Dropped {len(drop_labels)} humanizer rule(s): {preview}{suffix}")
            if args.verbose:
                print(f"Humanizer rules loaded: {len(humanizer_rules)} kept, {len(dropped_rules)} dropped")
    # Mask non-voice blocks and inline citations so they are preserved verbatim.
    input_md, frozen_blocks = mask_non_voice_blocks(input_md)
    input_md, citation_map = mask_inline_citations(input_md)
    section_blocks = extract_heading_blocks(input_md)
    section_blocks_restored: List[Dict[str, Any]] = []
    for block in section_blocks:
        restored = block["block"]
        restored = restore_placeholders(restored, frozen_blocks)
        restored = restore_placeholders(restored, html_map)
        restored = restore_placeholders(restored, math_map)
        restored = restore_placeholders(restored, entity_map)
        restored = restore_placeholders(restored, inline_code_map)
        restored = restore_placeholders(restored, quote_map)
        restored = restore_placeholders(restored, citation_map)
        restored = restore_base64_images(restored, base64_map, find_base64_placeholders(restored))
        section_blocks_restored.append({
            **block,
            "block": restored,
            "signature": section_signature(restored)
        })

    all_deviations: List[Any] = []
    outputs: List[str] = []

    def build_messages_for_chunk(
        md_chunk: str,
        style_feedback: Dict[str, Any] | None = None,
        for_estimate: bool = False,
        fingerprint_override: Dict[str, Any] | None = None,
        controller_overlay: Dict[str, Any] | None = None,
        previous_summary: str | None = None,
        summary_words: int | None = None,
        summary_enabled: bool = False
    ) -> List[Dict[str, str]]:
        # Build prompts per chunk using local measurements.
        if for_estimate:
            word_count = len(words(md_chunk))
            input_meas = {"totals": {"total_words_est": word_count}}
        else:
            input_meas = compute_measurements(filter_author_voice_text(md_chunk))
        fp_payload = fingerprint_override if isinstance(fingerprint_override, dict) else fingerprint
        summary_payload = None
        if summary_enabled:
            summary_payload = {
                "enabled": True,
                "summary_words": summary_words if isinstance(summary_words, int) else None,
                "previous_summary": previous_summary or ""
            }
        return build_apply_prompt(
            fp_payload,
            md_chunk,
            input_meas,
            cfg,
            prompts,
            style_feedback,
            humanizer_rules,
            controller_overlay,
            args.force_person,
            summary_payload
        )

    def rewrite_chunk(
        md_chunk: str,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        depth: int = 0,
        previous_summary: str | None = None,
        summary_words: int | None = None,
        summary_enabled: bool = False
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        # Rewrite a chunk with optional style retry.
        author_voice = filter_author_voice_text(md_chunk)
        if not author_voice.strip():
            if args.verbose and chunk_index is not None and chunk_total is not None:
                vprint(f"Chunk {chunk_index}/{chunk_total} has no author-voice content; skipping LLM.")
            out_obj = {
                "final_markdown": md_chunk,
                "deviations": [
                    {
                        "rule_or_field": "skip_llm",
                        "reason": "Chunk contains no author-voice content; preserved verbatim."
                    }
                ],
                "self_check": {
                    "notes": ["No author-voice content in chunk; preserved verbatim."]
                }
            }
            if summary_enabled:
                out_obj["chunk_summary"] = (previous_summary or "")
            return md_chunk, out_obj, {"score": 1.0, "deltas": []}
        attempts = 0
        fp_overlay = None
        controller_overlay = None
        if isinstance(tunables, dict):
            fp_overlay, controller_overlay = build_controller_overlay(
                fingerprint,
                tunables,
                chunk_index,
                md_chunk
            )
        if controller_overlay and args.verbose:
            vprint(f"Controller overlay for chunk {chunk_index}/{chunk_total}: {controller_overlay}")
        style_feedback: Dict[str, Any] | None = None
        last_out: Dict[str, Any] = {}
        while True:
            messages = build_messages_for_chunk(
                md_chunk,
                style_feedback,
                False,
                fp_overlay,
                controller_overlay,
                previous_summary,
                summary_words,
                summary_enabled
            )
            input_tokens = estimate_tokens_for_messages(messages)
            last_raw = ""
            last_usage: Dict[str, Any] | None = None
            last_err: Exception | None = None
            out_obj: Dict[str, Any] | None = None
            for attempt in range(cfg.max_retries + 1):
                try:
                    raw, usage = chat_completions(cfg, messages)
                    last_raw = raw
                    last_usage = usage
                    try:
                        out_obj = parse_json_strict(raw)
                    except Exception:
                        vprint("Invalid JSON returned; attempting repair...")
                        out_obj = repair_json_with_llm(cfg, raw, prompts)
                    final_md = out_obj.get("final_markdown") if isinstance(out_obj, dict) else None
                    if isinstance(final_md, str) and final_md.strip():
                        if attempt > 0:
                            print_warn(f"LLM output recovered after {attempt} retry(ies).")
                        break
                    last_err = RuntimeError("LLM did not return final_markdown")
                except Exception as exc:
                    last_err = exc
                if attempt >= cfg.max_retries:
                    break
                backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
                jitter = random.uniform(0, backoff * 0.2)
                sleep_s = backoff + jitter
                print_warn(
                    "LLM output invalid "
                    f"(attempt {attempt + 1}/{cfg.max_retries + 1}); "
                    f"retrying in {sleep_s:.1f}s. Error: {last_err}"
                )
                time.sleep(sleep_s)
            if (
                not isinstance(out_obj, dict)
                or not isinstance(out_obj.get("final_markdown"), str)
                or not out_obj.get("final_markdown", "").strip()
            ):
                print_error("LLM did not return final_markdown.")
                if last_raw and args.verbose:
                    vprint("Last LLM raw output (truncated):")
                    print(last_raw[:2000])

                # Recovery: split and retry with smaller pieces rather than aborting the entire run.
                # This guards against occasional model failures where it "claims" the input is empty.
                chunk_conf = tunables.get("chunking", {}) if isinstance(tunables, dict) else {}
                recovery_max_depth = 2
                recovery_min_chars = 800
                if isinstance(chunk_conf, dict):
                    try:
                        recovery_max_depth = int(chunk_conf.get("recovery_split_max_depth", recovery_max_depth))
                    except (TypeError, ValueError):
                        recovery_max_depth = recovery_max_depth
                    try:
                        recovery_min_chars = int(chunk_conf.get("recovery_split_min_chars", recovery_min_chars))
                    except (TypeError, ValueError):
                        recovery_min_chars = recovery_min_chars
                recovery_max_depth = max(0, recovery_max_depth)
                recovery_min_chars = max(0, recovery_min_chars)

                def split_for_recovery(text: str) -> List[str]:
                    blocks = split_markdown_blocks(text)
                    if len(blocks) >= 2:
                        mid = max(1, len(blocks) // 2)
                        left = "\n\n".join(blocks[:mid]).strip()
                        right = "\n\n".join(blocks[mid:]).strip()
                        if left and right:
                            return [left, right]
                    # Fallback: split on nearest newline.
                    mid = max(1, len(text) // 2)
                    split_idx = text.rfind("\n", 0, mid)
                    if split_idx == -1:
                        split_idx = text.find("\n", mid)
                    if split_idx == -1:
                        split_idx = mid
                    left = text[:split_idx].strip()
                    right = text[split_idx:].strip()
                    if left and right:
                        return [left, right]
                    return [text]

                if depth < recovery_max_depth and len(md_chunk) >= recovery_min_chars:
                    parts = split_for_recovery(md_chunk)
                    if len(parts) > 1:
                        label = None
                        if chunk_index is not None and chunk_total is not None:
                            label = f"{chunk_index}/{chunk_total}"
                        msg = f"Chunk {label} rewrite failed; splitting into {len(parts)} part(s) for recovery." if label else f"Chunk rewrite failed; splitting into {len(parts)} part(s) for recovery."
                        print_warn(msg)
                        recovered_chunks: List[str] = []
                        recovered_devs: List[Any] = []
                        summary_cursor = previous_summary
                        for j, part in enumerate(parts, start=1):
                            if args.verbose:
                                vprint(f"Rewriting recovery subchunk {j}/{len(parts)}...")
                            sub_md, sub_obj, _sub_comp = rewrite_chunk(
                                part,
                                None,
                                None,
                                depth + 1,
                                summary_cursor,
                                summary_words,
                                summary_enabled
                            )
                            recovered_chunks.append(sub_md)
                            sub_devs = sub_obj.get("deviations", []) if isinstance(sub_obj, dict) else []
                            if isinstance(sub_devs, list):
                                recovered_devs.extend(sub_devs)
                            if summary_enabled and isinstance(sub_obj, dict):
                                summary_cursor = sub_obj.get("chunk_summary", summary_cursor)
                        merged = "\n\n".join(s.strip() for s in recovered_chunks if isinstance(s, str) and s.strip()).strip()
                        if merged:
                            recovered_obj = {
                                "final_markdown": merged,
                                "deviations": [
                                    {
                                        "rule_or_field": "chunk_recovery_split",
                                        "reason": "Chunk was split and rewritten due to repeated invalid LLM output.",
                                        "parts": len(parts),
                                        "depth": depth + 1
                                    }
                                ] + recovered_devs,
                                "self_check": {
                                    "notes": [
                                        "Chunk rewritten via recovery split after repeated invalid LLM output."
                                    ]
                                }
                            }
                            recovered_comp = compute_style_compliance(fingerprint, filter_author_voice_text(merged))
                            if summary_enabled:
                                fallback_summary = build_fallback_summary(merged, summary_words)
                                recovered_obj["chunk_summary"] = fallback_summary
                            return merged, recovered_obj, recovered_comp

                # Last resort: preserve the original chunk verbatim and continue.
                print_warn("Preserving original chunk verbatim due to repeated invalid LLM output.")
                preserved_obj = {
                    "final_markdown": md_chunk,
                    "deviations": [
                        {
                            "rule_or_field": "chunk_rewrite_failed_preserved",
                            "reason": "LLM repeatedly failed to return final_markdown; original chunk preserved verbatim.",
                            "depth": depth,
                            "llm_raw_preview": (last_raw[:500] if isinstance(last_raw, str) else "")
                        }
                    ],
                    "self_check": {
                        "notes": [
                            "Chunk preserved verbatim due to repeated invalid LLM output."
                        ]
                    }
                }
                preserved_comp = compute_style_compliance(fingerprint, filter_author_voice_text(md_chunk))
                if summary_enabled:
                    preserved_obj["chunk_summary"] = (previous_summary or "")
                return md_chunk, preserved_obj, preserved_comp
            raw = last_raw
            usage = last_usage
            final_md = out_obj.get("final_markdown")

            # Optional stochastic micro-variation layer (bounded, deterministic).
            variance = {}
            if isinstance(tunables, dict):
                variance = tunables.get("humanizer_variance", {}) if isinstance(tunables.get("humanizer_variance", {}), dict) else {}
            if isinstance(variance, dict) and variance.get("enabled"):
                seed = int(variance.get("seed", 0))
                max_ops_per_1000w = float(variance.get("max_ops_per_1000w", 0.0))
                allowed_ops = variance.get("allowed_ops", ["swap_transition", "drop_filler"])
                if isinstance(allowed_ops, list):
                    allowed_ops = [str(op) for op in allowed_ops if isinstance(op, (str, int, float))]
                else:
                    allowed_ops = []
                final_md, ops_applied = apply_humanizer_variance(final_md, seed, max_ops_per_1000w, allowed_ops)
                if ops_applied:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "tunables.humanizer_variance",
                        "reason": "Applied bounded stochastic micro-variations.",
                        "ops": ops_applied
                    })

            if forbid_em_dashes:
                final_md, removed = enforce_no_em_dashes(final_md)
                if removed:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "punctuation.em_dashes",
                        "reason": "Em dashes removed to satisfy humanizer_mandatory constraint.",
                        "count": removed
                    })

            if emoji_policy and str(emoji_policy) != "none":
                final_md, removed_emoji, replaced_emoji = enforce_emoji_policy(final_md, str(emoji_policy))
                if removed_emoji or replaced_emoji:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "emoji_policy",
                        "policy": emoji_policy,
                        "removed": removed_emoji,
                        "replaced": replaced_emoji
                    })

            if args.force_person and not args.no_style_retry and attempts < args.max_style_retries:
                voice_text = filter_author_voice_text(final_md)
                override_eval = evaluate_pronoun_override(voice_text, args.force_person)
                if override_eval.get("violations"):
                    style_feedback = {
                        "pronoun_override": override_eval,
                        "notes": [
                            "Pronoun override violations detected. Rewrite to match forced narrative voice."
                        ]
                    }
                    attempts += 1
                    continue

            if summary_enabled:
                llm_summary = out_obj.get("chunk_summary") if isinstance(out_obj, dict) else None
                summary = llm_summary if isinstance(llm_summary, str) else ""
                if not summary.strip():
                    fallback = build_semantic_fallback_summary(final_md, summary_words)
                    summary = normalize_summary(fallback, summary_words)
                    if args.verbose and chunk_index is not None and chunk_total is not None:
                        vprint(
                            f"Chunk {chunk_index}/{chunk_total} summary fallback (LLM empty)."
                        )
                        vprint(f"  LLM summary: (empty)")
                        vprint(f"  Fallback summary: {summary}")
                else:
                    summary = normalize_summary(summary, summary_words)
                    if summary and summary_is_meta(summary):
                        fallback = build_semantic_fallback_summary(final_md, summary_words)
                        summary = normalize_summary(fallback, summary_words)
                        if args.verbose and chunk_index is not None and chunk_total is not None:
                            vprint(
                                f"Chunk {chunk_index}/{chunk_total} summary fallback (LLM meta/task-focused)."
                            )
                            vprint(f"  LLM summary: {normalize_summary(llm_summary, summary_words)}")
                            vprint(f"  Fallback summary: {summary}")
                out_obj["chunk_summary"] = summary
                if args.verbose and chunk_index is not None and chunk_total is not None:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} summary "
                        f"({len(summary.split())} words): {summary}"
                    )

            compliance = compute_style_compliance(fingerprint, filter_author_voice_text(final_md))
            if args.verbose and chunk_index is not None and chunk_total is not None:
                comp_score = compliance.get("score")
                if isinstance(comp_score, (int, float)):
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} attempt {attempts + 1} "
                        f"compliance score: {comp_score:.3f} "
                        f"(threshold {args.style_retry_threshold})"
                    )
                else:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} attempt {attempts + 1} "
                        f"compliance score: {comp_score} "
                        f"(threshold {args.style_retry_threshold})"
                    )
            if not args.no_style_retry and attempts < args.max_style_retries and compliance["score"] < args.style_retry_threshold:
                style_feedback = {
                    "score": compliance["score"],
                    "deltas": compliance.get("deltas", [])
                }
                if controller_overlay and isinstance(tunables, dict):
                    controller_conf = tunables.get("humanization_controller", {}) if isinstance(tunables.get("humanization_controller", {}), dict) else {}
                    if controller_conf.get("feedback_enabled", False):
                        max_feedback_retries = int(controller_conf.get("max_feedback_retries", args.max_style_retries))
                        if attempts < max_feedback_retries:
                            overlay_feedback = build_overlay_feedback(
                                controller_overlay,
                                filter_author_voice_text(final_md),
                                controller_conf
                            )
                            if overlay_feedback:
                                style_feedback["humanization_controller"] = overlay_feedback
                attempts += 1
                continue

            if chunk_index is not None and chunk_total is not None and args.verbose:
                if isinstance(usage, dict):
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} token usage: "
                        f"prompt={usage.get('prompt_tokens', 'n/a')}, "
                        f"completion={usage.get('completion_tokens', 'n/a')}, "
                        f"total={usage.get('total_tokens', 'n/a')}"
                    )
                else:
                    output_tokens = estimate_tokens_for_text(final_md)
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} token estimate: "
                        f"in={input_tokens}, out={output_tokens}"
                    )
            last_out = out_obj
            return final_md, out_obj, compliance

    summary_enabled = False
    summary_words = 25
    max_input_tokens = 0
    min_chunks_when_perturbing = 1
    chunk_split_on = "sentence"
    perturbations_active = False
    if isinstance(tunables, dict):
        chunk_conf = tunables.get("chunking", {})
        if isinstance(chunk_conf, dict):
            summary_conf = chunk_conf.get("chunk_summary", {})
            if isinstance(summary_conf, dict):
                if isinstance(summary_conf.get("enabled"), bool):
                    summary_enabled = summary_conf.get("enabled")
                try:
                    summary_words = int(summary_conf.get("summary_words", summary_words))
                except (TypeError, ValueError):
                    summary_words = summary_words
                summary_words = max(5, min(summary_words, 120))
            cap = chunk_conf.get("max_input_tokens")
            min_chunks = chunk_conf.get("min_chunks_when_perturbing")
            if isinstance(min_chunks, int) and min_chunks > 0:
                min_chunks_when_perturbing = min_chunks
            split_on = chunk_conf.get("chunk_split_on")
            if isinstance(split_on, str) and split_on.lower() in ("word", "sentence", "paragraph"):
                chunk_split_on = split_on.lower()
    summary_placeholder = None
    if summary_enabled:
        summary_placeholder = ("summary " * summary_words).strip()
    base_messages = build_messages_for_chunk(
        "",
        None,
        True,
        None,
        None,
        summary_placeholder,
        summary_words,
        summary_enabled
    )
    base_tokens = estimate_tokens_for_messages(base_messages)
    max_input_tokens = max(400, cfg.max_prompt_tokens - base_tokens)
    if isinstance(tunables, dict):
        chunk_conf = tunables.get("chunking", {})
        if isinstance(chunk_conf, dict):
            cap = chunk_conf.get("max_input_tokens")
            if isinstance(cap, int) and cap > 0:
                max_input_tokens = min(max_input_tokens, max(200, cap))
        humanizer_var = tunables.get("humanizer_variance", {})
        if isinstance(humanizer_var, dict) and humanizer_var.get("enabled", False):
            perturbations_active = True
        controller_conf = tunables.get("humanization_controller", {})
        if isinstance(controller_conf, dict) and controller_conf.get("enabled", False):
            perturbations_active = True
        factor = compute_variance_aware_factor(fingerprint, tunables)
        if factor < 1.0:
            max_input_tokens = max(200, int(max_input_tokens * factor))
            if args.verbose:
                vprint(f"Variance-aware chunking factor applied: {factor:.2f}")
    input_tokens = estimate_tokens_for_text(input_md)
    initial_tokens = input_tokens + base_tokens
    if args.verbose:
        vprint(
            f"Prompt overhead tokens: {base_tokens}; "
            f"input budget: {max_input_tokens}; input tokens: {input_tokens}"
        )
        if summary_enabled:
            vprint(f"Chunk summary chaining enabled: {summary_words} words")
        vprint(
            f"Chunk split strategy: {chunk_split_on} "
            "(fallback to sentence/word if oversized; list lines treated as sentences)"
        )
    force_min_chunks = perturbations_active and min_chunks_when_perturbing > 1
    summary_active = summary_enabled and (input_tokens > max_input_tokens or force_min_chunks)
    if input_tokens <= max_input_tokens and not force_min_chunks:
        vprint("Calling LLM to apply fingerprint...")
        try:
            final_md, out_obj, compliance = rewrite_chunk(
                input_md,
                None,
                None,
                0,
                None,
                summary_words,
                False
            )
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
        final_md = restore_placeholders(final_md, frozen_blocks)
        final_md = restore_placeholders(final_md, html_map)
        final_md = restore_placeholders(final_md, math_map)
        final_md = restore_placeholders(final_md, entity_map)
        final_md = restore_placeholders(final_md, inline_code_map)
        final_md = restore_placeholders(final_md, quote_map)
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
        if input_tokens > max_input_tokens:
            vprint(f"Prompt too large ({initial_tokens} tokens); chunking input...")
        elif force_min_chunks and args.verbose:
            vprint(f"Perturbations enabled; enforcing minimum chunk count: {min_chunks_when_perturbing}")
        def build_messages_for_chunk_est(md_chunk: str, style_feedback=None, for_estimate: bool = False, fingerprint_override=None, controller_overlay=None):
            return build_messages_for_chunk(
                md_chunk,
                style_feedback,
                for_estimate,
                fingerprint_override,
                controller_overlay,
                summary_placeholder if summary_active else None,
                summary_words,
                summary_active
            )
        chunks = chunk_markdown(
            input_md,
            build_messages_for_chunk_est,
            cfg.max_prompt_tokens,
            max_input_tokens_override=max_input_tokens,
            split_on=chunk_split_on
        )
        non_empty = [c for c in chunks if isinstance(c, str) and c.strip()]
        if len(non_empty) != len(chunks):
            vprint(f"Filtered out {len(chunks) - len(non_empty)} empty chunk(s).")
        chunks = enforce_min_chunks(input_md, non_empty, min_chunks_when_perturbing) if force_min_chunks else non_empty
        vprint(f"Chunked into {len(chunks)} parts.")
        running_summary = ""
        for idx, chunk in enumerate(chunks, start=1):
            vprint(f"Rewriting chunk {idx}/{len(chunks)}...")
            try:
                final_md, out_obj, compliance = rewrite_chunk(
                    chunk,
                    idx,
                    len(chunks),
                    0,
                    running_summary,
                    summary_words,
                    summary_active
                )
            except RuntimeError:
                return 3
            if summary_active and isinstance(out_obj, dict):
                running_summary = out_obj.get("chunk_summary", running_summary)
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
            final_md = restore_placeholders(final_md, frozen_blocks)
            final_md = restore_placeholders(final_md, html_map)
            final_md = restore_placeholders(final_md, math_map)
            final_md = restore_placeholders(final_md, entity_map)
            final_md = restore_placeholders(final_md, inline_code_map)
            final_md = restore_placeholders(final_md, quote_map)
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
    section_conf = tunables.get("section_restore", {}) if isinstance(tunables, dict) else {}
    restore_enabled = bool(section_conf.get("enabled", True))
    max_restore_sections = int(section_conf.get("max_restore_sections", 20))
    heading_similarity_threshold = float(section_conf.get("heading_similarity_threshold", 0.85))
    signature_similarity_threshold = float(section_conf.get("signature_similarity_threshold", 0.6))
    signature_min_overlap = int(section_conf.get("signature_min_overlap", 6))
    if max_restore_sections < 0:
        max_restore_sections = 0
    if heading_similarity_threshold < 0:
        heading_similarity_threshold = 0.0
    if signature_similarity_threshold < 0:
        signature_similarity_threshold = 0.0
    if signature_min_overlap < 0:
        signature_min_overlap = 0

    output_blocks = extract_heading_blocks(final_md)
    output_blocks_with_sig: List[Dict[str, Any]] = []
    for block in output_blocks:
        output_blocks_with_sig.append({
            **block,
            "signature": section_signature(block["block"]),
            "norm_title": normalize_heading(block["title"])
        })
    output_keys = {b["key"] for b in output_blocks_with_sig}

    used_output_idx: set[int] = set()
    matched_start_by_input: List[int | None] = []
    match_diagnostics: List[Dict[str, Any]] = []

    for block in section_blocks_restored:
        best_heading_idx = None
        best_heading_score = 0.0
        best_signature_idx = None
        best_signature_score = 0.0
        best_signature_overlap = 0
        for i, out_block in enumerate(output_blocks_with_sig):
            if out_block.get("level") != block.get("level"):
                continue
            score = heading_similarity(block.get("key", ""), out_block.get("key", ""))
            if score > best_heading_score:
                best_heading_score = score
                best_heading_idx = i
        if block["key"] in output_keys:
            out_idx = next(
                (i for i, b in enumerate(output_blocks_with_sig)
                 if b["key"] == block["key"] and b.get("level") == block.get("level")),
                None
            )
            if out_idx is not None:
                used_output_idx.add(out_idx)
                matched_start_by_input.append(output_blocks_with_sig[out_idx]["start_line"])
                match_diagnostics.append({
                    "method": "heading_exact",
                    "best_heading_score": best_heading_score,
                    "best_heading_title": output_blocks_with_sig[best_heading_idx]["title"] if best_heading_idx is not None else None,
                    "best_signature_score": best_signature_score,
                    "best_signature_overlap": best_signature_overlap,
                    "best_signature_title": None
                })
                continue
        if best_heading_idx is not None and best_heading_score >= heading_similarity_threshold:
            used_output_idx.add(best_heading_idx)
            matched_start_by_input.append(output_blocks_with_sig[best_heading_idx]["start_line"])
            match_diagnostics.append({
                "method": "heading_fuzzy",
                "best_heading_score": best_heading_score,
                "best_heading_title": output_blocks_with_sig[best_heading_idx]["title"],
                "best_signature_score": best_signature_score,
                "best_signature_overlap": best_signature_overlap,
                "best_signature_title": None
            })
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
        if best_idx is not None:
            best_signature_idx = best_idx
            best_signature_score = best_score
            best_signature_overlap = overlap
        if best_idx is not None and best_score >= signature_similarity_threshold and overlap >= signature_min_overlap:
            used_output_idx.add(best_idx)
            matched_start_by_input.append(output_blocks_with_sig[best_idx]["start_line"])
            match_diagnostics.append({
                "method": "signature",
                "best_heading_score": best_heading_score,
                "best_heading_title": output_blocks_with_sig[best_heading_idx]["title"] if best_heading_idx is not None else None,
                "best_signature_score": best_signature_score,
                "best_signature_overlap": best_signature_overlap,
                "best_signature_title": output_blocks_with_sig[best_idx]["title"]
            })
        else:
            matched_start_by_input.append(None)
            match_diagnostics.append({
                "method": "missing",
                "best_heading_score": best_heading_score,
                "best_heading_title": output_blocks_with_sig[best_heading_idx]["title"] if best_heading_idx is not None else None,
                "best_signature_score": best_signature_score,
                "best_signature_overlap": best_signature_overlap,
                "best_signature_title": output_blocks_with_sig[best_signature_idx]["title"] if best_signature_idx is not None else None
            })

    missing_sections = [b for idx, b in enumerate(section_blocks_restored) if matched_start_by_input[idx] is None]
    if missing_sections:
        preview_limit = 10
        for idx, block in enumerate(section_blocks_restored):
            if matched_start_by_input[idx] is not None:
                continue
            if block not in missing_sections:
                continue
            diag = match_diagnostics[idx] if idx < len(match_diagnostics) else {}
            best_heading_title = diag.get("best_heading_title")
            best_heading_score = diag.get("best_heading_score")
            best_signature_title = diag.get("best_signature_title")
            best_signature_score = diag.get("best_signature_score")
            best_signature_overlap = diag.get("best_signature_overlap")
            heading_part = (
                f"best heading match '{best_heading_title}' ({best_heading_score:.2f})"
                if best_heading_title and isinstance(best_heading_score, (float, int))
                else "no heading match"
            )
            signature_part = (
                f"best signature match '{best_signature_title}' ({best_signature_score:.2f}, overlap {best_signature_overlap})"
                if best_signature_title and isinstance(best_signature_score, (float, int))
                else "no signature match"
            )
            print_warn(f"Missing section '{block['title']}': {heading_part}; {signature_part}.")
            preview_limit -= 1
            if preview_limit <= 0:
                remaining = sum(
                    1
                    for j, b in enumerate(section_blocks_restored)
                    if matched_start_by_input[j] is None and b in missing_sections
                ) - 10
                if remaining > 0:
                    print_warn(f"...and {remaining} more missing section(s).")
                break
        if not restore_enabled:
            print_warn(
                f"Missing {len(missing_sections)} section(s); restoration disabled by tunables."
            )
            for idx, block in enumerate(section_blocks_restored):
                if matched_start_by_input[idx] is not None:
                    continue
                diag = match_diagnostics[idx] if idx < len(match_diagnostics) else {}
                all_deviations.append({
                    "rule_or_field": "missing_section",
                    "heading": block["title"],
                    "reason": "Section missing from LLM output; restoration disabled.",
                    "diagnostics": diag
                })
            missing_sections = []
        elif max_restore_sections == 0:
            print_warn(
                f"Missing {len(missing_sections)} section(s); restoration cap is 0."
            )
            for idx, block in enumerate(section_blocks_restored):
                if matched_start_by_input[idx] is not None:
                    continue
                diag = match_diagnostics[idx] if idx < len(match_diagnostics) else {}
                all_deviations.append({
                    "rule_or_field": "missing_section",
                    "heading": block["title"],
                    "reason": "Section missing from LLM output; restoration cap is 0.",
                    "diagnostics": diag
                })
            missing_sections = []
    if missing_sections:
        if len(missing_sections) > max_restore_sections:
            print_warn(
                f"Missing {len(missing_sections)} section(s); restoring first {max_restore_sections} due to cap."
            )
            missing_sections = missing_sections[:max_restore_sections]
        titles = ", ".join(b["title"] for b in missing_sections[:10])
        suffix = "..." if len(missing_sections) > 10 else ""
        print_warn(f"Restoring {len(missing_sections)} missing section(s) in original order: {titles}{suffix}")
        lines = final_md.splitlines()
        offset = 0
        for idx, block in enumerate(section_blocks_restored):
            if matched_start_by_input[idx] is not None:
                continue
            if block not in missing_sections:
                diag = match_diagnostics[idx] if idx < len(match_diagnostics) else {}
                all_deviations.append({
                    "rule_or_field": "missing_section",
                    "heading": block["title"],
                    "reason": "Section missing from LLM output; restoration cap exceeded.",
                    "diagnostics": diag
                })
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
            diag = match_diagnostics[idx] if idx < len(match_diagnostics) else {}
            all_deviations.append({
                "rule_or_field": "missing_section",
                "heading": block["title"],
                "reason": "Section missing from LLM output; restored at original position.",
                "diagnostics": diag
            })
        final_md = "\n".join(lines).strip()
    line_count_in = len(original_input_md.splitlines())
    line_count_out = len(final_md.splitlines())
    line_change_pct = ((line_count_out - line_count_in) / max(1, line_count_in)) * 100.0
    print(f"Line count change: {line_count_in} -> {line_count_out} ({line_change_pct:+.1f}%).")

    word_count_in = len(words(original_input_md))
    word_count_out = len(words(final_md))
    word_change_pct = ((word_count_out - word_count_in) / max(1, word_count_in)) * 100.0
    vprint(f"Word count change: {word_count_in} -> {word_count_out} ({word_change_pct:+.1f}%).")

    para_count_in = len(split_paragraphs(original_input_md))
    para_count_out = len(split_paragraphs(final_md))
    para_change_pct = ((para_count_out - para_count_in) / max(1, para_count_in)) * 100.0
    vprint(f"Paragraph count change: {para_count_in} -> {para_count_out} ({para_change_pct:+.1f}%).")

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

    if args.force_person:
        voice_text = filter_author_voice_text(final_md)
        override_eval = evaluate_pronoun_override(voice_text, args.force_person)
        all_deviations.append({
            "rule_or_field": "pronoun_override",
            "mode": override_eval.get("mode"),
            "allowed_count": override_eval.get("allowed_count"),
            "violations": override_eval.get("violations", {}),
            "note": "Pronoun override applied; violations indicate forbidden pronoun usage in author-voice text."
        })

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

    humanization_metrics = None
    if args.metrics:
        input_metrics = compute_humanization_metrics(original_input_md, fingerprint)
        output_metrics = compute_humanization_metrics(final_md, fingerprint)
        humanization_metrics = {
            "input": input_metrics,
            "output": output_metrics
        }
        input_weights = {}
        output_weights = {}
        if isinstance(tunables, dict):
            hm = tunables.get("humanization_metrics", {})
            if isinstance(hm, dict):
                weights = hm.get("weights", {})
                if isinstance(weights, dict):
                    input_weights = {str(k): float(v) for k, v in weights.items() if isinstance(v, (int, float))}
                    output_weights = dict(input_weights)
        input_metrics.update(compute_humanization_aggregate(input_metrics.get("scores", {}), input_weights))
        output_metrics.update(compute_humanization_aggregate(output_metrics.get("scores", {}), output_weights))
    if args.metrics and isinstance(humanization_metrics, dict):
        humanization_weights = {}
        if isinstance(tunables, dict):
            hm = tunables.get("humanization_metrics", {})
            if isinstance(hm, dict):
                weights = hm.get("weights", {})
                if isinstance(weights, dict):
                    humanization_weights = {str(k): float(v) for k, v in weights.items() if isinstance(v, (int, float))}
        aggregate = compute_humanization_aggregate(humanization_metrics.get("scores", {}), humanization_weights)
        humanization_metrics.update(aggregate)
        all_deviations.append({
            "rule_or_field": "humanization_metrics",
            "metrics": humanization_metrics
        })

    out_path = args.out or args.inp.with_suffix(args.inp.suffix + ".styled.md")
    vprint(f"Writing output: {out_path}")
    out_path.write_text(final_md, encoding="utf-8")
    print(f"Wrote rewritten markdown to: {out_path}")
    if args.metrics and isinstance(humanization_metrics, dict):
        for label in ("input", "output"):
            metrics = humanization_metrics.get(label, {})
            scores = metrics.get("scores", {}) if isinstance(metrics, dict) else {}
            aggregate_100 = metrics.get("aggregate_score_100") if isinstance(metrics, dict) else None
            if isinstance(scores, dict) and scores:
                print(
                    f"Humanization metrics ({label}, heuristic scores): "
                    f"lexical_diversity={scores.get('lexical_diversity', 0):.2f}, "
                    f"repetition_inverse={scores.get('repetition_inverse', 0):.2f}, "
                    f"sentence_burstiness={scores.get('sentence_burstiness', 0):.2f}, "
                    f"paragraph_burstiness={scores.get('paragraph_burstiness', 0):.2f}, "
                    f"punctuation_variety={scores.get('punctuation_variety', 0):.2f}"
                )
            if isinstance(aggregate_100, (int, float)):
                print(f"Humanization aggregate score ({label}, 0–100): {aggregate_100:.2f}")

    # Optionally also write deviations report
    if humanizer_debug:
        all_deviations.append(humanizer_debug)
    if input_style_signals:
        all_deviations.append({
            "rule_or_field": "input_style_signals",
            "signals": input_style_signals
        })
    if humanization_metrics:
        all_deviations.append({
            "rule_or_field": "humanization_metrics",
            "metrics": humanization_metrics
        })
    if all_deviations:
        rep = out_path.with_suffix(out_path.suffix + ".deviations.json")
        rep.write_text(json.dumps(all_deviations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote deviations report to: {rep}")
    vprint("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
