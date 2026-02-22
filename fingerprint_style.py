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
import math
import copy
import collections
import dataclasses
import io
import json
import hashlib
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

from utils import (
    approx_rate_per_1000_words,
    clamp01,
    histogram,
    safe_mean,
    safe_stdev,
    split_paragraphs as utils_split_paragraphs,
    split_sentences as utils_split_sentences,
    words,
)
from common import resolve_path_prefer_cwd, resolve_required_path

import requests

try:
    from docx import Document  # python-docx
except Exception:
    Document = None  # optional


# Function: Print warnings to stderr.
def print_warn(msg: str) -> None:
    print(msg, file=sys.stderr)

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
QUOTE_SPAN_RE = re.compile(r"\"([^\"]+)\"", re.S)
QUOTE_SPAN_CURLY_RE = re.compile(r"“([^”]+)”", re.S)
BOILERPLATE_LINE_RE = re.compile(
    r"(?i)\\b("
    r"copyright|all rights reserved|rights reserved|"
    r"terms of use|terms & conditions|terms and conditions|"
    r"privacy policy|conditions of use|legal notice|imprint|disclaimer|"
    r"permission(?:s)?|reproduction|published by|"
    r"©|\\(c\\)"
    r")\\b"
)
BOILERPLATE_HEADING_RE = re.compile(
    r"(?i)\\b("
    r"copyright|terms|conditions|privacy|legal|imprint|disclaimer|permissions"
    r")\\b"
)
DEFAULT_MAX_FILES = 2000
DEFAULT_MAX_BYTES_PER_FILE = 2_000_000  # 2 MB per file
DEFAULT_MAX_TOTAL_CHARS_FOR_LLM = 180_000  # excerpt cap; we send stats + representative excerpts
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
LICENSE_FILENAME = "LICENSE.md"
LEXICON_HINTS_FILENAME = "lexicon_hints.json"
AVOID_LIST_FILENAME = "config.avoid.txt"
COMMON_WORDS_FILENAME = "config.common_words.txt"
ENTITY_BLACKLIST_FILENAME = "config.entity_blacklist.txt"
LOCAL_SPELLING_RULES_FILENAME = "config.local_spelling_rules.json"
QUOTE_MODE: str | None = None
DEFAULT_COMMON_WORDS = {
    "ability","account","action","activity","address","agreement","answer","area","argument","article",
    "attention","audience","author","back","balance","base","basis","beginning","benefit","body",
    "book","business","case","cause","center","change","chapter","child","choice","city",
    "class","company","comparison","condition","consideration","contact","content","control","cost",
    "country","course","court","culture","data","day","decision","development","difference","direction",
    "discussion","door","drive","education","effect","effort","end","energy","environment","event",
    "evidence","example","experience","fact","family","field","figure","focus","food","force",
    "form","friend","function","future","game","goal","government","group","growth","hand",
    "health","history","home","hour","house","idea","importance","interest","issue","job",
    "judgment","kind","knowledge","law","level","life","line","list","literature","management",
    "market","meaning","method","mind","model","moment","money","month","music","name",
    "nature","need","number","object","office","opinion","order","organization","page","paper",
    "part","party","pattern","people","person","place","plan","point","policy","position",
    "power","price","problem","process","product","program","project","quality","question","rate",
    "reason","report","research","result","role","room","rule","school","science","section",
    "sense","service","side","situation","skill","society","solution","source","space","state",
    "story","strategy","study","subject","support","system","table","team","technology","term",
    "theory","thing","thought","time","title","tool","topic","trade","training","understanding",
    "value","view","voice","way","week","word","work","world","year",
    "accept","achieve","add","allow","appear","apply","argue","ask","avoid","become",
    "believe","bring","build","call","carry","change","choose","claim","compare","consider",
    "continue","create","deal","decide","define","develop","discover","discuss","drive","expect",
    "explain","feel","find","follow","form","gain","give","handle","help","hold","include",
    "increase","indicate","keep","know","learn","leave","lead","listen","look","make",
    "maintain","mean","meet","move","need","notice","offer","open","provide","raise",
    "reach","read","receive","reduce","reflect","remain","remember","report","require","return",
    "risk","run","say","see","seem","send","serve","set","show","speak",
    "start","state","stop","suggest","take","talk","teach","tell","think","turn",
    "understand","use","view","want","watch","work","write"
}

# Function: Load prompts.
def load_prompts() -> Dict[str, Any]:
    # Load externalized prompt templates located alongside this script.
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.json not found at {PROMPTS_PATH}")
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

# Function: Resolve license path.
def resolve_license_path() -> Path | None:
    # Resolve LICENSE.md from CWD or script directory.
    return resolve_path_prefer_cwd(LICENSE_FILENAME, __file__)


# Function: Render markdown.
def render_markdown(text: str) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except Exception:
        print(text)
        return
    console = Console()
    console.print(Markdown(text))


# Function: Print license and exit.
def print_license_and_exit() -> int:
    path = resolve_license_path()
    if not path:
        print(f"License file not found: {LICENSE_FILENAME}", file=sys.stderr)
        return 2
    render_markdown(path.read_text(encoding="utf-8"))
    return 0

# Function: Load optional lexicon hints.
def load_optional_lexicon_hints() -> Optional[Dict[str, Any]]:
    # Load optional lexicon hints from CWD or script directory.
    path = resolve_path_prefer_cwd(LEXICON_HINTS_FILENAME, __file__)
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# Function: Load tunables snapshot.
def load_tunables_snapshot() -> Optional[Dict[str, Any]]:
    # Load tunables for auditability snapshot (CWD or script directory).
    path = resolve_path_prefer_cwd("config.tunables.json", __file__)
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None

# Function: Parse list lines.
def parse_list_lines(text: str) -> List[str]:
    items: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


# Function: Parse avoid list.
def parse_avoid_list(text: str) -> List[str]:
    return parse_list_lines(text)


# Function: Load avoid list.
def load_avoid_list() -> List[str]:
    # Load optional avoid-word list from CWD or script directory.
    path = resolve_path_prefer_cwd(AVOID_LIST_FILENAME, __file__)
    if not path:
        return []
    try:
        return parse_list_lines(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# Function: Load common words.
def load_common_words() -> List[Tuple[str, Optional[float]]]:
    # Load optional common-words list from CWD or script directory.
    path = resolve_path_prefer_cwd(COMMON_WORDS_FILENAME, __file__)
    if path and path.exists():
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            raw_lines = []
        entries: List[Tuple[str, Optional[float]]] = []
        seen: set[str] = set()
        for raw in raw_lines:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if not parts:
                continue
            word = parts[0].strip().lower()
            if not word or word in seen:
                continue
            freq: Optional[float] = None
            if len(parts) >= 2:
                try:
                    freq = float(parts[1])
                except Exception:
                    freq = None
            seen.add(word)
            entries.append((word, freq))
        if entries:
            return entries
    return [(w, None) for w in sorted(DEFAULT_COMMON_WORDS)]


# Function: Normalize entity name.
def normalize_entity_name(value: str) -> str:
    try:
        import unicodedata
    except Exception:
        unicodedata = None
    lowered = value.lower()
    if unicodedata is not None:
        normalized = unicodedata.normalize("NFKD", lowered)
        lowered = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    tokens = re.findall(r"[A-Za-z0-9]+", lowered)
    return " ".join(tokens)


# Function: Load entity blacklist.
def load_entity_blacklist() -> List[str]:
    # Load optional entity blacklist from CWD or script directory.
    path = resolve_path_prefer_cwd(ENTITY_BLACKLIST_FILENAME, __file__)
    if not path:
        return []
    try:
        raw_items = parse_list_lines(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        norm = normalize_entity_name(item)
        if not norm:
            continue
        if " " not in norm and len(norm) < 3:
            continue
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


# Function: Build entity matcher.
def build_entity_matcher(entities: List[str]) -> tuple[set[str], set[tuple[str, ...]], int]:
    singles: set[str] = set()
    phrases: set[tuple[str, ...]] = set()
    max_len = 1
    for item in entities:
        parts = tuple(item.split())
        if not parts:
            continue
        if len(parts) == 1:
            singles.add(parts[0])
        else:
            phrases.add(parts)
            if len(parts) > max_len:
                max_len = len(parts)
    return singles, phrases, max_len


# Function: Apply case.
def _apply_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper() and template[1:].islower():
        return replacement.capitalize()
    if template.islower():
        return replacement.lower()
    return replacement


# Function: Load local spelling rules.
def load_local_spelling_rules() -> Dict[str, Any]:
    rules_path = resolve_path_prefer_cwd(LOCAL_SPELLING_RULES_FILENAME, __file__)
    if rules_path is None:
        rules_path = Path(__file__).resolve().parent / LOCAL_SPELLING_RULES_FILENAME
    if rules_path.exists():
        try:
            data = json.loads(rules_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


# Function: Build a spelling-variant mapping for a locale.
def build_local_spelling_map(rules: Dict[str, Any], locale: str) -> Dict[str, str]:
    if not isinstance(rules, dict):
        return {}
    locale = locale.lower()
    rules_conf = rules.get("rules", {}) if isinstance(rules.get("rules", {}), dict) else {}
    direct_variants = rules_conf.get("direct_variants", []) if isinstance(rules_conf.get("direct_variants", []), list) else []
    suffix_variants = rules_conf.get("suffix_variants", []) if isinstance(rules_conf.get("suffix_variants", []), list) else []
    context_variants = rules_conf.get("context_variants", []) if isinstance(rules_conf.get("context_variants", []), list) else []
    mapping: Dict[str, str] = {}

    # Build suffix forms, including optional dropped-e inflection (e.g., organize + ation -> organization).
    def _spelling_forms_for_suffix(base: str, suffix: str) -> tuple[str, str | None]:
        regular = base + suffix
        dropped = None
        if (
            suffix
            and base.endswith("e")
            and suffix[0].lower() in {"a", "e", "i", "o", "u", "y"}
            and len(base) > 1
        ):
            dropped = base[:-1] + suffix
        return regular, dropped

    for entry in direct_variants:
        variants = entry.get("variants") if isinstance(entry, dict) else None
        if not isinstance(variants, dict):
            continue
        target = variants.get(locale)
        if not isinstance(target, str):
            continue
        for _, variant in variants.items():
            if isinstance(variant, str):
                mapping[variant.lower()] = target
    for entry in suffix_variants:
        variants = entry.get("variants") if isinstance(entry, dict) else None
        suffixes = entry.get("suffixes") if isinstance(entry, dict) else None
        if not isinstance(variants, dict) or not isinstance(suffixes, list):
            continue
        target_base = variants.get(locale)
        if not isinstance(target_base, str):
            continue
        for _, variant_base in variants.items():
            if not isinstance(variant_base, str):
                continue
            for suffix in suffixes:
                if isinstance(suffix, str):
                    src_regular, src_drop = _spelling_forms_for_suffix(variant_base, suffix)
                    dst_regular, dst_drop = _spelling_forms_for_suffix(target_base, suffix)
                    mapping[src_regular.lower()] = dst_regular
                    if src_drop:
                        mapping[src_drop.lower()] = dst_drop or dst_regular
    for entry in context_variants:
        variants = entry.get("variants") if isinstance(entry, dict) else None
        if not isinstance(variants, dict):
            continue
        target = variants.get(locale)
        if not isinstance(target, str):
            continue
        for _, variant in variants.items():
            if isinstance(variant, str):
                mapping[variant.lower()] = target
    return mapping


# Function: Normalize spelling variants in text via a mapping.
def normalize_text_spelling(text: str, mapping: Dict[str, str]) -> str:
    if not text or not mapping:
        return text
    word_re = re.compile(r"[A-Za-z][A-Za-z']+")
    parts: list[str] = []
    last = 0
    for match in word_re.finditer(text):
        start, end = match.start(), match.end()
        word = match.group(0)
        parts.append(text[last:start])
        last = end
        replacement = mapping.get(word.lower())
        if replacement:
            parts.append(_apply_case(word, replacement))
        else:
            parts.append(word)
    parts.append(text[last:])
    return "".join(parts)


# Function: Normalize lexicon spellings to a US baseline.
def normalize_lexicon_spelling(
    fingerprint: Dict[str, Any],
    rules: Dict[str, Any],
    avoid_list: List[str]
) -> None:
    if not isinstance(fingerprint, dict):
        return
    lexicon = fingerprint.get("lexicon")
    if not isinstance(lexicon, dict):
        return
    mapping = build_local_spelling_map(rules, "us")
    if not mapping:
        return
    avoid_literal = {str(item).lower().strip() for item in avoid_list if isinstance(item, str)}

    # Function: Normalize list.
    def normalize_list(items: Any, skip_literal: bool = False) -> list[str]:
        if not isinstance(items, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            raw = item.strip()
            if not raw:
                continue
            if skip_literal and raw.lower() in avoid_literal:
                norm = raw
            else:
                norm = normalize_text_spelling(raw, mapping)
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
        return out

    if "preferred_words" in lexicon:
        lexicon["preferred_words"] = normalize_list(lexicon.get("preferred_words"))
    if "preferred_phrases" in lexicon:
        lexicon["preferred_phrases"] = normalize_list(lexicon.get("preferred_phrases"))
    if "avoid_words" in lexicon:
        lexicon["avoid_words"] = normalize_list(lexicon.get("avoid_words"), skip_literal=True)
    if "avoid_words_soft" in lexicon:
        lexicon["avoid_words_soft"] = normalize_list(lexicon.get("avoid_words_soft"))

    synonym_prefs = lexicon.get("synonym_preferences")
    if isinstance(synonym_prefs, dict):
        updated: Dict[str, Any] = {}
        for key, val in synonym_prefs.items():
            if not isinstance(key, str):
                continue
            norm_key = normalize_text_spelling(key, mapping)
            if isinstance(val, list):
                norm_vals = normalize_list(val)
            elif isinstance(val, str):
                norm_vals = normalize_list([val])
            else:
                norm_vals = []
            if norm_key in updated:
                existing = updated[norm_key]
                if isinstance(existing, list):
                    for v in norm_vals:
                        if v not in existing:
                            existing.append(v)
            else:
                updated[norm_key] = norm_vals if norm_vals else val
        lexicon["synonym_preferences"] = updated


# Function: Merge avoid lists and normalize US spellings.
def normalize_lexicon_avoids(
    fingerprint: Dict[str, Any],
    measurements: Dict[str, Any],
    lexicon_hints: Optional[Dict[str, Any]],
    avoid_list: List[str]
) -> None:
    lexicon = fingerprint.get("lexicon")
    if not isinstance(lexicon, dict):
        lexicon = {}
        fingerprint["lexicon"] = lexicon

    hard_seen: set[str] = set()
    hard_list: List[str] = []

    # Function: Add hard.
    def add_hard(item: Any) -> None:
        if not isinstance(item, str):
            return
        token = item.strip()
        if not token or token in hard_seen:
            return
        hard_seen.add(token)
        hard_list.append(token)

    if isinstance(lexicon_hints, dict):
        for key in ("avoid_words_hard", "avoid_words"):
            val = lexicon_hints.get(key)
            if isinstance(val, list):
                for item in val:
                    add_hard(item)
            elif isinstance(val, str):
                add_hard(val)

    for item in avoid_list:
        add_hard(item)

    soft_seen: set[str] = set()
    soft_list: List[str] = []

    avoid_category_set: set[str] = set()
    lex_categories = lexicon.get("avoid_categories_soft")
    if isinstance(lex_categories, list):
        for item in lex_categories:
            if isinstance(item, str) and item.strip():
                avoid_category_set.add(item.strip())
    lex_avoid_measurements = measurements.get("lexical_avoidance", {}) if isinstance(measurements, dict) else {}
    if isinstance(lex_avoid_measurements, dict):
        category_rates = lex_avoid_measurements.get("category_rates_per_1000w")
        if isinstance(category_rates, dict):
            for key in category_rates.keys():
                if isinstance(key, str) and key.strip():
                    avoid_category_set.add(key.strip())

    # Function: Add soft.
    def add_soft(item: Any) -> None:
        if not isinstance(item, str):
            return
        token = item.strip()
        if not token or token in hard_seen or token in soft_seen:
            return
        if token in avoid_category_set:
            return
        soft_seen.add(token)
        soft_list.append(token)

    measurement_words = lex_avoid_measurements.get("rare_words", []) if isinstance(lex_avoid_measurements, dict) else []
    for item in measurement_words:
        if isinstance(item, dict):
            add_soft(item.get("word"))
        elif isinstance(item, str):
            add_soft(item)

    existing_soft = lexicon.get("avoid_words_soft")
    if isinstance(existing_soft, list):
        for item in existing_soft:
            add_soft(item)
    elif isinstance(existing_soft, str):
        add_soft(existing_soft)

    existing_hard = lexicon.get("avoid_words")
    if isinstance(existing_hard, list):
        for item in existing_hard:
            if item in hard_seen:
                continue
            add_soft(item)
    elif isinstance(existing_hard, str):
        if existing_hard not in hard_seen:
            add_soft(existing_hard)

    lexicon["avoid_words"] = hard_list
    if soft_list:
        lexicon["avoid_words_soft"] = soft_list


# Function: Normalize rewrite policy text for display.
def normalize_rewrite_policy(text: str, conf: Optional[Dict[str, Any]] = None) -> str:
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
    # Split into clauses on sentence punctuation and directive starts.
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

    # Function: Normalize tokens.
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
        # Function: Split directive.
        def split_directive(clause: str) -> tuple[Optional[str], str]:
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

        # Function: Compute score for phrase.
        def score_phrase(s: str) -> int:
            return len([t for t in norm_tokens(s) if t])

        aspect_penalties: Dict[str, List[str]] = {
            "details": ["structure", "rhythm"],
            "structure": ["detail", "details", "rhythm"],
            "rhythm": ["detail", "details", "structure"],
        }

        # Function: Compute aspect score.
        def aspect_score(aspect: str, phrase: str) -> float:
            base = float(score_phrase(phrase))
            pl = phrase.lower()
            penalty = 0.0
            for kw in aspect_penalties.get(aspect, []):
                if kw in pl:
                    penalty += 2.0
            return base - penalty

        # Function: Select best.
        def pick_best(existing: Optional[str], candidate: str, aspect: Optional[str] = None) -> str:
            if not existing:
                return candidate
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

        # Function: Extract segment.
        def extract_segment(phrase: str, keyword: str) -> str:
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


# Function: Normalize priority order.
def normalize_priority_order(value: Any, conf: Optional[Dict[str, Any]] = None) -> List[str]:
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


# Function: Merge avoid list into hints.
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

    # Function: Add item.
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

# Function: Get prompt value.
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

# Function: Extract a corpus archive into a destination directory.
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

# Function: Read a .docx file using python-docx.
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


# Function: Read text file.
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

# Function: Normalize text.
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


# Function: Strip base64 images.
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


# Function: Normalize heading text.
def normalize_heading_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# Function: Get heading at.
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


# Function: Check whether reference heading.
def is_reference_heading(text: str) -> bool:
    return normalize_heading_text(text) in REFERENCE_HEADINGS


# Function: Find reference sections.
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


# Function: Strip non voice sections.
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


# Function: Strip boilerplate sections.
def strip_boilerplate_sections(text: str) -> str:
    # Remove legal/publishing boilerplate sections and paragraphs.
    lines = text.splitlines()
    boiler_sections: List[Tuple[int, int]] = []
    i = 0
    while i < len(lines):
        heading = get_heading_at(lines, i)
        if heading:
            level, title, span = heading
            if BOILERPLATE_HEADING_RE.search(title):
                start = i
                end = len(lines)
                j = i + span
                while j < len(lines):
                    next_h = get_heading_at(lines, j)
                    if next_h and next_h[0] <= level:
                        end = j
                        break
                    j += 1
                boiler_sections.append((start, end))
                i = end
                continue
        i += 1
    if boiler_sections:
        boiler_iter = iter(boiler_sections)
        current = next(boiler_iter, None)
        pruned: List[str] = []
        idx = 0
        while idx < len(lines):
            if current and idx == current[0]:
                idx = current[1]
                current = next(boiler_iter, None)
                continue
            pruned.append(lines[idx])
            idx += 1
        lines = pruned

    # Drop paragraphs that contain boilerplate lines.
    paragraphs: List[List[str]] = []
    current_para: List[str] = []
    for line in lines:
        if line.strip() == "":
            if current_para:
                paragraphs.append(current_para)
                current_para = []
            paragraphs.append([line])
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append(current_para)
    kept_lines: List[str] = []
    for para in paragraphs:
        if any(BOILERPLATE_LINE_RE.search(l) for l in para if l.strip()):
            continue
        kept_lines.extend(para)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Function: Strip fenced code blocks.
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


# Function: Strip inline code.
def strip_inline_code(text: str) -> str:
    # Remove inline code spans delimited by backticks.
    return re.sub(r"(``[^`\n]+``|`[^`\n]+`)", "", text)


# Function: Strip latex math.
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


# Function: Strip html.
def strip_html(text: str) -> str:
    # Remove HTML tags and block elements to exclude HTML from profiling.
    text = re.sub(r"(?is)<[A-Za-z][^>]*>.*?</[A-Za-z][^>]*>", "\n", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = HTML_TAG_RE.sub("", text)
    return text


# Function: Strip html entities.
def strip_html_entities(text: str) -> str:
    # Remove HTML entities (e.g., &nbsp;).
    return HTML_ENTITY_RE.sub("", text)


# Function: Find quote spans.
def _quote_spans(text: str) -> List[tuple[int, int, str]]:
    spans: List[tuple[int, int, str]] = []
    for match in QUOTE_SPAN_RE.finditer(text):
        spans.append((match.start(), match.end(), match.group(1)))
    for match in QUOTE_SPAN_CURLY_RE.finditer(text):
        spans.append((match.start(), match.end(), match.group(1)))
    spans.sort(key=lambda s: s[0])
    return spans


# Function: Check whether multiword quote.
def is_multiword_quote(inner: str) -> bool:
    return len(words(inner)) >= 2


# Function: Strip quoted passages.
def strip_quoted_passages(text: str) -> str:
    # Remove multi-word quoted passages (used for non-fiction).
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


# Function: Detect fiction from texts.
def detect_fiction_from_texts(
    texts: List[str],
    quote_span_min: int,
    quoted_ratio_min: float,
    quote_para_ratio_min: float,
    quoted_ratio_force: float
) -> bool:
    total_words = 0
    quoted_words = 0
    quote_spans = 0
    quote_para = 0
    total_para = 0
    for text in texts:
        total_words += len(words(text))
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


# Function: Check whether parenthetical citation.
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


# Function: Strip inline citations.
def strip_inline_citations(text: str) -> str:
    # Remove inline citation markers while keeping surrounding prose.
    text = INLINE_FOOTNOTE_RE.sub("", text)
    text = INLINE_NUMERIC_CITE_RE.sub("", text)

    # Function: Replacement helper for data.
    def repl(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        return "" if is_parenthetical_citation(inner) else match.group(0)

    text = PAREN_GROUP_RE.sub(repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# Function: Filter author voice text.
def filter_author_voice_text(text: str) -> str:
    # Remove non-author voice segments and inline citations for measurements/excerpts.
    text = strip_fenced_code_blocks(text)
    text = strip_non_voice_sections(text)
    text = strip_boilerplate_sections(text)
    if QUOTE_MODE == "non-fiction":
        text = strip_quoted_passages(text)
    text = strip_inline_code(text)
    text = strip_latex_math(text)
    text = strip_html(text)
    text = strip_html_entities(text)
    text = BASE64_PLACEHOLDER_RE.sub("", text)
    text = strip_inline_citations(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# Function: Iterate over corpus texts.
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

        # Skip tiny/empty
        if len(txt) < 200:
            continue

        items.append((str(p.relative_to(root)), txt))
        count += 1
    return items


# Function: Extract title.
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


# Function: Build corpus documents.
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

# Function: Split text into sentences using the shared heuristic.
split_sentences = utils_split_sentences
# Function: Split text into paragraphs using the shared heuristic.
split_paragraphs = utils_split_paragraphs


# Function: Estimate tokens.
def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token.
    return max(1, (len(text) + 3) // 4)


# Function: Estimate tokens for messages.
def estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    # Add a small per-message overhead to approximate chat tokenization.
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4
    return total + 2

# Function: Detect likely English spelling variant in text.
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

# Function: Compute stylometric measurements for a text corpus.
def compute_measurements(
    texts: List[str],
    rare_words_limit: int | None = None,
    rare_words_limit_avoidance: int | None = None,
    common_words: Optional[Iterable[str]] = None,
    local_spelling_rules: Dict[str, Any] | None = None
) -> Dict[str, Any]:
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
    cap_counts = collections.Counter(t.lower() for t in w if t and t[0].isupper())

    # Rare-word signals: low-frequency tokens that the author rarely uses.
    roman_re = re.compile(r"^[ivxlcdm]+$")

    # Function: Check whether repeated token.
    def is_repeated_token(token: str) -> bool:
        # Filter pathological tokens like "chairmanchairman" that are almost always OCR or tokenization noise.
        if len(token) < 8:
            return False
        for k in range(3, (len(token) // 2) + 1):
            if len(token) % k != 0:
                continue
            if token == token[:k] * (len(token) // k):
                return True
        if len(token) % 2 == 0:
            half = len(token) // 2
            if token[:half] == token[half:]:
                return True
        return False
    # Function: Check whether token looks like concatenated artifacts.
    def is_likely_concatenation(token: str) -> bool:
        # Filter tokens that begin with unusually long consonant clusters (often OCR/glue artifacts).
        if len(token) < 12:
            return False
        vowels = set("aeiouy")
        prefix_len = 0
        for ch in token:
            if ch in vowels:
                break
            prefix_len += 1
        return prefix_len >= 5
    # Function: Check whether candidate rare.
    def is_candidate_rare(token: str) -> bool:
        # Keep rare-word signals focused on lexical content rather than digits, numerals, or artifacts.
        if token in stop:
            return False
        if len(token) < 4:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        # Roman numerals (e.g., "xxiii") are excluded to avoid skew from chapter/section markers.
        if roman_re.fullmatch(token) and len(token) >= 2:
            return False
        if is_repeated_token(token):
            return False
        if is_likely_concatenation(token):
            return False
        return token.isalpha()
    # Function: Check whether candidate common.
    def is_candidate_common(token: str) -> bool:
        if token in stop:
            return False
        if len(token) < 3:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        return token.isalpha()

    max_count = max(2, int(total_words * 0.0001))
    max_count = min(5, max_count)
    common_entries: List[Tuple[str, Optional[float], int]] = []
    has_freq = False
    for idx, entry in enumerate(common_words or [(w, None) for w in DEFAULT_COMMON_WORDS]):
        if isinstance(entry, tuple):
            word = entry[0] if entry else ""
            freq = entry[1] if len(entry) > 1 else None
        else:
            word = entry
            freq = None
        if isinstance(word, str):
            word = word.strip().lower()
        else:
            word = ""
        if not word:
            continue
        if freq is not None:
            has_freq = True
        common_entries.append((word, freq, idx))
    if has_freq:
        common_entries.sort(key=lambda x: (x[1] if x[1] is not None else float("-inf")), reverse=True)
    else:
        common_entries.sort(key=lambda x: x[2])
    rare_candidates = [
        (token, count) for token, count in token_counts.items()
        if count <= max_count and is_candidate_rare(token)
    ]
    common_absent_candidates = [
        (token, token_counts.get(token, 0), freq) for token, freq, _ in common_entries
        if is_candidate_common(token) and token_counts.get(token, 0) == 0
    ]
    # Function: Compute a stable hash for token ordering.
    def stable_token_hash(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

    # Function: Estimate proper-name likelihood for a token.
    def proper_name_likelihood(token: str) -> float:
        total = token_counts.get(token, 0)
        if total <= 0:
            return 0.0
        return cap_counts.get(token, 0) / total

    # Prefer low frequency and low proper-name likelihood, then stabilize ordering with a hash.
    rare_candidates.sort(key=lambda x: (x[1], proper_name_likelihood(x[0]), stable_token_hash(x[0])))
    # Round-robin by initial letter to reduce alphabetical clustering when counts tie.
    by_initial: Dict[str, List[Tuple[str, int]]] = {}
    for token, count in rare_candidates:
        initial = token[0] if token else "#"
        by_initial.setdefault(initial, []).append((token, count))
    initials = sorted(by_initial.keys())
    selected: List[Tuple[str, int]] = []
    while len(selected) < 40 and initials:
        next_initials = []
        for initial in initials:
            bucket = by_initial.get(initial, [])
            if bucket:
                selected.append(bucket.pop(0))
            if bucket and len(selected) < 40:
                next_initials.append(initial)
        initials = next_initials
    limit_signals = rare_words_limit if isinstance(rare_words_limit, int) and rare_words_limit > 0 else 100
    limit_avoid = rare_words_limit_avoidance if isinstance(rare_words_limit_avoidance, int) and rare_words_limit_avoidance > 0 else 100
    pool_size = max(limit_signals, limit_avoid) * 2
    candidate_pool = selected[:pool_size]
    candidate_pool.sort(key=lambda x: (proper_name_likelihood(x[0]), x[1], stable_token_hash(x[0])))
    filtered = candidate_pool[:max(limit_signals, limit_avoid)]
    # LLM ranking is handled during common-phrase validation to avoid extra calls here.
    rules = local_spelling_rules if isinstance(local_spelling_rules, dict) else load_local_spelling_rules()
    # Normalize rare-word signals to a US baseline so fingerprints are comparable across locales.
    spelling_map = build_local_spelling_map(rules, "us")
    normalized_counts: Dict[str, int] = {}
    normalized_order: List[str] = []
    for token, count in filtered[:limit_signals]:
        norm = normalize_text_spelling(token, spelling_map) if spelling_map else token
        if norm not in normalized_counts:
            normalized_counts[norm] = count
            normalized_order.append(norm)
        else:
            normalized_counts[norm] += count
    rare_words_signals = [
        {
            "word": token,
            "count": normalized_counts[token],
            "rate_per_1000w": approx_rate_per_1000_words(normalized_counts[token], total_words)
        }
        for token in normalized_order
    ]
    rare_words_avoid = []
    for token, count, freq in common_absent_candidates[:limit_avoid]:
        item = {
            "word": token,
            "count": count,
            "rate_per_1000w": approx_rate_per_1000_words(count, total_words)
        }
        if freq is not None:
            item["zipf_frequency"] = freq
        rare_words_avoid.append(item)
    # Function: Generate n-grams from tokens.
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
    transition_start_hits = 0
    transition_mid_hits = 0
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

    # Function: Check whether a sentence contains a marker.
    def sentence_has_marker(sentence: str, markers: List[str]) -> bool:
        s = sentence.lower()
        return any(m in s for m in markers)

    claim_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["claim"]))
    evidence_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["evidence"]))
    counter_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["counterpoint"]))
    concession_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["concession"]))
    synthesis_hits = sum(1 for s in all_sents if sentence_has_marker(s, rhetoric_markers["synthesis"]))

    # Paragraph cadence profile.
    opening_lens: List[int] = []
    closing_lens: List[int] = []
    for t in texts:
        for p in split_paragraphs(t):
            sents = split_sentences(p)
            if not sents:
                continue
            opening_lens.append(len(words(sents[0])))
            closing_lens.append(len(words(sents[-1])))

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
    parenthetical_hits = combined.count("(") + combined.count(")")
    appositive_hits = len(re.findall(r",\s+(?:a|an|the|which|who|that)\b", combined.lower()))

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

    sent_bins = [(0,9),(10,17),(18,25),(26,40),(41,None)]
    sent_hist = histogram(all_sent_lens, sent_bins)

    para_bins = [(1,1),(2,3),(4,5),(6,8),(9,None)]
    para_hist = histogram(para_lens, para_bins)

    # Self-echo repetition rates (bigrams/trigrams reused above a threshold).
    # Function: Compute the fraction of repeated n-grams above a minimum count.
    def repeat_rate(ngram_list: List[str], min_count: int = 3) -> float:
        if not ngram_list:
            return 0.0
        counts = collections.Counter(ngram_list)
        repeat_tokens = sum(c for _, c in counts.items() if c >= min_count)
        return repeat_tokens / max(1, len(ngram_list))

    bigrams_all = list(ngrams(2))
    trigrams_all = list(ngrams(3))

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
        "lexical_signals": {
            "rare_words": rare_words_signals,
            "rare_word_max_count": max_count,
            "rare_word_min_length": 4
        },
        "lexical_avoidance": {
            "category_rates_per_1000w": avoidance_rates,
            "rare_words": rare_words_avoid
        },
        "repetition": {
            "bigram_repeat_rate": repeat_rate(bigrams_all),
            "trigram_repeat_rate": repeat_rate(trigrams_all),
            "min_repeat_count": 3
        },
        "common_phrases": {
            "bigrams_top": [{"phrase": p, "count": c} for p, c in big],
            "trigrams_top": [{"phrase": p, "count": c} for p, c in tri]
        }
    }
    return measurements


# Function: Compute entropy for counts.
def _entropy_counts(counts: Dict[str, int]) -> float:
    total = sum(v for v in counts.values() if isinstance(v, int) and v > 0)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counts.values():
        if not isinstance(v, int) or v <= 0:
            continue
        p = v / total
        ent -= p * math.log(p, 2)
    return ent


# Function: Compute quantile for data.
def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


# Function: Compute rolling humanization baselines from a corpus.
def compute_humanization_baseline(
    texts: List[str],
    conf: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute windowed (rolling) variability baselines from the corpus itself.

    This is embedded into the fingerprint for auditability/control, but is intended
    to be stripped from what the LLM "sees" during rewriting.
    """
    conf = conf or {}
    enabled = bool(conf.get("enabled", True))
    if not enabled:
        return {"enabled": False, "notes": ["Disabled via tunables."]}

    window_words = int(conf.get("window_words", 800))
    stride_words = int(conf.get("stride_words", 400))
    min_window_words = int(conf.get("min_window_words", 250))
    max_windows = int(conf.get("max_windows", 200))

    window_words = max(50, window_words)
    stride_words = max(25, stride_words)
    min_window_words = max(50, min_window_words)
    max_windows = max(10, max_windows)

    # Paragraph-driven rolling windows preserve local punctuation and cadence signals.
    paras: List[str] = []
    para_words: List[int] = []
    for t in texts:
        for p in split_paragraphs(t):
            if not p.strip():
                continue
            wc = len(words(p))
            if wc <= 0:
                continue
            paras.append(p)
            para_words.append(wc)

    if not paras:
        return {
            "enabled": True,
            "windowing": {
                "window_words": window_words,
                "stride_words": stride_words,
                "min_window_words": min_window_words,
                "max_windows": max_windows,
                "windows_used": 0
            },
            "metrics": {},
            "notes": ["No usable paragraphs for baseline computation."]
        }

    # Function: Compute windowed metrics.
    def window_metrics(text: str) -> Dict[str, float]:
        toks = [w.lower() for w in words(text)]
        total_w = len(toks)

        sents = split_sentences(text)
        sent_lens = [len(words(s)) for s in sents] if sents else []
        sent_mean = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
        sent_stdev = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
        sent_burst = (sent_stdev / sent_mean) if sent_mean > 0 else 0.0

        ps = split_paragraphs(text)
        para_sent_counts = [len(split_sentences(p)) for p in ps] if ps else []
        para_mean = (sum(para_sent_counts) / len(para_sent_counts)) if para_sent_counts else 0.0
        para_stdev = statistics.pstdev(para_sent_counts) if len(para_sent_counts) > 1 else 0.0
        para_burst = (para_stdev / para_mean) if para_mean > 0 else 0.0
        one_sent_rate = (sum(1 for n in para_sent_counts if n == 1) / max(1, len(para_sent_counts))) if para_sent_counts else 0.0

        # Repetition (simple n-gram uniqueness ratio).
        big_total = max(0, len(toks) - 1)
        tri_total = max(0, len(toks) - 2)
        big_unique = len(set(zip(toks, toks[1:]))) if big_total else 0
        tri_unique = len(set(zip(toks, toks[1:], toks[2:]))) if tri_total else 0
        big_repeat = (1.0 - (big_unique / big_total)) if big_total > 0 else 0.0
        tri_repeat = (1.0 - (tri_unique / tri_total)) if tri_total > 0 else 0.0

        # Punctuation entropy/variety and density rates.
        punct_keys = [",", ";", ":", "!", "?", "(", ")", "\"", "'", "—"]
        punct_counts = {k: text.count(k) for k in punct_keys}
        punct_entropy = _entropy_counts({k: int(v) for k, v in punct_counts.items() if isinstance(v, int)})
        punct_variety = float(sum(1 for v in punct_counts.values() if isinstance(v, int) and v > 0))
        commas = int(punct_counts.get(",", 0))
        semicolons = int(punct_counts.get(";", 0))
        colons = int(punct_counts.get(":", 0))
        exclamations = int(punct_counts.get("!", 0))
        questions = int(punct_counts.get("?", 0))
        em_dashes = int(punct_counts.get("—", 0))
        per_1000 = 1000.0 / max(1.0, float(total_w))
        commas_per_1000w = commas * per_1000
        semicolons_per_1000w = semicolons * per_1000
        colons_per_1000w = colons * per_1000
        exclamations_per_1000w = exclamations * per_1000
        questions_per_1000w = questions * per_1000
        em_dashes_per_1000w = em_dashes * per_1000
        comma_density_per_100w = commas / max(1.0, float(total_w)) * 100.0

        # Char trigram entropy (orthographic texture).
        letters_only = re.sub(r"[^a-zA-Z]+", "", text.lower())
        trigram_counts: Dict[str, int] = {}
        if len(letters_only) >= 3:
            trigram_counts = dict(collections.Counter(letters_only[i:i+3] for i in range(len(letters_only) - 3 + 1)))
        char_tri_entropy = _entropy_counts({k: int(v) for k, v in trigram_counts.items()})

        # Lexical diversity + average word length.
        ttr = (len(set(toks)) / total_w) if total_w > 0 else 0.0
        avg_word_len = (sum(len(w) for w in toks) / total_w) if total_w > 0 else 0.0

        return {
            "token_count": float(total_w),
            "type_token_ratio": float(ttr),
            "avg_word_length": float(avg_word_len),
            "sentence_length_mean": float(sent_mean),
            "sentence_length_stdev": float(sent_stdev),
            "sentence_burstiness": float(sent_burst),
            "paragraph_length_mean": float(para_mean),
            "paragraph_length_stdev": float(para_stdev),
            "paragraph_burstiness": float(para_burst),
            "one_sentence_paragraph_rate": float(one_sent_rate),
            "bigram_repeat_rate": float(big_repeat),
            "trigram_repeat_rate": float(tri_repeat),
            "punctuation_variety": float(punct_variety),
            "punctuation_entropy": float(punct_entropy),
            "punctuation_commas_per_1000w": float(commas_per_1000w),
            "punctuation_semicolons_per_1000w": float(semicolons_per_1000w),
            "punctuation_colons_per_1000w": float(colons_per_1000w),
            "punctuation_exclamations_per_1000w": float(exclamations_per_1000w),
            "punctuation_questions_per_1000w": float(questions_per_1000w),
            "punctuation_em_dashes_per_1000w": float(em_dashes_per_1000w),
            "comma_density_per_100w": float(comma_density_per_100w),
            "char_trigram_entropy": float(char_tri_entropy),
        }

    # Sliding window over paragraphs with stride in words.
    metrics_series: Dict[str, List[float]] = {}
    start = 0
    windows_used = 0
    while start < len(paras) and windows_used < max_windows:
        end = start
        wsum = 0
        while end < len(paras) and wsum < window_words:
            wsum += para_words[end]
            end += 1
        if wsum < min_window_words:
            break
        wtext = "\n\n".join(paras[start:end]).strip()
        if not wtext:
            break
        wm = window_metrics(wtext)
        for k, v in wm.items():
            metrics_series.setdefault(k, []).append(float(v))
        windows_used += 1

        # Advance start by stride_words (approx) using paragraph word counts.
        adv = 0
        while start < len(paras) and adv < stride_words:
            adv += para_words[start]
            start += 1

    metrics_summary: Dict[str, Any] = {}
    for key, vals in metrics_series.items():
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        metrics_summary[key] = {
            "min": float(vals_sorted[0]),
            "max": float(vals_sorted[-1]),
            "mean": float(sum(vals_sorted) / len(vals_sorted)),
            "stdev": float(statistics.pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0),
            "p10": _quantile(vals_sorted, 0.10),
            "p25": _quantile(vals_sorted, 0.25),
            "p50": _quantile(vals_sorted, 0.50),
            "p75": _quantile(vals_sorted, 0.75),
            "p90": _quantile(vals_sorted, 0.90),
            "n": len(vals_sorted)
        }

    return {
        "enabled": True,
        "windowing": {
            "window_words": window_words,
            "stride_words": stride_words,
            "min_window_words": min_window_words,
            "max_windows": max_windows,
            "windows_used": windows_used
        },
        "metrics": metrics_summary,
        "notes": [
            "Baseline metrics are computed over rolling paragraph windows to capture natural within-author variability.",
            "These are intended for controller logic and auditability, not for direct LLM prompting."
        ]
    }


# Function: Compute token capitalization ratios.
def compute_token_capitalization_ratios(
    texts: List[str],
    token_set: set[str]
) -> Dict[str, float]:
    totals: Dict[str, int] = {t: 0 for t in token_set}
    caps: Dict[str, int] = {t: 0 for t in token_set}
    if not token_set:
        return {}
    for text in texts:
        for tok in re.findall(r"[A-Za-z][A-Za-z'-]*", text):
            key = tok.lower()
            if key not in totals:
                continue
            totals[key] += 1
            if tok[0].isupper():
                caps[key] += 1
    ratios: Dict[str, float] = {}
    for t in token_set:
        total = totals.get(t, 0)
        if total > 0:
            ratios[t] = caps.get(t, 0) / total
    return ratios


# Function: Select representative excerpts.
def pick_representative_excerpts(files_and_texts: List[Tuple[str, str]], max_total_chars: int) -> List[Dict[str, str]]:
    """
    Pick excerpts from multiple files to show the LLM real style.
    We keep this bounded; the stats do most of the work.
    """
    excerpts: List[Dict[str, str]] = []
    used = 0

    # Function: Compute score for paragraph voice.
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
    timeout_seconds: int = 300
    extra_headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    max_prompt_tokens: int = 100000
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 20.0

# Function: Load config.
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
        timeout_seconds=int(data.get("timeout_seconds", 300)),
        extra_headers=dict(data.get("extra_headers", {})),
        max_prompt_tokens=int(data.get("max_prompt_tokens", max_tokens)),
        max_retries=int(data.get("max_retries", 6)),
        backoff_base_seconds=float(data.get("backoff_base_seconds", 2.0)),
        backoff_max_seconds=float(data.get("backoff_max_seconds", 20.0)),
    )

# Function: Call completions.
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
            try:
                content = data["choices"][0]["message"]["content"]
                if attempt > 0:
                    print(
                        f"LLM request succeeded after {attempt} retry(ies).",
                        file=sys.stderr
                    )
                return content, data.get("usage")
            except Exception:
                raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:2000]}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, RuntimeError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
            jitter = random.uniform(0, backoff * 0.2)
            sleep_s = backoff + jitter
            print(
                "LLM request failed "
                f"(attempt {attempt + 1}/{cfg.max_retries + 1}); "
                f"retrying in {sleep_s:.1f}s. Error: {exc}",
                file=sys.stderr
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {cfg.max_retries + 1} attempts: {last_err}")


# ----------------------------
# Prompting
# ----------------------------

# Function: Extract the fingerprint schema template from prompts.
def fingerprint_schema_template(prompts: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_prompt_value(prompts, "fingerprint", "schema_hint")
    if not isinstance(schema, dict):
        raise TypeError("prompts.fingerprint.schema_hint must be an object")
    return copy.deepcopy(schema)

# Function: Build fingerprint prompt.
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


# Function: Strip measurements from a fingerprint before merging.
def slim_fingerprint_for_merge(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(fingerprint)
    slim.pop("measurements", None)
    return slim


# Function: Build the LLM prompt for fingerprint merging.
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


# Function: Build the LLM prompt for phrase validation.
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


# Function: Extract LLM-ranked rare words from validation output.
def rank_rare_words_llm(
    validation_result: Dict[str, Any]
) -> List[str]:
    ranked = validation_result.get("ranked_rare_words") if isinstance(validation_result, dict) else None
    if not isinstance(ranked, list):
        return []
    seen: set[str] = set()
    deduped: List[str] = []
    for w in ranked:
        if not isinstance(w, str):
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(w)
    return deduped


# Function: Validate common phrases to remove OCR/citation noise.
def validate_common_phrases(
    cfg: LLMConfig,
    phrases: List[Dict[str, Any]],
    prompts: Dict[str, Any],
    rare_word_candidates: List[Dict[str, Any]] | None = None,
    token_cap_ratios: Dict[str, float] | None = None,
    entity_blacklist: List[str] | None = None
) -> Dict[str, Any]:
    # Ask the LLM to flag OCR/citation noise in common phrases.
    honorifics = {
        "mr", "mrs", "ms", "dr", "sir", "madam", "lord", "lady", "duke", "emir", "imam",
        "saint", "st", "president", "prime", "minister", "governor", "senator", "rep",
        "mp", "king", "queen", "prince", "princess"
    }

    month_names = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    }

    # Function: Check whether like date.
    def looks_like_date(phrase: str) -> bool:
        lowered = phrase.lower()
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", lowered)]
        if any(t in month_names for t in tokens):
            if any(t.isdigit() for t in tokens):
                return True
        # Numeric date patterns: 2024-06-20, 06/20/2024, 20.06.2024, 20240620
        if re.search(r"\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b", lowered):
            return True
        if re.search(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b", lowered):
            return True
        if re.search(r"\b\d{8}\b", lowered):
            return True
        # Ordinal dates like 13th, 2nd, 1st when paired with month names
        if any(t in month_names for t in tokens) and re.search(r"\b\d{1,2}(st|nd|rd|th)\b", lowered):
            return True
        return False

    entity_singles: set[str]
    entity_phrases: set[tuple[str, ...]]
    entity_max_len: int
    if entity_blacklist:
        entity_singles, entity_phrases, entity_max_len = build_entity_matcher(entity_blacklist)
    else:
        entity_singles, entity_phrases, entity_max_len = set(), set(), 1

    # Function: Check for entity.
    def contains_entity(phrase: str) -> bool:
        # Entity matching is optional; skip when no blacklist is loaded.
        if not entity_singles and not entity_phrases:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", phrase.lower())
        if not tokens:
            return False
        for tok in tokens:
            if tok in entity_singles:
                return True
        if not entity_phrases:
            return False
        max_len = min(entity_max_len, len(tokens))
        for start in range(0, len(tokens)):
            for length in range(2, max_len + 1):
                end = start + length
                if end > len(tokens):
                    break
                if tuple(tokens[start:end]) in entity_phrases:
                    return True
        return False

    # Function: Decide whether to drop proper name.
    def should_drop_proper_name(phrase: str) -> bool:
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]*", phrase)]
        if not tokens:
            return False
        if looks_like_date(phrase):
            return True
        if contains_entity(phrase):
            return True
        lower_tokens = [t.lower() for t in tokens]
        if any(t in honorifics for t in lower_tokens):
            return True
        cap_tokens = [t for t in tokens if t[0].isupper()]
        if len(cap_tokens) >= 2:
            return True
        if len(cap_tokens) == 1:
            # Drop single-capitalized token phrases unless it's a generic institution.
            generic = {"United", "States", "World", "Bank", "European", "Union", "Congress", "Parliament"}
            if cap_tokens[0] not in generic:
                return True
        if token_cap_ratios:
            high = [t for t in lower_tokens if token_cap_ratios.get(t, 0.0) >= 0.6]
            mid = [t for t in lower_tokens if token_cap_ratios.get(t, 0.0) >= 0.4]
            if len(high) >= 1:
                return True
            if len(mid) >= 2:
                return True
        return False

    filtered = []
    dropped = []
    for item in phrases:
        phrase = item.get("phrase", "") if isinstance(item, dict) else ""
        if isinstance(phrase, str) and should_drop_proper_name(phrase):
            dropped.append({
                "phrase": phrase,
                "ngram": item.get("ngram"),
                "count": item.get("count"),
                "reason": "Dropped by prefilter: likely proper name, entity name, or date."
            })
        else:
            filtered.append(item)
    messages = build_phrase_validation_prompt(filtered, prompts)
    if rare_word_candidates is not None:
        try:
            messages[1]["content"] = json.dumps(
                {
                    **json.loads(messages[1]["content"]),
                    "rare_word_candidates": rare_word_candidates
                },
                ensure_ascii=False
            )
        except Exception:
            pass
    result, _ = request_json_with_retries(cfg, messages, prompts, "phrase validation")
    if dropped:
        result.setdefault("decisions", [])
        for d in dropped:
            result["decisions"].append({
                "phrase": d["phrase"],
                "ngram": d["ngram"],
                "count": d["count"],
                "decision": "drop",
                "reason": d["reason"]
            })
    return result


# Function: Filter phrases that look like proper names or dates.
def filter_proper_phrase_candidates(
    phrases: List[Dict[str, Any]],
    token_cap_ratios: Dict[str, float] | None,
    entity_blacklist: List[str] | None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Deterministically drop phrases likely to be proper names, entities, or dates.
    honorifics = {
        "mr", "mrs", "ms", "dr", "sir", "madam", "lord", "lady", "duke", "emir", "imam",
        "saint", "st", "president", "prime", "minister", "governor", "senator", "rep",
        "mp", "king", "queen", "prince", "princess"
    }
    month_names = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    }

    # Function: Check whether like date.
    def looks_like_date(phrase: str) -> bool:
        lowered = phrase.lower()
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", lowered)]
        if any(t in month_names for t in tokens) and any(t.isdigit() for t in tokens):
            return True
        if re.search(r"\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b", lowered):
            return True
        if re.search(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b", lowered):
            return True
        if re.search(r"\b\d{8}\b", lowered):
            return True
        if any(t in month_names for t in tokens) and re.search(r"\b\d{1,2}(st|nd|rd|th)\b", lowered):
            return True
        return False

    entity_singles: set[str]
    entity_phrases: set[tuple[str, ...]]
    entity_max_len: int
    if entity_blacklist:
        entity_singles, entity_phrases, entity_max_len = build_entity_matcher(entity_blacklist)
    else:
        entity_singles, entity_phrases, entity_max_len = set(), set(), 1

    # Function: Check for entity.
    def contains_entity(phrase: str) -> bool:
        if not entity_singles and not entity_phrases:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", phrase.lower())
        if not tokens:
            return False
        for tok in tokens:
            if tok in entity_singles:
                return True
        if not entity_phrases:
            return False
        max_len = min(entity_max_len, len(tokens))
        for start in range(0, len(tokens)):
            for length in range(2, max_len + 1):
                end = start + length
                if end > len(tokens):
                    break
                if tuple(tokens[start:end]) in entity_phrases:
                    return True
        return False

    # Function: Decide whether to drop.
    def should_drop(phrase: str) -> bool:
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]*", phrase)]
        if not tokens:
            return False
        if looks_like_date(phrase):
            return True
        if contains_entity(phrase):
            return True
        lower_tokens = [t.lower() for t in tokens]
        if any(t in honorifics for t in lower_tokens):
            return True
        cap_tokens = [t for t in tokens if t[0].isupper()]
        if len(cap_tokens) >= 2:
            return True
        if len(cap_tokens) == 1:
            generic = {"United", "States", "World", "Bank", "European", "Union", "Congress", "Parliament"}
            if cap_tokens[0] not in generic:
                return True
        if token_cap_ratios:
            high = [t for t in lower_tokens if token_cap_ratios.get(t, 0.0) >= 0.6]
            mid = [t for t in lower_tokens if token_cap_ratios.get(t, 0.0) >= 0.4]
            if len(high) >= 1:
                return True
            if len(mid) >= 2:
                return True
        return False

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for item in phrases:
        phrase = item.get("phrase", "") if isinstance(item, dict) else ""
        if isinstance(phrase, str) and should_drop(phrase):
            dropped.append(item)
            continue
        kept.append(item)
    return kept, dropped


# Function: Chunk excerpts.
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


# Function: Extract JSON candidate.
def _extract_json_candidate(text: str) -> str | None:
    # Extract the first complete JSON object/array from a string.
    start = None
    stack: List[str] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{" or ch == "[":
                start = i
                stack.append(ch)
            continue
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == "\"":
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{" or ch == "[":
            stack.append(ch)
        elif ch == "}" or ch == "]":
            if not stack:
                continue
            open_ch = stack.pop()
            if not stack:
                return text[start:i + 1]
    return None


# Function: Parse JSON strict.
def parse_json_strict(s: str) -> Dict[str, Any]:
    # Strip code fences if present and parse strictly as JSON.
    s = s.strip()
    # Some models wrap in ```json ... ```; strip that if present.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        candidate = _extract_json_candidate(s)
        if candidate:
            return json.loads(candidate)
        raise

# Function: Repair JSON with LLM.
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


# Function: Call the LLM with retries and JSON parsing.
def request_json_with_retries(
    cfg: LLMConfig,
    messages: List[Dict[str, str]],
    prompts: Dict[str, Any],
    purpose: str
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    last_err: Exception | None = None
    last_raw = ""
    for attempt in range(cfg.max_retries + 1):
        try:
            raw, usage = chat_completions(cfg, messages)
            last_raw = raw
            try:
                result = parse_json_strict(raw)
            except Exception:
                result = repair_json_with_llm(cfg, raw, prompts)
            if isinstance(result, dict) and result:
                if attempt > 0:
                    print_warn(f"LLM output recovered after {attempt} retry(ies) ({purpose}).")
                return result, usage
            last_err = RuntimeError("LLM returned empty or invalid JSON")
        except Exception as exc:
            last_err = exc
        if attempt >= cfg.max_retries:
            break
        backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2 ** attempt))
        jitter = random.uniform(0, backoff * 0.2)
        sleep_s = backoff + jitter
        print_warn(
            f"LLM output invalid ({purpose}) "
            f"(attempt {attempt + 1}/{cfg.max_retries + 1}); "
            f"retrying in {sleep_s:.1f}s. Error: {last_err}"
        )
        time.sleep(sleep_s)
    raise RuntimeError(f"LLM output invalid after {cfg.max_retries + 1} attempts ({purpose}): {last_err}")


# ----------------------------
# Main
# ----------------------------

# Function: CLI entry point.
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
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fiction",
        action="store_true",
        help="Treat input as fiction (quoted passages are part of author voice)."
    )
    mode_group.add_argument(
        "--non-fiction",
        dest="non_fiction",
        action="store_true",
        help="Treat input as non-fiction (multi-word quotes excluded from author voice)."
    )
    ap.add_argument(
        "--license",
        action="store_true",
        help="Print LICENSE.md and exit"
    )
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--max-bytes-per-file", type=int, default=DEFAULT_MAX_BYTES_PER_FILE)
    ap.add_argument("--excerpt-char-budget", type=int, default=DEFAULT_MAX_TOTAL_CHARS_FOR_LLM)
    args = ap.parse_args()

    if args.out.suffix == "":
        args.out = args.out.with_suffix(".json")

    if args.config is None:
        args.config = resolve_required_path(args.config, "config.llm.json", __file__)

    # Function: Verbose-print when enabled.
    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    vprint(f"Using config: {args.config}")
    vprint(f"Output path: {args.out}")
    # print("Use --license to view license.")

    cfg = load_config(args.config)
    prompts = load_prompts()
    tunables_snapshot = load_tunables_snapshot()
    lexicon_hints = load_optional_lexicon_hints()
    avoid_list = load_avoid_list()
    common_words = load_common_words()
    entity_blacklist = load_entity_blacklist()
    local_spelling_rules = load_local_spelling_rules()
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
        raw_texts = [t for _, t in files_and_texts]
        fiction_conf = {}
        if isinstance(tunables_snapshot, dict):
            fiction_conf = tunables_snapshot.get("fiction_detection", {}) or {}
        quote_span_min = int(fiction_conf.get("quote_span_min", 8))
        quoted_ratio_min = float(fiction_conf.get("quoted_ratio_min", 0.03))
        quote_para_ratio_min = float(fiction_conf.get("quote_para_ratio_min", 0.2))
        quoted_ratio_force = float(fiction_conf.get("quoted_ratio_force", 0.08))
        if args.fiction:
            fiction_mode = True
            print("Assuming fiction: quoted passages will be included in fingerprinting.")
        elif args.non_fiction:
            fiction_mode = False
            print("Assuming non-fiction: multi-word quotations will be excluded from fingerprinting.")
        else:
            fiction_mode = detect_fiction_from_texts(
                raw_texts,
                quote_span_min=quote_span_min,
                quoted_ratio_min=quoted_ratio_min,
                quote_para_ratio_min=quote_para_ratio_min,
                quoted_ratio_force=quoted_ratio_force
            )
            if fiction_mode:
                print("Detected fiction: quoted passages will be included in fingerprinting.")
            else:
                print("Detected non-fiction: multi-word quotations will be excluded from fingerprinting.")
        global QUOTE_MODE
        QUOTE_MODE = "fiction" if fiction_mode else "non-fiction"
        filtered_files_and_texts: List[Tuple[str, str]] = []
        for rel, txt in files_and_texts:
            filtered = filter_author_voice_text(txt)
            if len(filtered) < 200:
                continue
            filtered_files_and_texts.append((rel, filtered))
        if not filtered_files_and_texts:
            print("No usable corpus text after filtering.", file=sys.stderr)
            return 3
        texts = [t for _, t in filtered_files_and_texts]
        rare_words_limit = None
        rare_words_limit_avoid = None
        if isinstance(tunables_snapshot, dict):
            lex = tunables_snapshot.get("lexical_signals", {})
            if isinstance(lex, dict):
                limit = lex.get("rare_words_limit")
                if isinstance(limit, int) and limit > 0:
                    rare_words_limit = limit
            lex_avoid = tunables_snapshot.get("lexical_avoidance", {})
            if isinstance(lex_avoid, dict):
                limit_avoid = lex_avoid.get("rare_words_limit")
                if isinstance(limit_avoid, int) and limit_avoid > 0:
                    rare_words_limit_avoid = limit_avoid
        measurements = compute_measurements(
            texts,
            rare_words_limit=rare_words_limit,
            rare_words_limit_avoidance=rare_words_limit_avoid,
            common_words=common_words,
            local_spelling_rules=local_spelling_rules
        )
        # Deterministically filter sentence/transition openers to reduce proper-name noise.
        template_signals = measurements.get("templates_signals", {})
        opener_candidates: List[Dict[str, Any]] = []
        for item in template_signals.get("sentence_openers_top", []) or []:
            opener_candidates.append(item)
        for item in template_signals.get("transition_openers_top", []) or []:
            opener_candidates.append(item)
        if opener_candidates:
            token_set: set[str] = set()
            for item in opener_candidates:
                phrase = item.get("phrase", "")
                if isinstance(phrase, str):
                    for tok in re.findall(r"[A-Za-z][A-Za-z'-]*", phrase):
                        token_set.add(tok.lower())
            token_cap_ratios = compute_token_capitalization_ratios(texts, token_set)
            filtered_sentence, dropped_sentence = filter_proper_phrase_candidates(
                template_signals.get("sentence_openers_top", []) or [],
                token_cap_ratios,
                entity_blacklist
            )
            filtered_transition, dropped_transition = filter_proper_phrase_candidates(
                template_signals.get("transition_openers_top", []) or [],
                token_cap_ratios,
                entity_blacklist
            )
            measurements["templates_signals"]["sentence_openers_top"] = filtered_sentence
            measurements["templates_signals"]["transition_openers_top"] = filtered_transition
            # if dropped_sentence:
            #     measurements["templates_signals"]["sentence_openers_dropped"] = dropped_sentence
            # if dropped_transition:
            #     measurements["templates_signals"]["transition_openers_dropped"] = dropped_transition
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
                token_set: set[str] = set()
                for item in candidates:
                    phrase = item.get("phrase", "")
                    if isinstance(phrase, str):
                        for tok in re.findall(r"[A-Za-z][A-Za-z'-]*", phrase):
                            token_set.add(tok.lower())
                token_cap_ratios = compute_token_capitalization_ratios(texts, token_set)
                rare_candidates = measurements.get("lexical_signals", {}).get("rare_words", [])
                validation = validate_common_phrases(
                    cfg,
                    candidates,
                    prompts,
                    rare_word_candidates=rare_candidates,
                    token_cap_ratios=token_cap_ratios,
                    entity_blacklist=entity_blacklist
                )
                decisions = validation.get("decisions", []) or []
                decision_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
                for d in decisions:
                    phrase = d.get("phrase")
                    ngram = d.get("ngram")
                    if isinstance(phrase, str) and isinstance(ngram, int):
                        decision_map[(phrase, ngram)] = d
                ranked = rank_rare_words_llm(validation)
                if ranked:
                    word_to_item = {item.get("word"): item for item in rare_candidates if isinstance(item, dict)}
                    ordered = [word_to_item[w] for w in ranked if w in word_to_item]
                    remaining = [item for item in rare_candidates if item.get("word") not in set(ranked)]
                    measurements["lexical_signals"]["rare_words"] = ordered + remaining
                ranked_phrases = validation.get("ranked_phrases") if isinstance(validation, dict) else None
                keep_phrase_set: set[str] | None = None
                if isinstance(ranked_phrases, list) and ranked_phrases:
                    keep_count = max(1, len(ranked_phrases) // 2)
                    keep_phrase_set = set(
                        p for p in ranked_phrases[:keep_count] if isinstance(p, str)
                    )

                validated_bi: List[Dict[str, Any]] = []
                validated_tri: List[Dict[str, Any]] = []
                dropped: List[Dict[str, Any]] = []

                for item in common.get("bigrams_top", []) or []:
                    phrase = item.get("phrase", "")
                    decision = decision_map.get((phrase, 2))
                    if keep_phrase_set is not None and phrase not in keep_phrase_set:
                        dropped.append({
                            "phrase": phrase,
                            "count": item.get("count", 0),
                            "ngram": 2,
                            "reason": "Dropped by phrase ranking (likely proper name)."
                        })
                    elif decision and decision.get("decision") == "drop":
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
                    if keep_phrase_set is not None and phrase not in keep_phrase_set:
                        dropped.append({
                            "phrase": phrase,
                            "count": item.get("count", 0),
                            "ngram": 3,
                            "reason": "Dropped by phrase ranking (likely proper name)."
                        })
                    elif decision and decision.get("decision") == "drop":
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
                    "notes": validation.get("notes", [])
                }
                measurements["common_phrases"] = {
                    "bigrams_top": validated_bi,
                    "trigrams_top": validated_tri
                }
            else:
                measurements["common_phrases_validation"] = {
                    "validated": {"bigrams_top": [], "trigrams_top": []},
                    "dropped": [],
                    "notes": ["No common phrases to validate."]
                }
        vprint("Selecting representative excerpts...")
        excerpts = pick_representative_excerpts(filtered_files_and_texts, max_total_chars=args.excerpt_char_budget)

        vprint("Calling LLM to synthesize fingerprint...")
        messages = build_fingerprint_prompt(measurements, excerpts, cfg, prompts, lexicon_hints)
        prompt_tokens = estimate_tokens_for_messages(messages)
        if prompt_tokens <= cfg.max_prompt_tokens:
            fingerprint, usage = request_json_with_retries(cfg, messages, prompts, "fingerprint synthesis")
        else:
            vprint(f"Prompt too large ({prompt_tokens} tokens); chunking excerpts...")
            batches = chunk_excerpts(excerpts, measurements, cfg, cfg.max_prompt_tokens, prompts, lexicon_hints)
            vprint(f"Chunked into {len(batches)} excerpt batches.")
            partials: List[Dict[str, Any]] = []
            for idx, batch in enumerate(batches, start=1):
                vprint(f"Synthesizing partial fingerprint {idx}/{len(batches)}...")
                batch_messages = build_fingerprint_prompt(measurements, batch, cfg, prompts, lexicon_hints)
                partial, usage = request_json_with_retries(cfg, batch_messages, prompts, "partial fingerprint")
                if usage and args.verbose:
                    vprint(
                        f"Partial {idx}/{len(batches)} token usage: "
                        f"prompt={usage.get('prompt_tokens', 'n/a')}, "
                        f"completion={usage.get('completion_tokens', 'n/a')}, "
                        f"total={usage.get('total_tokens', 'n/a')}"
                    )
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
                fingerprint, usage = request_json_with_retries(cfg, merge_messages, prompts, "merge fingerprint")
                if usage and args.verbose:
                    vprint(
                        f"Merge {idx}/{len(partials)} token usage: "
                        f"prompt={usage.get('prompt_tokens', 'n/a')}, "
                        f"completion={usage.get('completion_tokens', 'n/a')}, "
                        f"total={usage.get('total_tokens', 'n/a')}"
                    )

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
            fingerprint["metadata"]["extraction"]["model"] = cfg.model
            fingerprint["metadata"]["extraction"].setdefault("methods", ["hybrid"])
            fingerprint["metadata"]["extraction"].setdefault("confidence", "medium")
            if tunables_snapshot is not None:
                fingerprint["metadata"]["extraction"].setdefault("tunables_snapshot", tunables_snapshot)

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
        # Embed corpus-derived windowed baselines for later controller logic. This is
        # *not* intended to be sent to the LLM during rewriting.
        baseline_conf = {}
        if isinstance(tunables_snapshot, dict):
            baseline_conf = tunables_snapshot.get("humanization_baseline", {}) or {}
        baseline = compute_humanization_baseline(texts, baseline_conf if isinstance(baseline_conf, dict) else None)
        if isinstance(fingerprint.get("measurements"), dict):
            fingerprint["measurements"]["humanization_baseline"] = baseline
        normalize_lexicon_avoids(fingerprint, measurements, lexicon_hints, avoid_list)
        normalize_lexicon_spelling(fingerprint, local_spelling_rules, avoid_list)
        controls = fingerprint.get("controls")
        controls_norm = {}
        if isinstance(tunables_snapshot, dict):
            controls_norm = tunables_snapshot.get("controls_normalization", {}) or {}
        if isinstance(controls, dict) and isinstance(controls.get("rewrite_policy"), str):
            rewrite_conf = controls_norm.get("rewrite_policy", {}) if isinstance(controls_norm, dict) else {}
            controls["rewrite_policy"] = normalize_rewrite_policy(controls["rewrite_policy"], rewrite_conf)
        if isinstance(controls, dict) and controls.get("priority_order") is not None:
            priority_conf = controls_norm.get("priority_order", {}) if isinstance(controls_norm, dict) else {}
            controls["priority_order"] = normalize_priority_order(controls.get("priority_order"), priority_conf)

        vprint("Writing fingerprint JSON...")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
        vprint("Done.")
        print(f"Wrote fingerprint JSON to: {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
