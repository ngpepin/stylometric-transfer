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

from utils import (
    approx_rate_per_1000_words,
    histogram,
    safe_mean,
    safe_stdev,
    split_paragraphs,
    split_sentences,
    words,
)


# ---- same lightweight stats as fingerprint script ----
# Script overview:
# - Measure the input Markdown (lightweight, explainable stats)
# - Build a prompt using an externalized template (prompts.json)
# - Call an OpenAI-compatible LLM to rewrite while preserving meaning
# - Score style compliance and optionally retry with delta feedback
# - Handle oversized prompts by chunking the Markdown
# - Strip base64 images before prompting, then reinsert after rewriting
# - Preserve non-voice blocks (blockquotes, references, footnotes, citations) verbatim

BASE64_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\\s]+", re.IGNORECASE)
BASE64_PLACEHOLDER_RE = re.compile(r"\[\[BASE64_IMAGE_\d+\]\]")
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
LICENSE_FILENAME = "LICENSE.md"
HUMANIZER_GUIDELINES_FILENAME = "general-guidelines.md"
HUMANIZER_CACHE_FILENAME = "humanizer_rules.cache.json"
TUNABLES_FILENAME = "config.tunables.json"
AVOID_LIST_FILENAME = "config.avoid.txt"
LLM_ROSTER_FILENAME = "config.llm.roster.json"
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
DEFAULT_MAX_STYLE_RETRIES = 1
PERPLEXITY_LEVELS = ("default", "low", "medium", "high", "extreme")
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
        "emoji_policy": "remove",
        "normalize_double_quotes": True,
        "normalize_single_quotes": True,
        "force_local_spelling_LLM": "none",
        "force_local_spelling_rules": "none",
        "heading_case_normalization": "by-level",
        "heading_case_by_level": {
            "h1": "title-case",
            "h2": "sentence-case",
            "h3": "identical",
            "h4": "automatic",
            "h5": "caps",
            "h6": "lower",
            "h7": "automatic",
            "h8": "automatic"
        },
        "preserve_proper_name_case": True
    },
    "perplexity_level": "default",
    "perplexity_profiles": {
        "default": {
            "humanizer_variance": {
                "max_ops_per_1000w": 0.5
            },
            "humanization_controller": {
                "quantiles": [0.25, 0.5, 0.75],
                "range_pct": 0.15
            },
            "chunking": {
                "max_input_tokens": 5750,
                "min_chunks_when_perturbing": 2
            },
            "llm": {
                "temperature_multiplier": 1.0
            }
        },
        "low": {
            "humanizer_variance": {
                "max_ops_per_1000w": 1.0
            },
            "humanization_controller": {
                "quantiles": [0.2, 0.5, 0.8],
                "range_pct": 0.2
            },
            "chunking": {
                "max_input_tokens": 5200,
                "min_chunks_when_perturbing": 3
            },
            "llm": {
                "temperature_multiplier": 1.0
            }
        },
        "medium": {
            "humanizer_variance": {
                "max_ops_per_1000w": 1.5
            },
            "humanization_controller": {
                "quantiles": [0.15, 0.5, 0.85],
                "range_pct": 0.25
            },
            "chunking": {
                "max_input_tokens": 4700,
                "min_chunks_when_perturbing": 4
            },
            "llm": {
                "temperature_multiplier": 1.0
            }
        },
        "high": {
            "humanizer_variance": {
                "max_ops_per_1000w": 2.0
            },
            "humanization_controller": {
                "quantiles": [0.1, 0.5, 0.9],
                "range_pct": 0.3
            },
            "chunking": {
                "max_input_tokens": 4200,
                "min_chunks_when_perturbing": 5
            },
            "llm": {
                "temperature_multiplier": 1.0
            }
        },
        "extreme": {
            "humanizer_variance": {
                "max_ops_per_1000w": 2.0
            },
            "humanization_controller": {
                "quantiles": [0.1, 0.5, 0.9],
                "range_pct": 0.3
            },
            "chunking": {
                "max_input_tokens": 4200,
                "min_chunks_when_perturbing": 5
            },
            "llm": {
                "temperature_multiplier": 2.0
            }
        }
    },
    "section_restore": {
        "enabled": True,
        "max_restore_sections": 20,
        "heading_similarity_threshold": 0.75,
        "signature_similarity_threshold": 0.6,
        "signature_min_overlap": 6
    },
    "postprocess_redundancy": {
        "enabled": False,
        "paragraph_dedupe": {
            "enabled": True,
            "min_words": 30,
            "similarity_threshold": 0.985,
            "lookback_blocks": 20,
            "max_drop_ratio": 0.15
        },
        "list_density": {
            "enabled": True,
            "min_run_length": 9,
            "group_size": 2,
            "joiner": "; "
        }
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
ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,8})\s+(.+?)\s*$")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
UNORDERED_LIST_LINE_RE = re.compile(r"^(\s*)([-*+])\s+(.*\S)\s*$")
SETEXT_H1_RE = re.compile(r"^\s*=+\s*$")
SETEXT_H2_RE = re.compile(r"^\s*-+\s*$")
BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>")
FOOTNOTE_DEF_RE = re.compile(r"^\s*\[\^[^\]]+\]:")
MAX_CONFIG_HEADING_LEVEL = 8

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


# Function: Load prompts.
def load_prompts() -> Dict[str, Any]:
    # Load externalized prompt templates located alongside this script.
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.json not found at {PROMPTS_PATH}")
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


# Function: Colorize text for terminal output.
def colorize(text: str, color: str, stream: Any) -> str:
    if os.getenv("NO_COLOR") or os.getenv("CLICOLOR") == "0":
        return text
    return f"{color}{text}{ANSI_RESET}"


# Function: Print warnings to stderr.
def print_warn(msg: str) -> None:
    print(colorize(msg, ANSI_YELLOW, sys.stderr), file=sys.stderr)


# Function: Print errors to stderr.
def print_error(msg: str) -> None:
    print(colorize(msg, ANSI_RED, sys.stderr), file=sys.stderr)


# Function: Deep-merge nested dictionaries.
def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


# Function: Load emoji substitutions.
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


# Function: Get emoji substitutions.
def get_emoji_substitutions() -> list[tuple[str, str]]:
    global EMOJI_SUBSTITUTIONS
    if EMOJI_SUBSTITUTIONS is None:
        EMOJI_SUBSTITUTIONS = load_emoji_substitutions(Path(__file__).resolve().parent)
    return EMOJI_SUBSTITUTIONS


# Function: Load tunables config from disk, with fallbacks.
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


# Function: Normalize configured perplexity level token.
def normalize_perplexity_level(value: Any) -> str:
    level = str(value).strip().lower()
    if level not in PERPLEXITY_LEVELS:
        return "default"
    return level


# Function: Resolve and apply perplexity profile overrides.
def apply_perplexity_profile(
    tunables: Dict[str, Any],
    level_override: str | None = None,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    if not isinstance(tunables, dict):
        return dict(DEFAULT_TUNABLES), "default", {}
    configured_level = normalize_perplexity_level(tunables.get("perplexity_level", "default"))
    effective_level = normalize_perplexity_level(level_override) if level_override else configured_level

    profiles = tunables.get("perplexity_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    profile = profiles.get(effective_level, {})
    if not isinstance(profile, dict):
        profile = {}
    merged = deep_merge_dict(tunables, profile)
    merged["perplexity_level"] = effective_level

    hv_conf = merged.get("humanizer_variance", {}) if isinstance(merged.get("humanizer_variance", {}), dict) else {}
    hc_conf = merged.get("humanization_controller", {}) if isinstance(merged.get("humanization_controller", {}), dict) else {}
    chunk_conf = merged.get("chunking", {}) if isinstance(merged.get("chunking", {}), dict) else {}
    llm_conf = merged.get("llm", {}) if isinstance(merged.get("llm", {}), dict) else {}
    knob_values = {
        "humanizer_variance.max_ops_per_1000w": hv_conf.get("max_ops_per_1000w"),
        "humanization_controller.quantiles": hc_conf.get("quantiles"),
        "humanization_controller.range_pct": hc_conf.get("range_pct"),
        "chunking.max_input_tokens": chunk_conf.get("max_input_tokens"),
        "chunking.min_chunks_when_perturbing": chunk_conf.get("min_chunks_when_perturbing"),
        "llm.temperature_multiplier": llm_conf.get("temperature_multiplier", 1.0),
    }
    return merged, effective_level, knob_values


# Function: Apply perplexity-driven temperature multiplier.
def apply_temperature_multiplier(base_temperature: float, multiplier: Any) -> float:
    try:
        parsed_base = float(base_temperature)
    except (TypeError, ValueError):
        parsed_base = 0.2
    try:
        parsed_multiplier = float(multiplier)
    except (TypeError, ValueError):
        parsed_multiplier = 1.0
    if parsed_multiplier < 0:
        parsed_multiplier = 0.0
    return max(0.0, min(2.0, parsed_base * parsed_multiplier))


# Function: Extract a --query argument from raw argv.
def extract_query_arg(argv: List[str]) -> str | None:
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--query":
            if idx + 1 >= len(argv):
                return ""
            return argv[idx + 1]
        if arg.startswith("--query="):
            return arg.split("=", 1)[1]
        idx += 1
    return None


# Function: Handle lightweight query mode.
def handle_query(query_arg: str) -> int:
    query = str(query_arg).strip().lower()
    if not query:
        print_error("Error: --query requires a value.")
        return 2
    if query == "perplexity":
        tunables = load_tunables(None)
        _, level, _ = apply_perplexity_profile(tunables, None)
        print(level)
        return 0
    print_error(f"Error: unsupported --query value: {query}")
    return 2


# Function: Resolve style/voice retry budgets from tunables and CLI.
def resolve_retry_budgets(
    style_retry_conf: Dict[str, Any] | None,
    cli_style_retries: int,
    cli_default_style_retries: int = DEFAULT_MAX_STYLE_RETRIES
) -> tuple[int, int]:
    # Style retries are CLI-overridable; voice retries can be independently capped.
    # Backward compatibility:
    # - style_retry.max_retries still works (applies to style, and to voice if voice cap absent)
    # - style_retry.voice_max_retries overrides only the voice loop cap.
    style_cap = max(0, int(cli_style_retries))
    if isinstance(style_retry_conf, dict):
        style_from_tunables = style_retry_conf.get("style_max_retries")
        if not isinstance(style_from_tunables, int):
            style_from_tunables = style_retry_conf.get("max_retries")
        if cli_style_retries == cli_default_style_retries and isinstance(style_from_tunables, int):
            style_cap = max(0, int(style_from_tunables))

        voice_from_tunables = style_retry_conf.get("voice_max_retries")
        if isinstance(voice_from_tunables, int):
            voice_cap = max(0, int(voice_from_tunables))
        else:
            # If no explicit voice cap exists, preserve old "same budget" behavior.
            voice_cap = style_cap
    else:
        voice_cap = style_cap
    return style_cap, voice_cap


# Function: Parse avoid list.
def parse_avoid_list(text: str) -> List[str]:
    items: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


# Function: Load avoid list.
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


# Function: Merge avoid list into fingerprint.
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
        lexicon["avoid_words"] = merged
    return fingerprint


# Function: Normalize rewrite policy text for display.
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

        # Function: Compute score for phrase.
        def score_phrase(s: str) -> int:
            return len([t for t in norm_tokens(s) if t])

        # Prefer phrases that describe the intended aspect without dragging in other aspects.
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

        # Function: Extract segment.
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


# Function: Normalize priority order.
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


# Function: Compute baseline quantile.
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


# Function: Clamp range.
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


# Function: Build per-chunk controller overlay targets.
def build_controller_overlay(
    fingerprint: Dict[str, Any],
    tunables: Dict[str, Any] | None,
    chunk_index: int | None,
    chunk_text: str,
    seed_override: int | None = None
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

    # Seed is deterministic per run, with a per-chunk offset to vary overlays across chunks.
    seed = int(conf.get("seed", 0))
    if seed_override is not None:
        seed = int(seed_override)
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

    # Function: Set target.
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


# Function: Compute overlay observations.
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


# Function: Build overlay feedback.
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


# Function: Compute variance aware factor.
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


# Function: Decide whether to forbid em dashes.
def should_forbid_em_dashes(tunables: Dict[str, Any] | None) -> bool:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    return bool(mandatory.get("avoid_em_dashes", False))


# Function: Decide whether to normalize double quotes.
def should_normalize_double_quotes(tunables: Dict[str, Any] | None) -> bool:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    return bool(mandatory.get("normalize_double_quotes", False))


# Function: Decide whether to normalize single quotes.
def should_normalize_single_quotes(tunables: Dict[str, Any] | None) -> bool:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    return bool(mandatory.get("normalize_single_quotes", True))


# Function: Read heading qualifier sanitization settings.
def get_heading_qualifier_sanitize_conf(
    tunables: Dict[str, Any] | None
) -> tuple[bool, List[str]]:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    conf = mandatory.get("sanitize_heading_qualifiers", False)
    if isinstance(conf, dict):
        enabled = conf.get("enabled", True)
        allowlist = conf.get("allowlist", [])
        if not isinstance(allowlist, list):
            allowlist = []
        allowlist = [str(item) for item in allowlist if isinstance(item, (str, int, float))]
        return bool(enabled), allowlist
    return bool(conf), []


# Function: Read heading case-normalization settings.
def get_heading_case_normalization_conf(
    tunables: Dict[str, Any] | None
) -> tuple[str, Dict[int, str], bool]:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    mode = str(mandatory.get("heading_case_normalization", "")).strip().lower()
    # Backward compatibility with removed boolean.
    if not mode and "allow_heading_case_changes" in mandatory:
        mode = "automatic" if bool(mandatory.get("allow_heading_case_changes")) else "identical"
    if mode not in {"automatic", "identical", "by-level"}:
        mode = "by-level"

    raw_by_level = mandatory.get("heading_case_by_level", {})
    normalized_by_level: Dict[int, str] = {}
    default_by_level = {
        1: "title-case",
        2: "sentence-case",
        3: "identical",
        4: "automatic",
        5: "caps",
        6: "lower",
        7: "automatic",
        8: "automatic",
    }
    allowed_level_modes = {"automatic", "identical", "unchanged", "title-case", "sentence-case", "caps", "lower"}
    mode_aliases = {
        "upper": "caps",
        "uppercase": "caps",
        "title": "title-case",
        "sentence": "sentence-case",
    }
    if isinstance(raw_by_level, dict):
        for key, value in raw_by_level.items():
            level = None
            k = str(key).strip().lower()
            if k.startswith("h"):
                k = k[1:]
            if k.isdigit():
                parsed = int(k)
                if 1 <= parsed <= MAX_CONFIG_HEADING_LEVEL:
                    level = parsed
            if level is None:
                continue
            v = str(value).strip().lower()
            v = mode_aliases.get(v, v)
            if v not in allowed_level_modes:
                continue
            normalized_by_level[level] = v
    by_level: Dict[int, str] = {lvl: normalized_by_level.get(lvl, default) for lvl, default in default_by_level.items()}
    preserve_proper_name_case = bool(mandatory.get("preserve_proper_name_case", True))
    return mode, by_level, preserve_proper_name_case


# Function: Normalize a locale spelling mode token.
def normalize_spelling_locale(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in ("none", "canadian", "british", "australian", "us"):
        return "none"
    return mode


# Function: Resolve split spelling controls (LLM instructions vs deterministic rules).
def get_force_local_spelling_conf(tunables: Dict[str, Any] | None) -> tuple[str, str]:
    mandatory = tunables.get("humanizer_mandatory", {}) if isinstance(tunables, dict) else {}
    legacy = normalize_spelling_locale(mandatory.get("force_local_spelling", "none"))
    llm_raw = mandatory.get("force_local_spelling_LLM", None)
    rules_raw = mandatory.get("force_local_spelling_rules", None)
    llm = normalize_spelling_locale(llm_raw) if llm_raw is not None else legacy
    rules = normalize_spelling_locale(rules_raw) if rules_raw is not None else legacy
    return llm, rules


# Function: Enforce no em dashes.
def enforce_no_em_dashes(text: str) -> tuple[str, int]:
    # Replace em dashes with a spaced hyphen to preserve readability without em-dash glyphs.
    if EM_DASH_CHAR not in text:
        return text, 0
    count = text.count(EM_DASH_CHAR)
    text = re.sub(r"\s*—\s*", " - ", text)
    return text, count


DOUBLE_QUOTE_CHARS = ("“", "”", "„", "‟", "«", "»")
SINGLE_QUOTE_CHARS = ("‘", "’", "‚", "‛", "‹", "›")
LOCAL_SPELLING_RULES_FILENAME = "config.local_spelling_rules.json"


# Function: Enforce straight double quotes.
def enforce_straight_double_quotes(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    count = sum(text.count(ch) for ch in DOUBLE_QUOTE_CHARS)
    if count == 0:
        return text, 0
    trans = {ord(ch): ord('"') for ch in DOUBLE_QUOTE_CHARS}
    return text.translate(trans), count


# Function: Enforce straight single quotes while leaving backticks unchanged.
def enforce_straight_single_quotes(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    count = sum(text.count(ch) for ch in SINGLE_QUOTE_CHARS)
    if count == 0:
        return text, 0
    trans = {ord(ch): ord("'") for ch in SINGLE_QUOTE_CHARS}
    return text.translate(trans), count


HEADING_QUALIFIER_PREFIXES = (
    "why",
    "how",
    "what",
    "when",
    "where",
    "defined",
    "definition",
    "quick",
    "brief",
    "overview",
    "context",
    "rationale",
    "motivation",
    "intuition",
    "notes",
    "note",
    "summary",
    "takeaway",
    "key",
    "guide",
    "guidance",
    "in practice",
    "in theory",
    "in brief",
    "in short",
    "quick fix",
    "quick fixes",
    "example",
    "examples",
    "case study",
    "case studies"
)
HEADING_QUALIFIER_MAX_WORDS = 10
HEADING_QUALIFIER_MIN_WORDS = 2


# Function: Count words in a heading after stripping lightweight markup.
def heading_word_count(text: str) -> int:
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    cleaned = re.sub(r"[`*_~]", "", cleaned)
    return len(words(cleaned))


# Function: Detect trailing qualifier text in headings.
def looks_like_heading_qualifier(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.search(r"\d", stripped):
        return False
    word_count = len(words(stripped))
    if word_count == 0 or word_count > HEADING_QUALIFIER_MAX_WORDS:
        return False
    lower = stripped.lower()
    if any(lower.startswith(prefix) for prefix in HEADING_QUALIFIER_PREFIXES):
        return True
    return stripped[0].islower()


# Function: Remove trailing heading qualifiers when safe.
def strip_heading_qualifier(heading_text: str) -> str:
    if not heading_text.strip():
        return heading_text
    base_text = heading_text.strip()
    paren_match = re.match(r"^(?P<base>.+?)\s*\((?P<qual>[^()]+)\)\s*$", base_text)
    if paren_match:
        base = paren_match.group("base").strip()
        qualifier = paren_match.group("qual").strip()
        if looks_like_heading_qualifier(qualifier) and heading_word_count(base) >= HEADING_QUALIFIER_MIN_WORDS:
            return base
    comma_match = re.match(r"^(?P<base>.+?),\s+(?P<qual>[^,]+)$", base_text)
    if comma_match:
        base = comma_match.group("base").strip()
        qualifier = comma_match.group("qual").strip()
        if looks_like_heading_qualifier(qualifier) and heading_word_count(base) >= HEADING_QUALIFIER_MIN_WORDS:
            return base
    return heading_text


# Function: Compile heading allowlist regex patterns.
def compile_heading_allowlist(patterns: List[str]) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        if not pattern:
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    return compiled


# Function: Sanitize trailing qualifiers in Markdown headings.
def enforce_heading_qualifiers(
    text: str,
    allowlist: List[re.Pattern] | None = None
) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return text, 0
    in_code = False
    fence = ""
    updated: List[str] = []
    changes = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_code:
                in_code = True
                fence = stripped[:3]
            else:
                if stripped.startswith(fence):
                    in_code = False
                    fence = ""
            updated.append(line)
            i += 1
            continue
        if in_code:
            updated.append(line)
            i += 1
            continue
        heading_info = get_heading_at(lines, i)
        if not heading_info:
            updated.append(line)
            i += 1
            continue
        level, heading_text, span = heading_info
        if allowlist:
            if any(regex.search(heading_text) for regex in allowlist):
                updated.append(line)
                if span > 1:
                    updated.extend(lines[i + 1:i + span])
                i += span
                continue
        sanitized = strip_heading_qualifier(heading_text)
        if sanitized != heading_text:
            if span == 1:
                m = ATX_HEADING_RE.match(line)
                if m:
                    prefix = m.group(1)
                    updated.append(f"{prefix} {sanitized}")
                else:
                    updated.append(line)
            else:
                updated.append(sanitized)
                updated.extend(lines[i + 1:i + span])
            changes += 1
        else:
            updated.append(line)
            if span > 1:
                updated.extend(lines[i + 1:i + span])
        i += span
    return "\n".join(updated), changes


# Function: Load local spelling rules.
def load_local_spelling_rules() -> Dict[str, Any]:
    rules_path = Path(__file__).resolve().parent / LOCAL_SPELLING_RULES_FILENAME
    if rules_path.exists():
        try:
            data = json.loads(rules_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


LOCAL_SPELLING_NOUN_VERB_LOCALES = {"canadian", "british", "australian"}


# Function: Normalize lexicon token.
def _normalize_lexicon_token(token: str) -> str:
    token = re.sub(r"[\"“”'’()\\[\\]{}]", "", str(token).lower()).strip()
    return re.sub(r"\\s+", " ", token)


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


# Function: Normalize tokens to US spelling for avoidance checks.
def normalize_tokens_for_avoidance(tokens: set[str], rules: Dict[str, Any]) -> set[str]:
    if not tokens:
        return set()
    # Avoidance checks use a US-normalized baseline so soft avoids remain stable across locales.
    us_map = build_local_spelling_map(rules, "us")
    if not us_map:
        return set(_normalize_lexicon_token(t) for t in tokens if t)
    normalized: set[str] = set()
    for token in tokens:
        base = _normalize_lexicon_token(token)
        if not base:
            continue
        normalized.add(us_map.get(base, base))
    return normalized


# Function: Apply case.
def _apply_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper() and template[1:].islower():
        return replacement.capitalize()
    if template.islower():
        return replacement.lower()
    return replacement


# Function: Check whether a context window contains keywords.
def _context_has(tokens: list[str], idx: int, keywords: list[str], window: int) -> bool:
    if not keywords:
        return False
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    window_tokens = set(tokens[start:end])
    return any(k in window_tokens for k in keywords)


# Function: Replace base word with a locale suffix variant.
def _replace_with_suffix(word: str, base_us: str, base_ca: str, suffixes: list[str]) -> tuple[str, bool]:
    lower = word.lower()
    for suffix in suffixes:
        if lower == base_us + suffix:
            return _apply_case(word, base_ca + suffix), True
    return word, False


# Function: Return spelling forms for base+suffix, including optional dropped-e inflection.
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


# Function: Heuristic verb detection from adjacent tokens.
def _probable_verb(prev_word: str, next_word: str) -> bool:
    prev = prev_word.lower()
    nextw = next_word.lower()
    aux = {
        "to", "can", "could", "may", "might", "must", "shall", "should",
        "will", "would", "do", "does", "did", "don't", "doesn't", "didn't",
        "cannot", "can't", "won't", "shalln't", "shouldn't", "wouldn't", "couldn't"
    }
    return prev in aux or nextw in {"to", "me", "him", "her", "them", "it"}


# Function: Heuristic noun detection from adjacent tokens.
def _probable_noun(prev_word: str, next_word: str) -> bool:
    prev = prev_word.lower()
    det = {
        "a", "an", "the", "this", "that", "these", "those", "my", "your",
        "his", "her", "our", "their", "its"
    }
    if prev in det or prev.endswith("'s"):
        return True
    if next_word.lower() in {"of", "for", "in", "on"}:
        return True
    return False


# Function: Normalize practise license.
def _normalize_practise_license(word: str, prev_word: str, next_word: str) -> tuple[str, bool]:
    lower = word.lower()
    if lower in {"practice", "practise"}:
        if _probable_verb(prev_word, next_word):
            return _apply_case(word, "practise"), lower != "practise"
        if _probable_noun(prev_word, next_word):
            return _apply_case(word, "practice"), lower != "practice"
        return word, False
    if lower in {"license", "licence"}:
        if _probable_verb(prev_word, next_word):
            return _apply_case(word, "license"), lower != "license"
        if _probable_noun(prev_word, next_word):
            return _apply_case(word, "licence"), lower != "licence"
        return word, False
    return word, False


# Function: Apply locale spelling rules to text.
def enforce_local_spelling(text: str, locale: str, rules: Dict[str, Any]) -> tuple[str, int]:
    if not text:
        return text, 0
    locale = locale.lower()
    if locale == "none":
        return text, 0
    rules_conf = rules.get("rules", {}) if isinstance(rules, dict) else {}
    direct_variants = rules_conf.get("direct_variants", []) if isinstance(rules_conf.get("direct_variants", []), list) else []
    suffix_variants = rules_conf.get("suffix_variants", []) if isinstance(rules_conf.get("suffix_variants", []), list) else []
    context_variants = rules_conf.get("context_variants", []) if isinstance(rules_conf.get("context_variants", []), list) else []
    double_l_conf = rules_conf.get("double_l_inflection", {}) if isinstance(rules_conf.get("double_l_inflection", {}), dict) else {}
    double_l_bases = double_l_conf.get("bases", []) if isinstance(double_l_conf.get("bases", []), list) else []
    double_l_suffixes = double_l_conf.get("suffixes", []) if isinstance(double_l_conf.get("suffixes", []), list) else []

    direct_map: Dict[str, str] = {}
    for entry in direct_variants:
        variants = entry.get("variants") if isinstance(entry, dict) else None
        if not isinstance(variants, dict):
            continue
        target = variants.get(locale)
        if not isinstance(target, str):
            continue
        for _, variant in variants.items():
            if isinstance(variant, str):
                direct_map[variant.lower()] = target

    suffix_map: Dict[str, str] = {}
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
                if not isinstance(suffix, str):
                    continue
                src_regular, src_drop = _spelling_forms_for_suffix(variant_base, suffix)
                dst_regular, dst_drop = _spelling_forms_for_suffix(target_base, suffix)
                suffix_map[src_regular.lower()] = dst_regular
                if src_drop:
                    suffix_map[src_drop.lower()] = dst_drop or dst_regular

    word_re = re.compile(r"[A-Za-z][A-Za-z']+")
    matches = list(word_re.finditer(text))
    if not matches:
        return text, 0
    tokens = [m.group(0).lower() for m in matches]
    parts: list[str] = []
    last = 0
    replacements = 0

    # Function: Find the previous non-space character.
    def _prev_non_space(idx: int) -> str:
        j = idx - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        return text[j] if j >= 0 else ""

    # Function: Check whether sentence start.
    def _is_sentence_start(idx: int) -> bool:
        prev = _prev_non_space(idx)
        return prev == "" or prev in ".!?\n"

    # Function: Check whether mixed case.
    def _is_mixed_case(word: str) -> bool:
        if word.islower() or word.isupper():
            return False
        return not (word[:1].isupper() and word[1:].islower())

    # Function: Check whether like path.
    def _looks_like_path(start: int, end: int) -> bool:
        window = text[max(0, start - 6):min(len(text), end + 6)]
        if "://" in window or window.startswith(("./", "../", "~/")):
            return True
        if any(ch in window for ch in ("/", "\\")):
            return True
        prev = _prev_non_space(start)
        if prev == ".":
            back = text[max(0, start - 16):start]
            if "/" in back or "\\" in back:
                return True
        # Windows drive letter like C:\path
        back = text[max(0, start - 3):start]
        if len(back) >= 2 and back[-2].isalpha() and back[-1] == ":":
            return True
        return False

    # Function: Split possessive.
    def _split_possessive(word: str) -> tuple[str, str] | None:
        lowered = word.lower()
        if lowered.endswith(("'s", "’s")):
            return word[:-2], word[-2:]
        if lowered.endswith(("s'", "s’")) and len(word) > 2:
            return word[:-2], word[-2:]
        return None

    # Function: Attach a possessive suffix to a word.
    def _attach_possessive(replaced: str, suffix: str) -> str:
        return f"{replaced}{suffix}"
    # Build context-sensitive rules (e.g., noun/verb or domain-specific disambiguation).
    context_rules: list[dict[str, Any]] = []
    for entry in context_variants:
        if not isinstance(entry, dict):
            continue
        variants = entry.get("variants")
        if not isinstance(variants, dict):
            continue
        target = variants.get(locale)
        if not isinstance(target, str):
            continue
        rule = {
            "variants": {k: v for k, v in variants.items() if isinstance(v, str)},
            "target": target,
            "apply_if": entry.get("apply_if", {}),
            "avoid_if": entry.get("avoid_if", {}),
            "block_if": entry.get("block_if", {}),
            "window": int(entry.get("window", 6))
        }
        context_rules.append(rule)
    # Fast membership check so title-case words that are known spelling variants are still normalized.
    context_variant_tokens: set[str] = set()
    for rule in context_rules:
        variants = rule.get("variants")
        if isinstance(variants, dict):
            for variant in variants.values():
                if isinstance(variant, str):
                    context_variant_tokens.add(variant.lower())
    for idx, match in enumerate(matches):
        start, end = match.start(), match.end()
        word = match.group(0)
        parts.append(text[last:start])
        last = end
        # Skip placeholders, acronyms, URLs, and paths to avoid mangling identifiers.
        window = text[max(0, start - 3):min(len(text), end + 3)]
        if "__" in window or word.isupper() and len(word) > 2:
            parts.append(word)
            continue
        if any(ch in window for ch in ("/", "\\", "@")) or "http" in text[max(0, start - 8):start].lower():
            parts.append(word)
            continue
        if _looks_like_path(start, end):
            parts.append(word)
            continue
        if _is_mixed_case(word):
            parts.append(word)
            continue
        if word[:1].isupper() and not _is_sentence_start(start):
            lower_word = word.lower()
            if (
                lower_word not in direct_map
                and lower_word not in suffix_map
                and lower_word not in context_variant_tokens
            ):
                parts.append(word)
                continue
        # Avoid changing a word that is part of a larger alnum token (e.g., snake_case, hex IDs).
        prev_ch = text[start - 1] if start > 0 else ""
        next_ch = text[end] if end < len(text) else ""
        if (prev_ch.isalnum() or prev_ch == "_") or (next_ch.isalnum() or next_ch == "_"):
            parts.append(word)
            continue
        prev_word = matches[idx - 1].group(0) if idx > 0 else ""
        next_word = matches[idx + 1].group(0) if idx + 1 < len(matches) else ""
        if locale in LOCAL_SPELLING_NOUN_VERB_LOCALES:
            updated, changed = _normalize_practise_license(word, prev_word, next_word)
            if changed:
                replacements += 1
                parts.append(updated)
                continue
        elif locale == "us":
            lower = word.lower()
            if lower in {"licence", "licences"}:
                updated = _apply_case(word, lower.replace("ce", "se"))
                replacements += 1
                parts.append(updated)
                continue
            if lower in {"practise", "practises", "practised", "practising"}:
                updated = _apply_case(word, lower.replace("se", "ce"))
                replacements += 1
                parts.append(updated)
                continue
        poss = _split_possessive(word)
        base = poss[0] if poss else word
        suffix = poss[1] if poss else ""
        lower = base.lower()
        # Context-sensitive rules first (for ambiguous words like tire/tyre, practice/practise).
        applied_context = False
        blocked_context = False
        for rule in context_rules:
            variants = rule.get("variants", {})
            target = rule.get("target")
            if not isinstance(variants, dict) or not isinstance(target, str):
                continue
            variant_values = [v.lower() for v in variants.values() if isinstance(v, str)]
            if lower not in variant_values:
                continue
            apply_conf = rule.get("apply_if", {})
            avoid_conf = rule.get("avoid_if", {})
            block_conf = rule.get("block_if", {})
            window = int(rule.get("window", 6))
            apply_any = apply_conf.get("any", []) if isinstance(apply_conf, dict) else []
            avoid_any = avoid_conf.get("any", []) if isinstance(avoid_conf, dict) else []
            block_any = block_conf.get("any", []) if isinstance(block_conf, dict) else []
            apply_any = [str(k).lower() for k in apply_any if isinstance(k, (str, int, float))]
            avoid_any = [str(k).lower() for k in avoid_any if isinstance(k, (str, int, float))]
            block_any = [str(k).lower() for k in block_any if isinstance(k, (str, int, float))]
            if block_any and _context_has(tokens, idx, block_any, window):
                blocked_context = True
                break
            if apply_any and not _context_has(tokens, idx, apply_any, window):
                blocked_context = True
                continue
            if avoid_any and _context_has(tokens, idx, avoid_any, window):
                blocked_context = True
                break
            if lower != target.lower():
                updated = _apply_case(base, target)
                if suffix:
                    updated = _attach_possessive(updated, suffix)
                replacements += 1
                parts.append(updated)
                applied_context = True
            else:
                parts.append(word)
                applied_context = True
            break
        if blocked_context:
            parts.append(word)
            continue
        # Context rules take precedence over direct/suffix rules.
        if applied_context:
            continue
        # Direct variants are explicit word-for-word mappings.
        if lower in direct_map:
            updated = _apply_case(base, direct_map[lower])
            if suffix:
                updated = _attach_possessive(updated, suffix)
            replacements += 1
            parts.append(updated)
            continue
        # Suffix variants handle family patterns like -our/-or with shared suffixes.
        if lower in suffix_map:
            updated = _apply_case(base, suffix_map[lower])
            if suffix:
                updated = _attach_possessive(updated, suffix)
            replacements += 1
            parts.append(updated)
            continue
        # Double-l inflection handling (travelled vs traveled) for select locales.
        converted = False
        if double_l_bases and double_l_suffixes:
            for base in double_l_bases:
                if not isinstance(base, str):
                    continue
                base_lower = base.lower()
                for suffix in double_l_suffixes:
                    if not isinstance(suffix, str) or not suffix:
                        continue
                    us_form = base_lower + suffix
                    double_form = base_lower[:-1] + "ll" + suffix
                    if lower == us_form and locale in LOCAL_SPELLING_NOUN_VERB_LOCALES:
                        updated = _apply_case(word, double_form)
                        replacements += 1
                        parts.append(updated)
                        converted = True
                        break
                    if lower == double_form and locale == "us":
                        updated = _apply_case(word, us_form)
                        replacements += 1
                        parts.append(updated)
                        converted = True
                        break
                if converted:
                    break
        if converted:
            continue
        parts.append(word)
    parts.append(text[last:])
    return "".join(parts), replacements


# Function: Enforce local spelling while preserving protected regions.
def enforce_local_spelling_guarded(
    text: str,
    locale: str,
    rules: Dict[str, Any],
    preserve_multiword_quotes: bool = False,
) -> tuple[str, int]:
    # Guard deterministic post-processing so non-voice/citation/code regions remain verbatim.
    if not text or locale == "none":
        return text, 0
    guarded_text, base64_map = strip_base64_images(text)
    guarded_text, frozen_map = mask_non_voice_blocks(guarded_text)
    guarded_text, html_map = mask_html(guarded_text)
    guarded_text, math_map = mask_math_notation(guarded_text)
    guarded_text, entity_map = mask_html_entities(guarded_text)
    guarded_text, inline_code_map = mask_inline_code(guarded_text)
    quote_map: Dict[str, str] = {}
    if preserve_multiword_quotes:
        guarded_text, quote_map = mask_quoted_passages(guarded_text)
    guarded_text, citation_map = mask_inline_citations(guarded_text)
    guarded_text, replacements = enforce_local_spelling(guarded_text, locale, rules)
    guarded_text = restore_placeholders(guarded_text, citation_map)
    guarded_text = restore_placeholders(guarded_text, quote_map)
    guarded_text = restore_placeholders(guarded_text, inline_code_map)
    guarded_text = restore_placeholders(guarded_text, entity_map)
    guarded_text = restore_placeholders(guarded_text, math_map)
    guarded_text = restore_placeholders(guarded_text, html_map)
    guarded_text = restore_placeholders(guarded_text, frozen_map)
    guarded_text = restore_base64_images(guarded_text, base64_map, find_base64_placeholders(guarded_text))
    return guarded_text, replacements


# Function: Check whether heading or list line.
def is_heading_or_list_line(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith(">"):
        return True
    if re.match(r"([-*+]|\\d+[.)])\\s+", stripped):
        return True
    return False


# Function: Parse deterministic post-process settings for redundancy/list shaping.
def get_postprocess_redundancy_conf(tunables: Dict[str, Any] | None) -> Dict[str, Any]:
    conf: Dict[str, Any] = {
        "enabled": False,
        "paragraph_dedupe": {
            "enabled": True,
            "min_words": 30,
            "similarity_threshold": 0.985,
            "lookback_blocks": 20,
            "max_drop_ratio": 0.15,
        },
        "list_density": {
            "enabled": True,
            "min_run_length": 9,
            "group_size": 2,
            "joiner": "; ",
        },
    }
    if not isinstance(tunables, dict):
        return conf
    raw = tunables.get("postprocess_redundancy", {})
    if not isinstance(raw, dict):
        return conf
    conf["enabled"] = bool(raw.get("enabled", conf["enabled"]))
    pd_raw = raw.get("paragraph_dedupe", {})
    if isinstance(pd_raw, dict):
        pd_conf = conf["paragraph_dedupe"]
        pd_conf["enabled"] = bool(pd_raw.get("enabled", pd_conf["enabled"]))
        try:
            pd_conf["min_words"] = max(5, int(pd_raw.get("min_words", pd_conf["min_words"])))
        except (TypeError, ValueError):
            pass
        try:
            pd_conf["similarity_threshold"] = min(
                1.0,
                max(0.8, float(pd_raw.get("similarity_threshold", pd_conf["similarity_threshold"]))),
            )
        except (TypeError, ValueError):
            pass
        try:
            pd_conf["lookback_blocks"] = max(1, int(pd_raw.get("lookback_blocks", pd_conf["lookback_blocks"])))
        except (TypeError, ValueError):
            pass
        try:
            pd_conf["max_drop_ratio"] = min(
                0.5,
                max(0.01, float(pd_raw.get("max_drop_ratio", pd_conf["max_drop_ratio"]))),
            )
        except (TypeError, ValueError):
            pass
    ld_raw = raw.get("list_density", {})
    if isinstance(ld_raw, dict):
        ld_conf = conf["list_density"]
        ld_conf["enabled"] = bool(ld_raw.get("enabled", ld_conf["enabled"]))
        try:
            ld_conf["min_run_length"] = max(3, int(ld_raw.get("min_run_length", ld_conf["min_run_length"])))
        except (TypeError, ValueError):
            pass
        try:
            ld_conf["group_size"] = max(2, int(ld_raw.get("group_size", ld_conf["group_size"])))
        except (TypeError, ValueError):
            pass
        joiner = ld_raw.get("joiner", ld_conf["joiner"])
        if isinstance(joiner, str) and joiner:
            ld_conf["joiner"] = joiner
    return conf


# Function: Build a canonical prose fingerprint for near-duplicate detection.
def canonicalize_prose_block(text: str) -> str:
    collapsed = re.sub(r"`[^`]*`", " ", text)
    collapsed = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", collapsed)
    collapsed = re.sub(r"https?://\S+", " ", collapsed)
    collapsed = re.sub(r"[^a-zA-Z0-9\s]", " ", collapsed).lower()
    return " ".join(collapsed.split())


# Function: Decide whether a block is a safe prose candidate for deterministic dedupe.
def is_prose_dedupe_candidate(block: str, min_words: int) -> bool:
    if not block.strip() or is_code_block(block):
        return False
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    if ATX_HEADING_RE.match(lines[0].strip()):
        return False
    if any(LIST_LINE_RE.match(ln.strip()) for ln in lines):
        return False
    if any(ln.strip().startswith("|") for ln in lines):
        return False
    if len(words(block)) < min_words:
        return False
    if not re.search(r"[.!?]", block):
        return False
    return True


# Function: Remove near-duplicate prose blocks while preserving first occurrence.
def dedupe_redundant_prose_blocks(
    markdown: str,
    min_words: int = 30,
    similarity_threshold: float = 0.985,
    lookback_blocks: int = 20,
    max_drop_ratio: float = 0.15,
) -> tuple[str, int]:
    blocks = split_markdown_blocks(markdown)
    if not blocks:
        return markdown, 0
    kept: List[str] = []
    recent: List[str] = []
    dropped = 0
    max_drops = max(1, int(len(blocks) * max_drop_ratio))
    for block in blocks:
        if dropped >= max_drops or not is_prose_dedupe_candidate(block, min_words):
            kept.append(block)
            canon = canonicalize_prose_block(block)
            if canon:
                recent.append(canon)
                if len(recent) > lookback_blocks:
                    recent = recent[-lookback_blocks:]
            continue
        canon = canonicalize_prose_block(block)
        if not canon:
            kept.append(block)
            continue
        is_dup = False
        for prev in reversed(recent[-lookback_blocks:]):
            if canon == prev:
                is_dup = True
                break
            score = difflib.SequenceMatcher(None, canon, prev).ratio()
            if score >= similarity_threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(block)
        recent.append(canon)
        if len(recent) > lookback_blocks:
            recent = recent[-lookback_blocks:]
    merged = "\n\n".join(s.strip() for s in kept if s.strip()).strip()
    return merged, dropped


# Function: Reduce long unordered-list runs by grouping adjacent bullets.
def throttle_unordered_list_density(
    markdown: str,
    min_run_length: int = 9,
    group_size: int = 2,
    joiner: str = "; ",
) -> tuple[str, int, int]:
    blocks = split_markdown_blocks(markdown)
    if not blocks:
        return markdown, 0, 0
    out_blocks: List[str] = []
    run_count = 0
    merged_items = 0
    for block in blocks:
        if is_code_block(block):
            out_blocks.append(block)
            continue
        lines = block.splitlines()
        i = 0
        out_lines: List[str] = []
        while i < len(lines):
            match = UNORDERED_LIST_LINE_RE.match(lines[i])
            if not match:
                out_lines.append(lines[i])
                i += 1
                continue
            indent = match.group(1)
            run: List[str] = []
            j = i
            while j < len(lines):
                m = UNORDERED_LIST_LINE_RE.match(lines[j])
                if not m or m.group(1) != indent:
                    break
                run.append(m.group(3).strip())
                j += 1
            if len(run) < min_run_length:
                out_lines.extend(lines[i:j])
                i = j
                continue
            run_count += 1
            for k in range(0, len(run), group_size):
                chunk = run[k : k + group_size]
                if not chunk:
                    continue
                out_lines.append(f"{indent}- {joiner.join(chunk)}")
            merged_items += max(0, len(run) - ((len(run) + group_size - 1) // group_size))
            i = j
        out_blocks.append("\n".join(out_lines).strip("\n"))
    merged = "\n\n".join(s.strip() for s in out_blocks if s.strip()).strip()
    return merged, run_count, merged_items


# Function: Apply removed emoji punctuation.
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


# Function: Enforce emoji policy.
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


# Function: Apply bounded stochastic perturbations to text.
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

    # Function: Replace a transition token based on templates.
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

    # Function: Drop filler words from text.
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

# Function: Resolve general guidelines path.
def resolve_general_guidelines_path() -> Path | None:
    # Resolve optional humanizer guidelines from CWD or script directory.
    cwd_path = Path.cwd() / HUMANIZER_GUIDELINES_FILENAME
    script_path = Path(__file__).resolve().parent / HUMANIZER_GUIDELINES_FILENAME
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    return path


# Function: Load general guidelines.
def load_general_guidelines() -> tuple[str | None, Path | None]:
    path = resolve_general_guidelines_path()
    if not path:
        return None, None
    return path.read_text(encoding="utf-8"), path


# Function: Resolve license path.
def resolve_license_path() -> Path | None:
    # Resolve LICENSE.md from CWD or script directory.
    cwd_path = Path.cwd() / LICENSE_FILENAME
    script_path = Path(__file__).resolve().parent / LICENSE_FILENAME
    if cwd_path.exists():
        return cwd_path
    if script_path.exists():
        return script_path
    return None


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
        print_error(f"License file not found: {LICENSE_FILENAME}")
        return 2
    render_markdown(path.read_text(encoding="utf-8"))
    return 0

# Function: Get prompt value.
def get_prompt_value(prompts: Dict[str, Any], *path: str) -> Any:
    # Traverse a nested dict safely and fail fast if a key is missing.
    cur: Any = prompts
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing prompts key: {'.'.join(path)}")
        cur = cur[key]
    return cur

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


# Function: Detect meta/task-focused summaries.
def summary_is_meta(text: str) -> bool:
    lowered = text.lower()
    return any(pat in lowered for pat in META_SUMMARY_PATTERNS)


# Function: Decide when to request chunk summary generation.
def should_request_chunk_summary(
    attempt_no: int,
    summary_enabled: bool,
    is_last_chunk: bool,
    refresh_after_bad_first: bool = False,
) -> bool:
    if not summary_enabled or is_last_chunk:
        return False
    if attempt_no == 1:
        return True
    return bool(refresh_after_bad_first)


# Function: Normalize a chunk summary for continuity.
def normalize_summary(summary: str, max_words: int | None) -> str:
    if not isinstance(summary, str):
        return ""
    summary = " ".join(summary.split()).strip()
    if not summary:
        return ""
    # Deterministic cleanup: enforce prior-context phrasing and consistent tense for chaining.
    context_nouns = (
        "passage", "section", "chunk", "text", "document", "paper", "article",
        "report", "excerpt", "chapter", "part", "segment", "portion", "appendix",
        "material", "content", "discussion", "analysis", "overview", "summary"
    )
    noun_pattern = "|".join(context_nouns)
    # Replace prepositional references like "in this section".
    summary = re.sub(
        rf"\b(in|within|from|of|for|about)\s+(this|that|the current|the following|the above|the preceding)\s+({noun_pattern})\b",
        r"\1 the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    # Replace sentence-start references like "This section ..." or "The section ...".
    summary = re.sub(
        rf"(^|[.!?]\s+)(this|that|the current|the following|the above|the preceding|the)\s+({noun_pattern})\b",
        lambda m: f"{m.group(1)}the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    # Replace mid-sentence references like "the passage" or "this section".
    summary = re.sub(
        rf"\b(this|that|the current|the following|the above|the preceding)\s+({noun_pattern})\b",
        "the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    summary = re.sub(
        rf"\bthis\s+({noun_pattern})\b",
        "the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    summary = re.sub(
        rf"\bthe\s+above\s+({noun_pattern})\b",
        "the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    summary = re.sub(
        rf"\bthe\s+(?!previous\b)({noun_pattern})\b",
        "the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    # Normalize any lingering "the previous <noun>" to "the previous passage".
    summary = re.sub(
        rf"\bthe previous\s+({noun_pattern})\b",
        "the previous passage",
        summary,
        flags=re.IGNORECASE
    )
    summary = re.sub(r"\bthe previous passage is\b", "the previous passage was", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bthe previous passage are\b", "the previous passage were", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bin\s+this\s+summary\b", "in the previous passage", summary, flags=re.IGNORECASE)
    summary = re.sub(r"^the previous passage\b", "The previous passage", summary, flags=re.IGNORECASE)
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
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
        "includes": "included",
        "provides": "provided",
        "offers": "offered",
        "lays out": "laid out",
        "sets out": "set out",
        "summons": "summoned",
        "calls out": "called out",
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
        "calls": "called",
        "surveys": "surveyed",
        "reviews": "reviewed",
        "articulates": "articulated"
    }
    verb_pattern = "|".join(re.escape(k) for k in verb_map.keys())
    summary = re.sub(
        rf"\b(the previous passage)\s+({verb_pattern})\b",
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


# Function: Build fallback summary.
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


# Function: Build semantic fallback summary.
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


# Function: Detect fiction from text.
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

# Function: Strip base64 images.
def strip_base64_images(text: str) -> tuple[str, Dict[str, str]]:
    # Replace base64 images with placeholders to avoid prompt token blowups.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Replacement helper for data.
    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"[[BASE64_IMAGE_{counter}]]"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    stripped = BASE64_IMAGE_RE.sub(repl, text)
    return stripped, mapping


# Function: Restore base64 images.
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


# Function: Find base64 placeholders.
def find_base64_placeholders(text: str) -> List[str]:
    # Helper to detect placeholders in rewritten output.
    return re.findall(r"\[\[BASE64_IMAGE_\d+\]\]", text)


# Function: Normalize heading text.
def normalize_heading_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# Function: Get heading at.
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


# Function: Detect per-word case pattern.
def _word_case_pattern(word: str) -> str:
    if word.isupper():
        return "upper"
    if word.islower():
        return "lower"
    if word[:1].isupper() and word[1:].islower():
        return "capitalized"
    return "mixed"


# Function: Apply per-word case pattern.
def _apply_word_case_pattern(word: str, pattern: str) -> str:
    if pattern == "upper":
        return word.upper()
    if pattern == "lower":
        return word.lower()
    if pattern == "capitalized":
        return word[:1].upper() + word[1:].lower()
    return word


# Function: Detect likely proper-name token indices in heading text.
def _detect_proper_name_indices(text: str) -> set[int]:
    word_re = re.compile(r"[A-Za-z][A-Za-z']*")
    matches = list(word_re.finditer(text))
    if not matches:
        return set()
    stopwords = {
        "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "near", "nor",
        "of", "on", "or", "per", "the", "to", "vs", "via", "with", "without", "over", "under",
        "has", "have", "had", "is", "are", "was", "were", "be", "being", "been"
    }
    name_run_followers = {
        "has", "have", "had", "is", "are", "was", "were", "be", "being", "been",
        "does", "do", "did", "can", "could", "will", "would", "should", "may", "might", "must",
    }

    # Function: Check title-case token.
    def is_title_token(word: str) -> bool:
        return word[:1].isupper() and word[1:].islower()

    indices: set[int] = set()
    for idx, m in enumerate(matches):
        w = m.group(0)
        lower = w.lower()
        if lower in stopwords:
            continue
        if w.isupper() and len(w) > 1:
            indices.add(idx)
            continue
        if _word_case_pattern(w) == "mixed":
            indices.add(idx)
            continue
        if is_title_token(w):
            # Guard against false positives from Title Case headings by requiring a
            # name-like opening run ("John Black ...") followed by a verb/connective.
            if idx == 0:
                run_end = idx
                while run_end + 1 < len(matches):
                    nxt_word = matches[run_end + 1].group(0)
                    nxt_lower = nxt_word.lower()
                    if not is_title_token(nxt_word) or nxt_lower in stopwords:
                        break
                    run_end += 1
                if run_end > idx:
                    follower = matches[run_end + 1].group(0).lower() if run_end + 1 < len(matches) else ""
                    if follower in name_run_followers:
                        for j in range(idx, run_end + 1):
                            indices.add(j)
    return indices


# Function: Preserve source proper-name casing when rewriting heading case.
def _preserve_proper_name_case(source_heading: str, transformed_heading: str) -> str:
    word_re = re.compile(r"[A-Za-z][A-Za-z']*")
    src_matches = list(word_re.finditer(source_heading))
    dst_matches = list(word_re.finditer(transformed_heading))
    if not src_matches or not dst_matches:
        return transformed_heading
    name_indices = _detect_proper_name_indices(source_heading)
    if not name_indices:
        return transformed_heading
    out = transformed_heading
    for idx in sorted(name_indices):
        if idx >= len(src_matches) or idx >= len(dst_matches):
            break
        src_word = src_matches[idx].group(0)
        dst_match = dst_matches[idx]
        dst_word = dst_match.group(0)
        # Preserve casing only when both tokens are the same lexical item.
        # This avoids undoing deterministic spelling normalization
        # (e.g., Program -> Programme) when proper-name preservation is enabled.
        if src_word.lower() != dst_word.lower():
            continue
        out = out[:dst_match.start()] + src_word + out[dst_match.end():]
        # Refresh matches because replacement may shift offsets.
        dst_matches = list(word_re.finditer(out))
    return out


# Function: Apply an explicit case style to a heading.
def apply_heading_case_style(
    heading_text: str,
    mode: str,
    source_heading: str | None = None,
    preserve_proper_name_case: bool = True,
) -> str:
    mode = str(mode).strip().lower()
    if mode in {"upper", "uppercase"}:
        mode = "caps"
    elif mode == "title":
        mode = "title-case"
    elif mode == "sentence":
        mode = "sentence-case"
    if mode in {"automatic"}:
        return heading_text
    if mode in {"identical", "unchanged"}:
        if isinstance(source_heading, str):
            return transfer_heading_casing(source_heading, heading_text, preserve_proper_name_case)
        return heading_text

    word_re = re.compile(r"[A-Za-z][A-Za-z']*")
    if mode == "caps":
        transformed = heading_text.upper()
    elif mode == "lower":
        transformed = heading_text.lower()
    elif mode == "sentence-case":
        transformed = heading_text.lower()
        m = word_re.search(transformed)
        if m:
            transformed = transformed[:m.start()] + transformed[m.start():m.start() + 1].upper() + transformed[m.start() + 1:]
    elif mode == "title-case":
        small_words = {
            "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "nor", "of",
            "on", "or", "per", "the", "to", "vs", "via", "with"
        }
        matches = list(word_re.finditer(heading_text))
        total = len(matches)
        idx = 0

        # Function: Apply title casing to lexical tokens.
        def repl_title(match: re.Match[str]) -> str:
            nonlocal idx
            idx += 1
            word = match.group(0)
            lower = word.lower()
            if word.isupper() and len(word) > 1:
                return word
            if 1 < idx < total and lower in small_words:
                return lower
            return word[:1].upper() + word[1:].lower()

        transformed = word_re.sub(repl_title, heading_text)
    else:
        transformed = heading_text

    if preserve_proper_name_case and isinstance(source_heading, str):
        transformed = _preserve_proper_name_case(source_heading, transformed)
    return transformed


# Function: Transfer heading casing from source heading to rewritten heading text.
def transfer_heading_casing(
    source_heading: str,
    rewritten_heading: str,
    preserve_proper_name_case: bool = True
) -> str:
    # Prefer token-position case transfer; fallback to aligned-prefix transfer when token counts differ.
    word_re = re.compile(r"[A-Za-z][A-Za-z']*")
    src_matches = list(word_re.finditer(source_heading))
    dst_matches = list(word_re.finditer(rewritten_heading))
    if not dst_matches:
        return rewritten_heading

    if src_matches and len(src_matches) == len(dst_matches):
        patterns = [_word_case_pattern(m.group(0)) for m in src_matches]
        idx = 0

        # Function: Apply source pattern by word position.
        def repl(match: re.Match[str]) -> str:
            nonlocal idx
            pattern = patterns[idx] if idx < len(patterns) else "mixed"
            idx += 1
            return _apply_word_case_pattern(match.group(0), pattern)

        out = word_re.sub(repl, rewritten_heading)
        if preserve_proper_name_case:
            out = _preserve_proper_name_case(source_heading, out)
        return out

    if src_matches:
        # If token counts differ (e.g., qualifier removed), preserve case pattern for aligned prefix tokens.
        patterns = [_word_case_pattern(m.group(0)) for m in src_matches]
        idx = 0

        # Function: Apply source pattern to available prefix positions only.
        def repl_prefix(match: re.Match[str]) -> str:
            nonlocal idx
            word = match.group(0)
            if idx < len(patterns):
                out = _apply_word_case_pattern(word, patterns[idx])
            else:
                out = word
            idx += 1
            return out

        out = word_re.sub(repl_prefix, rewritten_heading)
        if preserve_proper_name_case:
            out = _preserve_proper_name_case(source_heading, out)
        return out
    return rewritten_heading


# Function: Resolve heading case mode for a specific level.
def _resolve_heading_case_mode_for_level(global_mode: str, by_level: Dict[int, str], level: int) -> str:
    mode = str(global_mode).strip().lower()
    if mode == "by-level":
        return str(by_level.get(level, "automatic")).strip().lower()
    return mode


# Function: Enforce heading case normalization policy from source markdown.
def enforce_heading_case_normalization_from_source(
    source_markdown: str,
    rewritten_markdown: str,
    global_mode: str,
    by_level: Dict[int, str],
    preserve_proper_name_case: bool = True,
) -> tuple[str, int]:
    src_lines = source_markdown.splitlines()
    out_lines = rewritten_markdown.splitlines()
    src_headings: List[tuple[int, str]] = []
    out_headings: List[tuple[int, int, str, int]] = []

    i = 0
    while i < len(src_lines):
        h = get_heading_at(src_lines, i)
        if h:
            level, heading_text, span = h
            src_headings.append((level, heading_text))
            i += span
            continue
        i += 1

    i = 0
    while i < len(out_lines):
        h = get_heading_at(out_lines, i)
        if h:
            level, heading_text, span = h
            out_headings.append((i, level, heading_text, span))
            i += span
            continue
        i += 1

    edits = 0
    limit = min(len(src_headings), len(out_headings))
    for idx in range(limit):
        src_level, src_text = src_headings[idx]
        out_idx, out_level, out_text, _out_span = out_headings[idx]
        if src_level != out_level:
            continue
        mode = _resolve_heading_case_mode_for_level(global_mode, by_level, out_level)
        recased = apply_heading_case_style(
            out_text,
            mode,
            source_heading=src_text,
            preserve_proper_name_case=preserve_proper_name_case,
        )
        if recased == out_text:
            continue
        m = ATX_HEADING_RE.match(out_lines[out_idx])
        if m:
            out_lines[out_idx] = out_lines[out_idx][:m.start(2)] + recased + out_lines[out_idx][m.end(2):]
            edits += 1
            continue
        prefix_match = re.match(r"^(\s*)", out_lines[out_idx])
        prefix = prefix_match.group(1) if prefix_match else ""
        out_lines[out_idx] = f"{prefix}{recased}"
        edits += 1
    return "\n".join(out_lines), edits


# Function: Enforce source heading casing in rewritten markdown.
def enforce_heading_casing_from_source(
    source_markdown: str,
    rewritten_markdown: str,
    preserve_proper_name_case: bool = True
) -> tuple[str, int]:
    # Backward-compatible wrapper for the old boolean behaviour ("identical").
    return enforce_heading_case_normalization_from_source(
        source_markdown,
        rewritten_markdown,
        "identical",
        {},
        preserve_proper_name_case=preserve_proper_name_case,
    )


# Function: Check whether reference heading.
def is_reference_heading(text: str) -> bool:
    return normalize_heading_text(text) in REFERENCE_HEADINGS


# Function: Find reference sections.
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
    return re.sub(r"(``[^`]+``|`[^`]+`)", "", text)


# Function: Strip html.
def strip_html(text: str) -> str:
    # Remove HTML tags and block elements to exclude HTML from profiling.
    text = re.sub(r"(?is)<[A-Za-z][^>]*>.*?</[A-Za-z][^>]*>", "\n", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = re.sub(r"<[A-Za-z/][^>]*>", "", text)
    return text


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


# Function: Strip html entities.
def strip_html_entities(text: str) -> str:
    # Remove HTML entities (e.g., &nbsp;).
    return re.sub(r"&[A-Za-z0-9#]+;", "", text)


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


# Function: Strip quoted passages.
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


# Function: Filter author voice text.
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


# Function: Mask non voice blocks.
def mask_non_voice_blocks(markdown: str) -> tuple[str, Dict[str, str]]:
    # Replace non-voice blocks with placeholders to preserve them verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Create placeholder.
    def make_placeholder(block: str) -> str:
        nonlocal counter
        placeholder = f"[[FROZEN_BLOCK_{counter}]]"
        mapping[placeholder] = block
        counter += 1
        return placeholder

    # Function: Replacement helper for html.
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


# Function: Mask inline citations.
def mask_inline_citations(text: str) -> tuple[str, Dict[str, str]]:
    # Replace inline citations with placeholders to preserve them verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Create placeholder.
    def make_placeholder(match_text: str) -> str:
        nonlocal counter
        placeholder = f"[[CITATION_{counter}]]"
        mapping[placeholder] = match_text
        counter += 1
        return placeholder

    # Function: Replacement helper for simple.
    def repl_simple(match: re.Match[str]) -> str:
        return make_placeholder(match.group(0))

    text = INLINE_FOOTNOTE_RE.sub(repl_simple, text)
    text = INLINE_NUMERIC_CITE_RE.sub(repl_simple, text)

    # Function: Replacement helper for paren.
    def repl_paren(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        return make_placeholder(match.group(0)) if is_parenthetical_citation(inner) else match.group(0)

    text = PAREN_GROUP_RE.sub(repl_paren, text)
    return text, mapping


# Function: Mask quoted passages.
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


# Function: Restore placeholders.
def restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    for placeholder, original in mapping.items():
        if placeholder in text:
            text = text.replace(placeholder, original)
    return text


# Function: Find placeholders.
def find_placeholders(text: str, pattern: re.Pattern[str]) -> List[str]:
    return pattern.findall(text)


# Function: Normalize heading.
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


# Function: Extract heading blocks.
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


# Function: Extract heading keys.
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


# Function: Build section signature.
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


# Function: Compute Jaccard similarity for token sets.
def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


# Function: Compute heading similarity for section matching.
def heading_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Function: Parse humanizer guidelines.
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


# Function: Normalize humanizer rules.
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


# Function: Build humanizer parse prompt.
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


# Function: Parse humanizer guidelines LLM.
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


# Function: Load humanizer rules cache.
def load_humanizer_rules_cache(cache_path: Path) -> Dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# Function: Write humanizer rules cache.
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


# Function: Analyze markdown style.
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


# Function: Filter humanizer rules.
def filter_humanizer_rules(
    rules: List[Dict[str, Any]],
    fingerprint: Dict[str, Any],
    input_style: Dict[str, Any] | None = None,
    tunables: Dict[str, Any] | None = None,
    local_spelling_rules: Dict[str, Any] | None = None,
    avoid_list: List[str] | None = None
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
    heading_case_mode = "by-level"
    heading_case_by_level: Dict[int, str] = {}
    if isinstance(tunables, dict):
        mandatory = tunables.get("humanizer_mandatory", {})
        if isinstance(mandatory, dict):
            em_dash_forbidden = bool(mandatory.get("avoid_em_dashes", False))
            emoji_policy = mandatory.get("emoji_policy")
    heading_case_mode, heading_case_by_level, _preserve_name_case = get_heading_case_normalization_conf(tunables)
    deterministic_heading_case = (
        heading_case_mode == "identical"
        or (
            heading_case_mode == "by-level"
            and any(
                str(heading_case_by_level.get(level, "automatic")).lower() not in {"automatic"}
                for level in range(1, MAX_CONFIG_HEADING_LEVEL + 1)
            )
        )
    )

    # Function: Collect style context.
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

    # Function: Expand words.
    def expand_words(words: List[str]) -> set[str]:
        out: set[str] = set()
        for w in words:
            if not w:
                continue
            parts = re.split(r"/|\\bor\\b", w)
            for part in parts:
                part = _normalize_lexicon_token(part)
                if part:
                    out.add(part)
        return out

    avoid_literal = {
        _normalize_lexicon_token(w)
        for w in (avoid_list or [])
        if isinstance(w, str)
    }
    avoid_words_norm = {w for w in avoid_words if w not in avoid_literal}
    avoid_words_us = normalize_tokens_for_avoidance(avoid_words_norm, local_spelling_rules or {})

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
        if tokens:
            if avoid_literal and any(t in avoid_literal for t in tokens):
                drop_reason = "Rule conflicts with global avoid words."
            elif avoid_words_us:
                tokens_us = normalize_tokens_for_avoidance(tokens, local_spelling_rules or {})
                if any(t in avoid_words_us for t in tokens_us):
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
            if "title case" in title and deterministic_heading_case:
                drop_reason = "Heading case normalization is deterministic via humanizer_mandatory."
            elif "title case" in title and title_case_rate >= heading_title_case_keep_rate:
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
        total += 4  # per-message overhead
    return total + 2


# Function: Estimate tokens for text.
def estimate_tokens_for_text(text: str) -> int:
    # Rough token estimate for plain text.
    return estimate_tokens(text)


# Function: Split markdown blocks.
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


# Function: Check whether code block.
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


# Function: Split text into words while preserving separators.
def split_words_preserve(text: str) -> List[str]:
    return re.findall(r"\S+", text)


# Function: Split sentences for chunking.
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


# Function: Split a block into units based on the chosen strategy.
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


# Function: Mask inline code.
def mask_inline_code(text: str) -> tuple[str, Dict[str, str]]:
    # Replace inline code spans with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Replacement helper for data.
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


# Function: Find inline code placeholders.
def find_inline_code_placeholders(text: str) -> List[str]:
    return re.findall(r"\[\[INLINE_CODE_\d+\]\]", text)


# Function: Mask html.
def mask_html(text: str) -> tuple[str, Dict[str, str]]:
    # Replace HTML blocks and tags with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Create placeholder.
    def make_placeholder(block: str) -> str:
        nonlocal counter
        placeholder = f"[[HTML_BLOCK_{counter}]]"
        mapping[placeholder] = block
        counter += 1
        return placeholder

    # Function: Replacement helper for block.
    def repl_block(match: re.Match[str]) -> str:
        return make_placeholder(match.group(0))

    # Mask block elements first.
    text = re.sub(r"(?is)<(script|style|table|pre|code|svg|math|div|section|article|header|footer|nav|aside)[^>]*>.*?</\\1>", repl_block, text)
    # Mask HTML comments.
    text = re.sub(r"(?is)<!--.*?-->", repl_block, text)
    # Mask any remaining tags (avoid matching inequalities like <10).
    text = re.sub(r"(?is)<[A-Za-z/][^>]*>", repl_block, text)
    return text, mapping


# Function: Mask math notation.
def mask_math_notation(text: str) -> tuple[str, Dict[str, str]]:
    # Replace LaTeX-style math with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter_inline = 0
    counter_display = 0

    # Function: Replacement helper for display.
    def repl_display(match: re.Match[str]) -> str:
        nonlocal counter_display
        placeholder = f"[[DISPLAY_MATH_{counter_display}]]"
        mapping[placeholder] = match.group(0)
        counter_display += 1
        return placeholder

    # Function: Replacement helper for inline.
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


# Function: Mask html entities.
def mask_html_entities(text: str) -> tuple[str, Dict[str, str]]:
    # Replace HTML entities with placeholders to preserve verbatim.
    mapping: Dict[str, str] = {}
    counter = 0

    # Function: Replacement helper for data.
    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"[[HTML_ENTITY_{counter}]]"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    stripped = re.sub(r"&[A-Za-z0-9#]+;", repl, text)
    return stripped, mapping


# Function: Split an oversized markdown block safely.
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


# Function: Enforce minimum chunks.
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


# Function: Chunk markdown into size-limited units.
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

    # Function: Flush data.
    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        if not block.strip():
            continue
        # Split by the requested unit (paragraph/sentence/word) with fallback if a unit is oversized.
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

    # Function: Check whether a sentence contains a marker.
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
    # Function: Compute the fraction of repeated n-grams above a minimum count.
    def repeat_rate(ngram_list: List[str], min_count: int = 3) -> float:
        if not ngram_list:
            return 0.0
        counts = collections.Counter(ngram_list)
        repeat_tokens = sum(c for _, c in counts.items() if c >= min_count)
        return repeat_tokens / max(1, len(ngram_list))

    # Function: Generate n-grams from tokens.
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


# Function: Compute L1 distance for distance.
def l1_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b))


# Function: Compute relative difference between values.
def relative_diff(a: float, b: float) -> float:
    denom = max(abs(b), 1.0)
    return abs(a - b) / denom


# Function: Compute style compliance.
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

    # Function: Add score.
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


# Function: Compute entropy for data.
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


# Function: Compute Jensen-Shannon divergence for divergence.
def _js_divergence(p: List[float], q: List[float]) -> float:
    if not p or not q or len(p) != len(q):
        return 0.0
    eps = 1e-12
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log((pi + eps) / (mi + eps), 2) for pi, mi in zip(p, m))
    kl_qm = sum(qi * math.log((qi + eps) / (mi + eps), 2) for qi, mi in zip(q, m))
    return (kl_pm + kl_qm) / 2.0


# Function: Compute humanization metrics.
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


# Function: Compute the aggregate humanization score.
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
    # Function: Initialize an LLM configuration instance.
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

# Function: Load config.
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


# Function: Resolve roster config path.
def resolve_roster_config_path(path: Path | None = None) -> Path | None:
    if isinstance(path, Path) and path.exists():
        return path
    cwd_path = Path.cwd() / LLM_ROSTER_FILENAME
    script_path = Path(__file__).resolve().parent / LLM_ROSTER_FILENAME
    if cwd_path.exists():
        return cwd_path
    if script_path.exists():
        return script_path
    return None


# Function: Parse optional roster seed.
def parse_roster_seed(value: str | None) -> int | None:
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    return int(token)


# Function: Build roster index schedule for chunks.
def build_roster_indices(roster_size: int, chunk_count: int, seed: int | None = None) -> List[int]:
    if roster_size <= 0 or chunk_count <= 0:
        return []
    if seed is None:
        return [idx % roster_size for idx in range(chunk_count)]
    rng = random.Random(int(seed))
    out: List[int] = []
    while len(out) < chunk_count:
        cycle = list(range(roster_size))
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:chunk_count]


# Function: Load model roster entries and map to runtime configs.
def load_llm_roster(path: Path, base_cfg: LLMConfig) -> List[LLMConfig]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse roster config at {path}: {exc}") from exc

    raw_entries: Any = data
    if isinstance(data, dict):
        raw_entries = data.get("roster")
    if not isinstance(raw_entries, list):
        raise ValueError("Roster config must be a JSON array or an object with a 'roster' array.")

    out: List[LLMConfig] = []
    for idx, entry in enumerate(raw_entries, start=1):
        overrides: Dict[str, Any]
        if isinstance(entry, str):
            overrides = {"model": entry}
        elif isinstance(entry, dict):
            overrides = entry
        else:
            continue
        try:
            max_tokens = int(overrides.get("max_tokens", base_cfg.max_tokens))
            max_prompt_tokens = int(overrides.get("max_prompt_tokens", base_cfg.max_prompt_tokens))
            timeout_seconds = int(overrides.get("timeout_seconds", base_cfg.timeout_seconds))
            max_retries = int(overrides.get("max_retries", base_cfg.max_retries))
            backoff_base_seconds = float(overrides.get("backoff_base_seconds", base_cfg.backoff_base_seconds))
            backoff_max_seconds = float(overrides.get("backoff_max_seconds", base_cfg.backoff_max_seconds))
            temperature = float(overrides.get("temperature", base_cfg.temperature))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric override in roster entry {idx}: {exc}") from exc

        base_extra_headers = dict(base_cfg.extra_headers) if isinstance(base_cfg.extra_headers, dict) else {}
        entry_headers = overrides.get("extra_headers")
        if isinstance(entry_headers, dict):
            merged_headers = {**base_extra_headers, **entry_headers}
        else:
            merged_headers = base_extra_headers
        model = str(overrides.get("model", base_cfg.model)).strip()
        if not model:
            raise ValueError(f"Roster entry {idx} has an empty model.")

        out.append(
            LLMConfig(
                api_key=str(overrides.get("api_key", base_cfg.api_key)),
                base_url=str(overrides.get("base_url", base_cfg.base_url)),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                extra_headers=merged_headers,
                max_prompt_tokens=max_prompt_tokens,
                max_retries=max_retries,
                backoff_base_seconds=backoff_base_seconds,
                backoff_max_seconds=backoff_max_seconds,
            )
        )

    if not out:
        raise ValueError("Roster config contains no valid entries.")
    return out

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

# Function: Parse JSON strict.
def parse_json_strict(s: str) -> Dict[str, Any]:
    # Strip code fences if present and parse strictly as JSON.
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)

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


# ---- Prompting: apply fingerprint ----

# Function: Build the LLM prompt for applying a fingerprint to a draft.
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
    chunk_summary: Dict[str, Any] | None = None,
    local_spelling_llm: str = "none",
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
    local_spelling_llm = normalize_spelling_locale(local_spelling_llm)
    if local_spelling_llm != "none":
        locale_labels = {
            "us": "US",
            "canadian": "Canadian",
            "british": "British",
            "australian": "Australian",
        }
        locale_label = locale_labels.get(local_spelling_llm, local_spelling_llm)
        if isinstance(user.get("rules"), list):
            user["rules"].append(
                f"Use {locale_label} English spelling consistently for lexical variants in final_markdown."
            )
        user["local_spelling_target"] = local_spelling_llm
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


# Function: Apply pronoun override.
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


# Function: Evaluate pronoun override.
def evaluate_pronoun_override(text: str, mode: str) -> Dict[str, Any]:
    # Normalize token-like forms so contractions map to their pronoun root.
    def normalize_pronoun_token(tok: str) -> str:
        t = tok.lower()
        if t in {"i", "we", "you", "he", "she", "they", "it", "me", "us", "him", "her", "them"}:
            return t
        if "'" in t:
            root = t.split("'", 1)[0]
            if root in {"i", "we", "you", "he", "she", "they", "it"}:
                return root
        return t

    pronoun_person: Dict[str, str] = {
        "i": "first", "me": "first", "my": "first", "mine": "first", "myself": "first",
        "we": "first", "us": "first", "our": "first", "ours": "first", "ourselves": "first",
        "you": "second", "your": "second", "yours": "second", "yourself": "second", "yourselves": "second",
        "he": "third", "him": "third", "his": "third", "himself": "third",
        "she": "third", "her": "third", "hers": "third", "herself": "third",
        "they": "third", "them": "third", "their": "third", "theirs": "third", "themselves": "third",
        "it": "third", "its": "third", "itself": "third",
    }
    subject_forms = {"i", "we", "you", "he", "she", "they", "it"}
    object_forms = {"me", "us", "him", "her", "them"}
    possessive_det_forms = {"my", "your", "his", "her", "our", "their", "its"}

    # Rough lexical cues to detect clause starts and predicate context.
    auxiliary_or_main_verbs = {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    }
    common_prepositions = {
        "to", "for", "with", "from", "of", "by", "on", "in", "at", "into", "onto",
        "over", "under", "about", "through", "across", "between", "among", "against",
        "toward", "towards", "without", "within", "before", "after", "during", "around",
    }
    common_transitive_verbs = {
        "help", "see", "know", "tell", "ask", "call", "watch", "hear", "find", "keep",
        "give", "show", "send", "bring", "take", "leave", "love", "hate", "trust",
        "support", "follow", "join", "thank", "warn", "teach", "remind", "meet",
        "chase", "save", "move", "lead", "guide", "blame", "praise", "inform",
    }
    determiners = {"a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every"}
    sentence_boundary = {".", "!", "?", ";", ":"}
    pronoun_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    punct_token_re = re.compile(r"[.!?,;:()\\[\\]{}\"“”‘’\\-]")
    stream = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,;:()\[\]{}\"“”‘’\-]", text)

    mode_label = mode if mode in {"first", "second", "third"} else "third"
    if mode_label == "first":
        disallowed = {"second", "third"}
    elif mode_label == "second":
        disallowed = {"first", "third"}
    else:
        disallowed = {"first", "second"}

    allowed_count = 0
    violations: Dict[str, int] = {}
    ignored_non_subject: Dict[str, int] = {}

    # Function: Find previous lexical token (word) inside current clause.
    def prev_word(idx: int) -> str:
        j = idx - 1
        while j >= 0:
            tok = stream[j]
            if punct_token_re.fullmatch(tok):
                if tok in sentence_boundary:
                    break
                j -= 1
                continue
            if pronoun_token_re.fullmatch(tok):
                return normalize_pronoun_token(tok)
            j -= 1
        return ""

    # Function: Find next lexical token (word) inside current clause.
    def next_word(idx: int) -> str:
        j = idx + 1
        while j < len(stream):
            tok = stream[j]
            if punct_token_re.fullmatch(tok):
                if tok in sentence_boundary:
                    break
                j += 1
                continue
            if pronoun_token_re.fullmatch(tok):
                return normalize_pronoun_token(tok)
            j += 1
        return ""

    # Function: Is a word likely functioning as a finite/main verb here?
    def is_verb_like(word: str) -> bool:
        if not word:
            return False
        if word in auxiliary_or_main_verbs or word in common_transitive_verbs:
            return True
        if word in determiners or word in possessive_det_forms or word in pronoun_person:
            return False
        if re.match(r".*(ed|ing|en)$", word):
            return True
        # Conservative fallback for present-tense verb morphology.
        if re.match(r".*(es|s)$", word) and len(word) > 3:
            return True
        return False

    # Function: Check if token starts a clause-like region.
    def is_clause_start(idx: int) -> bool:
        if idx == 0:
            return True
        j = idx - 1
        while j >= 0:
            tok = stream[j]
            if punct_token_re.fullmatch(tok):
                if tok in sentence_boundary or tok in {",", "(", "[", "{", "\"", "“", "”"}:
                    return True
                j -= 1
                continue
            if pronoun_token_re.fullmatch(tok):
                return tok.lower() in {"and", "but", "or", "so", "yet", "nor", "that", "because", "if", "when", "while", "although", "though"}
            j -= 1
        return True

    # Function: Is this pronoun token likely acting as a grammatical subject?
    def is_subject_like(idx: int, raw_tok: str, norm_tok: str) -> bool:
        clause_start = is_clause_start(idx)
        prev_sig = prev_word(idx)
        nxt = next_word(idx)

        # Core subject pronoun heuristic.
        if norm_tok in subject_forms:
            if clause_start:
                return True
            if nxt and is_verb_like(nxt):
                return True
            return False

        # Object pronouns are normally object/complement roles, but in non-standard
        # constructions ("Him and I ...", "Him was ...") they can still be subjects.
        if norm_tok in object_forms:
            if clause_start and (is_verb_like(nxt) or nxt in {"and", "or"}):
                return True
            if prev_sig in common_prepositions:
                return False
            if prev_sig and is_verb_like(prev_sig):
                return False
            return False

        # Possessive determiners/possessive pronouns should not be treated as subject triggers.
        return False

    for idx, raw_tok in enumerate(stream):
        if not pronoun_token_re.fullmatch(raw_tok):
            continue
        norm_tok = normalize_pronoun_token(raw_tok)
        person = pronoun_person.get(norm_tok)
        if not person:
            continue

        # Track target-person pronoun coverage regardless of role.
        if person == mode_label:
            allowed_count += 1

        if person not in disallowed:
            continue

        subject_like = is_subject_like(idx, raw_tok, norm_tok)

        if subject_like:
            label = f"{person}_person"
            violations[label] = int(violations.get(label, 0)) + 1
        else:
            # Keep diagnostics for disallowed pronouns that were allowed because they were likely
            # object/possessive references to other entities.
            label = f"{person}_person_non_subject"
            ignored_non_subject[label] = int(ignored_non_subject.get(label, 0)) + 1

    return {
        "mode": mode_label,
        "allowed_count": allowed_count,
        "violations": violations,
        "ignored_non_subject": ignored_non_subject,
    }


# Function: Render pronoun override diagnostics for verbose logging.
def format_pronoun_override_debug(override_eval: Dict[str, Any]) -> str:
    mode = str(override_eval.get("mode", "unknown"))
    allowed_count = override_eval.get("allowed_count")
    violations = override_eval.get("violations")
    parts: List[str] = [f"mode={mode}"]
    if isinstance(allowed_count, int):
        parts.append(f"allowed_count={allowed_count}")
    if isinstance(violations, dict) and violations:
        violation_parts: List[str] = []
        for label in ("first_person", "second_person", "third_person"):
            count = violations.get(label)
            if isinstance(count, int) and count > 0:
                violation_parts.append(f"{label}={count}")
        for label, count in violations.items():
            if label in {"first_person", "second_person", "third_person"}:
                continue
            if isinstance(count, int) and count > 0:
                violation_parts.append(f"{label}={count}")
        if violation_parts:
            parts.append("violations[" + ", ".join(violation_parts) + "]")
    ignored = override_eval.get("ignored_non_subject")
    if isinstance(ignored, dict) and ignored:
        ignored_parts: List[str] = []
        for label in ("first_person_non_subject", "second_person_non_subject", "third_person_non_subject"):
            count = ignored.get(label)
            if isinstance(count, int) and count > 0:
                ignored_parts.append(f"{label}={count}")
        for label, count in ignored.items():
            if label in {"first_person_non_subject", "second_person_non_subject", "third_person_non_subject"}:
                continue
            if isinstance(count, int) and count > 0:
                ignored_parts.append(f"{label}={count}")
        if ignored_parts:
            parts.append("ignored[" + ", ".join(ignored_parts) + "]")
    return "; ".join(parts)


# Function: Score pronoun override quality for best-attempt selection.
def pronoun_override_quality(override_eval: Dict[str, Any]) -> tuple[int, int]:
    violations = override_eval.get("violations")
    total_violations = 0
    if isinstance(violations, dict):
        for count in violations.values():
            if isinstance(count, int) and count > 0:
                total_violations += count
    allowed_count = override_eval.get("allowed_count")
    if not isinstance(allowed_count, int):
        allowed_count = 0
    # Higher is better: fewer violations first, then more allowed pronouns.
    return (-total_violations, allowed_count)


# Function: CLI entry point.
def main() -> int:
    query_arg = extract_query_arg(sys.argv[1:])
    if query_arg is not None:
        return handle_query(query_arg)
    if "--license" in sys.argv:
        return print_license_and_exit()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query",
        type=str,
        default=None,
        help="Return a lightweight value and exit; supported: perplexity"
    )
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
        "--local-spelling",
        dest="local_spelling",
        choices=["none", "canadian", "australian", "british", "us"],
        default=None,
        help="Override both tunables humanizer_mandatory.force_local_spelling_LLM and force_local_spelling_rules for this run"
    )
    ap.add_argument(
        "--local-spelling-llm",
        dest="local_spelling_llm",
        choices=["none", "canadian", "australian", "british", "us"],
        default=None,
        help="Override tunables humanizer_mandatory.force_local_spelling_LLM for this run"
    )
    ap.add_argument(
        "--local-spelling-rules",
        dest="local_spelling_rules",
        choices=["none", "canadian", "australian", "british", "us"],
        default=None,
        help="Override tunables humanizer_mandatory.force_local_spelling_rules for this run"
    )
    ap.add_argument(
        "--perplexity",
        choices=list(PERPLEXITY_LEVELS),
        default=None,
        help="Override tunables perplexity_level for this run (default|low|medium|high|extreme)"
    )
    ap.add_argument(
        "--seed",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Override tunables humanizer_variance.seed for this run (0 or omitted value = random seed)"
    )
    ap.add_argument(
        "--roster",
        nargs="?",
        const="",
        default=None,
        help=(
            "Use config.llm.roster.json and route one model per chunk in roster order. "
            "Optional integer value enables seeded random non-repeating roster cycles."
        )
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
        default=DEFAULT_MAX_STYLE_RETRIES,
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

    # Function: Verbose-print when enabled.
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
    tunables, perplexity_level, perplexity_knobs = apply_perplexity_profile(tunables, args.perplexity)
    base_temperature = cfg.temperature
    temp_multiplier = perplexity_knobs.get("llm.temperature_multiplier", 1.0)
    cfg.temperature = apply_temperature_multiplier(base_temperature, temp_multiplier)
    print(f"Perplexity level: {perplexity_level}")
    if args.verbose:
        if args.perplexity:
            vprint(f"Perplexity override: {args.perplexity} (CLI)")
        vprint(
            "Perplexity knobs: "
            f"humanizer_variance.max_ops_per_1000w={perplexity_knobs.get('humanizer_variance.max_ops_per_1000w')}, "
            f"humanization_controller.quantiles={perplexity_knobs.get('humanization_controller.quantiles')}, "
            f"humanization_controller.range_pct={perplexity_knobs.get('humanization_controller.range_pct')}, "
            f"chunking.max_input_tokens={perplexity_knobs.get('chunking.max_input_tokens')}, "
            "chunking.min_chunks_when_perturbing="
            f"{perplexity_knobs.get('chunking.min_chunks_when_perturbing')}, "
            f"llm.temperature_multiplier={perplexity_knobs.get('llm.temperature_multiplier')}, "
            f"llm.temperature_effective={cfg.temperature:.3f} (base={base_temperature:.3f})"
        )

    roster_cfgs: List[LLMConfig] = []
    roster_seed: int | None = None
    roster_enabled = args.roster is not None
    if roster_enabled:
        try:
            roster_seed = parse_roster_seed(args.roster)
        except ValueError:
            print_error("Error: --roster seed must be an integer when provided.")
            return 2
        roster_path = resolve_roster_config_path(None)
        if roster_path is None:
            print_error(f"Error: roster mode requested but {LLM_ROSTER_FILENAME} was not found.")
            return 2
        try:
            roster_cfgs = load_llm_roster(roster_path, cfg)
        except ValueError as exc:
            print_error(f"Error: {exc}")
            return 2
        if roster_seed is None:
            print(f"LLM roster: ordered ({len(roster_cfgs)} models)")
        else:
            print(f"LLM roster: random seed={roster_seed} ({len(roster_cfgs)} models)")
        if args.verbose:
            for idx, entry in enumerate(roster_cfgs, start=1):
                vprint(
                    f"  Roster[{idx}] model={entry.model} base_url={entry.base_url} "
                    f"temperature={entry.temperature:.3f}"
                )

    variance_seed_override: int | None = None
    if args.seed is not None:
        if args.seed == 0:
            variance_seed_override = random.SystemRandom().randint(1, 2**31 - 1)
            if args.verbose:
                vprint(f"Humanizer variance seed override (random): {variance_seed_override}")
        else:
            variance_seed_override = int(args.seed)
            if args.verbose:
                vprint(f"Humanizer variance seed override (CLI): {variance_seed_override}")
    if args.max_prompt_tokens is not None:
        # Allow CLI override for chunking threshold.
        cfg.max_prompt_tokens = args.max_prompt_tokens
    max_voice_retries = args.max_style_retries
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
            args.max_style_retries, max_voice_retries = resolve_retry_budgets(
                style_retry,
                args.max_style_retries,
                DEFAULT_MAX_STYLE_RETRIES
            )
    if args.verbose and not args.no_style_retry:
        max_chunk_attempts = 1 + args.max_style_retries + (max_voice_retries if args.force_person else 0)
        vprint(
            "Retry budgets: "
            f"voice retries={max_voice_retries}, style retries={args.max_style_retries}, "
            f"max chunk attempts={max_chunk_attempts}, threshold={args.style_retry_threshold:.3f}"
        )

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
    normalize_double_quotes = should_normalize_double_quotes(tunables)
    normalize_single_quotes = should_normalize_single_quotes(tunables)
    sanitize_heading_qualifiers, heading_allowlist_raw = get_heading_qualifier_sanitize_conf(tunables)
    heading_allowlist = compile_heading_allowlist(heading_allowlist_raw) if sanitize_heading_qualifiers else []
    heading_case_mode, heading_case_by_level, preserve_proper_name_case = get_heading_case_normalization_conf(tunables)
    force_local_spelling_llm, force_local_spelling_rules = get_force_local_spelling_conf(tunables)
    if args.local_spelling:
        # Backward-compatible umbrella override: applies to both LLM and deterministic rules.
        force_local_spelling_llm = args.local_spelling
        force_local_spelling_rules = args.local_spelling
    if args.local_spelling_llm:
        force_local_spelling_llm = args.local_spelling_llm
    if args.local_spelling_rules:
        force_local_spelling_rules = args.local_spelling_rules
    if args.verbose:
        if args.local_spelling:
            vprint(f"Local spelling override: {args.local_spelling} (CLI, applies to LLM + rules)")
        if args.local_spelling_llm:
            vprint(f"Local spelling LLM override: {args.local_spelling_llm} (CLI)")
        if args.local_spelling_rules:
            vprint(f"Local spelling rules override: {args.local_spelling_rules} (CLI)")
    local_spelling_rules = load_local_spelling_rules() if force_local_spelling_rules != "none" else {}
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
            if normalize_double_quotes:
                vprint("Hard constraint active: double quotes will be normalized to straight quotes (humanizer_mandatory).")
            if normalize_single_quotes:
                vprint("Hard constraint active: single quotes will be normalized to straight quotes (humanizer_mandatory).")
            if sanitize_heading_qualifiers:
                allowlist_note = f" (allowlist: {len(heading_allowlist)} pattern(s))" if heading_allowlist else ""
                vprint(
                    "Hard constraint active: heading qualifiers will be sanitized (humanizer_mandatory)."
                    + allowlist_note
                )
            if heading_case_mode == "identical":
                vprint("Hard constraint active: heading casing mode = identical (humanizer_mandatory).")
            elif heading_case_mode == "by-level":
                by_level_render = ", ".join(
                    f"H{lvl}={heading_case_by_level.get(lvl, 'automatic')}" for lvl in range(1, MAX_CONFIG_HEADING_LEVEL + 1)
                )
                preserve_note = "on" if preserve_proper_name_case else "off"
                vprint(
                    "Hard constraint active: heading casing mode = by-level "
                    f"({by_level_render}; preserve proper names={preserve_note})."
                )
            if force_local_spelling_llm != "none":
                vprint(f"Hard constraint active: LLM spelling target = {force_local_spelling_llm}.")
            else:
                vprint("Hard constraint active: LLM spelling target = none (no explicit spelling instruction).")
            if force_local_spelling_rules != "none":
                vprint(
                    "Hard constraint active: "
                    f"deterministic spelling rules target = {force_local_spelling_rules}."
                )
            else:
                vprint("Hard constraint active: deterministic spelling rules target = none.")
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
            humanizer_rules, dropped_rules = filter_humanizer_rules(
                parsed_rules,
                fingerprint,
                input_style,
                tunables,
                local_spelling_rules,
                avoid_list
            )
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

    # Function: Build messages for chunk.
    def build_messages_for_chunk(
        md_chunk: str,
        style_feedback: Dict[str, Any] | None = None,
        for_estimate: bool = False,
        fingerprint_override: Dict[str, Any] | None = None,
        controller_overlay: Dict[str, Any] | None = None,
        previous_summary: str | None = None,
        summary_words: int | None = None,
        summary_enabled: bool = False,
        llm_cfg: LLMConfig | None = None
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
        prompt_cfg = llm_cfg if isinstance(llm_cfg, LLMConfig) else cfg
        return build_apply_prompt(
            fp_payload,
            md_chunk,
            input_meas,
            prompt_cfg,
            prompts,
            style_feedback,
            humanizer_rules,
            controller_overlay,
            args.force_person,
            summary_payload,
            force_local_spelling_llm,
        )

    # Function: Rewrite a single chunk via the LLM.
    def rewrite_chunk(
        md_chunk: str,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        depth: int = 0,
        previous_summary: str | None = None,
        summary_words: int | None = None,
        summary_enabled: bool = False,
        chunk_cfg: LLMConfig | None = None
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        # Rewrite a chunk with optional style retry.
        active_cfg = chunk_cfg if isinstance(chunk_cfg, LLMConfig) else cfg
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
        # Track style-compliance retries separately from forced-person retries so that
        # voice enforcement doesn't consume the style retry budget (common with --1st-person).
        style_retries_used = 0
        voice_retries_used = 0
        # When forcing narrative person, we treat the "voice pass" output as a baseline that
        # generates feedback; style attempts begin on the *next* call so budgets add.
        style_phase_started = args.force_person is None
        attempt_global = 0
        best_attempt = 0
        best_score: float | None = None
        best_md: str | None = None
        best_out: Dict[str, Any] | None = None
        best_comp: Dict[str, Any] | None = None
        best_voice_key: tuple[int, int] | None = None
        best_voice_attempt = 0
        best_voice_md: str | None = None
        best_voice_out: Dict[str, Any] | None = None
        best_voice_eval: Dict[str, Any] | None = None
        fp_overlay = None
        controller_overlay = None
        if isinstance(tunables, dict):
            fp_overlay, controller_overlay = build_controller_overlay(
                fingerprint,
                tunables,
                chunk_index,
                md_chunk,
                variance_seed_override
            )
        if controller_overlay and args.verbose:
            vprint(f"Controller overlay for chunk {chunk_index}/{chunk_total}: {controller_overlay}")
        style_feedback: Dict[str, Any] | None = None
        last_out: Dict[str, Any] = {}
        max_chunk_attempts = 1 + args.max_style_retries + (max_voice_retries if args.force_person else 0)
        first_attempt_summary: str | None = None
        refresh_summary_after_bad_first = False
        is_last_chunk = bool(
            chunk_index is not None
            and chunk_total is not None
            and chunk_index == chunk_total
        )
        while True:
            if attempt_global >= max_chunk_attempts:
                if args.verbose and chunk_index is not None and chunk_total is not None:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} reached max chunk attempts "
                        f"({max_chunk_attempts}); stopping retries."
                    )
                if best_md is not None and best_out is not None and best_comp is not None:
                    return best_md, best_out, best_comp
                # Defensive fallback: should not occur because max attempts is at least 1.
                fallback_out = {
                    "final_markdown": md_chunk,
                    "deviations": [
                        {
                            "rule_or_field": "chunk_retry_cap_fallback",
                            "reason": "Chunk retry cap reached before a valid rewrite; original chunk preserved.",
                        }
                    ],
                    "self_check": {"notes": ["Original chunk preserved due to retry cap fallback."]},
                }
                return md_chunk, fallback_out, {"score": 0.0, "deltas": []}
            attempt_global += 1
            request_chunk_summary = should_request_chunk_summary(
                attempt_global,
                summary_enabled,
                is_last_chunk,
                refresh_summary_after_bad_first,
            )
            messages = build_messages_for_chunk(
                md_chunk,
                style_feedback,
                False,
                fp_overlay,
                controller_overlay,
                previous_summary,
                summary_words,
                request_chunk_summary,
                active_cfg
            )
            input_tokens = estimate_tokens_for_messages(messages)
            last_raw = ""
            last_usage: Dict[str, Any] | None = None
            last_err: Exception | None = None
            out_obj: Dict[str, Any] | None = None
            for attempt in range(active_cfg.max_retries + 1):
                try:
                    raw, usage = chat_completions(active_cfg, messages)
                    last_raw = raw
                    last_usage = usage
                    try:
                        out_obj = parse_json_strict(raw)
                    except Exception:
                        vprint("Invalid JSON returned; attempting repair...")
                        out_obj = repair_json_with_llm(active_cfg, raw, prompts)
                    final_md = out_obj.get("final_markdown") if isinstance(out_obj, dict) else None
                    if isinstance(final_md, str) and final_md.strip():
                        if attempt > 0:
                            print_warn(f"LLM output recovered after {attempt} retry(ies).")
                        break
                    last_err = RuntimeError("LLM did not return final_markdown")
                except Exception as exc:
                    last_err = exc
                if attempt >= active_cfg.max_retries:
                    break
                backoff = min(active_cfg.backoff_max_seconds, active_cfg.backoff_base_seconds * (2 ** attempt))
                jitter = random.uniform(0, backoff * 0.2)
                sleep_s = backoff + jitter
                print_warn(
                    "LLM output invalid "
                    f"(attempt {attempt + 1}/{active_cfg.max_retries + 1}); "
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

                # Function: Split for recovery.
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
                                summary_enabled,
                                active_cfg
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
                if variance_seed_override is not None:
                    seed = variance_seed_override
                if chunk_index is not None:
                    seed += int(chunk_index)
                else:
                    seed += abs(hash(md_chunk)) % 100000
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

            if normalize_double_quotes:
                final_md, replaced_quotes = enforce_straight_double_quotes(final_md)
                if replaced_quotes:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "punctuation.double_quotes",
                        "reason": "Curly double quotes normalized to straight quotes (humanizer_mandatory).",
                        "count": replaced_quotes
                    })
            if normalize_single_quotes:
                final_md, replaced_single_quotes = enforce_straight_single_quotes(final_md)
                if replaced_single_quotes:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "punctuation.single_quotes",
                        "reason": "Curly single quotes normalized to straight quotes (humanizer_mandatory).",
                        "count": replaced_single_quotes
                    })
            if sanitize_heading_qualifiers:
                final_md, sanitized_headings = enforce_heading_qualifiers(final_md, heading_allowlist)
                if sanitized_headings:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "structure.heading_qualifiers",
                        "reason": "Heading qualifiers removed when safe (humanizer_mandatory).",
                        "count": sanitized_headings
                    })
            if force_local_spelling_rules != "none":
                final_md, replacements = enforce_local_spelling(final_md, force_local_spelling_rules, local_spelling_rules)
                if replacements:
                    out_obj.setdefault("deviations", []).append({
                        "rule_or_field": "orthography.local_spelling",
                        "reason": (
                            f"Enforced {force_local_spelling_rules} spelling via deterministic rules "
                            "regardless of fingerprint."
                        ),
                        "count": replacements
                    })

            if args.force_person and not args.no_style_retry:
                voice_text = filter_author_voice_text(final_md)
                override_eval = evaluate_pronoun_override(voice_text, args.force_person)
                voice_key = pronoun_override_quality(override_eval)
                if best_voice_key is None or voice_key > best_voice_key:
                    best_voice_key = voice_key
                    best_voice_attempt = attempt_global
                    best_voice_md = final_md
                    best_voice_out = out_obj
                    best_voice_eval = override_eval
                if override_eval.get("violations") and voice_retries_used < max_voice_retries:
                    if args.verbose and chunk_index is not None and chunk_total is not None:
                        reason = format_pronoun_override_debug(override_eval)
                        vprint(
                            f"Chunk {chunk_index}/{chunk_total} pronoun override violations; "
                            f"retrying (voice retry {voice_retries_used + 1}/{max_voice_retries})."
                        )
                        vprint(f"  Pronoun override detail: {reason}")
                    style_feedback = {
                        "pronoun_override": override_eval,
                        "notes": [
                            "Pronoun override violations detected. Rewrite to match forced narrative voice."
                        ]
                    }
                    voice_retries_used += 1
                    continue
                if (
                    override_eval.get("violations")
                    and voice_retries_used >= max_voice_retries
                    and best_voice_md is not None
                    and best_voice_out is not None
                    and best_voice_eval is not None
                    and best_voice_key is not None
                    and voice_key < best_voice_key
                ):
                    final_md = best_voice_md
                    out_obj = best_voice_out
                    override_eval = best_voice_eval
                    if args.verbose and chunk_index is not None and chunk_total is not None:
                        vprint(
                            f"Chunk {chunk_index}/{chunk_total} using best voice attempt "
                            f"{best_voice_attempt} over attempt {attempt_global} after voice retries exhausted."
                        )
                        vprint(f"  Pronoun override detail: {format_pronoun_override_debug(override_eval)}")

            if summary_enabled:
                if request_chunk_summary:
                    llm_summary = out_obj.get("chunk_summary") if isinstance(out_obj, dict) else None
                    summary = llm_summary if isinstance(llm_summary, str) else ""
                    summary_was_fallback = False
                    if not summary.strip():
                        fallback = build_semantic_fallback_summary(final_md, summary_words)
                        summary = normalize_summary(fallback, summary_words)
                        summary_was_fallback = True
                        if args.verbose and chunk_index is not None and chunk_total is not None:
                            vprint(
                                f"Chunk {chunk_index}/{chunk_total} summary fallback (LLM empty)."
                            )
                            vprint("  LLM summary: (empty)")
                            vprint(f"  Fallback summary: {summary}")
                    else:
                        summary = normalize_summary(summary, summary_words)
                        if summary and summary_is_meta(summary):
                            fallback = build_semantic_fallback_summary(final_md, summary_words)
                            summary = normalize_summary(fallback, summary_words)
                            summary_was_fallback = True
                            if args.verbose and chunk_index is not None and chunk_total is not None:
                                vprint(
                                    f"Chunk {chunk_index}/{chunk_total} summary fallback (LLM meta/task-focused)."
                                )
                                vprint(f"  LLM summary: {normalize_summary(llm_summary, summary_words)}")
                                vprint(f"  Fallback summary: {summary}")
                    first_attempt_summary = summary
                    if attempt_global == 1 and summary_was_fallback:
                        # Allow one refresh attempt when first summary is empty/meta.
                        refresh_summary_after_bad_first = True
                    else:
                        refresh_summary_after_bad_first = False
                    out_obj["chunk_summary"] = summary
                    if args.verbose and chunk_index is not None and chunk_total is not None:
                        origin = "refresh" if attempt_global > 1 else "attempt 1"
                        vprint(
                            f"Chunk {chunk_index}/{chunk_total} summary "
                            f"({len(summary.split())} words, {origin}): {summary}"
                        )
                elif first_attempt_summary is not None:
                    out_obj["chunk_summary"] = first_attempt_summary

            compliance = compute_style_compliance(fingerprint, filter_author_voice_text(final_md))
            score_val = compliance.get("score") if isinstance(compliance, dict) else None
            if not isinstance(score_val, (int, float)):
                score_val = -1.0
            attempt_no = attempt_global
            if best_score is None or score_val > best_score:
                best_score = score_val
                best_attempt = attempt_no
                best_md = final_md
                best_out = out_obj
                best_comp = compliance
            if args.verbose and chunk_index is not None and chunk_total is not None:
                comp_score = compliance.get("score")
                if isinstance(comp_score, (int, float)):
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} attempt {attempt_no}/{max_chunk_attempts} "
                        f"compliance score: {comp_score:.3f} "
                        f"(threshold {args.style_retry_threshold})"
                    )
                else:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} attempt {attempt_no}/{max_chunk_attempts} "
                        f"compliance score: {comp_score} "
                        f"(threshold {args.style_retry_threshold})"
                    )

            # If we forced narrative person, the voice-phase output is used only to generate
            # compliance deltas; do not count it as a "style attempt" so budgets add.
            if (
                args.force_person
                and not style_phase_started
                and not args.no_style_retry
                and compliance["score"] < args.style_retry_threshold
            ):
                style_feedback = {
                    "score": compliance["score"],
                    "deltas": compliance.get("deltas", [])
                }
                if controller_overlay and isinstance(tunables, dict):
                    controller_conf = tunables.get("humanization_controller", {}) if isinstance(tunables.get("humanization_controller", {}), dict) else {}
                    if controller_conf.get("feedback_enabled", False):
                        max_feedback_retries = int(controller_conf.get("max_feedback_retries", args.max_style_retries))
                        if style_retries_used < max_feedback_retries:
                            overlay_feedback = build_overlay_feedback(
                                controller_overlay,
                                filter_author_voice_text(final_md),
                                controller_conf
                            )
                            if overlay_feedback:
                                style_feedback["humanization_controller"] = overlay_feedback
                # If the voice budget is exhausted but violations remain, carry them into the
                # style feedback so the model still gets a chance to fix voice while improving compliance.
                voice_text = filter_author_voice_text(final_md)
                override_eval = evaluate_pronoun_override(voice_text, args.force_person)
                if override_eval.get("violations"):
                    style_feedback["pronoun_override"] = override_eval
                    style_feedback.setdefault("notes", []).append(
                        "Pronoun override violations detected. Rewrite to match forced narrative voice."
                    )
                    if args.verbose and chunk_index is not None and chunk_total is not None:
                        reason = format_pronoun_override_debug(override_eval)
                        vprint(
                            f"Chunk {chunk_index}/{chunk_total} carrying pronoun override "
                            f"violations into style feedback: {reason}"
                        )
                style_phase_started = True
                continue

            if (
                not args.no_style_retry
                and style_phase_started
                and style_retries_used < args.max_style_retries
                and compliance["score"] < args.style_retry_threshold
            ):
                if args.verbose and chunk_index is not None and chunk_total is not None:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} below threshold; "
                        f"retrying (style retry {style_retries_used + 1}/{args.max_style_retries})."
                    )
                style_feedback = {
                    "score": compliance["score"],
                    "deltas": compliance.get("deltas", [])
                }
                if controller_overlay and isinstance(tunables, dict):
                    controller_conf = tunables.get("humanization_controller", {}) if isinstance(tunables.get("humanization_controller", {}), dict) else {}
                    if controller_conf.get("feedback_enabled", False):
                        max_feedback_retries = int(controller_conf.get("max_feedback_retries", args.max_style_retries))
                        if style_retries_used < max_feedback_retries:
                            overlay_feedback = build_overlay_feedback(
                                controller_overlay,
                                filter_author_voice_text(final_md),
                                controller_conf
                            )
                            if overlay_feedback:
                                style_feedback["humanization_controller"] = overlay_feedback
                style_retries_used += 1
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
            if best_md is not None and best_out is not None and best_comp is not None:
                if args.verbose and chunk_index is not None and chunk_total is not None and best_attempt != attempt_no:
                    vprint(
                        f"Chunk {chunk_index}/{chunk_total} using best attempt "
                        f"{best_attempt} (score {best_score:.3f}) over last attempt "
                        f"{attempt_no} (score {score_val:.3f})."
                    )
                return best_md, best_out, best_comp
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
        single_chunk_cfg = cfg
        if roster_cfgs:
            roster_idx = build_roster_indices(len(roster_cfgs), 1, roster_seed)[0]
            single_chunk_cfg = roster_cfgs[roster_idx]
            if args.verbose:
                vprint(
                    f"Single chunk roster model {roster_idx + 1}/{len(roster_cfgs)}: "
                    f"{single_chunk_cfg.model} @ {single_chunk_cfg.base_url}"
                )
        try:
            final_md, out_obj, compliance = rewrite_chunk(
                input_md,
                None,
                None,
                0,
                None,
                summary_words,
                False,
                single_chunk_cfg
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
        # Function: Build messages for chunk est.
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
        roster_indices: List[int] = []
        if roster_cfgs:
            roster_indices = build_roster_indices(len(roster_cfgs), len(chunks), roster_seed)
        running_summary = ""
        for idx, chunk in enumerate(chunks, start=1):
            chunk_cfg = cfg
            if roster_cfgs:
                roster_idx = roster_indices[idx - 1]
                chunk_cfg = roster_cfgs[roster_idx]
                if args.verbose:
                    vprint(
                        f"Chunk {idx}/{len(chunks)} roster model {roster_idx + 1}/{len(roster_cfgs)}: "
                        f"{chunk_cfg.model} @ {chunk_cfg.base_url}"
                    )
            vprint(f"Rewriting chunk {idx}/{len(chunks)}...")
            try:
                final_md, out_obj, compliance = rewrite_chunk(
                    chunk,
                    idx,
                    len(chunks),
                    0,
                    running_summary,
                    summary_words,
                    summary_active,
                    chunk_cfg
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
    # Section restoration can reintroduce original orthography; enforce locale spelling once more
    # on mutable regions only so preserved non-voice blocks and protected quotes stay verbatim.
    if force_local_spelling_rules != "none":
        final_md, post_restore_replacements = enforce_local_spelling_guarded(
            final_md,
            force_local_spelling_rules,
            local_spelling_rules,
            preserve_multiword_quotes=(not fiction_mode),
        )
        if post_restore_replacements:
            all_deviations.append({
                "rule_or_field": "orthography.local_spelling",
                "reason": (
                    f"Re-applied {force_local_spelling_rules} spelling via deterministic rules "
                    "after section restoration."
                ),
                "count": post_restore_replacements,
            })
    if (
        heading_case_mode == "identical"
        or (
            heading_case_mode == "by-level"
            and any(
                str(heading_case_by_level.get(lvl, "automatic")).lower() != "automatic"
                for lvl in range(1, MAX_CONFIG_HEADING_LEVEL + 1)
            )
        )
    ):
        final_md, heading_case_edits = enforce_heading_case_normalization_from_source(
            original_input_md,
            final_md,
            heading_case_mode,
            heading_case_by_level,
            preserve_proper_name_case=preserve_proper_name_case,
        )
        if heading_case_edits:
            all_deviations.append({
                "rule_or_field": "structure.heading_case",
                "reason": f"Applied heading case normalization mode '{heading_case_mode}' (humanizer_mandatory).",
                "count": heading_case_edits,
            })
    postprocess_conf = get_postprocess_redundancy_conf(tunables)
    postprocess_enabled = bool(postprocess_conf.get("enabled"))
    para_enabled = False
    list_enabled = False
    para_dropped_blocks = 0
    list_throttled_runs = 0
    list_merged_items = 0
    if postprocess_enabled:
        para_conf = postprocess_conf.get("paragraph_dedupe", {})
        if isinstance(para_conf, dict) and para_conf.get("enabled", True):
            para_enabled = True
            final_md, para_dropped_blocks = dedupe_redundant_prose_blocks(
                final_md,
                min_words=int(para_conf.get("min_words", 30)),
                similarity_threshold=float(para_conf.get("similarity_threshold", 0.985)),
                lookback_blocks=int(para_conf.get("lookback_blocks", 20)),
                max_drop_ratio=float(para_conf.get("max_drop_ratio", 0.15)),
            )
            if para_dropped_blocks:
                all_deviations.append({
                    "rule_or_field": "postprocess_redundancy.paragraph_dedupe",
                    "reason": "Removed near-duplicate prose blocks deterministically.",
                    "count": para_dropped_blocks,
                })
        list_conf = postprocess_conf.get("list_density", {})
        if isinstance(list_conf, dict) and list_conf.get("enabled", True):
            list_enabled = True
            final_md, list_throttled_runs, list_merged_items = throttle_unordered_list_density(
                final_md,
                min_run_length=int(list_conf.get("min_run_length", 9)),
                group_size=int(list_conf.get("group_size", 2)),
                joiner=str(list_conf.get("joiner", "; ")),
            )
            if list_merged_items:
                all_deviations.append({
                    "rule_or_field": "postprocess_redundancy.list_density",
                    "reason": "Grouped long unordered-list runs to reduce repetitive list density.",
                    "runs": list_throttled_runs,
                    "merged_items": list_merged_items,
                })
    if args.verbose:
        vprint(
            "Post-process redundancy: "
            f"enabled={postprocess_enabled}; "
            f"paragraph_dedupe(enabled={para_enabled}, dropped_blocks={para_dropped_blocks}); "
            f"list_density(enabled={list_enabled}, runs={list_throttled_runs}, merged_items={list_merged_items})."
        )
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

    roster_suffix = ""
    if roster_enabled:
        roster_suffix = "_roster"
        if roster_seed is not None:
            roster_suffix += str(roster_seed)
    out_path = args.out or args.inp.with_suffix(args.inp.suffix + f".styled{roster_suffix}.md")
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
