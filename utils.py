#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
from __future__ import annotations

import re
import statistics
from typing import List, Optional, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")



# Function: Tokenize text into simple word-like units.
def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


# Function: Split text into sentences using a simple heuristic.
def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


# Function: Split text into paragraphs separated by blank lines.
def split_paragraphs(text: str) -> List[str]:
    paras = PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]


# Function: Clamp a value into the [0, 1] range.
def clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


# Function: Build a normalized histogram from integer values and bins.
def histogram(values: List[int], bins: List[Tuple[int, Optional[int]]]) -> List[float]:
    if not bins:
        return []
    counts = [0 for _ in bins]
    for v in values:
        for i, (lo, hi) in enumerate(bins):
            if hi is None:
                if v >= lo:
                    counts[i] += 1
                    break
            elif lo <= v <= hi:
                counts[i] += 1
                break
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


# Function: Convert counts into per-1000-word rates.
def approx_rate_per_1000_words(count: int, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    return (count / total_words) * 1000.0


# Function: Compute a mean with a zero fallback.
def safe_mean(xs: List[int]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


# Function: Compute a population standard deviation with a zero fallback.
def safe_stdev(xs: List[int]) -> float:
    return float(statistics.pstdev(xs)) if xs else 0.0
