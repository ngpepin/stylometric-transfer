#!/usr/bin/env python3
"""
apply_fingerprint.py

Rewrite a Markdown file to match a style fingerprint JSON.

It:
1) Loads fingerprint JSON
2) Computes measurements of the input markdown (so the LLM can see deltas)
3) Calls an OpenAI-compatible LLM to rewrite the markdown
4) Writes rewritten markdown to an output file (default: <input>.styled.md)

Usage:
  python apply_fingerprint.py -c config.llm.json -f fingerprint.json -i draft.md
  python apply_fingerprint.py -c config.llm.json -f fingerprint.json -i draft.md -o draft.styled.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests


# ---- same lightweight stats as fingerprint script ----

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

def words(text: str) -> List[str]:
    return WORD_RE.findall(text)

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]

def split_paragraphs(text: str) -> List[str]:
    paras = PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]

def histogram(values: List[int], bins: List[tuple[int, int | None]]) -> List[float]:
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

def approx_rate_per_1000_words(count: int, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    return (count / total_words) * 1000.0

def compute_measurements(text: str) -> Dict[str, Any]:
    w = words(text)
    total_words = len(w)

    sent_lens = [len(words(s)) for s in split_sentences(text)]
    sent_bins = [(0,9),(10,17),(18,25),(26,40),(41,None)]
    sent_hist = histogram(sent_lens, sent_bins)

    paras = split_paragraphs(text)
    para_lens = [len(split_sentences(p)) for p in paras]
    para_bins = [(1,1),(2,3),(4,5),(6,8),(9,None)]
    para_hist = histogram(para_lens, para_bins)
    one_sentence_rate = sum(1 for n in para_lens if n == 1) / max(1, len(para_lens))

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
        }
    }


# ---- OpenAI-compatible client ----

class LLMConfig:
    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int, temperature: float, timeout_seconds: int, extra_headers: Dict[str, str]):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.extra_headers = extra_headers

def load_config(path: Path) -> LLMConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LLMConfig(
        api_key=data["api_key"],
        base_url=data["base_url"],
        model=data["model"],
        max_tokens=int(data.get("max_tokens", 6000)),
        temperature=float(data.get("temperature", 0.2)),
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        extra_headers=dict(data.get("extra_headers", {}))
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
    return data["choices"][0]["message"]["content"]

def parse_json_strict(s: str) -> Dict[str, Any]:
    s = s.strip()
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


# ---- Prompting: apply fingerprint ----

def build_apply_prompt(fingerprint: Dict[str, Any], input_md: str, input_meas: Dict[str, Any], cfg: LLMConfig) -> List[Dict[str, str]]:
    system = (
        "You are a style-transfer rewriting engine.\n"
        "You MUST output valid JSON only.\n"
        "Preserve meaning strictly. Do not add new facts, claims, or examples.\n"
        "Keep Markdown structure valid.\n"
        "If a conflict occurs, prioritize clarity and preservation of meaning, and record the deviation.\n"
    )

    user = {
        "task": "Rewrite INPUT_MARKDOWN to match STYLE_FINGERPRINT_JSON.",
        "output_format": {
            "final_markdown": "string",
            "deviations": [
                {"rule_or_field": "json.pointer", "reason": "string"}
            ],
            "self_check": {
                "notes": ["string"]
            }
        },
        "rules": [
            "Preserve meaning strictly; do not add new content.",
            "Preserve code blocks, links, and quoted material unless necessary for style and explicitly allowed.",
            "Follow controls.priority_order if present; otherwise prioritize persona > rhetoric > paragraph > sentence > lexical > punctuation > orthography.",
            "Avoid lexicon.avoid_phrases with severity 'hard' and avoid_words with severity 'hard' if present.",
            "Use preferred phrases sparingly (respect max_per_1000w if present)."
        ],
        "style_fingerprint_json": fingerprint,
        "input_measurements": input_meas,
        "input_markdown": input_md
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=Path, help="Path to config.llm.json")
    ap.add_argument("-f", "--fingerprint", required=True, type=Path, help="Fingerprint JSON from fingerprint_style.py")
    ap.add_argument("-i", "--in", dest="inp", required=True, type=Path, help="Input markdown file to rewrite")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output markdown path (default: <input>.styled.md)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not args.fingerprint.exists():
        print(f"Fingerprint not found: {args.fingerprint}", file=sys.stderr)
        return 2
    if not args.inp.exists():
        print(f"Input markdown not found: {args.inp}", file=sys.stderr)
        return 2

    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    input_md = args.inp.read_text(encoding="utf-8")
    input_meas = compute_measurements(input_md)

    messages = build_apply_prompt(fingerprint, input_md, input_meas, cfg)
    raw = chat_completions(cfg, messages)

    try:
        out_obj = parse_json_strict(raw)
    except Exception:
        out_obj = repair_json_with_llm(cfg, raw)

    final_md = out_obj.get("final_markdown")
    if not isinstance(final_md, str) or not final_md.strip():
        print("LLM did not return final_markdown.", file=sys.stderr)
        # As fallback, write raw response for inspection
        print(raw)
        return 3

    out_path = args.out or args.inp.with_suffix(args.inp.suffix + ".styled.md")
    out_path.write_text(final_md, encoding="utf-8")
    print(f"Wrote rewritten markdown to: {out_path}")

    # Optionally also write deviations report
    deviations = out_obj.get("deviations", [])
    if deviations:
        rep = out_path.with_suffix(out_path.suffix + ".deviations.json")
        rep.write_text(json.dumps(deviations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote deviations report to: {rep}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
