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
  python apply_fingerprint.py -f fingerprint.json -i draft.md
  python apply_fingerprint.py -f fingerprint.json -i draft.md -o draft.styled.md
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
BASE64_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\\s]+", re.IGNORECASE)

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


def strip_base64_images(text: str) -> tuple[str, Dict[str, str]]:
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
    if not mapping:
        return text
    if placeholders is None:
        placeholders = list(mapping.keys())
    for placeholder in placeholders:
        if placeholder in text:
            text = text.replace(placeholder, mapping[placeholder])
    return text


def find_base64_placeholders(text: str) -> List[str]:
    return re.findall(r"\\[\\[BASE64_IMAGE_\\d+\\]\\]", text)


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token.
    return max(1, (len(text) + 3) // 4)


def estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # per-message overhead
    return total + 2


def split_markdown_blocks(markdown: str) -> List[str]:
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
    lines = block.splitlines()
    if not lines:
        return False
    first = lines[0].strip()
    last = lines[-1].strip()
    if (first.startswith("```") or first.startswith("~~~")) and (last.startswith("```") or last.startswith("~~~")):
        return True
    return False


def split_oversize_block(block: str, build_messages_fn, max_prompt_tokens: int) -> List[str]:
    if estimate_tokens_for_messages(build_messages_fn(block)) <= max_prompt_tokens:
        return [block]

    if is_code_block(block):
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
            "Preserve any [[BASE64_IMAGE_N]] placeholders exactly; do not remove or alter them.",
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
        cwd_cfg = Path.cwd() / "config.llm.json"
        script_cfg = Path(__file__).resolve().parent / "config.llm.json"
        args.config = cwd_cfg if cwd_cfg.exists() else script_cfg

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    vprint(f"Using config: {args.config}")

    cfg = load_config(args.config)
    if args.max_prompt_tokens is not None:
        cfg.max_prompt_tokens = args.max_prompt_tokens

    if not args.fingerprint.exists():
        print(f"Fingerprint not found: {args.fingerprint}", file=sys.stderr)
        return 2
    if not args.inp.exists():
        print(f"Input markdown not found: {args.inp}", file=sys.stderr)
        return 2

    vprint("Loading fingerprint and input...")
    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    input_md = args.inp.read_text(encoding="utf-8")
    input_md, base64_map = strip_base64_images(input_md)
    if base64_map:
        vprint(f"Stripped {len(base64_map)} base64 image embeds from prompt.")

    all_deviations: List[Any] = []
    outputs: List[str] = []

    def build_messages_for_chunk(md_chunk: str) -> List[Dict[str, str]]:
        input_meas = compute_measurements(md_chunk)
        return build_apply_prompt(fingerprint, md_chunk, input_meas, cfg)

    initial_messages = build_messages_for_chunk(input_md)
    initial_tokens = estimate_tokens_for_messages(initial_messages)
    if initial_tokens <= cfg.max_prompt_tokens:
        vprint("Calling LLM to apply fingerprint...")
        raw = chat_completions(cfg, initial_messages)

        try:
            out_obj = parse_json_strict(raw)
        except Exception:
            vprint("Invalid JSON returned; attempting repair...")
            out_obj = repair_json_with_llm(cfg, raw)

        final_md = out_obj.get("final_markdown")
        if not isinstance(final_md, str) or not final_md.strip():
            print("LLM did not return final_markdown.", file=sys.stderr)
            print(raw)
            return 3
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
                out_obj = repair_json_with_llm(cfg, raw)

            final_md = out_obj.get("final_markdown")
            if not isinstance(final_md, str) or not final_md.strip():
                print("LLM did not return final_markdown.", file=sys.stderr)
                print(raw)
                return 3
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
