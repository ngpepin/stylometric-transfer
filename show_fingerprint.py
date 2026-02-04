#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
"""
show_fingerprint.py

Render a compact HTML dashboard for a style fingerprint JSON.

Usage:
  python show_fingerprint.py path/to/fingerprint.json
  python show_fingerprint.py path/to/fingerprint.json -o output.html
  python show_fingerprint.py path/to/fingerprint.json -o output.html --open
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import textwrap
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(text: Any) -> str:
    return html.escape(str(text))


def fmt_num(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return str(value)
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"
    return esc(value)


def safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def take_items(items: Iterable[Any], limit: int = 10) -> List[Any]:
    out: List[Any] = []
    for item in items:
        out.append(item)
        if len(out) >= limit:
            break
    return out


def load_common_word_frequencies() -> Dict[str, float]:
    # Load common-word Zipf frequencies from config.common_words.txt if present.
    filename = "config.common_words.txt"
    cwd_path = Path.cwd() / filename
    script_path = Path(__file__).resolve().parent / filename
    path = cwd_path if cwd_path.exists() else script_path if script_path.exists() else None
    if not path:
        return {}
    freq_map: Dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if not parts:
            continue
        word = parts[0].strip().lower()
        if not word or len(parts) < 2:
            continue
        try:
            freq_map[word] = float(parts[1])
        except Exception:
            continue
    return freq_map


def truncate_text(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "…"


def clean_work_label(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"[^\w\s\-&:'’]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_quality_label(text: str) -> bool:
    if not text:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    return alnum >= 4 and (alnum / max(1, len(text))) >= 0.35


def render_work_list(documents: List[Dict[str, Any]], limit: int = 8, max_len: int = 60) -> str:
    if not documents:
        return "<p class='muted'>No works listed</p>"
    labels: List[str] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        name = doc.get("name") or ""
        title = doc.get("title") or ""
        path = doc.get("path") or ""
        options: List[str] = []
        if isinstance(name, str):
            options.append(clean_work_label(name))
        if isinstance(title, str):
            options.append(clean_work_label(title))
        if isinstance(path, str) and path:
            options.append(clean_work_label(Path(path).stem))
        label = ""
        for opt in options:
            if is_quality_label(opt):
                label = opt
                break
        if not label:
            for opt in options:
                if opt:
                    label = opt
                    break
        label = label.strip()
        if label:
            labels.append(label)
    if not labels:
        return "<p class='muted'>No works listed</p>"
    shown = take_items(labels, limit)
    items = "".join(
        f"<li class='doc-item' title='{esc(name)}'>{esc(truncate_text(name, max_len))}</li>"
        for name in shown
    )
    remainder = len(labels) - len(shown)
    suffix = f"<li class='doc-item muted'>+{remainder} more</li>" if remainder > 0 else ""
    return f"<ul class='doc-list'>{items}{suffix}</ul>"


def svg_bar_chart(labels: List[str], values: List[float], width: int = 840, height: int = 300, color: str = "#2E6DD8") -> str:
    if not values:
        return ""
    max_val = max(values) if max(values) > 0 else 1.0
    bar_w = width / max(1, len(values))
    bars = []
    grid = []
    for i in range(5):
        y = (height - 30) * (i / 4) + 10
        grid.append(f"<line x1='0' y1='{y:.1f}' x2='{width}' y2='{y:.1f}' stroke='#1f2937' stroke-width='1' />")
    for i, v in enumerate(values):
        h = (v / max_val) * (height - 30)
        x = i * bar_w + 6
        y = height - h - 16
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w - 12:.1f}' height='{h:.1f}' rx='4' fill='{color}'>"
            f"<title>{esc(labels[i])}: {fmt_num(v, 3)}</title></rect>"
        )
        bars.append(
            f"<text x='{x + (bar_w - 12)/2:.1f}' y='{height - 2:.1f}' fill='#94a3b8' font-size='12' text-anchor='middle'>{esc(labels[i])}</text>"
        )
        bars.append(
            f"<text x='{x + (bar_w - 12)/2:.1f}' y='{max(16, y - 6):.1f}' fill='#e2e8f0' font-size='12' text-anchor='middle'>{fmt_num(v, 3)}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}'>{''.join(grid)}{''.join(bars)}</svg>"


def svg_dot_chart(labels: List[str], values: List[float], width: int = 840, height: int = 340, color: str = "#16A34A") -> str:
    if not values:
        return ""
    max_val = max(values) if max(values) > 0 else 1.0
    dots = []
    grid = []
    for i in range(5):
        x = (width - 60) * (i / 4) + 50
        grid.append(f"<line x1='{x:.1f}' y1='0' x2='{x:.1f}' y2='{height}' stroke='#1f2937' stroke-width='1' />")
    step = height / max(1, len(values))
    for i, v in enumerate(values):
        y = (i + 0.8) * step
        x = 50 + (v / max_val) * (width - 80)
        dots.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{color}'>"
                    f"<title>{esc(labels[i])}: {fmt_num(v, 3)}</title></circle>")
        dots.append(f"<text x='0' y='{y + 4:.1f}' fill='#94a3b8' font-size='12'>{esc(labels[i])}</text>")
        dots.append(f"<text x='{x + 12:.1f}' y='{y + 4:.1f}' fill='#e2e8f0' font-size='12'>{fmt_num(v, 3)}</text>")
    return f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}'>{''.join(grid)}{''.join(dots)}</svg>"


def card(title: str, body: str, accent: str | None = None) -> str:
    accent_style = f" style='border-top: 2px solid {accent};'" if accent else ""
    return f"""
    <section class='card'{accent_style}>
      <h3>{esc(title)}</h3>
      {body}
    </section>
    """


def list_kv(items: List[Tuple[str, Any]]) -> str:
    rows = []
    for k, v in items:
        rows.append(f"<div class='kv'><span class='k'>{esc(k)}</span><span class='v'>{esc(v)}</span></div>")
    return "".join(rows)


def chips(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return ""
    return "<div class='chip-list'>" + "".join(f"<span class='chip'>{esc(item)}</span>" for item in items) + "</div>"


def chips_with_titles(items: Iterable[Tuple[str, Optional[str]]]) -> str:
    items = list(items)
    if not items:
        return ""
    parts = []
    for label, title in items:
        title_attr = f" title='{esc(title)}'" if title else ""
        parts.append(f"<span class='chip'{title_attr}>{esc(label)}</span>")
    return "<div class='chip-list'>" + "".join(parts) + "</div>"


def json_block(obj: Any) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=True)
    except Exception:
        text = str(obj)
    return f"<pre>{esc(text)}</pre>"


def details_block(title: str, content: str, open_state: bool = False) -> str:
    open_attr = " open" if open_state else ""
    return f"<details{open_attr}><summary>{esc(title)}</summary>{content}</details>"


def render_dashboard(fp: Dict[str, Any], source_path: Path) -> str:
    profile_id = fp.get("profile_id", "(unknown)")
    schema_version = fp.get("schema_version", "")
    author = safe_get(fp, "metadata", "author", "name", default="")
    author_is_self = safe_get(fp, "metadata", "author", "is_self", default=None)
    totals = safe_get(fp, "measurements", "totals", default={}) or {}
    sent_len = safe_get(fp, "measurements", "sentence", "length_words", default={}) or {}
    para_hist_bins = safe_get(fp, "measurements", "paragraph", "length_sentences_histogram_bins", default=[]) or []
    para_hist_vals = safe_get(fp, "measurements", "paragraph", "length_sentences_histogram_p", default=[]) or []
    sentence_bins = safe_get(fp, "measurements", "sentence", "length_words", "histogram_bins", default=[]) or []
    sentence_vals = safe_get(fp, "measurements", "sentence", "length_words", "histogram_p", default=[]) or []
    punctuation_rates = safe_get(fp, "measurements", "punctuation", "rates_per_1000w", default={}) or {}
    function_words = safe_get(fp, "measurements", "function_words", "top", default=[]) or []
    stance = safe_get(fp, "measurements", "stance_signals", default={}) or {}
    lexicon = fp.get("lexicon", {}) or {}
    templates = fp.get("templates", {}) or {}
    controls = fp.get("controls", {}) or {}
    validators = fp.get("validators", {}) or {}
    targets = fp.get("targets", {}) or {}
    extraction = safe_get(fp, "metadata", "extraction", default={}) or {}
    corpus_docs = safe_get(fp, "metadata", "corpus", "documents", default=[]) or []

    header_kv = list_kv([
        ("Profile", profile_id),
        ("Schema", schema_version),
        ("Author", author if author else "(unknown)"),
        ("Self", author_is_self if author_is_self is not None else "(unknown)"),
        ("Source", source_path.name),
    ])

    totals_kv = list_kv([
        ("Documents", totals.get("documents_used", "")),
        ("Words", totals.get("total_words_est", "")),
        ("Sentences", totals.get("total_sentences_est", "")),
        ("Paragraphs", totals.get("total_paragraphs_est", "")),
    ])
    works_list = render_work_list(corpus_docs, limit=8, max_len=64)
    totals_block = totals_kv + f"<div class='subhead'>Works</div>{works_list}"

    sentence_chart = svg_bar_chart(
        [str(b) for b in sentence_bins],
        [float(v) for v in sentence_vals] if sentence_vals else [],
        color="#7C3AED"
    )
    para_chart = svg_bar_chart(
        [str(b) for b in para_hist_bins],
        [float(v) for v in para_hist_vals] if para_hist_vals else [],
        color="#0EA5E9"
    )

    punct_items = sorted(punctuation_rates.items(), key=lambda kv: kv[1], reverse=True)
    punct_items = take_items(punct_items, 8)
    punct_chart = svg_dot_chart(
        [k for k, _ in punct_items],
        [float(v) for _, v in punct_items],
        color="#F97316"
    ) if punct_items else ""

    fn_words = take_items(function_words, 10)
    fn_list = "".join(
        f"<div class='pill'><span>{esc(item.get('word',''))}</span><strong>{fmt_num(item.get('count',''))}</strong></div>"
        for item in fn_words
    )

    stance_items = [(k, fmt_num(v)) for k, v in stance.items()]
    stance_chart = svg_dot_chart(
        [k for k, _ in stance_items],
        [float(v) for _, v in stance_items],
        color="#22C55E"
    ) if stance_items else ""

    rhetoric = safe_get(fp, "measurements", "rhetoric_moves", default={}) or {}
    rhetoric_keys = ["claim_rate", "evidence_rate", "counterpoint_rate", "concession_rate", "synthesis_rate"]
    rhetoric_items = [(k.replace("_rate", ""), float(rhetoric.get(k, 0.0))) for k in rhetoric_keys if k in rhetoric]
    rhetoric_chart = svg_dot_chart(
        [k for k, _ in rhetoric_items],
        [v for _, v in rhetoric_items],
        color="#F472B6"
    ) if rhetoric_items else ""
    if rhetoric_chart and "claim_evidence_ratio" in rhetoric:
        ratio_label = fmt_num(rhetoric.get("claim_evidence_ratio", 0.0), 3)
        rhetoric_chart = (
            rhetoric_chart
            + f"<div class='ratio-pill'>claim/evidence ratio: {ratio_label}</div>"
        )

    freq_map = load_common_word_frequencies()
    lex_prefer = take_items(lexicon.get("prefer_words", []) or [], 100)

    def sort_by_freq(words: Iterable[str]) -> List[str]:
        def key(w: str) -> Tuple[float, str]:
            freq = freq_map.get(w.lower())
            freq_val = float(freq) if isinstance(freq, (int, float)) else -1.0
            return (-freq_val, w.lower())
        return sorted([w for w in words if isinstance(w, str)], key=key)

    lex_avoid_hard = take_items(sort_by_freq(lexicon.get("avoid_words", []) or []), 100)
    lex_avoid_soft = take_items(sort_by_freq(lexicon.get("avoid_words_soft", []) or []), 100)

    template_openers = take_items(templates.get("sentence_openers", []) or [], 8)
    template_trans = take_items(templates.get("transition_openers", []) or [], 8)

    controls_kv = list_kv([
        ("Rewrite policy", controls.get("rewrite_policy", "")),
        ("Priority order", ", ".join(controls.get("priority_order", []) or [])),
    ])
    strictness = controls.get("strictness", {}) or {}
    if isinstance(strictness, dict):
        strictness_kv = list_kv([(k, v) for k, v in strictness.items()])
    else:
        strictness_kv = list_kv([("Strictness", strictness)])

    validator_checks = take_items(validators.get("checks", []) or [], 8)
    validator_weights = validators.get("scoring_weights", validators.get("weights", {})) or {}
    if isinstance(validator_weights, dict):
        validator_kv = list_kv([(k, fmt_num(v)) for k, v in validator_weights.items()])
    else:
        validator_kv = list_kv([("Weights", validator_weights)])

    extraction_kv = list_kv([
        ("Model", extraction.get("model", "")),
        ("Date", extraction.get("date", "")),
        ("Confidence", extraction.get("confidence", "")),
    ])

    persona = safe_get(fp, "targets", "persona", default={}) or {}
    pronouns = safe_get(persona, "pronoun_preferences", default={}) or {}
    persona_kv = list_kv([
        ("Default pronouns", pronouns.get("default_set", "")),
        ("Allowed", ", ".join(pronouns.get("allowed_sets", []) or [])),
        ("Avoid", ", ".join(pronouns.get("avoid_sets", []) or [])),
        ("Strictness", pronouns.get("strictness", "")),
    ])

    rare_words = safe_get(fp, "measurements", "lexical_signals", "rare_words", default=[]) or []
    rare_words_sorted = sorted(
        [item for item in rare_words if isinstance(item, dict)],
        key=lambda x: str(x.get("word", "")).lower()
    )
    rare_words_sorted = take_items(rare_words_sorted, 100)
    rare_chip_items: List[Tuple[str, Optional[str]]] = []
    for item in rare_words_sorted:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        title = f"count: {fmt_num(item.get('count',''))}, rate/1k: {fmt_num(item.get('rate_per_1000w',''))}"
        rare_chip_items.append((word, title))
    rare_list = chips_with_titles(rare_chip_items)

    avoidance_words = safe_get(fp, "measurements", "lexical_avoidance", "rare_words", default=[]) or []
    avoidance_items = [item for item in avoidance_words if isinstance(item, dict)]
    def avoid_sort_key(item: Dict[str, Any]) -> Tuple[float, str]:
        word = str(item.get("word", "")).lower()
        freq = item.get("zipf_frequency")
        if freq is None:
            freq = freq_map.get(word)
        freq_val = float(freq) if isinstance(freq, (int, float)) else -1.0
        return (-freq_val, word)
    avoidance_sorted = sorted(avoidance_items, key=avoid_sort_key)
    avoidance_sorted = take_items(avoidance_sorted, 100)
    avoidance_chip_items: List[Tuple[str, Optional[str]]] = []
    for item in avoidance_sorted:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        freq = item.get("zipf_frequency")
        freq_label = f", zipf: {fmt_num(freq, 3)}" if isinstance(freq, (int, float)) else ""
        title = f"count: {fmt_num(item.get('count',''))}, rate/1k: {fmt_num(item.get('rate_per_1000w',''))}{freq_label}"
        avoidance_chip_items.append((word, title))
    avoidance_list = chips_with_titles(avoidance_chip_items)
    avoidance_card = ""
    if not lex_avoid_soft and avoidance_list:
        avoidance_card = card("Avoidance Words", avoidance_list, accent="#F59E0B")

    target_sections = []
    for key, block in targets.items():
        target_sections.append(details_block(key, json_block(block)))

    settings_sections = [
        ("Controls", controls),
        ("Targets", targets),
        ("Lexicon", lexicon),
        ("Templates", templates),
        ("Validators", validators),
        ("Derived Instructions", fp.get("derived_instructions", {})),
        ("Metadata", fp.get("metadata", {})),
        ("Measurements (raw)", fp.get("measurements", {})),
    ]
    settings_blocks = "".join(details_block(title, json_block(block)) for title, block in settings_sections)

    html_out = f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>Fingerprint Dashboard - {esc(profile_id)}</title>
<style>
:root {{
  --bg: #0f172a;
  --card: #111827;
  --muted: #94a3b8;
  --text: #e2e8f0;
  --accent: #2E6DD8;
  --green: #16A34A;
  --border: #1f2937;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at 20% 20%, #1e293b, #0b1020 55%); color: var(--text); }}
header {{ padding: 28px 36px; border-bottom: 1px solid var(--border); }}
header h1 {{ margin: 0 0 6px 0; font-size: 28px; letter-spacing: 0.2px; }}
header p {{ margin: 0; color: var(--muted); }}
main {{ padding: 24px 36px 48px; }}
.layout {{ display: grid; grid-template-columns: 900px 1fr; gap: 24px; align-items: start; }}
.charts-panel {{ display: flex; flex-direction: column; gap: 20px; }}
.grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(3,7,18,0.35); }}
.chart-card h3 {{ margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); }}
.ratio-pill {{ display: inline-block; margin-top: 8px; padding: 6px 10px; border-radius: 999px; background: #1f2937; color: #fbcfe8; font-size: 12px; border: 1px solid #3f3f46; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(3,7,18,0.35); position: relative; overflow: hidden; }}
.card:before {{ content: ""; position: absolute; inset: 0; opacity: 0.05; background: linear-gradient(135deg, #ffffff, transparent 60%); pointer-events: none; }}
.card h3 {{ margin: 0 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); }}
.kv {{ display:flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed #1f2937; font-size: 13px; }}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{ color: var(--muted); }}
.kv .v {{ color: var(--text); font-weight: 600; max-width: 60%; text-align: right; }}
.pill {{ display:flex; justify-content: space-between; background: #0b1225; border: 1px solid #1f2a44; border-radius: 10px; padding: 6px 10px; margin: 4px 0; font-size: 12px; }}
.subhead {{ margin-top: 12px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
.doc-list {{ list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 6px; }}
.doc-item {{ background: #0b1225; border: 1px solid #1f2a44; border-radius: 8px; padding: 6px 10px; font-size: 12px; color: var(--text); }}
.chip-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ display:inline-block; background: #1f2937; color: #e5e7eb; padding: 3px 8px; border-radius: 999px; font-size: 11px; line-height: 1.4; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.section {{ margin-top: 22px; }}
.section h2 {{ font-size: 18px; margin: 0 0 12px; color: #e5e7eb; }}
pre {{ white-space: pre-wrap; background: #0b1225; padding: 10px; border-radius: 8px; border: 1px solid #1f2a44; font-size: 11px; color: #d1d5db; }}
details summary {{ cursor: pointer; color: #93c5fd; margin-bottom: 6px; font-weight: 600; }}
footer {{ color: var(--muted); padding: 24px 36px; font-size: 12px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>Style Fingerprint Dashboard</h1>
  <p>{esc(profile_id)} - {esc(author)} - Schema {esc(schema_version)}</p>
</header>
<main>
  <div class='layout'>
    <div class='charts-panel'>
      <section class='chart-card'>
        <h3>Sentence Lengths</h3>
        {sentence_chart or "<p class='muted'>No data</p>"}
      </section>
      <section class='chart-card'>
        <h3>Paragraph Lengths</h3>
        {para_chart or "<p class='muted'>No data</p>"}
      </section>
      <section class='chart-card'>
        <h3>Punctuation Density</h3>
        {punct_chart or "<p class='muted'>No data</p>"}
      </section>
      <section class='chart-card'>
        <h3>Stance Signals</h3>
        {stance_chart or "<p class='muted'>No data</p>"}
      </section>
      <section class='chart-card'>
        <h3>Rhetorical Moves</h3>
        {rhetoric_chart or "<p class='muted'>No data</p>"}
      </section>
    </div>
    <div class='grid'>
      {card("Overview", header_kv, accent="#60A5FA")}
      {card("Corpus Totals", totals_block, accent="#38BDF8")}
      {card("Persona", persona_kv or "<p class='muted'>None</p>", accent="#4ADE80")}
      {card("Lexicon - Prefer", chips(lex_prefer) or "<p class='muted'>None</p>", accent="#38BDF8")}
      {card("Lexicon - Rare Words", rare_list or "<p class='muted'>None</p>", accent="#10B981")}
      {card("Lexicon - Avoid (Soft)", chips(lex_avoid_soft) or "<p class='muted'>None</p>", accent="#F87171")}
      {card("Templates - Openers", chips(template_openers) or "<p class='muted'>None</p>", accent="#818CF8")}
      {card("Templates - Transitions", chips(template_trans) or "<p class='muted'>None</p>", accent="#6366F1")}
      {card("Lexicon - Avoid (Hard)", chips(lex_avoid_hard) or "<p class='muted'>None</p>", accent="#EF4444")}
      {card("Function Words", fn_list or "<p class='muted'>No data</p>", accent="#94A3B8")}
      {card("Controls", controls_kv or "<p class='muted'>None</p>", accent="#94A3B8")}
      {card("Validator Weights", validator_kv or "<p class='muted'>None</p>", accent="#FACC15")}
      {card("Extraction", extraction_kv or "<p class='muted'>None</p>", accent="#22C55E")}
      {card("Strictness", strictness_kv or "<p class='muted'>None</p>", accent="#94A3B8")}
    </div>
  </div>

  <div class='section'>
    <h2>Target Blocks</h2>
    {''.join(target_sections) if target_sections else '<p class="muted">No target blocks available.</p>'}
  </div>

  <div class='section'>
    <h2>Validator Checks</h2>
    {''.join(f"<div class='pill'><span>{esc(c)}</span></div>" for c in validator_checks) if validator_checks else '<p class="muted">No checks.</p>'}
  </div>

  <div class='section'>
    <h2>All Settings (Raw)</h2>
    {settings_blocks}
  </div>
</main>
<footer>
  Generated by show_fingerprint.py - {esc(source_path)}
</footer>
</body>
</html>
"""
    return html_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fingerprint", type=Path, help="Path to fingerprint JSON")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output HTML path (default: <profile_id>_dashboard.html)")
    ap.add_argument("--open", action="store_true", help="Open the generated dashboard in a browser")
    args = ap.parse_args()

    fp = load_json(args.fingerprint)
    profile_id = fp.get("profile_id", "fingerprint")
    out_path = args.out or Path.cwd() / f"{profile_id}_dashboard.html"
    html_out = render_dashboard(fp, args.fingerprint)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote dashboard to: {out_path}")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
