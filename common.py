#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
from __future__ import annotations

import datetime
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

FINGERPRINT_STORE_DIRNAME = "fingerprint_store"
GUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-"
    r"[0-9a-fA-F]{12}$"
)


# Function: Find the repository root by walking up for a .git directory.
def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return cur


# Function: Resolve a filename from cwd first, then script directory.
def resolve_path_prefer_cwd(filename: str, script_file: str | Path) -> Path | None:
    cwd_path = Path.cwd() / filename
    script_path = Path(script_file).resolve().parent / filename
    if cwd_path.exists():
        return cwd_path
    if script_path.exists():
        return script_path
    return None


# Function: Resolve a required config path from CLI override or defaults.
def resolve_required_path(
    override_path: Path | None,
    default_filename: str,
    script_file: str | Path
) -> Path:
    if override_path is not None:
        return override_path
    cwd_path = Path.cwd() / default_filename
    script_path = Path(script_file).resolve().parent / default_filename
    return cwd_path if cwd_path.exists() else script_path


# Function: Resolve an optional path from CLI override or defaults.
def resolve_optional_path(
    override_path: Path | None,
    default_filename: str,
    script_file: str | Path
) -> Path | None:
    if override_path is not None:
        return override_path
    return resolve_path_prefer_cwd(default_filename, script_file)


# Function: Ensure a directory exists.
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Function: Validate a GUID-like identifier.
def is_valid_guid(value: str) -> bool:
    return bool(GUID_RE.match(value.strip()))


# Function: Generate a new GUID for a stored fingerprint.
def new_guid() -> str:
    return str(uuid.uuid4())


# Function: Resolve the fingerprint store directory.
def resolve_fingerprint_store_dir(script_file: str | Path, store_dir: Path | None = None) -> Path:
    if store_dir is not None:
        return ensure_dir(store_dir)
    root = find_repo_root(Path(script_file).resolve())
    return ensure_dir(root / FINGERPRINT_STORE_DIRNAME)


# Function: Return the canonical fingerprint JSON path for a GUID.
def fingerprint_store_json_path(store_dir: Path, guid: str) -> Path:
    return store_dir / f"{guid}.fingerprint.json"


# Function: Return the canonical fingerprint metadata path for a GUID.
def fingerprint_store_meta_path(store_dir: Path, guid: str) -> Path:
    return store_dir / f"{guid}.meta.json"


# Function: Save a fingerprint and metadata into the local GUID store.
def save_fingerprint_to_store(
    fingerprint: Dict[str, Any],
    store_dir: Path,
    guid: str | None = None,
    source: str = "api.make",
    extra_metadata: Dict[str, Any] | None = None
) -> Tuple[str, Path, Path]:
    effective_guid = guid or new_guid()
    if not is_valid_guid(effective_guid):
        raise ValueError("Invalid GUID format")

    ensure_dir(store_dir)
    fp_path = fingerprint_store_json_path(store_dir, effective_guid)
    meta_path = fingerprint_store_meta_path(store_dir, effective_guid)
    created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    profile_id = None
    if isinstance(fingerprint, dict):
        profile_id = fingerprint.get("profile_id")

    fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    meta: Dict[str, Any] = {
        "id": effective_guid,
        "created_at": created_at,
        "source": source,
        "profile_id": profile_id,
        "fingerprint_file": fp_path.name,
    }
    if isinstance(extra_metadata, dict):
        meta.update(extra_metadata)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return effective_guid, fp_path, meta_path


# Function: Load a fingerprint and metadata from the local GUID store.
def load_fingerprint_from_store(
    guid: str,
    store_dir: Path
) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    if not is_valid_guid(guid):
        raise ValueError("Invalid GUID format")
    fp_path = fingerprint_store_json_path(store_dir, guid)
    if not fp_path.exists():
        raise FileNotFoundError(f"Fingerprint id not found: {guid}")
    data = json.loads(fp_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Stored fingerprint JSON is invalid")
    meta_path = fingerprint_store_meta_path(store_dir, guid)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    return data, fp_path, meta


# Function: Clamp a value into [0, 1].
def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# Function: Numerically stable logistic function.
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Function: Convert a compliance score into a calibrated style-match probability.
def calibrated_style_match_probability(
    compliance_score: float,
    token_count: int,
    threshold: float = 0.75,
    slope: float = 0.12,
    evidence_tokens: int = 250
) -> Dict[str, Any]:
    comp = clamp01(float(compliance_score))
    safe_slope = max(1e-6, float(slope))
    threshold = clamp01(float(threshold))

    # Map compliance to an unshrunk probability with logistic calibration.
    unshrunk = sigmoid((comp - threshold) / safe_slope)

    # Shrink toward 0.5 for short segments to reflect weak evidence.
    n = max(0, int(token_count))
    denom = max(1, int(evidence_tokens))
    reliability = 1.0 - math.exp(-n / denom)
    probability = 0.5 + reliability * (unshrunk - 0.5)

    half_width = (1.0 - reliability) * 0.5
    ci_low = clamp01(probability - half_width)
    ci_high = clamp01(probability + half_width)
    probability = clamp01(probability)

    return {
        "probability": probability,
        "probability_percent": round(probability * 100.0, 2),
        "confidence_interval_90": [round(ci_low, 4), round(ci_high, 4)],
        "calibration": {
            "compliance_score": comp,
            "threshold": threshold,
            "slope": safe_slope,
            "unshrunk_probability": unshrunk,
            "token_count": n,
            "reliability": reliability,
            "evidence_tokens": denom,
            "method": "logistic_calibration_with_length_reliability_shrinkage",
        },
    }


# Function: Safely fetch nested dictionary values.
def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# Function: Normalize a vector into a probability distribution.
def _normalize_distribution(values: List[float]) -> List[float]:
    if not values:
        return []
    cleaned = [max(0.0, float(v)) for v in values]
    total = sum(cleaned)
    if total <= 0.0:
        uniform = 1.0 / len(cleaned)
        return [uniform for _ in cleaned]
    return [v / total for v in cleaned]


# Function: Compute Jensen-Shannon divergence in base-2 (bounded in [0,1] for two distributions).
def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    if not p or not q or len(p) != len(q):
        return 1.0
    pp = _normalize_distribution(p)
    qq = _normalize_distribution(q)
    m = [(a + b) / 2.0 for a, b in zip(pp, qq)]
    eps = 1e-12

    # Function: Compute KL divergence.
    def _kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0.0:
                continue
            out += ai * math.log((ai + eps) / (bi + eps), 2)
        return out

    jsd = 0.5 * _kl(pp, m) + 0.5 * _kl(qq, m)
    if not math.isfinite(jsd):
        return 1.0
    return max(0.0, min(1.0, jsd))


# Function: Compute cosine similarity between two vectors.
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    out = dot / (na * nb)
    return max(0.0, min(1.0, out))


# Function: Compute set Jaccard similarity.
def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


# Function: Convert an arbitrary mapping to numeric float map.
def _to_numeric_dict(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in value.items():
        if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
    return out


# Function: Compare two numeric values using symmetric relative similarity.
def _relative_similarity(a: float, b: float) -> float:
    av = float(a)
    bv = float(b)
    denom = max((abs(av) + abs(bv)) / 2.0, 1e-9)
    rel = abs(av - bv) / denom
    # 1/(1+rel) maps 0->1 and grows smoothly toward 0 for large differences.
    return 1.0 / (1.0 + rel)


# Function: Compare two numeric dictionaries and return aggregate similarity diagnostics.
def _compare_numeric_dicts(
    a: Dict[str, float],
    b: Dict[str, float]
) -> Dict[str, Any] | None:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return None
    rows: List[Dict[str, Any]] = []
    sims: List[float] = []
    for key in keys:
        sim = _relative_similarity(a[key], b[key])
        sims.append(sim)
        rows.append(
            {
                "key": key,
                "a": a[key],
                "b": b[key],
                "similarity": sim,
                "abs_diff": abs(a[key] - b[key]),
            }
        )
    rows.sort(key=lambda r: float(r.get("similarity", 0.0)))
    return {
        "similarity": sum(sims) / len(sims),
        "keys_compared": len(keys),
        "worst_keys": rows[:5],
    }


# Function: Compare two distributions represented as equal-length vectors.
def _compare_distribution_vectors(
    a: List[float],
    b: List[float]
) -> Dict[str, Any] | None:
    if not isinstance(a, list) or not isinstance(b, list) or not a or len(a) != len(b):
        return None
    aa = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0 for x in a]
    bb = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0 for x in b]
    jsd = jensen_shannon_divergence(aa, bb)
    l1 = sum(abs(x - y) for x, y in zip(_normalize_distribution(aa), _normalize_distribution(bb))) / 2.0
    return {
        "similarity": max(0.0, 1.0 - jsd),
        "js_divergence": jsd,
        "l1_distance_half": l1,
        "bins": len(a),
    }


# Function: Compare two sparse feature maps as distributions.
def _compare_distribution_dicts(
    a: Dict[str, float],
    b: Dict[str, float]
) -> Dict[str, Any] | None:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return None
    va = [float(a.get(k, 0.0)) for k in keys]
    vb = [float(b.get(k, 0.0)) for k in keys]
    jsd = jensen_shannon_divergence(va, vb)
    cos = cosine_similarity(va, vb)
    return {
        "similarity": max(0.0, 1.0 - jsd),
        "js_divergence": jsd,
        "cosine_similarity": cos,
        "dimensions": len(keys),
    }


# Function: Extract an estimated corpus word count from a fingerprint.
def _fingerprint_word_estimate(fp: Dict[str, Any]) -> int:
    candidates = [
        _safe_get(fp, "metadata", "corpus", "size", "words_est"),
        _safe_get(fp, "measurements", "totals", "total_words_est"),
    ]
    for value in candidates:
        if isinstance(value, int) and value >= 0:
            return int(value)
    return 0


# Function: Compute similarity between two fingerprints using interpretable component metrics.
def compute_fingerprint_similarity(
    fingerprint_a: Dict[str, Any],
    fingerprint_b: Dict[str, Any],
    component_weights: Dict[str, float] | None = None
) -> Dict[str, Any]:
    if not isinstance(fingerprint_a, dict) or not isinstance(fingerprint_b, dict):
        raise ValueError("Both fingerprints must be JSON objects.")

    default_weights: Dict[str, float] = {
        "function_words_distribution": 0.20,
        "sentence_histogram": 0.15,
        "paragraph_histogram": 0.10,
        "punctuation_rates": 0.10,
        "stance_signals": 0.08,
        "rhetoric_moves": 0.08,
        "syntax_texture": 0.08,
        "paragraph_cadence": 0.08,
        "repetition": 0.05,
        "lexicon_preferred_overlap": 0.04,
        "lexicon_avoid_overlap": 0.04,
    }
    if isinstance(component_weights, dict):
        for key, value in component_weights.items():
            if key in default_weights and isinstance(value, (int, float)) and not isinstance(value, bool):
                default_weights[key] = max(0.0, float(value))

    components: Dict[str, Dict[str, Any]] = {}

    # Distribution components.
    sent_a = _safe_get(fingerprint_a, "measurements", "sentence", "length_words", "histogram_p", default=[])
    sent_b = _safe_get(fingerprint_b, "measurements", "sentence", "length_words", "histogram_p", default=[])
    cmp_sent = _compare_distribution_vectors(sent_a, sent_b) if isinstance(sent_a, list) and isinstance(sent_b, list) else None
    if cmp_sent:
        components["sentence_histogram"] = cmp_sent

    para_a = _safe_get(fingerprint_a, "measurements", "paragraph", "length_sentences_histogram_p", default=[])
    para_b = _safe_get(fingerprint_b, "measurements", "paragraph", "length_sentences_histogram_p", default=[])
    cmp_para = _compare_distribution_vectors(para_a, para_b) if isinstance(para_a, list) and isinstance(para_b, list) else None
    if cmp_para:
        components["paragraph_histogram"] = cmp_para

    func_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "function_words", "rates_per_1000w", default={}))
    func_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "function_words", "rates_per_1000w", default={}))
    cmp_func = _compare_distribution_dicts(func_a, func_b)
    if cmp_func:
        components["function_words_distribution"] = cmp_func

    # Numeric dictionary components.
    punct_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "punctuation", "rates_per_1000w", default={}))
    punct_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "punctuation", "rates_per_1000w", default={}))
    cmp_punct = _compare_numeric_dicts(punct_a, punct_b)
    if cmp_punct:
        components["punctuation_rates"] = cmp_punct

    stance_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "stance_signals", default={}))
    stance_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "stance_signals", default={}))
    cmp_stance = _compare_numeric_dicts(stance_a, stance_b)
    if cmp_stance:
        components["stance_signals"] = cmp_stance

    rhet_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "rhetoric_moves", default={}))
    rhet_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "rhetoric_moves", default={}))
    cmp_rhet = _compare_numeric_dicts(rhet_a, rhet_b)
    if cmp_rhet:
        components["rhetoric_moves"] = cmp_rhet

    synt_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "syntax_texture", default={}))
    synt_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "syntax_texture", default={}))
    cmp_synt = _compare_numeric_dicts(synt_a, synt_b)
    if cmp_synt:
        components["syntax_texture"] = cmp_synt

    cadence_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "paragraph_cadence", default={}))
    cadence_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "paragraph_cadence", default={}))
    cmp_cadence = _compare_numeric_dicts(cadence_a, cadence_b)
    if cmp_cadence:
        components["paragraph_cadence"] = cmp_cadence

    rep_a = _to_numeric_dict(_safe_get(fingerprint_a, "measurements", "repetition", default={}))
    rep_b = _to_numeric_dict(_safe_get(fingerprint_b, "measurements", "repetition", default={}))
    rep_a = {k: v for k, v in rep_a.items() if k in {"bigram_repeat_rate", "trigram_repeat_rate"}}
    rep_b = {k: v for k, v in rep_b.items() if k in {"bigram_repeat_rate", "trigram_repeat_rate"}}
    cmp_rep = _compare_numeric_dicts(rep_a, rep_b)
    if cmp_rep:
        components["repetition"] = cmp_rep

    # Lexical overlap components.
    lex_a = _safe_get(fingerprint_a, "lexicon", default={})
    lex_b = _safe_get(fingerprint_b, "lexicon", default={})
    preferred_a = set()
    preferred_b = set()
    avoid_a = set()
    avoid_b = set()
    if isinstance(lex_a, dict):
        preferred_a = {
            str(x).strip().lower()
            for x in (lex_a.get("preferred_words", []) or []) + (lex_a.get("preferred_phrases", []) or [])
            if isinstance(x, str) and str(x).strip()
        }
        avoid_a = {str(x).strip().lower() for x in (lex_a.get("avoid_words", []) or []) if isinstance(x, str) and str(x).strip()}
    if isinstance(lex_b, dict):
        preferred_b = {
            str(x).strip().lower()
            for x in (lex_b.get("preferred_words", []) or []) + (lex_b.get("preferred_phrases", []) or [])
            if isinstance(x, str) and str(x).strip()
        }
        avoid_b = {str(x).strip().lower() for x in (lex_b.get("avoid_words", []) or []) if isinstance(x, str) and str(x).strip()}

    components["lexicon_preferred_overlap"] = {
        "similarity": jaccard_similarity(preferred_a, preferred_b),
        "size_a": len(preferred_a),
        "size_b": len(preferred_b),
        "intersection_size": len(preferred_a & preferred_b),
    }
    components["lexicon_avoid_overlap"] = {
        "similarity": jaccard_similarity(avoid_a, avoid_b),
        "size_a": len(avoid_a),
        "size_b": len(avoid_b),
        "intersection_size": len(avoid_a & avoid_b),
    }

    total_weight = sum(default_weights.values())
    used_weight = 0.0
    weighted_sum = 0.0
    missing: List[str] = []
    used_components: Dict[str, Dict[str, Any]] = {}
    for name, weight in default_weights.items():
        comp = components.get(name)
        if not comp or not isinstance(comp.get("similarity"), (int, float)):
            missing.append(name)
            continue
        sim = clamp01(float(comp["similarity"]))
        used_weight += weight
        weighted_sum += sim * weight
        used_components[name] = {**comp, "weight": weight}

    if used_weight > 0.0:
        overall = weighted_sum / used_weight
    else:
        overall = 0.0
    overall = clamp01(overall)

    words_a = _fingerprint_word_estimate(fingerprint_a)
    words_b = _fingerprint_word_estimate(fingerprint_b)
    min_words = min(words_a, words_b) if (words_a > 0 and words_b > 0) else 0
    evidence_factor = 1.0 - math.exp(-max(0, min_words) / 20000.0)
    coverage = used_weight / total_weight if total_weight > 0 else 0.0
    confidence_hint = clamp01((0.6 * coverage) + (0.4 * evidence_factor))

    ordered_components = sorted(
        (
            {"name": name, **spec}
            for name, spec in used_components.items()
        ),
        key=lambda row: float(row.get("similarity", 0.0))
    )
    top_differences = ordered_components[:5]

    return {
        "similarity_score": overall,
        "distance_score": 1.0 - overall,
        "coverage": {
            "components_available": len(used_components),
            "components_expected": len(default_weights),
            "missing_components": missing,
            "used_weight": used_weight,
            "total_weight": total_weight,
            "coverage_ratio": coverage,
        },
        "confidence_hint": confidence_hint,
        "evidence": {
            "words_est_a": words_a,
            "words_est_b": words_b,
            "min_words_est": min_words,
            "evidence_factor": evidence_factor,
        },
        "components": ordered_components,
        "top_differences": top_differences,
        "weights": default_weights,
    }
