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
from typing import Any, Dict, Tuple

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
