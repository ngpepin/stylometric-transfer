#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
"""
fingerprint_api.py

Local HTTP API for stylometric fingerprint workflows.

Endpoints:
  POST /make   -> Build a fingerprint from text and store it by GUID.
  POST /apply  -> Apply a stored fingerprint to new text.
  POST /rate   -> Score probability that text matches a stored fingerprint.
  POST /similarity -> Compare two stored fingerprints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import apply_fingerprint as af

from common import (
    calibrated_style_match_probability,
    compute_fingerprint_similarity,
    find_repo_root,
    load_fingerprint_from_store,
    resolve_fingerprint_store_dir,
    resolve_optional_path,
    resolve_required_path,
    save_fingerprint_to_store,
)


@dataclass
class APIState:
    repo_root: Path
    config_path: Path
    tunables_path: Path | None
    store_dir: Path
    verbose: bool = False
    style_retry_threshold: float = 0.75
    default_max_prompt_tokens: int | None = None
    store_lock: threading.Lock = field(default_factory=threading.Lock)


# Function: Build a command and execute it in the repository root.
def run_repo_command(repo_root: Path, cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


# Function: Parse and validate a JSON request body.
def parse_json_body(raw: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


# Function: Build arguments for optional quote-mode flags.
def quote_mode_args(payload: Dict[str, Any]) -> List[str]:
    mode = payload.get("quote_mode")
    fiction = payload.get("fiction")
    if isinstance(mode, str):
        lowered = mode.strip().lower()
        if lowered in {"fiction", "fictional"}:
            return ["--fiction"]
        if lowered in {"non-fiction", "nonfiction", "non_fiction"}:
            return ["--non-fiction"]
    if isinstance(fiction, bool):
        return ["--fiction"] if fiction else ["--non-fiction"]
    return []


class FingerprintAPIHandler(BaseHTTPRequestHandler):
    server: "FingerprintAPIServer"

    # Function: Write a JSON response.
    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Function: Parse request JSON.
    def read_json(self) -> Dict[str, Any]:
        length_raw = self.headers.get("Content-Length", "0").strip()
        if not length_raw.isdigit():
            raise ValueError("Invalid Content-Length header.")
        length = int(length_raw)
        if length <= 0:
            raise ValueError("Request body is required.")
        data = self.rfile.read(length)
        return parse_json_body(data)

    # Function: Send a standardized error payload.
    def fail(self, status: int, message: str, detail: Any = None) -> None:
        payload: Dict[str, Any] = {"error": message}
        if detail is not None:
            payload["detail"] = detail
        self.send_json(status, payload)

    # Function: Handle GET requests for health/docs helpers.
    def do_GET(self) -> None:  # noqa: N802
        state = self.server.state
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "fingerprint_api",
                    "store_dir": str(state.store_dir),
                },
            )
            return
        if self.path in {"/openapi.yaml", "/openapi.json"}:
            docs_path = state.repo_root / "api" / "swagger" / self.path.lstrip("/")
            if not docs_path.exists():
                self.fail(404, f"Not found: {self.path}")
                return
            content_type = "application/yaml" if docs_path.suffix == ".yaml" else "application/json"
            body = docs_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.fail(404, f"Unknown path: {self.path}")

    # Function: Route POST requests.
    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/make":
                self.handle_make()
                return
            if self.path == "/apply":
                self.handle_apply()
                return
            if self.path == "/rate":
                self.handle_rate()
                return
            if self.path == "/similarity":
                self.handle_similarity()
                return
            self.fail(404, f"Unknown path: {self.path}")
        except ValueError as exc:
            self.fail(400, str(exc))
        except FileNotFoundError as exc:
            self.fail(404, str(exc))
        except RuntimeError as exc:
            self.fail(502, str(exc))
        except Exception as exc:
            self.fail(500, "Internal server error", detail=str(exc))

    # Function: Build a fingerprint from text and store it in the GUID store.
    def handle_make(self) -> None:
        state = self.server.state
        payload = self.read_json()
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string.")

        no_phrase_validation = bool(payload.get("no_phrase_validation", False))
        profile_id = payload.get("profile_id")
        author_name = payload.get("author_name")
        max_prompt_tokens = payload.get("max_prompt_tokens")
        if max_prompt_tokens is None:
            max_prompt_tokens = state.default_max_prompt_tokens

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source_md = tmp / "input.md"
            source_md.write_text(text, encoding="utf-8")

            archive_path = tmp / "input.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(source_md, arcname=source_md.name)

            out_path = tmp / "fingerprint.json"
            cmd: List[str] = [
                sys.executable,
                str(state.repo_root / "fingerprint_style.py"),
                "-c",
                str(state.config_path),
                "-a",
                str(archive_path),
                "-o",
                str(out_path),
            ]
            cmd.extend(quote_mode_args(payload))
            if no_phrase_validation:
                cmd.append("--no-phrase-validation")
            if isinstance(profile_id, str) and profile_id.strip():
                cmd.extend(["--profile-id", profile_id.strip()])
            if isinstance(author_name, str) and author_name.strip():
                cmd.extend(["--author-name", author_name.strip()])
            if isinstance(max_prompt_tokens, int) and max_prompt_tokens > 0:
                cmd.extend(["--max-prompt-tokens", str(max_prompt_tokens)])

            result = run_repo_command(state.repo_root, cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    "Fingerprint generation failed: "
                    f"exit={result.returncode}; stderr={result.stderr.strip() or '(empty)'}"
                )
            fingerprint = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(fingerprint, dict):
                raise RuntimeError("Generated fingerprint is not a JSON object.")

        with state.store_lock:
            guid, fp_path, meta_path = save_fingerprint_to_store(
                fingerprint,
                state.store_dir,
                source="api.make",
                extra_metadata={
                    "make_stdout": result.stdout.strip(),
                    "make_stderr": result.stderr.strip(),
                },
            )

        self.send_json(
            200,
            {
                "id": guid,
                "profile_id": fingerprint.get("profile_id"),
                "stored_fingerprint": str(fp_path),
                "stored_metadata": str(meta_path),
            },
        )

    # Function: Apply a stored fingerprint to input text.
    def handle_apply(self) -> None:
        state = self.server.state
        payload = self.read_json()
        guid = payload.get("id")
        text = payload.get("text")
        if not isinstance(guid, str) or not guid.strip():
            raise ValueError("Field 'id' must be a non-empty GUID string.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string.")

        fingerprint, fp_path, _meta = load_fingerprint_from_store(guid.strip(), state.store_dir)
        max_prompt_tokens = payload.get("max_prompt_tokens")
        if max_prompt_tokens is None:
            max_prompt_tokens = state.default_max_prompt_tokens
        force_person = payload.get("force_person")
        local_spelling = payload.get("local_spelling")
        no_style_retry = bool(payload.get("no_style_retry", False))
        metrics = bool(payload.get("metrics", False))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_path = tmp / "input.md"
            output_path = tmp / "output.styled.md"
            input_path.write_text(text, encoding="utf-8")

            cmd: List[str] = [
                sys.executable,
                str(state.repo_root / "apply_fingerprint.py"),
                "-c",
                str(state.config_path),
                "-f",
                str(fp_path),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
            ]
            if state.tunables_path is not None and state.tunables_path.exists():
                cmd.extend(["--tunables", str(state.tunables_path)])
            cmd.extend(quote_mode_args(payload))
            if isinstance(max_prompt_tokens, int) and max_prompt_tokens > 0:
                cmd.extend(["--max-prompt-tokens", str(max_prompt_tokens)])
            if no_style_retry:
                cmd.append("--no-style-retry")
            if metrics:
                cmd.append("--metrics")
            if isinstance(local_spelling, str) and local_spelling.strip():
                cmd.extend(["--local-spelling", local_spelling.strip().lower()])
            if isinstance(force_person, str):
                lowered = force_person.strip().lower()
                if lowered in {"first", "1st", "one"}:
                    cmd.append("--1st-person")
                elif lowered in {"second", "2nd", "two"}:
                    cmd.append("--2nd-person")
                elif lowered in {"third", "3rd", "three"}:
                    cmd.append("--3rd-person")

            result = run_repo_command(state.repo_root, cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    "Fingerprint apply failed: "
                    f"exit={result.returncode}; stderr={result.stderr.strip() or '(empty)'}"
                )

            final_markdown = output_path.read_text(encoding="utf-8")
            deviations_path = output_path.with_suffix(output_path.suffix + ".deviations.json")
            deviations: List[Dict[str, Any]] = []
            if deviations_path.exists():
                try:
                    loaded = json.loads(deviations_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        deviations = [d for d in loaded if isinstance(d, dict)]
                except Exception:
                    deviations = []

        self.send_json(
            200,
            {
                "id": guid,
                "profile_id": fingerprint.get("profile_id"),
                "final_markdown": final_markdown,
                "deviations": deviations,
                "apply_stdout": result.stdout.strip(),
                "apply_stderr": result.stderr.strip(),
            },
        )

    # Function: Return probability that text matches a stored fingerprint.
    def handle_rate(self) -> None:
        state = self.server.state
        payload = self.read_json()
        guid = payload.get("id")
        text = payload.get("text")
        if not isinstance(guid, str) or not guid.strip():
            raise ValueError("Field 'id' must be a non-empty GUID string.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string.")

        fingerprint, _fp_path, _meta = load_fingerprint_from_store(guid.strip(), state.store_dir)
        compliance = af.compute_style_compliance(fingerprint, text)
        compliance_score = float(compliance.get("score", 0.0))
        output_measurements = compliance.get("output_measurements", {})
        token_count = 0
        if isinstance(output_measurements, dict):
            totals = output_measurements.get("totals", {})
            if isinstance(totals, dict):
                total_words_est = totals.get("total_words_est")
                if isinstance(total_words_est, int):
                    token_count = total_words_est
        if token_count <= 0:
            token_count = len(af.words(af.filter_author_voice_text(text)))

        threshold_raw = payload.get("threshold")
        slope_raw = payload.get("slope")
        evidence_tokens_raw = payload.get("evidence_tokens")
        threshold = state.style_retry_threshold
        if isinstance(threshold_raw, (int, float)):
            threshold = float(threshold_raw)
        slope = 0.12
        if isinstance(slope_raw, (int, float)):
            slope = float(slope_raw)
        evidence_tokens = 250
        if isinstance(evidence_tokens_raw, int) and evidence_tokens_raw > 0:
            evidence_tokens = evidence_tokens_raw

        probability = calibrated_style_match_probability(
            compliance_score=compliance_score,
            token_count=token_count,
            threshold=threshold,
            slope=slope,
            evidence_tokens=evidence_tokens,
        )
        deltas = compliance.get("deltas", [])
        top_deltas: List[Dict[str, Any]] = []
        if isinstance(deltas, list):
            for d in deltas:
                if isinstance(d, dict):
                    top_deltas.append(d)
                if len(top_deltas) >= 10:
                    break

        self.send_json(
            200,
            {
                "id": guid,
                "profile_id": fingerprint.get("profile_id"),
                "probability": probability["probability"],
                "probability_percent": probability["probability_percent"],
                "confidence_interval_90": probability["confidence_interval_90"],
                "compliance_score": compliance_score,
                "token_count": token_count,
                "top_deltas": top_deltas,
                "calibration": probability["calibration"],
            },
        )

    # Function: Compare two stored fingerprints and return component-level similarity diagnostics.
    def handle_similarity(self) -> None:
        state = self.server.state
        payload = self.read_json()
        guid_a = payload.get("id_a")
        guid_b = payload.get("id_b")
        if not isinstance(guid_a, str) or not guid_a.strip():
            raise ValueError("Field 'id_a' must be a non-empty GUID string.")
        if not isinstance(guid_b, str) or not guid_b.strip():
            raise ValueError("Field 'id_b' must be a non-empty GUID string.")

        fp_a, _path_a, _meta_a = load_fingerprint_from_store(guid_a.strip(), state.store_dir)
        fp_b, _path_b, _meta_b = load_fingerprint_from_store(guid_b.strip(), state.store_dir)

        component_weights = payload.get("component_weights")
        if component_weights is not None and not isinstance(component_weights, dict):
            raise ValueError("Field 'component_weights' must be an object when provided.")

        similarity = compute_fingerprint_similarity(fp_a, fp_b, component_weights=component_weights)
        self.send_json(
            200,
            {
                "id_a": guid_a.strip(),
                "id_b": guid_b.strip(),
                "profile_id_a": fp_a.get("profile_id"),
                "profile_id_b": fp_b.get("profile_id"),
                **similarity,
            },
        )

    # Function: Quiet default logging unless verbose is enabled.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if self.server.state.verbose:
            super().log_message(format, *args)


class FingerprintAPIServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, state: APIState):
        super().__init__((host, port), FingerprintAPIHandler)
        self.state = state


# Function: Entry point.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.llm.json (default: ./config.llm.json if present; else next to script)",
    )
    ap.add_argument(
        "--tunables",
        type=Path,
        default=None,
        help="Path to config.tunables.json (default: ./config.tunables.json if present; else next to script)",
    )
    ap.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help="Directory for GUID-tracked fingerprint files (default: <repo>/fingerprint_store)",
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host (default: 127.0.0.1)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP bind port (default: 8765)",
    )
    ap.add_argument(
        "--api",
        type=int,
        default=None,
        help="Optional alias for --port (API port number)",
    )
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Default max_prompt_tokens forwarded to make/apply unless overridden per request",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable HTTP request logging")
    args = ap.parse_args()

    port = int(args.api) if args.api is not None else int(args.port)
    if not (1 <= port <= 65535):
        print("Error: API port must be in 1..65535.", file=sys.stderr)
        return 2

    repo_root = find_repo_root(Path(__file__).resolve())
    config_path = resolve_required_path(args.config, "config.llm.json", __file__)
    tunables_path = resolve_optional_path(args.tunables, "config.tunables.json", __file__)
    store_dir = resolve_fingerprint_store_dir(__file__, args.store_dir)
    tunables = af.load_tunables(tunables_path if tunables_path and tunables_path.exists() else None)
    style_retry = tunables.get("style_retry", {}) if isinstance(tunables, dict) else {}
    style_retry_threshold = 0.75
    if isinstance(style_retry, dict) and isinstance(style_retry.get("threshold"), (int, float)):
        style_retry_threshold = float(style_retry["threshold"])

    state = APIState(
        repo_root=repo_root,
        config_path=config_path,
        tunables_path=tunables_path,
        store_dir=store_dir,
        verbose=args.verbose,
        style_retry_threshold=style_retry_threshold,
        default_max_prompt_tokens=args.max_prompt_tokens,
    )
    server = FingerprintAPIServer(args.host, port, state)
    print(f"fingerprint_api listening on http://{args.host}:{port}")
    print(f"Using config: {config_path}")
    if tunables_path is not None:
        print(f"Using tunables: {tunables_path}")
    print(f"Fingerprint store: {store_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down fingerprint_api.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
