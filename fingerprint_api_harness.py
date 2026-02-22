#!/usr/bin/env python3
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
"""
fingerprint_api_harness.py

Simple GUI demo harness for the local fingerprint API.
Allows users to call all primary endpoints without writing curl commands.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk
from typing import Any, Dict

import requests


# Function: Parse optional integer input.
def parse_optional_int(raw: str, field_name: str) -> int | None:
    token = raw.strip()
    if token == "":
        return None
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    return value


# Function: Parse optional float input.
def parse_optional_float(raw: str, field_name: str) -> float | None:
    token = raw.strip()
    if token == "":
        return None
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    return value


class FingerprintAPIHarness(tk.Tk):
    def __init__(self, host: str, api_port: int):
        super().__init__()
        self.title("Stylometric Transfer API Harness")
        self.geometry("1180x860")

        self.host_var = tk.StringVar(value=host)
        self.port_var = tk.StringVar(value=str(api_port))
        self.base_url_var = tk.StringVar()
        self._refresh_base_url()

        self._build_top_bar()
        self._build_tabs()
        self._build_output()

    # Function: Compute the base URL from host+port controls.
    def _refresh_base_url(self) -> None:
        self.base_url_var.set(f"http://{self.host_var.get().strip()}:{self.port_var.get().strip()}")

    # Function: Render common top controls.
    def _build_top_bar(self) -> None:
        frame = ttk.Frame(self, padding=8)
        frame.pack(fill="x")

        ttk.Label(frame, text="Host:").pack(side="left")
        ttk.Entry(frame, textvariable=self.host_var, width=20).pack(side="left", padx=(4, 12))
        ttk.Label(frame, text="API Port:").pack(side="left")
        ttk.Entry(frame, textvariable=self.port_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Button(frame, text="Apply Host/Port", command=self._on_apply_host_port).pack(side="left")
        ttk.Button(frame, text="Health Check", command=self._call_health).pack(side="left", padx=(8, 0))
        ttk.Label(frame, textvariable=self.base_url_var).pack(side="left", padx=(16, 0))

    # Function: Render endpoint tabs.
    def _build_tabs(self) -> None:
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=4)
        self._build_make_tab()
        self._build_apply_tab()
        self._build_rate_tab()
        self._build_similarity_tab()
        self._build_docs_tab()

    # Function: Render output console.
    def _build_output(self) -> None:
        frame = ttk.LabelFrame(self, text="Response", padding=8)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.output = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=16)
        self.output.pack(fill="both", expand=True)

    # Function: Build the /make tab.
    def _build_make_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="/make")

        ttk.Label(tab, text="Source Text").grid(row=0, column=0, sticky="w")
        self.make_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=110, height=11)
        self.make_text.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=(2, 8))

        ttk.Label(tab, text="profile_id").grid(row=2, column=0, sticky="w")
        self.make_profile = ttk.Entry(tab, width=24)
        self.make_profile.grid(row=2, column=1, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="author_name").grid(row=2, column=2, sticky="w")
        self.make_author = ttk.Entry(tab, width=24)
        self.make_author.grid(row=2, column=3, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="max_prompt_tokens").grid(row=2, column=4, sticky="w")
        self.make_max_prompt = ttk.Entry(tab, width=10)
        self.make_max_prompt.grid(row=2, column=5, sticky="w", padx=(4, 0))

        self.make_no_phrase = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab,
            text="no_phrase_validation",
            variable=self.make_no_phrase
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

        ttk.Label(tab, text="quote_mode").grid(row=3, column=2, sticky="w", pady=(6, 0))
        self.make_quote_mode = ttk.Combobox(tab, width=16, values=["", "fiction", "non-fiction"], state="readonly")
        self.make_quote_mode.current(0)
        self.make_quote_mode.grid(row=3, column=3, sticky="w", pady=(6, 0))

        ttk.Button(tab, text="POST /make", command=self._call_make).grid(row=4, column=0, sticky="w", pady=(10, 0))

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

    # Function: Build the /apply tab.
    def _build_apply_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="/apply")

        ttk.Label(tab, text="Fingerprint GUID (id)").grid(row=0, column=0, sticky="w")
        self.apply_id = ttk.Entry(tab, width=45)
        self.apply_id.grid(row=0, column=1, columnspan=2, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="Input Text").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.apply_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=110, height=11)
        self.apply_text.grid(row=2, column=0, columnspan=7, sticky="nsew", pady=(2, 8))

        ttk.Label(tab, text="local_spelling").grid(row=3, column=0, sticky="w")
        self.apply_local_spelling = ttk.Combobox(
            tab,
            width=14,
            values=["", "none", "canadian", "australian", "british", "us"],
            state="readonly",
        )
        self.apply_local_spelling.current(0)
        self.apply_local_spelling.grid(row=3, column=1, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="force_person").grid(row=3, column=2, sticky="w")
        self.apply_force_person = ttk.Combobox(
            tab,
            width=12,
            values=["", "first", "second", "third"],
            state="readonly",
        )
        self.apply_force_person.current(0)
        self.apply_force_person.grid(row=3, column=3, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="max_prompt_tokens").grid(row=3, column=4, sticky="w")
        self.apply_max_prompt = ttk.Entry(tab, width=10)
        self.apply_max_prompt.grid(row=3, column=5, sticky="w", padx=(4, 10))

        self.apply_no_style_retry = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="no_style_retry", variable=self.apply_no_style_retry).grid(row=4, column=0, sticky="w")
        self.apply_metrics = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="metrics", variable=self.apply_metrics).grid(row=4, column=1, sticky="w")

        ttk.Button(tab, text="POST /apply", command=self._call_apply).grid(row=5, column=0, sticky="w", pady=(10, 0))

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

    # Function: Build the /rate tab.
    def _build_rate_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="/rate")

        ttk.Label(tab, text="Fingerprint GUID (id)").grid(row=0, column=0, sticky="w")
        self.rate_id = ttk.Entry(tab, width=45)
        self.rate_id.grid(row=0, column=1, columnspan=2, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="Text Segment").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.rate_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=110, height=11)
        self.rate_text.grid(row=2, column=0, columnspan=8, sticky="nsew", pady=(2, 8))

        ttk.Label(tab, text="threshold").grid(row=3, column=0, sticky="w")
        self.rate_threshold = ttk.Entry(tab, width=10)
        self.rate_threshold.grid(row=3, column=1, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="slope").grid(row=3, column=2, sticky="w")
        self.rate_slope = ttk.Entry(tab, width=10)
        self.rate_slope.grid(row=3, column=3, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="evidence_tokens").grid(row=3, column=4, sticky="w")
        self.rate_evidence = ttk.Entry(tab, width=10)
        self.rate_evidence.grid(row=3, column=5, sticky="w", padx=(4, 10))

        ttk.Button(tab, text="POST /rate", command=self._call_rate).grid(row=4, column=0, sticky="w", pady=(10, 0))

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

    # Function: Build the /similarity tab.
    def _build_similarity_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="/similarity")

        ttk.Label(tab, text="GUID A (id_a)").grid(row=0, column=0, sticky="w")
        self.sim_id_a = ttk.Entry(tab, width=45)
        self.sim_id_a.grid(row=0, column=1, sticky="w", padx=(4, 10))

        ttk.Label(tab, text="GUID B (id_b)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.sim_id_b = ttk.Entry(tab, width=45)
        self.sim_id_b.grid(row=1, column=1, sticky="w", padx=(4, 10), pady=(8, 0))

        ttk.Label(tab, text="component_weights JSON (optional)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.sim_weights = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=80, height=8)
        self.sim_weights.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(2, 8))
        self.sim_weights.insert(
            "1.0",
            '{\n  "function_words_distribution": 0.2,\n  "sentence_histogram": 0.15\n}\n',
        )

        ttk.Button(tab, text="POST /similarity", command=self._call_similarity).grid(row=4, column=0, sticky="w", pady=(10, 0))

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)

    # Function: Build docs/utility tab.
    def _build_docs_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="Utilities")

        ttk.Label(tab, text="Click to fetch API docs payload from the running server.").pack(anchor="w")
        ttk.Button(tab, text="GET /openapi.json", command=self._call_openapi_json).pack(anchor="w", pady=(8, 4))
        ttk.Button(tab, text="GET /openapi.yaml", command=self._call_openapi_yaml).pack(anchor="w")

    # Function: Recompute and validate host/port.
    def _on_apply_host_port(self) -> bool:
        host = self.host_var.get().strip()
        port_text = self.port_var.get().strip()
        if host == "":
            messagebox.showerror("Invalid Host", "Host cannot be empty.")
            return False
        try:
            port_value = int(port_text)
        except ValueError:
            messagebox.showerror("Invalid Port", "API Port must be an integer.")
            return False
        if not (1 <= port_value <= 65535):
            messagebox.showerror("Invalid Port", "API Port must be in 1..65535.")
            return False
        self.port_var.set(str(port_value))
        self._refresh_base_url()
        return True

    # Function: Send an HTTP request and render response.
    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> None:
        if not self._on_apply_host_port():
            return
        url = f"{self.base_url_var.get()}{path}"
        try:
            if method == "GET":
                resp = requests.get(url, timeout=120)
            else:
                resp = requests.post(url, json=payload, timeout=180)
        except Exception as exc:
            self._render_output(f"{method} {url}\n\nRequest failed:\n{exc}")
            return

        body_text = resp.text
        parsed: Any = None
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        if parsed is not None:
            rendered_body = json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            rendered_body = body_text

        self._render_output(
            f"{method} {url}\n"
            f"Status: {resp.status_code}\n\n"
            f"{rendered_body}"
        )

    # Function: Display text in output box.
    def _render_output(self, text: str) -> None:
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.see("1.0")

    # Function: Call /health.
    def _call_health(self) -> None:
        self._request("GET", "/health")

    # Function: Call /openapi.json.
    def _call_openapi_json(self) -> None:
        self._request("GET", "/openapi.json")

    # Function: Call /openapi.yaml.
    def _call_openapi_yaml(self) -> None:
        self._request("GET", "/openapi.yaml")

    # Function: Call /make.
    def _call_make(self) -> None:
        text = self.make_text.get("1.0", tk.END).strip()
        if text == "":
            messagebox.showerror("Missing text", "Source text is required for /make.")
            return
        payload: Dict[str, Any] = {"text": text}
        profile_id = self.make_profile.get().strip()
        author_name = self.make_author.get().strip()
        if profile_id:
            payload["profile_id"] = profile_id
        if author_name:
            payload["author_name"] = author_name
        try:
            max_prompt_tokens = parse_optional_int(self.make_max_prompt.get(), "max_prompt_tokens")
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        if max_prompt_tokens is not None:
            payload["max_prompt_tokens"] = max_prompt_tokens
        if self.make_no_phrase.get():
            payload["no_phrase_validation"] = True
        quote_mode = self.make_quote_mode.get().strip()
        if quote_mode:
            payload["quote_mode"] = quote_mode
        self._request("POST", "/make", payload)

    # Function: Call /apply.
    def _call_apply(self) -> None:
        guid = self.apply_id.get().strip()
        text = self.apply_text.get("1.0", tk.END).strip()
        if guid == "" or text == "":
            messagebox.showerror("Missing fields", "Both GUID and input text are required for /apply.")
            return
        payload: Dict[str, Any] = {"id": guid, "text": text}
        try:
            max_prompt_tokens = parse_optional_int(self.apply_max_prompt.get(), "max_prompt_tokens")
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        if max_prompt_tokens is not None:
            payload["max_prompt_tokens"] = max_prompt_tokens
        local_spelling = self.apply_local_spelling.get().strip()
        if local_spelling:
            payload["local_spelling"] = local_spelling
        force_person = self.apply_force_person.get().strip()
        if force_person:
            payload["force_person"] = force_person
        if self.apply_no_style_retry.get():
            payload["no_style_retry"] = True
        if self.apply_metrics.get():
            payload["metrics"] = True
        self._request("POST", "/apply", payload)

    # Function: Call /rate.
    def _call_rate(self) -> None:
        guid = self.rate_id.get().strip()
        text = self.rate_text.get("1.0", tk.END).strip()
        if guid == "" or text == "":
            messagebox.showerror("Missing fields", "Both GUID and text segment are required for /rate.")
            return
        payload: Dict[str, Any] = {"id": guid, "text": text}
        try:
            threshold = parse_optional_float(self.rate_threshold.get(), "threshold")
            slope = parse_optional_float(self.rate_slope.get(), "slope")
            evidence = parse_optional_int(self.rate_evidence.get(), "evidence_tokens")
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        if threshold is not None:
            payload["threshold"] = threshold
        if slope is not None:
            payload["slope"] = slope
        if evidence is not None:
            payload["evidence_tokens"] = evidence
        self._request("POST", "/rate", payload)

    # Function: Call /similarity.
    def _call_similarity(self) -> None:
        id_a = self.sim_id_a.get().strip()
        id_b = self.sim_id_b.get().strip()
        if id_a == "" or id_b == "":
            messagebox.showerror("Missing fields", "Both id_a and id_b are required for /similarity.")
            return
        payload: Dict[str, Any] = {"id_a": id_a, "id_b": id_b}
        weights_text = self.sim_weights.get("1.0", tk.END).strip()
        if weights_text:
            try:
                weights_obj = json.loads(weights_text)
            except Exception as exc:
                messagebox.showerror("Invalid JSON", f"component_weights must be valid JSON.\n{exc}")
                return
            if not isinstance(weights_obj, dict):
                messagebox.showerror("Invalid JSON", "component_weights must decode to an object.")
                return
            payload["component_weights"] = weights_obj
        self._request("POST", "/similarity", payload)


# Function: Entry point.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    ap.add_argument("--api", type=int, default=8765, help="API port (default: 8765)")
    args = ap.parse_args()

    if not (1 <= int(args.api) <= 65535):
        raise SystemExit("Error: --api must be in 1..65535.")

    app = FingerprintAPIHarness(args.host, int(args.api))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
