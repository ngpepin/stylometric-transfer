import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import apply_fingerprint as af  # noqa: E402
from urllib.parse import urlparse
import socket


class TestLLMSmoke(unittest.TestCase):
    def test_llm_roundtrip(self) -> None:
        config_path = os.environ.get("LLM_SMOKE_CONFIG")
        if not config_path:
            raise RuntimeError("LLM_SMOKE_CONFIG not set for LLM smoke test.")
        cfg = af.load_config(Path(config_path))
        parsed = urlparse(cfg.base_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if host:
            try:
                with socket.create_connection((host, port), timeout=2.0):
                    pass
            except OSError as exc:
                raise unittest.SkipTest(f"LLM endpoint not reachable ({host}:{port}): {exc}")
        # Keep this minimal to reduce cost and latency.
        cfg = af.LLMConfig(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            max_tokens=64,
            temperature=0.0,
            timeout_seconds=cfg.timeout_seconds,
            extra_headers=cfg.extra_headers,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_retries=min(cfg.max_retries, 2),
            backoff_base_seconds=cfg.backoff_base_seconds,
            backoff_max_seconds=cfg.backoff_max_seconds,
        )
        messages = [
            {"role": "system", "content": "Return a brief response."},
            {"role": "user", "content": "Reply with OK."},
        ]
        raw, usage = af.chat_completions(cfg, messages)
        self.assertIsInstance(raw, str)
        self.assertTrue(raw.strip())
        if usage is not None:
            self.assertIsInstance(usage, dict)


if __name__ == "__main__":
    unittest.main()
