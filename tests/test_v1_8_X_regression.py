import sys
from pathlib import Path
import unittest
import io
import contextlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import apply_fingerprint as af  # noqa: E402


class TestV18XRegression(unittest.TestCase):
    def test_humanizer_mandatory_normalizes_single_quotes_without_touching_backticks(self) -> None:
        text = "It‘s called ‘alpha’, not ‚beta‛ or ‹gamma›. Keep `ticks` and ```code```."
        updated, count = af.enforce_straight_single_quotes(text)
        self.assertEqual(count, 7)
        for ch in ("‘", "’", "‚", "‛", "‹", "›"):
            self.assertNotIn(ch, updated)
        self.assertIn("It's called 'alpha'", updated)
        self.assertIn("`ticks`", updated)
        self.assertIn("```code```", updated)

    def test_perplexity_profile_default_uses_baseline_knobs(self) -> None:
        tunables = {
            "perplexity_level": "default",
            "perplexity_profiles": {
                "default": {
                    "humanizer_variance": {"max_ops_per_1000w": 0.5},
                    "humanization_controller": {"quantiles": [0.25, 0.5, 0.75], "range_pct": 0.15},
                    "chunking": {"max_input_tokens": 5750, "min_chunks_when_perturbing": 2},
                },
                "high": {
                    "humanizer_variance": {"max_ops_per_1000w": 2.0},
                    "humanization_controller": {"quantiles": [0.1, 0.5, 0.9], "range_pct": 0.3},
                    "chunking": {"max_input_tokens": 4200, "min_chunks_when_perturbing": 5},
                },
            },
            "humanizer_variance": {"max_ops_per_1000w": 0.5},
            "humanization_controller": {"quantiles": [0.25, 0.5, 0.75], "range_pct": 0.15},
            "chunking": {"max_input_tokens": 5750, "min_chunks_when_perturbing": 2},
        }
        applied, level, knobs = af.apply_perplexity_profile(tunables)
        self.assertEqual(level, "default")
        self.assertEqual(applied["humanizer_variance"]["max_ops_per_1000w"], 0.5)
        self.assertEqual(applied["humanization_controller"]["quantiles"], [0.25, 0.5, 0.75])
        self.assertEqual(applied["chunking"]["max_input_tokens"], 5750)
        self.assertEqual(knobs["chunking.min_chunks_when_perturbing"], 2)

    def test_perplexity_profile_cli_override(self) -> None:
        tunables = {
            "perplexity_level": "default",
            "perplexity_profiles": {
                "default": {
                    "humanizer_variance": {"max_ops_per_1000w": 0.5},
                    "humanization_controller": {"quantiles": [0.25, 0.5, 0.75], "range_pct": 0.15},
                    "chunking": {"max_input_tokens": 5750, "min_chunks_when_perturbing": 2},
                },
                "high": {
                    "humanizer_variance": {"max_ops_per_1000w": 2.0},
                    "humanization_controller": {"quantiles": [0.1, 0.5, 0.9], "range_pct": 0.3},
                    "chunking": {"max_input_tokens": 4200, "min_chunks_when_perturbing": 5},
                },
            },
            "humanizer_variance": {"max_ops_per_1000w": 0.5},
            "humanization_controller": {"quantiles": [0.25, 0.5, 0.75], "range_pct": 0.15},
            "chunking": {"max_input_tokens": 5750, "min_chunks_when_perturbing": 2},
        }
        applied, level, knobs = af.apply_perplexity_profile(tunables, "high")
        self.assertEqual(level, "high")
        self.assertEqual(applied["humanizer_variance"]["max_ops_per_1000w"], 2.0)
        self.assertEqual(applied["humanization_controller"]["quantiles"], [0.1, 0.5, 0.9])
        self.assertEqual(applied["humanization_controller"]["range_pct"], 0.3)
        self.assertEqual(applied["chunking"]["max_input_tokens"], 4200)
        self.assertEqual(knobs["chunking.min_chunks_when_perturbing"], 5)

    def test_perplexity_profile_extreme_temperature_multiplier(self) -> None:
        tunables = {
            "perplexity_level": "default",
            "perplexity_profiles": {
                "default": {
                    "llm": {"temperature_multiplier": 1.0}
                },
                "extreme": {
                    "llm": {"temperature_multiplier": 2.0}
                },
            },
        }
        _, level, knobs = af.apply_perplexity_profile(tunables, "extreme")
        self.assertEqual(level, "extreme")
        self.assertEqual(knobs["llm.temperature_multiplier"], 2.0)

    def test_apply_temperature_multiplier_scales_and_clamps(self) -> None:
        self.assertAlmostEqual(af.apply_temperature_multiplier(0.2, 2.0), 0.4)
        self.assertAlmostEqual(af.apply_temperature_multiplier(1.5, 2.0), 2.0)

    def test_extract_query_arg_supports_equals_and_space(self) -> None:
        self.assertEqual(af.extract_query_arg(["--query", "perplexity"]), "perplexity")
        self.assertEqual(af.extract_query_arg(["--query=perplexity"]), "perplexity")
        self.assertEqual(af.extract_query_arg(["--query"]), "")
        self.assertIsNone(af.extract_query_arg(["--perplexity", "high"]))

    def test_handle_query_perplexity_prints_single_token(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = af.handle_query("perplexity")
        self.assertEqual(code, 0)
        value = out.getvalue().strip()
        self.assertIn(value, ("default", "low", "medium", "high", "extreme"))


if __name__ == "__main__":
    unittest.main()
