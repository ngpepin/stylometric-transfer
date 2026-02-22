import sys
from pathlib import Path
import unittest
import io
import contextlib
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import apply_fingerprint as af  # noqa: E402
import common  # noqa: E402


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

    def test_parse_roster_seed(self) -> None:
        self.assertIsNone(af.parse_roster_seed(None))
        self.assertIsNone(af.parse_roster_seed(""))
        self.assertEqual(af.parse_roster_seed("1234"), 1234)
        with self.assertRaises(ValueError):
            af.parse_roster_seed("abc")

    def test_build_roster_indices_ordered_and_seeded(self) -> None:
        self.assertEqual(af.build_roster_indices(3, 7, None), [0, 1, 2, 0, 1, 2, 0])
        seeded = af.build_roster_indices(3, 6, 42)
        self.assertEqual(seeded[:3], [1, 0, 2])
        self.assertCountEqual(seeded[:3], [0, 1, 2])
        self.assertCountEqual(seeded[3:6], [0, 1, 2])

    def test_load_llm_roster(self) -> None:
        base_cfg = af.LLMConfig(
            api_key="k",
            base_url="http://localhost:4141/v1",
            model="gpt-4.1",
            max_tokens=1000,
            temperature=0.2,
            timeout_seconds=30,
            extra_headers={},
            max_prompt_tokens=1000,
            max_retries=2,
            backoff_base_seconds=1.0,
            backoff_max_seconds=2.0,
        )
        payload = {
            "roster": [
                {"model": "gpt-4.1"},
                {"model": "gpt-4.1-mini", "temperature": 0.3},
                "gpt-4.1",
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            roster_path = Path(tmpdir) / "config.llm.roster.json"
            roster_path.write_text(json.dumps(payload), encoding="utf-8")
            entries = af.load_llm_roster(roster_path, base_cfg)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].model, "gpt-4.1")
        self.assertEqual(entries[1].model, "gpt-4.1-mini")
        self.assertAlmostEqual(entries[1].temperature, 0.3)
        self.assertEqual(entries[2].base_url, "http://localhost:4141/v1")

    def test_common_calibrated_probability_shrinks_for_short_text(self) -> None:
        long_text = common.calibrated_style_match_probability(
            compliance_score=0.9,
            token_count=1000,
        )
        short_text = common.calibrated_style_match_probability(
            compliance_score=0.9,
            token_count=5,
        )
        self.assertGreater(long_text["probability"], short_text["probability"])
        self.assertGreater(long_text["probability"], 0.5)
        self.assertGreater(short_text["confidence_interval_90"][1] - short_text["confidence_interval_90"][0], 0.5)

    def test_common_store_roundtrip(self) -> None:
        payload = {"schema_version": "1.0.0", "profile_id": "test-profile"}
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "fingerprint_store"
            guid, fp_path, _meta_path = common.save_fingerprint_to_store(payload, store_dir)
            self.assertTrue(common.is_valid_guid(guid))
            self.assertTrue(fp_path.exists())
            loaded, loaded_path, meta = common.load_fingerprint_from_store(guid, store_dir)
            self.assertEqual(loaded["profile_id"], "test-profile")
            self.assertEqual(loaded_path, fp_path)
            self.assertEqual(meta.get("id"), guid)

    def test_common_fingerprint_similarity_identical(self) -> None:
        fp = {
            "metadata": {"corpus": {"size": {"words_est": 5000}}},
            "measurements": {
                "sentence": {"length_words": {"histogram_p": [0.2, 0.4, 0.3, 0.1]}},
                "paragraph": {"length_sentences_histogram_p": [0.3, 0.5, 0.2]},
                "function_words": {"rates_per_1000w": {"and": 30.0, "the": 70.0, "but": 4.0}},
                "punctuation": {"rates_per_1000w": {"commas": 35.0, "semicolons": 3.0}},
                "stance_signals": {"hedge_rate": 4.0, "booster_rate": 1.0},
                "rhetoric_moves": {"claim_rate": 6.0, "evidence_rate": 5.0},
                "syntax_texture": {"subordinate_clause_rate": 8.0},
                "paragraph_cadence": {"opening_sentence_length_mean": 14.0},
                "repetition": {"bigram_repeat_rate": 0.06, "trigram_repeat_rate": 0.03},
            },
            "lexicon": {
                "preferred_words": ["therefore"],
                "preferred_phrases": ["in practice"],
                "avoid_words": ["very"],
            },
        }
        result = common.compute_fingerprint_similarity(fp, fp)
        self.assertGreaterEqual(result["similarity_score"], 0.99)
        self.assertLessEqual(result["distance_score"], 0.01)

    def test_common_fingerprint_similarity_detects_difference(self) -> None:
        fp_a = {
            "metadata": {"corpus": {"size": {"words_est": 7000}}},
            "measurements": {
                "sentence": {"length_words": {"histogram_p": [0.7, 0.2, 0.1]}},
                "paragraph": {"length_sentences_histogram_p": [0.8, 0.2]},
                "function_words": {"rates_per_1000w": {"and": 60.0, "the": 80.0}},
                "punctuation": {"rates_per_1000w": {"commas": 5.0, "semicolons": 0.2}},
                "stance_signals": {"hedge_rate": 0.3},
                "rhetoric_moves": {"claim_rate": 2.0},
                "syntax_texture": {"subordinate_clause_rate": 1.0},
                "paragraph_cadence": {"opening_sentence_length_mean": 8.0},
                "repetition": {"bigram_repeat_rate": 0.02, "trigram_repeat_rate": 0.01},
            },
            "lexicon": {"preferred_words": ["quick"], "avoid_words": ["however"]},
        }
        fp_b = {
            "metadata": {"corpus": {"size": {"words_est": 7000}}},
            "measurements": {
                "sentence": {"length_words": {"histogram_p": [0.1, 0.2, 0.7]}},
                "paragraph": {"length_sentences_histogram_p": [0.1, 0.9]},
                "function_words": {"rates_per_1000w": {"and": 5.0, "the": 20.0}},
                "punctuation": {"rates_per_1000w": {"commas": 65.0, "semicolons": 9.0}},
                "stance_signals": {"hedge_rate": 9.0},
                "rhetoric_moves": {"claim_rate": 12.0},
                "syntax_texture": {"subordinate_clause_rate": 14.0},
                "paragraph_cadence": {"opening_sentence_length_mean": 28.0},
                "repetition": {"bigram_repeat_rate": 0.22, "trigram_repeat_rate": 0.14},
            },
            "lexicon": {"preferred_words": ["consequently"], "avoid_words": ["quick"]},
        }
        result = common.compute_fingerprint_similarity(fp_a, fp_b)
        self.assertLess(result["similarity_score"], 0.6)
        self.assertGreater(result["distance_score"], 0.4)


if __name__ == "__main__":
    unittest.main()
