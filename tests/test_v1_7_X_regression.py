import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fingerprint_style as fs  # noqa: E402
import apply_fingerprint as af  # noqa: E402


class TestV17XRegression(unittest.TestCase):
    def test_rewrite_policy_compaction_merges_preserve(self) -> None:
        text = (
            "preserve technical, military, and procedural tone; "
            "avoid informal slang, emotional adjectives, and intensifiers; "
            "preserve technical and procedural details; "
            "Avoid informal slang except in dialogue; "
            "avoid intensifiers and emotional adjectives unless in dialogue; "
            "Preserve technical detail and narrative rhythm; "
            "avoid introducing informal or emotional language; "
            "Preserve narrative structure and technical detail."
        )
        normalized = af.normalize_rewrite_policy(
            text,
            {
                "jaccard_threshold": 0.6,
                "dedupe_on_subset": True,
                "prefer_more_specific": True,
                "compress_directives": True,
                "directive_verbs": [
                    "preserve", "avoid", "maintain", "ensure", "keep", "favor", "use",
                    "prefer", "minimize", "maximize", "do not", "don't"
                ],
                "stopwords": [
                    "the", "and", "of", "to", "a", "an", "in", "on", "for", "with", "or", "but",
                    "as", "by", "from", "into", "at", "that", "this", "these", "those", "be", "is",
                    "are", "was", "were", "been", "being"
                ],
            },
        )
        lower = normalized.lower()
        self.assertNotIn("; preserve", lower)
        self.assertIn("preserve", lower)
        self.assertIn("avoid", lower)

    def test_priority_order_filters_generic_tokens(self) -> None:
        normalized = af.normalize_priority_order(
            ["templates", "lexical", "syntactic", "rhetorical", "paragraph_cadence"]
        )
        self.assertIn("templates", normalized)
        self.assertIn("paragraph_cadence", normalized)
        self.assertNotIn("lexical", normalized)
        self.assertNotIn("syntactic", normalized)
        self.assertNotIn("rhetorical", normalized)

    def test_humanization_baseline_includes_punctuation_metrics(self) -> None:
        text = (
            "Short sentence, with commas, and more commas. "
            "Another sentence; with a semicolon: and a colon. "
            "Question? Exclaim! "
        ) * 6
        baseline = fs.compute_humanization_baseline(
            [text],
            {"enabled": True, "window_words": 60, "stride_words": 30, "min_window_words": 50, "max_windows": 10},
        )
        metrics = baseline.get("metrics", {})
        self.assertIn("comma_density_per_100w", metrics)
        self.assertIn("punctuation_semicolons_per_1000w", metrics)
        self.assertIn("punctuation_colons_per_1000w", metrics)
        self.assertIn("punctuation_em_dashes_per_1000w", metrics)

    def test_controller_overlay_maps_punctuation_targets(self) -> None:
        fingerprint = {
            "measurements": {
                "humanization_baseline": {
                    "enabled": True,
                    "metrics": {
                        "comma_density_per_100w": {"p50": 3.2},
                        "punctuation_semicolons_per_1000w": {"p50": 1.1},
                        "punctuation_colons_per_1000w": {"p50": 1.4},
                        "punctuation_em_dashes_per_1000w": {"p50": 0.5}
                    }
                }
            },
            "targets": {}
        }
        tunables = {
            "humanization_controller": {
                "enabled": True,
                "seed": 42,
                "quantiles": [0.5],
                "range_pct": 0.1,
                "min_width": 0.05,
                "max_width": 2.0,
                "allowed_metrics": [
                    "comma_density_per_100w",
                    "punctuation_semicolons_per_1000w",
                    "punctuation_colons_per_1000w",
                    "punctuation_em_dashes_per_1000w"
                ],
            }
        }
        fp_overlay, overlay = af.build_controller_overlay(fingerprint, tunables, 0, "Sample text.")
        self.assertIsNotNone(fp_overlay)
        punct = fp_overlay.get("targets", {}).get("punctuation", {})
        self.assertIn("comma_density_per_100w", punct)
        self.assertIn("semicolons_per_1000w", punct)
        self.assertIn("colons_per_1000w", punct)
        self.assertIn("em_dashes_per_1000w", punct)
        self.assertIsNotNone(overlay)

    def test_chunk_markdown_no_empty_and_multiple(self) -> None:
        md = "\n\n".join([f"Paragraph {i}. " + ("Word " * 40) for i in range(10)])

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks = af.chunk_markdown(md, build_messages, max_prompt_tokens=200, max_input_tokens_override=80)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.strip() for c in chunks))

    def test_chunk_markdown_preserves_code_fence(self) -> None:
        md = "Intro text.\n\n```python\nprint('hello')\nprint('world')\n```\n\nOutro text."

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks = af.chunk_markdown(md, build_messages, max_prompt_tokens=300, max_input_tokens_override=120)
        found_fence = False
        for chunk in chunks:
            if "```python" in chunk:
                found_fence = True
                lines = [l.strip() for l in chunk.splitlines() if l.strip()]
                fence_lines = [l for l in lines if l.startswith("```")]
                self.assertGreaterEqual(len(fence_lines), 2)
        self.assertTrue(found_fence)

    def test_split_oversize_block_code_fence(self) -> None:
        code = "\n".join([f"line {i}" for i in range(40)])
        block = f"```txt\n{code}\n```"
        parts = af.split_oversize_block(block, af.estimate_tokens, max_input_tokens=30)
        self.assertGreater(len(parts), 1)
        for part in parts:
            lines = part.strip().splitlines()
            self.assertTrue(lines[0].startswith("```"))
            self.assertTrue(lines[-1].startswith("```"))

    def test_chunk_markdown_override_affects_count(self) -> None:
        lines = [("word " * 20).strip() for _ in range(200)]
        md = "\n".join(lines)

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks_small = af.chunk_markdown(md, build_messages, max_prompt_tokens=2000, max_input_tokens_override=200)
        chunks_large = af.chunk_markdown(md, build_messages, max_prompt_tokens=2000, max_input_tokens_override=1000)
        self.assertGreater(len(chunks_small), len(chunks_large))

    def test_chunk_split_on_paragraph_fallbacks(self) -> None:
        md = " ".join(["word"] * 1200)

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks = af.chunk_markdown(
            md,
            build_messages,
            max_prompt_tokens=500,
            max_input_tokens_override=60,
            split_on="paragraph",
        )
        max_chars = max(200, 60) * 4
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= max_chars for c in chunks))

    def test_chunk_split_on_sentence_treats_list_lines(self) -> None:
        md = "- item one\n- item two\n- item three"

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks = af.chunk_markdown(
            md,
            build_messages,
            max_prompt_tokens=500,
            max_input_tokens_override=200,
            split_on="sentence",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("\n- item two\n", chunks[0])

    def test_chunk_split_on_word_limits_size(self) -> None:
        md = " ".join(["word"] * 500)

        def build_messages(md_chunk: str, _feedback=None, _for_estimate=False):
            return [{"role": "user", "content": md_chunk}]

        chunks = af.chunk_markdown(
            md,
            build_messages,
            max_prompt_tokens=300,
            max_input_tokens_override=40,
            split_on="word",
        )
        max_chars = max(200, 40) * 4
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= max_chars for c in chunks))

    def test_enforce_min_chunks_splits(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = af.enforce_min_chunks(text, [text], 2)
        self.assertGreaterEqual(len(chunks), 2)

    def test_humanizer_mandatory_normalizes_double_quotes(self) -> None:
        text = 'He said “Hello”, then wrote «Oui», and „No”.'
        updated, count = af.enforce_straight_double_quotes(text)
        self.assertEqual(count, 6)
        for ch in ("“", "”", "„", "«", "»"):
            self.assertNotIn(ch, updated)
        self.assertIn('"', updated)

    def test_force_local_spelling_canadian(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "The color of the center meter is gray. She will license the program."
        updated, count = af.enforce_local_spelling(text, "canadian", rules)
        self.assertGreater(count, 0)
        self.assertIn("colour", updated)
        self.assertIn("centre", updated)
        self.assertIn("metre", updated)
        self.assertIn("grey", updated)
        self.assertIn("license", updated)

    def test_force_local_spelling_tyre_context(self) -> None:
        rules = af.load_local_spelling_rules()
        automotive = "The tire pressure dropped near the wheel and axle."
        fatigue = "I tire easily after a long day."
        updated_auto, _ = af.enforce_local_spelling(automotive, "british", rules)
        updated_fatigue, _ = af.enforce_local_spelling(fatigue, "british", rules)
        self.assertIn("tyre", updated_auto.lower())
        self.assertIn("tire", updated_fatigue.lower())

    def test_force_local_spelling_spelt_grain_context(self) -> None:
        rules = af.load_local_spelling_rules()
        grain = "We baked spelt bread with whole grain flour."
        verb = "She spelt the name correctly."
        updated_grain, _ = af.enforce_local_spelling(grain, "us", rules)
        updated_verb, _ = af.enforce_local_spelling(verb, "us", rules)
        self.assertIn("spelt", updated_grain.lower())
        self.assertIn("spelled", updated_verb.lower())

    def test_force_local_spelling_gue_rules_skip_canadian(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "The dialog was short. The catalogue was updated."
        updated, _ = af.enforce_local_spelling(text, "canadian", rules)
        # Canadian locale should not force -gue conversions.
        self.assertIn("dialog", updated.lower())
        self.assertIn("catalogue", updated.lower())

    def test_avoidance_normalizes_to_us(self) -> None:
        rules = af.load_local_spelling_rules()
        tokens = {"colour", "centre"}
        normalized = af.normalize_tokens_for_avoidance(tokens, rules)
        self.assertIn("color", normalized)
        self.assertIn("center", normalized)

    def test_fingerprint_lexicon_us_normalization(self) -> None:
        rules = fs.load_local_spelling_rules()
        fingerprint = {
            "lexicon": {
                "preferred_words": ["colour", "favourite"],
                "preferred_phrases": ["colour scheme"],
                "avoid_words": ["colour", "armor"],
                "avoid_words_soft": ["centre"],
                "synonym_preferences": {
                    "favourite": ["colour"]
                }
            }
        }
        avoid_list = ["colour"]
        fs.normalize_lexicon_spelling(fingerprint, rules, avoid_list)
        lexicon = fingerprint["lexicon"]
        self.assertIn("color", lexicon["preferred_words"])
        self.assertIn("favorite", lexicon["preferred_words"])
        self.assertIn("color scheme", lexicon["preferred_phrases"])
        self.assertIn("colour", lexicon["avoid_words"])
        self.assertIn("center", lexicon["avoid_words_soft"])
        self.assertIn("favorite", lexicon["synonym_preferences"])
        self.assertIn("color", lexicon["synonym_preferences"]["favorite"])

    def test_summary_normalizes_previous_passage(self) -> None:
        summary = "The passage introduces a system for stylometric profiling."
        normalized = af.normalize_summary(summary, 50)
        self.assertTrue(normalized.startswith("The previous passage"))

    def test_force_local_spelling_canadian_exceptions(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "He replaced the tyre whilst he spelled the word."
        updated, _ = af.enforce_local_spelling(text, "canadian", rules)
        self.assertIn("tire", updated.lower())
        self.assertIn("while", updated.lower())
        self.assertIn("spelled", updated.lower())

    def test_local_spelling_skips_proper_noun_and_path(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "The ColorGuard brand filed C:\\Program Files\\ColorGuard\\readme.txt."
        updated, _ = af.enforce_local_spelling(text, "british", rules)
        self.assertIn("ColorGuard", updated)
        self.assertIn("C:\\Program Files\\ColorGuard\\readme.txt", updated)

    def test_local_spelling_possessive_and_hyphen(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "The colour's hue and colour-grade were noted."
        updated, _ = af.enforce_local_spelling(text, "us", rules)
        self.assertIn("color's", updated.lower())
        self.assertIn("color-grade", updated.lower())


if __name__ == "__main__":
    unittest.main()
