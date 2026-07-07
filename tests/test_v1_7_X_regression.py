import sys
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fingerprint_style as fs  # noqa: E402
import apply_fingerprint as af  # noqa: E402
import utils as utils  # noqa: E402


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

    def test_heading_qualifier_sanitizer(self) -> None:
        md = (
            "# Humanisation, defined carefully\n"
            "## Code changes (quick fix)\n"
            "### A taxonomy of stylometric features (why these work)\n"
            "#### Methods, Results\n"
            "#### Data (2021)\n"
        )
        updated, count = af.enforce_heading_qualifiers(md)
        self.assertIn("# Humanisation, defined carefully", updated)
        self.assertIn("## Code changes", updated)
        self.assertIn("### A taxonomy of stylometric features", updated)
        self.assertIn("#### Methods, Results", updated)
        self.assertIn("#### Data (2021)", updated)
        self.assertEqual(count, 2)

    def test_heading_qualifier_allowlist(self) -> None:
        md = "## Code changes (quick fix)\n"
        allowlist = af.compile_heading_allowlist([r"Code changes"])
        updated, count = af.enforce_heading_qualifiers(md, allowlist)
        self.assertIn("## Code changes (quick fix)", updated)
        self.assertEqual(count, 0)

    def test_transfer_heading_casing_positionally(self) -> None:
        source = "Humanisation, defined carefully"
        rewritten = "humanization, Defined Carefully"
        updated = af.transfer_heading_casing(source, rewritten)
        self.assertEqual(updated, "Humanization, defined carefully")

    def test_enforce_heading_casing_from_source(self) -> None:
        source_md = "# Humanisation, defined carefully\n\n## CODE CHANGES (QUICK FIX)\n"
        rewritten_md = "# humanization, Defined carefully\n\n## code changes (quick fix)\n"
        updated, edits = af.enforce_heading_casing_from_source(source_md, rewritten_md)
        self.assertGreaterEqual(edits, 2)
        self.assertIn("# Humanization, defined carefully", updated)
        self.assertIn("## CODE CHANGES (QUICK FIX)", updated)

    def test_filter_humanizer_rules_drops_title_case_rule_when_heading_case_locked(self) -> None:
        rules = [{"title": "Title Case in Headings", "words_to_watch": []}]
        _, dropped = af.filter_humanizer_rules(
            rules,
            {},
            input_style={"heading_title_case_rate": 0.1, "boldface_per_1000w": 0.0, "inline_header_list_rate": 0.0},
            tunables={"humanizer_mandatory": {"heading_case_normalization": "identical"}},
        )
        self.assertEqual(len(dropped), 1)
        self.assertIn("Heading case normalization is deterministic", dropped[0].get("drop_reason", ""))

    def test_heading_case_by_level_modes(self) -> None:
        source_md = (
            "# alpha beta gamma\n"
            "## alpha beta gamma\n"
            "### Alpha Beta gamma\n"
            "#### Alpha Beta GammA\n"
            "##### Alpha Beta GammA\n"
            "###### Alpha Beta GammA\n"
        )
        rewritten_md = (
            "# random case\n"
            "## RANDOM CASE\n"
            "### alpha BETA gamma\n"
            "#### rAnDom cASe\n"
            "##### random case\n"
            "###### RANDOM CASE\n"
        )
        by_level = {
            1: "title-case",
            2: "sentence-case",
            3: "identical",
            4: "automatic",
            5: "caps",
            6: "lower",
        }
        updated, edits = af.enforce_heading_case_normalization_from_source(
            source_md,
            rewritten_md,
            "by-level",
            by_level,
            preserve_proper_name_case=False,
        )
        self.assertGreaterEqual(edits, 5)
        self.assertIn("# Random Case", updated)
        self.assertIn("## Random case", updated)
        self.assertIn("### Alpha Beta gamma", updated)
        self.assertIn("#### rAnDom cASe", updated)
        self.assertIn("##### RANDOM CASE", updated)
        self.assertIn("###### random case", updated)

    def test_heading_case_config_accepts_h7_h8(self) -> None:
        mode, by_level, preserve = af.get_heading_case_normalization_conf(
            {
                "humanizer_mandatory": {
                    "heading_case_normalization": "by-level",
                    "heading_case_by_level": {
                        "h7": "caps",
                        "h8": "upper",
                    },
                    "preserve_proper_name_case": False,
                }
            }
        )
        self.assertEqual(mode, "by-level")
        self.assertEqual(by_level.get(7), "caps")
        self.assertEqual(by_level.get(8), "caps")
        self.assertFalse(preserve)

    def test_heading_case_by_level_h7_supported(self) -> None:
        source_md = "####### Alpha Beta\n"
        rewritten_md = "####### mixed CASE\n"
        updated, edits = af.enforce_heading_case_normalization_from_source(
            source_md,
            rewritten_md,
            "by-level",
            {7: "lower"},
            preserve_proper_name_case=False,
        )
        self.assertEqual(edits, 1)
        self.assertIn("####### mixed case", updated)

    def test_heading_case_caps_preserves_proper_names_when_enabled(self) -> None:
        source_heading = "John Black has mary"
        rewritten_heading = "john black has mary"
        updated = af.apply_heading_case_style(
            rewritten_heading,
            "caps",
            source_heading=source_heading,
            preserve_proper_name_case=True,
        )
        self.assertEqual(updated, "John Black HAS MARY")

    def test_heading_case_sentence_case_does_not_preserve_generic_title_tokens(self) -> None:
        source_heading = "Runtime Deployment memory"
        rewritten_heading = "Runtime Deployment memory"
        updated = af.apply_heading_case_style(
            rewritten_heading,
            "sentence-case",
            source_heading=source_heading,
            preserve_proper_name_case=True,
        )
        self.assertEqual(updated, "Runtime deployment memory")

    def test_heading_case_sentence_case_downcases_conjunction_heading(self) -> None:
        source_heading = "Tool Connectivity and Protocols"
        rewritten_heading = "Tool Connectivity and Protocols"
        updated = af.apply_heading_case_style(
            rewritten_heading,
            "sentence-case",
            source_heading=source_heading,
            preserve_proper_name_case=True,
        )
        self.assertEqual(updated, "Tool connectivity and protocols")

    def test_heading_case_caps_does_not_preserve_proper_names_when_disabled(self) -> None:
        source_heading = "John Black and mary"
        rewritten_heading = "john black and mary"
        updated = af.apply_heading_case_style(
            rewritten_heading,
            "caps",
            source_heading=source_heading,
            preserve_proper_name_case=False,
        )
        self.assertEqual(updated, "JOHN BLACK AND MARY")

    def test_heading_case_lock_with_spelling_forced_qualifier_retained(self) -> None:
        rules = af.load_local_spelling_rules()
        source_md = "# John Black has a color in his name (take note!)\n"
        rewritten_md = "# john black Has a color in HIS name (take note!)\n"
        localized, _ = af.enforce_local_spelling_guarded(
            rewritten_md,
            "canadian",
            rules,
            preserve_multiword_quotes=False,
        )
        updated, edits = af.enforce_heading_casing_from_source(source_md, localized)
        self.assertGreaterEqual(edits, 1)
        self.assertIn("# John Black has a colour in his name (take note!)", updated)

    def test_heading_case_lock_with_spelling_forced_qualifier_removed(self) -> None:
        rules = af.load_local_spelling_rules()
        source_md = "# John Black has a color in his name (take note!)\n"
        rewritten_md = "# john black Has a color in HIS name\n"
        localized, _ = af.enforce_local_spelling_guarded(
            rewritten_md,
            "canadian",
            rules,
            preserve_multiword_quotes=False,
        )
        updated, edits = af.enforce_heading_casing_from_source(source_md, localized)
        self.assertGreaterEqual(edits, 1)
        self.assertIn("# John Black has a colour in his name", updated)
        self.assertNotIn("(take note!)", updated)

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

    def test_avoidance_normalizes_ise_ize_noun_forms_to_us(self) -> None:
        rules = af.load_local_spelling_rules()
        tokens = {"humanisation", "organisation", "reprioritisation"}
        normalized = af.normalize_tokens_for_avoidance(tokens, rules)
        self.assertIn("humanization", normalized)
        self.assertIn("organization", normalized)
        self.assertIn("reprioritization", normalized)

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

    def test_summary_requested_only_on_first_attempt(self) -> None:
        self.assertTrue(af.should_request_chunk_summary(1, True, False, False))
        self.assertFalse(af.should_request_chunk_summary(2, True, False, False))
        self.assertFalse(af.should_request_chunk_summary(1, False, False, False))
        self.assertFalse(af.should_request_chunk_summary(1, True, True, False))
        self.assertTrue(af.should_request_chunk_summary(2, True, False, True))

    def test_force_local_spelling_split_conf_defaults(self) -> None:
        llm_mode, rules_mode = af.get_force_local_spelling_conf({})
        self.assertEqual(llm_mode, "none")
        self.assertEqual(rules_mode, "none")

    def test_force_local_spelling_split_conf_legacy_fallback(self) -> None:
        llm_mode, rules_mode = af.get_force_local_spelling_conf(
            {"humanizer_mandatory": {"force_local_spelling": "canadian"}}
        )
        self.assertEqual(llm_mode, "canadian")
        self.assertEqual(rules_mode, "canadian")

    def test_force_local_spelling_split_conf_independent(self) -> None:
        llm_mode, rules_mode = af.get_force_local_spelling_conf(
            {
                "humanizer_mandatory": {
                    "force_local_spelling_LLM": "us",
                    "force_local_spelling_rules": "canadian",
                }
            }
        )
        self.assertEqual(llm_mode, "us")
        self.assertEqual(rules_mode, "canadian")

    def test_build_apply_prompt_includes_llm_spelling_instruction(self) -> None:
        cfg = af.LLMConfig(
            api_key="k",
            base_url="http://example.test/v1",
            model="m",
            max_tokens=1000,
            temperature=0.0,
            timeout_seconds=30,
            extra_headers={},
            max_prompt_tokens=1000,
            max_retries=0,
            backoff_base_seconds=0.1,
            backoff_max_seconds=0.1,
        )
        prompts = {
            "apply": {
                "system": "s",
                "user": {"rules": [], "output_format": {"final_markdown": "string"}},
            }
        }
        messages = af.build_apply_prompt(
            {},
            "input",
            {},
            cfg,
            prompts,
            local_spelling_llm="us",
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload.get("local_spelling_target"), "us")
        rules = payload.get("rules", [])
        self.assertTrue(any("US English spelling" in r for r in rules))

    def test_apply_prompt_instructs_llm_to_escape_double_quotes(self) -> None:
        cfg = af.LLMConfig(
            api_key="k",
            base_url="http://example.test/v1",
            model="m",
            max_tokens=1000,
            temperature=0.0,
            timeout_seconds=30,
            extra_headers={},
            max_prompt_tokens=1000,
            max_retries=0,
            backoff_base_seconds=0.1,
            backoff_max_seconds=0.1,
        )
        prompts = af.load_prompts()
        messages = af.build_apply_prompt({}, 'He said "hello".', {}, cfg, prompts)
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        rules = payload.get("rules", [])
        self.assertIn("escape any double quote characters inside final_markdown", system)
        self.assertTrue(any("escape each one" in r and "\\\"" in r for r in rules))

    def test_build_apply_prompt_uses_author_name_for_third_person(self) -> None:
        cfg = af.LLMConfig(
            api_key="k",
            base_url="http://example.test/v1",
            model="m",
            max_tokens=1000,
            temperature=0.0,
            timeout_seconds=30,
            extra_headers={},
            max_prompt_tokens=1000,
            max_retries=0,
            backoff_base_seconds=0.1,
            backoff_max_seconds=0.1,
        )
        prompts = {
            "apply": {
                "system": "s",
                "user": {"rules": [], "output_format": {"final_markdown": "string"}},
            }
        }
        messages = af.build_apply_prompt(
            {},
            "# Section\n\nHe makes the claim.",
            {},
            cfg,
            prompts,
            voice_override="third",
            author_name="Nico Pepin",
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload.get("author_name"), "Nico Pepin")
        rules = payload.get("rules", [])
        self.assertTrue(any("After each Markdown heading" in r and "Nico Pepin" in r for r in rules))

    def test_build_apply_prompt_ignores_author_name_without_third_person(self) -> None:
        cfg = af.LLMConfig(
            api_key="k",
            base_url="http://example.test/v1",
            model="m",
            max_tokens=1000,
            temperature=0.0,
            timeout_seconds=30,
            extra_headers={},
            max_prompt_tokens=1000,
            max_retries=0,
            backoff_base_seconds=0.1,
            backoff_max_seconds=0.1,
        )
        prompts = {
            "apply": {
                "system": "s",
                "user": {"rules": [], "output_format": {"final_markdown": "string"}},
            }
        }
        messages = af.build_apply_prompt(
            {},
            "input",
            {},
            cfg,
            prompts,
            voice_override="first",
            author_name="Nico Pepin",
        )
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("author_name", payload)

    def test_parse_json_strict_repairs_unescaped_final_markdown_quotes(self) -> None:
        raw = '''{
          "final_markdown": "A person who is seen as "difficult," then dismissed as "difficult" simply because.",
          "deviations": [],
          "self_check": {"notes": []}
        }'''
        parsed = af.parse_json_strict(raw)
        self.assertEqual(
            parsed["final_markdown"],
            'A person who is seen as "difficult," then dismissed as "difficult" simply because.',
        )

    def test_parse_json_strict_repairs_quote_comma_inside_final_markdown(self) -> None:
        raw = '''{
          "final_markdown": "She said "yes," then walked away.",
          "deviations": [],
          "self_check": {"notes": []}
        }'''
        parsed = af.parse_json_strict(raw)
        self.assertEqual(parsed["final_markdown"], 'She said "yes," then walked away.')

    def test_parse_json_strict_repairs_unescaped_deviation_reason_quotes(self) -> None:
        raw = '''{
          "final_markdown": "Done.",
          "deviations": [
            {
              "rule_or_field": "/input_markdown/headings/Example 2",
              "reason": "Corrected the apparent typographical heading "EExample 2" to "Example 2" without changing meaning."
            },
            {
              "rule_or_field": "/style_fingerprint_json/lexicon/spelling_preferences",
              "reason": "Used US spellings such as "colorism," "color," and "behavior" because the task required US English."
            }
          ],
          "self_check": {"notes": []}
        }'''
        parsed = af.parse_json_strict(raw)
        reasons = [item["reason"] for item in parsed["deviations"]]
        self.assertIn('"EExample 2" to "Example 2"', reasons[0])
        self.assertIn('"colorism," "color," and "behavior"', reasons[1])

    def test_parse_json_strict_repairs_unescaped_self_check_note_quotes(self) -> None:
        raw = '''{
          "final_markdown": "Done.",
          "deviations": [],
          "self_check": {
            "notes": [
              "Returned valid JSON only.",
              "Converted the passage to third-person narration and used "Francia" at the first natural author reference after each heading."
            ]
          }
        }'''
        parsed = af.parse_json_strict(raw)
        self.assertIn(
            'used "Francia" at the first natural author reference',
            parsed["self_check"]["notes"][1],
        )

    def test_force_local_spelling_canadian_exceptions(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "He replaced the tyre whilst he spelled the word."
        updated, _ = af.enforce_local_spelling(text, "canadian", rules)
        self.assertIn("tire", updated.lower())
        self.assertIn("while", updated.lower())
        self.assertIn("spelled", updated.lower())

    def test_force_local_spelling_canadian_programme_always_normalizes(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "AI Strategy and Programme Delivery with no broadcast context."
        updated, _ = af.enforce_local_spelling(text, "canadian", rules)
        self.assertIn("Program Delivery", updated)
        self.assertNotIn("Programme", updated)

    def test_force_local_spelling_handles_ise_ize_noun_forms(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "Humanisation and organisation improved after reprioritisation, specialisation, optimisation, and visualisation."
        updated, _ = af.enforce_local_spelling(text, "canadian", rules)
        self.assertIn("humanization", updated.lower())
        self.assertIn("organization", updated.lower())
        self.assertIn("reprioritization", updated.lower())
        self.assertIn("specialization", updated.lower())
        self.assertIn("optimization", updated.lower())
        self.assertIn("visualization", updated.lower())

    def test_heading_proper_name_preservation_does_not_undo_spelling(self) -> None:
        source = "AI Strategy and Programme Delivery"
        rewritten = "AI Strategy and Program Delivery"
        updated = af.apply_heading_case_style(
            rewritten,
            "title-case",
            source_heading=source,
            preserve_proper_name_case=True,
        )
        self.assertEqual(updated, "AI Strategy and Program Delivery")

    def test_force_local_spelling_guarded_preserves_nonfiction_multiword_quotes(self) -> None:
        rules = af.load_local_spelling_rules()
        text = (
            "## Humanisation\n\n"
            "\"humanisation should stay as-is\"\n\n"
            "Humanisation should normalize outside protected quotes."
        )
        updated, _ = af.enforce_local_spelling_guarded(
            text,
            "canadian",
            rules,
            preserve_multiword_quotes=True,
        )
        self.assertIn("## Humanization", updated)
        self.assertIn("\"humanisation should stay as-is\"", updated)
        self.assertIn("Humanization should normalize outside protected quotes.", updated)

    def test_force_local_spelling_guarded_converts_quotes_in_fiction_mode(self) -> None:
        rules = af.load_local_spelling_rules()
        text = "\"humanisation in dialogue should normalize\""
        updated, _ = af.enforce_local_spelling_guarded(
            text,
            "canadian",
            rules,
            preserve_multiword_quotes=False,
        )
        self.assertIn("humanization in dialogue should normalize", updated.lower())

    def test_pronoun_override_debug_format(self) -> None:
        debug = af.format_pronoun_override_debug(
            {
                "mode": "first",
                "allowed_count": 4,
                "violations": {"second_person": 2, "third_person": 7},
                "ignored_non_subject": {"third_person_non_subject": 3},
            }
        )
        self.assertIn("mode=first", debug)
        self.assertIn("allowed_count=4", debug)
        self.assertIn("second_person=2", debug)
        self.assertIn("third_person=7", debug)
        self.assertIn("third_person_non_subject=3", debug)

    def test_pronoun_override_first_person_allows_third_person_object(self) -> None:
        text = "I had to help him when he was injured."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        # "him" is object-like and should not be counted as a violation.
        self.assertGreaterEqual(eval_obj.get("ignored_non_subject", {}).get("third_person_non_subject", 0), 1)
        # "he was" is subject-like and should still count as a violation.
        self.assertGreaterEqual(eval_obj.get("violations", {}).get("third_person", 0), 1)

    def test_pronoun_override_first_person_subject_violation(self) -> None:
        text = "She opened the door, and I followed."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        self.assertGreaterEqual(eval_obj.get("violations", {}).get("third_person", 0), 1)
        sentence_violations = eval_obj.get("sentence_violations", [])
        self.assertTrue(sentence_violations)
        self.assertEqual(sentence_violations[0].get("target_voice"), "first")

    def test_pronoun_override_handles_contractions(self) -> None:
        text = "I'm ready, but she's late."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        self.assertGreaterEqual(eval_obj.get("allowed_count", 0), 1)
        self.assertGreaterEqual(eval_obj.get("violations", {}).get("third_person", 0), 1)

    def test_deterministic_pronoun_repair_reduces_third_person_violations(self) -> None:
        text = "You can see why the decision matters. The next point is unchanged."
        before = af.evaluate_pronoun_override(text, "third")
        repaired, repairs = af.apply_deterministic_pronoun_repairs(text, "third", before)
        after = af.evaluate_pronoun_override(repaired, "third")
        self.assertTrue(repairs)
        self.assertIn("It is possible to see", repaired)
        self.assertGreater(af.pronoun_override_quality(after), af.pronoun_override_quality(before))

    def test_pronoun_sentence_repair_result_applies_only_improvements(self) -> None:
        text = "You can see why it matters. She left afterward."
        eval_obj = af.evaluate_pronoun_override(text, "third")
        result = {
            "sentences": [
                {"index": 0, "replacement": "It is possible to see why it matters."},
                {"index": 1, "replacement": "I left afterward."},
            ],
            "deviations": [],
        }
        repaired, repairs = af.apply_pronoun_sentence_repair_result(
            text,
            result,
            eval_obj.get("sentence_violations", []),
            "third",
        )
        self.assertIn("It is possible to see why it matters.", repaired)
        self.assertNotIn("I left afterward.", repaired)
        self.assertEqual(len(repairs), 1)

    def test_pronoun_sentence_repair_prompt_includes_author_name_for_third_person(self) -> None:
        messages = af.build_pronoun_sentence_repair_prompt(
            "third",
            [{"index": 0, "text": "He argues the point.", "violations": {}, "tokens": ["He"]}],
            "Nico Pepin",
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload.get("author_name"), "Nico Pepin")
        self.assertTrue(any("Nico Pepin" in r for r in payload.get("rules", [])))

    def test_pronoun_override_quality_prefers_fewer_violations(self) -> None:
        low = {"allowed_count": 2, "violations": {"third_person": 1}}
        high = {"allowed_count": 1, "violations": {"third_person": 3}}
        self.assertGreater(af.pronoun_override_quality(low), af.pronoun_override_quality(high))

    def test_pronoun_override_quality_tiebreaks_by_allowed_count(self) -> None:
        a = {"allowed_count": 5, "violations": {"third_person": 2}}
        b = {"allowed_count": 3, "violations": {"third_person": 2}}
        self.assertGreater(af.pronoun_override_quality(a), af.pronoun_override_quality(b))

    def test_resolve_retry_budgets_separate_voice_cap(self) -> None:
        style_cap, voice_cap = af.resolve_retry_budgets(
            {"max_retries": 3, "voice_max_retries": 2},
            cli_style_retries=1,
            cli_default_style_retries=1,
        )
        self.assertEqual(style_cap, 3)
        self.assertEqual(voice_cap, 2)

    def test_resolve_retry_budgets_cli_style_override_keeps_voice_fallback(self) -> None:
        style_cap, voice_cap = af.resolve_retry_budgets(
            {"max_retries": 3},
            cli_style_retries=5,
            cli_default_style_retries=1,
        )
        # CLI overrides style cap; voice follows style when no explicit voice cap exists.
        self.assertEqual(style_cap, 5)
        self.assertEqual(voice_cap, 5)

    def test_pronoun_override_first_person_svo_object_pattern(self) -> None:
        text = "I saw her and thanked them for helping me."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        self.assertEqual(eval_obj.get("violations", {}).get("third_person", 0), 0)
        self.assertGreaterEqual(eval_obj.get("ignored_non_subject", {}).get("third_person_non_subject", 0), 2)

    def test_pronoun_override_first_person_preposition_object(self) -> None:
        text = "We spoke to them before the meeting."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        self.assertEqual(eval_obj.get("violations", {}).get("third_person", 0), 0)
        self.assertGreaterEqual(eval_obj.get("ignored_non_subject", {}).get("third_person_non_subject", 0), 1)

    def test_pronoun_override_first_person_nonstandard_subject_object_form(self) -> None:
        text = "Him and I left early."
        eval_obj = af.evaluate_pronoun_override(text, "first")
        self.assertGreaterEqual(eval_obj.get("violations", {}).get("third_person", 0), 1)

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

    def test_dedupe_redundant_prose_blocks_drops_near_duplicate(self) -> None:
        a = (
            "This section explains how the platform enforces policy constraints while preserving semantic intent. "
            "It covers execution flow, observability hooks, and guardrails for failure handling."
        )
        b = (
            "This section explains how the platform enforces policy constraints while preserving semantic intent. "
            "It covers execution flow, observability hooks, and guardrails for failure handling in production."
        )
        c = "This paragraph is different and should remain intact."
        text = f"{a}\n\n{b}\n\n{c}"
        updated, dropped = af.dedupe_redundant_prose_blocks(
            text,
            min_words=10,
            similarity_threshold=0.95,
            lookback_blocks=10,
            max_drop_ratio=0.5,
        )
        self.assertEqual(dropped, 1)
        self.assertIn(c, updated)

    def test_dedupe_redundant_prose_blocks_keeps_list_blocks(self) -> None:
        text = (
            "- alpha\n"
            "- beta\n"
            "- gamma\n\n"
            "- alpha\n"
            "- beta\n"
            "- gamma\n"
        )
        updated, dropped = af.dedupe_redundant_prose_blocks(text, min_words=1)
        self.assertEqual(dropped, 0)
        self.assertEqual(updated.count("- alpha"), 2)

    def test_throttle_unordered_list_density_groups_long_runs(self) -> None:
        text = "\n".join(
            [
                "- one",
                "- two",
                "- three",
                "- four",
                "- five",
                "- six",
                "- seven",
                "- eight",
            ]
        )
        updated, runs, merged_items = af.throttle_unordered_list_density(
            text,
            min_run_length=6,
            group_size=2,
            joiner="; ",
        )
        self.assertEqual(runs, 1)
        self.assertEqual(merged_items, 4)
        self.assertIn("- one; two", updated)
        self.assertEqual(updated.count("\n- "), 3)

    def test_postprocess_redundancy_conf_defaults_and_overrides(self) -> None:
        conf = af.get_postprocess_redundancy_conf(
            {
                "postprocess_redundancy": {
                    "enabled": True,
                    "paragraph_dedupe": {"min_words": 20},
                    "list_density": {"group_size": 3},
                }
            }
        )
        self.assertTrue(conf.get("enabled"))
        self.assertEqual(conf["paragraph_dedupe"]["min_words"], 20)
        self.assertEqual(conf["list_density"]["group_size"], 3)

    def test_utils_common_helpers(self) -> None:
        text = "One token, two tokens."
        self.assertEqual(utils.words(text), ["One", "token", "two", "tokens"])
        self.assertEqual(utils.clamp01(-0.2), 0.0)
        self.assertEqual(utils.clamp01(1.2), 1.0)
        hist = utils.histogram([1, 2, 2, 3, 10], [(1, 1), (2, 2), (3, 5), (6, None)])
        self.assertEqual(len(hist), 4)
        self.assertAlmostEqual(sum(hist), 1.0)
        self.assertAlmostEqual(hist[0], 0.2, places=2)
        self.assertAlmostEqual(utils.approx_rate_per_1000_words(5, 1000), 5.0)
        self.assertAlmostEqual(utils.safe_mean([1, 2, 3]), 2.0)
        self.assertGreaterEqual(utils.safe_stdev([1, 2, 3]), 0.0)
        self.assertEqual(utils.split_sentences("One. Two?"), ["One.", "Two?"])
        self.assertEqual(utils.split_paragraphs("A\n\nB"), ["A", "B"])

    def test_rare_words_filters_duplicates_and_roman_numerals(self) -> None:
        rules = fs.load_local_spelling_rules()
        text = (
            "chairmanchairman xxiii reprioritised reprioritised "
            "alpha beta gamma delta"
        )
        measurements = fs.compute_measurements(
            [text],
            rare_words_limit=20,
            common_words=[],
            local_spelling_rules=rules,
        )
        rare_words = [item.get("word") for item in measurements["lexical_signals"]["rare_words"]]
        self.assertNotIn("chairmanchairman", rare_words)
        self.assertNotIn("xxiii", rare_words)
        self.assertIn("reprioritized", rare_words)
        self.assertNotIn("reprioritised", rare_words)

    def test_rare_words_filters_concatenation(self) -> None:
        rules = fs.load_local_spelling_rules()
        text = "mrcbgseniorfellows alpha beta gamma delta"
        measurements = fs.compute_measurements(
            [text],
            rare_words_limit=20,
            common_words=[],
            local_spelling_rules=rules,
        )
        rare_words = [item.get("word") for item in measurements["lexical_signals"]["rare_words"]]
        self.assertNotIn("mrcbgseniorfellows", rare_words)


if __name__ == "__main__":
    unittest.main()
