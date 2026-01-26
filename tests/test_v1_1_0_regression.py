import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fingerprint_style as fs  # noqa: E402
import apply_fingerprint as af  # noqa: E402


class TestV110Regression(unittest.TestCase):
    def test_normalize_text_ocr(self) -> None:
        raw = "A sample ofﬁce text with hy-\nphen breaks.\n\n"
        out = fs.normalize_text(raw)
        self.assertIn("office", out)
        self.assertIn("hyphen", out)

    def test_parse_avoid_list(self) -> None:
        text = "# comment\nfoo\nbar # inline\n\n"
        self.assertEqual(fs.parse_avoid_list(text), ["foo", "bar"])
        self.assertEqual(af.parse_avoid_list(text), ["foo", "bar"])

    def test_merge_avoid_list_into_hints(self) -> None:
        hints = {"avoid_words": ["alpha"], "preferred_words": ["x"]}
        merged = fs.merge_avoid_list_into_hints(hints, ["beta", "alpha"])
        self.assertEqual(merged.get("avoid_words"), ["alpha", "beta"])

    def test_merge_avoid_list_into_fingerprint(self) -> None:
        fingerprint = {"lexicon": {"avoid_words": ["alpha"]}}
        merged = af.merge_avoid_list_into_fingerprint(fingerprint, ["beta", "alpha"])
        self.assertEqual(merged["lexicon"]["avoid_words"], ["alpha", "beta"])

    def test_enforce_no_em_dashes(self) -> None:
        fingerprint = {"targets": {"punctuation": {"em_dashes_per_1000w": {"target": [0.0, 0.0]}}}}
        self.assertTrue(af.should_forbid_em_dashes(fingerprint, []))
        text = "Alpha—beta."
        out, count = af.enforce_no_em_dashes(text)
        self.assertEqual(count, 1)
        self.assertNotIn("—", out)

    def test_normalize_humanizer_rules(self) -> None:
        raw = [{"title": "Test Rule", "problem": "Prob", "words_to_watch": "alpha, beta"}]
        norm = af.normalize_humanizer_rules(raw, "llm")
        self.assertEqual(norm[0]["words_to_watch"], ["alpha", "beta"])

    def test_extract_heading_blocks(self) -> None:
        md = "## Appendix C\nAlpha\n\n## Appendix D\nBeta\n"
        blocks = af.extract_heading_blocks(md)
        keys = af.extract_heading_keys(md)
        self.assertEqual(len(blocks), 2)
        self.assertIn("appendix c", keys)
        self.assertIn("appendix d", keys)

    def test_section_signature_similarity(self) -> None:
        a = "## Appendix C\nAlpha beta gamma\nMore text."
        b = "## Renamed Section\nAlpha beta gamma\nMore text."
        sig_a = af.section_signature(a)
        sig_b = af.section_signature(b)
        self.assertGreater(af.jaccard_similarity(sig_a, sig_b), 0.4)
        self.assertGreaterEqual(len(sig_a & sig_b), 3)

    def test_default_tunables_include_sanity_checks(self) -> None:
        tunables = af.DEFAULT_TUNABLES
        self.assertIn("sanity_checks", tunables)
        self.assertIn("line_count_warn_pct", tunables["sanity_checks"])
        self.assertIn("word_count_warn_pct", tunables["sanity_checks"])
        self.assertIn("paragraph_count_warn_pct", tunables["sanity_checks"])

    def test_filter_author_voice_text_removes_non_voice(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            "> Quoted material.\n"
            "> Still quoted.\n\n"
            "```python\n"
            "print('code block')\n"
            "```\n\n"
            "Inline `code span` should be removed.\n"
            "<div>HTML content</div>\n"
            "Inline $E=mc^2$ and \\alpha+\\beta should be removed.\n"
            "Inline $E=mc^2$ and \\alpha+\\beta should be removed.\n"
            "More prose (Smith 2020).\n\n"
            "## References\n"
            "- [1] Example reference\n\n"
            "[^1]: Footnote content\n"
        )
        cleaned = fs.filter_author_voice_text(text)
        self.assertNotIn("Quoted material", cleaned)
        self.assertNotIn("code block", cleaned)
        self.assertNotIn("code span", cleaned)
        self.assertNotIn("HTML content", cleaned)
        self.assertNotIn("E=mc^2", cleaned)
        self.assertNotIn("\\alpha+\\beta", cleaned)
        self.assertNotIn("References", cleaned)
        self.assertNotIn("Footnote content", cleaned)
        self.assertNotIn("Smith 2020", cleaned)
        self.assertIn("Intro paragraph", cleaned)
        self.assertIn("More prose", cleaned)

    def test_pick_representative_excerpts_prefers_clean_paragraph(self) -> None:
        doc = (
            "This is a clean paragraph with normal prose and few artifacts. "
            "It should be preferred for excerpts. The paragraph is long enough "
            "to meet the excerpt thresholds and includes several sentences to "
            "simulate real prose for scoring. It avoids obvious citation noise "
            "and is meant to be selected by the heuristic.\n\n"
            "ibid p 23 sup 99. This paragraph is full of citation junk and "
            "should be de-prioritized relative to the clean paragraph."
        )
        excerpts = fs.pick_representative_excerpts([("doc.md", doc)], max_total_chars=2000)
        self.assertTrue(excerpts)
        self.assertIn("clean paragraph", excerpts[0]["excerpt"])
        self.assertNotIn("ibid", excerpts[0]["excerpt"])

    def test_measurements_include_new_signals_fingerprint(self) -> None:
        meas = fs.compute_measurements(["We therefore argue that this is likely."])
        self.assertIn("function_words", meas)
        self.assertIn("stance_signals", meas)
        self.assertIn("templates_signals", meas)
        self.assertIn("lexical_signals", meas)

    def test_measurements_include_new_signals_apply(self) -> None:
        meas = af.compute_measurements("We therefore argue that this is likely.")
        self.assertIn("function_words", meas)
        self.assertIn("stance_signals", meas)
        self.assertIn("templates_signals", meas)

    def test_mask_and_restore_non_voice_blocks(self) -> None:
        md = "Intro.\n\n> Quote here\n\n[^1]: Footnote\n"
        masked, mapping = af.mask_non_voice_blocks(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored.strip(), md.strip())

    def test_mask_and_restore_inline_citations(self) -> None:
        md = "A sentence (Smith 2020) with [12] citations."
        masked, mapping = af.mask_inline_citations(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored, md)

    def test_mask_and_restore_inline_code(self) -> None:
        md = "Use `code` and ``more code`` inline."
        masked, mapping = af.mask_inline_code(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored, md)

    def test_mask_and_restore_html(self) -> None:
        md = "Inline <span>HTML</span> and <div>block</div> and <math>x^2</math>."
        masked, mapping = af.mask_html(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored, md)

    def test_mask_html_ignores_inequalities(self) -> None:
        md = "Bins are <10, 10-17 and score S < 0.75."
        masked, mapping = af.mask_html(md)
        self.assertFalse(mapping)
        self.assertEqual(masked, md)

    def test_mask_and_restore_math(self) -> None:
        md = "Inline $E=mc^2$ and $$x^2$$ plus \\(y\\) and \\[z\\] and \\begin{equation}a=b\\end{equation}."
        masked, mapping = af.mask_math_notation(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored, md)

    def test_mask_and_restore_html_entities(self) -> None:
        md = "Spacing&nbsp;and&nbsp;entities."
        masked, mapping = af.mask_html_entities(md)
        self.assertTrue(mapping)
        restored = af.restore_placeholders(masked, mapping)
        self.assertEqual(restored, md)

    def test_style_compliance_detects_drift(self) -> None:
        base_text = "This is a fairly long sentence with multiple clauses, therefore it reads like a report."
        fingerprint = {"measurements": fs.compute_measurements([base_text])}
        output_text = "Short sentence. Another short sentence."
        compliance = af.compute_style_compliance(fingerprint, output_text)
        self.assertLess(compliance["score"], 1.0)
        self.assertIsInstance(compliance.get("deltas"), list)

    def test_humanizer_rule_filtering(self) -> None:
        sample = (
            "### 13. Em Dash Overuse\n"
            "**Problem:** Overuse of em dash.\n"
            "**Words to watch:** em dash, —\n"
        )
        rules = af.parse_humanizer_guidelines(sample)
        fingerprint = {
            "measurements": {
                "punctuation": {"rates_per_1000w": {"em_dashes": 2.0}}
            }
        }
        kept, dropped = af.filter_humanizer_rules(rules, fingerprint)
        self.assertFalse(kept)
        self.assertTrue(dropped)

    def test_humanizer_rule_conflict_with_preferred_lexicon(self) -> None:
        sample = (
            "### 7. Overused \"AI Vocabulary\" Words\n"
            "**Words to watch:** therefore, however\n"
        )
        rules = af.parse_humanizer_guidelines(sample)
        fingerprint = {
            "measurements": {},
            "lexicon": {
                "preferred_words": ["therefore"]
            }
        }
        kept, dropped = af.filter_humanizer_rules(rules, fingerprint, {}, af.DEFAULT_TUNABLES)
        self.assertFalse(kept)
        self.assertTrue(dropped)

    def test_humanizer_heading_case_conflict(self) -> None:
        sample = "### 16. Title Case in Headings\n"
        rules = af.parse_humanizer_guidelines(sample)
        fingerprint = {"measurements": {}}
        input_style = {"heading_title_case_rate": 0.9}
        kept, dropped = af.filter_humanizer_rules(rules, fingerprint, input_style, af.DEFAULT_TUNABLES)
        self.assertFalse(kept)
        self.assertTrue(dropped)

    def test_humanizer_synonym_preferences_list(self) -> None:
        sample = (
            "### 7. Overused \"AI Vocabulary\" Words\n"
            "**Words to watch:** synergy\n"
        )
        rules = af.parse_humanizer_guidelines(sample)
        fingerprint = {
            "measurements": {},
            "lexicon": {
                "synonym_preferences": ["synergy"]
            }
        }
        kept, dropped = af.filter_humanizer_rules(rules, fingerprint, {}, af.DEFAULT_TUNABLES)
        self.assertFalse(kept)
        self.assertTrue(dropped)


if __name__ == "__main__":
    unittest.main()
