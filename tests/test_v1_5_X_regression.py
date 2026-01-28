import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fingerprint_style as fs  # noqa: E402
import apply_fingerprint as af  # noqa: E402


class TestV15XRegression(unittest.TestCase):
    def test_measurements_include_new_signals_fingerprint(self) -> None:
        meas = fs.compute_measurements(["We therefore argue that this is likely."])
        self.assertIn("rhetoric_moves", meas)
        self.assertIn("paragraph_cadence", meas)
        self.assertIn("epistemic_profile", meas)
        self.assertIn("syntax_texture", meas)
        self.assertIn("lexical_avoidance", meas)
        self.assertIn("repetition", meas)
        self.assertIn("transition_marker_positions", meas.get("templates_signals", {}))

    def test_measurements_include_new_signals_apply(self) -> None:
        meas = af.compute_measurements("We therefore argue that this is likely.")
        self.assertIn("rhetoric_moves", meas)
        self.assertIn("paragraph_cadence", meas)
        self.assertIn("epistemic_profile", meas)
        self.assertIn("syntax_texture", meas)
        self.assertIn("lexical_avoidance", meas)
        self.assertIn("repetition", meas)
        self.assertIn("transition_marker_positions", meas.get("templates_signals", {}))

    def test_default_tunables_include_section_restore(self) -> None:
        tunables = af.DEFAULT_TUNABLES
        self.assertIn("section_restore", tunables)
        self.assertIn("heading_similarity_threshold", tunables["section_restore"])

    def test_restore_inline_code_inside_blockquote(self) -> None:
        md = "> Guidance for `stylometric-transfer`"
        masked_inline, inline_map = af.mask_inline_code(md)
        masked_frozen, frozen_map = af.mask_non_voice_blocks(masked_inline)
        restored = af.restore_placeholders(masked_frozen, frozen_map)
        restored = af.restore_placeholders(restored, inline_map)
        self.assertEqual(restored.strip(), md.strip())

    def test_normalize_heading_strips_emoji_and_markup(self) -> None:
        text = "Design principle: treat formulas as *signals* 😀"
        norm = af.normalize_heading(text)
        self.assertEqual(norm, "design principle treat formulas as signals")

    def test_heading_similarity_paraphrase(self) -> None:
        a = "Excel recalculation resembles a dataflow graph scheduler"
        b = "Excel recalculation as a dataflow graph"
        score = af.heading_similarity(af.normalize_heading(a), af.normalize_heading(b))
        self.assertGreaterEqual(score, 0.7)

    def test_tunables_snapshot_loads_for_fingerprinting(self) -> None:
        snapshot = fs.load_tunables_snapshot()
        self.assertIsInstance(snapshot, dict)
        self.assertIn("humanizer_conflicts", snapshot)
        self.assertIn("style_retry", snapshot)


if __name__ == "__main__":
    unittest.main()
