from __future__ import annotations

import json
import unittest

from mwe_roberta.llm_tuning import CANDIDATES, _validate_disjointness


class LlmTuningProtocolTest(unittest.TestCase):
    def test_candidate_grid_covers_shots_sampling_and_thinking(self) -> None:
        self.assertEqual({candidate.shots for candidate in CANDIDATES}, {0, 2, 4})
        self.assertTrue(any(candidate.do_sample for candidate in CANDIDATES))
        self.assertTrue(any(candidate.thinking for candidate in CANDIDATES))

    def test_disjointness_check_rejects_demo_target_overlap(self) -> None:
        manifest = json.dumps(
            {
                "language": "FR",
                "sentence_index": 7,
                "polarity": "positive",
                "word_count": 4,
            }
        )

        with self.assertRaisesRegex(ValueError, "Calibration leakage"):
            _validate_disjointness(manifest, {"FR": [7]})


if __name__ == "__main__":
    unittest.main()
