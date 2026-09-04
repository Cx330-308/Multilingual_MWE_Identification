from __future__ import annotations

import unittest

from mwe_roberta.sampling import build_language_sampling_plan, create_weighted_sampler


class LanguageSamplingPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.languages = ["LARGE"] * 100 + ["SMALL"] * 4

    def test_alpha_one_preserves_natural_distribution(self) -> None:
        plan = build_language_sampling_plan(self.languages, alpha=1.0)

        self.assertFalse(plan.uses_weighted_sampling)
        self.assertAlmostEqual(plan.expected_probabilities["LARGE"], 100 / 104)
        self.assertAlmostEqual(plan.expected_probabilities["SMALL"], 4 / 104)
        self.assertEqual(set(plan.chunk_weights), {1.0})

    def test_alpha_half_uses_square_root_language_mass(self) -> None:
        plan = build_language_sampling_plan(self.languages, alpha=0.5)

        self.assertTrue(plan.uses_weighted_sampling)
        self.assertAlmostEqual(plan.language_weights["LARGE"], 0.1)
        self.assertAlmostEqual(plan.language_weights["SMALL"], 0.5)
        self.assertAlmostEqual(plan.expected_probabilities["LARGE"], 5 / 6)
        self.assertAlmostEqual(plan.expected_probabilities["SMALL"], 1 / 6)
        self.assertAlmostEqual(
            plan.expected_probabilities["LARGE"]
            / plan.expected_probabilities["SMALL"],
            5.0,
        )

    def test_alpha_zero_gives_equal_language_mass(self) -> None:
        plan = build_language_sampling_plan(self.languages, alpha=0.0)

        self.assertAlmostEqual(plan.expected_probabilities["LARGE"], 0.5)
        self.assertAlmostEqual(plan.expected_probabilities["SMALL"], 0.5)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_language_sampling_plan([], alpha=0.5)
        with self.assertRaises(ValueError):
            build_language_sampling_plan(["FR"], alpha=-0.1)
        with self.assertRaises(ValueError):
            build_language_sampling_plan(["FR"], alpha=1.1)

    def test_weighted_sampler_is_reproducible_and_keeps_epoch_length(self) -> None:
        plan = build_language_sampling_plan(self.languages, alpha=0.5)

        first = list(create_weighted_sampler(plan, len(self.languages), seed=13))
        second = list(create_weighted_sampler(plan, len(self.languages), seed=13))

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(self.languages))
        self.assertTrue(all(0 <= index < len(self.languages) for index in first))

    def test_weighted_sampler_rejects_a_dataset_mismatch(self) -> None:
        plan = build_language_sampling_plan(self.languages, alpha=0.5)

        with self.assertRaisesRegex(ValueError, "104 weights for 103 chunks"):
            create_weighted_sampler(plan, dataset_length=103, seed=13)
