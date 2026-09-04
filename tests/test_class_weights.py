from __future__ import annotations

import math
import unittest

import torch

from mwe_roberta.class_weights import (
    build_class_weight_plan,
    weighted_token_cross_entropy,
)


class ClassWeightPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = ["O", "B-X", "I-X", "B-UNSEEN"]
        self.chunks = [
            {"labels": [-100, 0, 0, 1]},
            {"labels": [0, 2]},
        ]

    def test_uses_expected_counts_from_the_language_sampler(self) -> None:
        plan = build_class_weight_plan(
            self.chunks,
            self.labels,
            alpha=0.5,
            min_weight=0.5,
            max_weight=2.0,
            chunk_sampling_weights=[1.0, 3.0],
        )

        self.assertEqual(plan.count_basis, "expected_sampled_epoch")
        self.assertAlmostEqual(plan.label_counts[0], 2.5)
        self.assertAlmostEqual(plan.label_counts[1], 0.5)
        self.assertAlmostEqual(plan.label_counts[2], 1.5)
        self.assertEqual(plan.label_counts[3], 0.0)
        self.assertAlmostEqual(plan.reference_count, 1.5)
        self.assertAlmostEqual(plan.weights[0], math.sqrt(1.5 / 2.5))
        self.assertAlmostEqual(plan.weights[1], math.sqrt(1.5 / 0.5))
        self.assertEqual(plan.weights[2], 1.0)
        self.assertEqual(plan.weights[3], 2.0)

    def test_alpha_zero_preserves_standard_cross_entropy(self) -> None:
        plan = build_class_weight_plan(
            self.chunks,
            self.labels,
            alpha=0.0,
            min_weight=0.75,
            max_weight=2.5,
        )

        self.assertFalse(plan.uses_weighted_loss)
        self.assertEqual(plan.weights, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(plan.summary()["strategy"], "standard_cross_entropy")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_class_weight_plan([], [], 0.25, 0.75, 2.5)
        with self.assertRaisesRegex(ValueError, "2 weights for 1 chunks"):
            build_class_weight_plan(
                self.chunks[:1],
                self.labels,
                0.25,
                0.75,
                2.5,
                chunk_sampling_weights=[1.0, 2.0],
            )
        with self.assertRaisesRegex(ValueError, "Invalid aligned label id"):
            build_class_weight_plan(
                [{"labels": [99]}],
                self.labels,
                0.25,
                0.75,
                2.5,
            )


class WeightedTokenLossTest(unittest.TestCase):
    def test_applies_weights_and_ignores_special_tokens(self) -> None:
        logits = torch.tensor(
            [[[2.0, 0.0], [2.0, 0.0], [100.0, -100.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, -100]])
        weights = torch.tensor([1.0, 2.0])

        loss = weighted_token_cross_entropy(logits, targets, weights)

        expected = (0.126928 + 2.0 * 2.126928) / 3.0
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward()
        self.assertIsNotNone(logits.grad)
