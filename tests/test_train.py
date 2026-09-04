from __future__ import annotations

import argparse
import unittest

import torch

from mwe_roberta.train import (
    _float32_load_kwargs,
    _serializable_arguments,
    _validate_training_controls,
)


class ExperimentArgumentsTest(unittest.TestCase):
    def test_excludes_cli_callback_from_saved_config(self) -> None:
        args = argparse.Namespace(
            epochs=3,
            selection_metric="official",
            func=lambda _: None,
        )

        serialized = _serializable_arguments(args)

        self.assertEqual(
            serialized,
            {"epochs": 3, "selection_metric": "official"},
        )

    def test_uses_version_appropriate_float32_loading_argument(self) -> None:
        self.assertEqual(
            _float32_load_kwargs(torch, "4.40.0"),
            {"torch_dtype": torch.float32},
        )
        self.assertEqual(
            _float32_load_kwargs(torch, "5.3.0"),
            {"dtype": torch.float32},
        )


class TrainingControlsTest(unittest.TestCase):
    def valid_args(self, **overrides: object) -> argparse.Namespace:
        values = {
            "epochs": 3.0,
            "batch_size": 8,
            "eval_batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_length": 256,
            "stride": 64,
            "gradient_accumulation_steps": 1,
            "language_sampling_alpha": 0.5,
            "class_weight_alpha": 0.0,
            "class_weight_min": 0.75,
            "class_weight_max": 2.5,
            "selection_metric": "official",
            "evaluation_strategy": "epoch",
            "early_stopping_patience": 1,
            "load_best_model_at_end": True,
            "save_strategy": "epoch",
            "eval_steps": 500,
            "save_steps": 500,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_accepts_controlled_tuning_configuration(self) -> None:
        _validate_training_controls(self.valid_args())

    def test_rejects_invalid_optimization_parameters(self) -> None:
        invalid_values = {
            "epochs": 0.0,
            "batch_size": 0,
            "eval_batch_size": 0,
            "learning_rate": 0.0,
            "weight_decay": -0.01,
            "warmup_ratio": 1.0,
            "gradient_accumulation_steps": 0,
        }

        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(SystemExit):
                _validate_training_controls(self.valid_args(**{name: value}))

    def test_rejects_stride_not_smaller_than_max_length(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_training_controls(
                self.valid_args(max_length=256, stride=256)
            )


if __name__ == "__main__":
    unittest.main()
