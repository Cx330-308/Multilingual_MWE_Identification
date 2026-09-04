from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mwe_roberta.lora_summary import sample_statistics, summarize


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def evaluation_payload(value: float, sentences: int) -> dict[str, object]:
    scores = {"precision": value, "recall": value, "f1": value}
    return {
        "metrics": {
            "llm": {
                "average": {"mwe": scores, "token": scores},
                "per_language": {"XX": {"mwe": scores, "token": scores}},
            }
        },
        "response_status": {"ok": sentences, "warning": 0, "error": 0},
        "truncated_prompts": 0,
    }


class LoRASummaryTest(unittest.TestCase):
    def test_uses_sample_standard_deviation(self) -> None:
        result = sample_statistics([1.0, 2.0, 3.0])

        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_standard_deviation"], 1.0)

    def test_validates_fixed_training_composition_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, value in ((13, 0.4), (42, 0.5), (2026, 0.6)):
                run = root / "artifacts/experiments" / f"run-{seed}"
                write_json(
                    run / "experiment_config.json",
                    {
                        "status": "completed",
                        "arguments": {
                            "seed": seed,
                            "training_subset_seed": 13,
                        },
                        "data": {
                            "selected_sentences": {"XX": 2},
                            "encoded_sentences": {"XX": 2},
                            "total_encoded_sentences": 2,
                        },
                        "result": {
                            "global_step": 1,
                            "completed_epochs": 1.0,
                            "train_metrics": {"train_loss": value},
                        },
                    },
                )
                write_json(
                    run / "dev-subset-seed42/metrics.json",
                    evaluation_payload(value, 2),
                )
                write_json(
                    run / "dev-full/metrics.json",
                    evaluation_payload(value, 4),
                )

            result = summarize(
                root,
                seeds=(13, 42, 2026),
                run_prefix="run-",
                expected_matched_sentences=2,
                expected_full_sentences=4,
            )

        aggregate = result["aggregate"]["complete_development"]["mwe"]["f1"]
        self.assertAlmostEqual(aggregate["mean"], 0.5)
        self.assertAlmostEqual(aggregate["sample_standard_deviation"], 0.1)
        self.assertTrue(result["protocol"]["training_composition_identical"])


if __name__ == "__main__":
    unittest.main()
