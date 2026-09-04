from __future__ import annotations

import unittest

import numpy as np

from mwe_roberta.data import SentenceExample
from mwe_roberta.hf_dataset import ChunkMetadata
from mwe_roberta.training_metrics import decode_chunk_predictions


class ChunkDecodingTest(unittest.TestCase):
    def test_averages_overlap_and_uses_first_subword(self) -> None:
        examples = [
            SentenceExample(
                language="FR",
                source_path="dev.cupt",
                sentence_index=0,
                words=["alpha", "beta", "gamma"],
            )
        ]
        metadata = [
            ChunkMetadata(example_index=0, word_ids=(None, 0, 0, 1, None)),
            ChunkMetadata(example_index=0, word_ids=(None, 1, 2, None)),
        ]
        logits = np.zeros((2, 5, 3), dtype=np.float32)
        logits[0, 1] = [0.0, 5.0, 0.0]
        logits[0, 2] = [0.0, 0.0, 100.0]
        logits[0, 3] = [3.0, 0.0, 0.0]
        logits[1, 1] = [0.0, 0.0, 5.0]
        logits[1, 2] = [0.0, 4.0, 0.0]

        decoded = decode_chunk_predictions(
            logits,
            examples,
            metadata,
            {0: "O", 1: "B-X", 2: "I-X"},
        )

        self.assertEqual(decoded, [["B-X", "I-X", "B-X"]])

    def test_rejects_chunk_count_mismatch(self) -> None:
        examples = [
            SentenceExample(
                language="FR",
                source_path="dev.cupt",
                sentence_index=0,
                words=["alpha"],
            )
        ]

        with self.assertRaisesRegex(ValueError, "prediction chunks"):
            decode_chunk_predictions(
                np.zeros((1, 2, 2), dtype=np.float32),
                examples,
                [],
                {0: "O", 1: "B-X"},
            )


if __name__ == "__main__":
    unittest.main()
