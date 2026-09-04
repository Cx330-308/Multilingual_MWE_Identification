from __future__ import annotations

import unittest

from mwe_roberta.data import SentenceExample
from mwe_roberta.hf_dataset import TokenClassificationDataset


class FakeEncoding(dict):
    def word_ids(self, batch_index: int) -> list[int | None]:
        return [None, 0, None]


class FakeTokenizer:
    def __call__(self, *args, **kwargs) -> FakeEncoding:
        return FakeEncoding(
            input_ids=[[0, 10, 2]],
            attention_mask=[[1, 1, 1]],
        )


class DatasetMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            SentenceExample(
                language="FR",
                source_path="dev.cupt",
                sentence_index=0,
                words=["bonjour"],
                labels=["O"],
            )
        ]

    def test_metadata_is_disabled_by_default(self) -> None:
        dataset = TokenClassificationDataset(
            self.examples,
            FakeTokenizer(),
            {"O": 0},
            max_length=16,
            stride=4,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.chunk_metadata, [])
        self.assertEqual(dataset.chunk_languages, [])

    def test_metadata_can_be_enabled_for_official_decoding(self) -> None:
        dataset = TokenClassificationDataset(
            self.examples,
            FakeTokenizer(),
            {"O": 0},
            max_length=16,
            stride=4,
            track_metadata=True,
        )

        self.assertEqual(len(dataset.chunk_metadata), 1)
        self.assertEqual(dataset.chunk_metadata[0].example_index, 0)
        self.assertEqual(dataset.chunk_metadata[0].word_ids, (None, 0, None))

    def test_chunk_languages_can_be_tracked_for_sampling(self) -> None:
        dataset = TokenClassificationDataset(
            self.examples,
            FakeTokenizer(),
            {"O": 0},
            max_length=16,
            stride=4,
            track_languages=True,
        )

        self.assertEqual(dataset.chunk_languages, ["FR"])


if __name__ == "__main__":
    unittest.main()
