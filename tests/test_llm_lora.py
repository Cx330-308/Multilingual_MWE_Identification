from __future__ import annotations

import argparse
import unittest

from mwe_roberta.cupt import CuptSentence, CuptToken
from mwe_roberta.llm_lora import (
    CompletionOnlyCollator,
    add_args,
    completion_text,
    encode_sft_example,
    select_sentence_indices,
    text_linear_module_names,
)


def sentence(*mwe_columns: str) -> CuptSentence:
    return CuptSentence(
        comments=[],
        tokens=[
            CuptToken(
                [str(index), f"w{index}", "_", "X", "_", "_", "0", "dep", "_", "_", mwe]
            )
            for index, mwe in enumerate(mwe_columns, start=1)
        ],
    )


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "PROMPT:" + "|".join(message["content"] for message in messages)

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [3 + (ord(character) % 89) for character in text]}


class CompletionObjectiveTest(unittest.TestCase):
    def test_completion_is_canonical_json(self) -> None:
        value = sentence("1:VID", "*", "1")

        self.assertEqual(completion_text(value), '{"mwes":[[1,3]]}')

    def test_prompt_tokens_are_masked_from_loss(self) -> None:
        example = encode_sft_example(
            FakeTokenizer(),
            language="FR",
            sentence_index=7,
            sentence=sentence("1:VID", "1"),
            max_length=2_000,
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertTrue(all(label == -100 for label in example.labels[: example.prompt_tokens]))
        self.assertEqual(
            example.labels[example.prompt_tokens :],
            example.input_ids[example.prompt_tokens :],
        )
        self.assertGreater(example.completion_tokens, 0)

    def test_overlength_example_is_excluded_without_truncating_answer(self) -> None:
        example = encode_sft_example(
            FakeTokenizer(),
            language="FR",
            sentence_index=0,
            sentence=sentence("1:VID", "1"),
            max_length=8,
        )

        self.assertIsNone(example)

    def test_collator_masks_padding_in_labels(self) -> None:
        batch = CompletionOnlyCollator(pad_token_id=0)(
            [
                {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
                {"input_ids": [6], "attention_mask": [1], "labels": [6]},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[4, 5], [6, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 5], [6, -100]])


class LoRASelectionTest(unittest.TestCase):
    def test_training_subset_is_deterministic_and_language_specific(self) -> None:
        first = select_sentence_indices(100, language="FR", limit=12, seed=13)
        second = select_sentence_indices(100, language="FR", limit=12, seed=13)
        other_language = select_sentence_indices(100, language="DE", limit=12, seed=13)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(first, sorted(first))
        self.assertNotEqual(first, other_language)

    def test_training_subset_seed_is_separate_and_backward_compatible(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)

        default = parser.parse_args([])
        explicit = parser.parse_args(
            ["--seed", "42", "--training-subset-seed", "13"]
        )

        self.assertIsNone(default.training_subset_seed)
        self.assertEqual(explicit.seed, 42)
        self.assertEqual(explicit.training_subset_seed, 13)

    def test_only_text_stack_linears_are_selected(self) -> None:
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.model.visual = torch.nn.Linear(2, 2)
                self.lm_head = torch.nn.Linear(2, 2)

        self.assertEqual(text_linear_module_names(Model()), ["model.layers.0"])


if __name__ == "__main__":
    unittest.main()
