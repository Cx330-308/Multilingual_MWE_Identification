from __future__ import annotations

import unittest

from mwe_roberta.cupt import CuptSentence, CuptToken
from mwe_roberta.llm_baseline import (
    _report_interpretation_text,
    _report_protocol_lines,
    _report_system_name,
    _validate_demonstrations,
    all_sentence_indices,
    gold_mwe_groups,
    mwe_groups_to_columns,
    parse_llm_response,
    sample_sentence_indices,
    select_demonstration_indices,
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


class LlmResponseParsingTest(unittest.TestCase):
    def test_extracts_json_after_reasoning_and_code_fence(self) -> None:
        parsed = parse_llm_response(
            "Analysis omitted.\n```json\n{\"mwes\":[[4,2],[7]]}\n```",
            word_count=7,
        )

        self.assertEqual(parsed.groups, ((2, 4), (7,)))
        self.assertEqual(parsed.status, "ok")

    def test_drops_invalid_and_duplicate_groups_with_warnings(self) -> None:
        parsed = parse_llm_response(
            '{"mwes":[[1,1,3],[0,2],[1,3],["4"],[]]}',
            word_count=4,
        )

        self.assertEqual(parsed.groups, ((1, 3),))
        self.assertEqual(parsed.status, "warning")
        self.assertEqual(len(parsed.warnings), 5)

    def test_unparseable_output_becomes_an_error(self) -> None:
        parsed = parse_llm_response("There are no expressions.", word_count=4)

        self.assertEqual(parsed.groups, ())
        self.assertEqual(parsed.status, "error")


class LlmReportLabelTest(unittest.TestCase):
    def test_marks_adapter_backed_model_as_lora(self) -> None:
        self.assertEqual(
            _report_system_name(
                {"model_name": "Qwen/Qwen3.5-4B", "adapter_dir": "/tmp/adapter"}
            ),
            "Qwen/Qwen3.5-4B + LoRA",
        )

    def test_preserves_base_model_label_without_adapter(self) -> None:
        self.assertEqual(
            _report_system_name(
                {"model_name": "Qwen/Qwen3.5-4B", "adapter_dir": None}
            ),
            "Qwen/Qwen3.5-4B",
        )

    def test_full_development_report_is_not_labelled_as_balanced_sample(self) -> None:
        experiment = {
            "all_targets": True,
            "languages": ["FR", "JA"],
            "shots": 0,
            "positive_per_language": 4,
            "negative_per_language": 4,
        }

        protocol = _report_protocol_lines(
            experiment, request_count=321, is_calibration=False
        )
        interpretation = _report_interpretation_text(
            experiment, is_calibration=False
        )

        self.assertIn("zero-shot inference", protocol[0])
        self.assertIn("complete split for 2 configured languages", protocol[1])
        self.assertNotIn("positive", " ".join(protocol))
        self.assertIn("complete-development result", interpretation)
        self.assertNotIn("not representative", interpretation)


class MweConversionTest(unittest.TestCase):
    def test_reads_overlapping_and_discontinuous_gold_groups(self) -> None:
        value = sentence("1:VID", "2:NID", "1;2", "1")

        self.assertEqual(gold_mwe_groups(value), ((1, 3, 4), (2, 3)))

    def test_serializes_overlapping_and_discontinuous_groups(self) -> None:
        columns = mwe_groups_to_columns(5, [(1, 3, 5), (3, 4)])

        self.assertEqual(columns, ["1:VID", "*", "1;2:VID", "2", "1"])


class FixedSamplingTest(unittest.TestCase):
    def test_full_split_selection_preserves_source_order(self) -> None:
        sentences = [sentence("*", "*") for _ in range(3)]

        selected = all_sentence_indices(sentences, language="FR")

        self.assertEqual([item.sentence_index for item in selected], [0, 1, 2])
        self.assertEqual([item.polarity for item in selected], ["negative"] * 3)

    def test_balanced_sample_is_deterministic_and_ordered(self) -> None:
        sentences = [sentence("1:VID", "1") for _ in range(8)]
        sentences.extend(sentence("*", "*") for _ in range(8))

        first = sample_sentence_indices(
            sentences,
            language="FR",
            positive_count=3,
            negative_count=3,
            seed=13,
            max_words=10,
        )
        second = sample_sentence_indices(
            sentences,
            language="FR",
            positive_count=3,
            negative_count=3,
            seed=13,
            max_words=10,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [item.sentence_index for item in first],
            sorted(item.sentence_index for item in first),
        )
        self.assertEqual(sum(item.polarity == "positive" for item in first), 3)
        self.assertEqual(sum(item.polarity == "negative" for item in first), 3)

    def test_two_shot_examples_are_positive_then_negative(self) -> None:
        sentences = [sentence("1:VID", "1") for _ in range(4)]
        sentences.extend(sentence("*", "*") for _ in range(4))

        indices = select_demonstration_indices(
            sentences,
            language="FR",
            shots=2,
            seed=13,
            max_words=10,
        )

        self.assertTrue(gold_mwe_groups(sentences[indices[0]]))
        self.assertFalse(gold_mwe_groups(sentences[indices[1]]))

    def test_calibration_targets_can_be_excluded_from_demonstrations(self) -> None:
        sentences = [sentence("1:VID", "1") for _ in range(8)]
        sentences.extend(sentence("*", "*") for _ in range(8))
        calibration = sample_sentence_indices(
            sentences,
            language="FR",
            positive_count=2,
            negative_count=2,
            seed=13,
            max_words=10,
            purpose="train-calibration",
        )
        target_indices = [item.sentence_index for item in calibration]

        demonstrations = select_demonstration_indices(
            sentences,
            language="FR",
            shots=4,
            seed=13,
            max_words=10,
            excluded_indices=target_indices,
        )

        self.assertFalse(set(target_indices) & set(demonstrations))

    def test_demonstration_validation_rejects_calibration_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap train calibration"):
            _validate_demonstrations(
                language="FR",
                sentence_count=10,
                indices=[1, 4],
                shots=2,
                forbidden_indices={4, 7},
            )


if __name__ == "__main__":
    unittest.main()
