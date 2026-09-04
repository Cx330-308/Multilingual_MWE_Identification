from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from mwe_roberta.cli import build_parser


CUPT_HEADER = (
    "# global.columns = ID FORM LEMMA UPOS XPOS FEATS "
    "HEAD DEPREL DEPS MISC PARSEME:MWE"
)


def _sentence(index: int, first: str, second: str, mwe: bool) -> str:
    first_mwe = "1:VID" if mwe else "*"
    second_mwe = "1" if mwe else "*"
    return "\n".join(
        [
            f"# source_sent_id = smoke {index}",
            f"# text = {first} {second}",
            f"1\t{first}\t{first}\tVERB\t_\t_\t0\troot\t_\t_\t{first_mwe}",
            f"2\t{second}\t{second}\tNOUN\t_\t_\t1\tobj\t_\t_\t{second_mwe}",
        ]
    )


class E3TrainingSmokeTest(unittest.TestCase):
    def test_weighted_loss_and_language_sampler_train_one_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            language_dir = data_dir / "FR"
            language_dir.mkdir(parents=True)

            train_sentences = [
                _sentence(1, "take", "place", True),
                _sentence(2, "work", "today", False),
                _sentence(3, "make", "decision", True),
                _sentence(4, "hello", "world", False),
            ]
            dev_sentences = [_sentence(5, "take", "decision", True)]
            language_dir.joinpath("train.cupt").write_text(
                CUPT_HEADER + "\n" + "\n\n".join(train_sentences) + "\n\n",
                encoding="utf-8",
            )
            language_dir.joinpath("dev.cupt").write_text(
                CUPT_HEADER + "\n" + "\n\n".join(dev_sentences) + "\n\n",
                encoding="utf-8",
            )

            model_dir = root / "tiny-model"
            vocabulary = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "take": 5,
                "place": 6,
                "work": 7,
                "today": 8,
                "make": 9,
                "decision": 10,
                "hello": 11,
                "world": 12,
            }
            tokenizer = BertTokenizerFast(vocab=vocabulary)
            tokenizer.save_pretrained(model_dir)
            model = BertForTokenClassification(
                BertConfig(
                    vocab_size=len(vocabulary),
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                    max_position_embeddings=64,
                    pad_token_id=0,
                    num_labels=3,
                )
            )
            model.half()
            model.save_pretrained(model_dir)

            output_dir = root / "output"
            args = build_parser().parse_args(
                [
                    "train",
                    "--data-dir",
                    str(data_dir),
                    "--languages",
                    "FR",
                    "--model-name",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "0.25",
                    "--batch-size",
                    "2",
                    "--eval-batch-size",
                    "1",
                    "--max-length",
                    "32",
                    "--stride",
                    "8",
                    "--logging-steps",
                    "1",
                    "--evaluation-strategy",
                    "no",
                    "--save-strategy",
                    "no",
                    "--language-sampling-alpha",
                    "0.5",
                    "--class-weight-alpha",
                    "0.25",
                    "--class-weight-min",
                    "0.75",
                    "--class-weight-max",
                    "2.5",
                    "--experiment-name",
                    "e3-smoke",
                ]
            )

            args.func(args)

            experiment = json.loads(
                output_dir.joinpath("experiment_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(experiment["status"], "completed")
            self.assertEqual(
                experiment["environment"]["model_parameter_dtypes"],
                ["torch.float32"],
            )
            self.assertEqual(experiment["sampling"]["strategy"], "language_power")
            self.assertEqual(experiment["loss"]["strategy"], "class_power")
            self.assertIn("train_loss", experiment["result"]["train_metrics"])
            self.assertGreater(experiment["result"]["train_metrics"]["epoch"], 0.0)
            self.assertTrue(output_dir.joinpath("best_model", "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
