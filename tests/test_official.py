from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mwe_roberta.evaluate_official import run
from mwe_roberta.official import OfficialMweScore, parse_global_mwe_score


class OfficialScoreParsingTest(unittest.TestCase):
    def test_parses_macro_average_with_standard_deviation(self) -> None:
        output = """
## Global evaluation
* MWE-based: P=0.6108(+-0.2187) R=0.5745(+-0.2060) F=0.5921  @(16/16)
* Tok-based: P=0.7449(+-0.1378) R=0.6246(+-0.1713) F=0.6795  @(16/16)
"""
        score = parse_global_mwe_score(output)

        self.assertEqual(
            score,
            OfficialMweScore(
                precision=0.6108,
                recall=0.5745,
                f1=0.5921,
                evaluated_languages=16,
                requested_languages=16,
            ),
        )

    def test_parses_single_language_fraction_format(self) -> None:
        output = """
## Global evaluation
* MWE-based: P=12/20=0.6000 R=12/24=0.5000 F=0.5455
"""
        score = parse_global_mwe_score(output)

        self.assertEqual(score.precision, 0.6)
        self.assertEqual(score.recall, 0.5)
        self.assertEqual(score.f1, 0.5455)
        self.assertEqual(score.evaluated_languages, 1)


class StandaloneEvaluationTest(unittest.TestCase):
    def test_seen_unseen_reference_uses_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            language_dir = data_dir / "FR"
            tools_dir = data_dir / "tools"
            prediction_dir = root / "predictions"
            language_dir.mkdir(parents=True)
            tools_dir.mkdir(parents=True)
            (prediction_dir / "FR").mkdir(parents=True)

            train_path = language_dir / "train.cupt"
            dev_path = language_dir / "dev.cupt"
            prediction_path = prediction_dir / "FR" / "dev.system.cupt"
            train_path.write_text("train\n", encoding="utf-8")
            dev_path.write_text("dev\n", encoding="utf-8")
            prediction_path.write_text("prediction\n", encoding="utf-8")
            (tools_dir / "parseme_evaluate.py").write_text("", encoding="utf-8")
            (tools_dir / "average_of_evaluations.py").write_text("", encoding="utf-8")

            args = argparse.Namespace(
                data_dir=str(data_dir),
                languages="FR",
                prediction_dir=str(prediction_dir),
                output_dir=str(root / "evaluation"),
            )
            average_score = OfficialMweScore(0.5, 0.5, 0.5, 1, 1)

            with (
                patch(
                    "mwe_roberta.evaluate_official.run_language_evaluation",
                    return_value=root / "evaluation" / "FR" / "eval.txt",
                ) as language_eval,
                patch(
                    "mwe_roberta.evaluate_official.average_evaluations",
                    return_value=average_score,
                ),
            ):
                run(args)

            self.assertEqual(language_eval.call_args.kwargs["train_path"], train_path)


if __name__ == "__main__":
    unittest.main()
