from __future__ import annotations

import argparse
from pathlib import Path

from .cupt import resolve_languages, split_path
from .official import average_evaluations, evaluator_paths, run_language_evaluation
from .paths import default_artifacts_dir, default_data_dir


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = resolve_languages(data_dir, args.languages)

    try:
        eval_script, average_script = evaluator_paths(data_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    eval_files: list[Path] = []
    for language in languages:
        gold_path = split_path(data_dir, language, "dev")
        pred_path = prediction_dir / language / "dev.system.cupt"
        train_path = split_path(data_dir, language, "train")
        if not gold_path.exists() or not pred_path.exists():
            print(f"Skipping {language}: need {gold_path} and {pred_path}")
            continue

        eval_out = output_dir / language / "eval.txt"
        try:
            run_language_evaluation(
                eval_script=eval_script,
                gold_path=gold_path,
                prediction_path=pred_path,
                train_path=train_path,
                output_path=eval_out,
            )
            print(f"{language}: wrote {eval_out}")
            eval_files.append(eval_out)
        except RuntimeError as exc:
            print(f"{language}: {exc}")

    if eval_files:
        average_out = output_dir / "average.txt"
        score = average_evaluations(
            average_script=average_script,
            evaluation_files=eval_files,
            output_path=average_out,
        )
        print(
            f"Wrote {average_out}: P={score.precision:.4f} "
            f"R={score.recall:.4f} F={score.f1:.4f}"
        )


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--prediction-dir", default=str(default_artifacts_dir() / "predictions"))
    parser.add_argument("--output-dir", default=str(default_artifacts_dir() / "evaluation"))
