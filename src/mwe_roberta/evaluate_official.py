from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .cupt import concatenate, resolve_languages, split_path
from .paths import default_artifacts_dir, default_data_dir


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = resolve_languages(data_dir, args.languages)

    eval_script = data_dir / "tools" / "parseme_evaluate.py"
    average_script = data_dir / "tools" / "average_of_evaluations.py"
    if not eval_script.exists():
        raise SystemExit(f"Missing official evaluator: {eval_script}")

    eval_files: list[Path] = []
    for language in languages:
        gold_path = split_path(data_dir, language, "dev")
        pred_path = prediction_dir / language / "dev.system.cupt"
        train_path = split_path(data_dir, language, "train")
        dev_path = split_path(data_dir, language, "dev")
        if not gold_path.exists() or not pred_path.exists():
            print(f"Skipping {language}: need {gold_path} and {pred_path}")
            continue

        train_dev_path = output_dir / "train-dev" / language / "train-dev.cupt"
        if train_path.exists() and dev_path.exists():
            concatenate([train_path, dev_path], train_dev_path)

        eval_out = output_dir / language / "eval.txt"
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(eval_script), "--gold", str(gold_path), "--pred", str(pred_path)]
        if train_dev_path.exists():
            command.extend(["--train", str(train_dev_path)])
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        eval_out.write_text(result.stdout, encoding="utf-8")
        if result.stderr:
            (eval_out.parent / "eval.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            print(f"{language}: official evaluator failed; see {eval_out}")
        else:
            print(f"{language}: wrote {eval_out}")
            eval_files.append(eval_out)

    if eval_files and average_script.exists():
        average_out = output_dir / "average.txt"
        result = subprocess.run(
            [sys.executable, str(average_script), *map(str, eval_files)],
            check=False,
            text=True,
            capture_output=True,
        )
        average_out.write_text(result.stdout + result.stderr, encoding="utf-8")
        print(f"Wrote {average_out}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--prediction-dir", default=str(default_artifacts_dir() / "predictions"))
    parser.add_argument("--output-dir", default=str(default_artifacts_dir() / "evaluation"))
