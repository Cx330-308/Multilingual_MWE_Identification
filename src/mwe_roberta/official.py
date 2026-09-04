from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OfficialMweScore:
    precision: float
    recall: float
    f1: float
    evaluated_languages: int
    requested_languages: int


def evaluator_paths(data_dir: Path | str) -> tuple[Path, Path]:
    tools_dir = Path(data_dir) / "tools"
    eval_script = tools_dir / "parseme_evaluate.py"
    average_script = tools_dir / "average_of_evaluations.py"
    missing = [str(path) for path in (eval_script, average_script) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official evaluator file(s): {', '.join(missing)}")
    return eval_script, average_script


def _stat_value(field: str) -> float:
    return float(field.rsplit("=", 1)[-1].split("(", 1)[0])


def parse_global_mwe_score(text: str) -> OfficialMweScore:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("* MWE-based:"):
            continue

        fields = stripped.split()
        values = {field.split("=", 1)[0]: field for field in fields if "=" in field}
        try:
            precision = _stat_value(values["P"])
            recall = _stat_value(values["R"])
            f1 = _stat_value(values["F"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Could not parse official MWE score line: {line}") from exc

        evaluated = requested = 1
        coverage = next((field for field in fields if field.startswith("@(")), None)
        if coverage is not None:
            numerator, denominator = coverage.removeprefix("@(").removesuffix(")").split("/", 1)
            evaluated = int(numerator)
            requested = int(denominator)
        return OfficialMweScore(precision, recall, f1, evaluated, requested)

    raise ValueError("Official evaluator output has no global MWE-based score.")


def run_language_evaluation(
    *,
    eval_script: Path,
    gold_path: Path,
    prediction_path: Path,
    output_path: Path,
    train_path: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(eval_script),
        "--gold",
        str(gold_path),
        "--pred",
        str(prediction_path),
    ]
    if train_path is not None and train_path.is_file():
        command.extend(["--train", str(train_path)])

    result = subprocess.run(command, check=False, text=True, capture_output=True)
    output_path.write_text(result.stdout, encoding="utf-8")
    stderr_path = output_path.with_name(f"{output_path.stem}.stderr.txt")
    if result.stderr:
        stderr_path.write_text(result.stderr, encoding="utf-8")
    elif stderr_path.exists():
        stderr_path.unlink()

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no evaluator output"
        raise RuntimeError(
            f"Official evaluator failed for {prediction_path} with exit code "
            f"{result.returncode}: {detail}"
        )
    return output_path


def average_evaluations(
    *,
    average_script: Path,
    evaluation_files: Sequence[Path],
    output_path: Path,
) -> OfficialMweScore:
    if not evaluation_files:
        raise ValueError("At least one evaluation file is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(average_script), *map(str, evaluation_files)],
        check=False,
        text=True,
        capture_output=True,
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    stderr_path = output_path.with_name(f"{output_path.stem}.stderr.txt")
    if result.stderr:
        stderr_path.write_text(result.stderr, encoding="utf-8")
    elif stderr_path.exists():
        stderr_path.unlink()

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no averaging output"
        raise RuntimeError(
            f"Official averaging script failed with exit code {result.returncode}: {detail}"
        )
    return parse_global_mwe_score(result.stdout)
