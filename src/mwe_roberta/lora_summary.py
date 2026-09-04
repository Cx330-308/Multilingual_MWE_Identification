from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN_PREFIX = (
    "lora-qwen3p5-4b-n500pl-ml640-sa0p5-r8-a16-d0p05-lr2em4-ep1-seed"
)
METRIC_NAMES = ("precision", "recall", "f1")
SCORE_TYPES = ("mwe", "token")


def sample_statistics(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if len(values) < 2:
        raise ValueError("At least two values are required for a sample standard deviation.")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    response_status = payload["response_status"]
    return {
        "sentences": sum(response_status.values()),
        "metrics": payload["metrics"]["llm"]["average"],
        "response_status": response_status,
        "truncated_prompts": payload["truncated_prompts"],
    }


def _collect_run(
    repo_root: Path,
    *,
    seed: int,
    run_prefix: str,
    expected_training_subset_seed: int,
    expected_matched_sentences: int,
    expected_full_sentences: int,
) -> dict[str, Any]:
    relative_dir = Path("artifacts/experiments") / f"{run_prefix}{seed}"
    run_dir = repo_root / relative_dir
    config = _read_json(run_dir / "experiment_config.json")
    matched = _evaluation_summary(
        _read_json(run_dir / "dev-subset-seed42" / "metrics.json")
    )
    full = _evaluation_summary(_read_json(run_dir / "dev-full" / "metrics.json"))

    if config["status"] != "completed":
        raise ValueError(f"Seed {seed} is not complete: {config['status']!r}.")
    if config["arguments"]["seed"] != seed:
        raise ValueError(
            f"Seed {seed} directory records optimization seed "
            f"{config['arguments']['seed']}."
        )
    subset_seed = config["arguments"].get("training_subset_seed")
    if subset_seed is None:
        subset_seed = seed
    if subset_seed != expected_training_subset_seed:
        raise ValueError(
            f"Seed {seed} used training-subset seed {subset_seed}, expected "
            f"{expected_training_subset_seed}."
        )
    if matched["sentences"] != expected_matched_sentences:
        raise ValueError(
            f"Seed {seed} matched evaluation has {matched['sentences']} sentences, "
            f"expected {expected_matched_sentences}."
        )
    if full["sentences"] != expected_full_sentences:
        raise ValueError(
            f"Seed {seed} full evaluation has {full['sentences']} sentences, "
            f"expected {expected_full_sentences}."
        )

    result = config["result"]
    return {
        "seed": seed,
        "experiment_dir": relative_dir.as_posix(),
        "training": {
            "training_subset_seed": subset_seed,
            "selected_sentences": config["data"]["selected_sentences"],
            "encoded_sentences": config["data"]["encoded_sentences"],
            "total_encoded_sentences": config["data"]["total_encoded_sentences"],
            "global_step": result["global_step"],
            "completed_epochs": result["completed_epochs"],
            "train_metrics": result["train_metrics"],
        },
        "matched_subset": matched,
        "complete_development": full,
        "per_language_complete_development": _read_json(
            run_dir / "dev-full" / "metrics.json"
        )["metrics"]["llm"]["per_language"],
    }


def _aggregate_scores(
    runs: list[dict[str, Any]], evaluation_key: str
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        score_type: {
            metric: sample_statistics(
                run[evaluation_key]["metrics"][score_type][metric] for run in runs
            )
            for metric in METRIC_NAMES
        }
        for score_type in SCORE_TYPES
    }


def _aggregate_per_language(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    languages = sorted(runs[0]["per_language_complete_development"])
    return {
        language: {
            score_type: {
                metric: sample_statistics(
                    run["per_language_complete_development"][language][score_type][
                        metric
                    ]
                    for run in runs
                )
                for metric in METRIC_NAMES
            }
            for score_type in SCORE_TYPES
        }
        for language in languages
    }


def summarize(
    repo_root: Path,
    *,
    seeds: tuple[int, ...] = (13, 42, 2026),
    run_prefix: str = DEFAULT_RUN_PREFIX,
    expected_training_subset_seed: int = 13,
    expected_matched_sentences: int = 128,
    expected_full_sentences: int = 20_163,
) -> dict[str, Any]:
    runs = [
        _collect_run(
            repo_root,
            seed=seed,
            run_prefix=run_prefix,
            expected_training_subset_seed=expected_training_subset_seed,
            expected_matched_sentences=expected_matched_sentences,
            expected_full_sentences=expected_full_sentences,
        )
        for seed in seeds
    ]

    selected = [run["training"]["selected_sentences"] for run in runs]
    encoded = [run["training"]["encoded_sentences"] for run in runs]
    if any(value != selected[0] for value in selected[1:]):
        raise ValueError("Confirmation runs used different selected training samples.")
    if any(value != encoded[0] for value in encoded[1:]):
        raise ValueError("Confirmation runs used different encoded training samples.")

    return {
        "schema_version": 1,
        "protocol": {
            "optimization_seeds": list(seeds),
            "training_subset_seed": expected_training_subset_seed,
            "training_composition_identical": True,
            "matched_subset_sentences": expected_matched_sentences,
            "complete_development_sentences": expected_full_sentences,
            "standard_deviation": "sample (n-1)",
        },
        "runs": runs,
        "aggregate": {
            "training": {
                "train_loss": sample_statistics(
                    run["training"]["train_metrics"]["train_loss"] for run in runs
                ),
                "total_encoded_sentences": runs[0]["training"][
                    "total_encoded_sentences"
                ],
            },
            "matched_subset": _aggregate_scores(runs, "matched_subset"),
            "complete_development": _aggregate_scores(
                runs, "complete_development"
            ),
            "per_language_complete_development": _aggregate_per_language(runs),
        },
    }


def _score_cells(metrics: dict[str, Any], score_type: str) -> list[str]:
    return [f"{metrics[score_type][metric]:.4f}" for metric in METRIC_NAMES]


def _aggregate_cells(metrics: dict[str, Any], score_type: str) -> list[str]:
    return [
        f"{metrics[score_type][metric]['mean']:.4f} +/- "
        f"{metrics[score_type][metric]['sample_standard_deviation']:.4f}"
        for metric in METRIC_NAMES
    ]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen LoRA Confirmation Summary",
        "",
        "All runs use training-subset seed 13. Only adapter initialisation, "
        "training order, and weighted-sampler randomness vary with the "
        "optimization seed. Standard deviations are sample standard "
        "deviations (n-1).",
        "",
        "## Complete Development Set",
        "",
        "| Seed | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 | "
        "Strict | Errors | Warnings |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        status = run["complete_development"]["response_status"]
        cells = _score_cells(run["complete_development"]["metrics"], "mwe")
        cells += _score_cells(run["complete_development"]["metrics"], "token")
        lines.append(
            f"| {run['seed']} | {' | '.join(cells)} | {status['ok']} | "
            f"{status['error']} | {status['warning']} |"
        )
    aggregate = summary["aggregate"]["complete_development"]
    cells = _aggregate_cells(aggregate, "mwe") + _aggregate_cells(aggregate, "token")
    lines.extend(
        [
            f"| Mean +/- SD | {' | '.join(cells)} | - | - | - |",
            "",
            "## Matched 128-Sentence Subset",
            "",
            "| Seed | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in summary["runs"]:
        cells = _score_cells(run["matched_subset"]["metrics"], "mwe")
        cells += _score_cells(run["matched_subset"]["metrics"], "token")
        lines.append(f"| {run['seed']} | {' | '.join(cells)} |")
    aggregate = summary["aggregate"]["matched_subset"]
    cells = _aggregate_cells(aggregate, "mwe") + _aggregate_cells(aggregate, "token")
    lines.extend(
        [
            f"| Mean +/- SD | {' | '.join(cells)} |",
            "",
            "## Complete-Development MWE Scores by Language",
            "",
            "| Language | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    per_language = summary["aggregate"]["per_language_complete_development"]
    for language, values in per_language.items():
        cells = _aggregate_cells(values, "mwe")
        lines.append(f"| {language} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the three Qwen LoRA confirmation runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (
        repo_root / "artifacts/analysis/qwen-lora-confirmations"
    )
    summary = summarize(repo_root, seeds=tuple(args.seeds))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
