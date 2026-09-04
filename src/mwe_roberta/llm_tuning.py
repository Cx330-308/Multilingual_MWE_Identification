from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import llm_baseline
from .paths import default_artifacts_dir, default_data_dir


TUNING_PROTOCOL_VERSION = "qwen-train-calibration-v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    shots: int
    thinking: bool
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int


CANDIDATES = (
    Candidate("nt-greedy-0shot", 0, False, False, 0.7, 0.8, 20, 96),
    Candidate("nt-greedy-2shot", 2, False, False, 0.7, 0.8, 20, 96),
    Candidate("nt-greedy-4shot", 4, False, False, 0.7, 0.8, 20, 96),
    Candidate("nt-sample-4shot", 4, False, True, 0.7, 0.8, 20, 96),
    Candidate("think-sample-4shot", 4, True, True, 1.0, 0.95, 20, 256),
)


def add_args(parser: argparse.ArgumentParser) -> None:
    default_output = (
        default_artifacts_dir() / "experiments" / "llm-qwen3p5-4b-tuning-seed13"
    )
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--reference-prediction-dir")
    parser.add_argument("--positive-per-language", type=int, default=4)
    parser.add_argument("--negative-per-language", type=int, default=4)
    parser.add_argument("--dev-positive-per-language", type=int, default=4)
    parser.add_argument("--dev-negative-per-language", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dev-seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--max-demo-words", type=int, default=40)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--skip-dev-evaluation", action="store_true")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    plan = _tuning_plan(args)
    _write_matching_json(output_dir / "tuning_plan.json", plan)

    for candidate in CANDIDATES:
        print(f"\n=== Calibration candidate: {candidate.name} ===", flush=True)
        llm_baseline.run(
            _baseline_args(
                args,
                candidate,
                output_dir=output_dir / "candidates" / candidate.name,
                target_split="train",
                seed=args.seed,
                positive_per_language=args.positive_per_language,
                negative_per_language=args.negative_per_language,
                sample_purpose=f"{TUNING_PROTOCOL_VERSION}-targets",
                fixed_demonstrations=None,
                reference_prediction_dir=None,
            )
        )

    if args.prepare_only:
        print(f"Prepared all candidates under {output_dir}; inference was not started.")
        return

    summary = _summarize_candidates(output_dir, plan)
    winner = _candidate_by_name(summary["selection"]["candidate"])
    frozen_path = output_dir / "frozen_demonstrations.json"
    _write_json(frozen_path, summary["selection"]["demonstration_indices"])
    _write_json(output_dir / "tuning_summary.json", summary)
    _write_tuning_report(output_dir, summary)

    if args.skip_dev_evaluation:
        print(
            f"Selected {winner.name}; development evaluation was skipped as requested."
        )
        return

    print(f"\n=== Frozen development evaluation: {winner.name} ===", flush=True)
    dev_output = output_dir / f"dev-evaluation-seed{args.dev_seed}"
    llm_baseline.run(
        _baseline_args(
            args,
            winner,
            output_dir=dev_output,
            target_split="dev",
            seed=args.dev_seed,
            positive_per_language=args.dev_positive_per_language,
            negative_per_language=args.dev_negative_per_language,
            sample_purpose=f"{TUNING_PROTOCOL_VERSION}-frozen-dev",
            fixed_demonstrations=frozen_path,
            reference_prediction_dir=args.reference_prediction_dir,
        )
    )
    dev_metrics = _read_json(dev_output / "metrics.json")
    summary["frozen_development_evaluation"] = {
        "output_dir": str(dev_output),
        "seed": args.dev_seed,
        "sample_size": (
            len(plan["languages"])
            * (args.dev_positive_per_language + args.dev_negative_per_language)
        ),
        "metrics": dev_metrics["metrics"],
        "response_status": dev_metrics["response_status"],
        "truncated_prompts": dev_metrics["truncated_prompts"],
    }
    _write_json(output_dir / "tuning_summary.json", summary)
    _write_tuning_report(output_dir, summary)
    score = summary["frozen_development_evaluation"]["metrics"]["llm"]["average"][
        "mwe"
    ]
    print(
        f"Frozen dev subset macro: P={score['precision']:.4f} "
        f"R={score['recall']:.4f} F={score['f1']:.4f}"
    )
    print(f"Wrote {output_dir / 'tuning_report.md'}")


def _tuning_plan(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    languages = llm_baseline._training_languages(data_dir, args.languages)
    return {
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "objective": "official PARSEME exact MWE macro-F1 on train-only calibration",
        "tie_break": [
            "higher exact MWE precision",
            "higher token macro-F1",
            "earlier pre-registered candidate order",
        ],
        "data_dir": str(data_dir),
        "languages": languages,
        "model_name": args.model_name,
        "revision": args.revision,
        "seed": args.seed,
        "calibration_per_language": {
            "positive": args.positive_per_language,
            "negative": args.negative_per_language,
        },
        "leakage_control": (
            "Few-shot demonstrations are selected from training only after calibration "
            "targets are fixed, and target indices are excluded from the demo pool."
        ),
        "prompt_version": llm_baseline.PROMPT_VERSION,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "sampling_note": (
            "Sampling candidates use Qwen's published temperature/top-p/top-k values. "
            "Transformers generation does not expose Qwen's presence_penalty setting, "
            "so it is omitted consistently."
        ),
        "thinking_limit_note": (
            "The thinking candidate is capped at 256 new tokens for local hardware; "
            "this limit is part of the pre-registered candidate."
        ),
    }


def _baseline_args(
    args: argparse.Namespace,
    candidate: Candidate,
    *,
    output_dir: Path,
    target_split: str,
    seed: int,
    positive_per_language: int,
    negative_per_language: int,
    sample_purpose: str,
    fixed_demonstrations: Path | None,
    reference_prediction_dir: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=args.data_dir,
        languages=args.languages,
        model_name=args.model_name,
        revision=args.revision,
        output_dir=str(output_dir),
        reference_prediction_dir=reference_prediction_dir,
        adapter_dir=None,
        target_split=target_split,
        all_targets=False,
        sample_purpose=sample_purpose,
        positive_per_language=positive_per_language,
        negative_per_language=negative_per_language,
        shots=candidate.shots,
        demo_pool_shots=4 if fixed_demonstrations is None else None,
        fixed_demonstration_indices=(
            str(fixed_demonstrations) if fixed_demonstrations else None
        ),
        seed=seed,
        max_words=args.max_words,
        max_demo_words=args.max_demo_words,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=candidate.max_new_tokens,
        thinking=candidate.thinking,
        do_sample=candidate.do_sample,
        temperature=candidate.temperature,
        top_p=candidate.top_p,
        top_k=candidate.top_k,
        min_p=0.0,
        repetition_penalty=1.0,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
        prepare_only=args.prepare_only,
        retry_errors=args.retry_errors,
    )


def _summarize_candidates(output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    canonical_manifest: str | None = None
    for order, candidate in enumerate(CANDIDATES):
        candidate_dir = output_dir / "candidates" / candidate.name
        manifest_path = candidate_dir / "sample_manifest.jsonl"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        if canonical_manifest is None:
            canonical_manifest = manifest_text
        elif manifest_text != canonical_manifest:
            raise ValueError("Tuning candidates do not share the same calibration sample.")

        metrics = _read_json(candidate_dir / "metrics.json")
        experiment = metrics["experiment"]
        if experiment.get("target_split") != "train":
            raise ValueError(f"{candidate.name} was not evaluated on train calibration.")
        _validate_candidate_configuration(candidate, experiment)
        _validate_disjointness(manifest_text, experiment["demonstration_indices"])
        average = metrics["metrics"]["llm"]["average"]
        results.append(
            {
                "candidate": candidate.name,
                "order": order,
                "configuration": asdict(candidate),
                "mwe": average["mwe"],
                "token": average["token"],
                "response_status": metrics["response_status"],
                "truncated_prompts": metrics["truncated_prompts"],
                "demonstration_indices": experiment["demonstration_indices"],
                "output_dir": str(candidate_dir),
            }
        )

    ranked = sorted(
        results,
        key=lambda result: (
            -result["mwe"]["f1"],
            -result["mwe"]["precision"],
            -result["token"]["f1"],
            result["order"],
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    winner = ranked[0]
    return {
        "protocol": plan,
        "candidate_results": ranked,
        "selection": {
            "candidate": winner["candidate"],
            "criterion": plan["objective"],
            "mwe": winner["mwe"],
            "token": winner["token"],
            "demonstration_indices": winner["demonstration_indices"],
            "configuration": winner["configuration"],
        },
    }


def _validate_disjointness(
    manifest_text: str, demonstration_indices: dict[str, list[int]]
) -> None:
    targets: dict[str, set[int]] = {}
    for line in manifest_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        targets.setdefault(record["language"], set()).add(record["sentence_index"])
    for language, indices in demonstration_indices.items():
        overlap = targets.get(language, set()) & set(indices)
        if overlap:
            raise ValueError(
                f"Calibration leakage for {language}; overlapping indices: {sorted(overlap)}"
            )


def _validate_candidate_configuration(
    candidate: Candidate, experiment: dict[str, Any]
) -> None:
    expected = {
        "shots": candidate.shots,
        "thinking": candidate.thinking,
        "decoding": "sample" if candidate.do_sample else "greedy",
        "max_new_tokens": candidate.max_new_tokens,
    }
    mismatches = {
        key: (experiment.get(key), expected_value)
        for key, expected_value in expected.items()
        if experiment.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Recorded configuration for {candidate.name} does not match the plan: "
            f"{mismatches}"
        )


def _write_tuning_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Independent Qwen Configuration Tuning",
        "",
        "## Selection Protocol",
        "",
        "- Candidate selection used only a balanced subset of each language's training split.",
        "- Calibration targets and few-shot demonstrations were verified to be disjoint.",
        "- Primary objective: official exact MWE macro-F1.",
        "- The prompt was frozen before this search; development labels were not used to rank candidates.",
        "",
        "## Train-Only Calibration",
        "",
        "| Rank | Candidate | Shots | Thinking | Decoding | MWE P | MWE R | MWE F1 | Token F1 | Parse errors |",
        "| ---: | --- | ---: | :---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["candidate_results"]:
        config = result["configuration"]
        errors = result["response_status"]["error"]
        lines.append(
            f"| {result['rank']} | {result['candidate']} | {config['shots']} | "
            f"{'yes' if config['thinking'] else 'no'} | "
            f"{'sample' if config['do_sample'] else 'greedy'} | "
            f"{result['mwe']['precision']:.4f} | {result['mwe']['recall']:.4f} | "
            f"{result['mwe']['f1']:.4f} | {result['token']['f1']:.4f} | {errors} |"
        )

    selected = summary["selection"]
    lines.extend(
        [
            "",
            "## Frozen Selection",
            "",
            f"Selected `{selected['candidate']}` with train-calibration exact macro-F1 "
            f"{selected['mwe']['f1']:.4f}.",
        ]
    )
    dev = summary.get("frozen_development_evaluation")
    if dev:
        llm_scores = dev["metrics"]["llm"]["average"]
        lines.extend(
            [
                "",
                "## Frozen Development Evaluation",
                "",
                f"The selected configuration was evaluated on a separately sampled, balanced "
                f"development subset of {dev['sample_size']} sentences (seed {dev['seed']}).",
                "",
                "| System | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                _score_row("Tuned Qwen", llm_scores),
            ]
        )
        reference = dev["metrics"].get("reference")
        if reference:
            lines.append(_score_row("Final XLM-R (same subset)", reference["average"]))
        status = dev["response_status"]
        lines.extend(
            [
                "",
                f"Output parsing: {status['ok']} strict, {status['warning']} warning, "
                f"{status['error']} error; truncated prompts: {dev['truncated_prompts']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Calibration scores are selection statistics, not held-out results. The development evaluation remains a balanced subset diagnostic rather than a full-corpus estimate. The fixed prompt had been developed before this configuration search, including earlier development smoke checks, so only the configuration-selection step is strictly train-only.",
            "",
        ]
    )
    (output_dir / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")


def _score_row(name: str, scores: dict[str, Any]) -> str:
    mwe = scores["mwe"]
    token = scores["token"]
    return (
        f"| {name} | {mwe['precision']:.4f} | {mwe['recall']:.4f} | "
        f"{mwe['f1']:.4f} | {token['precision']:.4f} | {token['recall']:.4f} | "
        f"{token['f1']:.4f} |"
    )


def _candidate_by_name(name: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"Unknown tuning candidate: {name}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_matching_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing tuning plan does not match: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
