from __future__ import annotations

import argparse

from . import (
    analysis,
    benchmark_analysis,
    evaluate_official,
    llm_baseline,
    llm_lora,
    llm_tuning,
    predict,
    prepare,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mwe-roberta",
        description="Transformer fine-tuning pipeline for PARSEME 2.0 Subtask 1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Inspect data and build labels.json.")
    prepare.add_args(prepare_parser)
    prepare_parser.set_defaults(func=prepare.run)

    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune a Hugging Face encoder for MWE token classification.",
    )
    train.add_args(train_parser)
    train_parser.set_defaults(func=train.run)

    predict_parser = subparsers.add_parser("predict", help="Predict PARSEME:MWE columns and write CUPT files.")
    predict.add_args(predict_parser)
    predict_parser.set_defaults(func=predict.run)

    eval_parser = subparsers.add_parser("evaluate", help="Run the official PARSEME evaluator on dev predictions.")
    evaluate_official.add_args(eval_parser)
    eval_parser.set_defaults(func=evaluate_official.run)

    analysis_parser = subparsers.add_parser(
        "analyze",
        help="Aggregate final per-language scores and official error breakdowns.",
    )
    analysis.add_args(analysis_parser)
    analysis_parser.set_defaults(func=analysis.run)

    benchmark_parser = subparsers.add_parser(
        "analyze-benchmark",
        help="Parse Codabench blind-test details and compare them with development scores.",
    )
    benchmark_analysis.add_args(benchmark_parser)
    benchmark_parser.set_defaults(func=benchmark_analysis.run)

    llm_parser = subparsers.add_parser(
        "llm-baseline",
        help="Run a reproducible prompted-LLM comparison on a fixed dev subset.",
    )
    llm_baseline.add_args(llm_parser)
    llm_parser.set_defaults(func=llm_baseline.run)

    llm_tuning_parser = subparsers.add_parser(
        "llm-tune",
        help="Tune prompted-Qwen settings on train-only calibration data.",
    )
    llm_tuning.add_args(llm_tuning_parser)
    llm_tuning_parser.set_defaults(func=llm_tuning.run)

    llm_lora_parser = subparsers.add_parser(
        "llm-lora-train",
        help="Fine-tune Qwen for generative MWE identification with PEFT LoRA.",
    )
    llm_lora.add_args(llm_lora_parser)
    llm_lora_parser.set_defaults(func=llm_lora.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
