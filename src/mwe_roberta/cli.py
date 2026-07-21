from __future__ import annotations

import argparse

from . import evaluate_official, predict, prepare, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mwe-roberta",
        description="XLM-RoBERTa fine-tuning pipeline for PARSEME 2.0 Subtask 1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Inspect data and build labels.json.")
    prepare.add_args(prepare_parser)
    prepare_parser.set_defaults(func=prepare.run)

    train_parser = subparsers.add_parser("train", help="Fine-tune XLM-RoBERTa for MWE token classification.")
    train.add_args(train_parser)
    train_parser.set_defaults(func=train.run)

    predict_parser = subparsers.add_parser("predict", help="Predict PARSEME:MWE columns and write CUPT files.")
    predict.add_args(predict_parser)
    predict_parser.set_defaults(func=predict.run)

    eval_parser = subparsers.add_parser("evaluate", help="Run the official PARSEME evaluator on dev predictions.")
    evaluate_official.add_args(eval_parser)
    eval_parser.set_defaults(func=evaluate_official.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
