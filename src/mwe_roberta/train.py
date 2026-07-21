from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

from .cupt import available_split_languages, resolve_languages
from .data import load_examples
from .hf_dataset import TokenClassificationDataset, token_metrics
from .labels import build_label_list, collect_categories, load_labels, save_labels
from .paths import default_artifacts_dir, default_data_dir
from .cupt import iter_cupt, split_path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _discover_or_load_labels(args: argparse.Namespace, train_languages: list[str]) -> list[str]:
    if args.labels_json:
        return load_labels(args.labels_json)

    categories = set()
    for language in train_languages:
        for split in ("train", "dev"):
            path = split_path(args.data_dir, language, split)
            if path.exists():
                categories.update(collect_categories(iter_cupt(path)).keys())
    return build_label_list(categories)


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Run:\n"
            "  cd /path/to/mwe_roberta_pipeline\n"
            "  python -m pip install -r requirements.txt\n"
            "Then run the training command again."
        ) from exc

    _set_seed(args.seed)
    data_dir = Path(args.data_dir)
    languages = resolve_languages(data_dir, args.languages)
    train_languages = available_split_languages(data_dir, languages, "train")
    dev_languages = available_split_languages(data_dir, languages, "dev")
    supervised_languages = [lang for lang in languages if lang in train_languages and lang in dev_languages]
    if not supervised_languages:
        raise SystemExit("No languages with both train.cupt and dev.cupt were found.")

    skipped = [lang for lang in languages if lang not in supervised_languages]
    if skipped:
        print(f"Skipping languages without train/dev for fine-tuning: {', '.join(skipped)}")

    labels = _discover_or_load_labels(args, supervised_languages)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_labels(labels, output_dir / "labels.json")

    train_examples = load_examples(
        data_dir,
        supervised_languages,
        "train",
        limit_per_language=args.max_train_sentences_per_language,
        require_gold=True,
    )
    dev_examples = load_examples(
        data_dir,
        supervised_languages,
        "dev",
        limit_per_language=args.max_dev_sentences_per_language,
        require_gold=True,
    )
    print(f"Training sentences: {len(train_examples)}")
    print(f"Development sentences: {len(dev_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("This pipeline requires a fast Hugging Face tokenizer for word alignment.")

    train_dataset = TokenClassificationDataset(
        train_examples,
        tokenizer,
        label2id,
        max_length=args.max_length,
        stride=args.stride,
        label_all_tokens=args.label_all_tokens,
    )
    dev_dataset = TokenClassificationDataset(
        dev_examples,
        tokenizer,
        label2id,
        max_length=args.max_length,
        stride=args.stride,
        label_all_tokens=args.label_all_tokens,
    )
    print(f"Training chunks: {len(train_dataset)}")
    print(f"Development chunks: {len(dev_dataset)}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    use_cuda = torch.cuda.is_available()
    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": args.save_strategy,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        # Reload explicitly below. Transformers 5.x Trainer can mishandle
        # XLM-R LayerNorm key aliases when it reloads a checkpoint internally.
        "load_best_model_at_end": False,
        "metric_for_best_model": "mwe_f1",
        "greater_is_better": True,
        "fp16": args.fp16 and use_cuda,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "report_to": [],
        "seed": args.seed,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "dataloader_pin_memory" in training_signature.parameters:
        training_kwargs["dataloader_pin_memory"] = use_cuda
    warmup_ratio_parameter = training_signature.parameters.get("warmup_ratio")
    if (
        "warmup_steps" in training_signature.parameters
        and warmup_ratio_parameter is not None
        and warmup_ratio_parameter.default is None
    ):
        training_kwargs["warmup_steps"] = args.warmup_ratio
    else:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    if "eval_strategy" in training_signature.parameters:
        training_kwargs["eval_strategy"] = args.evaluation_strategy
    else:
        training_kwargs["evaluation_strategy"] = args.evaluation_strategy
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": token_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise SystemExit(
            "Unsupported transformers Trainer API: neither 'processing_class' nor "
            "'tokenizer' is accepted."
        )

    trainer = Trainer(
        **trainer_kwargs,
    )
    trainer.train()
    selected_checkpoint = None
    best_model_dir = output_dir / "best_model"
    if args.load_best_model_at_end:
        selected_checkpoint = trainer.state.best_model_checkpoint
        if selected_checkpoint is None:
            raise SystemExit(
                "No best checkpoint was recorded. Use matching evaluation/save strategies "
                "and make sure evaluation runs at least once."
            )
        print(f"Reloading best checkpoint: {selected_checkpoint}")
        selected_model = AutoModelForTokenClassification.from_pretrained(selected_checkpoint)
        selected_model.save_pretrained(best_model_dir)
    else:
        trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metadata = {
        "model_name": args.model_name,
        "languages_requested": languages,
        "supervised_languages": supervised_languages,
        "skipped_languages": skipped,
        "num_labels": len(labels),
        "max_length": args.max_length,
        "stride": args.stride,
        "label_all_tokens": args.label_all_tokens,
        "selected_checkpoint": selected_checkpoint,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved model to {output_dir / 'best_model'}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir() / "xlmr-all"))
    parser.add_argument("--labels-json", default=None, help="Optional labels.json from the prepare command.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--label-all-tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-sentences-per-language", type=int, default=None)
    parser.add_argument("--max-dev-sentences-per-language", type=int, default=None)
