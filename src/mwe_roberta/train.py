from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from .class_weights import build_class_weight_plan, weighted_token_cross_entropy
from .cupt import available_split_languages, iter_cupt, resolve_languages, split_path
from .data import load_examples
from .hf_dataset import TokenClassificationDataset, token_metrics
from .labels import build_label_list, collect_categories, load_labels, save_labels
from .paths import default_artifacts_dir, default_data_dir
from .sampling import build_language_sampling_plan, create_weighted_sampler
from .training_metrics import OfficialMetricsComputer


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_snapshot(path: Path) -> dict[str, Any] | None:
    root_result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        capture_output=True,
    )
    if root_result.returncode != 0:
        return None

    root = Path(root_result.stdout.strip())
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "root": str(root.resolve()),
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "tracked_files_modified": bool(status_result.stdout.strip())
        if status_result.returncode == 0
        else None,
    }


def _device_summary(torch: Any) -> dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
        }
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"type": "mps", "name": "Apple Metal Performance Shaders"}
    return {"type": "cpu", "name": platform.processor() or platform.machine()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for name, value in vars(args).items():
        if callable(value):
            continue
        serializable[name] = str(value) if isinstance(value, Path) else value
    return serializable


def _float32_load_kwargs(torch_module: Any, transformers_version: str) -> dict[str, Any]:
    try:
        major_version = int(transformers_version.split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        major_version = 5
    argument_name = "dtype" if major_version >= 5 else "torch_dtype"
    return {argument_name: torch_module.float32}


def _validate_training_controls(args: argparse.Namespace) -> None:
    if not math.isfinite(args.epochs) or args.epochs <= 0.0:
        raise SystemExit("--epochs must be finite and greater than zero.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be greater than zero.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be finite and greater than zero.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be finite and zero or greater.")
    if not math.isfinite(args.warmup_ratio) or not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be finite and in the range [0.0, 1.0).")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be greater than zero.")
    if args.stride < 0 or args.stride >= args.max_length:
        raise SystemExit("--stride must be zero or greater and smaller than --max-length.")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be greater than zero.")
    if not math.isfinite(args.language_sampling_alpha):
        raise SystemExit("--language-sampling-alpha must be finite.")
    if not 0.0 <= args.language_sampling_alpha <= 1.0:
        raise SystemExit("--language-sampling-alpha must be between 0.0 and 1.0.")
    if not math.isfinite(args.class_weight_alpha):
        raise SystemExit("--class-weight-alpha must be finite.")
    if not 0.0 <= args.class_weight_alpha <= 1.0:
        raise SystemExit("--class-weight-alpha must be between 0.0 and 1.0.")
    if not math.isfinite(args.class_weight_min) or args.class_weight_min <= 0.0:
        raise SystemExit("--class-weight-min must be finite and greater than zero.")
    if (
        not math.isfinite(args.class_weight_max)
        or args.class_weight_max < args.class_weight_min
    ):
        raise SystemExit(
            "--class-weight-max must be finite and at least --class-weight-min."
        )
    if args.selection_metric == "official" and args.evaluation_strategy == "no":
        raise SystemExit("Official checkpoint selection requires evaluation to be enabled.")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience must be zero or greater.")
    if args.early_stopping_patience and not args.load_best_model_at_end:
        raise SystemExit(
            "Early stopping requires --load-best-model-at-end so the selected checkpoint "
            "is exported instead of the last model."
        )
    if args.load_best_model_at_end:
        if args.save_strategy == "no":
            raise SystemExit("--load-best-model-at-end requires checkpoint saving.")
        if args.evaluation_strategy != args.save_strategy:
            raise SystemExit(
                "--evaluation-strategy and --save-strategy must match when selecting "
                "the best checkpoint."
            )
        if args.evaluation_strategy == "steps" and args.eval_steps != args.save_steps:
            raise SystemExit(
                "--eval-steps and --save-steps must match when selecting the best checkpoint."
            )


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Run:\n"
            "  cd /path/to/mwe_roberta_pipeline\n"
            "  python -m pip install -r requirements.txt\n"
            "Then run the training command again."
        ) from exc

    _validate_training_controls(args)
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
        include_sentence=args.selection_metric == "official",
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
        track_languages=True,
    )
    dev_dataset = TokenClassificationDataset(
        dev_examples,
        tokenizer,
        label2id,
        max_length=args.max_length,
        stride=args.stride,
        label_all_tokens=args.label_all_tokens,
        track_metadata=args.selection_metric == "official",
    )
    print(f"Training chunks: {len(train_dataset)}")
    print(f"Development chunks: {len(dev_dataset)}")

    sampling_plan = build_language_sampling_plan(
        train_dataset.chunk_languages,
        args.language_sampling_alpha,
    )
    if sampling_plan.uses_weighted_sampling:
        expected_distribution = ", ".join(
            f"{language}={probability:.1%}"
            for language, probability in sampling_plan.expected_probabilities.items()
        )
        print(
            "Language-balanced sampling enabled: "
            f"alpha={sampling_plan.alpha:g}, replacement=True"
        )
        print(f"Expected language distribution per epoch: {expected_distribution}")
    else:
        print("Language sampling: natural dataset distribution (alpha=1).")

    class_weight_plan = build_class_weight_plan(
        train_dataset.chunks,
        labels,
        alpha=args.class_weight_alpha,
        min_weight=args.class_weight_min,
        max_weight=args.class_weight_max,
        chunk_sampling_weights=(
            sampling_plan.chunk_weights
            if sampling_plan.uses_weighted_sampling
            else None
        ),
    )
    if class_weight_plan.uses_weighted_loss:
        o_label_id = label2id["O"]
        print(
            "Class-weighted cross-entropy enabled: "
            f"alpha={class_weight_plan.alpha:g}, "
            f"clip=[{class_weight_plan.min_weight:g}, "
            f"{class_weight_plan.max_weight:g}], "
            f"O weight={class_weight_plan.weights[o_label_id]:.4f}"
        )
        absent_labels = class_weight_plan.summary()["absent_labels"]
        if absent_labels:
            print(
                "Labels absent from training data: "
                f"{', '.join(str(label) for label in absent_labels)}"
            )
    else:
        print("Loss: standard cross-entropy (class-weight alpha=0).")

    float32_load_kwargs = _float32_load_kwargs(torch, transformers.__version__)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        **float32_load_kwargs,
    )
    model_parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    if model_parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(
            "Training requires float32 model parameters, but loaded: "
            f"{', '.join(model_parameter_dtypes) or 'none'}"
        )
    print("Model parameter dtype: float32")

    use_cuda = torch.cuda.is_available()
    metric_for_best_model = (
        "official_mwe_macro_f1" if args.selection_metric == "official" else "mwe_f1"
    )
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
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": True,
        "fp16": args.fp16 and use_cuda,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "report_to": [],
        "seed": args.seed,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "logging_nan_inf_filter" in training_signature.parameters:
        training_kwargs["logging_nan_inf_filter"] = False
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

    official_metrics = None
    compute_metrics = token_metrics
    if args.selection_metric == "official":
        official_metrics = OfficialMetricsComputer(
            data_dir=data_dir,
            output_dir=output_dir / "official-evaluations",
            languages=supervised_languages,
            examples=dev_examples,
            dataset=dev_dataset,
            id2label=id2label,
        )
        compute_metrics = official_metrics

    callbacks = []
    early_stopping_callback = None
    if args.early_stopping_patience:
        metric_name = f"eval_{metric_for_best_model}"

        class MetricEarlyStoppingCallback(TrainerCallback):
            def __init__(self) -> None:
                self.best_metric: float | None = None
                self.patience_counter = 0

            def on_evaluate(self, training_args, state, control, metrics=None, **kwargs):
                available_metrics = metrics or {}
                non_finite_metrics = sorted(
                    name
                    for name, value in available_metrics.items()
                    if isinstance(value, Real) and not math.isfinite(float(value))
                )
                if non_finite_metrics:
                    raise FloatingPointError(
                        "Evaluation produced non-finite metrics: "
                        f"{', '.join(non_finite_metrics)}"
                    )
                metric_value = available_metrics.get(metric_name)
                if metric_value is None:
                    raise RuntimeError(
                        f"Early stopping metric {metric_name!r} is missing. "
                        f"Available metrics: {sorted(available_metrics)}"
                    )

                improved = self.best_metric is None
                if self.best_metric is not None:
                    improvement = (
                        metric_value - self.best_metric
                        if training_args.greater_is_better
                        else self.best_metric - metric_value
                    )
                    improved = improvement > args.early_stopping_threshold

                if improved:
                    self.best_metric = float(metric_value)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(
                        f"Early stopping: {metric_name} did not improve "
                        f"({self.patience_counter}/{args.early_stopping_patience})."
                    )
                    if self.patience_counter >= args.early_stopping_patience:
                        control.should_training_stop = True
                return control

        early_stopping_callback = MetricEarlyStoppingCallback()
        callbacks.append(early_stopping_callback)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
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

    class FiniteLossTrainer(Trainer):
        def training_step(self, model, inputs, *step_args, **step_kwargs):
            loss = super().training_step(model, inputs, *step_args, **step_kwargs)
            check_interval = max(1, self.args.logging_steps)
            if (self.state.global_step + 1) % check_interval == 0:
                if not bool(torch.isfinite(loss).all().item()):
                    raise FloatingPointError(
                        "Training produced a non-finite loss near global step "
                        f"{self.state.global_step + 1}. The run has been stopped "
                        "before it can produce invalid checkpoints."
                    )
            return loss

    trainer_class = FiniteLossTrainer
    if sampling_plan.uses_weighted_sampling:
        base_trainer_class = trainer_class

        class LanguageBalancedTrainer(base_trainer_class):
            def _get_train_sampler(self, train_dataset=None):
                dataset = train_dataset if train_dataset is not None else self.train_dataset
                if dataset is None:
                    return None
                return create_weighted_sampler(
                    sampling_plan,
                    dataset_length=len(dataset),
                    seed=args.seed,
                )

        trainer_class = LanguageBalancedTrainer

    if class_weight_plan.uses_weighted_loss:
        base_trainer_class = trainer_class

        class ClassWeightedTrainer(base_trainer_class):
            def __init__(self, *trainer_args, **trainer_kwargs):
                super().__init__(*trainer_args, **trainer_kwargs)
                self.model_accepts_loss_kwargs = False
                self._class_weights = torch.tensor(
                    class_weight_plan.weights,
                    dtype=torch.float32,
                )

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                model_inputs = dict(inputs)
                targets = model_inputs.pop("labels", None)
                if targets is None:
                    raise RuntimeError("Class-weighted loss requires token labels.")
                outputs = model(**model_inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs[0]
                loss = weighted_token_cross_entropy(
                    logits,
                    targets,
                    self._class_weights,
                )
                return (loss, outputs) if return_outputs else loss

        trainer_class = ClassWeightedTrainer

    trainer = trainer_class(**trainer_kwargs)

    project_dir = Path(__file__).resolve().parents[2]
    experiment_config_path = output_dir / "experiment_config.json"
    experiment_config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_name": args.experiment_name,
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "arguments": _serializable_arguments(args),
        "selection": {
            "metric": metric_for_best_model,
            "official_evaluation_during_training": args.selection_metric == "official",
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
        },
        "sampling": sampling_plan.summary(),
        "loss": class_weight_plan.summary(),
        "data": {
            "directory": str(data_dir.resolve()),
            "languages_requested": languages,
            "supervised_languages": supervised_languages,
            "skipped_languages": skipped,
            "training_sentences": len(train_examples),
            "development_sentences": len(dev_examples),
            "training_chunks": len(train_dataset),
            "development_chunks": len(dev_dataset),
            "git": _git_snapshot(data_dir),
        },
        "source": {
            "directory": str(project_dir),
            "git": _git_snapshot(project_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": _device_summary(torch),
            "model_parameter_dtypes": model_parameter_dtypes,
        },
    }
    _write_json(experiment_config_path, experiment_config)

    selected_checkpoint = None
    best_model_dir = output_dir / "best_model"
    try:
        train_result = trainer.train()
        if args.load_best_model_at_end:
            selected_checkpoint = trainer.state.best_model_checkpoint
            if selected_checkpoint is None and trainer.state.best_global_step is not None:
                fallback = output_dir / f"checkpoint-{trainer.state.best_global_step}"
                if fallback.is_dir():
                    selected_checkpoint = str(fallback)
            if selected_checkpoint is None:
                raise RuntimeError(
                    "No best checkpoint was recorded. Make sure evaluation and checkpoint "
                    "saving run at least once."
                )
            print(f"Reloading best checkpoint: {selected_checkpoint}")
            selected_model = AutoModelForTokenClassification.from_pretrained(
                selected_checkpoint,
                **float32_load_kwargs,
            )
            selected_model.save_pretrained(best_model_dir)
        else:
            trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    except Exception as exc:
        experiment_config.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(experiment_config_path, experiment_config)
        raise

    metadata = {
        "model_name": args.model_name,
        "languages_requested": languages,
        "supervised_languages": supervised_languages,
        "skipped_languages": skipped,
        "num_labels": len(labels),
        "max_length": args.max_length,
        "stride": args.stride,
        "label_all_tokens": args.label_all_tokens,
        "sampling": sampling_plan.summary(),
        "loss": class_weight_plan.summary(),
        "selection_metric": metric_for_best_model,
        "best_metric": trainer.state.best_metric,
        "best_global_step": trainer.state.best_global_step,
        "selected_checkpoint": selected_checkpoint,
        "completed_epochs": trainer.state.epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "official_evaluations": official_metrics.history if official_metrics else [],
    }
    _write_json(output_dir / "training_metadata.json", metadata)

    experiment_config.update(
        {
            "status": "completed",
            "finished_at_utc": _utc_now(),
            "result": {
                **metadata,
                "train_metrics": train_result.metrics,
                "early_stopping_best_metric": early_stopping_callback.best_metric
                if early_stopping_callback
                else None,
            },
        }
    )
    _write_json(experiment_config_path, experiment_config)
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
    parser.add_argument(
        "--language-sampling-alpha",
        type=float,
        default=1.0,
        help=(
            "Training-language exponent: 1.0 keeps the natural chunk distribution, "
            "0.5 uses square-root balancing, and 0.0 gives equal expected language mass."
        ),
    )
    parser.add_argument(
        "--class-weight-alpha",
        type=float,
        default=0.0,
        help=(
            "Inverse-frequency exponent for token-label loss weights; "
            "0.0 keeps standard cross-entropy."
        ),
    )
    parser.add_argument(
        "--class-weight-min",
        type=float,
        default=0.75,
        help="Lower clip bound for class weights.",
    )
    parser.add_argument(
        "--class-weight-max",
        type=float,
        default=2.5,
        help="Upper clip bound for class weights.",
    )
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument(
        "--selection-metric",
        default="token",
        choices=["token", "official"],
        help="Select checkpoints using token F1 or the official multilingual MWE macro-F1.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without improvement; zero disables it.",
    )
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--label-all-tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-sentences-per-language", type=int, default=None)
    parser.add_argument("--max-dev-sentences-per-language", type=int, default=None)
