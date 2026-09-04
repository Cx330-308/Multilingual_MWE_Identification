from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cupt import CuptSentence, read_cupt, resolve_languages, split_path
from .llm_baseline import (
    PROMPT_VERSION,
    _cached_model_revision,
    _render_chat,
    build_chat_messages,
    gold_mwe_groups,
)
from .paths import default_artifacts_dir, default_data_dir
from .qwen import load_text_causal_lm
from .sampling import build_language_sampling_plan, create_weighted_sampler


LORA_PROTOCOL_VERSION = "qwen-generative-lora-v1"


@dataclass(frozen=True)
class EncodedSftExample:
    language: str
    sentence_index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int


class CompletionOnlyDataset:
    def __init__(self, examples: Sequence[EncodedSftExample]):
        self.examples = list(examples)
        self.languages = [example.language for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": list(example.input_ids),
            "attention_mask": [1] * len(example.input_ids),
            "labels": list(example.labels),
        }


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch.")
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def completion_text(sentence: CuptSentence) -> str:
    return json.dumps(
        {"mwes": [list(group) for group in gold_mwe_groups(sentence)]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def encode_sft_example(
    tokenizer: Any,
    *,
    language: str,
    sentence_index: int,
    sentence: CuptSentence,
    max_length: int,
) -> EncodedSftExample | None:
    prompt = _render_chat(
        tokenizer,
        build_chat_messages(language, (), sentence),
        thinking=False,
    )
    eos_token = tokenizer.eos_token
    if not eos_token:
        raise ValueError("The tokenizer must define an EOS token for LoRA SFT.")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        completion_text(sentence) + eos_token,
        add_special_tokens=False,
    )["input_ids"]
    if not completion_ids or len(prompt_ids) + len(completion_ids) > max_length:
        return None
    input_ids = tuple(prompt_ids + completion_ids)
    labels = tuple([-100] * len(prompt_ids) + completion_ids)
    return EncodedSftExample(
        language=language,
        sentence_index=sentence_index,
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion_ids),
    )


def select_sentence_indices(
    sentence_count: int,
    *,
    language: str,
    limit: int | None,
    seed: int,
) -> list[int]:
    if limit is None or limit >= sentence_count:
        return list(range(sentence_count))
    if limit < 1:
        raise ValueError("The per-language sentence limit must be positive.")
    digest = hashlib.sha256(
        f"{LORA_PROTOCOL_VERSION}:{seed}:{language}:training-subset".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
    return sorted(rng.sample(range(sentence_count), limit))


def text_linear_module_names(model: Any) -> list[str]:
    import torch

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (
            ".language_model.layers." in name
            or name.startswith("model.layers.")
        )
        and not name.startswith("mtp.")
    ]
    if not names:
        raise RuntimeError("No text-stack linear modules were found for LoRA.")
    return sorted(names)


def _validate_args(args: argparse.Namespace) -> None:
    numeric_positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "logging_steps": args.logging_steps,
    }
    for name, value in numeric_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero.")
    if not 0.0 <= args.language_sampling_alpha <= 1.0:
        raise ValueError("--language-sampling-alpha must be between 0 and 1.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer.")
    if (
        args.max_train_sentences_per_language is not None
        and args.max_train_sentences_per_language < 1
    ):
        raise ValueError("--max-train-sentences-per-language must be positive.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not callable(value)
    }


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        device = requested
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _build_dataset(
    args: argparse.Namespace,
    tokenizer: Any,
    languages: Sequence[str],
) -> tuple[CompletionOnlyDataset, dict[str, Any]]:
    examples: list[EncodedSftExample] = []
    source_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    encoded_counts: dict[str, int] = {}
    overlength_counts: dict[str, int] = {}
    training_subset_seed = (
        args.seed if args.training_subset_seed is None else args.training_subset_seed
    )

    for language in languages:
        sentences = read_cupt(split_path(args.data_dir, language, "train"))
        source_counts[language] = len(sentences)
        indices = select_sentence_indices(
            len(sentences),
            language=language,
            limit=args.max_train_sentences_per_language,
            seed=training_subset_seed,
        )
        selected_counts[language] = len(indices)
        encoded = 0
        overlength = 0
        for sentence_index in indices:
            example = encode_sft_example(
                tokenizer,
                language=language,
                sentence_index=sentence_index,
                sentence=sentences[sentence_index],
                max_length=args.max_length,
            )
            if example is None:
                overlength += 1
                continue
            examples.append(example)
            encoded += 1
        encoded_counts[language] = encoded
        overlength_counts[language] = overlength

    if not examples:
        raise RuntimeError("No training examples fit the configured maximum length.")
    summary = {
        "training_subset_seed": training_subset_seed,
        "source_sentences": source_counts,
        "selected_sentences": selected_counts,
        "encoded_sentences": encoded_counts,
        "overlength_sentences_excluded": overlength_counts,
        "total_encoded_sentences": len(examples),
        "mean_sequence_tokens": sum(len(example.input_ids) for example in examples)
        / len(examples),
        "mean_completion_tokens": sum(
            example.completion_tokens for example in examples
        )
        / len(examples),
    }
    return CompletionOnlyDataset(examples), summary


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "LoRA training requires torch, transformers, accelerate and peft. "
            "Install requirements.txt in the Python 3.12 LLM environment."
        ) from exc

    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(torch, args.device)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    data_dir = Path(args.data_dir).expanduser().resolve()
    args.data_dir = str(data_dir)
    languages = [
        language
        for language in resolve_languages(data_dir, args.languages)
        if split_path(data_dir, language, "train").is_file()
        and split_path(data_dir, language, "dev").is_file()
    ]
    if not languages:
        raise SystemExit("No supervised languages with train and dev data were found.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError(
            f"Adapter output already exists and will not be overwritten: {adapter_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, data_summary = _build_dataset(args, tokenizer, languages)
    sampling_plan = build_language_sampling_plan(
        dataset.languages,
        args.language_sampling_alpha,
    )

    print(
        f"Loading text-only {args.model_name} on {device} "
        f"({str(dtype).removeprefix('torch.')})."
    )
    model = load_text_causal_lm(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device=device,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    target_modules = text_linear_module_names(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(
        f"LoRA targets {len(model.targeted_module_names)} text linear modules; "
        f"trainable parameters {trainable:,}/{total:,} ({100 * trainable / total:.3f}%)."
    )

    updates_per_epoch = math.ceil(
        len(dataset) / (args.batch_size * args.gradient_accumulation_steps)
    )
    estimated_update_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(updates_per_epoch * args.epochs)
    )
    warmup_steps = math.ceil(estimated_update_steps * args.warmup_ratio)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "max_steps": args.max_steps,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "use_cpu" in signature.parameters:
        training_kwargs["use_cpu"] = device == "cpu"
    training_args = TrainingArguments(**training_kwargs)

    class LanguageBalancedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            selected_dataset = (
                train_dataset if train_dataset is not None else self.train_dataset
            )
            if selected_dataset is None or not sampling_plan.uses_weighted_sampling:
                return super()._get_train_sampler(train_dataset)
            return create_weighted_sampler(
                sampling_plan,
                dataset_length=len(selected_dataset),
                seed=args.seed,
            )

        def training_step(self, model, inputs, *step_args, **step_kwargs):
            loss = super().training_step(model, inputs, *step_args, **step_kwargs)
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError("LoRA training produced a non-finite loss.")
            return loss

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": CompletionOnlyCollator(tokenizer.pad_token_id),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = LanguageBalancedTrainer(**trainer_kwargs)

    config = {
        "schema_version": 1,
        "protocol_version": LORA_PROTOCOL_VERSION,
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "arguments": _serializable_args(args),
        "model": {
            "name": args.model_name,
            "revision": args.revision or _cached_model_revision(args.model_name),
            "text_only_load": True,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(model.targeted_module_names),
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
        "objective": {
            "type": "completion-only causal language modelling",
            "prompt_version": PROMPT_VERSION,
            "output": '{"mwes":[[token_index,...],...]}',
        },
        "optimization": {
            "estimated_update_steps": estimated_update_steps,
            "warmup_steps": warmup_steps,
        },
        "data": data_summary,
        "sampling": sampling_plan.summary(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
        },
    }
    _write_json(output_dir / "experiment_config.json", config)

    try:
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    except Exception as exc:
        config.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(output_dir / "experiment_config.json", config)
        raise

    metadata = {
        "completed_epochs": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "train_metrics": result.metrics,
        "adapter_dir": str(adapter_dir),
    }
    _write_json(output_dir / "training_metadata.json", metadata)
    config.update(
        {
            "status": "completed",
            "finished_at_utc": _utc_now(),
            "result": metadata,
        }
    )
    _write_json(output_dir / "experiment_config.json", config)
    print(f"Saved LoRA adapter to {adapter_dir}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision")
    parser.add_argument(
        "--output-dir",
        default=str(
            default_artifacts_dir()
            / "experiments"
            / "lora-qwen3p5-4b-n500pl-ml640-sa0p5-r16-a32-d0p05-lr2em4-ep1-seed13"
        ),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--language-sampling-alpha", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-sentences-per-language", type=int, default=500)
    parser.add_argument(
        "--training-subset-seed",
        type=int,
        help=(
            "Seed used only to choose capped per-language training subsets. "
            "Defaults to --seed for backward compatibility."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint")
