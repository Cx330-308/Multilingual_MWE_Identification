from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cupt import read_cupt, resolve_languages, split_path, write_cupt
from .labels import bio_to_mwe_columns
from .paths import default_artifacts_dir, default_data_dir


def _prediction_filename(split: str) -> str:
    if split == "test.blind":
        return "test.system.cupt"
    return f"{split}.system.cupt"


def _normalise_id2label(id2label: dict[Any, str]) -> dict[int, str]:
    return {int(idx): label for idx, label in id2label.items()}


def _default_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict_sentence(words: list[str], tokenizer: Any, model: Any, device: Any, id2label: dict[int, str], args: argparse.Namespace) -> list[str]:
    import torch

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    word_logits: dict[int, list[Any]] = defaultdict(list)
    word_seen_in_chunk: set[tuple[int, int]] = set()
    with torch.no_grad():
        model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
        outputs = model(**model_inputs)
        logits = outputs.logits.detach().cpu()

    for batch_index in range(logits.shape[0]):
        word_ids = encoded.word_ids(batch_index=batch_index)
        for token_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            key = (batch_index, word_id)
            if key in word_seen_in_chunk:
                continue
            word_seen_in_chunk.add(key)
            word_logits[word_id].append(logits[batch_index, token_index])

    labels: list[str] = []
    for word_index in range(len(words)):
        if word_index not in word_logits:
            labels.append("O")
            continue
        averaged = torch.stack(word_logits[word_index]).mean(dim=0)
        labels.append(id2label[int(averaged.argmax().item())])
    return labels


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Missing prediction dependencies. Run:\n"
            "  cd mwe_roberta_pipeline\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    data_dir = Path(args.data_dir)
    languages = resolve_languages(data_dir, args.languages)
    model_dir = Path(args.model_dir).expanduser()
    output_dir = Path(args.output_dir)

    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"Trained model directory is missing or incomplete: {model_dir.resolve()}\n"
            "Training must finish successfully before prediction. Re-run the training command "
            "and check that best_model/config.json was created."
        )

    local_model_dir = str(model_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(local_model_dir)
    id2label = _normalise_id2label(model.config.id2label)
    device = torch.device(args.device) if args.device else _default_device(torch)
    model.to(device)
    model.eval()

    for language in languages:
        input_path = split_path(data_dir, language, args.split)
        if not input_path.exists():
            print(f"Skipping {language}: missing {input_path}")
            continue
        sentences = read_cupt(input_path, limit=args.max_sentences_per_language)
        predicted_sentences = []
        iterator = tqdm(sentences, desc=f"{language} {args.split}")
        for sentence in iterator:
            bio_labels = _predict_sentence(sentence.words, tokenizer, model, device, id2label, args)
            predicted_mwes = bio_to_mwe_columns(bio_labels)
            predicted_sentences.append(sentence.with_predicted_mwes(predicted_mwes))
        output_path = output_dir / language / _prediction_filename(args.split)
        write_cupt(predicted_sentences, output_path)
        print(f"Wrote {output_path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--split", default="test.blind", choices=["train", "dev", "test.blind"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=str(default_artifacts_dir() / "predictions"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", default=None, help="Override device, e.g. 'cpu', 'cuda', or 'mps'.")
    parser.add_argument("--max-sentences-per-language", type=int, default=None)
