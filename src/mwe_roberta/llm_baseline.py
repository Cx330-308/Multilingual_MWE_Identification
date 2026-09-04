from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cupt import CuptSentence, read_cupt, resolve_languages, split_path, write_cupt
from .official import average_evaluations, evaluator_paths, run_language_evaluation
from .paths import default_artifacts_dir, default_data_dir
from .qwen import load_text_causal_lm


PROMPT_VERSION = "parseme-identification-v2"
GENERIC_MWE_CATEGORY = "VID"
GLOBAL_COLUMNS_COMMENT = (
    "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS "
    "MISC PARSEME:MWE"
)
SYSTEM_PROMPT = """You annotate lexical multiword expressions (MWEs) using the PARSEME 2.0 identification guidelines.

Return exactly one JSON object with this schema:
{"mwes":[[token_index,...],...]}

Each inner list is the complete set of 1-based token indices for one MWE. Include every component token, including gaps in discontinuous MWEs. Separate overlapping MWEs into separate lists. Relevant constructions include idioms, light-verb constructions, and fixed adverbial, prepositional, or nominal expressions. Annotate only conventionalized lexical units, not ordinary compositional phrases or general syntactic constituents. Never mark a whole clause or sentence merely because its words belong together grammatically. Most MWEs are short and contain multiple indexed tokens. A one-index group is valid only when one surface token is itself a fused lexicalized expression; never return only the head of a longer expression. Use {"mwes":[]} when there is no MWE. Do not return categories, confidence scores, markdown, or explanations."""

LANGUAGE_NAMES = {
    "EGY": "Ancient Egyptian",
    "EL": "Greek",
    "FA": "Persian",
    "FR": "French",
    "HE": "Hebrew",
    "JA": "Japanese",
    "KA": "Georgian",
    "LV": "Latvian",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "SL": "Slovenian",
    "SR": "Serbian",
    "SV": "Swedish",
    "UK": "Ukrainian",
}


@dataclass(frozen=True)
class ParsedResponse:
    groups: tuple[tuple[int, ...], ...]
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SampleSelection:
    language: str
    sentence_index: int
    polarity: str
    word_count: int


@dataclass(frozen=True)
class MetricScore:
    precision: float
    recall: float
    f1: float


@dataclass
class LocalGenerator:
    model: Any
    tokenizer: Any
    torch: Any
    device: str
    dtype: str
    revision: str | None

    def generate(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        max_input_tokens: int,
        max_new_tokens: int,
        thinking: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        generation_seed: int,
    ) -> list[dict[str, Any]]:
        rendered = [
            _render_chat(self.tokenizer, messages, thinking=thinking)
            for messages in message_batches
        ]
        original_lengths = [
            len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in rendered
        ]
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = encoded.to(self.device)
        input_width = encoded["input_ids"].shape[1]
        input_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]

        generation_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            self.torch.manual_seed(generation_seed)
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
            if min_p > 0:
                generation_kwargs["min_p"] = min_p

        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        elapsed = time.perf_counter() - started

        outputs = generated[:, input_width:]
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [
            {
                "raw_response": text.strip(),
                "input_tokens": input_length,
                "output_tokens": len(
                    self.tokenizer(text, add_special_tokens=False)["input_ids"]
                ),
                "input_truncated": original_length > max_input_tokens,
                "batch_seconds": elapsed,
                "batch_size": len(rendered),
                "generation_seed": generation_seed if do_sample else None,
            }
            for text, input_length, original_length in zip(
                decoded, input_lengths, original_lengths
            )
        ]


def gold_mwe_groups(sentence: CuptSentence) -> tuple[tuple[int, ...], ...]:
    memberships: dict[str, list[int]] = {}
    for position, cell in enumerate(sentence.mwe_columns, start=1):
        if cell in {"", "_", "*"}:
            continue
        for annotation in cell.split(";"):
            mwe_id = annotation.split(":", 1)[0].strip()
            if not mwe_id.isdigit():
                continue
            memberships.setdefault(mwe_id, []).append(position)

    return tuple(
        tuple(sorted(set(memberships[mwe_id])))
        for mwe_id in sorted(memberships, key=int)
    )


def mwe_groups_to_columns(
    word_count: int,
    groups: Sequence[Sequence[int]],
    *,
    category: str = GENERIC_MWE_CATEGORY,
) -> list[str]:
    columns: list[list[str]] = [[] for _ in range(word_count)]
    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for group in groups:
        positions = tuple(sorted(set(group)))
        if not positions:
            raise ValueError("MWE groups cannot be empty.")
        if positions[0] < 1 or positions[-1] > word_count:
            raise ValueError(
                f"MWE positions {positions} are outside the 1..{word_count} token range."
            )
        if positions not in seen:
            normalized.append(positions)
            seen.add(positions)

    for mwe_id, positions in enumerate(normalized, start=1):
        for offset, position in enumerate(positions):
            annotation = f"{mwe_id}:{category}" if offset == 0 else str(mwe_id)
            columns[position - 1].append(annotation)

    return [";".join(memberships) if memberships else "*" for memberships in columns]


def parse_llm_response(text: str, word_count: int) -> ParsedResponse:
    payload = _find_json_payload(text)
    if payload is None:
        return ParsedResponse((), "error", ("No JSON object containing 'mwes' was found.",))

    raw_groups = payload.get("mwes")
    if not isinstance(raw_groups, list):
        return ParsedResponse((), "error", ("The 'mwes' value is not a JSON list.",))

    warnings: list[str] = []
    groups: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, list) or not raw_group:
            warnings.append(f"Dropped MWE {group_index}: expected a non-empty list.")
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_group):
            warnings.append(f"Dropped MWE {group_index}: token indices must be integers.")
            continue
        if any(value < 1 or value > word_count for value in raw_group):
            warnings.append(
                f"Dropped MWE {group_index}: an index is outside 1..{word_count}."
            )
            continue

        normalized = tuple(sorted(set(raw_group)))
        if len(normalized) != len(raw_group):
            warnings.append(f"Deduplicated repeated indices in MWE {group_index}.")
        if normalized in seen:
            warnings.append(f"Dropped duplicate MWE {group_index}.")
            continue
        groups.append(normalized)
        seen.add(normalized)

    status = "warning" if warnings else "ok"
    return ParsedResponse(tuple(groups), status, tuple(warnings))


def sample_sentence_indices(
    sentences: Sequence[CuptSentence],
    *,
    language: str,
    positive_count: int,
    negative_count: int,
    seed: int,
    max_words: int,
    purpose: str = "development-sample",
    excluded_indices: Sequence[int] = (),
) -> list[SampleSelection]:
    if positive_count < 0 or negative_count < 0:
        raise ValueError("Sample counts must be non-negative.")
    if max_words < 1:
        raise ValueError("max_words must be positive.")

    excluded = set(excluded_indices)
    positive = [
        index
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and len(sentence.words) <= max_words
        and gold_mwe_groups(sentence)
    ]
    negative = [
        index
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and len(sentence.words) <= max_words
        and not gold_mwe_groups(sentence)
    ]
    if len(positive) < positive_count or len(negative) < negative_count:
        raise ValueError(
            f"{language} has {len(positive)} eligible positive and {len(negative)} "
            f"eligible negative sentences; requested {positive_count} and {negative_count}."
        )

    rng = random.Random(_stable_seed(seed, language, purpose))
    chosen_positive = rng.sample(positive, positive_count)
    chosen_negative = rng.sample(negative, negative_count)
    selected = [
        SampleSelection(language, index, "positive", len(sentences[index].words))
        for index in chosen_positive
    ]
    selected.extend(
        SampleSelection(language, index, "negative", len(sentences[index].words))
        for index in chosen_negative
    )
    return sorted(selected, key=lambda item: item.sentence_index)


def all_sentence_indices(
    sentences: Sequence[CuptSentence],
    *,
    language: str,
) -> list[SampleSelection]:
    """Select a complete split in source order for official full-set evaluation."""
    return [
        SampleSelection(
            language=language,
            sentence_index=index,
            polarity="positive" if gold_mwe_groups(sentence) else "negative",
            word_count=len(sentence.words),
        )
        for index, sentence in enumerate(sentences)
    ]


def select_demonstration_indices(
    sentences: Sequence[CuptSentence],
    *,
    language: str,
    shots: int,
    seed: int,
    max_words: int,
    excluded_indices: Sequence[int] = (),
) -> list[int]:
    if shots < 0:
        raise ValueError("shots must be non-negative.")
    if shots == 0:
        return []

    excluded = set(excluded_indices)
    positive = [
        index
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and len(sentence.words) <= max_words
        and gold_mwe_groups(sentence)
    ]
    contextual_positive = [
        index
        for index in positive
        if len(gold_mwe_groups(sentences[index])) == 1
        and 2 <= len(gold_mwe_groups(sentences[index])[0]) <= 4
        and len(sentences[index].words) >= 6
        and len(gold_mwe_groups(sentences[index])[0]) <= len(sentences[index].words) - 3
    ]
    contextual_negative = [
        index
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and 6 <= len(sentence.words) <= max_words
        and not gold_mwe_groups(sentence)
    ]
    all_negative = [
        index
        for index, sentence in enumerate(sentences)
        if index not in excluded
        and len(sentence.words) <= max_words
        and not gold_mwe_groups(sentence)
    ]

    positive_needed = (shots + 1) // 2
    negative_needed = shots // 2
    positive_pool = (
        contextual_positive if len(contextual_positive) >= positive_needed else positive
    )
    negative_pool = (
        contextual_negative
        if len(contextual_negative) >= negative_needed
        else all_negative
    )
    if len(positive_pool) < positive_needed or len(negative_pool) < negative_needed:
        raise ValueError(
            f"{language} does not have enough training examples for {shots}-shot prompting."
        )

    rng = random.Random(_stable_seed(seed, language, "demonstrations"))
    chosen_positive = rng.sample(positive_pool, positive_needed)
    chosen_negative = rng.sample(negative_pool, negative_needed)
    demonstrations: list[int] = []
    for slot in range(shots):
        source = chosen_positive if slot % 2 == 0 else chosen_negative
        demonstrations.append(source[slot // 2])
    return demonstrations


def build_chat_messages(
    language: str,
    demonstrations: Sequence[CuptSentence],
    target: CuptSentence,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demonstration in demonstrations:
        messages.append(
            {"role": "user", "content": _sentence_request(language, demonstration)}
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"mwes": [list(group) for group in gold_mwe_groups(demonstration)]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    messages.append({"role": "user", "content": _sentence_request(language, target)})
    return messages


def parse_metric_score(text: str, label: str) -> MetricScore:
    prefix = f"* {label}:"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        fields = stripped.split()
        values = {field.split("=", 1)[0]: field for field in fields if "=" in field}
        try:
            return MetricScore(
                precision=_metric_value(values["P"]),
                recall=_metric_value(values["R"]),
                f1=_metric_value(values["F"]),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Could not parse official {label} score line: {line}") from exc
    raise ValueError(f"Official evaluator output has no {label} score.")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_dir = (
        Path(args.reference_prediction_dir).expanduser().resolve()
        if args.reference_prediction_dir
        else None
    )
    languages = _training_languages(data_dir, args.languages)
    target_split = args.target_split
    sample_purpose = args.sample_purpose or (
        "development-sample" if target_split == "dev" else "train-calibration-sample"
    )
    artifact_split = "dev" if target_split == "dev" else "train-calibration"
    fixed_demonstrations = _load_fixed_demonstrations(
        args.fixed_demonstration_indices, languages
    )

    targets: dict[str, list[CuptSentence]] = {}
    training: dict[str, list[CuptSentence]] = {}
    selections: dict[str, list[SampleSelection]] = {}
    demonstration_indices: dict[str, list[int]] = {}
    requests: list[dict[str, Any]] = []

    for language in languages:
        training[language] = read_cupt(split_path(data_dir, language, "train"))
        targets[language] = (
            read_cupt(split_path(data_dir, language, "dev"))
            if target_split == "dev"
            else training[language]
        )
        if getattr(args, "all_targets", False):
            selections[language] = all_sentence_indices(
                targets[language],
                language=language,
            )
        else:
            selections[language] = sample_sentence_indices(
                targets[language],
                language=language,
                positive_count=args.positive_per_language,
                negative_count=args.negative_per_language,
                seed=args.seed,
                max_words=args.max_words,
                purpose=sample_purpose,
            )
        target_indices = {
            selection.sentence_index for selection in selections[language]
        }
        if fixed_demonstrations is not None:
            demonstration_indices[language] = fixed_demonstrations[language]
        else:
            demo_pool_shots = args.demo_pool_shots or args.shots
            demonstration_indices[language] = select_demonstration_indices(
                training[language],
                language=language,
                shots=demo_pool_shots,
                seed=args.seed,
                max_words=args.max_demo_words,
                excluded_indices=target_indices if target_split == "train" else (),
            )[: args.shots]
        _validate_demonstrations(
            language=language,
            sentence_count=len(training[language]),
            indices=demonstration_indices[language],
            shots=args.shots,
            forbidden_indices=target_indices if target_split == "train" else set(),
        )
        demonstrations = [
            training[language][index] for index in demonstration_indices[language]
        ]
        for selection in selections[language]:
            target = targets[language][selection.sentence_index]
            requests.append(
                {
                    **asdict(selection),
                    "demonstration_indices": demonstration_indices[language],
                    "messages": build_chat_messages(language, demonstrations, target),
                }
            )

    experiment = _experiment_spec(
        args=args,
        data_dir=data_dir,
        reference_dir=reference_dir,
        languages=languages,
        demonstration_indices=demonstration_indices,
        sample_purpose=sample_purpose,
    )
    _initialize_experiment(output_dir, experiment, selections, requests)
    _write_gold_and_reference_subsets(
        output_dir=output_dir,
        reference_dir=reference_dir,
        targets=targets,
        selections=selections,
        split_name=artifact_split,
    )

    total = len(requests)
    print(
        f"Prepared {total} requests across {len(languages)} languages in {output_dir}."
    )
    if args.prepare_only:
        print("Preparation only: model inference was not started.")
        return

    response_path = output_dir / "responses.jsonl"
    responses = _load_responses(response_path)
    pending = [
        request
        for request in requests
        if (request["language"], request["sentence_index"]) not in responses
        or (
            args.retry_errors
            and responses[(request["language"], request["sentence_index"])]["status"]
            == "error"
        )
    ]

    if pending:
        generator = _load_local_generator(args)
        _update_runtime_config(output_dir, generator)
        completed = total - len(pending)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            generated = generator.generate(
                [request["messages"] for request in batch],
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                thinking=args.thinking,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                generation_seed=_stable_seed(
                    args.seed, "MULTILINGUAL", f"generation-batch-{start}"
                ),
            )
            records: list[dict[str, Any]] = []
            for request, generation in zip(batch, generated):
                parsed = parse_llm_response(
                    generation["raw_response"], request["word_count"]
                )
                record = {
                    "language": request["language"],
                    "sentence_index": request["sentence_index"],
                    "polarity": request["polarity"],
                    "word_count": request["word_count"],
                    "prompt": request["messages"],
                    **generation,
                    "parsed_mwes": [list(group) for group in parsed.groups],
                    "status": parsed.status,
                    "warnings": list(parsed.warnings),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
                responses[(record["language"], record["sentence_index"])] = record
            _append_jsonl(response_path, records)
            completed += len(batch)
            print(f"Generated {completed}/{total} responses.", flush=True)
        _release_local_generator(generator)
    else:
        print(f"Reusing all {total} responses from {response_path}.")

    _write_predictions(
        output_dir=output_dir,
        targets=targets,
        selections=selections,
        responses=responses,
        split_name=artifact_split,
    )
    metrics = _evaluate_systems(
        data_dir=data_dir,
        output_dir=output_dir,
        languages=languages,
        has_reference=reference_dir is not None,
        split_name=artifact_split,
    )
    _write_results(output_dir, experiment, requests, responses, metrics)
    llm_score = metrics["llm"]["average"]["mwe"]
    score_scope = "full-development" if experiment.get("all_targets") else "subset"
    print(
        f"LLM official {score_scope} macro: "
        f"P={llm_score['precision']:.4f} R={llm_score['recall']:.4f} "
        f"F={llm_score['f1']:.4f}"
    )
    print(f"Wrote {output_dir / 'report.md'}")


def add_args(parser: argparse.ArgumentParser) -> None:
    default_output = default_artifacts_dir() / "experiments" / "llm-qwen3p5-4b-4shot-seed13"
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--reference-prediction-dir")
    parser.add_argument(
        "--adapter-dir",
        help="Optional PEFT LoRA adapter to load on the named base model.",
    )
    parser.add_argument("--target-split", choices=("dev", "train"), default="dev")
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Evaluate every sentence in the development split for a full official score.",
    )
    parser.add_argument(
        "--sample-purpose",
        help="Stable sampling namespace; keep fixed across tuning candidates.",
    )
    parser.add_argument("--positive-per-language", type=int, default=4)
    parser.add_argument("--negative-per-language", type=int, default=4)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument(
        "--demo-pool-shots",
        type=int,
        help="Select this many demonstrations before truncating to --shots.",
    )
    parser.add_argument(
        "--fixed-demonstration-indices",
        help="JSON mapping from language to frozen zero-based training indices.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--max-demo-words", type=int, default=40)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_split == "train" and args.reference_prediction_dir:
        raise ValueError("Reference predictions are only supported for dev targets.")
    if getattr(args, "all_targets", False) and args.target_split != "dev":
        raise ValueError("--all-targets is supported only for development evaluation.")
    if args.shots < 0:
        raise ValueError("--shots must be non-negative.")
    if args.demo_pool_shots is not None and args.demo_pool_shots < args.shots:
        raise ValueError("--demo-pool-shots must be at least --shots.")
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("Input and output token limits must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if not 0 <= args.min_p <= 1:
        raise ValueError("--min-p must be in [0, 1].")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive.")


def _load_fixed_demonstrations(
    path_value: str | None, languages: Sequence[str]
) -> dict[str, list[int]] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Frozen demonstration file must contain a JSON object: {path}")

    result: dict[str, list[int]] = {}
    for language in languages:
        raw_indices = value.get(language)
        if not isinstance(raw_indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_indices
        ):
            raise ValueError(
                f"Frozen demonstrations for {language} must be a list of integers."
            )
        result[language] = raw_indices
    return result


def _validate_demonstrations(
    *,
    language: str,
    sentence_count: int,
    indices: Sequence[int],
    shots: int,
    forbidden_indices: set[int],
) -> None:
    if len(indices) != shots:
        raise ValueError(
            f"{language} has {len(indices)} frozen demonstrations; expected {shots}."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"{language} demonstration indices contain duplicates.")
    invalid = [index for index in indices if index < 0 or index >= sentence_count]
    if invalid:
        raise ValueError(f"{language} demonstration indices are out of range: {invalid}")
    overlap = sorted(set(indices) & forbidden_indices)
    if overlap:
        raise ValueError(
            f"{language} demonstrations overlap train calibration targets: {overlap}"
        )


def _find_json_payload(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "mwes" in value:
            return value
    return None


def _stable_seed(seed: int, language: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{language}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sentence_request(language: str, sentence: CuptSentence) -> str:
    language_name = LANGUAGE_NAMES.get(language, language)
    indexed_tokens = "\n".join(
        f"{position}\t{word}" for position, word in enumerate(sentence.words, start=1)
    )
    return (
        f"Language: {language_name} ({language})\n"
        "Tokens (index followed by surface form):\n"
        f"{indexed_tokens}\n"
        "Return the JSON annotation."
    )


def _render_chat(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    thinking: bool = False,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _metric_value(field: str) -> float:
    return float(field.rsplit("=", 1)[-1].split("(", 1)[0])


def _training_languages(data_dir: Path, requested: str) -> list[str]:
    resolved = resolve_languages(data_dir, requested)
    available = [
        language
        for language in resolved
        if split_path(data_dir, language, "train").is_file()
        and split_path(data_dir, language, "dev").is_file()
    ]
    missing = sorted(set(resolved) - set(available))
    if missing and requested.lower() != "all":
        raise ValueError(
            "LLM development evaluation requires train.cupt and dev.cupt; missing for "
            + ", ".join(missing)
        )
    return available


def _experiment_spec(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    reference_dir: Path | None,
    languages: Sequence[str],
    demonstration_indices: dict[str, list[int]],
    sample_purpose: str,
) -> dict[str, Any]:
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    experiment = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "model_name": args.model_name,
        "revision": args.revision,
        "adapter_dir": (
            str(Path(args.adapter_dir).expanduser().resolve())
            if getattr(args, "adapter_dir", None)
            else None
        ),
        "data_dir": str(data_dir),
        "languages": list(languages),
        "seed": args.seed,
        "positive_per_language": args.positive_per_language,
        "negative_per_language": args.negative_per_language,
        "shots": args.shots,
        "max_words": args.max_words,
        "max_demo_words": args.max_demo_words,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "decoding": "sample" if args.do_sample else "greedy",
        "generic_cupt_category": GENERIC_MWE_CATEGORY,
        "reference_prediction_dir": str(reference_dir) if reference_dir else None,
        "demonstration_indices": demonstration_indices,
        "all_targets": bool(getattr(args, "all_targets", False)),
    }
    extended_protocol = any(
        (
            args.target_split != "dev",
            args.sample_purpose is not None,
            args.demo_pool_shots is not None,
            args.fixed_demonstration_indices is not None,
            args.thinking,
            args.do_sample,
            args.repetition_penalty != 1.0,
            getattr(args, "adapter_dir", None) is not None,
            bool(getattr(args, "all_targets", False)),
        )
    )
    if extended_protocol:
        experiment.update(
            {
                "schema_version": 2,
                "target_split": args.target_split,
                "sample_purpose": sample_purpose,
                "demo_pool_shots": args.demo_pool_shots,
                "fixed_demonstration_indices": (
                    str(Path(args.fixed_demonstration_indices).expanduser().resolve())
                    if args.fixed_demonstration_indices
                    else None
                ),
                "thinking": args.thinking,
                "sampling": {
                    "enabled": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "top_k": args.top_k if args.do_sample else None,
                    "min_p": args.min_p if args.do_sample else None,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed if args.do_sample else None,
                },
            }
        )
    return experiment


def _initialize_experiment(
    output_dir: Path,
    experiment: dict[str, Any],
    selections: dict[str, list[SampleSelection]],
    requests: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("experiment") != experiment:
            raise ValueError(
                f"Existing experiment configuration does not match requested run: {config_path}"
            )
    else:
        _write_json(
            config_path,
            {
                "experiment": experiment,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "transformers": _package_version("transformers"),
                    "torch": _package_version("torch"),
                },
            },
        )

    manifest_records = [
        asdict(selection)
        for language in experiment["languages"]
        for selection in selections[language]
    ]
    request_records = [
        {
            "language": request["language"],
            "sentence_index": request["sentence_index"],
            "polarity": request["polarity"],
            "word_count": request["word_count"],
            "demonstration_indices": request["demonstration_indices"],
            "messages": request["messages"],
        }
        for request in requests
    ]
    _write_jsonl_if_matching(output_dir / "sample_manifest.jsonl", manifest_records)
    _write_jsonl_if_matching(output_dir / "requests.jsonl", request_records)


def _write_gold_and_reference_subsets(
    *,
    output_dir: Path,
    reference_dir: Path | None,
    targets: dict[str, list[CuptSentence]],
    selections: dict[str, list[SampleSelection]],
    split_name: str,
) -> None:
    for language, language_selections in selections.items():
        gold_subset = [
            _copy_sentence(targets[language][selection.sentence_index])
            for selection in language_selections
        ]
        _write_cupt_with_header(
            gold_subset, output_dir / "gold" / language / f"{split_name}.cupt"
        )

        if reference_dir is None:
            continue
        reference_path = reference_dir / language / "dev.system.cupt"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing reference prediction: {reference_path}")
        full_reference = read_cupt(reference_path)
        if len(full_reference) != len(targets[language]):
            raise ValueError(
                f"Reference sentence count differs for {language}: "
                f"{len(full_reference)} != {len(targets[language])}"
            )
        reference_subset: list[CuptSentence] = []
        for selection in language_selections:
            gold_sentence = targets[language][selection.sentence_index]
            reference_sentence = full_reference[selection.sentence_index]
            if gold_sentence.words != reference_sentence.words:
                raise ValueError(
                    f"Reference token mismatch for {language} sentence "
                    f"{selection.sentence_index}."
                )
            reference_subset.append(
                gold_sentence.with_predicted_mwes(reference_sentence.mwe_columns)
            )
        _write_cupt_with_header(
            reference_subset,
            output_dir
            / "reference_predictions"
            / language
            / f"{split_name}.system.cupt",
        )


def _copy_sentence(sentence: CuptSentence) -> CuptSentence:
    return sentence.with_predicted_mwes(sentence.mwe_columns)


def _write_cupt_with_header(sentences: Sequence[CuptSentence], path: Path) -> None:
    copied = [_copy_sentence(sentence) for sentence in sentences]
    if copied and not any(
        comment.startswith("# global.columns") for comment in copied[0].comments
    ):
        copied[0].comments.insert(0, GLOBAL_COLUMNS_COMMENT)
    write_cupt(copied, path)


def _load_local_generator(args: argparse.Namespace) -> LocalGenerator:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM inference requires the project's torch and transformers dependencies."
        ) from exc

    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if device == "cpu":
        dtype = torch.float32
    elif device == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_dir = getattr(args, "adapter_dir", None)
    print(f"Loading {args.model_name} on {device} ({str(dtype).removeprefix('torch.')}).")
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading a LoRA adapter requires the peft package.") from exc
        model = load_text_causal_lm(
            args.model_name,
            revision=args.revision,
            local_files_only=args.local_files_only,
            dtype=dtype,
            device=device,
        )
        model = PeftModel.from_pretrained(
            model,
            str(Path(adapter_dir).expanduser().resolve()),
            is_trainable=False,
        )
    else:
        load_kwargs: dict[str, Any] = {
            "local_files_only": args.local_files_only,
            "dtype": dtype,
        }
        if args.revision:
            load_kwargs["revision"] = args.revision
        if device != "cpu":
            load_kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        if device == "cpu":
            model.to(device)
    model.eval()
    revision = (
        getattr(model.config, "_commit_hash", None)
        or args.revision
        or _cached_model_revision(args.model_name)
    )
    return LocalGenerator(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        dtype=str(dtype).removeprefix("torch."),
        revision=revision,
    )


def _update_runtime_config(output_dir: Path, generator: LocalGenerator) -> None:
    path = output_dir / "experiment_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["runtime"].update(
        {
            "resolved_device": generator.device,
            "resolved_dtype": generator.dtype,
            "resolved_model_revision": generator.revision,
        }
    )
    _write_json(path, config)


def _release_local_generator(generator: LocalGenerator) -> None:
    torch = generator.torch
    device = generator.device
    if device == "mps":
        torch.mps.synchronize()
    generator.model = None
    generator.tokenizer = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _load_responses(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    responses: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return responses
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record["language"]), int(record["sentence_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed response at {path}:{line_number}") from exc
            responses[key] = record
    return responses


def _write_predictions(
    *,
    output_dir: Path,
    targets: dict[str, list[CuptSentence]],
    selections: dict[str, list[SampleSelection]],
    responses: dict[tuple[str, int], dict[str, Any]],
    split_name: str,
) -> None:
    for language, language_selections in selections.items():
        predictions: list[CuptSentence] = []
        for selection in language_selections:
            key = (language, selection.sentence_index)
            if key not in responses:
                raise ValueError(
                    f"Missing generated response for {language}:{selection.sentence_index}"
                )
            sentence = targets[language][selection.sentence_index]
            columns = mwe_groups_to_columns(
                len(sentence.words), responses[key].get("parsed_mwes", [])
            )
            predictions.append(sentence.with_predicted_mwes(columns))
        _write_cupt_with_header(
            predictions,
            output_dir / "predictions" / language / f"{split_name}.system.cupt",
        )


def _evaluate_systems(
    *,
    data_dir: Path,
    output_dir: Path,
    languages: Sequence[str],
    has_reference: bool,
    split_name: str,
) -> dict[str, Any]:
    eval_script, average_script = evaluator_paths(data_dir)
    systems = {
        "llm": (output_dir / "predictions", output_dir / "evaluation"),
    }
    if has_reference:
        systems["reference"] = (
            output_dir / "reference_predictions",
            output_dir / "reference_evaluation",
        )

    metrics: dict[str, Any] = {}
    for system_name, (prediction_dir, evaluation_dir) in systems.items():
        evaluation_files: list[Path] = []
        per_language: dict[str, Any] = {}
        for language in languages:
            evaluation_path = evaluation_dir / language / "eval.txt"
            run_language_evaluation(
                eval_script=eval_script,
                gold_path=output_dir / "gold" / language / f"{split_name}.cupt",
                prediction_path=prediction_dir
                / language
                / f"{split_name}.system.cupt",
                train_path=split_path(data_dir, language, "train"),
                output_path=evaluation_path,
            )
            evaluation_files.append(evaluation_path)
            evaluation_text = evaluation_path.read_text(encoding="utf-8")
            per_language[language] = {
                "mwe": asdict(parse_metric_score(evaluation_text, "MWE-based")),
                "token": asdict(parse_metric_score(evaluation_text, "Tok-based")),
            }

        average_path = evaluation_dir / "average.txt"
        average_mwe = average_evaluations(
            average_script=average_script,
            evaluation_files=evaluation_files,
            output_path=average_path,
        )
        average_text = average_path.read_text(encoding="utf-8")
        metrics[system_name] = {
            "average": {
                "mwe": {
                    "precision": average_mwe.precision,
                    "recall": average_mwe.recall,
                    "f1": average_mwe.f1,
                },
                "token": asdict(parse_metric_score(average_text, "Tok-based")),
            },
            "per_language": per_language,
        }
    return metrics


def _write_results(
    output_dir: Path,
    experiment: dict[str, Any],
    requests: Sequence[dict[str, Any]],
    responses: dict[tuple[str, int], dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    ordered_responses = [
        responses[(request["language"], request["sentence_index"])] for request in requests
    ]
    status_counts = {
        status: sum(response["status"] == status for response in ordered_responses)
        for status in ("ok", "warning", "error")
    }
    truncated = sum(bool(response.get("input_truncated")) for response in ordered_responses)
    result = {
        "experiment": experiment,
        "response_status": status_counts,
        "truncated_prompts": truncated,
        "metrics": metrics,
    }
    _write_json(output_dir / "metrics.json", result)

    llm = metrics["llm"]["average"]
    target_split = experiment.get("target_split", "dev")
    is_calibration = target_split == "train"
    is_lora = bool(experiment.get("adapter_dir"))
    system_name = _report_system_name(experiment)
    thinking = bool(experiment.get("thinking", False))
    decoding = experiment["decoding"]
    title = (
        "# Train-Only LLM Calibration Candidate"
        if is_calibration
        else (
            "# LoRA-Fine-Tuned Local LLM Evaluation"
            if is_lora
            else "# Exploratory Local LLM Baseline"
        )
    )
    mode_text = "thinking enabled" if thinking else "thinking disabled"
    lines = [
        title,
        "",
        "## Protocol",
        "",
        (
            f"- Model: `{system_name}` with {decoding} decoding and "
            f"{mode_text}."
        ),
        *_report_protocol_lines(experiment, len(requests), is_calibration),
        "",
        "## Aggregate Scores",
        "",
        "| System | MWE P | MWE R | MWE F1 | Token P | Token R | Token F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _score_row(system_name, llm),
    ]
    if "reference" in metrics:
        lines.append(_score_row("Final XLM-R (same subset)", metrics["reference"]["average"]))

    lines.extend(
        [
            "",
            "## Output Reliability",
            "",
            f"- Valid strict parses: {status_counts['ok']}/{len(requests)}.",
            f"- Parsed with recoverable warnings: {status_counts['warning']}/{len(requests)}.",
            f"- Unparseable outputs scored as empty predictions: {status_counts['error']}/{len(requests)}.",
            f"- Truncated prompts: {truncated}/{len(requests)}.",
            "",
            "## Per-Language MWE F1",
            "",
            f"| Language | {system_name} | Reference | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for language in experiment["languages"]:
        llm_f1 = metrics["llm"]["per_language"][language]["mwe"]["f1"]
        if "reference" in metrics:
            reference_f1 = metrics["reference"]["per_language"][language]["mwe"]["f1"]
            reference_text = f"{reference_f1:.4f}"
            difference = f"{llm_f1 - reference_f1:+.4f}"
        else:
            reference_text = "n/a"
            difference = "n/a"
        lines.append(f"| {language} | {llm_f1:.4f} | {reference_text} | {difference} |")

    lines.extend(
        [
            "",
            "## Interpretation Constraints",
            "",
            _report_interpretation_text(experiment, is_calibration),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _report_system_name(experiment: dict[str, Any]) -> str:
    model_name = str(experiment["model_name"])
    return f"{model_name} + LoRA" if experiment.get("adapter_dir") else model_name


def _report_protocol_lines(
    experiment: dict[str, Any], request_count: int, is_calibration: bool
) -> list[str]:
    shots = int(experiment.get("shots", 0))
    prompting = (
        "- Prompting: zero-shot inference with no in-context demonstrations."
        if shots == 0
        else f"- Prompting: {shots}-shot per language using fixed training examples."
    )
    if is_calibration:
        return [
            prompting,
            (
                "- Training calibration sample: "
                f"{experiment['positive_per_language']} positive and "
                f"{experiment['negative_per_language']} negative sentences per language, "
                f"{request_count} sentences total."
            ),
            "- Leakage control: calibration targets and few-shot demonstrations are disjoint.",
            "- Comparison: official PARSEME scorer on exactly the same sampled sentences.",
        ]
    if experiment.get("all_targets"):
        language_count = len(experiment.get("languages", []))
        return [
            prompting,
            (
                "- Development set: complete split for "
                f"{language_count} configured languages, {request_count} sentences total."
            ),
            "- Evaluation: official PARSEME scorer over the complete development split.",
        ]
    return [
        prompting,
        (
            "- Development sample: "
            f"{experiment['positive_per_language']} positive and "
            f"{experiment['negative_per_language']} negative sentences per language, "
            f"{request_count} sentences total."
        ),
        "- Comparison: official PARSEME scorer on exactly the same sampled sentences.",
    ]


def _report_interpretation_text(
    experiment: dict[str, Any], is_calibration: bool
) -> str:
    shots = int(experiment.get("shots", 0))
    if is_calibration:
        return (
            "This candidate is used only for train-only configuration selection. "
            f"The balanced calibration sample is disjoint from its {shots} few-shot "
            "demonstrations and must not be reported as held-out model performance. "
            "Development labels do not participate in candidate ranking."
        )

    demonstration_text = (
        "No in-context demonstrations were used."
        if shots == 0
        else (
            f"The {shots} demonstrations are taken only from each language's "
            "training split."
        )
    )
    if experiment.get("all_targets"):
        return (
            "This run covers every sentence in the configured development files and is "
            "therefore a complete-development result rather than a balanced-sample "
            "estimate. It is comparable with systems scored on the same language set, "
            "although differences in training budget and number of seeds must still be "
            f"reported. {demonstration_text} Because the model may have encountered public "
            "PARSEME material during pretraining, data contamination cannot be ruled out. "
            "The result remains development-set evidence and does not establish blind-test "
            "performance."
        )
    return (
        "The balanced sentence sample is not representative of the natural "
        "development-set prevalence, so scores are comparable only between systems on "
        f"this fixed subset. {demonstration_text} Because the model may have encountered "
        "public PARSEME material during pretraining, data contamination cannot be ruled "
        "out. The result supports conclusions only about this model, prompt, decoding "
        "setup, and sample."
    )


def _score_row(name: str, scores: dict[str, Any]) -> str:
    mwe = scores["mwe"]
    token = scores["token"]
    return (
        f"| {name} | {mwe['precision']:.4f} | {mwe['recall']:.4f} | "
        f"{mwe['f1']:.4f} | {token['precision']:.4f} | {token['recall']:.4f} | "
        f"{token['f1']:.4f} |"
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cached_model_revision(model_name: str) -> str | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    reference = (
        Path(HF_HUB_CACHE)
        / f"models--{model_name.replace('/', '--')}"
        / "refs"
        / "main"
    )
    if not reference.is_file():
        return None
    value = reference.read_text(encoding="utf-8").strip()
    return value or None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_text(records: Sequence[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )


def _write_jsonl_if_matching(path: Path, records: Sequence[dict[str, Any]]) -> None:
    content = _jsonl_text(records)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing reproducibility file does not match: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
