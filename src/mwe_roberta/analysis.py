from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from .cupt import CuptSentence, iter_cupt
from .official import OfficialMweScore, parse_global_mwe_score
from .paths import default_artifacts_dir, project_root


SECTION_DIMENSIONS = {
    "## Global evaluation": "global",
    "## Per-category evaluation": "category",
    "## MWE continuity": "continuity",
    "## Number of tokens": "token_count",
    "## Whether seen in train": "seen_in_train",
    "## Whether identical to train": "train_identity",
}
COUNTED_SCORE_RE = re.compile(
    r"^P=(?P<p_correct>\d+)/(?P<p_total>\d+)=(?P<precision>[0-9.]+) "
    r"R=(?P<r_correct>\d+)/(?P<r_total>\d+)=(?P<recall>[0-9.]+) "
    r"F=(?P<f1>[0-9.]+)"
)
TOKEN_RANGE_RE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)$")


@dataclass(frozen=True)
class RunSpec:
    model: str
    seed: int
    experiment_dir: Path


@dataclass(frozen=True)
class BreakdownScore:
    dimension: str
    group: str
    correct: int
    predicted: int
    gold: int
    precision: float
    recall: float
    f1: float

    @property
    def false_positives(self) -> int:
        return self.predicted - self.correct

    @property
    def false_negatives(self) -> int:
        return self.gold - self.correct


def parse_official_breakdown(text: str) -> list[BreakdownScore]:
    dimension: str | None = None
    scores: list[BreakdownScore] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            dimension = next(
                (value for prefix, value in SECTION_DIMENSIONS.items() if line.startswith(prefix)),
                None,
            )
            continue
        if dimension is None or not line.startswith("* ") or "MWE-based: " not in line:
            continue

        label_text, metric_text = line[2:].split("MWE-based: ", 1)
        match = COUNTED_SCORE_RE.match(metric_text)
        if match is None:
            continue
        if match.group("p_correct") != match.group("r_correct"):
            raise ValueError(f"Precision and recall numerators differ: {raw_line}")

        group = label_text.removesuffix(": ").strip() or "All"
        scores.append(
            BreakdownScore(
                dimension=dimension,
                group=group,
                correct=int(match.group("p_correct")),
                predicted=int(match.group("p_total")),
                gold=int(match.group("r_total")),
                precision=float(match.group("precision")),
                recall=float(match.group("recall")),
                f1=float(match.group("f1")),
            )
        )
    return scores


def _load_manifest(path: Path) -> tuple[list[RunSpec], list[str], tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_dir = (path.parent / payload.get("base_dir", ".")).resolve()
    runs = [
        RunSpec(
            model=str(item["model"]),
            seed=int(item["seed"]),
            experiment_dir=(base_dir / item["experiment_dir"]).resolve(),
        )
        for item in payload["runs"]
    ]
    if not runs:
        raise ValueError("Analysis manifest contains no runs.")

    model_order = [str(model) for model in payload.get("model_order", [])]
    discovered_models = list(dict.fromkeys(run.model for run in runs))
    if not model_order:
        model_order = discovered_models
    if set(model_order) != set(discovered_models):
        raise ValueError("model_order must contain every model exactly once.")

    qualitative = payload.get("qualitative_run", {})
    qualitative_key = (str(qualitative["model"]), int(qualitative["seed"]))
    if qualitative_key not in {(run.model, run.seed) for run in runs}:
        raise ValueError("qualitative_run does not match a manifest run.")
    return runs, model_order, qualitative_key


def _best_evaluation(experiment_dir: Path) -> tuple[Path, OfficialMweScore]:
    evaluation_root = experiment_dir / "official-evaluations"
    candidates: list[tuple[Path, OfficialMweScore]] = []
    for evaluation_dir in sorted(evaluation_root.glob("eval-*")):
        average_path = evaluation_dir / "average.txt"
        if average_path.is_file():
            candidates.append(
                (average_path.parent, parse_global_mwe_score(average_path.read_text(encoding="utf-8")))
            )
    if not candidates:
        raise FileNotFoundError(f"No official evaluations found under {evaluation_root}")
    return max(candidates, key=lambda item: (item[1].f1, item[0].name))


def _sample_sd(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _global_breakdown(scores: Sequence[BreakdownScore], source: Path) -> BreakdownScore:
    matches = [score for score in scores if score.dimension == "global" and score.group == "All"]
    if len(matches) != 1:
        raise ValueError(f"Expected one global MWE score in {source}, found {len(matches)}.")
    return matches[0]


def _split_codes(field: str) -> list[str]:
    if not field or field == "*":
        return []
    return [part for part in field.split(";") if part and part != "*"]


def extract_mwe_spans(sentence: CuptSentence) -> dict[tuple[int, ...], str]:
    token_id_to_position: dict[int, int] = {}
    codes_by_position: dict[int, list[str]] = defaultdict(list)
    ranges: list[tuple[int, int, list[str]]] = []

    position = 0
    for token in sentence.tokens:
        if token.is_syntactic_token:
            position += 1
            token_id_to_position[int(token.token_id)] = position
            codes_by_position[position].extend(_split_codes(token.mwe))
            continue
        range_match = TOKEN_RANGE_RE.match(token.token_id)
        if range_match and _split_codes(token.mwe):
            ranges.append(
                (
                    int(range_match.group("start")),
                    int(range_match.group("end")),
                    _split_codes(token.mwe),
                )
            )

    for start, end, codes in ranges:
        for token_id in range(start, end + 1):
            if token_id in token_id_to_position:
                codes_by_position[token_id_to_position[token_id]].extend(codes)

    spans_by_id: dict[str, set[int]] = defaultdict(set)
    category_by_id: dict[str, str] = {}
    for word_position in sorted(codes_by_position):
        for code in codes_by_position[word_position]:
            if ":" in code:
                mwe_id, category = code.split(":", 1)
                category_by_id.setdefault(mwe_id, category)
            else:
                mwe_id = code
            spans_by_id[mwe_id].add(word_position)

    result: dict[tuple[int, ...], str] = {}
    for mwe_id, positions in spans_by_id.items():
        if not positions:
            continue
        span = tuple(sorted(positions))
        category = category_by_id.get(mwe_id, "<unlabeled>")
        if span in result and category not in result[span].split("|"):
            result[span] = "|".join(sorted({result[span], category}))
        else:
            result.setdefault(span, category)
    return result


def _span_text(words: Sequence[str], span: Sequence[int]) -> str:
    return " ".join(words[index - 1] for index in span)


def _span_string(span: Sequence[int]) -> str:
    return " ".join(str(index) for index in span)


def _qualitative_row(
    *,
    model: str,
    seed: int,
    language: str,
    sentence_index: int,
    error_type: str,
    subtype: str,
    category: str,
    span: tuple[int, ...],
    counterpart_category: str,
    counterpart_span: tuple[int, ...] | None,
    words: Sequence[str],
) -> dict[str, Any]:
    return {
        "model": model,
        "seed": seed,
        "language": language,
        "sentence_index": sentence_index,
        "error_type": error_type,
        "subtype": subtype,
        "category": category,
        "token_indexes": _span_string(span),
        "expression": _span_text(words, span),
        "counterpart_category": counterpart_category,
        "counterpart_token_indexes": _span_string(counterpart_span or ()),
        "counterpart_expression": _span_text(words, counterpart_span or ()),
        "sentence": " ".join(words),
    }


def collect_qualitative_errors(
    *,
    model: str,
    seed: int,
    evaluation_dir: Path,
    official_globals: dict[str, BreakdownScore],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language in sorted(official_globals):
        gold_path = evaluation_dir / "gold" / language / "dev.cupt"
        prediction_path = evaluation_dir / "predictions" / language / "dev.system.cupt"
        gold_sentences = list(iter_cupt(gold_path))
        prediction_sentences = list(iter_cupt(prediction_path))
        if len(gold_sentences) != len(prediction_sentences):
            raise ValueError(f"Sentence count mismatch for {language}.")

        correct = predicted = gold = 0
        for sentence_index, (gold_sentence, prediction_sentence) in enumerate(
            zip(gold_sentences, prediction_sentences), start=1
        ):
            if gold_sentence.words != prediction_sentence.words:
                raise ValueError(f"Token mismatch for {language} sentence {sentence_index}.")
            words = gold_sentence.words
            gold_spans = extract_mwe_spans(gold_sentence)
            prediction_spans = extract_mwe_spans(prediction_sentence)
            matched = set(gold_spans) & set(prediction_spans)
            unmatched_gold = set(gold_spans) - matched
            unmatched_prediction = set(prediction_spans) - matched
            correct += len(matched)
            predicted += len(prediction_spans)
            gold += len(gold_spans)

            for span in sorted(matched):
                if gold_spans[span] == prediction_spans[span]:
                    continue
                rows.append(
                    _qualitative_row(
                        model=model,
                        seed=seed,
                        language=language,
                        sentence_index=sentence_index,
                        error_type="CATEGORY_MISMATCH",
                        subtype="exact_span",
                        category=gold_spans[span],
                        span=span,
                        counterpart_category=prediction_spans[span],
                        counterpart_span=span,
                        words=words,
                    )
                )

            for span in sorted(unmatched_gold):
                overlaps = sorted(
                    candidate for candidate in unmatched_prediction if set(span) & set(candidate)
                )
                counterpart = overlaps[0] if overlaps else None
                rows.append(
                    _qualitative_row(
                        model=model,
                        seed=seed,
                        language=language,
                        sentence_index=sentence_index,
                        error_type="FN",
                        subtype="partial_overlap" if counterpart else "missed",
                        category=gold_spans[span],
                        span=span,
                        counterpart_category=(prediction_spans[counterpart] if counterpart else ""),
                        counterpart_span=counterpart,
                        words=words,
                    )
                )

            for span in sorted(unmatched_prediction):
                overlaps = sorted(candidate for candidate in unmatched_gold if set(span) & set(candidate))
                counterpart = overlaps[0] if overlaps else None
                rows.append(
                    _qualitative_row(
                        model=model,
                        seed=seed,
                        language=language,
                        sentence_index=sentence_index,
                        error_type="FP",
                        subtype="partial_overlap" if counterpart else "spurious",
                        category=prediction_spans[span],
                        span=span,
                        counterpart_category=(gold_spans[counterpart] if counterpart else ""),
                        counterpart_span=counterpart,
                        words=words,
                    )
                )

        official = official_globals[language]
        observed = (correct, predicted, gold)
        expected = (official.correct, official.predicted, official.gold)
        if observed != expected:
            raise ValueError(
                f"Qualitative span counts do not reproduce official counts for {language}: "
                f"observed={observed}, expected={expected}."
            )
    return rows


def _summarize_models(run_rows: Sequence[dict[str, Any]], model_order: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["model"])].append(row)

    result: list[dict[str, Any]] = []
    for model in model_order:
        rows = grouped[model]
        precision = [float(row["precision"]) for row in rows]
        recall = [float(row["recall"]) for row in rows]
        f1 = [float(row["f1"]) for row in rows]
        result.append(
            {
                "model": model,
                "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
                "run_count": len(rows),
                "precision_mean": mean(precision),
                "precision_sample_sd": _sample_sd(precision),
                "recall_mean": mean(recall),
                "recall_sample_sd": _sample_sd(recall),
                "f1_mean": mean(f1),
                "f1_sample_sd": _sample_sd(f1),
            }
        )
    return result


def _summarize_languages(
    language_rows: Sequence[dict[str, Any]], model_order: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in language_rows:
        grouped[(str(row["model"]), str(row["language"]))].append(row)

    result: list[dict[str, Any]] = []
    for (model, language), rows in grouped.items():
        precision = [float(row["precision"]) for row in rows]
        recall = [float(row["recall"]) for row in rows]
        f1 = [float(row["f1"]) for row in rows]
        result.append(
            {
                "model": model,
                "language": language,
                "seed_count": len(rows),
                "gold_mwes": int(rows[0]["gold"]),
                "predicted_mwes_mean": mean(float(row["predicted"]) for row in rows),
                "precision_mean": mean(precision),
                "precision_sample_sd": _sample_sd(precision),
                "recall_mean": mean(recall),
                "recall_sample_sd": _sample_sd(recall),
                "f1_mean": mean(f1),
                "f1_sample_sd": _sample_sd(f1),
            }
        )

    model_position = {model: position for position, model in enumerate(model_order)}
    result.sort(key=lambda row: (model_position[row["model"]], row["language"]))
    for model in model_order:
        ranked = sorted(
            (row for row in result if row["model"] == model),
            key=lambda row: (-row["f1_mean"], row["language"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["f1_rank"] = rank
    return result


def _summarize_breakdowns(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["dimension"]), str(row["group"]))].append(row)

    result: list[dict[str, Any]] = []
    for (model, dimension, group), items in grouped.items():
        correct = sum(int(item["correct"]) for item in items)
        predicted = sum(int(item["predicted"]) for item in items)
        gold = sum(int(item["gold"]) for item in items)
        micro_precision = correct / predicted if predicted else 0.0
        micro_recall = correct / gold if gold else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        result.append(
            {
                "model": model,
                "dimension": dimension,
                "group": group,
                "evaluations": len(items),
                "correct": correct,
                "predicted": predicted,
                "gold": gold,
                "false_positives": predicted - correct,
                "false_negatives": gold - correct,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "macro_precision_mean": mean(float(item["precision"]) for item in items),
                "macro_recall_mean": mean(float(item["recall"]) for item in items),
                "macro_f1_mean": mean(float(item["f1"]) for item in items),
            }
        )
    result.sort(key=lambda row: (row["model"], row["dimension"], row["group"]))
    return result


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pm(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sample_sd']:.4f}"


def _report(
    *,
    path: Path,
    model_order: Sequence[str],
    primary_model: str,
    model_summary: Sequence[dict[str, Any]],
    language_summary: Sequence[dict[str, Any]],
    breakdown_summary: Sequence[dict[str, Any]],
    qualitative_rows: Sequence[dict[str, Any]],
    qualitative_key: tuple[str, int],
) -> None:
    model_by_name = {row["model"]: row for row in model_summary}
    language_by_key = {(row["model"], row["language"]): row for row in language_summary}
    languages = sorted({str(row["language"]) for row in language_summary})
    language_winners: dict[str, str] = {}
    winner_counts = {model: 0 for model in model_order}
    for language in languages:
        winner = max(
            model_order,
            key=lambda model: language_by_key[(model, language)]["f1_mean"],
        )
        language_winners[language] = winner
        winner_counts[winner] += 1

    lines = [
        "# Final Model Per-Language and Error Analysis",
        "",
        "All selection scores are official MWE-based macro metrics. Each run uses its best "
        "official-development checkpoint. Error-breakdown counts are pooled micro diagnostics "
        "over languages and seeds; they are not replacements for the macro selection metric.",
        "",
        "## Model Summary",
        "",
        "| Model | Precision mean +/- SD | Recall mean +/- SD | Macro-F1 mean +/- SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in model_order:
        row = model_by_name[model]
        lines.append(
            f"| {_markdown_escape(model)} | {_pm(row, 'precision')} | {_pm(row, 'recall')} | "
            f"{_pm(row, 'f1')} |"
        )

    lines.extend(
        [
            "",
            "## Per-Language Mean F1",
            "",
            "Values are mean +/- sample SD across the three seeds. Gold support is shown because "
            "macro-F1 gives every language equal weight regardless of development-set size.",
            "",
            "| Language | Gold MWEs | "
            + " | ".join(f"{_markdown_escape(model)} mean +/- SD" for model in model_order)
            + " | Best |",
            "| --- | ---: | " + " | ".join("---:" for _ in model_order) + " | --- |",
        ]
    )
    for language in sorted(
        languages, key=lambda item: language_by_key[(primary_model, item)]["f1_mean"]
    ):
        gold_mwes = language_by_key[(primary_model, language)]["gold_mwes"]
        lines.append(
            f"| {language} | {gold_mwes} | "
            + " | ".join(_pm(language_by_key[(model, language)], "f1") for model in model_order)
            + f" | {language_winners[language]} |"
        )

    low_support = sorted(
        (
            row
            for row in language_summary
            if row["model"] == primary_model and int(row["gold_mwes"]) < 50
        ),
        key=lambda row: row["gold_mwes"],
    )
    lines.extend(
        [
            "",
            "## Cross-Model Language Pattern",
            "",
            "Languages with the highest mean F1: "
            + ", ".join(f"{model} {winner_counts[model]}/{len(languages)}" for model in model_order)
            + ".",
        ]
    )
    if low_support:
        lines.extend(
            [
                "",
                "Low-support development sets require particular caution: "
                + ", ".join(
                    f"{row['language']} (gold={row['gold_mwes']}, {primary_model} F1 "
                    f"{row['f1_mean']:.4f} +/- {row['f1_sample_sd']:.4f})"
                    for row in low_support
                )
                + ". Their ordering can change after only one or two predictions.",
            ]
        )

    primary_languages = sorted(
        (row for row in language_summary if row["model"] == primary_model),
        key=lambda row: row["f1_mean"],
    )
    hardest = primary_languages[:5]
    easiest = list(reversed(primary_languages[-5:]))
    lines.extend(
        [
            "",
            f"## {primary_model} Language Extremes",
            "",
            "Hardest: "
            + ", ".join(
                f"{row['language']} ({row['f1_mean']:.4f} +/- {row['f1_sample_sd']:.4f}; "
                f"gold={row['gold_mwes']})"
                for row in hardest
            )
            + ".",
            "",
            "Easiest: "
            + ", ".join(
                f"{row['language']} ({row['f1_mean']:.4f} +/- {row['f1_sample_sd']:.4f}; "
                f"gold={row['gold_mwes']})"
                for row in easiest
            )
            + ".",
        ]
    )

    breakdown_by_key = {
        (row["model"], row["dimension"], row["group"]): row
        for row in breakdown_summary
    }
    profile_groups = [
        ("continuity", "Continuous"),
        ("continuity", "Discontinuous"),
        ("token_count", "Multi-token"),
        ("token_count", "Single-token"),
        ("seen_in_train", "Seen-in-train"),
        ("seen_in_train", "Unseen-in-train"),
        ("train_identity", "Identical-to-train"),
        ("train_identity", "Variant-of-train"),
    ]
    dimension_labels = {
        "continuity": "Continuity",
        "token_count": "Expression length",
        "seen_in_train": "Seen status",
        "train_identity": "Train identity",
    }
    lines.extend(
        [
            "",
            "## Cross-Model Error Profile",
            "",
            "Values are pooled micro-F1 over all 16 languages and three seeds. They diagnose "
            "error structure but are not the macro model-selection score.",
            "",
            "| Dimension | Group | "
            + " | ".join(_markdown_escape(model) for model in model_order)
            + " | Best |",
            "| --- | --- | " + " | ".join("---:" for _ in model_order) + " | --- |",
        ]
    )
    for dimension, group in profile_groups:
        rows = [breakdown_by_key[(model, dimension, group)] for model in model_order]
        best_model = model_order[
            max(range(len(rows)), key=lambda index: rows[index]["micro_f1"])
        ]
        lines.append(
            f"| {dimension_labels[dimension]} | {group} | "
            + " | ".join(f"{row['micro_f1']:.4f}" for row in rows)
            + f" | {best_model} |"
        )

    primary_breakdowns = [row for row in breakdown_summary if row["model"] == primary_model]
    for dimension, title in [
        ("continuity", "Continuity"),
        ("token_count", "Expression Length"),
        ("seen_in_train", "Seen vs Unseen"),
        ("train_identity", "Train Identity"),
    ]:
        rows = [row for row in primary_breakdowns if row["dimension"] == dimension]
        lines.extend(
            [
                "",
                f"## {primary_model} Error Breakdown: {title}",
                "",
                "| Group | Gold | TP | FP | FN | Micro P | Micro R | Micro F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(rows, key=lambda item: item["group"]):
            lines.append(
                f"| {_markdown_escape(row['group'])} | {row['gold']} | {row['correct']} | "
                f"{row['false_positives']} | {row['false_negatives']} | "
                f"{row['micro_precision']:.4f} | {row['micro_recall']:.4f} | {row['micro_f1']:.4f} |"
            )

    categories = sorted(
        (row for row in primary_breakdowns if row["dimension"] == "category"),
        key=lambda row: (-row["false_negatives"], -row["gold"], row["group"]),
    )
    lines.extend(
        [
            "",
            f"## {primary_model} Categories with Most False Negatives",
            "",
            "| Category | Gold | TP | FP | FN | Micro F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in categories[:10]:
        lines.append(
            f"| {_markdown_escape(row['group'])} | {row['gold']} | {row['correct']} | "
            f"{row['false_positives']} | {row['false_negatives']} | {row['micro_f1']:.4f} |"
        )

    qualitative_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in qualitative_rows:
        qualitative_counts[(str(row["error_type"]), str(row["subtype"]))] += 1
    qualitative_order = [
        ("FN", "missed"),
        ("FN", "partial_overlap"),
        ("FP", "spurious"),
        ("FP", "partial_overlap"),
        ("CATEGORY_MISMATCH", "exact_span"),
    ]
    lines.extend(
        [
            "",
            f"## Qualitative Error Inventory ({qualitative_key[0]}, seed {qualitative_key[1]})",
            "",
            "| Error type | Subtype | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for error_type, subtype in qualitative_order:
        lines.append(
            f"| {error_type} | {subtype} | {qualitative_counts[(error_type, subtype)]} |"
        )
    lines.extend(
        [
            "",
            "Category mismatches have an exactly matched span, so they do not lower the official "
            "global span F1; they remain relevant to category-specific analysis.",
        ]
    )

    lines.extend(
        [
            "",
            f"## Qualitative Examples ({qualitative_key[0]}, seed {qualitative_key[1]})",
            "",
            "Examples below are deterministic samples from the three hardest languages. The complete "
            "set is available in `qualitative_errors.csv`.",
            "",
            "| Language | Error | Subtype | Category | Expression | Counterpart | Sentence |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for language_row in hardest[:3]:
        language = language_row["language"]
        for error_type in ("FN", "FP"):
            candidates = sorted(
                (
                    row
                    for row in qualitative_rows
                    if row["language"] == language and row["error_type"] == error_type
                ),
                key=lambda row: (row["subtype"], row["category"], row["sentence_index"]),
            )
            for row in candidates[:2]:
                sentence = str(row["sentence"])
                if len(sentence) > 180:
                    sentence = sentence[:177] + "..."
                lines.append(
                    f"| {language} | {error_type} | {row['subtype']} | "
                    f"{_markdown_escape(row['category'])} | {_markdown_escape(row['expression'])} | "
                    f"{_markdown_escape(row['counterpart_expression'])} | {_markdown_escape(sentence)} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Model selection must use the official macro metric, not pooled micro counts.",
            "- Three seeds describe observed variation but do not support strong significance claims.",
            "- Qualitative examples come from one preselected run and should illustrate, not estimate, prevalence.",
            "- Blind-test data must remain untouched until model and analysis decisions are frozen.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Path]:
    runs, model_order, qualitative_key = _load_manifest(manifest_path)
    primary_model = model_order[0]
    run_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []

    for spec in runs:
        evaluation_dir, run_score = _best_evaluation(spec.experiment_dir)
        run_rows.append(
            {
                "model": spec.model,
                "seed": spec.seed,
                "experiment_dir": str(spec.experiment_dir),
                "evaluation": evaluation_dir.name,
                "precision": run_score.precision,
                "recall": run_score.recall,
                "f1": run_score.f1,
            }
        )

        official_globals: dict[str, BreakdownScore] = {}
        evaluation_files = sorted(evaluation_dir.glob("*/eval.txt"))
        if not evaluation_files:
            raise FileNotFoundError(f"No per-language evaluations in {evaluation_dir}")
        for evaluation_path in evaluation_files:
            language = evaluation_path.parent.name
            scores = parse_official_breakdown(evaluation_path.read_text(encoding="utf-8"))
            global_score = _global_breakdown(scores, evaluation_path)
            official_globals[language] = global_score
            language_rows.append(
                {
                    "model": spec.model,
                    "seed": spec.seed,
                    "evaluation": evaluation_dir.name,
                    "language": language,
                    "correct": global_score.correct,
                    "predicted": global_score.predicted,
                    "gold": global_score.gold,
                    "false_positives": global_score.false_positives,
                    "false_negatives": global_score.false_negatives,
                    "precision": global_score.precision,
                    "recall": global_score.recall,
                    "f1": global_score.f1,
                }
            )
            for score in scores:
                breakdown_rows.append(
                    {
                        "model": spec.model,
                        "seed": spec.seed,
                        "evaluation": evaluation_dir.name,
                        "language": language,
                        "dimension": score.dimension,
                        "group": score.group,
                        "correct": score.correct,
                        "predicted": score.predicted,
                        "gold": score.gold,
                        "false_positives": score.false_positives,
                        "false_negatives": score.false_negatives,
                        "precision": score.precision,
                        "recall": score.recall,
                        "f1": score.f1,
                    }
                )

        if (spec.model, spec.seed) == qualitative_key:
            qualitative_rows = collect_qualitative_errors(
                model=spec.model,
                seed=spec.seed,
                evaluation_dir=evaluation_dir,
                official_globals=official_globals,
            )

    model_summary = _summarize_models(run_rows, model_order)
    language_summary = _summarize_languages(language_rows, model_order)
    breakdown_summary = _summarize_breakdowns(breakdown_rows)

    outputs = {
        "run_scores": output_dir / "run_scores.csv",
        "model_summary": output_dir / "model_summary.csv",
        "per_language_seed": output_dir / "per_language_seed.csv",
        "per_language_summary": output_dir / "per_language_summary.csv",
        "error_breakdown_seed": output_dir / "error_breakdown_seed.csv",
        "error_breakdown_summary": output_dir / "error_breakdown_summary.csv",
        "qualitative_errors": output_dir / "qualitative_errors.csv",
        "report": output_dir / "report.md",
    }
    _write_csv(
        outputs["run_scores"],
        run_rows,
        ["model", "seed", "experiment_dir", "evaluation", "precision", "recall", "f1"],
    )
    _write_csv(
        outputs["model_summary"],
        model_summary,
        [
            "model",
            "seeds",
            "run_count",
            "precision_mean",
            "precision_sample_sd",
            "recall_mean",
            "recall_sample_sd",
            "f1_mean",
            "f1_sample_sd",
        ],
    )
    _write_csv(
        outputs["per_language_seed"],
        language_rows,
        [
            "model",
            "seed",
            "evaluation",
            "language",
            "correct",
            "predicted",
            "gold",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        ],
    )
    _write_csv(
        outputs["per_language_summary"],
        language_summary,
        [
            "model",
            "language",
            "seed_count",
            "gold_mwes",
            "predicted_mwes_mean",
            "precision_mean",
            "precision_sample_sd",
            "recall_mean",
            "recall_sample_sd",
            "f1_mean",
            "f1_sample_sd",
            "f1_rank",
        ],
    )
    _write_csv(
        outputs["error_breakdown_seed"],
        breakdown_rows,
        [
            "model",
            "seed",
            "evaluation",
            "language",
            "dimension",
            "group",
            "correct",
            "predicted",
            "gold",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        ],
    )
    _write_csv(
        outputs["error_breakdown_summary"],
        breakdown_summary,
        [
            "model",
            "dimension",
            "group",
            "evaluations",
            "correct",
            "predicted",
            "gold",
            "false_positives",
            "false_negatives",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision_mean",
            "macro_recall_mean",
            "macro_f1_mean",
        ],
    )
    _write_csv(
        outputs["qualitative_errors"],
        qualitative_rows,
        [
            "model",
            "seed",
            "language",
            "sentence_index",
            "error_type",
            "subtype",
            "category",
            "token_indexes",
            "expression",
            "counterpart_category",
            "counterpart_token_indexes",
            "counterpart_expression",
            "sentence",
        ],
    )
    _report(
        path=outputs["report"],
        model_order=model_order,
        primary_model=primary_model,
        model_summary=model_summary,
        language_summary=language_summary,
        breakdown_summary=breakdown_summary,
        qualitative_rows=qualitative_rows,
        qualitative_key=qualitative_key,
    )
    return outputs


def run(args: argparse.Namespace) -> None:
    outputs = analyze(Path(args.manifest).resolve(), Path(args.output_dir).resolve())
    print(f"Analysis complete: {outputs['report']}")
    for name, path in outputs.items():
        if name != "report":
            print(f"  {name}: {path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        default=str(project_root() / "configs" / "final_analysis.json"),
        help="JSON manifest listing model, seed, and experiment directory for each run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_artifacts_dir() / "analysis" / "final-models"),
        help="Directory for CSV files and the generated Markdown report.",
    )
