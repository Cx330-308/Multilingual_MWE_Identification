from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .cupt import CuptSentence

O_LABEL = "O"


def _split_mwe_field(field: str) -> list[str]:
    if not field or field == "*":
        return []
    return [part for part in field.split(";") if part and part != "*"]


def sentence_mwe_categories(sentence: CuptSentence) -> dict[str, str]:
    categories: dict[str, str] = {}
    for field in sentence.mwe_columns:
        for part in _split_mwe_field(field):
            if ":" in part:
                mwe_id, category = part.split(":", 1)
                categories.setdefault(mwe_id, category)
    return categories


def mwe_columns_to_bio(sentence: CuptSentence) -> list[str]:
    categories = sentence_mwe_categories(sentence)
    seen_ids: set[str] = set()
    labels: list[str] = []
    for field in sentence.mwe_columns:
        parts = _split_mwe_field(field)
        if not parts:
            labels.append(O_LABEL)
            continue

        primary = parts[0]
        if ":" in primary:
            mwe_id, category = primary.split(":", 1)
        else:
            mwe_id = primary
            category = categories.get(mwe_id, "MWE")

        prefix = "I" if mwe_id in seen_ids else "B"
        seen_ids.add(mwe_id)
        labels.append(f"{prefix}-{category}")
    return labels


def bio_to_mwe_columns(labels: Sequence[str]) -> list[str]:
    next_id = 1
    active_by_category: dict[str, int] = {}
    columns: list[str] = []

    for label in labels:
        if label == O_LABEL or "-" not in label:
            columns.append("*")
            continue

        prefix, category = label.split("-", 1)
        if prefix == "B" or category not in active_by_category:
            mwe_id = next_id
            next_id += 1
            active_by_category[category] = mwe_id
            columns.append(f"{mwe_id}:{category}")
        else:
            columns.append(str(active_by_category[category]))
    return columns


def collect_categories(sentences: Iterable[CuptSentence]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for category in sentence_mwe_categories(sentence).values():
            counts[category] += 1
    return counts


def build_label_list(categories: Iterable[str]) -> list[str]:
    labels = [O_LABEL]
    for category in sorted(set(categories)):
        labels.append(f"B-{category}")
        labels.append(f"I-{category}")
    return labels


def save_labels(labels: Sequence[str], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    label2id = {label: idx for idx, label in enumerate(labels)}
    payload = {"labels": list(labels), "label2id": label2id, "id2label": {idx: label for label, idx in label2id.items()}}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_labels(path: Path | str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["labels"])
