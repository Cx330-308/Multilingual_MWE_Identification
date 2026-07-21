from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .cupt import available_split_languages, iter_cupt, resolve_languages, split_path
from .labels import build_label_list, collect_categories, save_labels
from .paths import default_artifacts_dir, default_data_dir


def sentence_count(path: Path) -> int:
    return sum(1 for _ in iter_cupt(path))


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    languages = resolve_languages(data_dir, args.languages)
    train_languages = available_split_languages(data_dir, languages, "train")
    dev_languages = available_split_languages(data_dir, languages, "dev")
    test_languages = available_split_languages(data_dir, languages, "test.blind")

    categories: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    for language in languages:
        split_counts[language] = {}
        for split in ("train", "dev", "test.blind"):
            path = split_path(data_dir, language, split)
            if path.exists():
                split_counts[language][split] = sentence_count(path)
                if split in args.label_splits:
                    categories.update(collect_categories(iter_cupt(path)))

    labels = build_label_list(categories.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_labels(labels, output_dir / "labels.json")
    summary = {
        "data_dir": str(data_dir),
        "languages": languages,
        "train_languages": train_languages,
        "dev_languages": dev_languages,
        "test_languages": test_languages,
        "missing_train_or_dev": [lang for lang in languages if lang not in train_languages or lang not in dev_languages],
        "label_splits": args.label_splits,
        "num_categories": len(categories),
        "num_labels": len(labels),
        "categories": dict(sorted(categories.items())),
        "sentence_counts": split_counts,
    }
    (output_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'labels.json'}")
    print(f"Wrote {output_dir / 'data_summary.json'}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Train/dev languages: {', '.join(lang for lang in languages if lang in train_languages and lang in dev_languages)}")
    if summary["missing_train_or_dev"]:
        print(f"No train/dev for: {', '.join(summary['missing_train_or_dev'])}")
    print(f"Labels: {len(labels)} ({len(categories)} MWE categories + O/B/I)")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="PARSEME 2.0 subtask1 directory.")
    parser.add_argument("--languages", default="all", help="Comma-separated ISO codes, or 'all'.")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir() / "metadata"))
    parser.add_argument(
        "--label-splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "test.blind"],
        help="Splits used to discover gold label categories.",
    )
