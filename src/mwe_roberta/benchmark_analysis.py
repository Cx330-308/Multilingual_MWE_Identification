from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .paths import default_artifacts_dir, project_root


PHENOMENON_TABLES = {
    "Discontinuous vs Continuous VMWEs:": (
        "continuity",
        "Discontinuous",
        "Continuous",
    ),
    "Unseen-in-train vs Seen-in-train VMWEs:": (
        "seen_in_train",
        "Unseen-in-train",
        "Seen-in-train",
    ),
    "Variant-of-train vs Identical-to-train VMWEs:": (
        "train_identity",
        "Variant-of-train",
        "Identical-to-train",
    ),
    "Single-token vs Multi-token VMWEs:": (
        "token_count",
        "Single-token",
        "Multi-token",
    ),
}


@dataclass(frozen=True)
class HTMLTable:
    section: str
    title: str
    rows: list[list[str]]


@dataclass(frozen=True)
class BenchmarkResults:
    global_score: dict[str, Any]
    phenomena: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    categories: list[dict[str, Any]]


class _DetailedResultsHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.section = ""
        self.title = ""
        self.tables: list[HTMLTable] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._table_rows: list[list[str]] | None = None
        self._in_tbody = False
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    @staticmethod
    def _text(parts: Sequence[str]) -> str:
        return " ".join(" ".join(parts).split())

    def _finish_heading(self) -> None:
        if self._heading_tag is None:
            return
        value = self._text(self._heading_parts)
        if self._heading_tag == "h1":
            self.section = value
            self.title = ""
        else:
            self.title = value
        self._heading_tag = None
        self._heading_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2"}:
            self._finish_heading()
            self._heading_tag = tag
            self._heading_parts = []
        elif tag == "table":
            self._finish_heading()
            self._table_rows = []
        elif tag == "tbody" and self._table_rows is not None:
            self._in_tbody = True
        elif tag == "tr" and self._in_tbody:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._heading_tag is not None:
            self._heading_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2"} and self._heading_tag is not None:
            # The Codabench template closes its h1 elements with </h2>.
            self._finish_heading()
        elif tag == "td" and self._cell_parts is not None and self._row is not None:
            self._row.append(self._text(self._cell_parts))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if self._row and self._table_rows is not None:
                self._table_rows.append(self._row)
            self._row = None
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "table" and self._table_rows is not None:
            self.tables.append(HTMLTable(self.section, self.title, self._table_rows))
            self._table_rows = None


def _score(value: str) -> float:
    return float(value) / 100.0


def _single_table(tables: Sequence[HTMLTable], section: str, title: str) -> HTMLTable:
    matches = [table for table in tables if table.section == section and table.title == title]
    if len(matches) != 1:
        raise ValueError(f"Expected one table for {section!r}/{title!r}, found {len(matches)}.")
    return matches[0]


def parse_detailed_results(text: str) -> BenchmarkResults:
    parser = _DetailedResultsHTMLParser()
    parser.feed(text)
    parser._finish_heading()

    global_table = _single_table(parser.tables, "Cross-lingual macro-averages", "General ranking:")
    if len(global_table.rows) != 1 or len(global_table.rows[0]) != 7:
        raise ValueError("Unexpected global ranking table shape.")
    global_values = global_table.rows[0]
    global_score = {
        "language_coverage": global_values[0],
        "mwe_precision": _score(global_values[1]),
        "mwe_recall": _score(global_values[2]),
        "mwe_f1": _score(global_values[3]),
        "token_precision": _score(global_values[4]),
        "token_recall": _score(global_values[5]),
        "token_f1": _score(global_values[6]),
    }

    phenomena: list[dict[str, Any]] = []
    for title, (dimension, first_group, second_group) in PHENOMENON_TABLES.items():
        table = _single_table(parser.tables, "Cross-lingual macro-averages", title)
        if len(table.rows) != 1 or len(table.rows[0]) != 8:
            raise ValueError(f"Unexpected phenomenon table shape for {title}")
        values = table.rows[0]
        for offset, group in ((0, first_group), (4, second_group)):
            phenomena.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "language_coverage": values[offset],
                    "precision": _score(values[offset + 1]),
                    "recall": _score(values[offset + 2]),
                    "f1": _score(values[offset + 3]),
                }
            )

    language_table = _single_table(parser.tables, "Language-specific system ranking", "")
    languages: list[dict[str, Any]] = []
    for values in language_table.rows:
        if len(values) != 13:
            raise ValueError(f"Unexpected language row shape: {values}")
        languages.append(
            {
                "language": values[0],
                "index_richness": float(values[1]),
                "shannon_entropy": float(values[2]),
                "shannon_evenness": float(values[3]),
                "unseen_precision": _score(values[4]),
                "unseen_recall": _score(values[5]),
                "unseen_f1": _score(values[6]),
                "mwe_precision": _score(values[7]),
                "mwe_recall": _score(values[8]),
                "mwe_f1": _score(values[9]),
                "token_precision": _score(values[10]),
                "token_recall": _score(values[11]),
                "token_f1": _score(values[12]),
            }
        )

    categories: list[dict[str, Any]] = []
    category_tables = [
        table for table in parser.tables if table.section == "Results per VMWE category"
    ]
    for table in category_tables:
        for values in table.rows:
            if len(values) != 7:
                raise ValueError(f"Unexpected category row shape for {table.title}: {values}")
            categories.append(
                {
                    "language": table.title,
                    "category": values[0],
                    "mwe_precision": _score(values[1]),
                    "mwe_recall": _score(values[2]),
                    "mwe_f1": _score(values[3]),
                    "token_precision": _score(values[4]),
                    "token_recall": _score(values[5]),
                    "token_f1": _score(values[6]),
                }
            )

    return BenchmarkResults(global_score, phenomena, languages, categories)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metric(rows: Sequence[dict[str, Any]], metric: str) -> float:
    return mean(float(row[metric]) for row in rows)


def _harmonic_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _markdown_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _build_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    results: BenchmarkResults,
    comparisons: Sequence[dict[str, Any]],
    phenomenon_comparisons: Sequence[dict[str, Any]],
) -> None:
    snapshot = manifest["leaderboard_snapshot"]
    supervised = [row for row in comparisons if row["dev_f1"] is not None]
    dev_precision = _mean_metric(supervised, "dev_precision")
    dev_recall = _mean_metric(supervised, "dev_recall")
    dev_f1 = _harmonic_f1(dev_precision, dev_recall)
    test_16_precision = _mean_metric(supervised, "test_precision")
    test_16_recall = _mean_metric(supervised, "test_recall")
    test_16_f1 = _harmonic_f1(test_16_precision, test_16_recall)
    grc = next(row for row in comparisons if row["language"] == "GRC")
    leader = snapshot["leader"]

    ranked_languages = sorted(comparisons, key=lambda row: row["test_f1"])
    largest_drops = sorted(supervised, key=lambda row: row["f1_delta"])[:5]
    improvements = sorted(
        (row for row in supervised if row["f1_delta"] > 0),
        key=lambda row: row["f1_delta"],
        reverse=True,
    )

    lines = [
        "# Codabench Blind-Test Analysis",
        "",
        f"Submission `{manifest['submission_id']}` uses `{manifest['model_name']}`. "
        f"The leaderboard snapshot was captured at {snapshot['captured_at']}; open-benchmark "
        "ranks can change after later submissions.",
        "",
        "## Leaderboard Snapshot",
        "",
        "| Measure | This submission | Rank | Current leader | Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Exact MWE F1 | {results.global_score['mwe_f1']:.4f} | "
        f"{snapshot['mwe_f1_rank']}/{snapshot['leaderboard_entries']} | "
        f"{leader['mwe_f1']:.4f} | {results.global_score['mwe_f1'] - leader['mwe_f1']:+.4f} |",
        f"| Token F1 | {results.global_score['token_f1']:.4f} | "
        f"{snapshot['token_f1_rank']}/{snapshot['leaderboard_entries']} | "
        f"{leader['token_f1']:.4f} | {results.global_score['token_f1'] - leader['token_f1']:+.4f} |",
        "",
        "## Development to Blind Test",
        "",
        "| Scope | Languages | Precision | Recall | Exact MWE F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Development, supervised languages | 16 | {dev_precision:.4f} | "
        f"{dev_recall:.4f} | {dev_f1:.4f} |",
        f"| Blind test, same supervised languages | 16 | {test_16_precision:.4f} | "
        f"{test_16_recall:.4f} | {test_16_f1:.4f} |",
        f"| Blind test, full benchmark | 17 | {results.global_score['mwe_precision']:.4f} | "
        f"{results.global_score['mwe_recall']:.4f} | {results.global_score['mwe_f1']:.4f} |",
        f"| Ancient Greek zero-shot | 1 | {grc['test_precision']:.4f} | "
        f"{grc['test_recall']:.4f} | {grc['test_f1']:.4f} |",
        "",
        f"Across the same 16 languages, macro F1 changes by {test_16_f1 - dev_f1:+.4f}. "
        f"Adding zero-shot Ancient Greek changes the full macro score by "
        f"{results.global_score['mwe_f1'] - test_16_f1:+.4f}.",
        "",
        "## Phenomenon Comparison",
        "",
        "Development values use the preselected seed-13 run. Blind-test values are the "
        "Codabench cross-language macro scores and may cover a different number of languages.",
        "",
        "| Dimension | Group | Dev F1 | Test F1 | Delta | Test coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in phenomenon_comparisons:
        lines.append(
            f"| {row['dimension']} | {row['group']} | {row['dev_f1']:.4f} | "
            f"{row['test_f1']:.4f} | {row['f1_delta']:+.4f} | "
            f"{row['test_language_coverage']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Language Exact MWE F1",
            "",
            "| Language | Development | Blind test | Delta | Unseen test F1 | Token test F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ranked_languages:
        lines.append(
            f"| {row['language']} | {_markdown_score(row['dev_f1'])} | "
            f"{row['test_f1']:.4f} | {_markdown_score(row['f1_delta'])} | "
            f"{row['test_unseen_f1']:.4f} | {row['test_token_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Largest Supervised-Language Changes",
            "",
            "Largest drops: "
            + ", ".join(
                f"{row['language']} ({row['f1_delta']:+.4f})" for row in largest_drops
            )
            + ".",
            "",
            "Languages improving: "
            + ", ".join(
                f"{row['language']} ({row['f1_delta']:+.4f})" for row in improvements
            )
            + ".",
            "",
            "## Interpretation Guardrails",
            "",
            "- The blind test is for final evaluation, not a new hyperparameter-selection set.",
            "- Development and test deltas for very small development sets can be unstable.",
            "- Per-category tables lack support counts; category F1 should not be averaged blindly.",
            "- The complete per-category export is available in `blind_categories.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = (manifest_path.parent / manifest.get("base_dir", ".")).resolve()
    scoring_dir = (base_dir / manifest["scoring_dir"]).resolve()
    dev_language_csv = (base_dir / manifest["dev_language_csv"]).resolve()
    dev_breakdown_csv = (base_dir / manifest["dev_breakdown_csv"]).resolve()

    results = parse_detailed_results(
        (scoring_dir / "detailed_results.html").read_text(encoding="utf-8")
    )
    score_payload = json.loads((scoring_dir / "scores.json").read_text(encoding="utf-8"))
    if abs(results.global_score["mwe_f1"] - score_payload["Global MWE-based F"] / 100) > 5e-4:
        raise ValueError("HTML and scores.json global MWE F1 values disagree.")

    model = str(manifest["dev_model"])
    seed = str(manifest["dev_seed"])
    dev_languages = {
        row["language"]: row
        for row in _read_csv(dev_language_csv)
        if row["model"] == model and row["seed"] == seed
    }
    if len(dev_languages) != 16:
        raise ValueError(f"Expected 16 development languages, found {len(dev_languages)}.")

    comparisons: list[dict[str, Any]] = []
    for test_row in results.languages:
        dev_row = dev_languages.get(test_row["language"])
        dev_precision = float(dev_row["precision"]) if dev_row else None
        dev_recall = float(dev_row["recall"]) if dev_row else None
        dev_f1 = float(dev_row["f1"]) if dev_row else None
        comparisons.append(
            {
                "language": test_row["language"],
                "dev_precision": dev_precision,
                "dev_recall": dev_recall,
                "dev_f1": dev_f1,
                "test_precision": test_row["mwe_precision"],
                "test_recall": test_row["mwe_recall"],
                "test_f1": test_row["mwe_f1"],
                "f1_delta": test_row["mwe_f1"] - dev_f1 if dev_f1 is not None else None,
                "test_unseen_precision": test_row["unseen_precision"],
                "test_unseen_recall": test_row["unseen_recall"],
                "test_unseen_f1": test_row["unseen_f1"],
                "test_token_precision": test_row["token_precision"],
                "test_token_recall": test_row["token_recall"],
                "test_token_f1": test_row["token_f1"],
                "index_richness": test_row["index_richness"],
                "shannon_entropy": test_row["shannon_entropy"],
                "shannon_evenness": test_row["shannon_evenness"],
            }
        )

    dev_breakdowns: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in _read_csv(dev_breakdown_csv):
        if row["model"] == model and row["seed"] == seed:
            dev_breakdowns[(row["dimension"], row["group"])].append(
                (float(row["precision"]), float(row["recall"]))
            )

    phenomenon_comparisons: list[dict[str, Any]] = []
    for test_row in results.phenomena:
        key = (test_row["dimension"], test_row["group"])
        dev_scores = dev_breakdowns.get(key, [])
        if not dev_scores:
            raise ValueError(f"No development phenomenon rows for {key}.")
        dev_precision = mean(score[0] for score in dev_scores)
        dev_recall = mean(score[1] for score in dev_scores)
        dev_f1 = _harmonic_f1(dev_precision, dev_recall)
        phenomenon_comparisons.append(
            {
                "dimension": test_row["dimension"],
                "group": test_row["group"],
                "dev_language_count": len(dev_scores),
                "dev_precision": dev_precision,
                "dev_recall": dev_recall,
                "dev_f1": dev_f1,
                "test_language_coverage": test_row["language_coverage"],
                "test_precision": test_row["precision"],
                "test_recall": test_row["recall"],
                "test_f1": test_row["f1"],
                "f1_delta": test_row["f1"] - dev_f1,
            }
        )

    outputs = {
        "blind_global": output_dir / "blind_global.csv",
        "blind_phenomena": output_dir / "blind_phenomena.csv",
        "blind_per_language": output_dir / "blind_per_language.csv",
        "blind_categories": output_dir / "blind_categories.csv",
        "dev_test_comparison": output_dir / "dev_test_comparison.csv",
        "phenomenon_comparison": output_dir / "phenomenon_comparison.csv",
        "report": output_dir / "report.md",
    }
    global_row = {
        "submission_id": manifest["submission_id"],
        "model_name": manifest["model_name"],
        **results.global_score,
    }
    _write_csv(outputs["blind_global"], [global_row], list(global_row))
    _write_csv(
        outputs["blind_phenomena"],
        results.phenomena,
        ["dimension", "group", "language_coverage", "precision", "recall", "f1"],
    )
    _write_csv(outputs["blind_per_language"], results.languages, list(results.languages[0]))
    _write_csv(outputs["blind_categories"], results.categories, list(results.categories[0]))
    _write_csv(outputs["dev_test_comparison"], comparisons, list(comparisons[0]))
    _write_csv(
        outputs["phenomenon_comparison"],
        phenomenon_comparisons,
        list(phenomenon_comparisons[0]),
    )
    _build_report(
        path=outputs["report"],
        manifest=manifest,
        results=results,
        comparisons=comparisons,
        phenomenon_comparisons=phenomenon_comparisons,
    )
    return outputs


def run(args: argparse.Namespace) -> None:
    outputs = analyze(Path(args.manifest).resolve(), Path(args.output_dir).resolve())
    print(f"Benchmark analysis complete: {outputs['report']}")
    for name, path in outputs.items():
        if name != "report":
            print(f"  {name}: {path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        default=str(project_root() / "configs" / "codabench_884918.json"),
        help="JSON manifest describing the Codabench submission and local score files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_artifacts_dir() / "analysis" / "codabench-884918"),
        help="Directory for blind-test CSV files and the generated Markdown report.",
    )
