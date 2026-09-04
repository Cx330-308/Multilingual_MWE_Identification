#!/usr/bin/env python3
"""Generate dissertation figures from frozen CSV analysis artefacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

try:
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except ImportError as exc:  # pragma: no cover - dependency error is user-facing
    raise SystemExit(
        "reportlab is required to generate dissertation figures; "
        "install requirements-docs.txt"
    ) from exc


MODEL_ORDER = ("XLM-R", "mDeBERTa-v3", "mBERT")
MODEL_COLORS = {
    "XLM-R": colors.HexColor("#187B78"),
    "mDeBERTa-v3": colors.HexColor("#D05A47"),
    "mBERT": colors.HexColor("#385170"),
}
TEXT = colors.HexColor("#20252B")
MUTED = colors.HexColor("#687078")
GRID = colors.HexColor("#D9DEE2")
ROW = colors.HexColor("#F5F7F8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def draw_x_axis(
    page: canvas.Canvas,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    minimum: float,
    maximum: float,
    ticks: Iterable[float],
    label: str,
    tick_decimals: int = 1,
) -> None:
    page.setStrokeColor(GRID)
    page.setFillColor(MUTED)
    page.setFont("Helvetica", 8)
    for tick in ticks:
        x = left + (tick - minimum) / (maximum - minimum) * (right - left)
        page.line(x, bottom, x, top)
        page.drawCentredString(x, bottom - 13, f"{tick:.{tick_decimals}f}")
    page.setFillColor(TEXT)
    page.setFont("Helvetica", 9)
    page.drawCentredString((left + right) / 2, bottom - 30, label)


def x_position(value: float, left: float, right: float, minimum: float, maximum: float) -> float:
    return left + (value - minimum) / (maximum - minimum) * (right - left)


def draw_legend(page: canvas.Canvas, *, x: float, y: float, spacing: float = 96) -> None:
    page.setFont("Helvetica", 8.5)
    for index, model in enumerate(MODEL_ORDER):
        item_x = x + index * spacing
        page.setFillColor(MODEL_COLORS[model])
        page.circle(item_x, y + 2, 3.3, fill=1, stroke=0)
        page.setFillColor(TEXT)
        page.drawString(item_x + 7, y - 1, model)


def generate_encoder_comparison(source: Path, output: Path) -> None:
    rows = {row["model"]: row for row in read_csv(source)}
    width, height = 7.1 * inch, 3.15 * inch
    page = canvas.Canvas(str(output), pagesize=(width, height))
    page.setTitle("Encoder development macro-F1 comparison")

    left, right = 118, width - 35
    bottom, top = 53, height - 37
    minimum, maximum = 0.62, 0.72
    draw_x_axis(
        page,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        minimum=minimum,
        maximum=maximum,
        ticks=(0.62, 0.64, 0.66, 0.68, 0.70, 0.72),
        label="Official exact MWE macro-F1",
        tick_decimals=2,
    )

    page.setFont("Helvetica", 10)
    row_gap = (top - bottom) / len(MODEL_ORDER)
    for index, model in enumerate(MODEL_ORDER):
        row = rows[model]
        mean = float(row["f1_mean"])
        sd = float(row["f1_sample_sd"])
        y = top - (index + 0.5) * row_gap
        page.setFillColor(TEXT)
        page.drawRightString(left - 13, y - 3, model)
        x_mean = x_position(mean, left, right, minimum, maximum)
        x_low = x_position(max(minimum, mean - sd), left, right, minimum, maximum)
        x_high = x_position(min(maximum, mean + sd), left, right, minimum, maximum)
        page.setStrokeColor(MODEL_COLORS[model])
        page.setLineWidth(2)
        page.line(x_low, y, x_high, y)
        page.setLineWidth(1)
        page.line(x_low, y - 5, x_low, y + 5)
        page.line(x_high, y - 5, x_high, y + 5)
        page.setFillColor(MODEL_COLORS[model])
        page.circle(x_mean, y, 5, fill=1, stroke=0)
        page.setFillColor(TEXT)
        page.setFont("Helvetica-Bold", 8.5)
        page.drawString(min(x_high + 7, right - 45), y - 3, f"{mean:.4f}")

    page.setFillColor(MUTED)
    page.setFont("Helvetica", 7.5)
    page.drawRightString(right, height - 16, "Points: mean; whiskers: sample SD across three seeds")
    page.save()


def generate_per_language(source: Path, output: Path) -> None:
    rows = read_csv(source)
    by_language: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_language.setdefault(row["language"], {})[row["model"]] = row
    languages = sorted(
        by_language,
        key=lambda language: float(by_language[language]["XLM-R"]["f1_mean"]),
    )

    width, height = 7.1 * inch, 8.35 * inch
    page = canvas.Canvas(str(output), pagesize=(width, height))
    page.setTitle("Per-language development macro-F1")
    left, right = 98, width - 28
    bottom, top = 48, height - 53
    minimum, maximum = 0.0, 0.9
    draw_x_axis(
        page,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        minimum=minimum,
        maximum=maximum,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        label="Exact MWE F1",
    )
    draw_legend(page, x=left + 52, y=height - 28, spacing=112)

    row_gap = (top - bottom) / len(languages)
    offsets = {"XLM-R": 6.2, "mDeBERTa-v3": 0, "mBERT": -6.2}
    for index, language in enumerate(languages):
        y = top - (index + 0.5) * row_gap
        if index % 2 == 0:
            page.setFillColor(ROW)
            page.rect(left, y - row_gap / 2, right - left, row_gap, fill=1, stroke=0)
        support = by_language[language]["XLM-R"]["gold_mwes"]
        page.setFillColor(TEXT)
        page.setFont("Helvetica", 8.2)
        page.drawRightString(left - 9, y - 2.5, f"{language} (n={support})")
        for model in MODEL_ORDER:
            row = by_language[language][model]
            mean = float(row["f1_mean"])
            sd = float(row["f1_sample_sd"])
            item_y = y + offsets[model]
            x_mean = x_position(mean, left, right, minimum, maximum)
            x_low = x_position(max(minimum, mean - sd), left, right, minimum, maximum)
            x_high = x_position(min(maximum, mean + sd), left, right, minimum, maximum)
            page.setStrokeColor(MODEL_COLORS[model])
            page.setLineWidth(0.8)
            page.line(x_low, item_y, x_high, item_y)
            page.setFillColor(MODEL_COLORS[model])
            page.circle(x_mean, item_y, 2.6, fill=1, stroke=0)

    page.setFillColor(MUTED)
    page.setFont("Helvetica", 7.3)
    page.drawRightString(right, 10, "Languages ordered by XLM-R mean; n is the number of gold development MWEs")
    page.save()


def generate_phenomenon_profile(source: Path, output: Path) -> None:
    wanted = (
        ("continuity", "Continuous"),
        ("continuity", "Discontinuous"),
        ("token_count", "Multi-token"),
        ("token_count", "Single-token"),
        ("seen_in_train", "Seen-in-train"),
        ("seen_in_train", "Unseen-in-train"),
        ("train_identity", "Identical-to-train"),
        ("train_identity", "Variant-of-train"),
    )
    labels = {
        ("continuity", "Continuous"): "Continuous",
        ("continuity", "Discontinuous"): "Discontinuous",
        ("token_count", "Multi-token"): "Multi-token",
        ("token_count", "Single-token"): "Single-token",
        ("seen_in_train", "Seen-in-train"): "Seen in train",
        ("seen_in_train", "Unseen-in-train"): "Unseen in train",
        ("train_identity", "Identical-to-train"): "Identical to train",
        ("train_identity", "Variant-of-train"): "Variant of train",
    }
    rows = read_csv(source)
    values = {
        (row["dimension"], row["group"], row["model"]): float(row["micro_f1"])
        for row in rows
    }

    width, height = 7.1 * inch, 5.35 * inch
    page = canvas.Canvas(str(output), pagesize=(width, height))
    page.setTitle("Phenomenon-level error profile")
    left, right = 112, width - 28
    bottom, top = 48, height - 55
    minimum, maximum = 0.0, 1.0
    draw_x_axis(
        page,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        minimum=minimum,
        maximum=maximum,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        label="Pooled micro-F1 diagnostic",
    )
    draw_legend(page, x=left + 48, y=height - 29, spacing=112)

    row_gap = (top - bottom) / len(wanted)
    offsets = {"XLM-R": 5.5, "mDeBERTa-v3": 0, "mBERT": -5.5}
    for index, key in enumerate(wanted):
        y = top - (index + 0.5) * row_gap
        if index % 2 == 0:
            page.setFillColor(ROW)
            page.rect(left, y - row_gap / 2, right - left, row_gap, fill=1, stroke=0)
        page.setFillColor(TEXT)
        page.setFont("Helvetica", 8.5)
        page.drawRightString(left - 9, y - 2.5, labels[key])
        for model in MODEL_ORDER:
            value = values[(key[0], key[1], model)]
            x = x_position(value, left, right, minimum, maximum)
            page.setFillColor(MODEL_COLORS[model])
            page.circle(x, y + offsets[model], 3, fill=1, stroke=0)

    page.setFillColor(MUTED)
    page.setFont("Helvetica", 7.3)
    page.drawRightString(right, 10, "Counts pooled over 16 languages and three seeds; not the model-selection metric")
    page.save()


def generate_blind_shift(source: Path, output: Path) -> None:
    rows = [row for row in read_csv(source) if row["dev_f1"]]
    rows.sort(key=lambda row: float(row["f1_delta"]))

    width, height = 7.1 * inch, 7.25 * inch
    page = canvas.Canvas(str(output), pagesize=(width, height))
    page.setTitle("Development to blind-test per-language F1 shift")
    left, right = 67, width - 76
    bottom, top = 48, height - 56
    minimum, maximum = 0.0, 0.9
    draw_x_axis(
        page,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        minimum=minimum,
        maximum=maximum,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        label="Exact MWE F1",
    )

    page.setFont("Helvetica", 8.5)
    page.setFillColor(colors.HexColor("#555B61"))
    page.circle(left + 102, height - 28, 3.3, fill=1, stroke=0)
    page.setFillColor(TEXT)
    page.drawString(left + 110, height - 31, "Development")
    page.setFillColor(colors.HexColor("#D05A47"))
    page.circle(left + 211, height - 28, 3.3, fill=1, stroke=0)
    page.setFillColor(TEXT)
    page.drawString(left + 219, height - 31, "Blind test")

    row_gap = (top - bottom) / len(rows)
    for index, row in enumerate(rows):
        y = top - (index + 0.5) * row_gap
        if index % 2 == 0:
            page.setFillColor(ROW)
            page.rect(left, y - row_gap / 2, right - left, row_gap, fill=1, stroke=0)
        dev = float(row["dev_f1"])
        test = float(row["test_f1"])
        x_dev = x_position(dev, left, right, minimum, maximum)
        x_test = x_position(test, left, right, minimum, maximum)
        page.setStrokeColor(colors.HexColor("#AAB0B5"))
        page.setLineWidth(1.2)
        page.line(x_dev, y, x_test, y)
        page.setFillColor(colors.HexColor("#555B61"))
        page.circle(x_dev, y, 3.1, fill=1, stroke=0)
        page.setFillColor(colors.HexColor("#D05A47"))
        page.circle(x_test, y, 3.1, fill=1, stroke=0)
        page.setFillColor(TEXT)
        page.setFont("Helvetica", 8.3)
        page.drawRightString(left - 9, y - 2.5, row["language"])
        delta = float(row["f1_delta"])
        page.setFont("Helvetica", 7.3)
        page.setFillColor(colors.HexColor("#B14132") if delta < 0 else colors.HexColor("#187B78"))
        page.drawRightString(width - 10, y - 2.5, f"{delta:+.3f}")

    page.setFillColor(MUTED)
    page.setFont("Helvetica", 7.3)
    page.drawRightString(width - 10, 10, "Seed-13 development run versus blind test; right-hand labels show test minus development")
    page.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing artifacts/ and dissertation/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    analysis = root / "artifacts" / "analysis"
    output = root / "dissertation" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    jobs = (
        (
            generate_encoder_comparison,
            analysis / "final-models" / "model_summary.csv",
            output / "encoder_macro_f1.pdf",
        ),
        (
            generate_per_language,
            analysis / "final-models" / "per_language_summary.csv",
            output / "per_language_f1.pdf",
        ),
        (
            generate_phenomenon_profile,
            analysis / "final-models" / "error_breakdown_summary.csv",
            output / "phenomenon_error_profile.pdf",
        ),
        (
            generate_blind_shift,
            analysis / "codabench-884918" / "dev_test_comparison.csv",
            output / "blind_language_shift.pdf",
        ),
    )
    for generator, source, destination in jobs:
        generator(source, destination)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
