#!/usr/bin/env python3
"""Render aggregate result CSV files as LaTeX table fragments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


RAW_PREFIX = "__RAW__"


@dataclass(frozen=True)
class TableSpec:
    caption: str
    label: str
    size: str = r"\footnotesize"


TABLE_SPECS = {
    "table_1_experimental_conditions": TableSpec(
        caption="Experimental conditions used in the promoted comparison matrix.",
        label="tab:conditions",
    ),
    "table_2_main_validation_results": TableSpec(
        caption=(
            "Main validation results. Bootstrap intervals are case-level 95\\% percentile intervals. "
            "Boldface marks the numerically best value among promoted rows for the indicated metric and "
            "does not imply statistical significance."
        ),
        label="tab:validation",
        size=r"\scriptsize",
    ),
    "table_3_runtime_reliability": TableSpec(
        caption="Runtime and peak GPU memory summary for promoted rows.",
        label="tab:runtime",
    ),
    "table_4_category_leakage_summary": TableSpec(
        caption="Category leakage summary after filtering to promoted validation rows.",
        label="tab:category",
    ),
    "table_5_stress_set_results": TableSpec(
        caption="Synthetic local-vault stress-set results.",
        label="tab:stress",
    ),
    "table_6_opf_rules_delta": TableSpec(
        caption="Deterministic safety-rule deltas measured relative to OPF-D.",
        label="tab:rules-delta",
    ),
    "table_7_pipeline_ablation": TableSpec(
        caption=(
            "OPF decode and deterministic safety-pass ablation on validation. These rows are "
            "OPF-family conditions only; direct decoder LLM rows are reported in the main validation "
            "and stress tables."
        ),
        label="tab:pipeline-ablation",
    ),
    "table_8_failure_examples": TableSpec(
        caption="Representative promoted-system failure and tradeoff patterns.",
        label="tab:failure-examples",
        size=r"\scriptsize",
    ),
    "table_9_supporting_split_consistency": TableSpec(
        caption="Supporting split consistency evidence for rows with complete provenance.",
        label="tab:supporting",
    ),
}


def escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_cell(value: str) -> str:
    if value.startswith(RAW_PREFIX):
        return value[len(RAW_PREFIX) :]
    return escape_tex(value)


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0], rows[1:]


def column_spec(width: int) -> str:
    return "l" + ("r" * max(width - 1, 0))


def render_table(csv_path: Path, spec: TableSpec | None = None) -> str:
    header, rows = read_csv(csv_path)
    table_spec = spec or TableSpec(caption=f"Aggregate results from {escape_tex(csv_path.name)}.", label=f"tab:{csv_path.stem}")
    col_spec = column_spec(len(header))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        table_spec.size,
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(render_cell(item) for item in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append(" & ".join(render_cell(item) for item in padded[: len(header)]) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{table_spec.caption}}}",
            rf"\label{{{table_spec.label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results_summary"), help="Directory containing table_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_tables"), help="Directory for rendered .tex files.")
    parser.add_argument("--table", action="append", help="Optional table CSV stem to render. May be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.table:
        csv_paths = [args.input_dir / f"{stem}.csv" for stem in args.table]
    else:
        csv_paths = sorted(args.input_dir.glob("table_*.csv"))

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        content = render_table(csv_path, TABLE_SPECS.get(csv_path.stem))
        output_path = args.output_dir / f"{csv_path.stem}.tex"
        output_path.write_text(content, encoding="utf-8")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
