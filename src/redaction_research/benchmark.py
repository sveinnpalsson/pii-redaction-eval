"""Benchmark harness for prepared redaction fixtures."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redaction_research.metrics import RedactionSpan, compute_case_metrics, summarize_case_metrics

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.I)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
URL_PATTERN = re.compile(r"\bhttps?://[^\s)]+", flags=re.I)
LONG_DIGITS_PATTERN = re.compile(r"\b\d{12,19}\b")

DEFAULT_LABEL_ORDER = ("EMAIL", "PHONE", "URL", "ACCOUNT")


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    source_type: str
    text: str
    expected_redacted_text: str
    expected_placeholders: list[str]
    expected_spans: list[RedactionSpan]


@dataclass(slots=True)
class BenchmarkCaseResult:
    case_id: str
    source_type: str
    mode: str
    actual_redacted_text: str
    metrics: dict[str, Any]


@dataclass(slots=True)
class BenchmarkRunResult:
    mode: str
    fixture_path: str
    summary: dict[str, Any]
    cases: list[dict[str, Any]]


def load_fixture(path: Path) -> list[BenchmarkCase]:
    rows: list[BenchmarkCase] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                BenchmarkCase(
                    case_id=str(raw["case_id"]),
                    source_type=str(raw["source_type"]),
                    text=str(raw["text"]),
                    expected_redacted_text=str(raw.get("expected_redacted_text", "")),
                    expected_placeholders=list(raw.get("expected_placeholders", [])),
                    expected_spans=[
                        RedactionSpan(
                            start=int(span["start"]),
                            end=int(span["end"]),
                            label=str(span["label"]),
                            placeholder=str(span["placeholder"]),
                        )
                        for span in raw.get("expected_spans", [])
                    ],
                )
            )
    return rows


def _placeholder(label: str, ordinal: int) -> str:
    return f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"


def regex_redact(text: str) -> str:
    redacted = text
    patterns = [
        (EMAIL_PATTERN, "EMAIL"),
        (PHONE_PATTERN, "PHONE"),
        (URL_PATTERN, "URL"),
        (LONG_DIGITS_PATTERN, "ACCOUNT"),
    ]
    for pattern, label in patterns:
        ordinal = 0
        while True:
            match = pattern.search(redacted)
            if not match:
                break
            placeholder = _placeholder(label, ordinal)
            redacted = redacted[: match.start()] + placeholder + redacted[match.end() :]
            ordinal += 1
    return redacted


def evaluate_fixture(path: Path, mode: str) -> BenchmarkRunResult:
    fixture = load_fixture(path)
    case_results: list[BenchmarkCaseResult] = []
    metric_rows = []

    for case in fixture:
        if mode == "regex":
            actual_redacted_text = regex_redact(case.text)
        elif mode == "expected":
            actual_redacted_text = case.expected_redacted_text
        elif mode == "passthrough":
            actual_redacted_text = case.text
        else:
            raise ValueError(f"unsupported mode: {mode}")

        metrics = compute_case_metrics(
            case_id=case.case_id,
            source_text=case.text,
            expected_spans=case.expected_spans,
            actual_redacted_text=actual_redacted_text,
        )
        metric_rows.append(metrics)
        case_results.append(
            BenchmarkCaseResult(
                case_id=case.case_id,
                source_type=case.source_type,
                mode=mode,
                actual_redacted_text=actual_redacted_text,
                metrics=metrics.to_dict(),
            )
        )

    summary = summarize_case_metrics(metric_rows)
    return BenchmarkRunResult(
        mode=mode,
        fixture_path=str(path),
        summary=summary.to_dict(),
        cases=[asdict(item) for item in case_results],
    )


def write_run_result(result: BenchmarkRunResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
