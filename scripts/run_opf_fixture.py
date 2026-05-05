#!/usr/bin/env python3
"""Run OpenAI Privacy Filter on a repo fixture and score with local metrics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from redaction_research.benchmark import load_fixture
from redaction_research.metrics import RedactionSpan, compute_case_metrics, summarize_case_metrics
from redaction_research.safety_rules import SAFETY_RULE_VERSION, add_safety_rule_spans


OPF_LABEL_MAP = {
    "private_person": "PERSON",
    "private_address": "ADDRESS",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_url": "URL",
    "account_number": "ACCOUNT",
    "secret": "CUSTOM",
    "private_date": "DATE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI Privacy Filter on a redaction fixture.")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-root", default="results/opf")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", help="Optional OPF checkpoint directory. Defaults to OPF_CHECKPOINT or ~/.opf/privacy_filter.")
    parser.add_argument("--context-window-length", type=int)
    parser.add_argument("--decode-mode", choices=("viterbi", "argmax"), default="viterbi")
    parser.add_argument("--viterbi-calibration-path", help="Optional OPF Viterbi calibration artifact.")
    parser.add_argument("--discard-overlapping-predicted-spans", action="store_true")
    parser.add_argument("--safety-pass", action="store_true", help="Add deterministic high-risk safety-rule spans after OPF.")
    parser.add_argument("--limit", type=int, help="Optional first-N case limit.")
    return parser.parse_args()


def _placeholder(label: str, ordinal: int) -> str:
    return f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"


def _span_from_opf(raw_span: Any) -> tuple[int, int, str, str]:
    if hasattr(raw_span, "label"):
        return (
            int(raw_span.start),
            int(raw_span.end),
            str(raw_span.label),
            str(raw_span.text),
        )
    return (
        int(raw_span["start"]),
        int(raw_span["end"]),
        str(raw_span["label"]),
        str(raw_span.get("text", "")),
    )


def _non_overlapping_spans(spans: list[RedactionSpan]) -> list[RedactionSpan]:
    accepted: list[RedactionSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start), item.label)):
        if any(not (span.end <= kept.start or span.start >= kept.end) for kept in accepted):
            continue
        accepted.append(span)
    return accepted


def _render_redacted_text(text: str, spans: list[RedactionSpan]) -> str:
    redacted = text
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        redacted = redacted[: span.start] + span.placeholder + redacted[span.end :]
    return redacted


def _opf_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    if isinstance(result, dict):
        return result
    raise TypeError(f"unsupported OPF result type: {type(result)!r}")


def _to_predicted_spans(opf_payload: dict[str, Any]) -> list[RedactionSpan]:
    label_counts: Counter[str] = Counter()
    spans: list[RedactionSpan] = []
    for raw_span in opf_payload.get("detected_spans", []):
        start, end, raw_label, _ = _span_from_opf(raw_span)
        label = OPF_LABEL_MAP.get(raw_label, "CUSTOM")
        ordinal = label_counts[label]
        label_counts[label] += 1
        spans.append(
            RedactionSpan(
                start=start,
                end=end,
                label=label,
                placeholder=_placeholder(label, ordinal),
            )
        )
    return _non_overlapping_spans(spans)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: Path, payload: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
    else:
        line_count = 0
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return line_count + 1


def main() -> int:
    args = parse_args()

    try:
        from opf import OPF
    except ModuleNotFoundError:
        print(
            "Missing OPF package. Install with: git clone https://github.com/openai/privacy-filter && "
            "cd privacy-filter && python -m pip install -e .",
            file=sys.stderr,
        )
        return 2

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    if args.limit is not None:
        fixture = fixture[: args.limit]

    run_slug = args.run_label.strip().replace(" ", "-")
    artifact_dir = Path(args.output_root) / run_slug
    artifact_prefix = artifact_dir / f"opf_{run_slug}"
    raw_path = Path(f"{artifact_prefix}_raw_responses.jsonl")
    cases_path = Path(f"{artifact_prefix}_cases.jsonl")
    summary_path = Path(f"{artifact_prefix}_summary.json")
    metadata_path = Path(f"{artifact_prefix}_metadata.json")
    telemetry_path = Path(f"{artifact_prefix}_telemetry.json")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    for path in (raw_path, cases_path):
        path.write_text("")

    started_at = utc_now()
    redactor = OPF(
        model=args.checkpoint,
        context_window_length=args.context_window_length,
        device=args.device,
        output_mode="typed",
        decode_mode=args.decode_mode,
        discard_overlapping_predicted_spans=args.discard_overlapping_predicted_spans,
        output_text_only=False,
    )
    if args.viterbi_calibration_path:
        redactor.set_viterbi_decoder(calibration_path=args.viterbi_calibration_path)

    case_metrics = []
    telemetry_cases: list[dict[str, Any]] = []
    blocker: dict[str, Any] | None = None
    completed = 0

    for case in fixture:
        call_started = time.perf_counter()
        try:
            result = redactor.redact(case.text)
            latency_ms = round((time.perf_counter() - call_started) * 1000.0, 3)
            opf_payload = _opf_result_to_dict(result)
            predicted_spans = _to_predicted_spans(opf_payload)
            safety_matches = []
            if args.safety_pass:
                predicted_spans, safety_matches = add_safety_rule_spans(case.text, predicted_spans)
            actual_redacted_text = _render_redacted_text(opf_payload.get("text", case.text), predicted_spans)
            malformed_output = False
            error = None
        except Exception as exc:  # pragma: no cover
            latency_ms = round((time.perf_counter() - call_started) * 1000.0, 3)
            opf_payload = {}
            predicted_spans = []
            safety_matches = []
            actual_redacted_text = case.text
            malformed_output = True
            error = str(exc)
            blocker = {
                "case_id": case.case_id,
                "error": error,
                "latency_ms": latency_ms,
            }

        raw_line = append_jsonl(
            raw_path,
            {
                "case_id": case.case_id,
                "opf": opf_payload,
                "latency_ms": latency_ms,
                "error": error,
                "received_at_utc": utc_now(),
            },
        )

        metrics = compute_case_metrics(
            case_id=case.case_id,
            source_text=case.text,
            expected_spans=case.expected_spans,
            actual_redacted_text=actual_redacted_text,
        )
        case_metrics.append(metrics)
        completed += 1

        case_row = {
            "case_id": case.case_id,
            "source_type": case.source_type,
            "mode": "openai_privacy_filter",
            "pipeline_mode": "opf-plus-safety-pass" if args.safety_pass else "openai_privacy_filter",
            "actual_redacted_text": actual_redacted_text,
            "metrics": metrics.to_dict(),
            "predicted_spans": [asdict(span) for span in predicted_spans],
            "safety_rule_spans": [
                {
                    "rule": item.name,
                    "text": item.text,
                    "span": asdict(item.span),
                }
                for item in safety_matches
            ],
            "opf_detected_spans": opf_payload.get("detected_spans", []),
            "latency_ms": latency_ms,
            "parse_status": "ok" if not malformed_output else "blocked_or_error",
            "malformed_output": malformed_output,
            "timeout": False,
            "fail_open": malformed_output,
            "fail_closed": False,
            "raw_response_path": str(raw_path),
            "raw_response_line": raw_line,
        }
        append_jsonl(cases_path, case_row)
        telemetry_cases.append(
            {
                "case_id": case.case_id,
                "latency_ms": latency_ms,
                "parse_status": case_row["parse_status"],
                "malformed_output": malformed_output,
                "timeout": False,
                "fail_open_or_fail_closed": "fail_open" if malformed_output else None,
                "raw_response_artifact_path": str(raw_path),
                "raw_response_line": raw_line,
                "span_count": len(predicted_spans),
                "safety_rule_span_count": len(safety_matches),
                "decoded_mismatch": opf_payload.get("summary", {}).get("decoded_mismatch"),
            }
        )

        if blocker is not None:
            break

    completed_at = utc_now()
    summary = summarize_case_metrics(case_metrics).to_dict() if case_metrics else None
    summary_payload = {
        "status": "blocked" if blocker else "ok",
        "run_label": run_slug,
        "model": "openai/privacy-filter",
        "fixture_path": str(fixture_path),
        "cases_requested": len(fixture),
        "cases_completed": completed,
        "summary": summary,
        "blocker": blocker,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
    }
    write_json(summary_path, summary_payload)

    metadata_payload = {
        "status": summary_payload["status"],
        "run_label": run_slug,
        "model": "openai/privacy-filter",
        "checkpoint": args.checkpoint,
        "device": args.device,
        "decode_mode": args.decode_mode,
        "viterbi_calibration_path": args.viterbi_calibration_path,
        "context_window_length": args.context_window_length,
        "discard_overlapping_predicted_spans": args.discard_overlapping_predicted_spans,
        "safety_pass": bool(args.safety_pass),
        "safety_rule_version": SAFETY_RULE_VERSION if args.safety_pass else None,
        "fixture_path": str(fixture_path),
        "python": sys.version,
        "platform": platform.platform(),
        "artifacts": {
            "raw_responses_jsonl": str(raw_path),
            "cases_jsonl": str(cases_path),
            "summary_json": str(summary_path),
            "metadata_json": str(metadata_path),
            "telemetry_json": str(telemetry_path),
        },
        "label_map": OPF_LABEL_MAP,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "blocker": blocker,
    }
    write_json(metadata_path, metadata_payload)

    telemetry_payload = {
        "status": summary_payload["status"],
        "run_label": run_slug,
        "model": "openai/privacy-filter",
        "device": args.device,
        "decode_mode": args.decode_mode,
        "viterbi_calibration_path": args.viterbi_calibration_path,
        "safety_pass": bool(args.safety_pass),
        "safety_rule_version": SAFETY_RULE_VERSION if args.safety_pass else None,
        "cases_requested": len(fixture),
        "cases_completed": completed,
        "cases": telemetry_cases,
        "blocker": blocker,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
    }
    write_json(telemetry_path, telemetry_payload)

    print(json.dumps({"summary": summary_payload, "metadata": str(metadata_path)}, indent=2, sort_keys=True))
    return 0 if blocker is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
