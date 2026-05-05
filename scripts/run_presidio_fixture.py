#!/usr/bin/env python3
"""Run Presidio default or custom recognizers on a benchmark fixture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from redaction_research.presidio_smoke import CUSTOM_RECOGNIZER_SPECS, run_presidio_smoke


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path, help="Fixture JSONL path.")
    parser.add_argument("--run-label", required=True, help="Run label used for the output directory and artifact prefix.")
    parser.add_argument("--mode", choices=("default", "custom"), default="default", help="Presidio recognizer mode.")
    parser.add_argument("--output-root", type=Path, default=Path("results/presidio"), help="Output root.")
    parser.add_argument("--language", default="en", help="Presidio language code.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional Presidio score threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_slug = args.run_label.strip().replace(" ", "-")
    output_dir = args.output_root / run_slug
    prefix = output_dir / f"presidio_{run_slug}"

    started = datetime.now(timezone.utc)
    result = run_presidio_smoke(
        args.fixture,
        mode=args.mode,
        language=args.language,
        score_threshold=args.score_threshold,
    )
    completed = datetime.now(timezone.utc)
    payload = result.to_dict()

    cases_path = prefix.with_name(prefix.name + "_cases.jsonl")
    raw_path = prefix.with_name(prefix.name + "_raw_responses.jsonl")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    telemetry_path = prefix.with_name(prefix.name + "_telemetry.json")

    cases = list(payload.get("cases") or [])
    telemetry = list(payload.get("telemetry") or [])
    raw_rows = [
        {
            "case_id": case.get("case_id"),
            "source_type": case.get("source_type"),
            "raw_entities": case.get("raw_entities", []),
        }
        for case in cases
    ]

    summary_payload = {
        "status": payload.get("status"),
        "blocker": payload.get("blocker"),
        "model": "Microsoft Presidio",
        "mode": args.mode,
        "run_label": run_slug,
        "fixture_path": str(args.fixture),
        "cases_requested": len(cases),
        "cases_completed": len(cases) if payload.get("status") == "ok" else 0,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "completed_at_utc": completed.isoformat(timespec="seconds"),
        "summary": payload.get("summary"),
    }
    metadata_payload = {
        "status": payload.get("status"),
        "blocker": payload.get("blocker"),
        "run_label": run_slug,
        "fixture_path": str(args.fixture),
        "mode": args.mode,
        "language": args.language,
        "score_threshold": args.score_threshold,
        "recognizer_label_map": payload.get("recognizer_label_map"),
        "custom_recognizers": payload.get("custom_recognizers", []),
        "custom_recognizer_specs": CUSTOM_RECOGNIZER_SPECS if args.mode == "custom" else [],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "presidio-analyzer": _package_version("presidio-analyzer"),
            "presidio-anonymizer": _package_version("presidio-anonymizer"),
            "spacy": _package_version("spacy"),
        },
        "started_at_utc": started.isoformat(timespec="seconds"),
        "completed_at_utc": completed.isoformat(timespec="seconds"),
        "artifacts": {
            "cases_jsonl": str(cases_path),
            "raw_responses_jsonl": str(raw_path),
            "summary_json": str(summary_path),
            "metadata_json": str(metadata_path),
            "telemetry_json": str(telemetry_path),
        },
    }
    telemetry_payload = {
        "run_label": run_slug,
        "mode": args.mode,
        "cases": telemetry,
    }

    _write_jsonl(cases_path, cases)
    _write_jsonl(raw_path, raw_rows)
    _write_json(summary_path, summary_payload)
    _write_json(metadata_path, metadata_payload)
    _write_json(telemetry_path, telemetry_payload)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "metadata": str(metadata_path),
                "telemetry": str(telemetry_path),
                "status": payload.get("status"),
                "blocker": payload.get("blocker"),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
