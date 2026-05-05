#!/usr/bin/env python3
"""Build normalized benchmark fixtures from AI4Privacy PII-Masking-300k JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_MAP = {
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "URL": "URL",
    "WEBSITE": "URL",
    "LINK": "URL",
    "ACCOUNT": "ACCOUNT",
    "ACCOUNT_NUMBER": "ACCOUNT",
    "BANK_ACCOUNT": "ACCOUNT",
    "CREDIT_CARD": "ACCOUNT",
    "IBAN": "ACCOUNT",
    "PERSON": "PERSON",
    "NAME": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "ADDRESS": "ADDRESS",
    "STREET_ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "CITY": "ADDRESS",
    "DATE": "DATE",
    "BIRTHDATE": "DATE",
    "DOB": "DATE",
    "USERNAME": "CUSTOM",
    "USER_NAME": "CUSTOM",
    "USER": "CUSTOM",
    "ID": "CUSTOM",
    "SSN": "CUSTOM",
    "PASSPORT": "CUSTOM",
    "LICENSE": "CUSTOM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-cases", type=int, default=1065)
    parser.add_argument("--train-cases", type=int, default=3817)
    parser.add_argument("--source-revision", default="unrecorded")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(label: str) -> str | None:
    key = label.strip().upper().replace("-", "_")
    if key.startswith(("B_", "I_")):
        key = key[2:]
    if key.startswith("LABEL_"):
        key = key[6:]
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    for needle, mapped in (
        ("EMAIL", "EMAIL"),
        ("PHONE", "PHONE"),
        ("URL", "URL"),
        ("ACCOUNT", "ACCOUNT"),
        ("CARD", "ACCOUNT"),
        ("NAME", "PERSON"),
        ("ADDRESS", "ADDRESS"),
        ("DATE", "DATE"),
        ("USER", "CUSTOM"),
        ("PASSPORT", "CUSTOM"),
        ("SSN", "CUSTOM"),
        ("LICENSE", "CUSTOM"),
    ):
        if needle in key:
            return mapped
    return "CUSTOM"


def source_text(row: dict[str, Any]) -> str:
    for key in ("source_text", "text", "unmasked_text", "original_text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("row does not contain a source text field")


def source_id(row: dict[str, Any], line_number: int) -> str:
    for key in ("id", "case_id", "uid", "document_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(line_number)


def raw_span_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("privacy_mask", "spans", "span_labels", "labels"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, (dict, list, tuple))]
    return []


def span_from_item(item: Any, text: str) -> tuple[int, int, str] | None:
    if isinstance(item, dict):
        label = item.get("label") or item.get("entity") or item.get("type") or item.get("tag")
        start = item.get("start")
        end = item.get("end")
        value = item.get("value") or item.get("text")
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        start, end, label = item[0], item[1], item[2]
        value = None
    else:
        return None

    if label is None:
        return None
    mapped = normalize_label(str(label))
    if mapped is None:
        return None

    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        return start, end, mapped

    if isinstance(value, str) and value:
        start_index = text.find(value)
        if start_index >= 0:
            return start_index, start_index + len(value), mapped
    return None


def non_overlapping(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    ordered = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    accepted: list[tuple[int, int, str]] = []
    for span in ordered:
        start, end, _ = span
        if any(start < kept_end and kept_start < end for kept_start, kept_end, _ in accepted):
            continue
        accepted.append(span)
    return sorted(accepted, key=lambda item: item[0])


def render_case(row: dict[str, Any], line_number: int) -> dict[str, Any] | None:
    text = source_text(row)
    spans = non_overlapping(
        [
            parsed
            for item in raw_span_candidates(row)
            for parsed in [span_from_item(item, text)]
            if parsed is not None
        ]
    )
    if not spans:
        return None

    label_counts: Counter[str] = Counter()
    expected_spans = []
    placeholders = []
    redacted = text
    for start, end, label in reversed(spans):
        ordinal = label_counts[label]
        label_counts[label] += 1
        placeholder = f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"
        expected_spans.append({"start": start, "end": end, "label": label, "placeholder": placeholder})
        placeholders.append(label)
        redacted = redacted[:start] + placeholder + redacted[end:]

    expected_spans.reverse()
    placeholders.reverse()
    return {
        "case_id": f"ai4privacy-{source_id(row, line_number)}",
        "source_type": "docs",
        "text": text,
        "expected_redacted_text": redacted,
        "expected_placeholders": placeholders,
        "expected_spans": expected_spans,
    }


def build_fixture(source_path: Path, output_path: Path, max_cases: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = 0
    scanned = 0
    expected_spans = 0
    with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            scanned += 1
            row = render_case(json.loads(line), line_number)
            if row is None:
                continue
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            selected += 1
            expected_spans += len(row["expected_spans"])
            if selected >= max_cases:
                break
    return {
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_rows_scanned": scanned,
        "cases": selected,
        "expected_spans": expected_spans,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = build_fixture(
        args.validation_source,
        args.output_dir / "ai4privacy-validation-full.jsonl",
        args.validation_cases,
    )
    train = build_fixture(
        args.train_source,
        args.output_dir / "ai4privacy-train-full.jsonl",
        args.train_cases,
    )
    manifest = {
        "dataset": "ai4privacy/pii-masking-300k",
        "source_revision": args.source_revision,
        "source_files": {
            "validation": "data/validation/1english_openpii_8k.jsonl",
            "train": "data/train/1english_openpii_30k.jsonl",
        },
        "label_policy": "benchmark placeholder taxonomy",
        "validation": validation,
        "train": train,
    }
    manifest_path = args.output_dir / "ai4privacy_fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
