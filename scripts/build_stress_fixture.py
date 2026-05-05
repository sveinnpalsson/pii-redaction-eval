#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from redaction_research.stress_fixture import (
    SOURCE_TYPE_COUNTS,
    validate_fixture_file,
    write_stress_fixture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate the synthetic local-vault stress fixture.")
    parser.add_argument(
        "--output",
        default="data/fixtures/local-vault-stress-set-v1.jsonl",
        help="Output JSONL path for build mode.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=sum(SOURCE_TYPE_COUNTS.values()),
        help="Total case count. Must match the documented balanced bucket total.",
    )
    parser.add_argument("--seed", type=int, default=20260502, help="Deterministic generation seed.")
    parser.add_argument(
        "--validate-only",
        help="Validate an existing JSONL fixture instead of building a new one.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.validate_only:
        rows = validate_fixture_file(
            Path(args.validate_only),
            expected_case_count=args.cases,
            expected_source_type_counts=SOURCE_TYPE_COUNTS,
        )
        print(f"validated {len(rows)} cases from {args.validate_only}")
        return 0

    rows = write_stress_fixture(Path(args.output), cases=args.cases, seed=args.seed)
    validate_fixture_file(
        Path(args.output),
        expected_case_count=args.cases,
        expected_source_type_counts=SOURCE_TYPE_COUNTS,
    )
    print(f"wrote {len(rows)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
