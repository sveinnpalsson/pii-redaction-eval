"""CLI entry points for local redaction evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from redaction_research.benchmark import evaluate_fixture, write_run_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="redaction-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_fixture = subparsers.add_parser("eval-fixture", help="Run a minimal local fixture evaluation.")
    eval_fixture.add_argument("--fixture", required=True)
    eval_fixture.add_argument("--mode", choices=("regex", "expected", "passthrough"), default="regex")
    eval_fixture.add_argument("--output", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "eval-fixture":
        result = evaluate_fixture(Path(args.fixture), args.mode)
        write_run_result(result, Path(args.output))
        return 0

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
