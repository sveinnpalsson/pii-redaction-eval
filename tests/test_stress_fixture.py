from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from redaction_research.stress_fixture import (
    SOURCE_TYPE_COUNTS,
    StressFixtureValidationError,
    build_stress_fixture,
    validate_fixture_file,
    validate_fixture_rows,
    write_stress_fixture,
)


class StressFixtureTests(unittest.TestCase):
    def test_build_stress_fixture_is_deterministic_and_balanced(self) -> None:
        rows_a = build_stress_fixture(cases=200, seed=20260502)
        rows_b = build_stress_fixture(cases=200, seed=20260502)

        self.assertEqual(rows_a, rows_b)
        self.assertEqual(len(rows_a), 200)

        counts: dict[str, int] = {}
        for row in rows_a:
            counts[row["source_type"]] = counts.get(row["source_type"], 0) + 1
        self.assertEqual(counts, SOURCE_TYPE_COUNTS)

    def test_write_and_validate_fixture_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stress.jsonl"
            rows = write_stress_fixture(output_path, cases=200, seed=20260502)
            loaded_rows = validate_fixture_file(
                output_path,
                expected_case_count=200,
                expected_source_type_counts=SOURCE_TYPE_COUNTS,
            )

        self.assertEqual(rows, loaded_rows)
        self.assertTrue(any(span["label"] == "ADDRESS" for row in rows for span in row["expected_spans"]))
        self.assertTrue(any(span["label"] == "CUSTOM" for row in rows for span in row["expected_spans"]))

    def test_validator_rejects_duplicate_case_ids(self) -> None:
        rows = build_stress_fixture(cases=200, seed=20260502)
        rows[1]["case_id"] = rows[0]["case_id"]

        with self.assertRaises(StressFixtureValidationError):
            validate_fixture_rows(rows, expected_case_count=200, expected_source_type_counts=SOURCE_TYPE_COUNTS)

    def test_validator_rejects_expected_redacted_text_mismatch(self) -> None:
        rows = build_stress_fixture(cases=200, seed=20260502)
        rows[0]["expected_redacted_text"] = rows[0]["text"]

        with self.assertRaises(StressFixtureValidationError):
            validate_fixture_rows(rows, expected_case_count=200, expected_source_type_counts=SOURCE_TYPE_COUNTS)

    def test_validator_rejects_non_json_line(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "broken.jsonl"
            fixture_path.write_text(json.dumps({"case_id": "ok"}) + "\nnot-json\n", encoding="utf-8")

            with self.assertRaises(StressFixtureValidationError):
                validate_fixture_file(fixture_path)


if __name__ == "__main__":
    unittest.main()
