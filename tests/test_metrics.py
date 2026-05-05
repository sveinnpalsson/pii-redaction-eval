from __future__ import annotations

import unittest

from redaction_research.metrics import (
    RedactionSpan,
    compute_case_metrics,
    extract_normalized_placeholder_labels,
    f_score,
    normalize_benchmark_label,
    summarize_case_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_f_score_zero_safe(self) -> None:
        self.assertEqual(f_score(0.0, 0.0, beta=1.0), 0.0)

    def test_case_metrics_detects_correct_hide(self) -> None:
        spans = [RedactionSpan(start=6, end=22, label="EMAIL", placeholder="<REDACTED_EMAIL_A>")]
        metrics = compute_case_metrics(
            case_id="case-1",
            source_text="email alice@example.test",
            expected_spans=spans,
            actual_redacted_text="email <REDACTED_EMAIL_A>",
        )
        self.assertEqual(metrics.correct_label_hidden, 1)
        self.assertEqual(metrics.hidden_any_label, 1)
        self.assertEqual(metrics.still_visible, 0)
        self.assertEqual(metrics.wrong_label_hidden, 0)
        self.assertEqual(metrics.over_redacted, 0)

    def test_case_metrics_detects_visible_value(self) -> None:
        spans = [RedactionSpan(start=6, end=22, label="EMAIL", placeholder="<REDACTED_EMAIL_A>")]
        metrics = compute_case_metrics(
            case_id="case-2",
            source_text="email alice@example.test",
            expected_spans=spans,
            actual_redacted_text="email alice@example.test",
        )
        self.assertEqual(metrics.still_visible, 1)
        self.assertEqual(metrics.hidden_any_label, 0)

    def test_case_metrics_normalizes_model_placeholder_labels(self) -> None:
        spans = [
            RedactionSpan(start=0, end=17, label="EMAIL", placeholder="<REDACTED_EMAIL_A>"),
            RedactionSpan(start=18, end=30, label="PHONE", placeholder="<REDACTED_PHONE_A>"),
            RedactionSpan(start=31, end=39, label="PERSON", placeholder="<REDACTED_PERSON_A>"),
            RedactionSpan(start=40, end=49, label="ACCOUNT", placeholder="<REDACTED_ACCOUNT_A>"),
            RedactionSpan(start=50, end=58, label="ADDRESS", placeholder="<REDACTED_ADDRESS_A>"),
        ]
        metrics = compute_case_metrics(
            case_id="case-3",
            source_text="alice@example.test 555-123-4567 Jane Doe 123456789 Seaside",
            expected_spans=spans,
            actual_redacted_text=(
                "<REDACTED_EMAIL_1> <REDACTED_PHONE_NUMBER_1> <REDACTED_NAME_1> "
                "<REDACTED_ACCOUNT_NUMBER_1> <REDACTED_LABEL_CITY_1>"
            ),
        )
        self.assertEqual(metrics.correct_label_hidden, 5)
        self.assertEqual(metrics.hidden_any_label, 5)
        self.assertEqual(metrics.wrong_label_hidden, 0)
        self.assertEqual(metrics.still_visible, 0)

    def test_case_metrics_unknown_label_falls_back_to_custom_and_preserves_binary_hide(self) -> None:
        spans = [RedactionSpan(start=0, end=11, label="CUSTOM", placeholder="<REDACTED_CUSTOM_A>")]
        metrics = compute_case_metrics(
            case_id="case-4",
            source_text="user_handle",
            expected_spans=spans,
            actual_redacted_text="<REDACTED_LABEL_USERNAME_1>",
        )
        self.assertEqual(metrics.correct_label_hidden, 1)
        self.assertEqual(metrics.hidden_any_label, 1)
        self.assertEqual(metrics.wrong_label_hidden, 0)

    def test_case_metrics_wrong_label_still_counts_as_hidden_any_label(self) -> None:
        spans = [RedactionSpan(start=0, end=17, label="EMAIL", placeholder="<REDACTED_EMAIL_A>")]
        metrics = compute_case_metrics(
            case_id="case-5",
            source_text="alice@example.test",
            expected_spans=spans,
            actual_redacted_text="<REDACTED_USERNAME_1>",
        )
        self.assertEqual(metrics.correct_label_hidden, 0)
        self.assertEqual(metrics.hidden_any_label, 1)
        self.assertEqual(metrics.wrong_label_hidden, 1)
        self.assertEqual(metrics.still_visible, 0)

    def test_normalize_benchmark_label_observed_variants_and_unknown_custom(self) -> None:
        self.assertEqual(normalize_benchmark_label("EMAIL"), "EMAIL")
        self.assertEqual(normalize_benchmark_label("E_MAIL"), "EMAIL")
        self.assertEqual(normalize_benchmark_label("PHONE_NUMBER"), "PHONE")
        self.assertEqual(normalize_benchmark_label("WEBSITE"), "URL")
        self.assertEqual(normalize_benchmark_label("LABEL_ADDRESS"), "ADDRESS")
        self.assertEqual(normalize_benchmark_label("LABEL_CITY"), "ADDRESS")
        self.assertEqual(normalize_benchmark_label("NAME"), "PERSON")
        self.assertEqual(normalize_benchmark_label("LABEL_NAME"), "PERSON")
        self.assertEqual(normalize_benchmark_label("ACCOUNT_NUMBER"), "ACCOUNT")
        self.assertEqual(normalize_benchmark_label("LABEL_ACCOUNT_NUMBER"), "ACCOUNT")
        self.assertEqual(normalize_benchmark_label("PRIVATE_DATE"), "DATE")
        self.assertEqual(normalize_benchmark_label("LABEL_DOB"), "DATE")
        self.assertEqual(normalize_benchmark_label("USERNAME"), "CUSTOM")
        self.assertEqual(normalize_benchmark_label("LABEL_PASSPORT"), "CUSTOM")
        self.assertEqual(normalize_benchmark_label("SOMETHING_NEW"), "CUSTOM")

    def test_extract_normalized_placeholder_labels_parses_suffix_variants(self) -> None:
        labels = extract_normalized_placeholder_labels(
            "<REDACTED_EMAIL_A> <REDACTED_PHONE_NUMBER_1> <REDACTED_LABEL_ACCOUNT_NUMBER_9>"
        )
        self.assertEqual(labels, ["EMAIL", "PHONE", "ACCOUNT"])

    def test_summary_metrics_aggregate(self) -> None:
        spans = [RedactionSpan(start=6, end=22, label="EMAIL", placeholder="<REDACTED_EMAIL_A>")]
        rows = [
            compute_case_metrics(
                case_id="case-1",
                source_text="email alice@example.test",
                expected_spans=spans,
                actual_redacted_text="email <REDACTED_EMAIL_A>",
            ),
            compute_case_metrics(
                case_id="case-2",
                source_text="email alice@example.test",
                expected_spans=spans,
                actual_redacted_text="email alice@example.test",
            ),
        ]
        summary = summarize_case_metrics(rows)
        self.assertEqual(summary.cases_total, 2)
        self.assertEqual(summary.correct_label_hidden, 1)
        self.assertEqual(summary.still_visible, 1)
        self.assertAlmostEqual(summary.binary_hide_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
