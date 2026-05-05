"""Pure-Python metrics for local redaction evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


PLACEHOLDER_PATTERN = re.compile(r"<REDACTED_([^>]+)>")

# This mapping normalizes model-emitted placeholder labels into the benchmark
# taxonomy for scoring only. It is not a claim that all source labels are
# semantically identical in every context.
BENCHMARK_LABEL_NORMALIZATION = {
    "EMAIL": "EMAIL",
    "E_MAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "CONTACT_NUMBER": "PHONE",
    "URL": "URL",
    "WEBSITE": "URL",
    "LINK": "URL",
    "ADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "POSTCODE": "ADDRESS",
    "NAME": "PERSON",
    "PERSON": "PERSON",
    "FULL_NAME": "PERSON",
    "ACCOUNT": "ACCOUNT",
    "ACCOUNT_NUMBER": "ACCOUNT",
    "BANK_ACCOUNT": "ACCOUNT",
    "DATE": "DATE",
    "PRIVATE_DATE": "DATE",
    "BIRTHDATE": "DATE",
    "DOB": "DATE",
    "USERNAME": "CUSTOM",
    "USER": "CUSTOM",
    "PASSPORT": "CUSTOM",
    "SSN": "CUSTOM",
    "SOCIAL_NUMBER": "CUSTOM",
    "SOCIAL_SECURITY_NUMBER": "CUSTOM",
    "ID": "CUSTOM",
    "ID_CARD": "CUSTOM",
    "LABEL": "CUSTOM",
    "CUSTOM": "CUSTOM",
}


@dataclass(slots=True, frozen=True)
class RedactionSpan:
    start: int
    end: int
    label: str
    placeholder: str


@dataclass(slots=True)
class CaseMetrics:
    case_id: str
    total_expected: int
    correct_label_hidden: int
    hidden_any_label: int
    still_visible: int
    wrong_label_hidden: int
    over_redacted: int

    @property
    def false_positive(self) -> int:
        return self.wrong_label_hidden + self.over_redacted

    @property
    def false_negative(self) -> int:
        return self.still_visible + self.wrong_label_hidden

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SummaryMetrics:
    cases_total: int
    expected_total: int
    correct_label_hidden: int
    hidden_any_label: int
    still_visible: int
    wrong_label_hidden: int
    over_redacted: int
    precision: float
    recall: float
    f1: float
    f2: float
    binary_hide_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_placeholder_suffix(raw_label: str) -> str:
    parts = raw_label.upper().split("_")
    if len(parts) <= 1:
        return raw_label.upper()
    last = parts[-1]
    if last.isdigit() or (len(last) == 1 and last.isalpha()):
        return "_".join(parts[:-1])
    return raw_label.upper()


def normalize_benchmark_label(raw_label: str) -> str:
    normalized = _strip_placeholder_suffix(raw_label.strip().upper())
    if normalized.startswith("LABEL_") and normalized != "LABEL":
        return normalize_benchmark_label(normalized[len("LABEL_") :])
    return BENCHMARK_LABEL_NORMALIZATION.get(normalized, "CUSTOM")


def extract_normalized_placeholder_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in PLACEHOLDER_PATTERN.finditer(text):
        labels.append(normalize_benchmark_label(match.group(1)))
    return labels


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f_score(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def compute_case_metrics(
    case_id: str,
    source_text: str,
    expected_spans: list[RedactionSpan],
    actual_redacted_text: str,
) -> CaseMetrics:
    correct_label_hidden = 0
    hidden_any_label = 0
    still_visible = 0
    wrong_label_hidden = 0

    expected_placeholders = {span.placeholder for span in expected_spans}
    actual_placeholder_count = actual_redacted_text.count("<REDACTED_")
    actual_label_counts: dict[str, int] = {}
    for label in extract_normalized_placeholder_labels(actual_redacted_text):
        actual_label_counts[label] = actual_label_counts.get(label, 0) + 1
    matched_label_counts: dict[str, int] = {}

    for span in expected_spans:
        original_value = source_text[span.start : span.end]
        value_visible = original_value in actual_redacted_text
        expected_label = normalize_benchmark_label(span.label)
        matched_for_label = matched_label_counts.get(expected_label, 0)
        available_for_label = actual_label_counts.get(expected_label, 0)
        label_match_available = matched_for_label < available_for_label

        if value_visible:
            still_visible += 1
        elif label_match_available:
            correct_label_hidden += 1
            hidden_any_label += 1
            matched_label_counts[expected_label] = matched_for_label + 1
        else:
            wrong_label_hidden += 1
            hidden_any_label += 1

    over_redacted = max(actual_placeholder_count - len(expected_placeholders), 0)
    return CaseMetrics(
        case_id=case_id,
        total_expected=len(expected_spans),
        correct_label_hidden=correct_label_hidden,
        hidden_any_label=hidden_any_label,
        still_visible=still_visible,
        wrong_label_hidden=wrong_label_hidden,
        over_redacted=over_redacted,
    )


def summarize_case_metrics(case_metrics: list[CaseMetrics]) -> SummaryMetrics:
    cases_total = len(case_metrics)
    expected_total = sum(item.total_expected for item in case_metrics)
    correct_label_hidden = sum(item.correct_label_hidden for item in case_metrics)
    hidden_any_label = sum(item.hidden_any_label for item in case_metrics)
    still_visible = sum(item.still_visible for item in case_metrics)
    wrong_label_hidden = sum(item.wrong_label_hidden for item in case_metrics)
    over_redacted = sum(item.over_redacted for item in case_metrics)

    precision = safe_divide(correct_label_hidden, correct_label_hidden + wrong_label_hidden + over_redacted)
    recall = safe_divide(correct_label_hidden, expected_total)
    return SummaryMetrics(
        cases_total=cases_total,
        expected_total=expected_total,
        correct_label_hidden=correct_label_hidden,
        hidden_any_label=hidden_any_label,
        still_visible=still_visible,
        wrong_label_hidden=wrong_label_hidden,
        over_redacted=over_redacted,
        precision=precision,
        recall=recall,
        f1=f_score(precision, recall, beta=1.0),
        f2=f_score(precision, recall, beta=2.0),
        binary_hide_rate=safe_divide(hidden_any_label, expected_total),
    )
