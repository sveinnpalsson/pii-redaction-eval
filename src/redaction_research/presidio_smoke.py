"""Optional Presidio smoke evaluation for prepared fixtures."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any

from redaction_research.benchmark import load_fixture
from redaction_research.metrics import RedactionSpan, compute_case_metrics, summarize_case_metrics

ENTITY_LABEL_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "URL": "URL",
    "US_SSN": "ACCOUNT",
    "CREDIT_CARD": "ACCOUNT",
    "CRYPTO": "ACCOUNT",
    "IBAN_CODE": "ACCOUNT",
    "MEDICAL_LICENSE": "ACCOUNT",
    "US_BANK_NUMBER": "ACCOUNT",
    "US_DRIVER_LICENSE": "ACCOUNT",
    "US_ITIN": "ACCOUNT",
    "US_PASSPORT": "ACCOUNT",
    "US_VEHICLE_REGISTRATION": "ACCOUNT",
    "PERSON": "PERSON",
    "LOCATION": "ADDRESS",
    "STREET_ADDRESS": "ADDRESS",
    "DATE_TIME": "DATE",
    "IP_ADDRESS": "CUSTOM",
    "USER_HANDLE": "CUSTOM",
    "FILE_PATH": "CUSTOM",
    "ACCOUNT_ID": "ACCOUNT",
    "TICKET_ID": "CUSTOM",
    "SECRET_TOKEN": "CUSTOM",
    "JWT_TOKEN": "CUSTOM",
    "PRIVATE_KEY_HEADER": "CUSTOM",
    "CLOUD_RESOURCE_ID": "CUSTOM",
}

CUSTOM_RECOGNIZER_SPECS = [
    {
        "name": "local-vault-user-handle",
        "entity": "USER_HANDLE",
        "patterns": [
            {
                "name": "at_handle",
                "regex": r"(?<![\w./-])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,31}\b",
                "score": 0.72,
            }
        ],
    },
    {
        "name": "local-vault-file-path",
        "entity": "FILE_PATH",
        "patterns": [
            {
                "name": "posix_user_path",
                "regex": r"\b(?:/home|/Users)/[A-Za-z0-9._-]+(?:/[^\s,;:)\"']+)+",
                "score": 0.78,
            },
            {
                "name": "windows_user_path",
                "regex": r"\b[A-Za-z]:\\Users\\[A-Za-z0-9._-]+(?:\\[^\s,;:)\"']+)+",
                "score": 0.78,
            },
        ],
    },
    {
        "name": "local-vault-account-id",
        "entity": "ACCOUNT_ID",
        "patterns": [
            {
                "name": "account_prefixed_id",
                "regex": r"\b(?:ACCT|ACCOUNT|CUST|CUSTOMER|CLIENT|USER)[-_ :]?[A-Za-z0-9][A-Za-z0-9_-]{3,}\b",
                "score": 0.70,
            },
            {
                "name": "numeric_account_like",
                "regex": r"\b(?:acct|account|client|customer)[-_ ]?(?:id|number|no)?[-_ :=#]{0,3}\d{5,19}\b",
                "score": 0.72,
            },
        ],
    },
    {
        "name": "local-vault-ticket-id",
        "entity": "TICKET_ID",
        "patterns": [
            {
                "name": "jira_or_issue_key",
                "regex": r"\b[A-Z][A-Z0-9]{1,9}-\d{2,8}\b",
                "score": 0.68,
            },
            {
                "name": "ticket_prefixed_id",
                "regex": r"\b(?:ticket|case|issue|incident|ref)[-_ #:]*[A-Za-z0-9][A-Za-z0-9_-]{3,}\b",
                "score": 0.68,
            },
        ],
    },
    {
        "name": "local-vault-secret-token",
        "entity": "SECRET_TOKEN",
        "patterns": [
            {
                "name": "openai_style_key",
                "regex": r"\bsk-[A-Za-z0-9_-]{20,}\b",
                "score": 0.92,
            },
            {
                "name": "github_token",
                "regex": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
                "score": 0.92,
            },
            {
                "name": "slack_token",
                "regex": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
                "score": 0.90,
            },
            {
                "name": "aws_access_key",
                "regex": r"\bA(?:KI|SI)A[0-9A-Z]{16}\b",
                "score": 0.90,
            },
            {
                "name": "google_api_key",
                "regex": r"\bAIza[0-9A-Za-z_-]{20,}\b",
                "score": 0.88,
            },
            {
                "name": "labeled_secret_value",
                "regex": r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=:@-]{6,}",
                "score": 0.75,
            },
        ],
    },
    {
        "name": "local-vault-jwt-token",
        "entity": "JWT_TOKEN",
        "patterns": [
            {
                "name": "jwt_compact",
                "regex": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
                "score": 0.92,
            }
        ],
    },
    {
        "name": "local-vault-private-key-header",
        "entity": "PRIVATE_KEY_HEADER",
        "patterns": [
            {
                "name": "private_key_header",
                "regex": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                "score": 0.95,
            }
        ],
    },
    {
        "name": "local-vault-cloud-resource-id",
        "entity": "CLOUD_RESOURCE_ID",
        "patterns": [
            {
                "name": "aws_arn",
                "regex": r"\barn:aws:[^\s,;)\"']+",
                "score": 0.76,
            },
            {
                "name": "aws_resource_short_id",
                "regex": r"\b(?:i|sg|subnet|vol|ami)-[a-f0-9]{8,17}\b",
                "score": 0.72,
            },
            {
                "name": "gcp_resource_path",
                "regex": r"\bprojects/[A-Za-z0-9_-]+/[^\s,;)\"']+",
                "score": 0.72,
            },
        ],
    },
]


def _placeholder(label: str, ordinal: int) -> str:
    return f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"


def _non_overlapping_predictions(results: list[Any]) -> list[Any]:
    ordered = sorted(results, key=lambda item: (-float(item.score), item.start, -(item.end - item.start), item.entity_type))
    accepted: list[Any] = []
    for item in ordered:
        if any(not (item.end <= kept.start or item.start >= kept.end) for kept in accepted):
            continue
        accepted.append(item)
    return sorted(accepted, key=lambda item: item.start)


def _build_actual_redacted_text(text: str, spans: list[RedactionSpan]) -> str:
    redacted = text
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        redacted = redacted[: span.start] + span.placeholder + redacted[span.end :]
    return redacted


@dataclass(slots=True)
class PresidioSmokeResult:
    status: str
    fixture_path: str
    cases_total: int
    summary: dict[str, Any] | None
    cases: list[dict[str, Any]]
    recognizer_label_map: dict[str, str]
    blocker: str | None = None
    mode: str = "default"
    language: str = "en"
    custom_recognizers: list[str] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_custom_recognizers(pattern_recognizer_cls: Any, pattern_cls: Any) -> list[Any]:
    recognizers = []
    for spec in CUSTOM_RECOGNIZER_SPECS:
        recognizers.append(
            pattern_recognizer_cls(
                supported_entity=str(spec["entity"]),
                name=str(spec["name"]),
                patterns=[
                    pattern_cls(name=str(pattern["name"]), regex=str(pattern["regex"]), score=float(pattern["score"]))
                    for pattern in spec["patterns"]
                ],
            )
        )
    return recognizers


def run_presidio_smoke(
    fixture_path: Path,
    *,
    mode: str = "default",
    language: str = "en",
    score_threshold: float | None = None,
) -> PresidioSmokeResult:
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    except ModuleNotFoundError:
        return PresidioSmokeResult(
            status="blocked_missing_dependency",
            fixture_path=str(fixture_path),
            cases_total=0,
            summary=None,
            cases=[],
            recognizer_label_map=ENTITY_LABEL_MAP,
            blocker="presidio_analyzer is not installed in the current Python environment.",
            mode=mode,
            language=language,
        )
    if mode not in {"default", "custom"}:
        raise ValueError(f"unsupported Presidio mode: {mode}")

    analyzer = AnalyzerEngine()
    custom_recognizers: list[str] = []
    if mode == "custom":
        for recognizer in _build_custom_recognizers(PatternRecognizer, Pattern):
            analyzer.registry.add_recognizer(recognizer)
            custom_recognizers.append(str(recognizer.name))

    fixture = load_fixture(fixture_path)
    case_metrics = []
    case_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []

    for case in fixture:
        started = time.perf_counter()
        analyze_kwargs: dict[str, Any] = {"text": case.text, "language": language}
        if score_threshold is not None:
            analyze_kwargs["score_threshold"] = score_threshold
        raw_results = analyzer.analyze(**analyze_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_spans: list[RedactionSpan] = []
        label_counts: Counter[str] = Counter()

        for result in _non_overlapping_predictions(raw_results):
            label = ENTITY_LABEL_MAP.get(result.entity_type)
            if label is None:
                continue
            ordinal = label_counts[label]
            label_counts[label] += 1
            predicted_spans.append(
                RedactionSpan(
                    start=int(result.start),
                    end=int(result.end),
                    label=label,
                    placeholder=_placeholder(label, ordinal),
                )
            )

        actual_redacted_text = _build_actual_redacted_text(case.text, predicted_spans)
        metrics = compute_case_metrics(
            case_id=case.case_id,
            source_text=case.text,
            expected_spans=case.expected_spans,
            actual_redacted_text=actual_redacted_text,
        )
        case_metrics.append(metrics)
        case_rows.append(
            {
                "case_id": case.case_id,
                "source_type": case.source_type,
                "mode": f"presidio-{mode}",
                "actual_redacted_text": actual_redacted_text,
                "metrics": metrics.to_dict(),
                "predicted_spans": [asdict(span) for span in predicted_spans],
                "raw_entities": [
                    {
                        "entity_type": result.entity_type,
                        "start": int(result.start),
                        "end": int(result.end),
                        "score": float(result.score),
                    }
                    for result in raw_results
                ],
            }
        )
        telemetry_rows.append(
            {
                "case_id": case.case_id,
                "latency_ms": latency_ms,
                "raw_entity_count": len(raw_results),
                "accepted_span_count": len(predicted_spans),
                "timeout": False,
                "malformed_output": False,
                "fail_open_or_fail_closed": False,
                "parse_status": "ok",
            }
        )

    return PresidioSmokeResult(
        status="ok",
        fixture_path=str(fixture_path),
        cases_total=len(fixture),
        summary=summarize_case_metrics(case_metrics).to_dict(),
        cases=case_rows,
        recognizer_label_map=ENTITY_LABEL_MAP,
        mode=mode,
        language=language,
        custom_recognizers=custom_recognizers,
        telemetry=telemetry_rows,
    )
