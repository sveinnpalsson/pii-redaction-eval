"""Deterministic safety-pass recognizers for local redaction experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from redaction_research.metrics import RedactionSpan


SAFETY_RULE_VERSION = "2026-05-04-v1"


@dataclass(frozen=True, slots=True)
class SafetyRulePattern:
    name: str
    label: str
    pattern: re.Pattern[str]
    group: int = 0


@dataclass(frozen=True, slots=True)
class SafetyRuleMatch:
    name: str
    span: RedactionSpan
    text: str


_TOKEN_VALUE = r"[A-Za-z0-9][A-Za-z0-9._~:/+=@-]{5,}"

SAFETY_RULE_PATTERNS: tuple[SafetyRulePattern, ...] = (
    SafetyRulePattern(
        "private-key-header",
        "CUSTOM",
        re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----"),
    ),
    SafetyRulePattern(
        "jwt",
        "CUSTOM",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    SafetyRulePattern("openai-api-key", "CUSTOM", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    SafetyRulePattern("github-token", "CUSTOM", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    SafetyRulePattern("slack-token", "CUSTOM", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    SafetyRulePattern("aws-access-key", "CUSTOM", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    SafetyRulePattern("google-api-key", "CUSTOM", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    SafetyRulePattern(
        "labeled-secret",
        "CUSTOM",
        re.compile(
            rf"""(?ix)
            \b(?:api[_-]?key|access[_-]?token|client[_-]?secret|secret|password|token)
            \s*[:=#]\s*['"]?({_TOKEN_VALUE})
            """
        ),
        group=1,
    ),
    SafetyRulePattern(
        "home-path",
        "CUSTOM",
        re.compile(r"(?<!\w)/(?:home|Users)/[A-Za-z][A-Za-z0-9._-]{1,}(?:/[^\s\"'<>]*)?"),
    ),
    SafetyRulePattern(
        "windows-user-path",
        "CUSTOM",
        re.compile(r"\b[A-Za-z]:\\Users\\[A-Za-z][A-Za-z0-9._-]{1,}(?:\\[^\s\"'<>]*)?"),
    ),
    SafetyRulePattern("email", "EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    SafetyRulePattern(
        "url",
        "URL",
        re.compile(r"\bhttps?://[^\s<>\"')]+", re.I),
    ),
    SafetyRulePattern(
        "phone",
        "PHONE",
        re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
    ),
    SafetyRulePattern("ipv4-address", "CUSTOM", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    SafetyRulePattern("acct-token", "ACCOUNT", re.compile(r"\bACCT-\d{4,}-\d{4,}\b", re.I)),
    SafetyRulePattern(
        "labeled-account",
        "ACCOUNT",
        re.compile(
            rf"""(?ix)
            \b(?:acct|account|iban|route|card|ssn|passport|invoice|reference|ref|id)
            (?:\s+(?:number|no|num|id))?
            \s*[:=#-]?\s*({_TOKEN_VALUE})
            """
        ),
        group=1,
    ),
    SafetyRulePattern(
        "credit-card-like",
        "ACCOUNT",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    SafetyRulePattern(
        "handle",
        "CUSTOM",
        re.compile(r"(?<![\w/])@[A-Za-z][A-Za-z0-9._-]{2,}\b"),
    ),
    SafetyRulePattern(
        "cloud-resource-id",
        "CUSTOM",
        re.compile(r"\b(?:arn:aws:[A-Za-z0-9:/._+=,@-]+|projects/[A-Za-z0-9._-]+/[A-Za-z0-9/._-]+)\b"),
    ),
)


def spans_overlap(left: RedactionSpan, right: RedactionSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _placeholder(label: str, ordinal: int) -> str:
    return f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"


def _has_overlap(span: RedactionSpan, spans: list[RedactionSpan]) -> bool:
    return any(spans_overlap(span, existing) for existing in spans)


def _label_counts(spans: list[RedactionSpan]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for span in spans:
        counts[span.label] += 1
    return counts


def detect_safety_rule_matches(
    text: str,
    *,
    existing_spans: list[RedactionSpan] | None = None,
) -> list[SafetyRuleMatch]:
    accepted = sorted(existing_spans or [], key=lambda span: (span.start, span.end))
    label_counts = _label_counts(accepted)
    matches: list[SafetyRuleMatch] = []

    for rule in SAFETY_RULE_PATTERNS:
        for match in rule.pattern.finditer(text):
            try:
                start, end = match.span(rule.group)
            except IndexError:
                continue
            if start < 0 or end <= start:
                continue
            value = text[start:end].strip()
            if not value or value.startswith("<REDACTED_"):
                continue
            span = RedactionSpan(
                start=start,
                end=end,
                label=rule.label,
                placeholder=_placeholder(rule.label, label_counts[rule.label]),
            )
            if _has_overlap(span, accepted):
                continue
            accepted.append(span)
            accepted.sort(key=lambda item: (item.start, item.end))
            label_counts[rule.label] += 1
            matches.append(SafetyRuleMatch(name=rule.name, span=span, text=value))

    return matches


def add_safety_rule_spans(
    text: str,
    spans: list[RedactionSpan],
) -> tuple[list[RedactionSpan], list[SafetyRuleMatch]]:
    matches = detect_safety_rule_matches(text, existing_spans=spans)
    combined = sorted(
        [*spans, *(match.span for match in matches)],
        key=lambda span: (span.start, span.end, span.label),
    )
    return combined, matches
