from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_LABELS = {"EMAIL", "PHONE", "URL", "ACCOUNT", "PERSON", "ADDRESS", "CUSTOM"}

SOURCE_TYPE_COUNTS = {
    "personal_note": 20,
    "chat_log": 20,
    "email_thread": 20,
    "meeting_notes": 15,
    "calendar_snippet": 10,
    "receipt_or_invoice": 15,
    "file_path": 15,
    "issue_or_commit": 15,
    "markdown_doc": 15,
    "json_or_yaml_config": 20,
    "url_identifier": 15,
    "ocr_noise": 20,
}

TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
PLACEHOLDER_PATTERN = re.compile(r"<REDACTED_[A-Z0-9_]+>")

FIRST_NAMES = [
    "Alex",
    "Avery",
    "Bailey",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Harper",
    "Jordan",
    "Kai",
    "Lane",
    "Logan",
    "Morgan",
    "Parker",
    "Quinn",
    "Reese",
    "Riley",
    "Rowan",
    "Sage",
    "Taylor",
]
LAST_NAMES = [
    "Briar",
    "Carver",
    "Dawson",
    "Ellis",
    "Frost",
    "Hale",
    "Iverson",
    "Keene",
    "Marlow",
    "North",
    "Pryor",
    "Quill",
    "Reeve",
    "Sawyer",
    "Tanner",
    "Vale",
    "Wilder",
    "Yarrow",
]
DOMAINS = ["amber", "cinder", "glider", "harbor", "juniper", "lattice", "signal", "tidemark"]
STREETS = ["Maple Loop", "Copper Way", "Signal Row", "Willow Trace", "Lantern Path", "Harbor Bend"]
CITIES = ["Fable Heights", "Sample Junction", "North Relay", "Quiet Harbor", "Cedar Hollow", "Mica Point"]
STATES = ["CA", "CO", "ME", "OR", "UT", "WA"]
PROJECTS = ["northstar", "local-vault", "rag-cache", "notes-sync", "mail-bridge", "issue-scrubber"]
AREAS = ["billing", "calendar", "vault", "triage", "receipts", "ops", "search", "docs"]
NOUNS = ["lantern", "cedar", "echo", "harbor", "signal", "thicket", "drift", "anchor"]


@dataclass(frozen=True, slots=True)
class TokenValue:
    value: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedCase:
    case_id: str
    source_type: str
    text: str
    expected_redacted_text: str
    expected_placeholders: list[str]
    expected_spans: list[dict[str, Any]]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_type": self.source_type,
            "text": self.text,
            "expected_redacted_text": self.expected_redacted_text,
            "expected_placeholders": list(self.expected_placeholders),
            "expected_spans": list(self.expected_spans),
            "notes": self.notes,
        }


class StressFixtureValidationError(ValueError):
    pass


def build_stress_fixture(cases: int = 200, seed: int = 20260502) -> list[dict[str, Any]]:
    expected_total = sum(SOURCE_TYPE_COUNTS.values())
    if cases != expected_total:
        raise ValueError(f"cases must equal {expected_total} to preserve documented source-type balance; got {cases}")

    rows: list[dict[str, Any]] = []
    for source_type, count in SOURCE_TYPE_COUNTS.items():
        for index in range(count):
            rendered = _build_case(source_type, index, seed)
            rows.append(rendered.to_dict())

    validate_fixture_rows(rows, expected_case_count=cases, expected_source_type_counts=SOURCE_TYPE_COUNTS)
    return rows


def write_stress_fixture(output_path: Path, cases: int = 200, seed: int = 20260502) -> list[dict[str, Any]]:
    rows = build_stress_fixture(cases=cases, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return rows


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StressFixtureValidationError(
                    f"{path}:{line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise StressFixtureValidationError(f"{path}:{line_number} must decode to an object")
            rows.append(row)
    return rows


def validate_fixture_file(
    path: Path,
    *,
    expected_case_count: int | None = None,
    expected_source_type_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(path)
    validate_fixture_rows(
        rows,
        expected_case_count=expected_case_count,
        expected_source_type_counts=expected_source_type_counts,
    )
    return rows


def validate_fixture_rows(
    rows: list[dict[str, Any]],
    *,
    expected_case_count: int | None = None,
    expected_source_type_counts: dict[str, int] | None = None,
) -> None:
    seen_case_ids: set[str] = set()
    source_type_counter: Counter[str] = Counter()

    if expected_case_count is not None and len(rows) != expected_case_count:
        raise StressFixtureValidationError(
            f"expected {expected_case_count} cases but found {len(rows)}"
        )

    for row_index, row in enumerate(rows, start=1):
        _validate_row(row, row_index=row_index, seen_case_ids=seen_case_ids)
        source_type_counter[str(row["source_type"])] += 1

    if expected_source_type_counts is not None and dict(source_type_counter) != dict(expected_source_type_counts):
        raise StressFixtureValidationError(
            f"source_type counts mismatch: expected {dict(expected_source_type_counts)}, got {dict(source_type_counter)}"
        )


def _validate_row(row: dict[str, Any], *, row_index: int, seen_case_ids: set[str]) -> None:
    required_fields = {
        "case_id",
        "source_type",
        "text",
        "expected_redacted_text",
        "expected_placeholders",
        "expected_spans",
    }
    missing = sorted(required_fields.difference(row))
    if missing:
        raise StressFixtureValidationError(f"row {row_index} is missing required fields: {missing}")

    case_id = str(row["case_id"])
    if case_id in seen_case_ids:
        raise StressFixtureValidationError(f"duplicate case_id: {case_id}")
    seen_case_ids.add(case_id)

    text = row["text"]
    expected_redacted_text = row["expected_redacted_text"]
    expected_placeholders = row["expected_placeholders"]
    expected_spans = row["expected_spans"]
    source_type = row["source_type"]

    if not isinstance(source_type, str) or not source_type:
        raise StressFixtureValidationError(f"{case_id} has invalid source_type")
    if not isinstance(text, str) or not isinstance(expected_redacted_text, str):
        raise StressFixtureValidationError(f"{case_id} text fields must be strings")
    if not isinstance(expected_placeholders, list) or not all(isinstance(item, str) for item in expected_placeholders):
        raise StressFixtureValidationError(f"{case_id} expected_placeholders must be a list of strings")
    if not isinstance(expected_spans, list):
        raise StressFixtureValidationError(f"{case_id} expected_spans must be a list")

    last_end = 0
    rebuilt_parts: list[str] = []
    placeholder_order: list[str] = []

    for span_index, span in enumerate(expected_spans):
        if not isinstance(span, dict):
            raise StressFixtureValidationError(f"{case_id} span {span_index} must be an object")
        for field in ("start", "end", "label", "placeholder"):
            if field not in span:
                raise StressFixtureValidationError(f"{case_id} span {span_index} is missing {field}")

        start = span["start"]
        end = span["end"]
        label = span["label"]
        placeholder = span["placeholder"]

        if not isinstance(start, int) or not isinstance(end, int):
            raise StressFixtureValidationError(f"{case_id} span {span_index} start/end must be ints")
        if not isinstance(label, str) or label not in VALID_LABELS:
            raise StressFixtureValidationError(f"{case_id} span {span_index} has invalid label {label!r}")
        if not isinstance(placeholder, str) or not PLACEHOLDER_PATTERN.fullmatch(placeholder):
            raise StressFixtureValidationError(f"{case_id} span {span_index} has invalid placeholder {placeholder!r}")
        if start < 0 or end <= start or end > len(text):
            raise StressFixtureValidationError(f"{case_id} span {span_index} has invalid bounds ({start}, {end})")
        if start < last_end:
            raise StressFixtureValidationError(f"{case_id} span {span_index} overlaps or is out of order")

        sensitive_value = text[start:end]
        if not sensitive_value:
            raise StressFixtureValidationError(f"{case_id} span {span_index} resolves to an empty substring")

        rebuilt_parts.append(text[last_end:start])
        rebuilt_parts.append(placeholder)
        placeholder_order.append(placeholder)
        last_end = end

    rebuilt_parts.append(text[last_end:])
    rebuilt_expected = "".join(rebuilt_parts)
    if rebuilt_expected != expected_redacted_text:
        raise StressFixtureValidationError(
            f"{case_id} expected_redacted_text mismatch: rebuilt value does not match stored value"
        )

    if placeholder_order != expected_placeholders:
        raise StressFixtureValidationError(
            f"{case_id} expected_placeholders mismatch: spans imply {placeholder_order}, stored {expected_placeholders}"
        )

    placeholder_mentions = PLACEHOLDER_PATTERN.findall(expected_redacted_text)
    if placeholder_mentions != expected_placeholders:
        raise StressFixtureValidationError(
            f"{case_id} placeholder appearance order mismatch: text has {placeholder_mentions}, stored {expected_placeholders}"
        )


def _build_case(source_type: str, index: int, seed: int) -> RenderedCase:
    profile = _build_profile(source_type, index, seed)
    template, notes = _select_template(source_type, index)
    case_id = f"local-vault-stress-v1-{source_type.replace('_', '-')}-{index + 1:03d}"
    return _render_case(case_id=case_id, source_type=source_type, template=template, replacements=profile, notes=notes)


def _build_profile(source_type: str, index: int, seed: int) -> dict[str, TokenValue]:
    rng = random.Random(f"{seed}:{source_type}:{index}")
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    second_first = rng.choice([name for name in FIRST_NAMES if name != first])
    second_last = rng.choice([name for name in LAST_NAMES if name != last])
    full_name = f"{first} {last}"
    backup_name = f"{second_first} {second_last}"
    domain = rng.choice(DOMAINS)
    area = rng.choice(AREAS)
    project = rng.choice(PROJECTS)
    noun = rng.choice(NOUNS)
    user_slug = f"{first.lower()}.{last.lower()}{index + 1:02d}"
    backup_slug = f"{second_first.lower()}.{second_last.lower()}{index + 7:02d}"
    email = f"{user_slug}@{domain}.test"
    backup_email = f"{backup_slug}@{domain}.test"
    phone = f"555-{200 + ((index * 13) % 700):03d}-{1000 + ((index * 37) % 9000):04d}"
    backup_phone = f"555-{200 + ((index * 19 + 11) % 700):03d}-{1000 + ((index * 53 + 17) % 9000):04d}"
    host = f"{area}.{project}.{domain}.test"
    path_slug = f"{project}-{noun}-{index + 1:02d}"
    url = f"https://{host}/{area}/{path_slug}?case={index + 1:03d}"
    backup_url = f"https://{host}/review/{backup_slug}?ticket={100 + index:03d}"
    account = f"ACCT-{1000 + index:04d}-{2000 + ((index * 7) % 8000):04d}"
    backup_account = f"TKN-{3000 + index:04d}-{4000 + ((index * 9) % 5000):04d}"
    street_number = 100 + ((index * 17) % 800)
    street = rng.choice(STREETS)
    city = rng.choice(CITIES)
    state = rng.choice(STATES)
    zip_code = f"000{10 + (index % 80):02d}"
    address = f"{street_number} {street}, {city}, {state} {zip_code}"
    backup_address = f"{street_number + 7} {rng.choice(STREETS)}, {rng.choice(CITIES)}, {rng.choice(STATES)} 000{20 + (index % 70):02d}"
    handle = f"@{first.lower()}_{noun}_{index + 1:02d}"
    channel = f"#{area}-{noun}-{(index % 9) + 1}"
    file_id = f"{project}-{index + 1:03d}-{noun}"
    commit_id = f"{(index + 1):04x}{(index * 73 + 19):04x}"
    yaml_key = f"{area}_{noun}_{index + 1:02d}"
    noisy_name = f"{first} {last}".replace("i", "1").replace("o", "0")
    noisy_email = email.replace("a", "@").replace(".", " . ")
    noisy_phone = phone.replace("-", " ")
    replacements = {
        "NAME_A": TokenValue(full_name, "PERSON"),
        "NAME_B": TokenValue(backup_name, "PERSON"),
        "EMAIL_A": TokenValue(email, "EMAIL"),
        "EMAIL_B": TokenValue(backup_email, "EMAIL"),
        "PHONE_A": TokenValue(phone, "PHONE"),
        "PHONE_B": TokenValue(backup_phone, "PHONE"),
        "URL_A": TokenValue(url, "URL"),
        "URL_B": TokenValue(backup_url, "URL"),
        "ACCOUNT_A": TokenValue(account, "ACCOUNT"),
        "ACCOUNT_B": TokenValue(backup_account, "ACCOUNT"),
        "ADDRESS_A": TokenValue(address, "ADDRESS"),
        "ADDRESS_B": TokenValue(backup_address, "ADDRESS"),
        "CUSTOM_A": TokenValue(handle, "CUSTOM"),
        "CUSTOM_B": TokenValue(file_id, "CUSTOM"),
        "PLAIN_HOST": TokenValue(host),
        "PLAIN_PROJECT": TokenValue(project),
        "PLAIN_AREA": TokenValue(area),
        "PLAIN_NOUN": TokenValue(noun),
        "PLAIN_CHANNEL": TokenValue(channel),
        "PLAIN_COMMIT": TokenValue(commit_id),
        "PLAIN_FILE": TokenValue(path_slug),
        "PLAIN_KEY": TokenValue(yaml_key),
        "PLAIN_NOISY_NAME": TokenValue(noisy_name, "PERSON"),
        "PLAIN_NOISY_EMAIL": TokenValue(noisy_email, "EMAIL"),
        "PLAIN_NOISY_PHONE": TokenValue(noisy_phone, "PHONE"),
    }
    return replacements


def _select_template(source_type: str, index: int) -> tuple[str, str]:
    templates = SOURCE_TYPE_TEMPLATES[source_type]
    selected = templates[index % len(templates)]
    return selected["template"], selected["notes"]


def _render_case(
    *,
    case_id: str,
    source_type: str,
    template: str,
    replacements: dict[str, TokenValue],
    notes: str,
) -> RenderedCase:
    text_parts: list[str] = []
    redacted_parts: list[str] = []
    expected_spans: list[dict[str, Any]] = []
    expected_placeholders: list[str] = []
    label_counts: Counter[str] = Counter()
    cursor = 0

    for match in TOKEN_PATTERN.finditer(template):
        text_parts.append(template[cursor:match.start()])
        redacted_parts.append(template[cursor:match.start()])
        token = match.group(1)
        replacement = replacements[token]
        start = sum(len(part) for part in text_parts)
        text_parts.append(replacement.value)
        if replacement.label is None:
            redacted_parts.append(replacement.value)
        else:
            ordinal = label_counts[replacement.label]
            label_counts[replacement.label] += 1
            placeholder = _placeholder_for(replacement.label, ordinal)
            redacted_parts.append(placeholder)
            expected_placeholders.append(placeholder)
            expected_spans.append(
                {
                    "start": start,
                    "end": start + len(replacement.value),
                    "label": replacement.label,
                    "placeholder": placeholder,
                }
            )
        cursor = match.end()

    text_parts.append(template[cursor:])
    redacted_parts.append(template[cursor:])
    text = "".join(text_parts)
    expected_redacted_text = "".join(redacted_parts)
    return RenderedCase(
        case_id=case_id,
        source_type=source_type,
        text=text,
        expected_redacted_text=expected_redacted_text,
        expected_placeholders=expected_placeholders,
        expected_spans=expected_spans,
        notes=notes,
    )


def _placeholder_for(label: str, ordinal: int) -> str:
    return f"<REDACTED_{label}_{chr(ord('A') + ordinal)}>"


SOURCE_TYPE_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "personal_note": [
        {
            "template": "Reminder for {{NAME_A}}: send the vault recap to {{EMAIL_A}} before 18:00 and keep {{ACCOUNT_A}} off the shared board.",
            "notes": "Personal note with person, email, and account token.",
        },
        {
            "template": "Sticky note: call {{PHONE_A}} about the spare badge for {{NAME_A}} and confirm the drop point at {{ADDRESS_A}}.",
            "notes": "Personal note with phone and address in a casual reminder.",
        },
        {
            "template": "Journal line: {{CUSTOM_A}} said the draft lives at {{URL_A}}, but only {{NAME_A}} should move {{ACCOUNT_A}}.",
            "notes": "Personal note blending handle, URL, person, and account.",
        },
        {
            "template": "Checklist: pack receipt, ping {{EMAIL_A}}, and update the contact card for {{NAME_A}} / {{PHONE_A}}.",
            "notes": "Compact note with slash-separated contact data.",
        },
    ],
    "chat_log": [
        {
            "template": "[09:14] {{CUSTOM_A}}: can you send {{NAME_A}} the invite at {{EMAIL_A}}?\n[09:15] ops-bot: use {{URL_A}} and mask {{ACCOUNT_A}} in screenshots.",
            "notes": "Two-line chat log with handle, person, email, URL, and account.",
        },
        {
            "template": "{{PLAIN_CHANNEL}}\n{{CUSTOM_A}} -> backup line is {{PHONE_A}}\n{{CUSTOM_B}} -> ship to {{ADDRESS_A}} after review",
            "notes": "Chat channel transcript with custom IDs plus phone and address.",
        },
        {
            "template": "chat export:\n{{NAME_A}}: I moved the file to {{URL_A}}\n{{NAME_B}}: thanks, I will text {{PHONE_A}} if the sync stalls",
            "notes": "Chat export with alternating speakers and URL/phone spans.",
        },
        {
            "template": "[bot] mention {{CUSTOM_A}} only in the private lane; their backup contact is {{EMAIL_A}} / {{PHONE_A}}.",
            "notes": "Bot-flavored chat line with compact contact tuple.",
        },
    ],
    "email_thread": [
        {
            "template": "From: {{NAME_A}} <{{EMAIL_A}}>\nTo: {{NAME_B}} <{{EMAIL_B}}>\nSubject: local vault handoff\n\nPlease archive {{ACCOUNT_A}} and keep the callback page at {{URL_A}}.",
            "notes": "Email headers plus body with account and URL.",
        },
        {
            "template": "Reply-To: {{EMAIL_A}}\nCC: {{EMAIL_B}}\n\n{{NAME_A}}, confirm the shipment address {{ADDRESS_A}} before the receipt closes.",
            "notes": "Email thread fragment with address confirmation.",
        },
        {
            "template": "Forwarded message\nFrom {{NAME_A}} on behalf of {{CUSTOM_A}}\nReach me at {{PHONE_A}} if {{ACCOUNT_A}} fails again.",
            "notes": "Forwarded thread with handle, phone, and account token.",
        },
        {
            "template": "Sender {{EMAIL_A}} wrote:\n> {{NAME_A}} updated {{URL_A}}\n> fallback number {{PHONE_A}}\nPlease do not paste {{ACCOUNT_A}} into tickets.",
            "notes": "Quoted email text with multiple sensitive spans.",
        },
    ],
    "meeting_notes": [
        {
            "template": "Meeting notes\n- owner: {{NAME_A}}\n- callback: {{PHONE_A}}\n- review page: {{URL_A}}\n- keep {{ACCOUNT_A}} out of slides",
            "notes": "Bullet-style notes with owner, phone, URL, and account.",
        },
        {
            "template": "Retro summary: {{NAME_A}} will verify the mailing point {{ADDRESS_A}} and send updates from {{EMAIL_A}}.",
            "notes": "Meeting note sentence with address and email.",
        },
        {
            "template": "Attendees: {{NAME_A}}, {{NAME_B}}\nAction: rotate {{ACCOUNT_A}} after {{CUSTOM_A}} finishes the vault sync.",
            "notes": "Attendance plus action item with account and handle.",
        },
    ],
    "calendar_snippet": [
        {
            "template": "Tue 08:30-09:00 | sync with {{NAME_A}} | dial {{PHONE_A}} | room note: verify {{ACCOUNT_A}}",
            "notes": "Calendar row with attendee, dial-in style phone, and account.",
        },
        {
            "template": "All-day: courier to {{ADDRESS_A}}\nGuest: {{EMAIL_A}}\nReference: {{CUSTOM_A}}",
            "notes": "Calendar snippet with address, guest email, and custom reference.",
        },
        {
            "template": "Calendar hold\nwho={{NAME_A}}\nwhere={{URL_A}}\nbackup={{PHONE_A}}",
            "notes": "Structured calendar block with URL and phone.",
        },
    ],
    "receipt_or_invoice": [
        {
            "template": "Receipt {{ACCOUNT_A}}\nBill to: {{NAME_A}}\nSend copy to: {{EMAIL_A}}\nShip-to: {{ADDRESS_A}}",
            "notes": "Receipt header with account-like invoice ID and billing contact.",
        },
        {
            "template": "Invoice memo: paid by {{NAME_A}} using ref {{ACCOUNT_A}}; service callback {{PHONE_A}}.",
            "notes": "Invoice sentence with account reference and callback number.",
        },
        {
            "template": "POS export | customer={{NAME_A}} | route={{URL_A}} | clerk note={{CUSTOM_A}}",
            "notes": "Receipt-like export row with URL and handle-shaped clerk note.",
        },
    ],
    "file_path": [
        {
            "template": "/vault/users/{{CUSTOM_A}}/notes/{{PLAIN_PROJECT}}/{{PLAIN_FILE}}.md",
            "notes": "File path with username-like path component.",
        },
        {
            "template": "/archive/{{PLAIN_AREA}}/{{NAME_A}}/{{ACCOUNT_A}}/handoff.txt",
            "notes": "File path with person directory and account-like filename component.",
        },
        {
            "template": "C:\\sync\\{{PLAIN_PROJECT}}\\{{CUSTOM_B}}\\contact-{{PHONE_A}}.txt",
            "notes": "Windows path with custom folder and phone in filename.",
        },
        {
            "template": "/tmp/mail/{{EMAIL_A}}/{{ADDRESS_A}}/proof.json",
            "notes": "Path-shaped string embedding email and address folders.",
        },
    ],
    "issue_or_commit": [
        {
            "template": "issue: redact callback tokens\nreporter={{NAME_A}}\ncontact={{EMAIL_A}}\nleaked_ref={{ACCOUNT_A}}",
            "notes": "Issue body with reporter identity and leaked account reference.",
        },
        {
            "template": "commit {{PLAIN_COMMIT}}\nAuthor: {{NAME_A}}\nMessage: remove {{URL_A}} and swap {{CUSTOM_A}} from docs",
            "notes": "Commit text with author, URL, and custom handle.",
        },
        {
            "template": "PR note: reproduce using {{PHONE_A}}, then rotate {{ACCOUNT_A}} before {{NAME_A}} reviews.",
            "notes": "PR discussion text with phone and account rotation.",
        },
    ],
    "markdown_doc": [
        {
            "template": "# Handoff\nOwner: {{NAME_A}}\nContact: {{EMAIL_A}}\nReference: {{ACCOUNT_A}}\n",
            "notes": "Markdown front matter style contact block.",
        },
        {
            "template": "## Checklist\n- profile: {{CUSTOM_A}}\n- docs: {{URL_A}}\n- ship label: {{ADDRESS_A}}\n",
            "notes": "Markdown list with handle, URL, and address.",
        },
        {
            "template": "Paragraph: Ping **{{NAME_A}}** at `{{PHONE_A}}` before merging the local vault note.",
            "notes": "Markdown inline formatting around person and phone.",
        },
    ],
    "json_or_yaml_config": [
        {
            "template": "{\"owner\":\"{{NAME_A}}\",\"email\":\"{{EMAIL_A}}\",\"account\":\"{{ACCOUNT_A}}\"}",
            "notes": "Compact JSON config with person, email, and account.",
        },
        {
            "template": "owner: {{NAME_A}}\ncallback: {{PHONE_A}}\nurl: {{URL_A}}\nprofile: {{CUSTOM_A}}\n",
            "notes": "YAML-like config with phone, URL, and custom profile.",
        },
        {
            "template": "{\"ship_to\":\"{{ADDRESS_A}}\",\"guest\":\"{{EMAIL_A}}\",\"ticket\":\"{{CUSTOM_B}}\"}",
            "notes": "JSON config with address, email, and custom ticket key.",
        },
        {
            "template": "vars:\n  approver: {{NAME_A}}\n  backup_phone: {{PHONE_A}}\n  token: {{ACCOUNT_A}}\n",
            "notes": "Indented YAML block with approver and token.",
        },
    ],
    "url_identifier": [
        {
            "template": "lookup={{URL_A}}#owner={{CUSTOM_A}}",
            "notes": "URL with handle-like identifier in the fragment.",
        },
        {
            "template": "redirect {{URL_A}}?contact={{EMAIL_A}}&ref={{ACCOUNT_A}}",
            "notes": "URL plus query parameters carrying email and account values.",
        },
        {
            "template": "path token: {{URL_A}}/{{NAME_A}}/{{CUSTOM_B}}",
            "notes": "URL identifier with trailing person and custom path segments.",
        },
    ],
    "ocr_noise": [
        {
            "template": "sc4n n0te -> n@me {{PLAIN_NOISY_NAME}} ; m@il {{PLAIN_NOISY_EMAIL}} ; ph0ne {{PLAIN_NOISY_PHONE}}",
            "notes": "OCR-like note with noisy name, email, and phone formatting.",
        },
        {
            "template": "0CR rec3ipt ref {{ACCOUNT_A}} addr {{ADDRESS_A}} c0ntact {{NAME_A}}",
            "notes": "OCR-ish receipt line with account, address, and person.",
        },
        {
            "template": "b1urry log url={{URL_A}} usr={{CUSTOM_A}} eml={{EMAIL_A}}",
            "notes": "Noisy log line with URL, handle, and email.",
        },
        {
            "template": "fax copy: {{NAME_A}} // {{PHONE_A}} // {{ADDRESS_A}} // {{ACCOUNT_A}}",
            "notes": "OCR-adjacent slash-delimited contact strip.",
        },
    ],
}
