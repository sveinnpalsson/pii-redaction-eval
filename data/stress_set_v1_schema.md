# Synthetic Stress Set v1 Schema

This document defines the schema for the synthetic local-vault stress set used in the released aggregate results.

Target fixture:

- `data/fixtures/local-vault-stress-set-v1.jsonl`

Target size:

- 200 cases

## Goals

- resemble local-first private text more closely than benchmark-only fixture text
- cover source types such as notes, chat logs, file paths, issues, receipts, markdown, JSON/YAML, and OCR-like noise
- remain fully synthetic or hand-authored without real private data

## Required Fields

Each JSONL row should contain:

- `case_id`: stable unique identifier
- `source_type`: coarse source family such as `personal_note`, `chat_log`, `file_path`, `markdown`, `json_config`
- `text`: original source text
- `expected_redacted_text`: target redacted text, if available
- `expected_placeholders`: ordered list of placeholder tokens expected in the redacted output
- `expected_spans`: list of span objects with:
  - `start`
  - `end`
  - `label`
  - `placeholder`
- `notes`: optional annotation about why the case is included

## Label Set

The initial label inventory is:

- `EMAIL`
- `PHONE`
- `URL`
- `ACCOUNT`
- `PERSON`
- `ADDRESS`
- `CUSTOM`

## Source-Type Coverage Targets

The planned v1 release should cover these buckets at roughly these counts:

- `personal_note`: about 20
- `chat_log`: about 20
- `email_thread`: about 20
- `meeting_notes`: about 15
- `calendar_snippet`: about 10
- `receipt_or_invoice`: about 15
- `file_path`: about 15
- `issue_or_commit`: about 15
- `markdown_doc`: about 15
- `json_or_yaml_config`: about 20
- `url_identifier`: about 15
- `ocr_noise`: about 20

The repo-local builder for v1 uses these counts exactly and produces 200 total cases.

## Preparation Commands

Build the fixture from the repo root:

```bash
PYTHONPATH=src python3 scripts/build_stress_fixture.py \
  --output data/fixtures/local-vault-stress-set-v1.jsonl \
  --cases 200 \
  --seed 20260502
```

Validate an existing fixture file:

```bash
PYTHONPATH=src python3 scripts/build_stress_fixture.py \
  --validate-only data/fixtures/local-vault-stress-set-v1.jsonl \
  --cases 200
```

## Example Span Object

```json
{
  "start": 20,
  "end": 36,
  "label": "EMAIL",
  "placeholder": "<REDACTED_EMAIL_A>"
}
```

## Validation Notes

- spans should refer to offsets in `text`
- placeholder lists should align with `expected_redacted_text`
- rows should avoid duplicate `case_id` values
- no row should contain real private data
- each row should be hand-authored or produced from synthetic templates only

## Annotation Workflow

1. Draft the synthetic text and assign a stable `case_id`.
2. Identify each sensitive substring directly in `text`.
3. Compute exact character offsets for every sensitive span.
4. Assign a benchmark label from the label set above.
5. Assign a deterministic placeholder token such as `<REDACTED_EMAIL_A>`.
6. Build `expected_redacted_text` and `expected_placeholders`.
7. Add a short `notes` field when the case is included for a specific edge condition.

## Recommended Checks

- verify each line parses as JSON
- verify each `case_id` is unique
- verify each span substring matches the intended sensitive value
- verify placeholder order matches appearance order
- verify no real private data was used
- verify source-type counts match the documented 200-case bucket plan
