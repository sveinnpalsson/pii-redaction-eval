# PII Redaction Evaluation

This repository contains the experiment-reproduction code and instructions for
the paper: **Redaction Before Delegation: Benchmarking Local PII Filters on
Consumer Hardware**.

It contains:

- fixture construction code for the AI4Privacy-derived validation and train
  splits used in the paper
- the synthetic local-vault stress fixture and its deterministic builder
- scoring code for visible leakage, binary hide rate, wrong-label hides,
  over-redaction, precision, recall, F1, and F2
- local runners for the regex baseline, Presidio, OpenAI Privacy Filter, and
  Qwen GGUF direct redaction
- the Qwen direct-redaction prompt and GGUF runtime metadata
- aggregate result summaries in `results_summary/`
- model and runtime identifiers in `configs/models/experiment_models.json`
- a LaTeX table renderer for the aggregate result CSVs
- unit tests for the fixture builder, deterministic safety rules, metrics,
  GGUF runner helpers, and table renderer

## Data Inputs

The AI4Privacy-derived fixtures are not redistributed. To reproduce those
fixtures, obtain `ai4privacy/pii-masking-300k` separately and use these source
files from that dataset:

- validation source: `data/validation/1english_openpii_8k.jsonl`
- train source: `data/train/1english_openpii_30k.jsonl`

The paper uses:

- validation: 1,065 cases, 4,201 expected spans
- train/supporting: 3,817 cases, 16,148 expected spans
- synthetic stress: 200 cases, 656 expected spans

The expected fixture hashes are recorded in `results_summary/registry.json`.

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install optional runtime dependencies as needed:

```bash
python -m pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
python -m pip install llama-cpp-python huggingface-hub
```

OpenAI Privacy Filter must be installed according to its upstream instructions.
Model identifiers, quantization settings, prompt version, and GGUF artifact
metadata used for the paper are listed in
`configs/models/experiment_models.json` and
`configs/models/qwen_gguf_runtime.json`.

## Build Fixtures

Place the AI4Privacy source files in an untracked local directory, for example:

```text
local_data/ai4privacy/data/validation/1english_openpii_8k.jsonl
local_data/ai4privacy/data/train/1english_openpii_30k.jsonl
```

Then build the paper fixtures:

```bash
python scripts/build_ai4privacy_fixtures.py \
  --validation-source local_data/ai4privacy/data/validation/1english_openpii_8k.jsonl \
  --train-source local_data/ai4privacy/data/train/1english_openpii_30k.jsonl \
  --output-dir local_data/fixtures \
  --validation-cases 1065 \
  --train-cases 3817 \
  --source-revision <UPSTREAM_REVISION>
```

The script writes:

- `local_data/fixtures/ai4privacy-validation-full.jsonl`
- `local_data/fixtures/ai4privacy-train-full.jsonl`
- `local_data/fixtures/ai4privacy_fixture_manifest.json`

Rebuild the synthetic stress fixture:

```bash
PYTHONPATH=src python scripts/build_stress_fixture.py \
  --output data/fixtures/local-vault-stress-set-v1.jsonl \
  --cases 200 \
  --seed 20260502
```

Validate an existing stress fixture:

```bash
PYTHONPATH=src python scripts/build_stress_fixture.py \
  --validate-only data/fixtures/local-vault-stress-set-v1.jsonl \
  --cases 200
```

## Run Local Baselines

Regex baseline:

```bash
PYTHONPATH=src python -m redaction_research.cli eval-fixture \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --mode regex \
  --output runs/regex-validation.json
```

Presidio default:

```bash
PYTHONPATH=src python scripts/run_presidio_fixture.py \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --mode default \
  --run-label validation-presidio-default \
  --output-root runs/presidio
```

Presidio with the custom recognizers used in the paper:

```bash
PYTHONPATH=src python scripts/run_presidio_fixture.py \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --mode custom \
  --run-label validation-presidio-custom \
  --output-root runs/presidio
```

OpenAI Privacy Filter:

```bash
PYTHONPATH=src python scripts/run_opf_fixture.py \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --run-label validation-opf-default \
  --output-root runs/opf \
  --device cuda \
  --decode-mode viterbi
```

OpenAI Privacy Filter with the deterministic safety pass:

```bash
PYTHONPATH=src python scripts/run_opf_fixture.py \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --run-label validation-opf-rules \
  --output-root runs/opf \
  --device cuda \
  --decode-mode viterbi \
  --safety-pass
```

Qwen GGUF direct redaction:

```bash
PYTHONPATH=src python scripts/run_gguf_fixture.py \
  --fixture local_data/fixtures/ai4privacy-validation-full.jsonl \
  --run-label validation-qwen-gguf \
  --output-root runs/gguf \
  --config configs/models/qwen_gguf_runtime.json \
  --repo-id lmstudio-community/Qwen3.5-27B-GGUF \
  --filename Qwen3.5-27B-Q4_K_M.gguf
```

If the GGUF file is already present locally, replace the repository and filename
arguments with:

```bash
--model-path <LOCAL_GGUF_FILE>
```

Use the same commands with `ai4privacy-train-full.jsonl` and
`data/fixtures/local-vault-stress-set-v1.jsonl` for the supporting train split
and stress split.

## Render Table Inputs

Render the aggregate CSV summaries as LaTeX fragments:

```bash
PYTHONPATH=src python scripts/render_latex_tables.py \
  --input-dir results_summary \
  --output-dir generated_tables
```

## Check The Code

```bash
make check
```
