#!/usr/bin/env python3
"""Run a local GGUF decoder model on a benchmark fixture and score redaction output."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from redaction_research.benchmark import load_fixture
from redaction_research.metrics import compute_case_metrics, summarize_case_metrics


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "run"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percent / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def extract_assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    text = first.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def build_messages(system_prompt: str, source_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": source_text},
    ]


def load_runtime_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    required = ["condition_id", "gguf_repo_id", "gguf_filename", "prompt_path"]
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"missing required runtime config fields: {', '.join(missing)}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path, help="Fixture JSONL path.")
    parser.add_argument("--run-label", required=True, help="Run label used for artifact names.")
    parser.add_argument("--output-root", type=Path, default=Path("runs/gguf"), help="Output root.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen_gguf_runtime.json"), help="Runtime config JSON.")
    parser.add_argument("--prompt", type=Path, help="Prompt text file. Defaults to the config prompt_path.")
    parser.add_argument("--model-path", type=Path, help="Local GGUF file. If omitted, the Hugging Face repo and filename are used.")
    parser.add_argument("--model-sha256", help="Observed SHA256 for the local GGUF file, if already computed.")
    parser.add_argument("--repo-id", help="Hugging Face GGUF repository override.")
    parser.add_argument("--filename", help="GGUF filename override.")
    parser.add_argument("--n-ctx", type=int, help="Context length override.")
    parser.add_argument("--n-gpu-layers", type=int, help="GPU layer count override.")
    parser.add_argument("--main-gpu", type=int, help="Main GPU index.")
    parser.add_argument("--temperature", type=float, help="Temperature override.")
    parser.add_argument("--top-p", type=float, help="Top-p override.")
    parser.add_argument("--max-tokens", type=int, help="Maximum generated tokens override.")
    parser.add_argument("--chat-format", help="llama-cpp-python chat format override.")
    parser.add_argument("--limit", type=int, help="Optional first-N case limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config)

    try:
        from llama_cpp import Llama
    except ModuleNotFoundError:
        print(
            "Missing llama-cpp-python. Install the GGUF optional dependencies before running this condition.",
            file=sys.stderr,
        )
        return 2

    prompt_path = args.prompt or Path(config["prompt_path"])
    system_prompt = prompt_path.read_text(encoding="utf-8")
    fixture = load_fixture(args.fixture)
    if args.limit is not None:
        fixture = fixture[: args.limit]

    run_slug = slugify(args.run_label)
    output_dir = args.output_root / run_slug
    prefix = output_dir / f"gguf_{run_slug}"
    raw_path = prefix.with_name(prefix.name + "_raw_responses.jsonl")
    cases_path = prefix.with_name(prefix.name + "_cases.jsonl")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    telemetry_path = prefix.with_name(prefix.name + "_telemetry.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("", encoding="utf-8")
    cases_path.write_text("", encoding="utf-8")

    repo_id = args.repo_id or config["gguf_repo_id"]
    filename = args.filename or config["gguf_filename"]
    n_ctx = args.n_ctx if args.n_ctx is not None else int(config.get("context_length", 8192))
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else int(config.get("n_gpu_layers", -1))
    temperature = args.temperature if args.temperature is not None else float(config.get("temperature", 0))
    top_p = args.top_p if args.top_p is not None else float(config.get("top_p", 1))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(config.get("max_tokens", 512))
    chat_format = args.chat_format or config.get("chat_format")

    model_kwargs: dict[str, Any] = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
    }
    if args.main_gpu is not None:
        model_kwargs["main_gpu"] = args.main_gpu
    if chat_format:
        model_kwargs["chat_format"] = chat_format

    started_at = utc_now()
    load_started = time.perf_counter()
    if args.model_path:
        llm = Llama(model_path=str(args.model_path), **model_kwargs)
        model_source = {"model_path": str(args.model_path)}
    else:
        llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **model_kwargs)
        model_source = {"repo_id": repo_id, "filename": filename}
    load_ms = round((time.perf_counter() - load_started) * 1000.0, 3)

    metric_rows = []
    telemetry_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for case in fixture:
        messages = build_messages(system_prompt, case.text)
        call_started = time.perf_counter()
        raw_response: dict[str, Any]
        error: str | None = None
        try:
            raw_response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            actual_redacted_text = extract_assistant_text(raw_response)
            if not actual_redacted_text:
                error = "empty model output"
                actual_redacted_text = case.text
        except Exception as exc:  # pragma: no cover
            raw_response = {}
            error = str(exc)
            actual_redacted_text = case.text

        latency_ms = round((time.perf_counter() - call_started) * 1000.0, 3)
        metrics = compute_case_metrics(
            case_id=case.case_id,
            source_text=case.text,
            expected_spans=case.expected_spans,
            actual_redacted_text=actual_redacted_text,
        )
        metric_rows.append(metrics)

        raw_row = {
            "case_id": case.case_id,
            "source_type": case.source_type,
            "response": raw_response,
            "actual_redacted_text": actual_redacted_text,
            "latency_ms": latency_ms,
            "error": error,
        }
        case_row = {
            "case_id": case.case_id,
            "source_type": case.source_type,
            "actual_redacted_text": actual_redacted_text,
            "metrics": metrics.to_dict(),
            "latency_ms": latency_ms,
            "error": error,
        }
        telemetry_row = {
            "case_id": case.case_id,
            "latency_ms": latency_ms,
            "error": error,
        }
        if error is not None:
            errors.append({"case_id": case.case_id, "error": error})

        append_jsonl(raw_path, raw_row)
        append_jsonl(cases_path, case_row)
        telemetry_rows.append(telemetry_row)

    completed_at = utc_now()
    summary = summarize_case_metrics(metric_rows).to_dict()
    latencies = [row["latency_ms"] for row in telemetry_rows if row["latency_ms"] is not None]
    runtime_summary = {
        "cases_total": len(fixture),
        "error_count": len(errors),
        "load_ms": load_ms,
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "p50_ms": round(percentile(latencies, 50) or 0.0, 3) if latencies else None,
        "p95_ms": round(percentile(latencies, 95) or 0.0, 3) if latencies else None,
        "p99_ms": round(percentile(latencies, 99) or 0.0, 3) if latencies else None,
    }

    summary_payload = {
        "status": "ok" if not errors else "completed_with_errors",
        "run_label": run_slug,
        "condition_id": config.get("condition_id"),
        "fixture_path": str(args.fixture),
        "cases_requested": len(fixture),
        "cases_completed": len(fixture),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "summary": summary,
        "runtime_summary": runtime_summary,
    }
    metadata_payload = {
        "schema_version": config.get("schema_version"),
        "run_label": run_slug,
        "condition_id": config.get("condition_id"),
        "fixture_path": str(args.fixture),
        "model_source": model_source,
        "base_model_id": config.get("base_model_id"),
        "gguf_repo_id": repo_id,
        "gguf_filename": filename,
        "gguf_sha256": args.model_sha256 or config.get("gguf_sha256"),
        "gguf_source_url": config.get("gguf_source_url"),
        "quantization": config.get("quantization"),
        "runtime": config.get("runtime"),
        "runtime_commit_or_release": config.get("runtime_commit_or_release"),
        "prompt_version": config.get("prompt_version"),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_text(system_prompt),
        "settings": {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": args.main_gpu,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "chat_format": chat_format,
        },
        "packages": {
            "llama-cpp-python": package_version("llama-cpp-python"),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "artifacts": {
            "cases_jsonl": str(cases_path),
            "raw_responses_jsonl": str(raw_path),
            "summary_json": str(summary_path),
            "metadata_json": str(metadata_path),
            "telemetry_json": str(telemetry_path),
        },
    }
    telemetry_payload = {
        "run_label": run_slug,
        "condition_id": config.get("condition_id"),
        "runtime_summary": runtime_summary,
        "cases": telemetry_rows,
    }

    write_json(summary_path, summary_payload)
    write_json(metadata_path, metadata_payload)
    write_json(telemetry_path, telemetry_payload)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "metadata": str(metadata_path),
                "telemetry": str(telemetry_path),
                "status": summary_payload["status"],
                "errors": len(errors),
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
