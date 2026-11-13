"""Artifact contracts and validators for HW03."""

from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

REQUIRED_JSON = {
    "environment.json": {
        "python_version",
        "torch_version",
        "device",
        "cuda_available",
        "parameter_count",
        "checkpoint_sha256",
        "context_length",
        "vocabulary_size",
        "seed",
    },
    "cache_equivalence.json": {
        "prompt_length",
        "generated_tokens",
        "token_ids_equal",
        "max_logit_difference",
        "uncached_processed_token_positions",
        "cached_processed_token_positions",
        "processed_position_ratio",
    },
    "quantization_summary.json": {
        "original_serialized_bytes",
        "quantized_serialized_bytes",
        "compression_ratio",
        "quantized_tensor_count",
        "runtime_memory_claim",
    },
}
REQUIRED_CSV = {
    "uncached_trace.csv": {
        "step",
        "input_length",
        "selected_token_id",
        "selected_probability",
        "processed_token_positions",
    },
    "kv_cache_shapes.csv": {
        "layer",
        "tensor",
        "batch",
        "heads",
        "sequence_length",
        "head_dimension",
        "dtype",
        "bytes",
    },
    "prefill_decode_trace.csv": {
        "phase",
        "input_length",
        "past_length",
        "cache_length",
        "max_logit_difference",
    },
    "cached_generation_trace.csv": {
        "step",
        "phase",
        "input_length",
        "effective_context_length",
        "cache_length_before",
        "cache_length_after",
        "selected_token_id",
        "selected_probability",
        "prefill_call_count",
        "decode_call_count",
        "cache_rebuild_count",
        "processed_token_positions",
    },
    "inference_benchmark.csv": {
        "prompt_length",
        "generated_tokens",
        "uncached_seconds",
        "prefill_seconds",
        "mean_decode_seconds",
        "time_to_first_token_seconds",
        "cached_total_seconds",
        "uncached_processed_token_positions",
        "cached_processed_token_positions",
        "cache_bytes",
        "token_ids_equal",
    },
    "context_rebuild_trace.csv": {
        "step",
        "phase",
        "cache_rebuild_count",
        "cache_length_before",
        "cache_length_after",
        "processed_token_positions",
    },
    "quantization_tensor_metrics.csv": {
        "tensor_name",
        "numel",
        "scale",
        "mean_abs_error",
        "max_abs_error",
        "original_bytes",
        "quantized_value_bytes",
    },
    "quantization_quality.csv": {
        "prompt_id",
        "original_loss",
        "quantized_loss",
        "loss_delta",
        "top1_agreement",
        "total_variation_distance",
        "greedy_token_agreement",
    },
    "capacity_plan.csv": {
        "model_label",
        "parameter_count",
        "sequence_length",
        "weight_bytes_fp32",
        "kv_cache_bytes_per_request",
        "memory_budget_bytes",
        "reserved_bytes",
        "max_concurrent_requests_fp32",
    },
    "inference_strategy_summary.csv": {
        "strategy",
        "exact_or_approximate",
        "greedy_equivalent",
        "processed_token_work",
        "cache_memory",
        "checkpoint_storage",
        "main_advantage",
        "main_failure_mode",
    },
}
REQUIRED_FIGURES = {
    "kv_cache_growth.png",
    "processed_token_positions.png",
    "prefill_decode_latency.png",
    "quantization_error.png",
    "memory_capacity.png",
}
REQUIRED_RESULTS = tuple(list(REQUIRED_JSON) + list(REQUIRED_CSV))
REQUIRED_ARTIFACT_COUNT = len(REQUIRED_RESULTS) + len(REQUIRED_FIGURES)


def _finite(value):
    if isinstance(value, bool | str | int) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(x) for x in value)
    if isinstance(value, dict):
        return all(_finite(x) for x in value.values())
    return False


def _png(path: Path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
        raise ValueError(f"invalid PNG: {path.name}")
    return struct.unpack(">II", data[16:24])


def validate_artifacts(project_root: str | Path = "."):
    root = Path(project_root)
    results = root / "results"
    figures = root / "figures"
    for name, keys in REQUIRED_JSON.items():
        p = results / name
        if not p.is_file():
            raise ValueError(f"missing result: {name}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or not keys <= set(obj) or not _finite(obj):
            raise ValueError(f"invalid JSON: {name}")
    counts = {}
    for name, cols in REQUIRED_CSV.items():
        p = results / name
        if not p.is_file():
            raise ValueError(f"missing result: {name}")
        with p.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            fields = set(r.fieldnames or [])
            rows = list(r)
        if not cols <= fields or not rows:
            raise ValueError(f"invalid CSV: {name}")
        counts[name] = len(rows)
    for name in REQUIRED_FIGURES:
        p = figures / name
        if not p.is_file() or p.stat().st_size >= 500 * 1024:
            raise ValueError(f"missing/oversized figure: {name}")
        w, h = _png(p)
        if w < 300 or h < 200:
            raise ValueError(f"figure too small: {name}")
    env = json.loads((results / "environment.json").read_text())
    if env["parameter_count"] != 644736 or env["device"] != "cpu" or env["context_length"] != 128:
        raise ValueError("environment invariants failed")
    eq = json.loads((results / "cache_equivalence.json").read_text())
    if eq["token_ids_equal"] is not True or float(eq["max_logit_difference"]) > 1e-5:
        raise ValueError("cache equivalence failed")
    q = json.loads((results / "quantization_summary.json").read_text())
    if not 0 < float(q["compression_ratio"]) < 1:
        raise ValueError("quantized serialization not smaller")
    if q["runtime_memory_claim"] != "not_reduced_after_fp32_dequantization":
        raise ValueError("runtime-memory claim is inaccurate")
    return {
        "json_files": 3,
        "csv_files": 10,
        "figures": 5,
        "artifact_count": 18,
        "row_counts": counts,
    }
