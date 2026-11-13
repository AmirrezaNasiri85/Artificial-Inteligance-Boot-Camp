"""Strict validation of required notebook evidence."""

from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

REQUIRED_JSON = {
    "environment.json": {"python_version", "torch_version", "device", "cuda_available", "parameter_count", "checkpoint_sha256", "context_length", "vocabulary_size", "seed"},
    "softmax_stability.json": {"input_logits", "naive_output_finite", "stable_probabilities", "stable_probability_sum"},
    "deterministic_choices.json": {"model", "distribution_A", "distribution_B"},
    "toy_sequence_search.json": {"greedy", "optimal"},
}
REQUIRED_CSV = {
    "top_tokens.csv": {"rank", "token_id", "token_display", "logit", "probability", "cumulative_probability"},
    "temperature_metrics.csv": {"temperature", "top_token_probability", "entropy_nats", "support_size_90", "rank_order_consistent", "probability_sum"},
    "sampling_candidate_sets.csv": {"strategy", "parameter", "candidate_count", "retained_probability_mass", "selected_token_id", "selected_token_display", "model_probability", "sampling_probability", "rank", "seed"},
    "generation_trace.csv": {"run_id", "prompt_id", "strategy", "seed", "step", "input_length", "effective_context_length", "selected_token_id", "selected_token_display", "model_probability", "sampling_probability", "rank", "entropy_nats", "candidate_count", "forward_pass_count", "stopped_by_eos"},
    "second_best_comparison.csv": {"strategy", "step", "rank", "model_probability", "entropy_nats", "cumulative_sequence_log_probability", "adjacent_repeat_rate"},
    "generations.csv": {"run_id", "prompt_id", "category", "prompt", "strategy", "seed", "generated_text", "generated_token_count", "stop_reason", "sequence_log_probability", "average_log_probability"},
    "decoding_metrics.csv": {"run_id", "prompt_id", "strategy", "seed", "mean_selected_model_probability", "mean_selected_sampling_probability", "mean_selected_rank", "mean_entropy_nats", "mean_candidate_count", "token_distinct_1", "token_distinct_2", "adjacent_repeat_rate", "rank_1_selection_rate", "generated_length", "eos_reached"},
    "adaptive_decoder_results.csv": {"prompt_id", "step", "entropy_nats", "entropy_threshold", "branch", "selected_token_id", "selected_token_display", "model_probability", "sampling_probability", "rank", "candidate_count"},
    "decoding_strategy_summary.csv": {"strategy", "deterministic", "candidate_set_rule", "uses_randomness", "seed_sensitive", "can_select_low_ranked_tokens", "candidate_count_behavior", "typical_diversity_behavior", "major_failure_mode"},
}
REQUIRED_FIGURES = {
    "next_token_distribution.png",
    "temperature_distributions.png",
    "second_best_probability_decay.png",
    "entropy_by_step.png",
    "diversity_vs_probability.png",
}


def _finite_values(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_values(item) for item in value)
    return True


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", payload[16:24])


def validate_artifacts(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    results = root / "results"
    figures = root / "figures"
    for name, keys in REQUIRED_JSON.items():
        path = results / name
        if not path.is_file():
            raise ValueError(f"missing result: {name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not keys <= set(value) or not _finite_values(value):
            raise ValueError(f"invalid JSON schema or value: {name}")
    row_counts: dict[str, int] = {}
    for name, columns in REQUIRED_CSV.items():
        path = results / name
        if not path.is_file():
            raise ValueError(f"missing result: {name}")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not columns <= set(reader.fieldnames or []):
                raise ValueError(f"invalid CSV schema: {name}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"CSV is empty: {name}")
        row_counts[name] = len(rows)
    for name in REQUIRED_FIGURES:
        path = figures / name
        if not path.is_file() or path.stat().st_size >= 500 * 1024:
            raise ValueError(f"missing or oversized figure: {name}")
        width, height = _png_dimensions(path)
        if width < 300 or height < 200:
            raise ValueError(f"figure dimensions are too small: {name}")
    environment = json.loads((results / "environment.json").read_text(encoding="utf-8"))
    if environment["parameter_count"] != 644736 or environment["device"] != "cpu":
        raise ValueError("environment model invariants failed")
    tree = json.loads((results / "toy_sequence_search.json").read_text(encoding="utf-8"))
    if tree["optimal"]["sequence_probability"] <= tree["greedy"]["sequence_probability"]:
        raise ValueError("toy tree does not demonstrate greedy suboptimality")
    return {"json_files": len(REQUIRED_JSON), "csv_files": len(REQUIRED_CSV), "figures": len(REQUIRED_FIGURES), "row_counts": row_counts}

