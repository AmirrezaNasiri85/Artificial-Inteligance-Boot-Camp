"""Transparent generation metrics and benchmark aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


def token_metrics(token_ids: list[int]) -> dict[str, float | int]:
    """Compute distinct-n and adjacent repetition directly on token IDs."""
    length = len(token_ids)
    pairs = list(zip(token_ids, token_ids[1:], strict=False))
    return {
        "token_distinct_1": len(set(token_ids)) / length if length else 0.0,
        "token_distinct_2": len(set(pairs)) / len(pairs) if pairs else 0.0,
        "adjacent_repeat_rate": sum(left == right for left, right in pairs) / len(pairs) if pairs else 0.0,
    }


def trace_metrics(trace: list[dict[str, object]]) -> dict[str, float | int | bool]:
    """Aggregate one generation trace; likelihood always uses model probability."""
    if not trace:
        return {
            "mean_selected_model_probability": 0.0,
            "mean_selected_sampling_probability": 0.0,
            "mean_selected_rank": 0.0,
            "mean_entropy_nats": 0.0,
            "mean_candidate_count": 0.0,
            "rank_1_selection_rate": 0.0,
            "sequence_log_probability": 0.0,
            "average_log_probability": 0.0,
            "generated_length": 0,
            "eos_reached": False,
        }
    model_probabilities = [float(row["model_probability"]) for row in trace]
    if any(not math.isfinite(value) or value <= 0 or value > 1 for value in model_probabilities):
        raise ValueError("trace model probabilities must satisfy 0 < p <= 1")
    count = len(trace)
    sequence_log_probability = sum(math.log(value) for value in model_probabilities)
    return {
        "mean_selected_model_probability": sum(model_probabilities) / count,
        "mean_selected_sampling_probability": sum(float(row["sampling_probability"]) for row in trace) / count,
        "mean_selected_rank": sum(int(row["rank"]) for row in trace) / count,
        "mean_entropy_nats": sum(float(row["entropy_nats"]) for row in trace) / count,
        "mean_candidate_count": sum(int(row["candidate_count"]) for row in trace) / count,
        "rank_1_selection_rate": sum(int(row["rank"]) == 1 for row in trace) / count,
        "sequence_log_probability": sequence_log_probability,
        "average_log_probability": sequence_log_probability / count,
        "generated_length": count,
        "eos_reached": bool(trace[-1]["stopped_by_eos"]),
    }


def benchmark_metric_row(run_id: str, prompt_id: str, strategy: str, seed: int, generated_ids: list[int], trace: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "strategy": strategy,
        "seed": seed,
        **trace_metrics(trace),
        **token_metrics(generated_ids),
    }


def summarize_by_strategy(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Average numeric run metrics by stable strategy identifier."""
    frame = pd.DataFrame(rows)
    numeric = [column for column in frame.select_dtypes(include="number").columns if column != "seed"]
    return frame.groupby("strategy", as_index=False)[numeric].mean()

