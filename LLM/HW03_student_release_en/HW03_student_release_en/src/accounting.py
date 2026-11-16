"""Deterministic inference-work and memory accounting."""

from __future__ import annotations

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the exact storage bytes represented by one dense tensor."""
    return int(tensor.numel()) * int(tensor.element_size())


def estimate_kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    heads: int,
    sequence_length: int,
    head_dimension: int,
    bytes_per_element: int,
) -> int:
    """Estimate total key and value cache bytes."""
    for name, v in (("layers", layers), ("batch_size", batch_size),
                    ("heads", heads), ("sequence_length", sequence_length),
                    ("head_dimension", head_dimension),
                    ("bytes_per_element", bytes_per_element)):
        if v < 0:
            raise ValueError(f"{name} must be non-negative")

    return (2 * layers * batch_size * heads * sequence_length
            * head_dimension * bytes_per_element)


def uncached_processed_token_positions(
    prompt_length: int,
    generated_tokens: int,
    context_length: int,
) -> int:
    """Count positions processed by full-prefix autoregressive decoding."""
    if prompt_length < 0 or generated_tokens < 0 or context_length <= 0:
        raise ValueError("invalid lengths")
    total = 0
    for step in range(generated_tokens):

        total += min(prompt_length + step, context_length)
    return total


def cached_processed_token_positions(
    prompt_length: int,
    generated_tokens: int,
    context_length: int,
    rebuild_points: list[int] | None = None,
) -> int:
    """Count positions processed by prefill, decode, and cache rebuilds."""
    if prompt_length < 0 or generated_tokens < 0 or context_length <= 0:
        raise ValueError("invalid lengths")
    if generated_tokens == 0:
        return 0

    total = min(prompt_length, context_length)

    total += generated_tokens - 1

    for step in rebuild_points or []:
        if 0 < step < generated_tokens:
            total += min(prompt_length + step, context_length)
    return total


def max_concurrent_requests(
    *,
    memory_budget_bytes: int,
    reserved_bytes: int,
    model_bytes: int,
    cache_bytes_per_request: int,
) -> int:
    """Return the maximum whole requests that fit the memory budget."""
    if cache_bytes_per_request <= 0:
        raise ValueError("cache_bytes_per_request must be positive")
    available = memory_budget_bytes - reserved_bytes - model_bytes
    if available <= 0:
        return 0
    return available // cache_bytes_per_request
