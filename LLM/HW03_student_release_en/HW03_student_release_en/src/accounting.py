"""Deterministic inference-work and memory accounting."""

from __future__ import annotations

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the exact storage bytes represented by one dense tensor."""
    # TODO(HW03): implement exact dense-tensor byte accounting.
    raise NotImplementedError("TODO(HW03): tensor_nbytes")


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
    # TODO(HW03): implement the validated KV-cache memory formula.
    raise NotImplementedError("TODO(HW03): estimate_kv_cache_bytes")


def uncached_processed_token_positions(
    prompt_length: int,
    generated_tokens: int,
    context_length: int,
) -> int:
    """Count positions processed by full-prefix autoregressive decoding."""
    # TODO(HW03): account for repeated prefixes and context clipping.
    raise NotImplementedError("TODO(HW03): uncached_processed_token_positions")


def cached_processed_token_positions(
    prompt_length: int,
    generated_tokens: int,
    context_length: int,
    rebuild_points: list[int] | None = None,
) -> int:
    """Count positions processed by prefill, decode, and cache rebuilds."""
    # TODO(HW03): account for prefill, token decode, and rebuild work.
    raise NotImplementedError("TODO(HW03): cached_processed_token_positions")


def max_concurrent_requests(
    *,
    memory_budget_bytes: int,
    reserved_bytes: int,
    model_bytes: int,
    cache_bytes_per_request: int,
) -> int:
    """Return the maximum whole requests that fit the memory budget."""
    # TODO(HW03): implement safe capacity-floor accounting.
    raise NotImplementedError("TODO(HW03): max_concurrent_requests")
