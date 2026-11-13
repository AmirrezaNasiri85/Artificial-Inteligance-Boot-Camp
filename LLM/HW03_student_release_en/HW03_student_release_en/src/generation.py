"""Manual autoregressive generation using an explicit KV cache."""

from __future__ import annotations

import torch
from torch import nn

from .types import CachedForwardOutput, CachedGenerationResult


def generate_with_kv_cache(
    model: nn.Module,
    prompt_ids: list[int] | torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    context_length: int,
    seed: int = 0,
) -> CachedGenerationResult:
    """Generate greedily with prefill, incremental decode, and trace data."""
    # TODO(HW03): implement sequential cached greedy generation.
    raise NotImplementedError("TODO(HW03): generate_with_kv_cache")


def rebuild_cache_from_window(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    context_length: int,
) -> tuple[torch.Tensor, CachedForwardOutput]:
    """Retain the newest context window and rebuild its cache by prefill."""
    # TODO(HW03): discard stale cache and prefill the retained window.
    raise NotImplementedError("TODO(HW03): rebuild_cache_from_window")
