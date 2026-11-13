"""KV-cache tensor and mask helpers."""

from __future__ import annotations

import torch

from .types import LayerKVCache


def build_incremental_causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build a rectangular boolean causal mask for incremental attention."""
    # TODO(HW03): allow every past key and only nonfuture new keys.
    raise NotImplementedError("TODO(HW03): build_incremental_causal_mask")


def append_key_value(
    past_key_value: LayerKVCache | None,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> LayerKVCache:
    """Validate and append new key/value tensors on sequence dimension 2."""
    # TODO(HW03): validate shapes and append without mutating the past.
    raise NotImplementedError("TODO(HW03): append_key_value")
