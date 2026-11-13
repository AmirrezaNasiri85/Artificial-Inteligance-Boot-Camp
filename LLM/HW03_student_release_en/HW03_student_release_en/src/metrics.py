"""Deterministic metrics used by the HW03 experiments."""

from __future__ import annotations

import torch


def maximum_absolute_difference(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Return the maximum absolute tensor difference."""
    if first.shape != second.shape:
        raise ValueError("tensor shapes must match")
    if first.numel() == 0:
        return 0.0
    return float((first - second).abs().max().item())
