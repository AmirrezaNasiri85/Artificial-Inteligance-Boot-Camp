"""Student TODOs for probability primitives."""

from __future__ import annotations

import torch


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Convert logits to probabilities with max-shift numerical stability."""
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by a finite positive temperature without mutation."""
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def distribution_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return Shannon entropy in natural-log units."""
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def support_size_for_mass(probabilities: torch.Tensor, mass: float) -> int:
    """Return minimal descending-probability support reaching mass."""
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def token_rank(probabilities: torch.Tensor, token_id: int) -> int:
    """Return one-based rank; lower token ID wins probability ties."""
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")

