"""Student TODOs for probability primitives."""

from __future__ import annotations

import torch


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Convert logits to probabilities with max-shift numerical stability."""

    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by a finite positive temperature without mutation."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    return logits / temperature


def distribution_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return Shannon entropy in natural-log units."""
    positive = probabilities > 0
    safe = torch.where(positive, probabilities, torch.ones_like(probabilities))
    terms = torch.where(positive, probabilities * torch.log(safe), torch.zeros_like(probabilities))
    return -terms.sum(dim=dim)


def support_size_for_mass(probabilities: torch.Tensor, mass: float) -> int:
    """Return minimal descending-probability support reaching mass."""
    if isinstance(mass, bool) or not isinstance(mass, (int, float)):
        raise TypeError("mass must be a number")
    sorted_p, _ = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=0)
    reached = (cumulative >= mass - 1e-6).nonzero()
    if reached.numel() == 0:
        return int(probabilities.numel())
    return int(reached[0].item()) + 1


def token_rank(probabilities: torch.Tensor, token_id: int) -> int:
    """Return one-based rank; lower token ID wins probability ties."""
    p = probabilities[token_id]
    higher = int((probabilities > p).sum().item())
    ties_before = int((probabilities[:token_id] == p).sum().item())
    return higher + ties_before + 1

