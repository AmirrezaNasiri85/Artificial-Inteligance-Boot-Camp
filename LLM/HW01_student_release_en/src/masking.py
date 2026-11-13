"""Student TODO: attention-mask construction."""

import torch


def build_causal_mask(sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )