"""Configuration and reproducibility helpers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MiniGPTConfig:
    vocab_size: int = 260
    context_length: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    use_causal_mask: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def set_seed(seed: int = 1403) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
