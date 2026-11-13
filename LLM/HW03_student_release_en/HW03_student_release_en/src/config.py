"""Model configuration and deterministic seeding."""

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
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count unique trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


HW03_CHECKPOINT_SHA256 = "7151108ca27ebfed3077aed89b4d093696bf09a715152fce99d3bf64167fea32"

HW03_REQUIRED_RESULTS = (
    "environment.json",
    "uncached_trace.csv",
    "kv_cache_shapes.csv",
    "prefill_decode_trace.csv",
    "cache_equivalence.json",
    "cached_generation_trace.csv",
    "inference_benchmark.csv",
    "context_rebuild_trace.csv",
    "quantization_summary.json",
    "quantization_tensor_metrics.csv",
    "quantization_quality.csv",
    "capacity_plan.csv",
    "inference_strategy_summary.csv",
)
HW03_REQUIRED_FIGURES = (
    "kv_cache_growth.png",
    "processed_token_positions.png",
    "prefill_decode_latency.png",
    "quantization_error.png",
    "memory_capacity.png",
)

HW03_CONTEXT_LENGTH = 128
HW03_PARAMETER_COUNT = 644_736
HW03_CACHE_SEQUENCE_DIMENSION = 2
