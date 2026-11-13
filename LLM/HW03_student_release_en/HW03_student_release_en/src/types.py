"""Shared immutable result structures."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectionResult:
    token_id: int
    model_probability: float
    sampling_probability: float
    rank: int
    entropy_nats: float
    candidate_count: int
    strategy: str


@dataclass
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    decoded_text: str
    trace: list[dict[str, object]]
    stop_reason: str
    forward_passes: int


@dataclass(frozen=True)
class LayerKVCache:
    """One decoder layer's key/value cache.

    Tensors use shape:
    [batch, heads, sequence_length, head_dimension].
    """

    key: torch.Tensor
    value: torch.Tensor

    @property
    def sequence_length(self) -> int:
        """Return the cached sequence length."""
        return int(self.key.shape[2])


@dataclass(frozen=True)
class CachedForwardOutput:
    """Logits and the complete per-layer KV cache."""

    logits: torch.Tensor
    past_key_values: tuple[LayerKVCache, ...]


@dataclass
class CachedGenerationResult:
    """Output and evidence from manual cached generation."""

    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    decoded_text: str
    trace: list[dict[str, object]]
    stop_reason: str
    prefill_calls: int
    decode_calls: int
    cache_rebuilds: int
    processed_token_positions: int


@dataclass(frozen=True)
class QuantizedTensor:
    """A symmetric per-tensor INT8 representation."""

    values: torch.Tensor
    scale: float
    original_dtype: str
    original_shape: tuple[int, ...]
