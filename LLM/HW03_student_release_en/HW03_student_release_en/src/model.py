"""Transparent decoder-only MiniGPT copied for standalone HW02 use."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch import nn as _hw03_nn
from torch.nn import functional as F

from .config import MiniGPTConfig
from .types import (
    CachedForwardOutput as _HW03CachedForwardOutput,
)
from .types import (
    LayerKVCache as _HW03LayerKVCache,
)


def build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


@dataclass
class MiniGPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.ln_attention(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        hidden = hidden + self.residual_dropout(attended)
        return hidden + self.mlp(self.ln_mlp(hidden))


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | bool | None = None,
        targets: torch.Tensor | None = None,
    ) -> MiniGPTOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if not 1 <= sequence_length <= self.config.context_length:
            raise ValueError("sequence length is outside the configured context window")
        if attention_mask is None and self.config.use_causal_mask:
            mask = build_causal_mask(sequence_length, input_ids.device)
        elif attention_mask is False:
            mask = None
        elif isinstance(attention_mask, torch.Tensor):
            if attention_mask.shape != (sequence_length, sequence_length):
                raise ValueError("attention_mask must have shape [sequence, sequence]")
            mask = attention_mask.to(input_ids.device)
        else:
            mask = None
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, mask)
        logits = self.output_projection(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return MiniGPTOutput(logits=logits, loss=loss)


def attention_forward_with_cache(
    attention: _hw03_nn.Module,
    hidden_states: torch.Tensor,
    *,
    past_key_value: _HW03LayerKVCache | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, _HW03LayerKVCache]:
    """Run one attention module for new positions and return its updated cache."""
    # TODO(HW03): compute new Q/K/V and attend over past plus new K/V.
    raise NotImplementedError("TODO(HW03): attention_forward_with_cache")


def model_forward_with_cache(
    model: _hw03_nn.Module,
    input_ids: torch.Tensor,
    *,
    past_key_values: tuple[_HW03LayerKVCache, ...] | None = None,
) -> _HW03CachedForwardOutput:
    """Run prompt prefill or incremental decode without changing model state."""
    # TODO(HW03): apply position offsets and return every layer's cache.
    raise NotImplementedError("TODO(HW03): model_forward_with_cache")
