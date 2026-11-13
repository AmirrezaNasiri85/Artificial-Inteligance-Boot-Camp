"""Transparent decoder-only MiniGPT implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import MiniGPTConfig
from .masking import build_causal_mask


@dataclass
class MiniGPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class DecoderBlock(nn.Module):
    """Pre-layer-normalized causal self-attention block."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.ln_attention(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden = hidden + self.residual_dropout(attended)
        return hidden + self.mlp(self.ln_mlp(hidden))


class MiniGPT(nn.Module):
    """A small decoder-only Transformer used unchanged in training and inference."""

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
        if sequence_length < 1 or sequence_length > self.config.context_length:
            raise ValueError("sequence length is outside the configured context window")
        if attention_mask is None and self.config.use_causal_mask:
            resolved_mask = build_causal_mask(sequence_length, input_ids.device)
        elif attention_mask is False:
            resolved_mask = None
        elif isinstance(attention_mask, torch.Tensor):
            if attention_mask.shape != (sequence_length, sequence_length):
                raise ValueError("attention_mask must have shape [sequence, sequence]")
            resolved_mask = attention_mask.to(input_ids.device)
        else:
            resolved_mask = None

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, resolved_mask)
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
