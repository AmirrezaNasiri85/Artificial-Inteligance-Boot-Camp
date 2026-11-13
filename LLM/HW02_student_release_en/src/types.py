"""Shared immutable result structures."""

from __future__ import annotations

from dataclasses import dataclass


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

