"""Student TODO for manual autoregressive generation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .types import GenerationResult, SelectionResult


def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    selector: Callable[[torch.Tensor, torch.Generator], SelectionResult],
    max_new_tokens: int,
    eos_token_id: int,
    context_length: int,
    seed: int,
) -> GenerationResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")

