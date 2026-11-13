"""Student TODOs for token-selection policies."""

from __future__ import annotations

import torch

from .types import SelectionResult


def select_greedy(logits: torch.Tensor, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def select_second_best(logits: torch.Tensor, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def sample_multinomial(logits: torch.Tensor, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def sample_top_k(logits: torch.Tensor, k: int, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def sample_top_p(logits: torch.Tensor, p: float, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")


def select_entropy_adaptive(logits: torch.Tensor, entropy_threshold: float, top_p: float, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    # TODO(HW02): Implement this function.
    raise NotImplementedError("TODO(HW02)")

