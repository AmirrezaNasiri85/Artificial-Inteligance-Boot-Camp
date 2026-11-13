"""Student TODOs for token-selection policies."""

from __future__ import annotations

import torch

from .probability import apply_temperature, distribution_entropy, stable_softmax
from .types import SelectionResult


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty 1-D tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    return stable_softmax(apply_temperature(logits, temperature))

def _order(probs: torch.Tensor) -> torch.Tensor:
    # descending probability; stable sort => lower token id wins ties
    return torch.sort(probs, descending=True, stable=True).indices


def _rank(probs: torch.Tensor, token_id: int) -> int:
    p = probs[token_id]
    higher = int((probs > p).sum().item())
    ties_before = int((probs[:token_id] == p).sum().item())
    return higher + ties_before + 1


def _result(probs, token_id, kept_ids, strategy: str) -> SelectionResult:
    kept_mass = float(probs[kept_ids].sum())
    model_p = float(probs[token_id])
    return SelectionResult(
        token_id=int(token_id),
        model_probability=model_p,
        sampling_probability=model_p / kept_mass,  # renormalized over kept set
        rank=_rank(probs, int(token_id)),
        entropy_nats=float(distribution_entropy(probs)),
        candidate_count=int(kept_ids.numel()),
        strategy=strategy,
    )


def _sample(probs, kept_ids, generator, strategy: str) -> SelectionResult:
    weights = probs[kept_ids] / probs[kept_ids].sum()  # renormalize
    pick = torch.multinomial(weights, 1, generator=generator)
    return _result(probs, kept_ids[pick.item()], kept_ids, strategy)

def _top_p_ids(probs: torch.Tensor, p: float) -> torch.Tensor:
    order = _order(probs)
    cumulative = torch.cumsum(probs[order], dim=0)
    reached = (cumulative >= p - 1e-6).nonzero()
    count = int(reached[0].item()) + 1 if reached.numel() else probs.numel()
    return order[:count]

def select_greedy(logits: torch.Tensor, temperature: float = 1.0) -> SelectionResult:
    probs = _probs(logits, temperature)
    top = _order(probs)[:1]
    return _result(probs, top[0], top, "greedy")


def select_second_best(logits: torch.Tensor, temperature: float = 1.0) -> SelectionResult:
    probs = _probs(logits, temperature)
    if probs.numel() < 2:
        raise ValueError("need at least two tokens")
    second = _order(probs)[1]
    # no filtering: candidate set is the full vocabulary
    return _result(probs, second, torch.arange(probs.numel()), "second_best")


def sample_multinomial(logits: torch.Tensor, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    probs = _probs(logits, temperature)
    return _sample(probs, torch.arange(probs.numel()), generator, "multinomial")


def sample_top_k(logits: torch.Tensor, k: int, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")
    probs = _probs(logits, temperature)
    if not 1 <= k <= probs.numel():
        raise ValueError("k must be in [1, vocab size]")
    return _sample(probs, _order(probs)[:k], generator, "top_k")


def sample_top_p(logits: torch.Tensor, p: float, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    probs = _probs(logits, temperature)
    return _sample(probs, _top_p_ids(probs, p), generator, "top_p")


def select_entropy_adaptive(logits: torch.Tensor, entropy_threshold: float, top_p: float, generator: torch.Generator, temperature: float = 1.0) -> SelectionResult:
    if isinstance(entropy_threshold, bool) or not isinstance(entropy_threshold, (int, float)):
        raise TypeError("entropy_threshold must be a number")
    probs = _probs(logits, temperature)
    if float(distribution_entropy(probs)) > entropy_threshold:
        return _sample(probs, _top_p_ids(probs, top_p), generator, "entropy_adaptive_top_p")
    top = _order(probs)[:1]
    return _result(probs, top[0], top, "entropy_adaptive_greedy")
