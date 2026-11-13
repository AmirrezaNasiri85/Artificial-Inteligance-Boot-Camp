from __future__ import annotations

import pytest
import torch

from src.decoding import (
    sample_multinomial,
    sample_top_k,
    sample_top_p,
    select_greedy,
    select_second_best,
)


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def test_greedy_and_second_best_ranks() -> None:
    logits = torch.tensor([1.0, 4.0, 2.0])
    assert (select_greedy(logits).token_id, select_greedy(logits).rank) == (1, 1)
    assert (select_second_best(logits).token_id, select_second_best(logits).rank) == (2, 2)


def test_deterministic_tie_policy() -> None:
    logits = torch.tensor([2.0, 2.0, 2.0])
    assert select_greedy(logits).token_id == 0
    assert select_second_best(logits).token_id == 1


def test_multinomial_seed_reproducibility_and_variation() -> None:
    logits = torch.zeros(20)
    first = sample_multinomial(logits, generator(7)).token_id
    assert first == sample_multinomial(logits, generator(7)).token_id
    assert len({sample_multinomial(logits, generator(seed)).token_id for seed in range(12)}) > 1


def test_top_k_membership_count_and_normalization() -> None:
    logits = torch.arange(10, dtype=torch.float32)
    for seed in range(10):
        result = sample_top_k(logits, 3, generator(seed))
        assert result.token_id in {7, 8, 9}
        assert result.candidate_count == 3
        assert 0 < result.sampling_probability <= 1


def test_top_p_minimal_support_and_dominant_token() -> None:
    logits = torch.log(torch.tensor([0.60, 0.25, 0.10, 0.05]))
    assert sample_top_p(logits, 0.80, generator(1)).candidate_count == 2
    dominant = sample_top_p(torch.log(torch.tensor([0.97, 0.02, 0.01])), 0.90, generator(1))
    assert dominant.candidate_count == 1 and dominant.token_id == 0


def test_model_and_sampling_probabilities_are_distinct() -> None:
    result = sample_top_k(torch.tensor([0.0, 1.0, 2.0]), 1, generator(2))
    assert result.model_probability < 1
    assert result.sampling_probability == pytest.approx(1)


@pytest.mark.parametrize("k", [0, 4])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError):
        sample_top_k(torch.ones(3), k, generator(1))


@pytest.mark.parametrize("p", [0, -0.1, 1.1])
def test_invalid_p(p: float) -> None:
    with pytest.raises(ValueError):
        sample_top_p(torch.ones(3), p, generator(1))


def test_invalid_decoder_temperature() -> None:
    with pytest.raises(ValueError):
        select_greedy(torch.ones(3), 0)

