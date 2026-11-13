from __future__ import annotations

import ast
import inspect
import math

import pytest
import torch

from src.probability import (
    apply_temperature,
    distribution_entropy,
    stable_softmax,
    support_size_for_mass,
    token_rank,
)


def test_stable_softmax_shape_sum_and_finite() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 1002.0], [-2.0, 0.0, 2.0]])
    probabilities = stable_softmax(logits, dim=1)
    assert probabilities.shape == logits.shape
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(2))


def test_softmax_constant_shift_invariance() -> None:
    logits = torch.tensor([-4.0, 1.0, 8.0])
    assert torch.allclose(stable_softmax(logits), stable_softmax(logits + 10000))


def test_softmax_rejects_empty_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        stable_softmax(torch.tensor([]))
    with pytest.raises(ValueError):
        stable_softmax(torch.tensor([0.0, math.inf]))


def test_no_prohibited_softmax_call() -> None:
    tree = ast.parse(inspect.getsource(stable_softmax))
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(name in {"torch.softmax", "F.softmax", "torch.nn.functional.softmax"} for name in calls)


def test_temperature_validation_and_nonmutation() -> None:
    logits = torch.tensor([1.0, 2.0])
    clone = logits.clone()
    assert torch.equal(apply_temperature(logits, 2), torch.tensor([0.5, 1.0]))
    assert torch.equal(logits, clone)
    for value in [0, -1, math.inf, math.nan]:
        with pytest.raises(ValueError):
            apply_temperature(logits, value)


def test_temperature_sharpens_and_flattens() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    low = stable_softmax(apply_temperature(logits, 0.5))
    high = stable_softmax(apply_temperature(logits, 2.0))
    assert low.max() > stable_softmax(logits).max() > high.max()


def test_entropy_deterministic_and_uniform() -> None:
    assert distribution_entropy(torch.tensor([1.0, 0.0])).item() == pytest.approx(0)
    assert distribution_entropy(torch.ones(4) / 4).item() == pytest.approx(math.log(4))


@pytest.mark.parametrize("values", [[0.7, -0.1, 0.4], [0.2, 0.2], [math.nan, 1.0]])
def test_entropy_rejects_invalid_probabilities(values: list[float]) -> None:
    with pytest.raises(ValueError):
        distribution_entropy(torch.tensor(values))


def test_support_size_known_distribution() -> None:
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert support_size_for_mass(probabilities, 0.49) == 1
    assert support_size_for_mass(probabilities, 0.8) == 2
    assert support_size_for_mass(probabilities, 0.81) == 3
    assert support_size_for_mass(probabilities, 1.0) == 4


def test_token_rank_tie_policy_and_validation() -> None:
    probabilities = torch.tensor([0.4, 0.4, 0.2])
    assert token_rank(probabilities, 0) == 1
    assert token_rank(probabilities, 1) == 2
    assert token_rank(probabilities, 2) == 3
    with pytest.raises(ValueError):
        token_rank(probabilities, 3)

