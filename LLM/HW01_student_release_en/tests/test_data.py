import pytest
import torch

from src.data import create_lm_example, sample_batch


def test_shifted_example_has_expected_values_and_shapes():
    tokens = torch.tensor([1, 10, 11, 12, 2])
    inputs, labels = create_lm_example(tokens)
    assert torch.equal(inputs, torch.tensor([1, 10, 11, 12]))
    assert torch.equal(labels, torch.tensor([10, 11, 12, 2]))
    assert inputs.shape == labels.shape


def test_invalid_short_or_multidimensional_input():
    with pytest.raises(ValueError):
        create_lm_example(torch.tensor([1]))
    with pytest.raises(ValueError):
        create_lm_example(torch.ones((2, 2), dtype=torch.long))


def test_sample_batch_is_deterministic():
    tokens = torch.arange(100)
    first = sample_batch(tokens, 3, 8, torch.Generator().manual_seed(7))
    second = sample_batch(tokens, 3, 8, torch.Generator().manual_seed(7))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
