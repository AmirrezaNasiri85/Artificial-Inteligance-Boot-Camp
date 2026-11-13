import pytest
import torch

from src.masking import build_causal_mask


@pytest.mark.parametrize("length", [1, 2, 7, 64])
def test_causal_mask_shape_dtype_and_visibility(length):
    mask = build_causal_mask(length)
    assert mask.shape == (length, length)
    assert mask.dtype == torch.bool
    for query in range(length):
        for key in range(length):
            assert bool(mask[query, key]) is (key > query)


@pytest.mark.parametrize("length", [0, -1])
def test_invalid_length(length):
    with pytest.raises(ValueError):
        build_causal_mask(length)


def test_non_integer_length():
    with pytest.raises(TypeError):
        build_causal_mask(2.5)
