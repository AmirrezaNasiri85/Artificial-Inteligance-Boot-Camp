from __future__ import annotations

import pytest
import torch

from src.cache import append_key_value, build_incremental_causal_mask
from src.types import LayerKVCache


def test_prefill_mask_matches_square_causal_mask() -> None:
    actual = build_incremental_causal_mask(
        3,
        3,
        0,
        device=torch.device("cpu"),
    )
    expected = torch.tensor(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)


def test_single_token_decode_can_see_all_keys() -> None:
    mask = build_incremental_causal_mask(
        1,
        5,
        4,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 5)
    assert not mask.any()


def test_multi_token_incremental_mask_is_rectangular() -> None:
    actual = build_incremental_causal_mask(
        3,
        7,
        4,
        device=torch.device("cpu"),
    )
    expected = torch.tensor(
        [
            [False, False, False, False, False, True, True],
            [False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False],
        ]
    )
    assert torch.equal(actual, expected)


def test_mask_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError):
        build_incremental_causal_mask(
            2,
            8,
            5,
            device=torch.device("cpu"),
        )


def test_append_without_past_returns_new_pair() -> None:
    key = torch.randn(2, 4, 3, 8)
    value = torch.randn(2, 4, 3, 8)
    result = append_key_value(None, key, value)
    assert result.key is key
    assert result.value is value


def test_append_grows_sequence_without_mutating_past() -> None:
    past_key = torch.randn(1, 4, 3, 8)
    past_value = torch.randn(1, 4, 3, 8)
    original_key = past_key.clone()
    original_value = past_value.clone()
    past = LayerKVCache(past_key, past_value)
    result = append_key_value(
        past,
        torch.randn(1, 4, 2, 8),
        torch.randn(1, 4, 2, 8),
    )
    assert result.key.shape == (1, 4, 5, 8)
    assert result.value.shape == (1, 4, 5, 8)
    assert torch.equal(past.key, original_key)
    assert torch.equal(past.value, original_value)


@pytest.mark.parametrize(
    "new_key,new_value",
    [
        (torch.randn(1, 4, 2), torch.randn(1, 4, 2)),
        (torch.randn(1, 4, 2, 8), torch.randn(1, 4, 3, 8)),
    ],
)
def test_append_rejects_invalid_new_pair(
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        append_key_value(None, new_key, new_value)


def test_append_rejects_batch_head_and_width_mismatch() -> None:
    past = LayerKVCache(
        torch.randn(1, 4, 2, 8),
        torch.randn(1, 4, 2, 8),
    )
    for shape in [(2, 4, 1, 8), (1, 3, 1, 8), (1, 4, 1, 7)]:
        with pytest.raises(ValueError):
            append_key_value(
                past,
                torch.randn(*shape),
                torch.randn(*shape),
            )
