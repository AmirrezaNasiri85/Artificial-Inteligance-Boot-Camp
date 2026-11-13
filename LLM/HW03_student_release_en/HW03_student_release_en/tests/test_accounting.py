from __future__ import annotations

import pytest
import torch

from src.accounting import (
    cached_processed_token_positions,
    estimate_kv_cache_bytes,
    max_concurrent_requests,
    tensor_nbytes,
    uncached_processed_token_positions,
)


def test_tensor_nbytes_exact_and_empty() -> None:
    assert tensor_nbytes(torch.zeros(3, 4, dtype=torch.float32)) == 48
    assert tensor_nbytes(torch.empty(0, dtype=torch.float64)) == 0


def test_tensor_nbytes_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        tensor_nbytes([1, 2, 3])  # type: ignore[arg-type]


def test_kv_cache_formula_and_zero_length() -> None:
    assert (
        estimate_kv_cache_bytes(
            layers=3,
            batch_size=2,
            heads=4,
            sequence_length=10,
            head_dimension=32,
            bytes_per_element=4,
        )
        == 61_440
    )
    assert (
        estimate_kv_cache_bytes(
            layers=3,
            batch_size=1,
            heads=4,
            sequence_length=0,
            head_dimension=32,
            bytes_per_element=4,
        )
        == 0
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("layers", 0),
        ("batch_size", 0),
        ("heads", 0),
        ("sequence_length", -1),
        ("head_dimension", 0),
        ("bytes_per_element", 0),
    ],
)
def test_kv_cache_formula_rejects_invalid_dimensions(
    field: str,
    value: int,
) -> None:
    arguments = {
        "layers": 3,
        "batch_size": 1,
        "heads": 4,
        "sequence_length": 8,
        "head_dimension": 32,
        "bytes_per_element": 4,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(**arguments)


def test_uncached_work_without_and_with_context_clipping() -> None:
    assert uncached_processed_token_positions(4, 3, 10) == 15
    assert uncached_processed_token_positions(4, 4, 5) == 19
    assert uncached_processed_token_positions(20, 2, 5) == 10
    assert uncached_processed_token_positions(4, 0, 5) == 0


def test_cached_work_without_rebuilds() -> None:
    assert cached_processed_token_positions(4, 3, 10) == 6
    assert cached_processed_token_positions(20, 2, 5) == 6
    assert cached_processed_token_positions(4, 0, 5) == 0


def test_cached_work_with_rebuilds() -> None:
    assert (
        cached_processed_token_positions(
            4,
            4,
            5,
            rebuild_points=[5],
        )
        == 11
    )
    assert (
        cached_processed_token_positions(
            5,
            5,
            5,
            rebuild_points=[5, 5, 5, 5],
        )
        == 25
    )


def test_cached_work_rejects_impossible_rebuilds() -> None:
    with pytest.raises(ValueError):
        cached_processed_token_positions(4, 0, 5, rebuild_points=[5])
    with pytest.raises(ValueError):
        cached_processed_token_positions(4, 2, 5, rebuild_points=[5, 5])
    with pytest.raises(ValueError):
        cached_processed_token_positions(4, 2, 5, rebuild_points=[6])


def test_capacity_floor_and_insufficient_memory() -> None:
    assert (
        max_concurrent_requests(
            memory_budget_bytes=1_000,
            reserved_bytes=100,
            model_bytes=300,
            cache_bytes_per_request=128,
        )
        == 4
    )
    assert (
        max_concurrent_requests(
            memory_budget_bytes=500,
            reserved_bytes=200,
            model_bytes=300,
            cache_bytes_per_request=1,
        )
        == 0
    )


def test_capacity_rejects_zero_cache_size() -> None:
    with pytest.raises(ValueError):
        max_concurrent_requests(
            memory_budget_bytes=500,
            reserved_bytes=0,
            model_bytes=0,
            cache_bytes_per_request=0,
        )
