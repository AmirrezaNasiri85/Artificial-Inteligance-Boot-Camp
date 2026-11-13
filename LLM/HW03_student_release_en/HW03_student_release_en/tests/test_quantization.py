from __future__ import annotations

import io

import pytest
import torch

from src.config import MiniGPTConfig
from src.model import MiniGPT
from src.quantization import (
    dequantize_state_dict,
    dequantize_symmetric_int8,
    quantize_state_dict,
    quantize_symmetric_int8,
)


def _serialized_size(value: object) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.tell()


def test_known_tensor_quantization_and_range() -> None:
    tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    quantized = quantize_symmetric_int8(tensor)
    assert quantized.values.dtype == torch.int8
    assert int(quantized.values.min()) >= -127
    assert int(quantized.values.max()) <= 127
    assert quantized.values.tolist() == [-127, -64, 0, 64, 127]


def test_zero_and_empty_tensors_are_safe() -> None:
    zero = quantize_symmetric_int8(torch.zeros(4))
    empty = quantize_symmetric_int8(torch.empty(0))
    assert zero.scale == 1.0
    assert empty.scale == 1.0
    assert torch.equal(dequantize_symmetric_int8(zero), torch.zeros(4))
    assert dequantize_symmetric_int8(empty).numel() == 0


def test_round_trip_error_is_bounded() -> None:
    torch.manual_seed(5)
    tensor = torch.randn(100, dtype=torch.float32)
    quantized = quantize_symmetric_int8(tensor)
    restored = dequantize_symmetric_int8(quantized)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert float((restored - tensor).abs().max()) <= quantized.scale / 2 + 1e-6


def test_quantization_rejects_nonfloating_tensor() -> None:
    with pytest.raises(TypeError):
        quantize_symmetric_int8(torch.tensor([1, 2, 3]))


def test_state_dict_round_trip_and_nonfloating_preservation() -> None:
    state = {
        "weight": torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        "counter": torch.tensor([4], dtype=torch.int64),
    }
    original_weight = state["weight"].clone()
    original_counter = state["counter"].clone()
    packed = quantize_state_dict(state)
    restored = dequantize_state_dict(packed)
    assert set(restored) == set(state)
    assert restored["weight"].shape == state["weight"].shape
    assert restored["weight"].dtype == torch.float32
    assert torch.equal(restored["counter"], state["counter"])
    assert torch.equal(state["weight"], original_weight)
    assert torch.equal(state["counter"], original_counter)


def test_default_model_quantized_serialization_is_smaller() -> None:
    torch.manual_seed(3)
    state = MiniGPT(MiniGPTConfig()).state_dict()
    packed = quantize_state_dict(state)
    assert _serialized_size(packed) < _serialized_size(state)


def test_malformed_quantized_metadata_is_rejected() -> None:
    with pytest.raises(ValueError):
        dequantize_state_dict(
            {
                "format": "wrong",
                "tensors": {},
                "quantized_keys": (),
            }
        )
