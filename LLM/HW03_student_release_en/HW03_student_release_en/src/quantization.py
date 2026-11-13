"""Educational symmetric per-tensor INT8 weight quantization."""

from __future__ import annotations

import torch

from .types import QuantizedTensor


def quantize_symmetric_int8(
    tensor: torch.Tensor,
) -> QuantizedTensor:
    """Quantize one floating tensor into signed symmetric INT8."""
    # TODO(HW03): compute a safe scale and quantize into [-127, 127].
    raise NotImplementedError("TODO(HW03): quantize_symmetric_int8")


def dequantize_symmetric_int8(
    quantized: QuantizedTensor,
) -> torch.Tensor:
    """Reconstruct an approximate floating tensor."""
    # TODO(HW03): restore shape, dtype, and scaled values.
    raise NotImplementedError("TODO(HW03): dequantize_symmetric_int8")


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Quantize floating state-dict tensors without mutating the input."""
    # TODO(HW03): quantize floating tensors and preserve other tensors.
    raise NotImplementedError("TODO(HW03): quantize_state_dict")


def dequantize_state_dict(
    quantized_state: dict[str, object],
) -> dict[str, torch.Tensor]:
    """Reconstruct a loadable state dictionary."""
    # TODO(HW03): reconstruct all tensors and metadata deterministically.
    raise NotImplementedError("TODO(HW03): dequantize_state_dict")
