"""Student TODO: aligned teacher-forcing and error propagation."""

from typing import Any

import torch


def total_variation(first: torch.Tensor, second: torch.Tensor) -> float:
    # TODO(HW01): Validate and compare two probability distributions.
    raise NotImplementedError("TODO(HW01)")


@torch.inference_mode()
def measure_error_propagation(model: torch.nn.Module, reference_ids: torch.Tensor, injection_position: int) -> list[dict[str, Any]]:
    # TODO(HW01): Compare P(x_t | prefix before t) on aligned prefixes.
    raise NotImplementedError("TODO(HW01)")
