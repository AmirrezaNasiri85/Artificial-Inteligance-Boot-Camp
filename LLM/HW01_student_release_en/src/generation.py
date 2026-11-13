"""Student TODO: manual sequential autoregressive decoding."""

from typing import Any

import torch


def greedy_generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    context_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    # TODO(HW01): Implement this function. Do not use model.generate().
    raise NotImplementedError("TODO(HW01)")
