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
    """Generate tokens one at a time by taking the argmax of the last-position logits.

    Accepts a 1-D prompt of shape (P,) or a batch of one with shape (1, P).
    Returns the full sequence (prompt + generated tokens) with the same
    dimensionality as the input, and one metrics dict per generated token.
    """
    # validations
    if not isinstance(prompt_ids, torch.Tensor):
        raise TypeError("prompt_ids must be a torch.Tensor")
    if prompt_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("prompt_ids must have an integer dtype")
    if prompt_ids.ndim not in (1, 2):
        raise ValueError("prompt_ids must be 1-D or 2-D")
    if prompt_ids.ndim == 2 and prompt_ids.shape[0] != 1:
        raise ValueError("greedy_generate supports a batch of exactly one prompt")
    if prompt_ids.numel() < 1:
        raise ValueError("prompt_ids must contain at least one token")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("context_length must be an int")
    if context_length < 1:
        raise ValueError("context_length must be at least 1")
    if (prompt_ids < 0).any():
        raise ValueError("prompt_ids must contain non-negative token ids")

    input_was_1d = prompt_ids.ndim == 1
    sequence = prompt_ids.reshape(1, -1).clone().long()

    model.eval()
    metrics: list[dict[str, Any]] = []

    with torch.inference_mode():
        for step in range(max_new_tokens):
            # Keep only the last `context_length` tokens as the conditioning prefix.
            window = sequence[:, -context_length:]

            # One forward pass per generated token.
            logits = model(window).logits          # (1, window_len, vocab)

            # Only the final position predicts the next token.
            last_logits = logits[:, -1, :] 
            next_id = torch.argmax(last_logits, dim=-1, keepdim=True)  
            probability = torch.softmax(last_logits, dim=-1)[0, next_id.item()]

            sequence = torch.cat([sequence, next_id], dim=1)

            token_id = int(next_id.item())
            stopped = token_id == eos_token_id
            metrics.append(
                {
                    "step": step,
                    "selected_token_id": token_id,
                    "probability": float(probability),
                    "context_length_used": int(window.shape[1]),
                    "forward_pass_count": step + 1,
                    "sequence_length": int(sequence.shape[1]),
                    "stopped_by_eos": stopped,
                }
            )

            if stopped:
                break

    output = sequence[0] if input_was_1d else sequence
    return output, metrics
