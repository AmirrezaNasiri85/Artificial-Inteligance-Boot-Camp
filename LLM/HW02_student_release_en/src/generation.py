"""Student TODO for manual autoregressive generation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .types import GenerationResult, SelectionResult


def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    selector: Callable[[torch.Tensor, torch.Generator], SelectionResult],
    max_new_tokens: int,
    eos_token_id: int,
    context_length: int,
    seed: int,
) -> GenerationResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    context = prompt_ids.detach().to("cpu").clone()
    prompt_token_ids: list[int] = context[0].tolist()
    generated_token_ids: list[int] = []
    trace: list[dict[str, object]] = []
    stop_reason = "max_new_tokens"
    forward_passes = 0

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            input_length = int(context.shape[1])
            window = context[:, -context_length:]

            out = model(window)
            all_logits = out.logits if hasattr(out, "logits") else out
            logits = all_logits[:, -1, :]
            forward_passes += 1

            selection = selector(logits[0], generator)
            next_id = int(selection.token_id)
            is_eos = next_id == eos_token_id

            generated_token_ids.append(next_id)
            context = torch.cat(
                [context, torch.tensor([[next_id]], dtype=context.dtype)], dim=1
            )

            trace.append(
                {
                    "step": step,
                    "input_length": input_length,
                    "effective_context_length": int(window.shape[1]),
                    "selected_token_id": next_id,
                    "selected_token_display": str(next_id),
                    "model_probability": float(selection.model_probability),
                    "sampling_probability": float(selection.sampling_probability),
                    "rank": int(selection.rank),
                    "entropy_nats": float(selection.entropy_nats),
                    "candidate_count": int(selection.candidate_count),
                    "strategy": str(selection.strategy),
                    "forward_pass_count": forward_passes,
                    "stopped_by_eos": is_eos,
                }
            )
            if is_eos:
                stop_reason = "eos"
                break

    return GenerationResult(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        decoded_text="",
        trace=trace,
        stop_reason=stop_reason,
        forward_passes=forward_passes,
    )