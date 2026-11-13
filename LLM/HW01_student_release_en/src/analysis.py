"""Student TODO: aligned teacher-forcing and error propagation."""

from typing import Any

import torch


def total_variation(first: torch.Tensor, second: torch.Tensor) -> float:
    """TV(p, q) = 0.5 * sum |p - q| for two probability vectors."""
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("inputs must be torch tensors")
    if first.shape != second.shape:
        raise ValueError("distributions must have the same shape")
    if first.ndim != 1 or first.numel() < 1:
        raise ValueError("distributions must be non-empty 1-D tensors")
    p = first.detach().double()
    q = second.detach().double()
    if not (torch.isfinite(p).all() and torch.isfinite(q).all()):
        raise ValueError("distributions must be finite")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("probabilities must be non-negative")
    one = torch.tensor(1.0, dtype=torch.double)
    if not torch.isclose(p.sum(), one, atol=1e-4):
        raise ValueError("first distribution must sum to one")
    if not torch.isclose(q.sum(), one, atol=1e-4):
        raise ValueError("second distribution must sum to one")
    return float(0.5 * (p - q).abs().sum())


def _display(token_id: int) -> str:
    special = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
    if token_id in special:
        return special[token_id]
    if 4 <= token_id < 260:
        return repr(bytes([token_id - 4]).decode("latin-1"))
    return f"<{token_id}>"


@torch.inference_mode()
def measure_error_propagation(
    model: torch.nn.Module, reference_ids: torch.Tensor, injection_position: int
) -> list[dict[str, Any]]:
    """For each row t >= 1 compare P(x_t | prefix) on clean vs corrupted history."""

    length = reference_ids.numel()
    if not 0 <= injection_position < length:
        raise ValueError("injection_position out of range")

    model.eval()
    clean = reference_ids.long().clone()
    corrupted = clean.clone()
    original = int(clean[injection_position])
    # deterministic wrong byte token, always != original
    wrong = 4 + ((original - 4 + 1) % 256) if original >= 4 else 4
    corrupted[injection_position] = wrong

    rows: list[dict[str, Any]] = []
    for t in range(1, length):
        if t < injection_position:
            phase = "before"
        elif t == injection_position:
            phase = "injection"
        else:
            phase = "after"

        target = int(clean[t])
        free_prev = int(corrupted[t - 1])

        # predict x_t from the prefix x[:t]; use only the LAST position's logits
        p = torch.softmax(model(clean[:t].unsqueeze(0)).logits[0, -1], dim=-1)
        q = torch.softmax(model(corrupted[:t].unsqueeze(0)).logits[0, -1], dim=-1)

        rows.append(
            {
                "position": t,
                "phase": phase,
                "reference_token_id": target,
                "reference_token_display": _display(target),
                "free_prefix_token_id": free_prev,
                "free_prefix_token_display": _display(free_prev),
                "injected_token_id": wrong,
                "injected_token_display": _display(wrong),
                "teacher_reference_probability": float(p[target]),
                "free_reference_probability": float(q[target]),
                "total_variation": total_variation(p, q),
            }
        )
    return rows
