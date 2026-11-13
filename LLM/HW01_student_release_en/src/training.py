"""Explicit teacher-forced training utilities."""

from __future__ import annotations

import time
from typing import Any

import torch

from .data import sample_batch


def training_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one parallel teacher-forced optimization step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(input_ids, targets=labels)   # one forward pass
    loss = output.loss
    if loss is None:
        raise RuntimeError("model did not return a loss")

    loss.backward()

    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.detach().pow(2).sum())
    gradient_norm = grad_sq ** 0.5

    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "supervised_positions": float(labels.numel()),
    }


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    batches: int,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        inputs, labels = sample_batch(tokens, batch_size, sequence_length, generator)
        loss = model(inputs, targets=labels).loss
        if loss is None:
            raise RuntimeError("model did not return a loss")
        losses.append(float(loss))
    return sum(losses) / len(losses)


def train_fixed_steps(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    seed: int,
    validation_batches: int = 8,
) -> tuple[list[float], dict[str, Any]]:
    """Train for a fixed number of deterministic CPU steps and return measured metrics."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        inputs, labels = sample_batch(train_tokens, batch_size, sequence_length, generator)
        metrics = training_step(model, inputs, labels, optimizer)
        losses.append(metrics["loss"])
    wall_time = time.perf_counter() - started
    validation_loss = evaluate_loss(
        model,
        validation_tokens,
        batch_size,
        sequence_length,
        validation_batches,
        seed + 1,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, Any] = {
        "initial_loss": losses[0],
        "final_training_loss": losses[-1],
        "validation_loss": validation_loss,
        "training_steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "wall_clock_seconds": wall_time,
        "parameter_count": parameter_count,
        "fp32_parameter_memory_bytes": parameter_count * 4,
    }
    return losses, summary
