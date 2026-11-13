import math

import torch

from src.config import MiniGPTConfig, set_seed
from src.model import MiniGPT
from src.training import training_step


def test_training_step_changes_parameters_and_returns_plain_metrics():
    set_seed(3)
    config = MiniGPTConfig(context_length=8, n_layers=1, d_model=32, n_heads=4, d_ff=64)
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(4, 30, (2, 8))
    labels = torch.randint(4, 30, (2, 8))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = training_step(model, inputs, labels, optimizer)
    assert model.training
    assert all(isinstance(value, float) for value in metrics.values())
    assert math.isfinite(metrics["loss"])
    assert metrics["gradient_norm"] > 0
    assert metrics["supervised_positions"] == 16.0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_forward_shapes_and_loss():
    model = MiniGPT(MiniGPTConfig(context_length=8, n_layers=1, d_model=32, n_heads=4, d_ff=64))
    inputs = torch.randint(4, 30, (2, 8))
    output = model(inputs, targets=inputs)
    assert output.logits.shape == (2, 8, 260)
    assert output.loss is not None and output.loss.ndim == 0
