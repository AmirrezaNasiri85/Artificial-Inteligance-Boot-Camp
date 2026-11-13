from dataclasses import dataclass

import pytest
import torch
from torch import nn

from src.analysis import measure_error_propagation, total_variation


@dataclass
class Output:
    logits: torch.Tensor


class PrefixModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.grad_flags: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> Output:
        self.grad_flags.append(torch.is_grad_enabled())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -8.0)
        for row in range(batch):
            selected = (int(input_ids[row, -1]) + 1) % self.vocab_size
            logits[row, -1, selected] = 8.0
            logits[row, -1, (selected + 1) % self.vocab_size] = 4.0
        return Output(logits)


def test_total_variation_equal_and_known_distributions():
    first = torch.tensor([0.25, 0.75])
    assert total_variation(first, first) == pytest.approx(0.0)
    assert total_variation(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0])) == pytest.approx(0.5)


def test_total_variation_range_and_invalid_inputs():
    value = total_variation(torch.tensor([0.2, 0.8]), torch.tensor([0.7, 0.3]))
    assert 0.0 <= value <= 1.0
    with pytest.raises(ValueError, match="same shape"):
        total_variation(torch.tensor([1.0]), torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="sum to one"):
        total_variation(torch.tensor([0.2, 0.2]), torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="negative"):
        total_variation(torch.tensor([-0.1, 1.1]), torch.tensor([0.5, 0.5]))


def test_error_propagation_is_aligned_deterministic_and_inference_only():
    reference = torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.long)
    model = PrefixModel()
    first = measure_error_propagation(model, reference, injection_position=3)
    second = measure_error_propagation(model, reference, injection_position=3)
    assert first == second
    required = {
        "position",
        "phase",
        "reference_token_id",
        "reference_token_display",
        "free_prefix_token_id",
        "free_prefix_token_display",
        "injected_token_id",
        "injected_token_display",
        "teacher_reference_probability",
        "free_reference_probability",
        "total_variation",
    }
    assert all(required.issubset(row) for row in first)
    assert all(row["total_variation"] == pytest.approx(0.0) for row in first[:3])
    assert any(row["total_variation"] > 0.0 for row in first[3:])
    injection = next(row for row in first if row["phase"] == "injection")
    assert injection["injected_token_id"] != injection["reference_token_id"]
    assert injection["total_variation"] == pytest.approx(0.0)
    assert not any(model.grad_flags)
    assert model.training is False
