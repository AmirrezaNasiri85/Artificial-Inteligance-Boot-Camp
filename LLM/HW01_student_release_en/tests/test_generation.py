from dataclasses import dataclass

import torch
from torch import nn

from src.generation import greedy_generate


@dataclass
class Output:
    logits: torch.Tensor


class RecordingModel(nn.Module):
    def __init__(self, selections, vocab_size=12):
        super().__init__()
        self.selections = selections
        self.calls = []
        self.grad_flags = []

    def forward(self, input_ids):
        self.calls.append(input_ids.detach().clone())
        self.grad_flags.append(torch.is_grad_enabled())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 12), -100.0)
        logits[:, -1, self.selections[len(self.calls) - 1]] = 100.0
        logits[:, :-1, 11] = 1000.0
        return Output(logits)


def test_generation_uses_last_logits_is_greedy_and_has_no_gradients():
    model = RecordingModel([5, 6, 7])
    generated, trace = greedy_generate(model, torch.tensor([1, 4]), 3, 2, 8)
    assert generated.tolist() == [1, 4, 5, 6, 7]
    assert [row["selected_token_id"] for row in trace] == [5, 6, 7]
    assert not any(model.grad_flags)
    assert not model.training


def test_generation_stops_at_eos():
    model = RecordingModel([5, 2, 7])
    generated, trace = greedy_generate(model, torch.tensor([1]), 10, 2, 4)
    assert generated.tolist() == [1, 5, 2]
    assert len(trace) == 2
    assert trace[-1]["stopped_by_eos"] is True


def test_generation_crops_context_and_respects_maximum():
    model = RecordingModel([5, 6, 7])
    generated, trace = greedy_generate(model, torch.tensor([1, 4, 4, 4]), 3, 2, 3)
    assert generated.numel() == 7
    assert all(call.shape[1] == 3 for call in model.calls)
    assert [row["forward_pass_count"] for row in trace] == [1, 2, 3]
