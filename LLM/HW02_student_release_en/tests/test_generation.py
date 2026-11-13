from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import pytest
import torch

from src.generation import generate_tokens
from src.types import SelectionResult


@dataclass
class Output:
    logits: torch.Tensor


class RecordingModel(torch.nn.Module):
    def __init__(self, sequence: list[int] | None = None, vocab_size: int = 12) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.sequence = sequence or [4]
        self.vocab_size = vocab_size
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids: torch.Tensor) -> Output:
        self.calls.append({"shape": tuple(input_ids.shape), "grad": torch.is_grad_enabled(), "training": self.training})
        logits = torch.full((*input_ids.shape, self.vocab_size), -20.0)
        logits[:, :, 9] = 10.0
        logits[:, -1, self.sequence[min(len(self.calls) - 1, len(self.sequence) - 1)]] = 30.0
        return Output(logits)


def greedy_like(logits: torch.Tensor, generator: torch.Generator) -> SelectionResult:
    del generator
    token = int(torch.argmax(logits).item())
    return SelectionResult(token, 0.9, 1.0, 1, 0.2, 1, "mock_greedy")


def stochastic(logits: torch.Tensor, generator: torch.Generator) -> SelectionResult:
    del logits
    token = int(torch.randint(4, 10, (1,), generator=generator).item())
    return SelectionResult(token, 0.2, 1 / 6, token - 3, 1.5, 6, "mock_sampling")


def test_manual_forward_count_final_position_and_trace() -> None:
    model = RecordingModel([4, 5, 6])
    result = generate_tokens(model, torch.tensor([[4, 5]]), greedy_like, 3, 2, 8, 1)
    assert result.generated_token_ids == [4, 5, 6]
    assert result.forward_passes == len(result.generated_token_ids) == len(model.calls)
    assert [row["forward_pass_count"] for row in result.trace] == [1, 2, 3]
    required = {"step", "input_length", "effective_context_length", "selected_token_id", "selected_token_display", "model_probability", "sampling_probability", "rank", "entropy_nats", "candidate_count", "strategy", "forward_pass_count", "stopped_by_eos"}
    assert required <= set(result.trace[0])


def test_context_cropping_and_prompt_not_mutated() -> None:
    prompt = torch.tensor([[4, 5, 6, 7]])
    clone = prompt.clone()
    model = RecordingModel([4, 4])
    result = generate_tokens(model, prompt, greedy_like, 2, 2, 3, 1)
    assert all(call["shape"][1] == 3 for call in model.calls)
    assert result.prompt_token_ids == clone[0].tolist()
    assert torch.equal(prompt, clone)


def test_eos_and_max_token_stopping() -> None:
    eos = generate_tokens(RecordingModel([2, 4]), torch.tensor([[4]]), greedy_like, 4, 2, 8, 1)
    capped = generate_tokens(RecordingModel([4]), torch.tensor([[4]]), greedy_like, 2, 2, 8, 1)
    assert eos.stop_reason == "eos" and eos.forward_passes == 1
    assert capped.stop_reason == "max_new_tokens" and capped.forward_passes == 2


def test_eval_inference_mode_and_finite_probabilities() -> None:
    model = RecordingModel([4])
    result = generate_tokens(model, torch.tensor([[4]]), greedy_like, 1, 2, 8, 1)
    assert model.training is False
    assert model.calls == [{"shape": (1, 1), "grad": False, "training": False}]
    assert 0 < result.trace[0]["model_probability"] <= 1


def test_deterministic_and_stochastic_seed_behavior() -> None:
    prompt = torch.tensor([[4]])
    deterministic = [generate_tokens(RecordingModel([4]), prompt, greedy_like, 4, 2, 8, seed).generated_token_ids for seed in [1, 2]]
    same = [generate_tokens(RecordingModel(), prompt, stochastic, 8, 2, 8, 9).generated_token_ids for _ in range(2)]
    different = generate_tokens(RecordingModel(), prompt, stochastic, 8, 2, 8, 10).generated_token_ids
    assert deterministic[0] == deterministic[1]
    assert same[0] == same[1]
    assert same[0] != different


def test_no_model_generate_call() -> None:
    tree = ast.parse(inspect.getsource(generate_tokens))
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(name.endswith(".generate") for name in calls)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        generate_tokens(RecordingModel(), torch.tensor([4]), greedy_like, 1, 2, 8, 1)

