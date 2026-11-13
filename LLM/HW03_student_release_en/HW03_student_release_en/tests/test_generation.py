from __future__ import annotations

import torch

from src.config import MiniGPTConfig
from src.generation import generate_with_kv_cache, rebuild_cache_from_window
from src.model import MiniGPT


def _model() -> MiniGPT:
    torch.manual_seed(23)
    return MiniGPT(MiniGPTConfig(dropout=0.0)).eval()


def _uncached_greedy(
    model: MiniGPT,
    prompt: list[int],
    *,
    max_new_tokens: int,
    context_length: int,
) -> list[int]:
    token_ids = list(prompt)
    generated: list[int] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(
                token_ids[-context_length:],
                dtype=torch.long,
            ).unsqueeze(0)
            selected = int(torch.argmax(model(inputs).logits[:, -1, :], dim=-1).item())
            generated.append(selected)
            token_ids.append(selected)
    return generated


def test_cached_greedy_matches_uncached() -> None:
    model = _model()
    prompt = [1, 76, 77, 78]
    expected = _uncached_greedy(
        model,
        prompt,
        max_new_tokens=8,
        context_length=32,
    )
    result = generate_with_kv_cache(
        model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        context_length=32,
        seed=10,
    )
    assert result.generated_token_ids == expected
    assert result.prefill_calls == 1
    assert result.decode_calls == 7
    assert result.cache_rebuilds == 0
    assert result.processed_token_positions == len(prompt) + 7


def test_context_rebuild_matches_uncached_sliding_window() -> None:
    model = _model()
    prompt = [1, 80, 81, 82]
    expected = _uncached_greedy(
        model,
        prompt,
        max_new_tokens=10,
        context_length=5,
    )
    result = generate_with_kv_cache(
        model,
        prompt,
        max_new_tokens=10,
        eos_token_id=None,
        context_length=5,
    )
    assert result.generated_token_ids == expected
    assert result.cache_rebuilds > 0
    assert result.prefill_calls == result.cache_rebuilds + 1
    assert len(result.trace) == 10


def test_zero_new_tokens_performs_no_forward_call() -> None:
    model = _model()
    result = generate_with_kv_cache(
        model,
        [1, 7],
        max_new_tokens=0,
        eos_token_id=None,
        context_length=8,
    )
    assert result.generated_token_ids == []
    assert result.prefill_calls == 0
    assert result.decode_calls == 0
    assert result.processed_token_positions == 0


def test_eos_stops_immediately() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = generate_with_kv_cache(
        model,
        [1, 7],
        max_new_tokens=5,
        eos_token_id=0,
        context_length=8,
    )
    assert result.generated_token_ids == [0]
    assert result.stop_reason == "eos"
    assert result.decode_calls == 0


def test_rebuild_retains_latest_window() -> None:
    model = _model()
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    retained, output = rebuild_cache_from_window(
        model,
        token_ids,
        context_length=4,
    )
    assert retained.tolist() == [[3, 4, 5, 6]]
    assert all(cache.sequence_length == 4 for cache in output.past_key_values)


def test_trace_schema_and_no_gradients() -> None:
    model = _model()
    result = generate_with_kv_cache(
        model,
        [1, 7, 8],
        max_new_tokens=3,
        eos_token_id=None,
        context_length=8,
    )
    required = {
        "step",
        "phase",
        "input_length",
        "effective_context_length",
        "cache_length_before",
        "cache_length_after",
        "selected_token_id",
        "selected_token_display",
        "selected_probability",
        "prefill_call_count",
        "decode_call_count",
        "cache_rebuild_count",
        "processed_token_positions",
        "stopped_by_eos",
    }
    assert all(required <= row.keys() for row in result.trace)
    assert all(parameter.grad is None for parameter in model.parameters())
