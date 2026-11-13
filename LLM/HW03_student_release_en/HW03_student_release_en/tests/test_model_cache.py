from __future__ import annotations

import pytest
import torch

from src.config import MiniGPTConfig, count_parameters
from src.model import MiniGPT, model_forward_with_cache


@pytest.fixture
def model() -> MiniGPT:
    torch.manual_seed(7)
    return MiniGPT(MiniGPTConfig(dropout=0.0)).eval()


def test_prefill_matches_original_forward(model: MiniGPT) -> None:
    input_ids = torch.randint(0, model.config.vocab_size, (2, 11))
    with torch.inference_mode():
        expected = model(input_ids).logits
        actual = model_forward_with_cache(model, input_ids)
    assert actual.logits.shape == expected.shape
    assert torch.allclose(actual.logits, expected, atol=1e-5, rtol=1e-5)


def test_incremental_chunks_match_full_forward(model: MiniGPT) -> None:
    input_ids = torch.randint(0, model.config.vocab_size, (2, 13))
    with torch.inference_mode():
        expected = model(input_ids).logits
        first = model_forward_with_cache(model, input_ids[:, :7])
        second = model_forward_with_cache(
            model,
            input_ids[:, 7:],
            past_key_values=first.past_key_values,
        )
    assert torch.allclose(
        second.logits,
        expected[:, 7:],
        atol=1e-5,
        rtol=1e-5,
    )


def test_cache_shapes_and_growth(model: MiniGPT) -> None:
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    with torch.inference_mode():
        first = model_forward_with_cache(model, input_ids)
        second = model_forward_with_cache(
            model,
            torch.randint(0, model.config.vocab_size, (2, 2)),
            past_key_values=first.past_key_values,
        )
    assert len(first.past_key_values) == model.config.n_layers
    for cache in first.past_key_values:
        assert cache.key.shape == (2, 4, 5, 32)
        assert cache.value.shape == (2, 4, 5, 32)
    for cache in second.past_key_values:
        assert cache.key.shape == (2, 4, 7, 32)


def test_parameter_count_and_state_keys_do_not_change(model: MiniGPT) -> None:
    before_keys = tuple(model.state_dict())
    before_count = count_parameters(model)
    with torch.inference_mode():
        model_forward_with_cache(
            model,
            torch.randint(0, model.config.vocab_size, (1, 4)),
        )
    assert count_parameters(model) == before_count == 644_736
    assert tuple(model.state_dict()) == before_keys
    assert model.output_projection.weight is model.token_embedding.weight


def test_original_training_forward_remains_valid(model: MiniGPT) -> None:
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    output = model(input_ids, targets=input_ids)
    assert output.logits.shape == (2, 6, model.config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_cache_rejects_context_overflow(model: MiniGPT) -> None:
    with torch.inference_mode():
        first = model_forward_with_cache(
            model,
            torch.randint(
                0,
                model.config.vocab_size,
                (1, model.config.context_length),
            ),
        )
    with pytest.raises(ValueError):
        model_forward_with_cache(
            model,
            torch.ones((1, 1), dtype=torch.long),
            past_key_values=first.past_key_values,
        )


def test_cache_rejects_wrong_layer_count(model: MiniGPT) -> None:
    with torch.inference_mode():
        first = model_forward_with_cache(
            model,
            torch.randint(0, model.config.vocab_size, (1, 4)),
        )
    with pytest.raises(ValueError):
        model_forward_with_cache(
            model,
            torch.ones((1, 1), dtype=torch.long),
            past_key_values=first.past_key_values[:-1],
        )
