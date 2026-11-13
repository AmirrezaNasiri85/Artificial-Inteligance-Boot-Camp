from __future__ import annotations

import inspect

from src import accounting, cache, generation, model, quantization
from src.validation import (
    REQUIRED_ARTIFACT_COUNT,
    REQUIRED_FIGURES,
    REQUIRED_RESULTS,
)

EXPECTED_SIGNATURES = {
    "tensor_nbytes": ["tensor"],
    "estimate_kv_cache_bytes": [
        "layers",
        "batch_size",
        "heads",
        "sequence_length",
        "head_dimension",
        "bytes_per_element",
    ],
    "uncached_processed_token_positions": [
        "prompt_length",
        "generated_tokens",
        "context_length",
    ],
    "cached_processed_token_positions": [
        "prompt_length",
        "generated_tokens",
        "context_length",
        "rebuild_points",
    ],
    "max_concurrent_requests": [
        "memory_budget_bytes",
        "reserved_bytes",
        "model_bytes",
        "cache_bytes_per_request",
    ],
    "build_incremental_causal_mask": [
        "query_length",
        "key_length",
        "past_length",
        "device",
    ],
    "append_key_value": [
        "past_key_value",
        "new_key",
        "new_value",
    ],
    "attention_forward_with_cache": [
        "attention",
        "hidden_states",
        "past_key_value",
        "attention_mask",
    ],
    "model_forward_with_cache": [
        "model",
        "input_ids",
        "past_key_values",
    ],
    "generate_with_kv_cache": [
        "model",
        "prompt_ids",
        "max_new_tokens",
        "eos_token_id",
        "context_length",
        "seed",
    ],
    "rebuild_cache_from_window": [
        "model",
        "token_ids",
        "context_length",
    ],
    "quantize_symmetric_int8": ["tensor"],
    "dequantize_symmetric_int8": ["quantized"],
    "quantize_state_dict": ["state_dict"],
    "dequantize_state_dict": ["quantized_state"],
}

MODULES = [accounting, cache, model, generation, quantization]


def _function(name: str):
    for module in MODULES:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AssertionError(f"missing function: {name}")


def test_frozen_function_signatures() -> None:
    assert len(EXPECTED_SIGNATURES) == 15

    for name, parameters in EXPECTED_SIGNATURES.items():
        function = _function(name)
        actual = list(inspect.signature(function).parameters)
        assert actual == parameters, (name, actual, parameters)


def test_artifact_contract() -> None:
    assert len(REQUIRED_RESULTS) == 13
    assert len(REQUIRED_FIGURES) == 5
    assert REQUIRED_ARTIFACT_COUNT == 18
