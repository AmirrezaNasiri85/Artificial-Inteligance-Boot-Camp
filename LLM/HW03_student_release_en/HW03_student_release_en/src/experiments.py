"""Transparent evidence-generation helpers used by all three HW03 notebooks."""

from __future__ import annotations

import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch.nn import functional as F

from .accounting import (
    cached_processed_token_positions,
    estimate_kv_cache_bytes,
    max_concurrent_requests,
    tensor_nbytes,
    uncached_processed_token_positions,
)
from .cache import build_incremental_causal_mask
from .config import MiniGPTConfig, count_parameters, set_seed
from .generation import generate_with_kv_cache
from .model import MiniGPT, model_forward_with_cache
from .model_io import load_pretrained_model, verify_assets
from .plotting import (
    plot_kv_cache_growth,
    plot_memory_capacity,
    plot_prefill_decode_latency,
    plot_processed_positions,
    plot_quantization_error,
)
from .quantization import dequantize_state_dict, quantize_state_dict
from .tokenizer import encode_text
from .validation import validate_artifacts

SEED = 1403


def setup(project_root: str | Path = "."):
    root = Path(project_root)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    torch.set_num_threads(max(1, min(4, psutil.cpu_count(logical=False) or 1)))
    assets = verify_assets(root / "assets")
    model = load_pretrained_model(root / "assets").cpu()
    model.eval()
    assert (
        count_parameters(model) == 644736
        and model.output_projection.weight is model.token_embedding.weight
    )
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "parameter_count": count_parameters(model),
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "context_length": model.config.context_length,
        "vocabulary_size": model.config.vocab_size,
        "seed": SEED,
    }
    (root / "results/environment.json").write_text(
        json.dumps(env, indent=2) + "\n", encoding="utf-8"
    )
    return root, model, assets["prompts"], env


def _uncached(model, prompt, max_new_tokens, context=128):
    hist = list(prompt)
    out = []
    rows = []
    logits = []
    cumulative = 0
    with torch.inference_mode():
        for step in range(max_new_tokens):
            active = hist[-context:]
            ids = torch.tensor([active], dtype=torch.long)
            current = model(ids).logits[:, -1, :]
            p = torch.softmax(current, dim=-1)
            token = int(torch.argmax(p, dim=-1).item())
            cumulative += len(active)
            out.append(token)
            logits.append(current.detach().cpu())
            rows.append(
                {
                    "step": step,
                    "input_length": len(active),
                    "selected_token_id": token,
                    "selected_probability": float(p[0, token].item()),
                    "processed_token_positions": cumulative,
                }
            )
            hist.append(token)
    return out, rows, logits


def run_uncached_baseline(root, model, prompts):
    prompt = encode_text(prompts[1]["text"], add_bos=True)
    tokens, rows, _ = _uncached(model, prompt, 12)
    df = pd.DataFrame(rows)
    df.to_csv(root / "results/uncached_trace.csv", index=False)
    assert int(df.iloc[-1].processed_token_positions) == uncached_processed_token_positions(
        len(prompt), 12, 128
    )
    return df


def run_kv_structure(root, model, prompts):
    ids = torch.tensor([encode_text(prompts[0]["text"], add_bos=True)], dtype=torch.long)
    with torch.inference_mode():
        output = model_forward_with_cache(model, ids)
    rows = []
    for i, c in enumerate(output.past_key_values):
        for name, t in [("key", c.key), ("value", c.value)]:
            rows.append(
                {
                    "layer": i,
                    "tensor": name,
                    "batch": t.shape[0],
                    "heads": t.shape[1],
                    "sequence_length": t.shape[2],
                    "head_dimension": t.shape[3],
                    "dtype": str(t.dtype),
                    "bytes": tensor_nbytes(t),
                }
            )
    shapes = pd.DataFrame(rows)
    shapes.to_csv(root / "results/kv_cache_shapes.csv", index=False)
    growth = pd.DataFrame(
        [
            {
                "sequence_length": n,
                "cache_bytes": estimate_kv_cache_bytes(
                    layers=model.config.n_layers,
                    batch_size=1,
                    heads=model.config.n_heads,
                    sequence_length=n,
                    head_dimension=model.config.d_model // model.config.n_heads,
                    bytes_per_element=4,
                ),
            }
            for n in [1, 8, 16, 32, 64, 96, 128]
        ]
    )
    plot_kv_cache_growth(growth, root / "figures/kv_cache_growth.png")
    return shapes, growth


def inspect_incremental_mask():
    mask = build_incremental_causal_mask(3, 8, 5, device=torch.device("cpu"))
    assert mask.shape == (3, 8) and not mask[:, :5].any()
    return mask.to(torch.int8)


def run_prefill_decode(root, model):
    ids = torch.tensor(
        [encode_text("Cache mechanics and position offsets.", add_bos=True)], dtype=torch.long
    )
    split = min(12, ids.shape[1] - 1)
    with torch.inference_mode():
        full = model(ids).logits
        first = model_forward_with_cache(model, ids[:, :split])
        second = model_forward_with_cache(
            model, ids[:, split:], past_key_values=first.past_key_values
        )
    d1 = float((full[:, :split] - first.logits).abs().max().item())
    d2 = float((full[:, split:] - second.logits).abs().max().item())
    assert d1 <= 1e-5 and d2 <= 1e-5
    df = pd.DataFrame(
        [
            {
                "phase": "prefill",
                "input_length": split,
                "past_length": 0,
                "cache_length": first.past_key_values[0].sequence_length,
                "max_logit_difference": d1,
            },
            {
                "phase": "decode_chunk",
                "input_length": ids.shape[1] - split,
                "past_length": split,
                "cache_length": second.past_key_values[0].sequence_length,
                "max_logit_difference": d2,
            },
        ]
    )
    df.to_csv(root / "results/prefill_decode_trace.csv", index=False)
    return df


def run_cache_equivalence(root, model, prompts):
    prompt = encode_text(prompts[2]["text"], add_bos=True)
    uncached, _, uncached_logits = _uncached(model, prompt, 12)
    with torch.inference_mode():
        output = model_forward_with_cache(model, torch.tensor([prompt], dtype=torch.long))
        cached = []
        diffs = []
        for step in range(12):
            current = output.logits[:, -1, :]
            diffs.append(float((current.cpu() - uncached_logits[step]).abs().max().item()))
            token = int(torch.argmax(current, dim=-1).item())
            cached.append(token)
            if step != 11:
                output = model_forward_with_cache(
                    model,
                    torch.tensor([[token]], dtype=torch.long),
                    past_key_values=output.past_key_values,
                )
    result = generate_with_kv_cache(
        model, prompt, max_new_tokens=12, eos_token_id=None, context_length=128, seed=SEED
    )
    pd.DataFrame(result.trace).to_csv(root / "results/cached_generation_trace.csv", index=False)
    uw = uncached_processed_token_positions(len(prompt), 12, 128)
    cw = cached_processed_token_positions(len(prompt), 12, 128)
    obj = {
        "prompt_length": len(prompt),
        "generated_tokens": 12,
        "token_ids_equal": uncached == cached == result.generated_token_ids,
        "max_logit_difference": max(diffs),
        "uncached_processed_token_positions": uw,
        "cached_processed_token_positions": cw,
        "processed_position_ratio": cw / uw,
    }
    assert obj["token_ids_equal"] and obj["max_logit_difference"] <= 1e-5
    (root / "results/cache_equivalence.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8"
    )
    return obj


def run_benchmark(root, model):
    source = encode_text("Holmes inspected the evidence carefully. " * 8, add_bos=True)
    rows = []
    for plen in [16, 48, 96]:
        prompt = source[:plen]
        start = time.perf_counter()
        u, _, _ = _uncached(model, prompt, 8)
        uncached_seconds = time.perf_counter() - start
        ids = torch.tensor([prompt], dtype=torch.long)
        with torch.inference_mode():
            start = time.perf_counter()
            output = model_forward_with_cache(model, ids)
            prefill = time.perf_counter() - start
            c = []
            decode = []
            for step in range(8):
                token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
                c.append(token)
                if step != 7:
                    start = time.perf_counter()
                    output = model_forward_with_cache(
                        model,
                        torch.tensor([[token]], dtype=torch.long),
                        past_key_values=output.past_key_values,
                    )
                    decode.append(time.perf_counter() - start)
        uw = uncached_processed_token_positions(plen, 8, 128)
        cw = cached_processed_token_positions(plen, 8, 128)
        final_len = plen + 7
        cache = estimate_kv_cache_bytes(
            layers=model.config.n_layers,
            batch_size=1,
            heads=model.config.n_heads,
            sequence_length=final_len,
            head_dimension=model.config.d_model // model.config.n_heads,
            bytes_per_element=4,
        )
        rows.append(
            {
                "prompt_length": plen,
                "generated_tokens": 8,
                "uncached_seconds": uncached_seconds,
                "prefill_seconds": prefill,
                "mean_decode_seconds": float(np.mean(decode)),
                "time_to_first_token_seconds": prefill,
                "cached_total_seconds": prefill + sum(decode),
                "uncached_processed_token_positions": uw,
                "cached_processed_token_positions": cw,
                "cache_bytes": cache,
                "token_ids_equal": u == c,
            }
        )
    df = pd.DataFrame(rows)
    assert df.token_ids_equal.all()
    df.to_csv(root / "results/inference_benchmark.csv", index=False)
    plot_processed_positions(df, root / "figures/processed_token_positions.png")
    plot_prefill_decode_latency(df, root / "figures/prefill_decode_latency.png")
    return df


def run_context_rebuild(root, model):
    prompt = encode_text(
        "A deliberately small context demonstrates cache rebuilding.", add_bos=True
    )[:16]
    result = generate_with_kv_cache(
        model, prompt, max_new_tokens=16, eos_token_id=None, context_length=24, seed=SEED
    )
    df = pd.DataFrame(result.trace)
    mask = df.cache_rebuild_count.diff().fillna(df.cache_rebuild_count) > 0
    out = df.loc[
        mask,
        [
            "step",
            "phase",
            "cache_rebuild_count",
            "cache_length_before",
            "cache_length_after",
            "processed_token_positions",
        ],
    ].copy()
    assert not out.empty
    out.to_csv(root / "results/context_rebuild_trace.csv", index=False)
    return out


def run_quantization(root, model):
    state = model.state_dict()
    q = quantize_state_dict(state)
    restored = dequantize_state_dict(q)
    a = io.BytesIO()
    b = io.BytesIO()
    torch.save(state, a)
    torch.save(q, b)
    rows = []
    for name in q["quantized_keys"]:
        orig = state[name].detach().cpu()
        rec = restored[name].detach().cpu()
        entry = q["tensors"][name]["value"]
        diff = (orig.float() - rec.float()).abs()
        rows.append(
            {
                "tensor_name": name,
                "numel": orig.numel(),
                "scale": float(entry.scale),
                "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
                "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
                "original_bytes": tensor_nbytes(orig),
                "quantized_value_bytes": tensor_nbytes(entry.values),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(root / "results/quantization_tensor_metrics.csv", index=False)
    plot_quantization_error(df, root / "figures/quantization_error.png")
    obj = {
        "original_serialized_bytes": len(a.getvalue()),
        "quantized_serialized_bytes": len(b.getvalue()),
        "compression_ratio": len(b.getvalue()) / len(a.getvalue()),
        "quantized_tensor_count": len(q["quantized_keys"]),
        "runtime_memory_claim": "not_reduced_after_fp32_dequantization",
    }
    assert obj["compression_ratio"] < 1
    (root / "results/quantization_summary.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8"
    )
    return restored, obj


def run_quantization_quality(root, model, prompts, restored):
    qmodel = MiniGPT(MiniGPTConfig()).cpu()
    qmodel.load_state_dict(restored)
    qmodel.eval()
    rows = []
    for prompt in prompts:
        ids = encode_text(prompt["text"], add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            lo = model(x).logits
            lq = qmodel(x).logits
        targets = x[:, 1:]
        loss_o = float(
            F.cross_entropy(
                lo[:, :-1, :].reshape(-1, model.config.vocab_size), targets.reshape(-1)
            ).item()
        )
        loss_q = float(
            F.cross_entropy(
                lq[:, :-1, :].reshape(-1, model.config.vocab_size), targets.reshape(-1)
            ).item()
        )
        po = torch.softmax(lo[:, -1, :], dim=-1)
        pq = torch.softmax(lq[:, -1, :], dim=-1)
        top = bool(torch.argmax(po, dim=-1).item() == torch.argmax(pq, dim=-1).item())
        tv = float(0.5 * (po - pq).abs().sum().item())
        go = generate_with_kv_cache(
            model, ids, max_new_tokens=12, eos_token_id=None, context_length=128, seed=SEED
        )
        gq = generate_with_kv_cache(
            qmodel, ids, max_new_tokens=12, eos_token_id=None, context_length=128, seed=SEED
        )
        rows.append(
            {
                "prompt_id": prompt["id"],
                "original_loss": loss_o,
                "quantized_loss": loss_q,
                "loss_delta": loss_q - loss_o,
                "top1_agreement": top,
                "total_variation_distance": tv,
                "greedy_token_agreement": go.generated_token_ids == gq.generated_token_ids,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(root / "results/quantization_quality.csv", index=False)
    return df


def run_capacity(root):
    G = 1024**3
    budget = 16 * G
    reserved = 2 * G
    cfg = [
        ("MiniGPT 0.645M", 644736, 3, 4, 32),
        ("Hypothetical 100M", 100_000_000, 12, 12, 64),
        ("Hypothetical 350M", 350_000_000, 24, 16, 64),
        ("Hypothetical 700M", 700_000_000, 24, 16, 128),
    ]
    rows = []
    for label, params, layers, heads, hd in cfg:
        for seq in [128, 512, 2048]:
            weights = params * 4
            cache = estimate_kv_cache_bytes(
                layers=layers,
                batch_size=1,
                heads=heads,
                sequence_length=seq,
                head_dimension=hd,
                bytes_per_element=4,
            )
            rows.append(
                {
                    "model_label": label,
                    "parameter_count": params,
                    "sequence_length": seq,
                    "weight_bytes_fp32": weights,
                    "kv_cache_bytes_per_request": cache,
                    "memory_budget_bytes": budget,
                    "reserved_bytes": reserved,
                    "max_concurrent_requests_fp32": max_concurrent_requests(
                        memory_budget_bytes=budget,
                        reserved_bytes=reserved,
                        model_bytes=weights,
                        cache_bytes_per_request=cache,
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(root / "results/capacity_plan.csv", index=False)
    plot_memory_capacity(df, root / "figures/memory_capacity.png")
    return df


def write_summary(root, quality):
    eq = json.loads((root / "results/cache_equivalence.json").read_text())
    q = json.loads((root / "results/quantization_summary.json").read_text())
    overflow = pd.read_csv(root / "results/context_rebuild_trace.csv")
    df = pd.DataFrame(
        [
            {
                "strategy": "uncached_fp32",
                "exact_or_approximate": "exact",
                "greedy_equivalent": True,
                "processed_token_work": eq["uncached_processed_token_positions"],
                "cache_memory": "none",
                "checkpoint_storage": "FP32",
                "main_advantage": "simple baseline",
                "main_failure_mode": "recomputes previous representations",
            },
            {
                "strategy": "cached_fp32",
                "exact_or_approximate": "exact",
                "greedy_equivalent": eq["token_ids_equal"],
                "processed_token_work": eq["cached_processed_token_positions"],
                "cache_memory": "grows with layers and context",
                "checkpoint_storage": "FP32",
                "main_advantage": "reuses previous keys and values",
                "main_failure_mode": "requires KV memory",
            },
            {
                "strategy": "cached_quantized_dequantized",
                "exact_or_approximate": "approximate",
                "greedy_equivalent": bool(quality.greedy_token_agreement.all()),
                "processed_token_work": eq["cached_processed_token_positions"],
                "cache_memory": "FP32 in this educational implementation",
                "checkpoint_storage": f"{q['compression_ratio']:.3f}x serialized ratio",
                "main_advantage": "smaller serialized weights",
                "main_failure_mode": "approximation can alter predictions",
            },
            {
                "strategy": "cached_context_rebuild",
                "exact_or_approximate": "exact for retained window",
                "greedy_equivalent": True,
                "processed_token_work": int(overflow.processed_token_positions.max()),
                "cache_memory": "bounded by retained context",
                "checkpoint_storage": "FP32",
                "main_advantage": "valid absolute positions after overflow",
                "main_failure_mode": "rebuild overhead",
            },
        ]
    )
    df.to_csv(root / "results/inference_strategy_summary.csv", index=False)
    return validate_artifacts(root), df
