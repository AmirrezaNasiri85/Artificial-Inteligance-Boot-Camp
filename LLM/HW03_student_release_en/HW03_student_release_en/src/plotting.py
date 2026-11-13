"""Matplotlib evidence plots for HW03."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _save(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=140)
    plt.close()


def plot_kv_cache_growth(df, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(df.sequence_length, df.cache_bytes, marker="o")
    plt.xlabel("Cached sequence length (tokens)")
    plt.ylabel("KV-cache memory (bytes)")
    plt.title("KV-cache memory grows with context")
    plt.grid(alpha=0.25)
    _save(path)


def plot_processed_positions(df, path):
    plt.figure(figsize=(7, 4.5))
    x = list(range(len(df)))
    w = 0.36
    plt.bar(
        [i - w / 2 for i in x], df.uncached_processed_token_positions, width=w, label="Uncached"
    )
    plt.bar([i + w / 2 for i in x], df.cached_processed_token_positions, width=w, label="KV cached")
    plt.xticks(
        x,
        [f"P={p}, N={n}" for p, n in zip(df.prompt_length, df.generated_tokens, strict=False)],
        rotation=20,
    )
    plt.ylabel("Processed token positions")
    plt.title("Deterministic inference work")
    plt.legend()
    _save(path)


def plot_prefill_decode_latency(df, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(df.prompt_length, df.prefill_seconds, marker="o", label="Prefill")
    plt.plot(df.prompt_length, df.mean_decode_seconds, marker="o", label="Mean decode call")
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Observed CPU time (seconds)")
    plt.title("Prefill and decode workloads")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(path)


def plot_quantization_error(df, path):
    s = df.sort_values("max_abs_error", ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    plt.barh(s.tensor_name, s.max_abs_error)
    plt.gca().invert_yaxis()
    plt.xlabel("Maximum absolute reconstruction error")
    plt.ylabel("Tensor")
    plt.title("Largest INT8 reconstruction errors")
    _save(path)


def plot_memory_capacity(df, path):
    plt.figure(figsize=(8, 5))
    for label, g in df.groupby("model_label"):
        g = g.sort_values("sequence_length")
        plt.plot(g.sequence_length, g.max_concurrent_requests_fp32, marker="o", label=label)
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Maximum concurrent requests")
    plt.title("16 GiB capacity: FP32 weights + KV cache")
    plt.yscale("symlog", linthresh=1)
    plt.legend()
    plt.grid(alpha=0.25)
    _save(path)
