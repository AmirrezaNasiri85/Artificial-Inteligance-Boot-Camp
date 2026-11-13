"""Required matplotlib-only plotting infrastructure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_next_token_distribution(frame: pd.DataFrame, path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(frame["token_display"].astype(str), frame["probability"], color="#3569a8")
    axis.set(title="Highest-probability next tokens", xlabel="Token", ylabel="Probability")
    axis.tick_params(axis="x", rotation=55)
    _save(fig, path)


def plot_temperature_distributions(frame: pd.DataFrame, path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for temperature, group in frame.groupby("temperature"):
        axis.plot(group["rank"], group["probability"], marker="o", label=f"T={temperature}")
    axis.set(title="Temperature changes probability concentration", xlabel="Token rank", ylabel="Probability")
    axis.legend()
    _save(fig, path)


def plot_second_best_decay(frame: pd.DataFrame, path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    for strategy, group in frame.groupby("strategy"):
        axis.plot(group["step"], group["cumulative_sequence_log_probability"], label=strategy)
    axis.set(title="Cumulative sequence log-probability", xlabel="Generation step", ylabel="Log-probability (nats)")
    axis.legend()
    _save(fig, path)


def plot_entropy_by_step(frame: pd.DataFrame, path: str | Path) -> None:
    summary = frame.groupby(["strategy", "step"], as_index=False)["entropy_nats"].mean()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for strategy, group in summary.groupby("strategy"):
        axis.plot(group["step"], group["entropy_nats"], alpha=0.8, label=strategy)
    axis.set(title="Mean full-distribution entropy by generation step", xlabel="Step", ylabel="Entropy (nats)")
    axis.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_diversity_vs_probability(frame: pd.DataFrame, path: str | Path) -> None:
    summary = frame.groupby("strategy", as_index=False)[["token_distinct_2", "mean_selected_model_probability"]].mean()
    fig, axis = plt.subplots(figsize=(8.5, 5))
    axis.scatter(summary["mean_selected_model_probability"], summary["token_distinct_2"], color="#b4493d")
    for row in summary.itertuples(index=False):
        axis.annotate(row.strategy, (row.mean_selected_model_probability, row.token_distinct_2), fontsize=7)
    axis.set(title="Diversity versus selected model probability", xlabel="Mean selected model probability", ylabel="Token distinct-2")
    _save(fig, path)

