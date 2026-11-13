"""Small deterministic matplotlib figures for HW01."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_causal_mask(mask: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = mask.detach().cpu().numpy().astype(int)
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    image = axis.imshow(array, cmap="RdYlGn_r", vmin=0, vmax=1)
    axis.set(title="Causal attention mask", xlabel="Key position", ylabel="Query position")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["visible", "blocked"])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_training_loss(losses: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.2)
    axis.set(title="MiniGPT CPU training", xlabel="Optimization step", ylabel="Cross-entropy loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_error_propagation(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    injection_position = next(row["position"] for row in rows if row["phase"] == "injection")
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(
        [row["position"] for row in rows],
        [row["total_variation"] for row in rows],
        marker="o",
        markersize=3,
    )
    axis.axvline(
        injection_position,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="wrong token appended",
    )
    axis.set(
        title="Distribution divergence after one injected token error",
        xlabel="Sequence position",
        ylabel="Total variation distance",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
