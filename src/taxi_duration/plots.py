"""Thin figure helpers. Each takes metrics or arrays and saves a PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "reports/figures"


def _save(fig, folder, name) -> Path:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rmse_by_segment(metrics: dict, folder=FIGURES_DIR) -> Path:
    """Bars of RMSE by segment, chosen vs baseline. Empty segments plot as 0."""
    names = list(metrics["chosen"]["rmse_by_segment"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for shift, key in ((-0.2, "chosen"), (0.2, "baseline")):
        values = [metrics[key]["rmse_by_segment"][n] or 0.0 for n in names]
        ax.bar(x + shift, values, width=0.4, label=key)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("RMSE (minutes)")
    ax.legend()
    return _save(fig, folder, "rmse_by_segment.png")


def plot_predicted_vs_actual(actual, predicted, folder=FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, s=4, alpha=0.3)
    top = max(np.max(actual), np.max(predicted))
    ax.plot([0, top], [0, top], color="black", linewidth=1)
    ax.set_xlabel("Actual (minutes)")
    ax.set_ylabel("Predicted (minutes)")
    return _save(fig, folder, "predicted_vs_actual.png")


def plot_residual_hist(actual, predicted, folder=FIGURES_DIR) -> Path:
    residual = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=50)
    ax.set_xlabel("Predicted minus actual (minutes)")
    ax.set_ylabel("Trips")
    return _save(fig, folder, "residual_hist.png")
