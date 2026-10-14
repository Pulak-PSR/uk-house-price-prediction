"""Metrics and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = {"main": "#1F4E79", "accent": "#F18F01", "grey": "#9AA5B1"}


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Metrics on the original price scale (GBP) plus R2 on log scale.

    ``mdape`` is the median absolute percentage error, robust to expensive outliers.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    actual, pred = np.exp(y_true_log), np.exp(y_pred_log)
    err = pred - actual
    ss_res = float(np.sum((y_true_log - y_pred_log) ** 2))
    ss_tot = float(np.sum((y_true_log - y_true_log.mean()) ** 2))
    return {
        "mae_gbp": float(np.mean(np.abs(err))),
        "rmse_gbp": float(np.sqrt(np.mean(err**2))),
        "mdape_pct": float(np.median(np.abs(err) / actual) * 100),
        "within_20pct": float(np.mean(np.abs(err) / actual <= 0.2) * 100),
        "r2_log": 1 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    """Bar chart of MdAPE per model (lower is better)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = results.sort_values("mdape_pct", ascending=False)
    colors = [PALETTE["main"] if i == len(data) - 1 else PALETTE["grey"] for i in range(len(data))]
    ax.barh(data.index, data["mdape_pct"], color=colors)
    for i, v in enumerate(data["mdape_pct"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Median absolute percentage error on 2025 sales (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pred_vs_actual(y_true_log: np.ndarray, y_pred_log: np.ndarray, path: Path) -> None:
    """Hexbin of predicted vs actual price (log axes)."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hb = ax.hexbin(np.exp(y_true_log), np.exp(y_pred_log), gridsize=70, xscale="log", yscale="log",
                   bins="log", cmap="Blues", mincnt=1)  # fmt: skip
    lim = [5e4, 3e6]
    ax.plot(lim, lim, color=PALETTE["accent"], lw=1.5, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual price (GBP)")
    ax.set_ylabel("Predicted price (GBP)")
    ax.legend(frameon=False)
    fig.colorbar(hb, label="Sales (log count)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error_by_region(test: pd.DataFrame, y_pred_log: np.ndarray, path: Path) -> pd.DataFrame:
    """Median % error per region; returns the table too."""
    t = test[["region", "price"]].copy()
    t["ape"] = np.abs(np.exp(y_pred_log) - t["price"]) / t["price"] * 100
    by = t.groupby("region").agg(mdape_pct=("ape", "median"), sales=("ape", "size"))
    by = by.sort_values("mdape_pct")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by.index, by["mdape_pct"], color=PALETTE["main"])
    ax.set_xlabel("Median absolute percentage error (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return by


def plot_importance(importance: pd.Series, path: Path) -> None:
    """Permutation importance bar chart."""
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.to_numpy(), color=PALETTE["main"])
    ax.set_xlabel("Increase in log-price error when feature is shuffled")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
