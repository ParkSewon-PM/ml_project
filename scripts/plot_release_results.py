"""Render the latest README results figures.

Builds two standalone figures from the v0.5.x high-density evaluation:
1. aligned same-slice rerun by probe count (after_aligned)
2. deployable link-level fusion comparison by probe count (after_deploy)

Source of truth: results/multi_probe/high_density_full_eval_v2.json
This matches the numbers shown in the README results tables.

Usage:
    python scripts/plot_release_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "results/multi_probe/high_density_full_eval_v2.json"
ALIGNED_OUT_PATH = ROOT / "docs/images/aligned_research_results.png"
DEPLOY_OUT_PATH = ROOT / "docs/images/deployable_fusion_results.png"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_aligned(eval_data: dict, probes: list[int]) -> None:
    aligned = eval_data["after_aligned"]
    aligned_r2 = [aligned[f"N{n}_overall"]["r2"] for n in probes]
    aligned_mae = [aligned[f"N{n}_overall"]["mae"] for n in probes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.plot(probes, aligned_r2, color="#1f5aa6", marker="o", linewidth=2.8, markersize=8)
    ax.fill_between(
        probes,
        aligned_r2,
        [min(aligned_r2)] * len(aligned_r2),
        color="#1f5aa6",
        alpha=0.08,
    )
    ax.set_title("Aligned Research Setting", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Probes")
    ax.set_ylabel("R²")
    ax.set_xticks(probes)
    ax.set_ylim(0.92, 0.975)

    for x, y, mae in zip(probes, aligned_r2, aligned_mae):
        ax.annotate(
            f"R² {y:.3f}\nMAE {mae:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.text(
        0.03,
        0.05,
        "High-density extended testset\nsame 1 km slice, XGBoost",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#355070",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef4ff", "edgecolor": "#c9daf8"},
    )

    ALIGNED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ALIGNED_OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ALIGNED_OUT_PATH}")


def plot_deployable(eval_data: dict, probes: list[int]) -> None:
    deploy = eval_data["after_deploy"]
    deploy_r2 = {
        "Simple mean": [deploy[f"simple_mean_N{n}_overall"]["r2"] for n in probes],
        "CF-softmax": [deploy[f"cf_softmax_N{n}_overall"]["r2"] for n in probes],
        "Bayesian+CF": [deploy[f"bayesian_cf_N{n}_overall"]["r2"] for n in probes],
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    colors = {
        "Simple mean": "#adb5bd",
        "CF-softmax": "#f4a261",
        "Bayesian+CF": "#2a9d8f",
    }
    for label, values in deploy_r2.items():
        ax.plot(
            probes,
            values,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=colors[label],
            label=label,
        )
    ax.set_title("Deployable Link-Level Fusion", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Probes")
    ax.set_ylabel("R²")
    ax.set_xticks(probes)
    ax.set_ylim(0.92, 0.97)

    # Legend (top-left) already lists labels with colors — no need to repeat them
    # at line ends. Annotate only the final R² value per line, stacked vertically
    # so the three near-identical N=5 values don't overlap.
    x_end = probes[-1]
    final_vals = [(label, deploy_r2[label][-1]) for label in deploy_r2]
    # Sort top-to-bottom by R² so vertical offsets line up with line ordering
    final_vals.sort(key=lambda kv: kv[1], reverse=True)
    vertical_offsets = [16, 0, -16]  # pts, top to bottom
    for (label, y), dy in zip(final_vals, vertical_offsets):
        ax.annotate(
            f"R² {y:.3f}",
            (x_end, y),
            textcoords="offset points",
            xytext=(12, dy),
            ha="left",
            va="center",
            fontsize=9,
            color=colors[label],
        )

    ax.legend(frameon=False, loc="upper left")

    ax.text(
        0.03,
        0.05,
        "Unequal traversal boundaries\n32-input single-probe model",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#245c54",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#edf8f6", "edgecolor": "#b7e4dc"},
    )

    DEPLOY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEPLOY_OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DEPLOY_OUT_PATH}")


def main() -> None:
    eval_data = load_json(EVAL_PATH)
    probes = [1, 2, 3, 5]
    plot_aligned(eval_data, probes)
    plot_deployable(eval_data, probes)


if __name__ == "__main__":
    main()
