from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/Users/park/ml-project/docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBES = np.array([1, 2, 3, 5])

ALIGNED_R2 = {
    "1 lane": [0.430, 0.484, 0.520, 0.597],
    "2 lanes": [0.525, 0.672, 0.727, 0.779],
    "3 lanes": [0.490, 0.664, 0.729, 0.800],
}

ALIGNED_MAE = {
    "1 lane": [2.471, 2.364, 2.249, 2.039],
    "2 lanes": [2.373, 1.956, 1.777, 1.590],
    "3 lanes": [2.360, 1.889, 1.717, 1.493],
}

DEPLOY_BAYES_R2 = {
    "1 lane": [0.471, 0.501, 0.529, 0.547],
    "2 lanes": [0.538, 0.634, 0.662, 0.678],
    "3 lanes": [0.550, 0.667, 0.687, 0.715],
}

DEPLOY_BAYES_MAE = {
    "1 lane": [2.380, 2.257, 2.204, 2.161],
    "2 lanes": [2.284, 2.048, 1.990, 1.935],
    "3 lanes": [2.246, 2.029, 1.943, 1.844],
}

N5_R2 = {
    "Simple mean": [0.545, 0.659, 0.672],
    "CF-softmax": [0.548, 0.669, 0.688],
    "Bayesian+CF": [0.547, 0.678, 0.715],
}

N5_MAE = {
    "Simple mean": [2.266, 2.051, 1.969],
    "CF-softmax": [2.253, 2.016, 1.919],
    "Bayesian+CF": [2.226, 1.974, 1.833],
}


def save_aligned_r2() -> None:
    plt.figure(figsize=(8, 5))
    for lane, vals in ALIGNED_R2.items():
        plt.plot(PROBES, vals, marker="o", linewidth=2, label=lane)
    plt.title("Aligned 1 km Multi-Probe XGBoost")
    plt.xlabel("Number of Probes (N)")
    plt.ylabel("R²")
    plt.xticks(PROBES)
    plt.ylim(0.35, 0.85)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "week6_aligned_lane_r2.png", dpi=180)
    plt.close()


def save_aligned_mae() -> None:
    plt.figure(figsize=(8, 5))
    for lane, vals in ALIGNED_MAE.items():
        plt.plot(PROBES, vals, marker="o", linewidth=2, label=lane)
    plt.title("Aligned 1 km Multi-Probe XGBoost")
    plt.xlabel("Number of Probes (N)")
    plt.ylabel("MAE")
    plt.xticks(PROBES)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "week6_aligned_lane_mae.png", dpi=180)
    plt.close()


def save_deploy_bayes_r2() -> None:
    plt.figure(figsize=(8, 5))
    for lane, vals in DEPLOY_BAYES_R2.items():
        plt.plot(PROBES, vals, marker="o", linewidth=2, label=lane)
    plt.title("Deployable Link-Level Fusion (Bayesian+CF)")
    plt.xlabel("Number of Probes (N)")
    plt.ylabel("R²")
    plt.xticks(PROBES)
    plt.ylim(0.40, 0.75)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "week6_deploy_bayesian_lane_r2.png", dpi=180)
    plt.close()


def save_deploy_bayes_mae() -> None:
    plt.figure(figsize=(8, 5))
    for lane, vals in DEPLOY_BAYES_MAE.items():
        plt.plot(PROBES, vals, marker="o", linewidth=2, label=lane)
    plt.title("Deployable Link-Level Fusion (Bayesian+CF)")
    plt.xlabel("Number of Probes (N)")
    plt.ylabel("MAE")
    plt.xticks(PROBES)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "week6_deploy_bayesian_lane_mae.png", dpi=180)
    plt.close()


def save_n5_method_compare() -> None:
    lanes = np.arange(3)
    labels = ["1 lane", "2 lanes", "3 lanes"]
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for idx, (name, vals) in enumerate(N5_R2.items()):
        axes[0].bar(lanes + (idx - 1) * width, vals, width=width, label=name)
    axes[0].set_title("N=5 Method Comparison (R²)")
    axes[0].set_xticks(lanes)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.45, 0.75)
    axes[0].grid(True, axis="y", alpha=0.25)

    for idx, (name, vals) in enumerate(N5_MAE.items()):
        axes[1].bar(lanes + (idx - 1) * width, vals, width=width, label=name)
    axes[1].set_title("N=5 Method Comparison (MAE)")
    axes[1].set_xticks(lanes)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(OUT_DIR / "week6_deploy_n5_method_compare.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    save_aligned_r2()
    save_aligned_mae()
    save_deploy_bayes_r2()
    save_deploy_bayes_mae()
    save_n5_method_compare()
