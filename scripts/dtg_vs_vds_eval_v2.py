"""Re-run inference with smoothed features (v2) and compare vs v1 + vs k_gt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_high_density import (  # noqa: E402
    SEED, TARGET, get_feat_cols, train_single_model,
)

ORIG = Path("/Users/park/ml-project/data/features/dataset_1km.parquet")
BN10K = Path("/Users/park/ml-project/data/features/dataset_bottleneck_10k.parquet")
BN1L5K = Path("/Users/park/ml-project/data/features/dataset_bottleneck_1lane_5k.parquet")
OUT_DIR = Path("/Users/park/ml-project/results")


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true > 0.5
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() else 0
    return {
        "n": int(len(y_true)),
        "mae": float(round(mean_absolute_error(y_true, y_pred), 4)),
        "rmse": float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)),
        "mape": float(round(mape, 2)),
        "r2": float(round(r2_score(y_true, y_pred), 4)),
    }


def main() -> None:
    print("[1/3] training model (same as v1)...")
    df_orig = pd.read_parquet(ORIG)
    bns = []
    for p, offset in [(BN10K, 100000), (BN1L5K, 200000)]:
        if p.exists():
            d = pd.read_parquet(p); d["scenario_id"] = d["scenario_id"] + offset
            bns.append(d)
    df_bn = pd.concat(bns, ignore_index=True) if bns else pd.DataFrame()
    feat_cols = get_feat_cols(df_orig)
    for c in feat_cols:
        if c not in df_bn.columns:
            df_bn[c] = 0.0
    keep = list(dict.fromkeys(feat_cols + [TARGET, "scenario_id", "num_lanes", "speed_limit"]))
    df_train = pd.concat([df_orig[keep], df_bn[keep]], ignore_index=True)

    sids_orig = df_orig["scenario_id"].unique()
    tr_o, temp_o = train_test_split(sids_orig, test_size=0.3, random_state=SEED)
    va_o, _ = train_test_split(temp_o, test_size=0.5, random_state=SEED)
    sids_bn = df_train["scenario_id"].unique()
    sids_bn = sids_bn[sids_bn >= 100000]
    tr_bn, temp_bn = train_test_split(sids_bn, test_size=0.3, random_state=SEED)
    va_bn, _ = train_test_split(temp_bn, test_size=0.5, random_state=SEED)
    tr_a = np.concatenate([tr_o, tr_bn]); va_a = np.concatenate([va_o, va_bn])
    model = train_single_model(df_train, feat_cols, tr_a, va_a)

    print("[2/3] inferring on smoothed DTG features (v2)...")
    dtg = pd.read_parquet(OUT_DIR / "dtg_slices_features_smoothed.parquet")
    for c in feat_cols:
        if c not in dtg.columns:
            dtg[c] = 0.0
    dtg["k_hat"] = model.predict(dtg[feat_cols].values)
    print(f"  k_hat range: [{dtg['k_hat'].min():.2f}, {dtg['k_hat'].max():.2f}], mean {dtg['k_hat'].mean():.2f}")
    print(f"  k_gt_qu mean: {dtg['k_gt_qu'].mean():.2f}")

    # Save
    dtg.to_parquet(OUT_DIR / "dtg_vds_eval_v2.parquet", index=False)

    print("\n[3/3] comparing v1 (raw 3Hz) vs v2 (smoothed 1Hz) vs k_gt...")
    v1 = pd.read_parquet(OUT_DIR / "dtg_vds_eval.parquet")
    print(f"\n{'metric':>10} | {'v1 (raw 3Hz)':>20} | {'v2 (smoothed 1Hz)':>20}")
    print("-" * 60)
    for gt in ("k_gt_qu", "k_gt_occ"):
        s1 = v1.dropna(subset=[gt]); s1 = s1[s1[gt] >= 0]
        s2 = dtg.dropna(subset=[gt]); s2 = s2[s2[gt] >= 0]
        m1 = metrics(s1[gt], s1["k_hat"])
        m2 = metrics(s2[gt], s2["k_hat"])
        print(f"\n  vs {gt}:")
        print(f"  {'n':>10} | {m1['n']:>20} | {m2['n']:>20}")
        print(f"  {'MAE':>10} | {m1['mae']:>20.2f} | {m2['mae']:>20.2f}")
        print(f"  {'MAPE':>10} | {m1['mape']:>19.1f}% | {m2['mape']:>19.1f}%")
        print(f"  {'R²':>10} | {m1['r2']:>20.3f} | {m2['r2']:>20.3f}")

    # Per-bin (v2)
    print("\n[v2] per density bin (vs k_gt_qu):")
    sub = dtg.dropna(subset=["k_gt_qu"])
    sub = sub[sub["k_gt_qu"] >= 0]
    bins = [0, 8, 16, 24, 40, 1000]
    labels = ["0-8", "8-16", "16-24", "24-40", "40+"]
    sub["bin"] = pd.cut(sub["k_gt_qu"], bins=bins, labels=labels)
    for lab in labels:
        ss = sub[sub["bin"] == lab]
        if len(ss) >= 10:
            m = metrics(ss["k_gt_qu"], ss["k_hat"])
            print(f"  {lab:>6}  n={m['n']:>4}  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%  R²={m['r2']:.3f}")

    # Filtered subset (in distribution)
    in_dist = (dtg["num_lanes"] <= 3) & (dtg["speed_limit"] <= 28)
    sub_in = dtg[in_dist].dropna(subset=["k_gt_qu"])
    sub_in = sub_in[sub_in["k_gt_qu"] >= 0]
    if len(sub_in) > 50:
        m = metrics(sub_in["k_gt_qu"], sub_in["k_hat"])
        print(f"\n[v2 / in-distribution] num_lanes≤3 + speed_limit≤100km/h: n={m['n']:,}, "
              f"MAE={m['mae']:.2f}, R²={m['r2']:.3f}, MAPE={m['mape']:.1f}%")

    # Scatter v1 vs v2
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, df_eval, label in zip(axes, [v1, dtg], ["v1: raw 3Hz", "v2: smoothed 1Hz"]):
        s = df_eval.dropna(subset=["k_gt_qu"])
        s = s[s["k_gt_qu"] >= 0]
        ax.scatter(s["k_gt_qu"], s["k_hat"], s=3, alpha=0.3, color="#2b2d42")
        lim = max(s["k_gt_qu"].max(), s["k_hat"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
        m = metrics(s["k_gt_qu"], s["k_hat"])
        ax.set_xlabel(f"VDS ground truth — k_gt_qu  [veh/km/lane]")
        ax.set_ylabel("Model k_hat  [veh/km/lane]")
        ax.set_title(f"{label}\nn={m['n']:,}, MAE={m['mae']:.2f}, MAPE={m['mape']:.1f}%, R²={m['r2']:.3f}")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    out_png = OUT_DIR / "dtg_vds_scatter_v1_vs_v2.png"
    plt.savefig(out_png, dpi=120)
    print(f"\n→ wrote {out_png}")


if __name__ == "__main__":
    main()
