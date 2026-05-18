"""Train XGBoost on augmented SUMO + predict on DTG slices + compare with VDS k_gt.

Reuses train_single_model from eval_high_density.py to get the after_single model
(trained on combined dataset_1km + bottleneck augmentation), then predicts density
for every DTG 1km slice and writes the comparison table.

Output:
  results/dtg_vds_eval.parquet  — per-slice k_hat / k_gt_qu / k_gt_occ + features
  results/dtg_vds_eval_summary.json  — overall + per-bin metrics
  results/dtg_vds_scatter.png        — k_gt vs k_hat scatter
"""
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
    SEED,
    TARGET,
    get_feat_cols,
    train_single_model,
)

DTG_SLICES = Path("/Users/park/ml-project/results/dtg_slices_features.parquet")
ORIG = Path("/Users/park/ml-project/data/features/dataset_1km.parquet")
BN10K = Path("/Users/park/ml-project/data/features/dataset_bottleneck_10k.parquet")
BN1L5K = Path("/Users/park/ml-project/data/features/dataset_bottleneck_1lane_5k.parquet")
OUT_DIR = Path("/Users/park/ml-project/results")


def load_training_data() -> tuple[pd.DataFrame, list[str]]:
    df_orig = pd.read_parquet(ORIG)
    bns = []
    for p, offset in [(BN10K, 100000), (BN1L5K, 200000)]:
        if p.exists():
            d = pd.read_parquet(p)
            d["scenario_id"] = d["scenario_id"] + offset
            bns.append(d)
    df_bn = pd.concat(bns, ignore_index=True) if bns else pd.DataFrame()
    feat_cols = get_feat_cols(df_orig)
    for c in feat_cols:
        if c not in df_bn.columns:
            df_bn[c] = 0.0
    keep = list(dict.fromkeys(feat_cols + [TARGET, "scenario_id", "num_lanes", "speed_limit"]))
    df_combined = pd.concat([df_orig[keep], df_bn[keep]], ignore_index=True)
    return df_combined, feat_cols


def split_after(df_orig: pd.DataFrame, df_bn_count: int) -> tuple:
    sids_orig = df_orig["scenario_id"].unique()
    tr_o, temp_o = train_test_split(sids_orig, test_size=0.3, random_state=SEED)
    va_o, _ = train_test_split(temp_o, test_size=0.5, random_state=SEED)
    return tr_o, va_o


def main() -> None:
    print("[1/4] Loading training data + DTG slices...")
    df_train_combined, feat_cols = load_training_data()
    print(f"  combined training rows: {len(df_train_combined):,}, features: {len(feat_cols)}")
    dtg = pd.read_parquet(DTG_SLICES)
    print(f"  DTG slices: {len(dtg):,}")

    # Verify DTG has all required features
    missing = [c for c in feat_cols if c not in dtg.columns]
    if missing:
        print(f"  WARNING: DTG missing features: {missing}")
        for c in missing:
            dtg[c] = 0.0
    print(f"  feature alignment: OK")

    print("\n[2/4] Building train/val split (same as eval_high_density.py 'after')...")
    df_orig = pd.read_parquet(ORIG)
    sids_orig = df_orig["scenario_id"].unique()
    tr_o, temp_o = train_test_split(sids_orig, test_size=0.3, random_state=SEED)
    va_o, _ = train_test_split(temp_o, test_size=0.5, random_state=SEED)

    sids_bn = df_train_combined["scenario_id"].unique()
    sids_bn = sids_bn[sids_bn >= 100000]  # bottleneck only
    if len(sids_bn) > 0:
        tr_bn, temp_bn = train_test_split(sids_bn, test_size=0.3, random_state=SEED)
        va_bn, _ = train_test_split(temp_bn, test_size=0.5, random_state=SEED)
    else:
        tr_bn = va_bn = np.array([])
    tr_a = np.concatenate([tr_o, tr_bn])
    va_a = np.concatenate([va_o, va_bn])
    print(f"  train sids: {len(tr_a):,}, val sids: {len(va_a):,}")

    print("\n[3/4] Training XGBoost (after_single)...")
    model = train_single_model(df_train_combined, feat_cols, tr_a, va_a)
    print(f"  model trained.")

    print("\n[4/4] Predicting DTG slices...")
    X_dtg = dtg[feat_cols].values
    k_hat = model.predict(X_dtg)
    dtg["k_hat"] = k_hat
    print(f"  predicted {len(dtg):,} slices. k_hat range: [{k_hat.min():.2f}, {k_hat.max():.2f}]")

    out_pq = OUT_DIR / "dtg_vds_eval.parquet"
    dtg.to_parquet(out_pq, index=False)
    print(f"\n→ wrote {out_pq}")

    # Compute metrics
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

    summary = {}
    for gt_name in ("k_gt_qu", "k_gt_occ"):
        sub = dtg.dropna(subset=[gt_name, "k_hat"])
        sub = sub[sub[gt_name] >= 0]
        summary[gt_name] = {"overall": metrics(sub[gt_name], sub["k_hat"])}
        # per density bin
        bins = [0, 8, 16, 24, 40, 1000]
        labels = ["0-8", "8-16", "16-24", "24-40", "40+"]
        sub["bin"] = pd.cut(sub[gt_name], bins=bins, labels=labels)
        per_bin = {}
        for lab in labels:
            ss = sub[sub["bin"] == lab]
            if len(ss) >= 10:
                per_bin[lab] = metrics(ss[gt_name], ss["k_hat"])
        summary[gt_name]["per_bin"] = per_bin

    out_json = OUT_DIR / "dtg_vds_eval_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"→ wrote {out_json}")

    print("\n=== OVERALL METRICS ===")
    for gt_name, s in summary.items():
        ov = s["overall"]
        print(f"  vs {gt_name}: n={ov['n']:,}  MAE={ov['mae']:.2f}  RMSE={ov['rmse']:.2f}  "
              f"MAPE={ov['mape']:.1f}%  R²={ov['r2']:.3f}")

    print("\n=== Per-density-bin (vs k_gt_qu) ===")
    for lab, m in summary["k_gt_qu"]["per_bin"].items():
        print(f"  {lab:>6}  n={m['n']:>4}  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%  R²={m['r2']:.3f}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, gt in zip(axes, ("k_gt_qu", "k_gt_occ")):
        sub = dtg.dropna(subset=[gt, "k_hat"])
        sub = sub[sub[gt] >= 0]
        ax.scatter(sub[gt], sub["k_hat"], s=3, alpha=0.3, color="#2b2d42")
        lim = max(sub[gt].max(), sub["k_hat"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
        m = summary[gt]["overall"]
        ax.set_xlabel(f"VDS ground truth — {gt}  [veh/km/lane]")
        ax.set_ylabel("Model k_hat  [veh/km/lane]")
        ax.set_title(f"DTG slice predictions vs {gt}\nn={m['n']:,}, MAE={m['mae']:.2f}, MAPE={m['mape']:.1f}%, R²={m['r2']:.3f}")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out_png = OUT_DIR / "dtg_vds_scatter.png"
    plt.savefig(out_png, dpi=120)
    print(f"\n→ wrote {out_png}")


if __name__ == "__main__":
    main()
