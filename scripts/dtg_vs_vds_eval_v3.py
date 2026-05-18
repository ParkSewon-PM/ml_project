"""V3: V2 + clamp speed_limit/num_lanes to SUMO training range + normalize vy_energy by length.

Tries to maximally align inference inputs with SUMO training distribution
without retraining the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
        "mape": float(round(mape, 2)),
        "r2": float(round(r2_score(y_true, y_pred), 4)),
    }


def main() -> None:
    print("[1/4] training same model...")
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
    sids_bn_arr = df_train["scenario_id"].unique()
    sids_bn_arr = sids_bn_arr[sids_bn_arr >= 100000]
    tr_bn, temp_bn = train_test_split(sids_bn_arr, test_size=0.3, random_state=SEED)
    va_bn, _ = train_test_split(temp_bn, test_size=0.5, random_state=SEED)
    tr_a = np.concatenate([tr_o, tr_bn]); va_a = np.concatenate([va_o, va_bn])
    model = train_single_model(df_train, feat_cols, tr_a, va_a)

    # SUMO clamp targets
    SL_MAX = float(df_train["speed_limit"].max())   # 27.78 m/s
    SL_MIN = float(df_train["speed_limit"].min())
    LN_MAX = float(df_train["num_lanes"].max())     # 3
    LN_MIN = float(df_train["num_lanes"].min())     # 1
    print(f"  SUMO: speed_limit ∈ [{SL_MIN:.1f}, {SL_MAX:.1f}], num_lanes ∈ [{LN_MIN}, {LN_MAX}]")

    print("[2/4] loading v2 (smoothed) DTG features...")
    dtg = pd.read_parquet(OUT_DIR / "dtg_slices_features_smoothed.parquet").copy()
    n_orig = len(dtg)
    n_sl_oob = (dtg["speed_limit"] > SL_MAX).sum()
    n_ln_oob = (dtg["num_lanes"] > LN_MAX).sum()
    print(f"  pre-clip: {n_sl_oob} ({n_sl_oob/n_orig*100:.1f}%) speed_limit OOB, "
          f"{n_ln_oob} ({n_ln_oob/n_orig*100:.1f}%) num_lanes OOB")

    # clamp
    dtg["speed_limit"] = dtg["speed_limit"].clip(SL_MIN, SL_MAX)
    dtg["num_lanes"] = dtg["num_lanes"].clip(LN_MIN, LN_MAX)

    # normalize length-dependent vy_energy: divide by traversal_s (per-second energy)
    # then rescale to median SUMO trajectory length (~ 300s? actually variable)
    # SUMO mean vy_energy = 5.44 with ~300s trajectories ⇒ scaling factor
    sumo_traj_len_proxy = 100.0  # rough; actual varies
    # Cleaner option: use vy_variance (length-invariant) but model expects vy_energy
    # → rescale: vy_energy_norm = vy_energy * (sumo_proxy / dtg_traj_s)
    if "traversal_s" in dtg.columns:
        dtg["vy_energy"] = dtg["vy_energy"] * (sumo_traj_len_proxy / dtg["traversal_s"].clip(lower=10))
        print(f"  vy_energy rescaled: new mean {dtg['vy_energy'].mean():.2f} (SUMO {5.44})")

    for c in feat_cols:
        if c not in dtg.columns:
            dtg[c] = 0.0

    print("[3/4] inferring v3 (clamped + vy_energy normalized)...")
    dtg["k_hat"] = model.predict(dtg[feat_cols].values)
    print(f"  k_hat: range [{dtg['k_hat'].min():.2f}, {dtg['k_hat'].max():.2f}], "
          f"mean {dtg['k_hat'].mean():.2f}  (k_gt_qu mean {dtg['k_gt_qu'].mean():.2f})")

    dtg.to_parquet(OUT_DIR / "dtg_vds_eval_v3.parquet", index=False)

    print("\n[4/4] v1 vs v2 vs v3 comparison:")
    v1 = pd.read_parquet(OUT_DIR / "dtg_vds_eval.parquet")
    v2 = pd.read_parquet(OUT_DIR / "dtg_vds_eval_v2.parquet")
    print(f"\n{'':>8} | {'v1 (raw 3Hz)':>15} | {'v2 (smooth 1Hz)':>17} | {'v3 (+clamp +normE)':>20}")
    for gt in ("k_gt_qu", "k_gt_occ"):
        rows = []
        for nm, df in [("v1", v1), ("v2", v2), ("v3", dtg)]:
            s = df.dropna(subset=[gt]); s = s[s[gt] >= 0]
            rows.append(metrics(s[gt], s["k_hat"]))
        print(f"\n  vs {gt}:")
        print(f"  {'MAE':>8} | {rows[0]['mae']:>15.2f} | {rows[1]['mae']:>17.2f} | {rows[2]['mae']:>20.2f}")
        print(f"  {'MAPE':>8} | {rows[0]['mape']:>14.1f}% | {rows[1]['mape']:>16.1f}% | {rows[2]['mape']:>19.1f}%")
        print(f"  {'R²':>8} | {rows[0]['r2']:>15.3f} | {rows[1]['r2']:>17.3f} | {rows[2]['r2']:>20.3f}")

    # v3 per-bin
    print("\n[v3] per density bin (vs k_gt_qu):")
    sub = dtg.dropna(subset=["k_gt_qu"]); sub = sub[sub["k_gt_qu"] >= 0]
    bins = [0, 8, 16, 24, 40, 1000]; labels = ["0-8", "8-16", "16-24", "24-40", "40+"]
    sub["bin"] = pd.cut(sub["k_gt_qu"], bins=bins, labels=labels)
    for lab in labels:
        ss = sub[sub["bin"] == lab]
        if len(ss) >= 10:
            m = metrics(ss["k_gt_qu"], ss["k_hat"])
            print(f"  {lab:>6}  n={m['n']:>4}  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%  R²={m['r2']:.3f}")


if __name__ == "__main__":
    main()
