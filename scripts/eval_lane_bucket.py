"""Compute lane × density-bucket crossed metrics for the after_single model.

Reuses split logic from eval_high_density.py to produce the (lane, bucket) table
needed for Slide 20 (b) "차로별 × 밀도구간별 분해".

Output: results/multi_probe/lane_bucket_after_single.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_high_density import (  # noqa: E402
    SEED,
    TARGET,
    compute_metrics,
    get_feat_cols,
    train_single_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BUCKETS = [(0, 8, "0_8"), (8, 16, "8_16"), (16, 24, "16_24"), (24, 999, "24+")]


def main() -> None:
    df_orig = pd.read_parquet("data/features/dataset_1km.parquet")
    df_bn_list = []
    p10k = Path("data/features/dataset_bottleneck_10k.parquet")
    if p10k.exists():
        d = pd.read_parquet(p10k)
        d["scenario_id"] = d["scenario_id"] + 100000
        df_bn_list.append(d)
    p1lane = Path("data/features/dataset_bottleneck_1lane_5k.parquet")
    if p1lane.exists():
        d = pd.read_parquet(p1lane)
        d["scenario_id"] = d["scenario_id"] + 200000
        df_bn_list.append(d)
    df_bn = pd.concat(df_bn_list, ignore_index=True) if df_bn_list else pd.DataFrame()

    feat_cols = get_feat_cols(df_orig)
    for c in feat_cols:
        if c not in df_bn.columns:
            df_bn[c] = 0.0

    cols = list(dict.fromkeys(feat_cols + [TARGET, "scenario_id", "num_lanes", "speed_limit"]))
    df_combined = pd.concat([df_orig[cols], df_bn[cols]], ignore_index=True)

    sids_orig = df_orig["scenario_id"].unique()
    tr_o, temp_o = train_test_split(sids_orig, test_size=0.3, random_state=SEED)
    va_o, te_o = train_test_split(temp_o, test_size=0.5, random_state=SEED)

    sids_bn = df_bn["scenario_id"].unique() if len(df_bn) > 0 else np.array([])
    if len(sids_bn) > 0:
        tr_bn, temp_bn = train_test_split(sids_bn, test_size=0.3, random_state=SEED)
        va_bn, te_bn = train_test_split(temp_bn, test_size=0.5, random_state=SEED)
    else:
        tr_bn = va_bn = te_bn = np.array([])

    tr_a = np.concatenate([tr_o, tr_bn])
    va_a = np.concatenate([va_o, va_bn])
    te_a = np.concatenate([te_o, te_bn])

    logger.info("Training after_single model (combined dataset)...")
    m_after = train_single_model(df_combined, feat_cols, tr_a, va_a)

    te = df_combined[df_combined["scenario_id"].isin(set(te_a))].copy()
    te["pred"] = m_after.predict(te[feat_cols].values)

    results: dict[str, dict] = {}

    # Lane × bucket cross
    for lane in [1, 2, 3]:
        for lo, hi, label in BUCKETS:
            mask = (te["num_lanes"] == lane) & (te[TARGET] >= lo) & (te[TARGET] < hi)
            if mask.sum() > 10:
                key = f"{lane}L_{label}"
                results[key] = compute_metrics(te.loc[mask, TARGET].values, te.loc[mask, "pred"].values)
                results[key]["n"] = int(mask.sum())

    # Per-lane (overall, for sanity check vs JSON existing values)
    for lane in [1, 2, 3]:
        mask = te["num_lanes"] == lane
        results[f"{lane}L_overall"] = compute_metrics(te.loc[mask, TARGET].values, te.loc[mask, "pred"].values)
        results[f"{lane}L_overall"]["n"] = int(mask.sum())

    # Per-bucket (overall, for sanity check)
    for lo, hi, label in BUCKETS:
        mask = (te[TARGET] >= lo) & (te[TARGET] < hi)
        results[f"{label}_overall"] = compute_metrics(te.loc[mask, TARGET].values, te.loc[mask, "pred"].values)
        results[f"{label}_overall"]["n"] = int(mask.sum())

    out = Path("results/multi_probe/lane_bucket_after_single.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)

    # Pretty print
    print("\n=== Lane × Bucket (after_single) — MAE [veh/km/lane] (MAPE %) [n] ===")
    print(f"{'Bucket':>10} | {'1L':>20} | {'2L':>20} | {'3L':>20}")
    print("-" * 80)
    for lo, hi, label in BUCKETS:
        row = f"{label:>10} |"
        for lane in [1, 2, 3]:
            k = f"{lane}L_{label}"
            if k in results:
                m = results[k]
                row += f" {m['mae']:>5.2f} ({m['mape']:>5.1f}) [n={m['n']:>5}] |"
            else:
                row += f" {'--':>20} |"
        print(row)


if __name__ == "__main__":
    main()
