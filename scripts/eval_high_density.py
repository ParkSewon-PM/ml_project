"""High-density before/after evaluation.

Compares: original-only vs original+bottleneck training.
Evaluates: single probe, aligned ensemble (N=1~5), deploy ensemble (CF/Bayesian).
All using 30 features consistently.

Usage:
    python scripts/eval_high_density.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
RESULTS_DIR = Path("results/multi_probe")
TARGET = "density_per_lane"

EXCLUDE_COLS = {
    "scenario_id", "probe_idx", "density", "flow", "density_per_lane", "flow_per_lane",
    "demand_vehph", "traversal_time", "k_fd", "q_fd",
    "delta_density", "delta_flow", "gap_mean", "slow_duration_ratio",
}
# num_lanes, speed_limit are INCLUDED as features (32 total)


def get_feat_cols(df: pd.DataFrame) -> list[str]:
    """32 features for single/deploy: 30 trajectory + num_lanes + speed_limit."""
    return sorted([c for c in df.columns if c not in EXCLUDE_COLS])


def get_probe_feat_cols(df: pd.DataFrame) -> list[str]:
    """30 trajectory features only (for aligned aggregation — num_lanes/speed_limit added separately)."""
    return sorted([c for c in df.columns if c not in EXCLUDE_COLS and c not in {"num_lanes", "speed_limit"}])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true > 0.5
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else 0
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mape": round(mape, 2),
        "n": int(len(y_true)),
    }


def cf_additive(row: pd.Series) -> float:
    return row["ax_std"] + row["brake_time_ratio"] + row["speed_cv"]


def ensemble_bayesian_cf(preds, cf_scores, prior_mu=9.4, prior_sigma=4.0):
    mu = prior_mu
    var = prior_sigma ** 2
    for pred, cf in zip(preds, cf_scores):
        obs_sigma = 3.0 * np.exp(-cf)
        obs_var = obs_sigma ** 2
        new_var = 1.0 / (1.0 / var + 1.0 / obs_var)
        mu = new_var * (mu / var + pred / obs_var)
        var = new_var
    return float(mu)


def train_single_model(df, feat_cols, tr_sids, va_sids):
    tr = df[df["scenario_id"].isin(set(tr_sids))]
    va = df[df["scenario_id"].isin(set(va_sids))]
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        early_stopping_rounds=20, random_state=SEED, tree_method="hist",
    )
    m.fit(tr[feat_cols].values, tr[TARGET].values,
          eval_set=[(va[feat_cols].values, va[TARGET].values)], verbose=False)
    return m


def eval_single_probe(model, df, feat_cols, te_sids):
    te = df[df["scenario_id"].isin(set(te_sids))]
    pred = model.predict(te[feat_cols].values)
    results = {"overall": compute_metrics(te[TARGET].values, pred)}
    for lane in [1, 2, 3]:
        sub = te[te["num_lanes"] == lane]
        if len(sub) > 10:
            p = model.predict(sub[feat_cols].values)
            results[f"{lane}L"] = compute_metrics(sub[TARGET].values, p)
    for lo, hi, label in [(0, 8, "0_8"), (8, 16, "8_16"), (16, 24, "16_24"), (24, 999, "24+")]:
        mask = (te[TARGET] >= lo) & (te[TARGET] < hi)
        if mask.sum() > 10:
            results[label] = compute_metrics(te.loc[mask, TARGET].values, pred[mask.values])
    return results


def build_aligned_agg(df, feat_cols, n_probes, sids, seed):
    rng = np.random.RandomState(seed)
    sub = df[df["scenario_id"].isin(set(sids))]
    rows = []
    for _, g in sub.groupby("scenario_id"):
        if len(g) < n_probes:
            continue
        s = g.iloc[rng.choice(len(g), size=n_probes, replace=False)]
        row = {}
        for c in feat_cols:
            row[f"{c}_mean"] = s[c].mean()
            row[f"{c}_std"] = s[c].std() if n_probes > 1 else 0.0
        row["num_lanes"] = g["num_lanes"].iloc[0]
        row["speed_limit"] = g["speed_limit"].iloc[0]
        row["target"] = g[TARGET].iloc[0]
        row["scenario_id"] = g["scenario_id"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def eval_aligned(df, probe_feat_cols, tr5, va5, te5):
    results = {}
    for n in [1, 2, 3, 5]:
        tr_agg = build_aligned_agg(df, probe_feat_cols, n, tr5, SEED)
        va_agg = build_aligned_agg(df, probe_feat_cols, n, va5, SEED + 1)
        te_agg = build_aligned_agg(df, probe_feat_cols, n, te5, SEED + 2)
        if len(tr_agg) == 0 or len(te_agg) == 0:
            continue

        acols = [f"{c}_mean" for c in probe_feat_cols] + [f"{c}_std" for c in probe_feat_cols] + ["num_lanes", "speed_limit"]
        fm = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20,
            random_state=SEED, tree_method="hist",
        )
        fm.fit(tr_agg[acols].values, tr_agg["target"].values,
               eval_set=[(va_agg[acols].values, va_agg["target"].values)], verbose=False)
        fp = fm.predict(te_agg[acols].values)

        results[f"N{n}_overall"] = compute_metrics(te_agg["target"].values, fp)
        for lane in [1, 2, 3]:
            mask = te_agg["num_lanes"] == lane
            if mask.sum() > 10:
                results[f"N{n}_{lane}L"] = compute_metrics(te_agg.loc[mask, "target"].values, fp[mask.values])
    return results


def eval_deploy(model, df, feat_cols, te_sids):
    rng = np.random.RandomState(SEED + 2)
    te_sub = df[df["scenario_id"].isin(set(te_sids))]
    results = {}

    for n in [1, 2, 3, 5]:
        yt, yp_cf, yp_mean, yp_bayes = [], [], [], []
        meta = []
        for _, g in te_sub.groupby("scenario_id"):
            if len(g) < n:
                continue
            s = g.iloc[rng.choice(len(g), size=n, replace=False)]
            preds = model.predict(s[feat_cols].values)
            scores = np.array([cf_additive(s.iloc[i]) for i in range(len(s))])
            exps = np.exp(scores)
            w = exps / exps.sum()

            yt.append(g[TARGET].iloc[0])
            yp_cf.append(float((w * preds).sum()))
            yp_mean.append(float(preds.mean()))
            yp_bayes.append(ensemble_bayesian_cf(preds, scores))
            meta.append({"lanes": int(g["num_lanes"].iloc[0]), "density": g[TARGET].iloc[0]})

        yt = np.array(yt)
        meta_df = pd.DataFrame(meta)

        for method, yp in [("cf_softmax", yp_cf), ("simple_mean", yp_mean), ("bayesian_cf", yp_bayes)]:
            yp = np.array(yp)
            results[f"{method}_N{n}_overall"] = compute_metrics(yt, yp)
            for lane in [1, 2, 3]:
                mask = meta_df["lanes"] == lane
                if mask.sum() > 10:
                    results[f"{method}_N{n}_{lane}L"] = compute_metrics(yt[mask.values], yp[mask.values])
    return results


def main():
    df_orig = pd.read_parquet("data/features/dataset_1km.parquet")
    # Combine 10K (2-3lane) + 5K (1lane) bottleneck
    df_bn_list = []
    p10k = Path("data/features/dataset_bottleneck_10k.parquet")
    if p10k.exists():
        d = pd.read_parquet(p10k)
        d["scenario_id"] = d["scenario_id"] + 100000
        df_bn_list.append(d)
        logger.info("Loaded 10K bottleneck: %d rows", len(d))
    p1lane = Path("data/features/dataset_bottleneck_1lane_5k.parquet")
    if p1lane.exists():
        d = pd.read_parquet(p1lane)
        d["scenario_id"] = d["scenario_id"] + 200000
        df_bn_list.append(d)
        logger.info("Loaded 1lane 5K bottleneck: %d rows", len(d))
    df_bn = pd.concat(df_bn_list, ignore_index=True) if df_bn_list else pd.DataFrame()
    logger.info("Total bottleneck data: %d rows, %d scenarios", len(df_bn), df_bn["scenario_id"].nunique())

    feat_cols = get_feat_cols(df_orig)  # 32: 30 trajectory + num_lanes + speed_limit
    probe_feat_cols = get_probe_feat_cols(df_orig)  # 30: trajectory only (for aligned agg)
    logger.info("Single/deploy features: %d, aligned probe features: %d", len(feat_cols), len(probe_feat_cols))

    # Ensure bottleneck has same columns
    for c in feat_cols:
        if c not in df_bn.columns:
            df_bn[c] = 0.0

    cols = list(dict.fromkeys(feat_cols + [TARGET, "scenario_id", "num_lanes", "speed_limit"]))
    df_combined = pd.concat([df_orig[cols], df_bn[cols]], ignore_index=True)

    # === Split: derive after-splits from before-splits (prevents leakage) ===
    # 1) Split original sids once
    sids_orig = df_orig["scenario_id"].unique()
    tr_o, temp_o = train_test_split(sids_orig, test_size=0.3, random_state=SEED)
    va_o, te_o = train_test_split(temp_o, test_size=0.5, random_state=SEED)

    # 2) Split bottleneck sids independently
    sids_bn = df_bn["scenario_id"].unique() if len(df_bn) > 0 else np.array([])
    if len(sids_bn) > 0:
        tr_bn, temp_bn = train_test_split(sids_bn, test_size=0.3, random_state=SEED)
        va_bn, te_bn = train_test_split(temp_bn, test_size=0.5, random_state=SEED)
    else:
        tr_bn = va_bn = te_bn = np.array([])

    # 3) Compose after-splits so original sids keep their before assignments
    tr_a = np.concatenate([tr_o, tr_bn])
    va_a = np.concatenate([va_o, va_bn])
    te_a = np.concatenate([te_o, te_bn])

    # Invariant: te_o (original test) never intersects tr_a or va_a
    assert set(te_o).isdisjoint(set(tr_a) | set(va_a)), "leakage: te_o overlaps tr_a/va_a"

    # 4) Aligned 5-probe splits: filter each before/after assignment to scenarios with >=5 probes
    sids5_orig = set(df_orig.groupby("scenario_id").filter(lambda x: len(x) >= 5)["scenario_id"].unique())
    sids5_all = set(df_combined.groupby("scenario_id").filter(lambda x: len(x) >= 5)["scenario_id"].unique())

    tr5_b = np.array([s for s in tr_o if s in sids5_orig])
    va5_b = np.array([s for s in va_o if s in sids5_orig])
    te5_b = np.array([s for s in te_o if s in sids5_orig])

    tr5_a = np.array([s for s in tr_a if s in sids5_all])
    va5_a = np.array([s for s in va_a if s in sids5_all])
    te5_a = np.array([s for s in te_a if s in sids5_all])

    assert set(te5_b).isdisjoint(set(tr5_a) | set(va5_a)), "leakage: te5_b overlaps tr5_a/va5_a"

    all_results = {}

    # === BEFORE (original only) ===
    logger.info("=== BEFORE (original only) ===")
    m_before = train_single_model(df_orig, feat_cols, tr_o, va_o)

    all_results["before_single"] = eval_single_probe(m_before, df_orig, feat_cols, te_o)
    logger.info("Before single: %s", all_results["before_single"]["overall"])

    all_results["before_aligned"] = eval_aligned(df_orig, probe_feat_cols, tr5_b, va5_b, te5_b)
    all_results["before_deploy"] = eval_deploy(m_before, df_orig, feat_cols, te5_b)

    # === AFTER (combined) ===
    logger.info("=== AFTER (combined) ===")
    m_after = train_single_model(df_combined, feat_cols, tr_a, va_a)

    all_results["after_single"] = eval_single_probe(m_after, df_combined, feat_cols, te_a)
    logger.info("After single: %s", all_results["after_single"]["overall"])

    # After model on ORIGINAL test set only (fair comparison)
    te_orig_only = df_orig[df_orig["scenario_id"].isin(set(te_o))]
    pred_orig = m_after.predict(te_orig_only[feat_cols].values)
    all_results["after_on_original_testset"] = compute_metrics(te_orig_only[TARGET].values, pred_orig)
    logger.info("After → original test: %s", all_results["after_on_original_testset"])

    all_results["after_aligned"] = eval_aligned(df_combined, probe_feat_cols, tr5_a, va5_a, te5_a)
    all_results["after_deploy"] = eval_deploy(m_after, df_combined, feat_cols, te5_a)

    # Fair-comparison aligned/deploy: after model evaluated on original-only 5-probe test set
    all_results["after_aligned_on_original"] = eval_aligned(df_combined, probe_feat_cols, tr5_a, va5_a, te5_b)
    all_results["after_deploy_on_original"] = eval_deploy(m_after, df_combined, feat_cols, te5_b)

    # === Save ===
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "high_density_full_eval_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SINGLE PROBE (overall)")
    print(f"  Before: {all_results['before_single']['overall']}")
    print(f"  After:  {all_results['after_single']['overall']}")
    print(f"  After→orig test: {all_results['after_on_original_testset']}")

    print("\nALIGNED N=5 (overall)")
    print(f"  Before: {all_results['before_aligned'].get('N5_overall')}")
    print(f"  After:  {all_results['after_aligned'].get('N5_overall')}")

    print("\nDEPLOY CF-SOFTMAX N=5")
    print(f"  Before: {all_results['before_deploy'].get('cf_softmax_N5_overall')}")
    print(f"  After:  {all_results['after_deploy'].get('cf_softmax_N5_overall')}")

    print("\nDEPLOY BAYESIAN N=5")
    print(f"  Before: {all_results['before_deploy'].get('bayesian_cf_N5_overall')}")
    print(f"  After:  {all_results['after_deploy'].get('bayesian_cf_N5_overall')}")


if __name__ == "__main__":
    main()
