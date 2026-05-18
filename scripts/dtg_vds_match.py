"""Match DTG highway points to nearest VDS (≤300m), join with 09-11 5min stats,
derive ground-truth density.

Outputs:
  results/dtg_vds_matched.parquet  — DTG point × VDS × VDS 5min stats
  results/dtg_vds_segments.csv     — (vds_id, 5min_bin) summary with k_gt + DTG points

Density derivation (per-lane, veh/km/lane):
  k_gt_q_over_u = (VMTC × 12 / lanes) / AVRG_VE       [veh/hr/lane ÷ km/h]
  k_gt_share    = SHARE × 1000 / (avg_vehicle_length_m × 100)   # share is %
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

DTG_PTS = Path("/Users/park/ml-project/results/dtg_highway_points.parquet")
VDS_INFO = Path("/Users/park/ml-project/data/vds_info.csv")
VDS_STAT = Path("/Users/park/ml-project/data/TB_COL_EX_T_SMHS_VDS_BROFIFO_5MIN_20200911/TB_COL_EX_T_SMHS_VDS_BROFIFO_5MIN_20200911.csv")
OUT_DIR = Path("/Users/park/ml-project/results")

MAX_DIST_M = 300            # DTG point ≤ 300m from VDS counted as on that VDS
AVG_VEH_LEN_M = 5.5         # for SHARE→k conversion
DEFAULT_LANES = 2           # fallback when VDS lanes unknown


def main() -> None:
    print("loading inputs...")
    dtg = pd.read_parquet(DTG_PTS)
    vds_info = pd.read_csv(VDS_INFO)
    n_total = len(vds_info)
    vds_info = vds_info.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  DTG highway points:  {len(dtg):,}")
    print(f"  VDS info:            {len(vds_info):,} (dropped {n_total - len(vds_info)} with missing coords)")

    # KDTree on VDS (lat/lon → meters approximation, equirectangular)
    # for small distances (~300m) at Korea's latitude, simple scale works
    lat_m = 111000
    lon_m = 88000  # at ~36°N
    vds_xy = np.c_[vds_info["lat"] * lat_m, vds_info["lon"] * lon_m]
    dtg_xy = np.c_[dtg["lat"] * lat_m, dtg["lon"] * lon_m]
    tree = cKDTree(vds_xy)
    dist, idx = tree.query(dtg_xy, k=1, distance_upper_bound=MAX_DIST_M)

    matched_mask = dist < MAX_DIST_M
    dtg["nearest_vds_dist_m"] = dist
    dtg["nearest_vds_idx"] = idx
    print(f"  DTG points with VDS within {MAX_DIST_M}m: "
          f"{matched_mask.sum():,} ({matched_mask.mean()*100:.1f}%)")

    dtg_m = dtg.loc[matched_mask].copy()
    dtg_m["vdsId"] = vds_info.iloc[idx[matched_mask]]["vdsId"].values
    dtg_m["vds_lat"] = vds_info.iloc[idx[matched_mask]]["lat"].values
    dtg_m["vds_lon"] = vds_info.iloc[idx[matched_mask]]["lon"].values
    dtg_m["vds_routeName"] = vds_info.iloc[idx[matched_mask]]["routeName"].values
    dtg_m["vds_directionCode"] = vds_info.iloc[idx[matched_mask]]["directionCode"].values

    # 5min bin (HHMM int)
    dtg_m["aggr_hm"] = dtg_m["ts"].dt.hour * 100 + (dtg_m["ts"].dt.minute // 5) * 5

    # Load VDS stats (09-11)
    print("loading VDS 09-11 stats...")
    stats = pd.read_csv(VDS_STAT, encoding="utf-8-sig",
                        usecols=["AGGR_HM", "VDS_ID", "VMTC", "SHARE", "AVRG_VE"])
    stats = stats[(stats["VMTC"] >= 0) & (stats["AVRG_VE"] > 0)]  # drop -1 missing
    print(f"  VDS stats (non-missing): {len(stats):,}")

    # Join
    merged = dtg_m.merge(
        stats,
        left_on=["vdsId", "aggr_hm"],
        right_on=["VDS_ID", "AGGR_HM"],
        how="left",
    )
    has_gt = merged["VMTC"].notna()
    print(f"\n  DTG points joined with VDS 5min stats: {has_gt.sum():,} "
          f"({has_gt.mean()*100:.1f}% of matched)")

    # Density derivations
    # Lanes from DTG highway match (MOCT LANES is per-direction usually)
    merged["lanes"] = merged["LANES"].fillna(DEFAULT_LANES).astype(float).clip(lower=1)
    # k = (q_per_hr / lanes) / u
    #   VMTC = veh in 5min  → veh/hr = VMTC × 12
    merged["k_gt_qu"] = (merged["VMTC"] * 12 / merged["lanes"]) / merged["AVRG_VE"]
    # SHARE is in percent (0~100). occupancy = SHARE/100 (dimensionless)
    # k = occupancy / avg_veh_length  (veh per meter) × 1000 → veh/km/lane
    merged["k_gt_occ"] = (merged["SHARE"] / 100.0) / (AVG_VEH_LEN_M / 1000.0)

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pts = OUT_DIR / "dtg_vds_matched.parquet"
    merged.to_parquet(out_pts, index=False)
    print(f"\n→ wrote {out_pts}")

    # Per (vds, 5min) summary
    seg = (
        merged.dropna(subset=["VMTC"])
        .groupby(["vdsId", "aggr_hm"])
        .agg(
            n_dtg_points=("ts", "size"),
            n_dtg_vehicles=("plate", "nunique"),
            dtg_mean_speed=("speed_kmh", "mean"),
            vds_VMTC=("VMTC", "first"),
            vds_SHARE=("SHARE", "first"),
            vds_AVRG_VE=("AVRG_VE", "first"),
            lanes=("lanes", "first"),
            road=("vds_routeName", "first"),
            k_gt_qu=("k_gt_qu", "first"),
            k_gt_occ=("k_gt_occ", "first"),
        )
        .reset_index()
        .sort_values("n_dtg_points", ascending=False)
    )
    out_seg = OUT_DIR / "dtg_vds_segments.csv"
    seg.to_csv(out_seg, index=False)
    print(f"→ wrote {out_seg} ({len(seg):,} unique (vds × 5min) segments)")

    # Summary
    print(f"\n{'='*70}\nMATCH SUMMARY\n{'='*70}")
    print(f"DTG points matched to VDS (≤{MAX_DIST_M}m + non-missing GT): {has_gt.sum():,}")
    print(f"Unique (VDS × 5min) segments with DTG coverage: {len(seg):,}")
    print(f"  └ segments with ≥2 DTG vehicles: {(seg['n_dtg_vehicles']>=2).sum():,}")
    print(f"  └ segments with ≥3 DTG vehicles: {(seg['n_dtg_vehicles']>=3).sum():,}")
    print(f"\nGround-truth density distribution:")
    print(f"  k_gt_qu  [veh/km/lane]: mean {seg['k_gt_qu'].mean():.1f}, "
          f"median {seg['k_gt_qu'].median():.1f}, "
          f"p95 {seg['k_gt_qu'].quantile(0.95):.1f}, max {seg['k_gt_qu'].max():.1f}")
    print(f"  k_gt_occ [veh/km/lane]: mean {seg['k_gt_occ'].mean():.1f}, "
          f"median {seg['k_gt_occ'].median():.1f}, "
          f"p95 {seg['k_gt_occ'].quantile(0.95):.1f}, max {seg['k_gt_occ'].max():.1f}")
    print(f"\nk_gt by density bin (count per bin from q/u):")
    bins = [0, 8, 16, 24, 40, 1000]
    labels = ["0-8", "8-16", "16-24", "24-40", "40+"]
    seg["bin"] = pd.cut(seg["k_gt_qu"], bins=bins, labels=labels)
    print(seg["bin"].value_counts().sort_index().to_string())

    print(f"\nTop 15 segments by DTG coverage (most points):")
    cols = ["aggr_hm", "road", "n_dtg_vehicles", "n_dtg_points",
            "dtg_mean_speed", "vds_AVRG_VE", "k_gt_qu", "k_gt_occ"]
    print(seg[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
