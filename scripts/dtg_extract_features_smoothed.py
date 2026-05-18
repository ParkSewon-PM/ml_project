"""V2: Smooth DTG GPS (3-sec sampling) into 1Hz signal before feature extraction.

Pipeline diff vs v1:
  - GPS valid points (every 3s) → cubic-spline interpolated onto 1Hz grid
  - 1Hz (x, y) trajectory → Savitzky-Golay filter (window=5, polyorder=2) → smoothed (x, y)
  - VX, VY = central-diff at 1Hz
  - AX, AY, jerk = subsequent 1Hz diffs

Goal: match SUMO training distribution (1Hz, low-noise) better than v1 (3Hz raw + noisy).

Output: results/dtg_slices_features_smoothed.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtg_extract_features import (  # noqa: E402
    DTG_DIR, HWY_PTS, VDS_INFO, VDS_STAT,
    TRAVERSAL_M, MIN_SLICE_SEC, MAX_VDS_DIST_M, LANE_DEFAULT, SPEED_LIMIT_DEFAULT,
    load_dtg, cumulative_distance_m, slice_one_vehicle, sample_entropy,
)

OUT = Path("/Users/park/ml-project/results/dtg_slices_features_smoothed.parquet")
SAVGOL_WIN = 15           # 15 seconds — strong smoothing of 1Hz interpolated trajectory
SAVGOL_ORDER = 3
GPS_SMOOTH_WIN = 5        # smooth the raw 3-sec GPS points before spline (window in points = 15s)
GPS_SMOOTH_ORDER = 2


def extract_features_smoothed(slc: pd.DataFrame) -> dict:
    """Same 30 trajectory features, but VX/VY/AX/AY/jerk are computed at 1Hz
    after cubic-spline interpolation and Savitzky-Golay smoothing."""
    speed = slc["speed_ms"].fillna(0).clip(lower=0).values
    brake = slc["brake"].fillna(0).astype(int).values
    n = len(speed)
    feat: dict = {}

    # === SPEED stats (9) === unchanged (1Hz raw)
    feat["speed_mean"] = float(speed.mean())
    feat["speed_std"] = float(speed.std(ddof=1)) if n > 1 else 0.0
    feat["speed_cv"] = feat["speed_std"] / feat["speed_mean"] if feat["speed_mean"] > 0 else 0.0
    feat["speed_iqr"] = float(np.percentile(speed, 75) - np.percentile(speed, 25))
    feat["speed_min"] = float(speed.min())
    feat["speed_max"] = float(speed.max())
    feat["speed_median"] = float(np.median(speed))
    feat["speed_p10"] = float(np.percentile(speed, 10))
    feat["speed_p90"] = float(np.percentile(speed, 90))

    # === GPS → 1Hz smoothed trajectory ===
    gps = slc[slc["gps_ok"]].copy()
    vx = vy = ax = ay = jerk = np.array([0.0])
    if len(gps) >= 4:
        ts0 = slc["ts"].iloc[0]
        times_gps = (gps["ts"] - ts0).dt.total_seconds().values
        lon = gps["lon"].values
        lat = gps["lat"].values
        lat0 = lat.mean()
        x_m = (lon - lon[0]) * 111000 * np.cos(np.radians(lat0))
        y_m = (lat - lat[0]) * 111000

        # ensure strictly increasing times for spline
        order = np.argsort(times_gps)
        times_gps = times_gps[order]; x_m = x_m[order]; y_m = y_m[order]
        # dedupe collisions
        ok = np.concatenate([[True], np.diff(times_gps) > 0.001])
        times_gps = times_gps[ok]; x_m = x_m[ok]; y_m = y_m[ok]

        if len(times_gps) >= 4:
            try:
                # (i) Smooth raw 3-sec GPS points first (removes GPS noise)
                if len(x_m) >= GPS_SMOOTH_WIN:
                    x_m = savgol_filter(x_m, window_length=GPS_SMOOTH_WIN, polyorder=GPS_SMOOTH_ORDER, mode="nearest")
                    y_m = savgol_filter(y_m, window_length=GPS_SMOOTH_WIN, polyorder=GPS_SMOOTH_ORDER, mode="nearest")
                # (ii) Cubic-spline interpolate to 1Hz grid
                spx = CubicSpline(times_gps, x_m, extrapolate=False)
                spy = CubicSpline(times_gps, y_m, extrapolate=False)
                t_min = times_gps[0]
                t_max = times_gps[-1]
                t_grid = np.arange(t_min, t_max + 0.5, 1.0)
                x_g = spx(t_grid); y_g = spy(t_grid)
                # (iii) Wide Savitzky-Golay smoothing on 1Hz trajectory
                if len(x_g) >= SAVGOL_WIN:
                    x_g = savgol_filter(x_g, window_length=SAVGOL_WIN, polyorder=SAVGOL_ORDER, mode="nearest")
                    y_g = savgol_filter(y_g, window_length=SAVGOL_WIN, polyorder=SAVGOL_ORDER, mode="nearest")
                # derivatives at 1Hz (dt=1s)
                vx = np.diff(x_g)
                vy = np.diff(y_g)
                if len(vx) >= 2:
                    ax = np.diff(vx)
                    ay = np.diff(vy)
                    if len(ax) >= 2:
                        jerk = np.diff(ax)
            except Exception:
                pass

    # AX (2), AY (2), jerk (2)
    feat["ax_mean"] = float(ax.mean())
    feat["ax_std"] = float(ax.std(ddof=1)) if len(ax) > 1 else 0.0
    feat["ay_mean"] = float(ay.mean())
    feat["ay_std"] = float(ay.std(ddof=1)) if len(ay) > 1 else 0.0
    feat["jerk_mean"] = float(jerk.mean())
    feat["jerk_std"] = float(jerk.std(ddof=1)) if len(jerk) > 1 else 0.0

    # VY stats (6)
    feat["vy_mean"] = float(vy.mean())
    feat["vy_std"] = float(vy.std(ddof=1)) if len(vy) > 1 else 0.0
    feat["vy_min"] = float(vy.min())
    feat["vy_max"] = float(vy.max())
    feat["vy_variance"] = float(vy.var(ddof=1)) if len(vy) > 1 else 0.0
    feat["vy_energy"] = float(np.sum(vy ** 2))

    # === BRAKE patterns (3) === unchanged
    if n > 0:
        runs = np.diff(np.concatenate([[0], brake, [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        feat["brake_count"] = int(len(starts))
        feat["brake_time_ratio"] = float(brake.mean())
        feat["mean_brake_duration"] = float(np.mean(ends - starts)) if len(starts) > 0 else 0.0
    else:
        feat["brake_count"] = 0
        feat["brake_time_ratio"] = 0.0
        feat["mean_brake_duration"] = 0.0

    # === STOP patterns (3) === unchanged (speed-based)
    stop_mask = (speed < 0.5).astype(int)
    runs = np.diff(np.concatenate([[0], stop_mask, [0]]))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    feat["stop_count"] = int(len(starts))
    feat["stop_time_ratio"] = float(stop_mask.mean())
    feat["mean_stop_duration"] = float(np.mean(ends - starts)) if len(starts) > 0 else 0.0

    # === TIME-SERIES (3) === unchanged (speed-based)
    if n >= 3:
        try:
            cc = np.corrcoef(speed[:-1], speed[1:])[0, 1]
            feat["speed_autocorr_lag1"] = float(cc) if np.isfinite(cc) else 0.0
        except Exception:
            feat["speed_autocorr_lag1"] = 0.0
        try:
            sp = speed - speed.mean()
            mag = np.abs(np.fft.rfft(sp))
            if len(mag) > 1:
                k = int(np.argmax(mag[1:]) + 1)
                feat["speed_fft_dominant_freq"] = float(k / n)
            else:
                feat["speed_fft_dominant_freq"] = 0.0
        except Exception:
            feat["speed_fft_dominant_freq"] = 0.0
        feat["sample_entropy"] = sample_entropy(speed, m=2, r=0.2 * (speed.std() if speed.std() > 0 else 1.0))
    else:
        feat["speed_autocorr_lag1"] = 0.0
        feat["speed_fft_dominant_freq"] = 0.0
        feat["sample_entropy"] = 0.0

    return feat


def main() -> None:
    print("[1/3] loading inputs...")
    hwy = pd.read_parquet(HWY_PTS)
    vds_info = pd.read_csv(VDS_INFO).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    vds_xy = np.c_[vds_info["lat"] * 111000, vds_info["lon"] * 88000]
    tree = cKDTree(vds_xy)

    stats = pd.read_csv(VDS_STAT, encoding="utf-8-sig",
                        usecols=["AGGR_HM", "VDS_ID", "VMTC", "SHARE", "AVRG_VE"])
    stats = stats[(stats["VMTC"] >= 0) & (stats["AVRG_VE"] > 0)]
    stats = stats.drop_duplicates(subset=["VDS_ID", "AGGR_HM"], keep="first")
    stat_map = stats.set_index(["VDS_ID", "AGGR_HM"]).to_dict(orient="index")

    print("[2/3] re-extracting features with smoothed GPS...")
    rows: list[dict] = []
    slice_id = 0
    for p in sorted(DTG_DIR.glob("*.csv")):
        plate = p.stem.split("_")[1]
        df = load_dtg(p)
        slices = slice_one_vehicle(df)
        n_kept = 0
        for s in slices:
            slc = s["df"]
            valid_gps = slc[slc["gps_ok"]]
            if len(valid_gps) < 4:
                continue
            mid = valid_gps.iloc[len(valid_gps) // 2]
            xy = np.array([[mid["lat"] * 111000, mid["lon"] * 88000]])
            dist, idx = tree.query(xy, k=1, distance_upper_bound=MAX_VDS_DIST_M)
            if not np.isfinite(dist[0]):
                continue
            vds_row = vds_info.iloc[idx[0]]
            vds_id = vds_row["vdsId"]
            mid_ts = mid["ts"]
            aggr_hm = mid_ts.hour * 100 + (mid_ts.minute // 5) * 5
            stat = stat_map.get((vds_id, aggr_hm))
            if stat is None:
                continue

            nearest_hwy = hwy[(hwy["plate"] == plate) &
                              (hwy["ts"] >= slc["ts"].iloc[0]) &
                              (hwy["ts"] <= slc["ts"].iloc[-1])]
            if len(nearest_hwy) == 0:
                num_lanes = float(LANE_DEFAULT)
                speed_limit_ms = SPEED_LIMIT_DEFAULT / 3.6
            else:
                num_lanes = float(nearest_hwy["LANES"].mode().iloc[0]) if not nearest_hwy["LANES"].isna().all() else LANE_DEFAULT
                spd_kmh = nearest_hwy["MAX_SPD"].mode().iloc[0] if not nearest_hwy["MAX_SPD"].isna().all() else SPEED_LIMIT_DEFAULT
                speed_limit_ms = float(spd_kmh) / 3.6
            num_lanes = max(1.0, num_lanes)

            feat = extract_features_smoothed(slc)
            feat["num_lanes"] = num_lanes
            feat["speed_limit"] = speed_limit_ms

            vmtc = stat["VMTC"]; avrg_ve = stat["AVRG_VE"]; share = stat["SHARE"]
            lanes_for_k = max(1.0, num_lanes)
            k_gt_qu = (vmtc * 12.0 / lanes_for_k) / avrg_ve if avrg_ve > 0 else np.nan
            k_gt_occ = (share / 100.0) / (5.5 / 1000.0)

            rec = {
                "slice_id": slice_id,
                "plate": plate,
                "ts_start": slc["ts"].iloc[0],
                "ts_end": slc["ts"].iloc[-1],
                "lon_mid": float(mid["lon"]),
                "lat_mid": float(mid["lat"]),
                "traversal_m": float(s["traversal_m"]),
                "traversal_s": float(s["traversal_s"]),
                "dtg_mean_speed_kmh": float(slc["속도"].mean()),
                "nearest_vdsId": vds_id,
                "dist_to_vds_m": float(dist[0]),
                "vds_aggr_hm": int(aggr_hm),
                "vds_VMTC": float(vmtc),
                "vds_SHARE": float(share),
                "vds_AVRG_VE": float(avrg_ve),
                "k_gt_qu": float(k_gt_qu),
                "k_gt_occ": float(k_gt_occ),
                **feat,
            }
            rows.append(rec)
            slice_id += 1
            n_kept += 1
        print(f"  [{plate}] {len(slices)} slices → {n_kept} matched")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n[3/3] wrote {OUT} ({len(out):,} slices)")

    # Compare key feature distributions vs v1
    v1 = pd.read_parquet("/Users/park/ml-project/results/dtg_slices_features.parquet")
    keys = ["ax_std", "vy_std", "vy_energy", "jerk_std", "ay_std", "vy_variance"]
    print(f"\n=== Distribution change (v1 raw 3Hz → v2 smoothed 1Hz) ===")
    print(f"{'feature':20s} | {'v1 mean':>10} | {'v2 mean':>10} | {'reduction':>10}")
    for k in keys:
        if k in v1.columns and k in out.columns:
            v1m = v1[k].mean(); v2m = out[k].mean()
            red = (1 - v2m / v1m) * 100 if abs(v1m) > 1e-6 else 0
            print(f"{k:20s} | {v1m:>10.3f} | {v2m:>10.3f} | {red:>9.1f}%")


if __name__ == "__main__":
    main()
