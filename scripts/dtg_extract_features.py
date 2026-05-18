"""Slice DTG trajectories into 1km traversals and extract 32 features.

Pipeline:
  1. Load DTG 30-vehicle cohort (1Hz time, 3s GPS).
  2. Per vehicle, sort by ts. Compute haversine cumulative distance.
  3. Cut slices at every 1km mark. Each slice = one prediction sample.
  4. For each slice, compute:
     - speed series (km/h → m/s) at 1Hz from `속도` column.
     - GPS-derived VX, VY by haversine on consecutive (lon, lat) with valid GPS.
     - AX = dVX/dt, AY = dVY/dt (numerical derivative).
     - jerk = dAX/dt.
     - brake = `브레이크` column (True/False → 0/1) at 1Hz.
     - 30 trajectory features (matching training schema).
  5. Snap slice midpoint to nearest VDS (already done in dtg_vds_matched.parquet).
     Attach (vdsId, aggr_hm, k_gt_qu, k_gt_occ, lanes, max_spd).

Output:
  results/dtg_slices_features.parquet
    columns: slice_id, plate, ts_start, ts_end, lon_start, lat_start,
             traversal_m, traversal_s, num_lanes, speed_limit,
             <30 trajectory features>,
             nearest_vdsId, dist_to_vds_m, vds_aggr_hm,
             k_gt_qu, k_gt_occ, dtg_mean_speed_kmh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DTG_DIR = Path("/Users/park/ml-project/data/DTG")
HWY_PTS = Path("/Users/park/ml-project/results/dtg_highway_points.parquet")
VDS_INFO = Path("/Users/park/ml-project/data/vds_info.csv")
VDS_STAT = Path("/Users/park/ml-project/data/TB_COL_EX_T_SMHS_VDS_BROFIFO_5MIN_20200911/TB_COL_EX_T_SMHS_VDS_BROFIFO_5MIN_20200911.csv")
OUT = Path("/Users/park/ml-project/results/dtg_slices_features.parquet")

TRAVERSAL_M = 1000.0     # 1km
MIN_SLICE_SEC = 20       # skip very fast slices (<20s = >180km/h avg, unrealistic)
MAX_VDS_DIST_M = 300
LANE_DEFAULT = 2
SPEED_LIMIT_DEFAULT = 100.0  # km/h


def haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_dtg(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, encoding="cp949", low_memory=False)
    df["일시"] = df["일시"].astype(str).str.lstrip("'")
    df["ts"] = pd.to_datetime(df["일시"], errors="coerce")
    for c in ("속도", "X", "Y", "RPM"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # brake: 'False'/'True'/0/1 mixed
    if df["브레이크"].dtype == object:
        df["brake"] = df["브레이크"].astype(str).str.strip().isin(["True", "1", "true", "TRUE"]).astype(int)
    else:
        df["brake"] = (df["브레이크"] > 0).astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    df["lon"] = df["X"] / 1e6
    df["lat"] = df["Y"] / 1e6
    df["gps_ok"] = (df["X"] > 0) & (df["Y"] > 0)
    df["speed_ms"] = df["속도"] / 3.6
    return df


def cumulative_distance_m(df: pd.DataFrame) -> np.ndarray:
    """Compute cumulative distance via speed×dt (1Hz integration)."""
    ts0 = df["ts"].iloc[0]
    times = (df["ts"] - ts0).dt.total_seconds().values  # seconds, unit-safe
    dt = np.diff(times, prepend=times[0])
    dt[dt < 0] = 0
    dt[dt > 5] = 5
    speed_seg = df["speed_ms"].fillna(0).clip(lower=0).values * dt
    return np.cumsum(speed_seg)


def slice_one_vehicle(df: pd.DataFrame) -> list[dict]:
    """Cut into 1km traversals starting fresh each slice."""
    if len(df) < MIN_SLICE_SEC:
        return []
    cum = cumulative_distance_m(df)
    slices = []
    slice_start_idx = 0
    slice_start_cum = cum[0]
    for i in range(1, len(df)):
        if cum[i] - slice_start_cum >= TRAVERSAL_M:
            slc = df.iloc[slice_start_idx:i+1]
            duration = (slc["ts"].iloc[-1] - slc["ts"].iloc[0]).total_seconds()
            if duration >= MIN_SLICE_SEC:
                slices.append({
                    "df": slc,
                    "traversal_m": cum[i] - slice_start_cum,
                    "traversal_s": duration,
                })
            slice_start_idx = i
            slice_start_cum = cum[i]
    return slices


def extract_features(slc: pd.DataFrame) -> dict:
    """Compute 30 trajectory features matching training schema."""
    speed = slc["speed_ms"].fillna(0).clip(lower=0).values
    brake = slc["brake"].fillna(0).astype(int).values
    n = len(speed)
    feat: dict = {}

    # === SPEED stats (9) ===
    feat["speed_mean"] = float(speed.mean())
    feat["speed_std"] = float(speed.std(ddof=1)) if n > 1 else 0.0
    feat["speed_cv"] = feat["speed_std"] / feat["speed_mean"] if feat["speed_mean"] > 0 else 0.0
    feat["speed_iqr"] = float(np.percentile(speed, 75) - np.percentile(speed, 25))
    feat["speed_min"] = float(speed.min())
    feat["speed_max"] = float(speed.max())
    feat["speed_median"] = float(np.median(speed))
    feat["speed_p10"] = float(np.percentile(speed, 10))
    feat["speed_p90"] = float(np.percentile(speed, 90))

    # === GPS-derived VX, VY, AX, AY, jerk ===
    gps = slc[slc["gps_ok"]].copy()
    if len(gps) >= 3:
        times = (gps["ts"] - gps["ts"].iloc[0]).dt.total_seconds().values
        lon = gps["lon"].values
        lat = gps["lat"].values
        # local meter conversion (equirectangular at mean lat)
        lat0 = lat.mean()
        x_m = (lon - lon[0]) * 111000 * np.cos(np.radians(lat0))
        y_m = (lat - lat[0]) * 111000
        dt = np.diff(times); dt[dt == 0] = 1e-3
        vx = np.diff(x_m) / dt
        vy = np.diff(y_m) / dt
        if len(vx) >= 2:
            dt2 = dt[1:]; dt2[dt2 == 0] = 1e-3
            ax = np.diff(vx) / dt2
            ay = np.diff(vy) / dt2
            jerk = np.diff(ax) / (dt2[1:] if len(dt2) > 1 else dt2)
        else:
            ax = ay = jerk = np.array([0.0])
    else:
        vx = vy = ax = ay = jerk = np.array([0.0])

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

    # === BRAKE patterns (3) ===
    # brake_count = number of brake events (consecutive runs of brake==1)
    if n > 0:
        runs = np.diff(np.concatenate([[0], brake, [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        n_brake_events = len(starts)
        feat["brake_count"] = int(n_brake_events)
        feat["brake_time_ratio"] = float(brake.mean())
        feat["mean_brake_duration"] = float(np.mean(ends - starts)) if n_brake_events > 0 else 0.0
    else:
        feat["brake_count"] = 0
        feat["brake_time_ratio"] = 0.0
        feat["mean_brake_duration"] = 0.0

    # === STOP patterns (3) === speed < 0.5 m/s
    stop_mask = (speed < 0.5).astype(int)
    runs = np.diff(np.concatenate([[0], stop_mask, [0]]))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    feat["stop_count"] = int(len(starts))
    feat["stop_time_ratio"] = float(stop_mask.mean())
    feat["mean_stop_duration"] = float(np.mean(ends - starts)) if len(starts) > 0 else 0.0

    # === TIME-SERIES (3) ===
    if n >= 3:
        try:
            cc = np.corrcoef(speed[:-1], speed[1:])[0, 1]
            feat["speed_autocorr_lag1"] = float(cc) if np.isfinite(cc) else 0.0
        except Exception:
            feat["speed_autocorr_lag1"] = 0.0
        # FFT dominant freq
        try:
            sp = speed - speed.mean()
            mag = np.abs(np.fft.rfft(sp))
            if len(mag) > 1:
                k = int(np.argmax(mag[1:]) + 1)  # skip DC
                feat["speed_fft_dominant_freq"] = float(k / n)  # cycles/sample
            else:
                feat["speed_fft_dominant_freq"] = 0.0
        except Exception:
            feat["speed_fft_dominant_freq"] = 0.0
        # sample entropy approximation (m=2, r=0.2*std)
        feat["sample_entropy"] = sample_entropy(speed, m=2, r=0.2 * (speed.std() if speed.std() > 0 else 1.0))
    else:
        feat["speed_autocorr_lag1"] = 0.0
        feat["speed_fft_dominant_freq"] = 0.0
        feat["sample_entropy"] = 0.0

    return feat


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Standard SampEn. O(N^2); fine for N≤500."""
    n = len(x)
    if n < m + 2 or r <= 0:
        return 0.0
    def _phi(m_local):
        templates = np.array([x[i:i + m_local] for i in range(n - m_local + 1)])
        cnt = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            cnt += int(np.sum(dist < r))
        return cnt
    a = _phi(m + 1)
    b = _phi(m)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def main() -> None:
    print("loading inputs...")
    hwy = pd.read_parquet(HWY_PTS)
    vds_info = pd.read_csv(VDS_INFO).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  highway points: {len(hwy):,}, VDS info: {len(vds_info):,}")

    # KDTree on VDS for snap
    vds_xy = np.c_[vds_info["lat"] * 111000, vds_info["lon"] * 88000]
    tree = cKDTree(vds_xy)

    # VDS 09-11 stats (for k_gt later)
    print("loading VDS 09-11 stats...")
    stats = pd.read_csv(VDS_STAT, encoding="utf-8-sig",
                        usecols=["AGGR_HM", "VDS_ID", "VMTC", "SHARE", "AVRG_VE"])
    stats = stats[(stats["VMTC"] >= 0) & (stats["AVRG_VE"] > 0)]
    # dedupe (rare duplicate rows): keep the first
    stats = stats.drop_duplicates(subset=["VDS_ID", "AGGR_HM"], keep="first")
    stat_map = stats.set_index(["VDS_ID", "AGGR_HM"]).to_dict(orient="index")
    print(f"  VDS stats (dedup): {len(stats):,}")

    # iterate vehicles
    print("\nslicing 30 vehicles into 1km traversals...")
    rows: list[dict] = []
    slice_id = 0
    for p in sorted(DTG_DIR.glob("*.csv")):
        plate = p.stem.split("_")[1]
        df = load_dtg(p)
        slices = slice_one_vehicle(df)
        n_kept = 0
        for s in slices:
            slc = s["df"]
            # is this slice on highway? require >=50% of slice points to be in hwy match
            slc_keys = list(zip(slc["ts"], slc["lat"], slc["lon"]))
            # Use mid-point GPS to snap to VDS and to MOCT link
            valid_gps = slc[slc["gps_ok"]]
            if len(valid_gps) < 3:
                continue
            mid = valid_gps.iloc[len(valid_gps) // 2]
            xy = np.array([[mid["lat"] * 111000, mid["lon"] * 88000]])
            dist, idx = tree.query(xy, k=1, distance_upper_bound=MAX_VDS_DIST_M)
            if not np.isfinite(dist[0]):
                continue
            vds_row = vds_info.iloc[idx[0]]
            vds_id = vds_row["vdsId"]
            # 5min bin
            mid_ts = mid["ts"]
            aggr_hm = mid_ts.hour * 100 + (mid_ts.minute // 5) * 5
            stat = stat_map.get((vds_id, aggr_hm))
            if stat is None:
                continue

            # Need num_lanes, speed_limit. Use MOCT match — pull from nearest hwy point
            # Match by nearest hwy point time
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

            feat = extract_features(slc)
            feat["num_lanes"] = num_lanes
            feat["speed_limit"] = speed_limit_ms

            # k_gt
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
        print(f"  [{plate}] {len(slices)} slices → {n_kept} matched on highway with VDS ground truth")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n=== TOTAL ===")
    print(f"slices with VDS GT: {len(out):,}")
    if len(out) > 0:
        print(f"unique VDS×5min bins: {out.groupby(['nearest_vdsId','vds_aggr_hm']).ngroups:,}")
    print(f"k_gt_qu: mean {out['k_gt_qu'].mean():.1f}, median {out['k_gt_qu'].median():.1f}, "
          f"p95 {out['k_gt_qu'].quantile(0.95):.1f}, max {out['k_gt_qu'].max():.1f}")
    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
