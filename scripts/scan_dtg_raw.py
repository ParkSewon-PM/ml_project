"""Scan all 30 raw DTG CSVs and report per-file profile + cohort summary.

Format notes (verified from 1초별데이터_경북82아1208_20200911.csv):
  - 1Hz time axis, but GPS coordinates update every 3 seconds (33.3% of rows have X>0).
  - `총운행거리(Km)` is recorded only at ignition events (~0% nonzero); not usable for distance.
  - X/Y are micro-degrees (lon×1e6, lat×1e6).
  - 가속도X/Y units unclear (integer, may not be m/s²); leave raw.
  - 속도 unit: km/h.

Output:
  results/dtg_raw_scan.csv  (one row per file)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DTG_DIR = Path("/Users/park/ml-project/data/DTG")
OUT_DIR = Path("/Users/park/ml-project/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def scan_one(path: Path) -> dict:
    df = pd.read_csv(path, encoding="cp949", low_memory=False)
    for col in ("일자", "시각", "일시"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lstrip("'")

    df["속도"] = pd.to_numeric(df["속도"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["ts"] = pd.to_datetime(df["일시"], errors="coerce")

    parts = path.stem.split("_")
    plate = parts[1] if len(parts) >= 2 else ""
    date = parts[2] if len(parts) >= 3 else ""

    gps_ok = (df["X"] > 0) & (df["Y"] > 0)
    moving = df["속도"] >= 1

    lon = df.loc[gps_ok, "X"] / 1e6
    lat = df.loc[gps_ok, "Y"] / 1e6

    t_start = df["ts"].min()
    t_end = df["ts"].max()
    duration_hr = (t_end - t_start).total_seconds() / 3600 if pd.notna(t_start) else np.nan

    # distance from speed integration: sum(speed_kmh * dt_hr)
    # treat each row as 1s; speed in km/h
    dist_speed_km = (df["속도"].clip(lower=0).sum()) / 3600

    # distance from GPS (rough): sum great-circle between consecutive GPS-valid points
    gps_df = df.loc[gps_ok, ["X", "Y"]].copy().reset_index(drop=True)
    gps_df["lon"] = gps_df["X"] / 1e6
    gps_df["lat"] = gps_df["Y"] / 1e6
    if len(gps_df) >= 2:
        dlat = gps_df["lat"].diff() * 111.0
        dlon = gps_df["lon"].diff() * 111.0 * np.cos(np.radians(gps_df["lat"].mean()))
        seg = np.sqrt(dlat**2 + dlon**2)
        # drop unrealistic jumps (GPS noise → >10 km in one 3s step impossible)
        seg = seg[seg < 10]
        dist_gps_km = seg.sum()
    else:
        dist_gps_km = np.nan

    # highway-share heuristic: speed > 80 km/h share of moving points
    if moving.sum() > 0:
        hwy_share = (df.loc[moving, "속도"] > 80).mean() * 100
    else:
        hwy_share = 0.0

    return {
        "plate": plate,
        "date": date,
        "n_rows": len(df),
        "n_gps_points": int(gps_ok.sum()),
        "duration_hr": round(duration_hr, 2) if not np.isnan(duration_hr) else np.nan,
        "t_start": str(t_start)[:19],
        "t_end": str(t_end)[:19],
        "stop_pct": round((df["속도"] < 1).mean() * 100, 1),
        "speed_mean_moving": round(df.loc[moving, "속도"].mean(), 1),
        "speed_p50_moving": round(df.loc[moving, "속도"].median(), 1),
        "speed_p90": round(df["속도"].quantile(0.9), 1),
        "speed_max": int(df["속도"].max()) if pd.notna(df["속도"].max()) else None,
        "hwy_share_pct": round(hwy_share, 1),
        "dist_speed_km": round(dist_speed_km, 0),
        "dist_gps_km": round(dist_gps_km, 0) if not np.isnan(dist_gps_km) else np.nan,
        "lon_min": round(lon.min(), 4) if len(lon) else np.nan,
        "lon_max": round(lon.max(), 4) if len(lon) else np.nan,
        "lat_min": round(lat.min(), 4) if len(lat) else np.nan,
        "lat_max": round(lat.max(), 4) if len(lat) else np.nan,
        "lon_center": round(lon.mean(), 4) if len(lon) else np.nan,
        "lat_center": round(lat.mean(), 4) if len(lat) else np.nan,
        "trip_span_km": round(max((lon.max() - lon.min()) * 88 if len(lon) else 0,
                                    (lat.max() - lat.min()) * 111 if len(lat) else 0), 0),
        "file_mb": round(path.stat().st_size / 1024 / 1024, 1),
    }


def main() -> None:
    files = sorted(DTG_DIR.glob("*.csv"))
    print(f"scanning {len(files)} files...\n")
    rows = []
    for i, p in enumerate(files, 1):
        try:
            rec = scan_one(p)
            rows.append(rec)
            print(f"  [{i:2d}/{len(files)}] {rec['plate']:>10s} {rec['date']}: "
                  f"n={rec['n_rows']:>6,}  "
                  f"dur={rec['duration_hr']:>5.1f}h  "
                  f"speed={rec['speed_mean_moving']:>5.1f}km/h  "
                  f"hwy={rec['hwy_share_pct']:>5.1f}%  "
                  f"dist≈{rec['dist_speed_km']:>5.0f}km  "
                  f"trip span={rec['trip_span_km']:>4.0f}km")
        except Exception as e:
            print(f"  [{i:2d}/{len(files)}] {p.name}: ERROR {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "dtg_raw_scan.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}\nCOHORT SUMMARY ({len(out_df)} files)\n{'='*70}")
    print(f"total rows:           {out_df['n_rows'].sum():,}")
    print(f"total GPS points:     {out_df['n_gps_points'].sum():,}")
    print(f"total drive hours:    {out_df['duration_hr'].sum():.0f}")
    print(f"total distance (sp):  {out_df['dist_speed_km'].sum():,.0f} km")
    print(f"total distance (gps): {out_df['dist_gps_km'].sum():,.0f} km")

    print("\n속도 (운행 중 평균):")
    print(out_df["speed_mean_moving"].describe().round(1).to_string())
    print(f"\n고속도로 share% (>80km/h 운행점 비율):")
    print(out_df["hwy_share_pct"].describe().round(1).to_string())
    print(f"\n정지 비율%:")
    print(out_df["stop_pct"].describe().round(1).to_string())

    print(f"\ntrip span (한 차량의 운행 영역 대각선 거리):")
    print(out_df["trip_span_km"].describe().round(0).to_string())

    # rough region tags
    def region_tag(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return "unknown"
        if 35.7 < lat < 36.7 and 128.5 < lon < 129.5: return "경북"
        if 35.4 < lat < 35.8 and 129.1 < lon < 129.5: return "울산"
        if 34.5 < lat < 35.3 and 127.0 < lon < 128.5: return "전남/경남남부"
        if 36.0 < lat < 37.0 and 127.0 < lon < 128.0: return "충북/충남"
        if 35.5 < lat < 36.0 and 128.0 < lon < 128.5: return "경북남부/경남"
        return f"기타({lat:.1f},{lon:.1f})"

    out_df["region_tag"] = out_df.apply(
        lambda r: region_tag(r["lat_center"], r["lon_center"]), axis=1
    )
    print(f"\n실제 운행 지역 분포 (lat/lon 중심 기준):")
    print(out_df["region_tag"].value_counts().to_string())

    out_df.to_csv(out_csv, index=False)
    print(f"\n→ wrote {out_csv}")


if __name__ == "__main__":
    main()
