"""Find time × space overlaps across 30 DTG vehicles.

For density inference cross-check, we need cases where ≥2 vehicles
passed through the SAME road segment at the SAME time. The model
should predict similar density for them (sanity test before VDS).

Approach:
  1. Quantize each GPS point to (lat_bin, lon_bin, 5-min time bin).
     bins: 0.005° (≈ 500m) × 5 min.
  2. Count distinct plates per cell.
  3. Report cells with ≥2 plates (= co-presence candidates for inference).
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

DTG_DIR = Path("/Users/park/ml-project/data/DTG")
OUT = Path("/Users/park/ml-project/results/dtg_cohort_overlap.csv")

LAT_BIN = 0.005   # ~500m
LON_BIN = 0.005
TIME_BIN_MIN = 5  # 5-minute window


def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949", low_memory=False)
    df["일시"] = df["일시"].astype(str).str.lstrip("'")
    df["ts"] = pd.to_datetime(df["일시"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["속도"] = pd.to_numeric(df["속도"], errors="coerce")
    df = df[(df["X"] > 0) & (df["Y"] > 0) & df["ts"].notna()].copy()
    df["lon"] = df["X"] / 1e6
    df["lat"] = df["Y"] / 1e6
    parts = path.stem.split("_")
    df["plate"] = parts[1] if len(parts) >= 2 else path.stem
    return df[["ts", "lat", "lon", "속도", "plate"]]


def main() -> None:
    files = sorted(DTG_DIR.glob("*.csv"))
    print(f"loading {len(files)} files...")
    all_df = pd.concat([load_one(p) for p in files], ignore_index=True)
    print(f"total GPS points across cohort: {len(all_df):,}")

    # Quantize
    all_df["lat_bin"] = (all_df["lat"] / LAT_BIN).round().astype(int)
    all_df["lon_bin"] = (all_df["lon"] / LON_BIN).round().astype(int)
    all_df["time_bin"] = all_df["ts"].dt.floor(f"{TIME_BIN_MIN}min")

    # Count distinct plates per cell
    cells = (
        all_df.groupby(["lat_bin", "lon_bin", "time_bin"])
        .agg(
            n_plates=("plate", "nunique"),
            n_points=("plate", "size"),
            mean_speed=("속도", "mean"),
            plates=("plate", lambda s: ",".join(sorted(s.unique()))),
        )
        .reset_index()
    )

    overlap = cells[cells["n_plates"] >= 2].copy()
    overlap["lat_center"] = overlap["lat_bin"] * LAT_BIN
    overlap["lon_center"] = overlap["lon_bin"] * LON_BIN
    overlap = overlap.sort_values("n_plates", ascending=False)

    print(f"\nTotal cells (≥1 plate): {len(cells):,}")
    print(f"Co-presence cells (≥2 plates): {len(overlap):,}")
    print(f"Co-presence cells (≥3 plates): {(overlap['n_plates']>=3).sum():,}")
    print(f"Co-presence cells (≥5 plates): {(overlap['n_plates']>=5).sum():,}")

    print(f"\n=== Top 15 cells (most plates co-present) ===")
    print(overlap[["time_bin", "lat_center", "lon_center", "n_plates", "n_points", "mean_speed", "plates"]].head(15).to_string(index=False))

    # Distribution of co-presence
    print(f"\n=== n_plates distribution ===")
    print(overlap["n_plates"].value_counts().sort_index().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(OUT, index=False)
    print(f"\n→ wrote {OUT} ({len(overlap):,} rows)")


if __name__ == "__main__":
    main()
