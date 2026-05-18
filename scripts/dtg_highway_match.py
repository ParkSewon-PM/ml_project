"""Spatial-join DTG cohort GPS points against MOCT highway links (road_rank==101).

Output:
  results/dtg_highway_points.parquet — DTG points snapped to highway (≤50m)
  results/dtg_highway_overlap.csv    — co-presence cells on highway only
  results/dtg_highway_links.csv      — unique highway link_ids touched (VDS match candidates)
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=UserWarning)

DTG_DIR = Path("/Users/park/ml-project/data/DTG")
MOCT = Path("/Users/park/LLM-project/raw_data/[2026-01-13]NODELINKDATA/MOCT_LINK.shp")
OUT_DIR = Path("/Users/park/ml-project/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SNAP_DIST_M = 50  # within 50m of a highway link → counted as on-highway
LAT_BIN = 0.005   # ~500m
LON_BIN = 0.005
TIME_BIN_MIN = 5


def load_dtg_cohort() -> gpd.GeoDataFrame:
    rows = []
    for p in sorted(DTG_DIR.glob("*.csv")):
        df = pd.read_csv(p, encoding="cp949", low_memory=False)
        df["일시"] = df["일시"].astype(str).str.lstrip("'")
        df["ts"] = pd.to_datetime(df["일시"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df["속도"] = pd.to_numeric(df["속도"], errors="coerce")
        df = df[(df["X"] > 0) & (df["Y"] > 0) & df["ts"].notna()].copy()
        df["lon"] = df["X"] / 1e6
        df["lat"] = df["Y"] / 1e6
        parts = p.stem.split("_")
        df["plate"] = parts[1] if len(parts) >= 2 else p.stem
        rows.append(df[["ts", "lat", "lon", "속도", "plate"]])
    cohort = pd.concat(rows, ignore_index=True)
    cohort = cohort.rename(columns={"속도": "speed_kmh"})
    gdf = gpd.GeoDataFrame(
        cohort,
        geometry=[Point(lo, la) for lo, la in zip(cohort["lon"], cohort["lat"])],
        crs="EPSG:4326",
    )
    return gdf


def main() -> None:
    print(f"loading DTG cohort GPS points...")
    pts = load_dtg_cohort()
    print(f"  {len(pts):,} valid GPS points across {pts['plate'].nunique()} vehicles")

    # bbox for shapefile pre-filter
    minx, miny, maxx, maxy = pts.total_bounds
    print(f"  cohort bbox: lon [{minx:.3f}, {maxx:.3f}], lat [{miny:.3f}, {maxy:.3f}]")

    print(f"\nloading MOCT highway links (road_rank == '101')...")
    # bbox filter at read time (much faster than loading all 600K+ links)
    # MOCT is EPSG:5186; transform cohort bbox to 5186 first
    bbox_gdf = gpd.GeoDataFrame(geometry=[Point(minx, miny), Point(maxx, maxy)], crs="EPSG:4326")
    bbox_5186 = bbox_gdf.to_crs("EPSG:5186").total_bounds
    pad = 5000  # 5km padding
    bbox_5186_padded = (bbox_5186[0] - pad, bbox_5186[1] - pad,
                       bbox_5186[2] + pad, bbox_5186[3] + pad)
    print(f"  reading bbox 5186 (with 5km pad)...")
    links = gpd.read_file(MOCT, bbox=bbox_5186_padded)
    print(f"  loaded {len(links):,} links in cohort bbox")
    hwy = links[links["ROAD_RANK"] == "101"].copy()
    print(f"  filtered to {len(hwy):,} highway (road_rank=101) links")
    print(f"  highway road_names: {hwy['ROAD_NAME'].value_counts().head(8).to_dict()}")

    # reproject DTG points to EPSG:5186 for spatial join
    print(f"\nreprojecting DTG points to EPSG:5186...")
    pts_5186 = pts.to_crs("EPSG:5186")

    # sjoin_nearest with max distance
    print(f"sjoin_nearest (max {MAX_SNAP_DIST_M}m)...")
    joined = gpd.sjoin_nearest(
        pts_5186, hwy[["LINK_ID", "ROAD_NAME", "ROAD_NO", "LANES", "MAX_SPD", "geometry"]],
        how="left", max_distance=MAX_SNAP_DIST_M, distance_col="snap_dist_m"
    )

    on_hwy = joined[joined["LINK_ID"].notna()].copy()
    print(f"\nresult:")
    print(f"  total points:           {len(pts):,}")
    print(f"  matched to highway:     {len(on_hwy):,} ({len(on_hwy)/len(pts)*100:.1f}%)")
    print(f"  unique highway links:   {on_hwy['LINK_ID'].nunique():,}")
    print(f"  snap distance — median {on_hwy['snap_dist_m'].median():.1f}m, "
          f"p95 {on_hwy['snap_dist_m'].quantile(0.95):.1f}m")

    print(f"\n  per-vehicle highway point count:")
    per_veh = on_hwy.groupby("plate").size().sort_values(ascending=False)
    print(per_veh.to_string())

    # save highway-only points (back to WGS84 for downstream)
    on_hwy_wgs = on_hwy.copy()
    on_hwy_wgs["lon"] = on_hwy_wgs.geometry.to_crs("EPSG:4326").x
    on_hwy_wgs["lat"] = on_hwy_wgs.geometry.to_crs("EPSG:4326").y
    out_pts = OUT_DIR / "dtg_highway_points.parquet"
    on_hwy_wgs[["ts", "plate", "lat", "lon", "speed_kmh",
                 "LINK_ID", "ROAD_NAME", "ROAD_NO", "LANES", "MAX_SPD",
                 "snap_dist_m"]].to_parquet(out_pts, index=False)
    print(f"\n→ wrote {out_pts}")

    # co-presence on highway
    print(f"\n=== Co-presence on HIGHWAY ===")
    on_hwy_wgs["lat_bin"] = (on_hwy_wgs["lat"] / LAT_BIN).round().astype(int)
    on_hwy_wgs["lon_bin"] = (on_hwy_wgs["lon"] / LON_BIN).round().astype(int)
    on_hwy_wgs["time_bin"] = on_hwy_wgs["ts"].dt.floor(f"{TIME_BIN_MIN}min")

    cells = (
        on_hwy_wgs.groupby(["lat_bin", "lon_bin", "time_bin"])
        .agg(
            n_plates=("plate", "nunique"),
            n_points=("plate", "size"),
            mean_speed=("speed_kmh", "mean"),
            road_name=("ROAD_NAME", lambda s: s.mode().iloc[0] if len(s) else ""),
            link_ids=("LINK_ID", lambda s: ",".join(sorted(set(s)))),
            plates=("plate", lambda s: ",".join(sorted(s.unique()))),
        )
        .reset_index()
    )
    overlap = cells[cells["n_plates"] >= 2].copy()
    overlap["lat_center"] = overlap["lat_bin"] * LAT_BIN
    overlap["lon_center"] = overlap["lon_bin"] * LON_BIN
    overlap = overlap.sort_values(["n_plates", "n_points"], ascending=[False, False])

    print(f"  cells with ≥1 vehicle:  {len(cells):,}")
    print(f"  cells with ≥2 vehicles: {len(overlap):,}")
    print(f"  cells with ≥3 vehicles: {(overlap['n_plates']>=3).sum():,}")
    print(f"  cells with ≥5 vehicles: {(overlap['n_plates']>=5).sum():,}")

    print(f"\n  top 15 highway co-presence cells:")
    cols = ["time_bin", "road_name", "lat_center", "lon_center",
            "n_plates", "n_points", "mean_speed"]
    print(overlap[cols].head(15).to_string(index=False))

    out_overlap = OUT_DIR / "dtg_highway_overlap.csv"
    overlap.to_csv(out_overlap, index=False)
    print(f"\n→ wrote {out_overlap}")

    # unique highway link list for VDS matching
    unique_links = (
        on_hwy_wgs.groupby("LINK_ID")
        .agg(
            road_name=("ROAD_NAME", "first"),
            road_no=("ROAD_NO", "first"),
            lanes=("LANES", "first"),
            max_spd=("MAX_SPD", "first"),
            n_points=("plate", "size"),
            n_plates=("plate", "nunique"),
            lat_center=("lat", "mean"),
            lon_center=("lon", "mean"),
        )
        .reset_index()
        .sort_values(["n_plates", "n_points"], ascending=[False, False])
    )
    out_links = OUT_DIR / "dtg_highway_links.csv"
    unique_links.to_csv(out_links, index=False)
    print(f"→ wrote {out_links}  ({len(unique_links)} unique highway links)")

    print(f"\n  top 10 highway links by vehicle traversal:")
    print(unique_links.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
