"""Fetch all VDS install info from 한국도로공사 OpenAPI.

Usage:
  EX_API_KEY=<your_key> python scripts/fetch_vds_info.py

Output:
  data/vds_info.csv  (vdsId, lat, lon, route, shift, direction, road_grade, lanes...)

Pagination: ~7,500 VDS → 8 pages × 1,000 rows. ~30 seconds.

Coordinate note:
  grs80x/grs80y are GRS80 projected (EPSG:5186 ITRF2000 Central Belt TM).
  Same CRS as MOCT_LINK shapefile → spatial join is straightforward.
  We also derive WGS84 lat/lon for DTG matching.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from pyproj import Transformer

API_URL = "https://data.ex.co.kr/openapi/vdsinfo/vdsList"
OUT = Path("/Users/park/ml-project/data/vds_info.csv")
PER_PAGE = 1000


def fetch_page(key: str, page: int) -> dict:
    params = {"key": key, "type": "json", "numOfRows": PER_PAGE, "pageNo": page}
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    key = os.environ.get("EX_API_KEY")
    if not key:
        print("ERROR: set EX_API_KEY=<your_key> in env", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    page = 1
    while True:
        print(f"  page {page}...", end=" ", flush=True)
        data = fetch_page(key, page)
        # response shape per docs: { code, message, count, pageNo, numOfRows, pageSize, list:[...] }
        # list key name varies between API endpoints; try a few
        items = data.get("list") or data.get("data") or data.get("vdsList") or []
        if isinstance(items, dict):
            items = [items]
        if not items:
            print("(no more items)")
            break
        rows.extend(items)
        total = int(data.get("count", 0))
        page_size = int(data.get("pageSize", 0))
        print(f"got {len(items)} (cum {len(rows)}, expected {total})")
        if total and len(rows) >= total:
            break
        if page_size and page >= page_size:
            break
        page += 1
        time.sleep(0.2)  # polite

    if not rows:
        print("ERROR: empty response. Check API key & docs.", file=sys.stderr)
        print("First response sample (for debugging):")
        print(data)
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\ncolumns: {list(df.columns)}")
    print(f"total VDS: {len(df):,}")

    # Coerce numeric
    for col in ("grs80x", "grs80y", "shift", "vdsStartShift", "vdsEndShift", "vdsLength"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # GRS80 TM (EPSG:5186) → WGS84 (EPSG:4326)
    if "grs80x" in df.columns and "grs80y" in df.columns:
        tr = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
        lons, lats = tr.transform(df["grs80x"].values, df["grs80y"].values)
        df["lon"] = lons
        df["lat"] = lats
        print(f"  bbox lat [{df['lat'].min():.3f}, {df['lat'].max():.3f}], "
              f"lon [{df['lon'].min():.3f}, {df['lon'].max():.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n→ wrote {OUT}")
    print(df[["vdsId", "routeNo", "routeName", "shift", "directionCode", "lon", "lat"]].head())


if __name__ == "__main__":
    main()
