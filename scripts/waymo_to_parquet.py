"""Convert downloaded Waymo Motion shards into flat Parquet tables you can open in pandas/DuckDB.

Usage:
    python scripts/waymo_to_parquet.py                    # every shard in data/raw/waymo/
    python scripts/waymo_to_parquet.py --limit 50         # first 50 scenarios per shard (quick test)

Output (data/processed/waymo/):
    {split}_{shard}_tracks.parquet   one row per agent per timestep (x, y, heading, velocity, type, ...)
    {split}_{shard}_map.parquet      one row per map point (lanes, road edges, crosswalks, ...)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.waymo_reader import iter_scenarios, map_frame, tracks_frame  # noqa: E402

RAW = ROOT / "data" / "raw" / "waymo"
OUT = ROOT / "data" / "processed" / "waymo"


def convert(shard: Path, limit: int | None) -> None:
    split = shard.parent.name
    tag = shard.name.split("-")[1] if "-" in shard.name else shard.stem
    tracks, maps = [], []
    for sc in tqdm(iter_scenarios(shard, limit), desc=f"{split}/{shard.name}", unit="scenario"):
        tracks.append(tracks_frame(sc))
        maps.append(map_frame(sc))
    if not tracks:
        print(f"  no scenarios read from {shard.name}")
        return
    t, m = pd.concat(tracks, ignore_index=True), pd.concat(maps, ignore_index=True)
    t.to_parquet(OUT / f"{split}_{tag}_tracks.parquet", index=False)
    m.to_parquet(OUT / f"{split}_{tag}_map.parquet", index=False)
    print(f"  {len(tracks)} scenarios -> {len(t):,} track rows, {len(m):,} map points")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="max scenarios per shard")
    args = p.parse_args()
    shards = sorted(f for f in RAW.rglob("*tfrecord*") if f.is_file())
    if not shards:
        sys.exit(f"No shards found in {RAW}. Run scripts/download_waymo.sh first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for s in shards:
        convert(s, args.limit)
    print(f"\nDone. Parquet files are in {OUT}")


if __name__ == "__main__":
    main()
