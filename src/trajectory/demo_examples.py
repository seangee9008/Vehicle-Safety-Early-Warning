"""Export ~50 test agents with everything the demo needs to draw them, without Waymo data.

For each agent (validation_00000, the test shard): past path, true future, constant-velocity
path, the selected model's 6 paths + confidences (highest first), nearby lane centre lines, and
FDE@8s for CV / best guess / min of 6. All coordinates are in the agent's frame: metres, origin =
position now, +x = heading now. Invalid steps are null. Rounded to 1 cm.

Picked with a fixed seed among agents that moved at least 2 m (25 vehicles, 15 pedestrians,
10 cyclists), so reruns give the same set.

    python -m src.trajectory.demo_examples      # -> data/processed/waymo/demo_examples.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.trajectory.dataset import (
    TEST_NPZ,
    TEST_TRACKS,
    WAYMO_DIR,
    load,
    to_agent_frame,
)
from src.trajectory.evaluate import (
    SELECTED_CHECKPOINT,
    constant_velocity_paths_np,
    per_agent_table,
)
from src.trajectory.model import load_checkpoint, predict

OUT = WAYMO_DIR / "demo_examples.json"
PER_TYPE = {"vehicle": 25, "pedestrian": 15, "cyclist": 10}
MIN_TRAVEL_M = 2.0
LANE_RADIUS_M = 60.0
LANE_POINT_STEP = 3  # keep every 3rd lane point (~1.5 m apart)
SEED = 0
FRAME = "agent frame: metres, origin = position now, +x = heading now"


def _xy(points: np.ndarray, valid: np.ndarray | None = None) -> list:
    out = np.round(points.astype(np.float64), 2).tolist()
    if valid is not None:
        out = [p if v else None for p, v in zip(out, valid.tolist(), strict=True)]
    return out


def pick(agents: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    at8 = agents[(agents["horizon_s"] == 8) & (agents["travelled_m"] >= MIN_TRAVEL_M)]
    picks = []
    for object_type, n in PER_TYPE.items():
        pool = at8[at8["object_type"] == object_type]
        picks.append(pool.iloc[np.sort(rng.choice(len(pool), size=min(n, len(pool)), replace=False))])
    return pd.concat(picks)


def lane_polylines(lanes: pd.DataFrame, origin: np.ndarray, yaw: float) -> list:
    if lanes.empty:
        return []
    xy = to_agent_frame(lanes[["x", "y"]].to_numpy()[None], origin[None], np.array([yaw]))[0]
    near = np.linalg.norm(xy, axis=1) <= LANE_RADIUS_M
    frame = lanes.assign(ax=xy[:, 0], ay=xy[:, 1])[near]
    out = []
    for _, line in frame.sort_values(["feature_id", "point_idx"]).groupby("feature_id", sort=False):
        pts = line[["ax", "ay"]].to_numpy()[::LANE_POINT_STEP]
        if len(pts) >= 2:
            out.append(_xy(pts))
    return out


def build(test_npz: Path = TEST_NPZ, checkpoint: Path = SELECTED_CHECKPOINT) -> list[dict]:
    test = load(test_npz)
    model, info = load_checkpoint(checkpoint)
    paths, conf = predict(model, test, temperature=info.get("temperature", 1.0))
    paths = paths.astype(np.float64)
    chosen = pick(per_agent_table(test, paths, conf))
    index = {(s, int(t)): i for i, (s, t) in enumerate(zip(test["scenario_id"], test["track_id"]))}
    cv = constant_velocity_paths_np(test["past"])
    map_path = TEST_TRACKS.with_name(TEST_TRACKS.name.replace("_tracks", "_map"))
    lanes = pd.read_parquet(
        map_path, columns=["scenario_id", "feature_id", "point_idx", "x", "y"],
        filters=[("scenario_id", "in", sorted(set(chosen["scenario_id"]))),
                 ("feature_type", "==", "lane")],
    )
    examples = []
    for _, row in chosen.iterrows():
        i = index[(row["scenario_id"], int(row["track_id"]))]
        order = np.argsort(-conf[i])
        examples.append({
            "example_id": f"{row['scenario_id']}_{int(row['track_id'])}",
            "scenario_id": str(row["scenario_id"]),
            "track_id": int(row["track_id"]),
            "object_type": row["object_type"],
            "frame": FRAME,
            "past_path": _xy(test["past"][i, :, :2], test["past_valid"][i]),
            "true_future": _xy(test["future"][i], test["future_valid"][i]),
            "baseline_path": _xy(cv[i]),
            "model_paths": [_xy(paths[i, k]) for k in order],
            "model_confidences": np.round(conf[i, order].astype(np.float64), 4).tolist(),
            "lane_polylines": lane_polylines(lanes[lanes["scenario_id"] == row["scenario_id"]],
                                             test["origin"][i], float(test["yaw"][i])),
            "fde_8s_cv_m": round(float(row["cv_fde_m"]), 3),
            "fde_8s_best_m": round(float(row["best_fde_m"]), 3),
            "fde_8s_min_m": round(float(row["min_fde_m"]), 3),
            "travelled_m": round(float(row["travelled_m"]), 3),
            "checkpoint": checkpoint.name,
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    examples = build()
    args.out.write_text(json.dumps(examples), encoding="utf-8")
    counts = pd.Series([e["object_type"] for e in examples]).value_counts().to_dict()
    print(f"wrote {args.out} ({len(examples)} agents {counts}, {args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
