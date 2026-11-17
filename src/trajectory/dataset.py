"""Agent-centric examples for trajectory models, built from Waymo tracks parquet.

One example per agent with ``to_predict`` that is valid at the current step and has at least one
valid future step (the same agents the constant-velocity baseline scores). Everything is expressed
in the agent's own frame: current position at the origin, current heading along +x.

Arrays in each ``.npz``:
- ``past``          (N, 11, 5) float32: x, y, vx, vy, heading for steps -10..0 (zeros where invalid)
- ``past_valid``    (N, 11) bool
- ``future``        (N, 80, 2) float32: x, y for steps 1..80 (zeros where invalid)
- ``future_valid``  (N, 80) bool
- ``agent_type``    (N,) int8, index into ``AGENT_TYPES``
- ``origin``        (N, 2) float64 world position at step 0; ``yaw`` (N,) world heading at step 0
- ``scenario_id``   (N,) str; ``track_id`` (N,) int64

Training examples come only from ``training_*`` shards; ``validation_00000`` is test only.

    python -m src.trajectory.dataset        # writes data/processed/waymo/{train,validation}_examples.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WAYMO_DIR = ROOT / "data" / "processed" / "waymo"
TRAIN_NPZ = WAYMO_DIR / "train_examples.npz"
TEST_NPZ = WAYMO_DIR / "validation_00000_examples.npz"
TEST_TRACKS = WAYMO_DIR / "validation_00000_tracks.parquet"

PAST_STEPS = 11  # steps -10..0
FUTURE_STEPS = 80  # steps 1..80 (8 s at 10 Hz)
AGENT_TYPES = ("vehicle", "pedestrian", "cyclist", "other")
KEY = ["scenario_id", "track_id"]
COLUMNS = [
    "scenario_id", "track_id", "object_type", "to_predict",
    "steps_from_current", "valid", "x", "y", "heading", "vx", "vy",
]


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotate(xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate (..., T, 2) vectors by ``angle`` (shape (...,)) counter-clockwise."""
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def to_agent_frame(xy: np.ndarray, origin: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """World (N, T, 2) -> agent frame, given origin (N, 2) and yaw (N,)."""
    return rotate(xy - origin[:, None, :], -yaw)


def to_world_frame(xy: np.ndarray, origin: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Agent frame (N, ..., T, 2) -> world. Extra middle axes (e.g. K candidates) broadcast."""
    extra = xy.ndim - 3
    yaw_b = yaw.reshape(yaw.shape + (1,) * extra)
    origin_b = origin.reshape(origin.shape[:1] + (1,) * (extra + 1) + (2,))
    return rotate(xy, yaw_b) + origin_b


def build_examples(tracks: pd.DataFrame) -> dict[str, np.ndarray]:
    """Turn a tracks frame (columns in ``COLUMNS``) into agent-frame arrays."""
    t = tracks[tracks["to_predict"].astype(bool)]
    t = t[t["steps_from_current"].between(-(PAST_STEPS - 1), FUTURE_STEPS)]
    current = t[(t["steps_from_current"] == 0) & t["valid"].astype(bool)]
    keys = current[[*KEY, "object_type"]].drop_duplicates(KEY).sort_values(KEY, ignore_index=True)
    keys["idx"] = np.arange(len(keys))
    rows = t.merge(keys[[*KEY, "idx"]], on=KEY)

    n, steps = len(keys), PAST_STEPS + FUTURE_STEPS
    i = rows["idx"].to_numpy()
    s = rows["steps_from_current"].to_numpy() + PAST_STEPS - 1
    valid = np.zeros((n, steps), bool)
    pos = np.zeros((n, steps, 2))
    vel = np.zeros((n, steps, 2))
    head = np.zeros((n, steps))
    valid[i, s] = rows["valid"].astype(bool).to_numpy()
    pos[i, s] = rows[["x", "y"]].to_numpy(float)
    vel[i, s] = rows[["vx", "vy"]].to_numpy(float)
    head[i, s] = rows["heading"].to_numpy(float)

    now = PAST_STEPS - 1
    origin, yaw = pos[:, now].copy(), head[:, now].copy()
    local = to_agent_frame(pos, origin, yaw)
    local_vel = rotate(vel, -yaw)
    local_head = wrap_angle(head - yaw[:, None])
    past = np.concatenate([local, local_vel, local_head[..., None]], axis=-1)[:, :PAST_STEPS]
    past[~valid[:, :PAST_STEPS]] = 0.0
    future = local[:, PAST_STEPS:].copy()
    future[~valid[:, PAST_STEPS:]] = 0.0

    type_index = {name: k for k, name in enumerate(AGENT_TYPES)}
    examples = {
        "past": past.astype(np.float32),
        "past_valid": valid[:, :PAST_STEPS],
        "future": future.astype(np.float32),
        "future_valid": valid[:, PAST_STEPS:],
        "agent_type": keys["object_type"].map(type_index).fillna(3).to_numpy(np.int8),
        "origin": origin,
        "yaw": yaw,
        "scenario_id": keys["scenario_id"].to_numpy(str),
        "track_id": keys["track_id"].to_numpy(np.int64),
    }
    keep = examples["future_valid"].any(axis=1)
    return {name: arr[keep] for name, arr in examples.items()}


def concat_examples(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def load_tracks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=COLUMNS, filters=[("to_predict", "==", True)])


def training_track_files(waymo_dir: Path = WAYMO_DIR) -> list[Path]:
    """Only ``training_*`` shards: validation is never used for fitting."""
    return sorted(waymo_dir.glob("training_*_tracks.parquet"))


def save(examples: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **examples)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def describe(examples: dict[str, np.ndarray], label: str) -> str:
    types = pd.Series(np.asarray(AGENT_TYPES)[examples["agent_type"]]).value_counts()
    by_type = ", ".join(f"{name} {count}" for name, count in types.items())
    return (f"{label}: {len(np.unique(examples['scenario_id']))} scenarios, "
            f"{len(examples['agent_type'])} agents ({by_type})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-out", type=Path, default=TRAIN_NPZ)
    parser.add_argument("--test-out", type=Path, default=TEST_NPZ)
    args = parser.parse_args()

    files = training_track_files()
    if not files:
        raise SystemExit(f"No training_*_tracks.parquet in {WAYMO_DIR}. Run waymo_to_parquet.py.")
    train = concat_examples([build_examples(load_tracks(f)) for f in files])
    test = build_examples(load_tracks(TEST_TRACKS))
    overlap = set(train["scenario_id"]) & set(test["scenario_id"])
    if overlap:
        raise SystemExit(f"{len(overlap)} validation scenarios found in training data")
    save(train, args.train_out)
    save(test, args.test_out)
    print(describe(train, f"train ({', '.join(f.name for f in files)})"))
    print(describe(test, "test (validation_00000)"))
    print(f"Wrote {args.train_out} and {args.test_out}")


if __name__ == "__main__":
    main()
