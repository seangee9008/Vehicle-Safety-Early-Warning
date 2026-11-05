"""Read Waymo Open Motion Dataset Scenario TFRecords with plain Python (no TensorFlow).

Run `python scripts/build_waymo_protos.py` once first.

    from src.waymo_reader import iter_scenarios, tracks_frame, map_frame
    for sc in iter_scenarios("data/raw/waymo/validation/validation.tfrecord-00000-of-00150"):
        tracks = tracks_frame(sc)   # one row per agent per timestep
        lanes = map_frame(sc)       # one row per map point
        break
"""
import struct
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

_PB = Path(__file__).resolve().parent / "waymo_pb"
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))

OBJECT_TYPES = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}


def _scenario_cls():
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as e:
        raise ImportError("Waymo protos not compiled. Run: python scripts/build_waymo_protos.py") from e
    return scenario_pb2.Scenario


def iter_records(path: str | Path) -> Iterator[bytes]:
    """Yield raw records from an uncompressed TFRecord file.

    Format per record: uint64 length | uint32 crc | data | uint32 crc (CRCs not checked).
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise ValueError(f"{path}: truncated record header (is the download complete?)")
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            if len(data) < length:
                raise ValueError(f"{path}: truncated record (is the download complete?)")
            f.read(4)
            yield data


def iter_scenarios(path: str | Path, limit: int | None = None):
    Scenario = _scenario_cls()
    for n, raw in enumerate(iter_records(path)):
        if limit is not None and n >= limit:
            return
        sc = Scenario()
        sc.ParseFromString(raw)
        yield sc


def tracks_frame(sc) -> pd.DataFrame:
    """One row per (agent, timestep). 91 timesteps at 10 Hz; current_time_index is usually 10."""
    predict = {r.track_index for r in sc.tracks_to_predict}
    rows = []
    for ti, t in enumerate(sc.tracks):
        for step, s in enumerate(t.states):
            rows.append((sc.scenario_id, t.id, OBJECT_TYPES.get(t.object_type, "other"),
                         ti == sc.sdc_track_index, ti in predict, step,
                         sc.timestamps_seconds[step] if step < len(sc.timestamps_seconds) else None,
                         step - sc.current_time_index, s.valid,
                         s.center_x, s.center_y, s.center_z, s.length, s.width, s.height,
                         s.heading, s.velocity_x, s.velocity_y))
    return pd.DataFrame(rows, columns=[
        "scenario_id", "track_id", "object_type", "is_sdc", "to_predict", "timestep",
        "time_s", "steps_from_current", "valid", "x", "y", "z", "length", "width", "height",
        "heading", "vx", "vy"])


def map_frame(sc) -> pd.DataFrame:
    """One row per map point: lanes, road lines, road edges, crosswalks, speed bumps, driveways, stop signs."""
    rows = []
    for mf in sc.map_features:
        kind = mf.WhichOneof("feature_data")
        if kind is None:
            continue
        part = getattr(mf, kind)
        if kind == "stop_sign":
            points = [part.position]
        elif hasattr(part, "polyline"):
            points = part.polyline
        else:
            points = part.polygon
        for i, pt in enumerate(points):
            rows.append((sc.scenario_id, mf.id, kind, i, pt.x, pt.y, pt.z))
    return pd.DataFrame(rows, columns=["scenario_id", "feature_id", "feature_type", "point_idx", "x", "y", "z"])
