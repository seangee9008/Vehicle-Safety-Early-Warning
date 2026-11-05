"""Quick checks that the setup code works. Run: pytest -q"""
import struct
from pathlib import Path

import pytest

from src.nhtsa_schemas import COMPLAINTS, INVESTIGATIONS, RECALLS

ROOT = Path(__file__).resolve().parents[1]


def test_complaint_schema_matches_dictionary():
    assert len(COMPLAINTS) == 51
    assert COMPLAINTS[16] == "LDATE" and COMPLAINTS[19] == "CDESCR"


def test_investigation_schema_matches_dictionary():
    assert len(INVESTIGATIONS) == 11 and INVESTIGATIONS[8] == "CAMPNO"


def test_recall_schema_matches_dictionary():
    assert len(RECALLS) == 29
    assert RECALLS[-2:] == ["DO_NOT_DRIVE", "PARK_OUTSIDE"]


def _write_tfrecord(path: Path, records: list[bytes]) -> None:
    with open(path, "wb") as f:
        for r in records:
            f.write(struct.pack("<Q", len(r)) + b"\0" * 4 + r + b"\0" * 4)


@pytest.mark.skipif(not (ROOT / "src" / "waymo_pb" / "waymo_open_dataset").exists(),
                    reason="run scripts/build_waymo_protos.py first")
def test_waymo_reader_round_trip(tmp_path):
    from src.waymo_reader import _scenario_cls, iter_scenarios, map_frame, tracks_frame

    Scenario = _scenario_cls()
    sc = Scenario(scenario_id="abc", current_time_index=1, sdc_track_index=0,
                  timestamps_seconds=[0.0, 0.1, 0.2])
    t = sc.tracks.add(id=7, object_type=2)
    for i in range(3):
        t.states.add(center_x=float(i), center_y=0.0, velocity_x=1.0, valid=True)
    sc.tracks_to_predict.add(track_index=0)
    lane = sc.map_features.add(id=1).lane
    lane.polyline.add(x=0, y=0)
    lane.polyline.add(x=1, y=0)
    sc.map_features.add(id=2).stop_sign.position.x = 5.0

    shard = tmp_path / "validation.tfrecord-00000-of-00001"
    _write_tfrecord(shard, [sc.SerializeToString()] * 2)
    got = list(iter_scenarios(shard))
    assert len(got) == 2

    tr = tracks_frame(got[0])
    assert len(tr) == 3 and tr.object_type.iloc[0] == "pedestrian"
    assert tr.steps_from_current.tolist() == [-1, 0, 1] and tr.to_predict.all()
    mp = map_frame(got[0])
    assert set(mp.feature_type) == {"lane", "stop_sign"} and len(mp) == 3
