"""Waymo model tests: no network, no data files needed (the data check skips if absent)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.trajectory import dataset
from src.trajectory.dataset import build_examples, to_agent_frame, to_world_frame
from src.trajectory.evaluate import horizon_errors
from src.trajectory.model import K, TrajectoryMLP, features, scenario_holdout, wta_loss


def _tracks(scenario_id="s1", track_id=1, heading=0.7, speed=5.0, object_type="vehicle"):
    rows = []
    for step in range(-10, 81):
        t = step * 0.1
        vx, vy = speed * np.cos(heading), speed * np.sin(heading)
        rows.append({
            "scenario_id": scenario_id, "track_id": track_id, "object_type": object_type,
            "to_predict": True, "steps_from_current": step, "valid": True,
            "x": 100.0 + vx * t, "y": -40.0 + vy * t, "heading": heading, "vx": vx, "vy": vy,
        })
    return rows


def test_agent_frame_round_trip():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(5, 20, 2)) * 50
    origin = rng.normal(size=(5, 2)) * 1000
    yaw = rng.uniform(-np.pi, np.pi, size=5)
    local = to_agent_frame(xy, origin, yaw)
    assert np.allclose(to_world_frame(local, origin, yaw), xy)
    # Candidate axis (N, K, T, 2) broadcasts too.
    stacked = np.repeat(local[:, None], 3, axis=1)
    assert np.allclose(to_world_frame(stacked, origin, yaw), np.repeat(xy[:, None], 3, axis=1))


def test_build_examples_puts_agent_at_origin_heading_along_x():
    ex = build_examples(pd.DataFrame(_tracks(heading=0.7, speed=5.0)))
    assert ex["past"].shape == (1, 11, 5) and ex["future"].shape == (1, 80, 2)
    assert np.allclose(ex["past"][0, -1], [0, 0, 5.0, 0, 0], atol=1e-5)
    # Straight motion along the heading: future is on +x, y ~ 0, 8 s * 5 m/s = 40 m.
    assert np.allclose(ex["future"][0, -1], [40.0, 0.0], atol=1e-4)
    assert np.allclose(ex["past"][0, 0, :2], [-5.0, 0.0], atol=1e-4)
    world = to_world_frame(ex["future"].astype(float), ex["origin"], ex["yaw"])
    assert np.allclose(world[0, -1], [100 + 40 * np.cos(0.7), -40 + 40 * np.sin(0.7)], atol=1e-3)


def test_loss_is_near_zero_when_one_candidate_matches_truth():
    torch.manual_seed(0)
    future = torch.randn(4, 80, 2) * 10
    valid = torch.ones(4, 80, dtype=torch.bool)
    valid[0, 40:] = False
    paths = torch.randn(4, K, 80, 2) * 10
    paths[:, 2] = future
    logits = torch.full((4, K), -20.0)
    logits[:, 2] = 20.0
    reg, cls = wta_loss(paths, logits, future, valid)
    assert reg.item() == pytest.approx(0.0, abs=1e-6)
    assert cls.item() < 1e-6
    # Masked steps don't count: garbage after the last valid step leaves the loss at zero.
    paths[0, 2, 40:] += 1000.0
    reg, _ = wta_loss(paths, logits, future, valid)
    assert reg.item() == pytest.approx(0.0, abs=1e-6)


def test_untrained_model_starts_at_constant_velocity():
    ex = build_examples(pd.DataFrame(_tracks()))
    x = torch.from_numpy(features(ex["past"], ex["past_valid"], ex["agent_type"]))
    model = TrajectoryMLP(x.shape[1]).eval()
    paths, _ = model(x, torch.from_numpy(ex["past"]))
    assert torch.allclose(paths[0, 0], torch.from_numpy(ex["future"][0]), atol=1e-3)


def test_horizon_errors_use_last_valid_step():
    future = np.zeros((1, 80, 2))
    valid = np.zeros((1, 80), bool)
    valid[0, :25] = True
    paths = np.zeros((1, 2, 80, 2))
    paths[0, 1, 24] = [3.0, 0.0]
    ade, fde, has = horizon_errors(paths, future, valid, 8)
    assert has[0] and fde[0, 0] == 0 and fde[0, 1] == pytest.approx(3.0)
    assert ade[0, 1] == pytest.approx(3.0 / 25)


def test_holdout_splits_by_scenario():
    ids = np.repeat([f"s{i}" for i in range(50)], 3)
    held = scenario_holdout(ids)
    assert held.sum() == 15  # 5 of 50 scenarios, 3 agents each
    assert not set(ids[held]) & set(ids[~held])


def test_training_files_exclude_validation(tmp_path):
    for name in ("training_00000", "training_00001", "validation_00000"):
        (tmp_path / f"{name}_tracks.parquet").touch()
    files = dataset.training_track_files(tmp_path)
    assert [f.name for f in files] == ["training_00000_tracks.parquet",
                                       "training_00001_tracks.parquet"]


@pytest.mark.skipif(not (dataset.TRAIN_NPZ.exists() and dataset.TEST_NPZ.exists()),
                    reason="processed Waymo examples not on this machine")
def test_validation_scenarios_never_in_training_data():
    train = np.load(dataset.TRAIN_NPZ)["scenario_id"]
    test = np.load(dataset.TEST_NPZ)["scenario_id"]
    assert not set(train) & set(test)


def test_non_targets_need_full_future_and_test_mode_skips_them():
    target = _tracks(track_id=1)
    full = [dict(r, track_id=2, to_predict=False) for r in _tracks()]
    partial = [dict(r, track_id=3, to_predict=False, valid=r["steps_from_current"] <= 40)
               for r in _tracks()]
    frame = pd.DataFrame(target + full + partial)
    assert list(build_examples(frame)["track_id"]) == [1]
    ex = build_examples(frame, targets_only=False)
    assert list(ex["track_id"]) == [1, 2]
    assert list(ex["is_target"]) == [True, False]


def test_temperature_keeps_ranking_and_label_smoothing_raises_floor():
    from src.trajectory.model import fit_temperature, predict

    rows = [r for i in range(4) for r in _tracks(scenario_id=f"s{i}", heading=0.1 * i)]
    ex = build_examples(pd.DataFrame(rows))
    model = TrajectoryMLP(features(ex["past"], ex["past_valid"], ex["agent_type"]).shape[1])
    torch.nn.init.normal_(model.logits.weight)
    t = fit_temperature(model, ex, np.arange(len(ex["past"])))
    assert 0.25 <= t <= 8.0
    _, p1 = predict(model, ex)
    _, p2 = predict(model, ex, temperature=t)
    assert (p1.argmax(1) == p2.argmax(1)).all()

    future = torch.zeros(2, 80, 2)
    valid = torch.ones(2, 80, dtype=torch.bool)
    paths = torch.ones(2, K, 80, 2)
    paths[:, 0] = 0.0
    logits = torch.full((2, K), -20.0)
    logits[:, 0] = 20.0
    _, sharp = wta_loss(paths, logits, future, valid)
    _, smooth = wta_loss(paths, logits, future, valid, label_smoothing=0.1)
    assert sharp.item() < 1e-6 < smooth.item()
