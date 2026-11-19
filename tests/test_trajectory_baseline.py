import numpy as np
import pandas as pd
import pytest

from src.trajectory.baseline import per_agent_metrics, predict_constant_velocity


def _agent(track_id, *, vx=2.0, vy=1.0, object_type="vehicle", valid=None, to_predict=True,
           offsets=None):
    """Straight-line track for steps -10..80; offsets[step] = (dx, dy) adds error."""
    rows = []
    for step in range(-10, 81):
        t = step * 0.1
        dx, dy = (offsets or {}).get(step, (0.0, 0.0))
        rows.append({
            "scenario_id": "s1", "track_id": track_id, "object_type": object_type,
            "to_predict": to_predict, "steps_from_current": step,
            "valid": True if valid is None else valid(step),
            "x": 10.0 + vx * t + dx, "y": -5.0 + vy * t + dy, "vx": vx, "vy": vy,
        })
    return rows


def _metrics(rows):
    scored = predict_constant_velocity(pd.DataFrame(rows))
    return scored, per_agent_metrics(scored).set_index(["track_id", "horizon_s"])


def test_straight_line_agent_has_near_zero_error():
    scored, metrics = _metrics(_agent(1))

    assert len(scored) == 80
    assert np.allclose(metrics[["ade_m", "fde_m"]], 0.0, atol=1e-9)
    assert list(metrics.loc[1].index) == [3, 5, 8]


def test_fde_uses_last_valid_step_within_horizon():
    # Valid future only up to step 25 (2.5 s), with a 3 m error at step 25 and 1 m at step 20.
    rows = _agent(
        1,
        valid=lambda step: step <= 25,
        offsets={20: (1.0, 0.0), 25: (0.0, 3.0)},
    )
    _, metrics = _metrics(rows)

    assert metrics.loc[(1, 3), "fde_m"] == pytest.approx(3.0)
    assert metrics.loc[(1, 8), "fde_m"] == pytest.approx(3.0)
    assert metrics.loc[(1, 3), "ade_m"] == pytest.approx(4.0 / 25)


def test_agents_without_valid_future_or_current_are_skipped():
    rows = (
        _agent(1)
        + _agent(2, valid=lambda step: step <= 0)          # no valid future
        + _agent(3, valid=lambda step: step != 0)          # not valid now
        + _agent(4, to_predict=False)                      # not a prediction target
    )
    scored, metrics = _metrics(rows)

    assert set(scored["track_id"]) == {1}
    assert set(metrics.index.get_level_values("track_id")) == {1}
