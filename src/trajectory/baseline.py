"""Constant-velocity trajectory baseline on Waymo Motion tracks.

Each example is an agent with ``to_predict`` that is valid at the current step
(``steps_from_current == 0``). Its future position at step k (0.1 s per step) is
predicted as ``current (x, y) + (vx, vy) * 0.1 * k``. Errors are scored only on
valid future steps.

Metrics per horizon H (3, 5, 8 s), averaged over agents:
- ADE: mean displacement over the agent's valid future steps 1..10H.
- FDE: displacement at the agent's last valid step within 1..10H.
An agent counts toward a horizon only if it has at least one valid step in it.

    python -m src.trajectory.baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKS = ROOT / "data" / "processed" / "waymo" / "validation_00000_tracks.parquet"
DEFAULT_CSV = ROOT / "reports" / "waymo_baseline.csv"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "waymo_baseline_examples.png"

DT_S = 0.1
PAST_STEPS = 10
FUTURE_STEPS = 80
HORIZONS_S = (3, 5, 8)
KEY = ["scenario_id", "track_id"]
TRACK_COLUMNS = [
    "scenario_id", "track_id", "object_type", "to_predict",
    "steps_from_current", "valid", "x", "y", "vx", "vy",
]


def predict_constant_velocity(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return one row per scored future step with true and predicted positions.

    Columns: scenario_id, track_id, object_type, step, x, y, pred_x, pred_y, error.
    Agents with no valid future step are dropped.
    """
    current = tracks[
        tracks["to_predict"].astype(bool)
        & (tracks["steps_from_current"] == 0)
        & tracks["valid"].astype(bool)
    ][[*KEY, "object_type", "x", "y", "vx", "vy"]].rename(
        columns={"x": "x0", "y": "y0"}
    )
    future = tracks[
        tracks["steps_from_current"].between(1, FUTURE_STEPS)
        & tracks["valid"].astype(bool)
    ][[*KEY, "steps_from_current", "x", "y"]].rename(columns={"steps_from_current": "step"})
    scored = current.merge(future, on=KEY, how="inner")
    t = scored["step"] * DT_S
    scored["pred_x"] = scored["x0"] + scored["vx"] * t
    scored["pred_y"] = scored["y0"] + scored["vy"] * t
    scored["error"] = np.hypot(scored["pred_x"] - scored["x"], scored["pred_y"] - scored["y"])
    return scored[
        [*KEY, "object_type", "step", "x", "y", "pred_x", "pred_y", "error"]
    ].sort_values([*KEY, "step"], ignore_index=True)


def per_agent_metrics(scored: pd.DataFrame, horizons_s=HORIZONS_S) -> pd.DataFrame:
    """ADE/FDE per agent and horizon (long format)."""
    frames = []
    for horizon in horizons_s:
        within = scored[scored["step"] <= round(horizon / DT_S)]
        if within.empty:
            continue
        grouped = within.groupby(KEY, sort=False)
        last = within.loc[grouped["step"].idxmax()].set_index(KEY)
        agent = pd.DataFrame(
            {
                "object_type": last["object_type"],
                "ade_m": grouped["error"].mean(),
                "fde_m": last["error"],
            }
        ).reset_index()
        agent["horizon_s"] = horizon
        frames.append(agent)
    return pd.concat(frames, ignore_index=True)


def summarize(agent_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ADE/FDE overall and by object_type, with agent counts."""
    def _agg(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return frame.groupby(by).agg(
            n_agents=("ade_m", "size"), ade_m=("ade_m", "mean"), fde_m=("fde_m", "mean")
        ).reset_index()

    overall = _agg(agent_metrics, ["horizon_s"]).assign(object_type="all")
    by_type = _agg(agent_metrics, ["object_type", "horizon_s"])
    summary = pd.concat([overall, by_type], ignore_index=True)
    order = {"all": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3}
    summary["_order"] = summary["object_type"].map(order).fillna(9)
    summary = summary.sort_values(["_order", "object_type", "horizon_s"]).drop(columns="_order")
    return summary[["object_type", "horizon_s", "n_agents", "ade_m", "fde_m"]].reset_index(
        drop=True
    )


def format_table(summary: pd.DataFrame) -> str:
    wide = summary.pivot(index="object_type", columns="horizon_s")
    rows = []
    header = f"{'object_type':<12}" + "".join(
        f"{f'n@{h}s':>7}{f'ADE@{h}s':>9}{f'FDE@{h}s':>9}" for h in HORIZONS_S
    )
    rows.append(header)
    for object_type in summary["object_type"].drop_duplicates():
        cells = f"{object_type:<12}"
        for h in HORIZONS_S:
            n = wide.loc[object_type, ("n_agents", h)]
            cells += f"{int(n):>7}{wide.loc[object_type, ('ade_m', h)]:>9.2f}" \
                f"{wide.loc[object_type, ('fde_m', h)]:>9.2f}"
        rows.append(cells)
    return "\n".join(rows)


def _pick_examples(tracks: pd.DataFrame, scored: pd.DataFrame) -> list[tuple[str, int, str]]:
    """One moving agent per type with a full 8 s valid future, chosen deterministically."""
    full = scored.groupby(KEY).agg(object_type=("object_type", "first"), n=("step", "size"))
    full = full[full["n"] == FUTURE_STEPS].reset_index()
    current = tracks[tracks["steps_from_current"] == 0][[*KEY, "vx", "vy"]]
    full = full.merge(current, on=KEY)
    full["speed"] = np.hypot(full["vx"], full["vy"])
    min_speed = {"vehicle": 5.0, "pedestrian": 0.8, "cyclist": 2.0}
    picks = []
    for object_type in ("vehicle", "pedestrian", "cyclist"):
        candidates = full[
            (full["object_type"] == object_type)
            & (full["speed"] >= min_speed[object_type])
        ].sort_values(KEY)
        if candidates.empty:
            candidates = full[full["object_type"] == object_type].sort_values(KEY)
        if not candidates.empty:
            row = candidates.iloc[len(candidates) // 2]
            picks.append((row["scenario_id"], int(row["track_id"]), object_type))
    return picks


def plot_examples(
    tracks: pd.DataFrame, scored: pd.DataFrame, map_path: Path, out_path: Path
) -> list[tuple[str, int, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = _pick_examples(tracks, scored)
    lanes = pd.read_parquet(
        map_path,
        columns=["scenario_id", "feature_id", "feature_type", "point_idx", "x", "y"],
        filters=[
            ("scenario_id", "in", sorted({p[0] for p in picks})),
            ("feature_type", "==", "lane"),
        ],
    )
    fig, axes = plt.subplots(1, len(picks), figsize=(5.5 * len(picks), 5.5), squeeze=False)
    for ax, (scenario_id, track_id, object_type) in zip(axes[0], picks, strict=True):
        agent = tracks[(tracks["scenario_id"] == scenario_id) & (tracks["track_id"] == track_id)]
        past = agent[
            agent["steps_from_current"].between(-PAST_STEPS, 0) & agent["valid"]
        ].sort_values("steps_from_current")
        future = scored[(scored["scenario_id"] == scenario_id) & (scored["track_id"] == track_id)]
        points = pd.concat([past[["x", "y"]], future[["x", "y"]],
                            future[["pred_x", "pred_y"]].set_axis(["x", "y"], axis=1)])
        span = max(points["x"].max() - points["x"].min(), points["y"].max() - points["y"].min())
        margin = max(10.0, 0.25 * span)
        x_lo, x_hi = points["x"].min() - margin, points["x"].max() + margin
        y_lo, y_hi = points["y"].min() - margin, points["y"].max() + margin
        nearby = lanes[
            (lanes["scenario_id"] == scenario_id)
            & lanes["x"].between(x_lo, x_hi)
            & lanes["y"].between(y_lo, y_hi)
        ]
        ax.scatter(nearby["x"], nearby["y"], s=1, color="0.8", label="lane points")
        ax.plot(past["x"], past["y"], color="0.25", lw=2, label="past (1 s)")
        ax.plot(future["x"], future["y"], color="tab:blue", lw=2, label="true future (8 s)")
        ax.plot(future["pred_x"], future["pred_y"], color="tab:orange", lw=2, ls="--",
                label="constant velocity")
        ax.scatter(past["x"].iloc[-1], past["y"].iloc[-1], color="black", zorder=5, s=20)
        fde = future["error"].iloc[-1]
        ax.set_title(f"{object_type}  (scenario {scenario_id}, track {track_id})\n"
                     f"FDE@8s = {fde:.1f} m", fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("Waymo validation_00000: constant-velocity baseline examples")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tracks = pd.read_parquet(args.tracks, columns=TRACK_COLUMNS)
    scored = predict_constant_velocity(tracks)
    summary = summarize(per_agent_metrics(scored))
    print(format_table(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    if not args.no_plot:
        map_path = args.tracks.with_name(args.tracks.name.replace("_tracks", "_map"))
        picks = plot_examples(tracks, scored, map_path, args.figure)
        print(f"Wrote {args.figure} ({', '.join(p[2] for p in picks)})")


if __name__ == "__main__":
    main()
