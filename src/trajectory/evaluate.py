"""Score the MLP against the constant-velocity baseline on validation_00000 (test only).

Same agents as ``src.trajectory.baseline``: ``to_predict``, valid now, at least one valid future
step. Per horizon H (3, 5, 8 s), on valid future steps within H:
- ``cv_*``: constant velocity (one path).
- ``best_*``: the model's highest-confidence path (single best guess).
- ``min_*``: minADE / minFDE over the model's K=6 paths (each minimised separately).
ADE = mean error over valid steps; FDE = error at the last valid step within H.

Distance-adjusted error (8 s rows only): FDE@8s / distance the agent actually travelled
(path length through its valid future points, from the current position), over agents that moved
at least 1 m. Reported as the mean and median of the per-agent ratio.

    python -m src.trajectory.evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.trajectory.dataset import (
    AGENT_TYPES,
    FUTURE_STEPS,
    PAST_STEPS,
    TEST_NPZ,
    TEST_TRACKS,
    load,
    to_world_frame,
)
from src.trajectory.model import CHECKPOINT, load_checkpoint, predict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "reports" / "waymo_model_vs_baseline.csv"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "waymo_model_examples.png"
HORIZONS_S = (3, 5, 8)
MIN_TRAVEL_M = 1.0


def horizon_errors(paths: np.ndarray, future: np.ndarray, valid: np.ndarray, horizon_s: int):
    """Per-agent, per-candidate ADE and FDE within the horizon.

    paths (N, K, 80, 2), future (N, 80, 2), valid (N, 80). Returns (ade, fde, has) with ade/fde
    of shape (N, K) and ``has`` (N,) marking agents with a valid step in the horizon.
    """
    steps = round(horizon_s * 10)
    m = valid[:, :steps]
    err = np.linalg.norm(paths[:, :, :steps] - future[:, None, :steps], axis=-1)
    n_valid = m.sum(1)
    has = n_valid > 0
    ade = (err * m[:, None]).sum(-1) / np.maximum(n_valid, 1)[:, None]
    last = steps - 1 - np.argmax(m[:, ::-1], axis=1)
    fde = np.take_along_axis(err, last[:, None, None], axis=2)[..., 0]
    return ade, fde, has


def distance_travelled(future: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Path length from the origin through each agent's valid future points (agent frame)."""
    out = np.zeros(len(future))
    for i in range(len(future)):
        pts = np.vstack([np.zeros((1, 2)), future[i][valid[i]]])
        out[i] = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return out


def per_agent_table(test: dict, paths: np.ndarray, conf: np.ndarray) -> pd.DataFrame:
    """Long table: one row per agent and horizon with cv/best/min ADE and FDE."""
    future, valid = test["future"].astype(np.float64), test["future_valid"]
    cv = constant_velocity_paths_np(test["past"])[:, None]
    best = np.take_along_axis(paths, conf.argmax(1)[:, None, None, None], axis=1)
    travelled = distance_travelled(future, valid)
    frames = []
    for h in HORIZONS_S:
        cv_ade, cv_fde, has = horizon_errors(cv, future, valid, h)
        b_ade, b_fde, _ = horizon_errors(best, future, valid, h)
        k_ade, k_fde, _ = horizon_errors(paths, future, valid, h)
        frames.append(pd.DataFrame({
            "scenario_id": test["scenario_id"], "track_id": test["track_id"],
            "object_type": np.asarray(AGENT_TYPES)[test["agent_type"]], "horizon_s": h,
            "cv_ade_m": cv_ade[:, 0], "cv_fde_m": cv_fde[:, 0],
            "best_ade_m": b_ade[:, 0], "best_fde_m": b_fde[:, 0],
            "min_ade_m": k_ade.min(1), "min_fde_m": k_fde.min(1),
            "travelled_m": travelled,
        })[has])
    return pd.concat(frames, ignore_index=True)


def constant_velocity_paths_np(past: np.ndarray) -> np.ndarray:
    t = np.arange(1, FUTURE_STEPS + 1) * 0.1
    return past[:, -1, None, 2:4].astype(np.float64) * t[None, :, None]


def summarize(agents: pd.DataFrame) -> pd.DataFrame:
    error_cols = ["cv_ade_m", "cv_fde_m", "best_ade_m", "best_fde_m", "min_ade_m", "min_fde_m"]
    rows = []
    groups = [("all", agents)] + [
        (t, agents[agents["object_type"] == t]) for t in AGENT_TYPES
        if (agents["object_type"] == t).any()
    ]
    for object_type, frame in groups:
        for h, at_h in frame.groupby("horizon_s"):
            row = {"object_type": object_type, "horizon_s": h, "n_agents": len(at_h)}
            row.update(at_h[error_cols].mean().to_dict())
            if h == 8:
                moving = at_h[at_h["travelled_m"] >= MIN_TRAVEL_M]
                row["n_moving"] = len(moving)
                for prefix in ("cv", "best", "min"):
                    ratio = moving[f"{prefix}_fde_m"] / moving["travelled_m"]
                    row[f"{prefix}_fde_per_m_mean"] = ratio.mean()
                    row[f"{prefix}_fde_per_m_median"] = ratio.median()
            rows.append(row)
    out = pd.DataFrame(rows)
    out["best_fde_vs_cv_pct"] = 100 * (1 - out["best_fde_m"] / out["cv_fde_m"])
    out["min_fde_vs_cv_pct"] = 100 * (1 - out["min_fde_m"] / out["cv_fde_m"])
    return out


def format_table(summary: pd.DataFrame) -> str:
    cols = ["object_type", "horizon_s", "n_agents", "cv_ade_m", "cv_fde_m", "best_ade_m",
            "best_fde_m", "min_ade_m", "min_fde_m", "best_fde_vs_cv_pct", "min_fde_vs_cv_pct"]
    main = summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}")
    adj_cols = ["object_type", "n_moving"] + [
        f"{p}_fde_per_m_{s}" for p in ("cv", "best", "min") for s in ("mean", "median")
    ]
    adj = summary[summary["horizon_s"] == 8][adj_cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"
    )
    return f"{main}\n\nDistance-adjusted FDE@8s (FDE / metres travelled, agents moving >= 1 m)\n{adj}"


def plot_examples(test: dict, paths: np.ndarray, conf: np.ndarray, out_path: Path) -> list:
    """Same three agents as the baseline figure, with the model's six paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.trajectory.baseline import (
        TRACK_COLUMNS,
        _pick_examples,
        predict_constant_velocity,
    )

    tracks = pd.read_parquet(TEST_TRACKS, columns=TRACK_COLUMNS)
    picks = _pick_examples(tracks, predict_constant_velocity(tracks))
    map_path = TEST_TRACKS.with_name(TEST_TRACKS.name.replace("_tracks", "_map"))
    lanes = pd.read_parquet(
        map_path, columns=["scenario_id", "feature_type", "x", "y"],
        filters=[("scenario_id", "in", sorted({p[0] for p in picks})),
                 ("feature_type", "==", "lane")],
    )
    index = {(s, int(t)): i for i, (s, t) in enumerate(zip(test["scenario_id"], test["track_id"]))}
    world_paths = to_world_frame(paths, test["origin"], test["yaw"])
    world_future = to_world_frame(test["future"].astype(np.float64), test["origin"], test["yaw"])
    world_past = to_world_frame(test["past"][..., :2].astype(np.float64), test["origin"], test["yaw"])
    world_cv = to_world_frame(constant_velocity_paths_np(test["past"]), test["origin"], test["yaw"])

    fig, axes = plt.subplots(1, len(picks), figsize=(5.5 * len(picks), 5.5), squeeze=False)
    for ax, (scenario_id, track_id, object_type) in zip(axes[0], picks, strict=True):
        i = index[(scenario_id, track_id)]
        past = world_past[i][test["past_valid"][i]]
        fut = world_future[i][test["future_valid"][i]]
        cand = world_paths[i]
        pts = np.vstack([past, fut, cand.reshape(-1, 2), world_cv[i]])
        span = np.ptp(pts, axis=0).max()
        margin = max(10.0, 0.15 * span)
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        near = lanes[(lanes["scenario_id"] == scenario_id) & lanes["x"].between(lo[0], hi[0])
                     & lanes["y"].between(lo[1], hi[1])]
        ax.scatter(near["x"], near["y"], s=1, color="0.85", label="lane points")
        ax.plot(past[:, 0], past[:, 1], color="0.25", lw=2, label="past (1 s)")
        ax.plot(fut[:, 0], fut[:, 1], color="tab:blue", lw=2.5, label="true future (8 s)")
        ax.plot(world_cv[i][:, 0], world_cv[i][:, 1], color="tab:orange", lw=1.5, ls="--",
                label="constant velocity")
        order = np.argsort(-conf[i])
        for rank, k in enumerate(order):
            ax.plot(cand[k][:, 0], cand[k][:, 1], color="tab:red", lw=0.8 + 3 * conf[i, k],
                    alpha=0.35 + 0.65 * conf[i, k] / conf[i].max(),
                    label="model paths (width = confidence)" if rank == 0 else None)
            ax.annotate(f"{conf[i, k]:.2f}", cand[k][-1], fontsize=7, color="tab:red")
        ax.scatter(*past[-1], color="black", zorder=5, s=20)
        valid = test["future_valid"][i]
        last = np.flatnonzero(valid)[-1]
        err_best = np.linalg.norm(cand[order[0]][last] - world_future[i][last])
        err_min = np.linalg.norm(cand[:, last] - world_future[i][last], axis=-1).min()
        err_cv = np.linalg.norm(world_cv[i][last] - world_future[i][last])
        ax.set_title(f"{object_type}  (scenario {scenario_id}, track {track_id})\n"
                     f"FDE@8s  CV {err_cv:.1f} m | best guess {err_best:.1f} m | "
                     f"min of 6 {err_min:.1f} m", fontsize=9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    axes[0][0].legend(loc="best", fontsize=7)
    fig.suptitle("Waymo validation_00000: MLP (K=6) vs constant velocity, same agents as baseline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", type=Path, default=TEST_NPZ)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    test = load(args.test)
    assert test["past"].shape[1] == PAST_STEPS
    model, _ = load_checkpoint(args.checkpoint)
    paths, conf = predict(model, test)
    paths = paths.astype(np.float64)
    summary = summarize(per_agent_table(test, paths, conf))
    print(format_table(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    if not args.no_plot:
        picks = plot_examples(test, paths, conf, args.figure)
        print(f"Wrote {args.figure} ({', '.join(p[2] for p in picks)})")


if __name__ == "__main__":
    main()
