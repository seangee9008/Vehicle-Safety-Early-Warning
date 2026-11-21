"""Pre-render the Waymo demo examples for the public app: app/static/trajectory/<example_id>.png and
examples.csv (plain description + errors, no coordinates). Coordinates never leave the dev machine.

    python -m app.render_trajectories      # needs data/processed/waymo/demo_examples.json
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app import motion, theme
from app.data import STATIC_TRAJECTORY
from src import bq_publish

DPI = 110
SIZE_IN = 7.6


def _xy(points) -> np.ndarray:
    pts = [p for p in points if p is not None]
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def render(example: dict, path) -> None:
    color = theme.ROAD_USER.get(example["object_type"], theme.TRACK_B)
    fig, ax = plt.subplots(figsize=(SIZE_IN, SIZE_IN), dpi=DPI)
    fig.patch.set_facecolor(theme.CARD)
    ax.set_facecolor(theme.CARD)
    for line in example["lane_polylines"]:
        pts = _xy(line)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], color=theme.LANE, lw=0.9, zorder=1)
    paths = example["model_paths"]
    for k in range(len(paths) - 1, 0, -1):
        pts = _xy(paths[k])
        ax.plot(pts[:, 0], pts[:, 1], color=theme.TRACK_B, lw=1.3, alpha=0.55, zorder=2,
                label="Other guesses" if k == 1 else None)
    pts = _xy(example["baseline_path"])
    ax.plot(pts[:, 0], pts[:, 1], color=theme.BASELINE, lw=2, ls=(0, (5, 4)), zorder=3, label="Going straight")
    pts = _xy(paths[0])
    ax.plot(pts[:, 0], pts[:, 1], color=theme.TRACK_B, lw=3.6, zorder=4, label="Top guess")
    pts = _xy(example["past_path"])
    ax.plot(pts[:, 0], pts[:, 1], color=theme.PAST, lw=3, zorder=5, label="Past second")
    pts = _xy(example["true_future"])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=4.2, zorder=6, label="What really happened")
    ax.scatter([0], [0], color=theme.INK, s=36, zorder=7)

    every = np.vstack([_xy(p) for p in [example["past_path"], example["true_future"], example["baseline_path"],
                                        *paths]])
    lo, hi = every.min(0), every.max(0)
    center, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2 * 1.2, 12.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_color(theme.BORDER)
    ax.tick_params(colors=theme.MUTED, labelsize=11)
    ax.set_xlabel("Meters ahead", color=theme.MUTED, fontsize=12)
    ax.set_ylabel("Meters to the left", color=theme.MUTED, fontsize=12)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(theme.MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=theme.CARD)
    plt.close(fig)


def main() -> None:
    path = bq_publish.WAYMO_DEMO
    if not path.exists():
        raise SystemExit(f"{path} missing: run python -m src.trajectory.demo_examples")
    examples = json.loads(path.read_text(encoding="utf-8"))
    STATIC_TRAJECTORY.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in examples:
        render(e, STATIC_TRAJECTORY / f"{e['example_id']}.png")
        last = [p for p in e["true_future"] if p is not None][-1]
        rows.append({"example_id": e["example_id"], "object_type": e["object_type"],
                     "description": motion.describe(e), "fde_8s_cv_m": e["fde_8s_cv_m"],
                     "fde_8s_best_m": e["fde_8s_best_m"], "fde_8s_min_m": e["fde_8s_min_m"],
                     "travelled_m": e["travelled_m"], "turns": abs(last[1]) >= 5})
    pd.DataFrame(rows).to_csv(STATIC_TRAJECTORY / "examples.csv", index=False)
    size = sum(f.stat().st_size for f in STATIC_TRAJECTORY.iterdir()) / 1e6
    print(f"wrote {len(rows)} images + examples.csv to {STATIC_TRAJECTORY} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
