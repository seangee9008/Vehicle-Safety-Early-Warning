"""Compare model versions; select on held-out *training* scenarios, report on the test set.

Selection split: prediction targets in the 10% of training scenarios held out from fitting (same
agents for every version). Rule, fixed before looking at results: lowest best-guess FDE@8s there,
ties (within 1%) broken by minFDE@8s. The test set (validation_00000) is reported, never used to
choose.

    python -m src.trajectory.compare     # writes reports/waymo_v1_vs_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.trajectory.dataset import TEST_NPZ, TRAIN_NPZ, WAYMO_DIR, load
from src.trajectory.evaluate import per_agent_table, summarize
from src.trajectory.model import holdout_index, load_checkpoint, predict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "reports" / "waymo_v1_vs_v2.csv"
MODELS = {
    "v1": WAYMO_DIR / "mlp_k6.pt",
    "v2_all": WAYMO_DIR / "mlp_k6_v2_all.pt",
    "v2_moving": WAYMO_DIR / "mlp_k6_v2_moving.pt",
}


def subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {name: arr[idx] for name, arr in data.items()}


def winner_nll(data: dict[str, np.ndarray], paths: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """-log confidence of the candidate with the lowest masked ADE over 8 s, per agent."""
    valid = data["future_valid"]
    err = np.linalg.norm(paths - data["future"][:, None].astype(np.float64), axis=-1)
    ade = (err * valid[:, None]).sum(-1) / np.maximum(valid.sum(1), 1)[:, None]
    winner = ade.argmin(1)
    return -np.log(np.clip(conf[np.arange(len(conf)), winner], 1e-12, None))


def score(data: dict[str, np.ndarray], model, temperature: float) -> pd.DataFrame:
    paths, conf = predict(model, data, temperature=temperature)
    paths = paths.astype(np.float64)
    summary = summarize(per_agent_table(data, paths, conf))
    nll = pd.Series(winner_nll(data, paths, conf))
    types = pd.Series(np.asarray(["vehicle", "pedestrian", "cyclist", "other"])[data["agent_type"]])
    nll_by = {"all": nll.mean(), **nll.groupby(types).mean().to_dict()}
    summary["winner_nll"] = summary["object_type"].map(nll_by)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    train = load(TRAIN_NPZ)
    holdout = subset(train, holdout_index(train))
    test = load(TEST_NPZ)
    frames = []
    for name, path in MODELS.items():
        model, info = load_checkpoint(path)
        temperature = info.get("temperature", 1.0)
        for split, data in (("holdout", holdout), ("test", test)):
            frames.append(score(data, model, temperature).assign(
                model=name, split=split, temperature=temperature,
                train_agents=info["train_agents"]))
    table = pd.concat(frames, ignore_index=True)
    cols = ["split", "model", "object_type", "horizon_s", "n_agents", "best_ade_m", "best_fde_m",
            "min_ade_m", "min_fde_m", "cv_fde_m", "best_fde_vs_cv_pct", "min_fde_vs_cv_pct",
            "winner_nll", "temperature", "train_agents"]
    table = table[cols]

    sel = table[(table["split"] == "holdout") & (table["object_type"] == "all")
                & (table["horizon_s"] == 8)].set_index("model")
    best = sel["best_fde_m"].min()
    tied = sel[sel["best_fde_m"] <= best * 1.01]
    chosen = tied["min_fde_m"].idxmin()
    table["selected"] = table["model"] == chosen

    show = table[table["horizon_s"] == 8].drop(columns=["horizon_s"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSelected on holdout (best-guess FDE@8s, then minFDE@8s): {chosen} "
          f"-> {MODELS[chosen].name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
