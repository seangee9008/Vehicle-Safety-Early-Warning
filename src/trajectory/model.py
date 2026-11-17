"""Small multi-modal MLP for 8 s trajectory prediction (agent frame).

Input: the 11 past steps (x, y, vx, vy, heading + validity) and the agent type. Output: K=6
candidate paths of 80 (x, y) points plus K confidence logits. Each path is a learned offset added
to the constant-velocity extrapolation of the current velocity, so an untrained model starts at the
baseline.

Loss: winner-takes-all. For each agent, the candidate with the lowest masked ADE (mean distance over
valid future steps) is the winner; regression loss is the winner's masked ADE, plus cross-entropy
pushing the confidence logits toward the winner.

Training uses only ``train_examples.npz`` (training shards). 10% of training *scenarios* are held
out for early stopping.

    python -m src.trajectory.model          # writes data/processed/waymo/mlp_k6.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.trajectory.dataset import (
    AGENT_TYPES,
    FUTURE_STEPS,
    PAST_STEPS,
    TRAIN_NPZ,
    WAYMO_DIR,
    load,
)

CHECKPOINT = WAYMO_DIR / "mlp_k6.pt"
K = 6
DT_S = 0.1
SEED = 0


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def features(past: np.ndarray, past_valid: np.ndarray, agent_type: np.ndarray) -> np.ndarray:
    """Flat input vector: past (11x5), validity (11), one-hot type (4)."""
    one_hot = np.eye(len(AGENT_TYPES), dtype=np.float32)[agent_type]
    return np.concatenate(
        [past.reshape(len(past), -1), past_valid.astype(np.float32), one_hot], axis=1
    ).astype(np.float32)


def constant_velocity_paths(past: torch.Tensor) -> torch.Tensor:
    """(B, 80, 2) path from the current velocity (already in the agent frame)."""
    t = torch.arange(1, FUTURE_STEPS + 1, device=past.device, dtype=past.dtype) * DT_S
    return past[:, -1, None, 2:4] * t[None, :, None]


class TrajectoryMLP(nn.Module):
    def __init__(self, in_dim: int, k: int = K, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.paths = nn.Linear(hidden, k * FUTURE_STEPS * 2)
        self.logits = nn.Linear(hidden, k)
        nn.init.zeros_(self.paths.weight)
        nn.init.zeros_(self.paths.bias)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body((x - self.mean) / self.std)
        offsets = self.paths(h).view(-1, self.k, FUTURE_STEPS, 2)
        return constant_velocity_paths(past)[:, None] + offsets, self.logits(h)


def wta_loss(
    paths: torch.Tensor, logits: torch.Tensor, future: torch.Tensor, future_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Winner-takes-all regression (masked ADE of closest candidate) + cross-entropy on the winner.

    paths (B, K, T, 2), logits (B, K), future (B, T, 2), future_valid (B, T) bool.
    Returns (regression, classification), each averaged over the batch.
    """
    mask = future_valid.to(paths.dtype)
    dist = torch.linalg.vector_norm(paths - future[:, None], dim=-1)  # (B, K, T)
    ade = (dist * mask[:, None]).sum(-1) / mask.sum(-1).clamp(min=1.0)[:, None]
    winner = ade.argmin(dim=1)
    regression = ade.gather(1, winner[:, None]).squeeze(1).mean()
    classification = nn.functional.cross_entropy(logits, winner)
    return regression, classification


def scenario_holdout(scenario_id: np.ndarray, fraction: float = 0.1, seed: int = SEED) -> np.ndarray:
    """Boolean mask of examples whose scenario falls in the held-out ``fraction``."""
    scenarios = np.unique(scenario_id)
    rng = np.random.default_rng(seed)
    held = rng.choice(scenarios, size=max(1, round(fraction * len(scenarios))), replace=False)
    return np.isin(scenario_id, held)


def _tensors(data: dict[str, np.ndarray], idx: np.ndarray, device: torch.device):
    x = features(data["past"][idx], data["past_valid"][idx], data["agent_type"][idx])
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(data["past"][idx]).to(device),
        torch.from_numpy(data["future"][idx]).to(device),
        torch.from_numpy(data["future_valid"][idx]).to(device),
    )


@torch.no_grad()
def predict(model: TrajectoryMLP, data: dict[str, np.ndarray], batch: int = 4096):
    """Agent-frame paths (N, K, 80, 2) and softmax confidences (N, K) as numpy."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(features(data["past"], data["past_valid"], data["agent_type"]))
    past = torch.from_numpy(data["past"])
    paths, probs = [], []
    for start in range(0, len(x), batch):
        p, logit = model(x[start:start + batch].to(device), past[start:start + batch].to(device))
        paths.append(p.cpu().numpy())
        probs.append(torch.softmax(logit, dim=-1).cpu().numpy())
    return np.concatenate(paths), np.concatenate(probs)


def train(
    data: dict[str, np.ndarray],
    *,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 25,
    max_minutes: float = 18.0,
    device: torch.device | None = None,
    log=print,
) -> tuple[TrajectoryMLP, dict]:
    torch.manual_seed(SEED)
    device = device or pick_device()
    held = scenario_holdout(data["scenario_id"])
    train_idx, val_idx = np.flatnonzero(~held), np.flatnonzero(held)
    tr = _tensors(data, train_idx, device)
    va = _tensors(data, val_idx, device)

    model = TrajectoryMLP(tr[0].shape[1]).to(device)
    model.mean.copy_(tr[0].mean(0))
    model.std.copy_(tr[0].std(0).clamp(min=1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log(f"device={device}  train agents={len(train_idx)}  holdout agents={len(val_idx)} "
        f"({len(np.unique(data['scenario_id'][val_idx]))} scenarios)")

    best, best_state, best_epoch, start = float("inf"), None, -1, time.time()
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx), generator=generator).to(device)
        total = 0.0
        for b in range(0, len(order), batch_size):
            sel = order[b:b + batch_size]
            paths, logits = model(tr[0][sel], tr[1][sel])
            reg, cls = wta_loss(paths, logits, tr[2][sel], tr[3][sel])
            loss = reg + cls
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
        sched.step()

        model.eval()
        with torch.no_grad():
            paths, logits = model(va[0], va[1])
            reg, cls = wta_loss(paths, logits, va[2], va[3])
        val = reg.item() + cls.item()
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == best_epoch:
            log(f"epoch {epoch:3d}  train {total / len(order):.3f}  holdout minADE@8s "
                f"{reg.item():.3f}  ce {cls.item():.3f}  ({time.time() - start:.0f}s)")
        if epoch - best_epoch >= patience or (time.time() - start) / 60 > max_minutes:
            break

    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": epoch + 1, "holdout_loss": best,
            "seconds": time.time() - start, "train_agents": len(train_idx),
            "holdout_agents": len(val_idx), "device": str(device)}
    return model, info


def save_checkpoint(model: TrajectoryMLP, info: dict, path: Path = CHECKPOINT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": model.mean.numel(), "info": info}, path)


def load_checkpoint(path: Path = CHECKPOINT, device: torch.device | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TrajectoryMLP(ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or torch.device("cpu")), ckpt["info"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=TRAIN_NPZ)
    parser.add_argument("--out", type=Path, default=CHECKPOINT)
    parser.add_argument("--max-minutes", type=float, default=18.0)
    args = parser.parse_args()

    data = load(args.data)
    assert PAST_STEPS == data["past"].shape[1]
    model, info = train(data, max_minutes=args.max_minutes)
    save_checkpoint(model, info, args.out)
    print(f"best epoch {info['best_epoch']} of {info['epochs_run']}, "
          f"{info['seconds'] / 60:.1f} min on {info['device']}. Wrote {args.out}")


if __name__ == "__main__":
    main()
