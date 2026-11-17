# Track B: trajectory prediction

Owner: Sean (lead), Angelina from week 2. Constant-velocity baseline, PyTorch model, minADE/minFDE.
Reads `data/processed/waymo/*_tracks.parquet` (see `src/waymo_reader.py`).

## Constant-velocity baseline

```bash
python -m src.trajectory.baseline   # writes reports/waymo_baseline.csv and reports/figures/waymo_baseline_examples.png
```

Predicts `current (x, y) + (vx, vy) * t` for 0.1–8.0 s for every `to_predict` agent valid at the
current step. ADE/FDE at 3/5/8 s are scored on valid future steps only; FDE uses the last valid
step within each horizon. Agents with no valid future step are skipped.

## MLP model (K=6 paths)

```bash
python scripts/waymo_to_parquet.py --split training   # training shards -> parquet
python -m src.trajectory.dataset                       # agent-frame examples -> data/processed/waymo/*.npz
python -m src.trajectory.model                         # train (MPS if available), early stop on 10% of training scenarios
python -m src.trajectory.evaluate                      # reports/waymo_model_vs_baseline.csv + figures/waymo_model_examples.png
```

- `dataset.py`: one example per `to_predict` agent valid now with a valid future step. 11 past steps
  (x, y, vx, vy, heading) and 80 future (x, y) + validity, in the agent's frame (current position at
  the origin, heading along +x), plus agent type. Training examples come only from `training_*`
  shards; `validation_00000` is test only (checked when building and in the tests).
- `model.py`: 3-layer MLP. Outputs 6 paths as offsets on top of the constant-velocity path, plus 6
  confidence logits. Loss: winner-takes-all masked ADE on the closest path + cross-entropy on which
  path won. 10% of training scenarios are held out for early stopping.
- `evaluate.py`: same agents and ADE/FDE rules as the baseline. Reports minADE/minFDE over the 6
  paths, the single best guess (highest confidence), and the distance-adjusted FDE@8s
  (FDE / metres actually travelled, agents moving at least 1 m).
