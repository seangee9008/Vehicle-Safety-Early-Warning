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
