# Track A: recall risk model

Owner: Justine. Labels, features, baseline, LightGBM, rolling backtest, weeks of warning, watchlist.
Column names follow the hand-off contracts in AGENTS.md.

Every step takes `--source raw|dbt`. `raw` reads `raw.complaints` / `raw.recalls` with temporary
cleaning in `sources.py`; `dbt` reads `main.stg_complaints` / `main.stg_recalls` with the contract columns.

```bash
python -m src.recall_model.labels    --source raw   # data/processed/recall_labels_raw.parquet
python -m src.recall_model.features  --source raw   # monthly_features_raw.parquet + main.monthly_features
python -m src.recall_model.model     --source raw   # predictions, warning weeks, model, reports/recall_metrics_raw.csv
python -m src.recall_model.watchlist --source raw   # watchlist_raw.csv (agent input) + main.watchlist
```

- `monthly_features` starts with the contract columns (`cmpl_1m`, `cmpl_3m`, `cmpl_6m`, `severe_3m`,
  `vehicle_age`, `label` = recalled within 6 months), then extras (`cmpl_12m`, `growth_3m`,
  `share_of_car`, `crash_3m`, `fire_3m`, `injured_3m`, `deaths_3m`, `median_miles_12m`, `y_3m`, `y_12m`, ...).
- `watchlist_raw.csv` has exactly the 8 contract columns, so `python -m src.agent.run --watchlist` reads it.
  A month's row uses complaints received by the end of that month. `watchlist_raw_full.parquet` and
  `main.watchlist` also have each reason, `source` and `model_version`.
- Split: train from 2010, validate 2020-21, test 2022-24. For each test month the model is refit
  on rows whose 6-month label was already known (`month + 6 months <= test month`).
- Weeks of warning: recall date minus the end of the first month the cell was in the top 20,
  looking back up to 12 months.
- Temporary part map (fuel, engine, service brakes, visibility) lives in `sources.component_sql`;
  the dbt map replaces it at the swap.
