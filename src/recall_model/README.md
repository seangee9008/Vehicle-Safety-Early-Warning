# Track A: recall risk model

Owner: Justine. Labels, features, baseline, LightGBM, rolling backtest, weeks of warning, watchlist.

Every step takes `--source raw|stg`. `stg` reads Angelina's cleaned `stg.complaints` / `stg.recalls`
(build them with `python scripts/nhtsa_data_table_cleaning.py`). `raw` reads `raw.complaints` /
`raw.recalls` with temporary cleaning in `sources.py`. Both return the same columns, named like the
stg tables (`cmpl_id`, `odi_no`, `ldate`, `injured_count`, `deaths_count`, `camp_no`, ...). With
`--source stg`, each step first checks the stg tables have every column it reads.

```bash
python scripts/nhtsa_data_table_cleaning.py         # stg.complaints, stg.recalls (~30 s)
python -m src.recall_model.labels    --source stg   # data/processed/recall_labels_stg.parquet
python -m src.recall_model.features  --source stg   # monthly_features_stg.parquet + main.monthly_features
python -m src.recall_model.tune      --source stg   # optional: reports/recall_tuning_stg.csv (validation years only)
python -m src.recall_model.model     --source stg   # predictions, SHAP, warning weeks, model, reports/recall_metrics_stg.csv
python -m src.recall_model.watchlist --source stg   # watchlist_stg.csv (agent input) + main.watchlist
python -m src.recall_model.figures   --source stg   # reports/recall_warning_weeks_stg.png, recall_shap_summary_stg.png
```

- `monthly_features` starts with the AGENTS.md columns (`cmpl_1m`, `cmpl_3m`, `cmpl_6m`, `severe_3m`,
  `vehicle_age`, `label` = recalled within 6 months), then extras (`cmpl_12m`, `growth_3m`,
  `share_of_car`, `crash_3m`, `fire_3m`, `injured_3m`, `deaths_3m`, `median_miles_12m`,
  `months_since_recall`, keyword counts `kw_*_3m`, `y_3m`, `y_12m`, ...).
- Model v1 uses the count, severity, vehicle age, mileage and keyword features, with LightGBM settings
  picked on 2020-21 by `tune.py`.
- First recalls only: a car + part's complaints stop counting at its first recall (leakage rule 3), so
  months after it have nothing for the model to see. `model.py` and `tune.py` leave those rows out of
  training and ranking. `--all-rows` keeps them in and writes `*_all_rows` files; `months_since_recall`
  is in the table for that comparison.
- `watchlist_<source>.csv` has exactly the 8 watchlist columns, so `python -m src.agent.run --watchlist`
  reads it. A month's row uses complaints received by the end of that month. `top_reasons` are the
  features that pushed that row's score up the most (SHAP, from the model that scored it), written as
  plain facts. `watchlist_<source>_full.parquet` and `main.watchlist` also have each reason, `source`
  and `model_version`.
- `main.monthly_features` and `main.watchlist` hold whichever source ran last, so run `stg` last.
- Split: train from 2010, validate 2020-21, test 2022-24. For each test month the model is refit
  on rows whose 6-month label was already known (`month + 6 months <= test month`).
- Weeks of warning: recall date minus the end of the first month the cell was in the top 20,
  looking back up to 12 months and only after the cell's previous recall.
- The raw path's temporary part map lives in `sources.component_sql`; the stg tables use Angelina's map.
