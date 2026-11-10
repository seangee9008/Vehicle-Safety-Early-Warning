# STATUS

Rules, ownership and contracts are in [AGENTS.md](AGENTS.md). Update this file in the same branch
as the work. Status is one of: Not started / In progress / In review / Done.

Last updated: Nov 9.

| Task | Owner | Status | Branch or PR | Due |
| --- | --- | --- | --- | --- |
| Repo setup + NHTSA/Waymo data | Sean | Done | main | Nov 6 |
| BigQuery sandbox connected | Sean | Done | main | Nov 6 |
| Raw data summary (profiling notebook + interim cleaning script) | Angelina | Done | PR #2 | Nov 12 |
| Agent v0 + validator + cache + model fallback | Sean | In review | PR #1 (sean/agent-v0) | Nov 12 |
| Waymo constant-velocity baseline (go/no-go PASSED: FDE@8s 25.85 m overall, 28.72 vehicle, 2.98 pedestrian, 10.24 cyclist) | Sean | In review | sean/waymo-baseline | Nov 12 |
| Project hub (AGENTS.md, STATUS.md, run.py) | Sean | In review | sean/project-hub | Nov 12 |
| Contracts + source switches (after PR #1 merges) | Sean | Not started | | Nov 12 |
| CI | Sean | Not started | | Nov 12 |
| dbt `stg_complaints` + `stg_recalls` + tests | Angelina | Not started | | Nov 12 |
| EDA notebook + label/split proposal + complaint-count baseline | Justine | Not started | | Nov 12 |
| Dagster pipeline | Sean | Not started | | Nov 19 |
| Streamlit demo skeleton | Sean | Not started | | Nov 19 |
| Waymo: more shards + distance-adjusted error | Sean | Not started | | Nov 19 |
| Agent on the real watchlist | Sean | Not started | | Nov 18 |
| dbt intermediate models + snapshots | Angelina | Not started | | Nov 19 |
| `monthly_features` mart + LightGBM + backtest | Justine | Not started | | Nov 19 |
| Waymo PyTorch model | Sean | Not started | | Nov 24 |
| Final demo | Sean | Not started | | Nov 24 |
| Data quality tests + publish final tables to BigQuery | Angelina | Not started | | Nov 24 |
| SHAP + final metrics | Justine | Not started | | Nov 24 |
| Write-up sections + presentation | Everyone | Not started | | Nov 24 |

## Decisions

- Unit of prediction: make + model + model_year + component, per month.
- `component` = top-level part (text before the first `:` in `COMPDESC`).
- Label = recalled within 6 months.
- Split: train 2010-2019 / tune 2020-2021 / test 2022-2024.
- Free stack only: DuckDB, BigQuery sandbox, Gemini free tier.
- Track B kept: the Waymo go/no-go passed.
- `stg_complaints` includes `odino`. The agent cites and validates ODINO numbers, and `complaint_id`
  (CMPLID) is a different, per-row ID.

## Known follow-ups for "Contracts + source switches"

- The agent (PR #1) still reads its own column names (`description`, `crash`, `fire`,
  `injuries`). Switch it to the contract names (`narrative`, `is_crash`, `is_fire`, `injured`)
  once `main.stg_complaints` exists.
- The agent's sample watchlist has `POWER TRAIN:AUTOMATIC TRANSMISSION`. Under the component
  decision that becomes `POWER TRAIN`.
- `scripts/nhtsa_data_cleaning.py` (PR #2) writes `stg.complaints` and `int.monthly_features` with
  different column names. These are interim tables; the contract tables come from dbt in `main`.
- Until `main.watchlist` exists, `python run.py briefs` uses `src/agent/fixtures/watchlist_sample.csv`.
