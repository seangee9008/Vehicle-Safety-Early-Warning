# STATUS

Rules, ownership and contracts are in [AGENTS.md](AGENTS.md). Update this file in the same branch
as the work. Status is one of: Not started / In progress / In review / Done.

Last updated: Nov 17.

| Task | Owner | Status | Branch or PR | Due |
| --- | --- | --- | --- | --- |
| Repo setup + NHTSA/Waymo data | Sean | Done | main | Nov 6 |
| BigQuery sandbox connected | Sean | Done | main | Nov 6 |
| Raw data summary (profiling notebook + interim cleaning script) | Angelina | Done | PR #2 | Nov 12 |
| Cleaned `stg.complaints` + `stg.recalls` (standardized make/model/component) | Angelina | Done | PR #4 | Nov 12 |
| EDA + labels + split + complaint-count baseline | Justine | Done | PR #5 | Nov 12 |
| `monthly_features` + LightGBM + backtest + watchlist + SHAP reasons | Justine | Done | PR #5 | Nov 19 |
| Agent v0 + validator + cache + model fallback | Sean | In review | in the integration PR (close PR #1 when it merges) | Nov 12 |
| Waymo constant-velocity baseline (go/no-go PASSED: FDE@8s 25.85 m overall, 28.72 vehicle, 2.98 pedestrian, 10.24 cyclist) | Sean | In review | merged into sean/integration | Nov 12 |
| Project hub (AGENTS.md, STATUS.md, run.py) | Sean | In review | merged into sean/integration | Nov 12 |
| Integration: contracts + sources layer + agent on real tables + run.py | Sean | In review | sean/integration (PR to main) | Nov 12 |
| Agent on the real watchlist (Dec 2024 top 10: 78/78 citations valid, 8/10 pass all checks) | Sean | In review | sean/integration | Nov 18 |
| Agent: vehicle-scoped similar-recall search; enforce 3-6 evidence bullets | Sean | Not started | | Nov 19 |
| CI | Sean | Not started | | Nov 19 |
| Dagster pipeline | Sean | Not started | | Nov 19 |
| Streamlit demo skeleton | Sean | Not started | | Nov 19 |
| Waymo: more shards + distance-adjusted error | Sean | Not started | | Nov 19 |
| dbt staging models + tests (`stg_complaints`, `stg_recalls`) | Angelina | Not started | | Nov 19 |
| dbt intermediate models + snapshots | Angelina | Not started | | Nov 19 |
| Waymo PyTorch model | Sean | Not started | | Nov 24 |
| Final demo | Sean | Not started | | Nov 24 |
| Data quality tests + publish final tables to BigQuery | Angelina | Not started | | Nov 24 |
| SHAP + final metrics | Justine | In progress | PR #5 has SHAP reasons | Nov 24 |
| Write-up sections + presentation | Everyone | Not started | | Nov 24 |

## Tables (python run.py status)

All four contract tables are real and pass their contracts (`python run.py contracts`), once
these have run locally: `scripts/nhtsa_data_table_cleaning.py`, then Justine's `--source stg`
steps (see `src/recall_model/README.md`). Until then `src/sources.py` uses placeholders.

| Contract | Real table | Rows (Nov 17, complaints 2010-2026) |
| --- | --- | --- |
| complaints | `stg.complaints` | 1,487,991 |
| recalls | `stg.recalls` | 219,987 |
| monthly_features | `main.monthly_features` | 16,011,564 |
| watchlist | `main.watchlist` | 720 (top 20 x 36 test months, 2022-01 to 2024-12) |

## Results so far

**Recall model (Justine, stg, test 2022-24, rerun Nov 17 matches her committed CSV):**

| Scorer | precision@20 | PR-AUC | median weeks of warning | catch rate (first recalls with complaints) |
| --- | --- | --- | --- | --- |
| LightGBM v1 | 10.3% (committed: 11.25%) | 0.027 | 23.7 (committed: 23.1) | 3.7% |
| Complaint-count baseline | 8.75% | 0.013 | 35.1 | 1.7% |

Base rate (6-month label): 0.53% of all car + part months, 0.32% of rows the model ranks.
No leakage signs: time-based split with refits on already-known labels, features use complaints
received by month end, complaints after a car + part's first recall are dropped.

**Agent on Justine's watchlist (2024-12, top 10, all real tables):**

| Rank | Vehicle | Component | Model used | Citations valid | Evidence bullets | Uncited |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Chevrolet Equinox EV 2024 | VISIBILITY | gemini-3.5-flash-lite | 7/7 | 4 | 0 |
| 2 | Chevrolet Traverse 2024 | POWER TRAIN | gemini-3-flash-preview | 8/8 | 6 | 0 |
| 3 | Kia Niro 2023 | UNKNOWN OR OTHER | gemini-3.5-flash-lite | 5/5 | 5 | 0 |
| 4 | Chevrolet Blazer EV 2024 | ADAS AND STABILITY CONTROL | gemini-3-flash-preview | 6/6 | 5 | 0 |
| 5 | Toyota Tacoma 2024 | POWER TRAIN | gemini-3.5-flash-lite | 10/10 | **10 (fails 3-6)** | 0 |
| 6 | GMC Yukon 2021 | ENGINE AND ENGINE COOLING | gemini-3.5-flash-lite | 11/11 | 5 | 0 |
| 7 | Chrysler Pacifica 2022 | POWER TRAIN | gemini-3.5-flash-lite | 11/11 | 5 | 0 |
| 8 | Kia Sportage 2023 | ENGINE AND ENGINE COOLING | gemini-3.5-flash-lite | 8/8 | **8 (fails 3-6)** | 0 |
| 9 | Ram 5500 2022 | POWER TRAIN | gemini-3.5-flash-lite | 6/6 | 6 | 0 |
| 10 | Honda Accord 2024 | STEERING | gemini-3.5-flash-lite | 6/6 | 5 | 0 |

- 78/78 citations valid, no uncited bullets. 8/10 pass every check; Tacoma and Sportage have too
  many evidence bullets (the model ignored the 3-6 instruction).
- The first Tacoma brief cited `ODINO 1160969`, a 7-digit ID that doesn't exist. The validator
  caught it; the brief was regenerated from scratch (not edited) and the new one is 10/10 valid.
- `gemini-3.6-flash` returned 503/429 all day; the fallback wrote every brief with other free
  Flash models. Example: [docs/example_brief_real.md](docs/example_brief_real.md) (rank 7).
- 9/10 briefs say "none found" for similar recalls: the search matches the part across every make,
  so the model rightly ignores the results. Next fix: scope it to the make (and model).

## Decisions

- Unit of prediction: make + model + model_year + component, per month.
- `component` = Angelina's standardized top-level part (e.g. `ENGINE AND ENGINE COOLING`).
- Label = recalled within 6 months.
- Split: train 2010-2019 / tune 2020-2021 / test 2022-2024.
- Free stack only: DuckDB, BigQuery sandbox, Gemini free tier.
- Track B kept: the Waymo go/no-go passed.
- Contracts use the producers' real names (CONTRACTS.md). The agent cites `odi_no` (ODINO).
- A watchlist row for month M uses complaints received by the end of M; the agent's evidence
  cutoff is the first day of the next month.
- Local data: complaints 2010-2026 (`python run.py data` downloads them).

## Requested from teammates

**Justine**
- **Filter `UNKNOWN OR OTHER`:** 65 of 720 watchlist rows (9%) are this catch-all part, including
  rank 3 in 2024-12. It isn't actionable for an investigation. Leave it out of the watchlist (or the
  model)?
- **Top 50 / top 100:** precision@20 is one point on the curve. Precision and catch rate at top 50
  and top 100 per month would show whether a bigger watchlist catches meaningfully more recalls.
- **Vehicle age vs complaints:** `vehicle_age` carries about two thirds of the model's gain, and many
  top reasons are "vehicle is under a year old". A comparison of an age-only model and a
  complaints-only model against v1 would show how much the complaint signal adds.
- (Smaller) `main.monthly_features` and `main.watchlist` hold whichever `--source` ran last. A
  `source` column on `main.monthly_features` (the watchlist has one) would let the contract check
  refuse a raw-source table.

**Angelina**
- **Retire the old cleaning script:** `scripts/nhtsa_data_cleaning.py` also writes `stg.complaints`,
  with different columns. Running it after the new script breaks the complaints contract (no
  `cmpl_id`, `odi_no`, `ldate`) and `src/sources.py` falls back to the raw placeholder.
- **Fix the usage line:** `scripts/nhtsa_data_table_cleaning.py` says to run
  `nhtsa_data_cleaning_v2.py`; the file is `nhtsa_data_table_cleaning.py`.
- **Rename `int.monthly_features`:** it shares a name with Justine's `main.monthly_features` but
  not the columns (e.g. `int.complaints_monthly`).
