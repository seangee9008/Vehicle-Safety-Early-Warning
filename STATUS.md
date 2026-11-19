# STATUS

Rules, ownership and contracts are in [AGENTS.md](AGENTS.md). Update this file in the same branch
as the work. Status is one of: Not started / In progress / In review / Done.

Last updated: Nov 18.

| Task | Owner | Status | Branch or PR | Due |
| --- | --- | --- | --- | --- |
| Repo setup + NHTSA/Waymo data | Sean | Done | main | Nov 6 |
| BigQuery sandbox connected | Sean | Done | main | Nov 6 |
| Raw data summary (profiling notebook + interim cleaning script) | Angelina | Done | PR #2 | Nov 12 |
| Cleaned `stg.complaints` + `stg.recalls` (standardized make/model/component) | Angelina | Done | PR #4 | Nov 12 |
| EDA + labels + split + complaint-count baseline | Justine | Done | PR #5 | Nov 12 |
| `monthly_features` + LightGBM + backtest + watchlist + SHAP reasons | Justine | Done | PR #5 | Nov 19 |
| Final recall results: drop `UNKNOWN OR OTHER`, top-k, age vs complaints ablation, case studies | Justine | Done | PR #7 | Nov 19 |
| Agent v0 + validator + cache + model fallback | Sean | Done | PR #6 | Nov 12 |
| Waymo constant-velocity baseline (go/no-go PASSED: FDE@8s 25.85 m overall, 28.72 vehicle, 2.98 pedestrian, 10.24 cyclist) | Sean | Done | PR #6 | Nov 12 |
| Project hub (AGENTS.md, STATUS.md, run.py) | Sean | Done | PR #6 | Nov 12 |
| Integration: contracts + sources layer + agent on real tables + run.py | Sean | Done | PR #6 | Nov 12 |
| Agent v1: self-correction retry, same-make recall search, SHAP reasons check, timeouts (Dec 2024 top 10: 81/81 citations valid, 10/10 pass) | Sean | In review | sean/agent-v1 | Nov 19 |
| CI | Sean | Not started | | Nov 19 |
| Dagster pipeline | Sean | Not started | | Nov 19 |
| Streamlit demo skeleton | Sean | Not started | | Nov 19 |
| Waymo: more shards + distance-adjusted error | Sean | Not started | | Nov 19 |
| dbt staging models + tests (`stg_complaints`, `stg_recalls`) | Angelina | Not started | | Nov 19 |
| dbt intermediate models + snapshots | Angelina | Not started | | Nov 19 |
| Waymo PyTorch model | Sean | Not started | | Nov 24 |
| Final demo | Sean | Not started | | Nov 24 |
| Data quality tests + publish final tables to BigQuery | Angelina | Not started | | Nov 24 |
| SHAP + final metrics | Justine | Done | PR #5, PR #7 | Nov 24 |
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

**Recall model (Justine, PR #7, stg, test 2022-24; Sean's rerun Nov 18):**

| Scorer | precision@20 | PR-AUC | median weeks of warning | recalls caught in top 20 / 50 / 100 |
| --- | --- | --- | --- | --- |
| LightGBM v1 | 11.4% (her write-up: 11.8%) | 0.031 | 22.7 | 63 / 117 / 185 (3.7% / 6.9% / 11.0%) |
| Complaint-count baseline | 8.9% | 0.016 | 35.1 | 30 / 45 / 74 (1.8% / 2.7% / 4.4%) |

- `UNKNOWN OR OTHER` is no longer scored (was 65 of 720 watchlist rows). No table or column changes.
- Top 50 / 100: precision drops (9.6% / 8.2%) but the model catches about 2x / 3x more recalls.
- Age vs complaints (precision@20 / PR-AUC / caught in top 20): age only 7.6% / 0.026 / 35,
  complaints only 7.1% / 0.017 / 42, both (v1) 11.4% / 0.031 / 63. Neither alone comes close:
  age picks the pool of new vehicles, complaints rank within it.
- Base rate 0.3-0.5%, so a random top 20 catches about 0.1 recalls a month; v1 catches about 2.
- No leakage signs (precision@20 well under 60%, median warning well under 40 weeks).

**Agent v1 on Justine's watchlist (2024-12, top 10, all real tables, cache on):**

| Rank | Vehicle | Component | Model used | Citations valid | Uncited | Evidence bullets | Self-correction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Tesla Model Y 2025 | ADAS AND STABILITY CONTROL | gemini-3.6-flash | 11/11 | 0 | 5 | passed first try |
| 2 | Tesla Model Y 2024 | ADAS AND STABILITY CONTROL | gemini-3.6-flash | 4/4 | 0 | 4 | passed first try |
| 3 | Kia Carnival 2025 | ELECTRICAL SYSTEM | gemini-3.7-flash | 13/13 | 0 | 5 | passed first try |
| 4 | Chevrolet Traverse 2024 | POWER TRAIN | gemini-3-flash-preview | 8/8 | 0 | 6 | passed first try (v0, cached) |
| 5 | Cadillac Lyriq 2024 | ADAS AND STABILITY CONTROL | gemini-3.6-flash | 7/7 | 0 | 6 | passed first try |
| 6 | Kia EV9 2024 | ADAS AND STABILITY CONTROL | gemini-3.6-flash | 13/13 | 0 | 5 | passed first try |
| 7 | Chevrolet Blazer EV 2024 | ADAS AND STABILITY CONTROL | gemini-3-flash-preview | 6/6 | 0 | 5 | passed first try (v0, cached) |
| 8 | Chevrolet Equinox 2025 | ELECTRICAL SYSTEM | gemini-3.5-flash-lite | 4/4 | 0 | 4 | passed first try |
| 9 | Toyota Tacoma 2024 | POWER TRAIN | gemini-3.5-flash-lite | 10/10 | 0 | 5 | passed first try (regenerated) |
| 10 | Jeep Grand Cherokee 4xe 2024 | ENGINE AND ENGINE COOLING | gemini-3.5-flash-lite | 5/5 | 0 | 5 | passed first try |
| **Total** | | | | **81/81** | **0** | **49** | **10 first try, 0 after retry, 0 failing** |

- 10/10 pass every check. Self-correction was not needed this run (it is covered by mocked tests).
- Tacoma's cached v0 brief had 10 evidence bullets; the run regenerated it (5 bullets).
- Traverse and Blazer EV are cached v0 briefs that still pass, so they were kept; they lack the
  reasons check and searched recalls across all makes. `--force` regenerates them.
- 8 v1 briefs used the same-make recall search; none needed the all-makes fallback.
- Gemini: about 54 requests over three runs (under the 60 cap). One call hung for 2 hours (now a
  120 s timeout) and five got `504 DEADLINE_EXCEEDED` (now retried, then the next model).
- Example: [docs/example_brief_real.md](docs/example_brief_real.md) (Kia EV9, rank 6).

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
- **Commit `reports/recall_ablation_stg.csv`:** the write-up cites it, but it is not in the repo.
- (Done in PR #7: filter `UNKNOWN OR OTHER`, top 50 / top 100, age vs complaints.)
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
