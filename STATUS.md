# STATUS

Rules, ownership and contracts are in [AGENTS.md](AGENTS.md). Update this file in the same branch
as the work. Status is one of: Not started / In progress / In review / Done.

Last updated: Nov 20.

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
| Agent v1: self-correction retry, same-make recall search, SHAP reasons check, timeouts (Dec 2024 top 10: 81/81 citations valid, 10/10 pass) | Sean | Done | PR #8 | Nov 19 |
| Label audit + investigations experiment (read-only, Justine's code untouched) | Sean | Done | PR #9, PR #11 | Nov 20 |
| CI: GitHub Actions, ruff + offline pytest on every PR | Sean | Done | PR #12 | Nov 19 |
| Pipeline: `python run.py pipeline` / `all` (replaces Dagster, see Decisions) | Sean | Done | PR #12 | Nov 19 |
| Streamlit demo (`app/`, 6 tabs, BigQuery + local fallback, AppTest tests, slide screenshots) | Sean | In review | sean/demo | Nov 24 |
| Waymo: more shards + distance-adjusted error (2 training shards; FDE@8s per metre travelled in reports/waymo_model_vs_baseline.csv) | Sean | Done | PR #10, PR #11 | Nov 19 |
| dbt staging models + tests (`stg_complaints`, `stg_recalls`) | Angelina | Not started | | Nov 19 |
| dbt intermediate models + snapshots | Angelina | Not started | | Nov 19 |
| Waymo PyTorch model: v2 kept (MLP K=6, + moving non-target agents, label smoothing 0.1, temperature scaling; chosen on held-out training scenarios). validation_00000 FDE@8s best guess 18.97 m vs CV 25.85 m (-27%), minFDE 7.32 m (-72%). v1 was 18.32 m on test; v1 vs v2 in reports/waymo_v1_vs_v2.csv | Sean | Done | PR #10, PR #11 | Nov 24 |
| Final demo (presented Monday) | Everyone | In progress | | Nov 24 |
| BigQuery publish: 7 final tables + publish_log + 2 views (sandbox, 1.67 MB) + read helper | Sean | Done | PR #10, PR #11 | Nov 20 |
| Data quality tests | Angelina | Not started | | Nov 24 |
| SHAP + final metrics | Justine | Done | PR #5, PR #7 | Nov 24 |
| Public README (results, team, how to run) | Sean | In review | sean/docs-final | Nov 20 |
| Public deploy: live BigQuery -> saved copy -> final snapshot (daily Action), Streamlit Cloud + Docker | Sean | In review | sean/deploy | Nov 21 |
| Write-up sections + presentation | Everyone | Not started | | Nov 24 |

## Headline numbers

> **The single official values** (use these in the write-up, slides and demo; same as README.md).
>
> | Track | Metric | Value | Baseline | Source |
> | --- | --- | --- | --- | --- |
> | A: recall model, test 2022-24 | precision@20 | **11.8%** | 8.9% | `reports/recall_metrics_stg.csv` |
> | | PR-AUC | **0.031** | 0.016 | `reports/recall_metrics_stg.csv` |
> | | recalls caught, top 100 (of 1,685 first recalls with prior complaints) | **11.1%** (187) | 4.4% (74) | `reports/recall_topk_stg.csv` |
> | | median weeks of warning (top 20) | **23.2** | 35.1 | `reports/recall_metrics_stg.csv` |
> | A: agent, Dec 2024 top 10 | briefs passing every check | **10/10** | | `reports/briefs/watchlist_2024-12/validation.csv`, BigQuery `agent_briefs` |
> | | citations | **81 citations / 76 unique complaints, all valid** | | BigQuery `agent_citations` |
> | B: Waymo, 1,268 test agents | FDE@8s best guess | **18.97 m (-27%)** | 25.85 m (constant velocity) | `reports/waymo_model_vs_baseline.csv` |
> | | FDE@8s best of 6 | **7.32 m (-72%)** | 25.85 m | `reports/waymo_model_vs_baseline.csv` |
>
> Reruns of the recall backtest give 11.4% precision@20 (and PR-AUC 0.031, 22.7 weeks): run-to-run
> noise of about 1 point (Justine's write-up). 11.8% is the committed, official value. The label
> audit and the investigations experiment below were computed on the 11.4% rerun's predictions.

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

**Recall model (Justine, PR #7, stg, test 2022-24). Official values from her committed CSVs; Sean's Nov 18 rerun in brackets:**

| Scorer | precision@20 | PR-AUC | median weeks of warning | recalls caught in top 20 / 50 / 100 |
| --- | --- | --- | --- | --- |
| LightGBM v1 | 11.8% (rerun 11.4%) | 0.031 | 23.2 (rerun 22.7) | 62 / 126 / 187 (3.7% / 7.5% / 11.1%) |
| Complaint-count baseline | 8.9% | 0.016 | 35.1 | 30 / 45 / 74 (1.8% / 2.7% / 4.4%) |

- `UNKNOWN OR OTHER` is no longer scored (was 65 of 720 watchlist rows). No table or column changes.
- Top 50 / 100: precision drops (9.6% / 7.9%) but the model catches about 2x / 3x more recalls.
- Age vs complaints (precision@20 / PR-AUC / caught in top 20, `reports/recall_results_stg.md`):
  age only 7.6% / 0.026 / 35, complaints only 6.4% / 0.017 / 41, both (v1) 11.8% / 0.031 / 62.
  Neither alone comes close: age picks the pool of new vehicles, complaints rank within it.
- The limit is the data: 76% of missed recalls (1,241 of 1,623) had two or fewer complaints in the
  year before (`reports/recall_miss_reasons_stg.csv`).
- Base rate 0.3-0.5%, so a random top 20 catches about 0.1 recalls a month; v1 catches about 2.
- No leakage signs (precision@20 well under 60%, median warning well under 40 weeks).

**Label audit (Sean, read-only, [reports/label_audit.md](reports/label_audit.md), on the 11.4% rerun):**
the grading is fair. Model-name variants (e.g. CX-90 PHEV flagged, recall filed under CX-90) add about 2
points. 8% of the model's misses were recalled 6-12 months later (early, not wrong). 44% of misses
are new cars recalled for a different part, but that mostly reflects the base rate (lift falls 44x to 12x).

**Investigations experiment (Sean, `src/recall_audit/investigations.py`, Justine's code imported,
not edited):** adds "an NHTSA investigation (PE / EA / DP) is open on this car + part / car / make +
part at month end". Picked on validation, test run once:

| | precision@20 val | precision@20 test | PR-AUC test | median weeks of warning | first recalls caught |
| --- | --- | --- | --- | --- | --- |
| v1 | 7.1% | 11.4% | 0.031 | 22.7 | 3.7% |
| v1 + investigations | 8.75% | 14.0% | 0.031 | 23.4 | 2.6% |

- Precision@20 up 2.6 points on test, but PR-AUC is flat and it catches *fewer* different recalls:
  an open investigation stays open for months, so the same car + parts sit in the top 20 month after
  month. And those recalls NHTSA was already investigating, so less early warning for an investigator.
- Leakage risk is small: 2.5% of investigation vehicle rows were added to the scope after opening.
- Decision: keep v1 as the model. Use "open NHTSA investigation" as context in the agent's briefs
  instead (possible next step).

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

- 10/10 pass every check: 81 citations of 76 unique complaints (a few briefs cite the same
  complaint in two bullets), all valid. Self-correction was not needed this run (it is covered by
  mocked tests).
- Tacoma's cached v0 brief had 10 evidence bullets; the run regenerated it (5 bullets).
- Traverse and Blazer EV are cached v0 briefs that still pass, so they were kept; they lack the
  reasons check and searched recalls across all makes. `--force` regenerates them.
- 8 v1 briefs used the same-make recall search; none needed the all-makes fallback.
- Gemini: about 54 requests over three runs (under the 60 cap). One call hung for 2 hours (now a
  120 s timeout) and five got `504 DEADLINE_EXCEEDED` (now retried, then the next model).
- Example: [docs/example_brief_real.md](docs/example_brief_real.md) (Kia EV9, rank 6).

**BigQuery (Nov 19):** the final results are in `waymo-project-513802.vehicle_safety` (US). Read them
with `src/bq_reader.py` (`read_table`, `read_view`); setup and examples in
[CONTRACTS.md](CONTRACTS.md#reading-from-bigquery). Published: watchlist 720 rows, recall_metrics 108,
recall_audit 44, agent_briefs 10, agent_citations 76, waymo_results 36, waymo_demo_examples 50
(1.39 MB of the 1.67 MB total), publish_log, plus views `v_watchlist_with_briefs` and
`v_headline_metrics`. Row counts verified against local. Ledger: 1.67 MB of the 2 GB cap.
**Tables expire Jan 18, 2027: republish (`python run.py publish --publish`) before then.**

## Decisions

- Unit of prediction: make + model + model_year + component, per month.
- `component` = Angelina's standardized top-level part (e.g. `ENGINE AND ENGINE COOLING`).
- Label = recalled within 6 months.
- Split: train 2010-2019 / tune 2020-2021 / test 2022-2024.
- Free stack only: DuckDB, BigQuery sandbox, Gemini free tier.
- Track B kept: the Waymo go/no-go passed.
- No Dagster: with a week left it adds risk for little gain. `python run.py pipeline` runs the
  same steps in order with each owner's documented command and stops at the first failure.
- Waymo model is evaluated only on validation_00000; training uses only `training_*` shards, with
  10% of training scenarios held out for early stopping and for choosing between model versions
  (best-guess FDE@8s on held-out prediction targets). The test set is never used to choose.
- Contracts use the producers' real names (CONTRACTS.md). The agent cites `odi_no` (ODINO).
- A watchlist row for month M uses complaints received by the end of M; the agent's evidence
  cutoff is the first day of the next month.
- Local data: complaints 2010-2026 (`python run.py data` downloads them).

## Requested from teammates

**Justine**
- **Commit `reports/recall_ablation_stg.csv`:** the write-up cites it, but it is not in the repo
  (the BigQuery publish picks it up automatically once it is).
- (Done in PR #7: filter `UNKNOWN OR OTHER`, top 50 / top 100, age vs complaints.)
- (Smaller) `main.monthly_features` and `main.watchlist` hold whichever `--source` ran last. A
  `source` column on `main.monthly_features` (the watchlist has one) would let the contract check
  refuse a raw-source table.

**Angelina**
- **Lint (CI skips these until fixed):** `scripts/nhtsa_data_table_cleaning.py` has invalid escape
  sequences (`"\s"`, `"\-"` in regex strings; use raw strings `r"..."`, lines 23 and 84) and
  unsorted imports, as do `scripts/nhtsa_data_cleaning.py` and the profiling notebook.
  `ruff check --fix` on those files fixes all of them.
- **Retire the old cleaning script:** `scripts/nhtsa_data_cleaning.py` also writes `stg.complaints`,
  with different columns. Running it after the new script breaks the complaints contract (no
  `cmpl_id`, `odi_no`, `ldate`) and `src/sources.py` falls back to the raw placeholder.
- **Fix the usage line:** `scripts/nhtsa_data_table_cleaning.py` says to run
  `nhtsa_data_cleaning_v2.py`; the file is `nhtsa_data_table_cleaning.py`.
- **Rename `int.monthly_features`:** it shares a name with Justine's `main.monthly_features` but
  not the columns (e.g. `int.complaints_monthly`).
