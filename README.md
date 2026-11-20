# Vehicle Safety Early Warning

> **Start here:** `python run.py check`
>
> - [AGENTS.md](AGENTS.md): rules, ownership, data contracts and commands (the source of truth for people and AI assistants).
> - [STATUS.md](STATUS.md): who is doing what, and the headline numbers.

Spotting vehicle safety problems before they happen, at two scales. **Track A (fleet level)** ranks
every make + model + model year + part each month by how likely it is to be recalled within 6 months,
using only NHTSA consumer complaints filed *before* the recall. A Gemini agent then writes an
investigation brief for each car on the watchlist, citing the complaints it relies on as
`[ODINO n]`; a validator checks every cited ID. **Track B (street level)** predicts where cars,
pedestrians and cyclists in the Waymo Open Motion Dataset will be over the next 8 seconds, with an
MLP that proposes 6 possible paths, compared against a constant-velocity baseline.

## Results

| Track | Metric | Result | Baseline |
| --- | --- | --- | --- |
| A: recall model (LightGBM, test 2022-24) | precision@20 (share of each month's top 20 recalled within 6 months) | **11.8%** | 8.9% (3-month complaint count) |
| | PR-AUC | **0.031** | 0.016 |
| | recalls caught by a top-100 list (of 1,685 first recalls in 2022-24 with prior complaints) | **11.1%** (187) | 4.4% (74) |
| | median weeks of warning before the recall (top 20) | **~23 weeks** (23.2) | 35.1 (but it catches half as many) |
| A: investigation agent (Dec 2024 top 10) | briefs that pass every check | **10/10** | |
| | citations valid | **81/81** (76 unique complaints, all real and before the cutoff) | |
| B: trajectory model (1,268 agents in 286 test scenes) | FDE@8s, single best guess | **18.97 m (-27%)** | 25.85 m (constant velocity) |
| | FDE@8s, best of 6 paths | **7.32 m (-72%)** | 25.85 m |

Sources: [reports/recall_metrics_stg.csv](reports/recall_metrics_stg.csv),
[reports/recall_topk_stg.csv](reports/recall_topk_stg.csv),
[reports/recall_results_stg.md](reports/recall_results_stg.md),
[reports/waymo_model_vs_baseline.csv](reports/waymo_model_vs_baseline.csv); agent numbers from the
brief validation run (`python run.py briefs`, also in the BigQuery tables `agent_briefs` and
`agent_citations`). A rerun of the recall backtest gives 11.4% precision@20; run-to-run noise is
about 1 point, and 11.8% is the official, committed value.

**The limit is the data, not the model.** 76% of the recalls the model misses (1,241 of 1,623) had
two or fewer complaints in the year before, so there is little for any model to see. Vehicle age
alone gets 7.6% precision@20 and complaints alone 6.4%; together they reach 11.8%. An audit of the
label ([reports/label_audit.md](reports/label_audit.md)) showed the grading is fair, and adding open
NHTSA investigations raised precision but caught fewer distinct recalls, so the model was kept as is.

## Team

| Person | Built |
| --- | --- |
| Angelina Yang | Raw data profiling; cleaned `stg.complaints` / `stg.recalls` with standardized make, model and part; demo (with Justine) |
| Justine Li-Wu | Recall labels, EDA, `monthly_features`, LightGBM rolling backtest, watchlist with SHAP reasons, results and case studies; demo (with Angelina) |
| Sean Gee | Data download and loading, hand-off contracts, Gemini brief agent and citation validator, label audit and investigations experiment, Waymo baseline and MLP, BigQuery publish, CI, `run.py` pipeline |

## How to run

Setup once: follow **[SETUP.md](SETUP.md)** (Python 3.11 in `.venv`, `pip install -r requirements.txt`,
then `python run.py data` to download NHTSA data and load DuckDB).

| Command | What it does |
| --- | --- |
| `python run.py check` | Setup check, then `pytest -q`, then ruff (the same tests and lint CI runs on every PR) |
| `python run.py pipeline` | Rebuild everything in order: cleaning, labels, features, model, watchlist, results, contracts, label audit, agent briefs (cache on, at most 60 Gemini requests), Waymo evaluation and demo examples, BigQuery plan. `--dry-run` lists the steps |
| `python run.py publish` | BigQuery publish plan (dry run); `--publish` uploads the final small tables to the sandbox and verifies them. Reading them: [CONTRACTS.md](CONTRACTS.md#reading-from-bigquery) |
| `python run.py demo` | Streamlit demo: overview, watchlist, briefs, model results, trajectories, about. Reads BigQuery, falls back to local files ([app/README.md](app/README.md)) |

`python run.py all` runs `check` then `pipeline`. Every command is listed in [AGENTS.md](AGENTS.md#commands).

## What's here

```
run.py                    one entry point: check, data, pipeline, briefs, publish, ... (AGENTS.md)
scripts/
  check_setup.py          verify installs, keys and data
  download_nhtsa.py       NHTSA complaints, recalls, investigations (+ data dictionaries)
  load_nhtsa.py           raw .txt -> Parquet + DuckDB
  nhtsa_data_table_cleaning.py   stg.complaints + stg.recalls (standardized names)
  nhtsa_data_cleaning.py  earlier interim cleaning script
  download_waymo.sh       a few Waymo Motion shards via gsutil
  build_waymo_protos.py   compile Waymo's data format (one time)
  waymo_to_parquet.py     Waymo shards -> flat Parquet (tracks + map)
  publish_bigquery.py     final small tables -> BigQuery sandbox
src/
  sources.py, contracts.py  hand-off tables: real table if it passes its contract, else a placeholder
  recall_model/           Track A: labels, features, LightGBM backtest, watchlist, results, ablation
  recall_audit/           label audit + open-investigations experiment
  agent/                  Track A: Gemini brief agent, citation validator, cached runner
  trajectory/             Track B: constant-velocity baseline, MLP (6 paths), evaluation, demo examples
  bq_publish.py           BigQuery publish: table list, schemas, size guards, views
  bq_reader.py            read the published tables (for the demo)
  nhtsa_schemas.py        column names for NHTSA files
  waymo_reader.py         read Waymo scenarios in plain Python (no TensorFlow)
app/                      Streamlit demo (BigQuery, local fallback)
contracts/contracts.yml   machine-readable hand-off contracts (CONTRACTS.md)
dbt/                      dbt project on the NHTSA tables (sources wired)
docs/                     example agent briefs
notebooks/                EDA, recall hand-checks, data profiling
reports/                  results tables, write-ups and figures
tests/                    offline tests (no data, keys or network needed)
.github/workflows/ci.yml  CI: ruff + pytest on every PR
third_party/waymo_protos/ Waymo's .proto files (Apache 2.0)
data/                     downloaded data, git-ignored (created by the scripts)
```

## Data

| Table | Where | Grain |
| --- | --- | --- |
| `raw.complaints` | `data/nhtsa.duckdb` | one row per complaint per component (2010-2026) |
| `raw.recalls` | `data/nhtsa.duckdb` | one row per recall campaign per vehicle per component |
| `raw.investigations` | `data/nhtsa.duckdb` | one row per NHTSA defect investigation per vehicle |
| `stg.complaints`, `stg.recalls` | `data/nhtsa.duckdb` | cleaned, standardized make / model / part |
| `main.monthly_features`, `main.watchlist` | `data/nhtsa.duckdb` | one row per car + part per month; top 20 per month |
| `*_tracks.parquet` | `data/processed/waymo/` | one row per agent per timestep (10 Hz, 9.1 s scenes) |
| `*_map.parquet` | `data/processed/waymo/` | one row per map point (lanes, road edges, crosswalks...) |

Waymo Open Motion Dataset **v1.3.1**: two training shards for training, and one validation shard
(`validation.tfrecord-00000-of-00150`, 286 scenes) used only as the test set. The final small result
tables are shared in BigQuery (`waymo-project-513802.vehicle_safety`); raw data never leaves each
laptop.

## Data sources and licenses

- NHTSA: [Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis) (complaints, recalls,
  investigations), public domain.
- Waymo: [Open Motion Dataset](https://waymo.com/open/about), non-commercial license; each person
  accepts the terms before downloading. Data is never committed to this repo or published to BigQuery
  (only summary results and 50 example paths for the demo).
- Waymo's `.proto` files in `third_party/waymo_protos/`: Apache 2.0.
- Briefs are written with Google Gemini (free tier) and are not hand-edited.
