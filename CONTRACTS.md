# Hand-off contracts

The machine-readable version is [contracts/contracts.yml](contracts/contracts.yml); this page explains
it. Column names are the **producers' real names**: Angelina's `stg` tables and Justine's `main`
tables. Check them with `python run.py contracts`.

**Rules:** renaming or removing a listed column needs the whole team's OK. Adding columns is fine.
A table passes when it exists, has rows, and has every listed column with a compatible type.

## Tables

| Contract | Real table | Owner | Built by | Placeholder until it exists |
| --- | --- | --- | --- | --- |
| complaints | `stg.complaints` | Angelina | `python scripts/nhtsa_data_table_cleaning.py` | view over `raw.complaints` |
| recalls | `stg.recalls` | Angelina | `python scripts/nhtsa_data_table_cleaning.py` | view over `raw.recalls` |
| monthly_features | `main.monthly_features` | Justine | `python -m src.recall_model.features --source stg` | view over `raw.complaints` (label NULL) |
| watchlist | `main.watchlist` | Justine | `python -m src.recall_model.watchlist --source stg` | `src/agent/fixtures/watchlist_sample.csv` |

All live in `data/nhtsa.duckdb`. `src/sources.py` picks the real table when it passes its contract,
otherwise the placeholder, and logs which (`python -m src.sources`).

### complaints (`stg.complaints`, one row per complaint per component)

`cmpl_id`, `odi_no`, `make`, `model`, `model_year`, `component`, `ldate`, `is_crash`, `is_fire`,
`injured_count`, `deaths_count`, `miles`, `narrative`

- `odi_no` is NHTSA's ODINO: the complaint number the agent cites as `[ODINO 12345678]`.
  `cmpl_id` (CMPLID) is a different, per-row ID.
- `make`, `model`, `component` are standardized: `model` has punctuation and spaces removed
  (`F-150` becomes `F150`); `component` is a canonical top-level part
  (`ENGINE` becomes `ENGINE AND ENGINE COOLING`).
- `ldate` is the date NHTSA received the complaint. `is_crash` / `is_fire` are 1/0.
- Recall-generated (`RC`) and early-warning (`EWR`) complaints are excluded.

### recalls (`stg.recalls`, one row per campaign per vehicle per component)

`camp_no`, `make`, `model`, `model_year`, `component`, `recall_date`, `defect_summary`,
`consequence_summary`

### monthly_features (`main.monthly_features`, one row per car + part per month)

`make`, `model`, `model_year`, `component`, `month`, `cmpl_1m`, `cmpl_3m`, `cmpl_6m`, `severe_3m`,
`vehicle_age`, `label`

- A month's row uses complaints received by the **end** of that month.
- `label` = recalled within 6 months after the month ends. Justine's table has many more
  columns (see `src/recall_model/README.md`).

### watchlist (`main.watchlist`, top 20 car + parts per month)

`month`, `make`, `model`, `model_year`, `component`, `score`, `rank`, `top_reasons`

- A row for month M uses complaints received by the end of M. The agent therefore uses evidence
  received **before the first day of the next month** (`data_access.evidence_cutoff`).

## How consumers adapt (without changing producers' code)

- The agent compares models after the same normalization as `stg` (`F-150` = `F150`), and accepts
  a component that matches exactly, by its top-level part, or as a prefix (`ENGINE` matches
  `ENGINE AND ENGINE COOLING`). That keeps the sample watchlist working on the real tables.
- Placeholders produce the same columns and types, so code never branches on real vs placeholder.

## Not part of the contract

- `int.monthly_features` (from Angelina's cleaning script) is a different table from Justine's
  `main.monthly_features`, with different columns. Nothing reads it yet.
- `main.monthly_features` and `main.watchlist` hold whichever `--source` ran last; run `stg` last.

## Reading from BigQuery

The final small result tables are published to the BigQuery sandbox, so the demo and teammates
read one shared copy without local data: project `waymo-project-513802`, dataset
`vehicle_safety` (US). Sean publishes with `python run.py publish --publish` (dry run without the
flag); every publish is logged in `reports/bq_publish_ledger.csv` and the `publish_log` table.
Sandbox tables expire 60 days after a publish, so republish before then.

| Table / view | Rows | What |
| --- | --- | --- |
| `watchlist` | 720 | `main.watchlist`: top 20 car + parts per test month, with `top_reasons` |
| `recall_metrics` | 108 | Justine's committed metric CSVs, long format (`scorer`, `top_k`, `metric`, `value`) |
| `recall_audit` | 44 | Label audit + investigations experiment, long format |
| `agent_briefs` | 10 | One row per brief: Markdown, model, citation and self-correction results |
| `agent_citations` | 76 | One row per complaint cited in a brief (`brief_id`, `odino`, `valid`) |
| `waymo_results` | 36 | Model vs constant velocity by type x horizon x predictor |
| `waymo_demo_examples` | 50 | Agents to draw: paths, 6 model paths + confidences, lanes (JSON, agent frame) |
| `publish_log` | 1 per table per publish | When, which commit, rows, bytes |
| `v_watchlist_with_briefs` | view | `watchlist` LEFT JOIN `agent_briefs` on month + rank, `has_brief` |
| `v_headline_metrics` | view | One row of headline numbers for the overview page |

Every table and column has a description in BigQuery. Never published: raw/stg complaints and
recalls, `monthly_features`, Waymo tracks.

Setup once per laptop: ask Sean to give your Google account *BigQuery Data Viewer* and *BigQuery
Job User* on the project, then run `gcloud auth application-default login`. Reading is free.

```python
from src.bq_reader import read_table, read_view

headline = read_view("v_headline_metrics").iloc[0]
print(f"precision@20 {headline.model_precision_at_20:.1%} vs baseline {headline.baseline_precision_at_20:.1%}")

watch = read_view("v_watchlist_with_briefs")
latest = watch[watch.month == watch.month.max()].sort_values("rank")
for row in latest[latest.has_brief].itertuples():
    print(row.rank, row.make, row.model, row.model_year, row.component)
    print(row.brief_markdown)

examples = read_table("waymo_demo_examples")   # JSON columns are already Python lists
ex = examples.iloc[0]
best_path = ex.model_paths[0]                   # 80 [x, y] points, highest confidence first
```

Without the helper: `SELECT * FROM \`waymo-project-513802.vehicle_safety.v_headline_metrics\``
in the BigQuery console, or `bigquery.Client(project="waymo-project-513802").query(sql).to_dataframe()`.
