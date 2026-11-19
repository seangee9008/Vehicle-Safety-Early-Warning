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
