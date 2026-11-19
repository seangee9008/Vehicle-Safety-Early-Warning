# AGENTS.md: source of truth

Every teammate and every AI assistant works from this file. If something here is wrong, fix it in a
PR. Current tasks and who is doing them are in [STATUS.md](STATUS.md).

## Project: Vehicle Safety Early Warning

- **Track A (fleet level):** predict which make + model + model_year + component gets recalled
  within 6 months, using only NHTSA complaints filed *before* the recall. Measure how many weeks of
  warning that gives. An LLM agent (Gemini free tier) writes an investigation brief for each
  watchlist item, citing complaint IDs as `[ODINO 12345678]`.
- **Track B (street level):** Waymo Open Motion Dataset. Predict where agents are 8 s ahead and
  compare against a constant-velocity baseline.

## Who owns what

| Person | Owns |
| --- | --- |
| Angelina | Data cleaning: `stg.complaints`, `stg.recalls` (`scripts/nhtsa_data_cleaning.py`, `scripts/nhtsa_data_table_cleaning.py`, `dbt/models/staging/`) |
| Justine | Features + recall model: `notebooks/`, `src/recall_model/`, dbt mart `monthly_features` (`dbt/models/marts/`), table `main.watchlist` |
| Sean | Agent (`src/agent/`), Waymo (`src/trajectory/`), contracts, CI, pipeline, demo (`app/`) |

**AI agents must not edit folders owned by someone other than the person asking.** If a change is
needed there, say so in your summary instead.

## Data (git-ignored, lives on each laptop)

- `data/nhtsa.duckdb`: tables `raw.complaints`, `raw.recalls`, `raw.investigations`. All columns are
  TEXT exactly as NHTSA publishes them; dates are `YYYYMMDD`.
  Key complaint columns: `ODINO`, `MAKETXT`, `MODELTXT`, `YEARTXT`, `COMPDESC`, `LDATE`, `CDESCR`,
  `CRASH`, `FIRE`, `INJURED`, `DEATHS`, `MILES`, `CMPL_TYPE`.
  One ODINO (complaint) can have several rows, one per component.
- `data/processed/waymo/*_tracks.parquet`: `scenario_id`, `track_id`, `object_type`, `to_predict`,
  `steps_from_current`, `valid`, `x`, `y`, `heading`, `vx`, `vy` (10 Hz; step 0 = now).
  `*_map.parquet` holds map points (lanes, road edges, crosswalks...).

## Hand-off contracts (exact column names)

Details: [CONTRACTS.md](CONTRACTS.md); machine-readable: [contracts/contracts.yml](contracts/contracts.yml).
We use the producers' real names. All live in `data/nhtsa.duckdb`.

| Contract | Real table | Columns | Owner |
| --- | --- | --- | --- |
| complaints | `stg.complaints` | `cmpl_id`, `odi_no`, `make`, `model`, `model_year`, `component`, `ldate`, `is_crash`, `is_fire`, `injured_count`, `deaths_count`, `miles`, `narrative` | Angelina |
| recalls | `stg.recalls` | `camp_no`, `make`, `model`, `model_year`, `component`, `recall_date`, `defect_summary`, `consequence_summary` | Angelina |
| monthly_features | `main.monthly_features` | `make`, `model`, `model_year`, `component`, `month`, `cmpl_1m`, `cmpl_3m`, `cmpl_6m`, `severe_3m`, `vehicle_age`, `label` | Justine |
| watchlist | `main.watchlist` | `month`, `make`, `model`, `model_year`, `component`, `score`, `rank`, `top_reasons` | Justine |

- `odi_no` is NHTSA's ODINO, the complaint number the agent cites. `cmpl_id` (CMPLID) is per row.
- A `monthly_features` or `watchlist` row for month M uses complaints received by the end of M.
- **Renaming or removing a contract column needs the whole team's OK. Adding columns is fine.**
- Read these tables through `src/sources.py`: it uses the real table when it passes its contract,
  otherwise a placeholder, and says which. `python run.py contracts` checks them.

## Rules

- Python 3.11 in `.venv`.
- Free tools only. Never enable billing.
- Never print, log or commit secrets (`GEMINI_API_KEY` lives only in `.env`).
- Never commit `data/`, `.env`, `.venv/`, `src/waymo_pb/`, `reports/briefs/`.
- One branch per task, named `<name>/<task>`, branched from `main`.
- Open a PR and get one teammate's review before merging. Never push to `main`.
- Never hand-edit AI-generated briefs. If a brief fails validation, fix the agent or regenerate.
- Tests run without network access or API keys (mock Gemini, use tiny fixtures).
- CI (`.github/workflows/ci.yml`) runs ruff and `pytest -q` on every PR; keep it green.
- BigQuery sandbox (project `waymo-project-513802`, dataset `vehicle_safety`): never load raw
  data. Only the final small tables, in week 3, as full replacements. 10 GB limit, 60-day expiry.

## Timeline

| Week | Dates | Milestone |
| --- | --- | --- |
| 1 | Nov 4-12 | Nov 12: clean tables (`stg_*`) handed to features |
| 2 | Nov 13-19 | Nov 18: `watchlist` handed to the agent and demo |
| 3 | Nov 20-24 | Nov 24: code freeze, write-up, presentation |

## Commands

Run from the repo root with the `.venv` Python. Each command prints what it runs.

| Command | What it does |
| --- | --- |
| `python run.py check` | `scripts/check_setup.py`, then `pytest -q`, then the lint below |
| `python run.py data` | Download NHTSA files incl. complaints 2010-2026 (skips files already present), then load DuckDB if it doesn't exist yet (`--reload` to rebuild) |
| `python run.py dbt` | `dbt build` in `dbt/` (copies `profiles.yml.example` to `profiles.yml` if missing) |
| `python run.py test` | `pytest -q` |
| `python run.py lint` | `ruff check .`, skipping Angelina's cleaning scripts and notebook until their fixes land (same as CI) |
| `python run.py waymo` | Waymo constant-velocity baseline (`python -m src.trajectory.baseline`) |
| `python run.py briefs` | Agent briefs for `main.watchlist` (latest month, top 10), or the sample CSV if not built; cache on, never forced, at most 60 Gemini requests per run |
| `python run.py contracts` | Check the real tables against `contracts/contracts.yml` |
| `python run.py publish` | Final small tables -> BigQuery sandbox (`scripts/publish_bigquery.py`): dry run; `--publish` uploads, creates the views, verifies row counts. Read them with `src/bq_reader.py` (CONTRACTS.md) |
| `python run.py status` | Print the task table from STATUS.md, then real vs placeholder per table |
| `python run.py pipeline` | Rebuild everything in order, stopping at the first failure: stg cleaning, Justine's labels / features / model / watchlist / results, contracts, label audit, briefs (cache on, request cap), Waymo evaluate + demo examples, BigQuery plan (dry run). `--dry-run` lists the steps. Rewrites Justine's committed `reports/recall_*_stg.csv` (`git checkout -- reports` restores them) |
| `python run.py all` | `check`, then `pipeline` |
| `python run.py demo` | Not built yet (`app/`) |

The agent itself: `python -m src.agent.run --watchlist <csv> --top 5` (add `--force` only to
regenerate on purpose; it uses the free daily quota).

## How AI agents work here

1. Read AGENTS.md and STATUS.md first.
2. Do only the task you were asked. Stay inside the asking person's folders.
3. Run `python run.py test` (and `lint` on the files you changed) before committing.
4. Update STATUS.md in the same branch when the task is done.
5. End with a short summary: what changed, the PR link, and what's next.
