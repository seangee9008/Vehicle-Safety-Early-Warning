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
| Angelina | Data cleaning: dbt staging models `stg_complaints`, `stg_recalls` (`dbt/models/staging/`) |
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

All four are tables (or views) in `data/nhtsa.duckdb`, schema `main` (where dbt writes them).

| Table | Columns | Owner |
| --- | --- | --- |
| `stg_complaints` | `complaint_id`, `odino`, `make`, `model`, `model_year`, `component`, `received_date`, `is_crash`, `is_fire`, `injured`, `deaths`, `miles`, `narrative` | Angelina |
| `stg_recalls` | `campaign_id`, `make`, `model`, `model_year`, `component`, `recall_date`, `defect_summary` | Angelina |
| `monthly_features` | `make`, `model`, `model_year`, `component`, `month`, `cmpl_1m`, `cmpl_3m`, `cmpl_6m`, `severe_3m`, `vehicle_age`, `label` | Justine |
| `watchlist` | `month`, `make`, `model`, `model_year`, `component`, `score`, `rank`, `top_reasons` | Justine |

- `complaint_id` is `CMPLID` (one row). `odino` is `ODINO` (the complaint number the agent cites).
- **Renaming or removing a contract column needs the whole team's OK. Adding columns is fine.**
- Tables outside schema `main` (for example `stg.complaints` and `int.monthly_features` from
  `scripts/nhtsa_data_cleaning.py`) are interim work, not the contract.

## Rules

- Python 3.11 in `.venv`.
- Free tools only. Never enable billing.
- Never print, log or commit secrets (`GEMINI_API_KEY` lives only in `.env`).
- Never commit `data/`, `.env`, `.venv/`, `src/waymo_pb/`, `reports/briefs/`.
- One branch per task, named `<name>/<task>`, branched from `main`.
- Open a PR and get one teammate's review before merging. Never push to `main`.
- Never hand-edit AI-generated briefs. If a brief fails validation, fix the agent or regenerate.
- Tests run without network access or API keys (mock Gemini, use tiny fixtures).
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
| `python run.py check` | `scripts/check_setup.py`, then `pytest -q`, then `ruff check .` |
| `python run.py data` | Download NHTSA files (skips files already present), then load DuckDB if it doesn't exist yet (`--reload` to rebuild) |
| `python run.py dbt` | `dbt build` in `dbt/` (copies `profiles.yml.example` to `profiles.yml` if missing) |
| `python run.py test` | `pytest -q` |
| `python run.py lint` | `ruff check .` |
| `python run.py waymo` | Waymo constant-velocity baseline (`python -m src.trajectory.baseline`) |
| `python run.py briefs` | Agent briefs for the watchlist, cache on (existing briefs are revalidated, never forced) |
| `python run.py status` | Print the task table from STATUS.md |
| `python run.py contracts`, `pipeline`, `demo`, `all` | Not built yet |

The agent itself: `python -m src.agent.run --watchlist <csv> --top 5` (add `--force` only to
regenerate on purpose; it uses the free daily quota).

## How AI agents work here

1. Read AGENTS.md and STATUS.md first.
2. Do only the task you were asked. Stay inside the asking person's folders.
3. Run `python run.py test` (and `lint` on the files you changed) before committing.
4. Update STATUS.md in the same branch when the task is done.
5. End with a short summary: what changed, the PR link, and what's next.
