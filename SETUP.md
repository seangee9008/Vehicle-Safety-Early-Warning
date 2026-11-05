# Setup

About 30 minutes for the NHTSA part. The Waymo part adds about 15 minutes plus download time.
Mac commands are shown; Windows differences are noted.

## 1. Install these once

| Tool | Get it |
| --- | --- |
| Python 3.11 | Mac: `brew install python@3.11` · Windows: [python.org](https://www.python.org/downloads/) (tick "Add to PATH") |
| Git | Mac: comes with Xcode tools (`xcode-select --install`) · Windows: [git-scm.com](https://git-scm.com/downloads) |
| VS Code | [code.visualstudio.com](https://code.visualstudio.com/) |
| Google Cloud CLI | [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) (needed for Waymo downloads and BigQuery) |
| GitHub CLI (optional) | Mac: `brew install gh` · [cli.github.com](https://cli.github.com/) |

## 2. Put the repo on GitHub (Sean, once)

```bash
cd vehicle-safety-early-warning
git init -b main
git add .
git commit -m "Project setup skeleton"
gh repo create vehicle-safety-early-warning --private --source=. --push
```

No `gh`? Create an empty repo on github.com, then:
`git remote add origin https://github.com/<you>/vehicle-safety-early-warning.git && git push -u origin main`

Then on GitHub: **Settings → Collaborators** → add Angelina and Justine. **Settings → Branches** → protect `main` (require 1 review).

## 3. Everyone: clone, open, install

```bash
git clone https://github.com/<owner>/vehicle-safety-early-warning.git
cd vehicle-safety-early-warning
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt        # torch is the slow one
cp .env.example .env                   # Windows: copy .env.example .env
code .
```

On Mac, install the OpenMP runtime required by LightGBM with `brew install libomp`.

In VS Code:
1. Click **Install** on the "recommended extensions" pop-up (or open the Extensions panel and search `@recommended`).
2. `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.

Check it: `python scripts/check_setup.py` (data lines will show `[--]` until step 4).

## 4. NHTSA data (everyone)

```bash
python scripts/download_nhtsa.py                       # recalls, investigations, 2015-2019 complaints (~100 MB)
python scripts/load_nhtsa.py                           # -> data/nhtsa.duckdb, tables raw.*
```

More complaints when you need them:
`python scripts/download_nhtsa.py --complaints 2015-2019 2020-2024` or `--complaints all` (358 MB zipped), then re-run `load_nhtsa.py`.

Look at the data:

```bash
python -c "import duckdb; print(duckdb.connect('data/nhtsa.duckdb').sql('select MAKETXT, MODELTXT, YEARTXT, COMPDESC, LDATE, CDESCR from raw.complaints limit 5'))"
```

The recall schema matches all 29 fields in the downloaded `RCL.txt` dictionary, including the `DO_NOT_DRIVE` and `PARK_OUTSIDE` advisory fields.

## 5. dbt (Data owner first, others later)

```bash
cd dbt
cp profiles.yml.example profiles.yml
dbt debug --profiles-dir .
dbt build --profiles-dir .             # runs the one example model
cd ..
```

Optional, BigQuery instead of local DuckDB:

```bash
gcloud auth application-default login
gcloud config set project waymo-project-513802
gcloud auth application-default set-quota-project waymo-project-513802
bq --location=US mk --dataset waymo-project-513802:vehicle_safety
# Copy profiles.yml.example to profiles.yml, then run:
dbt build --profiles-dir . --target bigquery
```

### BigQuery (shared, free sandbox)

The team uses the `waymo-project-513802` BigQuery sandbox project with the
`vehicle_safety` dataset. Each teammate should:

1. Install the Google Cloud CLI (Mac: `brew install --cask google-cloud-sdk`).
2. Authenticate and select the shared project:

   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project waymo-project-513802
   gcloud auth application-default set-quota-project waymo-project-513802
   ```

3. Put these values in the local, git-ignored `.env`:

   ```dotenv
   GCP_PROJECT_ID=waymo-project-513802
   BQ_DATASET=vehicle_safety
   ```

4. Check access with `bq ls`; it should show `vehicle_safety`.
5. Copy `dbt/profiles.yml.example` to `dbt/profiles.yml`. DuckDB remains the
   default; select BigQuery explicitly with `dbt build --profiles-dir . --target bigquery`.

The sandbox is free, has a **10 GB lifetime storage limit**, and tables expire
after **60 days**. Never load raw data into this project. In week 3, publish only
the final small tables (features and watchlist), using full table replacements.
Do not enable billing or use paid APIs for this project.

## 6. Waymo data (Track B people)

1. Go to [waymo.com/open](https://waymo.com/open/), sign in with Google, and **accept the license terms**.
2. `gcloud auth login` with the same Google account.
3. On [waymo.com/open/download](https://waymo.com/open/download/), open the **Motion** dataset → **scenario** folder (uncompressed) and copy its `gs://` path (the folder with `training/` and `validation/`). Put it in `.env` as `WAYMO_MOTION_BUCKET`.

One validation shard is enough to start. Training shards are only needed in week 3; then download one with `N_TRAIN=1 bash scripts/download_waymo.sh`.

```bash
export WAYMO_MOTION_BUCKET="gs://<path you copied>"   # Windows PowerShell: $env:WAYMO_MOTION_BUCKET="gs://..."
bash scripts/download_waymo.sh            # 3 training + 1 validation shard (Windows: run in Git Bash)
python scripts/build_waymo_protos.py      # one time
python scripts/waymo_to_parquet.py --limit 50   # quick test, then run again without --limit
```

Look at the data:

```bash
python -c "import pandas as pd, glob; t = pd.read_parquet(glob.glob('data/processed/waymo/validation_*_tracks.parquet')[0]); print(t.head()); print(t.object_type.value_counts())"
```

No TensorFlow is needed. If you want a GPU later for training, upload the Parquet files to Google Colab.

## 7. Check everything

```bash
python scripts/check_setup.py
pytest -q
```

## 8. How we work

- One branch per task: `git checkout -b angelina/stg-complaints`
- Small pull requests, one reviewer, then merge to `main`
- Never commit `data/`, `.env`, or notebook outputs
- Pull often: `git pull origin main`
