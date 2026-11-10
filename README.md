# Vehicle Safety Early Warning

> **Start here:** `python run.py check`
>
> - [AGENTS.md](AGENTS.md): rules, ownership, data contracts and commands (the source of truth for people and AI assistants).
> - [STATUS.md](STATUS.md): who is doing what, and when it's due.

Spotting vehicle safety problems before they happen, at two scales:

- **Track A, fleet level:** predict which vehicle components will be recalled, using only NHTSA consumer complaints filed **before** each recall, and measure how many weeks of warning that gives.
- **Track B, street level:** use the Waymo Open Motion Dataset to predict where cars, cyclists and pedestrians move over the next 8 seconds, compared against a constant-velocity baseline.

**Team:** Sean Gee · Angelina Yang · Justine Li-Wu

> **Status:** setup skeleton. The repo downloads and loads the raw data; the analysis and models are ours to build.

## Get started

Follow **[SETUP.md](SETUP.md)**. When you're done, `python scripts/check_setup.py` should show `[ok]` everywhere.

## What's here

```
scripts/
  check_setup.py          verify installs, keys and data
  download_nhtsa.py       NHTSA complaints, recalls, investigations (+ data dictionaries)
  load_nhtsa.py           raw .txt -> Parquet + DuckDB (+ BigQuery optional)
  download_waymo.sh       a few Waymo Motion shards via gsutil
  build_waymo_protos.py   compile Waymo's data format (one time)
  waymo_to_parquet.py     Waymo shards -> flat Parquet (tracks + map)
src/
  nhtsa_schemas.py        column names for NHTSA files
  waymo_reader.py         read Waymo scenarios in plain Python (no TensorFlow)
  recall_model/           Track A model           (to build)
  agent/                  Track A brief agent     (to build)
  trajectory/             Track B model           (to build)
app/                      Streamlit demo          (to build)
dbt/                      SQL models on the raw NHTSA tables (to build; sources already wired)
notebooks/                exploration
reports/                  figures, write-up
third_party/waymo_protos/ Waymo's .proto files (Apache 2.0)
data/                     downloaded data, git-ignored (created by the scripts)
```

## The raw data you'll have

| Table | Where | Grain |
| --- | --- | --- |
| `raw.complaints` | `data/nhtsa.duckdb` | one row per complaint per component |
| `raw.recalls` | `data/nhtsa.duckdb` | one row per recall campaign per vehicle per component |
| `raw.investigations` | `data/nhtsa.duckdb` | one row per NHTSA defect investigation |
| `*_tracks.parquet` | `data/processed/waymo/` | one row per agent per timestep (10 Hz, 9.1 s scenes) |
| `*_map.parquet` | `data/processed/waymo/` | one row per map point (lanes, road edges, crosswalks...) |

NHTSA columns load as text exactly as published; dates are `YYYYMMDD`. Field meanings are in `data/raw/nhtsa/dictionaries/`.

## Waymo data

Waymo Open Motion Dataset **v1.3.1** is set up locally with one validation shard
(`validation.tfrecord-00000-of-00150`). It contains **286 scenarios** and converts
to **1,666,301 track rows**. The shard and its processed Parquet files use about
**431 MB** on disk. Data remains local and is not committed.

## Data sources and licenses

- NHTSA: [Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis), public domain.
- Waymo: [Open Motion Dataset](https://waymo.com/open/about), non-commercial license; each person accepts the terms before downloading. Data is never committed to this repo.
