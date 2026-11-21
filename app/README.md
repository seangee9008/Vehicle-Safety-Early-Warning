# Demo

```bash
python run.py demo                    # = streamlit run app/streamlit_app.py
VSE_DATA=local python run.py demo     # offline: skip BigQuery, use local files
```

Reads the published results from BigQuery (`src/bq_reader.py`, one-time setup in
[CONTRACTS.md](../CONTRACTS.md#reading-from-bigquery)). If BigQuery isn't reachable or you're not
logged in, it falls back to local files (DuckDB read-only, `reports/`, the Waymo demo examples) and
shows a **local data** badge. A table that is in neither place shows a notice in its tab; the rest
still works. Everything loads once (in parallel) and is cached; **Reload data** in About refreshes.
No Gemini calls: the briefs were written and validated beforehand. Every number comes from the data.

| Tab | Shows | Data |
| --- | --- | --- |
| Overview | Headline cards for both tracks + two-track diagram | `v_headline_metrics`, `recall_metrics`, `agent_citations`, `waymo_results` |
| Recall watchlist | Top 20 per month (2022-01 to 2024-12) with SHAP reasons and a brief flag | `v_watchlist_with_briefs` |
| Investigation brief | The brief with validation badges, cited complaint IDs | `v_watchlist_with_briefs`, `agent_citations` |
| Model results | Top 20/50/100 vs baseline, PR-AUC, warning weeks, age vs complaints, label audit + investigations, Justine's charts | `recall_metrics`, `recall_audit`, `reports/` |
| Trajectory | One agent: lanes, past, true future, constant velocity, 6 model paths; all-agent table at 3/5/8 s | `waymo_demo_examples`, `waymo_results` |
| About | Architecture, team, honest limitations | `recall_metrics`, `waymo_results`, `reports/recall_miss_reasons_stg.csv` |

Files: `streamlit_app.py` (tabs), `data.py` (BigQuery + local fallback), `charts.py` (Altair).
Tests: `tests/test_demo.py` (AppTest, BigQuery faked). Slide screenshots:
`python scripts/demo_screenshots.py` (needs `pip install playwright && python -m playwright install chromium`)
writes `reports/figures/demo_*.png`.

**Presenting:** open it once before the talk so the cache is warm (first BigQuery load ~5 s). No
wifi: `VSE_DATA=local`.

## Public link (Streamlit Community Cloud, free)

1. [share.streamlit.io](https://share.streamlit.io): sign in with GitHub, **Create app** → deploy from
   GitHub: repo `seangee9008/Vehicle-Safety-Early-Warning`, branch `main`, main file
   `app/streamlit_app.py`, pick a URL (e.g. `vehicle-safety-early-warning`).
2. **Advanced settings**: Python 3.11. **Secrets**: paste the read-only BigQuery login
   (`[gcp_service_account]`, a key for `vse-demo-reader@waymo-project-513802`, which can only read
   the `vehicle_safety` dataset and run queries). Sean has it; never commit it.

Cloud installs `app/requirements.txt` (the demo's few packages), not the full project requirements.
There is no local data on Cloud, so if BigQuery fails a tab shows a notice instead of numbers.
Sandbox tables expire Jan 18, 2027 (republish with `python run.py publish --publish`).
