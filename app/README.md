# Demo

```bash
python run.py demo                    # = streamlit run app/streamlit_app.py
VSE_DATA=local python run.py demo     # offline: skip BigQuery, use local files
```

Reads the published results from BigQuery (`src/bq_reader.py`, one-time setup in
[CONTRACTS.md](../CONTRACTS.md#reading-from-bigquery)). If BigQuery isn't reachable or you're not
logged in, it falls back to local files (DuckDB read-only, `reports/`, the Waymo demo examples) and
shows a **local data** badge. A table that is in neither place shows a notice in its tab; the rest
still works. Everything loads once (in parallel) and is cached; **Reload data** under "Technical details" refreshes.
No Gemini calls: the briefs were written and validated beforehand. Every number comes from the data.

| Tab | Shows | Data |
| --- | --- | --- |
| Overview | 4 KPI cards (hit rate, lead time, evidence check, position error), how both tracks work, data at a glance | `v_headline_metrics`, `recall_metrics`, `recall_audit`, `agent_citations`, `waymo_results`, `waymo_demo_examples`, `app/facts.json` |
| Recall watchlist | Top 20 per month with risk score, why flagged, brief ready; select a row and "View brief" | `v_watchlist_with_briefs` |
| Investigation brief | Header card with status chips, brief sections as cards, complaint IDs as chips | `v_watchlist_with_briefs`, `agent_citations` |
| Model results | 6 tiles (hit rate, coverage, lead time, what drives the score, grading, more signals), each with a detail view | `recall_metrics`, `recall_audit`, `reports/recall_overlap_stg.csv`, `reports/recall_miss_reasons_stg.csv`, `reports/recall_results_stg.md`, `reports/label_audit.md` |
| Trajectory | Road user pills, example picker, 3 KPI cards, the path plot, 3 tiles with detail views | `waymo_demo_examples`, `waymo_results` |
| Data and limitations | Data sources, honest limitations, how we kept it honest | `app/facts.json`, `recall_metrics`, `waymo_results`, `reports/` |

Files: `streamlit_app.py` (tabs), `data.py` (BigQuery + local fallback), `theme.py` (palette, chart
template, styles), `components.py` (KPI cards, tiles), `charts.py` (Plotly), `reports.py` (committed
report files), `build_facts.py` (writes `facts.json`, the dataset counts that aren't in BigQuery:
`python -m app.build_facts`).
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
