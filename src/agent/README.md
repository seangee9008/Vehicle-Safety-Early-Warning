# Track A: investigation agent

The agent writes cited investigation briefs for watchlist rows:

```bash
python -m src.agent.run --top 10                      # main.watchlist (latest month) if built, else the sample CSV
python -m src.agent.run --month 2024-11-01 --top 10   # a specific watchlist month
python -m src.agent.run --watchlist src/agent/fixtures/watchlist_sample.csv --top 5
```

Data comes through `src/sources.py`: Angelina's `stg.complaints` / `stg.recalls` and Justine's
`main.watchlist` when they exist and pass their contracts ([CONTRACTS.md](../../CONTRACTS.md)),
otherwise placeholder views over the raw tables and the sample CSV. Column names follow the
contracts (`odi_no`, `ldate`, `narrative`, `is_crash`, `is_fire`, `injured_count`, ...).

A watchlist row for month M uses complaints received by the end of M, so the agent's evidence
cutoff is the first day of the next month. Models are matched after the stg normalization
(`F-150` = `F150`), and a component matches exactly, by its top-level part, or as a prefix.
Citations are valid when the ODINO is a complaint for that car + part received before the
cutoff. Common direct identifiers in complaint text (email, phone, VIN, URL, street address)
are redacted before the text is sent to Gemini.

Briefs for the sample CSV go to `reports/briefs/`; briefs for Justine's watchlist go to
`reports/briefs/watchlist_<YYYY-MM>/`. Both are git-ignored.

Set `GEMINI_API_KEY` in the local `.env`; `GEMINI_MODEL` defaults to
`gemini-3.6-flash`.

To list models available to your local API key, run
`python -m src.agent.models`. Choose a Flash model returned by that command and
set its exact model ID as `GEMINI_MODEL` in `.env`.

### Citations

Each Evidence bullet must cite at least one complaint. The validator accepts
`[ODINO 111]`, grouped `[ODINO 111, 222]` / `[ODINO 111; ODINO 222]`, and adjacent
`[ODINO 111][ODINO 222]`. Every ID must be a complaint received before the watchlist month.

### Model fallback

`GEMINI_MODEL` is tried first. If it still returns HTTP 429 or 503 after two short
retries, the brief is restarted from scratch on the next Flash / Flash-Lite text model
from the key's model list (newest first, Flash before Flash-Lite). Each brief is
therefore written by a single model, which is recorded in `validation.csv`
(`model_used`) and in a `<brief>.meta.json` file next to the brief. Set
`GEMINI_FALLBACK_MODELS=model-a,model-b` to pin the fallback order.

### Cache

Briefs that already exist are not regenerated, which saves the free daily quota on
retries. They are revalidated against the database instead (`source=cached` in
`validation.csv`). Pass `--force` to regenerate every brief, or delete a single brief
file to regenerate only that one.
