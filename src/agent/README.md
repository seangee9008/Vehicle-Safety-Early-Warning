# Track A: investigation agent

The v0 agent creates cited investigation briefs from the sample Contract 3 watchlist:

```bash
python -m src.agent.run --watchlist src/agent/fixtures/watchlist_sample.csv --top 5
```

Contract 1 complaints expose `complaint_id`, `odino`, `make`, `model`, `model_year`,
`component`, `received_date`, `description`, `crash`, `fire`, `injuries`, and `deaths`.
The raw SQL path normalizes those fields and filters `received_date` strictly before
the watchlist month. Common direct identifiers in free-text descriptions (email,
phone, VIN, URL, and street address patterns) are redacted before the text is sent
to Gemini. Set `AGENT_SOURCE=stg` to read the same columns from
`main.stg_complaints` instead of `raw.complaints`.

The sample Contract 3 CSV has `month`, `make`, `model`, `model_year`, `component`,
`score`, `rank`, and `top_reasons`. Scores and reasons in the fixture are illustrative;
the agent is instructed not to treat them as complaint evidence.

Set `GEMINI_API_KEY` in the local `.env`; `GEMINI_MODEL` defaults to
`gemini-3.6-flash`. Briefs and citation checks are written under the git-ignored
`reports/briefs/` directory.

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
