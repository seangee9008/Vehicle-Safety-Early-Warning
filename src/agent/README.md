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
