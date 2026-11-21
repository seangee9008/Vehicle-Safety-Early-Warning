# Deploying the public demo

The app (`app/streamlit_app.py`) runs on [Streamlit Community Cloud](https://share.streamlit.io) for free
and keeps working after the BigQuery sandbox tables expire:

- **Live:** it reads BigQuery (`waymo-project-513802.vehicle_safety`) with a read-only service account.
  The header says "Live results · ends <date>", where the date is read from the table metadata.
- **Saved copy:** if BigQuery fails for any reason, it shows the latest snapshot (gray "Saved copy" chip).
- **Final:** a daily GitHub Action (`.github/workflows/final-snapshot.yml`) saves a final copy 3 days before
  the tables expire. From then on the app shows that copy, with a banner saying when live data ended.

How the app picks between these modes, and the commands to preview them: `app/data.py` (top of the file).

## 1. Read-only service account

It already exists: `vse-demo-reader@waymo-project-513802.iam.gserviceaccount.com`. It has
*BigQuery Job User* on the project and *reader* on the `vehicle_safety` dataset only, and it can't write.

To recreate it ([docs](https://cloud.google.com/iam/docs/service-accounts-create)):

```bash
gcloud iam service-accounts create vse-demo-reader --project waymo-project-513802
gcloud projects add-iam-policy-binding waymo-project-513802 \
  --member serviceAccount:vse-demo-reader@waymo-project-513802.iam.gserviceaccount.com --role roles/bigquery.jobUser
# Dataset reader: BigQuery console -> vehicle_safety -> Sharing -> Permissions -> add the account as BigQuery Data Viewer
```

Make one key for GitHub (step 2) and one for Streamlit (step 4), so each can be revoked on its own:

```bash
gcloud iam service-accounts keys create gcp-key-github.json \
  --iam-account vse-demo-reader@waymo-project-513802.iam.gserviceaccount.com
```

Key files match `.gitignore` (`gcp-key*.json`, `*-sa-key.json`). Delete them once they're stored (steps 2 and 4).

## 2. GitHub secret

```bash
gh secret set GCP_SA_KEY < gcp-key-github.json && rm gcp-key-github.json
```

The workflow writes the key to a temp file on the runner and never prints it
([docs](https://docs.github.com/actions/security-guides/using-secrets-in-github-actions)).

## 3. Let Actions push the snapshot branch and open PRs

Repo **Settings → Actions → General → Workflow permissions**: choose *Read and write permissions* and tick
*Allow GitHub Actions to create and approve pull requests*. Or run:

```bash
gh api -X PUT repos/seangee9008/Vehicle-Safety-Early-Warning/actions/permissions/workflow \
  -f default_workflow_permissions=write -F can_approve_pull_request_reviews=true
```

The workflow only uses `GITHUB_TOKEN`. Leave the `snapshot` branch unprotected: it only holds data.

## 4. Deploy on Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → **Create app**.
   - Repo `seangee9008/Vehicle-Safety-Early-Warning`, branch `main`, file `app/streamlit_app.py`.
   - Python 3.11.
2. **Secrets:** paste the `[gcp_service_account]` block. `.streamlit/secrets.toml.example` lists the field
   names; fill them from a key made as in step 1.
3. Streamlit installs `app/requirements.txt` only (no torch, dbt or LightGBM).
   [Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud).

## 5. Preview each mode locally

```bash
python run.py demo                          # live (gcloud login) or saved copy
FORCE_MODE=snapshot python run.py demo      # "Saved copy" chip
FORCE_MODE=final python run.py demo         # final banner
python run.py snapshot                      # refresh the bundled copy in app/snapshot/ from BigQuery
```

Under `FORCE_MODE` the Trajectory tab shows the pre-rendered images, exactly as the public app does.

## 6. What happens on the expiry date

1. **Every day:** the workflow reads the earliest table expiry. With more than 3 days left it logs
   "N days until expiry, nothing to do".
2. **3 days or fewer before expiry:** it runs `python run.py snapshot --final`, which:
   - exports every table
   - checks the row counts
   - loads every tab of the app from the copy (AppTest)

   The workflow then pushes the copy to the `snapshot` branch.
3. **The app** sees `final: true` within an hour and switches to the final copy for good. Nobody needs to merge.
4. **The workflow also opens a PR** so `main` bundles the same copy. Merge it when convenient.
5. **After expiry:** the tables go away and nothing changes for viewers.
6. **If BigQuery is already gone and no final copy exists**, the workflow fails (red X, email). The app
   keeps showing the last bundled saved copy.

To save the final copy early: **Actions → Final snapshot → Run workflow → final = true**.

## 7. Backup hosting with Docker

```bash
docker build -t vse-demo .
docker run -p 8501:8501 vse-demo            # saved copy, no secrets
```

The image runs the bundled snapshot. For live data, add `-e FORCE_MODE=live` and mount a
`secrets.toml` at `/srv/.streamlit/secrets.toml`.
