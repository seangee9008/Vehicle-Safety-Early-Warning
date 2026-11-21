# Backup hosting for the public demo (any Docker host), if Streamlit Cloud ever goes away.
#   docker build -t vse-demo . && docker run -p 8501:8501 vse-demo
# Runs on the bundled saved copy (app/snapshot/) by default: no secrets, no BigQuery needed.
# For live data instead: -e FORCE_MODE=live plus a mounted .streamlit/secrets.toml (DEPLOY.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 FORCE_MODE=snapshot
WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY app app
COPY src/__init__.py src/bq_reader.py src/
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
