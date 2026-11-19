"""Publish the final small result tables to the BigQuery sandbox. Dry run unless --publish.

    python scripts/publish_bigquery.py [--publish] [--verify-only]

Logic, table list, guards and schemas: src/bq_publish.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bq_publish import PublishRefused, main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except PublishRefused as error:
        print(error, file=sys.stderr)
        sys.exit(2)
