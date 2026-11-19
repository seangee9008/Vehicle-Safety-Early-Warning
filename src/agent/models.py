"""List Gemini models available to the local API key."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from src.agent.data_access import ROOT


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is empty; set it in the local .env file.")
    client = genai.Client(api_key=api_key)
    try:
        models = sorted(
            model.name
            for model in client.models.list()
            if getattr(model, "name", None)
        )
    finally:
        client.close()
    if not models:
        raise RuntimeError("The Gemini API returned no models for this key.")
    print("\n".join(models))


if __name__ == "__main__":
    main()
