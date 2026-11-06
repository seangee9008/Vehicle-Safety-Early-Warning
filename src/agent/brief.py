"""Gemini function-calling agent for cited vehicle safety investigation briefs."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping
from datetime import date
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.agent import data_access
from src.agent.validate import validate_brief

ROOT = data_access.ROOT
DEFAULT_MODEL = "gemini-3.8-flash"
MAX_TOOL_CALLS = 6
MAX_RATE_LIMIT_RETRIES = 5

load_dotenv(ROOT / ".env")

SYSTEM_INSTRUCTION = """You write concise vehicle-safety investigation briefs using only the supplied
complaint evidence and results from the available tools. Do not infer facts not supported by the
complaints. The watchlist month is the hard evidence cutoff: do not use or mention any item dated
on or after that month. Do not claim that a trend, rate, or comparison is measured unless the
complaint records provided establish it.

Return Markdown with exactly these sections:
## Failure pattern (2-3 sentences)
## Severity
## Evidence
## Similar past recalls
## Suggested next step

Provide 3-6 Evidence bullets. Every Evidence bullet must cite at least one complaint as
[ODINO 12345678]. Use only IDs returned by get_complaints. Severity values are computed by the application and supplied in the prompt;
repeat those counts without recalculating them. For similar recalls, list campaign IDs returned by
find_similar_recalls or write "none found". Keep suggestions clearly labeled as next steps, not as
established facts."""


def _schema(type_name: str, properties: dict[str, dict[str, str]], required: list[str]) -> dict[str, Any]:
    return {"type": type_name, "properties": properties, "required": required}


TOOL_CONFIG = [
    {
        "function_declarations": [
            {
                "name": "get_complaints",
                "description": "Get cleaned complaint evidence for the watchlist vehicle before its month.",
                "parameters": _schema(
                    "OBJECT",
                    {
                        "make": {"type": "STRING"},
                        "model": {"type": "STRING"},
                        "model_year": {"type": "INTEGER"},
                        "component": {"type": "STRING"},
                        "before_date": {"type": "STRING"},
                        "limit": {"type": "INTEGER"},
                    },
                    ["make", "model", "model_year", "component", "before_date"],
                ),
            },
            {
                "name": "find_similar_recalls",
                "description": "Find similar historical NHTSA recalls dated before the watchlist month.",
                "parameters": _schema(
                    "OBJECT",
                    {
                        "component": {"type": "STRING"},
                        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "before_date": {"type": "STRING"},
                        "limit": {"type": "INTEGER"},
                    },
                    ["component", "before_date"],
                ),
            },
        ]
    }
]


def _is_rate_limited(error: Exception) -> bool:
    return (
        getattr(error, "code", None) == 429
        or getattr(error, "status_code", None) == 429
        or "429" in str(error)
    )


def _generate_with_backoff(
    client: Any,
    *,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> Any:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except APIError as error:
            if not _is_rate_limited(error) or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = min(2**attempt + random.random(), 30.0)
            time.sleep(delay)
    raise RuntimeError("Gemini request retry loop ended unexpectedly")


def _plain_date(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _row_value(item: Mapping[str, Any], key: str) -> Any:
    value = item.get(key)
    if value is None or value == "":
        raise ValueError(f"Watchlist row is missing required field {key!r}")
    return value


def create_brief(
    item: Mapping[str, Any],
    *,
    client: Any | None = None,
    model_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate one markdown brief and its deterministic citation validation."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if client is None and not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty; set it in the local .env file.")
    owns_client = client is None
    if client is None:
        client = genai.Client(api_key=api_key)
    try:
        return _create_brief_with_client(item, client, model_name)
    finally:
        if owns_client:
            client.close()


def _create_brief_with_client(
    item: Mapping[str, Any],
    client: Any,
    model_name: str | None,
) -> tuple[str, dict[str, Any]]:
    make = str(_row_value(item, "make"))
    model = str(_row_value(item, "model"))
    model_year = int(_row_value(item, "model_year"))
    component = str(_row_value(item, "component"))
    cutoff = _plain_date(_row_value(item, "month"))
    model_id = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    complaints = data_access.get_complaints(
        make, model, model_year, component, cutoff, limit=40
    )
    severity = data_access.severity_counts(complaints)
    prompt_item = {
        "month": cutoff,
        "make": make,
        "model": model,
        "model_year": model_year,
        "component": component,
    }
    prompt = f"""Prepare a brief for this watchlist row:
{json.dumps(prompt_item, sort_keys=True)}

Watchlist month (exclusive evidence cutoff): {cutoff}
Severity counts computed from the complaint rows by the application:
{json.dumps(severity, sort_keys=True)}

Use get_complaints for evidence and find_similar_recalls for history. The application enforces
the target vehicle/component and cutoff even if a tool argument differs. Cite only ODINO IDs
returned by get_complaints. If no matching complaint rows are returned, state that explicitly.
"""
    contents = [
        types.Content(role="user", parts=[types.Part(text=prompt)]),
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=TOOL_CONFIG,
        temperature=0.2,
    )
    tool_calls = 0

    while True:
        response = _generate_with_backoff(
            client, model=model_id, contents=contents, config=config
        )
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            text = getattr(response, "text", None)
            if text:
                break
            raise RuntimeError("Gemini returned no candidate or text.")
        model_content = candidates[0].content
        contents.append(model_content)
        function_parts = [
            part for part in model_content.parts if getattr(part, "function_call", None)
        ]
        if not function_parts:
            text = getattr(response, "text", None) or "\n".join(
                part.text for part in model_content.parts if getattr(part, "text", None)
            )
            if not text:
                raise RuntimeError("Gemini returned an empty brief.")
            break
        if tool_calls + len(function_parts) > MAX_TOOL_CALLS:
            raise RuntimeError(f"Gemini exceeded the {MAX_TOOL_CALLS}-call tool limit.")

        tool_responses = []
        for part in function_parts:
            call = part.function_call
            args = dict(call.args or {})
            requested_limit = max(1, min(int(args.get("limit", 40)), 40))
            if call.name == "get_complaints":
                result = data_access.get_complaints(
                    make, model, model_year, component, cutoff, limit=requested_limit
                )
            elif call.name == "find_similar_recalls":
                result = data_access.find_similar_recalls(
                    component=component,
                    keywords=args.get("keywords") or [],
                    before_date=cutoff,
                    limit=max(1, min(int(args.get("limit", 5)), 5)),
                )
            else:
                raise RuntimeError(f"Gemini requested unknown tool {call.name!r}.")
            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )
            tool_calls += 1
        contents.append(types.Content(role="user", parts=tool_responses))

    validation = validate_brief(text, complaints, cutoff)
    return text.rstrip() + "\n", validation
