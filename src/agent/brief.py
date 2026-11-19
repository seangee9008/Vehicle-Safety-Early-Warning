"""Gemini function-calling agent for cited vehicle safety investigation briefs."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Mapping
from datetime import date
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.agent import data_access
from src.agent.validate import validate_brief, validation_errors

ROOT = data_access.ROOT
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_TOOL_CALLS = 8  # shared by the first try and the one self-correction retry
PASSED_FIRST_TRY = "passed_first_try"
PASSED_AFTER_RETRY = "passed_after_retry"
STILL_FAILING = "still_failing"
COMPLAINT_LIMIT = 40
MAX_RETRIES_PER_MODEL = 2
# A request that hangs (seen Nov 18: one call never returned) counts like a 503.
REQUEST_TIMEOUT_MS = 120_000

load_dotenv(ROOT / ".env")

SYSTEM_INSTRUCTION = """You write concise vehicle-safety investigation briefs using only the supplied
complaint evidence and results from the available tools. Do not infer facts not supported by the
complaints. The evidence cutoff date in the prompt is a hard limit: do not use or mention any item
dated on or after it. Do not claim that a trend, rate, or comparison is measured unless the
complaint records provided establish it.

Return Markdown with exactly these sections, in this order:
## Failure pattern (2-3 sentences)
## Severity
## Evidence
## Watchlist reasons check
## Similar past recalls
## Suggested next step

Provide 3-6 Evidence bullets, never more. Every Evidence bullet must cite at least one complaint as
[ODINO 12345678]. Use only IDs returned by get_complaints, copied digit for digit. If citing
multiple complaints in one bullet, format each citation separately, like [ODINO 12345678],
[ODINO 87654321]. Severity values are computed by the application and supplied in the prompt;
repeat those counts without recalculating them.

Watchlist reasons check: the prompt lists the risk model's top reasons for flagging this row. Give
one bullet per reason saying whether the complaints SUPPORT it, CONTRADICT it, or it is CONTEXT
(not evidence of a defect, e.g. vehicle age). Base this only on the complaint records.

Similar past recalls: call find_similar_recalls once. Its result says which search was used
("same_make" or "all_makes"). Start the section with "Search: same make (<MAKE>)" or
"Search: all makes (no <MAKE> recalls for this part)". Then list relevant campaign IDs with a
few words on the defect, or write "none relevant" if the results are about different defects.
Keep suggestions clearly labeled as next steps, not as established facts."""


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
                "description": "Find earlier NHTSA recalls for the same part: the same make first, all makes only if the make has none.",
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


RETRYABLE_STATUS_CODES = frozenset({429, 503})
_FLASH_MODEL_RE = re.compile(r"^gemini-[\w.]+-flash(?:-lite)?(?:-[\w.]+)?$")
_EXCLUDED_MODEL_WORDS = ("image", "tts", "live", "audio", "embedding", "thinking", "computer")


class RequestBudgetExceeded(RuntimeError):
    """The run used its Gemini request budget (``set_request_budget``)."""


_REQUESTS = {"used": 0, "limit": None}
# Models whose free-tier daily quota ran out in this process; they are not tried again.
_EXHAUSTED_TODAY: set[str] = set()


def set_request_budget(limit: int | None) -> None:
    """Cap generate_content calls (retries included) for this process; None = no cap."""
    _REQUESTS["used"] = 0
    _REQUESTS["limit"] = limit


def requests_used() -> int:
    return int(_REQUESTS["used"])


def _is_daily_quota(error: Exception) -> bool:
    text = str(error).lower().replace("_", "").replace(" ", "")
    return _status_code(error) == 429 and "perday" in text


class ModelUnavailableError(RuntimeError):
    """A model kept returning 429/503, so the caller should try another model."""

    def __init__(self, model: str, status: int | None, error: Exception) -> None:
        super().__init__(f"{model} unavailable (HTTP {status}): {error}")
        self.model = model
        self.status = status


def _status_code(error: Exception) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _is_retryable(error: Exception) -> bool:
    return _status_code(error) in RETRYABLE_STATUS_CODES


def _generate_with_backoff(
    client: Any,
    *,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> Any:
    """Call one model, retrying 429/503/timeouts briefly before giving up on that model."""
    for attempt in range(MAX_RETRIES_PER_MODEL + 1):
        limit = _REQUESTS["limit"]
        if limit is not None and _REQUESTS["used"] >= limit:
            raise RequestBudgetExceeded(f"Gemini request budget of {limit} used up")
        _REQUESTS["used"] += 1
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except (APIError, httpx.TimeoutException) as error:
            if isinstance(error, APIError) and not _is_retryable(error):
                raise
            if _is_daily_quota(error):  # retrying today cannot help
                _EXHAUSTED_TODAY.add(model)
                raise ModelUnavailableError(model, _status_code(error), error) from error
            if attempt == MAX_RETRIES_PER_MODEL:
                raise ModelUnavailableError(model, _status_code(error), error) from error
            time.sleep(min(2**attempt + random.random(), 30.0))
    raise RuntimeError("Gemini request retry loop ended unexpectedly")


def _flash_sort_key(name: str) -> tuple[Any, ...]:
    version = re.match(r"gemini-(\d+(?:\.\d+)*)-", name)
    numbers = tuple(int(n) for n in version.group(1).split(".")) if version else ()
    numbers = (numbers + (0, 0, 0))[:3] if numbers else ()  # gemini-3 sorts after gemini-3.6
    return (
        not numbers,  # names without a version number (e.g. gemini-omni-flash) go last
        tuple(-n for n in numbers),
        "-lite" in name,
        "preview" in name or "exp" in name,
        name,
    )


_FALLBACK_CACHE: dict[int, list[str]] = {}


def fallback_models(client: Any) -> list[str]:
    """Flash / Flash-Lite text models the key can call, newest first.

    GEMINI_FALLBACK_MODELS (comma-separated) overrides the API model list.
    """
    configured = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
    if configured:
        return [name.strip() for name in configured.split(",") if name.strip()]
    names = []
    for model in client.models.list():
        name = str(getattr(model, "name", "") or "").removeprefix("models/")
        actions = getattr(model, "supported_actions", None)
        if actions is not None and "generateContent" not in actions:
            continue
        if not _FLASH_MODEL_RE.match(name):
            continue
        if any(word in name for word in _EXCLUDED_MODEL_WORDS):
            continue
        names.append(name)
    return sorted(set(names), key=_flash_sort_key)


def _cached_fallback_models(client: Any) -> list[str]:
    """The model list is fetched once per client, not once per brief."""
    key = id(client)
    if key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[key] = fallback_models(client)
    return _FALLBACK_CACHE[key]


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
    """Generate one markdown brief and its deterministic citation validation.

    Tries GEMINI_MODEL (or ``model_name``) first. If it keeps returning 429/503, the
    brief is restarted from scratch on the next Flash / Flash-Lite model, so each
    brief is written by exactly one model. ``validation["model_used"]`` names it.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if client is None and not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty; set it in the local .env file.")
    owns_client = client is None
    if client is None:
        client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        )
    primary = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        tried: list[str] = []
        candidates: list[str] = [primary]
        fallbacks_loaded = False
        while candidates:
            model_id = candidates.pop(0)
            if model_id in tried or model_id in _EXHAUSTED_TODAY:
                if model_id in _EXHAUSTED_TODAY and not fallbacks_loaded:
                    candidates.extend(_cached_fallback_models(client))
                    fallbacks_loaded = True
                continue
            tried.append(model_id)
            try:
                text, validation = _create_brief_with_client(item, client, model_id)
            except ModelUnavailableError as error:
                print(f"  {error.model} returned HTTP {error.status}; trying next model")
                if not fallbacks_loaded:
                    candidates.extend(_cached_fallback_models(client))
                    fallbacks_loaded = True
                continue
            return text, {**validation, "model_used": model_id}
        raise RuntimeError(f"All models unavailable (429/503): {', '.join(tried)}")
    finally:
        if owns_client:
            client.close()


def _converse(
    client: Any,
    model_id: str,
    contents: list[Any],
    config: Any,
    run_tool: Any,
    tool_budget: list[int],
) -> str:
    """Generate until the model returns text, answering its tool calls. Appends to contents."""
    while True:
        response = _generate_with_backoff(
            client, model=model_id, contents=contents, config=config
        )
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            text = getattr(response, "text", None)
            if text:
                return text
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
            return text
        if tool_budget[0] + len(function_parts) > MAX_TOOL_CALLS:
            raise RuntimeError(f"Gemini exceeded the {MAX_TOOL_CALLS}-call tool limit.")
        tool_responses = []
        for part in function_parts:
            call = part.function_call
            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": run_tool(call.name, dict(call.args or {}))},
                )
            )
            tool_budget[0] += 1
        contents.append(types.Content(role="user", parts=tool_responses))


def _retry_prompt(errors: list[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return f"""The application's validator rejected this brief:
{listed}

Rewrite the complete brief with the same sections, fixing exactly these problems. Keep every
other rule. Return only the corrected brief."""


def _parse_reasons(value: Any) -> list[str]:
    return [reason.strip() for reason in str(value or "").split(";") if reason.strip()]


def _create_brief_with_client(
    item: Mapping[str, Any],
    client: Any,
    model_id: str,
) -> tuple[str, dict[str, Any]]:
    make = str(_row_value(item, "make"))
    model = str(_row_value(item, "model"))
    model_year = int(_row_value(item, "model_year"))
    component = str(_row_value(item, "component"))
    month = _plain_date(_row_value(item, "month"))
    reasons = _parse_reasons(item.get("top_reasons"))
    # A watchlist row for month M uses complaints received by the end of M (contracts.yml).
    cutoff = data_access.evidence_cutoff(month)

    complaints = data_access.get_complaints(
        make, model, model_year, component, cutoff, limit=COMPLAINT_LIMIT
    )
    severity = data_access.severity_counts(complaints)
    prompt_item = {
        "month": month,
        "make": make,
        "model": model,
        "model_year": model_year,
        "component": component,
    }
    reason_lines = "\n".join(f"- {reason}" for reason in reasons) or "- (none given)"
    prompt = f"""Prepare a brief for this watchlist row:
{json.dumps(prompt_item, sort_keys=True)}

Watchlist month: {month}. Evidence cutoff (exclusive, end of that month): {cutoff}
Severity counts computed from the complaint rows by the application:
{json.dumps(severity, sort_keys=True)}

The risk model's top reasons for flagging this row (from SHAP; check them against the complaints):
{reason_lines}

Use get_complaints for evidence and find_similar_recalls for history. The application enforces
the target vehicle/component and cutoff even if a tool argument differs. Cite only ODINO IDs
returned by get_complaints. If no matching complaint rows are returned, state that explicitly.
"""
    recall_scopes: list[str] = []

    def run_tool(name: str, args: dict[str, Any]) -> Any:
        if name == "get_complaints":
            requested_limit = max(1, min(int(args.get("limit", COMPLAINT_LIMIT)), COMPLAINT_LIMIT))
            return data_access.get_complaints(
                make, model, model_year, component, cutoff, limit=requested_limit
            )
        if name == "find_similar_recalls":
            result = data_access.find_similar_recalls(
                make=make,
                component=component,
                keywords=args.get("keywords") or [],
                before_date=cutoff,
                limit=max(1, min(int(args.get("limit", 5)), 5)),
            )
            recall_scopes.append(result["scope"])
            return result
        raise RuntimeError(f"Gemini requested unknown tool {name!r}.")

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=TOOL_CONFIG,
        temperature=0.2,
    )
    tool_budget = [0]
    eligible = data_access.eligible_complaints(make, model, model_year, component, cutoff)

    text = _converse(client, model_id, contents, config, run_tool, tool_budget)
    validation = validate_brief(text, eligible, cutoff)
    first_errors = validation_errors(validation)
    status = PASSED_FIRST_TRY
    if first_errors:
        # Self-correction: one retry with the validator's exact errors, then validate again.
        contents.append(types.Content(role="user", parts=[types.Part(text=_retry_prompt(first_errors))]))
        text = _converse(client, model_id, contents, config, run_tool, tool_budget)
        validation = validate_brief(text, eligible, cutoff)
        status = PASSED_AFTER_RETRY if not validation_errors(validation) else STILL_FAILING
    return text.rstrip() + "\n", {
        **validation,
        "self_correction": status,
        "first_try_errors": " | ".join(first_errors),
        "recall_search": recall_scopes[-1] if recall_scopes else "not_called",
    }
