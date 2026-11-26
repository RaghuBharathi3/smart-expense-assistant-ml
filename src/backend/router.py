"""Whitelisted backend router with Pydantic validation for Qwen2.5 & tool-call JSON."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.backend import tools
from src.classifier.predict import predict
from src.llm.respond import generate_response
from src.llm.router_llm import MONTH_MAP, route_query

ALLOWED_TOOLS: dict[str, Callable[..., Any]] = {
    "get_expenses": tools.get_expenses,
    "calculate_spending": tools.calculate_spending,
    "compare_months": tools.compare_months,
    "get_highest_category": tools.get_highest_category,
    "add_expense": tools.add_expense,
    "get_category_breakdown": tools.get_category_breakdown,
    "categorize_expense": predict,
}

INTENT_ALIASES: dict[str, str] = {
    "calculate_spending": "calculate_spending",
    "compare_months": "compare_months",
    "highest_category": "get_highest_category",
    "get_highest_category": "get_highest_category",
    "categorize_expense": "categorize_expense",
    "show_expenses": "get_expenses",
    "get_expenses": "get_expenses",
    "add_expense": "add_expense",
    "get_category_breakdown": "get_category_breakdown",
}

_LLM_HANDLER: Callable[[str], dict[str, Any] | str] | None = None


class QwenIntentPayload(BaseModel):
    """Pydantic schema validating experimental Qwen2.5-0.5B-Instruct JSON output."""

    model_config = ConfigDict(extra="ignore")

    intent: str
    category: str | None = None
    month: int | str | None = None
    month_a: int | str | None = None
    month_b: int | str | None = None
    months: list[int | str] | None = None
    year: int | None = Field(default=None, ge=2000, le=2100)
    amount: float | None = None
    description: str | None = None
    date: str | None = None
    expense_date: str | None = None


def register_llm_handler(handler: Callable[[str], dict[str, Any] | str] | None) -> None:
    """Register an external local LLM callable (e.g., Qwen2.5-0.5B in Colab)."""
    global _LLM_HANDLER
    _LLM_HANDLER = handler


def _parse_month_value(value: int | str | None) -> int | None:
    """Convert month integer, numeric string, or month name to 1-12."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if 1 <= value <= 12:
            return value
        raise ValueError(f"Month must be between 1 and 12, got {value}")
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned.isdigit():
            num = int(cleaned)
            if 1 <= num <= 12:
                return num
            raise ValueError(f"Month must be between 1 and 12, got {value}")
        if cleaned in MONTH_MAP:
            return MONTH_MAP[cleaned]
        for name, num in MONTH_MAP.items():
            if re.search(rf"\b{name}\b", cleaned):
                return num
        raise ValueError(f"Unrecognized month '{value}'")
    raise ValueError(f"Invalid month value: {value!r}")


def _extract_two_months(parsed: QwenIntentPayload) -> tuple[int, int]:
    """Resolve two month integers from Qwen compare_months variants."""
    if parsed.month_a is not None and parsed.month_b is not None:
        ma = _parse_month_value(parsed.month_a)
        mb = _parse_month_value(parsed.month_b)
        if ma and mb:
            return ma, mb

    if parsed.months and len(parsed.months) >= 2:
        ma = _parse_month_value(parsed.months[0])
        mb = _parse_month_value(parsed.months[1])
        if ma and mb:
            return ma, mb

    if isinstance(parsed.month, str):
        found: list[int] = []
        lower = parsed.month.lower()
        for name, num in MONTH_MAP.items():
            if re.search(rf"\b{name}\b", lower) and num not in found:
                found.append(num)
        if len(found) >= 2:
            return found[0], found[1]

    single = _parse_month_value(parsed.month)
    if single is not None:
        prev_month = 12 if single == 1 else single - 1
        return prev_month, single

    return 10, 11


def normalize_llm_payload(payload: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize Qwen intent JSON or function-call JSON into whitelisted tool format."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Tool payload must be a JSON object.")

    # Standard {"function": ..., "arguments": {...}} format
    if "function" in payload and "intent" not in payload:
        return payload

    if "intent" not in payload:
        raise ValueError("Payload must include 'intent' or 'function'.")

    try:
        parsed = QwenIntentPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid LLM JSON schema: {exc.errors()[0]['msg']}") from exc

    intent_key = parsed.intent.strip().lower()
    if intent_key not in INTENT_ALIASES:
        raise ValueError(f"Unsupported intent '{parsed.intent}'.")

    fn_name = INTENT_ALIASES[intent_key]
    args: dict[str, Any] = {}

    if fn_name in {"calculate_spending", "get_expenses"}:
        if parsed.category:
            args["category"] = parsed.category
        month_num = _parse_month_value(parsed.month)
        if month_num is not None:
            args["month"] = month_num
        if parsed.year is not None:
            args["year"] = parsed.year

    elif fn_name == "get_highest_category":
        month_num = _parse_month_value(parsed.month)
        if month_num is not None:
            args["month"] = month_num
        if parsed.year is not None:
            args["year"] = parsed.year

    elif fn_name == "compare_months":
        ma, mb = _extract_two_months(parsed)
        args["month_a"] = ma
        args["month_b"] = mb
        if parsed.year is not None:
            args["year"] = parsed.year

    elif fn_name == "categorize_expense":
        if not parsed.description or not parsed.description.strip():
            raise ValueError("categorize_expense requires a non-empty 'description'.")
        args["description"] = parsed.description.strip()

    elif fn_name == "add_expense":
        if not parsed.description or not parsed.description.strip():
            raise ValueError("add_expense requires a non-empty 'description'.")
        if parsed.amount is None:
            raise ValueError("add_expense requires a positive 'amount'.")
        args["description"] = parsed.description.strip()
        args["amount"] = parsed.amount
        args["expense_date"] = (parsed.expense_date or parsed.date or date.today().isoformat()).strip()
        if parsed.category:
            args["category"] = parsed.category

    return {"function": fn_name, "arguments": args}


def execute_tool_call(payload: dict[str, Any] | str) -> dict[str, Any]:
    """Validate LLM JSON payload against whitelist and execute predefined tool."""
    try:
        normalized = normalize_llm_payload(payload)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    fn_name = normalized.get("function")
    if not isinstance(fn_name, str) or fn_name not in ALLOWED_TOOLS:
        return {
            "ok": False,
            "error": f"Function '{fn_name}' is not whitelisted or supported.",
        }

    args = normalized.get("arguments", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object."}

    try:
        result = ALLOWED_TOOLS[fn_name](**args)
        return {"ok": True, "function": fn_name, "arguments": args, "result": result}
    except (TypeError, ValueError) as exc:
        return {"ok": False, "function": fn_name, "error": str(exc)}


def process_query(
    user_message: str,
    llm_json: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Backend entrypoint called by the interface (routes via Qwen/fallback -> tools)."""
    if llm_json is not None:
        tool_output = execute_tool_call(llm_json)
        return {**tool_output, "answer": generate_response(tool_output), "source": "qwen_json"}

    if not isinstance(user_message, str) or not user_message.strip():
        err = {"ok": False, "error": "Query cannot be empty."}
        return {**err, "answer": generate_response(err), "source": "validation"}

    routed: dict[str, Any] | str | None = None
    source = "fallback_router"

    if _LLM_HANDLER is not None:
        try:
            candidate = _LLM_HANDLER(user_message)
            tool_output = execute_tool_call(candidate)
            if tool_output.get("ok"):
                return {**tool_output, "answer": generate_response(tool_output), "source": "qwen_llm"}
        except Exception:  # noqa: BLE001
            pass

    routed = route_query(user_message)
    if not routed.get("function"):
        err = {"ok": False, "error": routed.get("error", "Unsupported query.")}
        return {**err, "answer": generate_response(err), "source": source}

    tool_output = execute_tool_call(routed)
    return {**tool_output, "answer": generate_response(tool_output), "source": source}
