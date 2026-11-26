"""Load configured LLM endpoint or deterministic offline fallback."""

from __future__ import annotations

from typing import Any

from src.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS


def load_llm_client() -> dict[str, Any]:
    """Return active LLM configuration without exposing credentials."""
    if LLM_API_URL and LLM_API_KEY:
        return {
            "mode": "api",
            "api_url": LLM_API_URL,
            "api_key": LLM_API_KEY,
            "model": LLM_MODEL,
            "timeout": LLM_TIMEOUT_SECONDS,
        }
    return {"mode": "deterministic_fallback", "model": "rule-based-json-router"}
