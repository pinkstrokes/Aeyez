"""ChatOpenAI factory functions for vLLM-hosted Translator and Reasoner endpoints.

vLLM serves an OpenAI-compatible API but accepts additional sampling parameters
(top_k, repetition_penalty, etc.) via the ``extra_body`` field. These MUST NOT
be passed as top-level kwargs to ChatOpenAI -- they would be silently dropped by
the OpenAI client, causing parity failure with no error message.

Standard OpenAI parameters (temperature, top_p, max_tokens) are top-level.
When callers omit values, defaults come from :class:`seeingeye.config.Settings`,
so ``SEEINGEYE_*`` env vars actually affect runtime behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

from src.seeingeye.config.settings import Settings


def _with_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _host(base_url: str) -> str:
    return (urlparse(base_url).hostname or "").lower()


def _uses_local_vllm(base_url: str) -> bool:
    host = _host(base_url)
    return host in {"localhost", "127.0.0.1"}


def _uses_openai_gpt5(base_url: str, model: str) -> bool:
    host = _host(base_url)
    return host == "api.openai.com" and model.lower().startswith("gpt-5")


def _resolve_reasoning_effort(
    base_url: str,
    model: str,
    explicit: str | None,
    fallback: str | None,
) -> str | None:
    value = explicit if explicit is not None else fallback
    if not value:
        return None
    if not _uses_openai_gpt5(base_url, model):
        return None
    return value


def _use_openai_responses_api(
    base_url: str,
    model: str,
    reasoning_effort: str | None,
) -> bool:
    return _uses_openai_gpt5(base_url, model) and bool(reasoning_effort)


def _qwen3_enable_thinking() -> bool:
    value = os.getenv("SEEINGEYE_QWEN3_ENABLE_THINKING", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_api_key(
    base_url: str,
    api_key: str | None,
    *,
    role_env_var: str | None = None,
) -> str:
    if api_key:
        return api_key
    if _uses_local_vllm(base_url):
        return "EMPTY"
    if role_env_var and os.getenv(role_env_var):
        return os.getenv(role_env_var) or "EMPTY"
    env_key = (
        os.getenv("SEEINGEYE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if env_key:
        return env_key
    env_path = Path(".env")
    if env_path.exists():
        values = dotenv_values(env_path)
        if role_env_var and values.get(role_env_var):
            return values.get(role_env_var) or "EMPTY"
        return (
            values.get("SEEINGEYE_API_KEY")
            or values.get("OPENAI_API_KEY")
            or values.get("OPENROUTER_API_KEY")
            or "EMPTY"
        )
    return "EMPTY"


def _build_extra_body(
    base_url: str,
    *,
    model: str,
    top_k: int,
    repetition_penalty: float,
) -> dict[str, Any] | None:
    if _uses_local_vllm(base_url):
        return {
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

    extra_body: dict[str, Any] = {}
    # DashScope-compatible Qwen3 chat endpoints require non-streaming requests
    # to set thinking explicitly, otherwise they return HTTP 400.
    if model.lower().startswith("qwen3"):
        extra_body["enable_thinking"] = _qwen3_enable_thinking()
    return extra_body or None


def create_translator_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client for the Translator (Qwen2.5-VL-3B on vLLM port 8000).

    ``top_k`` and ``repetition_penalty`` are vLLM sampling extensions routed
    through ``extra_body``.  ``temperature`` and ``top_p`` are standard OpenAI
    parameters passed as top-level constructor kwargs.
    """
    settings = Settings()
    resolved_base_url = _with_default(base_url, settings.translator_base_url)
    resolved_top_k = _with_default(top_k, settings.top_k)
    resolved_repetition_penalty = _with_default(
        repetition_penalty, settings.repetition_penalty
    )
    resolved_model = _with_default(model, settings.translator_model)
    resolved_reasoning_effort = _resolve_reasoning_effort(
        resolved_base_url,
        resolved_model,
        reasoning_effort,
        settings.translator_reasoning_effort,
    )
    resolved_top_p = (
        None
        if _uses_openai_gpt5(resolved_base_url, resolved_model)
        else _with_default(top_p, settings.top_p)
    )
    return ChatOpenAI(
        base_url=resolved_base_url,
        api_key=_resolve_api_key(
            resolved_base_url,
            api_key,
            role_env_var="SEEINGEYE_TRANSLATOR_API_KEY",
        ),
        model=resolved_model,
        reasoning_effort=resolved_reasoning_effort,
        use_responses_api=_use_openai_responses_api(
            resolved_base_url, resolved_model, resolved_reasoning_effort
        ),
        output_version=(
            "responses/v1"
            if _use_openai_responses_api(
                resolved_base_url, resolved_model, resolved_reasoning_effort
            )
            else None
        ),
        temperature=_with_default(temperature, settings.temperature),
        max_tokens=_with_default(max_tokens, settings.max_tokens),
        top_p=resolved_top_p,
        extra_body=_build_extra_body(
            resolved_base_url,
            model=resolved_model,
            top_k=resolved_top_k,
            repetition_penalty=resolved_repetition_penalty,
        ),
    )


def create_reasoner_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client for the Reasoner (Qwen3-8B on vLLM port 8001).

    Same parameter routing as the Translator -- vLLM extensions in ``extra_body``,
    standard params top-level.
    """
    settings = Settings()
    resolved_base_url = _with_default(base_url, settings.reasoner_base_url)
    resolved_top_k = _with_default(top_k, settings.top_k)
    resolved_repetition_penalty = _with_default(
        repetition_penalty, settings.repetition_penalty
    )
    resolved_model = _with_default(model, settings.reasoner_model)
    resolved_reasoning_effort = _resolve_reasoning_effort(
        resolved_base_url,
        resolved_model,
        reasoning_effort,
        settings.reasoner_reasoning_effort,
    )
    resolved_top_p = (
        None
        if _uses_openai_gpt5(resolved_base_url, resolved_model)
        else _with_default(top_p, settings.top_p)
    )
    return ChatOpenAI(
        base_url=resolved_base_url,
        api_key=_resolve_api_key(
            resolved_base_url,
            api_key,
            role_env_var="SEEINGEYE_REASONER_API_KEY",
        ),
        model=resolved_model,
        reasoning_effort=resolved_reasoning_effort,
        use_responses_api=_use_openai_responses_api(
            resolved_base_url, resolved_model, resolved_reasoning_effort
        ),
        output_version=(
            "responses/v1"
            if _use_openai_responses_api(
                resolved_base_url, resolved_model, resolved_reasoning_effort
            )
            else None
        ),
        temperature=_with_default(temperature, settings.temperature),
        max_tokens=_with_default(max_tokens, settings.max_tokens),
        top_p=resolved_top_p,
        extra_body=_build_extra_body(
            resolved_base_url,
            model=resolved_model,
            top_k=resolved_top_k,
            repetition_penalty=resolved_repetition_penalty,
        ),
    )
