"""Tests for LLMClient and ChatOpenAI factory functions.

These tests verify correct parameter routing for vLLM endpoints:
- top_k and repetition_penalty must go through extra_body (vLLM extensions)
- temperature and top_p must be top-level ChatOpenAI params
- No imports from agents/, graph/, tools/, or state modules
"""

import ast
import inspect
import os

import pytest
from langchain_openai import ChatOpenAI


@pytest.fixture(autouse=True)
def _clean_seeingeye_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEEINGEYE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SEEINGEYE_DISABLE_DOTENV", "1")


def test_translator_client_base_url_port_8000():
    """Test 1: create_translator_client() returns ChatOpenAI with port 8000."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(base_url="http://localhost:8000/v1")
    assert "8000" in client.openai_api_base


def test_reasoner_client_base_url_port_8001():
    """Test 2: create_reasoner_client() returns ChatOpenAI with port 8001."""
    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client(base_url="http://localhost:8001/v1")
    assert "8001" in client.openai_api_base


def test_translator_client_api_key_empty():
    """Test 3: create_translator_client() sets api_key='EMPTY'."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(base_url="http://localhost:8000/v1")
    assert client.openai_api_key.get_secret_value() == "EMPTY"


def test_temperature_top_level():
    """Test 4: ChatOpenAI instance has temperature=0.01 as top-level attribute."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert client.temperature == 0.01


def test_extra_body_contains_vllm_params():
    """Test 5: extra_body contains top_k and repetition_penalty."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(base_url="http://localhost:8000/v1")
    assert client.extra_body is not None
    assert client.extra_body["top_k"] == 1
    assert client.extra_body["repetition_penalty"] == 1.1


def test_top_k_not_top_level():
    """Test 6: top_k is NOT a top-level attribute on ChatOpenAI."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(base_url="http://localhost:8000/v1")
    assert not hasattr(client, "top_k")


def test_translator_model_name():
    """Test 7: Translator uses Qwen2.5-VL-3B-Instruct model."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert client.model_name == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_reasoner_model_name():
    """Test 8: Reasoner uses Qwen3-8B model."""
    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen3-8B",
    )
    assert client.model_name == "Qwen/Qwen3-8B"


def test_factory_accepts_overrides():
    """Test 9: Factory functions accept overrides for all defaults."""
    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="http://custom:9999/v1",
        model="custom-model",
        api_key="custom-key",
        temperature=0.5,
        top_p=0.9,
        max_tokens=2048,
        top_k=5,
        repetition_penalty=1.5,
    )
    assert "9999" in client.openai_api_base
    assert client.model_name == "custom-model"
    assert client.openai_api_key.get_secret_value() == "custom-key"
    assert client.temperature == 0.5
    assert client.top_p == 0.9
    assert client.max_tokens == 2048
    assert client.extra_body in (None, {})


def test_translator_client_uses_settings_env_overrides(monkeypatch):
    """Translator client defaults should come from Settings/env vars."""
    monkeypatch.setenv("SEEINGEYE_TRANSLATOR_BASE_URL", "http://env-host:9100/v1")
    monkeypatch.setenv("SEEINGEYE_TRANSLATOR_MODEL", "env-translator")
    monkeypatch.setenv("SEEINGEYE_TEMPERATURE", "0.2")
    monkeypatch.setenv("SEEINGEYE_TOP_P", "0.8")
    monkeypatch.setenv("SEEINGEYE_TOP_K", "7")
    monkeypatch.setenv("SEEINGEYE_REPETITION_PENALTY", "1.7")
    monkeypatch.setenv("SEEINGEYE_MAX_TOKENS", "1234")

    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client()
    assert client.openai_api_base == "http://env-host:9100/v1"
    assert client.model_name == "env-translator"
    assert client.temperature == 0.2
    assert client.top_p == 0.8
    assert client.max_tokens == 1234
    assert client.extra_body in (None, {})


def test_reasoner_client_uses_settings_env_overrides(monkeypatch):
    """Reasoner client defaults should come from Settings/env vars."""
    monkeypatch.setenv("SEEINGEYE_REASONER_BASE_URL", "http://env-host:9200/v1")
    monkeypatch.setenv("SEEINGEYE_REASONER_MODEL", "env-reasoner")

    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client()
    assert client.openai_api_base == "http://env-host:9200/v1"
    assert client.model_name == "env-reasoner"


def test_official_openai_compatible_endpoint_uses_env_api_key(monkeypatch):
    """Remote OpenAI-compatible endpoints should pick up env API keys."""
    monkeypatch.setenv("SEEINGEYE_API_KEY", "gemini-key")

    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
    )
    assert (
        client.openai_api_key.get_secret_value() == "gemini-key"
    )


def test_translator_client_prefers_role_specific_api_key(monkeypatch):
    """Translator/Reasoner can use different provider keys."""
    monkeypatch.setenv("SEEINGEYE_API_KEY", "shared-key")
    monkeypatch.setenv("SEEINGEYE_TRANSLATOR_API_KEY", "translator-key")

    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(base_url="https://api.k2think.ai/v1")
    assert client.openai_api_key.get_secret_value() == "translator-key"


def test_reasoner_client_prefers_role_specific_api_key(monkeypatch):
    """Reasoner can keep a separate OpenAI key when Translator uses another provider."""
    monkeypatch.setenv("SEEINGEYE_API_KEY", "shared-key")
    monkeypatch.setenv("SEEINGEYE_REASONER_API_KEY", "reasoner-key")

    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client(base_url="https://api.openai.com/v1")
    assert client.openai_api_key.get_secret_value() == "reasoner-key"


def test_remote_openai_compatible_endpoint_omits_vllm_extra_body(monkeypatch):
    """Remote OpenAI-style endpoints should not send vLLM-only extra_body fields."""
    monkeypatch.setenv("SEEINGEYE_API_KEY", "gemini-key")

    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client(
        base_url="https://example.com/v1",
        model="gemini-2.0-flash",
    )
    assert client.extra_body in (None, {})


def test_openai_gpt5_omits_top_p(monkeypatch):
    """OpenAI GPT-5 models reject top_p; omit it for the official endpoint."""
    monkeypatch.setenv("SEEINGEYE_API_KEY", "openai-key")

    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",
    )
    assert client.top_p is None


def test_openai_gpt5_uses_reasoning_effort_from_env(monkeypatch):
    """OpenAI GPT-5 clients should honor configured reasoning effort."""
    monkeypatch.setenv("SEEINGEYE_TRANSLATOR_REASONING_EFFORT", "xhigh")

    from src.seeingeye.llm.vllm_openai import create_translator_client

    client = create_translator_client(
        base_url="https://api.openai.com/v1",
        model="gpt-5.4-mini",
    )
    assert client.reasoning_effort == "xhigh"
    assert client.use_responses_api is True
    assert client.output_version == "responses/v1"


def test_non_openai_endpoint_ignores_reasoning_effort(monkeypatch):
    """Reasoning effort should not be forwarded to non-OpenAI GPT-5-like endpoints."""
    monkeypatch.setenv("SEEINGEYE_REASONER_REASONING_EFFORT", "xhigh")

    from src.seeingeye.llm.vllm_openai import create_reasoner_client

    client = create_reasoner_client(
        base_url="https://example.com/v1",
        model="gpt-5.4-mini",
    )
    assert client.reasoning_effort is None
    assert client.use_responses_api is False or client.use_responses_api is None
    assert client.output_version is None


def test_no_forbidden_imports():
    """Test 10: vllm_openai.py has zero imports from agents/, graph/, tools/, or state."""
    from src.seeingeye.llm import vllm_openai

    source_file = inspect.getfile(vllm_openai)
    with open(source_file, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    forbidden = {"agents", "graph", "tools", "state"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                assert not any(
                    p in forbidden for p in parts
                ), f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts = node.module.split(".")
                assert not any(
                    p in forbidden for p in parts
                ), f"Forbidden import found: from {node.module}"
