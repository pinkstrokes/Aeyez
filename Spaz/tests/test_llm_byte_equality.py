"""Byte-equality smoke tests proving extra_body parameters reach the HTTP wire.

Uses httpx transport hooks to capture the actual JSON request body sent by
ChatOpenAI, without requiring a live vLLM server.  This is the gate that
catches silent parameter dropping before any live benchmark run.

Verifies:
- top_k and repetition_penalty (vLLM extensions via extra_body) are present
  at the top level of the request JSON body
- temperature and top_p are present as top-level fields
- max_tokens reaches the wire (as max_completion_tokens per OpenAI spec)
- model name is correct for each factory function
- messages array is well-formed
- Parameter fields are stable across different prompts
- Both Translator and Reasoner client configurations are correct
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage

from src.seeingeye.llm.vllm_openai import create_reasoner_client, create_translator_client


@pytest.fixture(autouse=True)
def _clean_seeingeye_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SEEINGEYE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SEEINGEYE_DISABLE_DOTENV", "1")


# ---------------------------------------------------------------------------
# httpx transport hook -- captures request bodies, returns mock completions
# ---------------------------------------------------------------------------


class RequestCapture(httpx.AsyncBaseTransport):
    """Async transport that records every request body and returns a mock 200."""

    def __init__(self) -> None:
        self.captured_requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.captured_requests.append({"url": str(request.url), "body": body})
        mock_response = {
            "id": "mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return httpx.Response(200, json=mock_response)


# ---------------------------------------------------------------------------
# Helper -- invoke a client factory and return the captured request body
# ---------------------------------------------------------------------------


async def capture_request(client_factory, prompt: str = "hello") -> dict:
    """Create a client via *client_factory*, invoke it, return the wire body.

    The factory function (create_translator_client / create_reasoner_client)
    returns a ChatOpenAI instance.  We then create a second ChatOpenAI with
    the same config plus our custom httpx transport for request capture.
    """
    # Build the client with default/paper params
    original = client_factory()

    # Create a capture transport and inject it into a new ChatOpenAI
    # that mirrors the original's configuration
    capture = RequestCapture()
    http_client = httpx.AsyncClient(transport=capture)

    from langchain_openai import ChatOpenAI

    client = ChatOpenAI(
        base_url=original.openai_api_base,
        api_key=original.openai_api_key.get_secret_value(),
        model=original.model_name,
        temperature=original.temperature,
        max_tokens=original.max_tokens,
        top_p=original.top_p,
        extra_body=original.extra_body,
        http_async_client=http_client,
    )

    await client.ainvoke([HumanMessage(content=prompt)])

    assert len(capture.captured_requests) == 1, "Expected exactly one request"
    return capture.captured_requests[0]["body"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_k_in_request_body():
    """Test 1: Captured request body contains top_k: 1."""
    body = await capture_request(
        lambda: create_translator_client(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-VL-3B-Instruct",
        )
    )
    assert body["top_k"] == 1, f"top_k missing or wrong: {body.get('top_k')}"


@pytest.mark.asyncio
async def test_repetition_penalty_in_request_body():
    """Test 2: Captured request body contains repetition_penalty: 1.1."""
    body = await capture_request(
        lambda: create_translator_client(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-VL-3B-Instruct",
        )
    )
    assert body["repetition_penalty"] == 1.1, (
        f"repetition_penalty missing or wrong: {body.get('repetition_penalty')}"
    )


@pytest.mark.asyncio
async def test_temperature_in_request_body():
    """Test 3: Captured request body contains temperature: 0.01."""
    body = await capture_request(
        lambda: create_translator_client(model="Qwen/Qwen2.5-VL-3B-Instruct")
    )
    assert body["temperature"] == 0.01, (
        f"temperature missing or wrong: {body.get('temperature')}"
    )


@pytest.mark.asyncio
async def test_top_p_in_request_body():
    """Test 4: Captured request body contains top_p: 0.001."""
    body = await capture_request(
        lambda: create_translator_client(model="Qwen/Qwen2.5-VL-3B-Instruct")
    )
    assert body["top_p"] == 0.001, f"top_p missing or wrong: {body.get('top_p')}"


@pytest.mark.asyncio
async def test_max_tokens_in_request_body():
    """Test 5: Captured request body contains max_tokens (as max_completion_tokens): 4096."""
    body = await capture_request(
        lambda: create_translator_client(model="Qwen/Qwen2.5-VL-3B-Instruct")
    )
    # langchain-openai sends max_tokens as max_completion_tokens (OpenAI API convention)
    effective = body.get("max_tokens") or body.get("max_completion_tokens")
    assert effective == 4096, (
        f"max_tokens/max_completion_tokens missing or wrong: "
        f"max_tokens={body.get('max_tokens')}, max_completion_tokens={body.get('max_completion_tokens')}"
    )


@pytest.mark.asyncio
async def test_translator_model_in_request_body():
    """Test 6: Captured request body contains model: gpt-5.4-mini."""
    body = await capture_request(create_translator_client)
    assert body["model"] == "gpt-5.4-mini", (
        f"model wrong: {body.get('model')}"
    )


@pytest.mark.asyncio
async def test_messages_contain_user_role():
    """Test 7: Captured request body messages array has at least one user message."""
    body = await capture_request(create_translator_client, prompt="test prompt")
    messages = body.get("messages", [])
    assert any(m.get("role") == "user" for m in messages), (
        f"No user message found in messages: {messages}"
    )


@pytest.mark.asyncio
async def test_param_stability_across_prompts():
    """Test 8: For 3 different prompts, parameter fields are identical (only content differs)."""
    prompts = ["prompt one", "prompt two", "prompt three"]
    bodies = []
    for prompt in prompts:
        body = await capture_request(create_translator_client, prompt=prompt)
        bodies.append(body)

    # Extract non-message keys and their values
    def param_signature(body: dict) -> dict:
        return {k: v for k, v in body.items() if k != "messages"}

    sig0 = param_signature(bodies[0])
    for i, body in enumerate(bodies[1:], start=2):
        sig = param_signature(body)
        assert sig == sig0, (
            f"Prompt {i} produced different params: {sig} vs {sig0}"
        )


@pytest.mark.asyncio
async def test_reasoner_model_and_params():
    """Test 9: Reasoner client request body has correct model and sampling params."""
    body = await capture_request(
        lambda: create_reasoner_client(
            base_url="http://localhost:8001/v1",
            model="Qwen/Qwen3-8B",
        )
    )
    assert body["model"] == "Qwen/Qwen3-8B", f"model wrong: {body.get('model')}"
    assert body["top_k"] == 1
    assert body["repetition_penalty"] == 1.1
    assert body["temperature"] == 0.01
    assert body["top_p"] == 0.001


@pytest.mark.asyncio
async def test_top_k_same_level_as_temperature():
    """Test 10: top_k is at the same JSON nesting level as temperature (top-level).

    This proves extra_body was merged into the top-level request body, not
    nested under a sub-key like 'extra_body' or 'extra'.
    """
    body = await capture_request(
        lambda: create_translator_client(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-VL-3B-Instruct",
        )
    )
    # Both must be top-level keys in the body dict
    assert "top_k" in body, f"top_k not in body keys: {sorted(body.keys())}"
    assert "temperature" in body, f"temperature not in body keys: {sorted(body.keys())}"
    # Verify they're at the same level (both are direct keys of the body dict)
    # If extra_body were nested, top_k would be at body["extra_body"]["top_k"]
    assert isinstance(body["top_k"], (int, float)), "top_k should be a scalar, not nested"
    assert isinstance(body["temperature"], (int, float)), "temperature should be a scalar"
