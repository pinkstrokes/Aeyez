"""Tests for AGT-04 force-answer node (Plan 04-02 Task 2).

Covers signature, single-tool bind structural enforcement (Finding #5
deviation), prompt usage, and behavioural extraction (with mocked LLM).

The shared fixtures (``populated_sir``, ``sample_question``,
``sample_options``) come from ``tests/seeingeye/agents/conftest.py`` (created
by Plan 04-01 in the same wave).
"""

from __future__ import annotations

import inspect

import pytest
from langchain_core.messages import AIMessage

from src.seeingeye.agents.reasoner.force_answer import force_answer_node
from src.seeingeye.agents.reasoner.force_answer import (
    _normalize_answer_to_option,
    _render_reasoner_user_message,
)


# ---------------------------------------------------------------------------
# Fake LLM that asserts only-one-tool-bound and returns a canned response
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, response):
        self._response = response

    def bind_tools(self, tools, tool_choice="auto"):  # noqa: D401
        # Structural enforcement of Finding #5 deviation: 1 tool, not 3.
        assert len(tools) == 1, f"force_answer must bind exactly 1 tool, got {len(tools)}"
        assert tool_choice == "auto", f"must use tool_choice='auto', got {tool_choice!r}"
        return self

    async def ainvoke(self, messages):
        return self._response


@pytest.fixture
def fake_llm_factory(monkeypatch: pytest.MonkeyPatch):
    """Patches ``create_reasoner_client`` inside the force_answer module."""

    def _make(response):
        def _factory(*args, **kwargs):
            return _FakeLLM(response)

        monkeypatch.setattr(
            "src.seeingeye.agents.reasoner.force_answer.create_reasoner_client",
            _factory,
        )

    return _make


# ---------------------------------------------------------------------------
# Signature + structural source greps
# ---------------------------------------------------------------------------


def test_force_answer_node_signature() -> None:
    assert inspect.iscoroutinefunction(force_answer_node)
    sig = inspect.signature(force_answer_node)
    # Single positional/keyword parameter.
    params = list(sig.parameters.values())
    assert len(params) == 1, f"force_answer_node must take 1 param, got {len(params)}"


def test_force_answer_binds_only_terminate_and_answer() -> None:
    src = inspect.getsource(force_answer_node)
    # Finding #5 deviation: bind ONLY terminate_and_answer (1 tool, not 3).
    assert "terminate_and_answer" in src
    assert "terminate_and_ask_translator" not in src
    assert "continue_reasoning" not in src


def test_force_answer_uses_final_iteration_prompt() -> None:
    src = inspect.getsource(force_answer_node)
    assert "FINAL_ITERATION_PROMPT" in src

    # Module-level import must reference the canonical prompt module.
    import src.seeingeye.agents.reasoner.force_answer as fa_mod

    mod_src = inspect.getsource(fa_mod)
    assert "from src.seeingeye.prompts.force_answer import FINAL_ITERATION_PROMPT" in mod_src


def test_force_answer_does_not_use_streaming() -> None:
    src = inspect.getsource(force_answer_node)
    # Pitfall #1: vLLM hermes streaming bug. Must use .ainvoke, not .astream.
    assert ".astream(" not in src


def test_force_answer_uses_tool_choice_auto() -> None:
    src = inspect.getsource(force_answer_node)
    # Pitfall #1: vLLM hermes parser bug. Must use "auto", not "required".
    assert 'tool_choice="auto"' in src
    assert 'tool_choice="required"' not in src


def test_force_answer_user_message_lists_valid_option_letters() -> None:
    text = _render_reasoner_user_message(
        "Q?",
        ["A. alpha", "B. beta", "C. gamma"],
        "caption",
    )

    assert "Valid answer letters: A, B, C" in text
    assert "exactly one of these letters" in text


def test_force_answer_normalizes_answer_to_valid_option_letter() -> None:
    assert _normalize_answer_to_option("Option C", ["A. x", "C. y"]) == "C"


# ---------------------------------------------------------------------------
# Behavioural tests (with patched LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_answer_returns_final_answer_key(
    fake_llm_factory, populated_sir, sample_question, sample_options
) -> None:
    fake_llm_factory(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "terminate_and_answer",
                    "args": {
                        "answer": "C",
                        "confidence": "high",
                        "reasoning": "x",
                    },
                    "id": "c0",
                    "type": "tool_call",
                }
            ],
        )
    )
    state = {
        "sir": populated_sir,
        "question": sample_question,
        "options": sample_options,
    }
    result = await force_answer_node(state)
    assert result == {"final_answer": "C"}


@pytest.mark.asyncio
async def test_force_answer_handles_no_tool_call_fallback(
    fake_llm_factory, populated_sir, sample_question, sample_options
) -> None:
    # Per 04-RESEARCH.md §Pattern 3: if model emits text without a tool call,
    # fall back to using the message content as the final answer.
    fake_llm_factory(AIMessage(content="The answer is D", tool_calls=[]))
    state = {
        "sir": populated_sir,
        "question": sample_question,
        "options": sample_options,
    }
    result = await force_answer_node(state)
    assert result == {"final_answer": "D"}
