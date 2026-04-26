"""Node-level tests for the Reasoner subgraph (Plan 04-02).

Covers decision routing (5 cases), finalize_answer / finalize_feedback arg
extraction, and check_termination step-counter routing.

References:
  - 04-02-PLAN.md <behavior> for test_nodes.py (9 tests)
  - 04-RESEARCH.md Finding #8 (decision detection via tool_calls[0]['name'])
  - 04-RESEARCH.md Finding #4 (explicit step_count counter)
"""

from __future__ import annotations

import inspect

import pytest
from langchain_core.messages import AIMessage

from src.seeingeye.agents.reasoner.nodes import (
    REASONER_ADAPTER_PROMPT,
    ROUTE_ANSWER,
    ROUTE_CONTINUE,
    ROUTE_END,
    ROUTE_FEEDBACK,
    ROUTE_LOOP,
    _reasoner_adapter_prompt,
    check_termination_node,
    decision_router,
    finalize_answer_node,
    finalize_feedback_node,
    _render_reasoner_user_message,
    _normalize_answer_to_option,
    reasoner_model_node,
    should_continue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai_with_tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": "c0",
                "type": "tool_call",
            }
        ],
    )


def _state_with_last(msg) -> dict:
    return {"reasoner_messages": [msg]}


# ---------------------------------------------------------------------------
# decision_router (5 cases — covers all branches)
# ---------------------------------------------------------------------------


def test_decision_router_routes_answer_on_terminate_and_answer() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_answer",
        {"answer": "A", "confidence": "high", "reasoning": "x"},
    )
    assert decision_router(_state_with_last(msg)) == ROUTE_ANSWER


def test_decision_router_routes_answer_on_terminate_and_report_safety() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_report_safety",
        {
            "hazards": "exposed moving blade",
            "safest_next_action": "stop the machine",
            "safest_route": "step backward away from the machine",
            "route_risks": "moving parts remain exposed",
            "route_now_vs_after_clear": "route is blocked now but clear after the belt stops",
            "safest_solution": "de-energize and guard the blade",
            "confidence": "high",
            "reasoning": "guard missing",
        },
    )
    assert decision_router(_state_with_last(msg)) == ROUTE_ANSWER


def test_decision_router_routes_feedback_on_terminate_and_ask_translator() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_ask_translator",
        {"feedback": "more detail"},
    )
    assert decision_router(_state_with_last(msg)) == ROUTE_FEEDBACK


def test_decision_router_routes_continue_on_continue_reasoning() -> None:
    msg = _ai_with_tool_call("continue_reasoning", {"thought": "let me think"})
    assert decision_router(_state_with_last(msg)) == ROUTE_CONTINUE


def test_decision_router_routes_continue_on_no_tool_call() -> None:
    msg = AIMessage(content="just text", tool_calls=[])
    assert decision_router(_state_with_last(msg)) == ROUTE_CONTINUE


def test_decision_router_routes_continue_on_unknown_tool() -> None:
    msg = _ai_with_tool_call("not_a_real_tool", {"foo": "bar"})
    assert decision_router(_state_with_last(msg)) == ROUTE_CONTINUE


def test_reasoner_user_message_lists_valid_option_letters() -> None:
    text = _render_reasoner_user_message(
        "Which option is correct?",
        ["A. first", "B. second", "C. third"],
        "caption",
    )

    assert "Valid answer letters: A, B, C" in text
    assert "exactly one of these letters" in text


def test_reasoner_adapter_removes_python_execute_assumption() -> None:
    assert "do not have python_execute" in REASONER_ADAPTER_PROMPT
    assert "exactly one valid option letter" in REASONER_ADAPTER_PROMPT
    assert "terminate_and_report_safety" in REASONER_ADAPTER_PROMPT


def test_reasoner_adapter_adds_safety_workflow_when_enabled() -> None:
    class _StubSettings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "Please inspect the nearby area for hazards and identify the safest next action."
        video_reasoning_framework = "RSTR"

    prompt = _reasoner_adapter_prompt({"outer_iter": 1}, _StubSettings())

    assert "OSHA-style hazard identification workflow" in prompt
    assert "dynamic_route_v1 safety_navigation framework" in prompt
    assert "current_scene:" in prompt
    assert "dynamic_clear_scene:" in prompt
    assert "candidate_risk_scores:" in prompt
    assert "fallback:" in prompt
    assert "observed facts" in prompt
    assert "caught-in/between" in prompt
    assert "falling object" in prompt
    assert "overhead/suspended load" in prompt
    assert "terminate_and_report_safety" in prompt
    assert "route-risk model" in prompt
    assert "Predict the best route" in prompt
    assert "Do not inherit the first route candidate as final" in prompt
    assert "Avoid route fixation" in prompt
    assert "before/after dynamic-obstruction check" in prompt
    assert "what path may be revealed after it clears" in prompt
    assert "Score route candidates qualitatively" in prompt
    assert "falling-object/overhead-load exposure" in prompt
    assert "Do not select a route solely because its floor is wider" in prompt
    assert "current-state route safety" in prompt
    assert "guarded exit or turn" in prompt
    assert "exposed edge may be no-go" in prompt
    assert "possible path-control cues" in prompt
    assert "moving forward slightly and then turning left/right" in prompt
    assert "include it as a conditional candidate" in prompt
    assert "possible safest route after controls are restored" in prompt
    assert "wait for the temporary obstruction to clear" in prompt
    assert "egocentric step sequence" in prompt
    assert "Do not over-prioritize the shortest route" in prompt
    assert "do not finalize a route unless the SIR includes explicit route candidates" in prompt
    assert "turns behind temporary obstructions" in prompt
    assert "route now versus route after temporary obstructions clear" in prompt


def test_reasoner_adapter_later_pass_prioritizes_missing_hazard_details() -> None:
    class _StubSettings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "Please inspect the nearby area for hazards and identify the safest next action."
        video_reasoning_framework = "RSTR"

    prompt = _reasoner_adapter_prompt({"outer_iter": 2}, _StubSettings())

    assert "later outer-loop pass" in prompt
    assert "equipment energized state" in prompt


def test_reasoner_adapter_adds_rstr_for_video_tasks() -> None:
    class _StubSettings:
        analysis_mode = "default"
        video_reasoning_framework = "RSTR"

    prompt = _reasoner_adapter_prompt({"media_type": "video"}, _StubSettings())

    assert "RSTR / when-where-what decomposition" in prompt
    assert "WHEN:" in prompt
    assert "WHERE:" in prompt
    assert "WHAT:" in prompt


def test_normalize_answer_to_valid_option_letter() -> None:
    options = ["A. alpha", "B. beta", "C. gamma"]
    assert _normalize_answer_to_option("The answer is B.", options) == "B"
    assert _normalize_answer_to_option("D", options) == "D"


# ---------------------------------------------------------------------------
# finalize_answer / finalize_feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_answer_extracts_answer_arg() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_answer",
        {"answer": "B", "confidence": "high", "reasoning": "x"},
    )
    result = await finalize_answer_node(_state_with_last(msg))
    assert result["final_answer"] == "B"
    assert result["reasoner_messages"][0].tool_call_id == "c0"
    assert "FINAL ANSWER: B" in result["reasoner_messages"][0].content


@pytest.mark.asyncio
async def test_finalize_answer_formats_safety_report() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_report_safety",
        {
            "hazards": "unguarded pinch point",
            "safest_next_action": "step clear and stop the equipment",
            "safest_route": "move backward through the clear aisle",
            "route_risks": "avoid the exposed belt and wet floor",
            "route_now_vs_after_clear": "do not pass now; pass only after the belt is stopped and the aisle is clear",
            "current_scene": "machine running with narrow aisle",
            "dynamic_clear_scene": "aisle clears after belt stops",
            "route_candidates": "backward aisle, side aisle",
            "candidate_risk_scores": "backward aisle low after stop",
            "no_go_zones": "exposed belt zone",
            "verification_needed": "confirm lockout",
            "safest_now": "stop and step back",
            "safest_after_clear": "use backward aisle",
            "fallback": "wait for supervisor",
            "safest_solution": "lock out the machine and reinstall guarding",
            "confidence": "high",
            "reasoning": "hands are close to moving parts without a guard",
        },
    )
    result = await finalize_answer_node(_state_with_last(msg))

    assert "SAFETY REPORT" in result["final_answer"]
    assert "Current scene" in result["final_answer"]
    assert "Dynamic-clear scene" in result["final_answer"]
    assert "Candidate risk scores" in result["final_answer"]
    assert "Fallback" in result["final_answer"]
    assert "unguarded pinch point" in result["final_answer"]
    assert "Safest route" in result["final_answer"]
    assert "Route risks" in result["final_answer"]
    assert "Route now vs after clear" in result["final_answer"]
    assert "lock out the machine" in result["final_answer"]
    assert result["reasoner_messages"][0].tool_call_id == "c0"
    assert "Safest next action" in result["reasoner_messages"][0].content


@pytest.mark.asyncio
async def test_finalize_feedback_extracts_feedback_arg() -> None:
    msg = _ai_with_tool_call(
        "terminate_and_ask_translator", {"feedback": "need OCR"}
    )
    result = await finalize_feedback_node(_state_with_last(msg))
    assert result["reasoner_feedback"] == "need OCR"
    assert result["reasoner_messages"][0].tool_call_id == "c0"
    assert result["reasoner_messages"][0].content == "feedback: need OCR"


# ---------------------------------------------------------------------------
# check_termination via should_continue + MAX_ITERS
# ---------------------------------------------------------------------------


def test_check_termination_routes_end_at_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the Settings class used inside should_continue so n_r=3.
    import src.seeingeye.agents.reasoner.nodes as nodes_mod

    class _FakeSettings:
        n_r = 3

    monkeypatch.setattr(nodes_mod, "Settings", lambda: _FakeSettings())
    state = {"reasoner_step": 3, "reasoner_messages": []}
    assert should_continue(state) == ROUTE_END


def test_check_termination_routes_continue_below_max(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.seeingeye.agents.reasoner.nodes as nodes_mod

    class _FakeSettings:
        n_r = 3

    monkeypatch.setattr(nodes_mod, "Settings", lambda: _FakeSettings())
    state = {"reasoner_step": 1, "reasoner_messages": []}
    assert should_continue(state) == ROUTE_LOOP


@pytest.mark.asyncio
async def test_check_termination_node_is_passthrough() -> None:
    """check_termination_node itself returns no state change; routing is in should_continue."""
    result = await check_termination_node({"reasoner_step": 2, "reasoner_messages": []})
    assert result == {}


@pytest.mark.asyncio
async def test_check_termination_node_acknowledges_continue_tool_call() -> None:
    msg = _ai_with_tool_call("continue_reasoning", {"thought": "let me think"})
    result = await check_termination_node(
        {"reasoner_step": 1, "reasoner_messages": [msg]}
    )

    tool_messages = result["reasoner_messages"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c0"
    assert tool_messages[0].content == "continuing: let me think"


# ---------------------------------------------------------------------------
# Source-grep: reasoner_model_node uses bind_tools + tool_choice="auto" + .ainvoke
# (Pitfall #1 mitigations — see 04-RESEARCH.md §Finding #6 + Pitfall 1)
# ---------------------------------------------------------------------------


def test_reasoner_model_uses_bind_tools_with_tool_choice_auto() -> None:
    src = inspect.getsource(reasoner_model_node)
    assert "bind_tools" in src
    assert 'tool_choice="auto"' in src
    assert 'tool_choice="required"' not in src
    # Non-streaming requirement (Pitfall 1)
    assert ".astream(" not in src
    # Must call .ainvoke (async non-streaming)
    assert ".ainvoke(" in src
