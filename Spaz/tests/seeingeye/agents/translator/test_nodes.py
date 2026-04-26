"""Unit tests for the Translator subgraph node functions.

Coverage (per 04-01 PLAN <behavior>):
  1. refine_sir_node calls SIR.update() (NOT SIR.replace) — verified by
     observing the ``--- UPDATED SIR ---`` separator in the result.
  2. refine_sir_node returns a dict with exactly ``{"sir": ...}`` (does
     not bleed into translator_messages or translator_step).
  3. should_continue routes to ROUTE_END when translator_step >= n_t.
  4. should_continue routes to ROUTE_CONTINUE when translator_step < n_t.
  5. route_after_model returns ROUTE_TOOLS when the latest AIMessage
     contains a parseable VCoT tool call.
  6. route_after_model returns ROUTE_DONE when the latest AIMessage is
     plain prose.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.seeingeye.agents.translator.nodes import (
    ROUTE_CONTINUE,
    ROUTE_DONE,
    ROUTE_END,
    ROUTE_TOOLS,
    TRANSLATOR_ADAPTER_PROMPT,
    _translator_adapter_prompt,
    _render_multiframe_user_content,
    _render_translator_user_message,
    _looks_like_navigation_task,
    _select_translator_model,
    check_termination_node,
    refine_sir_node,
    route_after_model,
    translator_model_node,
    should_continue,
)
from src.seeingeye.state.sir import SIR


@pytest.mark.asyncio
async def test_refine_sir_uses_update_not_replace() -> None:
    state = {
        "sir": SIR(content="prior caption"),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            ToolMessage(
                content="OCR result: 'foo bar'",
                tool_call_id="call_0",
                name="ocr",
            )
        ],
        "translator_step": 1,
    }

    result = await refine_sir_node(state)

    assert "sir" in result
    new_content = result["sir"].content
    # SIR.update appends with the literal '--- UPDATED SIR ---' separator
    # (see src/seeingeye/state/sir.py:56). SIR.replace would have wiped
    # the prior caption — its presence proves update() was called.
    assert new_content.startswith("prior caption")
    assert "--- UPDATED SIR ---" in new_content
    assert new_content.endswith("OCR result: 'foo bar'")


@pytest.mark.asyncio
async def test_refine_sir_returns_sir_only() -> None:
    state = {
        "sir": SIR(content="prior"),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            ToolMessage(
                content="tool output",
                tool_call_id="call_0",
                name="ocr",
            )
        ],
        "translator_step": 1,
    }

    result = await refine_sir_node(state)

    # Must update SIR only — must not also write translator_messages or
    # translator_step (those are owned by the model + termination nodes).
    assert set(result.keys()) == {"sir"}


def test_check_termination_routes_end_at_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force settings.n_t = 3 (the paper hyperparameter); should_continue
    # must route to ROUTE_END once translator_step has reached that bound.
    from src.seeingeye.agents.translator import nodes as nodes_mod

    class _StubSettings:
        n_t = 3
        max_iters = 3

    monkeypatch.setattr(nodes_mod, "Settings", lambda: _StubSettings())

    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [],
        "translator_step": 3,
    }

    assert should_continue(state) == ROUTE_END


def test_check_termination_routes_continue_below_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.seeingeye.agents.translator import nodes as nodes_mod

    class _StubSettings:
        n_t = 3
        max_iters = 3

    monkeypatch.setattr(nodes_mod, "Settings", lambda: _StubSettings())

    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [],
        "translator_step": 1,
    }

    assert should_continue(state) == ROUTE_CONTINUE


def test_route_after_model_returns_tools_when_parser_finds_call() -> None:
    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            AIMessage(
                content=(
                    '<tool_call>{"name": "ocr", '
                    '"arguments": {"image_path": "/x"}}</tool_call>'
                )
            )
        ],
        "translator_step": 1,
    }

    assert route_after_model(state) == ROUTE_TOOLS


def test_route_after_model_returns_done_when_parser_empty() -> None:
    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            AIMessage(content="Just plain prose, no tool call here.")
        ],
        "translator_step": 1,
    }

    assert route_after_model(state) == ROUTE_DONE


def test_route_after_model_ignores_placeholder_tool_name() -> None:
    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            AIMessage(
                content='<tool_call>{"name": "...", "arguments": {}}</tool_call>'
            )
        ],
        "translator_step": 1,
    }

    assert route_after_model(state) == ROUTE_DONE


def test_route_after_model_disables_tools_for_video() -> None:
    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "media_type": "video",
        "translator_messages": [
            AIMessage(
                content=(
                    '<tool_call>{"name": "ocr", '
                    '"arguments": {"image_path": "/x"}}</tool_call>'
                )
            )
        ],
        "translator_step": 1,
    }

    assert route_after_model(state) == ROUTE_DONE


@pytest.mark.asyncio
async def test_check_termination_updates_sir_from_direct_caption() -> None:
    state = {
        "sir": SIR(),
        "question": "q",
        "options": None,
        "image_b64": None,
        "translator_messages": [
            AIMessage(content="The frames show the wearer walking into a yard.")
        ],
        "translator_step": 1,
    }

    result = await check_termination_node(state)

    assert result["sir"].content == "The frames show the wearer walking into a yard."


def test_video_prompt_mentions_chronological_differences() -> None:
    text = _render_translator_user_message(
        "What happens?",
        None,
        "",
        media_type="video",
        frame_count=3,
    )

    assert "chronological frames" in text
    assert "Compare adjacent frames" in text
    assert "what happens in the video" in text
    assert "when-where-what" in text
    assert "WHERE key people/objects/paths/hazards" in text


def test_translator_user_message_includes_safety_scan_prompt() -> None:
    text = _render_translator_user_message(
        "What should I do next?",
        None,
        "",
        media_type="video",
        frame_count=2,
        safety_scan_prompt="Please inspect the nearby area for hazards and identify the safest next action.",
    )

    assert "Safety directive:" in text
    assert "inspect the nearby area for hazards" in text


def test_translator_user_message_includes_route_surface_scan() -> None:
    text = _render_translator_user_message(
        "Which way should I go?",
        None,
        "",
        route_surface_scan='{"best_conditional_candidate":"wait then turn right"}',
    )

    assert "Automatic route surface scan" in text
    assert "wait then turn right" in text


def test_navigation_task_detection_for_safety_mode() -> None:
    class _Settings:
        analysis_mode = "safety"

    assert _looks_like_navigation_task("Which way should I go?", _Settings())
    assert _looks_like_navigation_task("我应该怎么走到出口？", _Settings())
    assert not _looks_like_navigation_task("What color is the sign?", _Settings())


@pytest.mark.asyncio
async def test_translator_model_seeds_sir_with_route_surface_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.seeingeye.agents.translator import nodes as nodes_mod

    class _Settings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "scan"
        video_reasoning_framework = "RSTR"
        translator_model = "gpt-5.4-mini"
        translator_escalation_model = "gpt-5.4-mini"

    class _LLM:
        async def ainvoke(self, _msgs):
            return AIMessage(content="visual caption")

    async def _fake_route_scan(_state, _settings):
        return '{"candidate_ranking_not_final":"compare alternatives"}'

    monkeypatch.setattr(nodes_mod, "Settings", lambda: _Settings())
    monkeypatch.setattr(nodes_mod, "_automatic_route_surface_scan", _fake_route_scan)
    monkeypatch.setattr(nodes_mod, "create_translator_client", lambda model=None: _LLM())

    result = await translator_model_node(
        {
            "sir": SIR(),
            "question": "Which way should I go?",
            "options": None,
            "image_b64": "aaa",
            "translator_messages": [],
            "translator_step": 0,
            "media_type": "image",
        }
    )

    assert "AUTOMATIC ROUTE SURFACE SCAN" in result["sir"].content
    assert "candidate_ranking_not_final" in result["sir"].content
    assert result["translator_step"] == 1


def test_multiframe_user_content_includes_frame_timestamps() -> None:
    content = _render_multiframe_user_content(
        "Describe the video.",
        [
            {"b64": "aaa", "timestamp_s": 0.0, "mime_type": "image/jpeg"},
            {"b64": "bbb", "timestamp_s": 0.5, "mime_type": "image/jpeg"},
        ],
    )

    assert content[0] == {"type": "text", "text": "Describe the video."}
    assert content[1] == {"type": "text", "text": "Frame 1 at 0.000s"}
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,aaa"
    assert content[3] == {"type": "text", "text": "Frame 2 at 0.500s"}
    assert content[4]["image_url"]["url"] == "data:image/jpeg;base64,bbb"


def test_translator_adapter_prefers_direct_caption_for_openai_compatible_vlms() -> None:
    assert "Produce the visual description directly" in TRANSLATOR_ADAPTER_PROMPT
    assert "do not mention or call terminate_and_output_caption" in TRANSLATOR_ADAPTER_PROMPT


def test_translator_adapter_adds_safety_guidance_when_enabled() -> None:
    class _StubSettings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "Please inspect the nearby area for hazards and identify the safest next action."
        video_reasoning_framework = "RSTR"

    prompt = _translator_adapter_prompt({"outer_iter": 1}, _StubSettings())

    assert "OSHA-style hazard identification lens" in prompt
    assert "dynamic_route_v1 safety_navigation schema" in prompt
    assert "current_scene" in prompt
    assert "candidate_risk_scores" in prompt
    assert "fallback" in prompt
    assert "line-of-fire exposure" in prompt
    assert "overhead stored materials" in prompt
    assert "falling-object zones" in prompt
    assert "near-term safety incident" in prompt
    assert "distinct SAFETY EVIDENCE BLOCK" in prompt
    assert "ROUTE MEMORY BLOCK" in prompt
    assert "always add a ROUTE MEMORY BLOCK" in prompt
    assert "best_route_candidate" in prompt
    assert "visible_route_candidates" in prompt
    assert "route_after_temporary_obstruction_clears" in prompt
    assert "exit_openings_or_turns" in prompt
    assert "Do not lock onto the first apparently safe" in prompt
    assert "scene before and after it clears" in prompt
    assert "remaining static blockers" in prompt
    assert "overhead_hazards_along_each_path" in prompt
    assert "A wider floor path is not safest" in prompt
    assert "routes that require moving forward before turning" in prompt
    assert "side openings before, beside, or just beyond a blockage" in prompt
    assert "protected exit/turn" in prompt
    assert "controlled pedestrian exit/turn" in prompt
    assert "Do not assume a stacked object" in prompt
    assert "conditional_route_candidate" in prompt
    assert "no-go zones separately from conditional routes" in prompt
    assert "temporary dynamic blockage" in prompt
    assert "visible exit/opening near a barricade or guardrail" in prompt
    assert "move forward, then turn left/right" in prompt


def test_translator_adapter_later_pass_requests_hazard_critical_details() -> None:
    class _StubSettings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "Please inspect the nearby area for hazards and identify the safest next action."
        video_reasoning_framework = "RSTR"

    prompt = _translator_adapter_prompt({"outer_iter": 2}, _StubSettings())

    assert "later review pass" in prompt
    assert "missing/incorrect PPE" in prompt


def test_translator_adapter_adds_rstr_guidance_for_safety_video() -> None:
    class _StubSettings:
        analysis_mode = "safety"
        safety_framework = "OSHA"
        safety_navigation_schema = "dynamic_route_v1"
        safety_prediction_horizon = "near-term"
        safety_scan_prompt = "Please inspect the nearby area for hazards and identify the safest next action."
        video_reasoning_framework = "RSTR"

    prompt = _translator_adapter_prompt(
        {"outer_iter": 1, "media_type": "video"}, _StubSettings()
    )

    assert "RSTR style decomposition" in prompt
    assert "WHEN:" in prompt
    assert "WHERE:" in prompt
    assert "WHAT:" in prompt
    assert "option-relevant SIR" in TRANSLATOR_ADAPTER_PROMPT
    assert "Never use placeholder names" in TRANSLATOR_ADAPTER_PROMPT
    assert "For video frames, tools are disabled" in TRANSLATOR_ADAPTER_PROMPT
    assert "route_surface_scan" in TRANSLATOR_ADAPTER_PROMPT


def test_translator_model_selection_escalates_after_first_outer_iter() -> None:
    class _Settings:
        translator_model = "gpt-5.4-mini"
        translator_escalation_model = "gpt-5.4-mini"

    assert _select_translator_model({"outer_iter": 1}, _Settings()) == "gpt-5.4-mini"
    assert _select_translator_model({"outer_iter": 2}, _Settings()) == "gpt-5.4-mini"
