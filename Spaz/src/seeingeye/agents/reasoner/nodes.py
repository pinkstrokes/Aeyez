"""Reasoner subgraph nodes + routers (Plan 04-02 / AGT-03).

Topology (Pattern 2 in 04-RESEARCH.md):

    START
      |
      v
    reasoner_model  <-- (loops from check_termination)
      | (conditional on tool_calls[0]['name'])
      |-- terminate_and_answer          --> finalize_answer   --> END
      |-- terminate_and_ask_translator  --> finalize_feedback --> END
      `-- continue_reasoning OR none    --> check_termination
                                            | (conditional on reasoner_step)
                                            |-- continue --> reasoner_model
                                            `-- end      --> END

Pitfall #1 mitigations (LOCKED):
  - ``tool_choice="auto"`` (NOT ``"required"``) — vLLM hermes parser bug.
  - ``.ainvoke`` (NOT ``.astream``) — vLLM hermes streaming bug.

Decision routing (Finding #6 / #8):
  Inspect ``AIMessage.tool_calls[0]['name']`` directly — clean structural
  routing, NOT brittle regex on ``msg.content`` like the old code did.

Messages wipe per outer iteration: subgraph-PRIVATE key, fresh on each
invocation. Mirrors old ``_reset_agent_memory_for_iteration()`` in
``src/multi-agent/app/flow/iterative_refinement.py:56-71``.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.seeingeye.agents.reasoner.state import ReasonerSubgraphState
from src.seeingeye.config.settings import Settings
from src.seeingeye.llm.vllm_openai import create_reasoner_client
from src.seeingeye.prompts import reasoner as reasoner_prompts
from src.seeingeye.tools.decisions import (
    continue_reasoning,
    terminate_and_answer,
    terminate_and_ask_translator,
    terminate_and_report_safety,
)

# Edge key constants (Pitfall 5 — no inline string literals in conditional edges).
ROUTE_ANSWER = "answer"
ROUTE_FEEDBACK = "feedback"
ROUTE_CONTINUE = "continue"
ROUTE_LOOP = "loop"
ROUTE_END = "end"

# The 3 Reasoner decision @tool functions, bound together (D-04 / D-09).
_DECISION_TOOLS = [
    terminate_and_answer,
    terminate_and_report_safety,
    terminate_and_ask_translator,
    continue_reasoning,
]

REASONER_ADAPTER_PROMPT = """Runtime adapter:
- You do not have python_execute in this runtime. Do any needed arithmetic yourself in text.
- For multiple-choice questions, the answer argument to terminate_and_answer must be exactly one valid option letter from the provided options, with no extra words.
- For safety-oriented tasks, use terminate_and_report_safety when you can state the safety_navigation schema, hazards, the safest next action, the safest route, route-specific risks, and the safest overall solution supported by the SIR.
- If the SIR lacks a number, label, table cell, region, or relation needed to distinguish valid options, call terminate_and_ask_translator with a precise visual request.
- Before answering, compare each plausible option against the SIR and rule out mismatches.
- For spatial tasks, require a candidate-by-candidate check: viewpoint/frame, support/contact, occlusion, path/shape alternatives, geometric consistency, and local zoom evidence.
- For construction/engineering safety, require mechanics hazard checks: load path, supports, stored energy, likely failure motion, line-of-fire, pinch/crush/shear, and safe standoff/route.
- For diagram/math tasks, separate visual extraction from calculation: list visible labels/units/geometry first, then compute or compare options. Ask the translator for spatial_intelligence_scan evidence if the SIR lacks crop-level detail."""


def _reasoner_adapter_prompt(
    state: ReasonerSubgraphState,
    settings: Settings,
) -> str:
    prompt = REASONER_ADAPTER_PROMPT
    if state.get("media_type") == "video":
        prompt += f"""
- For video tasks, use the {settings.video_reasoning_framework} / when-where-what decomposition before deciding:
  WHEN: which frames, timestamps, or moments contain the relevant change.
  WHERE: where the key people, objects, paths, rooms, or hazards are located relative to each other.
  WHAT: what action, interaction, state transition, or developing risk is occurring.
- If any of WHEN, WHERE, or WHAT is underspecified, ask the translator for that exact missing evidence."""

    if settings.analysis_mode.strip().lower() != "safety":
        return prompt

    prompt += f"""
- Safety analysis mode is active. Use an {settings.safety_framework}-style hazard identification workflow grounded in the SIR.
- Use the {settings.safety_navigation_schema} safety_navigation framework for any movement, navigation, access, egress, route, or "which way should I go" task.
- Follow this fixed instruction exactly as an additional task anchor: "{settings.safety_scan_prompt}"
- First separate: (1) observed facts, (2) inferred hazards, and (3) missing evidence.
- Build a hazard inventory using common occupational categories when relevant: struck-by, falling object, overhead/suspended load, caught-in/between, line-of-fire, falls, slips/trips, electrical, thermal, chemical, fire/explosion, vehicle/mobile equipment, stored energy, ergonomic strain, blocked egress, poor housekeeping, missing PPE, and missing guarding/barriers.
- Build a route-risk model when the scene involves movement: current position, destination/safe zone, exits, passable paths, obstacles, blocked paths, blind corners, moving equipment, unstable surfaces, and no-go zones.
- Build the safety_navigation schema before answering:
  current_scene: current walkable surface, people/equipment, static blockers, dynamic blockers, occlusions, and active hazards.
  dynamic_clear_scene: what changes after movable blockers clear, what space/route may be revealed, and what static hazards remain.
  route_candidates: all plausible routes, including wait/stop, step back, forward, left, right, forward-then-turn, alternate exit, and no-go.
  candidate_risk_scores: qualitative low/medium/high scores with reasons for immediate passability, after-clear passability, edge/fall risk, trip/slip risk, struck-by/falling-object risk, overhead/suspended-load exposure, clearance, visibility, footing, and retreat path.
  no_go_zones: areas to avoid under current conditions.
  verification_needed: checks required before using uncertain routes.
  safest_now: immediate action in the current scene.
  safest_after_clear: best route after dynamic blockers clear and verification passes.
  fallback: backup route/action if verification fails.
- Build an ego-centered route map before choosing movement:
  current_position, forward_path, left_candidate, right_candidate, blocked_now, passable_after_clear, no_go_zones, overhead_hazards, line_of_fire_zones, retreat_path, safest_next_step.
- Build a mechanics hazard model before final OSHA judgment:
  load_path, supports_and_connections, unstable_or_deforming_objects, stored_energy_sources, likely_failure_modes, exposed_people_or_routes, safe_standoff_or_route, verification_needed.
- Predict the best route by comparing candidate paths, minimizing exposure to hazards, preserving escape options, avoiding line-of-fire and moving equipment, and preferring clear, visible, stable paths.
- Do not inherit the first route candidate as final. Treat route-scan ranking as provisional and compare all visible alternatives before choosing.
- Avoid route fixation: if a center/left/right lane looks safer at first, still evaluate side openings before, beside, or just beyond obstructions and dynamic blockers.
- Run a before/after dynamic-obstruction check: for each worker, pedestrian, vehicle, cart, carried load, swinging object, door, or movable material, ask what path or opening is blocked now and what path may be revealed after it clears.
- Choose routes from the after-clear scene only when the blocker is plausibly temporary and the remaining static hazards can be verified or controlled. Keep the now-scene recommendation as stop/wait when immediate movement is unsafe.
- Score route candidates qualitatively by: immediate passability, future passability after temporary obstructions clear, fall/edge exposure, trip/slip risk, struck-by/line-of-fire exposure, falling-object/overhead-load exposure, clearance, visibility around turns, stability of footing, and availability of a retreat path.
- Do not select a route solely because its floor is wider or clearer. A route with overhead stacked pipes/materials, suspended loads, low protrusions, or likely falling-object exposure may be higher risk than a narrower protected path.
- Do not select a route solely because it is visually open. Reject or downgrade routes through line-of-fire, under unstable loads, beside unsupported stacks, near pressurized/electrical/rotating equipment, or through pinch/crush/shear zones.
- Separate current-state route safety from future-state route safety. A path may be unsafe now because a worker/load temporarily blocks it, but become the best route after the dynamic obstruction clears.
- Explicitly distinguish safe exits/openings next to barriers from unsafe open edges/drop-offs. Do not mark an entire side as no-go if a guarded exit or turn is visible after a temporary obstruction.
- When an edge-adjacent side opening is visible or plausible, reason locally: the exposed edge may be no-go, while a guarded opening/turn beside it may be a conditional route after clearance and verification.
- Treat barriers, cones, rails, and workers as possible path-control cues. If they form or imply a managed opening next to a hazardous edge, preserve that opening as a conditional route candidate instead of merging it with the edge no-go zone.
- Do not conclude "no route forward" solely because the direct center path is blocked; check whether the safe path requires moving forward slightly and then turning left/right around or before the blockage.
- If every immediate path is unsafe but a possible side opening/turn exists, include it as a conditional candidate: what must clear first, what must be verified, and how to move to it safely. Separate "do not go now" from "possible safest route after controls are restored."
- Prefer a conditional route when it is safest: wait for the temporary obstruction to clear, move forward to a visible landmark or opening, then turn left/right through the safe exit while avoiding edges, cords, unstable footing, and line-of-fire zones.
- In first-person scenes, express the route as an egocentric step sequence from the camera holder's current position. Include hold/wait points and re-check points before blind turns or edge-adjacent openings.
- Do not over-prioritize the shortest route. Prefer the route with the lowest expected incident risk, even if it requires waiting, stepping back, or asking the area to be cleared.
- When recommending a route, include conditional instructions such as wait, move forward to a landmark, then turn left/right, while identifying any remaining hazards.
- For movement/navigation tasks, do not finalize a route unless the SIR includes explicit route candidates or a ROUTE MEMORY BLOCK. If it does not, ask the translator to inspect visible exits/openings, turns behind temporary obstructions, no-go zones, conditional routes after waiting, and the best route candidate.
- If the task asks what may happen next, reason about the most likely {settings.safety_prediction_horizon} hazard progression or unsafe outcome supported by the observed precursor conditions.
- Use an OSHA-style verifier before finalizing: identify the hazard category, exposed person/asset, credible incident mechanism, severity, likelihood, existing/missing controls, immediate control action, and whether work should stop/wait/proceed with controls.
- If safety-critical evidence is missing, ask the translator for specific details instead of filling gaps with assumption."""

    prompt += """
- When evidence is sufficient, prefer terminate_and_report_safety over terminate_and_answer.
- The safety report must explicitly include the safety_navigation schema:
  current_scene,
  dynamic_clear_scene,
  route_candidates,
  candidate_risk_scores,
  no_go_zones,
  verification_needed,
  safest_now,
  safest_after_clear,
  fallback,
  plus:
  hazards,
  safest next action,
  safest route,
  route risks,
  route now versus route after temporary obstructions clear,
  safest solution,
  and a short evidence-based reasoning summary."""

    if state.get("outer_iter", 1) > 1:
        prompt += """
- Because this is a later outer-loop pass, prioritize requests that can reduce safety uncertainty: exact worker positions, hand placement, proximity to hazards, motion direction, tool orientation, equipment energized state, guard/barrier status, PPE presence/absence, labels/warnings, footing/surface condition, load stability, and available escape path."""
        prompt += """
- If route choice is uncertain, ask the translator to identify all visible exits/openings, passable paths, blocked paths, obstacles, hazards along each path, and the safest route candidate."""
        prompt += """
- Ask whether any apparent blockage is temporary and whether a route becomes safer after waiting for a worker, carried material, vehicle, or moving equipment to clear."""

    return prompt


def _tool_messages_for_calls(tool_calls: list[dict]) -> list[ToolMessage]:
    """Build OpenAI-required tool responses for assistant tool calls."""
    messages: list[ToolMessage] = []
    for tc in tool_calls:
        tool_call_id = tc.get("id")
        if not tool_call_id:
            continue
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        if name == "terminate_and_answer":
            content = terminate_and_answer.invoke(args)
        elif name == "terminate_and_ask_translator":
            content = terminate_and_ask_translator.invoke(args)
        elif name == "terminate_and_report_safety":
            content = terminate_and_report_safety.invoke(args)
        elif name == "continue_reasoning":
            content = continue_reasoning.invoke(args)
        else:
            content = "Tool call acknowledged."
        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=name or None,
            )
        )
    return messages


def _render_reasoner_user_message(
    question: str, options: list[str] | None, sir_content: str
) -> str:
    """Render the user-facing prompt body for the Reasoner LLM call.

    Reasoner is text-only (D-09): SIR content goes here as plain text, no
    image_b64 attachment.
    """
    parts = [f"Question: {question}"]
    if options:
        parts.append("Options:\n" + "\n".join(options))
        letters = [opt.split(".", 1)[0].strip() for opt in options if "." in opt]
        if letters:
            parts.append(
                "Valid answer letters: "
                + ", ".join(letters)
                + ". The final answer must be exactly one of these letters."
            )
    parts.append(f"Visual description (SIR):\n{sir_content}")
    return "\n\n".join(parts)


def _valid_option_letters(options: list[str] | None) -> set[str]:
    letters: set[str] = set()
    for opt in options or []:
        if "." not in opt:
            continue
        letter = opt.split(".", 1)[0].strip().upper()
        if letter:
            letters.add(letter)
    return letters


def _normalize_answer_to_option(answer: str, options: list[str] | None) -> str:
    valid = _valid_option_letters(options)
    if not valid:
        return answer
    text = (answer or "").strip().upper()
    if text in valid:
        return text
    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in valid:
        return match.group(1)
    return answer


def _tool_arg_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        return str(value)
    if value is None:
        return ""
    return str(value)


async def reasoner_model_node(state: ReasonerSubgraphState) -> dict:
    """Call the Reasoner LLM with bind_tools([3 decisions], tool_choice="auto").

    Non-streaming only — Pitfall #1 (vLLM hermes parser bug with .astream
    + tool calls). NO ``image_b64`` is ever passed (D-09: Reasoner is text-only).

    On the first invocation (empty ``reasoner_messages``), seeds the system
    prompt + a user message rendering question + options + SIR content.
    Subsequent loop iterations append the new AIMessage to the existing list
    (LangGraph ``add_messages`` reducer dedupes by ID).
    """
    existing = list(state.get("reasoner_messages") or [])
    if not existing:
        settings = Settings()
        seed = [
            SystemMessage(content=reasoner_prompts.SYSTEM_PROMPT),
            SystemMessage(content=_reasoner_adapter_prompt(state, settings)),
            HumanMessage(
                content=_render_reasoner_user_message(
                    state["question"],
                    state.get("options"),
                    state["sir"].content,
                )
            ),
        ]
        msgs_to_send = seed
        msgs_to_return = seed
    else:
        msgs_to_send = existing
        msgs_to_return = []

    llm = create_reasoner_client().bind_tools(_DECISION_TOOLS, tool_choice="auto")
    response = await llm.ainvoke(msgs_to_send)  # NEVER .astream — Pitfall #1

    return {
        "reasoner_messages": msgs_to_return + [response],
        "reasoner_step": state.get("reasoner_step", 0) + 1,
    }


def decision_router(state: ReasonerSubgraphState) -> str:
    """Inspect AIMessage.tool_calls[0]['name'] (Finding #8).

    Clean structural routing — NOT brittle regex on assistant content (which
    is what the old ``_determine_reasoning_decision`` did at
    ``iterative_refinement.py:1051-1095``).

    Routing:
      - ``terminate_and_answer``           -> ROUTE_ANSWER
      - ``terminate_and_report_safety``    -> ROUTE_ANSWER
      - ``terminate_and_ask_translator``   -> ROUTE_FEEDBACK
      - ``continue_reasoning`` / no call / unknown -> ROUTE_CONTINUE
    """
    last = state["reasoner_messages"][-1]
    tcs = getattr(last, "tool_calls", None) or []
    if not tcs:
        return ROUTE_CONTINUE
    name = tcs[0].get("name")
    if name in {"terminate_and_answer", "terminate_and_report_safety"}:
        return ROUTE_ANSWER
    if name == "terminate_and_ask_translator":
        return ROUTE_FEEDBACK
    return ROUTE_CONTINUE  # continue_reasoning OR unknown -> loop


async def finalize_answer_node(state: ReasonerSubgraphState) -> dict:
    """Extract ``answer`` arg from the terminate_and_answer call into ``final_answer``."""
    last = state["reasoner_messages"][-1]
    tc = last.tool_calls[0]
    if tc["name"] == "terminate_and_report_safety":
        answer = (
            "SAFETY REPORT\n\n"
            f"Current scene: {_tool_arg_text(tc['args'].get('current_scene', ''))}\n\n"
            f"Dynamic-clear scene: {_tool_arg_text(tc['args'].get('dynamic_clear_scene', ''))}\n\n"
            f"Route candidates: {_tool_arg_text(tc['args'].get('route_candidates', ''))}\n\n"
            f"Candidate risk scores: {_tool_arg_text(tc['args'].get('candidate_risk_scores', ''))}\n\n"
            f"No-go zones: {_tool_arg_text(tc['args'].get('no_go_zones', ''))}\n\n"
            f"Verification needed: {_tool_arg_text(tc['args'].get('verification_needed', ''))}\n\n"
            f"Safest now: {_tool_arg_text(tc['args'].get('safest_now', ''))}\n\n"
            f"Safest after clear: {_tool_arg_text(tc['args'].get('safest_after_clear', ''))}\n\n"
            f"Fallback: {_tool_arg_text(tc['args'].get('fallback', ''))}\n\n"
            f"Hazards: {_tool_arg_text(tc['args'].get('hazards', ''))}\n\n"
            f"Safest next action: {_tool_arg_text(tc['args'].get('safest_next_action', ''))}\n\n"
            f"Safest route: {_tool_arg_text(tc['args'].get('safest_route', ''))}\n\n"
            f"Route risks: {_tool_arg_text(tc['args'].get('route_risks', ''))}\n\n"
            f"Route now vs after clear: {_tool_arg_text(tc['args'].get('route_now_vs_after_clear', ''))}\n\n"
            f"Safest solution: {_tool_arg_text(tc['args'].get('safest_solution', ''))}\n\n"
            f"Confidence: {_tool_arg_text(tc['args'].get('confidence', ''))}\n\n"
            f"Reasoning: {_tool_arg_text(tc['args'].get('reasoning', ''))}"
        )
    else:
        answer = _normalize_answer_to_option(
            _tool_arg_text(tc["args"].get("answer", "")),
            state.get("options"),
        )
    return {
        "final_answer": answer,
        "reasoner_messages": _tool_messages_for_calls(last.tool_calls),
    }


async def finalize_feedback_node(state: ReasonerSubgraphState) -> dict:
    """Extract ``feedback`` arg into ``reasoner_feedback`` (subgraph output).

    The actual ``SIR.merge_feedback()`` call lives in Phase 5's outer-loop
    bookkeeping (Finding #3) — Phase 4 only EXTRACTS the string here.
    """
    last = state["reasoner_messages"][-1]
    tc = last.tool_calls[0]
    feedback = _tool_arg_text(tc["args"].get("feedback", ""))
    return {
        "reasoner_feedback": feedback,
        "reasoner_messages": _tool_messages_for_calls(last.tool_calls),
    }


async def check_termination_node(state: ReasonerSubgraphState) -> dict:
    """Acknowledge continue tool calls before the next Reasoner turn.

    OpenAI's Chat Completions API requires every assistant ``tool_calls`` item
    to be followed by a matching ``ToolMessage`` before another assistant turn.
    The Reasoner executes terminal tools in finalize nodes; this node only sees
    ``continue_reasoning`` / unknown tool calls routed through
    :func:`decision_router`, so it supplies a small acknowledgment and lets
    ``should_continue`` decide whether another model turn is allowed.
    """
    last = (
        state.get("reasoner_messages", [])[-1]
        if state.get("reasoner_messages")
        else None
    )
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {}
    tool_messages = _tool_messages_for_calls(tool_calls)
    return {"reasoner_messages": tool_messages} if tool_messages else {}


def should_continue(state: ReasonerSubgraphState) -> str:
    """Route to END when ``reasoner_step >= n_r`` (paper N_R = 3)."""
    settings = Settings()
    max_steps = getattr(settings, "n_r", 3)
    if state.get("reasoner_step", 0) >= max_steps:
        return ROUTE_END
    return ROUTE_LOOP
