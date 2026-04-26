"""Translator subgraph node functions.

Four async node functions plus two pure routers wire together the inner
loop ``translator_model -> tools -> refine_sir -> check_termination``.

Edge keys are exposed as module-level string constants (Pitfall #5 — no
inline string literals in ``builder.py``'s ``add_conditional_edges``
mappings).

Wipe-per-iter design: ``translator_messages`` is a subgraph-private key
on :class:`TranslatorSubgraphState`. LangGraph initializes it fresh on
each invocation, so we never see messages from a prior outer iteration.
This mirrors the old ``_reset_agent_memory_for_iteration()`` in
``src/multi-agent/app/flow/iterative_refinement.py:56-71``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.seeingeye.agents.translator.policy import (
    parse_tool_calls_multiple_formats,
    valid_translator_tool_calls,
)
from src.seeingeye.agents.translator.state import TranslatorSubgraphState
from src.seeingeye.config.settings import Settings
from src.seeingeye.llm.vllm_openai import create_translator_client
from src.seeingeye.prompts import translator as translator_prompts
from src.seeingeye.tools.action_motion_scan import analyze_action_motion
from src.seeingeye.tools.mechanics_hazard_scan import analyze_mechanics_hazard
from src.seeingeye.tools.route_surface_scan import analyze_route_surface
from src.seeingeye.tools.spatial_intelligence_scan import analyze_spatial_intelligence

# Edge key constants — Pitfall #5: avoid inline string-literal typos in
# the builder's add_conditional_edges mappings.
ROUTE_TOOLS = "tools"
ROUTE_DONE = "done"
ROUTE_CONTINUE = "continue"
ROUTE_END = "end"

TRANSLATOR_ADAPTER_PROMPT = """Runtime adapter for OpenAI-compatible VLMs:
- Produce the visual description directly as text; do not mention or call terminate_and_output_caption.
- Prefer a dense, option-relevant SIR over tool-call chatter.
- Include exact visible text, numbers, table cells, chart labels, legends, units, and spatial relations.
- If the question has answer options, explicitly describe visual evidence that distinguishes the options.
- Valid tool names are exactly: ocr, read_table, smart_grid_caption, action_motion_scan, route_surface_scan, mechanics_hazard_scan, spatial_intelligence_scan. Never use placeholder names such as "...".
- For first-person actions, action_motion_scan can model Action = hand pose + active object + contact target + temporal motion + scene context, and predict short-horizon moving-object paths.
- For spatial tasks, spatial_intelligence_scan can zoom into key regions and verify route/shape/diagram candidates.
- For construction or engineering safety tasks, mechanics_hazard_scan can verify load paths, supports, stored energy, failure modes, and line-of-fire risk.
- For video frames, tools are disabled; the frames are already attached and directly visible, so describe them directly.
- For single-image tasks only, use OCR/read_table/smart_grid_caption when direct vision is insufficient.
- Do not solve the question or name the final answer; only describe visible evidence."""


VSTAR_TOOL_COLLABORATION_SYSTEM = """You are the V* collaboration layer for SeeingEye.
Your job is not to answer. Your job is to coordinate tool evidence into one stronger visual evidence package.

Inputs may include:
- action_motion_scan: egocentric action model, hand/object/contact/motion/context, predicted motion paths.
- route_surface_scan: route, walkable surface, blockers, dynamic-clear, hazards.
- mechanics_hazard_scan: load paths, support points, stored energy, line-of-fire, failure modes.
- spatial_intelligence_scan: 4x4 guided visual search, crop/zoom evidence, candidate verification, geometric consistency.
- direct Translator vision: broad global image evidence.
- future optional tools: OCR/read_table/smart_grid for text, tables, labels, and focused regions.

Return concise JSON with these exact keys:
vstar_goal, tool_roles, tool_agreement, tool_conflicts, missing_evidence,
candidate_verification_plan, next_best_tool_requests, consolidated_sir_addendum.

Rules:
- Treat V* as the guided controller: decide which evidence should dominate and what still needs verification.
- Use spatial_intelligence_scan as the crop/zoom verifier, not as a final answer.
- Use action_motion_scan as the action/motion-path verifier, not as a final answer.
- Use route_surface_scan as the route/hazard verifier, not as a final answer.
- Use mechanics_hazard_scan as the mechanical-risk verifier, not as a final answer.
- If route and spatial scans disagree, preserve both hypotheses and state what visual detail would resolve the conflict.
- If mechanics and route scans disagree, preserve both: a route can be walkable but mechanically unsafe because of overhead load, line-of-fire, stored energy, or collapse risk.
- Prefer a small number of high-value follow-up requests over broad re-captioning.
- Keep candidate paths/shapes/options alive until tool evidence rules them out.
- Treat predicted motion paths as route constraints: a route can be open now but unsafe if a hand/tool/person/cart/vehicle/load is likely to move into it.
- For safety scenes, require before/after dynamic blocker reasoning and route-risk comparison.
- For diagram/math scenes, require visible label extraction before calculation."""


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
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
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            text = content.get(key)
            if isinstance(text, str):
                return text
        return str(content)
    if content is None:
        return ""
    return str(content)


def _translator_adapter_prompt(
    state: TranslatorSubgraphState,
    settings: Settings,
) -> str:
    prompt = TRANSLATOR_ADAPTER_PROMPT
    if settings.analysis_mode.strip().lower() != "safety":
        return prompt

    prompt += f"""
- Safety analysis mode is active. Use an {settings.safety_framework}-style hazard identification lens, but stay grounded in visible evidence.
- When movement or navigation is relevant, gather evidence for the {settings.safety_navigation_schema} safety_navigation schema: current_scene, dynamic_clear_scene, route_candidates, candidate_risk_scores, no_go_zones, verification_needed, safest_now, safest_after_clear, and fallback.
- Follow this fixed instruction exactly as an additional task anchor: "{settings.safety_scan_prompt}"
- In every pass, capture workers/people, tools, machines, vehicles, materials, energy sources, access/egress, housekeeping, barriers, guards, signage, and PPE that are visible or visibly absent.
- Note conditions that could contribute to incidents: line-of-fire exposure, unstable loads/objects, overhead stored materials, suspended loads, falling-object zones, low/protruding scaffold members, moving parts, pinch points, fall edges, slippery/cluttered surfaces, electrical sources, heat/flame, chemicals, blocked exits, blind spots, and conflicting paths of people/equipment.
- For video, compare frames for changes over time and note precursor events that could lead to a {settings.safety_prediction_horizon} safety incident.
- Keep observations factual. If a danger is not directly visible, describe the precursor condition instead of inventing a conclusion."""

    prompt += """
- Organize safety observations with a distinct SAFETY EVIDENCE BLOCK containing:
  observed_hazards, precursor_conditions, missing_or_failed_controls, exposed_people_or_assets, and safest_immediate_actions_supported_by_visible_evidence."""

    prompt += """
- When movement, navigation, access, egress, path choice, or "which way should I go" is relevant, always add a ROUTE MEMORY BLOCK containing:
  current_position, likely_destination_or_safe_zone, visible_route_candidates, passable_paths, route_now,
  temporary_obstructions, route_after_temporary_obstruction_clears, exit_openings_or_turns,
  blocked_or_high_risk_paths, hazards_along_each_path, stable_footing_areas, edges/drop-offs/openings,
  overhead_hazards_along_each_path, struck_by_or_falling_object_zones, barriers/guardrails,
  moving_people_or_equipment, visibility_or_occlusion_limits,
  current_scene, dynamic_clear_scene, route_candidates, candidate_risk_scores, no_go_zones,
  verification_needed, safest_now, safest_after_clear, fallback, conditional_step_sequence,
  and best_route_candidate.
- Build route candidates from the camera/person point of view using relative motion primitives: wait, stop, step back, move forward, keep left/right, turn left/right, or use another designated route.
- Score the entire body envelope of each route: floor/feet, side clearance, head/shoulder clearance, and overhead falling-object or struck-by exposure. A wider floor path is not safest if overhead material or suspended loads are above it.
- Do not lock onto the first apparently safe or open lane. Keep plausible alternatives until side openings, occlusions, and temporary blockers have been checked.
- For every visible dynamic blocker, describe the scene before and after it clears: what route, opening, hazard, or line-of-sight it currently hides/occupies, and what may become visible/passable after it moves.
- Separate current scene, after-dynamic-clear scene, and remaining static blockers. Do not treat a route blocked by a movable person/object as permanently blocked.
- Look for non-obvious route geometry: openings beside barricades, turns just beyond people/materials, gaps behind movable objects, alternate egress paths, and routes that require moving forward before turning.
- Explicitly inspect side openings before, beside, or just beyond a blockage; a stacked object, column, wall, cart, or person may hide the route rather than end it.
- For every side opening, lateral gap, or turn near a barrier/guardrail, describe whether it appears to be a protected exit/turn, an exposed fall edge, or uncertain. Do not collapse all edge-adjacent openings into no-go zones without describing the local barrier geometry.
- If cones, barricades, rails, or a worker position suggest a controlled pedestrian exit/turn beside an otherwise hazardous edge, describe the exit/turn separately from the edge hazard and keep the path geometry: move forward to the opening, then turn left/right.
- Do not assume a stacked object or wall blocks all forward travel if a lateral turn or opening is visible before, beside, or just beyond it.
- If a side opening or turn is visible but safety is uncertain, keep it as a conditional_route_candidate with the verification needed, rather than omitting it or treating the whole side as permanently blocked.
- Distinguish permanent blockage from temporary dynamic blockage such as a worker, carried material, vehicle, swinging load, or moving equipment.
- Distinguish a safe exit/opening beside a barrier from an unsafe open edge/drop-off; describe whether a route could become passable after the temporary obstruction clears.
- If a worker, carried object, or equipment is blocking a visible exit/opening near a barricade or guardrail, describe both the unsafe/impossible route now and the possible route after waiting for it to clear.
- Mark no-go zones separately from conditional routes. A no-go zone is a fall edge, pinch/crush zone, line-of-fire zone, energized/unguarded equipment area, or unstable surface; a conditional route may be acceptable only after a temporary hazard clears and the surface/edge protection is verified.
- Describe the safest route candidate using visible landmarks and relative directions, including conditional steps such as wait, move forward, then turn left/right when an exit is clear."""

    if state.get("media_type") == "video":
        prompt += f"""
- For video, follow the {settings.video_reasoning_framework} style decomposition:
  WHEN: identify the frames or moments where relevant changes occur.
  WHERE: describe locations and relative positions of key people, objects, rooms, paths, and hazards.
  WHAT: describe the interaction, event, or developing situation supported by those temporal and spatial cues.
- Prefer explicit timestamps/frame references and relative-position evidence over broad summary."""
    else:
        prompt += """
- For single images, explicitly describe relative positions, support surfaces, containment/on-top-of/inside relations, barriers, openings, and visibility constraints when they matter."""

    if state.get("outer_iter", 1) > 1:
        prompt += f"""
- This is a later review pass after reasoning feedback. Re-inspect specifically for hazard-critical details that may have been missed.
- Prioritize subtle safety cues: missing/incorrect PPE, missing guards, compromised footing, body positioning near hazards, clearances, load stability, energized equipment state, tool orientation, hand placement, escape routes, warning labels, and any developing condition that could soon create harm."""

    return prompt


def _render_translator_user_message(
    question: str,
    options: list[str] | None,
    sir_content: str,
    media_type: str = "image",
    frame_count: int = 1,
    safety_scan_prompt: str | None = None,
    action_motion_scan: str | None = None,
    route_surface_scan: str | None = None,
    mechanics_hazard_scan: str | None = None,
    spatial_intelligence_scan: str | None = None,
    vstar_collaboration: str | None = None,
) -> str:
    """Render the per-step user-text payload for the Translator VLM."""
    parts = [f"Question: {question}"]
    if options:
        parts.append("Options:\n" + "\n".join(options))
    if media_type == "video":
        parts.append(
            "Video input: the attached images are chronological frames sampled "
            f"from the same video ({frame_count} frames). Compare adjacent "
            "frames, identify important changes, infer actions/events over "
            "time, and describe what happens in the video."
        )
        parts.append(
            "Use a when-where-what evidence structure: WHEN relevant changes "
            "happen, WHERE key people/objects/paths/hazards are located, and "
            "WHAT interaction or event is occurring."
        )
    if safety_scan_prompt:
        parts.append(f"Safety directive: {safety_scan_prompt}")
    if action_motion_scan:
        parts.append(
            "Automatic action/motion scan (use this as egocentric action evidence "
            "and short-horizon moving-object path prediction; verify against the image/video):\n"
            + action_motion_scan
        )
    if route_surface_scan:
        parts.append(
            "Automatic route surface scan (use this as route-specific evidence, "
            "but verify against the image):\n" + route_surface_scan
        )
    if mechanics_hazard_scan:
        parts.append(
            "Automatic mechanics hazard scan (use this as load-path, stored-energy, "
            "failure-mode, and line-of-fire evidence; verify against the image):\n"
            + mechanics_hazard_scan
        )
    if spatial_intelligence_scan:
        parts.append(
            "Automatic spatial intelligence scan (guided crop/zoom evidence; "
            "use it to verify candidates, not as a final answer):\n"
            + spatial_intelligence_scan
        )
    if vstar_collaboration:
        parts.append(
            "V* tool collaboration synthesis (use this to coordinate tool "
            "evidence and identify unresolved conflicts):\n" + vstar_collaboration
        )
    if sir_content:
        parts.append(f"Current visual description (SIR):\n{sir_content}")
    return "\n\n".join(parts)


def _looks_like_navigation_task(question: str, settings: Settings) -> bool:
    if settings.analysis_mode.strip().lower() != "safety":
        return False
    text = question.lower()
    nav_terms = (
        "route",
        "path",
        "walk",
        "walkway",
        "move",
        "go",
        "exit",
        "opening",
        "turn",
        "way",
        "egress",
        "navigate",
        "through",
        "走",
        "路",
        "出口",
        "怎么走",
        "往哪",
        "右拐",
        "左拐",
        "通行",
    )
    return any(term in text for term in nav_terms)


def _looks_like_spatial_task(question: str, options: list[str] | None) -> bool:
    # In benchmarks and real-world use, many spatial questions do not say
    # "spatial" or "diagram" explicitly. Multiple-choice image questions are
    # cheap to misread by fixation, so bias toward high recall and let V* decide
    # what evidence is useful.
    if options:
        return True
    text = " ".join([question, *(options or [])]).lower()
    spatial_terms = (
        "spatial",
        "route",
        "path",
        "walk",
        "move",
        "exit",
        "opening",
        "turn",
        "way",
        "egress",
        "navigate",
        "view",
        "projection",
        "perspective",
        "front view",
        "top view",
        "side view",
        "section",
        "cross-section",
        "orthographic",
        "shape",
        "geometry",
        "diagram",
        "blueprint",
        "map",
        "floor plan",
        "figure",
        "image",
        "shown",
        "truss",
        "beam",
        "member",
        "joint",
        "support",
        "load",
        "force",
        "moment",
        "shear",
        "stress",
        "strain",
        "slope",
        "angle",
        "length",
        "distance",
        "axis",
        "plot",
        "chart",
        "graph",
        "contour",
        "cross section",
        "profile",
        "dimension",
        "relative position",
        "left",
        "right",
        "above",
        "below",
        "behind",
        "in front",
        "hazard",
        "danger",
        "safe",
        "safest",
        "走",
        "路",
        "出口",
        "怎么走",
        "往哪",
        "右拐",
        "左拐",
        "通行",
        "视角",
        "投影",
        "截面",
        "剖面",
        "形状",
        "几何",
        "图纸",
        "蓝图",
        "地图",
        "危险",
        "安全",
    )
    return any(term in text for term in spatial_terms)


def _looks_like_mechanics_hazard_task(question: str, settings: Settings) -> bool:
    if settings.analysis_mode.strip().lower() == "safety":
        return True
    text = question.lower()
    mechanics_terms = (
        "load",
        "force",
        "support",
        "beam",
        "truss",
        "scaffold",
        "ladder",
        "hoist",
        "crane",
        "suspended",
        "overhead",
        "falling object",
        "line of fire",
        "pinch",
        "crush",
        "shear",
        "pressure",
        "hydraulic",
        "pneumatic",
        "electrical",
        "energized",
        "rotating",
        "stored energy",
        "failure",
        "collapse",
        "tip",
        "roll",
        "slide",
        "工地",
        "危险",
        "安全",
        "受力",
        "支撑",
        "载荷",
        "坠落",
        "挤压",
        "剪切",
        "压力",
        "电",
        "机械",
        "脚手架",
    )
    return any(term in text for term in mechanics_terms)


def _looks_like_action_motion_task(question: str, settings: Settings, media_type: str) -> bool:
    if settings.analysis_mode.strip().lower() == "safety":
        return True
    if media_type == "video":
        return True
    text = question.lower()
    action_terms = (
        "action",
        "motion",
        "move",
        "moving",
        "walk",
        "carry",
        "carrying",
        "hand",
        "grip",
        "hold",
        "holding",
        "contact",
        "touch",
        "push",
        "pull",
        "tool",
        "object",
        "target",
        "trajectory",
        "path",
        "next",
        "happen",
        "发生",
        "动作",
        "移动",
        "行径",
        "路径",
        "手",
        "拿",
        "握",
        "接触",
        "推",
        "拉",
        "下一步",
    )
    return any(term in text for term in action_terms)


async def _automatic_route_surface_scan(
    state: TranslatorSubgraphState,
    settings: Settings,
) -> str | None:
    if not _looks_like_navigation_task(state.get("question", ""), settings):
        return None
    if state.get("media_type", "image") != "image":
        return None
    frames = list(state.get("image_frames") or [])
    image_b64 = None
    mime_type = "image/jpeg"
    if frames:
        image_b64 = frames[0].get("b64")
        mime_type = frames[0].get("mime_type") or mime_type
    if not image_b64:
        image_b64 = state.get("image_b64")
    if not image_b64:
        return None
    try:
        scan = await analyze_route_surface(
            image_b64=image_b64,
            question=state.get("question", ""),
            mime_type=mime_type,
        )
    except Exception as e:  # noqa: BLE001
        scan = f"route_surface_scan unavailable: {e}"
    return scan.strip() or None


async def _automatic_action_motion_scan(
    state: TranslatorSubgraphState,
    settings: Settings,
) -> str | None:
    media_type = state.get("media_type", "image")
    if not _looks_like_action_motion_task(state.get("question", ""), settings, media_type):
        return None
    frames = list(state.get("image_frames") or [])
    image_b64 = None
    mime_type = "image/jpeg"
    if frames:
        image_b64 = frames[0].get("b64")
        mime_type = frames[0].get("mime_type") or mime_type
    if not image_b64:
        image_b64 = state.get("image_b64")
    if not image_b64 and not frames:
        return None
    try:
        scan = await analyze_action_motion(
            image_b64=image_b64,
            frames=frames if media_type == "video" else None,
            question=state.get("question", ""),
            mime_type=mime_type,
        )
    except Exception as e:  # noqa: BLE001
        scan = f"action_motion_scan unavailable: {e}"
    return scan.strip() or None


async def _automatic_mechanics_hazard_scan(
    state: TranslatorSubgraphState,
    settings: Settings,
) -> str | None:
    if state.get("media_type", "image") != "image":
        return None
    if not _looks_like_mechanics_hazard_task(state.get("question", ""), settings):
        return None
    frames = list(state.get("image_frames") or [])
    image_b64 = None
    mime_type = "image/jpeg"
    if frames:
        image_b64 = frames[0].get("b64")
        mime_type = frames[0].get("mime_type") or mime_type
    if not image_b64:
        image_b64 = state.get("image_b64")
    if not image_b64:
        return None
    try:
        scan = await analyze_mechanics_hazard(
            image_b64=image_b64,
            question=state.get("question", ""),
            mime_type=mime_type,
        )
    except Exception as e:  # noqa: BLE001
        scan = f"mechanics_hazard_scan unavailable: {e}"
    return scan.strip() or None


async def _automatic_spatial_intelligence_scan(
    state: TranslatorSubgraphState,
) -> str | None:
    if state.get("media_type", "image") != "image":
        return None
    if not _looks_like_spatial_task(state.get("question", ""), state.get("options")):
        return None
    frames = list(state.get("image_frames") or [])
    image_b64 = None
    mime_type = "image/jpeg"
    if frames:
        image_b64 = frames[0].get("b64")
        mime_type = frames[0].get("mime_type") or mime_type
    if not image_b64:
        image_b64 = state.get("image_b64")
    if not image_b64:
        return None
    try:
        scan = await analyze_spatial_intelligence(
            image_b64=image_b64,
            question=state.get("question", ""),
            options="\n".join(state.get("options") or []),
            mime_type=mime_type,
        )
    except Exception as e:  # noqa: BLE001
        scan = f"spatial_intelligence_scan unavailable: {e}"
    return scan.strip() or None


async def _automatic_vstar_tool_collaboration(
    state: TranslatorSubgraphState,
    action_scan: str | None,
    route_scan: str | None,
    mechanics_scan: str | None,
    spatial_scan: str | None,
) -> str | None:
    if not action_scan and not route_scan and not mechanics_scan and not spatial_scan:
        return None
    llm = create_translator_client()
    payload = (
        "Question:\n"
        + state.get("question", "")
        + "\n\nOptions:\n"
        + "\n".join(state.get("options") or [])
        + "\n\naction_motion_scan:\n"
        + (action_scan or "not available")
        + "\n\nroute_surface_scan:\n"
        + (route_scan or "not available")
        + "\n\nmechanics_hazard_scan:\n"
        + (mechanics_scan or "not available")
        + "\n\nspatial_intelligence_scan:\n"
        + (spatial_scan or "not available")
        + "\n\nCoordinate these tool outputs for the next Translator/Reasoner pass."
    )
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=VSTAR_TOOL_COLLABORATION_SYSTEM),
                HumanMessage(content=payload),
            ]
        )
    except Exception as e:  # noqa: BLE001
        return f"vstar_tool_collaboration unavailable: {e}"
    text = _message_content_text(getattr(response, "content", "")).strip()
    return text or None


def _render_multiframe_user_content(
    user_text: str,
    frames: list[dict[str, Any]],
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": user_text}]
    for idx, frame in enumerate(frames, start=1):
        timestamp = frame.get("timestamp_s")
        if timestamp is not None:
            content.append(
                {"type": "text", "text": f"Frame {idx} at {timestamp:.3f}s"}
            )
        mime_type = frame.get("mime_type") or "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{frame['b64']}",
                },
            }
        )
    return content


def _select_translator_model(state: TranslatorSubgraphState, settings: Settings) -> str:
    """Use the fast Translator first, then escalate after Reasoner feedback."""
    if state.get("outer_iter", 1) > 1:
        return settings.translator_escalation_model
    return settings.translator_model


async def translator_model_node(state: TranslatorSubgraphState) -> dict:
    """Call the Translator VLM.

    On step 0, build the initial multimodal message list from
    question/options/image_b64 + SIR context. On subsequent steps,
    append only the LLM response.

    Increments ``translator_step``.

    NO ``bind_tools`` — Translator emits text only. Tool dispatch is
    handled by :func:`translator_tools_node` via :mod:`policy`. This is
    the Pitfall #1 mitigation for the broken vLLM tool-call parser on
    Qwen2.5-VL.
    """
    # Wipe-per-iter: translator_messages is subgraph-private and starts
    # empty on each subgraph invocation. See
    # src/multi-agent/app/flow/iterative_refinement.py:56-71.
    msgs = list(state.get("translator_messages") or [])
    settings = Settings()
    action_scan: str | None = None
    route_scan: str | None = None
    mechanics_scan: str | None = None
    spatial_scan: str | None = None
    vstar_collaboration: str | None = None
    if not msgs:
        # First step inside this subgraph invocation: build initial messages.
        sys_msg = SystemMessage(content=translator_prompts.SYSTEM_PROMPT)
        action_scan = await _automatic_action_motion_scan(state, settings)
        route_scan = await _automatic_route_surface_scan(state, settings)
        mechanics_scan = await _automatic_mechanics_hazard_scan(state, settings)
        spatial_scan = await _automatic_spatial_intelligence_scan(state)
        vstar_collaboration = await _automatic_vstar_tool_collaboration(
            state,
            action_scan,
            route_scan,
            mechanics_scan,
            spatial_scan,
        )
        user_text = _render_translator_user_message(
            state["question"],
            state.get("options"),
            state["sir"].content,
            state.get("media_type", "image"),
            len(state.get("image_frames") or []) or 1,
            settings.safety_scan_prompt
            if settings.analysis_mode.strip().lower() == "safety"
            else None,
            action_scan,
            route_scan,
            mechanics_scan,
            spatial_scan,
            vstar_collaboration,
        )
        frames = list(state.get("image_frames") or [])
        if not frames and state.get("image_b64"):
            frames = [
                {
                    "b64": state["image_b64"],
                    "timestamp_s": None,
                    "mime_type": "image/jpeg",
                }
            ]
        if frames:
            user_content: list[dict] | str = _render_multiframe_user_content(
                user_text, frames
            )
        else:
            user_content = user_text
        msgs = [
            sys_msg,
            SystemMessage(content=_translator_adapter_prompt(state, settings)),
            HumanMessage(content=user_content),
        ]
        initial_payload: list = msgs
    else:
        initial_payload = []  # nothing to append besides the response

    llm = create_translator_client(model=_select_translator_model(state, settings))
    response = await llm.ainvoke(msgs)

    update: dict[str, Any] = {
        "translator_messages": initial_payload + [response]
        if initial_payload
        else [response],
        "translator_step": state.get("translator_step", 0) + 1,
    }
    if not state["sir"].content and route_scan:
        content = "AUTOMATIC ROUTE SURFACE SCAN:\n" + route_scan
        if action_scan:
            content += "\n\nAUTOMATIC ACTION MOTION SCAN:\n" + action_scan
        if mechanics_scan:
            content += "\n\nAUTOMATIC MECHANICS HAZARD SCAN:\n" + mechanics_scan
        if spatial_scan:
            content += "\n\nAUTOMATIC SPATIAL INTELLIGENCE SCAN:\n" + spatial_scan
        if vstar_collaboration:
            content += "\n\nV* TOOL COLLABORATION SYNTHESIS:\n" + vstar_collaboration
        update["sir"] = state["sir"].update(content)
    elif not state["sir"].content and action_scan:
        content = "AUTOMATIC ACTION MOTION SCAN:\n" + action_scan
        if mechanics_scan:
            content += "\n\nAUTOMATIC MECHANICS HAZARD SCAN:\n" + mechanics_scan
        if spatial_scan:
            content += "\n\nAUTOMATIC SPATIAL INTELLIGENCE SCAN:\n" + spatial_scan
        if vstar_collaboration:
            content += "\n\nV* TOOL COLLABORATION SYNTHESIS:\n" + vstar_collaboration
        update["sir"] = state["sir"].update(content)
    elif not state["sir"].content and mechanics_scan:
        content = "AUTOMATIC MECHANICS HAZARD SCAN:\n" + mechanics_scan
        if spatial_scan:
            content += "\n\nAUTOMATIC SPATIAL INTELLIGENCE SCAN:\n" + spatial_scan
        if vstar_collaboration:
            content += "\n\nV* TOOL COLLABORATION SYNTHESIS:\n" + vstar_collaboration
        update["sir"] = state["sir"].update(content)
    elif not state["sir"].content and spatial_scan:
        content = "AUTOMATIC SPATIAL INTELLIGENCE SCAN:\n" + spatial_scan
        if vstar_collaboration:
            content += "\n\nV* TOOL COLLABORATION SYNTHESIS:\n" + vstar_collaboration
        update["sir"] = state["sir"].update(content)
    return update


def route_after_model(state: TranslatorSubgraphState) -> str:
    """Inspect the latest AIMessage; route to tools or to termination."""
    if state.get("media_type") == "video":
        return ROUTE_DONE
    last = state["translator_messages"][-1]
    parsed = parse_tool_calls_multiple_formats(
        _message_content_text(getattr(last, "content", ""))
    )
    return ROUTE_TOOLS if valid_translator_tool_calls(parsed) else ROUTE_DONE


async def refine_sir_node(state: TranslatorSubgraphState) -> dict:
    """Apply ``SIR.update()`` with the latest tool result.

    Uses ``SIR.update()`` (incremental, ``--- UPDATED SIR ---`` separator)
    per Finding #3 in 04-RESEARCH.md. ``SIR.replace()`` is Phase 5's
    outer-loop concern — NOT this inner loop's job.
    """
    last_tool_msg = next(
        (
            m
            for m in reversed(state["translator_messages"])
            if getattr(m, "type", None) == "tool"
        ),
        None,
    )
    if last_tool_msg is None:
        return {}
    new_sir = state["sir"].update(str(last_tool_msg.content))
    return {"sir": new_sir}


async def check_termination_node(state: TranslatorSubgraphState) -> dict:
    """Capture direct Translator prose into SIR before termination checks.

    Routing is decided by :func:`should_continue` which reads
    ``translator_step`` directly.
    """
    last = state["translator_messages"][-1] if state.get("translator_messages") else None
    if getattr(last, "type", None) != "ai":
        return {}
    content = _message_content_text(getattr(last, "content", "")).strip()
    if not content:
        return {}
    parsed = parse_tool_calls_multiple_formats(content)
    if valid_translator_tool_calls(parsed):
        return {}
    return {"sir": state["sir"].update(content)}


def should_continue(state: TranslatorSubgraphState) -> str:
    """Route to ROUTE_END once translator_step has reached the n_t bound.

    The paper specifies N_T = 3; :class:`Settings` exposes both
    ``n_t`` and ``max_iters`` (defaulted to 3) — we prefer ``n_t`` since
    it's the per-agent inner-loop bound, not the outer-loop bound.
    """
    settings = Settings()
    max_steps = getattr(settings, "n_t", None) or getattr(settings, "max_iters", 3)
    if state.get("translator_step", 0) >= max_steps:
        return ROUTE_END
    return ROUTE_CONTINUE
