"""Route surface scan tool for first-person safety navigation scenes.

The tool asks the configured Translator VLM to focus only on traversable
surface, exits/openings, candidate paths, and static/dynamic blockers. It is
used as an automatic pre-scan in safety mode so the Reasoner receives a
route-specific structure instead of relying on a broad caption alone.
"""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.seeingeye.llm.vllm_openai import create_translator_client


ROUTE_SURFACE_SCAN_SYSTEM = """You are a route-surface scanner for first-person safety navigation.
Only inspect visible spatial evidence. Do not give a final safety answer.
Return concise JSON with these exact keys:
walkable_surface, exit_candidates, protected_openings_or_turns, no_go_zones,
static_obstacles, dynamic_obstacles, dynamic_object_before_after,
space_revealed_after_dynamic_clear, route_now, route_after_dynamic_clear,
side_openings_before_or_beside_blockage, kept_alternatives, candidate_ranking_not_final,
ego_centered_route_map, predicted_motion_paths_by_entity, overhead_hazards_by_route, struck_by_or_falling_object_zones, verification_needed,
best_conditional_candidate.

Rules:
- Scan the floor/walkable surface first, then exits/openings/turns, then obstacles.
- Scan overhead hazards with the same priority as floor hazards: stored materials above the path, suspended loads, stacked pipes, loose scaffold parts, low beams, hooks, straps, protruding rods, or anything that could fall/swing/strike.
- For each route candidate, assess the full travel envelope from floor to head/shoulder height and overhead, not only the ground surface.
- Do not lock onto the first apparently safe lane. Keep all plausible route candidates until side openings and dynamic blockers are checked.
- Explicitly inspect openings or turns before, beside, or just beyond a blockage such as stacked materials, a wall, a column, a cart, or a worker.
- Separate exposed edges/drop-offs from protected openings or turns beside barriers.
- Treat workers, carried materials, vehicles, and moving equipment as dynamic blockers.
- For each dynamic blocker, infer what route geometry it may currently hide or occupy, and what may be revealed after it moves. This is a before/after scene-change hypothesis, not a final fact.
- For each movable person/object/vehicle/equipment item, predict short-horizon motion paths using visible pose, orientation, contact, support, gravity, and scene affordances.
- Always separate: current scene, scene after dynamic blockers clear, and remaining static blockers.
- If a side opening/turn is visible but uncertain, keep it as a conditional candidate and state what must be verified.
- If one candidate seems safest, still list at least one kept alternative when visible, and explain why it might become better after a temporary blocker clears.
- candidate_ranking_not_final must state that the final route should be chosen after comparing all candidates and verifying uncertain openings.
- Use egocentric directions from the camera holder: forward, left, right, step back, wait, turn left/right.
- ego_centered_route_map must summarize current_position, forward/left/right candidates, blocked_now, passable_after_clear, no_go_zones, overhead_hazards, retreat_path, and safest_next_step.
- predicted_motion_paths_by_entity must state likely path, contact/collision target, route conflict, and uncertainty for each likely moving entity.
- Prefer "wait then proceed" when a temporary blocker hides or occupies the likely route."""


DYNAMIC_CLEAR_VERIFIER_SYSTEM = """You are a dynamic-clear route verifier.
You receive the same image plus an initial route scan. Focus only on movable/dynamic blockers and the space they hide or occupy.
Return concise JSON with these exact keys:
dynamic_blockers, blocked_now, revealed_after_clear, side_openings_after_clear,
protected_turn_candidates, edge_or_drop_conflicts, remaining_static_blockers,
overhead_hazards_after_clear, after_clear_step_sequence, safest_after_clear_candidate,
uncertainty.

Rules:
- For each dynamic blocker, mentally remove it and infer what floor, opening, turn, hazard, or line-of-sight would be revealed.
- Inspect all directions, not only right or center. Check forward, left, right, and any diagonal/side opening near barriers, columns, carts, materials, or workers.
- Re-check overhead and shoulder-height hazards after dynamic blockers clear. A route can be wide on the floor but unsafe because of suspended/stored material above it.
- Do not choose the currently clearest lane by default. Choose the after-clear candidate only after comparing revealed space against remaining static hazards.
- A route can be "not safe now" but still become the best route after a person, carried material, cart, vehicle, or temporary object clears.
- If an opening is edge-adjacent, separate the exposed edge from a possible protected turn beside it.
- Express movement from the camera holder's view, such as: wait, move forward slightly, turn right/left, keep left/right, stop and verify."""


def _image_b64_from_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def analyze_route_surface(
    *,
    image_b64: str,
    question: str = "",
    mime_type: str = "image/jpeg",
) -> str:
    """Run a targeted VLM scan for route geometry and blockers."""
    llm = create_translator_client(max_tokens=1600)
    prompt = (
        "Question/context: "
        + (question or "Find the safest route through this scene.")
        + "\n\nFocus on route geometry, not general captioning."
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=ROUTE_SURFACE_SCAN_SYSTEM),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ]
            ),
        ]
    )
    initial_scan = str(getattr(response, "content", "") or "").strip()
    verifier = await llm.ainvoke(
        [
            SystemMessage(content=DYNAMIC_CLEAR_VERIFIER_SYSTEM),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Initial route scan:\n"
                            + initial_scan
                            + "\n\nVerify dynamic-clear routes for this same image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ]
            ),
        ]
    )
    dynamic_scan = str(getattr(verifier, "content", "") or "").strip()
    if not dynamic_scan:
        return initial_scan
    return (
        "INITIAL_ROUTE_SURFACE_SCAN:\n"
        + initial_scan
        + "\n\nDYNAMIC_CLEAR_VERIFIER:\n"
        + dynamic_scan
    )


@tool
async def route_surface_scan(
    image_path: str | None = None,
    image_b64: str | None = None,
    question: str = "",
) -> str:
    """Scan an image for walkable surface, exits/openings, blockers, and route candidates.

    Args:
        image_path: Local image path to scan. Optional if image_b64 is supplied.
        image_b64: Base64-encoded image bytes. Optional if image_path is supplied.
        question: Route or safety context to focus the scan.
    """
    try:
        payload = image_b64 or _image_b64_from_path(image_path or "")
        return await analyze_route_surface(image_b64=payload, question=question)
    except Exception as e:  # noqa: BLE001
        return f"Error in route_surface_scan: {e}"
