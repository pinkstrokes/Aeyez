"""Mechanics hazard scan tool for construction/engineering safety scenes."""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.seeingeye.llm.vllm_openai import create_translator_client


MECHANICS_HAZARD_SCAN_SYSTEM = """You are a mechanics hazard verifier for first-person construction and engineering safety.
Only inspect visible evidence. Do not give a final answer.
Return concise JSON with these exact keys:
mechanical_scene_model, load_path, supports_and_connections, unstable_or_deforming_objects,
stored_energy_sources, likely_failure_modes, predicted_motion_paths, line_of_fire_zones, pinch_crush_shear_zones,
falling_or_swinging_object_zones, exposed_people_or_routes, safe_standoff_or_route,
verification_needed.

Rules:
- Build a simple mechanics model before naming hazards: loads, supports, connections, direction of gravity/motion, and possible release paths.
- Look for unstable stacks, leaning materials, suspended or overhead items, temporary supports, scaffold members, ladders, hoists, carts, vehicles, pipes, hoses, cables, hydraulic/pneumatic systems, rotating tools, hot/pressurized equipment, and damaged/deformed parts.
- For each possible failure mode, state the likely motion: fall, swing, roll, slide, collapse, tip, pinch, crush, shear, recoil, pressure release, electrical contact, heat/steam release, or vehicle strike.
- predicted_motion_paths must map each movable or releasable object to a short-horizon path, contact/collision target, and exposed person/route if it moves.
- Identify line-of-fire zones and whether the camera holder or a visible worker/route is inside them.
- Separate visible evidence from inference. If the image does not show load/state/energy clearly, state the verification needed.
- Prefer safety-critical recall over neatness: include plausible severe hazards even if uncertain, but label uncertainty.
- Express route advice egocentrically: stop, wait, step back, keep left/right, avoid under/near, move forward only after verification."""


def _image_b64_from_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def analyze_mechanics_hazard(
    *,
    image_b64: str,
    question: str = "",
    mime_type: str = "image/jpeg",
) -> str:
    """Run a targeted VLM scan for mechanics-related safety hazards."""
    llm = create_translator_client(max_tokens=1800)
    prompt = (
        "Question/context: "
        + (question or "Identify mechanics-related construction safety hazards.")
        + "\n\nFocus on load paths, supports, stored energy, failure modes, and line-of-fire risk."
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=MECHANICS_HAZARD_SCAN_SYSTEM),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ]
            ),
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


@tool
async def mechanics_hazard_scan(
    image_path: str | None = None,
    image_b64: str | None = None,
    question: str = "",
) -> str:
    """Scan an image for load paths, stored energy, failure modes, and line-of-fire risk.

    Args:
        image_path: Local image path to scan. Optional if image_b64 is supplied.
        image_b64: Base64-encoded image bytes. Optional if image_path is supplied.
        question: Safety or engineering context to focus the scan.
    """
    try:
        payload = image_b64 or _image_b64_from_path(image_path or "")
        return await analyze_mechanics_hazard(image_b64=payload, question=question)
    except Exception as e:  # noqa: BLE001
        return f"Error in mechanics_hazard_scan: {e}"
