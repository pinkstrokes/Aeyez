"""Egocentric action and motion-path scan tool."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.seeingeye.llm.vllm_openai import create_translator_client


ACTION_MOTION_SCAN_SYSTEM = """You are an egocentric action and motion-path scanner.
Only inspect visible evidence. Do not give a final answer.
Use this action formula:
Action = hand pose + active object + contact target + temporal motion + scene context.

Return concise JSON with these exact keys:
egocentric_action_model, hand_pose, active_object, contact_target, temporal_motion,
scene_context, movable_entities, predicted_motion_paths, contact_or_collision_targets,
route_conflicts, line_of_fire_or_release_paths, action_risk_level, safest_next_action,
verification_needed.

Rules:
- For hands/arms/tools/materials: describe hand pose, grip, active object, contact target, and likely next motion.
- For movable objects/people/vehicles/equipment: predict the likely path from visible pose, orientation, contact, support, gravity, scene affordances, and recent frame changes if frames are provided.
- Separate visible facts from predicted motion. Use short horizons: next seconds, not long-term plans.
- A predicted path may be human walking, carried object swing, cart/vehicle movement, falling/rolling/sliding object, door/swing path, tool motion, hose/cable drag, or pressure/energy release.
- Identify whether the camera holder's route intersects any predicted motion path, contact target, line-of-fire, pinch/crush/shear zone, or falling-object zone.
- If motion is ambiguous from a single image, list competing likely paths and the visual checks needed."""


def _image_b64_from_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def analyze_action_motion(
    *,
    image_b64: str | None = None,
    frames: list[dict[str, Any]] | None = None,
    question: str = "",
    mime_type: str = "image/jpeg",
) -> str:
    """Run an egocentric action/motion-path scan on one image or ordered frames."""
    llm = create_translator_client(max_tokens=1800)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Question/context: "
                + (question or "Predict visible actions and short-horizon motion paths.")
                + "\n\nFocus on egocentric action structure and likely moving-object paths."
            ),
        }
    ]
    if frames:
        for idx, frame in enumerate(frames, start=1):
            ts = frame.get("timestamp_s")
            if ts is not None:
                content.append({"type": "text", "text": f"Frame {idx} at {ts:.3f}s"})
            frame_mime = frame.get("mime_type") or mime_type
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{frame_mime};base64,{frame['b64']}"},
                }
            )
    elif image_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            }
        )
    else:
        raise ValueError("image_b64 or frames must be supplied")

    response = await llm.ainvoke(
        [
            SystemMessage(content=ACTION_MOTION_SCAN_SYSTEM),
            HumanMessage(content=content),
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


@tool
async def action_motion_scan(
    image_path: str | None = None,
    image_b64: str | None = None,
    question: str = "",
) -> str:
    """Scan an image for egocentric action structure and likely moving-object paths.

    Args:
        image_path: Local image path to scan. Optional if image_b64 is supplied.
        image_b64: Base64-encoded image bytes. Optional if image_path is supplied.
        question: Action, safety, or route context to focus the scan.
    """
    try:
        payload = image_b64 or _image_b64_from_path(image_path or "")
        return await analyze_action_motion(image_b64=payload, question=question)
    except Exception as e:  # noqa: BLE001
        return f"Error in action_motion_scan: {e}"
