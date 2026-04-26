"""Spatial intelligence scan tool.

This tool implements a lightweight V*/SEAL-style visual search loop for
spatial tasks: first ask the VLM where to look, then zoom into those regions
and ask for candidate-by-candidate spatial evidence. It is intentionally
general: routes, engineering projections, diagrams, object relations, and
occlusion/dynamic-clear reasoning all use the same evidence structure.
"""

from __future__ import annotations

import ast
import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.seeingeye.llm.vllm_openai import create_translator_client


SPATIAL_REGION_SELECTION_SYSTEM = """You are a guided visual-search planner for spatial intelligence.
You receive an image with a numbered 4x4 grid. Select only the regions that need closer inspection.
Return JSON only with these exact keys:
regions, reason, must_verify.

Rules:
- Use grid cell ids 0-15. You may return single cells or rectangles as [top_left, bottom_right].
- Select regions that clarify spatial relations, occlusion, exits/openings, route candidates, object support/contact, perspective, projection, cross-section/hidden shape, labels/dimensions, hazards, or dynamic blockers.
- Prefer 2-4 focused regions. Use [0, 15] only if the whole image is genuinely needed.
- Include regions that could disprove the obvious first answer, not only regions that support it."""


SPATIAL_INTELLIGENCE_SYSTEM = """You are a spatial-intelligence observer.
Use only visible evidence from the full image and zoomed crops. Do not give the final answer.
Return concise JSON with these exact keys:
viewpoint_frame, global_layout, floor_or_support_surfaces, object_contact_and_containment,
occlusions_and_hidden_space, dynamic_before_after_hypotheses, route_candidates,
shape_or_projection_candidates, diagram_math_facts, candidate_verification_table,
local_zoom_evidence, geometric_consistency_checks, hazards_and_safety_controls,
uncertainties, translator_followup_needed.

Rules:
- Build a structural model of the scene/diagram before describing conclusions.
- For routes: enumerate all plausible candidates, including wait, step back, forward, left/right, forward-then-turn, side openings, and after-dynamic-clear routes. Compare floor, side clearance, head/shoulder clearance, overhead/falling-object exposure, edges, trip hazards, visibility, and retreat path.
- For movable people/objects/vehicles/equipment: explicitly reason about the current scene and the scene after they clear. A temporarily blocked opening may still be the best route later.
- For spatial puzzles, engineering drawings, diagrams, and blueprints: verify each candidate against visible features, projection rules, orientation, hidden edges, symmetry, scale, labels, and dimensions.
- For math/diagram questions: extract numeric labels, units, axes, legends, angles, loads, distances, and equations that a separate reasoner could compute from. Do not invent missing formulas.
- Do not fixate on the first visually salient path, shape, or option. Include evidence that could eliminate each plausible candidate.
- If a crop reveals ambiguity, say exactly what local detail remains unresolved."""


PROJECTION_SECTION_VERIFIER_SYSTEM = """You are a projection and cross-section verifier for engineering graphics.
Use only visible evidence from the image and zoomed crops. Do not give a final answer unless the evidence is decisive.
Return concise JSON with these exact keys:
source_view_features, section_cut_or_view_direction, hidden_and_visible_edges,
cross_section_fill_or_material_regions, option_alignment_table, eliminated_options,
best_supported_option_if_any, remaining_ambiguity, visual_checks_needed.

Rules:
- Reconstruct the requested viewpoint or section before comparing options.
- Track orientation carefully: left/right/front/top, viewer direction, cut-plane A-A arrows, rotation/mirroring, and hidden edges.
- For each answer option, compare concrete features: outer silhouette, holes/voids, ribs/webs, steps, arcs, symmetry, relative heights, line types, hatch/filled areas, and adjacency.
- Do not choose an option because it is visually salient. Eliminate options one by one using mismatched features.
- If option panels are small, state exactly which panel/feature needs a tighter crop.
- If the correct option cannot be determined, say why; do not overclaim."""


def _image_b64_from_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _decode_image(image_b64: str) -> np.ndarray:
    payload = base64.b64decode(image_b64)
    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def _encode_jpeg_b64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Could not encode crop as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _grid_overlay(image: np.ndarray, grid_size: int = 4) -> np.ndarray:
    overlay = image.copy()
    height, width = image.shape[:2]
    cell_h = height / grid_size
    cell_w = width / grid_size
    for i in range(1, grid_size):
        x = int(round(i * cell_w))
        y = int(round(i * cell_h))
        cv2.line(overlay, (x, 0), (x, height), (255, 0, 0), 3)
        cv2.line(overlay, (0, y), (width, y), (255, 0, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(2.2, min(width, height) / 850.0))
    thickness = max(2, int(font_scale * 1.4))
    for row in range(grid_size):
        for col in range(grid_size):
            cell_id = row * grid_size + col
            label = str(cell_id)
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                overlay,
                (cx - tw // 2 - 8, cy - th // 2 - 8),
                (cx + tw // 2 + 8, cy + th // 2 + 8),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (cx - tw // 2, cy + th // 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
    return cv2.addWeighted(image, 0.72, overlay, 0.28, 0)


def _parse_regions(text: str) -> list[list[int]]:
    try:
        obj = json.loads(text)
        regions = obj.get("regions", [])
    except Exception:  # noqa: BLE001
        match = re.search(r"\[[\s\d,\[\]-]+\]", text)
        if not match:
            return [[0, 15]]
        try:
            regions = ast.literal_eval(match.group(0))
        except Exception:  # noqa: BLE001
            return [[0, 15]]

    validated: list[list[int]] = []
    for item in regions:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            a, b = item
            if 0 <= a <= 15 and (b == -1 or 0 <= b <= 15):
                validated.append([a, b])
    return validated[:4] or [[0, 15]]


def _crop_region(image: np.ndarray, region: list[int], grid_size: int = 4) -> np.ndarray:
    height, width = image.shape[:2]
    a, b = region
    if b == -1:
        b = a
    r1, c1 = divmod(a, grid_size)
    r2, c2 = divmod(b, grid_size)
    top = min(r1, r2)
    bottom = max(r1, r2)
    left = min(c1, c2)
    right = max(c1, c2)
    cell_h = height / grid_size
    cell_w = width / grid_size
    margin_x = int(round(cell_w * 0.12))
    margin_y = int(round(cell_h * 0.12))
    x1 = max(0, int(round(left * cell_w)) - margin_x)
    y1 = max(0, int(round(top * cell_h)) - margin_y)
    x2 = min(width, int(round((right + 1) * cell_w)) + margin_x)
    y2 = min(height, int(round((bottom + 1) * cell_h)) + margin_y)
    return image[y1:y2, x1:x2]


def _looks_like_projection_task(question: str, options: str = "") -> bool:
    text = f"{question}\n{options}".lower()
    terms = (
        "left view",
        "right view",
        "front view",
        "top view",
        "view",
        "section",
        "cross-section",
        "cross section",
        "a-a",
        "projection",
        "orthographic",
        "hidden edge",
        "main view",
        "select the correct",
        "correct view",
        "correct section",
        "剖面",
        "截面",
        "左视图",
        "右视图",
        "主视图",
        "俯视图",
        "投影",
    )
    return any(term in text for term in terms)


async def analyze_spatial_intelligence(
    *,
    image_b64: str,
    question: str = "",
    options: str = "",
    mime_type: str = "image/jpeg",
) -> str:
    """Run guided visual search plus spatial candidate verification."""
    image = _decode_image(image_b64)
    overlay_b64 = _encode_jpeg_b64(_grid_overlay(image))
    llm = create_translator_client(max_tokens=1800)
    context = (
        "Question/context: "
        + (question or "Analyze the spatial structure of this image.")
        + ("\nOptions:\n" + options if options else "")
    )
    selection = await llm.ainvoke(
        [
            SystemMessage(content=SPATIAL_REGION_SELECTION_SYSTEM),
            HumanMessage(
                content=[
                    {"type": "text", "text": context},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{overlay_b64}"},
                    },
                ]
            ),
        ]
    )
    selection_text = str(getattr(selection, "content", "") or "").strip()
    regions = _parse_regions(selection_text)

    verifier_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                context
                + "\n\nGuided visual-search region selection:\n"
                + selection_text
                + "\n\nUse the full image plus each zoomed crop to build spatial evidence."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
        },
    ]
    for idx, region in enumerate(regions, start=1):
        crop_b64 = _encode_jpeg_b64(_crop_region(image, region))
        verifier_content.append(
            {"type": "text", "text": f"Zoom crop {idx}, grid region {region}"}
        )
        verifier_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"},
            }
        )

    verification = await llm.ainvoke(
        [
            SystemMessage(content=SPATIAL_INTELLIGENCE_SYSTEM),
            HumanMessage(content=verifier_content),
        ]
    )
    verification_text = str(getattr(verification, "content", "") or "").strip()
    projection_text = ""
    if _looks_like_projection_task(question, options):
        projection = await llm.ainvoke(
            [
                SystemMessage(content=PROJECTION_SECTION_VERIFIER_SYSTEM),
                HumanMessage(
                    content=[
                        *verifier_content,
                        {
                            "type": "text",
                            "text": (
                                "Now run the projection/section verifier. "
                                "Compare each visible answer option against the requested view or cut."
                            ),
                        },
                    ]
                ),
            ]
        )
        projection_text = str(getattr(projection, "content", "") or "").strip()
    return (
        "SPATIAL_GUIDED_REGION_SELECTION:\n"
        + selection_text
        + "\n\nSPATIAL_INTELLIGENCE_VERIFICATION:\n"
        + verification_text
        + (
            "\n\nPROJECTION_SECTION_VERIFICATION:\n" + projection_text
            if projection_text
            else ""
        )
    )


@tool
async def spatial_intelligence_scan(
    image_path: str | None = None,
    image_b64: str | None = None,
    question: str = "",
    options: str = "",
) -> str:
    """Zoom and verify spatial evidence for routes, diagrams, shapes, and hazards.

    Args:
        image_path: Local image path to scan. Optional if image_b64 is supplied.
        image_b64: Base64-encoded image bytes. Optional if image_path is supplied.
        question: Question or task context for selecting regions.
        options: Optional answer options as a newline-separated string.
    """
    try:
        payload = image_b64 or _image_b64_from_path(image_path or "")
        return await analyze_spatial_intelligence(
            image_b64=payload,
            question=question,
            options=options,
        )
    except Exception as e:  # noqa: BLE001
        return f"Error in spatial_intelligence_scan: {e}"
