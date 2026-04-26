"""SmartGridCaption tool — ported from ``src/multi-agent/app/tool/smart_grid_caption.py``.

TPS-03 (simplified 2026-04-13): algorithm + prompt-string fidelity with the
original; Phase 6 parity gate validates correctness on real benchmarks.

What was preserved verbatim:
  - 4x4 grid overlay geometry (BGR colors, alpha blend, patch numbering,
    adaptive font scaling, ``with_grid/`` subdirectory layout, and the
    ``_with_grid.png`` filename suffix).
  - ``_create_selection_prompt``, ``_create_caption_prompt``,
    ``_combine_captions`` — every character copied as-is.
  - ``_parse_patch_selection`` regex + ``ast.literal_eval`` fallback chain,
    including the ``[[5, 10]]`` ultimate fallback.
  - ``_calculate_crop_coordinates`` arithmetic and boundary handling.
  - ``_validate_and_fix_coordinates`` rectangle validation / fix-up.
  - The multi-line emoji output template (🔍 📊 ✅ 🖼️ 📝 🎯).
  - Default ``output_dir`` value ``"./smart_grid_output"``.

What changed (the point of Phase 2):
  - The two nested VLM calls (``_locate_relevant_regions`` and
    ``_generate_contextual_caption``) go through
    ``seeingeye.llm.vllm_openai.create_translator_client`` instead of
    the old ``app.llm.LLM()`` wrapper.  This keeps all vLLM
    ``extra_body`` routing centralized (Pitfall #1 mitigation).
"""

from __future__ import annotations

import ast
import base64
import re
from pathlib import Path
from typing import Any

import cv2
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.seeingeye.llm.vllm_openai import create_translator_client


# ---------------------------------------------------------------------------
# Step 1 — Grid overlay (verbatim geometry)
# ---------------------------------------------------------------------------


async def _generate_grid_overlay(image_path: str) -> dict[str, Any]:
    """Step 1: Generate 4x4 grid overlay on the image.

    Returns a dict with either an ``error`` key or the original's
    ``{processed_image_path, ...}`` payload.
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            return {"error": f"Image file does not exist: {image_path}"}

        image = cv2.imread(str(image_file))
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        height, width = image.shape[:2]
        overlay = image.copy()

        patch_height = height // 4
        patch_width = width // 4

        # BGR colors — verbatim from the old tool.
        grid_color = (255, 0, 0)  # Red grid lines
        text_color = (255, 255, 255)  # White text
        background_color = (0, 0, 0)  # Black text background

        # 3 interior vertical + 3 interior horizontal grid lines.
        for i in range(1, 4):
            cv2.line(
                overlay,
                (i * patch_width, 0),
                (i * patch_width, height),
                grid_color,
                3,
            )
            cv2.line(
                overlay,
                (0, i * patch_height),
                (width, i * patch_height),
                grid_color,
                3,
            )

        # Patch numbers 0..15.
        patch_id = 0
        for row in range(4):
            for col in range(4):
                center_x = col * patch_width + patch_width // 2
                center_y = row * patch_height + patch_height // 2

                font = cv2.FONT_HERSHEY_SIMPLEX
                base_font_scale = min(width, height) / 600.0
                font_scale = max(0.8, min(3.0, base_font_scale))
                thickness = max(2, int(font_scale * 1.5))

                text = str(patch_id)
                (text_width, text_height), _baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )

                bg_x1 = center_x - text_width // 2 - 10
                bg_y1 = center_y - text_height // 2 - 10
                bg_x2 = center_x + text_width // 2 + 10
                bg_y2 = center_y + text_height // 2 + 10
                cv2.rectangle(
                    overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1
                )

                text_x = center_x - text_width // 2
                text_y = center_y + text_height // 2
                cv2.putText(
                    overlay, text, (text_x, text_y), font, font_scale, text_color, thickness
                )

                patch_id += 1

        alpha = 0.5
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # ``with_grid/`` subdirectory — same file-layout contract as old code.
        image_dir = image_file.parent
        with_grid_dir = image_dir / "with_grid"
        with_grid_dir.mkdir(exist_ok=True)

        original_name = image_file.stem
        output_filename = f"{original_name}_with_grid.png"
        output_path = with_grid_dir / output_filename

        success = cv2.imwrite(str(output_path), blended)
        if not success:
            return {"error": f"Failed to save processed image to: {output_path}"}

        return {
            "processed_image_path": str(output_path),
            "original_image_path": image_path,
            "grid_size": "4x4",
            "total_patches": 16,
            "image_dimensions": {"width": width, "height": height},
            "patch_dimensions": {"width": patch_width, "height": patch_height},
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Error in grid generation: {str(e)}"}


# ---------------------------------------------------------------------------
# Step 2 — VLM patch selection (nested VLM call — Phase 2 client)
# ---------------------------------------------------------------------------


def _create_selection_prompt(
    question: str, options: list[str] | None, context: str
) -> str:
    """Create the patch selection prompt for LLM. Verbatim template.

    The text below references a 4x4 patch grid with patch numbers 0-15 —
    changing any character here would shift model behavior.
    """
    options_section = ""
    if options:
        options_section = f"\nAnswer Options: {', '.join(options)}"

    context_section = f"\n\nPrevious Context:\n{context}" if context else ""

    prompt = f"""You need to select the most relevant patches from this 4x4 grid-overlay image to answer the given question.

Question: {question}{options_section}{context_section}

The image shows a 4x4 grid with patches numbered 0-15:
- Row 1: Patches 0, 1, 2, 3 (top row)
- Row 2: Patches 4, 5, 6, 7
- Row 3: Patches 8, 9, 10, 11
- Row 4: Patches 12, 13, 14, 15 (bottom row)

Your task:
1. Analyze which patch(es) contain information most relevant to answering the question
2. You can select:
   - Single patches: [patch_number, -1]
   - Rectangular regions: [top_left_patch, bottom_right_patch]
   - Multiple separate regions

Instructions:
- Select the MINIMUM number of patches that contain the relevant information
- Prioritize quality over quantity - better to crop precisely than include noise
- Consider text, charts, diagrams, numbers, or other visual elements needed for the question
- If the whole image is needed, you can select the entire grid: [0, 15]

Format your response EXACTLY as a Python list:
[[top_left1, bottom_right1], [top_left2, bottom_right2], [single_patch, -1]]

Examples:
- Single patch 5: [[5, -1]]
- Rectangle from patch 1 to 6: [[1, 6]]
- Two separate regions: [[2, 6], [10, -1]]
- Whole image: [[0, 15]]

Your selection:"""

    return prompt


def _parse_patch_selection(llm_response: str) -> list[list[int]]:
    """Parse LLM response to extract patch coordinates. Verbatim logic
    including the regex, the ``ast.literal_eval`` path, and the
    ``[[5, 10]]`` ultimate fallback."""
    try:
        list_pattern = r"\[\s*\[.*?\]\s*\]"
        matches = re.findall(list_pattern, llm_response, re.DOTALL)

        if matches:
            list_str = matches[0]
            try:
                parsed_list = ast.literal_eval(list_str)
                if isinstance(parsed_list, list):
                    validated_list: list[list[int]] = []
                    for item in parsed_list:
                        if isinstance(item, list) and len(item) == 2:
                            top_left, bottom_right = item
                            if (
                                0 <= top_left <= 15
                                and (bottom_right == -1 or 0 <= bottom_right <= 15)
                            ):
                                validated_list.append([top_left, bottom_right])
                    if validated_list:
                        return validated_list
            except Exception:  # noqa: BLE001
                pass

        numbers = re.findall(r"\d+", llm_response)
        if numbers:
            patch_numbers = [int(n) for n in numbers if 0 <= int(n) <= 15]
            if patch_numbers:
                if len(patch_numbers) >= 2:
                    return [[patch_numbers[0], patch_numbers[1]]]
                else:
                    return [[patch_numbers[0], -1]]

        # Ultimate fallback: center region.
        return [[5, 10]]
    except Exception:  # noqa: BLE001
        return [[5, 10]]


async def _ask_translator_vlm(prompt_text: str, image_b64: str) -> str:
    """Shared helper: send ``prompt_text`` + a base64 image through the
    Phase 2 Translator client (Qwen2.5-VL-3B on vLLM port 8000).

    Returns the text content of the AI reply.
    """
    client = create_translator_client(temperature=0.1)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )
    ai_msg = await client.ainvoke([msg])
    return ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)


async def _locate_relevant_regions(
    grid_image_path: str,
    question: str,
    options: list[str] | None,
    context: str,
) -> dict[str, Any]:
    """Step 2: ask the Translator VLM which grid patches matter."""
    try:
        grid_image_file = Path(grid_image_path)
        if not grid_image_file.exists():
            return {"error": f"Grid image file does not exist: {grid_image_path}"}

        try:
            with open(grid_image_file, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
        except Exception as e:  # noqa: BLE001
            return {"error": f"Failed to read grid image: {str(e)}"}

        selection_prompt = _create_selection_prompt(question, options, context)
        response = await _ask_translator_vlm(selection_prompt, image_b64)

        patch_coordinates = _parse_patch_selection(response)

        return {
            "grid_image_path": grid_image_path,
            "question": question,
            "options": options,
            "context": context,
            "selected_patches": patch_coordinates,
            "llm_response": response,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Error in region location: {str(e)}"}


# ---------------------------------------------------------------------------
# Step 3 — Crop + contextual captions (verbatim pixel math + prompts)
# ---------------------------------------------------------------------------


def _calculate_crop_coordinates(
    top_left: int, bottom_right: int, width: int, height: int
) -> dict[str, int]:
    """Calculate pixel coordinates for cropping. Verbatim arithmetic."""
    patch_width = width // 4
    patch_height = height // 4

    if bottom_right == -1:
        row = top_left // 4
        col = top_left % 4

        x_start = col * patch_width
        x_end = (col + 1) * patch_width if col < 3 else width
        y_start = row * patch_height
        y_end = (row + 1) * patch_height if row < 3 else height
    else:
        top_left_row, top_left_col = top_left // 4, top_left % 4
        bottom_right_row, bottom_right_col = bottom_right // 4, bottom_right % 4

        x_start = top_left_col * patch_width
        x_end = (
            (bottom_right_col + 1) * patch_width if bottom_right_col < 3 else width
        )
        y_start = top_left_row * patch_height
        y_end = (
            (bottom_right_row + 1) * patch_height if bottom_right_row < 3 else height
        )

    return {
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
    }


def _create_caption_prompt(
    question: str,
    options: list[str] | None,
    context: str,
    region_id: int,
    top_left: int,
    bottom_right: int,
) -> str:
    """Create contextual caption prompt for the cropped region. Verbatim."""
    patch_description = (
        f"patch {top_left}" if bottom_right == -1 else f"patches {top_left} to {bottom_right}"
    )

    options_section = ""
    if options:
        options_section = f"\nAnswer Options: {', '.join(options)}"

    context_section = (
        f"\n\nPrevious Analysis Context:\n{context}" if context else ""
    )

    prompt = f"""Analyze this image region to answer the question: {question}{options_section}{context_section}

This is region {region_id} containing {patch_description}.

Describe exactly what you see in this image:

**Visual Description:**
What objects, people, colors, and details are visible in this image?

**Text/Numbers Extracted:**
Any text, labels, numbers, or written content in the image (write "None visible" if no text).

**Structural Analysis:**
How are elements arranged and positioned in the image?

**Relevance to Question:**
How does what you see help answer: {question}

**Option Support:**
Which answer option does this image support and why?

Be direct and specific about what you observe in the image."""

    return prompt


def _combine_captions(
    captions: list[str],
    question: str,
    options: list[str] | None,
    context: str,
) -> str:
    """Combine individual captions into comprehensive analysis. Verbatim
    template, including the ``'=' * 50`` separator and the trailing summary."""
    options_text = ""
    if options:
        options_text = f"\nAnswer Options: {', '.join(options)}"

    header = f"""COMPREHENSIVE VISUAL ANALYSIS

Question: {question}{options_text}

DETAILED REGION ANALYSIS:
"""

    combined = header

    for i, caption in enumerate(captions):
        combined += f"\n{'=' * 50}\nREGION {i}:\n{'=' * 50}\n{caption}\n"

    summary = """
You can now improve your caption based on region caption.
"""

    combined += summary

    return combined.strip()


def _validate_and_fix_coordinates(
    patch_coordinates: list[list[int]],
) -> list[list[int]]:
    """Validate and fix patch coordinates. Verbatim port."""
    valid_coordinates: list[list[int]] = []

    for coord_pair in patch_coordinates:
        if not isinstance(coord_pair, list) or len(coord_pair) != 2:
            continue

        top_left, bottom_right = coord_pair

        if not (0 <= top_left <= 15):
            continue

        if bottom_right == -1:
            valid_coordinates.append([top_left, -1])
            continue

        if top_left == bottom_right:
            valid_coordinates.append([top_left, -1])
            continue

        if not (0 <= bottom_right <= 15):
            continue

        top_left_row, top_left_col = top_left // 4, top_left % 4
        bottom_right_row, bottom_right_col = bottom_right // 4, bottom_right % 4

        if top_left_row <= bottom_right_row and top_left_col <= bottom_right_col:
            valid_coordinates.append([top_left, bottom_right])
        else:
            valid_coordinates.append([top_left, -1])
            if top_left != bottom_right:
                valid_coordinates.append([bottom_right, -1])

    return valid_coordinates


async def _generate_contextual_caption(
    crop_path: str,
    question: str,
    options: list[str] | None,
    context: str,
    region_id: int,
    top_left: int,
    bottom_right: int,
) -> str:
    """Generate detailed caption with full context for the cropped region."""
    try:
        with open(crop_path, "rb") as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        caption_prompt = _create_caption_prompt(
            question, options, context, region_id, top_left, bottom_right
        )
        caption = await _ask_translator_vlm(caption_prompt, image_b64)
        return caption
    except Exception as e:  # noqa: BLE001
        return f"Caption generation failed: {str(e)}"


async def _crop_and_generate_captions(
    original_image_path: str,
    patch_coordinates: list[list[int]],
    question: str,
    options: list[str] | None,
    context: str,
    output_dir: str,
) -> dict[str, Any]:
    """Step 3: Crop regions and generate detailed captions."""
    try:
        image_file = Path(original_image_path)
        if not image_file.exists():
            return {
                "error": f"Original image file does not exist: {original_image_path}"
            }

        image = cv2.imread(str(image_file))
        if image is None:
            return {
                "error": f"Could not load original image: {original_image_path}"
            }

        height, width = image.shape[:2]

        crop_dir = Path(output_dir) / "cropped_regions"
        crop_dir.mkdir(parents=True, exist_ok=True)

        valid_coordinates = _validate_and_fix_coordinates(patch_coordinates)
        if not valid_coordinates:
            return {
                "error": f"All patch coordinates are invalid: {patch_coordinates}"
            }

        cropped_regions: list[dict[str, Any]] = []
        captions: list[str] = []

        for i, (top_left, bottom_right) in enumerate(valid_coordinates):
            crop_coords = _calculate_crop_coordinates(
                top_left, bottom_right, width, height
            )

            cropped_image = image[
                crop_coords["y_start"] : crop_coords["y_end"],
                crop_coords["x_start"] : crop_coords["x_end"],
            ]

            if cropped_image.size == 0:
                return {
                    "error": (
                        f"Cropped region {i} is empty. Invalid coordinates: "
                        f"[{top_left},{bottom_right}]. Crop coords: {crop_coords}"
                    )
                }

            crop_filename = f"cropped_region_{i}_{top_left}_{bottom_right}.png"
            crop_path = crop_dir / crop_filename

            success = cv2.imwrite(str(crop_path), cropped_image)
            if not success:
                return {
                    "error": (
                        f"Failed to save cropped region {i}: {crop_path}. "
                        f"Image shape: {cropped_image.shape}"
                    )
                }

            caption = await _generate_contextual_caption(
                str(crop_path), question, options, context, i, top_left, bottom_right
            )

            cropped_regions.append(
                {
                    "region_id": i,
                    "patch_coordinates": [top_left, bottom_right],
                    "pixel_coordinates": crop_coords,
                    "cropped_image_path": str(crop_path),
                    "caption": caption,
                }
            )
            captions.append(caption)

        combined_analysis = _combine_captions(captions, question, options, context)

        return {
            "original_image_path": original_image_path,
            "question": question,
            "options": options,
            "context": context,
            "patch_coordinates": patch_coordinates,
            "cropped_regions": cropped_regions,
            "individual_captions": captions,
            "combined_analysis": combined_analysis,
            "output_directory": str(crop_dir),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Error in cropping and captioning: {str(e)}"}


# ---------------------------------------------------------------------------
# Public ``@tool`` entrypoint
# ---------------------------------------------------------------------------


@tool
async def smart_grid_caption(
    image_path: str,
    question: str,
    options: list[str] | None = None,
    context: str = "",
    output_dir: str = "./smart_grid_output",
) -> str:
    """Intelligent image analysis tool that creates grid overlay, selects relevant regions via LLM, and generates detailed captions. If you do not find any relevant information in the image, please use this tool to analyze the image and generate a caption.

    Args:
        image_path: Path to the input image file.
        question: The question that needs to be answered.
        options: List of answer options (optional).
        context: Additional context or previous analysis.
        output_dir: Directory to save intermediate and final outputs.
    """
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        options_text = ""
        if options:
            options_text = f"\nOptions: {options}"

        # Step 1: Generate 4x4 grid overlay.
        grid_data = await _generate_grid_overlay(image_path)
        if "error" in grid_data:
            return f"Grid generation failed: {grid_data['error']}"

        grid_image_path = grid_data.get("processed_image_path", "")

        # Step 2: LLM selects relevant patches.
        locate_data = await _locate_relevant_regions(
            grid_image_path, question, options, context
        )
        if "error" in locate_data:
            return f"Region location failed: {locate_data['error']}"

        patch_coordinates = locate_data.get("selected_patches", [])

        # Step 3: Crop + generate contextual captions.
        crop_data = await _crop_and_generate_captions(
            image_path, patch_coordinates, question, options, context, output_dir
        )
        if "error" in crop_data:
            return f"Cropping and captioning failed: {crop_data['error']}"

        output_text = f"""Smart Grid Caption Analysis Completed:

🔍 Question: {question}{options_text}

📊 Workflow Results:
1. ✅ Grid Overlay Generated: {grid_image_path}
2. ✅ Regions Selected: {patch_coordinates}
3. ✅ Crops Generated: {len(crop_data.get('cropped_regions', []))} regions

🖼️ Cropped Images:
{chr(10).join([f"- {region['cropped_image_path']}" for region in crop_data.get('cropped_regions', [])])}

📝 Combined Analysis:
{crop_data.get('combined_analysis', '')}

🎯 This analysis provides focused visual information relevant to answering the question."""

        return output_text

    except Exception as e:  # noqa: BLE001
        return f"Error in smart_grid_caption workflow: {str(e)}"
