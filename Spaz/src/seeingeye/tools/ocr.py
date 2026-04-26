"""OCR tool — ported from ``src/multi-agent/app/tool/ocr.py``.

TPS-03 (simplified 2026-04-13): semantic equivalence with the original.
Phase 6 parity gate validates correctness on real benchmarks.

The Azure endpoint URL, headers, default language (``unk``), and the
``_extract_text_from_result`` line-joining algorithm are preserved. The
subscription key is read from the environment.  The old ``ToolResult(output=...)`` wrapper is
collapsed into a plain ``str`` return — langchain-core's ``@tool``
wraps the string into the standard ``ToolMessage`` envelope that
``bind_tools`` consumers expect.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from langchain_core.tools import tool

# Azure Computer Vision configuration.
_SUBSCRIPTION_KEY = os.getenv("AZURE_OCR_SUBSCRIPTION_KEY", "")
_ENDPOINT = os.getenv(
    "AZURE_OCR_ENDPOINT",
    "https://haoqi.cognitiveservices.azure.com/",
)
_OCR_URL = _ENDPOINT + "vision/v2.1/ocr"
_HEADERS = {
    "Ocp-Apim-Subscription-Key": _SUBSCRIPTION_KEY,
    "Content-Type": "application/octet-stream",
}
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


def _extract_text_from_result(result: dict) -> str:
    """Extract plain text from Azure OCR API result. Verbatim port of the
    old ``OCR._extract_text_from_result`` method."""
    text_lines: list[str] = []
    for region in result.get("regions", []):
        for line in region.get("lines", []):
            line_text = ""
            for word in line.get("words", []):
                line_text += word.get("text", "") + " "
            if line_text.strip():
                text_lines.append(line_text.strip())
    return "\n".join(text_lines)


@tool
async def ocr(
    image_path: str,
    language: str = "unk",
    detect_orientation: bool = True,
) -> str:
    """Extract text content from image files, supports text recognition in multiple languages.

    Args:
        image_path: Path to the image file (supports relative and absolute paths).
        language: Language code for recognition, e.g., 'zh-Hans' (Simplified Chinese), 'en' (English), 'unk' (auto-detect).
        detect_orientation: Whether to detect image orientation.
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image file does not exist: {image_path}"

        if image_file.suffix.lower() not in _ALLOWED_EXTENSIONS:
            return (
                f"Unsupported image format: {image_file.suffix}. "
                f"Supported formats: {', '.join(_ALLOWED_EXTENSIONS)}"
            )

        if not _SUBSCRIPTION_KEY:
            return "Azure OCR is not configured: set AZURE_OCR_SUBSCRIPTION_KEY."

        params = {
            "language": language,
            "detectOrientation": str(detect_orientation).lower(),
        }

        with open(image_file, "rb") as f:
            image_data = f.read()

        response = requests.post(
            _OCR_URL,
            headers=_HEADERS,
            params=params,
            data=image_data,
            timeout=30,
        )

        if response.status_code != 200:
            return f"OCR API call failed: {response.status_code} - {response.text}"

        result = response.json()
        extracted_text = _extract_text_from_result(result)

        # Preserve the trailing blank line from the original f-string.
        return f"OCR Results:\nExtracted Text:\n{extracted_text}\n\n"

    except FileNotFoundError:
        return f"File not found: {image_path}"
    except requests.exceptions.RequestException as e:
        return f"Network request error: {str(e)}"
    except Exception as e:  # noqa: BLE001
        return f"OCR processing error: {str(e)}"
