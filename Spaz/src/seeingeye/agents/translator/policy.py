"""VCoT parser for the Translator agent — verbatim port of
``src/multi-agent/app/agent/toolcall.py:19-213``.

The Translator VLM (Qwen2.5-VL-3B served via vLLM) cannot reliably emit
OpenAI-shaped tool calls because the hermes parser has known bugs on this
model (Pitfall #1, QwenLM/Qwen3-VL#1093). Instead, the Translator emits
free-form text containing one of three call shapes, in this priority order:

    Priority 1 — standard XML:        <tool_call>{json}</tool_call>
    Priority 2 — repeated-tag XML:    <tool_call>{json}<tool_call>{json}
    Priority 3 — loose JSON:          {"name": ..., "arguments": ...}

The fallback at the bottom of :func:`parse_tool_calls_multiple_formats`
returns the first non-empty parse even when arguments are empty (rare but
real cases where the model emits a call with default args).

This module is intentionally a near-byte-identical port of the old code.
The only required change: the old code returns ``app.schema.ToolCall``
(Pydantic with nested ``Function(name, arguments_json_str)``); this port
returns a local :class:`ParsedToolCall` dataclass with ``args`` already a
``dict``. The downstream :func:`translator_tools_node` calls
``tool.ainvoke(args)`` directly, so keeping ``args`` as a dict avoids a
``json.dumps -> json.loads`` round trip.

The three-priority ordering and the truncated-JSON recovery in
:func:`process_tool_call_matches` are the paper-reproducibility contract.
DO NOT REORDER. DO NOT SIMPLIFY.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

try:  # loguru is the project standard; fall back to stdlib if missing.
    from loguru import logger
except ImportError:  # pragma: no cover - loguru is in requirements.txt
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


VALID_TRANSLATOR_TOOL_NAMES = frozenset(
    {
        "ocr",
        "read_table",
        "smart_grid_caption",
        "action_motion_scan",
        "route_surface_scan",
        "mechanics_hazard_scan",
        "spatial_intelligence_scan",
    }
)


@dataclass(frozen=True)
class ParsedToolCall:
    """Lightweight tool-call envelope decoupled from ``app.schema``.

    Replaces the old ``ToolCall(id=..., function=Function(name=...,
    arguments=json.dumps(args)))`` triple. ``args`` is the parsed Python
    ``dict`` — :func:`translator_tools_node` passes it straight into
    ``tool.ainvoke(args)``.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


def valid_translator_tool_calls(
    tool_calls: list[ParsedToolCall],
) -> list[ParsedToolCall]:
    """Keep only executable Translator tools.

    Some OpenAI-compatible models copy placeholder JSON such as
    ``{"name": "...", "arguments": {...}}`` from prompt examples. Treating
    those as real calls pollutes the SIR with "Unknown tool" errors and
    causes extra translator/reasoner turns.
    """
    return [tc for tc in tool_calls if tc.name in VALID_TRANSLATOR_TOOL_NAMES]


def parse_tool_calls_multiple_formats(content: str) -> list[ParsedToolCall]:
    """Try multiple parsing formats and use the first one that succeeds
    WITH VALID ARGUMENTS.

    Priority: 1) Standard XML format, 2) Repeated-tag XML format,
    3) Loose JSON format.
    """
    if not content or not isinstance(content, str):
        return []

    def has_valid_arguments(tool_calls: list[ParsedToolCall]) -> bool:
        """Check if any tool call has non-empty arguments."""
        return any(tc.args for tc in tool_calls)

    # Priority 1: Try XML format with proper closing tags
    tool_calls = parse_xml_tool_calls_standard(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Priority 2: Try XML format with repeated opening tags
    tool_calls = parse_xml_tool_calls_repeated_tags(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Priority 3: Try loose JSON extraction
    tool_calls = parse_loose_json_tool_calls(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Fallback: If we found tool calls but none with valid arguments, return
    # the first successful parse. This handles cases where empty arguments
    # are actually valid (rare but possible).
    for _, parser_func in [
        ("standard XML", parse_xml_tool_calls_standard),
        ("repeated-tag XML", parse_xml_tool_calls_repeated_tags),
        ("loose JSON", parse_loose_json_tool_calls),
    ]:
        tool_calls = parser_func(content)
        if tool_calls:
            return tool_calls
    return []


def parse_xml_tool_calls_standard(content: str) -> list[ParsedToolCall]:
    """Parse standard ``<tool_call>...</tool_call>`` XML format."""
    # Find all <tool_call>...</tool_call> blocks
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    # Also look for incomplete tool calls (missing closing tag due to truncation)
    incomplete_pattern = r"<tool_call>\s*(.*?)$"
    if not matches:
        incomplete_matches = re.findall(
            incomplete_pattern, content, re.DOTALL | re.IGNORECASE
        )
        if incomplete_matches:
            logger.warning("Found incomplete tool call (possibly truncated)")
            matches = incomplete_matches

    return process_tool_call_matches(matches, "standard XML")


def parse_xml_tool_calls_repeated_tags(content: str) -> list[ParsedToolCall]:
    """Parse ``<tool_call>...<tool_call>`` format (repeated opening tags).

    Extract JSON objects that appear between ``<tool_call>`` tags.
    """
    # Split by <tool_call> tags and extract JSON content
    segments = re.split(r"<tool_call>", content, flags=re.IGNORECASE)

    json_matches: list[str] = []
    for segment in segments[1:]:  # Skip the first segment (before any <tool_call>)
        segment = segment.strip()
        if not segment:
            continue

        # Look for JSON object at the start of the segment
        # Find the complete JSON object using brace matching
        json_content = extract_first_complete_json(segment)
        if json_content:
            json_matches.append(json_content)

    return process_tool_call_matches(json_matches, "repeated-tag XML")


def extract_first_complete_json(text: str) -> str:
    """Extract the first complete JSON object from text by matching braces."""
    text = text.strip()
    if not text.startswith("{"):
        return ""

    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    return text[: i + 1]

    return ""  # Incomplete or malformed JSON


def parse_loose_json_tool_calls(content: str) -> list[ParsedToolCall]:
    """Extract JSON objects that look like tool calls without XML tags."""
    # Look for JSON objects with "name" and "arguments" fields - improved
    # pattern. This pattern handles nested braces better.
    json_pattern = (
        r'\{(?:[^{}]|{[^{}]*})*"name"(?:[^{}]|{[^{}]*})*'
        r'"arguments"(?:[^{}]|{[^{}]*})*\}'
    )
    matches = re.findall(json_pattern, content, re.DOTALL)

    # If that doesn't work, try a simpler pattern
    if not matches:
        simple_pattern = r'\{.*?"name".*?"arguments".*?\}'
        matches = re.findall(simple_pattern, content, re.DOTALL)

    return process_tool_call_matches(matches, "loose JSON")


def process_tool_call_matches(
    matches: list[str], _: str
) -> list[ParsedToolCall]:
    """Process matched strings into :class:`ParsedToolCall` objects.

    Replicates the truncated-JSON recovery branch of the old code: when a
    match doesn't end with ``}``, count unbalanced opening braces and
    append closing braces accordingly. This recovery is part of the
    paper-reproducibility contract — do not simplify.
    """
    tool_calls: list[ParsedToolCall] = []

    for i, match in enumerate(matches):
        try:
            match_content = match.strip()

            # Try to fix common JSON truncation issues
            if not match_content.endswith("}"):
                # Attempt to close incomplete JSON
                brace_count = match_content.count("{") - match_content.count("}")
                if brace_count > 0:
                    match_content += "}" * brace_count
                    logger.warning(
                        "Attempted to fix truncated JSON by adding "
                        f"{brace_count} closing braces"
                    )

            # Parse the JSON content
            tool_data = json.loads(match_content)

            if isinstance(tool_data, dict) and "name" in tool_data:
                # Extract arguments properly — keep as dict (NOT json.dumps),
                # downstream translator_tools_node calls tool.ainvoke(args).
                arguments = tool_data.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}

                tool_calls.append(
                    ParsedToolCall(
                        id=f"call_{i}",
                        name=tool_data["name"],
                        args=arguments,
                    )
                )

        except json.JSONDecodeError:
            # Try to extract at least the function name for basic tool calling
            name_match = re.search(r'"name":\s*"([^"]+)"', match)
            if name_match:
                function_name = name_match.group(1)
                tool_calls.append(
                    ParsedToolCall(
                        id=f"call_{i}",
                        name=function_name,
                        args={},  # Empty args as fallback
                    )
                )
            continue
        except Exception:  # noqa: BLE001
            continue

    return tool_calls
