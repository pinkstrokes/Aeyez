"""Sanity tests for the VCoT parser ported from
``src/multi-agent/app/agent/toolcall.py:19-213`` to
``src/seeingeye/agents/translator/policy.py``.

Coverage (per 04-01 PLAN <behavior>):
  1. Standard XML happy-path extraction (Priority 1).
  2. Empty content returns ``[]`` cleanly (no exception).
  3. Plain text without any tool-call marker returns ``[]``.
  4. Truncated JSON inside ``<tool_call>`` recovers via the auto-close branch
     in ``process_tool_call_matches`` (one closing brace appended).
  5. Loose-JSON fallback (Priority 3) — no XML tags, just a JSON-shaped object
     with ``"name"`` and ``"arguments"`` keys.
  6. Priority ordering — when both Priority 1 (standard XML) and Priority 3
     (loose JSON) match, Priority 1 wins.
"""

from __future__ import annotations

from src.seeingeye.agents.translator.policy import (
    ParsedToolCall,
    parse_tool_calls_multiple_formats,
    valid_translator_tool_calls,
)


def test_standard_xml_extracts_ocr_call() -> None:
    content = (
        '<tool_call>{"name": "ocr", "arguments": {"image_path": "/tmp/x.png"}}'
        "</tool_call>"
    )

    result = parse_tool_calls_multiple_formats(content)

    assert len(result) == 1
    assert isinstance(result[0], ParsedToolCall)
    assert result[0].name == "ocr"
    assert result[0].args == {"image_path": "/tmp/x.png"}
    assert result[0].id == "call_0"


def test_empty_content_returns_empty_list() -> None:
    assert parse_tool_calls_multiple_formats("") == []
    assert parse_tool_calls_multiple_formats(None) == []  # type: ignore[arg-type]


def test_no_tool_call_in_content_returns_empty() -> None:
    result = parse_tool_calls_multiple_formats(
        "Just some plain text with no tool call markers"
    )
    assert result == []


def test_truncated_json_recovery() -> None:
    # Missing BOTH closing braces AND the </tool_call> tag — exactly the
    # case the old toolcall.py:166-171 truncated-JSON recovery branch
    # exists for. The standard-XML parser's incomplete-tag fallback
    # (line 68 in old code) captures everything from <tool_call> to EOL;
    # process_tool_call_matches then auto-closes the unbalanced braces
    # (count of '{' minus count of '}' = 2 here -> append "}}").
    content = (
        '<tool_call>{"name": "read_table", "arguments": '
        '{"image_path": "/tmp/y.png"'
    )

    result = parse_tool_calls_multiple_formats(content)

    assert len(result) == 1
    assert result[0].name == "read_table"
    assert result[0].args == {"image_path": "/tmp/y.png"}


def test_loose_json_fallback() -> None:
    # No <tool_call> tags — Priority 1 / Priority 2 parsers return []; the
    # loose-JSON regex (Priority 3) matches the embedded object.
    content = (
        'Some preamble {"name": "smart_grid_caption", '
        '"arguments": {"image_path": "/tmp/z.png"}} more text'
    )

    result = parse_tool_calls_multiple_formats(content)

    assert len(result) == 1
    assert result[0].name == "smart_grid_caption"
    assert result[0].args == {"image_path": "/tmp/z.png"}


def test_priority_ordering_standard_xml_wins() -> None:
    # Content carries BOTH a standard <tool_call>...</tool_call> wrapping ocr
    # AND a loose JSON for read_table later in the string. Priority 1 must
    # win — the parser must return ocr only.
    content = (
        '<tool_call>{"name": "ocr", "arguments": {"image_path": "/a.png"}}'
        '</tool_call>\n'
        'Then later: {"name": "read_table", "arguments": {"image_path": "/b.png"}}'
    )

    result = parse_tool_calls_multiple_formats(content)

    assert len(result) == 1
    assert result[0].name == "ocr"
    assert result[0].args == {"image_path": "/a.png"}


def test_valid_translator_tool_calls_filters_placeholders() -> None:
    calls = [
        ParsedToolCall(id="call_0", name="...", args={}),
        ParsedToolCall(id="call_1", name="ocr", args={"image_path": "/a.png"}),
    ]

    result = valid_translator_tool_calls(calls)

    assert result == [calls[1]]
