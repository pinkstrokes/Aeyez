"""Sanity tests for the Phase 3 tools layer (TPS-02 / TPS-03 / TPS-04).

Per the 2026-04-13 simplification pass:
  - Tools are langchain-core ``@tool`` functions — NOT a custom ``BaseTool``
    shim.
  - Semantic equivalence, NOT byte-identical fixture diffs — Phase 6 parity
    gate is the canonical correctness check.
  - ``REASONER_DECISIONS`` must be bindable to a ``ChatOpenAI`` client via
    ``bind_tools(...)`` without error (schema sanity check only — no live
    vLLM call is made).
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Imports / package shape
# ---------------------------------------------------------------------------


def test_public_symbols_importable_from_seeingeye_tools():
    """All tools + 2 aggregate lists must be re-exported from the package."""
    from src.seeingeye.tools import (  # noqa: F401
        ocr,
        read_table,
        action_motion_scan,
        mechanics_hazard_scan,
        route_surface_scan,
        smart_grid_caption,
        spatial_intelligence_scan,
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
        VISUAL_TOOLS,
        REASONER_DECISIONS,
    )


def test_all_visual_tools_are_basetool_instances():
    """langchain-core ``@tool`` decorator must wrap each function into a
    ``BaseTool`` instance so the Phase 4 Reasoner can ``bind_tools`` on them
    and the Phase 4 Translator policy can call ``.ainvoke(...)``."""
    from src.seeingeye.tools import (
        ocr,
        read_table,
        action_motion_scan,
        mechanics_hazard_scan,
        route_surface_scan,
        smart_grid_caption,
        spatial_intelligence_scan,
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
    )

    for obj in (
        ocr,
        read_table,
        action_motion_scan,
        mechanics_hazard_scan,
        route_surface_scan,
        smart_grid_caption,
        spatial_intelligence_scan,
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
    ):
        assert isinstance(obj, BaseTool), f"{obj!r} is not a BaseTool"


def test_aggregate_lists_have_expected_membership_and_order():
    """``VISUAL_TOOLS`` and ``REASONER_DECISIONS`` encode the surface the
    Phase 4 agents consume; both content and order matter."""
    from src.seeingeye.tools import (
        ocr,
        read_table,
        action_motion_scan,
        mechanics_hazard_scan,
        route_surface_scan,
        smart_grid_caption,
        spatial_intelligence_scan,
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
        VISUAL_TOOLS,
        REASONER_DECISIONS,
    )

    assert VISUAL_TOOLS == [
        ocr,
        read_table,
        smart_grid_caption,
        action_motion_scan,
        route_surface_scan,
        mechanics_hazard_scan,
        spatial_intelligence_scan,
    ]
    assert REASONER_DECISIONS == [
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
    ]


def test_tool_names_match_old_class_attributes():
    """Old tools advertised these exact ``name`` class attributes via the
    OpenManus ``BaseTool``; the ported ``@tool`` functions must keep the
    identical public names so the Reasoner sees no wire-level behavior
    drift when ``bind_tools(...)`` emits the JSON schema."""
    from src.seeingeye.tools import (
        ocr,
        read_table,
        action_motion_scan,
        mechanics_hazard_scan,
        route_surface_scan,
        smart_grid_caption,
        spatial_intelligence_scan,
        terminate_and_answer,
        terminate_and_ask_translator,
        continue_reasoning,
    )

    assert ocr.name == "ocr"
    assert read_table.name == "read_table"
    assert smart_grid_caption.name == "smart_grid_caption"
    assert action_motion_scan.name == "action_motion_scan"
    assert route_surface_scan.name == "route_surface_scan"
    assert mechanics_hazard_scan.name == "mechanics_hazard_scan"
    assert spatial_intelligence_scan.name == "spatial_intelligence_scan"
    assert terminate_and_answer.name == "terminate_and_answer"
    assert terminate_and_ask_translator.name == "terminate_and_ask_translator"
    assert continue_reasoning.name == "continue_reasoning"


# ---------------------------------------------------------------------------
# Visual tool error-path behavior (no live Azure / vLLM / Tesseract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_missing_file_returns_error_string():
    """OCR must gracefully return a human-readable string on a missing
    image (paper behavior — the old code returned
    ``ToolResult(error=...)``; the port collapses that into the string
    so ``ToolMessage.content`` carries the same semantic signal)."""
    from src.seeingeye.tools import ocr

    result = await ocr.ainvoke({"image_path": "/definitely/not/real.png"})
    assert isinstance(result, str)
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_read_table_missing_file_returns_error_string():
    """ReadTable mirrors OCR's missing-file contract."""
    from src.seeingeye.tools import read_table

    result = await read_table.ainvoke({"image_path": "/definitely/not/real.png"})
    assert isinstance(result, str)
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_action_motion_scan_missing_file_returns_error_string():
    from src.seeingeye.tools import action_motion_scan

    result = await action_motion_scan.ainvoke(
        {"image_path": "/definitely/not/real.png"}
    )
    assert isinstance(result, str)
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_route_surface_scan_missing_file_returns_error_string():
    from src.seeingeye.tools import route_surface_scan

    result = await route_surface_scan.ainvoke(
        {"image_path": "/definitely/not/real.png"}
    )
    assert isinstance(result, str)
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_mechanics_hazard_scan_missing_file_returns_error_string():
    from src.seeingeye.tools import mechanics_hazard_scan

    result = await mechanics_hazard_scan.ainvoke(
        {"image_path": "/definitely/not/real.png"}
    )
    assert isinstance(result, str)
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_spatial_intelligence_scan_missing_file_returns_error_string():
    from src.seeingeye.tools import spatial_intelligence_scan

    result = await spatial_intelligence_scan.ainvoke(
        {"image_path": "/definitely/not/real.png"}
    )
    assert isinstance(result, str)
    assert "does not exist" in result


def test_smart_grid_caption_is_async_tool():
    """SmartGridCaption orchestrates nested VLM calls, so the Phase 4
    translator policy must ``await`` it.  ``BaseTool.coroutine`` is
    populated by ``@tool`` when the wrapped function is ``async def``."""
    from src.seeingeye.tools import smart_grid_caption

    assert smart_grid_caption.coroutine is not None


# ---------------------------------------------------------------------------
# Decision-action return contracts (model-facing strings)
# ---------------------------------------------------------------------------


def test_terminate_and_answer_return_template_verbatim():
    """The return template is what the Reasoner's ``ToolMessage.content``
    carries back into the reasoning loop. It must match the old
    ``src/multi-agent/app/tool/terminate_and_answer.py`` format verbatim
    — the model has been trained against these exact strings."""
    from src.seeingeye.tools import terminate_and_answer

    result = terminate_and_answer.invoke(
        {"answer": "B", "confidence": "high", "reasoning": "per SIR row 3"}
    )
    assert isinstance(result, str)
    assert "FINAL ANSWER: B" in result
    assert "Confidence: high" in result
    assert "per SIR row 3" in result


def test_terminate_and_ask_translator_return_template_verbatim():
    """``feedback: {...}`` is the parsed-sentinel prefix the Phase 4
    outer loop switches on. Do not drop the prefix."""
    from src.seeingeye.tools import terminate_and_ask_translator

    result = terminate_and_ask_translator.invoke({"feedback": "need patch 5"})
    assert isinstance(result, str)
    assert "feedback: need patch 5" in result


def test_continue_reasoning_returns_nonempty_string():
    """``continue_reasoning`` is a Phase-3 addition (not in old code) so
    the Reasoner with ``tool_choice='auto'`` has a no-op terminal branch.
    The return value is narrative for tool-message logging — just
    non-empty."""
    from src.seeingeye.tools import continue_reasoning

    result = continue_reasoning.invoke({"thought": "I need to compute"})
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# bind_tools schema compatibility
# ---------------------------------------------------------------------------


def test_reasoner_decisions_bind_to_reasoner_client():
    """``ChatOpenAI.bind_tools(...)`` validates the JSON schema shape at
    bind-time without contacting the server.  If any decision's
    ``args_schema`` is malformed, this raises.  No live vLLM is needed."""
    from src.seeingeye.llm.vllm_openai import create_reasoner_client
    from src.seeingeye.tools import REASONER_DECISIONS

    client = create_reasoner_client()
    bound = client.bind_tools(REASONER_DECISIONS)
    assert bound is not None
