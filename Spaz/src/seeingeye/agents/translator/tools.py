"""Translator tools dispatcher node — Python-side dispatch of visual tools.

Mirrors ``src/multi-agent/app/agent/toolcall.py:393-495`` (the old
``act()`` + ``execute_tool()`` pattern): per-call try/except, error string
on unknown tool, plain string content on the resulting ``ToolMessage``.

The Translator VLM (Qwen2.5-VL-3B on vLLM) cannot reliably emit
OpenAI-shaped tool calls because of Pitfall #1 (QwenLM/Qwen3-VL#1093).
Instead, the model emits free-form text containing one of three
VCoT-shaped call formats; :func:`parse_tool_calls_multiple_formats`
extracts them and this node dispatches each one through the local
``@tool``-decorated visual tools.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from src.seeingeye.agents.translator.policy import (
    parse_tool_calls_multiple_formats,
    valid_translator_tool_calls,
)
from src.seeingeye.agents.translator.state import TranslatorSubgraphState
from src.seeingeye.tools.action_motion_scan import action_motion_scan
from src.seeingeye.tools.mechanics_hazard_scan import mechanics_hazard_scan
from src.seeingeye.tools.ocr import ocr
from src.seeingeye.tools.read_table import read_table
from src.seeingeye.tools.route_surface_scan import route_surface_scan
from src.seeingeye.tools.smart_grid_caption import smart_grid_caption
from src.seeingeye.tools.spatial_intelligence_scan import spatial_intelligence_scan

_TOOL_MAP = {
    "ocr": ocr,
    "read_table": read_table,
    "smart_grid_caption": smart_grid_caption,
    "action_motion_scan": action_motion_scan,
    "route_surface_scan": route_surface_scan,
    "mechanics_hazard_scan": mechanics_hazard_scan,
    "spatial_intelligence_scan": spatial_intelligence_scan,
}


async def translator_tools_node(state: TranslatorSubgraphState) -> dict:
    """Parse VCoT text from the latest AIMessage and dispatch each tool.

    Returns a partial state update with the new ``ToolMessage``s appended
    to ``translator_messages`` (LangGraph's ``add_messages`` reducer
    handles the append).
    """
    last_msg = state["translator_messages"][-1]
    parsed = valid_translator_tool_calls(
        parse_tool_calls_multiple_formats(getattr(last_msg, "content", "") or "")
    )
    if not parsed:
        return {}

    tool_messages: list[ToolMessage] = []
    for call in parsed:
        tool_fn = _TOOL_MAP.get(call.name)
        if tool_fn is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Unknown tool '{call.name}'",
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
            continue
        try:
            result = await tool_fn.ainvoke(call.args)
        except Exception as e:  # noqa: BLE001
            result = f"Error executing {call.name}: {e}"
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call.id,
                name=call.name,
            )
        )
    return {"translator_messages": tool_messages}
