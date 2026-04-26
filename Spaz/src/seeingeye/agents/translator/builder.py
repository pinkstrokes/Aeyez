"""Translator inner-loop subgraph builder.

Returns a compiled :class:`CompiledStateGraph` that Phase 5 wires as a
parent-graph node.

Graph topology:

    translator_model --[route_after_model]--> tools | check_termination
    tools --> refine_sir --> check_termination
    check_termination --[should_continue]--> translator_model | END

All edge keys are module-level constants (Pitfall #5 — no inline string
literals in the conditional-edge mappings below).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.seeingeye.agents.translator.nodes import (
    ROUTE_CONTINUE,
    ROUTE_DONE,
    ROUTE_END,
    ROUTE_TOOLS,
    check_termination_node,
    refine_sir_node,
    route_after_model,
    should_continue,
    translator_model_node,
)
from src.seeingeye.agents.translator.state import TranslatorSubgraphState
from src.seeingeye.agents.translator.tools import translator_tools_node


def build_translator_graph():
    """Compile the Translator inner-loop subgraph."""
    g = StateGraph(TranslatorSubgraphState)
    g.add_node("translator_model", translator_model_node)
    g.add_node("tools", translator_tools_node)
    g.add_node("refine_sir", refine_sir_node)
    g.add_node("check_termination", check_termination_node)

    g.set_entry_point("translator_model")
    g.add_conditional_edges(
        "translator_model",
        route_after_model,
        {ROUTE_TOOLS: "tools", ROUTE_DONE: "check_termination"},
    )
    g.add_edge("tools", "refine_sir")
    g.add_edge("refine_sir", "check_termination")
    g.add_conditional_edges(
        "check_termination",
        should_continue,
        {ROUTE_CONTINUE: "translator_model", ROUTE_END: END},
    )
    return g.compile()
