"""Reasoner inner-loop subgraph builder (Plan 04-02 / AGT-03).

Public entry point consumed by Phase 5's parent ``StateGraph``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.seeingeye.agents.reasoner.nodes import (
    ROUTE_ANSWER,
    ROUTE_CONTINUE,
    ROUTE_END,
    ROUTE_FEEDBACK,
    ROUTE_LOOP,
    check_termination_node,
    decision_router,
    finalize_answer_node,
    finalize_feedback_node,
    reasoner_model_node,
    should_continue,
)
from src.seeingeye.agents.reasoner.state import ReasonerSubgraphState


def build_reasoner_graph():
    """Compile and return the Reasoner inner-loop subgraph.

    Returns:
        CompiledStateGraph: invocable via ``.ainvoke({"sir": ..., "question": ...,
        "options": ..., "reasoner_step": 0})``.
    """
    g = StateGraph(ReasonerSubgraphState)

    g.add_node("reasoner_model", reasoner_model_node)
    g.add_node("finalize_answer", finalize_answer_node)
    g.add_node("finalize_feedback", finalize_feedback_node)
    g.add_node("check_termination", check_termination_node)

    g.set_entry_point("reasoner_model")

    g.add_conditional_edges(
        "reasoner_model",
        decision_router,
        {
            ROUTE_ANSWER: "finalize_answer",
            ROUTE_FEEDBACK: "finalize_feedback",
            ROUTE_CONTINUE: "check_termination",
        },
    )

    g.add_edge("finalize_answer", END)
    g.add_edge("finalize_feedback", END)

    g.add_conditional_edges(
        "check_termination",
        should_continue,
        {
            ROUTE_LOOP: "reasoner_model",
            ROUTE_END: END,
        },
    )

    return g.compile()
