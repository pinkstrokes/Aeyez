"""Parent StateGraph builder for the SeeingEye LangGraph runtime (GRF-01).

Topology (paper Algorithm 1):

    START
      v
    increment_iter_node
      v
    translator_subgraph
      v
    reasoner_subgraph
      v
    route_after_reasoner (conditional)
      |-- TERMINATE (END sentinel) ---------------------------------> END
      |-- LOOP_BACK_TRANSLATOR -> merge_feedback_node -> increment_iter_node
      +-- FORCE_ANSWER -> force_answer -> END

Pitfall mitigations:

- **Pitfall 11** — ``TERMINATE`` is the ``END`` sentinel, not the string ``"END"``.
- **Pitfall 13** — all edge destinations are module-level constants from
  :mod:`src.seeingeye.graph.routing`.
- **Pitfall 5** — routers do not mutate state;
  :func:`merge_feedback_node` handles the SIR merge on the loop-back path.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.seeingeye.agents.reasoner.builder import build_reasoner_graph
from src.seeingeye.agents.reasoner.force_answer import force_answer_node
from src.seeingeye.agents.translator.builder import build_translator_graph
from src.seeingeye.graph.nodes import increment_iter_node, merge_feedback_node
from src.seeingeye.graph.routing import (
    FORCE_ANSWER,
    LOOP_BACK_TRANSLATOR,
    TERMINATE,
    route_after_reasoner,
)
from src.seeingeye.state.graph_state import SeeingEyeState


def build_parent_graph():
    """Compile the SeeingEye parent StateGraph (GRF-01).

    Returns:
        CompiledStateGraph: invoke via
        ``await graph.ainvoke(state, config={"recursion_limit": 50})``
        (Pitfall 12).
    """
    g = StateGraph(SeeingEyeState)

    # Nodes
    g.add_node("increment_iter_node", increment_iter_node)
    g.add_node("translator_subgraph", build_translator_graph())
    g.add_node("reasoner_subgraph", build_reasoner_graph())
    g.add_node("merge_feedback_node", merge_feedback_node)
    g.add_node("force_answer", force_answer_node)

    # Linear path through the outer loop
    g.add_edge(START, "increment_iter_node")
    g.add_edge("increment_iter_node", "translator_subgraph")
    g.add_edge("translator_subgraph", "reasoner_subgraph")

    # 3-way conditional (D-09). TERMINATE is an alias for END; the mapping
    # entry ``END: END`` is LangGraph-idiomatic for "route directly to END".
    # LOOP_BACK_TRANSLATOR routes to merge_feedback_node first (NOT directly
    # to the translator) to keep the router pure (Pitfall 5). The constant
    # name communicates intent ("we are looping back to the translator
    # phase") even though the immediate destination is merge_feedback_node.
    g.add_conditional_edges(
        "reasoner_subgraph",
        route_after_reasoner,
        {
            TERMINATE: END,
            LOOP_BACK_TRANSLATOR: "merge_feedback_node",
            FORCE_ANSWER: "force_answer",
        },
    )

    # Loop-back path: merge feedback, then re-increment before Translator.
    g.add_edge("merge_feedback_node", "increment_iter_node")

    # Force-answer terminal.
    g.add_edge("force_answer", END)

    return g.compile()


if __name__ == "__main__":
    # GRF-06 / D-13: emit a mermaid diagram for reviewer trust.
    # Invoke via: python -m src.seeingeye.graph.builder
    from pathlib import Path

    graph = build_parent_graph()
    mermaid_src = graph.get_graph().draw_mermaid()
    out = Path(".planning/phases/05-graph-wiring-runtime/graph.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# SeeingEye Parent Graph (GRF-06)\n\n"
        "Generated via `graph.get_graph().draw_mermaid()`. Commit on every\n"
        "structural change to the parent graph.\n\n"
        "```mermaid\n"
        f"{mermaid_src}"
        "```\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")
