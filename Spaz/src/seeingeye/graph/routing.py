"""Routing constants and pure router function for the parent graph.

Uses module-level constants for conditional edges (Pitfall 13 — no inline
string literals in ``add_conditional_edges`` mappings). ``TERMINATE`` is
the ``langgraph.graph.END`` sentinel, not the string ``"END"`` (Pitfall 11).

Routers are PURE — they read state and return a constant; they MUST NOT
mutate state (Pitfall 5). Feedback merging lives in
:func:`src.seeingeye.graph.nodes.merge_feedback_node`.
"""

from __future__ import annotations

from langgraph.graph import END

from src.seeingeye.config.settings import Settings

# Conditional-edge destinations — MUST match ``add_conditional_edges`` mapping
# keys AND (for ``LOOP_BACK_TRANSLATOR`` and ``FORCE_ANSWER``) the node names
# registered on the parent StateGraph via ``add_node``.
#
# D-09 locked: TERMINATE is END sentinel (Pitfall 11).
TERMINATE = END
LOOP_BACK_TRANSLATOR = "translator_subgraph"  # matches ``add_node(...)``
FORCE_ANSWER = "force_answer"  # matches ``add_node(...)``


def route_after_reasoner(state) -> str:
    """Decide the next edge after the Reasoner subgraph returns.

    Priority order (matches paper Algorithm 1):

      1. If Reasoner produced ``final_answer`` -> ``TERMINATE`` (END sentinel).
      2. If ``outer_iter >= MAX_ITERS`` (exhausted) and no final_answer
         -> ``FORCE_ANSWER``.
      3. Otherwise (Reasoner produced feedback, budget remains)
         -> ``LOOP_BACK_TRANSLATOR``.

    Pure: reads state, returns a constant. Does NOT mutate state — SIR
    feedback merge happens in :func:`merge_feedback_node` between this
    router's ``LOOP_BACK_TRANSLATOR`` decision and the Translator subgraph
    re-entry (Pitfall 5).
    """
    if state.get("final_answer"):
        return TERMINATE
    settings = Settings()
    if state.get("outer_iter", 0) >= settings.max_iters:
        return FORCE_ANSWER
    return LOOP_BACK_TRANSLATOR
