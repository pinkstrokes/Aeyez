"""Parent-graph wrapper nodes.

Two small nodes that belong to the outer-loop wiring, NOT to either
agent's subgraph:

- :func:`increment_iter_node` — pre-translator wrapper that bumps
  ``outer_iter`` by 1. First pass enters with ``outer_iter=0``, becomes 1
  (paper Algorithm 1).
- :func:`merge_feedback_node` — applies
  :meth:`SIR.merge_feedback` on the loop-back path. Kept as a dedicated
  node (not inside the router) because routers MUST NOT mutate state
  (05-CONTEXT.md §specifics — Pitfall 5).
"""

from __future__ import annotations


async def increment_iter_node(state) -> dict:
    """Bump ``outer_iter`` by 1 before each Translator subgraph invocation."""
    return {"outer_iter": state.get("outer_iter", 0) + 1}


async def merge_feedback_node(state) -> dict:
    """Merge Reasoner feedback into the SIR before re-entering the Translator.

    Reads ``state["reasoner_feedback"]`` and ``state["sir"]``; returns a
    new SIR plus a cleared feedback field. :meth:`SIR.merge_feedback`
    handles the empty-feedback guard internally (returns ``self`` unchanged).

    Mirrors old ``iterative_refinement.py:307`` (the
    ``self._append_feedback_to_sir`` call site on the outer-loop side)
    per 04-RESEARCH.md Finding #3.
    """
    feedback = state.get("reasoner_feedback") or ""
    new_sir = state["sir"].merge_feedback(feedback)
    return {"sir": new_sir, "reasoner_feedback": None}
