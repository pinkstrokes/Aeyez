"""Reasoner subgraph state schema (Plan 04-02 / AGT-03).

Per D-09: NO ``image_b64`` field — Reasoner is text-only (Qwen3-8B on vLLM
port 8001).

Subgraph-private keys (``reasoner_messages``, ``reasoner_step``) are NOT in
the parent ``SeeingEyeState``. A fresh empty list per subgraph invocation
gives wipe-per-iter semantics for free (Finding #2 — see citation below).

Mirrors old ``_reset_agent_memory_for_iteration()`` in
``src/multi-agent/app/flow/iterative_refinement.py:56-71``.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.seeingeye.state.sir import SIR


class ReasonerSubgraphState(TypedDict, total=False):
    """Reasoner inner-loop subgraph state.

    Field semantics:
      - ``sir``: shared with parent ``SeeingEyeState``.
      - ``question``: shared with parent.
      - ``options``: shared with parent (multiple-choice or ``None``).
      - ``reasoner_messages``: subgraph-PRIVATE message history. Wipes per
        outer iteration because it is not declared on the parent schema.
      - ``reasoner_step``: explicit step counter (Finding #4 — explicit
        counter beats ``recursion_limit`` for the paper's ``N_R = 3``).
      - ``final_answer``: subgraph output, set by ``finalize_answer_node``.
      - ``reasoner_feedback``: subgraph output, set by ``finalize_feedback_node``.
        Phase 5's outer-loop bookkeeping consumes this string and calls
        ``state["sir"].merge_feedback(state["reasoner_feedback"])`` (Finding #3).
    """

    # Shared with parent SeeingEyeState (NO image_b64 — D-09)
    sir: SIR
    outer_iter: int
    question: str
    options: list[str] | None
    media_type: str
    # Subgraph-private — fresh on each invocation (wipe-per-iter)
    reasoner_messages: Annotated[list, add_messages]
    reasoner_step: int
    # Subgraph outputs (consumed by Phase 5 parent graph)
    final_answer: str | None
    reasoner_feedback: str | None
