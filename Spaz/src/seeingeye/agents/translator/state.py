"""Translator subgraph state schema.

The subgraph-private keys ``translator_messages`` and ``translator_step``
are NOT in the parent :class:`SeeingEyeState`. LangGraph isolates keys
that exist only in a subgraph schema per-invocation, so each entry into
this subgraph starts with a fresh empty message list and zero step
counter — wipe-per-iter semantics for free.

This mirrors the old ``_reset_agent_memory_for_iteration()`` in
``src/multi-agent/app/flow/iterative_refinement.py:56-71`` which called
``agent.memory.clear()`` at the start of every outer iteration.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.seeingeye.state.sir import SIR


class TranslatorSubgraphState(TypedDict):
    """Inner-loop state schema for the Translator subgraph.

    Shared with parent ``SeeingEyeState``:
      - ``sir``: current Structured Intermediate Representation
      - ``outer_iter``: parent outer-loop iteration; first Translator pass is 1
      - ``question``: original user question
      - ``options``: multiple-choice options or None
      - ``media_type``: "image" or "video"
      - ``image_b64``: base64-encoded image bytes for legacy single-image calls
      - ``image_frames``: ordered image/video frame payloads for the VLM

    Private to this subgraph (NOT in parent schema):
      - ``translator_messages``: subgraph-local message list — fresh on
        each invocation. Mirrors wipe-per-iter behavior of
        ``_reset_agent_memory_for_iteration()`` in
        ``src/multi-agent/app/flow/iterative_refinement.py:56-71``.
      - ``translator_step``: explicit counter for the paper's N_T = 3
        bound, incremented in ``translator_model_node``, checked by
        ``should_continue``.
    """

    # Shared with parent SeeingEyeState
    sir: SIR
    outer_iter: int
    question: str
    options: list[str] | None
    media_type: str
    image_b64: str | None
    image_frames: list[dict]
    # Subgraph-private — fresh on each invocation (wipe-per-iter).
    # Mirrors old _reset_agent_memory_for_iteration() in
    # src/multi-agent/app/flow/iterative_refinement.py:56-71.
    translator_messages: Annotated[list, add_messages]
    translator_step: int
