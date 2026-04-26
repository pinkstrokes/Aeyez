"""Parent-graph state schema for the SeeingEye LangGraph runtime.

TPS-01 (simplified 2026-04-13): plain TypedDict. Per-agent message
isolation via separate ``translator_messages`` / ``reasoner_messages``
keys, both reduced by LangGraph's default ``add_messages``. NO custom
token reducers, NO shared ``messages`` key on the parent schema.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.seeingeye.state.sir import SIR


class SeeingEyeState(TypedDict):
    """Parent StateGraph schema consumed by the Phase 5 builder.

    Field semantics:
      - sir: current Structured Intermediate Representation (Phase 2 model).
      - outer_iter: number of completed translator->reasoner outer-loop passes.
      - question: the original user question (immutable across the run).
      - options: multiple-choice options or None for short-answer.
      - media_type: "image" or "video".
      - image_b64: base64-encoded image bytes for legacy single-image calls.
      - image_frames: ordered frame payloads for image/video VLM calls.
      - translator_messages: Translator subgraph message history.
      - reasoner_messages: Reasoner subgraph message history.
      - reasoner_feedback: subgraph output set by finalize_feedback_node;
        consumed by merge_feedback_node on loop-back; None on
        terminate_and_answer or force_answer paths. (Phase 5 addition —
        without this key declared on the parent schema, LangGraph drops
        the value on subgraph exit. See 04-RESEARCH.md Finding #3.)
      - final_answer: terminal answer string, populated on terminate_and_answer
        or by the force-answer node at MAX_ITERS.
    """

    sir: SIR
    outer_iter: int
    question: str
    options: list[str] | None
    media_type: str
    image_b64: str | None
    image_frames: list[dict]
    translator_messages: Annotated[list, add_messages]
    reasoner_messages: Annotated[list, add_messages]
    reasoner_feedback: str | None
    final_answer: str | None
