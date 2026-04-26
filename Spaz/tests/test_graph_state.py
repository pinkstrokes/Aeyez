"""Sanity tests for the SeeingEyeState parent-graph schema (TPS-01 simplified).

Verifies the TypedDict shape, exact field set, reducer annotations on the
per-agent message lists, the presence of the Phase 2 SIR model, and the
regression guard against a top-level shared ``messages`` key.
"""

from __future__ import annotations

import typing

import pytest

from langgraph.graph.message import add_messages

from src.seeingeye.state.sir import SIR


EXPECTED_FIELDS = {
    "sir",
    "outer_iter",
    "question",
    "options",
    "media_type",
    "image_b64",
    "image_frames",
    "translator_messages",
    "reasoner_messages",
    "final_answer",
    # Added in Phase 5 (GRF-01): carries the Reasoner subgraph's feedback string
    # from the Reasoner → Translator loop-back edge, where the parent-graph
    # merge_feedback_node applies SIR.merge_feedback(feedback_str). See
    # .planning/phases/05-graph-wiring-runtime/05-CONTEXT.md §D-07.
    "reasoner_feedback",
}


def test_seeingeyestate_importable_from_module():
    """Importing SeeingEyeState directly from graph_state must succeed."""
    from src.seeingeye.state.graph_state import SeeingEyeState  # noqa: F401

    assert SeeingEyeState is not None


def test_seeingeyestate_is_typeddict():
    """SeeingEyeState must be a TypedDict — typing.get_type_hints returns a
    non-empty dict for TypedDicts."""
    from src.seeingeye.state.graph_state import SeeingEyeState

    hints = typing.get_type_hints(SeeingEyeState)
    assert isinstance(hints, dict)
    assert len(hints) > 0


def test_seeingeyestate_has_exact_field_set():
    """SeeingEyeState must expose exactly the 8 TPS-01-specified keys — no
    more, no less."""
    from src.seeingeye.state.graph_state import SeeingEyeState

    hints = typing.get_type_hints(SeeingEyeState)
    assert set(hints.keys()) == EXPECTED_FIELDS


def test_sir_field_is_phase2_sir_model():
    """The ``sir`` field must resolve to the exact SIR class from Phase 2,
    not a wrapper or stringified form."""
    from src.seeingeye.state.graph_state import SeeingEyeState

    hints = typing.get_type_hints(SeeingEyeState)
    assert hints["sir"] is SIR


def test_no_top_level_messages_key():
    """Regression guard: the parent schema must NOT declare a shared
    ``messages`` field (Pitfall #4 — message isolation)."""
    from src.seeingeye.state.graph_state import SeeingEyeState

    hints = typing.get_type_hints(SeeingEyeState)
    assert "messages" not in hints


@pytest.mark.parametrize("field", ["translator_messages", "reasoner_messages"])
def test_message_fields_use_add_messages_reducer(field):
    """Both per-agent message fields must be Annotated[list, add_messages]."""
    from src.seeingeye.state.graph_state import SeeingEyeState

    hints = typing.get_type_hints(SeeingEyeState, include_extras=True)
    args = typing.get_args(hints[field])
    assert args, f"{field} has no Annotated metadata"
    assert args[0] is list, f"{field} base type must be list, got {args[0]!r}"
    assert add_messages in args[1:], (
        f"{field} must carry add_messages in its Annotated metadata; "
        f"got metadata {args[1:]!r}"
    )


def test_seeingeyestate_reexported_from_package():
    """``from src.seeingeye.state import SeeingEyeState`` must succeed (public
    API re-export alongside SIR)."""
    from src.seeingeye.state import SeeingEyeState  # noqa: F401

    assert SeeingEyeState is not None
