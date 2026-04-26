"""Tests for the parent-graph router + routing constants (Pitfalls 11/13).

Asserts that the module-level constants are correct AND that the pure
router function makes the right routing decision for every branch of
paper Algorithm 1.
"""

from __future__ import annotations

from langgraph.graph import END

from src.seeingeye.graph.routing import (
    FORCE_ANSWER,
    LOOP_BACK_TRANSLATOR,
    TERMINATE,
    route_after_reasoner,
)
from src.seeingeye.state.sir import SIR


# ---------------------------------------------------------------------------
# Constants — locked by Pitfall 11 / 13 / D-09
# ---------------------------------------------------------------------------


def test_terminate_is_end_sentinel() -> None:
    """Pitfall 11: TERMINATE MUST be the langgraph.graph.END sentinel."""
    assert TERMINATE is END


def test_loop_back_constant() -> None:
    """LOOP_BACK_TRANSLATOR matches the node name registered in the builder."""
    assert LOOP_BACK_TRANSLATOR == "translator_subgraph"


def test_force_answer_constant() -> None:
    """FORCE_ANSWER matches the node name registered in the builder."""
    assert FORCE_ANSWER == "force_answer"


# ---------------------------------------------------------------------------
# route_after_reasoner — pure function, 3-way decision
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
    """Minimal state dict for router tests — only the keys the router reads."""
    state = {
        "final_answer": None,
        "reasoner_feedback": None,
        "outer_iter": 1,
        "sir": SIR(content=""),
    }
    state.update(overrides)
    return state


def test_route_after_reasoner_final_answer() -> None:
    """Terminate_and_answer path: final_answer set -> TERMINATE (END sentinel)."""
    state = _base_state(final_answer="A", outer_iter=1)
    assert route_after_reasoner(state) is TERMINATE


def test_route_after_reasoner_feedback_under_max() -> None:
    """Feedback path with budget remaining -> LOOP_BACK_TRANSLATOR."""
    state = _base_state(final_answer=None, reasoner_feedback="need more detail", outer_iter=1)
    assert route_after_reasoner(state) == LOOP_BACK_TRANSLATOR


def test_route_after_reasoner_max_iters_hit() -> None:
    """Budget exhausted without terminate_and_answer -> FORCE_ANSWER."""
    # Default Settings().max_iters = 3; outer_iter == 3 hits the boundary.
    state = _base_state(final_answer=None, reasoner_feedback="still unclear", outer_iter=3)
    assert route_after_reasoner(state) == FORCE_ANSWER


def test_route_after_reasoner_prefers_final_answer() -> None:
    """Anomaly guard: if BOTH final_answer and reasoner_feedback set, prefer TERMINATE."""
    state = _base_state(
        final_answer="A", reasoner_feedback="inconsistent ignore", outer_iter=1
    )
    assert route_after_reasoner(state) is TERMINATE
