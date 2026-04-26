"""Builder + state-schema tests for the Reasoner subgraph (Plan 04-02).

Covers compiled graph smoke, expected node set, and ReasonerSubgraphState
isolation (no image_b64, private reasoner_messages, no translator_messages).

References:
  - 04-02-PLAN.md <behavior> for test_builder.py (7 tests)
  - 04-CONTEXT.md D-09 (Reasoner is text-only — NO image_b64)
  - 04-RESEARCH.md §Pattern 2 (Reasoner subgraph topology)
"""

from __future__ import annotations

from src.seeingeye.agents.reasoner.builder import build_reasoner_graph
from src.seeingeye.agents.reasoner.state import ReasonerSubgraphState


# ---------------------------------------------------------------------------
# Compiled graph smoke
# ---------------------------------------------------------------------------


def test_build_reasoner_graph_returns_compiled() -> None:
    g = build_reasoner_graph()
    assert hasattr(g, "ainvoke")
    assert hasattr(g, "invoke")


def test_compiled_graph_has_expected_nodes() -> None:
    g = build_reasoner_graph()
    node_names = set(g.get_graph().nodes)
    expected = {
        "reasoner_model",
        "finalize_answer",
        "finalize_feedback",
        "check_termination",
    }
    assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"


# ---------------------------------------------------------------------------
# State schema isolation (D-09 + per-agent message isolation)
# ---------------------------------------------------------------------------


def test_reasoner_state_excludes_image_b64() -> None:
    # D-09: Reasoner is text-only; image_b64 must NOT be in subgraph schema.
    assert "image_b64" not in ReasonerSubgraphState.__annotations__


def test_reasoner_state_has_private_messages_key() -> None:
    annotations = ReasonerSubgraphState.__annotations__
    assert "reasoner_messages" in annotations
    assert "reasoner_step" in annotations


def test_reasoner_state_has_output_keys() -> None:
    annotations = ReasonerSubgraphState.__annotations__
    assert "final_answer" in annotations
    assert "reasoner_feedback" in annotations


def test_reasoner_state_does_not_redefine_translator_messages() -> None:
    # Per ROADMAP H/I rule: no cross-agent leakage between subgraph schemas.
    assert "translator_messages" not in ReasonerSubgraphState.__annotations__


def test_reasoner_state_carries_shared_keys() -> None:
    # The subgraph still needs sir + question + options from the parent.
    annotations = ReasonerSubgraphState.__annotations__
    assert "sir" in annotations
    assert "question" in annotations
    assert "options" in annotations
