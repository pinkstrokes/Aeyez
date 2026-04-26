"""Tests for the parent StateGraph builder + wrapper nodes (GRF-01).

Structural tests only — no live LLM invocation (integration test covers
end-to-end in a separate marker).
"""

from __future__ import annotations

import pytest

from src.seeingeye.graph.builder import build_parent_graph
from src.seeingeye.graph.nodes import increment_iter_node, merge_feedback_node
from src.seeingeye.state.sir import SIR


# ---------------------------------------------------------------------------
# Compiled parent graph structure
# ---------------------------------------------------------------------------


def test_parent_graph_compiles() -> None:
    """build_parent_graph() compiles without raising and exposes ainvoke."""
    graph = build_parent_graph()
    assert callable(getattr(graph, "ainvoke", None))


def test_parent_graph_has_expected_nodes() -> None:
    """The 5 topology-critical nodes are all registered."""
    graph = build_parent_graph()
    node_names = set(graph.get_graph().nodes)
    expected = {
        "increment_iter_node",
        "translator_subgraph",
        "reasoner_subgraph",
        "merge_feedback_node",
        "force_answer",
    }
    assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"


def test_parent_graph_entry_point() -> None:
    """START edges directly into increment_iter_node (first pass bumps outer_iter to 1)."""
    graph = build_parent_graph()
    edges = graph.get_graph().edges
    edge_pairs = {(e.source, e.target) for e in edges}
    # LangGraph uses "__start__" as the START sentinel string in the drawn graph.
    assert ("__start__", "increment_iter_node") in edge_pairs, (
        f"expected START -> increment_iter_node; got {edge_pairs}"
    )


# ---------------------------------------------------------------------------
# Wrapper nodes — pure behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_iter_node_pure() -> None:
    """increment_iter_node bumps outer_iter by 1 (paper Algorithm 1)."""
    assert (await increment_iter_node({"outer_iter": 0})) == {"outer_iter": 1}
    assert (await increment_iter_node({"outer_iter": 2})) == {"outer_iter": 3}


@pytest.mark.asyncio
async def test_increment_iter_node_missing_key_defaults_to_zero() -> None:
    """Defensive: a fresh state with no outer_iter yet still bumps to 1."""
    assert (await increment_iter_node({})) == {"outer_iter": 1}


@pytest.mark.asyncio
async def test_merge_feedback_node_applies_sir_merge() -> None:
    """merge_feedback_node applies SIR.merge_feedback and clears the feedback field."""
    state = {
        "sir": SIR(content="original"),
        "reasoner_feedback": "add this",
    }
    result = await merge_feedback_node(state)
    # SIR content contains both strings + the locked separator (D-03).
    assert "original" in result["sir"].content
    assert "add this" in result["sir"].content
    assert "--- REASONING FEEDBACK ---" in result["sir"].content
    # Feedback MUST be cleared so it does not re-trigger on the next pass.
    assert result["reasoner_feedback"] is None


@pytest.mark.asyncio
async def test_merge_feedback_node_empty_feedback_is_noop() -> None:
    """SIR.merge_feedback returns self when feedback is empty — node still clears key."""
    original_sir = SIR(content="original")
    state = {"sir": original_sir, "reasoner_feedback": None}
    result = await merge_feedback_node(state)
    assert result["sir"] is original_sir  # unchanged
    assert result["reasoner_feedback"] is None
