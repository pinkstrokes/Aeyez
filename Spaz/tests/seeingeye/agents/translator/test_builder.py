"""Compiled-subgraph smoke tests for build_translator_graph().

Coverage (per 04-01 PLAN <behavior>):
  1. build_translator_graph() returns an object with .ainvoke + .invoke
     (CompiledStateGraph duck-test).
  2. The compiled graph contains the four expected nodes:
     translator_model, tools, refine_sir, check_termination.
  3. TranslatorSubgraphState declares translator_messages and
     translator_step privately (PRIVATE to subgraph schema, NOT in parent
     SeeingEyeState).
  4. TranslatorSubgraphState does NOT redefine reasoner_messages — proves
     no cross-agent leakage at the schema level.
"""

from __future__ import annotations

from src.seeingeye.agents.translator.builder import build_translator_graph
from src.seeingeye.agents.translator.state import TranslatorSubgraphState


def test_build_translator_graph_returns_compiled() -> None:
    g = build_translator_graph()

    # CompiledStateGraph duck-test: must be callable both sync and async.
    assert hasattr(g, "ainvoke") and callable(g.ainvoke)
    assert hasattr(g, "invoke") and callable(g.invoke)


def test_subgraph_state_keys_include_private() -> None:
    g = build_translator_graph()

    nodes = set(g.get_graph().nodes)
    expected = {"translator_model", "tools", "refine_sir", "check_termination"}
    assert expected.issubset(nodes), (
        f"Compiled subgraph missing expected node names. "
        f"Expected {expected}, got {nodes}."
    )


def test_translator_state_has_private_messages_key() -> None:
    annotations = TranslatorSubgraphState.__annotations__
    assert "translator_messages" in annotations, (
        "TranslatorSubgraphState must declare translator_messages PRIVATELY "
        "(subgraph-private wipe-per-iter semantics depend on this)."
    )
    assert "translator_step" in annotations, (
        "TranslatorSubgraphState must declare translator_step PRIVATELY "
        "(explicit counter for paper N_T = 3 enforcement)."
    )


def test_translator_state_does_not_redefine_reasoner_messages() -> None:
    # Cross-agent leakage guard: the Translator subgraph schema must NOT
    # know about reasoner_messages. Phase 4 keeps the two agents
    # mechanically isolated at the schema level.
    annotations = TranslatorSubgraphState.__annotations__
    assert "reasoner_messages" not in annotations
