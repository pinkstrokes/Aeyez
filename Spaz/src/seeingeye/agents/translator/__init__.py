"""Translator inner-loop subgraph (Phase 4 AGT-01).

Public surface: :func:`build_translator_graph` returns a compiled
LangGraph subgraph that Phase 5 wires as a parent-graph node.
"""

from src.seeingeye.agents.translator.builder import build_translator_graph

__all__ = ["build_translator_graph"]
