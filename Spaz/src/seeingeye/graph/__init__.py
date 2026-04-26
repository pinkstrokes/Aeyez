"""Parent StateGraph assembly for the SeeingEye LangGraph runtime (GRF-01).

Re-exports the compiled-graph builder. Phase 5 Plan 05-01.
"""

from src.seeingeye.graph.builder import build_parent_graph

__all__ = ["build_parent_graph"]
