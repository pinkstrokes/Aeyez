"""Phase 3 tools layer — visual tools and Reasoner decision actions.

Per TPS-02 (simplified 2026-04-13), all tools are langchain-core ``@tool``
functions.  No custom ``BaseTool`` shim.  No decorator registry.  No
byte-identical envelope verification.

Public surface:
  - ``ocr``, ``read_table``, ``smart_grid_caption``, ``action_motion_scan``,
    ``route_surface_scan``, ``mechanics_hazard_scan``,
    ``spatial_intelligence_scan`` — visual tools the Translator dispatches to
    in Phase 4.
  - ``terminate_and_answer``, ``terminate_and_ask_translator``,
    ``continue_reasoning`` — the three Reasoner decision actions bound
    via ``ChatOpenAI.bind_tools([...])`` in Phase 4.
  - ``VISUAL_TOOLS`` / ``REASONER_DECISIONS`` — convenience aggregates
    in a fixed order for deterministic ``bind_tools`` schemas.
"""

from src.seeingeye.tools.decisions import (
    continue_reasoning,
    terminate_and_answer,
    terminate_and_ask_translator,
)
from src.seeingeye.tools.action_motion_scan import action_motion_scan
from src.seeingeye.tools.mechanics_hazard_scan import mechanics_hazard_scan
from src.seeingeye.tools.ocr import ocr
from src.seeingeye.tools.read_table import read_table
from src.seeingeye.tools.route_surface_scan import route_surface_scan
from src.seeingeye.tools.smart_grid_caption import smart_grid_caption
from src.seeingeye.tools.spatial_intelligence_scan import spatial_intelligence_scan

VISUAL_TOOLS = [
    ocr,
    read_table,
    smart_grid_caption,
    action_motion_scan,
    route_surface_scan,
    mechanics_hazard_scan,
    spatial_intelligence_scan,
]
REASONER_DECISIONS = [
    terminate_and_answer,
    terminate_and_ask_translator,
    continue_reasoning,
]

__all__ = [
    "ocr",
    "read_table",
    "smart_grid_caption",
    "action_motion_scan",
    "route_surface_scan",
    "mechanics_hazard_scan",
    "spatial_intelligence_scan",
    "terminate_and_answer",
    "terminate_and_ask_translator",
    "continue_reasoning",
    "VISUAL_TOOLS",
    "REASONER_DECISIONS",
]
