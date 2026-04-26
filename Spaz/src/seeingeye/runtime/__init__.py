"""Public runtime API for the SeeingEye LangGraph pipeline.

Re-exports:
  - :class:`SeeingEyeResult` (GRF-03 / D-11) — minimal 4-field result dataclass.
  - :func:`run_question` (GRF-02 / D-10) — async single-question public API
    (added in Task 2 of Plan 05-01).
"""

from src.seeingeye.runtime.result import SeeingEyeResult
from src.seeingeye.runtime.runner import run_question

__all__ = ["SeeingEyeResult", "run_question"]
