"""SeeingEyeResult — the public return type for :func:`run_question` (GRF-03).

Locked by D-11 to exactly 4 fields. No debug conveniences — langfuse
handles trace-level inspection if ever needed per REQUIREMENTS.md GRF-03
simplification (cut 2026-04-13).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.seeingeye.state.sir import SIR


@dataclass
class SeeingEyeResult:
    """Public return type for :func:`run_question` (GRF-03).

    Minimal surface per paper reproducibility scope:

    - ``answer``: final string answer (from ``terminate_and_answer`` or
      ``force_answer``).
    - ``sir``: final SIR snapshot at termination.
    - ``outer_iters_used``: number of completed outer-loop passes
      (1..MAX_ITERS).
    - ``total_tokens``: sum of ``usage_metadata.total_tokens`` across all
      AIMessages in both ``translator_messages`` and ``reasoner_messages``.

    NO ``decision_path``, NO per-agent token breakdown, NO
    ``translator_steps_total`` / ``reasoner_steps_total`` — cut
    2026-04-13 as debugging conveniences (GRF-03 simplification per
    REQUIREMENTS.md; langfuse handles trace-level inspection if ever
    needed).
    """

    answer: str
    sir: SIR
    outer_iters_used: int
    total_tokens: int
