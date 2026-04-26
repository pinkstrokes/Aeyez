"""Tests for the SeeingEyeResult dataclass (GRF-03 / D-11).

Asserts the minimal 4-field surface area exactly — no debug fields, no
per-agent token breakdown. See REQUIREMENTS.md GRF-03 simplification note
(cut 2026-04-13) and 05-CONTEXT.md §D-11.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from src.seeingeye.runtime.result import SeeingEyeResult
from src.seeingeye.state.sir import SIR


def test_seeing_eye_result_is_dataclass() -> None:
    """D-11 locks dataclass (NOT Pydantic) — keep the minimal surface."""
    assert is_dataclass(SeeingEyeResult)


def test_seeing_eye_result_has_exactly_four_fields_in_order() -> None:
    """GRF-03 locks fields: answer, sir, outer_iters_used, total_tokens."""
    names = [f.name for f in fields(SeeingEyeResult)]
    assert names == ["answer", "sir", "outer_iters_used", "total_tokens"], (
        f"SeeingEyeResult fields drifted from GRF-03 spec: {names}"
    )


def test_seeing_eye_result_constructible_with_keyword_args() -> None:
    result = SeeingEyeResult(
        answer="A",
        sir=SIR(content="test sir"),
        outer_iters_used=2,
        total_tokens=1234,
    )
    assert result.answer == "A"
    assert isinstance(result.sir, SIR)
    assert result.sir.content == "test sir"
    assert result.outer_iters_used == 2
    assert result.total_tokens == 1234
