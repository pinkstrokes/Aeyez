"""Shared fixtures for tests/seeingeye/agents/.

Reused by both translator and reasoner test trees (Plan 04-01 creates the
file; Plan 04-02 reuses it for reasoner-side tests).
"""

from __future__ import annotations

import pytest

from src.seeingeye.state.sir import SIR


@pytest.fixture
def empty_sir() -> SIR:
    return SIR(content="")


@pytest.fixture
def populated_sir() -> SIR:
    return SIR(content="Initial visual description: a chart with three bars.")


@pytest.fixture
def sample_question() -> str:
    return "Which bar is tallest?"


@pytest.fixture
def sample_options() -> list[str]:
    return ["A. Left", "B. Middle", "C. Right", "D. None"]
