"""Sanity tests for the paper-frozen prompt package (TPS-05).

Covers:
- Module + symbol presence for the four prompt modules.
- Presence of the ``# DO NOT EDIT -- paper-frozen`` header on each prompt file.

The byte-equality drift-detection tests that cross-checked the new prompt
modules against the old legacy tree were removed in Phase 7 (2026-04-17)
when that tree was deleted. The remaining tests are self-referential
against ``src/seeingeye/prompts/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Target prompt files                                                         #
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT_DIR = _REPO_ROOT / "src" / "seeingeye" / "prompts"
PROMPT_FILES = [
    PROMPT_DIR / "translator.py",
    PROMPT_DIR / "reasoner.py",
    PROMPT_DIR / "vqa_mmmu.py",
    PROMPT_DIR / "force_answer.py",
]


# --------------------------------------------------------------------------- #
# 1. Module import + symbol presence                                          #
# --------------------------------------------------------------------------- #


def test_package_imports():
    """All four prompt modules import from the seeingeye.prompts package."""
    from src.seeingeye.prompts import translator, reasoner, vqa_mmmu, force_answer

    # Each module is importable and has a module-level attribute namespace.
    for mod in (translator, reasoner, vqa_mmmu, force_answer):
        assert mod is not None
        assert hasattr(mod, "__name__")


def test_translator_symbols_are_nonempty_strings():
    from src.seeingeye.prompts import translator

    for name in ("SYSTEM_PROMPT", "FIRST_STEP_PROMPT", "NEXT_STEP_PROMPT", "FINAL_STEP_PROMPT"):
        value = getattr(translator, name)
        assert isinstance(value, str) and value.strip(), f"translator.{name} must be a non-empty str"


def test_reasoner_symbols_are_nonempty_strings():
    from src.seeingeye.prompts import reasoner

    for name in ("SYSTEM_PROMPT", "NEXT_STEP_PROMPT", "FINAL_STEP_PROMPT", "FINAL_ITERATION_PROMPT"):
        value = getattr(reasoner, name)
        assert isinstance(value, str) and value.strip(), f"reasoner.{name} must be a non-empty str"


def test_vqa_mmmu_symbols_are_nonempty_strings():
    from src.seeingeye.prompts import vqa_mmmu

    for name in (
        "SYSTEM_PROMPT_MULTIPLE_CHOICE",
        "SYSTEM_PROMPT_SHORT_ANSWER",
        "NEXT_STEP_PROMPT_MULTIPLE_CHOICE",
        "NEXT_STEP_PROMPT_SHORT_ANSWER",
    ):
        value = getattr(vqa_mmmu, name)
        assert isinstance(value, str) and value.strip(), f"vqa_mmmu.{name} must be a non-empty str"


def test_force_answer_symbol_is_nonempty_string():
    from src.seeingeye.prompts import force_answer

    value = force_answer.FINAL_ITERATION_PROMPT
    assert isinstance(value, str) and value.strip()


# --------------------------------------------------------------------------- #
# 2. DO NOT EDIT header presence                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", PROMPT_FILES, ids=[p.name for p in PROMPT_FILES])
def test_do_not_edit_header_present(path: Path):
    """Each prompt file carries ``# DO NOT EDIT`` and ``paper-frozen`` in the first 3 lines."""
    first_lines = path.read_text(encoding="utf-8").splitlines()[:3]
    joined = "\n".join(first_lines)
    assert "# DO NOT EDIT" in joined, f"{path.name} missing '# DO NOT EDIT' header: {joined!r}"
    assert "paper-frozen" in joined, f"{path.name} missing 'paper-frozen' marker: {joined!r}"


# --------------------------------------------------------------------------- #
# 3. No templating migration (Pitfall #2 guard)                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", PROMPT_FILES, ids=[p.name for p in PROMPT_FILES])
def test_no_chatprompttemplate_migration(path: Path):
    text = path.read_text(encoding="utf-8")
    assert "ChatPromptTemplate" not in text
    assert "PromptTemplate" not in text
    assert "from langchain" not in text
