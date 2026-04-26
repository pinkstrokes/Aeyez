"""Tests for the SIR (Structured Intermediate Representation) model (FND-02)."""

import ast
import inspect
import pathlib

import pytest

from src.seeingeye.state.sir import SIR


class TestSIRConstruction:
    """Tests 1-2: SIR construction."""

    def test_empty_sir(self):
        """Test 1: SIR() creates empty SIR with content=''."""
        sir = SIR()
        assert sir.content == ""

    def test_sir_with_content(self):
        """Test 2: SIR(content='hello') creates SIR with that content."""
        sir = SIR(content="hello")
        assert sir.content == "hello"


class TestMergeFeedback:
    """Tests 3-6: merge_feedback() semantics."""

    def test_merge_feedback_appends(self):
        """Test 3: merge_feedback returns NEW SIR with feedback appended."""
        sir = SIR(content="original")
        result = sir.merge_feedback("some feedback")
        assert result.content == "original\n\n--- REASONING FEEDBACK ---\nsome feedback"

    def test_merge_feedback_strips_prefix(self):
        """Test 4: merge_feedback strips 'FEEDBACK from reasoning agent:' prefix."""
        sir = SIR(content="original")
        result = sir.merge_feedback("FEEDBACK from reasoning agent: actual text")
        assert result.content.endswith("actual text")

    def test_merge_feedback_empty_feedback_noop(self):
        """Test 5: merge_feedback('') on non-empty SIR returns same SIR."""
        sir = SIR(content="existing")
        result = sir.merge_feedback("")
        assert result is sir

    def test_merge_feedback_empty_content_guard(self):
        """Test 6: merge_feedback on empty SIR returns same SIR (guard)."""
        sir = SIR(content="")
        result = sir.merge_feedback("text")
        assert result is sir


class TestUpdate:
    """Tests 7-8: update() semantics."""

    def test_update_appends(self):
        """Test 7: update on non-empty SIR appends with separator."""
        sir = SIR(content="original")
        result = sir.update("new info")
        assert result.content == "original\n\n--- UPDATED SIR ---\nnew info"

    def test_update_first_time(self):
        """Test 8: update on empty SIR sets content directly."""
        sir = SIR(content="")
        result = sir.update("new info")
        assert result.content == "new info"


class TestReplace:
    """Test 9: replace() semantics."""

    def test_replace(self):
        """Test 9: replace returns SIR with completely new content."""
        sir = SIR(content="old")
        result = sir.replace("completely new")
        assert result.content == "completely new"


class TestJSONRoundtrip:
    """Test 10: JSON serialization."""

    def test_json_roundtrip(self):
        """Test 10: SIR roundtrips through JSON losslessly."""
        sir = SIR(content="test content\nwith newlines\n\n--- REASONING FEEDBACK ---\nfeedback")
        json_str = sir.model_dump_json()
        restored = SIR.model_validate_json(json_str)
        assert restored == sir


class TestImmutability:
    """Test 11: Immutability check."""

    def test_original_unchanged_after_operations(self):
        """Test 11: Original SIR unchanged after merge_feedback/update/replace."""
        sir = SIR(content="original")
        _ = sir.merge_feedback("feedback")
        _ = sir.update("update")
        _ = sir.replace("replaced")
        assert sir.content == "original"


class TestZeroSeeingeyeImports:
    """Test 12: sir.py has no seeingeye imports."""

    def test_no_seeingeye_imports(self):
        """Test 12: sir.py imports ONLY from pydantic (no seeingeye imports)."""
        sir_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "seeingeye" / "state" / "sir.py"
        source = sir_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("seeingeye"), (
                        f"Found import of seeingeye module: {alias.name}"
                    )
                    assert not alias.name.startswith("src.seeingeye"), (
                        f"Found import of src.seeingeye module: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("seeingeye"), (
                        f"Found import from seeingeye module: {node.module}"
                    )
                    assert not node.module.startswith("src.seeingeye"), (
                        f"Found import from src.seeingeye module: {node.module}"
                    )
