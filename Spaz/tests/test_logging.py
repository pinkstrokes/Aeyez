"""Tests for seeingeye.observability.logging — loguru JSONL sink + content filter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from loguru import logger

from src.seeingeye.observability.logging import _content_filter, configure_logging


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove tracing env vars before each test so setdefault works fresh."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    # Also reset loguru sinks between tests
    logger.remove()
    yield


# ---------------------------------------------------------------------------
# Test 1-2: Environment variable defaults
# ---------------------------------------------------------------------------

def test_langsmith_tracing_defaults_to_false():
    """After configure_logging(), LANGSMITH_TRACING should be 'false'."""
    configure_logging(jsonl=False)
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_langchain_tracing_v2_defaults_to_false():
    """After configure_logging(), LANGCHAIN_TRACING_V2 should be 'false'."""
    configure_logging(jsonl=False)
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


# ---------------------------------------------------------------------------
# Test 3-5: _content_filter unit tests
# ---------------------------------------------------------------------------

def test_content_filter_rejects_base64():
    """_content_filter returns False for messages containing 'base64,'."""
    record = {"message": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}
    assert _content_filter(record) is False


def test_content_filter_rejects_oversized_message():
    """_content_filter returns False for messages > 4096 characters."""
    record = {"message": "x" * 4097}
    assert _content_filter(record) is False


def test_content_filter_passes_normal_message():
    """_content_filter returns True for a normal short message without base64."""
    record = {"message": "User submitted a question about math."}
    assert _content_filter(record) is True


# ---------------------------------------------------------------------------
# Test 6-7: JSONL sink creation + valid JSON
# ---------------------------------------------------------------------------

def test_jsonl_file_created(tmp_path: Path):
    """After configure_logging(jsonl=True), logging a message creates a JSONL file."""
    configure_logging(log_dir=tmp_path, jsonl=True)
    logger.info("hello from test")
    # loguru flushes on each write
    jsonl_file = tmp_path / "seeingeye.jsonl"
    assert jsonl_file.exists(), f"Expected JSONL file at {jsonl_file}"


def test_jsonl_contains_valid_json(tmp_path: Path):
    """Each line of the JSONL file is valid JSON."""
    configure_logging(log_dir=tmp_path, jsonl=True)
    logger.info("first message")
    logger.info("second message")

    jsonl_file = tmp_path / "seeingeye.jsonl"
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        parsed = json.loads(line)
        assert "text" in parsed  # loguru serialize puts message in "text"


# ---------------------------------------------------------------------------
# Test 8: base64 message does NOT appear in JSONL output
# ---------------------------------------------------------------------------

def test_base64_message_not_in_jsonl(tmp_path: Path):
    """A message containing 'base64,' must NOT appear in the JSONL output."""
    configure_logging(log_dir=tmp_path, jsonl=True)
    logger.info("normal log message")
    logger.info("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA")

    jsonl_file = tmp_path / "seeingeye.jsonl"
    content = jsonl_file.read_text(encoding="utf-8")
    assert "base64," not in content
    # Only the normal message should be present
    lines = content.strip().splitlines()
    assert len(lines) == 1
