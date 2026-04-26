"""Tests for the public async ``run_question()`` API (GRF-02 / D-10).

Structural signature + shape tests. NO live LLM — patches ``_GRAPH`` in
the runner module with a fake graph that records its ``ainvoke`` arguments.

Also verifies the mermaid diagram artifact was produced by the Task 2
step `python -m src.seeingeye.graph.builder` (GRF-06 / D-13).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
from pathlib import Path

import pytest

from src.seeingeye.runtime import SeeingEyeResult, run_question
from src.seeingeye.runtime import runner as runner_mod
from src.seeingeye.state.sir import SIR
from src.seeingeye.runtime.media import EncodedFrame


# ---------------------------------------------------------------------------
# Fake graph used across tests — records calls, returns canned final state
# ---------------------------------------------------------------------------


class _FakeAIMsg:
    """Minimal stand-in for langchain_core AIMessage with usage_metadata."""

    def __init__(self, tokens: int | None) -> None:
        if tokens is None:
            self.usage_metadata = None
        else:
            self.usage_metadata = {"total_tokens": tokens}


class _FakeMsgWithoutUsage:
    """Stand-in for HumanMessage / ToolMessage (no usage_metadata attribute)."""


class _FakeGraph:
    def __init__(self, final_state: dict | None = None) -> None:
        self.calls: list[tuple[dict, dict]] = []
        self._final_state = final_state or {
            "sir": SIR(content="final-sir"),
            "outer_iter": 1,
            "final_answer": "A",
            "translator_messages": [],
            "reasoner_messages": [],
        }

    async def ainvoke(self, state: dict, config: dict) -> dict:
        # Record a deep-ish snapshot (dict copy; SIR stays referenced — fine for tests).
        self.calls.append((dict(state), dict(config)))
        # Simulate graph run: carry forward the inputs, overlay canned final state.
        merged = dict(state)
        merged.update(self._final_state)
        return merged


@pytest.fixture
def fake_graph(monkeypatch):
    """Install a fresh _FakeGraph as the module-level cached graph."""
    graph = _FakeGraph()
    monkeypatch.setattr(runner_mod, "_GRAPH", graph, raising=False)
    return graph


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """Write a tiny 1x1 PNG for run_question to base64-encode."""
    # Smallest valid PNG (1x1 transparent) — 67 bytes.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "test.png"
    p.write_bytes(png_bytes)
    return p


# ---------------------------------------------------------------------------
# Signature / shape
# ---------------------------------------------------------------------------


def test_run_question_signature_async() -> None:
    """D-10 locks: ``async def run_question(question, image_path, options=None, **overrides)``."""
    assert inspect.iscoroutinefunction(run_question)
    sig = inspect.signature(run_question)
    params = list(sig.parameters.values())
    # Positional: question, image_path; keyword: options=None; variadic: **overrides
    names = [p.name for p in params]
    assert names[:2] == ["question", "image_path"]
    assert "options" in names
    assert sig.parameters["options"].default is None
    # **overrides present
    kinds = {p.name: p.kind for p in params}
    assert kinds["overrides"] is inspect.Parameter.VAR_KEYWORD


def test_run_question_returns_seeing_eye_result(fake_graph, tmp_image) -> None:
    fake_graph._final_state = {
        "sir": SIR(content="done"),
        "outer_iter": 2,
        "final_answer": "B",
        "translator_messages": [],
        "reasoner_messages": [],
    }
    result = asyncio.run(run_question("Q?", tmp_image))
    assert isinstance(result, SeeingEyeResult)
    assert result.answer == "B"
    assert result.sir.content == "done"
    assert result.outer_iters_used == 2
    assert result.total_tokens == 0  # no messages with usage_metadata


# ---------------------------------------------------------------------------
# Invocation contract
# ---------------------------------------------------------------------------


def test_run_question_converts_image_to_base64(fake_graph, tmp_image) -> None:
    asyncio.run(run_question("Q?", tmp_image))
    state, _config = fake_graph.calls[0]
    b64 = state["image_b64"]
    assert isinstance(b64, str) and len(b64) > 0
    # Decodes without raising and produces the original bytes.
    decoded = base64.b64decode(b64)
    assert decoded == tmp_image.read_bytes()


def test_run_question_passes_recursion_limit_50(fake_graph, tmp_image) -> None:
    """Pitfall 12: recursion_limit=50 explicit on every .ainvoke()."""
    asyncio.run(run_question("Q?", tmp_image))
    _state, config = fake_graph.calls[0]
    assert config == {"recursion_limit": 50}


def test_run_question_initial_state(fake_graph, tmp_image) -> None:
    asyncio.run(run_question("What is in this image?", tmp_image))
    state, _config = fake_graph.calls[0]
    assert state["question"] == "What is in this image?"
    assert state["outer_iter"] == 0  # bumped to 1 by increment_iter_node on entry
    assert isinstance(state["sir"], SIR)
    assert state["sir"].content == ""
    assert state["final_answer"] is None
    assert state["reasoner_feedback"] is None
    assert state["options"] is None
    assert state["media_type"] == "image"
    assert len(state["image_frames"]) == 1
    assert state["translator_messages"] == []
    assert state["reasoner_messages"] == []


def test_run_question_initial_state_with_options(fake_graph, tmp_image) -> None:
    opts = ["A. foo", "B. bar"]
    asyncio.run(run_question("Q?", tmp_image, options=opts))
    state, _config = fake_graph.calls[0]
    assert state["options"] == opts


def test_run_question_rejects_missing_media(fake_graph) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(run_question("Q?"))


def test_run_question_rejects_two_media_inputs(fake_graph, tmp_image) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(run_question("Q?", image_path=tmp_image, video_path=tmp_image))


def test_run_question_extracts_video_frames(fake_graph, tmp_path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    def _fake_extract(
        video_path,
        *,
        frame_interval_s,
        max_frames,
        frame_selection,
        scene_change_threshold,
    ):
        assert Path(video_path) == video
        assert frame_interval_s == 0.25
        assert max_frames == 64
        assert frame_selection == "change"
        assert scene_change_threshold == 6.0
        return [
            EncodedFrame(b64="aaa", timestamp_s=0.0),
            EncodedFrame(b64="bbb", timestamp_s=0.25),
        ]

    monkeypatch.setattr(runner_mod, "extract_video_frames", _fake_extract)

    asyncio.run(
        run_question(
            "What happens?",
            image_path=None,
            video_path=video,
            frame_interval_s=0.25,
        )
    )

    state, _config = fake_graph.calls[0]
    assert state["media_type"] == "video"
    assert state["image_b64"] is None
    assert [frame["timestamp_s"] for frame in state["image_frames"]] == [0.0, 0.25]


def test_run_question_ignores_overrides_with_warning(fake_graph, tmp_image, caplog) -> None:
    """D-10: overrides accepted but ignored (clamped to paper defaults)."""
    asyncio.run(run_question("Q?", tmp_image, max_iters=999, weird="foo"))
    # No exception; graph still invoked with stock recursion_limit=50.
    _state, config = fake_graph.calls[0]
    assert config == {"recursion_limit": 50}


# ---------------------------------------------------------------------------
# total_tokens aggregation
# ---------------------------------------------------------------------------


def test_total_tokens_aggregation(fake_graph, tmp_image) -> None:
    fake_graph._final_state = {
        "sir": SIR(content=""),
        "outer_iter": 1,
        "final_answer": "A",
        "translator_messages": [_FakeAIMsg(100)],
        "reasoner_messages": [_FakeAIMsg(100)],
    }
    result = asyncio.run(run_question("Q?", tmp_image))
    assert result.total_tokens == 200


def test_total_tokens_handles_missing_usage_metadata(fake_graph, tmp_image) -> None:
    fake_graph._final_state = {
        "sir": SIR(content=""),
        "outer_iter": 1,
        "final_answer": "A",
        "translator_messages": [
            _FakeAIMsg(50),
            _FakeMsgWithoutUsage(),
            _FakeAIMsg(None),
        ],
        "reasoner_messages": [_FakeAIMsg(70)],
    }
    result = asyncio.run(run_question("Q?", tmp_image))
    assert result.total_tokens == 120


def test_outer_iters_used_from_final_state(fake_graph, tmp_image) -> None:
    fake_graph._final_state = {
        "sir": SIR(content=""),
        "outer_iter": 2,
        "final_answer": "A",
        "translator_messages": [],
        "reasoner_messages": [],
    }
    result = asyncio.run(run_question("Q?", tmp_image))
    assert result.outer_iters_used == 2


# ---------------------------------------------------------------------------
# Mermaid diagram artifact (GRF-06 / D-13)
# ---------------------------------------------------------------------------


def test_mermaid_diagram_emitted() -> None:
    """GRF-06: Markdown-embedded mermaid at the canonical path exists."""
    out = Path(".planning/phases/05-graph-wiring-runtime/graph.md")
    assert out.exists(), (
        f"{out} missing — run `python -m src.seeingeye.graph.builder` first."
    )
    text = out.read_text(encoding="utf-8")
    assert "```mermaid" in text
