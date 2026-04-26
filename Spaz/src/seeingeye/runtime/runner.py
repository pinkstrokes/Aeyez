"""Public async ``run_question()`` API for the SeeingEye LangGraph runtime.

GRF-02 / D-10 signature lock (05-CONTEXT.md):

    async def run_question(
        question: str,
        image_path: str | Path,
        options: list[str] | None = None,
        **overrides,
    ) -> SeeingEyeResult

This is the single module-private entry point that Phase 6's
``FlowExecutor`` adapter (PAR-01) will wrap. The compiled LangGraph is a
private implementation detail.

``**overrides`` are accepted but **clamped to paper defaults** — they are
logged via loguru and ignored. Explicit opt-in for overrides is Phase 7's
concern (05-CONTEXT.md §D-10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.seeingeye.graph.builder import build_parent_graph
from src.seeingeye.observability.logging import configure_logging
from src.seeingeye.observability.safety_events import log_safety_event
from src.seeingeye.config.settings import Settings
from src.seeingeye.runtime.media import EncodedFrame, encode_image, extract_video_frames
from src.seeingeye.runtime.result import SeeingEyeResult
from src.seeingeye.state.sir import SIR


# Compile once per process. ``build_parent_graph`` is pure; per-call
# compilation is wasteful. Cached as a module-level attribute so tests can
# install a fake via monkeypatch.
_GRAPH: Any = None


def _get_graph() -> Any:
    """Lazy-compile the parent graph on first call."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_parent_graph()
    return _GRAPH


def _encode_image(image_path: str | Path) -> str:
    """Backward-compatible wrapper for tests and downstream imports."""
    return encode_image(image_path)


def _sum_total_tokens(state: dict) -> int:
    """Sum ``usage_metadata.total_tokens`` across both agent message lists.

    Defensive: messages without ``usage_metadata`` (e.g. ``ToolMessage``,
    ``HumanMessage``, system-only paths) contribute 0. Matches the GRF-03
    "single int, not per-agent breakdown" requirement.
    """
    total = 0
    for key in ("translator_messages", "reasoner_messages"):
        for msg in state.get(key, []) or []:
            meta = getattr(msg, "usage_metadata", None)
            if meta and isinstance(meta, dict):
                total += meta.get("total_tokens", 0) or 0
    return total


async def run_question(
    question: str,
    image_path: str | Path | None = None,
    options: list[str] | None = None,
    video_path: str | Path | None = None,
    frame_interval_s: float | None = None,
    frame_selection: str | None = None,
    scene_change_threshold: float | None = None,
    **overrides: Any,
) -> SeeingEyeResult:
    """Run the SeeingEye pipeline on one image or video question.

    Args:
        question: the user-facing question text.
        image_path: filesystem path to an image file (PNG/JPG bytes).
        options: optional multiple-choice options; ``None`` for short-answer.
        video_path: optional filesystem path to a video file. When present,
            frames are extracted every ``frame_interval_s`` seconds and sent to
            the Translator as ordered images.
        frame_interval_s: video sampling interval, constrained to 0.1-1.0
            seconds. Defaults to ``Settings.video_frame_interval_s``.
        frame_selection: video frame selection strategy, ``uniform`` or
            ``change``. Defaults to ``Settings.video_frame_selection``.
        scene_change_threshold: mean grayscale frame-difference threshold used
            by ``change`` frame selection.
        **overrides: accepted but IGNORED (D-10 clamped to paper defaults).
            A loguru warning is emitted on each call with the ignored keys.

    Returns:
        :class:`SeeingEyeResult` with ``answer``, ``sir``, ``outer_iters_used``,
        ``total_tokens``.

    Raises:
        FileNotFoundError: if the selected media path does not exist.

    LLM / vLLM connection errors propagate unchanged — the caller decides
    whether to retry or surface them.
    """
    configure_logging()

    if overrides:
        # D-10: clamped to paper defaults. Warn but do not fail.
        logger.warning(
            "run_question received overrides that will be ignored "
            "(clamped to paper defaults per D-10): {keys}",
            keys=list(overrides.keys()),
        )

    if bool(image_path) == bool(video_path):
        raise ValueError("provide exactly one of image_path or video_path")

    settings = Settings()
    graph = _get_graph()
    media_type = "image"
    image_b64: str | None = None
    image_frames: list[dict[str, Any]] = []
    if video_path:
        media_type = "video"
        frames = extract_video_frames(
            video_path,
            frame_interval_s=frame_interval_s or settings.video_frame_interval_s,
            max_frames=settings.video_max_frames,
            frame_selection=frame_selection or settings.video_frame_selection,
            scene_change_threshold=(
                scene_change_threshold
                if scene_change_threshold is not None
                else settings.video_scene_change_threshold
            ),
        )
        image_frames = [
            {
                "b64": frame.b64,
                "timestamp_s": frame.timestamp_s,
                "mime_type": frame.mime_type,
            }
            for frame in frames
        ]
    else:
        image_b64 = _encode_image(image_path or "")
        image_frames = [
            EncodedFrame(b64=image_b64, timestamp_s=None, mime_type="image/jpeg").__dict__
        ]

    initial_state: dict = {
        "sir": SIR(content=""),
        "outer_iter": 0,  # increment_iter_node bumps to 1 on first pass
        "question": question,
        "options": options,
        "media_type": media_type,
        "image_b64": image_b64,
        "image_frames": image_frames,
        "translator_messages": [],
        "reasoner_messages": [],
        "reasoner_feedback": None,
        "final_answer": None,
    }

    # Pitfall 12: recursion_limit=50 explicit; primary control is
    # ``outer_iter >= MAX_ITERS`` inside the graph.
    # Pitfall 18: .ainvoke (never .invoke).
    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    answer = final_state.get("final_answer") or ""
    sir = final_state["sir"]
    outer_iters_used = final_state.get("outer_iter", 0)
    total_tokens = _sum_total_tokens(final_state)

    if settings.safety_event_log_enabled and (
        settings.analysis_mode.strip().lower() == "safety"
        or "SAFETY REPORT" in answer
    ):
        log_safety_event(
            log_path=settings.safety_event_log_path,
            question=question,
            answer=answer,
            sir=sir.content,
            outer_iters_used=outer_iters_used,
            total_tokens=total_tokens,
            media_type=media_type,
        )

    return SeeingEyeResult(
        answer=answer,
        sir=sir,
        outer_iters_used=outer_iters_used,
        total_tokens=total_tokens,
    )
