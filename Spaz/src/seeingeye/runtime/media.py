"""Media loading helpers for SeeingEye runtime inputs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncodedFrame:
    """A single model-ready frame extracted from an image or video."""

    b64: str
    timestamp_s: float | None = None
    mime_type: str = "image/jpeg"


def encode_image(image_path: str | Path) -> str:
    """Read and base64-encode an image file with no data-URL prefix."""
    path = Path(image_path)
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _validate_frame_interval(frame_interval_s: float) -> float:
    interval = float(frame_interval_s)
    if not 0.1 <= interval <= 1.0:
        raise ValueError("frame_interval_s must be between 0.1 and 1.0 seconds")
    return interval


def _validate_frame_selection(selection: str) -> str:
    clean = selection.strip().lower()
    if clean not in {"uniform", "change"}:
        raise ValueError("frame selection must be 'uniform' or 'change'")
    return clean


def _jpeg_frame(frame: Any, timestamp_s: float) -> EncodedFrame:
    import cv2  # type: ignore[import-not-found]

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError(f"could not encode video frame at {timestamp_s:.3f}s")
    return EncodedFrame(
        b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        timestamp_s=round(timestamp_s, 3),
        mime_type="image/jpeg",
    )


def _frame_signature(frame: Any) -> Any:
    import cv2  # type: ignore[import-not-found]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)


def _change_score(previous_signature: Any | None, current_signature: Any) -> float:
    if previous_signature is None:
        return float("inf")
    import cv2  # type: ignore[import-not-found]

    return float(cv2.absdiff(previous_signature, current_signature).mean())


def _keep_change_frame(
    candidates: list[tuple[float, float, EncodedFrame]],
    candidate: tuple[float, float, EncodedFrame],
    max_frames: int | None,
) -> None:
    candidates.append(candidate)
    if max_frames is None or len(candidates) <= max_frames:
        return

    # Keep the first frame as an anchor, then drop the least-changing
    # non-anchor frame. This preserves temporal context while reducing repeats.
    removable = range(1, len(candidates))
    drop_idx = min(removable, key=lambda idx: candidates[idx][1])
    del candidates[drop_idx]


def extract_video_frames(
    video_path: str | Path,
    *,
    frame_interval_s: float,
    max_frames: int | None = None,
    frame_selection: str = "uniform",
    scene_change_threshold: float = 6.0,
) -> list[EncodedFrame]:
    """Extract JPEG frames from a video at a fixed wall-clock interval.

    ``frame_interval_s`` is intentionally clamped by validation to the user
    requested 0.1-1.0 second control range. Frames are encoded as JPEG because
    it is broadly accepted by OpenAI-compatible vision endpoints and keeps
    payload size lower than PNG.
    """
    interval = _validate_frame_interval(frame_interval_s)
    selection = _validate_frame_selection(frame_selection)
    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        frames: list[EncodedFrame] = []
        candidates: list[tuple[float, float, EncodedFrame]] = []
        previous_signature: Any | None = None
        timestamp_s = 0.0
        while True:
            if selection == "uniform" and max_frames is not None and len(frames) >= max_frames:
                break
            if duration_s > 0 and timestamp_s > duration_s + 1e-9:
                break

            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break

            if selection == "uniform":
                frames.append(_jpeg_frame(frame, timestamp_s))
            else:
                signature = _frame_signature(frame)
                score = _change_score(previous_signature, signature)
                previous_signature = signature
                if score == float("inf") or score >= scene_change_threshold:
                    _keep_change_frame(
                        candidates,
                        (timestamp_s, score, _jpeg_frame(frame, timestamp_s)),
                        max_frames,
                    )
            timestamp_s += interval

        if selection == "change":
            frames = [item[2] for item in sorted(candidates, key=lambda item: item[0])]

        if not frames:
            raise ValueError(f"no frames could be extracted from video: {path}")
        return frames
    finally:
        cap.release()
