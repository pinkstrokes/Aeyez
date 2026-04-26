from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SelectedFrame:
    timestamp_s: float
    reason: str
    data_url: str


@dataclass(frozen=True)
class EventSegment:
    start_s: float
    end_s: float
    peak_s: float
    peak_score: float
    danger_candidate: bool
    frame_timestamps_s: list[float]


@dataclass(frozen=True)
class DailySelection:
    duration_s: float
    fps: float
    baseline_interval_s: float
    baseline_target_count: int
    selected_frames: list[SelectedFrame]
    events: list[EventSegment]


def _jpeg_data_url(frame: Any, quality: int = 82) -> str:
    import cv2  # type: ignore[import-not-found]

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("could not encode video frame")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _signature(frame: Any) -> Any:
    import cv2  # type: ignore[import-not-found]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)


def _change_score(prev: Any | None, curr: Any) -> float:
    if prev is None:
        return 0.0
    import cv2  # type: ignore[import-not-found]

    return float(cv2.absdiff(prev, curr).mean())


def _capture_at(cap: Any, timestamp_s: float) -> Any | None:
    import cv2  # type: ignore[import-not-found]

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _merge_event_points(
    points: list[tuple[float, float]],
    *,
    merge_gap_s: float,
    event_padding_s: float,
    danger_threshold: float,
    duration_s: float,
) -> list[EventSegment]:
    if not points:
        return []

    segments: list[EventSegment] = []
    current: list[tuple[float, float]] = [points[0]]
    for point in points[1:]:
        if point[0] - current[-1][0] <= merge_gap_s:
            current.append(point)
        else:
            segments.append(
                _event_from_points(
                    current,
                    event_padding_s=event_padding_s,
                    danger_threshold=danger_threshold,
                    duration_s=duration_s,
                )
            )
            current = [point]
    segments.append(
        _event_from_points(
            current,
            event_padding_s=event_padding_s,
            danger_threshold=danger_threshold,
            duration_s=duration_s,
        )
    )
    return segments


def _event_from_points(
    points: list[tuple[float, float]],
    *,
    event_padding_s: float,
    danger_threshold: float,
    duration_s: float,
) -> EventSegment:
    peak_s, peak_score = max(points, key=lambda item: item[1])
    start_s = max(0.0, points[0][0] - event_padding_s)
    end_s = min(duration_s, points[-1][0] + event_padding_s)
    danger_candidate = peak_score >= danger_threshold
    count = 6 if danger_candidate else 3
    if count == 1 or end_s <= start_s:
        frame_ts = [peak_s]
    else:
        frame_ts = [
            round(start_s + (end_s - start_s) * i / (count - 1), 3)
            for i in range(count)
        ]
    return EventSegment(
        start_s=round(start_s, 3),
        end_s=round(end_s, 3),
        peak_s=round(peak_s, 3),
        peak_score=round(peak_score, 3),
        danger_candidate=danger_candidate,
        frame_timestamps_s=frame_ts,
    )


def _format_ts(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_seconds(seconds: float) -> str:
    return _format_ts(seconds)


def select_daily_video_frames(
    video_path: str | Path,
    *,
    baseline_frames_per_minute: int = 5,
    scan_interval_s: float = 1.0,
    change_threshold: float = 8.0,
    danger_threshold: float = 18.0,
    event_merge_gap_s: float = 8.0,
    event_padding_s: float = 3.0,
    context_interval_s: float | None = None,
    max_context_frames: int | None = None,
) -> DailySelection:
    """Select evidence frames for all-day summary.

    The retention rule is represented by ``baseline_target_count``: callers can
    persist that full plan later. The model-facing selected frames are bounded:
    all event/danger evidence is kept, plus sparse context frames across the day.
    """
    import cv2  # type: ignore[import-not-found]

    path = Path(video_path).expanduser()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
        if duration_s <= 0:
            raise ValueError("video duration could not be determined")

        baseline_interval_s = 60.0 / baseline_frames_per_minute
        baseline_target_count = int(duration_s // baseline_interval_s) + 1

        event_points: list[tuple[float, float]] = []
        prev_sig = None
        ts = 0.0
        while ts <= duration_s:
            frame = _capture_at(cap, ts)
            if frame is None:
                break
            sig = _signature(frame)
            score = _change_score(prev_sig, sig)
            prev_sig = sig
            if score >= change_threshold:
                event_points.append((ts, score))
            ts += scan_interval_s

        events = _merge_event_points(
            event_points,
            merge_gap_s=event_merge_gap_s,
            event_padding_s=event_padding_s,
            danger_threshold=danger_threshold,
            duration_s=duration_s,
        )

        selected_by_ts: dict[float, str] = {}
        context_interval_s = context_interval_s or baseline_interval_s
        context_ts = 0.0
        while context_ts <= duration_s:
            if max_context_frames is not None and len(selected_by_ts) >= max_context_frames:
                break
            selected_by_ts[round(context_ts, 3)] = "baseline-context"
            context_ts += context_interval_s

        for event in events:
            reason = "danger-candidate" if event.danger_candidate else "event"
            for frame_ts in event.frame_timestamps_s:
                selected_by_ts[round(frame_ts, 3)] = reason

        selected_frames: list[SelectedFrame] = []
        for frame_ts, reason in sorted(selected_by_ts.items()):
            frame = _capture_at(cap, frame_ts)
            if frame is None:
                continue
            selected_frames.append(
                SelectedFrame(
                    timestamp_s=frame_ts,
                    reason=reason,
                    data_url=_jpeg_data_url(frame),
                )
            )

        return DailySelection(
            duration_s=round(duration_s, 3),
            fps=round(fps, 3),
            baseline_interval_s=round(baseline_interval_s, 3),
            baseline_target_count=baseline_target_count,
            selected_frames=selected_frames,
            events=events,
        )
    finally:
        cap.release()


def frames_to_model_payload(frames: list[SelectedFrame]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": _format_ts(frame.timestamp_s),
            "timestamp_s": frame.timestamp_s,
            "reason": frame.reason,
        }
        for frame in frames
    ]


def frame_to_runtime_payload(frame: SelectedFrame) -> dict[str, Any]:
    header, _, payload = frame.data_url.partition(",")
    mime_type = "image/jpeg"
    if header.startswith("data:") and ";" in header:
        mime_type = header[5:].split(";", 1)[0] or mime_type
    return {
        "b64": payload,
        "timestamp_s": frame.timestamp_s,
        "mime_type": mime_type,
    }


def choose_segment_model_frames(
    selection: DailySelection,
    *,
    start_s: float,
    end_s: float,
    max_frames: int = 24,
) -> list[SelectedFrame]:
    in_segment = [
        frame for frame in selection.selected_frames
        if start_s <= frame.timestamp_s < end_s
    ]
    if len(in_segment) <= max_frames:
        return in_segment

    priority = {"danger-candidate": 0, "event": 1, "baseline-context": 2}
    chosen: list[SelectedFrame] = []
    seen: set[float] = set()
    for frame in sorted(in_segment, key=lambda item: (priority.get(item.reason, 9), item.timestamp_s)):
        if frame.reason == "baseline-context" and len(chosen) >= max_frames:
            continue
        if frame.timestamp_s not in seen:
            chosen.append(frame)
            seen.add(frame.timestamp_s)
        if len(chosen) >= max_frames and any(item.reason != "baseline-context" for item in in_segment):
            break

    if len(chosen) < max_frames:
        baselines = [frame for frame in in_segment if frame.timestamp_s not in seen]
        needed = max_frames - len(chosen)
        if baselines:
            if len(baselines) <= needed:
                chosen.extend(baselines)
            else:
                for i in range(needed):
                    idx = round(i * (len(baselines) - 1) / max(1, needed - 1))
                    chosen.append(baselines[idx])

    return sorted(chosen[:max_frames], key=lambda item: item.timestamp_s)
