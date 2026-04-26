from __future__ import annotations

import base64
from dataclasses import dataclass, field
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
    timeline_segments: list["TimelineSegment"] = field(default_factory=list)


@dataclass(frozen=True)
class TimelineSegment:
    start_s: float
    end_s: float
    reason: str
    risk_level: str
    peak_score: float
    event_count: int
    danger_count: int
    sample_interval_s: float
    anchor_s: float


@dataclass(frozen=True)
class AnalysisSegment:
    start_s: float
    end_s: float
    reason: str
    risk_level: str
    peak_score: float
    event_count: int
    danger_count: int
    sample_interval_s: float


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


def _classify_risk_level(*, danger_count: int, event_count: int, peak_score: float) -> str:
    if danger_count > 0 or peak_score >= 18.0:
        return "high"
    if event_count > 0 or peak_score >= 10.0:
        return "medium"
    return "low"


def _default_sample_interval_s(risk_level: str) -> float:
    if risk_level == "high":
        return 1.0
    if risk_level == "uncertain":
        return 2.0
    if risk_level == "medium":
        return 2.0
    return 3.0


def _add_uniform_segments(
    items: list[TimelineSegment],
    *,
    start_s: float,
    end_s: float,
    reason: str,
    risk_level: str,
    peak_score: float,
    event_count: int,
    danger_count: int,
    target_window_s: float,
) -> None:
    cursor = start_s
    while cursor < end_s:
        seg_end = min(end_s, cursor + target_window_s)
        items.append(
            TimelineSegment(
                start_s=round(cursor, 3),
                end_s=round(seg_end, 3),
                reason=reason,
                risk_level=risk_level,
                peak_score=round(peak_score, 3),
                event_count=event_count,
                danger_count=danger_count,
                sample_interval_s=_default_sample_interval_s(risk_level),
                anchor_s=round((cursor + seg_end) / 2.0, 3),
            )
        )
        cursor = seg_end


def build_global_timeline(
    selection: DailySelection,
    *,
    quiet_window_s: float = 30.0,
    active_window_s: float = 30.0,
    event_padding_s: float = 6.0,
    max_anchors_per_minute: int = 3,
) -> list[TimelineSegment]:
    timeline: list[TimelineSegment] = []
    events = sorted(selection.events, key=lambda item: item.peak_s)
    window_start = 0.0

    while window_start < selection.duration_s:
        window_end = min(selection.duration_s, window_start + quiet_window_s)
        window_events = [
            event for event in events
            if event.end_s > window_start and event.start_s < window_end
        ]
        ranked = sorted(
            window_events,
            key=lambda item: (not item.danger_candidate, -item.peak_score, item.peak_s),
        )[:max_anchors_per_minute]
        peak_score = max((item.peak_score for item in window_events), default=0.0)
        danger_count = sum(1 for item in window_events if item.danger_candidate)
        event_count = len(window_events)
        risk_level = _classify_risk_level(
            danger_count=danger_count,
            event_count=event_count,
            peak_score=peak_score,
        )
        reason = "global-motion-cluster" if ranked else "global-baseline"
        anchor_s = (
            ranked[0].peak_s if ranked else (window_start + window_end) / 2.0
        )
        timeline.append(
            TimelineSegment(
                start_s=round(window_start, 3),
                end_s=round(window_end, 3),
                reason=reason,
                risk_level=risk_level,
                peak_score=round(peak_score, 3),
                event_count=event_count,
                danger_count=danger_count,
                sample_interval_s=_default_sample_interval_s(risk_level),
                anchor_s=round(anchor_s, 3),
            )
        )
        window_start = window_end

    return timeline


def _events_for_window(
    events: list[EventSegment],
    *,
    start_s: float,
    end_s: float,
) -> list[EventSegment]:
    return [
        event for event in events
        if event.peak_s >= start_s and event.peak_s < end_s
    ]


def local_rescan_analysis_segments(
    video_path: str | Path,
    selection: DailySelection,
    segments: list[AnalysisSegment],
    *,
    scan_interval_s: float = 1.0,
    change_threshold: float = 8.0,
    danger_threshold: float = 18.0,
    high_motion_threshold: float = 80.0,
    medium_motion_threshold: float = 45.0,
) -> list[AnalysisSegment]:
    """Re-score each analysis window from its own local motion profile.

    The first global event pass can merge a long camera-motion stretch into one
    large event. This local pass prevents that one event from making every
    30-second window look equally high risk.
    """
    import cv2  # type: ignore[import-not-found]

    path = Path(video_path).expanduser()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return segments
    try:
        rescored: list[AnalysisSegment] = []
        for segment in segments:
            points: list[tuple[float, float]] = []
            prev_sig = None
            ts = segment.start_s
            while ts <= segment.end_s:
                frame = _capture_at(cap, ts)
                if frame is None:
                    break
                sig = _signature(frame)
                score = _change_score(prev_sig, sig)
                prev_sig = sig
                if score >= change_threshold:
                    points.append((round(ts, 3), score))
                ts += scan_interval_s

            local_peak = max((score for _ts, score in points), default=0.0)
            local_events = _events_for_window(
                selection.events,
                start_s=segment.start_s,
                end_s=segment.end_s,
            )
            local_danger = sum(1 for event in local_events if event.danger_candidate)
            local_count = len(points)
            selected_count = sum(
                1
                for frame in selection.selected_frames
                if segment.start_s <= frame.timestamp_s < segment.end_s
            )
            if local_peak >= high_motion_threshold or (
                local_danger > 0 and local_peak >= medium_motion_threshold
            ):
                risk_level = "high"
            elif local_peak >= medium_motion_threshold or local_count >= 2:
                risk_level = "medium"
            elif selected_count < 2:
                risk_level = "uncertain"
            else:
                risk_level = "low"

            reason = (
                "local-risk-rescan"
                if risk_level in {"high", "medium", "uncertain"}
                else "global-baseline"
            )
            rescored.append(
                AnalysisSegment(
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    reason=reason,
                    risk_level=risk_level,
                    peak_score=round(local_peak, 3),
                    event_count=local_count,
                    danger_count=local_danger,
                    sample_interval_s=_default_sample_interval_s(risk_level),
                )
            )
        return rescored
    finally:
        cap.release()


def select_daily_video_frames(
    video_path: str | Path,
    *,
    baseline_frames_per_minute: int = 8,
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
            timeline_segments=[],
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

    segment_events = [
        event for event in selection.events
        if event.end_s > start_s and event.start_s < end_s
    ]
    event_peaks = [event.peak_s for event in segment_events]
    priority = {"danger-candidate": 0, "event": 1, "baseline-context": 2}
    chosen: list[SelectedFrame] = []
    seen: set[float] = set()
    for frame in sorted(
        in_segment,
        key=lambda item: (
            priority.get(item.reason, 9),
            min((abs(item.timestamp_s - peak) for peak in event_peaks), default=9999.0),
            item.timestamp_s,
        ),
    ):
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


def choose_adaptive_analysis_segments(
    selection: DailySelection,
    *,
    base_window_s: float | None = None,
    normal_segment_s: float = 30.0,
    medium_segment_s: float = 30.0,
    high_segment_s: float = 30.0,
    min_segment_s: float = 30.0,
) -> list[AnalysisSegment]:
    _ = base_window_s
    segments: list[AnalysisSegment] = []
    timeline = selection.timeline_segments or build_global_timeline(selection)
    for item in timeline:
        segments.append(
            AnalysisSegment(
                start_s=round(item.start_s, 3),
                end_s=round(item.end_s, 3),
                reason=item.reason,
                risk_level=item.risk_level,
                peak_score=round(item.peak_score, 3),
                event_count=item.event_count,
                danger_count=item.danger_count,
                sample_interval_s=item.sample_interval_s,
            )
        )

    deduped: list[AnalysisSegment] = []
    seen: set[tuple[float, float, str, str]] = set()
    for segment in segments:
        key = (segment.start_s, segment.end_s, segment.reason, segment.risk_level)
        if key not in seen:
            deduped.append(segment)
            seen.add(key)
    return sorted(deduped, key=lambda item: (item.start_s, item.end_s))


def choose_timeline_model_frames(
    selection: DailySelection,
    *,
    start_s: float,
    end_s: float,
    min_frames_per_minute: int = 6,
    max_frames: int = 20,
) -> list[SelectedFrame]:
    in_segment = [
        frame for frame in selection.selected_frames
        if start_s <= frame.timestamp_s < end_s
    ]
    segment_duration = max(1.0, end_s - start_s)
    target = max(3, int((segment_duration / 60.0) * min_frames_per_minute + 0.999))
    target = min(max_frames, target)
    if len(in_segment) <= target:
        return in_segment

    priority = {"danger-candidate": 0, "event": 1, "baseline-context": 2}
    chosen: list[SelectedFrame] = []
    for frame in sorted(in_segment, key=lambda item: (priority.get(item.reason, 9), item.timestamp_s)):
        if len(chosen) >= target:
            break
        chosen.append(frame)
    return sorted(chosen, key=lambda item: item.timestamp_s)


def sample_analysis_segment_frames(
    video_path: str | Path,
    selection: DailySelection,
    segment: AnalysisSegment,
    *,
    max_frames: int = 20,
) -> list[SelectedFrame]:
    import cv2  # type: ignore[import-not-found]

    segment_duration = max(0.5, segment.end_s - segment.start_s)
    base_timestamps: list[tuple[float, str, int]] = []
    dense_ts = segment.start_s
    while dense_ts < segment.end_s:
        base_timestamps.append((round(dense_ts, 3), f"{segment.risk_level}-dense", 2))
        dense_ts += segment.sample_interval_s
    base_timestamps.append((round(segment.end_s, 3), f"{segment.risk_level}-dense", 2))

    for frame in selection.selected_frames:
        if segment.start_s <= frame.timestamp_s <= segment.end_s:
            priority = 0 if frame.reason == "danger-candidate" else 1
            base_timestamps.append((round(frame.timestamp_s, 3), frame.reason, priority))

    for event in selection.events:
        if event.end_s > segment.start_s and event.start_s < segment.end_s:
            for ts in event.frame_timestamps_s:
                if segment.start_s <= ts <= segment.end_s:
                    priority = 0 if event.danger_candidate else 1
                    reason = "danger-candidate" if event.danger_candidate else "event"
                    base_timestamps.append((round(ts, 3), reason, priority))

    deduped: list[tuple[float, str, int]] = []
    seen: set[float] = set()
    for ts, reason, priority in sorted(base_timestamps, key=lambda item: (item[2], item[0])):
        if ts not in seen:
            deduped.append((ts, reason, priority))
            seen.add(ts)

    if len(deduped) > max_frames:
        pinned = [item for item in deduped if item[2] == 0]
        remaining = [item for item in deduped if item[2] != 0]
        chosen = pinned[:max_frames]
        slots = max_frames - len(chosen)
        if slots > 0 and remaining:
            if len(remaining) <= slots:
                chosen.extend(remaining)
            else:
                for idx in range(slots):
                    pick = round(idx * (len(remaining) - 1) / max(1, slots - 1))
                    chosen.append(remaining[pick])
        deduped = sorted(chosen, key=lambda item: item[0])

    path = Path(video_path).expanduser()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return choose_timeline_model_frames(
            selection,
            start_s=segment.start_s,
            end_s=segment.end_s,
            min_frames_per_minute=6,
            max_frames=max_frames,
        )
    try:
        sampled: list[SelectedFrame] = []
        for ts, reason, _priority in deduped:
            frame = _capture_at(cap, ts)
            if frame is None:
                continue
            sampled.append(
                SelectedFrame(
                    timestamp_s=ts,
                    reason=reason,
                    data_url=_jpeg_data_url(frame),
                )
            )
        return sampled
    finally:
        cap.release()
