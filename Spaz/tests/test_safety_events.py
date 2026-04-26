from src.seeingeye.observability.safety_events import (
    log_safety_event,
    summarize_daily_events,
)


def test_log_safety_event_and_daily_summary(tmp_path):
    log_path = tmp_path / "safety_events.jsonl"

    event = log_safety_event(
        log_path=log_path,
        question="Which way should I go?",
        answer="SAFETY REPORT\nSafest route: wait.",
        sir="AUTOMATIC MECHANICS HAZARD SCAN\nmechanical_scene_model: ...\nego_centered_route_map: ...",
        outer_iters_used=2,
        total_tokens=123,
        media_type="image",
    )

    assert log_path.exists()
    assert event["has_safety_report"] is True
    assert event["has_route_map"] is True
    assert event["has_mechanics_scan"] is True

    summary = summarize_daily_events(
        log_path=log_path,
        date_utc=event["date_utc"],
    )
    assert summary["total_events"] == 1
    assert summary["tooling_flags"]["has_mechanics_scan"] == 1
