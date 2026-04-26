from __future__ import annotations


async def test_investigate_response_shape(client):
    r = await client.post("/investigate", json={"event": "_describe", "image_b64": "x"})
    assert r.status_code == 200
    body = r.json()
    assert {"event", "prompt", "response", "elapsed_seconds", "success"} <= body.keys()
    assert body["success"] is True
    assert isinstance(body["elapsed_seconds"], (int, float))


async def test_investigate_unauthenticated_does_not_persist(client):
    """Unauthenticated requests still work but aren't saved."""
    await client.post("/investigate", json={"event": "_describe", "image_b64": "x"})
    # No auth header → no way to read history, but we can register fresh and confirm empty
    r = await client.post("/auth/register", json={"username": "alice", "password": "secret"})
    headers = {"Authorization": f"Bearer {r.json()['token']}"}
    rows = (await client.get("/history", headers=headers)).json()
    assert rows == []


async def test_analyze_change_response_shape(client):
    r = await client.post("/analyze-change", json={"frame0_b64": "a", "frame1_b64": "b"})
    assert r.status_code == 200
    body = r.json()
    assert {"response", "elapsed_seconds", "success"} <= body.keys()
    assert body["success"] is True


async def test_chat_no_history_returns_text(client, auth_headers):
    r = await client.post(
        "/chat",
        json={"text": "hi", "image_b64": "data:image/jpeg;base64,eA=="},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hi"
    assert body["audio_b64"] is None  # ElevenLabs mocked off
    assert body["referenced_location"] is None
    assert body["response"] == "Temporal summary from fake Spaz."


async def test_chat_referenced_location_for_object_query(client, auth_headers):
    """When the user asks 'where did I see X' and history mentions X, the
    response should populate referenced_location with the matching capture."""
    # Seed: investigate at known coords; the stub response contains the word
    # "describe" — so a query for "describe" will match.
    await client.post(
        "/investigate",
        json={"event": "_describe", "image_b64": "x", "lat": 40.0, "lon": -74.0},
        headers=auth_headers,
    )
    r = await client.post(
        "/chat", json={"text": "where did I see describe?"}, headers=auth_headers
    )
    body = r.json()
    assert body["success"] is True
    ref = body["referenced_location"]
    assert ref is not None
    assert ref["lat"] == 40.0
    assert ref["lon"] == -74.0


async def test_chat_referenced_location_for_named_location(client, auth_headers):
    """Asking about a saved location by name should reference it."""
    await client.post(
        "/locations", json={"name": "kitchen", "lat": 1.0, "lon": 2.0}, headers=auth_headers
    )
    r = await client.post(
        "/chat", json={"text": "what's in my kitchen?"}, headers=auth_headers
    )
    body = r.json()
    assert body["referenced_location"] is not None
    assert body["referenced_location"]["name"] == "kitchen"


async def test_safe_mode_with_image(client, auth_headers):
    r = await client.post(
        "/safe-mode", json={"image_b64": "x"}, headers=auth_headers
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["response"] == "Temporal summary from fake Spaz."


async def test_safe_mode_with_recent_frames(client, auth_headers):
    r = await client.post(
        "/safe-mode", json={"recent_frames": ["a", "b", "c"]}, headers=auth_headers
    )
    assert r.status_code == 200
    assert r.json()["response"] == "Temporal summary from fake Spaz."


async def test_daily_video_summary_uses_keyframe_selector(client, auth_headers, monkeypatch):
    import server
    from video_analysis import DailySelection, EventSegment, SelectedFrame

    frames = [
        SelectedFrame(timestamp_s=0.0, reason="baseline", data_url="data:image/jpeg;base64,a"),
        SelectedFrame(timestamp_s=12.0, reason="baseline", data_url="data:image/jpeg;base64,b"),
        SelectedFrame(timestamp_s=24.0, reason="event", data_url="data:image/jpeg;base64,c"),
    ]
    event = EventSegment(
        start_s=20.0,
        end_s=28.0,
        peak_s=24.0,
        peak_score=22.5,
        danger_candidate=True,
        frame_timestamps_s=[20.0, 24.0, 28.0],
    )

    def _fake_select(*_args, **_kwargs):
        return DailySelection(
            duration_s=60.0,
            fps=30.0,
            baseline_interval_s=12.0,
            baseline_target_count=5,
            selected_frames=frames,
            events=[event],
        )

    async def _fake_k2(_prompt: str, max_tokens: int = 900):
        return (
            "1. Today Summary:\n- Demo summary.\n"
            "2. Dangers:\n- 00:20-00:28: Possible hazard near the center.\n"
            "3. Timeline:\n- 00:24: Motion changed.\n"
            "4. Confidence Notes:\n- Test."
        )

    monkeypatch.setattr(server, "select_daily_video_frames", _fake_select)
    monkeypatch.setattr(server, "_complete_with_k2", _fake_k2)

    r = await client.post(
        "/daily-video-summary",
        files={"video": ("demo.mp4", b"not a real video", "video/mp4")},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["duration_seconds"] == 60.0
    assert body["retained_frame_count"] == 3
    assert body["model_frame_count"] == 3
    assert body["baseline_frames_per_minute"] == 5
    assert body["events"][0]["danger_candidate"] is True
    assert body["timeline"]
    assert body["dangers"]


def test_adaptive_video_segments_split_risk_events_without_unbounded_calls():
    from video_analysis import DailySelection, EventSegment, choose_adaptive_analysis_segments

    selection = DailySelection(
        duration_s=600.0,
        fps=30.0,
        baseline_interval_s=12.0,
        baseline_target_count=51,
        selected_frames=[],
        events=[
            EventSegment(
                start_s=120.0,
                end_s=132.0,
                peak_s=126.0,
                peak_score=24.0,
                danger_candidate=True,
                frame_timestamps_s=[120.0, 126.0, 132.0],
            ),
            EventSegment(
                start_s=180.0,
                end_s=190.0,
                peak_s=184.0,
                peak_score=12.0,
                danger_candidate=False,
                frame_timestamps_s=[180.0, 184.0, 190.0],
            ),
        ],
    )

    segments = choose_adaptive_analysis_segments(selection, base_window_s=300.0)

    risk_segments = [segment for segment in segments if segment.reason == "global-motion-cluster"]
    assert risk_segments
    assert any(segment.danger_count >= 1 for segment in risk_segments)
    assert any(segment.end_s > 120.0 and segment.start_s < 132.0 for segment in risk_segments)
    assert len([segment for segment in segments if segment.start_s < 300.0]) <= 25
    assert any(segment.reason == "global-baseline" for segment in segments)


def test_safe_mode_prompt_includes_action_motion_path_model():
    import server

    prompt = server._safe_mode_prompt("recent context", "which way?", 3)

    assert "Action = hand pose + active object + contact target + temporal motion + scene context" in prompt
    assert "movable entities" in prompt
    assert "short-horizon motion paths" in prompt
    assert "line-of-fire" in prompt
    assert "temporary obstruction that may clear" in prompt
    assert "frames are chronological" in prompt
    assert "head or shoulder height are hard no-go overhead hazards" in prompt
    assert "Compare left, center, and right" in prompt


def test_daily_video_prompt_forbids_option_style_output():
    import server

    prompt = server._daily_video_segment_prompt(
        [{"timestamp": "00:00:12", "reason": "baseline"}],
        start_s=0.0,
        end_s=60.0,
        timeline_reason="global-motion-cluster",
        risk_level="medium",
    )

    assert "Do not output answer labels such as A, B, C, or D." in prompt
    assert "multiple-choice options" in prompt
    assert "Stage-1 timeline label: global-motion-cluster" in prompt
    assert "Local risk level: medium" in prompt
    assert "This is not a test question, benchmark item, or multiple-choice task." in prompt
    assert "using the same safety-focused mindset as safe mode" in prompt
    assert "Explain the scene directly to the user in plain language." in prompt
    assert "Do not include your own thoughts, meta commentary, reasoning process" in prompt
    assert "Only output information the user can directly use" in prompt
    assert "0. Summary:" in prompt
    assert "the most important danger, where that danger is" in prompt
    assert "the timestamp where the user can review it" in prompt
    assert "1. Segment action:" in prompt
    assert "2. Movement timeline:" in prompt
    assert "3. Hazard table:" in prompt
    assert "4. Route / obstruction notes:" in prompt
    assert "5. Confidence:" in prompt
    assert "Time range | Scene position | Hazard type | Why it is dangerous | Recommended next action" in prompt
    assert "Every hazard row must state where the danger is in the scene" in prompt
    assert "so the user can find and review it in the video" in prompt


def test_daily_video_final_prompt_uses_reference_template():
    import server
    from video_analysis import DailySelection, SelectedFrame

    selection = DailySelection(
        duration_s=240.0,
        fps=30.0,
        baseline_interval_s=12.0,
        baseline_target_count=20,
        selected_frames=[
            SelectedFrame(timestamp_s=0.0, reason="baseline", data_url="data:image/jpeg;base64,a"),
        ],
        events=[],
        timeline_segments=[],
    )

    prompt = server._daily_video_final_prompt(
        selection=selection,
        segment_summaries=[],
    )

    assert "=== FINAL SUMMARY ===" in prompt
    assert "Overall summary:" in prompt
    assert "Danger locations:" in prompt
    assert "<timestamp range>: <where in the scene> - <danger> - <why it is dangerous>" in prompt
    assert "timestamp, place in the scene, reason it is dangerous" in prompt
    assert "00:00:00-00:02:00:" in prompt
    assert "0. Summary" in prompt
    assert "1. Segment action" in prompt
    assert "2. Movement timeline" in prompt
    assert "3. Hazard table" in prompt
    assert "4. Route / obstruction notes" in prompt
    assert "5. Confidence" in prompt


def test_daily_video_rejects_option_like_output():
    import server

    assert server._is_unhelpful_video_summary_text("A") is True
    assert server._is_unhelpful_video_summary_text("Option B") is True
    assert server._is_unhelpful_video_summary_text("There are no answer options, so here is the summary.") is True
    assert server._is_unhelpful_video_summary_text("I’ll answer from the scene description alone. No calculation is needed because this is a visual-safety judgment, not a numeric problem.") is True
    assert server._is_unhelpful_video_summary_text("中文：工人在狭窄平台上施工") is True
    assert server._is_unhelpful_video_summary_text(
        "- Workers move blocks near the right side.\n- Hazard at 00:24 near the edge.\n- Wait for the route to clear."
    ) is False


def test_runtime_answer_guard_rejects_placeholders():
    import server

    assert server._is_unhelpful_runtime_answer("A") is True
    assert server._is_unhelpful_runtime_answer("Left") is True
    assert server._is_unhelpful_runtime_answer("Stub response — real model coming soon.") is True
    assert server._is_unhelpful_runtime_answer(
        "Right after a short wait; the left path has a low hanging chain in the head-level corridor."
    ) is False


def test_latest_frames_keep_chronological_current_frame_last():
    import server

    assert server._latest_frames_for_chat("now", ["old", "middle"]) == [
        "old",
        "middle",
        "now",
    ]


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["mode"] == "spaz"
    assert body["spaz_available"] is True
    assert body["spaz_root_found"] is True
    assert body["spaz_import_ok"] is True
    assert body["spaz_runtime_ok"] is True
    assert body["openai_ready"] is True
    assert body["k2_ready"] is True
    assert body["issues"] == []


async def test_health_returns_503_when_runtime_probe_fails(client, monkeypatch):
    import seeingeye_bridge

    monkeypatch.setattr(
        seeingeye_bridge,
        "runtime_probe",
        lambda: seeingeye_bridge.RuntimeProbe(
            ok=False,
            import_ok=True,
            runtime_ok=False,
            root_found=True,
            reason="runner import failed",
        ),
    )

    r = await client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert body["spaz_available"] is False
    assert body["spaz_runtime_ok"] is False
    assert "spaz_runtime_unavailable" in body["issues"]


async def test_security_headers_present(client):
    r = await client.get("/health")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in r.headers
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    # HSTS only set in prod (AEYEZ_ENV=prod); tests run in dev, so absent.
    assert "Strict-Transport-Security" not in r.headers
