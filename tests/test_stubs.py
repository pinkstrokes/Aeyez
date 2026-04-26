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
    assert body["dangers"]


def test_safe_mode_prompt_includes_action_motion_path_model():
    import server

    prompt = server._safe_mode_prompt("recent context", "which way?", 3)

    assert "Action = hand pose + active object + contact target + temporal motion + scene context" in prompt
    assert "movable entities" in prompt
    assert "short-horizon motion paths" in prompt
    assert "line-of-fire" in prompt
    assert "temporary obstruction that may clear" in prompt
    assert "frames are chronological" in prompt


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


async def test_security_headers_present(client):
    r = await client.get("/health")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in r.headers
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    # HSTS only set in prod (AEYEZ_ENV=prod); tests run in dev, so absent.
    assert "Strict-Transport-Security" not in r.headers
