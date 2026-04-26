"""Tests for the seeingeye Settings model (FND-01)."""

import os
import pytest


def _clean_seeingeye_env(monkeypatch):
    """Remove any SEEINGEYE_* env vars to ensure clean defaults."""
    for key in list(os.environ):
        if key.startswith("SEEINGEYE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SEEINGEYE_DISABLE_DOTENV", "1")


def test_settings_loads_paper_defaults(monkeypatch):
    """Settings() loads with paper hyperparameter defaults."""
    _clean_seeingeye_env(monkeypatch)
    from src.seeingeye.config.settings import Settings

    s = Settings()
    assert s.max_iters == 3
    assert s.n_t == 3
    assert s.n_r == 3
    assert s.translator_base_url == "https://api.openai.com/v1"
    assert s.reasoner_base_url == "https://api.openai.com/v1"
    assert s.temperature == 0.01
    assert s.top_p == 0.001
    assert s.top_k == 1
    assert s.repetition_penalty == 1.1
    assert s.max_tokens == 4096
    assert s.video_frame_interval_s == 0.5
    assert s.video_max_frames == 64
    assert s.video_frame_selection == "change"
    assert s.video_scene_change_threshold == 6.0
    assert s.safety_navigation_schema == "dynamic_route_v1"
    assert s.safety_event_log_enabled is True
    assert s.safety_event_log_path == "logs/safety_events.jsonl"


def test_settings_has_no_api_key_fields(monkeypatch):
    """Settings schema must NOT contain any field with 'api_key' in its name."""
    _clean_seeingeye_env(monkeypatch)
    from src.seeingeye.config.settings import Settings

    for field_name in Settings.model_fields:
        assert "api_key" not in field_name, (
            f"Settings must not have api_key fields, found: {field_name}"
        )


def test_env_var_override(monkeypatch):
    """Environment variables with SEEINGEYE_ prefix override TOML defaults."""
    _clean_seeingeye_env(monkeypatch)
    monkeypatch.setenv("SEEINGEYE_TRANSLATOR_BASE_URL", "http://custom:9000/v1")
    from src.seeingeye.config.settings import Settings

    s = Settings()
    assert s.translator_base_url == "http://custom:9000/v1"


def test_langsmith_tracing_default(monkeypatch):
    """Settings has langsmith_tracing field defaulting to False."""
    _clean_seeingeye_env(monkeypatch)
    from src.seeingeye.config.settings import Settings

    s = Settings()
    assert s.langsmith_tracing is False


def test_model_defaults(monkeypatch):
    """translator_model and reasoner_model have correct defaults."""
    _clean_seeingeye_env(monkeypatch)
    from src.seeingeye.config.settings import Settings

    s = Settings()
    assert s.translator_model == "gpt-5.4-mini"
    assert s.translator_escalation_model == "gpt-5.4-mini"
    assert s.reasoner_model == "gpt-5.4-mini"
    assert s.translator_reasoning_effort is None
    assert s.reasoner_reasoning_effort is None
