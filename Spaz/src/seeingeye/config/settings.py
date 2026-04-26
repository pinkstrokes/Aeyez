"""SeeingEye configuration via pydantic-settings with TOML + env-var sources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


_TOML_PATH = Path(__file__).parent / "seeingeye.toml"


class Settings(BaseSettings):
    """Configuration for the SeeingEye LangGraph pipeline.

    Values are loaded in priority order:
      1. Environment variables (prefix ``SEEINGEYE_``)
      2. TOML file (``seeingeye.toml`` next to this module)
      3. Field defaults below (mirrors the TOML for documentation)

    NO ``api_key`` fields — credentials are env-var only (D-04).
    """

    model_config = SettingsConfigDict(
        toml_file=_TOML_PATH,
        env_prefix="SEEINGEYE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Endpoint URLs
    translator_base_url: str = "https://api.openai.com/v1"
    reasoner_base_url: str = "https://api.openai.com/v1"

    # Model identifiers
    translator_model: str = "gpt-5.4-mini"
    translator_escalation_model: str = "gpt-5.4-mini"
    reasoner_model: str = "gpt-5.4-mini"
    translator_reasoning_effort: str | None = None
    reasoner_reasoning_effort: str | None = None

    # Analysis mode
    analysis_mode: str = "default"
    safety_framework: str = "OSHA"
    safety_prediction_horizon: str = "near-term"
    safety_navigation_schema: str = "dynamic_route_v1"
    safety_scan_prompt: str = (
        "Please inspect the nearby area for hazards and identify the safest next action."
    )
    video_reasoning_framework: str = "RSTR"

    # Paper hyperparameters (Section 4.1)
    max_iters: int = 3
    n_t: int = 3
    n_r: int = 3

    # vLLM sampling parameters
    temperature: float = 0.01
    top_p: float = 0.001
    top_k: int = 1
    repetition_penalty: float = 1.1
    max_tokens: int = 4096

    # Video frame sampling
    video_frame_interval_s: float = 0.5
    video_max_frames: int = 64
    video_frame_selection: str = "change"
    video_scene_change_threshold: float = 6.0

    # Observability
    langsmith_tracing: bool = False
    safety_event_log_enabled: bool = True
    safety_event_log_path: str = "logs/safety_events.jsonl"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > env vars > TOML file > field defaults."""
        if os.getenv("SEEINGEYE_DISABLE_DOTENV") == "1":
            return (
                init_settings,
                env_settings,
                TomlConfigSettingsSource(settings_cls),
                file_secret_settings,
            )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
