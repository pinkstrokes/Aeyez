from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

from src.seeingeye.config.settings import Settings
from src.seeingeye.runtime import run_question


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seeingeye-run",
        description="Run the SeeingEye pipeline on a single image or video question.",
    )
    parser.add_argument("--question", help="Question to answer.")
    parser.add_argument(
        "--image",
        help="Path to the image file to analyze.",
    )
    parser.add_argument(
        "--video",
        help="Path to the video file to analyze.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Video frame sampling interval in seconds (0.1 to 1.0).",
    )
    parser.add_argument(
        "--frame-selection",
        choices=["uniform", "change"],
        default=None,
        help="Video frame selection strategy. 'change' keeps high-motion/key-change frames.",
    )
    parser.add_argument(
        "--scene-change-threshold",
        type=float,
        default=None,
        help="Mean grayscale frame-difference threshold for --frame-selection change.",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="Multiple-choice option. Pass this flag multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON.",
    )
    parser.add_argument(
        "--show-sir",
        action="store_true",
        help="Also print the final SIR text in plain-text mode.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check configuration and backend readiness without running a question.",
    )
    return parser


def _is_local_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1"}


def _can_connect(base_url: str, timeout_s: float = 1.5) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _doctor() -> int:
    settings = Settings()
    api_key_present = bool(
        os.getenv("SEEINGEYE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if not api_key_present and Path(".env").exists():
        env_values = dotenv_values(".env")
        api_key_present = bool(
            env_values.get("SEEINGEYE_API_KEY")
            or env_values.get("OPENAI_API_KEY")
            or env_values.get("OPENROUTER_API_KEY")
        )

    print("SeeingEye doctor")
    print(f"translator_base_url: {settings.translator_base_url}")
    print(f"reasoner_base_url:   {settings.reasoner_base_url}")
    print(f"translator_model:    {settings.translator_model}")
    print(f"translator_escalation_model: {settings.translator_escalation_model}")
    print(f"reasoner_model:      {settings.reasoner_model}")
    print(f"video_frame_interval_s: {settings.video_frame_interval_s}")
    print(f"video_max_frames:    {settings.video_max_frames}")
    print(f"video_frame_selection: {settings.video_frame_selection}")
    print(f"video_scene_change_threshold: {settings.video_scene_change_threshold}")
    print(f"video_reasoning_framework: {settings.video_reasoning_framework}")
    print(f"analysis_mode:       {settings.analysis_mode}")
    print(f"api_key_present:     {'yes' if api_key_present else 'no'}")

    for name, base_url in (
        ("translator", settings.translator_base_url),
        ("reasoner", settings.reasoner_base_url),
    ):
        if _is_local_url(base_url):
            reachable = _can_connect(base_url)
            print(f"{name}_backend_reachable: {'yes' if reachable else 'no'}")
        else:
            print(f"{name}_backend_reachable: remote endpoint configured")

    if not api_key_present and not (
        _is_local_url(settings.translator_base_url)
        and _is_local_url(settings.reasoner_base_url)
    ):
        print()
        print("Missing API key. Set SEEINGEYE_API_KEY in .env or your shell.")
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        return _doctor()

    if not args.question:
        parser.error("--question is required unless --doctor is used")

    if bool(args.image) == bool(args.video):
        parser.error("provide exactly one of --image or --video")

    image_path = Path(args.image).expanduser() if args.image else None
    video_path = Path(args.video).expanduser() if args.video else None
    if image_path and not image_path.exists():
        parser.error(f"image not found: {image_path}")
    if video_path and not video_path.exists():
        parser.error(f"video not found: {video_path}")
    if args.frame_interval is not None and not 0.1 <= args.frame_interval <= 1.0:
        parser.error("--frame-interval must be between 0.1 and 1.0 seconds")

    options = args.option or None
    result = asyncio.run(
        run_question(
            question=args.question,
            image_path=image_path,
            options=options,
            video_path=video_path,
            frame_interval_s=args.frame_interval,
            frame_selection=args.frame_selection,
            scene_change_threshold=args.scene_change_threshold,
        )
    )

    if args.json:
        payload = {
            "answer": result.answer,
            "sir": result.sir.model_dump(),
            "outer_iters_used": result.outer_iters_used,
            "total_tokens": result.total_tokens,
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    print(result.answer)
    if args.show_sir:
        print()
        print("SIR:")
        print(result.sir.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
