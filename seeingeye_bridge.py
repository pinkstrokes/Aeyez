from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class SpazUnavailableError(RuntimeError):
    """Raised when the local Spaz checkout is missing or unusable."""


@dataclass(frozen=True)
class BridgeStatus:
    available: bool
    root: Path | None
    reason: str | None = None


def _candidate_roots() -> list[Path]:
    env_path = (
        os.environ.get("SPAZ_PATH") or os.environ.get("SEEINGEYE_PATH") or ""
    ).strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append((Path(__file__).resolve().parent.parent / "Spaz").resolve())
    candidates.append((Path(__file__).resolve().parent.parent / "seeingeye").resolve())
    return candidates


def _find_root() -> BridgeStatus:
    for root in _candidate_roots():
        if (root / "src" / "seeingeye" / "runtime" / "__init__.py").exists():
            return BridgeStatus(available=True, root=root)
    return BridgeStatus(
        available=False,
        root=None,
        reason=(
            "No local Spaz checkout found. Set SPAZ_PATH or place the repo "
            "next to Aeyez."
        ),
    )


STATUS = _find_root()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _mirror_spaz_env_to_runtime() -> None:
    """Let the renamed app accept SPAZ_* while the Python package is still src.seeingeye."""
    for key, value in list(os.environ.items()):
        if not key.startswith("SPAZ_") or key == "SPAZ_PATH":
            continue
        legacy_key = "SEEINGEYE_" + key[len("SPAZ_") :]
        os.environ.setdefault(legacy_key, value)


def _require_runtime():
    if not STATUS.available or STATUS.root is None:
        raise SpazUnavailableError(STATUS.reason or "Spaz is unavailable.")
    if str(STATUS.root) not in sys.path:
        sys.path.insert(0, str(STATUS.root))
    _load_env_file(STATUS.root / ".env")
    _mirror_spaz_env_to_runtime()
    try:
        from src.seeingeye.runtime import run_question  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on broken installs
        raise SpazUnavailableError(f"Failed to import Spaz runtime: {exc}") from exc
    return run_question


def _strip_data_url(value: str) -> tuple[bytes, str]:
    payload = value.strip()
    mime_type = "image/jpeg"
    if payload.startswith("data:"):
        header, _, payload = payload.partition(",")
        if ";" in header:
            mime_type = header[5:].split(";", 1)[0] or mime_type
    try:
        return base64.b64decode(payload, validate=False), mime_type
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def _image_suffix(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(mime_type.lower(), ".jpg")


def _write_temp_image(image_b64: str, directory: str, stem: str) -> Path:
    image_bytes, mime_type = _strip_data_url(image_b64)
    path = Path(directory) / f"{stem}{_image_suffix(mime_type)}"
    path.write_bytes(image_bytes)
    return path


def _write_temp_video(frames_b64: Iterable[str], directory: str) -> Path:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    decoded_frames = []
    for frame_b64 in frames_b64:
        frame_bytes, _mime_type = _strip_data_url(frame_b64)
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode one of the provided frames.")
        decoded_frames.append(frame)
    if not decoded_frames:
        raise ValueError("At least one frame is required.")

    height, width = decoded_frames[0].shape[:2]
    video_path = Path(directory) / "frames.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError("Could not create temporary video for Spaz.")
    try:
        for frame in decoded_frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return video_path


async def run_on_image(question: str, image_b64: str):
    run_question = _require_runtime()
    with tempfile.TemporaryDirectory(prefix="aeyez-spaz-") as tmpdir:
        image_path = _write_temp_image(image_b64, tmpdir, "image")
        return await run_question(question=question, image_path=image_path)


async def run_on_frames(question: str, frames_b64: Iterable[str]):
    run_question = _require_runtime()
    frames = [frame for frame in frames_b64 if frame]
    if not frames:
        raise ValueError("At least one frame is required.")
    if len(frames) == 1:
        return await run_on_image(question, frames[0])
    with tempfile.TemporaryDirectory(prefix="aeyez-spaz-") as tmpdir:
        video_path = _write_temp_video(frames, tmpdir)
        return await run_question(
            question=question,
            video_path=video_path,
            frame_interval_s=1.0,
            frame_selection="uniform",
        )


def run_sync(coro):
    return asyncio.run(coro)
