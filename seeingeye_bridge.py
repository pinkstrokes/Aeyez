from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class SpazUnavailableError(RuntimeError):
    """Raised when the local Spaz checkout is missing or unusable."""


@dataclass(frozen=True)
class BridgeStatus:
    available: bool
    root: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeProbe:
    ok: bool
    import_ok: bool
    runtime_ok: bool
    root_found: bool
    reason: str | None = None


def _candidate_roots() -> list[Path]:
    env_path = (
        os.environ.get("SPAZ_PATH") or os.environ.get("SEEINGEYE_PATH") or ""
    ).strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append((Path(__file__).resolve().parent / "Spaz").resolve())
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


def _require_runner_module():
    if not STATUS.available or STATUS.root is None:
        raise SpazUnavailableError(STATUS.reason or "Spaz is unavailable.")
    if str(STATUS.root) not in sys.path:
        sys.path.insert(0, str(STATUS.root))
    _load_env_file(STATUS.root / ".env")
    _mirror_spaz_env_to_runtime()
    try:
        from src.seeingeye.runtime import runner  # type: ignore
        from src.seeingeye.observability.logging import configure_logging  # type: ignore
        from src.seeingeye.runtime.result import SeeingEyeResult  # type: ignore
        from src.seeingeye.state.sir import SIR  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on broken installs
        raise SpazUnavailableError(f"Failed to import Spaz runtime: {exc}") from exc
    return runner, configure_logging, SeeingEyeResult, SIR


def runtime_probe() -> RuntimeProbe:
    """Best-effort readiness probe for the local Spaz runtime.

    This verifies more than "the folder exists": the runtime must be importable,
    and the graph runner module used by multi-frame requests must also import.
    """
    if not STATUS.available or STATUS.root is None:
        return RuntimeProbe(
            ok=False,
            import_ok=False,
            runtime_ok=False,
            root_found=False,
            reason=STATUS.reason or "Spaz root not found.",
        )

    try:
        _require_runtime()
    except Exception as exc:
        return RuntimeProbe(
            ok=False,
            import_ok=False,
            runtime_ok=False,
            root_found=True,
            reason=str(exc),
        )

    try:
        _require_runner_module()
    except Exception as exc:
        return RuntimeProbe(
            ok=False,
            import_ok=True,
            runtime_ok=False,
            root_found=True,
            reason=str(exc),
        )

    return RuntimeProbe(
        ok=True,
        import_ok=True,
        runtime_ok=True,
        root_found=True,
        reason=None,
    )


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


def _frame_from_b64(frame_b64: str, timestamp_s: float) -> dict:
    frame_bytes, mime_type = _strip_data_url(frame_b64)
    return {
        "b64": base64.b64encode(frame_bytes).decode("ascii"),
        "timestamp_s": round(timestamp_s, 3),
        "mime_type": mime_type,
    }


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


async def run_on_image(question: str, image_b64: str):
    run_question = _require_runtime()
    with tempfile.TemporaryDirectory(prefix="aeyez-spaz-") as tmpdir:
        image_path = _write_temp_image(image_b64, tmpdir, "image")
        return await run_question(question=question, image_path=image_path)


async def run_on_frames(question: str, frames_b64: Iterable[str]):
    frames = [frame for frame in frames_b64 if frame]
    if not frames:
        raise ValueError("At least one frame is required.")
    if len(frames) == 1:
        return await run_on_image(question, frames[0])

    runner, configure_logging, SeeingEyeResult, SIR = _require_runner_module()
    configure_logging()
    image_frames = [
        _frame_from_b64(frame_b64, timestamp_s=index * 1.5)
        for index, frame_b64 in enumerate(frames)
    ]
    graph = runner._get_graph()
    initial_state = {
        "sir": SIR(content=""),
        "outer_iter": 0,
        "question": question,
        "options": None,
        "media_type": "video",
        "image_b64": None,
        "image_frames": image_frames,
        "translator_messages": [],
        "reasoner_messages": [],
        "reasoner_feedback": None,
        "final_answer": None,
    }
    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 50})
    return SeeingEyeResult(
        answer=final_state.get("final_answer") or "",
        sir=final_state["sir"],
        outer_iters_used=final_state.get("outer_iter", 0),
        total_tokens=runner._sum_total_tokens(final_state),
    )


async def run_on_frame_payloads(question: str, frames: Iterable[dict[str, Any]]):
    payloads = [frame for frame in frames if frame.get("b64")]
    if not payloads:
        raise ValueError("At least one frame is required.")
    if len(payloads) == 1:
        frame = payloads[0]
        data_url = f"data:{frame.get('mime_type', 'image/jpeg')};base64,{frame['b64']}"
        return await run_on_image(question, data_url)

    runner, configure_logging, SeeingEyeResult, SIR = _require_runner_module()
    configure_logging()
    image_frames = [
        {
            "b64": frame["b64"],
            "timestamp_s": frame.get("timestamp_s"),
            "mime_type": frame.get("mime_type") or "image/jpeg",
        }
        for frame in payloads
    ]
    graph = runner._get_graph()
    initial_state = {
        "sir": SIR(content=""),
        "outer_iter": 0,
        "question": question,
        "options": None,
        "media_type": "video",
        "image_b64": None,
        "image_frames": image_frames,
        "translator_messages": [],
        "reasoner_messages": [],
        "reasoner_feedback": None,
        "final_answer": None,
    }
    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 50})
    return SeeingEyeResult(
        answer=final_state.get("final_answer") or "",
        sir=final_state["sir"],
        outer_iters_used=final_state.get("outer_iter", 0),
        total_tokens=runner._sum_total_tokens(final_state),
    )


def run_sync(coro):
    return asyncio.run(coro)
