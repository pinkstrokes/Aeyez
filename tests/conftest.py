"""
Per-test fresh SQLite via tmp_path. aiosqlite reopens connections per call so
:memory: doesn't share state across operations — file-on-disk in tmp is the
bulletproof option (and still <50ms per test).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bypass the prod JWT-secret guard for the entire test session.
os.environ.setdefault("AEYEZ_SKIP_SECRET_CHECK", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import database  # noqa: E402
import server  # noqa: E402
import seeingeye_bridge  # noqa: E402


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    await database.init_db()
    # Skip ElevenLabs: every chat/safe-mode test should get audio_b64=None.
    async def _no_tts(_text: str):
        return None
    monkeypatch.setattr(server, "_elevenlabs_tts", _no_tts)
    async def _no_summary(_text: str):
        return None
    monkeypatch.setattr(server, "_summarize_with_k2", _no_summary)
    async def _no_final_spoken(_text: str, **_kwargs):
        return None
    monkeypatch.setattr(server, "_final_spoken_with_k2", _no_final_spoken)
    monkeypatch.setattr(
        seeingeye_bridge,
        "STATUS",
        seeingeye_bridge.BridgeStatus(
            available=True,
            root=ROOT.parent / "seeingeye",
            reason=None,
        ),
    )

    class _FakeResult:
        def __init__(self, answer: str):
            self.answer = answer

    async def _fake_run_on_image(question: str, image_b64: str):
        assert question
        assert image_b64
        return _FakeResult("Describe surroundings from fake Spaz.")

    async def _fake_run_on_frames(question: str, frames_b64: list[str]):
        assert question
        assert frames_b64
        return _FakeResult("Temporal summary from fake Spaz.")

    async def _fake_run_on_frame_payloads(question: str, frames: list[dict]):
        assert question
        assert frames
        return _FakeResult("Temporal summary from fake Spaz.")

    monkeypatch.setattr(seeingeye_bridge, "run_on_image", _fake_run_on_image)
    monkeypatch.setattr(seeingeye_bridge, "run_on_frames", _fake_run_on_frames)
    monkeypatch.setattr(seeingeye_bridge, "run_on_frame_payloads", _fake_run_on_frame_payloads)
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client: AsyncClient, username: str = "alice", password: str = "secret") -> dict:
    r = await client.post("/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


@pytest_asyncio.fixture
async def auth_token(client) -> str:
    data = await _register(client)
    return data["token"]


@pytest_asyncio.fixture
async def auth_headers(auth_token) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def register_user():
    """Helper to register additional users in cross-user-isolation tests."""
    return _register
