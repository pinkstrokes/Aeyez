"""
Aeyez demo backend.

Endpoints:
  POST /auth/register       — create account
  POST /auth/login          — get JWT token
  GET  /profile             — fetch current user profile + history count
  PATCH /profile            — update display name and/or password
  GET  /history             — fetch user's event history (requires auth)
  GET  /locations           — list saved named locations (requires auth)
  POST /locations           — save a new named location (requires auth)
  DELETE /locations/{id}    — remove a saved location (requires auth)
  PATCH /locations/{id}     — rename a saved location (requires auth)
  POST /investigate         — single-frame scene description (optional auth → history saved)
  POST /analyze-change      — two-frame scene-diff narration   (optional auth → history saved)
  POST /chat                — voice chat + ElevenLabs TTS       (optional auth → history saved)
  POST /safe-mode           — active guidance mode with recent frames (optional auth → history saved)
  GET  /health              — readiness probe

Env vars:
  ELEVENLABS_API_KEY        — required for /chat audio output
  ELEVENLABS_VOICE_ID       — optional, defaults to Rachel (21m00Tcm4TlvDq8ikWAM)
  GOOGLE_MAPS_API_KEY       — optional, enables reverse geocoding for saved locations
  JWT_SECRET                — JWT signing secret (set a strong value in production)
"""
from __future__ import annotations

import base64
import math
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import auth
import database
import seeingeye_bridge

DEMO_DIR = Path(__file__).resolve().parent
STATIC_DIR = DEMO_DIR / "static"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(DEMO_DIR / ".env")

_ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
_ELEVENLABS_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
_GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
_AEYEZ_ENV = os.environ.get("AEYEZ_ENV") or os.environ.get("AEYES_ENV") or "dev"
_IS_PROD = _AEYEZ_ENV == "prod"
_K2_API_KEY = (
    os.environ.get("K2_API_KEY")
    or os.environ.get("AEYEZ_SUMMARIZER_API_KEY")
    or os.environ.get("AEYES_SUMMARIZER_API_KEY")
    or ""
).strip()
_K2_BASE_URL = (
    os.environ.get("K2_BASE_URL")
    or os.environ.get("AEYEZ_SUMMARIZER_BASE_URL")
    or os.environ.get("AEYES_SUMMARIZER_BASE_URL")
    or "https://api.k2think.ai/v1"
).rstrip("/")
_K2_SUMMARIZER_MODEL = (
    os.environ.get("AEYEZ_SUMMARIZER_MODEL")
    or os.environ.get("AEYES_SUMMARIZER_MODEL")
    or "k2thinkv2"
).strip()

_MATCH_RADIUS_METERS = 100


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await database.init_db()
    yield


app = FastAPI(title="Aeyez Demo", lifespan=lifespan)

# Rate limiter — enabled only in prod so tests + local dev aren't throttled.
# Token-keyed limiting is the upgrade path: replace `get_remote_address` with
# a `key_func` that prefers `Authorization` header so NAT'd users don't share
# a bucket. In-memory backend doesn't survive restarts and won't share across
# `--workers N` — needs Redis if scaled horizontally.
limiter = Limiter(key_func=get_remote_address, enabled=_IS_PROD)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if _IS_PROD:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # CSP — allow only what the app actually needs.
    # `unpkg.com` for Leaflet JS+CSS; `cartocdn.com` for OSM tiles; data: + blob:
    # for camera frames and ElevenLabs MP3 blobs. `'unsafe-inline'` on style-src
    # is needed because map.js injects an inline <style> for the position-marker
    # ping animation; follow-up cleanup is to move it to style.css.
    # No CORSMiddleware is added — the frontend is same-origin (served by
    # FastAPI's StaticFiles), and adding permissive CORS would be a regression.
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; "
        "img-src 'self' data: blob: https://*.basemaps.cartocdn.com; "
        "media-src 'self' blob:; "
        "connect-src 'self' https://api.elevenlabs.io; "
        "font-src 'self' https://fonts.gstatic.com;"
    )
    return resp


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

async def _elevenlabs_tts(text: str) -> Optional[str]:
    """Call ElevenLabs and return base64-encoded MP3, or None if key not set."""
    if not _ELEVENLABS_KEY:
        return None
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{_ELEVENLABS_VOICE}",
            headers={"xi-api-key": _ELEVENLABS_KEY, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=30.0,
        )
    if r.status_code != 200:
        return None
    return base64.b64encode(r.content).decode()


async def _summarize_with_k2(text: str) -> Optional[str]:
    """Return a short k2thinkv2 summary for the generated output, or None."""
    if not _K2_API_KEY or not text.strip():
        return None
    prompt = (
        "Summarize the following visual-assistant output for a blind user in one short sentence. "
        "Keep only the most actionable takeaway, under 20 words, and do not add preamble.\n\n"
        f"Output:\n{text.strip()}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{_K2_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {_K2_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _K2_SUMMARIZER_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You compress assistant outputs into crisp, practical summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 60,
                },
            )
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]["content"]
    except Exception:
        return None
    summary = (message or "").strip()
    return summary or None


# ── Geolocation helpers ───────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Return a human-readable address from Google Maps, or None on any failure."""
    if not _GMAPS_KEY:
        return None
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"latlng": f"{lat},{lon}", "key": _GMAPS_KEY,
                        "result_type": "street_address|premise"},
                timeout=5.0,
            )
        data = r.json()
        results = data.get("results", [])
        return results[0]["formatted_address"] if results else None
    except Exception:
        return None


async def _resolve_location(
    user_id: Optional[int],
    lat: Optional[float],
    lon: Optional[float],
) -> tuple[Optional[int], Optional[str]]:
    """Match lat/lon against user's saved locations. Returns (id, name) or (None, None)."""
    if not user_id or lat is None or lon is None:
        return None, None
    saved = await database.get_locations(user_id)
    for loc in saved:
        if _haversine_m(lat, lon, loc["lat"], loc["lon"]) <= _MATCH_RADIUS_METERS:
            return loc["id"], loc["name"]
    return None, None


# ── History context builder ───────────────────────────────────────────────────

def _build_context(history: list[dict]) -> str:
    """
    Format recent history into a prompt context string, grouped by location so
    the model can naturally answer location-scoped queries like "what's the
    state of my kitchen?". When the real model is integrated, inject this
    before the user's query so it has memory of past observations *and where
    they happened*. The model's output is always a visual description (audio-
    event triggering was removed), so "Saw" is the correct verb regardless of
    the original event field.
    """
    if not history:
        return ""

    grouped: dict[Optional[str], list[dict]] = {}
    for h in history[-12:]:
        grouped.setdefault(h.get("location_name"), []).append(h)

    lines = ["Recent observations (for context):"]
    # Render groups with named locations first, then untagged rows.
    keys = sorted(grouped.keys(), key=lambda k: (k is None, k or ""))
    for loc in keys:
        lines.append(f"  At {loc}:" if loc else "  Untagged location:")
        for h in grouped[loc]:
            ts = h["created_at"][:16].replace("T", " ")
            snippet = h["response"][:100].rstrip(".")
            if h["type"] == "chat":
                lines.append(f'    [{ts}] User asked: "{h["input_text"]}" → "{snippet}…"')
            elif h["type"] == "investigate":
                lines.append(f'    [{ts}] Saw: "{snippet}…"')
            elif h["type"] == "change":
                lines.append(f'    [{ts}] Scene changed: "{snippet}…"')
            elif h["type"] == "safe_mode":
                lines.append(f'    [{ts}] Safe mode → "{snippet}…"')
    return "\n".join(lines)


# ── Spatial-query stubs ───────────────────────────────────────────────────────
# In stub mode we do tiny pattern-matching so the demo can actually answer
# "where did I last see X" / "what's at the kitchen" without an LLM. The real
# model takes over once Spaz is wired in; these routes are best-effort
# fallbacks that produce something useful in the meantime.

# Matches: "where did I see/leave/put X", "where is X", "where are X",
# "where can I find X", "where's X". Captures whatever follows.
_WHERE_RE = re.compile(
    r"\bwhere(?:'s|\s+is|\s+are|\s+can\s+i\s+find|\s+did\s+i\s+(?:see|last\s+see|put|leave|find))\s+"
    r"(?:my\s+|the\s+|a\s+|an\s+|some\s+)?(.+?)\??\s*$",
    re.IGNORECASE,
)


def _match_object_query(text: str) -> Optional[str]:
    """If `text` looks like a 'where did I see X' question, return X. Else None."""
    m = _WHERE_RE.search(text.strip())
    if not m:
        return None
    obj = m.group(1).strip().lower().rstrip("?.,!")
    return obj or None


def _find_object_in_history(history: list[dict], object_name: str) -> Optional[dict]:
    """Most recent history row whose response mentions `object_name`. Else None."""
    needle = object_name.lower()
    for h in reversed(history):
        if needle in (h.get("response") or "").lower():
            return h
    return None


def _match_location_query(text: str, locations: list[dict]) -> Optional[dict]:
    """If text mentions a saved location name, return that location dict."""
    txt = text.lower()
    # Prefer the longest matching name so "main kitchen" beats "kitchen".
    best = None
    for loc in locations:
        name = loc["name"].lower()
        if name and name in txt and (best is None or len(name) > len(best["name"])):
            best = loc
    return best


def _summarize_at_location(history: list[dict], loc: dict) -> str:
    rows = [h for h in history if h.get("location_id") == loc["id"]]
    if not rows:
        return f"No observations recorded at {loc['name']} yet."
    last = rows[-1]
    snippet = (last.get("response") or "")[:120].rstrip(".")
    n = len(rows)
    return (
        f"At {loc['name']}, {n} observation{'s' if n != 1 else ''} on file. "
        f'Most recent: "{snippet}…"'
    )


def _relative_time(iso: str) -> str:
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return iso[:16].replace("T", " ")
    delta = datetime.now(timezone.utc) - when
    s = int(delta.total_seconds())
    if s < 60:    return f"{s}s ago"
    if s < 3600:  return f"{s // 60}m ago"
    if s < 86400: return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


# ── Auth endpoints ────────────────────────────────────────────────────────────

class AuthReq(BaseModel):
    username: str
    password: str


class AuthResp(BaseModel):
    token: str
    username: str
    display_name: str


@app.post("/auth/register", response_model=AuthResp)
@limiter.limit("5/minute")
async def register(request: Request, req: AuthReq) -> AuthResp:
    if len(req.username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Username ≥2 chars, password ≥4 chars.")
    if await database.get_user(req.username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    hashed = auth.hash_password(req.password)
    user_id = await database.create_user(req.username, hashed)
    return AuthResp(
        token=auth.create_token(user_id, req.username),
        username=req.username,
        display_name=req.username,
    )


@app.post("/auth/login", response_model=AuthResp)
@limiter.limit("5/minute")
async def login(request: Request, req: AuthReq) -> AuthResp:
    user = await database.get_user(req.username)
    if not user or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return AuthResp(
        token=auth.create_token(user["id"], req.username),
        username=user["username"],
        display_name=user.get("display_name") or user["username"],
    )


# ── History endpoint ──────────────────────────────────────────────────────────

@app.get("/history")
async def get_history(
    limit: int = 20,
    location_id: Optional[int] = None,
    current_user: dict = Depends(auth.require_user),
) -> list[dict]:
    return await database.get_history(current_user["id"], limit=limit, location_id=location_id)


# ── Profile endpoints ─────────────────────────────────────────────────────────

class ProfileResp(BaseModel):
    username: str
    display_name: str
    member_since: str
    history_count: int


class UpdateProfileReq(BaseModel):
    display_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


@app.get("/profile", response_model=ProfileResp)
async def get_profile(current_user: dict = Depends(auth.require_user)) -> ProfileResp:
    user = await database.get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    count = await database.count_history(current_user["id"])
    return ProfileResp(
        username=user["username"],
        display_name=user.get("display_name") or user["username"],
        member_since=user["created_at"][:10],
        history_count=count,
    )


@app.patch("/profile")
async def update_profile(
    req: UpdateProfileReq,
    current_user: dict = Depends(auth.require_user),
) -> dict:
    if req.display_name is not None:
        name = req.display_name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Display name cannot be empty.")
        await database.update_display_name(current_user["id"], name)
    if req.new_password:
        if not req.current_password:
            raise HTTPException(status_code=400, detail="Current password required.")
        user = await database.get_user_by_id(current_user["id"])
        if not user or not auth.verify_password(req.current_password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        if len(req.new_password) < 4:
            raise HTTPException(status_code=400, detail="New password must be ≥4 chars.")
        await database.update_password(current_user["id"], auth.hash_password(req.new_password))
    return {"ok": True}


# ── Location endpoints ────────────────────────────────────────────────────────

class CreateLocationReq(BaseModel):
    name: str
    lat: float
    lon: float


class UpdateLocationReq(BaseModel):
    name: str


@app.get("/locations")
async def list_locations(current_user: dict = Depends(auth.require_user)) -> list[dict]:
    return await database.get_locations(current_user["id"])


@app.post("/locations", status_code=201)
async def create_location(
    req: CreateLocationReq,
    current_user: dict = Depends(auth.require_user),
) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Location name cannot be empty.")
    if not (-90 <= req.lat <= 90) or not (-180 <= req.lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates.")
    address = await _reverse_geocode(req.lat, req.lon)
    loc_id = await database.add_location(current_user["id"], name, req.lat, req.lon, address)
    locs = await database.get_locations(current_user["id"])
    return next((l for l in locs if l["id"] == loc_id), {"id": loc_id, "name": name})


@app.patch("/locations/{location_id}")
async def rename_location(
    location_id: int,
    req: UpdateLocationReq,
    current_user: dict = Depends(auth.require_user),
) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Location name cannot be empty.")
    await database.update_location_name(location_id, current_user["id"], name)
    return {"ok": True}


@app.delete("/locations/{location_id}")
async def remove_location(
    location_id: int,
    current_user: dict = Depends(auth.require_user),
) -> dict:
    await database.delete_location(location_id, current_user["id"])
    return {"ok": True}


# ── Spaz prompt helpers ──────────────────────────────────────────────────

def _compose_question(
    instruction: str,
    *,
    context: str = "",
    user_text: Optional[str] = None,
) -> str:
    parts = [instruction.strip()]
    if context:
        parts.append(context.strip())
    if user_text:
        parts.append(f"User question: {user_text.strip()}")
    return "\n\n".join(part for part in parts if part)


def _describe_prompt(context: str) -> str:
    return _compose_question(
        (
            "You are describing the user's current surroundings for a blind user. "
            "Be concrete and concise. Mention the most important visible objects, people, "
            "text, layout cues, and immediate hazards or navigation-relevant details."
        ),
        context=context,
    )


def _change_prompt(context: str) -> str:
    return _compose_question(
        (
            "You are given an earlier frame followed by a later frame from the same scene. "
            "Describe what changed in plain language for a blind user. Focus on meaningful "
            "movement, new or missing objects, people entering or leaving, path obstructions, "
            "and any new hazards."
        ),
        context=context,
    )


def _chat_prompt(context: str, text: str) -> str:
    return _compose_question(
        (
            "Answer the user's spoken follow-up question using a short chronological visual window. "
            "When three frames are provided, interpret them as approximately 1.5 seconds before the "
            "question, the exact question moment, and 1.5 seconds after the question. Use the change "
            "across those frames to infer what just happened, what is happening now, and what the user "
            "should know next. Keep it direct, grounded in what is visible, and useful for a blind user."
        ),
        context=context,
        user_text=text,
    )


def _safe_mode_prompt(context: str, text: Optional[str], frame_count: int) -> str:
    instruction = (
        "Safe mode is active. Give immediate, practical mobility guidance for the next few seconds. "
        "Prioritize obstacles, drop-offs, crowding, open routes, doors, stairs, vehicles, and the safest "
        "next action. Use short, actionable sentences. Model egocentric action as: "
        "Action = hand pose + active object + contact target + temporal motion + scene context. "
        "Identify movable entities, predict their short-horizon motion paths, and warn when the user's "
        "route intersects a collision path, line-of-fire, pinch/crush/shear zone, falling-object zone, "
        "or a temporary obstruction that may clear if the user waits."
    )
    if frame_count > 1:
        instruction += (
            " The frames are chronological, so use before/after motion to infer what is moving, "
            "what will likely move next, and which route will become safest."
        )
    return _compose_question(instruction, context=context, user_text=text)


def _latest_frames_for_chat(
    image_b64: Optional[str],
    recent_frames: Optional[list[str]],
) -> list[str]:
    # Keep temporal order. Clients send recent_frames oldest -> newest; image_b64,
    # when present, is the explicit current frame and belongs at the end.
    frames = [frame for frame in (recent_frames or []) if frame]
    if image_b64:
        return [*frames, image_b64]
    return frames


async def _run_seeingeye_image(question: str, image_b64: str) -> str:
    result = await seeingeye_bridge.run_on_image(question, image_b64)
    return result.answer.strip()


async def _run_seeingeye_frames(question: str, frames_b64: list[str]) -> str:
    result = await seeingeye_bridge.run_on_frames(question, frames_b64)
    return result.answer.strip()


def _bridge_error_message(exc: Exception) -> str:
    if not seeingeye_bridge.STATUS.available:
        return seeingeye_bridge.STATUS.reason or "Spaz is not available on this machine yet."
    return (
        "Spaz is installed but could not answer this request right now. "
        f"{exc}"
    )


# ── Investigation endpoints ───────────────────────────────────────────────────

class InvestigateReq(BaseModel):
    event: str
    image_b64: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class InvestigateResp(BaseModel):
    event: str
    prompt: str
    response: str
    summary: Optional[str] = None
    elapsed_seconds: float
    success: bool


@app.post("/investigate", response_model=InvestigateResp)
@limiter.limit("30/minute")
async def investigate(
    request: Request,
    req: InvestigateReq,
    current_user: Optional[dict] = Depends(auth.optional_user),
) -> InvestigateResp:
    start = time.time()
    history = await database.get_history(current_user["id"]) if current_user else []
    context = _build_context(history)
    question = _describe_prompt(context)

    try:
        response = await _run_seeingeye_image(question, req.image_b64)
        success = True
    except Exception as exc:
        response = _bridge_error_message(exc)
        success = False
    summary = await _summarize_with_k2(response)

    if current_user:
        loc_id, loc_name = await _resolve_location(current_user["id"], req.lat, req.lon)
        await database.add_history(
            user_id=current_user["id"],
            type="investigate",
            response=response,
            input_text=req.event,
            event=req.event,
            lat=req.lat,
            lon=req.lon,
            location_id=loc_id,
            location_name=loc_name,
        )

    return InvestigateResp(
        event=req.event,
        prompt=question,
        response=response,
        summary=summary,
        elapsed_seconds=round(time.time() - start, 3),
        success=success,
    )


class ChangeReq(BaseModel):
    frame0_b64: str
    frame1_b64: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class ChangeResp(BaseModel):
    response: str
    summary: Optional[str] = None
    elapsed_seconds: float
    success: bool


@app.post("/analyze-change", response_model=ChangeResp)
@limiter.limit("30/minute")
async def analyze_change(
    request: Request,
    req: ChangeReq,
    current_user: Optional[dict] = Depends(auth.optional_user),
) -> ChangeResp:
    start = time.time()
    history = await database.get_history(current_user["id"]) if current_user else []
    context = _build_context(history)
    question = _change_prompt(context)

    try:
        response = await _run_seeingeye_frames(question, [req.frame0_b64, req.frame1_b64])
        success = True
    except Exception as exc:
        response = _bridge_error_message(exc)
        success = False
    summary = await _summarize_with_k2(response)

    if current_user:
        loc_id, loc_name = await _resolve_location(current_user["id"], req.lat, req.lon)
        await database.add_history(
            user_id=current_user["id"],
            type="change",
            response=response,
            lat=req.lat,
            lon=req.lon,
            location_id=loc_id,
            location_name=loc_name,
        )

    return ChangeResp(
        response=response,
        summary=summary,
        elapsed_seconds=round(time.time() - start, 3),
        success=success,
    )


# ── Chat endpoint ─────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    text: str
    image_b64: Optional[str] = None
    recent_frames: Optional[list[str]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class LocationRef(BaseModel):
    name: Optional[str] = None
    lat: float
    lon: float


class ChatResp(BaseModel):
    text: str
    response: str
    summary: Optional[str] = None
    audio_b64: Optional[str]
    success: bool
    referenced_location: Optional[LocationRef] = None


@app.post("/chat", response_model=ChatResp)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    req: ChatReq,
    current_user: Optional[dict] = Depends(auth.optional_user),
) -> ChatResp:
    history = await database.get_history(current_user["id"]) if current_user else []
    locations = await database.get_locations(current_user["id"]) if current_user else []
    context = _build_context(history)

    referenced: Optional[LocationRef] = None
    stub_response: str

    # Feature 1: "where did I (last) see X?" — find an object across history
    # and surface where it was last observed, plus a map pin for the client
    # to highlight.
    obj = _match_object_query(req.text)
    if obj and history:
        match = _find_object_in_history(history, obj)
        if match:
            loc_str = f" at {match['location_name']}" if match.get("location_name") else ""
            when = _relative_time(match["created_at"])
            snippet = (match.get("response") or "")[:120].rstrip(".")
            stub_response = (
                f'You last saw something matching "{obj}"{loc_str} {when}: "{snippet}…" '
                "(Stub response — real model coming soon.)"
            )
            if match.get("lat") is not None and match.get("lon") is not None:
                referenced = LocationRef(
                    name=match.get("location_name"),
                    lat=match["lat"],
                    lon=match["lon"],
                )
        else:
            stub_response = (
                f'I don\'t have any past observations matching "{obj}". '
                "(Stub: real model coming soon.)"
            )
    else:
        # Feature 3: "what's at <location>?" / "state of <location>?" — match
        # against the user's saved locations and return a per-location summary.
        loc_match = _match_location_query(req.text, locations) if locations else None
        if loc_match:
            stub_response = _summarize_at_location(history, loc_match)
            referenced = LocationRef(
                name=loc_match["name"],
                lat=loc_match["lat"],
                lon=loc_match["lon"],
            )
        else:
            frames = _latest_frames_for_chat(req.image_b64, req.recent_frames)
            question = _chat_prompt(context, req.text)
            try:
                if frames:
                    stub_response = await _run_seeingeye_frames(question, frames[:5])
                else:
                    stub_response = (
                        f'You asked: "{req.text}". '
                        "I do not have a recent camera frame yet, so I can only answer from saved history."
                    )
            except Exception as exc:
                stub_response = _bridge_error_message(exc)

    summary = await _summarize_with_k2(stub_response)
    audio_b64 = await _elevenlabs_tts(stub_response)

    if current_user:
        loc_id, loc_name = await _resolve_location(current_user["id"], req.lat, req.lon)
        await database.add_history(
            user_id=current_user["id"],
            type="chat",
            response=stub_response,
            input_text=req.text,
            lat=req.lat,
            lon=req.lon,
            location_id=loc_id,
            location_name=loc_name,
        )

    return ChatResp(
        text=req.text,
        response=stub_response,
        summary=summary,
        audio_b64=audio_b64,
        success=True,
        referenced_location=referenced,
    )


# ── Safe-mode endpoint ───────────────────────────────────────────────────────

class SafeModeReq(BaseModel):
    image_b64: Optional[str] = None
    recent_frames: Optional[list[str]] = None  # last N captured frames, newest first
    text: Optional[str] = None                 # voice phrase that triggered it, if any
    lat: Optional[float] = None
    lon: Optional[float] = None


class SafeModeResp(BaseModel):
    response: str
    summary: Optional[str] = None
    audio_b64: Optional[str]
    elapsed_seconds: float
    success: bool


@app.post("/safe-mode", response_model=SafeModeResp)
@limiter.limit("30/minute")
async def safe_mode(
    request: Request,
    req: SafeModeReq,
    current_user: Optional[dict] = Depends(auth.optional_user),
) -> SafeModeResp:
    start = time.time()
    history = await database.get_history(current_user["id"]) if current_user else []
    context = _build_context(history)

    n_frames = len(req.recent_frames) if req.recent_frames else (1 if req.image_b64 else 0)
    frames = _latest_frames_for_chat(req.image_b64, req.recent_frames)
    question = _safe_mode_prompt(context, req.text, len(frames))
    if frames:
        try:
            stub_response = await _run_seeingeye_frames(question, frames[:5])
            success = True
        except Exception as exc:
            stub_response = _bridge_error_message(exc)
            success = False
    else:
        stub_response = (
            "Safe mode is active, but I do not have any recent camera frame yet. "
            "Point the camera at the scene and try again."
        )
        success = False

    summary = await _summarize_with_k2(stub_response)
    audio_b64 = await _elevenlabs_tts(stub_response)

    if current_user:
        loc_id, loc_name = await _resolve_location(current_user["id"], req.lat, req.lon)
        await database.add_history(
            user_id=current_user["id"],
            type="safe_mode",
            response=stub_response,
            input_text=req.text,
            lat=req.lat,
            lon=req.lon,
            location_id=loc_id,
            location_name=loc_name,
        )

    return SafeModeResp(
        response=stub_response,
        summary=summary,
        audio_b64=audio_b64,
        elapsed_seconds=round(time.time() - start, 3),
        success=success,
    )


# ── Static + root ─────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "mode": "spaz" if seeingeye_bridge.STATUS.available else "stub",
        "spaz_available": seeingeye_bridge.STATUS.available,
        "spaz_path": str(seeingeye_bridge.STATUS.root) if seeingeye_bridge.STATUS.root else None,
        "spaz_reason": seeingeye_bridge.STATUS.reason,
        "elevenlabs": bool(_ELEVENLABS_KEY),
        "geocoding": bool(_GMAPS_KEY),
    }
