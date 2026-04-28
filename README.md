# Aeyez — Auto-Capture Visual Safety Narration

A tested web app that helps blind or low-vision users understand what is in
front of the camera, what changed, and what hazards may affect the safest next
move. The browser captures a still every 5 seconds (or on demand), sends it to
the bundled [Spaz](./Spaz/README.md) multi-agent pipeline for description, and
reads the answer aloud. **"What just changed?"** compares two frames sampled
~10 s apart, **"Safe mode"** asks for immediate route and hazard guidance, and
**"Hold to speak"** lets the user ask follow-up questions by voice. Logged-in
users get persistent per-account history that is fed back to the model as
memory.

Aeyez is inspired by the SeeingEye agentic visual-reasoning architecture. The
project extends that idea toward first-person safety assistance: route
understanding, obstruction tracking, long-video hazard summaries, and
actionable next-step guidance for real environments.

## Why no audio detection?

An earlier version listened for environmental sounds (glass breaking, alarms,
crashes) with on-device YAMNet and triggered investigations on those events.
That premise was wrong: blind users *aren't deaf*. They already hear the glass
break — what they need is for the model to tell them what they *can't* hear.
The audio-detection layer was removed; the value lives in the visual
description and in voice-driven Q&A, not in echoing what the user already
perceived.

> **Backend status — live Spaz integration.** `server.py` now looks for a
> local Spaz checkout next to this repo (or at `SPAZ_PATH`) and uses
> it for `/investigate`, `/analyze-change`, `/chat`, and `/safe-mode`. The
> browser remains the same: camera capture, rolling ClipBuffer, voice chat,
> auth, profile, and history all stay in place. ElevenLabs TTS for `/chat` is
> still real when `ELEVENLABS_API_KEY` is set; otherwise the browser falls back
> to `speechSynthesis`.

## Measured results

These numbers are from local Aeyez / Spaz runs, not from a paper claim or
mocked demo output.

| Evaluation | Result | Notes |
|---|---:|---|
| MMMU custom hardset | **220 / 250 correct = 88.0%** | Open-answer adjusted result recorded in [`benchmark_results/mmmutest.jsonl`](./benchmark_results/mmmutest.jsonl). |
| Long construction-video safety summaries | timestamped hazard tables + route guidance | Tested on masonry-site videos with 30 s windows, OpenCV keyframe selection, local risk rescans, and safe-mode escalation for high/uncertain windows. |

The MMMU hardset is curated toward the skills Aeyez needs in real scenes:
spatial relationships, route and obstruction reasoning, engineering/mechanics
visual verification, candidate-shape comparison, crop/zoom search, and
safety-relevant scene understanding.

## Architecture

```
Browser                                       Server (FastAPI)
─────────                                     ─────────────────
auth.js   ──login/register──→  /auth/login, /auth/register  ──→  SQLite (users)
                ↓                                                      │
            JWT in localStorage                                        │
                                                                       │
camera → ClipBuffer  ─10 s rolling window of JPEGs                     │
              │       (sampled at 1 fps, sample()-on-demand)           │
              ▼                                                        │
   5 s timer → 16×16 perceptual hash → diff vs last narration          │
                  │                                                    │
                  ├─ below CHANGE_THRESHOLD → stay silent               │
                  │                                                    │
                  └─ above → /analyze-change {prev, current}     ─→  Spaz
                              (first tick: /investigate _describe)─→  Spaz
   "What changed?" → /analyze-change {frame0, frame1}        ─────→  Spaz
   "Safe mode"     → /safe-mode {recent frames}              ─────→  Spaz safety analysis
   "Hold to speak" → /chat {text + recent frames}            ─────→  Spaz + ElevenLabs MP3
   video upload    → /daily-video-summary                    ─────→  OpenCV keyframes + Spaz
              │                                                        │
              │   each successful request appends to                   ▼
              │   ─────────────────────────────────────────→  SQLite (history)
              ▼
   speechSynthesis.speak(text)   OR   audio.play(audio_b64)
              │
              └── after each request: clip cache pruned to most recent frame only
              └── on success: window.refreshHistory() repopulates the panel
```

### Clip → frame sampling

The model takes images, not video, so the browser maintains a small rolling
window of recent frames and picks representative ones per request.
`static/clip_buffer.js` exposes a `ClipBuffer` class that owns the window and
the sampling policy:

| Strategy | Returns | Used by |
|---|---|---|
| `latest` | newest frame | on-demand current-scene descriptions |
| `edges` | `[oldest, newest]` | `/analyze-change` |
| `uniform` | N evenly spaced frames across the window | `/chat`, `/safe-mode`, and multi-frame context |

Adding a new strategy (e.g. perceptual-hash dedup, motion keyframes) is a local
edit to `ClipBuffer.sample()` — no orchestration changes needed.

### Cache hygiene

No frames are persisted to disk. Two distinct in-memory caches:

1. **Working buffer** (`ClipBuffer`). Fixed-size rolling window (max 11
   frames at 1 fps). Pruned to the single most recent frame after every
   `/investigate` and `/analyze-change`. Trigger-moment data URLs are
   released as soon as the request finishes. Worst-case footprint: ~100 KB.
2. **Recent captures sub-window**. A separate, time-bounded record of every
   frame the model actually saw, with timestamp + source label. Entries
   auto-evict after `CAPTURE_TTL_MS` (default 1 min) — both proactively as
   new captures arrive and via a periodic sweep every 10 s, so old entries
   vanish even if nothing else is happening. Worst-case at peak (auto-capture
   firing on every change for 1 min): ~12 frames × 50 KB ≈ 600 KB.

History is persisted server-side in `aeyez.db` only for authenticated users
(via `auth.optional_user`). Unauthenticated requests still work; they just
aren't recorded.

## Running it

### Prerequisites

```bash
pip install -r requirements.txt
```

This installs both the Aeyez web server dependencies and the bundled Spaz
runtime dependencies.

The Spaz runtime is bundled in this repo under `./Spaz`. The bridge looks in
this order:

```text
./Spaz
../Spaz
```

If your Spaz repo lives elsewhere, set:

```bash
export SPAZ_PATH=/absolute/path/to/Spaz
```

The SQLite DB and tables are created automatically on startup
(`database.init_db` runs in the FastAPI lifespan hook).

Optional environment variables:

| Var | Effect when set |
|---|---|
| `ELEVENLABS_API_KEY` | `/chat` returns a real MP3 voice reply (Rachel by default) instead of relying on browser TTS. |
| `ELEVENLABS_VOICE_ID` | Override the ElevenLabs voice ID (default: `21m00Tcm4TlvDq8ikWAM`, Rachel). |
| `JWT_SECRET` | Sign auth tokens with a real secret. **Required when `AEYEZ_ENV=prod`** — the dev default is `aeyez-dev-secret-change-in-prod`. |
| `AEYEZ_ENV` | `dev` (default) or `prod`. In `prod`, the server refuses to start unless `JWT_SECRET` is set to a non-default value. |
| `AEYEZ_DB_PATH` | Override the SQLite path. Defaults to `aeyez.db` next to `database.py`. Used by the Docker image to point at a mounted volume. |
| `AEYEZ_SKIP_SECRET_CHECK` | Set to `1` to bypass the JWT-secret check (test fixtures only — never in production). |
| `GOOGLE_MAPS_API_KEY` | Reverse-geocode lat/lon into addresses for saved locations. |

For live model responses, set `OPENAI_API_KEY` and any `SEEINGEYE_*` model
overrides in your shell, Docker `.env`, or `Spaz/.env`. Aeyez loads
`Spaz/.env` automatically if the environment is not already set. Use
`Spaz/.env.example` or the root `.env.example` as a starting point.

### Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest                                     # 19 tests, ~12 s on the current branch
pytest --cov=server --cov=auth --cov=database --cov-report=term
```

The published branch currently omits GitHub Actions so it can be pushed with
a normal token. Add a workflow later if you want hosted CI.

### LAN demo on a phone (HTTPS via mkcert)

Camera, mic, geolocation, and `SpeechRecognition` only work in a "secure
context" on phones. `localhost` is exempt; LAN IPs over plain HTTP are not.
The included launcher generates a per-LAN-IP cert via `mkcert` and starts
uvicorn with TLS.

**One-time setup on the laptop:**

```bash
choco install mkcert    # or: scoop install mkcert  (macOS: brew install mkcert)
mkcert -install         # installs a local root CA into the OS trust store
```

**One-time setup on each phone** — install the mkcert root CA so the phone
trusts certs you sign locally:

```bash
mkcert -CAROOT          # prints the dir holding rootCA.pem
```

- **iOS:** AirDrop `rootCA.pem` to the phone → tap to install profile →
  Settings → General → VPN & Device Management → install. Then Settings →
  General → About → Certificate Trust Settings → toggle on the "mkcert"
  root. (Apple buries this on purpose; it's a one-time pain.)
- **Android (Chrome):** copy `rootCA.pem` to the phone → Settings → Security
  → Encryption & credentials → Install a certificate → CA certificate →
  pick the file. Some Android versions only honor user-installed roots
  inside Chrome, not all apps — Chrome is what we need.

**Each demo:**

```bash
python scripts/dev_https.py
```

The script prints both the localhost URL and the LAN URL. Open the LAN URL
on the phone (must be on the same wifi). On first load the phone prompts
for camera + mic + location — accept all three. iOS users: Share → Add to
Home Screen, then launch from the icon for fullscreen.

**When wifi changes:** the LAN IP changes too, so the cert no longer
matches. Re-run with `--regen`:

```bash
python scripts/dev_https.py --regen
```

The cached IP lives in `certs/.issued-for`; the script auto-regenerates
when it sees a mismatch, so `--regen` is mostly only needed if the cert
gets out of sync with what mkcert thinks.

### Docker

```bash
JWT_SECRET=$(openssl rand -hex 32) docker compose up --build
```

The image runs in `AEYEZ_ENV=prod` mode and refuses to start without a real
`JWT_SECRET`. SQLite is stored in a named volume (`aeyez-data:/app/data`) so
data survives `down`/`up`.

### Public deploy for `aeyez.vision`

The checked-in [`docker-compose.yml`](/Users/pinkstrokes/Documents/New%20project/Aeyez/docker-compose.yml)
and [`Caddyfile`](/Users/pinkstrokes/Documents/New%20project/Aeyez/Caddyfile)
are ready for a public deploy with automatic HTTPS.

Before you start:

1. Point the DNS for `aeyez.vision` at your server.
2. Open inbound ports `80` and `443` on the server.
3. Fill a real `.env` with at least:

```bash
AEYEZ_DOMAIN=aeyez.vision
JWT_SECRET=replace_with_a_long_random_value
OPENAI_API_KEY=your_openai_api_key_here
SEEINGEYE_TRANSLATOR_MODEL=gpt-5.4-mini
SEEINGEYE_TRANSLATOR_ESCALATION_MODEL=gpt-5.4-mini
SEEINGEYE_REASONER_MODEL=gpt-5.4-mini
```

Then launch:

```bash
docker compose up -d --build
```

`Caddy` terminates TLS for `https://aeyez.vision` and proxies traffic to the
internal `aeyez:8000` service. The app container is no longer published
directly on host port `8000`; public traffic should go through `Caddy`.

### Accessibility

The target user is blind/low-vision, so accessibility is treated as a hard
requirement, not a nice-to-have. What's covered:
- Skip link first in tab order.
- ARIA tabs pattern with arrow-key roving tabindex on Camera/Map and
  Login/Register tabs.
- `aria-pressed` on every toggle button (auto-capture, voice, safe mode).
- `role="status"` + `aria-live` on the response and voice-response panels.
- Focus moves into overlays on open and back to the trigger on close
  (auth → username field, dropdown → first menuitem, profile → back button).
- Escape closes the user dropdown and returns focus to the menu button.
- `prefers-reduced-motion` short-circuits splash, panel-swap, and pulse
  animations.
- High-contrast focus ring on tabs whose `.active` background is the accent.
- Decorative `alt=""` on history thumbnails (the response text alongside is
  the semantic content) to avoid SR repetition.

Test with NVDA + Firefox on Windows for primary coverage; spot-check
VoiceOver on iOS Safari for the mobile install path.

Open <http://localhost:8000> in **Chrome** (camera + microphone permissions;
`SpeechRecognition` is Chromium-only). On first visit you'll be asked to
register; on subsequent visits with a stored token, auto-capture starts
automatically. The auto-capture loop is the default running state — use
**"Stop auto-capture"** to silence it. The manual buttons override on top:
**"Describe surroundings"** for a fresh on-demand description, **"What
changed?"** for an explicit diff, **"Hold to speak"** to ask the model a
question by voice. The **"Recent captures"** panel on the right shows
thumbnails of every frame the model actually saw, auto-evicting after 1 min.

`GET /health` returns whether Spaz was found locally, plus the resolved
path in `spaz_path`.

### vLLM (target — self-hosted Reasoner, ~16 GB VRAM)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

## Endpoints

| Method | Path | Auth | What it does |
|---|---|---|---|
| POST | `/auth/register` | none | Create a user, return a JWT. |
| POST | `/auth/login` | none | Verify credentials, return a JWT. |
| GET | `/profile` | required | Return user profile + history count. |
| PATCH | `/profile` | required | Update display name and/or password. |
| GET | `/history` | required | List the user's last N events. |
| POST | `/investigate` | optional | Single-frame Spaz description. Saves history when authenticated. |
| POST | `/analyze-change` | optional | Multi-frame Spaz comparison. Saves history when authenticated. |
| POST | `/chat` | optional | Voice question → text reply + optional ElevenLabs MP3. Saves history when authenticated. |
| POST | `/safe-mode` | optional | Safety-focused route, obstruction, and hazard guidance from recent frames. Saves history when authenticated. |
| POST | `/daily-video-summary` | optional | Uploaded long-video analysis with OpenCV keyframe selection, risk rescans, timestamped hazards, and final summary. Saves history when authenticated. |
| GET | `/health` | none | Liveness + Spaz availability/path. |

## Demo plan (stage-day)

1. Open the page → register a demo account in <10 s.
2. Click **"Start auto-capture"**. Stand in front of three pre-staged scenes
   for ~5 s each. The page speaks a description as soon as each one comes
   into frame.
3. Show **"What just changed?"** by quietly removing or adding an object
   during the demo, then pressing the button.
4. Hold **"Hold to speak"** and ask "what colour is the lamp on the desk?" —
   showcases voice Q&A with optional ElevenLabs voice if the key is set.
5. Open the corner avatar → **Profile** to show display-name + password
   editing and history count. Click into **History** to show that prior
   observations are remembered across sessions and (eventually) injected as
   context when the real model lands.
6. Privacy story: no frames leave the browser except for the single in-flight
   request, and the in-memory cache is pruned after each call. Only the
   text response is persisted.

## Knobs to tune

- `AUTO_CAPTURE_MS` (frontend, `app.js`): interval between automatic
  capture ticks. Default 5 s. Raise for less aggressive polling.
- `changeThreshold` (frontend, `app.js`, default 8 via `CHANGE_THRESHOLD_DEFAULT`):
  mean absolute brightness diff (0–255 scale) between consecutive 16×16 frame
  thumbnails required for the auto-capture loop to narrate. Below this, the
  tick stays silent and no request is sent. **The Calibration panel in the UI
  exposes this as a live slider** with a "Last diff" readout (green = above
  threshold, would have fired; gray = below, gated to silence). Static scenes
  typically diff at 1–3; an object moving usually diffs 10+. Tune in the room
  you'll demo in.
- `CAPTURE_TTL_MS` (frontend, `app.js`): how long thumbnails live in the
  "Recent captures" sub-window before being auto-evicted. Default 1 min.
- `CAPTURE_PRUNE_INTERVAL_MS` (frontend, `app.js`): how often the eviction
  sweep runs even when no new captures arrive. Default 10 s.
- `CLIP_WINDOW_MS` / `CLIP_FPS` (frontend, `app.js`): rolling-window length
  and sampling rate handed to `ClipBuffer`. Default 10 s @ 1 fps.
- `EVENT_PROMPTS` (backend, `server.py`): per-event investigative prompts
  used before frames are sent to Spaz.
- `_build_context(...)` (backend, `server.py`): formats recent history into
  a prompt prefix for live model calls, giving the model memory of past
  observations.
- `_EXPIRY_HOURS` (backend, `auth.py`): JWT lifetime, default 24 h.
- `/daily-video-summary` (backend, `server.py` + `video_analysis.py`): uses a
  tested two-stage video path: 5 baseline frames/min for the global timeline,
  30 s local windows, 64x64 grayscale OpenCV frame-difference scoring, local
  risk rescans, and denser frame sampling around high/uncertain risk windows.
- `SEEINGEYE_VIDEO_FRAME_INTERVAL_S` / `SEEINGEYE_VIDEO_MAX_FRAMES`: lower-level
  Spaz CLI/API video-frame sampling controls.

## File map

| Path | Purpose |
|---|---|
| `server.py` | FastAPI app — auth, history, profile, investigate, analyze-change, chat, health. Model endpoints call the bundled Spaz runtime through `seeingeye_bridge.py`. |
| `seeingeye_bridge.py` | Locates `./Spaz`, loads model environment, and adapts browser image frames into Spaz runtime calls. |
| `auth.py` | bcrypt password hashing + PyJWT token issuance/verification. `require_user` and `optional_user` dependencies. |
| `database.py` | aiosqlite wrappers for users (id, username, password_hash, display_name) and history (type, event, input, response, timestamp). Auto-creates schema on startup. |
| `Spaz/` | Bundled multimodal reasoning runtime and tests. Runtime secrets are intentionally not committed. |
| `benchmark_results/` | Measured local validation artifacts, including the 250-question MMMU custom hardset result at 88.0%. |
| `requirements.txt` | Aeyez web dependencies plus `Spaz/requirements.txt` for the live model bridge. |
| `static/index.html` | Auth overlay, two-column layout (camera left, app/profile right), corner user menu. |
| `static/app.js` | Camera init, ClipBuffer wiring, 5 s auto-capture loop, manual buttons, voice chat (`SpeechRecognition` → `/chat` → ElevenLabs audio), TTS, fetch + Bearer auth + cache pruning. |
| `static/auth.js` | Auth flows (login/register/logout), profile panel (display name, password change), history panel (server-driven via `/history`). Exposes `window.getAuthHeaders` and `window.refreshHistory`. |
| `static/clip_buffer.js` | `ClipBuffer` class — rolling 10 s frame window with named sampling strategies (`latest` / `edges` / `uniform`). The seam between "video coming in" and "images going to the model". |
| `static/style.css` | Dark theme, two-column layout, panels, animations. |

## Limits / honest caveats

- Chrome recommended. `SpeechRecognition` (used for voice chat) is
  Chromium-only; the rest works in any modern browser.
- Auto-capture narration is gated by a client-side perceptual hash
  (`CHANGE_THRESHOLD`). Static scenes are silent. The
  first tick of every auto-capture run still fires a baseline `/investigate`
  so the user gets an initial description; subsequent ticks route through
  `/analyze-change` for a "what changed" framing.
- Auth lives entirely in `localStorage` on the client; logging out clears
  it. The JWT `_SECRET` defaults to a dev value — set `JWT_SECRET` for any
  shared deployment.
- ElevenLabs is optional. Without the key, `/chat.audio_b64` is `null` and
  the browser falls back to `speechSynthesis`.
- Live model calls require a valid model API key. Without one, the web app
  still starts, but `/investigate`, `/analyze-change`, `/chat`, and
  `/safe-mode` cannot return real model answers.

### Fresh database

`init_db` (in `database.py`) creates the schema on startup if missing, so
deleting the file resets all accounts and history:

```bash
rm aeyez.db && uvicorn server:app --reload
```

`aeyez.db` is `.gitignore`d, so a fresh clone starts empty. The legacy
dev-account DB is still in git history from prior commits; if those
accounts are sensitive, scrub with `git filter-repo --invert-paths --path
aeyez.db` (force-push, coordinate with collaborators).

## Cleanup follow-ups

- **JWT secret rotation.** Tokens issued under the dev default
  (`aeyez-dev-secret-change-in-prod`) remain valid against the new secret
  only if the secret is unchanged. Rotate before the first prod deploy and
  expect existing sessions to be invalidated.
- **`aeyez.db` history.** The file is no longer tracked, but historical
  commits still contain it. See `git filter-repo` note above.
