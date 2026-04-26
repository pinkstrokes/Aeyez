// Aeyez demo — frontend orchestration.
//
// Pipelines:
//   (a) Auto-capture timer       -> /investigate (event=_describe), every AUTO_CAPTURE_MS
//   (b) Manual "describe"        -> /investigate (event=_describe)
//   (c) Manual "what changed"    -> /analyze-change (oldest+newest frame from rolling buffer)
//   (d) Hold-to-speak voice chat -> /chat (transcript + before/current/after frames)
//
// All capture paths funnel through `runInvestigation(event)` so status / TTS
// behave consistently. There is no audio classification: blind users already
// hear what's happening — the model's job is to describe what they cannot.
//
// History is server-driven when the user is logged in (auth.js owns the
// rendering via window.refreshHistory). When unauthenticated, requests still
// work; they just aren't persisted.
//
// No frames are persisted on disk. The ClipBuffer is a fixed-size rolling
// window in memory; it samples every 1.5s so safe mode can infer short-horizon
// motion paths while voice questions can compare
// the moment before the question, the question moment, and the moment after.

const AUTO_CAPTURE_MS = 5_000;
const QUESTION_CONTEXT_MS = 1_500;
const CLIP_WINDOW_MS = 10_000;
const CLIP_FRAME_INTERVAL_MS = QUESTION_CONTEXT_MS;
const SAFE_MODE_FRAME_COUNT = 6;

// Perceptual-hash threshold for the auto-capture change gate. Scale is the
// mean absolute brightness difference per pixel between two 16×16 thumbnails
// (0–255). ~3 corresponds to "no perceptible change", ~10+ corresponds to
// "an object moved or appeared". `let` (not `const`) so the calibration
// slider in the UI can adjust it live during stage rehearsal.
const CHANGE_THRESHOLD_DEFAULT = 8;
let changeThreshold = CHANGE_THRESHOLD_DEFAULT;


// "Recent captures" sub-window — keeps thumbnails of every frame the model
// actually saw, then auto-evicts once they cross CAPTURE_TTL_MS. The rolling
// ClipBuffer is still pruned aggressively after each request; this panel is
// a separate, time-bounded record for review/demo.
const CAPTURE_TTL_MS = 60_000;              // 1 minute
const CAPTURE_PRUNE_INTERVAL_MS = 10_000;   // re-sweep + re-render every 10 s

const $ = (id) => document.getElementById(id);
const statusEl = $("status-text");
const responseEl = $("response-text");
const summaryEl = $("summary-text");
const latencyChipEl = $("latency-chip");
const autoBtn = $("auto-btn");
const describeBtn = $("describe-btn");
const changeBtn = $("change-btn");
const safeModeBtn = $("safe-mode-btn");
const cameraEl = $("camera");
const captureCanvas = $("capture-canvas");
const lastFrameImg = $("last-frame");
const voiceBtn = $("voice-btn");
const voiceTranscriptEl = $("voice-transcript");
const voiceResponseEl = $("voice-response");
const voiceSummaryEl = $("voice-summary");
const videoSummaryForm = $("video-summary-form");
const videoUploadEl = $("video-upload");
const videoSummaryBtn = $("video-summary-btn");
const videoSummaryStatusEl = $("video-summary-status");
const videoSummaryOutputEl = $("video-summary-output");
const thresholdSliderEl = $("threshold-slider");
const thresholdValueEl = $("threshold-value");
const lastDiffEl = $("last-diff");

// In-memory cache of frames the model recently saw, keyed by capture time.
// auth.js's renderHistory looks up matching frames by timestamp via
// `window.getCaptureNear` and inlines the thumbnail. Frames auto-evict after
// CAPTURE_TTL_MS, after which the corresponding history rows render text-only.
const capturedFrames = []; // {ts, dataUrl, eventLabel} — newest first
const CAPTURE_MATCH_SLACK_MS = 8_000; // server clock vs client clock + roundtrip

const SAFE_MODE_PHRASES = [
  "tell me what to do", "guide me", "help me", "what should i do",
  "safe mode", "assist me", "i need help", "i need guidance",
  "安全模式", "帮我", "引导我", "我该怎么办", "告诉我怎么走",
];
const SPEECH_RECOGNITION_LANG = localStorage.getItem("aeyez_speech_lang") || "en-US";

let busy = false;
let clipBuffer = null;
let autoCaptureTimer = null;
let safeModeActive = false;
let lastNarrationHash = null;
let firstAutoTick = true;
let cameraStream = null;
let cameraRestartPromise = null;
let cameraWatchdogTimer = null;
let lastCameraFrameAt = 0;
let lastCameraCurrentTime = 0;
let cameraReadyOnce = false;
let lastFrameHideTimer = null;

// Reusable 16×16 canvas for the perceptual hash — avoids allocating per tick.
const HASH_CANVAS = document.createElement("canvas");
HASH_CANVAS.width = 16;
HASH_CANVAS.height = 16;

const CAMERA_STALL_MS = 8_000;
const CAMERA_WATCHDOG_MS = 4_000;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------- TTS ----------------
function speak(text, opts = {}) {
  if (!("speechSynthesis" in window)) return;
  if (opts.cancel !== false) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = opts.rate ?? 1.05;
  u.pitch = opts.pitch ?? 1.0;
  speechSynthesis.speak(u);
}

function setStatus(text, state) {
  statusEl.textContent = text;
  if (state) statusEl.dataset.state = state;
  else delete statusEl.dataset.state;
}

function showResponse(text) {
  responseEl.textContent = text;
  responseEl.hidden = !text;
}

function showSummary(text, target = summaryEl) {
  if (!target) return;
  target.textContent = text || "";
  target.hidden = !text;
}

function showLatency(seconds) {
  if (typeof seconds !== "number" || !isFinite(seconds)) {
    latencyChipEl.hidden = true;
    return;
  }
  latencyChipEl.textContent = `Responded in ${seconds.toFixed(2)}s`;
  latencyChipEl.hidden = false;
}

function formatVideoSummary(data) {
  const dangers = data.dangers?.length
    ? data.dangers.map((item) => `- ${item}`).join("\n")
    : "- No clear dangers detected.";
  const events = data.events?.length
    ? data.events
        .slice(0, 12)
        .map((event) => (
          `- ${event.start}-${event.end} peak ${event.peak} score ${event.peak_score}`
          + (event.danger_candidate ? " danger-candidate" : "")
        ))
        .join("\n")
    : "- No major motion events detected.";
  return [
    data.summary,
    "",
    "Dangers",
    dangers,
    "",
    "Selector",
    `- Duration: ${(data.duration_seconds / 60).toFixed(1)} min`,
    `- Retained frames: ${data.retained_frame_count}`,
    `- Model frames: ${data.model_frame_count}`,
    `- Baseline: ${data.baseline_frames_per_minute}/min`,
    "",
    "Detected events",
    events,
  ].join("\n");
}

// ---------------- Captured-frame cache ----------------
function recordCapture(_eventKey, dataUrl) {
  if (!dataUrl) return;
  capturedFrames.unshift({ ts: Date.now(), dataUrl });
  pruneCapturedFrames();
}

function pruneCapturedFrames() {
  const cutoff = Date.now() - CAPTURE_TTL_MS;
  // Newest-first array → old entries cluster at the tail.
  while (capturedFrames.length && capturedFrames[capturedFrames.length - 1].ts < cutoff) {
    capturedFrames.pop();
  }
}

// auth.js calls this from renderHistory to inline a thumbnail next to each
// matching history entry. Server timestamps lag client-capture timestamps by
// roundtrip + clock skew; CAPTURE_MATCH_SLACK_MS is the tolerance.
window.getCaptureNear = function (isoTimestamp) {
  if (!isoTimestamp) return null;
  const target = new Date(isoTimestamp).getTime();
  if (!Number.isFinite(target)) return null;
  let best = null;
  let bestDelta = CAPTURE_MATCH_SLACK_MS;
  for (const f of capturedFrames) {
    const d = Math.abs(f.ts - target);
    if (d <= bestDelta) {
      best = f;
      bestDelta = d;
    }
  }
  return best ? best.dataUrl : null;
};

// Periodic sweep so old frames vanish even when no new captures arrive. When
// any frame is evicted, ask auth.js to re-render history so its thumbnail
// disappears too — keeps the inline thumbnails honest about the TTL.
setInterval(() => {
  if (capturedFrames.length === 0) return;
  const before = capturedFrames.length;
  pruneCapturedFrames();
  if (capturedFrames.length !== before) window.refreshHistory?.();
}, CAPTURE_PRUNE_INTERVAL_MS);

// Privacy: clear cached frames on logout so a different account on the same
// browser tab doesn't see the previous user's thumbnails.
document.getElementById("logout-btn")?.addEventListener("click", () => {
  capturedFrames.length = 0;
});

// ---------------- Calibration ----------------
function updateLastDiffReadout(diff) {
  if (!lastDiffEl) return;
  lastDiffEl.textContent = diff.toFixed(1);
  lastDiffEl.dataset.state = diff >= changeThreshold ? "above" : "below";
}

function updateThresholdReadout() {
  if (thresholdValueEl) thresholdValueEl.textContent = changeThreshold.toFixed(1);
  // Re-color the last-diff readout against the new threshold without waiting
  // for the next tick, so the slider feels responsive.
  if (lastDiffEl && lastDiffEl.textContent && lastDiffEl.textContent !== "—") {
    const last = parseFloat(lastDiffEl.textContent);
    if (!Number.isNaN(last)) {
      lastDiffEl.dataset.state = last >= changeThreshold ? "above" : "below";
    }
  }
}

function initCalibration() {
  if (!thresholdSliderEl) return;
  thresholdSliderEl.value = String(changeThreshold);
  updateThresholdReadout();
  thresholdSliderEl.addEventListener("input", () => {
    const v = parseFloat(thresholdSliderEl.value);
    if (!Number.isNaN(v)) {
      changeThreshold = v;
      updateThresholdReadout();
    }
  });
}

// ---------------- Camera + frame buffer ----------------
function markCameraAlive() {
  lastCameraFrameAt = Date.now();
  if (Number.isFinite(cameraEl.currentTime)) {
    lastCameraCurrentTime = cameraEl.currentTime;
  }
}

function stopCameraStream() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
  }
  cameraStream = null;
  cameraEl.srcObject = null;
}

function ensureClipBuffer() {
  if (clipBuffer) return;
  clipBuffer = new window.ClipBuffer({
    windowMs: CLIP_WINDOW_MS,
    intervalMs: CLIP_FRAME_INTERVAL_MS,
    captureFn: () => (cameraEl.readyState < 2 ? null : captureFrameDataUrl()),
  });
}

async function startCameraStream() {
  // Prefer rear camera on mobile (the one pointed at the user's surroundings);
  // fall back if the device only has a front camera.
  const ideal = {
    video: { width: 640, height: 480, facingMode: { ideal: "environment" } },
    audio: false,
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(ideal);
  } catch {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  }
  stopCameraStream();
  cameraStream = stream;
  cameraEl.srcObject = stream;
  await new Promise((r) => (cameraEl.onloadedmetadata = r));
  await cameraEl.play().catch(() => {});
  captureCanvas.width = cameraEl.videoWidth;
  captureCanvas.height = cameraEl.videoHeight;
  markCameraAlive();
  cameraReadyOnce = true;

  for (const track of stream.getVideoTracks()) {
    track.addEventListener("ended", () => {
      if (!document.hidden) scheduleCameraRestart("track ended");
    });
    track.addEventListener("mute", () => {
      if (!document.hidden) scheduleCameraRestart("track muted");
    });
  }

  ensureClipBuffer();
  clipBuffer.start();
}

async function scheduleCameraRestart(reason = "camera stalled") {
  if (!cameraReadyOnce) return;
  if (cameraRestartPromise) return cameraRestartPromise;
  cameraRestartPromise = (async () => {
    setStatus("Reconnecting camera…", "investigating");
    if (clipBuffer) clipBuffer.stop();
    try {
      await startCameraStream();
      setStatus(
        safeModeActive ? "Safe mode." : autoCaptureTimer ? "Auto-capturing." : "Ready.",
        safeModeActive || autoCaptureTimer ? "listening" : null,
      );
    } catch (e) {
      console.error(`Camera restart failed: ${reason}`, e);
      setStatus("Camera disconnected. Reload and grant camera access.", "error");
      throw e;
    } finally {
      cameraRestartPromise = null;
    }
  })();
  return cameraRestartPromise;
}

function cameraLooksStalled() {
  if (!cameraReadyOnce) return false;
  if (document.hidden) return false;
  const track = cameraStream?.getVideoTracks?.()[0];
  if (!track || track.readyState !== "live" || track.muted) return true;
  if (cameraEl.readyState < 2) return true;

  const now = Date.now();
  const currentTime = Number.isFinite(cameraEl.currentTime) ? cameraEl.currentTime : 0;
  if (currentTime > lastCameraCurrentTime + 0.05) {
    markCameraAlive();
    return false;
  }
  return now - lastCameraFrameAt > CAMERA_STALL_MS;
}

function startCameraWatchdog() {
  if (cameraWatchdogTimer !== null) return;
  cameraWatchdogTimer = setInterval(() => {
    if (!cameraRestartPromise && cameraLooksStalled()) {
      scheduleCameraRestart("watchdog");
    }
  }, CAMERA_WATCHDOG_MS);
}

function bindCameraLifecycle() {
  cameraEl.addEventListener("playing", markCameraAlive);
  cameraEl.addEventListener("loadeddata", markCameraAlive);
  cameraEl.addEventListener("timeupdate", markCameraAlive);
  cameraEl.addEventListener("stalled", () => {
    if (!document.hidden) scheduleCameraRestart("video stalled");
  });
  cameraEl.addEventListener("emptied", () => {
    if (!document.hidden) scheduleCameraRestart("video emptied");
  });

  document.addEventListener("visibilitychange", () => {
    if (!cameraReadyOnce) return;
    if (document.hidden) {
      clipBuffer?.stop();
      return;
    }
    clipBuffer?.start();
    if (cameraLooksStalled()) {
      scheduleCameraRestart("tab became visible");
    } else {
      markCameraAlive();
    }
  });

  window.addEventListener("pagehide", () => {
    if (!cameraReadyOnce) return;
    clipBuffer?.stop();
  });

  window.addEventListener("pageshow", () => {
    if (!cameraReadyOnce) return;
    clipBuffer?.start();
    if (cameraLooksStalled()) {
      scheduleCameraRestart("page restored");
    }
  });

  window.addEventListener("beforeunload", () => {
    clipBuffer?.stop();
    stopCameraStream();
  });
}

function captureFrameDataUrl() {
  if (cameraEl.readyState < 2 || captureCanvas.width === 0 || captureCanvas.height === 0) {
    return null;
  }
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(cameraEl, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.85);
}

function showLastFrame(dataUrl) {
  lastFrameImg.src = dataUrl;
  lastFrameImg.hidden = false;
  if (lastFrameHideTimer !== null) clearTimeout(lastFrameHideTimer);
  lastFrameHideTimer = setTimeout(() => {
    lastFrameImg.hidden = true;
    lastFrameHideTimer = null;
  }, 1800);
}

// Drop everything except the most recent frame, so the rolling window doesn't
// hold sent frames longer than necessary. /analyze-change still has the latest
// to diff against on its next call.
function pruneClipCache() {
  if (!clipBuffer) return;
  const last = clipBuffer.latest();
  clipBuffer.frames.length = 0;
  if (last) clipBuffer.frames.push(last);
}

function buildQuestionFrames(currentFrame, afterFrame) {
  const beforeFrame = clipBuffer?.closestTo(
    currentFrame.ts - QUESTION_CONTEXT_MS,
    QUESTION_CONTEXT_MS,
  );
  const frames = [];
  if (beforeFrame) frames.push(beforeFrame);
  frames.push(currentFrame);
  if (afterFrame) frames.push(afterFrame);
  return frames
    .filter(Boolean)
    .filter((frame, idx, arr) => arr.findIndex((item) => item.ts === frame.ts) === idx);
}

// 16×16 grayscale thumbnail of the live camera frame. Returns a Uint8Array
// of 256 brightness values, or null if the camera isn't ready.
function frameHash() {
  if (cameraEl.readyState < 2) return null;
  const ctx = HASH_CANVAS.getContext("2d");
  ctx.drawImage(cameraEl, 0, 0, 16, 16);
  const data = ctx.getImageData(0, 0, 16, 16).data;
  const out = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    out[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
  }
  return out;
}

// Mean absolute difference between two 256-byte hashes. Range 0–255.
function hashDiff(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// ---------------- Geolocation ----------------
async function getCurrentCoords() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) { resolve(null); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      ()    => resolve(null),
      { timeout: 3000, maximumAge: 30_000 },
    );
  });
}

// ---------------- Trigger handlers ----------------
async function runInvestigation(eventKey, { showFrame = true } = {}) {
  if (busy) return;
  if (cameraEl.readyState < 2) {
    setStatus("Camera not ready", "error");
    return;
  }
  busy = true;
  showResponse("");
  showSummary("");
  showLatency(null);

  setStatus("Investigating…", "investigating");
  let [triggerFrame, coords] = await Promise.all([
    Promise.resolve(clipBuffer.captureNow()),
    getCurrentCoords(),
  ]);
  if (!triggerFrame) {
    setStatus("Camera not ready", "error");
    busy = false;
    return;
  }
  if (showFrame) showLastFrame(triggerFrame.dataUrl);

  const tStart = performance.now();
  try {
    const resp = await fetch("/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...window.getAuthHeaders?.() },
      body: JSON.stringify({ event: eventKey, image_b64: triggerFrame.dataUrl, ...(coords || {}) }),
    });
    const data = await resp.json();
    const wallElapsed = (performance.now() - tStart) / 1000;
    const elapsed = typeof data.elapsed_seconds === "number" ? data.elapsed_seconds : wallElapsed;
    if (!data.success) {
      setStatus("Something went wrong.", "error");
      showResponse(data.response || "");
      speak("Something went wrong with the investigation.");
    } else {
      setStatus(safeModeActive ? "Safe mode." : autoCaptureTimer ? "Auto-capturing." : "Ready.",
                safeModeActive || autoCaptureTimer ? "listening" : null);
      showResponse(data.response);
      showSummary(data.summary || "");
      showLatency(elapsed);
      recordCapture(eventKey, triggerFrame.dataUrl);
      speak(data.response);
      window.refreshHistory?.();
    }
  } catch (e) {
    setStatus("Network error.", "error");
    speak("Network error.");
    console.error(e);
  } finally {
    triggerFrame = null;
    pruneClipCache();
    busy = false;
  }
}

async function runChangeAnalysis({ showFrame = true } = {}) {
  if (busy) return;
  if (!clipBuffer || clipBuffer.length() < 2) {
    speak("Not enough video history yet. Wait a few more seconds.");
    return;
  }
  busy = true;
  setStatus("Comparing the last few seconds…", "investigating");
  speak("Comparing the scene.");
  showResponse("");
  showSummary("");
  showLatency(null);

  let [oldest, newest] = clipBuffer.sample("edges");
  if (showFrame) showLastFrame(newest.dataUrl);
  const coords = await getCurrentCoords();

  const tStart = performance.now();
  try {
    const resp = await fetch("/analyze-change", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...window.getAuthHeaders?.() },
      body: JSON.stringify({ frame0_b64: oldest.dataUrl, frame1_b64: newest.dataUrl, ...(coords || {}) }),
    });
    const data = await resp.json();
    const wallElapsed = (performance.now() - tStart) / 1000;
    const elapsed = typeof data.elapsed_seconds === "number" ? data.elapsed_seconds : wallElapsed;
    if (!data.success) {
      setStatus("Something went wrong.", "error");
      showResponse(data.response || "");
      speak("Could not compare the scene.");
    } else {
      setStatus(safeModeActive ? "Safe mode." : autoCaptureTimer ? "Auto-capturing." : "Ready.",
                safeModeActive || autoCaptureTimer ? "listening" : null);
      showResponse(data.response);
      showSummary(data.summary || "");
      showLatency(elapsed);
      recordCapture("_change", newest.dataUrl);
      speak(data.response);
      window.refreshHistory?.();
    }
  } catch (e) {
    setStatus("Network error.", "error");
    speak("Network error.");
    console.error(e);
  } finally {
    oldest = null;
    newest = null;
    pruneClipCache();
    busy = false;
  }
}

// ---------------- Auto-capture loop ----------------
//
function autoCaptureTick() {
  if (busy) return;
  const h = frameHash();
  if (!h) return;

  if (firstAutoTick || lastNarrationHash === null) {
    firstAutoTick = false;
    lastNarrationHash = h;
    runInvestigation("_describe", { showFrame: false });
    return;
  }

  const diff = hashDiff(lastNarrationHash, h);
  updateLastDiffReadout(diff);
  if (diff < changeThreshold) {
    return;
  }

  lastNarrationHash = h;
  if (clipBuffer && clipBuffer.length() >= 2) {
    runChangeAnalysis({ showFrame: false });
  } else {
    runInvestigation("_describe", { showFrame: false });
  }
}

function startAutoCapture({ silent = false } = {}) {
  if (autoCaptureTimer !== null) return;
  firstAutoTick = true;
  lastNarrationHash = null;
  autoCaptureTimer = setInterval(autoCaptureTick, AUTO_CAPTURE_MS);
  setStatus("Auto-capturing.", "listening");
  autoBtn.textContent = "Stop auto-capture";
  autoBtn.dataset.state = "running";
  autoBtn.setAttribute("aria-pressed", "true");
  if (!silent) speak("Auto capture started.");
}

function stopAutoCapture({ silent = false } = {}) {
  if (autoCaptureTimer === null) return;
  clearInterval(autoCaptureTimer);
  autoCaptureTimer = null;
  if (!silent) setStatus("Ready.");
  autoBtn.textContent = "Start auto-capture";
  delete autoBtn.dataset.state;
  autoBtn.setAttribute("aria-pressed", "false");
  if (!silent) speak("Auto capture stopped.");
}

// ---------------- Safe mode ----------------
function isSafeModePhrase(text) {
  const lower = text.toLowerCase();
  return SAFE_MODE_PHRASES.some((p) => lower.includes(p));
}

function startSafeMode() {
  if (safeModeActive) return;
  safeModeActive = true;
  document.body.classList.add("safe-mode-active");
  setStatus("Safe mode.", "listening");
  if (safeModeBtn) {
    safeModeBtn.textContent = "Stop safe mode";
    safeModeBtn.dataset.state = "safe";
    safeModeBtn.setAttribute("aria-pressed", "true");
  }
}

function stopSafeMode() {
  if (!safeModeActive) return;
  safeModeActive = false;
  document.body.classList.remove("safe-mode-active");
  setStatus(autoCaptureTimer ? "Auto-capturing." : "Ready.", autoCaptureTimer ? "listening" : null);
  if (safeModeBtn) {
    safeModeBtn.textContent = "Safe mode";
    delete safeModeBtn.dataset.state;
    safeModeBtn.setAttribute("aria-pressed", "false");
  }
}

async function runSafeMode(triggerText) {
  if (busy) return;
  busy = true;
  showResponse("");
  showSummary("");

  const [currentFrame, coords] = await Promise.all([
    Promise.resolve(clipBuffer?.captureNow() || null),
    getCurrentCoords(),
  ]);
  const recentFrames = clipBuffer
    ? clipBuffer
        .sample("uniform", SAFE_MODE_FRAME_COUNT)
        .filter((frame) => !currentFrame || frame.ts !== currentFrame.ts)
        .map((frame) => frame.dataUrl)
    : [];

  try {
    const resp = await fetch("/safe-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...window.getAuthHeaders?.() },
      body: JSON.stringify({
        image_b64: currentFrame?.dataUrl || null,
        recent_frames: recentFrames,
        text: triggerText || null,
        ...(coords || {}),
      }),
    });
    const data = await resp.json();
    voiceResponseEl.textContent = data.response;
    voiceResponseEl.hidden = false;
    showSummary(data.summary || "", voiceSummaryEl);
    showResponse(data.response);
    await speakReply(data.response, data.audio_b64);
    window.refreshHistory?.();
  } catch (e) {
    setStatus("Network error.", "error");
    console.error(e);
  } finally {
    if (safeModeBtn && safeModeActive) {
      safeModeBtn.textContent = "Stop safe mode";
      safeModeBtn.dataset.state = "safe";
    }
    setStatus(safeModeActive ? "Safe mode." : autoCaptureTimer ? "Auto-capturing." : "Ready.",
              safeModeActive || autoCaptureTimer ? "listening" : null);
    busy = false;
  }
}

// ---------------- Voice chat ----------------
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let voiceBusy = false;
let recognitionActive = false;
let recognitionStopRequested = false;
let recognitionStartedAt = 0;
let micPermissionReady = false;
let replyAudio = null;
let voiceRecorder = null;
let voiceRecorderStream = null;
let voiceChunks = [];
let voiceRecordingStartedAt = 0;

function initRecognition() {
  if (!SpeechRecognition) return null;
  const r = new SpeechRecognition();
  r.continuous = true;
  r.interimResults = true;
  r.maxAlternatives = 1;
  r.lang = SPEECH_RECOGNITION_LANG;
  return r;
}

async function playAudioB64(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/mpeg" });
  const url = URL.createObjectURL(blob);
  stopReplyAudio();
  const audio = new Audio(url);
  replyAudio = audio;
  const cleanup = () => {
    URL.revokeObjectURL(url);
    if (replyAudio === audio) replyAudio = null;
  };
  audio.onended = cleanup;
  audio.onerror = cleanup;
  try {
    await audio.play();
  } catch (e) {
    cleanup();
    throw e;
  }
}

async function speakReply(text, audioB64) {
  if (!audioB64) {
    speak(text, { cancel: false });
    return;
  }
  try {
    await playAudioB64(audioB64);
  } catch (e) {
    console.warn("ElevenLabs audio playback failed; falling back to browser TTS.", e);
    speak(text, { cancel: false });
  }
}

function stopReplyAudio() {
  if (!replyAudio) return;
  try {
    replyAudio.pause();
    replyAudio.currentTime = 0;
  } catch {}
  replyAudio = null;
}

async function ensureMicPermission() {
  if (micPermissionReady) return true;
  if (!navigator.mediaDevices?.getUserMedia) return true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    for (const track of stream.getTracks()) track.stop();
    micPermissionReady = true;
    return true;
  } catch (e) {
    console.error("Microphone permission failed", e);
    voiceTranscriptEl.textContent = "Mic permission denied. Allow microphone access, then try again.";
    voiceTranscriptEl.hidden = false;
    return false;
  }
}

function cleanupVoiceRecorder() {
  if (voiceRecorderStream) {
    for (const track of voiceRecorderStream.getTracks()) track.stop();
  }
  voiceRecorderStream = null;
  voiceRecorder = null;
  voiceChunks = [];
}

async function transcribeVoiceBlob(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "speech.webm");
  const resp = await fetch("/transcribe", {
    method: "POST",
    headers: { ...window.getAuthHeaders?.() },
    body: formData,
  });
  const data = await resp.json();
  if (!resp.ok || !data.success || !data.text) {
    throw new Error(data.detail || "No speech caught.");
  }
  return data.text.trim();
}

async function startVoiceCapture() {
  if (voiceBusy || voiceRecorder?.state === "recording") return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    voiceTranscriptEl.textContent = "Audio recording is not supported in this browser.";
    voiceTranscriptEl.hidden = false;
    return;
  }
  stopReplyAudio();
  window.speechSynthesis?.cancel?.();
  try {
    voiceRecorderStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    console.error("Microphone permission failed", e);
    voiceTranscriptEl.textContent = "Mic permission denied. Allow microphone access, then try again.";
    voiceTranscriptEl.hidden = false;
    return;
  }

  voiceChunks = [];
  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";
  voiceRecorder = new MediaRecorder(voiceRecorderStream, { mimeType });
  voiceRecorder.ondataavailable = (ev) => {
    if (ev.data?.size) voiceChunks.push(ev.data);
  };
  voiceRecorder.onstop = async () => {
    const chunks = voiceChunks.slice();
    cleanupVoiceRecorder();
    resetVoiceButton();
    if (Date.now() - voiceRecordingStartedAt < 450 || chunks.length === 0) {
      voiceTranscriptEl.textContent = "No speech caught. Hold the button, speak clearly, then release.";
      voiceTranscriptEl.hidden = false;
      return;
    }
    voiceBusy = true;
    voiceTranscriptEl.textContent = "Transcribing...";
    voiceTranscriptEl.hidden = false;
    try {
      const blob = new Blob(chunks, { type: "audio/webm" });
      const text = await transcribeVoiceBlob(blob);
      voiceTranscriptEl.textContent = `You: ${text}`;
      voiceTranscriptEl.hidden = false;
      if (isSafeModePhrase(text)) {
        startSafeMode();
        await runSafeMode(text);
      } else if (safeModeActive) {
        await runSafeMode(text);
      } else {
        await runChat(text);
      }
    } catch (e) {
      console.error(e);
      voiceTranscriptEl.textContent = "No speech caught. Hold the button, speak clearly, then release.";
      voiceTranscriptEl.hidden = false;
    } finally {
      voiceBusy = false;
    }
  };
  voiceRecorder.start();
  voiceRecordingStartedAt = Date.now();
  voiceBtn.classList.add("recording");
  voiceBtn.textContent = "Listening...";
  voiceBtn.setAttribute("aria-pressed", "true");
  voiceTranscriptEl.textContent = "Listening...";
  voiceTranscriptEl.hidden = false;
}

function stopVoiceCapture() {
  if (!voiceRecorder || voiceRecorder.state !== "recording") return;
  const elapsed = Date.now() - voiceRecordingStartedAt;
  const stopNow = () => {
    if (voiceRecorder?.state === "recording") voiceRecorder.stop();
  };
  if (elapsed < 700) setTimeout(stopNow, 700 - elapsed);
  else stopNow();
}

async function runChat(text) {
  if (busy) {
    voiceResponseEl.textContent = "Still finishing the last visual check.";
    voiceResponseEl.hidden = false;
    return;
  }
  voiceBusy = true;
  busy = true;
  voiceTranscriptEl.textContent = `You: ${text}`;
  voiceTranscriptEl.hidden = false;
  voiceResponseEl.textContent = "…";
  voiceResponseEl.hidden = false;
  showSummary("", voiceSummaryEl);

  const [coords, currentFrame] = await Promise.all([
    getCurrentCoords(),
    Promise.resolve(clipBuffer?.captureNow() || null),
  ]);
  if (!currentFrame) {
    voiceResponseEl.textContent = "Camera not ready.";
    busy = false;
    voiceBusy = false;
    return;
  }
  voiceResponseEl.textContent = "Watching the next moment…";
  await wait(QUESTION_CONTEXT_MS);
  const afterFrame = clipBuffer?.captureNow() || null;
  const questionFrames = buildQuestionFrames(currentFrame, afterFrame);
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...window.getAuthHeaders?.() },
      body: JSON.stringify({
        text,
        image_b64: null,
        recent_frames: questionFrames.map((frame) => frame.dataUrl),
        ...(coords || {}),
      }),
    });
    const data = await resp.json();
    voiceResponseEl.textContent = data.response;
    showSummary(data.summary || "", voiceSummaryEl);
    await speakReply(data.response, data.audio_b64);
    // Spatial-memory: when /chat answers a "where did I see X" or
    // "what's at <location>" query, the response carries a referenced
    // location — switch to the map tab and pulse a pin there.
    if (data.referenced_location) {
      const r = data.referenced_location;
      window.flashLocation?.(r.lat, r.lon, r.name);
    }
    window.refreshHistory?.();
  } catch (e) {
    voiceResponseEl.textContent = "Network error.";
    showSummary("", voiceSummaryEl);
    console.error(e);
  } finally {
    busy = false;
    voiceBusy = false;
  }
}

async function runVideoSummary(event) {
  event.preventDefault();
  const file = videoUploadEl?.files?.[0];
  if (!file) {
    videoSummaryStatusEl.textContent = "Choose a video first.";
    videoSummaryStatusEl.hidden = false;
    return;
  }

  videoSummaryBtn.disabled = true;
  videoSummaryOutputEl.hidden = true;
  videoSummaryOutputEl.textContent = "";
  videoSummaryStatusEl.textContent = "Selecting key frames and analyzing video…";
  videoSummaryStatusEl.hidden = false;

  const formData = new FormData();
  formData.append("video", file);
  const startedAt = performance.now();
  try {
    const resp = await fetch("/daily-video-summary", {
      method: "POST",
      headers: { ...window.getAuthHeaders?.() },
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok || !data.success) {
      videoSummaryStatusEl.textContent = data.detail || "Could not summarize this video.";
      return;
    }
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    videoSummaryStatusEl.textContent = `Finished in ${elapsed}s.`;
    videoSummaryOutputEl.textContent = formatVideoSummary(data);
    videoSummaryOutputEl.hidden = false;
    window.refreshHistory?.();
  } catch (e) {
    videoSummaryStatusEl.textContent = "Network error while summarizing video.";
    console.error(e);
  } finally {
    videoSummaryBtn.disabled = false;
  }
}

function resetVoiceButton() {
  recognitionActive = false;
  recognitionStopRequested = false;
  voiceBtn.classList.remove("recording");
  voiceBtn.textContent = "Hold to speak";
  voiceBtn.setAttribute("aria-pressed", "false");
}

function stopVoiceRecognition() {
  if (!recognition || !recognitionActive) return;
  recognitionStopRequested = true;
  const elapsed = Date.now() - recognitionStartedAt;
  const stopNow = () => {
    if (!recognition || !recognitionActive) return;
    try {
      recognition.stop();
    } catch (e) {
      console.warn("SpeechRecognition stop failed", e);
      resetVoiceButton();
    }
  };
  if (elapsed < 700) {
    setTimeout(stopNow, 700 - elapsed);
  } else {
    stopNow();
  }
}

async function startVoiceRecognition() {
  if (voiceBusy) return;
  if (!SpeechRecognition) {
    voiceTranscriptEl.textContent = "Speech recognition not supported in this browser. Use Chrome.";
    voiceTranscriptEl.hidden = false;
    return;
  }
  stopReplyAudio();
  window.speechSynthesis?.cancel?.();
  const micOk = await ensureMicPermission();
  if (!micOk) return;

  recognition = initRecognition();
  let finalTranscript = "";
  let latestTranscript = "";
  let gotError = false;
  recognitionStopRequested = false;
  voiceBtn.classList.add("recording");
  voiceBtn.textContent = "Listening…";
  voiceBtn.setAttribute("aria-pressed", "true");
  voiceTranscriptEl.textContent = "Listening…";
  voiceTranscriptEl.hidden = false;

  recognition.onresult = (ev) => {
    let interimTranscript = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const piece = ev.results[i]?.[0]?.transcript || "";
      if (ev.results[i].isFinal) {
        finalTranscript += ` ${piece}`;
      } else {
        interimTranscript += ` ${piece}`;
      }
    }
    latestTranscript = `${finalTranscript} ${interimTranscript}`.replace(/\s+/g, " ").trim();
    if (latestTranscript) {
      voiceTranscriptEl.textContent = `You: ${latestTranscript}`;
      voiceTranscriptEl.hidden = false;
    }
  };

  recognition.onerror = (ev) => {
    gotError = true;
    console.error("SpeechRecognition error", ev.error);
    if (ev.error === "aborted" && recognitionStopRequested) {
      voiceTranscriptEl.textContent = "Listening stopped.";
      voiceTranscriptEl.hidden = false;
      return;
    }
    const messages = {
      "not-allowed": "Mic permission denied. Allow microphone access, then try again.",
      "service-not-allowed": "Speech recognition is blocked in this browser. Try Chrome permissions.",
      "audio-capture": "No microphone was found.",
      "no-speech": "I didn't catch speech. Hold the button and speak after it says Listening.",
      network: "Speech recognition network error. Try again.",
      aborted: "Speech input was interrupted. Hold the button, speak, then release.",
    };
    voiceTranscriptEl.textContent = messages[ev.error] || `Mic error: ${ev.error}`;
    voiceTranscriptEl.hidden = false;
  };

  recognition.onend = () => {
    resetVoiceButton();
    const text = (finalTranscript || latestTranscript).replace(/\s+/g, " ").trim();
    if (text) {
      voiceTranscriptEl.textContent = `You: ${text}`;
      voiceTranscriptEl.hidden = false;
      if (isSafeModePhrase(text)) {
        startSafeMode();
        runSafeMode(text);
      } else if (safeModeActive) {
        runSafeMode(text);
      } else {
        runChat(text);
      }
    } else if (!gotError) {
      voiceTranscriptEl.textContent = "No speech caught. Hold the button, speak clearly, then release.";
      voiceTranscriptEl.hidden = false;
    }
  };

  try {
    recognition.start();
    recognitionActive = true;
    recognitionStartedAt = Date.now();
  } catch (e) {
    console.error("SpeechRecognition start failed", e);
    voiceTranscriptEl.textContent = "Mic could not start. Check browser microphone permission.";
    voiceTranscriptEl.hidden = false;
    resetVoiceButton();
  }
}

voiceBtn.addEventListener("mousedown", startVoiceCapture);

voiceBtn.addEventListener("mouseup", stopVoiceCapture);
window.addEventListener("mouseup", stopVoiceCapture);

// Touch-event parallels for mobile. preventDefault on touchstart suppresses
// the synthesized mousedown so the handler doesn't fire twice.
voiceBtn.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startVoiceCapture();
}, { passive: false });
voiceBtn.addEventListener("touchend",    stopVoiceCapture);
voiceBtn.addEventListener("touchcancel", stopVoiceCapture);

// ---------------- Wiring ----------------
autoBtn.addEventListener("click", () => {
  if (autoCaptureTimer === null) startAutoCapture();
  else stopAutoCapture();
});

safeModeBtn?.addEventListener("click", () => {
  if (safeModeActive) stopSafeMode();
  else startSafeMode();
});

describeBtn.addEventListener("click", () => runInvestigation("_describe"));
changeBtn.addEventListener("click", () => runChangeAnalysis());
videoSummaryForm?.addEventListener("submit", runVideoSummary);

// ---------------- Boot ----------------
(async () => {
  try {
    bindCameraLifecycle();
    await startCameraStream();
    startCameraWatchdog();
    describeBtn.disabled = false;
    changeBtn.disabled = false;
    autoBtn.disabled = false;
    if (safeModeBtn) safeModeBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Camera permission denied. Reload and grant camera access.", "error");
    return;
  }

  setStatus(localStorage.getItem("aeyez_token") ? "Ready." : "Sign in to begin.");

  initCalibration();
})();
