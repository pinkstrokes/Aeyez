// Rolling window of camera frames sampled at a fixed rate. Designed so any
// trigger can pull a coherent "last N seconds" clip and sample frames from it
// for model input. The model takes images, not video — this layer is where
// we decide which frames from a clip best represent it.
//
// Frame shape: { ts: epoch-millis, dataUrl: "data:image/jpeg;base64,..." }
//
// Sampling strategies:
//   'latest'  -> single most-recent frame        (investigate "now")
//   'edges'   -> [oldest, newest]                (what-changed diff)
//   'uniform' -> N evenly spaced across window   (multi-frame scene)
//
// Classic (non-module) script that exposes `window.ClipBuffer`.

(function () {
  class ClipBuffer {
    constructor({ windowMs = 10_000, fps = 1, intervalMs = null, captureFn } = {}) {
      if (typeof captureFn !== "function") {
        throw new Error("ClipBuffer requires a captureFn that returns a JPEG data URL");
      }
      this.windowMs = windowMs;
      this.intervalMs = intervalMs === null ? Math.round(1000 / fps) : Math.round(intervalMs);
      // +1 so a 10s window at 1fps holds frames at t=0,1,...,10.
      this.maxFrames = Math.ceil(windowMs / this.intervalMs) + 1;
      this.captureFn = captureFn;
      this.frames = [];
      this.timer = null;
    }

    start() {
      if (this.timer !== null) return;
      this._tick();
      this.timer = setInterval(() => this._tick(), this.intervalMs);
    }

    stop() {
      if (this.timer === null) return;
      clearInterval(this.timer);
      this.timer = null;
    }

    _tick() {
      const frame = this._safeCapture();
      if (frame) this._append(frame);
    }

    _safeCapture() {
      try {
        const dataUrl = this.captureFn();
        if (!dataUrl) return null;
        return { ts: Date.now(), dataUrl };
      } catch {
        return null;
      }
    }

    _append(frame) {
      this.frames.push(frame);
      while (this.frames.length > this.maxFrames) this.frames.shift();
    }

    // Snap an explicit frame *right now* (e.g. at the moment a trigger fires),
    // independent of the background ticker. The fresh frame is also appended
    // to the buffer so later samples include it.
    captureNow() {
      const frame = this._safeCapture();
      if (frame) this._append(frame);
      return frame;
    }

    length() {
      return this.frames.length;
    }

    latest() {
      return this.frames.length ? this.frames[this.frames.length - 1] : null;
    }

    closestTo(targetTs, maxDeltaMs = Infinity) {
      if (this.frames.length === 0 || !Number.isFinite(targetTs)) return null;
      let best = null;
      let bestDelta = maxDeltaMs;
      for (const frame of this.frames) {
        const delta = Math.abs(frame.ts - targetTs);
        if (delta <= bestDelta) {
          best = frame;
          bestDelta = delta;
        }
      }
      return best;
    }

    // Everything currently in the window, oldest first. Returns a shallow
    // copy so callers can mutate freely.
    clip() {
      return this.frames.slice();
    }

    // Pick representative frames from the current window. `n` only applies
    // to strategies that take a count ('uniform'). Returns [] if empty.
    sample(strategy, n = 2) {
      const frames = this.frames;
      if (frames.length === 0) return [];

      if (strategy === "latest") {
        return [frames[frames.length - 1]];
      }

      if (strategy === "edges") {
        if (frames.length === 1) return [frames[0]];
        return [frames[0], frames[frames.length - 1]];
      }

      if (strategy === "uniform") {
        const k = Math.max(1, Math.min(n, frames.length));
        if (k === 1) return [frames[frames.length - 1]];
        const out = [];
        for (let i = 0; i < k; i++) {
          const idx = Math.round((i * (frames.length - 1)) / (k - 1));
          out.push(frames[idx]);
        }
        return out;
      }

      throw new Error(`Unknown sampling strategy: ${strategy}`);
    }
  }

  window.ClipBuffer = ClipBuffer;
})();
