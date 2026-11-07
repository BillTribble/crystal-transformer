// The frame loop: timing, canvas sizing, mobile dynamic resolution governor, and background pause.

import { detectMobileDevice } from './gl.js';

const MOBILE_MAX_REST_DPR = 1.35;

export function createLoop(canvas, P, sketch) {
  const P_timeScale = P.timeScale;
  const P_renderScale = P.renderScale, P_maxDpr = P.maxDpr;
  const P_maxFps = P.maxFps;

  let raf = 0, last = 0, sceneTime = 0, realTime = 0;
  let fpsAccum = 0, fpsFrames = 0;
  let lastW = 0, lastH = 0, lastRatio = 0, frames = 0;

  let dynScale = 1;
  let running = false;
  let shaderFirstReadyAt = 0;
  let lastInteractAt = 0;

  function computeMobileDynScale(s, nowMs) {
    if (loop.fixed) return 1;
    if (!s || !s.shaderReady) {
      shaderFirstReadyAt = 0;
      return 0.40;
    }
    if (!shaderFirstReadyAt) shaderFirstReadyAt = nowMs;
    const readyElapsed = nowMs - shaderFirstReadyAt;
    // Progressive cold-start ramp on mobile: 0.40x on Frame 1 (< 5ms GPU draw), 0.72x over 150..420ms, 1.00x settled
    let startupScale = 1.0;
    if (readyElapsed < 150) startupScale = 0.40;
    else if (readyElapsed < 420) startupScale = 0.72;

    if (loop.interacting) {
      lastInteractAt = nowMs;
      return Math.min(startupScale, 0.68);
    }
    if (lastInteractAt > 0 && (nowMs - lastInteractAt) < 250) {
      return Math.min(startupScale, 0.68);
    }
    return startupScale;
  }

  function adapt(_dt) {
    const isMobile = detectMobileDevice();
    if (!isMobile || loop.fixed) {
      dynScale = 1;
      return;
    }
    dynScale = computeMobileDynScale(sketch(), performance.now());
  }

  function resizeIfNeeded(s) {
    const isMobile = detectMobileDevice();
    const rawDpr = window.devicePixelRatio || 1;
    const dpr = Math.min(rawDpr, isMobile ? MOBILE_MAX_REST_DPR : 2);
    const maxDpr = isMobile
      ? Math.min(P_maxDpr ? P_maxDpr.value : 1.25, MOBILE_MAX_REST_DPR)
      : (P_maxDpr ? P_maxDpr.value : 1.5);
    const scale = (P_renderScale ? P_renderScale.value : 1) * (isMobile && !loop.fixed ? dynScale : 1);
    const ratio = Math.min(dpr, maxDpr) * scale;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    const w = Math.max(1, Math.round(cw * ratio));
    const h = Math.max(1, Math.round(ch * ratio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const px = cw > 0 ? w / cw : 1;
    if (w !== lastW || h !== lastH || px !== lastRatio) {
      lastW = w;
      lastH = h;
      lastRatio = px;
      s.resize(w, h, px);
    }
  }

  function render(dt) {
    const s = sketch();
    if (!s) return;
    realTime = (realTime + dt) % 10000;
    const sdt = loop.paused ? 0 : dt * (P_timeScale ? P_timeScale.value : 1);
    sceneTime += sdt;
    adapt(dt);
    if (loop.onTick) loop.onTick(dt, sdt);
    adapt(dt);
    resizeIfNeeded(s);
    s.frame(sceneTime, sdt, realTime, dt);
    frames++;

    fpsAccum += dt;
    fpsFrames++;
    if (fpsAccum >= 0.5) {
      if (loop.onStats) {
        loop.onStats(canvas.width + '×' + canvas.height + 'dp @ ' + Math.round(fpsFrames / fpsAccum) + 'fps' +
          (dynScale < 0.99 ? ' (' + Math.round(dynScale * 100) + '%)' : ''));
      }
      fpsAccum = 0;
      fpsFrames = 0;
    }
  }

  function cappedFps() {
    return P_maxFps && !loop.fixed ? P_maxFps.value : 0;
  }

  function frame(now) {
    if (!running) {
      raf = 0;
      return;
    }
    if (typeof document !== 'undefined' && document.hidden) {
      raf = 0;
      return;
    }
    raf = requestAnimationFrame(frame);
    const s = sketch();
    const inStartup = s && (!s.shaderReady || s.fillInProgress < 1.0);
    const cap = inStartup ? 0 : cappedFps();
    if (cap > 0 && last && now - last < 900 / cap) return;
    const rawElapsed = last ? (now - last) * 0.001 : 0.016;
    const dt = rawElapsed > 0.12 ? 0.016 : rawElapsed;
    last = now;
    render(dt);
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        if (raf) {
          cancelAnimationFrame(raf);
          raf = 0;
        }
      } else if (running && !raf) {
        last = performance.now();
        raf = requestAnimationFrame(frame);
      }
    });
  }

  const loop = {
    fixed: false,
    interacting: false,
    onStats: null,
    onTick: null,
    paused: false,
    get frames() { return frames; },
    get dynScale() { return dynScale; },
    start() {
      running = true;
      cancelAnimationFrame(raf);
      if (frames === 0) render(0);
      last = performance.now();
      if (typeof document === 'undefined' || !document.hidden) {
        raf = requestAnimationFrame(frame);
      }
    },
    stop() { running = false; cancelAnimationFrame(raf); raf = 0; },
    step(dt) { render(dt); },
    degrade(factor) {
      if (detectMobileDevice() && !loop.fixed) {
        dynScale = Math.max(0.40, Math.min(1.0, Number(factor) || 0.68));
        lastInteractAt = performance.now();
      }
    },
    snapshot(width, height, pxRatio) {
      const s = sketch();
      if (!s) return Promise.reject(new Error('The sketch is not running.'));
      s.ensureReadySync?.();
      canvas.width = width;
      canvas.height = height;
      s.resize(width, height, pxRatio);
      s.frame(sceneTime, 0, realTime, 0);
      const png = new Promise((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('The browser could not encode the image.'))), 'image/png'));
      lastW = 0;
      resizeIfNeeded(s);
      s.frame(sceneTime, 0, realTime, 0);
      return png;
    },
  };
  return loop;
}
