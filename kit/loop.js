// The frame loop: timing, canvas sizing and adaptive quality. Allocation-free per frame.

const MIN_SCALE = 0.35;

export function createLoop(canvas, P, sketch) {
  const P_timeScale = P.timeScale;
  const P_renderScale = P.renderScale, P_maxDpr = P.maxDpr;
  const P_targetFps = P.targetFps, P_maxFps = P.maxFps;

  let raf = 0, last = 0, sceneTime = 0, realTime = 0;
  let fpsAccum = 0, fpsFrames = 0;
  let lastW = 0, lastH = 0, lastRatio = 0, frames = 0;

  let dynScale = 1;
  let winTime = 0, winFrames = 0, sinceChange = 0, ceiling = 1, ceilingAge = 0;
  let judgingDrop = false, preDropScale = 1, preDropAvg = 0, holdDown = 0, holdDownLen = 5;
  let settling = false;

  // Automatic resolution downgrading is disabled so watching on phones or desktops
  // never degrades render quality over time; resolution is controlled solely by P_renderScale & P_maxDpr.
  function adaptive() { return false; }

  function adapt(_dt) {
    dynScale = 1;
  }

  function resizeIfNeeded(s) {
    const isMobile = window.innerWidth <= 820 || (window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
    const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
    const maxDpr = P_maxDpr ? P_maxDpr.value : 1.5;
    const scale = P_renderScale ? P_renderScale.value : 1;
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
    resizeIfNeeded(s);
    s.frame(sceneTime, sdt, realTime, dt);
    frames++;

    fpsAccum += dt;
    fpsFrames++;
    if (fpsAccum >= 0.5) {
      if (loop.onStats) {
        loop.onStats(canvas.width + '×' + canvas.height + 'dp @ ' + Math.round(fpsFrames / fpsAccum) + 'fps' +
          (adaptive() ? ' (' + Math.round(dynScale * 100) + '%)' : ''));
      }
      fpsAccum = 0;
      fpsFrames = 0;
    }
  }

  function cappedFps() {
    return P_maxFps && !loop.fixed ? P_maxFps.value : 0;
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const cap = cappedFps();
    if (cap > 0 && last && now - last < 900 / cap) return;
    const dt = last ? Math.min((now - last) * 0.001, 0.1) : 0;
    last = now;
    render(dt);
  }

  const loop = {
    fixed: false,
    onStats: null,
    onTick: null,
    paused: false,
    get frames() { return frames; },
    start() {
      last = 0;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    },
    stop() { cancelAnimationFrame(raf); raf = 0; },
    step(dt) { render(dt); },
    degrade(_factor) { dynScale = 1; },
    snapshot(width, height, pxRatio) {
      const s = sketch();
      if (!s) return Promise.reject(new Error('The sketch is not running.'));
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
