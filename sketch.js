// Crystals — WebGL2 renderer + dedicated OffscreenCanvas Worker 3D Quartz Wireframe startup
// with wall-clock phase-locked camera rotation (yaw(t)), non-blocking gl.fenceSync PSO pre-warm,
// and seamless ~950ms cubic ease-in-out crossfade into the live ray-traced crystal.
// Built on Jack Bush's Geode renderer (jackbush/glimmers).

import SHADER from './shader.js';
import { buildCluster, cluster } from './cluster.js';
import { FULLSCREEN_VERT, detectMobileDevice, frameUniforms, probeFloatTargets, program, shaderHeader, shaderParams, uploadFrame, uploadParams } from './kit/gl.js';

const UNIT_INCL = 0, UNIT_HDR = 1, UNIT_BLOOM_A = 2, UNIT_BLOOM_B = 3;
const BLOOM_DIVISOR = 4;
const COMPLETION_STATUS_KHR = 0x91b1;
const STARTUP_SPIN_RAD_S = 0.28; // ~16.043 deg/s graceful slow 3D startup rotation
const STARTUP_SPIN_DEG_S = (STARTUP_SPIN_RAD_S * 180) / Math.PI;
const MIN_WIREFRAME_HOLD_MS = 900;
const CROSSFADE_DURATION_MS = 950;
const FENCE_TIMEOUT_MS = 8500;
const DEG = Math.PI / 180;
const TAU = Math.PI * 2;

function wallNow() {
  return (typeof performance !== 'undefined' && typeof performance.timeOrigin === 'number')
    ? performance.timeOrigin + performance.now()
    : Date.now();
}

function wrap360(deg) {
  return ((deg % 360) + 360) % 360;
}

function cubicEaseInOut(u) {
  const x = Math.max(0, Math.min(1, u));
  return x * x * (3 - 2 * x);
}

// Self-contained Dedicated Worker entry point for locked 60 FPS OffscreenCanvas 3D wireframe rendering.
// Zero external closures: communicates strictly via structured messages and shared monotonic epoch `bootEpochMs`.
function wireframeWorkerMain() {
  const W_DEG = Math.PI / 180;
  const W_TAU = Math.PI * 2;
  const M = new Float64Array(9);
  const A = new Float64Array(9);
  const B = new Float64Array(9);
  const T9 = new Float64Array(9);
  const cam = new Float64Array(9);
  const camPos = new Float64Array(3);
  const projX = new Float32Array(128);
  const projY = new Float32Array(128);
  const projZ = new Float32Array(128);
  const faceFront = new Uint8Array(24);
  const faceLight = new Float32Array(24);

  let offscreen = null;
  let ctx = null;
  let running = false;
  let bootEpochMs = 0;
  let cssW = 1280;
  let cssH = 800;
  let dpr = 1;
  let bodies = [];
  let camera = null;
  let tint = { r: 220, g: 236, b: 250 };
  let frames = 0;
  let rafId = 0;
  let timerId = 0;
  let lastDrawWallMs = 0;

  function nowEpoch() {
    return (self.performance && typeof self.performance.timeOrigin === 'number')
      ? self.performance.timeOrigin + self.performance.now()
      : Date.now();
  }

  function set9(m, a, b, c, d, e, f, g, h, i) {
    m[0] = a; m[1] = b; m[2] = c;
    m[3] = d; m[4] = e; m[5] = f;
    m[6] = g; m[7] = h; m[8] = i;
  }
  function rotX(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, 1, 0, 0, 0, c, -s, 0, s, c); }
  function rotY(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, c, 0, s, 0, 1, 0, -s, 0, c); }
  function rotZ(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, c, -s, 0, s, c, 0, 0, 0, 1); }
  function mul(out, a, b) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        T9[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
      }
    }
    out.set(T9);
  }

  function drawFrame() {
    if (!ctx || !offscreen || !camera) return;
    const wallMs = nowEpoch();
    lastDrawWallMs = wallMs;
    const elapsedMs = Math.max(0, wallMs - bootEpochMs);
    const elapsedSec = elapsedMs * 0.001;
    const yawDeg = camera.overrideYawDeg !== null && camera.overrideYawDeg !== undefined
      ? (((camera.overrideYawDeg % 360) + 360) % 360)
      : (((camera.baseYawDeg + camera.spinDegPerSec * elapsedSec) % 360) + 360) % 360;

    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (offscreen.width !== w || offscreen.height !== h) {
      offscreen.width = w;
      offscreen.height = h;
    }
    ctx.clearRect(0, 0, w, h);

    if (bodies.length > 0) {
      const sceneT = elapsedSec * (camera.timeScale || 1.5);
      const sw = (camera.swayDeg || 0) * W_DEG;
      const ss = W_TAU * (camera.swaySpeed || 0.07) * sceneT;
      rotY(M, yawDeg * W_DEG);
      rotZ(A, sw * Math.sin(ss));
      rotX(B, sw * 0.7 * Math.sin(ss * 1.618 + 1.3));
      mul(M, A, M);
      mul(M, B, M);
      rotX(A, (camera.tiltDeg || 0) * W_DEG);
      mul(M, A, M);

      const fov = (camera.fovDeg || 21) * W_DEG;
      const portraitZoom = w < h ? Math.min(1.36, Math.pow(h / Math.max(1, w), 0.38)) : 1.0;
      const focal = (0.5 / Math.tan(fov * 0.5)) * portraitZoom;
      const D = camera.distance || 2.15;
      for (let k = 0; k < 3; k++) {
        cam[k] = M[k];
        cam[3 + k] = M[3 + k];
        cam[6 + k] = -M[6 + k];
        const cOff = k === 0 ? (camera.centreX || 0) : k === 1 ? (camera.centreY || 0) : (camera.centreZ || 0);
        camPos[k] = D * M[6 + k] + cOff;
      }

      const halfW = w * 0.5, halfH = h * 0.5, minWH = Math.min(w, h);
      const camPos0 = camPos[0], camPos1 = camPos[1], camPos2 = camPos[2];
      const cam0 = cam[0], cam1 = cam[1], cam2 = cam[2];
      const cam3 = cam[3], cam4 = cam[4], cam5 = cam[5];
      const cam6 = cam[6], cam7 = cam[7], cam8 = cam[8];

      const keyX = -cam0 * 0.55 + cam3 * 0.65 - cam6 * 0.52;
      const keyY = -cam1 * 0.55 + cam4 * 0.65 - cam7 * 0.52;
      const keyZ = -cam2 * 0.55 + cam5 * 0.65 - cam8 * 0.52;

      for (let b = 0; b < bodies.length; b++) {
        const body = bodies[b];
        body.sphereZ = (body.cx - camPos0) * cam6 + (body.cy - camPos1) * cam7 + (body.cz - camPos2) * cam8;
      }
      const sorted = bodies.slice().sort((a, b) => b.sphereZ - a.sphereZ);
      const tintR = tint.r, tintG = tint.g, tintB = tint.b;

      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (let b = 0; b < sorted.length; b++) {
        const body = sorted[b];
        const bAlpha = Math.min(1, body.alpha);
        if (bAlpha <= 0.01) continue;

        const verts = body.verts;
        const nv = body.nv;
        const scale = focal * minWH;
        for (let u = 0; u < nv; u++) {
          const rx = verts[u * 3] - camPos0;
          const ry = verts[u * 3 + 1] - camPos1;
          const rz = verts[u * 3 + 2] - camPos2;
          const cx = rx * cam0 + ry * cam1 + rz * cam2;
          const cy = rx * cam3 + ry * cam4 + rz * cam5;
          const cz = Math.max(0.08, rx * cam6 + ry * cam7 + rz * cam8);
          const invZ = scale / cz;
          projX[u] = halfW + cx * invZ;
          projY[u] = halfH - cy * invZ;
          projZ[u] = cz;
        }

        const normals = body.normals;
        const faces = body.faces;
        faceFront.fill(0);
        for (let f = 0; f < faces.length; f++) {
          const face = faces[f];
          const p = face.plane;
          const nx = normals[p * 3], ny = normals[p * 3 + 1], nz = normals[p * 3 + 2];
          const vx = face.cx - camPos0, vy = face.cy - camPos1, vz = face.cz - camPos2;
          const isFront = (nx * vx + ny * vy + nz * vz) < 0;
          faceFront[p] = isFront ? 1 : 0;
          const kDot = Math.max(0, nx * keyX + ny * keyY + nz * keyZ);
          faceLight[p] = kDot;

          if (isFront) {
            const idx = face.indices;
            ctx.beginPath();
            ctx.moveTo(projX[idx[0]], projY[idx[0]]);
            for (let m = 1; m < idx.length; m++) {
              ctx.lineTo(projX[idx[m]], projY[idx[m]]);
            }
            ctx.closePath();
            const fAlpha = (0.024 + 0.068 * Math.pow(kDot, 1.8)) * bAlpha;
            ctx.fillStyle = 'rgba(' + tintR + ',' + tintG + ',' + tintB + ',' + fAlpha.toFixed(4) + ')';
            ctx.fill();
          }
        }

        const edges = body.edges;

        // Pass 1: Back-facing internal refracted crystal ridges
        ctx.beginPath();
        for (let e = 0; e < edges.length; e++) {
          const ed = edges[e];
          if (!faceFront[ed.pA] && !faceFront[ed.pB]) {
            ctx.moveTo(projX[ed.u], projY[ed.u]);
            ctx.lineTo(projX[ed.v], projY[ed.v]);
          }
        }
        ctx.strokeStyle = 'rgba(' + Math.round(tintR * 0.72) + ',' + Math.round(tintG * 0.84) + ',' + tintB + ',' + (0.26 * bAlpha).toFixed(3) + ')';
        ctx.lineWidth = 0.85 * dpr;
        ctx.stroke();

        // Pass 2: Front-facing crystallographic facet ridges
        ctx.beginPath();
        for (let e = 0; e < edges.length; e++) {
          const ed = edges[e];
          if (faceFront[ed.pA] && faceFront[ed.pB]) {
            ctx.moveTo(projX[ed.u], projY[ed.u]);
            ctx.lineTo(projX[ed.v], projY[ed.v]);
          }
        }
        ctx.strokeStyle = 'rgba(' + tintR + ',' + tintG + ',' + tintB + ',' + (0.70 * bAlpha).toFixed(3) + ')';
        ctx.lineWidth = 1.25 * dpr;
        ctx.stroke();

        // Pass 3: Outer 3D crystal silhouette edges (soft halo + crisp bright rim)
        ctx.beginPath();
        for (let e = 0; e < edges.length; e++) {
          const ed = edges[e];
          if (faceFront[ed.pA] !== faceFront[ed.pB]) {
            ctx.moveTo(projX[ed.u], projY[ed.u]);
            ctx.lineTo(projX[ed.v], projY[ed.v]);
          }
        }
        ctx.strokeStyle = 'rgba(' + Math.round(tintR * 0.78) + ',' + Math.round(tintG * 0.92) + ',255,' + (0.24 * bAlpha).toFixed(3) + ')';
        ctx.lineWidth = 3.4 * dpr;
        ctx.stroke();

        ctx.strokeStyle = 'rgba(236,247,255,' + (0.92 * bAlpha).toFixed(3) + ')';
        ctx.lineWidth = 1.6 * dpr;
        ctx.stroke();

        // Pass 4: Subtle vertex catch-light glints on termination tips
        const vertHgt = body.vertHgt;
        ctx.fillStyle = 'rgba(245,252,255,' + (0.82 * bAlpha).toFixed(3) + ')';
        const dotR = 1.55 * dpr;
        for (let u = 0; u < nv; u++) {
          if (vertHgt[u] >= 0.85) {
            ctx.beginPath();
            ctx.arc(projX[u], projY[u], dotR, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.restore();
    }

    frames++;
    self.postMessage({
      type: 'tick',
      frames,
      yawDeg,
      elapsedMs,
      wallMs,
    });
  }

  function scheduleNext() {
    if (!running) return;
    if (typeof self.requestAnimationFrame === 'function') {
      if (!rafId) {
        rafId = self.requestAnimationFrame(() => {
          rafId = 0;
          if (timerId) {
            self.clearTimeout(timerId);
            timerId = 0;
          }
          if (!running) return;
          drawFrame();
          scheduleNext();
        });
      }
      // Backpressure-safe fallback: if headless software compositor stalls RAF on CrGpuMain,
      // step at a bounded 90ms cadence while `rafId` is still pending so yaw advances continuously
      // without flooding >2GB of un-acked SharedBitmap frames; resumes 60fps RAF immediately once Viz acks.
      if (!timerId) {
        timerId = self.setTimeout(() => {
          timerId = 0;
          if (!running) return;
          if (nowEpoch() - lastDrawWallMs >= 80) {
            drawFrame();
          }
          scheduleNext();
        }, 90);
      }
    } else if (!timerId) {
      timerId = self.setTimeout(() => {
        timerId = 0;
        if (!running) return;
        drawFrame();
        scheduleNext();
      }, 16);
    }
  }

  self.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg) return;
    if (msg.type === 'init') {
      offscreen = msg.canvas;
      ctx = offscreen.getContext('2d');
      bootEpochMs = msg.bootEpochMs;
      cssW = msg.width;
      cssH = msg.height;
      dpr = msg.dpr;
      bodies = msg.bodies || [];
      camera = msg.camera;
      if (msg.tint) tint = msg.tint;
      running = true;
      drawFrame();
      scheduleNext();
    } else if (msg.type === 'resize') {
      cssW = msg.width;
      cssH = msg.height;
      dpr = msg.dpr;
      if (msg.bodies) bodies = msg.bodies;
      if (msg.camera) camera = msg.camera;
      if (msg.tint) tint = msg.tint;
    } else if (msg.type === 'camera') {
      if (msg.camera) camera = msg.camera;
      if (msg.tint) tint = msg.tint;
    } else if (msg.type === 'renderNow') {
      drawFrame();
    } else if (msg.type === 'stop') {
      running = false;
      if (rafId && typeof self.cancelAnimationFrame === 'function') self.cancelAnimationFrame(rafId);
      if (timerId) self.clearTimeout(timerId);
      rafId = 0;
      timerId = 0;
      bodies = [];
      camera = null;
    }
  };
}

export default function create({ canvas, P, model, degrade, onStartupHandoff }) {
  const sketchStartPerf = performance.now();
  const bootEpochMs = wallNow();
  const sp = new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
  const isCapture = sp.has('capture');
  const isMobile = detectMobileDevice();
  let holdWireframe = sp.has('wireframe');
  const pinnedFillIn = sp.has('fillin') ? Math.max(0, Math.min(1, parseFloat(sp.get('fillin')))) : null;
  const minHoldMs = (isCapture || pinnedFillIn !== null || sp.has('fast')) ? 0 : MIN_WIREFRAME_HOLD_MS;

  const baseYawDeg = Number(P.spinStart?.value ?? 280.757);
  const spinSign = (P.spin && Number(P.spin.value) < 0) ? -1 : 1;
  const startupSpinDegPerSec = spinSign * STARTUP_SPIN_DEG_S;

  let wfCanvas = document.getElementById('wfCanvas') || document.getElementById('wireframe-stage');
  let wfTransferred = false;
  function ensureWfCanvasElement() {
    if (wfCanvas && wfCanvas.isConnected && !wfTransferred) return wfCanvas;
    if (wfCanvas && wfCanvas.isConnected) wfCanvas.remove();
    wfCanvas = document.createElement('canvas');
    wfCanvas.id = 'wfCanvas';
    wfCanvas.setAttribute('aria-hidden', 'true');
    wfCanvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;display:block;will-change:opacity;';
    canvas.parentElement?.insertBefore(wfCanvas, canvas.nextSibling);
    wfTransferred = false;
    return wfCanvas;
  }
  ensureWfCanvasElement();

  let wfCtx = null;
  let wfWorker = null;
  let wfWorkerUrl = null;
  let usingOffscreenWorker = false;
  let userOverrodeCamera = false;
  let startupPhaseActive = !(isCapture && !holdWireframe) && pinnedFillIn === null;
  let handoffFired = !startupPhaseActive;

  let gl = null;
  let extParallel = null;
  function ensureGl() {
    if (gl && !gl.isContextLost()) return gl;
    gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: isMobile ? Boolean(isCapture) : true,
      powerPreference: isMobile ? 'default' : 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    extParallel = gl.getExtension('KHR_parallel_shader_compile');
    return gl;
  }

  const uniformParams = shaderParams(model.params);
  const { vals: frameVals, list: frameList } = frameUniforms(SHADER.frameUniforms);
  const INCL = isMobile ? 32 : SHADER.inclSize;
  const noiseVol = new Uint8Array(INCL * INCL * INCL * 4);

  let ready = false, shaderReady = false, buildStage = 0, vao = null, ubo = null;
  let passBackdrop = null, passTrace = null, passBright = null, passBlur = null, passComposite = null;
  let pendingPrograms = null;
  let inclTex = null, bakedSeed = NaN;
  let hdrTex = null, hdrFbo = null, bloomA = null, bloomAFbo = null, bloomB = null, bloomBFbo = null;
  let hdrW = 0, hdrH = 0, bloomW = 0, bloomH = 0, floatTargets = false;
  let clusterVersion = -1, uploadedClusterVer = -1, lastClusterW = 0, lastClusterH = 0, pxRatio = 1;
  let mainBootTicks = 0;
  let wireframeFrames = 0;
  let wireframeYawDeg = baseYawDeg;
  const wireframeLog = [];
  let fillInProgress = (isCapture && !holdWireframe) ? 1.0 : (pinnedFillIn !== null ? pinnedFillIn : 0.0);
  let wireframeAlpha = 1.0 - cubicEaseInOut(fillInProgress);
  let firstWireframeMs = null, firstWireframeNavMs = null, shaderReadyMs = null;
  let crossfadeStartWallMs = 0, crossfadeStarted = false;
  let prewarmSync = null, fenceCreatedWallMs = 0, fenceSignaled = false, fenceWaitMs = null;
  let bodyBackdropApplied = false;

  // Preallocated scratch buffers for main-thread 2D fallback wireframe projection
  const projX = new Float32Array(128);
  const projY = new Float32Array(128);
  const projZ = new Float32Array(128);
  const faceFront = new Uint8Array(24);
  const faceLight = new Float32Array(24);

  function startCompileAsync(name, src, samplers, postOnly = false) {
    const commonBlock = (postOnly && SHADER.commonPost) ? SHADER.commonPost : SHADER.common;
    const fragSrc = shaderHeader(uniformParams, frameList, isMobile) + commonBlock + src;
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, FULLSCREEN_VERT);
    gl.compileShader(vs);
    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    return { name, prog, vs, fs, samplers };
  }

  function finishPass(spec) {
    const { name, prog, vs, fs, samplers } = spec;
    const linked = gl.getProgramParameter(prog, gl.LINK_STATUS);
    if (!linked && !gl.isContextLost()) {
      const vsLog = !gl.getShaderParameter(vs, gl.COMPILE_STATUS) ? gl.getShaderInfoLog(vs) : '';
      const fsLog = !gl.getShaderParameter(fs, gl.COMPILE_STATUS) ? gl.getShaderInfoLog(fs) : '';
      const progLog = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error('Program "' + name + '" failed to link:\n' + (vsLog ? vsLog + '\n' : '') + (fsLog ? fsLog + '\n' : '') + progLog);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    const loc = (n) => gl.getUniformLocation(prog, n);
    gl.useProgram(prog);
    for (const s in samplers) gl.uniform1i(loc(s), samplers[s]);
    return {
      prog,
      version: -1,
      uRes: loc('u_res'), uScreen: loc('u_screen'), uTime: loc('u_time'), uRealTime: loc('u_realTime'),
      uSlice: loc('u_slice'), uDir: loc('u_dir'), uSrc: loc('u_src'),
      paramLocs: uniformParams.map((p) => loc('u_' + p.key)),
      frameLocs: frameList.map((f) => loc(f.name)),
    };
  }

  function texture2D(unit) {
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }
  function framebuffer(tex) {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return fbo;
  }
  function allocate(unit, w, h) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    if (floatTargets) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  function sizeTargets() {
    const w = canvas.width, h = canvas.height;
    if (w !== hdrW || h !== hdrH) { hdrW = w; hdrH = h; allocate(UNIT_HDR, w, h); }
    const bw = Math.max(1, Math.ceil(w / BLOOM_DIVISOR)), bh = Math.max(1, Math.ceil(h / BLOOM_DIVISOR));
    if (bw !== bloomW || bh !== bloomH) { bloomW = bw; bloomH = bh; allocate(UNIT_BLOOM_A, bw, bh); allocate(UNIT_BLOOM_B, bw, bh); }
  }

  function initGlResourcesAndBake() {
    if (buildStage >= 1) return;
    ensureGl();
    ubo = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
    gl.bufferData(gl.UNIFORM_BUFFER, cluster.data.byteLength, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);
    uploadedClusterVer = -1;

    inclTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + UNIT_INCL);
    gl.bindTexture(gl.TEXTURE_3D, inclTex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, INCL, INCL, INCL, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    bakedSeed = NaN;
    bake();

    // On mobile, use compact RGBA8 FBOs directly (encodeHdr stores [0,1) Reinhard-compressed HDR, saving 50% VRAM & skipping probeFloatTargets's synchronous gl.getError() IPC stall!)
    floatTargets = isMobile ? false : probeFloatTargets(gl);
    hdrTex = texture2D(UNIT_HDR);
    bloomA = texture2D(UNIT_BLOOM_A);
    bloomB = texture2D(UNIT_BLOOM_B);
    hdrW = hdrH = bloomW = bloomH = 0;
    sizeTargets();
    hdrFbo = framebuffer(hdrTex);
    bloomAFbo = framebuffer(bloomA);
    bloomBFbo = framebuffer(bloomB);
    if (floatTargets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, hdrFbo);
      const hdrOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, bloomAFbo);
      const bloomAOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, bloomBFbo);
      const bloomBOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      if (!hdrOk || !bloomAOk || !bloomBOk) {
        floatTargets = false;
        hdrW = hdrH = bloomW = bloomH = 0;
        sizeTargets();
        gl.bindFramebuffer(gl.FRAMEBUFFER, hdrFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, hdrTex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomAFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bloomA, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomBFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bloomB, 0);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    buildStage = 1;
  }

  function kickCompileAsync() {
    if (buildStage < 1) initGlResourcesAndBake();
    if (buildStage >= 2) return;
    pendingPrograms = {
      backdrop: startCompileAsync('backdrop', SHADER.backdrop, {}, false),
      trace: startCompileAsync('trace', SHADER.trace, { u_incl: UNIT_INCL }, false),
      bright: startCompileAsync('bright', SHADER.bright, { u_hdr: UNIT_HDR }, true),
      blur: startCompileAsync('blur', SHADER.blur, { u_src: UNIT_BLOOM_A }, true),
      composite: startCompileAsync('composite', SHADER.composite, { u_hdr: UNIT_HDR, u_bloomTex: UNIT_BLOOM_A }, true),
    };
    buildStage = 2;
  }

  function isParallelCompileComplete() {
    if (!pendingPrograms) return false;
    if (!extParallel) return true;
    for (const k in pendingPrograms) {
      if (!gl.getProgramParameter(pendingPrograms[k].prog, COMPLETION_STATUS_KHR)) {
        return false;
      }
    }
    return true;
  }

  function finalizeCompiledPrograms() {
    if (buildStage >= 3 || !pendingPrograms) return;
    passBackdrop = finishPass(pendingPrograms.backdrop);
    passTrace = finishPass(pendingPrograms.trace);
    passBright = finishPass(pendingPrograms.bright);
    passBlur = finishPass(pendingPrograms.blur);
    passComposite = finishPass(pendingPrograms.composite);
    pendingPrograms = null;

    gl.uniformBlockBinding(passTrace.prog, gl.getUniformBlockIndex(passTrace.prog, 'Bodies'), 0);
    buildStage = 3;
  }

  function cleanupFence() {
    if (prewarmSync && gl && !gl.isContextLost()) {
      gl.deleteSync(prewarmSync);
    }
    prewarmSync = null;
  }

  function applyStudioBackdropToBody() {
    if (typeof document === 'undefined' || !document.body) return;
    const exposure = P.exposure ? Number(P.exposure.value) : 1.0;
    const glow = frameVals.f_glow;
    const edgeRgb = P.studioEdge?.rgb || [0.043, 0.063, 0.094];
    const acesByte = (linVal) => {
      const v = Math.max(0, linVal);
      const a = Math.min(1, Math.max(0, (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)));
      return Math.min(255, Math.max(0, Math.round(Math.pow(a, 1 / 2.2) * 255)));
    };
    const eLinR = Math.pow(Math.max(0, edgeRgb[0]), 2.2);
    const eLinG = Math.pow(Math.max(0, edgeRgb[1]), 2.2);
    const eLinB = Math.pow(Math.max(0, edgeRgb[2]), 2.2);
    const cR = acesByte((eLinR * 0.12 + glow[0] * 0.88) * exposure);
    const cG = acesByte((eLinG * 0.12 + glow[1] * 0.88) * exposure);
    const cB = acesByte((eLinB * 0.12 + glow[2] * 0.88) * exposure);
    const eR = acesByte(eLinR * exposure);
    const eG = acesByte(eLinG * exposure);
    const eB = acesByte(eLinB * exposure);
    document.body.style.backgroundImage = `radial-gradient(circle at 50% 50%, rgb(${cR},${cG},${cB}) 0%, rgb(${eR},${eG},${eB}) 72%)`;
    bodyBackdropApplied = true;
  }

  function clearBodyBackdrop() {
    if (!bodyBackdropApplied) return;
    if (typeof document !== 'undefined' && document.body) {
      document.body.style.backgroundImage = '';
    }
    bodyBackdropApplied = false;
  }

  function getTintSnapshot() {
    const stoneRgb = P.stoneColour?.rgb || [0.92, 0.96, 1.0];
    return {
      r: Math.round(175 + stoneRgb[0] * 65),
      g: Math.round(205 + stoneRgb[1] * 45),
      b: Math.round(230 + stoneRgb[2] * 25),
    };
  }

  function getCameraSnapshot() {
    return {
      baseYawDeg,
      spinDegPerSec: startupSpinDegPerSec,
      overrideYawDeg: userOverrodeCamera ? Number(P.spinStart?.value ?? baseYawDeg) : null,
      fovDeg: Number(P.fov?.value ?? 21),
      tiltDeg: Number(P.tilt?.value ?? 6),
      distance: Number(P.distance?.value ?? 2.15),
      swayDeg: Number(P.sway?.value ?? 0),
      swaySpeed: Number(P.swaySpeed?.value ?? 0.07),
      timeScale: Number(P.timeScale?.value ?? 1.5),
      centreX: cluster.centreX || 0,
      centreY: cluster.centreY || 0,
      centreZ: cluster.centreZ || 0,
    };
  }

  function computePhaseLockedYaw(atWallMs = wallNow()) {
    if (userOverrodeCamera) {
      return wrap360(Number(P.spinStart?.value ?? baseYawDeg));
    }
    const elapsedSec = Math.max(0, (atWallMs - bootEpochMs) * 0.001);
    return wrap360(baseYawDeg + startupSpinDegPerSec * elapsedSec);
  }

  function syncMainCameraYaw(atWallMs = wallNow()) {
    const yaw = computePhaseLockedYaw(atWallMs);
    if (P.spinStart && !userOverrodeCamera) {
      P.spinStart.value = yaw;
    }
    return yaw;
  }

  function terminateWireframeAndRelease() {
    const workerToStop = wfWorker;
    const urlToRevoke = wfWorkerUrl;
    const canvasToHide = wfCanvas;
    const wasTransferred = wfTransferred;
    wfWorker = null;
    wfWorkerUrl = null;
    usingOffscreenWorker = false;
    wfCtx = null;

    if (workerToStop) {
      workerToStop.onmessage = null;
      workerToStop.onerror = null;
      try { workerToStop.postMessage({ type: 'stop' }); } catch { /* ignore */ }
    }
    if (urlToRevoke) {
      try { URL.revokeObjectURL(urlToRevoke); } catch { /* ignore */ }
    }
    if (canvasToHide) {
      canvasToHide.style.opacity = '0';
      canvasToHide.style.display = 'none';
      if (!wasTransferred) {
        canvasToHide.width = 1;
        canvasToHide.height = 1;
      }
    }
    clearBodyBackdrop();
  }

  function startWireframeRenderer(initW, initH) {
    ensureWfCanvasElement();
    wfCanvas.style.display = 'block';
    wfCanvas.style.opacity = '1';
    const wfDpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const canOffscreen =
      !isCapture &&
      pinnedFillIn === null &&
      !sp.has('mainthreadwf') &&
      typeof wfCanvas.transferControlToOffscreen === 'function' &&
      typeof Worker !== 'undefined' &&
      typeof Blob !== 'undefined';

    if (canOffscreen) {
      try {
        const offscreen = wfCanvas.transferControlToOffscreen();
        wfTransferred = true;
        const blob = new Blob(['(' + wireframeWorkerMain.toString() + ')();'], { type: 'application/javascript' });
        wfWorkerUrl = URL.createObjectURL(blob);
        wfWorker = new Worker(wfWorkerUrl);
        usingOffscreenWorker = true;
        wfWorker.onerror = (err) => {
          console.error('wfWorker error:', err?.message || err);
        };
        wfWorker.onmessage = (ev) => {
          const msg = ev.data;
          if (!msg || msg.type !== 'tick') return;
          wireframeFrames = msg.frames;
          wireframeYawDeg = msg.yawDeg;
          if (firstWireframeMs === null) {
            firstWireframeNavMs = Number(performance.now().toFixed(2));
            firstWireframeMs = Number(Math.max(0.1, performance.now() - sketchStartPerf).toFixed(2));
          }
          if (wireframeLog.length < 400) {
            wireframeLog.push({
              frame: msg.frames,
              elapsedMs: Number(msg.elapsedMs.toFixed(2)),
              wallMs: msg.wallMs,
              yawDeg: Number(msg.yawDeg.toFixed(4)),
            });
          }
        };
        const bodies = cluster.getWireframe?.() || [];
        wfWorker.postMessage(
          {
            type: 'init',
            canvas: offscreen,
            bootEpochMs,
            width: initW,
            height: initH,
            dpr: wfDpr,
            bodies,
            camera: getCameraSnapshot(),
            tint: getTintSnapshot(),
          },
          [offscreen],
        );
        return;
      } catch {
        // Fall through cleanly to main-thread 2D canvas if OffscreenCanvas/Worker fails (e.g. CSP)
        usingOffscreenWorker = false;
        if (wfWorker) { wfWorker.terminate(); wfWorker = null; }
        if (wfWorkerUrl) { URL.revokeObjectURL(wfWorkerUrl); wfWorkerUrl = null; }
        ensureWfCanvasElement();
      }
    }

    wfCtx = wfCanvas.getContext('2d');
    drawWireframeMainThread(1.0);
  }

  // Graceful main-thread 2D canvas fallback (also used for deterministic ?capture=1&fillin=... snapshots)
  function drawWireframeMainThread(wireAlpha = 1.0) {
    if (usingOffscreenWorker) {
      if (wfCanvas) {
        wfCanvas.style.opacity = Math.max(0, Math.min(1, wireAlpha)).toFixed(4);
      }
      wfWorker?.postMessage({ type: 'renderNow' });
      return;
    }
    if (!wfCtx || !wfCanvas) return;
    if (wireAlpha <= 0.001) {
      if (wfCanvas.style.display !== 'none') wfCanvas.style.display = 'none';
      return;
    }
    if (wfCanvas.style.display === 'none') wfCanvas.style.display = 'block';
    wfCanvas.style.opacity = Math.max(0, Math.min(1, wireAlpha)).toFixed(4);

    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const cw = canvas.clientWidth || window.innerWidth || (isMobile ? 390 : 1280);
    const ch = canvas.clientHeight || window.innerHeight || (isMobile ? 844 : 800);
    const w = Math.max(1, Math.round(cw * dpr));
    const h = Math.max(1, Math.round(ch * dpr));
    if (wfCanvas.width !== w || wfCanvas.height !== h) {
      wfCanvas.width = w;
      wfCanvas.height = h;
    }
    wfCtx.clearRect(0, 0, w, h);

    const bodies = cluster.getWireframe?.() || [];
    if (!bodies.length) return;

    const halfW = w * 0.5, halfH = h * 0.5, minWH = Math.min(w, h);
    const camPos = frameVals.f_camPos;
    const cam = frameVals.f_cam;
    const focal = frameVals.f_focal[0] || 1.2;
    const camPos0 = camPos[0], camPos1 = camPos[1], camPos2 = camPos[2];
    const cam0 = cam[0], cam1 = cam[1], cam2 = cam[2];
    const cam3 = cam[3], cam4 = cam[4], cam5 = cam[5];
    const cam6 = cam[6], cam7 = cam[7], cam8 = cam[8];

    const keyX = -cam0 * 0.55 + cam3 * 0.65 - cam6 * 0.52;
    const keyY = -cam1 * 0.55 + cam4 * 0.65 - cam7 * 0.52;
    const keyZ = -cam2 * 0.55 + cam5 * 0.65 - cam8 * 0.52;

    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b];
      body.sphereZ = (body.cx - camPos0) * cam6 + (body.cy - camPos1) * cam7 + (body.cz - camPos2) * cam8;
    }
    const sorted = bodies.slice().sort((a, b) => b.sphereZ - a.sphereZ);
    const { r: tintR, g: tintG, b: tintB } = getTintSnapshot();

    wfCtx.save();
    wfCtx.lineCap = 'round';
    wfCtx.lineJoin = 'round';

    for (let b = 0; b < sorted.length; b++) {
      const body = sorted[b];
      const bAlpha = Math.min(1, body.alpha);
      if (bAlpha <= 0.01) continue;

      const verts = body.verts;
      const nv = body.nv;
      const scale = focal * minWH;
      for (let u = 0; u < nv; u++) {
        const rx = verts[u * 3] - camPos0;
        const ry = verts[u * 3 + 1] - camPos1;
        const rz = verts[u * 3 + 2] - camPos2;
        const cx = rx * cam0 + ry * cam1 + rz * cam2;
        const cy = rx * cam3 + ry * cam4 + rz * cam5;
        const cz = Math.max(0.08, rx * cam6 + ry * cam7 + rz * cam8);
        const invZ = scale / cz;
        projX[u] = halfW + cx * invZ;
        projY[u] = halfH - cy * invZ;
        projZ[u] = cz;
      }

      const normals = body.normals;
      const faces = body.faces;
      faceFront.fill(0);
      for (let f = 0; f < faces.length; f++) {
        const face = faces[f];
        const p = face.plane;
        const nx = normals[p * 3], ny = normals[p * 3 + 1], nz = normals[p * 3 + 2];
        const vx = face.cx - camPos0, vy = face.cy - camPos1, vz = face.cz - camPos2;
        const isFront = (nx * vx + ny * vy + nz * vz) < 0;
        faceFront[p] = isFront ? 1 : 0;
        const kDot = Math.max(0, nx * keyX + ny * keyY + nz * keyZ);
        faceLight[p] = kDot;

        if (isFront) {
          const idx = face.indices;
          wfCtx.beginPath();
          wfCtx.moveTo(projX[idx[0]], projY[idx[0]]);
          for (let m = 1; m < idx.length; m++) {
            wfCtx.lineTo(projX[idx[m]], projY[idx[m]]);
          }
          wfCtx.closePath();
          const fAlpha = (0.024 + 0.068 * Math.pow(kDot, 1.8)) * bAlpha;
          wfCtx.fillStyle = `rgba(${tintR},${tintG},${tintB},${fAlpha.toFixed(4)})`;
          wfCtx.fill();
        }
      }

      const edges = body.edges;

      wfCtx.beginPath();
      for (let e = 0; e < edges.length; e++) {
        const ed = edges[e];
        if (!faceFront[ed.pA] && !faceFront[ed.pB]) {
          wfCtx.moveTo(projX[ed.u], projY[ed.u]);
          wfCtx.lineTo(projX[ed.v], projY[ed.v]);
        }
      }
      wfCtx.strokeStyle = `rgba(${Math.round(tintR * 0.72)},${Math.round(tintG * 0.84)},${tintB},${(0.26 * bAlpha).toFixed(3)})`;
      wfCtx.lineWidth = 0.85 * dpr;
      wfCtx.stroke();

      wfCtx.beginPath();
      for (let e = 0; e < edges.length; e++) {
        const ed = edges[e];
        if (faceFront[ed.pA] && faceFront[ed.pB]) {
          wfCtx.moveTo(projX[ed.u], projY[ed.u]);
          wfCtx.lineTo(projX[ed.v], projY[ed.v]);
        }
      }
      wfCtx.strokeStyle = `rgba(${tintR},${tintG},${tintB},${(0.70 * bAlpha).toFixed(3)})`;
      wfCtx.lineWidth = 1.25 * dpr;
      wfCtx.stroke();

      wfCtx.beginPath();
      for (let e = 0; e < edges.length; e++) {
        const ed = edges[e];
        if (faceFront[ed.pA] !== faceFront[ed.pB]) {
          wfCtx.moveTo(projX[ed.u], projY[ed.u]);
          wfCtx.lineTo(projX[ed.v], projY[ed.v]);
        }
      }
      wfCtx.strokeStyle = `rgba(${Math.round(tintR * 0.78)},${Math.round(tintG * 0.92)},255,${(0.24 * bAlpha).toFixed(3)})`;
      wfCtx.lineWidth = 3.4 * dpr;
      wfCtx.stroke();

      wfCtx.strokeStyle = `rgba(236,247,255,${(0.92 * bAlpha).toFixed(3)})`;
      wfCtx.lineWidth = 1.6 * dpr;
      wfCtx.stroke();

      const vertHgt = body.vertHgt;
      wfCtx.fillStyle = `rgba(245,252,255,${(0.82 * bAlpha).toFixed(3)})`;
      const dotR = 1.55 * dpr;
      for (let u = 0; u < nv; u++) {
        if (vertHgt[u] >= 0.85) {
          wfCtx.beginPath();
          wfCtx.arc(projX[u], projY[u], dotR, 0, Math.PI * 2);
          wfCtx.fill();
        }
      }
    }

    wfCtx.restore();
    wireframeFrames++;
    wireframeYawDeg = Number(P.spinStart?.value ?? baseYawDeg);
    if (firstWireframeMs === null) {
      firstWireframeNavMs = Number(performance.now().toFixed(2));
      firstWireframeMs = Number(Math.max(0.1, performance.now() - sketchStartPerf).toFixed(2));
    }
    const nowWall = wallNow();
    if (wireframeLog.length < 400) {
      wireframeLog.push({
        frame: wireframeFrames,
        elapsedMs: Number(Math.max(0, nowWall - bootEpochMs).toFixed(2)),
        wallMs: nowWall,
        yawDeg: Number(wireframeYawDeg.toFixed(4)),
      });
    }
  }

  let sceneTime = 0, realTime = 0;
  function bind(pass, fbo, w, h) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(pass.prog);
    gl.uniform2f(pass.uRes, w, h);
    gl.uniform2f(pass.uScreen, canvas.width, canvas.height);
    gl.uniform1f(pass.uTime, sceneTime);
    gl.uniform1f(pass.uRealTime, realTime);
    if (pass.version !== model.version) {
      uploadParams(gl, pass.paramLocs, uniformParams);
      pass.version = model.version;
    }
    uploadFrame(gl, pass.frameLocs, frameList);
  }
  function drawPass(pass, fbo, w, h) {
    bind(pass, fbo, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function bake() {
    SHADER.fillNoiseVolume(noiseVol, P.seed.value, INCL);
    gl.activeTexture(gl.TEXTURE0 + UNIT_INCL);
    gl.bindTexture(gl.TEXTURE_3D, inclTex);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, INCL, INCL, INCL, gl.RGBA, gl.UNSIGNED_BYTE, noiseVol);
    bakedSeed = P.seed.value;
  }

  function renderRaytracedScene() {
    if (uploadedClusterVer !== cluster.version) {
      gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, cluster.data);
      uploadedClusterVer = cluster.version;
    }
    if (bakedSeed !== P.seed.value) bake();
    drawPass(passBackdrop, hdrFbo, hdrW, hdrH);
    const box = SHADER.traceBox;
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(box[0], box[1], box[2], box[3]);
    drawPass(passTrace, hdrFbo, hdrW, hdrH);
    gl.disable(gl.SCISSOR_TEST);
    if ((P.bloom?.value ?? 0.2) > 0.005) {
      drawPass(passBright, bloomAFbo, bloomW, bloomH);
      bind(passBlur, bloomBFbo, bloomW, bloomH);
      gl.uniform1i(passBlur.uSrc, UNIT_BLOOM_A);
      gl.uniform2f(passBlur.uDir, 1, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      bind(passBlur, bloomAFbo, bloomW, bloomH);
      gl.uniform1i(passBlur.uSrc, UNIT_BLOOM_B);
      gl.uniform2f(passBlur.uDir, 0, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    drawPass(passComposite, null, canvas.width, canvas.height);
  }

  function ensureReadySync() {
    holdWireframe = false;
    if (buildStage === 0) initGlResourcesAndBake();
    if (buildStage === 1) kickCompileAsync();
    if (buildStage === 2) finalizeCompiledPrograms();
    cleanupFence();
    fenceSignaled = true;
    shaderReady = true;
    buildStage = 5;
    fillInProgress = 1.0;
    wireframeAlpha = 0.0;
    startupPhaseActive = false;
    canvas.style.opacity = '1';
    terminateWireframeAndRelease();
  }

  function buildAllSync() {
    buildStage = 0;
    shaderReady = false;
    initGlResourcesAndBake();
    kickCompileAsync();
    finalizeCompiledPrograms();
    fenceSignaled = true;
    shaderReady = true;
    buildStage = 5;
    startupPhaseActive = false;
    canvas.style.opacity = '1';
    terminateWireframeAndRelease();
    ready = true;
  }

  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    cleanupFence();
    ready = false;
    shaderReady = false;
    fenceSignaled = false;
    buildStage = 0;
    gl = null;
    extParallel = null;
    vao = ubo = inclTex = hdrTex = hdrFbo = bloomA = bloomAFbo = bloomB = bloomBFbo = null;
    passBackdrop = passTrace = passBright = passBlur = passComposite = null;
    pendingPrograms = null;
    fillInProgress = 0.0;
    wireframeAlpha = 1.0;
    crossfadeStarted = false;
    startupPhaseActive = true;
    canvas.style.opacity = '0';
    applyStudioBackdropToBody();
    const cw = canvas.clientWidth || window.innerWidth || (isMobile ? 390 : 1280);
    const ch = canvas.clientHeight || window.innerHeight || (isMobile ? 844 : 800);
    startWireframeRenderer(cw, ch);
    degrade(0.6);
  });
  canvas.addEventListener('webglcontextrestored', () => {
    if (isCapture && !holdWireframe) {
      buildAllSync();
    } else {
      buildStage = 0;
      shaderReady = false;
      fenceSignaled = false;
      mainBootTicks = 0;
      fillInProgress = 0.0;
      wireframeAlpha = 1.0;
      crossfadeStarted = false;
      startupPhaseActive = true;
      ready = true;
      canvas.style.opacity = '0';
      applyStudioBackdropToBody();
      if (!usingOffscreenWorker && !wfCtx) {
        const cw = canvas.clientWidth || window.innerWidth || (isMobile ? 390 : 1280);
        const ch = canvas.clientHeight || window.innerHeight || (isMobile ? 844 : 800);
        startWireframeRenderer(cw, ch);
      }
    }
  });

  // Boot: in capture mode (?capture=1), build synchronously so frame 1 is immediately ray-traced;
  // otherwise keep #stage at opacity:0, paint studio radial backdrop, and launch the 60 FPS OffscreenCanvas Worker at t=0!
  ready = true;
  if (isCapture && !holdWireframe && pinnedFillIn === null) {
    buildAllSync();
  } else {
    const initW = canvas.clientWidth || window.innerWidth || (isMobile ? 390 : 1280);
    const initH = canvas.clientHeight || window.innerHeight || (isMobile ? 844 : 800);
    const initDpr = Math.min(isMobile ? 1.25 : 1.5, window.devicePixelRatio || 1);
    pxRatio = initDpr;
    buildCluster(P, initW, initH);
    clusterVersion = model.version;
    lastClusterW = initW;
    lastClusterH = initH;
    syncMainCameraYaw(bootEpochMs);
    SHADER.perFrame(P, 0, 0, Math.round(initW * initDpr), Math.round(initH * initDpr), initDpr, frameVals);
    applyStudioBackdropToBody();
    canvas.style.opacity = pinnedFillIn !== null ? cubicEaseInOut(pinnedFillIn).toFixed(4) : '0.001';
    startWireframeRenderer(initW, initH);
  }

  return {
    get shaderReady() { return shaderReady; },
    get startupPhaseActive() { return startupPhaseActive; },
    get usingOffscreenWorker() { return usingOffscreenWorker; },
    get userOverrodeCamera() { return userOverrodeCamera; },
    get fenceSignaled() { return fenceSignaled; },
    get fenceWaitMs() { return fenceWaitMs; },
    get wireframeAlpha() { return wireframeAlpha; },
    get fillInProgress() { return fillInProgress; },
    get firstWireframeMs() { return firstWireframeMs; },
    get firstWireframeNavMs() { return firstWireframeNavMs; },
    get shaderReadyMs() { return shaderReadyMs; },
    get wireframeCanvas() { return wfCanvas; },
    get wireframeFrames() { return wireframeFrames; },
    get wireframeYawDeg() { return wireframeYawDeg; },
    get startupSpinDegPerSec() { return startupSpinDegPerSec; },
    get wireframeLog() { return wireframeLog; },
    get floatTargets() { return floatTargets; },
    get inclEdge() { return INCL; },
    syncMainCameraYaw,
    notifyUserCameraOverride() {
      if (!userOverrodeCamera) {
        if (startupPhaseActive && P.spinStart) {
          P.spinStart.value = computePhaseLockedYaw(wallNow());
        }
        userOverrodeCamera = true;
      }
    },
    syncWorkerCamera() {
      if (wfWorker) {
        wfWorker.postMessage({
          type: 'camera',
          camera: getCameraSnapshot(),
          tint: getTintSnapshot(),
        });
      }
    },
    drawWireframe: (_bgAlpha = 1.0, wfAlpha = 1.0) => drawWireframeMainThread(wfAlpha),
    ensureReadySync,
    resize(_w, _h, px) {
      pxRatio = px;
      if (buildStage >= 1) sizeTargets();
      if (wfWorker) {
        const cw = canvas.clientWidth || window.innerWidth || (isMobile ? 390 : 1280);
        const ch = canvas.clientHeight || window.innerHeight || (isMobile ? 844 : 800);
        if (cw !== lastClusterW || ch !== lastClusterH) {
          buildCluster(P, cw, ch);
          clusterVersion = model.version;
          lastClusterW = cw;
          lastClusterH = ch;
        }
        wfWorker.postMessage({
          type: 'resize',
          width: cw,
          height: ch,
          dpr: Math.min(2, Math.max(1, window.devicePixelRatio || 1)),
          bodies: cluster.getWireframe?.() || [],
          camera: getCameraSnapshot(),
          tint: getTintSnapshot(),
        });
      }
    },
    frame(t, _dt, rt, rawDt = 0.016) {
      if (!ready) return;
      const nowWall = wallNow();
      const bootElapsedMs = Math.max(0, nowWall - bootEpochMs);
      const bootElapsedSec = bootElapsedMs * 0.001;
      sceneTime = startupPhaseActive ? bootElapsedSec * Number(P.timeScale?.value ?? 1.5) : t;
      realTime = rt;

      const cw = canvas.clientWidth || canvas.width;
      const ch = canvas.clientHeight || canvas.height;
      if (clusterVersion !== model.version || lastClusterW !== cw || lastClusterH !== ch) {
        buildCluster(P, cw, ch);
        clusterVersion = model.version;
        lastClusterW = cw;
        lastClusterH = ch;
        if (wfWorker) {
          wfWorker.postMessage({
            type: 'resize',
            width: cw,
            height: ch,
            dpr: Math.min(2, Math.max(1, window.devicePixelRatio || 1)),
            bodies: cluster.getWireframe?.() || [],
            camera: getCameraSnapshot(),
            tint: getTintSnapshot(),
          });
        }
      }

      if (startupPhaseActive && !userOverrodeCamera) {
        syncMainCameraYaw(nowWall);
      }

      if (holdWireframe) {
        fillInProgress = 0.0;
        wireframeAlpha = 1.0;
        if (!usingOffscreenWorker) {
          SHADER.perFrame(P, sceneTime, rt, canvas.width, canvas.height, pxRatio, frameVals);
          drawWireframeMainThread(1.0);
        }
        return;
      }

      if (pinnedFillIn !== null) {
        if (buildStage < 3) {
          initGlResourcesAndBake();
          kickCompileAsync();
          finalizeCompiledPrograms();
          shaderReady = true;
          fenceSignaled = true;
          buildStage = 5;
        }
        SHADER.perFrame(P, sceneTime, rt, canvas.width, canvas.height, pxRatio, frameVals);
        renderRaytracedScene();
        fillInProgress = pinnedFillIn;
        const ease = cubicEaseInOut(fillInProgress);
        wireframeAlpha = 1.0 - ease;
        canvas.style.opacity = ease.toFixed(4);
        drawWireframeMainThread(wireframeAlpha);
        return;
      }

      // Staggered non-blocking WebGL compilation + hidden PSO pre-warm + gl.fenceSync gate
      // while the OffscreenCanvas Worker rotates #wfCanvas at a locked 60 FPS!
      if (buildStage < 5) {
        if (!usingOffscreenWorker) {
          SHADER.perFrame(P, sceneTime, rt, canvas.width, canvas.height, pxRatio, frameVals);
          drawWireframeMainThread(1.0);
        }
        mainBootTicks++;
        if (mainBootTicks <= 1 || (usingOffscreenWorker && wireframeFrames < 2 && mainBootTicks < 12)) {
          // Ensure HTML HUD + OffscreenCanvas Worker isolate are live and spinning at 60 FPS
          // before touching WebGL2 context creation or shader compilation!
          return;
        }
        if (buildStage === 0) {
          initGlResourcesAndBake();
          return;
        }
        if (buildStage === 1) {
          kickCompileAsync();
          return;
        }
        if (buildStage === 2) {
          if (isParallelCompileComplete()) {
            finalizeCompiledPrograms();
          }
          return;
        }
        if (buildStage === 3) {
          // Render 1 hidden full-pipeline pre-warm pass (#stage still opacity: 0) across all 5 shaders
          // & real FBO attachments (scissored to 8x8 so 100% of driver PSO JIT finishes without shading
          // millions of hidden pixels) and insert gl.fenceSync!
          sizeTargets();
          SHADER.perFrame(P, sceneTime, rt, canvas.width, canvas.height, pxRatio, frameVals);
          if (uploadedClusterVer !== cluster.version) {
            gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
            gl.bufferSubData(gl.UNIFORM_BUFFER, 0, cluster.data);
            uploadedClusterVer = cluster.version;
          }
          if (bakedSeed !== P.seed.value) bake();
          gl.enable(gl.SCISSOR_TEST);
          gl.scissor(0, 0, 8, 8);
          drawPass(passBackdrop, hdrFbo, hdrW, hdrH);
          drawPass(passTrace, hdrFbo, hdrW, hdrH);
          if ((P.bloom?.value ?? 0.2) > 0.005) {
            drawPass(passBright, bloomAFbo, bloomW, bloomH);
            bind(passBlur, bloomBFbo, bloomW, bloomH);
            gl.uniform1i(passBlur.uSrc, UNIT_BLOOM_A);
            gl.uniform2f(passBlur.uDir, 1, 0);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
            bind(passBlur, bloomAFbo, bloomW, bloomH);
            gl.uniform1i(passBlur.uSrc, UNIT_BLOOM_B);
            gl.uniform2f(passBlur.uDir, 0, 1);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
          }
          drawPass(passComposite, null, canvas.width, canvas.height);
          gl.disable(gl.SCISSOR_TEST);
          prewarmSync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
          gl.flush();
          fenceCreatedWallMs = wallNow();
          buildStage = 4;
          return;
        }
        if (buildStage === 4) {
          if (!fenceSignaled) {
            if (!prewarmSync || gl.isContextLost()) {
              fenceSignaled = true;
            } else {
              const waitRes = gl.clientWaitSync(prewarmSync, gl.SYNC_FLUSH_COMMANDS_BIT, 0);
              const signaled =
                waitRes === gl.ALREADY_SIGNALED ||
                waitRes === gl.CONDITION_SATISFIED ||
                gl.getSyncParameter(prewarmSync, gl.SYNC_STATUS) === gl.SIGNALED;
              if (signaled || (nowWall - fenceCreatedWallMs) > FENCE_TIMEOUT_MS) {
                fenceSignaled = true;
                fenceWaitMs = Number(Math.max(0, nowWall - fenceCreatedWallMs).toFixed(2));
                cleanupFence();
              }
            }
          }
          if (!fenceSignaled || bootElapsedMs < minHoldMs) {
            return;
          }
          // Both >= 900ms wireframe hold AND GPU PSO fence are satisfied -> begin phase-locked crossfade!
          buildStage = 5;
          shaderReady = true;
          if (shaderReadyMs === null) {
            shaderReadyMs = Number(performance.now().toFixed(2));
          }
          crossfadeStarted = true;
          crossfadeStartWallMs = nowWall;
        }
      }

      // Phase-locked crossfade (~950ms cubic ease-in-out) -> live raytraced crystal
      if (fillInProgress < 1.0) {
        if (!crossfadeStarted) {
          crossfadeStarted = true;
          crossfadeStartWallMs = nowWall;
        }
        const fadeElapsedMs = isCapture
          ? (fillInProgress * CROSSFADE_DURATION_MS + Math.min(Math.max(rawDt || 0.016, 0.004), 0.10) * 1000)
          : Math.max(0, nowWall - crossfadeStartWallMs);
        fillInProgress = Math.min(1.0, fadeElapsedMs / CROSSFADE_DURATION_MS);
        const ease = cubicEaseInOut(fillInProgress);
        wireframeAlpha = 1.0 - ease;

        if (!userOverrodeCamera) {
          syncMainCameraYaw(nowWall);
        }
        SHADER.perFrame(P, sceneTime, rt, canvas.width, canvas.height, pxRatio, frameVals);
        renderRaytracedScene();

        if (fillInProgress < 1.0) {
          canvas.style.opacity = ease.toFixed(4);
          if (usingOffscreenWorker) {
            if (wfCanvas) wfCanvas.style.opacity = wireframeAlpha.toFixed(4);
          } else {
            drawWireframeMainThread(wireframeAlpha);
          }
          return;
        }

        // Crossfade just reached 1.0: hand off seamlessly to live raytracer & release #wfCanvas + Worker!
        wireframeAlpha = 0.0;
        canvas.style.opacity = '1';
        startupPhaseActive = false;
        terminateWireframeAndRelease();
        if (!handoffFired) {
          handoffFired = true;
          onStartupHandoff?.(startupSpinDegPerSec);
        }
        return;
      }

      SHADER.perFrame(P, t, rt, canvas.width, canvas.height, pxRatio, frameVals);
      renderRaytracedScene();
    },
  };
}
