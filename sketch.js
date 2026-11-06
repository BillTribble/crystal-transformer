// Crystals — WebGL2 renderer + instant (<30ms) 3D Quartz Crystal Wireframe first-paint
// that smoothly fills in with full ray-traced detail once async shader compilation completes.
// Built on Jack Bush's Geode renderer (jackbush/glimmers).

import SHADER from './shader.js';
import { buildCluster, cluster } from './cluster.js';
import { FULLSCREEN_VERT, frameUniforms, probeFloatTargets, program, shaderHeader, shaderParams, uploadFrame, uploadParams } from './kit/gl.js';

const UNIT_INCL = 0, UNIT_HDR = 1, UNIT_BLOOM_A = 2, UNIT_BLOOM_B = 3;
const BLOOM_DIVISOR = 4;
const FILL_IN_DURATION = 0.45;
const COMPLETION_STATUS_KHR = 0x91b1;

export default function create({ canvas, P, model, degrade }) {
  const sketchStartPerf = performance.now();
  const sp = new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
  const isCapture = sp.has('capture');
  const holdWireframe = sp.has('wireframe');
  const pinnedFillIn = sp.has('fillin') ? Math.max(0, Math.min(1, parseFloat(sp.get('fillin')))) : null;

  // Lightweight 2D overlay canvas for instant (<30ms) 3D Quartz Crystal Wireframe first-paint
  // Initialized before WebGL2 context creation so Frame 1 paints in <10ms without waiting for GPU driver init.
  let wfCanvas = document.getElementById('wireframe-stage');
  if (!wfCanvas) {
    wfCanvas = document.createElement('canvas');
    wfCanvas.id = 'wireframe-stage';
    wfCanvas.setAttribute('aria-hidden', 'true');
    wfCanvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;display:block;';
    canvas.parentElement?.insertBefore(wfCanvas, canvas.nextSibling);
  }
  const wfCtx = wfCanvas.getContext('2d');

  let gl = null;
  let extParallel = null;
  function ensureGl() {
    if (gl) return gl;
    gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    extParallel = gl.getExtension('KHR_parallel_shader_compile');
    return gl;
  }

  const uniformParams = shaderParams(model.params);
  const { vals: frameVals, list: frameList } = frameUniforms(SHADER.frameUniforms);
  const INCL = SHADER.inclSize;
  const noiseVol = new Uint8Array(INCL * INCL * INCL * 4);

  let ready = false, shaderReady = false, buildStage = 0, vao = null, ubo = null;
  let passBackdrop = null, passTrace = null, passBright = null, passBlurH = null, passBlurV = null, passComposite = null;
  let pendingPrograms = null;
  let inclTex = null, bakedSeed = NaN;
  let hdrTex = null, hdrFbo = null, bloomA = null, bloomAFbo = null, bloomB = null, bloomBFbo = null;
  let hdrW = 0, hdrH = 0, bloomW = 0, bloomH = 0, floatTargets = false;
  let clusterVersion = -1, uploadedClusterVer = -1, lastClusterW = 0, lastClusterH = 0, pxRatio = 1;
  let wireframeFrames = 0, fillInProgress = isCapture && !holdWireframe ? 1.0 : 0.0, wireframeAlpha = isCapture && !holdWireframe ? 0.0 : 1.0;
  let firstWireframeMs = null, firstWireframeNavMs = null, shaderReadyMs = null, fillStartPerfMs = 0;

  // Preallocated scratch buffers for 60 FPS 3D wireframe projection
  const projX = new Float32Array(128);
  const projY = new Float32Array(128);
  const projZ = new Float32Array(128);
  const faceFront = new Uint8Array(24);
  const faceLight = new Float32Array(24);

  function startCompileAsync(name, src, samplers) {
    const fragSrc = shaderHeader(uniformParams, frameList) + SHADER.common + src;
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
    const fsLog = !gl.getShaderParameter(fs, gl.COMPILE_STATUS) ? gl.getShaderInfoLog(fs) : '';
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error('Program "' + name + '" failed to link:\n' + (fsLog ? fsLog + '\n' : '') + gl.getProgramInfoLog(prog));
    }
    const loc = (n) => gl.getUniformLocation(prog, n);
    gl.useProgram(prog);
    for (const s in samplers) gl.uniform1i(loc(s), samplers[s]);
    return {
      prog,
      version: -1,
      uRes: loc('u_res'), uScreen: loc('u_screen'), uTime: loc('u_time'), uRealTime: loc('u_realTime'),
      uSlice: loc('u_slice'), uDir: loc('u_dir'),
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

  function initResourcesAndKickCompile() {
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

    floatTargets = probeFloatTargets(gl);
    hdrTex = texture2D(UNIT_HDR);
    bloomA = texture2D(UNIT_BLOOM_A);
    bloomB = texture2D(UNIT_BLOOM_B);
    hdrW = hdrH = bloomW = bloomH = 0;
    sizeTargets();
    hdrFbo = framebuffer(hdrTex);
    const hdrOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    bloomAFbo = framebuffer(bloomA);
    const bloomAOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    bloomBFbo = framebuffer(bloomB);
    const bloomBOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    if (floatTargets && (!hdrOk || !bloomAOk || !bloomBOk)) {
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
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    pendingPrograms = {
      backdrop: startCompileAsync('backdrop', SHADER.backdrop, {}),
      trace: startCompileAsync('trace', SHADER.trace, { u_incl: UNIT_INCL }),
      bright: startCompileAsync('bright', SHADER.bright, { u_hdr: UNIT_HDR }),
      blurH: startCompileAsync('blurH', SHADER.blur, { u_src: UNIT_BLOOM_A }),
      blurV: startCompileAsync('blurV', SHADER.blur, { u_src: UNIT_BLOOM_B }),
      composite: startCompileAsync('composite', SHADER.composite, { u_hdr: UNIT_HDR, u_bloomTex: UNIT_BLOOM_A }),
    };
    buildStage = 1;
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
    if (buildStage >= 2 || !pendingPrograms) return;
    passBackdrop = finishPass(pendingPrograms.backdrop);
    passTrace = finishPass(pendingPrograms.trace);
    passBright = finishPass(pendingPrograms.bright);
    passBlurH = finishPass(pendingPrograms.blurH);
    passBlurV = finishPass(pendingPrograms.blurV);
    passComposite = finishPass(pendingPrograms.composite);
    pendingPrograms = null;

    gl.useProgram(passBlurH.prog);
    gl.uniform2f(passBlurH.uDir, 1, 0);
    gl.useProgram(passBlurV.prog);
    gl.uniform2f(passBlurV.uDir, 0, 1);

    gl.uniformBlockBinding(passTrace.prog, gl.getUniformBlockIndex(passTrace.prog, 'Bodies'), 0);
    buildStage = 2;
    shaderReady = true;
    if (shaderReadyMs === null) {
      shaderReadyMs = Number(performance.now().toFixed(2));
    }
    fillStartPerfMs = performance.now();
  }

  function ensureReadySync() {
    if (buildStage === 0) initResourcesAndKickCompile();
    if (buildStage === 1) finalizeCompiledPrograms();
    if (!holdWireframe) {
      fillInProgress = 1.0;
      wireframeAlpha = 0.0;
      if (wfCanvas) wfCanvas.style.display = 'none';
    }
  }

  function buildAllSync() {
    buildStage = 0;
    shaderReady = false;
    initResourcesAndKickCompile();
    finalizeCompiledPrograms();
    ready = true;
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
    SHADER.fillNoiseVolume(noiseVol, P.seed.value);
    gl.activeTexture(gl.TEXTURE0 + UNIT_INCL);
    gl.bindTexture(gl.TEXTURE_3D, inclTex);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, INCL, INCL, INCL, gl.RGBA, gl.UNSIGNED_BYTE, noiseVol);
    bakedSeed = P.seed.value;
  }

  // Renders the exact 3D Quartz Crystal cluster polytope wireframe projected with f_cam, f_camPos, f_focal
  function drawWireframeOverlay(backdropAlpha, wireAlpha) {
    if (!wfCtx || !wfCanvas) return;
    if (backdropAlpha <= 0.001 && wireAlpha <= 0.001) {
      if (wfCanvas.style.display !== 'none') wfCanvas.style.display = 'none';
      return;
    }
    if (wfCanvas.style.display === 'none') wfCanvas.style.display = 'block';
    const dpr = Math.max(1, pxRatio || window.devicePixelRatio || 1);
    const w = (canvas.width > 300 ? canvas.width : Math.round((window.innerWidth || 1280) * dpr));
    const h = (canvas.height > 150 ? canvas.height : Math.round((window.innerHeight || 800) * dpr));
    if (wfCanvas.width !== w || wfCanvas.height !== h) {
      wfCanvas.width = w;
      wfCanvas.height = h;
    }
    wfCtx.clearRect(0, 0, w, h);

    const halfW = w * 0.5, halfH = h * 0.5, minWH = Math.min(w, h);

    // 1. Studio radial backdrop glow via compositor-accelerated CSS radial-gradient (matches SHADER.backdrop + composite)
    if (backdropAlpha > 0.001) {
      const glow = frameVals.f_glow;
      const edgeRgb = P.studioEdge?.rgb || [0.043, 0.063, 0.094];
      const cR = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[0] + glow[0] * 0.42), 0.58) * 255));
      const cG = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[1] + glow[1] * 0.42), 0.58) * 255));
      const cB = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[2] + glow[2] * 0.42), 0.58) * 255));
      const eR = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[0]), 0.65) * 255));
      const eG = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[1]), 0.65) * 255));
      const eB = Math.min(255, Math.round(Math.pow(Math.max(0, edgeRgb[2]), 0.65) * 255));
      const aStr = backdropAlpha.toFixed(3);
      wfCanvas.style.background = `radial-gradient(circle at 50% 50%, rgba(${cR},${cG},${cB},${aStr}) 0%, rgba(${eR},${eG},${eB},${aStr}) 72%)`;
    } else if (wfCanvas.style.background) {
      wfCanvas.style.background = 'transparent';
    }

    if (wireAlpha <= 0.001) return;

    const bodies = cluster.getWireframe?.() || [];
    if (!bodies.length) return;

    const camPos = frameVals.f_camPos;
    const cam = frameVals.f_cam;
    const focal = frameVals.f_focal[0] || 1.2;
    const camPos0 = camPos[0], camPos1 = camPos[1], camPos2 = camPos[2];
    const cam0 = cam[0], cam1 = cam[1], cam2 = cam[2];
    const cam3 = cam[3], cam4 = cam[4], cam5 = cam[5];
    const cam6 = cam[6], cam7 = cam[7], cam8 = cam[8];

    // Studio key light direction in world space
    const keyX = -cam0 * 0.55 + cam3 * 0.65 - cam6 * 0.52;
    const keyY = -cam1 * 0.55 + cam4 * 0.65 - cam7 * 0.52;
    const keyZ = -cam2 * 0.55 + cam5 * 0.65 - cam8 * 0.52;

    // Sort crystals back-to-front by camera-space Z
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b];
      body.sphereZ = (body.cx - camPos0) * cam6 + (body.cy - camPos1) * cam7 + (body.cz - camPos2) * cam8;
    }
    const sorted = bodies.slice().sort((a, b) => b.sphereZ - a.sphereZ);

    const stoneRgb = P.stoneColour?.rgb || [0.92, 0.96, 1.0];
    const tintR = Math.round(175 + stoneRgb[0] * 65);
    const tintG = Math.round(205 + stoneRgb[1] * 45);
    const tintB = Math.round(230 + stoneRgb[2] * 25);

    wfCtx.save();
    wfCtx.lineCap = 'round';
    wfCtx.lineJoin = 'round';

    for (let b = 0; b < sorted.length; b++) {
      const body = sorted[b];
      const bAlpha = Math.min(1, body.alpha) * wireAlpha;
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

        // Subtle translucent quartz facet wash on front-facing facets
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

      // Pass 1: Back-facing internal refracted crystal ridges
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

      // Pass 2: Front-facing crystallographic facet ridges
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

      // Pass 3: Outer 3D crystal silhouette edges (soft halo + crisp bright rim)
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

      // Pass 4: Subtle vertex catch-light glints on termination tips
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
    if (firstWireframeMs === null) {
      firstWireframeNavMs = Number(performance.now().toFixed(2));
      firstWireframeMs = Number(Math.max(0.1, performance.now() - sketchStartPerf).toFixed(2));
    }
  }

  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    ready = false;
    shaderReady = false;
    buildStage = 0;
    vao = ubo = inclTex = hdrTex = hdrFbo = bloomA = bloomAFbo = bloomB = bloomBFbo = null;
    passBackdrop = passTrace = passBright = passBlurH = passBlurV = passComposite = null;
    pendingPrograms = null;
    degrade(0.6);
  });
  canvas.addEventListener('webglcontextrestored', buildAllSync);

  // In capture mode (?capture=1), build synchronously so frame 1 is immediately ray-traced;
  // in interactive mode, render the 3D Quartz Wireframe immediately (<5ms) and compile shaders asynchronously!
  ready = true;
  if (isCapture && !holdWireframe) {
    buildAllSync();
  } else {
    const initW = window.innerWidth || 1280;
    const initH = window.innerHeight || 800;
    const initDpr = Math.min(2, window.devicePixelRatio || 1);
    pxRatio = initDpr;
    buildCluster(P, initW, initH);
    clusterVersion = model.version;
    lastClusterW = initW;
    lastClusterH = initH;
    SHADER.perFrame(P, 0, 0, Math.round(initW * initDpr), Math.round(initH * initDpr), initDpr, frameVals);
    drawWireframeOverlay(1.0, 1.0);
  }

  return {
    get shaderReady() { return shaderReady; },
    get wireframeAlpha() { return wireframeAlpha; },
    get fillInProgress() { return fillInProgress; },
    get firstWireframeMs() { return firstWireframeMs; },
    get firstWireframeNavMs() { return firstWireframeNavMs; },
    get shaderReadyMs() { return shaderReadyMs; },
    get wireframeCanvas() { return wfCanvas; },
    ensureReadySync,
    resize(_w, _h, px) {
      pxRatio = px;
      if (buildStage >= 1) sizeTargets();
    },
    frame(t, _dt, rt) {
      if (!ready) return;
      sceneTime = t;
      realTime = rt;
      const cw = canvas.clientWidth || canvas.width;
      const ch = canvas.clientHeight || canvas.height;
      if (clusterVersion !== model.version || lastClusterW !== cw || lastClusterH !== ch) {
        buildCluster(P, cw, ch);
        clusterVersion = model.version;
        lastClusterW = cw;
        lastClusterH = ch;
      }
      SHADER.perFrame(P, t, rt, canvas.width, canvas.height, pxRatio, frameVals);

      // Frame 1 (<30ms): Immediately draw the 3D Quartz Crystal Wireframe and yield to browser paint
      // before kicking off WebGL shader compilation on Frame 2!
      if (!shaderReady) {
        drawWireframeOverlay(1.0, 1.0);
        wireframeFrames++;
        if (holdWireframe) return;
        if (buildStage === 0 && wireframeFrames >= 2) {
          initResourcesAndKickCompile();
          return;
        }
        if (buildStage === 1 && wireframeFrames >= 3 && isParallelCompileComplete()) {
          finalizeCompiledPrograms();
        } else {
          return;
        }
      }

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
        drawPass(passBlurH, bloomBFbo, bloomW, bloomH);
        drawPass(passBlurV, bloomAFbo, bloomW, bloomH);
      }
      drawPass(passComposite, null, canvas.width, canvas.height);

      // Smooth 450ms fill-in crossfade from 3D Quartz Wireframe -> full ray-traced crystal
      if (holdWireframe) {
        fillInProgress = 0.0;
        wireframeAlpha = 1.0;
        drawWireframeOverlay(1.0, 1.0);
      } else if (pinnedFillIn !== null) {
        fillInProgress = pinnedFillIn;
        const pBg = Math.min(1.0, fillInProgress / 0.72);
        const bgEase = pBg * pBg * (3.0 - 2.0 * pBg);
        const backdropAlpha = 1.0 - bgEase;
        const pWf = Math.max(0.0, Math.min(1.0, (fillInProgress - 0.18) / 0.82));
        const wfEase = pWf * pWf * (3.0 - 2.0 * pWf);
        wireframeAlpha = 1.0 - wfEase;
        drawWireframeOverlay(backdropAlpha, wireframeAlpha);
      } else if (fillInProgress < 1.0) {
        const elapsed = Math.max(0, (performance.now() - fillStartPerfMs) * 0.001);
        fillInProgress = Math.min(1.0, elapsed / FILL_IN_DURATION);
        // Smooth cubic ease-in-out for ray-traced crystal reveal underneath the wireframe
        const pBg = Math.min(1.0, fillInProgress / 0.72);
        const bgEase = pBg * pBg * (3.0 - 2.0 * pBg);
        const backdropAlpha = 1.0 - bgEase;
        // Hold wireframe crisp while crystal fills in (0..22%), then smoothly dissolve wireframe out (22%..100%)
        const pWf = Math.max(0.0, Math.min(1.0, (fillInProgress - 0.18) / 0.82));
        const wfEase = pWf * pWf * (3.0 - 2.0 * pWf);
        wireframeAlpha = 1.0 - wfEase;
        drawWireframeOverlay(backdropAlpha, wireframeAlpha);
      } else if (wfCanvas && wfCanvas.style.display !== 'none') {
        wireframeAlpha = 0.0;
        wfCanvas.style.display = 'none';
      }
    },
  };
}
