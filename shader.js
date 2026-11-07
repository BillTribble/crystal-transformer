// Crystals — WebGL2 ray-tracing shaders + per-frame CPU values.
// Built on Jack Bush's Geode shader (jackbush/glimmers), extended with:
// - Direct interactive orbit camera (azimuth + elevation + zoom)
// - Bi-colour core-to-rim / base-to-tip absorption zoning (f_sigmaCore + u_biColour)
// - Multi-generation nested phantom growth shells (u_phantomLayers)
// - Stepped skeletal hopper crystal terraces (u_hopper)

import { FILM_GLSL } from './kit/film.js';
import { cluster, CRYSTAL_PLANES, MAX_CRYSTALS, PLANES_AT, ROW_W, ROWS } from './cluster.js';
import { MIRROR_MODES } from './params.js';

const INCL = 48;           // inclusion volume edge, texels (trilinear filtered, period 2..4)
const LIGHTS = 4;          // key, fill, rim left, rim right
const LIGHT_STRIDE = 14;   // centre 3, u 3, v 3, colour 3, half width, half height
const DEG = Math.PI / 180;
const TAU = Math.PI * 2;

const lin = (c) => Math.pow(c, 2.2);

// Smoothed backdrop glow colours (see perFrame).
const glowState = { init: false, rt: 0, glow: new Float64Array(3), comp: new Float64Array(3) };

// Scratch rotation matrices, row-major m[r * 3 + c].
const A = new Float64Array(9), B = new Float64Array(9), M = new Float64Array(9);
function set9(m, a, b, c, d, e, f, g, h, i) { m[0] = a; m[1] = b; m[2] = c; m[3] = d; m[4] = e; m[5] = f; m[6] = g; m[7] = h; m[8] = i; }
function rotX(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, 1, 0, 0, 0, c, -s, 0, s, c); }
function rotY(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, c, 0, s, 0, 1, 0, -s, 0, c); }
function rotZ(m, a) { const c = Math.cos(a), s = Math.sin(a); set9(m, c, -s, 0, s, c, 0, 0, 0, 1); }
const T9 = new Float64Array(9);
function mul(out, a, b) {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) T9[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  out.set(T9);
}

function light(f, i, az, el, hw, hh, P, power) {
  const o = i * LIGHT_STRIDE, L = f.f_lights;
  const cx = Math.sin(az) * Math.cos(el), cy = Math.sin(el), cz = Math.cos(az) * Math.cos(el);
  let ux = cz, uz = -cx;
  const ul = Math.hypot(ux, uz) || 1;
  ux /= ul; uz /= ul;
  const vx = cy * uz, vy = cz * ux - cx * uz, vz = -cy * ux;
  L[o] = cx; L[o + 1] = cy; L[o + 2] = cz;
  L[o + 3] = ux; L[o + 4] = 0; L[o + 5] = uz;
  L[o + 6] = vx; L[o + 7] = vy; L[o + 8] = vz;
  const rgb = P.lightColour.rgb;
  L[o + 9] = lin(rgb[0]) * power; L[o + 10] = lin(rgb[1]) * power; L[o + 11] = lin(rgb[2]) * power;
  L[o + 12] = hw * P.lightSize.value; L[o + 13] = hh * P.lightSize.value;
}

function hslChannel(n, h, s, l) {
  const k = (n + h * 12) % 12;
  return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
}

export default {
  inclSize: INCL,
  traceBox: new Int32Array(4),

  frameUniforms: {
    f_px: 1,
    f_frame: 1,
    f_seed: 1,
    f_camPos: 3,
    f_cam: 9,
    f_toWorld: 9,
    f_focal: 1,
    f_pixAngle: 1,
    f_crystals: 1,
    f_lights: [LIGHTS * LIGHT_STRIDE],
    f_ambient: 3,
    f_glow: 3,
    f_compGlow: 3,
    f_sigma: 3,
    f_sigmaCore: 3,
    f_mirrorMode: 1,
    u_isMobile: 1,
  },

  perFrame(P, t, rt, width, height, px, f) {
    f.f_px[0] = px;
    f.f_frame[0] = Math.floor(rt * P.filmFps.value);
    f.f_seed[0] = P.seed.value;
    f.f_crystals[0] = cluster.crystals;
    const mIdx = MIRROR_MODES.indexOf(String(P.mirrorMode?.value || 'Plain (None)'));
    f.f_mirrorMode[0] = mIdx > 0 ? mIdx : 0;
    const isMob = typeof window !== 'undefined' && (
      (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) ||
      /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent || '')
    );
    f.u_isMobile[0] = isMob ? 1.0 : 0.0;

    // Orbit turn: spinStart is updated both by interactive mouse/touch drag and by autoRotate in run.js
    const sw = P.sway.value * DEG, ss = TAU * P.swaySpeed.value * t;
    rotY(M, P.spinStart.value * DEG);
    rotZ(A, sw * Math.sin(ss));
    rotX(B, sw * 0.7 * Math.sin(ss * 1.618 + 1.3));
    mul(M, A, M);
    mul(M, B, M);
    rotX(A, P.tilt.value * DEG);
    mul(M, A, M);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) f.f_toWorld[c * 3 + r] = M[r * 3 + c];

    const fov = P.fov.value * DEG;
    const portraitZoom = width < height ? Math.min(1.36, Math.pow(height / Math.max(1, width), 0.38)) : 1.0;
    const focal = (0.5 / Math.tan(fov / 2)) * portraitZoom;
    const D = P.distance.value, R = cluster.radius;
    f.f_focal[0] = focal;
    f.f_pixAngle[0] = 1 / (focal * Math.min(width, height));
    const half = D > R * 1.01 ? (focal * R) / Math.sqrt(D * D - R * R) : 1e3;
    const reach = Math.ceil(half * Math.min(width, height)) + 2;
    const box = this.traceBox;
    if (f.f_mirrorMode[0] > 0.5) {
      box[0] = 0;
      box[1] = 0;
      box[2] = width;
      box[3] = height;
    } else {
      box[0] = Math.max(0, Math.floor(width / 2) - reach);
      box[1] = Math.max(0, Math.floor(height / 2) - reach);
      box[2] = Math.min(width, Math.floor(width / 2) + reach) - box[0];
      box[3] = Math.min(height, Math.floor(height / 2) + reach) - box[1];
    }
    for (let k = 0; k < 3; k++) {
      f.f_cam[k] = M[k];
      f.f_cam[3 + k] = M[3 + k];
      f.f_cam[6 + k] = -M[6 + k];
      const cOff = k === 0 ? (cluster.centreX || 0) : k === 1 ? (cluster.centreY || 0) : (cluster.centreZ || 0);
      f.f_camPos[k] = D * M[6 + k] + cOff;
    }

    const ka = P.keyAngle.value * DEG, kh = P.keyHeight.value * DEG, ra = P.rimAngle.value * DEG;
    light(f, 0, ka, kh, 0.42, 0.3, P, P.keyLight.value);
    light(f, 1, -ka * 0.85, kh * 0.3, 0.75, 0.55, P, P.fillLight.value);
    light(f, 2, -ra, 0.25, 0.07, 0.75, P, P.rimLight.value);
    light(f, 3, ra, 0.35, 0.07, 0.75, P, P.rimLight.value);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let i = 0; i < LIGHTS; i++) {
        const o = i * LIGHT_STRIDE;
        sum += f.f_lights[o + 9 + c] * f.f_lights[o + 12] * f.f_lights[o + 13];
      }
      f.f_ambient[c] = sum * 0.25;
    }

    const s = P.stoneColour.rgb;
    const sc = P.coreColour ? P.coreColour.rgb : s;
    const mx = Math.max(s[0], s[1], s[2]), mn = Math.min(s[0], s[1], s[2]), d = mx - mn;
    const stoneSat = mx > 1e-5 ? d / mx : 0;
    const isNeutralStone = stoneSat < 0.22 || (P.colourDepth && P.colourDepth.value < 0.45);
    let h = 0.60;
    if (!isNeutralStone && d > 1e-5) {
      h = mx === s[0] ? ((s[1] - s[2]) / d) % 6 : mx === s[1] ? (s[2] - s[0]) / d + 2 : (s[0] - s[1]) / d + 4;
      h = ((h / 6) % 1 + 1) % 1;
      const isPurpleStone = h >= 0.68 && h <= 0.86;
      const defaultCompShift = isPurpleStone ? 268 : 180;
      const shiftDeg = (P.graduallyShift && P.graduallyShift.value === false)
        ? P.hueShift.value
        : (Math.abs(P.hueShift.value - 180) > 1 ? P.hueShift.value : defaultCompShift);
      h = (((h + shiftDeg / 360) % 1) + 1) % 1;
    }
    const effBackdropSat = isNeutralStone ? Math.min(P.backdropSat.value, 0.24) : P.backdropSat.value;
    const hComp = (h + 0.5) % 1;
    // Ease the backdrop glow colours over ~1 s of real time. The stone hue is numerically unstable
    // for near-white/grey stones (e.g. High-Dispersion Diamond, where 1-step RGB changes swap the
    // max channel), which used to flash the whole background between red-brown and olive.
    let dtR = rt - glowState.rt;
    if (!(dtR >= 0)) dtR = 0.016;
    glowState.rt = rt;
    const aG = (glowState.init && (!P.morphPresets || P.morphPresets.value !== false))
      ? 1 - Math.exp(-Math.min(dtR, 0.5) / 1.1)
      : 1;
    glowState.init = true;
    for (let c = 0; c < 3; c++) {
      const tg = lin(hslChannel(c === 0 ? 0 : c === 1 ? 8 : 4, h, effBackdropSat, 0.5)) * P.backdropLight.value;
      const tc = lin(hslChannel(c === 0 ? 0 : c === 1 ? 8 : 4, hComp, Math.max(0.58, effBackdropSat), 0.52)) * Math.max(0.11, P.backdropLight.value * 1.35);
      glowState.glow[c] += (tg - glowState.glow[c]) * aG;
      glowState.comp[c] += (tc - glowState.comp[c]) * aG;
      f.f_glow[c] = glowState.glow[c];
      f.f_compGlow[c] = glowState.comp[c];
      f.f_sigma[c] = -Math.log(Math.max(lin(s[c]), 0.002)) * P.colourDepth.value;
      f.f_sigmaCore[c] = -Math.log(Math.max(lin(sc[c]), 0.002)) * P.colourDepth.value;
    }
  },

  common: `
#define PI  3.14159265359
#define TAU 6.28318530718
${FILM_GLSL}
#define LIGHTS ${LIGHTS}
#define LS ${LIGHT_STRIDE}
#define u_mirrorMode int(f_mirrorMode + 0.5)
vec3 lin(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 backdrop(vec3 d) {
  float ang = acos(clamp(-d.z, -1.0, 1.0));
  float gRatio = ang / max(radians(u_glowSize), 1e-4);
  float g = exp(-gRatio * gRatio);
  return mix(lin(u_studioEdge), f_glow, g);
}
vec3 lights(vec3 d, float rough) {
  vec3 sum = vec3(0.0);
  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 c = vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]);
    float cs = dot(d, c);
    if (cs <= 0.15) continue;
    float hw = f_lights[o + 12], hh = f_lights[o + 13];
    float s = max(u_lightSoftness * min(hw, hh) + rough * 0.6, 1e-3);
    vec3 p = d / cs;
    float au = abs(dot(p, vec3(f_lights[o + 3], f_lights[o + 4], f_lights[o + 5])));
    if (au >= hw + s) continue;
    float av = abs(dot(p, vec3(f_lights[o + 6], f_lights[o + 7], f_lights[o + 8])));
    if (av >= hh + s) continue;
    float m = (1.0 - smoothstep(hw - s, hw + s, au)) * (1.0 - smoothstep(hh - s, hh + s, av));
    m *= 1.0 - 0.6 * clamp(dot(vec2(au / hw, av / hh), vec2(au / hw, av / hh)), 0.0, 1.0);
    m *= hw * hh / ((hw + s) * (hh + s));
    sum += m * vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
  }
  return sum;
}
vec3 env(vec3 dc, float rough) {
  vec3 d = f_toWorld * dc;
  return backdrop(d) + lights(d, rough);
}
float glimmerCurve(float g) {
  float g01 = clamp(g, 0.0, 1.0);
  return smoothstep(0.0, 0.45, g01) * mix(0.85, 1.0, smoothstep(0.45, 1.0, g01)) + max(0.0, g - 1.0);
}
// High-contrast analytic refraction strip lights around the refraction cone behind the cluster (active when u_glimmer > 0)
// Narrow, crisp studio rim strips (narrower than the RGB dispersion spread) so refracted wavelengths separate cleanly
// into focused jewel flashes (cyan, sapphire blue, warm gold, crimson-magenta) inside dark facets without white glare slabs.
vec3 glimmerStripLights(vec3 d, float rough) {
  float g = clamp(u_glimmer, 0.0, 2.0);
  if (g <= 0.001 || d.z >= -0.05) return vec3(0.0);
  float effG = glimmerCurve(g);
  vec3 lCol = vec3(f_lights[2 * LS + 9], f_lights[2 * LS + 10], f_lights[2 * LS + 11]);
  vec3 sum = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec3 c = (i == 0) ? vec3(-0.51449576,  0.19788302, -0.83424024)
           : (i == 1) ? vec3( 0.49811675, -0.17828389, -0.84857699)
           : (i == 2) ? vec3(-0.33864256, -0.41286135, -0.84549780)
                      : vec3( 0.35486045,  0.39644465, -0.84669103);
    float cs = dot(d, c);
    if (cs <= 0.35) continue;
    vec3 p = d / cs;
    vec3 uAx = normalize(vec3(c.z, 0.0, -c.x));
    vec3 vAx = cross(c, uAx);
    if (i >= 2) {
      float ca = 0.80;
      float sa = ((i & 1) == 0) ? 0.60 : -0.60;
      vec3 uRot = uAx * ca + vAx * sa;
      vAx = -uAx * sa + vAx * ca;
      uAx = uRot;
    }
    float hw = (i < 2) ? 0.025 : 0.020;
    float hh = (i < 2) ? 0.44 : 0.34;
    float s = max(u_lightSoftness * 0.50 * min(hw, hh) + rough * 0.40, 1e-3);
    float au = abs(dot(p, uAx));
    if (au >= hw + s) continue;
    float av = abs(dot(p, vAx));
    if (av >= hh + s) continue;
    float m = (1.0 - smoothstep(hw - s, hw + s, au)) * (1.0 - smoothstep(hh - s, hh + s, av));
    float vTaper = 1.0 - 0.65 * clamp((av / hh) * (av / hh), 0.0, 1.0);
    m *= (1.0 - 0.50 * clamp((au / hw) * (au / hw), 0.0, 1.0)) * vTaper;
    m *= hw * hh / ((hw + s) * (hh + s));
    sum += m * lCol * 0.40;
  }
  return sum * effG;
}
vec3 spectralRainbow(float t) {
  vec3 x = fract(t + vec3(0.0, 0.66667, 0.33333));
  vec3 wave = clamp(abs(x * 6.0 - 3.0) - 1.0, 0.0, 1.0);
  return wave * wave * (3.0 - 2.0 * wave);
}
vec4 encodeHdr(vec3 c) { return vec4(c / (1.0 + c), 1.0); }
vec3 decodeHdr(vec4 e) { return e.rgb / max(1.0 - e.rgb, 1e-4); }
`,

  fillNoiseVolume(vol, seed) {
    const N = INCL;
    const invN = 1.0 / N;
    const s = (Math.floor((Number(seed) || 1) * 1000) ^ 0x9e3779b9) >>> 0;
    const gSize = 16;
    const g = new Float32Array(gSize * gSize * gSize);
    let st = (s + 0x6d2b79f5) >>> 0;
    for (let i = 0; i < g.length; i++) {
      st = (Math.imul(st ^ (st >>> 15), 0x85ebca6b) + 0x6d2b79f5) >>> 0;
      g[i] = ((st >>> 8) * (1.0 / 8388608.0)) - 1.0;
    }
    function samplePeriodic(x, y, z, period) {
      const pm = period - 1;
      const ix0 = Math.floor(x), iy0 = Math.floor(y), iz0 = Math.floor(z);
      const fx = x - ix0, fy = y - iy0, fz = z - iz0;
      const ux = fx * fx * (3.0 - 2.0 * fx);
      const uy = fy * fy * (3.0 - 2.0 * fy);
      const uz = fz * fz * (3.0 - 2.0 * fz);
      const x0 = ix0 & pm, y0 = (iy0 & pm) << 4, z0 = (iz0 & pm) << 8;
      const x1 = (ix0 + 1) & pm, y1 = ((iy0 + 1) & pm) << 4, z1 = ((iz0 + 1) & pm) << 8;
      const c00 = g[z0 + y0 + x0] + (g[z0 + y0 + x1] - g[z0 + y0 + x0]) * ux;
      const c10 = g[z0 + y1 + x0] + (g[z0 + y1 + x1] - g[z0 + y1 + x0]) * ux;
      const c01 = g[z1 + y0 + x0] + (g[z1 + y0 + x1] - g[z1 + y0 + x0]) * ux;
      const c11 = g[z1 + y1 + x0] + (g[z1 + y1 + x1] - g[z1 + y1 + x0]) * ux;
      const c0 = c00 + (c10 - c00) * uy;
      const c1 = c01 + (c11 - c01) * uy;
      return c0 + (c1 - c0) * uz;
    }
    let idx = 0;
    let rng = (s ^ 0x27d4eb2d) >>> 0;
    for (let z = 0; z < N; z++) {
      const pz = (z + 0.5) * invN;
      for (let y = 0; y < N; y++) {
        const py = (y + 0.5) * invN;
        for (let x = 0; x < N; x++) {
          const px = (x + 0.5) * invN;
          const n2 = samplePeriodic(px * 2.0 + 9.0, py * 2.0 + 9.0, pz * 2.0 + 9.0, 2);
          const n4 = samplePeriodic(px * 4.0, py * 4.0, pz * 4.0, 4);
          const n8 = samplePeriodic(px * 8.0, py * 8.0, pz * 8.0, 8);
          const m = 0.58 * n4 + 0.30 * n8;
          const mc = Math.max(0.0, Math.min(1.0, (m + 0.05) * 1.6));
          const milk = mc * Math.sqrt(mc);
          const sn = Math.abs(samplePeriodic(px * 4.0 + n4 * 0.78, py * 4.0 + n2 * 0.78, pz * 4.0 - n4 * 0.78, 4));
          const sheet = sn >= 0.045 ? 0.0 : (1.0 - sn / 0.045);
          const pt = Math.max(0.0, Math.min(1.0, ((0.55 * n2 + 0.28 * n4) + 0.05) * (1.0 / 0.30)));
          const patchy = pt * pt * (3.0 - 2.0 * pt);
          rng = (Math.imul(rng ^ (rng >>> 15), 0x85ebca6b) + 0x6d2b79f5) >>> 0;
          const bubbles = (rng & 255) > 153 ? 1.0 : 0.25;
          const veil = sheet * patchy * bubbles;
          const mottle = Math.max(0.0, Math.min(1.0, (0.52 * n2 + 0.26 * n4 + 0.13 * n8) * 0.72 + 0.5));
          const sparkle = (rng >>> 8) & 255;
          vol[idx++] = Math.min(255, Math.max(0, (milk * 255.0 + 0.5) | 0));
          vol[idx++] = Math.min(255, Math.max(0, (veil * 255.0 + 0.5) | 0));
          vol[idx++] = Math.min(255, Math.max(0, (mottle * 255.0 + 0.5) | 0));
          vol[idx++] = sparkle;
        }
      }
    }
  },

  backdrop: `
out vec4 outColor;
void main() {
  vec2 q = (gl_FragCoord.xy - 0.5 * u_res) / min(u_res.x, u_res.y);
  outColor = encodeHdr(env(normalize(f_cam * vec3(q, f_focal)), 0.0));
}`,

  trace: `
#define MAX_CRYSTALS ${MAX_CRYSTALS}
#define CP ${CRYSTAL_PLANES}
#define PA ${PLANES_AT}
layout(std140) uniform Bodies { vec4 bodies[${ROW_W * ROWS}]; };
uniform highp sampler3D u_incl;
out vec4 outColor;

vec4 body(int row, int k) { return bodies[row * ${ROW_W} + k]; }

bool sphereHit(vec3 ro, vec3 rd, vec4 s, float tMax) {
  vec3 oc = ro - s.xyz;
  float b = dot(oc, rd), h = b * b - dot(oc, oc) + s.w * s.w;
  if (h < 0.0) return false;
  h = sqrt(h);
  return -b + h > 0.0 && -b - h < tMax;
}

// Stable occlusion partner for a crystal: one of the three permanent core crystals (uids 1..3 always
// occupy rows 0..2), chosen by the crystal's stable uid. The old (row + 1) % activeCount re-targeted
// every void the instant any crystal started or stopped drawing (or rows shifted), so big chunks of
// crystals popped in and out of existence.
int occPartnerRow(int row, int m) {
  int nCore = clamp(int(f_crystals + 0.5), 1, 3);
  int u = int(max(1.0, body(row, 2).z) + 0.5);
  return (u + m) % nCore;
}

// Occlusion voids GROW into existence and shrink away: their size scales with the occlusion level
// (0 -> full size) and their cut opacity ramps in early so a sub-pixel void can't twinkle.
// (Previously a full-size void cross-dissolved in and out, which read as popping.)
// Each crystal's void starts at a staggered moment in the first ~45% of the fade (by stable uid),
// then eases to full size: the voids are big next to a thin crystal, so if they all grew at once
// the whole cluster was eaten within ~4 s; staggered, the occlusion spreads over the whole fade
// and recedes in the reverse order.
float occGrowthFor(int row) {
  float x = clamp(u_occlusion / 0.92, 0.0, 1.0);
  float delay = 0.45 * fract(max(1.0, body(row, 2).z) * 0.6180339887);
  float y = clamp((x - delay) / (1.0 - delay), 0.0, 1.0);
  return y * y * (3.0 - 2.0 * y);
}
float occOpacityFor(int row) {
  float x = clamp(u_occlusion / 0.92, 0.0, 1.0);
  float delay = 0.45 * fract(max(1.0, body(row, 2).z) * 0.6180339887);
  return smoothstep(0.0, 0.25, (x - delay) / (1.0 - delay));
}

vec3 rotateAroundAxis(vec3 v, vec3 k, float cosA, float sinA) {
  return v * cosA + cross(k, v) * sinA + k * dot(k, v) * (1.0 - cosA);
}

// 1a. 3D CSG Ovoid + Wedge Void Interval (u_occlusionType == 0):
//     Computes ray interval [tc0, tc1], entry/exit 3D cavity wall normals, and interior crease distance exitSeam.
bool evalOvoidVoidInterval(int row, int m, vec3 ro, vec3 rd, out float tc0, out float tc1, out vec3 enterN, out vec3 exitN, out float exitSeam) {
  tc0 = -1e9; tc1 = 1e9;
  enterN = vec3(0.0, 1.0, 0.0);
  exitN = vec3(0.0, 1.0, 0.0);
  exitSeam = 1e9;
  if (u_occlusion <= 0.002) return false;
  float g = occGrowthFor(row) * smoothstep(0.55, 0.95, clamp(body(row, 2).x, 0.0, 1.0));
  if (g <= 1e-3) return false;

  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float Lh = max(body(row, 1).w, 0.08);
  float rh = max(body(row, 2).w, 0.02);
  vec3 p0 = body(row, PA).xyz;
  vec3 side1 = p0 - ax * dot(p0, ax);
  float sLen = length(side1);
  side1 = sLen > 1e-4 ? (side1 / sLen) : normalize(cross(ax, vec3(0.577, 0.577, 0.577)));
  vec3 side2 = cross(ax, side1);

  float uid = max(1.0, body(row, 2).z);
  float azV = uid * 2.399963 + float(m) * 2.65;
  vec3 outDir = side1 * cos(azV) + side2 * sin(azV);
  float hFrac = (m == 0) ? 0.42 : 0.72;
  vec3 axPt = rt + ax * (Lh * hFrac);
  float rad = rh * (1.32 + 0.24 * float(m)) * g;
  vec4 ec = vec4(axPt + outDir * (rh * 1.04), rad);
  if (ec.w <= 1e-4) return false;

  vec3 oc = ro - ec.xyz;
  float b = dot(oc, rd);
  float h = b * b - dot(oc, oc) + ec.w * ec.w;
  if (h <= 0.0) return false;
  float s = sqrt(h);
  tc0 = -b - s;
  tc1 = -b + s;
  int enterSurf = 0;
  int exitSurf = 0;

  vec3 nUp = normalize(ax + outDir * 0.44);
  vec3 nDn = normalize(-ax + outDir * 0.48);
  float dUp = ec.w * 0.66 - dot(nUp, oc);
  float dnUp = dot(nUp, rd);
  if (abs(dnUp) > 1e-6) {
    float tUp = dUp / dnUp;
    if (dnUp < 0.0) {
      if (tUp > tc0) { tc0 = tUp; enterSurf = 1; }
    } else {
      if (tUp < tc1) { tc1 = tUp; exitSurf = 1; }
    }
  } else if (dUp < 0.0) return false;

  float dDn = ec.w * 0.68 - dot(nDn, oc);
  float dnDn = dot(nDn, rd);
  if (abs(dnDn) > 1e-6) {
    float tDn = dDn / dnDn;
    if (dnDn < 0.0) {
      if (tDn > tc0) { tc0 = tDn; enterSurf = 2; }
    } else {
      if (tDn < tc1) { tc1 = tDn; exitSurf = 2; }
    }
  } else if (dDn < 0.0) return false;

  if (tc1 <= tc0) return false;

  vec3 rel0 = (ro + rd * tc0) - ec.xyz;
  enterN = (enterSurf == 1) ? nUp : ((enterSurf == 2) ? nDn : normalize(rel0 + 1e-6));
  vec3 rel1 = (ro + rd * tc1) - ec.xyz;
  exitN = (exitSurf == 1) ? nUp : ((exitSurf == 2) ? nDn : normalize(rel1 + 1e-6));

  float dSph = max(0.0, ec.w - length(rel1));
  float dU = max(0.0, ec.w * 0.66 - dot(nUp, rel1));
  float dD = max(0.0, ec.w * 0.68 - dot(nDn, rel1));
  exitSeam = (exitSurf == 0) ? min(dU, dD) : ((exitSurf == 1) ? min(dSph, dD) : min(dSph, dU));
  return true;
}

// 1b. 3D CSG Invisible Crystal Void Interval (u_occlusionType == 1):
//     Carves a 3D faceted crystal cavity of the exact 18-plane crystallographic habit of partner tRow.
bool evalCrystalVoidInterval(int row, int m, vec3 ro, vec3 rd, bool needSeam, out float tc0, out float tc1, out vec3 enterN, out vec3 exitN, out float exitSeam) {
  tc0 = -1e9; tc1 = 1e9;
  enterN = vec3(0.0, 1.0, 0.0);
  exitN = vec3(0.0, 1.0, 0.0);
  exitSeam = 1e9;
  if (u_occlusion <= 0.002) return false;
  float g = occGrowthFor(row) * smoothstep(0.55, 0.95, clamp(body(row, 2).x, 0.0, 1.0));
  if (g <= 1e-3) return false;

  vec3 axRow = body(row, 1).xyz;
  float LRow = max(body(row, 1).w, 0.08);
  float rRow = max(body(row, 2).w, 0.02);
  vec3 rtRow = body(row, 3).xyz;
  vec3 p0 = body(row, PA).xyz;
  vec3 side1 = p0 - axRow * dot(p0, axRow);
  float sLen = length(side1);
  side1 = sLen > 1e-4 ? (side1 / sLen) : normalize(cross(axRow, vec3(0.577, 0.577, 0.577)));
  vec3 side2 = cross(axRow, side1);

  int tRow = occPartnerRow(row, m);
  vec3 axTpl = body(tRow, 1).xyz;
  float LTpl = max(body(tRow, 1).w, 0.08);
  float rTpl = max(body(tRow, 2).w, 0.02);
  vec3 rtTpl = body(tRow, 3).xyz;
  vec3 cTpl = rtTpl + axTpl * (LTpl * 0.48);

  float uid = max(1.0, body(row, 2).z);
  float azV = uid * 2.399963 + float(m) * 2.45 + 0.55;
  vec3 outDir = side1 * cos(azV) + side2 * sin(azV);
  vec3 tanDir = -side1 * sin(azV) + side2 * cos(azV);
  float hFrac = (m == 0) ? 0.40 : 0.70;
  vec3 cAnchor = rtRow + axRow * (LRow * hFrac) + outDir * (rRow * 0.94);
  vec3 rotAxis = normalize(tanDir + axRow * ((m == 0) ? 0.28 : -0.24));
  float ang = (m == 0) ? 1.12 : -1.02;
  float cosA = cos(ang), sinA = sin(ang);
  float invScale = clamp((rTpl / max(rRow, 0.02)) * 0.76, 0.48, 1.28) / max(g, 0.02);

  vec3 roTpl = cTpl + rotateAroundAxis((ro - cAnchor) * invScale, rotAxis, cosA, sinA);
  vec3 rdTpl = rotateAroundAxis(rd * invScale, rotAxis, cosA, sinA);

  vec3 enterPlN = vec3(0.0, 1.0, 0.0);
  vec3 exitPlN = vec3(0.0, 1.0, 0.0);
  int kExit = 0;
  for (int k = 0; k < 18; k++) {
    vec4 pl = body(tRow, PA + k);
    float dn = dot(pl.xyz, rdTpl);
    float dist = pl.w - dot(pl.xyz, roTpl);
    if (abs(dn) < 1e-8) {
      if (dist < 0.0) return false;
      continue;
    }
    float tPl = dist / dn;
    if (dn < 0.0) {
      if (tPl > tc0) { tc0 = tPl; enterPlN = pl.xyz; }
    } else if (tPl < tc1) {
      tc1 = tPl; exitPlN = pl.xyz; kExit = k;
    }
    if (tc0 > tc1) return false;
  }
  if (tc1 <= tc0) return false;

  enterN = normalize(rotateAroundAxis(enterPlN, rotAxis, cosA, -sinA) + 1e-6);
  exitN = normalize(rotateAroundAxis(exitPlN, rotAxis, cosA, -sinA) + 1e-6);

  if (needSeam) {
    vec3 pTplExit = roTpl + rdTpl * tc1;
    float dMinSeam = 1e9;
    for (int k = 0; k < 18; k++) {
      if (k == kExit) continue;
      vec4 pl = body(tRow, PA + k);
      float rawD = max(0.0, pl.w - dot(pl.xyz, pTplExit));
      if (rawD >= dMinSeam * invScale) continue;
      float cosAng = dot(pl.xyz, exitPlN);
      if (cosAng > 0.995) continue;
      float sinAng = sqrt(max(1e-3, 1.0 - cosAng * cosAng));
      float dPl = rawD / (sinAng * invScale);
      if (dPl < dMinSeam) dMinSeam = dPl;
    }
    exitSeam = dMinSeam;
  }
  return true;
}

bool evalSingleVoidInterval(int row, int m, vec3 ro, vec3 rd, bool needSeam, out float tc0, out float tc1, out vec3 enterN, out vec3 exitN, out float exitSeam) {
  float kType = clamp(u_occlusionType, 0.0, 1.0);
  bool useCrystal = (kType >= 0.85) || (kType > 0.15 && m == 0);
  if (useCrystal) {
    return evalCrystalVoidInterval(row, m, ro, rd, needSeam, tc0, tc1, enterN, exitN, exitSeam);
  }
  return evalOvoidVoidInterval(row, m, ro, rd, tc0, tc1, enterN, exitN, exitSeam);
}

// Unified 3D CSG Occlusion Void Carve:
// Subtracts 3D ovoid/crystal void volumes [tc0, tc1] from crystal interval [tn, tf], advancing tn to tc1
// when a cavity is carved into the front face (with true 3D interior wall normal outVoidN and crisp 3D
// border distance outVoidBorder) or returning false when the void slices completely through the silhouette.
bool applyOcclusionVoidCSG(int row, vec3 ro, vec3 rd, inout float tn, inout float tf, inout int kn, out vec3 outVoidN, out float outVoidBorder, out float outVoidSeam) {
  outVoidN = (kn < CP) ? body(row, PA + kn).xyz : vec3(0.0, 1.0, 0.0);
  outVoidBorder = 1e9;
  outVoidSeam = 1e9;
  if (u_occlusion <= 0.002) return true;
  float rRow = max(body(row, 2).w, 0.02);
  float tnOrig = max(tn, 0.0);
  float tfOrig = tf;

  for (int m = 0; m < 2; m++) {
    float tc0, tc1, exitSeam;
    vec3 enterN, exitN;
    if (!evalSingleVoidInterval(row, m, ro, rd, true, tc0, tc1, enterN, exitN, exitSeam)) continue;

    float tStart = max(tn, 0.0);
    if (tc0 <= tStart + rRow * 0.06 && tc1 > tStart) {
      if (tc1 >= tf - 1e-4) return false;
      float dFront = max(0.0, tc1 - tnOrig);
      float dBack = max(0.0, tf - tc1);
      float dLip = (tc0 > tnOrig - rRow * 0.08) ? max(0.0, tnOrig + rRow * 0.06 - tc0) : 1e9;
      tn = tc1;
      kn = CP + m + 1;
      outVoidN = -exitN;
      outVoidBorder = min(dFront, min(dBack, dLip));
      outVoidSeam = exitSeam;
    } else if (tc0 > tStart && tc0 < tf && tc1 >= tf - rRow * 0.06) {
      float dBackCut = max(0.0, tfOrig - tc0);
      float dThru = max(0.0, tc0 - tStart);
      tf = tc0;
      if (tf <= max(tn, 0.0) + 1e-4) return false;
      outVoidBorder = min(outVoidBorder, min(dBackCut, dThru));
      outVoidSeam = min(outVoidSeam, exitSeam);
    }
  }
  return tf > max(tn, 0.0) + 1e-4;
}

float n3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
        mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
        mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}

float n1d(float x, float s) {
  float i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(hash(vec3(i, s, f_seed)), hash(vec3(i + 1.0, s, f_seed)), f);
}

float fbm1(float x, float s) {
  return 0.56 * n1d(x, s)
       + 0.29 * n1d(x * 2.17 + 1.9, s + 5.3)
       + 0.15 * n1d(x * 4.53 + 3.7, s + 11.7);
}

float steppedNoise(float x, float s) {
  float i = floor(x), f = fract(x);
  float h0 = hash(vec3(i, s + 17.0, f_seed));
  float h1 = hash(vec3(i + 1.0, s + 17.0, f_seed));
  float stepF = smoothstep(0.32, 0.68, f);
  float terrace = mix(h0, h1, stepF);
  float shelf = floor(terrace * 4.5 + 0.5) / 4.5;
  return mix(terrace, shelf, 0.65);
}

float cellularChip(float x, float s) {
  float i = floor(x), f = fract(x) - 0.5;
  float h = hash(vec3(i, s + 31.0, f_seed));
  float chipAct = step(0.28, h);
  float halfW = mix(0.18, 0.44, fract(h * 13.7));
  float u = clamp(abs(f) / halfW, 0.0, 1.0);
  return chipAct * (1.0 - u * u) * mix(0.45, 1.0, fract(h * 7.9));
}

float sn1(float x, float s) {
  return clamp((n1d(x, s) - 0.5) * 2.85, -1.0, 1.0);
}

// Two-Stage SDF Evaluation (Optimization 1):
// Stage 1 (crystalSDF_fast): coarse analytic 18-plane polyhedron SDF (zero 3D noise calls when far from surface, abs(d) >= 0.045).
// Stage 2 (crystalSDF_full / crystalSDF): detailed 3D CSG edge-chip, irregularity, and basal-cavity evaluation
// evaluated ONLY when near surface (abs(d) < 0.045) and inside calcNormal().
float crystalSDF_fast(vec3 p, int row, out float secondD, out float thirdD, out float ridgeBevel) {
  float d = -1e9;
  secondD = -1e9;
  thirdD = -1e9;
  for (int k = 0; k < 18; k++) {
    vec4 pl = body(row, PA + k);
    float dp = dot(pl.xyz, p) - pl.w;
    if (dp > d) {
      thirdD = secondD;
      secondD = d;
      d = dp;
    } else if (dp > secondD) {
      thirdD = secondD;
      secondD = dp;
    } else if (dp > thirdD) {
      thirdD = dp;
    }
  }
  ridgeBevel = (d + secondD) * 0.70710678;
  return d;
}

// 3D Architectural Bismuth Hopper & Ziggurat Morphological Carver (post-015895, post-015966):
// Pure branchless register-only O(1) evaluation in crystal-local q in [-1, 1]^3 coordinates.
// Precomputes per-crystal morphP once outside the march loop to model 5 distinct Bismuth morphologies:
//   0: Broad Open-Sided Cantilevered Hopper Wing (BGS specimen: 3-step bottom-left cutout, 6 nested steps)
//   1: 5-Tier Aztec Ziggurat Pyramid & Stepped Pagoda Spire (Encyclopedia & Wikipedia stepped pyramid/tower)
//   2: Open-Sided L-Bracket / U-Channel Greek-Key Tower (RSC Education opposite L-notches + open U-channel)
//   3: Diagonal Stair-Stepped Parallel-Ribbed Terrace ("Rainbow Metal" 8 parallel metallic comb ribs)
//   4: Asymmetric Cantilevered Stepped Fin Bar / Twin Hopper Bracket
vec4 getBismuthMorphParams(int uidInt) {
  int seedShift = int(mod(floor(abs(f_seed)), 5.0));
  int m = (uidInt <= 3) ? (uidInt - 1) : ((uidInt - 1 + seedShift) % 5);
  return (m == 0) ? vec4(0.36, -0.44, -6.0,  1.0)
       : (m == 1) ? vec4(0.46,  0.35,  5.0,  0.6)
       : (m == 2) ? vec4(0.44, -0.34,  4.0,  1.2)
       : (m == 3) ? vec4(0.45, -0.45, -7.0, -1.0)
                  : vec4(0.46,  0.24, -4.0,  0.8);
}

float quantStairFast(float t, float nSteps) {
  float s = clamp(t, 0.0, 0.9999) * nSteps;
  float idx = floor(s);
  return (idx + smoothstep(0.74, 0.95, s - idx)) / nSteps;
}

float evalBismuthFast(vec3 pQ, vec4 morphP, float hopperAmt, out float outStepIdx, out float outEdgeDist) {
  float nSt = abs(morphP.z);
  // 1. Broken-out 3D 90-degree stair-stepped silhouette across ALL 3 AXES (X, Y, Z)
  float yUp = clamp((pQ.y + 0.90) * 0.54, 0.0, 1.0);
  float yDn = clamp((0.90 - pQ.y) * 0.54, 0.0, 1.0);
  float xPos = clamp((pQ.x + 0.85) * 0.56, 0.0, 1.0);
  float zPos = clamp((pQ.z + 0.85) * 0.56, 0.0, 1.0);
  float stairA = quantStairFast(morphP.z > 0.0 ? yUp : yDn, 5.0);
  float stairB = quantStairFast(morphP.y >= 0.0 ? yUp : yDn, 4.0);
  float stairC = quantStairFast(xPos, 4.0);
  float stairZ = quantStairFast(zPos, 4.0);

  float maxPosX =  1.002 - morphP.x * stairA * hopperAmt;
  float minNegX = -1.002 + abs(morphP.y) * stairB * hopperAmt;
  float minNegY = -1.002 + 0.36 * stairC * hopperAmt;
  float maxPosY =  1.002 - 0.32 * stairZ * hopperAmt;
  float maxPosZ =  1.002 - 0.38 * stairA * hopperAmt;
  float minNegZ = -1.002 + 0.34 * stairB * hopperAmt;

  float wX = min(maxPosX - pQ.x, pQ.x - minNegX);
  float wY = min(maxPosY - pQ.y, pQ.y - minNegY);
  float wZ = min(maxPosZ - pQ.z, pQ.z - minNegZ);

  float wFace = min(wX, min(wY, wZ));
  float dContour = max(min(wX, wY), min(max(wX, wY), wZ));
  float polyAllow = 0.65 * max(0.0, 1.0 - hopperAmt / 0.88);

  // Active 2D face coordinates for internal G-spiral / L-bracket / parallel-rib walls
  vec2 uv = (wZ <= wX && wZ <= wY) ? pQ.xy
          : (wX <= wY)             ? vec2(pQ.z, pQ.y)
                                   : vec2(pQ.x, pQ.z);

  // Open-sided L-bracket / G-spiral breach that breaks through both the outer rim and inner terraces (post-016625)
  float openGap = smoothstep(0.05, 0.14, uv.x * sign(morphP.w)) * (1.0 - smoothstep(0.18, 0.30, abs(uv.y - 0.12 * sign(morphP.z))));
  float effRim = 0.115 - 0.088 * openGap;

  float insideWell = smoothstep(effRim, effRim + 0.030, dContour) * (1.0 - smoothstep(0.66, 0.86, abs(pQ.y) * (1.0 - 0.30 * openGap)));
  float wellEase = smoothstep(0.02, 0.38, hopperAmt);
  if (insideWell <= 1e-4) {
    outStepIdx = (0.9 * stairA + 0.6 * stairB + 0.5 * stairZ) * hopperAmt * wellEase;
    outEdgeDist = mix(1.0, abs(dContour - effRim), wellEase);
    return wFace + polyAllow;
  }

  // 2. Contour-following stepped hopper tiers + broken G-spiral / sub-chamber divider bars
  float normContour = clamp((dContour - effRim) / 0.68, 0.0, 1.0);
  float wellStair = quantStairFast(normContour, nSt);
  float sWell = normContour * nSt;
  float fWell = fract(sWell);

  float ridgePhase = abs(fWell - 0.36);
  float concentricRidge = (1.0 - smoothstep(0.13, 0.21, ridgePhase)) * (1.0 - 0.92 * openGap);

  // Parallel comb ribs (active when morphP.w < 0)
  float parCoord = (uv.x - uv.y * 0.16) * 4.0;
  float parWave = abs(fract(parCoord) - 0.5);
  float parallelRibs = 1.0 - smoothstep(0.14, 0.22, parWave);

  // Asymmetric Greek-key inner G-hook bar + sub-chamber partition bar
  float dHook1 = max(abs(uv.x - 0.05 * sign(morphP.w)) - 0.30, abs(uv.y - 0.10) - 0.052);
  float dHook2 = max(abs(uv.x + 0.14 * sign(morphP.w)) - 0.052, abs(uv.y + 0.03) - 0.28);
  float greekHook = 1.0 - smoothstep(-0.012, 0.012, min(dHook1, dHook2));

  float wallMask = (morphP.w < 0.0)
    ? max(parallelRibs * 0.88, max(concentricRidge * 0.72, greekHook * 0.65))
    : max(concentricRidge, greekHook);

  float floorDepth = (0.22 + 0.50 * wellStair) * hopperAmt * wellEase;
  float ridgeTopDepth = (0.05 + 0.13 * wellStair) * hopperAmt * wellEase;
  float targetDepth = mix(floorDepth, ridgeTopDepth, wallMask) * insideWell;

  outStepIdx = ((0.65 + 2.8 * wellStair + 1.2 * stairA - 0.55 * wallMask) * insideWell + abs(morphP.w) * 0.25) * hopperAmt * wellEase;
  float eWell = min(fWell, 1.0 - fWell) / nSt;
  float rawEdgeDist = min(min(eWell, abs(dContour - effRim)), (morphP.w < 0.0) ? abs(parWave - 0.18) * 0.25 : abs(ridgePhase - 0.16) / nSt);
  outEdgeDist = mix(1.0, rawEdgeDist, wellEase);
  return (wFace + polyAllow * (1.0 - insideWell)) - targetDepth;
}

float evalBismuthHopper(vec3 p, int row, float hopperAmt, out float outStepIdx, out float outEdgeDist) {
  outStepIdx = 0.0;
  outEdgeDist = 1.0;
  if (hopperAmt <= 0.01) return -1e9;

  vec3 ax = body(row, 1).xyz;
  float L = max(body(row, 1).w, 0.08);
  vec3 rt = body(row, 3).xyz;
  vec3 cMid = rt + ax * (0.52 * L);
  vec3 p0 = body(row, PA).xyz;
  vec3 sx = p0 - ax * dot(p0, ax);
  float sxLen = length(sx);
  sx = sxLen > 1e-4 ? (sx / sxLen) : normalize(cross(ax, vec3(0.577, 0.577, 0.577)));
  vec3 sz = cross(ax, sx);

  float dxp = body(row, PA + 0).w - dot(body(row, PA + 0).xyz, cMid);
  float dxn = body(row, PA + 4).w - dot(body(row, PA + 4).xyz, cMid);
  float dzp = body(row, PA + 2).w - dot(body(row, PA + 2).xyz, cMid);
  float dzn = (body(row, PA + 6).w - dot(body(row, PA + 6).xyz, cMid)) / max(0.5, abs(dot(body(row, PA + 6).xyz, sz)));
  float dyp = (body(row, PA + 17).w - dot(body(row, PA + 17).xyz, cMid)) / max(0.25, abs(dot(body(row, PA + 17).xyz, ax)));
  float dyn = (body(row, PA + 16).w - dot(body(row, PA + 16).xyz, cMid)) / max(0.25, abs(dot(body(row, PA + 16).xyz, ax)));
  vec3 cBox = cMid + sx * (0.5 * (dxp - dxn)) + ax * (0.5 * (dyp - dyn)) + sz * (0.5 * (dzp - dzn));

  float hx = max(0.035, 0.5 * (dxp + dxn));
  float hy = max(0.035, 0.5 * (dyp + dyn));
  float hz = max(0.035, 0.5 * (dzp + dzn));
  vec3 halfExt = vec3(hx, hy, hz);
  float fScale = min(hx, min(hy, hz));

  vec3 rel = p - cBox;
  vec3 q = vec3(dot(rel, sx), dot(rel, ax), dot(rel, sz)) / halfExt;
  int uidInt = int(max(1.0, body(row, 2).z) + 0.5);
  float sxSign = ((uidInt & 1) == 0) ? 1.0 : -1.0;
  vec4 morphP = getBismuthMorphParams(uidInt);
  float solidMargin = evalBismuthFast(vec3(q.x * sxSign, q.y, q.z), morphP, hopperAmt, outStepIdx, outEdgeDist);
  return -solidMargin * fScale;
}

// Fast branchless 3D raymarcher for varied Bismuth Hopper morphologies (ziggurat spires, cantilevered
// BGS wings, open L-brackets, parallel-ribbed plates & twin hoppers):
// Precomputes local ray coordinates (q0, dq) and morphP ONCE per crystal (zero UBO reads inside loop).
bool intersectBismuthHopper(int row, vec3 ro, vec3 rd, float hopperAmt, float tHitMax, inout float tn, float tf, inout int kn, inout vec3 outHopperN) {
  vec3 origN = (kn < CP) ? body(row, PA + kn).xyz : outHopperN;
  outHopperN = origN;
  float tStart = max(tn, 0.0);
  if (tStart >= tHitMax) return false;

  vec3 ax = body(row, 1).xyz;
  float L = max(body(row, 1).w, 0.08);
  vec3 rt = body(row, 3).xyz;
  vec3 cMid = rt + ax * (0.52 * L);
  vec3 p0 = body(row, PA).xyz;
  vec3 sx = p0 - ax * dot(p0, ax);
  float sxLen = length(sx);
  sx = sxLen > 1e-4 ? (sx / sxLen) : normalize(cross(ax, vec3(0.577, 0.577, 0.577)));
  vec3 sz = cross(ax, sx);

  float dxp = body(row, PA + 0).w - dot(body(row, PA + 0).xyz, cMid);
  float dxn = body(row, PA + 4).w - dot(body(row, PA + 4).xyz, cMid);
  float dzp = body(row, PA + 2).w - dot(body(row, PA + 2).xyz, cMid);
  float dzn = (body(row, PA + 6).w - dot(body(row, PA + 6).xyz, cMid)) / max(0.5, abs(dot(body(row, PA + 6).xyz, sz)));
  float dyp = (body(row, PA + 17).w - dot(body(row, PA + 17).xyz, cMid)) / max(0.25, abs(dot(body(row, PA + 17).xyz, ax)));
  float dyn = (body(row, PA + 16).w - dot(body(row, PA + 16).xyz, cMid)) / max(0.25, abs(dot(body(row, PA + 16).xyz, ax)));
  vec3 cBox = cMid + sx * (0.5 * (dxp - dxn)) + ax * (0.5 * (dyp - dyn)) + sz * (0.5 * (dzp - dzn));

  float hx = max(0.035, 0.5 * (dxp + dxn));
  float hy = max(0.035, 0.5 * (dyp + dyn));
  float hz = max(0.035, 0.5 * (dzp + dzn));
  vec3 invHalf = 1.0 / vec3(hx, hy, hz);

  int uidInt = int(max(1.0, body(row, 2).z) + 0.5);
  float sxSign = ((uidInt & 1) == 0) ? 1.0 : -1.0;
  vec4 morphP = getBismuthMorphParams(uidInt);

  vec3 rel0 = (ro + rd * tStart) - cBox;
  vec3 pQ0 = vec3(dot(rel0, sx) * sxSign, dot(rel0, ax), dot(rel0, sz)) * invHalf;
  vec3 dpQ = vec3(dot(rd, sx) * sxSign, dot(rd, ax), dot(rd, sz)) * invHalf;

  float dtMax = min(tf, tHitMax) - tStart;
  if (dtMax <= 1e-4) {
    float s0, e0;
    return evalBismuthFast(pQ0, morphP, hopperAmt, s0, e0) >= -1e-4;
  }

  // March along local ray pQ(dt) = pQ0 + dpQ * dt in 10 fast branchless register-only steps (7 on mobile)
  int nSteps = (u_isMobile > 0.5) ? 7 : 10;
  float dtStep = dtMax / float(nSteps);
  float dtPrev = 0.0;
  bool foundHit = false;
  float dtHit = 0.0;

  for (int ms = 0; ms <= 10; ms++) {
    if (ms > nSteps) break;
    float dt = float(ms) * dtStep;
    float sIdx, sEdge;
    if (evalBismuthFast(pQ0 + dpQ * dt, morphP, hopperAmt, sIdx, sEdge) >= -1e-4) {
      dtHit = dt;
      foundHit = true;
      break;
    }
    dtPrev = dt;
  }

  if (!foundHit) return false;

  if (dtHit > 1e-4) {
    float tLo = dtPrev, tHi = dtHit;
    for (int bs = 0; bs < 3; bs++) {
      float tMid = 0.5 * (tLo + tHi);
      float sIdx, sEdge;
      if (evalBismuthFast(pQ0 + dpQ * tMid, morphP, hopperAmt, sIdx, sEdge) >= 0.0) tHi = tMid;
      else tLo = tMid;
    }
    dtHit = tHi;
  }

  // Compute 3D normal in world space via forward differences of evalBismuthFast(pQ) (only 4 evals total!)
  vec3 pQH = pQ0 + dpQ * dtHit;
  float s0, e0;
  float m0 = evalBismuthFast(pQH, morphP, hopperAmt, s0, e0);
  if (dtHit > 1e-4 || m0 < 0.016) {
    float eps = 0.005;
    float sD, eD;
    float gx = (evalBismuthFast(pQH + vec3(eps, 0.0, 0.0), morphP, hopperAmt, sD, eD) - m0) * sxSign;
    float gy =  evalBismuthFast(pQH + vec3(0.0, eps, 0.0), morphP, hopperAmt, sD, eD) - m0;
    float gz =  evalBismuthFast(pQH + vec3(0.0, 0.0, eps), morphP, hopperAmt, sD, eD) - m0;
    vec3 gLocal = -vec3(gx, gy, gz) * invHalf;
    float gLen = length(gLocal);
    if (gLen > 1e-5) {
      tn = tStart + dtHit;
      kn = CP + 2000;
      vec3 nCand = normalize(sx * gLocal.x + ax * gLocal.y + sz * gLocal.z);
      float hopNBlend = smoothstep(0.02, 0.45, hopperAmt);
      nCand = normalize(mix(origN, nCand, hopNBlend));
      if (dot(nCand, rd) > -0.06) {
        nCand = normalize(nCand - rd * (dot(nCand, rd) + 0.06));
      }
      outHopperN = nCand;
    }
  }
  return true;
}

// 3D Elestial / Jacare Cathedral Spire Carver (task-0693, post-016223, post-016233):
// Carves a proud Central Nave spire column (u = 0) and Ridge-Anchored Flying-Buttress pinnacles (u = +/- 0.55)
// with flat, glassy vertical prism faces, crisp vertical Gothic flute bevels, and 3 tiered miniature
// pointed rhombohedral roof-peaks (with re-entrant V-notches behind each sub-termination tip) climbing up each spire flank.
void evalCathedralColumn(
  float du, float sideExcess, float inSideWall, float dColBase, float ly, float yBase0, float cathAmt, float baseRamp,
  out float outDCarve, out float outGu, out float outGax, out float outEdgeDist, out float outIsGable
) {
  float au = min(abs(du), 0.36);
  float tierStep = 0.220;
  float chevronSlope = 0.26;
  float netStepDepth = 0.105;
  float roofDrop = 0.165;
  float sideWallSlope = 1.65;
  float fRoofSpan = 0.26;
  float fNotchSpan = 0.08;

  float yGable0 = yBase0 - chevronSlope * au;
  float s = (ly - yGable0) / tierStep;
  float tierDepth = 0.0;
  float dYEdge = abs(ly - yGable0);
  float gableMask = 0.0;
  float notchMask = 0.0;

  if (s > 0.0) {
    float m = clamp(floor(s), 0.0, 2.0);
    float f = clamp(s - m, 0.0, 1.0);
    if (f < fRoofSpan) {
      float roofFrac = f / fRoofSpan;
      tierDepth = m * netStepDepth + roofFrac * roofDrop;
      dYEdge = min(f, fRoofSpan - f) * tierStep;
      gableMask = smoothstep(-0.006, 0.006, f) * (1.0 - smoothstep(fRoofSpan - 0.006, fRoofSpan + 0.006, f));
    } else if (f < fRoofSpan + fNotchSpan) {
      // Re-entrant V-notch right above the miniature rhombohedral roof-peak where the next vertical shaft emerges
      float notchFrac = (f - fRoofSpan) / fNotchSpan;
      tierDepth = mix(m * netStepDepth + roofDrop, (m + 1.0) * netStepDepth, smoothstep(0.0, 1.0, notchFrac));
      dYEdge = (f - fRoofSpan) * tierStep;
      notchMask = 1.0 - smoothstep(0.0, 1.0, notchFrac);
    } else {
      tierDepth = (m + 1.0) * netStepDepth;
      dYEdge = min(f - (fRoofSpan + fNotchSpan), 1.0 - f) * tierStep;
    }
  }

  float dRaw = dColBase + sideWallSlope * sideExcess + tierDepth + gableMask * 0.12 * au;
  outDCarve = dRaw * baseRamp * cathAmt;

  float smoothSignU = clamp(du / 0.010, -1.0, 1.0);
  float guRib = smoothSignU * (0.90 * inSideWall);
  float guGable = smoothSignU * mix(0.40, 0.90, inSideWall);
  outGu = mix(guRib, guGable, gableMask) * baseRamp * cathAmt;
  outGax = (gableMask * (1.0 - 0.35 * inSideWall) * 0.92 - notchMask * 0.42) * baseRamp * cathAmt;
  float dGableRidge = (gableMask > 0.05) ? min(au, dYEdge * 2.8) : (dYEdge * 2.8);
  outEdgeDist = dGableRidge;
  outIsGable = gableMask;
}

float evalCathedralFace(
  float u, float ly, float faceSeed, float cathAmt,
  out float outGu, out float outGax, out float outEdgeDist, out float outIsGable
) {
  float axSign = (ly >= 0.14) ? 1.0 : -1.0;
  float lyBilateral = 0.16 + abs(ly - 0.14) * 1.12;
  float baseRamp = smoothstep(0.05, 0.14, lyBilateral);
  if (baseRamp <= 1e-4 || cathAmt <= 1e-4) {
    outGu = 0.0; outGax = 0.0; outEdgeDist = 1.0; outIsGable = 0.0;
    return 0.0;
  }
  float fShift = (faceSeed - 0.5) * 0.028;
  float absU = abs(u);
  float sideSign = (u >= 0.0) ? 1.0 : -1.0;

  // 1. Broad Central Nave Spire Column (u = 0, wCol = 0.23, taller gable peaks on both upper and short lower halves)
  float sideEx0 = max(0.0, absU - 0.23);
  float inWall0 = smoothstep(0.22, 0.24, absU) * (1.0 - smoothstep(0.27, 0.29, absU));
  float d0, gu0, gax0, edge0, gab0;
  evalCathedralColumn(u, sideEx0, inWall0, 0.0, lyBilateral, 0.250 + fShift, cathAmt, baseRamp, d0, gu0, gax0, edge0, gab0);

  // 2. Flanking Buttress Spire Column (|u| >= 0.25 extending cleanly through the prism corner ridges with zero outer excess!)
  float sideEx1 = max(0.0, 0.25 - absU);
  float inWall1 = 1.0 - smoothstep(0.24, 0.27, absU);
  float d1, gu1, gax1, edge1, gab1;
  evalCathedralColumn(
    u - sideSign * 0.46, sideEx1, inWall1, 0.032, lyBilateral,
    0.195 + fShift * 0.5, cathAmt, baseRamp,
    d1, gu1, gax1, edge1, gab1
  );

  // CSG union of Central Nave and Flanking Buttress columns (= min of inward carve depths)
  float fluteDist = abs(absU - 0.24);
  float seamDist = min(fluteDist, abs(d0 - d1) * 0.55);
  if (d0 <= d1) {
    outGu = gu0; outGax = gax0 * axSign; outEdgeDist = min(edge0, seamDist); outIsGable = gab0;
    return d0;
  } else {
    outGu = gu1; outGax = gax1 * axSign; outEdgeDist = min(edge1, seamDist); outIsGable = gab1;
    return d1;
  }
}

bool intersectCathedralSpire(
  int row, vec3 ro, vec3 rd, float cathAmt, float tHitMax,
  inout float tn, inout float tf, inout int kn, out vec3 outCathN, out float outCathEdge
) {
  vec3 origN = (kn < CP) ? body(row, PA + kn).xyz : vec3(0.0, 1.0, 0.0);
  outCathN = origN;
  outCathEdge = 1e9;
  float tStart = max(tn, 0.0);
  if (tStart >= tHitMax || cathAmt <= 0.005) return tStart < tHitMax;

  vec3 ax = body(row, 1).xyz;
  float L = max(body(row, 1).w, 0.08);
  float r = max(body(row, 2).w, 0.02);
  float invR = 1.0 / r;
  float invL = 1.0 / L;
  vec3 rt = body(row, 3).xyz;
  float uid = max(1.0, body(row, 2).z);

  float tEnd = min(tf, tHitMax);
  float dtSpan = tEnd - tStart;
  if (dtSpan <= 1e-5) return true;

  vec3 pStart = ro + rd * tStart;
  float ly0 = dot(pStart - rt, ax) * invL;
  float dly = dot(rd, ax) * invL;
  float lyEnd = ly0 + dly * dtSpan;
  if (max(ly0, lyEnd) < -0.42) return true;

  // Find the 3 closest active prism planes (kA, kB, kC) at the chord midpoint pMid.
  // Because any chord through a hexagonal prism interacts with the 3 prism faces on its side of the axis,
  // the 3 closest prism planes at pMid are always the exact entry, grazed-silhouette, and exit prism flanks!
  vec3 pMid = ro + rd * (tStart + 0.5 * dtSpan);
  int kA = -1, kB = -1, kC = -1;
  float bestDA = 1e9, bestDB = 1e9, bestDC = 1e9;
  float maxValidOff = L * 1.35 + r * 2.2;
  bool isHexPrism = dot(body(row, PA + 1).xyz, body(row, PA + 3).xyz) > 0.25;
  for (int k = 0; k < 7; k++) {
    if (k == 2 && isHexPrism) continue;
    vec4 pl = body(row, PA + k);
    if (pl.w - dot(pl.xyz, rt) > maxValidOff) continue;
    float d = (pl.w - dot(pl.xyz, pMid)) * invR;
    if (d < bestDA) {
      bestDC = bestDB; kC = kB;
      bestDB = bestDA; kB = kA;
      bestDA = d;      kA = k;
    } else if (d < bestDB) {
      bestDC = bestDB; kC = kB;
      bestDB = d;      kB = k;
    } else if (d < bestDC) {
      bestDC = d;      kC = k;
    }
  }
  if (kA < 0) return true;
  if (kB < 0) kB = kA;
  if (kC < 0) kC = kB;

  vec4 plA = body(row, PA + kA);
  vec3 nA = plA.xyz;
  vec3 tUA = cross(ax, nA);
  float tUALen = length(tUA);
  tUA = tUALen > 1e-4 ? (tUA / tUALen) : normalize(cross(nA, vec3(0.577, 0.577, 0.577)));
  float dA0 = (plA.w - dot(nA, pStart)) * invR;
  float ddA = -dot(nA, rd) * invR;
  float uA0 = dot(pStart - rt, tUA) * invR;
  float duA = dot(rd, tUA) * invR;
  float seedA = fract(uid * 0.3183 + float(kA) * 0.173 + f_seed * 0.071);

  vec4 plB = body(row, PA + kB);
  vec3 nB = plB.xyz;
  vec3 tUB = cross(ax, nB);
  float tUBLen = length(tUB);
  tUB = tUBLen > 1e-4 ? (tUB / tUBLen) : tUA;
  float dB0 = (plB.w - dot(nB, pStart)) * invR;
  float ddB = -dot(nB, rd) * invR;
  float uB0 = dot(pStart - rt, tUB) * invR;
  float duB = dot(rd, tUB) * invR;
  float seedB = fract(uid * 0.3183 + float(kB) * 0.173 + f_seed * 0.071);

  vec4 plC = body(row, PA + kC);
  vec3 nC = plC.xyz;
  vec3 tUC = cross(ax, nC);
  float tUCLen = length(tUC);
  tUC = tUCLen > 1e-4 ? (tUC / tUCLen) : tUA;
  float dC0 = (plC.w - dot(nC, pStart)) * invR;
  float ddC = -dot(nC, rd) * invR;
  float uC0 = dot(pStart - rt, tUC) * invR;
  float duC = dot(rd, tUC) * invR;
  float seedC = fract(uid * 0.3183 + float(kC) * 0.173 + f_seed * 0.071);

  float maxCarve = 0.46 * cathAmt;
  // Fast bypass if ray enters top termination pyramid already deeper inside prism collar than maxCarve
  if (kn >= 7 && dA0 > maxCarve && dB0 > maxCarve && ddA >= -0.05 && ddB >= -0.05) {
    return true;
  }

  int nSteps = (u_isMobile > 0.5) ? 6 : 8;
  float dtStep = dtSpan / float(nSteps);
  float dtPrev = 0.0;
  float dtHit = 0.0;
  bool foundHit = false;

  for (int ms = 0; ms <= 8; ms++) {
    if (ms > nSteps) break;
    float dt = float(ms) * dtStep;
    float ly = ly0 + dly * dt;
    float guA, gaxA, edA, gabA;
    float cA = evalCathedralFace(uA0 + duA * dt, ly, seedA, cathAmt, guA, gaxA, edA, gabA);
    float mMin = (dA0 + ddA * dt) - cA;
    if (kB != kA) {
      float guB, gaxB, edB, gabB;
      float cB = evalCathedralFace(uB0 + duB * dt, ly, seedB, cathAmt, guB, gaxB, edB, gabB);
      mMin = min(mMin, (dB0 + ddB * dt) - cB);
    }
    if (kC != kB && kC != kA) {
      float guC, gaxC, edC, gabC;
      float cC = evalCathedralFace(uC0 + duC * dt, ly, seedC, cathAmt, guC, gaxC, edC, gabC);
      mMin = min(mMin, (dC0 + ddC * dt) - cC);
    }
    if (mMin >= -1e-4) {
      dtHit = dt;
      foundHit = true;
      break;
    }
    dtPrev = dt;
  }

  // If the ray never entered the carved cathedral spire along the entire chord [tStart, tEnd],
  // it passed completely through a 3D stepped sub-termination notch!
  if (!foundHit) return false;

  if (dtHit > 1e-5) {
    float tLo = dtPrev, tHi = dtHit;
    for (int bs = 0; bs < 3; bs++) {
      float tMid = 0.5 * (tLo + tHi);
      float ly = ly0 + dly * tMid;
      float guA, gaxA, edA, gabA;
      float cA = evalCathedralFace(uA0 + duA * tMid, ly, seedA, cathAmt, guA, gaxA, edA, gabA);
      float mMin = (dA0 + ddA * tMid) - cA;
      if (kB != kA) {
        float guB, gaxB, edB, gabB;
        float cB = evalCathedralFace(uB0 + duB * tMid, ly, seedB, cathAmt, guB, gaxB, edB, gabB);
        mMin = min(mMin, (dB0 + ddB * tMid) - cB);
      }
      if (kC != kB && kC != kA) {
        float guC, gaxC, edC, gabC;
        float cC = evalCathedralFace(uC0 + duC * tMid, ly, seedC, cathAmt, guC, gaxC, edC, gabC);
        mMin = min(mMin, (dC0 + ddC * tMid) - cC);
      }
      if (mMin >= 0.0) tHi = tMid;
      else tLo = tMid;
    }
    dtHit = tHi;
  }

  // If the ray entered a main termination face (kn >= 7) without being clipped by the cathedral flank (dtHit == 0):
  // Above ly = 0.85 (or on bottom termination faces), keep the termination plane untouched;
  // on the lower/mid shoulder of top rhombohedral faces (kn in 8..13, ly0 < 0.85), carry the Elestial cathedral
  // gable terraces seamlessly across the rhombohedral shoulder!
  if (kn >= 7 && dtHit <= 1e-5) {
    if (kn >= 8 && kn <= 13 && ly0 > 0.16 && ly0 < 0.85) {
      float termFade = (1.0 - smoothstep(0.64, 0.85, ly0)) * smoothstep(0.16, 0.28, ly0);
      float guT, gaxT, edT, gabT;
      evalCathedralFace(uA0, ly0, seedA, cathAmt, guT, gaxT, edT, gabT);
      vec3 nTermCath = normalize(origN + (guT * 0.48 * tUA - gaxT * 0.42 * ax) * termFade);
      if (dot(nTermCath, rd) > -0.06) {
        nTermCath = normalize(nTermCath - rd * (dot(nTermCath, rd) + 0.06));
      }
      kn = CP + 4000 + kn;
      outCathN = nTermCath;
      outCathEdge = edT * r;
    }
    return true;
  }

  float lyH = ly0 + dly * dtHit;
  float guA, gaxA, edA, gabA;
  float cA = evalCathedralFace(uA0 + duA * dtHit, lyH, seedA, cathAmt, guA, gaxA, edA, gabA);
  float mA = (dA0 + ddA * dtHit) - cA;

  int kWin = kA;
  vec3 nWin = normalize(nA + guA * tUA + gaxA * ax);
  float edgeWin = edA;
  float mWin = mA;

  if (kB != kA) {
    float guB, gaxB, edB, gabB;
    float cB = evalCathedralFace(uB0 + duB * dtHit, lyH, seedB, cathAmt, guB, gaxB, edB, gabB);
    float mB = (dB0 + ddB * dtHit) - cB;
    vec3 nCandB = normalize(nB + guB * tUB + gaxB * ax);
    float ridgeDiff = abs(mWin - mB);
    edgeWin = min(min(edgeWin, edB), ridgeDiff * 0.65);
    float wB = 1.0 - smoothstep(-0.012, 0.012, mB - mWin);
    nWin = normalize(mix(nWin, nCandB, wB));
    if (mB < mWin) { kWin = kB; mWin = mB; }
  }
  if (kC != kB && kC != kA) {
    float guC, gaxC, edC, gabC;
    float cC = evalCathedralFace(uC0 + duC * dtHit, lyH, seedC, cathAmt, guC, gaxC, edC, gabC);
    float mC = (dC0 + ddC * dtHit) - cC;
    vec3 nCandC = normalize(nC + guC * tUC + gaxC * ax);
    float ridgeDiff = abs(mWin - mC);
    edgeWin = min(min(edgeWin, edC), ridgeDiff * 0.65);
    float wC = 1.0 - smoothstep(-0.012, 0.012, mC - mWin);
    nWin = normalize(mix(nWin, nCandC, wC));
    if (mC < mWin) { kWin = kC; mWin = mC; }
  }

  if (dot(nWin, rd) > -0.06) {
    nWin = normalize(nWin - rd * (dot(nWin, rd) + 0.06));
  }
  tn = tStart + dtHit;
  kn = CP + 4000 + kWin;
  outCathN = nWin;
  outCathEdge = edgeWin * r;
  return tf > tn + 1e-4;
}

float crystalSDF_full(vec3 p, int row) {
  float secondD, thirdD, ridgeBevel;
  float d = crystalSDF_fast(p, row, secondD, thirdD, ridgeBevel);
  if (u_hopper > 0.01) {
    if (d >= 0.045) return d - 0.008;
    float hStep, hEdge;
    float dHopperVoid = evalBismuthHopper(p, row, u_hopper, hStep, hEdge);
    d = max(d, dHopperVoid);
    return d;
  }
  // Skip all 3D CSG edge & basal-cavity noise when far from surface (abs(d) >= 0.045)
  if (abs(d) >= 0.045) {
    return (d > 0.045) ? (d - 0.008) : d;
  }
  float r = max(body(row, 2).w, 0.02);
  float rawSurfScale = mix(0.15, 1.0, clamp(u_rawAmount / 0.4, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, (u_rawAmount - 0.4) / 1.6));
  float effEdgeMacro = max(0.0, u_patinaMacro * u_edgeMacro * rawSurfScale);
  float effChips = clamp(u_edgeChips * effEdgeMacro, 0.0, 2.0);
  float effIrreg = clamp((u_edgeIrregularity + 0.25 * u_edgeWear) * effEdgeMacro, 0.0, 2.0);
  if ((effChips > 0.005 || effIrreg > 0.005) && abs(d - secondD) < 0.045) {
    float uid = max(1.0, body(row, 2).z);
    float chipN = n3(p * (9.5 * clamp(u_edgeGranularity, 0.2, 3.5)) + uid * 7.3);
    float edgeCut = r * (0.004 * effIrreg + 0.008 * effChips * chipN);
    d = max(d, ridgeBevel - edgeCut);
  }
  return d;
}

float crystalSDF(vec3 p, int row) {
  return crystalSDF_full(p, row);
}

vec3 calcNormal(vec3 p, int row) {
  float eps = (u_hopper > 0.01) ? 0.00055 : 0.0015;
  vec2 e = vec2(eps, -eps);
  return normalize(
    e.xyy * crystalSDF_full(p + e.xyy, row) +
    e.yyx * crystalSDF_full(p + e.yyx, row) +
    e.yxy * crystalSDF_full(p + e.yxy, row) +
    e.xxx * crystalSDF_full(p + e.xxx, row)
  );
}

float evalEdgeContourFast(float alongEdge, float edgeSeed, float gran) {
  float g = clamp(gran, 0.2, 3.5);
  float x = alongEdge * (0.80 + 0.45 * g);
  float terrace = steppedNoise(x * 2.2, edgeSeed);
  float chipCell = cellularChip(x * 3.4 + 1.3, edgeSeed + 19.0);
  float macroWave = sn1(x * 1.8, edgeSeed + 7.0);
  return clamp((terrace - 0.36) * 0.85 + chipCell * 0.72 + macroWave * 0.22, 0.0, 1.35);
}

void evalEdgeContourSlopeAsym(float alongEdge, float edgeSeed, float gran, float fwAlong, float c0, out float outSlope, out float outAsym) {
  float g = clamp(gran, 0.2, 3.5);
  float x = alongEdge * (0.80 + 0.45 * g);
  float hiDamp = 1.0 / (1.0 + 12.0 * fwAlong);
  float dx = 0.03;
  float terraceP = steppedNoise((x + dx) * 2.2, edgeSeed);
  float chipCellP = cellularChip((x + dx) * 3.4 + 1.3, edgeSeed + 19.0);
  float macroWaveP = sn1((x + dx) * 1.8, edgeSeed + 7.0);
  float cPlus = clamp((terraceP - 0.36) * 0.85 + chipCellP * 0.72 + macroWaveP * 0.22, 0.0, 1.35);
  outSlope = clamp((cPlus - c0) / dx, -0.85, 0.85) * hiDamp;
  float stepAsym = (steppedNoise(x * 2.9 + 4.3, edgeSeed + 29.0) - 0.5) * 2.0;
  outAsym = clamp(stepAsym, -0.9, 0.9);
}

void evalSingleRidgeBevel(
  int ka, int kb, vec4 plA, vec4 plB,
  vec3 ro, vec3 rd, float tStart, inout float tf,
  vec3 ax, vec3 rt, float L, float r, float uid,
  float effChips, float effIrreg, float chipSz, float nodePresence, float gran,
  float cornerMask, float dCorner3, float fwBase,
  inout float bestTChip, inout int bestEnc, inout vec3 bestChipN
) {
  if (ka < 0 || kb < 0 || ka >= 16 || kb >= 16 || ka == kb) return;
  int kLo = min(ka, kb), kHi = max(ka, kb);
  vec4 plLo = (ka < kb) ? plA : plB;
  vec4 plHi = (ka < kb) ? plB : plA;
  vec3 nLo = plLo.xyz;
  vec3 nHi = plHi.xyz;
  vec3 sumN = nLo + nHi;
  float lenN = length(sumN);
  float dihedralW = smoothstep(0.45, 0.62, lenN) * (1.0 - smoothstep(1.90, 1.98, lenN));
  if (dihedralW <= 1e-4) return;

  vec3 nAB = sumN / lenN;
  float wAB = (plLo.w + plHi.w) / lenN;
  vec3 pStart = ro + rd * tStart;
  float dRidgeStart = wAB - dot(nAB, pStart);
  float maxPossibleCut = r * (0.016 + 0.026 * max(effChips, effIrreg)) * max(0.65, chipSz) * nodePresence;
  if (dRidgeStart >= min(0.045, maxPossibleCut)) return;

  vec3 edgeDir = cross(nLo, nHi);
  float edLen = length(edgeDir);
  if (edLen <= 1e-4) return;
  edgeDir /= edLen;

  float dnAB = dot(nAB, rd);
  float nomBevel = r * (0.0032 * effIrreg + 0.0028 * effChips) * (0.75 + 0.25 * chipSz) * nodePresence;
  float tNom = (abs(dnAB) > 1e-3) ? clamp((wAB - nomBevel - dot(nAB, ro)) / dnAB, tStart, min(tf, tStart + r * 0.08)) : tStart;
  vec3 pRidge = ro + rd * tNom;
  vec3 cMid = rt + ax * (0.52 * L);
  vec3 relRidge = pRidge - cMid;
  float hNorm = dot(relRidge, ax) / max(L * 0.45, 0.06);
  vec3 radEdgeDir = edgeDir - ax * dot(edgeDir, ax);
  float rNorm = dot(relRidge, radEdgeDir) / max(r * 1.15, 0.03);
  float alongEdge = 0.8 + hNorm * abs(dot(edgeDir, ax)) + rNorm;
  if (alongEdge < -1.1 || alongEdge > 2.6) return;

  float edgeSeed = uid * 13.7 + float(kLo) * 19.3 + float(kHi) * 7.1;
  float cutBest = 0.0;
  float rawContour = 0.0;

  if (effIrreg > 0.005 || effChips > 0.005) {
    rawContour = evalEdgeContourFast(alongEdge, edgeSeed, gran);
    float isTermEdge = (kHi >= 8) ? 0.05 : 0.0;
    float activeGate = smoothstep(0.04, 0.18, rawContour + isTermEdge + 0.12 * cornerMask);
    if (activeGate > 0.005) {
      float edgeContour = clamp(rawContour + isTermEdge + 0.14 * cornerMask, 0.0, 1.35);
      float cutDepth = r * (0.0014 + 0.022 * pow(edgeContour, 1.55)) * activeGate * max(effIrreg, effChips * 0.85) * nodePresence * (0.70 + 0.30 * chipSz);
      cutBest = max(cutBest, cutDepth);
    }
  }

  // Corner-anchored 3D shoulder fracture facet
  if (effChips > 0.005 && cornerMask > 0.08) {
    float hC = hash(vec3(uid * 5.0 + float(kLo) * 19.0 + float(kHi), float(kHi) * 13.0, 11.0));
    float threshC = 0.32;
    if (hC > threshC) {
      float spanR = r * mix(0.034, 0.082, fract(hC * 7.3)) * max(0.5, chipSz);
      float uCorner = clamp(dCorner3 / max(spanR, 1e-4), 0.0, 1.0);
      float scallopEnv = (1.0 - uCorner * uCorner) * (1.0 - uCorner);
      float sizeMult = mix(0.55, 1.35, pow(fract(hC * 19.3), 1.35));
      float baseCut = r * (0.0026 + 0.0068 * effChips) * (0.68 + 0.32 * chipSz) * nodePresence;
      float cutShoulder = baseCut * sizeMult * scallopEnv * smoothstep(threshC, threshC + 0.12, hC);
      cutBest = max(cutBest, cutShoulder);
    }
  }

  cutBest *= dihedralW;
  if (cutBest <= 1e-5) return;

  float startExcess = cutBest - dRidgeStart;
  if (startExcess <= 0.0) return;

  float tCand = tStart + startExcess / max(-dnAB, 0.28);
  if (tCand > bestTChip) {
    float cSlope = 0.0, cAsym = 0.0;
    if (u_isMobile <= 0.5 && (effIrreg > 0.005 || effChips > 0.005)) {
      evalEdgeContourSlopeAsym(alongEdge, edgeSeed, gran, fwBase, rawContour, cSlope, cAsym);
    }
    vec3 bevelTan = normalize(nLo - nHi + vec3(1e-5));
    vec3 nChip = normalize(nAB + bevelTan * (0.22 * cAsym) - edgeDir * (0.08 * cSlope));
    bestTChip = tCand;
    bestEnc = CP + 5000 + (kLo * 18 + kHi) * 2;
    bestChipN = nChip;
  }
}

// 2. Crystal-Locked 3D CSG Corner & Ridge Facets (u_edgeIrregularity & u_edgeChips):
//    Evaluates flat/stepped 3D CSG fracture planes at corners and ridges that clip both tn and tf
//    (returning false when tn >= tf for a true 3D silhouette bite) and outputs the true 3D facet normal outChipN.
bool applyTinyEdgeChip(int row, int k0, vec3 ro, vec3 rd, inout float tn, inout float tf, inout int kn, out float edgeSilAlpha, out vec3 outChipN) {
  edgeSilAlpha = 1.0;
  float rawSurfScale = mix(0.15, 1.0, clamp(u_rawAmount / 0.4, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, (u_rawAmount - 0.4) / 1.6));
  float effEdgeMacro = max(0.0, u_patinaMacro * u_edgeMacro * rawSurfScale) * (1.0 - smoothstep(0.002, 0.03, clamp(u_melt / 0.72, 0.0, 1.0)));
  float effChips = clamp(u_edgeChips * effEdgeMacro, 0.0, 2.0);
  float effIrreg = clamp((u_edgeIrregularity + 0.25 * u_edgeWear) * effEdgeMacro, 0.0, 2.0);
  vec4 pl0 = body(row, PA + k0);
  vec3 n0 = pl0.xyz;
  outChipN = n0;
  if ((effChips <= 0.005 && effIrreg <= 0.005) || k0 >= CP) return true;
  float r = max(body(row, 2).w, 0.02);
  float chipSz = clamp(u_chipSize, 0.1, 3.0);
  float tStart = max(tn, 0.0);
  vec3 pStart = ro + rd * tStart;

  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float L = max(body(row, 1).w, 0.08);
  float uid = max(1.0, body(row, 2).z);
  float nodePresence = clamp(body(row, 2).x, 0.0, 1.0);
  float gran = clamp(u_edgeGranularity, 0.2, 3.5);

  float d1 = 1e9, d2 = 1e9, d3 = 1e9;
  int k1 = -1, k2 = -1, k3 = -1;
  vec4 pl1 = vec4(0.0), pl2 = vec4(0.0);
  float maxValidOff = L * 1.35 + r * 2.2;

  for (int k = 0; k < 18; k++) {
    if (k == k0) continue;
    vec4 pl = body(row, PA + k);
    if (pl.w - dot(pl.xyz, rt) > maxValidOff) continue;
    float cAng = dot(pl.xyz, n0);
    if (cAng > 0.996 || cAng < -0.75) continue;
    float parallelPen = max(0.0, (cAng - 0.94) / 0.056) * r * 1.5;
    float d = max(0.0, pl.w - dot(pl.xyz, pStart)) + parallelPen;
    if (d < d1) {
      d3 = d2; k3 = k2;
      d2 = d1; k2 = k1; pl2 = pl1;
      d1 = d;  k1 = k;  pl1 = pl;
    } else if (d < d2) {
      d3 = d2; k3 = k2;
      d2 = d;  k2 = k;  pl2 = pl;
    } else if (d < d3) {
      d3 = d;  k3 = k;
    }
  }
  if (k1 < 0 || d1 >= r * 0.46 * max(0.45, max(chipSz, effIrreg))) return true;

  int v0 = -1, v1 = -1, v2 = -1;
  float whaleRoll = 0.0;
  float whaleMult = 1.0;
  if (k2 >= 0) {
    v0 = min(k0, min(k1, k2));
    v2 = max(k0, max(k1, k2));
    v1 = (k0 + k1 + k2) - v0 - v2;
    whaleRoll = hash(vec3(uid * 17.0 + float(v0) * 31.0, float(v1) * 23.0 + float(v2) * 13.0, 7.0));
    whaleMult = (whaleRoll >= 0.99) ? 20.0 : ((whaleRoll >= 0.93) ? 10.0 : 1.0);
  }
  if (whaleMult <= 1.0 && d1 >= min(0.045, r * 0.095 * max(0.45, max(chipSz, effIrreg)))) return true;

  float fwBase = max(tStart, 0.12) * f_pixAngle / max(L * 0.35, 0.05);
  float c3 = exp(-max(0.0, d2) / (r * 0.075 * max(0.45, chipSz)));
  float c4 = exp(-max(0.0, d3) / (r * 0.115 * max(0.45, chipSz)));
  float cornerMask = clamp(max(c3, c4 * 1.2), 0.0, 1.0);

  float bestTChip = tStart;
  int bestEnc = kn;
  vec3 bestChipN = n0;
  bool isSilhouetteChip = false;

  // Evaluate 3D CSG ridge bevel half-spaces on all 3 pairs of (k0, k1, k2)
  evalSingleRidgeBevel(
    k0, k1, pl0, pl1, ro, rd, tStart, tf, ax, rt, L, r, uid,
    effChips, effIrreg, chipSz, nodePresence, gran, cornerMask, d2, fwBase,
    bestTChip, bestEnc, bestChipN
  );
  if (bestEnc != kn) {
    isSilhouetteChip = (dot(pl0.xyz, rd) * dot(pl1.xyz, rd) < 0.0);
  }
  if (k2 >= 0 && d2 < r * 0.085 * max(0.45, max(chipSz, effIrreg))) {
    int prevEnc = bestEnc;
    evalSingleRidgeBevel(
      k0, k2, pl0, pl2, ro, rd, tStart, tf, ax, rt, L, r, uid,
      effChips, effIrreg, chipSz, nodePresence, gran, cornerMask, d1, fwBase,
      bestTChip, bestEnc, bestChipN
    );
    if (bestEnc != prevEnc) {
      isSilhouetteChip = (dot(pl0.xyz, rd) * dot(pl2.xyz, rd) < 0.0);
    }
    prevEnc = bestEnc;
    evalSingleRidgeBevel(
      k1, k2, pl1, pl2, ro, rd, tStart, tf, ax, rt, L, r, uid,
      effChips, effIrreg, chipSz, nodePresence, gran, cornerMask, 0.0, fwBase,
      bestTChip, bestEnc, bestChipN
    );
    if (bestEnc != prevEnc) {
      isSilhouetteChip = (dot(pl1.xyz, rd) * dot(pl2.xyz, rd) < 0.0);
    }
  }

  // Exact 3-face corner vertex 3D CSG fracture facet (including 6% 10x Whale / 1% 20x Mega-Whale 3D corner chips)
  bool isWhale = (whaleMult > 1.5);
  float maxCornerD2 = isWhale ? (r * 0.46 * max(0.5, chipSz)) : (r * 0.055 * max(0.5, chipSz));
  if (effChips > 0.005 && k2 >= 0 && d2 < maxCornerD2) {
    float hV = hash(vec3(uid * 11.0 + float(v0) * 29.0, float(v1) * 17.0 + float(v2), 43.0));
    if (isWhale || hV > 0.25) {
      vec4 pv0 = body(row, PA + v0);
      vec4 pv1 = body(row, PA + v1);
      vec4 pv2 = body(row, PA + v2);
      vec3 sumCornerN = pv0.xyz + pv1.xyz + pv2.xyz;
      float lenTri = max(length(sumCornerN), 1e-4);
      vec3 nCornerBase = sumCornerN / lenTri;
      vec3 tC1 = normalize(pv1.xyz - pv0.xyz + vec3(1e-5));
      vec3 tC2 = normalize(cross(nCornerBase, tC1));
      float tilt1 = (fract(hV * 13.7) - 0.5) * 0.26;
      float tilt2 = (fract(hV * 29.3) - 0.5) * 0.26;
      vec3 nTri = normalize(nCornerBase + tC1 * tilt1 + tC2 * tilt2);

      // Exact 3-plane Cramer intersection vertex pVert (fallback to weighted plane offset if nearly coplanar)
      vec3 c12 = cross(pv1.xyz, pv2.xyz);
      float det = dot(pv0.xyz, c12);
      float wTriVertex = (abs(det) > 1e-2)
        ? dot(nTri, (pv0.w * c12 + pv1.w * cross(pv2.xyz, pv0.xyz) + pv2.w * cross(pv0.xyz, pv1.xyz)) / det)
        : ((pv0.w + pv1.w + pv2.w) / lenTri);

      float nomCutV = isWhale
        ? min(r * 0.36, r * (0.0030 + 0.0072 * effChips) * (0.65 + 0.35 * chipSz) * nodePresence * whaleMult * mix(0.88, 1.12, fract(hV * 13.7)))
        : (r * (0.0032 + 0.0076 * effChips) * (0.65 + 0.35 * chipSz) * nodePresence
           * mix(0.70, 1.55, pow(fract(hV * 19.3), 1.35))
           * smoothstep(0.25, 0.40, hV));
      if (nomCutV > 1e-5) {
        float dTriStart = max(0.0, wTriVertex - dot(nTri, pStart));
        float startExcessTri = nomCutV - dTriStart;
        if (startExcessTri > 0.0) {
          float dnTri = dot(nTri, rd);
          float tCorner = tStart + startExcessTri / max(-dnTri, 0.28);
          if (tCorner > bestTChip) {
            bestTChip = tCorner;
            bestEnc = CP + 10000 + ((v0 * 18 + v1) * 18 + v2) * 4;
            bestChipN = nTri;
            float s0 = dot(pv0.xyz, rd), s1 = dot(pv1.xyz, rd), s2 = dot(pv2.xyz, rd);
            isSilhouetteChip = (max(s0, max(s1, s2)) > 0.0);
          }
        }
      }
    }
  }

  if (bestTChip <= tStart + 1e-6) return true;

  if (!isSilhouetteChip) {
    // Interior ridge/corner between front-facing facets: recess into a 3D facet without punching a hole through the crystal
    bestTChip = min(bestTChip, mix(tStart, tf, 0.45));
  } else if (tf <= bestTChip + 1e-5) {
    // True outer silhouette ridge/corner: ray passes completely through the 3D chipped notch!
    return false;
  }

  float featherW = max(max(tStart, 0.12) * f_pixAngle * 1.15, 1e-4);
  if (isSilhouetteChip && tf - bestTChip < featherW) {
    edgeSilAlpha = clamp((tf - bestTChip) / featherW, 0.0, 1.0);
    if (edgeSilAlpha <= 1e-3) return false;
  }
  if (dot(bestChipN, rd) > -0.18) {
    bestChipN = normalize(bestChipN - rd * (dot(bestChipN, rd) + 0.18));
  }
  tn = bestTChip;
  kn = bestEnc;
  outChipN = bestChipN;
  return tf > max(tn, 0.0);
}

bool convexHit(int row, vec3 ro, vec3 rd, out float tn, out float tf, out int kn, out int kf) {
  tn = -1e9; tf = 1e9; kn = 0; kf = 0;
  for (int k = 0; k < CP; k++) {
    vec4 pl = body(row, PA + k);
    float dn = dot(pl.xyz, rd);
    float dist = pl.w - dot(pl.xyz, ro);
    if (abs(dn) < 1e-8) { if (dist < 0.0) return false; continue; }
    float t = dist / dn;
    if (dn < 0.0) {
      if (t > tn) { tn = t; kn = k; }
    } else if (t < tf) {
      tf = t; kf = k;
    }
    if (tn > tf) return false;
  }
  return tf > max(tn, 0.0);
}

float exitCrystal(int row, vec3 ro, vec3 rd, out int kf, out vec3 outExitN) {
  float tf = 1e9; kf = 0;
  outExitN = body(row, PA).xyz;
  for (int k = 0; k < CP; k++) {
    vec4 pl = body(row, PA + k);
    float dn = dot(pl.xyz, rd);
    if (dn > 1e-8) {
      float t = (pl.w - dot(pl.xyz, ro)) / dn;
      if (t < tf) { tf = t; kf = k; outExitN = pl.xyz; }
    }
  }
  if (u_occlusion > 0.002) {
    for (int m = 0; m < 2; m++) {
      float tc0, tc1, exitSeam;
      vec3 enterN, exitN;
      if (!evalSingleVoidInterval(row, m, ro, rd, false, tc0, tc1, enterN, exitN, exitSeam)) continue;
      if (tc0 > 1e-4 && tc0 < tf) {
        tf = tc0;
        kf = CP + m + 1;
        outExitN = -enterN;
      }
    }
  }
  if (u_cathedral > 0.005 && u_hopper <= 0.01 && kf < 7) {
    vec3 ax = body(row, 1).xyz;
    float L = max(body(row, 1).w, 0.08);
    float r = max(body(row, 2).w, 0.02);
    vec3 rt = body(row, 3).xyz;
    float uid = max(1.0, body(row, 2).z);
    vec3 pEx = ro + rd * tf;
    float ly = dot(pEx - rt, ax) / L;
    vec3 nK = body(row, PA + kf).xyz;
    vec3 tUK = cross(ax, nK);
    float tUKLen = length(tUK);
    tUK = tUKLen > 1e-4 ? (tUK / tUKLen) : normalize(cross(nK, vec3(0.577, 0.577, 0.577)));
    float uEx = dot(pEx - rt, tUK) / r;
    float seedK = fract(uid * 0.3183 + float(kf) * 0.173 + f_seed * 0.071);
    float gu, gax, ed, gab;
    float cBack = evalCathedralFace(uEx, ly, seedK, u_cathedral, gu, gax, ed, gab);
    float dnK = max(dot(nK, rd), 0.25);
    tf = max(0.0, tf - (cBack * r) / dnK);
    outExitN = normalize(nK + 0.16 * gu * tUK + 0.22 * gax * ax);
  }
  return max(tf, 0.0);
}

// ---------------------------------------------------------------------------
// Solid-to-Liquid Gallium Melting SDF & Meniscus Collar (u_melt, post-016133):
// Combines upper orthorhombic/polyhedral crystal blocks with a lower multi-lobed
// high-surface-tension molten metal puddle via a polynomial smooth-minimum (smin)
// capillary meniscus collar.
// ---------------------------------------------------------------------------
vec3 toGalliumFrame(vec3 p) {
  return vec3(
    0.186646 * p.x - 0.982427 * p.z,
    p.y,
    0.982427 * p.x + 0.186646 * p.z
  );
}

float sminPoly(float a, float b, float k, out float outH) {
  float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-5), 0.0, 1.0);
  outH = 1.0 - h;
  return mix(b, a, h) - k * h * (1.0 - h);
}

float sminSimple(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-5), 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float smaxSimple(float a, float b, float k) {
  float h = clamp(0.5 - 0.5 * (b - a) / max(k, 1e-5), 0.0, 1.0);
  return mix(b, a, h) + k * h * (1.0 - h);
}

float evalGalliumPuddleSDF(vec3 qGa, float rScale, float tMorph, vec3 coreAnchor, out float outFloorClip, out vec2 outPuddleUV) {
  if (rScale <= 1e-4) {
    outFloorClip = -1e9;
    outPuddleUV = vec2(0.0);
    return 1e4;
  }
  // At resting Liquid Gallium (tMorph == 1.0, rScale == 1.0, meltAmt == 0.72), sRest = 1.02 and pCen = (-0.012, -0.014, 0.042).
  // As rScale contracts 1.0 -> 0.0 (leaving Gallium) or expands 0.0 -> 1.0 (entering Gallium), all 3 pool lobes scale
  // radially in exact world-unit coordinates (sRest = 1.02 constant so |grad| == 1 down to rScale = 0) while pCen
  // draws smoothly inward and upward into coreAnchor (the interior centroid of Hero Crystal #0).
  const float sRest = 1.02;
  vec3 pCen = mix(coreAnchor, vec3(-0.012, -0.014, 0.042), smoothstep(0.0, 0.94, rScale));

  // Tilted puddle plane (~34 deg forward slope toward viewer, matching reference photo 372c3f66)
  vec3 nP = vec3(0.055, 0.825, 0.562);
  vec3 uP = vec3(0.997, -0.020, -0.075);
  vec3 vP = vec3(-0.0112, -0.5618, 0.8272);

  vec3 rel = (qGa - pCen) / sRest;
  vec2 uv = vec2(dot(rel, uP), dot(rel, vP));
  float h = dot(rel, nP);
  vec2 uvNorm = uv / max(rScale, 0.04);
  outPuddleUV = uvNorm;

  // Broad, natural, flat-lying liquid-mercury/gallium pool (3 wide overlapping horizontal pancake lobes scaling radially from 0 <-> 1)
  vec2 pMain = uv - vec2(-0.018, 0.010) * rScale;
  float dPoolMain = (length(pMain * vec2(0.74, 1.24)) - 0.198 * rScale) * 0.86;

  vec2 pLeft = uv - vec2(-0.078, 0.024) * rScale;
  float dPoolLeft = (length(pLeft * vec2(0.84, 1.28)) - 0.158 * rScale) * 0.86;

  vec2 pRight = uv - vec2(0.082, 0.018) * rScale;
  float dPoolRight = (length(pRight * vec2(0.84, 1.28)) - 0.156 * rScale) * 0.86;

  float kLobe = 0.076 * rScale;
  float d2D = sminSimple(dPoolMain, sminSimple(dPoolLeft, dPoolRight, kLobe), kLobe);

  // Bulbous high-surface-tension capillary rim profile (thickness ~ 0.098 at rest) gathering slightly as it draws upward
  float vScale = mix(rScale, sqrt(rScale), 0.32 * smoothstep(0.05, 0.45, rScale));
  float T = (0.049 + 0.005 * smoothstep(-0.08 * max(rScale, 0.04), 0.0, d2D)) * vScale;
  float centerSeat = 0.026 * vScale * max(0.0, 1.0 - dot(uvNorm, uvNorm) * 14.0);
  float hCamber = centerSeat - 0.004 * vScale * smoothstep(0.04, 0.18, uvNorm.y);
  float dCore2D = max(d2D + T, 0.0);
  float dh = h - hCamber;
  float dPad = (sqrt(dCore2D * dCore2D + dh * dh) - T) * sRest;

  outFloorClip = -(h - (hCamber - T * 0.42)) * sRest * smoothstep(0.55, 0.92, tMorph);
  return dPad;
}

// Pure 3D evaluation of the molten gallium puddle + upward-climbing capillary meniscus collar around the 3D crystal axes
// (100% aligned with each crystal's 3D axis — zero bloated upper sausages and zero UBO reads inside the march loop!).
float evalGalliumFastStep(
  vec3 qGa, float meltAmt, float rScale, float tMorph, float blobBell, vec3 coreAnchor,
  vec3 pA0, vec3 ba0, float rCap0,
  vec3 pA1, vec3 ba1, float rCap1,
  vec3 pA2, vec3 ba2, float rCap2,
  out float outMeltFactor
) {
  float floorClip = -1e9;
  vec2 puddleUV = vec2(0.0);
  float dPuddle = evalGalliumPuddleSDF(qGa, rScale, tMorph, coreAnchor, floorClip, puddleUV);

  vec3 pa0 = qGa - pA0;
  float d0 = length(pa0 - ba0 * clamp(dot(pa0, ba0) / max(dot(ba0, ba0), 1e-4), 0.0, 1.0)) - rCap0;
  vec3 pa1 = qGa - pA1;
  float d1 = length(pa1 - ba1 * clamp(dot(pa1, ba1) / max(dot(ba1, ba1), 1e-4), 0.0, 1.0)) - rCap1;
  vec3 pa2 = qGa - pA2;
  float d2 = length(pa2 - ba2 * clamp(dot(pa2, ba2) / max(dot(ba2, ba2), 1e-4), 0.0, 1.0)) - rCap2;

  float kBand = 0.072 * max(rScale, 0.25);
  float dBase = sminSimple(d0, sminSimple(d1, d2, kBand), kBand);
  float kMeniscus = (0.082 + 0.098 * clamp(meltAmt, 0.0, 1.0) + 0.068 * blobBell) * smoothstep(0.04, 0.36, rScale);
  // C1-smooth 3D capillary meniscus collar that visibly climbs up and fillets the lower crystal walls
  float uMen = (kMeniscus > 1e-4) ? (1.0 - smoothstep(-0.060, kMeniscus, dBase)) : 0.0;
  float meniscusLift = (0.62 + 0.36 * blobBell) * kMeniscus * uMen * uMen;
  float dLiquid = dPuddle - meniscusLift;
  outMeltFactor = smoothstep(0.02, 0.35, uMen);
  return dLiquid;
}

bool intersectGalliumMelt(
  vec3 ro, vec3 rd, float meltAmt,
  out int outId, out float outTHit, out float outTfHit, out int outK,
  out vec3 outN, out float outMeltFactor, out float outEdgeDist
) {
  outId = 0; outK = 0; outTHit = 1e9; outTfHit = 1e9;
  outN = vec3(0.0, 1.0, 0.0);
  outMeltFactor = 0.0;
  outEdgeDist = 1.0;

  float tMorph = clamp(meltAmt / 0.72, 0.0, 1.0);
  float rScale = pow(tMorph, 1.12) * smoothstep(0.03, 0.11, tMorph);
  // Bell curve of 3D liquid surface-tension rounding across the crystal spires during transition (0 at resting Liquid Gallium & 0 at final crystal)
  float blobBell = sin(tMorph * PI) * smoothstep(0.03, 0.24, tMorph);
  float gaFacetStyle = smoothstep(0.03, 0.42, tMorph);
  vec3 c0 = toGalliumFrame(body(0, 0).xyz);
  vec3 coreAnchor = c0;

  // 1. Exact analytical convex polytope intersection across active crystals
  int nc = clamp(int(f_crystals + 0.5), 0, MAX_CRYSTALS);
  bool hasPolyHit = false;
  int polyId = 0, polyK = 0;
  float polyT = 1e9, polyTf = 1e9;
  for (int i = 0; i < nc; i++) {
    if (!sphereHit(ro, rd, body(i, 0), polyT)) continue;
    float tn, tf; int kn, kf;
    if (convexHit(i, ro, rd, tn, tf, kn, kf)) {
      float tCand = max(tn, 0.0);
      if (tf > tCand + 1e-4 && tCand < polyT) {
        vec3 qCand = toGalliumFrame(ro + rd * tCand);
        float fClip = -1e9;
        vec2 puv = vec2(0.0);
        evalGalliumPuddleSDF(qCand, rScale, tMorph, coreAnchor, fClip, puv);
        if (fClip < 0.015) {
          polyT = tCand; polyTf = tf; polyId = i; polyK = kn;
          hasPolyHit = true;
        }
      }
    }
  }

  // 2. Compute solid / liquid-rounded crystal normal & analytical Euclidean edge distance at polyT (single 18-plane loop)
  vec3 polyN = vec3(0.0, 1.0, 0.0);
  float polyEdgeDist = 1.0;
  if (hasPolyHit) {
    float rPoly = max(body(polyId, 2).w, 0.02);
    float rRound = rPoly * (0.022 * gaFacetStyle + 0.34 * blobBell);
    vec3 pPoly = ro + rd * mix(polyT, 0.5 * (polyT + polyTf), 0.30 * blobBell);
    polyN = body(polyId, PA + polyK).xyz;
    float d2nd = 1e9;
    vec3 n2nd = polyN;
    vec3 nRound = polyN * (rRound * rRound);
    for (int k = 0; k < 18; k++) {
      if (k == polyK) continue;
      vec4 pl = body(polyId, PA + k);
      float dSigned = dot(pl.xyz, pPoly) - pl.w;
      float wR = max(0.0, dSigned + rRound);
      nRound += pl.xyz * (wR * wR);
      float cAng = dot(pl.xyz, polyN);
      if (cAng > 0.992) continue;
      float sinAng = sqrt(max(0.08, 1.0 - cAng * cAng));
      float dPl = max(0.0, -dSigned) / sinAng;
      if (dPl < d2nd) {
        d2nd = dPl;
        n2nd = pl.xyz;
      }
    }
    polyEdgeDist = d2nd;
    // Subtle cast-metal facet camber + smooth physical edge bevel (fading cleanly to 0 as tMorph -> 0.03)
    vec3 relCen = (pPoly - body(polyId, 0).xyz) / max(rPoly, 0.05);
    vec3 camberN = (relCen - polyN * dot(relCen, polyN)) * (0.082 * gaFacetStyle);
    float bevelW = (1.0 - smoothstep(0.0, rPoly * 0.040, d2nd)) * (0.36 * gaFacetStyle);
    vec3 nSharpBevel = normalize(mix(polyN + camberN, normalize(polyN + n2nd), bevelW));
    vec3 nBlob = normalize(nRound + polyN * 1e-4);
    polyN = normalize(mix(nSharpBevel, nBlob, smoothstep(0.01, 0.26, blobBell) * 0.88));
  }

  // Once rScale <= 1e-4, the shrinking puddle has completely drawn inside Crystal #0 (dPuddle == 1e4), so return the crystal directly
  if (rScale <= 1e-4) {
    if (!hasPolyHit) return false;
    outId = polyId; outTHit = polyT; outTfHit = polyTf;
    outK = CP + 3000 + clamp(polyK, 0, 16);
    outN = polyN; outMeltFactor = 0.0; outEdgeDist = polyEdgeDist;
    return true;
  }

  // 3. Bounding sphere around the molten gallium puddle & meniscus collar (tracking coreAnchor as rScale -> 0)
  vec3 qRo = toGalliumFrame(ro);
  vec3 qRd = toGalliumFrame(rd);
  vec3 pudSphCen = mix(coreAnchor, vec3(-0.01, -0.065, 0.045), smoothstep(0.0, 0.94, rScale));
  float pudSphRad = mix(0.12, 0.42, rScale);
  vec3 oc = qRo - pudSphCen;
  float b = dot(oc, qRd);
  float h = b * b - dot(oc, oc) + pudSphRad * pudSphRad;
  if (h <= 0.0) {
    if (!hasPolyHit) return false;
    outId = polyId; outTHit = polyT; outTfHit = polyTf;
    outK = CP + 3000 + clamp(polyK, 0, 16);
    outN = polyN; outMeltFactor = 0.0; outEdgeDist = polyEdgeDist;
    return true;
  }

  float s = sqrt(h);
  float t0 = max(0.0, -b - s);
  // Never march behind a solid front crystal face (polyT), preventing internal puddle X-raying as the pool draws into Crystal #0
  float t1 = hasPolyHit ? min(-b + s, polyT) : (-b + s);

  // Precompute 3 coaxial 3D crystal spire axes in qGa frame ONCE outside the march loop for capillary meniscus lift
  float spireLenScale = mix(0.14, 0.44, blobBell);
  vec3 ax0 = toGalliumFrame(body(0, 1).xyz);
  float r0 = max(body(0, 2).w, 0.05);
  float L0 = max(body(0, 1).w, 0.12) * spireLenScale;
  vec3 pA0 = c0 - ax0 * (0.45 * L0);
  vec3 ba0 = ax0 * L0;
  float rCap0 = r0 * mix(0.92, 0.82, blobBell);

  vec3 c1 = (nc > 1) ? toGalliumFrame(body(1, 0).xyz) : c0;
  vec3 ax1 = (nc > 1) ? toGalliumFrame(body(1, 1).xyz) : ax0;
  float r1 = (nc > 1) ? max(body(1, 2).w, 0.05) : r0;
  float L1 = (nc > 1) ? (max(body(1, 1).w, 0.12) * spireLenScale) : L0;
  vec3 pA1 = c1 - ax1 * (0.45 * L1);
  vec3 ba1 = ax1 * L1;
  float rCap1 = r1 * mix(0.92, 0.82, blobBell);

  int idx2 = (nc > 3) ? 3 : 2;
  vec3 c2Spire = (nc > 2) ? toGalliumFrame(body(idx2, 0).xyz) : c0;
  vec3 ax2 = (nc > 2) ? toGalliumFrame(body(idx2, 1).xyz) : ax0;
  float r2 = (nc > 2) ? max(body(idx2, 2).w, 0.05) : r0;
  float L2 = (nc > 2) ? (max(body(idx2, 1).w, 0.12) * spireLenScale) : L0;
  float wSpire2 = smoothstep(0.02, 0.30, blobBell);
  vec3 saddleCen = mix(coreAnchor, vec3(-0.005, 0.038, 0.048), rScale);
  vec3 pA2 = mix(saddleCen, c2Spire - ax2 * (0.45 * L2), wSpire2);
  vec3 ba2 = ax2 * (L2 * wSpire2);
  float rCap2 = mix(0.122 * rScale, r2 * 0.82, wSpire2);

  // Early exit for upper crystal hits well above the puddle/meniscus zone when resting (blobBell < 0.15)
  if (hasPolyHit && blobBell < 0.15) {
    vec3 qPolyHit = qRo + qRd * polyT;
    float fClipP = -1e9;
    vec2 puvP = vec2(0.0);
    float dPudHit = evalGalliumPuddleSDF(qPolyHit, rScale, tMorph, coreAnchor, fClipP, puvP);
    if (dPudHit > 0.13) {
      outId = polyId; outTHit = polyT; outTfHit = polyTf;
      outK = CP + 3000 + clamp(polyK, 0, 16);
      outN = polyN; outMeltFactor = 0.0; outEdgeDist = polyEdgeDist;
      return true;
    }
  }

  float t = t0;
  float tPrev = t0;
  bool hitPuddle = false;

  if (t1 > t0) {
    for (int st = 0; st < 32; st++) {
      if (t > t1) break;
      float mF = 0.0;
      float d = evalGalliumFastStep(qRo + qRd * t, meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, mF);
      if (d <= 0.0006) {
        hitPuddle = true;
        break;
      }
      tPrev = t;
      t += max(0.0014, d * 0.86);
    }
  }

  if (!hitPuddle) {
    if (!hasPolyHit) return false;
    // Smooth capillary wetting fillet on the solid crystal wall right where it meets the liquid pool
    vec3 qPoly = qRo + qRd * polyT;
    float mFPoly = 0.0;
    float dLiqPoly = evalGalliumFastStep(qPoly, meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, mFPoly);
    float wetW = (1.0 - smoothstep(0.005, 0.068, dLiqPoly)) * smoothstep(0.48, 0.90, rScale);
    if (wetW > 0.005) {
      float epsW = 0.0018;
      float dm1, dm2, dm3;
      float gxW = evalGalliumFastStep(qPoly + vec3(epsW, 0.0, 0.0), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm1) - dLiqPoly;
      float gyW = evalGalliumFastStep(qPoly + vec3(0.0, epsW, 0.0), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm2) - dLiqPoly;
      float gzW = evalGalliumFastStep(qPoly + vec3(0.0, 0.0, epsW), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm3) - dLiqPoly;
      vec3 nGaW = normalize(vec3(gxW, gyW, gzW) + vec3(0.0, 1e-6, 0.0));
      vec3 nLiqW = normalize(vec3(
        0.186646 * nGaW.x + 0.982427 * nGaW.z,
        nGaW.y,
        -0.982427 * nGaW.x + 0.186646 * nGaW.z
      ));
      polyN = normalize(mix(polyN, nLiqW, wetW * 0.86));
    }
    outId = polyId; outTHit = polyT; outTfHit = polyTf;
    outK = CP + 3000 + clamp(polyK, 0, 16);
    outN = polyN; outMeltFactor = wetW * 0.92; outEdgeDist = polyEdgeDist;
    return true;
  }

  float tLo = tPrev, tHi = t;
  for (int bs = 0; bs < 3; bs++) {
    float tMid = 0.5 * (tLo + tHi);
    float mFMid = 0.0;
    if (evalGalliumFastStep(qRo + qRd * tMid, meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, mFMid) <= 0.0002) {
      tHi = tMid;
    } else {
      tLo = tMid;
    }
  }
  float tFinal = 0.5 * (tLo + tHi);
  vec3 qHit = qRo + qRd * tFinal;
  float mF0 = 1.0;
  float d0 = evalGalliumFastStep(qHit, meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, mF0);

  // Compute 3D gradient normal in qGa space using 3 forward differences (zero UBO reads!)
  float eps = 0.0018;
  float dm1, dm2, dm3;
  float gx = evalGalliumFastStep(qHit + vec3(eps, 0.0, 0.0), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm1) - d0;
  float gy = evalGalliumFastStep(qHit + vec3(0.0, eps, 0.0), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm2) - d0;
  float gz = evalGalliumFastStep(qHit + vec3(0.0, 0.0, eps), meltAmt, rScale, tMorph, blobBell, coreAnchor, pA0, ba0, rCap0, pA1, ba1, rCap1, pA2, ba2, rCap2, dm3) - d0;
  vec3 nGaLocal = normalize(vec3(gx, gy, gz) + vec3(0.0, 1e-6, 0.0));
  // Transform normal back from qGa to cluster space p
  vec3 nPuddleWorld = normalize(vec3(
    0.186646 * nGaLocal.x + 0.982427 * nGaLocal.z,
    nGaLocal.y,
    -0.982427 * nGaLocal.x + 0.186646 * nGaLocal.z
  ));

  vec3 nFinal = nPuddleWorld;
  float endLock = 1.0 - smoothstep(0.04, 0.16, rScale);
  if (hasPolyHit) {
    float wallProx = 1.0 - smoothstep(0.0, 0.018, abs(polyT - tFinal));
    float lockW = max(wallProx * 0.65, endLock);
    nFinal = normalize(mix(nPuddleWorld, polyN, lockW));
    tFinal = mix(tFinal, polyT, endLock);
  }
  if (dot(nFinal, rd) > -0.04) {
    nFinal = normalize(nFinal - rd * (dot(nFinal, rd) + 0.04));
  }

  outId = hasPolyHit ? polyId : 0;
  outTHit = tFinal;
  outTfHit = hasPolyHit ? max(tFinal + 0.12, polyTf) : (tFinal + 0.16);
  outK = CP + 3000 + 17;
  outN = nFinal;
  outMeltFactor = 1.0 - endLock;
  outEdgeDist = hasPolyHit ? mix(1.0, polyEdgeDist, endLock) : 1.0;
  return true;
}

bool traceScene(vec3 ro, vec3 rd, int skipMask, out int id, out float tHit, out float tfHit, out int kHit, out vec3 outVoidN, out float outVoidBorder, out float outVoidSeam) {
  bool hit = false; id = 0; kHit = 0; tHit = 1e9; tfHit = 1e9;
  outVoidN = vec3(0.0, 1.0, 0.0);
  outVoidBorder = 1e9;
  outVoidSeam = 1e9;
  if (u_melt > 0.0216 && skipMask == 0) {
    int mId, mK;
    float mT, mTf, mMeltF, mEdgeD;
    vec3 mN;
    if (intersectGalliumMelt(ro, rd, u_melt, mId, mT, mTf, mK, mN, mMeltF, mEdgeD)) {
      id = mId;
      tHit = mT;
      tfHit = mTf;
      kHit = mK;
      outVoidN = mN;
      outVoidBorder = mMeltF;
      outVoidSeam = mEdgeD;
      return true;
    }
  }
  float tn, tf; int kn, kf;
  int nc = clamp(int(f_crystals + 0.5), 0, MAX_CRYSTALS);
  for (int i = 0; i < nc; i++) {
    if ((skipMask & (1 << i)) != 0 || !sphereHit(ro, rd, body(i, 0), tHit)) continue;
    if (convexHit(i, ro, rd, tn, tf, kn, kf)) {
      if (max(tn, 0.0) >= tHit) continue;
      vec3 candVoidN = vec3(0.0, 1.0, 0.0);
      float candBorder = 1e9, candSeam = 1e9;
      if (u_occlusion > 0.002) {
        if (!applyOcclusionVoidCSG(i, ro, rd, tn, tf, kn, candVoidN, candBorder, candSeam)) continue;
      }
      if (u_hopper > 0.01) {
        if (!intersectBismuthHopper(i, ro, rd, u_hopper, tHit, tn, tf, kn, candVoidN)) continue;
      } else if (u_cathedral > 0.005) {
        float cathEdge = 1e9;
        if (!intersectCathedralSpire(i, ro, rd, u_cathedral, tHit, tn, tf, kn, candVoidN, cathEdge)) continue;
        candBorder = min(candBorder, cathEdge);
      }
      float tCand = max(tn, 0.0);
      if (tf > tCand + 1e-4 && tCand < tHit) {
        tHit = tCand; tfHit = tf; id = i; kHit = kn;
        outVoidN = candVoidN; outVoidBorder = candBorder; outVoidSeam = candSeam;
        hit = true;
      }
    }
  }
  return hit;
}

float fresnel(float cosi, float eta) {
  float r0 = (1.0 - eta) / (1.0 + eta);
  r0 *= r0;
  return r0 + (1.0 - r0) * pow(1.0 - clamp(cosi, 0.0, 1.0), 5.0);
}

float crystalUid(int id) {
  return max(1.0, body(id, 2).z);
}

// Monotone bottom-to-top height [0 = bottom of crystal/cluster, 1 = top termination tip].
// Lower 1/3 (hgt < 0.34) is clear quartz root; upper 2/3 (0.34 .. 1.0) grades smoothly into deep purple tip.
float crystalHeight01(int id, vec3 p) {
  vec4 ax = body(id, 1), rt = body(id, 3);
  float L = max(ax.w, 0.08);
  float axial = dot(p - rt.xyz, ax.xyz);
  float hAxial = clamp((axial + 0.42 * L) / (1.42 * L), 0.0, 1.0);
  float hWorld = clamp((p.y + 0.36 * L) / (1.18 * L), 0.0, 1.0);
  float zWeight = clamp(u_colourZoning, 0.0, 1.0);
  return clamp(mix(hAxial, min(hAxial, hWorld) * 0.65 + hWorld * 0.35, zWeight), 0.0, 1.0);
}

vec3 zonedGemSigma(float hgt) {
  float raw01 = clamp(u_rawAmount / 0.4, 0.0, 1.0);
  float zWeight = clamp(u_colourZoning, 0.0, 1.0) * raw01;
  float zRamp = smoothstep(0.24, 0.68, hgt);
  float cMix = clamp(u_biColour * (1.0 - zRamp) * zWeight, 0.0, 1.0);
  float zAtten = mix(1.0, 0.22 + 0.78 * zRamp, zWeight);
  return mix(f_sigma, f_sigmaCore, cMix) * zAtten;
}

vec3 insideVolume(int id, vec3 ro, vec3 rd, float len, float jitter, int maxSteps, out vec3 trans) {
  vec4 ax = body(id, 1), side = body(id, 2), rt = body(id, 3);
  vec3 axis = ax.xyz, root = rt.xyz;
  float L = ax.w, r = side.w, rnd = rt.w;
  float hueOff = side.y;
  vec3 off = vec3(rnd * 17.3, rnd * 31.1, rnd * 7.7);
  vec3 scat = vec3(0.0);
  vec3 T = vec3(1.0);
  vec3 lightIn = f_ambient * u_inclusionLight;
  float opticalScale = clamp(0.28 / max(r, 0.035), 1.5, 5.8);

  bool isPrimaryVol = (maxSteps > 5) || (u_isMobile > 0.5 && maxSteps >= 3);
  float hasFlash = (isPrimaryVol && u_flash > 0.01) ? step(0.68, fract(rnd * 17.37 + 0.23)) : 0.0;
  vec3 p0 = body(id, PA).xyz;
  vec3 t1 = p0 - axis * dot(p0, axis);
  float t1Len = length(t1);
  t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
  vec3 t2 = cross(axis, t1);
  float flashAz = rnd * TAU;
  float flashEl = (fract(rnd * 7.13) - 0.5) * 0.55;
  vec3 flashViewDir = normalize((t1 * cos(flashAz) + t2 * sin(flashAz)) * cos(flashEl) + axis * sin(flashEl));
  float cosFlashAng = max(0.0, dot(-rd, flashViewDir));
  float schillerResonance = hasFlash * pow(cosFlashAng, 42.0);
  float flashCenterHgt = 0.35 + 0.30 * fract(rnd * 9.13);
  vec3 flashCenterOffset = (t1 * cos(flashAz + 0.6) + t2 * sin(flashAz + 0.6)) * (r * 0.18);

  int n = max(2, maxSteps);
  float dt = len / float(n);
  float absorbDt = max(len, r * 0.45) / float(n);
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float glimmer = clamp(u_glimmer, 0.0, 2.0);
  float g1 = clamp(glimmer, 0.0, 1.0);
  float effOpticalScale = mix(opticalScale, 1.0, g1);
  float effAbsorbDt = mix(absorbDt, dt, g1);
  float rawAmt = clamp(u_rawAmount, 0.0, 2.0);
  float raw01 = clamp(rawAmt / 0.4, 0.0, 1.0);
  float rawScale = mix(0.08, 1.0, clamp(rawAmt, 0.0, 1.0)) * mix(1.0, 1.85, max(0.0, rawAmt - 1.0));
  float inclScale = mix(rawScale, max(rawScale, 0.85), g1);
  float effMilk = u_milk * (1.0 - 0.25 * dither) * inclScale;
  float effVeils = u_veils * (1.0 - 0.20 * dither) * inclScale;
  float hgtStart = crystalHeight01(id, ro);
  float hgtEnd = crystalHeight01(id, ro + rd * len);
  bool needInclTex = (effMilk > 0.01 || effVeils > 0.01 || u_iridescence > 0.01 || u_colourZoning > 0.01 || g1 > 0.01);
  vec3 cMidVol = root + axis * (0.52 * L);
  float invLVol = 1.0 / max(L, 0.14);
  float inclOctBlend = clamp((u_inclusionScale - 1.5) / 2.5, 0.0, 1.0);
  vec3 fGemBodyCol = exp(-f_sigma * 0.38);
  vec3 fGemBodyNorm = normalize(fGemBodyCol + vec3(0.04)) * 1.28;
  float fSigmaLen = length(f_sigma);
  bool hasBiCol = (u_biColour > 0.001);
  for (int s = 0; s < n; s++) {
    float stepF = float(s) + jitter;
    vec3 pp = ro + rd * (dt * stepF);
    vec3 rel = pp - root;
    float axial = dot(rel, axis);
    float hgtSample = mix(hgtStart, hgtEnd, stepF / float(n));
    float hgt = (u_colourZoning > 0.5) ? mix(hgtStart, hgtSample, 0.50) : hgtSample;
    if (!isPrimaryVol && u_colourZoning > 0.5) {
      hgt = min(hgt, hgtStart + 0.10);
    }
    float radFrac = clamp(length(rel - axis * axial) / max(r, 1e-3), 0.0, 1.0);
    float zRamp = smoothstep(0.24, 0.68, hgt);
    float baseZone = raw01 * (1.0 - zRamp);
    vec3 baseSigma = f_sigma;
    vec3 gemBodyNorm = fGemBodyNorm;
    float baseSigmaLen = fSigmaLen;
    if (hasBiCol) {
      float coreMix = clamp(u_biColour * max(baseZone * u_colourZoning, (1.0 - radFrac * 0.75) * 0.22), 0.0, 1.0);
      baseSigma = mix(f_sigma, f_sigmaCore, coreMix);
      vec3 gemBodyCol = exp(-baseSigma * 0.38);
      gemBodyNorm = normalize(gemBodyCol + vec3(0.04)) * 1.28;
      baseSigmaLen = length(baseSigma);
    }
    vec3 tintedLightIn = lightIn * mix(vec3(1.0), gemBodyNorm, clamp(baseSigmaLen * 0.24 * zRamp, 0.0, 0.88));
    vec3 effLightIn = mix(tintedLightIn, mix(lightIn, tintedLightIn, 0.28), g1);
    vec3 relMidV = (pp - cMidVol) * invLVol;
    vec3 pCry = vec3(dot(relMidV, t1), dot(relMidV, axis), dot(relMidV, t2));
    vec4 v = needInclTex
      ? mix(texture(u_incl, pCry * 1.25 + off), texture(u_incl, pCry * 2.50 + off + 0.37), inclOctBlend)
      : vec4(0.0, 0.0, 0.5, 0.0);
    float a = 0.0;
    if (effMilk > 0.01 || effVeils > 0.01) {
      float rootCloud = mix(1.0, (0.20 + 2.2 * pow(1.0 - hgt, 1.95)) * (0.45 + 0.85 * v.g), u_milkBase);
      float wisp = pow(max(v.r, 0.0), 1.25) * (0.45 + 0.75 * v.b);
      float milk = effMilk * mix(v.r, wisp, u_milkBase * baseZone) * rootCloud;
      float veil = effVeils * pow(max(v.g, 0.0), 1.15) * (0.65 + 0.65 * v.b);
      float dens = (milk + veil) * mix(5.2, 6.0, g1);
      a = 1.0 - exp(-dens * dt);
      float bloomSparkle = u_bloom > 0.0 ? (1.0 + u_bloom * 1.6 * pow(v.a, 8.0)) : 1.0;
      vec3 vSparkCol = mix(vec3(1.0), spectralRainbow(v.a * 7.3 + hgt * 2.1 + dot(rd, axis) * 1.7 + hueOff), 0.55 * g1) * (1.0 + 3.5 * glimmer * pow(v.a, 8.0));
      vec3 milkTint = mix(effLightIn, lightIn * vec3(0.96, 0.98, 1.0) * mix(1.0, 1.18, raw01), clamp(baseZone * u_milkBase * 0.85, 0.0, 0.90));
      scat += T * a * milkTint * (1.0 + mix(2.8, 3.2, g1) * veil * pow(v.a, 6.0) * bloomSparkle * vSparkCol);
    }
    if (schillerResonance > 1e-3) {
      vec3 radialVec = (rel - axis * axial) - flashCenterOffset;
      float dAxial = (hgt - flashCenterHgt) / 0.095;
      float dRadial = length(radialVec) / max(r * 0.26, 1e-3);
      float singleGlow = exp(-(dAxial * dAxial + dRadial * dRadial));
      float coreEnvelope = (1.0 - smoothstep(0.25, 0.92, radFrac)) * smoothstep(0.08, 0.24, hgt) * (1.0 - smoothstep(0.74, 0.92, hgt));
      vec3 coreGemGlow = gemBodyNorm;
      vec3 schillerTint = spectralRainbow(hueOff + cosFlashAng * 0.35);
      vec3 deepFlashCol = mix(coreGemGlow, schillerTint * (coreGemGlow + vec3(0.22)), 0.32);
      float flashPower = u_flash * (0.65 + 0.45 * u_flash);
      scat += T * coreEnvelope * singleGlow * schillerResonance * deepFlashCol * (f_ambient * 2.2 + vec3(0.38)) * flashPower * effOpticalScale * dt * 2.4;
    }
    if (u_iridescence > 0.01 && u_rainbow * (1.0 - 0.85 * g1) > 0.001) {
      float iridPhase = hgt * mix(1.85, 1.45, dither) + radFrac * mix(1.25, 0.95, dither) + dot(rd, axis) * mix(1.15, 0.95, dither) + hueOff + v.b * mix(0.75, 0.55, dither);
      vec3 iridWave = spectralRainbow(iridPhase);
      float veilBand = (0.03 + 0.14 * v.g + 0.06 * v.r) * clamp(u_iridescence * 0.48, 0.0, 1.3) * u_rainbow * (1.0 - 0.85 * g1) * (1.0 - 0.32 * dither);
      scat += T * iridWave * effLightIn * veilBand * dt * mix(2.1, 1.45, dither);
    }
    float chevronBand = 1.0 + 0.06 * rawAmt * u_colourZoning * sin((hgt * 10.0 - radFrac * 3.5) * PI) * zRamp;
    float cleanTipZone = 1.0 + 0.35 * u_colourZoning * (2.0 * hgt - 1.0);
    float rawTipZone = mix(1.0, 0.24 + 1.12 * pow(zRamp, 1.05), u_colourZoning);
    float tipZone = mix(cleanTipZone, rawTipZone, raw01);
    vec3 minQuartzSigma = vec3(0.28, 0.32, 0.26) * g1;
    vec3 sigma = max(baseSigma * effOpticalScale * tipZone * chevronBand * (0.76 + 0.48 * v.b), minQuartzSigma);
    T *= (1.0 - a) * exp(-sigma * effAbsorbDt);
  }

  // Warped 3D internal fracture / cleavage-crack sheets (replaces 1D wire-line rutilations & flat cleavage discs)
  // Subtle translucent hairline fracture planes (~6% veil) when off-angle; flares with golden/silvery/iridescent
  // specular reflection and thin-film Newton-ring glints when 3D crack normal catches the light as crystal rotates.
  float effHairline = clamp(u_hairlineFractures * u_patinaMacro * u_surfaceMacro, 0.0, 2.0);
  float effNeedles = clamp(u_needles / 6.0, 0.0, 2.0);
  float effCrackDrive = clamp(u_fractures * 1.35 + effNeedles * 0.65 + effHairline * 0.55 + (u_iridescence * 0.16 + u_flash * 0.18) * max(u_rainbow, g1), 0.0, 2.2);
  if (effCrackDrive > 0.01 && len > 1e-4) {
    float uid = crystalUid(id);
    float bounceCleaveW = isPrimaryVol ? 1.0 : mix(0.42 * (1.0 - 0.65 * dither), 0.70, g1);

    // Studio key, rim, and fill light directions transformed into crystal space
    vec3 lightDir1 = normalize(vec3(f_lights[0], f_lights[1], f_lights[2]) * f_toWorld);
    vec3 lightDir2 = normalize(vec3(f_lights[2 * LS], f_lights[2 * LS + 1], f_lights[2 * LS + 2]) * f_toWorld);
    vec3 lightDir3 = normalize(vec3(f_lights[3 * LS], f_lights[3 * LS + 1], f_lights[3 * LS + 2]) * f_toWorld);
    vec3 lightDir4 = normalize(vec3(f_lights[LS], f_lights[LS + 1], f_lights[LS + 2]) * f_toWorld);

    float needleTintAmt = clamp(u_needles * 0.15, 0.0, 0.88);
    vec3 rutileGold = normalize(lin(u_needleColour) + vec3(0.08)) * 1.38;
    vec3 baseCrackTint = mix(vec3(0.95, 0.97, 1.0), rutileGold, needleTintAmt);
    float widthScale = clamp(u_needleWidth / 0.0025, 0.55, 2.2);
    float crackRainbowGate = max(u_rainbow, g1);
    float invPatchScale = 1.0 / max(r * 1.45, 0.045);

    int maxCrackFamilies = isPrimaryVol
      ? ((u_isMobile > 0.5) ? 2 : ((u_needles >= 3.0 || u_fractures >= 0.30) ? 4 : 3))
      : 1;

    for (int k = 0; k < maxCrackFamilies; k++) {
      vec3 hSeed = vec3(uid * 19.3 + float(k) * 37.7, f_seed + float(k) * 17.1, 93.0 + float(k) * 23.0);
      float famRoll = hash(hSeed + 0.7);
      float famWeight = smoothstep(famRoll * 0.42 - 0.06, famRoll * 0.42 + 0.24, effCrackDrive) * bounceCleaveW;
      if (famWeight <= 1e-4) continue;

      // Rhombohedral / conchoidal 3D cleavage plane orientation in crystal space
      float azK = hash(hSeed) * TAU + float(k) * 2.094395;
      float tiltK = mix(0.42, 1.15, hash(hSeed + 1.3));
      float axialSign = ((k & 1) == 0) ? 1.0 : -1.0;
      vec3 nLocal0 = normalize(vec3(cos(azK) * sin(tiltK), axialSign * cos(tiltK), sin(azK) * sin(tiltK)));
      vec3 crackNormal_k = normalize(t1 * nLocal0.x + axis * nLocal0.y + t2 * nLocal0.z);
      vec3 crackTan = normalize(cross(crackNormal_k, axis + t1 * 0.35));
      vec3 crackBit = cross(crackNormal_k, crackTan);

      // Anchor point fc_k of this 3D internal fracture sheet inside the crystal body
      float hgtAnchor = mix(0.20, 0.78, (float(k) + 0.22 + 0.55 * hash(hSeed + 3.1)) / float(max(maxCrackFamilies, 2)));
      vec3 fc_k = root + axis * (L * hgtAnchor);
      vec3 seed = vec3(uid * 5.7 + float(k) * 19.1, f_seed * 0.73 + float(k) * 8.3, 41.7 + float(k) * 13.3);
      float offset_k = -dot(fc_k - root, crackNormal_k) - 0.05 * r;

      // Exact ray intersection with gently curved, monotone 3D cleavage-crack sheet (zero closed-loop bullseyes!)
      float dn = dot(rd, crackNormal_k);
      float signedDn = (dn >= 0.0 ? 1.0 : -1.0) * max(abs(dn), 0.08);
      float t = dot(fc_k - ro, crackNormal_k) / signedDn;
      if (t < -0.12 * len || t > 1.12 * len) continue;

      // 2-step Newton refinement on the gently warped 3D crack surface:
      // crackPlane = abs(dot(q, crackNormal_k) + offset_k + 0.18 * n3(q * 3.5))
      float invL = 0.55 / max(L, 0.15);
      for (int it = 0; it < 2; it++) {
        vec3 hpIt = ro + rd * clamp(t, 0.0, len);
        vec3 qIt = hpIt - root;
        vec3 relCIt = (hpIt - fc_k) * invL;
        vec3 qLocIt = vec3(dot(relCIt, t1), dot(relCIt, axis), dot(relCIt, t2));
        float uIt = dot(hpIt - fc_k, crackTan) * invPatchScale;
        float vIt = dot(hpIt - fc_k, crackBit) * invPatchScale;
        float gentleArc = r * (0.07 * (uIt * uIt - 0.30) + 0.04 * sin(vIt * 1.9 + seed.x));
        float signedSurface = dot(qIt, crackNormal_k) + offset_k + gentleArc + 0.10 * r * n3(qLocIt * 3.5 + seed);
        t -= signedSurface / signedDn;
      }
      if (t <= 0.005 * len || t >= 0.995 * len) continue;

      vec3 hp = ro + rd * t;
      vec3 q = hp - root;
      vec3 relC = (hp - fc_k) * invL;
      vec3 qLoc = vec3(dot(relC, t1), dot(relC, axis), dot(relC, t2));
      float u = dot(hp - fc_k, crackTan) * invPatchScale;
      float v = dot(hp - fc_k, crackBit) * invPatchScale;
      float gentleArc = r * (0.07 * (u * u - 0.30) + 0.04 * sin(v * 1.9 + seed.x));
      float crackPlane = abs(dot(q, crackNormal_k) + offset_k + gentleArc + 0.10 * r * n3(qLoc * 3.5 + seed));
      float sheetHit = 1.0 - smoothstep(0.004 * r, 0.024 * r, crackPlane);
      if (sheetHit <= 1e-4) continue;

      // Monotone wall-anchored 3D fracture patch envelope (zero closed loops, zero 1D painted lines!)
      // Each crack is a true 2D curved cleavage sheet inside the 3D crystal that feathers out organically.
      float uSign = ((k & 1) == 0) ? 1.0 : -1.0;
      float uCoord = u * uSign;
      float patchNoise = n3(qLoc * 2.2 + seed);
      float patchFine = n3(qLoc * 5.8 + seed * 1.73);
      float patchMicro = n3(qLoc * 12.5 + seed * 2.41);
      float uFront = mix(-0.02, 0.42, hash(hSeed + 5.3))
                   + 0.24 * sin(v * 2.1 + seed.x)
                   + 0.12 * cos(v * 4.3 + seed.y)
                   + 0.22 * (patchNoise - 0.5)
                   + 0.10 * (patchFine - 0.5);
      float frontDist = uFront - uCoord;
      float patchField = frontDist + 0.14 + 0.12 * (patchFine - 0.5);
      float patchEnv = smoothstep(0.08, 0.45, patchField);
      float vSpan = 1.0 - smoothstep(0.52, 0.94, abs(v + 0.18 * (patchNoise - 0.5)));
      float endTaper = smoothstep(0.008 * len, 0.05 * len, t) * smoothstep(0.008 * len, 0.05 * len, len - t);
      // Feathery crystalline cleavage sheet texture (internal air-film pockets & stepped cleavage terraces)
      float featherSheet = smoothstep(0.22, 0.76, patchFine * 0.62 + patchMicro * 0.38) * (0.55 + 0.45 * widthScale);
      float mask = famWeight * sheetHit * patchEnv * vSpan * endTaper;
      if (mask <= 1e-4) continue;

      // 3D crack sheet normal nCrack (gently curved conchoidal surface + smooth step-fracture facets, zero floor() jump)
      float epsP = 0.028;
      vec3 cTanLoc = vec3(dot(crackTan, t1), dot(crackTan, axis), dot(crackTan, t2));
      vec3 cBitLoc = vec3(dot(crackBit, t1), dot(crackBit, axis), dot(crackBit, t2));
      float dU = n3((qLoc + cTanLoc * epsP) * 5.2 + seed) - n3((qLoc - cTanLoc * epsP) * 5.2 + seed);
      float dV = n3((qLoc + cBitLoc * epsP) * 5.2 + seed) - n3((qLoc - cBitLoc * epsP) * 5.2 + seed);
      float pf5 = patchFine * 5.0;
      float smoothStair5 = pf5 - 0.12 * sin(TAU * pf5);
      float terraceStep = smoothStair5 * 0.025 + (patchMicro - 0.5) * 0.08;
      vec3 nCrack = normalize(
        crackNormal_k
        + crackTan * (0.10 * uCoord + dU * 0.22 + terraceStep)
        + crackBit * (0.07 * cos(v * 1.9 + seed.x) + dV * 0.22 - terraceStep * 0.7)
      );
      if (dot(rd, nCrack) > 0.0) nCrack = -nCrack;

      // View-dependent light-catching specular glint:
      // Almost invisible (~5-8% faint ghostly veil) when viewed off-angle;
      // flashes brilliantly when reflCrack = reflect(rd, nCrack) aligns with studio lights or grazing TIR.
      vec3 reflCrack = reflect(rd, nCrack);
      float catchLight = pow(max(0.0, dot(reflCrack, lightDir1)), 34.0)
                       + 0.8 * pow(max(0.0, dot(reflCrack, lightDir2)), 28.0)
                       + 0.16 * pow(1.0 - abs(dot(rd, nCrack)), 8.0);
      catchLight += 0.65 * pow(max(0.0, dot(reflCrack, lightDir3)), 30.0)
                  + 0.45 * pow(max(0.0, dot(reflCrack, lightDir4)), 28.0);

      // Modulate crack brightness by (0.04 + 2.4 * catchLight):
      // Off-angle: subtle translucent feathery fracture plane; On-angle: bright specular cleavage flash!
      float crackMod = (0.04 * (0.45 + 0.65 * featherSheet) + 2.4 * catchLight * (0.65 + 0.55 * featherSheet));

      // Subtle thin-film Newton-ring rainbow interference across the crack face when it catches the light
      float cosCrack = abs(dot(rd, nCrack));
      float filmPhase = (frontDist * 0.85 + v * 0.35 + patchNoise * 0.45 + patchFine * 0.25) * (0.45 + 0.55 * cosCrack) + hueOff + float(k) * 0.31;
      vec3 newtonRings = 0.5 + 0.5 * cos(TAU * (filmPhase + vec3(0.0, 0.33, 0.67)));
      vec3 thinFilmCol = mix(spectralRainbow(filmPhase), newtonRings, mix(0.45, 0.82, g1));
      float iridWeight = clamp((0.52 + 0.58 * u_iridescence + 0.28 * u_fractures) * max(crackRainbowGate, 0.78) * (1.0 - 0.38 * needleTintAmt), 0.0, 0.94);
      float glintIrid = iridWeight * smoothstep(0.08, 0.52, catchLight);
      vec3 crackCol = mix(baseCrackTint, thinFilmCol * (baseCrackTint + vec3(0.18)), glintIrid);

      vec3 rawRefl = env(reflCrack, mix(0.025, 0.045, dither));
      vec3 reflComp = rawRefl / (1.0 + 0.55 * max(rawRefl.r, max(rawRefl.g, rawRefl.b)));
      vec3 envIllum = mix(
        f_ambient * 0.45 + vec3(0.12),
        reflComp * 0.90 + f_ambient * 0.45 + vec3(0.24),
        clamp(catchLight * 1.5, 0.0, 1.0)
      );
      float glowScale = mix(max(u_fractureGlow, 1.15) * 0.62, min(max(u_fractureGlow, 1.15), 2.2) * 0.48, dither) * (1.0 + 0.20 * effNeedles);
      scat += T * mask * crackMod * glowScale * crackCol * envIllum;
      T *= 1.0 - clamp(mask * (0.025 * featherSheet + 0.24 * clamp(catchLight, 0.0, 1.5)), 0.0, 0.48);
    }
  }
  trans = T;
  return scat;
}

void insidePrimaryExtras(int id, vec3 ro, vec3 rd, float len, float travelled, inout vec3 scat, inout vec3 T) {
  vec4 ax = body(id, 1), side = body(id, 2), rt = body(id, 3);
  vec3 axis = ax.xyz, root = rt.xyz;
  float L = ax.w, r = side.w, rnd = rt.w;
  float uid = crystalUid(id);
  float hueOff = side.y;
  vec3 off = vec3(rnd * 17.3, rnd * 31.1, rnd * 7.7);
  vec3 lightIn = f_ambient * u_inclusionLight;
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float rawAmt = clamp(u_rawAmount, 0.0, 2.0);
  float effPhantom = u_phantom * (1.0 - 0.35 * dither) * mix(0.12, 1.0, clamp(rawAmt, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, rawAmt - 1.0));

  // Multi-generation nested phantoms (evaluated once on primary ray outside bounce loop)
  float phRateRoll = hash(vec3(uid, f_seed, 97.0));
  float phRateW = smoothstep(phRateRoll - 0.14, phRateRoll + 0.14, u_phantomRate);
  if (effPhantom > 0.01 && phRateW > 1e-4) {
    vec3 p0Ph = body(id, PA).xyz;
    vec3 t1Ph = p0Ph - axis * dot(p0Ph, axis);
    float t1PhLen = length(t1Ph);
    t1Ph = t1PhLen > 1e-4 ? (t1Ph / t1PhLen) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
    vec3 t2Ph = cross(axis, t1Ph);
    float invLPh = 1.0 / max(L, 0.14);
    vec3 phTint = exp(-mix(f_sigma, f_sigmaCore, 0.4) * 0.35);
    vec3 phLightIn = lightIn * mix(vec3(1.0), normalize(phTint + vec3(0.05)) * 1.25, clamp(length(f_sigma) * 0.22, 0.0, 0.85));
    int maxPhLayers = clamp(int(ceil(u_phantomLayers)), 0, (u_isMobile > 0.5) ? 2 : 4);
    for (int layer = 0; layer < maxPhLayers; layer++) {
      float layerWeight = clamp(u_phantomLayers - float(layer), 0.0, 1.0);
      if (layerWeight <= 1e-4) break;
      float pScale = u_phantomSize * (1.0 - 0.21 * float(layer));
      if (pScale <= 0.12) break;
      vec3 c = root + axis * L * (0.45 - 0.06 * float(layer));
      float tn = -1e9, tf = 1e9;
      vec3 nEnter = axis, nExit = axis;
      for (int k = 0; k < 18; k++) {
        vec4 pl = body(id, PA + k);
        float d = dot(pl.xyz, c) + (pl.w - dot(pl.xyz, c)) * pScale;
        float dn = dot(pl.xyz, rd), dist = d - dot(pl.xyz, ro);
        if (abs(dn) < 1e-8) { if (dist < 0.0) { tn = 1e9; } continue; }
        float t = dist / dn;
        if (dn < 0.0) { if (t > tn) { tn = t; nEnter = pl.xyz; } } else if (t < tf) { tf = t; nExit = pl.xyz; }
      }
      if (tf > tn) {
        for (int e = 0; e < 2; e++) {
          float t = e == 0 ? tn : tf;
          if (t <= 0.0 || t >= len) continue;
          vec3 hp = ro + rd * t;
          vec3 relPh = (hp - c) * (invLPh / max(pScale, 0.2));
          vec3 hpLoc = vec3(dot(relPh, t1Ph), dot(relPh, axis), dot(relPh, t2Ph));
          float cosk = abs(dot(rd, e == 0 ? nEnter : nExit));
          float dust = texture(u_incl, hpLoc * 2.4 + off + float(layer) * 0.37).r;
          float sheet = phRateW * layerWeight * clamp(effPhantom * (0.12 - 0.02 * float(layer)) * (0.3 + 1.7 * dust) / max(cosk, 0.12), 0.0, 1.0);
          float effIrid = u_iridescence * u_rainbow;
          vec3 phCol = effIrid > 0.01
            ? mix(phLightIn, spectralRainbow(cosk * 2.4 + float(layer) * 0.33 + hueOff) * (phLightIn + vec3(0.35)), clamp(effIrid * 0.6, 0.0, 1.0))
            : phLightIn;
          scat += T * sheet * phCol * 1.45;
          T *= 1.0 - 0.45 * sheet;
        }
      }
    }
  }
}

// Evaluates a crystal-locked 1D striation wave at fixed harmonic frequencies (so morphing u_striationScale
// blends harmonic weights instead of sweeping wave phase across the crystal face!).
float evalCrystalStriationWave(float cNorm, float uid, float fpNorm, float scaleBlend, float metalW, float aaCoarse, float aaFine, float aaBrush) {
  float fCoarse = 5.5;
  float fFine = 16.5;
  float fBrush = 31.0;
  float amp = (0.38 + 0.72 * texture(u_incl, vec3(cNorm * 1.8, uid * 0.37, 0.5)).b) * mix(1.0, 1.45, metalW);
  float wFine = mix(0.24, 0.46, scaleBlend);
  float wCoarse = 1.0 - wFine;
  float ridge = aaCoarse * wCoarse * sin(TAU * cNorm * fCoarse + 1.5 * sin(cNorm * 11.0))
              + aaFine * wFine * sin(TAU * cNorm * fFine);
  ridge += metalW * aaBrush * 0.48 * sin(TAU * cNorm * fBrush + 2.3 * cos(cNorm * 23.0));
  return amp * ridge;
}

// Bend a face's normal: growth ridges across prism faces (including orthogonal m-3 diploid cube-face striations on Pyrite),
// stepped hopper terraces, and smooth/Gaussian-dithered/Jack-Bush-druzy mineral frost relief.
vec3 faceNormal(int id, int k, vec3 n, vec3 p, float rough, float footprint, float dEdge) {
  vec4 ax = body(id, 1);
  float L = max(ax.w, 0.10);
  float r = max(body(id, 2).w, 0.02);
  vec3 rt = body(id, 3).xyz;
  vec3 cMid = rt + ax.xyz * (0.52 * L);
  vec3 relMid = p - cMid;
  vec3 p0 = body(id, PA).xyz;
  vec3 t1 = p0 - ax.xyz * dot(p0, ax.xyz);
  float t1Len = length(t1);
  t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(ax.xyz, vec3(0.577, 0.577, 0.577)));
  vec3 t2 = cross(ax.xyz, t1);
  vec3 pLoc = vec3(
    dot(relMid, t1) / max(r * 2.0, 0.04),
    dot(relMid, ax.xyz) / L,
    dot(relMid, t2) / max(r * 2.0, 0.04)
  );
  vec3 localN = normalize(vec3(dot(n, t1), dot(n, ax.xyz), dot(n, t2)) + vec3(1e-6));

  float uid = crystalUid(id);
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float glimmer = clamp(u_glimmer, 0.0, 2.0);
  float g1 = clamp(glimmer, 0.0, 1.0);
  float rawAmt = clamp(u_rawAmount, 0.0, 2.0);
  float rawSurfScale = mix(0.15, 1.0, clamp(rawAmt / 0.4, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, (rawAmt - 0.4) / 1.6));
  if (u_striations > 0.001) {
    float metalW = smoothstep(0.08, 0.55, u_metallic);
    float hopperAtten = 1.0 - smoothstep(0.15, 0.52, u_hopper);
    float prismFaceW = 1.0 - smoothstep(0.32, 0.72, abs(localN.y));
    float voidDamp = (k >= CP && k < CP + 5000 && k != CP + 2000) ? 0.0 : 1.0;
    float striFaceMask = mix(prismFaceW, 1.0, metalW) * hopperAtten * voidDamp;
    if (striFaceMask > 1e-4) {
      float fpNorm = footprint / L;
      float scaleBlend = clamp((u_striationScale - 40.0) / 140.0, 0.0, 1.0);
      float aaCoarse = clamp(1.0 - (fpNorm * 5.5) * 5.8, 0.0, 1.0);
      float aaFine = clamp(1.0 - (fpNorm * 16.5) * 5.8, 0.0, 1.0);
      float aaBrush = clamp(1.0 - (fpNorm * 31.0) * 4.5, 0.0, 1.0);

      // Smooth triplanar m-3 diploid cube-face blend (zero 90° axis-switch flick, zero u_hopper < 0.35 jump)
      float pyriteW = smoothstep(0.12, 0.60, u_metallic) * (1.0 - smoothstep(0.12, 0.48, u_hopper));
      vec3 wTri = pow(abs(localN), vec3(6.0));
      wTri /= max(wTri.x + wTri.y + wTri.z, 1e-5);
      vec3 axisWeights = mix(vec3(1.0, 0.0, 0.0), wTri, pyriteW);

      float waveY = evalCrystalStriationWave(pLoc.y, uid, fpNorm, scaleBlend, metalW, aaCoarse, aaFine, aaBrush);
      vec3 striPerturb = ax.xyz * (axisWeights.x * waveY);
      if (pyriteW > 1e-3) {
        float waveX = evalCrystalStriationWave(pLoc.x * 0.55, uid + 1.7, fpNorm, scaleBlend, metalW, aaCoarse, aaFine, aaBrush);
        float waveZ = evalCrystalStriationWave(pLoc.z * 0.55, uid + 3.3, fpNorm, scaleBlend, metalW, aaCoarse, aaFine, aaBrush);
        striPerturb += t1 * (axisWeights.z * waveX) + t2 * (axisWeights.y * waveZ);
      }
      // Project perturbation onto the facet tangent plane so normal length stays consistent
      striPerturb -= n * dot(n, striPerturb);
      float striStrength = u_striations * striFaceMask * rawSurfScale * (1.0 - 0.35 * dither) * mix(0.055, 0.125, g1);
      n = normalize(n + striPerturb * striStrength);
    }
  }
  if (rough > 0.0) {
    float frostAa = 1.0 / (1.0 + 48.0 * (footprint / L));
    // Crystal-locked frost & fine micro-crystalline dither coordinates (scaled in crystal-local frame using trilinear hardware u_incl lookups — zero floor() cellular jumps & zero hash() overhead)
    vec3 pFrost = vec3(pLoc.x * 0.32, pLoc.y, pLoc.z * 0.32);
    vec4 fTex = texture(u_incl, pFrost * 0.48 + uid * 0.19);
    vec3 fGradLoc = fTex.rgb - 0.5;
    vec3 fGrad = t1 * fGradLoc.x + ax.xyz * fGradLoc.y + t2 * fGradLoc.z;
    vec3 nSmooth = fGrad * (rough * 0.015 * frostAa);
    vec4 dTex = texture(u_incl, pFrost * 14.5 + uid * 0.31);
    vec3 nCell0 = dTex.rgb - 0.5;
    float patchMod = 0.55 + 0.80 * fTex.b;
    vec3 nDitherWorld = (t1 * nCell0.x + ax.xyz * nCell0.y + t2 * nCell0.z) * (rough * 0.058 * patchMod * frostAa);
    vec3 nJack = vec3(0.0);
    if (glimmer > 0.001) {
      vec3 hJb = texture(u_incl, pFrost * 18.5 + uid * 0.53 + 0.17).rgb - 0.5;
      nJack = (t1 * hJb.x + ax.xyz * hJb.y + t2 * hJb.z) * (rough * 0.038 * glimmer * frostAa);
    }
    n = normalize(n + mix(nSmooth, nDitherWorld, dither) * (1.0 - 0.38 * g1) + nJack);
  }
  return n;
}

// Evaluates a family of long, hair-thin (100:1+ aspect ratio), gently curved anti-aliased mineral scratches
// with per-scratch random angle and curvature so scratches never look like parallel ruled lines or alias into static.
float evalLongHairlineScratch(vec2 uv, vec3 lp, float angle, float stripFreq, float alongFreq, float densityThresh, float seed, float fwStrip, float baseN, out vec2 outGradUV) {
  float ca = cos(angle), sa = sin(angle);
  vec2 uvRot = vec2(uv.x * ca + uv.y * sa, -uv.x * sa + uv.y * ca);
  float warp = ((u_isMobile > 0.5 ? fract(baseN + seed * 0.13) : n3(lp * 3.4 + seed)) - 0.5) * 0.055;
  float uStrip = (uvRot.x + warp) * stripFreq;
  float vAlong = uvRot.y * alongFreq;
  float stripId = floor(uStrip);
  float segId = floor(vAlong);
  float uLocal = fract(uStrip) - 0.5;
  float vLocal = fract(vAlong) - 0.5;

  float h = hash(vec3(stripId, segId, seed));
  outGradUV = vec2(0.0);
  if (h < densityThresh) return 0.0;

  float uOff = (fract(h * 13.7) - 0.5) * 0.48;
  float skew = (fract(h * 29.3) - 0.5) * 0.62;
  float arc = (fract(h * 43.1) - 0.5) * 0.16 * (vLocal * vLocal - 0.08);
  float du = uLocal - (uOff + skew * vLocal + arc);
  float halfSpan = mix(0.18, 0.42, fract(h * 19.1));
  float vCenter = (fract(h * 41.7) - 0.5) * (0.44 - halfSpan * 0.5);
  float lenEnv = 1.0 - smoothstep(halfSpan * 0.25, halfSpan, abs(vLocal - vCenter));
  if (lenEnv <= 1e-4) return 0.0;
  float rawWidth = mix(0.020, 0.036, fract(h * 37.9));
  float effWidth = max(rawWidth, fwStrip * 1.35);
  if (abs(du) > effWidth * 3.2) return 0.0;
  float aaAtten = clamp(rawWidth / effWidth, 0.0, 1.0);
  float profile = exp(-(du * du) / (effWidth * effWidth)) * lenEnv * aaAtten;
  float pressure = (u_isMobile > 0.5)
    ? smoothstep(0.32, 0.74, fract(baseN * 1.7 + h * 7.3))
    : smoothstep(0.32, 0.74, n3(lp * 8.5 + seed + stripId * 1.3));
  profile *= (0.35 + 0.65 * pressure) * mix(0.50, 1.0, fract(h * 53.3));
  float dProfDu = (-2.0 * du / effWidth) * profile * aaAtten;
  outGradUV = vec2(ca, sa) * dProfDu;
  return profile;
}

// Realistic mineral surface patina: 4 orientations of sparse, varied-angle hair-thin mineral scratches
// and faint growth-hillock etch lines. Uses crystal-frame-locked normalized coordinates (lp) independent of
// facet normal tilt nFlat and plane index kFace so surface scratches and hillocks never flick during morphs!
void evalSurfacePatina(int id, int k, vec3 p, vec3 nBase, float footprint, float dEdge, out vec3 nPatina, out float patinaGlint, out float patinaRough) {
  nPatina = nBase;
  patinaGlint = 0.0;
  patinaRough = 0.0;
  float metalAtten = 1.0 - smoothstep(0.75, 0.96, clamp(u_metallic, 0.0, 1.0));
  float rawSurfScale = mix(0.15, 1.0, clamp(u_rawAmount / 0.4, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, (u_rawAmount - 0.4) / 1.6));
  float effPatina = clamp(u_patina * u_patinaMacro * u_surfaceMacro * rawSurfScale * metalAtten, 0.0, 2.0);
  if (effPatina <= 0.005) return;

  vec3 ax = body(id, 1).xyz;
  vec3 rt = body(id, 3).xyz;
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float uid = crystalUid(id);

  vec3 p0 = body(id, PA).xyz;
  vec3 t1 = p0 - ax * dot(p0, ax);
  float t1Len = length(t1);
  t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(ax, vec3(0.577, 0.577, 0.577)));
  vec3 t2 = cross(ax, t1);

  vec3 cMid = rt + ax * (0.52 * L);
  vec3 relMid = p - cMid;
  float scaleR = 1.0 / max(r * 2.2, 0.06);
  float scaleL = 1.0 / max(L * 0.72, 0.10);
  vec3 lp = vec3(dot(relMid, t1) * scaleR, dot(relMid, ax) * scaleL, dot(relMid, t2) * scaleR);

  // Singularity-free 3D-to-2D crystal-locked scratch projections (invariant to nFlat tilt and kFace index)
  float uFace = lp.x * 0.80 - lp.z * 0.60;
  float vFace = lp.y * 0.90 + (lp.x * 0.60 + lp.z * 0.80) * 0.28;
  vec2 uvFace = vec2(uFace, vFace);
  vec2 uvFaceAlt = vec2(lp.x * 0.60 + lp.z * 0.80, lp.y * 0.90 - uFace * 0.28);

  vec3 tU0 = normalize(t1 * 0.80 - t2 * 0.60 + ax * 0.25);
  vec3 tU = normalize(tU0 - nBase * dot(nBase, tU0) + vec3(1e-6));
  vec3 tV = normalize(cross(nBase, tU));

  float uvFootprint = max(footprint * scaleL, 0.0012);
  float aa = 1.0 / (1.0 + 45.0 * uvFootprint);
  float fSeed = uid * 17.3 + f_seed * 0.37;
  float edgeProx = 1.0 - smoothstep(0.0, r * 0.28, dEdge);
  // Single-octave 3D noise base sample on mobile; full multi-octave on desktop
  float baseN = n3(lp * 4.8 + fSeed);
  float patchEnv = smoothstep(0.42, 0.80, baseN + 0.26 * edgeProx);

  float a1 = 0.38 + 0.55 * (hash(vec3(uid, 1.0, 13.0)) - 0.5);
  float a3 = 1.15 + 0.45 * (hash(vec3(uid, 3.0, 47.0)) - 0.5);

  vec2 g1 = vec2(0.0), g2 = vec2(0.0), g3 = vec2(0.0), g4 = vec2(0.0);
  float scr1 = 0.0, scr2 = 0.0, scr4 = 0.0;
  float scr3 = evalLongHairlineScratch(uvFace + vec2(-0.19, 0.43), lp, a3, 7.5, 1.10, 0.86, fSeed + 23.9, uvFootprint * 7.5, baseN, g3);
  if (patchEnv > 0.001) {
    scr1 = evalLongHairlineScratch(uvFace, lp, a1, 9.5, 1.35, 0.82, fSeed + 3.1, uvFootprint * 9.5, baseN, g1) * patchEnv;
    g1 *= patchEnv;
    if (u_isMobile <= 0.5) {
      float a2 = -0.72 + 0.55 * (hash(vec3(uid, 2.0, 29.0)) - 0.5);
      float a4 = -1.35 + 0.45 * (hash(vec3(uid, 4.0, 71.0)) - 0.5);
      scr2 = evalLongHairlineScratch(uvFaceAlt + vec2(0.37, -0.21), lp, a2, 13.0, 1.65, 0.84, fSeed + 11.7, uvFootprint * 13.0, baseN, g2) * patchEnv;
      scr4 = evalLongHairlineScratch(uvFaceAlt + vec2(0.51, 0.17), lp, a4, 16.0, 1.85, 0.85, fSeed + 39.3, uvFootprint * 16.0, baseN, g4) * patchEnv;
      g2 *= patchEnv;
      g4 *= patchEnv;
    }
  }

  // Faint organic growth-hillock etch steps & horizontal striation surface ledges (anti-aliased by uvFootprint)
  float termFaceW = smoothstep(0.20, 0.65, abs(dot(nBase, ax)));
  float hillPatchN = (u_isMobile > 0.5) ? baseN : n3(lp * 3.8 + fSeed + 15.1);
  float hillPatch = smoothstep(0.54, 0.84, hillPatchN) * mix(0.48, 0.85, termFaceW) * aa;
  float hillock = 0.0;
  float dHill = 0.0;
  if (hillPatch > 0.001) {
    float hillField = (u_isMobile > 0.5)
      ? (vFace * 11.0 + 4.8 * baseN)
      : (vFace * 11.0 + 3.8 * n3(lp * 4.2 + fSeed) + 1.4 * n3(lp * 9.5 + fSeed * 1.7));
    float hillWave = sin(hillField);
    if (hillWave > 0.55) {
      hillock = pow(hillWave, 14.0) * hillPatch;
      dHill = cos(hillField) * hillock;
    }
  }

  float striaGlint = 0.0;
  float dStria = 0.0;
  float prismFaceW = 1.0 - smoothstep(0.35, 0.72, abs(dot(nBase, ax)));
  float striaGate = prismFaceW * smoothstep(0.08, 0.24, u_striations);
  if (striaGate > 0.001) {
    float sField = lp.y * 12.0 + 1.8 * sin(lp.y * 9.0 + uFace * 1.6 + fSeed);
    float sSin = sin(sField);
    if (sSin > 0.55) {
      float striaN = (u_isMobile > 0.5) ? baseN : n3(lp * 4.5 + fSeed + 7.3);
      float sWave = pow(sSin, 12.0) * smoothstep(0.38, 0.78, striaN) * aa;
      striaGlint = sWave * u_striations * 0.72 * striaGate;
      dStria = cos(sField) * striaGlint;
    }
  }

  vec2 gradTot = (g1 + g2 + g3 * 1.20 + g4 * 0.80) * 0.024 + vec2(0.0, (dHill + dStria) * 0.015);
  vec3 dN = tU * gradTot.x + tV * gradTot.y;

  nPatina = normalize(nBase + dN * min(effPatina, 1.6));
  float totalScratch = clamp(scr1 * 0.85 + scr2 * 0.80 + scr3 * 1.10 + scr4 * 0.70 + hillock * 0.42 + striaGlint * 0.55, 0.0, 1.6);
  patinaGlint = totalScratch * effPatina * aa;
  patinaRough = clamp(totalScratch * 0.020 * effPatina * aa, 0.0, 0.035);
}

float edgeDist(int id, int k, vec3 p, out vec3 adjN, out int outKLo, out int outKHi) {
  outKLo = -1;
  outKHi = -1;
  if (k >= CP + 10000) {
    int enc = k - (CP + 10000);
    int tri = enc / 4;
    int v2 = tri - (tri / 18) * 18;
    int rem = tri / 18;
    int v1 = rem - (rem / 18) * 18;
    int v0 = rem / 18;
    vec4 pl0 = body(id, PA + v0);
    vec4 pl1 = body(id, PA + v1);
    vec4 pl2 = body(id, PA + v2);
    adjN = normalize(pl0.xyz + pl1.xyz + pl2.xyz);
    float s0 = max(0.0, pl0.w - dot(pl0.xyz, p));
    float s1 = max(0.0, pl1.w - dot(pl1.xyz, p));
    float s2 = max(0.0, pl2.w - dot(pl2.xyz, p));
    outKLo = v0;
    outKHi = v1;
    return min(s0, min(s1, s2));
  }
  if (k >= CP + 5000) {
    int enc = k - (CP + 5000);
    int pair = enc / 2;
    int ka = pair / 18;
    int kb = pair - ka * 18;
    outKLo = ka;
    outKHi = kb;
    vec4 plLo = body(id, PA + ka);
    vec4 plHi = body(id, PA + kb);
    adjN = normalize(plLo.xyz + plHi.xyz);
    float cosAng = dot(plLo.xyz, plHi.xyz);
    float sinAng = sqrt(max(1e-3, 1.0 - cosAng * cosAng));
    float sLo = max(0.0, plLo.w - dot(plLo.xyz, p)) / sinAng;
    float sHi = max(0.0, plHi.w - dot(plHi.xyz, p)) / sinAng;
    return abs(sLo - sHi) * 0.5;
  }
  if (k >= CP && k < CP + 5000) {
    int m = clamp(k - CP - 1, 0, 1);
    int nextRow = occPartnerRow(id, m);
    vec4 rawEc = body(nextRow, m);
    vec4 ec = vec4(rawEc.xyz, rawEc.w * u_occlusion);
    adjN = normalize(ec.xyz - p + vec3(1e-5));
    return max(0.0, ec.w - length(p - ec.xyz));
  }
  float dmin = 1e9;
  int jBest = -1;
  vec3 nk = body(id, PA + k).xyz;
  adjN = nk;
  for (int j = 0; j < CP; j++) {
    if (j == k) continue;
    vec4 pl = body(id, PA + j);
    if (pl.w > 1e5) continue;
    float rawD = pl.w - dot(pl.xyz, p);
    if (rawD >= dmin) continue;
    float cosAng = dot(pl.xyz, nk);
    if (cosAng > 0.9995) continue;
    float sinAng = sqrt(max(1e-3, 1.0 - cosAng * cosAng));
    float d = rawD / sinAng;
    if (d < dmin) {
      dmin = d;
      adjN = pl.xyz;
      jBest = j;
    }
  }
  if (k < 16 && jBest >= 0 && jBest < 16) {
    outKLo = min(k, jBest);
    outKHi = max(k, jBest);
  }
  return max(dmin, 0.0);
}

float edgeDist(int id, int k, vec3 p, out vec3 adjN) {
  int kLo, kHi;
  return edgeDist(id, k, p, adjN, kLo, kHi);
}

// Softens & tints bright studio softbox glare on outer crystal faces (governed by u_faceGlare)
// so a lit front facet never blows out to solid opaque white and internal colourful reflections stay visible.
vec3 softFacetRefl(vec3 rawEnv, float hgt) {
  vec3 gemCol = exp(-zonedGemSigma(hgt) * 0.42);
  float lum = max(rawEnv.r, max(rawEnv.g, rawEnv.b));
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  vec3 compressed = rawEnv / (1.0 + (2.35 - 1.05 * glare) * lum);
  vec3 softCol = mix(compressed * (0.20 + 0.72 * gemCol), compressed * 0.85, 0.26 * glare);
  return mix(softCol, rawEnv / (1.0 + 0.65 * lum), 0.42 * clamp(u_glimmer, 0.0, 1.0));
}
vec3 softFacetRefl(vec3 rawEnv) {
  return softFacetRefl(rawEnv, 0.75);
}

vec3 refractedEnv(vec3 dir, float rough, float refAlpha, float hgt) {
  vec3 d = f_toWorld * dir;
  vec3 bg = backdrop(d);
  vec3 lt = lights(d, rough);
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  float ltLum = max(lt.r, max(lt.g, lt.b));
  vec3 gemCol = exp(-zonedGemSigma(hgt) * 0.48);
  vec3 softLt = (lt / (1.0 + (2.8 - 1.4 * glare) * ltLum)) * mix(gemCol * 0.76, gemCol * 0.52 + vec3(0.32), 0.20 * glare);
  vec3 curEnv = bg + mix(lt, softLt, refAlpha);
  float rearRimMask = 1.0 - smoothstep(-0.35, 0.05, d.z);
  vec3 jbRefrLt = mix(softLt * 0.26, lt * 0.22, rearRimMask) + glimmerStripLights(d, rough);
  vec3 jbEnv = bg + mix(lt, jbRefrLt, refAlpha);
  float jbBlend = clamp(glimmerCurve(u_glimmer), 0.0, 1.0);
  return mix(curEnv, jbEnv, jbBlend);
}
vec3 refractedEnv(vec3 dir, float rough, float refAlpha) {
  return refractedEnv(dir, rough, refAlpha, 0.75);
}

// Smooth crystal-frame facet orientation phase (replaces discrete float(k) * 0.19 so thin-film & frost never jump on k changes)
float crystalFacetPhase(int id, vec3 n) {
  return dot(n, vec3(0.61, 0.43, 0.79)) + float(id) * 0.37;
}

float evalCrystalFrostRoll(int id, vec3 n, float uid) {
  float v = sin(n.x * 3.7 + uid * 2.3 + f_seed * 0.31) * cos(n.y * 4.1 - n.z * 3.3 + uid * 1.7)
          + 0.5 * sin(n.z * 5.3 + n.x * 2.9 - uid * 3.1);
  return clamp(0.5 + 0.32 * v, 0.05, 0.95);
}

vec3 facetIridescence(int id, int k, vec3 p, vec3 rd, vec3 n, float dEdge, vec3 baseRefl) {
  float s = u_iridescence * u_rainbow;
  if (s <= 0.0) return baseRefl;
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float cosV = clamp(abs(dot(rd, n)), 0.0, 1.0);
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float hgt = dot(p - body(id, 3).xyz, body(id, 1).xyz) / L;
  float genShift = body(id, 2).y;
  float filmT = cosV * 2.4 + hgt * 1.35 + (dEdge / r) * 0.95 + crystalFacetPhase(id, n) + genShift;
  vec3 irid1 = spectralRainbow(filmT);
  vec3 irid2 = spectralRainbow(filmT * 1.62 + 0.29);
  vec3 specCol = normalize(mix(irid1, irid2, mix(0.32, 0.35, dither)) + vec3(1e-4)) * mix(1.22, 1.12, dither);
  float fresRim = pow(1.0 - cosV, 2.0);
  vec3 tinted = mix(baseRefl, baseRefl * specCol * mix(1.32, 1.12, dither), clamp(s * 0.82, 0.0, mix(0.85, 0.68, dither)));
  vec3 sheen = specCol * f_ambient * (0.28 + 0.72 * fresRim) * min(s, mix(2.2, 1.2, dither)) * mix(0.22, 0.14, dither);
  return tinted + sheen;
}

vec3 iridescentFilm(int id, int k, vec3 p, vec3 rd, vec3 n, float dEdge) {
  float s = u_iridescence * u_rainbow;
  if (s <= 0.0) return vec3(0.0);
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float cosV = clamp(abs(dot(rd, n)), 0.0, 1.0);
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float hgt = dot(p - body(id, 3).xyz, body(id, 1).xyz) / L;
  float fPhase = crystalFacetPhase(id, n);
  float filmT = cosV * 2.4 + hgt * 1.35 + (dEdge / r) * 0.95 + fPhase + body(id, 2).y;
  vec3 film = spectralRainbow(filmT * 1.18 + 0.11);
  float rim = pow(1.0 - cosV, 1.5);
  float coat = 0.6 + 0.4 * evalCrystalFrostRoll(id, n, body(id, 2).z * 3.0);
  return film * f_ambient * (0.25 + 0.75 * rim) * coat * min(s, 1.4) * 0.18 * (1.0 - 0.38 * dither);
}

// Surface facet catch-lights & bevel glints (governed by u_bloom and u_faceGlare)
vec3 crystalBloomGlint(int id, int k, vec3 p, vec3 rd, vec3 nSmooth, float dEdge, float rough) {
  if (u_bloom <= 0.0) return vec3(0.0);
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  vec3 rSmooth = normalize(f_toWorld * reflect(rd, nSmooth));
  float apexMask = smoothstep(0.78, 0.98, dot(p - body(id, 3).xyz, body(id, 1).xyz) / max(body(id, 1).w, 0.1));
  float polish = 1.0 - 0.55 * rough;

  // Tight crystallographic specular catch-lights without artificial edge-strip brightening
  float expFacet = mix(240.0, 90.0, clamp(u_bloom * 0.45, 0.0, 0.9));
  vec3 flashSum = vec3(0.0);
  float hgtGlint = crystalHeight01(id, p);
  vec3 gemTint = mix(exp(-zonedGemSigma(hgtGlint) * 0.38), vec3(1.0), 0.32 * glare);

  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 lDir = vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]);
    float cSmooth = dot(rSmooth, lDir);
    if (cSmooth <= 0.84) continue;
    vec3 lCol = vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
    float fGlint = pow(cSmooth, expFacet) * (1.0 + 0.8 * apexMask);
    flashSum += lCol * gemTint * (fGlint * 0.28 * glare);
  }

  // 6 pinpoint studio catch-lights around the orbit ring
  float fPhase = crystalFacetPhase(id, nSmooth);
  for (int j = 0; j < 6; j++) {
    vec3 spotDir = (j == 0) ? vec3( 0.373644, 0.352274,  0.858281)
                 : (j == 1) ? vec3( 0.770520, 0.628793,  0.104525)
                 : (j == 2) ? vec3( 0.525634, 0.352274, -0.774518)
                 : (j == 3) ? vec3(-0.309551, 0.628793, -0.713298)
                 : (j == 4) ? vec3(-0.899278, 0.352274, -0.083763)
                            : vec3(-0.460969, 0.628793,  0.608773);
    float sSmooth = dot(rSmooth, spotDir);
    if (sSmooth <= 0.86) continue;
    float facetHit = pow(sSmooth, expFacet * 1.2) * (1.0 + 0.8 * apexMask);
    vec3 prismTint = mix(gemTint, spectralRainbow(fPhase + float(j) * 0.17 + body(id, 2).y), clamp((0.25 + u_iridescence * 0.3) * u_rainbow, 0.0, 0.85));
    flashSum += prismTint * (facetHit * 1.15 * glare);
  }

  float overdrive = u_bloom * (0.30 + 0.28 * u_bloom) * (0.18 + 0.62 * glare);
  vec3 rawGlint = flashSum * polish * overdrive;
  return rawGlint / (1.0 + (1.8 - 0.85 * glare) * max(rawGlint.r, max(rawGlint.g, rawGlint.b)));
}

// Physical Thin-Film Bi2O3 Oxide Interference Palette (post-015895, post-015896):
// Maps optical path difference opd across the exact 5-zone Bismuth oxide interference spectrum:
// warm metallic gold -> hot magenta/rose -> royal violet -> electric peacock cyan -> emerald/chartreuse green.
vec3 evalBismuthOxidePalette(float opd) {
  float t = fract(opd);
  vec3 cGold    = vec3(1.00, 0.80, 0.16);
  vec3 cMagenta = vec3(0.96, 0.18, 0.64);
  vec3 cViolet  = vec3(0.42, 0.22, 0.96);
  vec3 cCyan    = vec3(0.06, 0.82, 0.96);
  vec3 cEmerald = vec3(0.18, 0.92, 0.36);

  float s = t * 5.0;
  vec3 Stops = (s < 1.0) ? mix(cGold, cMagenta, smoothstep(0.0, 1.0, s))
             : (s < 2.0) ? mix(cMagenta, cViolet, smoothstep(0.0, 1.0, s - 1.0))
             : (s < 3.0) ? mix(cViolet, cCyan, smoothstep(0.0, 1.0, s - 2.0))
             : (s < 4.0) ? mix(cCyan, cEmerald, smoothstep(0.0, 1.0, s - 3.0))
                         : mix(cEmerald, cGold, smoothstep(0.0, 1.0, s - 4.0));

  vec3 cosFilm = 0.5 + 0.5 * cos(6.28318 * (opd * vec3(1.0, 0.86, 0.72) + vec3(0.0, 0.20, 0.45)));
  return mix(Stops, cosFilm, 0.28) * 1.08;
}

float outerLipCatchLight(vec3 wN, vec3 wRefl, vec3 qGa) {
  float leftLip = (1.0 - smoothstep(-0.22, -0.08, qGa.x)) * smoothstep(-0.55, 0.65, wN.z - 0.35 * wN.x);
  float botBelly = (1.0 - smoothstep(-0.22, 0.02, qGa.y)) * smoothstep(-0.65, 0.55, -wN.y + 0.45 * wN.z);
  float rightCollar = smoothstep(0.08, 0.22, qGa.x) * smoothstep(-0.45, 0.65, wN.x + 0.35 * wN.z);
  return clamp(leftLip * 0.85 + botBelly * 0.95 + rightCollar * 0.75, 0.0, 1.4);
}

// Opaque Metallic Conductor & Thin-Film Oxide Surface Shader (Pyrite Cubes FeS2 + Bismuth Hopper Bi/Bi2O3 + Solid/Liquid Gallium Ga):
// Evaluates conductor Fresnel reflectance, studio softbox diffuser dome, GGX metallic specular lobes,
// and inter-crystal + intra-hopper multi-bounce metallic reflections.
vec3 shadeMetallicCrystal(int id, int k, vec3 pEntry, vec3 rd, vec3 nRefl, float reflRough, float dEdge, float footprint, float meltFactor) {
  float cosTheta = clamp(-dot(rd, nRefl), 0.02, 1.0);
  float hgt = crystalHeight01(id, pEntry);
  float uid = crystalUid(id);

  float hopperStepIdx = 0.0, hopperEdgeDist = 1.0;
  if (u_hopper > 0.01) {
    evalBismuthHopper(pEntry, id, u_hopper, hopperStepIdx, hopperEdgeDist);
  }

  // 1. Conductor normal-incidence reflectance F0:
  //    - Pyrite Cubes (FeS2, low iridescence): mirror-polished brass-gold F0 = vec3(0.92, 0.80, 0.48)
  //    - Bismuth Hopper (Bi2O3 oxide film, high iridescence): thin-film interference varying by step depth, height & angle
  vec3 pyriteF0 = vec3(0.92, 0.78, 0.42);
  float sin2Theta = max(0.0, 1.0 - cosTheta * cosTheta);
  float cosFilm = sqrt(max(0.22, 1.0 - sin2Theta / (2.42 * 2.42)));
  // Smooth thin-film Bi2O3 oxide thickness sweeping across the ziggurat cluster & Greek-key terrace steps:
  // Outer rims & raised G-bars gleam in Gold -> Hot Magenta, stepping through Violet -> Peacock Cyan -> Emerald Green
  vec3 axMet = body(id, 1).xyz;
  float LMet = max(body(id, 1).w, 0.12);
  vec3 cMidMet = body(id, 3).xyz + axMet * (0.52 * LMet);
  vec3 p0Met = body(id, PA).xyz;
  vec3 t1Met = p0Met - axMet * dot(p0Met, axMet);
  float t1MetLen = length(t1Met);
  t1Met = t1MetLen > 1e-4 ? (t1Met / t1MetLen) : normalize(cross(axMet, vec3(0.577, 0.577, 0.577)));
  vec3 relMet = (pEntry - cMidMet) / LMet;
  float spatialGrad = (1.0 - hgt) * 0.46 + (dot(relMet, t1Met) * 0.65 - dot(relMet, axMet) * 0.35) * 0.36 + float(id) * 0.11;
  float opd = (0.94 + hopperStepIdx * 0.22 + spatialGrad) * cosFilm;
  vec3 biOxideCol = evalBismuthOxidePalette(opd);
  vec3 biGrazingCol = evalBismuthOxidePalette(opd + 0.18 * (1.0 - cosTheta));

  float iridW = clamp((u_iridescence - 0.08) / 0.78, 0.0, 1.0);
  vec3 F0 = mix(pyriteF0, biOxideCol, iridW);
  vec3 F90 = mix(vec3(1.0, 0.94, 0.76), mix(biGrazingCol, biOxideCol * 1.15, 0.55), iridW);
  vec3 F_cond = mix(F0, F90, pow(1.0 - cosTheta, 3.8));

  // 2. High-contrast studio photo-tent reflection (dark studio shadows + bright overhead key softbox + rim strips)
  vec3 reflDir = reflect(rd, nRefl);
  vec3 wN = normalize(f_toWorld * nRefl);
  vec3 wRefl = normalize(f_toWorld * reflDir);
  float metalRough = clamp(reflRough * 0.42 + 0.018, 0.012, 0.26);

  vec3 baseEnv = env(reflDir, metalRough);
  float topSoftbox = pow(max(0.0, smoothstep(-0.05, 0.82, wRefl.y)), 1.6) * (0.55 + 0.45 * smoothstep(-0.3, 0.7, wRefl.x));
  float sideRimTent = pow(max(0.0, smoothstep(0.10, 0.85, abs(wRefl.x))), 1.8) * smoothstep(-0.25, 0.55, wRefl.z);
  float hzDiff = (wRefl.y + 0.12) * 3.8;
  float darkHorizonBand = 1.0 - 0.65 * exp(-hzDiff * hzDiff);
  float baseAmbFloor = mix(0.04, 0.14, iridW);
  vec3 studioDome = f_ambient * (baseAmbFloor + mix(0.62, 0.44, iridW) * topSoftbox + 0.26 * sideRimTent) * darkHorizonBand;
  vec3 reflLight = baseEnv * mix(0.48, 0.34, iridW) + studioDome;

  // 3. Multi-bounce metallic inter-reflections (analytical intra-hopper 90° corner reflection + neighboring crystal bounce)
  vec3 roBounce = pEntry + nRefl * 0.003;
  bool hitNeighbor = false;
  vec3 neighborCol = vec3(0.0);
  float seamAO = 1.0;

  if (u_hopper > 0.01 && hopperStepIdx > 0.05) {
    // 90° Greek-key step walls reflect adjacent terrace colors into each other
    float cornerBounceW = smoothstep(0.05, 0.35, u_hopper) * smoothstep(0.15, 3.5, hopperStepIdx) * (0.42 + 0.45 * (1.0 - cosTheta));
    vec3 bounceOxide = evalBismuthOxidePalette(opd + 0.22);
    reflLight += bounceOxide * f_ambient * (0.26 * cornerBounceW);
  }

  // 1 secondary metallic reflection bounce max
  int nc = clamp(int(f_crystals + 0.5), 0, MAX_CRYSTALS);
  float bestBT = 1e9;
  int bestBId = 0, bestBK = 0;
  for (int bi = 0; bi < nc; bi++) {
    if (bi == id || !sphereHit(roBounce, reflDir, body(bi, 0), bestBT)) continue;
    float bTn, bTf; int bKn, bKf;
    if (convexHit(bi, roBounce, reflDir, bTn, bTf, bKn, bKf)) {
      float tC = max(bTn, 0.0);
      if (bTf > tC + 1e-4 && tC < bestBT) {
        bestBT = tC; bestBId = bi; bestBK = bKn;
        hitNeighbor = true;
      }
    }
  }
  if (hitNeighbor) {
    vec3 pB = roBounce + reflDir * bestBT;
    vec3 nB = body(bestBId, PA + bestBK).xyz;
    float sIdxB = (u_hopper > 0.01) ? (hopperStepIdx + 0.8) : 0.0;
    float opdB = (0.08 + sIdxB * 0.17 + (1.0 - crystalHeight01(bestBId, pB)) * 0.52 + float(bestBId) * 0.14) * 0.82;
    vec3 colB = mix(pyriteF0, evalBismuthOxidePalette(opdB), iridW);
    vec3 reflDir2 = reflect(reflDir, nB);
    vec3 wRefl2 = normalize(f_toWorld * reflDir2);
    seamAO = smoothstep(0.002, 0.065, bestBT);
    neighborCol = colB * (f_ambient * 0.22 + lights(wRefl2, 0.05) * 0.48 + env(reflDir2, 0.05) * 0.36);
    reflLight = mix(reflLight * seamAO, neighborCol * (0.35 + 0.85 * seamAO), 0.68);
  }

  // 4. Broad & sharp GGX conductor specular lobes from the 4 studio area lights
  vec3 specBroad = vec3(0.0);
  vec3 specSharp = vec3(0.0);
  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 lDir = normalize(vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]));
    vec3 lCol = vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
    float cR = max(0.0, dot(wRefl, lDir));
    float cN = max(0.0, dot(wN, lDir));
    specBroad += lCol * (0.11 * pow(cN, 3.0) + 0.46 * pow(cR, 14.0));
    specSharp += lCol * pow(cR, 84.0);
  }

  // 5. Crisp metallic catch-light along 90° hopper step rims & cube facet edges (smoothly blended across u_hopper)
  float hopRimW = smoothstep(0.02, 0.28, u_hopper);
  float edgeRimGlow = (1.0 - smoothstep(0.0, 0.024, dEdge / max(body(id, 2).w, 0.04))) * 0.28;
  float hopRimGlow = (1.0 - smoothstep(0.002, 0.032, hopperEdgeDist)) * (0.42 + 0.48 * u_hopper);
  float stepRimGlow = mix(edgeRimGlow, hopRimGlow, hopRimW);

  vec3 highlightTint = mix(mix(F_cond, vec3(1.0, 0.96, 0.84), 0.32), F_cond * 1.15, iridW * 0.78);
  vec3 metalCol = F_cond * (reflLight + specBroad * mix(0.16, 0.12, iridW) * seamAO)
                + highlightTint * (specSharp * mix(0.18, 0.11, iridW) + stepRimGlow * f_ambient * mix(0.32, 0.24, iridW))
                + crystalBloomGlint(id, k, pEntry, rd, nRefl, dEdge, metalRough) * F_cond * mix(0.55, 0.35, iridW);

  // 6. Solid-to-Liquid Gallium Dual Surface Shader (post-016133, reference photo 372c3f66):
  //    - Upper solid orthorhombic crystal facets: frosty cast-gallium crystalline grain, crisp silver-white
  //      bevel ridges, warm champagne-amber top bevel reflections, cool silver-platinum side facets, and pewter shadows.
  //    - Lower smooth-min meniscus collar & molten gallium puddle: 100% mirror-smooth liquid mercury/gallium chrome
  //      reflecting a luminous warm cream-gold basin, a high-contrast dark studio horizon band along the curved rim,
  //      and a razor-sharp silver-white specular highlight along the bulbous outer capillary lip.
  vec3 stoneLin = exp(-f_sigma * 0.42);
  float sMx = max(stoneLin.r, max(stoneLin.g, stoneLin.b));
  float sMn = min(stoneLin.r, min(stoneLin.g, stoneLin.b));
  float stoneSat = (sMx - sMn) / max(sMx, 1e-4);
  float nonPyriteMetal = (1.0 - smoothstep(0.46, 0.66, u_striations)) * (1.0 - iridW);
  float gaModeW = clamp(max(max(smoothstep(0.002, 0.18, u_melt), nonPyriteMetal), (1.0 - smoothstep(0.16, 0.34, stoneSat)) * (1.0 - iridW)), 0.0, 1.0);
  if (gaModeW > 0.001) {
    float tGaRest = clamp(u_melt / 0.72, 0.0, 1.0);
    float restPuddleW = smoothstep(0.58, 0.98, tGaRest);
    float blobMidGa = sin(tGaRest * PI);
    float liqW = smoothstep(0.02, 0.36, meltFactor);
    vec3 qGa = toGalliumFrame(pEntry);
    int kCrystal = (k >= CP + 3000 && k < CP + 4000) ? (k - (CP + 3000)) : clamp(k, 0, 17);

    // Subtle anisotropic cast-gallium micro-grain & directional brushed striations on solid crystal facets (~0.008..0.014 normal perturbation)
    vec3 gTan1 = normalize(cross(nRefl, vec3(0.36, 0.86, -0.35)));
    vec3 gTan2 = cross(nRefl, gTan1);
    float uCoord = dot(qGa, gTan1);
    float vCoord = dot(qGa, gTan2);
    float brushFine = sin(uCoord * 240.0 + n3(qGa * 72.0 + float(kCrystal) * 3.1) * 3.2);
    float brushCross = sin((uCoord * 0.30 + vCoord * 0.95) * 180.0 + n3(qGa * 105.0 + float(id) * 4.7) * 2.6);
    float grainMicro = n3(qGa * 195.0 + float(kCrystal) * 5.3) - 0.5;
    float solidGrainMask = (1.0 - liqW) * (0.45 + 0.55 * clamp(u_frost * 4.5, 0.0, 1.2));

    vec3 nGa = normalize(
      nRefl + (gTan1 * (brushFine * 0.008 + grainMicro * 0.011) + gTan2 * (grainMicro * 0.012 + brushCross * 0.006)) * solidGrainMask
    );
    vec3 wNGa = normalize(f_toWorld * nGa);
    vec3 wNSmooth = normalize(f_toWorld * nRefl);
    vec3 wReflGa = normalize(f_toWorld * reflect(rd, nGa));
    float cosGa = clamp(-dot(rd, nGa), 0.02, 1.0);
    float cosSmooth = clamp(-dot(rd, nRefl), 0.02, 1.0);

    // High-reflectance mirror-chrome metallic gallium conductor palette (F0 ~ vec3(0.88, 0.90, 0.93)):
    vec3 gaF0 = vec3(0.90, 0.93, 0.97);
    vec3 warmGoldKey = vec3(0.88, 0.92, 0.98);
    vec3 coolSilverRim = vec3(0.76, 0.83, 0.92);
    vec3 brightSilverSpec = vec3(1.02, 1.06, 1.12);
    vec3 pewterShadow = vec3(0.025, 0.030, 0.042);

    // --- (A) Upper Solid Gallium Crystal Facet Shading (High-Contrast Mirror-Chrome Metallic Conductor) ---
    float topGoldAlign = smoothstep(0.10, 0.54, dot(wNGa, normalize(vec3(-0.28, 0.78, 0.56))));
    float sideSilverAlign = smoothstep(-0.12, 0.50, dot(wNGa, normalize(vec3(0.62, 0.08, 0.78))));
    float innerPewterAlign = smoothstep(-0.04, 0.44, dot(wNGa, normalize(vec3(-0.74, -0.14, 0.65)))) * (1.0 - 0.85 * topGoldAlign);
    float puddleUpBounce = (1.0 - smoothstep(-0.44, -0.06, wNGa.y)) * (1.0 - smoothstep(-0.06, 0.09, qGa.y)) * clamp(u_melt * 1.35, 0.0, 1.0);

    vec3 gaCondTint = mix(coolSilverRim, brightSilverSpec, sideSilverAlign);
    gaCondTint = mix(gaCondTint, pewterShadow * 1.35, innerPewterAlign * (1.0 - 0.75 * puddleUpBounce));
    gaCondTint = mix(gaCondTint, warmGoldKey, topGoldAlign);
    gaCondTint = mix(gaCondTint, mix(brightSilverSpec, warmGoldKey * 1.12, 0.55), puddleUpBounce * 0.80);
    vec3 gaFresnel = mix(gaCondTint, vec3(0.98, 1.00, 1.04), pow(max(0.0, 1.0 - cosGa), 3.6) * 0.62);

    vec3 rawGaEnv = env(reflect(rd, nGa), 0.014);
    vec3 gaEnvRefl = vec3(0.92, 0.96, 1.03) * dot(rawGaEnv, vec3(0.299, 0.587, 0.114));
    vec3 gaReflLight = vec3(0.92, 0.96, 1.03) * dot(reflLight, vec3(0.299, 0.587, 0.114));
    vec3 gaAmb = vec3(0.92, 0.96, 1.03) * max(dot(f_ambient, vec3(0.299, 0.587, 0.114)), 0.85);
    vec3 gaSpecBroad = vec3(0.94, 0.98, 1.04) * dot(specBroad, vec3(0.299, 0.587, 0.114));
    vec3 gaSpecSharp = vec3(0.96, 1.00, 1.06) * dot(specSharp, vec3(0.299, 0.587, 0.114));
    float keySoftboxGrad = pow(max(0.0, smoothstep(-0.18, 0.82, wReflGa.y + 0.32 * wReflGa.z - 0.18 * wReflGa.x)), 1.65);
    float rimStripGrad = pow(max(0.0, smoothstep(-0.12, 0.80, abs(wReflGa.x) * 0.65 + wReflGa.z * 0.72)), 1.65);
    float chDiff = (wReflGa.y + 0.06) * 3.8;
    float chBand = 1.0 - (0.84 - 0.56 * blobMidGa) * exp(-chDiff * chDiff);
    float darkTentGrad = (0.08 + 0.92 * smoothstep(-0.42, 0.36, wReflGa.y + 0.25 * wReflGa.z)) * chBand;
    float chromeSoftboxSpec = pow(max(0.0, smoothstep(0.08, 0.86, wReflGa.y * 0.65 + wReflGa.z * 0.60 - wReflGa.x * 0.25)), 2.2)
                            + 0.72 * pow(max(0.0, smoothstep(0.14, 0.88, abs(wReflGa.x) * 0.68 + wReflGa.z * 0.64)), 2.4);

    float subtleBrushMod = 0.96 + 0.04 * grainMicro + 0.03 * brushFine + 0.02 * brushCross;
    vec3 realMetalRefl = gaReflLight * mix(0.48, 0.94, sideSilverAlign * 0.65 + topGoldAlign * 0.75)
                       + gaEnvRefl * (0.42 + 0.54 * rimStripGrad + 0.56 * keySoftboxGrad);

    vec3 solidCol = gaFresnel * realMetalRefl * darkTentGrad * subtleBrushMod
                  + gaCondTint * gaAmb * (0.035 + 0.16 * keySoftboxGrad * topGoldAlign + 0.14 * rimStripGrad * sideSilverAlign) * chBand * subtleBrushMod
                  + vec3(0.96, 1.00, 1.08) * chromeSoftboxSpec * 0.42 * chBand
                  + min(gaSpecBroad, vec3(1.6)) * mix(brightSilverSpec, warmGoldKey * 1.15, topGoldAlign) * 0.24 * seamAO
                  + min(gaSpecSharp, vec3(1.4)) * vec3(1.16, 1.20, 1.28) * 0.24;
    solidCol = min(solidCol, vec3(1.36, 1.40, 1.46));

    // --- (B) Lower Molten Gallium Puddle & Meniscus Collar Shading (100% Mirror Chrome) ---
    vec3 wPuddleUp = normalize(vec3(0.055, 0.825, 0.562));
    float basinDot = dot(wNSmooth, wPuddleUp);

    // Outer grazing silhouette attenuation (cosSmooth < 0.24 reflects the dark studio floor/horizon, never a blown-out white halo!)
    float silhouetteDarkEdge = smoothstep(0.04, 0.28, cosSmooth);

    // 1. Broad tilted puddle basin reflects bright silvery-platinum studio diffuser (staying luminous mercury as it contracts/expands)
    float basinMask = smoothstep(mix(0.16, 0.46, restPuddleW), mix(0.58, 0.82, restPuddleW), basinDot) * silhouetteDarkEdge;
    vec3 puddleWarmPool = mix(vec3(0.56, 0.62, 0.72), vec3(0.86, 0.91, 0.98), smoothstep(mix(0.32, 0.62, restPuddleW), mix(0.75, 0.95, restPuddleW), basinDot));

    // 2. Concave meniscus crease right where solid crystal plunges into the molten pool
    float meniscusCrease = max(0.0, sin(clamp((meltFactor - 0.03) / 0.55, 0.0, 1.0) * PI));
    float meniscusDark = 1.0 - (0.32 + 0.46 * restPuddleW) * pow(meniscusCrease, 1.35);
    float meniscusFilletSpec = pow(meniscusCrease, 2.4) * smoothstep(0.08, 0.72, wReflGa.y + 0.35 * wReflGa.z) * silhouetteDarkEdge;

    // 3. High-contrast black-chrome studio horizon band along the resting capillary rim (relaxing into bright mercury during morph)
    float rimDiff = (basinDot - 0.52) / 0.20;
    float rimDarkRing = exp(-rimDiff * rimDiff) * (0.22 + 0.70 * restPuddleW);
    float outerGrazingDark = (1.0 - silhouetteDarkEdge) * (0.55 + 0.33 * restPuddleW);
    float darkBand = clamp(max(rimDarkRing, outerGrazingDark), 0.0, 0.95);

    // 4. Inward-set silver-white liquid-mercury specular highlight tube along the upper-outer curvature of the rim (cosSmooth ~ 0.28..0.68)
    float tubeDiff = (cosSmooth - 0.44) / 0.16;
    float inwardRimTube = exp(-tubeDiff * tubeDiff) * smoothstep(0.10, 0.30, cosSmooth) * (1.0 - 0.75 * basinMask);
    float lowerLipCatch = outerLipCatchLight(wNSmooth, wReflGa, qGa);
    float rimSpecHighlight = clamp(inwardRimTube * (0.68 + 0.72 * lowerLipCatch), 0.0, 1.35);

    vec3 rawMirrorEnv = env(reflect(rd, nRefl), 0.004) * (0.14 + 0.22 * silhouetteDarkEdge);
    vec3 mirrorEnv = vec3(0.92, 0.96, 1.03) * dot(rawMirrorEnv, vec3(0.299, 0.587, 0.114));
    vec3 liquidCol = puddleWarmPool * basinMask * meniscusDark * 1.02;
    liquidCol = mix(liquidCol, vec3(0.012, 0.014, 0.020), darkBand);
    liquidCol += mirrorEnv * (1.0 - 0.65 * darkBand) * vec3(0.88, 0.92, 0.98);
    liquidCol += vec3(1.16, 1.22, 1.30) * rimSpecHighlight * 0.96;
    liquidCol += vec3(0.96, 1.02, 1.12) * meniscusFilletSpec * 0.58;
    liquidCol += min(gaSpecSharp, vec3(1.4)) * vec3(1.06, 1.10, 1.16) * 0.24 * silhouetteDarkEdge;

    vec3 gaCol = mix(solidCol, liquidCol, liqW);
    metalCol = mix(metalCol, gaCol, gaModeW);
  }
  return metalCol;
}

vec3 shadeMetallicCrystal(int id, int k, vec3 pEntry, vec3 rd, vec3 nRefl, float reflRough, float dEdge, float footprint) {
  return shadeMetallicCrystal(id, k, pEntry, rd, nRefl, reflRough, dEdge, footprint, 0.0);
}

// Secondary continuation ray for GPU mirrored backgrounds (single-pass, zero nested hop loop)
vec3 shadeBehindRay(vec3 roIn, vec3 rdIn, int skipIn, float travIn, float jitter, out float firstHitMask) {
  firstHitMask = 0.0;
  int bId, bK; float bT, bTf;
  vec3 bVoidN; float bVoidBorder, bVoidSeam;
  if (!traceScene(roIn, rdIn, skipIn, bId, bT, bTf, bK, bVoidN, bVoidBorder, bVoidSeam)) {
    return refractedEnv(rdIn, 0.0, 0.0);
  }
  vec3 bP = roIn + rdIn * bT;
  float travB = travIn + bT;
  float bRawA = clamp(body(bId, 2).x, 0.0, 1.0);
  float bAlpha = bRawA * bRawA * (3.0 - 2.0 * bRawA);
  firstHitMask = bAlpha;
  vec3 bEnvCol = refractedEnv(rdIn, 0.0, 0.0);
  if (bAlpha <= 1e-5) return bEnvCol;

  float glare = clamp(u_faceGlare, 0.0, 1.0);
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float ditherFrost = smoothstep(0.0, 0.85, dither);
  float effFrostShare = mix(u_frostShare, max(u_frostShare, 0.58), ditherFrost);
  float effFrost = mix(u_frost, max(u_frost, 0.38), ditherFrost);
  float bUid = crystalUid(bId);
  bool bIsGalliumMelt = (bK >= CP + 3000 && bK < CP + 4000);
  bool bIsCathedral = (bK >= CP + 4000 && bK < CP + 4100);
  int bKBase = bIsGalliumMelt ? (bK - (CP + 3000)) : (bIsCathedral ? (bK - (CP + 4000)) : bK);
  float bMeltFactor = bIsGalliumMelt ? clamp(bVoidBorder, 0.0, 1.0) : 0.0;
  vec3 bRawN = (bK < CP) ? body(bId, PA + bK).xyz : bVoidN;
  float bFrostRoll = evalCrystalFrostRoll(bId, bRawN, bUid);
  float bRough = effFrost * smoothstep(bFrostRoll - 0.15, bFrostRoll + 0.15, effFrostShare) * (1.0 - smoothstep(0.0, 0.35, bMeltFactor));
  vec3 bN = faceNormal(bId, bKBase < CP ? bKBase : 18, bRawN, bP, bRough, travB * f_pixAngle, 1.0);
  if (bIsGalliumMelt) {
    bN = normalize(mix(bN, bRawN, smoothstep(0.0, 0.42, bMeltFactor)));
  }
  float bMetalW = smoothstep(0.0, 0.95, clamp(max(u_metallic, smoothstep(0.02, 0.35, bMeltFactor)), 0.0, 1.0));
  vec3 bMetalCol = (bMetalW > 0.005)
    ? shadeMetallicCrystal(bId, bKBase, bP, rdIn, bN, bRough, bIsGalliumMelt ? bVoidSeam : 1.0, travB * f_pixAngle, bMeltFactor)
    : vec3(0.0);
  if (bMetalW >= 0.999 || (u_melt > 0.45 && bMetalW >= 0.92)) {
    return mix(bEnvCol, bMetalCol, bAlpha);
  }
  float bF = fresnel(-dot(rdIn, bN), 1.0 / u_ior);
  vec3 bEntryRefl = facetIridescence(bId, bKBase, bP, rdIn, bN, 1.0, softFacetRefl(env(reflect(rdIn, bN), bRough)));
  float bIridBoost = clamp(bF + (u_iridescence * u_rainbow) * 0.06, 0.0, 0.42) * (0.15 + 0.85 * glare);
  vec3 colB = bIridBoost * bEntryRefl;
  vec3 thrB = vec3(1.0 - bF * (0.3 + 0.7 * glare));
  vec3 bRefrIn = refract(rdIn, bN, 1.0 / u_ior);
  vec3 rdB = dot(bRefrIn, bRefrIn) > 1e-6 ? normalize(bRefrIn) : rdIn;
  int bKf;
  vec3 bExitRawN;
  float bLen = exitCrystal(bId, bP, rdB, bKf, bExitRawN);
  vec3 bTrans;
  vec3 bInScat = insideVolume(bId, bP, rdB, bLen, jitter, clamp(int(u_inclusionSteps + 0.5), 2, 4), bTrans);
  colB += thrB * bInScat;
  thrB *= bTrans;
  travB += bLen;
  bP += rdB * bLen;
  float bExitRoll = evalCrystalFrostRoll(bId, bExitRawN, bUid);
  float bFr = effFrost * smoothstep(bExitRoll - 0.15, bExitRoll + 0.15, effFrostShare);
  vec3 bNf = faceNormal(bId, bKf < CP ? bKf : 18, bExitRawN, bP, bFr, travB * f_pixAngle, 1.0);
  float bFo = fresnel(dot(rdB, bNf), u_ior);
  vec3 bExitRefl = facetIridescence(bId, bKf, bP, rdB, -bNf, 1.0, softFacetRefl(env(reflect(rdB, -bNf), bFr)));
  colB += thrB * (bFo * bExitRefl * 0.42 * (0.15 + 0.85 * glare));
  thrB *= 1.0 - bFo * (0.3 + 0.7 * glare);
  vec3 bOut = refract(rdB, -bNf, u_ior);
  rdB = dot(bOut, bOut) > 1e-6 ? normalize(bOut) : reflect(rdB, -bNf);
  vec3 endEnv = refractedEnv(rdB, bFr, 1.0);
  float deflAng = acos(clamp(dot(rdB, rdIn), -1.0, 1.0));
  float glowRatio = deflAng / max(radians(u_glowSize * 1.15), 1e-4);
  float glowG = exp(-glowRatio * glowRatio);
  vec3 axisBg = mix(lin(u_studioEdge) * 0.14, f_glow * 1.65 + f_ambient * 0.22, glowG);
  endEnv = max(endEnv, axisBg);
  colB += thrB * endEnv;
  if (bMetalW > 0.005) {
    colB = mix(colB, bMetalCol, bMetalW);
  }
  return mix(bEnvCol, colB, bAlpha);
}

// Analytical GPU Mirrored Background:
// 0: Plain (None)
// 1: Above & Below (Classic)            6: Above & Below (Classic — Liquid)
// 2: Above & Below (Natural)            7: Above & Below (Natural — Liquid)
// 3: Cube (4-Wall)                      8: Cube (Liquid)
// 4: Hall of Mirrors (Stretched)        9: Hall of Mirrors (Stretched — Liquid)
// 5: Hexagon (Stretched)               10: Hexagon (Stretched — Liquid)
vec3 evalMirrorBackground(vec2 q, vec3 rd, float rough, float refAlpha, float jitter) {
  vec3 baseBg = refractedEnv(rd, rough, refAlpha);
  if (f_mirrorMode < 0.5 || u_mirrorStrength <= 0.001) return baseBg;

  int mMode = int(f_mirrorMode + 0.5);
  bool isLiquidMode = (mMode >= 6);
  int baseMode = isLiquidMode ? (mMode - 5) : mMode;
  float mScale = max(0.25, u_mirrorScale);
  vec3 camR = f_cam[0], camU = f_cam[1], camF = f_cam[2];
  vec3 cDir = vec3(dot(rd, camR), dot(rd, camU), dot(rd, camF));
  vec2 rawQ = (refAlpha > 0.01) ? (cDir.xy / max(0.18, cDir.z) * f_focal) : q;
  vec2 vq = rawQ / mScale;
  float camDist = length(f_camPos);
  float worldSpan = camDist / max(f_focal, 0.5);

  vec2 qRefl = rawQ;
  vec3 roRefl = f_camPos;
  float mirrorMask = 1.0;
  float archGlow = 0.0;
  float planeDist = abs(vq.y);
  float zP = 1.0;
  vec2 uvSurf = vec2(0.0);
  float fresnelM = 0.85;

  if (baseMode == 1) {
    // 1 & 6. Above & Below (Classic): original receding ceiling & floor mirror planes
    float ay = abs(vq.y);
    float sy = vq.y >= 0.0 ? 1.0 : -1.0;
    planeDist = ay;
    zP = 0.24 / max(ay, 0.014);
    uvSurf = vec2(vq.x * zP, zP);
    float grid = (smoothstep(0.92, 0.995, cos(uvSurf.x * 5.5)) + smoothstep(0.92, 0.995, cos(uvSurf.y * 4.0))) * exp(-zP * 0.22);
    float horizonBand = exp(-ay * 26.0);
    archGlow = grid * 0.32 + horizonBand * 0.55;
    mirrorMask = smoothstep(0.012, 0.075, ay);
    qRefl = vec2(vq.x * (0.84 + 0.32 * ay), -sy * (ay - 0.25) * 1.38);
    roRefl = f_camPos + camU * (sy * 0.52);
  } else if (baseMode == 2) {
    // 2 & 7. Above & Below (Natural): optical floor & ceiling planes above/below the crystal with natural horizontal-plane foreshortening
    float ay = abs(vq.y);
    float sy = rawQ.y >= 0.0 ? 1.0 : -1.0;
    planeDist = ay;
    zP = 0.24 / max(ay, 0.012);
    uvSurf = vec2(vq.x * zP, zP);
    float cosTheta = clamp(ay * 2.1, 0.02, 1.0);
    fresnelM = mix(0.62, 1.0, pow(1.0 - cosTheta, 3.0));
    archGlow = exp(-ay * 32.0) * 0.10;
    mirrorMask = smoothstep(0.018, 0.085, ay);
    float uVert = (abs(rawQ.y) - 0.20 * mScale) * 1.82;
    qRefl = vec2(rawQ.x * 1.08, -sy * uVert);
    roRefl = f_camPos + camU * (sy * 0.28 * mScale * worldSpan) - camF * (0.35 * mScale * worldSpan);
  } else if (baseMode == 3) {
    // 3 & 8. Cube (4-Wall): 4-walled square mirror tunnel (top, bottom, left, right walls) with true 3D planar reflections
    float ax = abs(vq.x), ay = abs(vq.y);
    float dCube = max(ax, ay);
    planeDist = dCube;
    zP = 0.24 / max(dCube, 0.012);
    float cosTheta = clamp(dCube * 2.1, 0.02, 1.0);
    fresnelM = mix(0.62, 1.0, pow(1.0 - cosTheta, 3.0));
    if (ay >= ax) {
      float sy = rawQ.y >= 0.0 ? 1.0 : -1.0;
      uvSurf = vec2((vq.x / max(ay, 0.012)) * 1.5, zP);
      float uVert = (abs(rawQ.y) - 0.20 * mScale) * 1.82;
      qRefl = vec2(rawQ.x * 1.08, -sy * uVert);
      roRefl = f_camPos + camU * (sy * 0.28 * mScale * worldSpan) - camF * (0.35 * mScale * worldSpan);
    } else {
      float sx = rawQ.x >= 0.0 ? 1.0 : -1.0;
      uvSurf = vec2((vq.y / max(ax, 0.012)) * 1.5, zP);
      float uHorz = (abs(rawQ.x) - 0.12 * mScale) * 1.38;
      qRefl = vec2(-sx * uHorz, rawQ.y * 1.12);
      roRefl = f_camPos + camR * (sx * 0.26 * mScale * worldSpan) - camF * (0.35 * mScale * worldSpan);
    }
    float cornerDist = abs(ax - ay) / max(dCube, 0.02);
    float cornerSeam = exp(-cornerDist * 30.0) * smoothstep(0.03, 0.12, dCube);
    archGlow = cornerSeam * 0.22 + exp(-dCube * 30.0) * 0.10;
    mirrorMask = smoothstep(0.025, 0.095, dCube);
  } else if (baseMode == 4) {
    // 4 & 9. Hall of Mirrors (Stretched): parallel floor & ceiling mirrors with anamorphically stretched reflections receding into depth
    float ay = abs(vq.y);
    float sy = rawQ.y >= 0.0 ? 1.0 : -1.0;
    planeDist = ay;
    zP = 0.24 / max(ay, 0.012);
    uvSurf = vec2(vq.x * zP, zP);
    float cosTheta = clamp(ay * 1.9, 0.02, 1.0);
    fresnelM = mix(0.65, 1.0, pow(1.0 - cosTheta, 3.0));
    float centerY = 0.26 * mScale;
    float stretchedY = centerY + (abs(rawQ.y) - centerY) * 0.28 + 0.024 * mScale * sin(zP * 4.4);
    qRefl = vec2(rawQ.x * (1.02 + 0.05 * cos(zP * 2.4)), -sy * stretchedY);
    roRefl = f_camPos + camU * (sy * 0.34 * mScale * worldSpan) - camF * (0.35 * mScale * worldSpan);
    float echoPulse = smoothstep(0.75, 0.99, cos(zP * 4.4)) * exp(-zP * 0.28);
    archGlow = echoPulse * 0.16 + exp(-ay * 30.0) * 0.12;
    mirrorMask = smoothstep(0.016, 0.08, ay);
  } else {
    // 5 & 10. Hexagon (Stretched): 6-walled hexagonal mirror tunnel with radially stretched crystal reflections
    float ang = atan(vq.y, vq.x);
    float sec = PI / 3.0;
    float secIdx = floor((ang + sec * 0.5) / sec);
    float aCenter = secIdx * sec;
    float aLocal = ang - aCenter;
    vec2 wDir = vec2(cos(aCenter), sin(aCenter));
    vec2 wTan = vec2(-sin(aCenter), cos(aCenter));
    vec3 camW = camR * wDir.x + camU * wDir.y;
    float dHex = max(dot(vq, wDir), 0.012);
    float tHex = dot(vq, wTan);
    planeDist = dHex;
    zP = 0.24 / dHex;
    float uWall = (tHex / dHex) / 0.57735027;
    uvSurf = vec2(uWall * 1.5, zP);
    float cosTheta = clamp(dHex * 1.9, 0.02, 1.0);
    fresnelM = mix(0.65, 1.0, pow(1.0 - cosTheta, 3.0));
    float rawDHex = max(dot(rawQ, wDir), 0.012);
    float rawTHex = dot(rawQ, wTan);
    float centerD = 0.26 * mScale;
    float stretchedD = centerD + (rawDHex - centerD) * 0.28 + 0.022 * mScale * sin(zP * 4.2);
    qRefl = -wDir * stretchedD + wTan * (rawTHex * 1.02);
    roRefl = f_camPos + camW * (0.34 * mScale * worldSpan) - camF * (0.35 * mScale * worldSpan);
    float hexSeam = exp(-abs(abs(aLocal) - sec * 0.5) * 42.0) * smoothstep(0.03, 0.12, dHex);
    archGlow = hexSeam * 0.22 + exp(-dHex * 30.0) * 0.10;
    mirrorMask = smoothstep(0.025, 0.095, dHex);
  }

  // Subtle liquid surface wave normal perturbation (active on Liquid modes 6..10, or when Liquid slider is raised above default on 1..5)
  float liqAmt = isLiquidMode ? clamp(u_mirrorLiquid, 0.0, 1.0) : max(0.0, u_mirrorLiquid - 0.22) * 0.85;
  float waveHighlight = 0.0;
  if (liqAmt > 0.001) {
    float wDepthDamp = exp(-max(0.0, zP - 0.45) * 0.22);
    float wt = u_realTime;
    float p1 = uvSurf.x * 3.4 + uvSurf.y * 2.2 - wt * 1.25;
    float p2 = uvSurf.x * 2.5 - uvSurf.y * 3.1 - wt * 0.92 + 1.57;
    vec2 radialVec = vec2(uvSurf.x, uvSurf.y - 1.15);
    float rPool = max(length(radialVec), 0.15);
    float p3 = rPool * 5.2 - wt * 1.45;
    vec2 gradW = vec2(
      cos(p1) * 3.4 + cos(p2) * 1.62 + cos(p3) * (radialVec.x / rPool) * 2.8,
      cos(p1) * 2.2 - cos(p2) * 2.02 + cos(p3) * (radialVec.y / rPool) * 2.8
    ) * 0.18;
    float amp = liqAmt * 0.036 * (baseMode == 1 ? 1.0 : mScale) * wDepthDamp;
    qRefl += gradW * amp;
    uvSurf += gradW * (liqAmt * 0.14 * wDepthDamp);
    waveHighlight = max(0.0, sin(p1) * 0.55 + sin(p3) * 0.45) * liqAmt * 0.20 * wDepthDamp;
  }

  // Mirror depth opacity fade (u_mirrorDepthFade: 0 = extends deep into horizon, 1 = fades out rapidly with depth)
  float fadeParam = clamp(u_mirrorDepthFade, 0.0, 1.0);
  float zDepth = max(0.0, zP - 0.38);
  float depthAtten = exp(-zDepth * (0.04 + 1.45 * fadeParam * fadeParam))
                   * mix(1.0, smoothstep(0.015, 0.08 + 0.24 * fadeParam, planeDist), fadeParam);

  // Anti-aliased 3D perspective checkerboard option (u_mirrorChecker, applies to all variations 1..10)
  float checkerStrength = clamp(u_mirrorChecker, 0.0, 1.0);
  float checkerLight = 0.0;
  float checkerDark = 0.0;
  if (checkerStrength > 0.001) {
    vec2 tc = uvSurf * 2.0;
    float fw = 0.055 + 0.055 * max(0.0, zP - 0.4);
    vec2 sc = sin(tc * PI) / max(fw * PI, 0.16);
    float chk = 0.5 + 0.5 * clamp(sc.x, -1.0, 1.0) * clamp(sc.y, -1.0, 1.0);
    checkerLight = chk * checkerStrength * depthAtten;
    checkerDark = (1.0 - chk) * checkerStrength * depthAtten;
  }

  vec3 rdRefl = normalize(f_cam * vec3(qRefl, f_focal));
  float hitMask = 0.0;
  vec3 reflScene = shadeBehindRay(roRefl, rdRefl, 0, 0.0, jitter, hitMask);

  vec3 compGlow = f_compGlow * (2.6 + 4.2 * exp(-dot(vq, vq) * 3.0));
  vec3 edgeLin = lin(u_studioEdge);
  vec3 compEdge = max(vec3(0.014, 0.012, 0.020), vec3(max(edgeLin.r, max(edgeLin.g, edgeLin.b)) + min(edgeLin.r, min(edgeLin.g, edgeLin.b))) - edgeLin);
  vec3 mirrorBaseBg = mix(baseBg, compEdge * 0.55 + compGlow, u_mirrorComplement * 0.88);
  vec3 compTintVec = mix(vec3(1.0), normalize(f_compGlow + vec3(0.16)) * 1.42, u_mirrorComplement * 0.65);
  float compHit = (baseMode == 1) ? 1.0 : hitMask;
  vec3 tintedRefl = reflScene * compTintVec + compGlow * (0.28 * u_mirrorComplement * compHit);
  vec3 archCol = mix(f_glow * 2.2 + f_ambient * 0.24, compGlow * 1.55 + f_ambient * 0.24, u_mirrorComplement);

  vec3 surfaceBase = mirrorBaseBg * (1.0 - 0.42 * checkerDark) + archCol * (0.38 * checkerLight + (archGlow + waveHighlight) * depthAtten);
  if (baseMode == 1) {
    vec3 mirrorCombined = surfaceBase * 0.42 + tintedRefl * (1.18 * depthAtten);
    float effMask = clamp(u_mirrorStrength * mirrorMask * mix(depthAtten, 1.0, 0.35 * checkerStrength), 0.0, 1.0);
    return mix(baseBg, mirrorCombined, effMask);
  }
  vec3 crystalReflCol = surfaceBase * 0.14 + tintedRefl * (1.18 * fresnelM);
  float hitBlend = clamp(hitMask * depthAtten * clamp(u_mirrorStrength * 1.25, 0.0, 1.0), 0.0, 1.0);
  float surfBlend = clamp(u_mirrorStrength * mirrorMask * mix(depthAtten, 1.0, 0.35 * checkerStrength), 0.0, 1.0);
  vec3 bgWithSurface = mix(baseBg, surfaceBase, surfBlend);
  return mix(bgWithSurface, crystalReflCol, hitBlend * mirrorMask);
}

void main() {
  vec2 fc = gl_FragCoord.xy;
  vec2 q = (fc - 0.5 * u_res) / min(u_res.x, u_res.y);
  vec3 rd = normalize(f_cam * vec3(q, f_focal));
  vec3 initialRd = rd;
  vec3 ro = f_camPos;
  float jitter = hash(vec3(fc, 17.0));
  float ior = u_ior;
  float glare = clamp(u_faceGlare, 0.0, 1.0);

  vec3 col = vec3(0.0), thr = vec3(1.0);
  float travelled = 0.0;
  int skipMask = 0;
  bool lastRefracted = false;
  vec3 lastIn = vec3(0.0), lastN = vec3(0.0);
  vec3 lastPathDisp = vec3(0.0);
  float lastRough = 0.0;
  float lastFireAlpha = 0.0;
  float lastHgt = 0.75;
  float fDepth = max(1.0, u_depth);
  int depth = int(ceil(fDepth)), bounces = int(ceil(u_bounces));
  if (u_isMobile > 0.5) {
    bounces = min(bounces, 3);
  }
  float hopCost = 0.0;
  float glimmer = clamp(u_glimmer, 0.0, 2.0);
  float g1 = clamp(glimmer, 0.0, 1.0);
  float effG1 = clamp(glimmerCurve(glimmer), 0.0, 1.0);

  int maxHops = clamp(depth + 1, 1, 5);
  bool primaryEvaluated = false;

  for (int hop = 0; hop < maxHops; hop++) {
    if (hopCost >= fDepth || dot(thr, thr) < 1e-6) break;
    int id, k; float t, tfHit;
    vec3 voidWallN = vec3(0.0, 1.0, 0.0);
    float voidBorder = 1e9, voidSeam = 1e9;
    if (!traceScene(ro, rd, skipMask, id, t, tfHit, k, voidWallN, voidBorder, voidSeam)) break;

    float rawAmt = clamp(u_rawAmount, 0.0, 2.0);
    float rawSurfScale = mix(0.15, 1.0, clamp(rawAmt / 0.4, 0.0, 1.0)) * mix(1.0, 1.65, max(0.0, (rawAmt - 0.4) / 1.6));
    float effEdgeMacro = max(0.0, u_patinaMacro * u_edgeMacro * rawSurfScale) * (1.0 - smoothstep(0.20, 0.55, u_hopper)) * (1.0 - smoothstep(0.0, 0.25, u_cathedral)) * (1.0 - smoothstep(0.002, 0.03, clamp(u_melt / 0.72, 0.0, 1.0)));
    float edgeSilAlpha = 1.0;
    vec3 chipFacetN = vec3(0.0, 1.0, 0.0);
    if (!primaryEvaluated && (u_edgeChips * effEdgeMacro > 0.005 || u_edgeIrregularity * effEdgeMacro > 0.005 || u_edgeWear * effEdgeMacro > 0.005) && k < CP) {
      if (!applyTinyEdgeChip(id, k, ro, rd, t, tfHit, k, edgeSilAlpha, chipFacetN)) {
        skipMask |= (1 << id);
        ro += rd * max(t, 0.0);
        travelled += max(t, 0.0);
        continue;
      }
    }

    vec3 pEntry = ro + rd * t;
    vec3 p = pEntry;
    travelled += t;
    skipMask |= (1 << id);

    float rawAlpha = clamp(body(id, 2).x, 0.0, 1.0) * edgeSilAlpha;
    primaryEvaluated = true;

    float budgetWeight = clamp(fDepth - hopCost, 0.0, 1.0);
    float alpha = rawAlpha * rawAlpha * (3.0 - 2.0 * rawAlpha) * budgetWeight;
    if (alpha <= 2e-3) {
      ro = pEntry;
      continue;
    }

    float effIor = ior;
    float uid = crystalUid(id);
    hopCost += alpha;

    float footprint = travelled * f_pixAngle;
    float dither = clamp(u_ditherBlend, 0.0, 1.0);
    float ditherFrost = smoothstep(0.0, 0.85, dither);
    float hgtEntry = crystalHeight01(id, pEntry);
    if (hop == 0) {
      lastHgt = hgtEntry;
    } else if (u_colourZoning > 0.5) {
      hgtEntry = min(hgtEntry, lastHgt + 0.12);
    }
    float baseFrost = u_colourZoning * clamp(rawSurfScale, 0.0, 1.5) * (1.0 - smoothstep(0.14, 0.48, hgtEntry));
    float rawFrostScale = mix(0.22, 1.0, clamp(rawAmt / 0.4, 0.0, 1.0)) * mix(1.0, 1.35, max(0.0, (rawAmt - 0.4) / 1.6));
    float effFrostShare = clamp((mix(u_frostShare, max(u_frostShare, 0.54), ditherFrost) + 0.14 * baseFrost) * rawFrostScale, 0.0, 0.88);
    float effFrost = clamp((mix(u_frost, max(u_frost, 0.32), ditherFrost) + 0.08 * baseFrost) * rawFrostScale, 0.0, 0.46);
    bool isGalliumMelt = (k >= CP + 3000 && k < CP + 4000);
    bool isCathedralTerrace = (k >= CP + 4000 && k < CP + 4100);
    int kBaseFace = isGalliumMelt ? (k - (CP + 3000)) : (isCathedralTerrace ? (k - (CP + 4000)) : k);
    int kCrystal = kBaseFace;
    bool isTinyEdgeChip = (k >= CP + 5000 && k < CP + 10000);
    bool isCornerVertexChip = (k >= CP + 10000);
    bool isHopperTerrace = (k == CP + 2000);
    float meltFactor = isGalliumMelt ? clamp(voidBorder, 0.0, 1.0) : 0.0;
    bool isOcclusionVoid = (k >= CP && k < CP + 5000 && !isHopperTerrace && !isGalliumMelt && !isCathedralTerrace);
    bool isEdgeIncursion = isTinyEdgeChip || isCornerVertexChip || isOcclusionVoid;
    bool isCornerNip = (k >= 18 && k < CP);
    vec3 rawFaceN = (isOcclusionVoid || isHopperTerrace || isGalliumMelt || isCathedralTerrace)
      ? voidWallN
      : ((isTinyEdgeChip || isCornerVertexChip) ? chipFacetN : body(id, PA + k).xyz);
    float frostRoll = evalCrystalFrostRoll(id, rawFaceN, uid);
    float frostW = smoothstep(frostRoll - 0.15, frostRoll + 0.15, effFrostShare);
    float rough = effFrost * frostW * mix(1.0, 0.25, smoothstep(0.12, 0.58, u_hopper));
    if (isOcclusionVoid) {
      // Keep 3D void cavity walls smooth & clear (zero noisy texture, pure 3D geometric delineation)
      rough *= 0.20;
    } else if (isGalliumMelt) {
      rough *= (1.0 - smoothstep(0.0, 0.35, meltFactor));
    } else if (isCathedralTerrace) {
      rough *= mix(1.0, 0.55, smoothstep(0.0, 0.35, u_cathedral));
    }

    vec3 adjN = rawFaceN;
    int ridgeKLo = -1, ridgeKHi = -1;
    float tMorphM = clamp(u_melt / 0.72, 0.0, 1.0);
    float gaHoldMetal = smoothstep(0.42, 0.92, tMorphM) * 0.96;
    float rawMetalW = clamp(max(max(u_metallic, gaHoldMetal), smoothstep(0.02, 0.35, meltFactor)), 0.0, 1.0);
    float metalW = smoothstep(0.0, 0.95, rawMetalW);
    bool isOpaqueMetal = (metalW >= 0.995 || (isGalliumMelt && kBaseFace == 17));
    bool needDEdge = !isGalliumMelt && !(isOpaqueMetal && u_hopper > 0.65) && (((hop == 0) && (u_patina > 0.01 || u_occlusion > 0.01 || u_hopper > 0.0 || u_cathedral > 0.0 || u_iridescence > 0.0 || u_bloom > 0.0)) || (g1 > 0.001 && u_edgeWear > 0.0));
    float dEdge = isGalliumMelt ? voidSeam : (needDEdge ? edgeDist(id, kBaseFace, p, adjN, ridgeKLo, ridgeKHi) : 1.0);

    vec3 axisMain = body(id, 1).xyz, rootMain = body(id, 3).xyz;
    float LMain = max(body(id, 1).w, 0.12);
    vec3 cMidMain = rootMain + axisMain * (0.52 * LMain);
    vec3 p0Main = body(id, PA).xyz;
    vec3 t1Main = p0Main - axisMain * dot(p0Main, axisMain);
    float t1MainLen = length(t1Main);
    t1Main = t1MainLen > 1e-4 ? (t1Main / t1MainLen) : normalize(cross(axisMain, vec3(0.577, 0.577, 0.577)));
    vec3 t2Main = cross(axisMain, t1Main);
    vec3 relMidEntry = (p - cMidMain) / LMain;
    vec3 lpEntry = vec3(dot(relMidEntry, t1Main), dot(relMidEntry, axisMain), dot(relMidEntry, t2Main));

    float jbChip = 0.0;
    if (!isOpaqueMetal && g1 > 0.001 && u_edgeWear > 0.0 && kBaseFace < CP) {
      float jbEdge = 1.0 - smoothstep(0.0, u_edgeWidth * body(id, 2).w, dEdge);
      float jbNz = texture(u_incl, lpEntry * 4.5 + 0.71).b * 0.75 + texture(u_incl, lpEntry * 15.0).b * 0.35;
      jbChip = jbEdge * smoothstep(1.0 - 0.45 * u_edgeWear, 1.05 - 0.45 * u_edgeWear, jbNz) * g1 * (1.0 - metalW);
      rough = max(rough, jbChip * 0.6);
    }
    if (isCathedralTerrace) {
      dEdge = min(dEdge, voidBorder);
    }

    // Pure 3D CSG refractive crystal facet shading (zero artificial white edge paint or cylindrical edge normal rounding)
    if (isTinyEdgeChip || isCornerVertexChip || isCornerNip) {
      rough = max(rough, 0.025 * clamp(u_edgeWear * effEdgeMacro, 0.0, 1.0));
    }
    vec3 nBaseForRefr = (isCathedralTerrace && !isOpaqueMetal && kBaseFace < CP)
      ? normalize(mix(body(id, PA + kBaseFace).xyz, rawFaceN, 0.45))
      : rawFaceN;
    vec3 n = faceNormal(id, isEdgeIncursion ? 18 : kBaseFace, nBaseForRefr, p, rough, footprint, dEdge);
    if (isGalliumMelt) {
      n = normalize(mix(n, rawFaceN, smoothstep(0.0, 0.40, meltFactor)));
    }
    if (!isOpaqueMetal && hop == 0 && (isTinyEdgeChip || isCornerVertexChip || isCornerNip)) {
      float nDamp = (1.0 - metalW) / (1.0 + 40.0 * (footprint / LMain));
      float conchRing = sin(length(lpEntry) * 18.0 + dot(lpEntry, rawFaceN) * 28.0) * nDamp;
      vec3 ringTan = normalize(cross(rawFaceN, axisMain + vec3(0.01, 0.0, 0.02)));
      n = normalize(n + ringTan * (0.022 * conchRing));
    }

    vec3 nSurfFull = isCathedralTerrace ? normalize(rawFaceN + (n - nBaseForRefr)) : n;
    vec3 nRefl = nSurfFull;
    float patinaGlint = 0.0, patinaRough = 0.0;
    float patinaFade = 1.0 - smoothstep(0.002, 0.03, tMorphM);
    if (!isOpaqueMetal && hop == 0 && !isOcclusionVoid && !isGalliumMelt && patinaFade > 0.001) {
      evalSurfacePatina(id, kBaseFace, p, nSurfFull, footprint, dEdge, nRefl, patinaGlint, patinaRough);
      nRefl = normalize(mix(nSurfFull, nRefl, patinaFade));
      patinaGlint *= patinaFade;
      patinaRough *= patinaFade;
      // Keep internal refraction 100% pure & clear by only applying a tiny fraction (6%) of surface scratch normal to n
      n = normalize(mix(n, n + (nRefl - nSurfFull), 0.06 * (1.0 - 0.85 * dither)));
    }
    float reflRough = max(rough, patinaRough * (1.0 - 0.5 * dither));

    vec3 hopCol = vec3(0.0);
    vec3 hopThr = vec3(1.0);
    vec3 metalSurfCol = (metalW > 0.005)
      ? shadeMetallicCrystal(id, kBaseFace, pEntry, rd, nRefl, reflRough, dEdge, footprint, meltFactor)
      : vec3(0.0);
    if (isOpaqueMetal) {
      col += thr * alpha * metalSurfCol;
      thr *= (1.0 - alpha);
      if (alpha >= 0.999) {
        lastRefracted = false;
        break;
      }
      ro = pEntry;
      continue;
    } else if (hop == 0 && metalW > 0.005) {
      // Smoothly blend opaque conductor reflection at primary surface without compounding inside internal bounces
      col += thr * (alpha * metalW) * metalSurfCol;
      thr *= (1.0 - alpha * metalW);
    }
    bool isChippedFacet = isTinyEdgeChip || isCornerVertexChip || isCornerNip;
    float effGlare = (isOcclusionVoid ? min(1.0, glare * 1.15 + 0.12) : (isChippedFacet ? (glare * 0.65) : glare)) * (1.0 - 0.30 * dither);
    float cosEntry = mix(max(0.32, -dot(rd, nRefl)), clamp(-dot(rd, nRefl), 0.0, 1.0), g1);
    float F = fresnel(cosEntry, 1.0 / effIor);
    vec3 rawEntryRefl = softFacetRefl(env(reflect(rd, nRefl), reflRough), hgtEntry);
    if (isChippedFacet) {
      rawEntryRefl *= mix(vec3(1.0), exp(-zonedGemSigma(hgtEntry) * 0.32), 0.68 * (1.0 - 0.55 * g1));
    }
    vec3 entryRefl = facetIridescence(id, kBaseFace, p, rd, nRefl, dEdge, rawEntryRefl);
    vec3 entryFlash = (hop == 0) ? crystalBloomGlint(id, kBaseFace, p, rd, nRefl, dEdge, reflRough) * 0.55 * (1.0 - 0.25 * dither) : vec3(0.0);
    float iridReflBoost = clamp(F + (u_iridescence * u_rainbow) * 0.04 + patinaGlint * 0.08 * (1.0 - 0.6 * dither), 0.0, mix(0.46, 0.38, dither)) * (0.18 + 0.76 * effGlare);
    float entryReflW = mix(iridReflBoost, F * 0.90, g1);
    vec3 patinaLight = (0.42 * f_ambient + 0.48 * entryRefl + vec3(0.05)) * vec3(0.96, 0.98, 1.0);
    float surfFrostSheen = (hop == 0) ? (rough * (0.04 + 0.05 * baseFrost) * (0.35 + 0.65 * pow(1.0 - cosEntry, 1.8))) : 0.0;
    vec3 frostSheenCol = (0.38 * f_ambient + 0.42 * rawEntryRefl + vec3(0.03)) * mix(vec3(0.88, 0.92, 0.98), exp(-zonedGemSigma(hgtEntry) * 0.28), 0.55);
    hopCol += (entryReflW * entryRefl + jbChip * 0.08 * f_ambient + patinaGlint * 0.12 * (1.0 - 0.55 * dither) * patinaLight + surfFrostSheen * frostSheenCol + entryFlash);
    hopCol += iridescentFilm(id, kBaseFace, p, rd, nRefl, dEdge);
    if (isCathedralTerrace) {
      float rCath = max(body(id, 2).w, 0.02);
      float gableTilt = smoothstep(0.18, 0.48, dot(rawFaceN, body(id, 1).xyz));
      float cathEdgeRim = exp(-voidBorder / max(rCath * 0.018, footprint * 1.6));
      vec3 gemColCath = exp(-zonedGemSigma(hgtEntry) * 0.34);
      vec3 goldViolet = mix(gemColCath, vec3(1.0, 0.84, 0.55), 0.28 * gableTilt);
      hopCol += (0.18 * gableTilt * entryRefl + 0.14 * cathEdgeRim * (0.55 * entryRefl + 0.45 * f_ambient)) * goldViolet * u_cathedral;
    }
    hopThr *= mix(1.0 - F * (0.3 + 0.7 * effGlare), (1.0 - F) * (1.0 - 0.6 * jbChip), g1);

    // Clear 3D physical delineation of voided spaces: crisp 3D border rim where the void cuts the
    // crystal face + distinct 3D interior cavity wall shading & interior facet seams (zero noisy texture)
    if (!isCathedralTerrace && !isGalliumMelt && u_occlusion > 0.002 && voidBorder < 1e4) {
      float rVoid = max(body(id, 2).w, 0.02);
      float borderW = max(rVoid * 0.032, footprint * 2.4);
      float borderRim = exp(-voidBorder / borderW);
      float seamW = max(rVoid * 0.022, footprint * 1.8);
      float seamRim = isOcclusionVoid ? exp(-voidSeam / seamW) : 0.0;
      float edge3D = clamp(borderRim * 1.20 + seamRim * 0.85, 0.0, 1.35);
      vec3 gemTint = mix(exp(-f_sigma * 0.22), vec3(0.96, 0.98, 1.0), 0.74);
      vec3 keyDir = normalize(vec3(f_lights[0], f_lights[1], f_lights[2]));
      float diffWall = clamp(dot(f_toWorld * nRefl, keyDir) * 0.5 + 0.5, 0.0, 1.0);
      vec3 rimCol = gemTint * (0.55 * f_ambient + 0.55 * entryRefl + vec3(0.32));
      hopCol += edge3D * (isOcclusionVoid ? 0.52 : 0.34) * rimCol;
      if (isOcclusionVoid) {
        float cavRecess = 1.0 - exp(-voidBorder / max(rVoid * 0.12, 1e-3));
        float grazeWall = pow(clamp(1.0 - abs(dot(rd, nRefl)), 0.0, 1.0), 2.0);
        vec3 wallLight = gemTint * (0.38 + 0.62 * diffWall) * (0.45 * f_ambient + 0.45 * entryRefl + vec3(0.20));
        hopCol += hopThr * (0.22 + 0.18 * grazeWall) * cavRecess * wallLight;
        hopThr *= (1.0 - 0.26 * cavRecess - 0.20 * borderRim);
      }
    }

    vec3 rdC = rd;
    vec3 refrIn = refract(rdC, n, 1.0 / effIor);
    rdC = dot(refrIn, refrIn) > 1e-6 ? normalize(refrIn) : rdC;
    vec3 pathDisp = lastPathDisp + (rdC - rd) / max(effIor - 1.0, 0.2);
    vec3 roC = p;
    float travC = travelled;
    int kf = 0;
    float fr = 0.0;
    vec3 nf = n;
    vec3 o = rdC;

    vec3 rawExitN0;
    float len0 = exitCrystal(id, roC, rdC, kf, rawExitN0);
    vec3 trans0;
    int steps0 = (u_isMobile > 0.5)
      ? ((hop == 0) ? min(int(u_inclusionSteps + 0.5), 3) : 2)
      : ((hop == 0) ? clamp(int(u_inclusionSteps + 0.5), 2, 16) : clamp(int(u_inclusionSteps + 0.5), 2, 6));
    vec3 inScat0 = insideVolume(id, roC, rdC, len0, jitter, steps0, trans0);
    if (hop == 0) {
      insidePrimaryExtras(id, roC, rdC, len0, travC, inScat0, trans0);
    }
    hopCol += hopThr * inScat0;
    hopThr *= trans0;
    travC += len0;
    p = roC + rdC * len0;
    float exitFrostRoll = evalCrystalFrostRoll(id, rawExitN0, uid);
    float exitRoughBase = effFrost * smoothstep(exitFrostRoll - 0.15, exitFrostRoll + 0.15, effFrostShare);
    float crispExitRough = smoothstep(exitFrostRoll - 0.12, exitFrostRoll + 0.12, max(u_frostShare * 0.68, effFrostShare * 0.52 * dither)) * (max(u_frost, effFrost * 0.65 * dither) * mix(0.24, 0.36, dither));
    fr = mix(exitRoughBase, crispExitRough, effG1 * (1.0 - 0.32 * dither));
    float dExit0 = 1.0;
    if (g1 > 0.001 && u_edgeWear > 0.0 && kf < CP) {
      vec3 tmpAdj;
      dExit0 = edgeDist(id, kf, p, tmpAdj);
      vec3 relEx0 = (p - cMidMain) / LMain;
      vec3 lpEx0 = vec3(dot(relEx0, t1Main), dot(relEx0, axisMain), dot(relEx0, t2Main));
      float jbExEdge = 1.0 - smoothstep(0.0, u_edgeWidth * body(id, 2).w, dExit0);
      float jbExNz = texture(u_incl, lpEx0 * 4.5 + 0.71).b * 0.75 + texture(u_incl, lpEx0 * 15.0).b * 0.35;
      float jbExChip = jbExEdge * smoothstep(1.0 - 0.45 * u_edgeWear, 1.05 - 0.45 * u_edgeWear, jbExNz) * g1;
      fr = max(fr, jbExChip * 0.55);
    }
    nf = faceNormal(id, kf < CP ? kf : 18, rawExitN0, p, kf < CP ? fr : fr * 0.2, travC * f_pixAngle, 1.0);
    pathDisp += (rdC - nf * dot(rdC, nf)) * 0.55;
    if (g1 > 0.01 && kf == 16 && dot(body(id, PA + 16).xyz, body(id, 1).xyz) < -0.92) {
      hopThr *= (1.0 - 0.88 * g1);
    }
    o = refract(rdC, -nf, effIor);

    int maxB = (u_isMobile > 0.5)
      ? clamp(hop == 0 ? (min(bounces, 3) + 1) : 2, 1, 4)
      : ((g1 > 0.01) ? clamp(bounces, 1, 8) : clamp(hop == 0 ? bounces : 2, 1, 6));
    int stepsB = (u_isMobile > 0.5) ? 2 : clamp(int(u_inclusionSteps + 0.5), 2, 6);
    for (int b = 1; b < maxB; b++) {
      if (dot(o, o) >= 1e-6 || dot(hopThr, hopThr) < 1e-4) break;
      float Fb = fresnel(dot(rdC, nf), effIor);
      vec3 reflBDir = reflect(rdC, -nf);
      vec3 bounceRefl = facetIridescence(id, kf, p, rdC, -nf, 1.0, softFacetRefl(env(reflBDir, fr), hgtEntry));
      hopCol += hopThr * (Fb * bounceRefl * 0.26 * (1.0 - 0.65 * g1) * (0.20 + 0.80 * glare));
      pathDisp = reflect(pathDisp, -nf) + (reflBDir - nf * dot(reflBDir, nf)) * 0.28;
      rdC = reflBDir;
      roC = p;
      vec3 rawExitNB;
      float len = exitCrystal(id, roC, rdC, kf, rawExitNB);
      vec3 trans;
      vec3 inScat = insideVolume(id, roC, rdC, len, jitter, stepsB, trans);
      hopCol += hopThr * inScat;
      hopThr *= trans;
      travC += len;
      p = roC + rdC * len;
      exitFrostRoll = evalCrystalFrostRoll(id, rawExitNB, uid);
      exitRoughBase = effFrost * smoothstep(exitFrostRoll - 0.15, exitFrostRoll + 0.15, effFrostShare);
      crispExitRough = smoothstep(exitFrostRoll - 0.12, exitFrostRoll + 0.12, max(u_frostShare * 0.68, effFrostShare * 0.52 * dither)) * (max(u_frost, effFrost * 0.65 * dither) * mix(0.24, 0.36, dither));
      fr = mix(exitRoughBase, crispExitRough, effG1 * (1.0 - 0.32 * dither));
      float dExitB = 1.0;
      if (g1 > 0.001 && u_edgeWear > 0.0 && kf < CP) {
        vec3 tmpAdjB;
        dExitB = edgeDist(id, kf, p, tmpAdjB);
        vec3 relExB = (p - cMidMain) / LMain;
        vec3 lpExB = vec3(dot(relExB, t1Main), dot(relExB, axisMain), dot(relExB, t2Main));
        float jbExEdgeB = 1.0 - smoothstep(0.0, u_edgeWidth * body(id, 2).w, dExitB);
        float jbExNzB = texture(u_incl, lpExB * 4.5 + 0.71).b * 0.75 + texture(u_incl, lpExB * 15.0).b * 0.35;
        float jbExChipB = jbExEdgeB * smoothstep(1.0 - 0.45 * u_edgeWear, 1.05 - 0.45 * u_edgeWear, jbExNzB) * g1;
        fr = max(fr, jbExChipB * 0.55);
      }
      nf = faceNormal(id, kf < CP ? kf : 18, rawExitNB, p, kf < CP ? fr : fr * 0.2, travC * f_pixAngle, 1.0);
      if (g1 > 0.01 && kf == 16 && dot(body(id, PA + 16).xyz, body(id, 1).xyz) < -0.92) {
        hopThr *= (1.0 - 0.88 * g1);
      }
      o = refract(rdC, -nf, effIor);
    }
    bool leftCrystal = (dot(o, o) >= 1e-6);
    if (!leftCrystal) {
      if (g1 >= 0.5) {
        col += thr * hopCol;
        lastRefracted = false;
        break;
      }
      o = rdC;
    }
    vec3 exitAdjN;
    float dExit = (hop == 0 && (u_hopper > 0.0 || (u_iridescence > 0.0 && u_rainbow > 0.0) || u_bloom > 0.0)) ? edgeDist(id, kf, p, exitAdjN) : dExit0;
    float cosExitI = dot(rdC, nf);
    float Fo = fresnel(cosExitI, effIor);
    float exitHgt = clamp(dot(p - body(id, 3).xyz, body(id, 1).xyz) / max(body(id, 1).w, 0.1), 0.0, 1.0);
    float isRootCut = ((kf == 16 && dot(body(id, PA + 16).xyz, body(id, 1).xyz) < -0.85) ? 1.0 : (1.0 - smoothstep(0.06, 0.20, exitHgt)));
    vec3 reflExitDir = reflect(rdC, -nf);
    vec3 baseExitEnv = softFacetRefl(env(reflExitDir, fr), hgtEntry);
    vec3 stripExitEnv = glimmerStripLights(f_toWorld * reflExitDir, fr) * (1.0 - isRootCut);
    if (u_fire > 0.0) {
      float effGlim = glimmerCurve(glimmer);
      vec3 exitTan = rdC - nf * cosExitI;
      vec3 rSpread = (reflect(pathDisp, -nf) * 0.48 + exitTan * 0.28) * (u_fire * 0.82 * effGlim);
      vec3 dR = normalize(reflExitDir - rSpread);
      vec3 dB = normalize(reflExitDir + rSpread);
      vec3 sR = glimmerStripLights(f_toWorld * dR, fr) * (1.0 - isRootCut);
      vec3 sB = glimmerStripLights(f_toWorld * dB, fr) * (1.0 - isRootCut);
      stripExitEnv = vec3(
        mix(sR.r, stripExitEnv.r, 0.12),
        mix(stripExitEnv.g, 0.5 * (sR.g + sB.g), 0.16),
        mix(sB.b, stripExitEnv.b, 0.12)
      );
    }
    vec3 rawExitEnv = baseExitEnv + stripExitEnv;
    vec3 exitRefl = facetIridescence(id, kf, p, rdC, -nf, dExit, rawExitEnv / (1.0 + 0.24 * max(rawExitEnv.r, max(rawExitEnv.g, rawExitEnv.b))));
    vec3 exitFlash = (hop == 0) ? crystalBloomGlint(id, kf, p, rdC, -nf, dExit, fr) * 0.25 * (1.0 - 0.25 * dither) : vec3(0.0);
    float exitReflW = mix(0.36 * (1.0 - 0.28 * dither) * (0.15 + 0.85 * glare), 0.38 * (1.0 - 0.85 * isRootCut), effG1);
    hopCol += hopThr * (Fo * exitRefl * exitReflW + exitFlash);
    hopThr *= mix(1.0 - Fo * (0.3 + 0.7 * glare * (1.0 - 0.35 * dither)), 1.0 - Fo, g1);

    if (hop > 0 && metalW > 0.005) {
      hopCol = mix(hopCol, metalSurfCol, metalW * 0.35);
    }

    if (alpha < 0.999) {
      // Partial transparency / occlusion void feather: blend refracted crystal contribution and continue straight behind!
      vec3 cExitDir = normalize(o);
      vec3 cEnv = refractedEnv(cExitDir, fr, 1.0, hgtEntry);
      col += thr * alpha * (hopCol + hopThr * cEnv);
      thr *= (1.0 - alpha);
      ro = pEntry;
      continue;
    }

    col += thr * hopCol;
    thr *= hopThr * (1.0 - 0.65 * isRootCut * g1);
    lastIn = rdC;
    lastN = nf;
    lastPathDisp = pathDisp;
    lastRough = max(fr, rough * 0.45);
    lastFireAlpha = 1.0 - 0.88 * isRootCut * g1;
    lastRefracted = true;
    rd = normalize(o);
    ro = p;
    travelled = travC;
  }

  if (dot(thr, thr) > 0.0) {
    float dither = clamp(u_ditherBlend, 0.0, 1.0);
    float effGlimmer = glimmerCurve(glimmer);
    float effIrid = u_iridescence * u_rainbow;
    float effFire = mix(u_fire, u_fire * 0.75, dither) * (1.0 + 0.15 * effGlimmer) + effIrid * 0.035;
    if (lastRefracted && effFire > 0.005 && lastFireAlpha > 1e-4) {
      float dispIor = mix(1.0, ior, lastFireAlpha);
      float dispFire = effFire * lastFireAlpha;
      int ns = (u_isMobile > 0.5) ? 5 : clamp(int(u_fireSamples + 0.5), 2, 9);
      vec3 acc = vec3(0.0), wsum = vec3(0.0);
      for (int i = 0; i < ns; i++) {
        float x = (float(i) + 0.5) / float(ns);
        vec3 dw = (x - vec3(0.14, 0.5, 0.86)) / 0.40;
        vec3 w = max(1.0 - dw * dw, 0.0);
        vec3 inWave = normalize(lastIn + lastPathDisp * (dispFire * (x - 0.5) * 0.36 * effG1));
        vec3 o = refract(inWave, -lastN, dispIor + dispFire * (x - 0.5) * 2.0);
        wsum += w;
        if (dot(o, o) > 1e-6) {
          float tirFeather = smoothstep(0.0, 0.18, abs(dot(o, lastN)));
          vec3 sampleDir = normalize(mix(rd, o, lastFireAlpha));
          vec3 eCol = refractedEnv(sampleDir, lastRough, lastFireAlpha, lastHgt) * mix(0.28, 1.0, tirFeather);
          if (effIrid > 0.0) {
            float defl = length(sampleDir - initialRd);
            vec3 prism = spectralRainbow(defl * 2.4 + x * 0.85);
            eCol += prism * f_ambient * effIrid * mix(0.32, 0.18, dither) * lastFireAlpha * smoothstep(0.04, 0.45, defl);
          }
          acc += w * eCol;
        }
      }
      col += thr * acc / max(wsum, vec3(1e-4));
    } else {
      col += thr * evalMirrorBackground(q, rd, lastRough, lastFireAlpha, jitter);
    }
  }
  outColor = encodeHdr(col);
}`,

  bright: `
uniform sampler2D u_hdr;
out vec4 outColor;
void main() {
  vec2 px = 1.0 / u_screen, uv = gl_FragCoord.xy / u_res;
  vec3 c = decodeHdr(texture(u_hdr, uv + px * vec2(-1, -1))) + decodeHdr(texture(u_hdr, uv + px * vec2(1, -1)))
         + decodeHdr(texture(u_hdr, uv + px * vec2(-1, 1))) + decodeHdr(texture(u_hdr, uv + px * vec2(1, 1)));
  c *= 0.25 * u_exposure;
  float l = max(c.r, max(c.g, c.b));
  float thresh = max(1.45, u_bloomThreshold);
  vec3 hdrBloom = c * max(l - thresh, 0.0) / max(l, 1e-4);
  outColor = encodeHdr(hdrBloom);
}`,

  blur: `
uniform sampler2D u_src;
uniform vec2 u_dir;
out vec4 outColor;
void main() {
  vec2 uv = gl_FragCoord.xy / u_res, d = u_dir / u_res * f_px;
  vec3 c = decodeHdr(texture(u_src, uv)) * 0.227;
  c += (decodeHdr(texture(u_src, uv + d * 1.385)) + decodeHdr(texture(u_src, uv - d * 1.385))) * 0.316;
  c += (decodeHdr(texture(u_src, uv + d * 3.231)) + decodeHdr(texture(u_src, uv - d * 3.231))) * 0.070;
  outColor = encodeHdr(c);
}`,

  composite: `
uniform sampler2D u_hdr;
uniform sampler2D u_bloomTex;
out vec4 outColor;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float fxaaLuma(vec3 c) {
  vec3 t = c / (1.0 + c);
  return dot(t, vec3(0.299, 0.587, 0.114));
}
vec3 sampleSceneFxaa(vec2 uv) {
  vec3 cM = decodeHdr(texture(u_hdr, uv)) * u_exposure;
  float fxaaStrength = clamp(u_fxaa, 0.0, 1.0) * (1.0 - 0.65 * clamp(u_ditherBlend, 0.0, 1.0)) * (1.0 - 0.55 * clamp(u_glimmer, 0.0, 1.0));
  if (u_isMobile > 0.5) {
    fxaaStrength = max(fxaaStrength, clamp(u_fxaa, 0.0, 1.0) * 0.88);
  }
  if (fxaaStrength <= 0.005) return cM;
  vec2 px = 1.0 / u_res;
  vec3 cN  = decodeHdr(texture(u_hdr, uv + vec2(0.0, -px.y))) * u_exposure;
  vec3 cS  = decodeHdr(texture(u_hdr, uv + vec2(0.0,  px.y))) * u_exposure;
  vec3 cW  = decodeHdr(texture(u_hdr, uv + vec2(-px.x, 0.0))) * u_exposure;
  vec3 cE  = decodeHdr(texture(u_hdr, uv + vec2( px.x, 0.0))) * u_exposure;
  float lM = fxaaLuma(cM);
  float lN = fxaaLuma(cN);
  float lS = fxaaLuma(cS);
  float lW = fxaaLuma(cW);
  float lE = fxaaLuma(cE);
  float lMin = min(lM, min(min(lN, lS), min(lW, lE)));
  float lMax = max(lM, max(max(lN, lS), max(lW, lE)));
  float lRange = lMax - lMin;
  if (lRange < max(0.02, lMax * 0.075)) return cM;

  vec3 cNW = decodeHdr(texture(u_hdr, uv + vec2(-px.x, -px.y))) * u_exposure;
  vec3 cNE = decodeHdr(texture(u_hdr, uv + vec2( px.x, -px.y))) * u_exposure;
  vec3 cSW = decodeHdr(texture(u_hdr, uv + vec2(-px.x,  px.y))) * u_exposure;
  vec3 cSE = decodeHdr(texture(u_hdr, uv + vec2( px.x,  px.y))) * u_exposure;
  float lNW = fxaaLuma(cNW);
  float lNE = fxaaLuma(cNE);
  float lSW = fxaaLuma(cSW);
  float lSE = fxaaLuma(cSE);

  vec2 dir;
  dir.x = -((lNW + lNE) - (lSW + lSE));
  dir.y =  ((lNW + lSW) - (lNE + lSE));
  float dirReduce = max((lNW + lNE + lSW + lSE) * (0.25 * 0.125), 1.0 / 128.0);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;

  vec3 rgbA = 0.5 * (
    decodeHdr(texture(u_hdr, uv + dir * (1.0 / 3.0 - 0.5))) +
    decodeHdr(texture(u_hdr, uv + dir * (2.0 / 3.0 - 0.5)))
  ) * u_exposure;
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    decodeHdr(texture(u_hdr, uv + dir * -0.5)) +
    decodeHdr(texture(u_hdr, uv + dir *  0.5))
  ) * u_exposure;
  float lB = fxaaLuma(rgbB);
  vec3 aaCol = (lB < lMin || lB > lMax) ? rgbA : rgbB;

  float subpix = clamp(abs(((lN + lS + lW + lE) * 0.25) - lM) / max(lRange, 1e-4), 0.0, 1.0);
  subpix = smoothstep(0.0, 1.0, subpix);
  vec3 avgRing = (cN + cS + cW + cE + cNW + cNE + cSW + cSE) * 0.125;
  aaCol = mix(aaCol, avgRing, subpix * subpix * 0.28);
  return mix(cM, aaCol, fxaaStrength);
}
void main() {
  vec2 fc = gl_FragCoord.xy, uv = fc / u_res;
  vec3 bloomCol = decodeHdr(texture(u_bloomTex, uv));
  float dither = clamp(u_ditherBlend, 0.0, 1.0);
  float effBloom = u_bloom * 0.85 + u_rainbowGlow * 0.55 * (1.0 - 0.55 * dither);
  if (effBloom > 0.05) {
    vec2 sp = f_px * 3.5 / u_res;
    vec3 star = (
      decodeHdr(texture(u_bloomTex, uv + vec2(sp.x, 0.0))) +
      decodeHdr(texture(u_bloomTex, uv - vec2(sp.x, 0.0))) +
      decodeHdr(texture(u_bloomTex, uv + vec2(0.0, sp.y))) +
      decodeHdr(texture(u_bloomTex, uv - vec2(0.0, sp.y)))
    ) * 0.25;
    bloomCol += max(star - vec3(0.15), vec3(0.0)) * u_bloom * 0.65;
  }
  vec3 baseCol = sampleSceneFxaa(uv);
  float baseLuma = max(baseCol.r, max(baseCol.g, baseCol.b));
  float bloomLuma = max(bloomCol.r, max(bloomCol.g, bloomCol.b));
  float g01 = clamp(u_glimmer, 0.0, 1.0);
  float effG1 = smoothstep(0.0, 0.45, g01) * mix(0.85, 1.0, smoothstep(0.45, 1.0, g01));
  vec3 curHalo = (bloomCol * effBloom) / (1.0 + 0.35 * bloomLuma + 1.8 * baseLuma * baseLuma);
  vec3 jbHalo = bloomCol * (max(u_bloom, 0.35 * effG1) + u_rainbowGlow * 0.18 * (1.0 - 0.55 * dither));
  vec3 haloCol = mix(curHalo, jbHalo, effG1);
  vec3 c = baseCol + haloCol;
  c = pow(max(aces(c), vec3(0.0)), vec3(1.0 / 2.2));
  vec2 q = (fc - 0.5 * u_res) / u_res;
  c = mix(c, u_studioEdge, clamp(u_vignette * dot(q, q) * 2.6, 0.0, 1.0));
  c = filmDust(c, fc, f_px, f_frame, u_dust);
  c = filmGrain(c, fc, f_px, f_frame, u_grainAmount, u_grainSize, u_grainSharpness, u_grainClump);
  c += (hash(vec3(fc, f_frame)) - 0.5) / 255.0;
  outColor = vec4(c, 1.0);
}`,
};
