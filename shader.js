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
  },

  perFrame(P, t, rt, width, height, px, f) {
    f.f_px[0] = px;
    f.f_frame[0] = Math.floor(rt * P.filmFps.value);
    f.f_seed[0] = P.seed.value;
    f.f_crystals[0] = cluster.crystals;
    const mIdx = MIRROR_MODES.indexOf(String(P.mirrorMode?.value || 'Plain (None)'));
    f.f_mirrorMode[0] = mIdx > 0 ? mIdx : 0;

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

    const fov = P.fov.value * DEG, focal = 0.5 / Math.tan(fov / 2);
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
    let h = 0.75;
    if (d > 1e-5) {
      h = mx === s[0] ? ((s[1] - s[2]) / d) % 6 : mx === s[1] ? (s[2] - s[0]) / d + 2 : (s[0] - s[1]) / d + 4;
      h = ((h / 6) % 1 + 1) % 1;
    }
    const shiftDeg = (P.graduallyShift && P.graduallyShift.value === false) ? P.hueShift.value : 180;
    h = (((h + shiftDeg / 360) % 1) + 1) % 1;
    const hComp = (h + 0.5) % 1;
    // Ease the backdrop glow colours over ~1 s of real time. The stone hue is numerically unstable
    // for near-white/grey stones (e.g. High-Dispersion Diamond, where 1-step RGB changes swap the
    // max channel), which used to flash the whole background between red-brown and olive.
    let dtR = rt - glowState.rt;
    if (!(dtR >= 0)) dtR = 0.016;
    glowState.rt = rt;
    const aG = glowState.init ? 1 - Math.exp(-Math.min(dtR, 0.5) / 1.1) : 1;
    glowState.init = true;
    for (let c = 0; c < 3; c++) {
      const tg = lin(hslChannel(c === 0 ? 0 : c === 1 ? 8 : 4, h, P.backdropSat.value, 0.5)) * P.backdropLight.value;
      const tc = lin(hslChannel(c === 0 ? 0 : c === 1 ? 8 : 4, hComp, Math.max(0.58, P.backdropSat.value), 0.52)) * Math.max(0.11, P.backdropLight.value * 1.35);
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
vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
vec3 backdrop(vec3 d) {
  float ang = acos(clamp(-d.z, -1.0, 1.0));
  float g = exp(-pow(ang / radians(u_glowSize), 2.0));
  return mix(lin(u_studioEdge), f_glow, g);
}
vec3 lights(vec3 d, float rough) {
  vec3 sum = vec3(0.0);
  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 c = vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]);
    float cs = dot(d, c);
    if (cs <= 0.05) continue;
    vec3 p = d / cs;
    float hw = f_lights[o + 12], hh = f_lights[o + 13];
    float s = max(u_lightSoftness * min(hw, hh) + rough * 0.6, 1e-3);
    float u = dot(p, vec3(f_lights[o + 3], f_lights[o + 4], f_lights[o + 5]));
    float v = dot(p, vec3(f_lights[o + 6], f_lights[o + 7], f_lights[o + 8]));
    float m = (1.0 - smoothstep(hw - s, hw + s, abs(u))) * (1.0 - smoothstep(hh - s, hh + s, abs(v)));
    m *= 1.0 - 0.6 * clamp(dot(vec2(u / hw, v / hh), vec2(u / hw, v / hh)), 0.0, 1.0);
    m *= hw * hh / ((hw + s) * (hh + s));
    sum += m * vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
  }
  return sum;
}
vec3 env(vec3 dc, float rough) {
  vec3 d = f_toWorld * dc;
  return backdrop(d) + lights(d, rough);
}
vec3 spectralRainbow(float t) {
  vec3 x = fract(t + vec3(0.0, 0.66667, 0.33333));
  vec3 wave = clamp(abs(x * 6.0 - 3.0) - 1.0, 0.0, 1.0);
  return wave * wave * (3.0 - 2.0 * wave);
}
vec4 encodeHdr(vec3 c) { return vec4(c / (1.0 + c), 1.0); }
vec3 decodeHdr(vec4 e) { return e.rgb / max(1.0 - e.rgb, 1e-4); }
`,

  bake: `
#define INCL ${INCL}.0
uniform float u_slice;
out vec4 outColor;

float pnoise(vec3 p, float period) {
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec3 s = vec3(f_seed * 17.3 + 7.0, f_seed * 31.1 + 3.0, f_seed * 13.7 + 11.0);
  vec3 i0 = mod(i, period) + s;
  vec3 i1 = mod(i + 1.0, period) + s;
  float n000 = hash(vec3(i0.x, i0.y, i0.z));
  float n100 = hash(vec3(i1.x, i0.y, i0.z));
  float n010 = hash(vec3(i0.x, i1.y, i0.z));
  float n110 = hash(vec3(i1.x, i1.y, i0.z));
  float n001 = hash(vec3(i0.x, i0.y, i1.z));
  float n101 = hash(vec3(i1.x, i0.y, i1.z));
  float n011 = hash(vec3(i0.x, i1.y, i1.z));
  float n111 = hash(vec3(i1.x, i1.y, i1.z));
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z) * 2.0 - 1.0;
}

void main() {
  vec3 p = vec3(gl_FragCoord.xy, u_slice + 0.5) / INCL;
  float n4 = pnoise(p * 4.0, 4.0);
  float n8 = pnoise(p * 8.0, 8.0);
  float m = 0.58 * n4 + 0.30 * n8;
  float milk = pow(clamp((m + 0.05) * 1.6, 0.0, 1.0), 1.8);
  vec3 w = vec3(n4, pnoise(p * 2.0 + 7.7, 2.0), -n4) * 0.65;
  float sheet = 1.0 - smoothstep(0.0, 0.045, abs(pnoise(p * 3.0 + w * 1.2, 3.0)));
  float n2 = pnoise(p * 2.0 + 9.0, 2.0);
  float patchy = smoothstep(-0.05, 0.25, 0.55 * n2 + 0.28 * n4);
  float bubbles = mix(0.25, 1.0, step(0.6, hash(vec3(gl_FragCoord.xy, u_slice) + f_seed * 977.0)));
  float veil = sheet * patchy * bubbles;
  float mottle = clamp((0.52 * n2 + 0.26 * n4 + 0.13 * n8) * 0.72 + 0.5, 0.0, 1.0);
  float sparkle = hash(vec3(gl_FragCoord.xy, u_slice + 4096.0) + f_seed * 389.0);
  outColor = vec4(milk, veil, mottle, sparkle);
}`,

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

// 1a. Massive Abstract Ovoid Occlusion Voids (u_occlusionType == 0): fixed-radius 3D displacing spheres/wedges
float evalOvoidOcclusionVoid(int row, vec3 ro, vec3 rd, float tEntry, out vec3 outWallN, out float outRim) {
  outWallN = vec3(0.0, 1.0, 0.0);
  outRim = 0.0;
  if (u_occlusion <= 0.001) return 0.0;
  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float Lh = max(body(row, 1).w, 0.02);
  float rh = max(body(row, 2).w, 0.01);
  vec3 sideC = body(row, PA).xyz - ax * dot(body(row, PA).xyz, ax);
  int nextRow = occPartnerRow(row, 0);
  float g = occGrowthFor(row);
  float bestCut = 0.0;
  for (int m = 0; m < 2; m++) {
    vec4 rawEc = body(nextRow, m);
    // Each void is a bubble seeded just under this crystal's surface (on the side facing the
    // partner crystal) that grows with the occlusion level and shrinks back to that seed. It opens
    // as a small hole that steadily widens, instead of appearing late and already huge (when it
    // grew from the partner's centre deep inside the cluster, whole crystals vanished in ~3 s).
    vec3 axPt = rt + ax * (Lh * (m == 0 ? 0.55 : 0.8));
    vec3 toC = rawEc.xyz - axPt;
    vec3 perpT = toC - ax * dot(toC, ax);
    vec3 outDir = normalize(perpT + sideC * max(0.0, rh - length(perpT)) + vec3(1e-6));
    float rad = rawEc.w * 0.92 * g;
    vec4 ec = vec4(axPt + outDir * (rh * 0.85 + 0.35 * rad), rad);
    if (ec.w <= 1e-4) continue;
    vec3 oc = ro - ec.xyz;
    float b = dot(oc, rd);
    float h = b * b - dot(oc, oc) + ec.w * ec.w;
    if (h <= 0.0) continue;
    float s = sqrt(h);
    float tc0 = -b - s, tc1 = -b + s;
    // Radial direction from this crystal's axis line to the void centre. When the centre sits on
    // (or crosses) the axis line this vector is ~0 and normalizing it flips the wedge planes from
    // frame to frame (whole chunks popped), so blend smoothly toward the crystal's own side axis.
    vec3 perpC = ec.xyz - rt - ax * dot(ec.xyz - rt, ax);
    vec3 radDir = normalize(perpC + sideC * max(0.0, ec.w * 0.35 - length(perpC)) + vec3(1e-6));
    vec3 nUp = normalize(ax + radDir * 0.42);
    vec3 nDn = normalize(-ax + radDir * 0.48);
    float dUp = ec.w * 0.62 - dot(nUp, oc);
    float dnUp = dot(nUp, rd);
    if (abs(dnUp) > 1e-6) {
      float tUp = dUp / dnUp;
      if (dnUp < 0.0) tc0 = max(tc0, tUp); else tc1 = min(tc1, tUp);
    } else if (dUp < 0.0) continue;
    float dDn = ec.w * 0.66 - dot(nDn, oc);
    float dnDn = dot(nDn, rd);
    if (abs(dnDn) > 1e-6) {
      float tDn = dDn / dnDn;
      if (dnDn < 0.0) tc0 = max(tc0, tDn); else tc1 = min(tc1, tDn);
    } else if (dDn < 0.0) continue;
    if (tc1 > tc0 && tc1 > tEntry) {
      float frontSoft = 1.0 - smoothstep(tEntry - ec.w * 0.06, tEntry + ec.w * 0.28, tc0);
      float span = max(0.0, tc1 - max(tc0, tEntry));
      float depthSoft = smoothstep(0.0, max(ec.w * 0.36, 1e-3), span);
      float cut = frontSoft * depthSoft;
      if (cut > bestCut) {
        bestCut = cut;
        vec3 pHit = ro + rd * max(tc0, tEntry);
        vec3 relC = pHit - ec.xyz;
        outWallN = -normalize(relC + 1e-5);
        outRim = clamp(cut * (1.0 - cut) * 4.0, 0.0, 1.0);
      }
    }
  }
  return clamp(bestCut, 0.0, 1.0);
}

vec3 rotateAroundAxis(vec3 v, vec3 k, float cosA, float sinA) {
  return v * cosA + cross(k, v) * sinA + k * dot(k, v) * (1.0 - cosA);
}

// 1b. Invisible Crystal Occlusions (u_occlusionType == 1): invisible crystals of the exact same shape & form (facet planes 0..17) as the current batch occluding visible crystals
float evalInvisibleCrystalVoid(int row, vec3 ro, vec3 rd, float tEntry, out vec3 outWallN, out float outRim) {
  outWallN = vec3(0.0, 1.0, 0.0);
  outRim = 0.0;
  if (u_occlusion <= 0.001) return 0.0;
  vec3 axRow = body(row, 1).xyz;
  float LRow = max(body(row, 1).w, 0.08);
  float rRow = max(body(row, 2).w, 0.02);
  vec3 rtRow = body(row, 3).xyz;
  vec3 p0 = body(row, PA).xyz;
  vec3 side1 = p0 - axRow * dot(p0, axRow);
  float sLen = length(side1);
  side1 = sLen > 1e-4 ? (side1 / sLen) : normalize(cross(axRow, vec3(0.577, 0.577, 0.577)));
  vec3 side2 = cross(axRow, side1);

  float bestCut = 0.0;

  for (int m = 0; m < 2; m++) {
    int tRow = occPartnerRow(row, m);
    vec3 axTpl = body(tRow, 1).xyz;
    float LTpl = max(body(tRow, 1).w, 0.08);
    float rTpl = max(body(tRow, 2).w, 0.02);
    vec3 rtTpl = body(tRow, 3).xyz;
    vec3 cTpl = rtTpl + axTpl * (LTpl * 0.48);

    float hFrac = (m == 0) ? 0.44 : 0.72;
    // While small, seed the void just under the crystal's surface so its hole is visible from the
    // start and widens steadily; ease back to the original interior anchor as it reaches full size.
    vec3 cAnchor = rtRow + axRow * (LRow * hFrac) + ((m == 0) ? side1 : side2) * (rRow * mix(0.92, 0.22, occGrowthFor(row)));
    vec3 rotAxis = (m == 0) ? normalize(side2 + axRow * 0.25) : normalize(side1 - axRow * 0.22);
    float ang = (m == 0) ? 1.08 : -0.96;
    float cosA = cos(ang), sinA = sin(ang);
    float invScale = clamp((rTpl / max(rRow, 0.02)) * 0.80, 0.55, 1.35) / max(occGrowthFor(row), 0.02);

    vec3 roTpl = cTpl + rotateAroundAxis((ro - cAnchor) * invScale, rotAxis, cosA, sinA);
    vec3 rdTpl = rotateAroundAxis(rd * invScale, rotAxis, cosA, sinA);

    float tc0 = -1e9, tc1 = 1e9;
    vec3 hitN = vec3(0.0, 1.0, 0.0);
    for (int k = 0; k < 18; k++) {
      vec4 pl = body(tRow, PA + k);
      float dn = dot(pl.xyz, rdTpl);
      float dist = pl.w - dot(pl.xyz, roTpl);
      if (abs(dn) < 1e-8) {
        if (dist < 0.0) { tc0 = 1e9; break; }
        continue;
      }
      float tPl = dist / dn;
      if (dn < 0.0) {
        if (tPl > tc0) { tc0 = tPl; hitN = pl.xyz; }
      } else if (tPl < tc1) {
        tc1 = tPl;
      }
      if (tc0 > tc1) break;
    }

    if (tc1 > tc0 && tc1 > tEntry) {
      float feather = rRow * 0.06;
      float frontSoft = 1.0 - smoothstep(tEntry - feather * 0.4, tEntry + feather * 1.4, tc0);
      float span = max(0.0, tc1 - max(tc0, tEntry));
      float depthSoft = smoothstep(0.0, max(rRow * 0.14, 1e-3), span);
      float cut = frontSoft * depthSoft;
      if (cut > bestCut) {
        bestCut = cut;
        vec3 worldWallN = rotateAroundAxis(hitN, rotAxis, cosA, -sinA);
        outWallN = -normalize(worldWallN + 1e-5);
        outRim = clamp(cut * (1.0 - cut) * 4.0, 0.0, 1.0);
      }
    }
  }
  return clamp(bestCut, 0.0, 1.0);
}

// Unified Occlusion Void Evaluator (blends Ovoids at u_occlusionType=0 and Invisible Crystals at u_occlusionType=1)
float evalMassiveOcclusionVoid(int row, vec3 ro, vec3 rd, float tEntry, out vec3 outWallN, out float outRim) {
  outWallN = vec3(0.0, 1.0, 0.0);
  outRim = 0.0;
  if (u_occlusion <= 0.002) return 0.0;
  float kType = clamp(u_occlusionType, 0.0, 1.0);
  vec3 nOvoid = vec3(0.0, 1.0, 0.0), nCryst = vec3(0.0, 1.0, 0.0);
  float rOvoid = 0.0, rCryst = 0.0;
  float cOvoid = (kType < 0.99) ? evalOvoidOcclusionVoid(row, ro, rd, tEntry, nOvoid, rOvoid) : 0.0;
  float cCryst = (kType > 0.01) ? evalInvisibleCrystalVoid(row, ro, rd, tEntry, nCryst, rCryst) : 0.0;
  outWallN = normalize(mix(nOvoid, nCryst, kType) + vec3(1e-5));
  outRim = mix(rOvoid, rCryst, kType);
  return mix(cOvoid, cCryst, kType);
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

float evalEdgeContourProfile(float alongEdge, float edgeSeed, float gran, float fwAlong, out float outSlope, out float outAsym) {
  float g = clamp(gran, 0.2, 3.5);
  float x = alongEdge * (0.80 + 0.45 * g);
  float hiDamp = 1.0 / (1.0 + 12.0 * fwAlong);
  float terrace = steppedNoise(x * 2.2, edgeSeed);
  float chipCell = cellularChip(x * 3.4 + 1.3, edgeSeed + 19.0);
  float macroWave = sn1(x * 1.8, edgeSeed + 7.0);
  // Intermittent stepped 3D crystallographic bevels & conchoidal chip bites (~40% untouched sharp edge)
  float c0 = clamp((terrace - 0.36) * 0.85 + chipCell * 0.72 + macroWave * 0.22, 0.0, 1.35);
  float dx = 0.03;
  float terraceP = steppedNoise((x + dx) * 2.2, edgeSeed);
  float chipCellP = cellularChip((x + dx) * 3.4 + 1.3, edgeSeed + 19.0);
  float macroWaveP = sn1((x + dx) * 1.8, edgeSeed + 7.0);
  float cPlus = clamp((terraceP - 0.36) * 0.85 + chipCellP * 0.72 + macroWaveP * 0.22, 0.0, 1.35);
  outSlope = clamp((cPlus - c0) / dx, -0.85, 0.85) * hiDamp;
  float stepAsym = (steppedNoise(x * 2.9 + 4.3, edgeSeed + 29.0) - 0.5) * 2.0;
  outAsym = clamp(stepAsym, -0.9, 0.9);
  return c0;
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
  if (dRidgeStart >= maxPossibleCut) return;

  vec3 edgeDir = cross(nLo, nHi);
  float edLen = length(edgeDir);
  if (edLen <= 1e-4) return;
  edgeDir /= edLen;
  if (dot(edgeDir, ax + (nLo - nHi) * 0.15) < 0.0) edgeDir = -edgeDir;

  float dnAB = dot(nAB, rd);
  float nomBevel = r * (0.0032 * effIrreg + 0.0028 * effChips) * (0.75 + 0.25 * chipSz) * nodePresence;
  float tNom = (abs(dnAB) > 1e-3) ? clamp((wAB - nomBevel - dot(nAB, ro)) / dnAB, tStart, min(tf, tStart + r * 0.08)) : tStart;
  vec3 pRidge = ro + rd * tNom;
  vec3 relRidge = pRidge - rt;
  float hNorm = dot(relRidge, ax) / max(L * 0.45, 0.06);
  vec3 radEdgeDir = edgeDir - ax * dot(edgeDir, ax);
  float rNorm = dot(relRidge, radEdgeDir) / max(r * 1.15, 0.03);
  float alongEdge = hNorm * abs(dot(edgeDir, ax)) + rNorm;
  if (alongEdge < -1.1 || alongEdge > 2.6) return;

  float edgeSeed = uid * 13.7 + float(kLo) * 19.3 + float(kHi) * 7.1;
  float cutBest = 0.0;
  float cSlope = 0.0, cAsym = 0.0;

  if (effIrreg > 0.005 || effChips > 0.005) {
    float rawContour = evalEdgeContourProfile(alongEdge, edgeSeed, gran, fwBase, cSlope, cAsym);
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

  // Invariant 3D CSG bisector half-space (wAB - cutBest) with stepped crystallographic facet normal nChip
  vec3 bevelTan = normalize(nLo - nHi + vec3(1e-5));
  vec3 nChip = normalize(nAB + bevelTan * (0.22 * cAsym) - edgeDir * (0.08 * cSlope));
  float tCand = tStart + startExcess / max(-dnAB, 0.28);
  if (tCand > bestTChip) {
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
  float effEdgeMacro = max(0.0, u_patinaMacro * u_edgeMacro);
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

  for (int k = 0; k < 16; k++) {
    if (k == k0) continue;
    vec4 pl = body(row, PA + k);
    if (pl.w - dot(pl.xyz, rt) > maxValidOff) continue;
    float cAng = dot(pl.xyz, n0);
    if (cAng > 0.992 || cAng < -0.75) continue;
    float d = max(0.0, pl.w - dot(pl.xyz, pStart));
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
  if (whaleMult <= 1.0 && d1 >= r * 0.095 * max(0.45, max(chipSz, effIrreg))) return true;

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

float exitCrystal(int row, vec3 ro, vec3 rd, out int kf) {
  float tf = 1e9; kf = 0;
  for (int k = 0; k < CP; k++) {
    vec4 pl = body(row, PA + k);
    float dn = dot(pl.xyz, rd);
    if (dn > 1e-8) {
      float t = (pl.w - dot(pl.xyz, ro)) / dn;
      if (t < tf) { tf = t; kf = k; }
    }
  }
  return max(tf, 0.0);
}

bool traceScene(vec3 ro, vec3 rd, int skipMask, out int id, out float tHit, out float tfHit, out int kHit) {
  bool hit = false; id = 0; kHit = 0; tHit = 1e9; tfHit = 1e9;
  float tn, tf; int kn, kf;
  int nc = clamp(int(f_crystals + 0.5), 0, MAX_CRYSTALS);
  for (int i = 0; i < nc; i++) {
    if ((skipMask & (1 << i)) != 0 || !sphereHit(ro, rd, body(i, 0), tHit)) continue;
    if (convexHit(i, ro, rd, tn, tf, kn, kf)) {
      float tCand = max(tn, 0.0);
      if (tf > tCand + 1e-4 && tCand < tHit) { tHit = tCand; tfHit = tf; id = i; kHit = kn; hit = true; }
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

  float hasFlash = (maxSteps > 5 && u_flash > 0.01) ? step(0.68, fract(rnd * 17.37 + 0.23)) : 0.0;
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
  bool needInclTex = (u_milk > 0.01 || u_veils > 0.01 || u_iridescence > 0.01 || u_colourZoning > 0.01);
  for (int s = 0; s < n; s++) {
    vec3 pp = ro + rd * (dt * (float(s) + jitter));
    vec3 rel = pp - root;
    float axial = dot(rel, axis);
    float hgt = clamp(axial / max(L, 1e-3), 0.0, 1.0);
    float radFrac = clamp(length(rel - axis * axial) / max(r, 1e-3), 0.0, 1.0);
    float coreMix = u_biColour * clamp((1.0 - radFrac * 0.85) * 0.65 + (1.0 - hgt) * 0.55, 0.0, 1.0);
    vec3 baseSigma = mix(f_sigma, f_sigmaCore, coreMix);
    vec3 gemBodyCol = exp(-baseSigma * 0.38);
    vec3 tintedLightIn = lightIn * mix(vec3(1.0), normalize(gemBodyCol + vec3(0.04)) * 1.28, clamp(length(baseSigma) * 0.24, 0.0, 0.88));
    vec4 v = needInclTex ? texture(u_incl, pp * u_inclusionScale + off) : vec4(0.0, 0.0, 0.5, 0.0);
    float a = 0.0;
    if (u_milk > 0.01 || u_veils > 0.01) {
      float milk = u_milk * v.r * mix(1.0, 2.2 * pow(1.0 - hgt, 2.0), u_milkBase);
      float veil = u_veils * v.g;
      float dens = (milk + veil) * 5.2;
      a = 1.0 - exp(-dens * dt);
      float bloomSparkle = u_bloom > 0.0 ? (1.0 + u_bloom * 1.6 * pow(v.a, 8.0)) : 1.0;
      scat += T * a * tintedLightIn * (1.0 + 2.8 * veil * pow(v.a, 6.0) * bloomSparkle);
    }
    if (schillerResonance > 1e-3) {
      vec3 radialVec = (rel - axis * axial) - flashCenterOffset;
      float dAxial = (hgt - flashCenterHgt) / 0.095;
      float dRadial = length(radialVec) / max(r * 0.26, 1e-3);
      float singleGlow = exp(-(dAxial * dAxial + dRadial * dRadial));
      float coreEnvelope = smoothstep(0.92, 0.25, radFrac) * smoothstep(0.08, 0.24, hgt) * smoothstep(0.92, 0.74, hgt);
      vec3 coreGemGlow = normalize(gemBodyCol + vec3(0.04)) * 1.28;
      vec3 schillerTint = spectralRainbow(hueOff + cosFlashAng * 0.35);
      vec3 deepFlashCol = mix(coreGemGlow, schillerTint * (coreGemGlow + vec3(0.22)), 0.32);
      float flashPower = u_flash * (0.65 + 0.45 * u_flash);
      scat += T * coreEnvelope * singleGlow * schillerResonance * deepFlashCol * (f_ambient * 2.2 + vec3(0.38)) * flashPower * opticalScale * dt * 2.4;
    }
    if (u_iridescence > 0.01 && u_rainbow > 0.001) {
      float iridPhase = hgt * 1.85 + radFrac * 1.25 + dot(rd, axis) * 1.15 + hueOff + v.b * 0.75;
      vec3 iridWave = spectralRainbow(iridPhase);
      float veilBand = (0.03 + 0.14 * v.g + 0.06 * v.r) * clamp(u_iridescence * 0.48, 0.0, 1.3) * u_rainbow;
      scat += T * iridWave * tintedLightIn * veilBand * dt * 2.1;
    }
    vec3 sigma = baseSigma * opticalScale * (1.0 + u_colourZoning * (2.0 * hgt - 1.0)) * (0.75 + 0.5 * v.b);
    T *= (1.0 - a) * exp(-sigma * absorbDt);
  }

  // Deep internal iridescent cleavage plane (full f50bae4 optical pipeline scaled by u_rainbow)
  if ((u_fractures > 0.01 || u_iridescence > 0.01 || u_flash > 0.01) && u_rainbow > 0.001) {
    float uid = crystalUid(id);
    float effFractures = clamp(u_fractures + u_iridescence * 0.22 + u_flash * 0.25, 0.0, 1.0);
    float fracRoll = hash(vec3(uid, f_seed, 91.0));
    float fracW = smoothstep(fracRoll - 0.14, fracRoll + 0.14, effFractures);
    if (fracW > 1e-4) {
      vec3 hj = vec3(uid, f_seed, 93.0);
      vec3 fn = normalize(vec3(hash(hj), hash(hj + 1.0), hash(hj + 2.0)) - 0.5);
      vec3 fc = root + axis * L * (0.25 + 0.4 * hash(hj + 3.0));
      float dn = dot(fn, rd);
      if (abs(dn) > 1e-4) {
        float t = dot(fn, fc - ro) / dn;
        if (t > 0.0 && t < len) {
          vec3 hp = ro + rd * t;
          vec4 v = texture(u_incl, hp * 2.5 + off);
          float extent = length(hp - fc) / (r * (1.6 + 0.35 * u_iridescence));
          float mask = fracW * (1.0 - smoothstep(0.55, 0.78, extent + 0.45 * (v.b - 0.5) + 0.25 * v.r));
          if (mask > 0.0) {
            float film = v.b * 2.5 + length(hp - fc) * 6.0 + hueOff + abs(dn) * 2.2;
            vec3 irid = spectralRainbow(film);
            float fr = 0.18 + 0.82 * pow(1.0 - abs(dn), 3.2);
            vec3 refl = env(reflect(rd, fn), 0.03) + f_ambient * (0.35 * u_flash + 0.2 * u_iridescence);
            float glowBoost = u_fractureGlow * (1.0 + u_iridescence * 1.2 + u_flash * 1.45) * u_rainbow;
            scat += T * mask * glowBoost * fr * irid * refl;
            T *= 1.0 - 0.55 * mask * fr * clamp(u_rainbow, 0.0, 1.0);
          }
        }
      }
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

  // Multi-generation nested phantoms (evaluated once on primary ray outside bounce loop)
  float phRateRoll = hash(vec3(uid, f_seed, 97.0));
  float phRateW = smoothstep(phRateRoll - 0.14, phRateRoll + 0.14, u_phantomRate);
  if (u_phantom > 0.01 && phRateW > 1e-4) {
    vec3 phTint = exp(-mix(f_sigma, f_sigmaCore, 0.4) * 0.35);
    vec3 phLightIn = lightIn * mix(vec3(1.0), normalize(phTint + vec3(0.05)) * 1.25, clamp(length(f_sigma) * 0.22, 0.0, 0.85));
    int maxPhLayers = clamp(int(ceil(u_phantomLayers)), 0, 4);
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
          float cosk = abs(dot(rd, e == 0 ? nEnter : nExit));
          float dust = texture(u_incl, hp * u_inclusionScale * 2.0 + off + float(layer) * 0.37).r;
          float sheet = phRateW * layerWeight * clamp(u_phantom * (0.12 - 0.02 * float(layer)) * (0.3 + 1.7 * dust) / max(cosk, 0.12), 0.0, 1.0);
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

  // Rutile needles (evaluated once on primary ray outside bounce loop)
  if (u_needles > 0.01) {
    int maxNeedles = clamp(int(ceil(u_needles)), 0, 12);
    for (int j = 0; j < maxNeedles; j++) {
      float needleWeight = clamp(u_needles - float(j), 0.0, 1.0);
      if (needleWeight <= 1e-4) break;
      vec3 hj = vec3(uid, float(j), f_seed);
      vec3 h3 = vec3(hash(hj + 0.5), hash(hj + 1.5), hash(hj + 2.5));
      vec3 h4 = vec3(hash(hj + 3.5), hash(hj + 4.5), hash(hj + 5.5));
      vec3 u = normalize(h4 - 0.5 + axis * 0.15);
      float nl = L * (0.25 + 0.5 * hash(hj + 6.5));
      vec3 radial = (h3 - 0.5) * 1.4 * r;
      radial -= axis * dot(radial, axis);
      vec3 a = root + axis * L * (0.15 + 0.6 * h3.y) + radial - u * nl * 0.5;
      vec3 w0 = ro - a;
      float b = dot(rd, u), d = dot(rd, w0), e = dot(u, w0);
      float den = max(1.0 - b * b, 1e-6);
      float s = clamp((b * e - d) / den, 0.0, len);
      float tq = clamp(e + b * s, 0.0, nl);
      s = clamp(b * tq - d, 0.0, len);
      vec3 dv = w0 + rd * s - u * tq;
      float dist = length(dv);
      float fp = (travelled + s) * f_pixAngle;
      float cov = needleWeight * (1.0 - smoothstep(u_needleWidth - fp, u_needleWidth + fp, dist));
      cov *= min(1.0, u_needleWidth / max(fp, 1e-6));
      if (cov > 0.001) {
        vec3 cn = dv / max(dist, 1e-6);
        vec3 gold = lin(u_needleColour);
        vec3 shade = gold * (lightIn * 0.4 + lights(f_toWorld * reflect(rd, cn), 0.05) * (0.8 + u_bloom * 1.4));
        scat += T * cov * shade;
        T *= 1.0 - cov;
      }
    }
  }

  // Occasional razor-thin internal & sub-surface hairline fractures (u_hairlineFractures)
  // Monotone single-valued gently curved cleavage traces near terminations & edges (zero closed-loop bullseye rings!)
  float effHairline = clamp(u_hairlineFractures * u_patinaMacro * u_surfaceMacro, 0.0, 2.0);
  if (effHairline > 0.005) {
    vec3 p0 = body(id, PA).xyz;
    vec3 t1 = p0 - axis * dot(p0, axis);
    float t1Len = length(t1);
    t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
    vec3 t2 = cross(axis, t1);
    float invScale = 1.0 / max(r * 1.35, 0.04);
    float fp = max(travelled * f_pixAngle, 0.0008);

    for (int fIdx = 0; fIdx < 2; fIdx++) {
      vec3 hSeed = vec3(uid * 19.0 + float(fIdx) * 37.0, f_seed + 71.0, 103.0);
      float azF = hash(hSeed) * TAU;
      float tiltF = mix(0.38, 0.68, hash(hSeed + 1.1));
      vec3 fracN = normalize(vec3(cos(azF) * sin(tiltF), cos(tiltF), sin(azF) * sin(tiltF)));
      vec3 fracTan = normalize(cross(fracN, vec3(-sin(azF), 0.25, cos(azF))));
      vec3 fracBit = cross(fracN, fracTan);
      vec3 fracWorldN = normalize(t1 * fracN.x + axis * fracN.y + t2 * fracN.z);
      float anchorH = mix(0.52, 0.86, hash(hSeed + 2.3));
      vec2 rimDir = vec2(cos(azF + 0.4), sin(azF + 0.4));
      vec3 lpAnchor = vec3(rimDir.x * r * 0.82 * invScale, anchorH * L * invScale, rimDir.y * r * 0.82 * invScale);
      vec3 seedOff = vec3(uid * 3.7 + float(fIdx) * 11.3, float(fIdx) * 7.1, f_seed * 0.29);

      // Sub-surface & shallow internal monotone hairline cleavage traces
      for (int sl = 0; sl < 2; sl++) {
        float tSlice = (sl == 0) ? min(len * 0.04, r * 0.028) : min(len * 0.16, r * 0.10);
        vec3 hp = ro + rd * tSlice;
        vec3 rel = hp - root;
        float axial = dot(rel, axis);
        float hgt = clamp(axial / max(L, 1e-3), 0.0, 1.0);
        float radFrac = clamp(length(rel - axis * axial) / max(r, 1e-3), 0.0, 1.2);
        vec3 lp = vec3(dot(rel, t1), axial, dot(rel, t2)) * invScale;

        vec3 dLp = lp - lpAnchor;
        float uAlong = dot(dLp, fracTan);
        float wAcross = dot(dLp, fracBit);
        // Natural conchoidal feather arc that is strictly monotone in dot(dLp, fracN) (df/dv == 1 -> never forms closed bullseye loops!)
        float gentleBend = 0.12 * sin(uAlong * 3.6 + seedOff.x)
                         + 0.045 * cos(uAlong * 7.2 + wAcross * 2.2 + seedOff.y)
                         + 0.28 * uAlong * uAlong;
        float fVal = dot(dLp, fracN) + gentleBend;

        float spanAlong = mix(0.32, 0.52, fract(hash(hSeed + 3.1) * 13.7));
        float uNorm = length(vec2(uAlong, wAcross * 0.75)) / spanAlong;
        float endTaper = 1.0 - smoothstep(0.25, 1.0, uNorm);
        float edgeBias = smoothstep(0.42, 0.85, radFrac * 0.72 + hgt * 0.48);
        float crystalGate = smoothstep(0.32, 0.56, hash(hSeed + 4.7) + 0.22 * min(effHairline, 1.2));
        float pocket = endTaper * edgeBias * crystalGate;

        if (pocket > 0.005) {
          float fwF = max(fp * invScale * 1.4, 0.006);
          float targetW = (0.0075 + 0.0035 * min(effHairline, 1.2)) * (sl == 1 ? 1.25 : 1.0);
          float effW = max(targetW, fwF);
          float aaScale = clamp(targetW / effW, 0.18, 1.0);
          float lineMask = (1.0 - smoothstep(effW * 0.20, effW, abs(fVal))) * aaScale * pocket;
          if (lineMask > 0.001) {
            float slWeight = (sl == 0) ? 0.82 : 0.42;
            float graze = 0.35 + 0.65 * pow(1.0 - abs(dot(rd, fracWorldN)), 2.2);
            vec3 tirRefl = env(reflect(rd, fracWorldN), 0.03);
            vec3 silvCol = vec3(0.95, 0.97, 1.0) * (0.42 + 0.40 * f_ambient + 0.55 * (tirRefl / (1.0 + max(tirRefl.r, max(tirRefl.g, tirRefl.b)))));
            float strength = clamp(lineMask * slWeight * graze * effHairline * 0.72, 0.0, 0.65);
            scat += T * strength * silvCol;
            T *= 1.0 - 0.15 * strength;
          }
        }
      }
    }
  }
}

// Bend a face's normal: growth ridges across prism faces, stepped hopper terraces, and smooth mineral frost relief.
vec3 faceNormal(int id, int k, vec3 n, vec3 p, float rough, float footprint, float dEdge) {
  vec4 ax = body(id, 1);
  float uid = crystalUid(id);
  if (k < 8 && u_striations > 0.0) {
    float axialY = dot(p - body(id, 3).xyz, ax.xyz);
    float fFine = u_striationScale * 0.30;
    float fCoarse = u_striationScale * 0.10;
    float aaFine = clamp(1.0 - (footprint * fFine) * 5.8, 0.0, 1.0);
    float aaCoarse = clamp(1.0 - (footprint * fCoarse) * 5.8, 0.0, 1.0);
    float amp = texture(u_incl, vec3(axialY * 1.8, uid * 0.37, 0.5)).b;
    float ridge = aaCoarse * 0.68 * sin(TAU * axialY * fCoarse + 1.5 * sin(axialY * 11.0))
                + aaFine * 0.32 * sin(TAU * axialY * fFine);
    n = normalize(n + ax.xyz * (u_striations * 0.026 * amp * ridge));
  }
  if (u_hopper > 0.0) {
    float r = max(body(id, 2).w, 0.02);
    float u = dEdge / (r * 0.22);
    float riser = smoothstep(0.72, 0.96, fract(u)) - smoothstep(0.0, 0.18, fract(u));
    float mask = smoothstep(0.06, 0.22, dEdge / r) * (1.0 - smoothstep(0.85, 1.15, dEdge / r));
    vec3 tangent = normalize(cross(n, ax.xyz) + ax.xyz * 0.35);
    n = normalize(n + tangent * (u_hopper * 0.28 * riser * mask));
  }
  if (rough > 0.0) {
    float L = max(ax.w, 0.1);
    vec3 pLoc = (p - body(id, 3).xyz) / L;
    float frostAa = 1.0 / (1.0 + 65.0 * (footprint / L));
    vec3 fGrad = vec3(
      n3(pLoc * 14.0 + uid * 1.7) - 0.5,
      n3(pLoc * 14.0 + uid * 3.1 + 7.3) - 0.5,
      n3(pLoc * 14.0 + uid * 5.3 + 13.7) - 0.5
    );
    n = normalize(n + fGrad * (rough * 0.014 * frostAa));
  }
  return n;
}

// Evaluates a family of long, hair-thin (100:1+ aspect ratio), gently curved anti-aliased mineral scratches
// with per-scratch random angle and curvature so scratches never look like parallel ruled lines or alias into static.
float evalLongHairlineScratch(vec2 uv, vec3 lp, float angle, float stripFreq, float alongFreq, float densityThresh, float seed, float fwStrip, out vec2 outGradUV) {
  float ca = cos(angle), sa = sin(angle);
  vec2 uvRot = vec2(uv.x * ca + uv.y * sa, -uv.x * sa + uv.y * ca);
  float warp = (n3(lp * 3.4 + seed) - 0.5) * 0.055;
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
  float rawWidth = mix(0.020, 0.036, fract(h * 37.9));
  float effWidth = max(rawWidth, fwStrip * 1.35);
  float aaAtten = clamp(rawWidth / effWidth, 0.0, 1.0);
  float profile = exp(-(du * du) / (effWidth * effWidth)) * lenEnv * aaAtten;
  float pressure = smoothstep(0.32, 0.74, n3(lp * 8.5 + seed + stripId * 1.3));
  profile *= (0.35 + 0.65 * pressure) * mix(0.50, 1.0, fract(h * 53.3));
  float dProfDu = (-2.0 * du / effWidth) * profile * aaAtten;
  outGradUV = vec2(ca, sa) * dProfDu;
  return profile;
}

// Realistic mineral surface patina: 4 orientations of sparse, varied-angle hair-thin mineral scratches
// and faint growth-hillock etch lines. Keeps crystal faces 96%+ glassy and water-clear with zero pixel static!
void evalSurfacePatina(int id, int k, vec3 p, vec3 nBase, float footprint, float dEdge, out vec3 nPatina, out float patinaGlint, out float patinaRough) {
  nPatina = nBase;
  patinaGlint = 0.0;
  patinaRough = 0.0;
  float effPatina = clamp(u_patina * u_patinaMacro * u_surfaceMacro, 0.0, 2.0);
  if (effPatina <= 0.005) return;

  vec3 ax = body(id, 1).xyz;
  vec3 rt = body(id, 3).xyz;
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float uid = crystalUid(id);
  int kFace = (k < CP) ? k : (k >= CP + 5000 && k < CP + 10000 ? ((k - (CP + 5000)) / 36) : 0);
  vec3 nFlat = (kFace < CP) ? body(id, PA + kFace).xyz : nBase;

  vec3 tU = cross(nFlat, ax);
  float tULen = length(tU);
  tU = tULen > 1e-4 ? (tU / tULen) : normalize(cross(nFlat, vec3(0.577, -0.577, 0.577)));
  vec3 tV = cross(nFlat, tU);

  vec3 rel = p - rt;
  float scaleUV = 1.0 / max(r * 2.2, 0.14);
  float uFace = dot(rel, tU) * scaleUV;
  float vFace = dot(rel, tV) * scaleUV;
  vec2 uvFace = vec2(uFace, vFace);
  vec3 lp = vec3(uFace, dot(rel, ax) / L, vFace);

  float uvFootprint = max(footprint * scaleUV, 0.0012);
  float aa = 1.0 / (1.0 + 45.0 * uvFootprint);
  float fSeed = uid * 17.3 + float(kFace) * 11.7 + f_seed * 0.37;
  float edgeProx = 1.0 - smoothstep(0.0, r * 0.28, dEdge);
  float patchEnv = smoothstep(0.42, 0.80, n3(lp * 4.8 + fSeed) + 0.26 * edgeProx);

  float a1 = 0.38 + 0.55 * (hash(vec3(uid, float(kFace), 13.0)) - 0.5);
  float a2 = -0.72 + 0.55 * (hash(vec3(uid, float(kFace), 29.0)) - 0.5);
  float a3 = 1.15 + 0.45 * (hash(vec3(uid, float(kFace), 47.0)) - 0.5);
  float a4 = -1.35 + 0.45 * (hash(vec3(uid, float(kFace), 71.0)) - 0.5);

  vec2 g1 = vec2(0.0), g2 = vec2(0.0), g3 = vec2(0.0), g4 = vec2(0.0);
  float scr1 = evalLongHairlineScratch(uvFace, lp, a1, 9.5, 1.35, 0.82, fSeed + 3.1, uvFootprint * 9.5, g1) * patchEnv;
  float scr2 = evalLongHairlineScratch(uvFace + vec2(0.37, -0.21), lp, a2, 13.0, 1.65, 0.84, fSeed + 11.7, uvFootprint * 13.0, g2) * patchEnv;
  float scr3 = evalLongHairlineScratch(uvFace + vec2(-0.19, 0.43), lp, a3, 7.5, 1.10, 0.86, fSeed + 23.9, uvFootprint * 7.5, g3);
  float scr4 = evalLongHairlineScratch(uvFace + vec2(0.51, 0.17), lp, a4, 16.0, 1.85, 0.85, fSeed + 39.3, uvFootprint * 16.0, g4) * patchEnv;

  // Faint organic growth-hillock etch steps (anti-aliased by uvFootprint)
  float hillField = vFace * 11.0 + 3.8 * n3(lp * 4.2 + fSeed) + 1.4 * n3(lp * 9.5 + fSeed * 1.7);
  float hillWave = sin(hillField);
  float hillPatch = smoothstep(0.62, 0.88, n3(lp * 3.8 + fSeed + 15.1)) * ((kFace >= 8) ? 0.75 : 0.35) * aa;
  float hillock = pow(max(0.0, hillWave), 18.0) * hillPatch;
  float dHill = cos(hillField) * hillock;

  vec2 gradTot = (g1 + g2 + g3 * 1.20 + g4 * 0.80) * 0.022 + vec2(0.0, dHill * 0.012);
  vec3 dN = tU * gradTot.x + tV * gradTot.y;

  nPatina = normalize(nBase + dN * min(effPatina, 1.5));
  float totalScratch = clamp(scr1 * 0.85 + scr2 * 0.80 + scr3 * 1.10 + scr4 * 0.70 + hillock * 0.30, 0.0, 1.5);
  patinaGlint = totalScratch * effPatina * aa;
  patinaRough = clamp(totalScratch * 0.018 * effPatina * aa, 0.0, 0.03);
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
  if (k == CP || k == CP + 1) {
    int nextRow = occPartnerRow(id, k - CP);
    vec4 rawEc = body(nextRow, k - CP);
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
    float cosAng = dot(pl.xyz, nk);
    if (cosAng > 0.9995) continue;
    float sinAng = sqrt(max(1e-3, 1.0 - cosAng * cosAng));
    float d = (pl.w - dot(pl.xyz, p)) / sinAng;
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

// Softens & tints bright studio softbox glare on crystal faces (governed by u_faceGlare)
// so a lit facet never blows out to solid opaque white and internal crystal details stay visible.
vec3 softFacetRefl(vec3 rawEnv) {
  vec3 gemCol = exp(-f_sigma * 0.42);
  float lum = max(rawEnv.r, max(rawEnv.g, rawEnv.b));
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  vec3 compressed = rawEnv / (1.0 + (2.1 - 1.2 * glare) * lum);
  return mix(compressed * (0.18 + 0.82 * gemCol), compressed, 0.28 * glare);
}

vec3 refractedEnv(vec3 dir, float rough, float refAlpha) {
  vec3 d = f_toWorld * dir;
  vec3 bg = backdrop(d);
  vec3 lt = lights(d, rough);
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  float ltLum = max(lt.r, max(lt.g, lt.b));
  vec3 gemCol = exp(-f_sigma * 0.48);
  vec3 softLt = (lt / (1.0 + (2.6 - 1.5 * glare) * ltLum)) * mix(gemCol * 0.88, gemCol * 0.55 + vec3(0.45), 0.25 * glare);
  return bg + mix(lt, softLt, refAlpha);
}

vec3 facetIridescence(int id, int k, vec3 p, vec3 rd, vec3 n, float dEdge, vec3 baseRefl) {
  float s = u_iridescence * u_rainbow;
  if (s <= 0.0) return baseRefl;
  float cosV = clamp(abs(dot(rd, n)), 0.0, 1.0);
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float hgt = dot(p - body(id, 3).xyz, body(id, 1).xyz) / L;
  float genShift = body(id, 2).y;
  float filmT = cosV * 2.4 + hgt * 1.35 + (dEdge / r) * 0.95 + float(k) * 0.19 + genShift;
  vec3 irid1 = spectralRainbow(filmT);
  vec3 irid2 = spectralRainbow(filmT * 1.62 + 0.29);
  vec3 specCol = normalize(mix(irid1, irid2, 0.32) + vec3(1e-4)) * 1.35;
  float fresRim = pow(1.0 - cosV, 2.0);
  vec3 tinted = mix(baseRefl, baseRefl * specCol * 1.85, clamp(s * 0.9, 0.0, 1.0));
  vec3 sheen = specCol * f_ambient * (0.32 + 0.68 * fresRim) * min(s, 3.0) * 0.5;
  return tinted + sheen;
}

vec3 iridescentFilm(int id, int k, vec3 p, vec3 rd, vec3 n, float dEdge) {
  float s = u_iridescence * u_rainbow;
  if (s <= 0.0) return vec3(0.0);
  float cosV = clamp(abs(dot(rd, n)), 0.0, 1.0);
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float hgt = dot(p - body(id, 3).xyz, body(id, 1).xyz) / L;
  float filmT = cosV * 2.4 + hgt * 1.35 + (dEdge / r) * 0.95 + float(k) * 0.19 + body(id, 2).y;
  vec3 film = spectralRainbow(filmT * 1.18 + 0.11);
  float rim = pow(1.0 - cosV, 1.5);
  float coat = 0.6 + 0.4 * hash(vec3(body(id, 2).z * 3.0, float(k) * 7.0, 61.0));
  return film * f_ambient * (0.3 + 0.7 * rim) * coat * min(s, 1.5) * 0.4;
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
  vec3 gemTint = mix(exp(-f_sigma * 0.38), vec3(1.0), 0.32 * glare);

  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 lDir = normalize(vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]));
    vec3 lCol = vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
    float cSmooth = max(0.0, dot(rSmooth, lDir));
    float fGlint = pow(cSmooth, expFacet) * (1.0 + 0.8 * apexMask);
    flashSum += lCol * gemTint * (fGlint * 0.28 * glare);
  }

  // 6 pinpoint studio catch-lights around the orbit ring
  for (int j = 0; j < 6; j++) {
    float az = float(j) * (TAU / 6.0) + 0.41;
    float el = (j & 1) == 0 ? 0.36 : 0.68;
    vec3 spotDir = vec3(sin(az) * cos(el), sin(el), cos(az) * cos(el));
    float sSmooth = max(0.0, dot(rSmooth, spotDir));
    float facetHit = pow(sSmooth, expFacet * 1.2) * (1.0 + 0.8 * apexMask);
    vec3 prismTint = mix(gemTint, spectralRainbow(float(k) * 0.21 + float(j) * 0.17 + body(id, 2).y), clamp((0.25 + u_iridescence * 0.3) * u_rainbow, 0.0, 0.85));
    flashSum += prismTint * (facetHit * 1.15 * glare);
  }

  float overdrive = u_bloom * (0.55 + 0.45 * u_bloom) * (0.22 + 0.78 * glare);
  vec3 rawGlint = flashSum * polish * overdrive;
  return rawGlint / (1.0 + (1.6 - 0.85 * glare) * max(rawGlint.r, max(rawGlint.g, rawGlint.b)));
}

// Secondary continuation ray for GPU mirrored backgrounds (single-pass, zero nested hop loop)
vec3 shadeBehindRay(vec3 roIn, vec3 rdIn, int skipIn, float travIn, float jitter, out float firstHitMask) {
  firstHitMask = 0.0;
  int bId, bK; float bT, bTf;
  if (!traceScene(roIn, rdIn, skipIn, bId, bT, bTf, bK)) {
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
  float bUid = crystalUid(bId);
  float bFrostRoll = hash(vec3(bUid, float(bK), f_seed + 0.5));
  float bRough = u_frost * smoothstep(bFrostRoll - 0.15, bFrostRoll + 0.15, u_frostShare);
  vec3 bRawN = (bK < CP) ? body(bId, PA + bK).xyz : normalize(bP - body(bId, 0).xyz);
  vec3 bN = faceNormal(bId, bK < CP ? bK : 18, bRawN, bP, bRough, travB * f_pixAngle, 1.0);
  float bF = fresnel(-dot(rdIn, bN), 1.0 / u_ior);
  vec3 bEntryRefl = facetIridescence(bId, bK, bP, rdIn, bN, 1.0, softFacetRefl(env(reflect(rdIn, bN), bRough)));
  float bIridBoost = clamp(bF + (u_iridescence * u_rainbow) * 0.06, 0.0, 0.42) * (0.15 + 0.85 * glare);
  vec3 colB = bIridBoost * bEntryRefl;
  vec3 thrB = vec3(1.0 - bF * (0.3 + 0.7 * glare));
  vec3 bRefrIn = refract(rdIn, bN, 1.0 / u_ior);
  vec3 rdB = dot(bRefrIn, bRefrIn) > 1e-6 ? normalize(bRefrIn) : rdIn;
  int bKf;
  float bLen = exitCrystal(bId, bP, rdB, bKf);
  vec3 bTrans;
  vec3 bInScat = insideVolume(bId, bP, rdB, bLen, jitter, clamp(int(u_inclusionSteps + 0.5), 2, 4), bTrans);
  colB += thrB * bInScat;
  thrB *= bTrans;
  travB += bLen;
  bP += rdB * bLen;
  float bExitRoll = hash(vec3(bUid, float(bKf), f_seed + 0.5));
  float bFr = u_frost * smoothstep(bExitRoll - 0.15, bExitRoll + 0.15, u_frostShare);
  vec3 bNf = faceNormal(bId, bKf, body(bId, PA + bKf).xyz, bP, bFr, travB * f_pixAngle, 1.0);
  float bFo = fresnel(dot(rdB, bNf), u_ior);
  vec3 bExitRefl = facetIridescence(bId, bKf, bP, rdB, -bNf, 1.0, softFacetRefl(env(reflect(rdB, -bNf), bFr)));
  colB += thrB * (bFo * bExitRefl * 0.42 * (0.15 + 0.85 * glare));
  thrB *= 1.0 - bFo * (0.3 + 0.7 * glare);
  vec3 bOut = refract(rdB, -bNf, u_ior);
  rdB = dot(bOut, bOut) > 1e-6 ? normalize(bOut) : reflect(rdB, -bNf);
  vec3 endEnv = refractedEnv(rdB, bFr, 1.0);
  float deflAng = acos(clamp(dot(rdB, rdIn), -1.0, 1.0));
  float glowG = exp(-pow(deflAng / radians(u_glowSize * 1.15), 2.0));
  vec3 axisBg = mix(lin(u_studioEdge) * 0.14, f_glow * 1.65 + f_ambient * 0.22, glowG);
  endEnv = max(endEnv, axisBg);
  colB += thrB * endEnv;
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
  float lastRough = 0.0;
  float lastFireAlpha = 0.0;
  int depth = int(u_depth + 0.5), bounces = int(u_bounces + 0.5);
  float hopCost = 0.0;

  int maxHops = clamp(depth + 1, 1, 5);
  bool primaryEvaluated = false;

  for (int hop = 0; hop < maxHops; hop++) {
    if (hopCost >= float(depth) || dot(thr, thr) < 1e-6) break;
    int id, k; float t, tfHit;
    if (!traceScene(ro, rd, skipMask, id, t, tfHit, k)) break;

    float effEdgeMacro = max(0.0, u_patinaMacro * u_edgeMacro);
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
    vec3 voidWallN = vec3(0.0, 1.0, 0.0);
    float voidRim = 0.0;
    if (!primaryEvaluated && u_occlusion > 0.001) {
      float occBlend = occOpacityFor(id) * smoothstep(0.6, 1.0, rawAlpha);
      float voidCut = evalMassiveOcclusionVoid(id, ro, rd, t, voidWallN, voidRim);
      rawAlpha *= (1.0 - min(0.995, occBlend * voidCut));
      voidRim *= occBlend;
    }
    primaryEvaluated = true;

    float budgetWeight = clamp(float(depth) - hopCost, 0.0, 1.0);
    float alpha = rawAlpha * rawAlpha * (3.0 - 2.0 * rawAlpha) * budgetWeight;
    if (alpha <= 2e-3) {
      ro = pEntry;
      continue;
    }

    float effIor = ior;
    float uid = crystalUid(id);
    hopCost += alpha;

    float footprint = travelled * f_pixAngle;
    int kFrostId = (k < CP) ? k : ((k >= CP + 10000) ? ((k - (CP + 10000)) / 4) : ((k >= CP + 5000) ? ((k - (CP + 5000)) / 2) : k));
    float frostRoll = hash(vec3(uid, float(kFrostId), f_seed + 0.5));
    float frostW = smoothstep(frostRoll - 0.15, frostRoll + 0.15, u_frostShare);
    float rough = u_frost * frostW;
    bool isTinyEdgeChip = (k >= CP + 5000 && k < CP + 10000);
    bool isCornerVertexChip = (k >= CP + 10000);
    bool isOcclusionVoid = (k >= CP && k < CP + 5000);
    bool isEdgeIncursion = isTinyEdgeChip || isCornerVertexChip || isOcclusionVoid;
    bool isCornerNip = (k >= 18 && k < CP);
    vec3 rawFaceN = (isTinyEdgeChip || isCornerVertexChip) ? chipFacetN : body(id, PA + k).xyz;

    vec3 adjN = rawFaceN;
    int ridgeKLo = -1, ridgeKHi = -1;
    float dEdge = ((hop == 0) && (u_patina > 0.01 || u_occlusion > 0.01 || u_hopper > 0.0 || u_iridescence > 0.0 || u_bloom > 0.0)) ? edgeDist(id, k, p, adjN, ridgeKLo, ridgeKHi) : 1.0;

    // Pure 3D CSG refractive crystal facet shading (zero artificial white edge paint or cylindrical edge normal rounding)
    if (isTinyEdgeChip || isCornerVertexChip || isCornerNip) {
      rough = max(rough, 0.025 * clamp(u_edgeWear * effEdgeMacro, 0.0, 1.0));
    }
    vec3 n = faceNormal(id, isEdgeIncursion ? 18 : k, rawFaceN, p, rough, footprint, dEdge);
    if (voidRim > 0.001) {
      n = normalize(mix(n, voidWallN, voidRim * 0.65));
      rough = max(rough, 0.18 * voidRim);
    }
    if (hop == 0 && (isTinyEdgeChip || isCornerVertexChip || isCornerNip)) {
      vec3 axis = body(id, 1).xyz, root = body(id, 3).xyz;
      float L = max(body(id, 1).w, 0.12);
      vec3 p0 = body(id, PA).xyz;
      vec3 t1 = p0 - axis * dot(p0, axis);
      float t1Len = length(t1);
      t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
      vec3 t2 = cross(axis, t1);
      vec3 rel = p - root;
      vec3 lp = vec3(dot(rel, t1), dot(rel, axis), dot(rel, t2)) / L;
      float nDamp = 1.0 / (1.0 + 40.0 * (footprint / L));
      float conchRing = sin(length(lp) * 18.0 + dot(lp, rawFaceN) * 28.0) * nDamp;
      vec3 ringTan = normalize(cross(rawFaceN, axis + vec3(0.01, 0.0, 0.02)));
      n = normalize(n + ringTan * (0.022 * conchRing));
    }

    vec3 nRefl = n;
    float patinaGlint = 0.0, patinaRough = 0.0;
    if (hop == 0) {
      evalSurfacePatina(id, k, p, n, footprint, dEdge, nRefl, patinaGlint, patinaRough);
      // Keep internal refraction 100% pure & clear by only applying a tiny fraction (6%) of surface scratch normal to n
      n = normalize(mix(n, nRefl, 0.06));
    }
    float reflRough = max(rough, patinaRough);

    vec3 hopCol = vec3(0.0);
    vec3 hopThr = vec3(1.0);
    bool isChippedFacet = isTinyEdgeChip || isCornerVertexChip || isCornerNip;
    float effGlare = isOcclusionVoid ? (glare * 0.45) : (isChippedFacet ? (glare * 0.65) : glare);
    float cosEntry = max(0.32, -dot(rd, nRefl));
    float F = fresnel(cosEntry, 1.0 / effIor);
    vec3 rawEntryRefl = softFacetRefl(env(reflect(rd, nRefl), reflRough));
    if (isChippedFacet) {
      rawEntryRefl *= mix(vec3(1.0), exp(-f_sigma * 0.32), 0.68);
    }
    vec3 entryRefl = facetIridescence(id, k, p, rd, nRefl, dEdge, rawEntryRefl);
    vec3 entryFlash = (hop == 0) ? crystalBloomGlint(id, k, p, rd, nRefl, dEdge, reflRough) : vec3(0.0);
    float iridReflBoost = clamp(F + (u_iridescence * u_rainbow) * 0.06 + patinaGlint * 0.11, 0.0, 0.58) * (0.18 + 0.82 * effGlare);
    vec3 patinaLight = (0.48 * f_ambient + 0.52 * entryRefl + vec3(0.10)) * vec3(0.96, 0.98, 1.0);
    hopCol += (iridReflBoost * entryRefl + patinaGlint * 0.14 * patinaLight + entryFlash);
    hopCol += iridescentFilm(id, k, p, rd, nRefl, dEdge);
    hopThr *= (1.0 - F * (0.3 + 0.7 * effGlare));

    vec3 rdC = rd;
    vec3 refrIn = refract(rdC, n, 1.0 / effIor);
    rdC = dot(refrIn, refrIn) > 1e-6 ? normalize(refrIn) : rdC;
    vec3 roC = p;
    float travC = travelled;
    int kf = 0;
    float fr = 0.0;
    vec3 nf = n;
    vec3 o = rdC;

    float len0 = exitCrystal(id, roC, rdC, kf);
    vec3 trans0;
    int steps0 = (hop == 0) ? clamp(int(u_inclusionSteps + 0.5), 2, 12) : clamp(int(u_inclusionSteps + 0.5), 2, 4);
    vec3 inScat0 = insideVolume(id, roC, rdC, len0, jitter, steps0, trans0);
    if (hop == 0) {
      insidePrimaryExtras(id, roC, rdC, len0, travC, inScat0, trans0);
    }
    hopCol += hopThr * inScat0;
    hopThr *= trans0;
    travC += len0;
    p = roC + rdC * len0;
    float exitFrostRoll = hash(vec3(uid, float(kf), f_seed + 0.5));
    fr = u_frost * smoothstep(exitFrostRoll - 0.15, exitFrostRoll + 0.15, u_frostShare);
    nf = faceNormal(id, kf, body(id, PA + kf).xyz, p, fr, travC * f_pixAngle, 1.0);
    o = refract(rdC, -nf, effIor);

    int maxB = clamp(hop == 0 ? bounces : 2, 1, 6);
    for (int b = 1; b < maxB; b++) {
      if (dot(o, o) >= 1e-6) break;
      float Fb = fresnel(dot(rdC, nf), effIor);
      vec3 bounceRefl = facetIridescence(id, kf, p, rdC, -nf, 1.0, softFacetRefl(env(reflect(rdC, -nf), fr)));
      hopCol += hopThr * (Fb * bounceRefl * 0.28 * (0.15 + 0.85 * glare));
      rdC = reflect(rdC, -nf);
      roC = p;
      float len = exitCrystal(id, roC, rdC, kf);
      vec3 trans;
      vec3 inScat = insideVolume(id, roC, rdC, len, jitter, clamp(int(u_inclusionSteps + 0.5), 2, 4), trans);
      hopCol += hopThr * inScat;
      hopThr *= trans;
      travC += len;
      p = roC + rdC * len;
      exitFrostRoll = hash(vec3(uid, float(kf), f_seed + 0.5));
      fr = u_frost * smoothstep(exitFrostRoll - 0.15, exitFrostRoll + 0.15, u_frostShare);
      nf = faceNormal(id, kf, body(id, PA + kf).xyz, p, fr, travC * f_pixAngle, 1.0);
      o = refract(rdC, -nf, effIor);
    }
    if (dot(o, o) < 1e-6) o = rdC;
    vec3 exitAdjN;
    float dExit = (hop == 0 && (u_hopper > 0.0 || (u_iridescence > 0.0 && u_rainbow > 0.0) || u_bloom > 0.0)) ? edgeDist(id, kf, p, exitAdjN) : 1.0;
    float Fo = fresnel(dot(rdC, nf), effIor);
    vec3 exitRefl = facetIridescence(id, kf, p, rdC, -nf, dExit, softFacetRefl(env(reflect(rdC, -nf), fr)));
    vec3 exitFlash = (hop == 0) ? crystalBloomGlint(id, kf, p, rdC, -nf, dExit, fr) * 0.4 : vec3(0.0);
    hopCol += hopThr * (Fo * exitRefl * 0.42 * (0.15 + 0.85 * glare) + exitFlash);
    hopThr *= 1.0 - Fo * (0.3 + 0.7 * glare);

    if (alpha < 0.999) {
      // Partial transparency / occlusion void feather: blend refracted crystal contribution and continue straight behind!
      vec3 cExitDir = normalize(o);
      vec3 cEnv = refractedEnv(cExitDir, fr, 1.0);
      col += thr * alpha * (hopCol + hopThr * cEnv);
      thr *= (1.0 - alpha);
      ro = pEntry;
      continue;
    }

    col += thr * hopCol;
    thr *= hopThr;
    lastIn = rdC;
    lastN = nf;
    lastRough = fr;
    lastFireAlpha = 1.0;
    lastRefracted = true;
    rd = normalize(o);
    ro = p;
    travelled = travC;
  }

  if (dot(thr, thr) > 0.0) {
    float effIrid = u_iridescence * u_rainbow;
    float effFire = u_fire + effIrid * 0.075;
    if (lastRefracted && effFire > 0.005 && lastFireAlpha > 1e-4) {
      int ns = clamp(int(u_fireSamples + 0.5), 2, 9);
      vec3 acc = vec3(0.0), wsum = vec3(0.0);
      float dispIor = mix(1.0, ior, lastFireAlpha);
      float dispFire = effFire * lastFireAlpha;
      for (int i = 0; i < ns; i++) {
        float x = (float(i) + 0.5) / float(ns);
        vec3 dw = (x - vec3(0.12, 0.5, 0.88)) / 0.38;
        vec3 w = max(1.0 - dw * dw, 0.0);
        vec3 o = refract(lastIn, -lastN, dispIor + dispFire * (x - 0.5) * 2.0);
        vec3 sampleDir = dot(o, o) > 1e-6 ? normalize(mix(rd, o, lastFireAlpha)) : rd;
        wsum += w;
        vec3 eCol = refractedEnv(sampleDir, lastRough, lastFireAlpha);
        if (effIrid > 0.0) {
          float defl = length(sampleDir - initialRd);
          vec3 prism = spectralRainbow(defl * 2.4 + x * 0.85);
          eCol += prism * f_ambient * effIrid * 0.45 * lastFireAlpha * smoothstep(0.04, 0.45, defl);
        }
        acc += w * eCol;
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
  float thresh = max(0.72, u_bloomThreshold);
  vec3 hdrBloom = c * max(l - thresh, 0.0) / max(l, 1e-4);
  float bgLum = max(u_studioEdge.r, max(u_studioEdge.g, u_studioEdge.b));
  float crystalLum = max(0.0, l - bgLum * 1.15);
  vec3 softKneeGlow = smoothstep(0.22, 0.85, crystalLum) * c * (0.45 * u_rainbowGlow + 0.35 * max(0.0, u_bloom - 0.2));
  outColor = encodeHdr(hdrBloom + softKneeGlow);
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
  float fxaaStrength = clamp(u_fxaa, 0.0, 1.0);
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
  float effBloom = u_bloom * 0.85 + u_rainbowGlow * 0.55;
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
  vec3 haloCol = (bloomCol * effBloom) / (1.0 + 0.35 * bloomLuma + 1.8 * baseLuma * baseLuma);
  vec3 c = baseCol + haloCol;
  c = pow(aces(c), vec3(1.0 / 2.2));
  vec2 q = (fc - 0.5 * u_res) / u_res;
  c = mix(c, u_studioEdge, clamp(u_vignette * dot(q, q) * 2.6, 0.0, 1.0));
  c = filmDust(c, fc, f_px, f_frame, u_dust);
  c = filmGrain(c, fc, f_px, f_frame, u_grainAmount, u_grainSize, u_grainSharpness, u_grainClump);
  c += (hash(vec3(fc, f_frame)) - 0.5) / 255.0;
  outColor = vec4(c, 1.0);
}`,
};
