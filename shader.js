// Crystals — WebGL2 ray-tracing shaders + per-frame CPU values.
// Built on Jack Bush's Geode shader (jackbush/glimmers), extended with:
// - Direct interactive orbit camera (azimuth + elevation + zoom)
// - Bi-colour core-to-rim / base-to-tip absorption zoning (f_sigmaCore + u_biColour)
// - Multi-generation nested phantom growth shells (u_phantomLayers)
// - Stepped skeletal hopper crystal terraces (u_hopper)

import { FILM_GLSL } from './kit/film.js';
import { cluster, CRYSTAL_PLANES, MAX_CRYSTALS, PLANES_AT, ROW_W, ROWS } from './cluster.js';

const INCL = 128;          // inclusion volume edge, texels
const LIGHTS = 4;          // key, fill, rim left, rim right
const LIGHT_STRIDE = 14;   // centre 3, u 3, v 3, colour 3, half width, half height
const DEG = Math.PI / 180;
const TAU = Math.PI * 2;

const lin = (c) => Math.pow(c, 2.2);

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
    f_sigma: 3,
    f_sigmaCore: 3,
  },

  perFrame(P, t, rt, width, height, px, f) {
    f.f_px[0] = px;
    f.f_frame[0] = Math.floor(rt * P.filmFps.value);
    f.f_seed[0] = P.seed.value;
    f.f_crystals[0] = cluster.crystals;

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
    box[0] = Math.max(0, Math.floor(width / 2) - reach);
    box[1] = Math.max(0, Math.floor(height / 2) - reach);
    box[2] = Math.min(width, Math.floor(width / 2) + reach) - box[0];
    box[3] = Math.min(height, Math.floor(height / 2) + reach) - box[1];
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
    for (let c = 0; c < 3; c++) {
      f.f_glow[c] = lin(hslChannel(c === 0 ? 0 : c === 1 ? 8 : 4, h, P.backdropSat.value, 0.5)) * P.backdropLight.value;
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
  vec3 raw = wave * wave * (3.0 - 2.0 * wave);
  return mix(vec3(0.95, 0.96, 1.0), raw, 0.78);
}
vec4 encodeHdr(vec3 c) { return vec4(c / (1.0 + c), 1.0); }
vec3 decodeHdr(vec4 e) { return e.rgb / max(1.0 - e.rgb, 1e-4); }
`,

  bake: `
#define INCL ${INCL}.0
uniform float u_slice;
out vec4 outColor;

vec3 grad(vec3 c, float period) {
  uvec3 u = uvec3(ivec3(mod(c, period)) + ivec3(int(f_seed) * 131 + 7, int(f_seed) * 71 + 3, int(f_seed) * 37 + 11));
  uint h = pcg(u.x + pcg(u.y + pcg(u.z)));
  return vec3(float(h & 1023u), float((h >> 10) & 1023u), float((h >> 20) & 1023u)) / 511.5 - 1.0;
}
float pnoise(vec3 p, float period) {
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n000 = dot(grad(i, period), f);
  float n100 = dot(grad(i + vec3(1, 0, 0), period), f - vec3(1, 0, 0));
  float n010 = dot(grad(i + vec3(0, 1, 0), period), f - vec3(0, 1, 0));
  float n110 = dot(grad(i + vec3(1, 1, 0), period), f - vec3(1, 1, 0));
  float n001 = dot(grad(i + vec3(0, 0, 1), period), f - vec3(0, 0, 1));
  float n101 = dot(grad(i + vec3(1, 0, 1), period), f - vec3(1, 0, 1));
  float n011 = dot(grad(i + vec3(0, 1, 1), period), f - vec3(0, 1, 1));
  float n111 = dot(grad(i + vec3(1, 1, 1), period), f - vec3(1, 1, 1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z) * 1.6;
}
float pfbm(vec3 p, float period, int octaves) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 6; i++) {
    if (i >= octaves) break;
    s += a * pnoise(p, period);
    p *= 2.0; period *= 2.0; a *= 0.5;
  }
  return s;
}

void main() {
  vec3 p = vec3(gl_FragCoord.xy, u_slice + 0.5) / INCL;
  float m = pfbm(p * 4.0, 4.0, 5);
  float milk = pow(clamp((m + 0.05) * 1.6, 0.0, 1.0), 1.8);
  vec3 w = vec3(pfbm(p * 2.0 + 3.1, 2.0, 2), pfbm(p * 2.0 + 7.7, 2.0, 2), pfbm(p * 2.0 + 1.3, 2.0, 2));
  float sheet = 1.0 - smoothstep(0.0, 0.045, abs(pnoise(p * 3.0 + w * 1.2, 3.0)));
  float patchy = smoothstep(-0.05, 0.25, pfbm(p * 2.0 + 9.0, 2.0, 2));
  float bubbles = mix(0.25, 1.0, step(0.6, hash(vec3(gl_FragCoord.xy, u_slice) + f_seed * 977.0)));
  float veil = sheet * patchy * bubbles;
  float mottle = clamp(pfbm(p * 2.0 + 5.0, 2.0, 3) * 0.8 + 0.5, 0.0, 1.0);
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

// 1. Large Architectural Occlusion Voids (u_occlusion): invisible 3D displacing wedges/spheres that carve large concave gaps into crystals
bool getOcclusionVoidPlanes(int row, int ka, int kb, float slotId, out vec4 mUp, out vec4 mDn) {
  mUp = vec4(0.0); mDn = vec4(0.0);
  if (u_occlusion <= 0.01 || ka >= 17 || kb >= 17 || ka == kb) return false;
  if (slotId < 0.5 || slotId > 3.5) return false;

  vec4 plA = body(row, PA + ka);
  vec4 plB = body(row, PA + kb);
  vec3 sumN = plA.xyz + plB.xyz;
  float lenN = length(sumN);
  if (lenN < 0.45 || lenN > 1.98) return false;
  vec3 nAB = sumN / lenN;
  float wAB = (plA.w + plB.w) / lenN;

  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float L = max(body(row, 1).w, 0.08);
  float r = max(body(row, 2).w, 0.02);
  float uid = max(1.0, body(row, 2).z);
  float nodePresence = clamp(body(row, 2).x, 0.0, 1.0);
  float rA = plA.w - dot(plA.xyz, rt);
  float rB = plB.w - dot(plB.xyz, rt);
  float symWeight = (1.0 - smoothstep(r * 1.45, r * 2.15, rA)) * (1.0 - smoothstep(r * 1.45, r * 2.15, rB));
  if (symWeight <= 1e-4) return false;

  int kLo = min(ka, kb), kHi = max(ka, kb);
  float roll = hash(vec3(uid * 3.0 + float(kLo) * 7.0 + float(kHi), slotId + 13.0, 5.0));
  float voidW = smoothstep(0.58, 0.74, roll) * clamp(u_occlusion * 1.5, 0.0, 1.0) * nodePresence * symWeight;
  if (voidW <= 1e-4) return false;

  float h0 = ((slotId + 0.5 + (hash(vec3(slotId, uid, float(kLo + kHi))) - 0.5) * 0.28) / 5.0) * L;
  float d0 = r * (0.18 + 0.42 * u_occlusion) * voidW;
  float sUp = 0.55 + 0.35 * fract(roll * 11.3);
  float sDn = 0.55 + 0.35 * fract(roll * 19.7);
  float asym = (hash(vec3(uid, slotId, 19.0)) - 0.5) * 0.18;
  vec3 diffVec = plA.xyz - plB.xyz;
  float diffLen = max(length(diffVec), 1e-4);
  vec3 diffN = diffVec / diffLen;
  float wDiff = (plA.w - plB.w) / diffLen;

  float axH0 = dot(ax, rt) + h0;
  vec3 vUp = nAB - ax * sUp + diffN * asym;
  float wUp = (wAB - d0) - sUp * axH0 + asym * wDiff;
  vec3 vDn = nAB + ax * sDn + diffN * asym;
  float wDn = (wAB - d0) + sDn * axH0 + asym * wDiff;
  mUp = vec4(vUp, wUp);
  mDn = vec4(vDn, wDn);
  return true;
}

bool applyOcclusionVoid(int row, int ka, int kb, vec3 ro, vec3 rd, inout float tn, inout float tf, inout int kn) {
  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float L = max(body(row, 1).w, 0.08);
  float tStart = max(tn, 0.0);
  vec3 pStart = ro + rd * tStart;
  float hNorm = dot(pStart - rt, ax) / L;
  float slotId = floor(clamp(hNorm, 0.0, 0.999) * 5.0);
  vec4 mUp, mDn;
  if (!getOcclusionVoidPlanes(row, ka, kb, slotId, mUp, mDn)) return true;

  float dnU = dot(mUp.xyz, rd), dnD = dot(mDn.xyz, rd);
  float valU = dot(mUp.xyz, pStart) - mUp.w;
  float valD = dot(mDn.xyz, pStart) - mDn.w;
  if (valU > 0.0 && valD > 0.0) {
    float tU = (dnU < -1e-6) ? (tStart - valU / dnU) : 1e9;
    float tD = (dnD < -1e-6) ? (tStart - valD / dnD) : 1e9;
    float tc1 = min(tU, tD);
    if (tc1 >= tf) return false;
    int wallExit = (tU < tD) ? 0 : 1;
    tn = tc1;
    kn = CP + (ka * 18 + kb) * 10 + int(slotId + 0.5) * 2 + wallExit;
  }

  vec3 pExit = ro + rd * tf;
  float valUe = dot(mUp.xyz, pExit) - mUp.w;
  float valDe = dot(mDn.xyz, pExit) - mDn.w;
  if (valUe > 0.0 && valDe > 0.0) {
    float tUb = (dnU > 1e-6) ? (tf - valUe / dnU) : -1e9;
    float tDb = (dnD > 1e-6) ? (tf - valDe / dnD) : -1e9;
    float tc0 = max(tUb, tDb);
    if (tc0 <= max(tn, 0.0)) return false;
    tf = tc0;
  }
  return tf > max(tn, 0.0);
}

// 2. Tiny Crystal-Locked 3D Edge Chips (u_edgeChips): ~10x smaller nicks strictly locked to crystal edges (d0, d1, alongEdge)
bool applyTinyEdgeChip(int row, int ka, int kb, vec3 ro, vec3 rd, inout float tn, inout float tf, inout int kn) {
  if (u_edgeChips <= 0.01 || ka >= 16 || kb >= 16 || ka == kb) return true;
  vec4 plA = body(row, PA + ka);
  vec4 plB = body(row, PA + kb);
  vec3 sumN = plA.xyz + plB.xyz;
  float lenN = length(sumN);
  if (lenN < 0.45 || lenN > 1.98) return true;
  vec3 nAB = sumN / lenN;
  float wAB = (plA.w + plB.w) / lenN;

  vec3 ax = body(row, 1).xyz;
  vec3 rt = body(row, 3).xyz;
  float L = max(body(row, 1).w, 0.08);
  float r = max(body(row, 2).w, 0.02);
  float uid = max(1.0, body(row, 2).z);
  float nodePresence = clamp(body(row, 2).x, 0.0, 1.0);
  float rA = plA.w - dot(plA.xyz, rt);
  float rB = plB.w - dot(plB.xyz, rt);
  float symWeight = (1.0 - smoothstep(r * 1.45, r * 2.15, rA)) * (1.0 - smoothstep(r * 1.45, r * 2.15, rB));
  if (symWeight <= 1e-4) return true;

  float tStart = max(tn, 0.0);
  vec3 pStart = ro + rd * tStart;
  float dRidge = wAB - dot(nAB, pStart);
  float maxChipCut = r * (0.010 + 0.022 * u_edgeChips) * nodePresence * symWeight;
  if (dRidge >= maxChipCut) return true;

  vec3 edgeDir = cross(plA.xyz, plB.xyz);
  float edLen = length(edgeDir);
  if (edLen < 1e-4) return true;
  edgeDir /= edLen;
  if (dot(edgeDir, ax) < 0.0) edgeDir = -edgeDir;

  float alongEdge = dot(pStart - rt, edgeDir) / L;
  if (alongEdge < 0.08 || alongEdge > 0.92) return true;
  float slotCoord = alongEdge * 18.0;
  float slotId = floor(slotCoord);
  float localU = fract(slotCoord) - 0.5;

  int kLo = min(ka, kb), kHi = max(ka, kb);
  float h = hash(vec3(uid * 5.0 + float(kLo) * 19.0 + float(kHi), slotId + 17.0, 11.0));
  // Occasional: only ~16% of the 18 edge slots have a tiny 3D chip
  if (h <= 0.84) return true;
  float notchProfile = max(0.0, 1.0 - (localU * localU) * 7.5);
  float chipCut = maxChipCut * (0.55 + 0.45 * fract(h * 13.7)) * smoothstep(0.84, 0.89, h) * notchProfile;
  if (dRidge >= chipCut) return true;

  float dnAB = dot(nAB, rd);
  if (dnAB < -0.02) {
    float tChip = tStart + (chipCut - dRidge) / (-dnAB);
    if (tChip >= tf) return false;
    tn = tChip;
    int wallBit = localU >= 0.0 ? 0 : 1;
    kn = CP + 5000 + (kLo * 18 + kHi) * 2 + wallBit;
  } else {
    return false;
  }
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
  if (tf <= max(tn, 0.0)) return false;
  if ((u_edgeChips > 0.01 || u_occlusion > 0.01) && kn < 16) {
    float r = max(body(row, 2).w, 0.02);
    vec3 pStart = ro + rd * max(tn, 0.0);
    vec3 nk = body(row, PA + kn).xyz;
    int kAdj = -1;
    float dAdjMin = 1e9;
    int kBase = (kn < 8) ? 0 : 8;
    for (int j = 0; j < 8; j++) {
      int kCand = kBase + j;
      if (kCand == kn) continue;
      vec4 plJ = body(row, PA + kCand);
      float cAng = dot(plJ.xyz, nk);
      if (cAng > 0.96 || cAng < 0.1) continue;
      float dJ = plJ.w - dot(plJ.xyz, pStart);
      if (dJ < dAdjMin) { dAdjMin = dJ; kAdj = kCand; }
    }
    if (kAdj >= 0) {
      if (u_occlusion > 0.01 && dAdjMin < r * 0.55) {
        if (!applyOcclusionVoid(row, kn, kAdj, ro, rd, tn, tf, kn)) return false;
      }
      if (u_edgeChips > 0.01 && kn < 16 && dAdjMin < r * 0.048) {
        if (!applyTinyEdgeChip(row, kn, kAdj, ro, rd, tn, tf, kn)) return false;
      }
    }
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

bool traceScene(vec3 ro, vec3 rd, int skipMask, out int id, out float tHit, out int kHit) {
  bool hit = false; id = 0; kHit = 0; tHit = 1e9;
  float tn, tf; int kn, kf;
  int nc = int(f_crystals + 0.5);
  for (int i = 0; i < MAX_CRYSTALS; i++) {
    if (i >= nc) break;
    if ((skipMask & (1 << i)) != 0 || !sphereHit(ro, rd, body(i, 0), tHit)) continue;
    if (convexHit(i, ro, rd, tn, tf, kn, kf)) {
      float tCand = max(tn, 0.0);
      if (tf > tCand + 1e-4 && tCand < tHit) { tHit = tCand; id = i; kHit = kn; hit = true; }
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

vec3 inside(int id, vec3 ro, vec3 rd, float len, float travelled, float jitter, int bounce, out vec3 trans) {
  vec4 ax = body(id, 1), side = body(id, 2), rt = body(id, 3);
  vec3 axis = ax.xyz, root = rt.xyz;
  float L = ax.w, r = side.w, rnd = rt.w;
  float uid = crystalUid(id);
  float hueOff = side.y;
  vec3 off = vec3(rnd * 17.3, rnd * 31.1, rnd * 7.7);
  vec3 scat = vec3(0.0);
  vec3 T = vec3(1.0);
  vec3 lightIn = f_ambient * u_inclusionLight;

  // Normalize Beer-Lambert optical path by crystal radius so slender and thick crystals alike show vivid gemstone saturation
  float opticalScale = clamp(0.28 / max(r, 0.035), 1.5, 5.8);

  // Occasional characteristic: only ~30% of crystals carry an internal flash pocket at all
  float hasFlash = step(0.68, fract(rnd * 17.37 + 0.23));
  vec3 p0 = body(id, PA).xyz;
  vec3 t1 = p0 - axis * dot(p0, axis);
  float t1Len = length(t1);
  t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
  vec3 t2 = cross(axis, t1);
  float flashAz = rnd * TAU;
  float flashEl = (fract(rnd * 7.13) - 0.5) * 0.55;
  vec3 flashViewDir = normalize((t1 * cos(flashAz) + t2 * sin(flashAz)) * cos(flashEl) + axis * sin(flashEl));

  // One-sided narrow angular reflection cone (NO abs() — only visible from ONE specific angle as the crystal turns)
  float cosFlashAng = max(0.0, dot(-rd, flashViewDir));
  float schillerResonance = hasFlash * pow(cosFlashAng, 42.0);
  float flashCenterHgt = 0.35 + 0.30 * fract(rnd * 9.13);
  vec3 flashCenterOffset = (t1 * cos(flashAz + 0.6) + t2 * sin(flashAz + 0.6)) * (r * 0.18);

  // GPU/Mobile optimization: 14 steps on primary refraction ray (bounce == 0), 6 steps on secondary reflection bounces
  int maxSteps = (bounce == 0) ? min(int(u_inclusionSteps + 0.5), 14) : 6;
  int n = max(2, maxSteps);
  float dt = len / float(n);
  bool needInclTex = (u_milk > 0.01 || u_veils > 0.01 || u_iridescence > 0.01 || u_colourZoning > 0.01);
  for (int s = 0; s < 16; s++) {
    if (s >= n) break;
    if (bounce > 0 && s >= 6) break;
    vec3 pp = ro + rd * (dt * (float(s) + jitter));
    vec3 rel = pp - root;
    float axial = dot(rel, axis);
    float hgt = clamp(axial / max(L, 1e-3), 0.0, 1.0);
    float radFrac = clamp(length(rel - axis * axial) / max(r, 1e-3), 0.0, 1.0);
    float coreMix = u_biColour * clamp((1.0 - radFrac * 0.85) * 0.65 + (1.0 - hgt) * 0.55, 0.0, 1.0);
    vec3 baseSigma = mix(f_sigma, f_sigmaCore, coreMix);

    // Tint internal inclusion scattering by the stone's body color so milky clouds & veils never wash out gemstone hues
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

    // Deep Internal Crystal Flash (u_flash): small, localized internal pocket glow visible only from one angle in ~30% of crystals
    if (u_flash > 0.01 && schillerResonance > 1e-3) {
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

    if (u_iridescence > 0.01) {
      float iridPhase = hgt * 1.45 + radFrac * 0.95 + dot(rd, axis) * 0.95 + hueOff + v.b * 0.55;
      vec3 iridWave = spectralRainbow(iridPhase);
      float veilBand = (0.022 + 0.11 * v.g + 0.045 * v.r) * clamp(u_iridescence * 0.45, 0.0, 0.75);
      scat += T * iridWave * tintedLightIn * veilBand * dt * 1.45;
    }

    vec3 sigma = baseSigma * opticalScale * (1.0 + u_colourZoning * (2.0 * hgt - 1.0)) * (0.75 + 0.5 * v.b);
    T *= (1.0 - a) * exp(-sigma * dt);
  }

  // Multi-generation nested phantoms (guarded behind u_phantom > 0.01; smoothly weighted per layer & phantomRate so morphing never snaps)
  float phRateRoll = hash(vec3(uid, f_seed, 97.0));
  float phRateW = smoothstep(phRateRoll - 0.14, phRateRoll + 0.14, u_phantomRate);
  if (u_phantom > 0.01 && phRateW > 1e-4) {
    vec3 phTint = exp(-mix(f_sigma, f_sigmaCore, 0.4) * 0.35);
    vec3 phLightIn = lightIn * mix(vec3(1.0), normalize(phTint + vec3(0.05)) * 1.25, clamp(length(f_sigma) * 0.22, 0.0, 0.85));
    for (int layer = 0; layer < 4; layer++) {
      float layerWeight = clamp(u_phantomLayers - float(layer), 0.0, 1.0);
      if (layerWeight <= 1e-4) break;
      float pScale = u_phantomSize * (1.0 - 0.21 * float(layer));
      if (pScale <= 0.12) break;
      vec3 c = root + axis * L * (0.45 - 0.06 * float(layer));
      float tn = -1e9, tf = 1e9; int kn = 0, kf = 0;
      for (int k = 0; k < 18; k++) {
        vec4 pl = body(id, PA + k);
        float d = dot(pl.xyz, c) + (pl.w - dot(pl.xyz, c)) * pScale;
        float dn = dot(pl.xyz, rd), dist = d - dot(pl.xyz, ro);
        if (abs(dn) < 1e-8) { if (dist < 0.0) { tn = 1e9; } continue; }
        float t = dist / dn;
        if (dn < 0.0) { if (t > tn) { tn = t; kn = k; } } else if (t < tf) { tf = t; kf = k; }
      }
      if (tf > tn) {
        for (int e = 0; e < 2; e++) {
          float t = e == 0 ? tn : tf;
          if (t <= 0.0 || t >= len) continue;
          vec3 hp = ro + rd * t;
          float cosk = abs(dot(rd, body(id, PA + (e == 0 ? kn : kf)).xyz));
          float dust = texture(u_incl, hp * u_inclusionScale * 2.0 + off + float(layer) * 0.37).r;
          float sheet = phRateW * layerWeight * clamp(u_phantom * (0.12 - 0.02 * float(layer)) * (0.3 + 1.7 * dust) / max(cosk, 0.12), 0.0, 1.0);
          vec3 phCol = u_iridescence > 0.01
            ? mix(phLightIn, spectralRainbow(cosk * 2.4 + float(layer) * 0.33 + hueOff) * (phLightIn + vec3(0.35)), clamp(u_iridescence * 0.6, 0.0, 1.0))
            : phLightIn;
          scat += T * sheet * phCol * 1.45;
          T *= 1.0 - 0.45 * sheet;
        }
      }
    }
  }

  // Rutile needles (guarded behind u_needles > 0.01 && bounce == 0; smoothly weighted per needle so morphing needles never snaps)
  if (u_needles > 0.01 && bounce == 0) {
    for (int j = 0; j < 12; j++) {
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

  // Deep internal iridescent cleavage plane (guarded behind u_fractures > 0.01 || u_iridescence > 0.01)
  if (u_fractures > 0.01 || u_iridescence > 0.01 || u_flash > 0.01) {
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
            float film = v.b * 0.85 + length(hp - fc) * 1.4 + hueOff + abs(dn) * 1.2;
            vec3 irid = mix(vec3(0.96, 0.97, 1.0), spectralRainbow(film), clamp(u_iridescence * 0.72, 0.0, 0.85));
            float fr = 0.12 + 0.78 * pow(1.0 - abs(dn), 3.4);
            vec3 rawRefl = env(reflect(rd, fn), 0.05);
            vec3 refl = (rawRefl / (1.0 + 1.1 * max(rawRefl.r, max(rawRefl.g, rawRefl.b)))) + f_ambient * (0.16 * u_flash + 0.18 * u_iridescence);
            float glowBoost = min(u_fractureGlow, 2.2) * 0.65;
            scat += T * mask * glowBoost * fr * irid * refl;
            T *= 1.0 - 0.35 * mask * fr;
          }
        }
      }
    }
  }

  trans = T;
  return scat;
}

// Bend a face's normal: growth ridges across prism faces, stepped hopper terraces, and frost grain.
vec3 faceNormal(int id, int k, vec3 n, vec3 p, float rough, float footprint, float dEdge) {
  vec4 ax = body(id, 1);
  float uid = crystalUid(id);
  if (k < 8 && u_striations > 0.0) {
    float h = dot(p - body(id, 3).xyz, ax.xyz) * u_striationScale;
    float amp = texture(u_incl, vec3(h * 0.013, uid * 0.37, 0.5)).b * (1.0 - clamp(footprint * u_striationScale * 4.0, 0.0, 1.0));
    float ridge = sin(TAU * h + 2.0 * sin(h * 0.7));
    n = normalize(n + ax.xyz * u_striations * 0.08 * amp * ridge);
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
    vec3 pLoc = p - body(id, 3).xyz;
    vec3 cell = floor(pLoc * 820.0);
    vec3 patchCell = floor(pLoc * 140.0);
    float patchMod = 0.55 + 0.75 * hash(patchCell + uid * 3.1);
    n = normalize(n + rough * 0.068 * patchMod * (vec3(hash(cell), hash(cell + 1.3), hash(cell + 2.7)) - 0.5));
  }
  return n;
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

float edgeDist(int id, int k, vec3 p, out vec3 adjN) {
  if (k >= CP + 5000) {
    int enc = k - (CP + 5000);
    int pair = enc / 2;
    int ka = pair / 18;
    int kb = pair - ka * 18;
    vec3 sumN = body(id, PA + ka).xyz + body(id, PA + kb).xyz;
    adjN = normalize(sumN);
    return max(body(id, 2).w, 0.02) * 0.012;
  }
  if (k >= CP) {
    int enc = k - CP;
    int wallExit = enc - (enc / 2) * 2;
    int rem = enc / 2;
    int slotInt = rem - (rem / 5) * 5;
    int pair = rem / 5;
    int ka = pair / 18;
    int kb = pair - ka * 18;
    vec4 mUp, mDn;
    getOcclusionVoidPlanes(id, ka, kb, float(slotInt), mUp, mDn);
    vec4 mOther = wallExit == 0 ? mDn : mUp;
    adjN = normalize(mOther.xyz);
    return max(0.0, (dot(mOther.xyz, p) - mOther.w) / max(length(mOther.xyz), 1e-4));
  }
  float dmin = 1e9;
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
    }
  }
  return max(dmin, 0.0);
}

float wearFromDist(int id, vec3 p, float dmin) {
  if (u_edgeWear <= 0.0) return 0.0;
  float edge = 1.0 - smoothstep(0.0, u_edgeWidth * 0.45 * body(id, 2).w, dmin);
  float nz = texture(u_incl, p * 6.0 + 0.71).b * 0.75 + texture(u_incl, p * 21.0).b * 0.35;
  return 0.25 * edge * smoothstep(1.02 - 0.32 * u_edgeWear, 1.08 - 0.32 * u_edgeWear, nz);
}

// Softens & tints bright studio softbox glare on crystal faces (governed by u_faceGlare)
// so a lit facet never blows out to solid opaque white and internal crystal details stay visible.
vec3 softFacetRefl(vec3 rawEnv) {
  vec3 gemCol = exp(-f_sigma * 0.32);
  float lum = max(rawEnv.r, max(rawEnv.g, rawEnv.b));
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  vec3 compressed = rawEnv / (1.0 + (1.85 - 1.2 * glare) * lum);
  return mix(compressed * (0.42 + 0.58 * gemCol), compressed, 0.42 * glare);
}

vec3 refractedEnv(vec3 dir, float rough, float refAlpha) {
  vec3 d = f_toWorld * dir;
  vec3 bg = backdrop(d);
  vec3 lt = lights(d, rough);
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  float ltLum = max(lt.r, max(lt.g, lt.b));
  vec3 gemCol = exp(-f_sigma * 0.35);
  vec3 softLt = (lt / (1.0 + (2.4 - 1.65 * glare) * ltLum)) * mix(0.22 + 0.48 * gemCol, vec3(1.0), glare);
  return bg + mix(lt, softLt, refAlpha);
}

vec3 facetIridescence(int id, int k, vec3 p, vec3 rd, vec3 n, float dEdge, vec3 baseRefl) {
  if (u_iridescence <= 0.0) return baseRefl;
  float cosV = clamp(abs(dot(rd, n)), 0.0, 1.0);
  float L = max(body(id, 1).w, 0.1);
  float r = max(body(id, 2).w, 0.02);
  float hgt = dot(p - body(id, 3).xyz, body(id, 1).xyz) / L;
  float genShift = body(id, 2).y;
  float filmT = cosV * 2.4 + hgt * 1.35 + (dEdge / r) * 0.95 + float(k) * 0.19 + genShift;
  vec3 irid1 = spectralRainbow(filmT);
  vec3 irid2 = spectralRainbow(filmT * 1.62 + 0.29);
  vec3 specCol = normalize(mix(irid1, irid2, 0.35) + vec3(1e-4)) * 1.55;
  float fresRim = pow(1.0 - cosV, 2.0);
  float s = u_iridescence;
  vec3 tinted = mix(baseRefl, baseRefl * specCol * 1.28, clamp(s * 0.58, 0.0, 0.78));
  vec3 sheen = specCol * f_ambient * (0.18 + 0.58 * fresRim) * min(s, 1.5) * 0.28;
  return tinted + sheen;
}

// Surface facet catch-lights & bevel glints (governed by u_bloom and u_faceGlare)
vec3 crystalBloomGlint(int id, int k, vec3 p, vec3 rd, vec3 nSmooth, float dEdge, float rough) {
  if (u_bloom <= 0.0) return vec3(0.0);
  float glare = clamp(u_faceGlare, 0.0, 1.0);
  vec3 nFlat = body(id, PA + k).xyz;
  vec3 rFlat = normalize(f_toWorld * reflect(rd, nFlat));
  vec3 rSmooth = normalize(f_toWorld * reflect(rd, nSmooth));
  float r = max(body(id, 2).w, 0.02);
  float edgeMask = 1.0 - smoothstep(0.0, r * 0.11, dEdge);
  float apexMask = smoothstep(0.72, 0.98, dot(p - body(id, 3).xyz, body(id, 1).xyz) / max(body(id, 1).w, 0.1));
  float polish = 1.0 - 0.55 * rough;

  // Tight crystallographic specular lobes on bevel edges while keeping flat faces translucent
  float expFacet = mix(220.0, 75.0, clamp(u_bloom * 0.45, 0.0, 0.9));
  float expEdge = mix(320.0, 110.0, clamp(u_bloom * 0.45, 0.0, 0.9));
  vec3 flashSum = vec3(0.0);

  for (int i = 0; i < LIGHTS; i++) {
    int o = i * LS;
    vec3 lDir = normalize(vec3(f_lights[o], f_lights[o + 1], f_lights[o + 2]));
    vec3 lCol = vec3(f_lights[o + 9], f_lights[o + 10], f_lights[o + 11]);
    float cFlat = max(0.0, dot(rFlat, lDir));
    float cSmooth = max(0.0, dot(rSmooth, lDir));
    float fGlint = pow(cFlat, expFacet);
    float eGlint = pow(max(cFlat, cSmooth), expEdge) * (0.4 + 2.4 * max(edgeMask, apexMask));
    flashSum += lCol * (fGlint * 0.22 * glare + eGlint * 0.75);
  }

  // 6 pinpoint studio catch-lights around the orbit ring
  for (int j = 0; j < 6; j++) {
    float az = float(j) * (TAU / 6.0) + 0.41;
    float el = (j & 1) == 0 ? 0.36 : 0.68;
    vec3 spotDir = vec3(sin(az) * cos(el), sin(el), cos(az) * cos(el));
    float sFlat = max(0.0, dot(rFlat, spotDir));
    float sSmooth = max(0.0, dot(rSmooth, spotDir));
    float facetHit = pow(sFlat, expFacet * 1.15);
    float edgeHit = pow(max(sFlat, sSmooth), expEdge) * (0.25 + 2.2 * max(edgeMask, apexMask));
    vec3 prismTint = mix(vec3(1.0, 0.98, 0.95), spectralRainbow(float(k) * 0.21 + float(j) * 0.17 + body(id, 2).y), clamp(0.25 + u_iridescence * 0.3, 0.0, 0.85));
    flashSum += prismTint * (facetHit * 1.05 * glare + edgeHit * 4.5);
  }

  float overdrive = u_bloom * (0.55 + 0.45 * u_bloom) * (0.22 + 0.78 * glare);
  vec3 rawGlint = flashSum * polish * overdrive;
  return rawGlint / (1.0 + (1.35 - 0.85 * glare) * max(rawGlint.r, max(rawGlint.g, rawGlint.b)));
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

  for (int hop = 0; hop < 10; hop++) {
    if (hopCost >= float(depth)) break;
    int id, k; float t;
    if (!traceScene(ro, rd, skipMask, id, t, k)) break;
    vec3 pEntry = ro + rd * t;
    vec3 p = pEntry;
    travelled += t;
    skipMask |= (1 << id);

    float rawAlpha = clamp(body(id, 2).x, 0.0, 1.0);
    float budgetWeight = clamp(float(depth) - hopCost, 0.0, 1.0);
    float alpha = rawAlpha * rawAlpha * (3.0 - 2.0 * rawAlpha) * budgetWeight;
    if (alpha <= 1e-5) {
      ro = pEntry;
      continue;
    }
    float effIor = mix(1.0, ior, alpha);
    float uid = crystalUid(id);
    hopCost += alpha;

    float footprint = travelled * f_pixAngle;
    float frostRoll = hash(vec3(uid, float(k), f_seed + 0.5));
    float frostW = smoothstep(frostRoll - 0.15, frostRoll + 0.15, u_frostShare);
    float rough = u_frost * frostW;
    bool isTinyEdgeChip = (k >= CP + 5000);
    bool isOcclusionVoid = (k >= CP && k < CP + 5000);
    bool isEdgeIncursion = isTinyEdgeChip || isOcclusionVoid;
    bool isCornerNip = (k >= 18 && k < CP);
    vec3 rawFaceN = vec3(0.0, 1.0, 0.0);
    if (isTinyEdgeChip) {
      int enc = k - (CP + 5000);
      int wallBit = enc - (enc / 2) * 2;
      int pair = enc / 2;
      int ka = pair / 18;
      int kb = pair - ka * 18;
      vec3 ax = body(id, 1).xyz;
      vec3 nAB = normalize(body(id, PA + ka).xyz + body(id, PA + kb).xyz);
      rawFaceN = normalize(nAB + ax * (wallBit == 0 ? -0.32 : 0.32));
    } else if (isOcclusionVoid) {
      int enc = k - CP;
      int wallExit = enc - (enc / 2) * 2;
      int rem = enc / 2;
      int slotInt = rem - (rem / 5) * 5;
      int pair = rem / 5;
      int ka = pair / 18;
      int kb = pair - ka * 18;
      vec4 mUp, mDn;
      getOcclusionVoidPlanes(id, ka, kb, float(slotInt), mUp, mDn);
      rawFaceN = normalize(wallExit == 0 ? mUp.xyz : mDn.xyz);
    } else {
      rawFaceN = body(id, PA + k).xyz;
    }
    vec3 adjN = rawFaceN;
    float dEdge = (u_edgeWear > 0.0 || u_edgeChips > 0.01 || u_occlusion > 0.01 || u_hopper > 0.0 || u_iridescence > 0.0 || u_bloom > 0.0) ? edgeDist(id, k, p, adjN) : 1.0;
    float eDist = dEdge;
    float chip = wearFromDist(id, p, dEdge);
    rough = max(rough, chip * 0.6);
    vec3 n = faceNormal(id, isEdgeIncursion ? 18 : k, rawFaceN, p, rough, footprint, dEdge);
    float inChip = 0.0;
    if ((u_edgeChips > 0.01 || u_occlusion > 0.01) && (isEdgeIncursion || isCornerNip)) {
      vec3 axis = body(id, 1).xyz, root = body(id, 3).xyz;
      float r = max(body(id, 2).w, 0.03);
      vec3 p0 = body(id, PA).xyz;
      vec3 t1 = p0 - axis * dot(p0, axis);
      float t1Len = length(t1);
      t1 = t1Len > 1e-4 ? (t1 / t1Len) : normalize(cross(axis, vec3(0.577, 0.577, 0.577)));
      vec3 t2 = cross(axis, t1);
      vec3 rel = p - root;
      vec3 lp = vec3(dot(rel, t1), dot(rel, axis), dot(rel, t2));
      if (isEdgeIncursion) {
        // Literal 3D V-notch / conchoidal smash incursion carved into the crystal edge (pure refractive crystal facet, no white paint)
        float incFeather = smoothstep(0.0, r * 0.015, eDist + r * 0.004);
        vec3 chipN = normalize(vec3(n3(lp * 62.0) - 0.5, n3(lp * 62.0 + 7.1) - 0.5, n3(lp * 62.0 + 13.3) - 0.5));
        n = normalize(rawFaceN + chipN * (0.085 * incFeather));
        inChip = 0.08 * incFeather * clamp(u_edgeChips * 1.4, 0.0, 1.0);
        rough = max(rough, 0.30 * incFeather);
      } else {
        // Literal 3D chipped corner facet: smooth feather near edge so merging in/out never pops
        float nipFeather = smoothstep(0.0, r * 0.012, eDist);
        vec3 chipN = normalize(vec3(n3(lp * 58.0) - 0.5, n3(lp * 58.0 + 7.1) - 0.5, n3(lp * 58.0 + 13.3) - 0.5));
        n = normalize(n + chipN * (0.065 * nipFeather));
        inChip = 0.08 * nipFeather * clamp(u_edgeChips * 1.4, 0.0, 1.0);
        rough = max(rough, 0.24 * nipFeather);
      }
    }
    float F = fresnel(-dot(rd, n), 1.0 / effIor) * alpha;
    vec3 entryRefl = facetIridescence(id, k, p, rd, n, dEdge, softFacetRefl(env(reflect(rd, n), rough)));
    vec3 entryFlash = crystalBloomGlint(id, k, p, rd, n, dEdge, rough);
    float iridReflBoost = clamp(F + u_iridescence * 0.09 * alpha, 0.0, 0.48 * alpha) * (0.15 + 0.85 * glare);
    col += thr * (iridReflBoost * entryRefl + alpha * (chip * 0.12 * f_ambient + entryFlash));
    thr *= (1.0 - F * (0.3 + 0.7 * glare)) * (1.0 - 0.6 * chip * alpha);
    if (inChip > 0.001) {
      vec3 stoneCol = exp(-f_sigma * 0.35);
      vec3 keyDir = normalize(vec3(f_lights[0], f_lights[1], f_lights[2]));
      float diffLight = clamp(dot(f_toWorld * n, keyDir) * 0.5 + 0.5, 0.0, 1.0);
      vec3 chipScatter = stoneCol * (0.35 + 0.65 * diffLight) * (0.35 + 0.65 * f_ambient);
      float chipMix = inChip * 0.25 * alpha;
      col += thr * chipMix * chipScatter;
    }
    vec3 refrIn = refract(rd, n, 1.0 / effIor);
    rd = dot(refrIn, refrIn) > 1e-6 ? normalize(mix(rd, refrIn, alpha)) : rd;
    ro = p;

    bool left = false;
    for (int b = 0; b < 8; b++) {
      if (b >= bounces) break;
      int kf;
      float len = exitCrystal(id, ro, rd, kf);
      vec3 trans;
      vec3 inScat = inside(id, ro, rd, len, travelled, jitter, b, trans);
      col += thr * alpha * inScat;
      thr *= mix(vec3(1.0), trans, alpha);
      float nextTravelled = travelled + len * alpha;
      p = ro + rd * len;

      float exitFrostRoll = hash(vec3(uid, float(kf), f_seed + 0.5));
      float exitFrostW = smoothstep(exitFrostRoll - 0.15, exitFrostRoll + 0.15, u_frostShare);
      float fr = u_frost * exitFrostW;
      vec3 exitAdjN;
      float dExit = (u_hopper > 0.0 || u_iridescence > 0.0 || u_bloom > 0.0) ? edgeDist(id, kf, p, exitAdjN) : 1.0;
      vec3 nf = faceNormal(id, kf, body(id, PA + kf).xyz, p, fr, nextTravelled * f_pixAngle, dExit);
      float cosi = dot(rd, nf);
      vec3 o = refract(rd, -nf, effIor);
      if (dot(o, o) < 1e-6) {
        if (b + 1 < bounces) { travelled = nextTravelled; rd = reflect(rd, -nf); ro = p; continue; }
        o = rd;
      }
      float Fo = fresnel(cosi, effIor) * alpha;
      vec3 exitRefl = facetIridescence(id, kf, p, rd, -nf, dExit, softFacetRefl(env(reflect(rd, -nf), fr)));
      vec3 exitFlash = crystalBloomGlint(id, kf, p, rd, -nf, dExit, fr) * 0.4;
      col += thr * (Fo * exitRefl * 0.42 * (0.15 + 0.85 * glare) + alpha * exitFlash);
      thr *= 1.0 - Fo * (0.3 + 0.7 * glare);
      lastIn = lastRefracted ? normalize(mix(lastIn, rd, alpha)) : rd;
      lastN = lastRefracted ? normalize(mix(lastN, nf, alpha)) : nf;
      lastRough = mix(lastRough, fr, alpha);
      lastFireAlpha = 1.0 - (1.0 - lastFireAlpha) * (1.0 - alpha);
      lastRefracted = lastFireAlpha > 1e-4;
      rd = normalize(mix(rd, o, alpha));
      ro = (b == 0) ? (p - rd * (len * (1.0 - alpha))) : p;
      travelled = nextTravelled;
      left = true;
      break;
    }
    if (!left) { break; }
    if (dot(thr, thr) < 1e-6) break;
  }

  if (dot(thr, thr) > 0.0) {
    float effFire = u_fire + u_iridescence * 0.028;
    if (lastRefracted && effFire > 0.005 && lastFireAlpha > 1e-4) {
      int ns = int(u_fireSamples + 0.5);
      vec3 acc = vec3(0.0), wsum = vec3(0.0);
      float dispIor = mix(1.0, ior, lastFireAlpha);
      float dispFire = effFire * lastFireAlpha;
      for (int i = 0; i < 9; i++) {
        if (i >= ns) break;
        float x = (float(i) + 0.5) / float(ns);
        vec3 dw = (x - vec3(0.12, 0.5, 0.88)) / 0.38;
        vec3 w = max(1.0 - dw * dw, 0.0);
        vec3 o = refract(lastIn, -lastN, dispIor + dispFire * (x - 0.5) * 2.0);
        vec3 sampleDir = dot(o, o) > 1e-6 ? normalize(mix(rd, o, lastFireAlpha)) : rd;
        wsum += w;
        vec3 eCol = refractedEnv(sampleDir, lastRough, lastFireAlpha);
        if (u_iridescence > 0.0) {
          float defl = length(sampleDir - initialRd);
          vec3 prism = spectralRainbow(defl * 2.4 + x * 0.85);
          eCol += prism * f_ambient * u_iridescence * 0.24 * lastFireAlpha * smoothstep(0.06, 0.48, defl);
        }
        acc += w * eCol;
      }
      col += thr * acc / max(wsum, vec3(1e-4));
    } else {
      col += thr * refractedEnv(rd, lastRough, lastFireAlpha);
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
  float thresh = max(0.95, u_bloomThreshold);
  outColor = encodeHdr(c * max(l - thresh, 0.0) / max(l, 1e-4));
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
void main() {
  vec2 fc = gl_FragCoord.xy, uv = fc / u_res;
  vec3 bloomCol = decodeHdr(texture(u_bloomTex, uv));
  if (u_bloom > 0.05) {
    vec2 sp = f_px * 3.5 / u_res;
    vec3 star = (
      decodeHdr(texture(u_bloomTex, uv + vec2(sp.x, 0.0))) +
      decodeHdr(texture(u_bloomTex, uv - vec2(sp.x, 0.0))) +
      decodeHdr(texture(u_bloomTex, uv + vec2(0.0, sp.y))) +
      decodeHdr(texture(u_bloomTex, uv - vec2(0.0, sp.y)))
    ) * 0.25;
    bloomCol += max(star - vec3(0.15), vec3(0.0)) * u_bloom * 0.65;
  }
  vec3 c = decodeHdr(texture(u_hdr, uv)) * u_exposure + bloomCol * (u_bloom * 1.22);
  c = pow(aces(c), vec3(1.0 / 2.2));
  vec2 q = (fc - 0.5 * u_res) / u_res;
  c = mix(c, u_studioEdge, clamp(u_vignette * dot(q, q) * 2.6, 0.0, 1.0));
  c = filmDust(c, fc, f_px, f_frame, u_dust);
  c = filmGrain(c, fc, f_px, f_frame, u_grainAmount, u_grainSize, u_grainSharpness, u_grainClump);
  c += (hash(vec3(fc, f_frame)) - 0.5) / 255.0;
  outColor = vec4(c, 1.0);
}`,
};
