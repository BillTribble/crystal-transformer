// Crystals — multi-habit procedural crystal & twin-cluster geometry builder with
// bounded sliding-window evolution (Phase 1: Seed -> Full Size, Phase 2: Continuous
// Frontier Branching + Smooth Old-Segment Dissolve & Garbage Collection).
//
// Every crystal body is the exact intersection of 21 half-spaces (CRYSTAL_PLANES = 21),
// written into `cluster.data` (a std140 uniform buffer read as `bodies[]` in the trace shader).
// Slot layout per crystal row (`ROW_W = 25` vec4s):
//   col 0: vec4(sphereCentre.xyz, sphereRadius)
//   col 1: vec4(axis.xyz, L)
//   col 2: vec4(alpha, hueOffset, stableUid, r)
//   col 3: vec4(root.xyz, rnd)
//   col 4..24: 21 half-space planes vec4(normal.xyz, planeOffset) (slots 18..20 = 3D corner-nip planes)

export const MAX_CRYSTALS = 28;
export const CRYSTAL_PLANES = 21;
export const PLANES_AT = 4; // first plane slot in a row
export const ROW_W = PLANES_AT + CRYSTAL_PLANES;
export const ROWS = MAX_CRYSTALS;

const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));

export const cluster = {
  data: new Float32Array(ROW_W * ROWS * 4),
  crystals: 0,
  floatCrystals: 9,
  activeVectors: 0,
  maxVectors: MAX_CRYSTALS * CRYSTAL_PLANES,
  maxCrystals: MAX_CRYSTALS,
  version: 0,
  phase: 'evolving', // 'evolving' (continuous whole-cluster transformation) | 'static'
  seedProgress: 1.0,
  evolveTime: 0,
  evoPhase: 0,
  evoBurstVel: 0,
  spawnedCount: 0,
  dissolvedCount: 0,
  generation: 0,
  nodes: [],
  centerInit: false,
  pendingCenterDt: 0,
  centreX: 0,
  centreY: 0,
  centreZ: 0,
  velCentreX: 0,
  velCentreY: 0,
  velCentreZ: 0,
  targetCentreX: 0,
  targetCentreY: 0,
  targetCentreZ: 0,
  radius: 1,
  radiusXZ: 0.5,
  halfY: 0.5,
  aspect: 1.6,
  growDirection: 'sideways',
  fillX: 0.5,
  fillY: 0.5,
  screenFill: 0.5,
};

// mulberry32 PRNG, allocation-free
let state = 0;
function seedRandom(seed) { state = Math.floor(seed * 1000) ^ 0x9e3779b9; }
function rand() {
  state = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(state ^ (state >>> 15), 1 | state);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const sym = () => rand() * 2 - 1;
// Quintic C2-continuous ease-in / ease-out S-curve (zero 1st & 2nd derivative at both endpoints)
const smooth01 = (x) => {
  const t = Math.max(0, Math.min(1, x));
  return t * t * t * (t * (t * 6 - 15) + 10);
};

// Scratch vectors for the current crystal frame
const ax = new Float32Array(3), sx = new Float32Array(3), sz = new Float32Array(3), root = new Float32Array(3);

function slot(row, col, x, y, z, w) {
  const o = (row * ROW_W + col) * 4;
  cluster.data[o] = x; cluster.data[o + 1] = y; cluster.data[o + 2] = z; cluster.data[o + 3] = w;
}

/** A plane from a unit normal in the crystal's frame (sx, ax, sz) and its offset from `root`. */
function crystalPlane(row, k, lx, ly, lz, offset) {
  const inv = 1 / Math.hypot(lx, ly, lz);
  lx *= inv; ly *= inv; lz *= inv;
  const nx = sx[0] * lx + ax[0] * ly + sz[0] * lz;
  const ny = sx[1] * lx + ax[1] * ly + sz[1] * lz;
  const nz = sx[2] * lx + ax[2] * ly + sz[2] * lz;
  slot(row, PLANES_AT + k, nx, ny, nz, nx * root[0] + ny * root[1] + nz * root[2] + offset);
}

/** A plane from a normal in the crystal's frame and its offset from the crystal's midpoint `root + 0.5 * L * ax`. */
function centeredPlane(row, k, L, lx, ly, lz, offsetFromMid) {
  const inv = 1 / Math.hypot(lx, ly, lz);
  lx *= inv; ly *= inv; lz *= inv;
  const nx = sx[0] * lx + ax[0] * ly + sz[0] * lz;
  const ny = sx[1] * lx + ax[1] * ly + sz[1] * lz;
  const nz = sx[2] * lx + ax[2] * ly + sz[2] * lz;
  const mx = root[0] + ax[0] * 0.5 * L;
  const my = root[1] + ax[1] * 0.5 * L;
  const mz = root[2] + ax[2] * 0.5 * L;
  slot(row, PLANES_AT + k, nx, ny, nz, nx * mx + ny * my + nz * mz + offsetFromMid);
}

// Scratch buffers for exact convex-polytope vertex enumeration (allocation-free).
const POLY_N = new Float64Array(18 * 3);
const POLY_W = new Float64Array(18);
const POLY_V = new Float64Array(3 * 816);

/**
 * Enumerates the exact vertices of crystal `rowIdx` (the intersection of its active half-spaces
 * 0..17, skipping the parked far-away slots above `nSym`) into POLY_V and returns the count.
 */
function enumerateCrystalVertices(rowIdx) {
  const base = (rowIdx * ROW_W + PLANES_AT) * 4;
  const d = cluster.data;
  let n = 0;
  for (let k = 0; k < 18; k++) {
    const o = base + k * 4;
    POLY_N[n * 3] = d[o]; POLY_N[n * 3 + 1] = d[o + 1]; POLY_N[n * 3 + 2] = d[o + 2];
    POLY_W[n] = d[o + 3];
    n++;
  }
  let count = 0;
  for (let i = 0; i < n - 2; i++) {
    const ax0 = POLY_N[i * 3], ay0 = POLY_N[i * 3 + 1], az0 = POLY_N[i * 3 + 2];
    for (let j = i + 1; j < n - 1; j++) {
      const bx = POLY_N[j * 3], by = POLY_N[j * 3 + 1], bz = POLY_N[j * 3 + 2];
      const abx = ay0 * bz - az0 * by, aby = az0 * bx - ax0 * bz, abz = ax0 * by - ay0 * bx;
      for (let k = j + 1; k < n; k++) {
        const cx = POLY_N[k * 3], cy = POLY_N[k * 3 + 1], cz = POLY_N[k * 3 + 2];
        const bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;
        const det = ax0 * bcx + ay0 * bcy + az0 * bcz;
        if (Math.abs(det) < 1e-7) continue;
        const cax = cy * az0 - cz * ay0, cay = cz * ax0 - cx * az0, caz = cx * ay0 - cy * ax0;
        const inv = 1 / det;
        const wi = POLY_W[i], wj = POLY_W[j], wk = POLY_W[k];
        const vx = (wi * bcx + wj * cax + wk * abx) * inv;
        const vy = (wi * bcy + wj * cay + wk * aby) * inv;
        const vz = (wi * bcz + wj * caz + wk * abz) * inv;
        let inside = true;
        for (let m = 0; m < n; m++) {
          if (m === i || m === j || m === k) continue;
          if (POLY_N[m * 3] * vx + POLY_N[m * 3 + 1] * vy + POLY_N[m * 3 + 2] * vz > POLY_W[m] + 1e-5) {
            inside = false;
            break;
          }
        }
        if (!inside) continue;
        POLY_V[count * 3] = vx; POLY_V[count * 3 + 1] = vy; POLY_V[count * 3 + 2] = vz;
        if (++count >= 816) return count;
      }
    }
  }
  return count;
}

/**
 * Writes a corner-nip half-space into `slotIdx` (18..20) that slices a cap of exactly `depth`
 * off the crystal's most extreme corner in the outward direction between prism faces
 * `kPrism0`/`kPrism1` (tilted up by cVert). The plane offset is the crystal's exact support
 * value in that direction minus `depth`, so the nip is always a tiny corner cap that moves and
 * grows continuously: it can never cull the whole crystal (the old ridge-vertex search could
 * land on a vertex outside the crystal and make the entire crystal vanish) and never toggles.
 */
function writeCornerNipPlane(rowIdx, slotIdx, kPrism0, kPrism1, cHoriz, cVert, depth, bias0, bias1, vertCount, L, r) {
  const base = (rowIdx * ROW_W + PLANES_AT) * 4;
  const d = cluster.data;
  const oP0 = base + kPrism0 * 4, oP1 = base + kPrism1 * 4;
  let hx = d[oP0] * bias0 + d[oP1] * bias1;
  let hy = d[oP0 + 1] * bias0 + d[oP1 + 1] * bias1;
  let hz = d[oP0 + 2] * bias0 + d[oP1 + 2] * bias1;
  const hProj = hx * ax[0] + hy * ax[1] + hz * ax[2];
  hx -= hProj * ax[0]; hy -= hProj * ax[1]; hz -= hProj * ax[2];
  let hl = Math.hypot(hx, hy, hz);
  if (hl < 1e-4) { hx = sx[0]; hy = sx[1]; hz = sx[2]; hl = 1; }
  hx /= hl; hy /= hl; hz /= hl;

  let nx = hx * cHoriz + ax[0] * cVert;
  let ny = hy * cHoriz + ax[1] * cVert;
  let nz = hz * cHoriz + ax[2] * cVert;
  const nl = Math.hypot(nx, ny, nz) || 1;
  nx /= nl; ny /= nl; nz /= nl;

  let support = -Infinity;
  let minSupport = Infinity;
  for (let v = 0; v < vertCount; v++) {
    const s = nx * POLY_V[v * 3] + ny * POLY_V[v * 3 + 1] + nz * POLY_V[v * 3 + 2];
    if (s > support) support = s;
    if (s < minSupport) minSupport = s;
  }
  if (!Number.isFinite(support)) {
    // Degenerate (sub-pixel) crystal: park the nip safely outside a generous bound.
    const mx = root[0] + ax[0] * 0.5 * L, my = root[1] + ax[1] * 0.5 * L, mz = root[2] + ax[2] * 0.5 * L;
    support = nx * mx + ny * my + nz * mz + L + 4 * r;
    minSupport = support - 2 * r;
  }
  const maxSafeDepth = (Number.isFinite(minSupport) && support > minSupport)
    ? Math.min(r * 0.14, (support - minSupport) * 0.12)
    : r * 0.10;
  const clampedDepth = Math.max(0, Math.min(depth, maxSafeDepth));
  slot(rowIdx, PLANES_AT + slotIdx, nx, ny, nz, support - clampedDepth);
}

function pcg32(v) {
  let state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  let word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/**
 * Deterministic corner-nip whale roll in [0, 1):
 * 6% (0.93 <= roll < 0.99) get a 10x 'Whale' corner-nip cut,
 * 1% (roll >= 0.99) get a 20x 'Mega-Whale' corner-nip cut.
 */
function cornerWhaleRoll(uid, slotIdx) {
  const ux = (Math.imul(uid | 0, 37) + Math.imul(slotIdx | 0, 19)) >>> 0;
  const uy = (Math.imul(slotIdx | 0, 53) + Math.imul(uid | 0, 11)) >>> 0;
  const uz = 12;
  return pcg32((ux + pcg32((uy + pcg32(uz)) >>> 0)) >>> 0) / 4294967295.0;
}

function buildSideBasis(twistAngle) {
  // Singularity-free quaternion half-angle orthonormal basis around +Y = (0, 1, 0).
  // Unlike cross([0, 1, 0], ax), which has a pole singularity right at ax = (0, 1, 0)
  // where upright crystals point (causing a 180-degree azimuth pirouette when a crystal
  // tilts across vertical), this shortest-arc mapping is C-infinity smooth across +Y.
  const ax0 = ax[0], ay0 = ax[1], az0 = ax[2];
  const denom = Math.max(1e-4, 1.0 + ay0);
  const ux0 = 1.0 - (ax0 * ax0) / denom;
  const uy0 = -ax0;
  const uz0 = -(ax0 * az0) / denom;
  const vx0 = -(ax0 * az0) / denom;
  const vy0 = -az0;
  const vz0 = 1.0 - (az0 * az0) / denom;
  const c = Math.cos(twistAngle), s = Math.sin(twistAngle);
  sx[0] = c * ux0 + s * vx0;
  sx[1] = c * uy0 + s * vy0;
  sx[2] = c * uz0 + s * vz0;
  const sl = Math.hypot(sx[0], sx[1], sx[2]) || 1;
  sx[0] /= sl; sx[1] /= sl; sx[2] /= sl;
  sz[0] = ax[1] * sx[2] - ax[2] * sx[1];
  sz[1] = ax[2] * sx[0] - ax[0] * sx[2];
  sz[2] = ax[0] * sx[1] - ax[1] * sx[0];
}

export function resolveHabitSystem(habit) {
  const h = String(habit || 'Trigonal');
  if (h.startsWith('Cubic') || h.startsWith('Isometric') || h === 'Octahedral Fluorite') return 'Cubic / Isometric';
  if (h.startsWith('Hexagonal') || h === 'Cathedral Spire' || h === 'Tabular Gem' || h === 'Dendritic Starburst') return 'Hexagonal';
  if (h.startsWith('Tetragonal')) return 'Tetragonal';
  if (h.startsWith('Orthorhombic')) return 'Orthorhombic';
  if (h.startsWith('Monoclinic')) return 'Monoclinic';
  if (h.startsWith('Triclinic')) return 'Triclinic';
  return 'Trigonal';
}

function primaryCountFor(n, habit, twinShare) {
  const sys = resolveHabitSystem(habit);
  let nPrimary = n;
  if (n > 2) {
    if (sys === 'Cubic / Isometric') {
      nPrimary = Math.max(2, Math.round(n * 0.72));
    } else if (sys === 'Monoclinic' || sys === 'Triclinic') {
      nPrimary = Math.max(2, Math.round(n * (1 - 0.38 * twinShare)));
    } else if (twinShare > 0.02) {
      nPrimary = Math.max(1, Math.round(n * (1 - 0.45 * twinShare)));
    }
  }
  return nPrimary;
}

function setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, k, lx, ly, lz, offR, offL) {
  const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
  outLx[k] = lx * inv;
  outLy[k] = ly * inv;
  outLz[k] = lz * inv;
  outOffR[k] = offR;
  outOffL[k] = offL;
}

/**
 * Computes the 18 primary crystallographic half-space planes (k = 0..17) in the crystal's
 * local (sx, ax, sz) coordinate frame for any of the 7 crystallographic crystal systems:
 *   1. Trigonal (3-fold rhombohedral r{10-11}/z{01-11} & ditrigonal/hexagonal prism)
 *   2. Cubic / Isometric (a = b = c, alpha = beta = gamma = 90°: octahedron {111} + cube {100} + dodecahedron {110})
 *   3. Hexagonal (6-fold c-axis, gamma = 120°: equal 6-sided prism {10-10} + flat {0001} pinacoid + {10-11} bipyramid)
 *   4. Tetragonal (4-fold c-axis, a = b != c, 90°: square prism {100}/{110} + steep bipyramid {101}/{211})
 *   5. Orthorhombic (a != b != c, alpha = beta = gamma = 90°: unequal rhombic prism {110}+{100}+{010} + chisel domes {101}/{011})
 *   6. Monoclinic (a != b != c, alpha = gamma = 90°, beta != 90°: flattened lath prism + obliquely slanted {001} & clinodomes)
 *   7. Triclinic (a != b != c, alpha != beta != gamma != 90°: non-90° skewed bladed prism + doubly-oblique asymmetric wedge)
 */
function computeSystemPlanes(
  habit, symVal, taperLy, latScale, nodeTilt, chamfer, dom, faceJitterAmt, tipJitterAmt, fs, L, r,
  outLx, outLy, outLz, outOffR, outOffL
) {
  const sys = resolveHabitSystem(habit);
  const fVar = (k, scale = 0.35) => Math.max(0.55, Math.min(1.45, 1 + faceJitterAmt * scale * fs[k]));
  const tVar = (k, scale = 0.28) => tipJitterAmt * scale * fs[8 + k];

  if (sys === 'Cubic / Isometric') {
    // Equant isometric polyhedron (a = b = c, alpha = beta = gamma = 90°) centered at y = 0.52 * L
    const isoScale = Math.max(1.18, Math.min(2.25, (0.34 * L) / Math.max(r, 1e-4)));
    const cosOct = 0.57735027; // 1 / sqrt(3)
    const sinOct = 0.81649658; // sqrt(2 / 3)
    const cos45 = 0.70710678;
    const cubeDist = isoScale * Math.max(0.96, 1.38 - 0.52 * chamfer);
    const dodDist = isoScale * Math.max(1.02, 1.24 - 0.24 * chamfer);
    for (let m = 0; m < 4; m++) {
      const aCube = m * 0.5 * Math.PI; // 0, 90, 180, 270 deg: vertical {100}/{001} cube faces
      const kCube = m * 2;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kCube,
        Math.cos(aCube), 0, Math.sin(aCube),
        cubeDist * fVar(kCube, 0.14), 0
      );
      const aOct = (m + 0.5) * 0.5 * Math.PI; // 45, 135, 225, 315 deg: octahedral {111} faces
      const kBotOct = m * 2 + 1;
      const dBot = isoScale * (1 + 0.10 * tipJitterAmt * fs[kBotOct]);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kBotOct,
        sinOct * Math.cos(aOct), -cosOct, sinOct * Math.sin(aOct),
        dBot, -0.52 * cosOct
      );
      const kTopOct = 8 + m * 2;
      const dTop = isoScale * (1 + 0.10 * tipJitterAmt * fs[kTopOct]);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kTopOct,
        sinOct * Math.cos(aOct), cosOct, sinOct * Math.sin(aOct),
        dTop, 0.52 * cosOct
      );
      const kDod = 8 + m * 2 + 1; // {110} rhombic dodecahedral bevels
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kDod,
        cos45 * Math.cos(aCube), cos45, cos45 * Math.sin(aCube),
        dodDist * fVar(kDod, 0.12), 0.52 * cos45
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, cubeDist, -0.52);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, cubeDist, 0.52);
    return;
  }

  if (sys === 'Hexagonal') {
    // 6-fold hexagonal prism {10-10} + prominent flat {0001} basal pinacoid + 6 equal {10-11} bipyramid bevels
    const hexTaper = taperLy * 0.45;
    const hexLat = Math.sqrt(Math.max(0.1, 1 - hexTaper * hexTaper));
    for (let k = 0; k < 6; k++) {
      const a = (k * TAU) / 6;
      const radR = 1.0 * fVar(k, 0.20);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        hexLat * Math.cos(a), hexTaper, hexLat * Math.sin(a),
        radR, 0.5 * hexTaper
      );
    }
    // Dihexagonal {11-20} accessory bevels at 30° and 210°
    for (let m = 0; m < 2; m++) {
      const k = 6 + m;
      const a = (m * Math.PI) + Math.PI / 6;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        hexLat * Math.cos(a), hexTaper, hexLat * Math.sin(a),
        (1.20 - 0.12 * chamfer) * fVar(k, 0.15), 0.5 * hexTaper
      );
    }
    const bipTilt = Math.max(30 * DEG, Math.min(58 * DEG, nodeTilt * 0.78));
    const cosB = Math.cos(bipTilt), sinB = Math.sin(bipTilt);
    const tableH = Math.max(0.68, 0.84 - 0.22 * chamfer);
    for (let j = 0; j < 6; j++) {
      const a = (j * TAU) / 6;
      const offR = (0.34 - 0.16 * chamfer) * sinB + tVar(j, 0.16);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        sinB * Math.cos(a), cosB, sinB * Math.sin(a),
        offR, tableH * cosB
      );
    }
    const bip2Tilt = Math.min(68 * DEG, bipTilt + 9 * DEG);
    const cosB2 = Math.cos(bip2Tilt), sinB2 = Math.sin(bip2Tilt);
    for (let m = 0; m < 2; m++) {
      const a = (m * Math.PI) + Math.PI / 6;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 14 + m,
        sinB2 * Math.cos(a), cosB2, sinB2 * Math.sin(a),
        0.48 * sinB2, tableH * cosB2
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, 0.04, 0);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableH);
    return;
  }

  if (sys === 'Tetragonal') {
    // 4-fold c-axis symmetry (a = b != c, alpha = beta = gamma = 90°): square {100} prism + {110} corner chamfers + {101}/{211} bipyramid
    const cornerChamfer = 1.14 - 0.22 * chamfer;
    for (let k = 0; k < 8; k++) {
      const a = (k * TAU) / 8; // 0, 45, 90, 135, 180, 225, 270, 315 deg
      const isCorner = (k & 1) === 1;
      const baseR = isCorner ? cornerChamfer : 0.90;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
        baseR * fVar(k, 0.16), 0.5 * taperLy
      );
    }
    const t101 = Math.max(38 * DEG, Math.min(66 * DEG, nodeTilt));
    const c101 = Math.cos(t101), s101 = Math.sin(t101);
    const t211 = Math.min(78 * DEG, t101 + 12 * DEG);
    const c211 = Math.cos(t211), s211 = Math.sin(t211);
    for (let j = 0; j < 8; j++) {
      const a = (j * TAU) / 8;
      const isSecondOrder = (j & 1) === 1;
      const ct = isSecondOrder ? c211 : c101;
      const st = isSecondOrder ? s211 : s101;
      const offR = (isSecondOrder ? (0.30 - 0.16 * chamfer) : 0.06) + tVar(j, 0.18);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        st * Math.cos(a), ct, st * Math.sin(a),
        offR, ct
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, 0.05, 0);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, Math.max(0.76, 1.08 - 0.42 * chamfer));
    return;
  }

  if (sys === 'Orthorhombic') {
    // 3 mutually perpendicular unequal axes (a != b != c, alpha = beta = gamma = 90°):
    // Short a-axis (0.68 r) along sx, wide b-axis (1.36 r) along sz, rhombic prism {110} + macrodomes {101} & brachydomes {011}
    const orthoAz = [0, 31 * DEG, 90 * DEG, 149 * DEG, 180 * DEG, 211 * DEG, 270 * DEG, 329 * DEG];
    const orthoRadii = [0.70, 0.82, 1.34, 0.82, 0.70, 0.82, 1.34, 0.82];
    for (let k = 0; k < 8; k++) {
      const a = orthoAz[k];
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
        orthoRadii[k] * fVar(k, 0.18), 0.5 * taperLy
      );
    }
    const tMacro = Math.min(76 * DEG, nodeTilt + 10 * DEG); // Steep macrodome {101} at 0°, 180°
    const tBrachy = Math.max(28 * DEG, nodeTilt - 16 * DEG); // Shallow chisel brachydome {011} at 90°, 270°
    const tBip = nodeTilt; // Rhombic bipyramid {111} at corners
    for (let j = 0; j < 8; j++) {
      const a = orthoAz[j];
      let tj = tBip;
      let baseOffR = 0.18 - 0.12 * chamfer;
      if (j === 0 || j === 4) {
        tj = tMacro;
        baseOffR = 0.04;
      } else if (j === 2 || j === 6) {
        tj = tBrachy;
        baseOffR = 0.14;
      }
      const ct = Math.cos(tj), st = Math.sin(tj);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        st * Math.cos(a), ct, st * Math.sin(a),
        baseOffR + tVar(j, 0.18), ct
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, 0.04, 0);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, Math.max(0.74, 0.98 - 0.34 * chamfer));
    return;
  }

  if (sys === 'Monoclinic') {
    // 1 oblique inclined clino-axis (a != b != c, alpha = gamma = 90°, beta ≈ 116° != 90°):
    // Flattened lath prism + obliquely slanted {001} termination & forward-sheared clinodomes {011}
    const monoAz = [0, 38 * DEG, 90 * DEG, 142 * DEG, 180 * DEG, 218 * DEG, 270 * DEG, 322 * DEG];
    const monoRadii = [0.60, 0.82, 1.38, 0.82, 0.60, 0.82, 1.38, 0.82];
    for (let k = 0; k < 8; k++) {
      const a = monoAz[k];
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
        monoRadii[k] * fVar(k, 0.20), 0.5 * taperLy
      );
    }
    const betaShear = 0.44; // sin(26°) monoclinic inclination along +sx (clino-axis a)
    for (let j = 0; j < 8; j++) {
      const a = monoAz[j];
      const isBack = Math.cos(a) < -0.15;
      const tj = isBack ? Math.max(26 * DEG, nodeTilt - 18 * DEG) : Math.min(72 * DEG, nodeTilt + 6 * DEG);
      const ct = Math.cos(tj), st = Math.sin(tj);
      const lx = st * Math.cos(a) + betaShear * ct;
      const ly = ct;
      const lz = st * Math.sin(a);
      const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
      const normLy = ly * inv;
      const offR = (isBack ? 0.28 : 0.12) + tVar(j, 0.20);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        lx * inv, normLy, lz * inv,
        offR, 0.90 * normLy
      );
    }
    // Lower & upper basal pinacoids {00-1} and {001} slanted at beta = 116° in the sx-ax plane (lz = 0 since alpha = gamma = 90°)
    const sinBeta = Math.sin(26 * DEG), cosBeta = Math.cos(26 * DEG);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, -sinBeta * 0.75, -cosBeta, 0, 0.28, 0);
    const monoTopH = Math.max(0.68, 0.84 - 0.18 * chamfer);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, sinBeta, cosBeta, 0, 0.08, monoTopH * cosBeta);
    return;
  }

  if (sys === 'Triclinic') {
    // All 3 axes and all 3 interaxial angles unequal & non-90° (a != b != c, alpha ≈ 72°, beta ≈ 108°, gamma ≈ 70°):
    // Skewed non-orthogonal parallelogram bladed prism + doubly-inclined asymmetric wedge termination
    const tricAz = [0, 34 * DEG, 70 * DEG, 128 * DEG, 180 * DEG, 214 * DEG, 250 * DEG, 308 * DEG];
    const tricRadii = [0.52, 0.78, 1.42, 1.06, 0.52, 0.78, 1.42, 1.06];
    const tricLySkew = [0.09, 0.08, 0.12, -0.07, -0.09, -0.08, -0.12, 0.07];
    for (let k = 0; k < 8; k++) {
      const a = tricAz[k];
      const ly = taperLy + tricLySkew[k];
      const ls = Math.sqrt(Math.max(0.1, 1 - ly * ly));
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        ls * Math.cos(a), ly, ls * Math.sin(a),
        tricRadii[k] * fVar(k, 0.22), 0.5 * ly
      );
    }
    const tricTiltOffsets = [8, -2, -14, 12, -18, -6, 6, -10];
    for (let j = 0; j < 8; j++) {
      const a = tricAz[j];
      const tj = Math.max(26 * DEG, Math.min(74 * DEG, nodeTilt + tricTiltOffsets[j] * DEG));
      const ct = Math.cos(tj), st = Math.sin(tj);
      // Doubly-oblique shear in both +sx (beta != 90°) and +sz (alpha != 90°)
      const lx = st * Math.cos(a) + 0.30 * ct;
      const ly = ct;
      const lz = st * Math.sin(a) + 0.26 * ct;
      const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
      const normLy = ly * inv;
      const offR = 0.14 + 0.06 * (j % 3) + tVar(j, 0.22);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        lx * inv, normLy, lz * inv,
        offR, 0.88 * normLy
      );
    }
    // Doubly-inclined basal pinacoids {00-1} and {001} (non-zero lx AND non-zero lz)
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, -0.34, -0.88, -0.33, 0.32, 0);
    const tricTopH = Math.max(0.66, 0.82 - 0.18 * chamfer);
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0.38, 0.84, 0.39, 0.10, tricTopH * 0.84);
    return;
  }

  // Default: Trigonal (Rhombohedral — 3-fold c-axis symmetry, e.g. Quartz, Corundum, Tourmaline)
  // Smoothly blends between 3-sided rounded ditrigonal Tourmaline prism (symVal ≈ 3) and 6-sided Quartz prism (symVal ≈ 6)
  const triRound = smooth01(Math.max(0, Math.min(1, (4.8 - symVal) / 1.5)));
  for (let k = 0; k < 6; k++) {
    const a = (k * TAU) / 6;
    const isMinorPrism = (k & 1) === 1;
    const baseR = isMinorPrism
      ? (1.0 + 0.16 * triRound + 0.04 * dom)
      : (1.0 - 0.16 * triRound - 0.04 * dom);
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, k,
      latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
      baseR * fVar(k, 0.35), 0.5 * taperLy
    );
  }
  // Accessory ditrigonal rounding bevels in slots 6..7 (at 90° and 270°)
  for (let m = 0; m < 2; m++) {
    const k = 6 + m;
    const a = (m * Math.PI) + 0.5 * Math.PI;
    const ditrigR = (1.10 * triRound + 1.28 * (1 - triRound)) * fVar(k, 0.20);
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, k,
      latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
      ditrigR, 0.5 * taperLy
    );
  }
  // 3-fold Rhombohedral termination: major r{10-11} (even j = 0,2,4) alternating with minor z{01-11} (odd j = 1,3,5)
  const effDom = Math.max(0.14, dom);
  const rTilt = nodeTilt;
  const zTilt = Math.min(82 * DEG, nodeTilt + (4 + 12 * effDom) * DEG);
  const cosR = Math.cos(rTilt), sinR = Math.sin(rTilt);
  const cosZ = Math.cos(zTilt), sinZ = Math.sin(zTilt);
  for (let j = 0; j < 6; j++) {
    const a = (j * TAU) / 6;
    const isMinorRhombo = (j & 1) === 1;
    const ct = isMinorRhombo ? cosZ : cosR;
    const st = isMinorRhombo ? sinZ : sinR;
    const offR = (isMinorRhombo ? (0.30 + 0.52 * effDom + 0.16 * triRound) : (0.04 - 0.18 * effDom)) + tVar(j, 0.32);
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, 8 + j,
      st * Math.cos(a), ct, st * Math.sin(a),
      offR, ct
    );
  }
  // Steep trigonal trapezohedron {51-61} shoulder facets in slots 14..15
  const trapTilt = Math.min(82 * DEG, nodeTilt + 14 * DEG);
  const cosT = Math.cos(trapTilt), sinT = Math.sin(trapTilt);
  for (let m = 0; m < 2; m++) {
    const a = (m * Math.PI) + 32 * DEG;
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, 14 + m,
      sinT * Math.cos(a), cosT, sinT * Math.sin(a),
      (0.58 - 0.22 * chamfer) + tVar(6 + m, 0.20), cosT
    );
  }
  const rootOffR = Math.max(0.02, effDom * 0.25 + Math.abs(tipJitterAmt * 0.15 * fs[16]));
  setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, rootOffR, 0);
  const tableFrac = Math.max(0.72, 1.14 - 0.54 * Math.max(0, chamfer));
  setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableFrac);
}

// Pre-allocated scratch buffers for target and morph-blended local crystal planes (0..17)
const tgtLx = new Float32Array(18), tgtLy = new Float32Array(18), tgtLz = new Float32Array(18);
const tgtOffR = new Float32Array(18), tgtOffL = new Float32Array(18);
const altLx = new Float32Array(18), altLy = new Float32Array(18), altLz = new Float32Array(18);
const altOffR = new Float32Array(18), altOffL = new Float32Array(18);

let nextUid = 0;
let spawnTimer = 0;
let masterTwist = 0;
let lastSeedVal = NaN;
let lastStaticCount = -1;

function rollFaceSeeds() {
  const fs = new Float32Array(24);
  for (let k = 0; k < 24; k++) fs[k] = sym();
  return fs;
}

function createInitialCohortNode(i, nBase, nPrimary, parentNode, startFromSeed, habit = 'Quartz Cluster', symmetry = 6) {
  const isSecondary = i >= nPrimary;
  const uid = i + 1;
  let birthThreshold = 0;
  let birthSpan = 0.56;
  if (i === 0) {
    birthThreshold = 0.0;
    birthSpan = 0.52;
  } else if (!isSecondary) {
    birthThreshold = 0.04 + 0.28 * (i / Math.max(1, nPrimary));
    birthSpan = 0.56;
  } else {
    birthThreshold = 0.24 + 0.36 * ((i - nPrimary) / Math.max(1, nBase - nPrimary));
    birthSpan = 0.38;
  }

  const uRho = rand();
  const rho = i === 0 ? 0.12 * uRho : Math.sqrt((i + 0.5 * uRho) / Math.max(1, nPrimary));
  const twinRoll = i < 2 ? 999 : 0.06 + 0.84 * (((i - 2) * 0.6180339887 + 0.23) % 1);
  const parentIdx = i < 2 ? -1 : ((i - 1) % Math.max(1, Math.min(4, i - 1)));
  const initSeedP = 0.24;
  const initGrow = startFromSeed
    ? (i === 0 ? Math.max(0.32, smooth01((initSeedP - birthThreshold) / birthSpan)) : smooth01((initSeedP - birthThreshold) / birthSpan))
    : 1.0;

  return {
    uid,
    id: uid,
    generation: isSecondary ? 1 : 0,
    age: 0,
    cohortIdx: i,
    isInitialCohort: true,
    isSecondary,
    parentIdx,
    parentUid: parentNode ? parentNode.uid : (parentIdx >= 0 ? parentIdx + 1 : -1),
    twinRoll,
    twinW: 0,
    smoothTwinW: 0,
    velTwinW: 0,
    presence: 1,
    smoothPresence: 1,
    velPresence: 0,
    smoothInit: false,
    frameInit: false,
    birthHabit: habit,
    birthSymmetry: symmetry,
    birthNPrimary: nPrimary,
    birthThreshold,
    birthSpan,
    growProgress: initGrow,
    dissolveProgress: 0,
    dissolveStartAlpha: 1.0,
    deadFrames: 0,
    alpha: startFromSeed ? smooth01(initGrow / 0.42) : 1.0,
    lifecycle: startFromSeed && initGrow < 0.999 ? 'spawning' : 'mature',
    // Morphological genes rolled once at birth so crystal shape is stable across all transitions
    uRho,
    rho,
    uAz: sym(),
    uLean: sym(),
    uLeanAz: sym(),
    uLen: rand(),
    uLenSym: sym(),
    uRadSym: sym(),
    uTwist: rand() * TAU,
    uRootU: sym(),
    uRootV: sym(),
    uStepSym: sym(),
    uScepterLen: rand(),
    uScepterRad: rand(),
    uParLen: rand(),
    uParRad: rand(),
    uParAttach: rand(),
    uBranchAz: sym(),
    uBranchAngle: rand(),
    uAttach: rand(),
    uBranchLen: rand(),
    uBranchRad: rand(),
    rnd: rand(),
    hueOffset: (uid * 0.16180339887) % 1,
    faceSeeds: rollFaceSeeds(),
    // Cached world frame (updated each frame so child branches can anchor smoothly)
    worldRoot: new Float32Array(3),
    worldAxis: new Float32Array([0, 1, 0]),
    worldSide: new Float32Array([1, 0, 0]),
    worldL: 0.5,
    worldFullL: 0.5,
    worldR: 0.08,
    worldTwist: 0,
    savedParentRoot: new Float32Array(3),
    savedParentAxis: new Float32Array([0, 1, 0]),
    savedParentFullL: 0.65,
    savedParentL: 0.65,
    savedParentR: 0.08,
    savedParentTwist: 0,
    driftOffset: new Float32Array(3),
  };
}

/**
 * Spawns a new frontier branch crystal during Phase 2 (`evolving`).
 * Branches off an active mature crystal (preferring newer frontier generations),
 * with an inward radial bounding envelope so the evolving creation stays centered.
 */
function spawnFrontierBranch(P) {
  if (cluster.nodes.length >= MAX_CRYSTALS) return null;
  const maturePool = cluster.nodes.filter((n) => n.lifecycle !== 'dissolving' && n.growProgress >= 0.45);
  const uid = ++nextUid;
  cluster.spawnedCount++;

  const habit = P?.habit ? String(P.habit.value) : 'Quartz Cluster';
  const symmetry = P?.symmetry ? Math.max(3, Math.min(6, Math.round(P.symmetry.value))) : 6;
  const desiredCount = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));
  const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
  const nPrimary = primaryCountFor(desiredCount, habit, twinShare);

  // Keep a healthy balance of full-sized primary crystals so the cluster never hollows out into tiny twigs
  let activePrimaries = 0;
  for (let i = 0; i < maturePool.length; i++) {
    if (!maturePool[i].isSecondary) activePrimaries++;
  }
  const minPrimaries = Math.max(4, Math.ceil(nPrimary * 0.6));
  const spawnCoreStem = maturePool.length === 0 || activePrimaries < minPrimaries || (cluster.spawnedCount % 2 === 0);
  if (spawnCoreStem) {
    const gen = cluster.generation + 1;
    cluster.generation = gen;
    const rho = 0.18 + 0.65 * Math.sqrt(rand());
    const node = {
      uid,
      id: uid,
      generation: gen,
      age: 0,
      cohortIdx: cluster.spawnedCount,
      isInitialCohort: false,
      isEvolvedPrimary: true,
      isSecondary: false,
      parentUid: -1,
      birthHabit: habit,
      birthSymmetry: symmetry,
      birthNPrimary: nPrimary,
      birthThreshold: 0,
      birthSpan: 0.5,
      growProgress: 0.0,
      dissolveProgress: 0.0,
      dissolveStartAlpha: 1.0,
      deadFrames: 0,
      alpha: 0.0,
      lifecycle: 'spawning',
      rho,
      uAz: sym(),
      uLean: sym(),
      uLeanAz: sym(),
      uLen: rand(),
      uLenSym: sym(),
      uRadSym: sym(),
      uTwist: rand() * TAU,
      uRootU: sym() * 0.6,
      uRootV: sym() * 0.6,
      uStepSym: sym(),
      uScepterLen: rand(),
      uScepterRad: rand(),
      uParLen: rand(),
      uParRad: rand(),
      uParAttach: rand(),
      uBranchAz: sym(),
      uBranchAngle: rand(),
      uAttach: rand(),
      uBranchLen: rand(),
      uBranchRad: rand(),
      rnd: rand(),
      hueOffset: (uid * 0.2180339887) % 1,
      faceSeeds: rollFaceSeeds(),
      worldRoot: new Float32Array(3),
      worldAxis: new Float32Array([0, 1, 0]),
      worldSide: new Float32Array([1, 0, 0]),
      worldL: 0.5,
      worldFullL: 0.5,
      worldR: 0.08,
      worldTwist: 0,
      savedParentRoot: new Float32Array(3),
      savedParentAxis: new Float32Array([0, 1, 0]),
      savedParentFullL: 0.65,
      savedParentL: 0.65,
      savedParentR: 0.08,
      savedParentTwist: 0,
      driftOffset: new Float32Array(3),
    };
    cluster.nodes.push(node);
    return node;
  }

  // Pick a parent weighted toward the newer frontier half of maturePool
  const startIdx = Math.floor(maturePool.length * 0.35 * rand());
  const parent = maturePool[Math.min(maturePool.length - 1, startIdx + Math.floor(rand() * (maturePool.length - startIdx)))];
  const gen = (parent.generation || 0) + 1;
  cluster.generation = Math.max(cluster.generation, gen);

  // If parent is already far from the origin, bias attachment closer to the base
  const pRootDist = Math.hypot(parent.worldRoot[0], parent.worldRoot[1], parent.worldRoot[2]);
  const attachRaw = pRootDist > 0.45 ? 0.18 + 0.28 * rand() : 0.28 + 0.46 * rand();

  const node = {
    uid,
    id: uid,
    generation: gen,
    age: 0,
    cohortIdx: cluster.spawnedCount,
    isInitialCohort: false,
    isEvolvedPrimary: false,
    isSecondary: true,
    parentUid: parent.uid,
    birthHabit: habit,
    birthSymmetry: symmetry,
    birthNPrimary: nPrimary,
    birthThreshold: 0,
    birthSpan: 0.5,
    growProgress: 0.0,
    dissolveProgress: 0.0,
    dissolveStartAlpha: 1.0,
    deadFrames: 0,
    alpha: 0.0,
    lifecycle: 'spawning',
    rho: Math.min(1, parent.rho + 0.12 * sym()),
    uAz: sym(),
    uLean: sym(),
    uLeanAz: sym(),
    uLen: rand(),
    uLenSym: sym(),
    uRadSym: sym(),
    uTwist: rand() * TAU,
    uRootU: sym(),
    uRootV: sym(),
    uStepSym: sym(),
    uScepterLen: rand(),
    uScepterRad: rand(),
    uParLen: rand(),
    uParRad: rand(),
    uParAttach: attachRaw,
    uBranchAz: sym(),
    uBranchAngle: rand(),
    uAttach: attachRaw,
    uBranchLen: 0.45 + 0.42 * rand(),
    uBranchRad: 0.68 + 0.26 * rand(),
    rnd: rand(),
    hueOffset: (uid * 0.2180339887) % 1,
    faceSeeds: rollFaceSeeds(),
    worldRoot: new Float32Array(3),
    worldAxis: new Float32Array([0, 1, 0]),
    worldSide: new Float32Array([1, 0, 0]),
    worldL: 0.5,
    worldFullL: 0.5,
    worldR: 0.08,
    worldTwist: 0,
    savedParentRoot: new Float32Array(parent.worldRoot),
    savedParentAxis: new Float32Array(parent.worldAxis),
    savedParentFullL: Math.max(0.58, Math.min(0.95, parent.worldFullL)),
    savedParentL: Math.max(0.52, parent.worldL),
    savedParentR: Math.max(0.065, Math.min(0.18, parent.worldR)),
    savedParentTwist: parent.worldTwist,
    driftOffset: new Float32Array(3),
  };

  cluster.nodes.push(node);
  return node;
}

const primRoot = new Float32Array(3), primAx = new Float32Array(3);
const twinRoot = new Float32Array(3), twinAx = new Float32Array(3);
const sphereCentres = new Float32Array(MAX_CRYSTALS * 4);

export function resetEvolution(P, _opts = {}) {
  const seedVal = P?.seed ? Number(P.seed.value) : 7;
  seedRandom(seedVal);
  lastSeedVal = seedVal;
  masterTwist = rand() * TAU;
  spawnTimer = 0;

  const evolveOn = P?.evolve ? !!P.evolve.value : true;
  const floatC = Math.max(1, Math.min(MAX_CRYSTALS, P?.crystals ? Number(P.crystals.value) : 9));
  lastStaticCount = floatC;
  const habit = P?.habit ? String(P.habit.value) : 'Quartz Cluster';
  const symmetry = P?.symmetry ? Math.max(3, Math.min(8, Number(P.symmetry.value))) : 6;
  const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
  const nPrimary = primaryCountFor(Math.round(floatC), habit, twinShare);

  cluster.nodes = [];
  cluster.evolveTime = 0;
  cluster.evoPhase = 0;
  cluster.evoBurstVel = 0;
  cluster.spawnedCount = MAX_CRYSTALS;
  cluster.dissolvedCount = 0;
  cluster.generation = 1;
  cluster.seedProgress = 1.0;
  cluster.phase = !evolveOn ? 'static' : 'evolving';

  // Pre-allocate all MAX_CRYSTALS nodes deterministically so float `crystals` count
  // smoothly scales crystal presence in/out without ever adding/removing/reassigning nodes.
  for (let i = 0; i < MAX_CRYSTALS; i++) {
    const pIdx = i < 2 ? -1 : ((i - 1) % Math.max(1, Math.min(4, i - 1)));
    const parentNode = pIdx >= 0 ? cluster.nodes[pIdx] : null;
    const node = createInitialCohortNode(i, MAX_CRYSTALS, nPrimary, parentNode, false, habit, symmetry);
    node.growProgress = 1.0;
    node.dissolveProgress = 0.0;
    node.alpha = 1.0;
    node.lifecycle = 'mature';
    cluster.nodes.push(node);
  }
}

export function triggerGrowBurst(_P) {
  cluster.evoBurstVel = Math.min(0.35, (cluster.evoBurstVel || 0) + 0.14);
  cluster.version++;
}

export function stepEvolution(P, dt, viewportW = 0, viewportH = 0) {
  if (cluster.nodes.length < MAX_CRYSTALS) {
    resetEvolution(P, { startFromSeed: false });
  }

  const safeDt = Math.max(0, Math.min(Number(dt) || 0, 30));
  cluster.pendingCenterDt = (cluster.pendingCenterDt || 0) + safeDt;
  cluster.pendingNodeDt = (cluster.pendingNodeDt || 0) + safeDt;

  const evolveOn = P?.evolve ? !!P.evolve.value : true;
  if (!evolveOn) {
    cluster.phase = 'static';
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].growProgress = 1.0;
      cluster.nodes[i].dissolveProgress = 0.0;
      cluster.nodes[i].alpha = 1.0;
      cluster.nodes[i].lifecycle = 'mature';
    }
    buildCluster(P, viewportW, viewportH);
    return;
  }

  cluster.phase = 'evolving';
  cluster.seedProgress = 1.0;

  let rem = safeDt;
  const evolveSpd = P?.evolveSpeed !== undefined ? Math.max(0, Number(P.evolveSpeed.value)) : 1;
  const growSpd = P?.growSpeed !== undefined ? Math.max(0, Number(P.growSpeed.value)) : 0.07;

  while (rem > 1e-6) {
    const stepDt = Math.min(rem, 0.2);
    rem -= stepDt;
    cluster.evolveTime += stepDt;

    let burstStep = 0;
    if (cluster.evoBurstVel > 1e-5) {
      burstStep = cluster.evoBurstVel * stepDt;
      cluster.evoBurstVel *= Math.exp(-1.6 * stepDt);
      if (cluster.evoBurstVel < 1e-4) cluster.evoBurstVel = 0;
    }

    cluster.evoPhase += stepDt * growSpd * evolveSpd * 2.2 + burstStep;

    for (let i = 0; i < cluster.nodes.length; i++) {
      const node = cluster.nodes[i];
      node.age += stepDt;
      node.growProgress = 1.0;
      node.dissolveProgress = 0.0;
      node.alpha = 1.0;
      node.lifecycle = 'mature';
    }
  }

  buildCluster(P, viewportW, viewportH);
}

export function buildCluster(P, viewportW = 0, viewportH = 0) {
  const seedVal = P?.seed ? Number(P.seed.value) : 7;
  const evolveOn = P?.evolve ? !!P.evolve.value : true;
  const floatC = Math.max(3.0, Math.min(MAX_CRYSTALS, P?.crystals ? Number(P.crystals.value) : 4.0));
  const nActive = Math.min(MAX_CRYSTALS, Math.ceil(floatC - 1e-4));
  cluster.floatCrystals = floatC;

  if (cluster.nodes.length < MAX_CRYSTALS || (!evolveOn && seedVal !== lastSeedVal)) {
    resetEvolution(P, { startFromSeed: false });
  } else if (seedVal !== lastSeedVal) {
    lastSeedVal = seedVal;
  }

  const vw = viewportW > 0 ? viewportW : (typeof window !== 'undefined' && window.innerWidth > 0 ? window.innerWidth : 1280);
  const vh = viewportH > 0 ? viewportH : (typeof window !== 'undefined' && window.innerHeight > 0 ? window.innerHeight : 800);
  const aspect = vw / Math.max(1, vh);
  const growSideways = aspect >= 1.0;
  cluster.aspect = aspect;
  cluster.growDirection = growSideways ? 'sideways' : 'upwards';

  const fovDeg = P.fov ? Number(P.fov.value) : 21;
  const D = P.distance ? Number(P.distance.value) : 2.9;
  const focal = 0.5 / Math.tan((fovDeg * DEG) / 2);
  const minDim = Math.max(1, Math.min(vw, vh));
  const halfFrustumMin = D / (2 * focal);
  const halfFrustumX = halfFrustumMin * (vw / minDim);
  const halfFrustumY = halfFrustumMin * (vh / minDim);

  const growth = P.growth ? Math.max(0.03, Math.min(4.5, Number(P.growth.value))) : 1;
  const dirProgress = 1.22 * Math.tanh(Math.max(0, growth - 0.75) * 0.72);
  const overScale = 1 + 0.48 * Math.tanh(Math.max(0, growth - 1.0) * 0.72);

  // Continuous float symmetry (3.0..8.0) for smooth 'opening-out' polygon transformation
  const symmetry = P.symmetry ? Math.max(3, Math.min(8, Number(P.symmetry.value))) : 6;
  const taperParam = P.taper ? Number(P.taper.value) : 0;
  const chamfer = P.chamfer ? Number(P.chamfer.value) : 0.12;
  const twinShare = P.twinShare ? Number(P.twinShare.value) : 0.3;
  const twinAngle = (P.twinAngle ? Number(P.twinAngle.value) : 52) * DEG;
  const cavityCurve = P.cavityCurve ? Number(P.cavityCurve.value) : 0;

  const fan = P.fan.value * DEG;
  const tilt = P.tipAngle.value * DEG;

  let minY = Infinity, maxY = -Infinity, maxR2 = 0;
  let weightedSumX = 0, weightedSumY = 0, weightedSumZ = 0, weightTotal = 0;

  const rawEffPrimaryCount = Math.max(1.8, floatC * (1.0 - 0.38 * twinShare));
  const nodeDt = Math.max(0, Math.min(cluster.pendingNodeDt || 0, 10));
  cluster.pendingNodeDt = 0;
  if (!Number.isFinite(cluster.smoothEffPrimaryCount)) {
    cluster.smoothEffPrimaryCount = rawEffPrimaryCount;
  } else if (nodeDt > 0) {
    cluster.smoothEffPrimaryCount += (rawEffPrimaryCount - cluster.smoothEffPrimaryCount) * (1 - Math.exp(-0.22 * nodeDt));
  }
  const effPrimaryCount = cluster.smoothEffPrimaryCount;
  let activeCount = 0;

  for (let idx = 0; idx < cluster.nodes.length; idx++) {
    if (activeCount >= MAX_CRYSTALS) break;
    const node = cluster.nodes[idx];
    const i = node.cohortIdx;
    const nodeSymmetry = symmetry;

    // Continuous presence & twinW with ultra-slow 2nd-order critically-damped spring (omega = 0.20..0.24 rad/s,
    // max velocity <= 0.065/s, taking ~15-18s — 3x longer than before) so appearing and disappearing
    // crystals bud and shrink very gradually across preset changes.
    const rawPresence = smooth01(Math.max(0.0, Math.min(1.0, floatC - i)));
    const rawTwinW = i < 2 ? 0.0 : smooth01(Math.max(0.0, Math.min(1.0, (twinShare - node.twinRoll + 0.44) / 0.88)));
    if (!node.smoothInit) {
      node.smoothPresence = rawPresence;
      node.velPresence = 0;
      node.smoothTwinW = rawTwinW;
      node.velTwinW = 0;
      node.smoothInit = true;
    } else if (nodeDt > 0) {
      let remN = nodeDt;
      const omegaN = 0.20 + 0.04 * (0.5 + 0.5 * (node.uLen || 0));
      const omegaN2 = omegaN * omegaN;
      const dampN = 2.0 * omegaN;
      const maxVelN = 0.065;
      while (remN > 1e-6) {
        const ndt = Math.min(remN, 0.05);
        remN -= ndt;
        node.velPresence += ((rawPresence - node.smoothPresence) * omegaN2 - dampN * node.velPresence) * ndt;
        if (node.velPresence > maxVelN) node.velPresence = maxVelN;
        else if (node.velPresence < -maxVelN) node.velPresence = -maxVelN;
        node.smoothPresence = Math.max(0, Math.min(1, node.smoothPresence + node.velPresence * ndt));

        node.velTwinW += ((rawTwinW - node.smoothTwinW) * omegaN2 - dampN * node.velTwinW) * ndt;
        if (node.velTwinW > maxVelN) node.velTwinW = maxVelN;
        else if (node.velTwinW < -maxVelN) node.velTwinW = -maxVelN;
        node.smoothTwinW = Math.max(0, Math.min(1, node.smoothTwinW + node.velTwinW * ndt));
      }
      if (rawPresence <= 1e-4 && node.smoothPresence < 1e-3) {
        node.smoothPresence = 0;
        node.velPresence = 0;
      } else if (rawPresence >= 1 - 1e-4 && node.smoothPresence > 1 - 1e-3) {
        node.smoothPresence = 1;
        node.velPresence = 0;
      }
    }

    const presence = node.smoothPresence;
    const twinW = node.smoothTwinW;
    node.presence = presence;
    node.twinW = twinW;
    node.isSecondary = twinW >= 0.5;

    // Combine node lifecycle growth with manual P.growth slider
    const manualProgress = (!evolveOn && growth < 1.0)
      ? smooth01((growth - node.birthThreshold * 0.85) / Math.max(0.15, node.birthSpan))
      : 1.0;
    const localGrowth = evolveOn ? node.growProgress : manualProgress;
    const gScale = (i === 0 && node.isInitialCohort ? Math.max(0.08, localGrowth) : localGrowth) * overScale;

    if (localGrowth <= 1e-4 || node.alpha <= 1e-4 || gScale <= 1e-4) {
      continue;
    }

    // Continuous whole-cluster organic phase waves for crystal i (strictly for axial L / radial r growth — zero axial rotation or tilt!)
    const phi = cluster.evoPhase || 0;
    const s1 = i * GOLDEN + node.uLen * TAU;
    const s2 = i * GOLDEN * 1.6180339887 + (node.uAz + 1) * Math.PI;
    const s3 = i * GOLDEN * 2.6180339887 + (node.uLean + 1) * Math.PI;
    const w1 = Math.sin(phi * 0.85 + s1);
    const w2 = Math.sin(phi * 0.53 + s2);
    const w3 = Math.cos(phi * 1.18 + s3);

    const lenWave = 0.62 * w1 + 0.38 * w3;
    const radWave = 0.64 * w2 - 0.36 * w1;
    const lenMod = 0.95 + 0.32 * lenWave;
    const radMod = 1.0 + 0.18 * radWave;

    // Compute target crystal unit axis (independent of evoPhase so crystals never wobble on their own axis),
    // and smoothly damp node.axis0 toward the target axis when fan, cavityCurve, twinAngle, or habit changes.
    const initPrimaryCount = Math.max(1.8, floatC * (1.0 - 0.38 * twinShare));
    const rho0 = i === 0 ? 0.12 * node.uRho : Math.min(1.0, Math.sqrt((i + 0.5 * node.uRho) / initPrimaryCount));
    node.rho = rho0;
    const az0 = i * GOLDEN + 0.55 * node.uAz;
    let lean0 = fan * Math.min(1, rho0 + 0.28 * node.uLean * (i > 0 ? 1 : 0.25));
    if (i === 0) lean0 = Math.max(0.02, lean0);

    if (growSideways) {
      if (i > 0) {
        const targetSideLean = Math.min(84 * DEG, Math.max(lean0, (56 + 26 * rho0) * DEG));
        lean0 = lean0 + (targetSideLean - lean0) * dirProgress * Math.min(1, 0.45 + 0.55 * rho0);
      }
    } else {
      lean0 = lean0 * (1 - 0.58 * dirProgress);
    }

    const leanAz0 = az0 + 0.35 * node.uLeanAz;
    primAx[0] = Math.sin(lean0) * Math.cos(leanAz0);
    primAx[1] = Math.cos(lean0);
    primAx[2] = Math.sin(lean0) * Math.sin(leanAz0);

    if (cavityCurve > 0.001) {
      const inTilt = cavityCurve * 0.45 * rho0;
      primAx[0] = primAx[0] * (1 - inTilt) - Math.cos(az0) * inTilt * 0.5;
      primAx[2] = primAx[2] * (1 - inTilt) - Math.sin(az0) * inTilt * 0.5;
      const al0 = Math.hypot(primAx[0], primAx[1], primAx[2]) || 1;
      primAx[0] /= al0; primAx[1] /= al0; primAx[2] /= al0;
    }

    if (!node.frameInit) {
      const isTwinNode = i >= 2 && rawTwinW >= 0.5;
      node.lockedTwinRole = isTwinNode ? 1 : 0;
      node.rot0 = node.uTwist;
      node.az0 = az0;
    }

    if (!node.lockedTwinRole) {
      ax[0] = primAx[0]; ax[1] = primAx[1]; ax[2] = primAx[2];
    } else {
      const pNode = cluster.nodes[node.parentIdx >= 0 ? node.parentIdx : 0];
      const pAxis = (pNode && pNode.axis0) ? pNode.axis0 : new Float32Array([0, 1, 0]);
      const pTwist = (pNode && Number.isFinite(pNode.rot0)) ? pNode.rot0 : 0;
      for (let k = 0; k < 3; k++) ax[k] = pAxis[k];
      buildSideBasis(pTwist);
      const branchAz = i * GOLDEN + 0.35 * node.uBranchAz;
      const bAngle = twinAngle * (0.85 + 0.3 * node.uBranchAngle);
      const cosB = Math.cos(bAngle), sinB = Math.sin(bAngle);
      const rx = sx[0] * Math.cos(branchAz) + sz[0] * Math.sin(branchAz);
      const ry = sx[1] * Math.cos(branchAz) + sz[1] * Math.sin(branchAz);
      const rz = sx[2] * Math.cos(branchAz) + sz[2] * Math.sin(branchAz);
      twinAx[0] = pAxis[0] * cosB + rx * sinB;
      twinAx[1] = pAxis[1] * cosB + ry * sinB;
      twinAx[2] = pAxis[2] * cosB + rz * sinB;
      const tal = Math.hypot(twinAx[0], twinAx[1], twinAx[2]) || 1;
      ax[0] = twinAx[0] / tal; ax[1] = twinAx[1] / tal; ax[2] = twinAx[2] / tal;
      if (!node.branchRadial0) node.branchRadial0 = new Float32Array(3);
      node.branchRadial0[0] = rx; node.branchRadial0[1] = ry; node.branchRadial0[2] = rz;
    }

    if (!node.frameInit || !node.axis0) {
      node.axis0 = new Float32Array(ax);
      buildSideBasis(node.rot0);
      node.u0 = new Float32Array(sx);
      node.v0 = new Float32Array(sz);
      node.frameInit = true;
    } else if (nodeDt > 0) {
      const aAxis = 1 - Math.exp(-2.8 * nodeDt);
      node.axis0[0] += (ax[0] - node.axis0[0]) * aAxis;
      node.axis0[1] += (ax[1] - node.axis0[1]) * aAxis;
      node.axis0[2] += (ax[2] - node.axis0[2]) * aAxis;
      const al = Math.hypot(node.axis0[0], node.axis0[1], node.axis0[2]) || 1;
      node.axis0[0] /= al; node.axis0[1] /= al; node.axis0[2] /= al;
      for (let k = 0; k < 3; k++) ax[k] = node.axis0[k];
      buildSideBasis(node.rot0);
      for (let k = 0; k < 3; k++) {
        node.u0[k] = sx[k];
        node.v0[k] = sz[k];
      }
    }

    for (let k = 0; k < 3; k++) {
      ax[k] = node.axis0[k];
      sx[k] = node.u0[k];
      sz[k] = node.v0[k];
    }
    const tw = node.rot0;
    const rho = node.rho;
    const az = node.az0 || (i * GOLDEN + 0.55 * node.uAz);

    let primBaseL = (i === 0 ? 1 : Math.max(0.18, 1 - P.lengthJitter.value * (0.4 * node.uLen + 0.6 * rho))) * lenMod;
    const horizComp = Math.hypot(ax[0], ax[2]);
    const vertComp = Math.abs(ax[1]);
    if (growSideways) {
      primBaseL *= 1 + dirProgress * (0.78 * horizComp * Math.min(1.9, aspect) - 0.16 * vertComp);
    } else {
      primBaseL *= 1 + dirProgress * (0.85 * vertComp * Math.min(1.9, 1 / aspect) - 0.16 * horizComp);
    }

    const primBaseR = P.thickness.value * Math.pow(primBaseL, 0.65) * Math.max(0.25, 1 + P.thicknessJitter.value * 0.55 * node.uRadSym) * radMod;

    let baseL = primBaseL;
    let baseR = primBaseR;

    if (!node.lockedTwinRole) {
      const curPrimL = primBaseL * gScale * presence;
      const curPrimR = primBaseR * Math.pow(gScale, 0.78) * presence;
      const back = P.crossAt.value * curPrimL;
      const u = node.uRootU * curPrimR * 1.05, v = node.uRootV * curPrimR * 1.05;
      for (let k = 0; k < 3; k++) {
        root[k] = -ax[k] * back + sx[k] * u + sz[k] * v + node.driftOffset[k];
      }
      if (dirProgress > 0.01 && i > 0) {
        if (growSideways) {
          const latDist = dirProgress * halfFrustumX * 0.96 * rho * localGrowth * presence;
          root[0] += Math.cos(az) * latDist;
          root[2] += Math.sin(az) * latDist;
          root[1] *= (1 - 0.42 * dirProgress);
        } else {
          const tier = ((i % 2 === 1) ? 1 : -0.88) * Math.pow(rho, 0.78);
          const vertDist = dirProgress * halfFrustumY * 0.98 * tier * localGrowth * presence;
          root[1] += vertDist;
          root[0] *= (1 - 0.42 * dirProgress);
          root[2] *= (1 - 0.42 * dirProgress);
        }
      }
      if (cavityCurve > 0.001) {
        const bowlR = 0.52 * cavityCurve * (0.35 + 0.65 * gScale) * presence;
        const ringR = bowlR * rho;
        const bowlY = -bowlR * Math.cos(rho * 1.15) + bowlR * 0.45;
        root[0] += Math.cos(az) * ringR;
        root[1] += bowlY;
        root[2] += Math.sin(az) * ringR;
      }
    } else {
      const pNode = cluster.nodes[node.parentIdx >= 0 ? node.parentIdx : 0];
      if (pNode && pNode.worldL > 0.005) {
        for (let k = 0; k < 3; k++) {
          node.savedParentRoot[k] = pNode.worldRoot[k];
          node.savedParentAxis[k] = pNode.worldAxis[k];
        }
        node.savedParentL = pNode.worldL;
        node.savedParentFullL = Math.max(0.35, Math.min(0.95, pNode.worldFullL));
        node.savedParentR = Math.max(0.035, Math.min(0.18, pNode.worldR));
      }
      const pRoot = node.savedParentRoot;
      const pAxis = node.savedParentAxis;
      const pL = node.savedParentL;
      const pFullL = node.savedParentFullL;
      const pR = node.savedParentR;
      const br = node.branchRadial0 || sx;
      const attachFrac = Math.max(0.14, Math.min(0.78, 0.24 + 0.44 * node.uAttach));
      const attachDist = pL * attachFrac;
      for (let k = 0; k < 3; k++) {
        root[k] = pRoot[k] + pAxis[k] * attachDist + br[k] * (pR * 0.35) + node.driftOffset[k];
      }
      baseL = pFullL * (0.44 + 0.32 * node.uBranchLen) * (1 + 0.32 * dirProgress) * lenMod;
      baseR = Math.max(0.035, P.thickness.value * 0.85) * (0.68 + 0.28 * node.uBranchRad) * radMod;
    }

    // Pull emerging/shrinking crystals smoothly into the heart of the cluster core (0, 0, 0)
    // so disappearing crystals retract inside the permanent core (crystals 0..2) before vanishing.
    const corePull = i < 3 ? 1.0 : smooth01(Math.min(1.0, presence / 0.42));
    root[0] *= corePull;
    root[1] *= corePull;
    root[2] *= corePull;

    // Apply procedural growth scaling and continuous crystal count presence to L and r
    const L = Math.max(1e-4, baseL * gScale * presence);
    const r = Math.max(1e-4, baseR * Math.pow(gScale, 0.78) * presence);

    // Save computed world frame on node for child branches
    for (let k = 0; k < 3; k++) {
      node.worldRoot[k] = root[k];
      node.worldAxis[k] = ax[k];
      node.worldSide[k] = sx[k];
    }
    node.worldL = presence < 2e-3 ? 0 : L;
    node.worldFullL = baseL * presence;
    node.worldR = presence < 2e-3 ? 0 : r;
    node.worldTwist = tw;
    if (presence < 2e-3) {
      continue;
    }

    const rowIdx = activeCount++;
    const fs = node.faceSeeds;
    let rMax = r * 1.45;

    const taperLy = Math.max(-0.32, Math.min(0.32, taperParam * 0.38));
    const latScale = Math.sqrt(Math.max(0.1, 1 - taperLy * taperLy));
    const faceJitterAmt = Math.max(0.08, P.faceJitter.value);
    const nodeTilt = Math.max(20 * DEG, Math.min(84 * DEG, tilt));
    const cosNT = Math.cos(nodeTilt);
    const dynDom = P.faceDominance.value;
    const dynTipJitter = Math.max(0.06, P.tipJitter.value);
    const dynChamfer = Math.max(0, chamfer);

    const curHabit = P.habit ? String(P.habit.value) : 'Trigonal';
    const morphBlend = Number(cluster.morphHabitBlend) || 0;
    const hFrom = cluster.morphHabitFrom ? resolveHabitSystem(cluster.morphHabitFrom) : '';
    const hTo = cluster.morphHabitTo ? resolveHabitSystem(cluster.morphHabitTo) : '';

    if (morphBlend > 1e-4 && morphBlend < 1 - 1e-4 && hFrom && hTo && hFrom !== hTo) {
      computeSystemPlanes(
        hFrom, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r,
        tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
      );
      computeSystemPlanes(
        hTo, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r,
        altLx, altLy, altLz, altOffR, altOffL
      );
      for (let k = 0; k < 18; k++) {
        const bx = tgtLx[k] + (altLx[k] - tgtLx[k]) * morphBlend;
        const by = tgtLy[k] + (altLy[k] - tgtLy[k]) * morphBlend;
        const bz = tgtLz[k] + (altLz[k] - tgtLz[k]) * morphBlend;
        const inv = 1 / (Math.hypot(bx, by, bz) || 1);
        tgtLx[k] = bx * inv;
        tgtLy[k] = by * inv;
        tgtLz[k] = bz * inv;
        tgtOffR[k] = tgtOffR[k] + (altOffR[k] - tgtOffR[k]) * morphBlend;
        tgtOffL[k] = tgtOffL[k] + (altOffL[k] - tgtOffL[k]) * morphBlend;
      }
    } else {
      computeSystemPlanes(
        curHabit, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r,
        tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
      );
    }

    if (!node.planesInit || !node.smoothLx) {
      node.smoothLx = new Float32Array(tgtLx);
      node.smoothLy = new Float32Array(tgtLy);
      node.smoothLz = new Float32Array(tgtLz);
      node.smoothOffR = new Float32Array(tgtOffR);
      node.smoothOffL = new Float32Array(tgtOffL);
      node.planesInit = true;
    } else if (nodeDt > 0) {
      const aPlane = 1 - Math.exp(-3.0 * nodeDt);
      for (let k = 0; k < 18; k++) {
        const nx = node.smoothLx[k] + (tgtLx[k] - node.smoothLx[k]) * aPlane;
        const ny = node.smoothLy[k] + (tgtLy[k] - node.smoothLy[k]) * aPlane;
        const nz = node.smoothLz[k] + (tgtLz[k] - node.smoothLz[k]) * aPlane;
        const inv = 1 / (Math.hypot(nx, ny, nz) || 1);
        node.smoothLx[k] = nx * inv;
        node.smoothLy[k] = ny * inv;
        node.smoothLz[k] = nz * inv;
        node.smoothOffR[k] += (tgtOffR[k] - node.smoothOffR[k]) * aPlane;
        node.smoothOffL[k] += (tgtOffL[k] - node.smoothOffL[k]) * aPlane;
      }
    }

    for (let k = 0; k < 18; k++) {
      const offVal = Math.max(-0.95 * L, node.smoothOffR[k] * r + node.smoothOffL[k] * L);
      if (k < 8) {
        rMax = Math.max(rMax, Math.abs(offVal));
      }
      crystalPlane(rowIdx, k, node.smoothLx[k], node.smoothLy[k], node.smoothLz[k], offVal);
    }

    // Slots 18..20: Literal 3D geometric corner-nip half-space planes (scaled by P.chipSize and macro faders)
    const macroEdgeScale = (P.patinaMacro ? Number(P.patinaMacro.value) : 1.0) * (P.edgeMacro ? Number(P.edgeMacro.value) : 1.0);
    const edgeChipsVal = Math.max(0, Math.min(2, (P.edgeChips ? Number(P.edgeChips.value) : 0.28) * macroEdgeScale));
    const chipSizeVal = P.chipSize ? Math.max(0.1, Math.min(3.0, Number(P.chipSize.value))) : 1.5;
    const rc = rMax * 1.25;
    crystalPlane(rowIdx, 18, 0, 1, 0, L * 12 + r * 12);
    crystalPlane(rowIdx, 19, 0, 1, 0, L * 12 + r * 12);
    crystalPlane(rowIdx, 20, 0, 1, 0, L * 12 + r * 12);

    const nipRoll1 = ((node.uid * 37 + 7) % 19) / 19;
    const nipRoll2 = ((node.uid * 53 + 3) % 23) / 23;
    const nipRoll3 = ((node.uid * 29 + 5) % 17) / 17;
    const rawChip1 = smooth01(Math.max(0, Math.min(1, (edgeChipsVal * 1.8 - nipRoll1 * 0.35 + 0.12) / 0.55)));
    const rawChip2 = smooth01(Math.max(0, Math.min(1, (edgeChipsVal * 1.7 - nipRoll2 * 0.42 + 0.12) / 0.55)));
    const rawChip3 = smooth01(Math.max(0, Math.min(1, (edgeChipsVal * 1.6 - nipRoll3 * 0.42 + 0.12) / 0.55)));

    if (!Number.isFinite(node.smoothChip1)) {
      node.smoothChip1 = rawChip1;
      node.smoothChip2 = rawChip2;
      node.smoothChip3 = rawChip3;
    } else if (nodeDt > 0) {
      const aChip = 1 - Math.exp(-0.22 * nodeDt);
      const maxStepChip = 0.065 * nodeDt;
      const d1 = rawChip1 - node.smoothChip1;
      node.smoothChip1 += Math.max(-maxStepChip, Math.min(maxStepChip, d1 * aChip));
      const d2 = rawChip2 - node.smoothChip2;
      node.smoothChip2 += Math.max(-maxStepChip, Math.min(maxStepChip, d2 * aChip));
      const d3 = rawChip3 - node.smoothChip3;
      node.smoothChip3 += Math.max(-maxStepChip, Math.min(maxStepChip, d3 * aChip));
    }

    // Lock corner-nip indices to fixed faces in {0, 1, 2} so they NEVER jump!
    const shVert1 = Math.max(0.08, Math.min(0.65, cosNT * 0.52));
    const shHoriz1 = Math.sqrt(Math.max(0.1, 1 - shVert1 * shVert1));
    const shVert2 = Math.max(0.08, Math.min(0.65, cosNT * 0.58));
    const shHoriz2 = Math.sqrt(Math.max(0.1, 1 - shVert2 * shVert2));
    const tipVert = Math.min(0.94, cosNT + (1.0 - cosNT) * 0.45);
    const tipHoriz = Math.sqrt(Math.max(0.05, 1 - tipVert * tipVert));

    const k0 = node.uid % 2;
    const k1 = k0 + 1;
    const k2 = (k0 + 1) % 3;
    const k3 = (k2 + 1) % 3;
    const kt0 = 1;
    const kt1 = 2;

    const whaleRoll1 = cornerWhaleRoll(node.uid, 18);
    const whaleRoll2 = cornerWhaleRoll(node.uid, 19);
    const whaleRoll3 = cornerWhaleRoll(node.uid, 20);
    const whaleMult1 = whaleRoll1 >= 0.99 ? 20.0 : (whaleRoll1 >= 0.93 ? 10.0 : 1.0);
    const whaleMult2 = whaleRoll2 >= 0.99 ? 20.0 : (whaleRoll2 >= 0.93 ? 10.0 : 1.0);
    const whaleMult3 = whaleRoll3 >= 0.99 ? 20.0 : (whaleRoll3 >= 0.93 ? 10.0 : 1.0);

    const nipSizeMult1 = (0.40 + 1.25 * Math.pow(((node.uid * 41 + 11) % 29) / 28, 1.35)) * whaleMult1;
    const nipSizeMult2 = (0.40 + 1.25 * Math.pow(((node.uid * 59 + 17) % 31) / 30, 1.35)) * whaleMult2;
    const nipSizeMult3 = (0.40 + 1.25 * Math.pow(((node.uid * 67 + 23) % 37) / 36, 1.35)) * whaleMult3;

    // Exact vertices of this crystal (planes 0..17) so every corner nip is a cap measured from the
    // crystal's true support in its direction: continuous, always a corner, never culls the crystal.
    const vertCount = enumerateCrystalVertices(rowIdx);

    const nipDepth1 = r * (0.0045 + 0.0105 * edgeChipsVal) * chipSizeVal * nipSizeMult1 * (whaleMult1 > 1.0 ? Math.max(0.65, node.smoothChip1) : node.smoothChip1);
    writeCornerNipPlane(rowIdx, 18, k0, k1, shHoriz1, shVert1, nipDepth1, 0.85, 1.15, vertCount, L, r);

    const nipDepth2 = r * (0.0040 + 0.0095 * edgeChipsVal) * chipSizeVal * nipSizeMult2 * (whaleMult2 > 1.0 ? Math.max(0.65, node.smoothChip2) : node.smoothChip2);
    writeCornerNipPlane(rowIdx, 19, k2, k3, shHoriz2, shVert2, nipDepth2, 1.12, 0.88, vertCount, L, r);

    const nipDepth3 = r * (0.0036 + 0.0088 * edgeChipsVal) * chipSizeVal * nipSizeMult3 * (whaleMult3 > 1.0 ? Math.max(0.65, node.smoothChip3) : node.smoothChip3);
    writeCornerNipPlane(rowIdx, 20, kt0, kt1, tipHoriz, tipVert, nipDepth3, 0.90, 1.10, vertCount, L, r);

    const half = 0.5 * L;
    const nodeAlpha = Math.max(0.0, Math.min(1.0, node.alpha * smooth01(Math.min(1.0, presence / 0.55))));
    const scx = root[0] + ax[0] * half;
    const scy = root[1] + ax[1] * half;
    const scz = root[2] + ax[2] * half;
    let maxVertDist2 = 0;
    for (let v = 0; v < vertCount; v++) {
      const dvx = POLY_V[v * 3] - scx;
      const dvy = POLY_V[v * 3 + 1] - scy;
      const dvz = POLY_V[v * 3 + 2] - scz;
      const d2 = dvx * dvx + dvy * dvy + dvz * dvz;
      if (d2 > maxVertDist2) maxVertDist2 = d2;
    }
    const sr = Math.max(Math.hypot(half + r * 0.65, rc), Math.sqrt(maxVertDist2)) * 1.05;
    sphereCentres[rowIdx * 4] = scx;
    sphereCentres[rowIdx * 4 + 1] = scy;
    sphereCentres[rowIdx * 4 + 2] = scz;
    sphereCentres[rowIdx * 4 + 3] = sr;

    slot(rowIdx, 0, scx, scy, scz, sr);
    slot(rowIdx, 1, ax[0], ax[1], ax[2], L);
    slot(rowIdx, 2, nodeAlpha, node.hueOffset, node.uid, r);
    slot(rowIdx, 3, root[0], root[1], root[2], node.rnd);

    // Smooth volume-weighted 3D centroid so emerging/shrinking crystals have zero derivative at birth/death
    const w = (nodeAlpha * nodeAlpha) * (presence * presence) * Math.max(1e-4, L * r * r);
    weightedSumX += scx * w;
    weightedSumY += scy * w;
    weightedSumZ += scz * w;
    weightTotal += w;

    const tipY = root[1] + ax[1] * L;
    const effPad = r * (0.1 + 0.35 * nodeAlpha);
    minY = Math.min(minY, Math.min(root[1], tipY) - effPad);
    maxY = Math.max(maxY, Math.max(root[1], tipY) + effPad);
    const rTip = Math.hypot(root[0] + ax[0] * L, root[2] + ax[2] * L) + effPad;
    const rRoot = Math.hypot(root[0], root[2]) + effPad;
    maxR2 = Math.max(maxR2, rTip * rTip, rRoot * rRoot);
  }

  cluster.crystals = Math.max(1, activeCount);
  cluster.activeVectors = cluster.crystals * CRYSTAL_PLANES;

  const targetCx = weightTotal > 1e-6 ? weightedSumX / weightTotal : 0;
  const targetCy = weightTotal > 1e-6 ? weightedSumY / weightTotal : (Number.isFinite(minY) ? 0.5 * (minY + maxY) : 0);
  const targetCz = weightTotal > 1e-6 ? weightedSumZ / weightTotal : 0;
  cluster.targetCentreX = targetCx;
  cluster.targetCentreY = targetCy;
  cluster.targetCentreZ = targetCz;

  if (!cluster.centerInit || !Number.isFinite(cluster.centreY)) {
    cluster.centreX = targetCx;
    cluster.centreY = targetCy;
    cluster.centreZ = targetCz;
    cluster.velCentreX = 0;
    cluster.velCentreY = 0;
    cluster.velCentreZ = 0;
    cluster.centerInit = true;
    cluster.pendingCenterDt = 0;
  } else {
    // Slow critically-damped 2nd-order spring-damper (true ease-in from rest + smooth ease-out deceleration, tau ~ 2.2s)
    let remC = Math.max(0, Math.min(cluster.pendingCenterDt > 0 ? cluster.pendingCenterDt : (!evolveOn ? 0.016 : 0), 10));
    cluster.pendingCenterDt = 0;
    const omega = 1.35;
    const omega2 = omega * omega;
    const damp = 2.0 * omega;
    while (remC > 1e-6) {
      const cdt = Math.min(remC, 0.05);
      remC -= cdt;
      cluster.velCentreX += ((targetCx - cluster.centreX) * omega2 - damp * cluster.velCentreX) * cdt;
      cluster.centreX += cluster.velCentreX * cdt;
      cluster.velCentreY += ((targetCy - cluster.centreY) * omega2 - damp * cluster.velCentreY) * cdt;
      cluster.centreY += cluster.velCentreY * cdt;
      cluster.velCentreZ += ((targetCz - cluster.centreZ) * omega2 - damp * cluster.velCentreZ) * cdt;
      cluster.centreZ += cluster.velCentreZ * cdt;
    }
  }

  let maxCamReach = 0.45;
  for (let k = 0; k < activeCount; k++) {
    const dx = sphereCentres[k * 4] - cluster.centreX;
    const dy = sphereCentres[k * 4 + 1] - cluster.centreY;
    const dz = sphereCentres[k * 4 + 2] - cluster.centreZ;
    const reach = Math.hypot(dx, dy, dz) + sphereCentres[k * 4 + 3];
    if (reach > maxCamReach) maxCamReach = reach;
  }

  const targetRadiusXZ = Math.max(0.18, Math.sqrt(maxR2));
  const targetHalfY = Math.max(0.18, Number.isFinite(minY) ? 0.5 * (maxY - minY) : 0.45);
  cluster.radiusXZ = Number.isFinite(cluster.radiusXZ)
    ? cluster.radiusXZ + (targetRadiusXZ - cluster.radiusXZ) * 0.04
    : targetRadiusXZ;
  cluster.halfY = Number.isFinite(cluster.halfY)
    ? cluster.halfY + (targetHalfY - cluster.halfY) * 0.04
    : targetHalfY;
  cluster.radius = Math.max(0.45, maxCamReach * 1.06, cluster.radiusXZ * 1.18, cluster.halfY * 1.18);
  cluster.version++;

  const projHalfX = (focal * cluster.radiusXZ) / Math.max(0.6, D);
  const projHalfY = (focal * cluster.halfY) / Math.max(0.6, D);
  const screenHalfW = 0.68 * (vw / minDim);
  const screenHalfH = 0.68 * (vh / minDim);
  cluster.fillX = projHalfX / Math.max(0.1, screenHalfW);
  cluster.fillY = projHalfY / Math.max(0.1, screenHalfH);
  cluster.screenFill = growSideways ? cluster.fillX : cluster.fillY;
}

cluster.resetEvolution = resetEvolution;
cluster.stepEvolution = stepEvolution;
cluster.applyManualParam = (key, P) => {
  cluster.morphHabitBlend = 0;
  if (key === 'habit' && P?.habit) {
    const h = String(P.habit.value);
    const desiredCount = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));
    const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
    const np = primaryCountFor(desiredCount, h, twinShare);
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].birthHabit = h;
      cluster.nodes[i].birthNPrimary = np;
    }
  } else if (key === 'symmetry' && P?.symmetry) {
    const s = Math.max(3, Math.min(8, Math.round(Number(P.symmetry.value))));
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].birthSymmetry = s;
    }
  }
};
cluster.snapToTarget = (P) => {
  cluster.morphHabitBlend = 0;
  for (let i = 0; i < cluster.nodes.length; i++) {
    cluster.nodes[i].planesInit = false;
    cluster.nodes[i].frameInit = false;
    cluster.nodes[i].smoothInit = false;
  }
  cluster.centerInit = false;
  if (P) buildCluster(P);
};
cluster.triggerGrowBurst = triggerGrowBurst;
cluster.surgeBranch = triggerGrowBurst;

