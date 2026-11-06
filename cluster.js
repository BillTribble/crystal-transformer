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
    if (d[o + 3] > 1e4) continue;
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

/**
 * Magnetic overlap repulsion weight by crystal system (post-015442, post-015454):
 * - Normal prismatic crystals (Trigonal & Hexagonal: Classic Quartz Prism, Amethyst Cathedral,
 *   Rutilated Smoky Scepter, Glacial Ice, Emerald, Tourmaline, Ruby Druzy, Aquamarine) = 0.0
 *   so their shared radiating cluster base is 100% untouched.
 * - Cubic / Isometric (Diamond) = 1.0 so equant octahedra don't clump concentric at the origin.
 * - Chunky non-hexagonal systems (Tetragonal, Orthorhombic, Monoclinic, Triclinic) = 0.50..0.62
 *   so heavily overlapping polyhedra push apart cleanly when sitting on top of each other.
 */
function habitRepelWeight(habit) {
  const sys = resolveHabitSystem(habit);
  if (sys === 'Cubic / Isometric') return 1.0;
  return 0.0;
}

const activeNodes = new Array(MAX_CRYSTALS);
const initRootX = new Float32Array(MAX_CRYSTALS);
const initRootY = new Float32Array(MAX_CRYSTALS);
const initRootZ = new Float32Array(MAX_CRYSTALS);

function setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, k, lx, ly, lz, offR, offL) {
  const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
  outLx[k] = lx * inv;
  outLy[k] = ly * inv;
  outLz[k] = lz * inv;
  outOffR[k] = offR;
  outOffL[k] = offL;
}

const cubeLx = new Float32Array(18), cubeLy = new Float32Array(18), cubeLz = new Float32Array(18);
const cubeOffR = new Float32Array(18), cubeOffL = new Float32Array(18);

/**
 * Orthogonal / pseudo-cubic & non-cubic Bismuth Hopper block planes (`prismSides = 4`, flat top/bottom)
 * used for Spanish Navajún Pyrite Cubes (`FeS2`, equant 1:1:1 cubes) and varied 3D Bismuth Hopper
 * morphologies (`Bi`, cantilevered wings, tall pagoda spires, L-brackets, parallel-ribbed slabs & fin bars).
 */
function computeCubeBlockPlanes(
  hopperBlend, meltBlend, morphType, chamfer, faceJitterAmt, fs, L, r, doubleTermBlend,
  outLx, outLy, outLz, outOffR, outOffL
) {
  const safeL = Math.max(L, 1e-4);
  const safeR = Math.max(r, 1e-4);
  const dtW = typeof doubleTermBlend === 'number'
    ? Math.max(0.0, Math.min(1.0, doubleTermBlend))
    : 0.0;
  const isoScale = Math.max(1.05, Math.min(4.2, (0.36 * safeL) / safeR));
  const cos45 = 0.70710678;
  const cubeHalf = isoScale * 0.92;

  // Non-cubic 3D aspect ratios (scaleX, scaleY, scaleZ) & trapezoidal flare per Bismuth morphology (0..4):
  //   0: Broad Cantilevered BGS Hopper Wing (wide horizontal span, shallower depth, upward trapezoidal flare)
  //   1: Tall Stepped Pagoda Spire / Ziggurat Tower (elongated vertical column tapering upward)
  //   2: Open-Sided L-Bracket / Greek-Key Tower (tall rectangular plate)
  //   3: Wide Parallel-Ribbed Terrace Slab ("Rainbow Metal" flat cantilevered comb plate)
  //   4: Stepped Rhombohedral Fin Bar / Twin Bracket (narrow front, deep side profile)
  const mType = ((morphType | 0) % 5 + 5) % 5;
  const biScaleX = mType === 0 ? 1.46 : mType === 1 ? 0.78 : mType === 2 ? 1.16 : mType === 3 ? 1.44 : 0.68;
  const biScaleY = mType === 0 ? 1.02 : mType === 1 ? 1.56 : mType === 2 ? 1.30 : mType === 3 ? 0.70 : 1.34;
  const biScaleZ = mType === 0 ? 0.62 : mType === 1 ? 0.76 : mType === 2 ? 0.66 : mType === 3 ? 0.54 : 1.12;
  const biTrapY  = mType === 0 ? 0.085 : mType === 1 ? -0.095 : mType === 2 ? 0.045 : mType === 3 ? -0.055 : 0.065;

  // Orthorhombic / truncated-bipyramidal Gallium block proportions (matching reference photo 372c3f66)
  const gaScaleX = mType === 0 ? 0.96 : mType === 1 ? 1.08 : mType === 2 ? 0.88 : mType === 3 ? 0.94 : 1.02;
  const gaScaleY = mType === 0 ? 1.34 : mType === 1 ? 1.16 : mType === 2 ? 1.12 : mType === 3 ? 1.04 : 1.18;
  const gaScaleZ = mType === 0 ? 0.88 : mType === 1 ? 0.96 : mType === 2 ? 0.84 : mType === 3 ? 0.86 : 0.92;
  const gaTrapY  = -0.085;

  const scaleX = 1.0 + (biScaleX - 1.0) * hopperBlend + (gaScaleX - 1.0) * meltBlend * (1.0 - hopperBlend);
  const scaleY = 1.0 + (biScaleY - 1.0) * hopperBlend + (gaScaleY - 1.0) * meltBlend * (1.0 - hopperBlend);
  const scaleZ = 1.0 + (biScaleZ - 1.0) * hopperBlend + (gaScaleZ - 1.0) * meltBlend * (1.0 - hopperBlend);
  const hx = cubeHalf * scaleX;
  const hy = cubeHalf * scaleY;
  const hz = cubeHalf * scaleZ;

  // Express cube offsets in L-units (`outOffL`) with small `outOffR` so plane smoothing is linear across r/L changes.
  const rOverL = safeR / safeL;
  const extraChamfer = Math.max(0.0, chamfer - 0.08);
  const fCubeVar = (k, scale = 0.07) => Math.max(0.88, Math.min(1.12, 1 + faceJitterAmt * scale * fs[k]));

  // Slots 0..6: Lateral side faces (0, 2, 4, 6 = cardinal +sx, +sz, -sx, -sz read by shader.js; 1, 3, 5 = vertical {110} corner bevels)
  for (let m = 0; m < 4; m++) {
    const aCube = m * 0.5 * Math.PI;
    const kCube = m * 2;
    const isXFace = (m & 1) === 0;
    const faceHalf = isXFace ? hx : hz;
    const rhoShear = (isXFace ? biTrapY : -0.042) * hopperBlend + gaTrapY * meltBlend * (1.0 - hopperBlend);
    const dSideL = faceHalf * fCubeVar(kCube, 0.08) * rOverL;
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, kCube,
      Math.cos(aCube), rhoShear, Math.sin(aCube),
      0.0, dSideL + 0.52 * rhoShear
    );
    if (m < 3) {
      const aCorner = (m + 0.5) * 0.5 * Math.PI;
      const kCorner = m * 2 + 1;
      const cornerDistL = (hx + hz) * cos45 * (0.988 - 0.056 * chamfer) * fCubeVar(kCorner, 0.05) * rOverL;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kCorner,
        Math.cos(aCorner), rhoShear, Math.sin(aCorner),
        0.0, cornerDistL + 0.52 * rhoShear
      );
    }
  }

  // Slots 8..13: 6 TOP bevel/crown planes (ly > 0) matching the 6 top termination slots of computeSystemPlanes
  for (let idx = 0; idx < 6; idx++) {
    const kTop = 8 + idx;
    const aTop = (idx * TAU) / 6;
    const faceHalf = Math.abs(Math.cos(aTop)) * hx + Math.abs(Math.sin(aTop)) * hz;
    const asymmBevel = (idx === 1 || idx === 2 || idx === 4) ? 1.18 : 0.88;
    const dodDistL = (faceHalf + hy) * cos45 * (0.988 - 0.056 * chamfer - 0.52 * extraChamfer * asymmBevel) * fCubeVar(kTop, 0.05) * rOverL;
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, kTop,
      cos45 * Math.cos(aTop), cos45, cos45 * Math.sin(aTop),
      0.0, dodDistL + 0.52 * cos45
    );
  }

  // Slots 7, 14, 15, 16: 4 BOTTOM planes (ly < 0) — Slot 16 is exact (0, -1, 0) read as `dyn` by shader.js;
  // Slots 7, 14, 15 are matching bottom planes with tiny outward azimuths so the 2-pole bottom pyramid
  // smoothly flattens and rises into the flat bottom of the cube/hopper block without ever flipping normal sign.
  const topBotHalfL = hy * fCubeVar(16, 0.06) * rOverL;
  const botOffL = (topBotHalfL - 0.52) * (1.0 - dtW) + (0.42 * dtW);
  const botSlots = [7, 14, 15];
  const botAz = [0, 0.5 * Math.PI, Math.PI];
  const sinBot = 0.02;
  const cosBot = Math.sqrt(1 - sinBot * sinBot);
  for (let b = 0; b < 3; b++) {
    const aB = botAz[b];
    const fHalfL = ((b & 1) === 0 ? hx : hz) * rOverL;
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, botSlots[b],
      sinBot * Math.cos(aB), -cosBot, sinBot * Math.sin(aB),
      0.0, botOffL * cosBot + sinBot * fHalfL * 0.5
    );
  }
  setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 16, 0, -1, 0, 0.0, botOffL);

  // Slot 17: Exact TOP (0, +1, 0) plane read as `dyp` by shader.js
  const topTableHalfL = topBotHalfL * (1.0 - 0.14 * extraChamfer);
  setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0.0, topTableHalfL + 0.52);
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
  habit, symVal, taperLy, latScale, nodeTilt, chamfer, dom, faceJitterAmt, tipJitterAmt, fs, L, r, doubleTermBlend,
  outLx, outLy, outLz, outOffR, outOffL
) {
  const sys = resolveHabitSystem(habit);
  const fVar = (k, scale = 0.35) => Math.max(0.55, Math.min(1.45, 1 + faceJitterAmt * scale * fs[k]));
  const tVar = (k, scale = 0.28) => tipJitterAmt * scale * fs[8 + k];
  const dtW = typeof doubleTermBlend === 'number'
    ? Math.max(0.0, Math.min(1.0, doubleTermBlend))
    : (doubleTermBlend ? 1.0 : 0.0);
  const botExt = 0.02 + 0.40 * dtW;
  taperLy *= (1.0 - 0.78 * dtW);
  const tableTop = Math.max(0.76, 1.14 - 0.50 * Math.max(0, chamfer));

  if (sys === 'Cubic / Isometric') {
    // Equant isometric polyhedron (a = b = c, alpha = beta = gamma = 90°) centered at y = 0.52 * L
    const isoScale = Math.max(1.18, Math.min(3.8, (0.34 * L) / Math.max(r, 1e-4)));
    const cosOct = 0.57735027; // 1 / sqrt(3)
    const sinOct = 0.81649658; // sqrt(2 / 3)
    const cos45 = 0.70710678;
    const cubeDist = isoScale * Math.max(0.96, 1.38 - 0.52 * chamfer);
    const dodDist = isoScale * Math.max(1.02, 1.24 - 0.24 * chamfer);
    // Slots 0..6: 4 vertical {100} cube faces (0, 2, 4, 6) + 3 vertical {110} bevels (1, 3, 5)
    for (let m = 0; m < 4; m++) {
      const aCube = m * 0.5 * Math.PI;
      const kCube = m * 2;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kCube,
        Math.cos(aCube), 0, Math.sin(aCube),
        cubeDist * fVar(kCube, 0.14), 0
      );
      if (m < 3) {
        const aCorner = (m + 0.5) * 0.5 * Math.PI;
        const kCorner = m * 2 + 1;
        setLocalPlane(
          outLx, outLy, outLz, outOffR, outOffL, kCorner,
          Math.cos(aCorner), 0, Math.sin(aCorner),
          dodDist * fVar(kCorner, 0.12), 0
        );
      }
    }
    // Slots 8..13: 4 top {111} octahedral faces (8..11) + 2 top {110} dodecahedral bevels (12..13)
    for (let m = 0; m < 4; m++) {
      const aOct = (m + 0.5) * 0.5 * Math.PI;
      const kTopOct = 8 + m;
      const dTop = isoScale * (1 + 0.10 * tipJitterAmt * fs[kTopOct]);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kTopOct,
        sinOct * Math.cos(aOct), cosOct, sinOct * Math.sin(aOct),
        dTop, 0.52 * cosOct
      );
    }
    for (let m = 0; m < 2; m++) {
      const aCube = m * Math.PI;
      const kDod = 12 + m;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kDod,
        cos45 * Math.cos(aCube), cos45, cos45 * Math.sin(aCube),
        dodDist * fVar(kDod, 0.12), 0.52 * cos45
      );
    }
    // Slots 7, 14, 15, 16: 4 bottom {111} octahedral faces (ly < 0)
    const botSlots = [7, 14, 15, 16];
    for (let m = 0; m < 4; m++) {
      const aOct = (m + 0.5) * 0.5 * Math.PI;
      const kBot = botSlots[m];
      const dBot = isoScale * (1 + 0.10 * tipJitterAmt * fs[m * 2 + 1]);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, kBot,
        sinOct * Math.cos(aOct), -cosOct, sinOct * Math.sin(aOct),
        dBot, -0.52 * cosOct
      );
    }
    // Slot 17: Top +ax cube face
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, cubeDist, 0.52);
    return;
  }

  if (sys === 'Hexagonal') {
    // 6-fold hexagonal prism {10-10} + {10-11} bipyramid bevels + continuous pointed bottom termination
    // Monotonically ordered slots k = 0..6 ([0°, 60°, 90°, 120°, 180°, 240°, 300°] -> [0°, 45°, 90°, 135°, 180°, 225°, 270°])
    const hexTaper = taperLy * 0.45;
    const hexLat = Math.sqrt(Math.max(0.1, 1 - hexTaper * hexTaper));
    const sqBlend = smooth01(Math.max(0, Math.min(1, (6.0 - symVal) / 2.0)));
    const hexAz = [0, 60 * DEG, 90 * DEG, 120 * DEG, 180 * DEG, 240 * DEG, 300 * DEG];
    const sqAz  = [0, 45 * DEG, 90 * DEG, 135 * DEG, 180 * DEG, 225 * DEG, 270 * DEG];
    const hexR  = [1.0, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0];
    const sqR   = [0.94, 1.14 - 0.16 * chamfer, 0.94, 1.14 - 0.16 * chamfer, 0.94, 1.14 - 0.16 * chamfer, 0.94];
    for (let k = 0; k < 7; k++) {
      const a = hexAz[k] + (sqAz[k] - hexAz[k]) * sqBlend;
      const radR = (hexR[k] + (sqR[k] - hexR[k]) * sqBlend) * fVar(k, 0.20);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        hexLat * Math.cos(a), hexTaper, hexLat * Math.sin(a),
        radR, 0.5 * hexTaper
      );
    }
    const bipTilt = Math.max(30 * DEG, Math.min(58 * DEG, nodeTilt * 0.78));
    const cosB = Math.cos(bipTilt), sinB = Math.sin(bipTilt);
    const tableH = Math.max(0.68, 0.84 - 0.22 * chamfer);
    for (let j = 0; j < 6; j++) {
      const a = (j * TAU) / 6;
      const offR = ((0.34 - 0.16 * chamfer) * (1 - dtW) + (0.14 - 0.08 * chamfer) * dtW) * sinB + tVar(j, 0.16);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + j,
        sinB * Math.cos(a), cosB, sinB * Math.sin(a),
        offR, tableH * cosB
      );
    }
    const botSlots = [7, 14, 15, 16];
    const botJ = [0, 1, 3, 5];
    for (let b = 0; b < 4; b++) {
      const j = botJ[b];
      const a = (j * TAU) / 6;
      const offR = (0.08 - 0.02 * dtW) * sinB + tVar(j, 0.14);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, botSlots[b],
        sinB * Math.cos(a), -cosB, sinB * Math.sin(a),
        offR, botExt * cosB
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableH + 0.18 * dtW);
    return;
  }

  if (sys === 'Tetragonal') {
    // 4-fold c-axis symmetry (a = b != c, alpha = beta = gamma = 90°): square {100} prism + {110} corner chamfers + top & bottom {101} bipyramids
    const cornerChamfer = 1.14 - 0.22 * chamfer;
    for (let k = 0; k < 7; k++) {
      const a = (k * TAU) / 8; // 0, 45, 90, 135, 180, 225, 270 deg (Slot 6 = 270°)
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
    const topAzIndices = [0, 1, 2, 4, 5, 6];
    for (let idx = 0; idx < 6; idx++) {
      const j = topAzIndices[idx];
      const a = (j * TAU) / 8;
      const isSecondOrder = (j & 1) === 1;
      const ct = isSecondOrder ? c211 : c101;
      const st = isSecondOrder ? s211 : s101;
      const offR = (isSecondOrder ? (0.30 - 0.16 * chamfer) : 0.06) + tVar(j, 0.18);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + idx,
        st * Math.cos(a), ct, st * Math.sin(a),
        offR, ct
      );
    }
    // 4-sided pointed bottom {10-1} bipyramid termination facets in slots [7, 14, 15, 16]
    const botSlots = [7, 14, 15, 16];
    for (let m = 0; m < 4; m++) {
      const a = (m * TAU) / 4;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, botSlots[m],
        s101 * Math.cos(a), -c101, s101 * Math.sin(a),
        0.06 + tVar(m, 0.14), botExt * c101
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableTop);
    return;
  }

  if (sys === 'Orthorhombic') {
    // 3 mutually perpendicular unequal axes (a != b != c, alpha = beta = gamma = 90°):
    // Short a-axis (0.68 r) along sx, wide b-axis (1.36 r) along sz, rhombic prism {110} + top & bottom macrodomes/brachydomes
    const orthoAz = [0, 31 * DEG, 90 * DEG, 149 * DEG, 180 * DEG, 211 * DEG, 270 * DEG, 329 * DEG];
    const orthoRadii = [0.70, 0.82, 1.34, 0.82, 0.70, 0.82, 1.34, 0.82];
    for (let k = 0; k < 7; k++) {
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
    const topIdx = [0, 1, 2, 4, 5, 6];
    for (let idx = 0; idx < 6; idx++) {
      const j = topIdx[idx];
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
        outLx, outLy, outLz, outOffR, outOffL, 8 + idx,
        st * Math.cos(a), ct, st * Math.sin(a),
        baseOffR + tVar(j, 0.18), ct
      );
    }
    // 4-sided pointed bottom rhombic dome/bipyramid termination in slots [7, 14, 15, 16]
    const botSlots = [7, 14, 15, 16];
    for (let m = 0; m < 4; m++) {
      const j = m * 2;
      const a = orthoAz[j];
      const tj = (j === 0 || j === 4) ? tMacro : tBrachy;
      const ct = Math.cos(tj), st = Math.sin(tj);
      const baseOffR = (j === 0 || j === 4) ? 0.05 : 0.14;
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, botSlots[m],
        st * Math.cos(a), -ct, st * Math.sin(a),
        baseOffR + tVar(m, 0.14), botExt * ct
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableTop);
    return;
  }

  if (sys === 'Monoclinic') {
    // 1 oblique inclined clino-axis (a != b != c, alpha = gamma = 90°, beta ≈ 116° != 90°):
    // Flattened lath prism + obliquely slanted top & bottom clinodome terminations
    const monoAz = [0, 38 * DEG, 90 * DEG, 142 * DEG, 180 * DEG, 218 * DEG, 270 * DEG, 322 * DEG];
    const monoRadii = [0.60, 0.82, 1.38, 0.82, 0.60, 0.82, 1.38, 0.82];
    for (let k = 0; k < 7; k++) {
      const a = monoAz[k];
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, k,
        latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
        monoRadii[k] * fVar(k, 0.20), 0.5 * taperLy
      );
    }
    const betaShear = 0.44; // sin(26°) monoclinic inclination along +sx (clino-axis a)
    const topIdx = [0, 1, 2, 4, 5, 6];
    for (let idx = 0; idx < 6; idx++) {
      const j = topIdx[idx];
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
        outLx, outLy, outLz, outOffR, outOffL, 8 + idx,
        lx * inv, normLy, lz * inv,
        offR, 0.90 * normLy
      );
    }
    const botSlots = [7, 14, 15, 16];
    for (let m = 0; m < 4; m++) {
      const j = m * 2;
      const a = monoAz[j];
      const isFront = Math.cos(a) > 0.15;
      const tj = isFront ? Math.max(26 * DEG, nodeTilt - 16 * DEG) : Math.min(70 * DEG, nodeTilt + 4 * DEG);
      const ct = Math.cos(tj), st = Math.sin(tj);
      const lx = st * Math.cos(a) - betaShear * ct;
      const ly = -ct;
      const lz = st * Math.sin(a);
      const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
      const normAbsLy = Math.abs(ly * inv);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, botSlots[m],
        lx * inv, ly * inv, lz * inv,
        0.12 + tVar(m, 0.16), botExt * normAbsLy
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableTop);
    return;
  }

  if (sys === 'Triclinic') {
    // All 3 axes and all 3 interaxial angles unequal & non-90° (a != b != c, alpha ≈ 72°, beta ≈ 108°, gamma ≈ 70°):
    // Skewed non-orthogonal parallelogram bladed prism + doubly-inclined asymmetric top & bottom wedge terminations
    const tricAz = [0, 34 * DEG, 70 * DEG, 128 * DEG, 180 * DEG, 214 * DEG, 250 * DEG, 308 * DEG];
    const tricRadii = [0.52, 0.78, 1.42, 1.06, 0.52, 0.78, 1.42, 1.06];
    const tricLySkew = [0.09, 0.08, 0.12, -0.07, -0.09, -0.08, -0.12, 0.07];
    for (let k = 0; k < 7; k++) {
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
    const topIdx = [0, 1, 2, 4, 5, 6];
    for (let idx = 0; idx < 6; idx++) {
      const j = topIdx[idx];
      const a = tricAz[j];
      const tj = Math.max(26 * DEG, Math.min(74 * DEG, nodeTilt + tricTiltOffsets[j] * DEG));
      const ct = Math.cos(tj), st = Math.sin(tj);
      const lx = st * Math.cos(a) + 0.30 * ct;
      const ly = ct;
      const lz = st * Math.sin(a) + 0.26 * ct;
      const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
      const normLy = ly * inv;
      const offR = 0.14 + 0.06 * (j % 3) + tVar(j, 0.22);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, 8 + idx,
        lx * inv, normLy, lz * inv,
        offR, 0.88 * normLy
      );
    }
    const botSlots = [7, 14, 15, 16];
    for (let m = 0; m < 4; m++) {
      const j = m * 2;
      const a = tricAz[j];
      const tj = Math.max(28 * DEG, Math.min(72 * DEG, nodeTilt + tricTiltOffsets[j] * DEG));
      const ct = Math.cos(tj), st = Math.sin(tj);
      const lx = st * Math.cos(a) - 0.30 * ct;
      const ly = -ct;
      const lz = st * Math.sin(a) - 0.26 * ct;
      const inv = 1 / (Math.hypot(lx, ly, lz) || 1);
      const normAbsLy = Math.abs(ly * inv);
      setLocalPlane(
        outLx, outLy, outLz, outOffR, outOffL, botSlots[m],
        lx * inv, ly * inv, lz * inv,
        0.14 + tVar(m, 0.16), botExt * normAbsLy
      );
    }
    setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableTop);
    return;
  }

  // Default: Trigonal (Rhombohedral — 3-fold c-axis symmetry, e.g. Quartz, Corundum, Tourmaline)
  // Smoothly blends across 3-sided Tourmaline (symVal ≈ 3), 4-sided square prism (symVal ≈ 4), and 6-sided Quartz prism (symVal ≈ 6)
  // using monotonically ordered azimuth slots k = 0..6 ([0°, 60°, 90°, 120°, 180°, 240°, 300°] -> [0°, 45°, 90°, 135°, 180°, 225°, 270°]).
  const triRound = smooth01(Math.max(0, Math.min(1, (3.8 - symVal) / 1.0)));
  const sqBlend = smooth01(Math.max(0, Math.min(1, (6.0 - symVal) / 2.0))) * (1.0 - triRound);
  const hexAz = [0, 60 * DEG, 90 * DEG, 120 * DEG, 180 * DEG, 240 * DEG, 300 * DEG];
  const sqAz  = [0, 45 * DEG, 90 * DEG, 135 * DEG, 180 * DEG, 225 * DEG, 270 * DEG];
  const minorMask = [0, 1, 0, 0, 1, 0, 1];
  for (let k = 0; k < 7; k++) {
    const a = hexAz[k] + (sqAz[k] - hexAz[k]) * sqBlend;
    const isMinorPrism = minorMask[k] === 1;
    const hexBaseR = k === 2
      ? (1.10 * triRound + 1.18 * (1 - triRound))
      : (isMinorPrism ? (1.0 + 0.16 * triRound + 0.04 * dom) : (1.0 - 0.16 * triRound - 0.04 * dom));
    const sqBaseR = (k & 1) === 1 ? (1.14 - 0.16 * chamfer) : 0.94;
    const baseR = hexBaseR + (sqBaseR - hexBaseR) * sqBlend;
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, k,
      latScale * Math.cos(a), taperLy, latScale * Math.sin(a),
      baseR * fVar(k, k === 2 ? 0.20 : 0.35), 0.5 * taperLy
    );
  }
  // 3-fold Rhombohedral top termination: major r{10-11} (even j = 0,2,4) alternating with minor z{01-11} (odd j = 1,3,5)
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
  // Pointed rhombohedral bottom termination in slots [7, 14, 15, 16] (all 3 major rhombohedra j = 1, 3, 5 at 60°, 180°, 300° + minor j = 0 at 0°),
  // scaled continuously by `botExt = 0.02 + 0.40 * dtW` so the lower half smoothly retracts/grows across doubleTermBlend in [0, 1].
  const botSlots = [7, 14, 15, 16];
  const botJ = [0, 1, 3, 5];
  for (let b = 0; b < 4; b++) {
    const j = botJ[b];
    const a = (j * TAU) / 6;
    const isMinorRhombo = (j & 1) === 0;
    const ct = isMinorRhombo ? cosZ : cosR;
    const st = isMinorRhombo ? sinZ : sinR;
    const offR = (isMinorRhombo ? (0.24 + 0.38 * effDom + 0.14 * triRound) : Math.max(0.06, 0.12 - 0.08 * effDom)) + Math.max(0, tVar(j, 0.24));
    setLocalPlane(
      outLx, outLy, outLz, outOffR, outOffL, botSlots[b],
      st * Math.cos(a), -ct, st * Math.sin(a),
      offR, botExt * ct
    );
  }
  setLocalPlane(outLx, outLy, outLz, outOffR, outOffL, 17, 0, 1, 0, 0, tableTop);
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
  const hopperVal = P?.hopper ? Math.max(0, Math.min(1, Number(P.hopper.value))) : 0;
  const meltVal = P?.melt ? Math.max(0, Math.min(1, Number(P.melt.value))) : 0;
  const metallicVal = P?.metallic ? Math.max(0, Math.min(1, Number(P.metallic.value))) : 0;
  const hopperBlend = Math.max(0, Math.min(1, hopperVal / 0.92));
  const meltBlend = smooth01(Math.max(0, Math.min(1, meltVal / 0.52)));
  const rawCount = P?.crystals ? Number(P.crystals.value) : (P?.count ? Number(P.count.value) : 4.0);
  const count = rawCount;
  const floatC = Math.max(3.0, Math.min(MAX_CRYSTALS, count));
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
  const portraitBlend = Math.max(0, Math.min(1, (1.0 - aspect) / 0.55));
  cluster.aspect = aspect;
  cluster.growDirection = growSideways ? 'sideways' : 'upwards';

  const fovDeg = P.fov ? Number(P.fov.value) : 21;
  const D = P.distance ? Number(P.distance.value) : 2.15;
  const portraitZoom = vw < vh ? Math.min(1.36, Math.pow(vh / Math.max(1, vw), 0.38)) : 1.0;
  const focal = (0.5 / Math.tan((fovDeg * DEG) / 2)) * portraitZoom;
  const minDim = Math.max(1, Math.min(vw, vh));
  const halfFrustumMin = D / (2 * focal);
  const halfFrustumX = halfFrustumMin * (vw / minDim);
  const halfFrustumY = halfFrustumMin * (vh / minDim);

  const growth = P.growth ? Math.max(0.03, Math.min(4.5, Number(P.growth.value))) : 1;
  const dirProgress = 1.22 * Math.tanh(Math.max(0, growth - 1.0) * 0.72);
  const overScale = 1 + 0.48 * Math.tanh(Math.max(0, growth - 1.0) * 0.72);

  const curHabit = P.habit ? String(P.habit.value) : 'Trigonal';
  const morphBlend = Number(cluster.morphHabitBlend) || 0;
  const hFrom = cluster.morphHabitFrom ? resolveHabitSystem(cluster.morphHabitFrom) : '';
  const hTo = cluster.morphHabitTo ? resolveHabitSystem(cluster.morphHabitTo) : '';
  const curSys = resolveHabitSystem(curHabit);
  const pyriteCubeBlend = Math.max(0, Math.min(1, (metallicVal - 0.08) / 0.88)) * (1 - hopperBlend);
  const cubeBlockBlend = Math.max(hopperBlend, pyriteCubeBlend);

  const baseRepel = (hFrom && hTo && hFrom !== hTo)
    ? habitRepelWeight(hFrom) * (1 - morphBlend) + habitRepelWeight(hTo) * morphBlend
    : habitRepelWeight(curSys);
  const repelStrength = hopperBlend > 1e-4
    ? baseRepel * (1 - hopperBlend) + 0.86 * hopperBlend
    : baseRepel * (1 - 0.66 * pyriteCubeBlend);
  const baseCubicBlend = (hFrom && hTo && hFrom !== hTo)
    ? (hFrom === 'Cubic / Isometric' ? 1 - morphBlend : 0) + (hTo === 'Cubic / Isometric' ? morphBlend : 0)
    : (curSys === 'Cubic / Isometric' ? 1 : 0);
  const cubicBlend = Math.max(baseCubicBlend, cubeBlockBlend);

  const paramDoubleTerm = P?.doubleTerminated !== undefined
    ? (typeof P.doubleTerminated.value === 'boolean'
      ? (P.doubleTerminated.value ? 1.0 : 0.0)
      : Math.max(0.0, Math.min(1.0, Number(P.doubleTerminated.value))))
    : 1.0;
  const rawDoubleTermBlend = paramDoubleTerm * (1.0 - cubicBlend);

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
  if (!Number.isFinite(cluster.doubleTermBlend) || !cluster.centerInit) {
    cluster.doubleTermBlend = rawDoubleTermBlend;
  } else if (nodeDt > 0) {
    cluster.doubleTermBlend += (rawDoubleTermBlend - cluster.doubleTermBlend) * (1 - Math.exp(-3.6 * nodeDt));
  } else {
    cluster.doubleTermBlend = rawDoubleTermBlend;
  }
  const doubleTermBlend = Math.max(0.0, Math.min(1.0, cluster.doubleTermBlend));

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

    // Continuous presence & twinW with critically-damped spring — wide overlapping fade windows for i >= 3
    // so outer strands i = 3, 4, 5 shrink and grow smoothly across the full morph rather than dying in the first 25%.
    const effTwinShare = twinShare * (1.0 - hopperBlend);
    const thrLo = i < 3 ? i : (i <= 5 ? (3.0 + (i - 3) * 0.22) : (i - 0.8));
    const thrHi = i + 1.0;
    const rawPresence = smooth01(Math.max(0.0, Math.min(1.0, (floatC - thrLo) / Math.max(1.0, thrHi - thrLo)))) * (i >= 6 ? Math.max(0.0, 1.0 - hopperBlend) : 1.0);
    const rawTwinW = i < 2 ? 0.0 : smooth01(Math.max(0.0, Math.min(1.0, (effTwinShare - node.twinRoll + 0.44) / 0.88))) * (1.0 - hopperBlend);
    if (!node.smoothInit) {
      node.smoothPresence = rawPresence;
      node.velPresence = 0;
      node.smoothTwinW = rawTwinW;
      node.velTwinW = 0;
      node.smoothInit = true;
    } else if (nodeDt > 0) {
      let remN = nodeDt;
      const omegaN = 1.15 + 0.15 * (0.5 + 0.5 * (node.uLen || 0));
      const omegaN2 = omegaN * omegaN;
      const dampN = 2.0 * omegaN;
      const maxVelN = 0.45;
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
    const rho0 = i === 0
      ? 0.26 + 0.10 * node.uRho
      : Math.min(1.0, Math.pow((i + 0.35 * node.uRho) / Math.max(1.4, initPrimaryCount - 0.4), 0.52));
    node.rho = rho0;
    const az0 = i * GOLDEN + 0.38 * node.uAz;
    let lean0 = fan * Math.min(1.05, (i === 0 ? 0.30 : 0.72 + 0.32 * rho0) + 0.16 * node.uLean);

    if (growSideways) {
      if (i > 0) {
        const baseSideLean = Math.min(78 * DEG, lean0 * 1.24 + 6 * DEG * rho0);
        const targetSideLean = Math.min(84 * DEG, Math.max(baseSideLean, (56 + 26 * rho0) * DEG));
        lean0 = baseSideLean + (targetSideLean - baseSideLean) * dirProgress * Math.min(1, 0.45 + 0.55 * rho0);
      }
    } else {
      const portraitFanScale = 1 - 0.38 * portraitBlend;
      lean0 = lean0 * portraitFanScale * (1 - 0.58 * dirProgress);
    }

    const leanAz0 = az0 + 0.35 * node.uLeanAz;
    const effLean = cavityCurve > 0.001 && i > 0
      ? Math.min(82 * DEG, lean0 + cavityCurve * 14 * DEG * rho0)
      : lean0;
    primAx[0] = Math.sin(effLean) * Math.cos(leanAz0);
    primAx[1] = Math.cos(effLean);
    primAx[2] = Math.sin(effLean) * Math.sin(leanAz0);

    if (!node.frameInit) {
      const isTwinNode = i >= 2 && rawTwinW >= 0.5;
      node.lockedTwinRole = isTwinNode ? 1 : 0;
      node.rot0 = node.uTwist;
      node.az0 = az0;
    }

    const effTwinW = i < 2 ? 0.0 : twinW * smooth01(Math.max(0.0, Math.min(1.0, effTwinShare / 0.22))) * (1.0 - hopperBlend);
    if (effTwinW <= 1e-4) {
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
      const tax0 = twinAx[0] / tal, tax1 = twinAx[1] / tal, tax2 = twinAx[2] / tal;
      const mixAx0 = primAx[0] * (1 - effTwinW) + tax0 * effTwinW;
      const mixAx1 = primAx[1] * (1 - effTwinW) + tax1 * effTwinW;
      const mixAx2 = primAx[2] * (1 - effTwinW) + tax2 * effTwinW;
      const mal = Math.hypot(mixAx0, mixAx1, mixAx2) || 1;
      ax[0] = mixAx0 / mal; ax[1] = mixAx1 / mal; ax[2] = mixAx2 / mal;
      if (!node.branchRadial0) node.branchRadial0 = new Float32Array(3);
      node.branchRadial0[0] = rx; node.branchRadial0[1] = ry; node.branchRadial0[2] = rz;
    }

    const seedInt = Math.floor(Math.abs(P.seed ? Number(P.seed.value) : 64));
    const nodeUidInt = Math.max(1, Math.round(node.uid || (i + 1)));
    // For the 3-block Bismuth Hopper cluster (i = 0, 1, 2):
    //   Block 0: Hero broad cantilevered terraced hopper wing (morph 0)
    //   Block 1: Tall stepped pagoda ziggurat tower interlocking at an angle (morph 1)
    //   Block 2: Open-sided L-bracket / Greek-key sub-block on the opposite flank (morph 2)
    const hopperMorph = i < 5 ? i : ((nodeUidInt - 1 + (seedInt % 5)) % 5);
    node.hopperMorph = hopperMorph;

    if (hopperBlend > 1e-4 && i < 6) {
      // Broken-out 6-block cascading Bismuth Hopper cluster (matching reference photo 1dbbb5bc):
      // Asymmetric stepped orientations forming a jagged diagonal cascade from upper-right crown to lower-left fingers
      const archMode = seedInt % 3;
      const azBase = (0.16 + 0.07 * archMode) * Math.PI;
      const azSpread = (i === 0) ? azBase
                     : (i === 1) ? (azBase + 0.38 * Math.PI)
                     : (i === 2) ? (azBase - 0.56 * Math.PI)
                     : (i === 3) ? (azBase + 0.82 * Math.PI)
                     : (i === 4) ? (azBase - 0.92 * Math.PI)
                                 : (azBase - 0.24 * Math.PI);
      const leanDeg = (i === 0) ? 18
                    : (i === 1) ? 15
                    : (i === 2) ? 26
                    : (i === 3) ? 22
                    : (i === 4) ? 38
                                : 44;
      const zigLean = leanDeg * DEG;
      const zAx0 = Math.sin(zigLean) * Math.cos(azSpread);
      const zAx1 = Math.cos(zigLean);
      const zAx2 = Math.sin(zigLean) * Math.sin(azSpread);
      const hopperFrameW = 0.88 * hopperBlend * smooth01(hopperBlend);
      ax[0] = ax[0] * (1 - hopperFrameW) + zAx0 * hopperFrameW;
      ax[1] = ax[1] * (1 - hopperFrameW) + zAx1 * hopperFrameW;
      ax[2] = ax[2] * (1 - hopperFrameW) + zAx2 * hopperFrameW;
      const zal = Math.hypot(ax[0], ax[1], ax[2]) || 1;
      ax[0] /= zal; ax[1] /= zal; ax[2] /= zal;
    }

    if (!node.frameInit || !node.axis0) {
      node.axis0 = new Float32Array(ax);
      buildSideBasis(node.rot0);
      node.u0 = new Float32Array(sx);
      node.v0 = new Float32Array(sz);
      node.frameInit = true;
    } else if (nodeDt > 0 || !evolveOn) {
      const aAxis = nodeDt > 0 ? (1 - Math.exp(-2.8 * nodeDt)) : 1.0;
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
    const twistStep = (hopperMorph === 3 || hopperMorph === 4) ? (0.25 * Math.PI) : (0.5 * Math.PI);
    const quantTwist = Math.round(node.rot0 / twistStep) * twistStep + 0.16 * ((nodeUidInt % 3) - 1);
    const tw = node.rot0 * (1 - 0.86 * hopperBlend) + quantTwist * (0.86 * hopperBlend);
    const rho = node.rho;
    const az = node.az0 || (i * GOLDEN + 0.38 * node.uAz);
    node.doubleTermBlend = doubleTermBlend;
    node.isDoubleTerm = doubleTermBlend >= 0.5;

    let primBaseL = (i === 0 ? 0.76 : Math.max(0.28, 0.80 - P.lengthJitter.value * (0.28 * node.uLen + 0.34 * rho))) * lenMod;
    const horizComp = Math.hypot(ax[0], ax[2]);
    const vertComp = Math.abs(ax[1]);
    if (growSideways) {
      primBaseL *= 1 + dirProgress * (0.78 * horizComp * Math.min(1.9, aspect) - 0.16 * vertComp);
    } else {
      primBaseL *= 1 + (dirProgress + 0.18 * portraitBlend) * (0.85 * vertComp * Math.min(1.9, 1 / aspect) - 0.16 * horizComp);
    }

    let primBaseR = P.thickness.value * Math.pow(primBaseL, 0.65) * Math.max(0.25, 1 + P.thicknessJitter.value * 0.55 * node.uRadSym) * radMod;
    if (hopperBlend > 1e-4) {
      // Varied sub-block scales across the 6 cascading Bismuth Hopper blocks (matching reference photo 1dbbb5bc)
      const biLenMult = i === 0 ? 1.18 : i === 1 ? 1.12 : i === 2 ? 1.06 : i === 3 ? 0.94 : i === 4 ? 1.02 : 0.86;
      const biRadMult = i === 0 ? 1.16 : i === 1 ? 0.92 : i === 2 ? 1.04 : i === 3 ? 0.88 : i === 4 ? 0.94 : 0.82;
      primBaseL *= 1.0 + (biLenMult - 1.0) * hopperBlend;
      primBaseR *= 1.0 + (biRadMult - 1.0) * hopperBlend;
    }

    // Softly bound extreme vertical projection of single nearly-upright outlier spikes so the camera
    // can sit ~25-30% closer without a lone vertical tip clipping the top/bottom viewport edge,
    // while keeping full prism thickness `primBaseR` and full horizontal reach on angled crystals.
    const vertSpanFactor = vertComp * (1.04 + 0.38 * doubleTermBlend) * (1 - 0.45 * cubicBlend);
    const spikeCap = growSideways ? 0.64 : 0.88;
    if (primBaseL * vertSpanFactor > spikeCap) {
      const rawVS = primBaseL * vertSpanFactor;
      const cappedVS = spikeCap + 0.12 * Math.tanh((rawVS - spikeCap) / 0.12);
      primBaseL *= cappedVS / rawVS;
    }

    let baseL = primBaseL;
    let baseR = primBaseR;

    const curPrimL = primBaseL * gScale * presence;
    const curPrimR = primBaseR * Math.pow(gScale, 0.78) * presence;
    // Non-diamond crystals cross through a shared root nexus (0, 0, 0) so their longer top points
    // (+0.86 * L) and shorter bottom points (-0.56 * L) extend out past the root in asymmetric 'X' / '*' shapes.
    const dtCrossFrac = Math.max(0.10, Math.min(0.22, P.crossAt.value * 0.45 + 0.07));
    const singleCrossFrac = Math.max(0.16, Math.min(0.32, P.crossAt.value));
    const baseCrossFrac = singleCrossFrac + (dtCrossFrac - singleCrossFrac) * doubleTermBlend;
    const back = (cubicBlend * P.crossAt.value + (1 - cubicBlend) * baseCrossFrac) * curPrimL;
    const uvSpread = 0.16 + 0.89 * cubicBlend;
    const u = node.uRootU * curPrimR * uvSpread, v = node.uRootV * curPrimR * uvSpread;
    for (let k = 0; k < 3; k++) {
      primRoot[k] = -ax[k] * back + sx[k] * u + sz[k] * v + node.driftOffset[k];
    }
    if (dirProgress > 0.01 && i > 0) {
      const spreadScale = 0.12 + 0.88 * cubicBlend;
      if (growSideways) {
        const latDist = Math.min(curPrimR * 0.45 + cubicBlend * halfFrustumX, dirProgress * halfFrustumX * 0.96 * rho * localGrowth * presence * spreadScale);
        primRoot[0] += Math.cos(az) * latDist;
        primRoot[2] += Math.sin(az) * latDist;
        primRoot[1] *= (1 - 0.42 * dirProgress * spreadScale);
      } else {
        const tier = ((i % 2 === 1) ? 1 : -0.88) * Math.pow(rho, 0.78);
        const vertDist = Math.min(curPrimR * 0.45 + cubicBlend * halfFrustumY, Math.abs(dirProgress * halfFrustumY * 0.98 * tier * localGrowth * presence * spreadScale)) * Math.sign(tier);
        primRoot[1] += vertDist;
        primRoot[0] *= (1 - 0.42 * dirProgress * spreadScale);
        primRoot[2] *= (1 - 0.42 * dirProgress * spreadScale);
      }
    }
    if (cavityCurve > 0.001) {
      // Keep base offset tightly bounded by curPrimR * 0.42 on non-diamond clusters so crystals
      // always grow out of a shared clumped base nexus rather than floating in separate mid-air orbits.
      const maxClumpBowlR = curPrimR * 0.42 + cubicBlend * 0.52 * cavityCurve * (0.35 + 0.65 * gScale) * presence;
      const bowlR = Math.min(maxClumpBowlR, 0.12 * cavityCurve * (0.35 + 0.65 * gScale) * presence);
      const ringR = bowlR * rho;
      const bowlY = (-bowlR * Math.cos(rho * 1.15) + bowlR * 0.45) * 0.35;
      primRoot[0] += Math.cos(az) * ringR;
      primRoot[1] += bowlY;
      primRoot[2] += Math.sin(az) * ringR;
    }

    if (effTwinW <= 1e-4) {
      root[0] = primRoot[0]; root[1] = primRoot[1]; root[2] = primRoot[2];
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
      const twinBaseL = pFullL * (0.48 + 0.34 * node.uBranchLen) * (1 + 0.32 * dirProgress) * lenMod;
      const twinBaseR = Math.max(0.035, P.thickness.value * 0.85) * (0.68 + 0.28 * node.uBranchRad) * radMod;
      baseL = primBaseL * (1 - effTwinW) + twinBaseL * effTwinW;
      baseR = primBaseR * (1 - effTwinW) + twinBaseR * effTwinW;
      const curTwinL = twinBaseL * gScale * presence;
      const attachFrac = Math.max(0.06, Math.min(0.22, 0.08 + 0.14 * node.uAttach));
      const attachDist = pL * attachFrac;
      const twinBack = (0.16 - 0.02 * doubleTermBlend) * curTwinL * (1 - cubicBlend);
      for (let k = 0; k < 3; k++) {
        twinRoot[k] = pRoot[k] + pAxis[k] * attachDist + br[k] * (pR * 0.10) - ax[k] * twinBack + node.driftOffset[k];
        root[k] = primRoot[k] * (1 - effTwinW) + twinRoot[k] * effTwinW;
      }
    }

    if (hopperBlend > 1e-4 && i < 6) {
      // Broken-out 6-block cascading Bismuth Hopper offsets (matching reference photo 1dbbb5bc):
      //   i = 0: Broad lower-right multi-chamber G-spiral foundation block
      //   i = 1: Upper-right G-spiral crown tower rising above Block 0
      //   i = 2: Mid-center / upper-left protruding G-spiral block
      //   i = 3: Mid-right cantilevered L-hook step block
      //   i = 4: Left-flank cascading crenellated stair-stepped block
      //   i = 5: Lower-left broken-out L-pillar finger outcrop
      const offX = (i === 0 ?  0.14 : i === 1 ?  0.11 : i === 2 ? -0.10 : i === 3 ?  0.27 : i === 4 ? -0.26 : -0.34) * baseL * gScale * presence;
      const offY = (i === 0 ? -0.16 : i === 1 ?  0.26 : i === 2 ?  0.08 : i === 3 ?  0.05 : i === 4 ? -0.02 : -0.18) * baseL * gScale * presence;
      const offZ = (i === 0 ?  0.12 : i === 1 ?  0.04 : i === 2 ?  0.14 : i === 3 ? -0.08 : i === 4 ? -0.04 :  0.08) * baseL * gScale * presence;
      const hopperRootW = hopperBlend * smooth01(hopperBlend);
      root[0] = root[0] * (1 - 0.82 * hopperRootW) + offX * hopperRootW;
      root[1] = root[1] * (1 - 0.82 * hopperRootW) + offY * hopperRootW;
      root[2] = root[2] * (1 - 0.82 * hopperRootW) + offZ * hopperRootW;
    }

    if (meltBlend > 1e-4 && hopperBlend < 0.5) {
      // Sculptural Gallium crystal cluster staging (matching reference photo 372c3f66):
      // Upper intergrown orthorhombic crystal blocks perched above the molten liquid pool
      const gaBlend = meltBlend * (1 - hopperBlend);
      const gaX = (i === 0 ? 0.01 : i === 1 ? 0.07 : i === 2 ? -0.10 : (i === 3 ? 0.24 : 0.0)) * baseL * gScale * presence;
      const gaY = (i === 0 ? 0.14 : i === 1 ? 0.06 : i === 2 ? 0.07 : (i === 3 ? 0.09 : 0.06)) * baseL * gScale * presence;
      const gaZ = (i === 0 ? -0.03 : i === 1 ? 0.09 : i === 2 ? 0.02 : (i === 3 ? -0.05 : 0.0)) * baseL * gScale * presence;
      root[0] = root[0] * (1 - 0.68 * gaBlend) + gaX * gaBlend;
      root[1] = root[1] * (1 - 0.58 * gaBlend) + (gaY + 0.095 * meltVal) * gaBlend;
      root[2] = root[2] * (1 - 0.68 * gaBlend) + gaZ * gaBlend;
    }

    // Pull emerging/shrinking crystals smoothly into the heart of the cluster core (0, 0, 0)
    // so disappearing crystals retract inside the permanent core (crystals 0..2) before vanishing.
    const corePull = i < 3 ? 1.0 : smooth01(Math.min(1.0, presence / 0.42));
    const latPull = corePull * (1 - 0.28 * portraitBlend);
    root[0] *= latPull;
    root[1] *= corePull;
    root[2] *= latPull;

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

    initRootX[activeCount] = root[0];
    initRootY[activeCount] = root[1];
    initRootZ[activeCount] = root[2];
    activeNodes[activeCount++] = node;
  }

  // Simple pairwise magnetic repulsion strictly for high-overlap shapes like Diamond (post-015442, post-015454)
  if (repelStrength > 1e-4 && activeCount > 1) {
    const centerFrac = 0.46 + 0.06 * cubicBlend;
    for (let iter = 0; iter < 4; iter++) {
      for (let a = 0; a < activeCount - 1; a++) {
        const nodeA = activeNodes[a];
        const rootA = nodeA.worldRoot, axA = nodeA.worldAxis;
        const La = nodeA.worldL, ra = nodeA.worldR;
        const radA = cubicBlend * Math.max(1.22 * ra, 0.34 * La) + (1 - cubicBlend) * Math.max(1.28 * ra, 0.22 * La);
        const cax = rootA[0] + axA[0] * centerFrac * La;
        const cay = rootA[1] + axA[1] * centerFrac * La;
        const caz = rootA[2] + axA[2] * centerFrac * La;
        const azA = nodeA.az0 || (nodeA.cohortIdx * GOLDEN);

        for (let b = a + 1; b < activeCount; b++) {
          const nodeB = activeNodes[b];
          const rootB = nodeB.worldRoot, axB = nodeB.worldAxis;
          const Lb = nodeB.worldL, rb = nodeB.worldR;
          const radB = cubicBlend * Math.max(1.22 * rb, 0.34 * Lb) + (1 - cubicBlend) * Math.max(1.28 * rb, 0.22 * Lb);
          const cbx = rootB[0] + axB[0] * centerFrac * Lb;
          const cby = rootB[1] + axB[1] * centerFrac * Lb;
          const cbz = rootB[2] + axB[2] * centerFrac * Lb;

          const dcx = cax - cbx, dcy = cay - cby, dcz = caz - cbz;
          const dist = Math.hypot(dcx, dcy, dcz);
          const pairPres = Math.min(nodeA.presence, nodeB.presence);
          const minSep = (radA + radB) * (0.72 + 0.32 * cubicBlend) * repelStrength * pairPres;

          if (dist < minSep) {
            const rawGap = minSep - dist;
            const softKnee = 0.08 * minSep + 1e-4;
            const overlap = rawGap < softKnee ? (rawGap * rawGap) / (2 * softKnee) : rawGap - 0.5 * softKnee;
            const azB = nodeB.az0 || (nodeB.cohortIdx * GOLDEN);

            let vx = dcx + 0.32 * (La * axA[0] - Lb * axB[0]) + 0.08 * (Math.cos(azA) - Math.cos(azB));
            let vy = (dcy + 0.16 * (La * axA[1] - Lb * axB[1])) * 0.42;
            let vz = dcz + 0.32 * (La * axA[2] - Lb * axB[2]) + 0.08 * (Math.sin(azA) - Math.sin(azB));
            const vl = Math.hypot(vx, vy, vz) || 1;
            vx /= vl; vy /= vl; vz /= vl;

            const push = overlap * 0.50;
            const wA = a === 0 ? 0.42 : 0.5;
            const wB = 1.0 - wA;
            rootA[0] += vx * push * wA;
            rootA[1] += vy * push * wA;
            rootA[2] += vz * push * wA;
            rootB[0] -= vx * push * wB;
            rootB[1] -= vy * push * wB;
            rootB[2] -= vz * push * wB;

            const tiltPush = (overlap / Math.max(0.35, La + Lb)) * 0.14 * repelStrength;
            axA[0] += vx * tiltPush * wA;
            axA[1] = Math.max(0.15, axA[1] + vy * tiltPush * wA);
            axA[2] += vz * tiltPush * wA;
            const alA = Math.hypot(axA[0], axA[1], axA[2]) || 1;
            axA[0] /= alA; axA[1] /= alA; axA[2] /= alA;

            axB[0] -= vx * tiltPush * wB;
            axB[1] = Math.max(0.15, axB[1] - vy * tiltPush * wB);
            axB[2] -= vz * tiltPush * wB;
            const alB = Math.hypot(axB[0], axB[1], axB[2]) || 1;
            axB[0] /= alB; axB[1] /= alB; axB[2] /= alB;
          }
        }
      }
    }

    // Re-center net magnetic translation so the cluster stays centered at the origin
    let shiftX = 0, shiftY = 0, shiftZ = 0, shiftW = 0;
    for (let a = 0; a < activeCount; a++) {
      const nA = activeNodes[a];
      const w = nA.presence * nA.worldL;
      shiftX += (nA.worldRoot[0] - initRootX[a]) * w;
      shiftY += (nA.worldRoot[1] - initRootY[a]) * w;
      shiftZ += (nA.worldRoot[2] - initRootZ[a]) * w;
      shiftW += w;
    }
    if (shiftW > 1e-6) {
      shiftX /= shiftW; shiftY /= shiftW; shiftZ /= shiftW;
      for (let a = 0; a < activeCount; a++) {
        activeNodes[a].worldRoot[0] -= shiftX;
        activeNodes[a].worldRoot[1] -= shiftY;
        activeNodes[a].worldRoot[2] -= shiftZ;
      }
    }
  }

  for (let rowIdx = 0; rowIdx < activeCount; rowIdx++) {
    const node = activeNodes[rowIdx];
    const presence = node.presence;
    const L = node.worldL;
    const r = node.worldR;
    const nodeSymmetry = symmetry;
    for (let k = 0; k < 3; k++) {
      ax[k] = node.worldAxis[k];
      root[k] = node.worldRoot[k];
    }
    buildSideBasis(node.worldTwist);
    for (let k = 0; k < 3; k++) {
      node.worldSide[k] = sx[k];
    }

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

    const nodeDoubleTerm = node.doubleTermBlend !== undefined ? node.doubleTermBlend : doubleTermBlend;

    if (hFrom && hTo && hFrom !== hTo) {
      if (morphBlend <= 1e-4) {
        computeSystemPlanes(
          hFrom, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r, nodeDoubleTerm,
          tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
        );
      } else if (morphBlend >= 1 - 1e-4) {
        computeSystemPlanes(
          hTo, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r, nodeDoubleTerm,
          tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
        );
      } else {
        computeSystemPlanes(
          hFrom, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r, nodeDoubleTerm,
          tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
        );
        computeSystemPlanes(
          hTo, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r, nodeDoubleTerm,
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
      }
    } else {
      computeSystemPlanes(
        curHabit, nodeSymmetry, taperLy, latScale, nodeTilt, dynChamfer, dynDom, faceJitterAmt, dynTipJitter, fs, L, r, nodeDoubleTerm,
        tgtLx, tgtLy, tgtLz, tgtOffR, tgtOffL
      );
    }

    if (cubeBlockBlend > 1e-4) {
      computeCubeBlockPlanes(
        hopperBlend, meltBlend, node.hopperMorph || 0, dynChamfer, faceJitterAmt, fs, L, r, nodeDoubleTerm,
        cubeLx, cubeLy, cubeLz, cubeOffR, cubeOffL
      );
      for (let k = 0; k < 18; k++) {
        const bx = tgtLx[k] + (cubeLx[k] - tgtLx[k]) * cubeBlockBlend;
        const by = tgtLy[k] + (cubeLy[k] - tgtLy[k]) * cubeBlockBlend;
        const bz = tgtLz[k] + (cubeLz[k] - tgtLz[k]) * cubeBlockBlend;
        const inv = 1 / (Math.hypot(bx, by, bz) || 1);
        tgtLx[k] = bx * inv;
        tgtLy[k] = by * inv;
        tgtLz[k] = bz * inv;
        tgtOffR[k] = tgtOffR[k] + (cubeOffR[k] - tgtOffR[k]) * cubeBlockBlend;
        tgtOffL[k] = tgtOffL[k] + (cubeOffL[k] - tgtOffL[k]) * cubeBlockBlend;
      }
    }

    if (!node.planesInit || !node.smoothLx) {
      node.smoothLx = new Float32Array(tgtLx);
      node.smoothLy = new Float32Array(tgtLy);
      node.smoothLz = new Float32Array(tgtLz);
      node.smoothOffR = new Float32Array(tgtOffR);
      node.smoothOffL = new Float32Array(tgtOffL);
      node.planesInit = true;
    } else if (nodeDt > 0 || !evolveOn) {
      const aPlane = nodeDt > 0 ? (1 - Math.exp(-3.0 * nodeDt)) : 1.0;
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
    const nipBlockDamp = 1 - 0.88 * cubeBlockBlend;

    const nipDepth1 = r * (0.0045 + 0.0105 * edgeChipsVal) * chipSizeVal * nipSizeMult1 * (whaleMult1 > 1.0 ? Math.max(0.65, node.smoothChip1) : node.smoothChip1) * nipBlockDamp;
    writeCornerNipPlane(rowIdx, 18, k0, k1, shHoriz1, shVert1, nipDepth1, 0.85, 1.15, vertCount, L, r);

    const nipDepth2 = r * (0.0040 + 0.0095 * edgeChipsVal) * chipSizeVal * nipSizeMult2 * (whaleMult2 > 1.0 ? Math.max(0.65, node.smoothChip2) : node.smoothChip2) * nipBlockDamp;
    writeCornerNipPlane(rowIdx, 19, k2, k3, shHoriz2, shVert2, nipDepth2, 1.12, 0.88, vertCount, L, r);

    const nipDepth3 = r * (0.0036 + 0.0088 * edgeChipsVal) * chipSizeVal * nipSizeMult3 * (whaleMult3 > 1.0 ? Math.max(0.65, node.smoothChip3) : node.smoothChip3) * nipBlockDamp;
    writeCornerNipPlane(rowIdx, 20, kt0, kt1, tipHoriz, tipVert, nipDepth3, 0.90, 1.10, vertCount, L, r);

    const half = (0.51 + 0.21 * nodeDoubleTerm) * L;
    const midFrac = cubicBlend * 0.52 + (1 - cubicBlend) * (0.49 - 0.20 * nodeDoubleTerm);
    const nodeAlpha = Math.max(0.0, Math.min(1.0, node.alpha * smooth01(Math.min(1.0, presence / 0.55))));
    const scx = root[0] + ax[0] * midFrac * L;
    const scy = root[1] + ax[1] * midFrac * L;
    const scz = root[2] + ax[2] * midFrac * L;
    let maxVertDist2 = 0;
    for (let v = 0; v < vertCount; v++) {
      const vx = POLY_V[v * 3];
      const vy = POLY_V[v * 3 + 1];
      const vz = POLY_V[v * 3 + 2];
      const dvx = vx - scx;
      const dvy = vy - scy;
      const dvz = vz - scz;
      const d2 = dvx * dvx + dvy * dvy + dvz * dvz;
      if (d2 > maxVertDist2) maxVertDist2 = d2;
      if (nodeAlpha > 0.05) {
        if (vy < minY) minY = vy;
        if (vy > maxY) maxY = vy;
        const r2 = vx * vx + vz * vz;
        if (r2 > maxR2) maxR2 = r2;
      }
    }
    const sr = (Math.max(Math.hypot(half + r * 0.65, rc), Math.sqrt(maxVertDist2)) + (rowIdx === 0 ? 0.28 * meltVal : 0.08 * meltVal)) * 1.05;
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
    weightedSumY += (scy - 0.045 * meltVal) * w;
    weightedSumZ += scz * w;
    weightTotal += w;

    if (meltVal > 1e-3 && rowIdx === 0) {
      const puddleReachY = -0.34 * Math.min(1.0, meltVal * 1.25);
      const puddleReachXZ = 0.34 * Math.min(1.0, meltVal * 1.25);
      minY = Math.min(minY, puddleReachY);
      maxR2 = Math.max(maxR2, puddleReachXZ * puddleReachXZ);
    }

    if (vertCount === 0) {
      const tipY = root[1] + ax[1] * L;
      const botReach = (1 - cubicBlend) * (0.04 + 0.40 * nodeDoubleTerm) * L;
      const botY = root[1] - ax[1] * botReach;
      const effPad = r * (0.1 + 0.35 * nodeAlpha);
      minY = Math.min(minY, Math.min(botY, tipY) - effPad);
      maxY = Math.max(maxY, Math.max(botY, tipY) + effPad);
      const rTip = Math.hypot(root[0] + ax[0] * L, root[2] + ax[2] * L) + effPad;
      const rRoot = Math.hypot(root[0] - ax[0] * botReach, root[2] - ax[2] * botReach) + effPad;
      maxR2 = Math.max(maxR2, rTip * rTip, rRoot * rRoot);
    }
  }

  // Smooth 3D bounding-box auto-fit normalization so every preset fills ~68-76% of viewport height
  // on both desktop and mobile without clipping termination tips during 360° rotation.
  const rawSpanY = Number.isFinite(minY) && maxY > minY ? (maxY - minY) : 0.58;
  const rawRadiusXZ = Math.max(0.12, Math.sqrt(maxR2));
  const targetSpanY = (growSideways ? 0.59 : 0.80) * overScale;
  const maxSafeSpanY = (growSideways ? 0.62 : 0.84) * overScale;
  const minSafeSpanY = (growSideways ? 0.55 : 0.76) * overScale;
  const maxSafeRadiusXZ = (growSideways ? 0.58 : 0.265) * overScale;
  let desiredFitScale = 1.0;
  if (rawSpanY > maxSafeSpanY) {
    desiredFitScale = maxSafeSpanY / rawSpanY;
  } else if (rawSpanY < minSafeSpanY) {
    desiredFitScale = Math.min(1.45, targetSpanY / rawSpanY);
  }
  if (rawRadiusXZ * desiredFitScale > maxSafeRadiusXZ) {
    desiredFitScale = Math.min(desiredFitScale, maxSafeRadiusXZ / rawRadiusXZ);
  }
  desiredFitScale = Math.max(0.42, Math.min(1.45, desiredFitScale));
  if (!cluster.centerInit || !Number.isFinite(cluster.fitScale)) {
    cluster.fitScale = desiredFitScale;
  } else {
    const aFit = nodeDt > 0 ? (1 - Math.exp(-2.4 * nodeDt)) : (!evolveOn ? 0.15 : 0.04);
    cluster.fitScale += (desiredFitScale - cluster.fitScale) * aFit;
  }
  const fitScale = cluster.fitScale;
  if (Math.abs(fitScale - 1.0) > 1e-4) {
    const d = cluster.data;
    for (let rowIdx = 0; rowIdx < activeCount; rowIdx++) {
      const rBase = rowIdx * ROW_W * 4;
      d[rBase] *= fitScale;
      d[rBase + 1] *= fitScale;
      d[rBase + 2] *= fitScale;
      d[rBase + 3] *= fitScale;
      d[rBase + 4 + 3] *= fitScale; // L
      d[rBase + 8 + 3] *= fitScale; // r
      d[rBase + 12] *= fitScale;    // root.x
      d[rBase + 13] *= fitScale;    // root.y
      d[rBase + 14] *= fitScale;    // root.z
      const pBase = (rowIdx * ROW_W + PLANES_AT) * 4;
      for (let k = 0; k < CRYSTAL_PLANES; k++) {
        d[pBase + k * 4 + 3] *= fitScale;
      }
      sphereCentres[rowIdx * 4] *= fitScale;
      sphereCentres[rowIdx * 4 + 1] *= fitScale;
      sphereCentres[rowIdx * 4 + 2] *= fitScale;
      sphereCentres[rowIdx * 4 + 3] *= fitScale;
    }
    weightedSumX *= fitScale;
    weightedSumY *= fitScale;
    weightedSumZ *= fitScale;
    if (Number.isFinite(minY)) {
      minY *= fitScale;
      maxY *= fitScale;
    }
    maxR2 *= fitScale * fitScale;
  }

  cluster.crystals = Math.max(1, activeCount);
  cluster.activeVectors = cluster.crystals * CRYSTAL_PLANES;

  const targetCx = weightTotal > 1e-6 ? weightedSumX / weightTotal : 0;
  const midY = Number.isFinite(minY) ? 0.5 * (minY + maxY) : 0;
  const volCy = weightTotal > 1e-6 ? weightedSumY / weightTotal : midY;
  const targetCy = Number.isFinite(minY) ? (0.42 * volCy + 0.58 * midY) : volCy;
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
  if (key === 'habit' && P?.habit) {
    cluster.morphHabitBlend = 0;
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

let cachedWireframeVersion = -1;
let cachedWireframeBodies = [];
const wfNx = new Float64Array(18);
const wfNy = new Float64Array(18);
const wfNz = new Float64Array(18);
const wfW = new Float64Array(18);
const wfVx = new Float64Array(128);
const wfVy = new Float64Array(128);
const wfVz = new Float64Array(128);
const wfMask = new Int32Array(128);

export function getClusterWireframe() {
  if (cachedWireframeVersion === cluster.version && cachedWireframeBodies.length > 0) {
    return cachedWireframeBodies;
  }
  const d = cluster.data;
  const nCrystals = cluster.crystals;
  const bodies = [];

  for (let row = 0; row < nCrystals; row++) {
    const rBase = row * ROW_W * 4;
    const alpha = d[rBase + 8];
    if (alpha <= 0.02) continue;
    const axX = d[rBase + 4], axY = d[rBase + 5], axZ = d[rBase + 6], L = d[rBase + 7];
    const r = d[rBase + 11];
    const rtX = d[rBase + 12], rtY = d[rBase + 13], rtZ = d[rBase + 14];
    const pBase = (row * ROW_W + PLANES_AT) * 4;
    let np = 0;
    for (let k = 0; k < 18; k++) {
      const o = pBase + k * 4;
      const w = d[o + 3];
      if (w > 1e4) continue;
      wfNx[np] = d[o];
      wfNy[np] = d[o + 1];
      wfNz[np] = d[o + 2];
      wfW[np] = w;
      np++;
    }
    if (np < 4) continue;

    const epsMerge = Math.max(1e-3, r * 0.018);
    const epsMerge2 = epsMerge * epsMerge;
    const epsPlane = Math.max(1.2e-3, r * 0.015);
    let nv = 0;

    for (let i = 0; i < np - 2; i++) {
      const ax0 = wfNx[i], ay0 = wfNy[i], az0 = wfNz[i], wi = wfW[i];
      for (let j = i + 1; j < np - 1; j++) {
        const bx = wfNx[j], by = wfNy[j], bz = wfNz[j], wj = wfW[j];
        const abx = ay0 * bz - az0 * by, aby = az0 * bx - ax0 * bz, abz = ax0 * by - ay0 * bx;
        for (let k = j + 1; k < np; k++) {
          const cx = wfNx[k], cy = wfNy[k], cz = wfNz[k], wk = wfW[k];
          const bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;
          const det = ax0 * bcx + ay0 * bcy + az0 * bcz;
          if (Math.abs(det) < 1e-6) continue;
          const cax = cy * az0 - cz * ay0, cay = cz * ax0 - cx * az0, caz = cx * ay0 - cy * ax0;
          const inv = 1 / det;
          const vx = (wi * bcx + wj * cax + wk * abx) * inv;
          const vy = (wi * bcy + wj * cay + wk * aby) * inv;
          const vz = (wi * bcz + wj * caz + wk * abz) * inv;
          let inside = true;
          for (let m = 0; m < np; m++) {
            if (m === i || m === j || m === k) continue;
            if (wfNx[m] * vx + wfNy[m] * vy + wfNz[m] * vz > wfW[m] + 2e-4) {
              inside = false;
              break;
            }
          }
          if (!inside) continue;

          let foundIdx = -1;
          for (let u = 0; u < nv; u++) {
            const dx = wfVx[u] - vx, dy = wfVy[u] - vy, dz = wfVz[u] - vz;
            if (dx * dx + dy * dy + dz * dz <= epsMerge2) {
              foundIdx = u;
              break;
            }
          }
          const bitTrip = (1 << i) | (1 << j) | (1 << k);
          if (foundIdx >= 0) {
            wfMask[foundIdx] |= bitTrip;
          } else if (nv < 128) {
            wfVx[nv] = vx; wfVy[nv] = vy; wfVz[nv] = vz;
            let mask = bitTrip;
            for (let m = 0; m < np; m++) {
              if (Math.abs(wfNx[m] * vx + wfNy[m] * vy + wfNz[m] * vz - wfW[m]) <= epsPlane) {
                mask |= (1 << m);
              }
            }
            wfMask[nv] = mask;
            nv++;
          }
        }
      }
    }
    if (nv < 4) continue;

    const verts = new Float32Array(nv * 3);
    const vertHgt = new Float32Array(nv);
    for (let u = 0; u < nv; u++) {
      verts[u * 3] = wfVx[u];
      verts[u * 3 + 1] = wfVy[u];
      verts[u * 3 + 2] = wfVz[u];
      const axial = (wfVx[u] - rtX) * axX + (wfVy[u] - rtY) * axY + (wfVz[u] - rtZ) * axZ;
      vertHgt[u] = Math.max(0, Math.min(1, axial / Math.max(L, 0.08)));
    }

    const normals = new Float32Array(np * 3);
    for (let p = 0; p < np; p++) {
      normals[p * 3] = wfNx[p];
      normals[p * 3 + 1] = wfNy[p];
      normals[p * 3 + 2] = wfNz[p];
    }

    const faces = [];
    const edgeSet = new Map();
    for (let p = 0; p < np; p++) {
      const bit = 1 << p;
      const fVerts = [];
      let cx = 0, cy = 0, cz = 0;
      for (let u = 0; u < nv; u++) {
        if ((wfMask[u] & bit) !== 0) {
          fVerts.push(u);
          cx += wfVx[u]; cy += wfVy[u]; cz += wfVz[u];
        }
      }
      if (fVerts.length < 3) continue;
      cx /= fVerts.length; cy /= fVerts.length; cz /= fVerts.length;
      const nx = wfNx[p], ny = wfNy[p], nz = wfNz[p];
      let t1x = wfVx[fVerts[0]] - cx, t1y = wfVy[fVerts[0]] - cy, t1z = wfVz[fVerts[0]] - cz;
      const t1l = Math.hypot(t1x, t1y, t1z) || 1;
      t1x /= t1l; t1y /= t1l; t1z /= t1l;
      const t2x = ny * t1z - nz * t1y, t2y = nz * t1x - nx * t1z, t2z = nx * t1y - ny * t1x;
      fVerts.sort((a, b) => {
        const dax = wfVx[a] - cx, day = wfVy[a] - cy, daz = wfVz[a] - cz;
        const dbx = wfVx[b] - cx, dby = wfVy[b] - cy, dbz = wfVz[b] - cz;
        return Math.atan2(dax * t2x + day * t2y + daz * t2z, dax * t1x + day * t1y + daz * t1z) -
               Math.atan2(dbx * t2x + dby * t2y + dbz * t2z, dbx * t1x + dby * t1y + dbz * t1z);
      });
      faces.push({ plane: p, indices: fVerts, cx, cy, cz });
      for (let m = 0; m < fVerts.length; m++) {
        const u = fVerts[m], v = fVerts[(m + 1) % fVerts.length];
        const lo = Math.min(u, v), hi = Math.max(u, v);
        const key = (lo << 8) | hi;
        const prev = edgeSet.get(key);
        if (!prev) edgeSet.set(key, { u: lo, v: hi, pA: p, pB: p });
        else prev.pB = p;
      }
    }

    const edges = [...edgeSet.values()];
    bodies.push({
      row,
      alpha,
      verts,
      vertHgt,
      nv,
      normals,
      np,
      faces,
      edges,
      sphereZ: 0,
      cx: d[rBase],
      cy: d[rBase + 1],
      cz: d[rBase + 2],
    });
  }

  cachedWireframeVersion = cluster.version;
  cachedWireframeBodies = bodies;
  return bodies;
}
cluster.getWireframe = getClusterWireframe;
