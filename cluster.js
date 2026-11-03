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
function enumerateCrystalVertices(rowIdx, nSym) {
  const base = (rowIdx * ROW_W + PLANES_AT) * 4;
  const d = cluster.data;
  let n = 0;
  for (let k = 0; k < 18; k++) {
    if ((k < 8 && k > nSym) || (k >= 8 && k < 16 && k - 8 > nSym)) continue;
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

function primaryCountFor(n, habit, twinShare) {
  let nPrimary = n;
  if (n > 2) {
    if (habit === 'Scepter Overgrowth') {
      nPrimary = Math.max(1, Math.ceil(n * 0.55));
    } else if (habit === 'Twin Branching') {
      nPrimary = Math.max(1, Math.round(n * (1 - 0.58 * Math.max(0.35, twinShare))));
    } else if (habit === 'Dendritic Starburst') {
      nPrimary = Math.min(n, 7);
    } else if (twinShare > 0.02) {
      nPrimary = Math.max(1, Math.round(n * (1 - 0.45 * twinShare)));
    }
  }
  return nPrimary;
}

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
  const growSpd = P?.growSpeed !== undefined ? Math.max(0, Number(P.growSpeed.value)) : 0.04;

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

    // Lock each crystal's unit axis (axis0), orthonormal basis (u0, v0), twist (rot0),
    // and integer face symmetry (baseSides + faceAngles) permanently at initialization
    // so individual crystals NEVER rotate on their own axis or tilt relative to the cluster.
    if (!node.frameInit) {
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

      const isTwinNode = i >= 2 && rawTwinW >= 0.5;
      node.lockedTwinRole = isTwinNode ? 1 : 0;
      node.rot0 = node.uTwist;

      if (!isTwinNode) {
        ax[0] = primAx[0]; ax[1] = primAx[1]; ax[2] = primAx[2];
        buildSideBasis(node.rot0);
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
        buildSideBasis(node.rot0);
        node.branchRadial0 = new Float32Array([rx, ry, rz]);
      }

      node.axis0 = new Float32Array(ax);
      node.u0 = new Float32Array(sx);
      node.v0 = new Float32Array(sz);
      node.az0 = az0;
      const lockedN = Math.max(3, Math.min(8, Math.round(Number(node.birthSymmetry ?? nodeSymmetry) || 6)));
      node.baseSides = lockedN;
      node.faceAngles = new Float32Array(8);
      for (let k = 0; k < 8; k++) {
        node.faceAngles[k] = k < lockedN ? (TAU * k) / lockedN : 0;
      }
      node.frameInit = true;
    }

    // Use the permanently locked crystal axis & orthonormal side basis (zero axial rotation!)
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
    let rMax = 0;

    // Fixed integer prism symmetry & fixed azimuth angles `node.faceAngles[k]` per crystal
    // so prism and termination facets NEVER revolve or sweep around the crystal's axis!
    const nSym = node.baseSides || Math.max(3, Math.min(8, Math.round(Number(nodeSymmetry) || 6)));
    const taperLy = Math.max(-0.32, Math.min(0.32, taperParam * 0.38));
    const latScale = Math.sqrt(Math.max(0.1, 1 - taperLy * taperLy));
    const faceJitterAmt = Math.max(0.08, P.faceJitter.value);

    // Slots 0..7: prism shaft faces at 100% FIXED azimuth angles `a` around the crystal axis
    for (let k = 0; k < 8; k++) {
      if (k < nSym) {
        const a = node.faceAngles ? node.faceAngles[k] : (TAU * k) / nSym;
        const fSeed = fs[k];
        const faceR = r * Math.max(0.35, 1 + faceJitterAmt * fSeed);
        const midOffset = Math.max(r * 0.22, faceR + 0.5 * L * taperLy);
        const topR = faceR - 0.5 * L * taperLy;
        const botR = faceR + 0.5 * L * taperLy;
        rMax = Math.max(rMax, faceR, topR, botR);
        crystalPlane(rowIdx, k, latScale * Math.cos(a), taperLy, latScale * Math.sin(a), midOffset);
      } else {
        crystalPlane(rowIdx, k, 0, 1, 0, L * 12 + r * 12);
      }
    }

    // Slots 8..15: crown pyramid termination faces at 100% FIXED azimuth angles `a`
    const nodeTilt = Math.max(20 * DEG, Math.min(84 * DEG, tilt));
    const cosNT = Math.cos(nodeTilt), sinNT = Math.sin(nodeTilt);
    const dynDom = P.faceDominance.value;
    const dynTipJitter = Math.max(0.06, P.tipJitter.value);
    for (let k = 0; k < 8; k++) {
      if (k < nSym) {
        const a = node.faceAngles ? node.faceAngles[k] : (TAU * k) / nSym;
        const domWeight = (k & 1) ? 0.85 : 0.25;
        const tSeed = fs[8 + k];
        const big = domWeight * dynDom * r;
        const tipOff = L * cosNT + big + dynTipJitter * r * 0.45 * tSeed;
        crystalPlane(rowIdx, 8 + k, sinNT * Math.cos(a), cosNT, sinNT * Math.sin(a), Math.max(0.02 * L, tipOff));
      } else {
        crystalPlane(rowIdx, 8 + k, 0, 1, 0, L * 12 + r * 12);
      }
    }

    // Slot 16: Root base termination plane
    const rootOff = Math.max(0.01 * r, dynDom * r * 0.35 + Math.abs(dynTipJitter * r * 0.2 * fs[16]));
    crystalPlane(rowIdx, 16, 0, -1, 0, rootOff);

    // Slot 17: Continuous top crown table / bevel truncation plane (no discrete chamfer threshold)
    const dynChamfer = Math.max(0, chamfer);
    const tableFrac = Math.max(0.68, 1.12 - 0.58 * dynChamfer);
    crystalPlane(rowIdx, 17, 0, 1, 0, L * tableFrac);

    // Slots 18..20: Literal 3D geometric corner-nip half-space planes (scaled by P.chipSize and macro faders)
    const macroEdgeScale = (P.patinaMacro ? Number(P.patinaMacro.value) : 1.0) * (P.edgeMacro ? Number(P.edgeMacro.value) : 1.0);
    const edgeChipsVal = Math.max(0, Math.min(2, (P.edgeChips ? Number(P.edgeChips.value) : 0.28) * macroEdgeScale));
    const chipSizeVal = P.chipSize ? Math.max(0.1, Math.min(3.0, Number(P.chipSize.value))) : 1.5;
    const rc = rMax / Math.cos(Math.PI / Math.max(3, nSym));
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
    const vertCount = enumerateCrystalVertices(rowIdx, nSym);

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
    const sr = Math.hypot(half + r * 0.65, rc) * 1.04;
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
  if (key === 'habit' && P?.habit) {
    const h = String(P.habit.value);
    const desiredCount = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));
    const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
    const np = primaryCountFor(desiredCount, h, twinShare);
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].birthHabit = h;
      cluster.nodes[i].birthNPrimary = np;
      cluster.nodes[i].frameInit = false;
    }
  } else if (key === 'symmetry' && P?.symmetry) {
    const s = Math.max(3, Math.min(8, Math.round(Number(P.symmetry.value))));
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].birthSymmetry = s;
      cluster.nodes[i].frameInit = false;
    }
  } else if (key === 'fan' || key === 'twinAngle' || key === 'cavityCurve') {
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].frameInit = false;
    }
  }
};
cluster.triggerGrowBurst = triggerGrowBurst;
cluster.surgeBranch = triggerGrowBurst;

