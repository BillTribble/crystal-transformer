// Crystals — multi-habit procedural crystal & twin-cluster geometry builder with
// bounded sliding-window evolution (Phase 1: Seed -> Full Size, Phase 2: Continuous
// Frontier Branching + Smooth Old-Segment Dissolve & Garbage Collection).
//
// Every crystal body is the exact intersection of 18 half-spaces (CRYSTAL_PLANES = 18),
// written into `cluster.data` (a std140 uniform buffer read as `bodies[]` in the trace shader).
// Slot layout per crystal row (`ROW_W = 22` vec4s):
//   col 0: vec4(sphereCentre.xyz, sphereRadius)
//   col 1: vec4(axis.xyz, L)
//   col 2: vec4(alpha, hueOffset, stableUid, r)
//   col 3: vec4(root.xyz, rnd)
//   col 4..21: 18 half-space planes vec4(normal.xyz, planeOffset)

export const MAX_CRYSTALS = 28;
export const CRYSTAL_PLANES = 18;
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

function buildSideBasis(twistAngle) {
  let px = -ax[2], pz = ax[0], pl = Math.hypot(px, pz);
  if (pl < 1e-4) { px = 1; pz = 0; pl = 1; }
  px /= pl; pz /= pl;
  const qx = ax[1] * pz, qy = ax[2] * px - ax[0] * pz, qz = -ax[1] * px;
  const c = Math.cos(twistAngle), s = Math.sin(twistAngle);
  sx[0] = c * px + s * qx; sx[1] = s * qy; sx[2] = c * pz + s * qz;
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
    presence: 1,
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
  cluster.evoBurstVel = Math.min(2.4, (cluster.evoBurstVel || 0) + 0.65);
  cluster.version++;
}

export function stepEvolution(P, dt, viewportW = 0, viewportH = 0) {
  if (cluster.nodes.length < MAX_CRYSTALS) {
    resetEvolution(P, { startFromSeed: false });
  }

  const safeDt = Math.max(0, Math.min(Number(dt) || 0, 30));
  cluster.pendingCenterDt = (cluster.pendingCenterDt || 0) + safeDt;

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
  const floatC = Math.max(1.0, Math.min(MAX_CRYSTALS, P?.crystals ? Number(P.crystals.value) : 9.0));
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

  const effPrimaryCount = Math.max(1.8, floatC * (1.0 - 0.38 * twinShare));
  let activeCount = 0;

  for (let idx = 0; idx < nActive; idx++) {
    if (activeCount >= MAX_CRYSTALS) break;
    const node = cluster.nodes[idx];
    const i = node.cohortIdx;
    const nodeSymmetry = symmetry;

    // Continuous presence factor for crystal idx: smoothly emerges from 0 out of the core
    // as floatC crosses idx..idx+1, and smoothly shrinks back to 0 before removal.
    const presence = smooth01(Math.max(0.0, Math.min(1.0, floatC - i)));
    node.presence = presence;
    if (presence < 1e-3) {
      node.worldL = 0;
      node.worldFullL = 0;
      node.worldR = 0;
      continue;
    }

    // Combine node lifecycle growth with manual P.growth slider
    const manualProgress = (!evolveOn && growth < 1.0)
      ? smooth01((growth - node.birthThreshold * 0.85) / Math.max(0.15, node.birthSpan))
      : 1.0;
    const localGrowth = evolveOn ? node.growProgress : manualProgress;
    const gScale = (i === 0 && node.isInitialCohort ? Math.max(0.08, localGrowth) : localGrowth) * overScale;

    if (localGrowth <= 1e-4 || node.alpha <= 1e-4 || gScale <= 1e-4) {
      continue;
    }

    // Continuous whole-cluster organic phase waves for crystal i (golden-ratio desynchronized C-infinity harmonics)
    const phi = cluster.evoPhase || 0;
    const s1 = i * GOLDEN + node.uLen * TAU;
    const s2 = i * GOLDEN * 1.6180339887 + (node.uAz + 1) * Math.PI;
    const s3 = i * GOLDEN * 2.6180339887 + (node.uLean + 1) * Math.PI;
    const w1 = Math.sin(phi * 0.85 + s1);
    const w2 = Math.sin(phi * 0.53 + s2);
    const w3 = Math.cos(phi * 1.18 + s3);
    const w4 = Math.sin(phi * 1.42 + s1 + s2 * 0.5);

    const lenWave = 0.62 * w1 + 0.38 * w3;
    const radWave = 0.64 * w2 - 0.36 * w1;
    const tiltWave = 0.60 * w3 + 0.40 * w2;
    const azWave = 0.65 * w2 + 0.35 * w4;
    const lenMod = 0.95 + 0.40 * lenWave;
    const radMod = 1.0 + 0.25 * radWave;

    // 1. Compute Primary Root Transform (primRoot, primAx, primBaseL, primBaseR, primTw)
    const rho = i === 0 ? 0.12 * node.uRho : Math.min(1.0, Math.sqrt((i + 0.5 * node.uRho) / effPrimaryCount));
    node.rho = rho;
    const az = i * GOLDEN + 0.55 * node.uAz + 0.14 * azWave;
    let lean = fan * Math.min(1, rho + 0.28 * node.uLean * (i > 0 ? 1 : 0.25));
    const smoothAbsTilt = Math.sqrt(tiltWave * tiltWave + 0.01) - 0.1;
    lean = i === 0
      ? Math.max(0.02, lean * (1 + 0.18 * tiltWave) + 0.10 * fan * smoothAbsTilt)
      : lean * (1 + 0.18 * tiltWave);

    if (growSideways) {
      if (i > 0) {
        const targetSideLean = Math.min(84 * DEG, Math.max(lean, (56 + 26 * rho) * DEG));
        lean = lean + (targetSideLean - lean) * dirProgress * Math.min(1, 0.45 + 0.55 * rho);
      }
    } else {
      lean = lean * (1 - 0.58 * dirProgress);
    }

    const leanAz = az + 0.35 * node.uLeanAz + 0.14 * tiltWave;
    primAx[0] = Math.sin(lean) * Math.cos(leanAz);
    primAx[1] = Math.cos(lean);
    primAx[2] = Math.sin(lean) * Math.sin(leanAz);

    let primBaseL = (i === 0 ? 1 : Math.max(0.18, 1 - P.lengthJitter.value * (0.4 * node.uLen + 0.6 * rho))) * lenMod;
    const horizComp = Math.hypot(primAx[0], primAx[2]);
    const vertComp = Math.abs(primAx[1]);
    if (growSideways) {
      primBaseL *= 1 + dirProgress * (0.78 * horizComp * Math.min(1.9, aspect) - 0.16 * vertComp);
    } else {
      primBaseL *= 1 + dirProgress * (0.85 * vertComp * Math.min(1.9, 1 / aspect) - 0.16 * horizComp);
    }

    const primBaseR = P.thickness.value * Math.pow(primBaseL, 0.65) * Math.max(0.25, 1 + P.thicknessJitter.value * 0.55 * node.uRadSym) * radMod;
    const primTw = node.uTwist + 0.22 * w4;

    ax[0] = primAx[0]; ax[1] = primAx[1]; ax[2] = primAx[2];
    buildSideBasis(primTw);

    const curPrimL = primBaseL * gScale * presence;
    const curPrimR = primBaseR * Math.pow(gScale, 0.78) * presence;
    const back = P.crossAt.value * curPrimL;
    const u = node.uRootU * curPrimR * 1.05, v = node.uRootV * curPrimR * 1.05;
    for (let k = 0; k < 3; k++) {
      primRoot[k] = -primAx[k] * back + sx[k] * u + sz[k] * v + node.driftOffset[k];
    }

    if (dirProgress > 0.01 && i > 0) {
      if (growSideways) {
        const latDist = dirProgress * halfFrustumX * 0.96 * rho * localGrowth * presence;
        primRoot[0] += Math.cos(az) * latDist;
        primRoot[2] += Math.sin(az) * latDist;
        primRoot[1] *= (1 - 0.42 * dirProgress);
      } else {
        const tier = ((i % 2 === 1) ? 1 : -0.88) * Math.pow(rho, 0.78);
        const vertDist = dirProgress * halfFrustumY * 0.98 * tier * localGrowth * presence;
        primRoot[1] += vertDist;
        primRoot[0] *= (1 - 0.42 * dirProgress);
        primRoot[2] *= (1 - 0.42 * dirProgress);
      }
    }

    const curve = cavityCurve;
    if (curve > 0.001) {
      const bowlR = 0.52 * curve * (0.35 + 0.65 * gScale) * presence;
      const ringR = bowlR * rho;
      const bowlY = -bowlR * Math.cos(rho * 1.15) + bowlR * 0.45;
      primRoot[0] += Math.cos(az) * ringR;
      primRoot[1] += bowlY;
      primRoot[2] += Math.sin(az) * ringR;
      const inTilt = curve * 0.45 * rho;
      primAx[0] = primAx[0] * (1 - inTilt) - Math.cos(az) * inTilt * 0.5;
      primAx[2] = primAx[2] * (1 - inTilt) - Math.sin(az) * inTilt * 0.5;
      const al = Math.hypot(primAx[0], primAx[1], primAx[2]) || 1;
      primAx[0] /= al; primAx[1] /= al; primAx[2] /= al;
    }

    // 2. Continuous twinShare transition (twinW in [0, 1]): smoothly blends between
    // primary root crystal and secondary twin branch with zero discrete jumps.
    const twinW = i < 2 ? 0.0 : smooth01(Math.max(0.0, Math.min(1.0, (twinShare - node.twinRoll + 0.18) / 0.36)));
    node.twinW = twinW;
    node.isSecondary = twinW >= 0.5;

    let baseL = primBaseL;
    let baseR = primBaseR;
    let tw = primTw;

    if (twinW <= 1e-5) {
      for (let k = 0; k < 3; k++) {
        root[k] = primRoot[k];
        ax[k] = primAx[k];
      }
      buildSideBasis(tw);
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
        node.savedParentTwist = pNode.worldTwist;
      }

      const pRoot = node.savedParentRoot;
      const pAxis = node.savedParentAxis;
      const pL = node.savedParentL;
      const pFullL = node.savedParentFullL;
      const pR = node.savedParentR;
      const pTwist = node.savedParentTwist;

      const branchAz = i * GOLDEN + 0.35 * node.uBranchAz + 0.14 * azWave;
      const bAngle = twinAngle * (0.85 + 0.3 * node.uBranchAngle) * (1 + 0.18 * tiltWave);
      const cosB = Math.cos(bAngle), sinB = Math.sin(bAngle);

      for (let k = 0; k < 3; k++) ax[k] = pAxis[k];
      buildSideBasis(pTwist);
      const rx = sx[0] * Math.cos(branchAz) + sz[0] * Math.sin(branchAz);
      const ry = sx[1] * Math.cos(branchAz) + sz[1] * Math.sin(branchAz);
      const rz = sx[2] * Math.cos(branchAz) + sz[2] * Math.sin(branchAz);

      const attachFrac = Math.max(0.14, Math.min(0.78, 0.24 + 0.44 * node.uAttach + 0.09 * lenWave));
      const attachDist = pL * attachFrac;
      for (let k = 0; k < 3; k++) {
        twinRoot[k] = pRoot[k] + pAxis[k] * attachDist + (k === 0 ? rx : k === 1 ? ry : rz) * (pR * 0.35) + node.driftOffset[k];
      }

      twinAx[0] = pAxis[0] * cosB + rx * sinB;
      twinAx[1] = pAxis[1] * cosB + ry * sinB;
      twinAx[2] = pAxis[2] * cosB + rz * sinB;

      // Smooth C2 radial bounding envelope
      const estL = pFullL * 0.58;
      const tipX = twinRoot[0] + twinAx[0] * estL;
      const tipY = twinRoot[1] + twinAx[1] * estL;
      const tipZ = twinRoot[2] + twinAx[2] * estL;
      const tipDist = Math.hypot(tipX, tipY, tipZ);
      if (tipDist > 0.72) {
        const steer = 0.68 * smooth01(Math.min(1.0, (tipDist - 0.72) / 0.45));
        const invR = 1 / Math.max(1e-4, Math.hypot(twinRoot[0], twinRoot[1], twinRoot[2]));
        twinAx[0] = twinAx[0] * (1 - steer) - twinRoot[0] * invR * steer;
        twinAx[1] = twinAx[1] * (1 - steer) - twinRoot[1] * invR * steer * 0.7;
        twinAx[2] = twinAx[2] * (1 - steer) - twinRoot[2] * invR * steer;
      }

      if (dirProgress > 0.01) {
        if (growSideways) {
          twinAx[1] *= (1 - 0.55 * dirProgress);
        } else {
          twinAx[0] *= (1 - 0.55 * dirProgress);
          twinAx[2] *= (1 - 0.55 * dirProgress);
          twinAx[1] += (0.72 - twinAx[1]) * 0.48 * dirProgress;
        }
      }
      const tal = Math.hypot(twinAx[0], twinAx[1], twinAx[2]) || 1;
      twinAx[0] /= tal; twinAx[1] /= tal; twinAx[2] /= tal;

      const twinTw = pTwist + branchAz + 0.22 * w4;
      const twinBaseL = pFullL * (0.44 + 0.32 * node.uBranchLen) * (1 + 0.32 * dirProgress) * lenMod;
      const twinBaseR = Math.max(0.035, P.thickness.value * 0.85) * (0.68 + 0.28 * node.uBranchRad) * radMod;

      for (let k = 0; k < 3; k++) {
        root[k] = primRoot[k] + (twinRoot[k] - primRoot[k]) * twinW;
        ax[k] = primAx[k] + (twinAx[k] - primAx[k]) * twinW;
      }
      const al = Math.hypot(ax[0], ax[1], ax[2]) || 1;
      ax[0] /= al; ax[1] /= al; ax[2] /= al;
      baseL = primBaseL + (twinBaseL - primBaseL) * twinW;
      baseR = primBaseR + (twinBaseR - primBaseR) * twinW;
      tw = primTw + (twinTw - primTw) * twinW;
      buildSideBasis(tw);
    }

    // Apply procedural growth scaling and continuous crystal count presence to L and r
    const L = Math.max(1e-4, baseL * gScale * presence);
    const r = Math.max(1e-4, baseR * Math.pow(gScale, 0.78) * presence);

    // Save computed world frame on node for child branches
    for (let k = 0; k < 3; k++) {
      node.worldRoot[k] = root[k];
      node.worldAxis[k] = ax[k];
      node.worldSide[k] = sx[k];
    }
    node.worldL = L;
    node.worldFullL = baseL * presence;
    node.worldR = r;
    node.worldTwist = tw;

    const rowIdx = activeCount++;
    const fs = node.faceSeeds;
    let rMax = 0;

    // Unified continuous "opening-out" polygon symmetry (3.0 -> 4.0 -> 5.0 -> 6.0 -> 8.0 faces)
    const sSym = Math.max(3.0, Math.min(8.0, Number(nodeSymmetry) || 6.0));
    const nSym = Math.min(7, Math.floor(sSym));
    const fSym = Math.max(0.0, Math.min(1.0, sSym - nSym));
    const wSym = smooth01(fSym);

    const taperLy = Math.max(-0.32, Math.min(0.32, (taperParam + 0.08 * w2) * 0.38));
    const latScale = Math.sqrt(Math.max(0.1, 1 - taperLy * taperLy));
    const dynFaceJitter = Math.max(0.08, P.faceJitter.value * (1 + 0.22 * radWave));

    // Slots 0..7: up to 8 prism shaft faces (smoothly morphing face offsets with evoPhase)
    for (let k = 0; k < 8; k++) {
      if (k <= nSym) {
        const a = (TAU * k) / sSym;
        const fWaveK = k < nSym
          ? Math.sin(phi * 1.15 + s1 + k * GOLDEN)
          : ((1 - wSym) * Math.sin(phi * 1.15 + s1) + wSym * Math.sin(phi * 1.15 + s1 + nSym * GOLDEN));
        const fSeedBase = k < nSym ? fs[k] : (fs[0] + (fs[nSym] - fs[0]) * wSym);
        const fSeed = 0.65 * fSeedBase + 0.35 * fWaveK;
        const emergePad = k === nSym ? (1 - wSym) * r * 0.06 : 0;
        const faceR = r * Math.max(0.35, 1 + dynFaceJitter * fSeed) + emergePad;
        const midOffset = Math.max(r * 0.22, faceR + 0.5 * L * taperLy);
        const topR = faceR - 0.5 * L * taperLy;
        const botR = faceR + 0.5 * L * taperLy;
        rMax = Math.max(rMax, faceR, topR, botR);
        crystalPlane(rowIdx, k, latScale * Math.cos(a), taperLy, latScale * Math.sin(a), midOffset);
      } else {
        crystalPlane(rowIdx, k, 0, 1, 0, L * 12 + r * 12);
      }
    }

    // Slots 8..15: up to 8 crown pyramid termination faces aligned with the opening-out prism faces
    const nodeTilt = Math.max(20 * DEG, Math.min(84 * DEG, tilt * (1 + 0.08 * tiltWave)));
    const cosNT = Math.cos(nodeTilt), sinNT = Math.sin(nodeTilt);
    const dynDom = P.faceDominance.value * (1 + 0.28 * w1);
    const dynTipJitter = Math.max(0.06, P.tipJitter.value * (1 + 0.22 * w3));
    const domPhase0 = 0.5 + 0.5 * Math.sin(phi * 0.95 + s3);
    for (let k = 0; k < 8; k++) {
      if (k <= nSym) {
        const a = (TAU * k) / sSym;
        const domPhase = 0.5 + 0.5 * Math.sin(phi * 0.95 + s3 + k * Math.PI);
        const domWeight0 = 0.35 * domPhase0;
        const domWeightK = 0.65 * (k & 1) + 0.35 * domPhase;
        const domWeight = k < nSym ? domWeightK : (domWeight0 + (domWeightK - domWeight0) * wSym);
        const tWaveK = k < nSym
          ? Math.cos(phi * 1.32 + s2 + k * GOLDEN)
          : ((1 - wSym) * Math.cos(phi * 1.32 + s2) + wSym * Math.cos(phi * 1.32 + s2 + nSym * GOLDEN));
        const tSeedBase = k < nSym ? fs[8 + k] : (fs[8] + (fs[8 + nSym] - fs[8]) * wSym);
        const tSeed = 0.62 * tSeedBase + 0.38 * tWaveK;
        const emergeTipPad = k === nSym ? (1 - wSym) * r * 0.06 : 0;
        const big = domWeight * dynDom * r;
        const tipOff = L * cosNT + big + dynTipJitter * r * 0.45 * tSeed + emergeTipPad;
        crystalPlane(rowIdx, 8 + k, sinNT * Math.cos(a), cosNT, sinNT * Math.sin(a), Math.max(0.02 * L, tipOff));
      } else {
        crystalPlane(rowIdx, 8 + k, 0, 1, 0, L * 12 + r * 12);
      }
    }

    // Slot 16: Root base termination plane
    const rootOff = Math.max(0.01 * r, dynDom * r * 0.35 + Math.abs(dynTipJitter * r * 0.2 * fs[16]));
    crystalPlane(rowIdx, 16, 0, -1, 0, rootOff);

    // Slot 17: Continuous top crown table / bevel truncation plane (no discrete chamfer threshold)
    const dynChamfer = Math.max(0, chamfer * (1 + 0.22 * w2));
    const tableFrac = Math.max(0.68, 1.12 - 0.58 * dynChamfer);
    crystalPlane(rowIdx, 17, 0, 1, 0, L * tableFrac);

    const half = 0.5 * L, rc = rMax / Math.cos(Math.PI / Math.max(3, sSym));
    const nodeAlpha = Math.max(0.0, Math.min(1.0, node.alpha * smooth01(Math.min(1.0, presence / 0.35))));
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
  for (let idx = nActive; idx < cluster.nodes.length; idx++) {
    cluster.nodes[idx].presence = 0;
    cluster.nodes[idx].worldL = 0;
    cluster.nodes[idx].worldFullL = 0;
    cluster.nodes[idx].worldR = 0;
  }

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
    }
  } else if (key === 'symmetry' && P?.symmetry) {
    const s = Math.max(3, Math.min(8, Number(P.symmetry.value)));
    for (let i = 0; i < cluster.nodes.length; i++) {
      cluster.nodes[i].birthSymmetry = s;
    }
  }
};
cluster.triggerGrowBurst = triggerGrowBurst;
cluster.surgeBranch = triggerGrowBurst;

