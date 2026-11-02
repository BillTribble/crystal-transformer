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
  centreY: 0,
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
  const uid = ++nextUid;
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

  const rho = i === 0 ? 0.12 * rand() : Math.sqrt((i + 0.5 * rand()) / Math.max(1, nPrimary));
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
    parentUid: parentNode ? parentNode.uid : -1,
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
    // Morphological genes rolled once at birth so crystal shape is stable across pool compaction
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
    // Cached world frame (updated each frame so child branches can anchor even after parent dissolves)
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

export function resetEvolution(P, _opts = {}) {
  const seedVal = P?.seed ? Number(P.seed.value) : 7;
  seedRandom(seedVal);
  lastSeedVal = seedVal;
  masterTwist = rand() * TAU;
  spawnTimer = 0;

  const evolveOn = P?.evolve ? !!P.evolve.value : true;
  const nBase = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));
  lastStaticCount = nBase;
  const habit = P?.habit ? String(P.habit.value) : 'Quartz Cluster';
  const symmetry = P?.symmetry ? Math.max(3, Math.min(8, Number(P.symmetry.value))) : 6;
  const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
  const nPrimary = primaryCountFor(nBase, habit, twinShare);

  cluster.nodes = [];
  cluster.evolveTime = 0;
  cluster.evoPhase = 0;
  cluster.evoBurstVel = 0;
  cluster.spawnedCount = nBase;
  cluster.dissolvedCount = 0;
  cluster.generation = nBase > nPrimary ? 1 : 0;
  cluster.seedProgress = 1.0;
  cluster.phase = !evolveOn ? 'static' : 'evolving';

  for (let i = 0; i < nBase; i++) {
    const isSecondary = i >= nPrimary;
    const parentNode = isSecondary ? cluster.nodes[(i - nPrimary) % nPrimary] : null;
    const node = createInitialCohortNode(i, nBase, nPrimary, parentNode, false, habit, symmetry);
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
  if (!cluster.nodes.length) {
    resetEvolution(P, { startFromSeed: false });
  }

  const evolveOn = P?.evolve ? !!P.evolve.value : true;
  const desiredCount = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));
  const habit = P?.habit ? String(P.habit.value) : 'Quartz Cluster';
  const symmetry = P?.symmetry ? Math.max(3, Math.min(8, Number(P.symmetry.value))) : 6;
  const twinShare = P?.twinShare ? Number(P.twinShare.value) : 0.3;
  const nPrimary = primaryCountFor(desiredCount, habit, twinShare);

  // Keep cluster.nodes matched to desiredCount without single-crystal popcorn spawning/dissolving
  if (cluster.nodes.length !== desiredCount) {
    while (cluster.nodes.length < desiredCount) {
      const idx = cluster.nodes.length;
      const isSec = idx >= nPrimary;
      const pNode = isSec ? cluster.nodes[(idx - nPrimary) % Math.max(1, nPrimary)] : null;
      const node = createInitialCohortNode(idx, desiredCount, nPrimary, pNode, false, habit, symmetry);
      node.growProgress = 1.0;
      node.dissolveProgress = 0.0;
      node.alpha = 1.0;
      node.lifecycle = 'mature';
      cluster.nodes.push(node);
    }
    if (cluster.nodes.length > desiredCount) {
      cluster.nodes.length = desiredCount;
    }
    for (let i = 0; i < cluster.nodes.length; i++) {
      const isSec = i >= nPrimary;
      cluster.nodes[i].cohortIdx = i;
      cluster.nodes[i].isSecondary = isSec;
      cluster.nodes[i].parentUid = isSec ? cluster.nodes[(i - nPrimary) % Math.max(1, nPrimary)].uid : -1;
    }
  }

  if (!evolveOn) {
    cluster.phase = 'static';
    lastStaticCount = desiredCount;
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

  let rem = Math.max(0, Math.min(dt, 30));
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
  const desiredCount = Math.max(1, Math.min(MAX_CRYSTALS, Math.round(P?.crystals ? Number(P.crystals.value) : 9)));

  if (evolveOn) {
    lastStaticCount = desiredCount;
    if (seedVal !== lastSeedVal && cluster.nodes.length > 0) {
      seedRandom(seedVal + cluster.spawnedCount * 31);
      lastSeedVal = seedVal;
    }
  }
  if (!cluster.nodes.length || (!evolveOn && (seedVal !== lastSeedVal || desiredCount !== lastStaticCount))) {
    resetEvolution(P, { startFromSeed: false });
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

  const habit = P.habit ? String(P.habit.value) : 'Quartz Cluster';
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
  const tilt = P.tipAngle.value * DEG, cosT = Math.cos(tilt), sinT = Math.sin(tilt);

  let minY = Infinity, maxY = -Infinity, maxR2 = 0;
  let weightedSumY = 0, weightTotal = 0;

  // Map uid -> node for live parent lookups
  const byUid = new Map();
  for (let i = 0; i < cluster.nodes.length; i++) {
    byUid.set(cluster.nodes[i].uid, cluster.nodes[i]);
  }

  const nPrimary = primaryCountFor(desiredCount, habit, twinShare);
  let activeCount = 0;

  for (let idx = 0; idx < cluster.nodes.length; idx++) {
    if (activeCount >= MAX_CRYSTALS) break;
    const node = cluster.nodes[idx];
    const i = node.cohortIdx;
    const nodeSymmetry = symmetry;

    // Combine node lifecycle growth/dissolve with manual P.growth slider
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

    // Smooth per-crystal length (0.55x..1.35x), thickness (0.75x..1.25x), tilt (±18%), azimuth (±0.14 rad), and twist modulation
    const lenWave = 0.62 * w1 + 0.38 * w3;
    const radWave = 0.64 * w2 - 0.36 * w1;
    const tiltWave = 0.60 * w3 + 0.40 * w2;
    const azWave = 0.65 * w2 + 0.35 * w4;
    const lenMod = 0.95 + 0.40 * lenWave;
    const radMod = 1.0 + 0.25 * radWave;

    let baseL = 1;
    let baseR = P.thickness.value;
    let tw = node.uTwist + 0.22 * w4;

    if (!node.isSecondary) {
      const rho = node.rho;
      const az = i * GOLDEN + 0.55 * node.uAz + 0.14 * azWave;
      let lean = fan * Math.min(1, rho + 0.28 * node.uLean * (i > 0 ? 1 : 0.25));
      lean = i === 0
        ? Math.max(0.02, lean * (1 + 0.18 * tiltWave) + 0.10 * fan * Math.abs(tiltWave))
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
      ax[0] = Math.sin(lean) * Math.cos(leanAz);
      ax[1] = Math.cos(lean);
      ax[2] = Math.sin(lean) * Math.sin(leanAz);

      baseL = (i === 0 ? 1 : Math.max(0.18, 1 - P.lengthJitter.value * (0.4 * node.uLen + 0.6 * rho))) * lenMod;

      const horizComp = Math.hypot(ax[0], ax[2]);
      const vertComp = Math.abs(ax[1]);
      if (growSideways) {
        baseL *= 1 + dirProgress * (0.78 * horizComp * Math.min(1.9, aspect) - 0.16 * vertComp);
      } else {
        baseL *= 1 + dirProgress * (0.85 * vertComp * Math.min(1.9, 1 / aspect) - 0.16 * horizComp);
      }

      baseR = P.thickness.value * Math.pow(baseL, 0.65) * Math.max(0.25, 1 + P.thicknessJitter.value * 0.55 * node.uRadSym) * radMod;

      buildSideBasis(tw);

      const curL = baseL * gScale;
      const curR = baseR * Math.pow(gScale, 0.78);
      const back = P.crossAt.value * curL;
      const u = node.uRootU * curR * 1.05, v = node.uRootV * curR * 1.05;
      for (let k = 0; k < 3; k++) root[k] = -ax[k] * back + sx[k] * u + sz[k] * v + node.driftOffset[k];

      if (dirProgress > 0.01 && i > 0) {
        if (growSideways) {
          const latDist = dirProgress * halfFrustumX * 0.96 * rho * localGrowth;
          root[0] += Math.cos(az) * latDist;
          root[2] += Math.sin(az) * latDist;
          root[1] *= (1 - 0.42 * dirProgress);
        } else {
          const tier = ((i % 2 === 1) ? 1 : -0.88) * Math.pow(rho, 0.78);
          const vertDist = dirProgress * halfFrustumY * 0.98 * tier * localGrowth;
          root[1] += vertDist;
          root[0] *= (1 - 0.42 * dirProgress);
          root[2] *= (1 - 0.42 * dirProgress);
        }
      }

      const curve = cavityCurve;
      if (curve > 0.001) {
        const bowlR = 0.52 * curve * (0.35 + 0.65 * gScale);
        const ringR = bowlR * rho;
        const bowlY = -bowlR * Math.cos(rho * 1.15) + bowlR * 0.45;
        root[0] += Math.cos(az) * ringR;
        root[1] += bowlY;
        root[2] += Math.sin(az) * ringR;
        const inTilt = curve * 0.45 * rho;
        ax[0] = ax[0] * (1 - inTilt) - Math.cos(az) * inTilt * 0.5;
        ax[2] = ax[2] * (1 - inTilt) - Math.sin(az) * inTilt * 0.5;
        const al = Math.hypot(ax[0], ax[1], ax[2]) || 1;
        ax[0] /= al; ax[1] /= al; ax[2] /= al;
        buildSideBasis(tw);
      }
    } else {
      // Secondary generation or twin branch: look up live parent or fallback to saved parent frame
      const liveParent = byUid.get(node.parentUid);
      if (liveParent && liveParent.worldL > 0.02) {
        for (let k = 0; k < 3; k++) {
          node.savedParentRoot[k] = liveParent.worldRoot[k];
          node.savedParentAxis[k] = liveParent.worldAxis[k];
        }
        node.savedParentL = liveParent.worldL;
        node.savedParentFullL = Math.max(0.35, Math.min(0.95, liveParent.worldFullL));
        node.savedParentR = Math.max(0.035, Math.min(0.18, liveParent.worldR));
        node.savedParentTwist = liveParent.worldTwist;
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
        root[k] = pRoot[k] + ax[k] * attachDist + (k === 0 ? rx : k === 1 ? ry : rz) * (pR * 0.35) + node.driftOffset[k];
      }

      ax[0] = pAxis[0] * cosB + rx * sinB;
      ax[1] = pAxis[1] * cosB + ry * sinB;
      ax[2] = pAxis[2] * cosB + rz * sinB;

      // Radial bounding envelope: if branch tip would exceed envelope radius, steer gently inward
      const estL = pFullL * 0.58;
      const tipX = root[0] + ax[0] * estL;
      const tipY = root[1] + ax[1] * estL;
      const tipZ = root[2] + ax[2] * estL;
      const tipDist = Math.hypot(tipX, tipY, tipZ);
      if (tipDist > 0.72) {
        const steer = Math.min(0.68, (tipDist - 0.72) * 0.95);
        const invR = 1 / Math.max(1e-4, Math.hypot(root[0], root[1], root[2]));
        ax[0] = ax[0] * (1 - steer) - root[0] * invR * steer;
        ax[1] = ax[1] * (1 - steer) - root[1] * invR * steer * 0.7;
        ax[2] = ax[2] * (1 - steer) - root[2] * invR * steer;
      }

      if (dirProgress > 0.01) {
        if (growSideways) {
          ax[1] *= (1 - 0.55 * dirProgress);
        } else {
          ax[0] *= (1 - 0.55 * dirProgress);
          ax[2] *= (1 - 0.55 * dirProgress);
          ax[1] = Math.sign(ax[1] || 1) * Math.max(Math.abs(ax[1]), 0.65 * dirProgress);
        }
      }
      const al = Math.hypot(ax[0], ax[1], ax[2]) || 1;
      ax[0] /= al; ax[1] /= al; ax[2] /= al;
      tw = pTwist + branchAz + 0.22 * w4;
      buildSideBasis(tw);

      baseL = pFullL * (0.44 + 0.32 * node.uBranchLen) * (1 + 0.32 * dirProgress) * lenMod;
      baseR = Math.max(0.035, P.thickness.value * 0.85) * (0.68 + 0.28 * node.uBranchRad) * radMod;
    }

    // Apply procedural growth scaling to L and r
    const L = Math.max(0.008, baseL * gScale);
    const r = Math.max(0.003, baseR * Math.pow(gScale, 0.78));

    // Save computed world frame on node for child branches
    for (let k = 0; k < 3; k++) {
      node.worldRoot[k] = root[k];
      node.worldAxis[k] = ax[k];
      node.worldSide[k] = sx[k];
    }
    node.worldL = L;
    node.worldFullL = baseL;
    node.worldR = r;
    node.worldTwist = tw;

    const rowIdx = activeCount++;
    const fs = node.faceSeeds;
    let rMax = 0;

    // Unified continuous "opening-out" polygon symmetry (3.0 -> 4.0 -> 5.0 -> 6.0 -> 8.0 faces):
    // For s = n + f (n = floor(s), f in [0, 1]):
    // - Faces k = 0..n-1 sit at angle 2*pi*k / s
    // - The emerging (n+1)-th face (k = n) sits at angle 2*pi*n / s (which equals 2*pi == 0 at f = 0,
    //   coincident with face 0!) and blends its faceJitter from face 0 (at f=0) to face n (at f=1),
    //   hinging smoothly open as f increases from 0 -> 1 with zero geometric discontinuity.
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
        const faceR = r * Math.max(0.35, 1 + dynFaceJitter * fSeed);
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
    for (let k = 0; k < 8; k++) {
      if (k <= nSym) {
        const a = (TAU * k) / sSym;
        const domPhase = 0.5 + 0.5 * Math.sin(phi * 0.95 + s3 + k * Math.PI);
        const domBase = k < nSym ? (k & 1) : ((nSym & 1) * wSym);
        const domWeight = 0.65 * domBase + 0.35 * (k < nSym ? domPhase : domPhase * wSym);
        const tWaveK = k < nSym
          ? Math.cos(phi * 1.32 + s2 + k * GOLDEN)
          : ((1 - wSym) * Math.cos(phi * 1.32 + s2) + wSym * Math.cos(phi * 1.32 + s2 + nSym * GOLDEN));
        const tSeedBase = k < nSym ? fs[8 + k] : (fs[8] + (fs[8 + nSym] - fs[8]) * wSym);
        const tSeed = 0.62 * tSeedBase + 0.38 * tWaveK;
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
    const dynChamfer = Math.max(0, chamfer * (1 + 0.22 * w2));
    const tableFrac = Math.max(0.68, 1.12 - 0.58 * dynChamfer);
    crystalPlane(rowIdx, 17, 0, 1, 0, L * tableFrac);

    const half = 0.5 * L, rc = rMax / Math.cos(Math.PI / Math.max(3, sSym));
    const nodeAlpha = Math.max(0.0, Math.min(1.0, node.alpha));
    slot(rowIdx, 0, root[0] + ax[0] * half, root[1] + ax[1] * half, root[2] + ax[2] * half, Math.hypot(half + r * 0.65, rc) * 1.04);
    slot(rowIdx, 1, ax[0], ax[1], ax[2], L);
    slot(rowIdx, 2, nodeAlpha, node.hueOffset, node.uid, r);
    slot(rowIdx, 3, root[0], root[1], root[2], node.rnd);

    // Smoothly cubic-weighted bounds so arriving/dissolving crystals have zero derivative at birth/death
    const w = nodeAlpha * nodeAlpha * Math.min(1.0, gScale);
    const tipY = root[1] + ax[1] * L;
    const midY = root[1] + ax[1] * half;
    weightedSumY += midY * w;
    weightTotal += w;

    const effPad = r * (0.1 + 0.35 * nodeAlpha);
    minY = Math.min(minY, Math.min(root[1], tipY) - effPad);
    maxY = Math.max(maxY, Math.max(root[1], tipY) + effPad);
    const rTip = Math.hypot(root[0] + ax[0] * L, root[2] + ax[2] * L) + effPad;
    const rRoot = Math.hypot(root[0], root[2]) + effPad;
    maxR2 = Math.max(maxR2, rTip * rTip, rRoot * rRoot);
  }

  cluster.crystals = Math.max(1, activeCount);
  cluster.activeVectors = cluster.crystals * CRYSTAL_PLANES;
  const targetCentreY = weightTotal > 1e-4 ? weightedSumY / weightTotal : (Number.isFinite(minY) ? 0.5 * (minY + maxY) : 0);
  cluster.centreY = Number.isFinite(cluster.centreY)
    ? cluster.centreY + (targetCentreY - cluster.centreY) * 0.04
    : targetCentreY;
  const targetRadiusXZ = Math.max(0.18, Math.sqrt(maxR2));
  const targetHalfY = Math.max(0.18, Number.isFinite(minY) ? 0.5 * (maxY - minY) : 0.45);
  cluster.radiusXZ = Number.isFinite(cluster.radiusXZ)
    ? Math.max(targetRadiusXZ, cluster.radiusXZ + (targetRadiusXZ - cluster.radiusXZ) * 0.08)
    : targetRadiusXZ;
  cluster.halfY = Number.isFinite(cluster.halfY)
    ? Math.max(targetHalfY, cluster.halfY + (targetHalfY - cluster.halfY) * 0.08)
    : targetHalfY;
  cluster.radius = Math.max(0.45, cluster.radiusXZ * 1.18, cluster.halfY * 1.18);
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

