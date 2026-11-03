// Boots the crystal ray-tracer: params -> saved overrides -> pane -> interactive orbit/zoom -> loop.

import { createModel } from './params.js';
import { buildPanel } from './panel.js';
import { setupChrome } from './chrome.js';
import { createLoop } from './loop.js';
import { hexToOkhsl, okhslToHex } from './okhsl.js';
import { buildCluster, cluster, resetEvolution, stepEvolution } from '../cluster.js';
import { createEnsoSound } from '../sound.js';

const DEFAULT_ZOOM_KEY = 'crystals_default_zoom';
const SOUND_PARAM_KEYS = new Set([
  'sound', 'soundVolume', 'washLevel', 'droneMode', 'washInstrument', 'washFade', 'washGap',
  'soundTrack', 'trackName', 'soundRotate', 'soundReverbMode',
]);
const NON_MORPH_KEYS = new Set([
  'renderScale', 'targetFps', 'maxFps', 'maxDpr', 'timeScale', 'fxaa',
  'defaultZoom', 'distance', 'fov', 'tilt', 'spin', 'spinStart', 'sway', 'swaySpeed',
  'evolve', 'evolveSpeed', 'morphPresets', 'morphTime', 'autoGrow', 'growth', 'growSpeed',
  'graduallyShift', 'complementaryBg', 'hueShift', 'studioEdge',
  'mirrorMode', 'mirrorScale', 'mirrorStrength', 'mirrorDepthFade', 'mirrorLiquid', 'mirrorChecker', 'mirrorComplement', 'occlusionStyle', 'occlusionType', 'occlusion',
  'seed', 'filmFps', 'fireSamples', 'bounces', 'depth', 'rainbow', 'rainbowGlow', 'bloom', 'faceGlare', 'chipSize', 'edgeGranularity',
  'patinaMacro', 'edgeMacro', 'surfaceMacro', 'edgeIrregularity', 'patina', 'hairlineFractures',
  'symmetry', 'fan', 'twinAngle', 'cavityCurve', 'taper', 'tipAngle',
  ...SOUND_PARAM_KEYS,
]);

export function computeDeviceDpr() {
  const isMobile = window.innerWidth <= 820 || (window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
  const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
  return dpr;
}

export function computeComplementaryBg(stoneHex) {
  const hsl = hexToOkhsl(String(stoneHex || '#a77fe0'));
  const stoneH = Number.isFinite(hsl.h) ? hsl.h : 0.75;
  const compH = (stoneH + 0.5) % 1.0;
  const compS = Math.max(0.45, Math.min(0.75, (hsl.s || 0.65) * 0.82));
  const compL = 0.075;
  return {
    stoneH,
    compH,
    compDeg: Number((compH * 360).toFixed(1)),
    stoneDeg: Number((stoneH * 360).toFixed(1)),
    hex: okhslToHex(compH, compS, compL),
  };
}

function parseHexUnit(hex) {
  const n = parseInt(String(hex || '#ffffff').slice(1), 16);
  if (!Number.isFinite(n)) return [1, 1, 1];
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export function run(slug, { manifest, params, presets, randomize, sketch }) {
  const sp = new URLSearchParams(location.search);
  const capture = sp.has('capture');
  const store = createModel(params, capture ? null : 'crystals:' + slug + ':overrides', presets);
  const P = store.byKey;
  if (!P.evolvePresets && P.morphPresets) P.evolvePresets = P.morphPresets;
  if (!P.complementaryBg && P.graduallyShift) P.complementaryBg = P.graduallyShift;
  if (!P.bgColour && P.studioEdge) P.bgColour = P.studioEdge;
  if (!P.trackName && P.soundTrack) P.trackName = P.soundTrack;

  const baseDist = P.distance ? P.distance.default : 2.9;
  function distanceForZoom(zoomVal) {
    const z = Math.max(0.4, Math.min(2.5, Number(zoomVal) || 1));
    const minD = P.distance ? P.distance.min : 1;
    const maxD = P.distance ? P.distance.max : 18;
    return Number(Math.max(minD, Math.min(maxD, baseDist / z)).toFixed(2));
  }

  function syncDefaultZoomToCamera(persist = true) {
    if (!P.defaultZoom || !P.distance) return;
    const z = Number(Math.max(P.defaultZoom.min, Math.min(P.defaultZoom.max, Number(P.defaultZoom.value) || 1)).toFixed(2));
    P.defaultZoom.value = z;
    P.defaultZoom.base = z;
    if (persist && !capture) {
      try { localStorage.setItem(DEFAULT_ZOOM_KEY, String(z)); } catch { /* ignore */ }
    }
    const d = distanceForZoom(z);
    store.setValue(P.distance, d);
    P.distance.base = d;
  }

  if (P.defaultZoom) {
    if (!capture && !sp.has('defaultZoom')) {
      try {
        const rawZoom = localStorage.getItem(DEFAULT_ZOOM_KEY);
        if (rawZoom !== null) {
          const parsed = Number(rawZoom);
          if (Number.isFinite(parsed) && parsed >= 0.4 && parsed <= 2.5) {
            store.setValue(P.defaultZoom, parsed);
            P.defaultZoom.base = P.defaultZoom.value;
          }
        }
      } catch { /* ignore */ }
    } else if (sp.has('defaultZoom') && !capture) {
      try { localStorage.setItem(DEFAULT_ZOOM_KEY, String(P.defaultZoom.value)); } catch { /* ignore */ }
    }
    if (!sp.has('distance') && !sp.has('preset')) {
      const d = distanceForZoom(P.defaultZoom.value);
      store.setValue(P.distance, d);
      P.distance.base = d;
    }
  }

  const canvas = document.getElementById('stage');
  const errorBox = document.getElementById('error');
  const fpsEl = document.getElementById('fps');
  const themeMeta = document.querySelector('meta[name="theme-color"]');

  function fail(err) {
    const msg = err instanceof Error ? err.message : String(err);
    errorBox.textContent = msg;
    errorBox.hidden = false;
    console.error(err);
  }

  const pageParam = manifest.pageColour ? P[manifest.pageColour] : null;
  function syncPageColour() {
    if (P.graduallyShift?.value !== false && P.stoneColour && P.studioEdge) {
      const comp = computeComplementaryBg(P.stoneColour.value);
      if (P.hueShift) {
        P.hueShift.value = 180;
        P.hueShift.base = 180;
      }
      if (P.studioEdge.value !== comp.hex) {
        store.setSmooth(P.studioEdge, comp.hex);
        P.studioEdge.base = comp.hex;
      }
    }
    if (!pageParam) return;
    const c = String(pageParam.value);
    document.documentElement.style.backgroundColor = c;
    document.body.style.backgroundColor = c;
    if (themeMeta) themeMeta.content = c;
  }

  const presetSeconds = manifest.presetSeconds ?? 0.65;
  let saveOnLanding = false;
  let instance = null;
  let growBoost = 0;
  let growBoostTotal = 0;
  let growBoostElapsed = 0;
  const growBoostDuration = 1.65;

  // ---- Secondary Evolve: Gradual Preset-to-Preset -> Random-Dice Morph Controller
  const standardPresetNames = Object.keys(presets || {});
  const morphState = {
    stageIndex: 0,
    diceRolls: 0,
    fromPreset: standardPresetNames[0] || 'Amethyst Cathedral',
    toPreset: standardPresetNames[1] || 'Glacial Ice',
    elapsed: 0,
    holdRemaining: 0,
    progress: 0,
    fromMap: Object.create(null),
    toMap: Object.create(null),
    fromOkhsl: Object.create(null),
    toOkhsl: Object.create(null),
  };

  function prepareMorphStage(fromName, nextStageIdx, holdSeconds = 0) {
    morphState.fromPreset = fromName || store.preset || standardPresetNames[0] || 'Current';
    morphState.stageIndex = nextStageIdx;
    morphState.elapsed = 0;
    morphState.holdRemaining = holdSeconds;
    morphState.progress = 0;
    morphState.fromLive = false;
    morphState.fromMap = Object.create(null);
    morphState.toMap = Object.create(null);
    morphState.fromOkhsl = Object.create(null);
    morphState.toOkhsl = Object.create(null);

    let targetRaw;
    if (nextStageIdx < standardPresetNames.length) {
      const targetName = standardPresetNames[nextStageIdx];
      targetRaw = presets[targetName] || {};
      morphState.toPreset = targetName;
    } else {
      morphState.diceRolls++;
      targetRaw = randomize ? randomize(store) : {};
      morphState.toPreset = 'Random Dice #' + morphState.diceRolls;
    }

    for (const p of store.params) {
      if (NON_MORPH_KEYS.has(p.key) || p.key === 'bloom' || p.noPreset || p.randomize === false) continue;
      const curVal = (fromName && presets && presets[fromName] && presets[fromName][p.key] !== undefined)
        ? presets[fromName][p.key]
        : (p.base !== undefined ? p.base : p.value);
      const toVal = targetRaw[p.key] !== undefined ? targetRaw[p.key] : p.default;
      if (p.type === 'color') {
        morphState.fromMap[p.key] = parseHexUnit(curVal);
        morphState.toMap[p.key] = parseHexUnit(toVal);
        morphState.fromOkhsl[p.key] = hexToOkhsl(String(curVal));
        morphState.toOkhsl[p.key] = hexToOkhsl(String(toVal));
      } else if (p.type === 'select' || p.type === 'bool') {
        morphState.fromMap[p.key] = curVal;
        morphState.toMap[p.key] = toVal;
      } else {
        morphState.fromMap[p.key] = Number(curVal);
        morphState.toMap[p.key] = Number(toVal);
      }
    }
  }

  function resetMorphFromPreset(presetName, holdSeconds = 4.0) {
    const idx = standardPresetNames.indexOf(presetName);
    const nextIdx = idx >= 0 ? idx + 1 : standardPresetNames.length;
    morphState.diceRolls = 0;
    prepareMorphStage(presetName, nextIdx, holdSeconds);
  }

  // Start every morph from the values actually on screen, so its first moving frame can never
  // snap (page-load defaults or saved overrides, a finished preset glide, a randomize landing).
  function captureLiveFrom() {
    for (const p of store.params) {
      if (NON_MORPH_KEYS.has(p.key) || p.key === 'bloom' || p.noPreset || p.randomize === false) continue;
      if (morphState.toMap[p.key] === undefined) continue;
      if (p.type === 'color') {
        morphState.fromMap[p.key] = parseHexUnit(p.value);
        morphState.fromOkhsl[p.key] = hexToOkhsl(String(p.value));
      } else if (p.type === 'select' || p.type === 'bool') {
        morphState.fromMap[p.key] = p.value;
      } else {
        morphState.fromMap[p.key] = Number(p.value);
      }
    }
    morphState.fromLive = true;
  }

  function tickMorphPresets(dt) {
    if (!P.morphPresets?.value || dt <= 0) return false;
    if (P.evolve?.value && cluster.phase === 'growing') return false;
    let rem = Math.min(dt, 600);
    if (morphState.holdRemaining > 0) {
      const holdTake = Math.min(rem, morphState.holdRemaining);
      morphState.holdRemaining -= holdTake;
      rem -= holdTake;
      if (rem <= 1e-6) return false;
    }
    let moved = false;

    while (rem > 1e-6) {
      if (!morphState.fromLive) captureLiveFrom();
      const duration = Math.max(5, Math.min(600, Number(P.morphTime?.value) || 30));
      const needed = Math.max(0, duration - morphState.elapsed);
      const take = Math.min(rem, needed > 1e-6 ? needed : duration);
      morphState.elapsed += take;
      rem -= take;

      const rawT = Math.min(1, morphState.elapsed / duration);
      morphState.progress = Number(rawT.toFixed(4));
      const k = rawT * rawT * rawT * (rawT * (rawT * 6 - 15) + 10);

      for (const p of store.params) {
        if (NON_MORPH_KEYS.has(p.key) || p.key === 'bloom' || p.noPreset || p.randomize === false) continue;
        const fromV = morphState.fromMap[p.key];
        const toV = morphState.toMap[p.key];
        if (fromV === undefined || toV === undefined) continue;

        if (p.type === 'color') {
          const h0 = morphState.fromOkhsl[p.key];
          const h1 = morphState.toOkhsl[p.key];
          if (h0 && h1 && Number.isFinite(h0.h) && Number.isFinite(h1.h) && h0.s > 0.12 && h1.s > 0.12) {
            const dh = ((h1.h - h0.h + 1.5) % 1) - 0.5;
            const h = (h0.h + dh * k + 1) % 1;
            const s = h0.s + (h1.s - h0.s) * k;
            const l = h0.l + (h1.l - h0.l) * k;
            const hex = okhslToHex(h, s, l);
            store.setSmooth(p, hex);
            p.base = p.value;
          } else {
            const r = fromV[0] + (toV[0] - fromV[0]) * k;
            const g = fromV[1] + (toV[1] - fromV[1]) * k;
            const b = fromV[2] + (toV[2] - fromV[2]) * k;
            store.setSmooth(p, null, [r, g, b]);
            p.base = p.value;
          }
        } else if (p.key === 'hueShift') {
          const dh = ((toV - fromV + 540) % 360) - 180;
          const hVal = ((fromV + dh * k) % 360 + 360) % 360;
          store.setSmooth(p, hVal);
          p.base = p.value;
        } else if (p.type === 'select' || p.type === 'bool') {
          const nextChoice = k >= 0.5 ? toV : fromV;
          if (p.value !== nextChoice) {
            store.setSmooth(p, nextChoice);
            p.base = p.value;
          }
        } else {
          const v = fromV + (toV - fromV) * k;
          store.setSmooth(p, v);
          p.base = p.value;
        }
      }
      moved = true;

      if (morphState.elapsed >= duration - 1e-5) {
        store.setPresetName(morphState.toPreset);
        prepareMorphStage(morphState.toPreset, morphState.stageIndex + 1);
      }
    }
    return moved;
  }

  // Initialize initial seed-to-full-size evolution & preset morph pipeline
  resetEvolution(P, { startFromSeed: !capture && !sp.has('growth') && !!P.evolve?.value });
  resetMorphFromPreset(store.preset || standardPresetNames[0] || 'Amethyst Cathedral');

  let occlusionDriftTime = 0;
  let occlusionManualHold = 0;
  let smoothOcclusion = Number(P.occlusion?.value) || 0;
  let targetOcclusion = smoothOcclusion;

  function doRandomize(seconds = 18.0) {
    if (!randomize) return;
    const dur = typeof seconds === 'number' && Number.isFinite(seconds) ? seconds : 18.0;
    saveOnLanding = true;
    const fromSnap = {};
    for (const p of store.params) fromSnap[p.key] = p.value;
    const targetValues = store.randomize(randomize, dur);
    prepareMorphStage('Randomized', standardPresetNames.length, 4.0);
    syncPageColour();
    panel.syncAll();
    panel.refreshStatus();
    store.scheduleSave();
    return { fromSnap, targetValues, duration: dur };
  }

  function growMore() {
    if (!P.growth) return;
    growBoost = Math.min(1.6, growBoost + 0.32);
    growBoostTotal = growBoost;
    growBoostElapsed = 0;
    const curG = Number(P.growth.value);
    const targetG = curG + growBoost;
    if (targetG > P.growth.max - 0.1) {
      store.setRange(P.growth, P.growth.min, Number(Math.min(4.5, targetG + 0.5).toFixed(2)));
    }
    if (P.evolve?.value) {
      cluster.surgeBranch?.(P);
    }
    syncHudButtons();
    panel.syncAll();
    panel.refreshStatus();
    store.scheduleSave();
  }

  const soundBtn = document.getElementById('sound-btn');
  const soundIconOn = document.getElementById('sound-icon-on');
  const soundIconOff = document.getElementById('sound-icon-off');
  const unmuteCenterBtn = document.getElementById('unmute-center-btn');

  let panel = null;

  function syncParamsFromTrack(track, { syncPanel = true } = {}) {
    if (!track) return;
    if (P.soundTrack && P.soundTrack.value !== track.label) {
      store.setValue(P.soundTrack, track.label);
      P.soundTrack.base = track.label;
    }
    if (P.droneMode && track.droneMode) {
      store.setValue(P.droneMode, track.droneMode);
      P.droneMode.base = track.droneMode;
    }
    if (P.washInstrument && track.washInstrument) {
      store.setValue(P.washInstrument, track.washInstrument);
      P.washInstrument.base = track.washInstrument;
    }
    if (P.washFade && Number.isFinite(track.washFade)) {
      store.setValue(P.washFade, track.washFade);
      P.washFade.base = track.washFade;
    }
    if (P.washGap && Number.isFinite(track.washGap)) {
      store.setValue(P.washGap, track.washGap);
      P.washGap.base = track.washGap;
    }
    if (syncPanel && panel) {
      panel.syncAll();
      panel.refreshStatus();
      store.scheduleSave();
    }
  }

  const sound = createEnsoSound({
    onTrackChange(nextLabel, nextTrack) {
      syncParamsFromTrack(nextTrack || sound.currentTrack, { syncPanel: true });
    },
    onStateChange() {
      syncHudButtons();
    },
  });

  // Sync initial Enso track and drone/wash defaults on page load
  if (P.soundTrack && !sp.has('soundTrack')) {
    const startTrack = sound.currentTrack;
    syncParamsFromTrack(startTrack, { syncPanel: false });
    P.soundTrack.default = startTrack.label;
  }

  function syncSoundFromParams(changedKey) {
    if (P.soundVolume) sound.setVolume(P.soundVolume.value);
    if (P.washLevel) sound.setWashLevel(P.washLevel.value);
    if (P.soundRotate) sound.setAutoRotate(P.soundRotate.value);
    if (P.soundReverbMode) sound.setReverbMode(P.soundReverbMode.value);
    if ((changedKey === 'soundTrack' || changedKey === 'trackName') && P.soundTrack) {
      sound.setTrack(P.soundTrack.value);
    } else if (!changedKey || changedKey === 'reset') {
      if (P.soundTrack && P.soundTrack.value !== sound.currentTrack.label) {
        sound.setTrack(P.soundTrack.value);
      }
      if (P.droneMode) sound.setDroneMode(P.droneMode.value);
      if (P.washInstrument) sound.setWashInstrument(P.washInstrument.value);
      if (P.washFade) sound.setWashFade(P.washFade.value);
      if (P.washGap) sound.setWashGap(P.washGap.value);
    } else if (changedKey === 'droneMode' && P.droneMode) {
      if (P.droneMode.value === 'Breathing wash' && P.washGap && Number(P.washGap.value) <= 0) {
        const defGap = Number.isFinite(sound.currentTrack?.washGap) && sound.currentTrack.washGap > 0 ? sound.currentTrack.washGap : 3.0;
        store.setValue(P.washGap, defGap);
      } else if (P.droneMode.value === 'Continuous drone' && P.washGap && Number(P.washGap.value) > 0) {
        store.setValue(P.washGap, 0);
      }
      sound.setDroneMode(P.droneMode.value);
      if (P.washGap) sound.setWashGap(P.washGap.value);
      panel?.syncAll();
    } else if (changedKey === 'washInstrument' && P.washInstrument) {
      sound.setWashInstrument(P.washInstrument.value);
    } else if (changedKey === 'washFade' && P.washFade) {
      sound.setWashFade(P.washFade.value);
    } else if (changedKey === 'washGap' && P.washGap) {
      sound.setWashGap(P.washGap.value);
    }
    if (P.sound) {
      sound.setEnabled(!!P.sound.value);
    }
    syncHudButtons();
  }

  function toggleSound() {
    if (!P.sound) return;
    const next = !P.sound.value;
    store.setValue(P.sound, next);
    P.sound.base = next;
    syncSoundFromParams('sound');
    panel.syncAll();
    panel.refreshStatus();
    store.scheduleSave();
  }

  const chrome = setupChrome(!capture, () => {}, {
    onRandomize: doRandomize,
    onGrow: growMore,
    onSound: toggleSound,
  });

  function applyPreset(name, seconds, save = true) {
    const all = { ...(presets || {}), ...(store.customPresets || {}) };
    if (!all[name]) return 0;
    const evolveOn = !!P.evolve?.value;
    const s = seconds !== undefined ? seconds : (instance?.presetSeconds?.(all[name]) ?? presetSeconds);
    saveOnLanding = save;
    const own = !!instance?.morph;
    store.applyPreset(name, s, !own && s > 0);
    if (own && s > 0) instance.morph(s, name);
    resetMorphFromPreset(name);
    syncPageColour();
    syncHudButtons();
    panel.syncAll();
    panel.refreshStatus();
    if (save) store.scheduleSave();
    return s;
  }

  panel = buildPanel({
    slug,
    store,
    presets,
    onPreset: (name) => applyPreset(name),
    onRandomize: doRandomize,
    onSkipTrack: () => sound.nextTrack(1),
    onEdited(changedKey) {
      if (changedKey === 'defaultZoom') {
        syncDefaultZoomToCamera(true);
        panel.syncAll();
      } else if (changedKey === 'reset') {
        resetEvolution(P, { startFromSeed: !!P.evolve?.value });
        resetMorphFromPreset(store.preset || standardPresetNames[0]);
        syncSoundFromParams('reset');
      } else if (changedKey && SOUND_PARAM_KEYS.has(changedKey)) {
        syncSoundFromParams(changedKey);
      } else if (changedKey === 'occlusionStyle' && P.occlusion) {
        occlusionManualHold = 0;
        const st = String(P.occlusionStyle?.value || 'Auto drift (1/3 time)');
        if (st === 'Always on (Massive)') targetOcclusion = 0.92;
        else if (st === 'Subtle') targetOcclusion = 0.38;
        else if (st === 'Off') targetOcclusion = 0.0;
        panel.syncAll();
      } else if (changedKey === 'occlusion') {
        occlusionManualHold = 30.0;
        smoothOcclusion = Number(P.occlusion?.value) || 0;
        targetOcclusion = smoothOcclusion;
      } else if (changedKey) {
        cluster.applyManualParam?.(changedKey, P);
        if (P[changedKey] && morphState.fromMap[changedKey] !== undefined) {
          if (P[changedKey].type === 'color') {
            const rgb = parseHexUnit(P[changedKey].value);
            morphState.fromMap[changedKey] = rgb;
            morphState.toMap[changedKey] = rgb;
            morphState.fromOkhsl[changedKey] = hexToOkhsl(String(P[changedKey].value));
            morphState.toOkhsl[changedKey] = hexToOkhsl(String(P[changedKey].value));
          } else if (P[changedKey].type === 'select' || P[changedKey].type === 'bool') {
            morphState.fromMap[changedKey] = P[changedKey].value;
            morphState.toMap[changedKey] = P[changedKey].value;
          } else {
            morphState.fromMap[changedKey] = Number(P[changedKey].value);
            morphState.toMap[changedKey] = Number(P[changedKey].value);
          }
        }
      }
      store.scheduleSave();
      syncPageColour();
      syncHudButtons();
      panel.refreshStatus();
    },
  });

  // ---- Top HUD buttons: Preset pill, Sound, Grow, Randomize
  const growBtn = document.getElementById('grow-btn');
  const randHudBtn = document.getElementById('rand-btn');
  const prevPresetBtn = document.getElementById('prev-preset');
  const nextPresetBtn = document.getElementById('next-preset');
  const hudPresetNameBtn = document.getElementById('hud-preset-name');

  function syncHudButtons() {
    if (growBtn) {
      const surging = growBoost > 0.01;
      growBtn.classList.toggle('active', surging);
    }
    if (soundBtn) {
      const isPlaying = !!sound.enabled;
      soundBtn.classList.toggle('active', isPlaying);
      soundBtn.setAttribute('aria-pressed', String(isPlaying));
      if (soundIconOn) soundIconOn.style.display = isPlaying ? 'block' : 'none';
      if (soundIconOff) soundIconOff.style.display = isPlaying ? 'none' : 'block';
    }
    if (unmuteCenterBtn) {
      const hidePrompt = capture || !!sound.enabled;
      unmuteCenterBtn.classList.toggle('is-unmuted', hidePrompt);
      unmuteCenterBtn.setAttribute('aria-hidden', String(hidePrompt));
    }
  }

  soundBtn?.addEventListener('click', toggleSound);
  unmuteCenterBtn?.addEventListener('click', () => {
    if (!sound.enabled) toggleSound();
  });
  growBtn?.addEventListener('click', growMore);
  randHudBtn?.addEventListener('click', () => doRandomize(18.0));
  syncSoundFromParams();

  function cyclePreset(dir) {
    const names = [...Object.keys(presets || {}), ...Object.keys(store.customPresets || {})];
    if (!names.length) return;
    const curIdx = names.indexOf(store.preset);
    const nextIdx = curIdx < 0 ? 0 : (curIdx + dir + names.length) % names.length;
    applyPreset(names[nextIdx]);
  }
  prevPresetBtn?.addEventListener('click', () => cyclePreset(-1));
  nextPresetBtn?.addEventListener('click', () => cyclePreset(1));
  hudPresetNameBtn?.addEventListener('click', () => chrome.togglePane());

  // ---- Interactive Mouse / Touch Orbit & Zoom on #stage
  const activePointers = new Map();
  let dragging = false;
  let lastX = 0, lastY = 0;
  let velYaw = 0, velPitch = 0;
  let pinchDist = 0;
  let lastMoveTime = 0;

  function wrap360(deg) {
    return ((deg % 360) + 360) % 360;
  }
  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (activePointers.size === 1) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      velYaw = 0;
      velPitch = 0;
      lastMoveTime = performance.now();
      canvas.classList.add('dragging');
    } else if (activePointers.size === 2) {
      const pts = [...activePointers.values()];
      pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!activePointers.has(e.pointerId)) return;
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (activePointers.size === 2 && P.distance) {
      const pts = [...activePointers.values()];
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      if (pinchDist > 10 && d > 10) {
        const ratio = pinchDist / d;
        const nextDist = clamp(Number(P.distance.value) * ratio, P.distance.min, P.distance.max);
        store.setValue(P.distance, nextDist);
        panel.syncAll();
      }
      pinchDist = d;
      return;
    }

    if (!dragging || activePointers.size !== 1) return;
    const now = performance.now();
    const dt = Math.max(0.004, Math.min(0.08, (now - lastMoveTime) * 0.001));
    lastMoveTime = now;

    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dx === 0 && dy === 0) return;

    const shortSide = Math.max(240, Math.min(window.innerWidth, window.innerHeight));
    const dYaw = (dx / shortSide) * 210;
    const dPitch = (dy / shortSide) * 165;

    if (P.spinStart) {
      store.setValue(P.spinStart, Number(wrap360(Number(P.spinStart.value) + dYaw).toFixed(3)));
    }
    if (P.tilt) {
      store.setValue(P.tilt, clamp(Number(P.tilt.value) + dPitch, P.tilt.min, P.tilt.max));
    }

    // Exponentially smoothed angular velocity (deg/s) for momentum release
    const instYaw = clamp(dYaw / dt, -520, 520);
    const instPitch = clamp(dPitch / dt, -420, 420);
    velYaw = 0.58 * instYaw + 0.42 * velYaw;
    velPitch = 0.58 * instPitch + 0.42 * velPitch;
    panel.syncAll();
    panel.refreshStatus();
  });

  function endPointer(e) {
    activePointers.delete(e.pointerId);
    if (activePointers.size < 2) pinchDist = 0;
    if (activePointers.size === 0 && dragging) {
      dragging = false;
      canvas.classList.remove('dragging');
      const idleMs = performance.now() - lastMoveTime;
      if (idleMs > 220) {
        const idleDecay = Math.exp(-(idleMs - 220) * 0.008);
        velYaw *= idleDecay;
        velPitch *= idleDecay;
      }
      if (P.spin && Math.abs(velYaw) > 4) {
        const mag = Math.abs(Number(P.spin.value)) || 1.5;
        store.setValue(P.spin, Math.sign(velYaw) * mag);
      }
      const baseSpin = P.spin ? Number(P.spin.value) : 0;
      velYaw = velYaw - baseSpin;
      store.scheduleSave();
    }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  canvas.addEventListener('wheel', (e) => {
    if (!P.distance) return;
    e.preventDefault();
    const factor = Math.exp(e.deltaY * 0.0012);
    const nextDist = clamp(Number(P.distance.value) * factor, P.distance.min, P.distance.max);
    store.setValue(P.distance, nextDist);
    panel.syncAll();
    panel.refreshStatus();
    store.scheduleSave();
  }, { passive: false });

  canvas.addEventListener('dblclick', () => {
    if (P.tilt && P.distance && P.spinStart) {
      store.setValue(P.tilt, P.tilt.base);
      const resetDist = P.defaultZoom ? distanceForZoom(P.defaultZoom.value) : P.distance.base;
      store.setValue(P.distance, resetDist);
      P.distance.base = resetDist;
      store.setValue(P.spinStart, P.spinStart.base);
      velYaw = 0;
      velPitch = 0;
      panel.syncAll();
      panel.refreshStatus();
      store.scheduleSave();
    }
  });

  panel.refreshStatus();
  syncPageColour();
  syncHudButtons();

  const loop = createLoop(canvas, P, () => instance);
  loop.fixed = capture || new URLSearchParams(location.search).has('fixed');
  loop.onStats = (text) => { if (chrome.paneOpen) fpsEl.textContent = text; };

  let paneWait = 0;
  loop.onTick = (dt, _sdt) => {
    let needsUiSync = false;

    // 1. Post-drag momentum that smoothly blends right back into continuous slow auto-rotation
    if (!dragging && dt > 0) {
      const baseSpin = P.spin ? Number(P.spin.value) : 0;
      const hasMomentum = Math.abs(velYaw) > 0.05 || Math.abs(velPitch) > 0.05;
      const yawRate = baseSpin + velYaw;
      if (P.spinStart && Math.abs(yawRate) > 1e-4) {
        const nextSpin = wrap360(Number(P.spinStart.value) + yawRate * dt);
        P.spinStart.value = Number(nextSpin.toFixed(3));
      }
      if (P.tilt && Math.abs(velPitch) > 0.05) {
        store.setValue(P.tilt, clamp(Number(P.tilt.value) + velPitch * dt, P.tilt.min, P.tilt.max));
      }
      if (hasMomentum) {
        velYaw *= Math.exp(-2.2 * dt);
        velPitch *= Math.exp(-2.8 * dt);
        if (Math.abs(velYaw) <= 0.05) velYaw = 0;
        if (Math.abs(velPitch) <= 0.05) velPitch = 0;
        needsUiSync = true;
      }
    }

    // 2. Viewport-Aware Auto-Growth + Click-to-Grow Surge:
    // Auto-grows sideways (when viewport width > height) or upwards (when height > width)
    // until the crystals fill the screen, then smoothly slows down. Clicking the Grow plant
    // button (#grow-btn) adds a `growBoost` surge that makes the crystals grow even more!
    if (P.growth && dt > 0 && (P.autoGrow?.value || growBoost > 0.001)) {
      buildCluster(P, canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight);
      const curG = Number(P.growth.value);
      const maxG = P.growth.max || 3.5;
      let deltaG = 0;

      if (P.autoGrow?.value) {
        const evolveSpd = P.evolveSpeed ? Number(P.evolveSpeed.value) : 1;
        const baseSpeed = (P.growSpeed ? Number(P.growSpeed.value) : 0.22) * (evolveSpd / 0.22);
        const fill = cluster.screenFill || 0.5;
        const fillSlowdown = fill < 0.72
          ? 1.0
          : Math.max(0.02, Math.pow(Math.max(0, 1.0 - (fill - 0.72) / 0.22), 1.85));
        const autoCap = 1.65;
        const capSlowdown = curG >= autoCap
          ? 0.01
          : curG < autoCap - 0.3
            ? 1.0
            : Math.max(0.015, (autoCap - curG) / 0.3);
        deltaG += baseSpeed * Math.min(fillSlowdown, capSlowdown) * dt;
      }

      if (growBoost > 0.001) {
        const t0 = Math.min(1, growBoostElapsed / growBoostDuration);
        growBoostElapsed += dt;
        const t1 = Math.min(1, growBoostElapsed / growBoostDuration);
        const s0 = t0 * t0 * t0 * (t0 * (t0 * 6 - 15) + 10);
        const s1 = t1 * t1 * t1 * (t1 * (t1 * 6 - 15) + 10);
        const stepBoost = Math.min(growBoost, Math.max(0, (s1 - s0) * Math.max(0.42, growBoostTotal)));
        deltaG += stepBoost;
        growBoost = Math.max(0, growBoost - stepBoost);
        if (growBoost <= 0.005 || t1 >= 1) {
          growBoost = 0;
          syncHudButtons();
        }
      }

      if (deltaG > 1e-5) {
        const nextG = Math.min(maxG, curG + deltaG);
        store.setValue(P.growth, Number(nextG.toFixed(4)));
        needsUiSync = true;
      }
    }

    const state = store.tick(dt);

    // 3. Secondary Evolve: gradual parameter morphing across standard presets -> random dice
    if (dt > 0 && state === 'idle' && P.morphPresets?.value) {
      if (tickMorphPresets(dt)) {
        needsUiSync = true;
      }
    }

    // 4. Primary Evolve: Phase 1 (seed -> full size) + Phase 2 (continuous branching & old-segment dissolve)
    if (dt > 0) {
      stepEvolution(P, dt, canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight);
      sound.tick(dt);
      if (P.occlusion && P.occlusionStyle) {
        const occStyle = String(P.occlusionStyle.value || 'Auto alternate (Ovoids & Invisible Crystals)');
        if (occlusionManualHold > 0) {
          occlusionManualHold = Math.max(0, occlusionManualHold - dt);
          targetOcclusion = Number(P.occlusion.value) || 0;
          smoothOcclusion = targetOcclusion;
        } else {
          if (occStyle === 'Always on (Invisible Crystals)') {
            targetOcclusion = 0.92;
            if (P.occlusionType && Math.abs(Number(P.occlusionType.value) - 1.0) > 1e-3) {
              store.setSmooth(P.occlusionType, 1.0);
              P.occlusionType.base = 1.0;
              needsUiSync = true;
            }
          } else if (occStyle === 'Always on (Ovoids)' || occStyle === 'Always on (Massive)') {
            targetOcclusion = 0.92;
            if (P.occlusionType && Math.abs(Number(P.occlusionType.value) - 0.0) > 1e-3) {
              store.setSmooth(P.occlusionType, 0.0);
              P.occlusionType.base = 0.0;
              needsUiSync = true;
            }
          } else if (occStyle === 'Subtle') {
            targetOcclusion = 0.38;
          } else if (occStyle === 'Off') {
            targetOcclusion = 0.0;
          } else if ((occStyle === 'Auto alternate (Ovoids & Invisible Crystals)' || occStyle === 'Auto drift (1/3 time)') && P.evolve?.value) {
            occlusionDriftTime += dt;
            // 90-second cycle (active 30s = 1/3 of the time) alternating between Ovoids (even cycles) and Invisible Crystals (odd cycles):
            // 60s clear at 0.0 (0..60s), 12s slow fade-in (60..72s), 6s hold in the middle at 0.92 (72..78s), 12s slow fade-out (78..90s)
            const cyc = occlusionDriftTime % 90.0;
            const cycleIdx = Math.floor(occlusionDriftTime / 90.0);
            const desiredType = (cycleIdx % 2 === 1) ? 1.0 : 0.0;
            if (P.occlusionType && smoothOcclusion < 0.02 && Math.abs(Number(P.occlusionType.value) - desiredType) > 1e-3) {
              store.setSmooth(P.occlusionType, desiredType);
              P.occlusionType.base = desiredType;
              needsUiSync = true;
            }
            if (cyc >= 60.0 && cyc < 72.0) {
              const u = (cyc - 60.0) / 12.0;
              targetOcclusion = 0.92 * (0.5 - 0.5 * Math.cos(Math.PI * u));
            } else if (cyc >= 72.0 && cyc < 78.0) {
              targetOcclusion = 0.92;
            } else if (cyc >= 78.0) {
              const u = (cyc - 78.0) / 12.0;
              targetOcclusion = 0.92 * (0.5 + 0.5 * Math.cos(Math.PI * u));
            } else {
              targetOcclusion = 0.0;
            }
          }
          // Rate-limit smoothOcclusion to at most 0.92 / 12.0 per second so voids always appear/disappear over a full 12.0 seconds
          const maxStep = (0.92 / 12.0) * dt;
          const diff = targetOcclusion - smoothOcclusion;
          if (Math.abs(diff) <= maxStep) {
            smoothOcclusion = targetOcclusion;
          } else {
            smoothOcclusion += Math.sign(diff) * maxStep;
          }
          const curOcc = Number(P.occlusion.value) || 0;
          if (Math.abs(smoothOcclusion - curOcc) > 5e-4) {
            store.setSmooth(P.occlusion, Number(smoothOcclusion.toFixed(4)));
            P.occlusion.base = P.occlusion.value;
            needsUiSync = true;
          }
        }
      }
    }

    if (state !== 'idle' || needsUiSync) {
      syncPageColour();
      paneWait -= dt;
      if (state === 'done' || paneWait <= 0) {
        paneWait = 0.05;
        panel.syncAll();
        panel.refreshStatus();
      }
      if (state === 'done' && saveOnLanding) store.scheduleSave();
    }
  };

  // Screenshot button
  const shotBtn = document.getElementById('shot');
  shotBtn.addEventListener('click', async () => {
    shotBtn.disabled = true;
    try {
      const png = await loop.snapshot(2880, 1864, 2);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(png);
      a.download = slug + '-' + new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-') + '.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (err) {
      fail(err);
    } finally {
      shotBtn.disabled = false;
    }
  });

  try {
    instance = sketch({ canvas, P, model: store, degrade: (f) => loop.degrade(f) });
  } catch (err) {
    fail(err);
    return;
  }

  window.__glimmer = {
    get frames() { return loop.frames; },
    get velYaw() { return velYaw; },
    get velPitch() { return velPitch; },
    cluster,
    morphState,
    sound,
    chrome,
    loop,
    store,
    P,
    step: (dt) => loop.step(dt),
    resetEvolution: (opts) => resetEvolution(P, opts),
    set: (key, value) => {
      if (P[key]) {
        store.setValue(P[key], value);
        if (SOUND_PARAM_KEYS.has(key)) {
          syncSoundFromParams(key);
        } else {
          cluster.applyManualParam?.(key, P);
        }
        if (morphState.fromMap[key] !== undefined) {
          if (P[key].type === 'color') {
            const rgb = parseHexUnit(P[key].value);
            morphState.fromMap[key] = rgb;
            morphState.toMap[key] = rgb;
            morphState.fromOkhsl[key] = hexToOkhsl(String(P[key].value));
            morphState.toOkhsl[key] = hexToOkhsl(String(P[key].value));
          } else if (P[key].type === 'select' || P[key].type === 'bool') {
            morphState.fromMap[key] = P[key].value;
            morphState.toMap[key] = P[key].value;
          } else {
            morphState.fromMap[key] = Number(P[key].value);
            morphState.toMap[key] = Number(P[key].value);
          }
        }
        if (key === 'defaultZoom') syncDefaultZoomToCamera(true);
        syncPageColour();
        panel.syncAll();
        panel.refreshStatus();
        store.scheduleSave();
      }
    },
    computeComplementaryBg,
    computeDeviceDpr,
    preset: (name, seconds = 0) => applyPreset(name, seconds),
    randomize: (seconds = 10.0) => doRandomize(seconds),
    growMore: () => growMore(),
    get growBoost() { return growBoost; },
    savePreset: (name) => { const s = store.saveCustomPreset(name); panel.syncAll(); panel.refreshStatus(); return s; },
  };
  if (!capture) loop.start();
}
