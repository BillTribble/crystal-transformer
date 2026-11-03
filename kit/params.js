// The param model: live values for params.js, overrides saved per browser,
// curated + user-saved custom presets in localStorage, URL state sharing, and smooth morphing.

function hexToRgb(h, out, scale = 1) {
  const n = parseInt(h.slice(1), 16);
  out[0] = ((n >> 16) & 255) * scale; out[1] = ((n >> 8) & 255) * scale; out[2] = (n & 255) * scale;
}
function rgbToHex(r, g, b) {
  const c = (v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}
function easeInOut(t) { return 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, t))); }

function decimalsOf(step) {
  const s = String(step ?? 1);
  return s.includes('e-') ? Number(s.split('e-')[1]) : (s.split('.')[1] || '').length;
}
function roundTo(v, d) { return Number(Number(v).toFixed(d)); }

function writeDerived(p) {
  if (p.type === 'color') hexToRgb(String(p.value), p.rgb, 1 / 255);
}

function coerce(p, v) {
  if (p.type === 'bool') return v === 'false' ? false : v === '0' ? false : !!v;
  if (p.type === 'color') return typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v) ? v.toLowerCase() : p.value;
  if (p.type === 'select') return typeof v === 'string' && p.options.includes(v) ? v : p.value;
  const n = Number(v);
  return Number.isFinite(n) ? roundTo(n, p.decimals) : p.value;
}

function signature(params) {
  const s = JSON.stringify(params.map((p) => [p.key, p.default]));
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

const CUSTOM_PRESETS_KEY = 'crystals:custom-presets';
const STICKY_USER_KEYS = new Set([
  'timeScale', 'fxaa', 'defaultZoom', 'evolve', 'evolveSpeed', 'morphPresets', 'morphTime',
  'autoGrow', 'growSpeed', 'growth', 'spinStart', 'spin', 'tilt', 'fov',
  'rainbow', 'rainbowGlow', 'bloom', 'faceGlare', 'chipSize', 'edgeGranularity',
  'patinaMacro', 'edgeMacro', 'surfaceMacro', 'edgeIrregularity', 'patina', 'hairlineFractures',
  'mirrorMode', 'mirrorScale', 'mirrorStrength', 'mirrorDepthFade', 'mirrorLiquid', 'mirrorChecker', 'mirrorComplement',
  'occlusionStyle', 'occlusionType', 'occlusion',
]);
const DISCRETE_INT_KEYS = new Set(['seed', 'fireSamples', 'bounces', 'depth', 'filmFps', 'targetFps', 'maxFps']);

export function createModel(schema, storageKey, presets = {}) {
  const params = [];
  const byKey = Object.create(null);
  const groups = [];
  let version = 0;

  for (const g of schema) {
    const live = [];
    for (const def of g.params) {
      if (byKey[def.key]) throw new Error('Duplicate param key: ' + def.key);
      const type = def.type || 'range';
      const p = {
        ...def,
        type,
        group: g.group,
        default: def.value,
        base: def.value,
        decimals: type === 'range' ? Math.max(decimalsOf(def.step), decimalsOf(def.value)) : 0,
        rgb: new Float32Array(3),
        parent: null,
        depth: 0,
        codeMin: def.min,
        codeMax: def.max,
      };
      if (def.when) {
        const parent = live.find((q) => q.key === def.when.key);
        if (!parent) throw new Error(`${def.key}: "when" names ${def.when.key}, which isn't earlier in its group`);
        p.parent = parent;
        p.depth = parent.depth + 1;
      }
      writeDerived(p);
      params.push(p);
      live.push(p);
      byKey[p.key] = p;
    }
    groups.push({ group: g.group, params: live });
  }

  const sig = signature(params);
  let saveTimer = 0;
  let preset = '';
  let held = null;
  const edited = (p) => (held && held.has(p) ? held.get(p) : p.value);

  let glides = [];
  let glideT = 0, glideLen = 0;

  // Load user-saved custom presets from localStorage
  let customPresets = {};
  if (storageKey) {
    try {
      const raw = JSON.parse(localStorage.getItem(CUSTOM_PRESETS_KEY) || '{}');
      if (raw && typeof raw === 'object') customPresets = raw;
    } catch { /* ignore */ }
  }

  function saveCustomPresetsToDisk() {
    if (!storageKey) return;
    try {
      localStorage.setItem(CUSTOM_PRESETS_KEY, JSON.stringify(customPresets));
    } catch { /* ignore */ }
  }

  function assign(p, v) {
    p.value = coerce(p, v);
    writeDerived(p);
    version++;
  }

  function assignSmooth(p, v, rgbUnit) {
    if (p.type === 'color' && rgbUnit) {
      p.value = rgbToHex(rgbUnit[0] * 255, rgbUnit[1] * 255, rgbUnit[2] * 255);
      p.rgb[0] = rgbUnit[0];
      p.rgb[1] = rgbUnit[1];
      p.rgb[2] = rgbUnit[2];
    } else if (p.type === 'range' && !DISCRETE_INT_KEYS.has(p.key)) {
      const n = Number(v);
      p.value = Number.isFinite(n) ? Number(n.toFixed(Math.max(4, p.decimals))) : p.value;
    } else {
      p.value = coerce(p, v);
      writeDerived(p);
    }
    version++;
  }

  const store = {
    params,
    byKey,
    groups,
    curatedPresets: presets,
    get customPresets() { return customPresets; },
    get version() { return version; },
    bumpVersion() { version++; },
    setSmooth(p, v, rgbUnit) { assignSmooth(p, v, rgbUnit); },
    setValue(p, v) {
      store.setBypass(false);
      if (glides.length) glides = glides.filter((g) => g.p !== p);
      assign(p, v);
    },
    get preset() { return preset; },
    setPresetName(name) { preset = name; },
    isChanged: (p) => edited(p) !== p.base,
    edited,
    get bypassed() { return !!held; },
    setBypass(on) {
      if (on === !!held) return;
      if (on) {
        glides = [];
        held = new Map();
        for (const p of params) if (p.value !== p.base) { held.set(p, p.value); assign(p, p.base); }
      } else {
        const back = held;
        held = null;
        for (const [p, v] of back) assign(p, v);
      }
    },
    setRange(p, min, max) {
      p.min = min;
      p.max = max;
      version++;
    },
    isShown(p) {
      for (let q = p; q && q.when; q = q.parent) {
        const v = edited(q.parent), w = q.when;
        if (w.is && !w.is.includes(v)) return false;
        if (w.not && w.not.includes(v)) return false;
      }
      return true;
    },
    changedCount() {
      let n = 0;
      for (const p of params) if (edited(p) !== p.base && store.isShown(p)) n++;
      return n;
    },
    resetAll() {
      glides = [];
      preset = '';
      held = null;
      for (const p of params) {
        p.base = p.default;
        p.min = p.codeMin;
        p.max = p.codeMax;
        assign(p, p.default);
      }
    },
    scheduleSave() {
      if (!storageKey) return;
      clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => {
        try {
          const values = {};
          const bases = {};
          const ranges = {};
          for (const p of params) {
            if (edited(p) !== p.default) values[p.key] = edited(p);
            if (p.base !== p.default) bases[p.key] = p.base;
            if (p.min !== p.codeMin || p.max !== p.codeMax) ranges[p.key] = [p.min, p.max];
          }
          localStorage.setItem(storageKey, JSON.stringify({ sig, values, bases, ranges, preset }));
        } catch { /* storage unavailable */ }
      }, 250);
    },
    applyValues(values, seconds = 0.6, presetName = '', glideNumbers = true) {
      if (!values) return;
      store.setBypass(false);
      preset = presetName;
      const resolveTarget = (p) => {
        if (p.key === 'bloom' || p.noPreset || p.randomize === false) return p.value;
        if (values[p.key] !== undefined) return values[p.key];
        if (STICKY_USER_KEYS.has(p.key)) return p.value;
        if (p.key === 'distance' && byKey.defaultZoom) {
          const z = Math.max(0.4, Math.min(2.5, Number(byKey.defaultZoom.value) || 1));
          return Math.max(p.min, Math.min(p.max, p.default / z));
        }
        return p.default;
      };
      for (const p of params) {
        if (p.key === 'renderScale' || p.key === 'targetFps' || p.key === 'maxFps' || p.key === 'maxDpr' || p.key === 'bloom' || p.noPreset) continue;
        p.base = coerce(p, resolveTarget(p));
      }
      glides = [];
      for (const p of params) {
        if (p.key === 'renderScale' || p.key === 'targetFps' || p.key === 'maxFps' || p.key === 'maxDpr' || p.key === 'bloom' || p.noPreset) continue;
        if (STICKY_USER_KEYS.has(p.key) && values[p.key] === undefined) continue;
        if (p.key === 'seed' && seconds > 0 && glideNumbers) continue;
        const to = coerce(p, resolveTarget(p));
        if (seconds <= 0 || p.type === 'select' || p.type === 'bool' || p.key === 'seed' || (p.type !== 'color' && !glideNumbers)) {
          assign(p, to);
          continue;
        }
        if (p.type === 'color') {
          const a = [0, 0, 0], b = [0, 0, 0];
          hexToRgb(String(p.value), a); hexToRgb(String(to), b);
          glides.push({ p, from: 0, to: 1, rgbFrom: a, rgbTo: b });
        } else if (Number(p.value) !== Number(to)) {
          glides.push({ p, from: Number(p.value), to: Number(to) });
        }
      }
      glideT = 0;
      glideLen = Math.max(seconds, 1e-3);
    },
    applyPreset(name, seconds = 0.65, glideNumbers = true) {
      const resolvedName = (name === 'Glacial Ice Dendrite' && presets?.['Glacial Ice']) ? 'Glacial Ice' : name;
      const values = customPresets[resolvedName] || presets?.[resolvedName];
      if (!values) return;
      store.applyValues(values, seconds, resolvedName, glideNumbers);
    },
    randomize(customRandomizer, seconds = 10.0) {
      const next = typeof customRandomizer === 'function' ? { ...customRandomizer(store) } : {};
      delete next.bloom;
      delete next.faceGlare;
      if (seconds > 0) delete next.seed;
      for (const p of params) {
        if (p.key === 'bloom' || p.key === 'faceGlare' || p.randomize === false || p.noPreset) {
          delete next[p.key];
        }
      }
      store.applyValues(next, seconds, 'Randomized', seconds > 0);
      return next;
    },
    get gliding() { return glides.length > 0; },
    get glideElapsed() { return glideT; },
    get glideDuration() { return glideLen; },
    get glideProgress() { return glideLen > 0 ? Math.min(1, glideT / glideLen) : 0; },
    saveCustomPreset(name) {
      const clean = String(name || '').trim();
      if (!clean) return null;
      const snap = {};
      for (const p of params) {
        if (p.key === 'renderScale' || p.key === 'targetFps' || p.key === 'maxFps' || p.key === 'maxDpr' || p.key === 'defaultZoom' || p.key === 'bloom' || p.noPreset) continue;
        snap[p.key] = edited(p);
      }
      customPresets[clean] = snap;
      preset = clean;
      for (const p of params) p.base = edited(p);
      saveCustomPresetsToDisk();
      store.scheduleSave();
      return clean;
    },
    deleteCustomPreset(name) {
      if (!customPresets[name]) return false;
      delete customPresets[name];
      if (preset === name) preset = '';
      saveCustomPresetsToDisk();
      store.scheduleSave();
      return true;
    },
    renameCustomPreset(oldName, newName) {
      const clean = String(newName || '').trim();
      if (!clean || !customPresets[oldName]) return null;
      if (clean !== oldName) {
        customPresets[clean] = customPresets[oldName];
        delete customPresets[oldName];
      }
      if (preset === oldName) preset = clean;
      saveCustomPresetsToDisk();
      store.scheduleSave();
      return clean;
    },
    exportSnapshot() {
      const snap = {};
      for (const p of params) {
        if (edited(p) !== p.default) snap[p.key] = edited(p);
      }
      return snap;
    },
    encodeShareUrl() {
      const url = new URL(window.location.href);
      // Clear previous param keys
      for (const p of params) url.searchParams.delete(p.key);
      url.searchParams.delete('preset');
      url.searchParams.delete('s');
      const diff = {};
      for (const p of params) {
        if (edited(p) !== p.default) diff[p.key] = edited(p);
      }
      if (preset) url.searchParams.set('preset', preset);
      if (Object.keys(diff).length > 0) {
        url.searchParams.set('s', btoa(unescape(encodeURIComponent(JSON.stringify(diff)))));
      }
      window.history.replaceState(null, '', url.toString());
      return url.toString();
    },
    loadFromUrl() {
      const sp = new URLSearchParams(window.location.search);
      let applied = false;
      const presetParam = sp.get('preset');
      if (presetParam && (customPresets[presetParam] || presets?.[presetParam])) {
        store.applyPreset(presetParam, 0);
        applied = true;
      }
      const sParam = sp.get('s');
      if (sParam) {
        try {
          const decoded = JSON.parse(decodeURIComponent(escape(atob(sParam))));
          if (decoded && typeof decoded === 'object') {
            for (const k in decoded) if (byKey[k]) {
              assign(byKey[k], decoded[k]);
              byKey[k].base = byKey[k].value;
            }
            applied = true;
          }
        } catch { /* ignore malformed s param */ }
      }
      for (const p of params) {
        if (sp.has(p.key)) {
          assign(p, sp.get(p.key));
          p.base = p.value;
          applied = true;
        }
      }
      return applied;
    },
    tick(dt) {
      if (!glides.length) return 'idle';
      glideT = Math.min(glideLen, glideT + dt);
      const done = glideT >= glideLen;
      const k = easeInOut(glideT / glideLen);
      for (const g of glides) {
        if (g.rgbFrom && g.rgbTo) {
          const a = g.rgbFrom, b = g.rgbTo;
          const r = a[0] + (b[0] - a[0]) * k, gr = a[1] + (b[1] - a[1]) * k, bl = a[2] + (b[2] - a[2]) * k;
          if (done) assign(g.p, rgbToHex(r, gr, bl));
          else assignSmooth(g.p, null, [r / 255, gr / 255, bl / 255]);
        } else if (done) {
          assign(g.p, g.to);
        } else {
          assignSmooth(g.p, g.from + (g.to - g.from) * k);
        }
      }
      if (!done) return 'moving';
      glides = [];
      return 'done';
    },
    clearSaved() {
      clearTimeout(saveTimer);
      if (!storageKey) return;
      try { localStorage.removeItem(storageKey); } catch { /* ignore */ }
    },
  };

  if (storageKey) {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (saved && typeof saved === 'object' && saved.values) {
        for (const k in saved.values) if (byKey[k]) assign(byKey[k], saved.values[k]);
        for (const k in saved.bases || {}) if (byKey[k]) byKey[k].base = coerce(byKey[k], saved.bases[k]);
        for (const k in saved.ranges || {}) {
          const r = saved.ranges[k];
          if (byKey[k] && Array.isArray(r) && Number.isFinite(r[0]) && Number.isFinite(r[1]) && r[0] < r[1]) [byKey[k].min, byKey[k].max] = r;
        }
        if (typeof saved.preset === 'string' && (customPresets[saved.preset] || presets?.[saved.preset])) preset = saved.preset;
      }
      if (saved && saved.values) {
        let migrated = saved.sig !== sig;
        if ((saved.values.rainbow === 1 || Number(saved.values.rainbow) === 1) && byKey.rainbow) {
          assign(byKey.rainbow, 0.1);
          byKey.rainbow.base = 0.1;
          delete saved.values.rainbow;
          if (saved.bases) delete saved.bases.rainbow;
          migrated = true;
        }
        if ((saved.values.rainbowGlow === 0.5 || Number(saved.values.rainbowGlow) === 0.5 || saved.values.rainbowGlow === 1 || Number(saved.values.rainbowGlow) === 1) && byKey.rainbowGlow) {
          assign(byKey.rainbowGlow, 1.3);
          byKey.rainbowGlow.base = 1.3;
          delete saved.values.rainbowGlow;
          if (saved.bases) delete saved.bases.rainbowGlow;
          migrated = true;
        }
        if ((saved.values.edgeMacro === 1 || Number(saved.values.edgeMacro) === 1) && byKey.edgeMacro) {
          assign(byKey.edgeMacro, 0.38);
          byKey.edgeMacro.base = 0.38;
          delete saved.values.edgeMacro;
          if (saved.bases) delete saved.bases.edgeMacro;
          migrated = true;
        }
        if ((saved.values.surfaceMacro === 1 || Number(saved.values.surfaceMacro) === 1) && byKey.surfaceMacro) {
          assign(byKey.surfaceMacro, 1.05);
          byKey.surfaceMacro.base = 1.05;
          delete saved.values.surfaceMacro;
          if (saved.bases) delete saved.bases.surfaceMacro;
          migrated = true;
        }
        if ((saved.values.chipSize === 0.55 || saved.values.chipSize === 1.5 || Number(saved.values.chipSize) === 0.55 || Number(saved.values.chipSize) === 1.5) && byKey.chipSize) {
          assign(byKey.chipSize, 3);
          byKey.chipSize.base = 3;
          delete saved.values.chipSize;
          if (saved.bases) delete saved.bases.chipSize;
          migrated = true;
        }
        if ((saved.values.edgeGranularity === 1.65 || saved.values.edgeGranularity === 1.4 || Number(saved.values.edgeGranularity) === 1.65 || Number(saved.values.edgeGranularity) === 1.4) && byKey.edgeGranularity) {
          assign(byKey.edgeGranularity, 3);
          byKey.edgeGranularity.base = 3;
          delete saved.values.edgeGranularity;
          if (saved.bases) delete saved.bases.edgeGranularity;
          migrated = true;
        }
        if ((saved.values.maxDpr === 1 || Number(saved.values.maxDpr) === 1) && byKey.maxDpr) {
          assign(byKey.maxDpr, 1.5);
          byKey.maxDpr.base = 1.5;
          delete saved.values.maxDpr;
          if (saved.bases) delete saved.bases.maxDpr;
          migrated = true;
        }
        if ((saved.values.targetFps === 60 || Number(saved.values.targetFps) === 60) && byKey.targetFps) {
          assign(byKey.targetFps, 30);
          byKey.targetFps.base = 30;
          delete saved.values.targetFps;
          if (saved.bases) delete saved.bases.targetFps;
          migrated = true;
        }
        if ((saved.values.iridescence === 0.38 || saved.values.iridescence === 0.22 || saved.values.iridescence === 0.29 || saved.values.iridescence === 0.46 || saved.values.iridescence === 0.14 || saved.values.iridescence === 0.42) && byKey.iridescence) {
          const pDef = presets?.[preset]?.iridescence ?? 0.35;
          assign(byKey.iridescence, pDef);
          byKey.iridescence.base = pDef;
          migrated = true;
        }
        if (preset && presets?.[preset]) {
          for (const rk of ['fire', 'fractures', 'fractureGlow', 'phantom', 'phantomRate', 'phantomLayers', 'frostShare', 'frost']) {
            if (presets[preset][rk] !== undefined && byKey[rk]) {
              assign(byKey[rk], presets[preset][rk]);
              byKey[rk].base = presets[preset][rk];
            }
          }
        }
        if (migrated) {
          saved.sig = sig;
          try { localStorage.setItem(storageKey, JSON.stringify(saved)); } catch { /* ignore */ }
        }
      }
      if (byKey.chipSize && byKey.chipSize.max < 3.0) byKey.chipSize.max = 3.0;
      if (byKey.edgeGranularity && byKey.edgeGranularity.max < 3.0) byKey.edgeGranularity.max = 3.0;
      if (byKey.iridescence && byKey.iridescence.max < 3.0) byKey.iridescence.max = 3.0;
    } catch { /* storage unavailable — run on defaults */ }
  }

  // URL parameters override localStorage if present
  store.loadFromUrl();

  return store;
}
