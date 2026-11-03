// Standard params the kit itself reads: playback speed and resolution/quality. (Pause isn't
// a param: it's the play/pause button next to the cog.)
//
// A sketch puts `playback()` first in its params.js. The kit looks these keys up by name,
// so every sketch gets the same controls with the same meaning. Override a default by
// passing it in, e.g. `playback({ timeScale: 0.41 }, { quality: true, extra: [seed] })`.
//
// All of them are `uniform: false`: they're read on the CPU, never by a shader.

/** @typedef {import('./types').ParamDef} ParamDef */
/** @typedef {import('./types').Group} Group */

/** @type {Record<string, ParamDef>} */
const STANDARD = {
  timeScale: { key: 'timeScale', label: 'Speed', value: 1.5, default: 1.5, min: 0, max: 4, step: 0.01, uniform: false, desc: 'Global speed of all scene motion.' },
  fxaa: { key: 'fxaa', label: 'Anti-aliasing', value: 1, default: 1, min: 0, max: 1, step: 0.05, randomize: false, noPreset: true, desc: 'Fast post-process directional edge-aware anti-aliasing (FXAA) across crystal silhouettes, facets, and specular highlights (1.0 = full anti-aliasing, 0 = off).' },
  renderScale: { key: 'renderScale', label: 'Render scale', value: 1, default: 1, min: 0.25, max: 1, step: 0.05, uniform: false, desc: 'Internal resolution ceiling. Locked at 1.0 unless manually lowered.' },
  targetFps: { key: 'targetFps', label: 'Target fps', value: 30, default: 30, min: 24, max: 120, step: 1, uniform: false, desc: 'Target frame rate reference.' },
  maxFps: { key: 'maxFps', label: 'Frame cap', value: 30, default: 30, min: 0, max: 120, step: 1, uniform: false, desc: 'Most frames per second to draw (0 = as many as the screen shows). A slow scene looks the same at 30 and costs half the GPU, battery and heat.' },
  maxDpr: { key: 'maxDpr', label: 'Max pixel ratio', value: 1.5, default: 1.5, min: 1, max: 3, step: 0.25, uniform: false, desc: 'Caps device pixel ratio on retina and mobile screens.' },
  spinStart: { key: 'spinStart', label: 'Orbit angle °', value: 280.757, default: 280.757, min: 0, max: 360, step: 0.5, uniform: false, desc: 'Current horizontal orbit angle around the cluster (drag horizontally on canvas).' },
  bloom: { key: 'bloom', label: 'Bloom', value: 0.2, default: 0.2, min: 0, max: 2, step: 0.01, randomize: false, noPreset: true, desc: 'Surface facet catch-lights, star-burst glints, and optical glow around the brightest crystal highlights.' },
};

/** Keys the kit owns. Sketches must not reuse them for anything else. */
export const STANDARD_KEYS = Object.keys(STANDARD);

/**
 * The Playback group: Speed, any sketch-specific `extra` params, then the
 * resolution controls. `quality: true` adds render scale, the target fps that adaptive
 * quality holds (adaptive quality is always on for those sketches) and a frame cap, which
 * only pay off for GPU-heavy sketches; every sketch gets the pixel-ratio cap.
 *
 * @param {Partial<Record<string, number | boolean>>} [defaults] default values for standard keys
 * @param {{ quality?: boolean, extra?: ParamDef[] }} [options]
 * @returns {Group}
 */
export function playback(defaults = {}, { quality = false, extra = [] } = {}) {
  const pick = (/** @type {string} */ key) => {
    const p = { ...STANDARD[key] };
    if (defaults[key] !== undefined) p.value = /** @type {number | boolean} */ (defaults[key]);
    return p;
  };
  return {
    group: 'Playback',
    params: [
      pick('timeScale'),
      ...extra,
      pick('fxaa'),
      ...(quality ? [pick('renderScale'), pick('targetFps'), pick('maxFps')] : []),
      pick('maxDpr'),
    ],
  };
}
