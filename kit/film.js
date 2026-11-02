// The film layer shared by the WebGL sketches: grain and dust in GLSL, and their params.
//
// A sketch puts FILM_GLSL in its shader preamble and calls the functions last in its
// composite pass, grain after everything else. Both take `fc` (gl_FragCoord.xy), `px` (canvas
// px per CSS px, so specks keep their size when adaptive quality changes resolution) and
// `frame` (the film frame index, floor(realTime × film fps), so grain keeps running while paused).

export const FILM_GLSL = `
// Integer hash (PCG) → 0..1 per integer cell.
uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
float hash(vec3 p) {
  uvec3 u = uvec3(ivec3(floor(p)));
  return float(pcg(u.x + pcg(u.y + pcg(u.z)))) * (1.0 / 4294967295.0);
}

// Dust: sparse white specks, one candidate per cell per frame.
vec3 filmDust(vec3 col, vec2 fc, float px, float frame, float amount) {
  float cell = 48.0 * px;
  vec2 cid = floor(fc / cell);
  if (hash(vec3(cid, frame + 101.0)) < amount * 0.05) {
    vec2 cpos = (cid + 0.2 + 0.6 * vec2(hash(vec3(cid, frame + 7.0)), hash(vec3(cid, frame + 13.0)))) * cell;
    vec2 dd = fc - cpos;
    dd.x /= mix(1.0, 3.0, hash(vec3(cid, frame + 29.0)));
    float rad = mix(0.7, 2.6, hash(vec3(cid, frame + 19.0))) * px;
    col += (1.0 - smoothstep(rad * 0.4, rad, length(dd))) * mix(0.4, 0.9, hash(vec3(cid, frame + 23.0)));
  }
  return col;
}

// Grain: white, per film frame, two scales, clumped, screened over the image.
vec3 filmGrain(vec3 col, vec2 fc, float px, float frame, float amount, float size, float sharpness, float clump) {
  float gs = max(size * px, 0.5);
  vec2 gp = floor(fc / gs);
  float gn = (hash(vec3(gp, frame)) + hash(vec3(gp, frame + 1000.0)) + hash(vec3(gp, frame + 2000.0))) / 3.0;
  vec2 gp2 = floor(fc / (gs * 2.3));
  float gn2 = (hash(vec3(gp2, frame + 3000.0)) + hash(vec3(gp2, frame + 4000.0))) * 0.5;
  float grain = pow(clamp((mix(gn, gn2, 0.3) - 0.5) * 3.2, 0.0, 1.0), sharpness);
  grain *= mix(1.0, 2.0 * hash(vec3(floor(fc / (gs * 4.0)), frame + 77.0)), clump) * amount;
  return 1.0 - (1.0 - clamp(col, 0.0, 1.0)) * (1.0 - clamp(grain, 0.0, 1.0));
}
`;

/** @typedef {import('./types').ParamDef} ParamDef */

/**
 * The Grain & damage group: grain amount, size, sparsity and clumping, then dust. Override a
 * default by passing it in, e.g. `film({ grainAmount: 0.2, dust: 0 })`. The shader reads them
 * as u_grainAmount, u_grainSize, u_grainSharpness, u_grainClump and u_dust.
 *
 * @param {Partial<Record<string, number>>} [defaults]
 * @returns {import('./types').Group}
 */
export function film(defaults = {}) {
  /** @type {ParamDef[]} */
  const params = [
    { key: 'grainAmount', label: 'Grain amount', value: 0.1, min: 0, max: 0.8, step: 0.005, desc: 'Strength of the light grain (screened over the image).' },
    { key: 'grainSize', label: 'Grain size (px)', value: 0.5, min: 0.5, max: 6, step: 0.05, desc: 'Size of individual grains in canvas pixels.', when: { key: 'grainAmount', not: [0] } },
    { key: 'grainSharpness', label: 'Grain sparsity', value: 1.4, min: 0.3, max: 4, step: 0.01, desc: 'Higher = fewer, brighter specks; lower = even haze.', when: { key: 'grainAmount', not: [0] } },
    { key: 'grainClump', label: 'Grain clumping', value: 0.5, min: 0, max: 1, step: 0.01, desc: 'Uneven grain density, like real emulsion.', when: { key: 'grainAmount', not: [0] } },
    { key: 'dust', label: 'Dust', value: 0.03, min: 0, max: 1, step: 0.01, desc: 'Density of white dust specks per frame.' },
  ];
  for (const p of params) if (defaults[p.key] !== undefined) p.value = /** @type {number} */ (defaults[p.key]);
  return { group: 'Grain & damage', params };
}
