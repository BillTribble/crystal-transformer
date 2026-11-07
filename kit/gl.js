// WebGL2 plumbing shared by the shader sketches: compiling and linking, a fullscreen
// triangle, params as uniforms, and per-frame CPU values (f_*) as uniforms. A param without `uniform: false` (selects excepted) is
// `uniform float u_<key>` in GLSL, or `vec3` for a colour.

/** Vertex shader for a pass that covers the target: one triangle from the vertex id, no buffers. */
export const FULLSCREEN_VERT = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/**
 * Detect mobile / tablet browsers (Android Chrome, iOS Safari, coarse-pointer touch devices)
 * with deterministic `?mobile=1` and `?desktop=1` URL query overrides for testing.
 */
export function detectMobileDevice() {
  if (typeof window === 'undefined') return false;
  try {
    const sp = new URLSearchParams(window.location ? window.location.search : '');
    const m = sp.get('mobile');
    if (m === '1' || m === 'true') return true;
    if (m === '0' || m === 'false' || sp.get('desktop') === '1' || sp.get('desktop') === 'true') return false;
  } catch { /* ignore */ }
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile/i.test(ua)) return true;
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return true;
  const sw = (window.screen && window.screen.width) || window.innerWidth || 1280;
  const sh = (window.screen && window.screen.height) || window.innerHeight || 800;
  return Math.max(sw, sh) <= 960 && Math.min(sw, sh) < 768;
}

/**
 * Compile both stages and link them. Throws with the info log on failure, unless the
 * context is lost (the sketch rebuilds when it's restored).
 * @param {WebGL2RenderingContext} gl
 * @param {string} vert
 * @param {string} frag
 * @param {string} name used in error messages
 */
export function program(gl, vert, frag, name) {
  const vs = /** @type {WebGLShader} */ (gl.createShader(gl.VERTEX_SHADER));
  gl.shaderSource(vs, vert);
  gl.compileShader(vs);
  const fs = /** @type {WebGLShader} */ (gl.createShader(gl.FRAGMENT_SHADER));
  gl.shaderSource(fs, frag);
  gl.compileShader(fs);
  const prog = /** @type {WebGLProgram} */ (gl.createProgram());
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost()) {
    const vsLog = !gl.getShaderParameter(vs, gl.COMPILE_STATUS) ? gl.getShaderInfoLog(vs) : '';
    const fsLog = !gl.getShaderParameter(fs, gl.COMPILE_STATUS) ? gl.getShaderInfoLog(fs) : '';
    const progLog = gl.getProgramInfoLog(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error('Program "' + name + '" failed to link:\n' + (vsLog ? vsLog + '\n' : '') + (fsLog ? fsLog + '\n' : '') + progLog);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

/**
 * The params a shader sees as uniforms.
 * @param {import('./types').Param[]} params
 */
export function shaderParams(params) {
  return params.filter((p) => p.uniform !== false && p.type !== 'select');
}

/**
 * GLSL declarations for those params, one per line.
 * @param {import('./types').Param[]} params from shaderParams()
 */
export function paramDecls(params) {
  return params.map((p) => 'uniform ' + (p.type === 'color' ? 'vec3' : 'float') + ' u_' + p.key + ';\n').join('');
}

/**
 * Upload every param's value (the program must be in use). Allocation-free.
 * @param {WebGL2RenderingContext} gl
 * @param {(WebGLUniformLocation | null)[]} locs one per param, in order
 * @param {import('./types').Param[]} params from shaderParams()
 */
export function uploadParams(gl, locs, params) {
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (p.type === 'color') gl.uniform3fv(locs[i], p.rgb);
    else gl.uniform1f(locs[i], p.type === 'bool' ? (p.value ? 1 : 0) : /** @type {number} */ (p.value));
  }
}

/**
 * @typedef {{ name: string, n: number, array: boolean, arr: Float32Array }} FrameUniform
 * One per-frame value: `n` floats, uploaded as float/vec2/vec3/vec4 (n 1–4), mat3 (n 9), or a
 * float array of length n (`array`).
 */

/**
 * Preallocate the per-frame values a sketch's CPU code writes each frame.
 * @param {Record<string, number | [number]>} spec name → component count (1–4 → float…vec4,
 *   9 → mat3, column-major) or [length] for a float array
 */
export function frameUniforms(spec) {
  /** @type {Record<string, Float32Array>} */
  const vals = {};
  /** @type {FrameUniform[]} */
  const list = [];
  for (const name in spec) {
    const s = spec[name];
    const array = Array.isArray(s);
    const n = array ? s[0] : s;
    vals[name] = new Float32Array(n);
    list.push({ name, n, array, arr: vals[name] });
  }
  return { vals, list };
}

/**
 * The GLSL preamble every pass shares: version and precision, the built-ins (u_res, u_screen,
 * u_time, u_realTime), one uniform per param and one per frame value.
 * @param {import('./types').Param[]} params from shaderParams()
 * @param {FrameUniform[]} list from frameUniforms()
 */
export function shaderHeader(params, list, mobileQuality = false) {
  const type = (/** @type {number} */ n) => (n === 9 ? 'mat3' : ['', 'float', 'vec2', 'vec3', 'vec4'][n]);
  return '#version 300 es\nprecision highp float;\nprecision highp int;\n' +
    '#define MOBILE_QUALITY ' + (mobileQuality ? '1' : '0') + '\n' +
    'uniform vec2 u_res;\nuniform vec2 u_screen;\nuniform float u_time;\nuniform float u_realTime;\n' +
    paramDecls(params) +
    list.map((f) => f.array ? 'uniform float ' + f.name + '[' + f.n + '];' : 'uniform ' + type(f.n) + ' ' + f.name + ';').join('\n') + '\n';
}

/**
 * Upload every frame value (the program must be in use). Allocation-free.
 * @param {WebGL2RenderingContext} gl
 * @param {(WebGLUniformLocation | null)[]} locs one per frame value, in order
 * @param {FrameUniform[]} list from frameUniforms()
 */
export function uploadFrame(gl, locs, list) {
  for (let i = 0; i < list.length; i++) {
    const f = list[i], loc = locs[i];
    if (f.array || f.n === 1) gl.uniform1fv(loc, f.arr);
    else if (f.n === 2) gl.uniform2fv(loc, f.arr);
    else if (f.n === 3) gl.uniform3fv(loc, f.arr);
    else if (f.n === 4) gl.uniform4fv(loc, f.arr);
    else gl.uniformMatrix3fv(loc, false, f.arr);
  }
}

/**
 * Probe whether RGBA16F / HALF_FLOAT colour attachments with linear filtering are complete
 * on this WebGL2 context. Falls back to false (RGBA8 / UNSIGNED_BYTE) if EXT_color_buffer_float
 * and EXT_color_buffer_half_float are missing or if a test framebuffer is incomplete.
 * @param {WebGL2RenderingContext} gl
 */
export function probeFloatTargets(gl) {
  const extFloat = gl.getExtension('EXT_color_buffer_float');
  const extHalf = gl.getExtension('EXT_color_buffer_half_float');
  gl.getExtension('OES_texture_float_linear');
  gl.getExtension('OES_texture_half_float_linear');
  if (!extFloat && !extHalf) return false;

  const tex = gl.createTexture();
  const fbo = gl.createFramebuffer();
  let ok = false;
  try {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 4, 4, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE && gl.getError() === gl.NO_ERROR;
  } catch {
    ok = false;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    if (fbo) gl.deleteFramebuffer(fbo);
    if (tex) gl.deleteTexture(tex);
  }
  return ok;
}

