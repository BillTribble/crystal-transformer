// Seeded random numbers (mulberry32), so a sketch with a Seed param lays out the same way
// every time — and covers come out the same on every capture.

/**
 * @param {number} seed any number; fractional parts count
 * @returns {() => number} a function returning floats in [0, 1)
 */
export function rng(seed) {
  let a = Math.floor(seed * 1000) ^ 0x9e3779b9;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Integer hash → [0, 1). Stateless, for values that must come out the same at any time
 * (a shooting star's slot, say) without stepping a generator.
 * @param {number} n an integer
 */
export function hash1(n) {
  n |= 0;
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n ^= n >>> 16;
  return (n >>> 0) / 4294967296;
}

/** Two integers → [0, 1). */
export function hash2(/** @type {number} */ a, /** @type {number} */ b) { return hash1(Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul(b | 0, 0x165667b1)); }
