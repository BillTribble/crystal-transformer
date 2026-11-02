// OKHSL <-> sRGB hex, for the colour picker: Björn Ottosson's OKLab-based HSL (MIT), whose
// square of saturation x lightness is all inside sRGB at every hue, so the picker has no
// unreachable corners. Hue, saturation and lightness are 0-1. Also plain OKLCH -> sRGB.

const K1 = 0.206, K2 = 0.03, K3 = (1 + K1) / (1 + K2);
const toe = (x) => 0.5 * (K3 * x - K1 + Math.sqrt((K3 * x - K1) * (K3 * x - K1) + 4 * K2 * K3 * x));
const toeInv = (x) => (x * x + K1 * x) / (K3 * (x + K2));

const lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const enc = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

function oklabToLinear(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}

function maxSaturation(a, b) {
  let k0, k1, k2, k3, k4, wl, wm, ws;
  if (-1.88170328 * a - 0.80936493 * b > 1) {
    k0 = 1.19086277; k1 = 1.76576728; k2 = 0.59662641; k3 = 0.75515197; k4 = 0.56771245;
    wl = 4.0767416621; wm = -3.3077115913; ws = 0.2309699292;
  } else if (1.81444104 * a - 1.19445276 * b > 1) {
    k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.12541070; k4 = 0.14503204;
    wl = -1.2684380046; wm = 2.6097574011; ws = -0.3413193965;
  } else {
    k0 = 1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = 0.00692167;
    wl = -0.0041960863; wm = -0.7034186147; ws = 1.7076147010;
  }
  let S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
  const kl = 0.3963377774 * a + 0.2158037573 * b;
  const km = -0.1055613458 * a - 0.0638541728 * b;
  const ks = -0.0894841775 * a - 1.2914855480 * b;
  const l_ = 1 + S * kl, m_ = 1 + S * km, s_ = 1 + S * ks;
  const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  const f = wl * l + wm * m + ws * s;
  const f1 = wl * 3 * kl * l_ * l_ + wm * 3 * km * m_ * m_ + ws * 3 * ks * s_ * s_;
  const f2 = wl * 6 * kl * kl * l_ + wm * 6 * km * km * m_ + ws * 6 * ks * ks * s_;
  S -= (f * f1) / (f1 * f1 - 0.5 * f * f2);
  return S;
}

function cusp(a, b) {
  const S = maxSaturation(a, b);
  const rgb = oklabToLinear(1, S * a, S * b);
  const L = Math.cbrt(1 / Math.max(rgb[0], rgb[1], rgb[2]));
  return [L, L * S];
}

function gamutIntersection(a, b, L1, C1, L0, [cl, cc]) {
  if ((L1 - L0) * cc - (cl - L0) * C1 <= 0) return (cc * L0) / (C1 * cl + cc * (L0 - L1));
  let t = (cc * (L0 - 1)) / (C1 * (cl - 1) + cc * (L0 - L1));
  const dL = L1 - L0, dC = C1;
  const kl = 0.3963377774 * a + 0.2158037573 * b;
  const km = -0.1055613458 * a - 0.0638541728 * b;
  const ks = -0.0894841775 * a - 1.2914855480 * b;
  const ldt = dL + dC * kl, mdt = dL + dC * km, sdt = dL + dC * ks;
  const L = L0 * (1 - t) + t * L1, C = t * C1;
  const l_ = L + C * kl, m_ = L + C * km, s_ = L + C * ks;
  const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  const l1 = 3 * ldt * l_ * l_, m1 = 3 * mdt * m_ * m_, s1 = 3 * sdt * s_ * s_;
  const l2 = 6 * ldt * ldt * l_, m2 = 6 * mdt * mdt * m_, s2 = 6 * sdt * sdt * s_;
  const step = (wl, wm, ws) => {
    const f = wl * l + wm * m + ws * s - 1, f1 = wl * l1 + wm * m1 + ws * s1, f2 = wl * l2 + wm * m2 + ws * s2;
    const u = f1 / (f1 * f1 - 0.5 * f * f2);
    return u >= 0 ? -f * u : 1e6;
  };
  t += Math.min(
    step(4.0767416621, -3.3077115913, 0.2309699292),
    step(-1.2684380046, 2.6097574011, -0.3413193965),
    step(-0.0041960863, -0.7034186147, 1.7076147010),
  );
  return t;
}

function chromas(L, a, b) {
  const cu = cusp(a, b);
  const Cmax = gamutIntersection(a, b, L, 1, L, cu);
  const Smax = cu[1] / cu[0], Tmax = cu[1] / (1 - cu[0]);
  const Smid = 0.11516993 + 1 / (7.44778970 + 4.15901240 * b + a * (-2.19557347 + 1.75198401 * b + a * (-2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
  const Tmid = 0.11239642 + 1 / (1.61320320 - 0.68124379 * b + a * (0.40370612 + 0.90148123 * b + a * (-0.27087943 + 0.61223990 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
  const k = Cmax / Math.min(L * Smax, (1 - L) * Tmax);
  const ca = L * Smid, cb = (1 - L) * Tmid;
  const Cmid = 0.9 * k * Math.sqrt(Math.sqrt(1 / (1 / (ca * ca * ca * ca) + 1 / (cb * cb * cb * cb))));
  const c0a = L * 0.4, c0b = (1 - L) * 0.8;
  const C0 = Math.sqrt(1 / (1 / (c0a * c0a) + 1 / (c0b * c0b)));
  return [C0, Cmid, Cmax];
}

export function oklchToRgb(l, c, h, out) {
  const a = c * Math.cos(h * Math.PI / 180), b = c * Math.sin(h * Math.PI / 180);
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.2914855480 * b;
  const L = l_ * l_ * l_, M = m_ * m_ * m_, S = s_ * s_ * s_;
  out[0] = enc(Math.min(1, Math.max(0, 4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S)));
  out[1] = enc(Math.min(1, Math.max(0, -1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S)));
  out[2] = enc(Math.min(1, Math.max(0, -0.0041960863 * L - 0.7034186147 * M + 1.7076147010 * S)));
}

const hex2 = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0');

export function okhslRow(h, l) {
  if (l >= 1) return () => '#ffffff';
  if (l <= 0) return () => '#000000';
  const a = Math.cos(2 * Math.PI * h), b = Math.sin(2 * Math.PI * h);
  const L = toeInv(l);
  const [C0, Cmid, Cmax] = chromas(L, a, b);
  return (s) => {
    let C;
    if (s < 0.8) {
      const t = 1.25 * s, k1 = 0.8 * C0, k2 = 1 - k1 / Cmid;
      C = (t * k1) / (1 - k2 * t);
    } else {
      const t = 5 * (s - 0.8), k1 = (0.2 * Cmid * Cmid * 1.25 * 1.25) / C0, k2 = 1 - k1 / (Cmax - Cmid);
      C = Cmid + (t * k1) / (1 - k2 * t);
    }
    const rgb = oklabToLinear(L, C * a, C * b);
    return '#' + hex2(enc(rgb[0])) + hex2(enc(rgb[1])) + hex2(enc(rgb[2]));
  };
}

export function okhslToHex(h, s, l) {
  return okhslRow(h, l)(s);
}

export function hexToOkhsl(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = lin(((n >> 16) & 255) / 255), g = lin(((n >> 8) & 255) / 255), bl = lin((n & 255) / 255);
  const l_ = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * bl);
  const m_ = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * bl);
  const s_ = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * bl);
  const L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  const A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
  const B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
  const C = Math.hypot(A, B), l = toe(L);
  if (C < 1e-5 || L <= 0 || L >= 1) return { h: NaN, s: 0, l: Math.min(1, Math.max(0, l)) };
  const a = A / C, b = B / C;
  const [C0, Cmid, Cmax] = chromas(L, a, b);
  let s;
  if (C < Cmid) {
    const k1 = 0.8 * C0, k2 = 1 - k1 / Cmid;
    s = 0.8 * (C / (k1 + k2 * C));
  } else {
    const k1 = (0.2 * Cmid * Cmid * 1.25 * 1.25) / C0, k2 = 1 - k1 / (Cmax - Cmid);
    s = 0.8 + 0.2 * ((C - Cmid) / (k1 + k2 * (C - Cmid)));
  }
  return { h: ((Math.atan2(B, A) / (2 * Math.PI)) + 1) % 1, s: Math.max(0, s), l };
}
