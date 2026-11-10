struct U {
  gain: vec3<f32>, n: u32,
  a_sh: f32, a_hi: f32, bp: f32, wp: f32,
  k: f32, use_tone: u32, use_curves: u32, lut_n: u32,
  pad0: u32, pad1: u32, pad2: u32, pad3: u32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<f32>;
@group(0) @binding(5) var<storage, read> lv: array<f32>;

const PR: u32 = 132u;
const PI: f32 = 3.14159265358979;
var<private> BANDS: array<f32, 8> = array<f32, 8>(29.0, 55.0, 100.0, 142.0, 195.0, 264.0, 295.0, 330.0);

fn rem(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }
fn pw(x: f32, y: f32) -> f32 { return select(0.0, pow(x, y), x > 0.0); }
fn sm(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
fn cbrt(x: f32) -> f32 { return sign(x) * pw(abs(x), 1.0 / 3.0); }
fn srgb_eotf(v: f32) -> f32 { return select(pw((v + 0.055) / 1.055, 2.4), v / 12.92, v <= 0.04045); }
fn srgb_oetf(l: f32) -> f32 { return select(1.055 * pw(l, 1.0 / 2.4) - 0.055, l * 12.92, l <= 0.0031308); }

fn oklab(p: vec3<f32>) -> vec3<f32> {
  let l = cbrt(0.6166884 * p.x + 0.3601591 * p.y + 0.0230433 * p.z);
  let m = cbrt(0.2651402 * p.x + 0.6358565 * p.y + 0.0990302 * p.z);
  let s = cbrt(0.1001506 * p.x + 0.2040043 * p.y + 0.6963247 * p.z);
  return vec3<f32>(
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}
fn from_oklab(c: vec3<f32>) -> vec3<f32> {
  let l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  let m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  let s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  let a = l * l * l; let b = m * m * m; let d = s * s * s;
  return vec3<f32>(
    2.1401404 * a - 1.2463560 * b + 0.1064317 * d,
    -0.8848325 * a + 2.1631727 * b - 0.2783616 * d,
    -0.0485791 * a - 0.4544909 * b + 1.5023563 * d);
}
fn enc(p: vec3<f32>) -> vec3<f32> {
  let r = vec3<f32>(
    1.6604910 * p.x - 0.5876411 * p.y - 0.0728499 * p.z,
    -0.1245505 * p.x + 1.1328999 * p.y - 0.0083494 * p.z,
    -0.0181508 * p.x - 0.1005789 * p.y + 1.1187297 * p.z);
  return vec3<f32>(srgb_oetf(clamp(r.x, 0.0, 1.0)), srgb_oetf(clamp(r.y, 0.0, 1.0)), srgb_oetf(clamp(r.z, 0.0, 1.0)));
}
fn dec(e: vec3<f32>) -> vec3<f32> {
  let q = vec3<f32>(srgb_eotf(e.x), srgb_eotf(e.y), srgb_eotf(e.z));
  return vec3<f32>(
    0.6274039 * q.x + 0.3292830 * q.y + 0.0433131 * q.z,
    0.0690973 * q.x + 0.9195404 * q.y + 0.0113623 * q.z,
    0.0163914 * q.x + 0.0880133 * q.y + 0.8955953 * q.z);
}

fn tone_ch(c: f32) -> f32 {
  var v = (sqrt(max(c, 0.0)) - u.bp) / (u.wp - u.bp);
  if (v > 0.0 && v < 1.0) { v = v + u.k * (v * v * (3.0 - 2.0 * v) - v); }
  v = max(v, 0.0);
  return v * v;
}

fn curve_ch(c: f32, ch: u32) -> f32 {
  let v = sqrt(max(c, 0.0));
  let base = ch * (u.lut_n + 1u);
  var o: f32;
  if (v >= 1.0) {
    o = lut[base + u.lut_n] + (v - 1.0);
  } else {
    let f = v * f32(u.lut_n);
    let i = u32(f);
    o = lut[base + i] + (lut[base + i + 1u] - lut[base + i]) * (f - f32(i));
  }
  return o * o;
}

fn color_ops(p: vec3<f32>) -> vec3<f32> {
  let lab = oklab(p);
  var l = lab.x;
  var c = length(lab.yz);
  var h = atan2(lab.z, lab.y);
  let sat = P[2]; let vib = P[3];
  if (sat != 0.0 || vib != 0.0) {
    let hd = rem(degrees(h), 360.0);
    let skin = 1.0 - 0.6 * sm(25.0, 0.0, abs(hd - 62.0));
    let v = vib * (1.0 - min(c / 0.25, 1.0)) * select(1.0, skin, vib > 0.0);
    c = c * max((1.0 + sat) * (1.0 + v), 0.0);
  }
  if (P[4] != 0.0) {
    let hd = rem(degrees(h), 360.0);
    var i = 7u;
    for (var k = 0u; k < 8u; k++) { if (BANDS[k] <= hd) { i = k; } }
    let j = (i + 1u) % 8u;
    let span = rem(BANDS[j] - BANDS[i], 360.0);
    let t = sm(0.0, 1.0, rem(hd - BANDS[i], 360.0) / span);
    let gate = sm(0.0, 0.06, c);
    let s0 = (P[5u + i * 3u] * (1.0 - t) + P[5u + j * 3u] * t) * gate;
    let s1 = (P[6u + i * 3u] * (1.0 - t) + P[6u + j * 3u] * t) * gate;
    let s2 = (P[7u + i * 3u] * (1.0 - t) + P[7u + j * 3u] * t) * gate;
    h += s0;
    c = c * max(1.0 + s1, 0.0);
    l += s2;
  }
  let nsw = u32(P[44]);
  for (var n = 0u; n < nsw; n++) {
    let o = 45u + n * 11u;
    let dh = rem(h - P[o] + PI, 2.0 * PI) - PI;
    let a = dh / P[o + 3u];
    let b = (c - P[o + 1u]) / P[o + 4u];
    let e = (l - P[o + 2u]) / P[o + 5u];
    let d = sqrt(a * a + b * b + e * e);
    let wgt = 1.0 - sm(P[o + 6u], P[o + 7u], d);
    if (wgt > 0.0) {
      h += P[o + 8u] * wgt;
      c = c * max(1.0 + P[o + 9u] * wgt, 0.0);
      l += P[o + 10u] * wgt;
    }
  }
  if (P[1] != 0.0) { c = 0.0; }
  var a2 = c * cos(h);
  var b2 = c * sin(h);
  if (P[29] != 0.0) {
    var lb = clamp(l, 0.0, 1.0);
    let bal = P[42]; let k = P[43];
    if (bal != 1.0) { lb = pw(lb, bal); }
    var w = vec3<f32>((1.0 - lb) * (1.0 - lb), 2.0 * lb * (1.0 - lb), lb * lb);
    if (k != 1.0) {
      w = vec3<f32>(pw(w.x, k), pw(w.y, k), pw(w.z, k));
      w = w / (w.x + w.y + w.z);
    }
    l += w.x * P[30] + w.y * P[33] + w.z * P[36];
    a2 += w.x * P[31] + w.y * P[34] + w.z * P[37];
    b2 += w.x * P[32] + w.y * P[35] + w.z * P[38];
    l += P[39]; a2 += P[40]; b2 += P[41];
  }
  return from_oklab(vec3<f32>(l, a2, b2));
}

fn pro_ops(pin: vec3<f32>) -> vec3<f32> {
  var p = pin;
  if (P[PR] != 0.0) {
    let e = enc(p);
    var o = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
      let f = e[c] * 1024.0;
      let i = min(u32(f), 1023u);
      let b = c * 1025u + i;
      o[c] = lv[b] + (lv[b + 1u] - lv[b]) * (f - f32(i));
    }
    p = dec(o);
  }
  if (P[PR + 1u] != 0.0) {
    let sn = P[PR + 2u]; let cs = P[PR + 3u]; let sat = P[PR + 4u]; let light = P[PR + 5u];
    let lab = oklab(p);
    var l = lab.x; var a = lab.y; var b = lab.z;
    if (P[PR + 6u] != 0.0) {
      let k = clamp(4.0 * l * (1.0 - l), 0.0, 1.0);
      a = P[PR + 7u] * k; b = P[PR + 8u] * k;
    } else {
      let na = (a * cs - b * sn) * sat;
      b = (a * sn + b * cs) * sat;
      a = na;
    }
    if (light > 0.0) { l += (1.0 - l) * light; } else { l *= 1.0 + light; }
    let k = 1.0 - abs(light);
    p = from_oklab(vec3<f32>(l, a * k, b * k));
  }
  if (P[PR + 9u] != 0.0) {
    let e = enc(p);
    let mx = max(e.x, max(e.y, e.z));
    let mn = min(e.x, min(e.y, e.z));
    let md = e.x + e.y + e.z - mx - mn;
    var wts = array<f32, 9>(
      select(0.0, mx - md, e.x == mx), select(0.0, md - mn, e.z == mn),
      select(0.0, mx - md, e.y == mx), select(0.0, md - mn, e.x == mn),
      select(0.0, mx - md, e.z == mx), select(0.0, md - mn, e.y == mn),
      max((mn - 0.5) * 2.0, 0.0), 1.0 - abs(mx - 0.5) - abs(mn - 0.5), max((0.5 - mx) * 2.0, 0.0));
    var o = e;
    let absolute = P[PR + 10u] != 0.0;
    for (var i = 0u; i < 9u; i++) {
      if (wts[i] > 0.0) {
        let f3 = P[PR + 14u + i * 4u];
        for (var c = 0u; c < 3u; c++) {
          let fc = P[PR + 11u + i * 4u + c];
          let ink = fc + f3 * (1.0 + fc);
          o[c] -= wts[i] * ink * select(1.0 - e[c], 1.0, absolute);
        }
      }
    }
    p = dec(clamp(o, vec3<f32>(0.0), vec3<f32>(1.0)));
  }
  if (P[PR + 47u] != 0.0) {
    let e = enc(p);
    var o = vec3<f32>(0.0);
    for (var r = 0u; r < 3u; r++) {
      let b = PR + 48u + r * 4u;
      o[r] = clamp(P[b] * e.x + P[b + 1u] * e.y + P[b + 2u] * e.z + P[b + 3u], 0.0, 1.0);
    }
    p = dec(o);
  }
  if (P[PR + 60u] != 0.0) {
    let q = oklab(p);
    let s = vec3<f32>(P[PR + 61u], P[PR + 62u], P[PR + 63u]);
    let h = vec3<f32>(P[PR + 64u], P[PR + 65u], P[PR + 66u]);
    let g = mix(s, h, clamp(q.x, 0.0, 1.0));
    p = from_oklab(mix(q, g, P[PR + 67u]));
  }
  if (P[PR + 68u] != 0.0 || P[PR + 69u] != 0.0 || P[PR + 71u] != 0.0) {
    var e = enc(p);
    if (P[PR + 68u] != 0.0) { e = vec3<f32>(1.0) - e; }
    let n = P[PR + 69u];
    if (n != 0.0) { e = floor(e * n + vec3<f32>(0.5)) / n; }
    if (P[PR + 71u] != 0.0) {
      let d = dec(e);
      let y = clamp(0.2627 * d.x + 0.6780 * d.y + 0.0593 * d.z, 0.0, 1.0);
      e = vec3<f32>(select(0.0, 1.0, srgb_oetf(y) >= P[PR + 70u]));
    }
    p = dec(e);
  }
  return p;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u.n) { return; }
  var p = vec3<f32>(src[3u*i], src[3u*i+1u], src[3u*i+2u]) * u.gain;
  if (u.use_tone != 0u) {
    if (u.a_sh != 0.0 || u.a_hi != 0.0) {
      let v = sqrt(max(max(max(p.x, p.y), p.z), 0.0));
      let ws = 1.0 - smoothstep(0.0, 0.5, v);
      let wh = smoothstep(0.35, 1.0, v);
      p = p * exp2(u.a_sh * ws + u.a_hi * wh);
    }
    p = vec3<f32>(tone_ch(p.x), tone_ch(p.y), tone_ch(p.z));
  }
  if (u.use_curves != 0u) {
    p = vec3<f32>(curve_ch(p.x, 0u), curve_ch(p.y, 1u), curve_ch(p.z, 2u));
  }
  if (P[0] != 0.0) { p = color_ops(p); }
  if (P[PR + 72u] != 0.0) { p = pro_ops(p); }
  dst[3u*i] = p.x; dst[3u*i+1u] = p.y; dst[3u*i+2u] = p.z;
}

