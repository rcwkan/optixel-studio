// Optixel Studio — a 3D LUT look (.cube), in place on the develop stack's result: each working-space pixel
// (linear Rec.2020) to sRGB display values, the domain looks are made in, trilinearly through the LUT, back,
// and blended by the amount. develop.rs `apply_look` does the same steps on the CPU.
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V at build time.

struct LookUniforms {
    count: u32,
    // entries per side; the table is size³ RGB, red fastest
    size: u32,
    amount: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> px: array<f32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<uniform> u: LookUniforms;

// develop.rs M2020_709 and M709_2020, by rows
const TO_709_R = vec3<f32>(1.6604910, -0.5876411, -0.0728499);
const TO_709_G = vec3<f32>(-0.1245505, 1.1328999, -0.0083494);
const TO_709_B = vec3<f32>(-0.0181508, -0.1005789, 1.1187297);
const TO_2020_R = vec3<f32>(0.6274039, 0.3292830, 0.0433131);
const TO_2020_G = vec3<f32>(0.0690973, 0.9195404, 0.0113623);
const TO_2020_B = vec3<f32>(0.0163914, 0.0880133, 0.8955953);

fn oetf(l: f32) -> f32 {
    if (l <= 0.0031308) {
        return l * 12.92;
    }
    return 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

fn eotf(v: f32) -> f32 {
    if (v <= 0.04045) {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn entry(r: u32, g: u32, b: u32) -> vec3<f32> {
    let k = ((b * u.size + g) * u.size + r) * 3u;
    return vec3<f32>(lut[k], lut[k + 1u], lut[k + 2u]);
}

fn mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.count || u.size < 2u) {
        return;
    }
    let i = gid.x * 3u;
    let p = vec3<f32>(px[i], px[i + 1u], px[i + 2u]);
    let s = vec3<f32>(dot(TO_709_R, p), dot(TO_709_G, p), dot(TO_709_B, p));
    let e = vec3<f32>(oetf(clamp(s.x, 0.0, 1.0)), oetf(clamp(s.y, 0.0, 1.0)), oetf(clamp(s.z, 0.0, 1.0)));
    let n = u.size;
    let pos = e * f32(n - 1u);
    let r0 = min(u32(floor(pos.x)), n - 1u);
    let g0 = min(u32(floor(pos.y)), n - 1u);
    let b0 = min(u32(floor(pos.z)), n - 1u);
    let r1 = min(r0 + 1u, n - 1u);
    let g1 = min(g0 + 1u, n - 1u);
    let b1 = min(b0 + 1u, n - 1u);
    let t = pos - vec3<f32>(f32(r0), f32(g0), f32(b0));
    let c0 = mix3(mix3(entry(r0, g0, b0), entry(r1, g0, b0), t.x), mix3(entry(r0, g1, b0), entry(r1, g1, b0), t.x), t.y);
    let c1 = mix3(mix3(entry(r0, g0, b1), entry(r1, g0, b1), t.x), mix3(entry(r0, g1, b1), entry(r1, g1, b1), t.x), t.y);
    let l = mix3(c0, c1, t.z);
    let lin = vec3<f32>(eotf(l.x), eotf(l.y), eotf(l.z));
    let back = vec3<f32>(dot(TO_2020_R, lin), dot(TO_2020_G, lin), dot(TO_2020_B, lin));
    let o = p + (back - p) * u.amount;
    px[i] = o.x;
    px[i + 1u] = o.y;
    px[i + 2u] = o.z;
}
