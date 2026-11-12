// Optixel Studio — lateral chromatic aberration, as develop.rs `lens` corrects it: red and blue rescaled
// about the centre (bilinear, edge-clamped) in linear sRGB, the channels a lens's fringes are in, green
// kept. Pixels are packed RGB floats in the working space (linear Rec.2020).
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V at build time.

struct CaUniforms {
    width: u32,
    height: u32,
    scale_red: f32,
    scale_blue: f32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: CaUniforms;

// develop.rs M2020_709 and M709_2020, by rows
const TO_709_R = vec3<f32>(1.6604910, -0.5876411, -0.0728499);
const TO_709_G = vec3<f32>(-0.1245505, 1.1328999, -0.0083494);
const TO_709_B = vec3<f32>(-0.0181508, -0.1005789, 1.1187297);
const TO_2020_R = vec3<f32>(0.6274039, 0.3292830, 0.0433131);
const TO_2020_G = vec3<f32>(0.0690973, 0.9195404, 0.0113623);
const TO_2020_B = vec3<f32>(0.0163914, 0.0880133, 0.8955953);

fn at(x: u32, y: u32) -> vec3<f32> {
    let i = (y * u.width + x) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

// `x`, `y` already inside the frame (pixel centres on whole numbers)
fn bilinear(x: f32, y: f32) -> vec3<f32> {
    let x0 = u32(x);
    let y0 = u32(y);
    let x1 = min(x0 + 1u, u.width - 1u);
    let y1 = min(y0 + 1u, u.height - 1u);
    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let a = at(x0, y0);
    let top = a + (at(x1, y0) - a) * tx;
    let c = at(x0, y1);
    let bottom = c + (at(x1, y1) - c) * tx;
    return top + (bottom - top) * ty;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let cx = f32(u.width) / 2.0;
    let cy = f32(u.height) / 2.0;
    let x = f32(gid.x) + 0.5 - cx;
    let y = f32(gid.y) + 0.5 - cy;
    let mx = f32(u.width - 1u);
    let my = f32(u.height - 1u);
    let red = bilinear(clamp(cx + x * u.scale_red - 0.5, 0.0, mx), clamp(cy + y * u.scale_red - 0.5, 0.0, my));
    let blue = bilinear(clamp(cx + x * u.scale_blue - 0.5, 0.0, mx), clamp(cy + y * u.scale_blue - 0.5, 0.0, my));
    let srgb = vec3<f32>(dot(TO_709_R, red), dot(TO_709_G, at(gid.x, gid.y)), dot(TO_709_B, blue));
    let i = (gid.y * u.width + gid.x) * 3u;
    dst[i] = dot(TO_2020_R, srgb);
    dst[i + 1u] = dot(TO_2020_G, srgb);
    dst[i + 2u] = dot(TO_2020_B, srgb);
}
