// Optixel Studio — Multiscale Local Clarity, Texture & Dark-Channel Dehaze Compute Shader
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V (.spv) at cargo build time.

struct ClarityDehazeUniforms {
    width: u32,
    height: u32,
    clarity: f32,
    dehaze: f32,
    texture_amt: f32,
    radius: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: ClarityDehazeUniforms;

fn sample_clamped(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    let idx = u32(cy) * u.width + u32(cx);
    return src_buf[idx].xyz;
}

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = gid.y * u.width + gid.x;
    let center4 = src_buf[idx];
    var rgb = center4.xyz;
    let y0 = luma(rgb);

    // 1. Fine-scale Laplacian pyramid band (Texture)
    if (abs(u.texture_amt) > 0.001) {
        let fine = (
            sample_clamped(x - 1, y) +
            sample_clamped(x + 1, y) +
            sample_clamped(x, y - 1) +
            sample_clamped(x, y + 1) +
            sample_clamped(x, y) * 4.0
        ) * 0.125;
        let detail_fine = y0 - luma(fine);
        let mid_weight = 4.0 * clamp(y0, 0.0, 1.0) * (1.0 - clamp(y0, 0.0, 1.0));
        rgb = rgb + vec3<f32>(detail_fine * (u.texture_amt * 0.012) * (0.35 + 0.65 * mid_weight));
    }

    // 2. Coarse-scale multiscale Laplacian pyramid band (Clarity)
    if (abs(u.clarity) > 0.001) {
        let r1 = max(i32(u.radius), 2);
        let r2 = r1 * 2;
        var sum_coarse = vec3<f32>(0.0);
        var w_sum = 0.0;
        for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
            for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
                let sample_c = sample_clamped(x + dx * r1, y + dy * r1);
                let dist2 = f32(dx * dx + dy * dy);
                let w = exp(-0.35 * dist2);
                sum_coarse = sum_coarse + sample_c * w;
                w_sum = w_sum + w;
            }
        }
        let coarse1 = sum_coarse / w_sum;
        let coarse2 = (
            sample_clamped(x - r2, y) +
            sample_clamped(x + r2, y) +
            sample_clamped(x, y - r2) +
            sample_clamped(x, y + r2) +
            coarse1 * 2.0
        ) / 6.0;
        let band_mid = luma(coarse1) - luma(coarse2);
        let band_hi = y0 - luma(coarse1);
        let mid_mask = 4.0 * clamp(y0, 0.0, 1.0) * (1.0 - clamp(y0, 0.0, 1.0));
        let delta = (0.65 * band_mid + 0.35 * band_hi) * (u.clarity * 0.01) * (0.25 + 0.75 * mid_mask);
        rgb = rgb + vec3<f32>(delta);
    }

    // 3. Dark-Channel Atmospheric Dehaze (He et al. 2011)
    if (abs(u.dehaze) > 0.001) {
        let step_px = 3;
        var dark = min(rgb.x, min(rgb.y, rgb.z));
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let p = sample_clamped(x + dx * step_px, y + dy * step_px);
                dark = min(dark, min(p.x, min(p.y, p.z)));
            }
        }
        let d = u.dehaze * 0.01;
        if (d > 0.0) {
            let airlight = vec3<f32>(0.92, 0.94, 0.97);
            let transmission = clamp(1.0 - d * 0.85 * dark, 0.18, 1.0);
            rgb = (rgb - airlight * (1.0 - transmission)) / transmission;
        } else {
            let fog = clamp(-d * 0.65, 0.0, 0.85);
            rgb = mix(rgb, vec3<f32>(0.92, 0.93, 0.95), fog * (1.0 - dark * 0.4));
        }
    }

    dst_buf[idx] = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), center4.w);
}
