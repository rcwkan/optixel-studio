// Optixel Studio — Lateral & Axial Chromatic Aberration Correction Compute Shader
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V (.spv) at cargo build time.

struct CaUniforms {
    width: u32,
    height: u32,
    ca_red_cyan: f32,
    ca_blue_yellow: f32,
    axial_defringe: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: CaUniforms;

fn fetch_px(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return src_buf[u32(cy) * u.width + u32(cx)].xyz;
}

fn sample_bilinear(fx: f32, fy: f32) -> vec3<f32> {
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let dx = fx - f32(x0);
    let dy = fy - f32(y0);
    let c00 = fetch_px(x0, y0);
    let c10 = fetch_px(x0 + 1, y0);
    let c01 = fetch_px(x0, y0 + 1);
    let c11 = fetch_px(x0 + 1, y0 + 1);
    let top = mix(c00, c10, dx);
    let bot = mix(c01, c11, dx);
    return mix(top, bot, dy);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let idx = gid.y * u.width + gid.x;
    let center4 = src_buf[idx];
    let cx = f32(u.width) * 0.5;
    let cy = f32(u.height) * 0.5;
    let dx = f32(gid.x) - cx;
    let dy = f32(gid.y) - cy;
    let inv_diag = 1.0 / max(sqrt(cx * cx + cy * cy), 1.0);
    let r2 = (dx * dx + dy * dy) * (inv_diag * inv_diag);

    // Lateral radial dispersion model: scale R and B relative to reference G
    let scale_r = 1.0 + (u.ca_red_cyan * 0.00025) * (0.65 + 0.35 * r2);
    let scale_b = 1.0 + (u.ca_blue_yellow * 0.00025) * (0.65 + 0.35 * r2);

    let r_sample = sample_bilinear(cx + dx * scale_r, cy + dy * scale_r).x;
    let g_sample = center4.y;
    let b_sample = sample_bilinear(cx + dx * scale_b, cy + dy * scale_b).z;

    var rgb = vec3<f32>(r_sample, g_sample, b_sample);

    // Axial purple/green fringe suppression on high-contrast edges
    if (u.axial_defringe > 0.001) {
        let x = i32(gid.x);
        let y = i32(gid.y);
        let g_left = fetch_px(x - 1, y).y;
        let g_right = fetch_px(x + 1, y).y;
        let g_up = fetch_px(x, y - 1).y;
        let g_down = fetch_px(x, y + 1).y;
        let edge = abs(g_right - g_left) + abs(g_down - g_up);
        let purple_excess = max(0.0, min(rgb.x - rgb.y, rgb.z - rgb.y));
        let suppress = clamp(edge * purple_excess * u.axial_defringe * 4.0, 0.0, 0.85);
        rgb.x = mix(rgb.x, rgb.y, suppress);
        rgb.z = mix(rgb.z, rgb.y, suppress);
    }

    dst_buf[idx] = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), center4.w);
}

