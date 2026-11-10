// Optixel Studio — Real-Time RGB Parade & Luma Waveform Scopes Compute Shader
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V (.spv) at cargo build time.

struct ScopeUniforms {
    src_w: u32,
    src_h: u32,
    scope_w: u32,
    scope_h: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> parade_bins: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> waveform_bins: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> u: ScopeUniforms;

@compute @workgroup_size(16, 16, 1)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.src_w || gid.y >= u.src_h) {
        return;
    }
    let idx = gid.y * u.src_w + gid.x;
    let rgb = clamp(src_buf[idx].xyz, vec3<f32>(0.0), vec3<f32>(1.0));

    let sw = u.scope_w;
    let sh = u.scope_h;
    let third = max(sw / 3u, 1u);

    let sub_x = min((gid.x * third) / max(u.src_w, 1u), third - 1u);
    let wf_x = min((gid.x * sw) / max(u.src_w, 1u), sw - 1u);

    let ry = min(u32((1.0 - rgb.x) * f32(sh - 1u) + 0.5), sh - 1u);
    let gy = min(u32((1.0 - rgb.y) * f32(sh - 1u) + 0.5), sh - 1u);
    let by = min(u32((1.0 - rgb.z) * f32(sh - 1u) + 0.5), sh - 1u);

    let px_r = sub_x;
    let px_g = min(third + sub_x, sw - 1u);
    let px_b = min(2u * third + sub_x, sw - 1u);

    atomicAdd(&parade_bins[(ry * sw + px_r) * 3u + 0u], 1u);
    atomicAdd(&parade_bins[(gy * sw + px_g) * 3u + 1u], 1u);
    atomicAdd(&parade_bins[(by * sw + px_b) * 3u + 2u], 1u);

    let luma = clamp(0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z, 0.0, 1.0);
    let wy = min(u32((1.0 - luma) * f32(sh - 1u) + 0.5), sh - 1u);
    let w_idx = (wy * sw + wf_x) * 3u;
    atomicAdd(&waveform_bins[w_idx + 0u], u32(rgb.x * 16.0 + 1.0));
    atomicAdd(&waveform_bins[w_idx + 1u], u32(rgb.y * 16.0 + 1.0));
    atomicAdd(&waveform_bins[w_idx + 2u], u32(rgb.z * 16.0 + 1.0));
}

