// Optixel Studio — 3D LUT Trilinear Interpolation Compute Shader
// Compiled Ahead-of-Time (AOT) into stripped SPIR-V (.spv) at cargo build time.

struct Lut3dUniforms {
    width: u32,
    height: u32,
    lut_size: u32,
    strength: f32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lut_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> u: Lut3dUniforms;

fn sample_lut(ri: u32, gi: u32, bi: u32) -> vec3<f32> {
    let n = u.lut_size;
    let idx = (bi * n + gi) * n + ri;
    return lut_buf[idx].xyz;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let idx = gid.y * u.width + gid.x;
    let src4 = src_buf[idx];
    let rgb = clamp(src4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let n = u.lut_size;
    if (n < 2u || u.strength <= 0.0001) {
        dst_buf[idx] = src4;
        return;
    }

    let scale = f32(n - 1u);
    let pos = rgb * scale;
    let r0 = min(u32(floor(pos.x)), n - 1u);
    let g0 = min(u32(floor(pos.y)), n - 1u);
    let b0 = min(u32(floor(pos.z)), n - 1u);
    let r1 = min(r0 + 1u, n - 1u);
    let g1 = min(g0 + 1u, n - 1u);
    let b1 = min(b0 + 1u, n - 1u);

    let fr = pos.x - f32(r0);
    let fg = pos.y - f32(g0);
    let fb = pos.z - f32(b0);

    let c000 = sample_lut(r0, g0, b0);
    let c100 = sample_lut(r1, g0, b0);
    let c010 = sample_lut(r0, g1, b0);
    let c110 = sample_lut(r1, g1, b0);
    let c001 = sample_lut(r0, g0, b1);
    let c101 = sample_lut(r1, g0, b1);
    let c011 = sample_lut(r0, g1, b1);
    let c111 = sample_lut(r1, g1, b1);

    let c00 = mix(c000, c100, fr);
    let c10 = mix(c010, c110, fr);
    let c01 = mix(c001, c101, fr);
    let c11 = mix(c011, c111, fr);

    let c0 = mix(c00, c10, fg);
    let c1 = mix(c01, c11, fg);
    let lut_rgb = mix(c0, c1, fb);

    let out_rgb = mix(rgb, lut_rgb, clamp(u.strength, 0.0, 1.0));
    dst_buf[idx] = vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), src4.w);
}

