// Optixel Studio — box blur, one pass along the rows or the columns: develop.rs `box_rows`, run on every
// line at once. Three passes each way make `develop::blur`, the Gaussian that clarity, texture, dehaze,
// defringe and the masks are built on. Compiled Ahead-of-Time (AOT) into stripped SPIR-V at build time.

struct BlurUniforms {
    width: u32,
    height: u32,
    radius: u32,
    // 0: along the rows, 1: down the columns
    axis: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: BlurUniforms;

// one line per invocation, a running sum along it, the ends repeated past the edges (as on the CPU,
// in the same order, so both give the same numbers)
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = u.axis == 1u;
    let lines = select(u.height, u.width, cols);
    if (gid.x >= lines) {
        return;
    }
    let n = select(u.width, u.height, cols);
    let stride = select(1u, u.width, cols);
    let base = select(gid.x * u.width, gid.x, cols);
    let last = n - 1u;
    let r = u.radius;
    let inv = 1.0 / f32(2u * r + 1u);
    var acc = src[base] * f32(r);
    for (var k = 0u; k <= r; k = k + 1u) {
        acc = acc + src[base + min(k, last) * stride];
    }
    for (var i = 0u; i < n; i = i + 1u) {
        dst[base + i * stride] = acc * inv;
        acc = acc + (src[base + min(i + r + 1u, last) * stride] - src[base + (i - min(i, r)) * stride]);
    }
}
