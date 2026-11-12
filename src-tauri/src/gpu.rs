//! Optional GPU acceleration (wgpu: DX12/Vulkan/Metal) using Ahead-of-Time (AOT) compiled
//! stripped SPIR-V (.spv) binary compute kernels generated at build time (`build.rs`): the develop stack
//! (with a 3D LUT look), the box blur under clarity, texture, dehaze and defringe, lateral chromatic
//! aberration, and the scopes. Each kernel does the CPU path's arithmetic, so a photo looks the same either way.
//! Detected once; every entry point returns `None` when there is no GPU or it is switched off,
//! and the caller keeps its CPU path.

use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};
use wgpu::util::DeviceExt;

type Px = [f32; 3];

/// Pixels per dispatch: 8M × 12 B stays under the default 128 MiB storage-binding limit.
const CHUNK: usize = 8 << 20;

/// Floats of colour parameters, then of Pro parameters, in the order `develop.rs` fills them
/// (mirrored by the offsets in `develop_stack.wgsl`). A zeroed section switches that stage off.
pub const COLOR_LEN: usize = 132;
pub const PRO_LEN: usize = 73;
/// Point-colour swatches that fit in the colour section (11 floats each).
pub const MAX_SW: usize = 8;

/// Ahead-of-Time (AOT) stripped SPIR-V compute kernels embedded at build time.
pub const DEVELOP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/develop_stack.spv"));
pub const BLUR_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/box_blur.spv"));
pub const LUT3D_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lut3d.spv"));
pub const CA_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/chromatic_aberration.spv"));
pub const SCOPES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scopes.spv"));

static ENABLED: AtomicBool = AtomicBool::new(true);
static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

struct Gpu {
    name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    lock: std::sync::Mutex<()>,
    pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    lut3d_pipeline: wgpu::ComputePipeline,
    ca_pipeline: wgpu::ComputePipeline,
    scopes_pipeline: wgpu::ComputePipeline,
    /// the largest storage buffer one binding takes (the adapter's own, not wgpu's 128 MiB default)
    max_binding: u64,
}

/// Global tone (`develop::Tone`) coefficients.
#[derive(Clone, Copy)]
pub struct ToneP {
    pub a_sh: f32,
    pub a_hi: f32,
    pub bp: f32,
    pub wp: f32,
    pub k: f32,
}

/// What to run on the GPU, in pipeline order.
pub struct Stack<'a> {
    pub gain: Px,
    pub tone: Option<ToneP>,
    pub curves: Option<&'a [Vec<f32>; 3]>,
    /// `COLOR_LEN + PRO_LEN` floats; a section left zero is skipped.
    pub params: Vec<f32>,
    /// Levels LUTs (1025 entries per channel) when the Pro section uses them.
    pub levels: Option<&'a [Vec<f32>; 3]>,
    /// A 3D LUT look on the result (`develop::Lut3d`), and how much of it.
    pub look: Option<(&'a crate::develop::Lut3d, f32)>,
}

fn make_pipeline(device: &wgpu::Device, label: &str, spv: &[u8], entry_point: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::util::make_spirv(spv),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn init() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return None; // software rasterizer: slower than our rayon path
    }
    // the adapter's buffer limits: a 36 MP float plane alone is past wgpu's default 128 MiB
    let al = adapter.limits();
    let required_limits = wgpu::Limits { max_storage_buffer_binding_size: al.max_storage_buffer_binding_size, max_buffer_size: al.max_buffer_size, ..wgpu::Limits::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits, ..Default::default() })).ok()?;
    let max_binding = (al.max_storage_buffer_binding_size as u64).min(al.max_buffer_size);
    let pipeline = make_pipeline(&device, "develop_stack_spv", DEVELOP_SPV, "main");
    let blur_pipeline = make_pipeline(&device, "box_blur_spv", BLUR_SPV, "main");
    let lut3d_pipeline = make_pipeline(&device, "lut3d_spv", LUT3D_SPV, "main");
    let ca_pipeline = make_pipeline(&device, "chromatic_aberration_spv", CA_SPV, "main");
    let scopes_pipeline = make_pipeline(&device, "scopes_spv", SCOPES_SPV, "accumulate");
    Some(Gpu {
        name: info.name,
        device,
        queue,
        lock: std::sync::Mutex::new(()),
        pipeline,
        blur_pipeline,
        lut3d_pipeline,
        ca_pipeline,
        scopes_pipeline,
        max_binding,
    })
}

fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(init).as_ref()
}

/// Adapter name when a usable GPU exists (probes on first call).
pub fn detect() -> Option<String> {
    gpu().map(|g| g.name.clone())
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Returns the list of embedded AOT SPIR-V binary modules and their sizes in bytes.
#[allow(dead_code)]
pub fn spirv_manifest() -> Vec<(&'static str, usize)> {
    vec![
        ("develop_stack.spv", DEVELOP_SPV.len()),
        ("box_blur.spv", BLUR_SPV.len()),
        ("lut3d.spv", LUT3D_SPV.len()),
        ("chromatic_aberration.spv", CA_SPV.len()),
        ("scopes.spv", SCOPES_SPV.len()),
    ]
}

/// Runs the stack over `src`. `None` = not available (off, no GPU, or a device error): use the CPU path.
pub fn run(src: &[Px], s: &Stack) -> Option<Vec<Px>> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let g = gpu()?;
    let buf = |v: &[f32]| {
        g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let lut_n = s.curves.map_or(0, |l| l[0].len() as u32 - 1);
    let lut = buf(&s.curves.map_or_else(|| vec![0.0; 3], |l| l.concat()));
    let lv = buf(&s.levels.map_or_else(|| vec![0.0; 3], |l| l.concat()));
    let prm = buf(&s.params);
    let look = s.look.map(|(l, _)| buf(&l.rgb));
    let mut out: Vec<Px> = Vec::with_capacity(src.len());
    for chunk in src.chunks(CHUNK) {
        out.extend(dispatch(g, chunk, s, lut_n, [&lut, &prm, &lv], look.as_ref())?);
    }
    Some(out)
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

fn dispatch(g: &Gpu, px: &[Px], s: &Stack, lut_n: u32, [lut, prm, lv]: [&wgpu::Buffer; 3], look: Option<&wgpu::Buffer>) -> Option<Vec<Px>> {
    let _guard = g.lock.lock().ok()?;
    let bytes = std::mem::size_of_val(px) as u64;
    let t = s.tone.unwrap_or(ToneP { a_sh: 0.0, a_hi: 0.0, bp: 0.0, wp: 1.0, k: 0.0 });
    // ints are passed as raw bits: the shader reads them as u32
    let bits = |v: u32| f32::from_bits(v);
    let u = [
        s.gain[0], s.gain[1], s.gain[2], bits(px.len() as u32),
        t.a_sh, t.a_hi, t.bp, t.wp,
        t.k, bits(s.tone.is_some() as u32), bits(s.curves.is_some() as u32), bits(lut_n),
        0.0, 0.0, 0.0, 0.0,
    ];
    let uni = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&u),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let src = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(px),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dst = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &uni), entry(1, &src), entry(2, &dst), entry(3, lut), entry(4, prm), entry(5, lv)],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((px.len() as u32).div_ceil(256), 1, 1);
    }
    // the look on the result, in place, before it leaves the GPU
    if let (Some(lb), Some((l, amount))) = (look, s.look) {
        let u = uniform(g, &[px.len() as u32, l.size as u32, amount.to_bits(), 0]);
        let look_bind = bind_group(g, &g.lut3d_pipeline, &[&dst, lb, &u]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.lut3d_pipeline);
        pass.set_bind_group(0, &look_bind, &[]);
        pass.dispatch_workgroups((px.len() as u32).div_ceil(256), 1, 1);
    }
    enc.copy_buffer_to_buffer(&dst, 0, &read, 0, bytes);
    g.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;
    let data = read.slice(..).get_mapped_range().ok()?;
    let out: Vec<Px> = bytemuck::cast_slice::<u8, f32>(&data).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    drop(data);
    read.unmap();
    Some(out)
}

fn uniform(g: &Gpu, words: &[u32; 4]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(words), usage: wgpu::BufferUsages::UNIFORM })
}

const RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);

/// A storage buffer holding `data`, which passes may also write and which can be read back.
fn storage(g: &Gpu, data: &[u8]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: RW })
}

/// An uninitialised storage buffer of `bytes` (nothing uploaded) for a pass to write.
fn scratch(g: &Gpu, bytes: u64) -> wgpu::Buffer {
    g.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: RW, mapped_at_creation: false })
}

fn bind_group(g: &Gpu, p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = bufs.iter().enumerate().map(|(i, b)| entry(i as u32, b)).collect();
    g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
}

/// Submits `enc`, then `buf`'s first `bytes`.
fn finish_and_read(g: &Gpu, mut enc: wgpu::CommandEncoder, buf: &wgpu::Buffer, bytes: u64) -> Option<Vec<u8>> {
    let read = g.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(buf, 0, &read, 0, bytes);
    g.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;
    let out = read.slice(..).get_mapped_range().ok()?.to_vec();
    read.unmap();
    Some(out)
}

/// `develop::blur`'s box passes on the GPU: radius `r`, three along the rows then three down the
/// columns, with the CPU's running sums. None: off, no GPU, or more than one buffer holds.
pub fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Option<Vec<f32>> {
    if !ENABLED.load(Ordering::Relaxed) || w == 0 || h == 0 || src.len() != w * h {
        return None;
    }
    let g = gpu()?;
    let bytes = std::mem::size_of_val(src) as u64;
    if bytes > g.max_binding {
        return None;
    }
    let _guard = g.lock.lock().ok()?;
    let (a, b) = (storage(g, bytemuck::cast_slice(src)), scratch(g, bytes));
    let mut enc = g.device.create_command_encoder(&Default::default());
    for (k, axis) in [0u32, 0, 0, 1, 1, 1].into_iter().enumerate() {
        let (from, to) = if k % 2 == 0 { (&a, &b) } else { (&b, &a) };
        let u = uniform(g, &[w as u32, h as u32, r as u32, axis]);
        let bg = bind_group(g, &g.blur_pipeline, &[from, to, &u]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.blur_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((if axis == 1 { w } else { h } as u32).div_ceil(64), 1, 1);
    }
    // six passes, so the result is back in `a`
    finish_and_read(g, enc, &a, bytes).map(|v| bytemuck::cast_slice(&v).to_vec())
}

/// `develop::lens`'s lateral chromatic aberration on the GPU: red and blue rescaled about the centre by
/// `scale_red` and `scale_blue`, in linear sRGB.
pub fn lateral_ca(px: &[Px], w: usize, h: usize, scale_red: f32, scale_blue: f32) -> Option<Vec<Px>> {
    if !ENABLED.load(Ordering::Relaxed) || w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let g = gpu()?;
    let bytes = std::mem::size_of_val(px) as u64;
    if bytes > g.max_binding {
        return None;
    }
    let _guard = g.lock.lock().ok()?;
    let (src, dst) = (storage(g, bytemuck::cast_slice(px)), scratch(g, bytes));
    let u = uniform(g, &[w as u32, h as u32, scale_red.to_bits(), scale_blue.to_bits()]);
    let bg = bind_group(g, &g.ca_pipeline, &[&src, &dst, &u]);
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.ca_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((w as u32).div_ceil(16), (h as u32).div_ceil(16), 1);
    }
    finish_and_read(g, enc, &dst, bytes).map(|v| bytemuck::cast_slice::<u8, f32>(&v).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

/// Runs the pre-compiled SPIR-V RGB Parade & Waveform scope accumulation kernel on the GPU.
pub fn run_scopes(src_rgba: &[u8], w: u32, h: u32, scope_w: u32, scope_h: u32) -> Option<(Vec<u8>, Vec<u8>)> {
    if !ENABLED.load(Ordering::Relaxed) || src_rgba.len() < (w as usize) * (h as usize) * 4 {
        return None;
    }
    let g = gpu()?;
    let _guard = g.lock.lock().ok()?;
    let px_count = (w as usize) * (h as usize);
    let mut src4 = Vec::with_capacity(px_count);
    for px in src_rgba[..px_count * 4].chunks_exact(4) {
        src4.push([
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
            1.0f32,
        ]);
    }
    let (p_len, w_len) = ((scope_w * scope_h * 3) as usize, (scope_w * scope_h * 4) as usize);
    let (p_bytes, w_bytes) = ((p_len * 4) as u64, (w_len * 4) as u64);

    let bits = f32::from_bits;
    let u = [bits(w), bits(h), bits(scope_w), bits(scope_h)];
    let uni = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&u),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let src_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&src4),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let parade_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vec![0u32; p_len]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let wf_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vec![0u32; w_len]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let read_parade = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: p_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_wf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: w_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.scopes_pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, &src_buf),
            entry(1, &parade_buf),
            entry(2, &wf_buf),
            entry(3, &uni),
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.scopes_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    enc.copy_buffer_to_buffer(&parade_buf, 0, &read_parade, 0, p_bytes);
    enc.copy_buffer_to_buffer(&wf_buf, 0, &read_wf, 0, w_bytes);
    g.queue.submit([enc.finish()]);

    let (tx1, rx1) = std::sync::mpsc::channel();
    read_parade.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx1.send(r);
    });
    let (tx2, rx2) = std::sync::mpsc::channel();
    read_wf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx2.send(r);
    });
    g.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx1.recv().ok()?.ok()?;
    rx2.recv().ok()?.ok()?;

    let p_data = read_parade.slice(..).get_mapped_range().ok()?;
    let w_data = read_wf.slice(..).get_mapped_range().ok()?;
    let out = render_scopes(bytemuck::cast_slice(&p_data), bytemuck::cast_slice(&w_data), scope_w as usize, scope_h as usize, px_count);
    drop(p_data);
    drop(w_data);
    read_parade.unmap();
    read_wf.unmap();
    Some(out)
}

/// Scope bins as `scopes.wgsl` (or the C++ fallback, `ol_scope_bins`) fills them, drawn as RGBA8 images
/// for a photo of `px` pixels. A bin's brightness follows its density against an even spread of the
/// photo's pixels, 1 - e^(-n / 2·even); the waveform takes its hue from its pixels' average colour.
/// Dark ground, graticule lines at 25 / 50 / 75%.
pub fn render_scopes(parade: &[u32], wave: &[u32], sw: usize, sh: usize, px: usize) -> (Vec<u8>, Vec<u8>) {
    let even_p = px as f32 / ((sw / 3).max(1) * sh) as f32;
    let even_w = px as f32 / (sw * sh) as f32;
    let tone = |n: f32, even: f32| 1.0 - (-n / (2.0 * even.max(1e-6))).exp();
    let grat = [sh / 4, sh / 2, sh * 3 / 4];
    let (mut p_rgba, mut w_rgba) = (vec![0u8; sw * sh * 4], vec![0u8; sw * sh * 4]);
    for i in 0..sw * sh {
        let ground = if grat.contains(&(i / sw)) { 34.0 } else { 14.0 };
        let px8 = |v: f32| (v * 255.0).clamp(ground, 255.0) as u8;
        for k in 0..3 {
            p_rgba[i * 4 + k] = px8(tone(parade[i * 3 + k] as f32, even_p));
        }
        // each pixel added 16 per full channel plus 1, and 1 to the count
        let n = wave[i * 4 + 3] as f32;
        let avg = [0, 1, 2].map(|k| if n > 0.0 { (wave[i * 4 + k] as f32 / n - 1.0) / 16.0 } else { 0.0 });
        let top = avg.iter().fold(0.0f32, |a, &b| a.max(b));
        let d = tone(n, even_w);
        for k in 0..3 {
            let tint = if top > 1e-3 { avg[k] / top } else { 1.0 };
            w_rgba[i * 4 + k] = px8(d * (0.3 + 0.7 * tint));
        }
        p_rgba[i * 4 + 3] = 255;
        w_rgba[i * 4 + 3] = 255;
    }
    (p_rgba, w_rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_binaries_have_valid_magic_headers() {
        assert_eq!(spirv_manifest().len(), 5);
        for (name, bytes) in [
            ("develop_stack.spv", DEVELOP_SPV),
            ("box_blur.spv", BLUR_SPV),
            ("lut3d.spv", LUT3D_SPV),
            ("chromatic_aberration.spv", CA_SPV),
            ("scopes.spv", SCOPES_SPV),
        ] {
            assert!(bytes.len() > 20, "{name} should be non-empty");
            assert_eq!(bytes.len() % 4, 0, "{name} length must be a multiple of 4");
            let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            assert_eq!(magic, 0x0723_0203, "{name} must start with SPIR-V magic number 0x07230203");
        }
    }

    #[test]
    fn gpu_kernels_execute_if_adapter_available() {
        if detect().is_none() {
            return;
        }
        let w = 32u32;
        let h = 24u32;
        let src: Vec<Px> = (0..(w * h))
            .map(|i| {
                let f = i as f32 / (w * h) as f32;
                [f * 0.8, f * 0.5 + 0.1, 1.0 - f * 0.7]
            })
            .collect();
        // the blur and the chromatic aberration give the CPU's numbers
        let plane: Vec<f32> = src.iter().map(|p| p[1]).collect();
        for r in [1usize, 3, 9] {
            let gpu = box_blur(&plane, w as usize, h as usize, r).expect("blur kernel");
            let cpu = crate::develop::box_blur_cpu(&plane, w as usize, h as usize, r);
            let worst = gpu.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-5, "blur r {r}: off by {worst}");
        }
        let ca = lateral_ca(&src, w as usize, h as usize, 1.002, 0.998).expect("ca kernel");
        let cpu = crate::develop::lateral_ca_cpu(&src, w as usize, h as usize, [1.002, 1.0, 0.998]);
        let worst = ca.iter().zip(&cpu).flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs())).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "chromatic aberration off by {worst}");

        let mut rgba = vec![128u8; (w * h * 4) as usize];
        for (i, p) in src.iter().enumerate() {
            rgba[i * 4] = (p[0] * 255.0) as u8;
            rgba[i * 4 + 1] = (p[1] * 255.0) as u8;
            rgba[i * 4 + 2] = (p[2] * 255.0) as u8;
            rgba[i * 4 + 3] = 255;
        }
        let (parade, wf) = run_scopes(&rgba, w, h, 48, 32).expect("scopes kernel");
        assert_eq!(parade.len(), 48 * 32 * 4);
        assert_eq!(wf.len(), 48 * 32 * 4);
        // the GPU and the C++ fallback draw the same scopes (a float rounding may move a pixel a bin)
        let (cp, cw) = crate::libraw::compute_scopes_cpu(&rgba, w as usize, h as usize, 48, 32);
        let diff = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / a.len() as f64;
        assert!(diff(&parade, &cp) < 1.0 && diff(&wf, &cw) < 1.0, "parade {} waveform {}", diff(&parade, &cp), diff(&wf, &cw));
    }
}


