//! Optional GPU acceleration (wgpu: DX12/Vulkan/Metal) using Ahead-of-Time (AOT) compiled
//! stripped SPIR-V (.spv) binary compute kernels generated at build time (`build.rs`).
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
pub const CLARITY_DEHAZE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/clarity_dehaze.spv"));
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
    #[allow(dead_code)]
    clarity_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    lut3d_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    ca_pipeline: wgpu::ComputePipeline,
    scopes_pipeline: wgpu::ComputePipeline,
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    let pipeline = make_pipeline(&device, "develop_stack_spv", DEVELOP_SPV, "main");
    let clarity_pipeline = make_pipeline(&device, "clarity_dehaze_spv", CLARITY_DEHAZE_SPV, "main");
    let lut3d_pipeline = make_pipeline(&device, "lut3d_spv", LUT3D_SPV, "main");
    let ca_pipeline = make_pipeline(&device, "chromatic_aberration_spv", CA_SPV, "main");
    let scopes_pipeline = make_pipeline(&device, "scopes_spv", SCOPES_SPV, "accumulate");
    Some(Gpu {
        name: info.name,
        device,
        queue,
        lock: std::sync::Mutex::new(()),
        pipeline,
        clarity_pipeline,
        lut3d_pipeline,
        ca_pipeline,
        scopes_pipeline,
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
        ("clarity_dehaze.spv", CLARITY_DEHAZE_SPV.len()),
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
    let mut out: Vec<Px> = Vec::with_capacity(src.len());
    for chunk in src.chunks(CHUNK) {
        out.extend(dispatch(g, chunk, s, lut_n, [&lut, &prm, &lv])?);
    }
    Some(out)
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

fn dispatch(g: &Gpu, px: &[Px], s: &Stack, lut_n: u32, [lut, prm, lv]: [&wgpu::Buffer; 3]) -> Option<Vec<Px>> {
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

#[allow(dead_code)]
fn px_to_vec4(src: &[Px]) -> Vec<[f32; 4]> {
    src.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect()
}

#[allow(dead_code)]
fn readback_vec4(g: &Gpu, dst: &wgpu::Buffer, count: usize) -> Option<Vec<Px>> {
    let bytes = (count * 16) as u64;
    let read = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(dst, 0, &read, 0, bytes);
    g.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;
    let data = read.slice(..).get_mapped_range().ok()?;
    let out: Vec<Px> = bytemuck::cast_slice::<u8, [f32; 4]>(&data)
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    drop(data);
    read.unmap();
    Some(out)
}

/// Runs the pre-compiled SPIR-V Local Clarity, Texture & Dark-Channel Dehaze compute kernel.
#[allow(dead_code)]
pub fn run_clarity_dehaze(src: &[Px], w: u32, h: u32, clarity: f32, dehaze: f32, texture_amt: f32) -> Option<Vec<Px>> {
    if !ENABLED.load(Ordering::Relaxed) || (w as usize) * (h as usize) != src.len() {
        return None;
    }
    let g = gpu()?;
    let _guard = g.lock.lock().ok()?;
    let src4 = px_to_vec4(src);
    let bytes = (src4.len() * 16) as u64;
    if bytes > 120 * 1024 * 1024 {
        return None;
    }
    let radius = ((w.max(h) as f32) * 0.004).round().clamp(2.0, 16.0) as u32;
    let bits = f32::from_bits;
    let u = [bits(w), bits(h), clarity, dehaze, texture_amt, bits(radius), 0.0, 0.0];
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
    let dst_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.clarity_pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &src_buf), entry(1, &dst_buf), entry(2, &uni)],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.clarity_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    g.queue.submit([enc.finish()]);
    readback_vec4(g, &dst_buf, src.len())
}

/// Runs the pre-compiled SPIR-V 3D LUT trilinear interpolation compute kernel.
#[allow(dead_code)]
pub fn run_lut3d(src: &[Px], w: u32, h: u32, lut_rgb: &[Px], lut_size: u32, strength: f32) -> Option<Vec<Px>> {
    if !ENABLED.load(Ordering::Relaxed) || (w as usize) * (h as usize) != src.len() {
        return None;
    }
    let g = gpu()?;
    let _guard = g.lock.lock().ok()?;
    let src4 = px_to_vec4(src);
    let lut4 = px_to_vec4(lut_rgb);
    let bytes = (src4.len() * 16) as u64;
    if bytes > 120 * 1024 * 1024 {
        return None;
    }
    let bits = f32::from_bits;
    let u = [bits(w), bits(h), bits(lut_size), strength];
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
    let lut_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&lut4),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dst_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.lut3d_pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &src_buf), entry(1, &lut_buf), entry(2, &dst_buf), entry(3, &uni)],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.lut3d_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    g.queue.submit([enc.finish()]);
    readback_vec4(g, &dst_buf, src.len())
}

/// Runs the pre-compiled SPIR-V Chromatic Aberration (lateral + axial defringe) compute kernel.
#[allow(dead_code)]
pub fn run_chromatic_aberration(
    src: &[Px],
    w: u32,
    h: u32,
    ca_red_cyan: f32,
    ca_blue_yellow: f32,
    axial_defringe: f32,
) -> Option<Vec<Px>> {
    if !ENABLED.load(Ordering::Relaxed) || (w as usize) * (h as usize) != src.len() {
        return None;
    }
    let g = gpu()?;
    let _guard = g.lock.lock().ok()?;
    let src4 = px_to_vec4(src);
    let bytes = (src4.len() * 16) as u64;
    if bytes > 120 * 1024 * 1024 {
        return None;
    }
    let bits = f32::from_bits;
    let u = [bits(w), bits(h), ca_red_cyan, ca_blue_yellow, axial_defringe, 0.0, 0.0, 0.0];
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
    let dst_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.ca_pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &src_buf), entry(1, &dst_buf), entry(2, &uni)],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.ca_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    g.queue.submit([enc.finish()]);
    readback_vec4(g, &dst_buf, src.len())
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
    let bin_len = (scope_w as usize) * (scope_h as usize) * 3;
    let bin_bytes = (bin_len * 4) as u64;
    let zeros = vec![0u32; bin_len];

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
        contents: bytemuck::cast_slice(&zeros),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let wf_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&zeros),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let read_parade = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bin_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_wf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bin_bytes,
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
    enc.copy_buffer_to_buffer(&parade_buf, 0, &read_parade, 0, bin_bytes);
    enc.copy_buffer_to_buffer(&wf_buf, 0, &read_wf, 0, bin_bytes);
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
    let p_bins: &[u32] = bytemuck::cast_slice(&p_data);
    let w_bins: &[u32] = bytemuck::cast_slice(&w_data);

    let sw = scope_w as usize;
    let sh = scope_h as usize;
    let scale = ((h as f32) / (sh as f32).max(1.0)).max(1.0) * 3.5;

    let mut parade_rgba = vec![0u8; sw * sh * 4];
    let mut wf_rgba = vec![0u8; sw * sh * 4];
    let graticule_rows = [sh / 4, sh / 2, (sh * 3) / 4];

    for y in 0..sh {
        let is_grat = graticule_rows.contains(&y);
        let bg = if is_grat { 34u8 } else { 14u8 };
        for x in 0..sw {
            let idx = y * sw + x;
            let pr = p_bins[idx * 3] as f32;
            let pg = p_bins[idx * 3 + 1] as f32;
            let pb = p_bins[idx * 3 + 2] as f32;

            let r_val = ((1.0 - (-pr / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            let g_val = ((1.0 - (-pg / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            let b_val = ((1.0 - (-pb / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            parade_rgba[idx * 4] = r_val;
            parade_rgba[idx * 4 + 1] = g_val;
            parade_rgba[idx * 4 + 2] = b_val;
            parade_rgba[idx * 4 + 3] = 255;

            let wr = w_bins[idx * 3] as f32 / 16.0;
            let wg = w_bins[idx * 3 + 1] as f32 / 16.0;
            let wb = w_bins[idx * 3 + 2] as f32 / 16.0;
            wf_rgba[idx * 4] = ((1.0 - (-wr / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            wf_rgba[idx * 4 + 1] = ((1.0 - (-wg / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            wf_rgba[idx * 4 + 2] = ((1.0 - (-wb / scale).exp()) * 255.0).clamp(bg as f32, 255.0) as u8;
            wf_rgba[idx * 4 + 3] = 255;
        }
    }

    drop(p_data);
    drop(w_data);
    read_parade.unmap();
    read_wf.unmap();
    Some((parade_rgba, wf_rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_binaries_have_valid_magic_headers() {
        assert_eq!(spirv_manifest().len(), 5);
        for (name, bytes) in [
            ("develop_stack.spv", DEVELOP_SPV),
            ("clarity_dehaze.spv", CLARITY_DEHAZE_SPV),
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
        let cd = run_clarity_dehaze(&src, w, h, 35.0, 25.0, 20.0).expect("clarity_dehaze kernel");
        assert_eq!(cd.len(), src.len());

        let mut lut = Vec::with_capacity(8 * 8 * 8);
        for b in 0..8 {
            for g in 0..8 {
                for r in 0..8 {
                    lut.push([r as f32 / 7.0, g as f32 / 7.0, b as f32 / 7.0]);
                }
            }
        }
        let l3d = run_lut3d(&src, w, h, &lut, 8, 1.0).expect("lut3d kernel");
        assert_eq!(l3d.len(), src.len());

        let ca = run_chromatic_aberration(&src, w, h, 15.0, -10.0, 0.5).expect("ca kernel");
        assert_eq!(ca.len(), src.len());

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
    }
}
