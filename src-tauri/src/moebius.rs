//! Moebius: 0.2B lightweight image inpainting framework (Duan & Xu et al., ECCV 2026).
//! ONNX exports by SimonW (https://huggingface.co/simonw/Moebius-ONNX).
//! Three graphs: VAE encoder, student UNet (RemovalModel), and VAE decoder.
//! Learned embedding conditioning (no text encoder / tokenizer needed).

use ort::{session::Session, value::Tensor};
use std::{path::Path, sync::Mutex};

pub const SIZE: usize = 512;
pub const LAT: usize = 64;
pub const SCALING_FACTOR: f32 = 0.13025;
pub const GUIDANCE: f32 = 2.0;
pub const NUM_TRAIN_TIMESTEPS: usize = 1000;
pub const BETA_START: f32 = 0.00085;
pub const BETA_END: f32 = 0.012;
pub const NUM_STEPS: usize = 20;
pub const NOISE_OFFSET: f32 = 0.0357;
pub const HALF_IDS: usize = 10;

// DDIM scheduler (scaled-linear betas, eta=0, clip_sample=false)
struct Ddim {
    acp: Vec<f32>,
    timesteps: Vec<i64>,
}

impl Ddim {
    fn new(num_steps: usize) -> Self {
        let (b0, b1) = (BETA_START.sqrt(), BETA_END.sqrt());
        let mut acp = Vec::with_capacity(NUM_TRAIN_TIMESTEPS);
        let mut prod = 1.0f32;
        for i in 0..NUM_TRAIN_TIMESTEPS {
            let b = (b0 + (b1 - b0) * (i as f32) / (NUM_TRAIN_TIMESTEPS - 1) as f32).powi(2);
            prod *= 1.0 - b;
            acp.push(prod);
        }
        let step_ratio = NUM_TRAIN_TIMESTEPS / num_steps;
        let mut ts: Vec<i64> = (0..num_steps).map(|i| (i * step_ratio) as i64).collect();
        ts.reverse(); // [950, 900, ..., 0]
        // strength 0.99 drops the first step
        let timesteps = ts[1..].to_vec(); // 19 steps: [900, 850, ..., 0]
        Ddim { acp, timesteps }
    }

    fn step(&self, eps: &[f32], t: usize, prev_t: i64, x: &mut [f32]) {
        let ac_t = self.acp[t];
        let ac_prev = if prev_t >= 0 { self.acp[prev_t as usize] } else { 1.0 };
        let (sa, sb, pa) = (ac_t.sqrt(), (1.0 - ac_t).sqrt(), (1.0 - ac_prev).sqrt());
        for (v, &e) in x.iter_mut().zip(eps) {
            let x0 = (*v - sb * e) / sa;
            *v = ac_prev.sqrt() * x0 + pa * e;
        }
    }
}

pub struct Moebius {
    id: String,
    venc: Session,
    unet: Session,
    vdec: Session,
}

static MOEBIUS: Mutex<Option<Moebius>> = Mutex::new(None);

pub fn reset() {
    *MOEBIUS.lock().unwrap() = None;
}

fn open_session(env: &ort::environment::Environment, path: &Path, key: &str, gpu: bool) -> Result<Session, String> {
    let fail = |e: String| format!("could not load {}: {e}", path.display());
    let try_open = |use_gpu: bool| -> Result<Session, String> {
        let mut b = Session::builder(env).map_err(|e| fail(e.to_string()))?;
        if use_gpu && std::env::var_os("OPTIXEL_AI_CPU").or_else(|| std::env::var_os("OPENLIGHT_AI_CPU")).is_none() {
            b = b.with_execution_providers([crate::aimodel::accel(key)]).map_err(|e| fail(e.to_string()))?;
        } else {
            b = b.with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU).map_err(|e| fail(e.to_string()))?;
        }
        b = b.with_memory_pattern(false).map_err(|e| fail(e.to_string()))?;
        #[cfg(target_os = "macos")]
        if let Some(n) = crate::aimodel::perf_cores() {
            b = b.with_intra_threads(n).map_err(|e| fail(e.to_string()))?;
        }
        let mut b = b.with_parallel_execution(false).map_err(|e| fail(e.to_string()))?;
        b.commit_from_file(path).map_err(|e| fail(e.to_string()))
    };

    if gpu {
        match try_open(true) {
            Ok(s) => Ok(s),
            Err(e) => {
                eprintln!("[moebius] GPU EP failed ({e}), falling back to CPU for {}", path.display());
                try_open(false)
            }
        }
    } else {
        try_open(false)
    }
}

impl Moebius {
    pub fn load(env: &ort::environment::Environment, dir: &Path, id: &str, gpu: bool) -> Result<Self, String> {
        let venc = open_session(env, &dir.join("vae_encoder.onnx"), "moebius-venc", gpu)?;
        let unet = open_session(env, &dir.join("unet.onnx"), "moebius-unet", gpu)?;
        let vdec = open_session(env, &dir.join("vae_decoder.onnx"), "moebius-vdec", gpu)?;
        Ok(Moebius {
            id: id.to_string(),
            venc,
            unet,
            vdec,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

fn randn(seed: u64, count: usize) -> Vec<f32> {
    let mut rng = seed | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..count)
        .map(|_| {
            let (u1, u2) = (next().max(1e-7), next());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

/// Fill the hole of a 512 px picture using Moebius.
/// `rgb_in`: interleaved 0-1, `m_in`: 1 = hole, 0 = keep. Returns interleaved 0-1.
/// `tick(step, steps)` returns false to cancel.
pub fn fill(
    data: &Path,
    id: &str,
    gpu: bool,
    rgb_in: &[f32],
    m_in: &[f32],
    tick: &dyn Fn(usize, usize) -> bool,
) -> Result<Vec<f32>, String> {
    crate::aimodel::clear_other_sessions();
    let mut guard = MOEBIUS.lock().unwrap();
    if guard.as_ref().is_none_or(|m| m.id() != id) {
        *guard = None;
        let model_dir = data.join("models").join(id);
        *guard = Some(Moebius::load(crate::sdfill::env()?, &model_dir, id, gpu)?);
    }
    let m = guard.as_mut().unwrap();

    let area = SIZE * SIZE;
    let lat_area = LAT * LAT;

    // 1. Prepare masked image in [-1, 1] space: (1 - mask) zeros out the hole
    let mut planes = vec![0.0f32; 3 * area];
    for p in 0..area {
        let keep = if m_in[p] >= 0.5 { 0.0 } else { 1.0 };
        for c in 0..3 {
            planes[c * area + p] = (rgb_in[p * 3 + c] * 2.0 - 1.0) * keep;
        }
    }

    // 2. VAE encode the masked image -> moments -> mean * SCALING_FACTOR
    let img_tensor = Tensor::from_array(([1usize, 3, SIZE, SIZE], planes)).map_err(|e| e.to_string())?;
    let enc_out = m.venc.run(ort::inputs!["image" => img_tensor]).map_err(|e| format!("VAE encoder: {e}"))?;
    let (shape, moments) = enc_out["moments"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    if shape.len() != 4 || shape[1] < 4 || shape[2] as usize != LAT || shape[3] as usize != LAT {
        return Err(format!("unexpected VAE encoder output shape {shape:?}"));
    }
    let masked_lat: Vec<f32> = moments[0..4 * lat_area].iter().map(|&v| v * SCALING_FACTOR).collect();
    drop(enc_out);

    // 3. Downsample mask to 64x64 latent size (nearest neighbor)
    let mask64: Vec<f32> = (0..lat_area)
        .map(|k| {
            let y = k / LAT;
            let x = k % LAT;
            if m_in[(y * 8) * SIZE + (x * 8)] >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect();

    // 4. Sample initial Gaussian noise + noise offset
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let mut latents = randn(seed, 4 * lat_area);
    let off = randn(seed ^ 0x9e3779b9, 4);
    for c in 0..4 {
        for p in 0..lat_area {
            latents[c * lat_area + p] += NOISE_OFFSET * off[c];
        }
    }

    // 5. Precompute input_ids for CFG (row 0: uncond [10..19], row 1: cond [0..9])
    let mut ids_vec = Vec::with_capacity(2 * HALF_IDS);
    for i in 0..HALF_IDS {
        ids_vec.push((HALF_IDS + i) as i64);
    }
    for i in 0..HALF_IDS {
        ids_vec.push(i as i64);
    }

    // 6. DDIM sampling loop (19 steps)
    let ddim = Ddim::new(NUM_STEPS);
    let steps_count = ddim.timesteps.len();
    for (step_idx, &t) in ddim.timesteps.iter().enumerate() {
        if !tick(step_idx, steps_count) {
            return Err("cancelled".into());
        }
        let prev_t = if step_idx + 1 < steps_count { ddim.timesteps[step_idx + 1] } else { -1 };

        // 9 channels: [latents (4), mask64 (1), masked_lat (4)] duplicated for batch=2
        let mut nine = Vec::with_capacity(2 * 9 * lat_area);
        for _ in 0..2 {
            nine.extend_from_slice(&latents[0..4 * lat_area]);
            nine.extend_from_slice(&mask64);
            nine.extend_from_slice(&masked_lat[0..4 * lat_area]);
        }

        let sample = Tensor::from_array(([2usize, 9, LAT, LAT], nine)).map_err(|e| e.to_string())?;
        let tstep = Tensor::from_array(([2usize], vec![t; 2])).map_err(|e| e.to_string())?;
        let ids = Tensor::from_array(([2usize, HALF_IDS], ids_vec.clone())).map_err(|e| e.to_string())?;

        let unet_out = m
            .unet
            .run(ort::inputs!["latent" => sample, "timesteps" => tstep, "input_ids" => ids])
            .map_err(|e| format!("UNet: {e}"))?;
        let (noise_shape, noise) = unet_out["noise"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        if noise_shape.len() != 4 || noise_shape[0] != 2 || noise_shape[1] != 4 || noise_shape[2] as usize != LAT || noise_shape[3] as usize != LAT {
            return Err(format!("unexpected UNet noise output shape {noise_shape:?}"));
        }
        let n = 4 * lat_area;
        let (un, co) = noise.split_at(n);
        let guided: Vec<f32> = un.iter().zip(co).map(|(u, c)| u + GUIDANCE * (c - u)).collect();
        drop(unet_out);

        ddim.step(&guided, t as usize, prev_t, &mut latents);
    }

    // 7. VAE decode latents / SCALING_FACTOR -> image in [-1, 1] -> [0, 1]
    let scaled: Vec<f32> = latents.iter().map(|&v| v / SCALING_FACTOR).collect();
    let dec_in = Tensor::from_array(([1usize, 4, LAT, LAT], scaled)).map_err(|e| e.to_string())?;
    let dec_out = m.vdec.run(ort::inputs!["latent" => dec_in]).map_err(|e| format!("VAE decoder: {e}"))?;
    let (img_shape, dec_img) = dec_out["image"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    if img_shape.len() != 4 || img_shape[1] != 3 || img_shape[2] as usize != SIZE || img_shape[3] as usize != SIZE {
        return Err(format!("unexpected VAE decoder output shape {img_shape:?}"));
    }

    let mut rgb_out = vec![0.0f32; area * 3];
    for p in 0..area {
        for c in 0..3 {
            rgb_out[p * 3 + c] = ((dec_img[c * area + p] + 1.0) / 2.0).clamp(0.0, 1.0);
        }
    }
    Ok(rgb_out)
}
