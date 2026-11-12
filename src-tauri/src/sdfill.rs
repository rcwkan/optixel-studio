//! Stable Diffusion inpainting (generative fill) on ONNX Runtime: CLIP tokenizer, text encoder, VAE, the
//! 9-channel inpainting UNet and a DDIM sampler, following diffusers' `OnnxStableDiffusionInpaintPipeline`
//! for the SD 1.5 inpainting export (fp16, 512 px, classifier-free guidance).
//! ponytail: SD 1.5 only (512 px, 4-channel latents); SD 2 / SDXL need a different text encoder and scale.

use half::f16;
use ort::{environment::Environment, session::Session, value::Tensor};
use regex::Regex;
use std::{collections::HashMap, fs, path::Path, sync::OnceLock};

/// Image size the model works at, and its latent size.
pub const SIZE: usize = 512;
const LAT: usize = SIZE / 8;
const SCALE: f32 = 0.18215;
const GUIDANCE: f32 = 6.0;
const CTX: usize = 77;

// ---------------------------------------------------------------------------
// CLIP tokenizer
// ---------------------------------------------------------------------------

pub(crate) struct Tokenizer {
    vocab: HashMap<String, i32>,
    ranks: HashMap<(String, String), usize>,
    bytes: Vec<char>,
    re: Regex,
}

impl Tokenizer {
    fn load(dir: &Path) -> Result<Self, String> {
        Self::load_files(&dir.join("tokenizer/vocab.json"), &dir.join("tokenizer/merges.txt"))
    }

    /// CLIP's BPE from its `vocab.json` and `merges.txt`.
    pub(crate) fn load_files(vocab: &Path, merges: &Path) -> Result<Self, String> {
        let vocab: HashMap<String, i32> = serde_json::from_str(&fs::read_to_string(vocab).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let merges = fs::read_to_string(merges).map_err(|e| e.to_string())?;
        let ranks = merges
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once(' '))
            .enumerate()
            .map(|(i, (a, b))| ((a.to_string(), b.to_string()), i))
            .collect();
        // GPT-2's byte -> printable character map
        let mut keep: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut chars = keep.clone();
        let mut extra = 0;
        for b in 0..256u32 {
            if !keep.contains(&b) {
                keep.push(b);
                chars.push(256 + extra);
                extra += 1;
            }
        }
        let mut bytes = vec!['\0'; 256];
        for (b, c) in keep.iter().zip(&chars) {
            bytes[*b as usize] = char::from_u32(*c).unwrap();
        }
        let re = Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+").map_err(|e| e.to_string())?;
        Ok(Tokenizer { vocab, ranks, bytes, re })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while word.len() > 1 {
            let best = (0..word.len() - 1).filter_map(|i| self.ranks.get(&(word[i].clone(), word[i + 1].clone())).map(|r| (*r, i))).min();
            let Some((_, i)) = best else { break };
            let (a, b) = (word[i].clone(), word[i + 1].clone());
            let mut out = Vec::with_capacity(word.len());
            let mut k = 0;
            while k < word.len() {
                if k + 1 < word.len() && word[k] == a && word[k + 1] == b {
                    out.push(format!("{a}{b}"));
                    k += 2;
                } else {
                    out.push(word[k].clone());
                    k += 1;
                }
            }
            word = out;
        }
        word
    }

    /// 77 token ids: start, the text (cut to fit), end, then padding with the end token.
    fn encode(&self, text: &str) -> Vec<i32> {
        let end = self.vocab["<|endoftext|>"];
        self.ids(text, CTX, end as i64).into_iter().map(|v| v as i32).collect()
    }

    /// `len` token ids: start, the text (cut to fit), end, then `pad`.
    pub(crate) fn ids(&self, text: &str, len: usize, pad: i64) -> Vec<i64> {
        let (start, end) = (self.vocab["<|startoftext|>"] as i64, self.vocab["<|endoftext|>"] as i64);
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![start];
        for m in self.re.find_iter(&clean) {
            let t: String = m.as_str().bytes().map(|b| self.bytes[b as usize]).collect();
            ids.extend(self.bpe(&t).iter().filter_map(|p| self.vocab.get(p)).map(|&v| v as i64));
        }
        ids.truncate(len.max(2) - 1);
        ids.push(end);
        ids.resize(len.max(2), pad);
        ids
    }
}

// ---------------------------------------------------------------------------
// DDIM sampler (scaled-linear betas, "leading" spacing, steps offset 1, epsilon prediction)
// ---------------------------------------------------------------------------

struct Ddim {
    acp: Vec<f32>,
    ratio: usize,
}

impl Ddim {
    fn new() -> Self {
        let n = 1000;
        let (b0, b1) = (0.00085f32.sqrt(), 0.012f32.sqrt());
        let mut acp = Vec::with_capacity(n);
        let mut prod = 1.0f32;
        for i in 0..n {
            let b = (b0 + (b1 - b0) * i as f32 / (n - 1) as f32).powi(2);
            prod *= 1.0 - b;
            acp.push(prod);
        }
        Ddim { acp, ratio: 0 }
    }
    fn timesteps(&mut self, steps: usize) -> Vec<usize> {
        self.ratio = 1000 / steps;
        (0..steps).rev().map(|i| i * self.ratio + 1).collect()
    }
    /// One deterministic step (eta = 0) from timestep `t`.
    fn step(&self, eps: &[f32], t: usize, x: &mut [f32]) {
        let prev = t as isize - self.ratio as isize;
        let a_t = self.acp[t];
        let a_prev = if prev >= 0 { self.acp[prev as usize] } else { self.acp[0] };
        let (sa, sb, pa) = (a_t.sqrt(), (1.0 - a_t).sqrt(), (1.0 - a_prev).sqrt());
        for (v, &e) in x.iter_mut().zip(eps) {
            let x0 = (*v - sb * e) / sa;
            *v = a_prev.sqrt() * x0 + pa * e;
        }
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

pub struct Sd {
    id: String,
    tok: Tokenizer,
    text: Session,
    venc: Session,
    vdec: Session,
    unet: Session,
}

fn open(env: &Environment, path: &Path, fixed: &[(&str, i64)]) -> Result<Session, String> {
    let fail = |e: String| format!("could not load {}: {e}", path.display());
    let b = Session::builder(env).map_err(|e| fail(e.to_string()))?;
    let b = b.with_execution_providers([crate::aimodel::accel(&format!("{}-{}", path.file_stem().and_then(|f| f.to_str()).unwrap_or("sd"), fixed.iter().map(|(_, v)| v.to_string()).collect::<Vec<_>>().join("x")))]).map_err(|e| fail(e.to_string()))?;
    let b = b.with_memory_pattern(false).map_err(|e| fail(e.to_string()))?;
    let mut b = b.with_parallel_execution(false).map_err(|e| fail(e.to_string()))?;
    // fixed input shapes keep DirectML from crashing on dynamic ones
    for (name, size) in fixed {
        b = b.with_dimension_override(*name, *size).map_err(|e| fail(e.to_string()))?;
    }
    b.commit_from_file(path).map_err(|e| fail(e.to_string()))
}

impl Sd {
    pub fn load(env: &Environment, dir: &Path, id: &str) -> Result<Sd, String> {
        let l = LAT as i64;
        Ok(Sd {
            id: id.to_string(),
            tok: Tokenizer::load(dir)?,
            text: open(env, &dir.join("text_encoder/model.onnx"), &[("batch", 1), ("sequence", CTX as i64)])?,
            venc: open(env, &dir.join("vae_encoder/model.onnx"), &[("batch", 1), ("channels", 3), ("height", SIZE as i64), ("width", SIZE as i64)])?,
            vdec: open(env, &dir.join("vae_decoder/model.onnx"), &[("batch", 1), ("channels", 4), ("height", l), ("width", l)])?,
            unet: open(env, &dir.join("unet/model.onnx"), &[("batch", 2), ("channels", 9), ("height", l), ("width", l), ("sequence", CTX as i64)])?,
        })
    }
    pub fn id(&self) -> &str {
        &self.id
    }

    fn embed(&mut self, prompt: &str) -> Result<Vec<f16>, String> {
        let ids = self.tok.encode(prompt);
        let t = Tensor::from_array(([1usize, CTX], ids)).map_err(|e| e.to_string())?;
        let out = self.text.run(ort::inputs!["input_ids" => t]).map_err(|e| format!("text encoder: {e}"))?;
        let (_, d) = out["last_hidden_state"].try_extract_tensor::<f16>().map_err(|e| e.to_string())?;
        Ok(d.to_vec())
    }

    /// Fill the hole of a 512 px picture. `rgb`: interleaved 0-1, `mask`: 1 = hole. Returns interleaved 0-1.
    /// `tick(step, steps)` returns false to cancel.
    pub fn fill(&mut self, rgb: &[f32], mask: &[f32], prompt: &str, negative: &str, steps: usize, seed: u64, tick: &dyn Fn(usize, usize) -> bool) -> Result<Vec<f32>, String> {
        let area = SIZE * SIZE;
        let (cond, uncond) = (self.embed(prompt)?, self.embed(negative)?);
        let ctx: Vec<f16> = uncond.iter().chain(&cond).copied().collect();

        // the picture with the hole blanked, as latents
        let mut planes = vec![f16::ZERO; 3 * area];
        for i in 0..area {
            let keep = if mask[i] >= 0.5 { 0.0 } else { 1.0 };
            for c in 0..3 {
                planes[c * area + i] = f16::from_f32((rgb[i * 3 + c] * 2.0 - 1.0) * keep);
            }
        }
        let t = Tensor::from_array(([1usize, 3, SIZE, SIZE], planes)).map_err(|e| e.to_string())?;
        let out = self.venc.run(ort::inputs!["sample" => t]).map_err(|e| format!("VAE encoder: {e}"))?;
        let (shape, lat) = out["latent_sample"].try_extract_tensor::<f16>().map_err(|e| e.to_string())?;
        if shape.len() != 4 || shape[1] != 4 || shape[2] as usize != LAT {
            return Err(format!("unexpected VAE encoder output {shape:?}"));
        }
        let masked: Vec<f32> = lat.iter().map(|v| v.to_f32() * SCALE).collect();
        drop(out);
        let lat_area = LAT * LAT;
        let mask_lat: Vec<f32> = (0..lat_area).map(|k| mask[(k / LAT * 8) * SIZE + (k % LAT) * 8]).collect();

        // start from noise
        let mut rng = seed | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut latents: Vec<f32> = (0..4 * lat_area)
            .map(|_| {
                let (u1, u2) = (next().max(1e-7), next());
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let mut sched = Ddim::new();
        let ts = sched.timesteps(steps);
        for (k, &t) in ts.iter().enumerate() {
            if !tick(k, steps) {
                return Err("cancelled".into());
            }
            // [noisy latents, mask, masked picture latents], for the unconditional and conditional halves
            let mut input = Vec::with_capacity(2 * 9 * lat_area);
            for _ in 0..2 {
                input.extend(latents.iter().map(|&v| f16::from_f32(v)));
                input.extend(mask_lat.iter().map(|&v| f16::from_f32(v)));
                input.extend(masked.iter().map(|&v| f16::from_f32(v)));
            }
            let sample = Tensor::from_array(([2usize, 9, LAT, LAT], input)).map_err(|e| e.to_string())?;
            let tstep = Tensor::from_array(([2usize], vec![f16::from_f32(t as f32); 2])).map_err(|e| e.to_string())?;
            let hidden = Tensor::from_array(([2usize, CTX, 768], ctx.clone())).map_err(|e| e.to_string())?;
            let out = self
                .unet
                .run(ort::inputs!["sample" => sample, "timestep" => tstep, "encoder_hidden_states" => hidden])
                .map_err(|e| format!("UNet: {e}"))?;
            let (_, eps) = out["out_sample"].try_extract_tensor::<f16>().map_err(|e| e.to_string())?;
            let (un, co) = eps.split_at(4 * lat_area);
            let guided: Vec<f32> = un.iter().zip(co).map(|(u, c)| u.to_f32() + GUIDANCE * (c.to_f32() - u.to_f32())).collect();
            drop(out);
            sched.step(&guided, t, &mut latents);
        }

        let dec: Vec<f16> = latents.iter().map(|&v| f16::from_f32(v / SCALE)).collect();
        let t = Tensor::from_array(([1usize, 4, LAT, LAT], dec)).map_err(|e| e.to_string())?;
        let out = self.vdec.run(ort::inputs!["latent_sample" => t]).map_err(|e| format!("VAE decoder: {e}"))?;
        let (_, img) = out["sample"].try_extract_tensor::<f16>().map_err(|e| e.to_string())?;
        let mut rgb_out = vec![0f32; area * 3];
        for i in 0..area {
            for c in 0..3 {
                rgb_out[i * 3 + c] = (img[c * area + i].to_f32() / 2.0 + 0.5).clamp(0.0, 1.0);
            }
        }
        Ok(rgb_out)
    }
}

pub fn env() -> Result<&'static Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("optixel-sd").build().map_err(|e| e.to_string())).as_ref().map_err(|e| e.clone())
}
