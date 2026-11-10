//! One-click AI auto adjustment. A small learned retouch model (Image-Adaptive 3D-LUT, Zeng et al., Apache-2.0,
//! trained on photographer edits from MIT-Adobe FiveK) looks at the photo and returns a colour lookup table that
//! retouches it. The sliders are then fitted to that result, so the adjustment stays editable in the Basic and
//! Presence panels and is one step in History. The model is built in (2 MB, runs on the CPU in well under a second).

use crate::aimodel::{encode, resized};
use crate::develop::{self, Develop, Img, Opts, Presence, Px};
use ort::{session::Session, value::Tensor};

const MODEL: &[u8] = include_bytes!("../models/autoretouch.onnx");
const DIM: usize = 33;
/// long edge of the picture the sliders are fitted on
const FIT: usize = 256;

fn shrink(img: &Img, edge: usize) -> Img {
    let k = edge as f32 / img.w.max(img.h) as f32;
    if k >= 1.0 {
        return img.clone();
    }
    let (nw, nh) = (((img.w as f32 * k).round() as usize).max(1), ((img.h as f32 * k).round() as usize).max(1));
    let flat = resized(img.px.iter().flatten().copied().collect(), img.w, img.h, 3, nw, nh);
    Img { w: nw, h: nh, px: flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() }
}

/// The model's lookup table for a picture (sRGB 0-1 pixels): `[channel][blue][green][red]`, 33 steps each.
fn lut_for(srgb: &[Px], w: usize, h: usize) -> Result<Vec<f32>, String> {
    let flat = resized(srgb.iter().flatten().copied().collect(), w, h, 3, 256, 256);
    let mut planes = vec![0f32; 3 * 256 * 256];
    for (i, p) in flat.chunks_exact(3).enumerate() {
        for c in 0..3 {
            planes[c * 256 * 256 + i] = p[c].clamp(0.0, 1.0);
        }
    }
    let fail = |e: String| format!("AI auto adjust: {e}");
    // the 3D-LUT model is tiny and CoreML cannot compile it (error code -1), so pin it to the CPU provider
    let mut session = Session::builder(crate::sdfill::env()?)
        .map_err(|e| fail(e.to_string()))?
        .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
        .map_err(|e| fail(e.to_string()))?
        .commit_from_memory(MODEL)
        .map_err(|e| fail(e.to_string()))?;
    let input = Tensor::from_array(([1usize, 3, 256, 256], planes)).map_err(|e| fail(e.to_string()))?;
    let out = session.run(ort::inputs!["image" => input]).map_err(|e| fail(e.to_string()))?;
    let (shape, lut) = out["lut"].try_extract_tensor::<f32>().map_err(|e| fail(e.to_string()))?;
    if lut.len() != 3 * DIM * DIM * DIM {
        return Err(fail(format!("unexpected model output {shape:?}")));
    }
    Ok(lut.to_vec())
}

/// Trilinear lookup, as the model was trained.
fn apply_lut(lut: &[f32], px: &[Px]) -> Vec<Px> {
    let n = (DIM - 1) as f32;
    let at = |c: usize, r: usize, g: usize, b: usize| lut[((c * DIM + b) * DIM + g) * DIM + r];
    px.iter()
        .map(|p| {
            let q = p.map(|v| v.clamp(0.0, 1.0) * n);
            let i = q.map(|v| (v.floor() as usize).min(DIM - 2));
            let f = [q[0] - i[0] as f32, q[1] - i[1] as f32, q[2] - i[2] as f32];
            std::array::from_fn(|c| {
                let mut s = 0.0;
                for (dr, dg, db) in (0..8).map(|k| (k & 1, (k >> 1) & 1, (k >> 2) & 1)) {
                    let w = [dr, dg, db].iter().enumerate().map(|(a, &d)| if d == 1 { f[a] } else { 1.0 - f[a] }).product::<f32>();
                    s += w * at(c, i[0] + dr, i[1] + dg, i[2] + db);
                }
                s
            })
        })
        .collect()
}

/// The sliders the fit moves: exposure, contrast, highlights, shadows, whites, blacks, ln temperature, tint, vibrance, saturation.
const N: usize = 10;
const LO: [f32; N] = [-3.0, -100.0, -100.0, -100.0, -100.0, -100.0, 7.6, -100.0, -100.0, -60.0];
const HI: [f32; N] = [3.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10.8, 100.0, 100.0, 60.0];
const STEP: [f32; N] = [0.4, 15.0, 15.0, 15.0, 15.0, 15.0, 0.08, 8.0, 15.0, 15.0];

fn params(d: &Develop) -> [f32; N] {
    let (t, w, p) = (&d.basic_tone, &d.white_balance, &d.presence);
    [t.exposure, t.contrast, t.highlights, t.shadows, t.whites, t.blacks, w.temperature_k.ln(), w.tint, p.vibrance, p.saturation]
}

fn with_params(d: &Develop, q: &[f32; N]) -> Develop {
    let mut d = d.clone();
    (d.basic_tone.exposure, d.basic_tone.contrast, d.basic_tone.highlights, d.basic_tone.shadows, d.basic_tone.whites, d.basic_tone.blacks) = (q[0], q[1], q[2], q[3], q[4], q[5]);
    (d.white_balance.temperature_k, d.white_balance.tint) = (q[6].exp(), q[7]);
    (d.presence.vibrance, d.presence.saturation) = (q[8], q[9]);
    d
}

/// The photo retouched by the model, as sliders: `cur` with white balance, Basic tone, vibrance and saturation
/// fitted. `amount` (0-1) is how far from the plain Auto result to go; `profile` adds that look on top.
pub fn adjust(img: &Img, cur: &Develop, profile: Option<&str>, strength: f32, amount: f32) -> Result<Develop, String> {
    // start from the engine's own Auto (white balance and tone), which the model then refines
    let white_balance = develop::auto_wb(img);
    let basic_tone = develop::auto_tone(img, &white_balance);
    let base = Develop { basic_tone, white_balance, presence: Presence { vibrance: 0.0, saturation: 0.0, ..cur.presence.clone() }, ..cur.clone() };
    let small = shrink(img, FIT);
    let opts = Opts::default();
    let shown = |d: &Develop| -> Vec<Px> { develop::render(&small, d, &opts).px.into_iter().map(encode).collect() };
    let start = shown(&base);
    let target = apply_lut(&lut_for(&start, small.w, small.h)?, &start);

    let p0 = params(&base);
    let loss = |q: &[f32; N]| -> f32 {
        let out = shown(&with_params(&base, q));
        let err = out.iter().zip(&target).map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>()).sum::<f32>() / out.len() as f32;
        // stay close to Auto unless the picture clearly asks for more
        err + (0..N).map(|i| ((q[i] - p0[i]) / (HI[i] - LO[i])).powi(2)).sum::<f32>() * 2e-4
    };
    let (mut p, mut step) = (p0, STEP);
    let mut best = loss(&p);
    for _ in 0..7 {
        for i in 0..N {
            for dir in [1.0f32, -1.0] {
                let mut q = p;
                q[i] = (p[i] + dir * step[i]).clamp(LO[i], HI[i]);
                let l = loss(&q);
                if l < best - 1e-7 {
                    (best, p) = (l, q);
                    break;
                }
            }
        }
        step.iter_mut().for_each(|s| *s *= 0.55);
    }
    // how far to go, then tidy numbers
    let mut q: [f32; N] = std::array::from_fn(|i| p0[i] + amount.clamp(0.0, 1.0) * (p[i] - p0[i]));
    q[0] = (q[0] * 20.0).round() / 20.0;
    (1..6).chain(7..10).for_each(|i| q[i] = q[i].round());
    q[6] = ((q[6].exp() / 10.0).round() * 10.0).ln();
    let fitted = with_params(&base, &q);
    match profile {
        None => Ok(fitted),
        Some(name) => {
            let mut d = develop::look_on_strength(&fitted, name, strength)?;
            // the look starts the presence panel over: keep the model's vibrance and saturation on top
            d.presence.vibrance = (d.presence.vibrance + fitted.presence.vibrance).clamp(-100.0, 100.0);
            d.presence.saturation = (d.presence.saturation + fitted.presence.saturation).clamp(-100.0, 100.0);
            Ok(d)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_runs_on_cpu_without_coreml_error() {
        let px = vec![[0.2, 0.4, 0.6]; 32 * 32];
        let lut = lut_for(&px, 32, 32).expect("3D-LUT inference should succeed on CPU");
        assert_eq!(lut.len(), 3 * DIM * DIM * DIM);
    }

    /// A photo through the model: OPTIXEL_TEST_RAW=<file>, OPTIXEL_TEST_OUT=<folder> for before/after PNGs.
    #[test]
    fn adjusts_a_real_photo() {
        let Some(p) = std::env::var_os("OPTIXEL_TEST_RAW") else { return };
        let img = develop::load(std::path::Path::new(&p), Some(1600)).unwrap();
        let t = std::time::Instant::now();
        let d = adjust(&img, &Develop::default(), None, 1.0, 1.0).unwrap();
        eprintln!("RETOUCH {} ms; tone {:?} wb {:?} presence {:?}", t.elapsed().as_millis(), d.basic_tone, d.white_balance, d.presence);
        let styled = adjust(&img, &Develop::default(), Some("vivid"), 1.0, 1.0).unwrap();
        eprintln!("RETOUCH vivid: tone {:?} presence {:?}", styled.basic_tone, styled.presence);
        if let Some(out) = std::env::var_os("OPTIXEL_TEST_OUT") {
            let small = shrink(&img, 900);
            let wb = develop::auto_wb(&img);
            let auto = Develop { basic_tone: develop::auto_tone(&img, &wb), white_balance: wb, ..Default::default() };
            for (name, dev) in [("auto", &auto), ("ai", &d), ("vivid", &styled)] {
                let r = develop::render(&small, dev, &Opts::default());
                let buf: Vec<u8> = r.px.iter().flat_map(|p| encode(*p).map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)).collect();
                image::RgbImage::from_raw(r.w as u32, r.h as u32, buf).unwrap().save(std::path::Path::new(&out).join(format!("{name}.png"))).unwrap();
            }
        }
    }
}
