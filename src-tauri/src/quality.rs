//! AI photo-quality and aesthetic scoring via downloadable ONNX models:
//! - Technical quality (`quality`): `lumina-v2-int8` (MIT, commercial use allowed) or `topiq-nr-koniq` (CC BY-NC-SA 4.0)
//! - Aesthetics (`aesthetic`): `laion-aesthetic-v2` (MIT + Apache-2.0, commercial use allowed) or `topiq-iaa-ava` (CC BY-NC-SA 4.0)

use crate::{aimodel, vision};
use image::{ImageBuffer, Rgb, imageops::FilterType};
use ort::{session::Session, value::Tensor};
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

pub const LUMINA: &str = "lumina-v2-int8";
pub const LUMINA_FILE: &str = "lumina_v2.int8.onnx";
pub const TOPIQ_TECH: &str = "topiq-nr-koniq";
pub const TOPIQ_TECH_FILE: &str = "model.onnx";

pub const LAION_AES: &str = "laion-aesthetic-v2";
pub const LAION_VISION_FILE: &str = "onnx/vision_model_uint8.onnx";
pub const LAION_MLP_FILE: &str = "aesthetic_score_mlp.onnx";
pub const TOPIQ_AES: &str = "topiq-iaa-ava";
pub const TOPIQ_AES_FILE: &str = "topiq_iaa_res50.onnx";

const TOPIQ_SIDE: usize = 384;
const CLIP_SIDE: usize = 224;
const MAX_CLIP_RESIZE: u32 = 4096;
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub type ScorePair = (Option<Result<f32, String>>, Option<Result<f32, String>>);

type Slot = Mutex<Option<(PathBuf, Session)>>;

static TECH: Slot = Mutex::new(None);
static AES: Slot = Mutex::new(None);
static CLIP_VIS: Slot = Mutex::new(None);

/// Drop cached ONNX sessions (for example when a model is removed).
pub fn reset() {
    for slot in [&TECH, &AES, &CLIP_VIS] {
        if let Ok(mut g) = slot.lock() {
            *g = None;
        }
    }
}

fn is_installed(data: &Path, id: &str) -> bool {
    match id {
        LUMINA => aimodel::model_file(data, LUMINA, LUMINA_FILE).is_some(),
        TOPIQ_TECH => aimodel::model_file(data, TOPIQ_TECH, TOPIQ_TECH_FILE).is_some(),
        LAION_AES => {
            aimodel::model_file(data, LAION_AES, LAION_VISION_FILE).is_some()
                && aimodel::model_file(data, LAION_AES, LAION_MLP_FILE).is_some()
        }
        TOPIQ_AES => aimodel::model_file(data, TOPIQ_AES, TOPIQ_AES_FILE).is_some(),
        _ => false,
    }
}

/// Pick the model to use for `kind` (`"quality"` or `"aesthetic"`): `chosen` if installed, else best installed.
pub fn pick(data: &Path, kind: &str, chosen: &str) -> Option<&'static str> {
    let order: &[&'static str] = if kind == "aesthetic" {
        &[LAION_AES, TOPIQ_AES]
    } else {
        &[LUMINA, TOPIQ_TECH]
    };
    if let Some(&m) = order.iter().find(|&&id| id == chosen && is_installed(data, id)) {
        return Some(m);
    }
    order.iter().copied().find(|&id| is_installed(data, id))
}

/// Active `(technical_model_id, aesthetic_model_id)` given the user's chosen IDs in Settings.
pub fn active(data: &Path, chosen_q: &str, chosen_a: &str) -> (Option<&'static str>, Option<&'static str>) {
    (pick(data, "quality", chosen_q), pick(data, "aesthetic", chosen_a))
}

fn to_image_buf(rgb: &vision::Rgb) -> Result<ImageBuffer<Rgb<f32>, Vec<f32>>, String> {
    if rgb.w == 0 || rgb.h == 0 {
        return Err("image has zero dimensions".into());
    }
    let expected = rgb.w.checked_mul(rgb.h).ok_or("image dimensions overflow")?;
    if rgb.px.len() != expected {
        return Err(format!("pixel buffer length {} does not match {}x{}", rgb.px.len(), rgb.w, rgb.h));
    }
    let (w, h) = (u32::try_from(rgb.w).map_err(|_| "image width overflow")?, u32::try_from(rgb.h).map_err(|_| "image height overflow")?);
    let mut flat = Vec::with_capacity(expected.saturating_mul(3));
    for p in &rgb.px {
        flat.push(if p[0].is_finite() { p[0].clamp(0.0, 1.0) } else { 0.0 });
        flat.push(if p[1].is_finite() { p[1].clamp(0.0, 1.0) } else { 0.0 });
        flat.push(if p[2].is_finite() { p[2].clamp(0.0, 1.0) } else { 0.0 });
    }
    ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(w, h, flat).ok_or_else(|| "failed to build RGB image buffer".to_string())
}

fn chw_planes(img: &ImageBuffer<Rgb<f32>, Vec<f32>>, scale: f32, norm: Option<([f32; 3], [f32; 3])>) -> Vec<f32> {
    let area = (img.width() * img.height()) as usize;
    let mut out = vec![0.0f32; 3 * area];
    let (r_plane, rest) = out.split_at_mut(area);
    let (g_plane, b_plane) = rest.split_at_mut(area);
    match norm {
        Some((mean, std)) => {
            for (i, p) in img.pixels().enumerate() {
                r_plane[i] = (p[0] * scale - mean[0]) / std[0];
                g_plane[i] = (p[1] * scale - mean[1]) / std[1];
                b_plane[i] = (p[2] * scale - mean[2]) / std[2];
            }
        }
        None => {
            for (i, p) in img.pixels().enumerate() {
                r_plane[i] = p[0] * scale;
                g_plane[i] = p[1] * scale;
                b_plane[i] = p[2] * scale;
            }
        }
    }
    out
}

/// Squash `rgb` (sRGB `0..1`) to `[1, 3, 384, 384]` CHW without ImageNet mean/std (normalisation is inside TOPIQ).
fn input_topiq(rgb: &vision::Rgb) -> Result<Vec<f32>, String> {
    let buf = to_image_buf(rgb)?;
    let resized = image::imageops::resize(&buf, TOPIQ_SIDE as u32, TOPIQ_SIDE as u32, FilterType::Triangle);
    Ok(chw_planes(&resized, 1.0, None))
}

/// Resize `rgb` at native aspect ratio (long side <= 512, short side <= 384, multiples of 32) in `[0, 255]` CHW for Lumina v2.
fn input_lumina(rgb: &vision::Rgb) -> Result<(usize, usize, Vec<f32>), String> {
    let buf = to_image_buf(rgb)?;
    let w = rgb.w as f32;
    let h = rgb.h as f32;
    let scale = (512.0 / w.max(h)).min(384.0 / w.min(h)).min(1.0);
    let nw = (((w * scale).round() as usize) / 32 * 32).max(32);
    let nh = (((h * scale).round() as usize) / 32 * 32).max(32);
    let resized = image::imageops::resize(&buf, nw as u32, nh as u32, FilterType::Triangle);
    Ok((nw, nh, chw_planes(&resized, 255.0, None)))
}

/// Resize shortest side to 224, center-crop `224x224`, and apply CLIP mean/std for ViT-L/14.
fn input_clip(rgb: &vision::Rgb) -> Result<Vec<f32>, String> {
    let buf = to_image_buf(rgb)?;
    let w = rgb.w as f32;
    let h = rgb.h as f32;
    let s = CLIP_SIDE as f32 / w.min(h);
    let rw = ((w * s).round().max(CLIP_SIDE as f32) as u32).min(MAX_CLIP_RESIZE);
    let rh = ((h * s).round().max(CLIP_SIDE as f32) as u32).min(MAX_CLIP_RESIZE);
    let mut resized = image::imageops::resize(&buf, rw, rh, FilterType::Triangle);
    let left = (rw - CLIP_SIDE as u32) / 2;
    let top = (rh - CLIP_SIDE as u32) / 2;
    let cropped = image::imageops::crop(&mut resized, left, top, CLIP_SIDE as u32, CLIP_SIDE as u32).to_image();
    Ok(chw_planes(&cropped, 1.0, Some((CLIP_MEAN, CLIP_STD))))
}

fn run_shape(slot: &Slot, path: &Path, shape: Vec<usize>, x: Vec<f32>) -> Result<Vec<f32>, String> {
    crate::auto::check_not_cancelled()?;
    let mut g = slot.lock().map_err(|_| "AI quality session lock poisoned".to_string())?;
    if g.as_ref().is_none_or(|(p, _)| p != path) {
        *g = Some((path.to_path_buf(), aimodel::open_session(path, None, 0, false)?));
    }
    let s = &mut g.as_mut().expect("session just opened").1;
    let input_tensor = Tensor::from_array((shape, x)).map_err(|e| format!("quality input: {e}"))?;
    let name = s.inputs().first().ok_or("the quality model has no inputs")?.name().to_string();
    let cancel_run = crate::auto::CancellableRun::new()?;
    let out = s
        .run_with_options(ort::inputs![name.as_str() => input_tensor], cancel_run.opts())
        .map_err(|e| crate::auto::map_run_err(e, "quality inference failed"))?;
    crate::auto::check_not_cancelled()?;
    if out.len() == 0 {
        return Err("the quality model returned no outputs".into());
    }
    let (_, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("quality output: {e}"))?;
    Ok(data.to_vec())
}

/// Map TOPIQ NR KonIQ-10k raw output (`~0..1`) to a `0..100` score.
pub fn technical_score(v: f32) -> f32 {
    if v.is_finite() { (v * 100.0).clamp(0.0, 100.0) } else { 0.0 }
}

/// Map Lumina v2 MOS output (`0..100`) to a clamped `0..100` score.
pub fn lumina_score(mos: f32) -> f32 {
    if mos.is_finite() { mos.clamp(0.0, 100.0) } else { 0.0 }
}

/// Map TOPIQ IAA AVA 10-bin score distribution (`1..10`) to a `0..100` score.
pub fn aesthetic_score(probs: &[f32]) -> Option<f32> {
    if probs.len() < 10 || probs[..10].iter().any(|p| !p.is_finite()) {
        return None;
    }
    let sum: f32 = probs[..10].iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    let mean: f32 = probs[..10]
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as f32 + 1.0) * (p / sum))
        .sum();
    Some(((mean - 3.0) / 4.5 * 100.0).clamp(0.0, 100.0))
}

/// Map LAION Aesthetic Predictor v2 raw score (`1..10` scale) to a `0..100` score.
pub fn laion_aesthetic_score(raw: f32) -> f32 {
    if raw.is_finite() { ((raw - 3.0) / 4.5 * 100.0).clamp(0.0, 100.0) } else { 0.0 }
}

fn run_laion(data: &Path, rgb: &vision::Rgb) -> Result<f32, String> {
    let vis_path = aimodel::model_file(data, LAION_AES, LAION_VISION_FILE)
        .ok_or("CLIP ViT-L/14 vision model missing")?;
    let mlp_path = aimodel::model_file(data, LAION_AES, LAION_MLP_FILE)
        .ok_or("LAION aesthetic MLP missing")?;
    let x = input_clip(rgb)?;
    let mut emb = run_shape(&CLIP_VIS, &vis_path, vec![1, 3, CLIP_SIDE, CLIP_SIDE], x)?;
    if emb.len() != 768 || emb.iter().any(|v| !v.is_finite()) {
        return Err(format!("expected finite 768-dim CLIP embedding, got len {}", emb.len()));
    }
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
    for v in &mut emb {
        *v /= norm;
    }
    let out = run_shape(&AES, &mlp_path, vec![1, 768], emb)?;
    out.first()
        .copied()
        .filter(|v| v.is_finite())
        .map(laion_aesthetic_score)
        .ok_or_else(|| "LAION aesthetic model returned empty or non-finite output".to_string())
}

/// Run the selected quality and aesthetic models on `rgb`; returns `(technical, aesthetic)`.
pub fn score(
    data: &Path,
    rgb: &vision::Rgb,
    tech_id: Option<&str>,
    aes_id: Option<&str>,
) -> ScorePair {
    let need_topiq = tech_id == Some(TOPIQ_TECH) || aes_id == Some(TOPIQ_AES);
    let topiq_x = need_topiq.then(|| input_topiq(rgb));

    let tech = tech_id.and_then(|id| match id {
        LUMINA => aimodel::model_file(data, LUMINA, LUMINA_FILE).map(|p| {
            let (nw, nh, x) = input_lumina(rgb)?;
            let v = run_shape(&TECH, &p, vec![1, 3, nh, nw], x)?;
            v.first()
                .copied()
                .filter(|s| s.is_finite())
                .map(lumina_score)
                .ok_or_else(|| "Lumina quality model returned empty or non-finite output".to_string())
        }),
        TOPIQ_TECH => aimodel::model_file(data, TOPIQ_TECH, TOPIQ_TECH_FILE).map(|p| {
            let x = topiq_x.as_ref().expect("topiq input built").clone()?;
            let v = run_shape(&TECH, &p, vec![1, 3, TOPIQ_SIDE, TOPIQ_SIDE], x)?;
            v.first()
                .copied()
                .filter(|s| s.is_finite())
                .map(technical_score)
                .ok_or_else(|| "technical quality model returned empty or non-finite output".to_string())
        }),
        _ => None,
    });

    let aes = aes_id.and_then(|id| match id {
        LAION_AES => is_installed(data, LAION_AES).then(|| run_laion(data, rgb)),
        TOPIQ_AES => aimodel::model_file(data, TOPIQ_AES, TOPIQ_AES_FILE).map(|p| {
            let x = topiq_x.expect("topiq input built")?;
            let v = run_shape(&AES, &p, vec![1, 3, TOPIQ_SIDE, TOPIQ_SIDE], x)?;
            aesthetic_score(&v)
                .ok_or_else(|| format!("aesthetic model returned unexpected output length {}", v.len()))
        }),
        _ => None,
    });

    (tech, aes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_mappings_clamp_and_weight_bins() {
        let mut p6 = [0.0f32; 10];
        p6[5] = 1.0; // bin 6 -> (6 - 3) / 4.5 * 100 = 66.6667
        let s6 = aesthetic_score(&p6).unwrap();
        assert!((s6 - 66.6667).abs() < 0.01, "got {s6}");

        let mut p1 = [0.0f32; 10];
        p1[0] = 1.0;
        assert_eq!(aesthetic_score(&p1), Some(0.0));

        let mut p10 = [0.0f32; 10];
        p10[9] = 1.0;
        assert_eq!(aesthetic_score(&p10), Some(100.0));

        assert_eq!(aesthetic_score(&[1.0; 5]), None);
        assert!((technical_score(0.7) - 70.0).abs() < 1e-4);
        assert_eq!(technical_score(1.3), 100.0);
        assert_eq!(technical_score(-0.2), 0.0);

        assert!((lumina_score(74.5) - 74.5).abs() < 1e-4);
        assert_eq!(lumina_score(120.0), 100.0);
        assert_eq!(lumina_score(-5.0), 0.0);

        assert!((laion_aesthetic_score(6.0) - 66.6667).abs() < 0.01);
        assert_eq!(laion_aesthetic_score(2.0), 0.0);
        assert_eq!(laion_aesthetic_score(8.5), 100.0);

        // Non-finite values must never propagate NaN/Inf.
        assert_eq!(technical_score(f32::NAN), 0.0);
        assert_eq!(lumina_score(f32::INFINITY), 0.0);
        assert_eq!(laion_aesthetic_score(f32::NAN), 0.0);
        let mut bad_probs = [0.1f32; 10];
        bad_probs[0] = f32::NAN;
        assert_eq!(aesthetic_score(&bad_probs), None);
    }

    #[test]
    fn input_layouts_match_model_expectations() {
        let rgb = vision::Rgb {
            w: 2,
            h: 1,
            px: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let x = input_topiq(&rgb).unwrap();
        let area = TOPIQ_SIDE * TOPIQ_SIDE;
        assert_eq!(x.len(), 3 * area);
        assert!(x[0] > 0.9 && x[2 * area] < 0.1, "left R={} B={}", x[0], x[2 * area]);
        assert!(x[TOPIQ_SIDE - 1] < 0.1 && x[2 * area + TOPIQ_SIDE - 1] > 0.9);

        let wide = vision::Rgb { w: 1200, h: 800, px: vec![[0.5, 0.25, 0.0]; 1200 * 800] };
        let (nw, nh, lx) = input_lumina(&wide).unwrap();
        assert_eq!((nw, nh), (512, 320));
        assert_eq!(nw % 32, 0);
        assert_eq!(nh % 32, 0);
        assert_eq!(lx.len(), 3 * nw * nh);
        assert!((lx[0] - 127.5).abs() < 1.0, "expected 0..255 scaling, got {}", lx[0]);

        let clip_x = input_clip(&wide).unwrap();
        assert_eq!(clip_x.len(), 3 * CLIP_SIDE * CLIP_SIDE);
        let expected_r = (0.5 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((clip_x[0] - expected_r).abs() < 1e-3, "got {}", clip_x[0]);

        // Malformed/empty buffers return Err instead of panicking.
        let empty = vision::Rgb { w: 10, h: 10, px: vec![] };
        assert!(input_topiq(&empty).is_err());
        assert!(input_lumina(&empty).is_err());
        assert!(input_clip(&empty).is_err());

        // Extreme aspect ratios are bounded instead of allocating huge resize buffers.
        let extreme = vision::Rgb { w: 5000, h: 1, px: vec![[0.5, 0.5, 0.5]; 5000] };
        assert_eq!(input_clip(&extreme).unwrap().len(), 3 * CLIP_SIDE * CLIP_SIDE);
    }

    /// Set `OPTIXEL_TEST_MODELS` to a data folder with culling models installed to run end-to-end ONNX inference.
    #[test]
    fn quality_end_to_end() {
        let Some(data) = std::env::var_os("OPTIXEL_TEST_MODELS").map(PathBuf::from) else {
            return;
        };
        let sample = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../html/demo-landscape.jpg");
        let img = image::open(&sample).expect("open demo-landscape.jpg");
        let blurred = img.blur(4.0);
        let rgb_sharp = vision::Rgb::from_image(&img, 768);
        let rgb_blur = vision::Rgb::from_image(&blurred, 768);

        let mut tested = 0;
        if is_installed(&data, LUMINA) && is_installed(&data, LAION_AES) {
            let (t_s, a_s) = score(&data, &rgb_sharp, Some(LUMINA), Some(LAION_AES));
            let (t_b, a_b) = score(&data, &rgb_blur, Some(LUMINA), Some(LAION_AES));
            let (t_s, t_b) = (t_s.unwrap().expect("lumina sharp"), t_b.unwrap().expect("lumina blur"));
            let (a_s, a_b) = (a_s.unwrap().expect("laion sharp"), a_b.unwrap().expect("laion blur"));
            println!("commercial (lumina, laion) sharp=({t_s:.2}, {a_s:.2}) blurred=({t_b:.2}, {a_b:.2})");
            assert!(t_s > t_b + 15.0, "expected lumina sharp ({t_s}) > blurred ({t_b}) + 15");
            assert!(a_s > a_b + 5.0, "expected laion sharp ({a_s}) > blurred ({a_b}) + 5");
            tested += 1;
        }
        if is_installed(&data, TOPIQ_TECH) && is_installed(&data, TOPIQ_AES) {
            let (t_s, a_s) = score(&data, &rgb_sharp, Some(TOPIQ_TECH), Some(TOPIQ_AES));
            let (t_b, a_b) = score(&data, &rgb_blur, Some(TOPIQ_TECH), Some(TOPIQ_AES));
            let (t_s, t_b) = (t_s.unwrap().expect("topiq tech sharp"), t_b.unwrap().expect("topiq tech blur"));
            let (a_s, a_b) = (a_s.unwrap().expect("topiq aes sharp"), a_b.unwrap().expect("topiq aes blur"));
            println!("non-commercial (topiq) sharp=({t_s:.2}, {a_s:.2}) blurred=({t_b:.2}, {a_b:.2})");
            assert!(t_s > t_b + 15.0, "expected topiq sharp ({t_s}) > blurred ({t_b}) + 15");
            assert!(a_s > a_b + 5.0, "expected topiq aes sharp ({a_s}) > blurred ({a_b}) + 5");
            tested += 1;
        }
        assert!(tested > 0, "OPTIXEL_TEST_MODELS set but no culling model pairs were found in {}", data.display());
    }
}
