//! Face swap with FaceFusion's ONNX models: YOLOFace finds the faces and their five landmarks, ArcFace turns
//! the source face into an identity vector, and HyperSwap redraws each face of the photo with that identity,
//! keeping its pose, expression and light. Each swapped face comes back as a patch layer like an object removal.
//!
//! Follows FaceFusion's own pipeline (face_detector.py, face_recognizer.py, face_swapper/core.py) with its
//! defaults: detector score 0.5, box mask with 0.3 blur, no pixel boost.
//! ponytail: every face in the photo is swapped (one layer each, so any can be turned off); the three models load
//! per run (about a second) instead of staying in memory, since this is a rare action. CPU only: one 256 px pass.

use crate::develop::{Img, Patch, Px, blur, srgb_to_working};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;
use std::path::Path;

pub const DETECT: &str = "yoloface_8n.onnx";
pub const RECOGNISE: &str = "arcface_w600k_r50.onnx";
pub const SWAP: &str = "hyperswap_1a_256.onnx";
pub const LANDMARKS: &str = "2dfan4.onnx";

/// Five landmarks: eyes, nose tip, mouth corners (photo pixels).
type Lm = [[f32; 2]; 5];
/// 2×3 affine, row-major: x' = m0·x + m1·y + m2, y' = m3·x + m4·y + m5.
pub(crate) type M = [f32; 6];

/// FaceFusion's warp templates (fractions of the crop side).
const ARCFACE_112_V2: Lm = [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.500225, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]];
const ARCFACE_128: Lm = [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]];
const DETECT_SIZE: usize = 640;
const SWAP_SIZE: usize = 256;

#[derive(Debug, Clone)]
pub struct Face {
    pub score: f32,
    /// x0, y0, x1, y1
    pub bbox: [f32; 4],
    pub lm: Lm,
}

/// Least-squares similarity transform (rotation, uniform scale, shift) taking `src` onto `dst`: what
/// cv2.estimateAffinePartial2D settles on for five points.
fn similarity(src: &Lm, dst: &Lm) -> M {
    let mean = |p: &Lm| p.iter().fold([0.0, 0.0], |a, q| [a[0] + q[0] / 5.0, a[1] + q[1] / 5.0]);
    let (ms, md) = (mean(src), mean(dst));
    let (mut a, mut b, mut den) = (0.0, 0.0, 0.0);
    for (p, q) in src.iter().zip(dst) {
        let (px, py, qx, qy) = (p[0] - ms[0], p[1] - ms[1], q[0] - md[0], q[1] - md[1]);
        a += px * qx + py * qy;
        b += px * qy - py * qx;
        den += px * px + py * py;
    }
    if den <= 1e-8 || den.is_nan() {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    }
    let (a, b) = (a / den, b / den);
    [a, -b, md[0] - (a * ms[0] - b * ms[1]), b, a, md[1] - (b * ms[0] + a * ms[1])]
}

fn invert(m: &M) -> M {
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() <= 1e-8 || det.is_nan() {
        return [1.0, 0.0, -m[2], 0.0, 1.0, -m[5]];
    }
    [m[4] / det, -m[1] / det, (m[1] * m[5] - m[4] * m[2]) / det, -m[3] / det, m[0] / det, (m[3] * m[2] - m[0] * m[5]) / det]
}

fn apply(m: &M, x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

fn template(t: &Lm, side: usize) -> Lm {
    t.map(|[x, y]| [x * side as f32, y * side as f32])
}

/// Bilinear sample of `ch`-channel interleaved `buf` (`w` × `h`), pixel centres on integers, edges repeated.
fn sample(buf: &[f32], w: usize, h: usize, ch: usize, x: f32, y: f32, out: &mut [f32]) {
    if w == 0 || h == 0 || ch == 0 || buf.len() < w * h * ch || out.len() < ch {
        out.fill(0.0);
        return;
    }
    let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, (w - 1) as f32) };
    let y = if y.is_nan() { 0.0 } else { y.clamp(0.0, (h - 1) as f32) };
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1, tx, ty) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1), x.fract(), y.fract());
    for c in 0..ch {
        let at = |xx: usize, yy: usize| buf.get((yy * w + xx) * ch + c).copied().unwrap_or(0.0);
        out[c] = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
    }
}

/// The `side`² crop of `img` that `m` (photo → crop) maps out, as interleaved sRGB 0-1. Each pixel averages n×n
/// samples, n the shrink factor, so a big face does not alias (FaceFusion uses INTER_AREA for this).
pub(crate) fn warp(img: &Img, m: &M, side: usize) -> Vec<f32> {
    if img.w == 0 || img.h == 0 || img.px.len() < img.w * img.h || side == 0 {
        return vec![0f32; side * side * 3];
    }
    let inv = invert(m);
    let n = (inv[0].hypot(inv[3]).ceil() as usize).clamp(1, 16);
    let flat: &[f32] = bytemuck::cast_slice(&img.px);
    let mut out = vec![0f32; side * side * 3];
    out.par_chunks_mut(side * 3).enumerate().for_each(|(v, row)| {
        let mut s = [0f32; 3];
        for u in 0..side {
            let mut acc = [0f32; 3];
            for j in 0..n {
                for i in 0..n {
                    let (cu, cv) = (u as f32 + (i as f32 + 0.5) / n as f32 - 0.5, v as f32 + (j as f32 + 0.5) / n as f32 - 0.5);
                    let (x, y) = apply(&inv, cu, cv);
                    sample(flat, img.w, img.h, 3, x, y, &mut s);
                    (0..3).for_each(|c| acc[c] += s[c]);
                }
            }
            let e = crate::aimodel::encode(acc.map(|a| a / (n * n) as f32));
            row[u * 3..u * 3 + 3].copy_from_slice(&e);
        }
    });
    out
}

/// Interleaved 0-1 RGB → NCHW planes through `f`, optionally swapping to BGR.
fn planes(rgb: &[f32], side: usize, bgr: bool, f: impl Fn(f32) -> f32) -> Vec<f32> {
    let area = side * side;
    let mut p = vec![0f32; 3 * area];
    for i in 0..area {
        for c in 0..3 {
            p[c * area + i] = f(rgb[i * 3 + if bgr { 2 - c } else { c }]);
        }
    }
    p
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let denom = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0) + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0) - inter;
    if denom <= 1e-6 || denom.is_nan() {
        return 0.0;
    }
    (inter / denom).clamp(0.0, 1.0)
}

/// 2DFAN4's 68 landmarks (photo pixels) of the face in `bbox` (x0, y0, x1, y1) and their confidence (0-1):
/// the face scaled so its longer side is 195 px of the 256 px input and centred, as FaceFusion crops it.
pub fn landmarks68(s: &mut Session, img: &Img, bbox: [f32; 4], bgr: bool) -> Result<(Vec<[f32; 2]>, f32), String> {
    const SIDE: usize = 256;
    let scale = 195.0 / (bbox[2] - bbox[0]).max(bbox[3] - bbox[1]).max(1.0);
    let (tx, ty) = ((SIDE as f32 - (bbox[0] + bbox[2]) * scale) * 0.5, (SIDE as f32 - (bbox[1] + bbox[3]) * scale) * 0.5);
    let rgb = warp(img, &[scale, 0.0, tx, 0.0, scale, ty], SIDE);
    let input = Tensor::from_array(([1usize, 3, SIDE, SIDE], planes(&rgb, SIDE, bgr, |v| v))).map_err(|e| format!("landmark input: {e}"))?;
    let name = s.inputs().first().ok_or("the landmark model has no inputs")?.name().to_string();
    let out = s.run(ort::inputs![name.as_str() => input]).map_err(|e| format!("the landmark model failed: {e}"))?;
    if out.len() < 2 {
        return Err("the landmark model returned too few outputs".into());
    }
    let (shape, lm) = out[0].try_extract_tensor::<f32>().map_err(|e| format!("landmark output: {e}"))?;
    let (hshape, hm) = out[1].try_extract_tensor::<f32>().map_err(|e| format!("landmark output: {e}"))?;
    if shape.len() != 3 || shape[1] != 68 || shape[2] < 2 || hshape.len() != 4 {
        return Err(format!("the landmark model returned an unexpected shape {shape:?}"));
    }
    let k = shape[2] as usize;
    // points come at the 64 px heatmap's scale
    let pts = (0..68).map(|i| [(lm[i * k] * 4.0 - tx) / scale, (lm[i * k + 1] * 4.0 - ty) / scale]).collect();
    let cell = (hshape[2] * hshape[3]) as usize;
    let peak: f32 = (0..68).map(|i| hm[i * cell..(i + 1) * cell].iter().fold(0f32, |a, &b| a.max(b))).sum::<f32>() / 68.0;
    Ok((pts, (peak / 0.9).clamp(0.0, 1.0)))
}

/// Eye aspect ratio (Soukupová & Čech) of 68 landmarks, the mean of the two eyes': the lids' opening over
/// the eye's width, about 0.3 open and under 0.2 shut.
pub fn eye_aspect(pts: &[[f32; 2]]) -> f32 {
    let d = |a: usize, b: usize| (pts[a][0] - pts[b][0]).hypot(pts[a][1] - pts[b][1]);
    let ear = |e: usize| (d(e + 1, e + 5) + d(e + 2, e + 4)) / (2.0 * d(e, e + 3)).max(1e-3);
    (ear(36) + ear(42)) / 2.0
}

/// Faces in `img`, best first (YOLOFace at 640, score > 0.5, NMS at IoU 0.4).
pub fn detect(s: &mut Session, img: &Img) -> Result<Vec<Face>, String> {
    if img.w == 0 || img.h == 0 {
        return Err("image has zero dimensions".into());
    }
    let r = (DETECT_SIZE as f32 / img.w.max(img.h) as f32).min(1.0);
    if r <= 0.0 || r.is_nan() {
        return Err("invalid scale for face detector".into());
    }
    let (nw, nh) = ((img.w as f32 * r).round() as usize, (img.h as f32 * r).round() as usize);
    let mut rgb = warp(img, &[r, 0.0, 0.0, 0.0, r, 0.0], DETECT_SIZE);
    // the picture sits top-left, the rest is black
    for (i, p) in rgb.chunks_mut(3).enumerate() {
        if i % DETECT_SIZE >= nw || i / DETECT_SIZE >= nh {
            p.fill(0.0);
        }
    }
    let input = Tensor::from_array(([1usize, 3, DETECT_SIZE, DETECT_SIZE], planes(&rgb, DETECT_SIZE, true, |v| v))).map_err(|e| format!("face detector input: {e}"))?;
    let name = s.inputs().first().ok_or("the face detector has no inputs")?.name().to_string();
    let out = s.run(ort::inputs![name.as_str() => input]).map_err(|e| format!("the face detector failed: {e}"))?;
    if out.len() == 0 {
        return Err("the face detector returned no outputs".into());
    }
    let (shape, o) = out[0].try_extract_tensor::<f32>().map_err(|e| format!("face detector output: {e}"))?;
    // [1, 20, anchors]: cx, cy, w, h, score, then five (x, y, visibility)
    if shape.len() != 3 || shape[1] != 20 {
        return Err(format!("the face detector returned an unexpected shape {shape:?}"));
    }
    let n = shape[2] as usize;
    if o.len() < 20 * n {
        return Err("the face detector returned truncated output tensor".into());
    }
    let at = |row: usize, i: usize| o[row * n + i] / r;
    let mut faces: Vec<Face> = (0..n)
        .filter(|&i| o[4 * n + i] > 0.5)
        .map(|i| Face {
            score: o[4 * n + i],
            bbox: [at(0, i) - at(2, i) / 2.0, at(1, i) - at(3, i) / 2.0, at(0, i) + at(2, i) / 2.0, at(1, i) + at(3, i) / 2.0],
            lm: std::array::from_fn(|k| [at(5 + k * 3, i), at(6 + k * 3, i)]),
        })
        .collect();
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        if kept.iter().all(|k| iou(&k.bbox, &f.bbox) < 0.4) {
            kept.push(f);
        }
    }
    Ok(kept)
}

/// The face's identity: ArcFace's 512-d vector, normalised.
pub fn embed(s: &mut Session, img: &Img, face: &Face) -> Result<Vec<f32>, String> {
    let crop = warp(img, &similarity(&face.lm, &template(&ARCFACE_112_V2, 112)), 112);
    let input = Tensor::from_array(([1usize, 3, 112, 112], planes(&crop, 112, false, |v| v * 2.0 - 1.0))).map_err(|e| format!("face recogniser input: {e}"))?;
    let name = s.inputs().first().ok_or("the face recogniser has no inputs")?.name().to_string();
    let out = s.run(ort::inputs![name.as_str() => input]).map_err(|e| format!("the face recogniser failed: {e}"))?;
    if out.len() == 0 {
        return Err("the face recogniser returned no outputs".into());
    }
    let (_, e) = out[0].try_extract_tensor::<f32>().map_err(|e| format!("face recogniser output: {e}"))?;
    let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm = if norm.is_nan() || norm < 1e-6 { 1.0 } else { norm };
    Ok(e.iter().map(|v| if v.is_nan() { 0.0 } else { v / norm }).collect())
}

/// Swap one face to the identity `source` (from `embed`) and return it as a feathered patch of the photo.
pub fn swap(s: &mut Session, img: &Img, face: &Face, source: &[f32]) -> Result<Patch, String> {
    let m = similarity(&face.lm, &template(&ARCFACE_128, SWAP_SIZE));
    let crop = warp(img, &m, SWAP_SIZE);
    let mut inputs: Vec<(String, ort::session::SessionInputValue)> = Vec::new();
    for i in s.inputs() {
        let v = match i.name() {
            "source" => Tensor::from_array(([1usize, source.len()], source.to_vec())).map_err(|e| format!("face swap input: {e}"))?.into(),
            "target" => Tensor::from_array(([1usize, 3, SWAP_SIZE, SWAP_SIZE], planes(&crop, SWAP_SIZE, false, |v| v * 2.0 - 1.0))).map_err(|e| format!("face swap input: {e}"))?.into(),
            other => return Err(format!("the face swap model wants an unknown input '{other}'")),
        };
        inputs.push((i.name().to_string(), v));
    }
    let out = s.run(inputs).map_err(|e| format!("the face swap model failed: {e}"))?;
    if out.len() == 0 {
        return Err("the face swap model returned no outputs".into());
    }
    let (shape, o) = out[0].try_extract_tensor::<f32>().map_err(|e| format!("face swap output: {e}"))?;
    if shape.len() != 4 || shape[1] != 3 || shape[2] as usize != SWAP_SIZE || shape[3] as usize != SWAP_SIZE {
        return Err(format!("the face swap model returned an unexpected shape {shape:?}"));
    }
    let area = SWAP_SIZE * SWAP_SIZE;
    if o.len() < area * 3 {
        return Err("the face swap model returned truncated output tensor".into());
    }
    let swapped: Vec<f32> = (0..area * 3).map(|k| (o[(k % 3) * area + k / 3] * 0.5 + 0.5).clamp(0.0, 1.0)).collect();
    drop(out);

    // FaceFusion's box mask: zero a border of side·0.3/4, then blur it with σ = side·0.3/8
    let border = SWAP_SIZE * 3 / 40;
    let hard: Vec<f32> = (0..area).map(|k| if (border..SWAP_SIZE - border).contains(&(k % SWAP_SIZE)) && (border..SWAP_SIZE - border).contains(&(k / SWAP_SIZE)) { 1.0 } else { 0.0 }).collect();
    let mask = blur(&hard, SWAP_SIZE, SWAP_SIZE, SWAP_SIZE as f32 * 0.3 / 8.0);

    // the crop's footprint on the photo, squared up (a patch is square)
    let inv = invert(&m);
    let corners = [(0.0, 0.0), (SWAP_SIZE as f32, 0.0), (0.0, SWAP_SIZE as f32), (SWAP_SIZE as f32, SWAP_SIZE as f32)].map(|(u, v)| apply(&inv, u, v));
    let (x0, x1) = corners.iter().fold((f32::MAX, f32::MIN), |(a, b), c| (a.min(c.0), b.max(c.0)));
    let (y0, y1) = corners.iter().fold((f32::MAX, f32::MIN), |(a, b), c| (a.min(c.1), b.max(c.1)));
    let side = ((x1 - x0).max(y1 - y0).ceil() as usize).clamp(1, img.w.min(img.h));
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let sx = ((cx - side as f32 / 2.0).max(0.0) as usize).min(img.w.saturating_sub(side));
    let sy = ((cy - side as f32 / 2.0).max(0.0) as usize).min(img.h.saturating_sub(side));
    let last = (SWAP_SIZE - 1) as f32;
    let data: Vec<[f32; 4]> = (0..side * side)
        .into_par_iter()
        .map(|k| {
            let (u, v) = apply(&m, (sx + k % side) as f32, (sy + k / side) as f32);
            if !(0.0..=last).contains(&u) || !(0.0..=last).contains(&v) {
                return [0.0; 4];
            }
            let (mut a, mut c) = ([0f32; 1], [0f32; 3]);
            sample(&mask, SWAP_SIZE, SWAP_SIZE, 1, u, v, &mut a);
            sample(&swapped, SWAP_SIZE, SWAP_SIZE, 3, u, v, &mut c);
            let p: Px = srgb_to_working([c[0], c[1], c[2]]);
            [p[0], p[1], p[2], a[0].clamp(0.0, 1.0)]
        })
        .collect();
    Ok(Patch { sx, sy, side, orig_w: img.w, orig_h: img.h, data })
}

/// Put the biggest face of `source` onto every face of `img`. `tick(label, done, total)` reports progress and
/// returns false to cancel.
pub fn run(dir: &Path, img: &Img, source: &Img, tick: &dyn Fn(&str, usize, usize) -> bool) -> Result<Vec<Patch>, String> {
    if img.w == 0 || img.h == 0 || source.w == 0 || source.h == 0 || img.px.len() < img.w * img.h || source.px.len() < source.w * source.h {
        return Err("invalid image dimensions or data".into());
    }
    let open = |f: &str| crate::aimodel::open_session(&dir.join(f), None, 0, false);
    let cancelled = || Err::<Vec<Patch>, String>("cancelled".into());
    if !tick("Finding faces", 0, 3) {
        return cancelled();
    }
    let mut det = open(DETECT)?;
    let src = detect(&mut det, source)?
        .into_iter()
        .max_by(|a, b| ((a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])).total_cmp(&((b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))))
        .ok_or("no face found in the source picture")?;
    let targets = detect(&mut det, img)?;
    if targets.is_empty() {
        return Err("no face found in this photo".into());
    }
    drop(det);
    let total = targets.len() + 2;
    if !tick("Reading the source face", 1, total) {
        return cancelled();
    }
    let identity = embed(&mut open(RECOGNISE)?, source, &src)?;
    let mut swapper = open(SWAP)?;
    let mut out = Vec::with_capacity(targets.len());
    for (i, f) in targets.iter().enumerate() {
        if !tick(&format!("Swapping face {} of {}", i + 1, targets.len()), i + 2, total) {
            return cancelled();
        }
        out.push(swap(&mut swapper, img, f, &identity)?);
    }
    tick("Done", total, total);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_recovers_a_known_transform() {
        let m: M = [0.8, -0.6, 10.0, 0.6, 0.8, -4.0]; // scale 1, rotated, shifted
        let src: Lm = [[10.0, 20.0], [50.0, 22.0], [30.0, 40.0], [15.0, 60.0], [45.0, 61.0]];
        let dst = src.map(|[x, y]| {
            let (u, v) = apply(&m, x, y);
            [u, v]
        });
        let got = similarity(&src, &dst);
        assert!(got.iter().zip(m).all(|(a, b)| (a - b).abs() < 1e-3), "{got:?}");
        let back = apply(&invert(&got), dst[2][0], dst[2][1]);
        assert!((back.0 - 30.0).abs() < 1e-3 && (back.1 - 40.0).abs() < 1e-3);
    }

    /// Needs OPTIXEL_TEST_AI=1 and OPTIXEL_TEST_FACES=<source image>,<target image>: downloads the face swap
    /// models into the app's data folder, swaps, and writes the result next to the target as *_swapped.jpg.
    #[test]
    fn face_swap_end_to_end() {
        let (Some(_), Ok(paths)) = (
            std::env::var_os("OPTIXEL_TEST_AI"),
            std::env::var("OPTIXEL_TEST_FACES"),
        ) else { return };
        let (src, dst) = paths.split_once(',').expect("source,target");
        let data = crate::data_dir();
        crate::aimodel::download(&data, "hyperswap-1a", &|d, t| eprint!("\rdownload {d}/{t}")).unwrap();
        let source = crate::develop::load(Path::new(src), Some(2048)).unwrap();
        let mut img = crate::develop::load(Path::new(dst), None).unwrap();
        let t0 = std::time::Instant::now();
        let patches = crate::aimodel::face_swap(&data, "hyperswap-1a", &img, &source, &|l, d, t| {
            eprintln!("{l} {d}/{t}");
            true
        })
        .unwrap();
        eprintln!("{} face(s) in {:?}", patches.len(), t0.elapsed());
        assert!(!patches.is_empty());
        for p in &patches {
            assert!(p.data.iter().any(|d| d[3] > 0.9), "a patch with no face in it");
            crate::develop::composite_patch(&mut img.px, img.w, img.h, p, [1.0, 1.0, 1.0]);
        }
        let rgb: Vec<u8> = img.px.iter().flat_map(|p| crate::aimodel::encode(*p).map(|v| (v * 255.0).round() as u8)).collect();
        let out = Path::new(dst).with_extension("swapped.jpg");
        image::RgbImage::from_raw(img.w as u32, img.h as u32, rgb).unwrap().save(&out).unwrap();
        eprintln!("wrote {}", out.display());
    }
}

