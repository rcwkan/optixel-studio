//! Image analysis for the auto functions (spec §3.1.2, §3.4.1): how sharp the photo and its main subject
//! are (edge blur measured by re-blurring), motion blur, noise, exposure, colour, a perceptual hash for
//! duplicates, and built-in outlines of the main subject and the sky.
//!
//! ponytail: classical estimators, no neural models. The subject is whatever differs from the colours along
//! the frame's border, weighted toward the centre (and, for masks, toward what is in focus), so a small
//! subject in a busy scene can be missed: the auto functions can ask the AI provider for it instead.

use crate::develop::{self, Img, Px};
use image::DynamicImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, f32::consts::PI};

/// Where the subject is taken to be when nothing stands out: the middle of the frame, [x, y, w, h].
pub const CENTRE: [f32; 4] = [0.2, 0.15, 0.6, 0.7];
/// A perfect step edge measures this blur (the gradient's own footprint); it is taken out of every reading.
const FLOOR: f32 = 0.75;

/// What the auto functions measure on a photo (cached per file in the catalog).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Metrics {
    /// Size the sharpness was measured at.
    pub w: u32,
    pub h: u32,
    /// 0-100 from the edge blur: the frame's sharpest part, and the main subject.
    pub sharpness: f32,
    pub subject_sharpness: f32,
    /// Edge blur behind the two scores: Gaussian σ in pixels as judged (2048 px, or 100% at full size).
    pub blur_px: f32,
    pub subject_blur_px: f32,
    /// Main subject [x, y, w, h] as fractions of the frame.
    pub subject: [f32; 4],
    /// How one-directional the subject's blur is (0-1; camera shake or a moving subject) and the
    /// direction of the motion in degrees (0 = horizontal).
    pub motion: f32,
    pub motion_angle: f32,
    /// Edge points measured; a featureless frame (fog, a plain sky) has too few to call it blurry.
    pub edges: u32,
    /// Noise σ in 8-bit steps at the measured size.
    pub noise: f32,
    /// Display brightness 0-1 (mean and 1st / 50th / 99th percentile), clipped highlights and crushed
    /// shadows in % of the frame.
    pub mean: f32,
    pub p01: f32,
    pub p50: f32,
    pub p99: f32,
    pub clipped_highlights: f32,
    pub clipped_shadows: f32,
    /// 0-100: little clipping and a usable brightness.
    pub exposure: f32,
    /// Hasler–Süsstrunk colourfulness (below ~8: black and white) and the dominant colours by name.
    pub colorfulness: f32,
    pub colors: Vec<String>,
    /// Perceptual (DCT) hash of the frame, 16 hex digits.
    pub phash: String,
    /// Native C++ focus-peaking score (0-100): over the eyes when there are faces, else the frame.
    pub focus_peaking_score: f32,
    /// The most closed face's eye aspect ratio (from its landmarks: about 0.3 open, under 0.2 shut); None
    /// without faces or without the landmark model.
    pub eye_openness: Option<f32>,
    /// A face (from the face scan) with both eyes shut, by its landmarks. Never without faces.
    pub blink_detected: bool,
}

/// Display-referred (sRGB-encoded) colour 0-1, row-major: what the analysis looks at.
#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Px>,
}

impl Rgb {
    /// `img` fitted inside `edge` × `edge` (0: full size). Float images are linear and get the sRGB curve.
    pub fn from_image(img: &DynamicImage, edge: u32) -> Rgb {
        let fit;
        let img = if edge > 0 && img.width().max(img.height()) > edge {
            fit = img.thumbnail(edge, edge);
            &fit
        } else {
            img
        };
        let linear = develop::is_float(img);
        let buf = img.to_rgb32f();
        let px = buf
            .as_raw()
            .par_chunks_exact(3)
            .map(|c| {
                let p = [c[0], c[1], c[2]];
                if linear { p.map(|v| develop::srgb_oetf(v.clamp(0.0, 1.0))) } else { p }
            })
            .collect();
        Rgb { w: buf.width() as usize, h: buf.height() as usize, px }
    }

    /// The develop working space (linear Rec.2020) as displayed.
    pub fn from_working(img: &Img, edge: u32) -> Rgb {
        Rgb::from_image(&crate::catalog::to_dynamic(img), edge)
    }

    /// Area-average copy fitted inside `edge` × `edge`.
    pub fn fit(&self, edge: usize) -> Rgb {
        let s = edge as f32 / self.w.max(self.h) as f32;
        if s >= 1.0 {
            return self.clone();
        }
        let (w, h) = (((self.w as f32 * s).round() as usize).max(1), ((self.h as f32 * s).round() as usize).max(1));
        let ch: [Vec<f32>; 3] = std::array::from_fn(|c| area(&self.px.iter().map(|p| p[c]).collect::<Vec<_>>(), self.w, self.h, w, h));
        Rgb { w, h, px: (0..w * h).map(|i| [ch[0][i], ch[1][i], ch[2][i]]).collect() }
    }

    /// Rec.709 luma of the encoded values (Y').
    pub fn luma(&self) -> Vec<f32> {
        self.px.par_iter().map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect()
    }

    fn lab(&self) -> Vec<Px> {
        self.px.par_iter().map(|p| develop::oklab(develop::srgb_to_working(*p))).collect()
    }
}

/// Area-average downscale of a plane: each source pixel adds to the output pixel its centre falls in
/// (nearest neighbour when enlarging).
fn area(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    if nw > w || nh > h {
        return (0..nw * nh).map(|i| src[(i / nw) * h / nh * w + (i % nw) * w / nw]).collect();
    }
    let (mut sum, mut n) = (vec![0f32; nw * nh], vec![0f32; nw * nh]);
    for y in 0..h {
        let row = y * nh / h * nw;
        for x in 0..w {
            sum[row + x * nw / w] += src[y * w + x];
            n[row + x * nw / w] += 1.0;
        }
    }
    sum.iter().zip(&n).map(|(s, n)| s / n.max(1.0)).collect()
}

/// Separable Gaussian with an exact kernel (the re-blur ratio relies on σ), edges clamped.
fn gauss(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / total).collect();
    let rows = |src: &[f32], w: usize, h: usize| -> Vec<f32> {
        let mut out = vec![0.0; w * h];
        out.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(o, s)| {
            for (x, v) in o.iter_mut().enumerate() {
                *v = k.iter().enumerate().map(|(j, kv)| kv * s[(x as isize + j as isize - r).clamp(0, w as isize - 1) as usize]).sum();
            }
        });
        out
    };
    let a = develop::transpose(&rows(src, w, h), w, h);
    develop::transpose(&rows(&a, h, w), h, w)
}

/// Sobel gradient as a per-pixel derivative, edges clamped.
fn sobel(v: &[f32], w: usize, h: usize) -> Vec<[f32; 2]> {
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (x0, x1, y0, y1) = (x.saturating_sub(1), (x + 1).min(w - 1), y.saturating_sub(1), (y + 1).min(h - 1));
            let at = |x: usize, y: usize| v[y * w + x];
            let gx = at(x1, y0) + 2.0 * at(x1, y) + at(x1, y1) - at(x0, y0) - 2.0 * at(x0, y) - at(x0, y1);
            let gy = at(x0, y1) + 2.0 * at(x, y1) + at(x1, y1) - at(x0, y0) - 2.0 * at(x, y0) - at(x1, y0);
            [gx / 8.0, gy / 8.0]
        })
        .collect()
}

/// Noise σ (Immerkær): the [[1,-2,1],[-2,4,-2],[1,-2,1]] response is 6σ on white noise. Its 30th
/// percentile (flat areas dominate it) is 0.385 × that for Gaussian noise, so texture and edges barely count.
fn noise(v: &[f32], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut r: Vec<f32> = (1..h - 1)
        .step_by(2)
        .flat_map(|y| (1..w - 1).step_by(2).map(move |x| (x, y)))
        .map(|(x, y)| {
            let at = |dx: isize, dy: isize| v[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
            (at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1) - 2.0 * (at(0, -1) + at(-1, 0) + at(1, 0) + at(0, 1)) + 4.0 * at(0, 0)).abs()
        })
        .collect();
    let k = r.len() * 3 / 10;
    *r.select_nth_unstable_by(k, f32::total_cmp).1 / (0.385 * 6.0)
}

/// One edge point: pixel, blur σ (px at the measured size), gradient direction (0°, 45°, 90°, 135°) and contrast.
struct Edge {
    i: usize,
    sigma: f32,
    dir: u8,
    g: f32,
}

/// Edge blur by re-blurring: at an edge point, blurring with σ0 lowers the gradient by R = √(σ² + σ0²) / σ,
/// so the edge's own blur is σ = σ0 / √(R² − 1), whatever the edge's contrast or colours. σ0 = 2 keeps R
/// well above 1 for blurred edges, so 8-bit steps barely move the reading. Edge points are where the
/// re-blurred gradient peaks across the edge (well above the noise): its own ripples, e.g. the steps JPEG
/// leaves inside each 8 × 8 block on a soft slope, are smoothed away there.
fn edges(v: &[f32], w: usize, h: usize, noise: f32) -> Vec<Edge> {
    const S0: f32 = 2.0;
    const ACROSS: [[(isize, isize); 2]; 4] = [[(1, 0), (-1, 0)], [(1, 1), (-1, -1)], [(0, 1), (0, -1)], [(-1, 1), (1, -1)]];
    let (g0, g1) = (sobel(v, w, h), sobel(&gauss(v, w, h, S0), w, h));
    let mag = |g: [f32; 2]| g[0].hypot(g[1]);
    let t = (6.0 * noise).max(0.01);
    (0..w * h)
        .into_par_iter()
        .filter_map(|i| {
            let (x, y) = (i % w, i / w);
            let (g, soft) = (mag(g0[i]), mag(g1[i]));
            if g < t || x < 2 || y < 2 || x + 2 >= w || y + 2 >= h {
                return None;
            }
            let dir = ((g1[i][1].atan2(g1[i][0]).rem_euclid(PI) / (PI / 4.0)).round() as usize % 4) as u8;
            let at = |(dx, dy): (isize, isize)| mag(g1[(y as isize + dy) as usize * w + (x as isize + dx) as usize]);
            let [p, q] = ACROSS[dir as usize];
            if soft < at(p) || soft <= at(q) {
                return None;
            }
            let r = g / soft.max(1e-6);
            let sigma = if r > 1.001 { (S0 / (r * r - 1.0).sqrt()).min(8.0) } else { 8.0 };
            Some(Edge { i, sigma, dir, g })
        })
        .collect()
}

/// Contrast-weighted percentile of the edges' blur; None with fewer than `min` edges.
fn blur_at(es: &[&Edge], q: f32, min: usize) -> Option<f32> {
    if es.len() < min.max(1) {
        return None;
    }
    let mut v: Vec<(f32, f32)> = es.iter().map(|e| (e.sigma, e.g)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = v.iter().map(|e| e.1).sum();
    let mut acc = 0.0;
    v.iter().find(|e| {
        acc += e.1;
        acc >= q * total
    })
    .or(v.last())
    .map(|e| e.0)
}

/// Blur of a set of edges: its q-th percentile, unless the blur is strongly one-directional (motion), when
/// the worst direction counts: motion leaves the edges along it sharp. Also the motion 0-1 and its angle.
fn region_blur(es: &[&Edge], q: f32) -> Option<(f32, f32, f32)> {
    let sigma = blur_at(es, q, 12)?;
    let dirs: Vec<(usize, f32)> = (0..4)
        .filter_map(|d| blur_at(&es.iter().copied().filter(|e| e.dir as usize == d).collect::<Vec<_>>(), 0.25, 25).map(|s| (d, s)))
        .collect();
    let Some(&(worst, max)) = dirs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).filter(|_| dirs.len() >= 2) else {
        return Some((sigma, 0.0, 0.0));
    };
    let min = dirs.iter().map(|d| d.1).fold(f32::MAX, f32::min);
    // deblurred, so two sharp directions read as equal rather than as a ratio of their measuring floors
    let clean = |s: f32| (s * s - FLOOR * FLOOR).max(0.0).sqrt();
    let motion = (1.0 - (clean(min) + 0.3) / (clean(max) + 0.3)).max(0.0);
    Some((if motion >= 0.4 { sigma.max(max) } else { sigma }, motion, worst as f32 * 45.0))
}

/// Blur of the frame's sharpest part: each of 6 × 6 tiles with enough edges is read on its own, and the
/// second sharpest counts (one tile alone may be a fluke); a global low percentile instead would miss a
/// small in-focus subject among many soft edges.
fn frame_blur(es: &[Edge], w: usize, h: usize) -> Option<f32> {
    let mut tiles: Vec<Vec<&Edge>> = vec![vec![]; 36];
    for e in es {
        tiles[(e.i / w * 6 / h) * 6 + (e.i % w) * 6 / w].push(e);
    }
    let mut read: Vec<f32> = tiles.iter().filter_map(|t| blur_at(t, 0.25, 25)).collect();
    read.sort_by(f32::total_cmp);
    match read.len() {
        0 => blur_at(&es.iter().collect::<Vec<_>>(), 0.1, 12),
        1 | 2 => Some(read[0]),
        _ => Some(read[1]),
    }
}

/// Sharpness 0-100 from an edge blur σ as judged: 0.5 px or less is 100, every further 0.25 px costs 10.
fn score(sigma: f32) -> f32 {
    (100.0 * (1.0 - (sigma - 0.5) / 2.5)).clamp(0.0, 100.0)
}

/// Noise σ in 8-bit steps.
pub fn noise_of(s: &Rgb) -> f32 {
    noise(&s.luma(), s.w, s.h) * 255.0
}

/// Measure a decoded photo. Sharpness is measured fitted inside `edge` (0: full size) and the blur judged
/// as it looks with the long edge `judge` px (2048 for a screen, the photo's own size for 100%; 0: as
/// measured). `subject` overrides the built-in subject box (e.g. the AI provider's).
/// `eyes`: of each face in the photo (`faces::photo_eyes`), for eye focus and blinks.
pub fn analyse(img: &DynamicImage, edge: u32, judge: u32, subject: Option<[f32; 4]>, eyes: &[[[f32; 2]; 2]]) -> Metrics {
    let detail = Rgb::from_image(img, edge);
    let (w, h) = (detail.w, detail.h);
    let small = detail.fit(512);
    let v = detail.luma();
    let n = noise(&v, w, h);
    let es = edges(&v, w, h, n);
    let norm = if judge > 0 { judge as f32 / w.max(h) as f32 } else { 1.0 };
    let judged = |s: f32| (s * s - FLOOR * FLOOR).max(0.0).sqrt() * norm;
    let frame = frame_blur(&es, w, h);
    // the subject's edges: its own texture well inside its region (an out-of-focus subject's soft halo
    // overlaps the background), else its region grown a little, so its outline counts; given only a box,
    // inside the ellipse the box frames, so background in the box's corners stays out
    let region = if subject.is_none() { subject_region(&small, 0.0) } else { None };
    let sbox = subject.or_else(|| region.as_ref().map(|m| bbox(m, small.w, small.h))).unwrap_or(CENTRE);
    let xy = |e: &Edge| ((e.i % w) as f32 / w as f32, (e.i / w) as f32 / h as f32);
    let subj = match region {
        Some(m) => {
            let soft = develop::blur(&m.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>(), small.w, small.h, 2.0);
            let within = |t: &dyn Fn(f32) -> bool| -> Vec<&Edge> {
                es.iter()
                    .filter(|e| {
                        let (x, y) = xy(e);
                        t(soft[((y * small.h as f32) as usize).min(small.h - 1) * small.w + ((x * small.w as f32) as usize).min(small.w - 1)])
                    })
                    .collect()
            };
            let core = within(&|v| v > 0.95);
            region_blur(&if core.len() >= 30 { core } else { within(&|v| v > 0.05) }, 0.25)
        }
        None => {
            let ellipse = |e: &&Edge| {
                let (x, y) = xy(e);
                ((x - sbox[0] - sbox[2] / 2.0) / (sbox[2] / 2.0)).powi(2) + ((y - sbox[1] - sbox[3] / 2.0) / (sbox[3] / 2.0)).powi(2) <= 1.0
            };
            region_blur(&es.iter().filter(ellipse).collect::<Vec<_>>(), 0.25)
        }
    }
    .or(frame.map(|f| (f, 0.0, 0.0)));
    // featureless frames (no edges at all) are not blurry: nothing to judge
    let (fb, sb) = (frame.map_or(0.0, judged), subj.map_or(0.0, |s| judged(s.0)));
    let mut m = Metrics {
        w: w as u32,
        h: h as u32,
        sharpness: score(fb),
        subject_sharpness: score(sb),
        blur_px: fb,
        subject_blur_px: sb,
        subject: sbox,
        motion: subj.map_or(0.0, |s| s.1),
        motion_angle: subj.map_or(0.0, |s| s.2),
        edges: es.len() as u32,
        noise: n * 255.0,
        phash: format!("{:016x}", phash(&small.luma(), small.w, small.h)),
        ..Default::default()
    };
    exposure(&small, &mut m);
    colours(&small, &mut m);
    focus_and_blink(&small, &detail, eyes, &mut m);
    m
}

/// Native C++ focus peaking: the eyes' sharpness when the face scan found faces (on the larger `detail`
/// copy, where a group's eyes are more than a few pixels), else the frame's. Closed eyes come from the
/// faces' landmarks (`faces::eye_openness`), not from here.
fn focus_and_blink(small: &Rgb, detail: &Rgb, eyes: &[[[f32; 2]; 2]], m: &mut Metrics) {
    let s = if eyes.is_empty() { small } else { detail };
    if s.w < 8 || s.h < 8 {
        return;
    }
    let flat_rgb: Vec<f32> = s.px.iter().flat_map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0), p[2].clamp(0.0, 1.0)]).collect();
    let patches: Vec<[f32; 8]> = eyes.iter().map(|e| crate::faces::eye_patches(*e, s.w, s.h)).collect();
    m.focus_peaking_score = crate::libraw::focus_peaking(&flat_rgb, s.w, s.h, 0.25, &patches).1;
}

/// Brightness percentiles, clipping and the exposure score, on the display values.
fn exposure(s: &Rgb, m: &mut Metrics) {
    let n = s.px.len().max(1) as f32;
    let mut l = s.luma();
    m.mean = l.iter().sum::<f32>() / n;
    l.sort_by(f32::total_cmp);
    let pct = |q: f32| l.get(((l.len().max(1) - 1) as f32 * q) as usize).copied().unwrap_or(0.0);
    (m.p01, m.p50, m.p99) = (pct(0.01), pct(0.5), pct(0.99));
    m.clipped_highlights = s.px.iter().filter(|p| p.iter().any(|&c| c >= 0.995)).count() as f32 / n * 100.0;
    m.clipped_shadows = s.px.iter().filter(|p| p.iter().all(|&c| c <= 0.01)).count() as f32 / n * 100.0;
    let dark = (0.12 - m.mean).max(0.0) + (m.mean - 0.85).max(0.0);
    m.exposure = (100.0 - 4.0 * (m.clipped_highlights - 0.5).max(0.0) - 3.0 * (m.clipped_shadows - 2.0).max(0.0) - 400.0 * dark).clamp(0.0, 100.0);
}

/// Colourfulness (on 0-255 values) and the colours covering at least 12% of the frame, most first.
fn colours(s: &Rgb, m: &mut Metrics) {
    let n = s.px.len().max(1) as f32;
    let (rg, yb): (Vec<f32>, Vec<f32>) = s.px.iter().map(|p| (255.0 * (p[0] - p[1]), 255.0 * (0.5 * (p[0] + p[1]) - p[2]))).unzip();
    let stat = |v: &[f32]| {
        let mean = v.iter().sum::<f32>() / n;
        (mean, (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt())
    };
    let ((mrg, srg), (myb, syb)) = (stat(&rg), stat(&yb));
    m.colorfulness = srg.hypot(syb) + 0.3 * mrg.hypot(myb);
    const NAMES: [(f32, &str); 8] = [(29.0, "red"), (55.0, "orange"), (100.0, "yellow"), (142.0, "green"), (195.0, "teal"), (264.0, "blue"), (295.0, "purple"), (330.0, "pink")];
    let mut count = [0usize; 9]; // + brown
    for [l, a, b] in s.lab() {
        let c = a.hypot(b);
        if c < 0.05 || !(0.2..0.97).contains(&l) {
            continue;
        }
        let hue = b.atan2(a).to_degrees().rem_euclid(360.0);
        let off = |c: f32| ((hue - c + 540.0).rem_euclid(360.0) - 180.0).abs();
        let k = (0..8).min_by(|&i, &j| off(NAMES[i].0).total_cmp(&off(NAMES[j].0))).unwrap_or(0);
        count[if k <= 2 && l < 0.55 && c < 0.12 { 8 } else { k }] += 1;
    }
    let mut named: Vec<(usize, &str)> = count.iter().zip(NAMES.iter().map(|n| n.1).chain(["brown"])).map(|(c, n)| (*c, n)).collect();
    named.sort_by_key(|c| std::cmp::Reverse(c.0));
    m.colors = named.into_iter().take_while(|(c, _)| *c as f32 >= 0.12 * n).take(3).map(|(_, n)| n.to_string()).collect();
}

/// 64-bit DCT hash: the 8×8 lowest frequencies of a 32×32 copy, each above or below their median.
fn phash(v: &[f32], w: usize, h: usize) -> u64 {
    let s = area(v, w, h, 32, 32);
    let c = |n: usize, k: usize| (PI / 32.0 * (n as f32 + 0.5) * k as f32).cos();
    let rows: Vec<[f32; 8]> = (0..32).map(|y| std::array::from_fn(|k| (0..32).map(|x| s[y * 32 + x] * c(x, k)).sum())).collect();
    let d: Vec<f32> = (0..64).map(|i| (0..32).map(|y| rows[y][i % 8] * c(y, i / 8)).sum()).collect();
    let mut ac = d[1..].to_vec();
    let median = *ac.select_nth_unstable_by(31, f32::total_cmp).1;
    (1..64).filter(|&i| d[i] > median).fold(0u64, |h, i| h | 1 << i)
}

/// pHash of any decoded image (e.g. a catalog thumbnail).
pub fn phash_of(img: &DynamicImage) -> u64 {
    let s = Rgb::from_image(img, 256);
    phash(&s.luma(), s.w, s.h)
}

/// Bits that differ between two hashes (hex strings as stored in Metrics); 64 when one is missing.
pub fn distance(a: &str, b: &str) -> u32 {
    match (u64::from_str_radix(a, 16), u64::from_str_radix(b, 16)) {
        (Ok(a), Ok(b)) => (a ^ b).count_ones(),
        _ => 64,
    }
}

// ---------------------------------------------------------------------------
// Regions: the main subject and the sky, and their outlines
// ---------------------------------------------------------------------------

fn dist(a: Px, b: Px) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// k-means palette of `pts` (seeded with the mean, then each farthest point), with each colour's share.
fn palette(pts: &[Px], k: usize) -> Vec<(Px, f32)> {
    if pts.is_empty() {
        return vec![];
    }
    let nearest = |p: &Px, c: &[Px]| c.iter().enumerate().map(|(i, q)| (i, dist(*p, *q))).fold((0, f32::MAX), |m, x| if x.1 < m.1 { x } else { m });
    let mean = |ps: &[&Px]| [0, 1, 2].map(|c| ps.iter().map(|p| p[c]).sum::<f32>() / ps.len().max(1) as f32);
    let mut c = vec![mean(&pts.iter().collect::<Vec<_>>())];
    while c.len() < k {
        let far = pts.iter().max_by(|a, b| nearest(a, &c).1.total_cmp(&nearest(b, &c).1)).copied().unwrap_or(c[0]);
        if nearest(&far, &c).1 < 0.03 {
            break;
        }
        c.push(far);
    }
    let mut share = vec![0f32; c.len()];
    for _ in 0..8 {
        let mut groups: Vec<Vec<&Px>> = vec![vec![]; c.len()];
        for p in pts {
            groups[nearest(p, &c).0].push(p);
        }
        for (i, g) in groups.iter().enumerate() {
            if !g.is_empty() {
                c[i] = mean(g);
            }
            share[i] = g.len() as f32 / pts.len() as f32;
        }
    }
    c.into_iter().zip(share).collect()
}

/// Values divided by their q-th quantile, clamped to 0-1.
fn normalise(v: &[f32], q: f32) -> Vec<f32> {
    let mut s = v.to_vec();
    let k = ((s.len().max(1) - 1) as f32 * q) as usize;
    let top = s.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-6);
    v.iter().map(|x| (x / top).clamp(0.0, 1.0)).collect()
}

/// Otsu threshold of values in 0-1.
fn otsu(v: &[f32]) -> f32 {
    let mut hist = [0f64; 64];
    for &x in v {
        hist[(x.clamp(0.0, 1.0) * 63.0).round() as usize] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let sum: f64 = hist.iter().enumerate().map(|(i, c)| i as f64 * c).sum();
    let (mut wb, mut sb, mut best, mut t) = (0.0, 0.0, -1.0, 0);
    for (i, &c) in hist.iter().enumerate() {
        wb += c;
        let wf = total - wb;
        if wb == 0.0 || wf == 0.0 {
            continue;
        }
        sb += i as f64 * c;
        let between = wb * wf * (sb / wb - (sum - sb) / wf).powi(2);
        if between > best {
            (best, t) = (between, i);
        }
    }
    (t as f32 + 0.5) / 63.0
}

const NONE: u32 = u32::MAX;

/// 4-connected regions of `m`: a label per pixel (NONE outside) and the region count.
fn components(m: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
    let mut label = vec![NONE; w * h];
    let mut n = 0;
    let mut queue = VecDeque::new();
    for s in 0..w * h {
        if !m[s] || label[s] != NONE {
            continue;
        }
        label[s] = n as u32;
        queue.push_back(s);
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if m[j] && label[j] == NONE {
                    label[j] = n as u32;
                    queue.push_back(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        n += 1;
    }
    (label, n)
}

/// `m` blurred by σ and thresholded at half: rounds corners and drops specks.
fn smoothed(m: &[bool], w: usize, h: usize, sigma: f32) -> Vec<bool> {
    develop::blur(&m.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>(), w, h, sigma).iter().map(|&v| v > 0.5).collect()
}

/// Bounding box [x, y, w, h] of the set pixels, grown by 3% of the frame on each side.
fn bbox(m: &[bool], w: usize, h: usize) -> [f32; 4] {
    let (mut lo, mut hi) = ([w, h], [0, 0]);
    for (i, _) in m.iter().enumerate().filter(|(_, v)| **v) {
        let (x, y) = (i % w, i / w);
        lo = [lo[0].min(x), lo[1].min(y)];
        hi = [hi[0].max(x + 1), hi[1].max(y + 1)];
    }
    if hi[0] <= lo[0] {
        return CENTRE;
    }
    let (x0, y0) = ((lo[0] as f32 / w as f32 - 0.03).max(0.0), (lo[1] as f32 / h as f32 - 0.03).max(0.0));
    let (x1, y1) = ((hi[0] as f32 / w as f32 + 0.03).min(1.0), (hi[1] as f32 / h as f32 + 0.03).min(1.0));
    [x0, y0, x1 - x0, y1 - y0]
}

/// The main subject on a small copy: pixels far from every common colour along the frame's border (a
/// k-means palette of it; rare colours are likely the subject crossing the border), plus `focus` × local
/// detail, weighted toward the centre. The salient region with the most weight wins; None when nothing
/// stands out.
fn subject_region(s: &Rgb, focus: f32) -> Option<Vec<bool>> {
    let (w, h) = (s.w, s.h);
    if w < 16 || h < 16 {
        return None;
    }
    let lab = s.lab();
    let ch: [Vec<f32>; 3] = std::array::from_fn(|c| develop::blur(&lab.iter().map(|p| p[c]).collect::<Vec<_>>(), w, h, 1.5));
    let at = |i: usize| [ch[0][i], ch[1][i], ch[2][i]];
    let band = ((w.min(h) as f32 * 0.06) as usize).max(2);
    let border: Vec<Px> = (0..w * h)
        .filter(|i| {
            let (x, y) = (i % w, i / w);
            x < band || y < band || x + band >= w || y + band >= h
        })
        .step_by(3)
        .map(at)
        .collect();
    let pal: Vec<Px> = palette(&border, 8).into_iter().filter(|c| c.1 >= 0.04).map(|c| c.0).collect();
    let far = normalise(&(0..w * h).into_par_iter().map(|i| pal.iter().map(|c| dist(at(i), *c)).fold(f32::MAX, f32::min)).collect::<Vec<_>>(), 0.98);
    let detail = if focus > 0.0 {
        let g: Vec<f32> = sobel(&s.luma(), w, h).iter().map(|g| g[0].hypot(g[1])).collect();
        normalise(&develop::blur(&g, w, h, 0.02 * w.max(h) as f32), 0.98)
    } else {
        vec![0.0; w * h]
    };
    let sal: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32 - 0.5, (i / w) as f32 / h as f32 - 0.5);
            ((far[i] + focus * detail[i]) * (0.3 + 0.7 * (-(x * x + y * y) / 0.18).exp())).min(1.0)
        })
        .collect();
    let t = otsu(&sal);
    let (label, n) = components(&smoothed(&sal.iter().map(|&v| v > t).collect::<Vec<_>>(), w, h, 1.5), w, h);
    let (mut mass, mut size) = (vec![0f32; n], vec![0usize; n]);
    for (i, &l) in label.iter().enumerate().filter(|(_, l)| **l != NONE) {
        mass[l as usize] += sal[i];
        size[l as usize] += 1;
    }
    let best = (0..n).max_by(|a, b| mass[*a].total_cmp(&mass[*b]))?;
    let (inside, outside) = (mass[best] / size[best] as f32, (sal.iter().sum::<f32>() - mass[best]) / (w * h - size[best]).max(1) as f32);
    let share = size[best] as f32 / (w * h) as f32;
    if !(0.003..=0.75).contains(&share) || inside < 1.6 * outside.max(1e-3) {
        return None;
    }
    Some(label.iter().map(|&l| l == best as u32).collect())
}

/// The sky on a small copy: smooth, bright, blue / grey (or sunlit, near the top) pixels grown down from
/// the top edge without crossing an edge; holes that touch neither the bottom nor the sides (birds,
/// clouds) are filled. None when the sky is small or not at the top.
fn sky_region(s: &Rgb) -> Option<Vec<bool>> {
    let (w, h) = (s.w, s.h);
    if w < 16 || h < 16 {
        return None;
    }
    let lab = s.lab();
    let l: Vec<f32> = lab.iter().map(|q| q[0]).collect();
    let (m1, m2) = (develop::blur(&l, w, h, 2.0), develop::blur(&l.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, 2.0));
    let candidate: Vec<bool> = (0..w * h)
        .map(|i| {
            let [ll, a, b] = lab[i];
            let texture = (m2[i] - m1[i] * m1[i]).max(0.0).sqrt();
            let (blue, grey, lit) = (b < -0.02 && a < 0.05, a.hypot(b) < 0.045, i / w < h / 2 && ll > 0.7);
            texture < 0.025 && ll > 0.5 && (blue || grey || lit)
        })
        .collect();
    let mut sky = vec![false; w * h];
    let mut queue: VecDeque<usize> = (0..2 * w).filter(|&i| candidate[i]).collect();
    for &i in &queue {
        sky[i] = true;
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
            if !sky[j] && candidate[j] && (l[j] - l[i]).abs() < 0.03 {
                sky[j] = true;
                queue.push_back(j);
            }
        }
    }
    let (holes, n) = components(&sky.iter().map(|v| !v).collect::<Vec<_>>(), w, h);
    let (mut ground, mut size) = (vec![false; n], vec![0usize; n]);
    for (i, &k) in holes.iter().enumerate().filter(|(_, k)| **k != NONE) {
        let (x, y) = (i % w, i / w);
        size[k as usize] += 1;
        ground[k as usize] |= y + 1 == h || x == 0 || x + 1 == w;
    }
    for (i, &k) in holes.iter().enumerate() {
        if k != NONE && !ground[k as usize] && size[k as usize] < w * h / 25 {
            sky[i] = true;
        }
    }
    let sky = smoothed(&sky, w, h, 1.0);
    let (label, n) = components(&sky, w, h);
    let mut size = vec![0usize; n];
    let mut top = vec![false; n];
    for (i, &k) in label.iter().enumerate().filter(|(_, k)| **k != NONE) {
        size[k as usize] += 1;
        top[k as usize] |= i < w;
    }
    let best = (0..n).filter(|&k| top[k]).max_by_key(|&k| size[k])?;
    let centre_y = label.iter().enumerate().filter(|(_, k)| **k == best as u32).map(|(i, _)| (i / w) as f32).sum::<f32>() / size[best] as f32 / h as f32;
    if (size[best] as f32) < 0.02 * (w * h) as f32 || centre_y > 0.55 {
        return None;
    }
    Some(label.iter().map(|&k| k == best as u32).collect())
}

/// Moore-neighbour trace of the outer boundary of the region holding the first set pixel.
fn trace(m: &[bool], w: usize, h: usize) -> Vec<(i32, i32)> {
    // clockwise from west
    const D: [(i32, i32); 8] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)];
    let set = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && m[y as usize * w + x as usize];
    let Some(s) = m.iter().position(|&v| v) else { return vec![] };
    let start = ((s % w) as i32, (s / w) as i32);
    let (mut c, mut dir, mut second) = (start, 0usize, None);
    let mut out = vec![start];
    for _ in 0..4 * w * h + 8 {
        let Some(d) = (0..8).map(|k| (dir + k) % 8).find(|&d| set(c.0 + D[d].0, c.1 + D[d].1)) else { break };
        let next = (c.0 + D[d].0, c.1 + D[d].1);
        if c == start && second == Some(next) {
            break;
        }
        second.get_or_insert(next);
        // resume from the last background pixel checked, as seen from the new pixel
        dir = ((d & !1) + 6) % 8;
        c = next;
        out.push(c);
    }
    if out.len() > 1 && out.last() == Some(&start) {
        out.pop();
    }
    out
}

/// Douglas–Peucker on an open polyline (iterative), keeping points farther than `eps` from the chords.
fn douglas(p: &[(i32, i32)], eps: f32) -> Vec<(i32, i32)> {
    let n = p.len();
    if n < 3 {
        return p.to_vec();
    }
    let seg = |q: (i32, i32), a: (i32, i32), b: (i32, i32)| {
        let (dx, dy) = ((b.0 - a.0) as f32, (b.1 - a.1) as f32);
        let len = dx.hypot(dy);
        if len < 1e-6 { ((q.0 - a.0) as f32).hypot((q.1 - a.1) as f32) } else { (dy * (q.0 - a.0) as f32 - dx * (q.1 - a.1) as f32).abs() / len }
    };
    let mut keep = vec![false; n];
    (keep[0], keep[n - 1]) = (true, true);
    let mut stack = vec![(0, n - 1)];
    while let Some((s, e)) = stack.pop() {
        let (i, d) = (s + 1..e).map(|i| (i, seg(p[i], p[s], p[e]))).fold((s, -1.0), |m, x| if x.1 > m.1 { x } else { m });
        if d > eps {
            keep[i] = true;
            stack.extend([(s, i), (i, e)]);
        }
    }
    p.iter().zip(&keep).filter(|(_, k)| **k).map(|(q, _)| *q).collect()
}

/// Outer boundary of the largest region of `m` as an outline of at most `max` points, as fractions of the
/// frame; boundary pixels on the frame's edge snap to it.
fn outline(m: &[bool], w: usize, h: usize, max: usize) -> Option<Vec<[f32; 2]>> {
    let (label, n) = components(m, w, h);
    let mut size = vec![0usize; n];
    for &k in label.iter().filter(|k| **k != NONE) {
        size[k as usize] += 1;
    }
    let best = (0..n).max_by_key(|&k| size[k])? as u32;
    ring(&label.iter().map(|&k| k == best).collect::<Vec<_>>(), w, h, max)
}

/// The boundary of the one region in `m`, simplified to at most `max` vertices (fractions of the frame).
fn ring(m: &[bool], w: usize, h: usize, max: usize) -> Option<Vec<[f32; 2]>> {
    let ring = trace(m, w, h);
    if ring.len() < 8 {
        return None;
    }
    // closed ring: split at the point farthest from the first, simplify both halves
    let far = (0..ring.len()).max_by_key(|&i| (ring[i].0 - ring[0].0).pow(2) + (ring[i].1 - ring[0].1).pow(2)).unwrap_or(0);
    let mut eps = 0.75;
    let poly = loop {
        let mut a = douglas(&ring[..=far], eps);
        let b = douglas(&[&ring[far..], &ring[..1]].concat(), eps);
        a.pop();
        a.extend(&b[..b.len() - 1]);
        if a.len() <= max {
            break a;
        }
        eps *= 1.4;
    };
    let at = |v: i32, n: usize| if v <= 0 { 0.0 } else if v as usize + 1 >= n { 1.0 } else { (v as f32 + 0.5) / n as f32 };
    let out: Vec<[f32; 2]> = poly.into_iter().map(|(x, y)| [at(x, w), at(y, h)]).collect();
    (out.len() >= 3).then_some(out)
}

/// Every part of `m` covering at least `min_frac` of the frame (largest first, at most `max_parts`) as an
/// outline (`false`), then every hole in them that large (`true`: enclosed background, off the frame's
/// edge), each simplified to `max` vertices (fractions of the frame). As mask components: parts add, holes
/// subtract.
pub(crate) fn regions(m: &[bool], w: usize, h: usize, max: usize, min_frac: f32, max_parts: usize) -> Vec<(bool, Vec<[f32; 2]>)> {
    let min = (((w * h) as f32 * min_frac) as usize).max(8);
    let mut out = vec![];
    for hole in [false, true] {
        let src: Vec<bool> = if hole { m.iter().map(|&v| !v).collect() } else { m.to_vec() };
        let (label, n) = components(&src, w, h);
        let (mut size, mut edge) = (vec![0usize; n], vec![false; n]);
        for (i, &k) in label.iter().enumerate().filter(|(_, k)| **k != NONE) {
            size[k as usize] += 1;
            let (x, y) = (i % w, i / w);
            edge[k as usize] |= x == 0 || y == 0 || x + 1 == w || y + 1 == h;
        }
        let mut keep: Vec<usize> = (0..n).filter(|&k| size[k] >= min && !(hole && edge[k])).collect();
        keep.sort_by_key(|&k| std::cmp::Reverse(size[k]));
        for k in keep.into_iter().take(max_parts) {
            if let Some(r) = ring(&label.iter().map(|&l| l == k as u32).collect::<Vec<_>>(), w, h, max) {
                out.push((hole, r));
            }
        }
    }
    out
}

/// Outline of the main subject (for an AI-free subject mask): colour against the border, centre and focus.
pub fn subject_outline(s: &Rgb) -> Option<Vec<[f32; 2]>> {
    outline(&subject_region(s, 0.35)?, s.w, s.h, 160)
}

/// Outline of the sky, or None when there is little or none.
pub fn sky_outline(s: &Rgb) -> Option<Vec<[f32; 2]>> {
    outline(&sky_region(s)?, s.w, s.h, 160)
}

/// Generates a 16-bit feathered alpha mask (`0..=65535`) for `subject`, `sky`, or `background`
/// using the native C++ O(N) He et al. edge-preserving Guided Filter (`ol_guided_feather_mask_16bit`).
pub fn semantic_mask_16bit(s: &Rgb, target: &str, feather: f32) -> Option<Vec<u16>> {
    let raw_bool = match target {
        "subject" => subject_region(s, 0.35)?,
        "sky" => sky_region(s)?,
        "background" => subject_region(s, 0.35)?.into_iter().map(|b| !b).collect(),
        _ => return None,
    };
    let raw_alpha: Vec<f32> = raw_bool.into_iter().map(|b| if b { 1.0f32 } else { 0.0f32 }).collect();
    let guide_luma = s.luma();
    let radius = ((feather.clamp(0.0, 100.0) * 0.25).round() as i32).max(4);
    let eps = 1e-3f32;
    let (a16, _af) = crate::libraw::guided_feather_16bit(&guide_luma, &raw_alpha, s.w, s.h, radius, eps);
    Some(a16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// "Dead leaves": overlapping discs of random greys, 4× supersampled, so edges run every way.
    fn leaves(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng(seed);
        let discs: Vec<(f32, f32, f32, f32)> = (0..400).map(|_| (r.next() * w as f32, r.next() * h as f32, 4.0 + 40.0 * r.next().powi(3), 0.1 + 0.8 * r.next())).collect();
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                let mut acc = 0.0;
                for s in 0..16 {
                    let (x, y) = ((i % w) as f32 + (s % 4) as f32 / 4.0 + 0.125, (i / w) as f32 + (s / 4) as f32 / 4.0 + 0.125);
                    acc += discs.iter().rev().find(|d| (x - d.0).hypot(y - d.1) < d.2).map_or(0.5, |d| d.3);
                }
                acc / 16.0
            })
            .collect()
    }

    fn grey(v: &[f32], w: usize, h: usize) -> DynamicImage {
        let rgb: Vec<u8> = v.iter().flat_map(|&g| [(g.clamp(0.0, 1.0) * 255.0).round() as u8; 3]).collect();
        DynamicImage::ImageRgb8(image::RgbImage::from_raw(w as u32, h as u32, rgb).unwrap())
    }

    /// Horizontal box blur of length `n` (camera moved sideways).
    fn smear(v: &[f32], w: usize, h: usize, n: usize) -> Vec<f32> {
        (0..w * h).map(|i| (0..n).map(|k| v[(i / w) * w + (i % w + k).saturating_sub(n / 2).min(w - 1)]).sum::<f32>() / n as f32).collect()
    }

    #[test]
    fn edge_blur_reads_the_blur_radius() {
        let (w, h) = (640, 480);
        let sharp = leaves(w, h, 7);
        let at = |v: &[f32]| analyse(&grey(v, w, h), 0, 0, Some([0.0, 0.0, 1.0, 1.0]), &[]);
        let m0 = at(&sharp);
        assert!(m0.sharpness > 90.0 && m0.blur_px < 0.8, "sharp: {m0:?}");
        // close edges (thin leaves) read a little low at large blur; the reading stays monotone
        for (sigma, lo, hi) in [(2.0, 1.5, 2.2), (3.5, 2.4, 3.6)] {
            let m = at(&gauss(&sharp, w, h, sigma));
            assert!((lo..hi).contains(&m.subject_blur_px), "σ {sigma}: measured {} ({m:?})", m.subject_blur_px);
            assert!(m.motion < 0.3, "isotropic blur read as motion: {}", m.motion);
        }
        assert!(at(&gauss(&sharp, w, h, 2.0)).sharpness < 60.0);
        // measured at half size and judged at full size, σ 2 still reads about 2
        let half = analyse(&grey(&gauss(&sharp, w, h, 2.0), w, h), 320, 640, Some([0.0, 0.0, 1.0, 1.0]), &[]);
        assert!((1.2..2.6).contains(&half.subject_blur_px) && half.subject_sharpness < 60.0, "{half:?}");
    }

    #[test]
    fn motion_blur_is_directional() {
        let (w, h) = (640, 480);
        let m = analyse(&grey(&smear(&leaves(w, h, 3), w, h, 11), w, h), 0, 0, Some([0.0, 0.0, 1.0, 1.0]), &[]);
        assert!(m.motion > 0.4 && (m.motion_angle == 0.0 || m.motion_angle == 180.0), "{m:?}");
        assert!(m.subject_sharpness < 60.0, "a smeared frame passed as sharp: {m:?}");
    }

    #[test]
    fn a_blurred_subject_in_a_sharp_frame_is_found() {
        // a textured red disc (the subject) in front of grey leaves; either layer out of focus
        let (w, h) = (640, 480);
        let (ground, skin) = (leaves(w, h, 11), leaves(w, h, 12));
        let disc: Vec<f32> = (0..w * h).map(|i| (((i % w) as f32 - 320.0).hypot((i / w) as f32 - 240.0) < 110.0) as u8 as f32).collect();
        let soft = |v: &[f32]| gauss(v, w, h, 3.5);
        let paint = |bg: &[f32], subject: &[f32], alpha: &[f32]| -> DynamicImage {
            let rgb: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let (a, g) = (alpha[i], bg[i] * 255.0);
                    [a * 200.0 + (1.0 - a) * g, a * (40.0 + 150.0 * subject[i]) + (1.0 - a) * g, a * 40.0 + (1.0 - a) * g].map(|v| v as u8)
                })
                .collect();
            DynamicImage::ImageRgb8(image::RgbImage::from_raw(w as u32, h as u32, rgb).unwrap())
        };
        let back_focus = analyse(&paint(&ground, &soft(&skin), &soft(&disc)), 0, 0, None, &[]);
        let b = back_focus.subject;
        assert!(b[0] < 0.36 && b[0] + b[2] > 0.64 && b[1] < 0.3 && b[1] + b[3] > 0.7 && b[2] < 0.6, "subject box {b:?}");
        assert!(back_focus.sharpness > 80.0 && back_focus.subject_sharpness < 40.0, "{back_focus:?}");
        let portrait = analyse(&paint(&soft(&ground), &skin, &disc), 0, 0, None, &[]);
        assert!(portrait.sharpness > 80.0 && portrait.subject_sharpness > 80.0, "{portrait:?}");
    }

    #[test]
    fn noise_exposure_colour_and_hashes() {
        let (w, h) = (400, 300);
        let mut r = Rng(5);
        let gauss_noise = |r: &mut Rng| (-2.0 * (r.next() + 1e-6).ln()).sqrt() * (2.0 * PI * r.next()).cos();
        let noisy: Vec<f32> = (0..w * h).map(|_| 0.5 + 0.02 * gauss_noise(&mut r)).collect();
        let n = noise(&noisy, w, h);
        assert!((n - 0.02).abs() < 0.003, "noise {n}");
        let texture = leaves(w, h, 9);
        assert!(noise(&texture, w, h) < 0.004, "texture read as noise: {}", noise(&texture, w, h));
        // a frame blown out on its right third
        let m = analyse(&grey(&(0..w * h).map(|i| if i % w > 266 { 1.0 } else { 0.4 }).collect::<Vec<_>>(), w, h), 0, 0, None, &[]);
        assert!((m.clipped_highlights - 33.0).abs() < 2.0 && m.exposure < 10.0, "{m:?}");
        assert!(m.colorfulness < 5.0 && m.colors.is_empty());
        // the same frame resized and brightened hashes close; another frame does not
        let a = grey(&texture, w, h);
        let b = DynamicImage::ImageRgb8(image::imageops::brighten(&a.resize(300, 225, image::imageops::FilterType::Triangle).to_rgb8(), 12));
        let c = grey(&leaves(w, h, 10), w, h);
        let hash = |i: &DynamicImage| format!("{:016x}", phash_of(i));
        assert!(distance(&hash(&a), &hash(&b)) <= 6, "near copy: {}", distance(&hash(&a), &hash(&b)));
        assert!(distance(&hash(&a), &hash(&c)) >= 18, "different photo: {}", distance(&hash(&a), &hash(&c)));
        assert_eq!(distance("zz", &hash(&a)), 64);
    }

    /// A ring traces as one part (to add) and one hole (to subtract) in the middle; a speck under the minimum
    /// area is left out.
    #[test]
    fn regions_trace_parts_and_holes() {
        let (w, h) = (64usize, 48usize);
        let m: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 32.0, (i / w) as f32 - 24.0);
                (8.0..18.0).contains(&x.hypot(y)) || (i % w == 2 && i / w == 2)
            })
            .collect();
        let r = regions(&m, w, h, 64, 0.01, 12);
        assert_eq!((r.iter().filter(|p| !p.0).count(), r.iter().filter(|p| p.0).count()), (1, 1), "{r:?}");
        let hole = &r.iter().find(|p| p.0).unwrap().1;
        let c = hole.iter().fold([0.0f32; 2], |a, p| [a[0] + p[0], a[1] + p[1]]).map(|v| v / hole.len() as f32);
        assert!((c[0] - 0.5).abs() < 0.05 && (c[1] - 0.5).abs() < 0.05, "{c:?}");
    }

    #[test]
    fn sky_and_subject_outlines() {
        // blue sky over a green field with a textured horizon, and an orange ball on the grass
        let (w, h) = (480, 320);
        let mut r = Rng(1);
        let px: Vec<Px> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let horizon = 120.0 + 12.0 * (x / 25.0).sin();
                if (x - 240.0).hypot(y - 230.0) < 45.0 {
                    [0.95, 0.55, 0.1]
                } else if y < horizon {
                    [0.35 + 0.1 * y / horizon, 0.6, 0.92]
                } else {
                    [0.15 + 0.1 * r.next(), 0.45 + 0.1 * r.next(), 0.12]
                }
            })
            .collect();
        let s = Rgb { w, h, px };
        let inside = |poly: &[[f32; 2]], p: [f32; 2]| {
            let mut c = false;
            for k in 0..poly.len() {
                let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
                if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
                    c = !c;
                }
            }
            c
        };
        let sky = sky_outline(&s).expect("a sky");
        assert!(sky.len() <= 160);
        for p in [[0.1, 0.1], [0.5, 0.2], [0.9, 0.25], [0.02, 0.02]] {
            assert!(inside(&sky, p), "sky misses {p:?}");
        }
        for p in [[0.5, 0.6], [0.1, 0.9], [0.5, 0.72]] {
            assert!(!inside(&sky, p), "sky spills to {p:?}");
        }
        let ball = subject_outline(&s).expect("a subject");
        assert!(inside(&ball, [0.5, 0.72]) && !inside(&ball, [0.2, 0.9]) && !inside(&ball, [0.5, 0.2]), "{ball:?}");
        // no sky in a frame of grass
        let grass = Rgb { w, h, px: (0..w * h).map(|_| [0.15 + 0.1 * r.next(), 0.45 + 0.1 * r.next(), 0.12]).collect() };
        assert!(sky_outline(&grass).is_none());
    }

    #[test]
    fn trace_follows_a_square() {
        let (w, h) = (10, 8);
        let m: Vec<bool> = (0..w * h).map(|i| (2..7).contains(&(i % w)) && (1..5).contains(&(i / w))).collect();
        let ring = trace(&m, w, h);
        assert_eq!(ring.len(), 14, "{ring:?}"); // 5 + 3 + 5 + 3 boundary pixels, corners once
        assert_eq!(ring[0], (2, 1));
        let poly = outline(&m, w, h, 10).unwrap();
        assert_eq!(poly.len(), 4, "{poly:?}");
    }
}
