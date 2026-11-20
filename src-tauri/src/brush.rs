//! Brush strokes to a selection: the painted area, with its edges moved onto the picture's own edges
//! where the colours change (the same snapping the AI selections use).
//! ponytail: the snap only refines the edge of what was painted (a band around it); it cannot grow a
//! scribble over part of an object into the whole object. A Segment Anything model would do that, but the
//! ONNX exports tried here returned the whole frame for any prompt (see git history), so paint the object.

use crate::develop::{self, Img};
use serde::{Deserialize, Serialize};

/// One brush stroke: the path (fractions of the frame) and the brush radius (fraction of its width).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Stroke {
    pub points: Vec<[f32; 2]>,
    pub radius: f32,
}

/// The painted area of the strokes over a `w` × `h` frame.
pub fn paint(strokes: &[Stroke], w: usize, h: usize) -> Vec<bool> {
    if w == 0 || h == 0 {
        return vec![];
    }
    fn disc(m: &mut [bool], w: usize, h: usize, cx: f32, cy: f32, r: f32) {
        if !cx.is_finite() || !cy.is_finite() || !r.is_finite() {
            return;
        }
        let (x0, x1) = ((cx - r).floor().max(0.0).min((w - 1) as f32) as usize, ((cx + r).ceil().max(0.0) as usize).min(w - 1));
        let (y0, y1) = ((cy - r).floor().max(0.0).min((h - 1) as f32) as usize, ((cy + r).ceil().max(0.0) as usize).min(h - 1));
        if x0 > x1 || y0 > y1 {
            return;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) <= r * r {
                    m[y * w + x] = true;
                }
            }
        }
    }
    let max_steps = w.saturating_add(h).saturating_mul(4).max(1);
    let mut m = vec![false; w * h];
    for s in strokes {
        if !s.radius.is_finite() {
            continue;
        }
        let r = (s.radius.clamp(0.0, 1.0) * w as f32).max(1.0);
        let px: Vec<[f32; 2]> = s
            .points
            .iter()
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .map(|p| [p[0].clamp(-0.5, 1.5) * w as f32, p[1].clamp(-0.5, 1.5) * h as f32])
            .collect();
        if let [only] = px[..] {
            disc(&mut m, w, h, only[0], only[1], r);
        }
        for pair in px.windows(2) {
            let steps = (((pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]) / (r * 0.5).max(1.0)).ceil().max(1.0) as usize).min(max_steps);
            for k in 0..=steps {
                let t = k as f32 / steps as f32;
                disc(&mut m, w, h, pair[0][0] + (pair[1][0] - pair[0][0]) * t, pair[0][1] + (pair[1][1] - pair[0][1]) * t, r);
            }
        }
    }
    m
}

/// The selection (0-1 over the frame of `img`) for the strokes, edges snapped to the picture.
pub fn snap(img: &Img, strokes: &[Stroke]) -> Result<Vec<f32>, String> {
    let (w, h) = (img.w, img.h);
    if w == 0 || h == 0 {
        return Err("empty image".into());
    }
    let painted = paint(strokes, w, h);
    if !painted.iter().any(|&p| p) {
        return Err("paint over the object first".into());
    }
    let thin = strokes
        .iter()
        .filter(|s| s.radius.is_finite())
        .map(|s| 2.0 * (s.radius.clamp(0.0, 1.0) * w as f32).max(1.0))
        .fold(f32::MAX, f32::min)
        .min(w.max(h) as f32);
    let (win, a) = develop::snap_painted(&painted, &img.px, w, h, 0.5, thin);
    let mut out = vec![0.0f32; w * h];
    for y in 0..win.h {
        for x in 0..win.w {
            out[(win.y0 + y) * w + win.x0 + x] = a[y * win.w + x];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clicks spread over the frame once vanished: the snap's blur was sized by the box around all of them and washed each one out.
    #[test]
    fn scattered_clicks_stay_selected() {
        let (w, h) = (1600, 1067); // the proxy's size: the blur's cap scales with it
        let img = Img { w, h, px: vec![[0.2, 0.3, 0.5]; w * h] };
        for radius in [0.005, 0.012, 0.05] {
            let strokes: Vec<Stroke> = [[0.2, 0.2], [0.7, 0.4], [0.4, 0.8]].iter().map(|&p| Stroke { points: vec![p], radius }).collect();
            let a = snap(&img, &strokes).unwrap();
            for s in &strokes {
                let (x, y) = ((s.points[0][0] * w as f32) as usize, (s.points[0][1] * h as f32) as usize);
                assert!(a[y * w + x] > 0.9, "radius {radius}: click at ({x}, {y}) vanished: {}", a[y * w + x]);
            }
        }
    }

    /// OPTIXEL_TEST_RAW=<file> OPTIXEL_TEST_OUT=<png> OPTIXEL_TEST_STROKE=x1,y1,x2,y2,radius (fractions): snaps a stroke.
    #[test]
    fn snap_real_photo() {
        let file = std::env::var_os("OPTIXEL_TEST_RAW");
        let out = std::env::var_os("OPTIXEL_TEST_OUT");
        let stroke = std::env::var("OPTIXEL_TEST_STROKE");
        let (Some(file), Some(out), Ok(st)) = (file, out, stroke) else { return };
        let v: Vec<f32> = st.split(',').map(|s| s.parse().unwrap()).collect();
        let img = develop::load(std::path::Path::new(&file), Some(1600)).unwrap();
        let (w, h) = (img.w, img.h);
        let strokes = vec![Stroke { points: vec![[v[0], v[1]], [(v[0] + v[2]) / 2.0, (v[1] + v[3]) / 2.0], [v[2], v[3]]], radius: v[4] }];
        let t0 = std::time::Instant::now();
        let a = snap(&img, &strokes).unwrap();
        eprintln!("snap in {:?}", t0.elapsed());
        let painted = paint(&strokes, w, h);
        let mut buf = image::RgbImage::new(w as u32, h as u32);
        for i in 0..w * h {
            let g = crate::aimodel::encode(img.px[i]);
            let k = 0.55 * a[i];
            let mut c = [g[0] * (1.0 - k) + 0.9 * k, g[1] * (1.0 - k), g[2] * (1.0 - k)];
            if painted[i] {
                c = [c[0].max(1.0), c[1] * 0.6, c[2] * 0.6];
            }
            buf.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb([(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8]));
        }
        buf.save(out).unwrap();
    }

    #[test]
    fn handles_empty_or_non_finite_strokes_safely() {
        let s = vec![Stroke { points: vec![[f32::NAN, 0.5], [1e20, -1e20]], radius: f32::INFINITY }];
        assert!(paint(&s, 0, 0).is_empty());
        let extreme = vec![Stroke { points: vec![[-100.0, 0.5], [100.0, 0.5]], radius: 1e10 }];
        let m = paint(&extreme, 16, 16);
        assert_eq!(m.len(), 256);
    }
}
