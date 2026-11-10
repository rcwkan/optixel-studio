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
    fn disc(m: &mut [bool], w: usize, h: usize, cx: f32, cy: f32, r: f32) {
        let (x0, x1) = ((cx - r).floor().max(0.0) as usize, ((cx + r).ceil().max(0.0) as usize).min(w - 1));
        let (y0, y1) = ((cy - r).floor().max(0.0) as usize, ((cy + r).ceil().max(0.0) as usize).min(h - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) <= r * r {
                    m[y * w + x] = true;
                }
            }
        }
    }
    let mut m = vec![false; w * h];
    for s in strokes {
        let r = (s.radius * w as f32).max(1.0);
        let px: Vec<[f32; 2]> = s.points.iter().map(|p| [p[0] * w as f32, p[1] * h as f32]).collect();
        if let [only] = px[..] {
            disc(&mut m, w, h, only[0], only[1], r);
        }
        for pair in px.windows(2) {
            let steps = ((pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]) / (r * 0.5).max(1.0)).ceil().max(1.0) as usize;
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
    let painted = paint(strokes, w, h);
    if !painted.iter().any(|&p| p) {
        return Err("paint over the object first".into());
    }
    let (win, a) = develop::snap_painted(&painted, &img.px, w, h, 0.5);
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
}
