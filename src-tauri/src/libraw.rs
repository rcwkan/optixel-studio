//! LibRaw (C++, vendored in vendor/libraw, built by build.rs) + clean-room C++ demosaics (RCD, AMaZE,
//! LMMSE), optical black-point calibration, MakerNotes & active AF-point extraction, real-time scopes,
//! focus-peaking/blink EAR, and 16-bit guided mask feathering.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, c_char, c_int, c_void};

#[repr(C)]
struct Opts {
    demosaic: c_int,
    highlight: c_int,
    proxy_edge: c_int,
    black_offset: c_int,
    raw_denoise: f32,
}

#[repr(C)]
struct Out {
    handle: *mut c_void,
    px: *const u16,
    width: c_int,
    height: c_int,
    colors: c_int,
    gain: f32,
    error: [c_char; 160],
}

const MAX_AF_POINTS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeAfPoint {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    active: u8,
    kind: u8,
}

#[repr(C)]
struct NativeMakerNotesOut {
    shutter_count: u32,
    af_count: u32,
    af_points: [NativeAfPoint; MAX_AF_POINTS],
    focus_distance_m: f32,
    is_xtrans: u8,
    vendor_id: u8,
    lens_serial: [c_char; 64],
    body_serial: [c_char; 64],
    camera_model: [c_char; 64],
    cfa_pattern: [c_char; 16],
    black_levels: [u32; 4],
    border_black_counts: [u32; 4],
    white_level: u32,
    linear_max: [u32; 4],
    cam_mul: [f32; 4],
    raw_width: u32,
    raw_height: u32,
}

// csrc/ol_raw.cpp & csrc/ol_demosaic.cpp
unsafe extern "C" {
    fn ol_raw_develop(data: *const u8, len: usize, o: *const Opts, out: *mut Out) -> c_int;
    fn ol_raw_free(handle: *mut c_void);
    fn ol_raw_camera_matrix(data: *const u8, len: usize, out: *mut f32) -> c_int;
    fn ol_parse_makernotes(data: *const u8, len: usize, out: *mut NativeMakerNotesOut) -> c_int;
    fn ol_scope_bins(rgba: *const u8, w: usize, h: usize, scope_w: usize, scope_h: usize, parade: *mut u32, waveform: *mut u32);
    fn ol_compute_focus_peaking(
        rgb: *const f32,
        w: usize,
        h: usize,
        threshold: f32,
        eyes_xywh: *const f32,
        num_faces: usize,
        out_peaking_mask: *mut u8,
        out_focus_score: *mut f32,
    );
    fn ol_guided_feather_mask_16bit(
        guide_luma: *const f32,
        raw_alpha: *const f32,
        w: usize,
        h: usize,
        radius: c_int,
        eps: f32,
        out_alpha16: *mut u16,
        out_alpha_f32: *mut f32,
    );
    fn ol_decode_heif_container(
        data: *const u8,
        len: usize,
        max_edge: c_int,
        out_rgb: *mut *mut u16,
        out_w: *mut u32,
        out_h: *mut u32,
        out_icc: *mut *mut u8,
        out_icc_len: *mut usize,
        err_buf: *mut c_char,
    ) -> c_int;
    fn ol_free_buf(p: *mut c_void);
}

/// Demosaic names (as `develop::Raw` stores them) and their engine codes.
/// - `rcd`: Ratio Corrected Demosaicing (general sharpness)
/// - `amaze`: Aliasing Minimization and Zipper Elimination (low-ISO maze suppression)
/// - `lmmse`: Linear Minimum Mean Square Error (high-ISO astro / night photography)
/// - `linear`..`aahd`: LibRaw's built-in Bayer & X-Trans Markesteijn demosaics.
pub const DEMOSAIC: &[(&str, i32)] = &[
    ("rcd", 101),
    ("amaze", 103),
    ("lmmse", 102),
    ("linear", 0),
    ("vng", 1),
    ("ppg", 2),
    ("ahd", 3),
    ("dcb", 4),
    ("dht", 11),
    ("aahd", 12),
];

/// Highlight modes and LibRaw's numbers: rebuild is 5, halfway between its whitest (3) and most colourful (9).
pub const HIGHLIGHTS: &[(&str, i32)] = &[("clip", 0), ("unclip", 1), ("blend", 2), ("rebuild", 5)];

fn pick(table: &[(&str, i32)], name: &str, default: i32) -> i32 {
    table.iter().find(|(n, _)| *n == name).map_or(default, |(_, v)| *v)
}

/// Active / candidate autofocus point decoded from camera MakerNotes (normalized 0..1 coordinates).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct AfPoint {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub active: bool,
    /// `"single"`, `"eye"`, `"face"`, or `"zone"`.
    pub kind: String,
}

/// Sensor optical calibration & MakerNotes metadata decoded from a RAW or JPEG file.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RawInfo {
    pub camera_model: String,
    pub vendor: String,
    pub cfa_pattern: String,
    pub is_xtrans: bool,
    pub raw_width: u32,
    pub raw_height: u32,
    pub black_levels: [u32; 4],
    pub border_black_counts: [u32; 4],
    pub white_level: u32,
    pub linear_max: [u32; 4],
    pub cam_wb: [f32; 4],
    pub shutter_count: u32,
    pub lens_serial: String,
    pub body_serial: String,
    pub focus_distance_m: f32,
    pub af_points: Vec<AfPoint>,
}

fn cstr_buf(buf: &[c_char]) -> String {
    // SAFETY: buffers in NativeMakerNotesOut are zero-initialized and written with bounded snprintf.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().trim().to_string()
}

/// Inspect a RAW (or JPEG/TIFF) buffer for sensor optical black/white levels, CFA pattern,
/// MakerNotes shutter count, lens/body serial numbers, and active AF points.
pub fn inspect(file: &[u8]) -> Result<RawInfo, String> {
    if file.len() < 16 {
        return Err("file too small".into());
    }
    let mut mn = std::mem::MaybeUninit::<NativeMakerNotesOut>::zeroed();
    let (rc, mn) = unsafe {
        (ol_parse_makernotes(file.as_ptr(), file.len(), mn.as_mut_ptr()), mn.assume_init())
    };
    if rc != 0 {
        return Err("unable to parse RAW/MakerNotes header".into());
    }
    let vendor = match mn.vendor_id {
        1 => "Canon",
        2 => "Nikon",
        3 => "Sony",
        4 => "Fujifilm",
        5 => "Olympus / OM",
        6 => "Panasonic",
        7 => "Pentax / Ricoh",
        _ => "Unknown",
    }
    .to_string();
    let count = (mn.af_count as usize).min(MAX_AF_POINTS);
    let af_points = mn.af_points[..count]
        .iter()
        .map(|p| AfPoint {
            x: p.x,
            y: p.y,
            w: p.w,
            h: p.h,
            active: p.active != 0,
            kind: match p.kind {
                1 => "eye",
                2 => "face",
                3 => "zone",
                _ => "single",
            }
            .to_string(),
        })
        .collect();
    Ok(RawInfo {
        camera_model: cstr_buf(&mn.camera_model),
        vendor,
        cfa_pattern: cstr_buf(&mn.cfa_pattern),
        is_xtrans: mn.is_xtrans != 0,
        raw_width: mn.raw_width,
        raw_height: mn.raw_height,
        black_levels: mn.black_levels,
        border_black_counts: mn.border_black_counts,
        white_level: mn.white_level,
        linear_max: mn.linear_max,
        cam_wb: mn.cam_mul,
        shutter_count: mn.shutter_count,
        lens_serial: cstr_buf(&mn.lens_serial),
        body_serial: cstr_buf(&mn.body_serial),
        focus_distance_m: mn.focus_distance_m,
        af_points,
    })
}

type M3 = crate::develop::M3;

fn matmul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn inverse(m: &M3) -> Option<M3> {
    let c = |i: usize, j: usize| m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3];
    let det: f32 = (0..3).map(|j| m[0][j] * c(0, j)).sum();
    // the transposed cofactors over the determinant
    (det.abs() > 1e-8).then(|| std::array::from_fn(|i| std::array::from_fn(|j| c(j, i) / det)))
}

/// The working space (linear Rec.2020) and a RAW's own white-balanced camera RGB, by the camera's matrix
/// (LibRaw's, from the file's metadata): (working → camera, camera → working). An AI denoiser sees noise there
/// as the sensor made it, before the colour matrix mixes the channels'. None when LibRaw has no matrix.
pub fn camera_space(file: &[u8]) -> Option<(M3, M3)> {
    let mut m = [0f32; 9];
    // SAFETY: `file` outlives the call; the shim writes 9 floats into `m`
    if unsafe { ol_raw_camera_matrix(file.as_ptr(), file.len(), m.as_mut_ptr()) } != 0 {
        return None;
    }
    let cam_to_srgb = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
    let from = matmul(&crate::develop::M709_2020, &cam_to_srgb);
    Some((inverse(&from)?, from))
}

/// Develops the RAW `file` (its bytes) to (width, height, interleaved linear Rec.2020), upright. Unknown
/// names mean LibRaw's defaults (AHD, clip). With `proxy_edge`, a file at least twice that size comes
/// back at half size without demosaicing, which is several times faster.
#[allow(dead_code)]
pub fn develop(file: &[u8], demosaic: &str, highlights: &str, proxy_edge: Option<u32>) -> Result<(usize, usize, Vec<f32>), String> {
    develop_ex(file, demosaic, highlights, 0, 0.0, proxy_edge)
}

/// Extended RAW develop with optional optical black-point ADU offset and linear sensor denoising.
pub fn develop_ex(
    file: &[u8],
    demosaic: &str,
    highlights: &str,
    black_offset: i32,
    raw_denoise: f32,
    proxy_edge: Option<u32>,
) -> Result<(usize, usize, Vec<f32>), String> {
    let o = Opts {
        demosaic: pick(DEMOSAIC, demosaic, -1),
        highlight: pick(HIGHLIGHTS, highlights, 0),
        proxy_edge: proxy_edge.map_or(0, |e| e.min(1 << 30) as c_int),
        black_offset: black_offset as c_int,
        raw_denoise: raw_denoise.clamp(0.0, 1.0),
    };
    let mut out = std::mem::MaybeUninit::<Out>::zeroed();
    // SAFETY: `file` and `o` outlive the call, and the shim writes nothing but `out`. All-zero bytes are a
    // valid `Out`, so it is initialised whether or not the shim got as far as filling it.
    let (rc, out) = unsafe { (ol_raw_develop(file.as_ptr(), file.len(), &o, out.as_mut_ptr()), out.assume_init()) };
    if rc != 0 {
        // SAFETY: the shim zeroes `error` and then writes it with snprintf, so it is NUL-terminated.
        return Err(format!("LibRaw: {}", unsafe { CStr::from_ptr(out.error.as_ptr()) }.to_string_lossy()));
    }
    let (w, h, n) = (out.width as usize, out.height as usize, out.colors as usize);
    // SAFETY: on success `px` points at w * h * colors u16 samples (16-bit aligned, 16 bytes into a
    // malloc'd block) that stay alive until `ol_raw_free` below.
    let samples = unsafe { std::slice::from_raw_parts(out.px, w * h * n) };
    let k = out.gain / 65535.0;
    let data: Vec<f32> = if n == 3 {
        samples.par_iter().map(|&v| v as f32 * k).collect()
    } else {
        samples.par_iter().flat_map_iter(|&v| [v as f32 * k; 3]).collect()
    };
    // SAFETY: `handle` came from the successful call above and is freed once.
    unsafe { ol_raw_free(out.handle) };
    Ok((w, h, data))
}

/// RGB parade and luma waveform scopes as RGBA8 images (`scope_w` x `scope_h` x 4 bytes each): the C++
/// fallback for `gpu::run_scopes`, drawing exactly what it draws.
pub fn compute_scopes_cpu(rgba: &[u8], w: usize, h: usize, scope_w: usize, scope_h: usize) -> (Vec<u8>, Vec<u8>) {
    let mut parade = vec![0u32; scope_w * scope_h * 3];
    let mut wave = vec![0u32; scope_w * scope_h * 4];
    if rgba.len() >= w * h * 4 && w > 0 && h > 0 && scope_w >= 3 && scope_h >= 2 {
        // SAFETY: `rgba` holds w * h pixels and the bins are the sizes `ol_scope_bins` fills
        unsafe { ol_scope_bins(rgba.as_ptr(), w, h, scope_w, scope_h, parade.as_mut_ptr(), wave.as_mut_ptr()) };
    }
    crate::gpu::render_scopes(&parade, &wave, scope_w, scope_h, w * h)
}

/// Focus-peaking mask (`0..255`) and focus score (`0..100`): of the eyes when `eyes` (per face, its two eye
/// patches: `faces::eye_patches`) has any, else of the frame.
pub fn focus_peaking(rgb: &[f32], w: usize, h: usize, threshold: f32, eyes: &[[f32; 8]]) -> (Vec<u8>, f32) {
    let mut mask = vec![0u8; w * h];
    let mut score = 0.0f32;
    if rgb.len() >= w * h * 3 && w >= 3 && h >= 3 {
        // SAFETY: `rgb` holds w * h pixels, `eyes` its own length of 8-float faces, `mask` w * h bytes
        unsafe {
            ol_compute_focus_peaking(
                rgb.as_ptr(),
                w,
                h,
                threshold,
                if eyes.is_empty() { std::ptr::null() } else { eyes.as_ptr() as *const f32 },
                eyes.len(),
                mask.as_mut_ptr(),
                &mut score,
            );
        }
    }
    (mask, score)
}

/// O(N) He et al. Guided Filter producing 16-bit (`0..65535`) and float (`0..1`) feathered alpha masks.
pub fn guided_feather_16bit(guide_luma: &[f32], raw_alpha: &[f32], w: usize, h: usize, radius: i32, eps: f32) -> (Vec<u16>, Vec<f32>) {
    let n = w * h;
    let mut a16 = vec![0u16; n];
    let mut af = vec![0.0f32; n];
    if guide_luma.len() >= n && raw_alpha.len() >= n && w > 0 && h > 0 {
        unsafe {
            ol_guided_feather_mask_16bit(
                guide_luma.as_ptr(),
                raw_alpha.as_ptr(),
                w,
                h,
                radius as c_int,
                eps,
                a16.as_mut_ptr(),
                af.as_mut_ptr(),
            );
        }
    }
    (a16, af)
}

/// Lossless JPEG transform (libjpeg-turbo): rotate or flip by moving the compressed DCT blocks, or crop on
/// block boundaries, without recompressing (`op`: `hflip`, `vflip`, `transpose`, `transverse`, `rot90`,
/// `rot180`, `rot270`, or anything else with `crop` [x, y, w, h] in pixels). A transform that would have to
/// move the partial blocks at the right or bottom edge is refused, not trimmed: the result replaces the
/// original file.
pub fn jpeg_lossless_transform(jpeg_bytes: &[u8], op: &str, crop: Option<[i32; 4]>) -> Result<Vec<u8>, String> {
    use turbojpeg::{Transform, TransformCrop, TransformOp};
    let mut t = Transform::op(match op {
        "hflip" => TransformOp::Hflip,
        "vflip" => TransformOp::Vflip,
        "transpose" => TransformOp::Transpose,
        "transverse" => TransformOp::Transverse,
        "rot90" | "90" => TransformOp::Rot90,
        "rot180" | "180" => TransformOp::Rot180,
        "rot270" | "270" | "-90" => TransformOp::Rot270,
        _ => TransformOp::None,
    });
    t.perfect = true;
    let header = turbojpeg::read_header(jpeg_bytes).map_err(|e| format!("not a JPEG libjpeg-turbo can read: {e}"))?;
    let (mw, mh) = (header.subsamp.mcu_width(), header.subsamp.mcu_height());
    if let Some([x, y, w, h]) = crop.filter(|c| c[2] > 0 && c[3] > 0) {
        // the origin snaps back to a block boundary, and the region grows to keep the requested edges
        let (x, y) = ((x.max(0) as usize).min(header.width - 1), (y.max(0) as usize).min(header.height - 1));
        let (ax, ay) = (x / mw * mw, y / mh * mh);
        let width = (x + w as usize - ax).clamp(1, header.width - ax);
        let height = (y + h as usize - ay).clamp(1, header.height - ay);
        t.crop = Some(TransformCrop { x: ax, y: ay, width: Some(width), height: Some(height) });
    }
    turbojpeg::transform(&t, jpeg_bytes).map(|b| b.to_vec()).map_err(|e| {
        if e.to_string().to_ascii_lowercase().contains("perfect") {
            format!(
                "{op} can't be lossless for this JPEG: its {}×{} px size isn't a whole number of its {mw}×{mh} px blocks, so the edge blocks would be lost",
                header.width, header.height
            )
        } else {
            format!("lossless JPEG transform: {e}")
        }
    })
}

/// Decode a HEIC / HEIF / AVIF container with ImageIO into `(width, height, 16-bit RGB, ICC profile)`,
/// upright. 16 bits keep a 10-bit file's levels.
pub fn decode_heif(data: &[u8], max_edge: Option<u32>) -> Result<(u32, u32, Vec<u16>, Option<Vec<u8>>), String> {
    let mut rgb_ptr: *mut u16 = std::ptr::null_mut();
    let mut w: u32 = 0;
    let mut h: u32 = 0;
    let mut icc_ptr: *mut u8 = std::ptr::null_mut();
    let mut icc_len: usize = 0;
    let mut err_buf = [0 as c_char; 160];
    let rc = unsafe {
        ol_decode_heif_container(
            data.as_ptr(),
            data.len(),
            max_edge.map_or(0, |e| e.min(16384) as c_int),
            &mut rgb_ptr,
            &mut w,
            &mut h,
            &mut icc_ptr,
            &mut icc_len,
            err_buf.as_mut_ptr(),
        )
    };
    if rc != 0 || rgb_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy().to_string();
        return Err(if msg.is_empty() { "HEIF/AVIF decode failed".into() } else { msg });
    }
    let rgb = unsafe { std::slice::from_raw_parts(rgb_ptr, (w as usize) * (h as usize) * 3) }.to_vec();
    unsafe { ol_free_buf(rgb_ptr as *mut c_void) };
    let icc = if !icc_ptr.is_null() && icc_len > 0 {
        let b = unsafe { std::slice::from_raw_parts(icc_ptr, icc_len) }.to_vec();
        unsafe { ol_free_buf(icc_ptr as *mut c_void) };
        Some(b)
    } else {
        None
    };
    Ok((w, h, rgb, icc))
}

#[cfg(test)]
mod tests {
    /// Not a RAW file: an error, not a crash or an exception escaping from C++.
    #[test]
    fn rejects_what_is_not_raw() {
        let e = super::develop(b"definitely not a camera raw file, just some text", "", "", None).unwrap_err();
        assert!(e.starts_with("LibRaw: open"), "{e}");
        assert!(super::develop(&[], "dht", "rebuild", Some(1600)).is_err());
    }

    #[test]
    fn native_scopes_peaking_and_guided_feather_work() {
        let rgba = vec![128u8, 64, 220, 255].repeat(32 * 32);
        let (parade, wave) = super::compute_scopes_cpu(&rgba, 32, 32, 48, 32);
        assert_eq!(parade.len(), 48 * 32 * 4);
        assert_eq!(wave.len(), 48 * 32 * 4);
        assert!(parade.iter().any(|&v| v > 20));

        let mut rgb = vec![0.2f32; 32 * 32 * 3];
        for r in 10..20 {
            for c in 10..20 {
                let i = (r * 32 + c) * 3;
                rgb[i] = 0.9;
                rgb[i + 1] = 0.9;
                rgb[i + 2] = 0.9;
            }
        }
        let (mask, score) = super::focus_peaking(&rgb, 32, 32, 0.05, &[[0.2, 0.2, 0.25, 0.2, 0.55, 0.2, 0.25, 0.2]]);
        assert_eq!(mask.len(), 32 * 32);
        assert!(score > 0.0 && mask.iter().any(|&m| m > 0));

        let guide: Vec<f32> = (0..32 * 32).map(|i| (i % 32) as f32 / 31.0).collect();
        let raw_a: Vec<f32> = (0..32 * 32).map(|i| if (i % 32) >= 16 { 1.0 } else { 0.0 }).collect();
        let (a16, af) = super::guided_feather_16bit(&guide, &raw_a, 32, 32, 3, 1e-3);
        assert_eq!(a16.len(), 32 * 32);
        assert!(af[0] < 0.2 && af[31] > 0.8);
        assert!(a16[31] > 50000);
    }

    /// Lossless JPEG transforms: a 90° turn swaps the size and decodes to the turned picture, a crop keeps
    /// the requested region (its origin snapped to a block), and a size that isn't whole blocks is refused.
    #[test]
    fn jpeg_lossless_transforms() {
        let jpeg = |w: u32, h: u32| {
            let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 4) as u8, (y * 5) as u8, ((x + y) * 2) as u8]));
            let mut out = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95).encode_image(&img).unwrap();
            out
        };
        let src = jpeg(64, 48);
        let rot = super::jpeg_lossless_transform(&src, "rot90", None).unwrap();
        let (a, b) = (image::load_from_memory(&src).unwrap().to_rgb8(), image::load_from_memory(&rot).unwrap().to_rgb8());
        assert_eq!(b.dimensions(), (48, 64));
        let turned = image::imageops::rotate90(&a);
        let worst = turned.as_raw().iter().zip(b.as_raw()).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(worst <= 2, "rotated pixels differ by {worst}");
        let crop = super::jpeg_lossless_transform(&src, "crop", Some([5, 5, 20, 20])).unwrap();
        assert_eq!(image::load_from_memory(&crop).unwrap().width(), 25);
        let e = super::jpeg_lossless_transform(&jpeg(70, 50), "rot90", None).unwrap_err();
        assert!(e.contains("can't be lossless"), "{e}");
    }

    /// Set OPTIXEL_TEST_HEIC to a 10-bit HEIC or AVIF: it decodes with more than 8 bits of levels.
    #[test]
    fn decodes_heif_at_16_bits() {
        let Some(p) = std::env::var_os("OPTIXEL_TEST_HEIC") else { return };
        let (w, h, rgb, icc) = super::decode_heif(&std::fs::read(&p).unwrap(), None).unwrap();
        let levels = rgb.iter().step_by(3).collect::<std::collections::HashSet<_>>().len();
        eprintln!("{w}x{h}, {levels} red levels, ICC {:?} bytes", icc.as_ref().map(Vec::len));
        assert_eq!(rgb.len(), (w * h * 3) as usize);
        assert!(levels > 256, "{levels} levels: decoded at 8 bits");
    }

    /// Our Bayer demosaics on synthetic scenes, for each of the four 2x2 layouts: a flat grey comes back
    /// flat (no maze, no colour cast), a scene of colour ramps and a soft-edged disc comes back close to the
    /// truth, and on a noisy mosaic LMMSE (the high-ISO choice) is at least as clean as RCD.
    #[test]
    fn demosaics_reconstruct_synthetic_scenes() {
        use std::ffi::c_int;
        unsafe extern "C" {
            fn ol_demosaic_bayer_rcd(cfa: *const f32, w: usize, h: usize, pat: c_int, out: *mut f32);
            fn ol_demosaic_bayer_amaze(cfa: *const f32, w: usize, h: usize, pat: c_int, out: *mut f32);
            fn ol_demosaic_bayer_lmmse(cfa: *const f32, w: usize, h: usize, pat: c_int, out: *mut f32);
        }
        type F = unsafe extern "C" fn(*const f32, usize, usize, c_int, *mut f32);
        let algos: [(&str, F); 3] = [("rcd", ol_demosaic_bayer_rcd), ("amaze", ol_demosaic_bayer_amaze), ("lmmse", ol_demosaic_bayer_lmmse)];
        let (w, h) = (160usize, 120usize);
        // colour of each 2x2 site per layout (0 RGGB, 1 BGGR, 2 GRBG, 3 GBRG), as `decode_cfa_pat`
        let layout = |p: usize, r: usize, c: usize| [[[0, 1], [1, 2]], [[2, 1], [1, 0]], [[1, 0], [2, 1]], [[1, 2], [0, 1]]][p][r & 1][c & 1];
        let scene = |r: usize, c: usize| {
            let (x, y) = (c as f32, r as f32);
            let d = ((x - 80.0).powi(2) + (y - 60.0).powi(2)).sqrt();
            let disc = 1.0 - 0.7 * (1.0 - ((d - 30.0) / 1.5).clamp(-1.0, 1.0)) / 2.0;
            [(0.15 + 0.6 * x / w as f32) * disc, (0.25 + 0.5 * y / h as f32) * disc, (0.5 + 0.25 * (x / 9.0).sin() * (y / 11.0).cos()) * disc]
        };
        let mut noise = 0x9e37_79b9_7f4a_7c15u64;
        let mut gauss = || {
            let mut s = 0.0;
            for _ in 0..12 {
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                s += (noise >> 40) as f32 / (1u64 << 24) as f32;
            }
            s - 6.0
        };
        // PSNR inside a 6 px margin
        let psnr = |out: &[f32], truth: &dyn Fn(usize, usize) -> [f32; 3]| {
            let (mut se, mut n) = (0.0f64, 0usize);
            for r in 6..h - 6 {
                for c in 6..w - 6 {
                    let t = truth(r, c);
                    for k in 0..3 {
                        se += ((out[(r * w + c) * 3 + k] - t[k]) as f64).powi(2);
                        n += 1;
                    }
                }
            }
            10.0 * (1.0 / (se / n as f64)).log10()
        };
        let mut out = vec![0.0f32; w * h * 3];
        for p in 0..4 {
            for (name, f) in algos {
                let flat = vec![0.4f32; w * h];
                unsafe { f(flat.as_ptr(), w, h, p as c_int, out.as_mut_ptr()) };
                let worst = out.iter().map(|v| (v - 0.4).abs()).fold(0.0f32, f32::max);
                assert!(worst < 1e-4, "{name} layout {p}: a flat field came back off by {worst}");

                let cfa: Vec<f32> = (0..w * h).map(|i| scene(i / w, i % w)[layout(p, i / w, i % w)]).collect();
                unsafe { f(cfa.as_ptr(), w, h, p as c_int, out.as_mut_ptr()) };
                let clean = psnr(&out, &scene);
                eprintln!("{name} layout {p}: {clean:.1} dB");
                assert!(clean > 30.0, "{name} layout {p}: {clean:.1} dB");
            }
        }
        // high ISO: the same scene with sensor noise
        let noisy: Vec<f32> = (0..w * h).map(|i| scene(i / w, i % w)[layout(0, i / w, i % w)] + 0.03 * gauss()).collect();
        let db: Vec<f64> = algos
            .iter()
            .map(|(_, f)| {
                unsafe { f(noisy.as_ptr(), w, h, 0, out.as_mut_ptr()) };
                psnr(&out, &scene)
            })
            .collect();
        eprintln!("noisy: rcd {:.1} dB, amaze {:.1} dB, lmmse {:.1} dB", db[0], db[1], db[2]);
        assert!(db[2] >= db[0] - 0.1, "LMMSE {:.1} dB vs RCD {:.1} dB on a noisy mosaic", db[2], db[0]);
    }

    /// Set OPTIXEL_TEST_RAW to a RAW file: its camera space round-trips the working space and keeps white
    /// white (the camera matrix keeps neutral neutral).
    #[test]
    fn camera_space_of_a_real_file() {
        let Some(p) = std::env::var_os("OPTIXEL_TEST_RAW") else { return };
        let (to, from) = super::camera_space(&std::fs::read(&p).unwrap()).expect("a camera matrix");
        let px = [0.2f32, 0.5, 0.8];
        let back = crate::develop::mul(&from, crate::develop::mul(&to, px));
        assert!((0..3).all(|c| (back[c] - px[c]).abs() < 1e-4), "{back:?}");
        let white = crate::develop::mul(&from, [1.0; 3]);
        assert!(white.iter().all(|&v| (v - 1.0).abs() < 0.02), "camera white lands on {white:?}");
    }

    /// Set OPTIXEL_TEST_RAW to a RAW file: every demosaic (including RCD, AMaZE, LMMSE) and highlight
    /// mode decodes it, kept highlights rise above white, proxy comes back at half size, and `inspect`
    /// extracts optical black levels, white level, CFA pattern, and MakerNotes.
    #[test]
    fn develops_a_real_file() {
        let Some(p) = std::env::var_os("OPTIXEL_TEST_RAW") else { return };
        let file = std::fs::read(&p).unwrap();
        let info = super::inspect(&file).unwrap();
        eprintln!(
            "inspect: model='{}' vendor='{}' cfa='{}' black={:?} white={} shutter={} af_points={}",
            info.camera_model, info.vendor, info.cfa_pattern, info.black_levels, info.white_level, info.shutter_count, info.af_points.len()
        );
        assert!(!info.camera_model.is_empty() && info.white_level > 0);

        let mean = |d: &[f32]| d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
        let t = std::time::Instant::now();
        let (w, h, clip) = super::develop(&file, "", "", None).unwrap();
        eprintln!("full {w}x{h} in {:?}, mean {:.4}, max {:.3}", t.elapsed(), mean(&clip), clip.iter().copied().fold(0.0, f32::max));
        assert!(w * h > 4_000_000 && clip.iter().all(|v| (0.0..=1.0).contains(v)));
        for (name, _) in super::DEMOSAIC {
            let t = std::time::Instant::now();
            let (dw, dh, d) = super::develop(&file, name, "", None).unwrap();
            eprintln!("  {name}: {:?}, mean {:.4}", t.elapsed(), mean(&d));
            assert_eq!((dw, dh), (w, h), "{name}");
            assert!((mean(&d) / mean(&clip) - 1.0).abs() < 0.03, "{name} changed the exposure");
        }
        for (name, _) in &super::HIGHLIGHTS[1..] {
            let (dw, dh, d) = super::develop(&file, "", name, None).unwrap();
            let max = d.iter().copied().fold(0.0, f32::max);
            eprintln!("  {name}: mean {:.4}, max {max:.3}", mean(&d));
            assert_eq!((dw, dh), (w, h), "{name}");
            // the same exposure as a clipped decode wherever nothing is blown (a blown sky rightly rises)
            let (a, b) = clip.chunks(3).zip(d.chunks(3)).filter(|(c, _)| c.iter().all(|&v| v < 0.9)).fold((0.0f64, 0.0f64), |(a, b), (c, e)| {
                (a + c.iter().sum::<f32>() as f64, b + e.iter().sum::<f32>() as f64)
            });
            assert!(max > 1.0 || name != &"unclip", "{name} kept nothing above white");
            assert!((b / a - 1.0).abs() < 0.05, "{name} changed the exposure of what is not blown: {:.4}", b / a);
        }
        let t = std::time::Instant::now();
        let (pw, ph, proxy) = super::develop(&file, "", "", Some(1600)).unwrap();
        eprintln!("proxy {pw}x{ph} in {:?}, mean {:.4}", t.elapsed(), mean(&proxy));
        assert!(pw.max(ph) >= 1600 && pw.max(ph) <= w.max(h) / 2 + 2, "proxy {pw}x{ph}");
        assert!((mean(&proxy) / mean(&clip) - 1.0).abs() < 0.02);
    }
}

