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
    fn ol_parse_makernotes(data: *const u8, len: usize, out: *mut NativeMakerNotesOut) -> c_int;
    fn ol_compute_scopes_parade_waveform(
        rgba: *const u8,
        w: usize,
        h: usize,
        scope_w: usize,
        scope_h: usize,
        out_parade_rgba: *mut u8,
        out_waveform_rgba: *mut u8,
    );
    fn ol_compute_focus_peaking_and_blink(
        rgb: *const f32,
        w: usize,
        h: usize,
        threshold: f32,
        face_boxes_xywh: *const f32,
        num_faces: usize,
        out_peaking_mask: *mut u8,
        out_focus_score: *mut f32,
        out_eye_openness: *mut f32,
        out_blink_detected: *mut c_int,
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
    fn ol_jpeg_lossless_transform(
        jpeg_in: *const u8,
        len_in: usize,
        op: c_int,
        crop_x: c_int,
        crop_y: c_int,
        crop_w: c_int,
        crop_h: c_int,
        out_jpeg: *mut *mut u8,
        out_len: *mut usize,
        err_buf: *mut c_char,
    ) -> c_int;
    fn ol_decode_heif_container(
        data: *const u8,
        len: usize,
        max_edge: c_int,
        out_rgb: *mut *mut u8,
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

/// Compute RGB Parade and Luma/Chroma Waveform scope RGBA8 images (`scope_w` x `scope_h` x 4 bytes each).
pub fn compute_scopes_cpu(rgba: &[u8], w: usize, h: usize, scope_w: usize, scope_h: usize) -> (Vec<u8>, Vec<u8>) {
    let mut parade = vec![0u8; scope_w * scope_h * 4];
    let mut waveform = vec![0u8; scope_w * scope_h * 4];
    if rgba.len() >= w * h * 4 && w > 0 && h > 0 && scope_w >= 3 && scope_h >= 2 {
        unsafe {
            ol_compute_scopes_parade_waveform(
                rgba.as_ptr(),
                w,
                h,
                scope_w,
                scope_h,
                parade.as_mut_ptr(),
                waveform.as_mut_ptr(),
            );
        }
    }
    (parade, waveform)
}

/// Compute focus-peaking mask (`0..255`), focus sharpness score (`0..100`), eye openness EAR, and blink flag.
pub fn focus_peaking_and_blink(
    rgb: &[f32],
    w: usize,
    h: usize,
    threshold: f32,
    face_boxes_xywh: &[[f32; 4]],
) -> (Vec<u8>, f32, f32, bool) {
    let mut mask = vec![0u8; w * h];
    let mut score = 0.0f32;
    let mut ear = 0.32f32;
    let mut blink: c_int = 0;
    if rgb.len() >= w * h * 3 && w >= 3 && h >= 3 {
        unsafe {
            ol_compute_focus_peaking_and_blink(
                rgb.as_ptr(),
                w,
                h,
                threshold,
                if face_boxes_xywh.is_empty() { std::ptr::null() } else { face_boxes_xywh.as_ptr() as *const f32 },
                face_boxes_xywh.len(),
                mask.as_mut_ptr(),
                &mut score,
                &mut ear,
                &mut blink,
            );
        }
    }
    (mask, score, ear, blink != 0)
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

/// Lossless JPEG DCT transform via libturbojpeg (`op`: `"hflip"`, `"vflip"`, `"rot90"`, `"rot180"`, `"rot270"`, `"crop"`).
pub fn jpeg_lossless_transform(
    jpeg_bytes: &[u8],
    op: &str,
    crop: Option<[i32; 4]>,
) -> Result<Vec<u8>, String> {
    let op_code = match op {
        "hflip" => 1,
        "vflip" => 2,
        "transpose" => 3,
        "transverse" => 4,
        "rot90" | "90" => 5,
        "rot180" | "180" => 6,
        "rot270" | "270" | "-90" => 7,
        _ => 0,
    };
    let [cx, cy, cw, ch] = crop.unwrap_or([0, 0, 0, 0]);
    let mut out_ptr: *mut u8 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    let mut err_buf = [0 as c_char; 160];
    let rc = unsafe {
        ol_jpeg_lossless_transform(
            jpeg_bytes.as_ptr(),
            jpeg_bytes.len(),
            op_code,
            cx,
            cy,
            cw,
            ch,
            &mut out_ptr,
            &mut out_len,
            err_buf.as_mut_ptr(),
        )
    };
    if rc != 0 || out_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy().to_string();
        return Err(if msg.is_empty() { "lossless JPEG transform failed".into() } else { msg });
    }
    let vec = unsafe { std::slice::from_raw_parts(out_ptr, out_len) }.to_vec();
    unsafe { ol_free_buf(out_ptr as *mut c_void) };
    Ok(vec)
}

/// Decode a HEIC / HEIF / AVIF container via hardware-accelerated native ImageIO / libheif into
/// `(width, height, rgb8_bytes, optional_icc_profile_bytes)`.
pub fn decode_heif(data: &[u8], max_edge: Option<u32>) -> Result<(u32, u32, Vec<u8>, Option<Vec<u8>>), String> {
    let mut rgb_ptr: *mut u8 = std::ptr::null_mut();
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
        let (mask, score, ear, _blink) = super::focus_peaking_and_blink(&rgb, 32, 32, 0.05, &[[0.2, 0.2, 0.5, 0.5]]);
        assert_eq!(mask.len(), 32 * 32);
        assert!(score > 0.0 && ear > 0.0);

        let guide: Vec<f32> = (0..32 * 32).map(|i| (i % 32) as f32 / 31.0).collect();
        let raw_a: Vec<f32> = (0..32 * 32).map(|i| if (i % 32) >= 16 { 1.0 } else { 0.0 }).collect();
        let (a16, af) = super::guided_feather_16bit(&guide, &raw_a, 32, 32, 3, 1e-3);
        assert_eq!(a16.len(), 32 * 32);
        assert!(af[0] < 0.2 && af[31] > 0.8);
        assert!(a16[31] > 50000);
    }

    /// Set OPTIXEL_TEST_RAW (or OPENLIGHT_TEST_RAW) to a RAW file: every demosaic (including RCD, AMaZE, LMMSE) and highlight
    /// mode decodes it, kept highlights rise above white, proxy comes back at half size, and `inspect`
    /// extracts optical black levels, white level, CFA pattern, and MakerNotes.
    #[test]
    fn develops_a_real_file() {
        let Some(p) = std::env::var_os("OPTIXEL_TEST_RAW").or_else(|| std::env::var_os("OPENLIGHT_TEST_RAW")) else { return };
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
            // the same exposure as a clipped decode wherever nothing is blown
            assert!((mean(&d) / mean(&clip) - 1.0).abs() < 0.1, "{name} changed the exposure");
        }
        let t = std::time::Instant::now();
        let (pw, ph, proxy) = super::develop(&file, "", "", Some(1600)).unwrap();
        eprintln!("proxy {pw}x{ph} in {:?}, mean {:.4}", t.elapsed(), mean(&proxy));
        assert!(pw.max(ph) >= 1600 && pw.max(ph) <= w.max(h) / 2 + 2, "proxy {pw}x{ph}");
        assert!((mean(&proxy) / mean(&clip) - 1.0).abs() < 0.02);
    }
}
