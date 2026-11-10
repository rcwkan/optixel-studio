//! Studio Colour Management & Soft-Proofing with Little CMS 2 (`lcms2` / `lcms2-sys` static C library):
//! - Embedded ICC profile -> 32-bit float linear working space (`to_working`)
//! - 32-bit float ProPhoto RGB (ROMM RGB) and ACEScg (AP1) wide-gamut transforms
//! - Hardware display profile conversion (Apple XDR Display P3, Adobe RGB, custom monitor `.icc`)
//! - Interactive printer/paper soft-proofing (`Transform::new_proofing_context`) with ink Black-Point
//!   Compensation (`Flags::BLACKPOINT_COMPENSATION`) and neon out-of-gamut warnings.

use crate::develop::{M709_2020, M2020_709, M2020_ADOBE, M2020_P3, Px, luma, mul};
use lcms2::{
    CIExyY, CIExyYTRIPLE, ColorSpaceSignature, DisallowCache, Flags, GlobalContext, InfoType, Intent, Locale,
    PixelFormat, Profile, ToneCurve, Transform,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Pixels per lcms2 call: big enough to amortise the call, small enough to spread over the cores.
const CHUNK: usize = 1 << 14;

pub(crate) const D65: CIExyY = CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 };
pub(crate) const D60: CIExyY = CIExyY { x: 0.32168, y: 0.33767, Y: 1.0 };
pub(crate) const D50: CIExyY = CIExyY { x: 0.34567, y: 0.35850, Y: 1.0 };
pub(crate) const D50_WARM_PAPER: CIExyY = CIExyY { x: 0.3520, y: 0.3640, Y: 0.94 };

pub(crate) const REC2020: [[f64; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];
pub(crate) const PROPHOTO: [[f64; 2]; 3] = [[0.7347, 0.2653], [0.1596, 0.8404], [0.0366, 0.0001]];
pub(crate) const ACESCG_AP1: [[f64; 2]; 3] = [[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]];
pub(crate) const DISPLAY_P3: [[f64; 2]; 3] = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
pub(crate) const ADOBE_RGB: [[f64; 2]; 3] = [[0.640, 0.330], [0.210, 0.710], [0.150, 0.060]];
pub(crate) const FINEART_MATTE: [[f64; 2]; 3] = [[0.615, 0.340], [0.245, 0.635], [0.165, 0.085]];
pub(crate) const BARYTA_GLOSS: [[f64; 2]; 3] = [[0.655, 0.328], [0.220, 0.685], [0.152, 0.068]];

/// A transform rayon can share between threads (lcms2's one-pixel cache off).
pub(crate) type Shared<I, O> = Transform<I, O, GlobalContext, DisallowCache>;

pub(crate) fn shared<I: Copy + bytemuck::Pod, O: Copy + bytemuck::Pod>(
    src: &Profile,
    fi: PixelFormat,
    dst: &Profile,
    fo: PixelFormat,
    intent: Intent,
) -> Result<Shared<I, O>, String> {
    Transform::new_flags_context(GlobalContext::new(), src, fi, dst, fo, intent, Flags::NO_CACHE)
        .map_err(|e| format!("ICC: {e}"))
}

/// An RGB matrix/shaper profile from primaries `[[x, y]; 3]`, a white and a pure gamma.
pub(crate) fn rgb_profile(primaries: [[f64; 2]; 3], white: &CIExyY, gamma: f64) -> Result<Profile, String> {
    let p = |[x, y]: [f64; 2]| CIExyY { x, y, Y: 1.0 };
    let tri = CIExyYTRIPLE { Red: p(primaries[0]), Green: p(primaries[1]), Blue: p(primaries[2]) };
    let curve = ToneCurve::new(gamma);
    Profile::new_rgb(white, &tri, &[&curve, &curve, &curve]).map_err(|e| format!("ICC: {e}"))
}

/// The working space: linear Rec.2020, D65.
pub(crate) fn working() -> Profile {
    rgb_profile(REC2020, &D65, 1.0).expect("linear Rec.2020 profile")
}

/// ProPhoto RGB (ROMM RGB, D50), either linear (`gamma = 1.0`) or standard 1.8 encoding.
pub fn prophoto_profile(linear: bool) -> Profile {
    rgb_profile(PROPHOTO, &D50, if linear { 1.0 } else { 1.8 }).expect("ProPhoto RGB profile")
}

/// ACEScg (Academy Color Encoding System AP1 primaries, D60, linear).
pub fn acescg_profile() -> Profile {
    rgb_profile(ACESCG_AP1, &D60, 1.0).expect("ACEScg profile")
}

/// Apple Display P3 (D65, 2.2 transfer curve).
pub fn display_p3_profile() -> Profile {
    rgb_profile(DISPLAY_P3, &D65, 2.2).expect("Display P3 profile")
}

/// Adobe RGB (1998) (D65, 2.19921875 gamma).
pub fn adobe_rgb_profile() -> Profile {
    rgb_profile(ADOBE_RGB, &D65, 2.19921875).expect("Adobe RGB profile")
}

/// Resolve a named or file-backed ICC profile (`sRGB`, `DisplayP3`, `AdobeRGB`, `ProPhotoRGB`, `ACEScg`,
/// `FineArtMatte`, `BarytaGloss`, or a path to a `.icc`/`.icm` file).
pub fn resolve_profile(name: &str, custom_dir: Option<&Path>) -> Result<Profile, String> {
    match name {
        "" | "sRGB" => Ok(Profile::new_srgb()),
        "DisplayP3" => Ok(display_p3_profile()),
        "AdobeRGB" => Ok(adobe_rgb_profile()),
        "ProPhotoRGB" => Ok(prophoto_profile(false)),
        "ProPhotoLinear" => Ok(prophoto_profile(true)),
        "ACEScg" => Ok(acescg_profile()),
        "Rec2020Linear" => Ok(working()),
        "FineArtMatte" => rgb_profile(FINEART_MATTE, &D50_WARM_PAPER, 2.0),
        "BarytaGloss" => rgb_profile(BARYTA_GLOSS, &D50, 2.15),
        other => {
            let p = Path::new(other);
            let bytes = if p.is_file() {
                std::fs::read(p).map_err(|e| format!("ICC read {other}: {e}"))?
            } else if let Some(dir) = custom_dir {
                let candidate = dir.join(other);
                std::fs::read(&candidate).map_err(|e| format!("ICC read {}: {e}", candidate.display()))?
            } else {
                return Err(format!("unknown ICC profile '{other}'"));
            };
            Profile::new_icc(&bytes).map_err(|e| format!("ICC parse {other}: {e}"))
        }
    }
}

pub fn parse_intent(intent: &str) -> Intent {
    match intent {
        "perceptual" => Intent::Perceptual,
        "saturation" => Intent::Saturation,
        "absolute" => Intent::AbsoluteColorimetric,
        _ => Intent::RelativeColorimetric,
    }
}

/// Metadata for a built-in or user-loaded `.icc` profile.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IccProfileEntry {
    pub id: String,
    pub label: String,
    pub kind: String, // "builtin" | "printer" | "custom"
    pub description: String,
}

/// List all built-in wide-gamut / printer profiles plus any custom `.icc`/`.icm` files in `<data_dir>/profiles/icc`.
pub fn list_profiles(data_dir: &Path) -> Vec<IccProfileEntry> {
    let mut out = vec![
        IccProfileEntry { id: "sRGB".into(), label: "sRGB IEC61966-2.1".into(), kind: "builtin".into(), description: "Standard web & SDR display space".into() },
        IccProfileEntry { id: "DisplayP3".into(), label: "Apple Display P3 / XDR".into(), kind: "builtin".into(), description: "Wide-gamut Apple XDR & modern P3 displays".into() },
        IccProfileEntry { id: "AdobeRGB".into(), label: "Adobe RGB (1998)".into(), kind: "builtin".into(), description: "Hardware-calibrated photo & prepress monitors".into() },
        IccProfileEntry { id: "ProPhotoRGB".into(), label: "ProPhoto RGB (ROMM D50)".into(), kind: "builtin".into(), description: "Ultra-wide archival working & export space".into() },
        IccProfileEntry { id: "ACEScg".into(), label: "ACEScg (Linear AP1 D60)".into(), kind: "builtin".into(), description: "Academy scene-linear VFX & grading space".into() },
        IccProfileEntry { id: "FineArtMatte".into(), label: "Hahnemühle FineArt Matte (Sim)".into(), kind: "printer".into(), description: "Warm cotton rag matte paper with raised ink black".into() },
        IccProfileEntry { id: "BarytaGloss".into(), label: "FineArt Baryta Gloss (Sim)".into(), kind: "printer".into(), description: "Exhibition baryta paper with deep Dmax blacks".into() },
    ];
    let dir = data_dir.join("profiles").join("icc");
    if let Ok(rd) = std::fs::read_dir(&dir) {
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| matches!(x.to_ascii_lowercase().as_str(), "icc" | "icm"))
            })
            .collect();
        files.sort();
        for f in files {
            let fname = f.file_name().unwrap_or_default().to_string_lossy().to_string();
            if let Ok(bytes) = std::fs::read(&f)
                && let Ok(prof) = Profile::new_icc(&bytes)
            {
                let desc = prof
                    .info(InfoType::Description, Locale::none())
                    .unwrap_or_else(|| fname.clone());
                out.push(IccProfileEntry {
                    id: fname,
                    label: desc.clone(),
                    kind: "custom".into(),
                    description: format!("Custom ICC ({:?})", prof.color_space()),
                });
            }
        }
    }
    out
}

/// Import a user `.icc` / `.icm` profile from disk into `<data_dir>/profiles/icc/`.
pub fn import_profile(data_dir: &Path, src_path: &Path) -> Result<IccProfileEntry, String> {
    let bytes = std::fs::read(src_path).map_err(|e| format!("read ICC: {e}"))?;
    let prof = Profile::new_icc(&bytes).map_err(|e| format!("invalid ICC profile: {e}"))?;
    let dir = data_dir.join("profiles").join("icc");
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let fname = src_path
        .file_name()
        .ok_or_else(|| "missing filename".to_string())?
        .to_string_lossy()
        .to_string();
    let dst = dir.join(&fname);
    std::fs::write(&dst, &bytes).map_err(|e| format!("write ICC: {e}"))?;
    let desc = prof
        .info(InfoType::Description, Locale::none())
        .unwrap_or_else(|| fname.clone());
    Ok(IccProfileEntry {
        id: fname,
        label: desc,
        kind: "custom".into(),
        description: format!("Custom ICC ({:?})", prof.color_space()),
    })
}

/// An ICC profile lcms2 can read, if it describes RGB data.
pub(crate) fn rgb_icc(icc: &[u8]) -> Result<Profile, String> {
    let p = Profile::new_icc(icc).map_err(|e| format!("ICC: {e}"))?;
    if p.color_space() != ColorSpaceSignature::RgbData {
        return Err(format!("ICC: {:?} profiles are not RGB", p.color_space()));
    }
    Ok(p)
}

/// Pixels encoded as `icc` describes (0–1 floats) to linear Rec.2020, in place. Relative colorimetric:
/// the file's white becomes D65 white and everything else keeps its measured colour.
pub fn to_working(icc: &[u8], px: &mut [Px]) -> Result<(), String> {
    let src = rgb_icc(icc)?;
    let t: Shared<Px, Px> = shared(&src, PixelFormat::RGB_FLT, &working(), PixelFormat::RGB_FLT, Intent::RelativeColorimetric)?;
    px.par_chunks_mut(CHUNK).for_each(|c| t.transform_in_place(c));
    Ok(())
}

/// Transform linear Rec.2020 working pixels into a target profile (`ProPhotoRGB`, `ACEScg`, `DisplayP3`,
/// `AdobeRGB`, `sRGB`, or custom `.icc`) and return the target profile's raw ICC bytes for embedding.
pub fn from_working(px: &[Px], profile_name: &str, custom_dir: Option<&Path>) -> Result<(Vec<Px>, Vec<u8>), String> {
    let dst = resolve_profile(profile_name, custom_dir)?;
    let icc_bytes = dst.icc().map_err(|e| format!("ICC export: {e}"))?;
    let t: Shared<Px, Px> = shared(&working(), PixelFormat::RGB_FLT, &dst, PixelFormat::RGB_FLT, Intent::RelativeColorimetric)?;
    let mut out = px.to_vec();
    out.par_chunks_mut(CHUNK).for_each(|c| t.transform_in_place(c));
    Ok((out, icc_bytes))
}

/// Transform working-space pixels in-place through a 32-bit floating-point ProPhoto RGB or ACEScg
/// intermediate space (for unclipped extreme highlight/shadow gamut mapping).
#[allow(dead_code)]
pub fn convert_working_space(px: &mut [Px], space: &str) -> Result<(), String> {
    let inter = match space {
        "ProPhotoRGB" | "prophoto" => prophoto_profile(true),
        "ACEScg" | "acescg" => acescg_profile(),
        _ => return Ok(()),
    };
    let w = working();
    let to_inter: Shared<Px, Px> = shared(&w, PixelFormat::RGB_FLT, &inter, PixelFormat::RGB_FLT, Intent::RelativeColorimetric)?;
    let back: Shared<Px, Px> = shared(&inter, PixelFormat::RGB_FLT, &w, PixelFormat::RGB_FLT, Intent::RelativeColorimetric)?;
    px.par_chunks_mut(CHUNK).for_each(|c| {
        to_inter.transform_in_place(c);
        back.transform_in_place(c);
    });
    Ok(())
}

/// Interactive Soft-Proofing & Out-of-Gamut / Clipping Warnings powered by LittleCMS2:
/// - When `simulate_paper` is true and `gamut` names a profile, uses `Transform::new_proofing_context`
///   with `Flags::SOFT_PROOFING` (and `Flags::BLACKPOINT_COMPENSATION` when `bpc` is true) to simulate
///   how the printer/paper or target profile renders colors and ink black point on screen.
/// - When `gamut_warning` is true, highlights out-of-gamut pixels in neon magenta/violet (`[0.85, 0.0, 1.0]`).
/// - When `clipping` is true, highlights clipped highlights in red and crushed shadows in blue.
pub fn soft_proof(
    px: &mut [Px],
    gamut: Option<&str>,
    intent: &str,
    bpc: bool,
    simulate_paper: bool,
    gamut_warning: bool,
    clipping: bool,
    custom_dir: Option<&Path>,
) {
    let prof_name = gamut.unwrap_or("").trim();
    if prof_name.is_empty() && !clipping {
        return;
    }

    // 1. Optional LittleCMS2 soft-proofing simulation (ink black-point compensation + rendering intent)
    if simulate_paper && !prof_name.is_empty()
        && let Ok(proof_prof) = resolve_profile(prof_name, custom_dir)
    {
        let w = working();
        let mut flags = Flags::NO_CACHE | Flags::SOFT_PROOFING;
        if bpc {
            flags = flags | Flags::BLACKPOINT_COMPENSATION;
        }
        let rendering_intent = parse_intent(intent);
        if let Ok(t) = Transform::<Px, Px, GlobalContext, DisallowCache>::new_proofing_context(
            GlobalContext::new(),
            &w,
            PixelFormat::RGB_FLT,
            &w,
            PixelFormat::RGB_FLT,
            &proof_prof,
            rendering_intent,
            Intent::RelativeColorimetric,
            flags,
        ) {
            px.par_chunks_mut(CHUNK).for_each(|c| t.transform_in_place(c));
        }
    }

    if !gamut_warning && !clipping {
        return;
    }

    // 2. Out-of-gamut detection (via exact matrix for standard spaces or lcms2 round-trip / target check for ICC profiles)
    let std_mat = match prof_name {
        "sRGB" => Some(M2020_709),
        "DisplayP3" => Some(M2020_P3),
        "AdobeRGB" => Some(M2020_ADOBE),
        _ => None,
    };

    let custom_xform: Option<Shared<Px, Px>> = if gamut_warning && std_mat.is_none() && !prof_name.is_empty() {
        resolve_profile(prof_name, custom_dir)
            .ok()
            .and_then(|p| shared(&working(), PixelFormat::RGB_FLT, &p, PixelFormat::RGB_FLT, parse_intent(intent)).ok())
    } else {
        None
    };

    // High-visibility neon warning colors in linear Rec.2020
    let paint = |c: Px| mul(&M709_2020, c);
    let (red, blue, neon_violet) = (paint([1.0, 0.0, 0.0]), paint([0.0, 0.25, 1.0]), paint([0.85, 0.0, 1.0]));
    let tol = match prof_name {
        "FineArtMatte" => 0.025, // matte paper compresses darkest 2.5% Dmax and saturated primaries
        "BarytaGloss" => 0.008,
        _ => 1e-3,
    };

    px.par_iter_mut().for_each(|p| {
        let s = mul(&M2020_709, *p);
        if clipping && s.iter().any(|&v| v >= 1.0) {
            *p = red;
            return;
        }
        if clipping && s.iter().all(|&v| v < 1.5e-4) {
            *p = blue;
            return;
        }
        if !gamut_warning {
            return;
        }
        if let Some(m) = &std_mat {
            let q = mul(m, *p);
            if q.iter().any(|&v| v < -1e-3) || (luma(*p) < 1.0 && q.iter().any(|&v| v > 1.0 + 1e-3)) {
                *p = neon_violet;
            }
        } else if let Some(xf) = &custom_xform {
            let mut q = [*p];
            xf.transform_in_place(&mut q);
            let q = q[0];
            if q.iter().any(|&v| v < -tol) || (luma(*p) < 1.0 && q.iter().any(|&v| v > 1.0 + tol)) {
                *p = neon_violet;
            }
        }
    });
}

/// An 8-bit RGB image in the colour space `icc` describes to sRGB, in place (perceptual, so a wide-gamut
/// file is compressed into sRGB rather than clipped).
pub fn to_srgb8(icc: &[u8], buf: &mut image::RgbImage) -> Result<(), String> {
    let src = rgb_icc(icc)?;
    let t: Shared<[u8; 3], [u8; 3]> = shared(&src, PixelFormat::RGB_8, &Profile::new_srgb(), PixelFormat::RGB_8, Intent::Perceptual)?;
    let px: &mut [[u8; 3]] = bytemuck::cast_slice_mut(&mut **buf);
    px.par_chunks_mut(CHUNK).for_each(|c| t.transform_in_place(c));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sRGB in, through its ICC profile: the same linear Rec.2020 the built-in sRGB path gives.
    #[test]
    fn srgb_icc_matches_the_built_in_path() {
        let icc = Profile::new_srgb().icc().unwrap();
        let samples: Vec<Px> = vec![[0.0; 3], [1.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.5, 0.8], [0.9, 0.4, 0.1]];
        let mut px = samples.clone();
        to_working(&icc, &mut px).unwrap();
        for (s, p) in samples.iter().zip(&px) {
            let want = crate::develop::srgb_to_working(*s);
            for c in 0..3 {
                assert!((p[c] - want[c]).abs() < 2e-3, "{s:?}: {p:?} vs {want:?}");
            }
        }
    }

    /// A pure Display P3 red is outside sRGB but inside Rec.2020: it must arrive unclipped and more
    /// saturated than sRGB red.
    #[test]
    fn wide_gamut_survives() {
        let p3 = rgb_profile(DISPLAY_P3, &D65, 2.2).unwrap().icc().unwrap();
        let mut px = vec![[1.0, 0.0, 0.0]];
        to_working(&p3, &mut px).unwrap();
        let srgb_red = crate::develop::srgb_to_working([1.0, 0.0, 0.0]);
        assert!(px[0][0] > srgb_red[0] - 0.01 && px[0][1] < srgb_red[1], "{:?} vs {srgb_red:?}", px[0]);
        // and to 8-bit sRGB it lands on (or right next to) pure red rather than a broken colour
        let mut img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
        to_srgb8(&p3, &mut img).unwrap();
        let [r, g, b] = img.get_pixel(0, 0).0;
        assert!(r > 240 && g < 40 && b < 40, "{r} {g} {b}");
    }

    #[test]
    fn prophoto_and_acescg_roundtrip_and_soft_proof() {
        let orig: Vec<Px> = vec![[0.18, 0.18, 0.18], [0.8, 0.2, 0.1], [0.1, 0.7, 0.3]];
        for space in ["ProPhotoRGB", "ACEScg"] {
            let mut px = orig.clone();
            convert_working_space(&mut px, space).unwrap();
            for (a, b) in orig.iter().zip(&px) {
                for c in 0..3 {
                    assert!((a[c] - b[c]).abs() < 5e-3, "{space}: {a:?} vs {b:?}");
                }
            }
            let (encoded, icc_bytes) = from_working(&orig, space, None).unwrap();
            assert_eq!(encoded.len(), orig.len());
            assert!(icc_bytes.len() > 200);
        }

        // Soft-proofing with FineArtMatte + BPC + gamut warning
        let mut proof_px = vec![[0.2, 0.2, 0.2], [1.0, 0.0, 0.0]]; // pure Rec.2020 red is out-of-gamut for matte paper
        soft_proof(&mut proof_px, Some("FineArtMatte"), "perceptual", true, true, true, false, None);
        // The out-of-gamut pure Rec.2020 red must be flagged with neon violet (high blue & red, low green)
        assert!(proof_px[1][2] > 0.5 && proof_px[1][1] < 0.2, "expected neon violet warning, got {:?}", proof_px[1]);
    }

    #[test]
    fn rejects_non_rgb_profiles() {
        let grey = Profile::new_gray(&D65, &ToneCurve::new(2.2)).unwrap().icc().unwrap();
        assert!(to_working(&grey, &mut [[0.5; 3]]).is_err());
        assert!(to_working(b"not a profile", &mut [[0.5; 3]]).is_err());
    }
}
