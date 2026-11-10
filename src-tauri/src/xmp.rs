//! Bidirectional Adobe Lightroom / Camera Raw `.xmp` sidecar parser and writer.
//! Preserves star ratings (`xmp:Rating`), pick/reject flags (`crs:Pick`), color labels (`xmp:Label`),
//! keywords (`dc:subject`), IPTC copyright (`dc:rights`) & creator (`dc:creator`), crop rectangles
//! (`crs:CropLeft`, `crs:CropTop`, `crs:CropRight`, `crs:CropBottom`, `crs:CropAngle`), and core
//! Camera Raw tone/color adjustments (`crs:Exposure2012`, `crs:Contrast2012`, etc.).

use crate::catalog::{Culling, Photo};
use crate::develop::Develop;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct AdobeXmp {
    pub rating: Option<i64>,
    pub flag: Option<String>,
    pub color_label: Option<String>,
    pub keywords: Vec<String>,
    pub copyright: Option<String>,
    pub creator: Option<String>,
    pub develop: Develop,
    pub has_develop_edits: bool,
}

/// Candidate `.xmp` sidecar paths for an image file (`DSC_0001.xmp`, `DSC_0001.XMP`, `DSC_0001.NEF.xmp`).
pub fn candidate_xmp_paths(photo_path: &Path) -> Vec<PathBuf> {
    let mut out = vec![
        photo_path.with_extension("xmp"),
        photo_path.with_extension("XMP"),
        PathBuf::from(format!("{}.xmp", photo_path.display())),
        PathBuf::from(format!("{}.XMP", photo_path.display())),
    ];
    out.dedup();
    out
}

/// Primary `.xmp` sidecar path to write (`<stem>.xmp`, matching Lightroom's convention).
pub fn primary_xmp_path(photo_path: &Path) -> PathBuf {
    photo_path.with_extension("xmp")
}

fn extract_tag_or_attr(xml: &str, ns_tag: &str) -> Option<String> {
    // 1. Attribute form: ns_tag="value" or ns_tag='value'
    let attr_pat = format!(r#"(?s)\b{}\s*=\s*["']([^"']*)["']"#, regex::escape(ns_tag));
    if let Ok(re) = Regex::new(&attr_pat)
        && let Some(c) = re.captures(xml)
    {
        return Some(xml_unescape(c.get(1)?.as_str().trim()));
    }
    // 2. Element form: <ns_tag>value</ns_tag> (or containing <rdf:Alt>/<rdf:Seq>/<rdf:Bag><rdf:li>value</rdf:li>)
    let elem_pat = format!(r#"(?s)<{}\b[^>]*>(.*?)</{}>"#, regex::escape(ns_tag), regex::escape(ns_tag));
    if let Ok(re) = Regex::new(&elem_pat)
        && let Some(c) = re.captures(xml)
    {
        let inner = c.get(1)?.as_str().trim();
        if let Ok(li_re) = Regex::new(r#"(?s)<rdf:li\b[^>]*>(.*?)</rdf:li>"#)
            && let Some(li) = li_re.captures(inner)
        {
            return Some(xml_unescape(li.get(1)?.as_str().trim()));
        }
        if !inner.starts_with('<') {
            return Some(xml_unescape(inner));
        }
    }
    None
}

fn extract_list(xml: &str, ns_tag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let elem_pat = format!(r#"(?s)<{}\b[^>]*>(.*?)</{}>"#, regex::escape(ns_tag), regex::escape(ns_tag));
    if let Ok(re) = Regex::new(&elem_pat)
        && let Some(c) = re.captures(xml)
    {
        let inner = c.get(1).map_or("", |m| m.as_str());
        if let Ok(li_re) = Regex::new(r#"(?s)<rdf:li\b[^>]*>(.*?)</rdf:li>"#) {
            for cap in li_re.captures_iter(inner) {
                if let Some(m) = cap.get(1) {
                    let s = xml_unescape(m.as_str().trim());
                    if !s.is_empty() && !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

fn xml_unescape(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Parse an Adobe Lightroom / Camera Raw `.xmp` XML string into `AdobeXmp`.
pub fn parse_adobe_xmp_str(xml: &str, base: Option<&Develop>) -> AdobeXmp {
    let mut out = AdobeXmp {
        develop: base.cloned().unwrap_or_default(),
        ..Default::default()
    };

    let f32_tag = |tag: &str| -> Option<f32> {
        extract_tag_or_attr(xml, tag)?.trim_start_matches('+').parse::<f32>().ok()
    };

    // Rating & Pick/Reject flag
    if let Some(r_str) = extract_tag_or_attr(xml, "xmp:Rating")
        && let Ok(r) = r_str.parse::<i64>()
    {
        if r < 0 {
            out.flag = Some("reject".into());
            out.rating = Some(0);
        } else {
            out.rating = Some(r.clamp(0, 5));
        }
    }
    if let Some(p_str) = extract_tag_or_attr(xml, "crs:Pick").or_else(|| extract_tag_or_attr(xml, "xmpDM:good")) {
        match p_str.trim() {
            "1" | "true" | "True" => out.flag = Some("pick".into()),
            "-1" | "false" | "False" => out.flag = Some("reject".into()),
            _ => {}
        }
    }

    // Color label
    if let Some(lbl) = extract_tag_or_attr(xml, "xmp:Label") {
        let lower = lbl.to_ascii_lowercase();
        let mapped = match lower.as_str() {
            "red" | "yellow" | "green" | "blue" | "purple" => Some(lower),
            _ => None,
        };
        if mapped.is_some() {
            out.color_label = mapped;
        }
    }

    // Keywords & IPTC copyright / creator
    out.keywords = extract_list(xml, "dc:subject");
    out.copyright = extract_tag_or_attr(xml, "dc:rights").or_else(|| extract_tag_or_attr(xml, "photoshop:Credit"));
    out.creator = extract_tag_or_attr(xml, "dc:creator").or_else(|| extract_tag_or_attr(xml, "tiff:Artist"));

    // Camera Raw (crs:) adjustments
    let d = &mut out.develop;
    let mut edited = false;
    let mut set = |dst: &mut f32, val: Option<f32>| {
        if let Some(v) = val {
            *dst = v;
            edited = true;
        }
    };

    set(&mut d.white_balance.temperature_k, f32_tag("crs:Temperature").or_else(|| f32_tag("crs:IncrementalTemperature").map(|t| 5500.0 + t * 50.0)));
    set(&mut d.white_balance.tint, f32_tag("crs:Tint").or_else(|| f32_tag("crs:IncrementalTint")));
    set(&mut d.basic_tone.exposure, f32_tag("crs:Exposure2012").or_else(|| f32_tag("crs:Exposure")));
    set(&mut d.basic_tone.contrast, f32_tag("crs:Contrast2012").or_else(|| f32_tag("crs:Contrast")));
    set(&mut d.basic_tone.highlights, f32_tag("crs:Highlights2012"));
    set(&mut d.basic_tone.shadows, f32_tag("crs:Shadows2012"));
    set(&mut d.basic_tone.whites, f32_tag("crs:Whites2012"));
    set(&mut d.basic_tone.blacks, f32_tag("crs:Blacks2012"));
    set(&mut d.presence.texture, f32_tag("crs:Texture"));
    set(&mut d.presence.clarity, f32_tag("crs:Clarity2012").or_else(|| f32_tag("crs:Clarity")));
    set(&mut d.presence.dehaze, f32_tag("crs:Dehaze"));
    set(&mut d.presence.vibrance, f32_tag("crs:Vibrance"));
    set(&mut d.presence.saturation, f32_tag("crs:Saturation"));
    set(&mut d.detail.sharpening, f32_tag("crs:Sharpness"));
    set(&mut d.detail.luminance_noise_reduction, f32_tag("crs:LuminanceSmoothing"));
    set(&mut d.detail.color_noise_reduction, f32_tag("crs:ColorNoiseReduction"));

    // Crop rectangle (crs:CropLeft, crs:CropTop, crs:CropRight, crs:CropBottom, crs:CropAngle)
    let has_crop = extract_tag_or_attr(xml, "crs:HasCrop")
        .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
        .unwrap_or(true);
    if has_crop {
        let left = f32_tag("crs:CropLeft");
        let top = f32_tag("crs:CropTop");
        let right = f32_tag("crs:CropRight");
        let bottom = f32_tag("crs:CropBottom");
        if let (Some(l), Some(t), Some(r), Some(b)) = (left, top, right, bottom)
            && r > l + 0.01
            && b > t + 0.01
        {
            d.crop.x = l.clamp(0.0, 0.98);
            d.crop.y = t.clamp(0.0, 0.98);
            d.crop.w = (r - l).clamp(0.02, 1.0 - d.crop.x);
            d.crop.h = (b - t).clamp(0.02, 1.0 - d.crop.y);
            edited = true;
        }
        if let Some(ang) = f32_tag("crs:CropAngle") {
            d.crop.angle = ang.clamp(-45.0, 45.0);
            edited = true;
        }
    }

    d.clamp();
    out.has_develop_edits = edited;
    out
}

/// Read an existing `.xmp` sidecar next to `photo_path`, if present.
pub fn read_adobe_xmp(photo_path: &Path, base: Option<&Develop>) -> Option<AdobeXmp> {
    for p in candidate_xmp_paths(photo_path) {
        if let Ok(xml) = fs::read_to_string(&p) {
            return Some(parse_adobe_xmp_str(&xml, base));
        }
    }
    None
}

/// Convert `AdobeXmp` catalog metadata into a `Culling` update if any catalog fields are set.
pub fn to_culling(xmp: &AdobeXmp) -> Option<Culling> {
    if xmp.rating.is_none() && xmp.flag.is_none() && xmp.color_label.is_none() && xmp.keywords.is_empty() {
        return None;
    }
    Some(Culling {
        rating: xmp.rating,
        flag: xmp.flag.clone(),
        color_label: xmp.color_label.clone(),
        keywords: if xmp.keywords.is_empty() {
            None
        } else {
            Some(crate::catalog::merge_keywords("", &xmp.keywords))
        },
    })
}

/// Format an Adobe Lightroom / Camera Raw `.xmp` XML document from `Photo` + `Develop`.
pub fn format_adobe_xmp(
    p: &Photo,
    d: &Develop,
    copyright: Option<&str>,
    creator: Option<&str>,
) -> String {
    let rating = if p.flag == -1 { -1 } else { p.rating.clamp(0, 5) };
    let pick = match p.flag {
        1 => 1,
        -1 => -1,
        _ => 0,
    };
    let label = match p.color_label.as_str() {
        "red" => "Red",
        "yellow" => "Yellow",
        "green" => "Green",
        "blue" => "Blue",
        "purple" => "Purple",
        _ => "",
    };
    let has_crop = d.crop.x > 1e-4 || d.crop.y > 1e-4 || d.crop.w < 0.9999 || d.crop.h < 0.9999 || d.crop.angle.abs() > 1e-4;
    let crop_right = (d.crop.x + d.crop.w).clamp(0.0, 1.0);
    let crop_bottom = (d.crop.y + d.crop.h).clamp(0.0, 1.0);

    let mut children = String::new();
    let kws: Vec<&str> = p.keywords.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if !kws.is_empty() {
        children.push_str("   <dc:subject>\n    <rdf:Bag>\n");
        for kw in kws {
            children.push_str(&format!("     <rdf:li>{}</rdf:li>\n", xml_escape(kw)));
        }
        children.push_str("    </rdf:Bag>\n   </dc:subject>\n");
    }
    if let Some(cr) = copyright.filter(|s| !s.trim().is_empty()) {
        children.push_str(&format!(
            "   <dc:rights>\n    <rdf:Alt>\n     <rdf:li xml:lang=\"x-default\">{}</rdf:li>\n    </rdf:Alt>\n   </dc:rights>\n",
            xml_escape(cr.trim())
        ));
    }
    if let Some(au) = creator.filter(|s| !s.trim().is_empty()) {
        children.push_str(&format!(
            "   <dc:creator>\n    <rdf:Seq>\n     <rdf:li>{}</rdf:li>\n    </rdf:Seq>\n   </dc:creator>\n",
            xml_escape(au.trim())
        ));
    }

    format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Optixel Studio">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   xmp:CreatorTool="Optixel Studio"
   xmp:Rating="{rating}"
   xmp:Label="{label}"
   crs:Version="16.0"
   crs:ProcessVersion="11.0"
   crs:Pick="{pick}"
   crs:Temperature="{temp:.0}"
   crs:Tint="{tint:+.0}"
   crs:Exposure2012="{exp:+.2}"
   crs:Contrast2012="{con:+.0}"
   crs:Highlights2012="{hi:+.0}"
   crs:Shadows2012="{sh:+.0}"
   crs:Whites2012="{wh:+.0}"
   crs:Blacks2012="{bl:+.0}"
   crs:Texture="{tex:+.0}"
   crs:Clarity2012="{cla:+.0}"
   crs:Dehaze="{deh:+.0}"
   crs:Vibrance="{vib:+.0}"
   crs:Saturation="{sat:+.0}"
   crs:Sharpness="{shp:.0}"
   crs:LuminanceSmoothing="{lnr:.0}"
   crs:ColorNoiseReduction="{cnr:.0}"
   crs:HasCrop="{has_crop}"
   crs:CropLeft="{cl:.6}"
   crs:CropTop="{ct:.6}"
   crs:CropRight="{cr:.6}"
   crs:CropBottom="{cb:.6}"
   crs:CropAngle="{ca:.2}">
{children}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"#,
        temp = d.white_balance.temperature_k,
        tint = d.white_balance.tint,
        exp = d.basic_tone.exposure,
        con = d.basic_tone.contrast,
        hi = d.basic_tone.highlights,
        sh = d.basic_tone.shadows,
        wh = d.basic_tone.whites,
        bl = d.basic_tone.blacks,
        tex = d.presence.texture,
        cla = d.presence.clarity,
        deh = d.presence.dehaze,
        vib = d.presence.vibrance,
        sat = d.presence.saturation,
        shp = d.detail.sharpening,
        lnr = d.detail.luminance_noise_reduction,
        cnr = d.detail.color_noise_reduction,
        has_crop = if has_crop { "True" } else { "False" },
        cl = d.crop.x,
        ct = d.crop.y,
        cr = crop_right,
        cb = crop_bottom,
        ca = d.crop.angle,
    )
}

/// Write an Adobe `.xmp` sidecar next to `p.path` (preserving any existing `dc:rights` / `dc:creator` if none supplied).
pub fn write_adobe_xmp(
    p: &Photo,
    d: &Develop,
    copyright: Option<&str>,
    creator: Option<&str>,
) -> Result<PathBuf, String> {
    if p.master_id.is_some() || crate::remote::immich_id(&p.path).is_some() {
        return Err("virtual copies and remote assets do not write disk .xmp sidecars".into());
    }
    let photo_path = Path::new(&p.path);
    let existing = read_adobe_xmp(photo_path, None);
    let eff_copyright = copyright.or_else(|| existing.as_ref().and_then(|x| x.copyright.as_deref()));
    let eff_creator = creator.or_else(|| existing.as_ref().and_then(|x| x.creator.as_deref()));
    let xml = format_adobe_xmp(p, d, eff_copyright, eff_creator);
    let out_path = primary_xmp_path(photo_path);
    fs::write(&out_path, xml).map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_lightroom_xmp_sidecar() {
        let mut d = Develop::default();
        d.basic_tone.exposure = 1.25;
        d.basic_tone.highlights = -45.0;
        d.basic_tone.shadows = 60.0;
        d.presence.clarity = 25.0;
        d.crop.x = 0.1;
        d.crop.y = 0.15;
        d.crop.w = 0.8;
        d.crop.h = 0.7;
        d.crop.angle = -2.5;

        let p = Photo {
            id: 42,
            master_id: None,
            copy_name: None,
            path: "/tmp/DSC_0042.NEF".into(),
            folder: "/tmp".into(),
            filename: "DSC_0042.NEF".into(),
            captured_at: Some("2026-05-01 12:00:00".into()),
            camera: Some("Nikon Z8".into()),
            lens: None,
            iso: Some(64),
            aperture: Some(2.8),
            shutter: Some("1/500".into()),
            focal_length: Some(50.0),
            width: 8256,
            height: 5504,
            rating: 5,
            flag: 1,
            color_label: "green".into(),
            keywords: "wedding, portrait".into(),
            edited: true,
            thumb_v: 1,
            deleted_at: None,
        };

        let xml = format_adobe_xmp(&p, &d, Some("© 2026 Optixel Studio"), Some("Jane Doe"));
        let parsed = parse_adobe_xmp_str(&xml, None);
        assert_eq!(parsed.rating, Some(5));
        assert_eq!(parsed.flag.as_deref(), Some("pick"));
        assert_eq!(parsed.color_label.as_deref(), Some("green"));
        assert_eq!(parsed.keywords, vec!["wedding", "portrait"]);
        assert_eq!(parsed.copyright.as_deref(), Some("© 2026 Optixel Studio"));
        assert_eq!(parsed.creator.as_deref(), Some("Jane Doe"));
        assert!((parsed.develop.basic_tone.exposure - 1.25).abs() < 1e-3);
        assert!((parsed.develop.basic_tone.highlights - -45.0).abs() < 1e-3);
        assert!((parsed.develop.crop.x - 0.1).abs() < 1e-4);
        assert!((parsed.develop.crop.w - 0.8).abs() < 1e-4);
        assert!((parsed.develop.crop.angle - -2.5).abs() < 1e-2);
    }
}
