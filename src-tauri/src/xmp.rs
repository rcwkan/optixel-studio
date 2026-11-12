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

const SKELETON: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Optixel Studio">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   xmp:CreatorTool="Optixel Studio">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"#;

const NAMESPACES: [(&str, &str); 3] = [
    ("xmp", "http://ns.adobe.com/xap/1.0/"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("crs", "http://ns.adobe.com/camera-raw-settings/1.0/"),
];

/// A sidecar this app wrote (rewritten freely). Any other (Lightroom's) is only ever merged into.
pub fn is_ours(xml: &str) -> bool {
    xml.contains(r#"x:xmptk="Optixel Studio""#)
}

/// Byte index of the `>` (or the `/` of `/>`) closing the first `<rdf:Description` start tag.
fn desc_tag_end(xml: &str) -> Option<usize> {
    let start = xml.find("<rdf:Description")?;
    let mut quote = None;
    for (i, c) in xml[start..].char_indices() {
        match (quote, c) {
            (Some(q), _) if c == q => quote = None,
            (None, '"' | '\'') => quote = Some(c),
            (None, '>') => return Some(start + if xml[..start + i].ends_with('/') { i - 1 } else { i }),
            _ => {}
        }
    }
    None
}

/// Property `name` (`prefix:Local`) set to `value`: replaced where it already is (attribute or simple
/// element), else added to the first `rdf:Description`, declaring its namespace there if nothing above does.
fn set_attr(xml: &mut String, name: &str, value: &str) {
    let v = xml_escape(value);
    let n = regex::escape(name);
    let attr = Regex::new(&format!(r#"(\s{n}\s*=\s*)(?:"[^"]*"|'[^']*')"#)).unwrap();
    if let Some(c) = attr.captures(xml) {
        let (m, pre) = (c.get(0).unwrap().range(), c[1].to_string());
        xml.replace_range(m, &format!("{pre}\"{v}\""));
        return;
    }
    let elem = Regex::new(&format!(r#"<{n}\b[^>]*>[^<]*</{n}>"#)).unwrap();
    if let Some(m) = elem.find(xml).map(|m| m.range()) {
        xml.replace_range(m, &format!("<{name}>{v}</{name}>"));
        return;
    }
    let Some(end) = desc_tag_end(xml) else { return };
    let mut add = format!("\n   {name}=\"{v}\"");
    if let Some((prefix, uri)) = NAMESPACES.iter().find(|(p, _)| name.starts_with(&format!("{p}:")))
        && !xml[..end].contains(&format!("xmlns:{prefix}="))
    {
        add = format!("\n    xmlns:{prefix}=\"{uri}\"{add}");
    }
    xml.insert_str(end, &add);
}

/// Element `name` replaced by (or, when missing, added as) `<name>inner</name>`.
fn set_elem(xml: &mut String, name: &str, inner: &str) {
    let new = format!("<{name}>{inner}</{name}>");
    let n = regex::escape(name);
    let re = Regex::new(&format!(r#"(?s)<{n}\b[^>]*>.*?</{n}>"#)).unwrap();
    if let Some(m) = re.find(xml).map(|m| m.range()) {
        xml.replace_range(m, &new);
    } else if let Some(i) = xml.find("</rdf:Description>") {
        xml.insert_str(i, &format!(" {new}\n  "));
    } else if let Some(end) = desc_tag_end(xml).filter(|&e| xml[e..].starts_with("/>")) {
        xml.replace_range(end..end + 2, &format!(">\n   {new}\n  </rdf:Description>"));
    }
}

/// Our catalog fields and the develop settings that differ from the defaults, written into `base` (an
/// existing sidecar, or a fresh skeleton). Everything else in `base` is kept, and keywords are added to its own.
/// Settings left at their defaults are not written, so Lightroom keeps (or applies) its own.
pub fn merge_adobe_xmp(base: &str, p: &Photo, d: &Develop, copyright: Option<&str>, creator: Option<&str>) -> String {
    let mut x = base.to_string();
    set_attr(&mut x, "xmp:Rating", &(if p.flag == -1 { -1 } else { p.rating.clamp(0, 5) }).to_string());
    let label = match p.color_label.as_str() {
        "red" => "Red",
        "yellow" => "Yellow",
        "green" => "Green",
        "blue" => "Blue",
        "purple" => "Purple",
        _ => "",
    };
    set_attr(&mut x, "xmp:Label", label);
    set_attr(&mut x, "crs:Pick", &p.flag.clamp(-1, 1).to_string());

    let def = Develop::default();
    if d.white_balance != def.white_balance {
        set_attr(&mut x, "crs:WhiteBalance", "Custom");
        set_attr(&mut x, "crs:Temperature", &format!("{:.0}", d.white_balance.temperature_k));
        set_attr(&mut x, "crs:Tint", &format!("{:+.0}", d.white_balance.tint));
    }
    if d.basic_tone.exposure != def.basic_tone.exposure {
        set_attr(&mut x, "crs:Exposure2012", &format!("{:+.2}", d.basic_tone.exposure));
    }
    for (name, v, dv) in [
        ("crs:Contrast2012", d.basic_tone.contrast, def.basic_tone.contrast),
        ("crs:Highlights2012", d.basic_tone.highlights, def.basic_tone.highlights),
        ("crs:Shadows2012", d.basic_tone.shadows, def.basic_tone.shadows),
        ("crs:Whites2012", d.basic_tone.whites, def.basic_tone.whites),
        ("crs:Blacks2012", d.basic_tone.blacks, def.basic_tone.blacks),
        ("crs:Texture", d.presence.texture, def.presence.texture),
        ("crs:Clarity2012", d.presence.clarity, def.presence.clarity),
        ("crs:Dehaze", d.presence.dehaze, def.presence.dehaze),
        ("crs:Vibrance", d.presence.vibrance, def.presence.vibrance),
        ("crs:Saturation", d.presence.saturation, def.presence.saturation),
    ] {
        if v != dv {
            set_attr(&mut x, name, &format!("{v:+.0}"));
        }
    }
    for (name, v, dv) in [
        ("crs:Sharpness", d.detail.sharpening, def.detail.sharpening),
        ("crs:LuminanceSmoothing", d.detail.luminance_noise_reduction, def.detail.luminance_noise_reduction),
        ("crs:ColorNoiseReduction", d.detail.color_noise_reduction, def.detail.color_noise_reduction),
    ] {
        if v != dv {
            set_attr(&mut x, name, &format!("{v:.0}"));
        }
    }
    let c = &d.crop;
    if c.x > 1e-4 || c.y > 1e-4 || c.w < 0.9999 || c.h < 0.9999 || c.angle.abs() > 1e-4 {
        set_attr(&mut x, "crs:HasCrop", "True");
        set_attr(&mut x, "crs:CropLeft", &format!("{:.6}", c.x));
        set_attr(&mut x, "crs:CropTop", &format!("{:.6}", c.y));
        set_attr(&mut x, "crs:CropRight", &format!("{:.6}", (c.x + c.w).clamp(0.0, 1.0)));
        set_attr(&mut x, "crs:CropBottom", &format!("{:.6}", (c.y + c.h).clamp(0.0, 1.0)));
        set_attr(&mut x, "crs:CropAngle", &format!("{:.2}", c.angle));
    }

    let mut kws = extract_list(&x, "dc:subject");
    for k in p.keywords.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if !kws.iter().any(|e| e == k) {
            kws.push(k.to_string());
        }
    }
    if !kws.is_empty() {
        let li: String = kws.iter().map(|k| format!("<rdf:li>{}</rdf:li>", xml_escape(k))).collect();
        set_elem(&mut x, "dc:subject", &format!("<rdf:Bag>{li}</rdf:Bag>"));
    }
    if let Some(cr) = copyright.map(str::trim).filter(|s| !s.is_empty()) {
        set_elem(&mut x, "dc:rights", &format!(r#"<rdf:Alt><rdf:li xml:lang="x-default">{}</rdf:li></rdf:Alt>"#, xml_escape(cr)));
    }
    if let Some(au) = creator.map(str::trim).filter(|s| !s.is_empty()) {
        set_elem(&mut x, "dc:creator", &format!("<rdf:Seq><rdf:li>{}</rdf:li></rdf:Seq>", xml_escape(au)));
    }
    x
}

/// A fresh Optixel `.xmp` document for `Photo` + `Develop`.
pub fn format_adobe_xmp(p: &Photo, d: &Develop, copyright: Option<&str>, creator: Option<&str>) -> String {
    merge_adobe_xmp(SKELETON, p, d, copyright, creator)
}

/// Write the `.xmp` sidecar next to `p.path`. One of ours is rewritten (keeping its `dc:rights` /
/// `dc:creator` if none are supplied); anyone else's (Lightroom's) is merged into, never replaced.
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
    let out_path = [primary_xmp_path(photo_path), photo_path.with_extension("XMP")]
        .into_iter()
        .find(|x| x.is_file())
        .unwrap_or_else(|| primary_xmp_path(photo_path));
    let existing = fs::read_to_string(&out_path).ok();
    let xml = match existing.as_deref() {
        Some(old) if !is_ours(old) => merge_adobe_xmp(old, p, d, copyright, creator),
        old => {
            let old = old.map(|x| parse_adobe_xmp_str(x, None));
            let cr = copyright.or_else(|| old.as_ref().and_then(|x| x.copyright.as_deref()));
            let au = creator.or_else(|| old.as_ref().and_then(|x| x.creator.as_deref()));
            format_adobe_xmp(p, d, cr, au)
        }
    };
    fs::write(&out_path, xml).map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo() -> Photo {
        Photo {
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
            latitude: None,
            longitude: None,
        }
    }

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

        let p = photo();

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
        assert!(is_ours(&xml) && !xml.contains("crs:Temperature") && !xml.contains("crs:Sharpness"), "{xml}");
    }

    /// A Lightroom sidecar is merged into, never replaced: its own settings and keywords survive, and
    /// only what we changed is written (declaring `crs:` where it was missing). Merging twice changes nothing.
    #[test]
    fn merges_into_a_lightroom_sidecar() {
        let lr = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
   xmp:Rating="2"
   xmp:Label="Red">
   <dc:subject><rdf:Bag><rdf:li>travel</rdf:li></rdf:Bag></dc:subject>
   <lr:hierarchicalSubject><rdf:Bag><rdf:li>places|Tokyo</rdf:li></rdf:Bag></lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        let mut d = Develop::default();
        d.basic_tone.exposure = 0.5;
        let p = Photo { rating: 4, flag: 0, keywords: "wedding, travel".into(), color_label: "".into(), ..photo() };
        let x = merge_adobe_xmp(lr, &p, &d, None, None);
        assert!(!is_ours(&x) && x.contains("places|Tokyo") && x.contains(r#"xmp:Rating="4""#) && x.contains(r#"xmp:Label="""#), "{x}");
        assert_eq!(x.matches("xmlns:crs=").count(), 1, "{x}");
        assert_eq!(x.matches("crs:Exposure2012=").count(), 1, "{x}");
        assert!(!x.contains("crs:Temperature") && !x.contains("crs:Contrast2012"), "{x}");
        let parsed = parse_adobe_xmp_str(&x, None);
        assert_eq!(parsed.keywords, vec!["travel", "wedding"]);
        assert!((parsed.develop.basic_tone.exposure - 0.5).abs() < 1e-3);
        assert_eq!(merge_adobe_xmp(&x, &p, &d, None, None), x);
    }
}
