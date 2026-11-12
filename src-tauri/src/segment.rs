//! Segment Anything, locally on ONNX Runtime: SAM 2.1 selects what is clicked, SAM 3 what is named ("sky",
//! "person", "red car": every instance). A model's image encoder runs once per picture and its output is kept
//! for the next prompt, so after the first one a click costs SAM 2.1's decoder milliseconds and a name SAM 3's
//! decoder about a second.
//! ponytail: one picture's encoding per model; switching photos re-encodes.

use crate::develop::{Img, es};
use ort::{session::Session, value::TensorRef};
use std::{path::Path, sync::Mutex};

/// A model's answer for one object: probabilities (0-1) over the whole frame at `w` × `h`, stretched as the
/// picture was to the model's square input.
#[derive(Clone)]
pub struct Prob {
    pub w: usize,
    pub h: usize,
    pub p: Vec<f32>,
}

/// An encoded picture: the encoder outputs the decoder takes (name, shape, values).
struct Encoding {
    key: String,
    outs: Vec<(String, Vec<i64>, Vec<f32>)>,
}

impl Encoding {
    fn get(&self, name: &str) -> Result<TensorRef<'_, f32>, String> {
        let (_, shape, v) = self.outs.iter().find(|o| o.0 == name).ok_or_else(|| format!("no {name} in the image encoding"))?;
        TensorRef::from_array_view((shape.clone(), v.as_slice())).map_err(|e| format!("{name}: {e}"))
    }
}

struct Model {
    id: String,
    sessions: Vec<Session>,
    enc: Option<Encoding>,
}

static SAM2: Mutex<Option<Model>> = Mutex::new(None);
static SAM3: Mutex<Option<Model>> = Mutex::new(None);

/// Let go of the loaded models (one was deleted, or memory is wanted back).
pub fn reset() {
    for m in [&SAM2, &SAM3] {
        if let Ok(mut g) = m.lock() {
            *g = None;
        }
    }
}

/// The ONNX graphs of a model, in the order encoder(, text encoder), decoder.
fn graphs(id: &str) -> Result<&'static [&'static str], String> {
    Ok(match id {
        "sam2.1-hiera-tiny" | "sam2.1-hiera-small" => &["onnx/vision_encoder.onnx", "onnx/prompt_encoder_mask_decoder.onnx"],
        "sam3-fp16" => &["sam3_image_encoder_fp16.onnx", "sam3_text_encoder_fp16.onnx", "sam3_decoder.onnx"],
        "sam3-tiny" => &["efficientsam3_ev_m_image_encoder.onnx", "efficientsam3_ev_m_text_encoder.onnx", "efficientsam3_ev_m_decoder.onnx"],
        _ => return Err(format!("{id} is not a Segment Anything model")),
    })
}

/// The model `id` in `slot`, its sessions opened the first time. On the CPU: CoreML (in this ONNX Runtime)
/// cannot build SAM 2.1's decoder or EfficientSAM3's encoder, fails at run time in SAM 2.1's encoder, and
/// was still compiling SAM 3's after ten minutes.
fn load<'a>(slot: &'a mut Option<Model>, data: &Path, id: &str) -> Result<&'a mut Model, String> {
    if slot.as_ref().is_none_or(|m| m.id != id) {
        *slot = None;
        let mut sessions = vec![];
        for g in graphs(id)? {
            let path = crate::aimodel::model_file(data, id, g).ok_or_else(|| format!("download {id} first (AI tab)"))?;
            sessions.push(crate::aimodel::open_session(&path, None, 0, false)?);
        }
        *slot = Some(Model { id: id.into(), sessions, enc: None });
    }
    slot.as_mut().ok_or_else(|| "model not loaded".to_string())
}

/// `img` stretched to `side` × `side`, display-encoded (sRGB), normalised by `mean` / `std`, as planes.
fn square(img: &Img, side: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let (sx, sy) = (side as f32 / img.w.max(1) as f32, side as f32 / img.h.max(1) as f32);
    let rgb = crate::faceswap::warp(img, &[sx, 0.0, 0.0, 0.0, sy, 0.0], side);
    let area = side * side;
    let mut p = vec![0f32; 3 * area];
    for i in 0..area {
        for c in 0..3 {
            p[c * area + i] = (rgb[i * 3 + c] - mean[c]) / std[c];
        }
    }
    p
}

/// Run the image encoder on `img` unless the encoding for `key` is the one kept; keep the outputs `keep`.
fn encode(m: &mut Model, key: &str, img: &Img, input: &str, side: usize, mean: [f32; 3], std: [f32; 3], keep: &[&str]) -> Result<(), String> {
    if m.enc.as_ref().is_some_and(|e| e.key == key) {
        return Ok(());
    }
    m.enc = None;
    let px = square(img, side, mean, std);
    let t = TensorRef::from_array_view(([1usize, 3, side, side], px.as_slice())).map_err(|e| format!("image input: {e}"))?;
    let out = m.sessions[0].run(ort::inputs![input => t]).map_err(|e| format!("the image encoder failed: {e}"))?;
    let mut outs = vec![];
    for &k in keep {
        let (shape, v) = out[k].try_extract_tensor::<f32>().map_err(|e| format!("{k}: {e}"))?;
        outs.push((k.to_string(), shape.iter().copied().collect(), v.to_vec()));
    }
    m.enc = Some(Encoding { key: key.into(), outs });
    Ok(())
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// SAM 2.1: the object at `points` (fractions of the frame; true includes, false excludes) in `img`. `key`
/// names the picture, so the next click on it reuses its encoding. `whole`: of SAM's three readings of an
/// ambiguous click, the largest (the whole object, the first reading on photos) rather than the surest
/// (often a part of it: a shirt rather than the person).
pub fn click(data: &Path, id: &str, img: &Img, key: &str, points: &[([f32; 2], bool)], whole: bool) -> Result<Prob, String> {
    if points.is_empty() {
        return Err("no point to select at".into());
    }
    let mut g = SAM2.lock().map_err(es)?;
    let m = load(&mut g, data, id)?;
    let emb = ["image_embeddings.0", "image_embeddings.1", "image_embeddings.2"];
    encode(m, key, img, "pixel_values", 1024, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], &emb)?;
    let n = points.len();
    let pts: Vec<f32> = points.iter().flat_map(|(p, _)| [p[0] * 1024.0, p[1] * 1024.0]).collect();
    let labels: Vec<i64> = points.iter().map(|(_, inc)| *inc as i64).collect();
    let no_box: [f32; 0] = [];
    let e = m.enc.as_ref().ok_or("no image encoding")?;
    let feed = ort::inputs![
        "input_points" => TensorRef::from_array_view(([1usize, 1, n, 2], pts.as_slice())).map_err(es)?,
        "input_labels" => TensorRef::from_array_view(([1usize, 1, n], labels.as_slice())).map_err(es)?,
        "input_boxes" => TensorRef::from_array_view(([1usize, 0, 4], &no_box[..])).map_err(es)?,
        "image_embeddings.0" => e.get(emb[0])?,
        "image_embeddings.1" => e.get(emb[1])?,
        "image_embeddings.2" => e.get(emb[2])?,
    ];
    let out = m.sessions[1].run(feed).map_err(|e| format!("SAM 2.1 failed: {e}"))?;
    let (_, iou) = out["iou_scores"].try_extract_tensor::<f32>().map_err(es)?;
    let (shape, masks) = out["pred_masks"].try_extract_tensor::<f32>().map_err(es)?;
    if shape.len() != 5 || iou.len() < 3 {
        return Err(format!("SAM 2.1 returned masks of an unexpected shape {shape:?}"));
    }
    let (k_n, h, w) = (shape[2] as usize, shape[3] as usize, shape[4] as usize);
    let plane = |k: usize| &masks[k * w * h..(k + 1) * w * h];
    let area = |k: usize| plane(k).iter().filter(|&&v| v > 0.0).count();
    let surest = (0..k_n).max_by(|&a, &b| iou[a].total_cmp(&iou[b])).unwrap_or(0);
    let k = if whole { (0..k_n).filter(|&k| iou[k] >= 0.15).max_by_key(|&k| area(k)).unwrap_or(surest) } else { surest };
    Ok(Prob { w, h, p: plane(k).iter().map(|&v| sigmoid(v)).collect() })
}

/// SAM 3: every instance of `what` ("sky", "person") in `img`, surest first, with its score (the instance's
/// own times SAM 3's confidence the thing is there at all); those under `min_score` are left out.
pub fn concept(data: &Path, id: &str, img: &Img, key: &str, what: &str, min_score: f32) -> Result<Vec<(f32, Prob)>, String> {
    let what = what.trim();
    if what.is_empty() {
        return Err("name what to select".into());
    }
    let tokenizer = {
        let file = |f: &str| crate::aimodel::model_file(data, id, f).ok_or_else(|| format!("download {id} first (AI tab)"));
        crate::sdfill::Tokenizer::load_files(&file("vocab.json")?, &file("merges.txt")?)?
    };
    let mut g = SAM3.lock().map_err(es)?;
    let m = load(&mut g, data, id)?;
    let feats = ["fpn_0", "fpn_1", "fpn_2", "pos_2"];
    encode(m, key, img, "image", 1008, [0.5; 3], [0.5; 3], &feats)?;
    // SAM 3 reads 32 tokens, its distilled text encoder 16; padding is 0, as in OpenAI's CLIP
    let len = if id == "sam3-tiny" { 16 } else { 32 };
    let tokens = tokenizer.ids(what, len, 0);
    let (lf_shape, lf, lp_shape, lp) = {
        let text = m.sessions[1]
            .run(ort::inputs!["tokens" => TensorRef::from_array_view(([1usize, len], tokens.as_slice())).map_err(es)?])
            .map_err(|e| format!("the text encoder failed: {e}"))?;
        let (fs, f) = text["language_features"].try_extract_tensor::<f32>().map_err(es)?;
        let (ps, p) = text["language_padding"].try_extract_tensor::<f32>().map_err(es)?;
        (fs.iter().copied().collect::<Vec<i64>>(), f.to_vec(), ps.iter().copied().collect::<Vec<i64>>(), p.to_vec())
    };
    // no exemplar boxes: all eight slots marked unused (1)
    let (boxes, labels, unused) = ([0f32; 32], [0i64; 8], [1f32; 8]);
    let e = m.enc.as_ref().ok_or("no image encoding")?;
    let feed = ort::inputs![
        "fpn_0" => e.get("fpn_0")?,
        "fpn_1" => e.get("fpn_1")?,
        "fpn_2" => e.get("fpn_2")?,
        "pos_2" => e.get("pos_2")?,
        "language_features" => TensorRef::from_array_view((lf_shape, lf.as_slice())).map_err(es)?,
        "language_padding" => TensorRef::from_array_view((lp_shape, lp.as_slice())).map_err(es)?,
        "exemplar_boxes" => TensorRef::from_array_view(([8usize, 1, 4], &boxes[..])).map_err(es)?,
        "exemplar_labels" => TensorRef::from_array_view(([8usize, 1], &labels[..])).map_err(es)?,
        "exemplar_mask" => TensorRef::from_array_view(([1usize, 8], &unused[..])).map_err(es)?,
    ];
    let out = m.sessions[2].run(feed).map_err(|e| format!("SAM 3 failed: {e}"))?;
    let (_, logits) = out["pred_logits"].try_extract_tensor::<f32>().map_err(es)?;
    let (_, presence) = out["presence_logit"].try_extract_tensor::<f32>().map_err(es)?;
    let (shape, masks) = out["pred_masks"].try_extract_tensor::<f32>().map_err(es)?;
    if shape.len() != 4 || presence.is_empty() {
        return Err(format!("SAM 3 returned masks of an unexpected shape {shape:?}"));
    }
    let (n, h, w) = (shape[1] as usize, shape[2] as usize, shape[3] as usize);
    let there = sigmoid(presence[0]);
    let mut found: Vec<(f32, usize)> = (0..n.min(logits.len())).map(|q| (sigmoid(logits[q]) * there, q)).filter(|(s, _)| *s >= min_score).collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(found.into_iter().map(|(s, q)| (s, Prob { w, h, p: masks[q * w * h..(q + 1) * w * h].iter().map(|&v| sigmoid(v)).collect() })).collect())
}

impl Prob {
    /// The union of several (a pixel's highest probability).
    pub fn union(probs: &[Prob]) -> Option<Prob> {
        let first = probs.first()?;
        let mut p = first.p.clone();
        for q in probs.iter().skip(1).filter(|q| q.w == first.w && q.h == first.h) {
            p.iter_mut().zip(&q.p).for_each(|(a, b)| *a = a.max(*b));
        }
        Some(Prob { w: first.w, h: first.h, p })
    }

    /// Probabilities at `w` × `h` (bilinear; the whole frame maps onto the whole frame).
    pub fn resize(&self, w: usize, h: usize) -> Vec<f32> {
        let (sx, sy) = (self.w as f32 / w as f32, self.h as f32 / h as f32);
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (self.h - 1) as f32);
            let (y0, ty) = (fy as usize, fy.fract());
            let y1 = (y0 + 1).min(self.h - 1);
            for x in 0..w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (self.w - 1) as f32);
                let (x0, tx) = (fx as usize, fx.fract());
                let x1 = (x0 + 1).min(self.w - 1);
                let at = |x: usize, y: usize| self.p[y * self.w + x];
                out[y * w + x] = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
            }
        }
        out
    }

    /// The selection as mask-component polygons (fractions of the frame), traced at about 1024 px on the
    /// long side of a frame `aspect` (width / height) wide: each part (add) and each hole in them
    /// (subtract), largest first, small specks left out.
    pub fn polygons(&self, aspect: f32) -> Vec<(bool, Vec<[f32; 2]>)> {
        let (w, h) = if aspect >= 1.0 { (1024, (1024.0 / aspect).round().max(8.0) as usize) } else { ((1024.0 * aspect).round().max(8.0) as usize, 1024) };
        let m: Vec<bool> = self.resize(w, h).into_iter().map(|v| v >= 0.5).collect();
        crate::vision::regions(&m, w, h, 256, 0.0015, 12)
    }
}

/// The SAM model to use for `kind` ("segment": SAM 2.1 clicks, "concept": SAM 3 names): `chosen` when it is
/// installed, else the best one that is.
pub fn pick(data: &Path, kind: &str, chosen: &str) -> Option<String> {
    let order: &[&str] = if kind == "segment" { &["sam2.1-hiera-small", "sam2.1-hiera-tiny"] } else { &["sam3-fp16", "sam3-tiny"] };
    let installed = |id: &str| graphs(id).ok().and_then(|g| crate::aimodel::model_file(data, id, g[0])).is_some();
    std::iter::once(chosen).chain(order.iter().copied()).find(|id| order.contains(id) && installed(id)).map(String::from)
}

/// What Auto masks mean by `target`, found by SAM 3 (`id`) in `img`, as masks (name, polygons): the sky; the
/// people, one mask each; the subject (and so the background): the people together, else the animals.
/// Empty when SAM 3 sees none.
pub fn target(data: &Path, id: &str, img: &Img, key: &str, target: &str) -> Result<Vec<(String, Vec<(bool, Vec<[f32; 2]>)>)>, String> {
    let aspect = img.w as f32 / img.h.max(1) as f32;
    let all = |what: &str| -> Result<Option<Prob>, String> {
        let found = concept(data, id, img, key, what, 0.5)?;
        Ok(Prob::union(&found.into_iter().map(|f| f.1).collect::<Vec<_>>()))
    };
    let one = |name: &str, p: Option<Prob>| p.map(|p| p.polygons(aspect)).filter(|v| !v.is_empty()).map(|v| vec![(name.to_string(), v)]).unwrap_or_default();
    Ok(match target {
        "sky" => one("Sky", all("sky")?),
        "people" => concept(data, id, img, key, "person", 0.5)?
            .into_iter()
            .map(|(_, p)| p.polygons(aspect))
            .filter(|v| !v.is_empty())
            .enumerate()
            .map(|(i, v)| (format!("Person {}", i + 1), v))
            .collect(),
        "subject" | "background" => {
            let name = if target == "subject" { "Subject" } else { "Background" };
            match all("person")? {
                Some(p) => one(name, Some(p)),
                None => one(name, all("animal")?),
            }
        }
        _ => vec![],
    })
}

/// `polys` (parts and holes, from `Prob::polygons`) as the mask `name` of photo `id`: a new mask (`invert`:
/// everything but them), or with `subtract` taken out of mask `mask` (its parts: an object's holes are
/// inside it). One history step; returns the edit and the mask's index.
pub fn commit(core: &std::sync::Arc<crate::Core>, id: i64, name: &str, polys: Vec<(bool, Vec<[f32; 2]>)>, mask: Option<usize>, subtract: bool, invert: bool, label: &str) -> Result<(crate::develop::Develop, usize), String> {
    commit_with(core, id, name, polys, mask, subtract, invert, None, label)
}

/// `commit`, a new mask starting with `adjust`.
#[allow(clippy::too_many_arguments)]
pub fn commit_with(
    core: &std::sync::Arc<crate::Core>,
    id: i64,
    name: &str,
    polys: Vec<(bool, Vec<[f32; 2]>)>,
    mask: Option<usize>,
    subtract: bool,
    invert: bool,
    adjust: Option<crate::develop::LocalAdjust>,
    label: &str,
) -> Result<(crate::develop::Develop, usize), String> {
    use crate::develop::{Mask, MaskComponent};
    let comp = |hole: bool, points: Vec<[f32; 2]>| MaskComponent { op: if hole || subtract { "subtract" } else { "add" }.into(), kind: "polygon".into(), name: name.to_lowercase(), points, ..Default::default() };
    let mut d = core.develop(id)?;
    let i = match mask.filter(|&i| subtract && i < d.masks.len()) {
        Some(i) => {
            d.masks[i].components.extend(polys.into_iter().filter(|(hole, _)| !hole).map(|(_, p)| comp(false, p)));
            i
        }
        None => {
            let components: Vec<MaskComponent> = polys.into_iter().map(|(hole, p)| comp(hole, p)).collect();
            let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
            let name = if subtract { format!("all but {name}") } else { name.to_string() };
            d.masks.push(Mask { mask_id: format!("mask_{ms:x}"), name, invert, components, adjustments: adjust.unwrap_or_default(), ..Default::default() });
            d.masks.len() - 1
        }
    };
    d.clamp();
    core.commit(id, &d, label)?;
    Ok((d, i))
}

/// Click selection with SAM 2.1 on photo `id`: the object at `at` (fractions of the full frame) as a new mask,
/// or with `subtract` taken out of mask `mask`. `part`: SAM's surest reading of the click (a shirt) rather than
/// the whole object (the person).
pub fn select_at(core: &std::sync::Arc<crate::Core>, id: i64, at: [f32; 2], mask: Option<usize>, subtract: bool, part: bool) -> Result<(crate::develop::Develop, usize), String> {
    let model = pick(&core.dir(), "segment", &core.settings.lock().map_err(es)?.ai_select_model).ok_or("download a SAM 2.1 model in the AI tab to select objects locally")?;
    // the full uncropped frame as edited: where mask coordinates live
    let img = core.render(id, &crate::develop::Opts { full_frame: true, ..Default::default() })?;
    let prob = click(&core.dir(), &model, &img, &format!("{id}:{}x{}", img.w, img.h), &[(at, true)], !part)?;
    let polys = prob.polygons(img.w as f32 / img.h.max(1) as f32);
    if polys.is_empty() {
        return Err("nothing to select there".into());
    }
    let n = core.develop(id)?.masks.len() + 1;
    let name = if part { format!("Part {n}") } else { format!("Object {n}") };
    let label = format!("SAM select: {}{name}", if subtract { "− " } else { "" });
    commit(core, id, &name, polys, mask, subtract, false, &label)
}

/// Select by name with SAM 3 on photo `id`: every instance of `what` ("sky", "person", "red car") as one mask
/// (a new one; or with `subtract` taken out of mask `mask`; `invert`: everything but them), starting with
/// `adjust`. Returns the edit, the mask's index and how many instances were found.
#[allow(clippy::too_many_arguments)]
pub fn select_by_name(
    core: &std::sync::Arc<crate::Core>,
    id: i64,
    what: &str,
    mask: Option<usize>,
    subtract: bool,
    invert: bool,
    adjust: Option<crate::develop::LocalAdjust>,
) -> Result<(crate::develop::Develop, usize, usize), String> {
    let model = pick(&core.dir(), "concept", &core.settings.lock().map_err(es)?.ai_concept_model).ok_or("download a SAM 3 model in the AI tab to select by name")?;
    // the full uncropped frame as edited: where mask coordinates live
    let img = core.render(id, &crate::develop::Opts { full_frame: true, ..Default::default() })?;
    let found = concept(&core.dir(), &model, &img, &format!("{id}:{}x{}", img.w, img.h), what, 0.5)?;
    let n = found.len();
    let prob = Prob::union(&found.into_iter().map(|f| f.1).collect::<Vec<_>>()).ok_or_else(|| format!("no {} found", what.trim()))?;
    let polys = prob.polygons(img.w as f32 / img.h.max(1) as f32);
    let name: String = what.trim().chars().take(40).collect();
    let name = name.chars().next().map(|c| c.to_uppercase().collect::<String>() + &name[c.len_utf8()..]).unwrap_or_default();
    let label = format!("SAM 3 select: {}{name}", if subtract { "− " } else { "" });
    let (d, i) = commit_with(core, id, &name, polys, mask, subtract, invert, adjust, &label)?;
    Ok((d, i, n))
}

#[cfg(test)]
mod tests {
    /// Set OPTIXEL_TEST_MODELS to a data folder with SAM 2.1 Tiny and SAM 3 Tiny installed and
    /// OPTIXEL_TEST_PHOTO to a photo of people under a sky: a click on a person selects them whole (and,
    /// part-wise, less), "sky" finds one sky over the top of the frame, "person" several people.
    #[test]
    fn segments_a_real_photo() {
        let (Some(data), Some(photo)) = (std::env::var_os("OPTIXEL_TEST_MODELS"), std::env::var_os("OPTIXEL_TEST_PHOTO")) else { return };
        let data = std::path::Path::new(&data);
        let img = crate::develop::load(std::path::Path::new(&photo), Some(1600)).unwrap();
        let cover = |p: &super::Prob| p.p.iter().filter(|&&v| v >= 0.5).count() as f32 / p.p.len() as f32;
        let aspect = img.w as f32 / img.h as f32;
        let t = std::time::Instant::now();
        let at = [0.586, 0.81];
        let whole = super::click(data, "sam2.1-hiera-tiny", &img, "test", &[(at, true)], true).unwrap();
        let t1 = t.elapsed();
        let part = super::click(data, "sam2.1-hiera-tiny", &img, "test", &[(at, true)], false).unwrap();
        eprintln!("SAM 2.1: first click {t1:?} (encoder + decoder), second {:?}; whole {:.3}, part {:.3} of the frame", t.elapsed() - t1, cover(&whole), cover(&part));
        assert!(cover(&whole) > 0.02 && cover(&part) < cover(&whole), "whole {} part {}", cover(&whole), cover(&part));
        assert!(whole.p[(0.81 * whole.h as f32) as usize * whole.w + (0.586 * whole.w as f32) as usize] > 0.5, "the clicked point is selected");
        let polys = whole.polygons(aspect);
        assert!(!polys.is_empty() && !polys[0].0, "{} polygons", polys.len());
        for id in ["sam3-tiny", "sam3-fp16"] {
            if crate::aimodel::model_file(data, id, "vocab.json").is_none() {
                continue;
            }
            let t = std::time::Instant::now();
            let sky = super::concept(data, id, &img, "test", "sky", 0.5).unwrap();
            let t1 = t.elapsed();
            let people = super::concept(data, id, &img, "test", "person", 0.5).unwrap();
            eprintln!("{id}: sky {t1:?} (encoder + text + decoder) -> {} at {:?}, person {:?} -> {} {:?}", sky.len(), sky.iter().map(|s| s.0).collect::<Vec<_>>(), t.elapsed() - t1, people.len(), people.iter().map(|s| (s.0 * 100.0).round() / 100.0).collect::<Vec<_>>());
            assert_eq!(sky.len(), 1, "{id}: one sky");
            let s = &sky[0].1;
            let top: f32 = s.p[..s.w * s.h / 4].iter().filter(|&&v| v >= 0.5).count() as f32 / (s.w * s.h / 4) as f32;
            assert!(top > 0.9 && cover(s) < 0.6, "{id}: sky covers {} of the top quarter, {} in all", top, cover(s));
            assert!(people.len() >= 3, "{id}: {} people", people.len());
            // what Auto masks make of it: a person each, the people together as the subject
            let each = super::target(data, id, &img, "test", "people").unwrap();
            let subject = super::target(data, id, &img, "test", "subject").unwrap();
            assert_eq!(each.len(), people.len(), "{id}");
            assert!(subject.len() == 1 && subject[0].1.iter().any(|p| !p.0), "{id}: subject {:?}", subject.iter().map(|s| s.1.len()).collect::<Vec<_>>());
        }
    }

    /// Set OPTIXEL_TEST_MODELS (SAM 2.1 and SAM 3 installed) and OPTIXEL_TEST_PHOTO (people under a sky): on a
    /// temporary catalog, a click selects a person and "sky" the sky, as masks that adjust just those pixels
    /// when the photo renders, and the sky inverted adjusts everything else.
    #[test]
    fn selections_edit_the_photo() {
        use crate::develop::{LocalAdjust, Opts};
        let (Some(models), Some(photo)) = (std::env::var_os("OPTIXEL_TEST_MODELS"), std::env::var_os("OPTIXEL_TEST_PHOTO")) else { return };
        let dir = std::env::temp_dir().join(format!("ols-sam-test-{}", crate::new_token()));
        let (data, photos) = (dir.join("data"), dir.join("photos"));
        std::fs::create_dir_all(&photos).unwrap();
        let core = std::sync::Arc::new(crate::Core::open(data.clone()).unwrap());
        // the installed models, and the photo linked in, so nothing is written next to the original
        std::os::unix::fs::symlink(std::path::Path::new(&models).join("models"), data.join("models")).unwrap();
        let photo = std::path::Path::new(&photo);
        std::os::unix::fs::symlink(photo, photos.join(photo.file_name().unwrap())).unwrap();
        let id = crate::catalog::import(&core, &photos, &|_, _| {}).unwrap()[0];
        let luma = |id: i64| {
            let img = core.render(id, &Opts { full_frame: true, ..Default::default() }).unwrap();
            let band = |y0: f32, y1: f32| {
                let rows = (img.h as f32 * y0) as usize..(img.h as f32 * y1) as usize;
                rows.clone().flat_map(|y| img.px[y * img.w..(y + 1) * img.w].iter()).map(|p| crate::develop::luma(*p)).sum::<f32>() / (rows.len() * img.w) as f32
            };
            (band(0.0, 0.25), band(0.75, 1.0))
        };
        let before = luma(id);

        let (d, i) = super::select_at(&core, id, [0.586, 0.81], None, false, false).unwrap();
        assert!(d.masks[i].name.starts_with("Object") && d.masks[i].components.iter().any(|c| c.op == "add"));
        let (mut d, i, found) = super::select_by_name(&core, id, "sky", None, false, false, Some(LocalAdjust { exposure: -1.0, ..Default::default() })).unwrap();
        assert_eq!((found, d.masks[i].name.as_str()), (1, "Sky"));
        core.commit(id, &d, "test").unwrap();
        let sky = luma(id);
        eprintln!("top / bottom luma: before {before:?}, sky -1 EV {sky:?}");
        assert!(sky.0 < before.0 * 0.75 && (sky.1 - before.1).abs() < before.1 * 0.05, "the sky darkens, the ground stays");
        d.masks[i].invert = true;
        core.commit(id, &d, "test").unwrap();
        let rest = luma(id);
        eprintln!("inverted: {rest:?}");
        assert!((rest.0 - before.0).abs() < before.0 * 0.1 && rest.1 < before.1 * 0.75, "inverted: the ground darkens, the sky stays");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
