//! Auto functions (spec §3.1.2, §3.4, §3.5): Catalog & Culling, Non-destructive Editing, AI Masking, Denoise &
//! Super Resolution, HDR Merge, Find blurred subjects and Find duplicates. Each has advanced settings (kept in
//! `Settings::auto`), runs over a list of photos as a cancellable job with `job-progress` events, and
//! returns a summary with a row per photo (or per group of duplicates) for the UI.

use crate::{
    Core, ai, aimodel,
    catalog::{self, Culling, Photo},
    develop::{self, Develop, LocalAdjust, Mask, MaskComponent, es},
    enhance, mcp, remote,
    vision::{self, Metrics, Rgb},
};
use image::{DynamicImage, metadata::Orientation};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

pub const FUNCTIONS: &[&str] = &["cull", "edit", "mask", "enhance", "hdr", "blur", "duplicates"];
/// Fewer edges than this and a frame (fog, a plain sky) is not judged sharp or blurred.
const MIN_EDGES: u32 = 50;
/// Bump when the analysis changes, so cached measurements are taken again.
// 2: blinks only on the face scan's faces
const ANALYSIS_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Advanced settings
// ---------------------------------------------------------------------------

/// Auto Catalog & Culling.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct CullSettings {
    /// Size sharpness is checked at: fast (1024 px) | screen (2048 px) | full (100%, slow).
    pub detail: String,
    /// Reject photos whose sharpness score (0-100) is below `blur_threshold`.
    pub reject_blurry: bool,
    pub blur_threshold: f32,
    /// Judge the main subject's sharpness rather than the frame's sharpest part.
    pub use_subject: bool,
    /// Reject photos with more than `clip_limit` % of the frame blown out or crushed to black.
    pub reject_exposure: bool,
    pub clip_limit: f32,
    /// Photos taken within this many seconds of each other that look alike (`similarity` 0-100) are a burst.
    pub burst_seconds: f32,
    pub similarity: f32,
    /// Flag the best photo of each burst as a pick; reject the others (else they get fewer stars).
    pub pick_best: bool,
    pub reject_similar: bool,
    /// Stars 1-5 from the overall score.
    pub stars: bool,
    /// Colour labels: green picks, yellow borderline, red rejects.
    pub labels: bool,
    /// Catalog keywords from capture data and the analysis (year, month, time of day, orientation,
    /// colours, black and white, high ISO, burst, blurry…).
    pub keywords: bool,
    /// Subject and scene keywords from the AI provider (one request per photo).
    pub ai_keywords: bool,
    /// Leave photos that already have a rating, flag or label alone.
    pub keep_manual: bool,
    /// Reject portrait/subject photos where closed eyes (blink) are detected.
    pub reject_blinks: bool,
}
impl Default for CullSettings {
    fn default() -> Self {
        Self {
            detail: "screen".into(),
            reject_blurry: true,
            blur_threshold: 35.0,
            use_subject: true,
            reject_exposure: true,
            clip_limit: 25.0,
            burst_seconds: 2.0,
            similarity: 60.0,
            pick_best: true,
            reject_similar: false,
            stars: true,
            labels: false,
            keywords: true,
            ai_keywords: false,
            keep_manual: true,
            reject_blinks: true,
        }
    }
}

/// Auto Non-destructive Editing.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct EditSettings {
    /// develop::PROFILES, or "type": each photo's own type (photo_type_of)
    pub profile: String,
    /// 0-100: how far toward the auto result.
    pub strength: f32,
    pub white_balance: bool,
    pub tone: bool,
    /// The profile's look (presence, colour, curve…).
    pub look: bool,
    /// off | level | auto | vertical | full
    pub upright: String,
    pub chromatic_aberration: bool,
    /// Noise reduction from the noise measured on the photo.
    pub noise_reduction: bool,
    /// Leave photos that already have edits alone.
    pub skip_edited: bool,
    /// Edit a new virtual copy instead of the photo.
    pub virtual_copy: bool,
    /// With the look "type": ask the AI provider what each photo shows (one request per photo); off, or
    /// when it fails, the photo's keywords and measurements decide.
    pub ai_type: bool,
}
impl Default for EditSettings {
    fn default() -> Self {
        Self {
            profile: "natural".into(),
            strength: 100.0,
            white_balance: true,
            tone: true,
            look: true,
            upright: "off".into(),
            chromatic_aberration: false,
            noise_reduction: true,
            skip_edited: true,
            virtual_copy: false,
            ai_type: true,
        }
    }
}

/// AI Masking.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct MaskSettings {
    /// subject | sky | background | people (people: AI method only)
    pub targets: Vec<String>,
    /// builtin (estimated on the photo, no model) | ai (the AI provider traces outlines) | sam (SAM 3, on this
    /// computer)
    pub method: String,
    pub feather: f32,
    pub edge_shift: f32,
    /// Replace selections of the same name (else add another).
    pub replace: bool,
    /// 0-100 of a starting adjustment: subject brighter, sky deeper, background darker.
    pub adjust: f32,
}
impl Default for MaskSettings {
    fn default() -> Self {
        Self { targets: vec!["subject".into(), "sky".into()], method: "builtin".into(), feather: 0.0, edge_shift: 0.0, replace: true, adjust: 0.0 }
    }
}

/// AI Denoise & Super Resolution (Enhance).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct EnhanceSettings {
    pub denoise: bool,
    /// 0-100: lightness noise thresholds; `detail` keeps the finest texture; `color` colour noise.
    pub strength: f32,
    pub detail: f32,
    pub color: f32,
    pub super_resolution: bool,
    /// Enlargement 1-4 (each side), capped so the result stays under `max_megapixels`.
    pub scale: f32,
    pub sharpen: f32,
    pub max_megapixels: f32,
    /// builtin | ai (a downloaded denoise model) | command (an external AI engine such as Real-ESRGAN, see `command`)
    pub engine: String,
    pub command: String,
    /// The AI denoise model (see aimodel.rs) and how much of its result is blended in, 0-100; used by engine "ai".
    pub ai_model: String,
    pub ai_strength: f32,
    /// Object removal (engine "ai", an inpaint or generate model): the brush strokes painted over the object, snapped
    /// to its edges, and how many pixels the hole grows. Empty: no removal.
    pub remove_strokes: Vec<crate::brush::Stroke>,
    pub remove_grow: f32,
    /// What to paint in (generative fill only; empty: whatever fits the surroundings).
    pub remove_prompt: String,
    /// auto (32-bit float for RAW files, else 16-bit) | float | tiff16
    pub format: String,
    /// Where the new files go; empty: next to the original (Pictures/Optixel Enhanced for remote photos).
    pub folder: String,
    /// The new photo starts with the original's edits, rating and keywords.
    pub copy_edits: bool,
}
impl Default for EnhanceSettings {
    fn default() -> Self {
        Self {
            denoise: true,
            strength: 50.0,
            detail: 50.0,
            color: 60.0,
            super_resolution: false,
            scale: 2.0,
            sharpen: 25.0,
            max_megapixels: 48.0,
            engine: "builtin".into(),
            command: "realesrgan-ncnn-vulkan -i {{INPUT}} -o {{OUTPUT}} -s {{SCALE}}".into(),
            ai_model: String::new(),
            ai_strength: 80.0,
            remove_strokes: vec![],
            remove_grow: 6.0,
            remove_prompt: String::new(),
            format: "auto".into(),
            folder: String::new(),
            copy_edits: true,
        }
    }
}

/// HDR Merge.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct HdrSettings {
    /// brackets (photos taken within `bracket_seconds` of each other, one merge each) | all (one merge)
    pub grouping: String,
    pub bracket_seconds: f32,
    /// Line up handheld frames.
    pub align: bool,
    /// off | low | medium | high: what moved between frames comes from the middle exposure alone.
    pub deghost: String,
    /// The new photo starts with auto white balance, tone and HDR toning (Merge exposures, fitted to its range).
    pub auto_tone: bool,
    /// Where the new files go; empty: next to the photos (Pictures/Optixel Enhanced for remote ones).
    pub folder: String,
}
impl Default for HdrSettings {
    fn default() -> Self {
        Self { grouping: "brackets".into(), bracket_seconds: 3.0, align: true, deghost: "low".into(), auto_tone: true, folder: String::new() }
    }
}

/// Find blurred main subjects.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct BlurSettings {
    /// fast | screen | full, as for culling.
    pub detail: String,
    /// A subject sharpness (0-100) below this is blurred.
    pub threshold: f32,
    /// builtin | ai: how the main subject is found.
    pub subject: String,
    /// Also report blur in one direction (camera shake, a moving subject) from this strength (0-100).
    pub motion: f32,
    /// Tell a missed focus (subject soft, something else sharp) from a blurred photo.
    pub missed_focus: bool,
    /// none | reject | label | keyword: what happens to the photos found.
    pub action: String,
}
impl Default for BlurSettings {
    fn default() -> Self {
        Self { detail: "screen".into(), threshold: 35.0, subject: "builtin".into(), motion: 50.0, missed_focus: true, action: "none".into() }
    }
}

/// Find duplicates.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct DuplicateSettings {
    /// exact (same file contents) | similar (look alike) | both
    pub mode: String,
    /// 0-100: how alike similar photos must look.
    pub similarity: f32,
    /// Compare only photos taken within this many minutes of each other (0: any time).
    pub window_minutes: f32,
    /// Which photo of a group stays: best (picks, stars, edits, size) | largest | sharpest | oldest | newest
    pub keep: String,
    /// none | reject | label | keyword: what happens to the others.
    pub action: String,
}
impl Default for DuplicateSettings {
    fn default() -> Self {
        Self { mode: "both".into(), similarity: 90.0, window_minutes: 0.0, keep: "best".into(), action: "none".into() }
    }
}

/// Run Auto Catalog & Culling on photos new to the catalog, by where they came from.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct OnImport {
    pub enabled: bool,
    /// Local folders and network drives.
    pub folders: bool,
    /// Cloud storage folders and Google Photos picks.
    pub cloud: bool,
    /// Immich albums (fetches each new photo's preview).
    pub immich: bool,
    /// Copies made by Duplicate.
    pub duplicates: bool,
}
impl Default for OnImport {
    fn default() -> Self {
        Self { enabled: false, folders: true, cloud: true, immich: false, duplicates: true }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AutoSettings {
    pub cull: CullSettings,
    pub edit: EditSettings,
    pub mask: MaskSettings,
    pub enhance: EnhanceSettings,
    pub hdr: HdrSettings,
    pub blur: BlurSettings,
    pub duplicates: DuplicateSettings,
    pub on_import: OnImport,
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// A cancellable job; its progress reaches the UI as `job-progress` events.
pub struct Job<'a> {
    core: &'a Core,
    pub id: String,
    cancel: Arc<AtomicBool>,
}

impl<'a> Job<'a> {
    pub fn start(core: &'a Core, id: &str) -> Job<'a> {
        let cancel = Arc::new(AtomicBool::new(false));
        core.jobs.lock().unwrap().insert(id.to_string(), cancel.clone());
        Job { core, id: id.to_string(), cancel }
    }
    pub fn progress(&self, label: &str, done: usize, total: usize) {
        self.core.emit("job-progress", json!({ "job": self.id, "label": label, "done": done, "total": total }));
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

impl Drop for Job<'_> {
    fn drop(&mut self) {
        let mut jobs = self.core.jobs.lock().unwrap();
        if jobs.get(&self.id).is_some_and(|c| Arc::ptr_eq(c, &self.cancel)) {
            jobs.remove(&self.id);
        }
        drop(jobs);
        self.core.emit("job-progress", json!({ "job": self.id, "done": 0, "total": 0, "finished": true }));
    }
}

/// Ask a running job to stop; it keeps what it finished. False when no such job runs.
pub fn cancel(core: &Core, id: &str) -> bool {
    core.jobs.lock().unwrap().get(id).map(|c| c.store(true, Ordering::Relaxed)).is_some()
}

/// Decodes are memory-heavy (a 24 MP photo is ~300 MB in flight): analyse four at a time.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(4).build().expect("analysis threads"))
}

/// `base` with the fields in `over` replaced (the UI sends a function's settings with the run).
fn with<T: Serialize + DeserializeOwned>(base: &T, over: &Value) -> Result<T, String> {
    let mut v = serde_json::to_value(base).map_err(es)?;
    if let (Some(o), Some(b)) = (over.as_object(), v.as_object_mut()) {
        for (k, x) in o {
            b.insert(k.clone(), x.clone());
        }
    }
    serde_json::from_value(v).map_err(|e| format!("invalid settings: {e}"))
}

/// Run auto function `function` on `ids` with `settings` (overriding the saved ones).
pub fn run(core: &Arc<Core>, function: &str, ids: &[i64], settings: &Value, job: &Job) -> Result<Value, String> {
    let (auto, provider) = {
        let s = core.settings.lock().unwrap();
        (s.auto.clone(), s.active_provider.clone())
    };
    if ids.is_empty() {
        return Err("no photos to work on".into());
    }
    let mut out = match function {
        "cull" => cull(core, ids, &with(&auto.cull, settings)?, &provider, job)?,
        "edit" => edit(core, ids, &with(&auto.edit, settings)?, &provider, job)?,
        "mask" => masks(core, ids, &with(&auto.mask, settings)?, &provider, job)?,
        "enhance" => enhance(core, ids, &with(&auto.enhance, settings)?, job)?,
        "hdr" => hdr(core, ids, &with(&auto.hdr, settings)?, job)?,
        "blur" => find_blurred(core, ids, &with(&auto.blur, settings)?, &provider, job)?,
        "duplicates" => find_duplicates(core, ids, &with(&auto.duplicates, settings)?, job)?,
        f => return Err(format!("unknown auto function '{f}' (use {})", FUNCTIONS.join(", "))),
    };
    out["function"] = json!(function);
    out["cancelled"] = json!(job.cancelled());
    core.emit("catalog-changed", json!({}));
    Ok(out)
}

/// New photos arrived from `source` (folders | cloud | immich | duplicates): when switched on, Auto Catalog &
/// Culling runs on them in the background.
pub fn after_import(core: &Arc<Core>, ids: Vec<i64>, source: &str) {
    crate::faces::after_import(core, &ids);
    let (s, provider) = {
        let s = core.settings.lock().unwrap();
        (s.auto.clone(), s.active_provider.clone())
    };
    let o = &s.on_import;
    let on = o.enabled
        && match source {
            "folders" => o.folders,
            "cloud" => o.cloud,
            "immich" => o.immich,
            "duplicates" => o.duplicates,
            _ => false,
        };
    if !on || ids.is_empty() {
        return;
    }
    let core = core.clone();
    std::thread::spawn(move || {
        let job = Job::start(&core, &format!("on-import-{}", crate::new_token()));
        match cull(&core, &ids, &s.cull, &provider, &job) {
            Ok(r) => {
                core.log("auto", format!("culled {} new photo(s): {} picks, {} rejects", ids.len(), r["picks"], r["rejects"]));
                core.emit("catalog-changed", json!({}));
            }
            Err(e) => core.emit("remote-error", json!(format!("Auto Catalog & Culling: {e}"))),
        }
    });
}

// ---------------------------------------------------------------------------
// Analysis (cached per file)
// ---------------------------------------------------------------------------

/// Masters among `ids`: virtual copies share their master's file (and its culling).
fn masters(core: &Core, ids: &[i64]) -> Result<Vec<Photo>, String> {
    Ok(catalog::photos(&core.db(), ids)?.into_iter().filter(|p| p.master_id.is_none()).collect())
}

/// Size sharpness is measured at, and the long edge it is judged at, for a photo `full` px long.
fn sizes(detail: &str, full: u32) -> (u32, u32) {
    match detail {
        "fast" => (1024, full.min(2048)),
        "full" => (0, full),
        _ => (2048, full.min(2048)),
    }
}

/// Measurement key: the analysis version, size and the file's size and date (Immich assets never change).
/// What a cached measurement depends on: the analysis, the detail asked for, the file, and its faces (a face
/// scan after the measurement changes the eyes it looks at).
fn stamp(p: &Photo, detail: &str, faces: usize) -> String {
    let file = match remote::immich_id(&p.path) {
        Some(_) => "immich".to_string(),
        None => fs::metadata(&p.path)
            .map(|m| format!("{}:{}", m.len(), m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs())))
            .unwrap_or_default(),
    };
    format!("v{ANALYSIS_VERSION}/{detail}/{file}/f{faces}")
}

/// The photo as the analysis sees it: a RAW's embedded preview when large enough (else the RAW itself), an
/// image file, or for an Immich photo not opened yet the server's preview.
fn picture(core: &Core, p: &Photo, edge: u32) -> Result<DynamicImage, String> {
    if let Some(asset) = remote::immich_id(&p.path) {
        return match remote::immich_cached(core, asset) {
            Some(file) => picture_of(core, &file, edge),
            None => remote::immich_preview(core, asset),
        };
    }
    picture_of(core, Path::new(&p.path), edge)
}

fn picture_of(core: &Core, path: &Path, edge: u32) -> Result<DynamicImage, String> {
    if !develop::is_raw(path) {
        return develop::decode_std(path);
    }
    let need = if edge == 0 { u32::MAX } else { edge.min(1600) };
    if let Some(mut img) = develop::raw_preview(path).filter(|i| i.width().max(i.height()) >= need) {
        if let Some(o) = Orientation::from_exif(catalog::exif_orientation(path)) {
            img.apply_orientation(o);
        }
        return Ok(img);
    }
    Ok(catalog::to_dynamic(&core.decode(path, (edge > 0).then_some(edge))?))
}

/// Measure one photo (`subject`: a box from the AI provider, measured fresh) or take its cached measurement.
fn measure(core: &Core, p: &Photo, detail: &str, subject: Option<[f32; 4]>) -> Result<Metrics, String> {
    let master = p.master_id.unwrap_or(p.id);
    // one statement each: a statement's DB lock is held to its end
    let eyes = crate::faces::photo_eyes(&core.db(), master);
    let boxes = crate::faces::photo_boxes(&core.db(), master);
    // installing the landmark model changes what is known about eyes
    let landmarks = crate::aimodel::model_file(&core.dir(), "face-landmarks-2dfan4", crate::faceswap::LANDMARKS).is_some();
    let key = format!("{}/l{}", stamp(p, detail, eyes.len()), landmarks as u8);
    if subject.is_none()
        && let Some(m) = catalog::analysis(&core.db(), &p.path, &key)
    {
        return Ok(m);
    }
    let full = p.width.max(p.height) as u32;
    let (edge, _) = sizes(detail, full);
    let img = picture(core, p, edge)?;
    let got = img.width().max(img.height());
    let full = if full > 0 { full } else { got };
    // 100% needs the full-size file: a smaller preview (an Immich photo not opened yet) is judged at screen size
    let mut m = vision::analyse(&img, edge, sizes(detail, full).1.min(got.max(2048)), subject, &eyes);
    // closed eyes: the faces' landmarks, the most closed face's eye aspect ratio
    if landmarks && !boxes.is_empty() {
        let rgb = vision::Rgb::from_image(&img, 2048);
        let work = develop::Img { w: rgb.w, h: rgb.h, px: rgb.px.iter().map(|&p| develop::srgb_to_working(p)).collect() };
        if let Some(ears) = crate::faces::eye_aspects(&core.dir(), &work, &boxes).filter(|e| !e.is_empty()) {
            let shut = ears.iter().copied().fold(f32::MAX, f32::min);
            m.eye_openness = Some(shut);
            m.blink_detected = shut < crate::faces::BLINK_EAR;
        }
    }
    if subject.is_none() {
        catalog::save_analysis(&core.db(), &p.path, &key, &m)?;
    }
    Ok(m)
}

/// Measure every photo, four at a time; a failure stays with its photo.
fn measure_all(core: &Core, photos: &[Photo], detail: &str, job: &Job, label: &str) -> Vec<Result<Metrics, String>> {
    let (done, total) = (AtomicUsize::new(0), photos.len());
    job.progress(label, 0, total);
    pool().install(|| {
        photos
            .par_iter()
            .map(|p| {
                if job.cancelled() {
                    return Err("cancelled".to_string());
                }
                let r = measure(core, p, detail, None);
                job.progress(label, done.fetch_add(1, Ordering::Relaxed) + 1, total);
                r
            })
            .collect()
    })
}

/// Seconds of a catalog capture time "YYYY-MM-DD HH:MM:SS" (any separators), counted from 1970; only
/// differences are used, so the time zone does not matter.
fn seconds(t: &str) -> Option<i64> {
    let n = |a: usize, b: usize| t.get(a..b)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, s) = (n(0, 4)?, n(5, 7)?, n(8, 10)?, n(11, 13)?, n(14, 16)?, n(17, 19)?);
    // days from the civil date (H. Hinnant)
    let y = if mo <= 2 { y - 1 } else { y };
    let (era, yoe) = (y.div_euclid(400), y.rem_euclid(400));
    let doy = (153 * ((mo + 9) % 12) + 2) / 5 + d - 1;
    let days = era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + s)
}

/// Hash bits two photos may differ by and still count as alike, from a 0-100 similarity.
fn max_bits(similarity: f32) -> u32 {
    ((100.0 - similarity.clamp(0.0, 100.0)) / 100.0 * 32.0).round() as u32
}

fn file_name(p: &Photo) -> String {
    p.filename.clone() + &p.copy_name.as_ref().map(|c| format!(" ({c})")).unwrap_or_default()
}

/// A rating, flag and label to set together.
type Verdict = (Option<i64>, Option<String>, Option<String>);

/// Ratings, flags and labels in as few updates as possible (each mirrored to Immich), then keywords.
fn apply(core: &Arc<Core>, changes: &[(i64, Culling)]) -> Result<(), String> {
    let mut groups: HashMap<Verdict, Vec<i64>> = HashMap::new();
    for (id, u) in changes {
        if u.rating.is_some() || u.flag.is_some() || u.color_label.is_some() {
            groups.entry((u.rating, u.flag.clone(), u.color_label.clone())).or_default().push(*id);
        }
    }
    for ((rating, flag, color_label), ids) in groups {
        core.cull(&ids, &Culling { rating, flag, color_label, keywords: None })?;
    }
    let db = core.db();
    for (id, u) in changes.iter().filter(|(_, u)| u.keywords.is_some()) {
        catalog::update(&db, &[*id], &Culling { keywords: u.keywords.clone(), ..Default::default() })?;
    }
    Ok(())
}

/// What happens to photos an auto finder found: none | reject | label (red) | keyword.
fn act(core: &Arc<Core>, photos: &[&Photo], action: &str, keyword: &str) -> Result<usize, String> {
    let changes: Vec<(i64, Culling)> = photos
        .iter()
        .filter_map(|p| {
            let u = match action {
                "reject" => Culling { flag: Some("reject".into()), ..Default::default() },
                "label" => Culling { color_label: Some("red".into()), ..Default::default() },
                "keyword" => Culling { keywords: Some(catalog::merge_keywords(&p.keywords, &[keyword.to_string()])), ..Default::default() },
                _ => return None,
            };
            Some((p.id, u))
        })
        .collect();
    apply(core, &changes)?;
    Ok(changes.len())
}

// ---------------------------------------------------------------------------
// Auto Catalog & Culling
// ---------------------------------------------------------------------------

/// Overall score 0-100: sharpness counts most, then exposure, then noise.
fn quality(m: &Metrics, subject: bool) -> f32 {
    let sharp = if m.edges < MIN_EDGES {
        60.0
    } else if subject {
        m.subject_sharpness
    } else {
        m.sharpness
    };
    let noise = ((m.noise - 2.0) * 10.0).clamp(0.0, 100.0);
    0.6 * sharp + 0.3 * m.exposure + 0.1 * (100.0 - noise)
}

fn stars(q: f32) -> i64 {
    match q {
        q if q >= 85.0 => 5,
        q if q >= 70.0 => 4,
        q if q >= 55.0 => 3,
        q if q >= 35.0 => 2,
        _ => 1,
    }
}

/// Keywords from capture data and the analysis.
fn keywords(p: &Photo, m: &Metrics) -> Vec<String> {
    const MONTHS: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
    let mut k: Vec<String> = vec![];
    if let Some(t) = p.captured_at.as_deref().filter(|t| t.len() >= 13 && !t.starts_with("0000")) {
        k.push(t[..4].to_string());
        if let Some(mo) = t.get(5..7).and_then(|m| m.parse::<usize>().ok()).filter(|m| (1..=12).contains(m)) {
            k.push(MONTHS[mo - 1].into());
        }
        if let Some(h) = t.get(11..13).and_then(|h| h.parse::<u32>().ok()) {
            k.push(match h {
                5..=10 => "morning",
                11..=16 => "afternoon",
                17..=20 => "evening",
                _ => "night",
            }.into());
        }
    }
    let (w, h) = (p.width.max(m.w as i64) as f32, p.height.max(m.h as i64) as f32);
    if w > 0.0 && h > 0.0 {
        k.push(match w / h {
            r if r >= 2.0 || r <= 0.5 => "panorama",
            r if r > 1.05 => "horizontal",
            r if r < 0.95 => "vertical",
            _ => "square",
        }.into());
    }
    if m.colorfulness < 8.0 {
        k.push("black and white".into());
    } else {
        k.extend(m.colors.iter().cloned());
    }
    if p.iso.is_some_and(|i| i >= 3200) {
        k.push("high iso".into());
    }
    let shutter = p.shutter.as_deref().and_then(|s| match s.split_once('/') {
        Some((a, b)) => Some(a.trim().parse::<f64>().ok()? / b.trim().parse::<f64>().ok()?),
        None => s.trim().parse::<f64>().ok(),
    });
    if shutter.is_some_and(|s| s >= 1.0) {
        k.push("long exposure".into());
    }
    match p.focal_length {
        Some(f) if f >= 200.0 => k.push("telephoto".into()),
        Some(f) if f > 0.0 && f <= 20.0 => k.push("wide angle".into()),
        _ => {}
    }
    k
}

pub fn cull(core: &Arc<Core>, ids: &[i64], s: &CullSettings, provider: &str, job: &Job) -> Result<Value, String> {
    let photos = masters(core, ids)?;
    let n = photos.len();
    let metrics = measure_all(core, &photos, &s.detail, job, "Analysing");
    // bursts: photos next to each other in time, taken close together, that look alike
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (photos[a].captured_at.as_deref().unwrap_or(""), &photos[a].filename).cmp(&(photos[b].captured_at.as_deref().unwrap_or(""), &photos[b].filename)));
    let bits = max_bits(s.similarity);
    let mut burst: Vec<usize> = (0..n).collect();
    for k in 1..order.len() {
        let (a, b) = (order[k - 1], order[k]);
        let t = |i: usize| photos[i].captured_at.as_deref().and_then(seconds);
        let close = matches!((t(a), t(b)), (Some(x), Some(y)) if ((y - x).abs() as f32) <= s.burst_seconds);
        let alike = matches!((&metrics[a], &metrics[b]), (Ok(x), Ok(y)) if vision::distance(&x.phash, &y.phash) <= bits);
        if close && alike {
            burst[b] = burst[a];
        }
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &b) in burst.iter().enumerate() {
        members.entry(b).or_default().push(i);
    }
    let q: Vec<f32> = metrics.iter().map(|m| m.as_ref().map_or(0.0, |m| quality(m, s.use_subject))).collect();
    // rejections first: the best of a burst is chosen among the rest
    let mut reasons: Vec<Vec<String>> = vec![vec![]; n];
    for (i, m) in metrics.iter().enumerate() {
        let Ok(m) = m else { continue };
        let sharp = if s.use_subject { m.subject_sharpness } else { m.sharpness };
        if s.reject_blurry && m.edges >= MIN_EDGES && sharp < s.blur_threshold {
            reasons[i].push(if m.motion >= 0.5 { "motion blur" } else { "blurry" }.into());
        }
        if s.reject_exposure {
            if m.clipped_highlights > s.clip_limit {
                reasons[i].push("blown highlights".into());
            }
            if m.clipped_shadows > s.clip_limit {
                reasons[i].push("crushed shadows".into());
            }
        }
        if s.reject_blinks && m.blink_detected {
            reasons[i].push("closed eyes".into());
        }
    }
    let best: HashMap<usize, usize> = members
        .iter()
        .filter(|(_, v)| v.len() > 1)
        .filter_map(|(g, v)| v.iter().copied().filter(|&i| reasons[i].is_empty() && metrics[i].is_ok()).max_by(|&a, &b| q[a].total_cmp(&q[b])).map(|b| (*g, b)))
        .collect();
    let (mut changes, mut rows) = (vec![], vec![]);
    let (mut picks, mut rejects, mut rated, mut labelled, mut tagged, mut kept, mut failed) = (0, 0, 0, 0, 0, 0, 0);
    for i in 0..n {
        let p = &photos[i];
        let m = match &metrics[i] {
            Ok(m) => m,
            Err(e) => {
                failed += 1;
                rows.push(json!({ "id": p.id, "file": file_name(p), "error": e }));
                continue;
            }
        };
        let group = &members[&burst[i]];
        let in_burst = group.len() > 1;
        let is_best = best.get(&burst[i]) == Some(&i);
        let mut why = reasons[i].clone();
        if in_burst && !is_best && s.reject_similar && why.is_empty() && best.contains_key(&burst[i]) {
            why.push("a better shot in the burst".into());
        }
        let manual = p.rating > 0 || p.flag != 0 || !p.color_label.is_empty();
        let mut u = Culling::default();
        let mut tags = vec![];
        if s.keywords {
            tags = keywords(p, m);
            if in_burst {
                tags.push("burst".into());
            }
            if reasons[i].iter().any(|r| r.contains("blur")) {
                tags.push(if m.motion >= 0.5 { "motion blur" } else { "blurry" }.into());
            }
        }
        if !(s.keep_manual && manual) {
            if !why.is_empty() {
                u.flag = Some("reject".into());
            } else if in_burst && is_best && s.pick_best {
                u.flag = Some("pick".into());
            }
            if s.stars && why.is_empty() {
                let mut st = stars(q[i]);
                if let Some(&b) = best.get(&burst[i]).filter(|&&b| b != i) {
                    st = st.min((stars(q[b]) - 1).max(1));
                }
                u.rating = Some(st);
            }
            if s.labels {
                let sharp = if s.use_subject { m.subject_sharpness } else { m.sharpness };
                u.color_label = Some(
                    if !why.is_empty() {
                        "red"
                    } else if u.flag.as_deref() == Some("pick") {
                        "green"
                    } else if (sharp - s.blur_threshold).abs() < 10.0 {
                        "yellow"
                    } else {
                        "none"
                    }
                    .into(),
                );
            }
        } else {
            kept += 1;
        }
        picks += (u.flag.as_deref() == Some("pick")) as usize;
        rejects += (u.flag.as_deref() == Some("reject")) as usize;
        rated += u.rating.is_some() as usize;
        labelled += u.color_label.as_deref().is_some_and(|l| l != "none") as usize;
        let merged = catalog::merge_keywords(&p.keywords, &tags);
        if merged != p.keywords {
            u.keywords = Some(merged);
            tagged += 1;
        }
        rows.push(json!({
            "id": p.id, "file": file_name(p), "quality": q[i].round(), "sharpness": m.sharpness.round(),
            "subject_sharpness": m.subject_sharpness.round(), "focus_peaking": m.focus_peaking_score.round(),
            "eye_openness": m.eye_openness.map(|e| (e * 100.0).round()), "blink": m.blink_detected,
            "exposure": m.exposure.round(), "noise": (m.noise * 10.0).round() / 10.0,
            "motion": (m.motion * 100.0).round(), "burst": in_burst.then_some(group.len()), "best": is_best,
            "flag": u.flag, "rating": u.rating, "label": u.color_label, "reasons": why, "keywords": tags,
            "kept": s.keep_manual && manual,
        }));
        changes.push((p.id, u));
    }
    // subject and scene keywords from the AI provider, one photo at a time
    let mut notes: Vec<String> = vec![];
    if s.ai_keywords && !job.cancelled() {
        let mut errors = 0;
        for (k, (id, u)) in changes.iter_mut().enumerate() {
            if job.cancelled() {
                break;
            }
            job.progress("Describing with AI", k, n);
            match ai::describe(core, *id, provider) {
                Ok(words) => {
                    let base = u.keywords.clone().unwrap_or_else(|| photos.iter().find(|p| p.id == *id).map(|p| p.keywords.clone()).unwrap_or_default());
                    u.keywords = Some(catalog::merge_keywords(&base, &words));
                    errors = 0;
                }
                Err(e) => {
                    errors += 1;
                    if errors == 3 {
                        notes.push(format!("AI keywords stopped after 3 failures: {e}"));
                        break;
                    }
                }
            }
        }
    }
    apply(core, &changes)?;
    Ok(json!({
        "total": n, "picks": picks, "rejects": rejects, "rated": rated, "labelled": labelled, "keyworded": tagged,
        "bursts": members.values().filter(|v| v.len() > 1).count(), "kept": kept, "failed": failed, "notes": notes, "rows": rows,
    }))
}

// ---------------------------------------------------------------------------
// Find blurred subjects
// ---------------------------------------------------------------------------

pub fn find_blurred(core: &Arc<Core>, ids: &[i64], s: &BlurSettings, provider: &str, job: &Job) -> Result<Value, String> {
    let photos = masters(core, ids)?;
    let n = photos.len();
    let metrics: Vec<Result<Metrics, String>> = if s.subject == "ai" {
        // the provider points out the subject; its box is measured on the photo
        let mut out = vec![];
        for (k, p) in photos.iter().enumerate() {
            if job.cancelled() {
                out.push(Err("cancelled".into()));
                continue;
            }
            job.progress("Finding subjects with AI", k, n);
            out.push(ai::locate(core, p.id, "the main subject", provider).and_then(|b| measure(core, p, &s.detail, Some(b))));
        }
        out
    } else {
        measure_all(core, &photos, &s.detail, job, "Checking sharpness")
    };
    let (mut rows, mut found) = (vec![], vec![]);
    for (p, m) in photos.iter().zip(&metrics) {
        let m = match m {
            Ok(m) => m,
            Err(e) => {
                rows.push(json!({ "id": p.id, "file": file_name(p), "error": e }));
                continue;
            }
        };
        let judged = m.edges >= MIN_EDGES;
        let blurred = judged && m.subject_sharpness < s.threshold;
        let shaken = judged && m.motion * 100.0 >= s.motion && m.subject_sharpness < s.threshold + 25.0;
        let reason = if shaken {
            "motion blur: camera shake or a moving subject"
        } else if blurred && s.missed_focus && m.sharpness >= s.threshold + 20.0 {
            "focus missed the subject: something else is sharp"
        } else if blurred {
            "out of focus"
        } else if !judged {
            "too little detail to judge"
        } else {
            ""
        };
        if blurred || shaken {
            found.push(p);
        }
        rows.push(json!({
            "id": p.id, "file": file_name(p), "found": blurred || shaken, "reason": reason,
            "sharpness": m.sharpness.round(), "subject_sharpness": m.subject_sharpness.round(), "subject_blur_px": (m.subject_blur_px * 10.0).round() / 10.0,
            "motion": (m.motion * 100.0).round(), "subject": m.subject,
        }));
    }
    rows.sort_by_key(|r| !r["found"].as_bool().unwrap_or(false));
    let acted = act(core, &found, &s.action, "blurry")?;
    Ok(json!({ "total": n, "found": found.len(), "acted": acted, "rows": rows }))
}

// ---------------------------------------------------------------------------
// Find duplicates
// ---------------------------------------------------------------------------

/// The file behind a photo when it is on this computer (an Immich photo only once downloaded).
fn local_file(core: &Core, p: &Photo) -> Option<PathBuf> {
    match remote::immich_id(&p.path) {
        Some(id) => remote::immich_cached(core, id),
        None => Some(PathBuf::from(&p.path)).filter(|f| f.is_file()),
    }
}

fn sha256(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(es)?;
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let k = f.read(&mut buf).map_err(es)?;
        if k == 0 {
            break;
        }
        ctx.update(&buf[..k]);
    }
    Ok(ctx.finish().as_ref().iter().map(|b| format!("{b:02x}")).collect())
}

struct Sets(Vec<usize>);
impl Sets {
    fn find(&mut self, i: usize) -> usize {
        let mut r = i;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut i = i;
        while self.0[i] != r {
            (self.0[i], i) = (r, self.0[i]);
        }
        r
    }
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b)] = a.min(b);
        }
    }
}

pub fn find_duplicates(core: &Arc<Core>, ids: &[i64], s: &DuplicateSettings, job: &Job) -> Result<Value, String> {
    let photos = masters(core, ids)?;
    let n = photos.len();
    let mut sets = Sets((0..n).collect());
    let mut content: Vec<Option<String>> = vec![None; n];
    if s.mode != "similar" {
        // same size first, then the same contents
        let files: Vec<Option<(PathBuf, u64)>> = photos.par_iter().map(|p| local_file(core, p).and_then(|f| fs::metadata(&f).ok().map(|m| (f, m.len())))).collect();
        let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            if let Some((_, len)) = f {
                by_size.entry(*len).or_default().push(i);
            }
        }
        let todo: Vec<usize> = by_size.into_values().filter(|v| v.len() > 1).flatten().collect();
        let done = AtomicUsize::new(0);
        let hashes: Vec<(usize, Option<String>)> = pool().install(|| {
            todo.par_iter()
                .map(|&i| {
                    let h = if job.cancelled() { None } else { files[i].as_ref().and_then(|(f, _)| sha256(f).ok()) };
                    job.progress("Comparing file contents", done.fetch_add(1, Ordering::Relaxed) + 1, todo.len());
                    (i, h)
                })
                .collect()
        });
        let mut first: HashMap<String, usize> = HashMap::new();
        for (i, h) in hashes {
            if let Some(h) = h {
                match first.get(&h) {
                    Some(&j) => sets.join(i, j),
                    None => {
                        first.insert(h.clone(), i);
                    }
                }
                content[i] = Some(h);
            }
        }
    }
    if s.mode != "exact" && !job.cancelled() {
        // the catalog's previews: fast, and they exist for every source
        let done = AtomicUsize::new(0);
        let hashes: Vec<Option<u64>> = photos
            .par_iter()
            .map(|p| {
                let h = image::open(core.thumb_path(p.id)).ok().map(|i| vision::phash_of(&i));
                job.progress("Comparing how photos look", done.fetch_add(1, Ordering::Relaxed) + 1, n);
                h
            })
            .collect();
        let times: Vec<Option<i64>> = photos.iter().map(|p| p.captured_at.as_deref().and_then(seconds)).collect();
        let (bits, window) = (max_bits(s.similarity), (s.window_minutes * 60.0) as i64);
        for i in 0..n {
            if job.cancelled() {
                break;
            }
            let Some(a) = hashes[i] else { continue };
            for j in i + 1..n {
                if let Some(b) = hashes[j]
                    && (a ^ b).count_ones() <= bits
                    && (window <= 0 || matches!((times[i], times[j]), (Some(x), Some(y)) if (x - y).abs() <= window))
                {
                    sets.join(i, j);
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = sets.find(i);
        groups.entry(r).or_default().push(i);
    }
    let mut groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    groups.sort_by_key(|g| g[0]);
    let mut out = vec![];
    let mut others: Vec<&Photo> = vec![];
    for g in &groups {
        let area = |i: usize| photos[i].width * photos[i].height;
        let sharp = |i: usize| if s.keep == "sharpest" { measure(core, &photos[i], "screen", None).map_or(0.0, |m| m.subject_sharpness) } else { 0.0 };
        let scores: Vec<(i64, f32, String)> = g.iter().map(|&i| (area(i), sharp(i), photos[i].captured_at.clone().unwrap_or_default())).collect();
        let pos = (0..g.len())
            .max_by(|&a, &b| {
                let (pa, pb) = (&photos[g[a]], &photos[g[b]]);
                let by_intent = (pa.flag, pa.rating, pa.edited).cmp(&(pb.flag, pb.rating, pb.edited));
                let earlier = pb.id.cmp(&pa.id); // the first imported wins ties
                match s.keep.as_str() {
                    "largest" => scores[a].0.cmp(&scores[b].0).then(earlier),
                    "sharpest" => scores[a].1.total_cmp(&scores[b].1).then(earlier),
                    "oldest" => scores[b].2.cmp(&scores[a].2).then(earlier),
                    "newest" => scores[a].2.cmp(&scores[b].2).then(earlier),
                    _ => by_intent.then(scores[a].0.cmp(&scores[b].0)).then(earlier),
                }
            })
            .unwrap_or(0);
        let keep = g[pos];
        let exact = g.iter().all(|&i| content[i].is_some() && content[i] == content[keep]);
        others.extend(g.iter().filter(|&&i| i != keep).map(|&i| &photos[i]));
        out.push(json!({
            "kind": if exact { "exact" } else { "similar" },
            "keep": photos[keep].id,
            "ids": g.iter().map(|&i| photos[i].id).collect::<Vec<_>>(),
            "files": g.iter().map(|&i| file_name(&photos[i])).collect::<Vec<_>>(),
        }));
    }
    let acted = act(core, &others, &s.action, "duplicate")?;
    Ok(json!({ "total": n, "groups": out.len(), "extra": others.len(), "acted": acted, "rows": out }))
}

// ---------------------------------------------------------------------------
// Auto Non-destructive Editing
// ---------------------------------------------------------------------------

/// `from` moved `t` (0-1) of the way to `to`: numbers in between, anything else switches halfway; colour
/// temperature moves in mireds, evenly to the eye.
fn blend(from: &Develop, to: &Develop, t: f32) -> Develop {
    if t >= 0.999 {
        return to.clone();
    }
    fn mix(a: &Value, b: &Value, t: f64) -> Value {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                if x.is_i64() && y.is_i64() {
                    let xi = x.as_i64().unwrap_or(0) as f64;
                    let yi = y.as_i64().unwrap_or(0) as f64;
                    json!((xi * (1.0 - t) + yi * t).round() as i64)
                } else if x.is_u64() && y.is_u64() {
                    let xi = x.as_u64().unwrap_or(0) as f64;
                    let yi = y.as_u64().unwrap_or(0) as f64;
                    json!((xi * (1.0 - t) + yi * t).round() as u64)
                } else {
                    json!(x.as_f64().unwrap_or(0.0) * (1.0 - t) + y.as_f64().unwrap_or(0.0) * t)
                }
            }
            (Value::Object(x), Value::Object(y)) => Value::Object(y.iter().map(|(k, v)| (k.clone(), x.get(k).map_or(v.clone(), |u| mix(u, v, t)))).collect()),
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => Value::Array(x.iter().zip(y).map(|(u, v)| mix(u, v, t)).collect()),
            _ => if t >= 0.5 { b.clone() } else { a.clone() },
        }
    }
    let (a, b) = (serde_json::to_value(from).unwrap_or_default(), serde_json::to_value(to).unwrap_or_default());
    let mut d: Develop = serde_json::from_value(mix(&a, &b, t as f64)).unwrap_or_else(|_| to.clone());
    let mired = |k: f32| 1e6 / k.max(1.0);
    d.white_balance.temperature_k = 1e6 / (mired(from.white_balance.temperature_k) * (1.0 - t) + mired(to.white_balance.temperature_k) * t);
    d
}

/// Photo types and the words (in a keyword) that point to them; ties go to the type listed first.
const TYPES: &[(&str, &[&str])] = &[
    ("night", &["night", "nighttime", "neon", "city lights", "star", "milky way", "astrophotography", "aurora", "moon", "firework", "dark"]),
    ("food", &["food", "dish", "meal", "plate", "dessert", "cake", "coffee", "drink", "cocktail", "breakfast", "lunch", "dinner", "fruit", "bread", "pizza", "salad", "burger", "pasta", "wine", "restaurant"]),
    ("portrait", &["portrait", "person", "people", "man", "men", "woman", "women", "girl", "boy", "child", "children", "face", "selfie", "model", "fashion", "bride", "couple", "smile", "headshot", "family"]),
    ("nature", &["macro", "flower", "insect", "bee", "butterfly", "leaf", "leaves", "plant", "mushroom", "bird", "wildlife", "animal", "dog", "cat", "petal", "moss", "dew", "garden"]),
    ("architecture", &["architecture", "building", "skyscraper", "interior", "facade", "bridge", "church", "cathedral", "tower", "staircase", "stairs", "house", "museum", "room", "ceiling", "cityscape", "skyline"]),
    ("street", &["street", "city", "urban", "crowd", "road", "traffic", "alley", "sidewalk", "pedestrian", "market", "car", "subway", "downtown", "taxi"]),
    ("landscape", &["landscape", "mountain", "lake", "sea", "ocean", "beach", "sunset", "sunrise", "sky", "forest", "desert", "valley", "river", "waterfall", "field", "hill", "coast", "snow", "cloud", "aerial", "horizon", "nature", "wave", "island"]),
];

/// The develop profile for a photo described by `keywords` (most telling first, as the AI lists them): black
/// and white outright, else the type most keywords point to (the first counts double), else natural.
/// ponytail: word lists; an image classifier would read scenes no keyword names.
pub fn photo_type(keywords: &[String]) -> &'static str {
    let keys: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
    let hits = |k: &str, t: &str| {
        if t.contains(' ') {
            k.contains(t)
        } else {
            k.split(|c: char| !c.is_alphanumeric()).any(|w| w == t || w.strip_suffix('s') == Some(t) || w.strip_suffix("es") == Some(t))
        }
    };
    if keys.iter().any(|k| hits(k, "black and white") || hits(k, "monochrome")) {
        return "bw";
    }
    let score = |words: &[&str]| keys.iter().enumerate().filter(|(_, k)| words.iter().any(|t| hits(k, t))).map(|(i, _)| if i == 0 { 2 } else { 1 }).sum::<usize>();
    TYPES.iter().rev().map(|(t, w)| (*t, score(w))).max_by_key(|x| x.1).filter(|x| x.1 > 0).map_or("natural", |x| x.0)
}

/// `p`'s photo type (photo_type) from the AI provider's description when `ask` (else, or when it fails,
/// without it), its catalog keywords and two measured cues: no colour, a dark frame. The error is the AI's.
pub fn photo_type_of(core: &Arc<Core>, p: &Photo, provider: &str, ask: bool) -> (&'static str, Option<String>) {
    let (mut k, err) = match ask.then(|| ai::describe(core, p.id, provider)) {
        Some(Ok(words)) => (words, None),
        Some(Err(e)) => (vec![], Some(e)),
        None => (vec![], None),
    };
    k.extend(p.keywords.split(',').map(|w| w.trim().to_string()));
    if let Ok(m) = measure(core, p, "fast", None) {
        if m.colorfulness < 8.0 {
            k.push("black and white".into());
        }
        if m.p50 < 0.15 {
            k.push("dark".into());
        }
    }
    (photo_type(&k), err)
}

pub fn edit(core: &Arc<Core>, ids: &[i64], s: &EditSettings, provider: &str, job: &Job) -> Result<Value, String> {
    let photos = catalog::photos(&core.db(), ids)?;
    let n = photos.len();
    let (mut rows, mut edited, mut ai_errors) = (vec![], 0, 0);
    for (k, p) in photos.iter().enumerate() {
        if job.cancelled() {
            break;
        }
        job.progress("Editing", k, n);
        let res = (|| -> Result<Value, String> {
            if s.skip_edited && p.edited {
                return Ok(json!({ "status": "skipped", "note": "already edited" }));
            }
            let img = core.proxy(&p.path)?;
            let target = if s.virtual_copy { core.virtual_copy(p.id)? } else { p.id };
            let cur = core.develop(target)?;
            let mut d = cur.clone();
            let mut notes = vec![];
            if s.white_balance {
                d.white_balance = develop::auto_wb(&img);
            }
            if s.tone {
                d.basic_tone = develop::auto_tone(&img, &d.white_balance);
            }
            let mut profile = s.profile.clone();
            if s.look {
                if profile == "type" {
                    // AI typing stops after 3 failures, like AI keywords
                    let (t, err) = photo_type_of(core, p, provider, s.ai_type && ai_errors < 3);
                    if let Some(e) = err {
                        ai_errors += 1;
                        notes.push(format!("AI: {e}"));
                    }
                    notes.push(format!("{t} look"));
                    profile = t.into();
                }
                d = develop::look_on(&d, &profile)?;
            }
            if s.upright != "off" {
                match develop::upright(&img, &d, &s.upright) {
                    Ok(t) => d.transform = t,
                    Err(e) => notes.push(format!("upright: {e}")),
                }
            }
            if s.chromatic_aberration {
                [d.lens.ca_red_cyan, d.lens.ca_blue_yellow] = develop::estimate_ca(&img);
            }
            if s.noise_reduction {
                // noise in 8-bit steps at the 1600 px proxy
                let noise = vision::noise_of(&Rgb::from_working(&img, 0));
                if noise >= 0.8 {
                    d.detail.luminance_noise_reduction = ((noise - 0.8) * 15.0).clamp(0.0, 60.0).round();
                    d.detail.color_noise_reduction = (20.0 + noise * 6.0).clamp(0.0, 60.0).round();
                }
            }
            let mut d = blend(&cur, &d, s.strength.clamp(0.0, 100.0) / 100.0);
            d.clamp();
            if d == cur {
                return Ok(json!({ "id": target, "status": "unchanged", "note": notes.join("; ") }));
            }
            core.commit(target, &d, &format!("Auto edit: {profile}"))?;
            Ok(json!({ "id": target, "status": "edited", "note": notes.join("; ") }))
        })();
        let mut row = res.unwrap_or_else(|e| json!({ "status": "failed", "note": e }));
        edited += (row["status"] == "edited") as usize;
        row["source"] = json!(p.id);
        if row.get("id").is_none() {
            row["id"] = json!(p.id);
        }
        row["file"] = json!(file_name(p));
        rows.push(row);
    }
    Ok(json!({ "total": n, "edited": edited, "rows": rows }))
}

// ---------------------------------------------------------------------------
// AI Masking
// ---------------------------------------------------------------------------

/// The starting adjustment of an auto selection at `amount` 0-100.
fn preset(target: &str, amount: f32) -> LocalAdjust {
    let k = amount.clamp(0.0, 100.0) / 100.0;
    let a = match target {
        "subject" => LocalAdjust { exposure: 0.3, contrast: 10.0, shadows: 15.0, saturation: 8.0, ..Default::default() },
        "sky" => LocalAdjust { exposure: -0.4, contrast: 10.0, highlights: -35.0, saturation: 15.0, ..Default::default() },
        "background" => LocalAdjust { exposure: -0.35, contrast: -5.0, saturation: -15.0, ..Default::default() },
        "people" => LocalAdjust { exposure: 0.2, shadows: 15.0, ..Default::default() },
        _ => LocalAdjust::default(),
    };
    LocalAdjust {
        exposure: a.exposure * k,
        contrast: a.contrast * k,
        highlights: a.highlights * k,
        shadows: a.shadows * k,
        whites: 0.0,
        blacks: 0.0,
        temperature_shift: 0.0,
        tint_shift: 0.0,
        saturation: a.saturation * k,
    }
}

pub fn masks(core: &Arc<Core>, ids: &[i64], s: &MaskSettings, provider: &str, job: &Job) -> Result<Value, String> {
    let photos = catalog::photos(&core.db(), ids)?;
    let n = photos.len();
    let (mut rows, mut total) = (vec![], 0);
    for (k, p) in photos.iter().enumerate() {
        if job.cancelled() {
            break;
        }
        job.progress(match s.method.as_str() { "ai" => "Selecting with AI", "sam" => "Selecting with Segment Anything", _ => "Selecting" }, k, n);
        let res = (|| -> Result<Value, String> {
            let small = Rgb::from_working(&*core.proxy(&p.path)?, 512);
            // per selection: its name, target, outline parts (true: a hole in them) and whether it is the inverse
            let mut found: Vec<(String, &str, Vec<(bool, Vec<[f32; 2]>)>, bool)> = vec![];
            let mut missed: Vec<String> = vec![];
            let one = |pts: Vec<[f32; 2]>| vec![(false, pts)];
            // SAM 3, run on the full frame as edited; its encoding is kept for the next target
            let sam = (s.method == "sam")
                .then(|| crate::segment::pick(&core.dir(), "concept", &core.settings.lock().unwrap().ai_concept_model))
                .flatten();
            if s.method == "sam" && sam.is_none() {
                return Err("download a SAM 3 model in the AI tab for Segment Anything masks".into());
            }
            let frame = sam.as_ref().map(|_| core.render(p.id, &crate::develop::Opts { full_frame: true, ..Default::default() })).transpose()?;
            for t in &s.targets {
                let before = found.len();
                let title = |t: &str| t[..1].to_uppercase() + &t[1..];
                match (s.method.as_str(), t.as_str()) {
                    ("ai", "people") => {
                        for (i, (_, pts)) in ai::outline(core, p.id, "each person", provider)?.into_iter().enumerate() {
                            found.push((format!("Person {}", i + 1), "people", one(pts), false));
                        }
                    }
                    ("ai", t @ ("subject" | "sky" | "background")) => {
                        let what = if t == "sky" { "the sky" } else { "the main subject" };
                        if let Some((_, pts)) = ai::outline(core, p.id, what, provider)?.into_iter().next() {
                            found.push((title(t), t, one(pts), t == "background"));
                        }
                    }
                    ("sam", t @ ("subject" | "sky" | "background" | "people")) => {
                        let (Some(model), Some(img)) = (&sam, &frame) else { continue };
                        let key = format!("{}:{}x{}", p.id, img.w, img.h);
                        for (name, polys) in crate::segment::target(&core.dir(), model, img, &key, t)? {
                            found.push((name, t, polys, t == "background"));
                        }
                    }
                    (_, "sky") => found.extend(vision::sky_outline(&small).map(|pts| ("Sky".into(), "sky", one(pts), false))),
                    (_, t @ ("subject" | "background")) => found.extend(vision::subject_outline(&small).map(|pts| (title(t), t, one(pts), t == "background"))),
                    (_, "people") => missed.push("people (needs the AI or Segment Anything method)".into()),
                    (_, t) => missed.push(format!("{t} (unknown target)")),
                }
                if found.len() == before && !missed.iter().any(|m| m.starts_with(t.as_str())) {
                    missed.push(t.clone());
                }
            }
            if found.is_empty() {
                return Ok(json!({ "made": [], "missed": missed }));
            }
            // read after the AI calls, so edits made meanwhile stay
            let mut d = core.develop(p.id)?;
            let mut made = vec![];
            for (name, target, parts, invert) in found {
                // the background: everything but the subject, as one inverse selection (holes and all)
                let components = parts
                    .into_iter()
                    .map(|(hole, points)| MaskComponent { op: if hole { "subtract" } else { "add" }.into(), kind: "polygon".into(), name: target.into(), feather: s.feather, edge_shift: s.edge_shift, points, ..Default::default() })
                    .collect();
                let mut m = Mask { mask_id: format!("mask_auto_{}", &crate::new_token()[..8]), name: name.clone(), invert, components, adjustments: preset(target, s.adjust), ..Default::default() };
                match d.masks.iter_mut().find(|x| x.name == name).filter(|_| s.replace) {
                    Some(x) => {
                        m.mask_id = x.mask_id.clone();
                        m.tuning = x.tuning.take();
                        *x = m;
                    }
                    None => d.masks.push(m),
                }
                made.push(name);
            }
            d.clamp();
            let label = format!("Auto masks: {}", made.join(", "));
            if s.method == "ai" {
                d.ai_provenance_log.push(mcp::provenance(core, provider, &label));
            }
            core.commit(p.id, &d, &label)?;
            Ok(json!({ "made": made, "missed": missed }))
        })();
        let mut row = res.unwrap_or_else(|e| json!({ "error": e }));
        total += row["made"].as_array().map_or(0, |a| a.len());
        row["id"] = json!(p.id);
        row["file"] = json!(file_name(p));
        rows.push(row);
    }
    Ok(json!({ "total": n, "masks": total, "rows": rows }))
}

// ---------------------------------------------------------------------------
// AI Denoise & Super Resolution (Enhance)
// ---------------------------------------------------------------------------

/// A name `stem<suffix>.tif` (e.g. -Enhanced-NR-SR, -HDR) in `dir` that is not taken.
fn enhanced_path(dir: &Path, p: &Photo, suffix: &str) -> PathBuf {
    let stem = Path::new(&p.filename).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "photo".into());
    let stem = format!("{stem}{suffix}");
    let mut out = dir.join(format!("{stem}.tif"));
    let mut k = 1;
    while out.exists() {
        out = dir.join(format!("{stem}-{k}.tif"));
        k += 1;
    }
    out
}

/// Where Enhance and HDR Merge write: `folder`, else next to a local original if that folder is writable,
/// else Pictures/Optixel Enhanced.
fn enhanced_dir(folder: &str, p: &Photo) -> Result<PathBuf, String> {
    let dir = if !folder.trim().is_empty() {
        PathBuf::from(folder.trim())
    } else {
        let next_to = Path::new(&p.path).parent().filter(|d| remote::immich_id(&p.path).is_none() && d.is_dir());
        match next_to.filter(|d| fs::metadata(d).is_ok_and(|m| !m.permissions().readonly())) {
            Some(d) => d.to_path_buf(),
            None => dirs::picture_dir().or_else(dirs::home_dir).unwrap_or_else(std::env::temp_dir).join("Optixel Enhanced"),
        }
    };
    fs::create_dir_all(&dir).map_err(|e| format!("cannot write to {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Catalogs `out`, made from `p` at `size`, as a new master with `p`'s capture data, edits `d` and the
/// keyword `tag`; with `keep`, also `p`'s rating, flag, label and keywords.
fn add_master(core: &Arc<Core>, p: &Photo, out: &Path, size: (usize, usize), d: &Develop, tag: &str, keep: bool) -> Result<i64, String> {
    let meta = catalog::Meta {
        captured_at: p.captured_at.clone(),
        camera: p.camera.clone(),
        lens: p.lens.clone(),
        iso: p.iso,
        aperture: p.aperture,
        shutter: p.shutter.clone(),
        focal_length: p.focal_length,
        orientation: 1,
        latitude: p.latitude,
        longitude: p.longitude,
    };
    let path = out.to_string_lossy().to_string();
    let folder = out.parent().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let name = out.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let id = catalog::add_photo(&core.db(), &path, &folder, &name, &meta, d)?.ok_or("the new file is already in the catalog")?;
    {
        let db = core.db();
        db.execute("UPDATE photos SET width = ?, height = ? WHERE id = ?", (size.0 as i64, size.1 as i64, id)).map_err(es)?;
        let words = catalog::merge_keywords(if keep { &p.keywords } else { "" }, &[tag.to_string()]);
        let label = if keep { Some(if p.color_label.is_empty() { "none".to_string() } else { p.color_label.clone() }) } else { None };
        let flag = if keep { Some(catalog::flag_name(p.flag).to_string()) } else { None };
        catalog::update(&db, &[id], &Culling { rating: keep.then_some(p.rating), flag, color_label: label, keywords: Some(words) })?;
    }
    core.refresh(id, d)?;
    Ok(id)
}

pub fn enhance(core: &Arc<Core>, ids: &[i64], s: &EnhanceSettings, job: &Job) -> Result<Value, String> {
    let removing = s.engine == "ai" && !s.remove_strokes.is_empty();
    if !s.denoise && !s.super_resolution && !removing {
        return Err("choose Denoise, Super Resolution or both".into());
    }
    let photos = masters(core, ids)?;
    let n = photos.len();
    let (mut rows, mut made) = (vec![], 0);
    for (k, p) in photos.iter().enumerate() {
        if job.cancelled() {
            break;
        }
        job.progress(if s.super_resolution { "Enhancing (denoise and enlarge)" } else { "Denoising" }, k, n);
        let res = (|| -> Result<Value, String> {
            let file = core.local_file(&p.path)?;
            let img = core.decode(&file, None)?;
            let (w0, h0) = (img.w, img.h);
            let fit = (s.max_megapixels.max(1.0) * 1e6 / (w0 * h0) as f32).sqrt();
            let scale = if s.super_resolution { s.scale.clamp(1.0, 4.0).min(fit).max(1.0) } else { 1.0 };
            let scaled = scale > 1.01;
            let img = if s.engine == "command" {
                let dir = std::env::temp_dir().join(format!("optixel-enhance-{}", crate::new_token()));
                fs::create_dir_all(&dir).map_err(es)?;
                let (input, output) = (dir.join("input.png"), dir.join("output.png"));
                let r = develop::export(&img, &input, "PNG_16", "sRGB", 100, None)
                    .and_then(|_| enhance::run_command(&s.command, &input, &output, scale.round().max(1.0) as u32, if s.denoise { s.strength } else { 0.0 }, Duration::from_secs(1800)))
                    .and_then(|_| develop::load(&output, None));
                let _ = fs::remove_dir_all(&dir);
                r?
            } else if s.engine == "ai" {
                let img = if removing {
                    // the selection the user confirmed, found again on the same preview frame, then at full size
                    let proxy = core.proxy(&p.path)?;
                    let alpha = crate::brush::snap(&proxy, &s.remove_strokes)?;
                    let hole = aimodel::hole_from_alpha(&alpha, proxy.w, proxy.h, img.w, img.h).ok_or("nothing is selected to remove")?;
                    job.progress("Removing the object", 0, 1);
                    aimodel::inpaint(&core.dir(), &s.ai_model, img, &hole, s.remove_grow.clamp(0.0, 40.0).round() as usize, &s.remove_prompt, &|done, total| {
                        job.progress("Filling in", done, total);
                        !job.cancelled()
                    })?
                } else {
                    img
                };
                let img = if s.denoise {
                    let vram = aimodel::hardware().vram_mb;
                    // a RAW is denoised in its camera's own RGB, before the colour matrix mixes the channels' noise
                    let space = develop::is_raw(&file).then(|| fs::read(&file).ok().and_then(|b| crate::libraw::camera_space(&b))).flatten();
                    aimodel::denoise_in(&core.dir(), &s.ai_model, img, s.ai_strength, vram, space, &|done, total| {
                        job.progress("AI denoising", done, total);
                        !job.cancelled()
                    })?
                } else {
                    img
                };
                if scaled { enhance::upscale(img, scale, s.sharpen) } else { img }
            } else {
                let img = if s.denoise { enhance::denoise(img, s.strength, s.detail, s.color) } else { img };
                if scaled { enhance::upscale(img, scale, s.sharpen) } else { img }
            };
            let out = enhanced_path(&enhanced_dir(&s.folder, p)?, p, &format!("-Enhanced{}{}{}", if s.denoise { "-NR" } else { "" }, if scaled { "-SR" } else { "" }, if removing { "-Removed" } else { "" }));
            let float = s.format == "float" || (s.format == "auto" && develop::is_raw(&file));
            if float {
                develop::export_linear(&img, &out)?;
            } else {
                develop::export(&img, &out, "TIFF_16", "sRGB", 100, None)?;
            }
            // the new master: the original's edits, rating and keywords
            let mut d = if s.copy_edits { core.develop(p.id)? } else { Develop::default() };
            if s.denoise {
                (d.detail.luminance_noise_reduction, d.detail.color_noise_reduction) = (0.0, 0.0);
            }
            let id = add_master(core, p, &out, (img.w, img.h), &d, "enhanced", s.copy_edits)?;
            Ok(json!({ "new_id": id, "output": out.to_string_lossy(), "size": format!("{w0} × {h0} → {} × {}", img.w, img.h) }))
        })();
        let mut row = res.unwrap_or_else(|e| json!({ "error": e }));
        made += row.get("new_id").is_some() as usize;
        row["id"] = json!(p.id);
        row["file"] = json!(file_name(p));
        rows.push(row);
    }
    core.emit("catalog-changed", json!({}));
    Ok(json!({ "total": n, "made": made, "rows": rows }))
}

// ---------------------------------------------------------------------------
// HDR Merge
// ---------------------------------------------------------------------------

/// Photos in capture order, split where more than `bracket_seconds` pass between two (a photo without a
/// capture time stands alone); "all": one group.
fn brackets<'a>(photos: &'a [Photo], s: &HdrSettings) -> Vec<Vec<&'a Photo>> {
    if s.grouping == "all" {
        return vec![photos.iter().collect()];
    }
    let (mut out, mut last): (Vec<Vec<&Photo>>, Option<i64>) = (vec![], None);
    for p in photos {
        let t = p.captured_at.as_deref().and_then(seconds);
        match (last, t, out.last_mut()) {
            (Some(a), Some(b), Some(g)) if (b - a) as f32 <= s.bracket_seconds => g.push(p),
            _ => out.push(vec![p]),
        }
        last = t;
    }
    out
}

/// Merge each exposure bracket among `ids` into a new 32-bit float master (…-HDR.tif) next to them.
pub fn hdr(core: &Arc<Core>, ids: &[i64], s: &HdrSettings, job: &Job) -> Result<Value, String> {
    let mut photos = masters(core, ids)?;
    photos.sort_by_key(|p| (p.captured_at.as_deref().and_then(seconds), p.filename.clone()));
    let groups = brackets(&photos, s);
    let deghost = match s.deghost.as_str() {
        "low" => 1.0,
        "medium" => 0.5,
        "high" => 0.25,
        _ => 0.0,
    };
    let n = groups.len();
    let (mut rows, mut made) = (vec![], 0);
    for (k, g) in groups.iter().enumerate() {
        if job.cancelled() {
            break;
        }
        job.progress("Merging to HDR", k, n);
        let res = (|| -> Result<Value, String> {
            if g.len() < 2 {
                return Err(format!("no bracket: nothing else was taken within {} s", s.bracket_seconds));
            }
            let frames = g.iter().map(|p| core.decode(&core.local_file(&p.path)?, None)).collect::<Result<Vec<_>, _>>()?;
            let m = enhance::merge_hdr(frames, s.align, deghost)?;
            let (lo, hi) = m.ev.iter().fold((0f32, 0f32), |(a, b), &e| (a.min(e), b.max(e)));
            if hi - lo < 0.5 {
                return Err("the photos have the same exposure: a burst, not a bracket".into());
            }
            let p = g[m.reference];
            let out = enhanced_path(&enhanced_dir(&s.folder, p)?, p, "-HDR");
            develop::export_linear(&m.img, &out)?;
            let mut d = if s.auto_tone { develop::auto(&m.img, &Develop::default(), "natural")? } else { Develop::default() };
            if s.auto_tone {
                d.hdr = develop::auto_hdr(&m.img);
            }
            let id = add_master(core, p, &out, (m.img.w, m.img.h), &d, "hdr", false)?;
            let evs: Vec<String> = m.ev.iter().map(|e| format!("{e:+.1}")).collect();
            let moved = m.shift.iter().map(|s| s.0.abs().max(s.1.abs())).max().unwrap_or(0);
            Ok(json!({ "new_id": id, "output": out.to_string_lossy(), "frames": g.len(), "evs": format!("{} EV", evs.join(", ")), "aligned_px": moved }))
        })();
        let mut row = res.unwrap_or_else(|e| json!({ "error": e }));
        made += row.get("new_id").is_some() as usize;
        row["id"] = json!(g[0].id);
        row["ids"] = json!(g.iter().map(|p| p.id).collect::<Vec<_>>());
        row["file"] = json!(g.iter().map(|p| file_name(p)).collect::<Vec<_>>().join(", "));
        rows.push(row);
    }
    core.emit("catalog-changed", json!({}));
    Ok(json!({ "total": photos.len(), "brackets": groups.iter().filter(|g| g.len() > 1).count(), "made": made, "rows": rows }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_times_and_similarity() {
        assert_eq!(seconds("2024-05-01 12:34:56").unwrap() - seconds("2024-05-01 12:34:54").unwrap(), 2);
        assert_eq!(seconds("2024-03-01 00:00:00").unwrap() - seconds("2024-02-28 23:59:59").unwrap(), 86_401, "leap day");
        assert_eq!(seconds("2024:05:01 12:34:56"), seconds("2024-05-01 12:34:56"));
        assert_eq!(seconds("1970-01-01 00:00:00"), Some(0));
        assert!(seconds("garbage").is_none());
        assert_eq!((max_bits(100.0), max_bits(90.0), max_bits(0.0)), (0, 3, 32));
    }

    #[test]
    fn blend_moves_part_of_the_way() {
        let mut to = Develop::default();
        to.basic_tone.exposure = 1.0;
        to.white_balance.temperature_k = 3000.0;
        to.profile = "Monochrome".into();
        let d = blend(&Develop::default(), &to, 0.5);
        assert!((d.basic_tone.exposure - 0.5).abs() < 1e-6);
        // halfway in mireds: 1e6 / ((181.8 + 333.3) / 2) ≈ 3882 K
        assert!((d.white_balance.temperature_k - 3882.0).abs() < 5.0, "{}", d.white_balance.temperature_k);
        assert_eq!(d.profile, "Monochrome");
        assert_eq!(blend(&Develop::default(), &to, 0.2).profile, "");
        assert_eq!(blend(&Develop::default(), &to, 1.0), to);
    }

    #[test]
    fn settings_take_overrides_and_keep_the_rest() {
        let s = with(&CullSettings::default(), &json!({ "blur_threshold": 50, "labels": true })).unwrap();
        assert_eq!((s.blur_threshold, s.labels, s.stars), (50.0, true, true));
        assert!(with(&CullSettings::default(), &json!({ "labels": "yes" })).is_err());
        // older settings files without the new fields still load
        let a: AutoSettings = serde_json::from_value(json!({ "cull": { "stars": false } })).unwrap();
        assert!(!a.cull.stars && a.cull.keywords && a.edit == EditSettings::default());
    }

    #[test]
    fn photo_types_map_to_profiles() {
        let t = |k: &[&str]| photo_type(&k.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(t(&["mountain", "lake", "sunrise"]), "landscape");
        assert_eq!(t(&["woman", "beach", "sunset"]), "portrait", "the first keyword counts double, ties go to the earlier type");
        assert_eq!(t(&["street", "people", "city"]), "street");
        assert_eq!(t(&["Pasta", "plate", "restaurant"]), "food");
        assert_eq!(t(&["skyscraper", "glass facade"]), "architecture");
        assert_eq!(t(&["city lights", "neon", "street"]), "night");
        assert_eq!(t(&["bee", "flowers", "garden"]), "nature");
        assert_eq!(t(&["portrait", "black and white"]), "bw");
        assert_eq!(t(&["2024", "may", "horizontal", "cathedrals"]), "architecture");
        assert_eq!(t(&["2024", "may", "horizontal"]), "natural");
        for (ty, _) in TYPES {
            assert!(develop::PROFILES.contains(ty), "{ty} is not a profile");
            assert!(develop::look_on(&Develop::default(), ty).unwrap() != Develop::default(), "{ty} has no look");
        }
    }

    fn photo(id: i64, w: i64, h: i64, captured: &str) -> Photo {
        Photo {
            id,
            path: format!("/p/{id}.jpg"),
            folder: "/p".into(),
            filename: format!("{id}.jpg"),
            master_id: None,
            copy_name: None,
            width: w,
            height: h,
            captured_at: Some(captured.into()),
            camera: None,
            lens: None,
            iso: Some(6400),
            aperture: None,
            shutter: Some("2".into()),
            focal_length: Some(16.0),
            rating: 0,
            flag: 0,
            color_label: String::new(),
            keywords: "trip".into(),
            edited: false,
            thumb_v: 0,
            deleted_at: None,
            latitude: None,
            longitude: None,
        }
    }

    #[test]
    fn keywords_from_capture_data_and_analysis() {
        let m = Metrics { colorfulness: 30.0, colors: vec!["blue".into()], ..Default::default() };
        let k = keywords(&photo(1, 6000, 4000, "2024-05-01 19:02:03"), &m);
        for w in ["2024", "may", "evening", "horizontal", "blue", "high iso", "long exposure", "wide angle"] {
            assert!(k.contains(&w.to_string()), "{w} missing from {k:?}");
        }
        let mono = keywords(&photo(1, 1000, 3000, "2024-12-01 03:00:00"), &Metrics { colorfulness: 2.0, ..Default::default() });
        assert!(mono.contains(&"black and white".to_string()) && mono.contains(&"panorama".to_string()) && mono.contains(&"night".to_string()));
        assert_eq!(catalog::merge_keywords("trip, Beach", &["beach".into(), "sunset".into()]), "trip, Beach, sunset");
    }

    /// End to end on a temporary catalog: a sharp photo and two near copies of a blurred one, culled, then
    /// searched for blur and duplicates, edited, masked and enhanced.
    #[test]
    fn auto_functions_on_a_catalog() {
        let dir = std::env::temp_dir().join(format!("ols-auto-test-{}", crate::new_token()));
        let core = Arc::new(Core::open(dir.join("data")).unwrap());
        let photos = dir.join("photos");
        fs::create_dir_all(&photos).unwrap();
        // textured squares on a sky gradient, sharp and blurred
        let (w, h) = (600u32, 400u32);
        let scene = |blur: f32, shift: u32| {
            let img = image::RgbImage::from_fn(w, h, |x, y| {
                let cell = ((x + shift) / 25 + y / 25).is_multiple_of(2);
                if y < 150 { image::Rgb([90, 150, 230]) } else if cell { image::Rgb([200, 120, 60]) } else { image::Rgb([60, 90, 40]) }
            });
            if blur > 0.0 { image::imageops::blur(&img, blur) } else { img }
        };
        scene(0.0, 0).save(photos.join("sharp.jpg")).unwrap();
        scene(4.0, 0).save(photos.join("soft.jpg")).unwrap();
        scene(4.0, 0).save(photos.join("soft copy.jpg")).unwrap();
        let ids = catalog::import(&core, &photos, &|_, _| {}).unwrap();
        assert_eq!(ids.len(), 3);
        let by_name = |name: &str| catalog::query(&core.db(), &Default::default()).unwrap().into_iter().find(|p| p.filename == name).unwrap();
        let job = Job::start(&core, "test");

        let r = run(&core, "cull", &ids, &json!({ "detail": "full" }), &job).unwrap();
        assert_eq!(r["total"], 3);
        let (sharp, soft) = (by_name("sharp.jpg"), by_name("soft.jpg"));
        assert!(sharp.flag != -1 && sharp.rating >= 3, "sharp photo: {sharp:?} {r}");
        assert_eq!(soft.flag, -1, "blurred photo not rejected: {r}");
        assert!(sharp.keywords.contains("horizontal") && soft.keywords.contains("blurry"), "{} / {}", sharp.keywords, soft.keywords);
        assert!(catalog::analysis(&core.db(), &sharp.path, &format!("{}/l{}", stamp(&sharp, "full", 0), crate::aimodel::model_file(&core.dir(), "face-landmarks-2dfan4", crate::faceswap::LANDMARKS).is_some() as u8)).is_some(), "measurements are cached");

        let r = run(&core, "blur", &ids, &json!({ "detail": "full", "action": "label" }), &job).unwrap();
        assert_eq!(r["found"], 2, "{r}");
        assert_eq!(by_name("soft copy.jpg").color_label, "red");
        assert_eq!(by_name("sharp.jpg").color_label, "");

        let r = run(&core, "duplicates", &ids, &json!({ "mode": "exact", "action": "keyword" }), &job).unwrap();
        assert_eq!((r["groups"].as_u64(), r["rows"][0]["kind"].as_str()), (Some(1), Some("exact")), "{r}");
        let copies = [by_name("soft.jpg"), by_name("soft copy.jpg")];
        assert_eq!(copies.iter().filter(|p| p.keywords.contains("duplicate")).count(), 1, "one copy stays unmarked");

        let r = run(&core, "edit", &[sharp.id], &json!({ "skip_edited": false, "strength": 50 }), &job).unwrap();
        assert_eq!(r["edited"], 1, "{r}");
        assert!(core.develop(sharp.id).unwrap() != Develop::default());
        assert_eq!(run(&core, "edit", &[sharp.id], &json!({}), &job).unwrap()["rows"][0]["status"], "skipped");
        let r = run(&core, "edit", &[soft.id], &json!({ "profile": "type", "ai_type": false }), &job).unwrap();
        assert!(r["rows"][0]["note"].as_str().unwrap().ends_with("natural look"), "{r}");

        let r = run(&core, "mask", &[sharp.id], &json!({ "targets": ["sky"], "adjust": 50 }), &job).unwrap();
        assert_eq!(r["masks"], 1, "{r}");
        let sky = core.develop(sharp.id).unwrap().masks.into_iter().find(|m| m.name == "Sky").expect("a Sky selection");
        assert!(sky.adjustments.exposure < 0.0 && sky.components[0].kind == "polygon");

        let r = run(&core, "enhance", &[soft.id], &json!({ "denoise": true, "super_resolution": true, "scale": 2, "max_megapixels": 1 }), &job).unwrap();
        let new = r["rows"][0]["new_id"].as_i64().unwrap_or_else(|| panic!("{r}"));
        let out = PathBuf::from(r["rows"][0]["output"].as_str().unwrap());
        assert!(out.starts_with(&photos) && out.file_name().unwrap().to_string_lossy() == "soft-Enhanced-NR-SR.tif", "{out:?}");
        let p = core.photo(new).unwrap();
        // a 1 MP cap: 600 × 400 grows by √(1e6 / 240000) ≈ 2.04, so by 2
        assert_eq!((p.width, p.height, p.flag), (1200, 800, -1));
        assert!(p.keywords.contains("enhanced"));
        let back = develop::load(&out, None).unwrap();
        assert_eq!((back.w, back.h), (1200, 800));
        // a two-shot bracket, 2 stops apart, a second after each other
        let bracket = photos.join("bracket");
        fs::create_dir_all(&bracket).unwrap();
        let sharp_scene = scene(0.0, 0);
        for (name, k) in [("under.jpg", 0.4f32), ("over.jpg", 1.6)] {
            let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(sharp_scene.get_pixel(x, y).0.map(|c| (c as f32 * k).min(255.0) as u8)));
            img.save(bracket.join(name)).unwrap();
        }
        let shots = catalog::import(&core, &bracket, &|_, _| {}).unwrap();
        let r = run(&core, "hdr", &shots, &json!({ "grouping": "all" }), &job).unwrap();
        let new = r["rows"][0]["new_id"].as_i64().unwrap_or_else(|| panic!("{r}"));
        let p = core.photo(new).unwrap();
        assert!(p.filename.ends_with("-HDR.tif") && p.keywords.contains("hdr") && (p.width, p.height) == (600, 400), "{p:?}");
        assert!(core.develop(new).unwrap().hdr.enabled, "the merge starts tone-mapped");
        assert!(run(&core, "hdr", &[shots[0]], &json!({}), &job).unwrap()["rows"][0]["error"].as_str().unwrap().contains("no bracket"));
        drop(job);
        assert!(core.jobs.lock().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
