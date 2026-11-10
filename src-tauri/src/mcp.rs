//! Built-in MCP server (spec §4). JSON-RPC 2.0 over Streamable HTTP
//! (`http://127.0.0.1:41920/mcp`, bearer token, JSON responses, no server-initiated
//! stream) and over stdio (`--mcp-stdio`: forwards to the running app, or runs headless
//! against the catalog when the app is closed).

use crate::{
    Core, Settings, auto, catalog,
    develop::{self, Opts},
};
use base64::Engine;
use serde_json::{Value, json};
use std::{
    io::{BufRead, Read, Write},
    sync::Arc,
    time::Duration,
};

const INSTRUCTIONS: &str = "Optixel Studio photo catalog and non-destructive RAW editor. \
Find photos with catalog_query, look at them with photo_inspect (preview image + histogram telemetry), \
edit with develop_apply_parameters / mask_create_and_edit, verify with photo_inspect again. \
Select an object by tracing it as a polygon mask; remove objects with retouch_remove_objects. \
Batch work (culling, finding blurred photos or duplicates, auto edits, auto masks, denoise and super \
resolution) runs through auto_run. Every mutation is one undoable history step in the app.";

pub fn tools() -> Value {
    let ids = json!({ "type": "array", "items": { "type": "string" } });
    let outline = json!({ "type": "array", "minItems": 3, "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
        "description": "outline vertices [x, y] as fractions (0-1) of the full uncropped frame, (0,0) = top-left; trace the edge closely, 12-60 points" });
    let wheel = json!({ "type": "object", "properties": { "hue": { "type": "number" }, "sat": { "type": "number" }, "lum": { "type": "number" } } });
    let parameters = json!({
        "type": "object",
        "properties": {
            "profile": { "type": "string", "enum": ["Color", "Monochrome"] },
            "white_balance": { "type": "object", "properties": {
                "temperature_k": { "type": "number", "minimum": 2000, "maximum": 50000, "description": "5500 = as shot; higher = warmer" },
                "tint": { "type": "number", "minimum": -150, "maximum": 150, "description": "+ magenta / - green" } } },
            "tone": { "type": "object", "properties": {
                "exposure": { "type": "number", "minimum": -5.0, "maximum": 5.0, "description": "EV stops" },
                "contrast": { "type": "number", "minimum": -100, "maximum": 100 },
                "highlights": { "type": "number", "minimum": -100, "maximum": 100 },
                "shadows": { "type": "number", "minimum": -100, "maximum": 100 },
                "whites": { "type": "number", "minimum": -100, "maximum": 100 },
                "blacks": { "type": "number", "minimum": -100, "maximum": 100 } } },
            "presence": { "type": "object", "properties": {
                "texture": { "type": "number", "minimum": -100, "maximum": 100 },
                "clarity": { "type": "number", "minimum": -100, "maximum": 100 },
                "dehaze": { "type": "number", "minimum": -100, "maximum": 100 },
                "vibrance": { "type": "number", "minimum": -100, "maximum": 100 },
                "saturation": { "type": "number", "minimum": -100, "maximum": 100 } } },
            "tone_curve": { "type": "object", "description": "parametric {shadows,darks,lights,highlights: -100..100}; point_luma/point_red/point_green/point_blue: [[in,out],...] in 0-255" },
            "color_mixer": { "type": "object", "description": "hue/saturation/luminance: arrays of 8 values (-100..100) for red, orange, yellow, green, aqua, blue, purple, magenta" },
            "color_grading_3way": { "type": "object", "properties": {
                "shadows": wheel, "midtones": wheel, "highlights": wheel, "global": wheel,
                "blending": { "type": "number", "minimum": 0, "maximum": 100 },
                "balance": { "type": "number", "minimum": -100, "maximum": 100 } },
                "description": "hue 0-360 (HSL wheel, 0 = red), sat 0-100, lum -100..100" },
            "point_color_swatches": { "type": "array", "items": { "type": "object", "properties": {
                "src_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[Oklch hue deg, chroma*250, L*100]" },
                "shift_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                "range_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                "falloff": { "type": "number", "minimum": 0, "maximum": 100 } } } },
            "detail": { "type": "object", "description": "sharpening 0-150, sharpen_radius 0.5-3, sharpen_masking 0-100, color_noise_reduction 0-100, luminance_noise_reduction 0-100, luminance_detail 0-100" },
            "effects": { "type": "object", "description": "vignette -100..100, vignette_midpoint 0-100, vignette_feather 0-100" },
            "grain": { "type": "object", "description": "amount 0-100, size 0-100, roughness 0-100" },
            "calibration": { "type": "object", "description": "shadows_tint (+ magenta), red_hue, red_saturation, green_hue, green_saturation, blue_hue, blue_saturation: -100..100" },
            "lens": { "type": "object", "description": "distortion -100..100 (+ corrects barrel), ca_red_cyan / ca_blue_yellow -100..100 (lateral chromatic aberration), defringe_purple / defringe_green 0-20 with defringe_purple_hue / defringe_green_hue [low, high] 0-100, vignetting -100..100 (+ brightens corners), vignetting_midpoint 0-100" },
            "transform": { "type": "object", "description": "perspective: vertical / horizontal -100..100 (keystone), rotate -10..10 degrees, aspect -100..100, scale 50-150, offset_x / offset_y -100..100, constrain_crop bool" },
            "levels": { "type": "object", "description": "Photoshop Levels: rgb (composite), red, green, blue, each {black 0-253, gamma 0.1-9.99, white 2-255, out_black 0-255, out_white 0-255} in display values" },
            "hue_saturation": { "type": "object", "description": "Photoshop Hue/Saturation: hue -180..180 degrees, saturation -100..100, lightness -100..100; colorize bool with colorize_hue 0-360 and colorize_saturation 0-100" },
            "selective_color": { "type": "object", "description": "Photoshop Selective Color: reds, yellows, greens, cyans, blues, magentas, whites, neutrals, blacks: [cyan, magenta, yellow, black] -100..100; absolute bool (default relative)" },
            "channel_mixer": { "type": "object", "description": "Photoshop Channel Mixer: red, green, blue outputs as [red, green, blue, constant] % -200..200 (identity [100,0,0,0] etc.); monochrome bool uses gray [r, g, b, constant]" },
            "gradient_map": { "type": "object", "description": "Photoshop Gradient Map: amount 0-100, shadows / highlights sRGB [r, g, b] 0-1" },
            "hdr": { "type": "object", "description": "HDR Toning: enabled bool; method fusion (default: merge a simulated bracket of frames 2-9 exposures ev_spacing 0.5-4 stops apart, weighted by exposedness / contrast_weight / saturation_weight 0-200; natural 0-100 keeps bright areas such as skies bright, 60 by default) | local (local adaptation) | equalize (equalize histogram) | gamma (exposure and gamma) | highlights (highlight compression); local only: radius 0-100 (region size), edge 0-100 (how firmly it stops at edges), smooth_edges bool (fewer halos), compression 0-100 (squeeze the range), detail -100..300 %, shadows / highlights -100..100; any method: exposure -5..5 EV, gamma 0.1-3 (above 1 opens the shadows), fine_detail -100..300 %, vibrance / saturation -100..100, curve [[x, y], ...] 0-255 on display brightness" },
            "filters": { "type": "object", "description": "blur 0-100 (Gaussian), invert bool, posterize levels (0 = off, 2-255), threshold (0 = off, 1-255)" },
            "image_rotation": { "type": "object", "description": "angle 0 / 90 / 180 / 270 clockwise, flip_h, flip_v bool; the crop is in the rotated frame" },
            "crop": { "type": "object", "description": "x, y, w, h normalised 0-1 in the straightened, rotated frame; angle -45..45 degrees" },
            "raw": { "type": "object", "description": "Native C++ LibRaw engine & optical calibration: engine (''=auto | 'libraw'), demosaic ('rcd' | 'amaze' | 'lmmse' | 'ahd' | 'dcb' | 'dht' | 'aahd' | 'ppg' | 'vng' | 'linear'), highlights ('clip' | 'unclip' | 'blend' | 'rebuild'), black_offset (-512..512 ADU), raw_denoise (0..100 linear Bayer/X-Trans pre-color sensor denoising)" }
        }
    });
    json!([
        {
            "name": "catalog_query",
            "description": "Search and filter the Optixel catalog by rating, flag, camera/lens and text. semantic_query is matched against filenames, keywords, camera and lens (no vision embedding model is installed in this build). min_eye_sharpness filters on the main subject's sharpness (0-100) measured by the auto functions; photos not analysed yet are left out.",
            "inputSchema": { "type": "object", "properties": {
                "semantic_query": { "type": "string" },
                "min_rating": { "type": "integer", "minimum": 0, "maximum": 5 },
                "flag_status": { "type": "string", "enum": ["pick", "unflagged", "reject", "any"] },
                "min_eye_sharpness": { "type": "number", "minimum": 0, "maximum": 100 },
                "eyes_open_only": { "type": "boolean" },
                "camera_or_lens": { "type": "string" },
                "limit": { "type": "integer", "default": 50 } } }
        },
        {
            "name": "culling_batch_update",
            "description": "Set star ratings, pick/reject flags, color labels, or create virtual copies for a list of photos.",
            "inputSchema": { "type": "object", "required": ["photo_ids"], "properties": {
                "photo_ids": ids,
                "rating": { "type": "integer", "minimum": 0, "maximum": 5 },
                "flag": { "type": "string", "enum": ["pick", "unflagged", "reject"] },
                "color_label": { "type": "string", "enum": ["none", "red", "yellow", "green", "blue", "purple"] },
                "create_virtual_copy": { "type": "boolean", "default": false } } }
        },
        {
            "name": "develop_apply_parameters",
            "description": "Apply non-destructive global white balance, tone, presence, curve, color mixer, 3-way color grading, Point Color, calibration, detail, effects, grain, lens correction, perspective transform, Photoshop-style (levels, hue/saturation, selective color, channel mixer, gradient map, filters), image rotation or crop adjustments to one or more photos. Unknown keys are reported back as ignored_keys.",
            "inputSchema": { "type": "object", "required": ["photo_ids", "parameters"], "properties": {
                "photo_ids": ids,
                "mode": { "type": "string", "enum": ["absolute", "relative_delta"], "default": "absolute" },
                "parameters": parameters } }
        },
        {
            "name": "mask_create_and_edit",
            "description": "Create (or replace, by mask_name) a boolean mask and its local adjustments. Supported source_type values in this build: polygon (points: outline of an object or area, e.g. to select the castle, a person or the sky; the edge snaps to the photo), linear_gradient (start/end: [x,y] 0-1, full effect at start), radial_gradient (center [x,y], radius [rx,ry] as fractions of width/height), luminance_range (range [low,high] 0-1), color_range (color: sRGB [r,g,b] 0-1, tolerance 0-100). Coordinates refer to the full uncropped frame (photo_inspect full_frame). AI semantic types (ai_subject, ai_sky…) need segmentation models that are not installed: trace a polygon instead. A selection with all-zero local_adjustments changes nothing until adjusted.",
            "inputSchema": { "type": "object", "required": ["photo_id", "mask_name", "operations", "local_adjustments"], "properties": {
                "photo_id": { "type": "string" },
                "mask_name": { "type": "string" },
                "operations": { "type": "array", "items": { "type": "object", "required": ["boolean_op", "source_type"], "properties": {
                    "boolean_op": { "type": "string", "enum": ["add", "subtract", "intersect"] },
                    "source_type": { "type": "string", "enum": develop::MASK_TYPES },
                    "invert": { "type": "boolean", "default": false },
                    "feather": { "type": "number", "minimum": -100, "maximum": 100, "default": 0 },
                    "edge_shift": { "type": "number", "minimum": -100, "maximum": 100, "default": 0 },
                    "start": { "type": "array", "items": { "type": "number" } },
                    "end": { "type": "array", "items": { "type": "number" } },
                    "center": { "type": "array", "items": { "type": "number" } },
                    "radius": { "type": "array", "items": { "type": "number" } },
                    "range": { "type": "array", "items": { "type": "number" } },
                    "color": { "type": "array", "items": { "type": "number" } },
                    "tolerance": { "type": "number", "minimum": 0, "maximum": 100 },
                    "points": outline,
                    "name": { "type": "string", "description": "polygon: what the region is, e.g. \"castle\" (defaults to mask_name)" } } } },
                "local_adjustments": { "type": "object", "description": "exposure (-5..5), contrast, highlights, shadows, whites, blacks, temperature_shift, tint_shift, saturation (-100..100)" } } }
        },
        {
            "name": "batch_export",
            "description": "Export photos at full resolution with their current edits. Formats: JPEG, AVIF, PNG_16, TIFF_16. Color spaces: sRGB, DisplayP3, AdobeRGB, ProPhotoRGB, ACEScg, Rec2020_HDR (PQ, PNG_16/TIFF_16 only). Existing files are never overwritten.",
            "inputSchema": { "type": "object", "required": ["photo_ids", "destination_folder", "format"], "properties": {
                "photo_ids": ids,
                "destination_folder": { "type": "string" },
                "format": { "type": "string", "enum": develop::EXPORT_FORMATS },
                "color_space": { "type": "string", "enum": ["sRGB", "DisplayP3", "AdobeRGB", "ProPhotoRGB", "ACEScg", "Rec2020_HDR"] },
                "quality": { "type": "integer", "minimum": 1, "maximum": 100, "default": 90 },
                "attach_c2pa_credentials": { "type": "boolean", "default": false } } }
        },
        {
            "name": "photo_inspect",
            "description": "Rendered preview (JPEG) of a photo with its current edits, plus telemetry: 256-bin RGB/luma histograms, clipping percentages, luma percentiles and dominant Oklch colours. Same data as the optixel://photo/{id}/preview and /telemetry resources.",
            "inputSchema": { "type": "object", "required": ["photo_id"], "properties": {
                "photo_id": { "type": "string" },
                "max_edge": { "type": "integer", "enum": [512, 1024, 1536], "default": 1024 },
                "full_frame": { "type": "boolean", "default": false, "description": "skip lens distortion, transform, straighten and crop: the frame that mask and retouch coordinates refer to" },
                "show_mask": { "type": "integer", "minimum": 0, "description": "tint the coverage of the mask at this index red, to check a selection" },
                "mark": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2, "description": "[x, y] in the full frame (with full_frame): draw a red ring there, e.g. where the user clicked" },
                "include_histogram_bins": { "type": "boolean", "default": false } } }
        },
        {
            "name": "develop_auto",
            "description": "Auto fine tune: measure white balance and tone on each photo and apply them with a look profile (natural = plain Auto). Replaces white balance, Basic, presence and colour/B&W, plus the panels the profile uses (film: tone curve, colour grading, grain; moody: vignette); other edits stay.",
            "inputSchema": { "type": "object", "required": ["photo_ids"], "properties": {
                "photo_ids": ids,
                "profile": { "type": "string", "enum": develop::PROFILES, "default": "natural" } } }
        },
        {
            "name": "auto_run",
            "description": "Run one of Optixel's auto functions on photos with its advanced settings (omitted settings keep the user's saved values): cull = Auto Catalog & Culling (stars, picks of each burst, rejects of blurred or badly exposed photos, labels, keywords); edit = auto non-destructive editing to a look profile; mask = AI masking (subject, sky, background, people selections); enhance = denoise and super resolution into a new master file; hdr = HDR Merge of exposure brackets into a new 32-bit float master; blur = find photos whose main subject is blurred; duplicates = find exact and similar duplicates. Returns a summary and a row per photo (or per duplicate group).",
            "inputSchema": { "type": "object", "required": ["function", "photo_ids"], "properties": {
                "function": { "type": "string", "enum": auto::FUNCTIONS },
                "photo_ids": ids,
                "settings": { "type": "object", "description": "cull: detail (fast|screen|full), reject_blurry, blur_threshold 0-100, use_subject, reject_exposure, clip_limit %, burst_seconds, similarity 0-100, pick_best, reject_similar, stars, labels, keywords, ai_keywords, keep_manual. edit: profile, strength 0-100, white_balance, tone, look, upright (off|level|auto|vertical|full), chromatic_aberration, noise_reduction, skip_edited, virtual_copy. mask: targets [subject|sky|background|people], method (builtin|ai), feather, edge_shift, replace, adjust 0-100. enhance: denoise, strength, detail, color, super_resolution, scale 1-4, sharpen, max_megapixels, engine (builtin|command), format (auto|float|tiff16), folder, copy_edits. hdr: grouping (brackets|all), bracket_seconds, align, deghost (off|low|medium|high), auto_tone, folder. blur: detail, threshold, subject (builtin|ai), motion, missed_focus, action (none|reject|label|keyword). duplicates: mode (exact|similar|both), similarity, window_minutes, keep (best|largest|sharpest|oldest|newest), action." } } }
        },
        {
            "name": "ai_reconstruct_and_retouch",
            "description": "AI Denoise or Super Resolution of one photo: writes an enhanced copy (a new master next to the original, added to the catalog with the original's edits) and returns it. amount is the denoise strength; super_resolution enlarges 2× (capped at the user's megapixel limit). The other tasks need generative models that are not installed: retouch_remove_objects heals people and distractions.",
            "inputSchema": { "type": "object", "required": ["photo_id", "task"], "properties": {
                "photo_id": { "type": "string" },
                "task": { "type": "string", "enum": ["ai_denoise", "super_resolution", "remove_reflections", "remove_background_people", "remove_sensor_dust", "ai_lens_blur", "generative_remove"] },
                "amount": { "type": "number", "minimum": 0, "maximum": 100, "default": 50 } } }
        },
        {
            "name": "retouch_remove_objects",
            "description": "Remove objects or distractions (people, cars, signs, litter, dust spots) from a photo. Each outline is refilled from a nearby area with matching surroundings and colour-matched to them (a non-generative heal, best on sand, sky, grass, water or walls; it cannot invent hidden structure). Outline each object generously, including its shadow. Removals are listed in the Develop view and are undoable.",
            "inputSchema": { "type": "object", "required": ["photo_id", "objects"], "properties": {
                "photo_id": { "type": "string" },
                "objects": { "type": "array", "items": { "type": "object", "required": ["name", "points"], "properties": {
                    "name": { "type": "string", "description": "what is removed, e.g. \"person left\"" },
                    "points": outline } } } } }
        }
    ])
}

/// Appends a heal per `{name, points}` object, each filled from an automatically chosen area.
pub fn remove_objects(core: &Core, id: i64, d: &mut develop::Develop, objects: &Value) -> Result<Vec<String>, String> {
    let objs = objects
        .as_array()
        .ok_or("objects must be an array")?
        .iter()
        .map(|o| {
            let points: Vec<[f32; 2]> = serde_json::from_value(o["points"].clone()).map_err(|e| format!("invalid points: {e}"))?;
            develop::check_outline(&points)?;
            Ok((o["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("object").to_string(), points))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let img = core.proxy(&core.photo(id)?.path)?;
    let objs = develop::group_outlines(objs, img.w, img.h);
    let taken: Vec<Vec<[f32; 2]>> = d.retouch.iter().map(|r| r.points.clone()).chain(objs.iter().map(|o| o.1.clone())).collect();
    let mut names = vec![];
    for (name, points) in objs {
        let avoid: Vec<&[[f32; 2]]> = taken.iter().filter(|t| **t != points).map(|t| &t[..]).collect();
        let source = develop::heal_source(&img, &points, &avoid).ok_or_else(|| format!("no clear area near '{name}' to fill it from"))?;
        names.push(name.clone());
        d.retouch.push(develop::Heal { name, points, source });
    }
    Ok(names)
}

/// Handle one JSON-RPC message. None for notifications and client responses.
pub fn handle(core: &Arc<Core>, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg.get("method")?.as_str().unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(json!({}));
    let result: Result<Value, (i64, String)> = match method {
        "initialize" => Ok(json!({
            "protocolVersion": params["protocolVersion"].as_str().unwrap_or("2025-06-18"),
            "capabilities": { "tools": {}, "resources": {} },
            "serverInfo": { "name": "optixel-studio", "version": env!("CARGO_PKG_VERSION") },
            "instructions": INSTRUCTIONS,
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or("");
            Ok(call_tool(core, name, params.get("arguments").unwrap_or(&json!({}))))
        }
        "resources/list" => Ok(json!({ "resources": [
            { "uri": "optixel://catalog/summary", "name": "Catalog summary", "mimeType": "application/json" },
            { "uri": "optixel://selection/active", "name": "Active selection", "mimeType": "application/json" },
        ] })),
        "resources/templates/list" => Ok(json!({ "resourceTemplates": [
            { "uriTemplate": "optixel://photo/{id}/state", "name": "Edit graph", "mimeType": "application/json" },
            { "uriTemplate": "optixel://photo/{id}/telemetry", "name": "Histogram telemetry", "mimeType": "application/json" },
            { "uriTemplate": "optixel://photo/{id}/preview", "name": "Rendered preview (?max_edge=1024|2048&mask=<index>)", "mimeType": "image/jpeg" },
        ] })),
        "resources/read" => {
            let uri = params["uri"].as_str().unwrap_or("");
            read_resource(core, uri).map(|c| json!({ "contents": [c] })).map_err(|e| (-32002, e))
        }
        "prompts/list" => Ok(json!({ "prompts": [] })),
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    })
}

fn text(v: Value) -> Value {
    json!({ "type": "text", "text": serde_json::to_string_pretty(&v).unwrap_or_default() })
}

fn parse_id(v: &Value) -> Result<i64, String> {
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.trim_start_matches("ols:").parse().ok()))
        .ok_or_else(|| format!("invalid photo id {v}"))
}

fn parse_ids(v: &Value) -> Result<Vec<i64>, String> {
    v.as_array().ok_or("photo_ids must be an array")?.iter().map(parse_id).collect()
}

pub fn now(core: &Core) -> String {
    core.db().query_row("SELECT strftime('%Y-%m-%dT%H:%M:%SZ', 'now')", [], |r| r.get(0)).unwrap_or_default()
}

pub fn provenance(core: &Core, provider: &str, action: &str) -> Value {
    json!({ "timestamp": now(core), "provider": provider, "action": action, "generative_pixels_modified": false })
}

fn call_tool(core: &Arc<Core>, name: &str, a: &Value) -> Value {
    let brief: String = a.to_string().chars().take(160).collect();
    core.log("mcp", format!("{name} {brief}"));
    let r: Result<Vec<Value>, String> = (|| match name {
        "catalog_query" => {
            let mut notes = vec![];
            let semantic = a["semantic_query"].as_str().filter(|s| !s.trim().is_empty());
            if semantic.is_some() {
                notes.push("semantic_query matched by keyword (no vision embedding model in this build)");
            }
            let eyes_open_only = a["eyes_open_only"].as_bool().unwrap_or(false);
            let min_sharp = a["min_eye_sharpness"].as_f64();
            if min_sharp.is_some() {
                notes.push("min_eye_sharpness uses the main subject's sharpness from the auto analysis; photos not analysed yet (auto_run blur or cull) are left out");
            }
            let f = catalog::Filter {
                min_rating: a["min_rating"].as_i64(),
                flag: a["flag_status"].as_str().map(String::from),
                text: a["camera_or_lens"].as_str().map(String::from),
                text_any: semantic.map(String::from),
                limit: Some(a["limit"].as_i64().unwrap_or(50).clamp(1, 1000)),
                ..Default::default()
            };
            let db = core.db();
            let photos = catalog::query(&db, &f)?;
            let photos: Vec<Value> = photos
                .iter()
                .filter_map(|p| {
                    let ana = catalog::any_analysis(&db, &p.path);
                    let sharp = ana.as_ref().map(|m| m.subject_sharpness.round());
                    if min_sharp.is_some_and(|min| sharp.is_none_or(|s| (s as f64) < min)) {
                        return None;
                    }
                    if eyes_open_only && ana.as_ref().is_some_and(|m| m.blink_detected) {
                        return None;
                    }
                    Some(json!({ "photo_id": p.id.to_string(), "filename": p.filename, "folder": p.folder, "virtual_copy": p.copy_name,
                        "rating": p.rating, "flag": catalog::flag_name(p.flag), "color_label": p.color_label,
                        "captured_at": p.captured_at, "camera": p.camera, "lens": p.lens, "iso": p.iso, "aperture": p.aperture,
                        "shutter": p.shutter, "focal_length": p.focal_length, "width": p.width, "height": p.height, "edited": p.edited,
                        "subject_sharpness": sharp, "blink_detected": ana.as_ref().map(|m| m.blink_detected) }))
                })
                .collect();
            Ok(vec![text(json!({ "count": photos.len(), "photos": photos, "notes": notes }))])
        }
        "culling_batch_update" => {
            let ids = parse_ids(&a["photo_ids"])?;
            let copies = if a["create_virtual_copy"].as_bool().unwrap_or(false) {
                ids.iter().map(|id| core.virtual_copy(*id).map(|n| n.to_string())).collect::<Result<Vec<_>, _>>()?
            } else {
                vec![]
            };
            let u = catalog::Culling {
                rating: a["rating"].as_i64(),
                flag: a["flag"].as_str().map(String::from),
                color_label: a["color_label"].as_str().map(String::from),
                keywords: None,
            };
            let n = core.cull(&ids, &u)?;
            core.emit("catalog-changed", json!({}));
            Ok(vec![text(json!({ "updated": n, "virtual_copy_ids": copies }))])
        }
        "develop_apply_parameters" => {
            let ids = parse_ids(&a["photo_ids"])?;
            let params = a.get("parameters").filter(|v| v.is_object()).ok_or("parameters must be an object")?;
            let relative = a["mode"].as_str() == Some("relative_delta");
            let mut out = vec![];
            let mut ignored = vec![];
            for id in ids {
                let (mut d, unknown) = develop::apply_patch(&core.develop(id)?, params, relative)?;
                d.ai_provenance_log.push(provenance(core, "mcp", "develop_apply_parameters"));
                core.commit(id, &d, "AI: develop_apply_parameters")?;
                ignored = unknown;
                out.push(json!({ "photo_id": id.to_string(), "white_balance": d.white_balance, "tone": d.basic_tone, "presence": d.presence }));
            }
            Ok(vec![text(json!({ "applied": out, "ignored_keys": ignored }))])
        }
        "mask_create_and_edit" => {
            let id = parse_id(&a["photo_id"])?;
            let name = a["mask_name"].as_str().filter(|s| !s.is_empty()).ok_or("mask_name is required")?;
            let (mut mask, ignored) = develop::mask_from_ops(name, &a["operations"], a.get("local_adjustments").unwrap_or(&json!({})))?;
            let mut d = core.develop(id)?;
            match d.masks.iter_mut().find(|m| m.name == name) {
                Some(m) => {
                    mask.mask_id = m.mask_id.clone();
                    mask.tuning = m.tuning.take();
                    *m = mask.clone();
                }
                None => {
                    mask.mask_id = format!("mask_{:02}", d.masks.len() + 1);
                    d.masks.push(mask.clone());
                }
            }
            d.clamp();
            d.ai_provenance_log.push(provenance(core, "mcp", &format!("mask_create_and_edit '{name}'")));
            core.commit(id, &d, &format!("AI: mask '{name}'"))?;
            Ok(vec![text(json!({ "photo_id": id.to_string(), "mask_id": mask.mask_id, "ignored_keys": ignored }))])
        }
        "batch_export" => {
            let ids = parse_ids(&a["photo_ids"])?;
            let folder = a["destination_folder"].as_str().ok_or("destination_folder is required")?;
            let format = a["format"].as_str().unwrap_or("JPEG");
            let cs = a["color_space"].as_str().unwrap_or("sRGB");
            let q = a["quality"].as_u64().unwrap_or(90).clamp(1, 100) as u8;
            let files = core.export(&ids, std::path::Path::new(folder), format, cs, q, None)?;
            let mut notes = vec![];
            if a["attach_c2pa_credentials"].as_bool() == Some(true) {
                notes.push("C2PA signing is not available in this build; files were exported without Content Credentials");
            }
            Ok(vec![text(json!({ "files": files, "notes": notes }))])
        }
        "develop_auto" => {
            let profile = a["profile"].as_str().unwrap_or("natural");
            let strength = a["strength"].as_f64().map(|s| (s / 100.0) as f32).unwrap_or(1.0);
            let mut out = vec![];
            for id in parse_ids(&a["photo_ids"])? {
                let mut d = develop::auto_with_strength(&*core.proxy(&core.photo(id)?.path)?, &core.develop(id)?, profile, strength)?;
                d.ai_provenance_log.push(provenance(core, "mcp", &format!("develop_auto: {profile}")));
                core.commit(id, &d, &format!("AI: auto {profile}"))?;
                out.push(json!({ "photo_id": id.to_string(), "white_balance": d.white_balance, "tone": d.basic_tone, "presence": d.presence }));
            }
            Ok(vec![text(json!({ "applied": out, "profile": profile }))])
        }
        "auto_run" => {
            let function = a["function"].as_str().ok_or("function is required")?;
            let ids = parse_ids(&a["photo_ids"])?;
            let job = auto::Job::start(core, &format!("mcp-{}", crate::new_token()));
            let mut r = auto::run(core, function, &ids, a.get("settings").unwrap_or(&json!({})), &job)?;
            if let Some(rows) = r["rows"].as_array_mut() {
                rows.truncate(200);
            }
            Ok(vec![text(r)])
        }
        "ai_reconstruct_and_retouch" => {
            let id = parse_id(&a["photo_id"])?;
            let settings = match a["task"].as_str().unwrap_or("") {
                "ai_denoise" => json!({ "denoise": true, "strength": a["amount"].as_f64().unwrap_or(50.0), "super_resolution": false }),
                "super_resolution" => json!({ "denoise": false, "super_resolution": true, "scale": 2 }),
                t => return Err(format!("'{t}' needs generative models that are not installed in this build: ai_denoise and super_resolution are available, and retouch_remove_objects heals people and distractions")),
            };
            let job = auto::Job::start(core, &format!("mcp-{}", crate::new_token()));
            let r = auto::run(core, "enhance", &[id], &settings, &job)?;
            let row = &r["rows"][0];
            if let Some(e) = row["error"].as_str() {
                return Err(e.to_string());
            }
            Ok(vec![text(json!({ "photo_id": id.to_string(), "enhanced_photo_id": row["new_id"].to_string(), "file": row["output"], "size": row["size"] }))])
        }
        "retouch_remove_objects" => {
            let id = parse_id(&a["photo_id"])?;
            let mut d = core.develop(id)?;
            let names = remove_objects(core, id, &mut d, &a["objects"])?;
            d.clamp();
            d.ai_provenance_log.push(provenance(core, "mcp", &format!("retouch_remove_objects: {}", names.join(", "))));
            core.commit(id, &d, &format!("AI: remove {}", names.join(", ")))?;
            Ok(vec![text(json!({ "photo_id": id.to_string(), "removed": names, "retouch_count": d.retouch.len() }))])
        }
        "photo_inspect" => {
            let id = parse_id(&a["photo_id"])?;
            let edge = a["max_edge"].as_u64().unwrap_or(1024).clamp(256, 2048) as u32;
            let opts = Opts { full_frame: a["full_frame"].as_bool().unwrap_or(false), overlay_mask: a["show_mask"].as_u64().map(|m| m as usize), ..Default::default() };
            let mut img = core.render(id, &opts)?;
            let mut t = develop::telemetry(&img);
            if !a["include_histogram_bins"].as_bool().unwrap_or(false) {
                t.as_object_mut().map(|o| o.remove("histogram"));
            }
            if opts.overlay_mask.is_some() {
                t = json!("not measured: the mask overlay tints the preview");
            }
            if let Ok(at) = serde_json::from_value::<[f32; 2]>(a["mark"].clone()) {
                develop::mark(&mut img, at);
            }
            let jpeg = develop::encode_jpeg(&img, edge, 85)?;
            let state = core.develop(id)?;
            Ok(vec![
                text(json!({ "photo_id": id.to_string(), "telemetry": t, "state": state })),
                json!({ "type": "image", "mimeType": "image/jpeg", "data": base64::engine::general_purpose::STANDARD.encode(jpeg) }),
            ])
        }
        _ => Err(format!("unknown tool '{name}'")),
    })();
    match r {
        Ok(content) => json!({ "content": content, "isError": false }),
        Err(e) => {
            core.log("mcp", format!("{name} failed: {e}"));
            json!({ "content": [{ "type": "text", "text": e }], "isError": true })
        }
    }
}

fn read_resource(core: &Arc<Core>, uri: &str) -> Result<Value, String> {
    let rest = uri.strip_prefix("optixel://")
        .or_else(|| uri.strip_prefix("openlight://"))
        .ok_or_else(|| format!("unknown resource {uri}"))?;
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let q = |k: &str| query.split('&').find_map(|kv| kv.strip_prefix(&format!("{k}=")).and_then(|v| v.parse::<u32>().ok()));
    let json_doc = |v: Value| json!({ "uri": uri, "mimeType": "application/json", "text": serde_json::to_string_pretty(&v).unwrap_or_default() });
    match path.split('/').collect::<Vec<_>>().as_slice() {
        ["catalog", "summary"] => Ok(json_doc(catalog::summary(&core.db())?)),
        ["selection", "active"] => {
            let ids = core.selection.lock().unwrap().clone();
            let f = catalog::Filter { ids: Some(ids), ..Default::default() };
            Ok(json_doc(json!({ "photos": catalog::query(&core.db(), &f)? })))
        }
        ["photo", id, "state"] => Ok(json_doc(serde_json::to_value(core.develop(parse_id(&json!(id))?)?).unwrap_or_default())),
        ["photo", id, "telemetry"] => Ok(json_doc(develop::telemetry(&core.render(parse_id(&json!(id))?, &Opts::default())?))),
        ["photo", id, "preview"] => {
            let opts = Opts { overlay_mask: q("mask").map(|m| m as usize), ..Default::default() };
            let img = core.render(parse_id(&json!(id))?, &opts)?;
            let jpeg = develop::encode_jpeg(&img, q("max_edge").unwrap_or(1024).clamp(256, 2048), 85)?;
            Ok(json!({ "uri": uri, "mimeType": "image/jpeg", "blob": base64::engine::general_purpose::STANDARD.encode(jpeg) }))
        }
        _ => Err(format!("unknown resource {uri}")),
    }
}

// ---------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------

const MAX_BODY: u64 = 16 << 20;

fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn valid_host(h: &str) -> bool {
    let host = h.trim();
    let host = if host.starts_with('[') {
        host.split(']').next().map(|s| s.trim_start_matches('[')).unwrap_or(host)
    } else {
        host.split(':').next().unwrap_or(host)
    };
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn local_origin(o: &str) -> bool {
    if let Ok(u) = tauri::Url::parse(o) {
        match u.scheme() {
            "http" | "https" => match u.host_str() {
                Some("localhost") | Some("127.0.0.1") | Some("[::1]") | Some("::1") | Some("tauri.localhost") => true,
                _ => false,
            },
            "tauri" => u.host_str() == Some("localhost"),
            _ => false,
        }
    } else {
        false
    }
}

pub fn start_http(core: Arc<Core>) {
    let (enabled, port) = {
        let s = core.settings.lock().unwrap();
        (s.mcp_enabled, s.mcp_port)
    };
    if !enabled {
        *core.mcp_status.lock().unwrap() = "disabled".into();
        return;
    }
    std::thread::spawn(move || {
        let server = match tiny_http::Server::http(("127.0.0.1", port)) {
            Ok(s) => s,
            Err(e) => {
                *core.mcp_status.lock().unwrap() = format!("error: port {port}: {e}");
                return;
            }
        };
        *core.mcp_status.lock().unwrap() = "running".into();
        for req in server.incoming_requests() {
            let core = core.clone();
            std::thread::spawn(move || serve(&core, req));
        }
    });
}

fn serve(core: &Arc<Core>, mut req: tiny_http::Request) {
    let header = |name: &str| req.headers().iter().find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name)).map(|h| h.value.as_str().to_string());
    let (host, origin, auth) = (header("Host"), header("Origin"), header("Authorization"));
    let reply = |req: tiny_http::Request, code: u16, body: String| {
        let ct = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header");
        let nosniff = tiny_http::Header::from_bytes(&b"X-Content-Type-Options"[..], &b"nosniff"[..]).expect("static header");
        let _ = req.respond(
            tiny_http::Response::from_string(body)
                .with_status_code(code)
                .with_header(ct)
                .with_header(nosniff),
        );
    };
    if req.url().split('?').next() != Some("/mcp") {
        return reply(req, 404, json!({ "error": "not found" }).to_string());
    }
    if host.as_deref().is_none_or(|h| !valid_host(h)) {
        return reply(req, 403, json!({ "error": "forbidden host" }).to_string());
    }
    if origin.is_some_and(|o| !local_origin(&o)) {
        return reply(req, 403, json!({ "error": "forbidden origin" }).to_string());
    }
    let token = core.settings.lock().unwrap().mcp_token.clone();
    if !auth.and_then(|a| a.strip_prefix("Bearer ").map(|t| ct_eq(t.trim(), &token))).unwrap_or(false) {
        return reply(req, 401, json!({ "error": "missing or invalid bearer token (Settings > Agent & MCP)" }).to_string());
    }
    if *req.method() != tiny_http::Method::Post {
        return reply(req, 405, String::new());
    }
    let mut body = String::new();
    if req.as_reader().take(MAX_BODY).read_to_string(&mut body).is_err() {
        return reply(req, 400, json!({ "error": "unreadable body" }).to_string());
    }
    let out = match serde_json::from_str::<Value>(&body) {
        Ok(Value::Array(batch)) => {
            let r: Vec<Value> = batch.iter().filter_map(|m| handle(core, m)).collect();
            (!r.is_empty()).then(|| Value::Array(r))
        }
        Ok(msg) => handle(core, &msg),
        Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } })),
    };
    match out {
        Some(v) => reply(req, 200, v.to_string()),
        None => {
            let _ = req.respond(tiny_http::Response::empty(202));
        }
    }
}

pub fn run_stdio() {
    let dir = crate::data_dir();
    let settings = Settings::load(&dir);
    let url = format!("http://127.0.0.1:{}/mcp", settings.mcp_port);
    let agent: ureq::Agent =
        ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(900))).build().into();
    let post = |body: &str| -> Option<(u16, String)> {
        let mut r = agent
            .post(&url)
            .header("Authorization", &format!("Bearer {}", settings.mcp_token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .send(body)
            .ok()?;
        Some((r.status().as_u16(), r.body_mut().read_to_string().unwrap_or_default()))
    };
    let attached = settings.mcp_enabled && post(r#"{"jsonrpc":"2.0","id":0,"method":"ping"}"#).is_some_and(|(s, _)| s == 200);
    let core = (!attached).then(|| Arc::new(Core::open(dir).expect("could not open the Optixel catalog")));
    eprintln!("optixel-studio MCP (stdio): {}", if attached { "attached to the running app" } else { "headless catalog" });
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let _ = writeln!(out, "{}", json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } }));
                continue;
            }
        };
        let reply = match &core {
            Some(core) => handle(core, &msg).map(|v| v.to_string()),
            None => match post(&line) {
                Some((200, body)) => Some(body),
                Some((202, _)) => None,
                other => msg.get("id").map(|id| {
                    let why = other.map(|(s, b)| format!("app returned HTTP {s}: {b}")).unwrap_or("app is not reachable".into());
                    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32000, "message": why } }).to_string()
                }),
            },
        };
        if let Some(r) = reply {
            let _ = writeln!(out, "{r}");
            let _ = out.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_header_validation() {
        assert!(valid_host("127.0.0.1"));
        assert!(valid_host("127.0.0.1:41920"));
        assert!(valid_host("localhost"));
        assert!(valid_host("localhost:41920"));
        assert!(valid_host("[::1]"));
        assert!(valid_host("[::1]:41920"));

        assert!(!valid_host("evil.com"));
        assert!(!valid_host("evil.com:41920"));
        assert!(!valid_host("localhost.evil.com"));
        assert!(!valid_host("127.0.0.1.nip.io"));
        assert!(!valid_host("192.168.1.1"));
        assert!(!valid_host("0.0.0.0"));
    }

    #[test]
    fn origin_header_validation() {
        assert!(local_origin("http://localhost"));
        assert!(local_origin("http://localhost:1420"));
        assert!(local_origin("http://127.0.0.1:41920"));
        assert!(local_origin("tauri://localhost"));
        assert!(local_origin("http://tauri.localhost"));

        assert!(!local_origin("http://localhost.evil.com"));
        assert!(!local_origin("http://localhost-evil.com"));
        assert!(!local_origin("http://127.0.0.1.nip.io"));
        assert!(!local_origin("https://attacker.com"));
        assert!(!local_origin("javascript:void(0)"));
    }
}
