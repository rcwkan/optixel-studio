//! DAM catalog (spec §3.1): SQLite `.olcat`, folder import, EXIF, thumbnails,
//! history (undo/redo), virtual copies and `.olxmp` sidecars.

use crate::{
    Core,
    develop::{self, Develop, es},
    vision::Metrics,
};
use image::{DynamicImage, metadata::Orientation};
use rayon::prelude::*;
use rusqlite::{Connection, params, params_from_iter, types::Value as Sql};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    path::{MAIN_SEPARATOR, Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const SCHEMA: &str = "
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
CREATE TABLE IF NOT EXISTS photos (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  folder TEXT NOT NULL,
  filename TEXT NOT NULL,
  master_id INTEGER,
  copy_name TEXT,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  captured_at TEXT,
  camera TEXT,
  lens TEXT,
  iso INTEGER,
  aperture REAL,
  shutter TEXT,
  focal_length REAL,
  rating INTEGER NOT NULL DEFAULT 0,
  flag INTEGER NOT NULL DEFAULT 0,
  color_label TEXT NOT NULL DEFAULT '',
  keywords TEXT NOT NULL DEFAULT '',
  develop_json TEXT NOT NULL DEFAULT '{}',
  history_pos INTEGER,
  thumb_v INTEGER NOT NULL DEFAULT 0,
  imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  edited_at TEXT,
  deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS photos_path ON photos(path) WHERE master_id IS NULL;
CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  develop_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS history_photo ON history(photo_id, id);
CREATE TABLE IF NOT EXISTS immich_albums (
  album TEXT NOT NULL,
  path TEXT NOT NULL,
  PRIMARY KEY (album, path)
);
CREATE TABLE IF NOT EXISTS analysis (
  path TEXT PRIMARY KEY,
  stamp TEXT NOT NULL,
  metrics TEXT NOT NULL
);
";

pub fn open(path: &Path) -> Result<Connection, String> {
    let c = Connection::open(path).map_err(es)?;
    c.execute_batch(SCHEMA).map_err(es)?;
    c.execute_batch(crate::faces::SCHEMA).map_err(es)?;
    let _ = c.execute("ALTER TABLE photos ADD COLUMN deleted_at TEXT", []);
    let _ = c.execute("CREATE INDEX IF NOT EXISTS photos_deleted ON photos(deleted_at)", []);
    let ids: Vec<i64> = c
        .prepare("SELECT id FROM photos WHERE json_array_length(json_extract(develop_json, '$.masks')) > 0")
        .and_then(|mut st| st.query_map([], |r| r.get(0)).map(|rows| rows.flatten().collect()))
        .unwrap_or_default();
    for id in ids {
        let _ = get_develop(&c, id);
    }
    Ok(c)
}

#[derive(Serialize, Clone, Debug)]
pub struct Photo {
    pub id: i64,
    pub path: String,
    pub folder: String,
    pub filename: String,
    pub master_id: Option<i64>,
    pub copy_name: Option<String>,
    pub width: i64,
    pub height: i64,
    pub captured_at: Option<String>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<i64>,
    pub aperture: Option<f64>,
    pub shutter: Option<String>,
    pub focal_length: Option<f64>,
    pub rating: i64,
    /// 1 pick, 0 unflagged, -1 reject
    pub flag: i64,
    pub color_label: String,
    pub keywords: String,
    pub edited: bool,
    pub thumb_v: i64,
    pub deleted_at: Option<String>,
}

const COLS: &str = "id, path, folder, filename, master_id, copy_name, width, height, captured_at, camera, lens, \
                    iso, aperture, shutter, focal_length, rating, flag, color_label, keywords, edited_at IS NOT NULL, thumb_v, deleted_at";

fn row(r: &rusqlite::Row) -> rusqlite::Result<Photo> {
    Ok(Photo {
        id: r.get(0)?,
        path: r.get(1)?,
        folder: r.get(2)?,
        filename: r.get(3)?,
        master_id: r.get(4)?,
        copy_name: r.get(5)?,
        width: r.get(6)?,
        height: r.get(7)?,
        captured_at: r.get(8)?,
        camera: r.get(9)?,
        lens: r.get(10)?,
        iso: r.get(11)?,
        aperture: r.get(12)?,
        shutter: r.get(13)?,
        focal_length: r.get(14)?,
        rating: r.get(15)?,
        flag: r.get(16)?,
        color_label: r.get(17)?,
        keywords: r.get(18)?,
        edited: r.get(19)?,
        thumb_v: r.get(20)?,
        deleted_at: r.get(21)?,
    })
}

#[derive(Deserialize, Default, Debug)]
#[serde(default)]
pub struct Filter {
    /// Folder and its subfolders (only the folder itself with `flat`).
    pub folder: Option<String>,
    pub flat: bool,
    /// Every word must match filename, camera, lens or keywords.
    pub text: Option<String>,
    /// Any word may match (keyword fallback for semantic queries).
    pub text_any: Option<String>,
    pub min_rating: Option<i64>,
    /// pick | unflagged | reject
    pub flag: Option<String>,
    pub color_label: Option<String>,
    pub edited: Option<bool>,
    /// original | copy | edited | unedited_original | edited_original
    pub copy_type: Option<String>,
    /// landscape | portrait | square
    pub aspect: Option<String>,
    pub ids: Option<Vec<i64>>,
    /// Pages of an Immich album loaded so far (album id, or "favorites").
    pub immich_album: Option<String>,
    pub limit: Option<i64>,
    /// Query items in the bin (true) or normal items (false / None).
    pub in_bin: Option<bool>,
    /// Only photos showing every one of these people (faces.rs).
    pub people: Option<Vec<i64>>,
}

const TEXT_COLS: &str = "(filename LIKE ? OR camera LIKE ? OR lens LIKE ? OR keywords LIKE ? \
                         OR COALESCE(master_id, id) IN (SELECT f.photo_id FROM faces f JOIN people p ON p.id = f.person_id WHERE p.name LIKE ?))";
/// Placeholders in TEXT_COLS.
const TEXT_ARGS: usize = 5;

/// pick | unflagged | reject for a stored flag.
pub fn flag_name(f: i64) -> &'static str {
    ["reject", "unflagged", "pick"][(f + 1).clamp(0, 2) as usize]
}

pub fn flag_value(s: &str) -> Result<i64, String> {
    match s {
        "pick" => Ok(1),
        "unflagged" => Ok(0),
        "reject" => Ok(-1),
        _ => Err(format!("flag must be pick, unflagged or reject (got '{s}')")),
    }
}

pub fn query(c: &Connection, f: &Filter) -> Result<Vec<Photo>, String> {
    let mut sql = format!("SELECT {COLS} FROM photos WHERE 1=1");
    let mut args: Vec<Sql> = vec![];
    if let (Some(folder), true) = (&f.folder, f.flat) {
        sql += " AND folder = ?";
        args.push(folder.clone().into());
    } else if let Some(folder) = &f.folder {
        let prefix = format!("{folder}{MAIN_SEPARATOR}");
        sql += " AND (folder = ? OR substr(folder, 1, ?) = ?)";
        args.extend([folder.clone().into(), (prefix.chars().count() as i64).into(), prefix.into()]);
    }
    let words = |s: &Option<String>| -> Vec<String> {
        s.as_deref().unwrap_or("").split_whitespace().map(|w| format!("%{w}%")).collect()
    };
    for w in words(&f.text) {
        sql += &format!(" AND {TEXT_COLS}");
        args.extend(std::iter::repeat_n(Sql::from(w), TEXT_ARGS));
    }
    let any = words(&f.text_any);
    if !any.is_empty() {
        sql += &format!(" AND ({})", vec![TEXT_COLS; any.len()].join(" OR "));
        for w in any {
            args.extend(std::iter::repeat_n(Sql::from(w), TEXT_ARGS));
        }
    }
    if let Some(r) = f.min_rating {
        sql += " AND rating >= ?";
        args.push(r.into());
    }
    if let Some(flag) = f.flag.as_deref().filter(|s| !s.is_empty() && *s != "any") {
        sql += " AND flag = ?";
        args.push(flag_value(flag)?.into());
    }
    if let Some(label) = f.color_label.as_deref().filter(|s| !s.is_empty() && *s != "any") {
        sql += " AND color_label = ?";
        args.push(if label == "none" { String::new() } else { label.to_string() }.into());
    }
    if let Some(e) = f.edited {
        sql += " AND (edited_at IS NOT NULL) = ?";
        args.push(e.into());
    }
    if let Some(ct) = f.copy_type.as_deref().filter(|s| !s.is_empty() && *s != "all") {
        match ct {
            "original" => sql += " AND master_id IS NULL",
            "copy" | "saved_copy" | "saved_new" => sql += " AND master_id IS NOT NULL",
            "edited" => sql += " AND (edited_at IS NOT NULL)",
            "unedited_original" => sql += " AND master_id IS NULL AND edited_at IS NULL",
            "edited_original" => sql += " AND master_id IS NULL AND edited_at IS NOT NULL",
            _ => {}
        }
    }
    if let Some(aspect) = f.aspect.as_deref().filter(|s| !s.is_empty() && *s != "all") {
        match aspect {
            "landscape" => sql += " AND width > height AND height > 0",
            "portrait" => sql += " AND height > width AND width > 0",
            "square" => sql += " AND width = height AND width > 0",
            _ => {}
        }
    }
    if let Some(people) = f.people.as_deref().filter(|p| !p.is_empty()) {
        sql += &crate::faces::filter_sql(people);
    }
    if let Some(album) = &f.immich_album {
        sql += " AND path IN (SELECT path FROM immich_albums WHERE album = ?)";
        args.push(album.clone().into());
    }
    if let Some(ids) = &f.ids {
        if ids.is_empty() {
            return Ok(vec![]); // e.g. an empty Immich album; `IN ()` is not valid SQL
        }
        // inline: a whole catalog's results can pass SQLite's limit of 32766 parameters
        sql += &format!(" AND id IN ({})", ids.iter().map(i64::to_string).collect::<Vec<_>>().join(","));
    }
    match f.in_bin {
        Some(true) => sql += " AND deleted_at IS NOT NULL",
        Some(false) => sql += " AND deleted_at IS NULL",
        None => {
            if f.ids.is_none() {
                sql += " AND deleted_at IS NULL";
            }
        }
    }
    sql += " ORDER BY COALESCE(captured_at, ''), filename, id LIMIT ?";
    args.push(f.limit.unwrap_or(1_000_000).into());
    let mut st = c.prepare(&sql).map_err(es)?;
    let rows = st.query_map(params_from_iter(args), row).map_err(es)?;
    rows.collect::<Result<_, _>>().map_err(es)
}

/// The photos with these ids, any number of them (SQLite caps the variables of one query).
pub fn photos(c: &Connection, ids: &[i64]) -> Result<Vec<Photo>, String> {
    let mut out = vec![];
    for chunk in ids.chunks(10_000) {
        out.extend(query(c, &Filter { ids: Some(chunk.to_vec()), ..Default::default() })?);
    }
    Ok(out)
}

/// Local folders only; Immich rows are browsed through their albums.
pub fn folders(c: &Connection) -> Result<Vec<(String, i64)>, String> {
    let mut st = c
        .prepare("SELECT folder, COUNT(*) FROM photos WHERE path NOT LIKE 'immich://%' AND deleted_at IS NULL GROUP BY folder ORDER BY folder")
        .map_err(es)?;
    let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).map_err(es)?;
    rows.collect::<Result<_, _>>().map_err(es)
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Culling {
    pub rating: Option<i64>,
    pub flag: Option<String>,
    pub color_label: Option<String>,
    pub keywords: Option<String>,
}

pub const COLOR_LABELS: &[&str] = &["none", "red", "yellow", "green", "blue", "purple"];

pub fn update(c: &Connection, ids: &[i64], u: &Culling) -> Result<usize, String> {
    let mut sets: Vec<(&str, Sql)> = vec![];
    if let Some(r) = u.rating {
        if !(0..=5).contains(&r) {
            return Err("rating must be 0-5".into());
        }
        sets.push(("rating", r.into()));
    }
    if let Some(f) = &u.flag {
        sets.push(("flag", flag_value(f)?.into()));
    }
    if let Some(l) = &u.color_label {
        if !COLOR_LABELS.contains(&l.as_str()) {
            return Err(format!("color_label must be one of {}", COLOR_LABELS.join(", ")));
        }
        sets.push(("color_label", if l == "none" { String::new() } else { l.clone() }.into()));
    }
    if let Some(k) = &u.keywords {
        sets.push(("keywords", k.trim().to_string().into()));
    }
    if sets.is_empty() || ids.is_empty() {
        return Ok(0);
    }
    let sql = format!(
        "UPDATE photos SET {} WHERE id IN ({})",
        sets.iter().map(|(k, _)| format!("{k} = ?")).collect::<Vec<_>>().join(", "),
        vec!["?"; ids.len()].join(",")
    );
    let args = sets.into_iter().map(|(_, v)| v).chain(ids.iter().map(|i| Sql::from(*i)));
    c.execute(&sql, params_from_iter(args)).map_err(es)
}

/// `add` merged into a comma-separated keyword list (case-insensitive; the existing order is kept).
pub fn merge_keywords(cur: &str, add: &[String]) -> String {
    let mut out: Vec<String> = cur.split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect();
    for k in add.iter().map(|k| k.trim()).filter(|k| !k.is_empty()) {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(k)) {
            out.push(k.to_string());
        }
    }
    out.join(", ")
}

/// Removes photos and their virtual copies; returns every id that went.
pub fn remove(c: &Connection, ids: &[i64]) -> Result<Vec<i64>, String> {
    let mut gone = vec![];
    for ids in ids.chunks(10_000) {
        // two variables per id; SQLite allows 32766
        let list = vec!["?"; ids.len()].join(",");
        let args: Vec<Sql> = ids.iter().chain(ids).map(|i| Sql::from(*i)).collect();
        let mut st = c
            .prepare(&format!("DELETE FROM photos WHERE id IN ({list}) OR master_id IN ({list}) RETURNING id"))
            .map_err(es)?;
        let rows = st.query_map(params_from_iter(args), |r| r.get(0)).map_err(es)?;
        gone.extend(rows.collect::<Result<Vec<i64>, _>>().map_err(es)?);
    }
    Ok(gone)
}

/// Returns the count of photos currently in the bin.
pub fn bin_count(c: &Connection) -> Result<i64, String> {
    c.query_row("SELECT COUNT(*) FROM photos WHERE deleted_at IS NOT NULL", [], |r| r.get(0)).map_err(es)
}

/// Moves photos and their virtual copies to the bin (soft delete); returns affected ids.
pub fn move_to_bin(c: &Connection, ids: &[i64]) -> Result<Vec<i64>, String> {
    let mut affected = vec![];
    for chunk in ids.chunks(10_000) {
        let list = vec!["?"; chunk.len()].join(",");
        let args: Vec<Sql> = chunk.iter().chain(chunk).map(|i| Sql::from(*i)).collect();
        let mut st = c
            .prepare(&format!(
                "UPDATE photos SET deleted_at = CURRENT_TIMESTAMP WHERE (id IN ({list}) OR master_id IN ({list})) AND deleted_at IS NULL RETURNING id"
            ))
            .map_err(es)?;
        let rows = st.query_map(params_from_iter(args), |r| r.get(0)).map_err(es)?;
        affected.extend(rows.collect::<Result<Vec<i64>, _>>().map_err(es)?);
    }
    Ok(affected)
}

/// Restores photos and their virtual copies from the bin; returns restored ids.
pub fn restore_from_bin(c: &Connection, ids: &[i64]) -> Result<Vec<i64>, String> {
    let mut affected = vec![];
    for chunk in ids.chunks(10_000) {
        let list = vec!["?"; chunk.len()].join(",");
        let args: Vec<Sql> = chunk.iter().chain(chunk).map(|i| Sql::from(*i)).collect();
        let mut st = c
            .prepare(&format!(
                "UPDATE photos SET deleted_at = NULL WHERE (id IN ({list}) OR master_id IN ({list})) AND deleted_at IS NOT NULL RETURNING id"
            ))
            .map_err(es)?;
        let rows = st.query_map(params_from_iter(args), |r| r.get(0)).map_err(es)?;
        affected.extend(rows.collect::<Result<Vec<i64>, _>>().map_err(es)?);
    }
    Ok(affected)
}

// ---------------------------------------------------------------------------
// Edit state + history
// ---------------------------------------------------------------------------

pub fn get_develop(c: &Connection, id: i64) -> Result<Develop, String> {
    let s: String = c
        .query_row("SELECT develop_json FROM photos WHERE id = ?", [id], |r| r.get(0))
        .map_err(|_| format!("photo {id} not found"))?;
    let raw: Develop = serde_json::from_str(&s).unwrap_or_default();
    let mut d = raw.clone();
    d.clamp();
    if d.masks != raw.masks
        && let Ok(json) = serde_json::to_string(&d)
    {
        let _ = c.execute("UPDATE photos SET develop_json = ? WHERE id = ?", params![json, id]);
    }
    Ok(d)
}

fn set_state(c: &Connection, id: i64, json: &str, hid: i64, edited: bool) -> Result<(), String> {
    c.execute(
        "UPDATE photos SET develop_json = ?, history_pos = ?, edited_at = CASE WHEN ? THEN datetime('now') END WHERE id = ?",
        params![json, hid, edited, id],
    )
    .map_err(es)?;
    Ok(())
}

/// New history step; drops any redo branch.
pub fn save_develop(c: &Connection, id: i64, d: &Develop, label: &str) -> Result<(), String> {
    save_develop_ext(c, id, d, label, None)
}

/// Save develop state to history, optionally replacing/updating the current step in-place
/// if its label matches `replace_prefix` (e.g. continuous slider or profile strength adjustments).
pub fn save_develop_ext(
    c: &Connection,
    id: i64,
    d: &Develop,
    label: &str,
    replace_prefix: Option<&str>,
) -> Result<(), String> {
    let json = serde_json::to_string(d).map_err(es)?;
    let pos: Option<i64> = c
        .query_row("SELECT history_pos FROM photos WHERE id = ?", [id], |r| r.get(0))
        .map_err(|_| format!("photo {id} not found"))?;

    if let (Some(cur_hid), Some(prefix)) = (pos, replace_prefix) {
        if !prefix.is_empty() {
            let cur_label: Option<String> = c
                .query_row("SELECT label FROM history WHERE id = ? AND photo_id = ?", params![cur_hid, id], |r| r.get(0))
                .ok();
            if let Some(cur) = cur_label {
                const PROTECTED: &[&str] = &["Import", "Virtual copy", "Save as new photo", "Duplicate"];
                if !PROTECTED.contains(&cur.as_str()) {
                    let matches = cur == prefix
                        || cur == label
                        || cur.starts_with(prefix)
                        || (prefix.starts_with("Profile: ")
                            && (cur == format!("Auto: {}", &prefix["Profile: ".len()..])
                                || cur.starts_with(&format!("Auto: {} (", &prefix["Profile: ".len()..]))));
                    if matches {
                        c.execute("DELETE FROM history WHERE photo_id = ? AND id > ?", params![id, cur_hid]).map_err(es)?;
                        c.execute(
                            "UPDATE history SET label = ?, develop_json = ?, created_at = datetime('now') WHERE id = ?",
                            params![label, json, cur_hid],
                        )
                        .map_err(es)?;
                        return set_state(c, id, &json, cur_hid, *d != Develop::default());
                    }
                }
            }
        }
    }

    if let Some(pos) = pos {
        c.execute("DELETE FROM history WHERE photo_id = ? AND id > ?", params![id, pos]).map_err(es)?;
    }
    c.execute("INSERT INTO history (photo_id, label, develop_json) VALUES (?, ?, ?)", params![id, label, json])
        .map_err(es)?;
    set_state(c, id, &json, c.last_insert_rowid(), *d != Develop::default())
}

pub fn history(c: &Connection, id: i64) -> Result<Vec<Value>, String> {
    let pos: Option<i64> =
        c.query_row("SELECT history_pos FROM photos WHERE id = ?", [id], |r| r.get(0)).map_err(es)?;
    let mut st =
        c.prepare("SELECT id, label, created_at FROM history WHERE photo_id = ? ORDER BY id DESC").map_err(es)?;
    let rows = st
        .query_map([id], |r| {
            let hid: i64 = r.get(0)?;
            Ok(json!({"id": hid, "label": r.get::<_, String>(1)?, "created_at": r.get::<_, String>(2)?, "current": Some(hid) == pos}))
        })
        .map_err(es)?;
    rows.collect::<Result<_, _>>().map_err(es)
}

pub fn goto(c: &Connection, id: i64, hid: i64) -> Result<Develop, String> {
    let json: String = c
        .query_row("SELECT develop_json FROM history WHERE id = ? AND photo_id = ?", [hid, id], |r| r.get(0))
        .map_err(|_| "history step not found".to_string())?;
    let mut d: Develop = serde_json::from_str(&json).unwrap_or_default();
    d.clamp();
    let json = serde_json::to_string(&d).map_err(es)?;
    set_state(c, id, &json, hid, d != Develop::default())?;
    Ok(d)
}

/// Undo (dir < 0) or redo (dir > 0). None when there is nothing to step to.
pub fn step(c: &Connection, id: i64, dir: i32) -> Result<Option<Develop>, String> {
    let pos: Option<i64> =
        c.query_row("SELECT history_pos FROM photos WHERE id = ?", [id], |r| r.get(0)).map_err(es)?;
    let sql = if dir < 0 {
        "SELECT MAX(id) FROM history WHERE photo_id = ? AND id < ?"
    } else {
        "SELECT MIN(id) FROM history WHERE photo_id = ? AND id > ?"
    };
    let target: Option<i64> = c.query_row(sql, [id, pos.unwrap_or(0)], |r| r.get(0)).map_err(es)?;
    target.map(|t| goto(c, id, t)).transpose()
}

/// Delete a specific history step. If deleting the current step, navigates to the adjacent step and returns its Develop state.
pub fn delete_history_step(c: &Connection, id: i64, hid: i64) -> Result<Option<Develop>, String> {
    let pos: Option<i64> =
        c.query_row("SELECT history_pos FROM photos WHERE id = ?", [id], |r| r.get(0)).map_err(es)?;
    let count: i64 =
        c.query_row("SELECT COUNT(*) FROM history WHERE photo_id = ?", [id], |r| r.get(0)).map_err(es)?;
    if count <= 1 {
        return Err("Cannot delete the only history step".into());
    }

    let mut new_dev = None;
    if pos == Some(hid) {
        let prev: Option<i64> = c
            .query_row("SELECT MAX(id) FROM history WHERE photo_id = ? AND id < ?", [id, hid], |r| r.get(0))
            .map_err(es)?;
        let target = match prev {
            Some(p) => p,
            None => c
                .query_row("SELECT MIN(id) FROM history WHERE photo_id = ? AND id > ?", [id, hid], |r| r.get(0))
                .map_err(es)?,
        };
        let d = goto(c, id, target)?;
        new_dev = Some(d);
    }

    c.execute("DELETE FROM history WHERE id = ? AND photo_id = ?", params![hid, id]).map_err(es)?;
    Ok(new_dev)
}

/// Undo steps: every history step but each photo's current one.
const PAST_STEPS: &str = "FROM history WHERE id NOT IN (SELECT history_pos FROM photos WHERE history_pos IS NOT NULL)";

/// (steps, bytes) that `clear_history` would drop.
pub fn past_steps(c: &Connection) -> Result<(i64, i64), String> {
    c.query_row(&format!("SELECT COUNT(*), COALESCE(SUM(LENGTH(develop_json)), 0) {PAST_STEPS}"), [], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(es)
}

/// Drops the undo steps; every photo keeps its current edit.
pub fn clear_history(c: &Connection) -> Result<usize, String> {
    c.execute(&format!("DELETE {PAST_STEPS}"), []).map_err(es)
}

/// Rewrites the catalog without its free pages (and analyses of photos no longer in it) and empties the
/// write-ahead log.
pub fn compact(c: &Connection) -> Result<(), String> {
    c.execute_batch("DELETE FROM analysis WHERE path NOT IN (SELECT path FROM photos); VACUUM; PRAGMA wal_checkpoint(TRUNCATE);").map_err(es)
}

/// What the auto functions measured on a file, if it was measured with these settings (`stamp`).
pub fn analysis(c: &Connection, path: &str, stamp: &str) -> Option<Metrics> {
    let json: String = c.query_row("SELECT metrics FROM analysis WHERE path = ? AND stamp = ?", [path, stamp], |r| r.get(0)).ok()?;
    serde_json::from_str(&json).ok()
}

/// The last measurement of a file, whatever its settings.
pub fn any_analysis(c: &Connection, path: &str) -> Option<Metrics> {
    let json: String = c.query_row("SELECT metrics FROM analysis WHERE path = ?", [path], |r| r.get(0)).ok()?;
    serde_json::from_str(&json).ok()
}

pub fn save_analysis(c: &Connection, path: &str, stamp: &str, m: &Metrics) -> Result<(), String> {
    let json = serde_json::to_string(m).map_err(es)?;
    c.execute("INSERT OR REPLACE INTO analysis (path, stamp, metrics) VALUES (?, ?, ?)", params![path, stamp, json]).map_err(es)?;
    Ok(())
}

pub fn virtual_copy(c: &Connection, id: i64) -> Result<i64, String> {
    let master: i64 =
        c.query_row("SELECT COALESCE(master_id, id) FROM photos WHERE id = ?", [id], |r| r.get(0)).map_err(es)?;
    let n: i64 =
        c.query_row("SELECT COUNT(*) FROM photos WHERE master_id = ?", [master], |r| r.get(0)).map_err(es)?;
    c.execute(
        "INSERT INTO photos (path, folder, filename, master_id, copy_name, width, height, captured_at, camera, lens, iso,
           aperture, shutter, focal_length, rating, flag, color_label, keywords, develop_json, edited_at)
         SELECT path, folder, filename, ?, ?, width, height, captured_at, camera, lens, iso,
           aperture, shutter, focal_length, rating, flag, color_label, keywords, develop_json, edited_at
         FROM photos WHERE id = ?",
        params![master, format!("Copy {}", n + 1), id],
    )
    .map_err(es)?;
    let new = c.last_insert_rowid();
    let d = get_develop(c, new)?;
    save_develop(c, new, &d, "Virtual copy")?;
    Ok(new)
}

pub fn save_as_new_photo(c: &Connection, id: i64, develop: Option<&Develop>, copy_name: Option<&str>) -> Result<i64, String> {
    let master: i64 =
        c.query_row("SELECT COALESCE(master_id, id) FROM photos WHERE id = ?", [id], |r| r.get(0)).map_err(es)?;
    let n: i64 =
        c.query_row("SELECT COUNT(*) FROM photos WHERE master_id = ?", [master], |r| r.get(0)).map_err(es)?;
    let name = match copy_name.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None => format!("Copy {}", n + 1),
    };
    c.execute(
        "INSERT INTO photos (path, folder, filename, master_id, copy_name, width, height, captured_at, camera, lens, iso,
           aperture, shutter, focal_length, rating, flag, color_label, keywords, develop_json, edited_at)
         SELECT path, folder, filename, ?, ?, width, height, captured_at, camera, lens, iso,
           aperture, shutter, focal_length, rating, flag, color_label, keywords, develop_json, CURRENT_TIMESTAMP
         FROM photos WHERE id = ?",
        params![master, name, id],
    )
    .map_err(es)?;
    let new = c.last_insert_rowid();
    if let Some(d) = develop {
        save_develop(c, new, d, "Save as new photo")?;
    } else {
        let d = get_develop(c, new)?;
        save_develop(c, new, &d, "Save as new photo")?;
    }
    Ok(new)
}

pub fn summary(c: &Connection) -> Result<Value, String> {
    let one = |sql: &str| -> Result<i64, String> { c.query_row(sql, [], |r| r.get(0)).map_err(es) };
    let top = |col: &str| -> Result<Vec<Value>, String> {
        let mut st = c
            .prepare(&format!(
                "SELECT {col}, COUNT(*) n FROM photos WHERE {col} IS NOT NULL AND {col} != '' GROUP BY {col} ORDER BY n DESC LIMIT 20"
            ))
            .map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"name": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)
    };

    let ratings: Vec<Value> = {
        let mut st = c.prepare("SELECT rating, COUNT(*) FROM photos GROUP BY rating ORDER BY rating").map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"rating": r.get::<_, i64>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let flags: Vec<Value> = {
        let mut st = c.prepare("SELECT flag, COUNT(*) FROM photos GROUP BY flag ORDER BY flag").map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"flag": r.get::<_, i64>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let years: Vec<Value> = {
        let mut st = c.prepare("SELECT SUBSTR(captured_at, 1, 4) y, COUNT(*) n FROM photos WHERE captured_at IS NOT NULL AND length(captured_at) >= 4 GROUP BY y ORDER BY y DESC LIMIT 12").map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"year": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let iso_ranges: Vec<Value> = {
        let sql = "SELECT 
          CASE 
            WHEN iso < 200 THEN '< 200'
            WHEN iso < 800 THEN '200–800'
            WHEN iso < 3200 THEN '800–3200'
            WHEN iso < 12800 THEN '3200–12800'
            ELSE '12800+'
          END AS range,
          COUNT(*) n
        FROM photos WHERE iso IS NOT NULL AND iso > 0
        GROUP BY range ORDER BY MIN(iso)";
        let mut st = c.prepare(sql).map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"range": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let aperture_ranges: Vec<Value> = {
        let sql = "SELECT 
          CASE 
            WHEN aperture <= 1.4 THEN '≤ f/1.4'
            WHEN aperture <= 2.8 THEN 'f/1.8–f/2.8'
            WHEN aperture <= 5.6 THEN 'f/4–f/5.6'
            WHEN aperture <= 11.0 THEN 'f/8–f/11'
            ELSE '≥ f/16'
          END AS range,
          COUNT(*) n
        FROM photos WHERE aperture IS NOT NULL AND aperture > 0
        GROUP BY range ORDER BY MIN(aperture)";
        let mut st = c.prepare(sql).map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"range": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let focal_ranges: Vec<Value> = {
        let sql = "SELECT 
          CASE 
            WHEN focal_length < 24 THEN 'Ultra-wide (<24mm)'
            WHEN focal_length < 50 THEN 'Wide (24–50mm)'
            WHEN focal_length < 85 THEN 'Standard (50–85mm)'
            WHEN focal_length < 200 THEN 'Telephoto (85–200mm)'
            ELSE 'Super-tele (200mm+)'
          END AS range,
          COUNT(*) n
        FROM photos WHERE focal_length IS NOT NULL AND focal_length > 0
        GROUP BY range ORDER BY MIN(focal_length)";
        let mut st = c.prepare(sql).map_err(es)?;
        let rows = st.query_map([], |r| Ok(json!({"range": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?})));
        rows.map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };

    let (formats, raw_count, std_count) = {
        let mut st = c.prepare("SELECT filename FROM photos").map_err(es)?;
        let mut map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut rows = st.query([]).map_err(es)?;
        let mut raw = 0usize;
        let mut std = 0usize;
        while let Some(row) = rows.next().map_err(es)? {
            let f: String = row.get(0).map_err(es)?;
            let ext = std::path::Path::new(&f)
                .extension()
                .map(|s| s.to_string_lossy().to_uppercase())
                .unwrap_or_else(|| "OTHER".into());
            let is_raw = matches!(ext.as_str(), "NEF" | "CR2" | "CR3" | "ARW" | "DNG" | "RAF" | "RW2" | "ORF" | "PEF");
            if is_raw {
                raw += 1;
            } else {
                std += 1;
            }
            *map.entry(ext).or_default() += 1;
        }
        let mut list: Vec<(String, usize)> = map.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1));
        let fmt_values: Vec<Value> = list.into_iter().take(8).map(|(k, v)| json!({"ext": k, "count": v})).collect();
        (fmt_values, raw, std)
    };

    let history_stats = {
        let total_steps: i64 = one("SELECT COUNT(*) FROM history")?;
        let max_steps: i64 = c.query_row("SELECT COALESCE(MAX(n), 0) FROM (SELECT COUNT(*) n FROM history GROUP BY photo_id)", [], |r| r.get(0)).unwrap_or(0);
        let edited_photos: i64 = one("SELECT COUNT(*) FROM photos WHERE edited_at IS NOT NULL")?;
        let avg_steps = if edited_photos > 0 {
            (total_steps as f64 / edited_photos as f64 * 10.0).round() / 10.0
        } else {
            0.0
        };
        json!({
            "total_steps": total_steps,
            "max_steps": max_steps,
            "avg_steps": avg_steps,
        })
    };

    Ok(json!({
        "total_photos": one("SELECT COUNT(*) FROM photos")?,
        "virtual_copies": one("SELECT COUNT(*) FROM photos WHERE master_id IS NOT NULL")?,
        "edited": one("SELECT COUNT(*) FROM photos WHERE edited_at IS NOT NULL")?,
        "picks": one("SELECT COUNT(*) FROM photos WHERE flag = 1")?,
        "rejects": one("SELECT COUNT(*) FROM photos WHERE flag = -1")?,
        "rated_3_plus": one("SELECT COUNT(*) FROM photos WHERE rating >= 3")?,
        "folders": folders(c)?.into_iter().map(|(f, n)| json!({"folder": f, "count": n})).collect::<Vec<_>>(),
        "cameras": top("camera")?,
        "lenses": top("lens")?,
        "ratings": ratings,
        "flags": flags,
        "years": years,
        "iso_ranges": iso_ranges,
        "aperture_ranges": aperture_ranges,
        "focal_ranges": focal_ranges,
        "formats": formats,
        "raw_count": raw_count,
        "std_count": std_count,
        "history_stats": history_stats,
    }))
}

// ---------------------------------------------------------------------------
// Sidecars (spec §6): `<file>.olxmp` JSON next to the original
// ---------------------------------------------------------------------------

pub fn sidecar_path(path: &str) -> PathBuf {
    PathBuf::from(format!("{path}.olxmp"))
}

/// The `.olxmp` document of a photo's edits.
pub fn sidecar_json(p: &Photo, d: &Develop) -> String {
    let doc = json!({"schema_version": "1.0.0", "photo_id": format!("ols:{}", p.id), "source": p.filename, "pipeline": d});
    serde_json::to_string_pretty(&doc).unwrap_or_default()
}

pub fn read_sidecar(path: &Path) -> Option<Develop> {
    read_sidecar_with_id(path).map(|(_, d)| d)
}

pub fn read_sidecar_with_id(path: &Path) -> Option<(Option<i64>, Develop)> {
    let s = fs::read_to_string(sidecar_path(&path.to_string_lossy())).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;
    let d: Develop = serde_json::from_value(v.get("pipeline")?.clone()).ok()?;
    let id: Option<i64> = v.get("photo_id")
        .and_then(|p| p.as_str())
        .and_then(|p| p.strip_prefix("ols:"))
        .and_then(|p| p.parse().ok());
    Some((id, d))
}

/// Masters only (virtual copies live in the catalog, as in Lightroom); Immich photos keep
/// their edits in the catalog. Best effort: read-only volumes just skip the sidecar.
/// ponytail: Optixel JSON only; read/write Adobe `crs:` XMP when ACR round-trips matter.
pub fn write_sidecar(p: &Photo, d: &Develop) {
    if p.master_id.is_some() || crate::remote::immich_id(&p.path).is_some() {
        return;
    }
    if let Err(e) = fs::write(sidecar_path(&p.path), sidecar_json(p, d)) {
        eprintln!("sidecar {}: {e}", p.path);
    }
    if crate::xmp::candidate_xmp_paths(Path::new(&p.path)).iter().any(|x| x.is_file()) {
        let _ = crate::xmp::write_adobe_xmp(p, d, None, None);
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Meta {
    pub captured_at: Option<String>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<i64>,
    pub aperture: Option<f64>,
    pub shutter: Option<String>,
    pub focal_length: Option<f64>,
    pub orientation: u8,
}

/// New master row with its first history step; None when `path` is already in the catalog.
pub fn add_photo(c: &Connection, path: &str, folder: &str, filename: &str, m: &Meta, d: &Develop) -> Result<Option<i64>, String> {
    add_photo_with_id(c, None, path, folder, filename, m, d)
}

/// New master row with an optionally specified id; None when `path` is already in the catalog.
pub fn add_photo_with_id(
    c: &Connection,
    specified_id: Option<i64>,
    path: &str,
    folder: &str,
    filename: &str,
    m: &Meta,
    d: &Develop,
) -> Result<Option<i64>, String> {
    let json = serde_json::to_string(d).map_err(es)?;
    let (n, id) = if let Some(req_id) = specified_id.filter(|&id| {
        // Only use specified_id if no row with this ID already exists
        !c.query_row("SELECT 1 FROM photos WHERE id = ?", [id], |_| Ok(true)).unwrap_or(false)
    }) {
        let n = c
            .execute(
                "INSERT OR IGNORE INTO photos (id, path, folder, filename, captured_at, camera, lens, iso, aperture, shutter,
                   focal_length, develop_json, edited_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ? THEN datetime('now') END)",
                params![
                    req_id,
                    path,
                    folder,
                    filename,
                    m.captured_at,
                    m.camera,
                    m.lens,
                    m.iso,
                    m.aperture,
                    m.shutter,
                    m.focal_length,
                    json,
                    *d != Develop::default(),
                ],
            )
            .map_err(es)?;
        (n, req_id)
    } else {
        let n = c
            .execute(
                "INSERT OR IGNORE INTO photos (path, folder, filename, captured_at, camera, lens, iso, aperture, shutter,
                   focal_length, develop_json, edited_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ? THEN datetime('now') END)",
                params![
                    path,
                    folder,
                    filename,
                    m.captured_at,
                    m.camera,
                    m.lens,
                    m.iso,
                    m.aperture,
                    m.shutter,
                    m.focal_length,
                    json,
                    *d != Develop::default(),
                ],
            )
            .map_err(es)?;
        (n, c.last_insert_rowid())
    };
    if n == 0 {
        return Ok(None);
    }
    c.execute("INSERT INTO history (photo_id, label, develop_json) VALUES (?, 'Import', ?)", params![id, json]).map_err(es)?;
    c.execute("UPDATE photos SET history_pos = ? WHERE id = ?", params![c.last_insert_rowid(), id]).map_err(es)?;
    Ok(Some(id))
}

/// "Make Model" without repeating the brand ("Canon" + "Canon EOS R5" → "Canon EOS R5").
pub fn camera_name(make: Option<String>, model: Option<String>) -> Option<String> {
    match (make, model) {
        (Some(mk), Some(md)) => {
            let brand = mk.split_whitespace().next().unwrap_or("").to_lowercase();
            Some(if md.to_lowercase().starts_with(&brand) { md } else { format!("{mk} {md}") })
        }
        (mk, md) => md.or(mk),
    }
}

/// EXIF orientation of a file (1 = upright).
pub fn exif_orientation(path: &Path) -> u8 {
    read_exif(path).orientation
}

fn read_exif(path: &Path) -> Meta {
    use exif::{In, Tag, Value as EV};
    let mut m = Meta { orientation: 1, ..Default::default() };
    let Ok(file) = fs::File::open(path) else { return m };
    let Ok(ex) = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)) else { return m };
    let field = |t: Tag| ex.get_field(t, In::PRIMARY);
    let s = |t: Tag| {
        field(t).map(|f| f.display_value().to_string().trim_matches('"').trim().to_string()).filter(|s| !s.is_empty())
    };
    let f = |t: Tag| {
        field(t).and_then(|f| match &f.value {
            EV::Rational(v) if !v.is_empty() && v[0].denom != 0 => Some(v[0].to_f64()),
            _ => None,
        })
    };
    m.captured_at = s(Tag::DateTimeOriginal).or_else(|| s(Tag::DateTime));
    m.camera = camera_name(s(Tag::Make), s(Tag::Model));
    m.lens = s(Tag::LensModel);
    m.iso = field(Tag::PhotographicSensitivity).and_then(|f| f.value.get_uint(0)).map(i64::from);
    m.aperture = f(Tag::FNumber);
    m.shutter = s(Tag::ExposureTime);
    m.focal_length = f(Tag::FocalLength);
    m.orientation = field(Tag::Orientation).and_then(|f| f.value.get_uint(0)).unwrap_or(1) as u8;
    m
}

pub const THUMB: u32 = 400;

/// Thumbnail plus the (oriented) pixel size of the decoded source. For RAW files the thumbnail
/// comes from the embedded preview, the size from the RAW header (the preview is smaller).
fn make_thumb(path: &Path, orientation: u8) -> Result<(DynamicImage, u32, u32), String> {
    let img = if develop::is_raw(path) {
        match develop::raw_preview(path) {
            Some(mut img) => {
                if let Some(o) = Orientation::from_exif(orientation) {
                    img.apply_orientation(o);
                }
                img
            }
            None => to_dynamic(&develop::load(path, None)?),
        }
    } else {
        develop::displayable(develop::decode_std(path)?)
    };
    let (w, h) = if develop::is_raw(path) { develop::raw_size(path).unwrap_or((img.width(), img.height())) } else { (img.width(), img.height()) };
    Ok((img.thumbnail(THUMB, THUMB), w, h))
}

pub fn to_dynamic(img: &develop::Img) -> DynamicImage {
    DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(img.w as u32, img.h as u32, develop::to_rgba8(img)).expect("rgba buffer size"),
    )
}

pub fn write_jpeg(path: &Path, img: &DynamicImage) -> Result<(), String> {
    let file = std::io::BufWriter::new(fs::File::create(path).map_err(es)?);
    img.to_rgb8().write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(file, 82)).map_err(es)
}

fn scan(dir: &Path, out: &mut Vec<PathBuf>, deep: bool) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if p.is_dir() {
            if deep {
                scan(&p, out, deep);
            }
        } else if develop::is_supported(&p) {
            out.push(p);
        }
    }
}

/// Classic catalog import: files stay where they are. Returns the new photos' ids.
pub fn import(core: &Core, folder: &Path, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<Vec<i64>, String> {
    let mut files = vec![];
    scan(folder, &mut files, true);
    import_files(core, files, progress)
}

/// Like `import`, but only the photos directly in `folder`: browsing opens one folder level at a time.
pub fn import_shallow(core: &Core, folder: &Path, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<Vec<i64>, String> {
    let mut files = vec![];
    scan(folder, &mut files, false);
    import_files(core, files, progress)
}

/// What opening `folder` would show: its subfolders and how many photos are directly in it.
pub fn browse(folder: &Path) -> Result<(Vec<String>, usize), String> {
    let (mut dirs, mut images) = (vec![], 0);
    for e in fs::read_dir(folder).map_err(es)?.flatten() {
        let p = e.path();
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if p.is_dir() {
            dirs.push(p.to_string_lossy().into_owned());
        } else if develop::is_supported(&p) {
            images += 1;
        }
    }
    dirs.sort_by_key(|d| d.to_lowercase());
    Ok((dirs, images))
}

/// Import these files (those already in the catalog are skipped); returns the new photos' ids.
pub fn import_files(core: &Core, mut files: Vec<PathBuf>, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<Vec<i64>, String> {
    let existing: HashSet<String> = {
        let c = core.db();
        let mut st = c.prepare("SELECT path FROM photos WHERE master_id IS NULL").map_err(es)?;
        let rows = st.query_map([], |r| r.get(0)).map_err(es)?;
        rows.collect::<Result<_, _>>().map_err(es)?
    };
    files.retain(|p| !existing.contains(&*p.to_string_lossy()));
    files.sort();
    // Photos are added a few at a time, each batch complete (row, thumbnail, size) before the next
    // starts, so the grid fills and can be worked in while the rest is still being read.
    // ponytail: 16 threads because reading a network share is waiting, not computing; make it a setting if a NAS needs fewer
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().map_err(es)?;
    let total = files.len();
    let done = AtomicUsize::new(0);
    let mut ids = vec![];
    let mut rest = &files[..];
    let mut size = 8;
    while !rest.is_empty() {
        let (batch, tail) = rest.split_at(size.min(rest.len()));
        rest = tail;
        size = 32;
        let metas: Vec<(PathBuf, Meta, Option<i64>, Develop, Option<Culling>)> = pool.install(|| {
            batch
                .par_iter()
                .map(|p| {
                    let m = read_exif(p);
                    let ax = crate::xmp::read_adobe_xmp(p, None);
                    let (sid, d) = read_sidecar_with_id(p)
                        .or_else(|| read_sidecar(p).map(|d| (None, d)))
                        .unwrap_or_else(|| (None, ax.as_ref().map(|x| x.develop.clone()).unwrap_or_default()));
                    let cull = ax.as_ref().and_then(crate::xmp::to_culling);
                    (p.clone(), m, sid, d, cull)
                })
                .collect()
        });
        let mut added: Vec<(i64, PathBuf, u8)> = vec![];
        let mut culls: Vec<(i64, Culling)> = vec![];
        {
            let mut c = core.db();
            let tx = c.transaction().map_err(es)?;
            for (p, m, sid, d, cull) in &metas {
                let folder = p.parent().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                let name = p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                if let Some(id) = add_photo_with_id(&tx, *sid, &p.to_string_lossy(), &folder, &name, m, d)? {
                    added.push((id, p.clone(), m.orientation));
                    if let Some(cu) = cull {
                        culls.push((id, cu.clone()));
                    }
                }
            }
            tx.commit().map_err(es)?;
            for (id, cu) in culls {
                let _ = update(&c, &[id], &cu);
            }
        }
        let dims: Vec<(i64, u32, u32)> = pool.install(|| {
            added
                .par_iter()
                .filter_map(|(id, p, o)| {
                    let thumb_p = core.thumb_path(*id);
                    // Always generate fresh thumbnail for newly added photo to avoid inheriting stale/mismatched thumbnails
                    let r = make_thumb(p, *o).and_then(|(t, w, h)| write_jpeg(&thumb_p, &t).map(|_| (w, h)));
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                    match r {
                        Ok((w, h)) => Some((*id, w, h)),
                        Err(e) => {
                            eprintln!("thumbnail {}: {e}", p.display());
                            None
                        }
                    }
                })
                .collect()
        });
        {
            let c = core.db();
            for (id, w, h) in dims {
                c.execute("UPDATE photos SET width = ?, height = ? WHERE id = ?", params![w, h, id]).map_err(es)?;
            }
        }
        ids.extend(added.into_iter().map(|a| a.0));
        core.emit("catalog-changed", serde_json::json!({}));
    }
    Ok(ids)
}

/// Regenerate thumbnails for specified photo IDs, a specific folder, or all catalog photos.
/// Re-extracts or re-renders previews from source files and increments `thumb_v` to bust cache.
pub fn regenerate_thumbnails(
    core: &Core,
    ids: Option<Vec<i64>>,
    folder: Option<String>,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<usize, String> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().map_err(es)?;
    let mut sql = "SELECT id, path, develop_json, edited_at, master_id FROM photos WHERE deleted_at IS NULL".to_string();
    let mut args: Vec<Sql> = vec![];

    if let Some(folder_path) = folder {
        sql += " AND folder = ?";
        args.push(folder_path.into());
    } else if let Some(target_ids) = ids {
        if target_ids.is_empty() {
            return Ok(0);
        }
        sql += &format!(" AND id IN ({})", target_ids.iter().map(i64::to_string).collect::<Vec<_>>().join(","));
    }

    struct TargetPhoto {
        id: i64,
        path: String,
        develop_json: String,
        is_edited: bool,
    }

    let photos: Vec<TargetPhoto> = {
        let c = core.db();
        let mut st = c.prepare(&sql).map_err(es)?;
        st.query_map(params_from_iter(args), |r| {
            let is_edited = r.get::<_, Option<String>>(3)?.is_some() || r.get::<_, Option<i64>>(4)?.is_some();
            Ok(TargetPhoto {
                id: r.get(0)?,
                path: r.get(1)?,
                develop_json: r.get(2)?,
                is_edited,
            })
        })
        .map_err(es)?
        .collect::<Result<_, _>>()
        .map_err(es)?
    };

    let total = photos.len();
    if total == 0 {
        return Ok(0);
    }
    progress(0, total);

    let done = AtomicUsize::new(0);
    let mut updated: Vec<(i64, u32, u32)> = vec![];

    for chunk in photos.chunks(32) {
        let res: Vec<(i64, u32, u32)> = pool.install(|| {
            chunk
                .par_iter()
                .filter_map(|ph| {
                    let p = Path::new(&ph.path);
                    if !p.is_file() {
                        return None;
                    }
                    let orientation = exif_orientation(p);
                    let thumb_p = core.thumb_path(ph.id);
                    let res = if ph.is_edited {
                        let dev = serde_json::from_str::<Develop>(&ph.develop_json).ok();
                        if let Some(d) = dev.filter(|d| *d != Develop::default()) {
                            if let Ok(proxy) = core.proxy(&ph.path) {
                                let out = develop::render(&*proxy, &d, &develop::Opts {
                                    ai_cache_dir: Some(core.ai_cache_dir(ph.id)),
                                    ..Default::default()
                                });
                                let thumb = to_dynamic(&out).thumbnail(THUMB, THUMB);
                                write_jpeg(&thumb_p, &thumb).map(|_| (proxy.w as u32, proxy.h as u32))
                            } else {
                                make_thumb(p, orientation).and_then(|(t, w, h)| write_jpeg(&thumb_p, &t).map(|_| (w, h)))
                            }
                        } else {
                            make_thumb(p, orientation).and_then(|(t, w, h)| write_jpeg(&thumb_p, &t).map(|_| (w, h)))
                        }
                    } else {
                        make_thumb(p, orientation).and_then(|(t, w, h)| write_jpeg(&thumb_p, &t).map(|_| (w, h)))
                    };

                    let cur = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(cur, total);

                    match res {
                        Ok((w, h)) => Some((ph.id, w, h)),
                        Err(e) => {
                            eprintln!("regenerate thumb {}: {e}", ph.path);
                            None
                        }
                    }
                })
                .collect()
        });

        if !res.is_empty() {
            let mut c = core.db();
            let tx = c.transaction().map_err(es)?;
            for (id, w, h) in &res {
                let _ = tx.execute(
                    "UPDATE photos SET thumb_v = thumb_v + 1, width = ?, height = ? WHERE id = ?",
                    params![w, h, id],
                );
            }
            tx.commit().map_err(es)?;
            for (id, _, _) in &res {
                core.emit("thumb-changed", json!({"id": id}));
            }
            updated.extend(res);
        }
    }

    progress(total, total);
    core.emit("catalog-changed", json!({}));
    Ok(updated.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regenerate_thumbnails() {
        let dir = std::env::temp_dir().join(format!("ols-test-regen-{}", crate::new_token()));
        let core = Core::open(dir.clone()).unwrap();
        let photos_dir = dir.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let photo_path = photos_dir.join("test.jpg");
        let img = DynamicImage::new_rgb8(100, 100);
        img.save(&photo_path).unwrap();

        let id = add_photo(&core.db(), &photo_path.to_string_lossy(), &photos_dir.to_string_lossy(), "test.jpg", &Default::default(), &Default::default()).unwrap().unwrap();
        assert_eq!(core.thumb_path(id).exists(), false);

        let count = regenerate_thumbnails(&core, Some(vec![id]), None, &|_, _| {}).unwrap();
        assert_eq!(count, 1);
        assert!(core.thumb_path(id).is_file());

        let p = core.photo(id).unwrap();
        assert_eq!(p.thumb_v, 1);
    }

    #[test]
    fn test_filter_and_save_as_new_photo() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();

        // Insert original unedited photo
        c.execute(
            "INSERT INTO photos (id, path, folder, filename, width, height) VALUES (1, '/photos/a.jpg', '/photos', 'a.jpg', 6000, 4000)",
            [],
        )
        .unwrap();

        // Insert original edited photo
        c.execute(
            "INSERT INTO photos (id, path, folder, filename, width, height, edited_at) VALUES (2, '/photos/b.jpg', '/photos', 'b.jpg', 4000, 6000, CURRENT_TIMESTAMP)",
            [],
        )
        .unwrap();

        // 1. Check all photos
        let all = query(&c, &Filter::default()).unwrap();
        assert_eq!(all.len(), 2);

        // 2. Filter original photos
        let orig = query(&c, &Filter { copy_type: Some("original".into()), ..Default::default() }).unwrap();
        assert_eq!(orig.len(), 2);

        // 3. Filter edited photos
        let edited = query(&c, &Filter { copy_type: Some("edited".into()), ..Default::default() }).unwrap();
        assert_eq!(edited.len(), 1);
        assert_eq!(edited[0].id, 2);

        // 4. Filter unedited original photos
        let unedited = query(&c, &Filter { copy_type: Some("unedited_original".into()), ..Default::default() }).unwrap();
        assert_eq!(unedited.len(), 1);
        assert_eq!(unedited[0].id, 1);

        // 5. Save as new photo from photo 1 with custom develop and custom name
        let mut d = Develop::default();
        d.basic_tone.exposure = 1.5;
        let new_id = save_as_new_photo(&c, 1, Some(&d), Some("B&W High Contrast")).unwrap();
        assert_eq!(new_id, 3);

        // Verify the saved new edited photo
        let p3 = query(&c, &Filter { ids: Some(vec![3]), ..Default::default() }).unwrap();
        assert_eq!(p3.len(), 1);
        assert_eq!(p3[0].master_id, Some(1));
        assert_eq!(p3[0].copy_name.as_deref(), Some("B&W High Contrast"));
        assert!(p3[0].edited);

        // 6. Filter saved new edited photos (copy)
        let copies = query(&c, &Filter { copy_type: Some("copy".into()), ..Default::default() }).unwrap();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].id, 3);

        // 7. Check aspect filter
        let landscape = query(&c, &Filter { aspect: Some("landscape".into()), ..Default::default() }).unwrap();
        assert_eq!(landscape.len(), 2); // id 1 (6000x4000) and id 3 (copied from 1)
        let portrait = query(&c, &Filter { aspect: Some("portrait".into()), ..Default::default() }).unwrap();
        assert_eq!(portrait.len(), 1); // id 2 (4000x6000)

        // 8. Test bin functions: move to bin, bin_count, query in_bin, restore_from_bin
        assert_eq!(bin_count(&c).unwrap(), 0);
        let moved = move_to_bin(&c, &[1]).unwrap();
        // Photo 1 and its virtual copy Photo 3 should both be moved to bin
        assert!(moved.contains(&1));
        assert!(moved.contains(&3));
        assert_eq!(bin_count(&c).unwrap(), 2);

        // Regular query now only returns photo 2
        let regular = query(&c, &Filter::default()).unwrap();
        assert_eq!(regular.len(), 1);
        assert_eq!(regular[0].id, 2);

        // Query bin explicitly
        let in_bin = query(&c, &Filter { in_bin: Some(true), ..Default::default() }).unwrap();
        assert_eq!(in_bin.len(), 2);

        // Restore photo 1 from bin
        let restored = restore_from_bin(&c, &[1]).unwrap();
        assert!(restored.contains(&1));
        assert!(restored.contains(&3));
        assert_eq!(bin_count(&c).unwrap(), 0);

        let regular_after = query(&c, &Filter::default()).unwrap();
        assert_eq!(regular_after.len(), 3);
    }

    #[test]
    fn test_migration_from_legacy_catalog_without_deleted_at() {
        let dir = std::env::temp_dir().join(format!("ols-migration-test-{}", crate::new_token()));
        fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("legacy.olcat");

        // Create a database with the old schema (without deleted_at column and without photos_deleted index)
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE photos (
                  id INTEGER PRIMARY KEY,
                  path TEXT NOT NULL,
                  folder TEXT NOT NULL,
                  filename TEXT NOT NULL,
                  master_id INTEGER,
                  copy_name TEXT,
                  width INTEGER NOT NULL DEFAULT 0,
                  height INTEGER NOT NULL DEFAULT 0,
                  captured_at TEXT,
                  camera TEXT,
                  lens TEXT,
                  iso INTEGER,
                  aperture REAL,
                  shutter TEXT,
                  focal_length REAL,
                  rating INTEGER NOT NULL DEFAULT 0,
                  flag INTEGER NOT NULL DEFAULT 0,
                  color_label TEXT NOT NULL DEFAULT '',
                  keywords TEXT NOT NULL DEFAULT '',
                  develop_json TEXT NOT NULL DEFAULT '{}',
                  history_pos INTEGER,
                  thumb_v INTEGER NOT NULL DEFAULT 0,
                  imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  edited_at TEXT
                );
                CREATE UNIQUE INDEX photos_path ON photos(path) WHERE master_id IS NULL;
                CREATE INDEX photos_folder ON photos(folder);
                INSERT INTO photos (path, folder, filename) VALUES ('/tmp/photo1.jpg', '/tmp', 'photo1.jpg');",
            )
            .unwrap();
        }

        // Now open the legacy database with catalog::open - this should successfully migrate it!
        let c = open(&db_path).expect("open should succeed and migrate the legacy database");

        // Verify that deleted_at column is now usable
        assert_eq!(bin_count(&c).unwrap(), 0);
        let list = query(&c, &Filter::default()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, 1);
        assert_eq!(list[0].deleted_at, None);

        // Verify that moving to bin works on the migrated catalog
        let moved = move_to_bin(&c, &[1]).unwrap();
        assert_eq!(moved, vec![1]);
        assert_eq!(bin_count(&c).unwrap(), 1);

        // Verify that restore works
        let restored = restore_from_bin(&c, &[1]).unwrap();
        assert_eq!(restored, vec![1]);
        assert_eq!(bin_count(&c).unwrap(), 0);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delete_history_step() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();

        c.execute(
            "INSERT INTO photos (id, path, folder, filename, width, height) VALUES (1, '/photos/a.jpg', '/photos', 'a.jpg', 6000, 4000)",
            [],
        )
        .unwrap();

        let mut d1 = Develop::default();
        save_develop(&c, 1, &d1, "Import").unwrap();

        // Only 1 step: deleting must fail
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 1);
        let hid1 = h[0]["id"].as_i64().unwrap();
        assert!(delete_history_step(&c, 1, hid1).is_err());

        // Add 2 more steps
        d1.basic_tone.exposure = 1.0;
        save_develop(&c, 1, &d1, "Exposure +1").unwrap();
        d1.basic_tone.contrast = 20.0;
        save_develop(&c, 1, &d1, "Contrast +20").unwrap();

        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 3);
        let hid3 = h[0]["id"].as_i64().unwrap(); // Contrast +20 (current)
        let hid2 = h[1]["id"].as_i64().unwrap(); // Exposure +1

        // Delete inactive step (hid2)
        let res = delete_history_step(&c, 1, hid2).unwrap();
        assert!(res.is_none()); // current step was unchanged
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0]["label"], "Contrast +20");
        assert!(h[0]["current"].as_bool().unwrap());

        // Delete active step (hid3): should move to previous step (hid1)
        let res = delete_history_step(&c, 1, hid3).unwrap();
        assert!(res.is_some());
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0]["label"], "Import");
        assert!(h[0]["current"].as_bool().unwrap());
    }

    #[test]
    fn test_save_develop_replace_profile_strength() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();

        c.execute(
            "INSERT INTO photos (id, path, folder, filename, width, height) VALUES (1, '/photos/a.jpg', '/photos', 'a.jpg', 6000, 4000)",
            [],
        )
        .unwrap();

        let mut d = Develop::default();
        save_develop(&c, 1, &d, "Import").unwrap();
        assert_eq!(history(&c, 1).unwrap().len(), 1);

        // Apply profile at 100%
        d.basic_tone.exposure = 0.2;
        save_develop_ext(&c, 1, &d, "Profile: Landscape (100%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0]["label"], "Profile: Landscape (100%)");

        // Adjust profile strength multiple times (e.g. 20%, 200%, 92%, 143%)
        d.basic_tone.exposure = 0.04;
        save_develop_ext(&c, 1, &d, "Profile: Landscape (20%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2, "Should keep single history entry when adjusting profile strength");
        assert_eq!(h[0]["label"], "Profile: Landscape (20%)");

        d.basic_tone.exposure = 0.3;
        save_develop_ext(&c, 1, &d, "Profile: Landscape (143%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2, "Should still be 2 entries");
        assert_eq!(h[0]["label"], "Profile: Landscape (143%)");

        // Now apply another tuning (e.g. Exposure +0.5)
        d.basic_tone.exposure = 0.8;
        save_develop(&c, 1, &d, "Exposure +0.5").unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 3, "New tuning creates new history entry");
        assert_eq!(h[0]["label"], "Exposure +0.5");

        // Adjusting profile strength now (after another tuning) creates a new history entry!
        d.basic_tone.exposure = 0.9;
        save_develop_ext(&c, 1, &d, "Profile: Landscape (120%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 4, "Profile adjustment after another tuning creates a new entry");
        assert_eq!(h[0]["label"], "Profile: Landscape (120%)");

        // And adjusting again updates this new 4th entry
        d.basic_tone.exposure = 0.85;
        save_develop_ext(&c, 1, &d, "Profile: Landscape (110%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(h[0]["label"], "Profile: Landscape (110%)");

        // Test replacement when previous step was legacy "Auto: Landscape"
        let d_auto = Develop::default();
        save_develop(&c, 1, &d_auto, "Auto: Landscape").unwrap();
        let h = history(&c, 1).unwrap();
        let len_before = h.len();
        save_develop_ext(&c, 1, &d_auto, "Profile: Landscape (143%)", Some("Profile: Landscape")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), len_before, "Should replace Auto: Landscape");
        assert_eq!(h[0]["label"], "Profile: Landscape (143%)");
    }

    #[test]
    fn test_save_develop_consecutive_tuning_coalesces() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();

        c.execute(
            "INSERT INTO photos (id, path, folder, filename, width, height) VALUES (1, '/photos/a.jpg', '/photos', 'a.jpg', 6000, 4000)",
            [],
        )
        .unwrap();

        let mut d = Develop::default();
        save_develop(&c, 1, &d, "Import").unwrap();
        assert_eq!(history(&c, 1).unwrap().len(), 1);

        // Adjusting "Natural" first time creates 1 history entry
        d.hdr.natural = 50.0;
        save_develop_ext(&c, 1, &d, "Natural", Some("Natural")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0]["label"], "Natural");

        // Repeatedly adjusting "Natural" coalesces in-place into the same history entry
        d.hdr.natural = 70.0;
        save_develop_ext(&c, 1, &d, "Natural", Some("Natural")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2, "Second Natural tweak must coalesce in-place");

        d.hdr.natural = 85.0;
        save_develop_ext(&c, 1, &d, "Natural", Some("Natural")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 2, "Third Natural tweak coalesces in-place");
        assert_eq!(h[0]["label"], "Natural");

        // Now change another tuning: "Contrast"
        d.hdr.contrast_weight = 30.0;
        save_develop_ext(&c, 1, &d, "Contrast", Some("Contrast")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 3, "Changing another tuning must create a new history step");
        assert_eq!(h[0]["label"], "Contrast");

        // Repeatedly adjusting "Contrast" coalesces into this Contrast step
        d.hdr.contrast_weight = 60.0;
        save_develop_ext(&c, 1, &d, "Contrast", Some("Contrast")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 3, "Repeated Contrast tweak coalesces");
        assert_eq!(h[0]["label"], "Contrast");

        // Adjusting "Natural" again after Contrast creates a new history entry
        d.hdr.natural = 95.0;
        save_develop_ext(&c, 1, &d, "Natural", Some("Natural")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 4, "Tuning Natural again creates a new history step");
        assert_eq!(h[0]["label"], "Natural");

        // Tweaking it again coalesces into that step
        d.hdr.natural = 100.0;
        save_develop_ext(&c, 1, &d, "Natural", Some("Natural")).unwrap();
        let h = history(&c, 1).unwrap();
        assert_eq!(h.len(), 4);

        // Discrete actions like "Add mask" do NOT coalesce
        save_develop(&c, 1, &d, "Add mask").unwrap();
        assert_eq!(history(&c, 1).unwrap().len(), 5);

        save_develop(&c, 1, &d, "Add mask").unwrap();
        assert_eq!(history(&c, 1).unwrap().len(), 6, "Adding a second mask must not overwrite the first");
    }
}
