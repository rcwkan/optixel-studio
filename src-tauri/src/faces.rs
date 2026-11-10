//! People: YOLOFace finds the faces in each photo, ArcFace turns each into a 512-d identity vector, and faces
//! whose vectors are close are grouped into people. The library can then be searched by person (or by name).
//!
//! ponytail: online nearest-centroid clustering, O(faces × people) per scan; fine for tens of thousands of faces.
//! People that split or merge wrongly are fixed by hand (merge, move faces, similar faces); a full re-cluster
//! (HDBSCAN) if that gets tedious at scale.

use crate::{Core, auto::Job, catalog, develop::es};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};
use std::{fs, path::PathBuf, sync::{Arc, Mutex}};

pub const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS people (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  hidden INTEGER NOT NULL DEFAULT 0,
  cover_face INTEGER
);
CREATE TABLE IF NOT EXISTS faces (
  id INTEGER PRIMARY KEY,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  score REAL NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS face_scans (
  photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
  faces INTEGER NOT NULL
);
";

/// Cosine similarity to a person's mean vector above which a face is that person.
/// ponytail: one global threshold; raise it if different people get merged, lower it if one person splits.
pub const MATCH: f32 = 0.42;
/// Faces smaller than this (pixels on the scan image) are too blurry to recognise.
const MIN_SIDE: f32 = 28.0;
const SCAN_EDGE: u32 = 1600;
const CROP: u32 = 160;

/// One scan at a time: two would find the same unscanned photos.
static SCANNING: Mutex<()> = Mutex::new(());

fn blob(e: &[f32]) -> Vec<u8> {
    e.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn crops_dir(core: &Core) -> PathBuf {
    core.dir().join("thumbs").join("faces")
}

/// Running sums of each person's face vectors: the direction of the sum is the person's centroid.
struct Centroids(Vec<(i64, Vec<f32>)>);

impl Centroids {
    fn load(c: &Connection) -> Result<Self, String> {
        let mut out: Vec<(i64, Vec<f32>)> = vec![];
        let mut st = c.prepare("SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL ORDER BY person_id").map_err(es)?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, vector(&r.get::<_, Vec<u8>>(1)?)))).map_err(es)?;
        for (p, e) in rows.flatten() {
            match out.last_mut() {
                Some((id, sum)) if *id == p => sum.iter_mut().zip(&e).for_each(|(s, v)| *s += v),
                _ => out.push((p, e)),
            }
        }
        Ok(Centroids(out))
    }

    /// The closest person and the similarity to their centroid.
    fn best(&self, e: &[f32]) -> Option<(i64, f32)> {
        self.0
            .iter()
            .map(|(id, sum)| (*id, dot(sum, e) / dot(sum, sum).sqrt().max(1e-6)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn add(&mut self, person: i64, e: &[f32]) {
        match self.0.iter_mut().find(|(id, _)| *id == person) {
            Some((_, sum)) => sum.iter_mut().zip(e).for_each(|(s, v)| *s += v),
            None => self.0.push((person, e.to_vec())),
        }
    }
}

/// Find and group the faces of the photos not scanned yet (all of them, or only `ids`).
pub fn scan(core: &Core, ids: Option<&[i64]>, job: &Job) -> Result<Value, String> {
    let dir = crate::aimodel::faces_dir(&core.dir())?;
    let todo: Vec<(i64, String)> = {
        let c = core.db();
        let mut st = c
            .prepare(
                "SELECT id, path FROM photos WHERE master_id IS NULL AND deleted_at IS NULL AND path NOT LIKE 'immich://%' \
                 AND id NOT IN (SELECT photo_id FROM face_scans) ORDER BY id",
            )
            .map_err(es)?;
        let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).map_err(es)?;
        rows.flatten().filter(|(id, _)| ids.is_none_or(|ids| ids.contains(id))).collect()
    };
    if todo.is_empty() {
        return Ok(json!({"photos": 0, "faces": 0, "failed": 0}));
    }
    let open = |f: &str| crate::aimodel::open_session(&dir.join(f), None, 0, false);
    let (mut det, mut rec) = (open(crate::faceswap::DETECT)?, open(crate::faceswap::RECOGNISE)?);
    let mut centroids = Centroids::load(&core.db())?;
    let crops = crops_dir(core);
    fs::create_dir_all(&crops).map_err(es)?;
    let (mut found, mut failed, mut done) = (0usize, 0usize, 0usize);
    for (i, (id, path)) in todo.iter().enumerate() {
        if job.cancelled() {
            break;
        }
        job.progress(&format!("Finding faces: photo {} of {}", i + 1, todo.len()), i, todo.len());
        // ponytail: sequential, one decode at a time (core.decode serialises them anyway); RAW decode dominates
        let img = match core.decode(std::path::Path::new(path), Some(SCAN_EDGE)) {
            Ok(img) if img.w > 0 && img.h > 0 => img,
            _ => {
                failed += 1; // offline or unreadable: not marked, so a later scan retries it
                continue;
            }
        };
        let faces: Vec<_> = crate::faceswap::detect(&mut det, &img)?
            .into_iter()
            .filter(|f| (f.bbox[2] - f.bbox[0]).min(f.bbox[3] - f.bbox[1]) >= MIN_SIDE)
            .collect();
        let pic = (!faces.is_empty()).then(|| catalog::to_dynamic(&img));
        let mut c = core.db();
        let tx = c.transaction().map_err(es)?;
        for f in &faces {
            let e = crate::faceswap::embed(&mut rec, &img, f)?;
            let person = match centroids.best(&e) {
                Some((p, s)) if s >= MATCH => p,
                _ => {
                    tx.execute("INSERT INTO people DEFAULT VALUES", []).map_err(es)?;
                    tx.last_insert_rowid()
                }
            };
            centroids.add(person, &e);
            let (w, h) = (img.w as f32, img.h as f32);
            let [x0, y0, x1, y1] = [f.bbox[0].max(0.0), f.bbox[1].max(0.0), f.bbox[2].min(w), f.bbox[3].min(h)];
            tx.execute(
                "INSERT INTO faces (photo_id, person_id, x, y, w, h, score, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                params![id, person, x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, f.score, blob(&e)],
            )
            .map_err(es)?;
            let face_id = tx.last_insert_rowid();
            if let Some(pic) = &pic {
                // a square around the face with some hair and chin
                let side = ((x1 - x0).max(y1 - y0) * 1.5).min(w.min(h));
                let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
                let sx = (cx - side / 2.0).clamp(0.0, w - side);
                let sy = (cy - side / 2.0).clamp(0.0, h - side);
                let crop = pic.crop_imm(sx as u32, sy as u32, side as u32, side as u32).thumbnail(CROP, CROP);
                let _ = catalog::write_jpeg(&crops.join(format!("{face_id}.jpg")), &crop);
            }
        }
        tx.execute("INSERT OR REPLACE INTO face_scans (photo_id, faces) VALUES (?, ?)", params![id, faces.len() as i64]).map_err(es)?;
        tx.commit().map_err(es)?;
        drop(c);
        found += faces.len();
        done += 1;
        if done % 25 == 0 {
            core.emit("faces-changed", json!({}));
        }
    }
    core.emit("faces-changed", json!({}));
    Ok(json!({"photos": done, "faces": found, "failed": failed}))
}

/// After an import: scan the new photos in the background when People is on and its model is installed.
pub fn after_import(core: &Arc<Core>, ids: &[i64]) {
    if ids.is_empty() || !core.settings.lock().unwrap().faces_enabled || crate::aimodel::faces_dir(&core.dir()).is_err() {
        return;
    }
    let (core, ids) = (core.clone(), ids.to_vec());
    std::thread::spawn(move || {
        let _one = SCANNING.lock().unwrap_or_else(|e| e.into_inner());
        let job = Job::start(&core, "faces-scan");
        if let Err(e) = scan(&core, Some(&ids), &job) {
            core.emit("remote-error", json!(format!("People: {e}")));
        }
    });
}

/// People with no faces left (after a merge, a move, or their photos were deleted) go.
fn prune(c: &Connection) -> Result<(), String> {
    c.execute("DELETE FROM people WHERE id NOT IN (SELECT person_id FROM faces WHERE person_id IS NOT NULL)", []).map_err(es)?;
    Ok(())
}

fn list(ids: &[i64]) -> String {
    if ids.is_empty() {
        return "NULL".into(); // `IN ()` is not valid SQL
    }
    ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
}

#[derive(Serialize)]
pub struct Person {
    pub id: i64,
    pub name: String,
    pub hidden: bool,
    pub faces: i64,
    pub photos: i64,
    pub cover: Option<i64>,
}

pub fn people(c: &Connection) -> Result<Vec<Person>, String> {
    prune(c)?;
    let mut st = c
        .prepare(
            "SELECT p.id, p.name, p.hidden, COUNT(f.id), COUNT(DISTINCT f.photo_id), \
             COALESCE((SELECT id FROM faces WHERE id = p.cover_face AND person_id = p.id), \
                      (SELECT id FROM faces WHERE person_id = p.id ORDER BY score DESC LIMIT 1)) \
             FROM people p JOIN faces f ON f.person_id = p.id JOIN photos ph ON ph.id = f.photo_id AND ph.deleted_at IS NULL \
             GROUP BY p.id ORDER BY COUNT(DISTINCT f.photo_id) DESC, p.name, p.id",
        )
        .map_err(es)?;
    let rows = st
        .query_map([], |r| Ok(Person { id: r.get(0)?, name: r.get(1)?, hidden: r.get(2)?, faces: r.get(3)?, photos: r.get(4)?, cover: r.get(5)? }))
        .map_err(es)?;
    rows.collect::<Result<_, _>>().map_err(es)
}

#[derive(Serialize)]
pub struct Face {
    pub id: i64,
    pub photo_id: i64,
    pub person_id: Option<i64>,
    /// x, y, w, h as fractions of the photo
    pub bbox: [f64; 4],
    pub score: f64,
    pub captured_at: Option<String>,
    pub filename: String,
    /// similarity to the face searched with (similar faces only)
    pub similarity: Option<f32>,
}

const FACE_COLS: &str = "f.id, f.photo_id, f.person_id, f.x, f.y, f.w, f.h, f.score, ph.captured_at, ph.filename";

fn face_row(r: &rusqlite::Row) -> rusqlite::Result<Face> {
    Ok(Face {
        id: r.get(0)?,
        photo_id: r.get(1)?,
        person_id: r.get(2)?,
        bbox: [r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?],
        score: r.get(7)?,
        captured_at: r.get(8)?,
        filename: r.get(9)?,
        similarity: None,
    })
}

pub fn faces_of(c: &Connection, person: i64) -> Result<Vec<Face>, String> {
    let mut st = c
        .prepare(&format!(
            "SELECT {FACE_COLS} FROM faces f JOIN photos ph ON ph.id = f.photo_id AND ph.deleted_at IS NULL \
             WHERE f.person_id = ? ORDER BY COALESCE(ph.captured_at, ''), f.id"
        ))
        .map_err(es)?;
    let rows = st.query_map([person], face_row).map_err(es)?;
    rows.collect::<Result<_, _>>().map_err(es)
}

/// Faces of other people that look like `face`, most alike first: for fixing a person that was split.
pub fn similar(c: &Connection, face: i64, limit: usize) -> Result<Vec<Face>, String> {
    let (e, person): (Vec<u8>, Option<i64>) = c.query_row("SELECT embedding, person_id FROM faces WHERE id = ?", [face], |r| Ok((r.get(0)?, r.get(1)?))).map_err(es)?;
    let e = vector(&e);
    let mut st = c
        .prepare(&format!("SELECT {FACE_COLS}, f.embedding FROM faces f JOIN photos ph ON ph.id = f.photo_id AND ph.deleted_at IS NULL WHERE f.id != ?"))
        .map_err(es)?;
    let mut out: Vec<Face> = st
        .query_map([face], |r| {
            let mut f = face_row(r)?;
            f.similarity = Some(dot(&e, &vector(&r.get::<_, Vec<u8>>(10)?)));
            Ok(f)
        })
        .map_err(es)?
        .flatten()
        .filter(|f| f.similarity.unwrap_or(0.0) >= MATCH - 0.12 && (person.is_none() || f.person_id != person))
        .collect();
    out.sort_by(|a, b| b.similarity.unwrap_or(0.0).total_cmp(&a.similarity.unwrap_or(0.0)));
    out.truncate(limit);
    Ok(out)
}

/// Name a person. A name someone else already has (ignoring case) means they are the same person: the two are
/// merged into the one that had it. Returns the person the faces end up in.
pub fn rename(c: &mut Connection, person: i64, name: &str) -> Result<i64, String> {
    let name = name.trim();
    let same: Option<i64> = if name.is_empty() {
        None
    } else {
        c.query_row("SELECT id FROM people WHERE name = ? COLLATE NOCASE AND id != ? ORDER BY id LIMIT 1", params![name, person], |r| r.get(0)).optional().map_err(es)?
    };
    match same {
        Some(into) => merge(c, &[into, person]),
        None => {
            c.execute("UPDATE people SET name = ? WHERE id = ?", params![name, person]).map_err(es)?;
            Ok(person)
        }
    }
}

pub fn set_hidden(c: &Connection, people: &[i64], hidden: bool) -> Result<(), String> {
    c.execute(&format!("UPDATE people SET hidden = ? WHERE id IN ({})", list(people)), [hidden]).map_err(es)?;
    Ok(())
}

pub fn set_cover(c: &Connection, person: i64, face: i64) -> Result<(), String> {
    c.execute("UPDATE people SET cover_face = ? WHERE id = ?", params![face, person]).map_err(es)?;
    Ok(())
}

/// Everyone in `people` becomes the first of them; it keeps the first name any of them has.
pub fn merge(c: &mut Connection, people: &[i64]) -> Result<i64, String> {
    let (&into, rest) = people.split_first().ok_or("pick the people to merge")?;
    if rest.is_empty() {
        return Ok(into);
    }
    let tx = c.transaction().map_err(es)?;
    let name: Option<String> = tx
        .query_row(&format!("SELECT name FROM people WHERE id IN ({}) AND name != '' ORDER BY id = ? DESC LIMIT 1", list(people)), [into], |r| r.get(0))
        .optional()
        .map_err(es)?;
    tx.execute(&format!("UPDATE faces SET person_id = ? WHERE person_id IN ({})", list(rest)), [into]).map_err(es)?;
    tx.execute("UPDATE people SET name = COALESCE(?, name) WHERE id = ?", params![name, into]).map_err(es)?;
    prune(&tx)?;
    tx.commit().map_err(es)?;
    Ok(into)
}

/// Move faces to `person`, or to a new person when None. Returns the person.
pub fn move_faces(c: &mut Connection, faces: &[i64], person: Option<i64>) -> Result<i64, String> {
    let tx = c.transaction().map_err(es)?;
    let person = match person {
        Some(p) => p,
        None => {
            tx.execute("INSERT INTO people DEFAULT VALUES", []).map_err(es)?;
            tx.last_insert_rowid()
        }
    };
    tx.execute(&format!("UPDATE faces SET person_id = ? WHERE id IN ({})", list(faces)), [person]).map_err(es)?;
    prune(&tx)?;
    tx.commit().map_err(es)?;
    Ok(person)
}

/// Forget faces (not a face, a stranger): their photos stay scanned, so they do not come back.
pub fn delete_faces(core: &Core, faces: &[i64]) -> Result<(), String> {
    let c = core.db();
    c.execute(&format!("DELETE FROM faces WHERE id IN ({})", list(faces)), []).map_err(es)?;
    prune(&c)?;
    faces.iter().for_each(|f| drop(fs::remove_file(crops_dir(core).join(format!("{f}.jpg")))));
    Ok(())
}

pub fn delete_people(core: &Core, people: &[i64]) -> Result<(), String> {
    let ids: Vec<i64> = {
        let c = core.db();
        let mut st = c.prepare(&format!("SELECT id FROM faces WHERE person_id IN ({})", list(people))).map_err(es)?;
        st.query_map([], |r| r.get(0)).map_err(es)?.collect::<Result<_, _>>().map_err(es)?
    };
    delete_faces(core, &ids)?;
    core.db().execute(&format!("DELETE FROM people WHERE id IN ({})", list(people)), []).map_err(es)?;
    Ok(())
}

/// Forget every face and person, so the next scan starts over.
pub fn reset(core: &Core) -> Result<(), String> {
    core.db().execute_batch("DELETE FROM faces; DELETE FROM people; DELETE FROM face_scans;").map_err(es)?;
    let _ = fs::remove_dir_all(crops_dir(core));
    Ok(())
}

pub fn stats(c: &Connection) -> Result<Value, String> {
    let one = |sql: &str| c.query_row(sql, [], |r| r.get::<_, i64>(0)).map_err(es);
    let pairs = |sql: &str| -> Result<Vec<(Value, i64)>, String> {
        let mut st = c.prepare(sql).map_err(es)?;
        let rows = st.query_map([], |r| Ok((r.get::<_, rusqlite::types::Value>(0)?, r.get(1)?))).map_err(es)?;
        Ok(rows
            .flatten()
            .map(|(k, n)| {
                let k = match k {
                    rusqlite::types::Value::Integer(i) => json!(i),
                    rusqlite::types::Value::Text(t) => json!(t),
                    _ => Value::Null,
                };
                (k, n)
            })
            .collect())
    };
    let live = "JOIN photos ph ON ph.id = f.photo_id AND ph.deleted_at IS NULL";
    Ok(json!({
        "eligible": one("SELECT COUNT(*) FROM photos WHERE master_id IS NULL AND deleted_at IS NULL AND path NOT LIKE 'immich://%'")?,
        "scanned": one("SELECT COUNT(*) FROM face_scans s JOIN photos ph ON ph.id = s.photo_id AND ph.deleted_at IS NULL")?,
        "with_faces": one(&format!("SELECT COUNT(DISTINCT f.photo_id) FROM faces f {live}"))?,
        "faces": one(&format!("SELECT COUNT(*) FROM faces f {live}"))?,
        "people": one("SELECT COUNT(DISTINCT person_id) FROM faces WHERE person_id IS NOT NULL")?,
        "named": one("SELECT COUNT(*) FROM people WHERE name != ''")?,
        "hidden": one("SELECT COUNT(*) FROM people WHERE hidden")?,
        "by_year": pairs(&format!("SELECT substr(ph.captured_at, 1, 4), COUNT(DISTINCT f.photo_id) FROM faces f {live} WHERE ph.captured_at IS NOT NULL GROUP BY 1 ORDER BY 1"))?,
        "per_photo": pairs("SELECT MIN(faces, 6), COUNT(*) FROM face_scans s JOIN photos ph ON ph.id = s.photo_id AND ph.deleted_at IS NULL GROUP BY MIN(faces, 6) ORDER BY 1")?,
    }))
}

// --- commands ---

type St<'a> = tauri::State<'a, Arc<Core>>;

#[tauri::command(async)]
pub fn faces_scan(core: St, ids: Option<Vec<i64>>) -> Result<Value, String> {
    let _one = SCANNING.try_lock().map_err(|_| "a face scan is already running")?;
    let job = Job::start(&core, "faces-scan");
    scan(&core, ids.as_deref(), &job)
}

#[tauri::command(async)]
pub fn faces_people(core: St) -> Result<Vec<Person>, String> {
    people(&core.db())
}

#[tauri::command(async)]
pub fn faces_of_person(core: St, person: i64) -> Result<Vec<Face>, String> {
    faces_of(&core.db(), person)
}

#[tauri::command(async)]
pub fn faces_similar(core: St, face: i64) -> Result<Vec<Face>, String> {
    similar(&core.db(), face, 200)
}

#[tauri::command(async)]
pub fn faces_stats(core: St) -> Result<Value, String> {
    stats(&core.db())
}

#[tauri::command(async)]
pub fn faces_rename(core: St, person: i64, name: String) -> Result<i64, String> {
    rename(&mut core.db(), person, &name)
}

#[tauri::command(async)]
pub fn faces_hide(core: St, people: Vec<i64>, hidden: bool) -> Result<(), String> {
    set_hidden(&core.db(), &people, hidden)
}

#[tauri::command(async)]
pub fn faces_set_cover(core: St, person: i64, face: i64) -> Result<(), String> {
    set_cover(&core.db(), person, face)
}

#[tauri::command(async)]
pub fn faces_merge(core: St, people: Vec<i64>) -> Result<i64, String> {
    merge(&mut core.db(), &people)
}

#[tauri::command(async)]
pub fn faces_move(core: St, faces: Vec<i64>, person: Option<i64>) -> Result<i64, String> {
    move_faces(&mut core.db(), &faces, person)
}

#[tauri::command(async)]
pub fn faces_delete(core: St, faces: Vec<i64>) -> Result<(), String> {
    delete_faces(&core, &faces)
}

#[tauri::command(async)]
pub fn faces_delete_people(core: St, people: Vec<i64>) -> Result<(), String> {
    delete_people(&core, &people)
}

#[tauri::command(async)]
pub fn faces_reset(core: St) -> Result<(), String> {
    reset(&core)
}

/// The photos that show every one of `people` (virtual copies included), for the library filter.
pub fn filter_sql(people: &[i64]) -> String {
    people.iter().map(|p| format!(" AND COALESCE(master_id, id) IN (SELECT photo_id FROM faces WHERE person_id = {p})")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE photos (id INTEGER PRIMARY KEY, master_id INTEGER, deleted_at TEXT, captured_at TEXT, filename TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '');").unwrap();
        c.execute_batch(SCHEMA).unwrap();
        c
    }

    fn face(c: &Connection, photo: i64, person: i64, e: &[f32]) -> i64 {
        c.execute("INSERT OR IGNORE INTO photos (id, captured_at) VALUES (?, ?)", params![photo, format!("{}:06:01", 2020 + photo % 2)]).unwrap();
        c.execute("INSERT OR IGNORE INTO people (id) VALUES (?)", [person]).unwrap();
        c.execute("INSERT INTO faces (photo_id, person_id, x, y, w, h, score, embedding) VALUES (?, ?, 0, 0, 0.1, 0.1, 0.9, ?)", params![photo, person, blob(e)]).unwrap();
        c.last_insert_rowid()
    }

    #[test]
    fn centroids_merge_move_prune_and_similar() {
        let mut c = db();
        let a = face(&c, 1, 10, &[1.0, 0.0]);
        face(&c, 2, 10, &[0.8, 0.6]);
        let b = face(&c, 3, 20, &[0.0, 1.0]);
        let cs = Centroids::load(&c).unwrap();
        assert_eq!(cs.best(&[1.0, 0.0]).unwrap().0, 10);
        assert_eq!(cs.best(&[0.0, 1.0]).unwrap().0, 20);
        // similar: face a looks like nothing in person 20 at this threshold, and never lists its own person
        assert!(similar(&c, a, 10).unwrap().iter().all(|f| f.person_id != Some(10)));
        rename(&mut c, 20, " Bo ").unwrap();
        assert_eq!(merge(&mut c, &[10, 20]).unwrap(), 10);
        let p = people(&c).unwrap();
        assert_eq!((p.len(), p[0].name.as_str(), p[0].photos), (1, "Bo", 3));
        let new = move_faces(&mut c, &[b], None).unwrap();
        assert_eq!(people(&c).unwrap().len(), 2);
        move_faces(&mut c, &[b], Some(10)).unwrap();
        assert!(people(&c).unwrap().iter().all(|p| p.id != new), "an emptied person is pruned");
        // the library filter: photos showing everyone listed
        // typing a name someone else has merges the two into that person
        let other = move_faces(&mut c, &[b], None).unwrap();
        assert_eq!(rename(&mut c, other, "bo").unwrap(), 10);
        let p = people(&c).unwrap();
        assert_eq!((p.len(), p[0].name.as_str(), p[0].photos), (1, "Bo", 3));
        let n: i64 = c.query_row(&format!("SELECT COUNT(*) FROM photos WHERE 1=1{}", filter_sql(&[10])), [], |r| r.get(0)).unwrap();
        assert_eq!(n, 3);
        let st = stats(&c).unwrap();
        assert!(st["faces"] == 3);
        // grouped by year, not by the faces' own `y` column
        assert_eq!(st["by_year"], json!([["2020", 1], ["2021", 2]]));
    }

    /// Needs OPTIXEL_TEST_AI=1 and the face model (or face swap model) installed in the app's data folder:
    /// scans two copies of the demo portrait and the demo landscape, and expects one person in two photos.
    #[test]
    #[cfg(unix)]
    fn scan_groups_the_same_face() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("ols-faces-test-{}", crate::new_token()));
        let pics = tmp.join("pics");
        fs::create_dir_all(&pics).unwrap();
        let html = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../html");
        fs::copy(html.join("demo-portrait.jpg"), pics.join("a.jpg")).unwrap();
        fs::copy(html.join("demo-portrait.jpg"), pics.join("b.jpg")).unwrap();
        fs::copy(html.join("demo-landscape.jpg"), pics.join("c.jpg")).unwrap();
        let core = Arc::new(Core::open(tmp.join("data")).unwrap());
        std::os::unix::fs::symlink(crate::data_dir().join("models"), tmp.join("data/models")).unwrap();
        catalog::import(&core, &pics, &|_, _| {}).unwrap();
        let r = scan(&core, None, &Job::start(&core, "faces-scan")).unwrap();
        eprintln!("{r}");
        assert_eq!(r["photos"], 3);
        let ps = people(&core.db()).unwrap();
        assert_eq!(ps[0].photos, 2, "the same face in two photos is one person");
        assert!(crops_dir(&core).join(format!("{}.jpg", ps[0].cover.unwrap())).is_file());
        rename(&mut core.db(), ps[0].id, "Demo").unwrap();
        let by_name = catalog::query(&core.db(), &catalog::Filter { text: Some("demo".into()), ..Default::default() }).unwrap();
        let by_person = catalog::query(&core.db(), &catalog::Filter { people: Some(vec![ps[0].id]), ..Default::default() }).unwrap();
        assert_eq!((by_name.len(), by_person.len()), (2, 2));
        // a rescan finds nothing new
        assert_eq!(scan(&core, None, &Job::start(&core, "faces-scan")).unwrap()["photos"], 0);
        let _ = fs::remove_dir_all(&tmp);
    }
}
