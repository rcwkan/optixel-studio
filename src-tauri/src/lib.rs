mod ai;
mod aimodel;
mod brush;
mod auto;
mod catalog;
mod cloud;
mod develop;
mod enhance;
mod faces;
mod faceswap;
mod gpu;
mod icc;
mod libraw;
mod mcp;
mod moebius;
mod perspective;
mod remote;
mod retouch;
mod sdfill;
mod storage;
mod updater;
mod vision;
mod xmp;

use develop::{Develop, Img, Opts, es};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, atomic::AtomicBool},
};
use tauri::{Emitter, Manager};

pub const APP_ID: &str = "org.optixel.studio";
/// Long edge of the in-memory develop proxy.
pub const PROXY_EDGE: u32 = 1600;

fn default_true() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub theme: String,
    pub mcp_enabled: bool,
    pub mcp_port: u16,
    pub mcp_token: String,
    pub providers: Vec<ai::Provider>,
    pub active_provider: String,
    /// Connected Immich server; its API key is in the keychain.
    pub immich_url: String,
    /// OAuth client of the signed-in Google account; secret and refresh token are in the keychain.
    pub google_client_id: String,
    /// Apple Photos Library or iCloud Photos sync path.
    pub apple_photos_path: String,
    /// Amazon Photos sync folder path or endpoint.
    pub amazon_photos_path: String,
    /// Adobe Creative Cloud / Lightroom Photos sync path.
    pub adobe_cloud_path: String,
    /// Advanced settings of the auto functions, and what runs on new photos.
    pub auto: auto::AutoSettings,
    /// Use the GPU for image processing when one is detected.
    pub gpu_accel: bool,
    /// The AI models the Develop AI tab uses (ids from aimodel.rs); empty: the best one installed.
    pub ai_denoise_model: String,
    pub ai_removal_model: String,
    pub ai_align_model: String,
    /// Shares on a NAS or server, browsed in place like folders.
    pub network_drives: Vec<storage::NetworkDrive>,
    /// Cloud storage accounts; their secrets are in the keychain.
    pub clouds: Vec<cloud::Account>,
    /// Automatically check for updates on startup.
    #[serde(default = "default_true")]
    pub auto_check_updates: bool,
    /// Commercial license key for pre-built binaries and automatic update entitlements.
    #[serde(default)]
    pub license_key: Option<String>,
    /// People: find and group faces (ArcFace), scan new imports, search by person.
    pub faces_enabled: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: "system".into(),
            mcp_enabled: true,
            mcp_port: 41920,
            mcp_token: new_token(),
            providers: ai::default_providers(),
            active_provider: "cli:claude".into(),
            immich_url: String::new(),
            google_client_id: String::new(),
            apple_photos_path: String::new(),
            amazon_photos_path: String::new(),
            adobe_cloud_path: String::new(),
            auto: Default::default(),
            gpu_accel: true,
            ai_denoise_model: String::new(),
            ai_removal_model: String::new(),
            ai_align_model: String::new(),
            network_drives: vec![],
            clouds: vec![],
            auto_check_updates: true,
            license_key: None,
            faces_enabled: false,
        }
    }
}

impl Settings {
    pub fn load(dir: &Path) -> Settings {
        let mut s: Settings = fs::read_to_string(dir.join("settings.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        for p in &mut s.providers {
            if p.id == "cli:antigravity" && (p.command.starts_with("antigravity run") || p.command.is_empty()) {
                p.command = "antigravity -p {{AGENT_PROMPT}} --dangerously-skip-permissions".into();
            }
        }
        s
    }
}

pub fn new_token() -> String {
    let mut b = [0u8; 24];
    getrandom::fill(&mut b).expect("OS random source");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Default catalog location in standard OS application data directory.
pub fn default_data_dir() -> PathBuf {
    #[cfg(test)]
    {
        std::env::temp_dir().join("optixel-test-default")
    }
    #[cfg(not(test))]
    {
        dirs::data_dir().unwrap_or_else(std::env::temp_dir).join(APP_ID)
    }
}

/// Pointer file stored in default application data directory referencing custom storage folder.
pub fn pointer_file() -> PathBuf {
    #[cfg(test)]
    {
        std::env::temp_dir().join("optixel-test-default").join("storage_location.txt")
    }
    #[cfg(not(test))]
    {
        default_data_dir().join("storage_location.txt")
    }
}

/// Check if a custom storage directory is configured via environment variable or pointer file.
pub fn configured_data_dir() -> Option<PathBuf> {
    if let Some(custom) = std::env::var_os("OPTIXEL_DATA_DIR") {
        let p = PathBuf::from(custom);
        if p.is_dir() {
            return Some(p);
        }
    }
    #[cfg(test)]
    {
        let p = pointer_file();
        if let Ok(content) = fs::read_to_string(&p) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                let path = PathBuf::from(trimmed);
                if path.is_dir() {
                    return Some(path);
                }
            }
        }
        return None;
    }
    #[cfg(not(test))]
    {
        let p = pointer_file();
        if let Ok(content) = fs::read_to_string(&p) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                let path = PathBuf::from(trimmed);
                if path.is_dir() {
                    return Some(path);
                } else {
                    // Stale or temp directory that no longer exists - remove stale pointer file
                    let _ = fs::remove_file(&p);
                }
            }
        }
        if let Some(config_dir) = dirs::config_dir() {
            let alt = config_dir.join(APP_ID).join("storage_location.txt");
            if alt != p {
                if let Ok(content) = fs::read_to_string(&alt) {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        let path = PathBuf::from(trimmed);
                        if path.is_dir() {
                            return Some(path);
                        } else {
                            let _ = fs::remove_file(&alt);
                        }
                    }
                }
            }
        }
        None
    }
}

/// Catalog location; `OPTIXEL_DATA_DIR` selects another catalog folder.
pub fn data_dir() -> PathBuf {
    configured_data_dir().unwrap_or_else(default_data_dir)
}

/// Persist custom storage pointer file, or remove if reverting to default.
pub fn persist_storage_location(target: Option<&Path>) -> Result<(), String> {
    #[cfg(test)]
    {
        let def = default_data_dir();
        let p = pointer_file();
        match target {
            Some(t) if t != def => {
                let _ = fs::create_dir_all(&def);
                fs::write(&p, t.to_string_lossy().as_bytes()).map_err(es)?;
            }
            _ => {
                let _ = fs::remove_file(&p);
            }
        }
        return Ok(());
    }
    #[cfg(not(test))]
    {
        let def = default_data_dir();
        let p = pointer_file();
        let norm_def = storage::canonicalize_or_parent(&def);
        let norm_target = target.map(storage::canonicalize_or_parent);
        let is_custom = match &norm_target {
            Some(t) => t != &norm_def,
            None => false,
        };

        if is_custom {
            if let Some(t) = target {
                let _ = fs::create_dir_all(&def);
                fs::write(&p, t.to_string_lossy().as_bytes()).map_err(es)?;
                if let Some(config_dir) = dirs::config_dir() {
                    let alt_dir = config_dir.join(APP_ID);
                    if alt_dir != def {
                        let _ = fs::create_dir_all(&alt_dir);
                        let _ = fs::write(alt_dir.join("storage_location.txt"), t.to_string_lossy().as_bytes());
                    }
                }
            }
        } else {
            let _ = fs::remove_file(&p);
            if let Some(config_dir) = dirs::config_dir() {
                let alt_dir = config_dir.join(APP_ID);
                let _ = fs::remove_file(alt_dir.join("storage_location.txt"));
            }
        }
        Ok(())
    }
}

/// Shared engine state used by the GUI commands, the MCP server and the AI router.
pub struct Core {
    db: Mutex<rusqlite::Connection>,
    dir: RwLock<PathBuf>,
    pub settings: Mutex<Settings>,
    pub selection: Mutex<Vec<i64>>,
    pub mcp_status: Mutex<String>,
    /// Cancel flags of the running jobs (auto functions, duplicating), by job id.
    pub jobs: Mutex<HashMap<String, Arc<AtomicBool>>>,
    cache: Mutex<Vec<((String, develop::Raw), Arc<Img>)>>,
    /// The one full-size decode kept for zoomed views (24 MP is ~300 MB of floats).
    full: Mutex<Option<((String, develop::Raw), Arc<Img>)>>,
    // ponytail: one full-size decode at a time bounds memory; a decode pool if imports/exports need throughput
    decode_lock: Mutex<()>,
    app: OnceLock<tauri::AppHandle>,
}

impl Core {
    pub fn open(dir: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(dir.join("thumbs")).map_err(es)?;
        let core = Core {
            db: Mutex::new(catalog::open(&dir.join("catalog.olcat"))?),
            settings: Mutex::new(Settings::load(&dir)),
            dir: RwLock::new(dir),
            selection: Default::default(),
            mcp_status: Mutex::new("not started".into()),
            jobs: Default::default(),
            cache: Default::default(),
            full: Default::default(),
            decode_lock: Mutex::new(()),
            app: OnceLock::new(),
        };
        core.save_settings()?; // persists the token generated on first run
        Ok(core)
    }

    pub fn dir(&self) -> PathBuf {
        self.dir.read().unwrap().clone()
    }

    pub fn switch_storage_engine(&self, new_dir: PathBuf, new_db: rusqlite::Connection) -> Result<(), String> {
        fs::create_dir_all(new_dir.join("thumbs")).map_err(es)?;
        {
            let mut db_guard = self.db.lock().unwrap();
            let mut dir_guard = self.dir.write().unwrap();
            *db_guard = new_db;
            *dir_guard = new_dir.clone();
        }
        // If the new directory has its own settings.json, load it;
        // otherwise preserve the current settings so preferences and credentials aren't lost
        let new_settings = if new_dir.join("settings.json").is_file() {
            Settings::load(&new_dir)
        } else {
            self.settings.lock().unwrap().clone()
        };
        *self.settings.lock().unwrap() = new_settings;
        self.save_settings()?;
        self.cache.lock().unwrap().clear();
        *self.full.lock().unwrap() = None;
        if let Some(app) = self.app.get() {
            let _ = app.asset_protocol_scope().allow_directory(new_dir.join("thumbs"), true);
            let _ = app.emit("storage-location-changed", serde_json::json!({
                "new_dir": new_dir.to_string_lossy(),
            }));
        }
        Ok(())
    }

    pub fn db(&self) -> MutexGuard<'_, rusqlite::Connection> {
        self.db.lock().unwrap()
    }

    pub fn save_settings(&self) -> Result<(), String> {
        gpu::set_enabled(self.settings.lock().unwrap().gpu_accel);
        let path = self.dir().join("settings.json");
        let s = serde_json::to_string_pretty(&*self.settings.lock().unwrap()).map_err(es)?;
        fs::write(&path, s).map_err(es)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600)); // holds the MCP token
        }
        Ok(())
    }

    pub fn thumb_path(&self, id: i64) -> PathBuf {
        self.dir().join("thumbs").join(format!("{id}.jpg"))
    }

    pub fn emit(&self, event: &str, payload: Value) {
        if let Some(app) = self.app.get() {
            let _ = app.emit(event, payload);
        }
    }

    /// Activity line for the Agent & MCP console.
    pub fn log(&self, source: &str, message: impl Into<String>) {
        self.emit("agent-log", json!({"source": source, "message": message.into()}));
    }

    pub fn photo(&self, id: i64) -> Result<catalog::Photo, String> {
        let f = catalog::Filter { ids: Some(vec![id]), ..Default::default() };
        catalog::query(&self.db(), &f)?.pop().ok_or_else(|| format!("photo {id} not found"))
    }

    pub fn develop(&self, id: i64) -> Result<Develop, String> {
        catalog::get_develop(&self.db(), id)
    }

    /// The file behind a catalog path; an `immich://` photo downloads on first use.
    pub fn local_file(&self, path: &str) -> Result<PathBuf, String> {
        match remote::immich_id(path) {
            Some(id) => remote::immich_original(self, id),
            None => Ok(PathBuf::from(path)),
        }
    }

    /// Full decode of a file, one at a time (full-size decodes are memory-heavy).
    pub fn decode(&self, file: &Path, max_edge: Option<u32>) -> Result<Img, String> {
        self.decode_with(file, max_edge, &develop::Raw::default())
    }

    pub fn decode_with(&self, file: &Path, max_edge: Option<u32>, raw: &develop::Raw) -> Result<Img, String> {
        let _g = self.decode_lock.lock().unwrap();
        develop::load_with(file, max_edge, raw)
    }

    /// Decoded working-space proxy, cached for the last few files.
    pub fn proxy(&self, path: &str) -> Result<Arc<Img>, String> {
        self.proxy_with(path, &develop::Raw::default())
    }

    pub fn proxy_with(&self, path: &str, raw: &develop::Raw) -> Result<Arc<Img>, String> {
        if let Some((_, img)) = self.cache.lock().unwrap().iter().find(|((p, r), _)| p == path && r == raw) {
            return Ok(img.clone());
        }
        let file = self.local_file(path)?;
        let img = Arc::new(self.decode_with(&file, Some(PROXY_EDGE), raw)?);
        let mut c = self.cache.lock().unwrap();
        let key = (path.to_string(), raw.clone());
        c.retain(|(k, _)| k != &key);
        c.insert(0, (key, img.clone()));
        c.truncate(12);
        Ok(img)
    }

    /// Full-size working image of a photo, for views zoomed past the proxy.
    pub fn full_image(&self, path: &str) -> Result<Arc<Img>, String> {
        self.full_image_with(path, &develop::Raw::default())
    }

    pub fn full_image_with(&self, path: &str, raw: &develop::Raw) -> Result<Arc<Img>, String> {
        if let Some((_, img)) = self.full.lock().unwrap().as_ref().filter(|((p, r), _)| p == path && r == raw) {
            return Ok(img.clone());
        }
        let img = Arc::new(self.decode_with(&self.local_file(path)?, None, raw)?);
        *self.full.lock().unwrap() = Some(((path.to_string(), raw.clone()), img.clone()));
        Ok(img)
    }

    pub fn ai_cache_dir(&self, id: i64) -> PathBuf {
        let p = self.dir().join("cache").join("ai").join(id.to_string());
        let _ = fs::create_dir_all(&p);
        p
    }

    /// Render the current edit of a photo at proxy size.
    pub fn render(&self, id: i64, opts: &Opts) -> Result<Img, String> {
        let p = self.photo(id)?;
        let d = self.develop(id)?;
        let mut opts = opts.clone();
        if opts.ai_cache_dir.is_none() {
            opts.ai_cache_dir = Some(self.ai_cache_dir(id));
        }
        Ok(develop::render(&*self.proxy_with(&p.path, &d.raw)?, &d, &opts))
    }

    /// Persist an edit as one history step (one undo = one GUI gesture or one AI/MCP call).
    pub fn commit(self: &Arc<Self>, id: i64, d: &Develop, label: &str) -> Result<(), String> {
        self.commit_with_replace(id, d, label, None)
    }

    pub fn commit_with_replace(
        self: &Arc<Self>,
        id: i64,
        d: &Develop,
        label: &str,
        replace_prefix: Option<&str>,
    ) -> Result<(), String> {
        catalog::save_develop_ext(&self.db(), id, d, label, replace_prefix)?;
        self.refresh(id, d)
    }

    /// Sidecar + thumbnail + UI notification after the stored state changed.
    pub fn refresh(self: &Arc<Self>, id: i64, d: &Develop) -> Result<(), String> {
        let photo = self.photo(id)?;
        catalog::write_sidecar(&photo, d);
        self.emit("photo-changed", json!({"id": id}));
        let (core, d) = (self.clone(), d.clone());
        std::thread::spawn(move || {
            let res = (|| -> Result<(), String> {
                let out = develop::render(&*core.proxy_with(&photo.path, &d.raw)?, &d, &Opts { ai_cache_dir: Some(core.ai_cache_dir(id)), ..Default::default() });
                let thumb = catalog::to_dynamic(&out).thumbnail(catalog::THUMB, catalog::THUMB);
                catalog::write_jpeg(&core.thumb_path(id), &thumb)?;
                core.db().execute("UPDATE photos SET thumb_v = thumb_v + 1 WHERE id = ?", [id]).map_err(es)?;
                Ok(())
            })();
            match res {
                Ok(()) => core.emit("thumb-changed", json!({"id": id})),
                Err(e) => eprintln!("thumbnail {id}: {e}"),
            }
        });
        Ok(())
    }

    /// Rating, flag, label or keyword change. Stars and pick/reject of Immich photos are
    /// mirrored to the server in the background; a failure shows in the status bar.
    pub fn cull(self: &Arc<Self>, ids: &[i64], u: &catalog::Culling) -> Result<usize, String> {
        let n = catalog::update(&self.db(), ids, u)?;
        if u.rating.is_some() || u.flag.is_some() {
            let (core, ids) = (self.clone(), ids.to_vec());
            std::thread::spawn(move || {
                if let Err(e) = remote::immich_push(&core, &ids) {
                    eprintln!("immich push: {e}");
                    core.emit("remote-error", json!(format!("Immich was not updated: {e}")));
                }
            });
        }
        Ok(n)
    }

    pub fn virtual_copy(&self, id: i64) -> Result<i64, String> {
        let new = catalog::virtual_copy(&self.db(), id)?;
        let _ = fs::copy(self.thumb_path(id), self.thumb_path(new));
        self.emit("catalog-changed", json!({}));
        Ok(new)
    }

    pub fn save_as_new_photo(self: &Arc<Self>, id: i64, develop: Option<Develop>, name: Option<String>) -> Result<catalog::Photo, String> {
        let mut d = develop.unwrap_or_else(|| self.develop(id).unwrap_or_default());
        d.clamp();
        let new_id = catalog::save_as_new_photo(&self.db(), id, Some(&d), name.as_deref())?;
        let _ = fs::copy(self.thumb_path(id), self.thumb_path(new_id));
        self.refresh(new_id, &d)?;
        self.emit("catalog-changed", json!({"id": new_id}));
        self.photo(new_id)
    }

    /// Removes photos and their virtual copies from the catalog; files on disk stay.
    pub fn remove(&self, ids: &[i64]) -> Result<usize, String> {
        let gone = catalog::remove(&self.db(), ids)?;
        for &id in &gone {
            let _ = fs::remove_file(self.thumb_path(id));
        }
        self.emit("catalog-changed", json!({}));
        Ok(gone.len())
    }

    /// Moves photos and their virtual copies to the bin (soft delete).
    pub fn move_to_bin(&self, ids: &[i64]) -> Result<usize, String> {
        let moved = catalog::move_to_bin(&self.db(), ids)?;
        self.emit("catalog-changed", json!({}));
        Ok(moved.len())
    }

    /// Restores photos and their virtual copies from the bin back to the library.
    pub fn restore_from_bin(&self, ids: &[i64]) -> Result<usize, String> {
        let restored = catalog::restore_from_bin(&self.db(), ids)?;
        self.emit("catalog-changed", json!({}));
        Ok(restored.len())
    }

    /// Permanently deletes photos from the bin, including actual files on disk for originals.
    pub fn permanent_delete(&self, ids: &[i64]) -> Result<usize, String> {
        // Look up photos to identify original files on disk
        let photos = catalog::photos(&self.db(), ids)?;
        for p in &photos {
            // Delete file from disk only if it is an original (not a virtual copy)
            if p.master_id.is_none() && !p.path.starts_with("immich://") {
                let path = Path::new(&p.path);
                if path.exists() {
                    let _ = fs::remove_file(path);
                }
                // Also remove possible sidecar .xmp files
                let xmp1 = path.with_extension("xmp");
                if xmp1 != path && xmp1.exists() {
                    let _ = fs::remove_file(xmp1);
                }
                let xmp2 = PathBuf::from(format!("{}.xmp", p.path));
                if xmp2 != path && xmp2.exists() {
                    let _ = fs::remove_file(xmp2);
                }
            }
        }
        let gone = catalog::remove(&self.db(), ids)?;
        for &id in &gone {
            let _ = fs::remove_file(self.thumb_path(id));
        }
        self.emit("catalog-changed", json!({}));
        Ok(gone.len())
    }

    /// Empties the bin by permanently deleting all trashed photos and their files on disk.
    pub fn empty_bin(&self) -> Result<usize, String> {
        let c = self.db();
        let mut st = c.prepare("SELECT id FROM photos WHERE deleted_at IS NOT NULL").map_err(es)?;
        let ids: Vec<i64> = st.query_map([], |r| r.get(0)).map_err(es)?.flatten().collect();
        if ids.is_empty() {
            return Ok(0);
        }
        self.permanent_delete(&ids)
    }

    /// Count of photos currently in the bin.
    pub fn bin_count(&self) -> Result<i64, String> {
        catalog::bin_count(&self.db())
    }

    /// Full-resolution export of the current edits. Never overwrites existing files.
    pub fn export(&self, ids: &[i64], folder: &Path, format: &str, color_space: &str, quality: u8, max_edge: Option<u32>) -> Result<Vec<String>, String> {
        fs::create_dir_all(folder).map_err(es)?;
        let ext = develop::file_ext(format);
        let mut out = vec![];
        for (i, &id) in ids.iter().enumerate() {
            let p = self.photo(id)?;
            let d = self.develop(id)?;
            let src = self.decode_with(&self.local_file(&p.path)?, max_edge, &d.raw)?;
            let opts = Opts { ai_cache_dir: Some(self.ai_cache_dir(id)), ..Default::default() };
            let img = develop::render(&src, &d, &opts);
            let stem = Path::new(&p.filename).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let stem = match &p.copy_name {
                Some(c) => format!("{stem}-{}", c.replace(' ', "")),
                None => stem,
            };
            let mut stem: String = stem.chars().filter(|c| !r#"/\:*?"<>|"#.contains(*c) && !c.is_control()).collect();
            if stem.is_empty() {
                stem = "export".into();
            }
            let mut target = folder.join(format!("{stem}.{ext}"));
            let mut n = 1;
            while target.exists() {
                target = folder.join(format!("{stem}-{n}.{ext}"));
                n += 1;
            }
            develop::export(&img, &target, format, color_space, quality, p.captured_at.as_deref())?;
            self.emit("export-progress", json!({"done": i + 1, "total": ids.len()}));
            out.push(target.display().to_string());
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

type St<'a> = tauri::State<'a, Arc<Core>>;

#[tauri::command]
fn app_info(core: St) -> Value {
    let s = core.settings.lock().unwrap();
    let dir = core.dir();
    json!({
        "data_dir": dir,
        "thumbs_dir": dir.join("thumbs"),
        "exe": std::env::current_exe().ok(),
        "mcp_url": format!("http://127.0.0.1:{}/mcp", s.mcp_port),
        "mcp_status": *core.mcp_status.lock().unwrap(),
    })
}

#[tauri::command(async)]
fn query_photos(core: St, filter: catalog::Filter) -> Result<Vec<catalog::Photo>, String> {
    catalog::query(&core.db(), &filter)
}

#[tauri::command(async)]
fn list_folders(core: St) -> Result<Vec<(String, i64)>, String> {
    catalog::folders(&core.db())
}

#[tauri::command(async)]
fn import_folder(core: St, path: String) -> Result<usize, String> {
    let c = core.inner().clone();
    let ids = catalog::import_shallow(&core, Path::new(&path), &|done, total| {
        c.emit("import-progress", json!({"label": "Reading folder", "done": done, "total": total}))
    })?;
    core.emit("catalog-changed", json!({}));
    let n = ids.len();
    auto::after_import(core.inner(), ids, "folders");
    Ok(n)
}

/// Subfolders of `path` and the number of photos directly in it, without reading any photo.
#[tauri::command(async)]
fn browse_folder(path: String) -> Result<(Vec<String>, usize), String> {
    catalog::browse(Path::new(&path))
}

#[tauri::command]
fn set_selection(core: St, ids: Vec<i64>) {
    *core.selection.lock().unwrap() = ids;
}

#[tauri::command(async)]
fn update_photos(core: St, ids: Vec<i64>, update: catalog::Culling) -> Result<usize, String> {
    core.cull(&ids, &update)
}

#[tauri::command(async)]
fn virtual_copy(core: St, id: i64) -> Result<i64, String> {
    core.virtual_copy(id)
}

#[tauri::command(async)]
fn save_as_new_photo(core: St, id: i64, develop: Option<Develop>, name: Option<String>) -> Result<catalog::Photo, String> {
    core.save_as_new_photo(id, develop, name)
}

#[tauri::command(async)]
fn remove_photos(core: St, ids: Vec<i64>) -> Result<usize, String> {
    core.remove(&ids)
}

#[tauri::command(async)]
fn bin_count(core: St) -> Result<i64, String> {
    core.bin_count()
}

#[tauri::command(async)]
fn move_to_bin(core: St, ids: Vec<i64>) -> Result<usize, String> {
    core.move_to_bin(&ids)
}

#[tauri::command(async)]
fn restore_from_bin(core: St, ids: Vec<i64>) -> Result<usize, String> {
    core.restore_from_bin(&ids)
}

#[tauri::command(async)]
fn permanent_delete(core: St, ids: Vec<i64>) -> Result<usize, String> {
    core.permanent_delete(&ids)
}

#[tauri::command(async)]
fn empty_bin(core: St) -> Result<usize, String> {
    core.empty_bin()
}

#[tauri::command(async)]
fn get_develop(core: St, id: i64) -> Result<Develop, String> {
    core.develop(id)
}

#[tauri::command]
fn default_develop() -> Develop {
    Develop::default()
}

/// Preview-only overlays: soft-proof gamut (sRGB | DisplayP3 | AdobeRGB | ProPhotoRGB | ACEScg | FineArtMatte | BarytaGloss | custom .icc),
/// LittleCMS2 rendering intent & black-point compensation, clipping warnings, and Visualize Spots threshold.
#[derive(Deserialize, Default)]
#[serde(default)]
struct Proof {
    gamut: Option<String>,
    clipping: bool,
    spots: Option<f32>,
    intent: Option<String>,
    bpc: Option<bool>,
    simulate_paper: Option<bool>,
    gamut_warn: Option<bool>,
}

/// Binary reply: u32 width, u32 height, 4x256 u32 histogram (R, G, B, luma), RGBA8 pixels.
/// The histogram is of the photo itself, not of the overlays.
#[tauri::command(async)]
fn render_preview(core: St, id: i64, develop: Develop, skip_crop: bool, overlay_mask: Option<usize>, full_frame: Option<bool>, proof: Option<Proof>, hires: Option<bool>) -> Result<tauri::ipc::Response, String> {
    let photo = core.photo(id)?;
    let opts = Opts { skip_crop, overlay_mask, full_frame: full_frame.unwrap_or(false), ai_cache_dir: Some(core.ai_cache_dir(id)) };
    let src = if hires == Some(true) { core.full_image_with(&photo.path, &develop.raw)? } else { core.proxy_with(&photo.path, &develop.raw)? };
    let mut out = develop::render(&src, &develop, &opts);
    drop(src);
    let mut rgba = develop::to_rgba8(&out);
    let hist = develop::histogram(&rgba);
    let proof = proof.unwrap_or_default();
    if proof.gamut.is_some() || proof.clipping || proof.spots.is_some() {
        if let Some(t) = proof.spots {
            develop::visualize_spots(&mut out, t);
        }
        if let Some(prof) = proof.gamut.as_deref().filter(|g| !g.is_empty()) {
            let use_lcms_proof = proof.simulate_paper.unwrap_or(false)
                || proof.intent.is_some()
                || matches!(prof, "ProPhotoRGB" | "ACEScg" | "FineArtMatte" | "BarytaGloss")
                || prof.to_ascii_lowercase().ends_with(".icc")
                || prof.to_ascii_lowercase().ends_with(".icm");
            if use_lcms_proof {
                let custom_dir = core.dir();
                icc::soft_proof(
                    &mut out.px,
                    Some(prof),
                    proof.intent.as_deref().unwrap_or("perceptual"),
                    proof.bpc.unwrap_or(true),
                    proof.simulate_paper.unwrap_or(true),
                    proof.gamut_warn.unwrap_or(true),
                    proof.clipping,
                    Some(&custom_dir),
                );
            } else {
                develop::proof(&mut out, Some(prof), proof.clipping);
            }
        } else if proof.clipping {
            develop::proof(&mut out, None, true);
        }
        rgba = develop::to_rgba8(&out);
    }
    let mut buf = Vec::with_capacity(8 + 4096 + rgba.len());
    buf.extend((out.w as u32).to_le_bytes());
    buf.extend((out.h as u32).to_le_bytes());
    for ch in hist {
        for v in ch {
            buf.extend(v.to_le_bytes());
        }
    }
    buf.extend(rgba);
    Ok(tauri::ipc::Response::new(buf))
}

/// Inspects a photo's native LibRaw sensor calibration (optical black levels, white point, CFA pattern)
/// and vendor MakerNotes (active AF points, Eye-AF box, mechanical shutter count, lens/body serials).
#[tauri::command(async)]
fn raw_info(core: St, id: i64) -> Result<libraw::RawInfo, String> {
    let p = core.photo(id)?;
    let local = core.local_file(&p.path)?;
    let bytes = fs::read(&local).map_err(es)?;
    libraw::inspect(&bytes)
}

/// Computes Real-Time RGB Parade & Luma/Chroma Waveform Scopes (binary reply: u32 scope_w, u32 scope_h,
/// followed by parade RGBA8 and waveform RGBA8). Uses the AOT SPIR-V GPU compute shader when enabled,
/// falling back to the multithreaded C++ scope kernel.
#[tauri::command(async)]
fn render_scopes(core: St, id: i64, develop: Develop, scope_w: Option<u32>, scope_h: Option<u32>) -> Result<tauri::ipc::Response, String> {
    let photo = core.photo(id)?;
    let opts = Opts { skip_crop: false, overlay_mask: None, full_frame: false, ai_cache_dir: Some(core.ai_cache_dir(id)) };
    let src = core.proxy_with(&photo.path, &develop.raw)?;
    let out = develop::render(&src, &develop, &opts);
    drop(src);
    let rgba = develop::to_rgba8(&out);
    let sw = scope_w.unwrap_or(288).clamp(64, 768);
    let sh = scope_h.unwrap_or(120).clamp(48, 512);
    let (parade, waveform) = gpu::run_scopes(&rgba, out.w as u32, out.h as u32, sw, sh)
        .unwrap_or_else(|| libraw::compute_scopes_cpu(&rgba, out.w, out.h, sw as usize, sh as usize));
    let mut buf = Vec::with_capacity(8 + parade.len() + waveform.len());
    buf.extend(sw.to_le_bytes());
    buf.extend(sh.to_le_bytes());
    buf.extend(parade);
    buf.extend(waveform);
    Ok(tauri::ipc::Response::new(buf))
}

/// Lists built-in wide-gamut / soft-proofing ICC profiles and user-imported `.icc` / `.icm` profiles.
#[tauri::command]
fn icc_profiles(core: St) -> Vec<icc::IccProfileEntry> {
    icc::list_profiles(&core.dir())
}

/// Imports a custom printer/paper/display `.icc` or `.icm` profile into `<data_dir>/icc_profiles/`.
#[tauri::command(async)]
fn icc_import_profile(core: St, path: String) -> Result<icc::IccProfileEntry, String> {
    icc::import_profile(&core.dir(), Path::new(&path))
}

/// Bidirectional Adobe Lightroom `.xmp` sidecar sync (`mode`: `"read"` | `"write"`).
#[tauri::command(async)]
fn xmp_sync(core: St, ids: Vec<i64>, mode: String, copyright: Option<String>, creator: Option<String>) -> Result<Value, String> {
    let photos = catalog::photos(&core.db(), &ids)?;
    let mut count = 0usize;
    for p in &photos {
        let path = Path::new(&p.path);
        if !path.exists() {
            continue;
        }
        if mode == "read" {
            let cur_dev = core.develop(p.id).unwrap_or_default();
            let Some(side) = xmp::read_adobe_xmp(path, Some(&cur_dev)) else { continue };
            if let Some(mut cull) = xmp::to_culling(&side) {
                if !side.keywords.is_empty() {
                    cull.keywords = Some(catalog::merge_keywords(&p.keywords, &side.keywords));
                }
                core.cull(&[p.id], &cull)?;
            }
            if side.has_develop_edits {
                core.commit(p.id, &side.develop, "Import XMP sidecar")?;
            }
            count += 1;
        } else {
            let dev = core.develop(p.id).unwrap_or_default();
            xmp::write_adobe_xmp(p, &dev, copyright.as_deref(), creator.as_deref())?;
            count += 1;
        }
    }
    if mode == "read" && count > 0 {
        core.emit("catalog-changed", json!({}));
    }
    Ok(json!({ "synced": count, "mode": mode }))
}

/// Lossless JPEG DCT block transform (`rot90`, `rot180`, `rot270`, `hflip`, `vflip`, `transpose`, `transverse`, `crop`).
#[tauri::command(async)]
fn jpeg_lossless(core: St, id: i64, op: String, crop: Option<[i32; 4]>) -> Result<Value, String> {
    let p = core.photo(id)?;
    let path = Path::new(&p.path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext != "jpg" && ext != "jpeg" {
        return Err("Lossless DCT transform requires a JPEG (.jpg/.jpeg) original".into());
    }
    let op_str = match op.as_str() {
        "rotate90" | "rot90" | "90" => "rot90",
        "rotate180" | "rot180" | "180" => "rot180",
        "rotate270" | "rot270" | "270" => "rot270",
        "flip_h" | "hflip" | "horizontal" => "hflip",
        "flip_v" | "vflip" | "vertical" => "vflip",
        "transpose" => "transpose",
        "transverse" => "transverse",
        "crop" => "crop",
        _ => return Err(format!("Unsupported lossless JPEG operation '{op}'")),
    };
    let jpeg_bytes = fs::read(path).map_err(es)?;
    let transformed = libraw::jpeg_lossless_transform(&jpeg_bytes, op_str, crop)?;
    let tmp = path.with_extension("lossless_tmp.jpg");
    fs::write(&tmp, &transformed).map_err(es)?;
    fs::rename(&tmp, path).map_err(es)?;
    core.cache.lock().unwrap().clear();
    *core.full.lock().unwrap() = None;
    let dev = core.develop(id).unwrap_or_default();
    let _ = core.refresh(id, &dev);
    core.emit("catalog-changed", json!({}));
    Ok(json!({ "id": id, "op": op, "path": p.path }))
}

/// Runs native C++ Focus-Peaking & Blink analysis on photo `id` and returns the focus score,
/// eye openness (EAR), blink flag, and an RGBA8 focus-peaking overlay.
#[tauri::command(async)]
fn focus_peaking(core: St, id: i64, threshold: Option<f32>) -> Result<Value, String> {
    let p = core.photo(id)?;
    let d = core.develop(id).unwrap_or_default();
    let opts = Opts { skip_crop: false, overlay_mask: None, full_frame: false, ai_cache_dir: Some(core.ai_cache_dir(id)) };
    let src = core.proxy_with(&p.path, &d.raw)?;
    let out = develop::render(&src, &d, &opts);
    drop(src);
    let rgba = develop::to_rgba8(&out);
    let flat_rgb: Vec<f32> = rgba
        .chunks_exact(4)
        .flat_map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();
    let (mask, score, ear, blink) = libraw::focus_peaking_and_blink(&flat_rgb, out.w, out.h, threshold.unwrap_or(0.22), &[]);
    let mut overlay = vec![0u8; out.w * out.h * 4];
    for (i, &m) in mask.iter().enumerate() {
        if m > 16 {
            overlay[i * 4] = 0;
            overlay[i * 4 + 1] = 255;
            overlay[i * 4 + 2] = 180;
            overlay[i * 4 + 3] = m;
        }
    }
    let png_img = image::RgbaImage::from_raw(out.w as u32, out.h as u32, overlay).ok_or("failed to build peaking image")?;
    let mut png_bytes = Vec::new();
    image::DynamicImage::ImageRgba8(png_img)
        .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(es)?;
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    Ok(json!({
        "focus_score": (score * 10.0).round() / 10.0,
        "eye_openness": (ear * 100.0).round() / 100.0,
        "blink_detected": blink,
        "overlay_data_url": format!("data:image/png;base64,{b64}"),
    }))
}

/// Generates a 16-bit feathered semantic mask (`subject`, `sky`, or `background`) using the native C++
/// He et al. Guided Filter and returns metadata + 16-bit PNG data URL.
#[tauri::command(async)]
fn semantic_mask_16bit(core: St, id: i64, target: String, feather: Option<f32>) -> Result<Value, String> {
    let p = core.photo(id)?;
    let proxy = core.proxy(&p.path)?;
    let rgb = vision::Rgb::from_working(&proxy, 512);
    let mask16 = vision::semantic_mask_16bit(&rgb, &target, feather.unwrap_or(20.0))
        .ok_or_else(|| format!("Could not detect {target} region in photo"))?;
    let coverage = mask16.iter().filter(|&&v| v > 32768).count() as f32 / mask16.len().max(1) as f32 * 100.0;
    Ok(json!({
        "id": id,
        "target": target,
        "width": rgb.w,
        "height": rgb.h,
        "bit_depth": 16,
        "coverage_pct": (coverage * 10.0).round() / 10.0,
    }))
}

/// Upright: the Transform that `mode` (off | auto | level | vertical | full | guided) gives `develop`.
#[tauri::command(async)]
fn upright(core: St, id: i64, develop: Develop, mode: String) -> Result<develop::Transform, String> {
    develop::upright(&*core.proxy(&core.photo(id)?.path)?, &develop, &mode)
}

/// AI Auto Align using PerspectiveFields (`mode`: `horizontal` | `vertical` | `both`).
#[tauri::command(async)]
fn ai_align(core: St, id: i64, develop: Develop, model: String, mode: String) -> Result<perspective::AlignResult, String> {
    let p = core.photo(id)?;
    let job = crate::auto::Job::start(&core, "ai-align");
    let proxy = core.proxy(&p.path)?;
    perspective::align(&core.dir(), &model, &proxy, &develop, &mode, &|label, done, total| {
        job.progress(label, done, total);
    })
}

/// Lateral chromatic aberration measured on the photo: [ca_red_cyan, ca_blue_yellow].
#[tauri::command(async)]
fn auto_ca(core: St, id: i64) -> Result<[f32; 2], String> {
    Ok(develop::estimate_ca(&*core.proxy(&core.photo(id)?.path)?))
}

/// Spot removal: heals a circle of `radius` (fraction of the long edge) at (x, y) of the full frame
/// from a matching nearby area; one history step.
#[tauri::command(async)]
fn heal_spot(core: St, id: i64, x: f32, y: f32, radius: f32) -> Result<Develop, String> {
    if !x.is_finite() || !y.is_finite() || !radius.is_finite() || radius <= 0.0 || radius > 1.0 {
        return Err("invalid spot coordinates or radius".to_string());
    }
    let img = core.proxy(&core.photo(id)?.path)?;
    let n = img.w.max(img.h) as f32;
    let (rx, ry) = (radius * n / img.w as f32, radius * n / img.h as f32);
    let points: Vec<[f32; 2]> = (0..24)
        .map(|k| {
            let a = k as f32 / 24.0 * std::f32::consts::TAU;
            [(x + rx * a.cos()).clamp(0.0, 1.0), (y + ry * a.sin()).clamp(0.0, 1.0)]
        })
        .collect();
    let mut d = core.develop(id)?;
    mcp::remove_objects(&core, id, &mut d, &json!([{ "name": "Spot", "points": points }]))?;
    d.clamp();
    core.commit(id, &d, "Spot removal")?;
    Ok(d)
}

/// Auto fine tune to a profile (develop::PROFILES): white balance and tone measured on the photo plus the look.
#[tauri::command(async)]
fn auto_develop(core: St, id: i64, develop: Develop, profile: String, strength: Option<f32>) -> Result<Develop, String> {
    develop::auto_with_strength(&*core.proxy(&core.photo(id)?.path)?, &develop, &profile, strength.unwrap_or(1.0))
}

/// One-click AI auto adjustment: the local retouch model's result as sliders, with `profile`'s look on top when given.
#[tauri::command(async)]
fn ai_adjust(core: St, id: i64, develop: Develop, profile: Option<String>, strength: Option<f32>, amount: Option<f32>) -> Result<Develop, String> {
    retouch::adjust(&*core.proxy(&core.photo(id)?.path)?, &develop, profile.as_deref(), strength.unwrap_or(1.0), amount.unwrap_or(1.0))
}

/// The profile for what photo `id` shows (auto::photo_type_of), asking the active AI provider; with the
/// AI's error when it could not, and the type came from the photo's keywords and measurements alone.
#[tauri::command(async)]
fn photo_type(core: St, id: i64) -> Result<(String, Option<String>), String> {
    let provider = core.settings.lock().unwrap().active_provider.clone();
    let (t, err) = auto::photo_type_of(core.inner(), &core.photo(id)?, &provider, true);
    Ok((t.into(), err))
}

#[tauri::command(async)]
fn auto_white_balance(core: St, id: i64) -> Result<develop::WhiteBalance, String> {
    Ok(develop::auto_wb(&*core.proxy(&core.photo(id)?.path)?))
}

/// Auto tone under the white balance of `develop`.
#[tauri::command(async)]
fn auto_tone(core: St, id: i64, develop: Develop) -> Result<develop::BasicTone, String> {
    Ok(develop::auto_tone(&*core.proxy(&core.photo(id)?.path)?, &develop.white_balance))
}

/// Auto HDR toning parameters for the photo proxy.
#[tauri::command(async)]
fn auto_hdr(core: St, id: i64) -> Result<develop::Hdr, String> {
    Ok(develop::auto_hdr(&*core.proxy(&core.photo(id)?.path)?))
}

/// Removes the objects in selection `mask` (its AI outlines), each healed from a matching nearby area,
/// and drops the selection; one history step.
#[tauri::command(async)]
fn remove_selection(core: St, id: i64, mask: usize) -> Result<Develop, String> {
    let mut d = core.develop(id)?;
    let m = d.masks.get(mask).ok_or("that selection no longer exists")?;
    let name = m.name.clone();
    let objs: Vec<Value> = m
        .components
        .iter()
        .filter(|c| c.kind == "polygon" && c.op == "add")
        .map(|c| json!({ "name": if c.name.is_empty() { &name } else { &c.name }, "points": c.points }))
        .collect();
    if objs.is_empty() {
        return Err(format!("'{name}' has no AI-selected object to remove: pick one with ✦ Select"));
    }
    mcp::remove_objects(&core, id, &mut d, &json!(objs))?;
    if mask < d.masks.len() {
        d.masks.remove(mask);
    }
    d.clamp();
    core.commit(id, &d, &format!("Remove {name}"))?;
    Ok(d)
}

#[tauri::command(async)]
fn save_develop(
    core: St,
    id: i64,
    develop: Develop,
    label: String,
    replace_prefix: Option<String>,
) -> Result<Develop, String> {
    let mut d = develop;
    d.clamp();
    core.commit_with_replace(id, &d, &label, replace_prefix.as_deref())?;
    Ok(d)
}

#[tauri::command(async)]
fn get_history(core: St, id: i64) -> Result<Vec<Value>, String> {
    catalog::history(&core.db(), id)
}

#[tauri::command(async)]
fn goto_history(core: St, id: i64, hid: i64) -> Result<Develop, String> {
    let d = catalog::goto(&core.db(), id, hid)?;
    core.refresh(id, &d)?;
    Ok(d)
}

#[tauri::command(async)]
fn delete_history(core: St, id: i64, hid: i64) -> Result<Option<Develop>, String> {
    let d = catalog::delete_history_step(&core.db(), id, hid)?;
    if let Some(d) = &d {
        core.refresh(id, d)?;
    }
    core.emit("catalog-changed", json!({}));
    Ok(d)
}

#[tauri::command(async)]
fn undo_redo(core: St, id: i64, dir: i32) -> Result<Option<Develop>, String> {
    let d = catalog::step(&core.db(), id, dir)?;
    if let Some(d) = &d {
        core.refresh(id, d)?;
    }
    Ok(d)
}

#[derive(Deserialize)]
struct ExportReq {
    ids: Vec<i64>,
    folder: String,
    format: String,
    color_space: String,
    quality: u8,
    max_edge: Option<u32>,
    /// `immich` | `google` | `cloud:<account id>`: upload the renders instead of keeping them in `folder`.
    upload: Option<String>,
    /// cloud: the folder in the account to upload into.
    #[serde(default)]
    upload_path: String,
}

#[tauri::command(async)]
fn export_photos(core: St, req: ExportReq) -> Result<Vec<String>, String> {
    let export = |folder: &Path| core.export(&req.ids, folder, &req.format, &req.color_space, req.quality.clamp(1, 100), req.max_edge);
    let Some(dest) = &req.upload else { return export(Path::new(&req.folder)) };
    let tmp = std::env::temp_dir().join(format!("optixel-upload-{}", new_token()));
    let res = export(&tmp).and_then(|files| match dest.strip_prefix("cloud:") {
        Some(id) => cloud::upload_files(&core, id, &req.upload_path, &files),
        None => remote::upload(&core, dest, &req.ids, &files),
    });
    let _ = fs::remove_dir_all(&tmp);
    res
}

#[tauri::command]
fn get_settings(core: St) -> Settings {
    core.settings.lock().unwrap().clone()
}

#[tauri::command]
fn save_settings(core: St, mut settings: Settings) -> Result<(), String> {
    {
        let mut s = core.settings.lock().unwrap();
        // drives and cloud accounts change through their own commands (and downloads add folders in the
        // background): the UI's copy of them may be out of date
        settings.network_drives = std::mem::take(&mut s.network_drives);
        settings.clouds = std::mem::take(&mut s.clouds);
        *s = settings;
    }
    core.save_settings()
}

/// This machine's GPU and the AI denoise models, with the best one for it marked.
#[tauri::command(async)]
fn ai_models(core: St) -> Value {
    let hw = aimodel::hardware();
    let models = aimodel::models(&core.dir(), &hw);
    json!({ "hardware": hw, "models": models })
}

/// Download an AI model (progress arrives as `ai-model-progress`).
#[tauri::command(async)]
fn ai_download(core: St, id: String) -> Result<(), String> {
    let c = core.inner().clone();
    let model = id.clone();
    aimodel::download(&core.dir(), &id, &|done, total| c.emit("ai-model-progress", json!({ "id": model, "done": done, "total": total })))
}

/// Snaps brush strokes painted over an object to the object, for the confirm step. Reply: u32 frame width and
/// height of the preview, u32 window x, y, width, height, then the window's selection as u8 (0-255).
#[tauri::command(async)]
fn snap_object(core: St, id: i64, strokes: Vec<brush::Stroke>) -> Result<tauri::ipc::Response, String> {
    let photo = core.photo(id)?;
    let img = core.proxy(&photo.path)?;
    let a = brush::snap(&img, &strokes)?;
    let (w, h) = (img.w, img.h);
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if a[y * w + x] > 0.01 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    let (ww, wh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut buf = Vec::with_capacity(24 + ww * wh);
    for v in [w, h, x0, y0, ww, wh] {
        buf.extend((v as u32).to_le_bytes());
    }
    for y in y0..y1 {
        buf.extend(a[y * w + x0..y * w + x1].iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    }
    Ok(tauri::ipc::Response::new(buf))
}

#[tauri::command(async)]
fn ai_remove(core: St, id: String) -> Result<(), String> {
    aimodel::remove(&core.dir(), &id)
}

#[tauri::command(async)]
fn ai_inpaint_inplace(
    core: St,
    id: i64,
    model: String,
    strokes: Vec<brush::Stroke>,
    grow: usize,
    prompt: String,
) -> Result<Develop, String> {
    if strokes.is_empty() {
        return Err("no strokes drawn to remove".into());
    }
    let p = core.photo(id)?;
    let job = crate::auto::Job::start(&core, "ai-remove");
    let file = core.local_file(&p.path)?;
    let img = core.decode(&file, None)?;
    let proxy = core.proxy(&p.path)?;
    let alpha = brush::snap(&proxy, &strokes)?;
    let hole = aimodel::hole_from_alpha(&alpha, proxy.w, proxy.h, img.w, img.h).ok_or("nothing is selected to remove")?;

    job.progress("Preparing hole & context", 0, 25);
    let patch = aimodel::inpaint_patch(&core.dir(), &model, &img, &hole, grow.min(40), &prompt, &|done, total| {
        let label = if total > 3 {
            format!("Diffusion step {} of {}", (done + 1).min(total), total)
        } else if total == 3 {
            match done {
                1 => "Extracting hole context".to_string(),
                2 => "Inpainting with neural network".to_string(),
                _ => "Feathering & blending patch".to_string(),
            }
        } else {
            "Removing object".to_string()
        };
        job.progress(&label, done, total);
        !job.cancelled()
    })?;

    let removal_id = new_token()[..12].to_string();
    let patch_filename = format!("patch_{removal_id}.bin");
    let ai_dir = core.ai_cache_dir(id);
    patch.save(&ai_dir.join(&patch_filename))?;

    let mut d = core.develop(id)?;
    let removal_label = if prompt.trim().is_empty() {
        format!("Removal {}", d.ai_removals.len() + 1)
    } else {
        format!("Fill: {}", prompt.trim())
    };
    let bounds = [
        patch.sx as f32 / img.w as f32,
        patch.sy as f32 / img.h as f32,
        patch.side as f32 / img.w as f32,
        patch.side as f32 / img.h as f32,
    ];
    d.ai_removals.push(develop::AiRemoval {
        id: removal_id,
        enabled: true,
        model: model.clone(),
        label: removal_label,
        prompt: prompt.clone(),
        bounds,
        patch_file: patch_filename,
    });
    core.commit(id, &d, "AI: Remove object")?;
    Ok(d)
}

/// Put the biggest face of library photo `source` onto every face of photo `id`: one layer per face, in the
/// same list as the object removals.
#[tauri::command(async)]
fn ai_face_swap(core: St, id: i64, model: String, source: i64) -> Result<Develop, String> {
    let p = core.photo(id)?;
    let s = core.photo(source)?;
    let job = crate::auto::Job::start(&core, "ai-faceswap");
    let src = core.decode(&core.local_file(&s.path)?, Some(2048))?;
    let img = core.decode(&core.local_file(&p.path)?, None)?;
    if img.w == 0 || img.h == 0 {
        return Err("target photo has invalid dimensions".into());
    }
    let patches = aimodel::face_swap(&core.dir(), &model, &img, &src, &|label, done, total| {
        job.progress(label, done, total);
        !job.cancelled()
    })?;
    let name = Path::new(&s.path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let mut d = core.develop(id)?;
    let ai_dir = core.ai_cache_dir(id);
    for patch in patches {
        let layer = new_token()[..12].to_string();
        let patch_file = format!("patch_{layer}.bin");
        patch.save(&ai_dir.join(&patch_file))?;
        let bounds = [patch.sx as f32 / img.w as f32, patch.sy as f32 / img.h as f32, patch.side as f32 / img.w as f32, patch.side as f32 / img.h as f32];
        d.ai_removals.push(develop::AiRemoval { id: layer, enabled: true, model: model.clone(), label: format!("Face swap: {name}"), prompt: String::new(), bounds, patch_file });
    }
    core.commit(id, &d, "AI: Face swap")?;
    Ok(d)
}

#[tauri::command(async)]
fn ai_denoise_inplace(
    core: St,
    id: i64,
    model: String,
    strength: f32,
) -> Result<Develop, String> {
    let p = core.photo(id)?;
    let job = crate::auto::Job::start(&core, "ai-denoise");
    let file = core.local_file(&p.path)?;
    let img = core.decode(&file, None)?;
    let vram = aimodel::hardware().vram_mb;

    job.progress("Starting AI denoise", 0, 1);
    let denoised_img = aimodel::denoise(&core.dir(), &model, img, 100.0, vram, &|done, total| {
        let label = if total > 1 {
            format!("Denoising tile {} of {}", (done + 1).min(total), total)
        } else {
            "Denoising photo".to_string()
        };
        job.progress(&label, done + 1, total);
        !job.cancelled()
    })?;

    let cache = develop::DenoisedCache {
        w: denoised_img.w,
        h: denoised_img.h,
        px: denoised_img.px,
    };
    let cache_file = core.ai_cache_dir(id).join(format!("denoise_{model}.bin"));
    cache.save(&cache_file)?;

    let mut d = core.develop(id)?;
    d.ai_denoise = develop::AiDenoise {
        enabled: true,
        model: model.clone(),
        strength,
    };
    core.commit(id, &d, "AI: Denoise")?;
    Ok(d)
}

#[tauri::command(async)]
fn ai_delete_removal(core: St, id: i64, removal_id: String) -> Result<Develop, String> {
    let mut d = core.develop(id)?;
    if let Some(pos) = d.ai_removals.iter().position(|r| r.id == removal_id) {
        let r = d.ai_removals.remove(pos);
        if let Some(file_name) = Path::new(&r.patch_file).file_name() {
            let patch_path = core.ai_cache_dir(id).join(file_name);
            let _ = std::fs::remove_file(patch_path);
        }
        core.commit(id, &d, "Delete AI removal")?;
    }
    Ok(d)
}

/// Detected GPU adapter name, or null when none is usable (the Settings toggle then has no effect).
#[tauri::command]
fn gpu_info() -> Option<String> {
    gpu::detect()
}

#[tauri::command]
fn regenerate_token(core: St) -> Result<String, String> {
    let t = new_token();
    core.settings.lock().unwrap().mcp_token = t.clone();
    core.save_settings()?;
    Ok(t)
}

#[tauri::command]
fn set_provider_key(core: St, id: String, key: String) -> Result<(), String> {
    ai::set_key(&id, &key)?;
    if let Some(p) = core.settings.lock().unwrap().providers.iter_mut().find(|p| p.id == id) {
        p.has_key = !key.is_empty();
    }
    core.save_settings()
}

#[tauri::command(async)]
fn ai_run(core: St, id: i64, prompt: String, provider: String) -> Result<String, String> {
    ai::run(core.inner(), id, &prompt, &provider)
}

/// AI selection: the object at (x, y) of the full frame joins mask `mask` (or a new one).
#[tauri::command(async)]
fn ai_select(core: St, id: i64, x: f32, y: f32, provider: String, mask: Option<usize>, subtract: bool) -> Result<Value, String> {
    if !x.is_finite() || !y.is_finite() || !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        return Err("invalid selection coordinates".to_string());
    }
    let (develop, mask) = ai::select(core.inner(), id, [x, y], &provider, mask, subtract)?;
    Ok(json!({ "develop": develop, "mask": mask }))
}

#[tauri::command(async)]
fn immich_connect(core: St, url: String, key: String) -> Result<Settings, String> {
    remote::immich_connect(&core, &url, &key)
}

#[tauri::command(async)]
fn immich_albums(core: St) -> Result<Vec<Value>, String> {
    remote::immich_albums(&core)
}

/// Browse an album (None: the favorites) in place; returns the catalog ids to show.
#[tauri::command(async)]
fn immich_open(core: St, album: Option<String>, page: Option<u32>) -> Result<Value, String> {
    remote::immich_open(core.inner(), album.as_deref(), page.unwrap_or(1))
}

#[tauri::command(async)]
fn google_connect(core: St, client_id: String, client_secret: String) -> Result<Settings, String> {
    remote::google_connect(&core, &client_id, &client_secret)
}

#[tauri::command(async)]
fn google_import(core: St) -> Result<Value, String> {
    remote::google_import(core.inner())
}

#[tauri::command]
fn apple_detect() -> Value {
    remote::apple_detect()
}

#[tauri::command(async)]
fn apple_connect(core: St, path: String) -> Result<Settings, String> {
    remote::apple_connect(&core, &path)
}

#[tauri::command(async)]
fn apple_import(core: St) -> Result<Value, String> {
    remote::apple_import(core.inner())
}

#[tauri::command]
fn amazon_detect() -> Value {
    remote::amazon_detect()
}

#[tauri::command(async)]
fn amazon_connect(core: St, path: String, account_name: Option<String>) -> Result<Settings, String> {
    remote::amazon_connect(&core, &path, account_name.as_deref().unwrap_or(""))
}

#[tauri::command(async)]
fn amazon_import(core: St) -> Result<Value, String> {
    remote::amazon_import(core.inner())
}

#[tauri::command]
fn adobe_detect() -> Value {
    remote::adobe_detect()
}

#[tauri::command(async)]
fn adobe_connect(core: St, path: String, account_name: Option<String>) -> Result<Settings, String> {
    remote::adobe_connect(&core, &path, account_name.as_deref().unwrap_or(""))
}

#[tauri::command(async)]
fn adobe_import(core: St) -> Result<Value, String> {
    remote::adobe_import(core.inner())
}

#[tauri::command]
fn remote_disconnect(core: St, source: String) -> Result<Settings, String> {
    remote::disconnect(&core, &source)
}

/// Run an auto function (auto::FUNCTIONS) on photos with its settings; `job` names it for cancel_job.
#[tauri::command(async)]
fn auto_run(core: St, function: String, ids: Vec<i64>, settings: Value, job: String) -> Result<Value, String> {
    let job = auto::Job::start(&core, &job);
    auto::run(core.inner(), &function, &ids, &settings, &job)
}

/// The auto functions' default settings (for Reset).
#[tauri::command]
fn auto_defaults() -> auto::AutoSettings {
    Default::default()
}

#[tauri::command]
fn cancel_job(core: St, job: String) -> bool {
    auto::cancel(&core, &job)
}

#[tauri::command(async)]
fn storage_report(core: St) -> Result<Value, String> {
    storage::report(&core)
}

/// `thumbs` | `originals` | `immich_albums` | `cloud` | `missing` | `history` | `compact` | `all`
#[tauri::command(async)]
fn clean_up(core: St, what: String) -> Result<usize, String> {
    storage::clean(&core, &what)
}

#[tauri::command(async)]
fn validate_storage_location(core: St, target: String) -> Result<Value, String> {
    storage::validate_location(&core, &target)
}

#[tauri::command(async)]
fn move_storage_location(core: St, target: String, mode: String, cleanup_old: Option<bool>) -> Result<Value, String> {
    storage::switch_storage_location(&core, PathBuf::from(target), &mode, cleanup_old.unwrap_or(false))
}

#[tauri::command(async)]
fn reset_storage_location(core: St, mode: String, cleanup_old: Option<bool>) -> Result<Value, String> {
    storage::reset_storage_location(&core, &mode, cleanup_old.unwrap_or(false))
}

#[tauri::command(async)]
fn sync_library_folders(core: St, folders: Option<Vec<String>>) -> Result<usize, String> {
    storage::sync_folders(&core, folders)
}

#[tauri::command(async)]
fn regenerate_thumbnails(
    core: St,
    ids: Option<Vec<i64>>,
    folder: Option<String>,
) -> Result<usize, String> {
    catalog::regenerate_thumbnails(&core, ids, folder, &|done, total| {
        core.emit("regenerate-thumbs-progress", json!({"done": done, "total": total}));
    })
}

/// Reveal a file or folder in the OS native file manager (macOS Finder, Windows File Explorer, Linux file manager).
#[tauri::command(async)]
fn show_in_folder(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        if let Some(parent) = p.parent() {
            if parent.exists() {
                return open_native_folder(parent);
            }
        }
        return Err(format!("Path does not exist on disk: {path}"));
    }

    #[cfg(target_os = "macos")]
    {
        let mut cmd = std::process::Command::new("open");
        if p.is_file() {
            cmd.arg("-R").arg(&path);
        } else {
            cmd.arg(&path);
        }
        cmd.spawn().map_err(|e| format!("Could not open Finder: {e}"))?;
    }

    #[cfg(target_os = "windows")]
    {
        let mut cmd = std::process::Command::new("explorer");
        if p.is_file() {
            cmd.arg(format!("/select,{}", path.trim_matches('"')));
        } else {
            cmd.arg(path.trim_matches('"'));
        }
        cmd.spawn().map_err(|e| format!("Could not open File Explorer: {e}"))?;
    }

    #[cfg(target_os = "linux")]
    {
        let target = if p.is_file() {
            p.parent().unwrap_or(&p)
        } else {
            &p
        };
        std::process::Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map_err(|e| format!("Could not open File Manager: {e}"))?;
    }

    Ok(())
}

fn open_native_folder(p: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Could not open Finder: {e}"))?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Could not open File Explorer: {e}"))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Could not open File Manager: {e}"))?;
    }

    Ok(())
}

#[tauri::command]
fn network_add(core: St, drive: storage::NetworkDrive) -> Result<Settings, String> {
    storage::add_drive(&core, drive)
}

#[tauri::command]
fn network_remove(core: St, id: String) -> Result<Settings, String> {
    storage::remove_drive(&core, &id)
}

/// Where a share usually appears once connected (to fill in the folder).
#[tauri::command]
fn network_mount_point(url: String) -> Option<String> {
    storage::mount_point(&url)
}

/// Connect the share if needed, without reading it into the catalog; returns its folder.
#[tauri::command(async)]
fn network_connect(core: St, id: String) -> Result<String, String> {
    storage::connect_drive(&core, &id)
}

#[tauri::command(async)]
fn network_status(core: St) -> Vec<Value> {
    storage::drives_online(&core)
}

/// Connect the share if needed and pick up new photos, like opening a folder.
#[tauri::command(async)]
fn network_open(core: St, id: String) -> Result<Value, String> {
    storage::open_drive(core.inner(), &id)
}

/// WebDAV / S3 account with its password or secret key.
#[tauri::command(async)]
fn cloud_connect(core: St, account: cloud::Account, secret: String) -> Result<Settings, String> {
    cloud::connect(&core, account, &secret)
}

#[tauri::command]
fn cloud_dropbox_start(app_key: String) -> Result<(), String> {
    cloud::dropbox_start(&app_key)
}

#[tauri::command(async)]
fn cloud_dropbox_finish(core: St, account: cloud::Account, code: String) -> Result<Settings, String> {
    cloud::dropbox_finish(&core, account, &code)
}

#[tauri::command(async)]
fn cloud_onedrive_connect(core: St, account: cloud::Account) -> Result<Settings, String> {
    cloud::onedrive_connect(&core, account)
}

#[tauri::command]
fn cloud_disconnect(core: St, id: String) -> Result<Settings, String> {
    cloud::disconnect(&core, &id)
}

#[tauri::command(async)]
fn cloud_browse(core: St, id: String, path: String) -> Result<Vec<cloud::Entry>, String> {
    cloud::open(&core, &id)?.1.list(&path)
}

/// Download what's new in a cloud folder and show it in the catalog.
#[tauri::command(async)]
fn cloud_sync(core: St, id: String, path: String) -> Result<Value, String> {
    cloud::sync(core.inner(), &id, &path)
}

/// Where each downloaded cloud folder is kept: [{id, path, local}].
#[tauri::command]
fn cloud_local_dirs(core: St) -> Vec<Value> {
    cloud::local_dirs(&core)
}

#[tauri::command]
fn cloud_forget_folder(core: St, id: String, path: String) -> Result<Settings, String> {
    cloud::forget_folder(&core, &id, &path)
}

/// Copy the originals at `from` to `to`; `job` names it for cancel_job.
#[tauri::command(async)]
fn duplicate(core: St, from: storage::Place, to: storage::Place, options: storage::DuplicateOptions, job: String) -> Result<Value, String> {
    let job = auto::Job::start(&core, &job);
    storage::duplicate(core.inner(), &from, &to, &options, &job)
}

pub fn run() {
    let core = Arc::new(Core::open(data_dir()).expect("could not open the Optixel catalog"));
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .menu(app_menu)
        .setup(move |app| {
            let _ = core.app.set(app.handle().clone());
            app.asset_protocol_scope().allow_directory(core.dir().join("thumbs"), true)?;
            app.manage(core.clone());
            mcp::start_http(core.clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            query_photos,
            list_folders,
            import_folder,
            browse_folder,
            set_selection,
            update_photos,
            virtual_copy,
            save_as_new_photo,
            remove_photos,
            bin_count,
            move_to_bin,
            restore_from_bin,
            permanent_delete,
            empty_bin,
            get_develop,
            default_develop,
            render_preview,
            raw_info,
            render_scopes,
            icc_profiles,
            icc_import_profile,
            xmp_sync,
            jpeg_lossless,
            focus_peaking,
            semantic_mask_16bit,
            upright,
            ai_align,
            auto_ca,
            heal_spot,
            auto_develop,
            ai_adjust,
            photo_type,
            auto_white_balance,
            auto_tone,
            auto_hdr,
            remove_selection,
            save_develop,
            get_history,
            goto_history,
            delete_history,
            undo_redo,
            export_photos,
            get_settings,
            save_settings,
            gpu_info,
            ai_models,
            ai_download,
            ai_remove,
            ai_inpaint_inplace,
            ai_face_swap,
            ai_denoise_inplace,
            ai_delete_removal,
            snap_object,
            regenerate_token,
            set_provider_key,
            ai_run,
            ai_select,
            immich_connect,
            immich_albums,
            immich_open,
            google_connect,
            google_import,
            apple_detect,
            apple_connect,
            apple_import,
            amazon_detect,
            amazon_connect,
            amazon_import,
            adobe_detect,
            adobe_connect,
            adobe_import,
            remote_disconnect,
            storage_report,
            clean_up,
            validate_storage_location,
            move_storage_location,
            reset_storage_location,
            sync_library_folders,
            regenerate_thumbnails,
            show_in_folder,
            auto_run,
            auto_defaults,
            cancel_job,
            network_add,
            network_remove,
            network_mount_point,
            network_status,
            network_connect,
            network_open,
            cloud_connect,
            cloud_dropbox_start,
            cloud_dropbox_finish,
            cloud_onedrive_connect,
            cloud_disconnect,
            cloud_browse,
            cloud_sync,
            cloud_local_dirs,
            cloud_forget_folder,
            duplicate,
            faces::faces_scan,
            faces::faces_people,
            faces::faces_of_person,
            faces::faces_similar,
            faces::faces_stats,
            faces::faces_rename,
            faces::faces_hide,
            faces::faces_set_cover,
            faces::faces_merge,
            faces::faces_move,
            faces::faces_delete,
            faces::faces_delete_people,
            faces::faces_reset,
            updater::check_for_updates,
            updater::download_update,
            updater::install_update,
            updater::open_external_url,
            updater::verify_license_key,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// `optixel-studio --import <folder>`: headless import into the catalog.
pub fn run_import(folder: &str) -> Result<usize, String> {
    let core = Core::open(data_dir())?;
    Ok(catalog::import(&core, Path::new(folder), &|done, total| eprint!("\rthumbnails {done}/{total}"))?.len())
}

/// `optixel-studio --mcp-stdio`: stdio MCP transport (spec §4.1).
pub fn run_mcp_stdio() {
    mcp::run_stdio();
}

/// The default menu, with the About panel showing the app icon, a description, disclaimer and developer.
fn app_menu(handle: &tauri::AppHandle) -> tauri::Result<tauri::menu::Menu<tauri::Wry>> {
    use tauri::menu::{AboutMetadata, Menu, MenuItemKind, PredefinedMenuItem};
    let menu = Menu::default(handle)?;
    let about = AboutMetadata {
        name: Some("Optixel Studio".into()),
        version: Some(handle.package_info().version.to_string()),
        authors: Some(vec!["Ross Kan <ross.kan@gmail.com>".into()]),
        comments: Some("Non-destructive RAW processor and photo catalog with a built-in MCP server and a pluggable AI/agent router.".into()),
        copyright: Some("Developed by Ross Kan (ross.kan@gmail.com)".into()),
        credits: Some(
            "Non-destructive RAW processor and photo catalog with a built-in MCP server and a pluggable AI/agent router.\n\n\
             Disclaimer: provided \"as is\", without warranty of any kind. The developer is not liable for any loss of or damage to photos or data. \
             Always keep backups of your originals. Third-party services and AI providers are used at your own discretion."
                .into(),
        ),
        icon: handle.default_window_icon().cloned(),
        ..Default::default()
    };
    // macOS puts About first in the app submenu; elsewhere it sits in the Help submenu
    for item in menu.items()? {
        if let MenuItemKind::Submenu(sub) = item {
            if let Some(i) = sub.items()?.iter().position(|it| matches!(it, MenuItemKind::Predefined(p) if p.text().is_ok_and(|t| t.starts_with("About")))) {
                sub.remove_at(i)?;
                sub.insert(&PredefinedMenuItem::about(handle, Some("About Optixel Studio"), Some(about.clone()))?, i)?;
            }
        }
    }
    Ok(menu)
}
