//! Remote photo sources: a local Immich server and Google Photos.
//!
//! Originals are downloaded to `<data dir>/remote/<source>/<album>/` and imported like any
//! folder, so develop, sidecars and export work unchanged; exports can be uploaded back.
//! Immich authenticates with an API key. Google Photos goes through the Picker API (the
//! Library API no longer lists a user's photos, but still accepts uploads) with the user's
//! own OAuth "Desktop app" client. Secrets live in the OS keychain.

use crate::{
    Core, Settings, ai, catalog,
    develop::{self, es},
};
use base64::Engine;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const IMMICH_KEY: &str = "remote:immich";
/// Keychain entry holding `{client_secret, refresh_token}`.
const GOOGLE_KEY: &str = "remote:google-photos";
const GOOGLE_TOKEN: &str = "https://oauth2.googleapis.com/token";
const PICKER: &str = "https://photospicker.googleapis.com/v1";

const APPLE_KEY: &str = "remote:apple-photos";
const AMAZON_KEY: &str = "remote:amazon-photos";
const ADOBE_KEY: &str = "remote:adobe-photos";

type Reply = Result<ureq::http::Response<ureq::Body>, ureq::Error>;

/// No overall deadline (originals can be large); a stalled read gives up after a minute.
pub(crate) fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .into()
}

pub(crate) fn json_of(what: &str, reply: Reply) -> Result<Value, String> {
    let mut r = reply.map_err(|e| format!("{what}: {e}"))?;
    let status = r.status().as_u16();
    let body = r.body_mut().read_to_string().map_err(es)?;
    if status >= 400 {
        return Err(format!("{what} returned HTTP {status}: {}", body.chars().take(300).collect::<String>()));
    }
    if body.trim().is_empty() {
        return Ok(Value::Null); // 204 No Content
    }
    serde_json::from_str(&body).map_err(|_| format!("{what} did not reply with JSON; check the address"))
}

pub(crate) fn update_settings(core: &Core, f: impl FnOnce(&mut Settings)) -> Result<Settings, String> {
    let s = {
        let mut s = core.settings.lock().unwrap();
        f(&mut s);
        s.clone()
    };
    core.save_settings()?;
    Ok(s)
}

pub fn disconnect(core: &Core, source: &str) -> Result<Settings, String> {
    match source {
        "immich" => {
            ai::set_key(IMMICH_KEY, "")?;
            update_settings(core, |s| s.immich_url.clear())
        }
        "google" => {
            ai::set_key(GOOGLE_KEY, "")?;
            update_settings(core, |s| s.google_client_id.clear())
        }
        "apple" => {
            ai::set_key(APPLE_KEY, "")?;
            update_settings(core, |s| s.apple_photos_path.clear())
        }
        "amazon" => {
            ai::set_key(AMAZON_KEY, "")?;
            update_settings(core, |s| s.amazon_photos_path.clear())
        }
        "adobe" => {
            ai::set_key(ADOBE_KEY, "")?;
            update_settings(core, |s| s.adobe_cloud_path.clear())
        }
        _ => Err(format!("unknown source '{source}'")),
    }
}

// ---------------------------------------------------------------------------
// Immich
// ---------------------------------------------------------------------------

/// `host:2283`, `http://host:2283/`, `…/api` → `http://host:2283`.
fn immich_base(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    let u = u.strip_suffix("/api").unwrap_or(u);
    if u.contains("://") { u.into() } else { format!("http://{u}") }
}

fn immich_albums_at(url: &str, key: &str) -> Result<Vec<Value>, String> {
    let v = json_of("Immich", agent().get(&format!("{url}/api/albums")).header("x-api-key", key).call())?;
    let albums = v.as_array().ok_or("Immich did not return an album list")?;
    Ok(albums.iter().map(|a| json!({"id": a["id"], "name": a["albumName"], "count": a["assetCount"]})).collect())
}

pub(crate) fn immich(core: &Core) -> Result<(String, String), String> {
    let url = core.settings.lock().unwrap().immich_url.clone();
    let key = ai::key(IMMICH_KEY).filter(|_| !url.is_empty()).ok_or("Immich is not connected")?;
    Ok((url, key))
}

/// Checks the address and key (listing albums needs `album.read`), then stores both.
pub fn immich_connect(core: &Core, url: &str, key: &str) -> Result<Settings, String> {
    let url = immich_base(url);
    let parsed = tauri::Url::parse(&url).map_err(|e| format!("invalid Immich address: {e}"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err("Immich address must use http or https".into());
    }
    let key = key.trim();
    immich_albums_at(&url, key)?;
    ai::set_key(IMMICH_KEY, key)?;
    update_settings(core, |s| s.immich_url = url)
}

pub fn immich_albums(core: &Core) -> Result<Vec<Value>, String> {
    let (url, key) = immich(core)?;
    immich_albums_at(&url, &key)
}

/// Catalog path of an Immich photo: `immich://<asset id>`.
pub fn immich_id(path: &str) -> Option<&str> {
    path.strip_prefix("immich://")
}

/// Photos per page when browsing an album; the grid asks for the next page as you scroll.
const PAGE: u32 = 250;

/// Every image of an album (the favorites when `album` is None), page by page.
pub fn immich_assets(core: &Core, album: Option<&str>) -> Result<Vec<Value>, String> {
    let (url, key) = immich(core)?;
    let (mut out, mut page) = (vec![], Some(1));
    while let Some(p) = page {
        let (assets, next) = immich_page(&url, &key, album, p)?;
        out.extend(assets);
        page = next;
    }
    Ok(out)
}

/// One page of an album's images (the favorites when `album` is None), oldest first like the
/// grid, plus the next page number if there is one.
fn immich_page(url: &str, key: &str, album: Option<&str>, page: u32) -> Result<(Vec<Value>, Option<u32>), String> {
    // ponytail: pre-3.2 search fields (deprecated, still served by Immich 3.x) keep older servers working; move to `filter` + `cursor` once those are rare
    let mut q = json!({"type": "IMAGE", "size": PAGE, "page": page, "order": "asc", "withExif": true});
    match album {
        Some(id) => q["albumIds"] = json!([id]),
        None => q["isFavorite"] = json!(true),
    }
    let r = json_of("Immich", agent().post(&format!("{url}/api/search/metadata")).header("x-api-key", key).send_json(&q))?;
    let mut assets = r["assets"]["items"].as_array().cloned().unwrap_or_default();
    if album.is_none() {
        assets.retain(|a| a["isFavorite"] == true); // a server that ignored the filter would list everything
    }
    let next = r["assets"]["nextPage"].as_str().and_then(|p| p.parse().ok()).filter(|n| *n > page);
    Ok((assets, next))
}

/// Browse an album (None: the favorites) in place, a page at a time. Each asset becomes an
/// `immich://<id>` catalog row with the server's thumbnail; nothing else downloads until a
/// photo is opened or exported. Page 1 starts the album over, so photos removed in Immich
/// drop out. Returns the next page number, or null at the end. Photos new to the catalog can be culled
/// automatically (Settings › Storage).
pub fn immich_open(core: &Arc<Core>, album: Option<&str>, page: u32) -> Result<Value, String> {
    let (url, key) = immich(core)?;
    let mut r = open_from_immich(core, &url, &key, album, page)?;
    let added: Vec<i64> = serde_json::from_value(r["added"].take()).unwrap_or_default();
    crate::auto::after_import(core, added, "immich");
    Ok(r)
}

fn open_from_immich(core: &Core, url: &str, key: &str, album: Option<&str>, page: u32) -> Result<Value, String> {
    let (assets, next) = immich_page(url, key, album, page.max(1))?;
    let member = album.unwrap_or("favorites");
    let (mut need_thumb, mut added) = (vec![], vec![]);
    {
        let mut c = core.db();
        let tx = c.transaction().map_err(es)?;
        if page <= 1 {
            tx.execute("DELETE FROM immich_albums WHERE album = ?", [member]).map_err(es)?;
        }
        for a in &assets {
            let (Some(asset), Some(name)) = (a["id"].as_str(), a["originalFileName"].as_str()) else { continue };
            let (path, x) = (format!("immich://{asset}"), &a["exifInfo"]);
            let m = catalog::Meta {
                // localDateTime is the wall-clock capture time, as the catalog stores it
                captured_at: a["localDateTime"].as_str().and_then(|t| t.get(..19)).map(|t| t.replace('T', " ")),
                camera: catalog::camera_name(x["make"].as_str().map(String::from), x["model"].as_str().map(String::from)),
                lens: x["lensModel"].as_str().map(String::from),
                iso: x["iso"].as_i64(),
                aperture: x["fNumber"].as_f64(),
                shutter: x["exposureTime"].as_str().map(String::from),
                focal_length: x["focalLength"].as_f64(),
                orientation: 1,
                latitude: x["latitude"].as_f64().filter(|v| v.is_finite() && (-90.0..=90.0).contains(v)),
                longitude: x["longitude"].as_f64().filter(|v| v.is_finite() && (-180.0..=180.0).contains(v)),
            };
            if let Some(id) = catalog::add_photo(&tx, &path, "Immich", name, &m, &develop::Develop::default())? {
                let (w, h) = (a["width"].as_i64().or(x["exifImageWidth"].as_i64()), a["height"].as_i64().or(x["exifImageHeight"].as_i64()));
                tx.execute("UPDATE photos SET width = ?, height = ? WHERE id = ?", (w.unwrap_or(0), h.unwrap_or(0), id)).map_err(es)?;
                added.push(id);
            }
            // A master's stars and pick/reject follow Immich (changes made here are pushed as they
            // happen): rating 1-5 = stars, -1 = rejected (stars kept), favorite = pick.
            let rating = x["rating"].as_i64();
            tx.execute(
                "UPDATE photos SET rating = CASE WHEN ?1 = -1 THEN rating ELSE ?2 END,
                   flag = CASE WHEN ?1 = -1 THEN -1 WHEN ?3 THEN 1 ELSE 0 END
                 WHERE path = ?4 AND master_id IS NULL",
                (rating, rating.unwrap_or(0).clamp(0, 5), a["isFavorite"] == true, &path),
            )
            .map_err(es)?;
            tx.execute("INSERT OR IGNORE INTO immich_albums (album, path) VALUES (?, ?)", (member, &path)).map_err(es)?;
            let mut st = tx.prepare_cached("SELECT id FROM photos WHERE path = ? AND master_id IS NULL").map_err(es)?;
            for id in st.query_map([&path], |r| r.get::<_, i64>(0)).map_err(es)?.flatten() {
                if !core.thumb_path(id).exists() {
                    need_thumb.push((id, asset));
                }
            }
        }
        tx.commit().map_err(es)?;
    }
    let (agent, done) = (agent(), AtomicUsize::new(0));
    need_thumb.par_iter().for_each(|(id, asset)| {
        let res = (|| {
            let mut r = agent.get(&format!("{url}/api/assets/{asset}/thumbnail")).header("x-api-key", key).call().map_err(es)?;
            if r.status().as_u16() >= 400 {
                return Err(format!("HTTP {}", r.status().as_u16()));
            }
            let img = image::load_from_memory(&r.body_mut().read_to_vec().map_err(es)?).map_err(es)?;
            catalog::write_jpeg(&core.thumb_path(*id), &img)
        })();
        if let Err(e) = res {
            eprintln!("immich thumbnail {asset}: {e}");
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        core.emit("import-progress", json!({"label": "Loading thumbnails", "done": n, "total": need_thumb.len()}));
    });
    core.emit("catalog-changed", json!({}));
    Ok(json!({"next": next, "added": added}))
}

/// Mirror these photos' stars and pick/reject to Immich: stars → rating 1-5 (none → unrated),
/// reject → rating -1, pick → favorite. Masters only; virtual copies stay local.
pub fn immich_push(core: &Core, ids: &[i64]) -> Result<(), String> {
    push_to_immich(core, ids, || immich(core))
}

fn push_to_immich(core: &Core, ids: &[i64], creds: impl FnOnce() -> Result<(String, String), String>) -> Result<(), String> {
    // One push at a time, each reading the catalog as it runs, so the last one sends the latest state.
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let photos = catalog::query(&core.db(), &catalog::Filter { ids: Some(ids.to_vec()), ..Default::default() })?;
    let mut groups: HashMap<(Option<i64>, bool), Vec<&str>> = HashMap::new();
    for p in photos.iter().filter(|p| p.master_id.is_none()) {
        let Some(asset) = immich_id(&p.path) else { continue };
        let rating = if p.flag == -1 { Some(-1) } else { Some(p.rating).filter(|r| *r > 0) };
        groups.entry((rating, p.flag == 1)).or_default().push(asset);
    }
    if groups.is_empty() {
        return Ok(());
    }
    let (url, key) = creds()?;
    let agent = agent();
    for ((rating, favorite), assets) in groups {
        let body = json!({"ids": assets, "rating": rating, "isFavorite": favorite});
        json_of("Immich", agent.put(&format!("{url}/api/assets")).header("x-api-key", &key).send_json(&body))?;
    }
    Ok(())
}

/// The file an `immich://<id>` photo edits from, downloaded once: the original, or for
/// formats Optixel can't decode (HEIC…) Immich's full-size JPEG/WebP rendition.
pub fn immich_original(core: &Core, id: &str) -> Result<PathBuf, String> {
    original_from_immich(core, id, || immich(core))
}

/// Cache of opened Immich photos; clearing it only means downloading them again.
pub fn originals_dir(core: &Core) -> PathBuf {
    core.dir().join("remote").join("Immich").join("originals")
}

/// The downloaded file of an Immich photo, if it was opened before (`Item` names it).
fn cached_original(core: &Core, id: &str) -> Result<(Item, Option<PathBuf>), String> {
    let name: String = core
        .db()
        .query_row("SELECT filename FROM photos WHERE path = ? LIMIT 1", [format!("immich://{id}")], |r| r.get(0))
        .map_err(es)?;
    let it = Item { id: id.into(), name, url: String::new() };
    let stem = local_stem(&it);
    let found = fs::read_dir(originals_dir(core))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| develop::is_supported(p) && p.file_stem().is_some_and(|s| s == stem.as_str()));
    Ok((it, found))
}

/// An Immich photo's original if it is downloaded already; never downloads.
pub fn immich_cached(core: &Core, id: &str) -> Option<PathBuf> {
    cached_original(core, id).ok()?.1
}

/// An Immich photo's preview (about 1440 px) straight from the server: enough to analyse it without
/// downloading the original.
pub fn immich_preview(core: &Core, id: &str) -> Result<image::DynamicImage, String> {
    let (url, key) = immich(core)?;
    let mut r = agent().get(&format!("{url}/api/assets/{id}/thumbnail?size=preview")).header("x-api-key", &key).call().map_err(es)?;
    if r.status().as_u16() >= 400 {
        return Err(format!("Immich preview: HTTP {}", r.status().as_u16()));
    }
    image::load_from_memory(&r.body_mut().read_to_vec().map_err(es)?).map_err(es)
}

/// `creds` is only asked for on a cache miss, so downloaded photos open offline.
fn original_from_immich(core: &Core, id: &str, creds: impl FnOnce() -> Result<(String, String), String>) -> Result<PathBuf, String> {
    let (mut it, cached) = cached_original(core, id)?;
    if let Some(p) = cached {
        return Ok(p);
    }
    let (dir, stem) = (originals_dir(core), local_stem(&it));
    fs::create_dir_all(&dir).map_err(es)?;
    let (url, key) = creds()?;
    let what = if save_ext(&it.name, "").is_some() { "original" } else { "thumbnail?size=fullsize" };
    it.url = format!("{url}/api/assets/{id}/{what}");
    download(&agent(), &it, ("x-api-key", &key), &dir, &stem)?.ok_or_else(|| format!("{}: Immich sent a format Optixel can't open", it.name))
}

// ---------------------------------------------------------------------------
// Google Photos
// ---------------------------------------------------------------------------

pub(crate) fn open_browser(url: &str) -> Result<(), String> {
    let parsed = tauri::Url::parse(url).map_err(|e| format!("invalid URL ({e}): {url}"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(format!("refusing to open non-http/https URL: {url}"));
    }
    let (prog, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        ("open", &["--"])
    } else if cfg!(windows) {
        ("rundll32", &["url.dll,FileProtocolHandler"])
    } else {
        ("xdg-open", &["--"])
    };
    let mut child =
        Command::new(prog).args(args).arg(url).spawn().map_err(|e| format!("could not open a browser ({e}); visit {url}"))?;
    std::thread::spawn(move || child.wait());
    Ok(())
}

/// Sign in through the browser: loopback redirect + PKCE (RFC 8252). Stores the refresh token.
pub fn google_connect(core: &Core, client_id: &str, client_secret: &str) -> Result<Settings, String> {
    let (client_id, client_secret) = (client_id.trim(), client_secret.trim());
    let server = tiny_http::Server::http("127.0.0.1:0").map_err(es)?;
    let port = server.server_addr().to_ip().ok_or("no loopback port")?.port();
    let redirect = format!("http://127.0.0.1:{port}");
    let (verifier, state) = (crate::new_token(), crate::new_token());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes()));
    let auth = tauri::Url::parse_with_params(
        "https://accounts.google.com/o/oauth2/v2/auth",
        [
            ("client_id", client_id),
            ("redirect_uri", redirect.as_str()),
            ("response_type", "code"),
            (
                "scope",
                "https://www.googleapis.com/auth/photospicker.mediaitems.readonly https://www.googleapis.com/auth/photoslibrary.appendonly",
            ),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("state", state.as_str()),
            ("access_type", "offline"),
            ("prompt", "consent"),
        ],
    )
    .map_err(es)?;
    open_browser(auth.as_str())?;
    let code = wait_for_code(&server, &state, "Google Photos")?;
    let tok = json_of(
        "Google sign-in",
        agent().post(GOOGLE_TOKEN).send_form([
            ("code", code.as_str()),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("redirect_uri", redirect.as_str()),
            ("grant_type", "authorization_code"),
            ("code_verifier", verifier.as_str()),
        ]),
    )?;
    let refresh = tok["refresh_token"].as_str().ok_or("Google returned no refresh token")?;
    ai::set_key(GOOGLE_KEY, &json!({"client_secret": client_secret, "refresh_token": refresh}).to_string())?;
    update_settings(core, |s| s.google_client_id = client_id.into())
}

/// Serve the OAuth redirect on the loopback listener and return the authorization code.
pub(crate) fn wait_for_code(server: &tiny_http::Server, state: &str, service: &str) -> Result<String, String> {
    let deadline = Instant::now() + Duration::from_secs(300);
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        let Some(req) = server.recv_timeout(left).map_err(es)? else { break };
        let url = tauri::Url::parse(&format!("http://127.0.0.1{}", req.url())).map_err(es)?;
        let q: HashMap<String, String> = url.query_pairs().into_owned().collect();
        if q.get("state").map(String::as_str) != Some(state) {
            let _ = req.respond(tiny_http::Response::empty(404)); // favicon, stray or forged requests
            continue;
        }
        let res = match (q.get("code"), q.get("error")) {
            (Some(code), _) => Ok(code.clone()),
            (_, err) => Err(format!("{service} sign-in failed: {}", err.map_or("no code returned", String::as_str))),
        };
        let page = if res.is_ok() { format!("Optixel Studio is connected to {service}.") } else { format!("{service} sign-in did not complete.") };
        let ct = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/plain; charset=utf-8"[..]).expect("static header");
        let nosniff = tiny_http::Header::from_bytes(&b"X-Content-Type-Options"[..], &b"nosniff"[..]).expect("static header");
        let _ = req.respond(
            tiny_http::Response::from_string(format!("{page} You can close this tab."))
                .with_header(ct)
                .with_header(nosniff),
        );
        return res;
    }
    Err(format!("timed out waiting for {service} sign-in"))
}

pub(crate) fn google_access_token(core: &Core) -> Result<String, String> {
    let client_id = core.settings.lock().unwrap().google_client_id.clone();
    let saved: Value = ai::key(GOOGLE_KEY)
        .filter(|_| !client_id.is_empty())
        .and_then(|s| serde_json::from_str(&s).ok())
        .ok_or("Google Photos is not connected")?;
    let tok = json_of(
        "Google sign-in",
        agent().post(GOOGLE_TOKEN).send_form([
            ("client_id", client_id.as_str()),
            ("client_secret", saved["client_secret"].as_str().unwrap_or("")),
            ("refresh_token", saved["refresh_token"].as_str().unwrap_or("")),
            ("grant_type", "refresh_token"),
        ]),
    )?;
    tok["access_token"].as_str().map(String::from).ok_or_else(|| "Google returned no access token".into())
}

/// Open the Google Photos picker in the browser, wait for the user's choice, then
/// download and import the picked photos.
pub fn google_import(core: &Arc<Core>) -> Result<Value, String> {
    let bearer = format!("Bearer {}", google_access_token(core)?);
    let agent = agent();
    let session = json_of(
        "Google Photos",
        agent.post(&format!("{PICKER}/sessions")).header("Authorization", &bearer).send_json(json!({})),
    )?;
    let id = session["id"].as_str().ok_or("Google Photos did not start a picking session")?;
    let res = (|| {
        // `open` would also launch a local path, so only hand it a web link.
        let picker = session["pickerUri"].as_str().filter(|u| u.starts_with("https://")).ok_or("Google Photos returned no picker link")?;
        open_browser(&format!("{picker}/autoclose"))?;
        // Durations arrive as "3.5s"; clamp so a bad value cannot stall or panic.
        let secs = |v: &Value| v.as_str().and_then(|s| s.trim_end_matches('s').parse::<f64>().ok()).map(|s| s.max(0.0).min(7200.0));
        let deadline = Instant::now() + Duration::from_secs_f64(secs(&session["pollingConfig"]["timeoutIn"]).unwrap_or(1800.0));
        let mut s = session.clone();
        while s["mediaItemsSet"] != true {
            if Instant::now() > deadline {
                return Err("timed out waiting for photos to be picked in Google Photos".to_string());
            }
            std::thread::sleep(Duration::from_secs_f64(secs(&s["pollingConfig"]["pollInterval"]).unwrap_or(5.0).clamp(1.0, 30.0)));
            s = json_of("Google Photos", agent.get(&format!("{PICKER}/sessions/{id}")).header("Authorization", &bearer).call())?;
        }
        let (mut items, mut picked, mut token) = (vec![], 0, String::new());
        loop {
            let mut req = agent
                .get(&format!("{PICKER}/mediaItems"))
                .header("Authorization", &bearer)
                .query("sessionId", id)
                .query("pageSize", "100");
            if !token.is_empty() {
                req = req.query("pageToken", &token);
            }
            let page = json_of("Google Photos", req.call())?;
            for m in page["mediaItems"].as_array().into_iter().flatten() {
                picked += 1;
                if let (true, Some(mid), Some(base)) = (m["type"] == "PHOTO", m["id"].as_str(), m["mediaFile"]["baseUrl"].as_str()) {
                    let name = m["mediaFile"]["filename"].as_str().unwrap_or_default();
                    items.push(Item { id: mid.into(), name: name.into(), url: format!("{base}=d") });
                }
            }
            match page["nextPageToken"].as_str() {
                Some(t) if !t.is_empty() => token = t.into(),
                _ => break,
            }
        }
        fetch_and_import(core, &["Google Photos"], picked - items.len(), items, ("Authorization", &bearer))
    })();
    let _ = agent.delete(&format!("{PICKER}/sessions/{id}")).header("Authorization", &bearer).call();
    res
}

// ---------------------------------------------------------------------------
// Apple iCloud Photos, Amazon Photos, Adobe Cloud Photos
// ---------------------------------------------------------------------------

fn copy_to_service_export(base_path: &Path, default_dest: &Path, file: &str) -> Result<String, String> {
    let source_path = Path::new(file);
    let name = file_name(file);
    let export_dir = if base_path.as_os_str().is_empty() {
        default_dest.to_path_buf()
    } else if base_path.is_dir() {
        base_path.join("Optixel Exports")
    } else {
        base_path.parent().map(|p| p.join("Optixel Exports")).unwrap_or_else(|| default_dest.to_path_buf())
    };
    fs::create_dir_all(&export_dir).map_err(es)?;
    let dest_file = export_dir.join(&name);
    fs::copy(source_path, &dest_file).map_err(es)?;
    Ok(name)
}

fn scan_supported_photos(dir: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut results = Vec::new();
    scan_supported_photos_inner(dir, 0, max_depth, &mut results);
    results
}

fn scan_supported_photos_inner(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && name != "derivatives" && name != "resources" {
                scan_supported_photos_inner(&path, depth + 1, max_depth, out);
            }
        } else if develop::is_supported(&path) {
            out.push(path);
        }
    }
}

fn sync_local_photos_into(
    core: &Arc<Core>,
    service_label: &str,
    src_dir: &Path,
    dest_subfolder: &str,
) -> Result<Value, String> {
    let dest = core.dir().join("remote").join(dest_subfolder);
    fs::create_dir_all(&dest).map_err(es)?;

    let scan_root = if src_dir.join("originals").is_dir() {
        src_dir.join("originals")
    } else if src_dir.join("Masters").is_dir() {
        src_dir.join("Masters")
    } else {
        src_dir.to_path_buf()
    };

    let files = scan_supported_photos(&scan_root, 6);
    let total = files.len();
    let mut skipped_count = 0;
    let mut failed_count = 0;

    let have: HashSet<String> = fs::read_dir(&dest)
        .map_err(es)?
        .flatten()
        .filter_map(|e| e.path().file_name().map(|s| s.to_string_lossy().into_owned()))
        .collect();

    for (i, file) in files.iter().enumerate() {
        let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.is_empty() {
            skipped_count += 1;
            continue;
        }

        if i % 10 == 0 || i + 1 == total {
            core.emit("import-progress", json!({
                "label": format!("Syncing {service_label}"),
                "done": i + 1,
                "total": total
            }));
        }

        let mut target_file = dest.join(&name);
        if have.contains(&name) || target_file.exists() {
            if fs::canonicalize(&target_file).ok() == fs::canonicalize(file).ok() {
                skipped_count += 1;
                continue;
            }
            let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("photo");
            let ext_str = file.extension().and_then(|e| e.to_str()).unwrap_or("jpg");
            let hash = crate::new_token();
            let short_hash = &hash[..8];
            target_file = dest.join(format!("{stem}_{short_hash}.{ext_str}"));
            if target_file.exists() {
                skipped_count += 1;
                continue;
            }
        }

        #[cfg(unix)]
        let link_res = std::os::unix::fs::symlink(file, &target_file);
        #[cfg(windows)]
        let link_res = std::os::windows::fs::symlink_file(file, &target_file);

        if link_res.is_err() {
            if let Err(e) = fs::copy(file, &target_file) {
                eprintln!("{service_label} copy error {}: {e}", file.display());
                failed_count += 1;
                continue;
            }
        }
    }

    let ids = catalog::import(core, &dest, &|done, tot| {
        core.emit("import-progress", json!({"done": done, "total": tot}));
    })?;

    core.emit("catalog-changed", json!({}));
    crate::auto::after_import(core, ids.clone(), "cloud");

    Ok(json!({
        "folder": dest,
        "imported": ids.len(),
        "skipped": skipped_count,
        "failed": failed_count,
        "error": Value::Null,
    }))
}

pub fn apple_detect() -> Value {
    let mut candidates = Vec::new();
    if let Some(pic) = dirs::picture_dir() {
        candidates.push(pic.join("Photos Library.photoslibrary"));
        candidates.push(pic.join("iCloud Photos"));
        candidates.push(pic.join("iCloud Photos/Photos"));
    }
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join("Pictures/Photos Library.photoslibrary"));
        candidates.push(home.join("Library/Mobile Documents/com~apple~CloudDocs/Photos"));
        candidates.push(home.join("Pictures/iCloud Photos"));
    }
    for c in &candidates {
        if c.exists() {
            return json!({ "found": true, "default_path": c.to_string_lossy() });
        }
    }
    let fallback = dirs::picture_dir()
        .map(|p| p.join("Photos Library.photoslibrary"))
        .unwrap_or_default();
    json!({ "found": false, "default_path": fallback.to_string_lossy() })
}

pub fn apple_connect(core: &Core, path: &str) -> Result<Settings, String> {
    let p = Path::new(path.trim());
    if !path.trim().is_empty() && !p.exists() {
        return Err(format!("Path does not exist: {}", path));
    }
    ai::set_key(APPLE_KEY, path.trim())?;
    update_settings(core, |s| s.apple_photos_path = path.trim().into())
}

pub fn apple_import(core: &Arc<Core>) -> Result<Value, String> {
    let path = core.settings.lock().unwrap().apple_photos_path.clone();
    if path.is_empty() {
        return Err("Apple iCloud Photos is not connected".into());
    }
    sync_local_photos_into(core, "Apple iCloud Photos", Path::new(&path), "Apple iCloud Photos")
}

pub fn amazon_detect() -> Value {
    let mut candidates = Vec::new();
    if let Some(pic) = dirs::picture_dir() {
        candidates.push(pic.join("Amazon Photos"));
        candidates.push(pic.join("Amazon Drive/Pictures"));
        candidates.push(pic.join("Amazon Drive"));
    }
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join("Amazon Photos"));
        candidates.push(home.join("Pictures/Amazon Photos"));
        candidates.push(home.join("Amazon Drive/Pictures"));
    }
    for c in &candidates {
        if c.exists() {
            return json!({ "found": true, "default_path": c.to_string_lossy() });
        }
    }
    let fallback = dirs::picture_dir()
        .map(|p| p.join("Amazon Photos"))
        .unwrap_or_default();
    json!({ "found": false, "default_path": fallback.to_string_lossy() })
}

pub fn amazon_connect(core: &Core, path: &str, account: &str) -> Result<Settings, String> {
    let p = Path::new(path.trim());
    if !path.trim().is_empty() && !p.exists() {
        return Err(format!("Path does not exist: {}", path));
    }
    let info = json!({ "path": path.trim(), "account": account.trim() }).to_string();
    ai::set_key(AMAZON_KEY, &info)?;
    update_settings(core, |s| s.amazon_photos_path = path.trim().into())
}

pub fn amazon_import(core: &Arc<Core>) -> Result<Value, String> {
    let path = core.settings.lock().unwrap().amazon_photos_path.clone();
    if path.is_empty() {
        return Err("Amazon Photos is not connected".into());
    }
    sync_local_photos_into(core, "Amazon Photos", Path::new(&path), "Amazon Photos")
}

pub fn adobe_detect() -> Value {
    let mut candidates = Vec::new();
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join("Creative Cloud Files"));
        candidates.push(home.join("Creative Cloud Files (Personal)"));
        candidates.push(home.join("Creative Cloud Files (Team)"));
    }
    for c in &candidates {
        if c.exists() {
            return json!({ "found": true, "default_path": c.to_string_lossy() });
        }
    }
    let fallback = dirs::home_dir()
        .map(|p| p.join("Creative Cloud Files"))
        .unwrap_or_default();
    json!({ "found": false, "default_path": fallback.to_string_lossy() })
}

pub fn adobe_connect(core: &Core, path: &str, account: &str) -> Result<Settings, String> {
    let p = Path::new(path.trim());
    if !path.trim().is_empty() && !p.exists() {
        return Err(format!("Path does not exist: {}", path));
    }
    let info = json!({ "path": path.trim(), "account": account.trim() }).to_string();
    ai::set_key(ADOBE_KEY, &info)?;
    update_settings(core, |s| s.adobe_cloud_path = path.trim().into())
}

pub fn adobe_import(core: &Arc<Core>) -> Result<Value, String> {
    let path = core.settings.lock().unwrap().adobe_cloud_path.clone();
    if path.is_empty() {
        return Err("Adobe Cloud Photos is not connected".into());
    }
    sync_local_photos_into(core, "Adobe Cloud Photos", Path::new(&path), "Adobe Cloud Photos")
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

/// Upload rendered exports (`files[i]` shows photo `ids[i]`) to `immich`, `google`, `apple`, `amazon`, or `adobe`.
/// Returns the uploaded file names.
pub fn upload(core: &Core, dest: &str, ids: &[i64], files: &[String]) -> Result<Vec<String>, String> {
    let agent = agent();
    let one: Box<dyn Fn(i64, &str) -> Result<String, String> + '_> = match dest {
        "immich" => {
            let (url, key) = immich(core)?;
            Box::new(move |id: i64, file: &str| immich_upload(core, &agent, &url, &key, id, file))
        }
        "google" => {
            let bearer = format!("Bearer {}", google_access_token(core)?);
            Box::new(move |_: i64, file: &str| google_upload(&agent, &bearer, file))
        }
        "apple" => {
            let path = core.settings.lock().unwrap().apple_photos_path.clone();
            let default_dest = core.dir().join("remote").join("Apple iCloud Photos").join("Exports");
            Box::new(move |_: i64, file: &str| copy_to_service_export(Path::new(&path), &default_dest, file))
        }
        "amazon" => {
            let path = core.settings.lock().unwrap().amazon_photos_path.clone();
            let default_dest = core.dir().join("remote").join("Amazon Photos").join("Exports");
            Box::new(move |_: i64, file: &str| copy_to_service_export(Path::new(&path), &default_dest, file))
        }
        "adobe" => {
            let path = core.settings.lock().unwrap().adobe_cloud_path.clone();
            let default_dest = core.dir().join("remote").join("Adobe Cloud Photos").join("Exports");
            Box::new(move |_: i64, file: &str| copy_to_service_export(Path::new(&path), &default_dest, file))
        }
        _ => return Err(format!("unknown upload destination '{dest}'")),
    };
    let (mut done, mut failed) = (vec![], vec![]);
    for (i, (&id, file)) in ids.iter().zip(files).enumerate() {
        match one(id, file) {
            Ok(name) => done.push(name),
            Err(e) => failed.push(format!("{}: {e}", file_name(file))),
        }
        core.emit("export-progress", json!({"label": "Uploading", "done": i + 1, "total": files.len()}));
    }
    if failed.is_empty() {
        return Ok(done);
    }
    Err(format!("uploaded {} of {}; {}", done.len(), files.len(), failed.into_iter().take(3).collect::<Vec<_>>().join("; ")))
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Catalog path, plus the capture time in UTC for Immich's required `fileCreatedAt` (EXIF
/// times are read as this computer's local time), else now. The file's own EXIF date wins.
pub(crate) const UPLOAD_ROW: &str = "SELECT path, COALESCE(strftime('%Y-%m-%dT%H:%M:%SZ', captured_at, 'utc'), \
                          strftime('%Y-%m-%dT%H:%M:%SZ', 'now')) FROM photos WHERE id = ?";

/// Upload one file as a new Immich asset taken at `taken` (UTC, RFC 3339). The reply's `status` is
/// "duplicate" when Immich already has these exact bytes.
pub(crate) fn immich_post(agent: &ureq::Agent, url: &str, key: &str, file: &Path, taken: &str) -> Result<Value, String> {
    let name = file_name(&file.to_string_lossy());
    let device_asset_id = format!("optixel-{}", crate::new_token());
    // deviceAssetId/deviceId are required before Immich 3 and ignored after.
    let fields = [("deviceAssetId", device_asset_id.as_str()), ("deviceId", "optixel-studio"), ("fileCreatedAt", taken), ("fileModifiedAt", taken)];
    let (content_type, body) = multipart(&fields, &name, &fs::read(file).map_err(es)?);
    let reply = agent.post(&format!("{url}/api/assets")).header("x-api-key", key).header("content-type", &content_type).send(&body[..]);
    json_of("Immich", reply) // 201 created, or 200 when Immich already has these exact bytes
}

fn immich_upload(core: &Core, agent: &ureq::Agent, url: &str, key: &str, id: i64, file: &str) -> Result<String, String> {
    let (path, taken): (String, String) = core.db().query_row(UPLOAD_ROW, [id], |r| Ok((r.get(0)?, r.get(1)?))).map_err(es)?;
    let name = file_name(file);
    let v = immich_post(agent, url, key, Path::new(file), &taken)?;
    // Saving an Immich photo back: the edit goes on top of its stack and the original stays.
    if let (Some(orig), Some(new)) = (immich_id(&path), v["id"].as_str()) {
        let asset = json_of("Immich", agent.get(&format!("{url}/api/assets/{orig}")).header("x-api-key", key).call())?;
        let top = asset["stack"]["primaryAssetId"].as_str().unwrap_or(orig);
        if new != top {
            let stack = agent.post(&format!("{url}/api/stacks")).header("x-api-key", key).send_json(json!({"assetIds": [new, top]}));
            json_of("Immich", stack)?;
        }
    }
    Ok(name)
}

/// `multipart/form-data`: text fields, then the file as `assetData`. Export names are
/// already free of quotes and control characters.
fn multipart(fields: &[(&str, &str)], file_name: &str, data: &[u8]) -> (String, Vec<u8>) {
    let b = format!("optixel{}", crate::new_token());
    let mut body = vec![];
    for (k, v) in fields {
        body.extend_from_slice(format!("--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend_from_slice(
        format!("--{b}\r\nContent-Disposition: form-data; name=\"assetData\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(data);
    body.extend_from_slice(format!("\r\n--{b}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={b}"), body)
}

/// Library API upload: raw bytes for an upload token, then create the media item.
/// ponytail: one batchCreate per file; send up to 50 per call if API quota ever matters
pub(crate) fn google_upload(agent: &ureq::Agent, bearer: &str, file: &str) -> Result<String, String> {
    let name = file_name(file);
    let mime = match Path::new(&name).extension().and_then(|e| e.to_str()).unwrap_or("") {
        "avif" => "image/avif",
        "png" => "image/png",
        "tif" => "image/tiff",
        _ => "image/jpeg",
    };
    let mut r = agent
        .post("https://photoslibrary.googleapis.com/v1/uploads")
        .header("Authorization", bearer)
        .header("Content-Type", "application/octet-stream")
        .header("X-Goog-Upload-Content-Type", mime)
        .header("X-Goog-Upload-Protocol", "raw")
        .send(fs::File::open(file).map_err(es)?)
        .map_err(es)?;
    let status = r.status().as_u16();
    let token = r.body_mut().read_to_string().map_err(es)?;
    if status >= 400 {
        // Signed in before uploads existed: the token lacks photoslibrary.appendonly.
        let hint = if token.contains("insufficient") { " (disconnect Google Photos and sign in again to allow uploads)" } else { "" };
        return Err(format!("HTTP {status}: {}{hint}", token.chars().take(300).collect::<String>()));
    }
    let v = json_of(
        "Google Photos",
        agent
            .post("https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate")
            .header("Authorization", bearer)
            .send_json(json!({"newMediaItems": [{"simpleMediaItem": {"fileName": name, "uploadToken": token}}]})),
    )?;
    let st = &v["newMediaItemResults"][0]["status"];
    match st["code"].as_i64().unwrap_or(0) {
        0 => Ok(name),
        _ => Err(st["message"].as_str().unwrap_or("Google Photos rejected the file").into()),
    }
}

// ---------------------------------------------------------------------------
// Download + import
// ---------------------------------------------------------------------------

struct Item {
    id: String,
    /// Remote file name (untrusted).
    name: String,
    url: String,
}

/// Remote names are untrusted: keep one plain, portable path component.
pub(crate) fn safe_name(s: &str) -> String {
    let s: String = s.chars().map(|c| if c.is_control() || r#"/\:*?"<>|"#.contains(c) { '_' } else { c }).take(100).collect();
    let s = s.trim().trim_matches('.').trim();
    if s.is_empty() { "untitled".into() } else { s.into() }
}

/// `IMG_0001_<id tail>`: stable across re-imports and unique per remote item.
fn local_stem(it: &Item) -> String {
    let name = safe_name(&it.name);
    let stem = Path::new(&name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let id: String = it.id.chars().filter(char::is_ascii_alphanumeric).collect();
    format!("{stem}_{}", &id[id.len().saturating_sub(8)..])
}

/// Extension to store a download under, None when Optixel can't decode it. The remote
/// name decides, unless the server sends a plain image type for a name we can't read as
/// one (Google may serve a JPEG of a HEIC or RAW original).
fn save_ext(name: &str, content_type: &str) -> Option<String> {
    let named = Path::new(name).extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let served = match content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "image/jpeg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/webp" => Some("webp"),
        "image/tiff" => Some("tif"),
        _ => None,
    };
    if named == "heic" || named == "heif" {
        return served.map(String::from);
    }
    let raw = develop::RAW_EXTS.contains(&named.as_str()) && matches!(served, None | Some("tif")); // DNG and most RAWs are TIFF containers
    if raw || develop::IMG_EXTS.contains(&named.as_str()) { Some(named) } else { served.map(String::from) }
}

/// Stream one file to `<dir>/<stem>.<ext>`; None when its format is unsupported.
fn download(agent: &ureq::Agent, it: &Item, auth: (&str, &str), dir: &Path, stem: &str) -> Result<Option<PathBuf>, String> {
    let mut r = agent.get(&it.url).header(auth.0, auth.1).call().map_err(es)?;
    let status = r.status().as_u16();
    if status >= 400 {
        return Err(format!("HTTP {status}"));
    }
    let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    let Some(ext) = save_ext(&it.name, ct) else { return Ok(None) };
    // Write aside, then rename: an interrupted download never looks complete.
    let (path, part) = (dir.join(format!("{stem}.{ext}")), dir.join(format!("{stem}.{ext}.part")));
    let res = fs::File::create(&part)
        .and_then(|mut f| io::copy(&mut r.body_mut().as_reader(), &mut f))
        .and_then(|_| fs::rename(&part, &path));
    if let Err(e) = res {
        let _ = fs::remove_file(&part);
        return Err(es(e));
    }
    Ok(Some(path))
}

/// Download what `remote/<folder…>` doesn't have yet, then import that folder.
fn fetch_and_import(core: &Arc<Core>, folder: &[&str], skipped: usize, items: Vec<Item>, auth: (&str, &str)) -> Result<Value, String> {
    let dir = folder.iter().fold(core.dir().join("remote"), |d, f| d.join(safe_name(f)));
    fs::create_dir_all(&dir).map_err(es)?;
    let have: HashSet<String> = fs::read_dir(&dir)
        .map_err(es)?
        .flatten()
        .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    let todo: Vec<(String, Item)> =
        items.into_iter().map(|it| (local_stem(&it), it)).filter(|(stem, _)| !have.contains(stem)).collect();
    let (agent, done) = (agent(), AtomicUsize::new(0));
    let results: Vec<Result<Option<PathBuf>, String>> = todo
        .par_iter()
        .map(|(stem, it)| {
            let r = download(&agent, it, auth, &dir, stem).map_err(|e| format!("{}: {e}", it.name));
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            core.emit("import-progress", json!({"label": "Downloading", "done": n, "total": todo.len()}));
            r
        })
        .collect();
    let failed: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    for e in &failed {
        eprintln!("remote download {e}");
    }
    let ids = catalog::import(core, &dir, &|done, total| core.emit("import-progress", json!({"done": done, "total": total})))?;
    let imported = ids.len();
    core.emit("catalog-changed", json!({}));
    crate::auto::after_import(core, ids, "cloud");
    Ok(json!({
        "folder": dir,
        "imported": imported,
        "skipped": skipped + results.iter().filter(|r| matches!(r, Ok(None))).count(),
        "failed": failed.len(),
        "error": failed.first(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_names_stay_inside_the_folder() {
        assert_eq!(safe_name("../../.zshrc"), "_.._.zshrc");
        assert_eq!(safe_name(".."), "untitled");
        assert_eq!(safe_name("a\\b:c/IMG_1.JPG"), "a_b_c_IMG_1.JPG");
        assert_eq!(safe_name("  Trip. "), "Trip");
        let it = Item { id: "0198c1d2-7e4f-7a3b-9c5d-1e2f3a4b5c6d".into(), name: "../IMG_0001.CR3".into(), url: String::new() };
        assert_eq!(local_stem(&it), "_IMG_0001_3a4b5c6d");
    }

    #[test]
    fn saved_extension_follows_the_bytes() {
        assert_eq!(save_ext("IMG_1.JPG", "").as_deref(), Some("jpg"));
        assert_eq!(save_ext("IMG_1.CR3", "application/octet-stream").as_deref(), Some("cr3"));
        assert_eq!(save_ext("IMG_1.DNG", "image/tiff").as_deref(), Some("dng"));
        assert_eq!(save_ext("IMG_1.DNG", "image/jpeg").as_deref(), Some("jpg"));
        assert_eq!(save_ext("IMG_1.HEIC", "image/jpeg; charset=binary").as_deref(), Some("jpg"));
        assert_eq!(save_ext("IMG_1.HEIC", "image/heic"), None);
        assert_eq!(save_ext("clip.mp4", "video/mp4"), None);
    }

    #[test]
    fn immich_addresses_normalise() {
        assert_eq!(immich_base("192.168.1.5:2283/api/"), "http://192.168.1.5:2283");
        assert_eq!(immich_base(" https://photos.example.net/ "), "https://photos.example.net");
    }

    /// Fake Immich on loopback: browse a two-page album in place, open a JPEG original and a
    /// HEIC (through its full-size rendition) on demand, then save an edit back as a stack.
    #[test]
    fn immich_album_opens_in_place_and_saves_back() {
        let mut jpeg = vec![];
        image::DynamicImage::new_rgb8(8, 8).write_to(&mut io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg).unwrap();
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
        let originals = std::sync::Arc::new(AtomicUsize::new(0));
        let hits = originals.clone();
        std::thread::spawn(move || {
            for mut req in server.incoming_requests() {
                assert!(req.headers().iter().any(|h| h.field.equiv("x-api-key") && h.value.as_str() == "k"));
                let mut bytes = vec![];
                req.as_reader().read_to_end(&mut bytes).unwrap();
                let body = String::from_utf8_lossy(&bytes);
                let json = |v: Value| ("application/json", v.to_string().into_bytes());
                let put = *req.method() == tiny_http::Method::Put;
                let (ct, data) = match req.url() {
                    "/api/albums" => json(json!([{"id": "a1", "albumName": "Trip", "assetCount": 2}])),
                    "/api/search/metadata" => {
                        let q = serde_json::from_str::<Value>(&body).unwrap();
                        assert_eq!((q["order"].as_str(), q["size"].as_u64()), (Some("asc"), Some(250)));
                        let first = q["page"] == 1;
                        let items = if first {
                            json!([{"id": "u-1", "originalFileName": "IMG_1.JPG", "localDateTime": "2024-05-01T12:34:56.000Z", "isFavorite": true,
                                    "exifInfo": {"make": "FUJIFILM", "model": "X-T5", "iso": 200, "rating": 4}}])
                        } else {
                            json!([{"id": "u-2", "originalFileName": "IMG_2.HEIC"}])
                        };
                        json(json!({"assets": {"items": items, "nextPage": if first { json!("2") } else { Value::Null }}}))
                    }
                    "/api/assets/u-1/thumbnail" | "/api/assets/u-2/thumbnail" | "/api/assets/u-2/thumbnail?size=fullsize" => ("image/jpeg", jpeg.clone()),
                    "/api/assets/u-1/original" => {
                        hits.fetch_add(1, Ordering::SeqCst);
                        ("image/jpeg", jpeg.clone())
                    }
                    "/api/assets" if put => {
                        let v = serde_json::from_str::<Value>(&body).unwrap();
                        assert_eq!(v, json!({"ids": ["u-1"], "rating": -1, "isFavorite": false}));
                        ("", vec![]) // 204 No Content, as Immich replies
                    }
                    "/api/assets" => {
                        // EXIF local time → UTC: the day can shift with this machine's time zone.
                        assert!(body.contains("name=\"fileCreatedAt\"\r\n\r\n2024-0"), "{body}");
                        json(json!({"id": "n1", "status": "created"}))
                    }
                    "/api/assets/u-1" => json(json!({"id": "u-1", "stack": null})),
                    "/api/stacks" => {
                        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["assetIds"], json!(["n1", "u-1"]));
                        json(json!({"id": "s1"}))
                    }
                    _ => ("text/plain", vec![]),
                };
                let code = if ct.is_empty() { 204 } else if data.is_empty() { 500 } else { 200 };
                let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap();
                let _ = req.respond(tiny_http::Response::from_data(data).with_status_code(code).with_header(h));
            }
        });
        let dir = std::env::temp_dir().join(format!("ols-remote-test-{}", crate::new_token()));
        let core = Core::open(dir.clone()).unwrap();
        let page = |n| open_from_immich(&core, &url, "k", Some("a1"), n).unwrap()["next"].as_u64();
        let album = || catalog::query(&core.db(), &catalog::Filter { immich_album: Some("a1".into()), ..Default::default() }).unwrap();
        assert_eq!((page(1), album().len()), (Some(2), 1), "one page at a time");
        assert_eq!((page(2), album().len()), (None, 2));
        assert!(album().iter().all(|p| core.thumb_path(p.id).exists()));
        let p = album().into_iter().find(|p| p.path == "immich://u-1").unwrap();
        assert_eq!((p.camera.as_deref(), p.captured_at.as_deref()), (Some("FUJIFILM X-T5"), Some("2024-05-01 12:34:56")));
        assert_eq!((p.rating, p.flag), (4, 1), "Immich stars and favorite arrive as rating and pick");
        assert_eq!((page(1), album().len()), (Some(2), 1), "reopening starts the album over");
        let creds = || Ok((url.clone(), "k".to_string()));
        let reject = catalog::Culling { rating: Some(2), flag: Some("reject".into()), ..Default::default() };
        catalog::update(&core.db(), &[p.id], &reject).unwrap();
        push_to_immich(&core, &[p.id], creds).unwrap();
        assert_eq!(originals.load(Ordering::SeqCst), 0, "browsing must not download originals");
        let file = original_from_immich(&core, "u-1", creds).unwrap();
        assert_eq!(original_from_immich(&core, "u-1", creds).unwrap(), file);
        assert_eq!(originals.load(Ordering::SeqCst), 1, "an opened original is downloaded once");
        assert!(original_from_immich(&core, "u-2", creds).unwrap().ends_with("IMG_2_u2.jpg"));
        assert_eq!(immich_upload(&core, &agent(), &url, "k", p.id, &file.display().to_string()), Ok("IMG_1_u1.jpg".to_string()));
        let _ = fs::remove_dir_all(&dir);
    }
}
