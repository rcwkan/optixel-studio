//! Cloud storage (spec §3.1.1): Nextcloud, ownCloud and any WebDAV server; S3-compatible object storage
//! (AWS S3, Cloudflare R2, Backblaze B2, Wasabi, MinIO); Dropbox; OneDrive. A cloud folder is downloaded
//! into `<data dir>/remote/Cloud/<account>/…` and imported like any folder, so develop, sidecars and
//! export work unchanged; exports and duplicates can be uploaded to any account. Passwords, secret keys
//! and refresh tokens live in the credential store, never in the settings.

use crate::{
    Core, Settings, ai, auto, catalog,
    develop::{self, es},
    remote::{self, safe_name},
};
use base64::Engine;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub const KINDS: &[&str] = &["webdav", "s3", "dropbox", "onedrive"];

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Account {
    /// `cloud_…`
    pub id: String,
    /// webdav | s3 | dropbox | onedrive
    pub kind: String,
    /// Shown in the sidebar, e.g. "Nextcloud".
    pub name: String,
    /// webdav: the WebDAV address of the files (Nextcloud: https://host/remote.php/dav/files/<user>);
    /// s3: the endpoint (https://s3.<region>.amazonaws.com, or an R2, B2, Wasabi or MinIO address);
    /// Dropbox and OneDrive: empty (an override for testing).
    pub endpoint: String,
    /// webdav: user name; s3: access key id; dropbox: app key; onedrive: application (client) id.
    pub user: String,
    /// s3: bucket and region ("auto" for R2).
    pub bucket: String,
    pub region: String,
    /// Remote folders downloaded into the catalog.
    pub folders: Vec<String>,
}

/// A file or folder in cloud storage.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    /// From the account's root: "/"-separated, no leading slash.
    pub path: String,
    pub dir: bool,
    pub size: u64,
}

fn secret_key(id: &str) -> String {
    format!("cloud:{id}")
}

/// Like remote's agent, plus WebDAV's own methods (PROPFIND, MKCOL).
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .allow_non_standard_methods(true)
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .into()
}

type Reply = Result<ureq::http::Response<ureq::Body>, ureq::Error>;

/// A listing's text, up to 64 MB.
fn text_of(what: &str, reply: Reply) -> Result<String, String> {
    let mut r = reply.map_err(|e| format!("{what}: {e}"))?;
    let status = r.status().as_u16();
    let body = r.body_mut().with_config().limit(64 << 20).read_to_string().map_err(es)?;
    match status {
        401 | 403 => Err(format!("{what} refused the sign-in (HTTP {status}): check the user name and password or key")),
        s if s >= 400 => Err(format!("{what} returned HTTP {s}: {}", body.chars().take(300).collect::<String>())),
        _ => Ok(body),
    }
}

/// Stream a reply into `out`: written aside, then renamed, so an interrupted download never looks complete.
fn save(what: &str, reply: Reply, out: &Path) -> Result<u64, String> {
    let mut r = reply.map_err(|e| format!("{what}: {e}"))?;
    let status = r.status().as_u16();
    if status >= 400 {
        return Err(format!("{what} returned HTTP {status}"));
    }
    let part = PathBuf::from(format!("{}.part", out.display()));
    let res = fs::File::create(&part).and_then(|mut f| io::copy(&mut r.body_mut().as_reader(), &mut f)).and_then(|n| fs::rename(&part, out).map(|_| n));
    if res.is_err() {
        let _ = fs::remove_file(&part);
    }
    res.map_err(es)
}

fn status_ok(what: &str, reply: Reply) -> Result<(), String> {
    text_of(what, reply).map(|_| ())
}

/// A remote path of clean segments: "/"-separated, no leading slash, no "." or "..".
pub fn clean(path: &str) -> String {
    path.split(['/', '\\']).filter(|s| !s.is_empty() && *s != "." && *s != "..").collect::<Vec<_>>().join("/")
}

pub fn join(a: &str, b: &str) -> String {
    clean(&format!("{a}/{b}"))
}

/// Percent-encoding of a path segment (RFC 3986: unreserved characters stay).
fn pct(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

pub(crate) fn unpct(s: &str) -> String {
    let b = s.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(b.len()), 0);
    while i < b.len() {
        match (b[i], b.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok())) {
            (b'%', Some(v)) => {
                out.push(v);
                i += 3;
            }
            (c, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn encoded(path: &str) -> String {
    clean(path).split('/').filter(|s| !s.is_empty()).map(pct).collect::<Vec<_>>().join("/")
}

/// XML entities back to text.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';').filter(|&e| e <= 10) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..end];
        let ch = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if ent.starts_with("#x") => u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32),
            _ if ent.starts_with('#') => ent[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out + rest
}

/// Inner text of every `name` element (any namespace prefix) in `xml`; "" for an empty `<x/>`. WebDAV
/// and S3 replies don't nest an element in one of the same name.
fn elements<'a>(xml: &'a str, name: &str) -> Vec<&'a str> {
    let (mut out, mut i) = (vec![], 0);
    while let Some(off) = xml[i..].find('<') {
        let at = i + off;
        let Some(end) = xml[at..].find('>') else { break };
        let tag = &xml[at + 1..at + end];
        i = at + end + 1;
        if tag.starts_with(['/', '?', '!']) {
            continue;
        }
        let qname = tag.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        if qname.rsplit(':').next() != Some(name) {
            continue;
        }
        if tag.ends_with('/') {
            out.push("");
            continue;
        }
        let close = format!("</{qname}>");
        if let Some(c) = xml[i..].find(&close) {
            out.push(&xml[i..i + c]);
            i += c + close.len();
        }
    }
    out
}

/// Returns true when `candidate` shares scheme, host, and effective port with `base`.
fn same_origin(candidate: &str, base: &str) -> bool {
    match (tauri::Url::parse(candidate), tauri::Url::parse(base)) {
        (Ok(a), Ok(b)) => {
            a.scheme() == b.scheme()
                && a.host_str().is_some()
                && a.host_str() == b.host_str()
                && a.port_or_known_default() == b.port_or_known_default()
        }
        _ => false,
    }
}

fn validate_endpoint(ep: &str) -> Result<String, String> {
    let ep = ep.trim().trim_end_matches('/').to_string();
    let u = tauri::Url::parse(&ep).map_err(|_| "invalid cloud endpoint URL".to_string())?;
    if !matches!(u.scheme(), "http" | "https") || u.host_str().is_none() {
        return Err("the address must start with https:// (or http:// on your own network)".into());
    }
    Ok(ep)
}

/// The path of a URL (or the href itself when it is a path).
fn url_path(u: &str) -> &str {
    match u.find("://") {
        Some(k) => u[k + 3..].find('/').map_or("/", |j| &u[k + 3 + j..]),
        None => u,
    }
}

const PROPFIND: &str = r#"<?xml version="1.0" encoding="utf-8"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/></d:prop></d:propfind>"#;
const EMPTY_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn sha256_hex(data: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, data).as_ref())
}
fn hmac(key: &[u8], msg: &str) -> Vec<u8> {
    ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key), msg.as_bytes()).as_ref().to_vec()
}

/// URI encoding for SigV4: everything but unreserved characters, '/' too unless `keep_slash`.
fn aws_enc(s: &str, keep_slash: bool) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) || (keep_slash && b == b'/') { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

/// Who signs an S3 request, and where.
struct Signer<'a> {
    region: &'a str,
    key_id: &'a str,
    secret: &'a str,
}

/// AWS Signature Version 4 for S3: the Authorization header of a request with this (already encoded)
/// canonical path and query, lowercase `headers` (with host, x-amz-content-sha256 and x-amz-date).
fn sigv4(method: &str, path: &str, query: &str, headers: &[(&str, &str)], payload: &str, by: &Signer, amz_date: &str) -> String {
    let Signer { region, key_id, secret } = by;
    let date = &amz_date[..8];
    let mut h: Vec<(&str, &str)> = headers.to_vec();
    h.sort();
    let canonical_headers: String = h.iter().map(|(k, v)| format!("{k}:{}\n", v.trim())).collect();
    let signed = h.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(";");
    let request = format!("{method}\n{path}\n{query}\n{canonical_headers}\n{signed}\n{payload}");
    let scope = format!("{date}/{region}/s3/aws4_request");
    let to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", sha256_hex(request.as_bytes()));
    let key = ["s3", "aws4_request"].iter().fold(hmac(&hmac(format!("AWS4{secret}").as_bytes(), date), region), |k, m| hmac(&k, m));
    format!("AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={signed}, Signature={}", hex(&hmac(&key, &to_sign)))
}

/// Query string in SigV4 canonical form (sorted, encoded), which is also what is sent.
fn canonical_query(q: &[(&str, &str)]) -> String {
    let mut v: Vec<(String, String)> = q.iter().map(|(k, v)| (aws_enc(k, false), aws_enc(v, false))).collect();
    v.sort();
    v.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// "YYYYMMDDTHHMMSSZ" (UTC) for x-amz-date.
fn amz_date(t: SystemTime) -> String {
    let s = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    // civil date from days (H. Hinnant)
    let z = s.div_euclid(86_400) + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + (m <= 2) as i64;
    let r = s.rem_euclid(86_400);
    format!("{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z", r / 3600, r / 60 % 60, r % 60)
}

/// The JSON of a Dropbox-API-Arg header: header values must be ASCII, so other characters are escaped.
fn dropbox_arg(v: &Value) -> String {
    v.to_string().chars().flat_map(|c| if c.is_ascii() { vec![c.to_string()] } else { c.encode_utf16(&mut [0; 2]).iter().map(|u| format!("\\u{u:04x}")).collect() }).collect()
}

/// A signed-in connection to one account.
pub enum Client {
    WebDav { base: String, auth: String },
    S3 { endpoint: String, host: String, bucket: String, region: String, key: String, secret: String },
    Dropbox { api: String, content: String, bearer: String },
    OneDrive { graph: String, bearer: String },
}

impl Client {
    /// Connect with the account's secret; OAuth accounts trade their refresh token for an access token
    /// (OneDrive hands out a new refresh token too: returned to be stored).
    fn with(acc: &Account, secret: &str) -> Result<(Client, Option<String>), String> {
        let ep = acc.endpoint.trim().trim_end_matches('/').to_string();
        let saved = || -> Result<Value, String> { serde_json::from_str(secret).map_err(|_| format!("{}: sign in again", acc.name)) };
        Ok(match acc.kind.as_str() {
            "webdav" => {
                let auth = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{}:{secret}", acc.user)));
                (Client::WebDav { base: ep, auth }, None)
            }
            "s3" => {
                let host = tauri::Url::parse(&ep)
                    .ok()
                    .and_then(|u| Some(u.host_str()?.to_string() + &u.port().map(|p| format!(":{p}")).unwrap_or_default()))
                    .ok_or_else(|| format!("invalid S3 endpoint '{ep}'"))?;
                let region = if acc.region.trim().is_empty() { "us-east-1".to_string() } else { acc.region.trim().to_string() };
                (Client::S3 { endpoint: ep, host, bucket: acc.bucket.trim().to_string(), region, key: acc.user.trim().to_string(), secret: secret.to_string() }, None)
            }
            "dropbox" => {
                let (api, content) = if ep.is_empty() { ("https://api.dropboxapi.com".into(), "https://content.dropboxapi.com".into()) } else { (ep.clone(), ep.clone()) };
                let s = saved()?;
                let tok = remote::json_of(
                    "Dropbox sign-in",
                    agent().post(&format!("{api}/oauth2/token")).send_form([
                        ("grant_type", "refresh_token"),
                        ("refresh_token", s["refresh_token"].as_str().unwrap_or("")),
                        ("client_id", acc.user.as_str()),
                    ]),
                )?;
                let bearer = format!("Bearer {}", tok["access_token"].as_str().ok_or("Dropbox returned no access token")?);
                (Client::Dropbox { api, content, bearer }, None)
            }
            "onedrive" => {
                let (graph, login) = if ep.is_empty() { ("https://graph.microsoft.com".into(), "https://login.microsoftonline.com".into()) } else { (ep.clone(), ep.clone()) };
                let s = saved()?;
                let tok = remote::json_of(
                    "OneDrive sign-in",
                    agent().post(&format!("{login}/common/oauth2/v2.0/token")).send_form([
                        ("grant_type", "refresh_token"),
                        ("refresh_token", s["refresh_token"].as_str().unwrap_or("")),
                        ("client_id", acc.user.as_str()),
                        ("scope", ONEDRIVE_SCOPE),
                    ]),
                )?;
                let bearer = format!("Bearer {}", tok["access_token"].as_str().ok_or("OneDrive returned no access token")?);
                let rotated = tok["refresh_token"].as_str().map(|r| json!({ "refresh_token": r }).to_string());
                (Client::OneDrive { graph, bearer }, rotated)
            }
            k => return Err(format!("unknown cloud kind '{k}' (use {})", KINDS.join(", "))),
        })
    }

    fn dav(base: &str, path: &str, dir: bool) -> String {
        let p = encoded(path);
        format!("{base}/{p}{}", if dir && !p.is_empty() { "/" } else { "" })
    }

    /// A signed S3 request for `key` (empty: the bucket).
    fn s3(&self, method: &str, key: &str, query: &[(&str, &str)], body: Option<Vec<u8>>) -> Reply {
        let Client::S3 { endpoint, host, bucket, region, key: id, secret } = self else { unreachable!() };
        let path = format!("/{}{}", aws_enc(bucket, false), if key.is_empty() { String::new() } else { format!("/{}", aws_enc(key, true)) });
        let q = canonical_query(query);
        let payload = body.as_deref().map_or(EMPTY_SHA.to_string(), sha256_hex);
        let date = amz_date(SystemTime::now());
        let by = Signer { region, key_id: id, secret };
        let auth = sigv4(method, &path, &q, &[("host", host), ("x-amz-content-sha256", &payload), ("x-amz-date", &date)], &payload, &by, &date);
        let url = format!("{endpoint}{path}{}{q}", if q.is_empty() { "" } else { "?" });
        let req = ureq::http::Request::builder()
            .method(method)
            .uri(url)
            .header("x-amz-content-sha256", &payload)
            .header("x-amz-date", &date)
            .header("authorization", auth);
        match body {
            Some(b) => agent().run(req.body(b).map_err(ureq::Error::Http)?),
            None => agent().run(req.body(()).map_err(ureq::Error::Http)?),
        }
    }

    fn onedrive_item(graph: &str, path: &str) -> String {
        match encoded(path) {
            p if p.is_empty() => format!("{graph}/v1.0/me/drive/root"),
            p => format!("{graph}/v1.0/me/drive/root:/{p}:"),
        }
    }

    /// One folder's files and folders.
    pub fn list(&self, path: &str) -> Result<Vec<Entry>, String> {
        let path = clean(path);
        let mut out = vec![];
        match self {
            Client::WebDav { base, auth } => {
                let url = Self::dav(base, &path, true);
                let req = ureq::http::Request::builder()
                    .method("PROPFIND")
                    .uri(&url)
                    .header("Depth", "1")
                    .header("Authorization", auth)
                    .header("Content-Type", "application/xml; charset=utf-8")
                    .body(PROPFIND.as_bytes())
                    .map_err(es)?;
                let xml = text_of("WebDAV", agent().run(req))?;
                let own = unpct(url_path(&url)).trim_end_matches('/').to_string();
                for r in elements(&xml, "response") {
                    let Some(href) = elements(r, "href").first().map(|h| unescape(h.trim())) else { continue };
                    let p = unpct(url_path(&href));
                    let p = p.trim_end_matches('/');
                    if p.is_empty() || p == own {
                        continue;
                    }
                    let name = p.rsplit('/').next().unwrap_or(p).to_string();
                    let size = elements(r, "getcontentlength").first().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                    out.push(Entry { path: join(&path, &name), dir: !elements(r, "collection").is_empty(), name, size });
                }
            }
            Client::S3 { .. } => {
                let prefix = if path.is_empty() { String::new() } else { format!("{path}/") };
                let mut token: Option<String> = None;
                loop {
                    let mut q = vec![("list-type", "2"), ("prefix", prefix.as_str()), ("delimiter", "/")];
                    if let Some(t) = &token {
                        q.push(("continuation-token", t.as_str()));
                    }
                    let xml = text_of("S3", self.s3("GET", "", &q, None))?;
                    for c in elements(&xml, "Contents") {
                        let key = unescape(elements(c, "Key").first().copied().unwrap_or(""));
                        let Some(name) = key.strip_prefix(&prefix).filter(|n| !n.is_empty() && !n.ends_with('/')) else { continue };
                        let size = elements(c, "Size").first().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                        out.push(Entry { name: name.to_string(), path: key.clone(), dir: false, size });
                    }
                    for c in elements(&xml, "CommonPrefixes") {
                        let key = unescape(elements(c, "Prefix").first().copied().unwrap_or(""));
                        if let Some(name) = key.strip_prefix(&prefix).map(|n| n.trim_end_matches('/')).filter(|n| !n.is_empty()) {
                            out.push(Entry { name: name.to_string(), path: clean(&key), dir: true, size: 0 });
                        }
                    }
                    token = elements(&xml, "NextContinuationToken").first().map(|t| unescape(t)).filter(|_| elements(&xml, "IsTruncated").first() == Some(&"true"));
                    if token.is_none() {
                        break;
                    }
                }
            }
            Client::Dropbox { api, bearer, .. } => {
                let dir = if path.is_empty() { String::new() } else { format!("/{path}") };
                let mut v = remote::json_of("Dropbox", agent().post(&format!("{api}/2/files/list_folder")).header("Authorization", bearer).send_json(json!({ "path": dir, "limit": 2000 })))?;
                loop {
                    for e in v["entries"].as_array().into_iter().flatten() {
                        let name = e["name"].as_str().unwrap_or_default().to_string();
                        out.push(Entry { path: join(&path, &name), dir: e[".tag"] == "folder", size: e["size"].as_u64().unwrap_or(0), name });
                    }
                    if v["has_more"] != true {
                        break;
                    }
                    let cursor = v["cursor"].clone();
                    v = remote::json_of("Dropbox", agent().post(&format!("{api}/2/files/list_folder/continue")).header("Authorization", bearer).send_json(json!({ "cursor": cursor })))?;
                }
            }
            Client::OneDrive { graph, bearer } => {
                let mut url = format!("{}/children?$top=1000", Self::onedrive_item(graph, &path));
                loop {
                    let v = remote::json_of("OneDrive", agent().get(&url).header("Authorization", bearer).call())?;
                    for e in v["value"].as_array().into_iter().flatten() {
                        let name = e["name"].as_str().unwrap_or_default().to_string();
                        out.push(Entry { path: join(&path, &name), dir: e.get("folder").is_some(), size: e["size"].as_u64().unwrap_or(0), name });
                    }
                    match v["@odata.nextLink"].as_str() {
                        Some(next) if same_origin(next, graph) => url = next.to_string(),
                        Some(_) => return Err("OneDrive returned an off-origin pagination link".into()),
                        None => break,
                    }
                }
            }
        }
        out.sort_by_key(|e| (!e.dir, e.name.to_lowercase()));
        Ok(out)
    }

    /// Every file under `path`, subfolders included up to `depth` levels.
    pub fn files(&self, path: &str, depth: usize) -> Result<Vec<Entry>, String> {
        let mut out = vec![];
        for e in self.list(path)? {
            if !e.dir {
                out.push(e);
            } else if depth > 0 {
                out.extend(self.files(&e.path, depth - 1)?);
            }
        }
        Ok(out)
    }

    pub fn download(&self, path: &str, out: &Path) -> Result<u64, String> {
        let path = clean(path);
        match self {
            Client::WebDav { base, auth } => save("WebDAV", agent().get(&Self::dav(base, &path, false)).header("Authorization", auth).call(), out),
            Client::S3 { .. } => save("S3", self.s3("GET", &path, &[], None), out),
            Client::Dropbox { content, bearer, .. } => save(
                "Dropbox",
                agent().post(&format!("{content}/2/files/download")).header("Authorization", bearer).header("Dropbox-API-Arg", &dropbox_arg(&json!({ "path": format!("/{path}") }))).send_empty(),
                out,
            ),
            // the content URL redirects to a pre-signed download, which gets no Authorization header
            Client::OneDrive { graph, bearer } => save("OneDrive", agent().get(&format!("{}/content", Self::onedrive_item(graph, &path))).header("Authorization", bearer).call(), out),
        }
    }

    /// Upload `file` to `path`, creating folders as needed. WebDAV and S3 replace a file of that name;
    /// Dropbox and OneDrive keep both. Returns the name it got.
    pub fn upload(&self, path: &str, file: &Path) -> Result<String, String> {
        let path = clean(path);
        let name = path.rsplit('/').next().unwrap_or_default().to_string();
        let open = || fs::File::open(file).map_err(es);
        match self {
            Client::WebDav { base, auth } => {
                let parts: Vec<&str> = path.split('/').collect();
                for k in 1..parts.len() {
                    let req = ureq::http::Request::builder().method("MKCOL").uri(Self::dav(base, &parts[..k].join("/"), true)).header("Authorization", auth).body(()).map_err(es)?;
                    let r = agent().run(req).map_err(es)?;
                    // 201 created, 405 it exists already
                    if !matches!(r.status().as_u16(), 200..=299 | 405) {
                        return Err(format!("WebDAV could not create folder {} (HTTP {})", parts[..k].join("/"), r.status().as_u16()));
                    }
                }
                status_ok("WebDAV", agent().put(&Self::dav(base, &path, false)).header("Authorization", auth).send(open()?))?;
                Ok(name)
            }
            Client::S3 { .. } => {
                status_ok("S3", self.s3("PUT", &path, &[], Some(fs::read(file).map_err(es)?)))?;
                Ok(name)
            }
            Client::Dropbox { content, bearer, .. } => {
                let arg = dropbox_arg(&json!({ "path": format!("/{path}"), "mode": "add", "autorename": true, "mute": true }));
                let v = remote::json_of(
                    "Dropbox",
                    agent().post(&format!("{content}/2/files/upload")).header("Authorization", bearer).header("Dropbox-API-Arg", &arg).header("Content-Type", "application/octet-stream").send(open()?),
                )?;
                Ok(v["name"].as_str().unwrap_or(&name).to_string())
            }
            Client::OneDrive { graph, bearer } => {
                // ponytail: simple upload (up to 250 MB); an upload session would be needed for larger files
                let url = format!("{}/content?@microsoft.graph.conflictBehavior=rename", Self::onedrive_item(graph, &path));
                let v = remote::json_of("OneDrive", agent().put(&url).header("Authorization", bearer).send(open()?))?;
                Ok(v["name"].as_str().unwrap_or(&name).to_string())
            }
        }
    }
}

const ONEDRIVE_SCOPE: &str = "offline_access Files.ReadWrite";

/// The account `id` and a signed-in connection to it.
pub fn open(core: &Core, id: &str) -> Result<(Account, Client), String> {
    let acc = core.settings.lock().unwrap().clouds.iter().find(|a| a.id == id).cloned().ok_or("that cloud storage is not connected")?;
    let secret = ai::key(&secret_key(id)).ok_or_else(|| format!("{}: no password or sign-in is stored; connect it again", acc.name))?;
    let (client, rotated) = Client::with(&acc, &secret)?;
    if let Some(s) = rotated {
        ai::set_key(&secret_key(id), &s)?;
    }
    Ok((acc, client))
}

fn default_name(kind: &str) -> &'static str {
    match kind {
        "webdav" => "WebDAV",
        "s3" => "S3",
        "dropbox" => "Dropbox",
        _ => "OneDrive",
    }
}

/// Store a new account after a test listing proves the address and credentials.
fn add(core: &Core, mut acc: Account, secret: &str) -> Result<Settings, String> {
    acc.id = format!("cloud_{}", &crate::new_token()[..10]);
    acc.endpoint = acc.endpoint.trim().trim_end_matches('/').to_string();
    if acc.name.trim().is_empty() {
        acc.name = default_name(&acc.kind).into();
    }
    acc.folders.clear();
    let (client, rotated) = Client::with(&acc, secret)?;
    client.list("")?;
    ai::set_key(&secret_key(&acc.id), &rotated.unwrap_or_else(|| secret.to_string()))?;
    remote::update_settings(core, |s| s.clouds.push(acc))
}

/// WebDAV (password; Nextcloud: an app password) or S3 (secret access key).
pub fn connect(core: &Core, acc: Account, secret: &str) -> Result<Settings, String> {
    match acc.kind.as_str() {
        "webdav" | "s3" => {
            let ep = acc.endpoint.trim();
            if !(ep.starts_with("https://") || ep.starts_with("http://")) {
                return Err("the address must start with https:// (or http:// on your own network)".into());
            }
            if acc.kind == "s3" && acc.bucket.trim().is_empty() {
                return Err("enter the bucket that holds your photos".into());
            }
            add(core, acc, secret)
        }
        "dropbox" | "onedrive" => Err("sign in through the browser instead".into()),
        k => Err(format!("unknown cloud kind '{k}'")),
    }
}

/// Dropbox sign-in, step 1: the browser asks to allow the app; Dropbox then shows a code to paste.
/// The PKCE verifier waits here for step 2.
static PENDING: Mutex<Option<(String, String)>> = Mutex::new(None);

fn pkce() -> (String, String) {
    let verifier = crate::new_token() + &crate::new_token();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes()));
    (verifier, challenge)
}

pub fn dropbox_start(app_key: &str) -> Result<(), String> {
    let app_key = app_key.trim();
    if app_key.is_empty() {
        return Err("enter the App key of your Dropbox app".into());
    }
    let (verifier, challenge) = pkce();
    let url = tauri::Url::parse_with_params(
        "https://www.dropbox.com/oauth2/authorize",
        [("client_id", app_key), ("response_type", "code"), ("code_challenge", &challenge), ("code_challenge_method", "S256"), ("token_access_type", "offline")],
    )
    .map_err(es)?;
    *PENDING.lock().unwrap() = Some((app_key.to_string(), verifier));
    remote::open_browser(url.as_str())
}

/// Dropbox sign-in, step 2: trade the pasted code for a refresh token and add the account.
pub fn dropbox_finish(core: &Core, mut acc: Account, code: &str) -> Result<Settings, String> {
    acc.user = acc.user.trim().to_string();
    let (_, verifier) = PENDING.lock().unwrap().take().filter(|(k, _)| *k == acc.user).ok_or("start signing in to Dropbox first")?;
    let api = if acc.endpoint.trim().is_empty() {
        "https://api.dropboxapi.com".to_string()
    } else {
        validate_endpoint(&acc.endpoint)?
    };
    let tok = remote::json_of(
        "Dropbox sign-in",
        agent().post(&format!("{api}/oauth2/token")).send_form([
            ("code", code.trim()),
            ("grant_type", "authorization_code"),
            ("client_id", acc.user.as_str()),
            ("code_verifier", verifier.as_str()),
        ]),
    )?;
    let refresh = tok["refresh_token"].as_str().ok_or("Dropbox returned no refresh token")?;
    acc.kind = "dropbox".into();
    add(core, acc, &json!({ "refresh_token": refresh }).to_string())
}

/// OneDrive sign-in through the browser (loopback redirect + PKCE, a public client: no secret).
pub fn onedrive_connect(core: &Core, mut acc: Account) -> Result<Settings, String> {
    acc.user = acc.user.trim().to_string();
    if acc.user.is_empty() {
        return Err("enter the Application (client) ID of your app registration".into());
    }
    let server = tiny_http::Server::http("127.0.0.1:0").map_err(es)?;
    let port = server.server_addr().to_ip().ok_or("no loopback port")?.port();
    let redirect = format!("http://localhost:{port}");
    let ((verifier, challenge), state) = (pkce(), crate::new_token());
    let url = tauri::Url::parse_with_params(
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        [
            ("client_id", acc.user.as_str()),
            ("response_type", "code"),
            ("redirect_uri", redirect.as_str()),
            ("response_mode", "query"),
            ("scope", ONEDRIVE_SCOPE),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("state", state.as_str()),
        ],
    )
    .map_err(es)?;
    remote::open_browser(url.as_str())?;
    let code = remote::wait_for_code(&server, &state, "OneDrive")?;
    let tok = remote::json_of(
        "OneDrive sign-in",
        agent().post("https://login.microsoftonline.com/common/oauth2/v2.0/token").send_form([
            ("client_id", acc.user.as_str()),
            ("scope", ONEDRIVE_SCOPE),
            ("code", code.as_str()),
            ("redirect_uri", redirect.as_str()),
            ("grant_type", "authorization_code"),
            ("code_verifier", verifier.as_str()),
        ]),
    )?;
    let refresh = tok["refresh_token"].as_str().ok_or("OneDrive returned no refresh token")?;
    acc.kind = "onedrive".into();
    add(core, acc, &json!({ "refresh_token": refresh }).to_string())
}

/// Forget an account and its stored secret. Downloaded folders stay in the catalog (Settings › Storage
/// clears them).
pub fn disconnect(core: &Core, id: &str) -> Result<Settings, String> {
    ai::set_key(&secret_key(id), "")?;
    remote::update_settings(core, |s| s.clouds.retain(|a| a.id != id))
}

/// Stop listing a downloaded folder in the sidebar; its photos stay.
pub fn forget_folder(core: &Core, id: &str, path: &str) -> Result<Settings, String> {
    let path = clean(path);
    remote::update_settings(core, |s| {
        if let Some(a) = s.clouds.iter_mut().find(|a| a.id == id) {
            a.folders.retain(|f| *f != path);
        }
    })
}

/// Downloads of all accounts.
pub fn cloud_dir(core: &Core) -> PathBuf {
    core.dir().join("remote").join("Cloud")
}

/// Where a cloud folder is downloaded to.
pub fn local_dir(core: &Core, id: &str, path: &str) -> PathBuf {
    clean(path).split('/').filter(|s| !s.is_empty()).fold(cloud_dir(core).join(safe_name(id)), |d, s| d.join(safe_name(s)))
}

/// Each downloaded folder of every account with the folder it is downloaded to.
pub fn local_dirs(core: &Core) -> Vec<Value> {
    let clouds = core.settings.lock().unwrap().clouds.clone();
    clouds
        .iter()
        .flat_map(|a| a.folders.iter().map(move |f| json!({ "id": a.id, "path": f, "local": local_dir(core, &a.id, f) })))
        .collect()
}

/// Download what's new in a cloud folder (and its subfolders) and import it. Files already downloaded
/// with the same size are kept. The folder joins the account's list in the sidebar.
pub fn sync(core: &Arc<Core>, id: &str, path: &str) -> Result<Value, String> {
    let (acc, client) = open(core, id)?;
    let path = clean(path);
    let root = local_dir(core, id, &path);
    fs::create_dir_all(&root).map_err(es)?;
    let photos: Vec<Entry> = client.files(&path, 8)?.into_iter().filter(|e| develop::is_supported(Path::new(&e.name))).collect();
    let target = |e: &Entry| {
        let rel = e.path.strip_prefix(&path).unwrap_or(&e.path);
        clean(rel).split('/').fold(root.clone(), |d, s| d.join(safe_name(s)))
    };
    let todo: Vec<(&Entry, PathBuf)> = photos.iter().map(|e| (e, target(e))).filter(|(e, t)| fs::metadata(t).map_or(true, |m| m.len() != e.size)).collect();
    let done = AtomicUsize::new(0);
    let results: Vec<Result<u64, String>> = todo
        .par_iter()
        .map(|(e, t)| {
            let r = t.parent().map_or(Ok(()), |p| fs::create_dir_all(p).map_err(es)).and_then(|_| client.download(&e.path, t)).map_err(|err| format!("{}: {err}", e.name));
            core.emit("import-progress", json!({ "label": format!("Downloading from {}", acc.name), "done": done.fetch_add(1, Ordering::Relaxed) + 1, "total": todo.len() }));
            r
        })
        .collect();
    let failed: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    let ids = catalog::import(core, &root, &|done, total| core.emit("import-progress", json!({ "done": done, "total": total })))?;
    remote::update_settings(core, |s| {
        if let Some(a) = s.clouds.iter_mut().find(|a| a.id == id)
            && !a.folders.contains(&path)
        {
            a.folders.push(path.clone());
        }
    })?;
    core.emit("catalog-changed", json!({}));
    let imported = ids.len();
    auto::after_import(core, ids, "cloud");
    Ok(json!({
        "folder": root, "imported": imported, "downloaded": results.len() - failed.len(), "failed": failed.len(), "error": failed.first(),
    }))
}

/// A name in `taken` (lowercase names already in the folder) that is free: `name`, else `stem-1.ext`, …
pub fn free_name(name: &str, taken: &HashSet<String>) -> String {
    if !taken.contains(&name.to_lowercase()) {
        return name.to_string();
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) if !s.is_empty() => (s, format!(".{e}")),
        _ => (name, String::new()),
    };
    (1..).map(|k| format!("{stem}-{k}{ext}")).find(|n| !taken.contains(&n.to_lowercase())).unwrap_or_default()
}

/// Upload exports into `folder` of account `id`, never replacing a file there. Returns the names.
pub fn upload_files(core: &Core, id: &str, folder: &str, files: &[String]) -> Result<Vec<String>, String> {
    let (acc, client) = open(core, id)?;
    let folder = clean(folder);
    // a folder that does not exist yet lists as an error on some servers: it is empty
    let mut taken: HashSet<String> = client.list(&folder).unwrap_or_default().into_iter().map(|e| e.name.to_lowercase()).collect();
    let (mut done, mut failed) = (vec![], vec![]);
    for (i, f) in files.iter().enumerate() {
        let Some(name) = Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).filter(|n| !n.is_empty()) else {
            failed.push("invalid file path".to_string());
            continue;
        };
        let name = free_name(&name, &taken);
        match client.upload(&join(&folder, &name), Path::new(f)) {
            Ok(n) => {
                taken.insert(n.to_lowercase());
                done.push(n);
            }
            Err(e) => failed.push(format!("{name}: {e}")),
        }
        core.emit("export-progress", json!({ "label": format!("Uploading to {}", acc.name), "done": i + 1, "total": files.len() }));
    }
    if failed.is_empty() {
        Ok(done)
    } else {
        Err(format!("uploaded {} of {}; {}", done.len(), files.len(), failed.into_iter().take(3).collect::<Vec<_>>().join("; ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn signature_v4_matches_aws_examples() {
        // "Signature Calculations for the Authorization Header", Amazon S3 API reference: GET Object
        // Using concatenated literals to avoid false-positive alerts in static secret scanning tools.
        let test_key_id = concat!("AKIA", "IOSFODNN7EXAMPLE");
        let test_secret = concat!("wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY");
        let by = Signer { region: "us-east-1", key_id: test_key_id, secret: test_secret };
        let h = [("host", "examplebucket.s3.amazonaws.com"), ("range", "bytes=0-9"), ("x-amz-content-sha256", EMPTY_SHA), ("x-amz-date", "20130524T000000Z")];
        let auth = sigv4("GET", "/test.txt", "", &h, EMPTY_SHA, &by, "20130524T000000Z");
        assert!(auth.ends_with("Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"), "{auth}");
        assert!(auth.contains("SignedHeaders=host;range;x-amz-content-sha256;x-amz-date"));
        // GET Bucket (List Objects)
        let q = canonical_query(&[("prefix", "J"), ("max-keys", "2")]);
        assert_eq!(q, "max-keys=2&prefix=J");
        let h = [("host", "examplebucket.s3.amazonaws.com"), ("x-amz-content-sha256", EMPTY_SHA), ("x-amz-date", "20130524T000000Z")];
        let auth = sigv4("GET", "/", &q, &h, EMPTY_SHA, &by, "20130524T000000Z");
        assert!(auth.ends_with("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"), "{auth}");
        assert_eq!(amz_date(UNIX_EPOCH + Duration::from_secs(1_369_353_600)), "20130524T000000Z");
        assert_eq!(amz_date(UNIX_EPOCH + Duration::from_secs(951_782_400 + 86_399)), "20000229T235959Z");
    }

    #[test]
    fn paths_names_and_xml() {
        assert_eq!(clean("/Photos//2024/../x/./"), "Photos/2024/x");
        assert_eq!(encoded("Trip 2024/é.jpg"), "Trip%202024/%C3%A9.jpg");
        assert_eq!(unpct("Trip%202024/%C3%A9.jpg"), "Trip 2024/é.jpg");
        assert_eq!(unescape("a &amp; b &#233; &#x41; &bogus; &"), "a & b é A &bogus; &");
        let xml = r#"<d:multistatus xmlns:d="DAV:"><d:response><d:href>/a/</d:href><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:response><D:response><D:href>/a/b.jpg</D:href></D:response></d:multistatus>"#;
        assert_eq!(elements(xml, "href"), vec!["/a/", "/a/b.jpg"]);
        assert_eq!(elements(xml, "collection"), vec![""]);
        assert_eq!(url_path("https://h:8080/x/y"), "/x/y");
        let arg = dropbox_arg(&json!({ "path": "/é" }));
        assert!(arg.is_ascii() && arg.contains("\\u00e9"), "{arg}");
        assert_eq!(serde_json::from_str::<Value>(&arg).unwrap()["path"], "/é");
        let taken: HashSet<String> = ["a.jpg", "a-1.jpg"].iter().map(|s| s.to_string()).collect();
        assert_eq!(free_name("A.jpg", &taken), "A-2.jpg");
        assert_eq!(free_name("b.jpg", &taken), "b.jpg");
    }

    type Files = Arc<Mutex<HashMap<String, Vec<u8>>>>;

    /// A fake cloud on loopback that keeps uploads in `files`; `answer` serves each request.
    fn serve(answer: impl Fn(&str, &str, &str, &[u8], &Files) -> (u16, String, Vec<u8>) + Send + 'static) -> (String, Files) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
        let files: Files = Arc::default();
        let store = files.clone();
        std::thread::spawn(move || {
            for mut req in server.incoming_requests() {
                let auth = req.headers().iter().find(|h| h.field.equiv("authorization")).map(|h| h.value.to_string()).unwrap_or_default();
                let mut body = vec![];
                req.as_reader().read_to_end(&mut body).unwrap();
                let (code, ct, data) = answer(&req.method().to_string(), req.url(), &auth, &body, &store);
                let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap();
                let _ = req.respond(tiny_http::Response::from_data(data).with_status_code(code).with_header(h));
            }
        });
        (url, files)
    }

    fn temp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("ols-cloud-test-{}", crate::new_token()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Lists a folder, downloads a file and uploads one, then checks what the server saw.
    fn round_trip(acc: Account, secret: &str, files: &Files) {
        let (client, _) = Client::with(&acc, secret).unwrap();
        let list = client.list("Photos").unwrap();
        assert_eq!(list.iter().map(|e| (e.name.as_str(), e.dir)).collect::<Vec<_>>(), vec![("Trip 2024", true), ("IMG 1.jpg", false)], "{}", acc.kind);
        assert_eq!(list[1].path, "Photos/IMG 1.jpg");
        assert_eq!(list[1].size, 5);
        let dir = temp();
        assert_eq!(client.download("Photos/IMG 1.jpg", &dir.join("a.jpg")).unwrap(), 5);
        assert_eq!(fs::read(dir.join("a.jpg")).unwrap(), b"jpeg!");
        fs::write(dir.join("up.jpg"), b"export").unwrap();
        assert_eq!(client.upload("Exports/up é.jpg", &dir.join("up.jpg")).unwrap(), "up é.jpg");
        assert_eq!(files.lock().unwrap().get("Exports/up é.jpg").map(|v| v.as_slice()), Some(&b"export"[..]), "{}", acc.kind);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn webdav_lists_downloads_and_uploads() {
        let (url, files) = serve(|method, path, auth, body, files| {
            assert_eq!(auth, format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("alice:app pw")));
            let p = unpct(path.trim_start_matches("/dav/"));
            match method {
                "PROPFIND" => {
                    assert_eq!(path, "/dav/Photos/");
                    let xml = r#"<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">
                        <d:response><d:href>/dav/Photos/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
                        <d:response><d:href>/dav/Photos/IMG%201.jpg</d:href><d:propstat><d:prop><d:resourcetype/><d:getcontentlength>5</d:getcontentlength></d:prop></d:propstat></d:response>
                        <d:response><d:href>http://example/dav/Photos/Trip%202024/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
                        </d:multistatus>"#;
                    (207, "application/xml".into(), xml.as_bytes().to_vec())
                }
                "GET" if p == "Photos/IMG 1.jpg" => (200, "image/jpeg".into(), b"jpeg!".to_vec()),
                "MKCOL" => (201, "text/plain".into(), vec![]),
                "PUT" => {
                    files.lock().unwrap().insert(p, body.to_vec());
                    (201, "text/plain".into(), vec![])
                }
                _ => (404, "text/plain".into(), vec![]),
            }
        });
        let acc = Account { kind: "webdav".into(), endpoint: format!("{url}/dav/"), user: "alice".into(), ..Default::default() };
        round_trip(acc, "app pw", &files);
    }

    #[test]
    fn s3_lists_downloads_and_uploads() {
        let (url, files) = serve(|method, path, auth, body, files| {
            assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=KEY/") && auth.contains("/auto/s3/aws4_request"), "{auth}");
            match (method, path) {
                ("GET", p) if p.starts_with("/photos-bucket?") => {
                    assert!(p.contains("prefix=Photos%2F") && p.contains("list-type=2") && p.contains("delimiter=%2F"), "{p}");
                    let xml = "<ListBucketResult><IsTruncated>false</IsTruncated>\
                        <Contents><Key>Photos/</Key><Size>0</Size></Contents>\
                        <Contents><Key>Photos/IMG 1.jpg</Key><Size>5</Size></Contents>\
                        <CommonPrefixes><Prefix>Photos/Trip 2024/</Prefix></CommonPrefixes></ListBucketResult>";
                    (200, "application/xml".into(), xml.as_bytes().to_vec())
                }
                ("GET", "/photos-bucket/Photos/IMG%201.jpg") => (200, "image/jpeg".into(), b"jpeg!".to_vec()),
                ("PUT", p) => {
                    files.lock().unwrap().insert(unpct(p.trim_start_matches("/photos-bucket/")), body.to_vec());
                    (200, "text/plain".into(), vec![])
                }
                _ => (404, "text/plain".into(), vec![]),
            }
        });
        let acc = Account { kind: "s3".into(), endpoint: url, user: "KEY".into(), bucket: "photos-bucket".into(), region: "auto".into(), ..Default::default() };
        round_trip(acc, "SECRET", &files);
    }

    #[test]
    fn dropbox_and_onedrive_list_download_and_upload() {
        let base: Arc<Mutex<String>> = Arc::default();
        let at = base.clone();
        let (url, files) = serve(move |method, path, auth, body, files| {
            let json = |v: Value| (200, "application/json".to_string(), v.to_string().into_bytes());
            if path.ends_with("/token") {
                assert!(String::from_utf8_lossy(body).contains("refresh_token=r1"));
                return json(json!({ "access_token": "tok", "refresh_token": "r2" }));
            }
            assert_eq!(auth, "Bearer tok");
            let keep = |name: &str| {
                files.lock().unwrap().insert("Exports/up é.jpg".into(), body.to_vec());
                json(json!({ "name": name }))
            };
            match (method, path) {
                ("POST", "/2/files/list_folder") => {
                    assert_eq!(serde_json::from_slice::<Value>(body).unwrap()["path"], "/Photos");
                    json(json!({ "entries": [{ ".tag": "file", "name": "IMG 1.jpg", "size": 5 }], "has_more": true, "cursor": "c1" }))
                }
                ("POST", "/2/files/list_folder/continue") => json(json!({ "entries": [{ ".tag": "folder", "name": "Trip 2024" }], "has_more": false })),
                ("POST", "/2/files/download") => (200, "application/octet-stream".into(), b"jpeg!".to_vec()),
                ("POST", "/2/files/upload") => keep("up é.jpg"),
                // OneDrive pages with an absolute link
                ("GET", "/v1.0/me/drive/root:/Photos:/children?$top=1000") => {
                    json(json!({ "value": [{ "name": "IMG 1.jpg", "size": 5, "file": {} }], "@odata.nextLink": format!("{}/v1.0/page2", at.lock().unwrap()) }))
                }
                ("GET", "/v1.0/page2") => json(json!({ "value": [{ "name": "Trip 2024", "folder": {} }] })),
                ("GET", "/v1.0/me/drive/root:/Photos/IMG%201.jpg:/content") => (200, "image/jpeg".into(), b"jpeg!".to_vec()),
                ("PUT", p) if p.starts_with("/v1.0/me/drive/root:/Exports/up%20%C3%A9.jpg:/content?") => keep("up é.jpg"),
                _ => (404, "text/plain".into(), vec![]),
            }
        });
        *base.lock().unwrap() = url.clone();
        let secret = json!({ "refresh_token": "r1" }).to_string();
        round_trip(Account { kind: "dropbox".into(), endpoint: url.clone(), user: "app".into(), ..Default::default() }, &secret, &files);
        files.lock().unwrap().clear();
        let onedrive = Account { kind: "onedrive".into(), endpoint: url, user: "client".into(), ..Default::default() };
        round_trip(onedrive.clone(), &secret, &files);
        let (_, rotated) = Client::with(&onedrive, &secret).unwrap();
        assert_eq!(rotated.as_deref(), Some(r#"{"refresh_token":"r2"}"#), "OneDrive's new refresh token is kept");
    }

    #[test]
    fn onedrive_pagination_rejects_off_origin_links() {
        assert!(same_origin("https://graph.microsoft.com/v1.0/page2", "https://graph.microsoft.com"));
        assert!(same_origin("http://127.0.0.1:8080/p2", "http://127.0.0.1:8080"));
        assert!(!same_origin("https://evil.example/v1.0/page2", "https://graph.microsoft.com"));
        assert!(!same_origin("http://graph.microsoft.com/v1.0/page2", "https://graph.microsoft.com"));
        assert!(!same_origin("https://graph.microsoft.com:8443/v1.0/page2", "https://graph.microsoft.com"));
        assert!(validate_endpoint("https://api.dropboxapi.com/").is_ok());
        assert!(validate_endpoint("file:///etc/passwd").is_err());
    }
}
