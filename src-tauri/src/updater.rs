use crate::develop::es;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};
use tauri::{Emitter, Manager};
use tauri_plugin_updater::UpdaterExt;

#[allow(dead_code)]
pub const GITHUB_REPO: &str = "rcwkan/optixel-studio";
pub const RELEASES_API_URL: &str = "https://api.github.com/repos/rcwkan/optixel-studio/releases/latest";
pub const RELEASES_WEB_URL: &str = "https://github.com/rcwkan/optixel-studio/releases";
pub const LICENSE_PURCHASE_URL: &str = "https://optixel.makecade.com/#pricing";
pub const DEFAULT_ENTITLEMENT_ENDPOINT: &str = "https://updates.optixel.studio/v1/license/verify";

static PENDING_PLUGIN_UPDATE: Mutex<Option<tauri_plugin_updater::Update>> = Mutex::new(None);

#[allow(dead_code)]
pub fn releases_page_url() -> &'static str {
    RELEASES_WEB_URL
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub size: u64,
    pub browser_download_url: String,
    pub content_type: String,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubAsset {
    name: String,
    size: u64,
    browser_download_url: String,
    #[serde(default)]
    content_type: String,
    #[serde(default)]
    digest: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct GithubRelease {
    tag_name: String,
    name: Option<String>,
    html_url: String,
    published_at: Option<String>,
    body: Option<String>,
    #[serde(default)]
    assets: Vec<GithubAsset>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub has_update: bool,
    pub release_name: String,
    pub release_tag: String,
    pub release_notes: String,
    pub published_at: String,
    pub html_url: String,
    pub recommended_asset: Option<ReleaseAsset>,
    pub all_assets: Vec<ReleaseAsset>,
    #[serde(default)]
    pub is_plugin_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgressPayload {
    pub status: String, // "downloading" | "completed" | "error"
    pub downloaded: u64,
    pub total: u64,
    pub percent: f32,
    pub path: Option<String>,
    pub error: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseStatus {
    pub key: String,
    pub valid: bool,
    pub tier: String,
    pub email: Option<String>,
    pub expires_at: Option<String>,
    pub message: String,
    pub purchase_url: Option<String>,
}

/// Parse semantic version numbers, ignoring prefix 'v' and suffix build/pre-release metadata.
pub fn parse_version(v: &str) -> Vec<u64> {
    let clean = v.trim().trim_start_matches('v');
    let base = clean.split('-').next().unwrap_or(clean);
    let base = base.split('+').next().unwrap_or(base);
    base.split('.')
        .filter_map(|s| s.parse::<u64>().ok())
        .collect()
}

/// Compare whether latest release is strictly newer than the current version.
pub fn is_newer_version(current: &str, latest: &str) -> bool {
    let curr_parts = parse_version(current);
    let latest_parts = parse_version(latest);
    if curr_parts.is_empty() || latest_parts.is_empty() {
        return false;
    }
    for (l, c) in latest_parts.iter().zip(curr_parts.iter()) {
        if l > c {
            return true;
        } else if l < c {
            return false;
        }
    }
    latest_parts.len() > curr_parts.len()
}

/// Select the most appropriate release asset for the current OS and CPU architecture.
pub fn pick_recommended_asset(assets: &[ReleaseAsset]) -> Option<ReleaseAsset> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match os {
        "macos" => {
            let is_arm = arch == "aarch64" || arch == "arm64";
            if is_arm {
                if let Some(a) = assets.iter().find(|a| {
                    let n = a.name.to_lowercase();
                    n.ends_with(".dmg") && (n.contains("aarch64") || n.contains("arm64"))
                }) {
                    return Some(a.clone());
                }
            } else if let Some(a) = assets.iter().find(|a| {
                let n = a.name.to_lowercase();
                n.ends_with(".dmg") && (n.contains("x86_64") || n.contains("x64") || n.contains("intel"))
            }) {
                return Some(a.clone());
            }

            // Fallback: any .dmg
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".dmg")) {
                return Some(a.clone());
            }

            // Fallback: macOS app tarball
            if let Some(a) = assets.iter().find(|a| {
                let n = a.name.to_lowercase();
                n.ends_with(".app.tar.gz") || (n.ends_with(".tar.gz") && (n.contains("macos") || n.contains("darwin") || n.contains("apple")))
            }) {
                return Some(a.clone());
            }
        }
        "windows" => {
            if let Some(a) = assets.iter().find(|a| {
                let n = a.name.to_lowercase();
                n.ends_with(".exe") && (n.contains("setup") || n.contains("installer") || n.contains("x64"))
            }) {
                return Some(a.clone());
            }
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".exe")) {
                return Some(a.clone());
            }
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".msi")) {
                return Some(a.clone());
            }
        }
        "linux" => {
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".appimage")) {
                return Some(a.clone());
            }
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".deb")) {
                return Some(a.clone());
            }
            if let Some(a) = assets.iter().find(|a| a.name.to_lowercase().ends_with(".rpm")) {
                return Some(a.clone());
            }
        }
        _ => {}
    }

    assets.first().cloned()
}

/// Query GitHub Releases API as a community / development fallback.
pub fn fetch_github_release() -> Result<UpdateInfo, String> {
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_per_read(Some(Duration::from_secs(30)))
        .build()
        .into();

    let mut resp = agent
        .get(RELEASES_API_URL)
        .header("User-Agent", &format!("Optixel-Studio/{current_version}"))
        .header("Accept", "application/vnd.github.v3+json")
        .call()
        .map_err(|e| format!("Failed to query GitHub releases: {e}"))?;

    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().map_err(es)?;
    if status != 200 {
        return Err(format!(
            "GitHub API returned HTTP {status}: {}",
            body.chars().take(200).collect::<String>()
        ));
    }

    let release: GithubRelease =
        serde_json::from_str(&body).map_err(|e| format!("Invalid release JSON from GitHub: {e}"))?;

    let latest_version = release.tag_name.trim().trim_start_matches('v').to_string();
    let has_update = is_newer_version(&current_version, &release.tag_name);

    let all_assets: Vec<ReleaseAsset> = release
        .assets
        .into_iter()
        .map(|a| {
            let sha256 = a
                .digest
                .as_deref()
                .and_then(|d| d.strip_prefix("sha256:"))
                .map(|s| s.to_string());
            ReleaseAsset {
                name: a.name,
                size: a.size,
                browser_download_url: a.browser_download_url,
                content_type: a.content_type,
                sha256,
            }
        })
        .collect();

    let recommended_asset = pick_recommended_asset(&all_assets);

    Ok(UpdateInfo {
        current_version,
        latest_version,
        has_update,
        release_name: release.name.unwrap_or_else(|| release.tag_name.clone()),
        release_tag: release.tag_name,
        release_notes: release.body.unwrap_or_default(),
        published_at: release.published_at.unwrap_or_default(),
        html_url: release.html_url,
        recommended_asset,
        all_assets,
        is_plugin_update: false,
    })
}

/// Download an update asset file, saving it into the Downloads or cache directory and emitting progress events.
pub fn download_update_file(
    app: &tauri::AppHandle,
    url: &str,
    filename: &str,
    expected_sha256: Option<&str>,
) -> Result<PathBuf, String> {
    // Security: verify download URL is from official domains
    let is_official = url.starts_with("https://github.com/rcwkan/optixel-studio/releases/download/")
        || url.starts_with("https://updates.optixel.studio/")
        || url.starts_with("https://optixel-studio-updates.workers.dev/");

    if !is_official {
        return Err("Download URL must originate from official Optixel Studio release servers".into());
    }

    let clean_filename: String = filename
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '_' || *c == '-')
        .collect();
    if clean_filename.is_empty()
        || clean_filename.starts_with('.')
        || clean_filename.starts_with('-')
        || clean_filename.contains("..")
    {
        return Err("Invalid filename for update asset".into());
    }

    let download_dir = dirs::download_dir().unwrap_or_else(|| {
        dirs::data_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join(crate::APP_ID)
            .join("updates")
    });
    let _ = fs::create_dir_all(&download_dir);

    let target_file = download_dir.join(&clean_filename);
    let part_file = download_dir.join(format!("{clean_filename}.part"));

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_per_read(Some(Duration::from_secs(120)))
        .build()
        .into();

    let mut resp = agent
        .get(url)
        .header("User-Agent", "Optixel-Studio-Updater")
        .call()
        .map_err(|e| format!("Failed to start download: {e}"))?;

    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let download_res = (|| -> Result<String, String> {
        let mut file = fs::File::create(&part_file).map_err(es)?;
        let mut reader = resp.body_mut().as_reader();
        let mut sha_ctx = ring::digest::Context::new(&ring::digest::SHA256);
        let mut buf = vec![0u8; 128 * 1024];
        let mut got: u64 = 0;
        let mut last_emit = std::time::Instant::now();

        loop {
            let n = reader.read(&mut buf).map_err(es)?;
            if n == 0 {
                break;
            }
            sha_ctx.update(&buf[..n]);
            file.write_all(&buf[..n]).map_err(es)?;
            got += n as u64;

            if last_emit.elapsed().as_millis() >= 100 || (total > 0 && got >= total) {
                last_emit = std::time::Instant::now();
                let percent = if total > 0 {
                    ((got as f64 / total as f64) * 100.0) as f32
                } else {
                    0.0
                };
                let _ = app.emit(
                    "update-download-progress",
                    DownloadProgressPayload {
                        status: "downloading".into(),
                        downloaded: got,
                        total,
                        percent,
                        path: None,
                        error: None,
                        sha256: None,
                    },
                );
            }
        }

        let computed_sha: String = sha_ctx
            .finish()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        if let Some(expected) = expected_sha256 {
            let exp_clean = expected.trim().to_lowercase();
            if !exp_clean.is_empty() && computed_sha.to_lowercase() != exp_clean {
                return Err(format!(
                    "SHA-256 checksum mismatch! Expected {}, but got {}",
                    exp_clean, computed_sha
                ));
            }
        }

        Ok(computed_sha)
    })();

    let computed_sha = match download_res {
        Ok(hash) => hash,
        Err(e) => {
            let _ = fs::remove_file(&part_file);
            let _ = app.emit(
                "update-download-progress",
                DownloadProgressPayload {
                    status: "error".into(),
                    downloaded: 0,
                    total,
                    percent: 0.0,
                    path: None,
                    error: Some(e.clone()),
                    sha256: None,
                },
            );
            return Err(e);
        }
    };

    if let Err(_e) = fs::rename(&part_file, &target_file) {
        fs::copy(&part_file, &target_file).map_err(es)?;
        let _ = fs::remove_file(&part_file);
    }

    let _ = app.emit(
        "update-download-progress",
        DownloadProgressPayload {
            status: "completed".into(),
            downloaded: total,
            total,
            percent: 100.0,
            path: Some(target_file.to_string_lossy().to_string()),
            error: None,
            sha256: Some(computed_sha),
        },
    );

    Ok(target_file)
}

/// Open/execute the downloaded package or installer according to the host platform.
pub fn launch_installer(path: &Path) -> Result<(), String> {
    if !path.exists() || !path.is_file() {
        return Err(format!("Installer file not found or is not a regular file: {}", path.display()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let valid_ext = matches!(
        ext.as_str(),
        "dmg" | "pkg" | "app" | "exe" | "msi" | "deb" | "rpm" | "appimage"
    );
    if !valid_ext {
        return Err(format!("Unsupported or unsafe installer file type: .{ext}"));
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| format!("Failed to open DMG installer: {e}"))?;
        let _ = std::process::Command::new("open").arg("-R").arg(path).spawn();
    }

    #[cfg(target_os = "windows")]
    {
        if ext == "msi" {
            std::process::Command::new("msiexec")
                .args(["/i", &path.to_string_lossy()])
                .spawn()
                .map_err(|e| format!("Failed to launch MSI installer: {e}"))?;
        } else {
            std::process::Command::new(path)
                .spawn()
                .map_err(|e| format!("Failed to launch Windows installer: {e}"))?;
        }
    }

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = fs::set_permissions(path, perms);
        }
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| format!("Failed to launch installer: {e}"))?;
    }

    Ok(())
}

/// Safely open an external HTTP or HTTPS URL in the user's default web browser.
pub fn open_browser_url(url: &str) -> Result<(), String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("Only http and https URLs can be opened".into());
    }

    if url.chars().any(|c| c.is_control() || "&|;\"'<>^%`$\\".contains(c)) {
        return Err("Invalid characters detected in URL".into());
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|e| format!("Could not open browser: {e}"))?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn()
            .map_err(|e| format!("Could not open browser: {e}"))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|e| format!("Could not open browser: {e}"))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tauri Commands
// ---------------------------------------------------------------------------

#[tauri::command(async)]
pub async fn check_for_updates(app: tauri::AppHandle) -> Result<UpdateInfo, String> {
    let current_version = env!("CARGO_PKG_VERSION").to_string();

    // Check user license key from Core settings or environment
    let license_key = {
        let core = app.try_state::<crate::St>();
        core.and_then(|c| c.settings.lock().ok().and_then(|s| s.license_key.clone()))
            .or_else(|| std::env::var("OPTIXEL_LICENSE_KEY").ok())
            .or_else(|| std::env::var("OPENLIGHT_LICENSE_KEY").ok())
    };

    // Attempt cryptographic update check via tauri-plugin-updater
    let updater_res = {
        let b = app.updater_builder();
        if let Some(ref key) = license_key {
            let clean = key.trim();
            if !clean.is_empty() {
                b.header("Authorization", format!("Bearer {clean}"))
                    .and_then(|b| b.header("X-Optixel-License-Key", clean))
                    .and_then(|b| b.build())
            } else {
                b.build()
            }
        } else {
            b.build()
        }
    };

    if let Ok(updater) = updater_res {
        match updater.check().await {
            Ok(Some(update)) => {
                let latest_version = update.version.clone();
                let release_notes = update.body.clone().unwrap_or_default();
                let published_at = update
                    .date
                    .map(|d| d.to_string())
                    .unwrap_or_default();

                *PENDING_PLUGIN_UPDATE.lock().unwrap() = Some(update);

                return Ok(UpdateInfo {
                    current_version: current_version.clone(),
                    latest_version: latest_version.clone(),
                    has_update: true,
                    release_name: format!("Optixel Studio v{latest_version}"),
                    release_tag: format!("v{latest_version}"),
                    release_notes,
                    published_at,
                    html_url: LICENSE_PURCHASE_URL.into(),
                    recommended_asset: Some(ReleaseAsset {
                        name: format!("Optixel-Studio-{latest_version}-update"),
                        size: 0,
                        browser_download_url: "tauri-plugin-updater://in-app".into(),
                        content_type: "application/octet-stream".into(),
                        sha256: None,
                    }),
                    all_assets: vec![],
                    is_plugin_update: true,
                });
            }
            Ok(None) => {
                *PENDING_PLUGIN_UPDATE.lock().unwrap() = None;
                return Ok(UpdateInfo {
                    current_version: current_version.clone(),
                    latest_version: current_version.clone(),
                    has_update: false,
                    release_name: format!("Optixel Studio v{current_version}"),
                    release_tag: format!("v{current_version}"),
                    release_notes: "You are running the latest version of Optixel Studio.".into(),
                    published_at: String::new(),
                    html_url: RELEASES_WEB_URL.into(),
                    recommended_asset: None,
                    all_assets: vec![],
                    is_plugin_update: true,
                });
            }
            Err(e) => {
                let err_msg = e.to_string();
                // If the error indicates entitlement denial (HTTP 402/403)
                if err_msg.contains("402") || err_msg.contains("Payment Required") || err_msg.contains("expired") {
                    return Err(format!(
                        "Active updates require a valid Optixel Studio license. Please renew or purchase at: {LICENSE_PURCHASE_URL}"
                    ));
                }
                // Fall back to GitHub public release check on network/server failure
            }
        }
    }

    // Community / fallback updater check
    fetch_github_release()
}

#[tauri::command(async)]
pub async fn download_update(
    app: tauri::AppHandle,
    url: String,
    filename: String,
    expected_sha256: Option<String>,
) -> Result<String, String> {
    // If this is an in-app signed update managed by tauri-plugin-updater
    if url == "tauri-plugin-updater://in-app" {
        let update_opt = PENDING_PLUGIN_UPDATE.lock().unwrap().take();
        if let Some(update) = update_opt {
            let app_handle = app.clone();
            let mut total_downloaded: u64 = 0;

            update
                .download_and_install(
                    move |chunk_len, content_len| {
                        total_downloaded += chunk_len as u64;
                        let total = content_len.unwrap_or(0);
                        let percent = if total > 0 {
                            ((total_downloaded as f64 / total as f64) * 100.0) as f32
                        } else {
                            0.0
                        };
                        let _ = app_handle.emit(
                            "update-download-progress",
                            DownloadProgressPayload {
                                status: "downloading".into(),
                                downloaded: total_downloaded,
                                total,
                                percent,
                                path: None,
                                error: None,
                                sha256: None,
                            },
                        );
                    },
                    move || {
                        let _ = app.emit(
                            "update-download-progress",
                            DownloadProgressPayload {
                                status: "completed".into(),
                                downloaded: total_downloaded,
                                total: total_downloaded,
                                percent: 100.0,
                                path: Some("installed".into()),
                                error: None,
                                sha256: None,
                            },
                        );
                    },
                )
                .await
                .map_err(|e| format!("Failed to download and install update: {e}"))?;

            return Ok("Update downloaded and installed. Please restart Optixel Studio.".into());
        }
    }

    download_update_file(&app, &url, &filename, expected_sha256.as_deref())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command(async)]
pub fn install_update(path: String) -> Result<(), String> {
    if path == "installed" {
        return Ok(());
    }
    launch_installer(Path::new(&path))
}

#[tauri::command(async)]
pub fn open_external_url(url: String) -> Result<(), String> {
    open_browser_url(&url)
}

#[tauri::command(async)]
pub fn verify_license_key(key: String) -> Result<LicenseStatus, String> {
    let clean = key.trim();
    if clean.is_empty() {
        return Ok(LicenseStatus {
            key: String::new(),
            valid: false,
            tier: "free".into(),
            email: None,
            expires_at: None,
            message: "No license key provided.".into(),
            purchase_url: Some(LICENSE_PURCHASE_URL.into()),
        });
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(5)))
        .timeout_per_read(Some(Duration::from_secs(10)))
        .build()
        .into();

    let url = format!("{DEFAULT_ENTITLEMENT_ENDPOINT}?key={clean}");
    match agent.get(&url).header("User-Agent", "Optixel-Studio").call() {
        Ok(mut resp) => {
            let body = resp.body_mut().read_to_string().map_err(es)?;
            if let Ok(status) = serde_json::from_str::<LicenseStatus>(&body) {
                return Ok(status);
            }
        }
        Err(_) => {
            // Offline or uncontactable: fallback check for well-formed key pattern
            if (clean.starts_with("OPT-") || clean.starts_with("OLS-")) && clean.len() >= 16 {
                return Ok(LicenseStatus {
                    key: clean.into(),
                    valid: true,
                    tier: "pro".into(),
                    email: None,
                    expires_at: None,
                    message: "Offline license active. Auto-updates will resume when connected.".into(),
                    purchase_url: None,
                });
            }
        }
    }

    Ok(LicenseStatus {
        key: clean.into(),
        valid: false,
        tier: "invalid".into(),
        email: None,
        expires_at: None,
        message: "Invalid or unverified license key.".into(),
        purchase_url: Some(LICENSE_PURCHASE_URL.into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parsing_and_comparison() {
        assert_eq!(parse_version("0.2.1"), vec![0, 2, 1]);
        assert_eq!(parse_version("v0.2.1"), vec![0, 2, 1]);
        assert_eq!(parse_version("v1.0.0-beta.1"), vec![1, 0, 0]);

        assert!(!is_newer_version("0.2.1", "0.2.1"));
        assert!(!is_newer_version("0.2.1", "v0.2.1"));
        assert!(!is_newer_version("0.2.1", "0.2.0"));
        assert!(!is_newer_version("0.2.1", "0.1.9"));

        assert!(is_newer_version("0.2.1", "0.2.2"));
        assert!(is_newer_version("0.2.1", "v0.2.2"));
        assert!(is_newer_version("0.2.1", "0.3.0"));
        assert!(is_newer_version("0.2.1", "1.0.0"));
        assert!(is_newer_version("0.2.1", "0.2.1.1"));
    }

    #[test]
    fn asset_recommendation_logic() {
        let assets = vec![
            ReleaseAsset {
                name: "Optixel.Studio_0.3.0_aarch64.dmg".into(),
                size: 1000,
                browser_download_url: "https://example.com/mac-arm.dmg".into(),
                content_type: "application/zip".into(),
                sha256: None,
            },
            ReleaseAsset {
                name: "Optixel.Studio_0.3.0_x64-setup.exe".into(),
                size: 1000,
                browser_download_url: "https://example.com/win-setup.exe".into(),
                content_type: "application/zip".into(),
                sha256: None,
            },
            ReleaseAsset {
                name: "Optixel.Studio_0.3.0_amd64.AppImage".into(),
                size: 1000,
                browser_download_url: "https://example.com/linux.AppImage".into(),
                content_type: "application/zip".into(),
                sha256: None,
            },
        ];

        let rec = pick_recommended_asset(&assets);
        assert!(rec.is_some());
        let name = rec.unwrap().name;
        #[cfg(target_os = "macos")]
        assert!(name.ends_with(".dmg"));
        #[cfg(target_os = "windows")]
        assert!(name.ends_with(".exe"));
        #[cfg(target_os = "linux")]
        assert!(name.ends_with(".AppImage"));
    }

    #[test]
    fn offline_license_verification_format() {
        let valid_key = "OLS-PRO-9988-7766-5544";
        let status = verify_license_key(valid_key.into()).unwrap();
        assert!(status.valid);
        assert_eq!(status.tier, "pro");

        let empty_status = verify_license_key("".into()).unwrap();
        assert!(!empty_status.valid);
    }
}
