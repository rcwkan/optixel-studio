//! Pluggable AI & CLI agent provider router (spec §5).
//!
//! CLI agents (Claude CLI, Antigravity, custom) get a snapshot of the photo plus an MCP
//! config and drive the edit themselves through the built-in MCP server; the GUI updates
//! live as their tool calls land. HTTP providers (Ollama, OpenAI-compatible, Anthropic,
//! Gemini) return an `AiTuningResponse` JSON that we validate, clamp and apply.

use crate::{
    Core,
    develop::{self, Opts, es},
    mcp,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Provider {
    /// e.g. `cli:claude`, `ollama:qwen2.5vl`, `api:anthropic`
    pub id: String,
    /// cli | ollama | openai | anthropic | gemini
    pub kind: String,
    pub name: String,
    /// cli: argv template ({{AGENT_PROMPT}}, {{USER_PROMPT}}, {{IMAGE_PATH}}, {{TELEMETRY_PATH}},
    /// {{HISTOGRAM_JSON}}, {{CURRENT_XMP_JSON}}, {{MCP_CONFIG_PATH}}, {{MCP_SSE_URL}}, {{MCP_TOKEN}}, {{PHOTO_ID}})
    pub command: String,
    pub endpoint: String,
    pub model: String,
    /// API key stored in the OS keychain under this provider id.
    pub has_key: bool,
    pub timeout_s: u64,
    /// HTTP providers: visual self-correction passes.
    pub max_passes: u32,
}

impl Default for Provider {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: "cli".into(),
            name: String::new(),
            command: String::new(),
            endpoint: String::new(),
            model: String::new(),
            has_key: false,
            timeout_s: 600,
            max_passes: 2,
        }
    }
}

pub fn default_providers() -> Vec<Provider> {
    let p = |id: &str, kind: &str, name: &str, command: &str, endpoint: &str, model: &str| Provider {
        id: id.into(),
        kind: kind.into(),
        name: name.into(),
        command: command.into(),
        endpoint: endpoint.into(),
        model: model.into(),
        ..Provider::default()
    };
    vec![
        p("cli:claude", "cli", "Claude CLI", "claude -p {{AGENT_PROMPT}} --mcp-config {{MCP_CONFIG_PATH}} --allowedTools mcp__optixel --max-turns 20 --output-format json", "", ""),
        p("cli:antigravity", "cli", "Antigravity CLI", "antigravity -p {{AGENT_PROMPT}} --dangerously-skip-permissions", "", ""),
        p("cli:custom", "cli", "Custom CLI agent", "", "", ""),
        p("ollama:qwen2.5vl", "ollama", "Ollama (local)", "", "http://127.0.0.1:11434", "qwen2.5vl:7b"),
        p("local-api:lmstudio", "openai", "LM Studio / vLLM / llama.cpp", "", "http://127.0.0.1:1234/v1", "local-model"),
        p("api:anthropic", "anthropic", "Anthropic API", "", "https://api.anthropic.com", "claude-opus-5-5"),
        p("api:gemini", "gemini", "Google Gemini API", "", "https://generativelanguage.googleapis.com", "gemini-2.5-flash"),
        p("api:openai", "openai", "OpenAI API", "", "https://api.openai.com/v1", "gpt-4o"),
        p("api:openrouter", "openai", "OpenRouter", "", "https://openrouter.ai/api/v1", "openrouter/auto"),
    ]
}

fn creds_path() -> PathBuf {
    crate::data_dir().join("credentials.json")
}

fn load_creds() -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(creds_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_creds(creds: &std::collections::HashMap<String, String>) -> Result<(), String> {
    let path = creds_path();
    let s = serde_json::to_string_pretty(creds).map_err(es)?;
    std::fs::write(&path, s).map_err(es)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

pub fn set_key(id: &str, key: &str) -> Result<(), String> {
    let mut creds = load_creds();
    if key.is_empty() {
        creds.remove(id);
    } else {
        creds.insert(id.to_string(), key.to_string());
    }
    save_creds(&creds)
}

pub fn key(id: &str) -> Option<String> {
    load_creds().get(id).cloned()
}

/// Keychain first, then the usual environment variable for that vendor.
fn get_key(p: &Provider) -> Option<String> {
    if p.has_key
        && let Some(k) = key(&p.id)
    {
        return Some(k);
    }
    let var = match p.id.as_str() {
        "api:anthropic" => "ANTHROPIC_API_KEY",
        "api:gemini" => "GEMINI_API_KEY",
        "api:openai" => "OPENAI_API_KEY",
        "api:openrouter" => "OPENROUTER_API_KEY",
        _ => return None,
    };
    std::env::var(var).ok().filter(|k| !k.is_empty())
}

/// Apps launched from the Dock get a minimal PATH; CLI agents usually live in the
/// user's login-shell PATH (~/.local/bin, Homebrew, npm globals).
pub(crate) fn user_path() -> &'static str {
    static PATH: OnceLock<String> = OnceLock::new();
    PATH.get_or_init(|| {
        let current = std::env::var("PATH").unwrap_or_default();
        #[cfg(unix)]
        {
            let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
            if let Ok(o) = Command::new(shell).args(["-lc", "printf %s \"$PATH\""]).stdin(Stdio::null()).output()
                && o.status.success()
                && !o.stdout.is_empty()
            {
                return String::from_utf8_lossy(&o.stdout).into_owned();
            }
        }
        current
    })
}

pub fn find_binary(prog: &str) -> PathBuf {
    let p = std::path::Path::new(prog);
    if p.is_absolute() || p.components().count() > 1 {
        return p.to_path_buf();
    }
    let aliases: &[&str] = match prog {
        "antigravity" => &["antigravity", "agy"],
        "agy" => &["agy", "antigravity"],
        _ => &[prog],
    };
    for name in aliases {
        for dir in std::env::split_paths(user_path()) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        if let Some(home) = dirs::home_dir() {
            for dir in [
                home.join(".local/bin"),
                home.join(".cargo/bin"),
                PathBuf::from("/opt/homebrew/bin"),
                PathBuf::from("/usr/local/bin"),
                home.join(".gemini/bin"),
            ] {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    p.to_path_buf()
}

/// Files and data handed to a provider for one pass.
struct Snapshot {
    dir: PathBuf,
    image: PathBuf,
    telemetry_path: PathBuf,
    jpeg_b64: String,
    telemetry: Value,
    state: develop::Develop,
}

/// The preview is the full uncropped frame, so coordinates the model reads off it are mask coordinates.
fn snapshot(core: &Core, id: i64, dir: &PathBuf) -> Result<Snapshot, String> {
    let img = core.render(id, &Opts { full_frame: true, ..Default::default() })?;
    let jpeg = develop::encode_jpeg(&img, 1536, 85)?;
    let telemetry = develop::telemetry(&img);
    let state = core.develop(id)?;
    let (image, telemetry_path) = (dir.join("preview.jpg"), dir.join("telemetry.json"));
    std::fs::write(&image, &jpeg).map_err(es)?;
    std::fs::write(&telemetry_path, telemetry.to_string()).map_err(es)?;
    std::fs::write(dir.join("state.json"), serde_json::to_string_pretty(&state).map_err(es)?).map_err(es)?;
    Ok(Snapshot {
        dir: dir.clone(),
        image,
        telemetry_path,
        jpeg_b64: base64::engine::general_purpose::STANDARD.encode(jpeg),
        telemetry,
        state,
    })
}

/// 32-bin luma histogram + stats; keeps prompts small for HTTP models.
fn compact(t: &Value) -> Value {
    let bins: Vec<u64> = t["histogram"]["luma"]
        .as_array()
        .map(|a| a.chunks(8).map(|c| c.iter().filter_map(Value::as_u64).sum()).collect())
        .unwrap_or_default();
    json!({ "luma_histogram_32": bins, "clipping": t["clipping"], "luma": t["luma"], "dominant_oklch": t["dominant_oklch"] })
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n).collect::<String>()) }
}

fn provider(core: &Core, id: &str) -> Result<Provider, String> {
    let s = core.settings.lock().unwrap();
    s.providers.iter().find(|p| p.id == id).cloned().ok_or_else(|| format!("provider '{id}' is not configured"))
}

fn session_dir(id: i64) -> Result<PathBuf, String> {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("optixel-session-{id}-{ts}"));
    std::fs::create_dir_all(&dir).map_err(es)?;
    Ok(dir)
}

/// Run one AI request against a photo. Returns a human-readable summary.
pub fn run(core: &Arc<Core>, id: i64, prompt: &str, provider_id: &str) -> Result<String, String> {
    let p = provider(core, provider_id)?;
    let dir = session_dir(id)?;
    core.log(&p.id, format!("photo {id}: {}", truncate(prompt, 120)));
    let res = if p.kind == "cli" { run_cli(core, &p, id, prompt, &dir, None) } else { run_http(core, &p, id, prompt, &dir) };
    match &res {
        Ok(s) => core.log(&p.id, format!("done: {}", truncate(s, 400))),
        Err(e) => core.log(&p.id, format!("error: {e}")),
    }
    let _ = std::fs::remove_dir_all(&dir);
    res
}

const TRACE: &str = "Identify the one object under the ring (the whole object: the whole castle, one person, one car) and \
    trace its outline. If the clicked object is in the background (e.g. a castle, building, mountain, wall, or sky) and a \
    foreground subject (e.g. a person, group of people, face, hair, or pet) stands in front of it, route the outline carefully \
    around the entire foreground subject (including their head, hair, and body) so the foreground subject is completely excluded \
    from the background selection. Reply with only a JSON object {\"name\": \"<short name>\", \"points\": [[x, y], ...]}: 20-80 points \
    in order around the object's visible edge, following it closely, as [x, y] fractions of the full frame, (0,0) = top-left.";

/// AI selection, step 1: the provider looks at the full frame with the click marked and returns
/// the name and outline of the object under it. Nothing is edited.
pub fn trace(core: &Arc<Core>, id: i64, at: [f32; 2], provider_id: &str) -> Result<(String, Vec<[f32; 2]>), String> {
    let p = provider(core, provider_id)?;
    let dir = session_dir(id)?;
    let [x, y] = at;
    core.log(&p.id, format!("photo {id}: select the object at [{x:.3}, {y:.3}]"));
    let res = (|| {
        let reply = if p.kind == "cli" {
            let agent = format!(
                "You are selecting an object in a photo in Optixel Studio through its MCP server \"optixel\". Photo id: {id}. \
                 The user clicked at [{x:.4}, {y:.4}] (fractions of the full frame). Call photo_inspect (photo_id \"{id}\", \
                 full_frame true, max_edge 1536, mark [{x:.4}, {y:.4}]): a red ring marks the click. Do not change the photo. {TRACE}"
            );
            run_cli(core, &p, id, &agent, &dir, Some(agent.clone()))?
        } else {
            let mut img = core.render(id, &Opts { full_frame: true, ..Default::default() })?;
            develop::mark(&mut img, at);
            let jpeg = base64::engine::general_purpose::STANDARD.encode(develop::encode_jpeg(&img, 1536, 85)?);
            let user = format!("The user clicked at [{x:.4}, {y:.4}]; a red ring marks the spot. {TRACE}");
            call_model(&p, "You outline objects in photos for a photo editor.", &user, &jpeg, &outline_schema())?
        };
        let v = parse_object(&reply)?;
        let points: Vec<[f32; 2]> = serde_json::from_value(v["points"].clone()).map_err(|e| format!("the AI's outline is not [[x, y], ...]: {e}"))?;
        develop::check_outline(&points)?;
        Ok((v["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("object").to_string(), points))
    })();
    match &res {
        Ok((name, pts)) => core.log(&p.id, format!("selected {name} ({} points)", pts.len())),
        Err(e) => core.log(&p.id, format!("error: {e}")),
    }
    let _ = std::fs::remove_dir_all(&dir);
    res
}

/// AI selection, step 2: each traced object starts its own selection (or is subtracted from `mask` on ⌥-click).
/// Returns the new state and the selection's index.
pub fn select(core: &Arc<Core>, id: i64, at: [f32; 2], provider_id: &str, mask: Option<usize>, subtract: bool) -> Result<(develop::Develop, usize), String> {
    let (name, points) = trace(core, id, at, provider_id)?;
    // read after the AI call so edits made while it ran are kept
    let mut d = core.develop(id)?;
    let op = if subtract { "subtract" } else { "add" };
    let c = develop::MaskComponent { op: op.into(), kind: "polygon".into(), name: name.clone(), points, ..Default::default() };
    let i = match mask.filter(|&i| subtract && i < d.masks.len()) {
        Some(i) => {
            d.masks[i].components.push(c);
            i
        }
        None => {
            let ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
            let name = if subtract { format!("all but {name}") } else { name.clone() };
            d.masks.push(develop::Mask { mask_id: format!("mask_{ms:x}"), name, components: vec![c], ..Default::default() });
            d.masks.len() - 1
        }
    };
    let label = format!("AI select: {}{name}", if subtract { "− " } else { "" });
    d.clamp();
    d.ai_provenance_log.push(mcp::provenance(core, provider_id, &label));
    core.commit(id, &d, &label)?;
    Ok((d, i))
}

/// Ask the provider about a photo without editing it: HTTP providers see the full-frame preview, CLI agents
/// look through photo_inspect. Returns the JSON object it replies with (`schema` describes it).
fn ask_json(core: &Arc<Core>, id: i64, provider_id: &str, task: &str, schema: &Value) -> Result<Value, String> {
    let p = provider(core, provider_id)?;
    let dir = session_dir(id)?;
    core.log(&p.id, format!("photo {id}: {}", truncate(task, 120)));
    let res = (|| {
        let reply = if p.kind == "cli" {
            let agent = format!(
                "You are looking at a photo in Optixel Studio through its MCP server \"optixel\". Photo id: {id}. Call \
                 photo_inspect (photo_id \"{id}\", full_frame true, max_edge 1024) to see it. Do not change the photo. {task} \
                 Reply with only a JSON object matching this schema: {schema}"
            );
            run_cli(core, &p, id, &agent, &dir, Some(agent.clone()))?
        } else {
            let img = core.render(id, &Opts { full_frame: true, ..Default::default() })?;
            let jpeg = base64::engine::general_purpose::STANDARD.encode(develop::encode_jpeg(&img, 1024, 85)?);
            call_model(&p, "You analyse photos for a photo editor and reply with JSON only.", task, &jpeg, schema)?
        };
        parse_object(&reply)
    })();
    if let Err(e) = &res {
        core.log(&p.id, format!("error: {e}"));
    }
    let _ = std::fs::remove_dir_all(&dir);
    res
}

/// Box [x, y, w, h] (fractions of the full frame) around `what`, e.g. "the main subject".
pub fn locate(core: &Arc<Core>, id: i64, what: &str, provider_id: &str) -> Result<[f32; 4], String> {
    let schema = json!({ "type": "object", "required": ["box"], "properties": { "box": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 } } });
    let task = format!(
        "Find {what}. Reply with {{\"box\": [x0, y0, x1, y1]}}: its bounding box as fractions of the full frame, (0,0) = top-left. \
         If there is none, reply with the box of the most prominent thing."
    );
    let v = ask_json(core, id, provider_id, &task, &schema)?;
    let b: [f32; 4] = serde_json::from_value(v["box"].clone()).map_err(|e| format!("the AI's box is not [x0, y0, x1, y1]: {e}"))?;
    let [x0, y0, x1, y1] = b.map(|v| v.clamp(0.0, 1.0));
    if x1 - x0 < 0.01 || y1 - y0 < 0.01 {
        return Err("the AI's box is empty".into());
    }
    Ok([x0, y0, x1 - x0, y1 - y0])
}

/// A traced object: its name and outline as fractions of the full frame.
pub type Outline = (String, Vec<[f32; 2]>);

/// Outlines of `what` (e.g. "the sky", "each person"), one per object. Empty when there is none.
pub fn outline(core: &Arc<Core>, id: i64, what: &str, provider_id: &str) -> Result<Vec<Outline>, String> {
    let schema = json!({ "type": "object", "required": ["objects"], "properties": { "objects": { "type": "array", "items": outline_schema() } } });
    let task = format!(
        "Trace {what}. Reply with {{\"objects\": [{{\"name\": \"<short name>\", \"points\": [[x, y], ...]}}]}}: one entry per \
         object, 20-80 points in order around its visible edge, following it closely, as [x, y] fractions of the full frame, \
         (0,0) = top-left. Reply with {{\"objects\": []}} if there is none."
    );
    let v = ask_json(core, id, provider_id, &task, &schema)?;
    let mut out = vec![];
    for o in v["objects"].as_array().into_iter().flatten() {
        let points: Vec<[f32; 2]> = serde_json::from_value(o["points"].clone()).map_err(|e| format!("the AI's outline is not [[x, y], ...]: {e}"))?;
        develop::check_outline(&points)?;
        out.push((o["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("object").to_string(), points));
    }
    Ok(out)
}

/// 3-8 lowercase keywords for the photo's subjects, scene and setting.
pub fn describe(core: &Arc<Core>, id: i64, provider_id: &str) -> Result<Vec<String>, String> {
    let schema = json!({ "type": "object", "required": ["keywords"], "properties": { "keywords": { "type": "array", "items": { "type": "string" } } } });
    let task = "List 3 to 8 short lowercase keywords for this photo: its main subjects, the scene and the setting (e.g. dog, beach, \
                sunset). Reply with {\"keywords\": [...]}.";
    let v = ask_json(core, id, provider_id, task, &schema)?;
    Ok(v["keywords"].as_array().into_iter().flatten().filter_map(|k| k.as_str()).map(|k| truncate(k.trim(), 40).to_lowercase()).filter(|k| !k.is_empty()).take(8).collect())
}

/// Strips ANSI escape sequences (CSI and OSC) from CLI agent output.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if (0x40..=0x7e).contains(&(next as u32)) {
                            break;
                        }
                    }
                }
                Some(&']') => {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next == '\x07' || next == '\x1b' {
                            break;
                        }
                    }
                }
                _ => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `agent`: the agent prompt; None builds the editing prompt for `prompt`.
fn run_cli(core: &Arc<Core>, p: &Provider, id: i64, prompt: &str, dir: &PathBuf, agent: Option<String>) -> Result<String, String> {
    let (enabled, url, token) = {
        let s = core.settings.lock().unwrap();
        (s.mcp_enabled, format!("http://127.0.0.1:{}/mcp", s.mcp_port), s.mcp_token.clone())
    };
    if !enabled || *core.mcp_status.lock().unwrap() != "running" {
        return Err("CLI agents edit through the MCP server; enable it in Agent & MCP settings (restart may be needed)".into());
    }
    if p.command.trim().is_empty() {
        return Err(format!("{} has no command template", p.name));
    }
    let snap = snapshot(core, id, dir)?;
    let mcp_config = dir.join("mcp.json");
    let cfg = json!({ "mcpServers": {
        "optixel": { "type": "http", "url": url, "headers": { "Authorization": format!("Bearer {token}") } },
        "openlight": { "type": "http", "url": url, "headers": { "Authorization": format!("Bearer {token}") } }
    } });
    std::fs::write(&mcp_config, cfg.to_string()).map_err(es)?;
    let agent_prompt = agent.unwrap_or_else(|| format!(
        "You are editing a photo in Optixel Studio through its MCP server \"optixel\". Photo id: {id}. \
         A rendered preview is at {} and histogram telemetry at {}. User goal: \"{prompt}\". \
         Call photo_inspect (photo_id \"{id}\", full_frame true) to see the image, apply non-destructive edits with develop_apply_parameters \
         and mask_create_and_edit, then photo_inspect again to verify (for example zero highlight clipping) and iterate if needed. \
         To select an object or area (\"select the castle\"), create a separate mask for each object named after it with one polygon operation tracing its outline \
         in the full-frame image, and leave local_adjustments at 0 unless an adjustment was asked for; check it with photo_inspect show_mask \
         (the mask's index) and fix the outline if it misses or spills. To remove objects (\"remove the people\"), call retouch_remove_objects \
         with a generous outline per object, then inspect the result. \
         Finish with a one-paragraph summary of what you changed.",
        snap.image.display(),
        snap.telemetry_path.display()
    ));
    let vars: [(&str, String); 10] = [
        ("AGENT_PROMPT", agent_prompt),
        ("USER_PROMPT", prompt.to_string()),
        ("IMAGE_PATH", snap.image.display().to_string()),
        ("TELEMETRY_PATH", snap.telemetry_path.display().to_string()),
        ("HISTOGRAM_JSON", snap.telemetry.to_string()),
        ("CURRENT_XMP_JSON", serde_json::to_string(&snap.state).map_err(es)?),
        ("MCP_CONFIG_PATH", mcp_config.display().to_string()),
        ("MCP_SSE_URL", url.clone()),
        ("MCP_TOKEN", token.clone()),
        ("PHOTO_ID", id.to_string()),
    ];
    // Split first, substitute second: prompt text never reaches a shell.
    let argv: Vec<String> = shlex::split(&p.command)
        .ok_or("command template has unbalanced quotes")?
        .into_iter()
        .map(|w| vars.iter().fold(w, |w, (k, v)| w.replace(&format!("{{{{{k}}}}}"), v)))
        .collect();
    let (prog, args) = argv.split_first().ok_or("empty command")?;
    let prog_path = find_binary(prog);
    core.log(&p.id, format!("running {} in {}", prog_path.display(), snap.dir.display()));
    let mut child = Command::new(&prog_path)
        .args(args)
        .current_dir(&snap.dir)
        .env("PATH", user_path())
        .env("OPTIXEL_MCP_URL", &url)
        .env("OPTIXEL_MCP_TOKEN", &token)
        .env("OPTIXEL_PHOTO_ID", id.to_string())
        .env("OPENLIGHT_MCP_URL", &url)
        .env("OPENLIGHT_MCP_TOKEN", &token)
        .env("OPENLIGHT_PHOTO_ID", id.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start '{prog}' (resolved to '{}'): {e}", prog_path.display()))?;
    let stream_lines = |core: Arc<Core>, source: String, r: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut full = String::new();
            if let Some(r) = r {
                let mut reader = BufReader::new(r);
                let mut buf = Vec::new();
                while let Ok(n) = reader.read_until(b'\n', &mut buf) {
                    if n == 0 {
                        break;
                    }
                    let line = String::from_utf8_lossy(&buf);
                    full.push_str(&line);
                    let mut last_chunk = None;
                    for chunk in line.split('\r') {
                        let cleaned = strip_ansi(chunk);
                        let trimmed = cleaned.trim();
                        if !trimmed.is_empty() {
                            last_chunk = Some(trimmed.to_string());
                        }
                    }
                    if let Some(msg) = last_chunk {
                        core.log(&source, msg);
                    }
                    buf.clear();
                }
            }
            full
        })
    };
    let out = stream_lines(core.clone(), p.id.clone(), child.stdout.take().map(|r| Box::new(r) as Box<dyn Read + Send>));
    let err = stream_lines(core.clone(), p.id.clone(), child.stderr.take().map(|r| Box::new(r) as Box<dyn Read + Send>));
    let deadline = Instant::now() + Duration::from_secs(p.timeout_s.max(10));
    let status = loop {
        if let Some(s) = child.try_wait().map_err(es)? {
            break s;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err(format!("{prog} timed out after {}s", p.timeout_s));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let (out, err) = (out.join().unwrap_or_default(), err.join().unwrap_or_default());
    if !status.success() {
        let msg = if !err.trim().is_empty() { err.trim() } else { out.trim() };
        return Err(format!("{prog} exited with {status}: {}", truncate(msg, 600)));
    }
    // `claude --output-format json` wraps the final message in {"result": ...}
    Ok(serde_json::from_str::<Value>(out.trim())
        .ok()
        .and_then(|v| v.get("result").and_then(Value::as_str).map(String::from))
        .unwrap_or_else(|| out.trim().to_string()))
}

fn response_schema() -> Value {
    let tools = mcp::tools();
    let props = |name: &str| {
        tools.as_array().and_then(|a| a.iter().find(|t| t["name"] == name)).map(|t| t["inputSchema"]["properties"].clone()).unwrap_or_default()
    };
    let (mask, remove) = (props("mask_create_and_edit"), props("retouch_remove_objects"));
    json!({
        "type": "object",
        "required": ["reasoning_summary", "global_parameter_patch", "mask_operations", "requires_another_visual_pass"],
        "properties": {
            "reasoning_summary": { "type": "string" },
            "global_parameter_patch": props("develop_apply_parameters")["parameters"],
            "mask_operations": { "type": "array", "items": { "type": "object", "properties": {
                "mask_name": mask["mask_name"], "operations": mask["operations"], "local_adjustments": mask["local_adjustments"] } } },
            "remove_objects": remove["objects"],
            "requires_another_visual_pass": { "type": "boolean" }
        }
    })
}

/// One traced object, `{name, points}` (the retouch tool's item schema).
fn outline_schema() -> Value {
    let tools = mcp::tools();
    let tool = tools.as_array().and_then(|a| a.iter().find(|t| t["name"] == "retouch_remove_objects"));
    tool.map(|t| t["inputSchema"]["properties"]["objects"]["items"].clone()).unwrap_or_default()
}

fn parse_object(s: &str) -> Result<Value, String> {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b > a => serde_json::from_str(&s[a..=b]).map_err(|e| format!("model returned invalid JSON: {e}")),
        _ => Err(format!("model reply had no JSON object: {}", truncate(s, 300))),
    }
}

/// `schema`: the reply's JSON schema (enforced by Ollama; the others are told in the prompt).
fn call_model(p: &Provider, system: &str, user: &str, image_b64: &str, schema: &Value) -> Result<String, String> {
    let key = get_key(p);
    let ep = p.endpoint.trim_end_matches('/');
    let mut headers: Vec<(&str, String)> = vec![];
    let (url, body) = match p.kind.as_str() {
        "ollama" => (
            format!("{ep}/api/chat"),
            json!({ "model": p.model, "stream": false, "format": schema, "options": { "temperature": 0.2 },
                "messages": [{ "role": "system", "content": system }, { "role": "user", "content": user, "images": [image_b64] }] }),
        ),
        "openai" => {
            if let Some(k) = &key {
                headers.push(("Authorization", format!("Bearer {k}")));
            }
            (
                format!("{ep}/chat/completions"),
                json!({ "model": p.model, "temperature": 0.2, "messages": [
                    { "role": "system", "content": system },
                    { "role": "user", "content": [
                        { "type": "text", "text": user },
                        { "type": "image_url", "image_url": { "url": format!("data:image/jpeg;base64,{image_b64}") } } ] } ] }),
            )
        }
        "anthropic" => {
            headers.push(("x-api-key", key.ok_or("no Anthropic API key (set one in Agent & MCP or ANTHROPIC_API_KEY)")?));
            headers.push(("anthropic-version", "2023-06-01".into()));
            (
                format!("{ep}/v1/messages"),
                json!({ "model": p.model, "max_tokens": 4096, "system": system, "messages": [{ "role": "user", "content": [
                    { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": image_b64 } },
                    { "type": "text", "text": user } ] }] }),
            )
        }
        "gemini" => {
            headers.push(("x-goog-api-key", key.ok_or("no Gemini API key (set one in Agent & MCP or GEMINI_API_KEY)")?));
            (
                format!("{ep}/v1beta/models/{}:generateContent", p.model),
                json!({ "systemInstruction": { "parts": [{ "text": system }] },
                    "contents": [{ "role": "user", "parts": [{ "inline_data": { "mime_type": "image/jpeg", "data": image_b64 } }, { "text": user }] }],
                    "generationConfig": { "responseMimeType": "application/json", "temperature": 0.2 } }),
            )
        }
        k => return Err(format!("unknown provider kind '{k}'")),
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(p.timeout_s.max(10))))
        .build()
        .into();
    let mut req = agent.post(&url).header("content-type", "application/json");
    for (k, v) in &headers {
        req = req.header(*k, v);
    }
    let mut resp = req.send_json(&body).map_err(|e| format!("{}: {e}", p.name))?;
    let status = resp.status().as_u16();
    let raw = resp.body_mut().read_to_string().map_err(es)?;
    if status >= 400 {
        return Err(format!("{} returned HTTP {status}: {}", p.name, truncate(&raw, 400)));
    }
    let v: Value = serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", p.name))?;
    let text = match p.kind.as_str() {
        "ollama" => v["message"]["content"].as_str().map(String::from),
        "openai" => v["choices"][0]["message"]["content"].as_str().map(String::from),
        "anthropic" => v["content"].as_array().map(|a| a.iter().filter_map(|b| b["text"].as_str()).collect()),
        _ => v["candidates"][0]["content"]["parts"].as_array().map(|a| a.iter().filter_map(|b| b["text"].as_str()).collect()),
    };
    text.ok_or_else(|| format!("{}: unexpected response {}", p.name, truncate(&raw, 300)))
}

fn run_http(core: &Arc<Core>, p: &Provider, id: i64, prompt: &str, dir: &PathBuf) -> Result<String, String> {
    let system = format!(
        "You are an expert photo retoucher controlling Optixel Studio, a non-destructive RAW editor. You see the current \
         rendered photo, its histogram telemetry and its full edit state. Decide parameter changes that achieve the user's goal. \
         global_parameter_patch contains ABSOLUTE values for only the parameters you change (omit everything else). \
         The image is the full uncropped frame; coordinates are [x, y] fractions of it, (0,0) top-left. To select an object or \
         area (\"select the castle\"), add a separate mask_operations entry for each object named after it with one polygon operation tracing its outline \
         (local_adjustments 0 unless an adjustment was asked for). To remove objects, list them in remove_objects with a generous \
         outline each. Keep highlights unclipped unless asked. Set requires_another_visual_pass to true only if you need to see the result \
         to refine it. Reply with only a JSON object matching this schema:\n{}",
        response_schema()
    );
    let mut summaries = vec![];
    for pass in 1..=p.max_passes.max(1) {
        let snap = snapshot(core, id, dir)?;
        let user = format!(
            "{}User goal: {prompt}\nCurrent edit state: {}\nTelemetry: {}",
            if pass > 1 { "Verification pass: the preview shows the result of your previous changes. Refine, or return an empty patch.\n" } else { "" },
            serde_json::to_string(&snap.state).map_err(es)?,
            compact(&snap.telemetry)
        );
        core.log(&p.id, format!("pass {pass}: asking {} ({})", p.name, p.model));
        let r = parse_object(&call_model(p, &system, &user, &snap.jpeg_b64, &response_schema())?)?;
        core.log(&p.id, format!("pass {pass}: applying model response..."));
        let mut d = snap.state.clone();
        let mut notes = vec![];
        if let Some(patch) = r.get("global_parameter_patch").filter(|v| v.as_object().is_some_and(|o| !o.is_empty())) {
            let (nd, unknown) = develop::apply_patch(&d, patch, false)?;
            d = nd;
            if !unknown.is_empty() {
                notes.push(format!("ignored {}", unknown.join(", ")));
            }
        }
        for op in r["mask_operations"].as_array().into_iter().flatten() {
            let name = op["mask_name"].as_str().unwrap_or("AI mask");
            match develop::mask_from_ops(name, &op["operations"], op.get("local_adjustments").unwrap_or(&json!({}))) {
                Ok((mut m, _)) => match d.masks.iter_mut().find(|x| x.name == name) {
                    Some(x) => {
                        m.mask_id = x.mask_id.clone();
                        m.tuning = x.tuning.take();
                        *x = m;
                    }
                    None => {
                        m.mask_id = format!("mask_{:02}", d.masks.len() + 1);
                        d.masks.push(m);
                    }
                },
                Err(e) => notes.push(e),
            }
        }
        if let Some(objs) = r.get("remove_objects").filter(|v| v.as_array().is_some_and(|a| !a.is_empty())) {
            let count = objs.as_array().map_or(0, |a| a.len());
            core.log(&p.id, format!("pass {pass}: removing {count} objects..."));
            if let Err(e) = mcp::remove_objects(core, id, &mut d, objs) {
                notes.push(e);
            }
        }
        let summary = r["reasoning_summary"].as_str().unwrap_or("").to_string();
        if d != snap.state {
            d.clamp();
            d.ai_provenance_log.push(mcp::provenance(core, &p.id, &truncate(&format!("{prompt} — {summary}"), 300)));
            core.commit(id, &d, &format!("AI ({}): {}", p.name, truncate(prompt, 40)))?;
        }
        core.log(&p.id, format!("pass {pass}: {}{}", summary, if notes.is_empty() { String::new() } else { format!(" [{}]", notes.join("; ")) }));
        summaries.push(summary);
        if !r["requires_another_visual_pass"].as_bool().unwrap_or(false) {
            break;
        }
    }
    Ok(summaries.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_stripping() {
        assert_eq!(strip_ansi("\x1b[32mhello world\x1b[0m"), "hello world");
        assert_eq!(strip_ansi("\x1b[1;31mError:\x1b[0m file not found"), "Error: file not found");
        assert_eq!(strip_ansi("clean text"), "clean text");
        assert_eq!(strip_ansi("\x1b]0;Title\x07rest"), "rest");
    }
}
