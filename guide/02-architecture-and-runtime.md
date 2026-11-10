# Chapter 2: System Architecture & Dual-Mode Runtime

This chapter details the high-level architecture of Optixel Studio, its execution modes, the IPC bridge between the frontend and backend, state ownership, and its concurrency model.

---

## 1. High-Level Architectural Flow

Optixel Studio separates user interface presentation, native system integration, and raw image processing into clean, isolated tiers:

```mermaid
flowchart TD
    subgraph UI ["Frontend Tier (Browser/WebView Context)"]
        ReactApp["React 19 + TypeScript Application"]
        CustomHooks["React State & Event Listeners"]
        ApiClient["api.ts (IPC Invocation Wrappers)"]
        ReactApp --> CustomHooks --> ApiClient
    end

    subgraph IPC ["Tauri v2 IPC Bridge"]
        InvokeChannel["tauri::invoke (Commands)"]
        EventChannel["tauri::Emitter (Events: job-progress)"]
        AssetProtocol["tauri-protocol-asset (asset:// protocol)"]
        ApiClient --> InvokeChannel
        EventChannel --> CustomHooks
        AssetProtocol --> ReactApp
    end

    subgraph RustCore ["Backend Tier (Native Rust Core Engine)"]
        LibRS["lib.rs (State Registry & Tauri Commands)"]
        CoreStruct["Core (Arc<Core> Shared State)"]
        DevelopRS["develop.rs (Non-Destructive DAG Engine)"]
        CatalogRS["catalog.rs (SQLite WAL & XMP Sidecars)"]
        AutoRS["auto.rs (Cancellable Job Runner)"]
        VisionRS["vision.rs (CV Metrics & pHash)"]
        McpServer["mcp.rs (Streamable HTTP & stdio)"]
        AiRouter["ai.rs (Agent & Provider Router)"]

        InvokeChannel --> LibRS
        LibRS --> CoreStruct
        CoreStruct --> DevelopRS
        CoreStruct --> CatalogRS
        CoreStruct --> AutoRS
        CoreStruct --> VisionRS
        CoreStruct --> McpServer
        CoreStruct --> AiRouter
        AutoRS -.-> EventChannel
    end

    subgraph StorageTier ["Persistence & Remote Sources"]
        SQLiteDB[("catalog.olcat (SQLite WAL)")]
        ThumbCache[("thumbs/*.jpg (Thumbnail Cache)")]
        LocalFiles[("Local RAW / JPEG Originals")]
        Sidecars[(".olxmp Sidecar Metadata")]
        ImmichCloud[("Immich / Google Photos / S3 / WebDAV")]

        CatalogRS --> SQLiteDB
        CatalogRS --> ThumbCache
        CatalogRS --> Sidecars
        DevelopRS --> LocalFiles
        CoreStruct --> ImmichCloud
    end
```

---

## 2. Multi-Mode Application Runtime

Optixel Studio is packaged as a single native binary that adapts its execution path depending on command-line arguments:

### Mode A: Full Desktop GUI Application (`optixel-studio`)
* Default launch mode when launched from Finder, application menus, or CLI without arguments.
* Initializes Tauri v2 desktop shell, creates the native application window (1280x800 default), registers system menus, and mounts the React 19 UI.
* Starts the in-process **Streamable HTTP MCP Server** on loopback `http://127.0.0.1:41920/mcp` with bearer authentication.
* Serves the `asset://` custom protocol to stream thumbnails and local photos securely into the WebView without CORS restrictions.

### Mode B: Headless CLI Folder Importer (`optixel-studio --import <folder>`)
* Executes without initializing GUI windows or WebViews.
* Recursively scans `<folder>` for supported RAW and bitmap formats (DNG, CR2, CR3, NEF, ARW, RAF, RW2, ORF, PEF, JPEG, TIFF, PNG, AVIF, HEIC).
* Extracts EXIF metadata, creates `.olcat` records, and generates cached thumbnails under `thumbs/`.
* Exits immediately upon completion with status code `0`, ideal for automated ingestion scripts and ingest hooks.

### Mode C: Headless or Forwarded MCP Stdio Server (`optixel-studio --mcp-stdio`)
* Provides a standard Model Context Protocol interface over `stdin` / `stdout`.
* **Smart Forwarding**:
  1. Checks if a GUI instance of Optixel Studio is currently running on `http://127.0.0.1:41920`.
  2. If running, forwards incoming JSON-RPC 2.0 messages directly to the active GUI app over HTTP, updating the user's open Develop canvas in real-time.
  3. If no GUI app is running, initializes a headless `Core` instance against `<OPTIXEL_DATA_DIR>/catalog.olcat` and performs catalog edits and exports in headless mode.

---

## 3. Tauri IPC & Communication Architecture

### 3.1 Command Invocation (`invoke`)
The frontend communicates with Rust through typed wrapper functions declared in [`src/api.ts`](../src/api.ts). Each function wraps `@tauri-apps/api/core::invoke`:

```typescript
// Frontend (src/api.ts)
export async function loadDevelop(photoId: number): Promise<DevelopResponse> {
  return invoke("load_develop", { photoId });
}

export async function saveDevelop(photoId: number, develop: Develop): Promise<void> {
  return invoke("save_develop", { photoId, develop });
}
```

On the Rust backend, commands are defined as async/sync functions registered in the `tauri::Builder` invoke handler in [`src-tauri/src/lib.rs`](../src-tauri/src/lib.rs):

```rust
// Backend (src-tauri/src/lib.rs)
#[tauri::command]
fn load_develop(state: tauri::State<'_, Arc<Core>>, photo_id: i64) -> Result<DevelopResponse, String> {
    state.load_develop(photo_id)
}
```

### 3.2 Real-Time Event Streaming (`emit` / `listen`)
Long-running asynchronous jobs (such as the 6 Auto functions in `auto.rs` or cloud downloads in `remote.rs`) do not block IPC commands. Instead, they stream progress events through `tauri::Emitter`:

* **`job-progress`**: Emitted during auto culling, AI masking, batch editing, and duplicate scans.
  ```typescript
  export type JobProgress = {
    function: string;      // "cull" | "edit" | "mask" | "enhance" | "blur" | "duplicates"
    done: number;
    total: number;
    current_photo?: string;
    summary?: any;
  };
  ```
* **`folder-progress`**: Emitted during large directory ingestion and cloud folder downloads.
* **`catalog-changed`**: Emitted when external operations (e.g. MCP tool mutations) modify ratings or settings, instructing the UI to refresh without a full reload.

### 3.3 The Asset Streaming Protocol
To display multi-gigabyte RAW image files and hundreds of cached thumbnails smoothly without base64 serialization overhead, Optixel Studio leverages Tauri's custom asset protocol:

```typescript
import { convertFileSrc } from "@tauri-apps/api/core";

// Converts local file path: /Users/name/photos/IMG_0001.CR3
// Into custom protocol URL: asset://localhost/Users/name/photos/IMG_0001.CR3
const imgSrc = convertFileSrc(photo.path);
```

---

## 4. State Ownership & Concurrency Model

### 4.1 The Central `Core` Struct
The entire backend engine state is consolidated inside the `Core` struct, managed as an `Arc<Core>` inside Tauri's managed state:

```rust
pub struct Core {
    pub dir: PathBuf,                     // Root OPTIXEL_DATA_DIR
    pub db: Mutex<rusqlite::Connection>,  // SQLite connection with WAL mode
    pub settings: Mutex<Settings>,        // App settings, active providers, cloud accounts
    pub cancel_tokens: Mutex<HashMap<String, Arc<AtomicBool>>>, // Cancellable auto jobs
    pub proxy_cache: Mutex<LruCache<PathBuf, Arc<Img>>>,        // In-memory 1600px develop proxies
}
```

### 4.2 SQLite Concurrency via WAL Mode
The database connection initiates with Write-Ahead Logging:
```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```
* **Read-Write Separation**: Multiple read transactions can execute concurrently with a single write transaction without table locks.
* **Crash Resilience**: Atomic commits guarantee that unexpected process termination leaves `.olcat` in a consistent state.

### 4.3 Rayon Parallelism for Pixel Processing
Tonal math, 3D Point Color lookup, bilateral filtering, and image encoding are computationally heavy. Optixel Studio avoids blocking the main event loop by using **Rayon** for data-parallel iteration:

```rust
// Parallel pixel processing across CPU cores
px.par_iter_mut().for_each(|p| {
    *p = tone.apply(*p);
});
```

Long-running jobs spawned in `auto.rs` run on dedicated background threads, checking their designated `AtomicBool` cancellation token on each iteration to permit instantaneous user cancellation.

