# Chapter 4: Backend Rust Core Engine & Module Deep-Dive

This chapter provides a comprehensive technical reference for the Rust backend located in [`src-tauri/src/`](../src-tauri/src/). It details each module's responsibility, data structures, thread safety, memory patterns, and database operations.

---

## 1. Rust Crate Architecture & Cargo Profiles

The backend is built as a hybrid binary/library crate configured in [`src-tauri/Cargo.toml`](../src-tauri/Cargo.toml):

```toml
[lib]
name = "optixel_studio_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[profile.release]
codegen-units = 1
lto = true
opt-level = 3
strip = true

# Pixel loops and image codecs are unusably slow unoptimised.
[profile.dev]
opt-level = 2

[profile.dev.package."*"]
opt-level = 3
```

> [!IMPORTANT]
> Unoptimized pixel processing in debug mode (`opt-level = 0`) runs at unacceptable latency for RAW demosaicing and 3D color lookup. Optixel Studio sets `opt-level = 2` even in development builds to guarantee near-instantaneous slider scrubbing and fast test execution.

---

## 2. Core Modules Breakdown

```
src-tauri/src/
├── lib.rs        # Tauri entrypoint, state management, IPC commands
├── main.rs       # CLI argument parsing and process bootstrap
├── develop.rs    # Non-destructive edit graph, rendering DAG, export encoders
├── catalog.rs    # SQLite database, EXIF parsing, thumbnail cache, XMP sync
├── vision.rs     # Computer vision: Laplacian edge-width blur, clipping, pHash
├── enhance.rs    # Wavelet denoise, Super-Resolution, external AI engine bridge
├── auto.rs       # 6 cancellable autonomous batch functions & telemetry
├── storage.rs    # Storage analyzer, duplicate finder, network share mounting
├── cloud.rs      # S3 (SigV4), WebDAV, Dropbox & OneDrive integration
├── remote.rs     # Immich REST client & Google Photos Picker API
├── ai.rs         # Provider router: CLI agents & HTTP vision APIs
└── mcp.rs        # Model Context Protocol server (HTTP & stdio)
```

---

### 2.1 [`lib.rs`](../src-tauri/src/lib.rs) — State Registry & Command Dispatcher
* **Central State**: Instantiates `Arc<Core>`, wrapping SQLite connection mutexes, settings, proxy caches, and cancellation tokens.
* **Tauri Commands**: Exposes high-level commands invoked from `api.ts`:
  * `catalog_list_photos`, `catalog_update_culling`, `catalog_create_virtual_copy`.
  * `load_develop`, `save_develop`, `render_preview`, `export_photos`.
  * `auto_start_job`, `auto_cancel_job`.
  * `storage_get_locations`, `storage_mount_drive`, `storage_duplicate`.
  * `ai_run_agent_prompt`, `mcp_get_token`.
* **Keychain Management**: Stores sensitive API keys and OAuth tokens in `<OPTIXEL_DATA_DIR>/credentials.json` with POSIX permissions locked to `0600` (read/write by owner only).

---

### 2.2 [`develop.rs`](../src-tauri/src/develop.rs) — The Non-Destructive Render Engine
* **`Develop` Struct**: The single source of truth for all photo modifications:
  * White balance, basic tone, presence (texture, clarity, dehaze), tone curve splines, 8-band color mixer, 3D Point Color swatches, 3-way color grading wheels, camera calibration, detail (noise reduction & sharpening), effects (vignette), grain, lens corrections, transform (Upright), Photoshop Pro adjustments (levels, hue/sat, selective color, channel mixer, gradient map, filters), image rotation, crop, retouch heals, and boolean masks.
* **`Img` Representation**:
  ```rust
  pub struct Img {
      pub w: usize,
      pub h: usize,
      pub px: Vec<[f32; 3]>, // Linear Rec.2020 32-bit floating-point RGB
  }
  ```
* **Render Function**: `pub fn render(src: &Img, d: &Develop, o: &Opts) -> Img` executes the deterministic 9-stage DAG pipeline.

---

### 2.3 [`catalog.rs`](../src-tauri/src/catalog.rs) — Digital Asset Manager & SQLite
* **SQLite Connection**: Operates in WAL mode with foreign keys enabled.
* **Table Schema**:
  * `photos`: Stores path, master/copy relationships, EXIF dimensions, capture date, camera/lens strings, ratings, flags, color labels, keywords, JSON develop state, and thumbnail version counter.
  * `history`: Audit log tracking every undoable operation with timestamp and snapshot.
* **Virtual Copies**: Zero-copy database records with `master_id` pointing to the original photo record, maintaining independent edit graphs.
* **Sidecar Synchronization**: Automatically exports edits to `.olxmp` sidecar files adjacent to original media files for interoperability.

---

### 2.4 [`vision.rs`](../src-tauri/src/vision.rs) — Computer Vision & Telemetry
* **Laplacian Edge-Width Sharpness**: Computes edge width and gradient magnitude across high-contrast contours to score subject focus independently of exposure brightness.
* **Subject Estimation**: Color contrast analysis with border pixels to estimate the main subject bounding box.
* **Clipping Metrics**: Calculates highlight clipping (>99% luminance) and shadow crushing (<1% luminance).
* **Perceptual Hashing (pHash)**: Computes 64-bit DCT-based perceptual hashes for grouping burst shots and finding near-duplicate images.

---

### 2.5 [`enhance.rs`](../src-tauri/src/enhance.rs) — Denoise & Super-Resolution
* **Wavelet Denoise**: Multi-scale *à trous* wavelet shrinkage operating in the perceptual **Oklab** color space, parameterized by measured noise floors at varying luminance levels.
* **Super Resolution**: Classical 2x–4x detail synthesis using Lanczos interpolation, iterative back-projection (IBP), and edge-preserving sharpening.
* **External AI Engine Bridge**: Spawns external CLI engines (such as `realesrgan-ncnn-vulkan`) using safe argv substitution (`{{INPUT}}`, `{{OUTPUT}}`, `{{SCALE}}`) without invoking a shell.

---

### 2.6 [`auto.rs`](../src-tauri/src/auto.rs) — Autonomous Batch Processing
* **Six Core Functions**:
  1. `cull`: Auto cataloging, sharpness scoring, burst selection, star rating, keyword tagging.
  2. `edit`: Measured white balance & auto-tone applied to natural, vivid, portrait, landscape, or monochrome profiles.
  3. `mask`: Automatic generation of Subject, Sky, Background, and People masks.
  4. `enhance`: Batch denoise and super-resolution generating linear TIFF/DNG masters.
  5. `blur`: Screening for missed focus or camera motion blur.
  6. `duplicates`: Identification of exact SHA-256 and perceptual duplicates.
* **Cancellation Support**: Every job checks an `Arc<AtomicBool>` token on every image, allowing instant interruption while safely retaining already-processed records.

---

### 2.7 [`storage.rs`](../src-tauri/src/storage.rs) & [`cloud.rs`](../src-tauri/src/cloud.rs)
* **Network Shares**: Discovers and mounts SMB, NFS, and WebDAV shares using native OS facilities (`mount_smbfs` on macOS, GVfs on Linux, UNC paths on Windows).
* **Cloud Storage**:
  * S3-compatible providers (AWS S3, Backblaze B2, Cloudflare R2, MinIO) with custom **AWS Signature V4** request signing.
  * WebDAV (Nextcloud, ownCloud, Synology).
  * Dropbox (OAuth 2.0 PKCE with refresh token rotation).
  * OneDrive (Microsoft Entra PKCE with offline access).
* **Duplicate Engine**: High-velocity bulk file copying with hash verification, metadata preservation, and sidecar sync.

---

### 2.8 [`remote.rs`](../src-tauri/src/remote.rs) — Immich & Google Photos
* **Immich Integration**:
  * Paginated streaming (250 items/page) allowing instant browsing of libraries with 100,000+ photos.
  * On-demand download of full-size originals for develop and export.
  * Bi-directional rating and flag synchronization.
  * Stacked edit upload (**Save to Immich** stacks the finished edit on top of the original asset).
* **Google Photos Integration**:
  * Integrates Google Photos Picker API for selecting user photos.
  * Library API integration for uploading edited renditions.

---

### 2.9 [`ai.rs`](../src-tauri/src/ai.rs) & [`mcp.rs`](../src-tauri/src/mcp.rs)
* **Agent Router**: Routes natural language instructions to either external CLI agents (Antigravity CLI, Claude CLI) via the built-in MCP server or HTTP multimodal APIs (Ollama, Gemini, Anthropic, OpenAI).
* **MCP Implementation**: Detailed in Chapter 6.

