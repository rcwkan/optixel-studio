# Chapter 1: Development Environment & Toolchain Setup

This chapter guides developers through setting up a complete, reproducible development environment for Optixel Studio on macOS, Linux, and Windows.

---

## 1. Prerequisites

### 1.1 Node.js Runtime
* **Version**: Node.js **v20.x or v22.x LTS** (Node 20+ required).
* **Package Manager**: `npm` (bundled with Node) or `pnpm` / `bun` (scripts default to standard `npm`).
* Verify installation:
  ```bash
  node --version   # v20.x.x or higher
  npm --version    # 10.x.x or higher
  ```

### 1.2 Rust Toolchain
* **Edition**: Rust **2024 edition** (Rust 1.85+ stable).
* **Components**: `cargo`, `rustc`, `rustup`.
* Install via official Rustup:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustup update stable
  ```
* Verify installation:
  ```bash
  rustc --version
  cargo --version
  ```

---

## 2. Platform-Specific System Dependencies

### 2.1 macOS (Apple Silicon & Intel)
1. Install **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```
2. Verify native disk image packaging utilities:
   `hdiutil` is built into macOS and is utilized by `scripts/build.sh` to package `.dmg` release installers.
3. Path configuration: Ensure `/opt/homebrew/bin` (Apple Silicon) or `/usr/local/bin` (Intel) is present in your `$PATH`.

### 2.2 Linux (Ubuntu / Debian / Fedora / Arch)
Tauri v2 requires native system webview and graphical libraries.

**Ubuntu / Debian**:
```bash
sudo apt-get update
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev
```

**Fedora**:
```bash
sudo dnf install \
  webkit2gtk4.1-devel \
  openssl-devel \
  curl \
  wget \
  libappindicator-gtk3-devel \
  librsvg2-devel
```

> [!TIP]
> For building Linux distributions without installing local GTK development headers, use the containerized build script: `./scripts/build-linux.sh`. This leverages `scripts/Dockerfile.linux` to build `.deb` packages inside an isolated Debian container.

### 2.3 Windows (x64 / ARM64)
1. Ensure **WebView2 Runtime** is installed (pre-installed on Windows 10/11).
2. Ensure Git Bash or PowerShell 7 is configured.

---

## 3. Repository Structure & Scripts

```
optixel_studio/
├── dist/                      # Compiled frontend static assets (Vite output)
├── dist-packages/             # Output installers (.dmg, .app, .deb)
│   ├── macos/                 # macOS .app bundle and .dmg disk image
│   └── linux/deb/             # Linux .deb package
├── guide/                     # Developer documentation & engineering guides
├── scripts/                   # Production & development automation scripts
│   ├── build.sh               # Native release packaging (macOS .dmg / Linux)
│   ├── build-linux.sh         # Linux containerized build launcher
│   ├── build-linux-container.sh# Script executed inside Docker container
│   ├── Dockerfile.linux       # Docker environment for Linux builds
│   ├── dev.sh                 # Fast dev launcher with automatic PATH checks
│   ├── import.sh              # Headless folder importer CLI
│   ├── mcp.sh                 # Headless MCP stdio launcher
│   └── test.sh                # Full test suite: cargo test + npm build
├── src/                       # React 19 Frontend source code
│   ├── App.tsx                # Main application frame, navigation, state
│   ├── Library.tsx            # DAM photo grid, collections, filtering
│   ├── Develop.tsx            # RAW develop canvas, adjustments, masks
│   ├── Advanced.tsx           # Optics, Upright, Calibration, Soft Proofing
│   ├── Pro.tsx                # Photoshop-grade adjustment layers
│   ├── AutoView.tsx           # Auto functions UI & job telemetry
│   ├── Agent.tsx              # AI prompt bar & provider controller
│   ├── Settings.tsx           # App settings, MCP port/token, AI providers
│   ├── Storage.tsx            # Locations, network shares, cloud storage, Duplicate
│   ├── controls.tsx           # Custom sliders, wheels, curves, inputs
│   └── api.ts                 # Tauri IPC bindings & TypeScript interfaces
├── src-tauri/                 # Rust Backend source code
│   ├── Cargo.toml             # Rust package configuration & dependencies
│   ├── tauri.conf.json        # Tauri v2 bundle and window configuration
│   └── src/
│       ├── lib.rs             # Tauri command registry & application state
│       ├── main.rs            # Desktop executable entrypoint
│       ├── develop.rs         # Non-destructive edit graph & rendering pipeline
│       ├── catalog.rs         # SQLite DAM, EXIF, thumbnails, XMP sidecars
│       ├── vision.rs          # Laplacian sharpness, clipping, pHash analysis
│       ├── enhance.rs         # Wavelet denoise & super resolution
│       ├── auto.rs            # Cancellable batch jobs for the 6 auto functions
│       ├── storage.rs         # Storage inspector, duplicate detector, network mounts
│       ├── cloud.rs           # S3 SigV4, WebDAV, Dropbox & OneDrive engines
│       ├── remote.rs          # Immich API client & Google Photos Picker API
│       ├── ai.rs              # Pluggable AI / CLI provider router
│       └── mcp.rs             # Built-in MCP server (Streamable HTTP & stdio)
├── package.json               # Node.js project manifest & scripts
├── vite.config.ts             # Vite configuration with Tailwind CSS v4 plugin
└── tsconfig.json              # TypeScript compilation configuration
```

---

## 4. Environment Variables & Data Directory

Optixel Studio isolates user data, catalogs, thumbnails, and cache files:

| Environment Variable | Default Value | Description |
|---|---|---|
| `OPTIXEL_DATA_DIR` | `~/Library/Application Support/org.optixel.studio/` (macOS)<br>`~/.local/share/org.optixel.studio/` (Linux)<br>`%APPDATA%\org.optixel.studio` (Windows) | Root directory containing `catalog.olcat`, `thumbs/`, `settings.json`, `credentials.json`, and `remote/`. |
| `ANTHROPIC_API_KEY` | *(None / OS Keychain)* | Fallback API key for Anthropic Claude provider. |
| `GEMINI_API_KEY` | *(None / OS Keychain)* | Fallback API key for Google Gemini API provider. |
| `OPENAI_API_KEY` | *(None / OS Keychain)* | Fallback API key for OpenAI / compatible endpoints. |
| `OPENROUTER_API_KEY` | *(None / OS Keychain)* | Fallback API key for OpenRouter API provider. |

### Data Directory Contents
```
<OPTIXEL_DATA_DIR>/
├── catalog.olcat          # SQLite database in WAL mode (DAM metadata & history)
├── catalog.olcat-wal      # SQLite Write-Ahead Log
├── catalog.olcat-shm      # SQLite shared memory
├── thumbs/                # Cached JPEG thumbnails named {photo_id}_{thumb_v}.jpg
├── settings.json          # User preferences, active provider, cloud accounts
├── credentials.json       # Encrypted/restricted (0600) API keys and cloud tokens
└── remote/                # Local cache for remote downloads
    ├── Cloud/             # On-demand downloads from S3, WebDAV, Dropbox, OneDrive
    └── Immich/            # On-demand downloads from connected Immich instances
        └── originals/     # Full-resolution originals downloaded for editing
```

---

## 5. Step-by-Step Setup & Verification

### Step 1: Clone and Install NPM Dependencies
```bash
git clone <repository-url>
cd optixel_studio
npm install
```

### Step 2: Validate Rust Backend
Run the backend test suite to verify compiler readiness and test assertions:
```bash
cargo test --manifest-path src-tauri/Cargo.toml
```

### Step 3: Launch in Development Mode
Start the Vite development server and Tauri native desktop window:
```bash
./scripts/dev.sh
# Or alternatively:
npm run tauri dev
```
Hot-module replacement (HMR) is active for all files in `src/`. Modifying files in `src-tauri/` automatically triggers Rust recompilation and restarts the backend.

### Step 4: Verify MCP Server Stdio Launcher
In a separate terminal, test the standalone Model Context Protocol server interface:
```bash
./scripts/mcp.sh
```
Send an MCP `initialize` JSON-RPC handshake over stdin:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}}
```
The server will respond with capabilities, supported tools, and server metadata. Press `Ctrl+C` to exit.

