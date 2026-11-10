<div align="center">

# Optixel Studio

**Professional, non-destructive RAW photo processor and digital asset manager (DAM) with a built-in Model Context Protocol (MCP) server that AI agents can drive.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v0.3.0-success.svg)](https://github.com/rcwkan/optixel-studio/releases)
[![Website](https://img.shields.io/badge/Website-optixel.makecade.com-25c6d8.svg)](https://optixel.makecade.com/)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/rcwkan/optixel-studio/releases)

[**Download App**](#-download--installation) · [**User Guide**](#-user-guide--key-features) · [**AI & Agents**](#-ai--agent-integration) · [**Developer Guide**](#-developer--contributor-guide) · [**Website**](https://optixel.makecade.com/)

</div>

---

## 📸 What is Optixel Studio?

Optixel Studio is a modern, local-first RAW photo editor and library organizer designed for photographers who want complete control over their images:

* 🔒 **Zero Lock-In & Local-First**: Your RAW files stay right where they are. Edits are stored in non-destructive `.olxmp` sidecars, standard Adobe `.xmp` sidecars, and a local SQLite catalog. No mandatory subscriptions, no vendor lock-in.
* 📷 **Native C/C++ Studio RAW Engine (LibRaw & LittleCMS2)**: Professional decoding for Canon CR3, Nikon HE .NEF, Sony .ARW, and Fujifilm X-Trans sensors with multi-algorithm demosaicing (AMaZE, RCD, LMMSE), sensor optical black-level calibration, multi-channel highlight reconstruction, and 32-bit floating-point ProPhoto RGB / ACEScg color management.
* 🌐 **Connect All Your Photos in One Place**: Browse local folders, NAS/network shares (SMB, NFS, WebDAV), self-hosted **Immich** servers, Google Photos picks, and cloud storage (Amazon S3, Backblaze B2, Cloudflare R2, Nextcloud, Dropbox, OneDrive).
* 🎨 **Ahead-of-Time (AOT) GPU Compute Shaders**: High-performance pre-compiled SPIR-V binary kernels for Local Clarity, Dark-Channel Dehaze, 3D LUT trilinear interpolation, axial defringing, and real-time RGB Parade & Waveform scopes.
* 🔍 **Lossless SIMD JPEG, Active AF Point Overlay & Soft-Proofing**: Lossless JPEG transforms with libjpeg-turbo without DCT recompression; proprietary MakerNotes active autofocus point overlay (<kbd>A</kbd>); interactive printer soft-proofing with custom `.icc` paper profiles and neon out-of-gamut warnings.
* 🧠 **In-Place AI Retouching & Neural Denoise**: Paint away objects with generative fill (Stable Diffusion 1.5, LaMa) and clean noise with NAFNet/SCUNet. Non-destructive adjustment layers that never create bloated duplicate master files.
* 🤖 **AI You Can Talk To**: A built-in Model Context Protocol (MCP) server allows external AI agents (Claude, Antigravity, local Ollama, LM Studio) to inspect histograms, trace precision masks, and execute photo editing workflows naturally.

---

## ⬇️ Download & Installation

Pre-built beta release packages are available for direct download from the [GitHub Releases Page](https://github.com/rcwkan/optixel-studio/releases):

| Operating System | Architecture | Package Format | Download Link |
| :--- | :---: | :---: | :--- |
| **macOS** (11.0+) | Apple Silicon (M1–M4) | `.dmg` Installer | [**Download for macOS (Apple Silicon)**](https://github.com/rcwkan/optixel-studio/releases/latest) — `Optixel.Studio_<version>_aarch64.dmg` |
| **Linux** (Debian/Ubuntu) | x86_64 (amd64) | `.deb` Package | [**Download for Linux (.deb)**](https://github.com/rcwkan/optixel-studio/releases/latest) — `Optixel.Studio_<version>_amd64.deb` |
| **Linux** (Universal) | x86_64 | `.AppImage` | [**Download for Linux (.AppImage)**](https://github.com/rcwkan/optixel-studio/releases/latest) — `Optixel.Studio_<version>_amd64.AppImage` |
| **Linux** (Fedora/RHEL) | x86_64 | `.rpm` Package | [**Download for Linux (.rpm)**](https://github.com/rcwkan/optixel-studio/releases/latest) — `Optixel.Studio-<version>-1.x86_64.rpm` |
| **Windows** (10/11) | x64 | `.exe` / `.msi` | [**Download for Windows**](https://github.com/rcwkan/optixel-studio/releases/latest) — `Optixel.Studio_<version>_x64-setup.exe` or `_x64_en-US.msi` |

### 🔐 Package Integrity & Checksums

Every GitHub release includes a canonical [`SHA256SUMS.txt`](https://github.com/rcwkan/optixel-studio/releases/latest) file to verify package authenticity:

```bash
# macOS
shasum -a 256 Optixel.Studio_0.2.1_aarch64.dmg

# Linux
sha256sum -c SHA256SUMS.txt

# Windows (PowerShell)
Get-FileHash .\Optixel.Studio_0.2.1_x64-setup.exe -Algorithm SHA256
```

| Latest Package (v0.2.1) | SHA-256 Checksum |
| :--- | :--- |
| `Optixel.Studio_0.2.1_aarch64.dmg` | `a8772dedefedf8c688d3cc81b44335a23f1d4fe7d19da4a8c511851852e45b91` |
| `Optixel.Studio_0.2.1_x64-setup.exe` | `61a86a3af545f81088365f17e54ef46603687e7b0ea35b208c37ca7d542fab60` |
| `Optixel.Studio_0.2.1_x64_en-US.msi` | `d3ded66a77324a9f6e33599f438d3b984f5677ede44239f079c40d3f0652fdff` |
| `Optixel.Studio_aarch64.app.tar.gz` | `875600356ec44c3e3d94ce692b8b78cde2dd2b73e7fb3a9d0120e970f58e5d43` |

### 🍎 Installing on macOS

Intel Macs are not supported: the AI engine (ONNX Runtime + CoreML) ships for Apple Silicon only.

1. Download `Optixel.Studio_<version>_aarch64.dmg` from the [latest release](https://github.com/rcwkan/optixel-studio/releases/latest).
2. Open the `.dmg` and drag **Optixel Studio** into **Applications**.
3. Launch it from Applications. On first launch macOS shows a warning because the app is not notarized by Apple:
   - **"Optixel Studio.app is damaged and can't be opened"** or **"cannot be opened"**: the download carries a quarantine flag. Remove it in Terminal, then open the app again:
     ```bash
     xattr -cr "/Applications/Optixel Studio.app"
     ```
   - **"Apple could not verify Optixel Studio is free of malware"**: click **Done**, open **System Settings → Privacy & Security**, scroll down and click **Open Anyway**, then confirm with your password. (On macOS 14 and earlier, Control-click the app and choose **Open** instead.)

> [!NOTE]
> Do not choose **Move to Trash** on the "damaged" dialog: the app is fine, macOS only blocks unsigned downloads. Run the `xattr` command above instead.

---

## 📖 User Guide & Key Features

### 1. Library & Photo Viewer
- **Zero-Import Browsing**: Add folders directly via **＋ › Folder…**. Files remain in their original directories; new images dropped into folders are picked up automatically.
- **Grid & Full Photo Viewer**:
  - Press <kbd>G</kbd> for the multi-column thumbnail grid with ratings, flags, and color labels.
  - Press <kbd>V</kbd> or <kbd>Enter</kbd> (or double-click) to open the high-resolution **Photo Viewer** with smooth pan, 100% pixel zoom, slideshow mode, and filmstrip navigation.
- **Fast Culling**: Rate photos with <kbd>0</kbd>–<kbd>5</kbd>, flag picks (<kbd>P</kbd>), unflag (<kbd>U</kbd>), or mark rejects (<kbd>X</kbd>).

### 2. Multi-Source DAM: Drives, Clouds & Immich
- **Network Drives (NAS)**: Mount SMB (`smb://nas.local/photos`), AFP, NFS, or WebDAV shares. Optixel displays connection health and edits photos in place across your network.
- **Self-Hosted Immich Integration**: Connect with your Immich server URL and API key. Browse your library and albums in place without duplicating storage. **Save to Immich** uploads edits and stacks them cleanly over the original. Star ratings and flags sync bidirectionally.
- **Cloud Storage**: Connect Nextcloud, ownCloud, Synology, pCloud, Koofr, Amazon S3, Cloudflare R2, Backblaze B2, Dropbox, or OneDrive. Downloads occur on-demand into a local cache; exports can upload directly to cloud folders.
- **Google Photos**: Import picks securely via the Google Photos Picker API without sharing your broader cloud account.
- **Duplicate & Backup** (*Settings › Storage › Duplicate*): Copy images between local folders, network shares, Immich, and cloud storage with SHA-256 verification, preserving metadata and edits.

### 3. Non-Destructive RAW Develop Workspace
The Develop panel groups every tool by the kind of tuning it does:

| Tab | What's in it |
| --- | --- |
| **Adjust** | Basic (white balance, tone, presence, Color/B&W), tone curve, levels, RAW engine settings |
| **Color** | Color mixer, point color, color grading, hue/saturation, selective color, channel mixer, gradient map, calibration, soft proofing |
| **Detail** | Sharpening, noise reduction, vignette & grain, filters, spot removal (<kbd>Q</kbd>) |
| **HDR** | Everything HDR in one place: on/off, ✦ Auto HDR, method, simulated bracket / edge glow, tone & detail, colour, toning curve |
| **Geometry** | Crop, rotate & flip, Upright transform, lens corrections (whole photo) |
| **Masks** | ✦ AI select (subject, sky, click objects), masks, remove a selected object |
| **Auto** | ✦ Auto, 16 photo styles (Bleach Bypass, Portra, Velvia, Nordic Noir, classic Teal & Orange + 5 variants, etc.) with strength slider, AI fine tune, one-at-a-time auto, and auto rotate |
| **AI** | Local models: AI auto adjust, PerspectiveFields auto align (horizon & verticals), neural denoise, object removal & face swap |

With a mask selected, the Adjust, Color, Detail and HDR tools tune only that mask.

- **Adjust Tab**: Exposure, contrast, highlights, shadows, whites, blacks, texture, clarity, dehaze, vibrance, saturation, and white balance.
- **Live 256-Bin Histogram & Native Scopes**: Real-time RGB and luminance histogram with clipping warnings (<kbd>J</kbd>), plus hardware-accelerated GPU RGB Parade and Waveform monitors.
- **Tone Curves & 8-Band Color Mixer**: Parametric and point tone curves (RGB, R, G, B) alongside 8-hue Hue/Saturation/Luminance control.
- **3D Point Color**: Click the eyedropper on any color in your image to isolate and modify that exact 3D color gamut with controllable falloff.
- **Color Grading**: 3-way split toning (shadows, midtones, highlights) with global luminance and balance sliders.

### 4. Native C/C++ Studio Processing & Optics Engine

Optixel Studio integrates production-grade C/C++ native libraries directly into the Rust backend:

#### A. Camera RAW Engine & Optical Calibration (LibRaw C++)
* **Proprietary Compressed RAWs**: Native decoding for Canon `.CR3` (C-RAW), Nikon High-Efficiency `.NEF`, Sony uncompressed/lossless `.ARW`, and Fujifilm X-Trans non-Bayer sensor arrays.
* **Multi-Algorithm Demosaicing Dropdown**:
  * **RCD (Ratio Corrected Demosaicing)**: Exceptional edge sharpness and fine structural detail for general photography.
  * **AMaZE (Aliasing Minimization and Zipper Elimination)**: Suppresses maze and zipper artifacts in low-ISO landscapes and architectural fine lines.
  * **LMMSE**: Directional, noise-adaptive demosaicing optimized for high-ISO shots, astro, and night photography.
  * **AHD, Linear, VNG, DCB**: Classic broadcast and studio interpolation fallbacks.
* **Sensor Black-Level & Multi-Channel Highlight Recovery**: Reads per-channel optical black-point calibration data directly from masked sensor border pixels, running multi-channel highlight reconstruction before tone mapping.

#### B. Studio Color Management & Soft-Proofing (LittleCMS2 & jxlcms)
* **32-Bit Floating-Point Wide-Gamut Spaces**: Transforms linear camera data into **ProPhoto RGB** or **ACEScg** working color spaces so extreme adjustments never clip hues, with on-the-fly conversion to your display profile (e.g. Apple XDR Display P3, calibrated AdobeRGB).
* **Interactive Printer Soft-Proofing & Out-of-Gamut Overlays**: Load custom `.icc` printer/paper profiles (e.g. Hahnemühle, Canson, Epson FineArt) to simulate ink black-point compensation on screen. Unprintable out-of-gamut pixels are highlighted in neon overlay colors.
* **Full JPEG XL Fidelity**: Pairs `jxl-oxide` with LittleCMS2 (`jxlcms`) to accurately decode and encode XYB perceptual color spaces and HDR PQ/HLG transfer functions.

#### C. Lossless Container I/O, Active AF Points & Sidecars
* **Encrypted Camera MakerNotes & Active AF Point Overlay (<kbd>A</kbd>)**: Decodes vendor-proprietary MakerNotes to draw exact active autofocus points and eye-AF bounding boxes directly over the photo during culling, alongside lens serial numbers and mechanical shutter counts.
* **Bidirectional Lightroom `.XMP` Sidecar Compatibility**: Reads and writes standard Adobe XMP sidecar files, preserving star ratings, color labels, copyright IPTC metadata, and crop rectangles when switching between Lightroom and Optixel Studio.
* **SIMD Lossless JPEG & Modern Codecs (libjpeg-turbo)**: Hardware-accelerated lossless JPEG cropping, 90°/180°/270° rotation, and flipping directly in DCT frequency space without recompression artifacts or generation loss; native Apple ImageIO / libheif for iPhone HEIC and 10-bit AVIF decoding.

#### D. Ahead-of-Time (AOT) GPU Compute Shaders (SPIR-V)
* **Pre-Compiled Binary Compute Kernels**: Compute shaders for Local Clarity/Texture/Dehaze (dark channel prior + bilateral-style edge-preserving contrast), 3D LUT trilinear interpolation, Chromatic Aberration lateral and axial defringing, and Real-Time RGB Parade / Waveform Scopes.
* **Build-Time SPIR-V Optimization**: Shaders in `src-tauri/shaders/*.wgsl` are validated, stripped of debug symbols, and compiled into binary `.spv` bytecode embedded directly into the executable at build time.

### 5. AI Retouching & Neural Denoise (The AI Tab)
Optixel Studio delivers deep-learning generative retouching and denoising directly inside the Develop workspace:
- **Generative AI Object Removal**:
  - Paint over unwanted distractions or people with the smart brush. The mask automatically snaps to edges.
  - Fine-tune brush size and outline dilation (`Grow edge (px)`).
  - Optionally specify what should fill the gap (e.g. *"green grass, mountain rocks, clear sky"*).
  - **Neural Backends**:
    - **Stable Diffusion 1.5 Inpainting**: Generative diffusion fill that synthesizes plausible background content from text descriptions or context.
    - **Moebius (RemovalModel)**: Lightweight 0.2B diffusion inpainting model with learned embedding conditioning for seamless object removal and realistic texture synthesis without needing text descriptions.
    - **LaMa**: High-speed, resolution-robust texture continuation.
  - **Live Progress & Cancellation**: Real-time progress bar displaying execution phase, completed steps, elapsed time, estimated completion time (ETA), and one-click cancel.
- **Neural AI Denoise**:
  - Tame high-ISO sensor noise and grain without softening fine hair, textures, or crisp subject contours.
  - Powered by local **NAFNet** (fast, gentle smartphone/camera denoise) and **SCUNet** (clean & sharp blind denoising) with an adjustable **Strength** slider (0–100).
- **In-Place Adjustment Layers & Zero-Duplicate Architecture**:
  - Traditional editors (like Adobe Lightroom) generate massive duplicate `.dng` or `.tiff` files on disk every time you denoise or remove an object, doubling or tripling catalog storage.
  - Optixel Studio treats AI operations as **in-place non-destructive adjustment layers**: parameters reside in `.olxmp` sidecar metadata, and rendered patches are stored in a dedicated local application cache. Your master RAW photos are never duplicated or overwritten.
  - **Granular History Management**: Remove individual adjustment steps at any time via the red **✕** button on each history row, or undo/redo at full fidelity.
  - **Storage & Relocation Management** (*Settings › Storage*): Interactive library storage breakdown, cache metrics, one-click cleanup, and full storage folder relocation (move catalog database, thumbnails, AI models, and caches to any internal or external drive with confirmation and seamless zero-restart engine switching).
- **Creative Profile Strength**:
  - All 18 creative and film profiles (Natural, Neutral, Vivid, Portrait, Landscape, B&W, Film, Moody, Street, Food, Architecture, Night, Nature, Teal & Orange, Editorial, Dark Moody, 35mm Film, Fine-Art B&W) include a real-time **Strength** slider (0% to 200%, default 100%).
  - Smoothly scales tone curves, HSL color mixer channels, 3-way color grading, and presence adjustments. Double-click the slider to reset to 100%.

### 5. Optics & Perspective (Geometry and Detail tabs)
- **Lens Corrections**: Distortion, lateral chromatic aberration (auto-measured along contrast edges), defringe, and vignetting.
- **Upright Perspective Solver**: Automatically solve camera pitch, roll, and yaw using Hough transform line detection (**Auto**, **Level**, **Vertical**, **Full**, or **Guided** with 2–4 lines).
- **Constrain Crop**: Automatically shrink crop boundaries to prevent transparent edges after perspective corrections.
- **Film Grain & Camera Calibration**: Fine-grained analog film grain and sensor calibration adjustments.

### 6. Photoshop Tools
Photoshop-style adjustments applied non-destructively on top of RAW develop, grouped with their kind (Adjust, Color, Detail):
- **Levels**: Per-channel RGB black, gamma, and white points with auto 0.1% clipping.
- **Selective Color**: Adjust CMYK ink balances across 9 color ranges (reds, yellows, greens, cyans, blues, magentas, whites, neutrals, blacks).
- **Channel Mixer & Gradient Maps**: Custom monochrome channel blends and dual-tone gradient maps.
- **Filters**: Gaussian blur, image invert, posterize, and threshold.

### 7. Automated Functions & Background Batch Jobs
Open the **Auto** bar on the right side of the Library to run background batch operations:
- **Auto Catalog & Culling**: Measures edge-width subject sharpness, highlights/shadows clipping, and noise; automatically detects bursts, selects the sharpest frame, and tags rejects.
- **Auto Horizon Levelling**: Detects tilted horizons and levels images as dedicated history steps.
- **Auto Look Profiles with Strength Slider**: One-click professional color profiling (Natural, Vivid, Portrait, Landscape, Street, Food, Architecture, Night, Nature, Film, Moody, B&W), or picked per photo by type with the AI, with 0–200% strength scaling.
- **Find Blurred Subjects & Duplicates**: Flag missed-focus or motion-blurred shots; identify duplicate and visually similar photos by perceptual hash.
- **Wavelet Denoise & Super Resolution**: Enhance images with multi-scale wavelet denoising and 1.5–4× enlargement into clean 32-bit float master TIFFs.
- **HDR Merge & HDR Toning**: Merge exposure brackets (aligned, deghosted) into a 32-bit float master; tone-map any photo with local adaptation and detail control in Pro.

---

## 🤖 AI & Agent Integration

Optixel Studio introduces the **Model Context Protocol (MCP)** directly into photo editing. External AI agents can inspect photos, read histogram telemetry, and adjust develop sliders autonomously.

### Built-in MCP Server
The app serves streamable HTTP MCP on loopback:
```text
Endpoint: http://127.0.0.1:41920/mcp
Auth:     Bearer <token> (found in Settings › Agent & MCP)
```

Add Optixel Studio to Claude Desktop or Claude Code:
```bash
claude mcp add --transport http optixel http://127.0.0.1:41920/mcp --header "Authorization: Bearer <your-token>"
```

### Pluggable AI Router & Local Neural Models
In the Develop view, open the **✦ AI** bar to execute natural language prompts:
- **Interactive Object Masking (✦ Select)**: Click an object (e.g. "castle", "sky", "person") — your selected AI provider traces the boundary and creates an isolated mask.
- **Smart Retouching (✦ Remove)**: Ask the AI to "remove the tourists in the background" — outlines are healed using context-aware texture matching.
- **Dedicated Neural Tooling (*The AI Tab*)**: Direct in-app controls for Generative Object Removal (Stable Diffusion 1.5, LaMa) and AI Denoise (NAFNet, SCUNet) with live progress tracking and cancel.
- **Local Neural Model Manager (*Settings › AI models*)**: Download and configure ONNX models directly from Hugging Face with automatic hardware recommendations (DirectML, Apple Silicon, CPU) and SHA-256 integrity verification.
- **Autonomous Agent MCP Tools**: Autonomous AI agents can query histograms, apply develops (`develop_auto`), perform non-destructive generative fill (`ai_reconstruct_and_retouch`), or execute inpainting object removal (`retouch_remove_objects`) via the standard MCP server.
- **Supported LLM Providers**:
  - **CLI Agents**: Claude CLI (`claude`), Antigravity CLI (`antigravity` / `agy`), custom CLI templates.
  - **Local Models**: Ollama (`qwen2.5-vl`, `llama3.2-vision`), LM Studio, vLLM, llama.cpp.
  - **Cloud APIs**: Anthropic Claude, Google Gemini, OpenAI, OpenRouter.

---

## 💻 Developer & Contributor Guide

### Tech Stack
- **Desktop Shell**: [Tauri v2](https://tauri.app) with custom asset protocol and restricted IPC capabilities.
- **Backend Core**: Rust (stable) with multi-threaded [rayon](https://crates.io/crates/rayon) parallelism and [rusqlite](https://crates.io/crates/rusqlite).
- **Native C/C++ RAW Decoder**: [LibRaw](https://www.libraw.org/) C++ integration for Canon CR3, Nikon HE .NEF, Sony .ARW, and Fuji X-Trans demosaicing (AMaZE, RCD, LMMSE), with [rawler](https://crates.io/crates/rawler) pure-Rust fallback.
- **Studio Color Management**: [LittleCMS 2](https://www.littlecms.com/) (static `lcms2`) & `jxl-oxide` (`jxlcms`) with 32-bit floating-point ProPhoto RGB, ACEScg, and custom `.icc` printer soft-proofing.
- **SIMD Image Codecs**: [libjpeg-turbo](https://libjpeg-turbo.org/) for lossless DCT transforms, and Apple ImageIO / [libheif](https://github.com/strukturag/libheif) for HEIC & 10-bit AVIF.
- **AOT GPU Pipeline**: [wgpu](https://wgpu.rs/) with Ahead-of-Time SPIR-V bytecode compiled and optimized at build time via [naga](https://crates.io/crates/naga).
- **Frontend**: [React 19](https://react.dev), [TypeScript](https://www.typescriptlang.org), [Tailwind CSS v4](https://tailwindcss.com), and [Vite](https://vite.dev).

### Prerequisites
- **Node.js**: v20 or newer (`node -v`)
- **Rust**: Stable toolchain (`rustup update stable`)

### Running in Development

```bash
# Clone the repository
git clone https://github.com/rcwkan/optixel-studio.git
cd optixel-studio

# Start development mode (Vite frontend + native Tauri window)
./scripts/dev.sh
```

### Building for Production

```bash
# Build macOS application bundle (.app) and DMG installer
./scripts/build.sh

# Run comprehensive test suite (Rust unit tests + TypeScript typecheck)
./scripts/test.sh

# Run headless MCP stdio server
./scripts/mcp.sh
```

### Headless CLI Usage

Optixel Studio can run headlessly for batch imports or as a stdio MCP server for agent scripts:

```bash
# Launch GUI
optixel-studio

# Headless recursive import (scans directory, generates previews in place)
optixel-studio --import /path/to/photos

# Headless MCP stdio bridge
optixel-studio --mcp-stdio
```

### Project Structure

```
├── src/                     # React 19 Frontend
│   ├── App.tsx              # App shell, navigation, global hotkeys
│   ├── Library.tsx          # Digital asset management & grid
│   ├── PhotoViewer.tsx      # Full-screen photo viewer, zoom, filmstrip
│   ├── Develop.tsx          # RAW develop workspace & histogram
│   ├── AiDenoise.tsx        # AI object removal, neural denoise & progress
│   ├── Advanced.tsx         # Whole-photo panels: lens, Upright, calibration, grain, spots, proofing
│   ├── Pro.tsx              # Levels, HDR tab, hue/sat, selective color, Photoshop tools
│   ├── Auto.tsx             # 15 photo styles, strength slider, auto fine tune & auto rotate
│   ├── AutoView.tsx         # Batch auto culling & processing jobs
│   ├── Storage.tsx          # Cloud storage, network drives, clean up
│   └── Agent.tsx            # AI provider router & MCP token settings
├── src-tauri/               # Rust Core Backend & Native C/C++ Engine
│   ├── csrc/                # Native C/C++ implementations (ol_raw.cpp, ol_demosaic.cpp, ol_codecs.cpp)
│   ├── shaders/             # AOT GPU compute shaders (*.wgsl -> binary *.spv)
│   ├── vendor/libraw/       # Embedded LibRaw C++ decoding library
│   ├── src/develop.rs       # Non-destructive edit graph & pixel pipeline
│   ├── src/libraw.rs        # LibRaw FFI, AMaZE/RCD/LMMSE demosaicing, black-point & highlights
│   ├── src/icc.rs           # LittleCMS 2 color management, ProPhoto/ACEScg, soft-proofing
│   ├── src/gpu.rs           # wgpu AOT SPIR-V compute pipeline (clarity, dehaze, lut3d, scopes)
│   ├── src/xmp.rs           # Bidirectional Adobe Lightroom .XMP sidecar sync
│   ├── src/aimodel.rs       # ONNX Runtime, NAFNet, SCUNet & SD inpainting
│   ├── src/catalog.rs       # SQLite schema, EXIF, .olxmp sidecars
│   ├── src/vision.rs        # Focus peaking, active AF point detection, blur & noise analysis
│   ├── src/auto.rs          # Auto culling, rating & batch functions
│   ├── src/mcp.rs           # Embedded HTTP & stdio MCP server
│   ├── src/cloud.rs         # S3, WebDAV, Dropbox & OneDrive storage
│   ├── src/remote.rs        # Immich integration & Google Photos picker
│   └── src/ai.rs            # AI agent router & command execution
├── html/                    # Marketing landing page & static site
├── cloudflare-worker/       # Cloudflare edge proxy & router
├── guide/                   # In-depth architectural & developer guides
└── scripts/                 # Build, test, run, and deployment scripts
```

For in-depth architectural specifications and component guides, consult [`guide/README.md`](guide/README.md) and the [Functional & System Specification](optixel_studio_functional_and_system_specification.md).

---

## 📄 License & Security

- **Application Core License**: Optixel Studio is released under the open-source [MIT License](LICENSE).
- **Security Policy**: See [`SECURITY.md`](SECURITY.md) for vulnerability disclosure guidelines.

### Third-Party AI Model Licenses & Disclaimers

Optixel Studio allows optional, on-demand downloading of third-party neural network weights for local inference. Each model operates under its own license and terms of use:

| Model | Purpose | Author / Lab | License | Terms & Disclaimer |
| :--- | :--- | :--- | :--- | :--- |
| **PerspectiveFields** | Auto Horizon & Vertical Alignment | Jin et al. (University of Michigan / CVPR 2023) | MIT License | Permissive open-source license ([jinlinyi/PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)). |
| **Stable Diffusion 1.5 Inpainting** | Generative Fill | RunwayML / Stability AI | CreativeML OpenRAIL-M | Subject to OpenRAIL-M use-based restrictions. |
| **Moebius (RemovalModel)** | Inpainting & Object Removal | Duan & Xu et al. (HUST) / SimonW | Apache-2.0 | Permissive open-source license. |
| **LaMa** | Inpainting & Object Removal | Suvorov et al. (Samsung Research) | Apache-2.0 | Permissive open-source license. |
| **NAFNet** | Fast Neural Denoise | Megvii Technology | MIT License | Permissive open-source license. |
| **SCUNet** | Real-World Camera Denoise | Zhang et al. | Apache-2.0 | Permissive open-source license. |
