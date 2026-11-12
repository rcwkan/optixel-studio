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
* 🗺️ **Multi-Modal Catalog Intelligence (Grid, Date, Map & Gear)**: Fast photo discovery across multi-column thumbnail Grid (<kbd>G</kbd>), chronological timeline By Date (<kbd>D</kbd>), interactive OpenStreetMap By Location (<kbd>L</kbd>) with ~25km cluster pins & GPS EXIF scanning, gear breakdown By Camera body & lens (<kbd>K</kbd>), and on-device **ArcFace** biometric face clustering and People search.
* 🎨 **Ahead-of-Time (AOT) GPU Compute Shaders**: High-performance pre-compiled SPIR-V binary kernels for Local Clarity, Dark-Channel Dehaze, 3D LUT trilinear interpolation, axial defringing, and real-time RGB Parade & Waveform scopes.
* 🔍 **Lossless SIMD JPEG, Active AF Point Overlay & Soft-Proofing**: Lossless JPEG transforms with libjpeg-turbo without DCT recompression; proprietary MakerNotes active autofocus point overlay (<kbd>A</kbd>); interactive printer soft-proofing with custom `.icc` paper profiles and neon out-of-gamut warnings.
* 🧠 **On-Device AI Models & SAM 2.1 Click-to-Select**: Point-and-click object segmentation with **SAM 2.1**; neural horizon and vertical alignment with **PerspectiveFields**; paint away distractions with **Moebius 0.2B**, **Stable Diffusion 1.5**, or **LaMa**; and clean sensor noise with **NAFNet** / **SCUNet**. Non-destructive in-place adjustment layers with zero duplicate files.
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

Every GitHub release includes a canonical [`SHA256SUMS.txt`](https://github.com/rcwkan/optixel-studio/releases/latest) file alongside Minisign signatures to verify package authenticity:

```bash
# macOS
shasum -a 256 Optixel.Studio_0.3.0_aarch64.dmg

# Linux
sha256sum -c SHA256SUMS.txt

# Windows (PowerShell)
Get-FileHash .\Optixel.Studio_0.3.0_x64-setup.exe -Algorithm SHA256
```

Official SHA-256 release checksums for all production builds are published in [`SHA256SUMS.txt`](https://github.com/rcwkan/optixel-studio/releases/latest) with each release tag. Verify downloaded installers against the signed release manifest before installing.

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

### 1. Library, Multi-Modal Views & People Management
- **Zero-Import Browsing**: Add folders directly via **＋ › Folder…**. Files remain in their original directories; new images dropped into folders are picked up automatically without importing into proprietary vaults.
- **Five Dedicated Library Views**:
  - **Grid View (<kbd>G</kbd>)**: High-performance multi-column thumbnail grid with ratings (<kbd>0</kbd>–<kbd>5</kbd>), flags (<kbd>P</kbd>, <kbd>X</kbd>, <kbd>U</kbd>), color labels, and adjustable thumbnail scaling (120px–320px).
  - **By Date Timeline (<kbd>D</kbd>)**: Chronological grouping by Year, Month, or Day. Collapsible date sections, quick-jump year pills, dated photo count badges, and single-click group selection.
  - **By Location Map (<kbd>L</kbd>)**: Interactive OpenStreetMap map view with pan, zoom, fit-to-bounds, and geographic clustering (~25km radius) with thumbnail previews and badge counts. Toggle Geotagged vs. Untagged photos and run on-demand GPS EXIF scan (`scan_missing_gps`).
  - **By Camera Equipment (<kbd>K</kbd>)**: Group photos by camera model, lens metadata, and ISO sensitivity tiers (<400 low noise, 400–1600 standard, 1600+ high sensitivity). Summary cards with photo counts, date ranges, lens breakdowns, and sample filmstrips.
  - **Photo Viewer (<kbd>V</kbd> or <kbd>Enter</kbd>)**: High-resolution viewer with smooth pan, 100% pixel zoom, slideshow mode, and filmstrip navigation. Press <kbd>Esc</kbd> or <kbd>G</kbd> to return to grid.
- **Biometric People Manager & Face Search** (*Library › People*):
  - On-device **YOLOFace** face detector and **ArcFace w600k** face recognizer.
  - Automatically clusters detected faces into distinct people profiles, allows naming individuals, merging duplicate profiles, and filtering the entire catalog by person or combinations of people.
  - Interactive face timeline displaying the chronological frequency of each person across shoot years.
- **Fast Keyboard Culling**: Rate photos with <kbd>0</kbd>–<kbd>5</kbd>, flag picks (<kbd>P</kbd>), unflag (<kbd>U</kbd>), or mark rejects (<kbd>X</kbd>). Filtering toolbar by rating, pick status, color labels, aspect ratio, and edit status.

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

### 5. On-Device AI Suite, Neural Models & Segmentation (AI & Masks Tabs)
Optixel Studio delivers production-grade on-device neural inference directly inside the Develop and Library workspaces, running locally with zero cloud subscriptions, zero network latency, and complete privacy:

#### A. Segment Anything (SAM 2.1 & SAM 3) — Click-to-Select Object Masks
- **Interactive Object Selection (<kbd>S</kbd>)**: Click any object in your photo (person, apparel, vehicle, building, sky, animal) to generate a pixel-precise, edge-snapped mask in under a second. Hold <kbd>⌥</kbd>/<kbd>Alt</kbd> and click to subtract regions from the selection.
- **Local Neural Engines**:
  - **SAM 2.1 Hiera-Tiny**: Meta's ultra-fast hierarchical vision transformer encoder. Runs smoothly on every click with minimal memory overhead.
  - **SAM 2.1 Hiera-Small**: Larger image encoder yielding pristine boundary fidelity along complex hair strands, tree branches, and transparent edges.
  - **SAM 3 & EfficientSAM3 (Concept Selection)**: Type what to select in natural language (e.g., *"red umbrella"*, *"mountain ridge"*, *"vintage car"*) and the neural model isolates every matching instance automatically.

#### B. PerspectiveFields — Neural Horizon & Vertical Alignment
- **ParamNet Camera Calibration & Gravity Fields**: Replaces brittle edge heuristics with deep perceptual surface normal and gravity field estimation (~0.1°–0.5° precision; CVPR 2023).
- **Specialized Weights**:
  - **GSV Model**: Trained on Google Street View; excels at architectural facades, urban street scenes, and converging skyscraper lines.
  - **EDINA / 360Cities Model**: Trained on diverse 360° panoramas; optimized for interior living spaces, horizon coastlines, and rolling landscapes.
- **One-Click Execution**: Apply horizontal auto-leveling or vertical perspective straightening as discrete, fully undoable history steps.

#### C. Generative AI Object Removal & Inpainting
- **Smart Brush & Edge Dilation**: Paint over unwanted distractions, power lines, or tourists with an edge-snapping brush and customizable dilation (`Grow edge (px)`).
- **Neural Backends**:
  - **Moebius (RemovalModel)**: Lightweight 0.2B inpainting diffusion model with learned embedding conditioning for realistic texture synthesis and seamless object removal without text prompts.
  - **Stable Diffusion 1.5 Inpainting**: Full generative diffusion model synthesizing context-aware fills from scratch with optional text guidance (*"clean cobblestone, morning light"*).
  - **LaMa**: High-speed Fourier convolution texture continuation for fast, crisp fills across large missing regions.
- **Live Progress & Cancellation**: Real-time progress bar displaying execution phase, completed diffusion steps, elapsed time, estimated completion time (ETA), and one-click cancel.

#### D. Neural AI Denoise
- **Clean High-ISO RAW Noise**: Eliminate sensor grain and chromatic blotches without smudging fine micro-contrast, fabric textures, or eyelashes.
- **Models**: Local **SCUNet** (clean & sharp blind denoising for real sensor noise) and **NAFNet** (lightweight half-precision denoising) with an interactive **Strength** slider (0–100%).

#### E. Biometric Face Recognition, Landmarks & Face Swap
- **People Manager (ArcFace w600k + YOLOFace)**: Automatically scans your library, detects faces, extracts deep biometric embeddings, and groups individuals into named people profiles with cross-shoot search.
- **Eye Landmarks & Blink Detection (2DFAN4)**: 68-point facial landmark regression measuring eyelid openness. Used by automated culling to flag and reject closed-eye frames in portrait bursts.
- **Face Swap (HyperSwap 1A)**: High-fidelity identity transfer preserving head pose, lighting conditions, and facial expressions.

#### F. Supported On-Device Neural Models Matrix
All models can be downloaded on-demand and configured under **Settings › AI models**:

| Model ID | Category | Architecture / Source | VRAM / RAM | Target Hardware & Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| `sam2.1-hiera-tiny` | Segmentation | Meta SAM 2.1 Hiera-Tiny (ONNX) | ~1 GB | Apple Silicon (CoreML) / DirectML / CPU |
| `sam2.1-hiera-small` | Segmentation | Meta SAM 2.1 Hiera-Small (ONNX) | ~1.5 GB | Apple Silicon (CoreML) / DirectML / CPU |
| `sam3-fp16` | Concept Select | Meta SAM 3 + CLIP ViT-B (ONNX) | ~4 GB | High-memory GPU / Apple Silicon |
| `sam3-tiny` | Concept Select | EfficientSAM3 distilled (PIXL) | ~1 GB | Apple Silicon / CPU / DirectML |
| `perspective-fields-gsv` | Alignment | ParamNet GSV (SegFormer-B3 + ConvNeXt) | ~1 GB | GPU / Apple Silicon / CPU |
| `perspective-fields-edina` | Alignment | ParamNet EDINA (360Cities) | ~1 GB | GPU / Apple Silicon / CPU |
| `moebius` | Inpainting | 0.2B Diffusion Inpainter (HUST/SimonW) | ~2.5 GB | Apple Silicon / Discrete GPU / DirectML |
| `lama-big` | Inpainting | Fast Fourier Convolutions (LaMa) | ~2 GB | CPU / Apple Silicon |
| `sd15-inpaint` | Generative Fill | Stable Diffusion 1.5 Inpaint FP16 | ~4 GB | Discrete GPU / Apple Silicon (CoreML) |
| `scunet-real-psnr` | Denoise | SCUNet Real-World Clean (Zhang et al.) | ~3 GB | GPU / Apple Silicon / CPU |
| `scunet-real-gan` | Denoise | SCUNet Real-World Sharp (Zhang et al.) | ~3 GB | GPU / Apple Silicon / CPU |
| `nafnet-sidd-w32-fp16` | Denoise | NAFNet Fast FP16 (Megvii) | ~1.5 GB | GPU / Apple Silicon / CPU |
| `arcface-faces` | People / Face | YOLOFace + ArcFace w600k | ~500 MB | CPU / Apple Silicon |
| `face-landmarks-2dfan4`| Eye Landmarks | 2DFAN4 68-Landmark Net | ~300 MB | CPU / Apple Silicon |
| `hyperswap-1a` | Face Swap | HyperSwap 1A (FaceFusion 3.3) | ~1 GB | CPU / Apple Silicon |

#### G. In-Place Adjustment Layers & Zero-Duplicate Architecture
- Traditional editors generate massive duplicate `.dng` or `.tiff` files on disk (200MB+ each) every time an AI denoise or removal is performed, quickly exhausting storage drives.
- Optixel Studio treats neural operations as **non-destructive in-place adjustment layers**: edit parameters reside in `.olxmp` metadata sidecars, and rendered patches are stored in a dedicated application cache. Master RAW photos remain untouched and zero duplicate masters are created.
- **Granular History Management**: Delete individual adjustment layers at any time via the red **✕** icon on each history step, or undo/redo at full 32-bit floating-point fidelity.

### 6. Optics & Perspective (Geometry and Detail tabs)
- **Lens Corrections**: Distortion, lateral chromatic aberration (auto-measured along contrast edges), defringe, and vignetting.
- **Upright Perspective Solver**: Automatically solve camera pitch, roll, and yaw using Hough transform line detection (**Auto**, **Level**, **Vertical**, **Full**, or **Guided** with 2–4 lines).
- **Constrain Crop**: Automatically shrink crop boundaries to prevent transparent edges after perspective corrections.
- **Film Grain & Camera Calibration**: Fine-grained analog film grain and sensor calibration adjustments.

### 7. Creative Color & Photoshop Tools (Color and Pro tabs)
Photoshop-style adjustments applied non-destructively on top of RAW develop:
- **16 Photo Grading Styles**: Instant professional looks (Bleach Bypass, Portra, Velvia, Nordic Noir, Teal & Orange variants, etc.) with real-time **0–200% Strength** scaling.
- **Levels**: Per-channel RGB black, gamma, and white points with auto 0.1% clipping.
- **Selective Color**: Adjust CMYK ink balances across 9 color ranges (reds, yellows, greens, cyans, blues, magentas, whites, neutrals, blacks).
- **Channel Mixer & Gradient Maps**: Custom monochrome channel blends and dual-tone gradient maps.
- **Filters**: Gaussian blur, image invert, posterize, and threshold.

### 8. Automated Functions & Background Batch Jobs
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
│   ├── Library.tsx          # Digital asset management & view orchestrator
│   ├── DateView.tsx         # Chronological timeline view (Year/Month/Day)
│   ├── LocationView.tsx     # OpenStreetMap map view & GPS cluster pins
│   ├── CameraView.tsx       # Equipment breakdown by camera body, lens, and ISO
│   ├── People.tsx           # Biometric face clustering & person search
│   ├── PhotoViewer.tsx      # Full-screen photo viewer, zoom, filmstrip
│   ├── Develop.tsx          # RAW develop workspace & 8-tab tool container
│   ├── AiDenoise.tsx        # AI object removal, neural denoise & progress
│   ├── Advanced.tsx         # Optics, Upright, calibration, grain, spots, proofing
│   ├── Pro.tsx              # Levels, HDR tab, hue/sat, selective color, Photoshop tools
│   ├── Auto.tsx             # 16 photo styles, strength slider, auto fine tune & auto rotate
│   ├── AutoView.tsx         # Batch auto culling & processing jobs
│   ├── Storage.tsx          # Cloud storage, network drives, clean up & duplicate
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
│   ├── src/aimodel.rs       # ONNX Runtime, SAM 2.1, PerspectiveFields, Moebius, NAFNet, SCUNet
│   ├── src/segment.rs       # Segment Anything 2.1 mask generation & point tracking
│   ├── src/perspective.rs   # PerspectiveFields PyTorch/ONNX inference & angle estimation
│   ├── src/faces.rs         # YOLOFace + ArcFace w600k face detection & recognition
│   ├── src/faceswap.rs      # HyperSwap 1A face swap & landmark alignment
│   ├── src/catalog.rs       # SQLite schema, EXIF GPS parsing, .olxmp sidecars
│   ├── src/vision.rs        # Focus peaking, active AF point detection, blur & noise analysis
│   ├── src/auto.rs          # Auto culling, rating & batch functions
│   ├── src/mcp.rs           # Embedded HTTP & stdio MCP server
│   ├── src/cloud.rs         # S3, WebDAV, Dropbox & OneDrive storage
│   ├── src/remote.rs        # Immich integration & Google Photos picker
│   └── src/ai.rs            # AI agent router & command execution
├── html/                    # Marketing landing page & static site
├── cloudflare-updater-worker/ # Cloudflare update manifest worker & download proxy
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
| **SAM 2.1 (Hiera-Tiny / Small)** | Click-to-Select Object Segmentation | Meta AI / onnx-community | Apache-2.0 | Permissive open-source license ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)). |
| **SAM 3 / EfficientSAM3** | Select by Name Concept Segmentation | Meta AI / Zeng et al. / PIXL | SAM License | Subject to Meta's SAM research terms and trade controls. |
| **PerspectiveFields (GSV & EDINA)** | Auto Horizon & Vertical Alignment | Jin et al. (University of Michigan / CVPR 2023) | MIT License | Permissive open-source license ([jinlinyi/PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)). |
| **Moebius (RemovalModel)** | Inpainting & Object Removal | Duan & Xu et al. (HUST) / SimonW | Apache-2.0 | Permissive open-source license. |
| **Stable Diffusion 1.5 Inpainting** | Generative Fill | RunwayML / Stability AI | CreativeML OpenRAIL-M | Subject to OpenRAIL-M use-based restrictions. |
| **LaMa** | Inpainting & Object Removal | Suvorov et al. (Samsung Research) | Apache-2.0 | Permissive open-source license. |
| **SCUNet** | Real-World Camera Denoise | Zhang et al. | Apache-2.0 | Permissive open-source license. |
| **NAFNet** | Fast Neural Denoise | Megvii Technology | MIT License | Permissive open-source license. |
| **YOLOFace + ArcFace w600k** | Biometric Face Detection & Recognition | InsightFace / FaceFusion | Non-Commercial Research | Research license; check terms for commercial deployments. |
| **2DFAN4** | 68-Point Eye Landmark Tracking | Bulat & Tzimiropoulos | Research License | Non-commercial research use. |
| **HyperSwap 1A** | Neural Face Swap | FaceFusion | Research License | Check FaceFusion's terms before commercial use. |
