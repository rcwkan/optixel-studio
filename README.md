<div align="center">

# Optixel Studio

**A free, open-source RAW photo editor and photo library that keeps your files and edits on your own computer, with on-device AI and your choice of AI assistant.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v0.3.1-success.svg)](https://github.com/rcwkan/optixel-studio/releases)
[![Website](https://img.shields.io/badge/Website-optixel.makecade.com-25c6d8.svg)](https://optixel.makecade.com/)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/rcwkan/optixel-studio/releases)

[**Download**](#-download--install) · [**Quick start**](#-quick-start) · [**Features**](#-features) · [**AI agents**](#-work-with-an-ai-agent) · [**Build from source**](#-build-from-source) · [**Website**](https://optixel.makecade.com/)

</div>

---

## 📸 What is Optixel Studio?

Optixel Studio organizes, culls and edits your photos where they already are: on your disk, a NAS, a cloud bucket, or your Immich server. There is no import step and your originals are never changed.

* 🔒 **Your files stay yours.** Edits are saved next to your photos (as `.olxmp` and Lightroom-compatible `.xmp` sidecars) and in a local catalog. No account, no subscription, no lock-in.
* 📷 **Serious RAW quality.** Decodes RAWs from Canon (CR3), Nikon (including High Efficiency NEF), Sony, Fujifilm X-Trans and many more, with a choice of demosaicing and highlight recovery, and edits in 32-bit floating point.
* 🗂️ **Find any photo fast.** Browse as a grid, by date, on a map, by camera and lens, or by the people in them.
* ⚡ **Cull a shoot in minutes.** Rate and flag from the keyboard, or let Auto group bursts, pick the sharpest frame and set aside the blurry and closed-eye shots.
* 🎨 **A full darkroom.** Tone, colour, curves, masks, HDR, lens and perspective correction, 16 film and colour-grade looks, and Photoshop-style adjustments, all undoable.
* 🧠 **AI that runs on your machine.** Click an object to select it, remove distractions, clean up high-ISO noise and straighten horizons, using models you download once and run offline. For plain-language editing, use a local model or a cloud AI such as Claude, Gemini or OpenAI.
* 🤖 **Let an AI agent do the busywork.** A built-in MCP server lets Claude and other AI assistants cull, edit and export for you, with every change undoable.

---

## ⬇️ Download & Install

Download the latest build from the [Releases page](https://github.com/rcwkan/optixel-studio/releases/latest):

| System | Download |
| :--- | :--- |
| **macOS 11+** (Apple Silicon, M1 or later) | `Optixel.Studio_<version>_aarch64.dmg` |
| **Windows 10/11** (x64) | `Optixel.Studio_<version>_x64-setup.exe` or `_x64_en-US.msi` |
| **Linux** Debian / Ubuntu | `Optixel.Studio_<version>_amd64.deb` |
| **Linux** Fedora / RHEL | `Optixel.Studio-<version>-1.x86_64.rpm` |
| **Linux** any distribution | `Optixel.Studio_<version>_amd64.AppImage` |

Intel Macs are not supported, because the on-device AI engine (ONNX Runtime with CoreML) ships for Apple Silicon only.

### Installing on macOS

1. Open the `.dmg` and drag **Optixel Studio** into **Applications**.
2. Open it from Applications. The app is not yet notarized by Apple, so macOS will show one of these warnings the first time:
   - **"Optixel Studio is damaged and can't be opened"**: macOS has quarantined the download. Run this in Terminal, then open the app again:
     ```bash
     xattr -cr "/Applications/Optixel Studio.app"
     ```
   - **"Apple could not verify Optixel Studio is free of malware"**: click **Done**, go to **System Settings → Privacy & Security**, scroll down and click **Open Anyway**. (On macOS 14 and earlier, Control-click the app and choose **Open**.)

> [!NOTE]
> Don't choose **Move to Trash** on the "damaged" warning. The app is fine; macOS blocks unsigned downloads by default. Use the `xattr` command above instead.

### Checking your download (optional)

Each release includes a `SHA256SUMS.txt` file. Compare it with the checksum of your download:

```bash
# macOS
shasum -a 256 Optixel.Studio_0.3.1_aarch64.dmg

# Linux
sha256sum -c SHA256SUMS.txt

# Windows (PowerShell)
Get-FileHash .\Optixel.Studio_0.3.1_x64-setup.exe -Algorithm SHA256
```

---

## 🚀 Quick Start

1. **Add your photos.** In the Library sidebar, press **＋ › Folder…** and pick a folder. Your photos stay where they are, and new files are picked up automatically.
2. **Cull.** Press <kbd>0</kbd>–<kbd>5</kbd> to rate, <kbd>P</kbd> to pick and <kbd>X</kbd> to reject. Or select a shoot, open **Auto › Catalog & Culling** and press **Run**.
3. **Edit.** Select a photo and click **Develop** at the top of the window. Start with **✦ Auto** or a look from the **Auto** tab, then fine-tune. <kbd>\\</kbd> shows the before and <kbd>⌘</kbd><kbd>Z</kbd> undoes anything.
4. **Export.** Save as JPEG, AVIF, or 16-bit PNG/TIFF to a folder, or upload straight to Immich, Google Photos or cloud storage.

---

## ✨ Features

### Organize your library

- **No importing.** Add a folder, a network drive or a cloud account and browse it in place.
- **Five ways to browse:**
  - **Grid** (<kbd>G</kbd>): thumbnails with ratings, flags and colour labels, at any size.
  - **By Date** (<kbd>D</kbd>): a timeline grouped by year, month or day.
  - **By Location** (<kbd>L</kbd>): your photos on a map, grouped into pins. **Scan GPS** fills in locations for photos added before.
  - **By Camera** (<kbd>K</kbd>): photos grouped by camera body, lens and ISO range.
  - **Viewer** (<kbd>V</kbd> or <kbd>Enter</kbd>): full-screen viewing with zoom to 100%, a filmstrip and a slideshow.
- **People** (*Library › People*): faces are found and grouped into people on your computer. Name them, merge duplicates, and show every photo of one person, or of several people together.
- **Compare** (<kbd>C</kbd> with two or more photos selected): view similar shots side by side and keep the best one.
- **Works with Lightroom.** Ratings, colour labels, copyright and crop are read from and written to standard `.xmp` sidecars, so you can move between the two apps.

### Cull faster

- **Keyboard culling:** <kbd>0</kbd>–<kbd>5</kbd> for stars, <kbd>P</kbd> pick, <kbd>X</kbd> reject, <kbd>U</kbd> unflag, <kbd>6</kbd>–<kbd>9</kbd> colour labels.
- **Filter** by rating, flag, colour label, aspect ratio or edit status.
- **Focus points:** turn on the AF overlay to see exactly where the camera focused, including eye-AF boxes (for cameras that record them).
- **Auto culling:** groups bursts, keeps the sharpest frame, rejects blurred, badly exposed and closed-eye shots, and rates the rest 1–5 stars. Turn it on for new photos and it runs in the background whenever a folder gets new files.

### Edit your photos

Every edit is non-destructive and shows up as a step in History. The Develop panel has eight tabs:

| Tab | What you can do |
| --- | --- |
| **Adjust** | White balance, exposure, contrast, highlights, shadows, whites, blacks, texture, clarity, dehaze, vibrance, saturation, B&W, tone curve, levels, and RAW decoding options |
| **Color** | 8-colour mixer, Point Color (pick a colour in the photo and change just that), 3-way colour grading, hue/saturation, selective color, channel mixer, gradient map, camera calibration, print soft-proofing |
| **Detail** | Sharpening, noise reduction, vignette, film grain, filters, spot removal (<kbd>Q</kbd>) |
| **HDR** | ✦ Auto HDR, HDR from a single photo, tone and detail controls, toning curve |
| **Geometry** | Crop (<kbd>R</kbd>), rotate and flip, Upright perspective correction (Auto, Level, Vertical, Full, or Guided with lines you draw), lens distortion, chromatic aberration, defringe, vignetting |
| **Masks** | ✦ Select subject, sky or background; click any object to select it (<kbd>S</kbd>); type what to select; linear, radial, brush, luminance and colour range masks |
| **Auto** | ✦ Auto, 16 looks (Portra 400, Fuji Velvia, Bleach Bypass, Nordic Noir, Golden Hour, Teal & Orange and five variants, and more) with a 0–200% strength slider, AI fine tune, auto rotate |
| **AI** | AI auto adjust, auto straighten horizon and verticals, AI denoise, remove objects, face swap |

Select a mask and the Adjust, Color, Detail and HDR tools change only that area.

**Helpful while editing:**
- A live histogram with clipping warnings (<kbd>J</kbd>), plus RGB Parade and Waveform scopes.
- Before/after (<kbd>\\</kbd>), side-by-side (<kbd>Y</kbd>) and split (<kbd>U</kbd>) views.
- Copy and paste settings between photos (<kbd>⌘</kbd><kbd>⇧</kbd><kbd>C</kbd> / <kbd>⌘</kbd><kbd>⇧</kbd><kbd>V</kbd>).
- **Soft-proofing:** load your printer's paper profile (`.icc`) to preview a print on screen, with colours the printer can't reproduce highlighted.
- **RAW options:** choose the demosaicing method (RCD, AMaZE, LMMSE and more), how blown highlights are recovered, the black point, and sensor-level noise reduction.

### AI tools that run on your computer

These run offline on your own hardware. Download the ones you want under **Settings › AI models**; each download is checked before use.

- **Click to select** (<kbd>S</kbd>): click any object (a person, a car, a building, the sky) to get a precise mask. <kbd>⌥</kbd>/<kbd>Alt</kbd>-click removes areas from it. You can also type what you want, such as *"red umbrella"*, and every match is selected. (Without a Segment Anything model installed, selection uses your chosen AI provider instead.)
- **Remove distractions:** paint over people, power lines or clutter and they are filled in to match the surroundings. Choose a fast model or a generative one, follow the progress, and cancel at any time.
- **AI denoise:** clean up high-ISO noise while keeping fine detail, with a strength slider.
- **Auto straighten:** levels horizons and straightens converging verticals in buildings, as an undoable step.
- **Blink detection:** finds closed eyes so Auto culling can reject those frames.
- **No extra files.** AI edits are stored as layers in your edit, not as separate 200 MB TIFF copies. Delete any of them from History at any time.

### Automate whole shoots

Open the **Auto** panel in the Library to run any of these on a selection, the current view or the whole catalog. Each can be cancelled, and results open in the Library.

| Function | What it does |
| --- | --- |
| **Catalog & Culling** | Rates, flags and keywords photos; groups bursts and keeps the best |
| **Non-destructive Editing** | Applies a look to each photo (Natural, Vivid, Portrait, Landscape, Film, Moody, B&W and more), or lets AI choose one per photo |
| **AI Masking** | Creates subject, sky, background or people masks |
| **AI Denoise & Super Resolution** | Removes noise and enlarges photos |
| **HDR Merge** | Merges bracketed exposures into one aligned, ghost-free HDR photo |
| **Find blurred subjects** | Flags missed focus and motion blur |
| **Find duplicates** | Finds exact and visually similar copies so you can keep one |

### Export and share

- **Formats:** JPEG, AVIF, 16-bit PNG and 16-bit TIFF.
- **Colour spaces:** sRGB, Display P3, Adobe RGB, ProPhoto RGB, ACEScg, and Rec.2020 PQ for HDR displays.
- **Destinations:** a folder, Immich, Google Photos, cloud storage, or your iCloud, Amazon Photos or Adobe sync folder. Exports keep the capture date, so they land on the right day.
- **Lossless JPEG rotate and crop:** JPEGs are rotated and cropped without re-compressing them.

### Storage and cloud services

| Type | Supported |
| --- | --- |
| **Local and network drives** | Folders and external drives; NAS over SMB, AFP, NFS or WebDAV (a dot shows whether it's connected) |
| **Self-hosted** | **Immich** (browse in place, ratings and flags sync both ways, saved edits stack on top of the original), Nextcloud, ownCloud, Synology, pCloud, Koofr |
| **Object storage** | Amazon S3, Cloudflare R2, Backblaze B2, Wasabi, MinIO |
| **Cloud services** | Dropbox, OneDrive, Google Photos (pick photos to import, upload exports) |
| **Desktop sync folders** | Apple iCloud Photos, Amazon Photos, Adobe Cloud Photos |

- Cloud photos download on demand into a local cache.
- **Duplicate** (*Settings › Storage › Duplicate*) copies or backs up photos between any two sources, checking every file and keeping your edits.
- Passwords, API keys and sign-in tokens are kept in your system keychain (macOS Keychain, Windows Credential Manager or Linux Secret Service), never in a settings file.

---

## 🤖 Work with an AI Agent

### The ✦ AI bar

In Develop, type what you want in plain words. Pick your AI provider in **Settings › Agent & MCP**.

```text
select the castle and warm it up a little
remove the people on the left
level the horizon and make it moody
```

Supported providers:
- **On your computer:** Ollama (for example `qwen2.5-vl` or `llama3.2-vision`), LM Studio, vLLM, llama.cpp
- **Cloud:** Anthropic Claude, Google Gemini, OpenAI, OpenRouter
- **Command-line agents:** Claude Code, Antigravity, or your own command

### Connect Claude or any MCP client

Optixel Studio runs an MCP server on your computer, so AI assistants can work with your library: find photos, cull, edit, mask, remove objects and export. Every change is a normal, undoable history step, and the open Develop view updates as the agent works.

Get your token and ready-to-copy setup from **Settings › Agent & MCP**. The green **MCP** dot in the title bar shows the server is running.

```bash
claude mcp add --transport http optixel http://127.0.0.1:41920/mcp --header "Authorization: Bearer <your-token>"
```

Then ask, for example: *"Cull the Iceland shoot, grade the picks for landscape, and export them for Immich."*

<details>
<summary><b>MCP tools</b></summary>

| Tool | What it does |
| --- | --- |
| `catalog_query` | Find photos by folder, rating, flag or keyword |
| `culling_batch_update` | Set ratings, flags, labels and keywords |
| `photo_inspect` | Get a preview, histogram and exposure data |
| `develop_apply_parameters` | Set any Develop value |
| `develop_auto` | Auto-edit to a look |
| `auto_run` | Run any of the seven Auto functions |
| `mask_create_and_edit` | Create and adjust masks |
| `mask_select_by_name` | Select objects by describing them |
| `retouch_remove_objects` | Remove objects |
| `ai_reconstruct_and_retouch` | AI denoise, super resolution and generative fill |
| `batch_export` | Render, save and upload |

</details>

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| --- | --- |
| <kbd>G</kbd> · <kbd>D</kbd> · <kbd>L</kbd> · <kbd>K</kbd> | Grid · By Date · By Location · By Camera |
| <kbd>V</kbd> / <kbd>Enter</kbd> | Open the viewer |
| <kbd>Esc</kbd> | Back to the grid, or deselect a mask |
| <kbd>C</kbd> | Compare the selected photos |
| <kbd>0</kbd>–<kbd>5</kbd> | Star rating |
| <kbd>P</kbd> · <kbd>X</kbd> · <kbd>U</kbd> | Pick · Reject · Unflag (in Develop, <kbd>U</kbd> toggles split view) |
| <kbd>6</kbd>–<kbd>9</kbd> | Colour label |
| <kbd>S</kbd> | Click-to-select tool |
| <kbd>Q</kbd> | Spot removal |
| <kbd>R</kbd> | Crop and straighten |
| <kbd>O</kbd> · <kbd>I</kbd> | Show mask overlay · Invert mask |
| <kbd>J</kbd> | Clipping warnings |
| <kbd>\\</kbd> · <kbd>Y</kbd> | Before/after · Side by side |
| <kbd>[</kbd> <kbd>]</kbd> | Rotate 90° left / right |
| <kbd>Z</kbd> · <kbd>F</kbd> | Zoom to 100% · Fit to screen (hold <kbd>Space</kbd> to pan) |
| <kbd>⌘</kbd><kbd>Z</kbd> · <kbd>⌘</kbd><kbd>⇧</kbd><kbd>Z</kbd> | Undo · Redo |
| <kbd>⌘</kbd><kbd>⇧</kbd><kbd>C</kbd> · <kbd>⌘</kbd><kbd>⇧</kbd><kbd>V</kbd> | Copy · Paste develop settings |

On Windows and Linux, use <kbd>Ctrl</kbd> in place of <kbd>⌘</kbd>.

---

## 🧠 AI Models

All models are optional and downloaded only when you choose them in **Settings › AI models**. Each has its own license; check it before commercial use.

| Model | Used for | Memory | License |
| :--- | :--- | :---: | :--- |
| SAM 2.1 Tiny / Small (Meta) | Click to select | ~1–1.5 GB | Apache-2.0 |
| SAM 3 / EfficientSAM3 (Meta, PIXL) | Select by typing what you want | ~1–4 GB | SAM License (Meta research terms) |
| PerspectiveFields GSV / EDINA (Univ. of Michigan) | Straighten horizons and verticals (GSV for city and architecture, EDINA for interiors and landscapes) | ~1 GB | MIT |
| Moebius 0.2B (HUST) | Remove objects | ~2.5 GB | Apache-2.0 |
| LaMa (Samsung Research) | Remove objects, fast | ~2 GB | Apache-2.0 |
| Stable Diffusion 1.5 Inpainting (RunwayML / Stability AI) | Generative fill, with optional text prompt | ~4 GB | CreativeML OpenRAIL-M |
| SCUNet Clean / Sharp | AI denoise | ~3 GB | Apache-2.0 |
| NAFNet (Megvii) | AI denoise, fast | ~1.5 GB | MIT |
| YOLOFace + ArcFace (InsightFace / FaceFusion) | Find and recognize people | ~500 MB | Non-commercial research |
| 2DFAN4 | Blink detection | ~300 MB | Non-commercial research |
| HyperSwap 1A (FaceFusion) | Face swap | ~1 GB | Research; check FaceFusion's terms |

Models run on Apple Silicon (CoreML), on Windows GPUs (DirectML), or on the CPU.

---

## 🛠️ Under the Hood

- **App:** [Tauri v2](https://tauri.app) with a Rust core and a [React 19](https://react.dev) + TypeScript + [Tailwind CSS v4](https://tailwindcss.com) interface. The catalog is SQLite.
- **RAW decoding:** [LibRaw](https://www.libraw.org/) built in, with RCD, AMaZE, LMMSE, AHD, DCB, VNG, PPG and bilinear demosaicing, per-channel black levels from the sensor's masked border, and several highlight-recovery modes. [rawler](https://crates.io/crates/rawler) is the pure-Rust fallback.
- **Colour:** one pipeline renders the viewport, previews, MCP previews and exports in 32-bit float linear Rec.2020, so what you see is what you export. [LittleCMS 2](https://www.littlecms.com/) handles ICC profiles and soft-proofing; `jxl-oxide` handles JPEG XL.
- **GPU:** [wgpu](https://wgpu.rs/) compute shaders for clarity, dehaze, 3D LUTs, defringing and scopes, compiled to SPIR-V at build time.
- **Images:** [libjpeg-turbo](https://libjpeg-turbo.org/) for lossless JPEG transforms; Apple ImageIO / [libheif](https://github.com/strukturag/libheif) for HEIC and 10-bit AVIF.
- **AI:** ONNX Runtime with CoreML, DirectML or CPU.

For architecture details, see [`guide/README.md`](guide/README.md) and the [Functional & System Specification](optixel_studio_functional_and_system_specification.md).

---

## 💻 Build from Source

You need **Node.js 20+** and stable **Rust** (`rustup update stable`).

```bash
git clone https://github.com/rcwkan/optixel-studio.git
cd optixel-studio
npm install

# Run in development mode
npm run tauri dev

# Build installers for your platform
npm run tauri build

# Run the tests
cargo test --manifest-path src-tauri/Cargo.toml
npm run build
```

The catalog is stored in `~/Library/Application Support/org.optixel.studio/` on macOS. Set `OPTIXEL_DATA_DIR` to use another folder.

### Command line

```bash
optixel-studio                          # open the app
optixel-studio --import /path/to/photos # add a folder and build previews, without the window
optixel-studio --mcp-stdio              # MCP over stdio (uses the running app, or the catalog directly if it's closed)
```

### Project structure

```
├── src/                   # React interface
│   ├── App.tsx            # App shell, navigation, shortcuts
│   ├── Library.tsx        # Library and its views (DateView, LocationView, CameraView, People, PhotoViewer)
│   ├── Develop.tsx        # Develop workspace and its tabs
│   ├── Advanced.tsx       # RAW options, optics, Upright, calibration, grain, spots, soft-proofing
│   ├── Pro.tsx            # Levels, HDR, hue/sat, selective color and other Photoshop-style tools
│   ├── Auto.tsx           # Looks, AI fine tune, auto rotate
│   ├── AutoView.tsx       # Batch Auto functions
│   ├── AiDenoise.tsx      # AI removal and denoise
│   ├── Storage.tsx        # Cloud storage, network drives, Duplicate
│   └── Agent.tsx          # AI providers and MCP settings
├── src-tauri/             # Rust core
│   ├── csrc/              # Native C/C++ code (RAW, demosaicing, codecs)
│   ├── shaders/           # GPU compute shaders
│   ├── vendor/libraw/     # LibRaw
│   └── src/               # develop, libraw, icc, gpu, xmp, catalog, aimodel, segment,
│                          # perspective, faces, auto, mcp, cloud, remote, ai, ...
└── guide/                 # Developer guides
```

---

## 📄 License & Security

- Optixel Studio is released under the [MIT License](LICENSE). Downloadable AI models have their own licenses (see [AI Models](#-ai-models)).
- To report a security issue, see [`SECURITY.md`](SECURITY.md).
