# Optixel Studio — Functional & System Specification

**Version:** 1.0.0-draft  
**License:** MIT License  
**Target Platforms:**  
- **Desktop Native:** macOS (Apple Silicon / Intel), Windows (x64 / ARM64), Linux (Wayland / X11) via **Tauri v2 + Rust + WebGPU (`wgpu`)**  
- **Web Application:** Progressive Web App (PWA) & Self-Hosted Server via **Rust-to-WASM + WebGPU + Origin Private File System (OPFS) + Optional Headless Studio Daemon**

---

## 1. Executive Summary & Architectural Vision

**Optixel Studio** is a professional, non-destructive RAW image processor, digital asset manager (DAM), and studio production suite designed to achieve 100% functional parity with both **Adobe Lightroom Classic** and **Adobe Lightroom (Cloud/Desktop/Mobile/Web)** while introducing two next-generation open architectures:

1. **Native Model Context Protocol (MCP) Server:** Every catalog query, culling metric, non-destructive develop slider, boolean AI mask, multi-shot merge, and export pipeline is exposed via a built-in MCP server (`stdio`, `SSE`, and `Streamable HTTP`). External AI agents can inspect histograms, view rendered previews, and orchestrate complex photo editing workflows autonomously.
2. **Pluggable AI & CLI Agent Provider Router:** Users can route AI-assisted image tuning, semantic search, prompt-based color grading, and generative retouching through their choice of execution backend:
   - **CLI Agents:** **Claude CLI** (`claude`), **Antigravity CLI** (`antigravity` / `agy`), or custom shell/CLI agent pipelines.
   - **Local Inference APIs:** **Ollama API** (`/api/chat` with vision models like `qwen2.5-vl`, `llama3.2-vision`, `gemma3`), **LM Studio**, **vLLM**, **llama.cpp**, and **ComfyUI** (for local diffusion inpainting/upscaling).
   - **Cloud APIs:** Anthropic API, Google Gemini API, OpenAI API, OpenRouter, and Replicate/Fal.ai.
   - **Embedded Local Neural Engine:** Built-in `ONNX Runtime` / `Candle` (Rust) execution for offline RAW AI Denoise, Super Resolution, Depth Map estimation, and SAM-2 segmentation.

---

## 2. System Architecture & Technology Stack

### 2.1 High-Level System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                  USER INTERFACE LAYER (Frontend)                                 |
|  React 19 + TypeScript + Tailwind CSS + WebGPU Canvas Viewport (SDR / 32-bit Float EDR/HDR)      |
|  Modules: Library | Develop | Map | Book | Print | Slideshow | Web | AI Agent & MCP Console      |
+------------------------------------------------+-------------------------------------------------+
                                                 | IPC (Tauri Commands) / WebSockets / WASM Bridge
+------------------------------------------------v-------------------------------------------------+
|                                 OPTIXEL CORE ENGINE (Rust)                                       |
|                                                                                                  |
|  +-----------------------------+  +-----------------------------+  +--------------------------+  |
|  |   Non-Destructive Develop   |  |     DAM & Vector Catalog    |  |   Studio & I/O Engine    |  |
|  |   DAG Pipeline (wgpu 32f)   |  |  SQLite + sqlite-vec + XMP  |  |  libgphoto2 / C2PA / ICC |  |
|  +-----------------------------+  +-----------------------------+  +--------------------------+  |
|                                                                                                  |
|  +---------------------------------------------+  +-------------------------------------------+  |
|  |         BUILT-IN MCP SERVER HOST            |  |      PLUGGABLE AI PROVIDER ROUTER         |  |
|  |  Transports: stdio | HTTP-SSE | WebSocket   |  |  - CLI Runners: Claude CLI, Antigravity   |  |
|  |  Exposes: Tools, Resources, Prompts         |  |  - Local API: Ollama, vLLM, LM Studio     |  |
|  |  Bi-directional state sync & undo stack     |  |  - Cloud API: Claude, Gemini, OpenAI      |  |
|  |                                             |  |  - Pixel AI: ONNX/Candle + ComfyUI Bridge |  |
|  +---------------------------------------------+  +-------------------------------------------+  |
+--------------------------------------------------------------------------------------------------+
```

### 2.2 Core Technology Stack

| Subsystem | Desktop Native Implementation | Web App Implementation |
|---|---|---|
| **Application Shell** | **Tauri v2** (Rust backend + OS WebView) | **PWA** + Service Workers + Optional Self-Hosted Rust Backend Daemon |
| **RAW Decoding** | **`LibRaw`** (Native C++ core: CR3, NEF HE, ARW, RAF X-Trans) + **`rawler`** (Pure Rust fallback) | **`rawler`** compiled to **WASM** (`wasm32-unknown-unknown` with SIMD128) |
| **Image Processing GPU** | **`wgpu`** AOT SPIR-V Compute Kernels over Metal (macOS), Vulkan (Linux/Windows), DX12 | **`wgpu`** over browser **WebGPU API** |
| **Working Color Space** | 32-bit Floating Point Linear **Rec. 2020** / **ProPhoto RGB** / **ACEScg** | 32-bit Floating Point Linear **Rec. 2020** in WebGPU compute shaders |
| **Color Management** | **`lcms2`** (LittleCMS 2 static) + **`moxcms`** for ICC v2/v4, printer soft-proofing, gamut warnings | **`qcms`** (WASM) + WebGPU 3D LUT tetrahedral interpolation shader |
| **Database & DAM** | **SQLite 3** (WAL mode) + **`sqlite-vec`** for vector embeddings | **SQLite-WASM** backed by **Origin Private File System (OPFS)** + IndexedDB |
| **Local Neural Inference** | **`ort`** (ONNX Runtime) with CoreML, DirectML, TensorRT, CUDA, ROCm | **`ort-web`** (WebGPU / WebNN execution providers) |
| **Tethering Engine** | **`gphoto2-rs`** (`libgphoto2` bindings) + Native Vendor SDK plugins | **WebUSB** / **WebTransport** PTP/IP camera bridge |
| **Provenance Signing** | **`c2pa-rs`** (Coalition for Content Provenance and Authenticity) | **`c2pa-wasm`** cryptographic manifest injector |

---

## 3. Complete Functional Specification

### 3.1 Module 1: Digital Asset Management (DAM) & Organization

#### 3.1.1 Hybrid Local Catalog & Cloud-Sync Architecture
* **Catalog Modes (`FR-DAM-01`):**
  * **Classic Catalog Mode:** Uses a local `.olcat` SQLite database tracking file paths on local/NAS volumes without moving files unless requested. Automatically reads/writes standard `.xmp` sidecar files for 100% compatibility with Adobe Camera Raw/Lightroom.
  * **Direct Local Folder Mode:** Browse and edit any directory on disk immediately without prior catalog import, caching metadata in a lightweight sidecar index.
  * **Cloud/Self-Hosted Sync Mode:** Syncs original RAWs, Smart Previews, and non-destructive edit graphs across Desktop, Web, and Mobile via S3-compatible object storage (AWS S3, Cloudflare R2, MinIO, Nextcloud) and CRDT (Conflict-Free Replicated Data Type) edit logs.
* **Smart Previews (`FR-DAM-02`):**
  * Generates lossy 2560px long-edge 10-bit DNG or AVIF floating-point proxies on import. Allows full offline tonal, color, and masking edits when external drives are disconnected.

#### 3.1.2 AI Assisted Culling & Facial Recognition
* **Assisted Culling Engine (`FR-DAM-03`):**
  * Runs an on-import or on-demand pass detecting all faces and subjects in a shoot.
  * **Eye Sharpness Score (0–100):** Computes Laplacian variance and high-frequency MTF across detected eye bounding boxes.
  * **Eyes Open Status (`OPEN`, `BLINK`, `OCCLUDED`):** Uses facial landmark Eye Aspect Ratio (EAR) + CNN classifier to flag closed eyes.
  * **Burst Grouping & Best-Shot Recommendation:** Groups bursts by capture timestamp proximity and visual perceptual hash (`pHash`), automatically highlighting the sharpest, eyes-open frame.
* **Facial Recognition / People View (`FR-DAM-04`):**
  * Extracts 512-dimensional facial embeddings (via local InsightFace/ArcFace ONNX model) stored in `sqlite-vec`.
  * Clusters unnamed faces via HDBSCAN, allowing one-click naming, merging, and filtering by person across the entire library.

#### 3.1.3 Search, Smart Collections & Comparison Views
* **Smart Collections & Metadata Filtering (`FR-DAM-05`):**
  * Boolean rule builder (`ALL`, `ANY`, `NONE`) filtering across EXIF (ISO, Aperture, Shutter Speed, Focal Length, Lens, Camera Serial), Star Ratings (`0–5`), Flags (`Pick`, `Unflagged`, `Reject`), Color Labels (`Red`, `Yellow`, `Green`, `Blue`, `Purple`), Aspect Ratio, and **AI Edit Status** (`Has AI Denoise`, `Has Generative Remove`, `Has AI Mask`, `Untouched`).
* **AI Semantic Natural-Language Search (`FR-DAM-06`):**
  * Generates image embeddings using a local CLIP/SigLIP vision encoder (or via configured Vision API/Ollama provider).
  * Supports natural language queries (e.g., *"bride laughing under umbrella at night"*) ranked by cosine similarity in `sqlite-vec`.
* **Hierarchical Keywording & IPTC Templates (`FR-DAM-07`):**
  * Unlimited parent-child keyword trees with synonym export controls, IPTC Extension metadata editors, and batch import presets.
* **Virtual Copies, Stacks, Survey & Compare Views (`FR-DAM-08`):**
  * **Virtual Copies:** Zero-copy database branches pointing to the same master RAW file with distinct edit histories.
  * **Survey View (`N`):** Dynamic $N$-up grid scaling to inspect and deselect candidates.
  * **Compare View (`C`):** Locked zoom/pan 2-up Select vs. Candidate view with pixel-level difference overlay.

---

### 3.2 Module 2: Core Global Editing & Tone Controls

All tonal math executes in a 32-bit floating-point linear working space before display transform.

```
[RAW Sensor Bayer/X-Trans]
   │ 1. Pre-Demosaic AI Denoise & Flat-Field Correction
   ▼
[Linear Demosaic & Camera Calibration (Primary R, G, B Hue/Sat)]
   │ 2. Lens Profile (Distortion, Vignette, CA) & Exposure / White Balance
   ▼
[RAW Profile / DCP / 3D LUT Base Transform]
   │ 3. Global Tone (Highlights, Shadows, Whites, Blacks) & Presence (Texture, Clarity, Dehaze)
   ▼
[Local AI Masks & Boolean Mask Stack (Per-Mask Tone, Point Color, Curve, Sharpness)]
   │ 4. Parametric & RGB Point Tone Curves (with Refine Saturation)
   ▼
[Color Mixer (8-Band HSL) + Point Color (3D HSL) + 3-Way Color Grading Wheels]
   │ 5. Detail (Sharpening/Noise) + AI Lens Blur (Depth Bokeh) + Grain/Vignette
   ▼
[Geometry / Guided Upright / Crop] ──► [Output Transform: SDR ICC Soft Proof OR HDR Rec.2020/P3]
```

* **RAW Profiles & Camera Matching (`FR-TONE-01`):**
  * Supports Adobe DCP (`.dcp`) dual-illuminant camera profiles, ICC profiles, open-source camera-matching film simulations (Fujifilm Provia/Velvia/Classic Chrome equivalents, Nikon/Canon/Sony equivalents), and custom `.cube` 3D LUTs with an **Amount** slider (0–200%).
* **Interactive Histogram & Basic Tone Panel (`FR-TONE-02`):**
  * Real-time GPU-computed RGB + Luminance histogram with clickable Shadow (`J`) and Highlight clipping indicators.
  * Dragging directly on the histogram adjusts **Blacks**, **Shadows**, **Exposure**, **Highlights**, and **Whites** (-100 to +100).
  * **White Balance:** Temperature (Kelvin `2000K–50000K`) and Tint (`-150` to `+150`) with neutral gray eyedropper and skin-tone target mode.
* **Tone Curve (`FR-TONE-03`):**
  * **Parametric Curve:** Highlights, Lights, Darks, Shadows sliders with adjustable region split points (`0–100`).
  * **Point Curve & Per-Channel RGB:** Cubic spline and linear interpolation modes across Luminance, Red, Green, and Blue channels.
  * **Refine Saturation Slider (`0–200`):** Decouples luminance contrast from chrominance shifts using perceptual Oklab/YCbCr compensation so steep S-curves do not over-saturate skin tones.
* **Presence Controls (`FR-TONE-04`):**
  * **Texture (`-100` to `+100`):** Multi-scale wavelet decomposition targeting mid-high spatial frequencies (`2px–8px` radius) while protecting noise floor and large edges.
  * **Clarity (`-100` to `+100`):** Local Laplacian pyramid midtone contrast enhancement without haloing.
  * **Dehaze (`-100` to `+100`):** Dark-channel prior atmospheric scattering estimation to remove or inject physical depth haze.
  * **Vibrance & Saturation (`-100` to `+100`):** Oklch chroma scaling where Vibrance applies a bell-curve weight that protects already-saturated hues and portrait skin-tone hue angles (`25°–55°`).
* **Native C++ RAW Processing & Multi-Algorithm Demosaicing (`FR-RAW-NATIVE`):**
  * Embedded **LibRaw** C++ decoding for proprietary compressed formats: Canon `.CR3` (C-RAW), Nikon High-Efficiency `.NEF`, Sony uncompressed/lossless `.ARW`, and Fujifilm X-Trans 6x6 arrays.
  * **Pro Demosaic Engine:** Switchable demosaicing algorithms: **RCD** (Ratio Corrected Demosaicing for maximum sharpness), **AMaZE** (Aliasing Minimization for low-ISO landscapes), **LMMSE** (directional noise-adaptive demosaicing for high-ISO and astrophotography), AHD, Linear, VNG, and DCB.
  * **Optical Calibration & Highlight Recovery:** Reads per-channel optical black-point borders from the physical sensor borders and executes multi-channel highlight reconstruction before tonemapping.
* **Studio Color Management & Printer Soft-Proofing (`FR-COLOR-STUDIO`):**
  * **LittleCMS 2 (`lcms2`)** pipeline supporting 32-bit floating-point **ProPhoto RGB** and **ACEScg** working color spaces.
  * **Interactive Soft-Proofing:** Load custom `.icc` paper/printer profiles (e.g. Hahnemühle, Epson, Canon) to simulate ink black-point compensation.
  * **Out-of-Gamut Warning:** Renders neon highlight overlays on pixels exceeding printable ink thresholds.
  * **JPEG XL XYB Fidelity:** Pairs `jxl-oxide` with LittleCMS 2 (`jxlcms`) for perceptual XYB color spaces and HDR PQ/HLG transfer functions.
* **Lossless Container I/O, Active AF Points & Sidecars (`FR-CONTAINER-NATIVE`):**
  * **Active AF Point Overlay (<kbd>A</kbd>):** Decodes encrypted camera MakerNotes via native Exiv2/LibRaw to render exact autofocus points and eye-AF bounding boxes over the photo during culling, alongside lens serial numbers and mechanical shutter counts.
  * **Bidirectional Lightroom `.XMP` Sync:** Reads and writes standard Adobe XMP sidecars preserving ratings, color labels, copyright IPTC metadata, and crop rectangles.
  * **SIMD Lossless JPEG (turbojpeg):** Hardware-accelerated lossless JPEG cropping, 90°/180°/270° rotation, and flipping directly in DCT frequency space without generation loss; Apple ImageIO & libheif for HEIC and 10-bit AVIF.
* **Ahead-of-Time (AOT) GPU Compute Shader Pipeline (`FR-GPU-AOT`):**
  * Pre-compiled binary SPIR-V bytecode generated at build time for Local Clarity, Dark-Channel Dehaze, 3D LUT trilinear interpolation, axial defringing, and Real-Time RGB Parade & Waveform Scopes.

---

### 3.3 Module 3: Advanced Color Grading & Precision Color Tools

* **Point Color — 3D HSL Swatching (`FR-COLOR-01`):**
  * Up to 8 independent color swatches per global panel or per local mask.
  * **Eyedropper Sampling:** Plots the sampled pixel in a 3D Hue-Saturation-Luminance visualizer (2D Hue/Sat plane + vertical Luminance bar).
  * **Shift Controls:** Hue Shift (`-100` to `+100`), Saturation Shift (`-100` to `+100`), Luminance Shift (`-100` to `+100`).
  * **3D Range & Falloff:** Independent **Hue Range**, **Saturation Range**, **Luminance Range**, and global **Falloff** sliders (`0–100`) defining a smooth 3D ellipsoid in color space.
  * **Visualize Range Toggle:** Renders unselected pixels in grayscale on the main canvas while adjusting range boundaries.
* **3-Way Color Grading Wheels (`FR-COLOR-02`):**
  * Dedicated color wheels for **Shadows**, **Midtones**, **Highlights**, and **Global** tinting.
  * Per-wheel **Hue (`0°–360°`)**, **Saturation (`0–100`)**, and **Luminance (`-100` to `+100`)** sliders with solo-wheel zoom and eyedropper pickers.
  * **Blending (`0–100`)** and **Balance (`-100` to `+100`)** sliders controlling the crossover curves between shadow, midtone, and highlight masks.
* **8-Band HSL / Color Mixer (`FR-COLOR-03`):**
  * Adjusts Hue, Saturation, and Luminance across Red, Orange, Yellow, Green, Aqua, Blue, Purple, and Magenta bands.
  * Includes on-canvas **Target Adjustment Tool (TAT)** allowing vertical/horizontal cursor drags directly on the photo to modify the underlying color band(s).
* **Camera Calibration Panel (`FR-COLOR-04`):**
  * Operates on the primary sensor R, G, B channels prior to profile mapping: **Shadows Tint**, **Red Primary (Hue/Sat)**, **Green Primary (Hue/Sat)**, and **Blue Primary (Hue/Sat)**.

---

### 3.4 Module 4: Advanced AI, Masking & Generative Pipeline

#### 3.4.1 Semantic AI Masking & Anatomical Parsing
* **Automated Mask Generators (`FR-AI-01`):**
  * **Select Subject / Select Background:** High-precision alpha matting (`SAM-2` + `BiRefNet`/`MODNet` hair-strand matting).
  * **Select Sky:** Horizon and tree-canopy aware sky alpha segmentation.
  * **Select People (Anatomical Sub-Masks):** Detects each person in the frame and exposes checkboxes to generate combined or separate masks for:
    1. *Entire Person*
    2. *Facial Skin*
    3. *Body Skin*
    4. *Eyebrows*
    5. *Eye Sclera*
    6. *Iris & Pupil*
    7. *Lips*
    8. *Teeth*
    9. *Hair*
    10. *Facial Hair / Beard*
    11. *Clothes*
  * **Select Landscape (Environmental Sub-Masks):** Segments outdoor scenes into:
    1. *Sky*
    2. *Mountains*
    3. *Architecture*
    4. *Vegetation*
    5. *Water*
    6. *Natural Ground*
    7. *Artificial Ground*
  * **Select Objects:** Interactive bounding-box or rough brush scribble prompt fed into `SAM-2` for instant object isolation.
* **Range & Parametric Masks (`FR-AI-02`):**
  * **Color Range Mask:** Eyedropper + Amount slider.
  * **Luminance Range Mask:** Upper/lower tonal threshold + feather ramp with monochromatic luminance map preview.
  * **Depth Range Mask:** Isolates foreground/midground/background layers using hardware depth maps (HEIC/DNG) or synthetic monocular depth (`Depth Anything V2`).
* **Boolean Mask Algebra & Non-Destructive Refinement (`FR-AI-03`):**
  * Every Mask Layer is a boolean tree of Mask Components (`Add`, `Subtract`, `Intersect`, `Invert`).
  * Any AI mask can be intersected or subtracted with **Linear Gradient**, **Radial Gradient**, **Brush**, or **Range Masks**.
  * **Mask Refinement Sliders:** Non-destructive **Feather (`-100` to `+100`)** and **Edge/Contract-Expand (`-100` to `+100`)** morphological guided-filter sliders on any mask component.
  * **Batch AI Mask Recalculation:** Copying masks across different photos recalculates the semantic AI masks on the target photo automatically.

#### 3.4.2 Generative Remove & Automated Distraction Removal
* **Generative Remove (`FR-AI-04`):**
  * Brush over an object (with adjustable brush size/opacity and add/subtract sub-strokes).
  * **Shadow & Reflection Detection:** Automatically expands the mask to include cast shadows and water/floor reflections associated with the brushed object.
  * **Variations & Provider Routing:** Generates 3 candidate variations per removal pass using either the local diffusion backend (`ComfyUI` / `Flux-Fill` / `SDXL-Inpaint` / `LaMa` ONNX) or a configured Cloud/CLI AI provider, stored non-destructively in a patch layer.
* **Automated Distraction Removal (`FR-AI-05`):**
  * **Reflection Removal:** Dual-layer separation model that separates transmitted scene light from glass/window reflections with a **Reflection Amount (`0–100`)** slider.
  * **People Removal:** Automatically detects non-primary background pedestrians/tourists, previews highlighted overlays, and removes them in one click.
  * **Sensor Dust Removal:** Automatically detects circular low-contrast sensor dust spots across sky/uniform regions and heals them non-destructively (with a **Visualize Spots** threshold overlay).
* **Prompt-to-Edit & Generative Expand (`FR-AI-06`):**
  * **Prompt-to-Edit:** Natural language instruction bar inside the Develop module. Depending on user mode, it either translates the prompt into non-destructive parametric edits + AI masks (via the AI Provider Router) or runs a localized instruction-guided generative edit.
  * **Generative Expand (Outpainting):** Extending the Crop box outside the original image canvas synthesizes matching border pixels to fill the target aspect ratio.

#### 3.4.3 AI Image Reconstruction & Optical Simulation
* **AI Denoise (`FR-AI-07`):**
  * Operates directly on raw Bayer or X-Trans mosaic data before demosaicing, outputting either a non-destructive virtual layer or a 16-bit Linear DNG (`Amount: 0–100`).
* **Super Resolution & Generative Upscale (`FR-AI-08`):**
  * **Super Resolution:** 2x linear (4x total megapixel) RAW detail synthesis.
  * **Generative Upscale:** Restores heavily cropped or low-resolution legacy JPEGs/TIFFs up to 4x using local ESRGAN/SwinIR/SUPIR or external provider APIs.
* **AI Lens Blur (`FR-AI-09`):**
  * Computes a 16-bit disparity depth map via `Depth Anything V2` (or native camera Lidar/Dual-Pixel depth).
  * **Controls:** Blur Amount (`0–100`), **5 Bokeh Shapes** (*Circle*, *Bubble / Soap*, *5-Blade Polygonal*, *Ring / Catadioptric*, *Cat Eye / Optical Vignette*), **Bokeh Boost** (physically amplifies highlight specular orbs before convolving), interactive **Focal Range** depth histogram slider, and manual **Focus / Blur Refinement Brushes**.

---

### 3.5 Module 5: High Dynamic Range (HDR) & Multi-Shot Composites

* **True HDR Editing & Display Pipeline (`FR-HDR-01`):**
  * One-click **HDR Toggle** switches the WebGPU swapchain to `rgba16float` extended dynamic range (EDR on macOS, Windows ACM HDR, Wayland HDR, Chrome WebGPU `toneMapping: { mode: "extended" }`).
  * **Split SDR/HDR Histogram:** Displays standard dynamic range (`0.0–1.0`) alongside +4.0 EV stops of HDR highlight headroom (`1.0–16.0`), with a **Visualize HDR by Stop** false-color overlay.
  * **SDR Preview & Tone Mapping:** Dedicated collapsible sub-panel (`SDR Brightness`, `SDR Clarity`, `SDR Contrast`, `Highlight Saturation`) to control how the HDR master renders on standard SDR displays.
* **Next-Gen HDR Export (`FR-HDR-02`):**
  * Exports HDR photos with ISO 21496-1 Gain Maps or native PQ/HLG transfer functions in **AVIF**, **JPEG XL (`.jxl`)**, **HEIC**, **JPEG + Gain Map**, **16-bit PNG**, and **32-bit/16-bit TIFF** across Rec. 2020, Display P3, and sRGB.
* **Multi-Shot Composite Engine (`FR-HDR-03`):**
  * **HDR Merge:** Aligns handheld exposure brackets, applies 3-level optical flow **Deghosting** (`Off`, `Low`, `Medium`, `High`) with deghost mask overlay, and outputs a 32-bit floating-point DNG.
  * **Panorama Merge:** Stitches multi-row panoramas with **Spherical**, **Cylindrical**, and **Perspective** projections, **Boundary Warp (`0–100`)** mesh warping, and **Fill Edges** generative/content-aware synthesis.
  * **HDR Panorama Merge:** Single-step fusion of multi-bracketed, multi-column panoramic shoots into a unified 32-bit DNG.

---

### 3.6 Module 6: Optics, Geometry & Technical Calibration

* **Automated Lens Profile Corrections (`FR-OPT-01`):**
  * Integrates the open-source **Lensfun** database, Adobe LCP parser, and embedded EXIF opcodes (Sony/Fujifilm/Micro Four Thirds/Canon/Nikon) with manual Distortion and Vignetting overrides.
* **Chromatic Aberration & Defringe (`FR-OPT-02`):**
  * Automatic sub-pixel lateral R/B channel warp + axial Purple/Green defringe hue-range sliders and on-canvas Fringe Eyedropper.
* **Upright & Guided Upright (`FR-OPT-03`):**
  * **Upright Modes:** `Off`, `Auto` (balanced perspective), `Level` (horizon only), `Vertical` (converging verticals), `Full` (horizontal + vertical).
  * **Guided Upright:** Interactive loupe tool allowing the user to draw 2 to 4 alignment guides along architectural edges to solve homography matrices exactly.
* **Flat-Field Correction (`FR-OPT-04`):**
  * Pairs target RAW frames with an interleaved or selected white diffuser calibration frame to remove asymmetrical sensor/lens color cast (e.g., technical view cameras with wide-angle tilt/shift lenses) and sensor dust shading.

---

### 3.7 Module 7: Studio Production, Batch Workflows & Output Modules

* **Tethered Capture & Live View (`FR-PROD-01`):**
  * Real-time USB and Wi-Fi/Ethernet PTP tethering via `libgphoto2` (Desktop) or WebUSB (Web).
  * Floating HUD bar with **Live View** video feed, click-to-autofocus, remote Shutter trigger, ISO/Aperture/Shutter Speed/White Balance selectors, session auto-naming, and instant develop preset application.
* **Soft Proofing & ICC Gamut Warnings (`FR-PROD-02`):**
  * Simulates any destination ICC printer/paper profile on screen using **Perceptual** or **Relative Colorimetric** rendering intents.
  * **Simulate Paper & Ink:** Adjusts display white point and black density to match physical paper stock.
  * Dual real-time overlays: **Monitor Gamut Warning** (Red) and **Destination Printer Gamut Warning** (Blue).
* **Reference View & High-Velocity Batch Sync (`FR-PROD-03`):**
  * **Reference View (`Shift+R`):** Split-screen locked Reference Photo + Active Editable Photo.
  * **Auto-Sync Mode:** When enabled, any slider, curve, color wheel, or mask adjustment applies simultaneously to all selected photos in the filmstrip.
  * **Subset Copy/Paste:** Granular modal with **"Modified Settings Only"** quick filter and automatic AI Mask recalculation on paste.
* **Content Credentials — C2PA Provenance (`FR-PROD-04`):**
  * Cryptographically signs exported images using `c2pa-rs`, recording creator identity, camera hardware capture assertions, and a transparent audit log of non-AI vs. Generative AI edits.
* **Video Trimming, Grading & 4K Export (`FR-PROD-05`):**
  * Non-destructive in/out trim points, full application of Optixel tone/color grading parameters to video frames via WebGPU shader pipeline, and H.264/H.265/ProRes/AV1 export up to 4K 60fps.
* **Dedicated Output Modules (`FR-PROD-06`):**
  * **Map Module:** Interactive vector/satellite map (MapLibre GL) displaying geotagged pins, reverse-geocoding search, drag-and-drop photo geotagging, and **GPX Tracklog** import/time-sync.
  * **Book Module:** Multi-page photo book layout engine (powered by an embedded **Typst** / PDF/X-4 renderer) with Blurb and standard print-on-demand trim sizes, auto-layout, spine calculator, and rich typography.
  * **Print Module:** WYSIWYG print engine supporting Single Image/Contact Sheet, Picture Package, and Custom Package layouts, 16-bit printing, custom print sharpening (`Low`/`Standard`/`High` for `Matte`/`Glossy` media), and custom identity plates/watermarks.
  * **Slideshow & Web Modules:** 4K audio-synced video slideshow renderer with pan-and-zoom Ken Burns effects, plus a static responsive HTML5/WebGL portfolio gallery generator.

---

## 4. Model Context Protocol (MCP) Server Specification

Optixel Studio embeds a full **Model Context Protocol (MCP) v1.0 Server** inside the application core so external AI agents (such as Claude Desktop, Claude Code CLI, Antigravity CLI, Cursor, Windsurf, or custom orchestrators) can inspect the photo library, analyze image telemetry, and execute non-destructive edits.

### 4.1 MCP Transports & Security Model
* **`stdio` Transport:** Launched via `optixel-studio --mcp-stdio` (attaches to the running GUI instance over a local Unix domain socket / named pipe, or runs headless against a specified `.olcat` catalog).
* **`Streamable HTTP` & `SSE` Transport:** Exposed locally at `http://127.0.0.1:41920/mcp` (protected by a scoped bearer token generated in Settings).
* **Non-Destructive Safety Guarantee:** Every MCP tool mutation is wrapped in an atomic **AI Transaction Group** on the undo/redo stack and optionally creates a **Virtual Copy** before modifying parameters.

### 4.2 MCP Resources (`optixel://...`)

| Resource URI Template | MIME Type | Description |
|---|---|---|
| `optixel://catalog/summary` | `application/json` | Total photo count, collections, camera/lens breakdown, and current filter state. |
| `optixel://selection/active` | `application/json` | Array of currently selected `photo_id`s in the UI filmstrip with EXIF & culling scores. |
| `optixel://photo/{id}/state` | `application/json` | Complete non-destructive edit graph (tone, curves, Point Color, 3-way wheels, masks, crop). |
| `optixel://photo/{id}/telemetry` | `application/json` | RGB/Luminance histogram bins (256 buckets), clipping percentages, dominant Oklch color palette, and detected face/eye sharpness metrics. |
| `optixel://photo/{id}/preview` | `image/jpeg` | Rendered viewport preview (configurable `max_edge: 1024px` or `2048px`) reflecting current edits, plus optional mask overlay. |

### 4.3 MCP Tool Definitions

Below is the canonical tool registry exposed by the Optixel MCP Server:

```json
{
  "tools": [
    {
      "name": "catalog_query",
      "description": "Search and filter the Optixel catalog using SQL metadata filters, culling scores, and natural-language semantic search.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "semantic_query": { "type": "string", "description": "Natural language visual description (e.g., 'bride smiling outdoors at golden hour')" },
          "min_rating": { "type": "integer", "minimum": 0, "maximum": 5 },
          "flag_status": { "type": "string", "enum": ["pick", "unflagged", "reject", "any"] },
          "min_eye_sharpness": { "type": "number", "minimum": 0, "maximum": 100 },
          "eyes_open_only": { "type": "boolean" },
          "camera_or_lens": { "type": "string" },
          "limit": { "type": "integer", "default": 50 }
        }
      }
    },
    {
      "name": "culling_batch_update",
      "description": "Set star ratings, pick/reject flags, color labels, or create stacks/virtual copies for a list of photos.",
      "inputSchema": {
        "type": "object",
        "required": ["photo_ids"],
        "properties": {
          "photo_ids": { "type": "array", "items": { "type": "string" } },
          "rating": { "type": "integer", "minimum": 0, "maximum": 5 },
          "flag": { "type": "string", "enum": ["pick", "unflagged", "reject"] },
          "color_label": { "type": "string", "enum": ["none", "red", "yellow", "green", "blue", "purple"] },
          "create_virtual_copy": { "type": "boolean", "default": false }
        }
      }
    },
    {
      "name": "develop_apply_parameters",
      "description": "Apply non-destructive global tone, presence, curve, 3-way color grading, Point Color, or calibration adjustments to one or more photos.",
      "inputSchema": {
        "type": "object",
        "required": ["photo_ids", "parameters"],
        "properties": {
          "photo_ids": { "type": "array", "items": { "type": "string" } },
          "mode": { "type": "string", "enum": ["absolute", "relative_delta"], "default": "absolute" },
          "parameters": {
            "type": "object",
            "properties": {
              "profile": { "type": "string" },
              "hdr_mode": { "type": "boolean" },
              "white_balance": {
                "type": "object",
                "properties": { "temperature_k": { "type": "number" }, "tint": { "type": "number" } }
              },
              "tone": {
                "type": "object",
                "properties": {
                  "exposure": { "type": "number", "minimum": -5.0, "maximum": 5.0 },
                  "contrast": { "type": "number", "minimum": -100, "maximum": 100 },
                  "highlights": { "type": "number", "minimum": -100, "maximum": 100 },
                  "shadows": { "type": "number", "minimum": -100, "maximum": 100 },
                  "whites": { "type": "number", "minimum": -100, "maximum": 100 },
                  "blacks": { "type": "number", "minimum": -100, "maximum": 100 }
                }
              },
              "presence": {
                "type": "object",
                "properties": {
                  "texture": { "type": "number", "minimum": -100, "maximum": 100 },
                  "clarity": { "type": "number", "minimum": -100, "maximum": 100 },
                  "dehaze": { "type": "number", "minimum": -100, "maximum": 100 },
                  "vibrance": { "type": "number", "minimum": -100, "maximum": 100 },
                  "saturation": { "type": "number", "minimum": -100, "maximum": 100 }
                }
              },
              "color_grading_3way": {
                "type": "object",
                "properties": {
                  "shadows": { "type": "object", "properties": { "hue": { "type": "number" }, "sat": { "type": "number" }, "lum": { "type": "number" } } },
                  "midtones": { "type": "object", "properties": { "hue": { "type": "number" }, "sat": { "type": "number" }, "lum": { "type": "number" } } },
                  "highlights": { "type": "object", "properties": { "hue": { "type": "number" }, "sat": { "type": "number" }, "lum": { "type": "number" } } },
                  "blending": { "type": "number", "minimum": 0, "maximum": 100 },
                  "balance": { "type": "number", "minimum": -100, "maximum": 100 }
                }
              },
              "point_color_swatches": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "src_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "shift_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "range_hsl": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "falloff": { "type": "number", "minimum": 0, "maximum": 100 }
                  }
                }
              }
            }
          }
        }
      }
    },
    {
      "name": "mask_create_and_edit",
      "description": "Create a boolean AI mask (Subject, Sky, Background, People anatomy, Landscape elements, Range, or Gradient) and apply local develop adjustments.",
      "inputSchema": {
        "type": "object",
        "required": ["photo_id", "mask_name", "operations", "local_adjustments"],
        "properties": {
          "photo_id": { "type": "string" },
          "mask_name": { "type": "string" },
          "operations": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["boolean_op", "source_type"],
              "properties": {
                "boolean_op": { "type": "string", "enum": ["add", "subtract", "intersect"] },
                "source_type": {
                  "type": "string",
                  "enum": [
                    "ai_subject", "ai_sky", "ai_background",
                    "ai_people_face_skin", "ai_people_body_skin", "ai_people_iris_pupil", "ai_people_sclera", "ai_people_lips", "ai_people_teeth", "ai_people_hair", "ai_people_clothes",
                    "ai_landscape_sky", "ai_landscape_mountains", "ai_landscape_architecture", "ai_landscape_vegetation", "ai_landscape_water", "ai_landscape_natural_ground",
                    "linear_gradient", "radial_gradient", "luminance_range", "color_range", "depth_range"
                  ]
                },
                "invert": { "type": "boolean", "default": false },
                "feather": { "type": "number", "minimum": -100, "maximum": 100, "default": 0 },
                "edge_shift": { "type": "number", "minimum": -100, "maximum": 100, "default": 0 }
              }
            }
          },
          "local_adjustments": { "type": "object", "description": "Any tone, presence, curve, or Point Color parameters to apply inside this mask." }
        }
      }
    },
    {
      "name": "ai_reconstruct_and_retouch",
      "description": "Execute AI Denoise, Super Resolution, AI Lens Blur, Reflection Removal, People Removal, or Generative Remove.",
      "inputSchema": {
        "type": "object",
        "required": ["photo_id", "task"],
        "properties": {
          "photo_id": { "type": "string" },
          "task": {
            "type": "string",
            "enum": ["ai_denoise", "super_resolution", "remove_reflections", "remove_background_people", "remove_sensor_dust", "ai_lens_blur", "generative_remove"]
          },
          "amount": { "type": "number", "minimum": 0, "maximum": 100, "default": 50 },
          "lens_blur_config": {
            "type": "object",
            "properties": {
              "bokeh_shape": { "type": "string", "enum": ["circle", "bubble", "blade_5", "ring", "cat_eye"] },
              "bokeh_boost": { "type": "number", "minimum": 0, "maximum": 100 },
              "focal_distance": { "type": "number", "minimum": 0, "maximum": 100 }
            }
          },
          "generative_prompt": { "type": "string" },
          "provider_override": { "type": "string", "description": "Optional provider ID from the AI Provider Router (e.g., 'ollama:qwen2.5-vl', 'cli:claude', 'cli:antigravity')" }
        }
      }
    },
    {
      "name": "batch_export",
      "description": "Export photos with specified format (JPEG, AVIF, JXL, TIFF, DNG), color space, sizing, sharpening, and C2PA Content Credentials.",
      "inputSchema": {
        "type": "object",
        "required": ["photo_ids", "destination_folder", "format"],
        "properties": {
          "photo_ids": { "type": "array", "items": { "type": "string" } },
          "destination_folder": { "type": "string" },
          "format": { "type": "string", "enum": ["JPEG", "AVIF", "JXL", "TIFF_16", "PNG_16", "DNG"] },
          "color_space": { "type": "string", "enum": ["sRGB", "DisplayP3", "AdobeRGB", "Rec2020_HDR"] },
          "attach_c2pa_credentials": { "type": "boolean", "default": true }
        }
      }
    }
  ]
}
```

---

## 5. Pluggable AI & CLI Agent Provider Router Specification

Optixel Studio decouples its AI features from any single vendor. In **Settings > AI & Agent Router**, the user can configure and assign different backends per task category (**Auto-Tone & Color Tuning**, **Natural-Language Prompt-to-Edit**, **Semantic Culling/Search**, and **Generative Inpainting**).

### 5.1 Supported Provider Types

| Provider Category | Identifier Prefix | Execution Mechanism & Supported Tools |
|---|---|---|
| **Claude CLI** | `cli:claude` | Spawns local `claude` CLI subprocess (`claude -p "<prompt>" --output-format json --allowedTools "mcp__optixel__*"`). Automatically passes the temporary rendered viewport image, histogram JSON, and active MCP server configuration so Claude CLI can iteratively inspect and tune the image. |
| **Antigravity CLI** | `cli:antigravity` | Spawns the **Antigravity CLI** (`antigravity` / `agy`) agent runner in headless/non-interactive mode with Optixel's local MCP server attached, allowing the Antigravity agent to execute multi-step photo grading, culling, and masking workflows. |
| **Custom CLI Agent** | `cli:custom` | User-defined command template (e.g., `codex`, `aider`, `goose`, or custom Python/Rust script) using template variables: `{{IMAGE_PATH}}`, `{{HISTOGRAM_JSON}}`, `{{CURRENT_XMP_JSON}}`, `{{USER_PROMPT}}`, and `{{MCP_SSE_URL}}`. |
| **Ollama Local API** | `ollama:<model>` | Connects to local or LAN Ollama server (`http://localhost:11434/api/chat`) using multimodal Vision-Language Models (`qwen2.5-vl`, `llama3.2-vision`, `gemma3:27b`, `llava`) with strict JSON Schema structured output (`format: schema`). |
| **OpenAI-Compatible Local API** | `local-api:<name>` | Connects to **LM Studio** (`http://localhost:1234/v1`), **vLLM**, **llama.cpp server**, or **LocalAI** via `/v1/chat/completions` with base64 image payloads and structured JSON tool calls. |
| **Cloud Commercial APIs** | `api:anthropic`<br>`api:gemini`<br>`api:openai`<br>`api:openrouter` | Direct HTTPS REST calls to Anthropic (`claude-sonnet-4-20250514` / `claude-3-7-sonnet`), Google Gemini (`gemini-2.5-pro` / `gemini-2.5-flash`), OpenAI (`gpt-4o`), or OpenRouter using user-supplied API keys stored in the OS Keychain. |
| **Local Pixel / Diffusion API** | `pixel:comfyui`<br>`pixel:onnx-local` | Connects to a local **ComfyUI** WebSocket API (`http://127.0.0.1:8188`) or built-in **ONNX Runtime** models for pixel-level Generative Remove, Outpainting, Reflection Removal, and Upscaling. |

### 5.2 Core Rust Trait & Router Architecture

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderBackend {
    ClaudeCli {
        binary_path: String,          // default: "claude"
        model_flag: Option<String>,   // e.g., "sonnet" or "opus"
        max_turns: u8,                // e.g., 3 iterative visual feedback turns
    },
    AntigravityCli {
        binary_path: String,          // default: "antigravity" or "agy"
        agent_profile: Option<String>,
        mcp_transport: String,        // "sse" or "stdio"
    },
    CustomCli {
        command_template: String,     // e.g., "my-agent --image {{IMAGE_PATH}} --prompt {{USER_PROMPT}}"
        timeout_ms: u64,
    },
    OllamaApi {
        endpoint: String,             // default: "http://127.0.0.1:11434"
        model: String,                // e.g., "qwen2.5-vl:7b" or "gemma3:27b"
        keep_alive: String,           // e.g., "5m"
    },
    OpenAiCompatibleApi {
        name: String,                 // e.g., "LM Studio", "vLLM", "OpenRouter", "OpenAI"
        base_url: String,             // e.g., "http://127.0.0.1:1234/v1"
        api_key_ref: Option<String>,  // OS Keychain reference
        model: String,
    },
    AnthropicApi {
        api_key_ref: String,
        model: String,
    },
    GeminiApi {
        api_key_ref: String,
        model: String,
    },
    ComfyUiWorkflow {
        endpoint: String,             // default: "http://127.0.0.1:8188"
        workflow_json_template: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AiTuningContext {
    pub photo_id: String,
    pub user_instruction: String,
    pub rendered_preview_jpeg_path: String,
    pub rendered_preview_base64: String,
    pub histogram_telemetry: HistogramTelemetry,
    pub current_develop_state: DevelopStateJson,
    pub reference_image_base64: Option<String>, // Used when matching a Reference View photo
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AiTuningResponse {
    pub reasoning_summary: String,
    pub global_parameter_patch: Option<DevelopParameterPatch>,
    pub mask_operations: Vec<MaskOperationCommand>,
    pub requires_another_visual_pass: bool,
}

#[async_trait]
pub trait AiProviderAdapter: Send + Sync {
    async fn evaluate_and_tune(
        &self,
        ctx: &AiTuningContext,
    ) -> Result<AiTuningResponse, ProviderError>;
}
```

### 5.3 Concrete Execution Flows for CLI & Ollama Providers

#### Flow A: Claude CLI (`cli:claude`) & Antigravity CLI (`cli:antigravity`) Closed-Loop Tuning
When the user selects **Claude CLI** or **Antigravity CLI** in the Develop AI Assistant bar and types:  
*"Match the warm Kodak Portra skin tones of the reference image, recover the blown clouds in the sky using an intersected sky mask, and add subtle halation in the highlights"*

1. **Snapshot & Telemetry Export:** Optixel renders a 1536px JPEG of the current canvas (`/tmp/optixel_preview_881.jpg`), exports the Reference Photo preview (`/tmp/optixel_ref_881.jpg`), and dumps the 256-bin RGB histogram + clipping metrics + current edit state JSON.
2. **CLI Invocation:**
   - **For Claude CLI:**
     ```bash
     claude -p "You are controlling Optixel Studio via MCP. Inspect /tmp/optixel_preview_881.jpg and reference /tmp/optixel_ref_881.jpg. Current telemetry: $(cat /tmp/optixel_telemetry_881.json). User goal: 'Match the warm Kodak Portra skin tones...'. Use mcp__optixel__develop_apply_parameters and mcp__optixel__mask_create_and_edit, then inspect the updated preview to verify zero highlight clipping." \
       --mcp-config /tmp/optixel_mcp_config.json \
       --allowedTools "mcp__optixel__*" \
       --output-format json
     ```
   - **For Antigravity CLI:**
     ```bash
     antigravity run \
       --mcp-server "http://127.0.0.1:41920/mcp" \
       --context-file "/tmp/optixel_preview_881.jpg" \
       --context-file "/tmp/optixel_telemetry_881.json" \
       --prompt "Execute non-destructive color grading and AI sky masking on photo_id=881 to match the user prompt..." \
       --format json
     ```
3. **Real-Time Viewport Feedback:** As the CLI agent calls `develop_apply_parameters` and `mask_create_and_edit` over the local MCP socket, the user sees the WebGPU canvas, tone curves, 3-Way Color Wheels, and Mask Panel update live at 60fps.
4. **Visual Verification Pass:** The CLI agent reads `optixel://photo/881/preview` and `optixel://photo/881/telemetry` after its first edit pass, verifies that highlight clipping dropped from `4.2%` to `0.0%` and skin-tone Oklch hue angle sits at `42°`, and completes the transaction.

#### Flow B: Ollama Local API (`ollama:<model>`) Direct Structured Tuning
When running offline with **Ollama** (e.g., `qwen2.5-vl:7b` or `gemma3:27b`):
1. Optixel sends a `POST http://127.0.0.1:11434/api/chat` request containing:
   - `images`: `[rendered_preview_base64]`
   - `messages`: System prompt containing the exact parameter ranges + current histogram telemetry JSON + user prompt.
   - `format`: The strict JSON Schema for `AiTuningResponse`.
2. Ollama returns a deterministic JSON payload containing `global_parameter_patch` and `mask_operations`.
3. Optixel's Rust engine validates parameter bounds, applies the patch to the WebGPU DAG pipeline, re-renders the histogram, and optionally runs a second self-correction pass if clipping exceeds thresholds.

---

## 6. Non-Destructive Edit Graph & XMP Data Schema

Every photo's edit state is stored both in SQLite (`photos.develop_json`) and in sidecar `.xmp` / `.olxmp` JSON-LD files so edits are portable, human-readable, and version-controllable via Git.

```json
{
  "schema_version": "1.0.0",
  "photo_id": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "pipeline": {
    "working_space": "Rec2020_Linear_32f",
    "hdr_display_mode": true,
    "hdr_limit_ev": 3.5,
    "raw_profile": "Camera Matching - Classic Chrome",
    "ai_denoise": { "enabled": true, "amount": 42, "model": "ols-bayer-denoise-v2.onnx" },
    "white_balance": { "mode": "Custom", "temperature_k": 5450, "tint": 8 },
    "basic_tone": {
      "exposure": 0.35,
      "contrast": 12,
      "highlights": -64,
      "shadows": 38,
      "whites": 15,
      "blacks": -18
    },
    "presence": {
      "texture": 15,
      "clarity": 8,
      "dehaze": 10,
      "vibrance": 18,
      "saturation": -4
    },
    "tone_curve": {
      "refine_saturation": 85,
      "parametric": { "shadows": -8, "darks": -5, "lights": 10, "highlights": 4, "splits": [25, 50, 75] },
      "point_luma": [[0, 6], [64, 58], [190, 202], [255, 250]],
      "point_red": [[0, 0], [128, 132], [255, 255]],
      "point_green": [[0, 0], [128, 128], [255, 255]],
      "point_blue": [[0, 4], [128, 124], [255, 248]]
    },
    "point_color": [
      {
        "id": "swatch_skin_1",
        "src_hsl": [24.5, 48.0, 62.0],
        "shift_hsl": [-4.0, -8.0, 5.0],
        "range_hsl": [35.0, 50.0, 60.0],
        "falloff": 65.0
      }
    ],
    "color_grading_3way": {
      "shadows": { "hue": 205, "sat": 14, "lum": -6 },
      "midtones": { "hue": 34, "sat": 10, "lum": 2 },
      "highlights": { "hue": 48, "sat": 18, "lum": 0 },
      "blending": 62,
      "balance": 12
    },
    "ai_lens_blur": {
      "enabled": true,
      "amount": 35,
      "bokeh_shape": "cat_eye",
      "bokeh_boost": 45,
      "focal_range": [18, 42]
    },
    "masks": [
      {
        "mask_id": "mask_01",
        "name": "Graduated Sky Recovery",
        "enabled": true,
        "components": [
          { "op": "add", "type": "ai_sky", "feather": 12, "edge_shift": -4 },
          { "op": "intersect", "type": "linear_gradient", "start": [0.5, 0.0], "end": [0.5, 0.55] }
        ],
        "adjustments": {
          "exposure": -0.65,
          "highlights": -45,
          "dehaze": 22,
          "temperature_shift": -12
        }
      }
    ],
    "ai_provenance_log": [
      {
        "timestamp": "2026-10-27T15:30:00Z",
        "provider": "cli:antigravity",
        "action": "Applied Graduated Sky Recovery mask and 3-way color grade",
        "generative_pixels_modified": false
      }
    ]
  }
}
```

---

## 7. Performance, Security & Cross-Platform SLAs

| Metric / Requirement | Target Specification |
|---|---|
| **Interactive Slider Frame Rate** | **60 FPS (`<16.6ms` frame time)** at 2560×1440 viewport resolution on 60MP RAW files using tiled MIP-mapped WebGPU compute shaders. |
| **Catalog Scale** | Smooth scrolling (120 FPS virtualized grid) and `<50ms` filter query response on catalogs containing **500,000+ photos**. |
| **RAW Import & Culling Throughput** | Parallel decoding + Smart Preview generation + Eye Sharpness AI scoring at **8–15 RAW files/second** on Apple M-series or modern discrete GPUs. |
| **Web App Feature Parity** | 95%+ feature parity in browser via WebGPU + WASM + OPFS; when paired with the local `optixel-daemon` (or Tauri desktop shell), 100% parity including native USB tethering (`libgphoto2`) and local CLI execution (`claude`, `antigravity`). |
| **CLI & MCP Sandboxing** | Subprocess CLI runners (`claude`, `antigravity`, custom scripts) execute with scoped environment variables, restricted working directories (`/tmp/optixel-session-*`), and configurable read/write catalog permissions. |