# Chapter 5: RAW Processing Pipeline, Color Science & Mathematics

This chapter explores the mathematical foundations, color spaces, and execution stages of Optixel Studio's non-destructive image rendering pipeline implemented in [`src-tauri/src/develop.rs`](../src-tauri/src/develop.rs).

---

## 1. Color Spaces & Working Architecture

Professional RAW processing demands a wide-gamut, high-dynamic-range color space to prevent hue shifts and gamut clipping during extreme tonal adjustments:

```
+--------------------------------------------------------------------------------------------------+
| RAW Sensor Data (Bayer / X-Trans CFA)                                                            |
+--------------------------------------------------------------------------------------------------+
                                                 | Camera Matrix Transform
+------------------------------------------------v-------------------------------------------------+
| WORKING COLOR SPACE: Linear Rec. 2020 (32-bit Float)                                             |
| - Primary Gamut: Rec. 2020 (encompasses virtually all real-world surface colors)                 |
| - Transfer Function: Linear (physically linear scene light values, 1.0 = 18% gray reference)    |
| - Precision: IEEE 754 32-bit single-precision floating point ([f32; 3] per pixel)              |
+--------------------------------------------------------------------------------------------------+
          |                                                                |
          v (Perceptual Color Ops)                                         v (Final Display/Export)
+------------------------------------+                           +---------------------------------+
| PERCEPTUAL SPACE: Oklab / Oklch    |                           | OUTPUT ENCODING: moxcms ICC     |
| - Point Color 3D Ellipsoid Gamut   |                           | - sRGB (Web / Standard display) |
| - Wavelet Denoise in Oklab L*a*b*  |                           | - Display P3 (Wide Apple / OLED)|
| - Vibrance Skin-Tone Preservation  |                           | - Adobe RGB (Fine art printing) |
| - Color Mixer 8-Band Hue Shifts    |                           | - Rec. 2020 PQ HDR (10/12-bit)  |
+------------------------------------+                           +---------------------------------+
```

---

## 2. The 9-Stage Non-Destructive Render Pipeline

The render pipeline in `develop::render` transforms an input linear image `src: &Img` into an output image through 9 sequential, deterministic stages:

```mermaid
flowchart TD
    Stage0["Input: Linear Rec.2020 Float Proxy (1600px) or Full RAW"]
    Stage1["1. White Balance & Linear Exposure Gain"]
    Stage1a["1a. Lens Corrections, Camera Calibration, Retouch Heal"]
    Stage2["2. Atmospheric Dehaze"]
    Stage3["3. Global Tone (Highlights, Shadows, Whites, Blacks, Contrast)"]
    Stage4["4. Presence: Texture & Clarity (Local Contrast Pyramid)"]
    Stage5["5. Boolean Mask Coverage Calculation & Local Adjustments"]
    Stage6["6. Curves, 8-Band Mixer, Point Color 3D HSL, 3-Way Grading"]
    Stage7["7. Detail: Luminance NR, Color NR, Edge Sharpening"]
    Stage7a["7a. Photoshop Pro Adjustments (HDR Toning, Levels, Selective Color, Filters)"]
    Stage8["8. Mask Tuning (Recursive Secondary Develop on Mask Coverage)"]
    Stage9["9. Geometry (Distortion, Upright, Straighten, Crop) & Post-Crop Effects"]

    Stage0 --> Stage1 --> Stage1a --> Stage2 --> Stage3 --> Stage4
    Stage4 --> Stage5 --> Stage6 --> Stage7 --> Stage7a --> Stage8 --> Stage9
```

---

### Stage 1: White Balance & Linear Exposure Gain
* **White Balance**: Computes RGB channel gains from Correlated Color Temperature ($T \in [2000K, 50000K]$) and Tint ($\Delta \in [-150, 150]$) using Planckian locus approximations:
  $$\mathbf{g}_{WB} = \text{wb\_gains}(T, \Delta)$$
* **Exposure**: Applies physically linear exposure scaling:
  $$g_{\text{exp}} = 2^{\text{exposure}}$$
  $$\mathbf{P}_1 = \mathbf{P}_0 \odot (\mathbf{g}_{WB} \cdot g_{\text{exp}})$$

### Stage 1a: Lens Corrections, Calibration & Object Removal
* **Lens Distortion**: Corrects radial barrel/pincushion distortion:
  $$r' = r \cdot (1 + k_1 r^2 + k_2 r^4)$$
* **Lateral Chromatic Aberration**: Warps red and blue color channels radially relative to green.
* **Camera Calibration**: Modulates primary sensor hue and saturation angles using white-preserving $3 \times 3$ matrices.
* **Retouch Heal**: Refills marked polygon areas from an automatically selected source offset, matching boundary gradient and color luminance.

### Stage 2: Atmospheric Dehaze
* Estimates transmission maps via dark-channel prior approximations. Inverts atmospheric scattering equation:
  $$\mathbf{J}(x) = \frac{\mathbf{I}(x) - \mathbf{A}}{\max(t(x), t_0)} + \mathbf{A}$$

### Stage 3: Global Tone & Contrast
* Calculates toe and shoulder compression curves. Adjusts **Highlights** (compressing specular highlights without hue shift), **Shadows** (lifting shadow detail), **Whites** (setting upper clipping limit), and **Blacks** (anchoring black point).

### Stage 4: Presence (Texture & Clarity)
* Decomposes luminance into high- and mid-frequency spatial layers using multi-scale difference-of-Gaussians / Laplacian filtering.
* **Texture**: Boosts fine spatial frequencies (2px–8px) while protecting pores and noise floors.
* **Clarity**: Amplifies mid-frequency contrast without producing halo artifacts along high-contrast boundaries.

### Stage 5 & Stage 8: Boolean Masks & Recursive Mask Tuning
* Evaluates boolean operations (`Add`, `Subtract`, `Intersect`, `Invert`) across active mask components (Polygons, Linear Gradients, Radial Gradients, Luminance Ranges, and Color Ranges).
* Computes continuous weight maps $\alpha(x, y) \in [0.0, 1.0]$ with morphological **Feather** and **Edge Shift**.
* **Stage 8 Mask Tuning**: Evaluates a secondary `Develop` pass inside the mask coverage weight, alpha-blending with base pixels:
  $$\mathbf{P}_{\text{final}} = (1 - \alpha) \mathbf{P}_{\text{base}} + \alpha \mathbf{P}_{\text{tuned}}$$

### Stage 6: Curves, Point Color & Color Grading
* **Tone Curve**: Parametric 4-zone splines and RGB cubic splines.
* **8-Band Color Mixer**: Partitions the color spectrum into Red, Orange, Yellow, Green, Aqua, Blue, Purple, and Magenta bands, interpolating hue, saturation, and luminance shifts.
* **Point Color 3D HSL**: Converts pixels to Oklch. Checks if the pixel falls inside an ellipsoid centered at $\mathbf{c}_{\text{src}}$:
  $$D = \left(\frac{\Delta H}{R_H}\right)^2 + \left(\frac{\Delta C}{R_C}\right)^2 + \left(\frac{\Delta L}{R_L}\right)^2$$
  If $D \le 1.0$, applies smooth Hermite falloff and shifts the color vector toward $\mathbf{c}_{\text{target}}$.
* **3-Way Color Grading**: Applies shadow, midtone, and highlight color balance with adjustable blending and crossover balance.

### Stage 7 & 7a: Detail & Photoshop Pro Adjustments
* **Noise Reduction**: Bilateral edge-preserving filtering on luminance, and median filtering on chrominance channels.
* **Sharpening**: Unsharp masking with user-specified radius and edge masking.
* **HDR Toning**: *Merge exposures* (the default) simulates an exposure bracket from the photo (2–9 shots, 0.5–4 EV apart, each clipped like a camera's) and merges it by exposure fusion (Mertens, Kautz & Van Reeth): per-pixel weights for well-exposedness, local contrast and saturation, blended in a Laplacian pyramid so no seams show. Shots beyond the photo's own range are dropped, and *Natural* blends the coarse pyramid levels toward the photo's own tones (extended Reinhard), since fusion alone favours mid grey: it would grey a bright sky and glow along strong edges. HDR Merge of real brackets starts its result with the same method, fitted to the merged range. Photoshop's four methods are there too, on log luminance. *Local adaptation*: a self-guided filter splits an edge-aware base (Radius, Edge strength, optional second Smooth-edges pass against halos) from the detail; the base range is compressed around the photo's key and shifted (Shadows, Highlights) and the detail scaled back in (Detail). *Equalize histogram* maps each brightness by its rank; *Exposure and gamma* is global; *Highlight compression* is Reinhard's roll-off to white. Then Exposure, Gamma (around white), Fine detail (1–2 px texture), the Toning curve on display brightness (drawn over the live histogram) and Vibrance / Saturation; every step changes luminance only, so hues stay.
* **Photoshop Adjustments**: Applies Levels (black/gamma/white input and output remapping), Hue/Saturation/Lightness, Selective Color CMYK offsets, and Channel Mixer.

### Stage 9: Geometry, Post-Crop Vignette & Film Grain
* Resolves Upright perspective homography matrices and straightens canvas by rotation angle $\theta$.
* Crops normalized coordinates $(x, y, w, h)$.
* Adds post-crop cosine vignette and procedural noise film grain scaled to output resolution.

---

## 3. Proxy Rendering vs. Full-Resolution Export

To maintain 60fps interactivity during slider adjustments on commodity CPUs, Optixel Studio separates the interactive viewport from export rendering:

| Attribute | Viewport Develop Proxy | Batch Export Renderer |
|---|---|---|
| **Resolution** | 1600px on long edge (`PROXY_EDGE`) | Full camera resolution (up to 100+ Megapixels) |
| **Pipeline Implementation** | [`develop::render`](../src-tauri/src/develop.rs#L1141) | [`develop::render`](../src-tauri/src/develop.rs#L1141) |
| **Execution Target** | Multithreaded CPU via `rayon` | Multithreaded CPU via `rayon` |
| **Latency** | < 16ms (instantaneous UI feedback) | ~200ms – 1.5s per photo |
| **Output Format** | Viewport RGBA buffer / JPEG telemetry | JPEG, AVIF, 16-bit PNG, 16-bit TIFF |
| **Color Profile** | System sRGB / Display P3 | Embedded ICC profile (sRGB, P3, Adobe RGB, Rec.2020) |

Because both modes execute the **exact same Rust code path**, developers and AI agents are guaranteed that what is previewed on screen is identical to the exported master image.


---

## 4. Photo-Type Auto Tuning

**✦ By photo type** (Develop › Auto) and the *By photo type* look in Auto › Non-destructive Editing first tell what a photo shows, then apply that type's profile on top of the measured white balance and tone (`develop::auto` / `look_on`). `auto::photo_type_of` asks the active AI provider for keywords (`ai::describe`), adds the photo's catalog keywords and two measured cues (no colour → B&W, a dark median → night), and `auto::photo_type` maps them to a profile; without an AI the keywords and cues decide alone.

The profiles follow the looks that are common on Unsplash for each kind of photo:

| Type | White balance | Tone & curve | Colour (HSL, grading) | Presence & effects |
|---|---|---|---|---|
| Landscape | as measured | contrast +, highlights −25, shadows + | deeper blues, greens toward yellow and calmer, aqua toward blue, warm highlights | clarity, dehaze, vibrance + |
| Portrait | warmer (+200 K) | soft contrast, faded black (curve lift) | calmer red/orange, brighter orange (skin), muted greens | texture and clarity −, light vibrance |
| Street | as measured | punchy midtones, matte black | muted (saturation −15), teal shadows / warm highlights | clarity +, vignette, light grain |
| Food | warmer (+150 K) | bright, open shadows | richer reds, oranges, greens; cool blues calmed | texture +, vibrance + |
| Architecture | cooler (−150 K) | contrast +, even (highlights −, shadows +, whites +) | muted yellows/greens/blues, lighter (pastel) sky | clarity and texture + |
| Night | cooler (−400 K) | neon held back (highlights −30), deep blacks | greens to aqua, yellows to orange; teal shadows / warm highlights | clarity, vibrance + |
| Nature & macro | warmer (+100 K) | gentle | lush yellows and greens | vibrance +30, texture +, soft vignette |
| B&W | — | contrast + | Monochrome | clarity + |

Unmatched photos get *Natural*. The keyword lists live in `auto::TYPES`; `photo_types_map_to_profiles` checks the mapping.

---

## 5. Native C/C++ Acceleration & Color Science Extensions

To achieve studio-grade parity with professional RAW developers, Optixel Studio links native C/C++ engines alongside the Rust pipeline:

### A. LibRaw C++ & Multi-Algorithm Demosaicing
* **Proprietary Sensor Arrays**: Native parsing for Canon `.CR3`, Nikon High-Efficiency `.NEF`, Sony lossless `.ARW`, and Fujifilm X-Trans non-Bayer color filter arrays (`src-tauri/csrc/ol_raw.cpp`).
* **Multi-Algorithm Demosaic Selector**:
  * **RCD**: Ratio Corrected Demosaicing for maximum structural edge acuity.
  * **AMaZE**: Directional interpolation suppressing zipper and maze artifacts in low-ISO landscapes.
  * **LMMSE**: High-ISO noise-adaptive interpolation preserving micro-textures in astrophotography and low-light portraits.
* **Optical Black-Level & Multi-Channel Highlight Recovery**: Directly reads optical calibration black points from masked sensor margins and executes highlight reconstruction across saturated channels prior to tonemapping.

### B. LittleCMS 2 Studio Color Management & Soft-Proofing
* **Working Color Spaces**: 32-bit floating-point **ProPhoto RGB** and **ACEScg** color space transforms implemented via static LittleCMS 2 (`src-tauri/src/icc.rs`).
* **Interactive Printer Soft-Proofing**: Simulates physical paper/ink black-point compensation using custom `.icc` profiles, highlighting unprintable out-of-gamut pixels with high-visibility neon overlays.
* **JPEG XL XYB Fidelity**: Pairs `jxl-oxide` with LittleCMS 2 (`jxlcms`) to support perceptual XYB color decoding and HDR PQ/HLG transfer curves.

### C. Ahead-of-Time (AOT) GPU Compute Shader Pipeline
* **Build-Time SPIR-V Bytecode**: Naga validates and compiles `src-tauri/shaders/*.wgsl` into stripped binary SPIR-V bytecode embedded directly into the Rust executable.
* **Real-Time GPU Compute Kernels**: Dispatches high-throughput GPU kernels for Local Clarity/Texture/Dehaze, 3D LUT trilinear interpolation, lateral and axial chromatic aberration defringing, and 60 FPS RGB Parade and Waveform scopes (`src-tauri/src/gpu.rs`).

### D. Lossless SIMD DCT Transforms & Adobe XMP Sync
* **libjpeg-turbo**: Lossless 90°/180°/270° rotation, flipping, and cropping executed directly in DCT frequency space without recompressing or degrading JPEG image quality.
* **MakerNotes AF Point Overlay (<kbd>A</kbd>)**: Decodes vendor-specific MakerNotes to overlay active autofocus points and eye-AF bounding boxes during culling.
* **Bidirectional Lightroom `.XMP` Sync**: Parses and writes Adobe XMP sidecars to synchronize ratings, color labels, crop rectangles, and IPTC copyright tags with Lightroom Classic (`src-tauri/src/xmp.rs`).
