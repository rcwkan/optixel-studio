# Third-Party Notices and Open Source Attributions

Optixel Studio incorporates, links against, or interacts with various third-party open-source libraries, native C/C++ engines, and neural model architectures. We gratefully acknowledge the creators and contributors of these foundational technologies.

Below is an itemized attribution and license summary for each major dependency and third-party component.

---

## 1. Native C/C++ Libraries & Media Engines

### LibRaw (Camera RAW Decoding & Demosaicing)
- **Copyright**: (c) 2008-2025 LibRaw LLC (http://www.libraw.org, info@libraw.org)
- **License**: Common Development and Distribution License (CDDL) Version 1.0 (elected) or GNU Lesser General Public License (LGPL) Version 2.1.
- **Notice**: Optixel Studio distributes and links LibRaw under the terms of the **CDDL Version 1.0**. The source code for LibRaw and any modifications to its files are maintained at `src-tauri/vendor/libraw` and are available in accordance with the CDDL. LibRaw also incorporates code from:
  - `dcraw.c` (Dave Coffin, copyright 1997-2018)
  - DCB demosaic and FBDD denoise (Jacek Gozdz, BSD-3-Clause)
  - X3F library (Roland Karlsson, BSD-style)
  - Adobe DNG SDK 1.4 (Adobe Systems Inc., MIT license)

### LittleCMS 2 (Color Management System)
- **Copyright**: (c) 1998-2024 Marti Maria Saguer
- **License**: MIT License
- **Notice**: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

### ONNX Runtime C/C++ Engine
- **Copyright**: (c) Microsoft Corporation. All rights reserved.
- **License**: MIT License
- **Notice**: Hardware-accelerated inference execution provider (Apple CoreML on macOS, DirectML on Windows, and CPU fallback).

### libjpeg-turbo & turbojpeg
- **Copyright**: (c) 1991-2024 The libjpeg-turbo Project and Independent JPEG Group.
- **License**: IJG License / Modified BSD License / zlib License.
- **Notice**: This software is based in part on the work of the Independent JPEG Group.

---

## 2. Rust Core Libraries

The following Rust crates are compiled into Optixel Studio:

| Crate | Primary Purpose | License | Copyright / Maintainer |
| :--- | :--- | :--- | :--- |
| **tauri / tauri-build** | Desktop application shell & IPC | MIT OR Apache-2.0 | Tauri Programme within The Commons Conservancy |
| **tauri-plugin-updater** | Cryptographic in-app updates | MIT OR Apache-2.0 | Tauri Contributors |
| **tauri-plugin-dialog** | Native OS file picker dialogs | MIT OR Apache-2.0 | Tauri Contributors |
| **wgpu / naga** | WebGPU compute & shader pipeline | MIT OR Apache-2.0 | wgpu graphical developers |
| **rawler** | Pure-Rust RAW fallback decoder | LGPL-2.1 | (c) 2018-2024 dnglab contributors |
| **rusqlite** | SQLite database wrapper | MIT | (c) 2014-2024 The Rusqlite Developers |
| **image** | Pixel buffer manipulation & codecs | MIT OR Apache-2.0 | The PistonDevelopers |
| **rayon** | Multi-threaded parallel processing | MIT OR Apache-2.0 | Niko Matsakis and rayon contributors |
| **kamadak-exif** | EXIF metadata parsing | BSD-2-Clause | (c) 2015-2022 Kamyar Kanani |
| **jxl-oxide** | JPEG XL image decoding | MIT OR Apache-2.0 | (c) 2023-2024 jxl-oxide contributors |
| **moxcms** | ICC color profile parsing | BSD-3-Clause OR Apache-2.0 | (c) 2024 moxcms developers |
| **ring** | Cryptographic hashing & SHA-256 | ISC / OpenSSL-like | Brian Smith and Ring contributors |
| **tiny_http** | Embedded local MCP server daemon | MIT | tiny-http developers |
| **dirs** | Cross-platform system paths | MIT OR Apache-2.0 | Simon Ochsenreither |
| **serde / serde_json** | Data serialization | MIT OR Apache-2.0 | Erick Tryzelaar and David Tolnay |

> **Notice Regarding `rawler` (LGPL-2.1)**:  
> In compliance with the GNU Lesser General Public License Version 2.1, Section 6, the complete source code of Optixel Studio and all build scripts necessary to re-compile and re-link the application against an alternative or modified version of `rawler` are freely available at:  
> https://github.com/rcwkan/optixel-studio

---

## 3. Deep Learning & AI Model Architectures

Optixel Studio provides client-side neural processing for photographic retouching. Deep learning model checkpoints are downloaded dynamically by the user upon request and are governed by their respective upstream licenses:

1. **LaMa (Large Mask Inpainting)**
   - *License*: Apache License 2.0
   - *Attribution*: Samsung AI Center & Roman Suvorov et al. ("Resolution-robust Large Mask Inpainting with Fourier Convolutions").

2. **NAFNet (Nonlinear Activation-Free Network)**
   - *License*: MIT License
   - *Attribution*: Megvii Research ("Simple Baselines for Image Restoration").

3. **SCUNet (Swin-Conv UNet)**
   - *License*: Apache License 2.0
   - *Attribution*: Kai Zhang et al. ("Practical Blind Denoising via Swin-Conv UNet").

4. **PerspectiveFields (Camera Calibration & Geometry)**
   - *License*: MIT License
   - *Attribution*: Sony AI & Linyi Jin et al. ("Perspective Fields for Single Image Camera Calibration").

5. **Moebius (Segment Anything / Masking)**
   - *License*: Apache License 2.0
   - *Attribution*: Meta AI Research / open-source SAM community.

6. **Stable Diffusion 1.5 Inpainting**
   - *License*: CreativeML OpenRAIL-M License
   - *Attribution*: RunwayML & CompVis.
   - *Ethical Use Requirements*: Use of Stable Diffusion weights is subject to the OpenRAIL-M Use Restrictions (prohibiting defamatory, harassing, or unlawful generation).

---

## 4. Frontend UI Stack

The graphical interface is built using standard web technologies:

- **React 19 & React-DOM** — (c) Meta Platforms, Inc. — MIT License
- **Tailwind CSS v4** — (c) Tailwind Labs, Inc. — MIT License
- **Vite** — (c) Evan You & Vite Contributors — MIT License
- **TypeScript** — (c) Microsoft Corporation — Apache License 2.0

---

## 5. Summary License Texts

### MIT License
```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Common Development and Distribution License (CDDL-1.0)
Full license text available at: https://opensource.org/licenses/CDDL-1.0

### GNU Lesser General Public License Version 2.1 (LGPL-2.1)
Full license text available at: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html

### Apache License 2.0
Full license text available at: https://www.apache.org/licenses/LICENSE-2.0

