# Chapter 6: Model Context Protocol (MCP) Server & Pluggable AI Router

This chapter is the technical guide to Optixel Studio's built-in **Model Context Protocol (MCP)** server and its **Pluggable AI & CLI Agent Router** implemented in [`src-tauri/src/mcp.rs`](../src-tauri/src/mcp.rs) and [`src-tauri/src/ai.rs`](../src-tauri/src/ai.rs).

---

## 1. MCP Server Architecture & Transports

Optixel Studio implements the **Model Context Protocol (MCP)** v1.0 specification, exposing every catalog query, culling rating, non-destructive develop parameter, boolean mask, and export pipeline to external AI agents.

```
+--------------------------------------------------------------------------------------------------+
| EXTERNAL AI AGENTS (Google Antigravity, Claude Code CLI, Cursor, Windsurf, Custom Orchestrators) |
+--------------------------------------------------------------------------------------------------+
                               |                                 |
              [Streamable HTTP / Bearer Auth]           [stdio JSON-RPC 2.0]
              http://127.0.0.1:41920/mcp                optixel-studio --mcp-stdio
                               |                                 |
+------------------------------v---------------------------------v---------------------------------+
|                                 OPTIXEL MCP SERVER (mcp.rs)                                      |
|   - JSON-RPC 2.0 Dispatcher (initialize, tools/list, tools/call, resources/list, resources/read)|
|   - Authentication & Scoped Bearer Token Validation                                              |
|   - Atomic Undo/Redo Step Wrapping & AI Provenance Log Recording                                |
+--------------------------------------------------------------------------------------------------+
                               |
+------------------------------v-------------------------------------------------------------------+
| OPTIXEL CORE ENGINE (Catalog Database + 32-bit Float Develop DAG Pipeline)                      |
+--------------------------------------------------------------------------------------------------+
```

### 1.1 Transport Options
1. **Streamable HTTP Transport (Loopback)**:
   * **Endpoint**: `http://127.0.0.1:41920/mcp`
   * **Authentication**: Bearer token generated on first launch and displayed in **Settings › AI & MCP**.
   * Suitable for long-lived desktop sessions, Claude Desktop integration, and Antigravity local agent connections.
2. **stdio Transport**:
   * Invoked via `optixel-studio --mcp-stdio` (or helper `./scripts/mcp.sh`).
   * If a desktop GUI instance is running, messages forward automatically to the GUI's HTTP endpoint.
   * If the GUI is closed, executes headless against the local SQLite catalog.

---

## 2. MCP Resources Reference (`optixel://...`)

| Resource URI Template | MIME Type | Description |
|---|---|---|
| `optixel://catalog/summary` | `application/json` | Total photo count, folders, cameras, lenses, ratings, and current filter state. |
| `optixel://selection/active` | `application/json` | List of currently selected `photo_id`s in the filmstrip with EXIF metadata. |
| `optixel://photo/{id}/state` | `application/json` | Full non-destructive edit graph (tone, curves, Point Color, 3-way wheels, masks, crop). |
| `optixel://photo/{id}/telemetry` | `application/json` | 256-bin RGB/Luminance histograms, clipping percentages, percentiles, and dominant Oklch colors. |
| `optixel://photo/{id}/preview` | `image/jpeg` | Rendered viewport preview reflecting current edits (supports `?max_edge=1024\|2048&mask=<idx>`). |

---

## 3. The 10 MCP Tools Reference

### 1. `catalog_query`
Search and filter the catalog by rating, flags, camera, lens, and text.
* **Inputs**:
  * `semantic_query` (string): Matched against filenames, keywords, camera, and lens.
  * `min_rating` (0–5): Minimum star rating.
  * `flag_status` (`"pick"` | `"unflagged"` | `"reject"` | `"any"`).
  * `min_eye_sharpness` (0–100): Filter by subject sharpness measured by auto culling.
  * `limit` (default: 50).

### 2. `culling_batch_update`
Batch update photo ratings, flags, or labels.
* **Inputs**:
  * `photo_ids` (array of strings/numbers).
  * `rating` (0–5).
  * `flag` (`"pick"` | `"unflagged"` | `"reject"`).
  * `color_label` (`"none"` | `"red"` | `"yellow"` | `"green"` | `"blue"` | `"purple"`).
  * `create_virtual_copy` (boolean): Optionally duplicate records before applying.

### 3. `develop_apply_parameters`
Apply non-destructive global tone, curve, color, calibration, optics, or Photoshop Pro adjustments.
* **Inputs**:
  * `photo_ids` (array).
  * `mode` (`"absolute"` | `"relative_delta"`).
  * `parameters`: Object containing keys: `white_balance`, `tone`, `presence`, `tone_curve`, `color_mixer`, `color_grading_3way`, `point_color_swatches`, `calibration`, `detail`, `effects`, `grain`, `lens`, `transform`, `levels`, `hue_saturation`, `selective_color`, `channel_mixer`, `gradient_map`, `filters`, `image_rotation`, `crop`.

### 4. `mask_create_and_edit`
Create or update boolean masks and apply local develop adjustments inside them.
* **Inputs**:
  * `photo_id` (string/number).
  * `mask_name` (string): Identifies the mask layer.
  * `operations`: Array of mask components with `boolean_op` (`"add"`, `"subtract"`, `"intersect"`), `source_type` (`"polygon"`, `"linear_gradient"`, `"radial_gradient"`, `"luminance_range"`, `"color_range"`), `feather`, `edge_shift`, `points` (for polygons), `start`/`end` (gradients), `range`, or `color`.
  * `local_adjustments`: Tonal adjustments applied inside the mask.

### 5. `photo_inspect`
Inspect image rendering and optical telemetry.
* **Inputs**:
  * `photo_id` (string/number).
  * `max_edge` (512, 1024, or 1536).
  * `full_frame` (boolean): If true, returns the uncropped, undistorted frame matching mask coordinate systems.
  * `show_mask` (optional integer): Overlays mask coverage in red.
  * `mark` (optional `[x, y]`): Draws an indicator ring at specific coordinates (e.g. click coordinates).

### 6. `develop_auto`
Autonomous white balance and tone estimation applied with a style look.
* **Inputs**:
  * `photo_ids` (array).
  * `profile`: `"natural"` (neutral auto), `"neutral"`, `"vivid"`, `"portrait"`, `"landscape"`, `"bw"`, `"film"`, `"moody"`.

### 7. `auto_run`
Executes one of Optixel's 6 autonomous batch functions with custom settings overrides.
* **Inputs**:
  * `function`: `"cull"`, `"edit"`, `"mask"`, `"enhance"`, `"blur"`, `"duplicates"`.
  * `photo_ids` (array).
  * `settings`: Settings object overriding defaults.

### 8. `retouch_remove_objects`
Removes objects, people, or sensor dust blemishes.
* **Inputs**:
  * `photo_id` (string/number).
  * `objects`: Array of `{ name: string, points: [[x, y], ...] }` defining polygon boundaries in normalized coordinates.

### 9. `ai_reconstruct_and_retouch`
Executes classical/neural reconstruction (Denoise or Super-Resolution).
* **Inputs**:
  * `photo_id` (string/number).
  * `task`: `"ai_denoise"` | `"super_resolution"`.
  * `amount`: 0–100 strength slider.

### 10. `batch_export`
Renders full-resolution images with current edits applied.
* **Inputs**:
  * `photo_ids` (array).
  * `destination_folder` (string).
  * `format`: `"JPEG"`, `"AVIF"`, `"PNG_16"`, `"TIFF_16"`.
  * `color_space`: `"sRGB"`, `"DisplayP3"`, `"AdobeRGB"`, `"Rec2020_HDR"`.
  * `quality`: 1–100.

---

## 4. Pluggable AI Provider Router ([`ai.rs`](../src-tauri/src/ai.rs))

When the user enters a prompt in the Develop module's **✦ AI** bar, Optixel routes the request according to the active provider configured in Settings:

### 4.1 CLI Agent Execution (Claude CLI & Antigravity CLI)
* **Antigravity CLI Configuration**:
  ```bash
  antigravity -p {{AGENT_PROMPT}} --dangerously-skip-permissions
  ```
* **Claude CLI Configuration**:
  ```bash
  claude -p {{AGENT_PROMPT}} --mcp-config {{MCP_CONFIG_PATH}} --allowedTools mcp__optixel --max-turns 20 --output-format json
  ```
* **Execution Flow**:
  1. Optixel writes a temporary MCP configuration file containing the loopback HTTP bearer token.
  2. Spawns the CLI agent with the user's prompt and photo ID.
  3. The agent calls `photo_inspect`, analyzes the image, issues `develop_apply_parameters` or `mask_create_and_edit` tool calls, and inspects the result.
  4. Each tool call lands in the UI via IPC in real-time.

### 4.2 HTTP Multimodal APIs
* **Supported Backends**: Ollama (`qwen2.5vl:7b`), LM Studio / vLLM, Anthropic API, Google Gemini API, OpenAI API, OpenRouter.
* **Structured Output Schema**: Optixel enforces an `AiTuningResponse` JSON schema:
  ```json
  {
    "explanation": "Increased exposure by +0.65 EV to lift underexposed shadows and warmed white balance.",
    "adjustments": {
      "exposure": 0.65,
      "shadows": 25,
      "highlights": -15,
      "temperature_k": 5800
    },
    "masks": []
  }
  ```
* **Verification Pass**: If `max_passes > 1`, Optixel applies the adjustments to the proxy, renders a new preview, and asks the model to verify whether highlight clipping or color shifts require refinement.

