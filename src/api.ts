import { convertFileSrc, invoke } from "@tauri-apps/api/core";

export type Photo = {
  id: number;
  path: string;
  folder: string;
  filename: string;
  master_id: number | null;
  copy_name: string | null;
  width: number;
  height: number;
  captured_at: string | null;
  camera: string | null;
  lens: string | null;
  iso: number | null;
  aperture: number | null;
  shutter: string | null;
  focal_length: number | null;
  rating: number;
  flag: number;
  color_label: string;
  keywords: string;
  edited: boolean;
  thumb_v: number;
  deleted_at: string | null;
};

export type CopyType = "all" | "original" | "copy" | "edited" | "unedited_original" | "edited_original";
export type AspectFilter = "all" | "landscape" | "portrait" | "square";

export type Filter = {
  folder?: string | null;
  /** only the folder's own photos, not its subfolders */
  flat?: boolean;
  text?: string | null;
  min_rating?: number | null;
  flag?: string | null;
  color_label?: string | null;
  edited?: boolean | null;
  /** original (master) | copy (saved new edited photo) | edited | unedited_original */
  copy_type?: CopyType | null;
  /** landscape | portrait | square */
  aspect?: AspectFilter | null;
  /** Loaded pages of an Immich album (album id, or "favorites") */
  immich_album?: string | null;
  /** Only these photos (e.g. the results of an auto function) */
  ids?: number[] | null;
  /** Sidebar row being shown; UI only, the backend ignores it */
  view?: string | null;
  /** Query items in the bin (trash) */
  in_bin?: boolean | null;
  /** at most this many photos */
  limit?: number | null;
  /** only photos showing every one of these people */
  people?: number[] | null;
};

export type FolderProgress = {
  label?: string;
  done: number;
  total: number;
};

export type FolderLoading = {
  viewId: string;
  name: string;
  progress?: FolderProgress;
};

export type Culling = { rating?: number; flag?: string; color_label?: string; keywords?: string };

export type Wheel = { hue: number; sat: number; lum: number };
export type MaskComponent = {
  op: string;
  type: string;
  /** tag of an AI region, e.g. "castle" */
  name: string;
  invert: boolean;
  feather: number;
  edge_shift: number;
  start: number[];
  end: number[];
  center: number[];
  radius: number[];
  range: number[];
  color: number[];
  tolerance: number;
  /** polygon: AI outline [x, y] in the full frame */
  points: number[][];
};
/** Object removal: the outline is refilled from the area at `source` (normalised offset). */
export type Heal = { name: string; points: number[][]; source: number[] };
export type LocalAdjust = {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  temperature_shift: number;
  tint_shift: number;
  saturation: number;
};
/** `tuning`: develop settings applied on top of the photo inside the mask only (absent = none). */
/** `invert`: the inverse of the whole selection (everything but the object) */
export type Mask = { mask_id: string; name: string; enabled: boolean; invert?: boolean; components: MaskComponent[]; adjustments: LocalAdjust; tuning?: Develop | null };
export type Swatch = { id: string; src_hsl: number[]; shift_hsl: number[]; range_hsl: number[]; falloff: number };
export type Level = { black: number; gamma: number; white: number; out_black: number; out_white: number };
/** Selective Color families, in Photoshop's order */
export const FAMILIES = ["reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks"] as const;

export type AiRemoval = {
  id: string;
  label: string;
  model: string;
  prompt: string;
  enabled: boolean;
  bounds: [number, number, number, number];
  patch_file: string;
};

export type AiDenoise = {
  enabled: boolean;
  model: string;
  strength: number;
};

export type Develop = {
  profile: string;
  white_balance: { temperature_k: number; tint: number };
  basic_tone: { exposure: number; contrast: number; highlights: number; shadows: number; whites: number; blacks: number };
  presence: { texture: number; clarity: number; dehaze: number; vibrance: number; saturation: number };
  tone_curve: {
    parametric: { shadows: number; darks: number; lights: number; highlights: number; splits: number[] };
    point_luma: number[][];
    point_red: number[][];
    point_green: number[][];
    point_blue: number[][];
  };
  color_mixer: { hue: number[]; saturation: number[]; luminance: number[] };
  point_color: Swatch[];
  color_grading_3way: { shadows: Wheel; midtones: Wheel; highlights: Wheel; global: Wheel; blending: number; balance: number };
  calibration: { shadows_tint: number; red_hue: number; red_saturation: number; green_hue: number; green_saturation: number; blue_hue: number; blue_saturation: number };
  detail: { sharpening: number; sharpen_radius: number; sharpen_masking: number; color_noise_reduction: number; luminance_noise_reduction: number; luminance_detail: number };
  effects: { vignette: number; vignette_midpoint: number; vignette_feather: number };
  grain: { amount: number; size: number; roughness: number };
  /** a creative .cube 3D LUT look (a file name in the LUT folder; "" for none), applied last */
  look: { lut: string; amount: number };
  lens: {
    distortion: number;
    ca_red_cyan: number;
    ca_blue_yellow: number;
    defringe_purple: number;
    defringe_purple_hue: number[];
    defringe_green: number;
    defringe_green_hue: number[];
    vignetting: number;
    vignetting_midpoint: number;
  };
  /** upright: the Upright mode that set vertical/horizontal/rotate; guides: [[x, y], [x, y]] in the full frame */
  transform: {
    upright: string;
    guides: number[][][];
    vertical: number;
    horizontal: number;
    rotate: number;
    aspect: number;
    scale: number;
    offset_x: number;
    offset_y: number;
    constrain_crop: boolean;
  };
  /** Pro (Photoshop adjustments). Levels in 0-255 display values; selective color [c, m, y, k]; mixer rows [r, g, b, constant] % */
  levels: Record<"rgb" | "red" | "green" | "blue", Level>;
  hue_saturation: { hue: number; saturation: number; lightness: number; colorize: boolean; colorize_hue: number; colorize_saturation: number };
  selective_color: { absolute: boolean } & Record<(typeof FAMILIES)[number], number[]>;
  channel_mixer: { red: number[]; green: number[]; blue: number[]; monochrome: boolean; gray: number[] };
  /** colours are sRGB 0-1 */
  gradient_map: { amount: number; shadows: number[]; highlights: number[] };
  /** HDR Toning: method fusion (simulated bracket: frames 2-9, ev_spacing 0.5-4, exposedness / contrast_weight /
   *  saturation_weight 0-200, natural 0-100) | local | equalize | gamma | highlights; radius / edge / compression 0-100, exposure EV,
   *  gamma 0.1-3, detail / fine_detail -100..300 %, the rest -100..100; curve [x, y] 0-255 on display brightness */
  hdr: {
    enabled: boolean;
    method: string;
    frames: number;
    ev_spacing: number;
    exposedness: number;
    contrast_weight: number;
    saturation_weight: number;
    natural: number;
    radius: number;
    edge: number;
    smooth_edges: boolean;
    compression: number;
    exposure: number;
    gamma: number;
    detail: number;
    fine_detail: number;
    shadows: number;
    highlights: number;
    vibrance: number;
    saturation: number;
    curve: number[][];
  };
  filters: { blur: number; invert: boolean; posterize: number; threshold: number };
  /** angle 0/90/180/270 clockwise, then the mirrors; the crop is in the rotated frame */
  image_rotation: { angle: number; flip_h: boolean; flip_v: boolean };
  /** in the straightened, rotated frame */
  crop: { x: number; y: number; w: number; h: number; angle: number };
  /** RAW sensor decode & calibration (LibRaw C++ engine, demosaic algorithm, highlight reconstruction, optical black offset, linear sensor denoise) */
  raw?: {
    engine: string;
    demosaic: string;
    highlights: string;
    black_offset: number;
    raw_denoise: number;
  };
  retouch: Heal[];
  ai_removals?: AiRemoval[];
  ai_denoise?: AiDenoise;
  masks: Mask[];
  ai_provenance_log: { timestamp?: string; provider?: string; action?: string }[];
};

export type Provider = {
  id: string;
  kind: string;
  name: string;
  command: string;
  endpoint: string;
  model: string;
  has_key: boolean;
  timeout_s: number;
  max_passes: number;
};
export type Theme = "system" | "dark" | "light" | "slate" | "warm" | "nordic" | "cyber" | "emerald" | "oled" | "slate_light" | "warm_light" | "nordic_light";

export type Settings = {
  theme?: Theme;
  mcp_enabled: boolean;
  gpu_accel?: boolean;
  ai_denoise_model?: string;
  ai_removal_model?: string;
  ai_align_model?: string;
  /** Segment Anything: SAM 2.1 for click selection, SAM 3 for select by name / Auto masks; empty: best installed */
  ai_select_model?: string;
  ai_concept_model?: string;
  mcp_port: number;
  mcp_token: string;
  providers: Provider[];
  active_provider: string;
  immich_url: string;
  google_client_id: string;
  apple_photos_path?: string;
  amazon_photos_path?: string;
  adobe_cloud_path?: string;
  auto: AutoSettings;
  network_drives: NetworkDrive[];
  clouds: CloudAccount[];
  auto_check_updates?: boolean;
  /** People: find and group faces, search by person */
  faces_enabled?: boolean;
};

// --- people (faces.rs) ---
export type Person = { id: number; name: string; hidden: boolean; faces: number; photos: number; cover: number | null };
/** bbox: x, y, w, h as fractions of the photo */
export type Face = { id: number; photo_id: number; person_id: number | null; bbox: [number, number, number, number]; score: number; captured_at: string | null; filename: string; similarity: number | null };
export type FaceStats = {
  eligible: number;
  scanned: number;
  with_faces: number;
  faces: number;
  people: number;
  named: number;
  hidden: number;
  by_year: [string, number][];
  /** [faces in the photo (6 = 6 or more), photos] */
  per_photo: [number, number][];
};

export type ReleaseAsset = {
  name: string;
  size: number;
  browser_download_url: string;
  content_type: string;
  sha256?: string | null;
};

export type UpdateInfo = {
  current_version: string;
  latest_version: string;
  has_update: boolean;
  release_name: string;
  release_tag: string;
  release_notes: string;
  published_at: string;
  html_url: string;
  recommended_asset: ReleaseAsset | null;
  all_assets: ReleaseAsset[];
};

export type DownloadProgress = {
  status: "downloading" | "completed" | "error";
  downloaded: number;
  total: number;
  percent: number;
  path?: string | null;
  error?: string | null;
  sha256?: string | null;
};

// --- auto functions: their advanced settings, as kept in Settings.auto ---
export type AutoFunction = "cull" | "edit" | "mask" | "enhance" | "hdr" | "blur" | "duplicates";
/** detail: where sharpness is checked: fast (1024 px) | screen (2048 px) | full (100%) */
export type CullSettings = {
  detail: string;
  reject_blurry: boolean;
  blur_threshold: number;
  use_subject: boolean;
  reject_blinks?: boolean;
  reject_exposure: boolean;
  clip_limit: number;
  burst_seconds: number;
  similarity: number;
  pick_best: boolean;
  reject_similar: boolean;
  stars: boolean;
  labels: boolean;
  keywords: boolean;
  ai_keywords: boolean;
  keep_manual: boolean;
};
export type EditSettings = {
  profile: string;
  strength: number;
  white_balance: boolean;
  tone: boolean;
  look: boolean;
  upright: string;
  chromatic_aberration: boolean;
  noise_reduction: boolean;
  skip_edited: boolean;
  virtual_copy: boolean;
  ai_type: boolean;
};
export type MaskSettings = { targets: string[]; method: string; feather: number; edge_shift: number; replace: boolean; adjust: number };
export type EnhanceSettings = {
  denoise: boolean;
  strength: number;
  detail: number;
  color: number;
  super_resolution: boolean;
  scale: number;
  sharpen: number;
  max_megapixels: number;
  engine: string;
  command: string;
  format: string;
  folder: string;
  copy_edits: boolean;
};
/** grouping: brackets (taken within bracket_seconds) | all; deghost: off | low | medium | high */
export type HdrSettings = { grouping: string; bracket_seconds: number; align: boolean; deghost: string; auto_tone: boolean; folder: string };
export type BlurSettings = { detail: string; threshold: number; subject: string; motion: number; missed_focus: boolean; action: string };
export type DuplicateSettings = { mode: string; similarity: number; window_minutes: number; keep: string; action: string };
/** Run Auto Catalog & Culling on photos new to the catalog, by where they came from */
export type OnImport = { enabled: boolean; folders: boolean; cloud: boolean; immich: boolean; duplicates: boolean };
export type AutoSettings = {
  cull: CullSettings;
  edit: EditSettings;
  mask: MaskSettings;
  enhance: EnhanceSettings;
  hdr: HdrSettings;
  blur: BlurSettings;
  duplicates: DuplicateSettings;
  on_import: OnImport;
};
/** An auto function's summary; `rows` holds one entry per photo (per group for duplicates) */
export type AutoResult = { function: AutoFunction; total: number; cancelled: boolean; rows: Record<string, unknown>[] } & Record<string, unknown>;
/** Progress of a running job (auto function, duplicate, culling of new photos) */
export type JobProgress = { job: string; label?: string; done: number; total: number; finished?: boolean };

// --- storage ---
export type NetworkDrive = { id: string; name: string; url: string; path: string };
export type CloudKind = "webdav" | "s3" | "dropbox" | "onedrive";
export type CloudAccount = { id: string; kind: CloudKind; name: string; endpoint: string; user: string; bucket: string; region: string; folders: string[] };
export type CloudEntry = { name: string; path: string; dir: boolean; size: number };
export type CloudFolder = { id: string; path: string; local: string };
/** Where photos are duplicated from or to */
export type Place =
  | { kind: "folder"; path: string }
  | { kind: "photos"; ids: number[] }
  | { kind: "immich"; album: string | null }
  | { kind: "google" }
  | { kind: "cloud"; account: string; path: string };
export type DuplicateOptions = {
  subfolders: boolean;
  only_picks: boolean;
  min_rating: number;
  skip_existing: boolean;
  sidecars: boolean;
  verify: boolean;
  keep_dates: boolean;
  catalog: boolean;
  /** then Auto Catalog & Culling on the copies; unset: as Settings.auto.on_import says for duplicates */
  cull?: boolean;
};
export type DuplicateResult = {
  listed: number;
  selected: number;
  copied: number;
  skipped: number;
  failed: number;
  bytes: number;
  errors: string[];
  cataloged: number;
  culled: { picks: number; rejects: number; rated: number } | null;
  cancelled: boolean;
};
export type Source = "immich" | "google" | "apple" | "amazon" | "adobe";
export type ImmichAlbum = { id: string; name: string; count: number };
export type RemoteImport = { folder: string; imported: number; skipped: number; failed: number; error: string | null };
export type AppInfo = { data_dir: string; thumbs_dir: string; exe: string | null; mcp_url: string; mcp_status: string };
export type HistoryEntry = { id: number; label: string; created_at: string; current: boolean };
export type Rendered = { hist: Uint32Array[]; image: ImageData };
/** Preview overlays: soft-proof gamut (sRGB | DisplayP3 | AdobeRGB | ProPhotoRGB | ACEScg | FineArtMatte | BarytaGloss | custom .icc), intent, BPC, clipping, Visualize Spots threshold. */
export type Proof = {
  gamut: string | null;
  clipping: boolean;
  spots: number | null;
  intent?: string | null;
  bpc?: boolean;
  simulate_paper?: boolean;
  gamut_warn?: boolean;
};
export type AfPoint = {
  x: number;
  y: number;
  w: number;
  h: number;
  active: boolean;
  kind: "single" | "eye" | "face" | "zone" | string;
};
export type RawInfo = {
  camera_model: string;
  vendor: string;
  cfa_pattern: string;
  is_xtrans: boolean;
  raw_width: number;
  raw_height: number;
  black_levels: [number, number, number, number];
  border_black_counts: [number, number, number, number];
  white_level: number;
  linear_max: [number, number, number, number];
  cam_wb: [number, number, number, number];
  shutter_count: number;
  lens_serial: string;
  body_serial: string;
  focus_distance_m: number;
  af_points: AfPoint[];
};
export type IccProfileEntry = {
  id: string;
  label: string;
  kind: "builtin" | "printer" | "custom" | string;
  description: string;
};
export type ScopesData = {
  parade: ImageData;
  waveform: ImageData;
};
export type FocusPeakingResult = {
  /** of the eyes when the face scan found faces, else of the frame */
  focus_score: number;
  faces: number;
  /** eye aspect ratio of the most closed face, from its landmarks; null without faces or the landmark model */
  eye_openness: number | null;
  blink_detected: boolean | null;
  overlay_data_url: string;
};
/** upload: send the renders to Immich / Google Photos / `cloud:<account id>` (into `upload_path`) instead of keeping them in `folder` */
export type ExportReq = { ids: number[]; folder: string; format: string; color_space: string; quality: number; max_edge: number | null; upload?: string | null; upload_path?: string };
export type CleanUp = "thumbs" | "originals" | "immich_albums" | "cloud" | "ai_models" | "ai_cache" | "missing" | "history" | "compact" | "all";
type Named = { name: string; count: number }[];
export type ImmichAlbumStorage = { name: string; path: string; bytes: number; photos: number };
export type AiModelDetail = { id: string; name: string; kind: string; bytes: number; size_mb: number };
export type OtherDetail = { name: string; bytes: number };
export type CatalogDetails = { db: number; wal: number; shm: number; free: number };

/** Sizes in bytes. `folders` are local folders; an offline one is not there at all (e.g. an unplugged drive). */
export type StorageReport = {
  data_dir: string;
  default_data_dir?: string;
  is_custom_location?: boolean;
  total: number;
  catalog: number;
  catalog_details?: CatalogDetails;
  thumbs: number;
  originals: number;
  immich_albums?: ImmichAlbumStorage[];
  immich_albums_bytes?: number;
  google: number;
  cloud?: number;
  ai_models?: number;
  ai_models_count?: number;
  ai_models_details?: AiModelDetail[];
  ai_cache?: { bytes: number; files: number };
  other?: number;
  other_details?: OtherDetail[];
  summary: {
    total_photos: number;
    virtual_copies: number;
    edited: number;
    picks: number;
    rejects: number;
    rated_3_plus: number;
    cameras: Named;
    lenses: Named;
    ratings?: { rating: number; count: number }[];
    flags?: { flag: number; count: number }[];
    years?: { year: string; count: number }[];
    iso_ranges?: { range: string; count: number }[];
    aperture_ranges?: { range: string; count: number }[];
    focal_ranges?: { range: string; count: number }[];
    formats?: { ext: string; count: number }[];
    raw_count?: number;
    std_count?: number;
    history_stats?: { total_steps: number; max_steps: number; avg_steps: number };
  };
  immich_photos: number;
  history_steps: number;
  ai: { photos: number; edits: number };
  folders: { folder: string; photos: number; bytes: number; missing: number; online: boolean }[];
  clean: Record<CleanUp, { count: number; bytes: number }>;
};

export type StorageValidation = {
  valid: boolean;
  error?: string;
  is_same?: boolean;
  has_existing_catalog?: boolean;
  has_existing_config?: boolean;
  existing_catalog_bytes?: number;
  existing_photos_count?: number | null;
  is_empty?: boolean;
  target?: string;
  current_dir?: string;
  current_bytes?: number;
  default_dir?: string;
  is_custom?: boolean;
};

export type StorageMoveMode = "copy" | "move" | "existing" | "switch" | "fresh";

export type StorageMoveResult = {
  success: boolean;
  data_dir: string;
  is_custom: boolean;
  total_bytes: number;
  message: string;
};

function parseRender(buf: ArrayBuffer): Rendered {
  const dv = new DataView(buf);
  const w = dv.getUint32(0, true);
  const h = dv.getUint32(4, true);
  const hist = [0, 1, 2, 3].map((i) => new Uint32Array(buf.slice(8 + i * 1024, 8 + (i + 1) * 1024)));
  return { hist, image: new ImageData(new Uint8ClampedArray(buf, 8 + 4096, w * h * 4), w, h) };
}

function parseScopes(buf: ArrayBuffer): ScopesData {
  const dv = new DataView(buf);
  const sw = dv.getUint32(0, true);
  const sh = dv.getUint32(4, true);
  const byteLen = sw * sh * 4;
  const parade = new ImageData(new Uint8ClampedArray(buf, 8, byteLen), sw, sh);
  const waveform = new ImageData(new Uint8ClampedArray(buf, 8 + byteLen, byteLen), sw, sh);
  return { parade, waveform };
}

export type AiHardware = { gpu: string | null; vram_mb: number | null; device: string };
export type AiModel = {
  id: string;
  /** denoise | inpaint | generate | faces | faceswap | align */
  kind: string;
  name: string;
  blurb: string;
  license: string;
  source: string;
  size_mb: number;
  installed: boolean;
  fits: boolean;
  recommended: boolean;
  note: string;
};

export type AlignResult = {
  transform: Develop["transform"];
  roll_deg: number;
  pitch_deg: number;
  vfov_deg: number;
  confidence: number;
  mode: "horizontal" | "vertical" | "both" | string;
};

/** One brush stroke over the photo: its path as fractions of the frame, and the brush radius as a fraction of the frame's width */
export type Stroke = { points: [number, number][]; radius: number };

/** A painted selection snapped to the picture's edges: the matte (0-255) inside a window of the preview frame. */
export type Snap = { w: number; h: number; x0: number; y0: number; ww: number; wh: number; alpha: Uint8Array };

export const api = {
  /** Snap a rough outline (fractions of the frame) to the object's edges, for confirming a removal */
  snapObject: async (id: number, strokes: Stroke[]): Promise<Snap> => {
    const buf = await invoke<ArrayBuffer>("snap_object", { id, strokes });
    const v = new DataView(buf);
    const n = (i: number) => v.getUint32(i * 4, true);
    return { w: n(0), h: n(1), x0: n(2), y0: n(3), ww: n(4), wh: n(5), alpha: new Uint8Array(buf, 24) };
  },
  /** This machine's GPU and the AI denoise models, the best one for it marked */
  aiModels: () => invoke<{ hardware: AiHardware; models: AiModel[] }>("ai_models"),
  aiDownload: (id: string) => invoke<void>("ai_download", { id }),
  aiRemove: (id: string) => invoke<void>("ai_remove", { id }),
  appInfo: () => invoke<AppInfo>("app_info"),
  query: (filter: Filter) => invoke<Photo[]>("query_photos", { filter }),
  photo: async (id: number) => (await invoke<Photo[]>("query_photos", { filter: { ids: [id] } }))[0],
  folders: () => invoke<[string, number][]>("list_folders"),
  importFolder: (path: string) => invoke<number>("import_folder", { path }),
  /** subfolders of a folder, and how many photos are directly in it */
  browseFolder: (path: string) => invoke<[string[], number]>("browse_folder", { path }),
  setSelection: (ids: number[]) => invoke("set_selection", { ids }),
  update: (ids: number[], update: Culling) => invoke<number>("update_photos", { ids, update }),
  virtualCopy: (id: number) => invoke<number>("virtual_copy", { id }),
  /** Save current edit settings as a new photo entry (virtual copy) in catalog */
  saveAsNewPhoto: (id: number, develop?: Develop, name?: string) =>
    invoke<Photo>("save_as_new_photo", { id, develop, name: name?.trim() || null }),
  remove: (ids: number[]) => invoke("remove_photos", { ids }),
  binCount: () => invoke<number>("bin_count"),
  moveToBin: (ids: number[]) => invoke<number>("move_to_bin", { ids }),
  restoreFromBin: (ids: number[]) => invoke<number>("restore_from_bin", { ids }),
  permanentDelete: (ids: number[]) => invoke<number>("permanent_delete", { ids }),
  emptyBin: () => invoke<number>("empty_bin"),
  getDevelop: (id: number) => invoke<Develop>("get_develop", { id }),
  defaultDevelop: () => invoke<Develop>("default_develop"),
  /** fullFrame: no crop or straighten, so view coordinates are mask coordinates; hires: the full-size photo instead of the 1600 px proxy */
  render: async (id: number, develop: Develop, skipCrop: boolean, overlayMask: number | null, fullFrame = false, proof: Proof | null = null, hires = false) =>
    parseRender(await invoke<ArrayBuffer>("render_preview", { id, develop, skipCrop, overlayMask, fullFrame, proof, hires })),
  /** Real-Time RGB Parade & Luma/Chroma Waveform Scopes (AOT SPIR-V GPU compute shader / C++ fallback) */
  renderScopes: async (id: number, develop: Develop, scopeW = 288, scopeH = 120): Promise<ScopesData> =>
    parseScopes(await invoke<ArrayBuffer>("render_scopes", { id, develop, scopeW, scopeH })),
  /** LibRaw C++ sensor optical calibration & MakerNotes (active AF points, Eye-AF, shutter count, lens/body serial) */
  rawInfo: (id: number) => invoke<RawInfo>("raw_info", { id }),
  /** LittleCMS2 built-in wide-gamut / printer profiles and user-imported .icc/.icm profiles */
  iccProfiles: () => invoke<IccProfileEntry[]>("icc_profiles"),
  iccImportProfile: (path: string) => invoke<IccProfileEntry>("icc_import_profile", { path }),
  lutList: () => invoke<string[]>("lut_list"),
  lutImport: (path: string) => invoke<string>("lut_import", { path }),
  /** Bidirectional Adobe Lightroom .XMP sidecar sync (read / write) */
  xmpSync: (ids: number[], mode: "read" | "write", copyright?: string, creator?: string) =>
    invoke<{ synced: number; mode: string }>("xmp_sync", { ids, mode, copyright: copyright ?? null, creator: creator ?? null }),
  /** SIMD lossless JPEG DCT block transform via libturbojpeg */
  jpegLossless: (id: number, op: string, crop?: [number, number, number, number]) =>
    invoke<{ id: number; op: string; path: string }>("jpeg_lossless", { id, op, crop: crop ?? null }),
  /** Native C++ Focus-Peaking & Blink / EAR scoring */
  focusPeaking: (id: number, threshold?: number) =>
    invoke<FocusPeakingResult>("focus_peaking", { id, threshold: threshold ?? null }),
  /** 16-bit feathered semantic mask (subject, sky, background) via He et al. Guided Filter */
  semanticMask16Bit: (id: number, target: string, feather?: number) =>
    invoke<{ id: number; target: string; width: number; height: number; bit_depth: number; coverage_pct: number }>("semantic_mask_16bit", {
      id,
      target,
      feather: feather ?? null,
    }),
  /** Upright mode: off | auto | level | vertical | full | guided (uses develop.transform.guides) */
  upright: (id: number, develop: Develop, mode: string) => invoke<Develop["transform"]>("upright", { id, develop, mode }),
  /** AI Auto Align using PerspectiveFields (horizontal | vertical | both) */
  aiAlign: (id: number, develop: Develop, model: string, mode: "horizontal" | "vertical" | "both") =>
    invoke<AlignResult>("ai_align", { id, develop, model, mode }),
  /** Measured lateral chromatic aberration: [ca_red_cyan, ca_blue_yellow] */
  autoCa: (id: number) => invoke<[number, number]>("auto_ca", { id }),
  /** Heal a circle at (x, y) of the full frame; radius is a fraction of the long edge. Commits one history step. */
  healSpot: (id: number, x: number, y: number, radius: number) => invoke<Develop>("heal_spot", { id, x, y, radius }),
  /** Auto fine tune to a profile (natural | neutral | vivid | portrait | landscape | bw | film | moody); not saved */
  autoDevelop: (id: number, develop: Develop, profile: string, strength?: number) =>
    invoke<Develop>("auto_develop", { id, develop, profile, strength: strength != null ? strength / 100 : 1.0 }),
  /** One-click AI auto adjustment: the local retouch model's result as sliders (`amount` and `strength` 0-100), with a profile's look on top when given; not saved */
  aiAdjust: (id: number, develop: Develop, profile: string | null, strength: number, amount: number) =>
    invoke<Develop>("ai_adjust", { id, develop, profile, strength: strength / 100, amount: amount / 100 }),
  /** the develop profile for what the photo shows, and the AI's error when it fell back to keywords */
  photoType: (id: number) => invoke<[string, string | null]>("photo_type", { id }),
  autoWhiteBalance: (id: number) => invoke<Develop["white_balance"]>("auto_white_balance", { id }),
  /** under develop's white balance */
  autoTone: (id: number, develop: Develop) => invoke<Develop["basic_tone"]>("auto_tone", { id, develop }),
  /** Auto HDR toning from image dynamic range analysis */
  autoHdr: (id: number) => invoke<Develop["hdr"]>("auto_hdr", { id }),
  /** Heal away the AI-selected objects of selection `mask` and drop it. Commits one history step. */
  removeSelection: (id: number, mask: number) => invoke<Develop>("remove_selection", { id, mask }),
  save: (id: number, develop: Develop, label: string, replacePrefix?: string) =>
    invoke<Develop>("save_develop", { id, develop, label, replacePrefix: replacePrefix ?? null }),
  history: (id: number) => invoke<HistoryEntry[]>("get_history", { id }),
  gotoHistory: (id: number, hid: number) => invoke<Develop>("goto_history", { id, hid }),
  deleteHistory: (id: number, hid: number) => invoke<Develop | null>("delete_history", { id, hid }),
  undoRedo: (id: number, dir: number) => invoke<Develop | null>("undo_redo", { id, dir }),
  export: (req: ExportReq) => invoke<string[]>("export_photos", { req }),
  settings: () => invoke<Settings>("get_settings"),
  saveSettings: (settings: Settings) => invoke("save_settings", { settings }),
  gpuInfo: () => invoke<string | null>("gpu_info"),
  regenerateToken: () => invoke<string>("regenerate_token"),
  setKey: (id: string, key: string) => invoke("set_provider_key", { id, key }),
  aiRun: (id: number, prompt: string, provider: string) => invoke<string>("ai_run", { id, prompt, provider }),
  /** AI selection: each clicked object at (x, y) of the full frame starts its own selection (or subtracts from `mask` on ⌥-click) */
  /** SAM 2.1 on this computer: the object at (x, y) as a new mask (or taken out of `mask`); `part`: just the clicked part */
  samSelect: (id: number, x: number, y: number, mask: number | null, subtract: boolean, part: boolean) =>
    invoke<{ develop: Develop; mask: number }>("sam_select", { id, x, y, mask, subtract, part }),
  /** SAM 3 on this computer: every instance of `what` ("sky", "person") as one new mask */
  samSelectConcept: (id: number, what: string, mask: number | null, subtract: boolean, invert = false) =>
    invoke<{ develop: Develop; mask: number; found: number }>("sam_select_concept", { id, what, mask, subtract, invert }),
  aiSelect: (id: number, x: number, y: number, provider: string, mask: number | null, subtract: boolean) =>
    invoke<{ develop: Develop; mask: number }>("ai_select", { id, x, y, provider, mask, subtract }),
  aiInpaintInplace: (id: number, model: string, strokes: Stroke[], grow: number, prompt: string) =>
    invoke<Develop>("ai_inpaint_inplace", { id, model, strokes, grow, prompt }),
  /** Put the biggest face of library photo `source` onto every face of photo `id`: one AI layer per face */
  aiFaceSwap: (id: number, model: string, source: number) => invoke<Develop>("ai_face_swap", { id, model, source }),
  aiDenoiseInplace: (id: number, model: string, strength: number) =>
    invoke<Develop>("ai_denoise_inplace", { id, model, strength }),
  aiDeleteRemoval: (id: number, removalId: string) =>
    invoke<Develop>("ai_delete_removal", { id, removalId }),
  immichConnect: (url: string, key: string) => invoke<Settings>("immich_connect", { url, key }),
  immichAlbums: () => invoke<ImmichAlbum[]>("immich_albums"),
  /** Load one page of an album (null = favorites) into the catalog; page 1 starts it over */
  immichOpen: (album: string | null, page: number) => invoke<{ next: number | null }>("immich_open", { album, page }),
  googleConnect: (clientId: string, clientSecret: string) => invoke<Settings>("google_connect", { clientId, clientSecret }),
  googleImport: () => invoke<RemoteImport>("google_import"),
  appleDetect: () => invoke<{ found: boolean; default_path: string }>("apple_detect"),
  appleConnect: (path: string) => invoke<Settings>("apple_connect", { path }),
  appleImport: () => invoke<RemoteImport>("apple_import"),
  amazonDetect: () => invoke<{ found: boolean; default_path: string }>("amazon_detect"),
  amazonConnect: (path: string, accountName?: string) => invoke<Settings>("amazon_connect", { path, accountName: accountName ?? "" }),
  amazonImport: () => invoke<RemoteImport>("amazon_import"),
  adobeDetect: () => invoke<{ found: boolean; default_path: string }>("adobe_detect"),
  adobeConnect: (path: string, accountName?: string) => invoke<Settings>("adobe_connect", { path, accountName: accountName ?? "" }),
  adobeImport: () => invoke<RemoteImport>("adobe_import"),
  remoteDisconnect: (source: Source) => invoke<Settings>("remote_disconnect", { source }),
  storage: () => invoke<StorageReport>("storage_report"),
  validateStorageLocation: (target: string) => invoke<StorageValidation>("validate_storage_location", { target }),
  moveStorageLocation: (target: string, mode: StorageMoveMode, cleanup_old?: boolean) =>
    invoke<StorageMoveResult>("move_storage_location", { target, mode, cleanupOld: cleanup_old ?? false }),
  resetStorageLocation: (mode: StorageMoveMode = "copy", cleanup_old?: boolean) =>
    invoke<StorageMoveResult>("reset_storage_location", { mode, cleanupOld: cleanup_old ?? false }),
  syncLibraryFolders: (folders?: string[]) =>
    invoke<number>("sync_library_folders", { folders: folders ?? null }),
  /** Re-extract or re-render thumbnails from source files on disk to fix missing or mismatched previews */
  regenerateThumbnails: (opts?: { ids?: number[]; folder?: string }) =>
    invoke<number>("regenerate_thumbnails", { ids: opts?.ids ?? null, folder: opts?.folder ?? null }),
  /** Open/reveal file or directory in OS native file manager (macOS Finder, Windows File Explorer, Linux file manager) */
  showInFolder: (path: string) => invoke<void>("show_in_folder", { path }),
  /** Returns how many files, photos or history steps went */
  cleanUp: (what: CleanUp | string) => invoke<number>("clean_up", { what }),
  /** Run an auto function on photos; `job` names it for cancelJob */
  autoRun: (fn: AutoFunction, ids: number[], settings: object, job: string) => invoke<AutoResult>("auto_run", { function: fn, ids, settings, job }),
  cancelJob: (job: string) => invoke<boolean>("cancel_job", { job }),
  autoDefaults: () => invoke<AutoSettings>("auto_defaults"),
  networkAdd: (drive: Partial<NetworkDrive>) => invoke<Settings>("network_add", { drive }),
  networkRemove: (id: string) => invoke<Settings>("network_remove", { id }),
  /** Where a share usually appears once connected, or null */
  networkMountPoint: (url: string) => invoke<string | null>("network_mount_point", { url }),
  networkStatus: () => invoke<{ id: string; online: boolean }[]>("network_status"),
  /** Connect the share if needed; returns its folder */
  networkConnect: (id: string) => invoke<string>("network_connect", { id }),
  /** Connect if needed and pick up new photos, like opening a folder */
  networkOpen: (id: string) => invoke<{ path: string; new: number }>("network_open", { id }),
  /** WebDAV or S3 account, checked by listing it */
  cloudConnect: (account: Partial<CloudAccount>, secret: string) => invoke<Settings>("cloud_connect", { account, secret }),
  cloudDropboxStart: (appKey: string) => invoke("cloud_dropbox_start", { appKey }),
  cloudDropboxFinish: (account: Partial<CloudAccount>, code: string) => invoke<Settings>("cloud_dropbox_finish", { account, code }),
  cloudOnedriveConnect: (account: Partial<CloudAccount>) => invoke<Settings>("cloud_onedrive_connect", { account }),
  cloudDisconnect: (id: string) => invoke<Settings>("cloud_disconnect", { id }),
  cloudBrowse: (id: string, path: string) => invoke<CloudEntry[]>("cloud_browse", { id, path }),
  /** Download what's new in a cloud folder and add it to the catalog */
  cloudSync: (id: string, path: string) => invoke<{ folder: string; imported: number; downloaded: number; failed: number; error: string | null }>("cloud_sync", { id, path }),
  cloudLocalDirs: () => invoke<CloudFolder[]>("cloud_local_dirs"),
  cloudForgetFolder: (id: string, path: string) => invoke<Settings>("cloud_forget_folder", { id, path }),
  duplicate: (from: Place, to: Place, options: DuplicateOptions, job: string) => invoke<DuplicateResult>("duplicate", { from, to, options, job }),
  /** Check for the latest version of Optixel Studio */
  checkForUpdates: () => invoke<UpdateInfo>("check_for_updates"),
  /** Download an update installer asset with progress tracking and SHA-256 verification */
  downloadUpdate: (url: string, filename: string, expectedSha256?: string | null) =>
    invoke<string>("download_update", { url, filename, expectedSha256: expectedSha256 ?? null }),
  /** Launch or open the downloaded update installer / package */
  installUpdate: (path: string) => invoke<void>("install_update", { path }),
  /** Find and group faces in photos not scanned yet (job "faces-scan") */
  facesScan: (ids?: number[]) => invoke<{ photos: number; faces: number; failed: number }>("faces_scan", { ids: ids ?? null }),
  people: () => invoke<Person[]>("faces_people"),
  facesOf: (person: number) => invoke<Face[]>("faces_of_person", { person }),
  /** faces of other people that look like this one, most alike first */
  facesSimilar: (face: number) => invoke<Face[]>("faces_similar", { face }),
  faceStats: () => invoke<FaceStats>("faces_stats"),
  /** a name someone else has merges the two; returns the person the faces end up in */
  renamePerson: (person: number, name: string) => invoke<number>("faces_rename", { person, name }),
  hidePeople: (people: number[], hidden: boolean) => invoke<void>("faces_hide", { people, hidden }),
  setCover: (person: number, face: number) => invoke<void>("faces_set_cover", { person, face }),
  /** everyone into the first; returns it */
  mergePeople: (people: number[]) => invoke<number>("faces_merge", { people }),
  /** to `person`, or a new person when null; returns the person */
  moveFaces: (faces: number[], person: number | null) => invoke<number>("faces_move", { faces, person }),
  deleteFaces: (faces: number[]) => invoke<void>("faces_delete", { faces }),
  deletePeople: (people: number[]) => invoke<void>("faces_delete_people", { people }),
  resetFaces: () => invoke<void>("faces_reset"),
  /** Open an external URL safely in the default system browser */
  openExternalUrl: (url: string) => invoke<void>("open_external_url", { url }),
};

/** Decimal units, as Finder shows them. */
export const size = (n: number) => {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  for (; n >= 1000 && i < units.length - 1; i++) n /= 1000;
  return `${n.toFixed(i && n < 10 ? 1 : 0)} ${units[i]}`;
};

let thumbsDir = "";
export function setThumbsDir(dir: string) {
  thumbsDir = dir + (dir.includes("\\") ? "\\" : "/");
}
export const faceUrl = (face: number) => convertFileSrc(`${thumbsDir}faces${thumbsDir.slice(-1)}${face}.jpg`);
export const thumbUrl = (p: Photo) => `${convertFileSrc(`${thumbsDir}${p.id}.jpg`)}?v=${p.thumb_v}`;

export const FLAGS = ["reject", "unflagged", "pick"] as const;
export const flagName = (f: number) => FLAGS[f + 1];
export const LABELS = ["red", "yellow", "green", "blue", "purple"] as const;
export const LABEL_COLORS: Record<string, string> = {
  red: "#e5484d",
  yellow: "#f5d90a",
  green: "#46a758",
  blue: "#3e63dd",
  purple: "#8e4ec6",
};

/** Immutable set of a nested value, e.g. setPath(d, ["basic_tone", "exposure"], 1). */
export function setPath<T>(obj: T, path: (string | number)[], v: unknown): T {
  const copy = structuredClone(obj);
  let o = copy as Record<string | number, unknown>;
  for (const k of path.slice(0, -1)) o = o[k] as Record<string | number, unknown>;
  o[path[path.length - 1]] = v;
  return copy;
}
export function getPath(obj: unknown, path: (string | number)[]): unknown {
  return path.reduce<unknown>((o, k) => (o == null ? undefined : (o as Record<string | number, unknown>)[k]), obj);
}

/** sRGB 8-bit -> Oklch [hue°, chroma, L] (same maths as the Rust pipeline). */
export function srgbToOklch(r: number, g: number, b: number): [number, number, number] {
  const lin = [r, g, b].map((v) => {
    v /= 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  const l = Math.cbrt(0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]);
  const m = Math.cbrt(0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]);
  const s = Math.cbrt(0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  return [((Math.atan2(B, A) * 180) / Math.PI + 360) % 360, Math.hypot(A, B), L];
}

/** Monotone cubic through curve points (0–255), same as the Rust Spline. */
export function spline(pts: number[][]): (x: number) => number {
  const p = [...pts].sort((a, b) => a[0] - b[0]);
  const n = p.length;
  if (n < 2) return (x) => x;
  const x = p.map((q) => q[0]);
  const y = p.map((q) => q[1]);
  const d = x.slice(0, -1).map((_, k) => (y[k + 1] - y[k]) / Math.max(1e-6, x[k + 1] - x[k]));
  const m = x.map((_, k) => (k === 0 ? d[0] : k === n - 1 ? d[n - 2] : d[k - 1] * d[k] <= 0 ? 0 : (d[k - 1] + d[k]) / 2));
  for (let k = 0; k < n - 1; k++) {
    if (d[k] === 0) {
      m[k] = m[k + 1] = 0;
      continue;
    }
    const a = m[k] / d[k];
    const b = m[k + 1] / d[k];
    const s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      m[k] = t * a * d[k];
      m[k + 1] = t * b * d[k];
    }
  }
  return (v) => {
    if (v <= x[0]) return y[0];
    if (v >= x[n - 1]) return y[n - 1];
    let k = 0;
    while (x[k + 1] <= v) k++;
    const h = x[k + 1] - x[k];
    const t = (v - x[k]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h * m[k] + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h * m[k + 1];
  };
}

// Folders added with "Add folder" (the roots of the sidebar tree) are remembered per viewer.
const ROOTS_KEY = "optixel.folderRoots";
export const loadRoots = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem(ROOTS_KEY) ?? "[]");
  } catch {
    return [];
  }
};
const saveRoots = (r: string[]) => {
  try {
    localStorage.setItem(ROOTS_KEY, JSON.stringify(r));
  } catch {
    /* storage unavailable: the roots last for this session only */
  }
  return r;
};
export const rememberRoot = (p: string) => saveRoots([...loadRoots().filter((r) => r !== p), p]);
export const forgetRoot = (p: string) => saveRoots(loadRoots().filter((r) => r !== p));

const LAST_FOLDER_KEY = "optixel.lastFolder";
export const loadLastFolder = (): string | null => {
  try {
    return localStorage.getItem(LAST_FOLDER_KEY);
  } catch {
    return null;
  }
};
export const rememberLastFolder = (folder: string | null) => {
  try {
    if (folder) {
      localStorage.setItem(LAST_FOLDER_KEY, folder);
    } else {
      localStorage.removeItem(LAST_FOLDER_KEY);
    }
  } catch {
    /* storage unavailable */
  }
};

/**
 * Returns the first item that appears in the Folders section of the sidebar:
 * 1. The first added root folder (if any).
 * 2. Otherwise, the first local catalog folder (excluding remote synced cloud folders and network drives).
 * 3. Otherwise, the first network drive.
 * 4. Otherwise, the first folder in the catalog list.
 */
export const getFirstFolder = (
  roots: string[],
  folders: [string, number][],
  dataDir: string,
  drives: NetworkDrive[] = []
): string | null => {
  if (roots.length > 0) {
    return roots[0];
  }
  const sep = dataDir.includes("\\") ? "\\" : "/";
  const remoteDir = dataDir ? `${dataDir}${sep}remote` : "";
  const under = (folder: string, root: string) =>
    Boolean(root) && (folder === root || folder.startsWith(root.replace(/[\\/]+$/, "") + sep));
  const local = folders.filter(
    ([folder]) => (!remoteDir || !under(folder, remoteDir)) && !drives.some((d) => under(folder, d.path))
  );
  if (local.length > 0) {
    return local[0][0];
  }
  if (drives.length > 0) {
    return drives[0].path;
  }
  return folders[0]?.[0] ?? null;
};

/** Name of native OS file manager (Finder on macOS, File Explorer on Windows, Folder on Linux) */
export function fileManagerName(): string {
  if (typeof navigator !== "undefined") {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("mac")) return "Finder";
    if (ua.includes("win")) return "File Explorer";
  }
  return "Folder";
}

