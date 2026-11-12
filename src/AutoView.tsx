import { ask, open } from "@tauri-apps/plugin-dialog";
import { ReactNode, useEffect, useRef, useState } from "react";
import { api, AutoFunction, AutoResult, AutoSettings, JobProgress, LABEL_COLORS, Photo, Settings, thumbUrl } from "./api";
import { CompareGroup, CompareModal } from "./CompareModal";
import { Btn, HUE, Hue, Icon, Seg, Stars } from "./controls";

type Rec = Record<string, unknown>;
type When = (s: Rec) => boolean;
type Field =
  | { kind: "check"; k: string; label: string; hint?: string; basic?: boolean; when?: When }
  | { kind: "slider"; k: string; label: string; min: number; max: number; step?: number; digits?: number; unit?: string; hint?: string; basic?: boolean; when?: When }
  | { kind: "select"; k: string; label: string; options: [string, string][]; numeric?: boolean; hint?: string; basic?: boolean; when?: When }
  | { kind: "text"; k: string; label: string; placeholder?: string; folder?: boolean; hint?: string; basic?: boolean; when?: When }
  | { kind: "targets"; k: string; label: string; options: [string, string][]; hint?: string; basic?: boolean; when?: When }
  | { kind: "heading"; label: string; when?: When };

const PROFILES: [string, string][] = [
  ["type", "By photo type (AI)"],
  ["natural", "Natural"],
  ["neutral", "Neutral"],
  ["vivid", "Vivid"],
  ["portrait", "Portrait"],
  ["landscape", "Landscape"],
  ["bw", "B&W"],
  ["film", "Film"],
  ["moody", "Moody"],
  ["street", "Street"],
  ["food", "Food"],
  ["architecture", "Architecture"],
  ["night", "Night"],
  ["nature", "Nature & macro"],
  ["teal_orange", "Teal & Orange"],
  ["soft_editorial", "Soft editorial"],
  ["dark_moody", "Dark moody"],
  ["analog_film", "35mm film"],
  ["fine_art_bw", "Fine-art B&W"],
];
const DETAIL: [string, string][] = [
  ["fast", "Fast (1024 px)"],
  ["screen", "Screen size (2048 px)"],
  ["full", "100% (full size, slow)"],
];
const actions = (keyword: string): [string, string][] => [
  ["none", "Only list them"],
  ["reject", "Flag them as rejects"],
  ["label", "Label them red"],
  ["keyword", `Add the keyword “${keyword}”`],
];
const on = (k: string) => (s: Rec) => !!s[k];

/** The auto functions: what each does, its settings (basic ones show first, the rest under Advanced) and
 *  whether it asks the AI provider. `short`, `icon` (an Icon path) and `hue` show it in the Library's right bar. */
export const FUNCS: { id: AutoFunction; title: string; short: string; icon: string; hue: Hue; verb: string; blurb: string; fields: Field[]; ai?: When }[] = [
  {
    id: "cull",
    title: "Catalog & Culling",
    short: "Cull",
    hue: "amber",
    icon: "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z",
    verb: "Cull",
    blurb:
      "Scores each photo's sharpness (on its main subject), exposure and noise, groups bursts and picks the best of each, rejects blurred or badly exposed photos, rates the rest 1–5 stars and adds catalog keywords.",
    ai: on("ai_keywords"),
    fields: [
      { kind: "check", k: "reject_blurry", label: "Reject blurred photos", basic: true },
      { kind: "slider", k: "blur_threshold", label: "Blurred below", min: 0, max: 100, basic: true, when: on("reject_blurry"), hint: "Sharpness 0–100; 35 is about an edge blurred over 2 px at screen size" },
      { kind: "check", k: "pick_best", label: "Pick the best photo of each burst", basic: true },
      { kind: "check", k: "stars", label: "Rate 1–5 stars", basic: true },
      { kind: "check", k: "keywords", label: "Add catalog keywords", basic: true, hint: "Year, month, time of day, orientation, colours, black and white, high ISO, long exposure, burst, blurry" },
      { kind: "heading", label: "Sharpness" },
      { kind: "select", k: "detail", label: "Check sharpness at", options: DETAIL },
      { kind: "check", k: "use_subject", label: "Judge the main subject", hint: "Off: the photo's sharpest part counts" },
      { kind: "heading", label: "Exposure" },
      { kind: "check", k: "reject_exposure", label: "Reject blown-out or black photos" },
      { kind: "slider", k: "clip_limit", label: "Clipped area over", min: 1, max: 80, unit: "%", when: on("reject_exposure") },
      { kind: "heading", label: "Bursts" },
      { kind: "slider", k: "burst_seconds", label: "Taken within", min: 0.5, max: 10, step: 0.5, digits: 1, unit: " s" },
      { kind: "slider", k: "similarity", label: "Looking alike from", min: 0, max: 100, unit: "%" },
      { kind: "check", k: "reject_similar", label: "Reject the rest of a burst", hint: "Off: they get fewer stars than the best" },
      { kind: "heading", label: "Catalog" },
      { kind: "check", k: "labels", label: "Colour labels", hint: "Green picks, yellow borderline, red rejects" },
      { kind: "check", k: "ai_keywords", label: "Subject and scene keywords from the AI provider", hint: "One request per photo" },
      { kind: "check", k: "keep_manual", label: "Keep my ratings, flags and labels", hint: "Photos you rated, flagged or labelled are left alone" },
    ],
  },
  {
    id: "edit",
    title: "Non-destructive Editing",
    short: "Edit",
    hue: "blue",
    icon: "M4 7h9 M17 7h3 M15 5v4 M4 12h3 M11 12h9 M9 10v4 M4 17h11 M19 17h1 M17 15v4",
    verb: "Edit",
    blurb:
      "Sets white balance and tone from each photo, then a look; can also straighten, remove colour fringes and reduce noise. Each photo gets one history step you can undo; files are never changed.",
    ai: (s) => s.profile === "type" && !!s.ai_type,
    fields: [
      { kind: "select", k: "profile", label: "Look", options: PROFILES, basic: true, hint: "By photo type: landscape, portrait, street, food, architecture, night, nature or B&W, each with its own look" },
      { kind: "check", k: "ai_type", label: "Ask the AI provider what each photo shows", basic: true, when: (s) => s.profile === "type", hint: "One request per photo; off, its keywords and measurements decide" },
      { kind: "slider", k: "strength", label: "Strength", min: 0, max: 100, unit: "%", basic: true },
      { kind: "check", k: "skip_edited", label: "Skip photos I have edited", basic: true },
      { kind: "heading", label: "What changes" },
      { kind: "check", k: "white_balance", label: "White balance" },
      { kind: "check", k: "tone", label: "Tone: exposure, contrast, highlights, shadows, whites, blacks" },
      { kind: "check", k: "look", label: "The look's presence and colour" },
      {
        kind: "select",
        k: "upright",
        label: "Straighten",
        options: [
          ["off", "Off"],
          ["level", "Level the horizon"],
          ["auto", "Upright: balanced"],
          ["vertical", "Upright: verticals"],
          ["full", "Upright: full perspective"],
        ],
      },
      { kind: "check", k: "chromatic_aberration", label: "Remove colour fringes" },
      { kind: "check", k: "noise_reduction", label: "Noise reduction from the noise measured on the photo" },
      { kind: "heading", label: "Where" },
      { kind: "check", k: "virtual_copy", label: "Edit a new virtual copy", hint: "Each photo's own edits stay as they are" },
    ],
  },
  {
    id: "mask",
    title: "AI Masking",
    short: "Mask",
    hue: "violet",
    icon: "M4 8V5a1 1 0 0 1 1-1h3 M16 4h3a1 1 0 0 1 1 1v3 M20 16v3a1 1 0 0 1-1 1h-3 M8 20H5a1 1 0 0 1-1-1v-3 M12 7.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5z M7.5 17a4.5 4.5 0 0 1 9 0",
    verb: "Select",
    blurb: "Adds selections to tune in Develop: the main subject, the sky, the background, or each person. Each lands in the photo's Selections, ready for its own adjustments.",
    ai: (s) => s.method === "ai",
    fields: [
      {
        kind: "targets",
        k: "targets",
        label: "Select",
        basic: true,
        options: [
          ["subject", "Subject"],
          ["sky", "Sky"],
          ["background", "Background"],
          ["people", "People"],
        ],
      },
      {
        kind: "select",
        k: "method",
        label: "How",
        basic: true,
        options: [
          ["builtin", "Built-in estimate (fast, no AI model)"],
          ["sam", "Segment Anything 3, on this computer"],
          ["ai", "AI provider traces the outlines"],
        ],
        hint: "People need Segment Anything or the AI provider. Segment Anything needs a SAM 3 model (AI tab) and keeps holes (a tree in the sky). All snap the outline to the photo's edges.",
      },
      { kind: "heading", label: "Selections" },
      { kind: "slider", k: "adjust", label: "Starting adjustment", min: 0, max: 100, unit: "%", hint: "Subject brighter, sky deeper, background darker; 0 only selects" },
      { kind: "slider", k: "feather", label: "Feather", min: -100, max: 100 },
      { kind: "slider", k: "edge_shift", label: "Edge shift", min: -100, max: 100 },
      { kind: "check", k: "replace", label: "Replace selections of the same name" },
    ],
  },
  {
    id: "enhance",
    title: "AI Denoise & Super Resolution",
    short: "Enhance",
    hue: "emerald",
    icon: "M11 3l1.8 5.2L18 10l-5.2 1.8L11 17l-1.8-5.2L4 10l5.2-1.8z M18 14l.7 2.3L21 17l-2.3.7L18 20l-.7-2.3L15 17l2.3-.7z",
    verb: "Enhance",
    blurb: "Writes a cleaner, optionally larger copy as a new photo next to the original (…-Enhanced.tif) with the original's edits. The original stays untouched.",
    fields: [
      { kind: "check", k: "denoise", label: "Denoise", basic: true },
      { kind: "slider", k: "strength", label: "Strength", min: 0, max: 100, basic: true, when: on("denoise") },
      { kind: "check", k: "super_resolution", label: "Super resolution", basic: true },
      {
        kind: "select",
        k: "scale",
        label: "Enlarge",
        numeric: true,
        basic: true,
        when: on("super_resolution"),
        options: [
          ["1.5", "1.5 × each side"],
          ["2", "2 × (4 × the pixels)"],
          ["3", "3 ×"],
          ["4", "4 ×"],
        ],
      },
      { kind: "heading", label: "Denoise", when: on("denoise") },
      { kind: "slider", k: "detail", label: "Keep fine detail", min: 0, max: 100, when: on("denoise") },
      { kind: "slider", k: "color", label: "Colour noise", min: 0, max: 100, when: on("denoise") },
      { kind: "heading", label: "Super resolution", when: on("super_resolution") },
      { kind: "slider", k: "sharpen", label: "Sharpen", min: 0, max: 100, when: on("super_resolution") },
      { kind: "slider", k: "max_megapixels", label: "At most", min: 4, max: 200, unit: " MP", when: on("super_resolution"), hint: "Large photos are enlarged less to stay under this" },
      { kind: "heading", label: "Engine and file" },
      {
        kind: "select",
        k: "engine",
        label: "Engine",
        options: [
          ["builtin", "Built-in: wavelets and Lanczos (no AI model)"],
          ["command", "External AI engine (a command)"],
        ],
      },
      {
        kind: "text",
        k: "command",
        label: "Command",
        when: (s) => s.engine === "command",
        hint: "Run without a shell; {{INPUT}}, {{OUTPUT}}, {{SCALE}}, {{STRENGTH}} and {{NOISE}} are filled in. Real-ESRGAN: realesrgan-ncnn-vulkan -i {{INPUT}} -o {{OUTPUT}} -s {{SCALE}}",
      },
      {
        kind: "select",
        k: "format",
        label: "File",
        options: [
          ["auto", "Auto: 32-bit float for RAW, else 16-bit"],
          ["float", "32-bit float TIFF (linear light)"],
          ["tiff16", "16-bit TIFF"],
        ],
      },
      { kind: "text", k: "folder", label: "Folder", folder: true, placeholder: "Next to the original", hint: "Photos from Immich or the cloud go to Pictures/Optixel Enhanced" },
      { kind: "check", k: "copy_edits", label: "Copy the edits, rating and keywords" },
    ],
  },
  {
    id: "hdr",
    title: "HDR Merge",
    short: "HDR",
    hue: "orange",
    icon: "M12 4a8 8 0 1 0 0 16z M12 4a8 8 0 0 1 0 16 M3.5 12h5 M15.5 12h5",
    verb: "Merge",
    blurb:
      "Merges each exposure bracket (photos taken a few seconds apart at different exposures) into one high dynamic range photo: lined up, ghosts of moving things removed, written as a new 32-bit float photo (…-HDR.tif). The originals stay untouched.",
    fields: [
      {
        kind: "select",
        k: "grouping",
        label: "Merge",
        basic: true,
        options: [
          ["brackets", "Each bracket among the photos"],
          ["all", "All the photos as one bracket"],
        ],
      },
      { kind: "slider", k: "bracket_seconds", label: "A bracket is taken within", min: 1, max: 30, unit: " s", basic: true, when: (s) => s.grouping === "brackets", hint: "Between one shot and the next" },
      {
        kind: "select",
        k: "deghost",
        label: "Deghost",
        basic: true,
        options: [
          ["off", "Off"],
          ["low", "Low"],
          ["medium", "Medium"],
          ["high", "High"],
        ],
        hint: "What moved between shots (people, leaves, water) comes from the middle exposure alone",
      },
      { kind: "heading", label: "Merging" },
      { kind: "check", k: "align", label: "Line up handheld shots" },
      { kind: "check", k: "auto_tone", label: "Start with auto white balance, tone and HDR toning", hint: "HDR toning (Merge exposures) fits the merged range to the screen; tune it in Develop › Pro" },
      { kind: "text", k: "folder", label: "Folder", folder: true, placeholder: "Next to the photos", hint: "Photos from Immich or the cloud go to Pictures/Optixel Enhanced" },
    ],
  },
  {
    id: "blur",
    title: "Find blurred subjects",
    short: "Blur",
    hue: "rose",
    icon: "M12 7a5 5 0 1 1 0 10 5 5 0 0 1 0-10z M12 2.5v2 M12 19.5v2 M2.5 12h2 M19.5 12h2",
    verb: "Check",
    blurb: "Finds photos whose main subject is out of focus or shaken, including missed focus, where something else in the frame is sharp.",
    ai: (s) => s.subject === "ai",
    fields: [
      { kind: "slider", k: "threshold", label: "Blurred below", min: 0, max: 100, basic: true, hint: "Subject sharpness 0–100" },
      { kind: "select", k: "action", label: "Then", options: actions("blurry"), basic: true },
      { kind: "heading", label: "Finding" },
      { kind: "select", k: "detail", label: "Check sharpness at", options: DETAIL },
      {
        kind: "select",
        k: "subject",
        label: "Find the subject with",
        options: [
          ["builtin", "Built-in estimate (fast)"],
          ["ai", "AI provider (slower, more accurate)"],
        ],
      },
      { kind: "slider", k: "motion", label: "Motion blur from", min: 0, max: 100, unit: "%", hint: "Blur in one direction: camera shake or a moving subject" },
      { kind: "check", k: "missed_focus", label: "Tell missed focus apart" },
    ],
  },
  {
    id: "duplicates",
    title: "Find duplicates",
    short: "Duplicates",
    hue: "cyan",
    icon: "M9 8h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z M4 16V5a1 1 0 0 1 1-1h11",
    verb: "Compare",
    blurb: "Finds copies of the same file across folders and sources, and photos that look the same; the best of each group is kept.",
    fields: [
      {
        kind: "select",
        k: "mode",
        label: "Find",
        basic: true,
        options: [
          ["both", "Exact copies and look-alikes"],
          ["exact", "Exact copies (same file contents)"],
          ["similar", "Look-alikes"],
        ],
      },
      { kind: "select", k: "action", label: "Then, for the extra copies", options: actions("duplicate"), basic: true },
      { kind: "heading", label: "Comparing" },
      { kind: "slider", k: "similarity", label: "Looking alike from", min: 50, max: 100, unit: "%", when: (s) => s.mode !== "exact" },
      { kind: "slider", k: "window_minutes", label: "Taken within", min: 0, max: 240, unit: " min", when: (s) => s.mode !== "exact", hint: "0: compare any two photos" },
      {
        kind: "select",
        k: "keep",
        label: "Keep",
        options: [
          ["best", "The best: picks, stars, edits, then size"],
          ["largest", "The largest"],
          ["sharpest", "The sharpest"],
          ["oldest", "The oldest"],
          ["newest", "The newest"],
        ],
      },
    ],
  },
];

const fieldCls = "rounded bg-neutral-800 px-2 py-1 outline-none focus:ring-1 focus:ring-sky-600";
const num = (n: unknown) => Number(n ?? 0).toLocaleString();

function FieldRow({ f, s, set }: { f: Field; s: Rec; set: (patch: Rec) => void }) {
  if (f.kind === "heading") return <div className="mb-0.5 mt-3 text-[10px] font-semibold uppercase tracking-wider text-neutral-500">{f.label}</div>;
  const v = s[f.k];
  const hint = "hint" in f && f.hint ? <div className="text-[10px] text-neutral-500">{f.hint}</div> : null;
  if (f.kind === "check")
    return (
      <label className="flex items-start gap-2 py-1">
        <input type="checkbox" className="mt-0.5" checked={!!v} onChange={(e) => set({ [f.k]: e.target.checked })} />
        <span>
          <span className="text-neutral-200">{f.label}</span>
          {hint}
        </span>
      </label>
    );
  // the label sits above its control (and a slider's value beside the label) to fit the narrow panel
  const row = (control: ReactNode, value?: ReactNode) => (
    <div className="py-1">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-neutral-400">{f.label}</span>
        {value}
      </div>
      {control}
      {hint}
    </div>
  );
  if (f.kind === "slider") {
    const n = Number(v ?? 0);
    return row(
      <input type="range" className="ol-range w-full" min={f.min} max={f.max} step={f.step ?? 1} value={n} onChange={(e) => set({ [f.k]: +e.target.value })} />,
      <span className="tabular-nums text-neutral-200">
        {n.toFixed(f.digits ?? 0)}
        {f.unit ?? ""}
      </span>,
    );
  }
  if (f.kind === "select")
    return row(
      <select className={`${fieldCls} w-full`} value={String(v)} onChange={(e) => set({ [f.k]: f.numeric ? +e.target.value : e.target.value })}>
        {f.options.map(([k, l]) => (
          <option key={k} value={k}>
            {l}
          </option>
        ))}
      </select>,
    );
  if (f.kind === "targets") {
    const list = (v as string[]) ?? [];
    return row(
      <div className="flex flex-wrap gap-1">
        {f.options.map(([k, l]) => (
          <Btn key={k} active={list.includes(k)} onClick={() => set({ [f.k]: list.includes(k) ? list.filter((x) => x !== k) : [...list, k] })}>
            {l}
          </Btn>
        ))}
      </div>,
    );
  }
  return row(
    <div className="flex gap-1">
      <input className={`${fieldCls} min-w-0 flex-1 font-mono text-[11px]`} value={String(v ?? "")} placeholder={f.placeholder} onChange={(e) => set({ [f.k]: e.target.value })} />
      {f.folder && (
        <Btn
          onClick={async () => {
            const d = await open({ directory: true, title: f.label });
            if (typeof d === "string") set({ [f.k]: d });
          }}
        >
          Choose…
        </Btn>
      )}
    </div>,
  );
}

function Thumb({ photo, ring }: { photo?: Photo; ring?: string }) {
  return (
    <div className={`flex h-12 w-16 shrink-0 items-center justify-center rounded bg-neutral-950 ${ring ?? ""}`}>
      {photo && <img src={thumbUrl(photo)} loading="lazy" className="max-h-full max-w-full object-contain" alt="" />}
    </div>
  );
}

/** One line per function: what the last run did. */
function summary(r: AutoResult): string {
  const n = (k: string) => num(r[k]);
  switch (r.function) {
    case "cull":
      return `${n("picks")} picks · ${n("rejects")} rejects · ${n("rated")} rated · ${n("bursts")} bursts${Number(r.kept) ? ` · ${n("kept")} kept as you set them` : ""}`;
    case "edit":
      return `${n("edited")} of ${n("total")} edited`;
    case "mask":
      return `${n("masks")} selections on ${n("total")} photos`;
    case "enhance":
      return `${n("made")} enhanced copies of ${n("total")}`;
    case "hdr":
      return `${n("made")} HDR photos from ${n("brackets")} brackets of ${n("total")} photos`;
    case "blur":
      return `${n("found")} of ${n("total")} have a blurred subject`;
    case "duplicates":
      return `${n("groups")} groups · ${n("extra")} extra copies`;
  }
}

/** The photos the quick actions act on: for duplicates the extra copies, not the ones kept. */
function extraIds(r: AutoResult): number[] {
  if (r.function !== "duplicates") return resultIds(r);
  return (r.rows as Rec[]).flatMap((g) => (g.ids as number[]).filter((id) => id !== g.keep));
}

/** The photos a result is about, for Show in Library. */
function resultIds(r: AutoResult): number[] {
  const rows = r.rows as Rec[];
  switch (r.function) {
    case "blur":
      return rows.filter((x) => x.found).map((x) => x.id as number);
    case "duplicates":
      return rows.flatMap((x) => x.ids as number[]);
    case "enhance":
    case "hdr":
      return rows.flatMap((x) => (x.new_id ? [x.new_id as number] : []));
    default:
      return rows.map((x) => x.id as number);
  }
}

export type AutoViewProps = {
  settings: Settings | null;
  setSettings: (s: Settings) => void;
  selected: number[];
  view: Photo[];
  jobs: Record<string, JobProgress>;
  onShow: (ids: number[], label: string) => void;
  onAiSettings: () => void;
  onStorage: () => void;
  fn: AutoFunction;
};

/** One auto function in the Library's right panel, next to the grid: its settings (the advanced ones folded
 *  away), run on the selected photos, the photos in view or the whole catalog, and what the last run found.
 *  The Library keeps it mounted while hidden, so runs and results survive switching functions. */
export function AutoView(p: AutoViewProps) {
  const fn = p.fn;
  const [scope, setScope] = useState<"selected" | "view" | "all">(p.selected.length ? "selected" : "view");
  useEffect(() => {
    if (scope === "selected" && p.selected.length === 0) {
      setScope("view");
    }
  }, [p.selected.length, scope]);
  const [total, setTotal] = useState<number | null>(null);
  const [running, setRunning] = useState<Partial<Record<AutoFunction, string>>>({});
  const [results, setResults] = useState<Partial<Record<AutoFunction, AutoResult>>>({});
  const [errors, setErrors] = useState<Partial<Record<AutoFunction, string>>>({});
  const [photos, setPhotos] = useState<Record<number, Photo>>({});
  const [defaults, setDefaults] = useState<AutoSettings | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [filter, setFilter] = useState("all");
  const saveTimer = useRef(0);
  const latest = useRef(p.settings);
  latest.current = p.settings;
  useEffect(() => void api.autoDefaults().then(setDefaults), []);
  useEffect(() => void api.query({}).then((ps) => setTotal(ps.length)), [results]);
  useEffect(() => setFilter(fn === "blur" ? "found" : "all"), [fn]);
  const resultRef = useRef<HTMLDivElement>(null);
  if (!p.settings) return null;

  const def = FUNCS.find((f) => f.id === fn)!;
  const hue = HUE[def.hue];
  const s = p.settings.auto[fn] as unknown as Rec;
  const save = (next: Settings) => {
    p.setSettings(next);
    clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => api.saveSettings(next), 400);
  };
  const set = (patch: Rec) => {
    const cur = latest.current!;
    save({ ...cur, auto: { ...cur.auto, [fn]: { ...(cur.auto[fn] as object), ...patch } } });
  };
  const visible = (f: Field) => !f.when || f.when(s);
  const basic = def.fields.filter((f) => "basic" in f && f.basic && visible(f));
  const more = def.fields.filter((f) => !("basic" in f && f.basic) && visible(f));
  const changed = defaults ? more.filter((f) => f.kind !== "heading" && JSON.stringify(s[f.k]) !== JSON.stringify((defaults[fn] as unknown as Rec)[f.k])).length : 0;
  const provider = p.settings.providers.find((x) => x.id === p.settings!.active_provider);
  const counts = { selected: p.selected.length, view: p.view.length, all: total ?? 0 };

  const run = async (f: AutoFunction) => {
    const settings = latest.current!.auto[f];
    setErrors((e) => ({ ...e, [f]: undefined }));
    const ids = scope === "selected" ? p.selected : scope === "view" ? p.view.map((x) => x.id) : (await api.query({})).map((x) => x.id);
    if (!ids.length) return setErrors((e) => ({ ...e, [f]: scope === "selected" ? "Select photos in the Library first." : "There are no photos here." }));
    const job = `${f}-${Date.now()}`;
    setRunning((r) => ({ ...r, [f]: job }));
    try {
      const r = await api.autoRun(f, ids, settings, job);
      setResults((x) => ({ ...x, [f]: r }));
      setFilter(f === "blur" ? "found" : "all");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
      const need = [...new Set([...resultIds(r), ...(r.rows as Rec[]).map((x) => x.id as number)])].filter((id) => id != null);
      const got = await api.query({ ids: need });
      setPhotos((cur) => ({ ...cur, ...Object.fromEntries(got.map((ph) => [ph.id, ph])) }));
    } catch (e) {
      setErrors((x) => ({ ...x, [f]: String(e) }));
    } finally {
      setRunning(({ [f]: _, ...rest }) => rest);
    }
  };
  const job = running[fn];
  const progress = job ? p.jobs[job] : undefined;
  const result = results[fn];
  const ai = def.ai?.(s);

  return (
    <div className="flex flex-col gap-3 p-3 text-[11px]">
      {/* the function's colour, as on its button in the right bar */}
      <header className={`-mx-3 -mt-3 border-t-2 px-3 pt-3 ${hue.border}`}>
        <h2 className="flex items-center gap-2 text-[12px] font-semibold text-neutral-100">
          <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${hue.soft} ${hue.text}`}>
            <Icon d={def.icon} className="h-4 w-4" />
          </span>
          {def.title}
        </h2>
        <p className="mt-1.5 text-neutral-400">{def.blurb}</p>
      </header>

      <div>
        <div className="mb-1 text-neutral-400">Photos</div>
        <Seg
          options={[
            ["selected", `Selected ${num(counts.selected)}`],
            ["view", `In view ${num(counts.view)}`],
            ["all", `All ${total == null ? "…" : num(total)}`],
          ]}
          value={scope}
          onChange={setScope}
        />
      </div>

      <div>
        {basic.map((f, i) => (
          <FieldRow key={"k" in f ? f.k : i} f={f} s={s} set={set} />
        ))}
        <div className="mt-2 flex items-center justify-between border-t border-neutral-800 pt-2">
          <button className="font-medium text-sky-400 hover:text-sky-300" onClick={() => setAdvanced(!advanced)}>
            {advanced ? "▾" : "▸"} Advanced settings{changed ? ` (${changed} changed)` : ""}
          </button>
          {defaults && (
            <button className="text-[10px] text-neutral-500 hover:text-neutral-200" onClick={() => set(defaults[fn] as unknown as Rec)} title="Every setting of this function back to its default">
              Reset to defaults
            </button>
          )}
        </div>
        {advanced && (
          <div className="mt-1">
            {more.map((f, i) => (
              <FieldRow key={"k" in f ? f.k : `h${i}`} f={f} s={s} set={set} />
            ))}
          </div>
        )}
      </div>

      {ai && (
        <p className="rounded bg-violet-950/40 px-2 py-1.5 text-violet-200">
          ✦ Uses the AI provider <b>{provider?.name ?? p.settings.active_provider}</b>, one request per photo.{" "}
          <button className="text-sky-400 hover:text-sky-300" onClick={p.onAiSettings}>
            Change…
          </button>
        </p>
      )}

      <Btn active className="w-full" disabled={!!job || (scope === "selected" && !p.selected.length)} onClick={() => run(fn)}>
        {job ? "Working…" : `${def.verb} ${scope === "selected" ? num(counts.selected) : scope === "view" ? num(counts.view) : "all"} photos`}
      </Btn>
      {job && (
        <div className="flex items-center gap-2">
          <div className="min-w-0 flex-1">
            <div className="mb-0.5 flex justify-between text-[10px] text-neutral-400">
              <span className="truncate">{progress?.label ?? "Starting"}…</span>
              {progress && progress.total > 0 && (
                <span className="tabular-nums">
                  {num(progress.done)} / {num(progress.total)}
                </span>
              )}
            </div>
            <div className="relative h-1 overflow-hidden rounded bg-neutral-800">
              {progress && progress.total > 0 ? (
                <div className={`h-full transition-all ${hue.bg}`} style={{ width: `${(progress.done / progress.total) * 100}%` }} />
              ) : (
                <div className={`animate-indeterminate absolute inset-y-0 w-1/3 rounded ${hue.bg}`} />
              )}
            </div>
          </div>
          <Btn onClick={() => api.cancelJob(job)} title="Stop after the current photo; what is done stays">
            Cancel
          </Btn>
        </div>
      )}
      {errors[fn] && <p className="break-words text-red-400">{errors[fn]}</p>}

      <div ref={resultRef}>{result && <Results r={result} photos={photos} filter={filter} setFilter={setFilter} onShow={p.onShow} title={def.title} />}</div>

      {fn === "cull" && (
        <button onClick={p.onStorage} className="text-left text-[10px] text-neutral-500 hover:text-neutral-200">
          <span className={p.settings.auto.on_import.enabled ? "text-emerald-400" : "text-neutral-500"}>●</span> Culling new photos as they arrive:{" "}
          {p.settings.auto.on_import.enabled ? "on" : "off"} · Settings › Locations
        </button>
      )}
    </div>
  );
}

function Results({ r, photos, filter, setFilter, onShow, title }: { r: AutoResult; photos: Record<number, Photo>; filter: string; setFilter: (f: string) => void; onShow: (ids: number[], label: string) => void; title: string }) {
  const rows = r.rows as Rec[];
  const ids = resultIds(r);
  const extra = extraIds(r);
  const [acted, setActed] = useState("");
  const [compareGroupIdx, setCompareGroupIdx] = useState<number | null>(null);

  const duplicateGroups: CompareGroup[] =
    r.function === "duplicates"
      ? rows.map((g, idx) => ({
          id: idx,
          title: g.kind === "exact" ? "Exact Duplicate" : "Look-alike Group",
          kind: g.kind as "exact" | "similar",
          photos: (g.ids as number[]).map((id) => photos[id]).filter(Boolean),
          keepId: g.keep as number,
        }))
      : [];

  const act = async (label: string, u: Parameters<typeof api.update>[1]) => {
    try {
      await api.update(extra, u);
      setActed(`${label}: ${num(extra.length)} photos`);
    } catch (e) {
      setActed(String(e));
    }
  };
  const filters: [string, string][] | null =
    r.function === "cull"
      ? [
          ["all", "All"],
          ["pick", "Picks"],
          ["reject", "Rejects"],
        ]
      : r.function === "blur"
        ? [
            ["found", "Blurred"],
            ["all", "All checked"],
          ]
        : null;
  const shown = rows.filter((x) => {
    if (r.function === "cull" && filter !== "all") return x.flag === filter;
    if (r.function === "blur" && filter !== "all") return !!x.found;
    return true;
  });
  const name = (x: Rec) => <span className="truncate text-neutral-200">{String(x.file ?? "")}</span>;
  return (
    <section className="flex flex-col gap-2 border-t border-neutral-800 pt-3">
      <div className="flex items-baseline gap-2">
        <h3 className="font-semibold text-neutral-100">Result</h3>
        <span className="text-neutral-400">
          {summary(r)}
          {r.cancelled ? " (cancelled)" : ""}
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {r.function === "duplicates" && rows.length > 0 && (
          <Btn active onClick={() => setCompareGroupIdx(0)} title="Open full-screen side-by-side comparison to inspect and decide">
            ⧉ Compare {rows.length} groups
          </Btn>
        )}
        {(r.function === "blur" || r.function === "duplicates") && extra.length > 0 && (
          <>
            <Btn onClick={() => act("Rejected", { flag: "reject" })} title={r.function === "duplicates" ? "Flag the extra copies as rejects; the ones kept stay as they are" : "Flag them as rejects"}>
              Reject {r.function === "duplicates" ? "extra copies" : ""}
            </Btn>
            <Btn onClick={() => act("Labelled red", { color_label: "red" })} title={r.function === "duplicates" ? "Label the extra copies red" : "Label them red"}>
              Label red
            </Btn>
          </>
        )}
        <Btn disabled={!ids.length} onClick={() => onShow(ids, title)} title="Show these photos in the grid">
          Show in Library
        </Btn>
      </div>
      {acted && <p className="text-emerald-400">{acted}</p>}
      {filters && <Seg options={filters} value={filters.some(([k]) => k === filter) ? filter : filters[0][0]} onChange={setFilter} />}
      {Array.isArray(r.notes) && (r.notes as string[]).map((n) => <p key={n} className="text-amber-300">{n}</p>)}
      <div className="flex flex-col gap-1">
        {r.function === "duplicates"
          ? rows.map((g, i) => (
              <div key={i} className="rounded bg-neutral-950 p-2">
                <div className="mb-1.5 flex items-center justify-between">
                  <span className={`text-[10px] uppercase font-bold ${g.kind === "exact" ? "text-amber-300" : "text-sky-300"}`}>
                    {g.kind === "exact" ? "exact copies" : "look alike"}
                  </span>
                  <button
                    onClick={() => setCompareGroupIdx(i)}
                    className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-sky-600 hover:text-white px-2 py-0.5 text-[10px] text-sky-300 border border-neutral-700/80 transition-colors cursor-pointer"
                    title="Open side by side comparison for this group"
                  >
                    <span>⧉</span>
                    <span>Compare</span>
                  </button>
                </div>
                <div className="flex gap-2 overflow-x-auto">
                  {(g.ids as number[]).map((id, k) => (
                    <div
                      key={id}
                      onClick={() => setCompareGroupIdx(i)}
                      className="flex w-16 shrink-0 flex-col items-center gap-0.5 cursor-pointer group"
                      title={`Click to compare: ${(g.files as string[])[k]}`}
                    >
                      <Thumb photo={photos[id]} ring={id === g.keep ? "ring-2 ring-emerald-500 shadow-xs" : "group-hover:ring-1 group-hover:ring-neutral-500"} />
                      <span className={`w-full truncate text-center text-[10px] ${id === g.keep ? "text-emerald-400 font-semibold" : "text-neutral-400"}`}>
                        {id === g.keep ? "✓ keep" : (g.files as string[])[k]}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          : shown.map((x) => (
              <div key={String(x.id)} className="flex items-center gap-3 rounded bg-neutral-950 px-2 py-1">
                <Thumb photo={photos[x.id as number]} />
                <div className="flex min-w-0 flex-1 flex-col">
                  {name(x)}
                  <Detail r={r} x={x} />
                </div>
                {(r.function === "enhance" || r.function === "hdr") && x.new_id != null && <Thumb photo={photos[x.new_id as number]} ring="ring-1 ring-emerald-600" />}
              </div>
            ))}
        {r.function !== "duplicates" && !shown.length && <p className="text-neutral-500">Nothing to show.</p>}
        {r.function === "duplicates" && !rows.length && <p className="text-neutral-500">No duplicates found.</p>}
      </div>

      {compareGroupIdx !== null && duplicateGroups.length > 0 && (
        <CompareModal
          groups={duplicateGroups}
          initialIndex={compareGroupIdx}
          onClose={() => setCompareGroupIdx(null)}
          onSetKeep={(gIdx, photoId) => {
            if (rows[gIdx]) {
              rows[gIdx].keep = photoId;
              setActed(`Group ${gIdx + 1}: selected photo ${photoId} to keep`);
            }
          }}
          onCull={async (photoId, culling) => {
            try {
              await api.update([photoId], culling);
              if (photos[photoId]) {
                Object.assign(photos[photoId], culling);
              }
            } catch (err) {
              console.error(err);
            }
          }}
          onMoveToBin={async (photoId) => {
            if (!(await ask("Move this photo to the Bin?\n\nYou can restore it from the Bin later.", { title: "Move to Bin", kind: "warning" }))) return;
            try {
              await api.moveToBin([photoId]);
            } catch (err) {
              console.error(err);
            }
          }}
        />
      )}
    </section>
  );
}

/** A result row's details. */
function Detail({ r, x }: { r: AutoResult; x: Rec }) {
  const small = "text-[10px] text-neutral-500";
  if (x.error) return <span className="text-[10px] text-red-400">{String(x.error)}</span>;
  switch (r.function) {
    case "cull":
      return (
        <span className={`flex flex-wrap items-center gap-2 ${small}`}>
          {x.flag === "pick" && <span className="text-neutral-100">⚑ Pick</span>}
          {x.flag === "reject" && <span className="text-red-400">✕ Reject{(x.reasons as string[]).length ? `: ${(x.reasons as string[]).join(", ")}` : ""}</span>}
          {x.rating != null && <Stars value={x.rating as number} />}
          {x.label != null && x.label !== "none" && <span className="h-2 w-2 rounded-full" style={{ background: LABEL_COLORS[x.label as string] }} />}
          <span>
            quality {String(x.quality)} · sharp {String(x.subject_sharpness)} · exposure {String(x.exposure)}
          </span>
          {x.burst != null && <span>burst of {String(x.burst)}{x.best ? " (best)" : ""}</span>}
          {x.kept === true && <span className="text-amber-300">kept as you set it</span>}
        </span>
      );
    case "blur":
      return (
        <span className={small}>
          <span className={x.found ? "text-amber-300" : ""}>{String(x.reason || "sharp")}</span> · subject {String(x.subject_sharpness)} · photo {String(x.sharpness)}
          {Number(x.motion) >= 30 ? ` · motion ${String(x.motion)}%` : ""}
        </span>
      );
    case "edit":
      return (
        <span className={small}>
          {String(x.status)}
          {x.note ? ` · ${String(x.note)}` : ""}
        </span>
      );
    case "mask":
      return (
        <span className={small}>
          {(x.made as string[]).length ? `selected ${(x.made as string[]).join(", ")}` : "nothing selected"}
          {(x.missed as string[]).length ? ` · not found: ${(x.missed as string[]).join(", ")}` : ""}
        </span>
      );
    case "enhance":
      return (
        <span className={`truncate ${small}`} title={String(x.output ?? "")}>
          {String(x.size ?? "")} · {String(x.output ?? "")}
        </span>
      );
    case "hdr":
      return (
        <span className={`truncate ${small}`} title={String(x.output ?? "")}>
          {String(x.frames)} exposures ({String(x.evs)}){Number(x.aligned_px) ? ` · lined up by ${String(x.aligned_px)} px` : ""} · {String(x.output ?? "")}
        </span>
      );
    default:
      return null;
  }
}
