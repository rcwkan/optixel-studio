import { open } from "@tauri-apps/plugin-dialog";
import { Fragment, ReactNode, useEffect, useState } from "react";
import { api, Develop as Dev, IccProfileEntry, Proof, RawInfo, setPath } from "./api";
import { Btn, Panel, Seg, Slider } from "./controls";

type SliderFor = (label: string, path: (string | number)[], min: number, max: number, extra?: Partial<Parameters<typeof Slider>[0]>) => ReactNode;

const UPRIGHT: [string, string, string][] = [
  ["off", "Off", "No Upright correction"],
  ["auto", "Auto", "Balanced level, vertical and horizontal correction"],
  ["level", "Level", "Level the horizon only"],
  ["vertical", "Vertical", "Level and make vertical lines parallel"],
  ["full", "Full", "Level, vertical and horizontal perspective"],
];
const sub = "mb-0.5 mt-2.5 flex items-center justify-between text-[10px] uppercase tracking-wider text-neutral-500";
const hint = "mt-1 text-[10px] text-neutral-500";

/** Lightroom's advanced Develop panels, always for the whole photo: lens corrections, Transform with
 *  Upright, calibration, noise reduction, grain, spot removal and soft proofing. Each is its own panel
 *  so the Develop tabs can group them by kind (Geometry, Color, Detail). */
export type WholeProps = {
  dev: Dev;
  defaults: Dev;
  change: (d: Dev, label: string, replacePrefix?: string) => void;
  /** sliders bound to the whole photo */
  S: SliderFor;
};

const resetBtn = (p: WholeProps, key: keyof Dev, label: string) => (
  <button className="text-[10px] font-normal normal-case text-neutral-500 hover:text-neutral-200" onClick={() => p.change({ ...p.dev, [key]: structuredClone(p.defaults[key]) }, `Reset ${label}`)}>
    reset
  </button>
);
const check = (label: string, checked: boolean, onChange: (v: boolean) => void, title?: string) => (
  <label className="mt-1.5 flex items-center gap-1.5 text-[11px] text-neutral-300" title={title}>
    <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    {label}
  </label>
);

export function LensPanel(p: WholeProps & { autoCa: () => void }) {
  const { S } = p;
  return (
    <Panel title="Lens corrections" open={false} right={resetBtn(p, "lens", "lens corrections")}>
      {S("Distortion", ["lens", "distortion"], -100, 100)}
      <div className={sub}>
        Chromatic aberration
        <button className="normal-case text-sky-400 hover:text-sky-300" onClick={p.autoCa} title="Measure the colour fringes along edges and line red and blue up with green">
          auto
        </button>
      </div>
      {S("Red / cyan", ["lens", "ca_red_cyan"], -100, 100)}
      {S("Blue / yellow", ["lens", "ca_blue_yellow"], -100, 100)}
      <div className={sub}>Defringe</div>
      {S("Purple amount", ["lens", "defringe_purple"], 0, 20)}
      {S("Purple from", ["lens", "defringe_purple_hue", 0], 0, 100, { track: "track-fringe-purple" })}
      {S("Purple to", ["lens", "defringe_purple_hue", 1], 0, 100, { track: "track-fringe-purple" })}
      {S("Green amount", ["lens", "defringe_green"], 0, 20)}
      {S("Green from", ["lens", "defringe_green_hue", 0], 0, 100, { track: "track-fringe-green" })}
      {S("Green to", ["lens", "defringe_green_hue", 1], 0, 100, { track: "track-fringe-green" })}
      <div className={sub}>Lens vignetting</div>
      {S("Amount", ["lens", "vignetting"], -100, 100)}
      {S("Midpoint", ["lens", "vignetting_midpoint"], 0, 100)}
    </Panel>
  );
}

export function TransformPanel(p: WholeProps & { tool: string; setTool: (t: "none" | "guides") => void; upright: (mode: string) => void }) {
  const { dev, change, S } = p;
  const t = dev.transform;
  const clearGuides = () =>
    change({ ...dev, transform: { ...t, guides: [], ...(t.upright === "guided" && { upright: "off", vertical: 0, horizontal: 0, rotate: 0 }) } }, "Clear guides");
  return (
    <Panel title="Transform (Upright)" right={resetBtn(p, "transform", "transform")}>
      <div className="mb-1 grid grid-cols-3 gap-1">
        {UPRIGHT.map(([m, label, title]) => (
          <Btn key={m} active={t.upright === m && p.tool !== "guides"} title={title} onClick={() => p.upright(m)}>
            {label}
          </Btn>
        ))}
        <Btn active={p.tool === "guides" || t.upright === "guided"} title="Draw 2–4 guides along lines that should be vertical or horizontal" onClick={() => p.setTool(p.tool === "guides" ? "none" : "guides")}>
          Guided
        </Btn>
      </div>
      {(p.tool === "guides" || t.upright === "guided") && (
        <p className={hint}>
          {p.tool === "guides" ? `Draw guides on the photo (${t.guides.length}/4); press Guided again to see the result.` : `Corrected along ${t.guides.length} guides.`}{" "}
          {t.guides.length > 0 && (
            <button className="text-sky-400 hover:text-sky-300" onClick={clearGuides}>
              clear
            </button>
          )}
        </p>
      )}
      {S("Vertical", ["transform", "vertical"], -100, 100)}
      {S("Horizontal", ["transform", "horizontal"], -100, 100)}
      {S("Rotate", ["transform", "rotate"], -10, 10, { step: 0.1, digits: 1 })}
      {S("Aspect", ["transform", "aspect"], -100, 100)}
      {S("Scale", ["transform", "scale"], 50, 150)}
      {S("X offset", ["transform", "offset_x"], -100, 100, { step: 0.1, digits: 1 })}
      {S("Y offset", ["transform", "offset_y"], -100, 100, { step: 0.1, digits: 1 })}
      {check("Constrain crop", t.constrain_crop, (v) => change(setPath(dev, ["transform", "constrain_crop"], v), "Constrain crop"), "Shrink the crop until no empty corners show")}
    </Panel>
  );
}

export function CalibrationPanel(p: WholeProps) {
  const { S } = p;
  return (
    <Panel title="Calibration" open={false} right={resetBtn(p, "calibration", "calibration")}>
      {S("Shadows tint", ["calibration", "shadows_tint"], -100, 100, { track: "track-tint" })}
      {(["red", "green", "blue"] as const).map((c) => (
        <Fragment key={c}>
          <div className={sub}>{c} primary</div>
          {S("Hue", ["calibration", `${c}_hue`], -100, 100, { track: `track-${c}-hue` })}
          {S("Saturation", ["calibration", `${c}_saturation`], -100, 100)}
        </Fragment>
      ))}
    </Panel>
  );
}

/** a creative .cube 3D LUT look, last in the pipeline (on the GPU, with the develop stack, when it can be) */
export function LookPanel(p: WholeProps) {
  const { dev, change, S } = p;
  const [luts, setLuts] = useState<string[]>([]);
  const [err, setErr] = useState("");
  useEffect(() => {
    api.lutList().then(setLuts).catch(() => {});
  }, []);
  const pick = (lut: string) => change({ ...dev, look: { ...dev.look, lut } }, lut ? `Look: ${lut}` : "Remove look");
  const importLut = async () => {
    setErr("");
    const f = await open({ multiple: false, title: "Load a 3D LUT look", filters: [{ name: "3D LUT", extensions: ["cube", "CUBE"] }] });
    if (typeof f !== "string") return;
    try {
      const name = await api.lutImport(f);
      setLuts(await api.lutList());
      pick(name);
    } catch (e) {
      setErr(String(e));
    }
  };
  return (
    <Panel title="Look (3D LUT)" open={!!dev.look.lut} right={resetBtn(p, "look", "look")}>
      <div className="flex gap-1">
        <select className="min-w-0 flex-1 rounded bg-neutral-800 px-1 py-0.5 text-[11px]" value={dev.look.lut} onChange={(e) => pick(e.target.value)}>
          <option value="">None</option>
          {luts.map((l) => (
            <option key={l} value={l}>
              {l.replace(/\.cube$/i, "")}
            </option>
          ))}
        </select>
        <Btn onClick={importLut} title="Import a .cube 3D LUT made for sRGB / Rec.709 display values">
          Import…
        </Btn>
      </div>
      {dev.look.lut && S("Amount", ["look", "amount"], 0, 100)}
      {err && <p className="text-[10px] text-rose-400">{err}</p>}
    </Panel>
  );
}

export function GrainPanel(p: WholeProps) {
  const { S } = p;
  return (
    <>
      <div className={sub}>
        Grain
        {resetBtn(p, "grain", "grain")}
      </div>
      {S("Amount", ["grain", "amount"], 0, 100)}
      {S("Size", ["grain", "size"], 0, 100)}
      {S("Roughness", ["grain", "roughness"], 0, 100)}
    </>
  );
}

/** heal spots by clicking, and the list of every removal (spots and the AI bar's) */
export function SpotPanel(p: WholeProps & { tool: string; setTool: (t: "none" | "spot") => void; proof: Proof; setProof: (p: Proof) => void; spotSize: number; setSpotSize: (v: number) => void }) {
  const { dev, change } = p;
  return (
    <Panel title="Spot removal" open={p.tool === "spot" || dev.retouch.length > 0} right={resetBtn(p, "retouch", "removals")}>
      <Btn active={p.tool === "spot"} className="w-full" onClick={() => p.setTool(p.tool === "spot" ? "none" : "spot")} title="Click spots on the photo to heal them from a matching nearby area (Q)">
        {p.tool === "spot" ? "⊙ Click spots on the photo… (Q to stop)" : "⊙ Heal spots (Q)"}
      </Btn>
      <Slider label="Size %" value={p.spotSize} min={0.3} max={8} step={0.1} def={1.5} digits={1} onChange={p.setSpotSize} />
      {check("Visualize spots", p.proof.spots != null, (v) => p.setProof({ ...p.proof, spots: v ? 50 : null }), "While healing, show edges in black and white so dust stands out")}
      {p.proof.spots != null && <Slider label="Threshold" value={p.proof.spots} min={0} max={100} def={50} onChange={(v) => p.setProof({ ...p.proof, spots: v })} />}
      {dev.retouch.length === 0 && <p className={hint}>Healed spots and removals asked of the AI bar (e.g. “remove the people on the sand”) are listed here.</p>}
      <div className="mt-2 space-y-1">
        {dev.retouch.map((r, i) => (
          <div key={i} className="flex items-center gap-2 rounded bg-neutral-800 px-2 py-1 text-[11px]">
            <span className="flex-1 truncate" title={r.name}>
              {r.name}
            </span>
            <button className="text-neutral-500 hover:text-white" title="Undo this removal" onClick={() => change({ ...dev, retouch: dev.retouch.filter((_, k) => k !== i) }, "Delete removal")}>
              ✕
            </button>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/** view-only warnings & LittleCMS2 interactive printer/display soft-proofing */
export function ProofPanel(p: { proof: Proof; setProof: (p: Proof) => void }) {
  const [profiles, setProfiles] = useState<IccProfileEntry[]>([]);
  const [loadingIcc, setLoadingIcc] = useState(false);

  useEffect(() => {
    api.iccProfiles().then(setProfiles).catch(() => {});
  }, []);

  const handleImportIcc = async () => {
    try {
      const selected = await open({
        multiple: false,
        title: "Load Custom Display / Printer ICC Profile",
        filters: [{ name: "ICC Color Profile", extensions: ["icc", "icm", "ICC", "ICM"] }],
      });
      if (typeof selected === "string") {
        setLoadingIcc(true);
        const entry = await api.iccImportProfile(selected);
        const list = await api.iccProfiles();
        setProfiles(list);
        p.setProof({
          ...p.proof,
          gamut: entry.id,
          intent: p.proof.intent ?? "perceptual",
          bpc: p.proof.bpc ?? true,
          simulate_paper: p.proof.simulate_paper ?? true,
          gamut_warn: p.proof.gamut_warn ?? true,
        });
      }
    } catch (e) {
      console.warn("ICC import error:", e);
    } finally {
      setLoadingIcc(false);
    }
  };

  const activeProfile = profiles.find((pr) => pr.id === p.proof.gamut);

  return (
    <Panel title="Studio Soft-Proofing (LittleCMS2)" open={false}>
      <div className="mb-1 flex items-center justify-between text-[10px] text-neutral-400">
        <span>Target Display / Printer Profile</span>
        <button
          type="button"
          onClick={handleImportIcc}
          disabled={loadingIcc}
          className="text-sky-400 hover:text-sky-300 font-medium cursor-pointer"
          title="Import custom .icc or .icm paper/printer/monitor profile"
        >
          {loadingIcc ? "Loading…" : "＋ Load .ICC…"}
        </button>
      </div>
      <select
        className="mb-2 w-full rounded bg-neutral-800 px-2 py-1 text-[11px] text-neutral-100 outline-none border border-neutral-700/70"
        value={p.proof.gamut ?? ""}
        onChange={(e) => {
          const g = e.target.value || null;
          p.setProof({
            ...p.proof,
            gamut: g,
            intent: p.proof.intent ?? "perceptual",
            bpc: p.proof.bpc ?? true,
            simulate_paper: p.proof.simulate_paper ?? (g === "FineArtMatte" || g === "BarytaGloss"),
            gamut_warn: p.proof.gamut_warn ?? true,
          });
        }}
      >
        <option value="">Off (Working Rec.2020 → Display)</option>
        <optgroup label="Wide-Gamut & Standard Spaces">
          <option value="sRGB">sRGB IEC61966-2.1</option>
          <option value="DisplayP3">Apple XDR / Display P3</option>
          <option value="AdobeRGB">Adobe RGB (1998)</option>
          <option value="ProPhotoRGB">ProPhoto RGB (ROMM D50 32-bit)</option>
          <option value="ACEScg">ACEScg (Linear AP1 D60 32-bit)</option>
        </optgroup>
        <optgroup label="Simulated Fine-Art Papers">
          <option value="FineArtMatte">Hahnemühle FineArt Matte (Warm Rag)</option>
          <option value="BarytaGloss">FineArt Baryta Photo Gloss</option>
        </optgroup>
        {profiles.some((pr) => pr.kind === "custom") && (
          <optgroup label="Custom Imported .ICC Profiles">
            {profiles
              .filter((pr) => pr.kind === "custom")
              .map((pr) => (
                <option key={pr.id} value={pr.id}>
                  {pr.label}
                </option>
              ))}
          </optgroup>
        )}
      </select>

      {activeProfile && (
        <p className="mb-2 rounded bg-neutral-950/70 px-2 py-1 text-[10px] text-neutral-400 border border-neutral-800">
          {activeProfile.description}
        </p>
      )}

      {p.proof.gamut && (
        <>
          <div className="mb-1 text-[10px] text-neutral-500">Rendering Intent</div>
          <Seg
            options={[
              ["perceptual", "Perceptual"],
              ["relative", "Relative"],
              ["saturation", "Sat"],
              ["absolute", "Absolute"],
            ]}
            value={p.proof.intent ?? "perceptual"}
            onChange={(intent) => p.setProof({ ...p.proof, intent })}
          />
          {check(
            "Ink Black-Point Compensation (BPC)",
            p.proof.bpc ?? true,
            (bpc) => p.setProof({ ...p.proof, bpc }),
            "Map working space black to printer/paper ink Dmax via LittleCMS2",
          )}
          {check(
            "Simulate Paper Tint & Ink Dynamic Range",
            p.proof.simulate_paper ?? false,
            (simulate_paper) => p.setProof({ ...p.proof, simulate_paper }),
            "Preview compressed paper contrast and warm cotton rag white point on screen",
          )}
          {check(
            "Neon Out-of-Gamut Overlay Warning",
            p.proof.gamut_warn ?? true,
            (gamut_warn) => p.setProof({ ...p.proof, gamut_warn }),
            "Highlight unprintable out-of-gamut pixels in neon magenta",
          )}
        </>
      )}

      {check("Highlight / Shadow Clipping Warning (J)", p.proof.clipping, (v) => p.setProof({ ...p.proof, clipping: v }))}
      <p className={hint}>Neon magenta: colours outside target gamut. Red: clipped highlights; blue: crushed shadows.</p>
    </Panel>
  );
}

const DEMOSAIC_ALGOS: [string, string][] = [
  ["rcd", "RCD — Ratio Corrected (General Sharpness)"],
  ["amaze", "AMaZE — Low-ISO Maze Artifact Suppression"],
  ["lmmse", "LMMSE — High-ISO Astro / Night Photography"],
  ["ahd", "AHD — Adaptive Homogeneity-Directed"],
  ["dcb", "DCB — Detail-Preserving Interpolation"],
  ["dht", "DHT — Directional Homogeneity"],
  ["aahd", "AAHD — Enhanced Adaptive AHD"],
  ["ppg", "PPG — Patterned Pixel Grouping"],
  ["vng", "VNG — Variable Number of Gradients"],
  ["linear", "Bilinear — Fast Linear"],
];

const HIGHLIGHT_MODES: [string, string][] = [
  ["clip", "Clip — Standard White Clipping"],
  ["unclip", "Unclip — Raw Unclipped Channels"],
  ["blend", "Blend — Multi-Channel Highlight Blend"],
  ["rebuild", "Rebuild — Multi-Channel Highlight Recovery"],
];

/** Camera RAW Engine (LibRaw C++), Multi-Algorithm Demosaicing (RCD / AMaZE / LMMSE),
 *  Sensor Optical Black-Point Calibration, Highlight Reconstruction & MakerNotes Telemetry. */
export function RawEnginePanel(
  p: WholeProps & {
    rawInfo: RawInfo | null;
    showAfOverlay: boolean;
    setShowAfOverlay: (v: boolean) => void;
  },
) {
  const { dev, change } = p;
  const raw = dev.raw ?? { engine: "", demosaic: "", highlights: "", black_offset: 0, raw_denoise: 0 };
  const setRaw = (patch: Partial<typeof raw>, label: string) => {
    const next = { ...raw, ...patch };
    // Automatically enable LibRaw C++ engine when picking a pro demosaic, highlight mode, black offset, or linear denoise
    if (
      (patch.demosaic && patch.demosaic !== "") ||
      (patch.highlights && patch.highlights !== "" && patch.highlights !== "clip") ||
      (patch.black_offset != null && patch.black_offset !== 0) ||
      (patch.raw_denoise != null && patch.raw_denoise > 0)
    ) {
      if (!next.engine) next.engine = "libraw";
    }
    change({ ...dev, raw: next }, label, label);
  };

  const info = p.rawInfo;

  return (
    <Panel
      title="Camera RAW & Optical Calibration"
      open={false}
      right={
        <button
          className="text-[10px] font-normal normal-case text-neutral-500 hover:text-neutral-200"
          onClick={() =>
            change(
              { ...dev, raw: { engine: "", demosaic: "", highlights: "", black_offset: 0, raw_denoise: 0 } },
              "Reset RAW engine",
            )
          }
        >
          reset
        </button>
      }
    >
      <div className="mb-1 text-[10px] text-neutral-400">RAW Pipeline Engine</div>
      <Seg
        options={[
          ["", "Auto"],
          ["libraw", "LibRaw C++ Studio"],
        ]}
        value={raw.engine || ""}
        onChange={(engine) => setRaw({ engine }, "RAW engine")}
      />

      <div className={sub}>Demosaicing Algorithm</div>
      <select
        className="mb-1.5 w-full rounded bg-neutral-800 px-2 py-1 text-[11px] text-neutral-100 outline-none border border-neutral-700/70"
        value={raw.demosaic || "rcd"}
        onChange={(e) => setRaw({ demosaic: e.target.value, engine: "libraw" }, `Demosaic: ${e.target.value.toUpperCase()}`)}
      >
        {DEMOSAIC_ALGOS.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>

      <div className={sub}>Highlight Reconstruction</div>
      <select
        className="mb-2 w-full rounded bg-neutral-800 px-2 py-1 text-[11px] text-neutral-100 outline-none border border-neutral-700/70"
        value={raw.highlights || "clip"}
        onChange={(e) => setRaw({ highlights: e.target.value, engine: "libraw" }, `Highlight recovery: ${e.target.value}`)}
      >
        {HIGHLIGHT_MODES.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>

      <Slider
        label="Optical Black Offset"
        value={raw.black_offset ?? 0}
        def={0}
        min={-256}
        max={256}
        step={1}
        onChange={(v) => setRaw({ black_offset: Math.round(v) }, "Optical black offset")}
      />
      <Slider
        label="Linear Sensor Denoise"
        value={raw.raw_denoise ?? 0}
        def={0}
        min={0}
        max={100}
        step={1}
        onChange={(v) => setRaw({ raw_denoise: Math.round(v) }, "RAW sensor denoise")}
      />

      {info && (
        <div className="mt-2.5 space-y-1.5 rounded border border-neutral-800 bg-neutral-950/80 p-2 text-[10px]">
          <div className="flex items-center justify-between border-b border-neutral-800/80 pb-1 font-semibold text-neutral-300">
            <span>Sensor & MakerNotes (C++)</span>
            <span className="rounded bg-sky-950/80 px-1.5 py-0.2 font-mono text-[9px] text-sky-300 border border-sky-800/50">
              {info.is_xtrans ? "X-Trans 6×6" : info.cfa_pattern || "Bayer"}
            </span>
          </div>
          <div className="grid grid-cols-[92px_1fr] gap-x-2 gap-y-0.5 text-neutral-400">
            <span>Optical Black</span>
            <span className="font-mono text-neutral-200">{info.black_levels.join(" / ")} ADU</span>
            <span>White Level</span>
            <span className="font-mono text-neutral-200">{info.white_level} ADU</span>
            {info.shutter_count > 0 && (
              <>
                <span>Shutter Count</span>
                <span className="font-mono text-emerald-300">{info.shutter_count.toLocaleString()} actuations</span>
              </>
            )}
            {info.lens_serial && (
              <>
                <span>Lens Serial</span>
                <span className="font-mono text-neutral-200 truncate">{info.lens_serial}</span>
              </>
            )}
            {info.body_serial && (
              <>
                <span>Body Serial</span>
                <span className="font-mono text-neutral-200 truncate">{info.body_serial}</span>
              </>
            )}
            {info.focus_distance_m > 0 && (
              <>
                <span>Focus Distance</span>
                <span className="font-mono text-neutral-200">{info.focus_distance_m.toFixed(2)} m</span>
              </>
            )}
          </div>
          {info.af_points.length > 0 && (
            <div className="pt-1 border-t border-neutral-800/80 flex items-center justify-between">
              <span className="text-neutral-400">
                Active AF Points ({info.af_points.filter((a) => a.active).length}/{info.af_points.length})
              </span>
              <Btn active={p.showAfOverlay} onClick={() => p.setShowAfOverlay(!p.showAfOverlay)}>
                {p.showAfOverlay ? "Hide AF Overlay" : "Show AF Points"}
              </Btn>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
