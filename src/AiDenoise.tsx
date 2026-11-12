import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AiModel, AlignResult, api, Develop as Dev, Photo, Settings, Stroke, thumbUrl } from "./api";
import { PROFILES } from "./Auto";
import { Btn, HUE, Icon, Panel, Slider } from "./controls";

const hint = "mt-1 text-[10px] text-neutral-500";

type Transform = Dev["transform"];

const defTransform: Transform = {
  upright: "off",
  guides: [],
  rotate: 0,
  vertical: 0,
  horizontal: 0,
  aspect: 0,
  scale: 100,
  offset_x: 0,
  offset_y: 0,
  constrain_crop: true,
};

export type AiJobProgress = {
  label: string;
  done: number;
  total: number;
  percent: number;
  startTime: number;
  elapsedSec: number;
};

/** Progress bar with percentage, phase details, elapsed timer, ETA, and cancellation. */
function AiProgressBar({
  title,
  label,
  done,
  total,
  percent,
  modelName,
  elapsedSec,
  onCancel,
}: {
  title: string;
  label?: string;
  done: number;
  total: number;
  percent: number;
  modelName?: string;
  elapsedSec: number;
  onCancel?: () => void;
}) {
  const isIndeterminate = total <= 1 && done === 0;

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const remainingSec =
    done > 0 && total > done && elapsedSec > 0
      ? Math.round((elapsedSec / done) * (total - done))
      : null;

  return (
    <div className="mt-2.5 rounded-lg border border-sky-800/60 bg-neutral-950 p-2.5 shadow-md space-y-2">
      {/* Title & percentage */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="h-2 w-2 rounded-full bg-sky-400 animate-pulse shrink-0" />
          <span className="font-semibold text-neutral-100 truncate">{title}</span>
        </div>
        <div className="tabular-nums font-mono font-bold text-sky-400 text-xs shrink-0">
          {isIndeterminate ? (
            <span className="text-[10px] font-normal text-neutral-400 animate-pulse">Initializing…</span>
          ) : (
            `${percent}%`
          )}
        </div>
      </div>

      {/* Progress Track */}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-neutral-800">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500 transition-all duration-300 ${
            isIndeterminate ? "w-full animate-pulse" : ""
          }`}
          style={{ width: isIndeterminate ? "100%" : `${Math.max(4, Math.min(100, percent))}%` }}
        />
      </div>

      {/* Live step label & timer */}
      <div className="flex items-center justify-between gap-2 text-[10px] text-neutral-400">
        <div className="truncate">
          {label ? (
            <span className="text-neutral-200 font-medium">{label}</span>
          ) : total > 1 ? (
            <span>Step {done} of {total}</span>
          ) : (
            <span>Executing neural pass…</span>
          )}
          {modelName && <span className="ml-1 text-neutral-500">({modelName})</span>}
        </div>
        <div className="flex items-center gap-1.5 shrink-0 font-mono text-[10px] text-neutral-400">
          <span>⏱ {formatTime(elapsedSec)}</span>
          {remainingSec != null && remainingSec > 0 && (
            <span className="text-neutral-500">~{remainingSec}s left</span>
          )}
        </div>
      </div>

      {/* Cancel button */}
      {onCancel && (
        <div className="flex justify-end pt-1 border-t border-neutral-900">
          <button
            type="button"
            onClick={onCancel}
            className="rounded px-2 py-0.5 text-[10px] text-neutral-400 hover:text-red-400 hover:bg-neutral-900 border border-neutral-800 transition-colors"
            title="Cancel this AI operation"
          >
            ✕ Cancel
          </button>
        </div>
      )}
    </div>
  );
}

/** Photo AI tools organized into workflow groups:
 *  1. Tone & Color (Auto Adjust)
 *  2. Geometry & Perspective (PerspectiveFields Auto Align)
 *  3. Detail & Restoration (Neural Denoise)
 *  4. Retouch & Faces (Object Removal, Generative Fill, Face Swap & AI Layers)
 *  Adjustments are stored non-destructively inside Develop with pixel patches cached on disk. */
export function AiTab(p: {
  photoId: number | null;
  settings: Settings | null;
  dev: Dev | null;
  onChange: (d: Dev, label: string, replacePrefix?: string) => void;
  setDev: (d: Dev) => void;
  /** the brush is on: paint over an object on the photo */
  drawing: boolean;
  onDraw: () => void;
  /** the brush strokes over the object, once snapped to its edges and shown for confirming */
  selection: Stroke[] | null;
  /** brush radius as a fraction of the photo's width */
  brush: number;
  setBrush: (v: number) => void;
  onClear: () => void;
  onOpenModels: () => void;
  /** show a photo this tool just made */
  onOpenPhoto?: (id: number) => void;
  /** run the local AI auto adjustment on the photo: a style (profile id or none), its strength and how far to go, 0-100 */
  onAutoAdjust: (profile: string | null, style: number, amount: number) => Promise<void>;
}) {
  const photoId = p.photoId;
  const [models, setModels] = useState<AiModel[]>([]);
  const [strength, setStrength] = useState(p.dev?.ai_denoise?.strength ?? 80);
  const [grow, setGrow] = useState(6);
  const [prompt, setPrompt] = useState("");
  const [job, setJob] = useState<AiJobProgress | null>(null);
  const [alignJob, setAlignJob] = useState<AiJobProgress | null>(null);
  const [alignInfo, setAlignInfo] = useState<AlignResult | null>(null);
  const [removing, setRemoving] = useState(false);
  const [fillJob, setFillJob] = useState<AiJobProgress | null>(null);
  const [style, setStyle] = useState("");
  const [styleStrength, setStyleStrength] = useState(100);
  const [amount, setAmount] = useState(100);
  const [adjusting, setAdjusting] = useState(false);
  const [swapJob, setSwapJob] = useState<AiJobProgress | null>(null);
  /** the library face picker: open while not null, holding the search text */
  const [faceQuery, setFaceQuery] = useState<string | null>(null);
  const [faces, setFaces] = useState<Photo[]>([]);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  /** which tool is showing: one at a time, so nothing needs scrolling to reach */
  const [sub, setSub] = useState<"tone" | "align" | "denoise" | "retouch">("tone");
  const [rsub, setRsub] = useState<"remove" | "face">("remove");

  useEffect(() => void api.aiModels().then((r) => setModels(r.models), () => setModels([])), []);
  useEffect(() => {
    setAlignInfo(null);
  }, [photoId]);

  useEffect(() => {
    const un = listen<{ job: string; label?: string; done: number; total: number; finished?: boolean }>("job-progress", (e) => {
      const ev = e.payload;
      if (ev.finished) {
        if (ev.job === "ai-denoise") setJob(null);
        if (ev.job === "ai-align") setAlignJob(null);
        if (ev.job === "ai-faceswap") setSwapJob(null);
        if (ev.job === "ai-remove") {
          setFillJob(null);
          setRemoving(false);
        }
        return;
      }
      const pct = ev.total > 0 ? Math.min(100, Math.round((ev.done / ev.total) * 100)) : 0;
      if (ev.job === "ai-denoise") {
        setJob((prev) => ({
          label: ev.label || "AI Denoising…",
          done: ev.done,
          total: ev.total,
          percent: pct,
          startTime: prev?.startTime ?? Date.now(),
          elapsedSec: Math.floor((Date.now() - (prev?.startTime ?? Date.now())) / 1000),
        }));
      }
      if (ev.job === "ai-align") {
        setAlignJob((prev) => ({
          label: ev.label || "Estimating perspective fields…",
          done: ev.done,
          total: ev.total,
          percent: pct,
          startTime: prev?.startTime ?? Date.now(),
          elapsedSec: Math.floor((Date.now() - (prev?.startTime ?? Date.now())) / 1000),
        }));
      }
      if (ev.job === "ai-faceswap") {
        setSwapJob((prev) => ({
          label: ev.label || "Swapping faces…",
          done: ev.done,
          total: ev.total,
          percent: pct,
          startTime: prev?.startTime ?? Date.now(),
          elapsedSec: Math.floor((Date.now() - (prev?.startTime ?? Date.now())) / 1000),
        }));
      }
      if (ev.job === "ai-remove") {
        setFillJob((prev) => ({
          label: ev.label || "Removing object…",
          done: ev.done,
          total: ev.total,
          percent: pct,
          startTime: prev?.startTime ?? Date.now(),
          elapsedSec: Math.floor((Date.now() - (prev?.startTime ?? Date.now())) / 1000),
        }));
      }
    });
    return () => void un.then((f) => f());
  }, []);

  useEffect(() => {
    if (!job && !alignJob && !fillJob && !swapJob && !removing) return;
    const interval = setInterval(() => {
      if (job) {
        setJob((prev) => (prev ? { ...prev, elapsedSec: Math.floor((Date.now() - prev.startTime) / 1000) } : null));
      }
      if (alignJob) {
        setAlignJob((prev) => (prev ? { ...prev, elapsedSec: Math.floor((Date.now() - prev.startTime) / 1000) } : null));
      }
      if (fillJob) {
        setFillJob((prev) => (prev ? { ...prev, elapsedSec: Math.floor((Date.now() - prev.startTime) / 1000) } : null));
      }
      if (swapJob) {
        setSwapJob((prev) => (prev ? { ...prev, elapsedSec: Math.floor((Date.now() - prev.startTime) / 1000) } : null));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [job != null, alignJob != null, fillJob != null, swapJob != null, removing]);

  useEffect(() => {
    if (faceQuery == null) return;
    const t = setTimeout(() => void api.query({ text: faceQuery.trim() || null, limit: 120 }).then(setFaces, () => setFaces([])), 200);
    return () => clearTimeout(t);
  }, [faceQuery]);

  useEffect(() => {
    if (p.dev?.ai_denoise?.strength != null) {
      setStrength(p.dev.ai_denoise.strength);
    }
  }, [p.dev?.ai_denoise?.strength]);

  // what each tool uses: the model chosen in Settings if it is installed, else the best installed one
  const use = (group: AiModel[], chosen?: string) => group.find((m) => m.id === chosen && m.installed) ?? group.find((m) => m.installed);
  const alignModel = use(models.filter((m) => m.kind === "align"), p.settings?.ai_align_model);
  const denoiseModel = use(models.filter((m) => m.kind === "denoise"), p.settings?.ai_denoise_model);
  const removalModel = use(models.filter((m) => m.kind === "inpaint" || m.kind === "generate"), p.settings?.ai_removal_model);
  const faceModel = use(models.filter((m) => m.kind === "faceswap"));
  const generative = removalModel?.kind === "generate";

  const autoAdjust = async () => {
    setMsg(null);
    setAdjusting(true);
    try {
      await p.onAutoAdjust(style || null, styleStrength, amount);
    } finally {
      setAdjusting(false);
    }
  };

  const runAlign = async (mode: "horizontal" | "vertical" | "both") => {
    if (photoId == null || !alignModel || !p.dev) return;
    setMsg(null);
    setAlignJob({
      label: "Estimating dense gravity & latitude fields…",
      done: 0,
      total: 3,
      percent: 15,
      startTime: Date.now(),
      elapsedSec: 0,
    });
    try {
      const res = await api.aiAlign(photoId, p.dev, alignModel.id, mode);
      setAlignInfo(res);
      const label =
        mode === "horizontal"
          ? "AI Align Horizon"
          : mode === "vertical"
          ? "AI Align Vertical"
          : "AI Align Horizon & Vertical";
      p.onChange({ ...p.dev, transform: res.transform }, label);
      const parts: string[] = [];
      if (mode === "horizontal" || mode === "both") {
        parts.push(`roll ${res.roll_deg >= 0 ? "+" : ""}${res.roll_deg.toFixed(2)}°`);
      }
      if (mode === "vertical" || mode === "both") {
        parts.push(`pitch ${res.pitch_deg >= 0 ? "+" : ""}${res.pitch_deg.toFixed(2)}° (vertical ${res.transform.vertical >= 0 ? "+" : ""}${res.transform.vertical.toFixed(0)})`);
      }
      setMsg({
        ok: true,
        text: `Aligned (${parts.join(", ")} · vFOV ${res.vfov_deg.toFixed(1)}°).`,
      });
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    } finally {
      setAlignJob(null);
    }
  };

  const setTransform = (patch: Partial<Transform>, label: string, replacePrefix?: string) => {
    if (!p.dev) return;
    const cur = p.dev.transform ?? defTransform;
    p.onChange({ ...p.dev, transform: { ...cur, ...patch } }, label, replacePrefix);
  };

  const denoise = async () => {
    if (photoId == null || !denoiseModel) return;
    setMsg(null);
    setJob({
      label: "Initializing denoise model…",
      done: 0,
      total: 1,
      percent: 0,
      startTime: Date.now(),
      elapsedSec: 0,
    });
    try {
      const newDev = await api.aiDenoiseInplace(photoId, denoiseModel.id, strength);
      p.setDev(newDev);
      setMsg({ ok: true, text: "AI Denoise applied in-place. Adjust strength or toggle anytime." });
    } catch (e) {
      if (String(e).toLowerCase().includes("cancel")) {
        setMsg({ ok: false, text: "AI Denoise cancelled." });
      } else {
        setMsg({ ok: false, text: String(e) });
      }
    } finally {
      setJob(null);
    }
  };

  const remove = async () => {
    if (photoId == null || !p.selection || !removalModel) return;
    setMsg(null);
    setRemoving(true);
    const isMoebius = removalModel.id === "moebius" || removalModel.id.startsWith("moebius");
    setFillJob({
      label: generative ? "Initializing diffusion model…" : isMoebius ? "Initializing Moebius model…" : "Preparing object removal…",
      done: 0,
      total: generative ? 25 : isMoebius ? 19 : 3,
      percent: 0,
      startTime: Date.now(),
      elapsedSec: 0,
    });
    try {
      const newDev = await api.aiInpaintInplace(photoId, removalModel.id, p.selection, grow, generative ? prompt : "");
      p.setDev(newDev);
      p.onClear();
      setMsg({ ok: true, text: "Object removed in-place. You can toggle or delete it below." });
    } catch (e) {
      if (String(e).toLowerCase().includes("cancel")) {
        setMsg({ ok: false, text: "Object removal cancelled." });
      } else {
        setMsg({ ok: false, text: String(e) });
      }
    } finally {
      setRemoving(false);
      setFillJob(null);
    }
  };

  const faceSwap = async (source: number) => {
    if (photoId == null || !faceModel) return;
    setFaceQuery(null);
    setMsg(null);
    setSwapJob({ label: "Loading face models…", done: 0, total: 1, percent: 0, startTime: Date.now(), elapsedSec: 0 });
    try {
      p.setDev(await api.aiFaceSwap(photoId, faceModel.id, source));
      setMsg({ ok: true, text: "Face swapped in-place. Each face is its own layer below: toggle or delete it." });
    } catch (e) {
      setMsg({ ok: false, text: String(e).toLowerCase().includes("cancel") ? "Face swap cancelled." : String(e) });
    } finally {
      setSwapJob(null);
    }
  };

  const toggleRemoval = (removalId: string) => {
    if (!p.dev || !p.dev.ai_removals) return;
    const nextRemovals = p.dev.ai_removals.map((r) =>
      r.id === removalId ? { ...r, enabled: !r.enabled } : r,
    );
    const updated = { ...p.dev, ai_removals: nextRemovals };
    p.onChange(updated, "Toggle AI removal");
  };

  const deleteRemoval = async (removalId: string) => {
    if (photoId == null) return;
    try {
      const newDev = await api.aiDeleteRemoval(photoId, removalId);
      p.setDev(newDev);
      setMsg({ ok: true, text: "AI layer deleted." });
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    }
  };

  const onStrengthChange = (val: number) => {
    setStrength(val);
    if (p.dev && p.dev.ai_denoise?.model) {
      const updated = { ...p.dev, ai_denoise: { ...p.dev.ai_denoise, strength: val } };
      p.onChange(updated, "AI Denoise Strength", "AI Denoise Strength");
    }
  };

  const toggleDenoise = () => {
    if (!p.dev || !p.dev.ai_denoise) return;
    const updated = {
      ...p.dev,
      ai_denoise: { ...p.dev.ai_denoise, enabled: !p.dev.ai_denoise.enabled },
    };
    p.onChange(updated, updated.ai_denoise.enabled ? "Enable AI Denoise" : "Disable AI Denoise");
  };

  const hasDenoised = !!p.dev?.ai_denoise?.model;
  const denoiseEnabled = p.dev?.ai_denoise?.enabled ?? false;
  const tr = p.dev?.transform ?? defTransform;
  const hasAlignment = Math.abs(tr.rotate) > 0.01 || Math.abs(tr.vertical) > 0.1;
  const layerCount = p.dev?.ai_removals?.length ?? 0;

  const modelsLink = (
    <button type="button" className="underline decoration-dotted hover:text-neutral-200" onClick={p.onOpenModels}>
      Settings → AI models
    </button>
  );

  return (
    <div className="text-[11px] text-neutral-300">
      {/* Top status bar */}
      <div className="flex items-center justify-between border-b border-neutral-800 bg-neutral-900/50 px-3 py-1.5 text-[10px] text-neutral-400">
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          <span>Non-destructive local AI</span>
        </span>
        <button
          type="button"
          onClick={p.onOpenModels}
          className="rounded border border-neutral-700/80 bg-neutral-800/80 px-2 py-0.5 text-[10px] text-neutral-200 hover:border-sky-600 hover:text-white transition-colors"
        >
          ⚙ Manage AI Models
        </button>
      </div>

      {/* Tool sub-menu: a dot marks a tool that is applied to this photo, pulsing while it runs */}
      <div className="border-b border-neutral-800 px-3 py-2">
        <div className="grid grid-cols-4 overflow-hidden rounded bg-neutral-800 text-[11px]" role="tablist">
          {(
            [
              ["tone", "Enhance", "Tone & color: AI auto adjust and creative looks", adjusting, false, "text-amber-500"],
              ["align", "Align", "Geometry: level the horizon and straighten verticals", !!alignJob, hasAlignment, "text-cyan-500"],
              ["denoise", "Denoise", "Detail: neural denoise", !!job, hasDenoised && denoiseEnabled, "text-emerald-500"],
              ["retouch", "Retouch", "Object removal, generative fill, face swap and their layers", removing || !!swapJob, layerCount > 0, "text-violet-500"],
            ] as const
          ).map(([t, label, title, busy, on, hue]) => (
            <button
              key={t}
              role="tab"
              aria-selected={sub === t}
              title={title}
              onClick={() => setSub(t)}
              className={`whitespace-nowrap px-1.5 py-1 ${sub === t ? "bg-neutral-600 text-white" : "text-neutral-400 hover:text-neutral-200"}`}
            >
              {label}
              {t === "retouch" && layerCount > 0 && !busy ? ` (${layerCount})` : busy || on ? <span className={`ml-0.5 ${hue} ${busy ? "animate-pulse" : ""}`}>●</span> : null}
            </button>
          ))}
        </div>
      </div>

      {msg && (
        <div
          className={`mx-3 mt-2 flex items-start justify-between gap-2 rounded border px-2.5 py-1.5 text-[10px] ${
            msg.ok
              ? "border-emerald-800/60 bg-emerald-950/30 text-emerald-300"
              : "border-red-800/60 bg-red-950/30 text-red-300"
          }`}
        >
          <span className="break-words">{msg.text}</span>
          <button
            type="button"
            onClick={() => setMsg(null)}
            className="shrink-0 text-neutral-400 hover:text-neutral-200"
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {/* 1. TONE & COLOR — AI AUTO ADJUST */}
      {sub === "tone" && (
      <Panel
        title={
          <span className="flex items-center gap-1.5">
            <span className={HUE.amber.text}>
              <Icon d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6 2.1-2.1" />
            </span>
            <span>Tone & Color · Auto Adjust</span>
          </span>
        }
        right={<span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-medium text-amber-300">Built-in 3D-LUT</span>}
      >
        <p className="text-neutral-400 leading-relaxed">
          Sets white balance, exposure, tone and colour using an image-adaptive 3D-LUT model trained on photographers&apos; edits. Every slider stays editable.
        </p>
        <div className="mt-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-neutral-500">Creative Look</div>
          <select
            className="w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-[11px] text-neutral-100 outline-none focus:border-sky-600"
            value={style}
            onChange={(e) => setStyle(e.target.value)}
          >
            <option value="">None · Natural AI look</option>
            {PROFILES.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
          {style && <p className={hint}>{PROFILES.find(([id]) => id === style)?.[2]}</p>}
        </div>
        <div className="mt-1.5">
          <Slider label="Adjustment" value={amount} min={0} max={100} def={100} onChange={setAmount} />
          {style && <Slider label="Style strength" value={styleStrength} min={0} max={100} def={100} onChange={setStyleStrength} />}
        </div>
        <Btn
          active
          className="mt-2 w-full"
          disabled={photoId == null || adjusting}
          onClick={() => void autoAdjust()}
          title="Sets the tone & color sliders on this photo; Ctrl+Z undoes it"
        >
          {adjusting ? "Adjusting…" : "✦ AI Auto Adjust"}
        </Btn>
      </Panel>
      )}

      {/* 2. GEOMETRY & PERSPECTIVE — PERSPECTIVEFIELDS AUTO ALIGN */}
      {sub === "align" && (
      <Panel
        title={
          <span className="flex items-center gap-1.5">
            <span className={HUE.cyan.text}>
              <Icon d="M3 12h18M12 3v18M5 5l14 0 0 14-14 0z" />
            </span>
            <span>Geometry · Auto Align</span>
          </span>
        }
        right={
          <span
            className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
              alignModel
                ? hasAlignment
                  ? "bg-cyan-500/20 text-cyan-300"
                  : "bg-neutral-800 text-neutral-300"
                : "bg-amber-500/15 text-amber-300"
            }`}
          >
            {alignModel ? (hasAlignment ? `${tr.rotate.toFixed(1)}° / ${tr.vertical.toFixed(0)}` : "PerspectiveFields") : "Not installed"}
          </span>
        }
      >
        {alignModel ? (
          <>
            <p className="text-neutral-400 leading-relaxed">
              Predicts dense per-pixel gravity (roll) and latitude (pitch) vector fields (~0.1°–0.5° accuracy) to level horizons or straighten converging verticals.
            </p>
            <div className="mt-2 grid grid-cols-3 gap-1.5">
              <Btn
                active
                disabled={photoId == null || !!alignJob}
                onClick={() => void runAlign("horizontal")}
                title="Level the horizon using PerspectiveFields dense gravity vectors (sets Rotate)"
                className="flex flex-col items-center justify-center py-1.5"
              >
                <span className="text-xs leading-none">↔</span>
                <span className="mt-0.5 text-[10px]">Horizon</span>
              </Btn>
              <Btn
                active
                disabled={photoId == null || !!alignJob}
                onClick={() => void runAlign("vertical")}
                title="Correct vertical keystoning using PerspectiveFields latitude & pitch (sets Vertical)"
                className="flex flex-col items-center justify-center py-1.5"
              >
                <span className="text-xs leading-none">↕</span>
                <span className="mt-0.5 text-[10px]">Vertical</span>
              </Btn>
              <Btn
                active
                disabled={photoId == null || !!alignJob}
                onClick={() => void runAlign("both")}
                title="Auto-align both horizontal roll and vertical perspective simultaneously"
                className="flex flex-col items-center justify-center py-1.5"
              >
                <span className="text-xs leading-none">⊞</span>
                <span className="mt-0.5 text-[10px]">Align Both</span>
              </Btn>
            </div>

            {alignJob && (
              <AiProgressBar
                title="PerspectiveFields Align"
                label={alignJob.label}
                done={alignJob.done}
                total={alignJob.total}
                percent={alignJob.percent}
                modelName={alignModel.name}
                elapsedSec={alignJob.elapsedSec}
              />
            )}

            {alignInfo && (
              <div className="mt-2 grid grid-cols-4 gap-1 rounded border border-cyan-900/50 bg-cyan-950/20 px-2 py-1.5 text-center text-[10px]">
                <div>
                  <div className="text-[9px] uppercase text-neutral-500">Roll</div>
                  <div className="font-mono font-semibold text-cyan-300">
                    {alignInfo.roll_deg >= 0 ? "+" : ""}
                    {alignInfo.roll_deg.toFixed(2)}°
                  </div>
                </div>
                <div>
                  <div className="text-[9px] uppercase text-neutral-500">Pitch</div>
                  <div className="font-mono font-semibold text-cyan-300">
                    {alignInfo.pitch_deg >= 0 ? "+" : ""}
                    {alignInfo.pitch_deg.toFixed(2)}°
                  </div>
                </div>
                <div>
                  <div className="text-[9px] uppercase text-neutral-500">vFOV</div>
                  <div className="font-mono font-semibold text-neutral-200">{alignInfo.vfov_deg.toFixed(1)}°</div>
                </div>
                <div>
                  <div className="text-[9px] uppercase text-neutral-500">Conf</div>
                  <div className="font-mono font-semibold text-emerald-300">{Math.round(alignInfo.confidence * 100)}%</div>
                </div>
              </div>
            )}

            <div className="mt-2 border-t border-neutral-800/80 pt-1.5">
              <Slider
                label="Horizon (°)"
                value={tr.rotate}
                min={-15}
                max={15}
                step={0.05}
                digits={2}
                def={0}
                onChange={(v) => setTransform({ rotate: v }, "Rotate", "Rotate")}
              />
              <Slider
                label="Vertical"
                value={tr.vertical}
                min={-100}
                max={100}
                step={1}
                digits={0}
                def={0}
                onChange={(v) => setTransform({ vertical: v }, "Vertical", "Vertical")}
              />
              <div className="mt-1 flex items-center justify-between">
                <label className="flex items-center gap-1.5 cursor-pointer select-none text-[10px] text-neutral-400">
                  <input
                    type="checkbox"
                    checked={tr.constrain_crop}
                    onChange={(e) => setTransform({ constrain_crop: e.target.checked }, "Constrain crop")}
                    className="rounded border-neutral-700 text-sky-500 focus:ring-0"
                  />
                  <span>Constrain crop to image</span>
                </label>
                {hasAlignment && (
                  <button
                    type="button"
                    onClick={() => {
                      setAlignInfo(null);
                      setTransform({ rotate: 0, vertical: 0 }, "Reset alignment");
                    }}
                    className="rounded px-1.5 py-0.5 text-[10px] text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
            <p className={hint}>
              Using <span className="text-neutral-300">{alignModel.name}</span>. Switch in {modelsLink}.
            </p>
          </>
        ) : (
          <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2.5">
            <div className="font-medium text-neutral-200">PerspectiveFields · Dense Gravity &amp; Roll Vector</div>
            <p className="mt-0.5 text-neutral-400">
              Auto-aligns horizons and vertical perspective (~0.1°–0.5°) for architecture, room scenes, and landscapes using SegFormer + ConvNeXt (~200 MB).
            </p>
            <Btn active className="mt-2" onClick={p.onOpenModels}>
              Download in Settings → AI models
            </Btn>
          </div>
        )}
      </Panel>
      )}

      {/* 3. DETAIL & RESTORATION — AI DENOISE */}
      {sub === "denoise" && (
      <Panel
        title={
          <span className="flex items-center gap-1.5">
            <span className={HUE.emerald.text}>
              <Icon d="M4 6h4v4H4zM10 6h4v4h-4zM16 6h4v4h-4zM4 14h4v4H4zM10 14h4v4h-4zM16 14h4v4h-4z" />
            </span>
            <span>Detail · Neural Denoise</span>
          </span>
        }
        right={
          <span
            className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
              hasDenoised && denoiseEnabled
                ? "bg-emerald-500/20 text-emerald-300"
                : denoiseModel
                ? "bg-neutral-800 text-neutral-300"
                : "bg-amber-500/15 text-amber-300"
            }`}
          >
            {hasDenoised ? (denoiseEnabled ? `Active · ${Math.round(strength)}%` : "Bypassed") : denoiseModel ? "Ready" : "Not installed"}
          </span>
        }
      >
        {denoiseModel ? (
          <>
            <p className="text-neutral-400 leading-relaxed">
              Removes high-ISO sensor noise and grain while preserving fine edges and textures as a non-destructive layer.
            </p>
            {hasDenoised && (
              <label className="mt-2 mb-1 flex items-center gap-2 cursor-pointer select-none rounded border border-neutral-800 bg-neutral-900/60 px-2 py-1">
                <input
                  type="checkbox"
                  checked={denoiseEnabled}
                  onChange={toggleDenoise}
                  className="rounded border-neutral-700 text-sky-500 focus:ring-0"
                />
                <span className={`font-medium ${denoiseEnabled ? "text-emerald-300" : "text-neutral-500"}`}>
                  {denoiseEnabled ? "AI Denoise layer enabled" : "AI Denoise layer bypassed"}
                </span>
              </label>
            )}

            <div className="mt-1.5">
              <Slider
                label="Strength"
                value={Math.round(strength)}
                min={0}
                max={100}
                def={80}
                onChange={onStrengthChange}
              />
            </div>

            <div className="mt-2 flex items-center gap-2">
              <Btn
                active
                className="flex-1"
                disabled={photoId == null || !!job}
                onClick={() => void denoise()}
                title="Runs AI Denoise non-destructively in-place on this photo"
              >
                {job
                  ? job.total > 1
                    ? `Denoising… ${job.percent}%`
                    : "Starting…"
                  : hasDenoised
                  ? "✦ Re-run AI Denoise"
                  : "✦ AI Denoise"}
              </Btn>
              {job && (
                <Btn onClick={() => void api.cancelJob("ai-denoise")} title="Stop">
                  Cancel
                </Btn>
              )}
            </div>
            {job && (
              <AiProgressBar
                title="AI Denoise"
                label={job.label}
                done={job.done}
                total={job.total}
                percent={job.percent}
                modelName={denoiseModel.name}
                elapsedSec={job.elapsedSec}
                onCancel={() => void api.cancelJob("ai-denoise")}
              />
            )}
            <p className={hint}>
              Using <span className="text-neutral-300">{denoiseModel.name}</span>. Switch in {modelsLink}.
            </p>
          </>
        ) : (
          <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2.5">
            <p className="text-neutral-400">Needs an AI Denoise model (SCUNet or NAFNet).</p>
            <Btn active className="mt-2" onClick={p.onOpenModels}>
              Download in Settings → AI models
            </Btn>
          </div>
        )}
      </Panel>
      )}

      {/* 4. RETOUCH & FACES — OBJECT REMOVAL, FACE SWAP & AI LAYERS */}
      {sub === "retouch" && (
      <Panel
        title={
          <span className="flex items-center gap-1.5">
            <span className={HUE.violet.text}>
              <Icon d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
            </span>
            <span>Retouch &amp; Faces · Layers</span>
          </span>
        }
        right={
          layerCount > 0 ? (
            <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[9px] font-medium text-violet-300">
              {layerCount} {layerCount === 1 ? "layer" : "layers"}
            </span>
          ) : null
        }
      >
        <div className="mb-2.5 grid grid-cols-2 overflow-hidden rounded bg-neutral-800 text-[11px]" role="tablist">
          {(
            [
              ["remove", "Remove & Fill", removing],
              ["face", "Face Swap", !!swapJob],
            ] as const
          ).map(([t, label, busy]) => (
            <button
              key={t}
              role="tab"
              aria-selected={rsub === t}
              onClick={() => setRsub(t)}
              className={`px-1.5 py-1 ${rsub === t ? "bg-neutral-600 text-white" : "text-neutral-400 hover:text-neutral-200"}`}
            >
              {label}
              {busy && <span className="ml-0.5 animate-pulse text-violet-500">●</span>}
            </button>
          ))}
        </div>

        {/* Sub-card: Object Removal & Generative Fill */}
        {rsub === "remove" && (
        <div className="rounded-md border border-neutral-800/90 bg-neutral-900/40 p-2.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-neutral-200">Object Removal &amp; Fill</span>
            {removalModel && (
              <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[9px] text-neutral-400">
                {generative ? "Generative Fill" : "Inpaint"}
              </span>
            )}
          </div>
          {removalModel ? (
            <>
              <p className="mt-0.5 text-neutral-400">
                Paint over unwanted objects; edges snap automatically before you confirm.
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <Btn
                  active={p.drawing}
                  disabled={photoId == null || removing}
                  onClick={p.onDraw}
                  title="Paint over the object on the photo (Esc cancels)"
                  className="flex-1"
                >
                  🖌 {p.drawing ? "Painting… drag on photo" : p.selection ? "Paint more" : "Paint over object"}
                </Btn>
                {p.selection && (
                  <Btn onClick={p.onClear} disabled={removing}>
                    Clear
                  </Btn>
                )}
              </div>
              <div className="mt-1">
                <Slider
                  label="Brush size"
                  value={Math.round(p.brush * 1000)}
                  min={5}
                  max={120}
                  def={12}
                  onChange={(v) => p.setBrush(v / 1000)}
                />
              </div>
              {p.selection && (
                <div className="mt-2 rounded border border-emerald-800/50 bg-emerald-950/20 p-2">
                  <div className="text-[10px] font-medium text-emerald-300">Selection ready — red area will be replaced</div>
                  <Slider label="Grow edge (px)" value={grow} min={0} max={30} def={6} onChange={setGrow} />
                  {generative && (
                    <input
                      className="mt-1 w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-[11px] text-neutral-100 outline-none focus:border-sky-600"
                      placeholder="What should fill the gap? (optional: grass, sky…)"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                    />
                  )}
                  <Btn
                    active
                    className="mt-2 w-full"
                    disabled={removing}
                    onClick={() => void remove()}
                    title="Removes the object non-destructively in-place on this photo"
                  >
                    {removing
                      ? fillJob && fillJob.total > 1
                        ? `Removing… ${fillJob.percent}%`
                        : "Removing…"
                      : generative
                      ? "✦ Generative Fill"
                      : "✦ Remove Object"}
                  </Btn>
                  {removing && (
                    <AiProgressBar
                      title="Removing Object"
                      label={
                        fillJob?.label ||
                        (generative
                          ? "Synthesizing generative fill…"
                          : removalModel.id === "moebius" || removalModel.id.startsWith("moebius")
                          ? "Synthesizing diffusion fill…"
                          : "Inpainting patch…")
                      }
                      done={fillJob?.done ?? 0}
                      total={
                        fillJob?.total ??
                        (generative ? 25 : removalModel.id === "moebius" || removalModel.id.startsWith("moebius") ? 19 : 3)
                      }
                      percent={
                        fillJob?.percent ??
                        (generative ? 0 : removalModel.id === "moebius" || removalModel.id.startsWith("moebius") ? 0 : 33)
                      }
                      modelName={removalModel.name}
                      elapsedSec={fillJob?.elapsedSec ?? 0}
                      onCancel={() => void api.cancelJob("ai-remove")}
                    />
                  )}
                </div>
              )}
              <p className={hint}>
                Using <span className="text-neutral-300">{removalModel.name}</span>. Switch in {modelsLink}.
              </p>
            </>
          ) : (
            <p className="mt-1 text-neutral-400">
              Needs an object removal model. Download one in {modelsLink}.
            </p>
          )}
        </div>
        )}

        {/* Sub-card: Face Swap */}
        {rsub === "face" && (
        <div className="rounded-md border border-neutral-800/90 bg-neutral-900/40 p-2.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-neutral-200">Face Swap</span>
            <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[9px] text-neutral-400">
              {faceModel ? "HyperSwap 1A" : "Not installed"}
            </span>
          </div>
          {faceModel ? (
            <>
              <p className="mt-0.5 text-neutral-400">
                Pick a photo from your library to transfer its face onto detected faces in this photo.
              </p>
              <Btn
                active
                className="mt-2 w-full"
                disabled={photoId == null || !!swapJob}
                onClick={() => setFaceQuery("")}
                title="Choose the library photo with the face to use"
              >
                {swapJob ? `Swapping… ${swapJob.percent}%` : "✦ Choose Face from Library…"}
              </Btn>
              {faceQuery != null &&
                createPortal(
                  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setFaceQuery(null)}>
                    <div
                      className="flex max-h-full w-[760px] max-w-full flex-col rounded-lg border border-neutral-700 bg-neutral-900 p-4 text-[12px] text-neutral-300 shadow-2xl"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <h2 className="text-sm font-semibold text-neutral-100">Choose the face to use</h2>
                        <Btn onClick={() => setFaceQuery(null)}>Close</Btn>
                      </div>
                      <input
                        autoFocus
                        className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-neutral-100 outline-none focus:border-sky-600"
                        placeholder="Search the library (name, keyword, camera…)"
                        value={faceQuery}
                        onChange={(e) => setFaceQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Escape" && setFaceQuery(null)}
                      />
                      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
                        <div className="grid grid-cols-[repeat(auto-fill,minmax(130px,1fr))] gap-2">
                          {faces
                            .filter((ph) => ph.id !== photoId)
                            .map((ph) => (
                              <button
                                key={ph.id}
                                type="button"
                                onClick={() => void faceSwap(ph.id)}
                                title={`Use the face in ${ph.filename}`}
                                className="group rounded border border-neutral-800 bg-neutral-950 p-1 text-left hover:border-sky-500 focus:border-sky-500 focus:outline-none"
                              >
                                <img src={thumbUrl(ph)} alt={ph.filename} loading="lazy" className="aspect-square w-full rounded-sm object-cover" />
                                <div className="mt-1 truncate text-[10px] text-neutral-400 group-hover:text-neutral-200">{ph.filename}</div>
                              </button>
                            ))}
                        </div>
                        {faces.length === 0 && <p className="py-6 text-center text-neutral-500">No photos match.</p>}
                      </div>
                    </div>
                  </div>,
                  document.body,
                )}
              {swapJob && (
                <AiProgressBar
                  title="Face Swap"
                  label={swapJob.label}
                  done={swapJob.done}
                  total={swapJob.total}
                  percent={swapJob.percent}
                  modelName={faceModel.name}
                  elapsedSec={swapJob.elapsedSec}
                  onCancel={() => void api.cancelJob("ai-faceswap")}
                />
              )}
              <p className={hint}>Only swap faces of people who agreed to it.</p>
            </>
          ) : (
            <p className="mt-1 text-neutral-400">
              Needs the Face Swap model. Download it in {modelsLink}.
            </p>
          )}
        </div>
        )}

        {/* Sub-card: AI Layers Stack */}
        {p.dev?.ai_removals && p.dev.ai_removals.length > 0 && (
          <div className="mt-2.5 space-y-1.5 rounded-md border border-neutral-800/90 bg-neutral-900/40 p-2.5">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>AI Retouch Layers ({p.dev.ai_removals.length})</span>
              <span className="font-normal normal-case text-neutral-500">Toggle or remove</span>
            </div>
            <div className="max-h-48 space-y-1 overflow-y-auto pr-0.5">
              {p.dev.ai_removals.map((r, i) => (
                <div
                  key={r.id}
                  className="group flex items-center justify-between rounded border border-neutral-800 bg-neutral-950/70 px-2 py-1 text-[11px] hover:border-neutral-700"
                >
                  <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2">
                    <input
                      type="checkbox"
                      checked={r.enabled}
                      onChange={() => toggleRemoval(r.id)}
                      className="rounded border-neutral-700 text-sky-500 focus:ring-0"
                    />
                    <span className={`truncate ${r.enabled ? "text-neutral-200" : "text-neutral-500 line-through"}`}>
                      {r.label || `Removal ${i + 1}`}
                    </span>
                  </label>
                  <div className="ml-2 flex shrink-0 items-center gap-1.5">
                    <span className="rounded bg-neutral-800 px-1 py-0.5 text-[9px] text-neutral-400">{r.model}</span>
                    <button
                      type="button"
                      onClick={() => void deleteRemoval(r.id)}
                      className="rounded p-0.5 text-neutral-500 hover:text-red-400"
                      title="Delete this layer"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Panel>
      )}
    </div>
  );
}
