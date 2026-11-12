import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import { AiHardware, AiModel, api, Settings } from "./api";
import { Btn, HUE, Icon } from "./controls";

const hint = "text-[10px] text-neutral-500";

type SettingModelKey = "ai_align_model" | "ai_denoise_model" | "ai_removal_model" | "ai_select_model" | "ai_concept_model";
type CategoryFilter = "all" | "geometry" | "denoise" | "retouch" | "masks" | "people" | "installed";

/** Settings → AI models: hardware summary, category filter bar, and organized AI model groups
 *  matching the Develop → AI workflow order (Tone & Geometry, Detail & Denoise, Retouch & Fill, People & Faces). */
export function AiModelsSettings({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const [hw, setHw] = useState<AiHardware | null>(null);
  const [models, setModels] = useState<AiModel[]>([]);
  const [dl, setDl] = useState<{ id: string; done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<CategoryFilter>("all");

  const load = () => api.aiModels().then((r) => (setHw(r.hardware), setModels(r.models)));
  useEffect(() => void load().catch((e) => setError(String(e))), []);
  useEffect(() => {
    const un = listen<{ id: string; done: number; total: number }>("ai-model-progress", (e) => setDl(e.payload));
    return () => void un.then((f) => f());
  }, []);

  const pick = (key: SettingModelKey, id: string) => {
    const updated = { ...settings, [key]: id };
    setSettings(updated);
    void api.saveSettings(updated);
  };
  const download = async (id: string) => {
    setError(null);
    setDl({ id, done: 0, total: 1 });
    try {
      await api.aiDownload(id);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setDl(null);
    }
  };
  const remove = (id: string) => void api.aiRemove(id).then(load);

  const fmtSize = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`);
  const gb = (mb: number) => `${(mb / 1024).toFixed(0)} GB`;

  // the model a tool uses: the one chosen here if it is installed, else the best installed one
  const inUse = (group: AiModel[], chosen?: string) =>
    (group.find((m) => m.id === chosen && m.installed) ?? group.find((m) => m.installed))?.id;

  const installedModels = models.filter((m) => m.installed);
  const installedMb = installedModels.reduce((acc, m) => acc + m.size_mb, 0);

  const renderModelList = (group: AiModel[], key?: SettingModelKey) => {
    const visible = filter === "installed" ? group.filter((m) => m.installed) : group;
    if (visible.length === 0) {
      return <p className="text-[11px] text-neutral-500">No installed models in this group yet.</p>;
    }
    const current = inUse(group, key && settings[key]);
    return (
      <div className="space-y-2">
        {visible.map((m) => {
          const busy = dl?.id === m.id;
          const pct = busy ? Math.round((dl!.done / Math.max(1, dl!.total)) * 100) : 0;
          const isCurrent = current === m.id;
          return (
            <div
              key={m.id}
              className={`rounded-md border px-3 py-2.5 transition-colors ${
                isCurrent
                  ? "border-sky-600/80 bg-sky-950/25"
                  : m.installed
                  ? "border-neutral-700/80 bg-neutral-900/70"
                  : "border-neutral-800 bg-neutral-950/40"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-medium text-neutral-100">{m.name}</span>
                  {m.recommended && (
                    <span className="rounded bg-emerald-600/80 px-1.5 py-0.5 text-[9px] font-semibold text-white">
                      Recommended for your GPU
                    </span>
                  )}
                  {!m.fits && (
                    <span className="rounded bg-amber-600/70 px-1.5 py-0.5 text-[9px] text-white">Needs more memory</span>
                  )}
                  {isCurrent && (
                    <span className="rounded bg-sky-600/90 px-1.5 py-0.5 text-[9px] font-semibold text-white">Active</span>
                  )}
                  {m.installed && !isCurrent && (
                    <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[9px] text-neutral-300">Installed</span>
                  )}
                </div>

                <div className="flex items-center gap-1.5">
                  {m.installed ? (
                    <>
                      {!isCurrent && key && (
                        <Btn active onClick={() => pick(key, m.id)}>
                          Use this model
                        </Btn>
                      )}
                      <Btn disabled={!!dl} onClick={() => remove(m.id)} title="Delete the downloaded files">
                        Remove
                      </Btn>
                    </>
                  ) : (
                    <Btn active={busy} disabled={!!dl} onClick={() => void download(m.id)} title={`Downloads from ${m.source}`}>
                      {busy ? `Downloading… ${pct}%` : `Download (${fmtSize(m.size_mb)})`}
                    </Btn>
                  )}
                </div>
              </div>

              <div className="mt-1 text-neutral-300">{m.blurb}</div>
              <div className={`${hint} mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5`}>
                <span className="text-neutral-400">{m.note}</span>
                <span>·</span>
                <span>{fmtSize(m.size_mb)}</span>
                <span>·</span>
                <span>{m.license}</span>
                <span>·</span>
                <span className="font-mono text-[9px] text-neutral-500">{m.source}</span>
              </div>

              {busy && (
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-neutral-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-200"
                    style={{ width: `${Math.max(4, Math.min(100, pct))}%` }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const alignModels = models.filter((m) => m.kind === "align");
  const denoiseModels = models.filter((m) => m.kind === "denoise");
  const removalModels = models.filter((m) => m.kind === "inpaint" || m.kind === "generate");
  const faceRecModels = models.filter((m) => m.kind === "faces");
  const faceSwapModels = models.filter((m) => m.kind === "faceswap");
  const landmarkModels = models.filter((m) => m.kind === "landmarks");
  const clickModels = models.filter((m) => m.kind === "segment");
  const conceptModels = models.filter((m) => m.kind === "concept");

  const showGeometry = filter === "all" || filter === "geometry" || filter === "installed";
  const showDenoise = filter === "all" || filter === "denoise" || filter === "installed";
  const showRetouch = filter === "all" || filter === "retouch" || filter === "installed";
  const showPeople = filter === "all" || filter === "people" || filter === "installed";
  const showMasks = filter === "all" || filter === "masks" || filter === "installed";

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-4xl space-y-5 text-[11px] text-neutral-300">
        {/* Hardware & Summary Header */}
        <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className={HUE.blue.text}>
                  <Icon d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7z" />
                </span>
                <h2 className="text-xs font-semibold text-neutral-100">Local AI Hardware &amp; Model Manager</h2>
              </div>
              {hw ? (
                <div className="mt-1.5">
                  <div className="font-medium text-neutral-100">{hw.gpu ?? "CPU Execution"}</div>
                  <div className="text-neutral-400">
                    {hw.vram_mb ? `${gb(hw.vram_mb)} memory · ` : ""}Inference runs on <span className="text-neutral-200">{hw.device}</span>
                  </div>
                </div>
              ) : (
                <p className="mt-1 text-neutral-500">Checking hardware capabilities…</p>
              )}
            </div>

            <div className="flex items-center gap-3 rounded-md border border-neutral-800 bg-neutral-950/60 px-3 py-2 text-right">
              <div>
                <div className="text-[9px] uppercase tracking-wider text-neutral-500">Installed Models</div>
                <div className="font-mono text-xs font-semibold text-neutral-100">
                  {installedModels.length} / {models.length} <span className="text-neutral-500">+ 1 built-in</span>
                </div>
              </div>
              <div className="h-6 w-px bg-neutral-800" />
              <div>
                <div className="text-[9px] uppercase tracking-wider text-neutral-500">Disk Usage</div>
                <div className="font-mono text-xs font-semibold text-sky-400">{fmtSize(installedMb + 2)}</div>
              </div>
            </div>
          </div>

          <p className={`${hint} mt-2`}>
            All neural models run 100% locally on your computer. Nothing downloads until you click Download, and every weight file is verified with SHA-256.
          </p>

          {/* Category filter pills */}
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-neutral-800 pt-3">
            {(
              [
                ["all", "All Categories"],
                ["geometry", "Tone & Geometry"],
                ["denoise", "Detail & Denoise"],
                ["retouch", "Retouch & Fill"],
                ["masks", "Selection & Masks"],
                ["people", "People & Faces"],
                ["installed", `Installed (${installedModels.length + 1})`],
              ] as [CategoryFilter, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setFilter(id)}
                className={`rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors ${
                  filter === id
                    ? "bg-sky-600 text-white"
                    : "bg-neutral-800/80 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="rounded-lg border border-red-800/70 bg-red-950/30 p-3 text-red-300">
            <div className="font-semibold">Model operation failed</div>
            <p className="mt-0.5 break-words text-[10px]">{error}</p>
          </div>
        )}

        {/* GROUP 1: TONE, COLOR & GEOMETRY */}
        {showGeometry && (
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className={HUE.cyan.text}>
                  <Icon d="M3 12h18M12 3v18M5 5l14 0 0 14-14 0z" />
                </span>
                <div>
                  <h2 className="text-xs font-semibold text-neutral-100">1. Tone, Color &amp; Geometry</h2>
                  <p className={hint}>Used by Develop → AI → Auto Adjust &amp; Auto Align</p>
                </div>
              </div>
              <span className="rounded bg-cyan-500/15 px-2 py-0.5 text-[10px] font-medium text-cyan-300">
                Tone &amp; Perspective
              </span>
            </div>

            {/* Sub-section: Built-in Auto Adjust */}
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <h3 className="font-semibold text-neutral-200">Auto Adjust · Tone &amp; Color</h3>
                <span className="text-[10px] text-neutral-500">Develop → AI → Tone &amp; Color</span>
              </div>
              <div className="rounded-md border border-neutral-800 bg-neutral-950/50 px-3 py-2.5">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-medium text-neutral-100">Image-Adaptive 3D-LUT · Photo Retouch</span>
                  <span className="rounded bg-sky-600/80 px-1.5 py-0.5 text-[9px] font-semibold text-white">Built in</span>
                </div>
                <div className="mt-1 text-neutral-300">
                  Predicts an image-adaptive 3D color lookup table trained on expert photographers&apos; edits, then maps it onto non-destructive Develop sliders.
                </div>
                <div className={`${hint} mt-1`}>
                  2 MB · Apache-2.0 (Zeng et al.) · trained on MIT-Adobe FiveK
                </div>
              </div>
            </div>

            {/* Sub-section: PerspectiveFields Auto Align */}
            <div>
              <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="font-semibold text-neutral-200">
                    Auto Align · Horizon &amp; Verticals (PerspectiveFields)
                  </h3>
                  <p className={hint}>
                    Predicts dense per-pixel gravity (roll) &amp; latitude (pitch) fields to auto-level horizons and straighten architectural verticals.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-1 text-[9px]">
                  <span className="rounded border border-cyan-800/60 bg-cyan-950/40 px-1.5 py-0.5 text-cyan-300">
                    Dense Gravity &amp; Roll Vector
                  </span>
                  <span className="rounded border border-neutral-700 bg-neutral-800 px-1.5 py-0.5 text-neutral-300">
                    ~0.1° – 0.5°
                  </span>
                  <span className="rounded border border-neutral-700 bg-neutral-800 px-1.5 py-0.5 text-neutral-300">
                    ConvNeXt / SegFormer
                  </span>
                  <span className="rounded border border-neutral-700 bg-neutral-800 px-1.5 py-0.5 text-neutral-400">
                    PyTorch-to-ONNX
                  </span>
                </div>
              </div>
              {renderModelList(alignModels, "ai_align_model")}
            </div>
          </section>
        )}

        {/* GROUP 2: DETAIL & RESTORATION */}
        {showDenoise && (
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className={HUE.emerald.text}>
                  <Icon d="M4 6h4v4H4zM10 6h4v4h-4zM16 6h4v4h-4zM4 14h4v4H4zM10 14h4v4h-4zM16 14h4v4h-4z" />
                </span>
                <div>
                  <h2 className="text-xs font-semibold text-neutral-100">2. Detail &amp; Restoration · Neural Denoise</h2>
                  <p className={hint}>
                    Used by Develop → AI → Detail · Neural Denoise. Cleans sensor noise and grain as a non-destructive in-place layer with adjustable strength.
                  </p>
                </div>
              </div>
              <span className="rounded bg-emerald-500/15 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
                Non-destructive layer
              </span>
            </div>
            {renderModelList(denoiseModels, "ai_denoise_model")}
          </section>
        )}

        {/* GROUP 3: RETOUCH & GENERATIVE INPAINTING */}
        {showRetouch && (
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className={HUE.violet.text}>
                  <Icon d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
                </span>
                <div>
                  <h2 className="text-xs font-semibold text-neutral-100">3. Retouch &amp; Generative Inpainting</h2>
                  <p className={hint}>
                    Used by Develop → AI → Retouch &amp; Faces. Generative fill synthesizes new detail into painted regions; fast inpainting continues surrounding textures.
                  </p>
                </div>
              </div>
              <span className="rounded bg-violet-500/15 px-2 py-0.5 text-[10px] font-medium text-violet-300">
                Inpaint &amp; Diffusion
              </span>
            </div>
            {renderModelList(removalModels, "ai_removal_model")}
          </section>
        )}

        {/* GROUP 4: PEOPLE & FACES */}
        {showPeople && (
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className={HUE.rose.text}>
                  <Icon d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
                </span>
                <div>
                  <h2 className="text-xs font-semibold text-neutral-100">4. People &amp; Faces</h2>
                  <p className={hint}>Used by Library → People view and Develop → AI → Face Swap</p>
                </div>
              </div>
              <span className="rounded bg-rose-500/15 px-2 py-0.5 text-[10px] font-medium text-rose-300">
                Recognition &amp; Swap
              </span>
            </div>

            {/* People indexing toggle */}
            <div className="rounded-md border border-neutral-800 bg-neutral-950/50 p-3">
              <label className="flex cursor-pointer items-center gap-2 font-semibold text-neutral-100">
                <input
                  type="checkbox"
                  checked={!!settings.faces_enabled}
                  onChange={(e) => {
                    const updated = { ...settings, faces_enabled: e.target.checked };
                    setSettings(updated);
                    void api.saveSettings(updated);
                  }}
                  className="rounded border-neutral-700 text-sky-500 focus:ring-0"
                />
                <span>Enable People view · Find faces and search library by person</span>
              </label>
              <p className={`${hint} mt-1`}>
                Detects faces with YOLOFace and groups them into people with ArcFace automatically on import. Runs locally on this computer and uses the Face Recognition model below (or the Face Swap bundle, which includes it).
              </p>
            </div>

            <div>
              <h3 className="mb-1.5 font-semibold text-neutral-200">Face Recognition (Library → People)</h3>
              {renderModelList(faceRecModels)}
            </div>

            <div>
              <h3 className="mb-1.5 font-semibold text-neutral-200">Face Swap (Develop → AI → Face Swap)</h3>
              {renderModelList(faceSwapModels)}
            </div>

            <div>
              <h3 className="mb-1.5 font-semibold text-neutral-200">Closed Eyes (Auto → culling, Develop → Focus)</h3>
              <p className={`${hint} mb-1.5`}>
                Reads each scanned face&apos;s eyelids from 68 landmarks: a blink is a face whose eye aspect ratio falls under 0.2. Without it, nothing is judged about eyes.
              </p>
              {renderModelList(landmarkModels)}
            </div>
          </section>
        )}

        {/* GROUP 5: SELECTION & MASKING (SEGMENT ANYTHING) */}
        {showMasks && (
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className={HUE.violet.text}>
                  <Icon d="M4 4h6v6H4zM14 14h6v6h-6zM14 4l6 6M4 14l6 6" />
                </span>
                <div>
                  <h2 className="text-xs font-semibold text-neutral-100">5. Selection &amp; Masking · Segment Anything</h2>
                  <p className={hint}>Used by Develop → Masks (click and select by name) and Auto → Masks. Runs on this computer: nothing is uploaded.</p>
                </div>
              </div>
              <span className="rounded bg-violet-500/15 px-2 py-0.5 text-[10px] font-medium text-violet-300">Local masks</span>
            </div>
            <div>
              <h3 className="mb-1.5 font-semibold text-neutral-200">Click to select · SAM 2.1</h3>
              <p className={`${hint} mb-1.5`}>Click an object: it is selected whole (Shift-click: just the part). After the first click on a photo, each one takes milliseconds.</p>
              {renderModelList(clickModels, "ai_select_model")}
            </div>
            <div>
              <h3 className="mb-1.5 font-semibold text-neutral-200">Select by name &amp; Auto masks · SAM 3</h3>
              <p className={`${hint} mb-1.5`}>Type what to select (&ldquo;sky&rdquo;, &ldquo;person&rdquo;, &ldquo;red car&rdquo;): every instance is found. Auto masks use it for subject, sky, people and background.</p>
              {renderModelList(conceptModels, "ai_concept_model")}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
