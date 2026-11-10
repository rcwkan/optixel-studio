import { ask, open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { useEffect, useState } from "react";
import { Agent, Log } from "./Agent";
import { api, AppInfo, CleanUp, JobProgress, Settings, size, Source, StorageMoveMode, StorageReport, StorageValidation, UpdateInfo } from "./api";
import { Btn } from "./controls";
import { Duplicate, Locations } from "./Storage";
import { AiModelsSettings } from "./AiModels";
import { applyTheme, getStoredTheme, isSystemDark, Theme, THEMES } from "./theme";

const card = "rounded-lg border border-neutral-800 bg-neutral-900 p-4";
const heading = "mb-1 text-[11px] font-semibold uppercase tracking-wider text-neutral-400";

const num = (n: number) => n.toLocaleString();
const plural = (n: number, word: string) => `${num(n)} ${word}${n === 1 ? "" : "s"}`;
const short = (folder: string) => folder.split(/[\\/]/).slice(-2).join("/");

export type SettingsTab = "appearance" | "locations" | "duplicate" | "storage" | "usage" | "ai" | "agent" | "about";
const TABS: [SettingsTab, string][] = [
  ["appearance", "Appearance"],
  ["locations", "Locations"],
  ["duplicate", "Duplicate"],
  ["storage", "Storage"],
  ["ai", "AI models"],
  ["agent", "Agent & MCP"],
  ["about", "About"],
];

/** Settings: Appearance & theme selection, storage locations, duplicate detection, library clean-up, and Agent & MCP configuration. */
export function SettingsView(p: {
  tab: SettingsTab;
  setTab: (t: SettingsTab) => void;
  settings: Settings | null;
  setSettings: (s: Settings) => void;
  selected: number[];
  jobs: Record<string, JobProgress>;
  onConnect: (s: Source) => void;
  onAuto: () => void;
  info?: AppInfo | null;
  logs?: Log[];
  clearLogs?: () => void;
  onCheckUpdate?: () => void;
  updateAvailable?: boolean;
  updateInfo?: UpdateInfo | null;
  updateLoading?: boolean;
}) {
  const reload = () => void api.settings().then(p.setSettings);
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <nav className="flex shrink-0 items-center gap-1 border-b border-neutral-800 bg-neutral-900 px-4 py-1.5 text-[12px]">
        <span className="mr-3 font-semibold text-neutral-300">Settings</span>
        {TABS.map(([t, label]) => (
          <button key={t} onClick={() => p.setTab(t)} className={`flex items-center gap-1.5 rounded px-3 py-1 ${p.tab === t ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-neutral-200"}`}>
            <span>{label}</span>
            {t === "about" && p.updateAvailable && (
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--theme-accent,#0ea5e9)] animate-pulse" />
            )}
          </button>
        ))}
      </nav>
      {p.settings && p.tab === "appearance" && <Appearance settings={p.settings} setSettings={p.setSettings} />}
      {p.settings && p.tab === "locations" && <Locations settings={p.settings} setSettings={p.setSettings} onConnect={p.onConnect} onAuto={p.onAuto} />}
      {p.settings && p.tab === "duplicate" && <Duplicate settings={p.settings} selected={p.selected} jobs={p.jobs} />}
      {p.settings && p.tab === "ai" && <AiModelsSettings settings={p.settings} setSettings={p.setSettings} />}
      {(p.tab === "storage" || p.tab === "usage") && <Usage onSettingsChanged={reload} onGoToAiModels={() => p.setTab("ai")} />}
      {p.tab === "agent" && (
        <Agent
          info={p.info ?? null}
          settings={p.settings}
          setSettings={p.setSettings}
          logs={p.logs ?? []}
          clearLogs={p.clearLogs ?? (() => { })}
        />
      )}
      {p.tab === "about" && (
        <About
          settings={p.settings}
          setSettings={p.setSettings}
          onCheckUpdate={p.onCheckUpdate}
          updateAvailable={p.updateAvailable}
          updateInfo={p.updateInfo}
          updateLoading={p.updateLoading}
        />
      )}
    </div>
  );
}

/** GPU acceleration: on by default, effective only when a GPU is detected. */
function GpuToggle({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const [gpu, setGpu] = useState<string | null | undefined>(undefined);
  useEffect(() => void api.gpuInfo().then(setGpu).catch(() => setGpu(null)), []);
  const on = settings.gpu_accel !== false;
  const change = (v: boolean) => {
    const updated = { ...settings, gpu_accel: v };
    setSettings(updated);
    api.saveSettings(updated);
  };
  return (
    <label className="flex items-center gap-3 rounded-lg border border-neutral-800 bg-neutral-900 p-3 text-xs text-neutral-300">
      <input type="checkbox" checked={on && !!gpu} disabled={!gpu} onChange={(e) => change(e.target.checked)} />
      <span className="font-semibold text-neutral-100">GPU acceleration</span>
      <span className="text-neutral-400">{gpu === undefined ? "Detecting GPU…" : gpu ? `Detected: ${gpu}` : "No compatible GPU found, using CPU"}</span>
    </label>
  );
}

/** Appearance settings: theme selection (9 distinct palettes including Studio Dark, Light, Cool Slate, Warm Darkroom, Nordic Frost, Midnight Violet, Forest Sage, OLED Obsidian, and System OS sync),
 *  with category filtering, dynamic accent glowing cards, and live previews. */
function Appearance({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const currentTheme: Theme = (settings.theme as Theme) || getStoredTheme();
  const [activeTheme, setActiveTheme] = useState<Theme>(currentTheme);
  const [previewExposure, setPreviewExposure] = useState<number>(0.35);
  const [categoryFilter, setCategoryFilter] = useState<string>("All");
  const sysDark = isSystemDark();

  const handleSelect = (t: Theme) => {
    setActiveTheme(t);
    applyTheme(t);
    const updated: Settings = { ...settings, theme: t };
    setSettings(updated);
    api.saveSettings(updated);
  };

  const categories = ["All", "Dark", "Light", "Neutral", "Warm", "Modern", "Vibrant"];
  const filteredThemes = THEMES.filter((th) => {
    if (categoryFilter === "All") return true;
    if (categoryFilter === "Dark") return th.mode === "dark";
    if (categoryFilter === "Light") return th.mode === "light";
    if (categoryFilter === "Neutral") return th.category === "Neutral" || th.id === "system";
    return th.category === categoryFilter;
  });

  const activeThemeMeta = THEMES.find((t) => t.id === activeTheme) || THEMES[0];

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <GpuToggle settings={settings} setSettings={setSettings} />
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
              <span>Appearance & Color Themes</span>
              <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] font-medium text-neutral-400">
                {THEMES.length} Themes
              </span>
            </h2>
            <p className="mt-0.5 text-xs text-neutral-400">
              Modern color palettes crafted for studio color grading, analog warmth, high-contrast daylight, and HDR/OLED displays.
            </p>
          </div>

          {/* Operating system status banner */}
          <div className="flex items-center gap-3 rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-1.5 text-xs">
            <span className="flex items-center gap-1.5 text-neutral-300">
              <span>🖥️ OS:</span>
              <strong className="text-neutral-100 font-medium">{sysDark ? "Dark" : "Light"}</strong>
            </span>
            {activeTheme !== "system" && (
              <button
                onClick={() => handleSelect("system")}
                className="text-[11px] font-medium text-sky-400 hover:text-sky-300 transition-colors"
                title="Synchronize appearance with your system appearance settings"
              >
                Sync OS
              </button>
            )}
          </div>
        </div>

        {/* Category Filter Pills */}
        <div className="flex items-center gap-1.5 border-b border-neutral-800/80 pb-3 overflow-x-auto">
          {categories.map((cat) => {
            const count =
              cat === "All"
                ? THEMES.length
                : cat === "Dark"
                ? THEMES.filter((t) => t.mode === "dark").length
                : cat === "Light"
                ? THEMES.filter((t) => t.mode === "light").length
                : cat === "Neutral"
                ? THEMES.filter((t) => t.category === "Neutral" || t.id === "system").length
                : THEMES.filter((t) => t.category === cat).length;
            const isCur = categoryFilter === cat;
            return (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-all ${isCur
                    ? "bg-neutral-100 text-neutral-900 shadow-sm"
                    : "bg-neutral-900/80 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200 border border-neutral-800"
                  }`}
              >
                <span>{cat}</span>
                <span className={`text-[10px] px-1 rounded-full ${isCur ? "bg-neutral-300/80 text-neutral-900" : "bg-neutral-800 text-neutral-400"}`}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Theme selection grid */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredThemes.map((th) => {
            const isSelected = activeTheme === th.id;
            return (
              <div
                key={th.id}
                onClick={() => handleSelect(th.id)}
                style={
                  isSelected
                    ? {
                      borderColor: th.colors.accent,
                      boxShadow: `0 0 18px ${th.colors.accentGlow}`,
                    }
                    : undefined
                }
                className={`group relative flex cursor-pointer flex-col justify-between rounded-xl border p-4 transition-all duration-200 ${isSelected
                    ? "bg-neutral-900/90 ring-1"
                    : "border-neutral-800/90 bg-neutral-900/40 hover:border-neutral-700 hover:bg-neutral-900/80 hover:shadow-md"
                  }`}
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2.5">
                      <div
                        className="flex h-8 w-8 items-center justify-center rounded-lg border border-neutral-700/60 text-base shadow-xs"
                        style={{ backgroundColor: th.colors.card }}
                      >
                        {th.icon}
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-neutral-100 group-hover:text-white flex items-center gap-1.5">
                          {th.label}
                        </div>
                        <div className="text-[10px] font-medium" style={{ color: th.colors.accent }}>
                          {th.tagline}
                        </div>
                      </div>
                    </div>

                    {isSelected ? (
                      <span
                        className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide shadow-xs"
                        style={{
                          backgroundColor: th.colors.accent,
                          color: "#ffffff",
                        }}
                      >
                        ✓ Active
                      </span>
                    ) : (
                      <span
                        className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                          th.mode === "light"
                            ? "bg-amber-500/10 text-amber-500 border border-amber-500/20"
                            : th.mode === "dark"
                            ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                            : "bg-neutral-800/90 text-neutral-400"
                        }`}
                      >
                        <span>{th.mode === "light" ? "☀️" : th.mode === "dark" ? "🌙" : "💻"}</span>
                        <span>{th.mode === "system" ? "Auto" : th.mode === "light" ? "Light" : "Dark"}</span>
                      </span>
                    )}
                  </div>

                  <p className="mt-3 text-[11px] leading-relaxed text-neutral-400">
                    {th.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-neutral-800/80">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-neutral-500 font-mono uppercase tracking-wider">Palette</span>
                    <div className="flex items-center gap-1.5">
                      <span
                        className="h-3.5 w-3.5 rounded-full border border-neutral-700/80 shadow-xs"
                        style={{ backgroundColor: th.colors.bg }}
                        title={`Canvas Canvas: ${th.colors.bg}`}
                      />
                      <span
                        className="h-3.5 w-3.5 rounded-full border border-neutral-700/80 shadow-xs"
                        style={{ backgroundColor: th.colors.card }}
                        title={`Panel Surface: ${th.colors.card}`}
                      />
                      <span
                        className="h-3.5 w-3.5 rounded-full border border-neutral-700/80 shadow-xs"
                        style={{ backgroundColor: th.colors.text }}
                        title={`Primary Text: ${th.colors.text}`}
                      />
                      <span
                        className="h-4 w-4 rounded-full border border-white/20 shadow-sm ring-1 ring-white/10"
                        style={{ backgroundColor: th.colors.accent, boxShadow: `0 0 8px ${th.colors.accentGlow}` }}
                        title={`Signature Accent: ${th.colors.accent}`}
                      />
                      {th.colors.secondaryAccent && (
                        <span
                          className="h-3 w-3 rounded-full border border-white/15 shadow-xs"
                          style={{ backgroundColor: th.colors.secondaryAccent }}
                          title={`Secondary Accent: ${th.colors.secondaryAccent}`}
                        />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Live Interface Preview */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/90 p-5 shadow-lg space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between border-b border-neutral-800 pb-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">
                  Live Theme Preview
                </h3>
                <span
                  className="rounded px-2 py-0.5 text-[10px] font-semibold"
                  style={{
                    backgroundColor: activeThemeMeta.colors.accent,
                    color: activeThemeMeta.colors.bg.startsWith("#f") ? "#0f172a" : "#ffffff",
                  }}
                >
                  {activeThemeMeta.label}
                </span>
              </div>
              <p className="text-[11px] text-neutral-400 mt-0.5">
                {activeThemeMeta.tagline} — Active studio color tokens rendered in real time.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-neutral-400">
              <span className="flex items-center gap-1.5 rounded bg-neutral-800 px-2 py-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>Engine Ready</span>
              </span>
            </div>
          </div>

          <div className="space-y-4 rounded-lg border border-neutral-800 bg-neutral-950 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <button className="rounded bg-neutral-800 px-3 py-1.5 text-xs font-medium text-neutral-200 border border-neutral-700/60 hover:bg-neutral-700">
                  Default Button
                </button>
                <button
                  className="rounded px-3 py-1.5 text-xs font-medium shadow-md transition-shadow"
                  style={{
                    backgroundColor: "var(--theme-accent, #0ea5e9)",
                    color: "var(--theme-accent-contrast, #ffffff)",
                    boxShadow: "0 0 12px var(--theme-accent-glow, rgba(14,165,233,0.3))",
                  }}
                >
                  Active Accent Button
                </button>
                <button
                  className="rounded border px-3 py-1.5 text-xs font-medium"
                  style={{
                    borderColor: "var(--theme-accent, #0ea5e9)",
                    color: "var(--theme-accent, #0ea5e9)",
                    backgroundColor: "var(--theme-accent-subtle, rgba(14,165,233,0.1))",
                  }}
                >
                  Subtle Accent Pill
                </button>
              </div>

              {/* Color Labels Preview */}
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-neutral-500 mr-1">Labels:</span>
                <span className="h-3.5 w-3.5 rounded-full bg-red-500 shadow-xs" title="Red label" />
                <span className="h-3.5 w-3.5 rounded-full bg-amber-400 shadow-xs" title="Yellow label" />
                <span className="h-3.5 w-3.5 rounded-full bg-emerald-500 shadow-xs" title="Green label" />
                <span className="h-3.5 w-3.5 rounded-full bg-sky-500 shadow-xs" title="Blue label" />
                <span className="h-3.5 w-3.5 rounded-full bg-purple-500 shadow-xs" title="Purple label" />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              <div className="rounded-md border border-neutral-800/80 bg-neutral-900/60 p-3">
                <div className="mb-1 flex justify-between text-xs text-neutral-400">
                  <span>Interactive Exposure Slider</span>
                  <span className="font-mono text-neutral-200 font-medium">
                    {previewExposure > 0 ? `+${previewExposure.toFixed(2)}` : previewExposure.toFixed(2)} EV
                  </span>
                </div>
                <input
                  type="range"
                  className="ol-range w-full"
                  min={-3}
                  max={3}
                  step={0.05}
                  value={previewExposure}
                  onChange={(e) => setPreviewExposure(+e.target.value)}
                />
              </div>

              <div className="flex items-center justify-between rounded-md border border-neutral-800/80 bg-neutral-900/60 p-3 text-xs text-neutral-400">
                <div className="flex items-center gap-2">
                  <span className="text-base">{activeThemeMeta.icon}</span>
                  <div>
                    <div className="font-medium text-neutral-200">{activeThemeMeta.label}</div>
                    <div className="text-[10px] text-neutral-500">Contrast ratio compliant (WCAG AA)</div>
                  </div>
                </div>
                <span
                  className="h-5 w-5 rounded-full border border-white/20 shadow-xs"
                  style={{ backgroundColor: activeThemeMeta.colors.accent }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Visual horizontal bar for statistics with percentage and count. */
function StatBar({
  label,
  count,
  total,
  color = "bg-sky-500",
  extra,
}: {
  label: string | React.ReactNode;
  count: number;
  total: number;
  color?: string;
  extra?: string;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="group flex items-center gap-2 py-0.5 text-[11px]">
      <div className="w-32 shrink-0 truncate text-neutral-300" title={typeof label === "string" ? label : undefined}>
        {label}
      </div>
      <div className="relative h-2 flex-1 rounded-full bg-neutral-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${Math.max(pct > 0 ? 1 : 0, Math.min(100, pct))}%` }}
        />
      </div>
      <div className="w-24 shrink-0 text-right tabular-nums text-neutral-400">
        <span>{num(count)}</span>
        <span className="ml-1 text-[10px] text-neutral-500">({pct.toFixed(1)}%)</span>
      </div>
      {extra && <span className="text-[10px] text-neutral-500">{extra}</span>}
    </div>
  );
}

/** Star ratings 5 down to 1 and Unrated breakdown. */
function RatingBreakdown({
  ratings,
  total,
}: {
  ratings?: { rating: number; count: number }[];
  total: number;
}) {
  const map: Record<number, number> = { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0 };
  if (ratings) {
    for (const r of ratings) {
      if (r.rating in map) map[r.rating] = r.count;
      else map[0] = (map[0] || 0) + r.count;
    }
  }
  const rows = [
    { stars: 5, label: "★★★★★ 5 Stars", color: "bg-amber-400" },
    { stars: 4, label: "★★★★☆ 4 Stars", color: "bg-amber-400" },
    { stars: 3, label: "★★★☆☆ 3 Stars", color: "bg-amber-500" },
    { stars: 2, label: "★★☆☆☆ 2 Stars", color: "bg-amber-600" },
    { stars: 1, label: "★☆☆☆☆ 1 Star", color: "bg-amber-700" },
    { stars: 0, label: "☆☆☆☆☆ Unrated", color: "bg-neutral-600" },
  ];
  return (
    <div className="space-y-1">
      {rows.map((row) => (
        <StatBar
          key={row.stars}
          label={<span className={row.stars > 0 ? "text-amber-400 font-medium" : "text-neutral-500"}>{row.label}</span>}
          count={map[row.stars] || 0}
          total={total}
          color={row.color}
        />
      ))}
    </div>
  );
}

/** Segmented culling status bar: Picks (green), Unflagged (gray), Rejects (red). */
function CullingBar({ picks, rejects, total }: { picks: number; rejects: number; total: number }) {
  const unflagged = Math.max(0, total - picks - rejects);
  const picksPct = total > 0 ? (picks / total) * 100 : 0;
  const rejectsPct = total > 0 ? (rejects / total) * 100 : 0;
  const unflaggedPct = total > 0 ? (unflagged / total) * 100 : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex h-3 w-full overflow-hidden rounded bg-neutral-800">
        {picks > 0 && (
          <div
            className="bg-emerald-500 transition-all"
            style={{ width: `${picksPct}%` }}
            title={`Picks: ${num(picks)} (${picksPct.toFixed(1)}%)`}
          />
        )}
        {unflagged > 0 && (
          <div
            className="bg-neutral-700 transition-all"
            style={{ width: `${unflaggedPct}%` }}
            title={`Unflagged: ${num(unflagged)} (${unflaggedPct.toFixed(1)}%)`}
          />
        )}
        {rejects > 0 && (
          <div
            className="bg-rose-500 transition-all"
            style={{ width: `${rejectsPct}%` }}
            title={`Rejects: ${num(rejects)} (${rejectsPct.toFixed(1)}%)`}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-neutral-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          <span>Picks: <strong className="text-neutral-200">{num(picks)}</strong> ({picksPct.toFixed(1)}%)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-neutral-600" />
          <span>Unflagged: <strong className="text-neutral-200">{num(unflagged)}</strong> ({unflaggedPct.toFixed(1)}%)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-rose-500" />
          <span>Rejects: <strong className="text-neutral-200">{num(rejects)}</strong> ({rejectsPct.toFixed(1)}%)</span>
        </span>
      </div>
    </div>
  );
}

/** Capture timeline histogram (photos shot by year). */
function TimelineHistogram({
  years,
  total,
  height = "h-24",
}: {
  years?: { year: string; count: number }[];
  total: number;
  height?: string;
}) {
  if (!years || years.length === 0) {
    return <p className="text-[11px] text-neutral-500 italic">No capture dates recorded.</p>;
  }
  const sorted = [...years].sort((a, b) => a.year.localeCompare(b.year));
  const max = Math.max(1, ...sorted.map((y) => y.count));

  return (
    <div className="space-y-2">
      <div className={`flex ${height} items-end gap-1.5 rounded-lg border border-neutral-800/80 bg-neutral-950 p-2.5`}>
        {sorted.map((y) => {
          const heightPct = Math.max(8, Math.round((y.count / max) * 100));
          const pct = ((y.count / Math.max(1, total)) * 100).toFixed(1);
          return (
            <div
              key={y.year}
              className="group relative flex flex-1 flex-col items-center justify-end h-full min-w-0"
              title={`${y.year}: ${num(y.count)} photos (${pct}%)`}
            >
              <div className="absolute -top-6 opacity-0 group-hover:opacity-100 transition-opacity bg-neutral-800 px-1.5 py-0.5 rounded text-[9px] tabular-nums text-neutral-200 pointer-events-none z-10 whitespace-nowrap shadow-md border border-neutral-700">
                {num(y.count)} ({pct}%)
              </div>
              <div
                className="w-full max-w-[32px] rounded-t bg-sky-500/85 transition-all group-hover:bg-sky-400 group-hover:shadow-[0_0_8px_rgba(56,189,248,0.5)]"
                style={{ height: `${heightPct}%` }}
              />
              <span className="mt-1 truncate text-[9px] font-medium text-neutral-400 group-hover:text-neutral-200">
                {y.year.slice(-2)}'
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-neutral-500">
        <span>Oldest: {sorted[0]?.year}</span>
        <span>Timeline span: {sorted.length} years</span>
        <span>Latest: {sorted[sorted.length - 1]?.year}</span>
      </div>
    </div>
  );
}

/** RAW formats vs Standard formats distribution. */
function FormatsSection({
  rawCount,
  stdCount,
  formats,
  total,
}: {
  rawCount: number;
  stdCount: number;
  formats?: { ext: string; count: number }[];
  total: number;
}) {
  const totalFmt = Math.max(1, rawCount + stdCount);
  const rawPct = (rawCount / totalFmt) * 100;
  const stdPct = (stdCount / totalFmt) * 100;

  return (
    <div className="space-y-2">
      <div className="flex h-3 w-full overflow-hidden rounded bg-neutral-800">
        {rawCount > 0 && (
          <div
            className="bg-indigo-500 transition-all"
            style={{ width: `${rawPct}%` }}
            title={`RAW formats: ${num(rawCount)} (${rawPct.toFixed(1)}%)`}
          />
        )}
        {stdCount > 0 && (
          <div
            className="bg-teal-500 transition-all"
            style={{ width: `${stdPct}%` }}
            title={`Standard formats: ${num(stdCount)} (${stdPct.toFixed(1)}%)`}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px]">
        <span className="flex items-center gap-1.5 text-indigo-400 font-medium">
          <span className="h-2 w-2 rounded-full bg-indigo-500" />
          <span>RAW files: <strong className="text-neutral-200">{num(rawCount)}</strong> ({rawPct.toFixed(1)}%)</span>
        </span>
        <span className="flex items-center gap-1.5 text-teal-400 font-medium">
          <span className="h-2 w-2 rounded-full bg-teal-500" />
          <span>Standard (JPG, PNG, TIF): <strong className="text-neutral-200">{num(stdCount)}</strong> ({stdPct.toFixed(1)}%)</span>
        </span>
      </div>
      {formats && formats.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {formats.map((f) => (
            <span
              key={f.ext}
              className="inline-flex items-center gap-1 rounded bg-neutral-950 px-2 py-0.5 text-[10px] text-neutral-300 border border-neutral-800"
            >
              <strong className="text-neutral-100">{f.ext}</strong>
              <span className="text-neutral-500">· {num(f.count)}</span>
              <span className="text-neutral-600">({((f.count / Math.max(1, total)) * 100).toFixed(1)}%)</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

type AnalysisTab = "overview" | "culling" | "timeline" | "gear" | "exposure" | "folders";

/** Storage usage of the catalog folder, rich library analysis and clean-up. */
function Usage({ onSettingsChanged, onGoToAiModels }: { onSettingsChanged: () => void; onGoToAiModels?: () => void }) {
  const [r, setR] = useState<StorageReport | null>(null);
  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");
  const [showDetails, setShowDetails] = useState(false);
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("overview");
  const [folderSearch, setFolderSearch] = useState("");
  const [folderSort, setFolderSort] = useState<"bytes" | "photos" | "name">("bytes");
  const [pendingRelocate, setPendingRelocate] = useState<StorageValidation | null>(null);
  const [cleanupOld, setCleanupOld] = useState(false);
  const [reconnectBusy, setReconnectBusy] = useState(false);
  const [relocating, setRelocating] = useState<{ phase: string; file: string; done_bytes: number; total_bytes: number } | null>(null);
  const [regenProgress, setRegenProgress] = useState<{ done: number; total: number } | null>(null);

  useEffect(() => void api.storage().then(setR, (e) => setNote(`Could not measure storage: ${e}`)), []);

  useEffect(() => {
    const unlistenPromise = listen<{ phase: string; file?: string; done_bytes?: number; total_bytes?: number; message?: string }>(
      "storage-move-progress",
      (event) => {
        const p = event.payload;
        setRelocating((prev) => ({
          phase: p.phase || prev?.phase || "copying",
          file: p.file || prev?.file || "",
          done_bytes: p.done_bytes ?? prev?.done_bytes ?? 0,
          total_bytes: p.total_bytes ?? prev?.total_bytes ?? 0,
        }));
      }
    );
    const unlistenRegen = listen<{ done: number; total: number }>(
      "regenerate-thumbs-progress",
      (event) => {
        setRegenProgress(event.payload);
      }
    );
    return () => {
      void unlistenPromise.then((unlisten) => unlisten());
      void unlistenRegen.then((unlisten) => unlisten());
    };
  }, []);

  const handlePickNewLocation = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select New Storage Location for Optixel Studio",
      });
      if (!selected || typeof selected !== "string") return;
      const check = await api.validateStorageLocation(selected);
      if (!check.valid) {
        await ask(check.error || "The selected folder cannot be used.", {
          title: "Invalid Storage Location",
          kind: "error",
        });
        return;
      }
      setPendingRelocate(check);
      setCleanupOld(false);
    } catch (e) {
      setNote(`Location selection failed: ${e}`);
    }
  };

  const handleConfirmRelocate = async (mode: StorageMoveMode) => {
    if (!pendingRelocate?.target) return;
    const target = pendingRelocate.target;
    const isResettingToDefault = !pendingRelocate.is_custom || target === r?.default_data_dir;
    setRelocating({
      phase: "starting",
      file: "",
      done_bytes: 0,
      total_bytes: (mode === "copy" || mode === "move") ? (pendingRelocate.current_bytes ?? r?.total ?? 0) : 0,
    });
    setNote("");
    try {
      const res = isResettingToDefault
        ? await api.resetStorageLocation(mode, cleanupOld)
        : await api.moveStorageLocation(target, mode, cleanupOld);
      setPendingRelocate(null);
      setCleanupOld(false);
      onSettingsChanged();
      const next = await api.storage();
      setR(next);
      setNote(`Storage successfully updated: ${res.data_dir}. (${mode} mode)`);
    } catch (e) {
      setNote(`Failed to update storage: ${e}`);
    } finally {
      setRelocating(null);
    }
  };

  const handleResetLocation = async () => {
    if (!r?.default_data_dir) return;
    try {
      const check = await api.validateStorageLocation(r.default_data_dir);
      if (!check.valid) {
        await ask(check.error || "The default folder cannot be validated.", {
          title: "Invalid Storage Location",
          kind: "error",
        });
        return;
      }
      setPendingRelocate(check);
      setCleanupOld(false);
    } catch (e) {
      setNote(`Default location check failed: ${e}`);
    }
  };

  const handleSyncFolders = async () => {
    setReconnectBusy(true);
    setNote("Scanning library folders for photos and adjustments…");
    try {
      const count = await api.syncLibraryFolders();
      const next = await api.storage();
      setR(next);
      setNote(`Library scan complete: ${num(count)} photo(s) synchronized and reconnected.`);
      onSettingsChanged();
    } catch (e) {
      setNote(`Sync failed: ${e}`);
    } finally {
      setReconnectBusy(false);
    }
  };
  if (!r) return <p className="p-4 text-[12px] text-neutral-500">{note || "Measuring storage…"}</p>;

  const run = async (key: string, title: string, job: () => Promise<number>, confirm?: string) => {
    if (confirm && !(await ask(confirm, { title, kind: "warning" }))) return;
    setBusy(key);
    setNote("");
    try {
      const n = await job();
      if (key === "cloud") onSettingsChanged(); // the accounts forget their downloaded folders
      const next = await api.storage();
      setNote(`${title}: ${n ? `${num(n)} removed, ` : ""}${size(Math.max(0, r.total - next.total))} freed`);
      setR(next);
    } catch (e) {
      setNote(`${title} failed: ${e}`);
    }
    setBusy("");
  };
  const removeFolder = async (folder: string) => {
    const ids = (await api.query({ folder })).map((p) => p.id);
    const msg = `${folder} is not available. If it is on a drive that isn't connected, connect the drive instead.\n\nRemove its ${ids.length} photo(s) from the catalog? Their ratings, keywords and edit history are lost.`;
    run(folder, "Remove offline folder", () => api.remove(ids).then(() => ids.length), msg);
  };

  const s = r.summary;
  const totalPhotos = Math.max(1, s.total_photos);
  const sep = r.data_dir.includes("\\") ? "\\" : "/";
  const own = r.folders.filter((f) => !f.folder.startsWith(r.data_dir + sep));
  const google = r.folders.filter((f) => f.folder.startsWith(`${r.data_dir}${sep}remote${sep}Google Photos`)).reduce((n, f) => n + f.photos, 0);
  const immichAlbumsBytes = r.immich_albums_bytes ?? 0;
  const cloudDir = `${r.data_dir}${sep}remote${sep}Cloud`;
  const cloudPhotos = r.folders.filter((f) => f.folder === cloudDir || f.folder.startsWith(cloudDir + sep)).reduce((n, f) => n + f.photos, 0);
  const aiModelsBytes = r.ai_models ?? 0;
  const aiCacheBytes = r.ai_cache?.bytes ?? 0;
  const otherBytes = r.other ?? Math.max(0, r.total - (r.catalog + r.thumbs + r.originals + immichAlbumsBytes + r.google + (r.cloud ?? 0) + aiModelsBytes + aiCacheBytes));

  const parts: [string, number, string][] = [
    ["Catalog", r.catalog, "bg-sky-500"],
    ["Thumbnails", r.thumbs, "bg-violet-500"],
    ...(immichAlbumsBytes > 0 ? [["Downloaded Immich albums", immichAlbumsBytes, "bg-cyan-500"] as [string, number, string]] : []),
    ["Immich photo cache", r.originals, "bg-amber-500"],
    ["Google Photos picks", r.google, "bg-emerald-500"],
    ...((r.cloud ?? 0) > 0 ? [["Cloud downloads", r.cloud ?? 0, "bg-rose-500"] as [string, number, string]] : []),
    ...(aiModelsBytes > 0 ? [["Downloaded AI models", aiModelsBytes, "bg-purple-500"] as [string, number, string]] : []),
    ...(aiCacheBytes > 0 ? [["AI layers & patch cache", aiCacheBytes, "bg-pink-500"] as [string, number, string]] : []),
    ["Other", otherBytes, "bg-neutral-500"],
  ];

  const tiles: [string, number, string?, string?][] = [
    ["Photos", s.total_photos - s.virtual_copies, `${num(s.total_photos)} total in catalog`],
    ["Virtual copies", s.virtual_copies, s.virtual_copies > 0 ? `${((s.virtual_copies / totalPhotos) * 100).toFixed(1)}% of photos` : "No virtual copies"],
    ["Edited", s.edited, `${((s.edited / totalPhotos) * 100).toFixed(1)}% of catalog`],
    ["Edited with AI", r.ai.photos, `${num(r.ai.edits)} AI & MCP edits`],
    ["Picks", s.picks, `${((s.picks / totalPhotos) * 100).toFixed(1)}% of catalog`],
    ["Rejects", s.rejects, `${((s.rejects / totalPhotos) * 100).toFixed(1)}% of catalog`],
    ["Rated ★3+", s.rated_3_plus, `${((s.rated_3_plus / totalPhotos) * 100).toFixed(1)}% of catalog`],
    ["History steps", r.history_steps, s.history_stats ? `avg ${s.history_stats.avg_steps} · peak ${s.history_stats.max_steps}` : undefined],
  ];

  const c = r.clean;
  // key, title, what it is, amount, button, confirmation for what can't be recovered
  const cleans: [CleanUp, string, string, string, string, string?][] = [
    ["thumbs", "Orphaned thumbnails", "Previews of photos that are no longer in the catalog.", `${plural(c.thumbs.count, "file")} · ${size(c.thumbs.bytes)}`, "Delete"],
    ["originals", "Immich photo cache", "Kept after you open or export an Immich photo; downloaded again when needed while Immich is connected.", `${plural(c.originals.count, "file")} · ${size(c.originals.bytes)}`, "Clear"],
    ...(c.immich_albums && c.immich_albums.bytes > 0
      ? [
        [
          "immich_albums" as CleanUp,
          "Downloaded Immich albums",
          "Full albums downloaded from Immich. Photos remain safe on your Immich server.",
          `${plural(c.immich_albums.count, "photo")} · ${size(c.immich_albums.bytes)}`,
          "Clear",
          `Delete the downloaded Immich albums (${plural(c.immich_albums.count, "photo")}, ${size(c.immich_albums.bytes)})?\n\nTheir photos are also removed from the catalog, with any edits, ratings and keywords you made to them here. The originals stay on your Immich server, where you can browse them in place.`,
        ] as [CleanUp, string, string, string, string, string?],
      ]
      : []),
    ...(c.cloud && c.cloud.bytes > 0
      ? [
        [
          "cloud" as CleanUp,
          "Cloud downloads",
          "Folders downloaded from cloud storage. The originals stay in the cloud.",
          `${plural(c.cloud.count, "file")} · ${size(c.cloud.bytes)}`,
          "Clear",
          `Delete the downloaded cloud folders (${plural(c.cloud.count, "file")}, ${size(c.cloud.bytes)})?\n\nTheir photos are also removed from the catalog, with any edits, ratings and keywords you made to them here. The originals stay in your cloud storage, where you can download them again.`,
        ] as [CleanUp, string, string, string, string, string?],
      ]
      : []),
    ...(c.ai_models && (c.ai_models.count > 0 || c.ai_models.bytes > 0)
      ? [
        [
          "ai_models" as CleanUp,
          "Downloaded AI models",
          "Neural network weights for AI denoise and generative object removal.",
          `${plural(c.ai_models.count, "model")} · ${size(c.ai_models.bytes)}`,
          "Clear",
          `Delete all downloaded AI models (${plural(c.ai_models.count, "model")}, ${size(c.ai_models.bytes)})?\n\nYou can re-download them anytime from the AI models tab.`,
        ] as [CleanUp, string, string, string, string, string?],
      ]
      : []),
    ...(c.ai_cache && (c.ai_cache.count > 0 || c.ai_cache.bytes > 0)
      ? [
        [
          "ai_cache" as CleanUp,
          "AI layers & patch cache",
          "Rendered pixel patches for AI object removal and denoise. Safely re-rendered on demand.",
          `${plural(c.ai_cache.count, "patch")} · ${size(c.ai_cache.bytes)}`,
          "Clear",
          `Clear ${plural(c.ai_cache.count, "AI patch")} (${size(c.ai_cache.bytes)})?\n\nPhotos retain their AI adjustment layers in Develop history, and patches will re-render automatically on demand.`,
        ] as [CleanUp, string, string, string, string, string?],
      ]
      : []),
    [
      "missing",
      "Missing photos",
      "Catalog entries whose file was deleted or moved. Offline folders are left alone.",
      plural(c.missing.count, "photo"),
      "Remove",
      `Remove ${c.missing.count} missing photo(s) and their virtual copies from the catalog? Their ratings, keywords and edit history are lost.`,
    ],
    ["history", "Edit history", "Undo steps. Every photo keeps its current edits.", `${plural(c.history.count, "step")} · ${size(c.history.bytes)}`, "Clear", `Clear ${c.history.count} undo step(s)? Every photo keeps its current edits, but earlier steps can no longer be restored.`],
    ["compact", "Compact catalog", "Rewrites the catalog file without its unused space.", `${size(c.compact.bytes)} unused`, "Compact"],
  ];

  const filteredFolders = [...r.folders]
    .filter((f) => !folderSearch || f.folder.toLowerCase().includes(folderSearch.toLowerCase()))
    .sort((a, b) => {
      if (folderSort === "bytes") return b.bytes - a.bytes;
      if (folderSort === "photos") return b.photos - a.photos;
      return a.folder.localeCompare(b.folder);
    });

  const ANALYSIS_TABS: { id: AnalysisTab; label: string; count?: number | string }[] = [
    { id: "overview", label: "Overview" },
    { id: "culling", label: "Ratings & Culling" },
    { id: "timeline", label: "Capture Timeline", count: s.years?.length ? `${s.years.length} yrs` : undefined },
    { id: "gear", label: "Gear & Formats", count: `${s.cameras.length} cams` },
    { id: "exposure", label: "Exposure" },
    { id: "folders", label: "Folders", count: r.folders.length },
  ];

  return (
    <div className="flex min-h-0 flex-1 gap-4 overflow-y-auto p-4 text-[12px]">
      <div className="flex w-[500px] shrink-0 flex-col gap-4">
        {note && (
          <div className="rounded-md border border-neutral-700 bg-neutral-900/90 px-3 py-2 text-[11px] text-neutral-300 flex items-center justify-between shadow-sm">
            <span>{note}</span>
            <button type="button" onClick={() => setNote("")} className="text-neutral-500 hover:text-neutral-300 ml-2">✕</button>
          </div>
        )}
        <section className={card}>
          <div className="flex items-baseline justify-between mb-2">
            <h2 className="font-semibold text-neutral-100">Storage</h2>
            <span className="tabular-nums font-semibold text-neutral-200">{size(r.total)}</span>
          </div>

          <div className="mb-3 rounded-md border border-neutral-800 bg-neutral-950/70 p-2.5">
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="text-[11px] font-medium text-neutral-400">Library Location</span>
              {r.is_custom_location ? (
                <span className="rounded bg-sky-950 px-1.5 py-0.5 text-[10px] font-medium text-sky-300 border border-sky-800/60">
                  Custom Location
                </span>
              ) : (
                <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400">
                  Default OS Location
                </span>
              )}
            </div>
            <p className="truncate font-mono text-[11px] text-neutral-300 select-text mb-2.5" title={r.data_dir}>
              {r.data_dir}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handlePickNewLocation}
                disabled={!!relocating}
                className="rounded bg-neutral-800 px-2.5 py-1 text-[11px] font-medium text-neutral-200 hover:bg-neutral-700 hover:text-white border border-neutral-700/80 transition-colors cursor-pointer"
              >
                Change Location…
              </button>
              {r.is_custom_location && (
                <button
                  type="button"
                  onClick={handleResetLocation}
                  disabled={!!relocating}
                  className="rounded px-2.5 py-1 text-[11px] text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 transition-colors cursor-pointer"
                >
                  Reset to Default
                </button>
              )}
              <button
                type="button"
                onClick={handleSyncFolders}
                disabled={reconnectBusy || !!relocating}
                className="rounded px-2.5 py-1 text-[11px] text-sky-400 hover:text-sky-300 hover:bg-sky-950/40 border border-sky-800/60 transition-colors cursor-pointer ml-auto flex items-center gap-1"
                title="Scan and reconnect library photo folders"
              >
                <span>⚡</span>
                <span>{reconnectBusy ? "Scanning…" : "Scan & Reconnect Library"}</span>
              </button>
            </div>
          </div>
          <div className="flex h-3 overflow-hidden rounded bg-neutral-800">
            {parts.map(([label, n, color]) => (
              <div key={label} className={color} style={{ width: `${(n / Math.max(1, r.total)) * 100}%` }} title={`${label}: ${size(n)}`} />
            ))}
          </div>
          <div className="mt-3 space-y-1">
            {parts.map(([label, n, color]) => (
              <div key={label} className="flex items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-sm ${color}`} />
                <span className="flex-1 text-neutral-300">{label}</span>
                <span className="tabular-nums text-neutral-400">{size(n)}</span>
              </div>
            ))}
          </div>

          <div className="mt-3 flex items-center justify-between border-t border-neutral-800/80 pt-2.5">
            <button
              type="button"
              onClick={() => setShowDetails((d) => !d)}
              className="flex items-center gap-1.5 text-[11px] font-medium text-sky-400 hover:text-sky-300"
            >
              <span>{showDetails ? "Hide Storage Analysis & Details" : "Show Storage Analysis & Details"}</span>
              <span className="text-[9px]">{showDetails ? "▲" : "▼"}</span>
            </button>
            <span className="text-[10px] text-neutral-500">
              {r.ai_models_details?.length ? `${plural(r.ai_models_count ?? r.ai_models_details.length, "model")} · ` : ""}
              {r.immich_albums?.length ? `${r.immich_albums.length} Immich album(s) · ` : ""}
              {size(r.catalog)} DB
            </span>
          </div>

          {showDetails && (
            <div className="mt-3 space-y-3 rounded border border-neutral-800 bg-neutral-950 p-3 text-[11px]">
              <div>
                <div className="mb-1 flex items-center justify-between font-semibold text-sky-400">
                  <span>Catalog Database</span>
                  <span className="tabular-nums text-neutral-300">{size(r.catalog)}</span>
                </div>
                <div className="space-y-0.5 pl-2 font-mono text-[10px] text-neutral-400">
                  {r.catalog_details && (
                    <>
                      <div className="flex justify-between">
                        <span>• Database (catalog.olcat):</span>
                        <span>{size(r.catalog_details.db)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>• Write-Ahead Log (catalog.olcat-wal):</span>
                        <span>{size(r.catalog_details.wal)}</span>
                      </div>
                      {r.catalog_details.free > 0 && (
                        <div className="flex justify-between text-neutral-500">
                          <span>• Unused space in pages:</span>
                          <span>{size(r.catalog_details.free)}</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {r.immich_albums && r.immich_albums.length > 0 && (
                <div>
                  <div className="mb-1 flex items-center justify-between font-semibold text-cyan-400">
                    <span>Immich Synced Albums ({r.immich_albums.length})</span>
                    <span className="tabular-nums text-neutral-300">{size(immichAlbumsBytes)}</span>
                  </div>
                  <div className="space-y-1 pl-2 font-mono text-[10px] text-neutral-400">
                    {r.immich_albums.map((alb) => (
                      <div key={alb.path} className="flex items-center justify-between gap-2 border-b border-neutral-900 pb-1 last:border-b-0">
                        <div className="truncate">
                          <span className="text-neutral-200">{alb.name}</span>
                          <span className="ml-1.5 text-neutral-500">({plural(alb.photos, "photo")})</span>
                        </div>
                        <span className="shrink-0 tabular-nums text-neutral-300">{size(alb.bytes)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {r.ai_models_details && r.ai_models_details.length > 0 && (
                <div>
                  <div className="mb-1 flex items-center justify-between font-semibold text-purple-400">
                    <div className="flex items-center gap-2">
                      <span>Downloaded AI Models ({r.ai_models_count ?? r.ai_models_details.length})</span>
                      {onGoToAiModels && (
                        <button
                          type="button"
                          onClick={onGoToAiModels}
                          className="text-[10px] font-normal text-sky-400 underline hover:text-sky-300"
                        >
                          Manage in AI models tab →
                        </button>
                      )}
                    </div>
                    <span className="tabular-nums text-neutral-300">{size(aiModelsBytes)}</span>
                  </div>
                  <div className="space-y-1.5 pl-2 font-mono text-[10px] text-neutral-400">
                    {r.ai_models_details.map((m) => (
                      <div key={m.id} className="flex items-center justify-between gap-2 border-b border-neutral-900 pb-1 last:border-b-0">
                        <div className="flex min-w-0 items-center gap-1.5 truncate">
                          <span className="truncate font-sans text-neutral-200">{m.name}</span>
                          <span className="shrink-0 rounded bg-neutral-800 px-1 py-0.5 font-sans text-[9px] uppercase text-neutral-400">
                            {m.kind}
                          </span>
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          <span className="tabular-nums text-neutral-300">{size(m.bytes)}</span>
                          <button
                            type="button"
                            disabled={!!busy}
                            onClick={() =>
                              run(
                                `ai_model:${m.id}`,
                                `Remove ${m.name}`,
                                () => api.cleanUp(`ai_model:${m.id}`),
                                `Remove downloaded model "${m.name}" (${size(m.bytes)})?\n\nYou can re-download it anytime from the AI models tab.`
                              )
                            }
                            className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-red-400 hover:bg-neutral-700 hover:text-red-300 disabled:opacity-40"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {r.ai_cache && (r.ai_cache.bytes > 0 || r.ai_cache.files > 0) && (
                <div>
                  <div className="mb-1 flex items-center justify-between font-semibold text-pink-400">
                    <span>AI Adjustment Layers & Patch Cache ({plural(r.ai_cache.files, "patch")})</span>
                    <span className="tabular-nums text-neutral-300">{size(r.ai_cache.bytes)}</span>
                  </div>
                  <div className="space-y-0.5 pl-2 font-mono text-[10px] text-neutral-400">
                    <div className="flex justify-between">
                      <span>• Rendered in-place pixel patches:</span>
                      <span>{size(r.ai_cache.bytes)}</span>
                    </div>
                    <div className="text-[10px] text-neutral-500 italic mt-0.5">
                      Cached pixel patches for in-place AI removal & denoise. Safely re-rendered on demand.
                    </div>
                  </div>
                </div>
              )}

              <div>
                <div className="mb-1 flex items-center justify-between font-semibold text-neutral-300">
                  <span>Analysis of "Other" Storage</span>
                  <span className="tabular-nums text-neutral-300">{size(otherBytes)}</span>
                </div>
                <div className="space-y-0.5 pl-2 font-mono text-[10px] text-neutral-400">
                  {r.other_details && r.other_details.length > 0 ? (
                    r.other_details.map((item) => (
                      <div key={item.name} className="flex justify-between">
                        <span className="truncate">• {item.name}:</span>
                        <span className="shrink-0 tabular-nums text-neutral-400">{size(item.bytes)}</span>
                      </div>
                    ))
                  ) : (
                    <div className="italic text-neutral-500">No miscellaneous files found.</div>
                  )}
                </div>
              </div>
            </div>
          )}

          <p className="mt-3 text-[11px] text-neutral-500">
            Not counted above: your photo files, {size(own.reduce((n, f) => n + f.bytes, 0))} in {own.length} folder(s). Optixel edits them in place and never copies them.
          </p>
        </section>

        <section className={card}>
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-neutral-100">Clean up</h2>
              {c.all && (c.all.count > 0 || c.all.bytes > 0) && (
                <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400">
                  {size(c.all.bytes)} recoverable
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="truncate text-[11px] text-emerald-400" title={note}>
                {note}
              </span>
              {regenProgress && regenProgress.total > 0 && (
                <span className="font-mono text-[11px] font-medium text-sky-400 tabular-nums">
                  {Math.round((regenProgress.done / regenProgress.total) * 100)}%
                </span>
              )}
              <Btn
                active
                className="px-2.5 py-0.5 font-medium text-violet-200 hover:text-white"
                disabled={!!busy || !(c.all?.count || c.all?.bytes)}
                onClick={() =>
                  run(
                    "all",
                    "Clear all caches",
                    () => api.cleanUp("all"),
                    `Clean up all caches, orphaned thumbnails, AI layer patches, and edit history, and compact the catalog database?\n\nEstimated space to free: ${size(c.all?.bytes ?? 0)}.\n(Your local and synced photo masters are kept safe).`
                  )
                }
              >
                {busy === "all" ? "Cleaning…" : "Clear All"}
              </Btn>
            </div>
          </div>
          {cleans.map(([key, title, detail, amount, verb, confirm]) => (
            <div key={key} className="flex items-center gap-3 border-t border-neutral-800 py-2 first-of-type:border-t-0">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-neutral-200 font-medium">{title}</span>
                  {key === "ai_models" && onGoToAiModels && (
                    <button
                      type="button"
                      onClick={onGoToAiModels}
                      className="text-[10px] text-sky-400 underline hover:text-sky-300 font-normal"
                    >
                      AI models tab →
                    </button>
                  )}
                </div>
                <div className="text-[11px] text-neutral-500">{detail}</div>
              </div>
              <span className="shrink-0 text-right tabular-nums text-neutral-400">{amount}</span>
              <Btn className="w-16 shrink-0" disabled={!!busy || !(c[key]?.count || c[key]?.bytes)} onClick={() => run(key, title, () => api.cleanUp(key), confirm)}>
                {busy === key ? "…" : verb}
              </Btn>
            </div>
          ))}
          <div className="flex flex-col gap-2 border-t border-neutral-800 py-2.5">
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-neutral-200 font-medium">Regenerate thumbnails</span>
                  {regenProgress && (
                    <span className="rounded bg-sky-950 px-1.5 py-0.5 text-[10px] font-medium text-sky-300 border border-sky-800/60 animate-pulse">
                      In progress…
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-neutral-500">
                  Re-render all thumbnails directly from original photo files on disk. Fixes mismatched, outdated, or missing previews.
                </div>
              </div>
              <span className="shrink-0 text-right tabular-nums text-neutral-400">
                {regenProgress ? `${num(regenProgress.done)} / ${num(regenProgress.total)}` : plural(r.summary.total_photos, "photo")}
              </span>
              <Btn
                className="w-24 shrink-0"
                disabled={!!busy || !!regenProgress}
                onClick={async () => {
                  if (busy || regenProgress) return;
                  setBusy("regen_thumbs");
                  setNote("Regenerating thumbnails…");
                  try {
                    const count = await api.regenerateThumbnails();
                    const next = await api.storage();
                    setR(next);
                    setNote(`Regenerated ${num(count)} thumbnails.`);
                  } catch (e) {
                    setNote(`Failed to regenerate: ${e}`);
                  } finally {
                    setBusy("");
                    setTimeout(() => setRegenProgress(null), 2500);
                  }
                }}
              >
                {busy === "regen_thumbs" || regenProgress ? "Working…" : "Regenerate"}
              </Btn>
            </div>
            {regenProgress && regenProgress.total > 0 && (
              <div className="space-y-1.5 rounded-lg border border-sky-800/80 bg-sky-950/40 p-2.5 text-sky-200">
                <div className="flex justify-between text-[11px] font-medium">
                  <span className="flex items-center gap-1.5">
                    <svg className="h-3.5 w-3.5 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                    </svg>
                    <span>Regenerating photo thumbnails…</span>
                  </span>
                  <span className="font-mono text-sky-300">
                    {Math.round((regenProgress.done / regenProgress.total) * 100)}% ({num(regenProgress.done)} / {num(regenProgress.total)})
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-800">
                  <div
                    className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-150"
                    style={{
                      width: `${Math.min(100, Math.max(3, (regenProgress.done / regenProgress.total) * 100))}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className={`${card} flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto`}>
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 pb-3">
          <div>
            <h2 className="text-sm font-semibold text-neutral-100">Library Analysis & Statistics</h2>
            <p className="text-[11px] text-neutral-400 mt-0.5">
              From folders {num(own.reduce((n, f) => n + f.photos, 0))} · Immich {num(r.immich_photos)} · Google Photos {num(google)}
              {cloudPhotos ? ` · Cloud storage ${num(cloudPhotos)}` : ""}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1 rounded-lg bg-neutral-950 p-1 border border-neutral-800/80">
            {ANALYSIS_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setAnalysisTab(t.id)}
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  analysisTab === t.id
                    ? "bg-neutral-800 text-white shadow-xs"
                    : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900"
                }`}
              >
                <span>{t.label}</span>
                {t.count !== undefined && (
                  <span className="rounded bg-neutral-900 px-1 py-0.2 text-[10px] text-neutral-500 font-mono">
                    {t.count}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Empty Catalog Notice / Quick Reconnect */}
        {s.total_photos === 0 && (
          <div className="rounded-lg border border-sky-800/80 bg-sky-950/40 p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-sm">
            <div className="space-y-0.5">
              <div className="flex items-center gap-1.5 font-semibold text-sky-200 text-xs">
                <span>⚡</span>
                <span>Catalog has 0 indexed photos</span>
              </div>
              <p className="text-[11px] text-sky-300/80 leading-relaxed">
                Previews, thumbnails, and sidecars exist in storage. Scan and reconnect your photo folders to load full library statistics.
              </p>
            </div>
            <button
              type="button"
              onClick={handleSyncFolders}
              disabled={reconnectBusy}
              className="shrink-0 rounded bg-sky-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50 cursor-pointer shadow-xs transition-colors flex items-center gap-1.5"
            >
              <span>{reconnectBusy ? "Scanning Folders…" : "⚡ Reconnect & Scan Library"}</span>
            </button>
          </div>
        )}

        {/* Global Summary Metric Tiles */}
        <div className="grid grid-cols-4 gap-2">
          {tiles.map(([label, n, hint]) => (
            <div key={label} className="rounded bg-neutral-950 px-3 py-2 border border-neutral-800/60" title={hint}>
              <div className="flex items-baseline justify-between">
                <div className="text-lg tabular-nums text-neutral-100 font-semibold">{num(n)}</div>
                {hint && <span className="text-[10px] text-neutral-500 font-mono">{hint}</span>}
              </div>
              <div className="text-[11px] text-neutral-400">{label}</div>
            </div>
          ))}
        </div>

        {/* Overview Tab */}
        {analysisTab === "overview" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-4">
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className={heading}>Culling & Pick Ratio</h3>
                    <button
                      type="button"
                      onClick={() => setAnalysisTab("culling")}
                      className="text-[10px] text-sky-400 hover:underline"
                    >
                      Details →
                    </button>
                  </div>
                  <CullingBar picks={s.picks} rejects={s.rejects} total={totalPhotos} />
                  <div className="pt-2 border-t border-neutral-900">
                    <div className="mb-1 text-[11px] font-medium text-neutral-300">Rating Breakdown</div>
                    <RatingBreakdown ratings={s.ratings} total={totalPhotos} />
                  </div>
                </div>

                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className={heading}>Capture Timeline (By Year)</h3>
                    <button
                      type="button"
                      onClick={() => setAnalysisTab("timeline")}
                      className="text-[10px] text-sky-400 hover:underline"
                    >
                      Full Timeline →
                    </button>
                  </div>
                  <TimelineHistogram years={s.years} total={totalPhotos} />
                </div>
              </div>

              <div className="space-y-4">
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className={heading}>File Formats & RAW Ratio</h3>
                    <button
                      type="button"
                      onClick={() => setAnalysisTab("gear")}
                      className="text-[10px] text-sky-400 hover:underline"
                    >
                      All Gear →
                    </button>
                  </div>
                  <FormatsSection
                    rawCount={s.raw_count ?? 0}
                    stdCount={s.std_count ?? 0}
                    formats={s.formats}
                    total={totalPhotos}
                  />
                </div>

                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 space-y-3">
                  <h3 className={heading}>Top Cameras & Lenses</h3>
                  <div className="space-y-3">
                    <div>
                      <div className="mb-1 text-[11px] font-medium text-neutral-300">Cameras</div>
                      {s.cameras.slice(0, 4).map((c) => (
                        <StatBar key={c.name} label={c.name} count={c.count} total={totalPhotos} color="bg-sky-500" />
                      ))}
                      {s.cameras.length === 0 && <p className="text-neutral-600 text-[11px]">None recorded</p>}
                    </div>
                    <div className="pt-2 border-t border-neutral-900">
                      <div className="mb-1 text-[11px] font-medium text-neutral-300">Lenses</div>
                      {s.lenses.slice(0, 4).map((l) => (
                        <StatBar key={l.name} label={l.name} count={l.count} total={totalPhotos} color="bg-emerald-500" />
                      ))}
                      {s.lenses.length === 0 && <p className="text-neutral-600 text-[11px]">None recorded</p>}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Edit & AI Insights */}
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5">
              <h3 className={heading}>Edit Depth & Non-Destructive AI Adjustments</h3>
              <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                <div className="rounded bg-neutral-900/80 p-2.5 border border-neutral-800">
                  <div className="text-neutral-400 text-[10px]">Total History Steps</div>
                  <div className="text-base font-semibold text-neutral-100 tabular-nums">{num(r.history_steps)}</div>
                  <div className="text-[10px] text-neutral-500">{size(c.history?.bytes ?? 0)} stored in DB</div>
                </div>
                <div className="rounded bg-neutral-900/80 p-2.5 border border-neutral-800">
                  <div className="text-neutral-400 text-[10px]">Avg Steps / Edited Photo</div>
                  <div className="text-base font-semibold text-neutral-100 tabular-nums">{s.history_stats?.avg_steps ?? 0}</div>
                  <div className="text-[10px] text-neutral-500">Across {num(s.edited)} edited files</div>
                </div>
                <div className="rounded bg-neutral-900/80 p-2.5 border border-neutral-800">
                  <div className="text-neutral-400 text-[10px]">Peak History on 1 Photo</div>
                  <div className="text-base font-semibold text-neutral-100 tabular-nums">{s.history_stats?.max_steps ?? 0}</div>
                  <div className="text-[10px] text-neutral-500">Deepest edit revision stack</div>
                </div>
                <div className="rounded bg-neutral-900/80 p-2.5 border border-neutral-800">
                  <div className="text-neutral-400 text-[10px]">AI Layers & Patch Cache</div>
                  <div className="text-base font-semibold text-pink-400 tabular-nums">{size(r.ai_cache?.bytes ?? 0)}</div>
                  <div className="text-[10px] text-neutral-500">{plural(r.ai_cache?.files ?? 0, "patch")} cached</div>
                </div>
              </div>
            </div>

            {/* Folders Summary */}
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className={heading}>Folders by Size ({r.folders.length} Total)</h3>
                <button
                  type="button"
                  onClick={() => setAnalysisTab("folders")}
                  className="text-[10px] text-sky-400 hover:underline"
                >
                  Manage all folders →
                </button>
              </div>
              <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] gap-x-4 gap-y-1.5 pt-1">
                {r.folders.slice(0, 5).map((f) => (
                  <div key={f.folder} className="contents text-[11px]">
                    <span className="truncate text-neutral-300" title={f.folder}>
                      {short(f.folder)}
                    </span>
                    <span className="text-right tabular-nums text-neutral-400">{num(f.photos)} photos</span>
                    <span className="text-right tabular-nums text-neutral-300 font-medium">{f.online ? size(f.bytes) : "—"}</span>
                    <span className="text-right">
                      {!f.online ? (
                        <span className="text-amber-400">offline</span>
                      ) : f.missing ? (
                        <span className="text-red-400">{num(f.missing)} missing</span>
                      ) : (
                        <span className="text-emerald-500 text-[10px]">online</span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Ratings & Culling Tab */}
        {analysisTab === "culling" && (
          <div className="space-y-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
              <h3 className={heading}>Culling Ratios & Flags</h3>
              <CullingBar picks={s.picks} rejects={s.rejects} total={totalPhotos} />
              <div className="grid grid-cols-3 gap-3 pt-3">
                <div className="rounded border border-emerald-900/50 bg-emerald-950/20 p-3">
                  <div className="text-emerald-400 font-semibold text-lg tabular-nums">{num(s.picks)}</div>
                  <div className="text-[11px] text-neutral-300">Picks (Flag = 1)</div>
                  <div className="text-[10px] text-neutral-500 mt-1">{((s.picks / totalPhotos) * 100).toFixed(2)}% of library</div>
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/30 p-3">
                  <div className="text-neutral-200 font-semibold text-lg tabular-nums">{num(Math.max(0, totalPhotos - s.picks - s.rejects))}</div>
                  <div className="text-[11px] text-neutral-300">Unflagged (Flag = 0)</div>
                  <div className="text-[10px] text-neutral-500 mt-1">{((Math.max(0, totalPhotos - s.picks - s.rejects) / totalPhotos) * 100).toFixed(2)}% of library</div>
                </div>
                <div className="rounded border border-rose-900/50 bg-rose-950/20 p-3">
                  <div className="text-rose-400 font-semibold text-lg tabular-nums">{num(s.rejects)}</div>
                  <div className="text-[11px] text-neutral-300">Rejects (Flag = -1)</div>
                  <div className="text-[10px] text-neutral-500 mt-1">{((s.rejects / totalPhotos) * 100).toFixed(2)}% of library</div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
              <h3 className={heading}>Star Ratings Distribution</h3>
              <RatingBreakdown ratings={s.ratings} total={totalPhotos} />
            </div>
          </div>
        )}

        {/* Capture Timeline Tab */}
        {analysisTab === "timeline" && (
          <div className="space-y-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className={heading}>Capture Timeline Histogram</h3>
                <span className="text-[11px] text-neutral-400">{s.years?.length ?? 0} active capture years</span>
              </div>
              <TimelineHistogram years={s.years} total={totalPhotos} height="h-40" />
            </div>

            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
              <h3 className={heading}>Chronological Yearly Summary</h3>
              <div className="space-y-1.5">
                {[...(s.years ?? [])]
                  .sort((a, b) => b.year.localeCompare(a.year))
                  .map((y) => (
                    <StatBar
                      key={y.year}
                      label={<span className="font-mono text-neutral-200 font-medium">Year {y.year}</span>}
                      count={y.count}
                      total={totalPhotos}
                      color="bg-sky-500"
                    />
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Gear & Formats Tab */}
        {analysisTab === "gear" && (
          <div className="space-y-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
              <h3 className={heading}>File Format Distribution & RAW vs Standard</h3>
              <FormatsSection
                rawCount={s.raw_count ?? 0}
                stdCount={s.std_count ?? 0}
                formats={s.formats}
                total={totalPhotos}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                <h3 className={heading}>Cameras ({s.cameras.length})</h3>
                {s.cameras.map((c) => (
                  <StatBar key={c.name} label={c.name} count={c.count} total={totalPhotos} color="bg-sky-500" />
                ))}
                {s.cameras.length === 0 && <p className="text-neutral-600 text-[11px]">No cameras recorded</p>}
              </div>

              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                <h3 className={heading}>Lenses ({s.lenses.length})</h3>
                {s.lenses.map((l) => (
                  <StatBar key={l.name} label={l.name} count={l.count} total={totalPhotos} color="bg-emerald-500" />
                ))}
                {s.lenses.length === 0 && <p className="text-neutral-600 text-[11px]">No lenses recorded</p>}
              </div>
            </div>
          </div>
        )}

        {/* Exposure Analytics Tab */}
        {analysisTab === "exposure" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                <div>
                  <h3 className={heading}>ISO Sensitivity</h3>
                  <p className="text-[10px] text-neutral-500">Daylight vs Low-light Distribution</p>
                </div>
                <div className="space-y-1">
                  {(s.iso_ranges ?? []).map((ir) => (
                    <StatBar key={ir.range} label={`ISO ${ir.range}`} count={ir.count} total={totalPhotos} color="bg-amber-500" />
                  ))}
                  {(!s.iso_ranges || s.iso_ranges.length === 0) && <p className="text-neutral-600 text-[11px]">No ISO metadata</p>}
                </div>
              </div>

              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                <div>
                  <h3 className={heading}>Aperture & Depth of Field</h3>
                  <p className="text-[10px] text-neutral-500">Wide Aperture vs Landscape F-Stops</p>
                </div>
                <div className="space-y-1">
                  {(s.aperture_ranges ?? []).map((ar) => (
                    <StatBar key={ar.range} label={ar.range} count={ar.count} total={totalPhotos} color="bg-purple-500" />
                  ))}
                  {(!s.aperture_ranges || s.aperture_ranges.length === 0) && <p className="text-neutral-600 text-[11px]">No aperture metadata</p>}
                </div>
              </div>

              <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                <div>
                  <h3 className={heading}>Focal Length</h3>
                  <p className="text-[10px] text-neutral-500">Ultra-wide, Normal, to Super-tele</p>
                </div>
                <div className="space-y-1">
                  {(s.focal_ranges ?? []).map((fr) => (
                    <StatBar key={fr.range} label={fr.range} count={fr.count} total={totalPhotos} color="bg-sky-500" />
                  ))}
                  {(!s.focal_ranges || s.focal_ranges.length === 0) && <p className="text-neutral-600 text-[11px]">No focal length metadata</p>}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Folders Tab */}
        {analysisTab === "folders" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-neutral-800 bg-neutral-950/60 p-3">
              <input
                type="text"
                placeholder="Search folders…"
                value={folderSearch}
                onChange={(e) => setFolderSearch(e.target.value)}
                className="w-64 rounded bg-neutral-900 px-2.5 py-1 text-xs text-neutral-200 border border-neutral-800 focus:border-sky-500 focus:outline-hidden"
              />
              <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                <span className="text-[11px]">Sort:</span>
                <button
                  type="button"
                  onClick={() => setFolderSort("bytes")}
                  className={`rounded px-2 py-0.5 ${folderSort === "bytes" ? "bg-neutral-800 text-neutral-100 font-medium" : "hover:text-neutral-200"}`}
                >
                  Size
                </button>
                <button
                  type="button"
                  onClick={() => setFolderSort("photos")}
                  className={`rounded px-2 py-0.5 ${folderSort === "photos" ? "bg-neutral-800 text-neutral-100 font-medium" : "hover:text-neutral-200"}`}
                >
                  Photos
                </button>
                <button
                  type="button"
                  onClick={() => setFolderSort("name")}
                  className={`rounded px-2 py-0.5 ${folderSort === "name" ? "bg-neutral-800 text-neutral-100 font-medium" : "hover:text-neutral-200"}`}
                >
                  Path
                </button>
              </div>
            </div>

            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-2">
              <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] gap-x-4 border-b border-neutral-800 pb-2 text-[10px] uppercase font-semibold text-neutral-400">
                <span>Folder</span>
                <span className="text-right">Photos</span>
                <span className="text-right">Disk Size</span>
                <span className="text-right">Status</span>
              </div>
              {filteredFolders.length === 0 && <p className="text-neutral-600 text-[11px] py-4 text-center">No folders match filter.</p>}
              <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] gap-x-4 gap-y-2 pt-1 overflow-y-auto max-h-[500px]">
                {filteredFolders.map((f) => (
                  <div key={f.folder} className="contents text-[11px]">
                    <span className="truncate text-neutral-300" title={f.folder}>
                      {short(f.folder)}
                      <span className="ml-2 font-mono text-[10px] text-neutral-600">{f.folder}</span>
                    </span>
                    <span className="text-right tabular-nums text-neutral-400">{num(f.photos)}</span>
                    <span className="text-right tabular-nums text-neutral-300 font-medium">{f.online ? size(f.bytes) : "—"}</span>
                    <span className="text-right">
                      {!f.online ? (
                        <button
                          className="text-amber-400 hover:text-amber-300 disabled:opacity-40"
                          disabled={!!busy}
                          title="Folder not found; remove its photos from the catalog…"
                          onClick={() => removeFolder(f.folder)}
                        >
                          offline · remove…
                        </button>
                      ) : f.missing ? (
                        <span className="text-red-400">{num(f.missing)} missing</span>
                      ) : (
                        <span className="text-emerald-500 text-[10px]">online</span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Relocation Confirmation & Progress Modal */}
      {pendingRelocate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-xs">
          <div className="w-full max-w-xl rounded-xl border border-neutral-700 bg-neutral-900 p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
              <h3 className="text-sm font-semibold text-neutral-100 flex items-center gap-2">
                <span>📁</span>
                <span>Storage Location Configuration</span>
              </h3>
              {!relocating && (
                <button
                  type="button"
                  onClick={() => setPendingRelocate(null)}
                  className="text-neutral-400 hover:text-neutral-200 text-lg leading-none cursor-pointer"
                >
                  ✕
                </button>
              )}
            </div>

            <div className="space-y-2.5 rounded-lg border border-neutral-800 bg-neutral-950/70 p-3 text-[11px]">
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">Current Storage</span>
                <p className="font-mono text-neutral-300 truncate mt-0.5 select-text" title={pendingRelocate.current_dir}>
                  {pendingRelocate.current_dir} ({size(pendingRelocate.current_bytes ?? r.total)})
                </p>
              </div>
              <div className="pt-2 border-t border-neutral-800/60">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">Target Location</span>
                <p className="font-mono text-emerald-300 font-medium truncate mt-0.5 select-text" title={pendingRelocate.target}>
                  {pendingRelocate.target}
                </p>
              </div>
            </div>

            {relocating ? (
              <div className="space-y-2.5 rounded-lg border border-sky-800/80 bg-sky-950/40 p-4 text-sky-200">
                <div className="flex justify-between text-xs font-medium">
                  <span>
                    {relocating.phase === "checkpoint"
                      ? "Flushing database transactions…"
                      : relocating.phase === "switching"
                      ? "Opening existing catalog…"
                      : relocating.phase === "fresh"
                      ? "Initializing fresh library…"
                      : relocating.phase === "done"
                      ? "Finishing…"
                      : "Transferring files to new location…"}
                  </span>
                  <span className="font-mono text-sky-300">
                    {relocating.total_bytes > 0
                      ? `${Math.round(((relocating.done_bytes || 0) / relocating.total_bytes) * 100)}%`
                      : "Working…"}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-800">
                  <div
                    className="h-full bg-sky-500 transition-all duration-200"
                    style={{
                      width: `${relocating.total_bytes > 0 ? Math.min(100, Math.max(5, ((relocating.done_bytes || 0) / relocating.total_bytes) * 100)) : 100}%`,
                    }}
                  />
                </div>
                {relocating.file && (
                  <p className="truncate font-mono text-[10px] text-neutral-400">
                    Copying: {relocating.file} ({size(relocating.done_bytes)} of {size(relocating.total_bytes)})
                  </p>
                )}
                <p className="text-[10px] text-sky-400/80">Please keep Optixel Studio open while switching storage.</p>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-[11px] text-neutral-300 font-medium">
                  Choose how you want to configure this location:
                </p>

                {/* Option 1: Copy Everything */}
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 hover:border-neutral-700 transition-colors">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 font-semibold text-neutral-100 text-xs">
                        <span className="text-sky-400">📦</span>
                        <span>Copy everything to new location</span>
                        <span className="rounded bg-sky-950 px-1.5 py-0.5 text-[10px] text-sky-400 border border-sky-800/60 font-mono">
                          {size(pendingRelocate.current_bytes ?? r.total)}
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-400 leading-relaxed">
                        Transfers all catalog data (<code className="font-mono">catalog.olcat</code>), edit history, rendered previews/thumbnails, neural models, and settings.
                      </p>
                      <label className="flex items-center gap-2 pt-1 text-[11px] text-neutral-300 cursor-pointer select-none">
                        <input
                          type="checkbox"
                          checked={cleanupOld}
                          onChange={(e) => setCleanupOld(e.target.checked)}
                          className="rounded border-neutral-700 bg-neutral-900 text-sky-500 focus:ring-sky-500/20"
                        />
                        <span>Delete files from previous location after successful copy</span>
                      </label>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleConfirmRelocate("copy")}
                      className="shrink-0 rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 cursor-pointer shadow-sm transition-colors"
                    >
                      Copy & Switch
                    </button>
                  </div>
                </div>

                {/* Option 2: Use Existing Configuration & Catalog */}
                <div
                  className={`rounded-lg border p-3.5 transition-colors ${
                    pendingRelocate.has_existing_catalog
                      ? "border-amber-800/80 bg-amber-950/30 hover:border-amber-700"
                      : "border-neutral-800/50 bg-neutral-950/30 opacity-60"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 font-semibold text-neutral-100 text-xs">
                        <span className="text-amber-400">📂</span>
                        <span>Use existing configuration & catalog</span>
                        {pendingRelocate.has_existing_catalog && (
                          <span className="rounded bg-amber-950 px-1.5 py-0.5 text-[10px] text-amber-300 border border-amber-800/60 font-mono">
                            Found {size(pendingRelocate.existing_catalog_bytes ?? 0)}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-neutral-400 leading-relaxed">
                        {pendingRelocate.has_existing_catalog
                          ? `Existing catalog detected (${pendingRelocate.existing_photos_count != null ? `${num(pendingRelocate.existing_photos_count)} photo(s)` : "catalog.olcat"}). Open it directly without transferring or overwriting anything.`
                          : "No existing catalog.olcat was found in the target folder."}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleConfirmRelocate("existing")}
                      disabled={!pendingRelocate.has_existing_catalog}
                      className="shrink-0 rounded bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shadow-sm transition-colors"
                    >
                      Use Existing
                    </button>
                  </div>
                </div>

                {/* Option 3: Start Fresh */}
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3.5 hover:border-neutral-700 transition-colors">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 font-semibold text-neutral-100 text-xs">
                        <span className="text-emerald-400">✨</span>
                        <span>Start fresh from a new start</span>
                        <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400 font-mono">
                          New Library
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-400 leading-relaxed">
                        Initializes a fresh, empty library at the target location. All current catalog photos and data remain safely preserved at the current location.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleConfirmRelocate("fresh")}
                      className="shrink-0 rounded bg-neutral-800 px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-neutral-700 hover:text-white border border-neutral-700 cursor-pointer transition-colors"
                    >
                      Start Fresh
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between border-t border-neutral-800 pt-3">
              <p className="text-[10px] text-neutral-500">
                Your original RAW/JPEG photo files are never moved; they remain in their original folders.
              </p>
              {!relocating && (
                <button
                  type="button"
                  onClick={() => setPendingRelocate(null)}
                  className="rounded bg-neutral-800 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700 cursor-pointer"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** About: app name and version, software updates, disclaimer, and developer contact. */
function About({
  settings,
  setSettings,
  onCheckUpdate,
  updateAvailable,
  updateInfo,
  updateLoading,
}: {
  settings?: Settings | null;
  setSettings?: (s: Settings) => void;
  onCheckUpdate?: () => void;
  updateAvailable?: boolean;
  updateInfo?: UpdateInfo | null;
  updateLoading?: boolean;
}) {
  const [version, setVersion] = useState("");
  useEffect(() => void getVersion().then(setVersion, () => {}), []);

  const openReleases = () => {
    const url = "https://github.com/rcwkan/optixel-studio/releases";
    api.openExternalUrl(url).catch(() => window.open(url, "_blank"));
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 text-[12px]">
      <div className="flex w-[520px] flex-col gap-4">
        <section className={card}>
          <h3 className={heading}>App info</h3>
          <p className="text-[15px] font-semibold text-neutral-100">Optixel Studio</p>
          <p className="text-neutral-400">Version {version || "—"}</p>
          <p className="mt-2 text-neutral-300">Non-destructive RAW processor and photo catalog with a built-in MCP server and a pluggable AI/agent router.</p>
        </section>

        <section className={card}>
          <div className="flex items-center justify-between">
            <h3 className={heading}>Software updates</h3>
            {updateAvailable && (
              <span className="rounded-full bg-[var(--theme-accent,#0ea5e9)]/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[var(--theme-accent,#0ea5e9)] border border-[var(--theme-accent,#0ea5e9)]/30">
                Update available
              </span>
            )}
          </div>
          <p className="text-neutral-300">
            Releases and installation packages are published directly via GitHub Releases.
          </p>

          {updateInfo?.has_update && (
            <div className="mt-3 rounded-md bg-[var(--theme-accent,#0ea5e9)]/10 border border-[var(--theme-accent,#0ea5e9)]/30 p-3 space-y-1">
              <p className="font-semibold text-white">
                Version {updateInfo.release_tag} is available!
              </p>
              <p className="text-[11px] text-neutral-300">
                Current version is v{version || updateInfo.current_version}. Click below to review the update and download the installer package.
              </p>
            </div>
          )}

          <div className="mt-3 flex items-center gap-2.5">
            <button
              type="button"
              onClick={onCheckUpdate}
              disabled={updateLoading}
              className="rounded-lg bg-[var(--theme-accent,#0ea5e9)] hover:brightness-110 disabled:opacity-50 px-3.5 py-1.5 font-semibold text-white transition-all cursor-pointer flex items-center gap-2 shadow-xs"
            >
              {updateLoading ? (
                <>
                  <span className="h-3 w-3 animate-spin rounded-full border border-white border-t-transparent" />
                  <span>Checking GitHub…</span>
                </>
              ) : (
                <>
                  <span>🔄</span>
                  <span>{updateAvailable ? "Review & Install Update" : "Check for Updates"}</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={openReleases}
              className="rounded-lg border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 px-3 py-1.5 font-medium text-neutral-300 transition-colors cursor-pointer"
            >
              View on GitHub ↗
            </button>
          </div>

          {settings && setSettings && (
            <label className="mt-3.5 flex items-center gap-2 cursor-pointer text-neutral-400 select-none">
              <input
                type="checkbox"
                checked={settings.auto_check_updates !== false}
                onChange={(e) => {
                  const updated = { ...settings, auto_check_updates: e.target.checked };
                  setSettings(updated);
                  api.saveSettings(updated);
                }}
              />
              <span className="text-[11px] text-neutral-300">Automatically check for updates on startup</span>
            </label>
          )}
        </section>

        <section className={card}>
          <h3 className={heading}>Disclaimer</h3>
          <p className="text-neutral-300">
            Optixel Studio is provided "as is", without warranty of any kind, express or implied. The developer is not liable for any loss of or damage to
            photos, data or equipment arising from its use. Edits are non-destructive, but always keep backups of your originals. AI-assisted edits and
            third-party services (Immich, Google Photos, cloud storage, AI providers) are used at your own discretion and subject to their own terms.
          </p>
        </section>
        <section className={card}>
          <h3 className={heading}>Developer</h3>
          <p className="select-text text-neutral-200">Ross Kan</p>
          <p className="select-text text-neutral-400">ross.kan@gmail.com</p>
        </section>
      </div>
    </div>
  );
}
