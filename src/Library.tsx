import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { ReactNode, useEffect, useId, useRef, useState } from "react";
import { api, AutoFunction, CloudAccount, CloudFolder, Culling, fileManagerName, Filter, flagName, FolderLoading, FolderProgress, ImmichAlbum, JobProgress, LABEL_COLORS, LABELS, NetworkDrive, Photo, Settings, Source, thumbUrl } from "./api";
import { AutoView, FUNCS } from "./AutoView";
import { Btn, HUE, Hue, Icon, Panel, Stars } from "./controls";
import { FaceSearch } from "./People";
import { PhotoViewer } from "./PhotoViewer";
import { CloudBrowser } from "./Storage";

export type SelectMods = { shift: boolean; toggle: boolean };

type Props = {
  photos: Photo[];
  folders: [string, number][];
  filter: Filter;
  setFilter: (f: Filter) => void;
  selected: number[];
  active: Photo | null;
  select: (id: number, mods: SelectMods) => void;
  openDevelop: (id: number) => void;
  cull: (u: Culling) => void;
  settings: Settings | null;
  setSettings?: (s: Settings) => void;
  dataDir: string;
  loadingFolder?: FolderLoading | null;
  onShowAll?: () => void;
  onAddFolder: () => void;
  onOpenFolder: (folder: string) => void;
  /** folders added with Add folder: the roots of the sidebar tree */
  roots: string[];
  onRemoveRoot: (folder: string) => void;
  onImmich: (album: string | null, name: string) => void;
  /** Next page of the Immich album being shown, null when it is fully loaded */
  nextPage: number | null;
  onLoadMore: () => void;
  onPickGoogle: () => void;
  onSyncApple?: () => void;
  onSyncAmazon?: () => void;
  onSyncAdobe?: () => void;
  /** Connects the drive when needed; settles once its new photos are read */
  onOpenDrive: (drive: NetworkDrive) => Promise<void>;
  onOpenCloud: (folder: CloudFolder, account: string) => void;
  /** Download a folder of a cloud account into the catalog */
  onAddCloudFolder: (account: string, path: string, name: string) => Promise<void>;
  /** Settings › Storage, where drives and cloud storage are added */
  onStorage: () => void;
  onConnect: (source: Source) => void;
  onExport: () => void;
  onVirtualCopy: () => void;
  onRemove: () => void;
  binCount: number;
  onOpenBin: () => void;
  onMoveToBin: () => void;
  onRestoreFromBin: () => void;
  onPermanentDelete: () => void;
  onEmptyBin?: () => void;
  onOpenCompare?: () => void;
  autoFn?: AutoFunction | null;
  onSelectAutoFn?: (fn: AutoFunction | null) => void;
  jobs?: Record<string, JobProgress>;
  onAiSettings?: () => void;
  workingIds?: number[];
  busyAiIds?: number[];
  onAddWorking?: (ids: number[]) => void;
  onOpenViewer?: (id: number) => void;
};

const short = (folder: string) => folder.split(/[\\/]/).slice(-2).join("/");

export function Library(p: Props) {
  // The right panel shows Info (autoFn null) or an auto function; `panel` false folds it away.
  const [internalAutoFn, setInternalAutoFn] = useState<AutoFunction | null>(null);
  const autoFn = p.autoFn !== undefined ? p.autoFn : internalAutoFn;
  const setAutoFn = (fn: AutoFunction | null) => {
    if (p.onSelectAutoFn) p.onSelectAutoFn(fn);
    else setInternalAutoFn(fn);
  };
  const [panel, setPanel] = useState(true);
  const openPanel = (fn: AutoFunction | null) => (setAutoFn(fn), setPanel(true));

  const [size, setSize] = useState(170);
  const [viewMode, setViewMode] = useState<"grid" | "viewer">("grid");
  const [albums, setAlbums] = useState<ImmichAlbum[]>([]);
  const [online, setOnline] = useState<Record<string, boolean>>({});
  const [cloudDirs, setCloudDirs] = useState<CloudFolder[]>([]);
  const [browse, setBrowse] = useState<CloudAccount | null>(null);
  const [showAdvFilter, setShowAdvFilter] = useState(false);
  const f = p.filter;
  const hasAdvFilters = Boolean(f.copy_type || f.aspect || f.edited != null);
  const advFilterCount = [f.copy_type, f.aspect, f.edited != null ? f.edited : null].filter(Boolean).length;
  const immichUrl = p.settings?.immich_url;
  const drives = p.settings?.network_drives ?? [];
  const clouds = p.settings?.clouds ?? [];
  useEffect(() => {
    if (immichUrl) api.immichAlbums().then(setAlbums, () => setAlbums([]));
  }, [immichUrl]);
  const driveKey = drives.map((d) => d.id).join();
  const checkDrives = () => api.networkStatus().then((l) => setOnline(Object.fromEntries(l.map((d) => [d.id, d.online]))));
  useEffect(() => void (driveKey && checkDrives()), [driveKey]);
  const cloudKey = JSON.stringify(clouds.map((a) => [a.id, a.folders]));
  useEffect(() => void api.cloudLocalDirs().then(setCloudDirs), [cloudKey]);
  // Google Photos picks are downloaded into the catalog folder; list them under Google Photos, not Folders.
  const sep = p.dataDir.includes("\\") ? "\\" : "/";
  const remoteDir = `${p.dataDir}${sep}remote`;
  const googleDir = `${remoteDir}${sep}Google Photos`;
  const appleDir = `${remoteDir}${sep}Apple iCloud Photos`;
  const amazonDir = `${remoteDir}${sep}Amazon Photos`;
  const adobeDir = `${remoteDir}${sep}Adobe Cloud Photos`;
  const under = (folder: string, root: string) => folder === root || folder.startsWith(root.replace(/[\\/]+$/, "") + sep);
  const count = (root: string) => p.folders.filter(([folder]) => under(folder, root)).reduce((n, [, k]) => n + k, 0);
  // network drives are listed after the local folders
  const local = p.folders.filter(([folder]) => !under(folder, remoteDir) && !drives.some((d) => under(folder, d.path)));
  const show = (patch: Filter) => p.setFilter({ ...f, folder: null, immich_album: null, ids: null, view: null, ...patch });
  const gridRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const loadMore = useRef(p.onLoadMore);
  loadMore.current = p.onLoadMore;
  // Large Immich albums load a page at a time: ask for the next one when the end of the grid
  // comes near. Observing again after each page keeps loading until the grid overflows.
  useEffect(() => {
    if (!p.nextPage || !endRef.current) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && loadMore.current(), { root: gridRef.current, rootMargin: "600px" });
    io.observe(endRef.current);
    return () => io.disconnect();
  }, [p.nextPage, p.photos.length]);

  // Global hotkeys to toggle between Grid mode and Viewer mode
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if ((k === "v" || k === "enter") && viewMode === "grid") {
        if (p.active || p.photos.length > 0) {
          e.preventDefault();
          if (!p.active && p.photos.length > 0) {
            p.select(p.photos[0].id, { shift: false, toggle: false });
          }
          setViewMode("viewer");
        }
      } else if ((k === "g" || k === "escape") && viewMode === "viewer") {
        e.preventDefault();
        setViewMode("grid");
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [viewMode, p.active, p.photos, p.select]);
  const side = "px-1 text-neutral-500 hover:text-neutral-100";
  const manage = (source: Source) => (
    <button title="Connection" className={side} onClick={() => p.onConnect(source)}>
      ⋯
    </button>
  );
  // Photo services: the connected ones show under Cloud & services, the others are offered by its ＋.
  // All but Immich are synced into a folder of the catalog: one row each, with its pick or sync.
  const services: { source: Source; label: string; hint: string; on: boolean; dir?: string; sync?: [string, string, (() => void) | undefined] }[] = [
    { source: "immich", label: "Immich", hint: "Browse the albums on your Immich server", on: !!immichUrl },
    { source: "google", label: "Google Photos", hint: "Pick photos to download and edit", on: !!p.settings?.google_client_id, dir: googleDir, sync: ["＋", "Pick photos…", p.onPickGoogle] },
    { source: "apple", label: "Apple iCloud Photos", hint: "Your Photos library or iCloud folder", on: !!p.settings?.apple_photos_path, dir: appleDir, sync: ["↻", "Sync iCloud Photos", p.onSyncApple] },
    { source: "amazon", label: "Amazon Photos", hint: "Your Amazon Photos sync folder", on: !!p.settings?.amazon_photos_path, dir: amazonDir, sync: ["↻", "Sync Amazon Photos", p.onSyncAmazon] },
    { source: "adobe", label: "Adobe Cloud Photos", hint: "Creative Cloud Files or Lightroom", on: !!p.settings?.adobe_cloud_path, dir: adobeDir, sync: ["↻", "Sync Adobe Cloud", p.onSyncAdobe] },
  ];
  const running = (fn: AutoFunction) => Object.values(p.jobs ?? {}).some((j) => j.job.startsWith(`${fn}-`) || (fn === "cull" && j.job.startsWith("on-import-")));
  // A button of the right bar, in its function's colour (Info: the theme's accent); choosing the one that is
  // open folds the panel away.
  const rail = (fn: AutoFunction | null, d: string, label: string, title: string, h?: (typeof HUE)[Hue], busy = false) => {
    const on = panel && autoFn === fn;
    return (
      <button
        key={label}
        title={title}
        aria-pressed={on}
        onClick={() => (on ? setPanel(false) : openPanel(fn))}
        className={`relative flex w-12 flex-col items-center gap-1 rounded-md py-1.5 text-[9px] transition-colors ${
          on ? `${h?.soft ?? "bg-[var(--theme-accent-subtle)]"} font-medium text-neutral-100` : "text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200"
        }`}
      >
        {on && <span className={`absolute -left-1 bottom-2 top-2 w-0.5 rounded-full ${h?.bg ?? "bg-[var(--theme-accent)]"}`} />}
        <span className={h?.text ?? "text-[var(--theme-accent,#0ea5e9)]"}>
          <Icon d={d} className="h-4 w-4" />
        </span>
        {label}
        {busy && <span className={`absolute right-1.5 top-1.5 h-1.5 w-1.5 animate-pulse rounded-full ${h?.bg ?? "bg-sky-400"}`} />}
      </button>
    );
  };
  // Photos shown from an auto function's result carry its colour in the toolbar
  const resultTag = f.view?.startsWith("auto:") ? f.view.slice(5) : "";
  const resultFn = FUNCS.find((x) => x.id === resultTag || x.title.toLowerCase().includes(resultTag.toLowerCase())) ?? FUNCS[0];
  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-neutral-800 bg-neutral-900">
        <Panel title={<Heading label="Library" hue="blue" />}>
          <FolderRow
            label="All Photographs"
            hue="blue"
            icon={<Icon d={ICONS.photos} />}
            active={(f.view === "all" || (!f.folder && !f.immich_album && !f.ids && !f.people && !f.in_bin && f.view !== "bin")) && f.view !== "loading" && !p.loadingFolder}
            loading={p.loadingFolder?.viewId === "all"}
            progress={p.loadingFolder?.viewId === "all" ? p.loadingFolder.progress : undefined}
            onClick={() => (p.onShowAll ? p.onShowAll() : show({ view: "all" }))}
          />
          <FolderRow
            label="Bin"
            hue="rose"
            icon={<Icon d={ICONS.trash} />}
            count={p.binCount || undefined}
            active={f.view === "bin" || f.in_bin === true}
            loading={p.loadingFolder?.viewId === "bin"}
            onClick={p.onOpenBin}
            title="Recycle bin for deleted photos"
          />
        </Panel>

        <Panel
          title={<Heading label="Folders" hue="amber" />}
          right={
            <AddMenu
              title="Add a folder or a network drive"
              items={[
                ["Folder…", "On this computer or an attached disk", p.onAddFolder],
                ["Network drive…", "A share on a NAS or another computer", p.onStorage],
              ]}
            />
          }
        >
          {local.length === 0 && drives.length === 0 && <p className="text-[11px] text-neutral-500">Add a folder to browse and edit its photos in place.</p>}
          {p.roots.map((r) => (
            <FolderNode key={r} path={r} depth={0} p={p} onRemove={() => p.onRemoveRoot(r)} />
          ))}
          {local
            .filter(([folder]) => !p.roots.some((r) => under(folder, r)))
            .map(([folder, n]) => {
              const isCurLoading = p.loadingFolder?.viewId === `folder:${folder}`;
              return (
                <FolderRow
                  key={folder}
                  label={short(folder)}
                  title={folder}
                  hue="amber"
                  icon={<Icon d={ICONS.folder} />}
                  count={n}
                  active={f.folder === folder || f.view === `folder:${folder}`}
                  loading={isCurLoading}
                  progress={isCurLoading ? p.loadingFolder?.progress : undefined}
                  onClick={() => p.onOpenFolder(folder)}
                />
              );
            })}
          {drives.map((d) => {
            const isCurLoading = p.loadingFolder?.viewId === `drive:${d.id}`;
            return (
              <FolderNode
                key={d.id}
                path={d.path}
                depth={0}
                p={p}
                root={{
                  label: d.name,
                  title: `${d.url ? `${d.url} → ` : ""}${d.path}${online[d.id] ? "" : " (not connected: opening it connects it)"}`,
                  icon: (
                    <span className="relative flex">
                      <Icon d={ICONS.drive} />
                      <span className={`absolute -bottom-px -right-px h-1.5 w-1.5 rounded-full ring-1 ring-neutral-900 ${online[d.id] ? "bg-emerald-500" : "bg-neutral-600"}`} />
                    </span>
                  ),
                  count: count(d.path) || undefined,
                  active: f.view === `drive:${d.id}` || f.folder === d.path,
                  loading: isCurLoading,
                  progress: isCurLoading ? p.loadingFolder?.progress : undefined,
                  onClick: () => p.onOpenDrive(d).then(checkDrives),
                }}
              />
            );
          })}
        </Panel>

        <Panel
          title={<Heading label="Cloud & services" hue="violet" />}
          right={
            <AddMenu
              title="Connect cloud storage or a photo service"
              items={[
                ["Cloud storage…", "Dropbox, OneDrive, S3, WebDAV or Nextcloud", p.onStorage],
                ...services.filter((s) => !s.on).map((s): MenuItem => [`${s.label}…`, s.hint, () => p.onConnect(s.source)]),
              ]}
            />
          }
        >
          {!clouds.length && !services.some((s) => s.on) && <p className="text-[11px] text-neutral-500">Use ＋ to connect Immich, Google Photos, iCloud, Dropbox and more.</p>}
          {services
            .filter((s) => s.on && s.dir)
            .map((s) => {
              const dir = s.dir!;
              const [glyph, what, sync] = s.sync!;
              const isCurLoading = p.loadingFolder?.viewId === s.source || p.loadingFolder?.viewId === `folder:${dir}`;
              return (
                <FolderRow
                  key={s.source}
                  label={s.label}
                  hue="violet"
                  icon={<Icon d={ICONS.cloud} />}
                  count={p.folders.find(([d]) => d === dir)?.[1] ?? 0}
                  active={f.folder === dir}
                  loading={isCurLoading}
                  progress={isCurLoading ? p.loadingFolder?.progress : undefined}
                  onClick={() => p.onOpenFolder(dir)}
                  actions={
                    <>
                      <button title={what} className={side} onClick={sync}>
                        {glyph}
                      </button>
                      {manage(s.source)}
                    </>
                  }
                />
              );
            })}
          {immichUrl && (
            <>
              <SubHead label="Immich">{manage("immich")}</SubHead>
              <FolderRow
                indent
                label="Favorites"
                hue="violet"
                icon={<Icon d={ICONS.star} />}
                active={f.view === "immich:"}
                loading={p.loadingFolder?.viewId === "immich:"}
                progress={p.loadingFolder?.viewId === "immich:" ? p.loadingFolder.progress : undefined}
                onClick={() => p.onImmich(null, "Favorites")}
              />
              {albums.map((a) => {
                const isCurLoading = p.loadingFolder?.viewId === `immich:${a.id}`;
                return (
                  <FolderRow
                    indent
                    key={a.id}
                    label={a.name}
                    hue="violet"
                    icon={<Icon d={ICONS.album} />}
                    count={a.count}
                    active={f.view === `immich:${a.id}`}
                    loading={isCurLoading}
                    progress={isCurLoading ? p.loadingFolder?.progress : undefined}
                    onClick={() => p.onImmich(a.id, a.name)}
                  />
                );
              })}
            </>
          )}
          {clouds.map((a) => (
            <div key={a.id}>
              <SubHead label={a.name}>
                <button title={`Download a folder of ${a.name}`} className={side} onClick={() => setBrowse(a)}>
                  ＋
                </button>
              </SubHead>
              {cloudDirs
                .filter((c) => c.id === a.id)
                .map((c) => {
                  const isCurLoading = p.loadingFolder?.viewId === `cloud:${a.id}:${c.path}`;
                  return (
                    <FolderRow
                      indent
                      key={c.path}
                      label={c.path || "All files"}
                      hue="violet"
                      title={`${a.name} › ${c.path}`}
                      icon={<Icon d={ICONS.folder} />}
                      count={count(c.local)}
                      active={f.view === `cloud:${a.id}:${c.path}`}
                      loading={isCurLoading}
                      progress={isCurLoading ? p.loadingFolder?.progress : undefined}
                      onClick={() => p.onOpenCloud(c, a.name)}
                    />
                  );
                })}
              {!a.folders.length && (
                <button className="pl-4 text-[11px] text-sky-400 hover:text-sky-300" onClick={() => setBrowse(a)}>
                  Choose a folder…
                </button>
              )}
            </div>
          ))}
        </Panel>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2 border-b border-neutral-800 bg-neutral-900 px-3 py-2 text-[11px]">
          <input
            className="w-56 rounded bg-neutral-800 px-2 py-1 outline-none placeholder:text-neutral-500 focus:ring-1 focus:ring-sky-600"
            placeholder="Search filename, camera, lens, keywords, people"
            value={f.text ?? ""}
            onChange={(e) => p.setFilter({ ...f, text: e.target.value })}
          />
          {(p.settings?.faces_enabled || !!f.people?.length) && (
            <FaceSearch
              people={f.people ?? []}
              label={f.view?.startsWith("people:") ? f.view.slice(7) : undefined}
              onChange={(people, label) =>
                // from the People view the whole library is shown; picked here, the current folder or album is kept
                p.setFilter({ ...f, people: people.length ? people : null, view: f.view?.startsWith("people:") ? (people.length ? `people:${label}` : "all") : f.view })
              }
            />
          )}
          <label className="flex items-center gap-1 text-neutral-400">
            Rating ≥
            <select className="rounded bg-neutral-800 px-1 py-1" value={f.min_rating ?? 0} onChange={(e) => p.setFilter({ ...f, min_rating: +e.target.value || null })}>
              {[0, 1, 2, 3, 4, 5].map((r) => (
                <option key={r} value={r}>
                  {r ? "★".repeat(r) : "any"}
                </option>
              ))}
            </select>
          </label>
          <select className="rounded bg-neutral-800 px-1 py-1" value={f.flag ?? ""} onChange={(e) => p.setFilter({ ...f, flag: e.target.value || null })}>
            <option value="">Any flag</option>
            <option value="pick">Picks</option>
            <option value="unflagged">Unflagged</option>
            <option value="reject">Rejects</option>
          </select>
          <select className="rounded bg-neutral-800 px-1 py-1" value={f.color_label ?? ""} onChange={(e) => p.setFilter({ ...f, color_label: e.target.value || null })}>
            <option value="">Any label</option>
            <option value="none">No label</option>
            {LABELS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>

          <button
            onClick={() => setShowAdvFilter(!showAdvFilter)}
            className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
              showAdvFilter || hasAdvFilters
                ? "bg-[var(--theme-accent-subtle)] text-[var(--theme-accent)] border border-[var(--theme-accent)]"
                : "bg-neutral-800 text-neutral-300 border border-neutral-700/60 hover:bg-neutral-700 hover:text-white"
            }`}
            title="More filters: photo type and orientation"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M2.628 1.601C5.028 1.206 7.49 1 10 1s4.972.206 7.372.601a.75.75 0 01.628.74v2.288a2.25 2.25 0 01-.659 1.59l-4.682 4.683a2.25 2.25 0 00-.659 1.59v3.037c0 .684-.31 1.33-.844 1.757l-1.937 1.55A.75.75 0 018 18.25v-5.757a2.25 2.25 0 00-.659-1.591L2.659 6.22A2.25 2.25 0 012 4.629V2.34a.75.75 0 01.628-.74z" clipRule="evenodd" />
            </svg>
            <span>Filters</span>
            {hasAdvFilters && (
              <span className="flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-sky-500 px-1 text-[9px] font-bold text-black">
                {advFilterCount}
              </span>
            )}
            <span className="text-[9px] text-neutral-400">{showAdvFilter ? "▲" : "▼"}</span>
          </button>

          {p.selected.length >= 2 && p.onOpenCompare && (
            <button
              onClick={p.onOpenCompare}
              className="flex items-center gap-1.5 rounded bg-[var(--theme-accent)] px-2.5 py-1 text-[11px] font-semibold text-[var(--theme-accent-contrast)] hover:bg-[var(--theme-accent-hover)] shadow-xs transition-colors cursor-pointer"
              title="Compare selected photos side-by-side (Shortcut: C)"
            >
              <span>⧉</span> Compare ({p.selected.length})
            </button>
          )}

          {/* ACDSee-style View Mode Switcher: Grid (Manage) vs Viewer (View) */}
          <div className="flex items-center rounded-md bg-neutral-800 p-0.5 border border-neutral-700/60 shadow-xs">
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-[11px] font-medium transition-colors cursor-pointer ${
                viewMode === "grid"
                  ? "bg-[var(--theme-accent,#0ea5e9)] text-[var(--theme-accent-contrast,#ffffff)] font-semibold shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
              title="Grid View — Browse thumbnails (Shortcut: G)"
            >
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
              </svg>
              <span>Grid</span>
            </button>
            <button
              type="button"
              onClick={() => {
                if (!p.active && p.photos.length > 0) {
                  p.select(p.photos[0].id, { shift: false, toggle: false });
                }
                setViewMode("viewer");
              }}
              disabled={p.photos.length === 0}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-[11px] font-medium transition-colors disabled:opacity-30 cursor-pointer ${
                viewMode === "viewer"
                  ? "bg-[var(--theme-accent,#0ea5e9)] text-[var(--theme-accent-contrast,#ffffff)] font-semibold shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
              title="Photo Viewer — ACDSee folder image viewing with zoom, pan, filmstrip, slideshow (Shortcut: V or Enter)"
            >
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              <span>Viewer</span>
            </button>
          </div>

          {f.ids ? (
            <span className={`ml-auto flex items-center gap-1.5 rounded px-2 py-0.5 text-neutral-100 ${HUE[resultFn.hue].soft}`} title="Photos chosen by an auto function">
              <span className={HUE[resultFn.hue].text}>
                <Icon d={resultFn.icon} className="h-3 w-3" />
              </span>
              {p.photos.length} from {resultTag || "a result"}
              <button
                title="Open the auto function in the right panel"
                className="ml-1 text-[10px] text-neutral-400 underline hover:text-neutral-100 cursor-pointer"
                onClick={() => openPanel(resultFn.id)}
              >
                view in auto
              </button>
              <button title="Show all photos again" className="text-neutral-400 hover:text-neutral-100 cursor-pointer" onClick={() => show({})}>
                ✕
              </button>
            </span>
          ) : (
            <span className="ml-auto text-neutral-500">
              {p.selected.length > 1 ? `${p.selected.length} selected / ` : ""}
              {p.photos.length}
              {p.nextPage ? "+" : ""} photos
            </span>
          )}
          <input type="range" className="ol-range w-24" min={110} max={320} value={size} onChange={(e) => setSize(+e.target.value)} title="Thumbnail size" />
        </div>

        {showAdvFilter && (
          <div className="flex flex-col gap-2 border-b border-neutral-800 bg-neutral-900/95 px-3 py-2 text-[11px]">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400 mr-1">
                  Type:
                </span>
                {[
                  { id: "all", label: "All Photos", desc: "Show all photos in view" },
                  { id: "edited", label: "Edited photo", desc: "Show photos with adjustments applied" },
                  { id: "original", label: "Original photo", desc: "Show original master imports" },
                  { id: "copy", label: "Save new edited photo", desc: "Show saved new edited photos (virtual copies)" },
                  { id: "unedited_original", label: "Untouched original", desc: "Show master photos with zero edits" },
                ].map((item) => {
                  const active =
                    item.id === "all"
                      ? !f.copy_type && f.edited == null
                      : item.id === "edited"
                      ? f.copy_type === "edited" || f.edited === true
                      : f.copy_type === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => {
                        if (item.id === "all") {
                          p.setFilter({ ...f, copy_type: null, edited: null });
                        } else if (item.id === "edited") {
                          p.setFilter({ ...f, copy_type: "edited", edited: null });
                        } else {
                          p.setFilter({ ...f, copy_type: item.id as any, edited: null });
                        }
                      }}
                      title={item.desc}
                      className={`rounded px-2.5 py-1 text-[11px] transition-colors ${
                        active
                          ? "bg-sky-600 text-white font-medium shadow-sm ring-1 ring-sky-400/40"
                          : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-white"
                      }`}
                    >
                      {item.label}
                    </button>
                  );
                })}
              </div>

              {(hasAdvFilters || f.text || f.min_rating || f.flag || f.color_label) && (
                <button
                  onClick={() =>
                    p.setFilter({
                      ...f,
                      copy_type: null,
                      aspect: null,
                      edited: null,
                      min_rating: null,
                      flag: null,
                      color_label: null,
                      text: "",
                    })
                  }
                  className="flex items-center gap-1 rounded bg-neutral-800 px-2 py-0.5 text-[10px] text-neutral-400 hover:bg-neutral-700 hover:text-white"
                  title="Clear all filters"
                >
                  <span>✕</span> Reset filters
                </button>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-1 border-t border-neutral-800/60 text-neutral-400">
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
                  Orientation:
                </span>
                {(
                  [
                    ["all", "All"],
                    ["landscape", "Landscape ▱"],
                    ["portrait", "Portrait ▯"],
                    ["square", "Square ◻"],
                  ] as const
                ).map(([id, label]) => {
                  const active = (f.aspect || "all") === id;
                  return (
                    <button
                      key={id}
                      onClick={() => p.setFilter({ ...f, aspect: id === "all" ? null : (id as any) })}
                      className={`rounded px-2 py-0.5 text-[10px] ${
                        active
                          ? "bg-neutral-700 text-white font-medium ring-1 ring-neutral-500"
                          : "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              <div className="ml-auto flex items-center gap-1.5 text-[10px]">
                {f.copy_type && (
                  <span className="flex items-center gap-1 rounded bg-sky-950/80 border border-sky-800/80 px-2 py-0.5 text-sky-200">
                    Type: {f.copy_type === "copy" ? "Save new edited photo" : f.copy_type === "original" ? "Original photo" : f.copy_type === "edited" ? "Edited photo" : f.copy_type}
                    <button
                      onClick={() => p.setFilter({ ...f, copy_type: null })}
                      className="text-sky-400 hover:text-white"
                      title="Clear type filter"
                    >
                      ✕
                    </button>
                  </span>
                )}
                {f.aspect && (
                  <span className="flex items-center gap-1 rounded bg-sky-950/80 border border-sky-800/80 px-2 py-0.5 text-sky-200">
                    Orientation: {f.aspect}
                    <button
                      onClick={() => p.setFilter({ ...f, aspect: null })}
                      className="text-sky-400 hover:text-white"
                      title="Clear orientation filter"
                    >
                      ✕
                    </button>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {f.view === "bin" && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-red-900/50 bg-red-950/40 px-4 py-2.5 text-xs">
            <div className="flex items-center gap-2.5">
              <span className="text-base">🗑️</span>
              <span className="font-semibold text-white">
                Bin ({p.photos.length} item{p.photos.length === 1 ? "" : "s"})
              </span>
              <span className="text-neutral-300 text-[11px] hidden sm:inline">
                Photos here are kept safely until restored or permanently deleted.
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <button
                disabled={!p.selected.length}
                onClick={p.onRestoreFromBin}
                className="flex items-center gap-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white px-3.5 py-1.5 text-xs font-semibold shadow-xs border border-emerald-500 transition-colors disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
                title="Restore selected photos back to their original folders"
              >
                <span>↩ Restore</span>
                {p.selected.length > 0 && (
                  <span className="rounded bg-emerald-700/90 px-1.5 py-0.5 text-[10px] font-bold text-white">
                    ({p.selected.length})
                  </span>
                )}
              </button>
              <button
                disabled={!p.selected.length}
                onClick={p.onPermanentDelete}
                className="flex items-center gap-1.5 rounded-md bg-red-600 hover:bg-red-500 text-white px-3.5 py-1.5 text-xs font-semibold shadow-xs border border-red-500 transition-colors disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
                title="Permanently delete selected photos from disk"
              >
                <span>✕ Delete Permanently</span>
                {p.selected.length > 0 && (
                  <span className="rounded bg-red-700/90 px-1.5 py-0.5 text-[10px] font-bold text-white">
                    ({p.selected.length})
                  </span>
                )}
              </button>
            </div>
          </div>
        )}

        {viewMode === "viewer" ? (
          <PhotoViewer
            photos={p.photos}
            active={p.active}
            onSelectPhoto={(id) => p.select(id, { shift: false, toggle: false })}
            onClose={() => setViewMode("grid")}
            openDevelop={p.openDevelop}
            cull={p.cull}
            onMoveToBin={p.onMoveToBin}
            onOpenCompare={p.onOpenCompare}
            onVirtualCopy={p.onVirtualCopy}
            selectedIds={p.selected}
            folderName={f.folder ? short(f.folder) : f.view?.startsWith("people:") ? f.view.slice(7) : f.view === "bin" ? "Bin" : f.view === "loading" ? "" : "All Photographs"}
          />
        ) : p.loadingFolder && p.photos.length === 0 ? (
          <FolderLoadingScreen loading={p.loadingFolder} />
        ) : p.photos.length === 0 && !p.nextPage ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-neutral-500">
            {f.view === "bin" ? (
              <>
                <span className="text-3xl">🗑️</span>
                <p className="text-xs text-neutral-300 font-medium">Bin is empty</p>
                <p className="text-[11px] text-neutral-500 max-w-sm text-center">
                  Photos you delete from your folders will appear here. You can restore them anytime or permanently delete them.
                </p>
              </>
            ) : (
              <>
                <p>{p.folders.length || f.immich_album ? "No photos here." : "Add a folder of JPEG, TIFF, PNG or RAW files, or connect Immich or Google Photos, to start."}</p>
                {!p.folders.length && !f.immich_album && <Btn onClick={p.onAddFolder}>Add folder…</Btn>}
              </>
            )}
          </div>
        ) : (
          <div className="relative flex min-h-0 flex-1 flex-col">
            {p.loadingFolder && (
              <div className="sticky top-0 z-20 flex items-center justify-between border-b border-sky-900/60 bg-sky-950/90 px-4 py-2 backdrop-blur text-[11px] text-sky-200 shadow-sm">
                <div className="flex items-center gap-2">
                  <svg className="h-3.5 w-3.5 animate-spin text-sky-400 shrink-0" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  <span className="font-medium text-white">{p.loadingFolder.name}</span>
                  <span className="text-sky-300/80">· {p.loadingFolder.progress?.label ?? "Loading…"}</span>
                </div>
                {p.loadingFolder.progress && p.loadingFolder.progress.total > 0 && (
                  <div className="flex items-center gap-3">
                    <div className="h-1.5 w-32 overflow-hidden rounded-full bg-sky-900/80">
                      <div
                        className="h-full rounded-full bg-sky-400 transition-all duration-150"
                        style={{ width: `${Math.min(100, Math.max(2, (p.loadingFolder.progress.done / p.loadingFolder.progress.total) * 100))}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-sky-300 font-mono text-[10px]">
                      {p.loadingFolder.progress.done} / {p.loadingFolder.progress.total}
                    </span>
                  </div>
                )}
              </div>
            )}
            <div ref={gridRef} className="grid flex-1 content-start gap-2 overflow-y-auto p-3" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}>
              {p.photos.map((ph) => (
                <Cell
                  key={ph.id}
                  photo={ph}
                  size={size}
                  selected={p.selected.includes(ph.id)}
                  active={p.active?.id === ph.id}
                  inWorking={p.workingIds?.includes(ph.id)}
                  aiBusy={p.busyAiIds?.includes(ph.id)}
                  onClick={(m) => p.select(ph.id, m)}
                  onOpen={() => p.openDevelop(ph.id)}
                  onView={() => {
                    p.select(ph.id, { shift: false, toggle: false });
                    setViewMode("viewer");
                  }}
                />
              ))}
              {p.nextPage && (
                <div ref={endRef} className="col-span-full py-4 text-center text-[11px] text-neutral-400 flex items-center justify-center gap-2">
                  <svg className="h-3.5 w-3.5 animate-spin text-neutral-400" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  Loading more photos…
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <aside className={panel ? "w-72 shrink-0 overflow-y-auto border-l border-neutral-800 bg-neutral-900" : "hidden"}>
        <div className={autoFn ? "hidden" : ""}>
          {p.active ? (
            <Metadata
              {...p}
              photo={p.active}
              onOpenViewer={(id) => {
                p.select(id, { shift: false, toggle: false });
                setViewMode("viewer");
              }}
            />
          ) : (
            <p className="p-4 text-[11px] text-neutral-500">Select a photo to rate it, see its details and act on it.</p>
          )}
        </div>
        {p.settings && (
          <div className={autoFn ? "" : "hidden"}>
            <AutoView
              settings={p.settings}
              setSettings={p.setSettings ?? (() => {})}
              selected={p.selected}
              view={p.photos}
              jobs={p.jobs ?? {}}
              onShow={(ids, label) => p.setFilter({ ...p.filter, ids, view: `auto:${label}`, folder: null, immich_album: null })}
              onAiSettings={p.onAiSettings ?? (() => {})}
              onStorage={p.onStorage}
              fn={autoFn ?? "cull"}
            />
          </div>
        )}
      </aside>
      <nav aria-label="Panels" className="flex w-14 shrink-0 flex-col items-center gap-1 overflow-y-auto border-l border-neutral-800 bg-neutral-950 py-2">
        {rail(null, ICONS.info, "Info", "Rating, details and actions of the photo")}
        <div className="mt-2 text-[8px] font-semibold uppercase tracking-wider text-neutral-500">Auto</div>
        {FUNCS.map((x) => rail(x.id, x.icon, x.short, `${x.title}: ${x.blurb}`, HUE[x.hue], running(x.id)))}
      </nav>
      {browse && (
        <CloudBrowser
          account={browse}
          action="Download this folder"
          onPick={(path) => {
            setBrowse(null);
            p.onAddCloudFolder(browse.id, path, browse.name);
          }}
          onClose={() => setBrowse(null)}
        />
      )}
    </div>
  );
}

export function FolderLoadingScreen({ loading }: { loading: FolderLoading }) {
  const p = loading.progress;
  const hasTotal = p && p.total > 0;
  const pct = hasTotal ? Math.min(100, Math.max(0, Math.round((p.done / p.total) * 100))) : null;

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-8 select-none text-center">
      <div className="relative mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-neutral-800/80 border border-neutral-700/60 shadow-xl">
        <svg className="h-8 w-8 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
          <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
      </div>

      <h3 className="mb-1 text-base font-medium text-neutral-100 tracking-wide">
        Loading {loading.name}
      </h3>

      <p className="mb-5 text-xs text-neutral-400">
        {p?.label ? `${p.label}…` : "Reading photos and thumbnails…"}
      </p>

      <div className="w-72 max-w-full">
        <div className="relative h-1.5 overflow-hidden rounded-full bg-neutral-800 border border-neutral-700/50 shadow-inner">
          {hasTotal ? (
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-500 via-sky-400 to-indigo-500 transition-all duration-150 ease-out"
              style={{ width: `${Math.max(2, pct!)}%` }}
            />
          ) : (
            <div className="animate-indeterminate absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r from-sky-500 to-indigo-500" />
          )}
        </div>

        <div className="mt-2.5 flex items-center justify-between text-[11px] text-neutral-400 font-mono tabular-nums">
          <span className="text-neutral-500 font-sans">{p?.label ?? "Progress"}</span>
          {hasTotal ? (
            <span>
              {p.done} / {p.total} <span className="text-sky-400 font-semibold">({pct}%)</span>
            </span>
          ) : (
            <span className="text-neutral-500 font-sans">Please wait…</span>
          )}
        </div>
      </div>
    </div>
  );
}

/** A folder in the sidebar tree: its subfolders are listed (one level, read when it opens) and its photos are read only when it is clicked. */
function FolderNode({
  path,
  depth,
  p,
  root,
  onRemove,
}: {
  path: string;
  depth: number;
  p: Props;
  /** a drive's own row, instead of the plain folder row */
  root?: { label: string; title: string; icon: ReactNode; count?: number; active: boolean; loading: boolean; progress?: FolderProgress; onClick: () => void };
  onRemove?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [kids, setKids] = useState<string[]>([]);
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null);
  const show = (v: boolean) => {
    setOpen(v);
    if (v) void api.browseFolder(path).then(([d]) => setKids(d), () => setKids([]));
  };
  const loading = p.loadingFolder?.viewId === `folder:${path}`;
  const row = root ?? {
    label: depth ? (path.split(/[\\/]/).filter(Boolean).pop() ?? path) : short(path),
    title: path,
    icon: <Icon d={ICONS.folder} />,
    count: p.folders.find(([d]) => d === path)?.[1],
    active: p.filter.view === `folder:${path}` || p.filter.folder === path,
    loading,
    progress: loading ? p.loadingFolder?.progress : undefined,
    onClick: () => p.onOpenFolder(path),
  };
  return (
    <>
      <div className="flex items-center" style={{ paddingLeft: depth * 12 }}>
        <button className="w-4 shrink-0 text-[10px] text-neutral-500 hover:text-neutral-200" title={open ? "Hide subfolders" : "Show subfolders"} onClick={() => show(!open)}>
          {open ? "▾" : "▸"}
        </button>
        <div className="min-w-0 flex-1">
          <FolderRow
            {...row}
            hue="amber"
            onContextMenu={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setMenuPos({ x: e.clientX, y: e.clientY });
            }}
            actions={
              <div className="flex items-center gap-0.5">
                <button
                  type="button"
                  className="px-1 text-neutral-400 hover:text-neutral-200"
                  title="Folder actions"
                  onClick={(e) => {
                    e.stopPropagation();
                    const rect = e.currentTarget.getBoundingClientRect();
                    setMenuPos({ x: rect.right, y: rect.bottom });
                  }}
                >
                  ···
                </button>
                {onRemove && (
                  <button type="button" className="px-1 text-neutral-400 hover:text-red-400" title="Remove from the list; nothing is deleted" onClick={(e) => { e.stopPropagation(); onRemove(); }}>
                    ×
                  </button>
                )}
              </div>
            }
            onClick={() => {
              row.onClick();
              if (!open) show(true);
            }}
          />
        </div>
      </div>
      {menuPos && (
        <div className="fixed inset-0 z-50 cursor-default" onClick={(e) => { e.stopPropagation(); setMenuPos(null); }}>
          <div
            className="fixed z-50 min-w-[200px] rounded-lg border border-neutral-700 bg-neutral-900/95 p-1 text-xs text-neutral-200 shadow-2xl backdrop-blur-md"
            style={{
              left: Math.min(menuPos.x, Math.max(10, window.innerWidth - 220)),
              top: Math.min(menuPos.y, Math.max(10, window.innerHeight - 170)),
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-neutral-500 truncate border-b border-neutral-800 mb-1">
              {path.split(/[\\/]/).filter(Boolean).pop() ?? path}
            </div>
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-neutral-200 hover:bg-neutral-800 hover:text-white cursor-pointer"
              onClick={() => {
                setMenuPos(null);
                void api.showInFolder(path);
              }}
            >
              <span className="text-neutral-400">↗</span>
              <span>Open in {fileManagerName()}</span>
            </button>
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-neutral-200 hover:bg-neutral-800 hover:text-white cursor-pointer"
              onClick={() => {
                setMenuPos(null);
                p.onOpenFolder(path);
              }}
            >
              <span className="text-neutral-400">↻</span>
              <span>Rescan Folder</span>
            </button>
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-neutral-200 hover:bg-neutral-800 hover:text-white cursor-pointer"
              onClick={async () => {
                setMenuPos(null);
                await api.regenerateThumbnails({ folder: path });
              }}
            >
              <span className="text-neutral-400">⚡</span>
              <span>Regenerate Thumbnails</span>
            </button>
            {onRemove && (
              <>
                <div className="my-1 border-t border-neutral-800" />
                <button
                  type="button"
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-red-400 hover:bg-red-500/10 cursor-pointer"
                  onClick={() => {
                    setMenuPos(null);
                    onRemove();
                  }}
                >
                  <span>×</span>
                  <span>Remove from Sidebar</span>
                </button>
              </>
            )}
          </div>
        </div>
      )}
      {open && kids.map((k) => <FolderNode key={k} path={k} depth={depth + 1} p={p} />)}
    </>
  );
}

function FolderRow({
  label,
  count,
  active,
  loading,
  progress,
  onClick,
  onContextMenu,
  title,
  icon,
  actions,
  indent,
  hue,
}: {
  label: string;
  count?: number;
  active: boolean;
  loading?: boolean;
  progress?: FolderProgress;
  onClick: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
  title?: string;
  icon?: ReactNode;
  /** small buttons shown in place of the count on hover or focus */
  actions?: ReactNode;
  /** under a sub-heading (an account, Immich) */
  indent?: boolean;
  /** its group's colour */
  hue: Hue;
}) {
  const h = HUE[hue];
  return (
    <div
      role="button"
      tabIndex={0}
      title={title}
      onClick={onClick}
      onContextMenu={onContextMenu}
      onKeyDown={(e) => e.target === e.currentTarget && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onClick())}
      className={`group flex cursor-pointer items-center gap-1.5 rounded py-1 pr-1.5 text-[11px] outline-none transition-colors duration-150 focus-visible:ring-1 focus-visible:ring-[var(--theme-accent,#0ea5e9)] ${indent ? "pl-4" : "pl-2"} ${
        active ? `${h.soft} font-medium text-neutral-100` : loading ? "bg-neutral-800/80 text-neutral-100" : "text-neutral-300 hover:bg-neutral-800/60"
      }`}
    >
      {loading ? (
        <svg className={`h-3.5 w-3.5 shrink-0 animate-spin ${h.text}`} viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
      ) : (
        <span className={`flex shrink-0 ${h.text}`}>{icon}</span>
      )}
      <span className="flex-1 truncate">{label}</span>
      {loading ? (
        <span className={`flex items-center gap-1 font-mono text-[10px] tabular-nums ${h.text}`}>
          {progress && progress.total > 0 ? `${progress.done}/${progress.total}` : count != null ? count : "…"}
        </span>
      ) : (
        <>
          {count != null && <span className={`tabular-nums ${actions ? "group-focus-within:hidden group-hover:hidden" : ""} ${active ? "text-neutral-300" : "text-neutral-500"}`}>{count}</span>}
          {actions && (
            <span className="hidden items-center group-focus-within:flex group-hover:flex" onClick={(e) => e.stopPropagation()}>
              {actions}
            </span>
          )}
        </>
      )}
    </div>
  );
}

/** A panel's heading in its colour: a swatch for a sidebar group, the section's icon given `d`. */
function Heading({ label, hue, d }: { label: string; hue: Hue; d?: string }) {
  return (
    <span className="flex items-center gap-1.5">
      {d ? <Icon d={d} className={`h-3.5 w-3.5 ${HUE[hue].text}`} /> : <span className={`h-2 w-2 shrink-0 rounded-[3px] ${HUE[hue].bg}`} />}
      {label}
    </span>
  );
}

/** An account or service inside a sidebar group, with its own small buttons. */
function SubHead({ label, children }: { label: string; children?: ReactNode }) {
  return (
    <div className="mt-2 flex items-center justify-between pl-2 text-[10px] uppercase tracking-wider text-neutral-500">
      <span className="truncate">{label}</span>
      {children}
    </div>
  );
}

type MenuItem = [label: string, hint: string, run: () => void];

/** ＋ on a sidebar heading: what can be added there, in a popover that a click outside or Esc closes. */
function AddMenu({ title, items }: { title: string; items: MenuItem[] }) {
  const id = useId();
  const menu = useRef<HTMLDivElement>(null);
  return (
    <>
      <button
        title={title}
        popoverTarget={id}
        className="px-1 text-neutral-500 hover:text-neutral-100"
        // runs before the popover opens: under the button, kept inside the window
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          Object.assign(menu.current!.style, { top: `${r.bottom + 4}px`, left: `${Math.max(8, r.right - 240)}px` });
        }}
      >
        ＋
      </button>
      <div id={id} ref={menu} popover="auto" className="inset-auto m-0 w-60 rounded-lg border border-neutral-700 bg-neutral-900 p-1 text-[11px] font-normal normal-case tracking-normal text-neutral-200 shadow-2xl">
        {items.map(([label, hint, run]) => (
          <button key={label} className="flex w-full flex-col items-start rounded-md px-2 py-1.5 text-left hover:bg-neutral-800" onClick={() => (menu.current!.hidePopover(), run())}>
            <span className="font-medium text-neutral-100">{label}</span>
            <span className="text-[10px] text-neutral-500">{hint}</span>
          </button>
        ))}
      </div>
    </>
  );
}

/** Sidebar and right bar icons (see Icon) */
const ICONS = {
  photos: "M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z M3 16l5-5 4 4 3-3 6 6 M16 7.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z",
  trash: "M4 7h16 M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2 M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12 M10 11v6 M14 11v6",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  drive: "M3 13l3-8h12l3 8 M3 13h18v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M7 16.5h2",
  cloud: "M7 18a4 4 0 0 1-.6-7.95A6 6 0 0 1 18 9.5a4.25 4.25 0 0 1-.5 8.5z",
  star: "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z",
  album: "M7 4h10 M5 7h14 M4 10h16v10H4z",
  info: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z M12 11v5 M12 7.5v.5",
  bolt: "M13 3L5 13.5h6L10 21l8-10.5h-6z",
};

export function Cell({
  photo,
  size,
  selected,
  active,
  inWorking,
  aiBusy,
  onClick,
  onOpen,
  onView,
}: {
  photo: Photo;
  size: number;
  selected: boolean;
  active: boolean;
  inWorking?: boolean;
  aiBusy?: boolean;
  onClick: (m: SelectMods) => void;
  onOpen: () => void;
  onView?: () => void;
}) {
  return (
    <div
      onClick={(e) => onClick({ shift: e.shiftKey, toggle: e.metaKey || e.ctrlKey })}
      onDoubleClick={onView || onOpen}
      className={`group relative flex cursor-pointer flex-col rounded-md p-1.5 ${
        active ? "bg-neutral-600" : selected ? "bg-neutral-700" : "bg-neutral-800 hover:bg-neutral-700/60"
      } ${aiBusy ? "ring-1 ring-violet-500/80" : ""} ${photo.flag === -1 ? "opacity-40" : ""}`}
    >
      <div className="relative flex items-center justify-center overflow-hidden rounded" style={{ height: size * 0.72 }}>
        <img src={thumbUrl(photo)} loading="lazy" draggable={false} className="max-h-full max-w-full object-contain shadow-md" alt="" />
        
        {/* Quick action overlay buttons on hover */}
        <div className="absolute inset-0 flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 backdrop-blur-xs rounded pointer-events-none">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              if (onView) onView();
              else onOpen();
            }}
            className="pointer-events-auto flex items-center gap-1.5 rounded-md bg-white hover:bg-slate-100 text-slate-900 px-2.5 py-1 text-[11px] font-semibold border border-white/90 shadow-lg cursor-pointer transition-all active:scale-95"
            title="Open in Photo Viewer (ACDSee folder image view - V)"
          >
            <svg className="h-3.5 w-3.5 text-slate-700" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
            <span>View</span>
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpen();
            }}
            className="pointer-events-auto flex items-center gap-1.5 rounded-md bg-[var(--theme-accent,#0ea5e9)] hover:bg-[var(--theme-accent-hover,#0284c7)] text-[var(--theme-accent-contrast,#ffffff)] px-2.5 py-1 text-[11px] font-semibold shadow-lg cursor-pointer transition-all active:scale-95"
            title="Open in Develop module (D)"
          >
            <Icon d="M12 20h9 M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" className="h-3.5 w-3.5" />
            <span>Develop</span>
          </button>
        </div>

        {aiBusy ? (
          <span
            className="pointer-events-none absolute top-1 right-1 flex items-center gap-1 rounded-full bg-violet-950/95 px-2 py-0.5 text-[9px] font-semibold text-violet-200 border border-violet-600/80 shadow-md z-10"
            title="AI fine tune / edit is working on this photo in Develop"
          >
            <span className="h-1.5 w-1.5 animate-ping rounded-full bg-violet-300" />
            ✦ AI working…
          </span>
        ) : inWorking ? (
          <span
            className="pointer-events-none absolute top-1 right-1 rounded bg-sky-950/90 px-1.5 py-0.5 text-[9px] font-medium text-sky-300 border border-sky-800/70 shadow-sm z-10"
            title="Open in Develop working images"
          >
            Develop
          </span>
        ) : null}
      </div>
      <div className="mt-1 flex items-center gap-1 text-[10px] text-neutral-400">
        <Stars value={photo.rating} />
        {photo.flag === 1 && <span className="text-neutral-100" title="Pick">⚑</span>}
        {photo.flag === -1 && <span className="text-red-400" title="Reject">✕</span>}
        {photo.color_label && <span className="h-2 w-2 rounded-full" style={{ background: LABEL_COLORS[photo.color_label] }} />}
        <span className="ml-auto flex gap-1 items-center">
          {photo.copy_name && (
            <span
              className="rounded bg-sky-950/90 px-1 py-0.2 text-[9px] font-medium text-sky-300 border border-sky-800/60"
              title={`Saved new edited photo: ${photo.copy_name}`}
            >
              {photo.copy_name}
            </span>
          )}
          {photo.edited && (
            <span
              className="badge-edited"
              title="Edited photo (adjustments applied)"
            >
              ±
            </span>
          )}
        </span>
      </div>
      <div className="truncate text-[10px] text-neutral-500">{photo.filename}</div>
    </div>
  );
}

function Metadata(p: Props & { photo: Photo }) {
  const ph = p.photo;
  const [keywords, setKeywords] = useState(ph.keywords);
  useEffect(() => setKeywords(ph.keywords), [ph.id, ph.keywords]);
  const rows: [string, string | number | null][] = [
    ["File", ph.filename + (ph.copy_name ? ` (${ph.copy_name})` : "")],
    ["Folder", ph.folder],
    ["Size", ph.width ? `${ph.width} × ${ph.height}` : null],
    ["Captured", ph.captured_at],
    ["Camera", ph.camera],
    ["Lens", ph.lens],
    ["ISO", ph.iso],
    ["Aperture", ph.aperture ? `f/${ph.aperture.toFixed(1)}` : null],
    ["Shutter", ph.shutter ? `${ph.shutter} s` : null],
    ["Focal length", ph.focal_length ? `${Math.round(ph.focal_length)} mm` : null],
  ];
  return (
    <>
      <Panel title={<Heading label="Quick culling" hue="amber" d={ICONS.star} />}>
        <div className="space-y-2 text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-neutral-400">Rating</span>
            <span className="text-base">
              <Stars value={ph.rating} onChange={(r) => p.cull({ rating: r })} />
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-neutral-400">Flag</span>
            <span className="flex gap-1">
              {(["pick", "unflagged", "reject"] as const).map((fl) => (
                <Btn key={fl} active={flagName(ph.flag) === fl} onClick={() => p.cull({ flag: fl })}>
                  {fl === "pick" ? "⚑ Pick" : fl === "reject" ? "✕ Reject" : "—"}
                </Btn>
              ))}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-neutral-400">Label</span>
            <span className="flex gap-1.5">
              {LABELS.map((l) => (
                <button
                  key={l}
                  title={l}
                  onClick={() => p.cull({ color_label: ph.color_label === l ? "none" : l })}
                  className={`h-4 w-4 rounded-full ${ph.color_label === l ? "ring-2 ring-white" : ""}`}
                  style={{ background: LABEL_COLORS[l] }}
                />
              ))}
            </span>
          </div>
          <input
            className="w-full rounded bg-neutral-800 px-2 py-1 outline-none placeholder:text-neutral-500"
            placeholder="Keywords, comma separated"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            onBlur={() => keywords !== ph.keywords && p.cull({ keywords })}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          />
          <p className="text-[10px] text-neutral-500">Keys: 0–5 rating · P/X/U flag · 6–9 label · D develop · ←/→ navigate</p>
        </div>
      </Panel>
      <Panel title={<Heading label="Metadata" hue="blue" d={ICONS.info} />}>
        <dl className="grid grid-cols-[80px_1fr] gap-x-2 gap-y-1 text-[11px]">
          {rows
            .filter(([, v]) => v != null && v !== "")
            .map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-neutral-500">{k}</dt>
                {k === "Folder" ? (
                  <dd className="break-all text-neutral-200 flex items-center justify-between gap-1 group">
                    <span className="truncate" title={String(v)}>{v}</span>
                    <button
                      type="button"
                      onClick={() => void api.showInFolder(ph.path)}
                      className="shrink-0 text-neutral-400 hover:text-sky-400 p-0.5 rounded cursor-pointer transition-colors"
                      title={`Open in ${fileManagerName()}`}
                    >
                      ↗
                    </button>
                  </dd>
                ) : (
                  <dd className="break-all text-neutral-200">{v}</dd>
                )}
              </div>
            ))}
        </dl>
      </Panel>
      <Panel title={<Heading label="Actions" hue="emerald" d={ICONS.bolt} />}>
        <div className="flex flex-wrap gap-1">
          {p.filter.view === "bin" ? (
            <>
              <Btn onClick={p.onRestoreFromBin} className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold">
                ↩ Restore
              </Btn>
              <Btn onClick={p.onPermanentDelete} className="bg-red-600 hover:bg-red-500 text-white font-semibold" title="Permanently delete original file(s) from disk">
                ✕ Delete Permanently
              </Btn>
            </>
          ) : (
            <>
              <Btn
                onClick={() => p.onOpenViewer?.(ph.id)}
                title="Open photo in Photo Viewer (ACDSee folder image view - V or Enter)"
                className="bg-[var(--theme-accent,#0ea5e9)] text-[var(--theme-accent-contrast,#ffffff)] font-semibold shadow-xs"
              >
                ▣ View Photo
              </Btn>
              <Btn onClick={() => p.openDevelop(ph.id)} title={p.selected.length > 1 ? `Open all ${p.selected.length} selected photos as working images in Develop` : "Open photo in Develop (D)"}>
                Develop{p.selected.length > 1 ? ` (${p.selected.length})` : ""}
              </Btn>
              {p.onAddWorking && (() => {
                const targets = p.selected.length ? p.selected : [ph.id];
                const newIds = targets.filter((id) => !p.workingIds?.includes(id));
                return newIds.length > 0 ? (
                  <Btn
                    onClick={() => p.onAddWorking?.(newIds)}
                    title="Add selected photo(s) to Develop working images while staying in Library"
                  >
                    ＋ Add to Develop{newIds.length > 1 ? ` (${newIds.length})` : ""}
                  </Btn>
                ) : null;
              })()}
              <Btn onClick={p.onExport}>
                {ph.path.startsWith("immich://") ? "Save to Immich" : "Export"} {p.selected.length > 1 ? `${p.selected.length}` : ""}…
              </Btn>
              <Btn onClick={p.onVirtualCopy}>Virtual copy</Btn>
              <Btn
                onClick={() => void api.showInFolder(ph.path)}
                title={`Open and reveal file in ${fileManagerName()}`}
              >
                Open in {fileManagerName()}
              </Btn>
              <Btn
                onClick={async () => {
                  const targets = p.selected.length ? p.selected : [ph.id];
                  await api.regenerateThumbnails({ ids: targets });
                }}
                title="Re-render thumbnail directly from source file on disk"
              >
                Regenerate thumbnail{p.selected.length > 1 ? `s (${p.selected.length})` : ""}
              </Btn>
              <Btn
                onClick={async () => {
                  const targets = p.selected.length ? p.selected : [ph.id];
                  await api.xmpSync(targets, "write");
                }}
                title="Write Adobe Lightroom .XMP sidecar (ratings, flags, color labels, keywords, develop & crop)"
              >
                Write .XMP{p.selected.length > 1 ? ` (${p.selected.length})` : ""}
              </Btn>
              <Btn
                onClick={async () => {
                  const targets = p.selected.length ? p.selected : [ph.id];
                  await api.xmpSync(targets, "read");
                }}
                title="Import ratings, labels, keywords & edits from Adobe Lightroom .XMP sidecar"
              >
                Read .XMP
              </Btn>
              {/\.(jpe?g)$/i.test(ph.filename) && (
                <Btn
                  onClick={async () => {
                    await api.jpegLossless(ph.id, "rot90");
                  }}
                  title="SIMD Lossless JPEG 90° DCT block rotation (libturbojpeg, zero recompression)"
                >
                  Lossless 90° ⟳
                </Btn>
              )}
              {/* the ones that take photos away, on their own line under the everyday ones */}
              <div className="mt-1 flex w-full flex-wrap gap-1 border-t border-neutral-800 pt-2">
                <Btn onClick={p.onMoveToBin} title="Move photo to Bin (can be restored later)" className="text-amber-400 hover:text-amber-300">
                  Move to Bin
                </Btn>
                <Btn onClick={p.onRemove} title="Removes from the catalog only; files stay on disk">
                  Remove from catalog
                </Btn>
                <Btn onClick={p.onPermanentDelete} title="Permanently delete photo file from disk" className="text-red-400 hover:text-red-300">
                  Delete permanently
                </Btn>
              </div>
            </>
          )}
        </div>
      </Panel>
    </>
  );
}

const lastFolder = () => {
  try {
    return localStorage.getItem("ols.exportFolder") ?? "";
  } catch {
    return "";
  }
};

/** dest: "" (a folder) | immich | google | cloud:<account id> */
export function ExportDialog({ ids, settings, dest: initialDest = "", onClose }: { ids: number[]; settings: Settings | null; dest?: string; onClose: () => void }) {
  const [format, setFormat] = useState("JPEG");
  const [space, setSpace] = useState("sRGB");
  const [quality, setQuality] = useState(90);
  const [edge, setEdge] = useState("");
  const [folder, setFolder] = useState(lastFolder);
  const [dest, setDest] = useState(initialDest); // "" = folder
  const [uploadPath, setUploadPath] = useState("Optixel exports");
  const account = dest.startsWith("cloud:") ? settings?.clouds.find((a) => `cloud:${a.id}` === dest) : undefined;
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState<string | null>(null);
  useEffect(() => {
    const un = listen<{ label?: string; done: number; total: number }>("export-progress", (e) => setProgress(`${e.payload.label ?? "Exporting"} ${e.payload.done}/${e.payload.total}`));
    return () => void un.then((f) => f());
  }, []);
  const hdrOk = format === "PNG_16" || format === "TIFF_16";
  const run = async () => {
    setBusy(true);
    setResult(null);
    setProgress("");
    try {
      localStorage.setItem("ols.exportFolder", folder);
    } catch {
      /* per-viewer convenience only */
    }
    try {
      const files = await api.export({ ids, folder, format, color_space: space, quality, max_edge: edge ? +edge : null, upload: dest || null, upload_path: uploadPath });
      const where = account
        ? `${account.name} › ${uploadPath.trim() || "/"}`
        : dest === "immich"
          ? "Immich"
          : dest === "google"
            ? "Google Photos"
            : dest === "apple"
              ? "Apple Photos / iCloud"
              : dest === "amazon"
                ? "Amazon Photos"
                : dest === "adobe"
                  ? "Adobe Creative Cloud"
                  : dest;
      setResult(dest ? `Uploaded ${files.length} photo(s) to ${where}` : `Exported ${files.length} file(s) to ${folder}`);
    } catch (e) {
      setResult(`Export failed: ${e}`);
    }
    setBusy(false);
  };
  const field = "rounded bg-neutral-800 px-2 py-1 outline-none";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => !busy && onClose()}>
      <div className="w-[440px] space-y-3 rounded-lg border border-neutral-700 bg-neutral-900 p-5 text-[12px] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-sm font-semibold">Export {ids.length} photo(s)</h2>
        <div className="grid grid-cols-[110px_minmax(0,1fr)] items-center gap-2">
          <span className="text-neutral-400">Format</span>
          <select className={field} value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="JPEG">JPEG</option>
            <option value="AVIF">AVIF</option>
            <option value="PNG_16">PNG 16-bit</option>
            <option value="TIFF_16">TIFF 16-bit</option>
          </select>
          <span className="text-neutral-400">Color space</span>
          <select className={field} value={space} onChange={(e) => setSpace(e.target.value)}>
            <option value="sRGB">sRGB IEC61966-2.1</option>
            <option value="DisplayP3">Apple Display P3 / XDR</option>
            <option value="AdobeRGB">Adobe RGB (1998)</option>
            <option value="ProPhotoRGB">ProPhoto RGB (ROMM D50 Wide-Gamut)</option>
            <option value="ACEScg" disabled={!hdrOk}>
              ACEScg (Linear AP1 D60){hdrOk ? "" : " — 16-bit formats only"}
            </option>
            <option value="Rec2020_HDR" disabled={!hdrOk}>
              Rec. 2020 HDR (PQ){hdrOk ? "" : " — 16-bit formats only"}
            </option>
          </select>
          {(format === "JPEG" || format === "AVIF") && (
            <>
              <span className="text-neutral-400">Quality {quality}</span>
              <input type="range" className="ol-range" min={40} max={100} value={quality} onChange={(e) => setQuality(+e.target.value)} />
            </>
          )}
          <span className="text-neutral-400">Long edge (px)</span>
          <input className={field} placeholder="Full resolution" value={edge} onChange={(e) => setEdge(e.target.value.replace(/\D/g, ""))} />
          <span className="text-neutral-400">Destination</span>
          <select className={field} value={dest} onChange={(e) => setDest(e.target.value)}>
            <option value="">Folder</option>
            <option value="immich" disabled={!settings?.immich_url}>
              Upload to Immich{settings?.immich_url ? "" : " — connect it in Library first"}
            </option>
            <option value="google" disabled={!settings?.google_client_id}>
              Upload to Google Photos{settings?.google_client_id ? "" : " — connect it in Library first"}
            </option>
            <option value="apple" disabled={!settings?.apple_photos_path}>
              Save to Apple Photos / iCloud{settings?.apple_photos_path ? "" : " — connect it in Library first"}
            </option>
            <option value="amazon" disabled={!settings?.amazon_photos_path}>
              Upload to Amazon Photos{settings?.amazon_photos_path ? "" : " — connect it in Library first"}
            </option>
            <option value="adobe" disabled={!settings?.adobe_cloud_path}>
              Save to Adobe Creative Cloud{settings?.adobe_cloud_path ? "" : " — connect it in Library first"}
            </option>
            {settings?.clouds.map((a) => (
              <option key={a.id} value={`cloud:${a.id}`}>
                Upload to {a.name}
              </option>
            ))}
          </select>
          {account && (
            <>
              <span className="text-neutral-400">Into folder</span>
              <input className={field} value={uploadPath} onChange={(e) => setUploadPath(e.target.value)} placeholder="The top of the storage" />
            </>
          )}
          {!dest && (
            <>
              <span className="text-neutral-400">Folder</span>
              <div className="flex gap-1">
                <input className={`${field} min-w-0 flex-1`} value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="Choose a destination" />
                <Btn
                  onClick={async () => {
                    const d = await open({ directory: true, defaultPath: folder || undefined });
                    if (typeof d === "string") setFolder(d);
                  }}
                >
                  Choose…
                </Btn>
              </div>
            </>
          )}
        </div>
        {result && <p className={result.startsWith("Export failed") ? "text-red-400" : "text-emerald-400"}>{result}</p>}
        <div className="flex justify-end gap-2">
          <Btn onClick={onClose} disabled={busy}>
            Close
          </Btn>
          <Btn active onClick={run} disabled={busy || (!dest && !folder) || (space === "Rec2020_HDR" && !hdrOk)}>
            {busy ? progress || "Exporting…" : dest ? "Upload" : "Export"}
          </Btn>
        </div>
      </div>
    </div>
  );
}

/** Connect or disconnect Immich / Google Photos / Apple iCloud / Amazon Photos / Adobe Cloud Photos. */
export function RemoteDialog({ source, settings, setSettings, onClose }: { source: Source; settings: Settings; setSettings: (s: Settings) => void; onClose: () => void }) {
  const titles: Record<Source, string> = {
    immich: "Immich",
    google: "Google Photos",
    apple: "Apple iCloud Photos",
    amazon: "Amazon Photos",
    adobe: "Adobe Cloud Photos",
  };
  const isConnected = () => {
    switch (source) {
      case "immich":
        return !!settings.immich_url;
      case "google":
        return !!settings.google_client_id;
      case "apple":
        return !!settings.apple_photos_path;
      case "amazon":
        return !!settings.amazon_photos_path;
      case "adobe":
        return !!settings.adobe_cloud_path;
    }
  };
  const connected = isConnected();
  const [url, setUrl] = useState(settings.immich_url);
  const [clientId, setClientId] = useState(settings.google_client_id);
  const [path, setPath] = useState(
    source === "apple"
      ? settings.apple_photos_path || ""
      : source === "amazon"
        ? settings.amazon_photos_path || ""
        : source === "adobe"
          ? settings.adobe_cloud_path || ""
          : "",
  );
  const [accountName, setAccountName] = useState("");
  const [secret, setSecret] = useState(""); // Immich API key or Google client secret
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!path) {
      if (source === "apple") {
        api.appleDetect().then((r) => r.found && setPath(r.default_path)).catch(() => {});
      } else if (source === "amazon") {
        api.amazonDetect().then((r) => r.found && setPath(r.default_path)).catch(() => {});
      } else if (source === "adobe") {
        api.adobeDetect().then((r) => r.found && setPath(r.default_path)).catch(() => {});
      }
    }
  }, [source, path]);

  const handleDetect = async () => {
    try {
      if (source === "apple") {
        const r = await api.appleDetect();
        if (r.default_path) setPath(r.default_path);
      } else if (source === "amazon") {
        const r = await api.amazonDetect();
        if (r.default_path) setPath(r.default_path);
      } else if (source === "adobe") {
        const r = await api.adobeDetect();
        if (r.default_path) setPath(r.default_path);
      }
    } catch (_) {}
  };

  const handleBrowse = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: `Choose ${titles[source]} folder or library`,
      });
      if (selected && typeof selected === "string") {
        setPath(selected);
      }
    } catch (_) {}
  };

  const run = async (label: string, job: () => Promise<Settings>) => {
    setBusy(label);
    setError("");
    try {
      setSettings(await job());
      onClose();
    } catch (e) {
      setError(String(e));
    }
    setBusy("");
  };

  const connectAction = () => {
    switch (source) {
      case "immich":
        return run("Connecting…", () => api.immichConnect(url, secret));
      case "google":
        return run("Finish signing in to Google in your browser…", () => api.googleConnect(clientId, secret));
      case "apple":
        return run("Connecting Apple iCloud Photos…", () => api.appleConnect(path));
      case "amazon":
        return run("Connecting Amazon Photos…", () => api.amazonConnect(path, accountName));
      case "adobe":
        return run("Connecting Adobe Cloud Photos…", () => api.adobeConnect(path, accountName));
    }
  };

  const canConnect = () => {
    if (busy) return false;
    switch (source) {
      case "immich":
        return !!url.trim() && !!secret.trim();
      case "google":
        return !!clientId.trim() && !!secret.trim();
      case "apple":
      case "amazon":
      case "adobe":
        return !!path.trim();
    }
  };

  const field = "rounded bg-neutral-800 px-2 py-1 outline-none placeholder:text-neutral-500";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div className="w-[490px] space-y-3 rounded-lg border border-neutral-700 bg-neutral-900 p-5 text-[12px] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">{titles[source]}</h2>
          {connected && (
            <Btn disabled={!!busy} onClick={() => run("Disconnecting…", () => api.remoteDisconnect(source))}>
              Disconnect
            </Btn>
          )}
        </div>
        {!connected ? (
          <>
            {source === "immich" && (
              <div className="grid grid-cols-[110px_1fr] items-center gap-2">
                <span className="text-neutral-400">Server URL</span>
                <input className={field} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="http://192.168.1.10:2283" />
                <span className="text-neutral-400">API key</span>
                <input className={field} type="password" value={secret} onChange={(e) => setSecret(e.target.value)} />
              </div>
            )}
            {source === "google" && (
              <div className="grid grid-cols-[110px_1fr] items-center gap-2">
                <span className="text-neutral-400">Client ID</span>
                <input className={field} value={clientId} onChange={(e) => setClientId(e.target.value)} placeholder="….apps.googleusercontent.com" />
                <span className="text-neutral-400">Client secret</span>
                <input className={field} type="password" value={secret} onChange={(e) => setSecret(e.target.value)} />
              </div>
            )}
            {(source === "apple" || source === "amazon" || source === "adobe") && (
              <div className="grid grid-cols-[110px_1fr] items-center gap-2">
                <span className="text-neutral-400">
                  {source === "apple" ? "Library or Folder" : "Sync Directory"}
                </span>
                <div className="flex items-center gap-1.5">
                  <input
                    className={`${field} min-w-0 flex-1`}
                    value={path}
                    onChange={(e) => setPath(e.target.value)}
                    placeholder={
                      source === "apple"
                        ? "~/Pictures/Photos Library.photoslibrary"
                        : source === "amazon"
                          ? "~/Pictures/Amazon Photos"
                          : "~/Creative Cloud Files"
                    }
                  />
                  <Btn onClick={handleBrowse} title="Choose directory with file dialog">
                    Choose…
                  </Btn>
                  <Btn onClick={handleDetect} title="Auto-detect standard path on this computer">
                    Auto-detect
                  </Btn>
                </div>
                {(source === "amazon" || source === "adobe") && (
                  <>
                    <span className="text-neutral-400">Account (optional)</span>
                    <input
                      className={field}
                      value={accountName}
                      onChange={(e) => setAccountName(e.target.value)}
                      placeholder="e.g. Personal or Studio"
                    />
                  </>
                )}
              </div>
            )}
            <p className="text-[11px] leading-relaxed text-neutral-400">
              {source === "immich" &&
                "Create a key in Immich under Account Settings → API Keys with album.read, asset.read, asset.view, asset.download, asset.upload, asset.update and stack.create. Stored securely in OS keychain."}
              {source === "google" &&
                "In Google Cloud Console, enable the Photos Picker API (and Photos Library API to upload), create an OAuth client of type Desktop app. Signing in opens your browser."}
              {source === "apple" &&
                "Connect your Apple Photos Library (.photoslibrary package) or iCloud Photos sync directory. Photos and RAW originals are indexed non-destructively; exports can be saved directly back."}
              {source === "amazon" &&
                "Connect your Amazon Photos or Amazon Drive local synchronization directory. Optixel indexes your full-resolution photos and RAWs; edits can be uploaded back."}
              {source === "adobe" &&
                "Connect your Adobe Creative Cloud Files or Lightroom photography synchronization folder. Changes and exported photos automatically sync across Creative Cloud apps."}
            </p>
          </>
        ) : (
          <div className="space-y-1 text-neutral-300">
            {source === "immich" && (
              <p>
                Connected to <span className="text-neutral-100">{settings.immich_url}</span>. Albums appear in the sidebar. Opening a photo downloads it once; Save to Immich stacks your edit on top of the original.
              </p>
            )}
            {source === "google" && (
              <p>
                Signed in with Google Photos. Pick photos opens the picker in your browser. Picked photos download into your catalog and can be developed.
              </p>
            )}
            {source === "apple" && (
              <p>
                Connected to <span className="font-mono text-neutral-100">{settings.apple_photos_path}</span>. Syncing indexes your Photos Library originals. Exporting can save rendered photos back to Apple Photos.
              </p>
            )}
            {source === "amazon" && (
              <p>
                Connected to <span className="font-mono text-neutral-100">{settings.amazon_photos_path}</span>. Syncing reads full-resolution photos & RAW files from Amazon Photos.
              </p>
            )}
            {source === "adobe" && (
              <p>
                Connected to <span className="font-mono text-neutral-100">{settings.adobe_cloud_path}</span>. Synced photos appear directly in Optixel Studio, and exports sync with Lightroom & Creative Cloud.
              </p>
            )}
          </div>
        )}
        {busy && <p className="text-sky-300">{busy}</p>}
        {error && <p className="break-words text-red-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <Btn onClick={onClose}>Close</Btn>
          {!connected && (
            <Btn
              active
              disabled={!canConnect()}
              onClick={connectAction}
            >
              {source === "immich"
                ? "Connect"
                : source === "google"
                  ? "Sign in with Google"
                  : `Connect ${titles[source]}`}
            </Btn>
          )}
        </div>
      </div>
    </div>
  );
}
