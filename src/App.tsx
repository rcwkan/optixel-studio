import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import { useCallback, useEffect, useRef, useState } from "react";
import { Log } from "./Agent";
import { api, AppInfo, AutoFunction, CloudFolder, Culling, Develop as Dev, Filter, FolderLoading, forgetRoot, getFirstFolder, loadLastFolder, loadRoots, rememberLastFolder, rememberRoot, JobProgress, LABELS, NetworkDrive, Photo, setThumbsDir, Settings, Source, UpdateInfo } from "./api";
import { AiTaskState, Develop } from "./Develop";
import { ExportDialog, Library, RemoteDialog, SelectMods } from "./Library";
import { CompareModal } from "./CompareModal";
import { SettingsTab, SettingsView } from "./Settings";
import { UpdateModal } from "./UpdateModal";
import { People } from "./People";
import { applyTheme, getStoredTheme } from "./theme";

type Module = "library" | "people" | "develop" | "settings";
const MODULES: [Module, string][] = [
  ["library", "Library"],
  ["people", "People"],
  ["develop", "Develop"],
  ["settings", "Settings"],
];

export default function App() {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [module, setModule] = useState<Module>("library");
  const [libraryAutoFn, setLibraryAutoFn] = useState<AutoFunction | null>(null);
  const [filter, setFilter] = useState<Filter>(() => {
    const last = loadLastFolder();
    if (last) {
      return { folder: last, flat: true, view: `folder:${last}` };
    }
    const r = loadRoots();
    if (r.length > 0) {
      return { folder: r[0], flat: true, view: `folder:${r[0]}` };
    }
    return { view: "loading" };
  });
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [folders, setFolders] = useState<[string, number][]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [defaults, setDefaults] = useState<Dev | null>(null);
  const [logs, setLogs] = useState<Log[]>([]);
  const [status, setStatus] = useState("");
  const [regenProgress, setRegenProgress] = useState<{ done: number; total: number } | null>(null);
  const [exportIds, setExportIds] = useState<number[] | null>(null);
  const [remote, setRemote] = useState<Source | null>(null);
  const [albumPages, setAlbumPages] = useState<{ album: string | null; next: number | null } | null>(null);
  const loadingPage = useRef(false);
  const [jobs, setJobs] = useState<Record<string, JobProgress>>({});
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("appearance");
  const [loadingFolder, setLoadingFolder] = useState<FolderLoading | null>(null);
  const [binCount, setBinCount] = useState<number>(0);
  const [compareOpen, setCompareOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<{ ids: number[] } | null>(null);
  const [workingIds, setWorkingIds] = useState<number[]>([]);
  const [developId, setDevelopId] = useState<number | null>(null);
  const [workingPhotosMap, setWorkingPhotosMap] = useState<Record<number, Photo>>({});
  const [aiTasks, setAiTasks] = useState<Record<number, AiTaskState>>({});
  const [initialLoading, setInitialLoading] = useState(true);

  // App software updater states
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [updateLoading, setUpdateLoading] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [showUpdateModal, setShowUpdateModal] = useState(false);

  const checkForUpdates = useCallback(async (manual = false) => {
    try {
      setUpdateLoading(true);
      setUpdateError(null);
      const res = await api.checkForUpdates();
      setUpdateInfo(res);
      if (manual) {
        setShowUpdateModal(true);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setUpdateError(msg);
      if (manual) {
        setShowUpdateModal(true);
      }
    } finally {
      setUpdateLoading(false);
    }
  }, []);

  // Background check on startup (slightly delayed to prevent blocking catalog setup)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (settings?.auto_check_updates !== false) {
        checkForUpdates(false);
      }
    }, 3000);
    return () => clearTimeout(timer);
  }, [settings?.auto_check_updates, checkForUpdates]);

  const initialResolved = useRef(false);

  const reload = useCallback(async () => {
    if (!initialResolved.current) return;
    const [ps, fs, bc] = await Promise.all([
      api.query(filter),
      api.folders(),
      api.binCount().catch(() => 0),
    ]);
    setPhotos(ps);
    setFolders(fs);
    setBinCount(bc);
  }, [filter]);
  const reloadRef = useRef(reload);
  reloadRef.current = reload;
  useEffect(() => void reload(), [reload]);

  useEffect(() => {
    if (!photos.length) return;
    setWorkingPhotosMap((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const p of photos) {
        if (next[p.id] !== p) {
          next[p.id] = p;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [photos]);

  const refreshPhoto = useCallback(async (id: number) => {
    const ph = await api.photo(id).catch(() => null);
    if (ph) {
      setPhotos((ps) => ps.map((p) => (p.id === ph.id ? ph : p)));
      setWorkingPhotosMap((prev) => ({ ...prev, [ph.id]: ph }));
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      const storedLast = loadLastFolder();
      const currentRoots = loadRoots();
      const initialCandidate = storedLast || currentRoots[0] || null;

      const [appInfoRes, settingsRes, , initialPhotos, fs, bc] = await Promise.all([
        api.appInfo().then((i) => {
          setThumbsDir(i.thumbs_dir);
          setInfo(i);
          return i;
        }).catch(() => null),
        api.settings().then((s) => {
          setSettings(s);
          if (s.theme) {
            applyTheme(s.theme);
          } else {
            applyTheme(getStoredTheme());
          }
          return s;
        }).catch(() => null),
        api.defaultDevelop().then(setDefaults).catch(() => null),
        initialCandidate ? api.query({ folder: initialCandidate, flat: true }).catch(() => [] as Photo[]) : Promise.resolve([] as Photo[]),
        api.folders().catch(() => [] as [string, number][]),
        api.binCount().catch(() => 0),
      ]);

      if (!mounted) return;
      setFolders(fs);
      setBinCount(bc);

      const drives = settingsRes?.network_drives ?? [];
      const dataDir = appInfoRes?.data_dir ?? "";
      let targetFolder: string | null = null;

      if (storedLast && (fs.some(([f]) => f === storedLast) || currentRoots.some((r) => storedLast === r || storedLast.startsWith(r)))) {
        targetFolder = storedLast;
      } else {
        targetFolder = getFirstFolder(currentRoots, fs, dataDir, drives);
      }

      initialResolved.current = true;

      if (targetFolder) {
        rememberLastFolder(targetFolder);
        const nextFilter: Filter = { folder: targetFolder, flat: true, view: `folder:${targetFolder}` };
        setFilter(nextFilter);
        if (targetFolder === initialCandidate) {
          setPhotos(initialPhotos);
        } else {
          const ps = await api.query(nextFilter).catch(() => []);
          if (mounted) setPhotos(ps);
        }
      } else {
        setFilter({});
        const ps = await api.query({}).catch(() => []);
        if (mounted) setPhotos(ps);
      }

      setTimeout(() => {
        if (mounted) setInitialLoading(false);
      }, 200);
    })();

    const handleContextMenu = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      e.preventDefault();
    };
    window.addEventListener("contextmenu", handleContextMenu);

    const subs = [
      listen("catalog-changed", () => reloadRef.current()),
      listen<{ new_dir: string; is_custom: boolean; mode: string }>("storage-location-changed", async (e) => {
        try {
          const info = await api.appInfo();
          setInfo(info);
          setThumbsDir(info.thumbs_dir);
        } catch {
          const sep = e.payload.new_dir.includes("\\") ? "\\" : "/";
          setThumbsDir(`${e.payload.new_dir}${sep}thumbs`);
        }
        void api.settings().then(setSettings);
        reloadRef.current();
      }),
      listen<{ done: number; total: number }>("regenerate-thumbs-progress", (e) => {
        setRegenProgress(e.payload.done >= e.payload.total ? null : e.payload);
        setStatus(`Regenerating thumbnails: ${e.payload.done} / ${e.payload.total}`);
        if (e.payload.done >= e.payload.total) {
          setTimeout(() => {
            setStatus("");
            setRegenProgress(null);
          }, 2500);
        }
      }),
      listen<{ id: number }>("thumb-changed", async (e) => {
        const ph = await api.photo(e.payload.id);
        if (ph) {
          setPhotos((ps) => ps.map((p) => (p.id === ph.id ? ph : p)));
          setWorkingPhotosMap((prev) => ({ ...prev, [ph.id]: ph }));
        }
      }),
      listen<{ label?: string; done: number; total: number }>("import-progress", (e) => {
        setStatus(`${e.payload.label ?? "Adding photos"} ${e.payload.done} / ${e.payload.total}`);
        setLoadingFolder((cur) => (cur ? { ...cur, progress: e.payload } : cur));
      }),
      listen<string>("remote-error", (e) => setStatus(e.payload)),
      listen<JobProgress>("job-progress", (e) =>
        setJobs((js) => {
          const { [e.payload.job]: _, ...rest } = js;
          return e.payload.finished ? rest : { ...rest, [e.payload.job]: e.payload };
        }),
      ),
      listen<{ source: string; message: string }>("agent-log", (e) => setLogs((l) => [...l.slice(-499), { t: new Date().toLocaleTimeString(), ...e.payload }])),
    ];
    return () => {
      mounted = false;
      window.removeEventListener("contextmenu", handleContextMenu);
      subs.forEach((s) => s.then((f) => f()));
    };
  }, []);
  useEffect(() => void (module === "settings" && settingsTab === "agent" && api.appInfo().then(setInfo)), [module, settingsTab]);
  useEffect(() => void api.setSelection(selected), [selected]);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      const cur = settings?.theme ?? getStoredTheme();
      if (cur === "system") {
        applyTheme("system");
      }
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [settings?.theme]);

  const active = photos.find((p) => p.id === activeId) ?? null;
  const workingPhotos = workingIds
    .map((id) => photos.find((p) => p.id === id) ?? workingPhotosMap[id])
    .filter((p): p is Photo => Boolean(p));
  const developActive =
    (developId != null ? workingPhotos.find((p) => p.id === developId) ?? photos.find((p) => p.id === developId) ?? workingPhotosMap[developId] : null) ??
    workingPhotos[0] ??
    active;
  const busyAiIds = Object.entries(aiTasks)
    .filter(([, t]) => t.busy)
    .map(([id]) => Number(id));
  // Auto Catalog & Culling of new photos runs in the background
  const culling = Object.values(jobs).find((j) => j.job.startsWith("on-import-"));
  const background = culling ? `Culling new photos${culling.total ? ` ${culling.done} / ${culling.total}` : "…"}` : "";
  const targets = () => (selected.length ? selected : activeId != null ? [activeId] : []);

  const addWorkingImages = useCallback(
    (ids: number[], focusTarget?: number) => {
      const valid = ids.filter((id) => id != null);
      if (!valid.length) return;
      setWorkingIds((prev) => {
        const next = [...prev];
        for (const id of valid) {
          if (!next.includes(id)) next.push(id);
        }
        return next;
      });
      for (const id of valid) {
        const found = photos.find((p) => p.id === id);
        if (found) {
          setWorkingPhotosMap((prev) => ({ ...prev, [id]: found }));
        } else {
          void api.photo(id).then((ph) => {
            if (ph) setWorkingPhotosMap((prev) => ({ ...prev, [id]: ph }));
          });
        }
      }
      if (focusTarget != null) {
        setDevelopId(focusTarget);
        setActiveId(focusTarget);
      }
    },
    [photos],
  );

  const openInDevelop = (targetId?: number) => {
    const toAdd =
      targetId != null
        ? selected.length > 1 && selected.includes(targetId)
          ? selected
          : [targetId]
        : selected.length > 0
          ? selected
          : activeId != null
            ? [activeId]
            : [];
    const primary = targetId ?? activeId ?? toAdd[0] ?? developId;
    if (toAdd.length) {
      addWorkingImages(toAdd, primary ?? undefined);
    } else if (primary != null) {
      setDevelopId(primary);
    }
    setModule("develop");
  };

  const closeWorkingImage = (id: number) => {
    setWorkingIds((prev) => {
      const idx = prev.indexOf(id);
      if (idx === -1) return prev;
      const next = prev.filter((x) => x !== id);
      if (developId === id) {
        const fallback = next[Math.min(idx, next.length - 1)] ?? null;
        setDevelopId(fallback);
        if (fallback != null) setActiveId(fallback);
      }
      if (next.length === 0) {
        setModule("library");
      }
      return next;
    });
  };

  const closeOtherWorkingImages = (keepId: number) => {
    setWorkingIds([keepId]);
    setDevelopId(keepId);
    setActiveId(keepId);
  };

  const select = (id: number, m: SelectMods) => {
    setSelected((sel) => {
      if (m.toggle) return sel.includes(id) ? sel.filter((s) => s !== id) : [...sel, id];
      if (m.shift && activeId != null) {
        const [a, b] = [photos.findIndex((p) => p.id === activeId), photos.findIndex((p) => p.id === id)];
        return photos.slice(Math.min(a, b), Math.max(a, b) + 1).map((p) => p.id);
      }
      return [id];
    });
    if (!m.shift) setActiveId(id);
  };
  const focus = (id: number) => (setActiveId(id), setSelected([id]));
  const focusDevelop = (id: number) => {
    addWorkingImages([id], id);
    setSelected([id]);
  };

  const cull = async (u: Culling, idsOverride?: number[]) => {
    const ids = idsOverride || targets();
    if (!ids.length) return;
    try {
      await api.update(ids, u);
    } catch (e) {
      return setStatus(String(e));
    }
    setPhotos((ps) =>
      ps.map((p) =>
        ids.includes(p.id)
          ? {
            ...p,
            ...(u.rating != null && { rating: u.rating }),
            ...(u.flag && { flag: u.flag === "pick" ? 1 : u.flag === "reject" ? -1 : 0 }),
            ...(u.color_label && { color_label: u.color_label === "none" ? "" : u.color_label }),
            ...(u.keywords != null && { keywords: u.keywords }),
          }
          : p,
      ),
    );
  };

  const [roots, setRoots] = useState<string[]>(loadRoots);
  const shortFolder = (f: string) => f.split(/[\\/]/).filter(Boolean).slice(-2).join("/") || f;

  // A folder is a live source: opening it picks up files added since it was last read.
  const openFolder = async (folder: string) => {
    rememberLastFolder(folder);
    const viewId = `folder:${folder}`;
    const name = shortFolder(folder);
    const isSwitching = filter.folder !== folder || filter.immich_album != null || filter.ids != null;
    // photos are read only when this folder itself holds some; its subfolders are in the sidebar tree
    const images = await api.browseFolder(folder).then(([, n]) => n, () => 1);
    if (isSwitching) {
      setPhotos([]);
    }
    setFilter((f) => ({ ...f, folder, flat: true, immich_album: null, ids: null, people: null, view: viewId, in_bin: null }));
    if (!images) {
      setStatus("");
      await reloadRef.current();
      return;
    }
    setLoadingFolder({ viewId, name });
    setStatus(`Reading ${name}…`);
    try {
      const n = await api.importFolder(folder);
      setStatus(n ? `${n} new photo(s) in ${name}` : "");
      setTimeout(() => setStatus(""), 8000);
    } catch (e) {
      setStatus(`Could not read ${folder}: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
  };

  const openAll = async () => {
    const viewId = "all";
    const isSwitching = filter.folder != null || filter.immich_album != null || filter.ids != null;
    setLoadingFolder({ viewId, name: "All Photographs" });
    if (isSwitching) {
      setPhotos([]);
    }
    setFilter((f) => ({ ...f, folder: null, immich_album: null, ids: null, people: null, view: "all", in_bin: null }));
    try {
      await reloadRef.current();
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
    }
  };

  const addFolder = async () => {
    const dir = await open({ directory: true, title: "Add a folder of photos" });
    if (typeof dir === "string") {
      setRoots(rememberRoot(dir));
      openFolder(dir);
    }
  };

  const openImmich = async (album: string | null, name: string) => {
    const viewId = `immich:${album ?? ""}`;
    const isSwitching = filter.view !== viewId;
    setLoadingFolder({ viewId, name });
    if (isSwitching) {
      setPhotos([]);
    }
    setFilter((f) => ({ ...f, folder: null, immich_album: album ?? "favorites", ids: null, people: null, view: viewId, in_bin: null }));
    setStatus(`Opening ${name}…`);
    try {
      const { next } = await api.immichOpen(album, 1);
      setAlbumPages({ album, next });
      setStatus("");
    } catch (e) {
      setStatus(`Immich: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
  };

  // Albums arrive a page at a time; the grid asks for the next page as its end comes near.
  const nextPage = albumPages?.next && filter.view === `immich:${albumPages.album ?? ""}` ? albumPages.next : null;
  const loadMore = async () => {
    const cur = albumPages;
    if (!nextPage || !cur || loadingPage.current) return;
    loadingPage.current = true;
    try {
      const { next } = await api.immichOpen(cur.album, nextPage);
      setAlbumPages((s) => (s === cur ? { ...cur, next } : s)); // ignore if the album was reopened meanwhile
    } catch (e) {
      setStatus(`Immich: ${e}`);
      setAlbumPages((s) => (s === cur ? { ...cur, next: null } : s)); // stop rather than retry on every scroll
    }
    loadingPage.current = false;
  };

  const pickGoogle = async () => {
    const viewId = "google";
    setLoadingFolder({ viewId, name: "Google Photos" });
    setStatus("Pick photos in the Google Photos tab, then press Done…");
    try {
      const r = await api.googleImport();
      const notes = [r.skipped && `${r.skipped} skipped (format not supported)`, r.failed && `${r.failed} failed (${r.error})`];
      setStatus([`Added ${r.imported} photo(s) from Google Photos`, ...notes.filter(Boolean)].join(", "));
      setFilter((f) => ({ ...f, folder: r.folder, immich_album: null, ids: null, people: null, view: "google", in_bin: null }));
    } catch (e) {
      setStatus(`Google Photos: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
    setTimeout(() => setStatus(""), 8000);
  };

  const syncApple = async () => {
    const viewId = "apple";
    setLoadingFolder({ viewId, name: "Apple iCloud Photos" });
    setStatus("Syncing Apple iCloud Photos…");
    try {
      const r = await api.appleImport();
      const notes = [r.skipped && `${r.skipped} up to date`, r.failed && `${r.failed} failed`];
      setStatus([`Synced ${r.imported} photo(s) from Apple iCloud Photos`, ...notes.filter(Boolean)].join(", "));
      setFilter((f) => ({ ...f, folder: r.folder, immich_album: null, ids: null, people: null, view: "apple", in_bin: null }));
    } catch (e) {
      setStatus(`Apple iCloud Photos: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
    setTimeout(() => setStatus(""), 8000);
  };

  const syncAmazon = async () => {
    const viewId = "amazon";
    setLoadingFolder({ viewId, name: "Amazon Photos" });
    setStatus("Syncing Amazon Photos…");
    try {
      const r = await api.amazonImport();
      const notes = [r.skipped && `${r.skipped} up to date`, r.failed && `${r.failed} failed`];
      setStatus([`Synced ${r.imported} photo(s) from Amazon Photos`, ...notes.filter(Boolean)].join(", "));
      setFilter((f) => ({ ...f, folder: r.folder, immich_album: null, ids: null, people: null, view: "amazon", in_bin: null }));
    } catch (e) {
      setStatus(`Amazon Photos: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
    setTimeout(() => setStatus(""), 8000);
  };

  const syncAdobe = async () => {
    const viewId = "adobe";
    setLoadingFolder({ viewId, name: "Adobe Cloud Photos" });
    setStatus("Syncing Adobe Cloud Photos…");
    try {
      const r = await api.adobeImport();
      const notes = [r.skipped && `${r.skipped} up to date`, r.failed && `${r.failed} failed`];
      setStatus([`Synced ${r.imported} photo(s) from Adobe Cloud Photos`, ...notes.filter(Boolean)].join(", "));
      setFilter((f) => ({ ...f, folder: r.folder, immich_album: null, ids: null, people: null, view: "adobe", in_bin: null }));
    } catch (e) {
      setStatus(`Adobe Cloud Photos: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
    setTimeout(() => setStatus(""), 8000);
  };

  // A network drive opens like a folder; it is connected first when it isn't mounted.
  const openDrive = async (d: NetworkDrive) => {
    rememberLastFolder(d.path);
    const viewId = `drive:${d.id}`;
    const isSwitching = filter.view !== viewId;
    setLoadingFolder({ viewId, name: d.name });
    if (isSwitching) {
      setPhotos([]);
    }
    setFilter((f) => ({ ...f, folder: d.path, flat: true, immich_album: null, ids: null, people: null, view: viewId, in_bin: null }));
    setStatus(`Opening ${d.name}…`);
    try {
      const r = await api.networkOpen(d.id);
      setStatus(r.new ? `${r.new} new photo(s) on ${d.name}` : "");
      setTimeout(() => setStatus(""), 8000);
    } catch (e) {
      setStatus(`${d.name}: ${e}`);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
  };

  // A downloaded cloud folder shows at once; what's new in the cloud is downloaded meanwhile.
  const openCloud = async (c: CloudFolder, name: string) => {
    const viewId = `cloud:${c.id}:${c.path}`;
    const folderName = `${name} › ${c.path || "All files"}`;
    const isSwitching = filter.view !== viewId;
    setLoadingFolder({ viewId, name: folderName });
    if (isSwitching) {
      setPhotos([]);
    }
    setFilter((f) => ({ ...f, folder: c.local, immich_album: null, ids: null, people: null, view: viewId, in_bin: null }));
    try {
      await syncCloud(c.id, c.path, name);
    } finally {
      setLoadingFolder((cur) => (cur?.viewId === viewId ? null : cur));
      await reloadRef.current();
    }
  };
  const syncing = useRef(new Set<string>());
  const syncCloud = async (id: string, path: string, name: string) => {
    const key = `${id}:${path}`;
    if (syncing.current.has(key)) return; // one download of a folder at a time
    syncing.current.add(key);
    setStatus(`Checking ${name}…`);
    try {
      const r = await api.cloudSync(id, path);
      setStatus([`${r.imported} new photo(s) from ${name}`, r.failed && `${r.failed} failed (${r.error})`].filter(Boolean).join(", "));
      api.settings().then(setSettings); // the folder joins the account's list
    } catch (e) {
      setStatus(`${name}: ${e}`);
    }
    syncing.current.delete(key);
    setTimeout(() => setStatus(""), 8000);
  };
  // Saving an Immich photo uploads the edit back to Immich by default.
  const exportDest = (ids: number[]) => {
    const ps = photos.filter((p) => ids.includes(p.id));
    return ps.length > 0 && ps.every((p) => p.path.startsWith("immich://")) ? "immich" : "";
  };
  const virtualCopy = async () => {
    if (activeId == null) return;
    const n = await api.virtualCopy(activeId);
    await reload();
    focus(n);
  };
  const handleSaveAsNewPhoto = async (newPhoto: Photo) => {
    setWorkingPhotosMap((prev) => ({ ...prev, [newPhoto.id]: newPhoto }));
    await reload();
    addWorkingImages([newPhoto.id], newPhoto.id);
    focus(newPhoto.id);
    const label = newPhoto.copy_name ? `${newPhoto.filename} (${newPhoto.copy_name})` : newPhoto.filename;
    setStatus(`Saved edit as new photo: ${label}`);
    setTimeout(() => setStatus(""), 4000);
  };
  const openBin = () => {
    setSelected([]);
    setActiveId(null);
    setFilter({ folder: null, immich_album: null, ids: null, people: null, view: "bin", in_bin: true });
  };

  // a second, explicit "are you sure" before acting on this many photos at once
  const BULK_DELETE = 20;
  const confirmBulk = async (n: number, action: string) =>
    n < BULK_DELETE ||
    ask(`You are about to ${action} ${n} photos at once.\n\nThis is a large number of photos. Are you absolutely sure?`, { title: `Confirm ${n} photos`, kind: "warning" });

  const moveToBin = async (idsOverride?: number[]) => {
    // buttons pass their click event here, so only trust a real id list
    const ids = Array.isArray(idsOverride) ? idsOverride : targets();
    if (!ids.length) return;
    if (!(await ask(`Move ${ids.length} photo(s) to the Bin?\n\nYou can restore them from the Bin later.`, { title: "Move to Bin", kind: "warning" }))) return;
    if (!(await confirmBulk(ids.length, "move to the Bin"))) return;
    try {
      const n = await api.moveToBin(ids);
      setStatus(`Moved ${n} photo(s) to Bin`);
      setSelected((prev) => prev.filter((id) => !ids.includes(id)));
      setWorkingIds((prev) => prev.filter((id) => !ids.includes(id)));
      if (developId && ids.includes(developId)) setDevelopId(null);
      if (activeId && ids.includes(activeId)) setActiveId(null);
      await reload();
      setTimeout(() => setStatus(""), 4000);
    } catch (e) {
      setStatus(`Move to Bin failed: ${e}`);
    }
  };

  const restoreFromBin = async () => {
    const ids = targets();
    if (!ids.length) return;
    try {
      const n = await api.restoreFromBin(ids);
      setStatus(`Restored ${n} photo(s) from Bin`);
      setSelected([]);
      setActiveId(null);
      await reload();
      setTimeout(() => setStatus(""), 4000);
    } catch (e) {
      setStatus(`Restore failed: ${e}`);
    }
  };

  const permanentDelete = () => {
    const ids = targets();
    if (!ids.length) return;
    setConfirmDelete({ ids });
  };

  const executePermanentDelete = async () => {
    if (!confirmDelete) return;
    const ids = confirmDelete.ids;
    setConfirmDelete(null);
    if (!(await confirmBulk(ids.length, "permanently delete from disk"))) return;
    try {
      const n = await api.permanentDelete(ids);
      setStatus(`Permanently deleted ${n} photo(s)`);
      setSelected([]);
      setWorkingIds((prev) => prev.filter((id) => !ids.includes(id)));
      if (developId && ids.includes(developId)) setDevelopId(null);
      setActiveId(null);
      await reload();
      setTimeout(() => setStatus(""), 4000);
    } catch (e) {
      setStatus(`Permanent delete failed: ${e}`);
    }
  };

  const emptyBin = async () => {
    const count = binCount || photos.length;
    if (count === 0) return;
    const ok = await ask(
      `Empty Bin (${count} photo(s))?\n\nAll photos in the Bin will be permanently deleted from your disk. This action cannot be undone.`,
      { title: "Empty Bin", kind: "error" }
    );
    if (!ok || !(await confirmBulk(count, "permanently delete from disk"))) return;
    try {
      const n = await api.emptyBin();
      setStatus(`Emptied Bin (${n} photo(s) permanently deleted)`);
      setSelected([]);
      setActiveId(null);
      await reload();
      setTimeout(() => setStatus(""), 4000);
    } catch (e) {
      setStatus(`Empty Bin failed: ${e}`);
    }
  };

  const remove = async () => {
    const ids = targets();
    if (!ids.length || !(await ask(`Remove ${ids.length} photo(s) from the catalog? Files on disk are not touched.`, { title: "Remove from catalog", kind: "warning" })) || !(await confirmBulk(ids.length, "remove from the catalog"))) return;
    await api.remove(ids);
    setSelected([]);
    setWorkingIds((prev) => prev.filter((id) => !ids.includes(id)));
    if (developId && ids.includes(developId)) setDevelopId(null);
    setActiveId(null);
  };

  const canDevelop = Boolean(active || workingPhotos.length > 0 || developActive);

  const keys = useRef<(e: KeyboardEvent) => void>(() => { });
  keys.current = (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || module === "settings") return;
    const k = e.key.toLowerCase();
    if (k === "c" && selected.length >= 2) {
      e.preventDefault();
      setCompareOpen((prev) => !prev);
    } else if (k === "g") setModule("library");
    else if (k === "d" && canDevelop) openInDevelop();
    else if (/^[0-5]$/.test(k)) cull({ rating: +k }, module === "develop" && developActive ? [developActive.id] : undefined);
    else if (k === "p") cull({ flag: "pick" }, module === "develop" && developActive ? [developActive.id] : undefined);
    else if (k === "x") cull({ flag: "reject" }, module === "develop" && developActive ? [developActive.id] : undefined);
    else if (k === "u") cull({ flag: "unflagged" }, module === "develop" && developActive ? [developActive.id] : undefined);
    else if (/^[6-9]$/.test(k)) {
      const label = LABELS[+k - 6];
      const curTarget = module === "develop" ? developActive : active;
      cull({ color_label: curTarget?.color_label === label ? "none" : label }, curTarget ? [curTarget.id] : undefined);
    } else if (k === "arrowright" || k === "arrowleft") {
      e.preventDefault();
      if (module === "develop" && workingPhotos.length > 1) {
        const curIdx = workingPhotos.findIndex((p) => p.id === developActive?.id);
        const nextPhoto = workingPhotos[curIdx + (k === "arrowright" ? 1 : -1)];
        if (nextPhoto) focusDevelop(nextPhoto.id);
      } else {
        const curId = module === "develop" ? developActive?.id ?? activeId : activeId;
        const i = photos.findIndex((p) => p.id === curId) + (k === "arrowright" ? 1 : -1);
        if (photos[i]) {
          if (module === "develop") focusDevelop(photos[i].id);
          else focus(photos[i].id);
        }
      }
    } else if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      if (filter.view === "bin" || e.shiftKey) {
        permanentDelete();
      } else {
        moveToBin();
      }
    }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (!["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) keys.current(e);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  if (initialLoading) {
    return (
      <div className="flex h-screen w-screen flex-col items-center justify-center bg-neutral-950 text-neutral-100 select-none">
        <div className="relative mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-neutral-900 border border-neutral-800 shadow-2xl">
          <div className="absolute inset-0 rounded-3xl bg-cyan-500/10 blur-xl animate-pulse" />
          <img
            src="/icon.png"
            alt="Optixel Studio"
            className="h-10 w-10 rounded-xl object-cover shadow-lg relative z-10"
          />
          <svg className="absolute -inset-1.5 h-23 w-23 animate-spin text-cyan-400" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-10" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
            <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v3a5 5 0 00-5 5H4z" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold tracking-tight text-white mb-1.5">Optixel Studio</h1>
        <div className="flex items-center gap-2 text-xs text-neutral-400 font-mono mb-4">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400 animate-ping" />
          <span>Starting 32-bit floating-point engine &amp; catalog…</span>
        </div>
        <div className="h-1 w-48 overflow-hidden rounded-full bg-neutral-800">
          <div className="h-full w-1/3 rounded-full bg-gradient-to-r from-cyan-500 to-sky-400 animate-[pulse_1.5s_ease-in-out_infinite]" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="relative flex h-10 shrink-0 items-center gap-5 border-b border-neutral-800 bg-neutral-950 px-4">
        <span className="font-semibold tracking-wide text-neutral-100 flex items-center gap-2">
          <img
            src="/icon.png"
            alt="Optixel Studio"
            className="h-5 w-5 rounded object-cover shadow-xs"
          />
          <span>
            Optixel <span className="font-normal text-neutral-400">Studio</span>
          </span>
        </span>
        <nav className="flex gap-1">
          {MODULES.filter(([m]) => m !== "people" || settings?.faces_enabled).map(([m, label]) => {
            const isCur = module === m;
            return (
              <button
                key={m}
                onClick={() => {
                  if (m === "develop") {
                    openInDevelop();
                  } else {
                    setModule(m);
                  }
                }}
                disabled={m === "develop" && !canDevelop}
                style={
                  isCur
                    ? {
                      backgroundColor: "var(--theme-accent, #0ea5e9)",
                      color: "var(--theme-accent-contrast, #ffffff)",
                      boxShadow: "0 0 10px var(--theme-accent-glow, rgba(14,165,233,0.3))",
                    }
                    : undefined
                }
                className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-[12px] transition-all duration-150 disabled:opacity-40 ${isCur ? "font-semibold shadow-xs" : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900"
                  }`}
              >
                <span>{label}</span>
                {m === "settings" && updateInfo?.has_update && (
                  <span
                    className="h-2 w-2 rounded-full bg-[var(--theme-accent,#0ea5e9)] animate-pulse"
                    title={`Update ${updateInfo.release_tag} available`}
                  />
                )}
                {m === "develop" && workingPhotos.length > 0 && (
                  <span
                    className={`rounded-full px-1.5 py-0.2 text-[10px] font-semibold leading-tight ${
                      isCur ? "bg-black/25 text-white" : "bg-neutral-800 text-neutral-300"
                    }`}
                    title={`${workingPhotos.length} working image(s) open in Develop`}
                  >
                    {workingPhotos.length}
                  </span>
                )}
                {m === "develop" && busyAiIds.length > 0 && (
                  <span
                    className={`flex items-center gap-1 rounded-full px-1.5 py-0.2 text-[10px] font-semibold animate-pulse ${
                      isCur ? "bg-white/25 text-white" : "bg-violet-600/30 text-violet-300 border border-violet-500/40"
                    }`}
                    title={`AI fine tune / edit running on ${busyAiIds.length} image(s)`}
                  >
                    ✦ {busyAiIds.length > 1 ? busyAiIds.length : ""}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
        {busyAiIds.length > 0 && module !== "develop" && (
          <button
            type="button"
            onClick={() => {
              setDevelopId(busyAiIds[0]);
              setModule("develop");
            }}
            className="flex items-center gap-1.5 rounded-full border border-violet-500/40 bg-violet-950/60 px-2.5 py-0.5 text-[11px] font-medium text-violet-200 hover:bg-violet-900/70 transition-colors cursor-pointer animate-pulse"
            title="Click to view AI progress in Develop"
          >
            <span>✦ AI working on {busyAiIds.length} {busyAiIds.length === 1 ? "photo" : "photos"} in Develop…</span>
          </button>
        )}
        <span className="ml-auto truncate text-[11px] text-neutral-400 flex items-center gap-2" title={status}>
          {status || background}
          {regenProgress && regenProgress.total > 0 && (
            <span className="flex items-center gap-1.5 font-mono text-[10px] text-sky-400">
              <span className="inline-block w-16 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                <span
                  className="block h-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-150"
                  style={{ width: `${Math.min(100, Math.max(3, (regenProgress.done / regenProgress.total) * 100))}%` }}
                />
              </span>
              <span>{Math.round((regenProgress.done / regenProgress.total) * 100)}%</span>
            </span>
          )}
        </span>
        {!status && culling && (
          <button className="text-[11px] text-neutral-500 hover:text-white" title="Stop culling the new photos; what is done stays" onClick={() => api.cancelJob(culling.job)}>
            ✕
          </button>
        )}
        {updateInfo?.has_update && (
          <button
            type="button"
            onClick={() => setShowUpdateModal(true)}
            className="flex items-center gap-1.5 rounded-full border border-[var(--theme-accent,#0ea5e9)]/40 bg-[var(--theme-accent,#0ea5e9)]/20 px-2.5 py-0.5 text-[11px] font-semibold text-[var(--theme-accent,#0ea5e9)] hover:bg-[var(--theme-accent,#0ea5e9)]/30 transition-all cursor-pointer shadow-xs"
            title={`Optixel Studio update ${updateInfo.release_tag} is available! Click to review and download.`}
          >
            <span className="animate-pulse">✨</span>
            <span>Update {updateInfo.release_tag}</span>
          </button>
        )}
        <button
          className="flex items-center gap-1.5 text-[11px] text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer"
          title={`${info?.mcp_url ?? "MCP server"} — Click to open Agent & MCP Settings`}
          onClick={() => {
            setSettingsTab("agent");
            setModule("settings");
          }}
        >
          <span
            className={`h-2 w-2 rounded-full transition-all ${info?.mcp_status === "running" ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]" : "bg-neutral-600"
              }`}
          />
          <span className="font-mono text-[10px]">MCP</span>
        </button>
        {regenProgress && regenProgress.total > 0 && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-neutral-800/80 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-150"
              style={{ width: `${Math.min(100, Math.max(2, (regenProgress.done / regenProgress.total) * 100))}%` }}
            />
          </div>
        )}
      </header>

      {module === "library" && (
        <Library
          photos={photos}
          folders={folders}
          filter={filter}
          setFilter={setFilter}
          selected={selected}
          active={active}
          select={select}
          openDevelop={(id) => openInDevelop(id)}
          workingIds={workingIds}
          busyAiIds={busyAiIds}
          onAddWorking={(ids) => {
            addWorkingImages(ids, developId ?? ids[0]);
            setStatus(`Added ${ids.length} photo(s) to Develop working images`);
            setTimeout(() => setStatus(""), 3500);
          }}
          cull={cull}
          settings={settings}
          setSettings={setSettings}
          dataDir={info?.data_dir ?? ""}
          loadingFolder={loadingFolder}
          onShowAll={openAll}
          onAddFolder={addFolder}
          onOpenFolder={openFolder}
          roots={roots}
          onRemoveRoot={(r) => setRoots(forgetRoot(r))}
          onImmich={openImmich}
          nextPage={nextPage}
          onLoadMore={loadMore}
          onPickGoogle={pickGoogle}
          onSyncApple={syncApple}
          onSyncAmazon={syncAmazon}
          onSyncAdobe={syncAdobe}
          onOpenDrive={openDrive}
          onOpenCloud={openCloud}
          onAddCloudFolder={syncCloud}
          onStorage={() => (setSettingsTab("locations"), setModule("settings"))}
          onConnect={setRemote}
          onExport={() => setExportIds(targets())}
          onVirtualCopy={virtualCopy}
          onRemove={remove}
          binCount={binCount}
          onOpenBin={openBin}
          onMoveToBin={moveToBin}
          onRestoreFromBin={restoreFromBin}
          onPermanentDelete={permanentDelete}
          onEmptyBin={emptyBin}
          onOpenCompare={() => setCompareOpen(true)}
          autoFn={libraryAutoFn}
          onSelectAutoFn={setLibraryAutoFn}
          jobs={jobs}
          onAiSettings={() => {
            setSettingsTab("agent");
            setModule("settings");
          }}
        />
      )}
      {module === "people" && (
        <People
          jobs={jobs}
          onOpenSettings={() => (setSettingsTab("ai"), setModule("settings"))}
          onShowPhotos={(people, label) => {
            setFilter({ people, view: `people:${label}`, folder: null, immich_album: null, ids: null, in_bin: null });
            setModule("library");
          }}
        />
      )}
      {defaults && (
        <div className={module === "develop" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
          <Develop
            onOpenModels={() => (setSettingsTab("ai"), setModule("settings"))}
            onOpenPhoto={(id) => setTimeout(() => openInDevelop(id), 400)}
            visible={module === "develop"}
            photos={photos}
            workingPhotos={workingPhotos}
            active={developActive}
            selected={selected}
            setActive={focusDevelop}
            onCloseWorking={closeWorkingImage}
            onCloseOtherWorking={closeOtherWorkingImages}
            onAddWorking={(ids) => addWorkingImages(ids, ids[0])}
            aiTasks={aiTasks}
            setAiTasks={setAiTasks}
            onPhotoUpdated={refreshPhoto}
            defaults={defaults}
            settings={settings}
            setSettings={setSettings}
            onExport={() => setExportIds(developActive ? [developActive.id] : targets())}
            onSaveAsNewPhoto={handleSaveAsNewPhoto}
            logs={logs}
            clearLogs={() => setLogs([])}
          />
        </div>
      )}
      {module === "settings" && (
        <SettingsView
          tab={settingsTab}
          setTab={setSettingsTab}
          settings={settings}
          setSettings={setSettings}
          selected={selected}
          jobs={jobs}
          onConnect={setRemote}
          onAuto={() => {
            setModule("library");
            setLibraryAutoFn("cull");
          }}
          info={info}
          logs={logs}
          clearLogs={() => setLogs([])}
          onCheckUpdate={() => checkForUpdates(true)}
          updateAvailable={Boolean(updateInfo?.has_update)}
          updateInfo={updateInfo}
          updateLoading={updateLoading}
        />
      )}
      {exportIds && exportIds.length > 0 && <ExportDialog ids={exportIds} settings={settings} dest={exportDest(exportIds)} onClose={() => setExportIds(null)} />}
      {remote && settings && <RemoteDialog source={remote} settings={settings} setSettings={setSettings} onClose={() => setRemote(null)} />}
      {compareOpen && selected.length >= 2 && (
        <CompareModal
          groups={[
            {
              id: "selection",
              title: `Selected Photos (${selected.length})`,
              kind: "manual",
              photos: selected.map((id) => photos.find((p) => p.id === id)).filter(Boolean) as Photo[],
              keepId: selected[0],
            },
          ]}
          initialIndex={0}
          onClose={() => setCompareOpen(false)}
          onCull={(id, culling) => cull(culling, [id])}
          onMoveToBin={(id) => moveToBin([id])}
        />
      )}
      {confirmDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4"
          onClick={() => setConfirmDelete(null)}
        >
          <div
            className="w-full max-w-md rounded-xl border border-red-500/40 bg-neutral-900 p-6 text-neutral-100 shadow-2xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start gap-3.5">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-red-500/20 text-red-400 border border-red-500/30 text-xl font-bold">
                ⚠️
              </div>
              <div className="space-y-1.5 flex-1">
                <h3 className="text-base font-semibold text-white">
                  Permanently Delete {confirmDelete.ids.length} Photo{confirmDelete.ids.length === 1 ? "" : "s"}?
                </h3>
                <p className="text-xs text-neutral-300 leading-relaxed">
                  This will permanently delete the original file{confirmDelete.ids.length === 1 ? "" : "s"} from your disk. All associated edit history, sidecar metadata, and cached thumbnails will also be removed.
                </p>
                <div className="rounded-md bg-red-950/60 border border-red-800/60 px-3 py-2 text-xs font-semibold text-red-300">
                  This action cannot be undone.
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-neutral-800">
              <button
                type="button"
                autoFocus
                onClick={() => setConfirmDelete(null)}
                className="rounded-lg border border-neutral-700 bg-neutral-800 px-4 py-2 text-xs font-medium text-neutral-200 hover:bg-neutral-700 hover:text-white transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={executePermanentDelete}
                className="rounded-lg bg-red-600 hover:bg-red-500 px-4 py-2 text-xs font-semibold text-white shadow-md transition-colors cursor-pointer flex items-center gap-1.5 border border-red-500"
              >
                <span>✕ Delete Permanently</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {showUpdateModal && (
        <UpdateModal
          info={updateInfo}
          loading={updateLoading}
          error={updateError}
          onCheckAgain={() => checkForUpdates(true)}
          onClose={() => setShowUpdateModal(false)}
        />
      )}
    </div>
  );
}
