import { useMemo, useState } from "react";
import { Photo, thumbUrl } from "./api";
import { Cell, SelectMods } from "./Library";

interface CameraViewProps {
  photos: Photo[];
  selected: number[];
  active: Photo | null;
  workingIds?: number[];
  busyAiIds?: number[];
  size: number;
  onSelect: (id: number, mods: SelectMods) => void;
  onOpenDevelop: (id: number) => void;
  onOpenViewer: (id: number) => void;
  onSelectMultiple?: (ids: number[]) => void;
}

interface CameraGroup {
  name: string;
  count: number;
  photos: Photo[];
  lenses: { name: string; count: number }[];
  dateRange: string;
  isoRange: string;
  sampleThumbs: Photo[];
}

export function CameraView({
  photos,
  selected,
  active,
  workingIds,
  busyAiIds,
  size,
  onSelect,
  onOpenDevelop,
  onOpenViewer,
  onSelectMultiple,
}: CameraViewProps) {
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [selectedLens, setSelectedLens] = useState<string | null>(null);
  const [selectedIso, setSelectedIso] = useState<string | null>(null);

  // Group photos by camera model
  const { cameraGroups, totalCameras, totalLenses } = useMemo(() => {
    const map = new Map<string, Photo[]>();
    const allLensesSet = new Set<string>();

    for (const p of photos) {
      const cam = p.camera && p.camera.trim() ? p.camera.trim() : "Unknown Camera";
      if (!map.has(cam)) map.set(cam, []);
      map.get(cam)!.push(p);

      if (p.lens && p.lens.trim()) {
        allLensesSet.add(p.lens.trim());
      }
    }

    const groups: CameraGroup[] = [];
    for (const [name, camPhotos] of map.entries()) {
      // Sort photos by capture date descending
      camPhotos.sort((a, b) => (b.captured_at || "").localeCompare(a.captured_at || ""));

      // Lenses breakdown
      const lensCountMap = new Map<string, number>();
      let minIso = Infinity;
      let maxIso = -Infinity;
      let earliestDate = "";
      let latestDate = "";

      for (const p of camPhotos) {
        const l = p.lens && p.lens.trim() ? p.lens.trim() : "Other / Prime";
        lensCountMap.set(l, (lensCountMap.get(l) || 0) + 1);

        if (typeof p.iso === "number" && p.iso > 0) {
          if (p.iso < minIso) minIso = p.iso;
          if (p.iso > maxIso) maxIso = p.iso;
        }

        if (p.captured_at && p.captured_at.length >= 4) {
          const y = p.captured_at.slice(0, 4);
          if (!earliestDate || y < earliestDate) earliestDate = y;
          if (!latestDate || y > latestDate) latestDate = y;
        }
      }

      const sortedLenses = Array.from(lensCountMap.entries())
        .map(([lensName, count]) => ({ name: lensName, count }))
        .sort((a, b) => b.count - a.count);

      const dateRange =
        earliestDate && latestDate
          ? earliestDate === latestDate
            ? earliestDate
            : `${earliestDate} – ${latestDate}`
          : "Various dates";

      const isoRange =
        minIso !== Infinity && maxIso !== -Infinity
          ? minIso === maxIso
            ? `ISO ${minIso}`
            : `ISO ${minIso} – ${maxIso}`
          : "";

      groups.push({
        name,
        count: camPhotos.length,
        photos: camPhotos,
        lenses: sortedLenses,
        dateRange,
        isoRange,
        sampleThumbs: camPhotos.slice(0, 6),
      });
    }

    // Sort camera groups by photo count descending
    groups.sort((a, b) => b.count - a.count);

    return {
      cameraGroups: groups,
      totalCameras: groups.length,
      totalLenses: allLensesSet.size,
    };
  }, [photos]);

  // Filtered photos based on camera, lens, and ISO selection
  const filteredPhotos = useMemo(() => {
    let list = photos;
    if (selectedCamera) {
      list = list.filter((p) => {
        const cam = p.camera && p.camera.trim() ? p.camera.trim() : "Unknown Camera";
        return cam === selectedCamera;
      });
    }
    if (selectedLens) {
      list = list.filter((p) => {
        const l = p.lens && p.lens.trim() ? p.lens.trim() : "Other / Prime";
        return l === selectedLens;
      });
    }
    if (selectedIso) {
      list = list.filter((p) => {
        if (typeof p.iso !== "number") return false;
        if (selectedIso === "<400") return p.iso < 400;
        if (selectedIso === "400-1600") return p.iso >= 400 && p.iso <= 1600;
        if (selectedIso === "1600+") return p.iso > 1600;
        return true;
      });
    }
    return list;
  }, [photos, selectedCamera, selectedLens, selectedIso]);

  const activeGroup = useMemo(() => {
    if (!selectedCamera) return null;
    return cameraGroups.find((g) => g.name === selectedCamera) || null;
  }, [selectedCamera, cameraGroups]);

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-neutral-950">
      {/* Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 bg-neutral-900/90 px-4 py-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-neutral-200">📷 By Camera</span>
          <span className="text-neutral-500">·</span>
          <span className="text-neutral-400">
            {totalCameras} camera{totalCameras === 1 ? "" : "s"} · {totalLenses} lens{totalLenses === 1 ? "" : "es"} across {photos.length} photo{photos.length === 1 ? "" : "s"}
          </span>
        </div>

        {selectedCamera && (
          <button
            type="button"
            onClick={() => {
              setSelectedCamera(null);
              setSelectedLens(null);
              setSelectedIso(null);
            }}
            className="flex items-center gap-1.5 rounded bg-neutral-800 hover:bg-neutral-700 px-2.5 py-1 text-[11px] font-medium text-neutral-200 transition-colors cursor-pointer"
          >
            <span>✕ Show All Cameras</span>
          </button>
        )}
      </div>

      {/* Camera Model Cards (shown when no specific camera is focused) */}
      {!selectedCamera && (
        <div className="border-b border-neutral-800 bg-neutral-900/50 p-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {cameraGroups.map((cg) => {
              const pct = photos.length > 0 ? Math.round((cg.count / photos.length) * 100) : 0;
              return (
                <div
                  key={cg.name}
                  onClick={() => {
                    setSelectedCamera(cg.name);
                    setSelectedLens(null);
                    setSelectedIso(null);
                    if (onSelectMultiple) {
                      onSelectMultiple(cg.photos.map((p) => p.id));
                    }
                  }}
                  className="group relative flex flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900/80 p-3.5 transition-all hover:border-neutral-700 hover:bg-neutral-800/80 hover:shadow-lg cursor-pointer"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="flex h-8 w-8 items-center justify-center rounded-md bg-neutral-800 text-base text-neutral-300 group-hover:bg-[var(--theme-accent,#0ea5e9)] group-hover:text-white transition-colors">
                          📷
                        </span>
                        <div>
                          <h3 className="font-semibold text-neutral-100 group-hover:text-[var(--theme-accent,#0ea5e9)] transition-colors line-clamp-1 text-xs">
                            {cg.name}
                          </h3>
                          <p className="text-[10px] text-neutral-400">{cg.dateRange}</p>
                        </div>
                      </div>
                      <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] font-mono font-semibold text-neutral-300">
                        {cg.count} ({pct}%)
                      </span>
                    </div>

                    {/* Lenses info */}
                    <div className="mt-3 space-y-1">
                      {cg.lenses.slice(0, 2).map((l) => (
                        <div key={l.name} className="flex items-center justify-between text-[10px] text-neutral-400">
                          <span className="truncate max-w-[170px]" title={l.name}>
                            🔍 {l.name}
                          </span>
                          <span className="font-mono text-neutral-500">{l.count}</span>
                        </div>
                      ))}
                      {cg.lenses.length > 2 && (
                        <p className="text-[9px] text-neutral-500 italic">
                          +{cg.lenses.length - 2} more lens{cg.lenses.length - 2 === 1 ? "" : "es"}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Thumbnail filmstrip */}
                  <div className="mt-3 flex gap-1 overflow-hidden rounded">
                    {cg.sampleThumbs.map((p) => (
                      <div key={p.id} className="h-10 w-10 shrink-0 overflow-hidden rounded bg-neutral-800">
                        <img
                          src={thumbUrl(p)}
                          alt=""
                          loading="lazy"
                          className="h-full w-full object-cover transition-transform group-hover:scale-105"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Focused Camera View with Sub-filters */}
      {selectedCamera && activeGroup && (
        <div className="border-b border-neutral-800 bg-neutral-900/60 px-4 py-2.5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="text-xl">📷</span>
              <div>
                <h2 className="text-sm font-bold text-white flex items-center gap-2">
                  <span>{activeGroup.name}</span>
                  <span className="rounded bg-[var(--theme-accent,#0ea5e9)]/20 text-[var(--theme-accent,#0ea5e9)] border border-[var(--theme-accent,#0ea5e9)]/40 px-2 py-0.2 text-[10px] font-semibold">
                    {filteredPhotos.length} photo{filteredPhotos.length === 1 ? "" : "s"}
                  </span>
                </h2>
                <p className="text-[11px] text-neutral-400">
                  {activeGroup.dateRange} {activeGroup.isoRange ? `· ${activeGroup.isoRange}` : ""}
                </p>
              </div>
            </div>

            {/* Lens and ISO Sub-filters */}
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              {/* Lens Filter */}
              <div className="flex items-center gap-1 text-neutral-400">
                <span>Lens:</span>
                <select
                  value={selectedLens || ""}
                  onChange={(e) => setSelectedLens(e.target.value || null)}
                  className="rounded bg-neutral-800 px-2 py-1 text-neutral-200 border border-neutral-700/60 outline-none"
                >
                  <option value="">All Lenses ({activeGroup.photos.length})</option>
                  {activeGroup.lenses.map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.name} ({l.count})
                    </option>
                  ))}
                </select>
              </div>

              {/* ISO Filter */}
              <div className="flex items-center gap-1 text-neutral-400">
                <span>ISO:</span>
                <select
                  value={selectedIso || ""}
                  onChange={(e) => setSelectedIso(e.target.value || null)}
                  className="rounded bg-neutral-800 px-2 py-1 text-neutral-200 border border-neutral-700/60 outline-none"
                >
                  <option value="">All ISO</option>
                  <option value="<400">&lt; 400 (Low noise)</option>
                  <option value="400-1600">400 – 1600 (Standard)</option>
                  <option value="1600+">1600+ (High sensitivity)</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Photos Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {filteredPhotos.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-2 text-neutral-500">
            <span className="text-3xl">📷</span>
            <p className="text-sm font-medium text-neutral-300">No photos match the selected camera filters</p>
            <button
              type="button"
              onClick={() => {
                setSelectedCamera(null);
                setSelectedLens(null);
                setSelectedIso(null);
              }}
              className="mt-2 rounded bg-neutral-800 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700"
            >
              Reset Camera Filters
            </button>
          </div>
        ) : (
          <div
            className="grid gap-2"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}
          >
            {filteredPhotos.map((ph) => (
              <Cell
                key={ph.id}
                photo={ph}
                size={size}
                selected={selected.includes(ph.id)}
                active={active?.id === ph.id}
                inWorking={workingIds?.includes(ph.id)}
                aiBusy={busyAiIds?.includes(ph.id)}
                onClick={(m) => onSelect(ph.id, m)}
                onOpen={() => onOpenDevelop(ph.id)}
                onView={() => {
                  onSelect(ph.id, { shift: false, toggle: false });
                  onOpenViewer(ph.id);
                }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
