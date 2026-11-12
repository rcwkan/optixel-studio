import { useMemo, useState, useRef, useEffect, useCallback, MouseEvent, WheelEvent } from "react";
import { api, Photo, thumbUrl } from "./api";
import { Cell, SelectMods } from "./Library";

interface LocationViewProps {
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

interface GeoPoint {
  photo: Photo;
  lat: number;
  lon: number;
}

interface PlaceCluster {
  id: string;
  name: string;
  centerLat: number;
  centerLon: number;
  photos: Photo[];
}

export function LocationView({
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
}: LocationViewProps) {
  const [filterMode, setFilterMode] = useState<"all" | "geotagged" | "untagged">("geotagged");
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanMessage, setScanMessage] = useState<string | null>(null);

  // Map state
  const [zoom, setZoom] = useState<number>(3);
  const [center, setCenter] = useState<{ lat: number; lon: number }>({ lat: 25, lon: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const mapContainerRef = useRef<HTMLDivElement>(null);

  // Separate geotagged vs untagged photos
  const { geotagged, untagged, clusters } = useMemo(() => {
    const geo: GeoPoint[] = [];
    const un: Photo[] = [];

    for (const p of photos) {
      if (typeof p.latitude === "number" && typeof p.longitude === "number" && !isNaN(p.latitude) && !isNaN(p.longitude)) {
        geo.push({ photo: p, lat: p.latitude, lon: p.longitude });
      } else {
        un.push(p);
      }
    }

    // Cluster points within ~0.25 degree radius (~25km)
    const clusterList: PlaceCluster[] = [];
    const used = new Set<number>();

    for (let i = 0; i < geo.length; i++) {
      if (used.has(geo[i].photo.id)) continue;
      const pt = geo[i];
      const matchedPhotos: Photo[] = [pt.photo];
      used.add(pt.photo.id);

      let sumLat = pt.lat;
      let sumLon = pt.lon;

      for (let j = i + 1; j < geo.length; j++) {
        if (used.has(geo[j].photo.id)) continue;
        const other = geo[j];
        const dLat = Math.abs(pt.lat - other.lat);
        const dLon = Math.abs(pt.lon - other.lon);
        if (dLat < 0.25 && dLon < 0.25) {
          matchedPhotos.push(other.photo);
          used.add(other.photo.id);
          sumLat += other.lat;
          sumLon += other.lon;
        }
      }

      const count = matchedPhotos.length;
      const centerLat = sumLat / count;
      const centerLon = sumLon / count;
      const latStr = centerLat >= 0 ? `${centerLat.toFixed(2)}°N` : `${Math.abs(centerLat).toFixed(2)}°S`;
      const lonStr = centerLon >= 0 ? `${centerLon.toFixed(2)}°E` : `${Math.abs(centerLon).toFixed(2)}°W`;
      
      // Determine a friendly name: check folder name or coordinate
      const folderPart = pt.photo.folder ? pt.photo.folder.split("/").pop() || "" : "";
      const name = folderPart && folderPart !== "DCIM" && folderPart.length > 2
        ? `${folderPart} (${latStr}, ${lonStr})`
        : `${latStr}, ${lonStr}`;

      clusterList.push({
        id: `cluster-${i}`,
        name,
        centerLat,
        centerLon,
        photos: matchedPhotos,
      });
    }

    // Sort clusters by photo count descending
    clusterList.sort((a, b) => b.photos.length - a.photos.length);

    return {
      geotagged: geo,
      untagged: un,
      clusters: clusterList,
    };
  }, [photos]);

  // Center map on photos if available
  useEffect(() => {
    if (geotagged.length > 0) {
      let sumLat = 0;
      let sumLon = 0;
      for (const g of geotagged) {
        sumLat += g.lat;
        sumLon += g.lon;
      }
      setCenter({
        lat: sumLat / geotagged.length,
        lon: sumLon / geotagged.length,
      });
      setZoom(geotagged.length > 1 ? 5 : 12);
    }
  }, [geotagged.length]);

  const handleScanGps = async () => {
    setScanning(true);
    setScanMessage("Scanning photo files for GPS metadata…");
    try {
      const count = await api.scanLocations();
      setScanMessage(count > 0 ? `Found GPS in ${count} photo(s)! Reloading…` : "No new GPS metadata found.");
      setTimeout(() => setScanMessage(null), 3500);
    } catch (e) {
      setScanMessage(`Scan error: ${e}`);
    } finally {
      setScanning(false);
    }
  };

  // Convert lat/lon to map pixel coordinates
  const project = useCallback(
    (lat: number, lon: number, w: number, h: number) => {
      const scale = 256 * Math.pow(2, zoom);
      const x = ((lon + 180) / 360) * scale;
      const sinLat = Math.sin((lat * Math.PI) / 180);
      const clampedSin = Math.min(Math.max(sinLat, -0.9999), 0.9999);
      const y = (0.5 - Math.log((1 + clampedSin) / (1 - clampedSin)) / (4 * Math.PI)) * scale;

      const centerX = ((center.lon + 180) / 360) * scale;
      const centerSin = Math.min(Math.max(Math.sin((center.lat * Math.PI) / 180), -0.9999), 0.9999);
      const centerY = (0.5 - Math.log((1 + centerSin) / (1 - centerSin)) / (4 * Math.PI)) * scale;

      return {
        x: w / 2 + (x - centerX),
        y: h / 2 + (y - centerY),
      };
    },
    [zoom, center]
  );

  // Unproject map pixel back to lat/lon for pan drag
  const unprojectOffset = useCallback(
    (dx: number, dy: number) => {
      const scale = 256 * Math.pow(2, zoom);
      const dLon = (-dx / scale) * 360;
      const dLat = (dy / scale) * 180;
      setCenter((c) => ({
        lat: Math.max(-85, Math.min(85, c.lat + dLat)),
        lon: ((c.lon + dLon + 540) % 360) - 180,
      }));
    },
    [zoom]
  );

  // Drag handlers
  const handleMouseDown = (e: MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setDragStart({ x: e.clientX, y: e.clientY });
    unprojectOffset(dx, dy);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(18, z + 1));
    } else if (e.deltaY > 0) {
      setZoom((z) => Math.max(2, z - 1));
    }
  };

  const activePhotos = useMemo(() => {
    if (filterMode === "untagged") return untagged;
    if (selectedClusterId) {
      const c = clusters.find((x) => x.id === selectedClusterId);
      if (c) return c.photos;
    }
    return filterMode === "geotagged" ? geotagged.map((g) => g.photo) : photos;
  }, [filterMode, selectedClusterId, clusters, geotagged, untagged, photos]);

  const mapW = mapContainerRef.current?.clientWidth || 800;
  const mapH = mapContainerRef.current?.clientHeight || 400;

  // Compute tile URLs for current center and zoom
  const tiles = useMemo(() => {
    const tileCount = Math.pow(2, zoom);
    const scale = 256 * tileCount;
    const centerX = ((center.lon + 180) / 360) * scale;
    const centerSin = Math.min(Math.max(Math.sin((center.lat * Math.PI) / 180), -0.9999), 0.9999);
    const centerY = (0.5 - Math.log((1 + centerSin) / (1 - centerSin)) / (4 * Math.PI)) * scale;

    const startX = Math.floor((centerX - mapW / 2) / 256);
    const endX = Math.floor((centerX + mapW / 2) / 256);
    const startY = Math.floor((centerY - mapH / 2) / 256);
    const endY = Math.floor((centerY + mapH / 2) / 256);

    const list: { url: string; x: number; y: number; key: string }[] = [];
    for (let ty = Math.max(0, startY); ty <= Math.min(tileCount - 1, endY); ty++) {
      for (let tx = startX; tx <= endX; tx++) {
        const wrappedTx = ((tx % tileCount) + tileCount) % tileCount;
        const posX = mapW / 2 + (tx * 256 - centerX);
        const posY = mapH / 2 + (ty * 256 - centerY);
        list.push({
          url: `https://tile.openstreetmap.org/${zoom}/${wrappedTx}/${ty}.png`,
          x: posX,
          y: posY,
          key: `${zoom}-${wrappedTx}-${ty}`,
        });
      }
    }
    return list;
  }, [zoom, center, mapW, mapH]);

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-neutral-950">
      {/* Top Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 bg-neutral-900/90 px-4 py-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-neutral-200">📍 By Location</span>
          <span className="text-neutral-500">·</span>
          <span className="text-neutral-400">
            {geotagged.length} geotagged {geotagged.length === 1 ? "photo" : "photos"} in {clusters.length} {clusters.length === 1 ? "place" : "places"}
          </span>
          {untagged.length > 0 && (
            <span className="text-neutral-500 text-[11px]">
              ({untagged.length} without GPS)
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center rounded-md border border-neutral-700/60 bg-neutral-800 p-0.5">
            <button
              type="button"
              onClick={() => {
                setFilterMode("geotagged");
                setSelectedClusterId(null);
              }}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                filterMode === "geotagged" && !selectedClusterId
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Geotagged ({geotagged.length})
            </button>
            <button
              type="button"
              onClick={() => {
                setFilterMode("all");
                setSelectedClusterId(null);
              }}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                filterMode === "all" && !selectedClusterId
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              All Photos ({photos.length})
            </button>
            <button
              type="button"
              onClick={() => {
                setFilterMode("untagged");
                setSelectedClusterId(null);
              }}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
                filterMode === "untagged"
                  ? "bg-[var(--theme-accent,#0ea5e9)] font-semibold text-white shadow-xs"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              No GPS ({untagged.length})
            </button>
          </div>

          <button
            type="button"
            onClick={handleScanGps}
            disabled={scanning}
            className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 px-2 py-1 text-[11px] font-medium transition-colors cursor-pointer disabled:opacity-40"
            title="Scan photos to extract GPS coordinates from camera EXIF"
          >
            {scanning ? "Scanning…" : "🔍 Scan GPS"}
          </button>
        </div>
      </div>

      {scanMessage && (
        <div className="bg-sky-950/80 border-b border-sky-800/80 px-4 py-1 text-[11px] text-sky-200 flex items-center justify-between">
          <span>{scanMessage}</span>
          <button onClick={() => setScanMessage(null)} className="text-sky-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Main Split: Interactive Map on Top, Photo Grid on Bottom */}
      <div className="flex min-h-0 flex-1 flex-col">
        {/* Map View Pane */}
        {filterMode !== "untagged" && (
          <div
            ref={mapContainerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
            className="relative h-72 w-full overflow-hidden border-b border-neutral-800 bg-neutral-900 select-none cursor-grab active:cursor-grabbing"
          >
            {/* OpenStreetMap Tiles */}
            <div className="absolute inset-0 pointer-events-none">
              {tiles.map((t) => (
                <img
                  key={t.key}
                  src={t.url}
                  alt=""
                  draggable={false}
                  className="absolute h-64 w-64 opacity-85 brightness-90 contrast-105"
                  style={{ left: t.x, top: t.y }}
                />
              ))}
            </div>

            {/* Map Cluster Markers */}
            <div className="absolute inset-0 pointer-events-none">
              {clusters.map((cluster) => {
                const pos = project(cluster.centerLat, cluster.centerLon, mapW, mapH);
                if (pos.x < -40 || pos.x > mapW + 40 || pos.y < -40 || pos.y > mapH + 40) return null;
                const isSelected = selectedClusterId === cluster.id;
                const thumb = cluster.photos[0] ? thumbUrl(cluster.photos[0]) : null;

                return (
                  <div
                    key={cluster.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = isSelected ? null : cluster.id;
                      setSelectedClusterId(next);
                      setCenter({ lat: cluster.centerLat, lon: cluster.centerLon });
                      setZoom((z) => Math.max(z, 8));
                      if (next && onSelectMultiple) {
                        onSelectMultiple(cluster.photos.map((p) => p.id));
                      }
                    }}
                    style={{ left: pos.x, top: pos.y }}
                    className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group transition-transform hover:scale-110 active:scale-95"
                    title={`${cluster.name}: ${cluster.photos.length} photo(s)`}
                  >
                    <div
                      className={`relative flex items-center justify-center rounded-full p-1 shadow-lg transition-all ${
                        isSelected
                          ? "bg-[var(--theme-accent,#0ea5e9)] ring-4 ring-[var(--theme-accent,#0ea5e9)]/40 scale-110"
                          : "bg-neutral-900/90 hover:bg-neutral-800 border-2 border-white/80 ring-2 ring-black/40"
                      }`}
                    >
                      {thumb ? (
                        <div className="h-10 w-10 overflow-hidden rounded-full bg-neutral-800">
                          <img src={thumb} alt="" className="h-full w-full object-cover" />
                        </div>
                      ) : (
                        <span className="px-2 text-xs">📍</span>
                      )}
                      <span className="absolute -top-1.5 -right-1.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-rose-600 px-1 font-mono text-[10px] font-bold text-white shadow-md ring-1 ring-white/50">
                        {cluster.photos.length}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Map Controls */}
            <div className="absolute bottom-3 right-3 flex flex-col gap-1 rounded-md bg-neutral-900/90 p-1 backdrop-blur-md border border-neutral-700/60 shadow-lg">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(18, z + 1))}
                className="flex h-7 w-7 items-center justify-center rounded text-neutral-300 hover:bg-neutral-800 hover:text-white font-bold cursor-pointer"
                title="Zoom in"
              >
                +
              </button>
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(2, z - 1))}
                className="flex h-7 w-7 items-center justify-center rounded text-neutral-300 hover:bg-neutral-800 hover:text-white font-bold cursor-pointer"
                title="Zoom out"
              >
                −
              </button>
              {geotagged.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    let sumLat = 0;
                    let sumLon = 0;
                    for (const g of geotagged) {
                      sumLat += g.lat;
                      sumLon += g.lon;
                    }
                    setCenter({ lat: sumLat / geotagged.length, lon: sumLon / geotagged.length });
                    setZoom(geotagged.length > 1 ? 5 : 12);
                    setSelectedClusterId(null);
                  }}
                  className="flex h-7 w-7 items-center justify-center rounded text-neutral-300 hover:bg-neutral-800 hover:text-white text-xs cursor-pointer"
                  title="Fit all geotagged photos"
                >
                  ⌖
                </button>
              )}
            </div>

            {/* Map Attribution */}
            <div className="absolute bottom-1 left-2 text-[9px] text-neutral-500/80 bg-neutral-950/60 px-1.5 py-0.5 rounded backdrop-blur-xs">
              ©{" "}
              <a
                href="https://www.openstreetmap.org/copyright"
                onClick={(e) => {
                  e.preventDefault();
                  api.openExternalUrl("https://www.openstreetmap.org/copyright");
                }}
                className="hover:underline cursor-pointer"
              >
                OpenStreetMap
              </a>{" "}
              contributors
            </div>
          </div>
        )}

        {/* Selected Cluster Header or Places Carousel */}
        {selectedClusterId && (
          <div className="flex items-center justify-between border-b border-neutral-800 bg-sky-950/40 px-4 py-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-sm">📍</span>
              <span className="font-semibold text-white">
                {clusters.find((c) => c.id === selectedClusterId)?.name}
              </span>
              <span className="text-neutral-400">
                ({activePhotos.length} photo{activePhotos.length === 1 ? "" : "s"})
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSelectedClusterId(null)}
              className="rounded bg-neutral-800 px-2 py-0.5 text-[11px] font-medium text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors cursor-pointer"
            >
              ✕ Show All Locations
            </button>
          </div>
        )}

        {/* Photo Grid of Location */}
        <div className="flex-1 overflow-y-auto p-4">
          {activePhotos.length === 0 ? (
            <div className="flex h-64 flex-col items-center justify-center gap-3 text-neutral-500">
              <span className="text-3xl">📍</span>
              <p className="text-sm font-medium text-neutral-300">
                {filterMode === "geotagged" ? "No geotagged photos found" : "No photos to display"}
              </p>
              <p className="text-xs text-neutral-500 max-w-md text-center">
                Photos taken with smartphones or GPS-enabled cameras automatically carry geographic coordinates in EXIF metadata.
              </p>
              <button
                type="button"
                onClick={handleScanGps}
                disabled={scanning}
                className="mt-1 rounded bg-[var(--theme-accent,#0ea5e9)] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[var(--theme-accent-hover,#0284c7)] transition-colors cursor-pointer"
              >
                Scan Library for GPS Metadata
              </button>
            </div>
          ) : (
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}
            >
              {activePhotos.map((ph) => (
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
    </div>
  );
}
