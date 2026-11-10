import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Culling, fileManagerName, flagName, FocusPeakingResult, LABEL_COLORS, LABELS, Photo, RawInfo, thumbUrl } from "./api";
import { Btn, Icon, Stars } from "./controls";

export interface PhotoViewerProps {
  photos: Photo[];
  active: Photo | null;
  onSelectPhoto: (id: number) => void;
  onClose: () => void;
  openDevelop: (id: number) => void;
  cull: (u: Culling, ids?: number[]) => void;
  onMoveToBin?: () => void;
  onOpenCompare?: () => void;
  onVirtualCopy?: () => void;
  selectedIds?: number[];
  folderName?: string;
}

export function PhotoViewer({
  photos,
  active,
  onSelectPhoto,
  onClose,
  openDevelop,
  cull,
  onMoveToBin,
  onOpenCompare,
  onVirtualCopy,
  selectedIds = [],
  folderName = "Folder",
}: PhotoViewerProps) {
  const photo = active || photos[0] || null;
  const currentIndex = photos.findIndex((p) => p.id === photo?.id);

  // Viewport and container state
  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const filmstripRef = useRef<HTMLDivElement>(null);
  const activeThumbRef = useRef<HTMLDivElement>(null);

  // Zoom and pan
  const [zoom, setZoom] = useState(1);
  const [zoomMode, setZoomMode] = useState<"fit" | "1:1" | "custom">("fit");
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [viewportSize, setViewportSize] = useState({ w: 900, h: 600 });

  // Orientation & Transform
  const [rotation, setRotation] = useState(0);
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);

  // Toggles and modes (info HUD popup is hidden by default)
  const [showInfo, setShowInfo] = useState(() => {
    try {
      const stored = localStorage.getItem("optixel_viewer_show_info") ?? localStorage.getItem("openlight_viewer_show_info");
      return stored === "true"; // defaults to false
    } catch {
      return false;
    }
  });

  const toggleInfo = useCallback((val?: boolean) => {
    setShowInfo((prev) => {
      const next = typeof val === "boolean" ? val : !prev;
      try {
        localStorage.setItem("optixel_viewer_show_info", String(next));
      } catch {}
      return next;
    });
  }, []);
  const [showFilmstrip, setShowFilmstrip] = useState(true);
  const [slideshow, setSlideshow] = useState(false);
  const [slideshowInterval] = useState(3000); // 3 seconds
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [highResReady, setHighResReady] = useState(false);
  const [loadingHighRes, setLoadingHighRes] = useState(false);
  const [rawInfo, setRawInfo] = useState<RawInfo | null>(null);
  const [showAfOverlay, setShowAfOverlay] = useState(false);
  const [focusPeaking, setFocusPeaking] = useState<FocusPeakingResult | null>(null);
  const [showFocusPeaking, setShowFocusPeaking] = useState(false);

  useEffect(() => {
    if (!photo) {
      setRawInfo(null);
      setFocusPeaking(null);
      return;
    }
    let cancelled = false;
    api
      .rawInfo(photo.id)
      .then((info) => !cancelled && setRawInfo(info))
      .catch(() => !cancelled && setRawInfo(null));
    return () => {
      cancelled = true;
    };
  }, [photo?.id]);

  useEffect(() => {
    if (!photo || !showFocusPeaking) return;
    let cancelled = false;
    api
      .focusPeaking(photo.id)
      .then((res) => !cancelled && setFocusPeaking(res))
      .catch(() => !cancelled && setFocusPeaking(null));
    return () => {
      cancelled = true;
    };
  }, [photo?.id, showFocusPeaking]);

  // Measure viewport size
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setViewportSize({ w: rect.width, h: rect.height });
      }
    };
    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Compute base fit dimensions
  const aspect = useMemo(() => {
    if (!photo || !photo.width || !photo.height) return 1.5;
    return photo.width / photo.height;
  }, [photo]);

  const fitDimensions = useMemo(() => {
    const margin = 32;
    const maxW = Math.max(100, viewportSize.w - margin * 2);
    const maxH = Math.max(100, viewportSize.h - margin * 2);

    let w = maxW;
    let h = maxW / aspect;
    if (h > maxH) {
      h = maxH;
      w = maxH * aspect;
    }
    return { w, h };
  }, [viewportSize, aspect]);

  // Actual 1:1 scale relative to fit
  const actualScale = useMemo(() => {
    if (!photo || !photo.width || fitDimensions.w <= 0) return 2.5;
    // 100%: one photo pixel per screen pixel (a CSS pixel is devicePixelRatio of them)
    return Math.max(1, photo.width / (window.devicePixelRatio || 1) / fitDimensions.w);
  }, [photo, fitDimensions.w]);

  // Reset view to Fit
  const resetFit = useCallback(() => {
    setZoom(1);
    setZoomMode("fit");
    setPan({ x: 0, y: 0 });
  }, []);

  // Past 1:1 of the proxy the canvas is stretched: render the full-size photo once the view is zoomed in
  const fullDone = useRef<number | null>(null);
  useEffect(() => {
    if (!photo || zoom <= 1.05 || fullDone.current === photo.id) return;
    const id = photo.id;
    let cancelled = false;
    setLoadingHighRes(true);
    api
      .getDevelop(id)
      .then((dev) => api.render(id, dev, false, null, false, null, true))
      .then((r) => {
        const canvas = canvasRef.current;
        if (cancelled || !canvas) return;
        canvas.width = r.image.width;
        canvas.height = r.image.height;
        canvas.getContext("2d")?.putImageData(r.image, 0, 0);
        fullDone.current = id;
      })
      .catch((err) => console.warn("Full-size render:", err))
      .finally(() => !cancelled && setLoadingHighRes(false));
    return () => {
      cancelled = true;
    };
  }, [photo?.id, zoom > 1.05]);

  // High resolution render on photo change
  useEffect(() => {
    if (!photo) return;
    resetFit();
    setRotation(0);
    setFlipH(false);
    setFlipV(false);
    setHighResReady(false);
    setLoadingHighRes(true);
    fullDone.current = null;

    let cancelled = false;
    api
      .getDevelop(photo.id)
      .then((dev) => {
        if (cancelled) return null;
        return api.render(photo.id, dev, false, null);
      })
      .then((rendered) => {
        if (cancelled || !rendered) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = rendered.image.width;
        canvas.height = rendered.image.height;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.putImageData(rendered.image, 0, 0);
          setHighResReady(true);
        }
      })
      .catch((err) => {
        console.warn("High-res render fallback to thumb:", err);
      })
      .finally(() => {
        if (!cancelled) setLoadingHighRes(false);
      });

    return () => {
      cancelled = true;
    };
  }, [photo?.id, resetFit]);

  // Navigation handlers
  const prevPhoto = useCallback(() => {
    if (photos.length <= 1) return;
    const prevIdx = currentIndex > 0 ? currentIndex - 1 : photos.length - 1;
    onSelectPhoto(photos[prevIdx].id);
  }, [photos, currentIndex, onSelectPhoto]);

  const nextPhoto = useCallback(() => {
    if (photos.length <= 1) return;
    const nextIdx = currentIndex < photos.length - 1 ? currentIndex + 1 : 0;
    onSelectPhoto(photos[nextIdx].id);
  }, [photos, currentIndex, onSelectPhoto]);

  // Slideshow auto-advance
  useEffect(() => {
    if (!slideshow) return;
    const timer = setInterval(() => {
      nextPhoto();
    }, slideshowInterval);
    return () => clearInterval(timer);
  }, [slideshow, nextPhoto, slideshowInterval]);

  // Auto-scroll active photo in filmstrip
  useEffect(() => {
    if (activeThumbRef.current) {
      activeThumbRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
    }
  }, [photo?.id]);

  // Mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((prev) => {
      const next = Math.max(0.2, Math.min(12, prev * factor));
      if (Math.abs(next - 1) < 0.04 && factor < 1) {
        setZoomMode("fit");
        setPan({ x: 0, y: 0 });
        return 1;
      }
      setZoomMode("custom");
      return next;
    });
  };

  // Drag to pan
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest("button, select, input, .no-drag")) return;
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    panStart.current = { ...pan };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    setPan({ x: panStart.current.x + dx, y: panStart.current.y + dy });
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  // Double click toggles between Fit and 1:1
  const handleDoubleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest("button, select, input")) return;
    if (zoomMode === "fit" || zoom <= 1.05) {
      setZoom(actualScale);
      setZoomMode("1:1");
    } else {
      resetFit();
    }
  };

  // Rotate handlers
  const rotateCCW = () => setRotation((r) => (r - 90 + 360) % 360);
  const rotateCW = () => setRotation((r) => (r + 90) % 360);
  const toggleFlipH = () => setFlipH((f) => !f);
  const toggleFlipV = () => setFlipV((f) => !f);

  // Fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
      setIsFullscreen(false);
    }
  };

  // Listen to external fullscreen exit (e.g. Esc on OS level)
  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;

      const k = e.key.toLowerCase();
      if (k === "escape") {
        e.preventDefault();
        if (isFullscreen) {
          if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
          setIsFullscreen(false);
        } else {
          onClose();
        }
      } else if (k === "arrowleft" || k === "pageup") {
        e.preventDefault();
        prevPhoto();
      } else if (k === "arrowright" || k === "pagedown") {
        e.preventDefault();
        nextPhoto();
      } else if (k === "home") {
        e.preventDefault();
        if (photos.length > 0) onSelectPhoto(photos[0].id);
      } else if (k === "end") {
        e.preventDefault();
        if (photos.length > 0) onSelectPhoto(photos[photos.length - 1].id);
      } else if (k === " ") {
        e.preventDefault();
        if (slideshow) {
          setSlideshow(false);
        } else {
          nextPhoto();
        }
      } else if (k === "z") {
        e.preventDefault();
        if (zoomMode === "fit" || zoom <= 1.05) {
          setZoom(actualScale);
          setZoomMode("1:1");
        } else {
          resetFit();
        }
      } else if (k === "r") {
        e.preventDefault();
        rotateCW();
      } else if (k === "l") {
        e.preventDefault();
        rotateCCW();
      } else if (k === "i") {
        e.preventDefault();
        toggleInfo();
      } else if (k === "f") {
        e.preventDefault();
        toggleFullscreen();
      } else if (k === "s" && e.altKey) {
        e.preventDefault();
        setSlideshow((s) => !s);
      } else if (k === "t") {
        e.preventDefault();
        setShowFilmstrip((s) => !s);
      } else if (k === "d" && photo) {
        e.preventDefault();
        openDevelop(photo.id);
      } else if (k === "c" && onOpenCompare) {
        e.preventDefault();
        onOpenCompare();
      } else if (k === "o" && photo) {
        e.preventDefault();
        void api.showInFolder(photo.path);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isFullscreen,
    onClose,
    prevPhoto,
    nextPhoto,
    photos,
    onSelectPhoto,
    slideshow,
    zoomMode,
    zoom,
    actualScale,
    resetFit,
    photo,
    openDevelop,
    onOpenCompare,
  ]);

  if (!photo) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-neutral-950 p-6 text-neutral-400">
        <p className="text-sm">No photo available to view.</p>
        <Btn onClick={onClose}>← Back to Grid</Btn>
      </div>
    );
  }

  // Megapixels
  const megapixels = photo.width && photo.height ? ((photo.width * photo.height) / 1000000).toFixed(1) : null;

  // Zoom percentage readout
  const zoomPercent = Math.round((zoom / actualScale) * 100);

  return (
    <div
      className={`flex flex-col bg-neutral-950 text-neutral-100 select-none ${
        isFullscreen ? "fixed inset-0 z-50 h-screen w-screen" : "relative flex-1 min-h-0"
      }`}
    >
      {/* Top Header Toolbar */}
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-neutral-800/90 bg-neutral-900/95 px-3 py-1.5 text-xs backdrop-blur-sm z-20">
        {/* Left: Back button & Breadcrumb */}
        <div className="flex items-center gap-2.5 min-w-0">
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 px-2.5 py-1 text-xs font-semibold text-neutral-200 transition-colors shadow-xs border border-neutral-700/70 cursor-pointer"
            title="Return to thumbnail grid view (Esc or G)"
          >
            <span>←</span>
            <span>Grid</span>
          </button>

          <div className="flex items-center gap-1.5 truncate text-[11px] text-neutral-400">
            <button
              type="button"
              onClick={() => void api.showInFolder(photo.path)}
              className="flex items-center gap-1 rounded hover:bg-neutral-800 px-1.5 py-0.5 text-neutral-300 hover:text-white transition-colors cursor-pointer group"
              title={`Open in ${fileManagerName()} (Reveal file)`}
            >
              <span className="truncate font-medium text-neutral-300 group-hover:text-white" title={photo.folder}>
                📁 {folderName}
              </span>
              <span className="text-[10px] text-neutral-500 group-hover:text-sky-400">↗</span>
            </button>
            <span className="text-neutral-600">›</span>
            <span className="truncate font-semibold text-white" title={photo.filename}>
              {photo.filename}
            </span>
            {photo.copy_name && (
              <span className="rounded bg-sky-950/80 px-1.5 py-0.2 text-[9px] font-medium text-sky-300 border border-sky-800/60">
                {photo.copy_name}
              </span>
            )}
            {photo.edited && (
              <span className="badge-edited" title="Edited photo (adjustments applied)">
                ±
              </span>
            )}
          </div>
        </div>

        {/* Center: Rating & Flag Quick Culling */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 rounded bg-neutral-800/80 px-2 py-0.5 border border-neutral-700/60">
            <Stars value={photo.rating} onChange={(r) => cull({ rating: r }, [photo.id])} />
          </div>

          <div className="flex items-center gap-0.5 rounded bg-neutral-800/80 p-0.5 border border-neutral-700/60">
            {(["pick", "unflagged", "reject"] as const).map((fl) => (
              <button
                key={fl}
                onClick={() => cull({ flag: fl }, [photo.id])}
                className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                  flagName(photo.flag) === fl
                    ? fl === "pick"
                      ? "bg-emerald-600 text-white font-bold"
                      : fl === "reject"
                      ? "bg-rose-600 text-white font-bold"
                      : "bg-neutral-600 text-white font-bold"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
                title={`Flag as ${fl} (${fl === "pick" ? "P" : fl === "reject" ? "X" : "U"})`}
              >
                {fl === "pick" ? "⚑ Pick" : fl === "reject" ? "✕ Reject" : "—"}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1">
            {LABELS.map((l) => (
              <button
                key={l}
                title={`Label ${l}`}
                onClick={() => cull({ color_label: photo.color_label === l ? "none" : l }, [photo.id])}
                className={`h-3.5 w-3.5 rounded-full transition-transform hover:scale-110 ${
                  photo.color_label === l ? "ring-2 ring-white scale-110" : ""
                }`}
                style={{ background: LABEL_COLORS[l] }}
              />
            ))}
          </div>
        </div>

        {/* Right: Actions & Module links */}
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-neutral-400 tabular-nums">
            {currentIndex + 1} / {photos.length}
          </span>

          <button
            type="button"
            onClick={() => void api.showInFolder(photo.path)}
            className="flex items-center gap-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 text-neutral-200 px-2.5 py-1 text-xs border border-neutral-700/80 transition-colors shadow-xs cursor-pointer"
            title={`Reveal photo in ${fileManagerName()} (Shortcut: O)`}
          >
            <svg className="h-3.5 w-3.5 text-neutral-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
            <span className="hidden sm:inline">Open in {fileManagerName()}</span>
          </button>

          <button
            onClick={() => openDevelop(photo.id)}
            className="flex items-center gap-1.5 rounded-md bg-[var(--theme-accent,#0ea5e9)] hover:bg-[var(--theme-accent-hover,#0284c7)] text-[var(--theme-accent-contrast,#ffffff)] px-3 py-1 text-xs font-semibold shadow-xs transition-colors cursor-pointer"
            title="Open photo in Develop module for full editing (D)"
          >
            <Icon d="M12 20h9 M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            <span>Develop</span>
          </button>

          {selectedIds.length >= 2 && onOpenCompare && (
            <button
              onClick={onOpenCompare}
              className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 px-2 py-1 text-xs border border-neutral-700"
              title="Compare selected photos (C)"
            >
              <span>⧉ Compare</span>
            </button>
          )}

          {onVirtualCopy && (
            <button
              onClick={onVirtualCopy}
              className="rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 px-2 py-1 text-[11px] border border-neutral-700"
              title="Create a virtual copy of this photo"
            >
              Copy
            </button>
          )}

          {onMoveToBin && (
            <button
              onClick={onMoveToBin}
              className="rounded bg-neutral-800 hover:bg-rose-950/60 text-neutral-400 hover:text-rose-400 px-2 py-1 text-[11px] border border-neutral-700 transition-colors"
              title="Move photo to Bin (Delete / Backspace)"
            >
              🗑️
            </button>
          )}
        </div>
      </header>

      {/* Main Center Viewport Stage */}
      <div
        ref={viewportRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        className={`relative flex flex-1 items-center justify-center overflow-hidden bg-neutral-950 select-none ${
          zoom > 1.05 ? (isDragging.current ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
        }`}
      >
        {/* Floating Prev / Next Chevrons on Hover */}
        {photos.length > 1 && (
          <>
            <button
              onClick={(e) => {
                e.stopPropagation();
                prevPhoto();
              }}
              className="no-drag absolute left-4 top-1/2 -translate-y-1/2 z-30 flex h-12 w-12 items-center justify-center rounded-full bg-black/40 hover:bg-black/80 text-white/70 hover:text-white border border-white/10 hover:border-white/30 backdrop-blur-md transition-all duration-150 shadow-xl cursor-pointer"
              title="Previous photo (← Arrow)"
            >
              <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                nextPhoto();
              }}
              className="no-drag absolute right-4 top-1/2 -translate-y-1/2 z-30 flex h-12 w-12 items-center justify-center rounded-full bg-black/40 hover:bg-black/80 text-white/70 hover:text-white border border-white/10 hover:border-white/30 backdrop-blur-md transition-all duration-150 shadow-xl cursor-pointer"
              title="Next photo (→ Arrow or Space)"
            >
              <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </>
        )}

        {/* Display Canvas & Image Container */}
        <div
          className="relative transition-transform duration-75 ease-out"
          style={{
            width: fitDimensions.w,
            height: fitDimensions.h,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom}) rotate(${rotation}deg) scaleX(${
              flipH ? -1 : 1
            }) scaleY(${flipV ? -1 : 1})`,
            transformOrigin: "center center",
          }}
        >
          {/* Instant Thumbnail layer (Always ready) */}
          <img
            src={thumbUrl(photo)}
            alt={photo.filename}
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain pointer-events-none shadow-2xl"
          />

          {/* High-res rendered Canvas layer (fades in once rendered) */}
          <canvas
            ref={canvasRef}
            className={`absolute inset-0 h-full w-full object-contain pointer-events-none transition-opacity duration-200 ${
              highResReady ? "opacity-100" : "opacity-0"
            }`}
          />

          {/* C++ Focus-Peaking Overlay */}
          {showFocusPeaking && focusPeaking && (
            <img
              src={focusPeaking.overlay_data_url}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain mix-blend-screen"
            />
          )}

          {/* Active AF Points & Eye-AF Overlay (MakerNotes) */}
          {showAfOverlay && rawInfo && rawInfo.af_points.length > 0 && (
            <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
              {rawInfo.af_points.map((af, idx) => {
                const isEye = af.kind === "eye" || af.kind === "face";
                const stroke = isEye ? "#10b981" : af.active ? "#ef4444" : "rgba(255,255,255,0.38)";
                const sw = isEye || af.active ? 2 : 1;
                return (
                  <rect
                    key={idx}
                    x={af.x - af.w / 2}
                    y={af.y - af.h / 2}
                    width={af.w}
                    height={af.h}
                    fill={af.active ? (isEye ? "rgba(16,185,129,0.14)" : "rgba(239,68,68,0.14)") : "none"}
                    stroke={stroke}
                    strokeWidth={sw}
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
            </svg>
          )}
        </div>

        {/* Loading Indicator for High-Res */}
        {loadingHighRes && !highResReady && (
          <div className="pointer-events-none absolute bottom-4 left-4 z-20 flex items-center gap-2 rounded-full bg-black/70 px-3 py-1 text-[11px] text-neutral-300 backdrop-blur-md border border-neutral-800">
            <svg className="h-3 w-3 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
            </svg>
            <span>Rendering full resolution…</span>
          </div>
        )}

        {/* Slideshow Active Indicator */}
        {slideshow && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 rounded-full bg-sky-950/90 border border-sky-600/80 px-4 py-1.5 text-xs font-semibold text-sky-200 shadow-xl backdrop-blur-md animate-pulse">
            <span className="h-2 w-2 rounded-full bg-sky-400" />
            <span>Slideshow Playing ({slideshowInterval / 1000}s)</span>
            <button
              onClick={() => setSlideshow(false)}
              className="ml-1 rounded bg-sky-800/80 hover:bg-sky-700 px-2 py-0.5 text-[10px] text-white"
            >
              Pause
            </button>
          </div>
        )}

        {/* ACDSee-style Information HUD / OSD Overlay */}
        {showInfo && (
          <div className="no-drag absolute top-4 left-4 z-20 max-w-sm rounded-xl border border-neutral-800/90 bg-neutral-900/85 p-3.5 text-xs text-neutral-200 shadow-2xl backdrop-blur-md transition-all">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-2 mb-2">
              <div className="flex items-center gap-1.5 font-semibold text-white truncate">
                <Icon d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <span className="truncate">{photo.filename}</span>
              </div>
              <button
                onClick={() => toggleInfo(false)}
                className="text-neutral-500 hover:text-white transition-colors"
                title="Hide Info HUD (I)"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-[80px_1fr] gap-x-2 gap-y-1 text-[11px]">
              <span className="text-neutral-500">Folder</span>
              <button
                type="button"
                onClick={() => void api.showInFolder(photo.path)}
                className="text-left text-neutral-300 hover:text-sky-400 hover:underline truncate cursor-pointer flex items-center justify-between gap-1 group"
                title={`Open folder in ${fileManagerName()}`}
              >
                <span className="truncate" title={photo.folder}>{photo.folder}</span>
                <span className="text-[10px] text-neutral-500 group-hover:text-sky-400 shrink-0">↗</span>
              </button>

              {photo.width && photo.height && (
                <>
                  <span className="text-neutral-500">Dimensions</span>
                  <span className="text-neutral-300 font-mono">
                    {photo.width} × {photo.height} {megapixels ? `(${megapixels} MP)` : ""}
                  </span>
                </>
              )}

              {photo.captured_at && (
                <>
                  <span className="text-neutral-500">Captured</span>
                  <span className="text-neutral-300">{photo.captured_at}</span>
                </>
              )}

              {photo.camera && (
                <>
                  <span className="text-neutral-500">Camera</span>
                  <span className="text-neutral-300 truncate">{photo.camera}</span>
                </>
              )}

              {photo.lens && (
                <>
                  <span className="text-neutral-500">Lens</span>
                  <span className="text-neutral-300 truncate">{photo.lens}</span>
                </>
              )}

              {(photo.shutter || photo.aperture || photo.iso) && (
                <>
                  <span className="text-neutral-500">Exposure</span>
                  <span className="text-neutral-200 font-mono font-medium">
                    {[
                      photo.shutter ? `${photo.shutter}s` : null,
                      photo.aperture ? `f/${photo.aperture.toFixed(1)}` : null,
                      photo.iso ? `ISO ${photo.iso}` : null,
                      photo.focal_length ? `${Math.round(photo.focal_length)}mm` : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </>
              )}

              <span className="text-neutral-500">Status</span>
              <span className="text-neutral-300">
                {photo.rating ? `${"★".repeat(photo.rating)} ` : "No rating "}
                · {flagName(photo.flag)}
                {photo.color_label ? ` · ${photo.color_label}` : ""}
              </span>
              {rawInfo && rawInfo.shutter_count > 0 && (
                <>
                  <span className="text-neutral-500">Shutter Count</span>
                  <span className="text-emerald-300 font-mono">{rawInfo.shutter_count.toLocaleString()}</span>
                </>
              )}
              {rawInfo && rawInfo.lens_serial && (
                <>
                  <span className="text-neutral-500">Lens Serial</span>
                  <span className="text-neutral-300 font-mono truncate">{rawInfo.lens_serial}</span>
                </>
              )}
              {focusPeaking && (
                <>
                  <span className="text-neutral-500">Focus / Blink</span>
                  <span className="text-emerald-300 font-mono">
                    {focusPeaking.focus_score}% · {focusPeaking.blink_detected ? "⚠ Blink" : "✓ Open"}
                  </span>
                </>
              )}
            </div>

            <div className="mt-2.5 pt-2 border-t border-neutral-800/80 flex items-center justify-between text-[10px] text-neutral-500">
              <span>Zoom: {zoomPercent}%</span>
              <span>Double-click to toggle 1:1</span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Floating Control Bar */}
      <div className="flex shrink-0 items-center justify-between border-t border-neutral-800/90 bg-neutral-900/95 px-3 py-1.5 text-xs backdrop-blur-sm z-20">
        {/* Left: Navigation & Slideshow */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={prevPhoto}
            className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-neutral-700 px-2.5 py-1 text-xs text-neutral-200 border border-neutral-700/60"
            title="Previous Photo (← Arrow)"
          >
            ◀ Prev
          </button>
          <button
            onClick={nextPhoto}
            className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-neutral-700 px-2.5 py-1 text-xs text-neutral-200 border border-neutral-700/60"
            title="Next Photo (→ Arrow)"
          >
            Next ▶
          </button>

          <span className="text-neutral-600 px-1">|</span>

          <button
            onClick={() => setSlideshow(!slideshow)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors ${
              slideshow
                ? "bg-amber-600 text-white font-semibold"
                : "bg-neutral-800 hover:bg-neutral-700 text-neutral-300 border border-neutral-700/60"
            }`}
            title="Toggle automated photo slideshow (Space or Alt+S)"
          >
            <span>{slideshow ? "⏸ Pause" : "▶ Slideshow"}</span>
          </button>
        </div>

        {/* Center: Zoom Controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center rounded bg-neutral-800 p-0.5 border border-neutral-700/60">
            <button
              onClick={resetFit}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors ${
                zoomMode === "fit" ? "bg-neutral-600 text-white font-semibold" : "text-neutral-400 hover:text-white"
              }`}
              title="Fit to Window"
            >
              Fit
            </button>
            <button
              onClick={() => {
                setZoom(actualScale);
                setZoomMode("1:1");
                setPan({ x: 0, y: 0 });
              }}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors ${
                zoomMode === "1:1" ? "bg-neutral-600 text-white font-semibold" : "text-neutral-400 hover:text-white"
              }`}
              title="Actual Size 1:1 (Z)"
            >
              1:1
            </button>
          </div>

          <div className="flex items-center gap-1 text-neutral-400">
            <button
              onClick={() => {
                setZoom((z) => Math.max(0.2, z / 1.25));
                setZoomMode("custom");
              }}
              className="flex h-6 w-6 items-center justify-center rounded bg-neutral-800 hover:bg-neutral-700 text-sm font-bold text-neutral-300 border border-neutral-700/60"
              title="Zoom out (-)"
            >
              −
            </button>
            <span className="w-12 text-center font-mono text-[11px] text-neutral-300 tabular-nums">
              {zoomPercent}%
            </span>
            <button
              onClick={() => {
                setZoom((z) => Math.min(12, z * 1.25));
                setZoomMode("custom");
              }}
              className="flex h-6 w-6 items-center justify-center rounded bg-neutral-800 hover:bg-neutral-700 text-sm font-bold text-neutral-300 border border-neutral-700/60"
              title="Zoom in (+)"
            >
              +
            </button>
          </div>
        </div>

        {/* Right: Rotate, Flip, HUD, Filmstrip & Fullscreen */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={rotateCCW}
            className="flex h-7 w-7 items-center justify-center rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 border border-neutral-700/60 transition-colors"
            title="Rotate 90° Counter-Clockwise (L)"
          >
            ↺
          </button>
          <button
            onClick={rotateCW}
            className="flex h-7 w-7 items-center justify-center rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 border border-neutral-700/60 transition-colors"
            title="Rotate 90° Clockwise (R)"
          >
            ↻
          </button>
          <button
            onClick={toggleFlipH}
            className={`rounded px-1.5 py-1 text-[11px] font-medium border border-neutral-700/60 transition-colors ${
              flipH ? "bg-sky-600 text-white" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
            }`}
            title="Flip Horizontal (H)"
          >
            ⇄ Flip H
          </button>
          <button
            onClick={toggleFlipV}
            className={`rounded px-1.5 py-1 text-[11px] font-medium border border-neutral-700/60 transition-colors ${
              flipV ? "bg-sky-600 text-white" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
            }`}
            title="Flip Vertical (V)"
          >
            ⇅ Flip V
          </button>

          <span className="text-neutral-600 px-0.5">|</span>

          {rawInfo && rawInfo.af_points.length > 0 && (
            <button
              onClick={() => setShowAfOverlay(!showAfOverlay)}
              className={`rounded px-2 py-1 text-xs font-medium border border-neutral-700/60 transition-colors ${
                showAfOverlay ? "bg-rose-600 text-white font-semibold" : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
              title="Toggle Active AF Points & Eye-AF Overlay (decoded from camera MakerNotes)"
            >
              ⌖ AF ({rawInfo.af_points.filter((a) => a.active).length})
            </button>
          )}

          <button
            onClick={() => setShowFocusPeaking(!showFocusPeaking)}
            className={`rounded px-2 py-1 text-xs font-medium border border-neutral-700/60 transition-colors ${
              showFocusPeaking ? "bg-emerald-600 text-white font-semibold" : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
            title="Toggle C++ Focus-Peaking & Blink / EAR Culling Overlay"
          >
            ◉ Peaking
          </button>

          <button
            onClick={() => toggleInfo()}
            className={`rounded px-2 py-1 text-xs font-medium border border-neutral-700/60 transition-colors ${
              showInfo ? "bg-neutral-700 text-white font-semibold" : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
            title="Toggle EXIF Info HUD (I)"
          >
            ℹ Info
          </button>

          <button
            onClick={() => setShowFilmstrip(!showFilmstrip)}
            className={`rounded px-2 py-1 text-xs font-medium border border-neutral-700/60 transition-colors ${
              showFilmstrip ? "bg-neutral-700 text-white font-semibold" : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
            title="Toggle Filmstrip (T)"
          >
            ▦ Filmstrip
          </button>

          <button
            onClick={toggleFullscreen}
            className="flex items-center gap-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 px-2 py-1 text-xs border border-neutral-700/60"
            title="Toggle Fullscreen (F)"
          >
            <span>{isFullscreen ? "⛶ Exit" : "⛶ Fullscreen"}</span>
          </button>
        </div>
      </div>

      {/* Bottom Filmstrip Thumbnail Strip */}
      {showFilmstrip && photos.length > 0 && (
        <div
          ref={filmstripRef}
          className="flex h-20 shrink-0 items-center gap-2 overflow-x-auto border-t border-neutral-800 bg-neutral-900/90 px-3 py-2 z-20"
        >
          {photos.map((p, idx) => {
            const isActive = p.id === photo.id;
            return (
              <div
                key={p.id}
                ref={isActive ? activeThumbRef : undefined}
                onClick={() => onSelectPhoto(p.id)}
                className={`group relative flex h-16 w-20 shrink-0 cursor-pointer flex-col items-center justify-center rounded-md p-1 transition-all ${
                  isActive
                    ? "bg-neutral-700 ring-2 ring-[var(--theme-accent,#0ea5e9)] scale-105 shadow-lg"
                    : "bg-neutral-800 hover:bg-neutral-700/70 opacity-75 hover:opacity-100"
                }`}
                title={`#${idx + 1}: ${p.filename}`}
              >
                <img
                  src={thumbUrl(p)}
                  alt={p.filename}
                  loading="lazy"
                  draggable={false}
                  className="max-h-11 max-w-full object-contain pointer-events-none rounded"
                />

                <div className="mt-0.5 flex w-full items-center justify-between text-[9px] text-neutral-400 px-0.5">
                  <span className="font-mono text-neutral-500 tabular-nums">#{idx + 1}</span>
                  {p.rating > 0 && <span className="text-amber-400 font-bold">★{p.rating}</span>}
                  {p.flag === 1 && <span className="text-emerald-400">⚑</span>}
                  {p.flag === -1 && <span className="text-rose-400">✕</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
