import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import { api, DownloadProgress, ReleaseAsset, size, UpdateInfo } from "./api";

export function UpdateModal({
  info,
  loading,
  error,
  onCheckAgain,
  onClose,
}: {
  info: UpdateInfo | null;
  loading: boolean;
  error: string | null;
  onCheckAgain: () => void;
  onClose: () => void;
}) {
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadedPath, setDownloadedPath] = useState<string | null>(null);
  const [activeAsset, setActiveAsset] = useState<string | null>(null);
  const [showAllAssets, setShowAllAssets] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("update-download-progress", (event) => {
      setDownloadProgress(event.payload);
      if (event.payload.status === "completed") {
        setDownloading(false);
        if (event.payload.path) {
          setDownloadedPath(event.payload.path);
        }
      } else if (event.payload.status === "error") {
        setDownloading(false);
        setInstallError(event.payload.error ?? "Download failed");
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const handleDownload = async (asset: ReleaseAsset) => {
    try {
      setInstallError(null);
      setDownloading(true);
      setActiveAsset(asset.name);
      setDownloadProgress({
        status: "downloading",
        downloaded: 0,
        total: asset.size,
        percent: 0,
      });

      const path = await api.downloadUpdate(asset.browser_download_url, asset.name, asset.sha256);
      setDownloadedPath(path);
      setDownloading(false);
    } catch (e: unknown) {
      setDownloading(false);
      setInstallError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleInstall = async () => {
    if (!downloadedPath) return;
    try {
      setInstallError(null);
      await api.installUpdate(downloadedPath);
    } catch (e: unknown) {
      setInstallError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleShowInFolder = async () => {
    if (!downloadedPath) return;
    try {
      await api.showInFolder(downloadedPath);
    } catch (e: unknown) {
      setInstallError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleOpenGitHub = () => {
    const url = info?.html_url || "https://github.com/rcwkan/optixel-studio/releases";
    api.openExternalUrl(url).catch(() => window.open(url, "_blank"));
  };

  const formatReleaseDate = (isoStr?: string) => {
    if (!isoStr) return "";
    try {
      return new Date(isoStr).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl max-h-[85vh] flex flex-col rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <header className="flex shrink-0 items-center justify-between border-b border-neutral-800 px-5 py-3.5 bg-neutral-900/90">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--theme-accent,#0ea5e9)]/20 text-[var(--theme-accent,#0ea5e9)] text-sm font-semibold">
              🚀
            </span>
            <div>
              <h2 className="text-sm font-semibold text-white">Software Update</h2>
              <p className="text-[11px] text-neutral-400">Optixel Studio Release Channel</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1 text-neutral-400 hover:bg-neutral-800 hover:text-white transition-colors cursor-pointer"
            title="Close"
          >
            ✕
          </button>
        </header>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 text-xs">
          {loading && (
            <div className="flex flex-col items-center justify-center py-10 space-y-3">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-600 border-t-[var(--theme-accent,#0ea5e9)]" />
              <p className="text-neutral-300 font-medium">Checking for latest release from GitHub…</p>
              <p className="text-[11px] text-neutral-500">https://github.com/rcwkan/optixel-studio/releases</p>
            </div>
          )}

          {!loading && error && (
            <div className="rounded-lg border border-red-500/30 bg-red-950/20 p-4 space-y-3">
              <div className="flex items-start gap-2.5">
                <span className="text-base text-red-400">⚠️</span>
                <div className="space-y-1 flex-1">
                  <h3 className="font-semibold text-red-300">Unable to Check for Updates</h3>
                  <p className="text-neutral-300 text-[11px] leading-relaxed">{error}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={onCheckAgain}
                  className="rounded bg-neutral-800 hover:bg-neutral-700 px-3 py-1.5 font-medium text-neutral-200 transition-colors cursor-pointer"
                >
                  Try Again
                </button>
                <button
                  onClick={handleOpenGitHub}
                  className="rounded bg-neutral-800 hover:bg-neutral-700 px-3 py-1.5 font-medium text-neutral-300 transition-colors cursor-pointer"
                >
                  Open GitHub Releases
                </button>
              </div>
            </div>
          )}

          {!loading && !error && info && (
            <>
              {/* Status Banner */}
              {info.has_update ? (
                <div className="rounded-lg border border-[var(--theme-accent,#0ea5e9)]/40 bg-[var(--theme-accent,#0ea5e9)]/10 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <span className="inline-block rounded-full bg-[var(--theme-accent,#0ea5e9)]/30 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[var(--theme-accent,#0ea5e9)]">
                        New Update Available
                      </span>
                      <h3 className="mt-1.5 text-base font-semibold text-white">
                        {info.release_name || `Version ${info.latest_version}`}
                      </h3>
                      <div className="mt-1 flex items-center gap-3 text-neutral-400 text-[11px]">
                        <span>Installed: v{info.current_version}</span>
                        <span>•</span>
                        <span className="font-medium text-white">Latest: {info.release_tag}</span>
                        {info.published_at && (
                          <>
                            <span>•</span>
                            <span>{formatReleaseDate(info.published_at)}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/20 p-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 text-lg">
                      ✓
                    </div>
                    <div>
                      <h3 className="font-semibold text-emerald-300">Optixel Studio is up to date</h3>
                      <p className="text-neutral-400 text-[11px] mt-0.5">
                        You are running version <span className="text-neutral-200 font-medium">{info.current_version}</span>, which is the latest available release.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Download / Installation Progress & Action Card */}
              {info.recommended_asset && (
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                        {info.has_update ? "Recommended Package for your System" : "Installer Package"}
                      </span>
                      <p className="mt-0.5 font-medium text-neutral-200 truncate max-w-sm">
                        {info.recommended_asset.name}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[11px] text-neutral-500">
                          {size(info.recommended_asset.size)}
                        </span>
                        {info.recommended_asset.sha256 && (
                          <span
                            className="font-mono text-[10px] text-neutral-400 bg-neutral-900 px-1.5 py-0.2 rounded border border-neutral-800 truncate max-w-[200px]"
                            title={`SHA-256: ${info.recommended_asset.sha256}`}
                          >
                            SHA: {info.recommended_asset.sha256.slice(0, 10)}…
                          </span>
                        )}
                      </div>
                    </div>

                    {!downloading && !downloadedPath && (
                      <button
                        onClick={() => handleDownload(info.recommended_asset!)}
                        className="rounded-lg bg-[var(--theme-accent,#0ea5e9)] hover:brightness-110 px-4 py-2 text-xs font-semibold text-white shadow-md transition-all cursor-pointer flex items-center gap-1.5"
                      >
                        <span>⬇ Download {info.has_update ? "Update" : "Package"}</span>
                      </button>
                    )}
                  </div>

                  {/* Downloading Progress */}
                  {downloading && downloadProgress && (
                    <div className="space-y-2 pt-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-neutral-300 font-medium">
                          Downloading {activeAsset}…
                        </span>
                        <span className="text-neutral-400 tabular-nums">
                          {downloadProgress.percent.toFixed(0)}% ({size(downloadProgress.downloaded)} / {size(downloadProgress.total)})
                        </span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-800">
                        <div
                          className="h-full bg-[var(--theme-accent,#0ea5e9)] transition-all duration-150"
                          style={{ width: `${Math.min(100, Math.max(0, downloadProgress.percent))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Download Completed Actions */}
                  {downloadedPath && (
                    <div className="flex flex-col gap-2 pt-2 border-t border-neutral-800">
                      <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium">
                        <span>✓ Downloaded & SHA-256 verified</span>
                        <span className="text-[11px] text-neutral-400 truncate">({downloadedPath})</span>
                      </div>
                      <div className="flex items-center gap-2 pt-1">
                        <button
                          onClick={handleInstall}
                          className="rounded-lg bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-xs font-semibold text-white shadow-md transition-colors cursor-pointer flex items-center gap-1.5"
                        >
                          <span>🚀 Open / Run Installer</span>
                        </button>
                        <button
                          onClick={handleShowInFolder}
                          className="rounded-lg border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 px-3 py-2 text-xs font-medium text-neutral-200 transition-colors cursor-pointer"
                        >
                          Reveal in Folder
                        </button>
                      </div>
                    </div>
                  )}

                  {installError && (
                    <p className="text-xs text-red-400 pt-1">
                      ⚠️ {installError}
                    </p>
                  )}
                </div>
              )}

              {/* Release Notes */}
              {info.release_notes && (
                <div className="space-y-1.5">
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                    Release Notes
                  </h4>
                  <div className="max-h-48 overflow-y-auto rounded-lg border border-neutral-800 bg-neutral-950/80 p-3.5 text-neutral-300 font-mono text-[11px] leading-relaxed whitespace-pre-wrap select-text">
                    {info.release_notes}
                  </div>
                </div>
              )}

              {/* All Assets Toggle */}
              {info.all_assets && info.all_assets.length > 0 && (
                <div className="border-t border-neutral-800 pt-3">
                  <button
                    onClick={() => setShowAllAssets(!showAllAssets)}
                    className="flex items-center gap-1.5 text-[11px] font-medium text-neutral-400 hover:text-neutral-200 cursor-pointer"
                  >
                    <span>{showAllAssets ? "▾" : "▸"}</span>
                    <span>All Available Release Packages ({info.all_assets.length})</span>
                  </button>

                  {showAllAssets && (
                    <div className="mt-2 space-y-1.5 pl-3">
                      {info.all_assets.map((asset) => (
                        <div
                          key={asset.name}
                          className="flex items-center justify-between rounded border border-neutral-800/80 bg-neutral-950/40 px-3 py-1.5"
                        >
                          <div className="truncate mr-2">
                            <span className="font-mono text-neutral-300 text-[11px]">{asset.name}</span>
                            <span className="ml-2 text-neutral-500 text-[10px]">{size(asset.size)}</span>
                            {asset.sha256 && (
                              <span
                                className="ml-2 font-mono text-[9px] text-neutral-500 select-all"
                                title={`SHA-256: ${asset.sha256}`}
                              >
                                {asset.sha256.slice(0, 8)}…
                              </span>
                            )}
                          </div>
                          <button
                            onClick={() => handleDownload(asset)}
                            disabled={downloading}
                            className="shrink-0 rounded bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 px-2 py-0.5 text-[10px] font-medium text-neutral-200 cursor-pointer"
                          >
                            Download
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Modal Footer */}
        <footer className="flex shrink-0 items-center justify-between border-t border-neutral-800 px-5 py-3 bg-neutral-900/90">
          <button
            onClick={handleOpenGitHub}
            className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer"
          >
            <span>View Release on GitHub</span>
            <span>↗</span>
          </button>
          <div className="flex items-center gap-2">
            {!loading && (
              <button
                onClick={onCheckAgain}
                className="rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors cursor-pointer"
              >
                Check Again
              </button>
            )}
            <button
              onClick={onClose}
              className="rounded-lg bg-neutral-800 hover:bg-neutral-700 px-4 py-1.5 text-xs font-medium text-white transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

