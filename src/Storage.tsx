import { ask, open } from "@tauri-apps/plugin-dialog";
import { ReactNode, useEffect, useState } from "react";
import { api, CloudAccount, CloudEntry, CloudKind, DuplicateOptions, DuplicateResult, ImmichAlbum, JobProgress, OnImport, Place, Settings, size, Source } from "./api";
import { Btn } from "./controls";

const card = "rounded-lg border border-neutral-800 bg-neutral-900 p-4";
const field = "rounded bg-neutral-800 px-2 py-1 outline-none placeholder:text-neutral-500 focus:ring-1 focus:ring-sky-600";
const hint = "text-[11px] text-neutral-500";
const form = "grid grid-cols-[120px_minmax(0,1fr)] items-center gap-2";

export const KIND_NAMES: Record<CloudKind, string> = { webdav: "WebDAV", s3: "S3", dropbox: "Dropbox", onedrive: "OneDrive" };
const KINDS: [CloudKind, string][] = [
  ["webdav", "WebDAV · Nextcloud"],
  ["s3", "S3 · B2 · R2"],
  ["dropbox", "Dropbox"],
  ["onedrive", "OneDrive"],
];
/** Common services: where their files are (REGION is filled in from the region) and how to sign in. */
type Preset = { name: string; endpoint: string; region?: string; note: string };
const PRESETS: Record<"webdav" | "s3", Preset[]> = {
  webdav: [
    { name: "Nextcloud", endpoint: "https://YOUR-SERVER/remote.php/dav/files/YOUR-USER", note: "Sign in with an app password: Settings › Security › Devices & sessions." },
    { name: "ownCloud", endpoint: "https://YOUR-SERVER/remote.php/webdav", note: "With two-factor sign-in on, use an app password." },
    { name: "Synology NAS", endpoint: "https://YOUR-NAS:5006", note: "Install WebDAV Server in Package Center and switch on HTTPS." },
    { name: "pCloud", endpoint: "https://webdav.pcloud.com", note: "Accounts in the EU region: https://ewebdav.pcloud.com" },
    { name: "Koofr", endpoint: "https://app.koofr.net/dav/Koofr", note: "Sign in with an app password: Preferences › Password." },
    { name: "WebDAV", endpoint: "", note: "Any WebDAV server: its address, user name and password." },
  ],
  s3: [
    { name: "Amazon S3", endpoint: "https://s3.REGION.amazonaws.com", region: "us-east-1", note: "The access key of an IAM user allowed s3:ListBucket, s3:GetObject and s3:PutObject on the bucket." },
    { name: "Backblaze B2", endpoint: "https://s3.REGION.backblazeb2.com", region: "us-west-004", note: "An application key with access to the bucket; the bucket's page shows its endpoint." },
    { name: "Cloudflare R2", endpoint: "https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com", region: "auto", note: "An R2 API token with Object Read & Write." },
    { name: "Wasabi", endpoint: "https://s3.REGION.wasabisys.com", region: "us-east-1", note: "The access key of a user allowed to read and write the bucket." },
    { name: "MinIO", endpoint: "http://YOUR-SERVER:9000", region: "us-east-1", note: "The access key of a user allowed to read and write the bucket." },
  ],
};
/** What Optixel opens (develop::RAW_EXTS and IMG_EXTS) */
const PHOTO = /\.(jpe?g|png|tiff?|webp|dng|cr[23w]|nef|nrw|arw|sr[f2]|raf|orf|rw[2l]|pef|srw|3fr|iiq|erf|mef|mos|kdc|dcr)$/i;

/** Run one action at a time: `busy` says what is going on, a failure lands in `error`. */
function useAction() {
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const run = async (label: string, job: () => Promise<unknown>) => {
    setBusy(label || "…");
    setError("");
    try {
      await job();
    } catch (e) {
      setError(String(e));
    }
    setBusy("");
  };
  return { busy, error, run };
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <span className="text-neutral-400">{label}</span>
      {children}
    </>
  );
}

/** Where photos come from: network drives, cloud storage, photo services, and what happens to new photos. */
export function Locations(p: { settings: Settings; setSettings: (s: Settings) => void; onConnect: (s: Source) => void; onAuto: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-wrap content-start gap-4 overflow-y-auto p-4 text-[12px]">
      <div className="flex w-[560px] max-w-full flex-col gap-4">
        <NetworkDrives {...p} />
        <CloudStorage {...p} />
      </div>
      <div className="flex w-[440px] max-w-full flex-col gap-4">
        <Automation {...p} />
        <Services {...p} />
      </div>
    </div>
  );
}

/** A UNC path (backslashes, server first), `host` or `host/share` -> smb://host[/share]; an address with a scheme is kept. */
const toShareUrl = (v: string) => {
  v = v.trim();
  if (!v || /^[a-z][a-z0-9+.-]*:\/\//i.test(v)) return v;
  const [host, share] = v.replace(/\\/g, "/").replace(/^\/+/, "").split("/");
  return `smb://${host}${share ? "/" + encodeURIComponent(share) : ""}`;
};
/** smb://host/share -> the UNC path where the folder dialog should start browsing. */
const shareToUnc = (u: string) => {
  const m = /^smb:\/\/([^/]+)(?:\/([^/]+))?/i.exec(u.trim());
  return m ? `\\\\${m[1]}${m[2] ? "\\" + decodeURIComponent(m[2]) : ""}` : undefined;
};

function NetworkDrives({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const [online, setOnline] = useState<Record<string, boolean>>({});
  const [draft, setDraft] = useState({ name: "", url: "", path: "" });
  const [suggest, setSuggest] = useState<string | null>(null);
  const { busy, error, run } = useAction();
  const drives = settings.network_drives;
  const refresh = () => api.networkStatus().then((l) => setOnline(Object.fromEntries(l.map((d) => [d.id, d.online]))));
  useEffect(() => void refresh(), [drives.length]);
  // where the share will appear, as a hint for the folder
  useEffect(() => {
    const t = setTimeout(() => (draft.url.trim() ? api.networkMountPoint(draft.url.trim()).then(setSuggest, () => setSuggest(null)) : setSuggest(null)), 300);
    return () => clearTimeout(t);
  }, [draft.url]);
  const patch = (x: Partial<typeof draft>) => setDraft((d) => ({ ...d, ...x }));
  return (
    <section className={card}>
      <h2 className="font-semibold">Network drives</h2>
      <p className={`${hint} mb-2`}>Shares on a NAS or another computer. They show in the Library like folders and are connected when you open them; photos are read and edited in place.</p>
      {drives.length === 0 && <p className="py-1 text-neutral-500">No network drives yet.</p>}
      {drives.map((d) => (
        <div key={d.id} className="flex items-center gap-2 border-t border-neutral-800 py-2 first-of-type:border-t-0">
          <span className={`h-2 w-2 shrink-0 rounded-full ${online[d.id] ? "bg-emerald-500" : "bg-neutral-600"}`} title={online[d.id] ? "Connected" : "Not connected"} />
          <div className="min-w-0 flex-1">
            <div className="text-neutral-200">{d.name}</div>
            <div className="truncate font-mono text-[10px] text-neutral-500" title={d.path}>
              {d.url ? `${d.url} → ` : ""}
              {d.path}
            </div>
          </div>
          {!online[d.id] && d.url && (
            <Btn disabled={!!busy} onClick={() => run(`Connecting ${d.name}…`, () => api.networkConnect(d.id).then(refresh))}>
              Connect
            </Btn>
          )}
          <Btn disabled={!!busy} title="Forget the drive; its photos stay in the catalog" onClick={() => run("", async () => setSettings(await api.networkRemove(d.id)))}>
            Remove
          </Btn>
        </div>
      ))}
      <div className="mt-2 flex items-center gap-3 border-t border-neutral-800 pt-3">
        <Btn
          active
          disabled={!!busy}
          onClick={async () => {
            const dir = await open({ directory: true, title: "Choose a folder on your network (look under Network, or type \\\\server\\share)" });
            if (typeof dir !== "string") return;
            // a network path (\\server\share\...) also gives the address that connects it again later
            const unc = dir.startsWith("\\\\");
            run("Adding…", async () => setSettings(await api.networkAdd({ name: "", url: unc ? toShareUrl(dir) : "", path: dir })));
          }}
        >
          Add network folder…
        </Btn>
        <p className={hint}>Pick the folder in the Windows dialog; it is added as it is. Your system asks for the password if the share needs one.</p>
      </div>
      <details className="mt-2">
        <summary className={`${hint} cursor-pointer`}>Enter an address by hand instead</summary>
      <div className={`${form} mt-2`}>
        <Row label="Address">
          <input
            className={`${field} font-mono text-[11px]`}
            value={draft.url}
            placeholder="IP or name (192.168.2.101), \\server\share, or smb://nas.local/photos"
            onChange={(e) => patch({ url: e.target.value })}
            onBlur={() => {
              const v = draft.url.trim();
              // a pasted UNC path to a subfolder also gives the folder
              const sub = v.startsWith("\\\\") && v.split("\\").filter(Boolean).length > 2 && !draft.path.trim();
              patch({ url: toShareUrl(v), ...(sub ? { path: v } : {}) });
            }}
          />
        </Row>
        <Row label="Folder">
          <div className="flex min-w-0 gap-1">
            <input
              className={`${field} min-w-0 flex-1 font-mono text-[11px]`}
              value={draft.path}
              placeholder={suggest ?? "Where its files appear once connected"}
              onChange={(e) => patch({ path: e.target.value })}
            />
            <Btn
              onClick={async () => {
                const dir = await open({ directory: true, defaultPath: shareToUnc(draft.url) ?? (draft.path || undefined), title: "The folder of the connected share" });
                // a folder picked on a share fills in the share's address too
                if (typeof dir === "string") patch({ path: dir, ...(!draft.url.trim() && dir.startsWith("\\\\") ? { url: toShareUrl(dir) } : {}) });
              }}
            >
              Choose…
            </Btn>
          </div>
        </Row>
        <Row label="Name">
          <input className={field} value={draft.name} placeholder={(draft.path || suggest || "").split(/[\\/]/).filter(Boolean).pop() ?? "Network drive"} onChange={(e) => patch({ name: e.target.value })} />
        </Row>
      </div>
      <div className="mt-2 flex items-start justify-between gap-3">
        <p className={hint}>
          smb:// for Windows, Samba and most NAS · afp:// for older Macs · nfs:// · https:// for WebDAV. A share connected some other way (a mapped drive) needs only its folder. Your system asks for
          the password and remembers it; Optixel never sees it.
        </p>
        <Btn
          active
          className="shrink-0"
          disabled={!!busy || !(draft.url.trim() || draft.path.trim())}
          onClick={() =>
            run("Adding…", async () => {
              setSettings(await api.networkAdd(draft));
              setDraft({ name: "", url: "", path: "" });
            })
          }
        >
          Add drive
        </Btn>
      </div>
      </details>
      {busy &&<p className="mt-1 text-sky-300">{busy}</p>}
      {error && <p className="mt-1 break-words text-red-400">{error}</p>}
    </section>
  );
}

function CloudStorage({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const [kind, setKind] = useState<CloudKind>("webdav");
  const [preset, setPreset] = useState(0);
  const [name, setName] = useState(PRESETS.webdav[0].name);
  const [endpoint, setEndpoint] = useState(PRESETS.webdav[0].endpoint);
  const [user, setUser] = useState("");
  const [bucket, setBucket] = useState("");
  const [region, setRegion] = useState("");
  const [secret, setSecret] = useState("");
  const [code, setCode] = useState("");
  const [started, setStarted] = useState(false);
  const [note, setNote] = useState("");
  const [browse, setBrowse] = useState<CloudAccount | null>(null);
  const { busy, error, run } = useAction();
  const presets = kind === "webdav" || kind === "s3" ? PRESETS[kind] : [];
  const pre = presets[preset] as Preset | undefined;
  const choose = (k: CloudKind, i: number) => {
    const x = k === "webdav" || k === "s3" ? PRESETS[k][i] : undefined;
    setKind(k);
    setPreset(i);
    setName(x?.name ?? KIND_NAMES[k]);
    setEndpoint(x ? x.endpoint.replace("REGION", x.region ?? "") : "");
    setRegion(x?.region ?? "");
    setUser("");
    setBucket("");
    setSecret("");
    setCode("");
    setStarted(false);
  };
  // the endpoint follows the region while it is the preset's
  const changeRegion = (r: string) => {
    if (pre?.endpoint.includes("REGION") && endpoint === pre.endpoint.replace("REGION", region)) setEndpoint(pre.endpoint.replace("REGION", r));
    setRegion(r);
  };
  const account = (): Partial<CloudAccount> => ({ kind, name: name.trim(), endpoint: endpoint.trim(), user: user.trim(), bucket: bucket.trim(), region: region.trim() });
  const connected = (s: Settings) => {
    setSettings(s);
    choose(kind, preset);
    setNote("");
    setBrowse(s.clouds[s.clouds.length - 1] ?? null); // pick a folder of photos right away
  };
  const sync = (a: CloudAccount, path: string) => {
    setBrowse(null);
    run(`Downloading ${a.name}${path ? ` › ${path}` : ""}…`, async () => {
      const r = await api.cloudSync(a.id, path);
      setSettings(await api.settings());
      setNote(`${r.imported} new photo(s) from ${a.name}${r.failed ? `, ${r.failed} failed (${r.error})` : ""}. The folder is in the Library under Cloud storage.`);
    });
  };
  const ready =
    kind === "webdav" ? !!(endpoint.trim() && user.trim() && secret) : kind === "s3" ? !!(endpoint.trim() && bucket.trim() && user.trim() && secret) : kind === "dropbox" ? !!user.trim() && (!started || !!code.trim()) : !!user.trim();
  const connect = () => {
    if (kind === "webdav" || kind === "s3") return run("Connecting…", async () => connected(await api.cloudConnect(account(), secret)));
    if (kind === "onedrive") return run("Finish signing in to Microsoft in your browser…", async () => connected(await api.cloudOnedriveConnect(account())));
    if (!started) return run("Opening Dropbox…", async () => (await api.cloudDropboxStart(user.trim()), setStarted(true)));
    return run("Signing in…", async () => connected(await api.cloudDropboxFinish(account(), code)));
  };
  return (
    <section className={card}>
      <h2 className="font-semibold">Cloud storage</h2>
      <p className={`${hint} mb-2`}>Folders of photos in your cloud storage are downloaded into the catalog (only what's new, each time you open them); exports can be uploaded back.</p>
      {settings.clouds.length === 0 && <p className="py-1 text-neutral-500">No cloud storage connected yet.</p>}
      {settings.clouds.map((a) => (
        <div key={a.id} className="border-t border-neutral-800 py-2 first-of-type:border-t-0">
          <div className="flex items-center gap-2">
            <span className="text-neutral-200">{a.name}</span>
            <span className="truncate text-[10px] text-neutral-500">
              {KIND_NAMES[a.kind] ?? a.kind}
              {a.bucket ? ` · ${a.bucket}` : ""}
              {a.kind === "webdav" || a.kind === "s3" ? ` · ${a.endpoint}` : ""}
            </span>
            <span className="ml-auto flex shrink-0 gap-1">
              <Btn disabled={!!busy} onClick={() => setBrowse(a)}>
                Add a folder…
              </Btn>
              <Btn
                disabled={!!busy}
                onClick={async () => {
                  if (await ask(`Disconnect ${a.name}? Its sign-in is forgotten; downloaded photos stay in the catalog.`, { title: "Disconnect", kind: "warning" }))
                    run("", async () => setSettings(await api.cloudDisconnect(a.id)));
                }}
              >
                Disconnect
              </Btn>
            </span>
          </div>
          {a.folders.map((f) => (
            <div key={f} className="flex items-center gap-2 pl-3 text-[11px] text-neutral-400">
              <span className="text-amber-300">▸</span>
              <span className="min-w-0 flex-1 truncate">{f || "All files"}</span>
              <button className="text-neutral-500 hover:text-white" title="Download what's new" disabled={!!busy} onClick={() => sync(a, f)}>
                ↻
              </button>
              <button className="text-neutral-500 hover:text-white" title="Stop listing this folder; its photos stay" disabled={!!busy} onClick={() => run("", async () => setSettings(await api.cloudForgetFolder(a.id, f)))}>
                ✕
              </button>
            </div>
          ))}
        </div>
      ))}

      <div className="mt-2 flex flex-wrap gap-1 border-t border-neutral-800 pt-3">
        {KINDS.map(([k, label]) => (
          <Btn key={k} active={kind === k} onClick={() => choose(k, 0)}>
            {label}
          </Btn>
        ))}
      </div>
      <div className={`${form} mt-3`}>
        {presets.length > 0 && (
          <Row label="Service">
            <select className={field} value={preset} onChange={(e) => choose(kind, +e.target.value)}>
              {presets.map((x, i) => (
                <option key={x.name} value={i}>
                  {x.name}
                </option>
              ))}
            </select>
          </Row>
        )}
        {(kind === "webdav" || kind === "s3") && (
          <Row label={kind === "s3" ? "Endpoint" : "Address"}>
            <input className={`${field} font-mono text-[11px]`} value={endpoint} placeholder="https://…" onChange={(e) => setEndpoint(e.target.value)} />
          </Row>
        )}
        {kind === "s3" && (
          <>
            <Row label="Bucket">
              <input className={field} value={bucket} onChange={(e) => setBucket(e.target.value)} />
            </Row>
            <Row label="Region">
              <input className={field} value={region} placeholder="us-east-1" onChange={(e) => changeRegion(e.target.value)} />
            </Row>
          </>
        )}
        <Row label={{ webdav: "User name", s3: "Access key ID", dropbox: "App key", onedrive: "Application ID" }[kind]}>
          <input className={field} value={user} disabled={started} placeholder={kind === "onedrive" ? "00000000-0000-0000-0000-000000000000" : ""} onChange={(e) => setUser(e.target.value)} />
        </Row>
        {(kind === "webdav" || kind === "s3") && (
          <Row label={kind === "s3" ? "Secret key" : "Password"}>
            <input className={field} type="password" value={secret} onChange={(e) => setSecret(e.target.value)} />
          </Row>
        )}
        {kind === "dropbox" && started && (
          <Row label="Code">
            <input className={`${field} font-mono`} value={code} placeholder="The code Dropbox shows" onChange={(e) => setCode(e.target.value)} />
          </Row>
        )}
        <Row label="Name">
          <input className={field} value={name} onChange={(e) => setName(e.target.value)} />
        </Row>
      </div>
      <div className="mt-2 flex items-start justify-between gap-3">
        <p className={hint}>
          {kind === "dropbox"
            ? "Create an app at dropbox.com/developers/apps (Scoped access, Full Dropbox) with files.metadata.read, files.content.read and files.content.write, and paste its App key. Allow it in your browser, then paste the code Dropbox shows."
            : kind === "onedrive"
              ? "Register an app at entra.microsoft.com for personal and work accounts, add the platform Mobile and desktop applications with the redirect URI http://localhost and the delegated permissions Files.ReadWrite and offline_access, and paste its Application (client) ID. Signing in opens your browser."
              : pre?.note}{" "}
          Passwords and sign-ins are kept with Optixel's other keys, never in its settings file.
        </p>
        <span className="flex shrink-0 gap-1">
          {started && (
            <Btn disabled={!!busy} onClick={() => choose(kind, preset)}>
              Start over
            </Btn>
          )}
          <Btn active disabled={!!busy || !ready} onClick={connect}>
            {kind === "dropbox" ? (started ? "Finish" : "Sign in with Dropbox") : kind === "onedrive" ? "Sign in with Microsoft" : "Connect"}
          </Btn>
        </span>
      </div>
      {busy && <p className="mt-1 text-sky-300">{busy}</p>}
      {error && <p className="mt-1 break-words text-red-400">{error}</p>}
      {note && <p className="mt-1 text-emerald-400">{note}</p>}
      {browse && <CloudBrowser account={browse} action="Download this folder" onPick={(path) => sync(browse, path)} onClose={() => setBrowse(null)} />}
    </section>
  );
}

/** Browse the folders of a cloud account and choose one. */
export function CloudBrowser({ account, action, onPick, onClose }: { account: CloudAccount; action: string; onPick: (path: string) => void; onClose: () => void }) {
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<CloudEntry[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setEntries(null);
    setError("");
    api.cloudBrowse(account.id, path).then(
      (e) => live && setEntries(e),
      (e) => live && setError(String(e)),
    );
    return () => void (live = false);
  }, [account.id, path]);
  const parts = path.split("/").filter(Boolean);
  const dirs = (entries ?? []).filter((e) => e.dir).sort((a, b) => a.name.localeCompare(b.name));
  const photos = (entries ?? []).filter((e) => !e.dir && PHOTO.test(e.name));
  const crumb = (label: string, to: string) => (
    <button key={to} className="rounded px-1 text-sky-400 hover:bg-neutral-800 hover:text-sky-300" onClick={() => setPath(to)}>
      {label}
    </button>
  );
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div className="flex max-h-[80vh] w-[520px] max-w-[calc(100vw-2rem)] flex-col gap-3 rounded-lg border border-neutral-700 bg-neutral-900 p-5 text-[12px] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-sm font-semibold">Choose a folder in {account.name}</h2>
        <div className="flex flex-wrap items-center gap-0.5 text-[11px] text-neutral-500">
          {crumb(account.name, "")}
          {parts.map((s, i) => (
            <span key={i} className="flex items-center gap-0.5">
              ›{crumb(s, parts.slice(0, i + 1).join("/"))}
            </span>
          ))}
        </div>
        <div className="min-h-[12rem] flex-1 overflow-y-auto rounded bg-neutral-950 p-1">
          {error && <p className="break-words p-2 text-red-400">{error}</p>}
          {!entries && !error && (
            <div className="flex flex-col items-center justify-center py-10 text-neutral-400 gap-2">
              <svg className="h-5 w-5 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" />
                <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              <span className="text-[11px] font-medium text-neutral-300">Scanning remote directory…</span>
              <span className="text-[10px] text-neutral-500 font-mono">Fetching folder metadata</span>
            </div>
          )}
          {dirs.map((d) => (
            <button key={d.path} className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-neutral-800" onClick={() => setPath(d.path)}>
              <span className="text-amber-300">▸</span>
              <span className="truncate">{d.name}</span>
            </button>
          ))}
          {entries && !dirs.length && <p className="p-2 text-neutral-500">No folders inside.</p>}
        </div>
        <p className={hint}>
          {!entries
            ? ""
            : photos.length
              ? `${photos.length} photo(s) here (${size(photos.reduce((n, e) => n + e.size, 0))})${dirs.length ? `, and ${dirs.length} folder(s) inside` : ""}.`
              : dirs.length
                ? `No photos directly in this folder; ${dirs.length} folder(s) inside.`
                : "No photos here."}
          {entries && dirs.length > 0 ? " Subfolders come along." : ""}
        </p>
        <div className="flex justify-end gap-2">
          <Btn onClick={onClose}>Cancel</Btn>
          <Btn active disabled={!entries} onClick={() => onPick(path)}>
            {action}
          </Btn>
        </div>
      </div>
    </div>
  );
}

function Services({ settings, onConnect }: { settings: Settings; onConnect: (s: Source) => void }) {
  const rows: [Source, string, string, boolean][] = [
    ["immich", "Immich", settings.immich_url ? `Connected to ${settings.immich_url}` : "Browse albums in place; edits are saved back as stacks.", !!settings.immich_url],
    ["google", "Google Photos", settings.google_client_id ? "Signed in" : "Pick photos to download; edits are uploaded as new photos.", !!settings.google_client_id],
    [
      "apple",
      "Apple iCloud Photos",
      settings.apple_photos_path ? `Connected to ${settings.apple_photos_path}` : "Index your Apple Photos library or iCloud sync folder; export edits directly back.",
      !!settings.apple_photos_path,
    ],
    [
      "amazon",
      "Amazon Photos",
      settings.amazon_photos_path ? `Connected to ${settings.amazon_photos_path}` : "Sync full-resolution photos and RAW files from Amazon Photos / Amazon Drive.",
      !!settings.amazon_photos_path,
    ],
    [
      "adobe",
      "Adobe Cloud Photos",
      settings.adobe_cloud_path ? `Connected to ${settings.adobe_cloud_path}` : "Sync with Adobe Creative Cloud Files & Lightroom photography library.",
      !!settings.adobe_cloud_path,
    ],
  ];
  return (
    <section className={card}>
      <h2 className="mb-1 font-semibold">Photo services</h2>
      {rows.map(([source, title, detail, on]) => {
        return (
          <div key={source} className="flex items-center gap-3 border-t border-neutral-800 py-2 first-of-type:border-t-0">
            <span className={`h-2 w-2 shrink-0 rounded-full ${on ? "bg-emerald-500" : "bg-neutral-600"}`} />
            <div className="min-w-0 flex-1">
              <div className="text-neutral-200">{title}</div>
              <div className="truncate text-[11px] text-neutral-500" title={detail}>
                {detail}
              </div>
            </div>
            <Btn onClick={() => onConnect(source)}>{on ? "Manage…" : "Connect…"}</Btn>
          </div>
        );
      })}
    </section>
  );
}

function Automation({ settings, setSettings, onAuto }: { settings: Settings; setSettings: (s: Settings) => void; onAuto: () => void }) {
  const o = settings.auto.on_import;
  const set = (patch: Partial<OnImport>) => {
    const next = { ...settings, auto: { ...settings.auto, on_import: { ...o, ...patch } } };
    setSettings(next);
    api.saveSettings(next);
  };
  const sources: [keyof OnImport, string][] = [
    ["folders", "Folders and network drives"],
    ["cloud", "Cloud storage and Google Photos picks"],
    ["immich", "Immich albums (fetches each new photo's preview)"],
    ["duplicates", "Copies made by Duplicate"],
  ];
  return (
    <section className={card}>
      <h2 className="font-semibold">Automation</h2>
      <label className="mt-2 flex items-start gap-2">
        <input type="checkbox" className="mt-0.5" checked={o.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
        <span>
          <span className="text-neutral-200">Auto Catalog & Culling of new photos</span>
          <span className={`block ${hint}`}>In the background, photos new to the catalog get stars, the best of each burst a pick, blurred or badly exposed ones a reject, and catalog keywords.</span>
        </span>
      </label>
      <div className={`ml-6 mt-2 space-y-1 ${o.enabled ? "" : "opacity-40"}`}>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">New photos from</div>
        {sources.map(([k, label]) => (
          <label key={k} className="flex items-center gap-2">
            <input type="checkbox" disabled={!o.enabled} checked={o[k]} onChange={(e) => set({ [k]: e.target.checked })} />
            {label}
          </label>
        ))}
      </div>
      <p className={`${hint} mt-3`}>
        It culls with the settings of Auto › Catalog & Culling, where you can also run it on photos already in the catalog.{" "}
        <button className="text-sky-400 hover:text-sky-300" onClick={onAuto}>
          Open Auto…
        </button>
      </p>
    </section>
  );
}

// --- Duplicate ---

type Kind = "folder" | "drive" | "photos" | "immich" | "google" | "apple" | "amazon" | "adobe" | "cloud";
type Draft = { kind: Kind; path: string; drive: string; album: string; account: string; cloudPath: string };
const EMPTY: Draft = { kind: "folder", path: "", drive: "", album: "favorites", account: "", cloudPath: "" };

/** Copy photos from one place to another: folders, network drives, catalog photos, Immich, Google Photos, cloud storage. */
export function Duplicate({ settings, selected, jobs }: { settings: Settings; selected: number[]; jobs: Record<string, JobProgress> }) {
  const [from, setFrom] = useState<Draft>(EMPTY);
  const [to, setTo] = useState<Draft>(EMPTY);
  const [o, setO] = useState<DuplicateOptions & { cull: boolean }>({
    subfolders: true,
    only_picks: false,
    min_rating: 0,
    skip_existing: true,
    sidecars: true,
    verify: false,
    keep_dates: true,
    catalog: true,
    cull: settings.auto.on_import.enabled && settings.auto.on_import.duplicates,
  });
  const [albums, setAlbums] = useState<ImmichAlbum[]>([]);
  const [job, setJob] = useState("");
  const [result, setResult] = useState<DuplicateResult | null>(null);
  const [error, setError] = useState("");
  const [browse, setBrowse] = useState<"from" | "to" | null>(null);
  useEffect(() => {
    if (settings.immich_url) api.immichAlbums().then(setAlbums, () => setAlbums([]));
  }, [settings.immich_url]);

  // a network drive is connected first; then it is a folder
  const place = async (d: Draft): Promise<Place> => {
    switch (d.kind) {
      case "folder":
        return { kind: "folder", path: d.path.trim() };
      case "drive":
        return { kind: "folder", path: await api.networkConnect(d.drive) };
      case "photos":
        return { kind: "photos", ids: selected };
      case "immich":
        return { kind: "immich", album: d.album === "favorites" ? null : d.album };
      case "google":
        return { kind: "google" };
      case "apple":
        return { kind: "folder", path: settings.apple_photos_path || "" };
      case "amazon":
        return { kind: "folder", path: settings.amazon_photos_path || "" };
      case "adobe":
        return { kind: "folder", path: settings.adobe_cloud_path || "" };
      case "cloud":
        return { kind: "cloud", account: d.account, path: d.cloudPath.trim() };
    }
  };
  const complete = (d: Draft) =>
    d.kind === "folder" ? !!d.path.trim() : d.kind === "drive" ? !!d.drive : d.kind === "cloud" ? !!d.account : d.kind === "photos" ? selected.length > 0 : true;
  const toFolder = to.kind === "folder" || to.kind === "drive";
  const run = async () => {
    const id = `duplicate-${Date.now()}`;
    setJob(id);
    setResult(null);
    setError("");
    try {
      setResult(await api.duplicate(await place(from), await place(to), o, id));
    } catch (e) {
      setError(String(e));
    }
    setJob("");
  };
  const progress = job ? jobs[job] : undefined;
  const check = (k: keyof DuplicateOptions, label: string, detail?: string, disabled?: boolean) => (
    <label className={`flex items-start gap-2 py-0.5 ${disabled ? "opacity-40" : ""}`}>
      <input type="checkbox" className="mt-0.5" disabled={disabled} checked={!!o[k]} onChange={(e) => setO({ ...o, [k]: e.target.checked })} />
      <span>
        <span className="text-neutral-200">{label}</span>
        {detail && <span className={`block ${hint}`}>{detail}</span>}
      </span>
    </label>
  );
  const accounts = settings.clouds;
  const side = (title: string, d: Draft, set: (d: Draft) => void, which: "from" | "to") => {
    const kinds: [Kind, string, boolean][] = [
      ["folder", "Folder", true],
      ["drive", "Network drive", settings.network_drives.length > 0],
      ...(which === "from" ? ([["photos", `Selected photos (${selected.length})`, selected.length > 0]] as [Kind, string, boolean][]) : []),
      ["immich", which === "from" ? "Immich album" : "Immich", !!settings.immich_url],
      ["google", which === "from" ? "Google Photos picks" : "Google Photos", !!settings.google_client_id],
      ["apple", which === "from" ? "Apple iCloud Photos" : "Apple Photos", !!settings.apple_photos_path],
      ["amazon", which === "from" ? "Amazon Photos" : "Amazon Photos", !!settings.amazon_photos_path],
      ["adobe", which === "from" ? "Adobe Cloud Photos" : "Adobe Creative Cloud", !!settings.adobe_cloud_path],
      ["cloud", "Cloud storage", accounts.length > 0],
    ];
    const patch = (x: Partial<Draft>) => set({ ...d, ...x });
    return (
      <section className={`${card} min-w-[280px] flex-1`}>
        <h3 className="mb-2 font-semibold">{title}</h3>
        <select
          className={`${field} mb-2 w-full`}
          value={d.kind}
          onChange={(e) => {
            const kind = e.target.value as Kind;
            patch({ kind, drive: d.drive || settings.network_drives[0]?.id || "", account: d.account || accounts[0]?.id || "" });
          }}
        >
          {kinds.map(([k, label, ok]) => (
            <option key={k} value={k} disabled={!ok}>
              {label}
              {ok ? "" : k === "photos" ? " — select photos in the Library" : " — add it in Locations"}
            </option>
          ))}
        </select>
        {d.kind === "folder" && (
          <div className="flex gap-1">
            <input className={`${field} min-w-0 flex-1 font-mono text-[11px]`} value={d.path} placeholder="A folder or a drive" onChange={(e) => patch({ path: e.target.value })} />
            <Btn
              onClick={async () => {
                const dir = await open({ directory: true, defaultPath: d.path || undefined, title });
                if (typeof dir === "string") patch({ path: dir });
              }}
            >
              Choose…
            </Btn>
          </div>
        )}
        {d.kind === "drive" && (
          <select className={`${field} w-full`} value={d.drive} onChange={(e) => patch({ drive: e.target.value })}>
            {settings.network_drives.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} — {x.path}
              </option>
            ))}
          </select>
        )}
        {d.kind === "photos" && <p className="text-neutral-400">The files of the {selected.length} photo(s) selected in the Library, with their culling and edits.</p>}
        {d.kind === "immich" &&
          (which === "from" ? (
            <select className={`${field} w-full`} value={d.album} onChange={(e) => patch({ album: e.target.value })}>
              <option value="favorites">★ Favorites</option>
              {albums.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} ({a.count})
                </option>
              ))}
            </select>
          ) : (
            <p className="text-neutral-400">Uploaded to your Immich library; files it has already are skipped.</p>
          ))}
        {d.kind === "google" && (
          <p className="text-neutral-400">{which === "from" ? "The photos you picked from Google Photos." : "Uploaded to your Google Photos library as new photos."}</p>
        )}
        {d.kind === "cloud" && (
          <div className="space-y-1">
            <select className={`${field} w-full`} value={d.account} onChange={(e) => patch({ account: e.target.value, cloudPath: "" })}>
              {accounts.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} ({KIND_NAMES[a.kind] ?? a.kind})
                </option>
              ))}
            </select>
            <div className="flex gap-1">
              <input className={`${field} min-w-0 flex-1 font-mono text-[11px]`} value={d.cloudPath} placeholder="Folder (empty: the top)" onChange={(e) => patch({ cloudPath: e.target.value })} />
              <Btn disabled={!d.account} onClick={() => setBrowse(which)}>
                Browse…
              </Btn>
            </div>
          </div>
        )}
      </section>
    );
  };
  const browsing = browse && accounts.find((a) => a.id === (browse === "from" ? from : to).account);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 text-[12px]">
      <p className="max-w-4xl text-neutral-400">
        Copy photos from one place to another — between folders and network drives, or to and from Immich, Google Photos and cloud storage. Nothing is overwritten or deleted: a file the
        destination has already is skipped, a different one with the same name is kept beside it.
      </p>
      <div className="flex max-w-4xl flex-wrap items-stretch gap-3">
        {side("From", from, setFrom, "from")}
        <div className="flex items-center text-lg text-neutral-500">→</div>
        {side("To", to, setTo, "to")}
      </div>
      <section className={`${card} grid max-w-4xl grid-cols-1 gap-x-6 sm:grid-cols-2`}>
        <div>
          <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-neutral-500">What</h3>
          {check("subfolders", "Include subfolders")}
          {check("only_picks", "Only picks")}
          <label className="flex items-center gap-2 py-0.5">
            <span className="text-neutral-200">Rated at least</span>
            <select className={field} value={o.min_rating} onChange={(e) => setO({ ...o, min_rating: +e.target.value })}>
              {[0, 1, 2, 3, 4, 5].map((r) => (
                <option key={r} value={r}>
                  {r ? "★".repeat(r) : "any"}
                </option>
              ))}
            </select>
          </label>
          <p className={hint}>Picks and ratings come from the catalog (Immich: favorites and its ratings).</p>
          {check("skip_existing", "Skip files the destination has", "Same name and size (and date, when dates are kept)")}
        </div>
        <div>
          <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-neutral-500">How</h3>
          {check("sidecars", "Bring the edits along", "As .olxmp sidecars next to the copies", !(toFolder || to.kind === "cloud"))}
          {check("keep_dates", "Keep the files' dates", undefined, !toFolder)}
          {check("verify", "Verify each copy", "Read it back and compare checksums", !toFolder)}
          {check("catalog", "Add the copies to the catalog", "With the originals' ratings, flags, labels and keywords", !toFolder)}
          {check("cull", "Then run Auto Catalog & Culling on them", "With the settings of Auto › Catalog & Culling", !toFolder || !o.catalog)}
        </div>
      </section>
      <div className="flex max-w-4xl items-center gap-3">
        <Btn active className="px-4 py-1.5 text-[12px] font-medium" disabled={!!job || !complete(from) || !complete(to)} onClick={run}>
          {job ? "Copying…" : "Duplicate"}
        </Btn>
        {job && (
          <>
            <div className="min-w-0 flex-1">
              <div className="mb-0.5 flex justify-between text-[10px] text-neutral-400">
                <span>{progress?.label ?? "Starting"}…</span>
                {progress && progress.total > 0 && (
                  <span className="tabular-nums">
                    {progress.done} / {progress.total}
                  </span>
                )}
              </div>
              <div className="relative h-1 overflow-hidden rounded bg-neutral-800">
                {progress && progress.total > 0 ? (
                  <div className="h-full bg-sky-500 transition-all" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
                ) : (
                  <div className="animate-indeterminate absolute inset-y-0 w-1/3 rounded bg-sky-500" />
                )}
              </div>
            </div>
            <Btn onClick={() => api.cancelJob(job)} title="Stop after the current file; what is copied stays">
              Cancel
            </Btn>
          </>
        )}
        {error && <span className="break-words text-red-400">{error}</span>}
      </div>
      {result && (
        <section className={`${card} max-w-4xl space-y-1`}>
          <h3 className="font-semibold">
            {result.cancelled ? "Stopped" : "Done"}: {result.copied} copied ({size(result.bytes)}), {result.skipped} already there
            {result.failed ? `, ${result.failed} failed` : ""}
          </h3>
          <p className="text-neutral-400">
            {result.selected} of {result.listed} photo(s) chosen.
            {result.cataloged ? ` ${result.cataloged} added to the catalog.` : ""}
            {result.culled ? ` Culled: ${result.culled.picks} picks, ${result.culled.rejects} rejects, ${result.culled.rated} rated.` : ""}
          </p>
          {result.errors.map((e) => (
            <p key={e} className="break-words text-[11px] text-red-400">
              {e}
            </p>
          ))}
        </section>
      )}
      {browse && browsing && (
        <CloudBrowser
          account={browsing}
          action={browse === "from" ? "Copy from this folder" : "Copy into this folder"}
          onPick={(path) => {
            (browse === "from" ? setFrom : setTo)((d) => ({ ...d, cloudPath: path }));
            setBrowse(null);
          }}
          onClose={() => setBrowse(null)}
        />
      )}
    </div>
  );
}
