import { useState } from "react";
import { api, AppInfo, Provider, Settings } from "./api";
import { Btn, Seg } from "./controls";

export type Log = { t: string; source: string; message: string };

const KINDS: [string, string][] = [
  ["cli", "CLI agent (uses MCP)"],
  ["ollama", "Ollama /api/chat"],
  ["openai", "OpenAI-compatible /v1"],
  ["anthropic", "Anthropic Messages API"],
  ["gemini", "Google Gemini API"],
];

const card = "rounded-lg border border-neutral-800 bg-neutral-900 p-4";
const field = "w-full rounded bg-neutral-800 px-2 py-1 font-mono text-[11px] outline-none focus:ring-1 focus:ring-sky-600";

export function Agent({ info, settings, setSettings, logs, clearLogs }: { info: AppInfo | null; settings: Settings | null; setSettings: (s: Settings) => void; logs: Log[]; clearLogs: () => void }) {
  const [reveal, setReveal] = useState(false);
  const [client, setClient] = useState<"code" | "desktop" | "json">("code");
  const [sel, setSel] = useState(0);
  const [key, setKey] = useState("");
  const [note, setNote] = useState("");
  if (!settings || !info) return null;

  const save = async (s: Settings, msg = "Saved") => {
    setSettings(s);
    await api.saveSettings(s);
    setNote(msg);
    setTimeout(() => setNote(""), 2500);
  };
  const url = `http://127.0.0.1:${settings.mcp_port}/mcp`;
  const token = settings.mcp_token;
  const snippets = {
    code: `claude mcp add --transport http optixel ${url} --header "Authorization: Bearer ${token}"`,
    desktop: JSON.stringify({ mcpServers: { optixel: { command: info.exe ?? "optixel-studio", args: ["--mcp-stdio"] } } }, null, 2),
    json: JSON.stringify({ mcpServers: { optixel: { type: "http", url, headers: { Authorization: `Bearer ${token}` } } } }, null, 2),
  };
  const prov: Provider | undefined = settings.providers[sel];
  const setProv = (patch: Partial<Provider>) => setSettings({ ...settings, providers: settings.providers.map((p, i) => (i === sel ? { ...p, ...patch } : p)) });

  return (
    <div className="flex min-h-0 flex-1 gap-4 overflow-y-auto p-4 text-[12px]">
      <div className="flex w-[520px] shrink-0 flex-col gap-4">
        <section className={card}>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold">MCP server</h2>
            <span className={`rounded px-2 py-0.5 text-[11px] ${info.mcp_status === "running" ? "bg-emerald-900 text-emerald-300" : "bg-neutral-800 text-neutral-400"}`}>{info.mcp_status}</span>
          </div>
          <div className="grid grid-cols-[90px_1fr] items-center gap-2">
            <span className="text-neutral-400">Endpoint</span>
            <code className="select-text text-sky-300">{url}</code>
            <span className="text-neutral-400">Bearer token</span>
            <div className="flex items-center gap-1">
              <code className="min-w-0 flex-1 truncate select-text">{reveal ? token : `${token.slice(0, 6)}••••••••••••`}</code>
              <Btn onClick={() => setReveal(!reveal)}>{reveal ? "Hide" : "Show"}</Btn>
              <Btn onClick={() => navigator.clipboard.writeText(token)}>Copy</Btn>
              <Btn onClick={async () => setSettings({ ...settings, mcp_token: await api.regenerateToken() })} title="Existing clients must be reconfigured">
                New
              </Btn>
            </div>
            <span className="text-neutral-400">Transport</span>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={settings.mcp_enabled} onChange={(e) => setSettings({ ...settings, mcp_enabled: e.target.checked })} />
                Enabled
              </label>
              <label className="flex items-center gap-1 text-neutral-400">
                Port
                <input className="w-20 rounded bg-neutral-800 px-2 py-1" value={settings.mcp_port} onChange={(e) => setSettings({ ...settings, mcp_port: +e.target.value.replace(/\D/g, "") || 41920 })} />
              </label>
              <Btn onClick={() => save(settings, "Saved — restart Optixel to apply transport changes")}>Save</Btn>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-neutral-500">
            Streamable HTTP on loopback only, protected by the token. stdio: run <code>optixel-studio --mcp-stdio</code> — it attaches to this window when open, else edits the catalog headless.
            Every tool call is one undoable history step.
          </p>
        </section>

        <section className={card}>
          <h2 className="mb-3 font-semibold">Connect an agent</h2>
          <Seg options={[["code", "Claude Code"], ["desktop", "Claude Desktop (stdio)"], ["json", "Generic HTTP"]]} value={client} onChange={setClient} />
          <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-all rounded bg-neutral-950 p-3 font-mono text-[11px] text-neutral-300 select-text">{reveal ? snippets[client] : snippets[client].replaceAll(token, "<token>")}</pre>
          <Btn className="mt-2" onClick={() => navigator.clipboard.writeText(snippets[client])}>
            Copy with token
          </Btn>
          <p className="mt-2 text-[11px] text-neutral-500">Tools: catalog_query, culling_batch_update, develop_apply_parameters, develop_auto, auto_run, mask_create_and_edit, retouch_remove_objects, ai_reconstruct_and_retouch, batch_export, photo_inspect. Resources: optixel://catalog/summary, selection/active, photo/{"{id}"}/state|telemetry|preview.</p>
        </section>

        <section className={`${card} flex min-h-64 flex-1 flex-col`}>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="font-semibold">Activity</h2>
            <Btn onClick={clearLogs}>Clear</Btn>
          </div>
          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto rounded bg-neutral-950 p-2 font-mono text-[11px]">
            {logs.length === 0 && <p className="text-neutral-600">MCP tool calls and AI runs appear here.</p>}
            {logs.map((l, i) => (
              <div key={i} className="select-text whitespace-pre-wrap break-all">
                <span className="text-neutral-600">{l.t}</span> <span className={l.source === "mcp" ? "text-sky-400" : "text-violet-300"}>{l.source}</span> <span className="text-neutral-300">{l.message}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className={`${card} flex min-w-0 flex-1 flex-col`}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold">AI & agent router</h2>
          <span className="text-[11px] text-emerald-400">{note}</span>
        </div>
        <div className="flex min-h-0 flex-1 gap-4">
          <div className="w-52 shrink-0 space-y-1">
            {settings.providers.map((p, i) => (
              <div key={p.id + i} onClick={() => (setSel(i), setKey(""))} className={`cursor-pointer rounded px-2 py-1.5 ${i === sel ? "bg-neutral-700" : "hover:bg-neutral-800"}`}>
                <div className="flex items-center justify-between">
                  <span className="truncate">{p.name}</span>
                  {settings.active_provider === p.id && <span className="text-[10px] text-violet-300">active</span>}
                </div>
                <div className="truncate font-mono text-[10px] text-neutral-500">{p.id}</div>
              </div>
            ))}
            <Btn
              className="mt-2 w-full"
              onClick={() => {
                const n = settings.providers.length + 1;
                setSettings({ ...settings, providers: [...settings.providers, { id: `cli:custom-${n}`, kind: "cli", name: `Custom agent ${n}`, command: "", endpoint: "", model: "", has_key: false, timeout_s: 600, max_passes: 2 }] });
                setSel(settings.providers.length);
              }}
            >
              + Add provider
            </Btn>
          </div>
          {prov && (
            <div className="min-w-0 flex-1 space-y-2">
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <span className="text-neutral-400">Name</span>
                <input className={field} value={prov.name} onChange={(e) => setProv({ name: e.target.value })} />
                <span className="text-neutral-400">Provider id</span>
                <input className={field} value={prov.id} onChange={(e) => setProv({ id: e.target.value })} />
                <span className="text-neutral-400">Type</span>
                <select className={field} value={prov.kind} onChange={(e) => setProv({ kind: e.target.value })}>
                  {KINDS.map(([k, l]) => (
                    <option key={k} value={k}>
                      {l}
                    </option>
                  ))}
                </select>
                {prov.kind === "cli" ? (
                  <>
                    <span className="self-start pt-1 text-neutral-400">Command</span>
                    <div>
                      <textarea className={`${field} h-24`} value={prov.command} onChange={(e) => setProv({ command: e.target.value })} placeholder="my-agent --image {{IMAGE_PATH}} --prompt {{USER_PROMPT}}" />
                      <p className="mt-1 text-[10px] text-neutral-500">
                        Split like a shell, then each {"{{VAR}}"} is substituted — no shell runs. Vars: AGENT_PROMPT, USER_PROMPT, IMAGE_PATH, TELEMETRY_PATH, HISTOGRAM_JSON, CURRENT_XMP_JSON, MCP_CONFIG_PATH, MCP_SSE_URL, MCP_TOKEN, PHOTO_ID.
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <span className="text-neutral-400">Endpoint</span>
                    <input className={field} value={prov.endpoint} onChange={(e) => setProv({ endpoint: e.target.value })} />
                    <span className="text-neutral-400">Model</span>
                    <input className={field} value={prov.model} onChange={(e) => setProv({ model: e.target.value })} />
                    <span className="text-neutral-400">Visual passes</span>
                    <input className={field} value={prov.max_passes} onChange={(e) => setProv({ max_passes: +e.target.value.replace(/\D/g, "") || 1 })} />
                    <span className="text-neutral-400">API key</span>
                    <div className="flex gap-1">
                      <input className={field} type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={prov.has_key ? "stored in OS keychain" : "optional for local servers"} />
                      <Btn
                        onClick={async () => {
                          await api.setKey(prov.id, key);
                          setProv({ has_key: !!key });
                          setKey("");
                          setNote(key ? "Key stored in the OS keychain" : "Key removed");
                        }}
                      >
                        {key ? "Store" : "Clear"}
                      </Btn>
                    </div>
                  </>
                )}
                <span className="text-neutral-400">Timeout (s)</span>
                <input className={field} value={prov.timeout_s} onChange={(e) => setProv({ timeout_s: +e.target.value.replace(/\D/g, "") || 60 })} />
              </div>
              <div className="flex gap-2 pt-2">
                <Btn active onClick={() => save(settings)}>
                  Save
                </Btn>
                <Btn onClick={() => save({ ...settings, active_provider: prov.id }, `${prov.name} is now the default`)}>Use in Develop</Btn>
                <Btn onClick={() => (save({ ...settings, providers: settings.providers.filter((_, i) => i !== sel) }, "Removed"), setSel(0))}>Remove</Btn>
              </div>
              <p className="pt-2 text-[11px] text-neutral-500">
                {prov.kind === "cli"
                  ? "CLI agents receive a rendered preview, telemetry and an MCP config for this app, then edit through MCP tools — the Develop view updates live."
                  : "HTTP providers get the preview image, histogram and edit state, and must answer with an AiTuningResponse JSON that Optixel validates, clamps and applies (with an optional verification pass)."}
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
