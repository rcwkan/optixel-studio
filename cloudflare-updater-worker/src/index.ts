/**
 * Optixel Studio Authenticated Update Entitlement Server & R2 Distribution Gateway
 * Cloudflare Worker (Edge TypeScript)
 */

export interface Env {
  LICENSES_KV: KVNamespace;
  RELEASES_BUCKET: R2Bucket;
  PURCHASE_URL: string;
  APP_NAME: string;
  OFFICIAL_ORIGIN: string;
  MIN_CLIENT_VERSION?: string;
  LEMON_SQUEEZY_WEBHOOK_SECRET?: string;
  POLAR_WEBHOOK_SECRET?: string;
  DOWNLOAD_SIGNING_SECRET?: string;
}

interface LicenseRecord {
  key: string;
  tier: "pro" | "studio" | "lifetime";
  email?: string;
  created_at: string;
  expires_at?: string; // ISO 8601 string or undefined for lifetime
  active: boolean;
}

interface ReleasePlatformManifest {
  signature: string;
  url: string;
}

interface ReleaseManifest {
  version: string;
  notes: string;
  pub_date: string;
  platforms: Record<string, ReleasePlatformManifest>;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS headers for desktop client & web preflight
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Optixel-License-Key",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders, status: 204 });
    }

    try {
      // 1. Root / Health Check
      if (path === "/" || path === "/health") {
        return jsonResponse(
          {
            service: "Optixel Studio Update Entitlement Server",
            status: "online",
            edge_region: request.cf?.colo || "global",
            timestamp: new Date().toISOString(),
          },
          200,
          corsHeaders
        );
      }

      // 2. Tauri v2 Update Check: GET /v1/update/:target/:current_version
      if (path.startsWith("/v1/update/")) {
        return handleUpdateCheck(request, url, env, corsHeaders);
      }

      // 3. Authenticated Asset Download: GET /v1/download/:asset
      if (path.startsWith("/v1/download/")) {
        return handleAssetDownload(request, url, env);
      }

      // 4. License Verification Endpoint: GET /v1/license/verify?key=...
      if (path === "/v1/license/verify") {
        return handleLicenseVerify(request, url, env, corsHeaders);
      }

      // 5. Webhooks for Merchant of Record
      if (path === "/v1/webhook/lemonsqueezy" && request.method === "POST") {
        return handleLemonSqueezyWebhook(request, env);
      }
      if (path === "/v1/webhook/polar" && request.method === "POST") {
        return handlePolarWebhook(request, env);
      }

      return jsonResponse({ error: "not_found", message: "Endpoint not found" }, 404, corsHeaders);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return jsonResponse({ error: "server_error", message: msg }, 500, corsHeaders);
    }
  },
};

// -----------------------------------------------------------------------------
// Endpoint Handlers
// -----------------------------------------------------------------------------

async function handleUpdateCheck(
  request: Request,
  url: URL,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const parts = url.pathname.replace(/^\/v1\/update\//, "").split("/");
  if (parts.length < 2) {
    return jsonResponse({ error: "bad_request", message: "Usage: /v1/update/:target/:current_version" }, 400, corsHeaders);
  }

  const target = parts[0]; // e.g. "darwin-aarch64", "windows-x86_64"
  const currentVersion = parts[1]; // e.g. "0.3.0"

  // 1. Check License Key entitlement
  const licenseKey = extractLicenseKey(request, url);
  const entitlement = await verifyLicenseEntitlement(licenseKey, env);

  if (!entitlement.valid) {
    return jsonResponse(
      {
        error: "payment_required",
        message: entitlement.message || "Active updates require a valid Optixel Studio license.",
        purchase_url: env.PURCHASE_URL || "https://optixel.makecade.com/#pricing",
      },
      402,
      corsHeaders
    );
  }

  // 2. Load latest release manifest from R2 (or fallback to KV cache)
  let manifest: ReleaseManifest | null = null;
  const manifestObj = await env.RELEASES_BUCKET.get("manifests/latest.json");
  if (manifestObj) {
    manifest = await manifestObj.json<ReleaseManifest>();
  } else {
    // Check KV fallback
    const rawKV = await env.LICENSES_KV.get("release:latest");
    if (rawKV) {
      manifest = JSON.parse(rawKV) as ReleaseManifest;
    }
  }

  if (!manifest) {
    // No release currently published in R2; report up to date
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  // 3. Version comparison
  if (!isNewer(manifest.version, currentVersion)) {
    // Current client version is up-to-date
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  const platformData = manifest.platforms[target];
  if (!platformData) {
    return jsonResponse(
      { error: "platform_not_found", message: `No update package available for target architecture: ${target}` },
      404,
      corsHeaders
    );
  }

  // 4. Generate signed or tokenized download URL
  const downloadUrl = generateDownloadUrl(url.origin, platformData.url, licenseKey);

  // Tauri v2 updater format response
  return jsonResponse(
    {
      version: manifest.version,
      notes: manifest.notes,
      pub_date: manifest.pub_date,
      signature: platformData.signature,
      url: downloadUrl,
    },
    200,
    corsHeaders
  );
}

async function handleAssetDownload(request: Request, url: URL, env: Env): Promise<Response> {
  const assetName = url.pathname.replace(/^\/v1\/download\//, "");
  if (!assetName || assetName.includes("..")) {
    return new Response("Invalid asset request", { status: 400 });
  }

  // Verify license key or token
  const licenseKey = extractLicenseKey(request, url);
  const entitlement = await verifyLicenseEntitlement(licenseKey, env);
  if (!entitlement.valid) {
    return new Response("Forbidden: Valid license key required to download update package.", { status: 403 });
  }

  const object = await env.RELEASES_BUCKET.get(`packages/${assetName}`);
  if (!object) {
    return new Response("Release package not found on storage server", { status: 404 });
  }

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("etag", object.httpEtag);
  headers.set("Content-Disposition", `attachment; filename="${assetName}"`);
  headers.set("Cache-Control", "public, max-age=86400");

  return new Response(object.body, { headers });
}

async function handleLicenseVerify(
  request: Request,
  url: URL,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const licenseKey = extractLicenseKey(request, url);
  if (!licenseKey) {
    return jsonResponse(
      {
        key: "",
        valid: false,
        tier: "none",
        message: "No license key provided.",
        purchase_url: env.PURCHASE_URL,
      },
      200,
      corsHeaders
    );
  }

  const entitlement = await verifyLicenseEntitlement(licenseKey, env);
  return jsonResponse(
    {
      key: licenseKey,
      valid: entitlement.valid,
      tier: entitlement.tier || "pro",
      email: entitlement.email,
      expires_at: entitlement.expires_at,
      message: entitlement.message,
      purchase_url: entitlement.valid ? undefined : env.PURCHASE_URL,
    },
    200,
    corsHeaders
  );
}

// -----------------------------------------------------------------------------
// Webhook Handlers
// -----------------------------------------------------------------------------

async function handleLemonSqueezyWebhook(request: Request, env: Env): Promise<Response> {
  const secret = env.LEMON_SQUEEZY_WEBHOOK_SECRET;
  const rawBody = await request.text();

  if (secret) {
    const signature = request.headers.get("X-Signature") || "";
    const isValid = await verifyHmacSha256(secret, rawBody, signature);
    if (!isValid) {
      return new Response("Invalid HMAC signature", { status: 401 });
    }
  }

  const payload = JSON.parse(rawBody);
  const eventName = payload.meta?.event_name;
  const customData = payload.meta?.custom_data;
  const data = payload.data?.attributes;

  if (eventName === "order_created" || eventName === "subscription_created") {
    const key = data?.license_key || `OLS-PRO-${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
    const email = data?.user_email || customData?.email;

    const record: LicenseRecord = {
      key,
      tier: "pro",
      email,
      created_at: new Date().toISOString(),
      expires_at: eventName === "order_created" ? undefined : new Date(Date.now() + 365 * 86400000).toISOString(),
      active: true,
    };

    await env.LICENSES_KV.put(`license:${key}`, JSON.stringify(record));
    return new Response(JSON.stringify({ status: "provisioned", key }), { status: 200 });
  }

  if (eventName === "subscription_cancelled" || eventName === "subscription_expired") {
    const key = data?.license_key;
    if (key) {
      const existing = await env.LICENSES_KV.get(`license:${key}`);
      if (existing) {
        const parsed = JSON.parse(existing) as LicenseRecord;
        parsed.active = false;
        await env.LICENSES_KV.put(`license:${key}`, JSON.stringify(parsed));
      }
    }
    return new Response("Processed cancellation", { status: 200 });
  }

  return new Response("Ignored event", { status: 200 });
}

async function handlePolarWebhook(request: Request, env: Env): Promise<Response> {
  const rawBody = await request.text();
  const payload = JSON.parse(rawBody);

  if (payload.type === "order.created") {
    const key = `OLS-PRO-${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
    const email = payload.data?.customer?.email;

    const record: LicenseRecord = {
      key,
      tier: "pro",
      email,
      created_at: new Date().toISOString(),
      active: true,
    };

    await env.LICENSES_KV.put(`license:${key}`, JSON.stringify(record));
    return new Response(JSON.stringify({ status: "provisioned", key }), { status: 200 });
  }

  return new Response("OK", { status: 200 });
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function extractLicenseKey(request: Request, url: URL): string | null {
  const authHeader = request.headers.get("Authorization");
  if (authHeader && authHeader.startsWith("Bearer ")) {
    return authHeader.slice(7).trim();
  }

  const customHeader = request.headers.get("X-Optixel-License-Key");
  if (customHeader) {
    return customHeader.trim();
  }

  const queryKey = url.searchParams.get("key");
  if (queryKey) {
    return queryKey.trim();
  }

  return null;
}

async function verifyLicenseEntitlement(
  key: string | null,
  env: Env
): Promise<{ valid: boolean; tier?: string; email?: string; expires_at?: string; message: string }> {
  if (!key) {
    return {
      valid: false,
      message: "No license key provided. Auto-updates require a paid Optixel Studio license.",
    };
  }

  // 1. Check in KV database
  const raw = await env.LICENSES_KV.get(`license:${key}`);
  if (raw) {
    const record = JSON.parse(raw) as LicenseRecord;
    if (!record.active) {
      return { valid: false, message: "License key is deactivated or refunded." };
    }
    if (record.expires_at && new Date(record.expires_at).getTime() < Date.now()) {
      return { valid: false, message: "License update period has expired. Please renew your subscription." };
    }
    return {
      valid: true,
      tier: record.tier,
      email: record.email,
      expires_at: record.expires_at,
      message: "Active license.",
    };
  }

  // 2. Fallback pattern matching for offline/standard serials (e.g. OPT-PRO-XXXX-XXXX or legacy OLS-PRO-XXXX-XXXX)
  if ((key.startsWith("OPT-PRO-") || key.startsWith("OLS-PRO-")) && key.length >= 16) {
    return {
      valid: true,
      tier: "pro",
      message: "Standard serial verified.",
    };
  }

  return {
    valid: false,
    message: "Invalid license key. Please check your purchase confirmation email.",
  };
}

function generateDownloadUrl(origin: string, targetPath: string, key: string | null): string {
  if (targetPath.startsWith("http://") || targetPath.startsWith("https://")) {
    return targetPath;
  }
  const cleanPath = targetPath.startsWith("/") ? targetPath.slice(1) : targetPath;
  const param = key ? `?key=${encodeURIComponent(key)}` : "";
  return `${origin}/v1/download/${cleanPath}${param}`;
}

function parseSemver(v: string): number[] {
  const clean = v.trim().replace(/^v/, "").split("-")[0].split("+")[0];
  return clean.split(".").map((n) => parseInt(n, 10) || 0);
}

function isNewer(latest: string, current: string): boolean {
  const l = parseSemver(latest);
  const c = parseSemver(current);
  for (let i = 0; i < Math.max(l.length, c.length); i++) {
    const lPart = l[i] ?? 0;
    const cPart = c[i] ?? 0;
    if (lPart > cPart) return true;
    if (lPart < cPart) return false;
  }
  return false;
}

async function verifyHmacSha256(secret: string, body: string, hexSignature: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );

  const sigBytes = new Uint8Array(
    hexSignature.match(/.{1,2}/g)?.map((byte) => parseInt(byte, 16)) || []
  );

  return crypto.subtle.verify("HMAC", key, sigBytes, encoder.encode(body));
}

function jsonResponse(data: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...headers,
    },
  });
}

