# Optixel Studio Update Entitlement Worker

This Cloudflare Worker provides cryptographic, entitlement-gated auto-updates for paid pre-built binaries of Optixel Studio via **`tauri-plugin-updater`** and **Cloudflare R2**.

---

## Architecture Overview

```
[ Optixel Desktop App (Tauri v2) ]
          |
          | 1. GET /v1/update/:target/:current_version
          |    Header: Authorization: Bearer OPT-PRO-XXXX
          v
[ Cloudflare Worker (this worker) ]
    ├──> Validates license key against [ Cloudflare KV (LICENSES_KV) ]
    ├──> Reads latest release metadata from [ Cloudflare R2 (RELEASES_BUCKET) ]
    └──> Returns signed Tauri v2 JSON manifest
          |
          | 2. App verifies ed25519 signature & streams installer
          v
[ Cloudflare R2 Bucket (optixel-releases) ]
```

---

## 1. Prerequisites & Setup

1. Install dependencies:
   ```bash
   cd cloudflare-updater-worker
   npm install
   ```

2. Login to your Cloudflare account via Wrangler:
   ```bash
   npx wrangler login
   ```

3. Create the KV Namespace for License Keys:
   ```bash
   npx wrangler kv namespace create LICENSES_KV
   ```
   Copy the generated `id` into `wrangler.jsonc` under `kv_namespaces`.

4. Create the R2 Storage Bucket for binaries:
   ```bash
   npx wrangler r2 bucket create optixel-releases
   ```

---

## 2. Generating Tauri v2 Cryptographic Keys

Tauri v2 uses **minisign / ed25519** public key cryptography:

1. Generate keypair:
   ```bash
   npx @tauri-apps/cli signer generate -w ~/.optixel/updater.key
   ```
   This generates:
   - **Private key**: Stored safely in `~/.optixel/updater.key` (or your CI/CD secret manager).
   - **Public key string**: e.g., `dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk...`

2. Place the **public key** into `src-tauri/tauri.conf.json`:
   ```json
   "plugins": {
     "updater": {
       "pubkey": "<YOUR_PUBLIC_KEY_HERE>",
       "endpoints": [
         "https://updates.optixel.studio/v1/update/{{target}}/{{current_version}}"
       ]
     }
   }
   ```

---

## 3. Uploading a Release to R2

When building a release:
```bash
./scripts/build.sh
```

Tauri will generate:
- `Optixel Studio.app.tar.gz`
- `Optixel Studio.app.tar.gz.sig` (signature)

Upload both to your R2 bucket:
```bash
npx wrangler r2 object put optixel-releases/packages/optixel-studio_0.3.1_aarch64.app.tar.gz --file=...
```

And upload the release manifest:
```json
// manifests/latest.json
{
  "version": "0.3.1",
  "notes": "## What's New in v0.3.1\n- Native C++ LibRaw demosaicing\n- LittleCMS 2 soft-proofing\n- GPU Waveform and RGB Parade scopes",
  "pub_date": "2026-11-10T12:00:00Z",
  "platforms": {
    "darwin-aarch64": {
      "signature": "<CONTENTS_OF_.app.tar.gz.sig>",
      "url": "optixel-studio_0.3.1_aarch64.app.tar.gz"
    },
    "windows-x86_64": {
      "signature": "<CONTENTS_OF_.nsis.zip.sig>",
      "url": "optixel-studio_0.3.1_x64-setup.nsis.zip"
    }
  }
}
```
Upload manifest:
```bash
npx wrangler r2 object put optixel-releases/manifests/latest.json --file=manifest.json
```

---

## 4. Merchant of Record Webhooks

To automatically grant updates upon purchase:
- In **Lemon Squeezy** or **Polar.sh**, configure your webhook URL to:
  `https://updates.optixel.studio/v1/webhook/lemonsqueezy`
- Set secret via Wrangler:
  ```bash
  npx wrangler secret put LEMON_SQUEEZY_WEBHOOK_SECRET
  ```

---

## 5. Local Testing & Deployment

- Test locally:
  ```bash
  npm run dev
  ```
- Deploy to edge network:
  ```bash
  npm run deploy
  ```

