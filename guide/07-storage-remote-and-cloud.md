# Chapter 7: Storage Architecture, Remote Sources & Cloud Sync

This chapter documents Optixel Studio's storage architecture, in-place local editing, network share mounting, Immich integration, Google Photos Picker API, and multi-cloud synchronization implemented in [`src-tauri/src/storage.rs`](../src-tauri/src/storage.rs), [`src-tauri/src/cloud.rs`](../src-tauri/src/cloud.rs), and [`src-tauri/src/remote.rs`](../src-tauri/src/remote.rs).

---

## 1. Storage Architecture & Location Strategy

Optixel Studio implements a strictly **non-destructive, local-first storage model**:
1. **Zero Mandatory Ingestion**: Local and network folders are browsed in-place. Media files are never renamed, converted, or copied to an opaque library bundle.
2. **Sidecar Synchronization**: Edit recipes, star ratings, flags, and color labels are saved both to SQLite (`catalog.olcat`) and synced to standard `.olxmp` sidecar files in the same directory as the original file.
3. **On-Demand Remote Caching**: Remote cloud assets and Immich originals are downloaded only when opened for develop or export, stored under `<OPTIXEL_DATA_DIR>/remote/`.

---

## 2. Network Drives (SMB, NFS, WebDAV)

Network shares allow studios and teams to edit photos stored on central NAS systems or file servers:

* **Supported Protocols**:
  * `smb://nas.local/photos` (Windows SMB, Samba, Synology, QNAP, TrueNAS).
  * `afp://` (Apple Filing Protocol).
  * `nfs://` (Network File System).
  * `https://` / `http://` (WebDAV shares).
* **OS-Level Mounting Facilities**:
  * **macOS**: Invokes `/sbin/mount_smbfs` or Finder's system connection dialog. Shares appear under `/Volumes/<share_name>`.
  * **Linux**: Mounts via GVfs daemon (`gio mount smb://...`) or `/mnt`.
  * **Windows**: Resolves UNC paths (`\\nas.local\photos`).
* **Connection Indicators**:
  The Library sidebar displays a live status dot: **Green** (connected & accessible), **Gray** (disconnected). Clicking a disconnected share triggers an OS mount request.

---

## 3. Immich Integration ([`remote.rs`](../src-tauri/src/remote.rs))

Optixel Studio features first-class integration with [Immich](https://immich.app), the popular self-hosted photo management server:

```
+---------------------------+                      +---------------------------+
|    Optixel Studio       |                      |       Immich Server       |
|  - Star Ratings (1-5)     | <==================> |  - Asset Rating (1-5)     |
|  - Pick Flag (P)          | <==================> |  - Favorite (true/false)  |
|  - Reject Flag (X)        | <==================> |  - Archive / Rejected     |
|  - Save to Immich         | -------------------> |  - Asset Upload & Stack   |
+---------------------------+                      +---------------------------+
```

### 3.1 Lazy Streaming & Pagination
* Rather than downloading the entire library on startup, Optixel streams Immich album contents **250 photos per page**.
* When browsing, thumbnail renditions are fetched directly from the Immich server's fast preview endpoints.
* Opening a photo in Develop automatically downloads the original RAW/HEIC into `<OPTIXEL_DATA_DIR>/remote/Immich/originals/`.

### 3.2 Bi-Directional Metadata Sync
* Updating a rating or flag in Optixel immediately dispatches an async HTTP `PUT /api/assets/{id}` to the Immich server.
* Re-opening an album pulls latest ratings from Immich, resolving conflicts in favor of recent timestamps.

### 3.3 Stacked Edits
* Clicking **Save to Immich…** renders the current edit and uploads it as a new asset stacked directly on top of the original master. In Immich, the edit displays as the primary photo while keeping the camera RAW original intact underneath.

---

## 4. Google Photos Integration

Due to Google Photos API restrictions that disallow full library listing or modification of existing photos, Optixel implements the modern **Google Photos Picker API**:

1. **OAuth 2.0 PKCE Flow**: The user signs in via a local browser loopback callback (`http://localhost`).
2. **Picking Session**: **Pick photos…** opens Google's native picker web UI.
3. **Downloading Picks**: Selected photos download into `<OPTIXEL_DATA_DIR>/remote/Cloud/GooglePhotos/` and appear under *Picked photos*.
4. **Export Uploads**: Edited photos are uploaded via the Google Photos Library API as new assets with original EXIF capture timestamps preserved.

---

## 5. Multi-Cloud Storage ([`cloud.rs`](../src-tauri/src/cloud.rs))

Optixel Studio supports direct connection to four primary cloud storage backends:

| Backend | Protocol & Authentication | Features |
|---|---|---|
| **S3-Compatible** | Custom **AWS Signature V4** signing over HTTPS | Supports AWS S3, Backblaze B2, Cloudflare R2, Wasabi, and local MinIO instances. Configurable bucket and region. |
| **WebDAV** | Basic / Digest HTTP Auth | Nextcloud, ownCloud, Synology WebDAV, pCloud, Koofr. In-place directory tree browsing. |
| **Dropbox** | OAuth 2.0 PKCE (Scoped App) | Browse user folders; on-demand chunked download; non-overwriting uploads. |
| **OneDrive** | Microsoft Entra OAuth 2.0 PKCE | Personal & Microsoft 365 Work/School accounts. Delegated `Files.ReadWrite` access. |

---

## 6. Duplicate Engine & Storage Audit ([`storage.rs`](../src-tauri/src/storage.rs))

The **Duplicate** facility (**Settings › Storage › Duplicate**) provides safe cross-source photo synchronization:

* **Source & Destination Matrix**:
  * Copy between any combination of Local Folders, Network Drives, Immich Albums, Google Photos picks, or Cloud Storage buckets.
* **Integrity & Verification**:
  * Exact matching via SHA-256 cryptographic hashing.
  * Perceptual look-alike matching via visual perceptual hash (`pHash`).
* **Conflict Resolution**:
  * **Strict Non-Destructive Rule**: Destination files with identical names but differing content are never overwritten; they are saved as `filename-1.ext`.
  * Preserves original file modification dates and EXIF metadata.

