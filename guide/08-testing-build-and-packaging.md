# Chapter 8: Testing, Verification, Multi-Platform Packaging & CI/CD

This chapter provides comprehensive instructions for running automated tests, validating TypeScript/Rust builds, packaging production installers across macOS, Linux, and Windows, and configuring CI/CD pipelines.

---

## 1. Automated Testing & Verification

Optixel Studio uses a dual-layer testing strategy: Rust cargo unit/integration tests for backend math and database operations, and TypeScript/Vite compilation for frontend type safety.

### 1.1 The Master Test Script
Run all tests and build validations with a single command:
```bash
npm test
# Or directly:
./scripts/test.sh
```

This script:
1. Runs `cargo test --manifest-path src-tauri/Cargo.toml`.
2. Runs `npm run build` (`tsc && vite build`) to verify that all React components and TypeScript types compile without errors.

### 1.2 Running Rust Backend Tests
To test specific Rust modules or test cases:

```bash
# Run all tests in the workspace
cargo test --manifest-path src-tauri/Cargo.toml

# Run tests in develop.rs (tonal math, color matrix, spline tests)
cargo test --manifest-path src-tauri/Cargo.toml develop::tests

# Run tests in catalog.rs (SQLite WAL, XMP sidecars, migrations)
cargo test --manifest-path src-tauri/Cargo.toml catalog::tests

# Run tests with stdout output enabled
cargo test --manifest-path src-tauri/Cargo.toml -- --nocapture
```

### 1.3 Running Frontend Type Checks & Production Bundle
```bash
# Typecheck only (no emit)
npx tsc --noEmit

# Full production build (generates dist/ folder)
npm run build
```

---

## 2. Multi-Platform Packaging & Distribution

Optixel Studio packages native production distributions for **macOS**, **Linux**, and **Windows**.

### 2.1 macOS Production Packaging (`scripts/build.sh`)
On macOS, [`scripts/build.sh`](../scripts/build.sh) compiles the optimized binary, bundles the `.app`, and creates a signed/compressed `.dmg` using macOS `hdiutil`:

```bash
./scripts/build.sh
```

**Packaging Steps Executed**:
1. Builds frontend assets with Vite into `dist/`.
2. Compiles optimized Rust binary (`profile.release` with LTO).
3. Packages the macOS Application Bundle via Tauri: `src-tauri/target/release/bundle/macos/Optixel Studio.app`.
4. Creates a compressed DMG installer (`UDZO` format): `src-tauri/target/release/bundle/dmg/Optixel Studio_0.1.0_<arch>.dmg`.
5. Copies finalized artifacts into `dist-packages/macos/`.

### 2.2 Linux Production Packaging
There are two ways to package for Linux:

#### Option A: Containerized Docker Build (`scripts/build-linux.sh`)
Recommended on any platform (including macOS) to build clean Linux `.deb` packages without installing local GTK development headers:
```bash
./scripts/build-linux.sh
```
This builds an isolated builder image via [`scripts/Dockerfile.linux`](../scripts/Dockerfile.linux) based on Debian, compiles the binary, and outputs the `.deb` package to `dist-packages/linux/deb/Optixel Studio_0.1.0_<arch>.deb`.

#### Option B: Native Linux System Build
On a native Linux workstation with prerequisites installed:
```bash
npx tauri build
```
Artifacts are generated under:
* `src-tauri/target/release/bundle/deb/*.deb`
* `src-tauri/target/release/bundle/appimage/*.AppImage`

### 2.3 Windows Packaging
On a Windows 10/11 system:
```bash
npm run tauri build
```
Artifacts generated:
* `src-tauri/target/release/bundle/msi/*.msi`
* `src-tauri/target/release/bundle/nsis/*.exe`

---

## 3. Production Release Packaging & Local Build Automation

Starting with **v0.3.0** and all subsequent releases, production packaging is performed locally and via containerized builders rather than automated GitHub cloud runner workflows:

* **macOS Production Releases**: Executed locally via [`./scripts/build.sh`](../scripts/build.sh), which bundles the macOS `.app`, creates a signed `.dmg` installer using `hdiutil`, and places release assets into `dist-packages/macos/`.
* **Linux Production Releases**: Executed via [`./scripts/build-linux.sh`](../scripts/build-linux.sh) (using [`scripts/Dockerfile.linux`](../scripts/Dockerfile.linux)), compiling clean `.deb` packages inside an isolated container with all native dependencies pre-configured and outputting to `dist-packages/linux/deb/`.
* **Windows Production Releases**: Executed on Windows via `npm run tauri build`.

This ensures fast, deterministic, reproducible releases with full developer inspection prior to distribution.
6. **Tauri Action**: Runs `tauri-apps/tauri-action@v0`, building frontend and native bundles.
7. **Artifact Upload**: Publishes zipped build artifacts for all platforms.

