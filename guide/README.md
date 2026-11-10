# Optixel Studio — Developer Documentation & Engineering Guidelines

Welcome to the **Optixel Studio** developer guide. This documentation is the authoritative reference for software engineers, systems programmers, and AI coding agents (including Google Antigravity) developing, extending, testing, and packaging Optixel Studio.

Optixel Studio is a professional, non-destructive RAW photo processor, digital asset manager (DAM), and studio production suite built on **Tauri v2**, **Rust**, **React 19**, **TypeScript**, **Tailwind CSS**, and **SQLite (WAL)**. It features a built-in **Model Context Protocol (MCP)** server and a **pluggable AI/CLI agent router** that enables autonomous agents and local/cloud neural models to inspect, cull, edit, and export photos alongside human photographers.

---

## Guide Structure & Table of Contents

| Chapter | Document | Core Focus |
|---|---|---|
| **01** | [01-environment-and-setup.md](./01-environment-and-setup.md) | Prerequisites (Rust 2024, Node 20+), OS toolchains (macOS, Linux, Windows), repository scripts, environment variables, and initial run. |
| **02** | [02-architecture-and-runtime.md](./02-architecture-and-runtime.md) | High-level system architecture, dual-mode runtime (Desktop GUI vs. Headless CLI vs. stdio MCP), IPC bridge, and state ownership. |
| **03** | [03-frontend-architecture.md](./03-frontend-architecture.md) | React 19 component hierarchy (`App`, `Library`, `Develop`, `Advanced`, `Pro`, `AutoView`, `Settings`, `Storage`), UI controls, Tailwind CSS, and 60fps rendering guidelines. |
| **04** | [04-backend-rust-core.md](./04-backend-rust-core.md) | Rust crate structure (`src-tauri/src/`), core modules (`lib.rs`, `develop.rs`, `catalog.rs`, `vision.rs`, `enhance.rs`, `auto.rs`), thread safety (`rayon`), and memory layout. |
| **05** | [05-image-pipeline-and-color.md](./05-image-pipeline-and-color.md) | Non-destructive DAG pipeline, 32-bit linear Rec.2020 working space, Oklab perceptual space, camera matrix/WB, tone curves, 3D Point Color, and export color management. |
| **06** | [06-mcp-server-and-ai-router.md](./06-mcp-server-and-ai-router.md) | Built-in MCP Server (Streamable HTTP & stdio), Tool & Resource definitions, agent loop, and the Pluggable AI Router (Claude CLI, Antigravity CLI, Ollama, Gemini, OpenAI). |
| **07** | [07-storage-remote-and-cloud.md](./07-storage-remote-and-cloud.md) | Storage engine, local folders, network shares (SMB/NFS/WebDAV), Immich bi-directional API integration, Google Photos Picker API, S3/Dropbox/OneDrive, and Duplicate engine. |
| **08** | [08-testing-build-and-packaging.md](./08-testing-build-and-packaging.md) | Rust unit/integration tests, TypeScript validation, macOS `.app` & `.dmg` packaging, Linux Docker containerized builds, and CI/CD pipelines. |
| **09** | [09-antigravity-and-ai-workflow.md](./09-antigravity-and-ai-workflow.md) | AI-assisted development with Google Antigravity (AGY), slash commands, skills (`antigravity-guide`), subagents, MCP debugging, and contribution guidelines. |

---

## High-Level System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    USER INTERFACE (React 19 + Vite)                               |
|   Library View   |   Develop View   |   Advanced & Pro   |   Auto Module   |   Settings & Storage   |
+------------------------------------------------+-------------------------------------------------+
                                                 | Tauri IPC (invoke / emit)
+------------------------------------------------v-------------------------------------------------+
|                                    TAURI v2 APPLICATION SHELL                                    |
|   Window Management  |  Menu System  |  Native Dialogs  |  Asset Protocol  |  Lifecycle Hooks    |
+------------------------------------------------+-------------------------------------------------+
                                                 |
+------------------------------------------------v-------------------------------------------------+
|                                     RUST BACKEND CORE ENGINE                                     |
|                                                                                                  |
|   +--------------------------+  +--------------------------+  +-------------------------------+  |
|   |   Non-Destructive DAG    |  |     DAM SQLite WAL       |  |      Vision & Analysis        |  |
|   |   Linear Rec.2020 Float  |  |   catalog.olcat + XMP    |  |   Laplacian, pHash, MTF       |  |
|   +--------------------------+  +--------------------------+  +-------------------------------+  |
|                                                                                                  |
|   +--------------------------+  +--------------------------+  +-------------------------------+  |
|   |    Auto Batch Engine     |  |   Remote & Cloud Sync    |  |     Wavelet & Super-Res       |  |
|   |   6 Cancellable Jobs     |  |   Immich / GPhotos / S3  |  |   à trous, Lanczos, Ext AI    |  |
|   +--------------------------+  +--------------------------+  +-------------------------------+  |
|                                                                                                  |
|   +--------------------------------------------------------+  +-------------------------------+  |
|   |             BUILT-IN MCP SERVER (mcp.rs)               |  |     AI PROVIDER ROUTER        |  |
|   |   Streamable HTTP (41920) + stdio IPC                  |  |  Claude CLI, Antigravity CLI  |  |
|   |   10 Tools | 5 Resources | AI Provenance Tracking      |  |  Ollama, Gemini, OpenAI APIs  |  |
|   +--------------------------------------------------------+  +-------------------------------+  |
+--------------------------------------------------------------------------------------------------+
```

---

## Quick Reference: Everyday Developer Commands

All major operations have helper scripts located in `./scripts/` and npm scripts in `package.json`:

```bash
# 1. Install dependencies
npm install

# 2. Run Optixel Studio in development mode (hot-reloading React + live native desktop)
npm run app
# Or directly:
./scripts/dev.sh

# 3. Run all tests and builds (Rust cargo tests + TypeScript compile)
npm test
# Or:
./scripts/test.sh

# 4. Run Rust unit tests specifically
npm run test:rs

# 5. Build production packages
# macOS (builds .app and packages .dmg into dist-packages/macos/):
./scripts/build.sh

# Linux (builds .deb inside Docker container into dist-packages/linux/deb/):
./scripts/build-linux.sh

# 6. Launch headless MCP stdio server
npm run mcp
# Or:
./scripts/mcp.sh
```

---

## Core Tenets & Engineering Principles

1. **Non-Destructive Guarantee**: Original RAW and bitmap files are **strictly read-only**. All crop, tone, color, geometry, and retouch operations are stored as mathematical recipes in SQLite (`catalog.olcat`) and synced to `.olxmp` sidecar files.
2. **Single Truth Pipeline**: The same Rust rendering pipeline (`develop::render`) generates the 1600px interactive viewport proxy, full-resolution 16-bit exports, and MCP agent previews. What the AI sees through `photo_inspect` or `optixel://photo/{id}/preview` matches the human photographer's screen pixel-for-pixel.
3. **Pluggable AI Co-Piloting**: AI is a first-class collaborator. Through MCP (port `41920`) or CLI providers (Antigravity, Claude), agents can inspect image histograms, detect clipping, generate boolean polygon masks, heal blemishes, and apply precise parametric corrections with an automatic undo/redo stack.
4. **Resilient Local-First DAM**: No forced cloud account or vendor lock-in. Folders are browsed in-place without copying. Remote cloud folders (S3, Dropbox, OneDrive) and photo services (Immich, Google Photos) are downloaded on demand into a structured cache.

