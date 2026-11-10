# LibRaw 0.22.3 (vendored)

Unmodified `libraw/`, `src/` and `internal/` from the upstream release:

- Source: https://github.com/LibRaw/LibRaw/archive/refs/tags/0.22.3.tar.gz
- SHA-256 of that archive: `7edf96c5b70cff88980a02ab0fc8af125cb6a4a377985573c654439d9d02355c`
- License: LGPL-2.1 or CDDL-1.0, at the choice of whoever redistributes it (`COPYRIGHT`, `LICENSE.LGPL`,
  `LICENSE.CDDL`).

`src-tauri/build.rs` compiles every `src/**/*.cpp` except the `*_ph.cpp` placeholders (the same list as upstream's
makefiles) together with `csrc/ol_raw.cpp`, the only code that calls into it. Built without zlib, libjpeg, the DNG SDK
or OpenMP.

To update: replace the three folders with those of a newer release and correct the version and checksum above.
