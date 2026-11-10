use std::path::{Path, PathBuf};

fn main() {
    // ort's CoreML code uses @available, whose ___isPlatformVersionAtLeast lives in clang's runtime, which rustc does not link
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let out = std::process::Command::new("xcrun").args(["clang", "--print-resource-dir"]).output().expect("xcrun clang");
        let dir = String::from_utf8(out.stdout).unwrap();
        println!("cargo:rustc-link-search=native={}/lib/darwin", dir.trim());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
        println!("cargo:rustc-link-lib=framework=ImageIO");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    }
    compile_shaders();
    libraw();
    tauri_build::build()
}

/// Ahead-of-Time (AOT) GPU Compute Shader Pipeline:
/// Compiles shaders/*.wgsl into stripped binary SPIR-V (.spv) bytecode in OUT_DIR.
fn compile_shaders() {
    use naga::back::spv;
    use naga::front::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let shader_dir = Path::new("shaders");
    println!("cargo:rerun-if-changed=shaders");
    if !shader_dir.exists() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let mut entries: Vec<PathBuf> = std::fs::read_dir(shader_dir)
        .expect("read shaders/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "wgsl"))
        .collect();
    entries.sort();

    for path in entries {
        println!("cargo:rerun-if-changed={}", path.display());
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read shader {}: {e}", path.display()));
        let module = wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("WGSL parse error in {}:\n{}", path.display(), e.emit_to_string(&src)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("SPIR-V validation error in {}: {e:?}", path.display()));

        // Strip all debug symbols (WriterFlags::empty()) and emit binary SPIR-V words
        let spv_opts = spv::Options {
            lang_version: (1, 3),
            flags: spv::WriterFlags::empty(),
            ..Default::default()
        };
        let words = spv::write_vec(&module, &info, &spv_opts, None)
            .unwrap_or_else(|e| panic!("SPIR-V codegen error in {}: {e:?}", path.display()));
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let stem = path.file_stem().unwrap().to_string_lossy();
        let spv_path = out_dir.join(format!("{stem}.spv"));
        std::fs::write(&spv_path, &bytes)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", spv_path.display()));
    }
}

/// LibRaw (vendor/libraw: every source but upstream's `_ph` placeholders, as its makefiles build it) and
/// csrc/*.cpp (ol_raw.cpp, ol_demosaic.cpp, ol_codecs.cpp).
fn libraw() {
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).expect("vendor/libraw/src").flatten() {
            let p = e.path();
            if p.is_dir() {
                sources(&p, out);
            } else if p.extension().is_some_and(|x| x == "cpp") && !p.file_stem().is_some_and(|s| s.to_string_lossy().ends_with("_ph")) {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    sources(Path::new("csrc"), &mut files);
    sources(Path::new("vendor/libraw/src"), &mut files);
    files.sort();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include("csrc")
        .include("vendor/libraw")
        .define("LIBRAW_NODLL", None) // a static library: no dllimport/dllexport on Windows
        .flag_if_supported("-pthread")
        .warnings(false);

    // Detect libjpeg-turbo (SIMD lossless DCT transform API) if installed locally
    let turbo_candidates = [
        std::env::var("TURBOJPEG_ROOT").ok().map(PathBuf::from),
        Some(PathBuf::from("/opt/homebrew/opt/jpeg-turbo")),
        Some(PathBuf::from("/usr/local/opt/jpeg-turbo")),
        Some(PathBuf::from("/usr")),
        Some(PathBuf::from("/usr/local")),
    ];
    'outer: for cand in turbo_candidates.into_iter().flatten() {
        let inc = cand.join("include");
        let lib_dirs = [
            cand.join("lib"),
            cand.join("lib/x86_64-linux-gnu"),
            cand.join("lib/aarch64-linux-gnu"),
            cand.join("lib64"),
        ];
        if inc.join("turbojpeg.h").exists() {
            for lib in lib_dirs {
                if lib.join("libturbojpeg.a").exists() {
                    build.include(&inc);
                    build.define("OL_HAVE_TURBOJPEG", "1");
                    println!("cargo:rustc-link-search=native={}", lib.display());
                    println!("cargo:rustc-link-lib=static=turbojpeg");
                    break 'outer;
                } else if lib.join("libturbojpeg.so").exists() {
                    build.include(&inc);
                    build.define("OL_HAVE_TURBOJPEG", "1");
                    println!("cargo:rustc-link-search=native={}", lib.display());
                    println!("cargo:rustc-link-lib=turbojpeg");
                    break 'outer;
                }
            }
        }
    }

    build.files(files).compile("olraw");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=vendor/libraw");
}
