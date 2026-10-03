// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Build script: obtain llama.cpp, build it for *this* machine, and generate
//! the FFI bindings.
//!
//! # Why the library is built rather than shipped
//!
//! A prebuilt llama.cpp is only usable on a machine like the one that produced
//! it. Its CPU backend is compiled for a specific instruction set, and its CUDA
//! backend for specific GPU architectures. Shipping one build means choosing
//! between two bad outcomes: target the lowest common denominator and waste the
//! hardware, or target something recent and crash with `Illegal instruction`
//! on everything older. That is not hypothetical -- the prebuilt artifacts this
//! project started from contained AVX-512 instructions and died on any CPU
//! without AVX-512, which is every 12th- and 13th-generation Intel desktop
//! part.
//!
//! So the library is compiled here, on the machine that will run it, where
//! `-march=native` and the CUDA architecture list both mean what they should.
//! The cost is build time, paid once.
//!
//! # What this script does
//!
//! 1. Clone llama.cpp at the pinned commit if it is not already present.
//! 2. Configure and build it with CMake into `OUT_DIR/llama.cpp-build`.
//! 3. Point the linker at the result and write the FFI bindings.
//!
//! Steps 1 and 2 are skipped when the build directory is already up to date,
//! so an ordinary `cargo build` after a source edit does not rebuild it.
//!
//! # Environment overrides
//!
//! | variable | effect |
//! |---|---|
//! | `R2T2_LIB_DIR` | link against prebuilt libraries, skipping the build |
//! | `R2T2_LLAMA_DIR` | use an existing llama.cpp checkout instead of cloning |
//! | `R2T2_CUDA=0` | force a CPU-only build |
//! | `R2T2_CUDA_ARCHS` | override the CUDA architecture list |
//! | `R2T2_FORCE_REBUILD=1` | rebuild even if the cached build looks current |

use std::path::{Path, PathBuf};
use std::process::Command;

/// llama.cpp revision this project is built and tested against.
///
/// Pinned rather than tracking a branch: `mtmd`'s C API changes between
/// releases, and the call sequence in `src/engine.rs` is written against this
/// one.
const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp";
const LLAMA_CPP_COMMIT: &str = "ad6c66839af3c5646fba8c6c2e2087a1e4e38948";

/// CUDA architectures compiled when CUDA is enabled and the caller does not say
/// otherwise.
///
/// `-virtual` entries embed PTX, which the driver just-in-time compiles for
/// whatever GPU is present, so older cards work without a native binary for
/// each. Only the current mainstream architectures are built natively, for
/// speed on the hardware most likely to be in use.
const DEFAULT_CUDA_ARCHS: &str = "75-virtual;80-virtual;86-real;89-real";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "R2T2_LIB_DIR",
        "R2T2_LLAMA_DIR",
        "R2T2_CUDA",
        "R2T2_CUDA_ARCHS",
        "R2T2_FORCE_REBUILD",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // A caller-provided library directory short-circuits the build: useful for a
    // distribution that ships its own llama.cpp, or for testing without waiting
    // for a compile. The headers still come from a checkout, since bindgen needs
    // them and they are not part of a library-only distribution.
    let (lib_dir, source) = match std::env::var_os("R2T2_LIB_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            println!(
                "cargo:warning=using prebuilt llama.cpp from {}; \
                 this build is not tuned for the current machine",
                dir.display()
            );
            (dir, source_dir(&manifest))
        }
        None => build_llama_cpp(&manifest, &out_dir),
    };

    link(&lib_dir);
    bind(&source, &out_dir);
}

// --------------------------------------------------------------------------- //
// building llama.cpp
// --------------------------------------------------------------------------- //

/// The llama.cpp checkout to use, cloning it if necessary.
///
/// `R2T2_LLAMA_DIR` points at an existing checkout; otherwise the pinned
/// revision is fetched into `third_party/`, which is not tracked.
fn source_dir(manifest: &Path) -> PathBuf {
    let dir = match std::env::var_os("R2T2_LLAMA_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let dir = manifest.join("third_party/llama.cpp");
            ensure_checkout(&dir);
            dir
        }
    };
    if !dir.join("CMakeLists.txt").is_file() {
        panic!("{} does not look like a llama.cpp checkout", dir.display());
    }
    dir
}

/// Ensure llama.cpp is built for this machine, returning where it landed.
fn build_llama_cpp(manifest: &Path, out_dir: &Path) -> (PathBuf, PathBuf) {
    let source = source_dir(manifest);

    let build = out_dir.join("llama.cpp-build");
    let stamp = build.join(".r2t2-built");

    let force = std::env::var("R2T2_FORCE_REBUILD").is_ok_and(|v| v == "1");
    if force || !stamp.is_file() {
        configure_and_build(&source, &build);
        std::fs::write(&stamp, LLAMA_CPP_COMMIT).expect("could not write the build stamp");
    }

    let lib_dir = build.join("bin");
    if !lib_dir.is_dir() {
        panic!(
            "llama.cpp built but {} does not exist; the layout may have changed",
            lib_dir.display()
        );
    }

    (lib_dir, source)
}

/// Clone llama.cpp at the pinned revision if it is not already there.
fn ensure_checkout(dir: &Path) {
    if dir.join("CMakeLists.txt").is_file() {
        return;
    }
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).expect("could not create third_party/");
    }

    eprintln!(
        "r2t2: fetching llama.cpp ({}) into {}",
        &LLAMA_CPP_COMMIT[..12],
        dir.display()
    );
    run(
        Command::new("git")
            .args(["clone", "--filter=blob:none", LLAMA_CPP_REPO])
            .arg(dir),
        "could not clone llama.cpp (is git installed, and is the network reachable?)",
    );
    run(
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["checkout", "--quiet", LLAMA_CPP_COMMIT]),
        "could not check out the pinned llama.cpp revision",
    );
}

/// Check that a program exists, with advice the platform understands.
///
/// A missing tool otherwise surfaces as `No such file or directory` from a
/// command the reader never typed, which says what failed but not what to
/// install.
fn require(program: &str, install: &str) {
    let found = Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match found {
        Ok(status) if status.success() => {}
        _ => panic!(
            "`{program}` is required to build llama.cpp but was not found.\n\
             Install it with:\n\
             \x20 {install}"
        ),
    }
}

/// Configure and compile llama.cpp.
fn configure_and_build(source: &Path, build: &Path) {
    #[cfg(target_os = "macos")]
    require("cmake", "brew install cmake");
    #[cfg(not(target_os = "macos"))]
    require("cmake", "your package manager, e.g. apt install cmake");

    let cuda = cuda_requested();

    eprintln!(
        "r2t2: building llama.cpp for this machine ({}) — takes a few minutes, once",
        backend_name(cuda)
    );

    let mut cmake = Command::new("cmake");
    cmake.arg("-S").arg(source).arg("-B").arg(build).args([
        "-DCMAKE_BUILD_TYPE=Release",
        // Only the two libraries this project links, plus their dependencies.
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        // The unified `llama` binary; it needs headers generated by a step
        // this build does not run, and nothing here calls it.
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_CURL=OFF",
        // Shared libraries, so one build serves every binary and the rpath
        // story stays simple.
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_BUILD_TESTS=OFF",
        "-DGGML_BUILD_EXAMPLES=OFF",
        // mtmd is the audio path: it carries the mel front end and the
        // Conformer encoder, so it is not optional here.
        "-DLLAMA_BUILD_MTMD=ON",
        // Adapt to the CPU this is being built on. This is the whole point of
        // building locally.
        "-DGGML_NATIVE=ON",
    ]);

    if cuda {
        cmake.arg("-DGGML_CUDA=ON");
        let archs = std::env::var("R2T2_CUDA_ARCHS")
            .unwrap_or_else(|_| DEFAULT_CUDA_ARCHS.to_string());
        cmake.arg(format!("-DCMAKE_CUDA_ARCHITECTURES={archs}"));
    } else {
        cmake.arg("-DGGML_CUDA=OFF");
    }

    // Metal is what llama.cpp uses on macOS. It defaults to on there, but the
    // default is someone else's to change, and a silent switch to the CPU would
    // look like this program being inexplicably slow rather than misconfigured.
    if cfg!(target_os = "macos") {
        cmake.arg("-DGGML_METAL=ON");
    }

    run(
        &mut cmake,
        "could not configure llama.cpp (is cmake installed and recent enough?)",
    );

    // Cap the parallelism: CUDA translation units are memory-hungry, and a
    // machine with many cores can exhaust memory compiling them all at once.
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(build)
            .arg("--parallel")
            .arg(jobs.to_string()),
        "could not compile llama.cpp",
    );
}

/// Whether to build the CUDA backend.
///
/// Enabled when `nvcc` is present and the caller has not opted out. A missing
/// toolkit is not an error and not a fallback to the CPU: on macOS the Metal
/// backend is what llama.cpp selects by default, and on Linux a CPU build still
/// transcribes, just slower. Saying "CPU only" there would be wrong twice over.
fn cuda_requested() -> bool {
    if std::env::var("R2T2_CUDA").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false")) {
        return false;
    }
    if cfg!(target_os = "macos") {
        // `nvcc` does not exist on macOS and CUDA is not the backend there.
        return false;
    }
    let found = Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !found {
        eprintln!("r2t2: nvcc not found; the CUDA backend will be skipped");
    }
    found
}

/// Which accelerator the build will actually use, for the progress message.
fn backend_name(cuda: bool) -> &'static str {
    if cuda {
        "CUDA + CPU"
    } else if cfg!(target_os = "macos") {
        "Metal + CPU"
    } else {
        "CPU"
    }
}

// --------------------------------------------------------------------------- //
// linking and bindings
// --------------------------------------------------------------------------- //

/// Tell the linker where the libraries are and how to find them at runtime.
fn link(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");

    // Relative rpaths, so the binary keeps working when the tree is moved.
    //
    // The spelling of "the directory holding this executable" depends on the
    // object format: ELF, used on Linux, writes `$ORIGIN`, while Mach-O, used
    // on macOS, writes `@loader_path`. Getting it wrong is not cosmetic — the
    // linker rejects the unknown syntax, or the binary builds and then dies at
    // startup with "Library not loaded".
    println!("cargo:rustc-link-arg=-Wl,-rpath,{ORIGIN}/../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{ORIGIN}");
    // The build directory, for running straight out of target/<profile>/.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}

/// How the platform spells "the directory holding this executable".
#[cfg(target_os = "macos")]
const ORIGIN: &str = "@loader_path";
#[cfg(not(target_os = "macos"))]
const ORIGIN: &str = "$ORIGIN";

/// Generate the FFI declarations from the headers.
///
/// The three headers live in three different places in a llama.cpp checkout,
/// so each is named by its full path and each directory is passed to clang as a
/// search path — the headers include each other by bare name, so clang must be
/// able to resolve `ggml.h` from inside `llama.h`.
fn bind(source: &Path, out_dir: &Path) {
    let llama_h = source.join("include/llama.h");
    let mtmd_h = source.join("tools/mtmd/mtmd.h");
    let mtmd_helper_h = source.join("tools/mtmd/mtmd-helper.h");

    for header in [&llama_h, &mtmd_h, &mtmd_helper_h] {
        if !header.is_file() {
            panic!(
                "missing header: {}\n\
                 The llama.cpp layout may have changed; this project expects a \
                 checkout of {}.",
                header.display(),
                LLAMA_CPP_COMMIT
            );
        }
    }

    println!("cargo:rerun-if-changed={}", llama_h.display());

    let out = out_dir.join("bindings.rs");
    let bindings = bindgen::Builder::default()
        .header(llama_h.to_string_lossy())
        .header(mtmd_h.to_string_lossy())
        .header(mtmd_helper_h.to_string_lossy())
        // Where each header's own includes resolve from.
        .clang_arg("-I")
        .clang_arg(source.join("include").to_string_lossy())
        .clang_arg("-I")
        .clang_arg(source.join("ggml/include").to_string_lossy())
        .clang_arg("-I")
        .clang_arg(source.join("tools/mtmd").to_string_lossy())
        // Parse as C, not C++: `mtmd-helper.h` guards a `namespace mtmd_helper`
        // block behind `#ifdef __cplusplus`, and emitting that as well produces
        // duplicate definitions. Everything called here lives in the
        // `extern "C"` block, which both language modes expose.
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg("-std=c11")
        .allowlist_function("llama_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("MTMD_.*")
        .allowlist_var("GGML_.*")
        .generate()
        .unwrap_or_else(|e| {
            panic!(
                "bindgen could not read the llama.cpp headers under {}: {e}\n\
                 (libclang must be installed)",
                source.display()
            )
        });

    bindings
        .write_to_file(&out)
        .expect("could not write bindings.rs");
}

/// Run a command, reporting the exact invocation when it fails.
fn run(command: &mut Command, context: &str) {
    let display = format!("{command:?}");
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("{context}\n  while running: {display}\n  error: {e}"));
    if !status.success() {
        panic!("{context}\n  command failed with {status}: {display}");
    }
}
