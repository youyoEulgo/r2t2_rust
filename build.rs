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
    let (lib_dirs, source) = match std::env::var_os("R2T2_LIB_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            println!(
                "cargo:warning=using prebuilt llama.cpp from {}; \
                 this build is not tuned for the current machine",
                dir.display()
            );
            (vec![dir], source_dir(&manifest))
        }
        None => build_llama_cpp(&manifest, &out_dir),
    };

    link(&lib_dirs);
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
///
/// The library directories, not one directory: a multi-config generator writes
/// each target's archive beside that target instead of gathering them, so
/// `llama.lib` sits under `src/` and the ggml archives it needs under
/// `ggml/src/`.
fn build_llama_cpp(manifest: &Path, out_dir: &Path) -> (Vec<PathBuf>, PathBuf) {
    let source = source_dir(manifest);

    let build = out_dir.join("llama.cpp-build");
    let stamp = build.join(".r2t2-built");

    let force = std::env::var("R2T2_FORCE_REBUILD").is_ok_and(|v| v == "1");
    if force || !stamp.is_file() {
        configure_and_build(&source, &build);
        std::fs::write(&stamp, LLAMA_CPP_COMMIT).expect("could not write the build stamp");
    }

    (find_libraries(&build), source)
}

/// Locate the archives this project links, wherever the generator put them.
///
/// `build/bin` is where a single-config Ninja build gathers them and is checked
/// first, but it is not the only layout: the Visual Studio generator writes each
/// target's archive beside that target, under a per-configuration directory.
/// A layout that is not handled here fails at the link step with unresolved
/// symbols instead of saying what is missing.
#[cfg(windows)]
fn find_libraries(build: &Path) -> Vec<PathBuf> {
    // Dependencies before dependents, so the linker resolves each archive's
    // undefined symbols from the ones that follow. On Windows the static
    // archives are separate: `llama.lib` alone is not the whole library, and
    // `ggml.lib` needs the per-backend archives, of which CUDA's is the largest
    // and the only one that is optional.
    let wanted = ["ggml-cpu", "ggml-base", "ggml-cuda", "ggml", "vendor-hash", "mtmd", "llama"];
    let bin = build.join("bin");

    let mut dirs = Vec::new();
    for name in wanted {
        // No CUDA toolkit means no CUDA archive, which is a CPU build rather
        // than a broken one; every other archive is required.
        if name == "ggml-cuda" && !has_cuda(build) {
            continue;
        }
        let file = format!("{name}.lib");
        let found = if bin.join(&file).is_file() {
            Some(bin.clone())
        } else {
            search_for(build, &file)
        };
        match found {
            Some(dir) if !dirs.contains(&dir) => dirs.push(dir),
            Some(_) => {}
            None => panic!(
                "llama.cpp built but {file} was not found under {}; \
                 the layout may have changed",
                build.display()
            ),
        }
    }

    // The CUDA runtime and cuBLAS are not built here, they come from the
    // toolkit, and `ggml-cuda.lib` leaves their symbols undefined.
    if has_cuda(build) {
        match cuda_library_dir(build) {
            Some(dir) if !dirs.contains(&dir) => dirs.push(dir),
            _ => panic!(
                "llama.cpp was built with CUDA but the toolkit library directory \
                 was not found in {}; the layout may have changed",
                build.join("CMakeCache.txt").display()
            ),
        }
    }

    dirs
}

/// Whether the build that lives under `build` was configured with CUDA.
#[cfg(windows)]
fn has_cuda(build: &Path) -> bool {
    std::fs::read_to_string(build.join("CMakeCache.txt"))
        .map(|cache| cache.contains("GGML_CUDA:BOOL=ON"))
        .unwrap_or(false)
}

/// Where the CUDA toolkit keeps its libraries, as CMake found it.
///
/// Not guessed from `CUDA_PATH`: the build used whichever toolkit CMake
/// located, and linking against a different one's import libraries is the kind
/// of mismatch that produces a DLL that will not load.
#[cfg(windows)]
fn cuda_library_dir(build: &Path) -> Option<PathBuf> {
    let cache = std::fs::read_to_string(build.join("CMakeCache.txt")).ok()?;

    // `FindCUDAToolkit` records where `nvcc` is, from which the import
    // libraries sit next door, and a project-provided variable if the caller set
    // one. Each candidate is checked for a file only the toolkit's library
    // directory has, so a directory that merely exists is not enough.
    let key = |name: &str| {
        cache
            .lines()
            .find_map(|line| line.strip_prefix(name).map(|rest| PathBuf::from(rest.trim())))
    };
    let mut candidates = Vec::new();
    if let Some(bin) = key("CUDAToolkit_BIN_DIR:PATH=") {
        candidates.push(bin.join("../lib/x64"));
    }
    if let Some(lib) = key("CUDAToolkit_LIBRARY_DIR:PATH=") {
        candidates.push(lib);
    }

    candidates.into_iter().find(|dir| {
        dir.join("cudart_static.lib").is_file() && dir.join("cublas.lib").is_file()
    })
}

/// Find the directory holding `name` anywhere under `root`.
#[cfg(windows)]
fn search_for(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    let mut subdirectories = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirectories.push(path);
        } else if entry.file_name().to_string_lossy() == name {
            return path.parent().map(Path::to_path_buf);
        }
    }
    subdirectories
        .into_iter()
        .find_map(|directory| search_for(&directory, name))
}

/// The one directory a Unix build links from, which carries everything above it.
#[cfg(not(windows))]
fn find_libraries(build: &Path) -> Vec<PathBuf> {
    let bin = build.join("bin");
    if !bin.is_dir() {
        panic!(
            "llama.cpp built but {} does not exist; the layout may have changed",
            bin.display()
        );
    }
    vec![bin]
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
        "-DGGML_BUILD_TESTS=OFF",
        "-DGGML_BUILD_EXAMPLES=OFF",
        // mtmd is the audio path: it carries the mel front end and the
        // Conformer encoder, so it is not optional here.
        "-DLLAMA_BUILD_MTMD=ON",
        // Adapt to the CPU this is being built on. This is the whole point of
        // building locally.
        "-DGGML_NATIVE=ON",
    ]);

    // Windows gets the Visual Studio generator when Ninja is not on `PATH`, and
    // that generator is multi-config: it ignores CMAKE_BUILD_TYPE and takes the
    // configuration from CMAKE_CONFIGURATION_TYPES, whose default is "Debug".
    // Narrowing it to the one configuration wanted here is what makes
    // `--config Release` below mean something. On the single-config generators
    // used elsewhere the value is not read, so the Unix build passes nothing.
    #[cfg(windows)]
    cmake.arg("-DCMAKE_CONFIGURATION_TYPES=Release");

    // Shared libraries keep Unix runtime loading simple. On Windows, static
    // linking is more convenient for a double-clickable executable: the
    // equivalent DLLs would have to be copied beside the exe after every
    // Cargo build, while MSVC static libraries can be linked directly.
    if cfg!(windows) {
        cmake.arg("-DBUILD_SHARED_LIBS=OFF");
    } else {
        cmake.arg("-DBUILD_SHARED_LIBS=ON");
    }

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
            .args(config_args())
            .arg("--parallel")
            .arg(jobs.to_string()),
        "could not compile llama.cpp",
    );
}

/// The configuration to name on the build step.
///
/// A multi-config generator: which one Windows gets, as noted above. It defaults
/// to a configuration of its own choosing and stops with a message about the
/// combination not existing when that one is not in CMAKE_CONFIGURATION_TYPES,
/// so this build has to name the one it configured for. A single-config
/// generator, which the Unix build gets, already knows its configuration from
/// CMAKE_BUILD_TYPE and takes no such argument.
#[cfg(windows)]
fn config_args() -> [&'static str; 2] {
    ["--config", "Release"]
}

/// No configuration to name: this generator reads CMAKE_BUILD_TYPE.
#[cfg(not(windows))]
fn config_args() -> [&'static str; 0] {
    []
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
///
/// On Windows the archives are static and separate, so every directory holding
/// one of them is a search path, and the ggml archives have to be named too:
/// `llama.lib` leaves their symbols undefined.
fn link(lib_dirs: &[PathBuf]) {
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    #[cfg(windows)]
    {
        // The CUDA archives are only present in a build that had a toolkit, and
        // naming one the linker cannot find is an error, so they are linked
        // only when the toolkit's own directory is among the search paths.
        if lib_dirs.iter().any(|dir| dir.join("cublas.lib").is_file()) {
            // Dependencies before dependents, for the same reason as in
            // `find_libraries`. `cudart_static` is the runtime API, `cuda` the
            // driver API, and both resolve symbols that `ggml-cuda.lib` leaves
            // open; `cublas` is the matrix library its kernels call.
            for name in ["cudadevrt", "cudart_static", "cuda", "cublas", "cublasLt"] {
                println!("cargo:rustc-link-lib=static={name}");
            }
        }
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
        // Absent in a CPU-only build, where nothing references its symbols.
        if lib_dirs.iter().any(|dir| dir.join("ggml-cuda.lib").is_file()) {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
        }
        for name in ["ggml", "vendor-hash", "mtmd", "llama"] {
            println!("cargo:rustc-link-lib=static={name}");
        }
        // The runtime and cuBLAS archives request the dynamic CRT; naming the
        // static one as well is an error, so say which is meant.
        println!("cargo:rustc-link-arg=/DEFAULTLIB:msvcrt");
    }
    #[cfg(not(windows))]
    {
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=mtmd");
    }

    // Relative rpaths, so the binary keeps working when the tree is moved.
    //
    // ELF (Linux) spells the executable directory `$ORIGIN`; Mach-O (macOS)
    // spells it `@loader_path`. Windows has no rpath at all: the normal DLL
    // search order includes the executable's directory, so adding Unix linker
    // flags there would make an MSVC build fail.
    #[cfg(unix)]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{ORIGIN}/../lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{ORIGIN}");
        // The build directory, for running straight out of target/<profile>/.
        for dir in lib_dirs {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
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
