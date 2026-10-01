// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Build script: generate FFI bindings for llama.cpp and wire up linking.
//!
//! Two things happen here:
//!
//! 1. `bindgen` parses the vendored C headers (`llama.h`, `mtmd.h`, `ggml.h`)
//!    and emits Rust declarations into `OUT_DIR/bindings.rs`. Generating rather
//!    than hand-writing matters because `llama_context_params` has ~40 fields;
//!    a single mis-ordered field would silently corrupt every call through it.
//!
//! 2. The linker is pointed at `vendor/lib`, where the prebuilt llama.cpp
//!    shared libraries live. The rpath is `$ORIGIN/../lib` (relative to the
//!    executable in `target/<profile>/`) so the binary keeps working when the
//!    tree is moved or copied.
//!
//! Set `R2T2_LIB_DIR` to link against libraries somewhere else.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_dir = manifest_dir.join("vendor/include");
    let lib_dir = env::var("R2T2_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("vendor/lib"));

    // Re-run when headers or libraries change.
    println!("cargo:rerun-if-changed=vendor/include");
    println!("cargo:rerun-if-changed=build.rs");

    // ---------------------------------------------------------------- linking
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");

    // Relative rpath so the built binary finds the libs next to it.
    // `$ORIGIN` is the directory holding the executable; cargo puts binaries in
    // target/<profile>/, but vendored libs live at <root>/vendor/lib, so we
    // cover both the in-tree layout and a deployed layout.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../vendor/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    // Escape hatch for running straight out of target/<profile>/ during dev.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // --------------------------------------------------------------- bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    let bindings = bindgen::Builder::default()
        .header(include_dir.join("llama.h").to_string_lossy())
        .header(include_dir.join("mtmd.h").to_string_lossy())
        .header(include_dir.join("mtmd-helper.h").to_string_lossy())
        // Keep the generated surface close to what the CLI actually uses;
        // a full dump is thousands of lines of unused declarations.
        .allowlist_function("llama_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("MTMD_.*")
        .allowlist_var("GGML_.*")
        // C enums must stay plain integers: llama.cpp returns and accepts them
        // as ints, and a Rust enum would be unsound for values this header
        // version does not know about. bindgen's default (a type alias plus
        // constants) is exactly what we want, so enums are left at default.
        //
        // Parse as C, not C++: `mtmd-helper.h` guards a
        // `namespace mtmd_helper { ... }` block behind `#ifdef __cplusplus`.
        // Compiling as C++ makes bindgen emit it as well, producing duplicate
        // definitions (the C++ and C overloads share names) and C++-only types
        // such as `_NodeHandle`. Everything we call is in the `extern "C"`
        // block, which both language modes expose.
        .clang_arg("-I")
        .clang_arg(include_dir.to_string_lossy())
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg("-std=c11")
        .generate()
        .expect("bindgen failed to generate bindings from vendor/include");

    bindings
        .write_to_file(&out_path)
        .expect("could not write bindings.rs");
}
