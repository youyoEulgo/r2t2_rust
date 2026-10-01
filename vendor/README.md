# vendor/ — llama.cpp build products

These are **tracked on purpose**. They are what lets the project build and run
without a llama.cpp checkout, and they carry the CPU-targeting decisions that
took real effort to get right.

## What is here

```
include/    C headers for libllama, libmtmd and ggml
lib/        the shared libraries themselves
```

`build.rs` feeds `include/` to bindgen and links against `lib/`.

| | |
|---|---|
| llama.cpp | 0.4.0 (commit `ad6c66839af3c5646fba8c6c2e2087a1e4e38948`) |
| ggml | 0.23.0 |
| CUDA toolkit | 13.4 |
| compiler | GNU 16.2.1 |
| CUDA arch | `sm_89` (Ada Lovelace) |
| platform | Linux x86_64 |

## Why they were rebuilt locally

The upstream project ships prebuilt CUDA artifacts, and they **crash with
`Illegal instruction` on this class of CPU**.

They were compiled with `-march=native` on a machine that had AVX-512, so the
binary contains instructions such as `vpternlogd` (AVX512F only). Intel removed
AVX-512 from its 12th- and 13th-generation consumer parts, so on an i9-13900HX
every GGUF load died at the first instruction the CPU did not recognise.

Rebuilding on the target machine fixes it: `-march=native` then adapts to what
that CPU actually has, keeping AVX-VNNI and GFNI while omitting AVX-512. After
the rebuild the libraries contain **zero** AVX512-only instructions.

### The tradeoff this encodes

Building with `-march=native` means these libraries run **only on CPUs at least
as capable as the build machine**. That is fine for a local build and wrong for
a general release. To ship binaries more widely, rebuild with

```sh
-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON
```

or use `-DGGML_CPU_ALL_VARIANTS=ON`, which packs several CPU variants into one
library and selects at runtime by CPUID.

## rpath

Every `.so` here uses a **relative** rpath (`$ORIGIN`), so the tree keeps working
after being moved or copied. An absolute rpath into the build directory works
only while that directory exists — removing it breaks every load with no
obvious cause.

The extension in the parent crate resolves its libraries with
`$ORIGIN/../bin`, matching the layout the upstream project ships.

## Rebuilding

```sh
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout ad6c66839af3c5646fba8c6c2e2087a1e4e38948

cmake -S /path/to/r2t2_llama -B build \
    -DLLAMA_CPP_DIR="$PWD/third_party/llama.cpp" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

cp -a build/bin/lib*.so* vendor/lib/
```

Then confirm the rpath is relative:

```sh
objdump -x vendor/lib/libllama.so.0.4.0 | grep -E 'RUNPATH|RPATH'
```

Headers come from the same checkout:

```sh
cp third_party/llama.cpp/include/*.h            vendor/include/
cp third_party/llama.cpp/ggml/include/*.h       vendor/include/
cp third_party/llama.cpp/tools/mtmd/*.h         vendor/include/
```

## Runtime requirements

| library | provided by |
|---|---|
| `libcublas`, `libcudart` | system CUDA toolkit (`/opt/cuda`) |
| `libcuda.so.1` | the NVIDIA driver — **not** redistributable |
| `libggml*`, `libllama`, `libmtmd` | this directory |

So the NVIDIA driver is the one thing a deployment cannot bundle.
