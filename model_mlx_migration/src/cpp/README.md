# Zipformer MLX (C++)

This directory contains an in-progress C++ port of Zipformer + pruned RNN-T inference using the MLX C++ API.

## Build

Prereqs:
- macOS with MLX installed (headers + libs discoverable by CMake)
- CMake 3.20+

```bash
cmake -S src/cpp -B src/cpp/build
cmake --build src/cpp/build -j
```

## Run tests

```bash
./src/cpp/build/test_zipformer
```

## Run CLI (placeholder)

```bash
./src/cpp/build/zipformer_infer --help
```

## Status / Known Gaps

- `SelfAttention` implements basic multi-head attention (relative positional term is TODO).
- `Conv2dSubsampling` is currently a simple 4x temporal pooling + linear projection placeholder (real 2D conv subsampling TODO).
- `ZipformerEncoder` currently executes the base-scale stage only (full multi-scale U-Net structure TODO).

