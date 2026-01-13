# Hole 1 Evidence: Is the Bottleneck Really the Metal Driver?

**Date:** 2025-12-20  
**Hardware/OS:** Apple M4 Max (40 GPU cores), macOS 15.7.3 (24G419)  
**Metal visibility:** ✅ (`./tests/metal_diagnostics.sh` reports 1 device, non-nil)

## Executive Summary

The repo’s story currently asserts an “Apple Metal driver efficiency ceiling (13–30% at 8 threads)”. This report attempted to **prove or falsify** that claim with:

1. **Instruments (Metal System Trace)** on a minimal PyTorch MPS workload  
2. **A pure Metal repro** with per-thread command queues and per-call timing

**Main finding:** the strongest serialization point we can *conclusively* demonstrate is **not the driver**, but **device-wide synchronization semantics**:

- `torch.mps.synchronize()` is explicitly **device-wide** (“all kernels in all streams”), implemented by synchronizing **all active streams** (`MPSStreamPool::synchronizeAllStreams()`).
- Calling it in each worker thread (as many benchmarks do) forces cross-thread barriers that can destroy scaling for small/medium workloads.

For large matmuls, scaling remains limited even with per-stream synchronization, consistent with **GPU saturation** rather than a driver-side “mutex ceiling”.

## Key Evidence

### 1) `torch.mps.synchronize()` is device-wide (cross-thread barrier)

- Python API documents device-wide behavior: `pytorch-mps-fork/torch/mps/__init__.py`
- C++ implementation explicitly synchronizes all streams: `pytorch-mps-fork/aten/src/ATen/mps/MPSHooks.mm` (`deviceSynchronize`)
- Stream-pool API makes this explicit: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h` (`synchronizeAllStreams`)

**Practical impact:** any benchmark that does per-iteration `torch.mps.synchronize()` in each thread is measuring a globally-synchronized execution model, which strongly limits throughput scaling.

### 2) Per-stream `Event.synchronize()` materially improves scaling for small matmul

`tests/profile_metal_trace.py` now supports:
- `--sync-mode device`: uses `torch.mps.synchronize()` (device-wide)
- `--sync-mode event`: uses per-thread `torch.mps.Event().record(); event.synchronize()`

Observed on matmul `512x512` (FP16):
- **Device sync:** ~`6.3k ops/s` at 1 thread → ~`6.6k ops/s` at 8 threads (~1.04x)
- **Event sync:** ~`6.3k ops/s` at 1 thread → ~`18.3k ops/s` at 8 threads (~2.9x)

This directly falsifies the idea of a universal “Metal driver ceiling” for this workload shape; the prior ceiling was primarily a synchronization artifact.

### 3) Instruments trace export: command buffer submissions are per-thread, but throughput is flat under device sync

Traces captured (Metal System Trace):
- `reports/traces/mps_matmul_2048_fp16_1t.trace`
- `reports/traces/mps_matmul_2048_fp16_8t.trace`

Exported summary:
- `reports/main/metal_system_trace_summary_2025-12-20.json` (generated via `tests/summarize_metal_system_trace.py`)

Notable points (device-sync runs):
- Command buffer submissions occur on **multiple Python worker threads** in the 8-thread run (not a single “submission thread” bottleneck).
- Despite that, **submissions/sec are ~flat** (~510/s @ 1T vs ~526/s @ 8T), consistent with “GPU can only complete ~this many of these command buffers per second under this synchronization model”.

### 4) Pure Metal repro scales substantially better than PyTorch matmul for comparable per-command-buffer GPU time

Pure Metal timing repro:
- Source: `tests/metal_pure_objc_repro/main.mm`
- Runner: `tests/metal_api_timing.py`
- Example JSON output: `reports/main/metal_api_timing_2025-12-20_axpy12m.json`

With a single-kernel command buffer sized to yield ~1ms GPU time per command buffer, the pure Metal repro achieves **~4x throughput speedup at 8 threads** (≈50% efficiency), indicating:
- Metal can accept parallel submissions with per-thread command queues
- A hard “13–30% ceiling” is **not intrinsic** to Metal submission itself

## Conclusion (Hole 1 Status)

**We did not find evidence supporting a general “Metal driver bottleneck ceiling” as currently claimed.** The strongest verified contributors to “no scaling” are:

1. **Device-wide sync semantics** (`torch.mps.synchronize()`), which many benchmarks use per-iteration in each thread
2. **GPU saturation** for large kernels (matmul 2048), where even per-stream sync yields limited speedup

If the repo’s public narrative continues to claim “the bottleneck is the Metal driver”, it should be revised to reflect:
- which workloads were tested,
- which synchronization model was used (device vs per-stream),
- and whether the observed limit is GPU saturation vs a driver serialization point.

## Repro Commands

Preflight:
```bash
./tests/metal_diagnostics.sh
```

Per-stream sync vs device sync comparison:
```bash
python3 tests/profile_metal_trace.py --op matmul --size 512 --dtype float16 --threads 8 --iters 2000 --sync-mode device
python3 tests/profile_metal_trace.py --op matmul --size 512 --dtype float16 --threads 8 --iters 2000 --sync-mode event
```

Metal System Trace capture (note: xctrace requires an absolute python path under Instruments):
```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace record \
  --template "Metal System Trace" \
  --output reports/traces/mps_parallel.trace \
  --launch -- /opt/homebrew/bin/python3 tests/profile_metal_trace.py --threads 8 --iters 200 --size 2048 --sync-mode device
```

Summarize trace to JSON:
```bash
python3 tests/summarize_metal_system_trace.py --trace reports/traces/mps_parallel.trace
```

Pure Metal call timing sweep:
```bash
python3 tests/metal_api_timing.py --threads 1,2,4,8 --iters 30 --elements 12000000
```

