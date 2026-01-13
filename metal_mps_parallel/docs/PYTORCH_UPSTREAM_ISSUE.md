# PyTorch Upstream: MPS Multi-Threading Serialization Issue

**Created by Andrew Yates**
**Date**: 2025-12-20
**Status**: Draft for PyTorch issue

---

## Summary

MPS backend has additional serialization beyond `torch.mps.synchronize()`. Raw Metal API achieves 62% efficiency at 8 threads, but PyTorch MPS achieves only 34% even with proper event-based synchronization. This indicates internal serialization in MPS or PyTorch's MPS backend.

**Key Finding**: The bottleneck is in MPS/PyTorch, NOT Apple's Metal driver.

---

## Issue Description

### Current Behavior

`torch.mps.synchronize()` synchronizes ALL pending GPU operations across ALL command queues/streams, not just the operations submitted by the calling thread.

In multi-threaded code, this creates cross-thread barriers:

```
Thread 1: compute → sync → wait(ALL streams) → continue
Thread 2: compute → sync → wait(ALL streams) → continue
Thread 3: compute → sync → wait(ALL streams) → continue
```

Each thread's synchronization blocks until ALL other threads' GPU work completes, serializing execution.

### Expected Behavior

Users expect `torch.mps.synchronize()` in one thread to wait only for that thread's GPU operations, similar to how CUDA streams work when each thread has its own stream.

### Solution

Use `torch.mps.Event()` for per-stream synchronization:

```python
def worker():
    event = torch.mps.Event(enable_timing=False)
    output = model(input)
    event.record()
    event.synchronize()  # Waits only for THIS stream
    return output
```

---

## Evidence: Raw Metal vs PyTorch MPS

**Hardware:** Apple M4 Max (40 GPU cores), macOS 15.7.3

### Raw Metal API (No PyTorch, No MPS)

Using custom Metal compute shader test (`fixes/metal_interpose/test_raw_metal.m`):

| Threads | Separate Queues | Shared Queue |
|---------|-----------------|--------------|
| 1       | 58,615 ops/s    | (baseline)   |
| 2       | 121,007 ops/s (103% eff) | 128,045 ops/s (109% eff) |
| 4       | 166,759 ops/s (71% eff)  | 209,036 ops/s (89% eff)  |
| 8       | 199,639 ops/s (43% eff)  | 291,180 ops/s (**62% eff**) |

**Raw Metal scales to 62% efficiency at 8 threads.**

### PyTorch MPS (Same Hardware)

| Threads | Device Sync | Event Sync | Raw Metal (ref) |
|---------|-------------|------------|-----------------|
| 1       | 6,065 ops/s | 6,279 ops/s | 58,615 ops/s |
| 2       | 6,460 ops/s | 12,751 ops/s | 128,045 ops/s |
| 4       | 6,948 ops/s | 15,870 ops/s | 209,036 ops/s |
| 8       | 6,672 ops/s (**14% eff**) | 17,256 ops/s (**34% eff**) | 291,180 ops/s (**62% eff**) |

**The gap: Raw Metal achieves 62% efficiency, MPS event sync achieves only 34%.**

This 28 percentage point gap proves additional serialization exists in MPS/PyTorch beyond
the `torch.mps.synchronize()` issue.

---

## Benchmarks

**Hardware:** Apple M4 Max (40 GPU cores), macOS 15.7.3, PyTorch nightly

### 512x512 FP16 Matrix Multiplication

| Threads | Device Sync (ops/s) | Event Sync (ops/s) | Improvement |
|---------|---------------------|--------------------| ----------- |
| 1       | 6,065               | 6,279              | 1.04x       |
| 2       | 6,460               | 12,751             | 1.97x       |
| 4       | 6,948               | 15,870             | 2.28x       |
| 8       | 6,672               | 17,256             | 2.59x       |

**Results:**
- At 1 thread: Both sync modes are equivalent (~1x)
- Device sync: **No scaling** (1.10x at 8 threads vs 1 thread, 14% efficiency)
- Event sync: **2.75x scaling** at 8 threads vs 1 thread (34% efficiency)
- Event sync is **2.59x faster** than device sync at 8 threads
- **But raw Metal achieves 62% efficiency** - MPS has additional overhead

### Neural Network Inference (512→1024→512 Linear)

| Method | 8-Thread Throughput | Efficiency |
|--------|---------------------|------------|
| Device sync (`torch.mps.synchronize()`) | 4,607 ops/s | 19.2% |
| Event sync (`torch.mps.Event`) | 7,065 ops/s | 29.4% |

Event sync provides **1.53x improvement** over device sync.

### Sync Frequency Impact

Synchronization frequency significantly impacts throughput. Results vary by model complexity:

| Model | Sync Every 1 | Sync Every 256 | Speedup |
|-------|--------------|----------------|---------|
| 3-layer NN (512→1024→512) | 4,450 ops/s | 10,500 ops/s | **2.4x** |
| 1-layer NN (512→512) | 8,100 ops/s | 41,300 ops/s | **5.1x** |
| Pure matmul 512x512 | 5,900 ops/s | 29,700 ops/s | **5.0x** |

**Note**: Simpler operations benefit more from reduced sync frequency. The 3-layer model used in other benchmarks sees ~2.4x improvement.

For maximum throughput, **batch operations before synchronizing** rather than syncing after every operation.

---

## Minimal Reproduction

```python
#!/usr/bin/env python3
"""
Demonstrates torch.mps.synchronize() vs torch.mps.Event for multi-threading.

Run:
    python3 mps_sync_comparison.py

Expected output shows event sync achieves ~2-3x better throughput at 8 threads.
"""

import torch
import time
from concurrent.futures import ThreadPoolExecutor

def benchmark_device_sync(num_threads: int, iters_per_thread: int = 500):
    """Benchmark using torch.mps.synchronize() - device-wide sync."""
    def worker(thread_id):
        a = torch.randn(512, 512, dtype=torch.float16, device='mps')
        b = torch.randn(512, 512, dtype=torch.float16, device='mps')
        for _ in range(iters_per_thread):
            c = torch.mm(a, b)
            torch.mps.synchronize()  # Syncs ALL streams!
        return iters_per_thread

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))
    elapsed = time.perf_counter() - start
    total_ops = sum(results)
    return total_ops / elapsed

def benchmark_event_sync(num_threads: int, iters_per_thread: int = 500):
    """Benchmark using torch.mps.Event - per-stream sync."""
    def worker(thread_id):
        a = torch.randn(512, 512, dtype=torch.float16, device='mps')
        b = torch.randn(512, 512, dtype=torch.float16, device='mps')
        event = torch.mps.Event(enable_timing=False)
        for _ in range(iters_per_thread):
            c = torch.mm(a, b)
            event.record()
            event.synchronize()  # Syncs only THIS stream
        return iters_per_thread

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))
    elapsed = time.perf_counter() - start
    total_ops = sum(results)
    return total_ops / elapsed

if __name__ == "__main__":
    print("MPS Synchronization Comparison")
    print("=" * 60)

    for n_threads in [1, 2, 4, 8]:
        device_ops = benchmark_device_sync(n_threads)
        event_ops = benchmark_event_sync(n_threads)
        ratio = event_ops / device_ops
        print(f"{n_threads} threads: device={device_ops:.0f} ops/s, "
              f"event={event_ops:.0f} ops/s, ratio={ratio:.2f}x")
```

---

## Proposed Documentation Update

Add to the MPS documentation (torch.mps.synchronize docstring or MPS page):

### Warning for Multi-Threaded Workloads

`torch.mps.synchronize()` waits for ALL pending GPU operations across all command queues, not just the calling thread's operations. For multi-threaded workloads, use `torch.mps.Event` for per-stream synchronization:

```python
# Single-threaded: OK to use device sync
def single_thread_inference():
    output = model(input)
    torch.mps.synchronize()
    return output

# Multi-threaded: Use event sync
def multi_thread_worker():
    event = torch.mps.Event(enable_timing=False)
    output = model(input)
    event.record()
    event.synchronize()  # Per-stream sync
    return output
```

### Performance Impact

| Sync Method | 8-Thread Efficiency | Use Case |
|-------------|---------------------|----------|
| `torch.mps.synchronize()` | ~13-19% | Single-threaded only |
| `torch.mps.Event.synchronize()` | ~29-35% | Multi-threaded |

---

## Related

- CUDA has explicit per-stream synchronization (cudaStreamSynchronize) vs device-wide (cudaDeviceSynchronize)
- Metal has similar semantics with MTLCommandQueue vs MTLDevice synchronization
- This is expected behavior, but should be documented for PyTorch users

---

## Filing Options

### Option 1: Documentation Issue

File issue at https://github.com/pytorch/pytorch/issues with label `module: docs`

Title: `[MPS] Document torch.mps.synchronize() device-wide semantics for multi-threaded workloads`

### Option 2: Documentation PR

Submit PR to update:
- `torch/_C/_mps/__init__.pyi` - Add docstring note
- `docs/source/notes/mps.rst` - Add multi-threading section

### Option 3: Feature Request

Request `torch.mps.stream_synchronize()` API for explicit per-stream sync (if Event API is considered workaround rather than solution).

---

## References

- This project: https://github.com/dropbox/dML/metal_mps_parallel
- Raw Metal test: `fixes/metal_interpose/test_raw_metal.m` - Proves Metal driver scales to 62% efficiency
- Analysis report: `reports/main/metal_vs_mps_serialization_N1408_2025-12-20.md`
- Guide: `docs/MPS_SYNC_MODE_GUIDE.md`

## Key Conclusion

The serialization is NOT in Apple's Metal driver. Raw Metal achieves 62% efficiency at 8 threads.
The bottleneck is in MetalPerformanceShaders (MPS) or PyTorch's MPS backend integration.
This should be investigated in PyTorch's `aten/src/ATen/mps/` code.
