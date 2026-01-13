# MPS Synchronization Mode Guide

**Created by Andrew Yates**
**Worker N=1400**
**Date: 2025-12-20**

---

## Executive Summary

PyTorch MPS provides two synchronization mechanisms with dramatically different parallel performance:

| Sync Mode | 8-Thread Efficiency | Use Case |
|-----------|---------------------|----------|
| Device Sync | ~13-19% | Single-threaded only |
| Event Sync | ~29-35% | Multi-threaded workloads |
| Process Pool | ~96% | Maximum throughput |

**Key Insight**: `torch.mps.synchronize()` synchronizes ALL streams device-wide, creating cross-thread barriers that destroy parallelism. Use per-stream event synchronization for multi-threaded workloads.

---

## The Problem: Device-Wide Synchronization

When you call `torch.mps.synchronize()`, it waits for ALL pending GPU operations across ALL streams to complete. In a multi-threaded context:

```
Thread 1: op1 → sync → wait(all streams) → continue
Thread 2: op2 → sync → wait(all streams) → continue
Thread 3: op3 → sync → wait(all streams) → continue
```

Each thread's sync waits for ALL other threads, serializing execution.

---

## The Solution: Per-Stream Event Synchronization

Using `torch.mps.Event()` allows each thread to sync only its own operations:

```
Thread 1: op1 → event.record() → event.synchronize() → continue
Thread 2: op2 → event.record() → event.synchronize() → continue
Thread 3: op3 → event.record() → event.synchronize() → continue
```

Each thread waits only for its own operations, enabling true parallelism.

---

## Implementation Patterns

### Pattern 1: Device Sync (Single-Threaded Only)

```python
# ONLY use for single-threaded workloads
def single_threaded_inference():
    with torch.no_grad():
        output = model(input_tensor)
    torch.mps.synchronize()  # OK for single thread
    return output
```

### Pattern 2: Event Sync (Multi-Threaded)

```python
# Use for multi-threaded workloads - avoids device-wide barriers
def threaded_inference():
    # Create one event per thread
    event = torch.mps.Event(enable_timing=False)

    with torch.no_grad():
        output = model(input_tensor)

    # Sync only this thread's operations
    event.record()
    event.synchronize()
    return output
```

### Pattern 3: Process Pool (Maximum Throughput)

```python
# Use for maximum throughput - 96% efficiency at 8 workers
from fixes.process_pool.mps_process_pool import MPSProcessPool

pool = MPSProcessPool(num_workers=8)
results = pool.forward_parallel(num_requests=100)
pool.shutdown()
```

---

## Complete Multi-Threaded Example

```python
#!/usr/bin/env python3
"""Proper multi-threaded MPS inference with event synchronization."""

import torch
import torch.nn as nn
import threading
from concurrent.futures import ThreadPoolExecutor

def create_model():
    return nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).to('mps').eval()

# Create thread-local models (optional but recommended)
models = [create_model() for _ in range(8)]

def worker(thread_id: int, num_iters: int) -> int:
    """Worker with proper event synchronization."""
    model = models[thread_id]

    # Create thread-local event for per-stream sync
    event = torch.mps.Event(enable_timing=False)

    x = torch.randn(32, 512, device='mps')

    for _ in range(num_iters):
        with torch.no_grad():
            y = model(x)

        # CORRECT: Per-stream event sync
        event.record()
        event.synchronize()

        # WRONG: Device-wide sync kills parallelism
        # torch.mps.synchronize()

    return num_iters

# Run parallel inference
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(lambda tid: worker(tid, 100), range(8)))
```

---

## Performance Comparison

Tested on Apple M4 Max (40 GPU cores), macOS 15.7.3, PyTorch MPS.

### Small Matrix (512x512 FP16 matmul)

| Threads | Device Sync (ops/s) | Event Sync (ops/s) | Event/Device Ratio |
|---------|---------------------|--------------------|--------------------|
| 1       | 6,298               | 6,605              | 1.05x              |
| 2       | 6,240               | 12,749             | 2.04x              |
| 4       | 6,148               | 15,870             | 2.58x              |
| 8       | 6,342               | 17,228             | 2.72x              |

**Key observation**: At 1 thread, both sync modes are equivalent (~1x). The advantage of event sync appears at 2+ threads where device sync's cross-thread barriers destroy parallelism.

**8-thread scaling vs 1-thread baseline**: device sync 1.01x, event sync 2.61x

### Neural Network Inference (Linear 512→1024→512)

| Method                    | Ops/s   | Efficiency |
|---------------------------|---------|------------|
| Single-thread baseline    | 3,005   | 100%       |
| 8 threads + device sync   | 4,607   | 19.2%      |
| 8 threads + event sync    | 7,065   | 29.4%      |
| Process pool 8 workers    | 23,035  | 95.8%      |

---

## Decision Matrix

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| Single request, low latency | Device sync | Lowest overhead |
| Multiple threads, shared model | Event sync | 4x better than device sync |
| Maximum throughput | Process pool | Near-linear scaling |
| GPU-saturated ops (2048x2048+) | Batching | Neither sync helps at saturation |
| Latency-sensitive streaming | Event sync | Per-request sync without global barrier |

---

## Common Mistakes

### Mistake 1: Using device sync in worker threads

```python
# WRONG - kills parallelism
def worker():
    output = model(input)
    torch.mps.synchronize()  # Waits for ALL threads!
```

### Mistake 2: Not syncing at all

```python
# WRONG - may return incomplete results
def worker():
    output = model(input)
    return output  # GPU may not be done yet!
```

### Mistake 3: Sharing events across threads

```python
# WRONG - event is not thread-safe
shared_event = torch.mps.Event()

def worker():
    output = model(input)
    shared_event.record()  # Race condition!
    shared_event.synchronize()
```

---

## When Event Sync Doesn't Help

Event sync won't improve performance when:

1. **GPU is saturated**: Large matrices (2048x2048+) already use 100% GPU
2. **Compute-bound**: The bottleneck is GPU compute, not synchronization
3. **Single-threaded**: No parallelism to enable

In these cases, focus on:
- Larger batch sizes (GPU internal parallelism)
- Mixed precision (FP16)
- Model optimization

---

## Verification Commands

```bash
# Compare device vs event sync (prints scaling/efficiency)
python3 tests/mps_sync_comparison.py

# Capture Instruments trace: device-wide sync
python3 tests/profile_metal_trace.py \
    --op matmul --size 512 --dtype float16 \
    --threads 8 --iters 2000 --sync-mode device

# Capture Instruments trace: per-stream event sync
python3 tests/profile_metal_trace.py \
    --op matmul --size 512 --dtype float16 \
    --threads 8 --iters 2000 --sync-mode event

# Run complete benchmark
python3 tests/benchmark_complete.py
```

---

## References

- `reports/main/mps_sync_comparison_update_2025-12-20-15-35.md` - Updated sync comparison numbers (correct 1-thread baseline)
- `reports/main/event_sync_verification_N1399_2025-12-20.md` - Detailed benchmarks
- `reports/main/metal_bottleneck_proof_2025-12-20.md` - Analysis of the bottleneck
- `fixes/process_pool/mps_process_pool.py` - Process pool implementation
- `tests/profile_metal_trace.py` - Benchmark tool with sync mode selection
