# User Guide: Parallel MPS Inference

**Version**: 1.3
**Date**: 2025-12-20

## Overview

This patch enables thread-safe parallel PyTorch MPS inference on Apple Silicon. Multiple threads can run forward() passes concurrently without crashes or data corruption.

**IMPORTANT**: For maximum multi-threaded performance, use per-stream event synchronization instead of `torch.mps.synchronize()`. See `docs/MPS_SYNC_MODE_GUIDE.md` for details.

## Quick Start

```python
import torch
import threading

# Verify MPS is available
assert torch.backends.mps.is_available()

# Create model on MPS
model = torch.nn.Linear(256, 128).to('mps')
model.eval()

# Run inference from multiple threads with EVENT sync
def worker(thread_id):
    # Create thread-local event for per-stream synchronization
    event = torch.mps.Event(enable_timing=False)

    x = torch.randn(32, 256, device='mps')
    with torch.no_grad():
        y = model(x)

    # Use event sync (NOT torch.mps.synchronize() which syncs ALL threads)
    event.record()
    event.synchronize()
    return y

# Safe to run from multiple threads!
threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Thread Safety Contract

### Safe Operations

| Operation | Thread Safety |
|-----------|---------------|
| `torch.randn()` on MPS | Thread-safe |
| `model.forward()` / `model(x)` | Thread-safe |
| `torch.matmul()` | Thread-safe |
| `torch.mps.synchronize()` | Thread-safe but **syncs ALL streams** - avoid in multi-threaded code |
| `torch.mps.Event.synchronize()` | Thread-safe, per-stream - **preferred for multi-threading** |
| `torch.mps.empty_cache()` | Thread-safe |
| Tensor creation on MPS | Thread-safe |
| Tensor operations on MPS | Thread-safe |

### Limitations

| Operation | Limit | Notes |
|-----------|-------|-------|
| Concurrent threads | 31 max | Pool has 32 slots, 1 reserved |
| TransformerEncoderLayer | 2-4 threads | Apple Metal LayerNorm limitation |

## Configuration

### Environment Variables

```bash
# Set stream pool size (default: 32)
export MPS_STREAM_POOL_SIZE=16

# Enable verbose logging
export PYTORCH_MPS_DEBUG=1
```

### Pool Exhaustion Modes

When more than 31 threads request streams:

```python
import os

# Mode 1: Throw exception (default)
# Thread gets immediate error if no stream available

# Mode 2: Backpressure (infinite wait)
os.environ["MPS_POOL_BACKPRESSURE"] = "1"
# Thread waits until stream becomes available

# Mode 3: Finite timeout
os.environ["MPS_POOL_TIMEOUT_MS"] = "5000"  # 5 second timeout
# Thread waits up to timeout, then gets error
```

## Best Practices

### 1. Use `torch.no_grad()` for Inference

```python
def inference_worker(model, data):
    with torch.no_grad():  # Disable gradient computation
        return model(data)
```

### 2. Use Event Sync in Multi-Threaded Code

```python
def worker():
    event = torch.mps.Event(enable_timing=False)
    y = model(x)
    event.record()
    event.synchronize()  # Wait for THIS thread's GPU operations
    result = y.cpu()  # Now safe to transfer to CPU
    return result

# For single-threaded code, torch.mps.synchronize() is fine:
def single_thread_inference():
    y = model(x)
    torch.mps.synchronize()  # OK for single-threaded
    return y.cpu()
```

### 3. Create Separate Model Per Thread (for Large Models)

```python
def worker(thread_id):
    # Each thread has its own model copy
    local_model = create_model().to('mps')
    local_model.eval()

    for data in get_data():
        result = local_model(data)
        process(result)
```

### 4. Monitor Memory Usage

```python
import torch

# Check current allocation
mem = torch.mps.current_allocated_memory()
print(f"MPS memory: {mem / 1e6:.1f} MB")

# Clear unused memory
torch.mps.empty_cache()
```

## Common Patterns

### Pattern 1: Thread Pool with Shared Model

```python
from concurrent.futures import ThreadPoolExecutor
import threading

model = torch.nn.Linear(256, 128).to('mps')
model.eval()

# Thread-local storage for events
_local = threading.local()

def inference(batch):
    # Get or create thread-local event
    if not hasattr(_local, 'event'):
        _local.event = torch.mps.Event(enable_timing=False)

    with torch.no_grad():
        x = batch.to('mps')
        y = model(x)
        _local.event.record()
        _local.event.synchronize()  # Per-stream sync for parallelism
        return y.cpu()

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(inference, batches))
```

### Pattern 2: Batch Processing with Workers

```python
import queue
import threading

work_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    model = torch.nn.Linear(256, 128).to('mps')
    model.eval()
    # Create thread-local event
    event = torch.mps.Event(enable_timing=False)

    while True:
        batch = work_queue.get()
        if batch is None:
            break
        with torch.no_grad():
            y = model(batch.to('mps'))
            event.record()
            event.synchronize()  # Per-stream sync
            result_queue.put(y.cpu())
        work_queue.task_done()

# Start workers
threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()

# Submit work
for batch in batches:
    work_queue.put(batch)

# Wait for completion
work_queue.join()

# Stop workers
for _ in threads:
    work_queue.put(None)
for t in threads:
    t.join()
```

## Troubleshooting

### "Pool exhausted" Error

**Cause**: More than 31 threads trying to acquire streams simultaneously.

**Solutions**:
1. Reduce thread count to ≤31
2. Enable backpressure mode: `MPS_POOL_BACKPRESSURE=1`
3. Use thread pooling to limit concurrency

### TransformerEncoderLayer Crashes at 4+ Threads

**Cause**: Apple Metal framework limitation with LayerNorm operations.

**Solutions**:
1. Use ≤2 threads for Transformer models
2. Use batching instead of parallel threads
3. Consider CPU for highly parallel Transformer inference

### Memory Growth Over Time

**Cause**: Tensors not being properly released.

**Solutions**:
```python
# Explicitly delete tensors when done
del tensor
torch.mps.empty_cache()

# Use context managers
with torch.no_grad():
    # ... operations ...
    pass  # Tensors released after block
```

### Hangs or Deadlocks

**Cause**: Synchronization issues or pool exhaustion without timeout.

**Solutions**:
1. Use `torch.mps.Event.synchronize()` for multi-threaded code (per-stream)
2. Only use `torch.mps.synchronize()` in single-threaded code (device-wide)
3. Set a timeout: `MPS_POOL_TIMEOUT_MS=10000`
4. Check thread count isn't exceeding pool size

## Performance Tuning

### Sync Frequency is Critical

**Synchronization frequency significantly impacts throughput.** Results vary by model complexity:

| Model Complexity | Sync Every 1 | Sync Every 256 | Speedup |
|------------------|--------------|----------------|---------|
| 3-layer NN (benchmark) | 4,450 ops/s | 10,500 ops/s | **2.4x** |
| 1-layer NN | 8,100 ops/s | 41,300 ops/s | **5.1x** |
| Pure matmul | 5,900 ops/s | 29,700 ops/s | **5.0x** |

**Note**: Simpler operations benefit more from reduced sync frequency.

```python
# BAD - sync after every operation
for x in inputs:
    y = model(x)
    torch.mps.synchronize()  # ~4,500 ops/s (3-layer model)

# BETTER - accumulate then sync
results = []
for i, x in enumerate(inputs):
    results.append(model(x))
    if (i + 1) % 32 == 0:  # Sync every 32 ops
        torch.mps.synchronize()
        # Process results batch here
# Final sync
torch.mps.synchronize()  # ~8,000 ops/s (1.8x faster)

# BEST - use batching
batched_input = torch.stack(inputs)
result = model(batched_input)  # One kernel call
torch.mps.synchronize()  # ~10,500 ops/s (2.4x faster)
```

### Finding Optimal Thread Count

```python
import time

def benchmark_threads(model, data, thread_counts):
    results = {}

    for n_threads in thread_counts:
        start = time.perf_counter()
        threads = [threading.Thread(target=inference, args=(model, data))
                   for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        results[n_threads] = elapsed

    return results

# Typical findings:
# - Small workloads: Peak at 4-8 threads
# - Large workloads: Peak at 1-2 threads (GPU saturated)
```

### GPU Saturation Detection

If adding more threads doesn't increase throughput, the GPU is saturated:

| Workload | Optimal Threads | Reason |
|----------|-----------------|--------|
| 128x128 matmul | 4-8 | CPU-bound, GPU underutilized |
| 512x512 matmul | 1-2 | GPU saturates at low thread count |
| 1024x1024 matmul | 1 | GPU fully utilized by single thread |

## API Reference

### Thread-Safe Functions

```python
# Create tensor on MPS
torch.randn(size, device='mps')
torch.zeros(size, device='mps')
torch.ones(size, device='mps')
tensor.to('mps')

# Operations
torch.matmul(a, b)
torch.nn.functional.*  # Most operations
model(x)  # Forward pass

# Synchronization - IMPORTANT DISTINCTION:
event = torch.mps.Event(enable_timing=False)  # Create per-thread event
event.record()  # Record completion point on stream
event.synchronize()  # Wait for THIS stream only (use in multi-threaded code)

torch.mps.synchronize()  # Waits for ALL streams (use in single-threaded code only)
torch.mps.empty_cache()  # Release unused memory

# Query
torch.mps.current_allocated_memory()
torch.mps.is_available()
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.3 | 2025-12-20 | Corrected sync frequency numbers - 2.4x for benchmark model (not 6.8x) |
| 1.2 | 2025-12-20 | Added sync frequency section |
| 1.1 | 2025-12-20 | Updated to use event sync for multi-threaded examples (critical for performance) |
| 1.0 | 2025-12-19 | Initial release |

## Support

For issues and feature requests:
- GitHub: https://github.com/dropbox/dML/metal_mps_parallel
