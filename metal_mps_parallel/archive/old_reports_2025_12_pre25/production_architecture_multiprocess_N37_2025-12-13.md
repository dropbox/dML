# Production Architecture Guidance: Multi-Process MPS Inference (N=37)

**Date**: 2025-12-13  
**Context**: Phase 15 thread-limit root cause (`reports/main/thread_limit_investigation_N36_2025-12-13.md`)

## Summary

Apple's MPS/MPSGraph framework is not thread-safe for **3+ concurrent `nn.Module` executions** within a single process. For applications that need >2 concurrent model inferences, the practical workaround is **multi-process parallelism** (process pool), not multi-threading.

This repo’s MPS stream pool fixes PyTorch’s single-stream bottleneck and supports high thread counts for **raw tensor ops** (e.g., `torch.mm`), but it cannot make Apple’s closed-source MPSGraph thread-safe at 3+ threads.

## Recommended Architecture (High Level)

**Goal**: Concurrency via *processes*, not *threads*.

1. Run **N worker processes** (N ≈ required concurrency / desired throughput).
2. Each worker process:
   - Initializes MPS/Metal **inside the worker** (do not share MPS state across processes).
   - Loads the model once and keeps it resident on `mps`.
   - Serves requests in a loop (RPC/IPC) or via a process pool API.
3. The parent process:
   - Routes requests to workers (round-robin, least-loaded, or queue-based).
   - Collects results and performs any CPU-side postprocessing.

## Practical Patterns

### Python

**Recommended**: `concurrent.futures.ProcessPoolExecutor` with an `initializer` that loads the model once per process.

Key points:
- Use `spawn` start method explicitly (`multiprocessing.get_context("spawn")`).
- Keep worker processes **persistent** (avoid per-request `subprocess` spawning).
- Avoid sending MPS tensors across processes. Send CPU data (bytes/NumPy) and move to `mps` inside the worker.
- Control CPU oversubscription per process:
  - `torch.set_num_threads(1)`
  - `torch.set_num_interop_threads(1)`
  - Optionally set env vars (`OMP_NUM_THREADS=1`, etc.) depending on your stack.

Example implementation: `tests/multiprocess_inference_pool.py`

### C++ / libtorch

**Recommended**: a worker-process model (one `libtorch`+MPS instance per process), with IPC between a coordinator and workers.

Key points:
- Prefer **exec/spawn** of worker executables over `fork()` after any MPS/Metal initialization.
- Use an IPC mechanism appropriate for payload sizes and latency:
  - Unix domain sockets for request/response
  - Shared memory (plus a control socket) for large tensors/audio buffers
- Keep models loaded and warmed up in each worker to amortize compilation/caching costs.

## Tradeoffs / Sizing

- **Memory**: each process has its own MPS state and model weights on GPU, so memory scales ~linearly with process count.
- **Throughput**: scaling improves with more processes until GPU saturation.
- **Latency**: persistent workers reduce tail latency compared to spawning processes per request.

## Operational Guidance

- If you only need **2 concurrent inferences**: threads may be sufficient (single process).
- If you need **3+ concurrent inferences** of `nn.Module`: use a **process pool**.
- If your workload is **raw tensor ops** (no MPSGraph): high thread counts can work in-process (still bounded by GPU saturation).

## Verified Results (N=38)

**Test**: `tests/multiprocess_inference_pool.py`
**Configuration**: 4 worker processes, 16 tasks, 10 iterations/task
**Model**: MLP (256→512→256)

```
workers: 4  tasks: 16  iters/task: 10  total iters: 160
model: mlp  batch: 4  in_features: 256  out_features: 256
wall time: 0.901s  throughput: 177.5 forwards/s
mean task time: 0.003s
```

**Conclusion**: Multi-process parallelism works reliably with 4+ concurrent model instances, bypassing Apple's 2-thread limitation for nn.Module operations.
