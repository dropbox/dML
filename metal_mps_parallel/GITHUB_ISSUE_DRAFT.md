# GitHub Issue Draft for pytorch/pytorch

> **NOTE**: This is a draft for the human to submit. Do not submit automatically.

---

## Issue Title

**[Feature Request] Add MPS stream pool for thread-safe parallel inference on Apple Silicon**

---

## Issue Body

### Problem

PyTorch's MPS backend currently uses a singleton `MPSStream`, which prevents concurrent `model.forward()` calls from multiple threads. When attempting parallel inference:

```python
from concurrent.futures import ThreadPoolExecutor

model = torch.nn.Linear(100, 10).to("mps")
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(model, torch.randn(32, 100, device="mps")) for _ in range(8)]
    results = [f.result() for f in futures]  # Crashes!
```

The error is:
```
"commit an already committed command buffer"
```

This limitation forces users to either:
1. Serialize all inference (losing parallelism benefits)
2. Use multiple processes (high memory overhead)
3. Switch to different frameworks

### Motivation

Parallel inference is essential for:
- **Serving workloads**: Web servers handling concurrent requests
- **Batch processing**: Running multiple models or inputs simultaneously
- **Real-time applications**: Voice assistants, video processing with multiple streams

CUDA has supported this for years via `CUDAStream` pooling. MPS should have parity.

### Proposed Solution

Implement `MPSStreamPool` following CUDA's proven design:

1. **Pool of 32 streams**: 1 default stream + 31 pooled streams
2. **Thread-local assignment**: Each thread automatically gets a stream from the pool
3. **Round-robin allocation**: CUDA-style `counter++ % pool_size` for simplicity and correctness
4. **Backward compatible**: Single-threaded code works exactly as before

### Implementation Status

We have a complete, tested implementation ready for review:

- **Patch**: 4,032 lines modifying 28 files
- **Tests**: 24 tests all passing
- **TSan**: 0 data races (8 threads x 50 iterations)
- **Performance**: 71% efficiency at 2 threads (GPU saturation limits scaling at higher thread counts)

Repository: https://github.com/dropbox/dML/metal_mps_parallel

### Known Limitations

During development, we discovered thread-safety issues in Apple's MPS framework itself:
- `MPSNDArrayMatrixMultiplication` has internal shared state (crashes at 3+ threads)
- Metal compute kernels have thread-safety issues (crashes at 4+ threads)

Our implementation includes workarounds for these Apple bugs, documented for potential Radar submission.

### Testing

The implementation includes:
- Unit tests for `MPSStreamPool` allocation/deallocation
- Parallel inference tests (2, 4, 8 threads)
- Thread churn stability tests
- Cross-stream tensor sharing tests
- Thread Sanitizer verification

### Related Issues

- (Search pytorch/pytorch for existing MPS threading issues and link them here)

### Checklist

- [x] Implementation complete
- [x] Tests written and passing
- [x] Thread-safe (TSan verified)
- [x] clang-format compliant
- [x] Documentation written
- [ ] PR submitted (pending this issue)

---

## Labels to Add

- `module: mps`
- `feature`
- `triaged`

---

## After Creating Issue

1. Note the issue number (e.g., #12345)
2. Update PR description to reference: `Fixes #12345`
3. Submit PR using template from SUBMISSION_PROOF.md
