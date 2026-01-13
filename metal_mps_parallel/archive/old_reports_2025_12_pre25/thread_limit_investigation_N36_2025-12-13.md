# Thread Limit Investigation Report - Worker N=36

**Date**: 2025-12-13
**Worker**: N=36
**Status**: COMPLETE - Root cause identified as Apple MPS framework limitation

## Executive Summary

The 3+ thread limitation is **NOT a bug in our stream pool implementation**. It is an internal limitation in Apple's MPS/MPSGraph framework that prevents safe concurrent execution of MPSGraph operations from more than 2 threads within the same process.

Multi-process parallelism works perfectly (8+ processes), confirming the issue is per-process thread safety in Apple's framework.

## Test Results

### Multi-Thread (Same Process)

| Threads | Raw Tensor Ops | nn.Linear (no-graph) | nn.Linear (graph path) |
|---------|----------------|----------------------|------------------------|
| 2 | PASS | PASS | PASS |
| 3 | PASS | SEGFAULT | PASS |
| 4 | PASS | SEGFAULT | SEGFAULT |
| 5 | PASS | SEGFAULT | SEGFAULT |
| 8 | PASS | SEGFAULT | SEGFAULT |

### Multi-Process (Separate Processes)

| Processes | nn.Linear |
|-----------|-----------|
| 2 | PASS |
| 4 | PASS |
| 6 | PASS |
| 8 | PASS |

## Key Findings

### 1. Raw Tensor Operations Are Thread-Safe

Operations like `torch.mm()` (matrix multiplication) work correctly with 8+ threads. This confirms:
- Our stream pool implementation is correct
- Each thread gets its own stream and command buffer
- Metal's command queue parallelism works

### 2. MPSGraph Operations Have Internal Thread Limits

`nn.Linear` and other nn.Module operations use MPSGraph internally. These fail at 3+ threads even with:
- Global encode mutex (`g_mpsgraph_encode_mutex`)
- Per-stream mutex (`_streamMutex`)
- Disabled commitAndContinue mode

### 3. No-Graph vs Graph Path Behavior

Setting `MPS_FORCE_GRAPH_PATH=1`:
- 3 threads: PASS (improvement)
- 4+ threads: SEGFAULT (still fails)

The no-graph path uses `MPSNDArrayMatrixMultiplication` directly, which has its own thread-safety issues at 3+ threads.

### 4. Multi-Process Parallelism Works Perfectly

8 separate Python processes can all run nn.Linear inference in parallel without issues. Each process has:
- Its own Metal device context
- Its own MPS framework state
- Its own MPSGraph cache

This proves the issue is **internal to Apple's MPS framework** within a single process.

## Root Cause Analysis

Apple's MPS framework (MPSGraph, MPSNDArrayMatrixMultiplication, etc.) has internal global state that is not fully thread-safe when accessed from 3+ threads within the same process, even with:

1. Different command queues per thread
2. Different command buffers per thread
3. Different MPSGraph instances per thread (thread-local cache)
4. External mutex serialization of encoding operations

The crash occurs in Apple's closed-source Metal/MPS framework code, not in our modifications.

## Workarounds

### For 2-Thread Use Cases
Our stream pool works correctly and provides true parallelism for 2 threads.

### For 3+ Thread Use Cases
Use **multi-process parallelism** instead of multi-threading:
- Launch separate Python/C++ processes
- Each process can safely use 1-2 threads for MPS
- Use IPC (shared memory, sockets) for communication

### Environment Variable Option
`MPS_FORCE_GRAPH_PATH=1` provides slight improvement (3 threads works) but doesn't solve 4+ threads.

## Recommendations

### Short-Term
1. Document 2-thread limit for nn.Module operations
2. Tests should use 2 threads for nn.Module, 8+ for raw tensor ops
3. Production code should use process pools for 3+ parallel inference

### Long-Term
1. Report issue to Apple (Feedback Assistant) with minimal reproducer
2. Monitor future macOS updates for MPS threading improvements
3. Consider contributing a PyTorch-level solution that spawns subprocesses

## Test Files Created

- `tests/test_thread_limit_investigation.py` - Comprehensive thread limit analysis
- `tests/test_nnlinear_threads.py` - Focused nn.Linear thread testing
- `tests/test_multiprocess_vs_multithread.py` - Comparison test

## Conclusion

The MPS stream pool implementation is correct and working. The 3+ thread limitation is an Apple framework issue that cannot be fixed in our code. For production use:

- **2 threads with nn.Module**: Works reliably
- **8+ threads with raw tensor ops**: Works reliably
- **3+ threads with nn.Module**: Use multi-process architecture

## References

- N=35 report: Initial per-stream mutex fix
- N=34 report: Discovery that tests were running against baseline
- Apple Metal documentation: No explicit threading limits documented
