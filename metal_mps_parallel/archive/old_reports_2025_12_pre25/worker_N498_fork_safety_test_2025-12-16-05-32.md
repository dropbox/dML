# Worker N=498 - Fork Safety Regression Test

Date: 2025-12-16

## Summary

Added a repo-level regression test that validates the MPS "bad fork" guard: after `fork()`, the child process must have `torch.mps._is_in_bad_fork()` return True.

## Key Finding

During implementation, discovered that PyTorch's MPS backend lacks the `_lazy_init()` guard that CUDA has. When `_is_in_bad_fork()` is true:
- **CUDA**: Raises `RuntimeError("Cannot re-initialize CUDA in forked subprocess...")`
- **MPS**: Crashes with SIGSEGV when attempting to use Metal resources

The test was modified to only verify the bad_fork flag is correctly set (which our fork handler does correctly), rather than testing graceful exception handling (which is a pre-existing PyTorch limitation).

## Verification (N=498)

- Fork safety test: PASS
- Full test suite (12 tests): 12/12 PASS
- Run on machine with Metal access

