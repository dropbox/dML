#!/usr/bin/env python3
"""
Test MPS stream pool wraparound behavior.

The MPS stream pool provides 32 streams (0 default + 31 pooled). Pooled streams
are recycled when threads exit, so creating more than 31 threads sequentially
should NOT exhaust the pool.
"""

import threading


def test_pool_wraparound_sequential(n_threads: int = 64) -> None:
    import torch

    # Warmup on main thread (default stream)
    _ = torch.randn(4, 4, device="mps")
    torch.mps.synchronize()

    errors = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            _ = torch.randn(4, 4, device="mps")
            torch.mps.synchronize()
            # Clear cached stream binding before thread exit (optional).
            # The function may not exist in all builds.
            if hasattr(torch.mps, 'release_current_thread_slot'):
                torch.mps.release_current_thread_slot()
        except Exception as exc:
            with lock:
                errors.append((tid, f"{type(exc).__name__}: {exc}"))

    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        t.join()
        if errors:
            break

    assert not errors, f"First failure: thread {errors[0][0]}: {errors[0][1]}"


if __name__ == "__main__":
    import torch

    print("=" * 70)
    print("MPS Stream Pool Wraparound Test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("")

    test_pool_wraparound_sequential()
    print("PASS")
