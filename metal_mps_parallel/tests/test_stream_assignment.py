#!/usr/bin/env python3
"""
Test to verify stream assignment per thread.
"""
import threading
import time

import torch
import torch.mps

def get_stream_info():
    """Get current stream info - this will help debug."""
    # Verify MPS is available
    if not torch.backends.mps.is_available():
        return "MPS not available"

    # Try to get stream info by doing a sync
    torch.mps.synchronize()
    return f"thread={threading.current_thread().name}"

def test_simple_sequential():
    """Test sequential MPS operations on main thread."""
    print("=== Sequential Test ===")
    for i in range(3):
        x = torch.randn(10, 10, device='mps')
        y = x + x
        torch.mps.synchronize()
        print(f"  Iteration {i}: OK, result sum = {y.sum().item():.2f}")
    print("Sequential test: PASS")

def test_parallel_simple():
    """Very simple parallel test - just 2 threads, 1 operation each."""
    print("\n=== Simple Parallel Test (2 threads, 1 op each) ===")

    results = []
    errors = []
    lock = threading.Lock()

    def worker(tid):
        try:
            x = torch.randn(10, 10, device='mps')
            y = x + x
            torch.mps.synchronize()
            with lock:
                results.append((tid, y.sum().item()))
        except Exception as e:
            with lock:
                errors.append((tid, str(e)))

    threads = [
        threading.Thread(target=worker, args=(0,)),
        threading.Thread(target=worker, args=(1,)),
    ]

    for t in threads:
        t.start()

    # Small delay to let threads start
    time.sleep(0.1)

    for t in threads:
        t.join(timeout=10)

    print(f"  Results: {len(results)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        for tid, err in errors:
            print(f"    Thread {tid}: {err}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print("Simple parallel test: PASS")


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print()

    test_simple_sequential()
    test_parallel_simple()
