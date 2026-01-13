#!/usr/bin/env python3
"""
Test: Large Workload Efficiency Validation

Validates that scaling efficiency meets the 50%+ target for appropriately-sized
workloads. Small workloads saturate the GPU early and show low efficiency;
this test uses large workloads where CPU/software bottlenecks would be visible.

Pass criteria:
- 2-thread efficiency >= 50% for linear (2048x2048, batch=64)

Reference: MPS_PARALLEL_INFERENCE_PLAN.md - "Measured large-workload point (N=161)"
"""

import subprocess
import sys
import json

def run_efficiency_benchmark(threads: int, iterations: int = 20, batch: int = 64,
                              in_features: int = 2048, out_features: int = 2048):
    """Run benchmark and return ops/s."""
    code = f'''
import torch
import torch.nn as nn
import threading
import time
import json

torch.zeros(1, device="mps")
torch.mps.synchronize()

results = []
lock = threading.Lock()

def worker(tid):
    model = nn.Linear({in_features}, {out_features}).to("mps")
    model.eval()
    for i in range({iterations}):
        x = torch.randn({batch}, {in_features}, device="mps")
        with torch.no_grad():
            _ = model(x)
            torch.mps.synchronize()
        with lock:
            results.append(1)

threads_list = [threading.Thread(target=worker, args=(i,)) for i in range({threads})]
start = time.perf_counter()
for t in threads_list: t.start()
for t in threads_list: t.join()
total_time = time.perf_counter() - start

ops_per_sec = len(results) / total_time if total_time > 0 else 0
print(json.dumps({{"ops_per_sec": ops_per_sec, "total_ops": len(results)}}))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        return {"ops_per_sec": 0, "error": f"Exit {result.returncode}"}
    except Exception as e:
        return {"ops_per_sec": 0, "error": str(e)}


def test_large_workload_efficiency():
    """
    Test that 2-thread efficiency >= 50% for large linear workload.

    Small workloads (batch=4, 256x128) saturate the GPU quickly, giving
    low efficiency numbers. Large workloads reveal software bottlenecks.
    """
    print("Testing large workload efficiency (2048x2048, batch=64)...")

    # Run baseline
    r1 = run_efficiency_benchmark(threads=1)
    assert r1.get("ops_per_sec", 0) > 0, f"1-thread baseline failed: {r1}"
    baseline = r1["ops_per_sec"]
    print(f"  1 thread: {baseline:.0f} ops/s")

    # Run 2-thread
    r2 = run_efficiency_benchmark(threads=2)
    assert r2.get("ops_per_sec", 0) > 0, f"2-thread failed: {r2}"
    ops_2t = r2["ops_per_sec"]
    print(f"  2 threads: {ops_2t:.0f} ops/s")

    # Calculate efficiency
    speedup = ops_2t / baseline
    efficiency = speedup / 2 * 100
    print(f"  Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%")

    # Assert efficiency >= 50%
    assert efficiency >= 50.0, (
        f"2-thread efficiency {efficiency:.1f}% < 50% target. "
        f"This suggests a software bottleneck (not GPU saturation). "
        f"Speedup={speedup:.2f}x, baseline={baseline:.0f}, 2t={ops_2t:.0f}"
    )
    print(f"  PASS: {efficiency:.1f}% >= 50%")


def main():
    """Run efficiency validation test."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("SKIP: MPS not available")
            return 0
    except ImportError:
        print("ERROR: PyTorch not found")
        return 2

    try:
        test_large_workload_efficiency()
        print("\nPASS: Large workload efficiency test")
        return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
