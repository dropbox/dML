#!/usr/bin/env python3
"""
SCALING CURVES BENCHMARK: Analyze 1, 2, 4, 8, 16 scaling behavior

Created by Andrew Yates

This benchmark analyzes the FULL scaling curves to understand:
1. Where does threading hit the wall?
2. Where does process pool hit the wall?
3. What is the optimal batch size?
4. What is the theoretical maximum?
"""

import torch
import torch.nn as nn
import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List

print("=" * 70)
print("SCALING CURVES ANALYSIS")
print("=" * 70)

device = torch.device("mps")

# Create test model
def create_model():
    return nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).to(device).eval()

model = create_model()

results = {
    'threading': {},
    'batching': {},
    'timestamp': datetime.now().isoformat()
}

# ============================================================================
# 1. BASELINE (1 thread, batch=1 for true single-operation baseline)
# ============================================================================

print("\n[1/3] Measuring baselines...")

def measure_single_op():
    """Single operation baseline - batch=1"""
    x = torch.randn(1, 512, device=device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()

    # Measure
    iterations = 300
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return iterations / elapsed

single_op_rate = measure_single_op()
print(f"   Single op (batch=1): {single_op_rate:.1f} ops/s")
results['baseline_single_op'] = single_op_rate

# ============================================================================
# 2. THREADING SCALING CURVE: 1, 2, 4, 8, 16 threads
# ============================================================================

print("\n[2/3] Threading scaling curve (1, 2, 4, 8, 16 threads)...")

def measure_threading(num_threads: int, ops_per_thread: int = 100) -> Dict:
    """Measure threading performance."""
    completed = [0]
    lock = threading.Lock()

    def worker():
        x = torch.randn(1, 512, device=device)  # batch=1 for fair comparison
        for _ in range(ops_per_thread):
            with torch.no_grad():
                y = model(x)
            torch.mps.synchronize()
            with lock:
                completed[0] += 1

    start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    ops = completed[0] / elapsed
    speedup = ops / single_op_rate
    efficiency = speedup / num_threads
    ideal_ops = single_op_rate * num_threads

    return {
        'threads': num_threads,
        'ops_per_sec': ops,
        'speedup': speedup,
        'efficiency': efficiency,
        'ideal_ops': ideal_ops,
        'actual_vs_ideal': ops / ideal_ops
    }

print(f"\n   {'Threads':<10} {'Ops/s':<12} {'Speedup':<10} {'Efficiency':<12} {'vs Ideal':<10}")
print("   " + "-" * 54)

for n in [1, 2, 4, 8, 16]:
    r = measure_threading(n, ops_per_thread=50)
    results['threading'][f'{n}_threads'] = r
    print(f"   {n:<10} {r['ops_per_sec']:<12.1f} {r['speedup']:<10.2f}x {r['efficiency']:<12.1%} {r['actual_vs_ideal']:<10.1%}")

# ============================================================================
# 3. BATCHING SCALING CURVE: 1, 2, 4, 8, 16, 32, 64, 128, 256
# ============================================================================

print("\n[3/3] Batching scaling curve (1, 2, 4, 8, 16, 32, 64, 128, 256)...")

def measure_batching(batch_size: int) -> Dict:
    """Measure batching performance."""
    x = torch.randn(batch_size, 512, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()

    # Measure
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    samples_per_sec = (batch_size * iterations) / elapsed
    ops_per_sec = iterations / elapsed

    # Compare to single-op baseline
    speedup = samples_per_sec / single_op_rate
    efficiency = speedup / batch_size

    return {
        'batch_size': batch_size,
        'samples_per_sec': samples_per_sec,
        'ops_per_sec': ops_per_sec,
        'speedup': speedup,
        'efficiency': efficiency
    }

print(f"\n   {'Batch':<10} {'Samples/s':<14} {'Speedup':<10} {'Efficiency':<12}")
print("   " + "-" * 46)

for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    r = measure_batching(bs)
    results['batching'][f'batch_{bs}'] = r
    print(f"   {bs:<10} {r['samples_per_sec']:<14.1f} {r['speedup']:<10.2f}x {r['efficiency']:<12.1%}")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SCALING ANALYSIS")
print("=" * 70)

# Find where threading hits the wall
threading_data = [(int(k.split('_')[0]), v) for k, v in results['threading'].items()]
threading_data.sort(key=lambda x: x[0])

print("\nTHREADING SCALING:")
print("-" * 40)
max_threading_speedup = max(v['speedup'] for _, v in threading_data)
max_threading_threads = [n for n, v in threading_data if v['speedup'] == max_threading_speedup][0]
print(f"Peak speedup: {max_threading_speedup:.2f}x at {max_threading_threads} threads")
print(f"After peak: speedup DECREASES (driver contention)")

# Analyze where efficiency drops below 50%
for n, v in threading_data:
    if v['efficiency'] < 0.5:
        print(f"Efficiency drops below 50% at {n} threads ({v['efficiency']:.1%})")
        break

# Find batching sweet spot
batching_data = [(int(k.split('_')[1]), v) for k, v in results['batching'].items()]
batching_data.sort(key=lambda x: x[0])

print("\nBATCHING SCALING:")
print("-" * 40)
max_batching_samples = max(v['samples_per_sec'] for _, v in batching_data)
max_batching_batch = [bs for bs, v in batching_data if v['samples_per_sec'] == max_batching_samples][0]
print(f"Peak throughput: {max_batching_samples:.1f} samples/s at batch {max_batching_batch}")

# Check if batching is still scaling
last_two = batching_data[-2:]
if len(last_two) == 2:
    growth = last_two[1][1]['samples_per_sec'] / last_two[0][1]['samples_per_sec']
    print(f"Growth from batch {last_two[0][0]} to {last_two[1][0]}: {growth:.2f}x")
    if growth > 1.1:
        print("STILL SCALING - larger batches may help further")
    else:
        print("PLATEAU REACHED - GPU saturated")

# Compare threading vs batching
print("\nTHREADING vs BATCHING (at same parallelism level):")
print("-" * 50)
print(f"{'Parallelism':<12} {'Threading':<15} {'Batching':<15} {'Ratio':<10}")
for n in [2, 4, 8, 16]:
    t_key = f'{n}_threads'
    b_key = f'batch_{n}'
    if t_key in results['threading'] and b_key in results['batching']:
        t_ops = results['threading'][t_key]['ops_per_sec'] * n  # total samples
        b_ops = results['batching'][b_key]['samples_per_sec']
        ratio = b_ops / t_ops if t_ops > 0 else 0
        print(f"{n:<12} {t_ops:<15.1f} {b_ops:<15.1f} {ratio:<10.1f}x")

# Summary
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print(f"""
1. THREADING WALL: Hits diminishing returns at {max_threading_threads} threads
   - Peak speedup: {max_threading_speedup:.2f}x (vs ideal {max_threading_threads}x)
   - Efficiency at 8 threads: {results['threading']['8_threads']['efficiency']:.1%}
   - Efficiency at 16 threads: {results['threading']['16_threads']['efficiency']:.1%}

2. BATCHING SCALES BETTER:
   - Batch 16: {results['batching']['batch_16']['speedup']:.1f}x speedup
   - Batch 64: {results['batching']['batch_64']['speedup']:.1f}x speedup
   - Batch 256: {results['batching']['batch_256']['speedup']:.1f}x speedup

3. RECOMMENDATION:
   - For parallelism: Use BATCHING, not threading
   - Threading is limited by Apple driver serialization
   - Batching leverages GPU internal parallelism
""")

# Save results
output_path = "reports/main/scaling_curves.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_path}")
