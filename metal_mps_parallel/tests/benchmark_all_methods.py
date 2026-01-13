#!/usr/bin/env python3
"""
COMPREHENSIVE BENCHMARK: Find the Absolute Best MPS Performance

Created by Andrew Yates

This benchmark tests EVERY method to find the absolute best performance:
1. Single-threaded baseline
2. Threading (2, 4, 8 threads) - limited by Apple driver
3. Process pool (2, 4, 8 processes) - bypasses driver serialization
4. Batching (batch sizes 1-128) - GPU internal parallelism
5. BatchQueue pipelining - async submission
6. Hybrid: Batching + Threading
7. Hybrid: Batching + Process Pool

THE GOAL: Find the combination that achieves MAXIMUM throughput.
"""

import torch
import torch.nn as nn
import time
import threading
import multiprocessing as mp
import statistics
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Tuple

print("=" * 70)
print("COMPREHENSIVE MPS BENCHMARK - FINDING ABSOLUTE BEST PERFORMANCE")
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

results = {}

# ============================================================================
# 1. SINGLE-THREADED BASELINE
# ============================================================================

print("\n[1/7] Single-threaded baseline...")

def measure_baseline():
    x = torch.randn(32, 512, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()

    # Measure
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return iterations / elapsed

baseline_ops = measure_baseline()
results['baseline'] = {
    'ops_per_sec': baseline_ops,
    'samples_per_sec': baseline_ops * 32,
    'description': 'Single-threaded, batch=32'
}
print(f"   Baseline: {baseline_ops:.1f} ops/s ({baseline_ops * 32:.1f} samples/s)")

# ============================================================================
# 2. THREADING (Limited by Apple driver)
# ============================================================================

print("\n[2/7] Threading (various thread counts)...")

def measure_threading(num_threads: int) -> Dict:
    completed = [0]
    lock = threading.Lock()

    def worker():
        x = torch.randn(32, 512, device=device)
        for _ in range(50):
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
    speedup = ops / baseline_ops
    efficiency = speedup / num_threads

    return {
        'ops_per_sec': ops,
        'speedup': speedup,
        'efficiency': efficiency,
        'samples_per_sec': ops * 32
    }

threading_results = {}
for n in [2, 4, 8]:
    r = measure_threading(n)
    threading_results[f'{n}_threads'] = r
    print(f"   {n} threads: {r['ops_per_sec']:.1f} ops/s, {r['speedup']:.2f}x speedup, {r['efficiency']:.1%} efficiency")

results['threading'] = threading_results

# ============================================================================
# 3. PROCESS POOL (Bypasses driver serialization)
# ============================================================================

print("\n[3/7] Process pool (bypasses driver)...")

def worker_process_simple(worker_id, ready_event, start_event, result_queue, iterations):
    """Simple worker that signals when ready, waits for start, does work."""
    import torch
    import torch.nn as nn

    device = torch.device('mps')
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).to(device).eval()

    x = torch.randn(32, 512, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()

    # Signal ready
    ready_event.set()

    # Wait for start signal
    start_event.wait()

    # Do work
    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time

    result_queue.put({
        'worker_id': worker_id,
        'iterations': iterations,
        'elapsed': elapsed
    })

def measure_process_pool(num_processes: int, iterations_per_process: int = 50) -> Dict:
    """Measure process pool performance."""
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass

    ready_events = [mp.Event() for _ in range(num_processes)]
    start_event = mp.Event()
    result_queue = mp.Queue()

    # Start workers
    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker_process_simple,
            args=(i, ready_events[i], start_event, result_queue, iterations_per_process)
        )
        p.start()
        processes.append(p)

    # Wait for all workers to be ready
    for e in ready_events:
        e.wait(timeout=30)

    # Start all workers simultaneously
    overall_start = time.perf_counter()
    start_event.set()

    # Collect results
    results_list = []
    for _ in range(num_processes):
        r = result_queue.get(timeout=60)
        results_list.append(r)

    overall_elapsed = time.perf_counter() - overall_start

    # Cleanup
    for p in processes:
        p.join(timeout=5)

    # Calculate metrics
    total_ops = num_processes * iterations_per_process
    ops_per_sec = total_ops / overall_elapsed
    speedup = ops_per_sec / baseline_ops
    efficiency = speedup / num_processes

    return {
        'ops_per_sec': ops_per_sec,
        'speedup': speedup,
        'efficiency': efficiency,
        'samples_per_sec': ops_per_sec * 32,
        'total_ops': total_ops,
        'elapsed': overall_elapsed
    }

process_results = {}
for n in [2, 4, 8]:
    try:
        r = measure_process_pool(n, iterations_per_process=30)
        process_results[f'{n}_processes'] = r
        print(f"   {n} processes: {r['ops_per_sec']:.1f} ops/s, {r['speedup']:.2f}x speedup, {r['efficiency']:.1%} efficiency")
    except Exception as e:
        print(f"   {n} processes: ERROR - {e}")
        process_results[f'{n}_processes'] = {'error': str(e)}

results['process_pool'] = process_results

# ============================================================================
# 4. BATCHING (GPU internal parallelism)
# ============================================================================

print("\n[4/7] Batching (GPU parallelism)...")

def measure_batching(batch_size: int) -> Dict:
    x = torch.randn(batch_size, 512, device=device)

    # Warmup
    for _ in range(5):
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

    return {
        'batch_size': batch_size,
        'samples_per_sec': samples_per_sec,
        'ops_per_sec': ops_per_sec,
        'samples_scaling': samples_per_sec / (baseline_ops * 32)
    }

batching_results = {}
for bs in [1, 8, 32, 64, 128]:
    r = measure_batching(bs)
    batching_results[f'batch_{bs}'] = r
    print(f"   Batch {bs}: {r['samples_per_sec']:.1f} samples/s, {r['samples_scaling']:.1f}x scaling")

results['batching'] = batching_results

# ============================================================================
# 5. BATCHQUEUE PIPELINING
# ============================================================================

print("\n[5/7] BatchQueue pipelining...")

def measure_batchqueue() -> Dict:
    """Test torch.mps.BatchQueue if available."""
    try:
        # Check if BatchQueue exists
        if not hasattr(torch.mps, 'BatchQueue'):
            return {'error': 'BatchQueue not available'}

        bq = torch.mps.BatchQueue(num_workers=1)

        x = torch.randn(32, 512, device=device)

        # Warmup
        for _ in range(10):
            future = bq.submit(model, x)
            future.result()

        # Measure
        iterations = 100
        start = time.perf_counter()
        futures = []
        for _ in range(iterations):
            futures.append(bq.submit(model, x))
        for f in futures:
            f.result()
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        speedup = ops_per_sec / baseline_ops

        return {
            'ops_per_sec': ops_per_sec,
            'speedup': speedup,
            'samples_per_sec': ops_per_sec * 32
        }
    except Exception as e:
        return {'error': str(e)}

batchqueue_result = measure_batchqueue()
results['batchqueue'] = batchqueue_result
if 'error' not in batchqueue_result:
    print(f"   BatchQueue: {batchqueue_result['ops_per_sec']:.1f} ops/s, {batchqueue_result['speedup']:.2f}x speedup")
else:
    print(f"   BatchQueue: {batchqueue_result['error']}")

# ============================================================================
# 6. HYBRID: Batching + Threading
# ============================================================================

print("\n[6/7] Hybrid: Batching + Threading...")

def measure_hybrid_batch_thread(num_threads: int, batch_size: int) -> Dict:
    completed = [0]
    lock = threading.Lock()

    def worker():
        x = torch.randn(batch_size, 512, device=device)
        for _ in range(30):
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

    total_samples = completed[0] * batch_size
    samples_per_sec = total_samples / elapsed
    ops_per_sec = completed[0] / elapsed

    return {
        'samples_per_sec': samples_per_sec,
        'ops_per_sec': ops_per_sec,
        'threads': num_threads,
        'batch_size': batch_size
    }

hybrid_results = {}
for threads, batch in [(2, 32), (4, 16), (2, 64)]:
    key = f'{threads}t_batch{batch}'
    r = measure_hybrid_batch_thread(threads, batch)
    hybrid_results[key] = r
    print(f"   {threads} threads × batch {batch}: {r['samples_per_sec']:.1f} samples/s")

results['hybrid_batch_thread'] = hybrid_results

# ============================================================================
# 7. SUMMARY: FIND THE BEST
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: FINDING THE ABSOLUTE BEST")
print("=" * 70)

# Collect all samples/s metrics
all_methods = []

# Baseline
all_methods.append(('Baseline (1 thread, batch 32)', results['baseline']['samples_per_sec']))

# Threading
for k, v in results['threading'].items():
    if 'samples_per_sec' in v:
        all_methods.append((f'Threading {k}', v['samples_per_sec']))

# Process pool
for k, v in results['process_pool'].items():
    if 'samples_per_sec' in v:
        all_methods.append((f'Process Pool {k}', v['samples_per_sec']))

# Batching
for k, v in results['batching'].items():
    all_methods.append((f'Batching {k}', v['samples_per_sec']))

# BatchQueue
if 'samples_per_sec' in results['batchqueue']:
    all_methods.append(('BatchQueue', results['batchqueue']['samples_per_sec']))

# Hybrid
for k, v in results['hybrid_batch_thread'].items():
    all_methods.append((f'Hybrid {k}', v['samples_per_sec']))

# Sort by performance
all_methods.sort(key=lambda x: x[1], reverse=True)

print("\nRANKED BY SAMPLES/SECOND:")
print("-" * 50)
for i, (method, throughput) in enumerate(all_methods[:10], 1):
    baseline_multiple = throughput / results['baseline']['samples_per_sec']
    print(f"{i:2}. {method:35} {throughput:>10.1f} samples/s ({baseline_multiple:.1f}x)")

best_method = all_methods[0]
print("\n" + "=" * 70)
print(f"BEST METHOD: {best_method[0]}")
print(f"THROUGHPUT:  {best_method[1]:.1f} samples/s")
print(f"VS BASELINE: {best_method[1] / results['baseline']['samples_per_sec']:.1f}x")
print("=" * 70)

# Threading vs Process Pool comparison
print("\nTHREADING vs PROCESS POOL (at 8 workers):")
threading_8 = results['threading'].get('8_threads', {})
process_8 = results['process_pool'].get('8_processes', {})

if 'efficiency' in threading_8 and 'efficiency' in process_8:
    print(f"   Threading 8:   {threading_8['efficiency']:.1%} efficiency, {threading_8['speedup']:.2f}x speedup")
    print(f"   Process Pool 8: {process_8['efficiency']:.1%} efficiency, {process_8['speedup']:.2f}x speedup")
    improvement = process_8['efficiency'] / threading_8['efficiency'] if threading_8['efficiency'] > 0 else 0
    print(f"   IMPROVEMENT:    {improvement:.1f}x better efficiency with process pool!")

# Save results
results['ranked'] = [{'method': m, 'samples_per_sec': s} for m, s in all_methods]
results['best'] = {'method': best_method[0], 'samples_per_sec': best_method[1]}
results['timestamp'] = datetime.now().isoformat()

output_path = "reports/main/comprehensive_benchmark.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
