#!/usr/bin/env python3
"""
Prove or disprove: Is the system GPU-bound or CPU-bound?

Method:
1. Measure GPU kernel time vs wall time
2. Measure CPU time in our code vs Metal framework
3. Calculate actual utilization percentages
"""
import torch
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

# Verify MPS
assert torch.backends.mps.is_available(), "MPS not available"
device = torch.device("mps")

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

def create_model(layers=6):
    """Create a 6-layer transformer."""
    blocks = torch.nn.ModuleList([TransformerBlock() for _ in range(layers)])
    return torch.nn.Sequential(*blocks).to(device).eval()

def measure_single_thread_timing():
    """Measure detailed timing for single-threaded execution."""
    model = create_model()
    x = torch.randn(4, 128, 256, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    torch.mps.synchronize()
    
    # Measure with sync points to isolate GPU time
    wall_times = []
    gpu_only_times = []
    
    for _ in range(50):
        torch.mps.synchronize()
        
        # Wall time (includes CPU overhead + GPU)
        wall_start = time.perf_counter()
        with torch.no_grad():
            result = model(x)
        torch.mps.synchronize()
        wall_end = time.perf_counter()
        wall_times.append(wall_end - wall_start)
        
        # Now measure pure GPU time by timing the sync
        torch.mps.synchronize()
        gpu_start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        # Time how long sync takes (this is GPU execution time)
        sync_start = time.perf_counter()
        torch.mps.synchronize()
        sync_end = time.perf_counter()
        gpu_only_times.append(sync_end - gpu_start)
    
    return {
        'wall_time_ms': statistics.mean(wall_times) * 1000,
        'wall_time_std': statistics.stdev(wall_times) * 1000,
        'gpu_time_ms': statistics.mean(gpu_only_times) * 1000,
        'gpu_time_std': statistics.stdev(gpu_only_times) * 1000,
    }

def measure_parallel_overhead(num_threads):
    """Measure overhead when running parallel threads."""
    models = [create_model() for _ in range(num_threads)]
    inputs = [torch.randn(4, 128, 256, device=device) for _ in range(num_threads)]
    
    # Warmup
    for m, x in zip(models, inputs):
        with torch.no_grad():
            _ = m(x)
    torch.mps.synchronize()
    
    results = {'thread_times': [], 'wall_time': 0}
    barrier = threading.Barrier(num_threads)
    
    def worker(tid):
        times = []
        for _ in range(20):
            barrier.wait()  # Sync all threads
            start = time.perf_counter()
            with torch.no_grad():
                _ = models[tid](inputs[tid])
            torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        return times
    
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            results['thread_times'].append(f.result())
    wall_end = time.perf_counter()
    results['wall_time'] = wall_end - wall_start
    
    return results

def calculate_gpu_utilization():
    """Calculate actual GPU utilization at different thread counts."""
    print("=" * 70)
    print("GPU UTILIZATION PROOF")
    print("=" * 70)
    
    # Single thread baseline
    single = measure_single_thread_timing()
    print(f"\n1 THREAD BASELINE:")
    print(f"  Wall time:     {single['wall_time_ms']:.3f} ± {single['wall_time_std']:.3f} ms")
    print(f"  GPU time:      {single['gpu_time_ms']:.3f} ± {single['gpu_time_std']:.3f} ms")
    print(f"  CPU overhead:  {single['wall_time_ms'] - single['gpu_time_ms']:.3f} ms")
    print(f"  GPU util:      {100 * single['gpu_time_ms'] / single['wall_time_ms']:.1f}%")
    
    single_throughput = 1000 / single['wall_time_ms']
    print(f"  Throughput:    {single_throughput:.1f} ops/s")
    
    # Multi-thread measurements
    print(f"\nPARALLEL SCALING ANALYSIS:")
    print("-" * 70)
    
    for n_threads in [2, 4, 8]:
        results = measure_parallel_overhead(n_threads)
        
        # Average time per operation across all threads
        all_times = [t for times in results['thread_times'] for t in times]
        avg_time = statistics.mean(all_times) * 1000
        
        # Throughput = total ops / wall time
        total_ops = n_threads * 20
        throughput = total_ops / results['wall_time']
        speedup = throughput / single_throughput
        
        # Theoretical max: if GPU does N ops in same time as 1 op
        # (perfect parallelism until saturation)
        theoretical_speedup = min(n_threads, 4.0)  # Assume GPU saturates at 4
        efficiency = speedup / theoretical_speedup
        
        print(f"\n{n_threads} THREADS:")
        print(f"  Avg op time:      {avg_time:.3f} ms (vs {single['wall_time_ms']:.3f} ms single)")
        print(f"  Throughput:       {throughput:.1f} ops/s")
        print(f"  Actual speedup:   {speedup:.2f}x")
        print(f"  Theoretical max:  {theoretical_speedup:.1f}x (GPU saturation model)")
        print(f"  Efficiency:       {100*efficiency:.1f}%")
        
        # Calculate contention overhead
        expected_time = single['wall_time_ms']  # If no contention
        contention_overhead = (avg_time - expected_time) / expected_time * 100
        print(f"  Time overhead:    {contention_overhead:+.1f}%")

def prove_saturation_point():
    """Prove where GPU saturates by measuring marginal throughput."""
    print("\n" + "=" * 70)
    print("GPU SATURATION PROOF")
    print("=" * 70)
    
    model = create_model()
    x = torch.randn(4, 128, 256, device=device)
    
    # Measure how throughput scales with batch size (proxy for GPU load)
    print("\nBatch size scaling (single thread):")
    print("-" * 50)
    
    for batch_size in [1, 2, 4, 8, 16]:
        x_batch = torch.randn(batch_size, 128, 256, device=device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x_batch)
        torch.mps.synchronize()
        
        # Measure
        times = []
        for _ in range(20):
            torch.mps.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x_batch)
            torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_ms = statistics.mean(times) * 1000
        throughput = batch_size * 1000 / avg_ms
        throughput_per_sample = 1000 / avg_ms
        
        print(f"  Batch {batch_size:2d}: {avg_ms:7.2f} ms, {throughput:7.1f} samples/s, {throughput_per_sample:.1f} batches/s")
    
    print("\nIf GPU is saturated, doubling batch size should ~double time.")
    print("If GPU is NOT saturated, doubling batch size should take SAME time.")

def main():
    print("SKEPTICAL ANALYSIS: Proving GPU-Bound Claims")
    print("=" * 70)
    
    calculate_gpu_utilization()
    prove_saturation_point()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
To prove GPU-bound:
1. Time overhead from 1→N threads should be minimal (< 20%)
2. Throughput should plateau as threads increase (saturation)
3. Batch size scaling should show linear time increase (GPU doing real work)

If these conditions are NOT met, the system is CPU-bound or contention-bound.
""")

if __name__ == "__main__":
    main()
