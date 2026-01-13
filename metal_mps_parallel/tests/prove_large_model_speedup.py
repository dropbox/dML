#!/usr/bin/env python3
"""
Verify 3.64x speedup claim with large model.
Report from N=1261: 6-layer Transformer, d=512, 8 heads
"""
import torch
import torch.nn as nn
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

device = torch.device("mps")

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

def create_large_model(layers=6):
    """Create 6-layer transformer (same as report)."""
    blocks = nn.ModuleList([TransformerBlock(d_model=512, nhead=8) for _ in range(layers)])
    return nn.Sequential(*blocks).to(device).eval()

def benchmark_throughput(num_threads, iterations=50):
    """Measure throughput at given thread count."""
    models = [create_large_model() for _ in range(num_threads)]
    inputs = [torch.randn(4, 128, 512, device=device) for _ in range(num_threads)]
    
    # Warmup
    for m, x in zip(models, inputs):
        with torch.no_grad():
            _ = m(x)
    torch.mps.synchronize()
    
    completed = [0] * num_threads
    
    def worker(tid):
        for _ in range(iterations):
            with torch.no_grad():
                _ = models[tid](inputs[tid])
            torch.mps.synchronize()
            completed[tid] += 1
    
    torch.mps.synchronize()
    start = time.perf_counter()
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    end = time.perf_counter()
    total_ops = sum(completed)
    throughput = total_ops / (end - start)
    
    return throughput, end - start

print("=" * 70)
print("VERIFYING 3.64x SPEEDUP CLAIM")
print("Model: 6-layer Transformer (d=512, 8 heads)")
print("=" * 70)

results = {}
for n in [1, 2, 4, 8]:
    print(f"\nTesting {n} threads...", end=" ", flush=True)
    throughput, elapsed = benchmark_throughput(n, iterations=30)
    results[n] = throughput
    print(f"{throughput:.1f} ops/s ({elapsed:.2f}s)")

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)
baseline = results[1]
for n in [1, 2, 4, 8]:
    speedup = results[n] / baseline
    print(f"  {n} threads: {results[n]:.1f} ops/s ({speedup:.2f}x)")

print(f"\nReport claimed: 3.64x at 8 threads")
print(f"Measured:       {results[8]/baseline:.2f}x at 8 threads")

if results[8]/baseline > 3.0:
    print("\n✓ VERIFIED: >3x speedup achieved")
elif results[8]/baseline > 2.0:
    print("\n⚠ PARTIAL: 2-3x speedup (below claim)")
else:
    print("\n✗ FAILED: <2x speedup (claim refuted)")
