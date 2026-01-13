#!/usr/bin/env python3
"""
Investigate: "95% efficiency" - 95% of WHAT?

Created by Andrew Yates

The blog claims batching achieves ~95% efficiency. But efficiency requires:
1. A baseline (100%)
2. A measurement

Possible interpretations:
A) 95% GPU utilization (Metal profiler would show this)
B) 95% of theoretical peak FLOPS
C) 95% of some reference throughput
D) Made-up number

Let's measure what we can actually measure.
"""

import torch
import torch.nn as nn
import time
import statistics

device = torch.device("mps")

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                       batch_first=True, dropout=0)
            for _ in range(layers)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

model = TransformerBlock().to(device).eval()

print("=" * 70)
print("INVESTIGATING BATCHING EFFICIENCY")
print("=" * 70)
print()

# Measure throughput at different batch sizes
print("Batch Scaling Analysis:")
print("-" * 50)
print("Batch Size | Samples/s | Batches/s | Time/batch")

results = {}
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    # Warmup
    x = torch.randn(batch_size, 32, 256, device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    torch.mps.synchronize()
    
    # Measure
    iterations = 30
    times = []
    for _ in range(iterations):
        x = torch.randn(batch_size, 32, 256, device=device)
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = statistics.mean(times)
    samples_per_sec = batch_size / avg_time
    batches_per_sec = 1 / avg_time
    
    results[batch_size] = {
        'time_ms': avg_time * 1000,
        'samples_s': samples_per_sec,
        'batches_s': batches_per_sec
    }
    
    print(f"    {batch_size:3d}    | {samples_per_sec:9.1f} | {batches_per_sec:9.1f} | {avg_time*1000:7.2f} ms")

print()
print("ANALYSIS:")
print("-" * 50)

# If GPU is fully utilized:
# - Doubling batch size should approximately double time
# - Samples/sec should stay roughly constant after saturation

baseline_1 = results[1]['samples_s']
baseline_8 = results[8]['samples_s']
baseline_64 = results[64]['samples_s']

print(f"Batch 1:  {baseline_1:.1f} samples/s (baseline)")
print(f"Batch 8:  {baseline_8:.1f} samples/s ({baseline_8/baseline_1:.2f}x of batch 1)")
print(f"Batch 64: {baseline_64:.1f} samples/s ({baseline_64/baseline_1:.2f}x of batch 1)")

print()
print("EFFICIENCY INTERPRETATION:")
print("-" * 50)

# Linear scaling would mean: batch N gives N * baseline_1 samples/s
# Efficiency = actual / theoretical * 100%

for batch_size in [8, 16, 32, 64]:
    theoretical = batch_size * baseline_1
    actual = results[batch_size]['samples_s']
    efficiency = actual / theoretical * 100
    print(f"Batch {batch_size:2d}: {efficiency:.1f}% vs linear scaling")

print()
print("CONCLUSION:")
print("-" * 50)
print("""
The "95% efficiency" claim in the blog is VAGUE.

What we CAN measure:
- Samples/sec at different batch sizes
- Scaling efficiency (actual vs linear extrapolation)

What we CANNOT measure without profilers:
- GPU utilization %
- Memory bandwidth utilization %
- Compute unit saturation %

The batching advantage is REAL (10x vs threading), but
quantifying "efficiency %" requires defining a baseline.
""")
