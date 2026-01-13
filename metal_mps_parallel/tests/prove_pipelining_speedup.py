#!/usr/bin/env python3
"""
Prove pipelining speedup (not GPU parallelism).

The 3.64x claim:
- 8 user threads submit work
- 1 worker thread processes GPU
- Speedup is from overlapping CPU prep with GPU execution

This is PIPELINING, not PARALLEL GPU.
"""
import torch
import torch.nn as nn
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, Future

device = torch.device("mps")

class LargeTransformer(nn.Module):
    """6-layer transformer (same as report)."""
    def __init__(self, d_model=512, nhead=8, layers=6):
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

def sequential_benchmark(model, iterations):
    """True sequential: submit, wait, repeat."""
    torch.mps.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            x = torch.randn(4, 128, 512, device=device)
            _ = model(x)
            torch.mps.synchronize()  # Wait after each
    
    elapsed = time.perf_counter() - start
    return iterations / elapsed

def pipelined_benchmark(model, iterations, num_user_threads):
    """
    Pipelined: multiple user threads prepare inputs,
    but GPU execution is serialized (1 worker).
    """
    input_queue = Queue(maxsize=num_user_threads * 2)
    output_queue = Queue()
    stop_flag = threading.Event()
    
    def gpu_worker():
        """Single GPU worker - processes inputs sequentially."""
        processed = 0
        while not stop_flag.is_set() or not input_queue.empty():
            try:
                x = input_queue.get(timeout=0.1)
                with torch.no_grad():
                    result = model(x)
                torch.mps.synchronize()
                output_queue.put(result)
                processed += 1
            except:
                continue
        return processed
    
    def input_producer(tid, count):
        """User thread - prepares inputs and submits to queue."""
        for _ in range(count):
            x = torch.randn(4, 128, 512, device=device)
            input_queue.put(x)
        return count
    
    # Warmup
    with torch.no_grad():
        _ = model(torch.randn(4, 128, 512, device=device))
    torch.mps.synchronize()
    
    # Start benchmark
    torch.mps.synchronize()
    start = time.perf_counter()
    
    # Start GPU worker
    gpu_thread = threading.Thread(target=gpu_worker)
    gpu_thread.start()
    
    # Start input producers
    per_thread = iterations // num_user_threads
    with ThreadPoolExecutor(max_workers=num_user_threads) as ex:
        futures = [ex.submit(input_producer, i, per_thread) for i in range(num_user_threads)]
        total_submitted = sum(f.result() for f in futures)
    
    # Wait for all outputs
    results_received = 0
    while results_received < total_submitted:
        try:
            _ = output_queue.get(timeout=5.0)
            results_received += 1
        except:
            break
    
    stop_flag.set()
    gpu_thread.join()
    
    elapsed = time.perf_counter() - start
    return results_received / elapsed

print("=" * 70)
print("PIPELINING vs SEQUENTIAL PROOF")
print("=" * 70)
print("\nModel: 6-layer Transformer (d=512, 8 heads)")
print("This tests PIPELINING (CPU/GPU overlap), NOT parallel GPU")
print()

model = LargeTransformer().to(device).eval()

# Sequential baseline
print("Testing sequential (no pipelining)...", end=" ", flush=True)
seq_throughput = sequential_benchmark(model, 30)
print(f"{seq_throughput:.1f} ops/s")

# Pipelined with different user thread counts
print()
for n_threads in [2, 4, 8]:
    print(f"Testing pipelined ({n_threads} user → 1 GPU worker)...", end=" ", flush=True)
    pipe_throughput = pipelined_benchmark(model, 30, n_threads)
    speedup = pipe_throughput / seq_throughput
    print(f"{pipe_throughput:.1f} ops/s ({speedup:.2f}x)")

print()
print("=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("""
If pipelining shows >1x speedup, it means:
- CPU preparation time overlaps with GPU execution
- Speedup is from hiding CPU latency, NOT from parallel GPU

The 3.64x claim is from this pipelining effect, not from
running 8 parallel GPU streams simultaneously.

True parallel GPU would require each thread's work to execute
concurrently on GPU - which Apple MPS doesn't support safely.
""")
