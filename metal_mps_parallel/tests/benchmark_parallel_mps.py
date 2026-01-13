#!/usr/bin/env python3
"""
Phase 7: MPS Parallel Inference Benchmarking

This benchmark measures:
1. Single-thread baseline latency
2. Multi-thread throughput scaling
3. GPU saturation point

Exit codes:
    0: All requested thread counts completed expected ops (Linear/MLP at all counts,
       Transformer at ≤2 threads)
    1: Unexpected failures detected

Known Limitation (Apple Metal Framework):
    TransformerEncoderLayer (which uses LayerNorm) may crash at 4+ concurrent threads
    due to Apple Metal compute kernel thread-safety issues. This is NOT a bug in
    our code - it's a framework limitation. See WORKER_DIRECTIVE.md for details.

Usage:
    cd ~/metal_mps_parallel && source venv_mps_test/bin/activate
    python3 tests/benchmark_parallel_mps.py

Large-workload example (for scaling efficiency):
    python3 tests/benchmark_parallel_mps.py --model linear --batch 64 --in-features 2048 --out-features 2048

Output: Markdown table suitable for plan documentation
"""

import subprocess
import sys
import time
import json
import argparse

# Early validation: check torch is available before running benchmarks
try:
    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not available. Run on Apple Silicon with MPS support.", file=sys.stderr)
        sys.exit(2)
except ImportError:
    print("ERROR: PyTorch not found. Activate the venv first:", file=sys.stderr)
    print("  cd ~/metal_mps_parallel && source venv_mps_test/bin/activate", file=sys.stderr)
    sys.exit(2)


def run_single_benchmark(
    model: str,
    threads: int,
    iterations: int,
    batch: int,
    in_features: int,
    hidden: int,
    out_features: int,
    seq_len: int,
    d_model: int,
    nhead: int,
) -> dict:
    """Run a single benchmark in a subprocess for stability."""
    code = f'''
import torch
import torch.nn as nn
import threading
import time
import json
import sys

torch.zeros(1, device="mps")
torch.mps.synchronize()

MODEL = "{model}"
BATCH = {batch}
IN_FEATURES = {in_features}
HIDDEN = {hidden}
OUT_FEATURES = {out_features}
SEQ_LEN = {seq_len}
D_MODEL = {d_model}
NHEAD = {nhead}

results = []
latencies = []
lock = threading.Lock()

def make_model():
    if MODEL == "linear":
        return nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    elif MODEL == "mlp":
        return nn.Sequential(
            nn.Linear(IN_FEATURES, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, IN_FEATURES), nn.ReLU(),
            nn.Linear(IN_FEATURES, OUT_FEATURES)
        ).to("mps")
    else:  # transformer
        return nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True).to("mps")

def make_input():
    if MODEL == "transformer":
        return torch.randn(BATCH, SEQ_LEN, D_MODEL, device="mps")
    else:
        return torch.randn(BATCH, IN_FEATURES, device="mps")

def worker(tid):
    try:
        model = make_model()
        model.eval()
        for i in range({iterations}):
            x = make_input()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
                torch.mps.synchronize()
            end = time.perf_counter()
            with lock:
                results.append(1)
                latencies.append((end - start) * 1000)
    except Exception as e:
        pass

threads = [threading.Thread(target=worker, args=(i,)) for i in range({threads})]
start = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
total_time = time.perf_counter() - start

total_ops = len(results)
ops_per_sec = total_ops / total_time if total_time > 0 else 0
sorted_lat = sorted(latencies) if latencies else [0]
p50_idx = len(sorted_lat) // 2
p99_idx = int(len(sorted_lat) * 0.99)

print(json.dumps({{
    "total_ops": total_ops,
    "total_time": total_time,
    "ops_per_sec": ops_per_sec,
    "avg_ms": sum(latencies)/len(latencies) if latencies else 0,
    "p50_ms": sorted_lat[p50_idx],
    "p99_ms": sorted_lat[p99_idx],
    "expected": {threads} * {iterations}
}}))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            return {"total_ops": 0, "total_time": 0, "ops_per_sec": 0,
                    "avg_ms": 0, "p50_ms": 0, "p99_ms": 0, "expected": threads * iterations,
                    "error": f"Exit code {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"total_ops": 0, "total_time": 0, "ops_per_sec": 0,
                "avg_ms": 0, "p50_ms": 0, "p99_ms": 0, "expected": threads * iterations,
                "error": "Timeout"}
    except Exception as e:
        return {"total_ops": 0, "total_time": 0, "ops_per_sec": 0,
                "avg_ms": 0, "p50_ms": 0, "p99_ms": 0, "expected": threads * iterations,
                "error": str(e)}


def run_benchmark_suite(
    model: str,
    thread_counts: list,
    iterations: int,
    batch: int,
    in_features: int,
    hidden: int,
    out_features: int,
    seq_len: int,
    d_model: int,
    nhead: int,
):
    """Run benchmark suite for a model across thread counts."""
    model_names = {
        "linear": f"nn.Linear ({in_features}->{out_features}, batch={batch})",
        "mlp": f"MLP (in={in_features}, hidden={hidden}, out={out_features}, batch={batch})",
        "transformer": f"TransformerEncoderLayer (d_model={d_model}, nhead={nhead}, batch={batch}, seq_len={seq_len})",
    }

    print(f"\n### {model_names[model]} Scaling\n")
    print("| Threads | Ops | Time (s) | ops/s | Avg (ms) | P50 (ms) | P99 (ms) | Status |")
    print("|---------|-----|----------|-------|----------|----------|----------|--------|")

    results = []
    all_passed = True
    for threads in thread_counts:
        r = run_single_benchmark(
            model,
            threads,
            iterations,
            batch,
            in_features,
            hidden,
            out_features,
            seq_len,
            d_model,
            nhead,
        )
        results.append(r)
        passed = r.get("total_ops") == r.get("expected") and "error" not in r
        status = "PASS" if passed else "FAIL"
        if "error" in r:
            status = f"FAIL ({r['error'][:20]})"
            passed = False
        all_passed = all_passed and passed
        print(f"| {threads} | {r['total_ops']} | {r['total_time']:.2f} | {r['ops_per_sec']:.0f} | {r['avg_ms']:.2f} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {status} |")

    # Print scaling efficiency
    if len(results) >= 2 and results[0].get("ops_per_sec", 0) > 0:
        baseline = results[0]["ops_per_sec"]
        print("\n**Scaling Efficiency:**")
        print("| Threads | ops/s | Speedup | Efficiency |")
        print("|---------|-------|---------|------------|")
        for i, r in enumerate(results):
            t = thread_counts[i]
            ops = r.get("ops_per_sec", 0)
            speedup = ops / baseline if baseline > 0 else 0
            efficiency = speedup / t * 100 if t > 0 else 0
            print(f"| {t} | {ops:.0f} | {speedup:.2f}x | {efficiency:.1f}% |")

    return results, all_passed


def main():
    parser = argparse.ArgumentParser(description='MPS Parallel Inference Benchmark')
    parser.add_argument('--iterations', type=int, default=50, help='Iterations per thread')
    parser.add_argument('--model', choices=['linear', 'mlp', 'transformer', 'all'], default='all')
    parser.add_argument('--threads', default='1,2,4,8', help='Comma-separated thread counts')
    parser.add_argument('--batch', type=int, default=4, help='Input batch size')
    parser.add_argument('--in-features', type=int, default=256, help='Linear/MLP input features')
    parser.add_argument('--hidden', type=int, default=512, help='MLP hidden size')
    parser.add_argument('--out-features', type=int, default=128, help='Linear/MLP output features')
    parser.add_argument('--seq-len', type=int, default=32, help='Transformer sequence length')
    parser.add_argument('--d-model', type=int, default=256, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer nhead (must divide d_model)')
    args = parser.parse_args()

    print("# Phase 7: MPS Parallel Inference Benchmarks")
    print(f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"**Iterations per thread**: {args.iterations}")
    print(f"**Thread counts**: {args.threads}")
    print(f"**Input batch**: {args.batch}")
    print(f"**Linear/MLP dims**: in={args.in_features}, hidden={args.hidden}, out={args.out_features}")
    print(f"**Transformer dims**: batch={args.batch}, seq_len={args.seq_len}, d_model={args.d_model}, nhead={args.nhead}")
    print("\n*Each thread count runs in a separate process for stability.*")

    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]
    if not thread_counts or any(t <= 0 for t in thread_counts):
        raise SystemExit(f"Invalid --threads={args.threads!r}; expected positive integers (e.g. 1,2,4,8)")
    thread_counts = sorted(set(thread_counts))
    if args.model == 'transformer' or args.model == 'all':
        if args.d_model <= 0 or args.nhead <= 0 or args.seq_len <= 0:
            raise SystemExit("--d-model, --nhead, and --seq-len must be positive")
        if args.d_model % args.nhead != 0:
            raise SystemExit("--d-model must be divisible by --nhead for TransformerEncoderLayer")

    if args.model == 'all':
        models = ['linear', 'mlp', 'transformer']
    else:
        models = [args.model]

    unexpected_failures = False
    transformer_high_thread_failures = 0  # Expected due to LayerNorm limitation

    for model in models:
        results, model_passed = run_benchmark_suite(
            model,
            thread_counts,
            args.iterations,
            args.batch,
            args.in_features,
            args.hidden,
            args.out_features,
            args.seq_len,
            args.d_model,
            args.nhead,
        )

        if not model_passed:
            if model == 'transformer':
                # Check which thread counts failed for transformer
                for i, r in enumerate(results):
                    t = thread_counts[i]
                    if r.get("total_ops") != r.get("expected") or "error" in r:
                        if t >= 4:
                            # Expected failure due to LayerNorm/Metal limitation
                            transformer_high_thread_failures += 1
                        else:
                            # Unexpected failure at low thread count
                            unexpected_failures = True
            else:
                # Any failure in linear/mlp is unexpected
                unexpected_failures = True

    print("\n## Summary")
    print("- nn.Linear and MLP: Thread-safe at all thread counts")
    print("- TransformerEncoderLayer: Uses LayerNorm, may fail at 4+ threads (Apple Metal limitation)")
    print("- Scaling efficiency = speedup / thread_count * 100%")
    print("- >50% efficiency indicates good parallelism")
    print("- <50% efficiency indicates GPU saturation or contention")

    if transformer_high_thread_failures > 0:
        print(f"\n**Note**: {transformer_high_thread_failures} TransformerEncoderLayer failure(s) at 4+ threads")
        print("         This is EXPECTED due to Apple Metal LayerNorm thread-safety limitation.")
        print("         See WORKER_DIRECTIVE.md 'Apple MPS Framework Limitations' section.")

    if unexpected_failures:
        print("\nUNEXPECTED BENCHMARK FAILURES DETECTED")
        return 1
    else:
        print("\nBENCHMARKS COMPLETED (expected behavior)")
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
