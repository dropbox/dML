#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark: Impact of skipping compute_logprobs in greedy decoding.

F10 optimization: Skip softmax/logsumexp for greedy decoding when not needed.
"""

import time
import mlx.core as mx
import numpy as np

# Simulate typical Whisper vocab and decode loop parameters
VOCAB_SIZE = 51865  # Whisper large vocab
BATCH_SIZE = 1
NUM_DECODE_STEPS = 50  # Typical for 30s audio
WARMUP_ITERS = 5
BENCH_ITERS = 20


def compute_logprobs(logits: mx.array) -> mx.array:
    """Current implementation: full logsumexp."""
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def compute_single_logprob(logits: mx.array, token_idx: int) -> float:
    """Optimized: compute logprob for single token only."""
    log_normalizer = mx.logsumexp(logits, axis=-1)
    return float(logits[0, token_idx] - log_normalizer[0])


def greedy_decode(logits: mx.array) -> mx.array:
    """Greedy decoding (no softmax)."""
    return mx.argmax(logits, axis=-1)


def benchmark_full_logprobs():
    """Benchmark decode loop WITH full logprobs computation."""
    # Create random logits to simulate decoder output
    logits_list = [mx.random.normal((BATCH_SIZE, VOCAB_SIZE)) for _ in range(NUM_DECODE_STEPS)]
    mx.eval(logits_list)

    # Warmup
    for i in range(WARMUP_ITERS):
        for logits in logits_list:
            next_token = greedy_decode(logits)
            logprobs = compute_logprobs(logits)
            token_logprob = float(logprobs[0, int(next_token[0])])
            mx.eval(next_token, logprobs)

    # Benchmark
    times = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        total_logprob = 0.0
        for logits in logits_list:
            next_token = greedy_decode(logits)
            logprobs = compute_logprobs(logits)
            token_logprob = float(logprobs[0, int(next_token[0])])
            total_logprob += token_logprob
            mx.eval(next_token, logprobs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_single_logprob():
    """Benchmark decode loop with single token logprob only."""
    logits_list = [mx.random.normal((BATCH_SIZE, VOCAB_SIZE)) for _ in range(NUM_DECODE_STEPS)]
    mx.eval(logits_list)

    # Warmup
    for i in range(WARMUP_ITERS):
        for logits in logits_list:
            next_token = greedy_decode(logits)
            token_logprob = compute_single_logprob(logits, int(next_token[0]))
            mx.eval(next_token)

    # Benchmark
    times = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        total_logprob = 0.0
        for logits in logits_list:
            next_token = greedy_decode(logits)
            token_logprob = compute_single_logprob(logits, int(next_token[0]))
            total_logprob += token_logprob
            mx.eval(next_token)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_skip_logprobs():
    """Benchmark decode loop WITHOUT logprobs computation."""
    logits_list = [mx.random.normal((BATCH_SIZE, VOCAB_SIZE)) for _ in range(NUM_DECODE_STEPS)]
    mx.eval(logits_list)

    # Warmup
    for i in range(WARMUP_ITERS):
        for logits in logits_list:
            next_token = greedy_decode(logits)
            mx.eval(next_token)

    # Benchmark
    times = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        for logits in logits_list:
            next_token = greedy_decode(logits)
            mx.eval(next_token)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def main():
    print("=" * 60)
    print("F10 Benchmark: Skip Logprobs in Greedy Decoding")
    print("=" * 60)
    print(f"Config: vocab={VOCAB_SIZE}, decode_steps={NUM_DECODE_STEPS}")
    print(f"Warmup: {WARMUP_ITERS}, Benchmark iters: {BENCH_ITERS}")
    print()

    # Run benchmarks
    full_mean, full_std = benchmark_full_logprobs()
    print(f"Full logprobs:   {full_mean*1000:.2f}ms ± {full_std*1000:.2f}ms")

    single_mean, single_std = benchmark_single_logprob()
    print(f"Single logprob:  {single_mean*1000:.2f}ms ± {single_std*1000:.2f}ms")

    skip_mean, skip_std = benchmark_skip_logprobs()
    print(f"Skip logprobs:   {skip_mean*1000:.2f}ms ± {skip_std*1000:.2f}ms")

    print()
    print("-" * 60)
    print("Results:")
    print(f"  Skip vs Full:   {full_mean/skip_mean:.2f}x speedup ({(full_mean-skip_mean)/full_mean*100:.1f}% reduction)")
    print(f"  Single vs Full: {full_mean/single_mean:.2f}x speedup ({(full_mean-single_mean)/full_mean*100:.1f}% reduction)")
    print()

    # Calculate per-token overhead
    per_token_full = (full_mean - skip_mean) * 1000 / NUM_DECODE_STEPS
    per_token_single = (single_mean - skip_mean) * 1000 / NUM_DECODE_STEPS
    print("Per-token overhead:")
    print(f"  Full logprobs:  {per_token_full:.3f}ms/token")
    print(f"  Single logprob: {per_token_single:.3f}ms/token")
    print()

    # Verdict
    if full_mean / skip_mean > 1.05:
        print("✅ RECOMMENDATION: Implement skip_logprobs option")
        print(f"   Expected speedup: {full_mean/skip_mean:.1f}x for decode loop")
    else:
        print("⚠️ RESULT: Logprobs overhead is minimal (<5%)")
        print("   May not be worth implementation complexity")


if __name__ == "__main__":
    main()
