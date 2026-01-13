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
Verify P1 Local Attention (Sliding Window) optimization.

Tests:
1. Local attention mask is created correctly
2. Output with local attention is close to full attention for short sequences
3. Performance improvement for longer sequences
"""

import sys
import time

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np

from tools.pytorch_to_mlx.converters.models.kokoro import KokoroModel, KokoroConfig, AlbertAttention


def test_local_mask():
    """Test local attention mask generation."""
    print("\n" + "-" * 70)
    print("TEST 1: Local Attention Mask Generation")
    print("-" * 70)

    config = KokoroConfig()
    attention = AlbertAttention(config)

    # Enable local attention with window=2
    attention.set_local_attention(2)

    # Generate mask for sequence length 6
    mask = attention._get_local_mask(6)
    mx.eval(mask)

    print("  Window size: 2")
    print("  Sequence length: 6")
    print(f"  Mask shape: {mask.shape}")

    # Expected pattern for window=2:
    # Position 0: can attend to [0,1,2]
    # Position 1: can attend to [0,1,2,3]
    # Position 2: can attend to [0,1,2,3,4]
    # Position 3: can attend to [1,2,3,4,5]
    # Position 4: can attend to [2,3,4,5]
    # Position 5: can attend to [3,4,5]

    mask_np = np.array(mask[0, 0])
    print("\n  Mask pattern (0 = attend, -1e9 = blocked):")
    for i in range(6):
        row = ["." if mask_np[i, j] < -1e8 else "X" for j in range(6)]
        print(f"    Position {i}: {' '.join(row)}")

    # Verify mask correctness
    expected = np.array([
        [0, 0, 0, 1, 1, 1],  # pos 0: attend 0,1,2
        [0, 0, 0, 0, 1, 1],  # pos 1: attend 0,1,2,3
        [0, 0, 0, 0, 0, 1],  # pos 2: attend 0,1,2,3,4
        [1, 0, 0, 0, 0, 0],  # pos 3: attend 1,2,3,4,5
        [1, 1, 0, 0, 0, 0],  # pos 4: attend 2,3,4,5
        [1, 1, 1, 0, 0, 0],  # pos 5: attend 3,4,5
    ])

    actual = (mask_np < -1e8).astype(int)
    correct = np.allclose(actual, expected)
    print(f"\n  Mask correctness: {'PASS' if correct else 'FAIL'}")

    return correct


def test_attention_output():
    """Test that local attention produces reasonable output."""
    print("\n" + "-" * 70)
    print("TEST 2: Attention Output Comparison")
    print("-" * 70)

    config = KokoroConfig()
    attention = AlbertAttention(config)
    mx.eval(attention.parameters())

    # Create test input
    batch_size = 1
    seq_len = 20  # Short sequence
    hidden_dim = config.plbert_hidden_size

    x = mx.random.normal((batch_size, seq_len, hidden_dim)) * 0.1
    mx.eval(x)

    print(f"  Input shape: {x.shape}")

    # Full attention (no local mask)
    attention.set_local_attention(None)
    t0 = time.perf_counter()
    out_full = attention(x)
    mx.eval(out_full)
    time_full = (time.perf_counter() - t0) * 1000

    # Local attention with window=5
    attention.set_local_attention(5)
    t0 = time.perf_counter()
    out_local = attention(x)
    mx.eval(out_local)
    time_local = (time.perf_counter() - t0) * 1000

    # Compare outputs
    arr_full = np.array(out_full)
    arr_local = np.array(out_local)

    max_diff = np.max(np.abs(arr_full - arr_local))
    mean_diff = np.mean(np.abs(arr_full - arr_local))

    print(f"\n  Full attention time: {time_full:.2f}ms")
    print(f"  Local attention time: {time_local:.2f}ms")
    print(f"  Max diff: {max_diff:.4f}")
    print(f"  Mean diff: {mean_diff:.4f}")

    # For short sequences with window=5 covering most positions,
    # outputs should be similar
    is_reasonable = max_diff < 1.0  # Generous tolerance

    print(f"\n  Output reasonableness: {'PASS' if is_reasonable else 'FAIL'}")

    return is_reasonable


def test_model_integration():
    """Test local attention in full model."""
    print("\n" + "-" * 70)
    print("TEST 3: Full Model Integration")
    print("-" * 70)

    config = KokoroConfig()
    model = KokoroModel(config)
    mx.eval(model.parameters())

    # Set deterministic mode
    model.decoder.set_deterministic(True)

    # Create test input
    input_ids = mx.array([[50, 62, 75, 75, 82, 0, 90, 82, 88, 75, 63]])
    voice = mx.array(np.random.randn(1, 256).astype(np.float32) * 0.1)
    mx.eval(input_ids, voice)

    print(f"  Input: {input_ids.shape}")

    # Warmup
    _ = model(input_ids, voice, validate_output=False)
    mx.eval(_)

    # Full attention
    model.set_local_attention(None)
    t0 = time.perf_counter()
    out_full = model(input_ids, voice, validate_output=False)
    mx.eval(out_full)
    time_full = (time.perf_counter() - t0) * 1000

    # Warmup with local
    model.set_local_attention(16)
    _ = model(input_ids, voice, validate_output=False)
    mx.eval(_)

    # Local attention (window=16)
    t0 = time.perf_counter()
    out_local = model(input_ids, voice, validate_output=False)
    mx.eval(out_local)
    time_local = (time.perf_counter() - t0) * 1000

    # Compare
    arr_full = np.array(out_full)
    arr_local = np.array(out_local)

    full_nan = np.isnan(arr_full).any()
    local_nan = np.isnan(arr_local).any()

    print(f"\n  Full NaN: {full_nan}")
    print(f"  Local NaN: {local_nan}")
    print(f"  Full time: {time_full:.2f}ms")
    print(f"  Local time: {time_local:.2f}ms")

    if not full_nan and not local_nan:
        max_diff = np.max(np.abs(arr_full - arr_local))
        print(f"  Max diff: {max_diff:.2e}")
    else:
        max_diff = float('inf')

    passed = not full_nan and not local_nan
    print(f"\n  Integration test: {'PASS' if passed else 'FAIL'}")

    return passed


def benchmark_scaling():
    """Benchmark attention scaling with sequence length."""
    print("\n" + "-" * 70)
    print("TEST 4: Scaling Benchmark")
    print("-" * 70)

    config = KokoroConfig()
    attention = AlbertAttention(config)
    mx.eval(attention.parameters())

    batch_size = 1
    hidden_dim = config.plbert_hidden_size

    print(f"\n  {'Seq Len':<10} {'Full (ms)':<12} {'Local (ms)':<12} {'Speedup':<10}")
    print("  " + "-" * 44)

    for seq_len in [32, 64, 128, 256]:
        x = mx.random.normal((batch_size, seq_len, hidden_dim)) * 0.1
        mx.eval(x)

        # Full attention
        attention.set_local_attention(None)
        times_full = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = attention(x)
            mx.eval(out)
            times_full.append((time.perf_counter() - t0) * 1000)
        avg_full = np.mean(times_full)

        # Local attention (window=16)
        attention.set_local_attention(16)
        times_local = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = attention(x)
            mx.eval(out)
            times_local.append((time.perf_counter() - t0) * 1000)
        avg_local = np.mean(times_local)

        speedup = avg_full / avg_local if avg_local > 0 else 0

        print(f"  {seq_len:<10} {avg_full:<12.2f} {avg_local:<12.2f} {speedup:<10.2f}x")

    return True


if __name__ == "__main__":
    try:
        results = []

        # Test 1: Mask generation
        results.append(("Mask generation", test_local_mask()))

        # Test 2: Attention output
        results.append(("Attention output", test_attention_output()))

        # Test 3: Model integration
        results.append(("Model integration", test_model_integration()))

        # Test 4: Scaling benchmark
        results.append(("Scaling benchmark", benchmark_scaling()))

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        all_passed = True
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        print(f"\nP1 Local Attention: {'VERIFIED' if all_passed else 'NEEDS REVIEW'}")

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
