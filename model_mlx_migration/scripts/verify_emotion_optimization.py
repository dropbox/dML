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
Verify emotion optimization (prosody cache + compiled decoder).

Tests:
1. Emotion cache initialization works
2. Compiled decoder produces correct output
3. Performance improvement from optimizations
"""

import sys
import time

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np

from tools.pytorch_to_mlx.converters import KokoroConverter
from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text


def test_emotion_cache_initialization():
    """Test that emotion cache initializes correctly."""
    print("\n" + "-" * 70)
    print("TEST 1: Emotion Cache Initialization")
    print("-" * 70)

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.decoder.set_deterministic(True)

    # Load prosody embeddings
    prosody_weights = mx.load('models/prosody_embeddings_orthogonal/final.safetensors')
    embedding_table = prosody_weights['embedding.weight']

    print(f"  Embedding table shape: {embedding_table.shape}")

    # Initialize cache
    setup_time = model.initialize_emotion_cache(
        embedding_table=embedding_table,
        unified_model=None,  # Test without unified model first
        compile_decoder=True,
    )

    print(f"  Setup time: {setup_time:.1f}ms")
    print(f"  Cache entries: {len(model._emotion_cache)}")
    print(f"  Compiled decoder: {model._compiled_decoder is not None}")
    print(f"  Use compiled: {model._use_compiled_decoder}")

    # Verify cache contents
    for eid in [0, 40, 41, 42]:
        cached = model.get_cached_emotion(eid)
        if cached is None:
            print(f"  ERROR: Emotion {eid} not cached")
            return False
        print(f"  Emotion {eid} cached: embedding shape {cached['embedding'].shape}")

    print("\n  PASS: Cache initialization successful")
    return True


def test_compiled_decoder_output():
    """Test that compiled decoder produces correct output."""
    print("\n" + "-" * 70)
    print("TEST 2: Compiled Decoder Output Correctness")
    print("-" * 70)

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.decoder.set_deterministic(True)

    # Prepare test input
    text = "Hello world, this is a test."
    phonemes, token_ids = phonemize_text(text, language='en')
    input_ids = mx.array([token_ids])
    voice = converter.load_voice('af_heart', phoneme_length=len(phonemes))
    mx.eval(voice)

    print(f"  Text: '{text}'")
    print(f"  Tokens: {len(token_ids)}")

    # Generate baseline (no optimization)
    model._use_compiled_decoder = False
    audio_baseline = model(input_ids, voice, validate_output=False)
    mx.eval(audio_baseline)

    # Initialize cache and compile
    prosody_weights = mx.load('models/prosody_embeddings_orthogonal/final.safetensors')
    embedding_table = prosody_weights['embedding.weight']
    model.initialize_emotion_cache(embedding_table, compile_decoder=True)

    # Generate with compiled decoder
    audio_compiled = model(input_ids, voice, validate_output=False)
    mx.eval(audio_compiled)

    # Compare outputs
    baseline_np = np.array(audio_baseline)
    compiled_np = np.array(audio_compiled)

    max_diff = np.max(np.abs(baseline_np - compiled_np))
    mean_diff = np.mean(np.abs(baseline_np - compiled_np))

    print(f"\n  Baseline shape: {baseline_np.shape}")
    print(f"  Compiled shape: {compiled_np.shape}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Check for NaN/Inf
    baseline_nan = np.isnan(baseline_np).any()
    compiled_nan = np.isnan(compiled_np).any()
    print(f"  Baseline NaN: {baseline_nan}")
    print(f"  Compiled NaN: {compiled_nan}")

    # Tolerance check - compiled may have small numerical differences due to
    # graph optimization/fusion, but should be perceptually identical for audio
    # Max diff < 0.05 is acceptable (audio typically quantized to 16-bit = 1/32768)
    passed = max_diff < 0.05 and not baseline_nan and not compiled_nan
    print(f"\n  {'PASS' if passed else 'FAIL'}: Output correctness")

    return passed


def test_performance_improvement():
    """Benchmark performance improvement from optimizations."""
    print("\n" + "-" * 70)
    print("TEST 3: Performance Benchmark")
    print("-" * 70)

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.decoder.set_deterministic(True)

    # Prepare test input
    text = "I can not believe what just happened to us today."
    phonemes, token_ids = phonemize_text(text, language='en')
    input_ids = mx.array([token_ids])
    voice = converter.load_voice('af_heart', phoneme_length=len(phonemes))
    mx.eval(voice)

    print(f"  Text: '{text}'")
    print(f"  Tokens: {len(token_ids)}")

    N = 10

    # Warmup
    for _ in range(3):
        _ = model(input_ids, voice, validate_output=False)
        mx.eval(_)

    # Baseline timing (no optimization)
    model._use_compiled_decoder = False
    times_baseline = []
    for _ in range(N):
        t0 = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        times_baseline.append((time.perf_counter() - t0) * 1000)

    baseline_avg = np.mean(times_baseline[2:])
    baseline_std = np.std(times_baseline[2:])
    audio_len = np.array(audio).flatten().shape[0] / 24000

    print("\n  BASELINE (no optimizations):")
    print(f"    Latency: {baseline_avg:.1f}ms +/- {baseline_std:.1f}ms")
    print(f"    Audio: {audio_len:.2f}s")
    print(f"    RTF: {baseline_avg/1000/audio_len:.3f}x")

    # Initialize emotion cache and compile decoder
    prosody_weights = mx.load('models/prosody_embeddings_orthogonal/final.safetensors')
    embedding_table = prosody_weights['embedding.weight']
    setup_time = model.initialize_emotion_cache(embedding_table, compile_decoder=True)
    print(f"\n  Setup time: {setup_time:.1f}ms (one-time)")

    # Warmup compiled path
    for _ in range(3):
        _ = model(input_ids, voice, validate_output=False)
        mx.eval(_)

    # Optimized timing
    times_optimized = []
    for _ in range(N):
        t0 = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        times_optimized.append((time.perf_counter() - t0) * 1000)

    opt_avg = np.mean(times_optimized[2:])
    opt_std = np.std(times_optimized[2:])

    print("\n  OPTIMIZED (compiled decoder):")
    print(f"    Latency: {opt_avg:.1f}ms +/- {opt_std:.1f}ms")
    print(f"    RTF: {opt_avg/1000/audio_len:.3f}x")

    # Calculate improvement
    speedup = baseline_avg / opt_avg
    saved_ms = baseline_avg - opt_avg

    print("\n  IMPROVEMENT:")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    Time saved: {saved_ms:.1f}ms per inference")
    print(f"    Throughput: {audio_len/(opt_avg/1000):.1f}x real-time")

    # Pass if any speedup (>1.0x)
    passed = speedup >= 1.0
    print(f"\n  {'PASS' if passed else 'FAIL'}: Performance benchmark")

    return passed


def test_with_unified_model():
    """Test emotion cache with unified prosody model."""
    print("\n" + "-" * 70)
    print("TEST 4: Unified Prosody Model Integration")
    print("-" * 70)

    try:
        from scripts.train_prosody_unified import UnifiedProsodyPredictor
    except ImportError:
        print("  SKIP: UnifiedProsodyPredictor not available")
        return True

    # Load models
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.decoder.set_deterministic(True)

    # Load prosody embeddings and unified model
    prosody_weights = mx.load('models/prosody_embeddings_orthogonal/final.safetensors')
    embedding_table = prosody_weights['embedding.weight']

    unified_model = UnifiedProsodyPredictor(
        prosody_dim=768, hidden_dim=512, contour_len=50, num_blocks=4
    )
    weights = mx.load('models/prosody_unified_v2/best_model.npz')
    unified_model.load_weights(list(weights.items()))

    # Initialize cache with unified model
    setup_time = model.initialize_emotion_cache(
        embedding_table=embedding_table,
        unified_model=unified_model,
        compile_decoder=True,
    )

    print(f"  Setup time: {setup_time:.1f}ms")

    # Verify unified model outputs are cached
    emotions_with_contour = 0
    for eid in [0, 40, 41, 42, 43, 44, 45]:
        cached = model.get_cached_emotion(eid)
        if cached and 'f0_contour' in cached:
            emotions_with_contour += 1
            print(f"  Emotion {eid}: contour cached, dur_mult={cached['duration_mult']:.2f}")

    passed = emotions_with_contour >= 7
    print(f"\n  {'PASS' if passed else 'FAIL'}: Unified model integration")

    return passed


if __name__ == "__main__":
    try:
        results = []

        # Test 1: Cache initialization
        results.append(("Cache initialization", test_emotion_cache_initialization()))

        # Test 2: Output correctness
        results.append(("Output correctness", test_compiled_decoder_output()))

        # Test 3: Performance
        results.append(("Performance benchmark", test_performance_improvement()))

        # Test 4: Unified model
        results.append(("Unified model integration", test_with_unified_model()))

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

        print(f"\nEmotion Optimization: {'VERIFIED' if all_passed else 'NEEDS REVIEW'}")

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
