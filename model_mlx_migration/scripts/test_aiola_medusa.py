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
Test aiola/whisper-medusa-v1 weights with WhisperMLX.

This script evaluates whether the aiola Medusa heads (trained on whisper-large-v2)
can work with whisper-large-v3.

Key insight: aiola uses ResBlock architecture (1280→1280) with shared proj_out.
The Medusa head weights are model-agnostic - only proj_out differs between v2/v3.
Since v3 provides its own proj_out (51866 vocab), we just need to verify that
the v2-trained heads produce useful hidden state transformations.

Test cases:
1. Load weights and verify forward pass works
2. Run Medusa transcription on test audio
3. Compare outputs between Medusa and standard decoding
4. Benchmark speedup
"""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration')

import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def test_forward_pass():
    """Test that aiola Medusa weights load and forward pass works."""
    from tools.whisper_mlx import WhisperMLX

    print("\n" + "="*60)
    print("TEST 1: Forward Pass")
    print("="*60)

    weights_path = Path("/Users/ayates/model_mlx_migration/checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz")

    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Run: python scripts/convert_aiola_medusa.py first")
        return False

    # Load model
    print("Loading whisper-large-v3...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Load Medusa heads (11 heads, aiola architecture)
    print(f"Loading Medusa heads from {weights_path}...")
    model.load_medusa_heads(
        str(weights_path),
        n_heads=11,
        use_aiola=True,
        tree_structure=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],  # 11 levels for 11 heads
        top_k=5,
    )

    print(f"  - Medusa loaded: {model.medusa_loaded}")
    print(f"  - Number of heads: {model._medusa_module.n_heads}")
    print("  - Architecture: aiola ResBlock")

    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    dummy_hidden = mx.random.normal((1, 10, 1280))
    try:
        outputs = model._medusa_module(dummy_hidden)
        mx.eval(outputs)
        print(f"  - Input shape: {dummy_hidden.shape}")
        print(f"  - Output count: {len(outputs)}")
        print(f"  - Output shapes: {[o.shape for o in outputs]}")
        print("  - Forward pass: SUCCESS")
        return True
    except Exception as e:
        print(f"  - Forward pass FAILED: {e}")
        return False


def test_transcription():
    """Test Medusa transcription on test audio."""
    from tools.whisper_mlx import WhisperMLX

    print("\n" + "="*60)
    print("TEST 2: Medusa Transcription")
    print("="*60)

    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz"
    audio_path = "/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean/3570/5696/3570-5696-0002.flac"

    if not Path(audio_path).exists():
        print(f"ERROR: Test audio not found at {audio_path}")
        return False, None, None

    # Load model
    print("Loading whisper-large-v3...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Test standard transcription first (baseline)
    print("\nBaseline (standard decoding)...")
    t0 = time.perf_counter()
    baseline_result = model.transcribe(audio_path)
    baseline_time = time.perf_counter() - t0
    print(f"  - Time: {baseline_time*1000:.0f}ms")
    print(f"  - Text: {baseline_result['text'][:80]}...")

    # Load Medusa heads
    print("\nLoading Medusa heads...")
    model.load_medusa_heads(
        weights_path,
        n_heads=11,
        use_aiola=True,
        tree_structure=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        top_k=5,
    )

    # Test Medusa transcription
    print("\nMedusa decoding...")
    t0 = time.perf_counter()
    try:
        medusa_result = model.transcribe_medusa(audio_path, verbose=True)
        medusa_time = time.perf_counter() - t0
        print(f"  - Time: {medusa_time*1000:.0f}ms")
        print(f"  - Text: {medusa_result['text'][:80]}...")
        print(f"  - Acceptance rate: {medusa_result['acceptance_rate']:.1%}")
        print(f"  - Tokens per step: {medusa_result['tokens_per_step']:.2f}")

        # Compare outputs
        print("\nOutput comparison:")
        baseline_text = baseline_result['text'].strip()
        medusa_text = medusa_result['text'].strip()
        match = baseline_text == medusa_text
        print(f"  - Exact match: {match}")

        if not match:
            print(f"  - Baseline: {baseline_text[:100]}")
            print(f"  - Medusa:   {medusa_text[:100]}")

        return True, baseline_time, medusa_time
    except Exception as e:
        print(f"  - Medusa transcription FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, baseline_time, None


def test_v2_model():
    """Test with whisper-large-v2 (should work better since aiola was trained on v2)."""
    from tools.whisper_mlx import WhisperMLX

    print("\n" + "="*60)
    print("TEST 3: Test with whisper-large-v2 (native compatibility)")
    print("="*60)

    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz"
    audio_path = "/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean/3570/5696/3570-5696-0002.flac"

    if not Path(audio_path).exists():
        print("ERROR: Test audio not found")
        return False

    # Load model (v2 this time)
    print("Loading whisper-large-v2...")
    try:
        model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v2-mlx")
    except Exception as e:
        print(f"  - Could not load v2: {e}")
        print("  - Trying large-v3 instead...")
        model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    print(f"  - Vocab size: {model.config.n_vocab}")

    # Baseline
    print("\nBaseline (standard decoding)...")
    t0 = time.perf_counter()
    baseline_result = model.transcribe(audio_path)
    baseline_time = time.perf_counter() - t0
    print(f"  - Time: {baseline_time*1000:.0f}ms")
    print(f"  - Text: {baseline_result['text'][:80]}...")

    # Load Medusa and test
    print("\nLoading Medusa heads...")
    model.load_medusa_heads(
        weights_path,
        n_heads=11,
        use_aiola=True,
        tree_structure=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        top_k=5,
    )

    print("\nMedusa decoding...")
    t0 = time.perf_counter()
    try:
        medusa_result = model.transcribe_medusa(audio_path, verbose=True)
        medusa_time = time.perf_counter() - t0

        speedup = baseline_time / medusa_time if medusa_time > 0 else 0
        print("\nResults:")
        print(f"  - Baseline time: {baseline_time*1000:.0f}ms")
        print(f"  - Medusa time: {medusa_time*1000:.0f}ms")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Acceptance rate: {medusa_result['acceptance_rate']:.1%}")
        print(f"  - Tokens per step: {medusa_result['tokens_per_step']:.2f}")

        return True
    except Exception as e:
        print(f"  - Medusa FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_files():
    """Benchmark on multiple test files."""
    from tools.whisper_mlx import WhisperMLX

    print("\n" + "="*60)
    print("TEST 4: Multi-file Benchmark")
    print("="*60)

    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz"
    test_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")

    # Collect test files
    audio_files = list(test_dir.glob("**/*.flac"))[:5]  # First 5 files
    if not audio_files:
        print("ERROR: No test audio files found")
        return False

    print(f"Testing on {len(audio_files)} files...")

    # Load model
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
    model.load_medusa_heads(
        weights_path,
        n_heads=11,
        use_aiola=True,
        tree_structure=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        top_k=5,
    )

    baseline_times = []
    medusa_times = []
    acceptance_rates = []
    tokens_per_step = []
    matches = []

    for i, audio_file in enumerate(audio_files):
        print(f"\nFile {i+1}/{len(audio_files)}: {audio_file.name}")

        # Baseline
        model.unload_medusa_heads()
        t0 = time.perf_counter()
        baseline = model.transcribe(str(audio_file))
        baseline_time = time.perf_counter() - t0
        baseline_times.append(baseline_time)

        # Reload Medusa
        model.load_medusa_heads(
            weights_path,
            n_heads=11,
            use_aiola=True,
            tree_structure=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
            top_k=5,
        )

        # Medusa
        try:
            t0 = time.perf_counter()
            medusa = model.transcribe_medusa(str(audio_file))
            medusa_time = time.perf_counter() - t0
            medusa_times.append(medusa_time)
            acceptance_rates.append(medusa['acceptance_rate'])
            tokens_per_step.append(medusa['tokens_per_step'])
            matches.append(baseline['text'].strip() == medusa['text'].strip())

            speedup = baseline_time / medusa_time
            print(f"  Baseline: {baseline_time*1000:.0f}ms, Medusa: {medusa_time*1000:.0f}ms, Speedup: {speedup:.2f}x")
            print(f"  Acceptance: {medusa['acceptance_rate']:.1%}, TPS: {medusa['tokens_per_step']:.2f}")
            print(f"  Match: {matches[-1]}")
        except Exception as e:
            print(f"  Medusa FAILED: {e}")
            medusa_times.append(float('inf'))
            acceptance_rates.append(0)
            tokens_per_step.append(0)
            matches.append(False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_baseline = np.mean(baseline_times)
    avg_medusa = np.mean([t for t in medusa_times if t < float('inf')])
    avg_speedup = avg_baseline / avg_medusa if avg_medusa > 0 else 0
    avg_acceptance = np.mean([a for a in acceptance_rates if a > 0])
    avg_tps = np.mean([t for t in tokens_per_step if t > 0])
    match_rate = sum(matches) / len(matches) if matches else 0

    print(f"Average baseline time: {avg_baseline*1000:.0f}ms")
    print(f"Average Medusa time: {avg_medusa*1000:.0f}ms")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average acceptance rate: {avg_acceptance:.1%}")
    print(f"Average tokens per step: {avg_tps:.2f}")
    print(f"Output match rate: {match_rate:.0%}")

    return avg_speedup > 1.0


def main():
    print("Testing aiola/whisper-medusa-v1 with WhisperMLX")
    print("="*60)

    # Test 1: Forward pass
    pass_1 = test_forward_pass()

    # Test 2: Single file transcription
    pass_2, baseline_time, medusa_time = test_transcription()

    # Test 3: v2 model (skip for now, focus on v3)
    # pass_3 = test_v2_model()

    # Test 4: Multi-file benchmark (only if single file passed)
    if pass_2:
        pass_4 = test_multiple_files()
    else:
        pass_4 = False

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test 1 (Forward Pass): {'PASS' if pass_1 else 'FAIL'}")
    print(f"Test 2 (Transcription): {'PASS' if pass_2 else 'FAIL'}")
    print(f"Test 4 (Benchmark): {'PASS' if pass_4 else 'FAIL'}")

    if pass_2 and medusa_time:
        speedup = baseline_time / medusa_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1.0:
            print("CONCLUSION: aiola Medusa heads provide speedup on v3")
        else:
            print("CONCLUSION: aiola Medusa heads are SLOWER on v3 - need fine-tuning")
    else:
        print("\nCONCLUSION: Medusa transcription failed - need debugging")


if __name__ == "__main__":
    main()
