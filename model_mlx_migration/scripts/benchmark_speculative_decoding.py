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
Benchmark speculative decoding for WhisperMLX (OPT-W5).

Compares:
1. Standard WhisperMLX decoding (baseline)
2. Speculative decoding with distil-whisper-large-v3

Expected speedup: 1.5-2x for decoder (decoding is ~90% of total time)
"""

import sys
import time

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx


def benchmark_speculative():
    """Run speculative decoding benchmark."""
    from tools.whisper_mlx import WhisperMLX

    # Test audio
    test_audio = "tests/fixtures/audio/test_speech.wav"

    print("=" * 60)
    print("WhisperMLX Speculative Decoding Benchmark (OPT-W5)")
    print("=" * 60)
    print()

    # Load main model
    print("Loading main model (large-v3)...")
    t0 = time.perf_counter()
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
    mx.eval(model.parameters())
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
    print()

    # Warmup
    print("Warmup run (standard)...")
    _ = model.transcribe(test_audio)
    print()

    # Benchmark standard decoding
    print("Benchmarking standard decoding...")
    times_standard = []
    for i in range(3):
        t0 = time.perf_counter()
        result_standard = model.transcribe(test_audio)
        elapsed = time.perf_counter() - t0
        times_standard.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms")

    avg_standard = sum(times_standard) / len(times_standard)
    print(f"  Average: {avg_standard*1000:.1f}ms")
    print(f"  Text: {result_standard['text'][:80]}...")
    print()

    # Load draft model for speculative decoding
    print("Loading draft model (distil-whisper-large-v3)...")
    t0 = time.perf_counter()
    model.load_draft_model(
        draft_model_name="mlx-community/distil-whisper-large-v3",
        draft_tokens=5,
    )
    mx.eval(model._draft_model.parameters())
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
    print()

    # Warmup speculative
    print("Warmup run (speculative)...")
    _ = model.transcribe_speculative(test_audio)
    print()

    # Benchmark speculative decoding
    print("Benchmarking speculative decoding...")
    times_speculative = []
    acceptance_rates = []
    tokens_per_iter = []
    for i in range(3):
        t0 = time.perf_counter()
        result_speculative = model.transcribe_speculative(test_audio)
        elapsed = time.perf_counter() - t0
        times_speculative.append(elapsed)
        acceptance_rates.append(result_speculative.get("acceptance_rate", 0))
        tokens_per_iter.append(result_speculative.get("tokens_per_iteration", 0))
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms (accept: {acceptance_rates[-1]:.1%})")

    avg_speculative = sum(times_speculative) / len(times_speculative)
    avg_acceptance = sum(acceptance_rates) / len(acceptance_rates)
    avg_tokens_per_iter = sum(tokens_per_iter) / len(tokens_per_iter)
    print(f"  Average: {avg_speculative*1000:.1f}ms")
    print(f"  Avg acceptance rate: {avg_acceptance:.1%}")
    print(f"  Avg tokens/iteration: {avg_tokens_per_iter:.2f}")
    print(f"  Text: {result_speculative['text'][:80]}...")
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Standard decoding:    {avg_standard*1000:.1f}ms")
    print(f"Speculative decoding: {avg_speculative*1000:.1f}ms")
    speedup = avg_standard / avg_speculative
    print(f"Speedup: {speedup:.2f}x")
    print()

    # Text comparison
    print("Text comparison:")
    text_standard = result_standard["text"].strip()
    text_speculative = result_speculative["text"].strip()
    if text_standard == text_speculative:
        print("  MATCH - Speculative decoding is LOSSLESS")
    else:
        print("  MISMATCH - Need investigation")
        print(f"  Standard:    {text_standard}")
        print(f"  Speculative: {text_speculative}")
    print()

    return speedup, avg_acceptance


def benchmark_variable_length():
    """Benchmark with variable-length mode."""
    from tools.whisper_mlx import WhisperMLX

    test_audio = "tests/fixtures/audio/test_speech.wav"

    print("=" * 60)
    print("Variable-Length + Speculative Decoding")
    print("=" * 60)
    print()

    # Load model with draft model
    print("Loading models...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
    model.load_draft_model(draft_tokens=5)
    mx.eval(model.parameters())
    mx.eval(model._draft_model.parameters())
    print()

    # Warmup
    _ = model.transcribe(test_audio, variable_length=True)
    _ = model.transcribe_speculative(test_audio, variable_length=True)

    # Benchmark standard + variable-length
    print("Standard + Variable-length mode:")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        model.transcribe(test_audio, variable_length=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg_var = sum(times) / len(times)
    print(f"  Average: {avg_var*1000:.1f}ms")
    print()

    # Benchmark speculative + variable-length
    print("Speculative + Variable-length mode:")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        result_spec = model.transcribe_speculative(test_audio, variable_length=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg_spec_var = sum(times) / len(times)
    print(f"  Average: {avg_spec_var*1000:.1f}ms")
    print(f"  Acceptance rate: {result_spec.get('acceptance_rate', 0):.1%}")
    print()

    print("Variable-length speedup:")
    print(f"  Standard -> Speculative: {avg_var / avg_spec_var:.2f}x")
    print()

    return avg_var / avg_spec_var


if __name__ == "__main__":
    speedup, acceptance = benchmark_speculative()
    print()

    speedup_var = benchmark_variable_length()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Standard mode speedup: {speedup:.2f}x")
    print(f"Variable-length mode speedup: {speedup_var:.2f}x")
    print(f"Draft acceptance rate: {acceptance:.1%}")

    if speedup > 1.2:
        print("\nOPT-W5 SUCCESS: Speculative decoding provides meaningful speedup")
    else:
        print("\nOPT-W5 NEEDS INVESTIGATION: Speedup lower than expected")
