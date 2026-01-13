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
Verify speculative decoding v2 speedup across multiple audio samples.

Tests:
1. Speedup is consistent across different audio files
2. Output is LOSSLESS (identical to standard decoding)
3. Acceptance rate is ≥70%
4. Total speedup is ≥1.3x (gate criteria)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx


def find_test_files(max_files: int = 10) -> list:
    """Find LibriSpeech test files."""
    librispeech_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")
    files = list(librispeech_dir.glob("**/*.flac"))[:max_files]

    if not files:
        # Fall back to test fixture
        test_audio = Path("/Users/ayates/model_mlx_migration/tests/fixtures/audio/test_speech.wav")
        if test_audio.exists():
            return [test_audio]
    return files


def verify_speculative_decoding():
    """Run comprehensive verification of speculative decoding."""
    from tools.whisper_mlx import WhisperMLX

    print("=" * 70)
    print("SPECULATIVE DECODING V2 VERIFICATION")
    print("=" * 70)
    print()

    # Find test files
    test_files = find_test_files(max_files=5)
    print(f"Found {len(test_files)} test files")
    print()

    # Load model
    print("Loading main model (large-v3)...")
    t0 = time.perf_counter()
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
    mx.eval(model.parameters())
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
    print()

    # Load draft model
    print("Loading draft model (distil-whisper-large-v3)...")
    t0 = time.perf_counter()
    model.load_draft_model(
        draft_model_name="mlx-community/distil-whisper-large-v3",
        draft_tokens=5,
    )
    mx.eval(model._draft_model.parameters())
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
    print()

    # Warmup
    print("Warmup runs...")
    _ = model.transcribe(str(test_files[0]))
    _ = model.transcribe_speculative(str(test_files[0]))
    print()

    # Test each file
    results = []
    print("Testing files...")
    print("-" * 70)

    for i, audio_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] {audio_file.name}")

        # Standard decoding
        t0 = time.perf_counter()
        result_std = model.transcribe(str(audio_file))
        time_std = time.perf_counter() - t0

        # Speculative decoding
        t0 = time.perf_counter()
        result_spec = model.transcribe_speculative(str(audio_file))
        time_spec = time.perf_counter() - t0

        speedup = time_std / time_spec
        accept_rate = result_spec.get("acceptance_rate", 0)
        text_match = result_std["text"].strip() == result_spec["text"].strip()

        print(f"  Standard:    {time_std*1000:.1f}ms")
        print(f"  Speculative: {time_spec*1000:.1f}ms")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Accept rate: {accept_rate:.1%}")
        print(f"  Text match:  {'✓ PASS' if text_match else '✗ FAIL'}")

        if not text_match:
            print(f"    STD:  {result_std['text'][:60]}...")
            print(f"    SPEC: {result_spec['text'][:60]}...")

        results.append({
            "file": audio_file.name,
            "time_std": time_std,
            "time_spec": time_spec,
            "speedup": speedup,
            "accept_rate": accept_rate,
            "text_match": text_match,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_accept = sum(r["accept_rate"] for r in results) / len(results)
    all_match = all(r["text_match"] for r in results)

    print(f"Files tested:       {len(results)}")
    print(f"Average speedup:    {avg_speedup:.2f}x")
    print(f"Average acceptance: {avg_accept:.1%}")
    print(f"All texts match:    {'✓ YES' if all_match else '✗ NO'}")
    print()

    # Gate criteria
    print("GATE CRITERIA:")
    gate_speedup = avg_speedup >= 1.3
    gate_accept = avg_accept >= 0.70
    gate_lossless = all_match

    print(f"  [{'✓' if gate_speedup else '✗'}] Speedup ≥1.3x:     {avg_speedup:.2f}x")
    print(f"  [{'✓' if gate_accept else '✗'}] Acceptance ≥70%:   {avg_accept:.1%}")
    print(f"  [{'✓' if gate_lossless else '✗'}] Lossless output:  {all_match}")
    print()

    all_pass = gate_speedup and gate_accept and gate_lossless
    if all_pass:
        print("RESULT: ✓ ALL GATES PASSED - Speculative decoding v2 is VERIFIED")
    else:
        print("RESULT: ✗ SOME GATES FAILED - See above")

    return all_pass, avg_speedup, avg_accept


if __name__ == "__main__":
    success, speedup, acceptance = verify_speculative_decoding()
    sys.exit(0 if success else 1)
