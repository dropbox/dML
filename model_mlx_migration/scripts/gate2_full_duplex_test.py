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
Gate 2: Full Duplex Validation Test

Tests the system's ability to:
1. Reject its own TTS output (echo cancellation)
2. Recognize user speech during TTS playback (barge-in)
3. Cancel TTS within latency target (<80ms)

Pass Criteria (from STREAMING_STT_MASTER_PLAN.md):
- TTS playing + user silent = empty transcript
- TTS playing + user speaks = user tracked, barge-in works
- Barge-in latency <80ms p95

Usage:
    python scripts/gate2_full_duplex_test.py
    python scripts/gate2_full_duplex_test.py --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_speech(
    duration_s: float,
    sample_rate: int = 16000,
    freq_hz: float = 440,
    modulation: bool = True,
) -> np.ndarray:
    """Generate synthetic speech-like signal for testing.

    Uses AM/FM modulation to create more realistic speech-like characteristics.
    """
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))

    # Base carrier
    signal = np.sin(2 * np.pi * freq_hz * t)

    if modulation:
        # Amplitude modulation (speech envelope)
        envelope = 0.5 * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
        signal = signal * envelope

        # Add harmonics for speech-like quality
        signal += 0.3 * np.sin(2 * np.pi * freq_hz * 2 * t)  # 2nd harmonic
        signal += 0.15 * np.sin(2 * np.pi * freq_hz * 3 * t)  # 3rd harmonic

    return signal.astype(np.float32)


def test_echo_rejection(
    canceller,
    tts_audio: np.ndarray,
    sample_rate: int,
    room_delay_ms: float = 50,
    echo_gain: float = 0.3,
    verbose: bool = False,
) -> dict:
    """Test 1: TTS playing + user silent = empty/minimal output

    Simulates TTS audio playing through speaker and picked up by microphone.
    After echo cancellation, result should have minimal energy.
    """

    # Add TTS to reference buffer (simulating TTS playback)
    canceller.add_reference(tts_audio)

    # Simulate mic input: delayed and attenuated TTS (room echo)
    delay_samples = int(room_delay_ms * sample_rate / 1000)
    mic_input = np.zeros(len(tts_audio) + delay_samples, dtype=np.float32)
    mic_input[delay_samples:delay_samples + len(tts_audio)] = echo_gain * tts_audio

    # Add small amount of room noise
    noise = 0.001 * np.random.randn(len(mic_input)).astype(np.float32)
    mic_input += noise

    # Process through echo canceller
    result = canceller.process(mic_input[delay_samples:])

    # Calculate residual energy
    input_rms = np.sqrt(np.mean(mic_input[delay_samples:]**2))
    output_rms = np.sqrt(np.mean(result.cleaned_audio**2))
    reduction_ratio = output_rms / max(input_rms, 1e-8)

    # Pass if residual is <10% of input (>20dB reduction)
    passed = reduction_ratio < 0.1

    if verbose:
        print(f"\n  Input RMS: {input_rms:.4f}")
        print(f"  Output RMS: {output_rms:.4f}")
        print(f"  Reduction ratio: {reduction_ratio:.4f} (target <0.1)")
        print(f"  Echo reduction: {result.echo_reduction_db:.1f}dB")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")

    return {
        "test": "echo_rejection",
        "passed": passed,
        "input_rms": float(input_rms),
        "output_rms": float(output_rms),
        "reduction_ratio": float(reduction_ratio),
        "echo_reduction_db": float(result.echo_reduction_db),
        "processing_time_ms": float(result.processing_time_ms),
    }


def test_barge_in_detection(
    canceller,
    tts_audio: np.ndarray,
    user_audio: np.ndarray,
    sample_rate: int,
    room_delay_ms: float = 50,
    echo_gain: float = 0.3,
    user_delay_ms: float = 500,  # User starts speaking 500ms after TTS starts
    verbose: bool = False,
) -> dict:
    """Test 2: TTS playing + user speaks = user speech preserved

    Simulates user speaking while TTS is playing. After echo cancellation,
    user speech should be preserved while TTS echo is removed.
    """
    # Add TTS to reference buffer
    canceller.add_reference(tts_audio)

    # Create mic input with both TTS echo and user speech
    delay_samples = int(room_delay_ms * sample_rate / 1000)
    user_start_samples = int(user_delay_ms * sample_rate / 1000)

    # Calculate total length
    total_length = max(len(tts_audio) + delay_samples, user_start_samples + len(user_audio))
    mic_input = np.zeros(total_length, dtype=np.float32)

    # Add TTS echo
    mic_input[delay_samples:delay_samples + len(tts_audio)] += echo_gain * tts_audio

    # Add user speech
    user_end = min(user_start_samples + len(user_audio), total_length)
    user_len = user_end - user_start_samples
    mic_input[user_start_samples:user_end] += user_audio[:user_len]

    # Process through echo canceller
    result = canceller.process(mic_input[delay_samples:])

    # Measure user speech preservation
    # User speech region starts at (user_start_samples - delay_samples)
    user_region_start = max(0, user_start_samples - delay_samples)
    user_region_end = min(len(result.cleaned_audio), user_region_start + len(user_audio))

    if user_region_end > user_region_start:
        user_region = result.cleaned_audio[user_region_start:user_region_end]
        user_rms = np.sqrt(np.mean(user_region**2))

        # Compare with original user audio
        original_user_rms = np.sqrt(np.mean(user_audio[:user_region_end - user_region_start]**2))
        preservation_ratio = user_rms / max(original_user_rms, 1e-8)
    else:
        preservation_ratio = 0.0

    # Pass if user speech is >50% preserved
    passed = preservation_ratio > 0.5

    if verbose:
        print(f"\n  User speech preservation ratio: {preservation_ratio:.2f} (target >0.5)")
        print(f"  Echo reduction: {result.echo_reduction_db:.1f}dB")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")

    return {
        "test": "barge_in_detection",
        "passed": passed,
        "preservation_ratio": float(preservation_ratio),
        "echo_reduction_db": float(result.echo_reduction_db),
        "processing_time_ms": float(result.processing_time_ms),
    }


def test_barge_in_latency(
    canceller,
    sample_rate: int,
    num_iterations: int = 50,
    target_latency_ms: float = 80,
    verbose: bool = False,
) -> dict:
    """Test 3: Barge-in detection latency <80ms p95

    Measures the time to process audio through echo cancellation.
    """
    latencies = []

    # Generate test signals
    tts_duration = 1.0
    user_duration = 0.5
    tts_audio = generate_synthetic_speech(tts_duration, sample_rate, freq_hz=440)
    user_audio = generate_synthetic_speech(user_duration, sample_rate, freq_hz=220)

    for _ in range(num_iterations):
        canceller.clear()
        canceller.add_reference(tts_audio)

        # Create mixed input
        mic_input = 0.3 * tts_audio[:len(user_audio)] + user_audio

        # Time the processing
        start = time.perf_counter()
        result = canceller.process(mic_input)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    passed = p95 < target_latency_ms

    if verbose:
        print(f"\n  Iterations: {num_iterations}")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P95 latency: {p95:.2f}ms (target <{target_latency_ms}ms)")
        print(f"  Min: {np.min(latencies):.2f}ms, Max: {np.max(latencies):.2f}ms")

    return {
        "test": "barge_in_latency",
        "passed": passed,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "target_ms": float(target_latency_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 2: Full Duplex Validation Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--iterations", type=int, default=50, help="Latency test iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("Gate 2: Full Duplex Validation Test")
    print("=" * 70)

    # Import echo canceller
    from tools.dashvoice.echo_cancel import EchoCanceller

    sample_rate = 16000  # STT typically uses 16kHz

    # Create and warm up echo canceller
    print("\n1. Initializing echo canceller...")
    canceller = EchoCanceller(sample_rate=sample_rate)
    canceller.warmup()
    print("   Warmup complete")

    # Generate test signals
    print("\n2. Generating test signals...")
    tts_audio = generate_synthetic_speech(2.0, sample_rate, freq_hz=440)
    user_audio = generate_synthetic_speech(1.0, sample_rate, freq_hz=220)
    print(f"   TTS audio: {len(tts_audio)/sample_rate:.1f}s")
    print(f"   User audio: {len(user_audio)/sample_rate:.1f}s")

    results = []

    # Test 1: Echo rejection (TTS only, no user)
    print("\n" + "-" * 70)
    print("TEST 1: Echo Rejection (TTS playing + user silent)")
    print("-" * 70)
    canceller.clear()
    result1 = test_echo_rejection(
        canceller, tts_audio, sample_rate, verbose=args.verbose
    )
    results.append(result1)
    status = "PASS" if result1["passed"] else "FAIL"
    print(f"Result: {status} (residual ratio: {result1['reduction_ratio']:.3f})")

    # Test 2: Barge-in detection (TTS + user speech)
    print("\n" + "-" * 70)
    print("TEST 2: Barge-in Detection (TTS playing + user speaks)")
    print("-" * 70)
    canceller.clear()
    result2 = test_barge_in_detection(
        canceller, tts_audio, user_audio, sample_rate, verbose=args.verbose
    )
    results.append(result2)
    status = "PASS" if result2["passed"] else "FAIL"
    print(f"Result: {status} (user preservation: {result2['preservation_ratio']:.2f})")

    # Test 3: Latency
    print("\n" + "-" * 70)
    print("TEST 3: Barge-in Latency (<80ms p95)")
    print("-" * 70)
    result3 = test_barge_in_latency(
        canceller, sample_rate, num_iterations=args.iterations, verbose=args.verbose
    )
    results.append(result3)
    status = "PASS" if result3["passed"] else "FAIL"
    print(f"Result: {status} (p95: {result3['p95_ms']:.2f}ms)")

    # Summary
    print("\n" + "=" * 70)
    print("GATE 2 SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    print()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['test']}")

    if passed == total:
        print("\n*** GATE 2: PASS ***")
        return 0
    else:
        print(f"\n*** GATE 2: FAIL ({passed}/{total} tests passed) ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
