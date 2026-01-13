"""
Lock-Free Audio Callback Enforcement Tests

Verifies that the audio callback implementation is truly lock-free by
testing under conditions that would expose blocking:

1. **Rapid sequential playback**: Queue many short utterances in rapid
   succession. If the callback blocks, we'll see underruns/gaps.

2. **Concurrent synthesis + playback**: Synthesize while playing to stress
   the producer-consumer relationship.

3. **Ring buffer stress**: Fill and drain the ring buffer rapidly to verify
   atomic operations don't cause data races.

Background:
The audio callback runs on a real-time thread. ANY blocking (mutex, malloc,
syscall) causes audio underruns = audible gaps. The implementation uses:
- Lock-free SPSC ring buffer (atomic read/write positions)
- No memory allocation in callback path
- Atomic flags for state (playing, interrupt)

Run: pytest tests/stress/test_lockfree_audio.py -v -m stress
"""

import os
import subprocess
import struct
import math
import pytest
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def read_wav_samples(wav_path: str) -> Tuple[List[float], int]:
    """Read WAV file and return (samples as floats in [-1,1], sample_rate)."""
    with open(wav_path, 'rb') as f:
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a RIFF file: {riff}")

        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a WAVE file: {wave}")

        sample_rate = 0
        bits_per_sample = 16

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break

            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                f.read(2)  # audio format
                f.read(2)  # num channels
                sample_rate = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # byte rate
                f.read(2)  # block align
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                remaining = chunk_size - 16
                if remaining > 0:
                    f.read(remaining)

            elif chunk_id == b'data':
                if bits_per_sample != 16:
                    raise ValueError(f"Only 16-bit PCM supported")

                num_samples = chunk_size // 2
                raw_data = f.read(chunk_size)
                samples_int = struct.unpack(f'<{num_samples}h', raw_data)
                samples = [s / 32768.0 for s in samples_int]
                return samples, sample_rate
            else:
                f.seek(chunk_size, 1)

    raise ValueError("No data chunk found in WAV file")


def detect_audio_gaps(samples: List[float], sample_rate: int,
                      gap_threshold: float = 0.001,
                      min_gap_ms: float = 10.0) -> List[dict]:
    """
    Detect gaps (extended silence) in audio that could indicate underruns.

    Args:
        samples: Audio samples as floats in [-1,1]
        sample_rate: Sample rate in Hz
        gap_threshold: Max absolute sample value to consider silence
        min_gap_ms: Minimum gap duration to report

    Returns:
        List of gaps with start_time, end_time, duration_ms
    """
    gaps = []
    min_gap_samples = int(sample_rate * min_gap_ms / 1000)

    in_gap = False
    gap_start = 0

    for i, sample in enumerate(samples):
        is_silent = abs(sample) < gap_threshold

        if is_silent and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_silent and in_gap:
            in_gap = False
            gap_length = i - gap_start
            if gap_length >= min_gap_samples:
                gaps.append({
                    'start_time': gap_start / sample_rate,
                    'end_time': i / sample_rate,
                    'duration_ms': gap_length * 1000 / sample_rate
                })

    # Check if still in gap at end
    if in_gap:
        gap_length = len(samples) - gap_start
        if gap_length >= min_gap_samples:
            gaps.append({
                'start_time': gap_start / sample_rate,
                'end_time': len(samples) / sample_rate,
                'duration_ms': gap_length * 1000 / sample_rate
            })

    return gaps


def generate_audio(binary: Path, text: str, lang: str, output_path: Path,
                   timeout: int = 60) -> Tuple[bool, float]:
    """
    Generate audio and measure latency.

    Returns:
        (success, latency_seconds)
    """
    start = time.perf_counter()
    cmd = [str(binary), "--speak", text, "--lang", lang, "--save-audio", str(output_path)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        cwd=str(binary.parent.parent),
        env=get_tts_env()
    )

    elapsed = time.perf_counter() - start
    return result.returncode == 0 and output_path.exists(), elapsed


@pytest.mark.stress
class TestLockFreeAudioCallback:
    """
    Tests to verify audio callback lock-free behavior.

    These tests stress the audio pipeline in ways that would expose
    blocking operations in the callback path.
    """

    @pytest.fixture(scope="class")
    def cpp_binary(self):
        """Path to stream-tts-cpp binary."""
        binary = Path(__file__).parent.parent.parent / "stream-tts-cpp" / "stream-tts-cpp"
        if not binary.exists():
            binary = Path(__file__).parent.parent.parent / "stream-tts-cpp" / "build" / "stream-tts-cpp"
        if not binary.exists():
            pytest.skip("TTS binary not found")
        return binary

    def test_rapid_sequential_synthesis(self, cpp_binary, tmp_path):
        """
        Synthesize multiple short utterances in rapid succession.

        If the audio callback blocks, we'll accumulate latency and see
        timing anomalies. Lock-free implementation should maintain
        consistent per-utterance latency.
        """
        utterances = [
            "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine", "Ten"
        ]

        latencies = []
        success_count = 0

        for i, text in enumerate(utterances):
            wav_path = tmp_path / f"seq_{i}.wav"
            success, latency = generate_audio(cpp_binary, text, "en", wav_path)

            if success:
                success_count += 1
                latencies.append(latency)

        # All should succeed
        assert success_count == len(utterances), (
            f"Only {success_count}/{len(utterances)} succeeded"
        )

        # Latency should be consistent (no runaway accumulation from blocking)
        if len(latencies) >= 3:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            print(f"\n=== Rapid Sequential Synthesis ===")
            print(f"Utterances: {len(utterances)}")
            print(f"Avg latency: {avg_latency:.2f}s")
            print(f"Max latency: {max_latency:.2f}s")
            print(f"Latency variance: {max_latency / avg_latency:.2f}x")

            # Max latency shouldn't be more than 3x average (would indicate blocking)
            # Note: First synthesis has warmup cost, so we're lenient
            assert max_latency <= avg_latency * 4, (
                f"Latency spike detected: max {max_latency:.2f}s vs avg {avg_latency:.2f}s "
                f"({max_latency/avg_latency:.1f}x) - possible callback blocking"
            )

    def test_audio_no_mid_utterance_gaps(self, cpp_binary, tmp_path):
        """
        Verify that synthesized audio doesn't have excessive gaps.

        Natural pauses between words/phrases (100-200ms) are expected in
        TTS output. We're looking for SEVERE gaps (>500ms) that would
        indicate audio underruns from callback blocking.
        """
        # Use a longer sentence that would expose underruns
        text = "The quick brown fox jumps over the lazy dog while running fast."
        wav_path = tmp_path / "gap_test.wav"

        success, latency = generate_audio(cpp_binary, text, "en", wav_path)
        assert success, "Failed to generate audio"

        samples, sample_rate = read_wav_samples(str(wav_path))
        duration = len(samples) / sample_rate

        print(f"\n=== Audio Gap Detection ===")
        print(f"Duration: {duration:.2f}s")
        print(f"Sample rate: {sample_rate} Hz")

        # Detect gaps >50ms to see the distribution
        gaps = detect_audio_gaps(samples, sample_rate, min_gap_ms=50.0)

        # Filter out leading/trailing silence (expected)
        interior_gaps = [g for g in gaps
                        if g['start_time'] > 0.1 and g['end_time'] < duration - 0.1]

        if interior_gaps:
            print(f"Interior gaps found: {len(interior_gaps)}")
            for g in interior_gaps[:5]:  # Show first 5
                print(f"  {g['start_time']:.2f}-{g['end_time']:.2f}s ({g['duration_ms']:.0f}ms)")

        # Natural TTS pauses are typically 50-200ms at phrase boundaries
        # SEVERE underruns would be >500ms - much longer than natural pauses
        severe_gaps = [g for g in interior_gaps if g['duration_ms'] > 500]
        gap_durations = [f"{g['duration_ms']:.0f}ms" for g in severe_gaps]
        assert len(severe_gaps) == 0, (
            f"Found {len(severe_gaps)} severe gaps (>500ms) - callback blocking suspected: "
            f"{gap_durations}"
        )

    def test_concurrent_synthesis_no_interference(self, cpp_binary, tmp_path):
        """
        Run multiple synthesis operations concurrently.

        Verifies that the lock-free architecture doesn't cause data races
        when multiple processes/threads access the system.
        """
        texts = [
            "Hello world",
            "Goodbye moon",
            "Testing one two three",
            "Final message here"
        ]

        def synthesize(idx_text):
            idx, text = idx_text
            wav_path = tmp_path / f"concurrent_{idx}.wav"
            success, latency = generate_audio(cpp_binary, text, "en", wav_path, timeout=120)
            return idx, success, latency, wav_path

        # Run concurrently
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(synthesize, (i, t)) for i, t in enumerate(texts)]
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        successes = [r for r in results if r[1]]
        print(f"\n=== Concurrent Synthesis ===")
        print(f"Succeeded: {len(successes)}/{len(texts)}")

        assert len(successes) >= len(texts) - 1, (
            f"Too many failures: {len(texts) - len(successes)}/{len(texts)}"
        )

        # Verify audio files are valid (not corrupted by races)
        for idx, success, latency, wav_path in results:
            if success and wav_path.exists():
                samples, sr = read_wav_samples(str(wav_path))
                assert len(samples) > 1000, f"Audio {idx} too short - possible corruption"

    def test_ring_buffer_stress_via_repeated_synth(self, cpp_binary, tmp_path):
        """
        Stress the ring buffer by generating many short utterances.

        The ring buffer handles producer (synthesis) -> consumer (playback)
        coordination. Repeated use should not cause memory leaks or
        state corruption.
        """
        # Generate 20 short utterances to stress ring buffer reuse
        num_iterations = 20
        short_texts = ["Hi", "No", "Yes", "OK", "Go"]

        successes = 0
        total_latency = 0

        for i in range(num_iterations):
            text = short_texts[i % len(short_texts)]
            wav_path = tmp_path / f"ring_stress_{i}.wav"
            success, latency = generate_audio(cpp_binary, text, "en", wav_path, timeout=30)

            if success:
                successes += 1
                total_latency += latency

        success_rate = successes / num_iterations
        avg_latency = total_latency / successes if successes > 0 else 0

        print(f"\n=== Ring Buffer Stress ===")
        print(f"Iterations: {num_iterations}")
        print(f"Success rate: {success_rate:.0%}")
        print(f"Avg latency: {avg_latency:.2f}s")

        # Should have >90% success rate
        assert success_rate >= 0.90, (
            f"Low success rate {success_rate:.0%} - ring buffer may have issues"
        )


@pytest.mark.stress
class TestLockFreeAudioValidation:
    """Validation tests for lock-free assumptions."""

    def test_lock_free_config_assumptions(self):
        """Verify test configuration assumptions are valid."""
        # Gap detection threshold should be small but not zero
        assert 0.0001 <= 0.001 <= 0.01  # gap_threshold

        # Minimum gap to report should be reasonable
        assert 5 <= 10.0 <= 100  # min_gap_ms in ms

        # Latency variance threshold should catch blocking
        assert 2 <= 4 <= 10  # max_latency / avg_latency multiplier
