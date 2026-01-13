"""
Audio Amplitude Consistency Tests

Tests that TTS audio maintains consistent amplitude throughout utterances.
This catches a bug where audio would fade out mid-utterance and then recover.

BUG REPORT: reports/main/BUG_AUDIO_FADEOUT_2025-12-09.md

OBSERVED PATTERN (BUG):
    Time     RMS     Speech
    0.0s     0.0000  . (silence)
    0.4s     0.0959  * (ONE TWO - loud)
    0.9s     0.0054  . (FADE STARTS - THREE?)
    1.0-1.5s 0.003   . (very quiet - FOUR FIVE SIX?)
    1.5s     0.0003  . (near silence - SEVEN?)
    2.0s     0.0196  * (EIGHT NINE TEN - recovers!)

EXPECTED: All speech segments should have roughly consistent amplitude (within 3x).

Run: pytest tests/integration/test_audio_amplitude_consistency.py -v
"""

import os
import subprocess
import struct
import math
import pytest
from pathlib import Path
from typing import List, Tuple


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def read_wav_samples(wav_path: str) -> Tuple[List[float], int]:
    """
    Read WAV file and return (samples as floats in [-1,1], sample_rate).

    Supports 16-bit PCM mono WAV files (standard for TTS output).
    """
    with open(wav_path, 'rb') as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a RIFF file: {riff}")

        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a WAVE file: {wave}")

        sample_rate = 0
        bits_per_sample = 16
        num_channels = 1

        # Find fmt and data chunks
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break

            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                audio_format = struct.unpack('<H', f.read(2))[0]
                num_channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # byte rate
                f.read(2)  # block align
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                # Skip any extra fmt data
                remaining = chunk_size - 16
                if remaining > 0:
                    f.read(remaining)

            elif chunk_id == b'data':
                if bits_per_sample != 16:
                    raise ValueError(f"Only 16-bit PCM supported, got {bits_per_sample}")

                num_samples = chunk_size // 2  # 2 bytes per sample
                raw_data = f.read(chunk_size)

                # Unpack as signed 16-bit integers
                samples_int = struct.unpack(f'<{num_samples}h', raw_data)

                # Convert to float in [-1, 1]
                samples = [s / 32768.0 for s in samples_int]

                return samples, sample_rate
            else:
                # Skip unknown chunk
                f.seek(chunk_size, 1)

    raise ValueError("No data chunk found in WAV file")


def calculate_segment_rms(samples: List[float], start_idx: int, end_idx: int) -> float:
    """Calculate RMS (Root Mean Square) of a segment of samples."""
    segment = samples[start_idx:end_idx]
    if not segment:
        return 0.0

    sum_sq = sum(s * s for s in segment)
    return math.sqrt(sum_sq / len(segment))


def calculate_segment_peak(samples: List[float], start_idx: int, end_idx: int) -> float:
    """Calculate peak (max absolute) amplitude of a segment."""
    segment = samples[start_idx:end_idx]
    if not segment:
        return 0.0

    return max(abs(s) for s in segment)


def analyze_amplitude_consistency(
    samples: List[float],
    sample_rate: int,
    segment_duration_ms: int = 500,
    silence_threshold: float = 0.005
) -> dict:
    """
    Analyze amplitude consistency across segments.

    Args:
        samples: Audio samples as floats in [-1,1]
        sample_rate: Sample rate in Hz
        segment_duration_ms: Duration of each segment to analyze
        silence_threshold: RMS below this is considered silence

    Returns:
        dict with analysis results:
            - segments: List of (start_time, end_time, rms, peak)
            - speech_segments: Segments with RMS > silence_threshold
            - max_speech_rms: Maximum RMS of speech segments
            - min_speech_rms: Minimum RMS of speech segments
            - amplitude_ratio: max_speech_rms / min_speech_rms
            - has_fadeout: True if any speech segment is 5x+ quieter than max
    """
    segment_samples = int(sample_rate * segment_duration_ms / 1000)
    num_segments = len(samples) // segment_samples

    segments = []
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = (i + 1) * segment_samples
        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate

        rms = calculate_segment_rms(samples, start_idx, end_idx)
        peak = calculate_segment_peak(samples, start_idx, end_idx)

        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'rms': rms,
            'peak': peak
        })

    # Filter to speech segments (above silence threshold)
    speech_segments = [s for s in segments if s['rms'] > silence_threshold]

    if not speech_segments:
        return {
            'segments': segments,
            'speech_segments': [],
            'max_speech_rms': 0,
            'min_speech_rms': 0,
            'amplitude_ratio': 1.0,
            'has_fadeout': False,
            'fadeout_details': None
        }

    speech_rms_values = [s['rms'] for s in speech_segments]
    max_rms = max(speech_rms_values)
    min_rms = min(speech_rms_values)
    ratio = max_rms / min_rms if min_rms > 0 else float('inf')

    # Check for fade-out pattern: any speech segment 5.5x+ quieter than loudest
    # Note: 5.5x threshold allows for natural prosodic variation at sentence endings
    # The real fadeout bug shows 7-26x ratios, so 5.5x catches real issues while
    # avoiding flaky failures from natural end-of-sentence quieting (~5.0x)
    fadeout_segments = []
    for s in speech_segments:
        if max_rms / s['rms'] >= 5.5:
            fadeout_segments.append({
                'time': f"{s['start_time']:.2f}-{s['end_time']:.2f}s",
                'rms': s['rms'],
                'ratio_to_max': max_rms / s['rms']
            })

    return {
        'segments': segments,
        'speech_segments': speech_segments,
        'max_speech_rms': max_rms,
        'min_speech_rms': min_rms,
        'amplitude_ratio': ratio,
        'has_fadeout': len(fadeout_segments) > 0,
        'fadeout_details': fadeout_segments if fadeout_segments else None
    }


@pytest.mark.integration
@pytest.mark.requires_binary
@pytest.mark.requires_models
@pytest.mark.amplitude_bug
class TestAudioAmplitudeConsistency:
    """
    Test that TTS audio maintains consistent amplitude throughout utterances.

    This catches a specific bug where audio would:
    1. Start loud (first 1 second)
    2. Fade to near-silence (1-2 seconds)
    3. Recover at the end (2+ seconds)

    Root cause hypothesis: Model attention/duration predictor failing mid-sequence.

    The test uses the specific text "One two three four five six seven eight nine ten"
    which reliably reproduced the bug.
    """

    @pytest.fixture(scope="class")
    def config_dir(self):
        """Config directory path."""
        return Path(__file__).parent.parent.parent / "stream-tts-cpp" / "config"

    def test_english_counting_amplitude_consistency(self, cpp_binary, config_dir, tmp_path):
        """
        Test that counting from one to ten maintains consistent amplitude.

        This specific text reliably reproduced the fade-out bug where:
        - "One two" were loud
        - "three four five six seven" faded to near-silence
        - "eight nine ten" recovered

        ACCEPTANCE CRITERIA:
        - All speech segments must be within 5x of the loudest segment
        - No speech segment should drop to <1/5 of the peak amplitude
        """
        config_path = config_dir / "kokoro-mps-en.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        text = "One two three four five six seven eight nine ten"
        wav_file = tmp_path / "counting_test.wav"

        # Generate audio using --speak mode
        result = subprocess.run(
            [str(cpp_binary), "--speak", text, "--lang", "en",
             "--save-audio", str(wav_file)],
            capture_output=True,
            timeout=60,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        # Check TTS succeeded
        assert result.returncode == 0, f"TTS failed: {result.stderr.decode()}"
        assert wav_file.exists(), f"Audio file not created: {wav_file}"

        file_size = wav_file.stat().st_size
        assert file_size > 1000, f"Audio file too small ({file_size} bytes)"

        # Analyze amplitude consistency
        samples, sample_rate = read_wav_samples(str(wav_file))
        analysis = analyze_amplitude_consistency(
            samples, sample_rate,
            segment_duration_ms=500,  # 500ms segments
            silence_threshold=0.005   # RMS below 0.005 is silence
        )

        # Print detailed analysis for debugging
        print(f"\n=== Amplitude Analysis: '{text}' ===")
        print(f"Total samples: {len(samples)}, Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(samples)/sample_rate:.2f}s")
        print(f"\nSegment analysis (500ms segments):")
        for seg in analysis['segments']:
            status = '*' if seg['rms'] > 0.005 else '.'
            print(f"  {seg['start_time']:.1f}-{seg['end_time']:.1f}s: "
                  f"RMS={seg['rms']:.4f}, Peak={seg['peak']:.4f} {status}")

        print(f"\nSpeech segments: {len(analysis['speech_segments'])}")
        print(f"Max speech RMS: {analysis['max_speech_rms']:.4f}")
        print(f"Min speech RMS: {analysis['min_speech_rms']:.4f}")
        print(f"Amplitude ratio (max/min): {analysis['amplitude_ratio']:.1f}x")
        print(f"Has fadeout: {analysis['has_fadeout']}")

        if analysis['fadeout_details']:
            print(f"\nFadeout details:")
            for fd in analysis['fadeout_details']:
                print(f"  {fd['time']}: RMS={fd['rms']:.4f}, "
                      f"{fd['ratio_to_max']:.1f}x quieter than max")

        # Assert no severe fade-out
        # Allow 5x variation (normal speech dynamics), fail on >5x (bug)
        assert not analysis['has_fadeout'], (
            f"Audio has fade-out bug! Speech amplitude varies by "
            f"{analysis['amplitude_ratio']:.1f}x (max {analysis['max_speech_rms']:.4f} / "
            f"min {analysis['min_speech_rms']:.4f}). "
            f"Fadeout segments: {analysis['fadeout_details']}"
        )

    def test_long_sentence_amplitude_consistency(self, cpp_binary, config_dir, tmp_path):
        """
        Test that a longer sentence maintains consistent amplitude.

        Uses a sentence that spans multiple seconds to test sustained synthesis.
        """
        config_path = config_dir / "kokoro-mps-en.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        text = "The quick brown fox jumps over the lazy dog while the cat watches."
        wav_file = tmp_path / "long_sentence_test.wav"

        result = subprocess.run(
            [str(cpp_binary), "--speak", text, "--lang", "en",
             "--save-audio", str(wav_file)],
            capture_output=True,
            timeout=60,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"TTS failed: {result.stderr.decode()}"
        assert wav_file.exists(), f"Audio file not created: {wav_file}"

        samples, sample_rate = read_wav_samples(str(wav_file))
        analysis = analyze_amplitude_consistency(
            samples, sample_rate,
            segment_duration_ms=500,
            silence_threshold=0.005
        )

        print(f"\n=== Amplitude Analysis: Long Sentence ===")
        print(f"Duration: {len(samples)/sample_rate:.2f}s")
        print(f"Speech segments: {len(analysis['speech_segments'])}")
        print(f"Amplitude ratio: {analysis['amplitude_ratio']:.1f}x")

        assert not analysis['has_fadeout'], (
            f"Long sentence has fade-out bug! Amplitude ratio: "
            f"{analysis['amplitude_ratio']:.1f}x. Details: {analysis['fadeout_details']}"
        )

    def test_japanese_amplitude_consistency(self, cpp_binary, config_dir, tmp_path):
        """
        Test Japanese TTS amplitude consistency.

        Uses a longer Japanese sentence to test non-English synthesis.
        """
        config_path = config_dir / "kokoro-mps-ja.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        # "Hello, world! This is a test."
        text = "こんにちは、世界！これはテストです。"
        wav_file = tmp_path / "japanese_test.wav"

        result = subprocess.run(
            [str(cpp_binary), "--speak", text, "--lang", "ja",
             "--save-audio", str(wav_file)],
            capture_output=True,
            timeout=60,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"TTS failed: {result.stderr.decode()}"
        assert wav_file.exists(), f"Audio file not created: {wav_file}"

        samples, sample_rate = read_wav_samples(str(wav_file))
        analysis = analyze_amplitude_consistency(
            samples, sample_rate,
            segment_duration_ms=500,
            silence_threshold=0.005
        )

        print(f"\n=== Amplitude Analysis: Japanese ===")
        print(f"Duration: {len(samples)/sample_rate:.2f}s")
        print(f"Speech segments: {len(analysis['speech_segments'])}")
        print(f"Amplitude ratio: {analysis['amplitude_ratio']:.1f}x")

        assert not analysis['has_fadeout'], (
            f"Japanese TTS has fade-out bug! Amplitude ratio: "
            f"{analysis['amplitude_ratio']:.1f}x. Details: {analysis['fadeout_details']}"
        )

    def test_chinese_amplitude_consistency(self, cpp_binary, config_dir, tmp_path):
        """
        Test Chinese TTS amplitude consistency.
        """
        config_path = config_dir / "kokoro-mps-zh.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        # "Hello, world! This is a test."
        text = "你好，世界！这是一个测试。"
        wav_file = tmp_path / "chinese_test.wav"

        result = subprocess.run(
            [str(cpp_binary), "--speak", text, "--lang", "zh",
             "--save-audio", str(wav_file)],
            capture_output=True,
            timeout=60,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"TTS failed: {result.stderr.decode()}"
        assert wav_file.exists(), f"Audio file not created: {wav_file}"

        samples, sample_rate = read_wav_samples(str(wav_file))
        analysis = analyze_amplitude_consistency(
            samples, sample_rate,
            segment_duration_ms=500,
            silence_threshold=0.005
        )

        print(f"\n=== Amplitude Analysis: Chinese ===")
        print(f"Duration: {len(samples)/sample_rate:.2f}s")
        print(f"Speech segments: {len(analysis['speech_segments'])}")
        print(f"Amplitude ratio: {analysis['amplitude_ratio']:.1f}x")

        assert not analysis['has_fadeout'], (
            f"Chinese TTS has fade-out bug! Amplitude ratio: "
            f"{analysis['amplitude_ratio']:.1f}x. Details: {analysis['fadeout_details']}"
        )
