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
Prosody Audio Testing - Phase A Validation

Tests that prosody markers produce audible differences in synthesized audio.
This validates the Phase A implementation:
1. Break insertion works correctly
2. Breaks produce silence at correct positions
3. Audio timing/duration changes appropriately

Uses mlx_audio Kokoro with Python prosody parser.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.prosody.test_prosody_parser import (
    PROSODY_ADJUSTMENTS,
    ParsedProsody,
    parse_prosody_markers,
)

# =============================================================================
# Audio Post-Processing (Python port of C++ prosody_adjust)
# =============================================================================

def insert_breaks_audio(
    audio: np.ndarray,
    sample_rate: int,
    parsed: ParsedProsody,
) -> np.ndarray:
    """
    Insert silence breaks into audio.

    Python port of kokoro.cpp insert_breaks_audio function.

    Args:
        audio: Audio samples (1D numpy array)
        sample_rate: Sample rate (e.g., 24000)
        parsed: Parsed prosody with break information

    Returns:
        New audio array with silence inserted
    """
    if not parsed.breaks or len(parsed.clean_text) == 0:
        return audio

    text_length = len(parsed.clean_text)
    samples_per_char = len(audio) / text_length

    # Build insertions list: (sample_pos, silence_samples)
    insertions = []
    for brk in parsed.breaks:
        sample_pos = int(brk.after_char * samples_per_char)
        sample_pos = min(sample_pos, len(audio))
        silence_samples = (brk.duration_ms * sample_rate) // 1000
        if silence_samples > 0:
            insertions.append((sample_pos, silence_samples))

    # Sort by position
    insertions.sort(key=lambda x: x[0])

    # Build result
    result = []
    audio_pos = 0

    fade_samples = min(int(5 * sample_rate / 1000), len(audio))  # 5ms fade

    for insert_pos, silence_samples in insertions:
        # Copy audio up to insertion point
        if insert_pos > audio_pos:
            chunk = audio[audio_pos:insert_pos]
            result.extend(chunk)
            audio_pos = insert_pos

        # Apply fade-out (last 5ms before silence)
        if len(result) >= fade_samples:
            for i in range(fade_samples):
                idx = len(result) - fade_samples + i
                fade = (fade_samples - i) / fade_samples
                result[idx] = result[idx] * fade

        # Insert silence
        result.extend([0.0] * silence_samples)

    # Copy remaining audio
    if audio_pos < len(audio):
        result.extend(audio[audio_pos:])

    return np.array(result, dtype=np.float32)


def apply_volume_envelope(
    audio: np.ndarray,
    sample_rate: int,
    parsed: ParsedProsody,
) -> np.ndarray:
    """
    Apply volume multipliers based on prosody annotations.

    Note: This is a crude approximation that operates on the final audio.
    True prosody adjustment requires modifying duration/F0 at the model level.
    """
    if not parsed.annotations or len(parsed.clean_text) == 0:
        return audio.copy()

    result = audio.copy()
    text_length = len(parsed.clean_text)
    samples_per_char = len(audio) / text_length

    for ann in parsed.annotations:
        if ann.type not in PROSODY_ADJUSTMENTS:
            continue

        _, _, _, volume_mult = PROSODY_ADJUSTMENTS[ann.type]

        if abs(volume_mult - 1.0) < 0.01:
            continue

        start_sample = int(ann.char_start * samples_per_char)
        end_sample = int(ann.char_end * samples_per_char)

        start_sample = max(0, min(start_sample, len(result)))
        end_sample = max(0, min(end_sample, len(result)))

        if start_sample < end_sample:
            result[start_sample:end_sample] *= volume_mult

    return result


# =============================================================================
# TTS Synthesis Helper
# =============================================================================

@dataclass
class AudioResult:
    """Result of audio synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    text: str
    clean_text: str


class ProsodyTTS:
    """Kokoro TTS with prosody marker support."""

    def __init__(self, voice: str = "af_bella"):
        self.voice = voice
        self.sample_rate = 24000
        self.model = None

    def load_model(self):
        """Load Kokoro TTS model."""
        if self.model is not None:
            return

        print("Loading Kokoro TTS model...")
        from mlx_audio.tts.utils import load_model
        self.model = load_model("prince-canuma/Kokoro-82M")

        # Warmup
        print("Warming up...")
        for _ in range(2):
            for _result in self.model.generate(text="Hello.", voice=self.voice):
                pass
        print("Model ready.")

    def synthesize(self, text: str, apply_prosody: bool = True) -> AudioResult:
        """
        Synthesize text with optional prosody processing.

        Args:
            text: Input text (may contain prosody markers)
            apply_prosody: If True, parse markers and apply post-processing

        Returns:
            AudioResult with synthesized audio
        """
        self.load_model()

        # Parse prosody markers
        parsed = parse_prosody_markers(text)
        clean_text = parsed.clean_text if apply_prosody else text

        # Synthesize clean text
        audio_chunks = [
            result.audio
            for result in self.model.generate(
                text=clean_text,
                voice=self.voice,
                speed=1.0,
                verbose=False,
            )
        ]

        audio = np.concatenate(audio_chunks, axis=-1).flatten()

        # Apply prosody post-processing
        if apply_prosody:
            # Insert breaks
            if parsed.breaks:
                audio = insert_breaks_audio(audio, self.sample_rate, parsed)

            # Apply volume envelope (crude)
            if parsed.annotations:
                audio = apply_volume_envelope(audio, self.sample_rate, parsed)

        return AudioResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=len(audio) / self.sample_rate,
            text=text,
            clean_text=parsed.clean_text,
        )


# =============================================================================
# Test Cases
# =============================================================================

@dataclass
class ProsodyTestCase:
    """A prosody audio test case."""
    id: int
    description: str
    text_with_prosody: str
    expected_effect: str


TEST_CASES = [
    # Break insertion tests
    ProsodyTestCase(
        id=1,
        description="Short break (300ms)",
        text_with_prosody="Hello<break time='300ms'/>world",
        expected_effect="300ms silence between 'Hello' and 'world'",
    ),
    ProsodyTestCase(
        id=2,
        description="Medium break (500ms)",
        text_with_prosody="I understand.<break time='500ms'/>Let me help.",
        expected_effect="500ms pause between sentences",
    ),
    ProsodyTestCase(
        id=3,
        description="Long break (1s)",
        text_with_prosody="Wait<break time='1s'/>okay now continue.",
        expected_effect="1 second pause after 'Wait'",
    ),
    ProsodyTestCase(
        id=4,
        description="Multiple breaks",
        text_with_prosody="One<break time='200ms'/>two<break time='200ms'/>three",
        expected_effect="200ms pauses between each word",
    ),

    # Emphasis tests (volume only at Phase A)
    ProsodyTestCase(
        id=5,
        description="Simple emphasis",
        text_with_prosody="I <em>really</em> need this",
        expected_effect="'really' should be louder",
    ),
    ProsodyTestCase(
        id=6,
        description="Strong emphasis",
        text_with_prosody="This is <strong>critical</strong> information",
        expected_effect="'critical' should be noticeably louder",
    ),

    # Emotion tests (volume approximation)
    ProsodyTestCase(
        id=7,
        description="Excited emotion",
        text_with_prosody="<emotion type='excited'>This is amazing!</emotion>",
        expected_effect="Higher volume for excitement",
    ),
    ProsodyTestCase(
        id=8,
        description="Calm emotion",
        text_with_prosody="<emotion type='calm'>Everything is fine.</emotion>",
        expected_effect="Lower volume for calmness",
    ),

    # Combined markers
    ProsodyTestCase(
        id=9,
        description="Combined emphasis and break",
        text_with_prosody="I <em>really</em> understand.<break time='300ms'/>Let me help.",
        expected_effect="Emphasis on 'really' plus 300ms pause",
    ),

    # Whisper/Loud
    ProsodyTestCase(
        id=10,
        description="Whisper marker",
        text_with_prosody="<whisper>This is a secret.</whisper>",
        expected_effect="Much lower volume (0.3x)",
    ),
    ProsodyTestCase(
        id=11,
        description="Loud marker",
        text_with_prosody="<loud>Pay attention!</loud>",
        expected_effect="Higher volume (1.5x)",
    ),
]


# =============================================================================
# Test Runner
# =============================================================================

def analyze_break_insertion(
    audio_with: np.ndarray,
    audio_without: np.ndarray,
    expected_break_ms: int,
    sample_rate: int,
) -> dict:
    """
    Analyze if break insertion produced expected silence.

    Returns dict with:
    - duration_diff_ms: difference in duration
    - expected_diff_ms: expected difference
    - break_detected: True if silence was detected
    - silence_quality: Assessment of the silence insertion
    """
    duration_with = len(audio_with) / sample_rate * 1000
    duration_without = len(audio_without) / sample_rate * 1000
    duration_diff = duration_with - duration_without

    # Check if duration increased approximately by expected break
    tolerance = 50  # 50ms tolerance
    break_detected = abs(duration_diff - expected_break_ms) < tolerance

    # Simple RMS analysis to detect silence
    rms_with = np.sqrt(np.mean(audio_with**2))
    rms_without = np.sqrt(np.mean(audio_without**2))

    return {
        "duration_with_ms": duration_with,
        "duration_without_ms": duration_without,
        "duration_diff_ms": duration_diff,
        "expected_diff_ms": expected_break_ms,
        "break_detected": break_detected,
        "rms_with": rms_with,
        "rms_without": rms_without,
    }


def analyze_volume_change(
    audio_with: np.ndarray,
    audio_without: np.ndarray,
    expected_mult: float,
) -> dict:
    """
    Analyze if volume envelope produced expected change.
    """
    rms_with = np.sqrt(np.mean(audio_with**2))
    rms_without = np.sqrt(np.mean(audio_without**2))

    actual_ratio = rms_with / rms_without if rms_without > 0 else 0

    # Since we apply multiplier to only part of audio, ratio will be partial
    # Just check if it's moving in the right direction
    if expected_mult > 1.0:
        effect_detected = actual_ratio > 1.0
    elif expected_mult < 1.0:
        effect_detected = actual_ratio < 1.0
    else:
        effect_detected = True

    return {
        "rms_with": rms_with,
        "rms_without": rms_without,
        "actual_ratio": actual_ratio,
        "expected_mult": expected_mult,
        "effect_detected": effect_detected,
    }


def run_prosody_audio_tests(
    save_audio: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """
    Run all prosody audio tests.

    Args:
        save_audio: If True, save audio files for manual review
        output_dir: Directory for audio files

    Returns:
        Test results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "prosody_audio_output"

    if save_audio:
        output_dir.mkdir(parents=True, exist_ok=True)

    tts = ProsodyTTS()
    results = []

    print("\n" + "="*60)
    print("PROSODY AUDIO TESTS - Phase A Validation")
    print("="*60 + "\n")

    for test in TEST_CASES:
        print(f"Test {test.id}: {test.description}")
        print(f"  Text: {test.text_with_prosody[:50]}...")

        start_time = time.time()

        # Synthesize with prosody
        result_with = tts.synthesize(test.text_with_prosody, apply_prosody=True)

        # Synthesize without prosody (clean text only)
        result_without = tts.synthesize(result_with.clean_text, apply_prosody=False)

        synthesis_time = time.time() - start_time

        # Analyze differences
        parsed = parse_prosody_markers(test.text_with_prosody)

        test_result = {
            "id": test.id,
            "description": test.description,
            "text_with_prosody": test.text_with_prosody,
            "clean_text": result_with.clean_text,
            "duration_with": result_with.duration_seconds,
            "duration_without": result_without.duration_seconds,
            "synthesis_time": synthesis_time,
            "has_breaks": len(parsed.breaks) > 0,
            "has_annotations": len(parsed.annotations) > 0,
        }

        # Analyze break insertion
        if parsed.breaks:
            total_break_ms = sum(b.duration_ms for b in parsed.breaks)
            break_analysis = analyze_break_insertion(
                result_with.audio,
                result_without.audio,
                total_break_ms,
                tts.sample_rate,
            )
            test_result["break_analysis"] = break_analysis

            status = "✓ PASS" if break_analysis["break_detected"] else "✗ FAIL"
            print(f"  Break insertion: {status}")
            print(f"    Expected: +{total_break_ms}ms, Actual: +{break_analysis['duration_diff_ms']:.1f}ms")

        # Analyze volume changes
        if parsed.annotations:
            # Get dominant annotation type
            ann = parsed.annotations[0]
            if ann.type in PROSODY_ADJUSTMENTS:
                _, _, _, expected_mult = PROSODY_ADJUSTMENTS[ann.type]
                vol_analysis = analyze_volume_change(
                    result_with.audio,
                    result_without.audio,
                    expected_mult,
                )
                test_result["volume_analysis"] = vol_analysis

                status = "✓ PASS" if vol_analysis["effect_detected"] else "✗ FAIL"
                print(f"  Volume change: {status}")
                print(f"    Expected mult: {expected_mult:.2f}, Actual ratio: {vol_analysis['actual_ratio']:.3f}")

        # Save audio if requested
        if save_audio:
            sf.write(
                output_dir / f"test_{test.id:02d}_with_prosody.wav",
                result_with.audio,
                tts.sample_rate,
            )
            sf.write(
                output_dir / f"test_{test.id:02d}_without_prosody.wav",
                result_without.audio,
                tts.sample_rate,
            )

        results.append(test_result)
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    break_tests = [r for r in results if r.get("has_breaks")]
    break_passed = sum(1 for r in break_tests if r.get("break_analysis", {}).get("break_detected"))

    vol_tests = [r for r in results if r.get("has_annotations")]
    vol_passed = sum(1 for r in vol_tests if r.get("volume_analysis", {}).get("effect_detected"))

    print(f"\nBreak insertion tests: {break_passed}/{len(break_tests)} passed")
    print(f"Volume change tests: {vol_passed}/{len(vol_tests)} passed")

    if save_audio:
        print(f"\nAudio files saved to: {output_dir}")

    return {
        "test_results": results,
        "break_tests_passed": break_passed,
        "break_tests_total": len(break_tests),
        "volume_tests_passed": vol_passed,
        "volume_tests_total": len(vol_tests),
    }


def measure_latency_overhead(num_runs: int = 10) -> dict:
    """
    Measure prosody processing latency overhead.

    Compares synthesis time with and without prosody parsing.
    Phase A requirement: < 1ms overhead.
    """
    tts = ProsodyTTS()

    test_text_plain = "Hello, this is a test sentence for measuring latency."
    test_text_prosody = "Hello, <em>this</em> is a <break time='100ms'/>test sentence for measuring latency."

    print("\n" + "="*60)
    print("LATENCY OVERHEAD MEASUREMENT")
    print("="*60 + "\n")

    # Warmup
    tts.synthesize(test_text_plain, apply_prosody=False)
    tts.synthesize(test_text_prosody, apply_prosody=True)

    times_plain = []
    times_prosody = []

    for _i in range(num_runs):
        # Plain text
        start = time.perf_counter()
        tts.synthesize(test_text_plain, apply_prosody=False)
        times_plain.append(time.perf_counter() - start)

        # With prosody
        start = time.perf_counter()
        tts.synthesize(test_text_prosody, apply_prosody=True)
        times_prosody.append(time.perf_counter() - start)

    avg_plain = np.mean(times_plain) * 1000
    avg_prosody = np.mean(times_prosody) * 1000
    overhead = avg_prosody - avg_plain

    print(f"Average synthesis time (plain): {avg_plain:.2f}ms")
    print(f"Average synthesis time (prosody): {avg_prosody:.2f}ms")
    print(f"Overhead: {overhead:.2f}ms")
    print("Requirement: < 1ms")
    print(f"Status: {'✓ PASS' if overhead < 1.0 else '✗ FAIL'}")

    return {
        "avg_plain_ms": avg_plain,
        "avg_prosody_ms": avg_prosody,
        "overhead_ms": overhead,
        "passed": overhead < 1.0,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prosody Audio Testing")
    parser.add_argument("--save-audio", action="store_true", help="Save audio files")
    parser.add_argument("--latency", action="store_true", help="Run latency measurement")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.all or not (args.latency):
        results = run_prosody_audio_tests(save_audio=args.save_audio)

    if args.latency or args.all:
        latency = measure_latency_overhead()
