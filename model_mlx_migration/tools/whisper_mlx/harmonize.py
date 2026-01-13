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
Harmonization Module for Gate 5 - Real-time Singing Harmony Generation.

This module implements real-time harmony generation when singing is detected:
1. Detect singing using SingingHead (99.6% accuracy on RAVDESS)
2. Track pitch using PitchHead
3. Generate harmony by pitch-shifting input audio

Musical intervals supported:
- Major third (+4 semitones)
- Perfect fifth (+7 semitones)
- Octave (+12 semitones)
- Minor third (+3 semitones)
- Perfect fourth (+5 semitones)

Architecture:
    Audio -> SingingHead (is singing?) -> PitchHead (track F0) -> Pitch Shift -> Harmony

References:
- RAVDESS Dataset: https://zenodo.org/record/1188976
- Librosa pitch shifting: https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.whisper_mlx.model import WhisperMLX

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


class HarmonyInterval(Enum):
    """Musical intervals for harmony generation."""
    UNISON = 0           # Same pitch (no shift)
    MINOR_SECOND = 1     # +1 semitone
    MAJOR_SECOND = 2     # +2 semitones
    MINOR_THIRD = 3      # +3 semitones (sad/dark harmony)
    MAJOR_THIRD = 4      # +4 semitones (bright/happy harmony)
    PERFECT_FOURTH = 5   # +5 semitones
    TRITONE = 6          # +6 semitones (dissonant)
    PERFECT_FIFTH = 7    # +7 semitones (power chord)
    MINOR_SIXTH = 8      # +8 semitones
    MAJOR_SIXTH = 9      # +9 semitones
    MINOR_SEVENTH = 10   # +10 semitones
    MAJOR_SEVENTH = 11   # +11 semitones
    OCTAVE = 12          # +12 semitones (doubling)

    # Negative intervals (below)
    OCTAVE_BELOW = -12   # -12 semitones
    FIFTH_BELOW = -5     # -5 semitones (power chord below)
    THIRD_BELOW = -4     # -4 semitones (major third below)


# Common chord harmonies
CHORD_MAJOR = [HarmonyInterval.UNISON, HarmonyInterval.MAJOR_THIRD, HarmonyInterval.PERFECT_FIFTH]
CHORD_MINOR = [HarmonyInterval.UNISON, HarmonyInterval.MINOR_THIRD, HarmonyInterval.PERFECT_FIFTH]
CHORD_POWER = [HarmonyInterval.UNISON, HarmonyInterval.PERFECT_FIFTH, HarmonyInterval.OCTAVE]
CHORD_OCTAVE = [HarmonyInterval.UNISON, HarmonyInterval.OCTAVE]

# Common two-part harmonies
HARMONY_THIRD_ABOVE = [HarmonyInterval.UNISON, HarmonyInterval.MAJOR_THIRD]
HARMONY_FIFTH_ABOVE = [HarmonyInterval.UNISON, HarmonyInterval.PERFECT_FIFTH]
HARMONY_OCTAVE_ABOVE = [HarmonyInterval.UNISON, HarmonyInterval.OCTAVE]
HARMONY_THIRD_BELOW = [HarmonyInterval.UNISON, HarmonyInterval.THIRD_BELOW]


@dataclass
class HarmonyConfig:
    """Configuration for harmony generation."""

    # Sample rate for audio processing
    sample_rate: int = 16000

    # Default intervals to generate (major third + perfect fifth)
    default_intervals: list[HarmonyInterval] = None

    # Mix ratio: 0.0 = only original, 1.0 = only harmony
    # Default 0.3 means 70% original + 30% harmony
    harmony_mix: float = 0.3

    # Singing detection threshold (probability)
    singing_threshold: float = 0.5

    # Minimum singing duration to trigger harmony (seconds)
    min_singing_duration: float = 0.3

    # Pitch shift quality: 'fast' or 'high'
    pitch_shift_quality: str = 'fast'

    def __post_init__(self):
        if self.default_intervals is None:
            self.default_intervals = [HarmonyInterval.MAJOR_THIRD, HarmonyInterval.PERFECT_FIFTH]


def semitones_to_ratio(semitones: int) -> float:
    """
    Convert semitones to frequency ratio.

    Equal temperament: ratio = 2^(semitones/12)

    Args:
        semitones: Number of semitones (positive = up, negative = down)

    Returns:
        Frequency ratio (e.g., 1.0595 for +1 semitone)
    """
    return 2.0 ** (semitones / 12.0)


def hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI note number (float for microtonal accuracy)."""
    if hz <= 0:
        return 0.0
    return 69 + 12 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name (e.g., C4, A#3)."""
    if midi <= 0:
        return ""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


def shift_pitch_hz(pitch_hz: float, interval: HarmonyInterval) -> float:
    """
    Calculate harmony pitch from original pitch and interval.

    Args:
        pitch_hz: Original pitch in Hz
        interval: Musical interval to shift by

    Returns:
        Harmony pitch in Hz
    """
    ratio = semitones_to_ratio(interval.value)
    return pitch_hz * ratio


def pitch_shift_audio(
    audio: np.ndarray,
    semitones: float,
    sample_rate: int = 16000,
    quality: str = 'fast',
) -> np.ndarray:
    """
    Shift the pitch of audio by a given number of semitones.

    Uses librosa's pitch_shift which applies time-stretching + resampling.
    This is a high-quality but not real-time approach.

    Args:
        audio: Audio waveform as float32 numpy array
        semitones: Number of semitones to shift (can be fractional)
        sample_rate: Audio sample rate
        quality: 'fast' (default) or 'high' quality

    Returns:
        Pitch-shifted audio as float32 numpy array
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for pitch shifting: pip install librosa") from None

    # n_steps is in semitones for librosa
    # bins_per_octave affects quality vs speed
    bins_per_octave = 12 if quality == 'fast' else 24

    shifted = librosa.effects.pitch_shift(
        audio,
        sr=sample_rate,
        n_steps=semitones,
        bins_per_octave=bins_per_octave,
    )

    return shifted.astype(np.float32)


def generate_harmony(
    audio: np.ndarray,
    intervals: list[HarmonyInterval],
    sample_rate: int = 16000,
    mix_ratios: list[float] | None = None,
    quality: str = 'fast',
) -> np.ndarray:
    """
    Generate harmony by pitch-shifting audio at multiple intervals.

    Args:
        audio: Original audio waveform
        intervals: List of intervals to generate (e.g., [MAJOR_THIRD, PERFECT_FIFTH])
        sample_rate: Audio sample rate
        mix_ratios: Mix ratio for each interval (default: equal mix)
        quality: Pitch shift quality

    Returns:
        Mixed audio with original + harmonies
    """
    if mix_ratios is None:
        # Default: equal mix of all voices
        mix_ratios = [1.0 / (len(intervals) + 1)] * len(intervals)

    # Start with original audio
    original_ratio = 1.0 - sum(mix_ratios)
    result = audio * original_ratio

    # Add each harmony voice
    for interval, ratio in zip(intervals, mix_ratios, strict=False):
        if interval.value != 0:  # Skip unison
            shifted = pitch_shift_audio(audio, interval.value, sample_rate, quality)
            result = result + shifted * ratio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val

    return result.astype(np.float32)


class HarmonyGenerator:
    """
    Real-time harmony generator integrating singing detection and pitch tracking.

    Pipeline:
        1. SingingHead detects if user is singing
        2. If singing, PitchHead tracks the melody
        3. Pitch-shift audio to create harmony voices
        4. Mix original + harmony

    Usage:
        generator = HarmonyGenerator(config)
        generator.load_models(whisper, checkpoint_path)

        # Process audio
        result = generator.process(audio_chunk)
        if result.is_singing:
            harmonized_audio = result.harmonized_audio
    """

    def __init__(self, config: HarmonyConfig | None = None):
        """Initialize harmony generator."""
        self.config = config or HarmonyConfig()
        self.whisper = None
        self.singing_head = None
        self.pitch_head = None
        self._model_loaded = False

    def load_models(
        self,
        whisper_model: "WhisperMLX",
        checkpoint_path: str,
    ):
        """
        Load singing detection and pitch tracking models.

        Args:
            whisper_model: Pre-loaded Whisper MLX model
            checkpoint_path: Path to multi-head checkpoint (.npz)
        """
        from .multi_head import create_multi_head

        self.whisper = whisper_model

        # Create multi-head architecture
        multi_head = create_multi_head("large-v3")

        # Load checkpoint weights
        weights = dict(np.load(checkpoint_path, allow_pickle=True))

        # Load singing head weights
        singing_weights = []
        for k, v in weights.items():
            if k.startswith("singing."):
                new_key = k[len("singing."):]
                singing_weights.append((new_key, mx.array(v)))

        if singing_weights:
            multi_head.singing_head.load_weights(singing_weights)

        # Load pitch head weights (if available)
        pitch_weights = []
        for k, v in weights.items():
            if k.startswith("pitch."):
                new_key = k[len("pitch."):]
                pitch_weights.append((new_key, mx.array(v)))

        if pitch_weights:
            multi_head.pitch_head.load_weights(pitch_weights)

        self.singing_head = multi_head.singing_head
        self.pitch_head = multi_head.pitch_head
        self._model_loaded = True

    def detect_singing(
        self,
        encoder_output: "mx.array",
    ) -> tuple[bool, float]:
        """
        Detect if audio contains singing.

        Args:
            encoder_output: Whisper encoder output (batch, T, d_model)

        Returns:
            Tuple of (is_singing, confidence)
        """
        if not self._model_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.singing_head.predict(encoder_output, self.config.singing_threshold)

    def track_pitch(
        self,
        encoder_output: "mx.array",
        frame_rate: float = 50.0,
    ) -> list[tuple[float, float, float]]:
        """
        Track pitch over time.

        Args:
            encoder_output: Whisper encoder output
            frame_rate: Frames per second (50 for Whisper)

        Returns:
            List of (time_seconds, pitch_hz, voicing_confidence)
        """
        if not self._model_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.pitch_head.predict_melody(encoder_output, frame_rate)

    def get_mean_pitch(self, melody: list[tuple[float, float, float]]) -> float:
        """
        Get mean pitch from melody (excluding unvoiced frames).

        Args:
            melody: Output from track_pitch()

        Returns:
            Mean pitch in Hz (0 if no voiced frames)
        """
        voiced = [hz for _, hz, conf in melody if hz > 0 and conf > 0.5]
        return float(np.mean(voiced)) if voiced else 0.0

    def process(
        self,
        audio: np.ndarray,
        intervals: list[HarmonyInterval] | None = None,
    ) -> "HarmonyResult":
        """
        Process audio chunk: detect singing, track pitch, generate harmony.

        Args:
            audio: Audio waveform as float32 numpy array
            intervals: Intervals to harmonize (default from config)

        Returns:
            HarmonyResult with detection, pitch, and harmonized audio
        """
        from .audio import log_mel_spectrogram

        if not self._model_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        intervals = intervals or self.config.default_intervals

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=128)

        # Pad to 3000 frames if needed
        target_frames = 3000
        if mel.shape[0] < target_frames:
            mel = mx.pad(mel, [(0, target_frames - mel.shape[0]), (0, 0)])
        else:
            mel = mel[:target_frames, :]

        mel = mel[None, ...]  # Add batch dimension

        # Get encoder output
        encoder_output = self.whisper.embed_audio(mel)

        # Detect singing
        is_singing, singing_confidence = self.detect_singing(encoder_output)

        # Track pitch
        melody = self.track_pitch(encoder_output)
        mean_pitch = self.get_mean_pitch(melody)

        # Generate harmony if singing detected
        harmonized_audio = None
        if is_singing:
            harmonized_audio = generate_harmony(
                audio,
                intervals,
                self.config.sample_rate,
                quality=self.config.pitch_shift_quality,
            )

        return HarmonyResult(
            is_singing=is_singing,
            singing_confidence=singing_confidence,
            mean_pitch_hz=mean_pitch,
            pitch_note=midi_to_note_name(int(hz_to_midi(mean_pitch))) if mean_pitch > 0 else "",
            melody=melody,
            intervals=intervals,
            original_audio=audio,
            harmonized_audio=harmonized_audio,
        )


@dataclass
class HarmonyResult:
    """Result from harmony generation."""

    # Singing detection
    is_singing: bool
    singing_confidence: float

    # Pitch tracking
    mean_pitch_hz: float
    pitch_note: str
    melody: list[tuple[float, float, float]]  # (time, hz, confidence)

    # Harmony output
    intervals: list[HarmonyInterval]
    original_audio: np.ndarray
    harmonized_audio: np.ndarray | None  # None if not singing

    def get_harmony_notes(self) -> list[str]:
        """Get the note names for each harmony voice."""
        if self.mean_pitch_hz <= 0:
            return []

        notes = []
        for interval in self.intervals:
            harmony_hz = shift_pitch_hz(self.mean_pitch_hz, interval)
            midi = int(hz_to_midi(harmony_hz))
            notes.append(midi_to_note_name(midi))

        return notes


def harmonize_file(
    audio_path: str,
    checkpoint_path: str,
    output_path: str | None = None,
    intervals: list[HarmonyInterval] | None = None,
    whisper_model: str = "mlx-community/whisper-large-v3-mlx",
) -> HarmonyResult:
    """
    Convenience function to harmonize an audio file.

    Args:
        audio_path: Path to input audio file
        checkpoint_path: Path to multi-head checkpoint
        output_path: Path to save harmonized audio (optional)
        intervals: Harmony intervals (default: major third + perfect fifth)
        whisper_model: Whisper model to use

    Returns:
        HarmonyResult with all outputs
    """
    from .audio import load_audio
    from .model import WhisperMLX

    # Load audio
    audio = load_audio(audio_path, sample_rate=16000)

    # Load Whisper
    whisper = WhisperMLX.from_pretrained(whisper_model)

    # Create generator and load models
    config = HarmonyConfig()
    generator = HarmonyGenerator(config)
    generator.load_models(whisper, checkpoint_path)

    # Process
    result = generator.process(audio, intervals)

    # Save output if requested
    if output_path and result.harmonized_audio is not None:
        try:
            import soundfile as sf
            sf.write(output_path, result.harmonized_audio, 16000)
            print(f"Saved harmonized audio to: {output_path}")
        except ImportError:
            print("soundfile not available, cannot save output")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python harmonize.py <audio_path> <checkpoint_path> [output_path]")
        print("\nExample:")
        print("  python harmonize.py data/prosody/ravdess/Actor_01/03-02-01-01-01-01-01.wav \\")
        print("                      checkpoints/multi_head_ravdess/best.npz \\")
        print("                      output_harmony.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    print("=" * 60)
    print("Gate 5: Harmonization")
    print("=" * 60)

    result = harmonize_file(audio_path, checkpoint_path, output_path)

    print(f"\nSinging detected: {result.is_singing} (confidence: {result.singing_confidence:.3f})")
    print(f"Mean pitch: {result.mean_pitch_hz:.1f} Hz ({result.pitch_note})")

    if result.is_singing:
        print(f"\nHarmony intervals: {[i.name for i in result.intervals]}")
        print(f"Harmony notes: {result.get_harmony_notes()}")

        if output_path:
            print(f"\nHarmonized audio saved to: {output_path}")
    else:
        print("\nNo singing detected - no harmony generated")
