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
Silero VAD Integration for WhisperMLX
=====================================

Bundled Voice Activity Detection using Silero VAD.
VAD is ALWAYS ON by default - this is NOT optional.

This module provides speech segment extraction that:
1. Filters out silence from audio before transcription
2. Speeds up transcription 2-4x on audio with silence
3. Has near-zero quality cost (VAD is very accurate)
4. Has negligible overhead (~1-2ms for 30s audio)

The only configurable parameter is `aggressiveness` (0-3):
- 0: Most conservative (keeps more audio, higher quality)
- 1: Balanced conservative
- 2: Balanced (default, recommended)
- 3: Most aggressive (filters more, faster, slight risk of cutting)

Usage:
    from tools.whisper_mlx.silero_vad import SileroVADProcessor

    # Create processor (loads model on first use)
    vad = SileroVADProcessor(aggressiveness=2)

    # Get speech segments from audio
    speech_segments = vad.get_speech_segments(audio)

    # Or extract only speech audio
    speech_audio = vad.extract_speech(audio)
"""

from dataclasses import dataclass

import numpy as np

# Sample rate expected by Silero VAD
SILERO_SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float      # Start time in seconds
    end: float        # End time in seconds
    start_sample: int # Start sample index
    end_sample: int   # End sample index

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start


@dataclass
class VADResult:
    """Result of VAD processing."""
    segments: list[SpeechSegment]
    speech_ratio: float      # Fraction of audio that is speech
    total_duration: float    # Total audio duration in seconds
    speech_duration: float   # Total speech duration in seconds

    @property
    def silence_ratio(self) -> float:
        """Fraction of audio that is silence."""
        return 1.0 - self.speech_ratio

    @property
    def is_mostly_silent(self) -> bool:
        """True if less than 5% of audio is speech."""
        return self.speech_ratio < 0.05


# Aggressiveness to Silero threshold mapping
# Higher aggressiveness = higher threshold = more filtering
AGGRESSIVENESS_THRESHOLDS = {
    0: 0.25,  # Conservative: keep more audio
    1: 0.35,  # Balanced conservative
    2: 0.50,  # Balanced (default)
    3: 0.65,  # Aggressive: filter more
}

# Aggressiveness to min silence duration mapping
# Higher aggressiveness = shorter silence duration = more splitting
AGGRESSIVENESS_SILENCE_MS = {
    0: 700,   # Conservative: longer pauses kept together
    1: 500,   # Balanced conservative
    2: 300,   # Balanced (default)
    3: 100,   # Aggressive: split on short pauses
}


class SileroVADProcessor:
    """
    Silero VAD processor for WhisperMLX.

    Loads Silero VAD model (lazy loading on first use) and provides
    speech segment extraction for audio preprocessing.

    This class is designed to be used as a singleton - create once
    and reuse for all transcriptions.
    """

    # Class-level model cache (singleton pattern)
    _model = None
    _utils = None
    _load_error = None

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = SILERO_SAMPLE_RATE,
        min_speech_duration_ms: int = 250,
    ):
        """
        Initialize VAD processor.

        Args:
            aggressiveness: VAD aggressiveness (0-3). Higher = more filtering.
                0: Most conservative (keeps more audio)
                1: Balanced conservative
                2: Balanced (default, recommended)
                3: Most aggressive (filters more)
            sample_rate: Expected sample rate (should be 16000 for Whisper)
            min_speech_duration_ms: Minimum speech duration to keep (ms)
        """
        if aggressiveness not in AGGRESSIVENESS_THRESHOLDS:
            raise ValueError(f"aggressiveness must be 0-3, got {aggressiveness}")

        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms

        # Get threshold and silence duration from aggressiveness
        self.threshold = AGGRESSIVENESS_THRESHOLDS[aggressiveness]
        self.min_silence_duration_ms = AGGRESSIVENESS_SILENCE_MS[aggressiveness]

    @classmethod
    def _load_model(cls) -> tuple[any, any]:
        """
        Load Silero VAD model (class-level singleton).

        Returns:
            Tuple of (model, utils) or raises ImportError if unavailable.
        """
        if cls._load_error is not None:
            raise cls._load_error

        if cls._model is not None:
            return cls._model, cls._utils

        try:
            import torch

            # Load Silero VAD from torch hub
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )

            cls._model = model
            cls._utils = utils
            return model, utils

        except ImportError as e:
            cls._load_error = ImportError(
                "Silero VAD requires PyTorch. Install with: pip install torch\n"
                f"Original error: {e}",
            )
            raise cls._load_error from e
        except Exception as e:
            cls._load_error = ImportError(
                f"Failed to load Silero VAD model: {e}\n"
                "Ensure torch is installed and you have internet access.",
            )
            raise cls._load_error from e

    def get_speech_segments(
        self,
        audio: np.ndarray,
        return_probs: bool = False,
    ) -> VADResult:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio waveform (float32, mono, 16kHz)
            return_probs: If True, also return per-frame probabilities

        Returns:
            VADResult with detected speech segments
        """
        import torch

        model, utils = self._load_model()
        get_speech_timestamps = utils[0]

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to torch tensor
        audio_tensor = torch.tensor(audio)

        # Get speech timestamps from Silero
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            sampling_rate=self.sample_rate,
        )

        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            segments.append(SpeechSegment(
                start=start_time,
                end=end_time,
                start_sample=start_sample,
                end_sample=end_sample,
            ))

        # Calculate statistics
        total_duration = len(audio) / self.sample_rate
        speech_duration = sum(seg.duration for seg in segments)
        speech_ratio = speech_duration / max(total_duration, 0.001)

        return VADResult(
            segments=segments,
            speech_ratio=speech_ratio,
            total_duration=total_duration,
            speech_duration=speech_duration,
        )

    def extract_speech(
        self,
        audio: np.ndarray,
        padding_ms: int = 50,
    ) -> tuple[np.ndarray, VADResult]:
        """
        Extract speech segments from audio, removing silence.

        This concatenates all detected speech segments with optional padding
        to create a shorter audio that contains only speech.

        Args:
            audio: Audio waveform (float32, mono, 16kHz)
            padding_ms: Padding to add around each segment (ms)

        Returns:
            Tuple of (speech_audio, vad_result)
            - speech_audio: Concatenated speech segments
            - vad_result: VAD detection result with segment info
        """
        vad_result = self.get_speech_segments(audio)

        if not vad_result.segments:
            # No speech detected - return empty array
            return np.array([], dtype=np.float32), vad_result

        # Extract and concatenate speech segments with padding
        padding_samples = int(padding_ms * self.sample_rate / 1000)
        speech_chunks = []

        for seg in vad_result.segments:
            # Add padding (clamp to audio bounds)
            start = max(0, seg.start_sample - padding_samples)
            end = min(len(audio), seg.end_sample + padding_samples)
            speech_chunks.append(audio[start:end])

        speech_audio = np.concatenate(speech_chunks)
        return speech_audio, vad_result

    def extract_speech_with_timestamps(
        self,
        audio: np.ndarray,
        padding_ms: int = 50,
    ) -> tuple[np.ndarray, VADResult, list[tuple[float, float]]]:
        """
        Extract speech segments with timestamp mapping.

        Like extract_speech, but also returns a mapping from output audio
        positions to original audio positions. This is needed to restore
        correct timestamps after transcription.

        Args:
            audio: Audio waveform (float32, mono, 16kHz)
            padding_ms: Padding to add around each segment (ms)

        Returns:
            Tuple of (speech_audio, vad_result, timestamp_map)
            - speech_audio: Concatenated speech segments
            - vad_result: VAD detection result
            - timestamp_map: List of (output_time, original_time) pairs
        """
        vad_result = self.get_speech_segments(audio)

        if not vad_result.segments:
            return np.array([], dtype=np.float32), vad_result, []

        padding_samples = int(padding_ms * self.sample_rate / 1000)
        speech_chunks = []
        timestamp_map = []
        output_pos = 0.0

        for seg in vad_result.segments:
            start = max(0, seg.start_sample - padding_samples)
            end = min(len(audio), seg.end_sample + padding_samples)
            chunk = audio[start:end]
            speech_chunks.append(chunk)

            # Map output position to original position
            original_start = start / self.sample_rate
            timestamp_map.append((output_pos, original_start))
            output_pos += len(chunk) / self.sample_rate

        speech_audio = np.concatenate(speech_chunks)
        return speech_audio, vad_result, timestamp_map

    def should_skip_transcription(self, audio: np.ndarray) -> bool:
        """
        Check if audio should skip transcription (mostly silence).

        This is a fast check that runs VAD and returns True if less than 5%
        of the audio is speech.

        Args:
            audio: Audio waveform

        Returns:
            True if audio is mostly silence and should be skipped
        """
        vad_result = self.get_speech_segments(audio)
        return vad_result.is_mostly_silent


# Global singleton for efficient reuse
_vad_processor: SileroVADProcessor | None = None


def get_vad_processor(aggressiveness: int = 2) -> SileroVADProcessor:
    """
    Get global VAD processor singleton.

    This ensures the Silero model is only loaded once per process.

    Args:
        aggressiveness: VAD aggressiveness (0-3)

    Returns:
        SileroVADProcessor instance
    """
    global _vad_processor

    if _vad_processor is None or _vad_processor.aggressiveness != aggressiveness:
        _vad_processor = SileroVADProcessor(aggressiveness=aggressiveness)

    return _vad_processor


def preprocess_audio_with_vad(
    audio: np.ndarray,
    aggressiveness: int = 2,
    padding_ms: int = 50,
) -> tuple[np.ndarray, VADResult]:
    """
    Convenience function to preprocess audio with VAD.

    This is the main entry point for VAD preprocessing in transcribe().

    Args:
        audio: Audio waveform (float32, mono, 16kHz)
        aggressiveness: VAD aggressiveness (0-3)
        padding_ms: Padding around speech segments

    Returns:
        Tuple of (processed_audio, vad_result)
    """
    processor = get_vad_processor(aggressiveness)
    return processor.extract_speech(audio, padding_ms=padding_ms)


__all__ = [
    "SileroVADProcessor",
    "SpeechSegment",
    "VADResult",
    "get_vad_processor",
    "preprocess_audio_with_vad",
    "SILERO_SAMPLE_RATE",
    "AGGRESSIVENESS_THRESHOLDS",
]
