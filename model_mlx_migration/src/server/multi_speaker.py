# Copyright 2024-2026 Andrew Yates
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
Multi-speaker pipeline for SOTA++ Voice Server (Phase 9.2).

Handles detection and separation of multiple speakers using FLASepformer/MossFormer2.
When overlap is detected, audio is separated into individual speaker streams
which are then processed in parallel by the ASR pipeline.

Architecture:
    Mixed Audio → Overlap Detection → [Single Speaker] → Direct ASR
                                   → [Multi Speaker] → Separation → Parallel ASR → Merge
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

try:
    import mlx.core as mx  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from ..models.flasepformer import SourceSeparator, create_separator
    HAS_SEPARATOR = True
except ImportError:
    HAS_SEPARATOR = False


class OverlapStatus(str, Enum):
    """Speaker overlap detection status."""
    SINGLE = "single"  # Single speaker or no overlap
    OVERLAP = "overlap"  # Multiple overlapping speakers
    SILENCE = "silence"  # No speech detected


@dataclass
class SpeakerSegment:
    """A separated audio segment from one speaker."""
    speaker_id: int  # 0-indexed speaker number from separation
    audio: np.ndarray  # Separated audio samples
    start_time_ms: float
    end_time_ms: float
    confidence: float = 1.0  # Separation confidence


@dataclass
class MultiSpeakerResult:
    """Result of multi-speaker detection and separation."""
    status: OverlapStatus
    num_speakers: int
    segments: list[SpeakerSegment] = field(default_factory=list)
    original_audio: np.ndarray | None = None


@dataclass
class MultiSpeakerConfig:
    """Configuration for multi-speaker pipeline."""
    # Enable/disable multi-speaker handling
    enabled: bool = True

    # Maximum speakers to separate
    max_speakers: int = 2  # MossFormer2 supports 2 or 3

    # Overlap detection threshold (energy-based)
    overlap_threshold: float = 0.3

    # Minimum duration for overlap detection (ms)
    min_overlap_duration_ms: float = 200.0

    # Sample rate
    sample_rate: int = 16000

    # Whether to keep original mixed audio
    keep_original: bool = True


class OverlapDetector:
    """
    Simple overlap detection using energy analysis.

    For production, this would use a trained VAD or overlap detector,
    but for now we use energy-based heuristics.
    """

    def __init__(self, config: MultiSpeakerConfig):
        self.config = config

    def detect(self, audio: np.ndarray) -> OverlapStatus:
        """
        Detect speaker overlap in audio.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            OverlapStatus indicating single, overlap, or silence
        """
        # Compute frame-level energy
        frame_size = int(0.025 * self.config.sample_rate)  # 25ms frames
        hop_size = int(0.010 * self.config.sample_rate)  # 10ms hop

        energies = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)

        if not energies:
            return OverlapStatus.SILENCE

        energies = np.array(energies)
        mean_energy = np.mean(energies)

        # Silence detection
        if mean_energy < 1e-6:
            return OverlapStatus.SILENCE

        # Overlap detection heuristic:
        # High variance in energy + high peaks can indicate overlapping speech
        energy_std = np.std(energies)
        energy_max = np.max(energies)

        # Compute kurtosis - overlapping speech tends to have higher kurtosis
        if mean_energy > 0:
            normalized = (energies - mean_energy) / (energy_std + 1e-8)
            kurtosis = np.mean(normalized ** 4) - 3  # Excess kurtosis
        else:
            kurtosis = 0

        # Simple heuristic: high kurtosis + high max/mean ratio suggests overlap
        max_mean_ratio = energy_max / (mean_energy + 1e-8)

        if kurtosis > 2.0 and max_mean_ratio > 5.0:
            return OverlapStatus.OVERLAP

        return OverlapStatus.SINGLE


class MultiSpeakerPipeline:
    """
    Multi-speaker detection and separation pipeline.

    Integrates overlap detection with MossFormer2 source separation.
    """

    def __init__(
        self,
        config: MultiSpeakerConfig | None = None,
        separator: Optional["SourceSeparator"] = None,
    ):
        self.config = config or MultiSpeakerConfig()
        self._separator = separator
        self._separator_loaded = False
        self._detector = OverlapDetector(self.config)

    @property
    def separator(self) -> Optional["SourceSeparator"]:
        """Lazy-load separator on first use."""
        if not self.config.enabled:
            return None

        if not HAS_SEPARATOR:
            return None

        if self._separator is None and not self._separator_loaded:
            try:
                self._separator = create_separator(
                    num_speakers=self.config.max_speakers,
                    sample_rate=self.config.sample_rate,
                )
                self._separator_loaded = True
            except Exception:
                self._separator_loaded = True  # Don't retry on error
                return None

        return self._separator

    def detect_overlap(self, audio: np.ndarray) -> OverlapStatus:
        """
        Detect if audio contains overlapping speakers.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            OverlapStatus
        """
        return self._detector.detect(audio)

    def separate(
        self,
        audio: np.ndarray,
        start_time_ms: float = 0.0,
    ) -> MultiSpeakerResult:
        """
        Separate mixed audio into speaker streams.

        Args:
            audio: Mixed audio samples at 16kHz
            start_time_ms: Start time of this audio segment

        Returns:
            MultiSpeakerResult with separated speaker segments
        """
        duration_ms = len(audio) / self.config.sample_rate * 1000

        # Check if separation is enabled and available
        if not self.config.enabled or self.separator is None:
            return MultiSpeakerResult(
                status=OverlapStatus.SINGLE,
                num_speakers=1,
                segments=[
                    SpeakerSegment(
                        speaker_id=0,
                        audio=audio,
                        start_time_ms=start_time_ms,
                        end_time_ms=start_time_ms + duration_ms,
                    ),
                ],
                original_audio=audio if self.config.keep_original else None,
            )

        # Detect overlap first
        status = self.detect_overlap(audio)

        if status == OverlapStatus.SILENCE:
            return MultiSpeakerResult(
                status=OverlapStatus.SILENCE,
                num_speakers=0,
                segments=[],
                original_audio=audio if self.config.keep_original else None,
            )

        if status == OverlapStatus.SINGLE:
            return MultiSpeakerResult(
                status=OverlapStatus.SINGLE,
                num_speakers=1,
                segments=[
                    SpeakerSegment(
                        speaker_id=0,
                        audio=audio,
                        start_time_ms=start_time_ms,
                        end_time_ms=start_time_ms + duration_ms,
                    ),
                ],
                original_audio=audio if self.config.keep_original else None,
            )

        # Run separation
        try:
            separated = self.separator.separate_to_numpy(audio)

            segments = []
            for idx, src_audio in enumerate(separated):
                segments.append(
                    SpeakerSegment(
                        speaker_id=idx,
                        audio=src_audio,
                        start_time_ms=start_time_ms,
                        end_time_ms=start_time_ms + duration_ms,
                    ),
                )

            return MultiSpeakerResult(
                status=OverlapStatus.OVERLAP,
                num_speakers=len(separated),
                segments=segments,
                original_audio=audio if self.config.keep_original else None,
            )

        except Exception:
            # Fallback to single speaker on separation error
            return MultiSpeakerResult(
                status=OverlapStatus.SINGLE,
                num_speakers=1,
                segments=[
                    SpeakerSegment(
                        speaker_id=0,
                        audio=audio,
                        start_time_ms=start_time_ms,
                        end_time_ms=start_time_ms + duration_ms,
                    ),
                ],
                original_audio=audio if self.config.keep_original else None,
            )

    async def separate_async(
        self,
        audio: np.ndarray,
        start_time_ms: float = 0.0,
    ) -> MultiSpeakerResult:
        """
        Async version of separate() for use in async server.

        Note: Actual MLX inference is synchronous, but this allows
        the separation to be awaited in async context.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.separate(audio, start_time_ms),
        )


class MockMultiSpeakerPipeline(MultiSpeakerPipeline):
    """
    Mock multi-speaker pipeline for testing.

    Returns deterministic results without loading models.
    """

    def __init__(self, config: MultiSpeakerConfig | None = None):
        super().__init__(config=config)
        self._mock_overlap = OverlapStatus.SINGLE
        self._mock_num_speakers = 1

    def set_mock_result(self, status: OverlapStatus, num_speakers: int = 1):
        """Configure mock result."""
        self._mock_overlap = status
        self._mock_num_speakers = num_speakers

    def detect_overlap(self, audio: np.ndarray) -> OverlapStatus:
        """Return mock overlap status."""
        return self._mock_overlap

    def separate(
        self,
        audio: np.ndarray,
        start_time_ms: float = 0.0,
    ) -> MultiSpeakerResult:
        """Return mock separation result."""
        duration_ms = len(audio) / self.config.sample_rate * 1000

        if self._mock_overlap == OverlapStatus.SILENCE:
            return MultiSpeakerResult(
                status=OverlapStatus.SILENCE,
                num_speakers=0,
                segments=[],
                original_audio=audio if self.config.keep_original else None,
            )

        segments = []
        for i in range(self._mock_num_speakers):
            # Mock separation by scaling audio differently per speaker
            separated = audio * (0.8 + 0.1 * i)
            segments.append(
                SpeakerSegment(
                    speaker_id=i,
                    audio=separated,
                    start_time_ms=start_time_ms,
                    end_time_ms=start_time_ms + duration_ms,
                ),
            )

        return MultiSpeakerResult(
            status=self._mock_overlap,
            num_speakers=self._mock_num_speakers,
            segments=segments,
            original_audio=audio if self.config.keep_original else None,
        )
