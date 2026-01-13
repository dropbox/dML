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

"""Tests for multi-speaker pipeline (Phase 9.2)."""

import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

from src.server.multi_speaker import (
    MockMultiSpeakerPipeline,
    MultiSpeakerConfig,
    MultiSpeakerPipeline,
    MultiSpeakerResult,
    OverlapDetector,
    OverlapStatus,
    SpeakerSegment,
)


class TestOverlapStatus:
    """Tests for OverlapStatus enum."""

    def test_all_statuses(self):
        assert OverlapStatus.SINGLE.value == "single"
        assert OverlapStatus.OVERLAP.value == "overlap"
        assert OverlapStatus.SILENCE.value == "silence"


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_basic_creation(self):
        audio = _rng.standard_normal(16000).astype(np.float32)
        segment = SpeakerSegment(
            speaker_id=0,
            audio=audio,
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        assert segment.speaker_id == 0
        assert len(segment.audio) == 16000
        assert segment.confidence == 1.0  # Default

    def test_with_confidence(self):
        audio = np.zeros(1600, dtype=np.float32)
        segment = SpeakerSegment(
            speaker_id=1,
            audio=audio,
            start_time_ms=1000.0,
            end_time_ms=1100.0,
            confidence=0.95,
        )
        assert segment.confidence == 0.95


class TestMultiSpeakerConfig:
    """Tests for MultiSpeakerConfig dataclass."""

    def test_default_config(self):
        config = MultiSpeakerConfig()
        assert config.enabled
        assert config.max_speakers == 2
        assert config.sample_rate == 16000
        assert config.keep_original

    def test_custom_config(self):
        config = MultiSpeakerConfig(
            enabled=False,
            max_speakers=3,
            overlap_threshold=0.5,
        )
        assert not config.enabled
        assert config.max_speakers == 3
        assert config.overlap_threshold == 0.5


class TestMultiSpeakerResult:
    """Tests for MultiSpeakerResult dataclass."""

    def test_single_speaker(self):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = MultiSpeakerResult(
            status=OverlapStatus.SINGLE,
            num_speakers=1,
            segments=[
                SpeakerSegment(
                    speaker_id=0,
                    audio=audio,
                    start_time_ms=0.0,
                    end_time_ms=1000.0,
                ),
            ],
        )
        assert result.status == OverlapStatus.SINGLE
        assert result.num_speakers == 1
        assert len(result.segments) == 1

    def test_multi_speaker(self):
        audio1 = _rng.standard_normal(16000).astype(np.float32)
        audio2 = _rng.standard_normal(16000).astype(np.float32)
        result = MultiSpeakerResult(
            status=OverlapStatus.OVERLAP,
            num_speakers=2,
            segments=[
                SpeakerSegment(0, audio1, 0.0, 1000.0),
                SpeakerSegment(1, audio2, 0.0, 1000.0),
            ],
        )
        assert result.status == OverlapStatus.OVERLAP
        assert result.num_speakers == 2
        assert len(result.segments) == 2

    def test_silence(self):
        result = MultiSpeakerResult(
            status=OverlapStatus.SILENCE,
            num_speakers=0,
            segments=[],
        )
        assert result.status == OverlapStatus.SILENCE
        assert result.num_speakers == 0
        assert len(result.segments) == 0


class TestOverlapDetector:
    """Tests for OverlapDetector."""

    @pytest.fixture
    def detector(self):
        config = MultiSpeakerConfig()
        return OverlapDetector(config)

    def test_detect_silence(self, detector):
        # Very quiet audio
        audio = np.zeros(16000, dtype=np.float32) + 1e-8
        result = detector.detect(audio)
        assert result == OverlapStatus.SILENCE

    def test_detect_single_speaker(self, detector):
        # Steady speech-like signal
        t = np.arange(16000) / 16000
        audio = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        result = detector.detect(audio)
        # Most steady signals will be detected as single speaker
        assert result in [OverlapStatus.SINGLE, OverlapStatus.OVERLAP]

    def test_detect_empty_audio(self, detector):
        audio = np.array([], dtype=np.float32)
        result = detector.detect(audio)
        assert result == OverlapStatus.SILENCE

    def test_detect_short_audio(self, detector):
        # Very short audio
        audio = _rng.standard_normal(100).astype(np.float32) * 0.1
        result = detector.detect(audio)
        # Short audio might be detected as silence
        assert result in [OverlapStatus.SILENCE, OverlapStatus.SINGLE]


class TestMultiSpeakerPipeline:
    """Tests for MultiSpeakerPipeline."""

    @pytest.fixture
    def pipeline(self):
        # Create pipeline without loading model (separator=None)
        config = MultiSpeakerConfig(enabled=True)
        return MultiSpeakerPipeline(config=config)

    @pytest.fixture
    def disabled_pipeline(self):
        config = MultiSpeakerConfig(enabled=False)
        return MultiSpeakerPipeline(config=config)

    def test_disabled_pipeline(self, disabled_pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = disabled_pipeline.separate(audio)
        # Disabled pipeline returns single speaker
        assert result.status == OverlapStatus.SINGLE
        assert result.num_speakers == 1
        assert len(result.segments) == 1

    def test_separate_returns_result(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = pipeline.separate(audio)
        assert isinstance(result, MultiSpeakerResult)
        assert result.num_speakers >= 0

    def test_separate_with_start_time(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = pipeline.separate(audio, start_time_ms=5000.0)
        if result.segments:
            assert result.segments[0].start_time_ms == 5000.0

    def test_detect_overlap(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        status = pipeline.detect_overlap(audio)
        assert isinstance(status, OverlapStatus)

    @pytest.mark.asyncio
    async def test_separate_async(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = await pipeline.separate_async(audio)
        assert isinstance(result, MultiSpeakerResult)


class TestMockMultiSpeakerPipeline:
    """Tests for MockMultiSpeakerPipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        return MockMultiSpeakerPipeline()

    def test_default_single_speaker(self, mock_pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_pipeline.separate(audio)
        assert result.status == OverlapStatus.SINGLE
        assert result.num_speakers == 1

    def test_set_mock_overlap(self, mock_pipeline):
        mock_pipeline.set_mock_result(OverlapStatus.OVERLAP, num_speakers=2)
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_pipeline.separate(audio)
        assert result.status == OverlapStatus.OVERLAP
        assert result.num_speakers == 2
        assert len(result.segments) == 2

    def test_set_mock_silence(self, mock_pipeline):
        mock_pipeline.set_mock_result(OverlapStatus.SILENCE)
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_pipeline.separate(audio)
        assert result.status == OverlapStatus.SILENCE
        assert result.num_speakers == 0
        assert len(result.segments) == 0

    def test_mock_detection(self, mock_pipeline):
        mock_pipeline.set_mock_result(OverlapStatus.OVERLAP, num_speakers=3)
        audio = _rng.standard_normal(16000).astype(np.float32)
        status = mock_pipeline.detect_overlap(audio)
        assert status == OverlapStatus.OVERLAP

    def test_mock_keeps_original(self, mock_pipeline):
        mock_pipeline.config.keep_original = True
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_pipeline.separate(audio)
        assert result.original_audio is not None
        assert len(result.original_audio) == len(audio)

    def test_mock_segments_have_audio(self, mock_pipeline):
        mock_pipeline.set_mock_result(OverlapStatus.OVERLAP, num_speakers=2)
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_pipeline.separate(audio)
        for segment in result.segments:
            assert len(segment.audio) == len(audio)
            assert segment.audio is not audio  # Should be modified copy


class TestMultiSpeakerIntegration:
    """Integration tests for multi-speaker pipeline."""

    def test_pipeline_with_config(self):
        config = MultiSpeakerConfig(
            enabled=True,
            max_speakers=2,
            overlap_threshold=0.4,
            sample_rate=16000,
        )
        pipeline = MockMultiSpeakerPipeline(config)
        assert pipeline.config.max_speakers == 2

    def test_segment_timing(self):
        mock = MockMultiSpeakerPipeline()
        mock.set_mock_result(OverlapStatus.OVERLAP, num_speakers=2)

        audio = _rng.standard_normal(32000).astype(np.float32)  # 2 seconds
        result = mock.separate(audio, start_time_ms=1000.0)

        for segment in result.segments:
            assert segment.start_time_ms == 1000.0
            assert segment.end_time_ms == 3000.0  # 1000 + 2000ms

    def test_audio_duration_calculation(self):
        config = MultiSpeakerConfig(sample_rate=16000)
        mock = MockMultiSpeakerPipeline(config)

        # 1.5 seconds of audio
        audio = np.zeros(24000, dtype=np.float32)
        result = mock.separate(audio, start_time_ms=500.0)

        if result.segments:
            duration = result.segments[0].end_time_ms - result.segments[0].start_time_ms
            assert abs(duration - 1500.0) < 1.0  # Should be ~1500ms
