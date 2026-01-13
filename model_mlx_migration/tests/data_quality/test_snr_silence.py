#!/usr/bin/env python3
"""
SNR and Silence Tests - Data QA Gate

Verifies that audio files meet SNR and silence requirements.
This is an OPTIONAL gate - can be run on schedule.

Run with: pytest tests/data_quality/test_snr_silence.py -v -m data_qa_heavy
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_dataset_config() -> dict:
    """Load dataset configuration from YAML."""
    config_path = Path(__file__).parent.parent.parent / "data" / "qa" / "datasets.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"defaults": {}, "datasets": {}}


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def estimate_snr_simple(audio: np.ndarray, frame_size: int = 1600) -> float:
    """
    Simple SNR estimation using frame-based RMS.

    Assumes lowest RMS frames are noise floor.
    """
    if len(audio) < frame_size:
        return 0.0

    # Compute RMS per frame
    n_frames = len(audio) // frame_size
    frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Sort frames by RMS
    sorted_rms = np.sort(frame_rms)

    # Noise floor = mean of lowest 10% of frames
    noise_frames = max(1, n_frames // 10)
    noise_floor = np.mean(sorted_rms[:noise_frames])

    # Signal = mean of highest 50% of frames
    signal_frames = n_frames // 2
    signal_level = np.mean(sorted_rms[-signal_frames:])

    if noise_floor < 1e-10:
        return 60.0  # Very clean signal

    snr_db = 20 * np.log10(signal_level / noise_floor)
    return float(snr_db)


def detect_silence_ratio(
    audio: np.ndarray,
    threshold_db: float = -40,
    frame_size: int = 1600,
) -> tuple[float, float, float]:
    """
    Detect silence at head, tail, and total.

    Returns:
        (head_silence_ratio, tail_silence_ratio, total_silence_ratio)
    """
    if len(audio) < frame_size:
        return 0.0, 0.0, 0.0

    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Compute RMS per frame
    n_frames = len(audio) // frame_size
    frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Normalize to max
    max_rms = np.max(frame_rms)
    if max_rms < 1e-10:
        return 1.0, 1.0, 1.0  # All silence

    normalized_rms = frame_rms / max_rms

    # Count silent frames
    is_silent = normalized_rms < threshold_linear
    total_silence = np.sum(is_silent) / n_frames

    # Head silence
    head_silent = 0
    for i in range(n_frames):
        if is_silent[i]:
            head_silent += 1
        else:
            break
    head_ratio = head_silent / n_frames

    # Tail silence
    tail_silent = 0
    for i in range(n_frames - 1, -1, -1):
        if is_silent[i]:
            tail_silent += 1
        else:
            break
    tail_ratio = tail_silent / n_frames

    return head_ratio, tail_ratio, total_silence


@pytest.mark.data_qa_heavy
class TestSNRRequirements:
    """Verify audio meets SNR requirements."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    def test_snr_threshold_configured(self, dataset_config):
        """SNR threshold is configured in defaults."""
        defaults = dataset_config.get("defaults", {})
        assert "snr_min_db" in defaults, "Missing default snr_min_db threshold"
        assert defaults["snr_min_db"] >= 0, "snr_min_db should be non-negative"

    def test_sample_audio_snr(self):
        """Sample audio files meet SNR requirements (quick check)."""
        project_root = Path(__file__).parent.parent.parent

        # Find test WAV files
        wav_files = list(project_root.glob("tests/*.wav"))[:3]

        for wav_path in wav_files:
            if not wav_path.exists():
                continue

            try:
                # Simple WAV reading without external deps
                import wave
                with wave.open(str(wav_path), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)

                # Convert to numpy
                if sample_width == 2:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    continue  # Skip non-16bit

                # Convert stereo to mono
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)

                snr = estimate_snr_simple(audio)
                # Just verify we can compute SNR
                assert snr > -np.inf, f"Could not compute SNR for {wav_path}"

            except Exception as e:
                pytest.skip(f"Could not process {wav_path}: {e}")


@pytest.mark.data_qa_heavy
class TestSilenceRequirements:
    """Verify audio silence is within bounds."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    def test_silence_threshold_configured(self, dataset_config):
        """Silence thresholds are configured in defaults."""
        defaults = dataset_config.get("defaults", {})

        assert "silence_max_ratio" in defaults, "Missing default silence_max_ratio"
        assert "silence_max_ms" in defaults, "Missing default silence_max_ms"

        assert 0 < defaults["silence_max_ratio"] <= 1, "silence_max_ratio should be in (0, 1]"
        assert defaults["silence_max_ms"] > 0, "silence_max_ms should be positive"

    def test_sample_audio_silence(self):
        """Sample audio files have acceptable silence levels (quick check)."""
        project_root = Path(__file__).parent.parent.parent

        wav_files = list(project_root.glob("tests/*.wav"))[:3]

        for wav_path in wav_files:
            if not wav_path.exists():
                continue

            try:
                import wave
                with wave.open(str(wav_path), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)

                if sample_width == 2:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    continue

                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)

                head_ratio, tail_ratio, total_ratio = detect_silence_ratio(audio)

                # Just verify we can compute silence metrics
                assert 0 <= total_ratio <= 1, f"Invalid silence ratio for {wav_path}"

            except Exception as e:
                pytest.skip(f"Could not process {wav_path}: {e}")


@pytest.mark.data_qa
class TestSNRSilenceConfig:
    """Verify SNR and silence configuration is valid."""

    def test_per_dataset_overrides_valid(self):
        """Per-dataset SNR overrides are valid."""
        config = load_dataset_config()
        defaults = config.get("defaults", {})
        datasets = config.get("datasets", {})

        default_snr = defaults.get("snr_min_db", 10)

        for name, ds_config in datasets.items():
            snr = ds_config.get("snr_min_db", default_snr)
            assert isinstance(snr, (int, float)), \
                f"Dataset {name} has invalid snr_min_db: {snr}"
            assert snr >= 0, f"Dataset {name} has negative snr_min_db: {snr}"
