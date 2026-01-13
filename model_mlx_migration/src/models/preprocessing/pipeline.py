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
Audio preprocessing pipeline for SOTA++ Voice Server.

CRITICAL: No denoising for ASR per "Denoising Hurts ASR" paper.
Enhancement is bypassed entirely for the ASR path.
"""

import time
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

try:
    import mlx.core as mx  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing pipeline."""

    # Target sample rate (16kHz for ASR)
    target_sample_rate: int = 16000

    # LUFS normalization target (-23 LUFS is broadcast standard)
    target_lufs: float = -23.0
    lufs_block_size: float = 0.4  # 400ms block for LUFS calculation

    # DC removal
    enable_dc_removal: bool = True

    # VAD parameters
    enable_vad: bool = True
    vad_threshold: float = 0.5  # Probability threshold for speech
    vad_min_speech_ms: int = 250  # Minimum speech duration
    vad_min_silence_ms: int = 100  # Minimum silence duration
    vad_window_ms: int = 32  # VAD window size
    vad_padding_ms: int = 100  # Padding around speech segments

    # Chunking parameters
    chunk_size_ms: int = 320  # 320ms chunks for streaming ASR
    chunk_overlap_ms: int = 0  # No overlap by default

    # CRITICAL: Denoising is DISABLED for ASR
    # Per "Denoising Hurts ASR" paper, enhancement harms recognition
    enable_denoising: bool = False  # DO NOT CHANGE


@dataclass
class AudioChunk:
    """A chunk of preprocessed audio ready for ASR."""

    # Audio samples at target sample rate
    samples: np.ndarray

    # Sample rate (always target_sample_rate)
    sample_rate: int

    # Timing information
    start_time_ms: float
    end_time_ms: float

    # VAD information
    is_speech: bool = True
    speech_probability: float = 1.0

    # Chunk index in stream
    chunk_index: int = 0

    # Whether this is the final chunk
    is_final: bool = False


@dataclass
class VADResult:
    """Result from Voice Activity Detection."""

    # Speech segments as list of (start_ms, end_ms) tuples
    speech_segments: list[tuple[float, float]] = field(default_factory=list)

    # Frame-level probabilities
    frame_probs: np.ndarray | None = None

    # Total speech duration in ms
    total_speech_ms: float = 0.0

    # Total silence duration in ms
    total_silence_ms: float = 0.0


class PreprocessingPipeline:
    """
    Audio preprocessing pipeline for streaming ASR.

    Pipeline stages:
    1. Resample to 16kHz (sinc interpolation via torchaudio)
    2. DC removal (subtract mean)
    3. AGC (LUFS normalization)
    4. VAD (Silero or similar)
    5. Chunking (320ms chunks)

    NOTE: Denoising is DISABLED by design - it hurts ASR performance.

    Usage:
        pipeline = PreprocessingPipeline()
        for chunk in pipeline.process_streaming(audio, sample_rate):
            # Process each chunk with ASR
            text = asr.transcribe_chunk(chunk.samples)
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize preprocessing pipeline."""
        self.config = config or PreprocessingConfig()
        self._vad_model = None
        self._resampler_cache = {}

        # Verify critical constraint
        if self.config.enable_denoising:
            raise ValueError(
                "Denoising MUST be disabled for ASR. "
                "Per 'Denoising Hurts ASR' paper, enhancement harms recognition. "
                "Set enable_denoising=False.",
            )

    def _get_resampler(self, source_sr: int, target_sr: int):
        """Get or create cached resampler."""
        if not HAS_TORCHAUDIO:
            raise ImportError("torchaudio required for resampling")

        key = (source_sr, target_sr)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = torchaudio.transforms.Resample(
                orig_freq=source_sr,
                new_freq=target_sr,
            )
        return self._resampler_cache[key]

    def resample(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int | None = None,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses sinc interpolation via torchaudio for high quality.
        Latency: ~1ms for typical audio lengths.

        Args:
            audio: Audio samples as 1D numpy array.
            source_sr: Source sample rate.
            target_sr: Target sample rate. Defaults to config.target_sample_rate.

        Returns:
            Resampled audio at target sample rate.
        """
        target_sr = target_sr or self.config.target_sample_rate

        if source_sr == target_sr:
            return audio

        if not HAS_TORCHAUDIO:
            raise ImportError(
                f"Audio is {source_sr}Hz but target is {target_sr}Hz. "
                "torchaudio required for resampling.",
            )

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample using cached resampler
        resampler = self._get_resampler(source_sr, target_sr)
        resampled = resampler(audio_tensor)

        return resampled.squeeze(0).numpy()

    def remove_dc(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio.

        Simple mean subtraction - extremely fast (<1ms).

        Args:
            audio: Audio samples.

        Returns:
            Audio with DC offset removed.
        """
        if not self.config.enable_dc_removal:
            return audio
        return audio - np.mean(audio)

    def compute_lufs(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Compute integrated LUFS loudness.

        Uses ITU-R BS.1770-4 momentary loudness calculation.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Integrated LUFS value.
        """
        if len(audio) == 0:
            return -70.0  # Silence

        # Pre-filter (simplified K-weighting)
        # Full K-weighting uses two-stage filter but this is adequate for normalization

        # Calculate RMS in blocks
        block_samples = int(self.config.lufs_block_size * sample_rate)
        if block_samples == 0 or len(audio) < block_samples:
            block_samples = len(audio)

        # Compute mean square per block
        num_blocks = max(1, len(audio) // block_samples)
        block_powers = []

        for i in range(num_blocks):
            start = i * block_samples
            end = min(start + block_samples, len(audio))
            block = audio[start:end]
            if len(block) > 0:
                # Mean square
                mean_sq = np.mean(block ** 2)
                if mean_sq > 0:
                    block_powers.append(mean_sq)

        if not block_powers:
            return -70.0

        # Gated loudness (simplified - use all blocks above threshold)
        threshold_power = 10 ** (-70.0 / 10.0)  # -70 LUFS absolute gate
        gated_powers = [p for p in block_powers if p > threshold_power]

        if not gated_powers:
            return -70.0

        # Mean power
        mean_power = np.mean(gated_powers)

        # Convert to LUFS
        lufs = 10 * np.log10(mean_power + 1e-10) - 0.691

        return float(lufs)

    def normalize_lufs(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_lufs: float | None = None,
    ) -> np.ndarray:
        """
        Normalize audio to target LUFS loudness.

        Latency: <1ms for typical audio lengths.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.
            target_lufs: Target LUFS. Defaults to config.target_lufs.

        Returns:
            Normalized audio.
        """
        target_lufs = target_lufs or self.config.target_lufs

        current_lufs = self.compute_lufs(audio, sample_rate)

        # Avoid extreme adjustments
        if current_lufs < -60.0:  # Too quiet (probably silence)
            return audio

        # Calculate gain
        gain_db = target_lufs - current_lufs

        # Limit gain to avoid clipping/distortion
        gain_db = np.clip(gain_db, -20.0, 20.0)

        gain_linear = 10 ** (gain_db / 20.0)

        return audio * gain_linear

    def _load_vad_model(self):
        """Load Silero VAD model."""
        if self._vad_model is not None:
            return

        try:
            # Try silero-vad package first
            import silero_vad
            self._vad_model = silero_vad.load_silero_vad()
            self._vad_type = "silero_package"
        except ImportError:
            try:
                # Fall back to torch.hub
                if not HAS_TORCHAUDIO:
                    raise ImportError("torchaudio required for Silero VAD")

                self._vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True,
                )
                self._get_speech_timestamps = utils[0]
                self._vad_type = "torch_hub"
            except Exception as e:
                raise ImportError(
                    f"Could not load Silero VAD: {e}. "
                    "Install with: pip install silero-vad",
                ) from e

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VADResult:
        """
        Detect speech segments in audio using Silero VAD.

        Latency: ~10ms for typical audio lengths.

        Args:
            audio: Audio samples at any sample rate.
            sample_rate: Sample rate of audio.

        Returns:
            VADResult with speech segments and probabilities.
        """
        if not self.config.enable_vad:
            # Return full audio as speech
            duration_ms = len(audio) / sample_rate * 1000
            return VADResult(
                speech_segments=[(0.0, duration_ms)],
                total_speech_ms=duration_ms,
                total_silence_ms=0.0,
            )

        self._load_vad_model()

        # Silero VAD expects 16kHz audio
        if sample_rate != 16000:
            audio = self.resample(audio, sample_rate, 16000)
            sample_rate = 16000

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        if self._vad_type == "silero_package":
            # Use silero-vad package API
            import silero_vad
            speech_timestamps = silero_vad.get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=sample_rate,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_ms,
                min_silence_duration_ms=self.config.vad_min_silence_ms,
            )
        else:
            # Use torch.hub utils
            speech_timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=sample_rate,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_ms,
                min_silence_duration_ms=self.config.vad_min_silence_ms,
            )

        # Convert to ms
        speech_segments = []
        total_speech_ms = 0.0

        for segment in speech_timestamps:
            start_ms = segment['start'] / sample_rate * 1000
            end_ms = segment['end'] / sample_rate * 1000

            # Add padding
            start_ms = max(0, start_ms - self.config.vad_padding_ms)
            end_ms = min(len(audio) / sample_rate * 1000, end_ms + self.config.vad_padding_ms)

            speech_segments.append((start_ms, end_ms))
            total_speech_ms += end_ms - start_ms

        total_duration_ms = len(audio) / sample_rate * 1000

        return VADResult(
            speech_segments=speech_segments,
            total_speech_ms=total_speech_ms,
            total_silence_ms=total_duration_ms - total_speech_ms,
        )

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_size_ms: int | None = None,
        chunk_overlap_ms: int | None = None,
    ) -> Generator[AudioChunk, None, None]:
        """
        Split audio into chunks for streaming processing.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.
            chunk_size_ms: Chunk size in ms. Defaults to config.chunk_size_ms.
            chunk_overlap_ms: Overlap between chunks. Defaults to config.chunk_overlap_ms.

        Yields:
            AudioChunk objects.
        """
        chunk_size_ms = chunk_size_ms or self.config.chunk_size_ms
        chunk_overlap_ms = chunk_overlap_ms or self.config.chunk_overlap_ms

        chunk_samples = int(chunk_size_ms * sample_rate / 1000)
        overlap_samples = int(chunk_overlap_ms * sample_rate / 1000)
        step_samples = chunk_samples - overlap_samples

        total_samples = len(audio)
        chunk_index = 0
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_audio = audio[start_sample:end_sample]

            # Pad last chunk if needed
            is_final = end_sample >= total_samples
            if len(chunk_audio) < chunk_samples and not is_final:
                # This shouldn't happen with the logic above, but handle it
                padding = np.zeros(chunk_samples - len(chunk_audio), dtype=audio.dtype)
                chunk_audio = np.concatenate([chunk_audio, padding])

            start_ms = start_sample / sample_rate * 1000
            end_ms = end_sample / sample_rate * 1000

            yield AudioChunk(
                samples=chunk_audio,
                sample_rate=sample_rate,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                chunk_index=chunk_index,
                is_final=is_final,
            )

            chunk_index += 1
            start_sample += step_samples

            # Handle final partial chunk
            if start_sample >= total_samples:
                break

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Apply full preprocessing to audio (without chunking).

        Stages:
        1. Resample to 16kHz
        2. DC removal
        3. LUFS normalization

        NOTE: No denoising - this is intentional!

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Preprocessed audio at target sample rate.
        """
        # 1. Resample
        audio = self.resample(audio, sample_rate, self.config.target_sample_rate)

        # 2. DC removal
        audio = self.remove_dc(audio)

        # 3. LUFS normalization
        audio = self.normalize_lufs(audio, self.config.target_sample_rate)

        return audio

    def process_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int,
        skip_silence: bool = False,
    ) -> Generator[AudioChunk, None, None]:
        """
        Process audio for streaming ASR.

        Full pipeline:
        1. Resample to 16kHz
        2. DC removal
        3. LUFS normalization
        4. VAD (optional filtering)
        5. Chunking

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.
            skip_silence: If True, skip chunks detected as silence.

        Yields:
            AudioChunk objects ready for ASR.
        """
        # Apply preprocessing
        preprocessed = self.preprocess(audio, sample_rate)
        target_sr = self.config.target_sample_rate

        # Get VAD results if needed
        vad_result = None
        if self.config.enable_vad and skip_silence:
            vad_result = self.detect_speech(preprocessed, target_sr)

        # Generate chunks
        for chunk in self.chunk_audio(preprocessed, target_sr):
            # Check if chunk is speech (if VAD enabled and skip_silence)
            if vad_result is not None and skip_silence:
                chunk_mid_ms = (chunk.start_time_ms + chunk.end_time_ms) / 2
                is_speech = any(
                    start <= chunk_mid_ms <= end
                    for start, end in vad_result.speech_segments
                )
                if not is_speech:
                    continue
                chunk.is_speech = is_speech

            yield chunk

    def benchmark(
        self,
        audio: np.ndarray,
        sample_rate: int,
        iterations: int = 10,
    ) -> dict:
        """
        Benchmark preprocessing latency.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.
            iterations: Number of iterations for timing.

        Returns:
            Dict with timing results for each stage.
        """
        results = {}

        # Resample timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            resampled = self.resample(audio, sample_rate)
            times.append((time.perf_counter() - start) * 1000)
        results['resample_ms'] = np.mean(times)

        # DC removal timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            dc_removed = self.remove_dc(resampled)
            times.append((time.perf_counter() - start) * 1000)
        results['dc_removal_ms'] = np.mean(times)

        # LUFS timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            normalized = self.normalize_lufs(dc_removed, self.config.target_sample_rate)
            times.append((time.perf_counter() - start) * 1000)
        results['lufs_normalize_ms'] = np.mean(times)

        # VAD timing
        if self.config.enable_vad:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                self.detect_speech(normalized, self.config.target_sample_rate)
                times.append((time.perf_counter() - start) * 1000)
            results['vad_ms'] = np.mean(times)

        # Total preprocessing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.preprocess(audio, sample_rate)
            times.append((time.perf_counter() - start) * 1000)
        results['total_preprocess_ms'] = np.mean(times)

        # Audio duration
        results['audio_duration_ms'] = len(audio) / sample_rate * 1000
        results['real_time_factor'] = results['total_preprocess_ms'] / results['audio_duration_ms']

        return results


def load_audio(
    path: str,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate. If None, returns original sample rate.

    Returns:
        Tuple of (audio, sample_rate).
    """
    if HAS_SOUNDFILE:
        audio, sr = sf.read(path)

        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

    elif HAS_TORCHAUDIO:
        audio, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)

        audio = audio.numpy()

    else:
        raise ImportError("soundfile or torchaudio required for audio loading")

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        if not HAS_TORCHAUDIO:
            raise ImportError(f"torchaudio required for resampling from {sr}Hz to {target_sr}Hz")

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio_tensor).squeeze(0).numpy()
        sr = target_sr

    return audio, sr
