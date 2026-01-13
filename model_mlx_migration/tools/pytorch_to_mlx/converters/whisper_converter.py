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
Whisper STT Converter

Wraps mlx-whisper for speech-to-text transcription on Apple Silicon.
Provides a consistent interface with our converter tool.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx_whisper

    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

import numpy as np


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    success: bool
    text: str
    language: str | None = None
    segments: list[dict[str, Any]] | None = None
    duration_seconds: float = 0.0
    transcription_time_seconds: float = 0.0
    real_time_factor: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark result for transcription performance."""

    audio_duration_seconds: float
    transcription_time_seconds: float
    real_time_factor: float
    words_per_second: float
    model: str


# Available models from mlx-community
AVAILABLE_MODELS = [
    "mlx-community/whisper-tiny",
    "mlx-community/whisper-tiny.en",
    "mlx-community/whisper-base",
    "mlx-community/whisper-base.en",
    "mlx-community/whisper-small",
    "mlx-community/whisper-small.en",
    "mlx-community/whisper-medium",
    "mlx-community/whisper-medium.en",
    "mlx-community/whisper-large-v2",
    "mlx-community/whisper-large-v3",
    "mlx-community/whisper-large-v3-turbo",
]


class WhisperConverter:
    """
    Whisper STT converter using mlx-whisper.

    Wraps mlx-whisper to provide speech-to-text transcription
    with a consistent API for our converter tool.

    Example:
        converter = WhisperConverter()
        result = converter.transcribe("audio.mp3")
        print(result.text)
    """

    DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

    def __init__(self) -> None:
        """Initialize converter."""
        if not MLX_WHISPER_AVAILABLE:
            raise ImportError(
                "mlx-whisper is required. Install with: pip install mlx-whisper",
            )

    def transcribe(
        self,
        audio_path: str,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        verbose: bool = False,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            model: Model to use (default: whisper-large-v3-turbo)
            language: Language code (auto-detected if not specified)
            verbose: Print progress during transcription
            word_timestamps: Include word-level timestamps
            initial_prompt: Optional prompt to condition the model
            temperature: Sampling temperature (0.0 for greedy)

        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            return TranscriptionResult(
                success=False, text="", error=f"Audio file not found: {audio_path}",
            )

        try:
            # Build decode options
            decode_options = {}
            if language is not None:
                decode_options["language"] = language

            # For WAV/FLAC files, try loading with soundfile first (avoids ffmpeg dep)
            suffix = audio_file.suffix.lower()
            use_array_fallback = False

            if suffix in [".wav", ".flac", ".ogg"] and SOUNDFILE_AVAILABLE:
                try:
                    audio_data, sample_rate = sf.read(str(audio_file), dtype="float32")
                    # Convert stereo to mono if needed
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    use_array_fallback = True
                except Exception:
                    # Fallback to mlx_whisper file loading
                    use_array_fallback = False

            # Transcribe using array or file
            start_time = time.perf_counter()

            if use_array_fallback:
                # Use array-based transcription
                return self.transcribe_array(
                    audio_data,
                    sample_rate=sample_rate,
                    model=model,
                    language=language,
                    verbose=verbose,
                )
            # Use mlx_whisper file-based transcription (may require ffmpeg)
            result = mlx_whisper.transcribe(
                str(audio_file),
                path_or_hf_repo=model,
                verbose=verbose,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                temperature=temperature,
                **decode_options,
            )
            transcription_time = time.perf_counter() - start_time

            # Extract results
            text = result.get("text", "").strip()
            detected_language = result.get("language", None)
            segments = result.get("segments", [])

            # Calculate audio duration from segments
            audio_duration = 0.0
            if segments:
                audio_duration = max(s.get("end", 0) for s in segments)

            # Calculate real-time factor
            rtf = transcription_time / audio_duration if audio_duration > 0 else 0.0

            return TranscriptionResult(
                success=True,
                text=text,
                language=detected_language,
                segments=segments,
                duration_seconds=audio_duration,
                transcription_time_seconds=transcription_time,
                real_time_factor=rtf,
            )

        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message for ffmpeg issues
            if "ffmpeg" in error_msg.lower() or "No such file" in error_msg:
                suffix = audio_file.suffix.lower()
                if suffix not in [".wav", ".flac", ".ogg"]:
                    error_msg = (
                        f"Failed to load audio file: {error_msg}. "
                        f"For {suffix} files, ffmpeg is required. "
                        f"Install ffmpeg or convert to WAV format."
                    )
                elif not SOUNDFILE_AVAILABLE:
                    error_msg = (
                        f"Failed to load audio file: {error_msg}. "
                        f"Install soundfile for WAV support: pip install soundfile"
                    )
            return TranscriptionResult(success=False, text="", error=error_msg)

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio from numpy array.

        Args:
            audio: Audio waveform as numpy array (mono, float32)
            sample_rate: Sample rate of audio (default 16kHz)
            model: Model to use
            language: Language code (auto-detected if not specified)
            verbose: Print progress during transcription

        Returns:
            TranscriptionResult with transcribed text
        """
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling - for production use librosa or scipy
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio,
                )

            # Convert to MLX array
            audio_mx = mx.array(audio.astype(np.float32))

            decode_options = {}
            if language is not None:
                decode_options["language"] = language

            start_time = time.perf_counter()
            result = mlx_whisper.transcribe(
                audio_mx, path_or_hf_repo=model, verbose=verbose, **decode_options,
            )
            transcription_time = time.perf_counter() - start_time

            text = result.get("text", "").strip()
            detected_language = result.get("language", None)
            segments = result.get("segments", [])

            audio_duration = len(audio) / 16000
            rtf = transcription_time / audio_duration if audio_duration > 0 else 0.0

            return TranscriptionResult(
                success=True,
                text=text,
                language=detected_language,
                segments=segments,
                duration_seconds=audio_duration,
                transcription_time_seconds=transcription_time,
                real_time_factor=rtf,
            )

        except Exception as e:
            return TranscriptionResult(success=False, text="", error=str(e))

    def transcribe_batch(
        self,
        audio_paths: list[str],
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        verbose: bool = False,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Processes files sequentially - Whisper's autoregressive decoder
        is the bottleneck, similar to other encoder-decoder models.

        Performance: ~0.05x RTF (20x faster than real-time) per file
        with large-v3-turbo model.

        Args:
            audio_paths: List of paths to audio files
            model: Model to use (default: whisper-large-v3-turbo)
            language: Language code (auto-detected if not specified)
            verbose: Print progress during transcription

        Returns:
            List of TranscriptionResult objects

        Example:
            >>> converter = WhisperConverter()
            >>> results = converter.transcribe_batch(["audio1.wav", "audio2.wav"])
            >>> for r in results:
            ...     print(r.text)
        """
        if not audio_paths:
            return []

        return [
            self.transcribe(
                audio_path=path,
                model=model,
                language=language,
                verbose=verbose,
            )
            for path in audio_paths
        ]

    def benchmark(
        self, audio_path: str, model: str = DEFAULT_MODEL, runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark transcription performance.

        Args:
            audio_path: Path to audio file
            model: Model to benchmark
            runs: Number of runs for averaging

        Returns:
            BenchmarkResult with performance metrics
        """
        times = []
        audio_duration = 0.0
        word_count = 0

        for _i in range(runs):
            result = self.transcribe(audio_path, model=model, verbose=False)
            if not result.success:
                raise RuntimeError(f"Transcription failed: {result.error}")

            times.append(result.transcription_time_seconds)
            audio_duration = result.duration_seconds
            word_count = len(result.text.split())

        avg_time = sum(times) / len(times)
        rtf = avg_time / audio_duration if audio_duration > 0 else 0.0
        wps = word_count / avg_time if avg_time > 0 else 0.0

        return BenchmarkResult(
            audio_duration_seconds=audio_duration,
            transcription_time_seconds=avg_time,
            real_time_factor=rtf,
            words_per_second=wps,
            model=model,
        )

    def format_output(self, result: TranscriptionResult, format: str = "text") -> str:
        """
        Format transcription result.

        Args:
            result: TranscriptionResult to format
            format: Output format (text, json, srt, vtt)

        Returns:
            Formatted string
        """
        if format == "text":
            return result.text

        if format == "json":
            import json

            return json.dumps(
                {
                    "text": result.text,
                    "language": result.language,
                    "duration": result.duration_seconds,
                    "segments": result.segments,
                },
                indent=2,
            )

        if format == "srt":
            return self._format_srt(result.segments or [])

        if format == "vtt":
            return self._format_vtt(result.segments or [])

        return result.text

    def _format_srt(self, segments: list[dict[str, Any]]) -> str:
        """Format segments as SRT subtitles."""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_timestamp_srt(seg.get("start", 0))
            end = self._format_timestamp_srt(seg.get("end", 0))
            text = seg.get("text", "").strip()
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _format_vtt(self, segments: list[dict[str, Any]]) -> str:
        """Format segments as WebVTT subtitles."""
        lines = ["WEBVTT", ""]
        for seg in segments:
            start = self._format_timestamp_vtt(seg.get("start", 0))
            end = self._format_timestamp_vtt(seg.get("end", 0))
            text = seg.get("text", "").strip()
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def list_models() -> list[str]:
        """
        List available Whisper models.

        Returns:
            List of HuggingFace model paths
        """
        return AVAILABLE_MODELS.copy()

    @staticmethod
    def get_model_info(model: str) -> dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Model path

        Returns:
            Dict with model information
        """
        # Model size estimates
        model_sizes = {
            "tiny": "39M params, ~150MB",
            "base": "74M params, ~290MB",
            "small": "244M params, ~970MB",
            "medium": "769M params, ~3GB",
            "large-v2": "1550M params, ~6GB",
            "large-v3": "1550M params, ~6GB",
            "large-v3-turbo": "809M params, ~3.1GB",
        }

        # Determine model variant from path (check more specific keys first)
        variant = "unknown"
        # Order by length descending to match more specific variants first
        for key in sorted(model_sizes.keys(), key=len, reverse=True):
            if key in model:
                variant = key
                break

        return {
            "path": model,
            "variant": variant,
            "size": model_sizes.get(variant, "unknown"),
            "multilingual": not model.endswith(".en"),
        }
