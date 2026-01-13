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
DashVoice Multi-Speaker Pipeline

End-to-end pipeline for processing audio with multiple speakers:
1. Voice Activity Detection (VAD) - detect speech segments
2. Echo Cancellation (Layer 1) - remove our own TTS output
3. Voice Fingerprinting (Layer 2) - detect DashVoice-generated audio
4. Speaker Identification - identify known speakers
5. Source Separation - separate overlapping speakers (if needed)

Performance Targets (from DASHVOICE_MASTER_PLAN):
- Echo Cancellation: <5ms latency
- Voice Fingerprint: <100ms for 1s audio
- VAD: Real-time capable
- Overall Pipeline: <200ms for 1s audio segment

Usage:
    from tools.dashvoice.pipeline import DashVoicePipeline

    pipeline = DashVoicePipeline()
    result = pipeline.process(audio, sample_rate=16000)

    for segment in result.segments:
        print(f"{segment.speaker}: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
        if segment.is_dashvoice:
            print("  (DashVoice generated)")
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpeechSegment:
    """A segment of speech with metadata."""

    start_time: float  # seconds
    end_time: float  # seconds
    audio: np.ndarray
    speaker: str | None = None  # speaker ID or name
    is_dashvoice: bool = False  # True if detected as DashVoice-generated
    dashvoice_voice: str | None = None  # matched voice name if is_dashvoice
    confidence: float = 0.0  # confidence in speaker ID
    transcription: str | None = None  # Whisper transcription
    language: str | None = None  # detected language


@dataclass
class PipelineResult:
    """Result of processing audio through the pipeline."""

    segments: list[SpeechSegment]
    processing_time_ms: float
    sample_rate: int
    echo_cancelled: bool = False
    echo_reduction_db: float | None = None
    num_speakers_detected: int = 0
    had_overlapping_speech: bool = False
    noise_reduced: bool = False


@dataclass
class VADResult:
    """Result of Voice Activity Detection."""

    segments: list[tuple[float, float]]  # List of (start_time, end_time)
    speech_probability: float  # Overall speech probability


class VADProcessor:
    """Voice Activity Detection using Silero VAD."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 100,
        window_size_samples: int = 512,
        sample_rate: int = 16000,
    ):
        """Initialize VAD processor.

        Args:
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration
            window_size_samples: VAD window size (512 for 16kHz)
            sample_rate: Audio sample rate (Silero VAD requires 16kHz)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.sample_rate = sample_rate
        self._model = None
        self._utils = None

    def _load_model(self):
        """Load Silero VAD model (lazy loading)."""
        if self._model is not None:
            return

        try:
            import torch

            # Load model from torch hub
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            print("Loaded Silero VAD model")
        except Exception as e:
            print(f"Warning: Could not load Silero VAD: {e}")
            print("Using fallback energy-based VAD")
            self._model = "fallback"

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> VADResult:
        """Detect speech segments in audio.

        Args:
            audio: Audio array (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            VADResult with speech segments
        """
        self._load_model()

        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # Use fallback if model not available
        if self._model == "fallback":
            return self._energy_vad(audio)

        # Use Silero VAD
        try:
            import torch

            # Get speech timestamps
            get_speech_timestamps = self._utils[0]
            audio_tensor = torch.tensor(audio)

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                sampling_rate=self.sample_rate,
            )

            # Convert to time in seconds
            segments = []
            for ts in speech_timestamps:
                start_time = ts["start"] / self.sample_rate
                end_time = ts["end"] / self.sample_rate
                segments.append((start_time, end_time))

            # Calculate overall speech probability
            total_speech = sum(end - start for start, end in segments)
            total_duration = len(audio) / self.sample_rate
            speech_prob = total_speech / max(total_duration, 0.001)

            return VADResult(segments=segments, speech_probability=speech_prob)

        except Exception as e:
            print(f"VAD error, using fallback: {e}")
            return self._energy_vad(audio)

    def _energy_vad(self, audio: np.ndarray) -> VADResult:
        """Fallback energy-based VAD."""
        # Simple energy-based detection
        frame_size = int(0.02 * self.sample_rate)  # 20ms frames
        hop_size = int(0.01 * self.sample_rate)  # 10ms hop

        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            energy = np.sum(frame**2) / frame_size
            energies.append(energy)

        energies = np.array(energies)
        if len(energies) == 0:
            return VADResult(segments=[], speech_probability=0.0)

        # Threshold based on median energy
        threshold = np.median(energies) * 2
        speech_frames = energies > threshold

        # Group into segments
        segments = []
        in_speech = False
        start_frame = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # Segment end
                start_time = start_frame * hop_size / self.sample_rate
                end_time = i * hop_size / self.sample_rate
                if (end_time - start_time) * 1000 >= self.min_speech_duration_ms:
                    segments.append((start_time, end_time))
                in_speech = False

        # Handle segment at end
        if in_speech:
            start_time = start_frame * hop_size / self.sample_rate
            end_time = len(energies) * hop_size / self.sample_rate
            if (end_time - start_time) * 1000 >= self.min_speech_duration_ms:
                segments.append((start_time, end_time))

        speech_prob = np.mean(speech_frames) if len(speech_frames) > 0 else 0.0
        return VADResult(segments=segments, speech_probability=float(speech_prob))

    def _resample(
        self, audio: np.ndarray, src_rate: int, dst_rate: int,
    ) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / src_rate
        new_length = int(duration * dst_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class SpeakerDiarizer:
    """Speaker diarization using Resemblyzer embeddings + clustering."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_segment_duration: float = 0.5,
    ):
        """Initialize speaker diarizer.

        Args:
            similarity_threshold: Cosine similarity threshold for same speaker
            min_segment_duration: Minimum segment duration in seconds
        """
        self.similarity_threshold = similarity_threshold
        self.min_segment_duration = min_segment_duration
        self._encoder = None
        self._known_speakers: dict[str, np.ndarray] = {}  # name -> embedding

    def _load_encoder(self):
        """Load Resemblyzer voice encoder (lazy loading)."""
        if self._encoder is not None:
            return

        try:
            from resemblyzer import VoiceEncoder

            self._encoder = VoiceEncoder()
            print("Loaded Resemblyzer speaker encoder")
        except ImportError:
            print("Warning: resemblyzer not installed. Speaker diarization disabled.")
            self._encoder = "fallback"
        except Exception as e:
            print(f"Warning: Could not load speaker encoder: {e}")
            self._encoder = "fallback"

    def register_speaker(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ):
        """Register a known speaker for identification.

        Args:
            name: Speaker name/ID
            audio: Reference audio sample
            sample_rate: Audio sample rate
        """
        self._load_encoder()
        if self._encoder == "fallback":
            return

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        try:
            from resemblyzer import preprocess_wav

            wav = preprocess_wav(audio, source_sr=16000)
            embedding = self._encoder.embed_utterance(wav)
            self._known_speakers[name] = embedding
            print(f"Registered speaker: {name}")
        except Exception as e:
            print(f"Could not register speaker {name}: {e}")

    def get_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Get speaker embedding for audio segment.

        Args:
            audio: Audio segment
            sample_rate: Audio sample rate

        Returns:
            Speaker embedding (256-dim) or None if failed
        """
        self._load_encoder()
        if self._encoder == "fallback":
            return None

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        try:
            from resemblyzer import preprocess_wav

            wav = preprocess_wav(audio, source_sr=16000)
            if len(wav) < 1600:  # Less than 0.1s
                return None
            return self._encoder.embed_utterance(wav)
        except Exception as e:
            print(f"Could not get embedding: {e}")
            return None

    def identify_speaker(
        self,
        embedding: np.ndarray,
    ) -> tuple[str | None, float]:
        """Identify speaker from embedding against known speakers.

        Args:
            embedding: Speaker embedding

        Returns:
            Tuple of (speaker_name, confidence) or (None, 0.0) if unknown
        """
        if not self._known_speakers:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for name, known_emb in self._known_speakers.items():
            # Cosine similarity
            similarity = np.dot(embedding, known_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-10
            )
            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_score >= self.similarity_threshold:
            return best_match, float(best_score)
        return None, 0.0

    def diarize_segments(
        self,
        segments: list["SpeechSegment"],
        sample_rate: int = 16000,
    ) -> list["SpeechSegment"]:
        """Assign speaker IDs to segments using clustering.

        Args:
            segments: List of speech segments
            sample_rate: Audio sample rate

        Returns:
            Updated segments with speaker IDs
        """
        self._load_encoder()
        if self._encoder == "fallback" or len(segments) == 0:
            return segments

        # Get embeddings for each segment
        embeddings = []
        valid_indices = []

        for i, seg in enumerate(segments):
            duration = seg.end_time - seg.start_time
            if duration < self.min_segment_duration:
                continue

            emb = self.get_embedding(seg.audio, sample_rate)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if len(embeddings) == 0:
            return segments

        embeddings_array = np.array(embeddings)

        # First, try to identify known speakers
        known_assignments = {}
        for i, emb in enumerate(embeddings):
            name, confidence = self.identify_speaker(emb)
            if name:
                known_assignments[valid_indices[i]] = (name, confidence)

        # For remaining segments, use agglomerative clustering
        unknown_indices = [
            i for i in range(len(embeddings)) if valid_indices[i] not in known_assignments
        ]

        if len(unknown_indices) > 1:
            try:
                from scipy.cluster.hierarchy import fcluster, linkage

                unknown_embeddings = embeddings_array[unknown_indices]

                # Use cosine distance
                linkage_matrix = linkage(
                    unknown_embeddings, method="average", metric="cosine",
                )

                # Cluster at threshold distance
                threshold = 1 - self.similarity_threshold
                labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

                # Assign speaker IDs
                for i, label in zip(unknown_indices, labels, strict=False):
                    orig_idx = valid_indices[i]
                    segments[orig_idx].speaker = f"Speaker_{label}"
                    segments[orig_idx].confidence = 0.5  # Unknown confidence

            except ImportError:
                # Fallback: assign sequential IDs
                for i, idx in enumerate(unknown_indices):
                    orig_idx = valid_indices[idx]
                    segments[orig_idx].speaker = f"Speaker_{i + 1}"
                    segments[orig_idx].confidence = 0.5

        elif len(unknown_indices) == 1:
            # Single unknown segment
            orig_idx = valid_indices[unknown_indices[0]]
            segments[orig_idx].speaker = "Speaker_1"
            segments[orig_idx].confidence = 0.5

        # Apply known speaker assignments
        for idx, (name, confidence) in known_assignments.items():
            segments[idx].speaker = name
            segments[idx].confidence = confidence

        return segments

    def _resample(
        self, audio: np.ndarray, src_rate: int, dst_rate: int,
    ) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio

        duration = len(audio) / src_rate
        new_length = int(duration * dst_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class NoiseReducer:
    """Noise reduction using DeepFilterNet3.

    DeepFilterNet is a state-of-the-art deep learning noise suppression model.
    It achieves >45dB noise reduction while preserving speech quality.

    Requirements:
    - DeepFilterNet installed: pip install deepfilternet
    - torchaudio 2.x compatibility patch applied (see scripts/patch_deepfilternet.py)
    """

    def __init__(self):
        """Initialize noise reducer."""
        self._model = None
        self._df_state = None
        self._enhance = None

    def _load_model(self):
        """Load DeepFilterNet model (lazy loading)."""
        if self._model is not None:
            return True

        try:
            from df import enhance, init_df

            self._model, self._df_state, _ = init_df()
            self._enhance = enhance
            print("Loaded DeepFilterNet3 noise reduction model")
            return True
        except ImportError:
            print("Warning: DeepFilterNet not installed. Noise reduction disabled.")
            print("Install with: pip install deepfilternet")
            self._model = "fallback"
            return False
        except Exception as e:
            print(f"Warning: Could not load DeepFilterNet: {e}")
            self._model = "fallback"
            return False

    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Reduce noise in audio using DeepFilterNet.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            Noise-reduced audio
        """
        if not self._load_model():
            return audio

        if self._model == "fallback":
            return audio

        try:
            import torch

            # DeepFilterNet expects 48kHz audio
            target_sr = self._df_state.sr()

            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Resample to 48kHz if needed
            if sample_rate != target_sr:
                import torchaudio.functional as F

                audio_tensor = F.resample(audio_tensor, sample_rate, target_sr)

            # Run enhancement
            enhanced = self._enhance(self._model, self._df_state, audio_tensor)

            # Resample back to original rate if needed
            if sample_rate != target_sr:
                enhanced = F.resample(enhanced, target_sr, sample_rate)

            # Convert back to numpy
            result = enhanced.squeeze().numpy()
            return result.astype(np.float32)

        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio


class SourceSeparator:
    """Source separation for overlapping speech.

    Separates multiple speakers from mixed audio. Uses SpeechBrain's SepFormer
    when available, falls back to simple spectral masking otherwise.

    Note: SpeechBrain requires torchaudio compatibility which may need fixing.
    When torchaudio issues are resolved, SepFormer will be used automatically.
    """

    def __init__(self):
        """Initialize source separator."""
        self._model = None
        self._use_sepformer = False

    def _load_model(self):
        """Load separation model (lazy loading)."""
        if self._model is not None:
            return

        # Try SpeechBrain SepFormer first
        try:
            from speechbrain.inference.separation import SepformerSeparation

            self._model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir="models/sepformer",
            )
            self._use_sepformer = True
            print("Loaded SepFormer source separation model")
            return
        except Exception as e:
            print(f"SepFormer not available ({e}), using spectral fallback")

        # Fallback: use spectral masking
        self._model = "spectral_fallback"
        self._use_sepformer = False

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: int = 2,
    ) -> list[np.ndarray]:
        """Separate mixed audio into individual sources.

        Args:
            audio: Mixed audio (float32, mono)
            sample_rate: Audio sample rate
            num_speakers: Expected number of speakers

        Returns:
            List of separated audio arrays (one per speaker)
        """
        self._load_model()

        if self._use_sepformer:
            return self._separate_sepformer(audio, sample_rate, num_speakers)
        return self._separate_spectral(audio, sample_rate, num_speakers)

    def _separate_sepformer(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int,
    ) -> list[np.ndarray]:
        """Separate using SpeechBrain SepFormer."""
        import tempfile
        from pathlib import Path

        import soundfile as sf

        # SepFormer expects 8kHz audio
        if sample_rate != 8000:
            # Simple resampling
            duration = len(audio) / sample_rate
            new_length = int(duration * 8000)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 8000)
            temp_path = f.name

        try:
            # Run separation
            est_sources = self._model.separate_file(path=temp_path)
            sources = []
            for i in range(est_sources.shape[2]):
                source = est_sources[:, :, i].squeeze().numpy()
                # Resample back to original rate
                if sample_rate != 8000:
                    duration = len(source) / 8000
                    new_length = int(duration * sample_rate)
                    indices = np.linspace(0, len(source) - 1, new_length)
                    source = np.interp(indices, np.arange(len(source)), source)
                sources.append(source.astype(np.float32))
            return sources[:num_speakers]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _separate_spectral(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int,
    ) -> list[np.ndarray]:
        """Simple spectral masking separation (fallback).

        This is a basic approach that won't work well for overlapping speech,
        but provides the interface for testing. Use SepFormer for real separation.
        """
        # Simple approach: return the original audio for first speaker
        # and attenuated versions for others (placeholder)
        return [audio] + [np.zeros_like(audio) for _ in range(1, num_speakers)]


class WhisperSTT:
    """Speech-to-Text using MLX Whisper."""

    def __init__(
        self,
        model: str = "mlx-community/whisper-large-v3-turbo",
    ):
        """Initialize Whisper STT.

        Args:
            model: HuggingFace model path for MLX Whisper
        """
        self.model_path = model
        self._loaded = False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, str | None]:
        """Transcribe audio to text.

        Args:
            audio: Audio array (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            Tuple of (transcription, detected_language)
        """
        try:
            import tempfile

            import mlx_whisper
            import soundfile as sf
        except ImportError:
            return "", None

        # Save to temp file (mlx_whisper expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            temp_path = f.name

        try:
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=self.model_path,
            )
            text = result.get("text", "").strip()
            language = result.get("language", None)
            self._loaded = True
            return text, language
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return "", None
        finally:
            Path(temp_path).unlink(missing_ok=True)


class DashVoicePipeline:
    """
    Complete pipeline for multi-speaker audio processing.

    Integrates:
    - VAD (Voice Activity Detection)
    - Noise Reduction (DeepFilterNet3)
    - Echo Cancellation (Layer 1)
    - Voice Fingerprinting (Layer 2)
    - Speaker Identification
    """

    def __init__(
        self,
        enable_echo_cancel: bool = True,
        enable_voice_fingerprint: bool = True,
        enable_vad: bool = True,
        enable_stt: bool = False,  # Disabled by default (slower)
        enable_diarization: bool = True,
        enable_source_separation: bool = False,  # Disabled by default (slow, needs SpeechBrain)
        enable_noise_reduction: bool = False,  # Disabled by default (requires DeepFilterNet)
        vad_threshold: float = 0.5,
        diarization_threshold: float = 0.75,
    ):
        """Initialize pipeline.

        Args:
            enable_echo_cancel: Enable echo cancellation
            enable_voice_fingerprint: Enable DashVoice detection
            enable_vad: Enable VAD segmentation
            enable_stt: Enable Whisper transcription (slower)
            enable_diarization: Enable speaker diarization
            enable_source_separation: Enable source separation for overlapping speech
            enable_noise_reduction: Enable DeepFilterNet3 noise reduction
            vad_threshold: VAD speech probability threshold
            diarization_threshold: Similarity threshold for speaker clustering
        """
        self.enable_echo_cancel = enable_echo_cancel
        self.enable_voice_fingerprint = enable_voice_fingerprint
        self.enable_vad = enable_vad
        self.enable_stt = enable_stt
        self.enable_diarization = enable_diarization
        self.enable_source_separation = enable_source_separation
        self.enable_noise_reduction = enable_noise_reduction

        # Lazy-load components
        self._echo_canceller = None
        self._voice_db = None
        self._stt = None
        self._diarizer = None
        self._separator = None
        self._noise_reducer = None
        self._vad = VADProcessor(threshold=vad_threshold)
        self._diarization_threshold = diarization_threshold

    def _get_stt(self):
        """Get or create Whisper STT."""
        if self._stt is None:
            self._stt = WhisperSTT()
        return self._stt

    def _get_diarizer(self):
        """Get or create speaker diarizer."""
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(
                similarity_threshold=self._diarization_threshold,
            )
        return self._diarizer

    def _get_separator(self):
        """Get or create source separator."""
        if self._separator is None:
            self._separator = SourceSeparator()
        return self._separator

    def _get_noise_reducer(self):
        """Get or create noise reducer."""
        if self._noise_reducer is None:
            self._noise_reducer = NoiseReducer()
        return self._noise_reducer

    def register_speaker(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ):
        """Register a known speaker for identification.

        Args:
            name: Speaker name/ID
            audio: Reference audio sample
            sample_rate: Audio sample rate
        """
        diarizer = self._get_diarizer()
        diarizer.register_speaker(name, audio, sample_rate)

    def separate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: int = 2,
    ) -> list[np.ndarray]:
        """Separate overlapping speech into individual speaker tracks.

        Args:
            audio: Mixed audio with overlapping speech
            sample_rate: Audio sample rate
            num_speakers: Expected number of speakers

        Returns:
            List of separated audio arrays (one per speaker)
        """
        separator = self._get_separator()
        return separator.separate(audio, sample_rate, num_speakers)

    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Reduce noise in audio using DeepFilterNet3.

        Standalone method for noise reduction without full pipeline.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            Noise-reduced audio
        """
        noise_reducer = self._get_noise_reducer()
        return noise_reducer.reduce_noise(audio, sample_rate)

    def _get_echo_canceller(self):
        """Get or create echo canceller."""
        if self._echo_canceller is None:
            from tools.dashvoice.echo_cancel import EchoCanceller

            self._echo_canceller = EchoCanceller()
        return self._echo_canceller

    def _get_voice_db(self):
        """Get or create voice database."""
        if self._voice_db is None:
            from tools.dashvoice.voice_database import VoiceDatabase

            self._voice_db = VoiceDatabase()
        return self._voice_db

    def add_tts_reference(self, audio: np.ndarray, sample_rate: int = 24000):
        """Add TTS output to reference buffer for echo cancellation.

        Call this whenever TTS generates audio.

        Args:
            audio: Generated TTS audio
            sample_rate: Sample rate of audio
        """
        if self.enable_echo_cancel:
            canceller = self._get_echo_canceller()
            canceller.add_reference(audio)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> PipelineResult:
        """Process audio through the complete pipeline.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            PipelineResult with processed segments
        """
        start_time = time.perf_counter()

        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32)
        max_val = np.abs(audio).max()
        if max_val > 100.0:
            # Likely int16 format (range -32768 to 32767)
            audio = audio / 32768.0
        elif max_val > 1.0:
            # Float audio slightly above 1.0, normalize to [-1, 1]
            audio = audio / max_val

        # Step 1: VAD to find speech segments
        segments = []
        if self.enable_vad:
            vad_result = self._vad.detect(audio, sample_rate)
            vad_segments = vad_result.segments
        else:
            # Treat whole audio as one segment
            vad_segments = [(0.0, len(audio) / sample_rate)]

        # Step 2: Process each segment
        echo_cancelled = False
        echo_reduction_db = None

        for start, end in vad_segments:
            # Extract segment audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < 100:
                continue

            # Echo cancellation
            if self.enable_echo_cancel:
                canceller = self._get_echo_canceller()
                result = canceller.process(segment_audio)
                segment_audio = result.cleaned_audio
                if result.had_echo:
                    echo_cancelled = True
                    echo_reduction_db = result.echo_reduction_db

            # Noise reduction using DeepFilterNet3
            if self.enable_noise_reduction:
                noise_reducer = self._get_noise_reducer()
                segment_audio = noise_reducer.reduce_noise(segment_audio, sample_rate)

            # Voice fingerprinting - check if DashVoice
            is_dashvoice = False
            dashvoice_voice = None

            if self.enable_voice_fingerprint:
                voice_db = self._get_voice_db()
                is_dashvoice, voice_name, confidence = voice_db.is_dashvoice(
                    segment_audio, sample_rate,
                )
                if is_dashvoice:
                    dashvoice_voice = voice_name

            # Speech-to-text transcription
            transcription = None
            language = None

            if self.enable_stt:
                stt = self._get_stt()
                transcription, language = stt.transcribe(segment_audio, sample_rate)

            segments.append(
                SpeechSegment(
                    start_time=start,
                    end_time=end,
                    audio=segment_audio,
                    speaker=None,  # Will be set by diarization
                    is_dashvoice=is_dashvoice,
                    dashvoice_voice=dashvoice_voice,
                    confidence=confidence if is_dashvoice else 0.0,
                    transcription=transcription,
                    language=language,
                ),
            )

        # Step 3: Speaker diarization (assign speaker IDs to segments)
        if self.enable_diarization and len(segments) > 0:
            diarizer = self._get_diarizer()
            segments = diarizer.diarize_segments(segments, sample_rate)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            segments=segments,
            processing_time_ms=processing_time_ms,
            sample_rate=sample_rate,
            echo_cancelled=echo_cancelled,
            echo_reduction_db=echo_reduction_db,
            num_speakers_detected=len({s.speaker for s in segments if s.speaker}),
            had_overlapping_speech=False,  # Not yet implemented
            noise_reduced=self.enable_noise_reduction,
        )

    def process_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> SpeechSegment | None:
        """Process audio chunk in streaming mode.

        For real-time processing, call this with small audio chunks.

        Args:
            audio_chunk: Audio chunk (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            SpeechSegment if speech detected, None otherwise
        """
        # Simplified streaming: just check if chunk contains speech
        vad_result = self._vad.detect(audio_chunk, sample_rate)

        if not vad_result.segments:
            return None

        # Process the chunk
        result = self.process(audio_chunk, sample_rate)
        if result.segments:
            return result.segments[0]
        return None


def run_pipeline_benchmark():
    """Run benchmark of the pipeline."""
    print("DashVoice Pipeline Benchmark")
    print("=" * 60)

    # Create test audio
    sample_rate = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))

    # Simulate speech-like signal
    audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    audio = (audio * 0.5).astype(np.float32)

    # Create pipeline
    pipeline = DashVoicePipeline()

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        pipeline.process(audio, sample_rate)

    # Benchmark
    print("Running benchmark (10 iterations)...")
    times = []
    for _ in range(10):
        result = pipeline.process(audio, sample_rate)
        times.append(result.processing_time_ms)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print("\nResults:")
    print(f"  Average processing time: {avg_time:.2f}ms +/- {std_time:.2f}ms")
    print(f"  Audio duration: {duration_s * 1000:.0f}ms")
    print(f"  Real-time factor: {(duration_s * 1000) / avg_time:.1f}x")
    print(f"  Segments found: {len(result.segments)}")
    print(f"  Echo cancelled: {result.echo_cancelled}")

    # Check performance target
    target_ms = 200.0
    if avg_time < target_ms:
        print(f"\n  [PASS] Meets target (<{target_ms}ms)")
    else:
        print(f"\n  [FAIL] Exceeds target ({target_ms}ms)")


if __name__ == "__main__":
    run_pipeline_benchmark()
