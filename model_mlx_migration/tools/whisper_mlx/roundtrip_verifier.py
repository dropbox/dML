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
Production Round-Trip Verification for Streaming ASR

Uses TTS (Kokoro) to verify STT confidence by synthesizing the transcript
and comparing mel spectrograms with the original audio.

Key insight: Kokoro is 38x real-time when warmed. A 2-second phrase takes ~52ms
to synthesize. Full round-trip verification is fast enough for streaming.

Usage:
    from tools.whisper_mlx.roundtrip_verifier import RoundTripVerifier

    # Initialize once (loads and warms Kokoro)
    verifier = RoundTripVerifier()

    # Fast verification (~50-100ms for typical phrases)
    result = verifier.verify(audio_chunk, transcript)
    print(f"Confidence: {result.confidence:.3f}")  # Higher = more likely correct

Design for streaming:
    - Kokoro stays loaded and warm (no cold start)
    - Verification runs async while processing next chunk
    - ~50-100ms latency fits in 200ms chunk budget
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


@dataclass
class VerificationResult:
    """Result of round-trip verification."""
    confidence: float              # Similarity score [0, 1], higher = more likely correct
    synthesis_time_ms: float       # Time to synthesize with Kokoro
    comparison_time_ms: float      # Time to compare mels
    total_time_ms: float           # Total verification time
    input_duration_sec: float      # Duration of input audio
    synth_duration_sec: float      # Duration of synthesized audio
    alignment_cost: float          # DTW alignment cost (lower = better match)


class RoundTripVerifier:
    """
    Production round-trip verification using Kokoro TTS.

    Keeps Kokoro loaded and warm for fast inference.
    Target: <100ms verification for 2-second phrases.
    """

    def __init__(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        voice: str = "af_bella",
        sample_rate: int = 16000,
        n_mels: int = 128,
        normalize_mels: bool = True,
    ):
        """
        Initialize the verifier (loads and warms Kokoro).

        Args:
            model_id: HuggingFace model ID for Kokoro
            voice: Voice to use for synthesis
            sample_rate: Target sample rate for comparison (16kHz for Whisper)
            n_mels: Number of mel bands
            normalize_mels: Apply voice normalization to mels
        """
        if not HAS_MLX:
            raise ImportError("MLX is required")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.normalize_mels = normalize_mels

        # Load Kokoro
        print("Loading Kokoro for round-trip verification...")
        t0 = time.perf_counter()

        from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        self._converter = KokoroConverter()
        self._model, self._config, _ = self._converter.load_from_hf(model_id)
        self._voice_pack = self._converter.load_voice_pack(voice, model_id)
        self._phonemize = phonemize_text

        load_time = (time.perf_counter() - t0) * 1000
        print(f"  Loaded in {load_time:.0f}ms")

        # Warmup
        print("Warming up Kokoro...")
        t0 = time.perf_counter()
        self._warmup()
        warmup_time = (time.perf_counter() - t0) * 1000
        print(f"  Warmed up in {warmup_time:.0f}ms")
        print("Round-trip verifier ready.")

    def _warmup(self):
        """Warmup Kokoro with a short synthesis."""
        _ = self._synthesize("hello world")

    def _synthesize_long(self, text: str, max_chars: int = 250) -> np.ndarray:
        """
        Synthesize long text by splitting into chunks.

        Args:
            text: Long text to synthesize
            max_chars: Maximum characters per chunk

        Returns:
            Concatenated audio from all chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            test_chunk = " ".join(current_chunk + [word])
            if len(test_chunk) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Synthesize each chunk
        audio_chunks = []
        for chunk in chunks:
            audio = self._synthesize_chunk(chunk)
            audio_chunks.append(audio)

        # Concatenate with small silence between chunks
        silence = np.zeros(int(0.05 * self.sample_rate), dtype=np.float32)
        result = []
        for i, audio in enumerate(audio_chunks):
            result.append(audio)
            if i < len(audio_chunks) - 1:
                result.append(silence)

        return np.concatenate(result)

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesize a single chunk (must be under character limit)."""
        try:
            phoneme_str, token_ids = self._phonemize(text, language="en")
        except Exception:
            # Phonemizer failed (e.g., unknown word) - return silence
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        if len(token_ids) == 0:
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        input_ids = mx.array([token_ids], dtype=mx.int32)
        voice_embedding = self._converter.select_voice_embedding(
            self._voice_pack, len(token_ids),
        )
        if voice_embedding.ndim == 1:
            voice_embedding = voice_embedding[None, :]

        audio = self._model(input_ids, voice_embedding)
        mx.eval(audio)
        audio_np = np.array(audio).flatten().astype(np.float32)

        if self.sample_rate != 24000:
            from scipy import signal
            audio_np = signal.resample(
                audio_np,
                int(len(audio_np) * self.sample_rate / 24000),
            ).astype(np.float32)

        return audio_np

    def _synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize audio from text using Kokoro.

        Returns:
            audio: float32 array at self.sample_rate
        """
        # Split long texts to avoid phonemizer limits (~250 chars to be safe)
        # Phonemize and synthesize in chunks, then concatenate
        max_chars = 250
        if len(text) > max_chars:
            return self._synthesize_long(text, max_chars)

        # Phonemize
        try:
            phoneme_str, token_ids = self._phonemize(text, language="en")
        except Exception:
            # Phonemizer failed - return silence
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        if len(token_ids) == 0:
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        # Prepare inputs
        input_ids = mx.array([token_ids], dtype=mx.int32)

        # Get voice embedding
        voice_embedding = self._converter.select_voice_embedding(
            self._voice_pack, len(token_ids),
        )
        if voice_embedding.ndim == 1:
            voice_embedding = voice_embedding[None, :]

        # Synthesize (Kokoro outputs at 24kHz)
        audio = self._model(input_ids, voice_embedding)
        mx.eval(audio)

        # Convert to numpy
        audio_np = np.array(audio).flatten().astype(np.float32)

        # Resample from 24kHz to target rate
        if self.sample_rate != 24000:
            from scipy import signal
            audio_np = signal.resample(
                audio_np,
                int(len(audio_np) * self.sample_rate / 24000),
            ).astype(np.float32)

        return audio_np

    def _extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.

        Returns:
            mel: [n_frames, n_mels] log mel spectrogram
        """
        from tools.whisper_mlx.audio import log_mel_spectrogram

        mel = log_mel_spectrogram(audio, n_mels=self.n_mels)
        if isinstance(mel, mx.array):
            mel = np.array(mel)

        return mel

    def _normalize_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Normalize mel spectrogram for voice-agnostic comparison.

        Applies:
        - Per-utterance mean/variance normalization
        - Removes overall energy differences
        """
        # Mean normalization (removes DC offset / overall energy)
        mel = mel - mel.mean(axis=0, keepdims=True)

        # Variance normalization (normalizes dynamic range)
        std = mel.std(axis=0, keepdims=True) + 1e-8
        return mel / std


    def _compare_mels(
        self,
        mel1: np.ndarray,
        mel2: np.ndarray,
        radius: int = 50,
    ) -> tuple[float, float]:
        """
        Compare two mel spectrograms using DTW.

        Args:
            mel1: First mel spectrogram [T1, n_mels]
            mel2: Second mel spectrogram [T2, n_mels]
            radius: Sakoe-Chiba band radius for fast DTW

        Returns:
            Tuple of (similarity [0,1], alignment_cost)
        """
        # Apply normalization if enabled
        if self.normalize_mels:
            mel1 = self._normalize_mel(mel1)
            mel2 = self._normalize_mel(mel2)

        T1, D = mel1.shape
        T2 = mel2.shape[0]

        # Fast DTW with Sakoe-Chiba band constraint
        dtw = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0

        for i in range(1, T1 + 1):
            # Compute band limits
            j_center = int(i * T2 / T1)
            j_min = max(1, j_center - radius)
            j_max = min(T2, j_center + radius)

            for j in range(j_min, j_max + 1):
                # Euclidean distance
                cost = np.sqrt(np.sum((mel1[i-1] - mel2[j-1]) ** 2))
                dtw[i, j] = cost + min(
                    dtw[i-1, j],      # insertion
                    dtw[i, j-1],      # deletion
                    dtw[i-1, j-1],    # match
                )

        # Normalized alignment cost
        alignment_cost = dtw[T1, T2] / (T1 + T2)

        # Convert to similarity using exponential decay
        # Tuned based on typical mel distances:
        # - Same content: cost ~0.5-1.5
        # - Different content: cost ~2.0-4.0
        scale = 1.5
        similarity = float(np.exp(-alignment_cost / scale))

        return similarity, float(alignment_cost)

    def verify(
        self,
        audio: np.ndarray,
        transcript: str,
    ) -> VerificationResult:
        """
        Verify a transcript against audio using round-trip synthesis.

        Args:
            audio: Input audio (float32, at self.sample_rate)
            transcript: STT transcript to verify

        Returns:
            VerificationResult with confidence score and timing info
        """
        total_t0 = time.perf_counter()

        # Validate inputs
        if len(audio) == 0 or not transcript or not transcript.strip():
            return VerificationResult(
                confidence=0.0,
                synthesis_time_ms=0.0,
                comparison_time_ms=0.0,
                total_time_ms=0.0,
                input_duration_sec=0.0,
                synth_duration_sec=0.0,
                alignment_cost=float('inf'),
            )

        input_duration = len(audio) / self.sample_rate

        # Extract mel from input audio
        input_mel = self._extract_mel(audio)

        # Synthesize audio from transcript
        synth_t0 = time.perf_counter()
        synth_audio = self._synthesize(transcript.strip())
        synthesis_time_ms = (time.perf_counter() - synth_t0) * 1000

        synth_duration = len(synth_audio) / self.sample_rate

        # Extract mel from synthesized audio
        synth_mel = self._extract_mel(synth_audio)

        # Compare mels
        compare_t0 = time.perf_counter()
        confidence, alignment_cost = self._compare_mels(input_mel, synth_mel)
        comparison_time_ms = (time.perf_counter() - compare_t0) * 1000

        total_time_ms = (time.perf_counter() - total_t0) * 1000

        return VerificationResult(
            confidence=confidence,
            synthesis_time_ms=synthesis_time_ms,
            comparison_time_ms=comparison_time_ms,
            total_time_ms=total_time_ms,
            input_duration_sec=input_duration,
            synth_duration_sec=synth_duration,
            alignment_cost=alignment_cost,
        )

    def should_commit(
        self,
        audio: np.ndarray,
        transcript: str,
        threshold: float = 0.5,
    ) -> tuple[bool, VerificationResult]:
        """
        Decide whether to commit a transcript based on verification.

        Args:
            audio: Input audio
            transcript: STT transcript
            threshold: Confidence threshold for commit decision

        Returns:
            Tuple of (should_commit, VerificationResult)
        """
        result = self.verify(audio, transcript)
        return result.confidence >= threshold, result


def benchmark_verifier(max_samples: int = 20):
    """
    Benchmark the round-trip verifier on LibriSpeech samples.
    """
    from pathlib import Path

    import soundfile as sf

    print("=" * 60)
    print("Round-Trip Verifier Benchmark")
    print("=" * 60)

    # Initialize verifier
    verifier = RoundTripVerifier()

    # Load LibriSpeech samples
    librispeech_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")

    if not librispeech_dir.exists():
        print(f"LibriSpeech not found at {librispeech_dir}")
        return

    samples = []
    for trans_file in librispeech_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, transcript = parts
                    audio_path = trans_file.parent / f"{audio_id}.flac"
                    if audio_path.exists():
                        samples.append({
                            "audio_path": str(audio_path),
                            "reference": transcript.lower(),
                        })
                        if len(samples) >= max_samples:
                            break
        if len(samples) >= max_samples:
            break

    print(f"\nLoaded {len(samples)} samples")

    # Process samples
    results = []
    correct_confidences = []
    wrong_confidences = []

    for i, sample in enumerate(samples):
        print(f"\r  Processing {i+1}/{len(samples)}...", end="", flush=True)

        # Load audio
        audio, sr = sf.read(sample["audio_path"])
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        audio = audio.astype(np.float32)

        # Test with correct transcript
        correct_result = verifier.verify(audio, sample["reference"])
        correct_confidences.append(correct_result.confidence)

        # Test with wrong transcript (shuffle words)
        words = sample["reference"].split()
        if len(words) > 2:
            wrong_transcript = " ".join(words[1:] + words[:1])
        else:
            wrong_transcript = "completely different words here"

        wrong_result = verifier.verify(audio, wrong_transcript)
        wrong_confidences.append(wrong_result.confidence)

        results.append({
            "correct_conf": correct_result.confidence,
            "wrong_conf": wrong_result.confidence,
            "synthesis_ms": correct_result.synthesis_time_ms,
            "total_ms": correct_result.total_time_ms,
            "audio_duration": correct_result.input_duration_sec,
        })

    print("\n")

    # Compute metrics
    correct_confidences = np.array(correct_confidences)
    wrong_confidences = np.array(wrong_confidences)

    # Discrimination: correct should be higher than wrong
    discrimination = np.mean(correct_confidences > wrong_confidences)

    # Effect size
    mean_diff = np.mean(correct_confidences) - np.mean(wrong_confidences)
    pooled_std = np.sqrt((np.var(correct_confidences) + np.var(wrong_confidences)) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-8)

    # Timing
    synthesis_times = [r["synthesis_ms"] for r in results]
    total_times = [r["total_ms"] for r in results]
    audio_durations = [r["audio_duration"] for r in results]

    # RTF
    rtf = np.mean(synthesis_times) / (np.mean(audio_durations) * 1000)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples: {len(results)}")
    print()
    print("Discrimination:")
    print(f"  Mean confidence (correct): {np.mean(correct_confidences):.3f}")
    print(f"  Mean confidence (wrong):   {np.mean(wrong_confidences):.3f}")
    print(f"  Discrimination rate:       {discrimination:.1%}")
    print(f"  Cohen's d effect size:     {cohens_d:.2f}")
    print()
    print("Timing:")
    print(f"  Mean synthesis time:       {np.mean(synthesis_times):.1f}ms")
    print(f"  Mean total time:           {np.mean(total_times):.1f}ms")
    print(f"  Mean audio duration:       {np.mean(audio_durations):.2f}s")
    print(f"  Real-time factor:          {rtf:.3f}x")
    print("=" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    if discrimination > 0.7:
        print(f"  DISCRIMINATION: {discrimination:.0%} - GOOD, can distinguish correct vs wrong")
    elif discrimination > 0.5:
        print(f"  DISCRIMINATION: {discrimination:.0%} - WEAK, some signal")
    else:
        print(f"  DISCRIMINATION: {discrimination:.0%} - POOR, not useful")

    if np.mean(total_times) < 100:
        print(f"  SPEED: {np.mean(total_times):.0f}ms - EXCELLENT, well under 100ms target")
    elif np.mean(total_times) < 200:
        print(f"  SPEED: {np.mean(total_times):.0f}ms - GOOD, under 200ms")
    else:
        print(f"  SPEED: {np.mean(total_times):.0f}ms - SLOW, exceeds target")


if __name__ == "__main__":
    benchmark_verifier(max_samples=20)
