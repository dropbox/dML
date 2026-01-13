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
Hybrid WhisperMLX: CoreML encoder + MLX decoder.

This module provides a hybrid implementation that uses:
- CoreML encoder (runs on GPU or ANE via CoreML)
- MLX decoder (runs on GPU via MLX)

Performance on M4 Max (30s audio, large-v3):
- CoreML encoder (GPU): ~200ms
- MLX encoder (GPU): ~210ms
- Both are similar; main benefit is flexibility

The hybrid approach allows offloading encoder computation to CoreML
while keeping the optimized MLX decoder. This is useful when:
1. ANE is faster on certain hardware (M2/M3)
2. GPU is busy with other work
3. You want to leverage pre-converted WhisperKit models
"""

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..whisper_mlx.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
from ..whisper_mlx.model import WhisperMLX
from .encoder import CoreMLEncoder

logger = logging.getLogger(__name__)


class HybridWhisperMLX:
    """
    Hybrid Whisper model: CoreML encoder + MLX decoder.

    This provides the same interface as WhisperMLX but uses CoreML
    for the encoder, which can run on ANE or GPU via CoreML.

    Example:
        model = HybridWhisperMLX.from_pretrained("large-v3")
        result = model.transcribe("audio.wav")
    """

    def __init__(
        self,
        mlx_model: WhisperMLX,
        coreml_encoder: CoreMLEncoder,
    ):
        """
        Initialize hybrid model.

        Args:
            mlx_model: WhisperMLX model (for decoder)
            coreml_encoder: CoreML encoder
        """
        self.mlx_model = mlx_model
        self.coreml_encoder = coreml_encoder

        # Proxy attributes from MLX model
        self.config = mlx_model.config
        self.decoder = mlx_model.decoder
        self.encoder = mlx_model.encoder  # Keep MLX encoder for fallback

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "large-v3",
        *,
        mlx_model_path: str | None = None,
        coreml_model_dir: Path | None = None,
        compute_units: str = "CPU_AND_GPU",
        auto_download: bool = True,
    ) -> "HybridWhisperMLX":
        """
        Load a pretrained hybrid model.

        Args:
            model_name: Model name (e.g., "large-v3", "turbo")
            mlx_model_path: Path to MLX model (uses HuggingFace default if None)
            coreml_model_dir: Directory containing CoreML models
            compute_units: CoreML compute units (CPU_AND_GPU recommended for M4)
            auto_download: Automatically download models if not found

        Returns:
            HybridWhisperMLX instance
        """
        # Load MLX model
        if mlx_model_path is None:
            # Map common names to HuggingFace paths
            mlx_paths = {
                "large-v3": "mlx-community/whisper-large-v3-mlx",
                "turbo": "mlx-community/whisper-large-v3-turbo",
                "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            }
            mlx_model_path = mlx_paths.get(model_name, f"mlx-community/whisper-{model_name}-mlx")

        logger.info("Loading MLX model from %s", mlx_model_path)
        mlx_model = WhisperMLX.from_pretrained(mlx_model_path)

        # Load CoreML encoder
        logger.info("Loading CoreML encoder with compute_units=%s", compute_units)
        coreml_encoder = CoreMLEncoder.from_pretrained(
            model_name=model_name,
            models_dir=coreml_model_dir,
            compute_units=compute_units,
            auto_download=auto_download,
        )

        return cls(mlx_model, coreml_encoder)

    def encode(
        self,
        mel: np.ndarray | mx.array,
        variable_length: bool = False,
    ) -> mx.array:
        """
        Encode mel spectrogram using CoreML encoder.

        Args:
            mel: Mel spectrogram (n_frames, n_mels) or (batch, n_frames, n_mels)
            variable_length: Ignored (for API compatibility)

        Returns:
            Encoder output as MLX array (batch, seq_len, n_state)
        """
        # Convert MLX to numpy for CoreML
        if isinstance(mel, mx.array):
            mel_np = np.array(mel)
        else:
            mel_np = mel

        # Run CoreML encoder (returns numpy)
        encoder_output_np = self.coreml_encoder(mel_np)

        # Convert back to MLX
        return mx.array(encoder_output_np)


    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        max_initial_timestamp: float = 1.0,
        verbose: bool = False,
        compression_ratio_threshold: float | None = 2.4,
        logprob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
    ) -> dict:
        """
        Transcribe audio using hybrid model.

        Uses CoreML encoder then MLX decoder.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            temperature: Sampling temperature(s)
            max_initial_timestamp: Maximum timestamp at start
            verbose: Print progress
            compression_ratio_threshold: Quality threshold
            logprob_threshold: Quality threshold
            no_speech_threshold: Silence threshold

        Returns:
            Dictionary with "text", "segments", "language"
        """
        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Check for silent audio
        from ..whisper_mlx.audio import is_silent_audio
        if is_silent_audio(audio):
            if verbose:
                print("Silent audio detected")
            return {
                "text": "",
                "segments": [],
                "language": language or "en",
            }

        # Compute mel spectrogram (returns MLX array)
        mel_mx = log_mel_spectrogram(audio, n_mels=self.coreml_encoder.n_mels)
        mel = np.array(mel_mx)  # Convert to numpy for CoreML

        # Pad to 30s (3000 frames) as expected by Whisper
        n_frames = mel.shape[0]
        if n_frames < 3000:
            mel = np.pad(mel, ((0, 3000 - n_frames), (0, 0)), mode='constant')
        elif n_frames > 3000:
            mel = mel[:3000]

        # Add batch dimension
        mel = mel[np.newaxis, ...]  # (1, 3000, n_mels)

        # Encode with CoreML
        encoder_output = self.encode(mel)

        # Use MLX decoder via the model's decode method
        # We need to call the underlying transcribe but with pre-computed encoder output
        # For now, we'll use the existing transcribe method with a trick:
        # Replace the encoder temporarily

        original_encoder = self.mlx_model.encoder

        class PrecomputedEncoder:
            """Dummy encoder that returns precomputed output."""
            def __init__(self, output):
                self.output = output
            def __call__(self, mel, variable_length=False):
                return self.output

        try:
            # Replace encoder with precomputed output
            self.mlx_model.encoder = PrecomputedEncoder(encoder_output)

            # Call MLX model's transcribe
            return self.mlx_model.transcribe(
                audio,
                language=language,
                task=task,
                temperature=temperature,
                max_initial_timestamp=max_initial_timestamp,
                verbose=verbose,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
            )


        finally:
            # Restore original encoder
            self.mlx_model.encoder = original_encoder

    def benchmark(
        self,
        audio: str | np.ndarray | None = None,
        n_iterations: int = 5,
        warmup: int = 2,
    ) -> dict:
        """
        Benchmark the hybrid model.

        Args:
            audio: Audio to benchmark (uses random mel if None)
            n_iterations: Number of timed iterations
            warmup: Number of warmup iterations

        Returns:
            Dict with timing statistics
        """
        import time

        # Generate test mel if not provided
        if audio is None:
            rng = np.random.default_rng()
            mel = rng.standard_normal((1, 3000, self.coreml_encoder.n_mels)).astype(np.float32)
        else:
            if isinstance(audio, str):
                audio = load_audio(audio, sample_rate=SAMPLE_RATE)
            mel_mx = log_mel_spectrogram(audio, n_mels=self.coreml_encoder.n_mels)
            mel = np.array(mel_mx)
            mel = np.pad(mel, ((0, max(0, 3000 - mel.shape[0])), (0, 0)))[:3000]
            mel = mel[np.newaxis, ...]

        # Benchmark encoder
        for _ in range(warmup):
            _ = self.coreml_encoder(mel)

        encoder_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            encoder_output = self.coreml_encoder(mel)
            encoder_times.append(time.perf_counter() - start)

        # Convert to MLX
        encoder_output_mx = mx.array(encoder_output)

        # Benchmark decoder (just one token generation for timing)
        tokens = mx.array([[50258, 50259, 50359, 50363]])  # <|startoftranscript|> etc.

        for _ in range(warmup):
            _ = self.decoder(tokens, encoder_output_mx)
            mx.eval(_)

        decoder_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = self.decoder(tokens, encoder_output_mx)
            mx.eval(_)
            decoder_times.append(time.perf_counter() - start)

        return {
            "encoder_mean_ms": np.mean(encoder_times) * 1000,
            "encoder_std_ms": np.std(encoder_times) * 1000,
            "decoder_mean_ms": np.mean(decoder_times) * 1000,
            "decoder_std_ms": np.std(decoder_times) * 1000,
            "compute_units": self.coreml_encoder.compute_units_str,
        }


def test_hybrid():
    """Quick test of hybrid model."""
    import time

    print("Testing hybrid WhisperMLX...")

    try:
        model = HybridWhisperMLX.from_pretrained(
            "large-v3",
            compute_units="CPU_AND_GPU",
            auto_download=False,
        )
    except Exception as e:
        print(f"Failed to load: {e}")
        return False

    print("Loaded hybrid model")

    # Test with random audio
    rng = np.random.default_rng()
    audio = rng.standard_normal(SAMPLE_RATE * 5).astype(np.float32) * 0.1  # 5s noise

    print("\nTranscribing 5s test audio...")
    start = time.perf_counter()
    result = model.transcribe(audio, verbose=True)
    elapsed = time.perf_counter() - start

    print(f"Result: {result.get('text', 'N/A')}")
    print(f"Time: {elapsed*1000:.1f}ms")

    # Benchmark
    print("\nBenchmarking...")
    bench = model.benchmark()
    print(f"Encoder: {bench['encoder_mean_ms']:.1f}ms +/- {bench['encoder_std_ms']:.1f}ms")
    print(f"Decoder (1 token): {bench['decoder_mean_ms']:.1f}ms +/- {bench['decoder_std_ms']:.1f}ms")

    return True


if __name__ == "__main__":
    test_hybrid()
