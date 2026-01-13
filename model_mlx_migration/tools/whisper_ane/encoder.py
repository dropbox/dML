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
CoreML Audio Encoder for WhisperMLX.

Wraps the WhisperKit CoreML encoder model for hardware acceleration.

Performance on M4 Max (30s audio, large-v3):
- CPU_AND_GPU: 200ms (FASTEST - use this on M4)
- CPU_AND_NE: 596ms (ANE is slower on M4 Max)
- CPU_ONLY: 757ms
- MLX GPU: 210ms (comparable to CoreML GPU)

The GPU path is fastest on M4 Max. ANE may be faster on M2/M3 or
when the GPU is busy with other work (e.g., decoder).

Input/Output Format:
- Input: Mel spectrogram (batch, n_frames, n_mels) or (n_frames, n_mels)
- Output: Encoder hidden states (batch, seq_len, n_state)

The interface matches tools.whisper_mlx.encoder.AudioEncoder for drop-in replacement.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Default model paths
DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "whisperkit"
AVAILABLE_MODELS = {
    "large-v3": "openai_whisper-large-v3",
    "large-v3-turbo": "openai_whisper-large-v3_turbo",
    "turbo": "openai_whisper-large-v3_turbo",
}


class CoreMLEncoder:
    """
    CoreML-based audio encoder that runs on Apple Neural Engine.

    This provides the same interface as whisper_mlx.encoder.AudioEncoder
    but uses CoreML for ANE acceleration.

    Example:
        encoder = CoreMLEncoder.from_pretrained("large-v3")
        encoder_output = encoder(mel_spectrogram)  # Runs on ANE
    """

    def __init__(
        self,
        model_path: str | Path,
        compute_units: str = "CPU_AND_GPU",
    ):
        """
        Initialize CoreML encoder.

        Args:
            model_path: Path to model directory containing AudioEncoder.mlmodelc
            compute_units: CoreML compute units. Options:
                - "CPU_AND_GPU": CPU and GPU (default, fastest on M4)
                - "CPU_AND_NE": Use Neural Engine (may be slower on M4 Max)
                - "CPU_ONLY": CPU only
                - "ALL": All available (CPU, GPU, NE)
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools is required for CoreML encoder. "
                "Install with: pip install coremltools",
            ) from None

        self._ct = ct
        self.model_path = Path(model_path)
        self.compute_units_str = compute_units

        # Load encoder model
        encoder_dir = self.model_path / "AudioEncoder.mlmodelc"
        if not encoder_dir.exists():
            raise FileNotFoundError(
                f"AudioEncoder.mlmodelc not found at {encoder_dir}. "
                "Run scripts/download_whisperkit_models.py to download.",
            )

        # WhisperKit models are stored as mlmodelc directories with:
        # - model.mlmodel: The model spec
        # - weights/weight.bin: The weights
        mlmodel_path = encoder_dir / "model.mlmodel"
        weights_dir = encoder_dir / "weights"

        if not mlmodel_path.exists():
            raise FileNotFoundError(
                f"model.mlmodel not found in {encoder_dir}. "
                "The model may be corrupted or incomplete.",
            )

        # Set compute units
        compute_unit_map = {
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "ALL": ct.ComputeUnit.ALL,
        }
        compute_unit = compute_unit_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)

        logger.info("Loading CoreML encoder from %s", mlmodel_path)
        logger.info("Compute units: %s", compute_units)

        # Load with weights directory
        self.encoder = ct.models.MLModel(
            str(mlmodel_path),
            weights_dir=str(weights_dir) if weights_dir.exists() else None,
            compute_units=compute_unit,
        )

        # Load config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Extract model parameters from config
        self.n_state = self.config.get("d_model", 1280)
        self.n_mels = self.config.get("num_mel_bins", 128)
        self.n_ctx = self.config.get("max_source_positions", 1500)

        # Get input/output spec
        self._input_spec = self.encoder.get_spec().description.input
        self._output_spec = self.encoder.get_spec().description.output
        self._input_names = [inp.name for inp in self._input_spec]
        self._output_names = [out.name for out in self._output_spec]

        logger.info("Encoder input names: %s", self._input_names)
        logger.info("Encoder output names: %s", self._output_names)
        logger.info("Model params: n_state=%d, n_mels=%d, n_ctx=%d", self.n_state, self.n_mels, self.n_ctx)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "large-v3",
        models_dir: Path | None = None,
        compute_units: str = "CPU_AND_NE",
        auto_download: bool = True,
    ) -> "CoreMLEncoder":
        """
        Load a pretrained CoreML encoder.

        Args:
            model_name: Model name (e.g., "large-v3", "turbo")
            models_dir: Directory containing models (default: models/whisperkit)
            compute_units: CoreML compute units
            auto_download: If True, download model if not found

        Returns:
            CoreMLEncoder instance
        """
        if models_dir is None:
            models_dir = DEFAULT_MODELS_DIR
        models_dir = Path(models_dir)

        # Get full model path
        if model_name in AVAILABLE_MODELS:
            model_subdir = AVAILABLE_MODELS[model_name]
        else:
            model_subdir = model_name

        model_path = models_dir / model_subdir

        # Check if model exists
        encoder_path = model_path / "AudioEncoder.mlmodelc"
        if not encoder_path.exists():
            if auto_download:
                logger.info("Model not found, downloading %s...", model_name)
                from scripts.download_whisperkit_models import download_model
                download_model(model_name, models_dir, encoder_only=True)
            else:
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Run scripts/download_whisperkit_models.py or set auto_download=True.",
                )

        return cls(model_path, compute_units=compute_units)

    def __call__(
        self,
        mel: np.ndarray,
        variable_length: bool = True,
    ) -> np.ndarray:
        """
        Encode audio mel spectrogram.

        Args:
            mel: Mel spectrogram, shape (n_frames, n_mels) or (batch, n_frames, n_mels)
            variable_length: Ignored (CoreML model has fixed input size)

        Returns:
            Encoded audio features as numpy array, shape (batch, seq_len, n_state)

        Note:
            The output can be converted to MLX array with mx.array(output).

        WhisperKit CoreML model format:
            Input:  melspectrogram_features (batch, n_mels, 1, n_frames) FLOAT16
            Output: encoder_output_embeds (batch, n_state, 1, seq_len) FLOAT16
        """
        # Handle MLX arrays
        try:
            import mlx.core as mx
            if isinstance(mel, mx.array):
                mel = np.array(mel)
        except ImportError:
            pass

        # Ensure correct dtype - CoreML uses float16
        mel = np.asarray(mel, dtype=np.float16)

        # Add batch dimension if needed
        # Input: (n_frames, n_mels) -> (batch, n_frames, n_mels)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]  # (1, n_frames, n_mels)
        else:
            pass

        mel.shape[0]
        n_frames = mel.shape[1]
        mel.shape[2]

        # Pad to expected 3000 frames if needed (30s audio)
        expected_frames = 3000
        if n_frames < expected_frames:
            # Pad with zeros
            pad_width = ((0, 0), (0, expected_frames - n_frames), (0, 0))
            mel = np.pad(mel, pad_width, mode='constant', constant_values=0)
            n_frames = expected_frames
        elif n_frames > expected_frames:
            # Truncate (should not happen for proper audio)
            logger.warning("Input %d frames exceeds max %d, truncating", n_frames, expected_frames)
            mel = mel[:, :expected_frames, :]
            n_frames = expected_frames

        # Convert from (batch, n_frames, n_mels) to WhisperKit format (batch, n_mels, 1, n_frames)
        mel_4d = mel.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (batch, n_mels, 1, n_frames)

        # Prepare input dict
        input_name = self._input_names[0] if self._input_names else "melspectrogram_features"

        # Run inference
        prediction = self.encoder.predict({input_name: mel_4d})

        # Get output - shape is (batch, n_state, 1, seq_len)
        output_name = self._output_names[0] if self._output_names else "encoder_output_embeds"
        output = prediction[output_name]

        # Convert from (batch, n_state, 1, seq_len) to (batch, seq_len, n_state)
        # Squeeze the extra dimension and transpose
        if output.ndim == 4:
            output = output[:, :, 0, :]  # (batch, n_state, seq_len)
            output = output.transpose(0, 2, 1)  # (batch, seq_len, n_state)

        # Convert back to float32 for downstream processing
        return output.astype(np.float32)


    def get_output_length(self, n_mel_frames: int) -> int:
        """
        Calculate encoder output length for given mel spectrogram length.

        Same calculation as MLX encoder: (n_frames + 1) // 2

        Args:
            n_mel_frames: Number of mel spectrogram frames

        Returns:
            Encoder output sequence length
        """
        return (n_mel_frames + 1) // 2

    @property
    def available_compute_units(self) -> list:
        """List available compute units for debugging."""
        return ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]

    def benchmark(
        self,
        mel: np.ndarray | None = None,
        n_iterations: int = 10,
        warmup: int = 3,
    ) -> dict:
        """
        Benchmark encoder performance.

        Args:
            mel: Input mel spectrogram (or generate random if None)
            n_iterations: Number of timed iterations
            warmup: Number of warmup iterations

        Returns:
            Dict with timing statistics
        """
        import time

        # Generate test input if not provided
        if mel is None:
            # 30s audio = 3000 mel frames
            rng = np.random.default_rng()
            mel = rng.standard_normal((1, 3000, self.n_mels)).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            _ = self(mel)

        # Timed iterations
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = self(mel)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "input_shape": mel.shape,
            "compute_units": self.compute_units_str,
        }


def test_encoder():
    """Quick test of CoreML encoder."""
    import time

    print("Testing CoreML encoder...")

    # Load encoder
    try:
        encoder = CoreMLEncoder.from_pretrained("large-v3", auto_download=False)
    except Exception as e:
        print(f"Failed to load encoder: {e}")
        return False

    print(f"Loaded encoder: n_state={encoder.n_state}, n_mels={encoder.n_mels}")
    print(f"Input names: {encoder._input_names}")
    print(f"Output names: {encoder._output_names}")

    # Test with random input (standard Whisper mel format)
    # Input: (batch, n_frames, n_mels) = (1, 3000, 128)
    rng = np.random.default_rng()
    mel = rng.standard_normal((1, 3000, encoder.n_mels)).astype(np.float32)
    print(f"Input shape: {mel.shape}")

    # Run inference
    start = time.perf_counter()
    output = encoder(mel)
    elapsed = time.perf_counter() - start

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Time: {elapsed*1000:.1f}ms")

    # Verify output shape
    # Output should be (batch, seq_len, n_state) = (1, 1500, 1280)
    expected_seq_len = 1500  # Fixed for WhisperKit
    actual_seq_len = output.shape[1]
    print(f"Expected seq_len: {expected_seq_len}, actual: {actual_seq_len}")

    expected_n_state = encoder.n_state
    actual_n_state = output.shape[2]
    print(f"Expected n_state: {expected_n_state}, actual: {actual_n_state}")

    # Verify shapes match
    if output.shape == (1, 1500, 1280):
        print("Output shape correct!")
    else:
        print(f"WARNING: Unexpected output shape {output.shape}")

    # Benchmark
    print("\nBenchmarking (10 iterations)...")
    results = encoder.benchmark(mel, n_iterations=10, warmup=3)
    print(f"Mean: {results['mean_ms']:.1f}ms, Std: {results['std_ms']:.1f}ms")
    print(f"Min: {results['min_ms']:.1f}ms, Max: {results['max_ms']:.1f}ms")

    return True


if __name__ == "__main__":
    test_encoder()
