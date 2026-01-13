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
Wake Word Detector Converter

Converts ONNX wake word models to MLX format.

Architecture:
The wake word detector uses a 3-stage pipeline:
1. Mel Spectrogram: Audio -> Mel features (melspectrogram.onnx, ~1.0 MB)
2. Embedding: Mel features -> Embedding vector (embedding_model.onnx, ~1.3 MB)
3. Classifier: Embedding -> Wake word probability (hey_agent.onnx, ~0.8 MB)

Expected model location: ~/voice/models/wakeword/
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    MLX_AVAILABLE = False

# Import MLX wake word models (lazy import to avoid circular deps)
WakeWordPipeline = None

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class DetectionResult:
    """Result of wake word detection."""

    success: bool
    detected: bool
    probability: float
    inference_time_seconds: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark result for wake word detection."""

    total_frames: int
    total_time_seconds: float
    avg_frame_time_ms: float
    fps: float
    model_type: str  # "onnx" or "mlx"


# Default model paths
DEFAULT_MODEL_DIR = Path.home() / "voice" / "models" / "wakeword"
DEFAULT_MELSPEC_PATH = DEFAULT_MODEL_DIR / "melspectrogram.onnx"
DEFAULT_EMBEDDING_PATH = DEFAULT_MODEL_DIR / "embedding_model.onnx"
DEFAULT_CLASSIFIER_PATH = DEFAULT_MODEL_DIR / "hey_agent.onnx"


class WakeWordConverter:
    """
    Wake Word detector converter.

    Converts ONNX wake word models to MLX format for efficient
    inference on Apple Silicon. Falls back to ONNX Runtime when
    MLX models are not available.

    Example:
        converter = WakeWordConverter()

        # Check if models exist
        if converter.models_available():
            result = converter.detect(audio_samples)
            print(f"Detected: {result.detected}, Probability: {result.probability:.3f}")
        else:
            print("Wake word models not found")
    """

    def __init__(
        self,
        melspec_path: Path | None = None,
        embedding_path: Path | None = None,
        classifier_path: Path | None = None,
        use_mlx: bool = True,
    ):
        """
        Initialize wake word converter.

        Args:
            melspec_path: Path to mel spectrogram ONNX model
            embedding_path: Path to embedding ONNX model
            classifier_path: Path to wake word classifier ONNX model
            use_mlx: Use MLX if available, otherwise ONNX Runtime
        """
        self.melspec_path = melspec_path or DEFAULT_MELSPEC_PATH
        self.embedding_path = embedding_path or DEFAULT_EMBEDDING_PATH
        self.classifier_path = classifier_path or DEFAULT_CLASSIFIER_PATH
        self.use_mlx = use_mlx and MLX_AVAILABLE

        # Model sessions (loaded lazily)
        self._melspec_session: ort.InferenceSession | None = None
        self._embedding_session: ort.InferenceSession | None = None
        self._classifier_session: ort.InferenceSession | None = None

        # MLX models (when converted) - Any type due to lazy imports
        self._mlx_melspec: Any = None
        self._mlx_embedding: Any = None
        self._mlx_classifier: Any = None
        self._mlx_pipeline: Any = None

    def load_mlx_models(self) -> bool:
        """
        Load MLX models from ONNX files.

        Returns:
            True if models loaded successfully, False otherwise
        """
        if not MLX_AVAILABLE:
            return False

        if not self.models_available():
            return False

        try:
            # Lazy import to avoid circular dependencies
            global WakeWordPipeline
            if WakeWordPipeline is None:
                from tools.pytorch_to_mlx.converters.wakeword_mlx_models import (
                    WakeWordPipeline as WP,
                )

                WakeWordPipeline = WP

            # Load pipeline from ONNX
            model_dir = self.melspec_path.parent
            self._mlx_pipeline = WakeWordPipeline.from_onnx(str(model_dir))

            # Set individual components for backward compatibility
            self._mlx_melspec = self._mlx_pipeline.mel
            self._mlx_embedding = self._mlx_pipeline.embedding
            self._mlx_classifier = self._mlx_pipeline.classifier

            return True
        except Exception:
            return False

    def models_available(self) -> bool:
        """Check if all required ONNX models exist."""
        return (
            self.melspec_path.exists()
            and self.embedding_path.exists()
            and self.classifier_path.exists()
        )

    def get_model_status(self) -> dict[str, Any]:
        """
        Get status of model files.

        Returns:
            Dict with model status information
        """
        return {
            "melspec": {
                "path": str(self.melspec_path),
                "exists": self.melspec_path.exists(),
                "size_mb": self.melspec_path.stat().st_size / 1e6
                if self.melspec_path.exists()
                else 0,
            },
            "embedding": {
                "path": str(self.embedding_path),
                "exists": self.embedding_path.exists(),
                "size_mb": self.embedding_path.stat().st_size / 1e6
                if self.embedding_path.exists()
                else 0,
            },
            "classifier": {
                "path": str(self.classifier_path),
                "exists": self.classifier_path.exists(),
                "size_mb": self.classifier_path.stat().st_size / 1e6
                if self.classifier_path.exists()
                else 0,
            },
            "onnx_available": ONNX_AVAILABLE,
            "mlx_available": MLX_AVAILABLE,
        }

    def _load_onnx_sessions(self) -> None:
        """Load ONNX Runtime sessions for all models."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime",
            )

        if not self.models_available():
            raise FileNotFoundError(
                f"Wake word models not found. Expected:\n"
                f"  - {self.melspec_path}\n"
                f"  - {self.embedding_path}\n"
                f"  - {self.classifier_path}",
            )

        # Use CoreML execution provider on Apple Silicon if available
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        if self._melspec_session is None:
            self._melspec_session = ort.InferenceSession(
                str(self.melspec_path), providers=providers,
            )
        if self._embedding_session is None:
            self._embedding_session = ort.InferenceSession(
                str(self.embedding_path), providers=providers,
            )
        if self._classifier_session is None:
            self._classifier_session = ort.InferenceSession(
                str(self.classifier_path), providers=providers,
            )

    def detect_onnx(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> DetectionResult:
        """
        Detect wake word using ONNX Runtime.

        Pipeline architecture:
        1. Mel spectrogram: audio [batch, samples] -> mel [1, 1, T, 32]
        2. Embedding: mel [batch, 76, 32, 1] -> emb [batch, 1, 1, 96]
        3. Classifier: embeddings [1, 16, 96] -> logits [1, 1]

        The embedding model requires exactly 76 mel frames (~780ms audio).
        The classifier requires 16 embeddings (sliding window detection).
        For single-shot detection, we generate embeddings from overlapping
        windows and accumulate them into the classifier buffer.

        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz (default 16000)

        Returns:
            DetectionResult with detection status and probability
        """
        try:
            self._load_onnx_sessions()

            # Type narrowing - _load_onnx_sessions ensures these are set
            assert self._melspec_session is not None
            assert self._embedding_session is not None
            assert self._classifier_session is not None

            start_time = time.perf_counter()

            # Constants for OpenWakeWord models
            MEL_FRAMES_REQUIRED = 76  # Frames needed per embedding
            EMBEDDINGS_REQUIRED = 16  # Embeddings needed for classifier
            MIN_SAMPLES_FOR_76_FRAMES = 12800  # ~800ms at 16kHz

            # Ensure audio is long enough for at least one embedding
            audio_flat = audio.flatten().astype(np.float32)
            if len(audio_flat) < MIN_SAMPLES_FOR_76_FRAMES:
                # Pad with zeros if too short
                padding = np.zeros(
                    MIN_SAMPLES_FOR_76_FRAMES - len(audio_flat), dtype=np.float32,
                )
                audio_flat = np.concatenate([audio_flat, padding])

            # Stage 1: Mel spectrogram
            mel_input_name = self._melspec_session.get_inputs()[0].name
            mel_output = self._melspec_session.run(
                None, {mel_input_name: audio_flat.reshape(1, -1)},
            )[0]
            # mel_output shape: [1, 1, T, 32]

            total_mel_frames = mel_output.shape[2]
            if total_mel_frames < MEL_FRAMES_REQUIRED:
                return DetectionResult(
                    success=False,
                    detected=False,
                    probability=0.0,
                    inference_time_seconds=0.0,
                    error=f"Not enough mel frames: got {total_mel_frames}, need {MEL_FRAMES_REQUIRED}",
                )

            # Stage 2: Generate embeddings from sliding windows
            # Calculate how many embeddings we can generate
            emb_input_name = self._embedding_session.get_inputs()[0].name
            embeddings_buffer = np.zeros((1, EMBEDDINGS_REQUIRED, 96), dtype=np.float32)

            # Determine frame step for overlapping windows
            num_possible_embeddings = total_mel_frames - MEL_FRAMES_REQUIRED + 1
            if num_possible_embeddings >= EMBEDDINGS_REQUIRED:
                # We have enough frames - use evenly spaced windows
                step = max(1, (num_possible_embeddings - 1) // (EMBEDDINGS_REQUIRED - 1))
                frame_starts = [i * step for i in range(EMBEDDINGS_REQUIRED)]
            else:
                # Not enough frames - duplicate embeddings to fill buffer
                frame_starts = list(range(num_possible_embeddings))
                # Pad by repeating last embedding position
                while len(frame_starts) < EMBEDDINGS_REQUIRED:
                    frame_starts.append(frame_starts[-1])

            for i, frame_start in enumerate(frame_starts):
                # Extract 76-frame window
                mel_window = mel_output[:, :, frame_start : frame_start + MEL_FRAMES_REQUIRED, :]
                # Reshape: [1, 1, 76, 32] -> [1, 76, 32, 1]
                mel_for_emb = mel_window.squeeze(1)[:, :, :, np.newaxis]

                # Run embedding
                emb = self._embedding_session.run(None, {emb_input_name: mel_for_emb})[0]
                # emb shape: [1, 1, 1, 96] -> flatten to 96
                embeddings_buffer[0, i, :] = emb.reshape(96)

            # Stage 3: Classification
            cls_input_name = self._classifier_session.get_inputs()[0].name
            logits = self._classifier_session.run(
                None, {cls_input_name: embeddings_buffer},
            )[0]

            inference_time = time.perf_counter() - start_time

            # Convert logits to probability
            probability = float(1.0 / (1.0 + np.exp(-logits[0, 0])))
            detected = probability > 0.5

            return DetectionResult(
                success=True,
                detected=detected,
                probability=probability,
                inference_time_seconds=inference_time,
            )

        except Exception as e:
            return DetectionResult(
                success=False,
                detected=False,
                probability=0.0,
                inference_time_seconds=0.0,
                error=str(e),
            )

    def detect_mlx(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> DetectionResult:
        """
        Detect wake word using MLX.

        Uses the same pipeline architecture as ONNX:
        1. Mel spectrogram: audio -> mel features
        2. Embedding: mel windows -> 96-dim embeddings
        3. Classifier: 16 embeddings -> probability

        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz (default 16000)

        Returns:
            DetectionResult with detection status and probability
        """
        if not MLX_AVAILABLE:
            return DetectionResult(
                success=False,
                detected=False,
                probability=0.0,
                inference_time_seconds=0.0,
                error="MLX not available. Install with: pip install mlx",
            )

        if self._mlx_pipeline is None:
            return DetectionResult(
                success=False,
                detected=False,
                probability=0.0,
                inference_time_seconds=0.0,
                error="MLX models not loaded. Use load_mlx_models() first.",
            )

        try:
            start_time = time.perf_counter()

            # Constants matching ONNX pipeline
            MEL_FRAMES_REQUIRED = 76
            EMBEDDINGS_REQUIRED = 16
            MIN_SAMPLES_FOR_76_FRAMES = 12800

            # Ensure audio is long enough
            audio_flat = audio.flatten().astype(np.float32)
            if len(audio_flat) < MIN_SAMPLES_FOR_76_FRAMES:
                padding = np.zeros(
                    MIN_SAMPLES_FOR_76_FRAMES - len(audio_flat), dtype=np.float32,
                )
                audio_flat = np.concatenate([audio_flat, padding])

            # Convert to MLX array
            audio_mx = mx.array(audio_flat)

            # Stage 1: Mel spectrogram
            mel = self._mlx_melspec(audio_mx)
            mx.eval(mel)
            # mel shape: [batch, 1, time, n_mels]

            total_mel_frames = mel.shape[2]
            if total_mel_frames < MEL_FRAMES_REQUIRED:
                return DetectionResult(
                    success=False,
                    detected=False,
                    probability=0.0,
                    inference_time_seconds=0.0,
                    error=f"Not enough mel frames: got {total_mel_frames}, need {MEL_FRAMES_REQUIRED}",
                )

            # Stage 2: Generate embeddings from sliding windows
            embeddings_buffer = mx.zeros((1, EMBEDDINGS_REQUIRED, 96))

            num_possible = total_mel_frames - MEL_FRAMES_REQUIRED + 1
            if num_possible >= EMBEDDINGS_REQUIRED:
                step = max(1, (num_possible - 1) // (EMBEDDINGS_REQUIRED - 1))
                frame_starts = [i * step for i in range(EMBEDDINGS_REQUIRED)]
            else:
                frame_starts = list(range(num_possible))
                while len(frame_starts) < EMBEDDINGS_REQUIRED:
                    frame_starts.append(frame_starts[-1])

            # Build embeddings list and stack
            embeddings_list = []
            for _i, frame_start in enumerate(frame_starts):
                # Extract window: [batch, 1, 76, 32]
                mel_window = mel[:, :, frame_start : frame_start + MEL_FRAMES_REQUIRED, :]
                # Reshape for embedding: [batch, 76, 32, 1]
                mel_emb = mx.transpose(mel_window, (0, 2, 3, 1))

                # Run embedding
                emb = self._mlx_embedding(mel_emb)  # [batch, 1, 1, 96]
                embeddings_list.append(emb.reshape(96))

            # Stack embeddings: [16, 96] then reshape to [1, 16, 96]
            embeddings_buffer = mx.stack(embeddings_list, axis=0).reshape(
                1, EMBEDDINGS_REQUIRED, 96,
            )
            mx.eval(embeddings_buffer)

            # Stage 3: Classification
            prob = self._mlx_classifier(embeddings_buffer)
            mx.eval(prob)

            inference_time = time.perf_counter() - start_time

            probability = float(prob[0, 0])
            detected = probability > 0.5

            return DetectionResult(
                success=True,
                detected=detected,
                probability=probability,
                inference_time_seconds=inference_time,
            )

        except Exception as e:
            return DetectionResult(
                success=False,
                detected=False,
                probability=0.0,
                inference_time_seconds=0.0,
                error=str(e),
            )

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> DetectionResult:
        """
        Detect wake word using best available backend.

        Uses MLX if available and models are loaded, otherwise ONNX Runtime.

        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz (default 16000)

        Returns:
            DetectionResult with detection status and probability
        """
        if self.use_mlx and self._mlx_melspec is not None:
            return self.detect_mlx(audio, sample_rate)
        return self.detect_onnx(audio, sample_rate)

    def analyze_onnx_model(self, model_path: Path) -> dict[str, Any]:
        """
        Analyze ONNX model structure.

        Args:
            model_path: Path to ONNX model file

        Returns:
            Dict with model information (inputs, outputs, ops)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx is required. Install with: pip install onnx")

        model = onnx.load(str(model_path))

        # Extract input/output info
        inputs = []
        for inp in model.graph.input:
            shape = [
                d.dim_value if d.dim_value > 0 else "dynamic"
                for d in inp.type.tensor_type.shape.dim
            ]
            inputs.append(
                {
                    "name": inp.name,
                    "shape": shape,
                    "dtype": onnx.TensorProto.DataType.Name(
                        inp.type.tensor_type.elem_type,
                    ),
                },
            )

        outputs = []
        for out in model.graph.output:
            shape = [
                d.dim_value if d.dim_value > 0 else "dynamic"
                for d in out.type.tensor_type.shape.dim
            ]
            outputs.append(
                {
                    "name": out.name,
                    "shape": shape,
                    "dtype": onnx.TensorProto.DataType.Name(
                        out.type.tensor_type.elem_type,
                    ),
                },
            )

        # Count ops
        op_counts: dict[str, int] = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        return {
            "path": str(model_path),
            "inputs": inputs,
            "outputs": outputs,
            "op_counts": op_counts,
            "total_ops": len(model.graph.node),
        }

    def analyze_all_models(self) -> dict[str, Any]:
        """
        Analyze all wake word ONNX models.

        Returns:
            Dict with analysis for each model
        """
        if not self.models_available():
            return {
                "error": "Models not found",
                "status": self.get_model_status(),
            }

        return {
            "melspec": self.analyze_onnx_model(self.melspec_path),
            "embedding": self.analyze_onnx_model(self.embedding_path),
            "classifier": self.analyze_onnx_model(self.classifier_path),
        }

    def convert_to_mlx(self, output_dir: Path | None = None) -> dict[str, Any]:
        """
        Convert ONNX models to MLX format.

        Note: This is a placeholder for future implementation.
        Currently returns the MLX model architecture that would be needed.

        Args:
            output_dir: Directory to save MLX models

        Returns:
            Dict with conversion status and generated code paths
        """
        if not self.models_available():
            return {
                "success": False,
                "error": "ONNX models not found",
                "status": self.get_model_status(),
            }

        # Analyze models to understand structure
        analysis = self.analyze_all_models()

        # Generate MLX model code templates
        mlx_code = self._generate_mlx_templates(analysis)

        return {
            "success": True,
            "message": "MLX conversion templates generated",
            "analysis": analysis,
            "mlx_templates": mlx_code,
            "next_steps": [
                "1. Implement MLX modules based on ONNX op analysis",
                "2. Convert weights from ONNX to MLX format",
                "3. Validate numerical equivalence",
                "4. Benchmark performance",
            ],
        }

    def _generate_mlx_templates(self, analysis: dict[str, Any]) -> dict[str, str]:
        """Generate MLX model code templates based on ONNX analysis."""
        templates = {}

        # Mel spectrogram template
        if "melspec" in analysis:
            melspec_ops = analysis["melspec"].get("op_counts", {})
            templates["melspec"] = f'''
# MLX Mel Spectrogram Model
# Ops detected: {melspec_ops}

class MelSpectrogram(nn.Module):
    """Mel spectrogram extractor in MLX."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, n_mels: int = 80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        # TODO: Implement mel filterbank

    def __call__(self, audio: mx.array) -> mx.array:
        # TODO: Implement STFT + mel filterbank
        pass
'''

        # Embedding template
        if "embedding" in analysis:
            emb_ops = analysis["embedding"].get("op_counts", {})
            templates["embedding"] = f'''
# MLX Embedding Model
# Ops detected: {emb_ops}

class WakeWordEmbedding(nn.Module):
    """Wake word embedding extractor in MLX."""

    def __init__(self, input_dim: int, embedding_dim: int = 256):
        super().__init__()
        # TODO: Define layers based on ONNX structure

    def __call__(self, mel: mx.array) -> mx.array:
        # TODO: Implement forward pass
        pass
'''

        # Classifier template
        if "classifier" in analysis:
            cls_ops = analysis["classifier"].get("op_counts", {})
            templates["classifier"] = f'''
# MLX Wake Word Classifier
# Ops detected: {cls_ops}

class WakeWordClassifier(nn.Module):
    """Wake word classifier in MLX."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # TODO: Define classifier layers

    def __call__(self, embedding: mx.array) -> mx.array:
        # TODO: Implement forward pass returning logits
        pass
'''

        return templates

    def benchmark(
        self, duration_seconds: float = 10.0, sample_rate: int = 16000,
    ) -> BenchmarkResult:
        """
        Benchmark wake word detection performance.

        Args:
            duration_seconds: Duration of synthetic audio to process
            sample_rate: Sample rate in Hz

        Returns:
            BenchmarkResult with performance metrics
        """
        if not self.models_available():
            raise FileNotFoundError("Wake word models not found")

        # Generate synthetic audio
        num_samples = int(duration_seconds * sample_rate)
        rng = np.random.default_rng()
        audio = rng.standard_normal(num_samples).astype(np.float32) * 0.01

        # Process in chunks (typical wake word window size)
        chunk_size = int(0.5 * sample_rate)  # 500ms chunks
        num_chunks = num_samples // chunk_size

        start_time = time.perf_counter()

        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            result = self.detect(chunk, sample_rate)
            if not result.success:
                raise RuntimeError(f"Detection failed: {result.error}")

        total_time = time.perf_counter() - start_time

        return BenchmarkResult(
            total_frames=num_chunks,
            total_time_seconds=total_time,
            avg_frame_time_ms=(total_time / num_chunks) * 1000,
            fps=num_chunks / total_time,
            model_type="mlx"
            if (self.use_mlx and self._mlx_melspec is not None)
            else "onnx",
        )

    @staticmethod
    def get_expected_model_paths() -> dict[str, str]:
        """Get expected model file paths."""
        return {
            "melspec": str(DEFAULT_MELSPEC_PATH),
            "embedding": str(DEFAULT_EMBEDDING_PATH),
            "classifier": str(DEFAULT_CLASSIFIER_PATH),
        }
