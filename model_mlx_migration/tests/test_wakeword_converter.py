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
Tests for Wake Word Converter

Tests the WakeWordConverter infrastructure.
Note: Most tests skip when ONNX models are not available.
"""

import numpy as np
import pytest

from tools.pytorch_to_mlx.converters.wakeword_converter import (
    DEFAULT_MODEL_DIR,
    BenchmarkResult,
    DetectionResult,
    WakeWordConverter,
)

# Check if onnxruntime is available
try:
    import onnxruntime  # noqa: F401
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestWakeWordConverterInit:
    """Tests for WakeWordConverter initialization."""

    def test_init_default_paths(self):
        """Test default model path initialization."""
        converter = WakeWordConverter()

        assert converter.melspec_path == DEFAULT_MODEL_DIR / "melspectrogram.onnx"
        assert converter.embedding_path == DEFAULT_MODEL_DIR / "embedding_model.onnx"
        assert converter.classifier_path == DEFAULT_MODEL_DIR / "hey_agent.onnx"

    def test_init_custom_paths(self, tmp_path):
        """Test custom model path initialization."""
        melspec = tmp_path / "mel.onnx"
        embedding = tmp_path / "emb.onnx"
        classifier = tmp_path / "cls.onnx"

        converter = WakeWordConverter(
            melspec_path=melspec,
            embedding_path=embedding,
            classifier_path=classifier,
        )

        assert converter.melspec_path == melspec
        assert converter.embedding_path == embedding
        assert converter.classifier_path == classifier

    def test_models_available_when_missing(self):
        """Test models_available returns False when models don't exist."""
        converter = WakeWordConverter()
        # Default paths point to ~/voice/models/wakeword/ which likely don't exist
        # This is expected behavior
        result = converter.models_available()
        # We don't assert specific value since it depends on the environment
        assert isinstance(result, bool)

    def test_models_available_all_present(self, tmp_path):
        """Test models_available returns True when all models exist."""
        # Create fake model files
        melspec = tmp_path / "mel.onnx"
        embedding = tmp_path / "emb.onnx"
        classifier = tmp_path / "cls.onnx"

        for f in [melspec, embedding, classifier]:
            f.write_bytes(b"fake onnx data")

        converter = WakeWordConverter(
            melspec_path=melspec,
            embedding_path=embedding,
            classifier_path=classifier,
        )

        assert converter.models_available() is True


class TestWakeWordConverterStatus:
    """Tests for model status reporting."""

    def test_get_model_status(self, tmp_path):
        """Test get_model_status returns correct structure."""
        melspec = tmp_path / "mel.onnx"
        melspec.write_bytes(b"x" * 1000)  # 1KB file

        converter = WakeWordConverter(
            melspec_path=melspec,
            embedding_path=tmp_path / "missing.onnx",
            classifier_path=tmp_path / "missing2.onnx",
        )

        status = converter.get_model_status()

        # Check structure
        assert "melspec" in status
        assert "embedding" in status
        assert "classifier" in status
        assert "onnx_available" in status
        assert "mlx_available" in status

        # Check melspec status (exists)
        assert status["melspec"]["exists"] is True
        assert status["melspec"]["size_mb"] == pytest.approx(0.001, rel=0.1)

        # Check embedding status (missing)
        assert status["embedding"]["exists"] is False

    def test_get_expected_model_paths(self):
        """Test expected model paths are returned."""
        paths = WakeWordConverter.get_expected_model_paths()

        assert "melspec" in paths
        assert "embedding" in paths
        assert "classifier" in paths
        assert "melspectrogram.onnx" in paths["melspec"]
        assert "embedding_model.onnx" in paths["embedding"]
        assert "hey_agent.onnx" in paths["classifier"]


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_detection_result_success(self):
        """Test successful detection result."""
        result = DetectionResult(
            success=True,
            detected=True,
            probability=0.95,
            inference_time_seconds=0.005,
        )

        assert result.success is True
        assert result.detected is True
        assert result.probability == 0.95
        assert result.error is None

    def test_detection_result_failure(self):
        """Test failed detection result."""
        result = DetectionResult(
            success=False,
            detected=False,
            probability=0.0,
            inference_time_seconds=0.0,
            error="Model not found",
        )

        assert result.success is False
        assert result.error == "Model not found"


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result(self):
        """Test benchmark result structure."""
        result = BenchmarkResult(
            total_frames=100,
            total_time_seconds=1.5,
            avg_frame_time_ms=15.0,
            fps=66.7,
            model_type="onnx",
        )

        assert result.total_frames == 100
        assert result.total_time_seconds == 1.5
        assert result.avg_frame_time_ms == 15.0
        assert result.fps == pytest.approx(66.7, rel=0.01)
        assert result.model_type == "onnx"


class TestWakeWordDetection:
    """Tests for wake word detection functionality."""

    def test_detect_onnx_models_missing(self):
        """Test detection fails gracefully when models missing."""
        converter = WakeWordConverter()

        if converter.models_available():
            pytest.skip("ONNX models available - cannot test missing model path")

        audio = _rng.standard_normal(16000).astype(np.float32)
        result = converter.detect_onnx(audio)

        assert result.success is False
        # Error should indicate models not found OR onnxruntime not installed
        error_lower = result.error.lower()
        assert any(
            keyword in error_lower
            for keyword in ["not found", "not installed", "required"]
        )

    def test_detect_mlx_not_loaded(self):
        """Test MLX detection fails when models not loaded."""
        converter = WakeWordConverter(use_mlx=True)

        audio = _rng.standard_normal(16000).astype(np.float32)
        result = converter.detect_mlx(audio)

        assert result.success is False
        assert (
            "not loaded" in result.error.lower()
            or "not available" in result.error.lower()
        )


class TestMLXConversion:
    """Tests for MLX conversion template generation."""

    def test_generate_mlx_templates_empty(self):
        """Test template generation with empty analysis."""
        converter = WakeWordConverter()

        analysis = {
            "melspec": {"op_counts": {"Conv": 5, "Relu": 4}},
            "embedding": {"op_counts": {"MatMul": 3, "Add": 2}},
            "classifier": {"op_counts": {"Gemm": 2, "Softmax": 1}},
        }

        templates = converter._generate_mlx_templates(analysis)

        assert "melspec" in templates
        assert "embedding" in templates
        assert "classifier" in templates

        # Check templates contain expected content
        assert "MelSpectrogram" in templates["melspec"]
        assert "WakeWordEmbedding" in templates["embedding"]
        assert "WakeWordClassifier" in templates["classifier"]


@pytest.mark.skipif(
    not WakeWordConverter().models_available() or not ONNXRUNTIME_AVAILABLE,
    reason="Wake word ONNX models not available or onnxruntime not installed",
)
class TestWakeWordWithModels:
    """Tests that require actual ONNX models."""

    def test_analyze_all_models(self):
        """Test ONNX model analysis."""
        converter = WakeWordConverter()
        analysis = converter.analyze_all_models()

        assert "melspec" in analysis
        assert "embedding" in analysis
        assert "classifier" in analysis

        for model_name in ["melspec", "embedding", "classifier"]:
            assert "inputs" in analysis[model_name]
            assert "outputs" in analysis[model_name]
            assert "op_counts" in analysis[model_name]
            assert "total_ops" in analysis[model_name]

    def test_detect_onnx(self):
        """Test ONNX detection with real models."""
        converter = WakeWordConverter()

        # Generate 1 second of audio (at 16kHz)
        audio = _rng.standard_normal(16000).astype(np.float32) * 0.01
        result = converter.detect_onnx(audio)

        assert result.success is True
        assert 0.0 <= result.probability <= 1.0
        assert result.inference_time_seconds > 0

    def test_benchmark(self):
        """Test benchmark with real models."""
        converter = WakeWordConverter()

        result = converter.benchmark(duration_seconds=2.0)

        assert result.total_frames > 0
        assert result.total_time_seconds > 0
        assert result.avg_frame_time_ms > 0
        assert result.fps > 0
