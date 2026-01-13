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
Tests for Wake Word Training Pipeline

Tests the custom wake word training script including:
- Model architectures (CNN and FC)
- ONNX export for both architectures
- Mel spectrogram computation
"""

import sys
from pathlib import Path

import numpy as np

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

# pytest is optional for quick tests
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    pytest = None
    PYTEST_AVAILABLE = False

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "wakeword_training"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for MLX
MLX_AVAILABLE = False
try:
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401
    MLX_AVAILABLE = True
except ImportError:
    pass

# Check for ONNX
ONNX_AVAILABLE = False
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    pass


# Create skipif decorator that works without pytest
def skipif(condition, reason=""):
    """Skip decorator that works with or without pytest."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(condition, reason=reason)
    def decorator(cls_or_func):
        return cls_or_func
    return decorator


@skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestWakeWordClassifier:
    """Test FC baseline model."""

    def test_import(self):
        """Test that WakeWordClassifier can be imported."""
        from scripts.wakeword_training.train_hey_agent import WakeWordClassifier
        assert WakeWordClassifier is not None

    def test_create_model(self):
        """Test model creation."""
        from scripts.wakeword_training.train_hey_agent import WakeWordClassifier
        model = WakeWordClassifier()
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with random input."""
        from scripts.wakeword_training.train_hey_agent import WakeWordClassifier
        model = WakeWordClassifier()

        # Input: [batch, frames, mels]
        test_input = mx.array(_rng.standard_normal((2, 76, 32)).astype(np.float32))
        output = model(test_input)
        mx.eval(output)

        assert output.shape == (2, 1)
        # Output should be between 0 and 1 (sigmoid)
        assert float(mx.min(output)) >= 0
        assert float(mx.max(output)) <= 1

    def test_has_expected_layers(self):
        """Test that model has expected layers."""
        from scripts.wakeword_training.train_hey_agent import WakeWordClassifier
        model = WakeWordClassifier()

        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert hasattr(model, 'dropout')


@skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestWakeWordCNN:
    """Test CNN model."""

    def test_import(self):
        """Test that WakeWordCNN can be imported."""
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        assert WakeWordCNN is not None

    def test_create_model(self):
        """Test model creation."""
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        model = WakeWordCNN()
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with random input."""
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        model = WakeWordCNN()

        # Input: [batch, frames, mels]
        test_input = mx.array(_rng.standard_normal((2, 76, 32)).astype(np.float32))
        output = model(test_input)
        mx.eval(output)

        assert output.shape == (2, 1)
        # Output should be between 0 and 1 (sigmoid)
        assert float(mx.min(output)) >= 0
        assert float(mx.max(output)) <= 1

    def test_has_expected_layers(self):
        """Test that model has expected conv and fc layers."""
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        model = WakeWordCNN()

        # Conv layers
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'conv4')

        # BatchNorm layers
        assert hasattr(model, 'bn1')
        assert hasattr(model, 'bn2')
        assert hasattr(model, 'bn3')
        assert hasattr(model, 'bn4')

        # FC layers
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')

    def test_batch_invariance(self):
        """Test that single and batched inputs produce consistent outputs."""
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        model = WakeWordCNN()
        model.eval()  # Disable dropout for deterministic output

        # Create same input for single and batch
        single_input = mx.array(_rng.standard_normal((1, 76, 32)).astype(np.float32))
        batch_input = mx.concatenate([single_input, single_input], axis=0)

        single_out = model(single_input)
        batch_out = model(batch_input)
        mx.eval(single_out, batch_out)

        # First output in batch should match single output
        assert np.allclose(np.array(single_out), np.array(batch_out[0:1]), rtol=1e-5)


@skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMelSpectrogramComputation:
    """Test mel spectrogram computation."""

    def test_import(self):
        """Test that compute_mel_spectrogram can be imported."""
        from scripts.wakeword_training.train_hey_agent import compute_mel_spectrogram
        assert compute_mel_spectrogram is not None

    def test_basic_computation(self):
        """Test basic mel spectrogram computation."""
        from scripts.wakeword_training.train_hey_agent import compute_mel_spectrogram

        # Create a 1-second audio at 16kHz
        audio = _rng.standard_normal(16000).astype(np.float32)
        mel = compute_mel_spectrogram(audio)

        assert mel is not None
        assert mel.shape[1] == 32  # n_mels default
        assert len(mel.shape) == 2

    def test_output_shape(self):
        """Test mel spectrogram output shape."""
        from scripts.wakeword_training.train_hey_agent import compute_mel_spectrogram

        # Create audio of specific length
        sample_rate = 16000
        audio = _rng.standard_normal(sample_rate).astype(np.float32)  # 1 second
        mel = compute_mel_spectrogram(audio, sample_rate=sample_rate)

        # Expected frames: (16000 - 512) / 160 + 1 = 97.something
        assert mel.shape[0] > 90
        assert mel.shape[0] < 105


@skipif(not MLX_AVAILABLE or not ONNX_AVAILABLE, reason="MLX or ONNX not available")
class TestFCONNXExport:
    """Test FC model ONNX export."""

    def test_export_creates_file(self, tmp_path):
        """Test that export creates ONNX file."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordClassifier,
            export_to_onnx,
        )

        model = WakeWordClassifier()
        output_path = tmp_path / "test_fc.onnx"
        export_to_onnx(model, output_path, model_type="fc")

        assert output_path.exists()

    def test_export_valid_onnx(self, tmp_path):
        """Test that exported ONNX model is valid."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordClassifier,
            export_to_onnx,
        )

        model = WakeWordClassifier()
        output_path = tmp_path / "test_fc.onnx"
        export_to_onnx(model, output_path, model_type="fc")

        # Validate with ONNX checker
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_export_correct_io(self, tmp_path):
        """Test that exported model has correct input/output shapes."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordClassifier,
            export_to_onnx,
        )

        model = WakeWordClassifier()
        output_path = tmp_path / "test_fc.onnx"
        export_to_onnx(model, output_path, model_type="fc")

        onnx_model = onnx.load(str(output_path))

        # Check input shape
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        assert input_shape == [1, 76, 32]

        # Check output shape
        output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        assert output_shape == [1, 1]


@skipif(not MLX_AVAILABLE or not ONNX_AVAILABLE, reason="MLX or ONNX not available")
class TestCNNONNXExport:
    """Test CNN model ONNX export."""

    def test_export_creates_file(self, tmp_path):
        """Test that export creates ONNX file."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordCNN,
            export_cnn_to_onnx,
        )

        model = WakeWordCNN()
        output_path = tmp_path / "test_cnn.onnx"
        export_cnn_to_onnx(model, output_path)

        assert output_path.exists()

    def test_export_valid_onnx(self, tmp_path):
        """Test that exported ONNX model is valid."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordCNN,
            export_cnn_to_onnx,
        )

        model = WakeWordCNN()
        output_path = tmp_path / "test_cnn.onnx"
        export_cnn_to_onnx(model, output_path)

        # Validate with ONNX checker
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_export_correct_io(self, tmp_path):
        """Test that exported model has correct input/output shapes."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordCNN,
            export_cnn_to_onnx,
        )

        model = WakeWordCNN()
        output_path = tmp_path / "test_cnn.onnx"
        export_cnn_to_onnx(model, output_path)

        onnx_model = onnx.load(str(output_path))

        # Check input shape
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        assert input_shape == [1, 76, 32]

        # Check output shape
        output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        assert output_shape == [1, 1]

    def test_export_has_expected_nodes(self, tmp_path):
        """Test that exported CNN has expected node types."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordCNN,
            export_cnn_to_onnx,
        )

        model = WakeWordCNN()
        output_path = tmp_path / "test_cnn.onnx"
        export_cnn_to_onnx(model, output_path)

        onnx_model = onnx.load(str(output_path))

        node_types = {node.op_type for node in onnx_model.graph.node}

        # Expected node types for CNN
        assert "Transpose" in node_types  # For input format conversion
        assert "Conv" in node_types
        assert "BatchNormalization" in node_types
        assert "Relu" in node_types
        # MLX uses x[:, ::2, :] (subsampling), not MaxPool, so ONNX uses Slice
        assert "Slice" in node_types
        assert "GlobalAveragePool" in node_types
        assert "Squeeze" in node_types
        assert "MatMul" in node_types
        assert "Sigmoid" in node_types


@skipif(not MLX_AVAILABLE or not ONNX_AVAILABLE, reason="MLX or ONNX not available")
class TestTrainedModelExport:
    """Test ONNX export with actual trained model weights."""

    def test_load_and_export_cnn(self):
        """Test loading saved CNN model and exporting to ONNX."""
        from scripts.wakeword_training.train_hey_agent import (
            WakeWordCNN,
            export_cnn_to_onnx,
        )

        model_path = Path("models/wakeword/hey_agent/hey_agent_mlx.safetensors")
        if not model_path.exists():
            if PYTEST_AVAILABLE:
                pytest.skip("Trained model not available")
            else:
                print("Skipping: Trained model not available")
                return

        # Load model
        model = WakeWordCNN()
        flat_weights = dict(mx.load(str(model_path)))

        def unflatten_to_nested(flat):
            result = {}
            for key, value in flat.items():
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        model.update(unflatten_to_nested(flat_weights))
        mx.eval(model.parameters())

        # Export and verify
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_cnn_to_onnx(model, output_path)
            assert output_path.exists()

            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            # Check node count matches expected architecture
            assert len(onnx_model.graph.node) == 24
        finally:
            output_path.unlink(missing_ok=True)


# Check for Silero VAD (via torch.hub, since silero-vad pip requires onnxruntime
# which is not available on Python 3.14+ macOS ARM64)
SILERO_VAD_AVAILABLE = False
try:
    import torch
    # Quick check that torch.hub can load the model (uses cache if available)
    torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
    SILERO_VAD_AVAILABLE = True
except Exception:
    pass


@skipif(not MLX_AVAILABLE or not SILERO_VAD_AVAILABLE, reason="MLX or Silero VAD not available")
class TestVADIntegration:
    """Test VAD integration with wake word detector."""

    def test_import_silero_vad(self):
        """Test that SileroVAD class can be imported."""
        from scripts.wakeword_training.inference import SileroVAD
        assert SileroVAD is not None

    def test_silero_vad_init(self):
        """Test SileroVAD initialization."""
        from scripts.wakeword_training.inference import SileroVAD
        vad = SileroVAD(sampling_rate=16000)
        assert vad.sampling_rate == 16000
        assert vad._model is None  # Lazy loading

    def test_silero_vad_lazy_load(self):
        """Test that VAD model is lazy-loaded."""
        from scripts.wakeword_training.inference import SileroVAD
        vad = SileroVAD(sampling_rate=16000)
        # Access model to trigger lazy load
        model = vad.model
        assert model is not None

    def test_vad_silence_detection(self):
        """Test that VAD correctly detects no speech in silence."""
        from scripts.wakeword_training.inference import SileroVAD
        vad = SileroVAD(sampling_rate=16000)

        silence = np.zeros(16000, dtype=np.float32)
        assert vad.has_speech(silence) is False

    def test_vad_noise_detection(self):
        """Test that VAD correctly detects no speech in random noise."""
        from scripts.wakeword_training.inference import SileroVAD
        vad = SileroVAD(sampling_rate=16000)

        noise = (_rng.standard_normal(16000) * 0.1).astype(np.float32)
        assert vad.has_speech(noise) is False

    def test_detector_with_vad_silence(self):
        """Test that detector with VAD returns 0.0 for silence."""
        from scripts.wakeword_training.inference import WakeWordDetector

        model_path = Path("models/wakeword/hey_agent/hey_agent_mlx.safetensors")
        if not model_path.exists():
            if PYTEST_AVAILABLE:
                pytest.skip("Trained model not available")
            else:
                return

        detector = WakeWordDetector.from_mlx(model_path, use_vad=True)
        silence = np.zeros(16000, dtype=np.float32)
        prob = detector.detect(silence)
        assert prob == 0.0, f"Expected 0.0 for silence, got {prob}"

    def test_detector_with_vad_noise(self):
        """Test that detector with VAD returns 0.0 for random noise."""
        from scripts.wakeword_training.inference import WakeWordDetector

        model_path = Path("models/wakeword/hey_agent/hey_agent_mlx.safetensors")
        if not model_path.exists():
            if PYTEST_AVAILABLE:
                pytest.skip("Trained model not available")
            else:
                return

        detector = WakeWordDetector.from_mlx(model_path, use_vad=True)
        noise = (_rng.standard_normal(16000) * 0.5).astype(np.float32)
        prob = detector.detect(noise)
        assert prob == 0.0, f"Expected 0.0 for noise, got {prob}"

    def test_detector_without_vad_false_positive(self):
        """Test that detector WITHOUT VAD has false positives on silence/noise."""
        from scripts.wakeword_training.inference import WakeWordDetector

        model_path = Path("models/wakeword/hey_agent/hey_agent_mlx.safetensors")
        if not model_path.exists():
            if PYTEST_AVAILABLE:
                pytest.skip("Trained model not available")
            else:
                return

        detector = WakeWordDetector.from_mlx(model_path, use_vad=False)
        silence = np.zeros(16000, dtype=np.float32)
        prob = detector.detect(silence)
        # Without VAD, model may have false positives
        assert prob > 0.5, f"Without VAD, silence should trigger false positive, got {prob}"


def run_quick_test():
    """Quick test without pytest."""
    print("Testing Wake Word Training Pipeline...")
    print("=" * 50)

    if not MLX_AVAILABLE:
        print("MLX not available, skipping tests")
        return True

    # Test CNN model
    print("\n1. Testing WakeWordCNN...")
    try:
        from scripts.wakeword_training.train_hey_agent import WakeWordCNN
        model = WakeWordCNN()
        test_input = mx.array(_rng.standard_normal((1, 76, 32)).astype(np.float32))
        output = model(test_input)
        mx.eval(output)
        print(f"   CNN forward pass: shape={output.shape}, output={float(output[0, 0]):.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test FC model
    print("\n2. Testing WakeWordClassifier...")
    try:
        from scripts.wakeword_training.train_hey_agent import WakeWordClassifier
        model = WakeWordClassifier()
        output = model(test_input)
        mx.eval(output)
        print(f"   FC forward pass: shape={output.shape}, output={float(output[0, 0]):.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test ONNX export if available
    if ONNX_AVAILABLE:
        print("\n3. Testing CNN ONNX export...")
        try:
            import tempfile

            from scripts.wakeword_training.train_hey_agent import (
                WakeWordCNN,
                export_cnn_to_onnx,
            )

            model = WakeWordCNN()
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = Path(f.name)

            export_cnn_to_onnx(model, output_path)

            # Validate
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("   ONNX export and validation: PASSED")
            output_path.unlink()
        except Exception as e:
            print(f"   Failed: {e}")
            return False
    else:
        print("\n3. Skipping ONNX tests (ONNX not available)")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
