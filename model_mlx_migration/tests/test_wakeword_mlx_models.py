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
Tests for Wake Word MLX Models

Tests the MLX wake word model implementations without requiring ONNX weights for most tests.
For full validation tests, run with actual ONNX models present.
"""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestWakeWordImports:
    """Test module imports and initialization."""

    def test_import_module(self):
        """Test that module can be imported."""
        from pytorch_to_mlx.converters import wakeword_mlx_models

        assert wakeword_mlx_models is not None

    def test_import_mel_spectrogram(self):
        """Test that MelSpectrogram can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import MelSpectrogram

        assert MelSpectrogram is not None

    def test_import_embedding_model(self):
        """Test that EmbeddingModel can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import EmbeddingModel

        assert EmbeddingModel is not None

    def test_import_classifier(self):
        """Test that Classifier can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import Classifier

        assert Classifier is not None

    def test_import_pipeline(self):
        """Test that WakeWordPipeline can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import WakeWordPipeline

        assert WakeWordPipeline is not None

    def test_import_depthwise_separable_conv(self):
        """Test that DepthwiseSeparableConv2d can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import (
            DepthwiseSeparableConv2d,
        )

        assert DepthwiseSeparableConv2d is not None

    def test_import_validate_function(self):
        """Test that validate_against_onnx can be imported."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import validate_against_onnx

        assert validate_against_onnx is not None
        assert callable(validate_against_onnx)


class TestMelSpectrogram:
    """Test MelSpectrogram model."""

    def test_init_default(self):
        """Test MelSpectrogram with default parameters."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import MelSpectrogram

        model = MelSpectrogram()
        assert model.n_fft == 512
        assert model.hop_length == 160
        assert model.n_mels == 32
        assert model.n_freq == 257  # n_fft // 2 + 1

    def test_init_custom(self):
        """Test MelSpectrogram with custom parameters."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import MelSpectrogram

        model = MelSpectrogram(n_fft=1024, hop_length=256, n_mels=64)
        assert model.n_fft == 1024
        assert model.hop_length == 256
        assert model.n_mels == 64
        assert model.n_freq == 513  # 1024 // 2 + 1

    def test_has_conv_layers(self):
        """Test that MelSpectrogram has conv layers."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import MelSpectrogram

        model = MelSpectrogram()
        assert hasattr(model, "conv_real")
        assert hasattr(model, "conv_imag")
        assert hasattr(model, "mel_filterbank")

    def test_has_normalization_constants(self):
        """Test that MelSpectrogram has normalization constants."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import MelSpectrogram

        model = MelSpectrogram()
        assert hasattr(model, "clip_min")
        assert hasattr(model, "clip_max")
        assert model.clip_min == 1e-10
        assert model.clip_max == 1e10


class TestDepthwiseSeparableConv2d:
    """Test DepthwiseSeparableConv2d module."""

    def test_init(self):
        """Test DepthwiseSeparableConv2d initialization."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import (
            DepthwiseSeparableConv2d,
        )

        model = DepthwiseSeparableConv2d(
            in_channels=16, out_channels=32, kernel_size=3,
        )
        # Has a conv layer internally
        assert hasattr(model, "conv")
        assert model.conv is not None

    def test_conv_is_conv2d(self):
        """Test that the internal conv is a Conv2d layer."""
        import mlx.nn as nn
        from pytorch_to_mlx.converters.wakeword_mlx_models import (
            DepthwiseSeparableConv2d,
        )

        model = DepthwiseSeparableConv2d(
            in_channels=16, out_channels=32, kernel_size=3,
        )
        assert isinstance(model.conv, nn.Conv2d)


class TestEmbeddingModel:
    """Test EmbeddingModel."""

    def test_init(self):
        """Test EmbeddingModel initialization."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import EmbeddingModel

        model = EmbeddingModel()
        assert model is not None

    def test_has_conv_layers(self):
        """Test that EmbeddingModel has expected conv layers."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import EmbeddingModel

        model = EmbeddingModel()
        # Model uses conv0, conv1, etc. naming convention
        assert hasattr(model, "conv0")
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")

    def test_has_pooling_layers(self):
        """Test that EmbeddingModel has pooling layers."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import EmbeddingModel

        model = EmbeddingModel()
        # Model has multiple pooling layers
        assert hasattr(model, "pool1")
        assert hasattr(model, "pool2")

    def test_has_leaky_relu(self):
        """Test that EmbeddingModel has leaky relu method."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import EmbeddingModel

        model = EmbeddingModel()
        assert hasattr(model, "_leaky_relu")
        assert callable(model._leaky_relu)


class TestClassifier:
    """Test Classifier model."""

    def test_init(self):
        """Test Classifier initialization."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import Classifier

        model = Classifier()
        assert model is not None

    def test_has_dense_layers(self):
        """Test that Classifier has dense layers."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import Classifier

        model = Classifier()
        # Should have multiple dense layers
        assert hasattr(model, "dense1")
        assert hasattr(model, "dense2")
        assert hasattr(model, "dense3")

    def test_has_layer_norms(self):
        """Test that Classifier has layer normalization."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import Classifier

        model = Classifier()
        assert hasattr(model, "ln1")
        assert hasattr(model, "ln2")


class TestWakeWordPipeline:
    """Test WakeWordPipeline."""

    def test_init(self):
        """Test WakeWordPipeline initialization."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import WakeWordPipeline

        pipeline = WakeWordPipeline()
        assert pipeline is not None

    def test_has_components(self):
        """Test that WakeWordPipeline has all components."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import WakeWordPipeline

        pipeline = WakeWordPipeline()
        # Pipeline uses mel, embedding, classifier naming
        assert hasattr(pipeline, "mel")
        assert hasattr(pipeline, "embedding")
        assert hasattr(pipeline, "classifier")

    def test_from_onnx_method_exists(self):
        """Test that from_onnx factory method exists."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import WakeWordPipeline

        assert hasattr(WakeWordPipeline, "from_onnx")
        assert callable(WakeWordPipeline.from_onnx)

    def test_components_are_correct_types(self):
        """Test that pipeline components have correct types."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import (
            Classifier,
            EmbeddingModel,
            MelSpectrogram,
            WakeWordPipeline,
        )

        pipeline = WakeWordPipeline()
        assert isinstance(pipeline.mel, MelSpectrogram)
        assert isinstance(pipeline.embedding, EmbeddingModel)
        assert isinstance(pipeline.classifier, Classifier)


class TestValidateFunction:
    """Test validate_against_onnx function."""

    def test_function_exists(self):
        """Test that validate function exists."""
        from pytorch_to_mlx.converters.wakeword_mlx_models import validate_against_onnx

        assert validate_against_onnx is not None
        assert callable(validate_against_onnx)

    def test_function_signature(self):
        """Test validate function has expected parameters."""
        import inspect

        from pytorch_to_mlx.converters.wakeword_mlx_models import validate_against_onnx

        sig = inspect.signature(validate_against_onnx)
        params = list(sig.parameters.keys())
        assert "model_dir" in params
        assert "audio" in params


class TestONNXAvailability:
    """Test ONNX availability flag."""

    def test_onnx_flag_exists(self):
        """Test that ONNX_AVAILABLE flag exists."""
        from pytorch_to_mlx.converters import wakeword_mlx_models

        assert hasattr(wakeword_mlx_models, "ONNX_AVAILABLE")

    def test_onnx_flag_is_bool(self):
        """Test that ONNX_AVAILABLE is boolean."""
        from pytorch_to_mlx.converters import wakeword_mlx_models

        assert isinstance(wakeword_mlx_models.ONNX_AVAILABLE, bool)


def run_quick_test():
    """Quick test without pytest."""
    print("Testing Wake Word MLX Models...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.converters.wakeword_mlx_models import (
            Classifier,
            EmbeddingModel,
            MelSpectrogram,
            WakeWordPipeline,
        )

        print("   Imports successful!")
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False

    # Test MelSpectrogram
    print("\n2. Testing MelSpectrogram...")
    try:
        mel = MelSpectrogram()
        print(f"   Created MelSpectrogram: n_fft={mel.n_fft}, n_mels={mel.n_mels}")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test EmbeddingModel
    print("\n3. Testing EmbeddingModel...")
    try:
        _ = EmbeddingModel()
        print("   Created EmbeddingModel")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test Classifier
    print("\n4. Testing Classifier...")
    try:
        _ = Classifier()
        print("   Created Classifier")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test WakeWordPipeline
    print("\n5. Testing WakeWordPipeline...")
    try:
        _ = WakeWordPipeline()
        print("   Created WakeWordPipeline")
        print("   Components: mel_spec, embedding, classifier")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo test with actual ONNX models:")
    print('  python -c "')
    print("    from pytorch_to_mlx.converters.wakeword_mlx_models import WakeWordPipeline")
    print("    pipeline = WakeWordPipeline.from_onnx('~/voice/models/wakeword')")
    print('    print(\"Loaded successfully!\")"')

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
