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
Tests for TorchScriptAnalyzer module.

Tests for tools/pytorch_to_mlx/analyzer/torchscript_analyzer.py
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tools.pytorch_to_mlx.analyzer.torchscript_analyzer import (
    LayerInfo,
    ModelArchitecture,
    OpCategory,
    OpInfo,
    TorchScriptAnalyzer,
    WeightInfo,
)

# =============================================================================
# OpCategory Tests
# =============================================================================


class TestOpCategory:
    """Tests for OpCategory enum."""

    def test_op_category_values(self):
        """Test that all expected category values exist."""
        assert OpCategory.LAYER.value == "layer"
        assert OpCategory.ACTIVATION.value == "activation"
        assert OpCategory.NORMALIZATION.value == "norm"
        assert OpCategory.ATTENTION.value == "attention"
        assert OpCategory.POOLING.value == "pooling"
        assert OpCategory.ARITHMETIC.value == "arithmetic"
        assert OpCategory.TENSOR.value == "tensor"
        assert OpCategory.REDUCTION.value == "reduction"
        assert OpCategory.COMPARISON.value == "comparison"
        assert OpCategory.CUSTOM.value == "custom"
        assert OpCategory.UNKNOWN.value == "unknown"

    def test_op_category_from_string(self):
        """Test creating category from string value."""
        assert OpCategory("layer") == OpCategory.LAYER
        assert OpCategory("activation") == OpCategory.ACTIVATION


# =============================================================================
# OpInfo Tests
# =============================================================================


class TestOpInfo:
    """Tests for OpInfo dataclass."""

    def test_create_op_info(self):
        """Test creating OpInfo instance."""
        op = OpInfo(
            name="aten::linear",
            category=OpCategory.LAYER,
            input_types=["Tensor", "Tensor", "Tensor"],
            output_types=["Tensor"],
            attributes={"bias": True},
            count=5,
        )
        assert op.name == "aten::linear"
        assert op.category == OpCategory.LAYER
        assert op.count == 5

    def test_op_info_default_count(self):
        """Test OpInfo default count is 1."""
        op = OpInfo(
            name="aten::relu",
            category=OpCategory.ACTIVATION,
            input_types=["Tensor"],
            output_types=["Tensor"],
        )
        assert op.count == 1

    def test_op_info_default_attributes(self):
        """Test OpInfo default attributes is empty dict."""
        op = OpInfo(
            name="aten::add",
            category=OpCategory.ARITHMETIC,
            input_types=["Tensor", "Tensor"],
            output_types=["Tensor"],
        )
        assert op.attributes == {}


# =============================================================================
# LayerInfo Tests
# =============================================================================


class TestLayerInfo:
    """Tests for LayerInfo dataclass."""

    def test_create_layer_info(self):
        """Test creating LayerInfo instance."""
        layer = LayerInfo(
            name="encoder.layer.0.self_attn",
            layer_type="MultiheadAttention",
            input_shapes=[(1, 512, 768)],
            output_shapes=[(1, 512, 768)],
            params={"in_proj_weight": (2304, 768), "out_proj.weight": (768, 768)},
            attributes={"num_heads": 12, "embed_dim": 768},
        )
        assert layer.name == "encoder.layer.0.self_attn"
        assert layer.layer_type == "MultiheadAttention"
        assert layer.attributes["num_heads"] == 12

    def test_layer_info_default_attributes(self):
        """Test LayerInfo default attributes is empty dict."""
        layer = LayerInfo(
            name="fc",
            layer_type="Linear",
            input_shapes=[],
            output_shapes=[],
            params={"weight": (10, 20)},
        )
        assert layer.attributes == {}


# =============================================================================
# WeightInfo Tests
# =============================================================================


class TestWeightInfo:
    """Tests for WeightInfo dataclass."""

    def test_create_weight_info(self):
        """Test creating WeightInfo instance."""
        weight = WeightInfo(
            name="encoder.weight",
            shape=(768, 512),
            dtype="torch.float32",
            requires_grad=True,
            size_bytes=768 * 512 * 4,
        )
        assert weight.name == "encoder.weight"
        assert weight.shape == (768, 512)
        assert weight.requires_grad is True

    def test_weight_info_buffer(self):
        """Test WeightInfo for non-trainable buffer."""
        weight = WeightInfo(
            name="running_mean",
            shape=(256,),
            dtype="torch.float32",
            requires_grad=False,
            size_bytes=256 * 4,
        )
        assert weight.requires_grad is False


# =============================================================================
# ModelArchitecture Tests
# =============================================================================


class TestModelArchitecture:
    """Tests for ModelArchitecture dataclass."""

    def test_create_model_architecture(self):
        """Test creating ModelArchitecture instance."""
        arch = ModelArchitecture(
            name="test_model",
            layers=[],
            ops=[],
            weights=[],
            input_shapes=[(1, 3, 224, 224)],
            output_shapes=[(1, 1000)],
            total_params=25000000,
            total_size_bytes=100000000,
        )
        assert arch.name == "test_model"
        assert arch.total_params == 25000000
        assert arch.input_shapes == [(1, 3, 224, 224)]


# =============================================================================
# TorchScriptAnalyzer Categorization Tests
# =============================================================================


class TestTorchScriptAnalyzerCategorization:
    """Tests for TorchScriptAnalyzer operation categorization."""

    def test_categorize_layer_ops(self):
        """Test categorization of layer operations."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::linear") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::conv1d") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::conv2d") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::embedding") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::lstm") == OpCategory.LAYER

    def test_categorize_activation_ops(self):
        """Test categorization of activation operations."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::relu") == OpCategory.ACTIVATION
        assert analyzer._categorize_op("aten::gelu") == OpCategory.ACTIVATION
        assert analyzer._categorize_op("aten::sigmoid") == OpCategory.ACTIVATION
        assert analyzer._categorize_op("aten::softmax") == OpCategory.ACTIVATION

    def test_categorize_norm_ops(self):
        """Test categorization of normalization operations."""
        analyzer = TorchScriptAnalyzer()

        # Most norm ops are also in LAYER_OPS, so they return LAYER category
        # Only rms_norm is uniquely in NORM_OPS
        assert analyzer._categorize_op("aten::rms_norm") == OpCategory.NORMALIZATION
        # These are in LAYER_OPS so return LAYER (checked first)
        assert analyzer._categorize_op("aten::layer_norm") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::batch_norm") == OpCategory.LAYER

    def test_categorize_tensor_ops(self):
        """Test categorization of tensor operations."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::reshape") == OpCategory.TENSOR
        assert analyzer._categorize_op("aten::view") == OpCategory.TENSOR
        assert analyzer._categorize_op("aten::transpose") == OpCategory.TENSOR
        assert analyzer._categorize_op("aten::cat") == OpCategory.TENSOR

    def test_categorize_arithmetic_ops(self):
        """Test categorization of arithmetic operations."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::add") == OpCategory.ARITHMETIC
        assert analyzer._categorize_op("aten::matmul") == OpCategory.ARITHMETIC
        assert analyzer._categorize_op("aten::einsum") == OpCategory.ARITHMETIC

    def test_categorize_reduction_ops(self):
        """Test categorization of reduction operations."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::sum") == OpCategory.REDUCTION
        assert analyzer._categorize_op("aten::mean") == OpCategory.REDUCTION
        assert analyzer._categorize_op("aten::max") == OpCategory.REDUCTION

    def test_categorize_unknown_aten_op(self):
        """Test categorization of unknown aten operation."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("aten::unknown_op") == OpCategory.UNKNOWN

    def test_categorize_custom_op(self):
        """Test categorization of custom operation."""
        analyzer = TorchScriptAnalyzer()

        assert analyzer._categorize_op("custom::my_op") == OpCategory.CUSTOM


# =============================================================================
# TorchScriptAnalyzer Loading Tests
# =============================================================================


class TestTorchScriptAnalyzerLoading:
    """Tests for TorchScriptAnalyzer model loading."""

    def test_init_without_model(self):
        """Test initializing analyzer without model."""
        analyzer = TorchScriptAnalyzer()
        assert analyzer.model is None
        assert analyzer.model_path is None

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        analyzer = TorchScriptAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.load("/nonexistent/path/model.pt")

    def test_get_ops_without_model(self):
        """Test get_ops raises error without model loaded."""
        analyzer = TorchScriptAnalyzer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            analyzer.get_ops()

    def test_get_weights_without_model(self):
        """Test get_weights raises error without model loaded."""
        analyzer = TorchScriptAnalyzer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            analyzer.get_weights()

    def test_get_architecture_without_model(self):
        """Test get_architecture raises error without model loaded."""
        analyzer = TorchScriptAnalyzer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            analyzer.get_architecture()


# =============================================================================
# TorchScriptAnalyzer Integration Tests
# =============================================================================


class TestTorchScriptAnalyzerIntegration:
    """Integration tests for TorchScriptAnalyzer with real models."""

    @pytest.fixture
    def simple_linear_model(self, tmp_path):
        """Create a simple linear model for testing."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleModel()
        model.eval()
        # Use trace instead of script for Python 3.14 compatibility
        example_input = torch.randn(1, 10)
        traced = torch.jit.trace(model, example_input)
        model_path = tmp_path / "simple_model.pt"
        traced.save(str(model_path))
        return model_path

    @pytest.fixture
    def conv_model(self, tmp_path):
        """Create a conv model for testing."""

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.fc = nn.Linear(16 * 8 * 8, 10)

            def forward(self, x):
                x = torch.relu(self.bn(self.conv(x)))
                x = nn.functional.adaptive_avg_pool2d(x, (8, 8))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = ConvModel()
        model.eval()
        # Use trace instead of script for Python 3.14 compatibility
        example_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, example_input)
        model_path = tmp_path / "conv_model.pt"
        traced.save(str(model_path))
        return model_path

    def test_load_simple_model(self, simple_linear_model):
        """Test loading a simple TorchScript model."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))

        assert analyzer.model is not None
        assert analyzer.model_path == simple_linear_model

    def test_get_weights_simple_model(self, simple_linear_model):
        """Test getting weights from simple model."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        weights = analyzer.get_weights()

        # Should have weights and biases for both linear layers
        assert len(weights) >= 4

        weight_names = [w.name for w in weights]
        assert any("fc1" in name and "weight" in name for name in weight_names)
        assert any("fc2" in name and "weight" in name for name in weight_names)

    def test_get_weight_mapping(self, simple_linear_model):
        """Test getting weight mapping from model."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        mapping = analyzer.get_weight_mapping()

        assert isinstance(mapping, dict)
        assert all(isinstance(v, WeightInfo) for v in mapping.values())

    def test_get_ops_simple_model(self, simple_linear_model):
        """Test getting operations from simple model."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        ops = analyzer.get_ops()

        assert len(ops) > 0
        op_names = [op.name for op in ops]
        # Should have linear and relu operations
        assert any("linear" in name.lower() for name in op_names)

    def test_get_architecture_simple_model(self, simple_linear_model):
        """Test getting architecture from simple model."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        arch = analyzer.get_architecture()

        assert arch.name == "simple_model"
        assert arch.total_params > 0
        assert arch.total_size_bytes > 0

    def test_get_architecture_with_sample_input(self, simple_linear_model):
        """Test getting architecture with sample input for shape inference."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        sample_input = torch.randn(1, 10)
        arch = analyzer.get_architecture(sample_input=sample_input)

        assert arch.input_shapes == [(1, 10)]
        assert len(arch.output_shapes) > 0

    def test_summarize_model(self, simple_linear_model):
        """Test model summarization."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        summary = analyzer.summarize()

        assert "Model: simple_model" in summary
        assert "Total Parameters:" in summary
        assert "Operations:" in summary

    def test_get_unsupported_ops(self, simple_linear_model):
        """Test getting unsupported operations."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))
        unsupported = analyzer.get_unsupported_ops()

        # Simple model should have mostly supported ops
        assert isinstance(unsupported, list)

    def test_ops_caching(self, simple_linear_model):
        """Test that ops are cached after first call."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))

        ops1 = analyzer.get_ops()
        ops2 = analyzer.get_ops()

        # The cache returns a new list each time, but the underlying cache dict is reused
        # Verify by checking the internal cache is populated and values match
        assert analyzer._ops_cache  # Cache should be populated
        assert ops1 == ops2  # Values should be equal

    def test_weights_caching(self, simple_linear_model):
        """Test that weights are cached after first call."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))

        weights1 = analyzer.get_weights()
        weights2 = analyzer.get_weights()

        # Should return same cached values
        assert weights1 is weights2

    def test_load_clears_caches(self, simple_linear_model, conv_model):
        """Test that loading a new model clears caches."""
        analyzer = TorchScriptAnalyzer(str(simple_linear_model))

        # Populate caches
        _ = analyzer.get_ops()
        _ = analyzer.get_weights()

        # Verify caches are populated
        assert len(analyzer._ops_cache) > 0
        assert len(analyzer._weights_cache) > 0
        assert len(analyzer._layers_cache) == 0  # Not yet populated

        # Load different model
        analyzer.load(str(conv_model))

        # Cache should be cleared after load
        assert len(analyzer._ops_cache) == 0
        assert len(analyzer._weights_cache) == 0
        assert len(analyzer._layers_cache) == 0

        # Model path should change
        assert analyzer.model_path == conv_model

        # New caches should be populated after get calls
        _ = analyzer.get_ops()
        _ = analyzer.get_weights()
        assert len(analyzer._ops_cache) > 0
        assert len(analyzer._weights_cache) > 0

    def test_conv_model_ops(self, conv_model):
        """Test analyzing conv model operations."""
        analyzer = TorchScriptAnalyzer(str(conv_model))
        ops = analyzer.get_ops()

        op_names = [op.name for op in ops]
        # Should have conv and batch norm related operations
        assert any("conv" in name.lower() for name in op_names)

    def test_conv_model_weights(self, conv_model):
        """Test analyzing conv model weights."""
        analyzer = TorchScriptAnalyzer(str(conv_model))
        weights = analyzer.get_weights()

        # Should have conv weights, bn weights, and linear weights
        weight_names = [w.name for w in weights]
        assert any("conv" in name for name in weight_names)
        assert any("bn" in name for name in weight_names)
        assert any("fc" in name for name in weight_names)


class TestTorchScriptAnalyzerLayerExtraction:
    """Tests for layer extraction from TorchScript models."""

    @pytest.fixture
    def multi_layer_model(self, tmp_path):
        """Create a multi-layer model for testing."""

        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 64)
                self.encoder = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )
                self.output = nn.Linear(64, 10)

            def forward(self, x):
                x = self.embedding(x)
                x = self.encoder(x)
                return self.output(x.mean(dim=1))

        model = MultiLayerModel()
        model.eval()
        # Use trace instead of script for Python 3.14 compatibility
        example_input = torch.randint(0, 1000, (1, 10))
        traced = torch.jit.trace(model, example_input)
        model_path = tmp_path / "multi_layer_model.pt"
        traced.save(str(model_path))
        return model_path

    def test_extract_layers(self, multi_layer_model):
        """Test extracting layers from multi-layer model."""
        analyzer = TorchScriptAnalyzer(str(multi_layer_model))
        arch = analyzer.get_architecture()

        # Should have multiple layers
        assert len(arch.layers) > 0

        layer_names = [layer.name for layer in arch.layers]
        # Check some expected layer names exist
        assert any("embedding" in name for name in layer_names)
        assert any("encoder" in name for name in layer_names)
        assert any("output" in name for name in layer_names)

    def test_layer_attributes(self, multi_layer_model):
        """Test that layer attributes are extracted."""
        analyzer = TorchScriptAnalyzer(str(multi_layer_model))
        arch = analyzer.get_architecture()

        # Find the embedding layer
        embedding_layers = [
            layer for layer in arch.layers if "embedding" in layer.name.lower()
        ]
        if embedding_layers:
            layer = embedding_layers[0]
            # Embedding should have params
            assert len(layer.params) > 0


class TestTorchScriptAnalyzerTotalParams:
    """Tests for total parameter counting."""

    @pytest.fixture
    def model_with_buffers(self, tmp_path):
        """Create a model with both parameters and buffers."""

        class ModelWithBuffers(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)  # 10*10 + 10 = 110 trainable params
                self.bn = nn.BatchNorm1d(10)  # Has running_mean, running_var buffers

            def forward(self, x):
                return self.bn(self.fc(x))

        model = ModelWithBuffers()
        model.eval()
        # Use trace instead of script for Python 3.14 compatibility
        example_input = torch.randn(2, 10)  # batch size > 1 for BatchNorm
        traced = torch.jit.trace(model, example_input)
        model_path = tmp_path / "model_with_buffers.pt"
        traced.save(str(model_path))
        return model_path

    def test_total_params_excludes_buffers(self, model_with_buffers):
        """Test that total_params only counts trainable parameters."""
        analyzer = TorchScriptAnalyzer(str(model_with_buffers))
        arch = analyzer.get_architecture()

        # Get only trainable weights
        trainable_weights = [w for w in arch.weights if w.requires_grad]

        # Calculate expected trainable params
        expected_trainable = sum(
            np.prod(w.shape) if w.shape else 1 for w in trainable_weights
        )

        assert arch.total_params == expected_trainable

    def test_weights_include_buffers(self, model_with_buffers):
        """Test that get_weights() returns both params and buffers."""
        analyzer = TorchScriptAnalyzer(str(model_with_buffers))
        weights = analyzer.get_weights()

        # Should have both trainable and non-trainable weights
        trainable = [w for w in weights if w.requires_grad]
        non_trainable = [w for w in weights if not w.requires_grad]

        assert len(trainable) > 0
        assert len(non_trainable) > 0  # BatchNorm has running_mean, running_var
