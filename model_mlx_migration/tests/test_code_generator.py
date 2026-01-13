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

"""Tests for the MLX Code Generator module."""

import sys
from pathlib import Path

import pytest

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from pytorch_to_mlx.analyzer.torchscript_analyzer import (
    LayerInfo,
    ModelArchitecture,
    OpCategory,
    OpInfo,
    WeightInfo,
)
from pytorch_to_mlx.generator.mlx_code_generator import GeneratedModel, MLXCodeGenerator


class TestMLXCodeGenerator:
    """Test suite for MLXCodeGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return MLXCodeGenerator()

    @pytest.fixture
    def simple_architecture(self) -> ModelArchitecture:
        """Create a simple model architecture for testing."""
        layers = [
            LayerInfo(
                name="encoder.embedding",
                layer_type="Embedding",
                input_shapes=[],
                output_shapes=[],
                params={"weight": (10000, 512)},
                attributes={"num_embeddings": 10000, "embedding_dim": 512},
            ),
            LayerInfo(
                name="encoder.linear",
                layer_type="Linear",
                input_shapes=[],
                output_shapes=[],
                params={"weight": (512, 256), "bias": (256,)},
                attributes={"in_features": 512, "out_features": 256, "bias": True},
            ),
            LayerInfo(
                name="encoder.norm",
                layer_type="LayerNorm",
                input_shapes=[],
                output_shapes=[],
                params={"weight": (256,), "bias": (256,)},
                attributes={"normalized_shape": 256, "eps": 1e-5},
            ),
        ]

        ops = [
            OpInfo(
                name="aten::embedding",
                category=OpCategory.LAYER,
                input_types=[],
                output_types=[],
                count=1,
            ),
            OpInfo(
                name="aten::linear",
                category=OpCategory.LAYER,
                input_types=[],
                output_types=[],
                count=1,
            ),
            OpInfo(
                name="aten::layer_norm",
                category=OpCategory.NORMALIZATION,
                input_types=[],
                output_types=[],
                count=1,
            ),
            OpInfo(
                name="aten::relu",
                category=OpCategory.ACTIVATION,
                input_types=[],
                output_types=[],
                count=1,
            ),
        ]

        weights = [
            WeightInfo(
                name="encoder.embedding.weight",
                shape=(10000, 512),
                dtype="torch.float32",
                requires_grad=True,
                size_bytes=10000 * 512 * 4,
            ),
            WeightInfo(
                name="encoder.linear.weight",
                shape=(512, 256),
                dtype="torch.float32",
                requires_grad=True,
                size_bytes=512 * 256 * 4,
            ),
            WeightInfo(
                name="encoder.linear.bias",
                shape=(256,),
                dtype="torch.float32",
                requires_grad=True,
                size_bytes=256 * 4,
            ),
            WeightInfo(
                name="encoder.norm.weight",
                shape=(256,),
                dtype="torch.float32",
                requires_grad=True,
                size_bytes=256 * 4,
            ),
            WeightInfo(
                name="encoder.norm.bias",
                shape=(256,),
                dtype="torch.float32",
                requires_grad=True,
                size_bytes=256 * 4,
            ),
        ]

        return ModelArchitecture(
            name="TestModel",
            layers=layers,
            ops=ops,
            weights=weights,
            input_shapes=[(1, 512)],
            output_shapes=[(1, 256)],
            total_params=10000 * 512 + 512 * 256 + 256 + 256 + 256,
            total_size_bytes=(10000 * 512 + 512 * 256 + 256 + 256 + 256) * 4,
        )

    def test_init(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.op_mapper is not None

    def test_generate_basic(self, generator, simple_architecture):
        """Test basic code generation."""
        result = generator.generate(simple_architecture)

        assert isinstance(result, GeneratedModel)
        assert result.model_code is not None
        assert result.config_code is not None
        assert result.weight_names is not None

    def test_generate_model_class(self, generator, simple_architecture):
        """Test model class code generation."""
        result = generator.generate(simple_architecture, class_name="TestMLX")

        # Check class name
        assert "class TestMLX(nn.Module):" in result.model_code

        # Check init method
        assert "def __init__" in result.model_code
        assert "super().__init__()" in result.model_code

        # Check call method
        assert "def __call__" in result.model_code

    def test_generate_config(self, generator, simple_architecture):
        """Test config generation."""
        result = generator.generate(simple_architecture, class_name="TestMLX")

        # Check dataclass
        assert "@dataclass" in result.config_code
        assert "class TestMLXConfig:" in result.config_code

    def test_generate_layers(self, generator, simple_architecture):
        """Test layer generation."""
        result = generator.generate(simple_architecture)

        # Should generate layer initialization
        assert "nn.Embedding" in result.model_code or "Embedding" in result.model_code
        assert "nn.Linear" in result.model_code or "Linear" in result.model_code
        assert "nn.LayerNorm" in result.model_code or "LayerNorm" in result.model_code

    def test_weight_mapping(self, generator, simple_architecture):
        """Test weight name mapping."""
        result = generator.generate(simple_architecture)

        # All original weights should have mappings
        for weight in simple_architecture.weights:
            assert weight.name in result.weight_names

    def test_generate_stub(self, generator, simple_architecture):
        """Test stub generation."""
        stub = generator.generate_stub(simple_architecture)

        assert "class TestModelMLX(nn.Module):" in stub
        assert "raise NotImplementedError()" in stub
        assert "Operation coverage:" in stub

    def test_imports(self, generator, simple_architecture):
        """Test required imports are tracked."""
        result = generator.generate(simple_architecture)

        assert "mlx.core as mx" in result.imports
        assert "mlx.nn as nn" in result.imports

    def test_helper_code_for_custom_ops(self, generator):
        """Test helper code is generated for custom ops."""
        # Create architecture with custom op
        ops = [
            OpInfo(
                name="aten::stft",
                category=OpCategory.CUSTOM,
                input_types=[],
                output_types=[],
                count=1,
            ),
        ]
        arch = ModelArchitecture(
            name="CustomOpModel",
            layers=[],
            ops=ops,
            weights=[],
            input_shapes=[],
            output_shapes=[],
            total_params=0,
            total_size_bytes=0,
        )

        result = generator.generate(arch)

        # Should have helper code for STFT
        assert (
            "stft_mlx" in result.helper_code
            or "Custom implementations" in result.helper_code
        )

    def test_sanitize_name(self, generator):
        """Test name sanitization."""
        # Test various name patterns
        assert generator._sanitize_name("model_v1.pt") == "ModelV1"
        assert generator._sanitize_name("my-model") == "MyModel"
        assert generator._sanitize_name("123_model") == "_123Model"

    def test_dtype_mappings(self, generator):
        """Test dtype mapping dictionary."""
        dtypes = generator.DTYPE_MAPPINGS

        assert "torch.float32" in dtypes
        assert "torch.float16" in dtypes
        assert dtypes["torch.float32"] == "mx.float32"

    def test_layer_mappings(self, generator):
        """Test layer mapping dictionary."""
        mappings = generator.LAYER_MAPPINGS

        assert "Linear" in mappings
        assert "Conv2d" in mappings
        assert "LayerNorm" in mappings
        assert mappings["Linear"] == "nn.Linear"


class TestGeneratedModel:
    """Test GeneratedModel dataclass."""

    def test_dataclass_fields(self):
        """Test GeneratedModel has expected fields."""
        model = GeneratedModel(
            model_code="class Test: pass",
            helper_code="",
            config_code="@dataclass\nclass Config: pass",
            weight_names={"a": "b"},
            imports={"mlx.core"},
        )

        assert model.model_code is not None
        assert model.helper_code == ""
        assert model.config_code is not None
        assert model.weight_names == {"a": "b"}
        assert "mlx.core" in model.imports


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
