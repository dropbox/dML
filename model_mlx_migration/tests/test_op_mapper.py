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

"""Tests for the Op Mapper module."""

import sys
from pathlib import Path

import pytest

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from pytorch_to_mlx.analyzer.op_mapper import MappingType, OpMapper


class TestOpMapper:
    """Test suite for OpMapper."""

    def test_init(self):
        """Test OpMapper initialization."""
        mapper = OpMapper()
        assert mapper is not None

    def test_direct_mapping(self):
        """Test direct operation mapping."""
        mapper = OpMapper()

        # Test a common direct mapping
        mapping = mapper.map_op("aten::linear")
        assert mapping.mapping_type == MappingType.DIRECT
        assert mapping.mlx_op == "mx.nn.Linear"

    def test_activation_mapping(self):
        """Test activation function mapping."""
        mapper = OpMapper()

        activations = ["aten::relu", "aten::gelu", "aten::sigmoid", "aten::tanh"]
        for op in activations:
            mapping = mapper.map_op(op)
            assert mapping.mapping_type == MappingType.DIRECT, (
                f"{op} should be direct mapping"
            )

    def test_decomposed_mapping(self):
        """Test decomposed operation mapping."""
        mapper = OpMapper()

        # Test addmm (commonly decomposed)
        mapping = mapper.map_op("aten::addmm")
        assert mapping.mapping_type == MappingType.DECOMPOSED
        assert mapping.decomposition is not None
        assert "def addmm_mlx" in mapping.decomposition

    def test_attention_decomposition(self):
        """Test attention operations are decomposed."""
        mapper = OpMapper()

        mapping = mapper.map_op("aten::scaled_dot_product_attention")
        assert mapping.mapping_type == MappingType.DECOMPOSED
        assert "softmax" in mapping.decomposition.lower()

    def test_custom_op_mapping(self):
        """Test custom operation mapping."""
        mapper = OpMapper()

        # STFT requires custom implementation
        mapping = mapper.map_op("aten::stft")
        assert mapping.mapping_type == MappingType.CUSTOM
        assert mapping.custom_impl is not None

    def test_unsupported_op(self):
        """Test handling of unsupported operations."""
        mapper = OpMapper()

        # Made-up operation should be unsupported
        mapping = mapper.map_op("aten::nonexistent_op")
        assert mapping.mapping_type == MappingType.UNSUPPORTED

    def test_noop_mapping(self):
        """Test operations that become no-ops in MLX."""
        mapper = OpMapper()

        # Contiguous is a no-op in MLX due to lazy evaluation
        mapping = mapper.map_op("aten::contiguous")
        assert mapping.mapping_type == MappingType.DIRECT
        assert mapping.mlx_op == ""  # Empty means no-op

    def test_tensor_operations(self):
        """Test tensor manipulation operations."""
        mapper = OpMapper()

        tensor_ops = {
            "aten::reshape": "mx.reshape",
            "aten::transpose": "mx.transpose",
            "aten::squeeze": "mx.squeeze",
            "aten::unsqueeze": "mx.expand_dims",
            "aten::cat": "mx.concatenate",
        }

        for torch_op, expected_mlx in tensor_ops.items():
            mapping = mapper.map_op(torch_op)
            assert mapping.mlx_op == expected_mlx, (
                f"{torch_op} should map to {expected_mlx}"
            )

    def test_arithmetic_operations(self):
        """Test arithmetic operations mapping."""
        mapper = OpMapper()

        arith_ops = {
            "aten::add": "mx.add",
            "aten::sub": "mx.subtract",
            "aten::mul": "mx.multiply",
            "aten::div": "mx.divide",
            "aten::matmul": "mx.matmul",
        }

        for torch_op, expected_mlx in arith_ops.items():
            mapping = mapper.map_op(torch_op)
            assert mapping.mlx_op == expected_mlx

    def test_reduction_operations(self):
        """Test reduction operations mapping."""
        mapper = OpMapper()

        reduction_ops = {
            "aten::sum": "mx.sum",
            "aten::mean": "mx.mean",
            "aten::max": "mx.max",
            "aten::argmax": "mx.argmax",
        }

        for torch_op, expected_mlx in reduction_ops.items():
            mapping = mapper.map_op(torch_op)
            assert mapping.mlx_op == expected_mlx

    def test_coverage_report(self):
        """Test coverage report generation."""
        mapper = OpMapper()

        ops = ["aten::linear", "aten::relu", "aten::stft", "aten::unknown_op"]
        report = mapper.get_coverage_report(ops)

        assert "total_ops" in report
        assert "supported_ops" in report
        assert "coverage_percent" in report
        assert report["total_ops"] == 4
        assert report["supported_ops"] == 3  # linear, relu, stft
        assert report["unsupported_ops"] == 1  # unknown_op

    def test_generate_conversion_code(self):
        """Test code generation for custom ops."""
        mapper = OpMapper()

        ops = ["aten::addmm", "aten::stft"]
        code = mapper.generate_conversion_code(ops)

        assert "import mlx.core as mx" in code
        assert "addmm_mlx" in code
        assert "stft_mlx" in code

    def test_list_supported_ops(self):
        """Test listing all supported operations."""
        supported = OpMapper.list_supported_ops()

        assert len(supported) > 50  # Should have many supported ops
        assert "aten::linear" in supported
        assert "aten::relu" in supported
        assert "aten::matmul" in supported

    def test_caching(self):
        """Test that mappings are cached."""
        mapper = OpMapper()

        # First call
        mapping1 = mapper.map_op("aten::linear")
        # Second call should return cached result
        mapping2 = mapper.map_op("aten::linear")

        assert mapping1 is mapping2  # Same object

    def test_batch_mapping(self):
        """Test batch mapping of operations."""
        mapper = OpMapper()

        ops = ["aten::linear", "aten::relu", "aten::softmax"]
        mappings = mapper.get_all_mappings(ops)

        assert len(mappings) == 3
        assert all(m.mapping_type == MappingType.DIRECT for m in mappings.values())


class TestMappingType:
    """Test MappingType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert MappingType.DIRECT.value == "direct"
        assert MappingType.RENAMED.value == "renamed"
        assert MappingType.DECOMPOSED.value == "decomposed"
        assert MappingType.CUSTOM.value == "custom"
        assert MappingType.UNSUPPORTED.value == "unsupported"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
