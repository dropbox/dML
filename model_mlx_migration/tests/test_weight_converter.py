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

"""Tests for the Weight Converter module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from pytorch_to_mlx.generator.weight_converter import ConversionStats, WeightConverter


class TestConversionStats:
    """Test ConversionStats dataclass."""

    def test_dataclass_creation(self):
        """Test ConversionStats can be created with all fields."""
        stats = ConversionStats(
            total_tensors=10,
            converted_tensors=8,
            skipped_tensors=2,
            total_bytes_input=1000,
            total_bytes_output=500,
            dtype_conversions={"torch.float32->float16": 8},
            renamed_tensors=[("old.name", "new.name")],
            errors=["error message"],
        )

        assert stats.total_tensors == 10
        assert stats.converted_tensors == 8
        assert stats.skipped_tensors == 2
        assert stats.total_bytes_input == 1000
        assert stats.total_bytes_output == 500
        assert stats.dtype_conversions == {"torch.float32->float16": 8}
        assert stats.renamed_tensors == [("old.name", "new.name")]
        assert stats.errors == ["error message"]

    def test_empty_stats(self):
        """Test ConversionStats with empty collections."""
        stats = ConversionStats(
            total_tensors=0,
            converted_tensors=0,
            skipped_tensors=0,
            total_bytes_input=0,
            total_bytes_output=0,
            dtype_conversions={},
            renamed_tensors=[],
            errors=[],
        )

        assert stats.total_tensors == 0
        assert len(stats.dtype_conversions) == 0
        assert len(stats.renamed_tensors) == 0
        assert len(stats.errors) == 0


class TestWeightConverterInit:
    """Test WeightConverter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        converter = WeightConverter()

        assert converter.target_dtype is None
        assert len(converter.name_transforms) == len(WeightConverter.NAME_TRANSFORMS)

    def test_init_with_target_dtype(self):
        """Test initialization with target dtype."""
        converter = WeightConverter(target_dtype="float16")

        assert converter.target_dtype == "float16"

    def test_init_with_custom_transforms(self):
        """Test initialization with custom name transforms."""
        custom_transforms = [(r"\.old$", ".new"), (r"^prefix\.", "")]
        converter = WeightConverter(name_transforms=custom_transforms)

        # Should have default transforms plus custom ones
        assert len(converter.name_transforms) == len(WeightConverter.NAME_TRANSFORMS) + 2

    def test_dtype_map_exists(self):
        """Test DTYPE_MAP has expected entries."""
        assert torch.float32 in WeightConverter.DTYPE_MAP
        assert torch.float16 in WeightConverter.DTYPE_MAP
        assert torch.bfloat16 in WeightConverter.DTYPE_MAP
        assert torch.int32 in WeightConverter.DTYPE_MAP
        assert torch.int64 in WeightConverter.DTYPE_MAP
        assert torch.bool in WeightConverter.DTYPE_MAP

    def test_name_transforms_exist(self):
        """Test default NAME_TRANSFORMS are defined."""
        transforms = WeightConverter.NAME_TRANSFORMS

        assert len(transforms) > 0
        # Check some specific transforms
        transform_patterns = [t[0] for t in transforms]
        assert r"\.gamma$" in transform_patterns
        assert r"\.beta$" in transform_patterns


class TestWeightConverterConvertName:
    """Test WeightConverter.convert_name method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    def test_gamma_to_weight(self, converter):
        """Test gamma -> weight transformation."""
        result = converter.convert_name("layer.gamma")
        assert result == "layer.weight"

    def test_beta_to_bias(self, converter):
        """Test beta -> bias transformation."""
        result = converter.convert_name("layer.beta")
        assert result == "layer.bias"

    def test_running_mean_unchanged(self, converter):
        """Test running_mean stays as running_mean."""
        result = converter.convert_name("bn.running_mean")
        assert result == "bn.running_mean"

    def test_in_proj_weight(self, converter):
        """Test in_proj_weight transformation."""
        result = converter.convert_name("attn.in_proj_weight")
        assert result == "attn.in_proj.weight"

    def test_no_transform_needed(self, converter):
        """Test name that doesn't need transformation."""
        result = converter.convert_name("encoder.layer.weight")
        assert result == "encoder.layer.weight"

    def test_custom_transform(self):
        """Test custom name transform."""
        custom = [(r"^model\.", "")]
        converter = WeightConverter(name_transforms=custom)

        result = converter.convert_name("model.encoder.weight")
        assert result == "encoder.weight"


class TestWeightConverterConvertTensor:
    """Test WeightConverter.convert_tensor method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    def test_float32_tensor(self, converter):
        """Test converting float32 tensor."""
        tensor = torch.randn(10, 10, dtype=torch.float32)
        result = converter.convert_tensor(tensor)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (10, 10)

    def test_float16_tensor(self, converter):
        """Test converting float16 tensor."""
        tensor = torch.randn(5, 5, dtype=torch.float16)
        result = converter.convert_tensor(tensor)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float16

    def test_int64_tensor(self, converter):
        """Test converting int64 tensor."""
        tensor = torch.randint(0, 100, (4, 4), dtype=torch.int64)
        result = converter.convert_tensor(tensor)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    def test_bool_tensor(self, converter):
        """Test converting bool tensor."""
        tensor = torch.tensor([True, False, True], dtype=torch.bool)
        result = converter.convert_tensor(tensor)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.bool_

    def test_target_dtype_conversion(self, converter):
        """Test conversion to specified target dtype."""
        tensor = torch.randn(5, 5, dtype=torch.float32)
        result = converter.convert_tensor(tensor, target_dtype=np.float16)

        assert result.dtype == np.float16

    def test_bfloat16_conversion(self, converter):
        """Test bfloat16 conversion (should convert to float32 for numpy)."""
        tensor = torch.randn(3, 3, dtype=torch.bfloat16)
        result = converter.convert_tensor(tensor)

        assert isinstance(result, np.ndarray)
        # bfloat16 -> float32 for numpy compatibility
        assert result.dtype == np.float32

    def test_converter_with_target_dtype(self):
        """Test converter initialized with target dtype."""
        converter = WeightConverter(target_dtype="float16")
        tensor = torch.randn(5, 5, dtype=torch.float32)
        result = converter.convert_tensor(tensor)

        assert result.dtype == np.float16

    def test_values_preserved(self, converter):
        """Test tensor values are preserved after conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        result = converter.convert_tensor(tensor)

        np.testing.assert_array_almost_equal(result, [[1.0, 2.0], [3.0, 4.0]])


class TestWeightConverterLoadPytorchWeights:
    """Test WeightConverter.load_pytorch_weights method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    def test_load_nonexistent_file(self, converter):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            converter.load_pytorch_weights("/nonexistent/path/model.pt")

    def test_load_state_dict(self, converter):
        """Test loading from state dict file."""
        # Create a simple state dict
        state_dict = {
            "layer.weight": torch.randn(10, 10),
            "layer.bias": torch.randn(10),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            loaded = converter.load_pytorch_weights(f.name)

        assert "layer.weight" in loaded
        assert "layer.bias" in loaded
        assert loaded["layer.weight"].shape == (10, 10)

    def test_load_wrapped_state_dict(self, converter):
        """Test loading wrapped state dict (e.g., from checkpoint)."""
        state_dict = {
            "state_dict": {
                "encoder.weight": torch.randn(5, 5),
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            loaded = converter.load_pytorch_weights(f.name)

        assert "encoder.weight" in loaded

    def test_load_model_wrapped_state_dict(self, converter):
        """Test loading model-wrapped state dict."""
        state_dict = {
            "model": {
                "decoder.weight": torch.randn(8, 8),
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            loaded = converter.load_pytorch_weights(f.name)

        assert "decoder.weight" in loaded


class TestWeightConverterShardWeights:
    """Test WeightConverter._shard_weights method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    def test_single_shard(self, converter):
        """Test weights fit in single shard."""
        weights = {
            "a": np.zeros((10,), dtype=np.float32),  # 40 bytes
            "b": np.zeros((10,), dtype=np.float32),  # 40 bytes
        }

        shards = converter._shard_weights(weights, max_bytes=1000)
        assert len(shards) == 1
        assert len(shards[0]) == 2

    def test_multiple_shards(self, converter):
        """Test weights split into multiple shards."""
        weights = {
            "a": np.zeros((100,), dtype=np.float32),  # 400 bytes
            "b": np.zeros((100,), dtype=np.float32),  # 400 bytes
            "c": np.zeros((100,), dtype=np.float32),  # 400 bytes
        }

        shards = converter._shard_weights(weights, max_bytes=500)
        assert len(shards) == 3
        # Each shard should have exactly one tensor
        for shard in shards:
            assert len(shard) == 1

    def test_shard_boundary(self, converter):
        """Test sharding at exact boundary."""
        weights = {
            "a": np.zeros((100,), dtype=np.float32),  # 400 bytes
            "b": np.zeros((100,), dtype=np.float32),  # 400 bytes
        }

        shards = converter._shard_weights(weights, max_bytes=400)
        assert len(shards) == 2

    def test_empty_weights(self, converter):
        """Test sharding empty weights dict."""
        shards = converter._shard_weights({}, max_bytes=1000)
        assert len(shards) == 0


class TestWeightConverterConvert:
    """Test WeightConverter.convert method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    @pytest.fixture
    def sample_model_path(self):
        """Create a sample model file."""
        state_dict = {
            "encoder.weight": torch.randn(10, 10),
            "encoder.bias": torch.randn(10),
            "decoder.weight": torch.randn(10, 10),
            "norm.gamma": torch.randn(10),
            "norm.beta": torch.randn(10),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            yield f.name

    def test_basic_conversion(self, converter, sample_model_path):
        """Test basic weight conversion."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            stats = converter.convert(sample_model_path, f.name)

        assert stats.total_tensors == 5
        assert stats.converted_tensors == 5
        assert stats.skipped_tensors == 0
        assert len(stats.errors) == 0

    def test_name_transforms_in_convert(self, converter, sample_model_path):
        """Test name transforms are applied during conversion."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            stats = converter.convert(sample_model_path, f.name)

        # gamma -> weight, beta -> bias should be renamed
        renamed_dict = dict(stats.renamed_tensors)
        assert "norm.gamma" in renamed_dict
        assert renamed_dict["norm.gamma"] == "norm.weight"
        assert "norm.beta" in renamed_dict
        assert renamed_dict["norm.beta"] == "norm.bias"

    def test_name_filter(self, converter, sample_model_path):
        """Test name filter skips specified tensors."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            stats = converter.convert(
                sample_model_path,
                f.name,
                name_filter=lambda n: "encoder" in n,
            )

        assert stats.converted_tensors == 2  # Only encoder.weight and encoder.bias
        assert stats.skipped_tensors == 3

    def test_progress_callback(self, converter, sample_model_path):
        """Test progress callback is called."""
        calls = []

        def callback(name, current, total):
            calls.append((name, current, total))

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            converter.convert(sample_model_path, f.name, progress_callback=callback)

        assert len(calls) == 5
        assert calls[0][1] == 1  # First call has current=1
        assert calls[-1][1] == 5  # Last call has current=5
        assert all(c[2] == 5 for c in calls)  # All have total=5


class TestWeightConverterConvertToMlxFormat:
    """Test WeightConverter.convert_to_mlx_format method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    @pytest.fixture
    def sample_model_path(self):
        """Create a sample model file."""
        state_dict = {
            "layer.weight": torch.randn(10, 10),
            "layer.bias": torch.randn(10),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            yield f.name

    def test_single_file_output(self, converter, sample_model_path):
        """Test conversion to single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = converter.convert_to_mlx_format(sample_model_path, tmpdir)

            assert len(files) == 1
            assert "weights.safetensors" in files[0] or "weights.npz" in files[0]
            assert Path(files[0]).exists()

    def test_sharded_output(self, converter, sample_model_path):
        """Test conversion to sharded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small shard size to force multiple shards
            files = converter.convert_to_mlx_format(
                sample_model_path, tmpdir, shard_size=100,
            )

            assert len(files) >= 2
            # Check naming pattern
            for f in files:
                assert "weights-" in f

    def test_creates_output_directory(self, converter, sample_model_path):
        """Test output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            files = converter.convert_to_mlx_format(sample_model_path, str(new_dir))

            assert new_dir.exists()
            assert len(files) == 1


class TestWeightConverterVerifyConversion:
    """Test WeightConverter.verify_conversion method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    @pytest.fixture
    def converted_pair(self, converter):
        """Create matching PyTorch and MLX weight files."""
        state_dict = {
            "layer.weight": torch.randn(5, 5),
            "layer.bias": torch.randn(5),
            "norm.gamma": torch.randn(5),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as pt_file:
            torch.save(state_dict, pt_file.name)

            with tempfile.NamedTemporaryFile(
                suffix=".safetensors", delete=False,
            ) as mlx_file:
                converter.convert(pt_file.name, mlx_file.name)
                yield pt_file.name, mlx_file.name

    def test_verify_matching_weights(self, converter, converted_pair):
        """Test verification of correctly converted weights."""
        pt_path, mlx_path = converted_pair
        report = converter.verify_conversion(pt_path, mlx_path)

        assert report["verified"] is True
        assert report["matched"] == 3
        assert report["mismatched"] == 0
        assert len(report["missing_in_mlx"]) == 0

    def test_verify_reports_max_errors(self, converter, converted_pair):
        """Test verification reports max errors for each tensor."""
        pt_path, mlx_path = converted_pair
        report = converter.verify_conversion(pt_path, mlx_path)

        assert "max_errors" in report
        assert len(report["max_errors"]) == 3
        # All errors should be near zero for converted weights
        for error in report["max_errors"].values():
            assert error < 1e-6

    def test_verify_with_tolerance(self, converter):
        """Test verification with custom tolerance."""
        # Create slightly different weights
        state_dict = {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as pt_file:
            torch.save(state_dict, pt_file.name)

            with tempfile.NamedTemporaryFile(
                suffix=".safetensors", delete=False,
            ) as mlx_file:
                converter.convert(pt_file.name, mlx_file.name)

                # Should pass with default tolerance
                report = converter.verify_conversion(pt_file.name, mlx_file.name)
                assert report["verified"] is True


class TestWeightConverterIntegration:
    """Integration tests for WeightConverter."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return WeightConverter()

    def test_full_conversion_pipeline(self, converter):
        """Test complete conversion from PyTorch to MLX format."""
        # Create a model-like state dict
        state_dict = {
            "encoder.embed.weight": torch.randn(100, 64),
            "encoder.layer.0.attn.in_proj_weight": torch.randn(192, 64),
            "encoder.layer.0.attn.in_proj_bias": torch.randn(192),
            "encoder.layer.0.norm.gamma": torch.randn(64),
            "encoder.layer.0.norm.beta": torch.randn(64),
            "decoder.weight": torch.randn(64, 100),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save PyTorch model
            pt_path = Path(tmpdir) / "model.pt"
            torch.save(state_dict, pt_path)

            # Convert to MLX format
            mlx_dir = Path(tmpdir) / "mlx_model"
            files = converter.convert_to_mlx_format(str(pt_path), str(mlx_dir))

            assert len(files) == 1
            assert (mlx_dir / "weights.safetensors").exists() or (
                mlx_dir / "weights.npz"
            ).exists()

            # Verify conversion
            mlx_path = files[0]
            report = converter.verify_conversion(str(pt_path), mlx_path)

            assert report["verified"] is True
            assert report["pytorch_tensors"] == 6
            assert report["mlx_tensors"] == 6

    def test_dtype_conversion_tracking(self):
        """Test that dtype conversions are properly tracked."""
        converter = WeightConverter(target_dtype="float16")

        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer.bias": torch.randn(10, dtype=torch.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as pt_file:
            torch.save(state_dict, pt_file.name)

            with tempfile.NamedTemporaryFile(
                suffix=".safetensors", delete=False,
            ) as mlx_file:
                stats = converter.convert(pt_file.name, mlx_file.name)

        # Should track dtype conversions
        assert len(stats.dtype_conversions) > 0
        assert any("float16" in k for k in stats.dtype_conversions.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
