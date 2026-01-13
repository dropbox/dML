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
Weight Converter

Converts PyTorch model weights to MLX-compatible format (safetensors).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ConversionStats:
    """Statistics from weight conversion."""

    total_tensors: int
    converted_tensors: int
    skipped_tensors: int
    total_bytes_input: int
    total_bytes_output: int
    dtype_conversions: dict[str, int]  # e.g., {"float32->float16": 5}
    renamed_tensors: list[tuple[str, str]]  # (old_name, new_name)
    errors: list[str]


class WeightConverter:
    """
    Converts PyTorch model weights to MLX format.

    Handles:
    1. Loading weights from TorchScript or state_dict
    2. Converting tensor dtypes as needed
    3. Renaming parameters to MLX conventions
    4. Saving in safetensors format (MLX native)
    """

    # dtype conversions from PyTorch to MLX-compatible numpy
    DTYPE_MAP = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # MLX supports bfloat16, but safetensors may not
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }

    # Common weight name transformations (PyTorch -> MLX)
    NAME_TRANSFORMS = [
        # LayerNorm naming
        (r"\.gamma$", ".weight"),
        (r"\.beta$", ".bias"),
        # Running stats
        (r"\.running_mean$", ".running_mean"),
        (r"\.running_var$", ".running_var"),
        # Attention naming
        (r"\.in_proj_weight$", ".in_proj.weight"),
        (r"\.in_proj_bias$", ".in_proj.bias"),
        (r"\.out_proj\.weight$", ".out_proj.weight"),
        (r"\.out_proj\.bias$", ".out_proj.bias"),
    ]

    def __init__(
        self,
        target_dtype: str | None = None,
        name_transforms: list[tuple[str, str]] | None = None,
    ):
        """
        Initialize weight converter.

        Args:
            target_dtype: Target dtype for all weights ('float32', 'float16', etc.)
            name_transforms: Additional (regex, replacement) pairs for renaming
        """
        self.target_dtype = target_dtype
        self.name_transforms = list(self.NAME_TRANSFORMS)
        if name_transforms:
            self.name_transforms.extend(name_transforms)

    def load_pytorch_weights(self, model_path: str) -> dict[str, torch.Tensor]:
        """
        Load weights from PyTorch model file.

        Args:
            model_path: Path to .pt, .pth, or .bin file

        Returns:
            Dictionary of parameter names to tensors
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try loading as TorchScript first
        try:
            model = torch.jit.load(str(path), map_location="cpu")
            weights = dict(model.named_parameters())
            # Also include buffers
            weights.update(dict(model.named_buffers()))
            return weights
        except RuntimeError:
            pass

        # Try loading as state_dict
        try:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict):
                # Check if it's a wrapped state dict
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                return state_dict  # type: ignore[no-any-return]
        except Exception:
            pass

        # Try loading as pickle
        state_dict = torch.load(str(path), map_location="cpu")
        if hasattr(state_dict, "state_dict"):
            return state_dict.state_dict()  # type: ignore[no-any-return]
        if isinstance(state_dict, dict):
            return state_dict

        raise ValueError(f"Could not load weights from {model_path}")

    def convert_name(self, pytorch_name: str) -> str:
        """
        Convert PyTorch parameter name to MLX convention.

        Args:
            pytorch_name: Original parameter name

        Returns:
            Converted name for MLX
        """
        mlx_name = pytorch_name

        for pattern, replacement in self.name_transforms:
            mlx_name = re.sub(pattern, replacement, mlx_name)

        return mlx_name

    def convert_tensor(
        self,
        tensor: torch.Tensor,
        target_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array for MLX.

        Args:
            tensor: PyTorch tensor
            target_dtype: Target numpy dtype (optional)

        Returns:
            Numpy array compatible with MLX
        """
        # Get source dtype mapping
        src_dtype = tensor.dtype
        self.DTYPE_MAP.get(src_dtype, np.float32)

        # Convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Handle bfloat16 specially
        if src_dtype == torch.bfloat16:
            arr = tensor.float().numpy()
        else:
            arr = tensor.detach().numpy()

        # Apply target dtype conversion
        if target_dtype is not None:
            arr = arr.astype(target_dtype)
        elif self.target_dtype is not None:
            dtype_map = {
                "float32": np.float32,
                "float16": np.float16,
                "bfloat16": np.float32,  # safetensors compatibility
                "int32": np.int32,
                "int64": np.int64,
            }
            if self.target_dtype in dtype_map:
                arr = arr.astype(dtype_map[self.target_dtype])

        return arr

    def convert(
        self,
        input_path: str,
        output_path: str,
        name_filter: Callable[[str], bool] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ConversionStats:
        """
        Convert PyTorch weights to MLX format.

        Args:
            input_path: Path to PyTorch model file
            output_path: Path for output safetensors file
            name_filter: Optional filter function for parameter names
            progress_callback: Optional callback(name, current, total)

        Returns:
            ConversionStats with conversion details
        """
        # Load PyTorch weights
        pytorch_weights = self.load_pytorch_weights(input_path)

        # Prepare output
        mlx_weights: dict[str, np.ndarray] = {}
        stats = ConversionStats(
            total_tensors=len(pytorch_weights),
            converted_tensors=0,
            skipped_tensors=0,
            total_bytes_input=0,
            total_bytes_output=0,
            dtype_conversions={},
            renamed_tensors=[],
            errors=[],
        )

        # Convert each tensor
        for i, (name, tensor) in enumerate(pytorch_weights.items()):
            if progress_callback:
                progress_callback(name, i + 1, len(pytorch_weights))

            # Apply filter
            if name_filter and not name_filter(name):
                stats.skipped_tensors += 1
                continue

            try:
                # Convert name
                mlx_name = self.convert_name(name)
                if mlx_name != name:
                    stats.renamed_tensors.append((name, mlx_name))

                # Convert tensor
                src_dtype = str(tensor.dtype)
                arr = self.convert_tensor(tensor)
                dst_dtype = str(arr.dtype)

                # Track dtype conversion
                if src_dtype != dst_dtype:
                    key = f"{src_dtype}->{dst_dtype}"
                    stats.dtype_conversions[key] = (
                        stats.dtype_conversions.get(key, 0) + 1
                    )

                # Track sizes
                stats.total_bytes_input += tensor.numel() * tensor.element_size()
                stats.total_bytes_output += arr.nbytes

                # Store converted weight
                mlx_weights[mlx_name] = arr
                stats.converted_tensors += 1

            except Exception as e:
                stats.errors.append(f"{name}: {str(e)}")

        # Save to safetensors
        self._save_safetensors(mlx_weights, output_path)

        return stats

    def _save_safetensors(
        self, weights: dict[str, np.ndarray], output_path: str,
    ) -> None:
        """Save weights to safetensors format."""
        try:
            from safetensors.numpy import save_file

            save_file(weights, output_path)
        except ImportError:
            # Fallback to numpy format if safetensors not available
            output_path = output_path.replace(".safetensors", ".npz")
            np.savez(output_path, **weights)  # type: ignore[arg-type]

    def convert_to_mlx_format(
        self,
        input_path: str,
        output_dir: str,
        shard_size: int | None = None,
    ) -> list[str]:
        """
        Convert PyTorch model to MLX directory format.

        MLX models are typically stored as:
        - model_dir/
          - weights.safetensors (or sharded: weights-00001-of-00002.safetensors)
          - config.json

        Args:
            input_path: Path to PyTorch model
            output_dir: Output directory for MLX model
            shard_size: Max bytes per shard (None = no sharding)

        Returns:
            List of created file paths
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # Load weights
        pytorch_weights = self.load_pytorch_weights(input_path)

        # Convert all weights
        mlx_weights: dict[str, np.ndarray] = {}
        for name, tensor in pytorch_weights.items():
            mlx_name = self.convert_name(name)
            mlx_weights[mlx_name] = self.convert_tensor(tensor)

        created_files = []

        if shard_size is None:
            # Single file
            weights_path = output / "weights.safetensors"
            self._save_safetensors(mlx_weights, str(weights_path))
            created_files.append(str(weights_path))
        else:
            # Sharded files
            shards = self._shard_weights(mlx_weights, shard_size)
            num_shards = len(shards)
            for i, shard in enumerate(shards, 1):
                shard_path = output / f"weights-{i:05d}-of-{num_shards:05d}.safetensors"
                self._save_safetensors(shard, str(shard_path))
                created_files.append(str(shard_path))

        return created_files

    def _shard_weights(
        self,
        weights: dict[str, np.ndarray],
        max_bytes: int,
    ) -> list[dict[str, np.ndarray]]:
        """Split weights into shards by size."""
        shards = []
        current_shard: dict[str, np.ndarray] = {}
        current_size = 0

        for name, arr in weights.items():
            arr_size = arr.nbytes

            # Start new shard if this would exceed limit
            if current_size + arr_size > max_bytes and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0

            current_shard[name] = arr
            current_size += arr_size

        # Don't forget the last shard
        if current_shard:
            shards.append(current_shard)

        return shards

    def verify_conversion(
        self,
        pytorch_path: str,
        mlx_path: str,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """
        Verify converted weights match original.

        Args:
            pytorch_path: Original PyTorch weights
            mlx_path: Converted MLX weights
            tolerance: Numerical tolerance for comparison

        Returns:
            Verification report
        """
        # Load both
        pytorch_weights = self.load_pytorch_weights(pytorch_path)

        try:
            from safetensors.numpy import load_file

            mlx_weights = load_file(mlx_path)
        except ImportError:
            mlx_weights = dict(np.load(mlx_path.replace(".safetensors", ".npz")))

        report: dict[str, Any] = {
            "pytorch_tensors": len(pytorch_weights),
            "mlx_tensors": len(mlx_weights),
            "matched": 0,
            "mismatched": 0,
            "missing_in_mlx": [],
            "extra_in_mlx": [],
            "max_errors": {},
        }

        # Check each PyTorch weight
        for pt_name, pt_tensor in pytorch_weights.items():
            mlx_name = self.convert_name(pt_name)

            if mlx_name not in mlx_weights:
                report["missing_in_mlx"].append(pt_name)
                continue

            # Compare values
            pt_arr = pt_tensor.detach().numpy()
            mlx_arr = mlx_weights[mlx_name]

            # Handle dtype differences
            if pt_arr.dtype != mlx_arr.dtype:
                pt_arr = pt_arr.astype(np.float64)
                mlx_arr = mlx_arr.astype(np.float64)

            max_error = np.max(np.abs(pt_arr - mlx_arr))
            report["max_errors"][pt_name] = float(max_error)

            if max_error <= tolerance:
                report["matched"] += 1
            else:
                report["mismatched"] += 1

        # Check for extra MLX weights
        converted_names = {self.convert_name(n) for n in pytorch_weights.keys()}
        for mlx_name in mlx_weights.keys():
            if mlx_name not in converted_names:
                report["extra_in_mlx"].append(mlx_name)

        report["verified"] = (
            report["mismatched"] == 0 and len(report["missing_in_mlx"]) == 0
        )

        return report
