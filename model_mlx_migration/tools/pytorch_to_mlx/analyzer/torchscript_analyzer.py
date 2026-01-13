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
TorchScript Model Analyzer

Parses TorchScript models to extract architecture, operations, and weight information.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.jit


class OpCategory(Enum):
    """Categories of operations for mapping purposes."""

    LAYER = "layer"  # nn.Module layers (Linear, Conv, etc.)
    ACTIVATION = "activation"  # Activation functions
    NORMALIZATION = "norm"  # Normalization layers
    ATTENTION = "attention"  # Attention mechanisms
    POOLING = "pooling"  # Pooling operations
    ARITHMETIC = "arithmetic"  # Math operations (+, -, *, /)
    TENSOR = "tensor"  # Tensor manipulation (reshape, transpose)
    REDUCTION = "reduction"  # Reduction ops (sum, mean, max)
    COMPARISON = "comparison"  # Comparison ops (>, <, ==)
    CUSTOM = "custom"  # Custom/unsupported ops
    UNKNOWN = "unknown"  # Cannot categorize


@dataclass
class OpInfo:
    """Information about an operation in the model."""

    name: str  # Operation name (e.g., 'aten::linear')
    category: OpCategory  # Category for mapping
    input_types: list[str]  # Input tensor types
    output_types: list[str]  # Output tensor types
    attributes: dict[str, Any] = field(default_factory=dict)  # Static attributes
    count: int = 1  # Number of occurrences


@dataclass
class LayerInfo:
    """Information about a layer in the model."""

    name: str  # Layer name (e.g., 'encoder.layer.0')
    layer_type: str  # Type (e.g., 'Linear', 'Conv1d')
    input_shapes: list[tuple[int, ...]]  # Expected input shapes
    output_shapes: list[tuple[int, ...]]  # Expected output shapes
    params: dict[str, tuple[int, ...]]  # Parameter name -> shape
    attributes: dict[str, Any] = field(default_factory=dict)  # Layer config


@dataclass
class WeightInfo:
    """Information about model weights."""

    name: str  # Parameter name
    shape: tuple[int, ...]  # Weight shape
    dtype: str  # Data type
    requires_grad: bool  # Trainable flag
    size_bytes: int  # Size in bytes


@dataclass
class ModelArchitecture:
    """Complete model architecture extracted from TorchScript."""

    name: str  # Model name
    layers: list[LayerInfo]  # All layers in order
    ops: list[OpInfo]  # All unique operations
    weights: list[WeightInfo]  # All model weights
    input_shapes: list[tuple[int, ...]]  # Model input shapes
    output_shapes: list[tuple[int, ...]]  # Model output shapes
    total_params: int  # Total parameter count
    total_size_bytes: int  # Total model size


class TorchScriptAnalyzer:
    """Analyzes TorchScript models to extract architecture and operations."""

    # Common TorchScript operation patterns
    LAYER_OPS = {
        "aten::linear",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv3d",
        "aten::embedding",
        "aten::lstm",
        "aten::gru",
        "aten::batch_norm",
        "aten::layer_norm",
        "aten::group_norm",
        "aten::instance_norm",
    }

    ACTIVATION_OPS = {
        "aten::relu",
        "aten::relu_",
        "aten::gelu",
        "aten::silu",
        "aten::swish",
        "aten::sigmoid",
        "aten::tanh",
        "aten::softmax",
        "aten::log_softmax",
        "aten::leaky_relu",
        "aten::elu",
        "aten::hardswish",
        "aten::mish",
    }

    NORM_OPS = {
        "aten::layer_norm",
        "aten::batch_norm",
        "aten::group_norm",
        "aten::instance_norm",
        "aten::rms_norm",
    }

    ATTENTION_OPS = {
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_attention",
        "aten::multi_head_attention_forward",
    }

    TENSOR_OPS = {
        "aten::reshape",
        "aten::view",
        "aten::transpose",
        "aten::permute",
        "aten::contiguous",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::cat",
        "aten::stack",
        "aten::split",
        "aten::chunk",
        "aten::flatten",
    }

    ARITHMETIC_OPS = {
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::matmul",
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::einsum",
        "aten::pow",
        "aten::sqrt",
        "aten::rsqrt",
        "aten::exp",
        "aten::log",
    }

    REDUCTION_OPS = {
        "aten::sum",
        "aten::mean",
        "aten::max",
        "aten::min",
        "aten::prod",
        "aten::argmax",
        "aten::argmin",
        "aten::all",
        "aten::any",
    }

    def __init__(self, model_path: str | None = None):
        """
        Initialize analyzer with optional model path.

        Args:
            model_path: Path to TorchScript model file (.pt or .pth)
        """
        self.model: torch.jit.ScriptModule | None = None
        self.model_path: Path | None = None
        self._ops_cache: dict[str, OpInfo] = {}
        self._layers_cache: list[LayerInfo] = []
        self._weights_cache: list[WeightInfo] = []

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """
        Load a TorchScript model.

        Args:
            model_path: Path to the model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = torch.jit.load(str(self.model_path), map_location="cpu")
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}") from e

        # Clear caches for new model
        self._ops_cache.clear()
        self._layers_cache.clear()
        self._weights_cache.clear()

    def _categorize_op(self, op_name: str) -> OpCategory:
        """Categorize an operation by name."""
        if op_name in self.LAYER_OPS:
            return OpCategory.LAYER
        if op_name in self.ACTIVATION_OPS:
            return OpCategory.ACTIVATION
        if op_name in self.NORM_OPS:
            return OpCategory.NORMALIZATION
        if op_name in self.ATTENTION_OPS:
            return OpCategory.ATTENTION
        if op_name in self.TENSOR_OPS:
            return OpCategory.TENSOR
        if op_name in self.ARITHMETIC_OPS:
            return OpCategory.ARITHMETIC
        if op_name in self.REDUCTION_OPS:
            return OpCategory.REDUCTION
        if op_name.startswith("aten::"):
            return OpCategory.UNKNOWN
        return OpCategory.CUSTOM

    def _extract_ops_from_graph(self, graph: torch.Graph) -> dict[str, OpInfo]:
        """Extract all operations from a TorchScript graph."""
        ops: dict[str, OpInfo] = {}

        for node in graph.nodes():
            op_name = node.kind()

            # Skip control flow and metadata nodes
            if op_name.startswith("prim::"):
                continue

            if op_name in ops:
                ops[op_name].count += 1
            else:
                # Extract input/output types
                input_types = [str(inp.type()) for inp in node.inputs()]
                output_types = [str(out.type()) for out in node.outputs()]

                # Extract attributes
                attributes = {}
                for attr_name in node.attributeNames():
                    try:
                        attributes[attr_name] = node[attr_name]
                    except RuntimeError:
                        pass

                ops[op_name] = OpInfo(
                    name=op_name,
                    category=self._categorize_op(op_name),
                    input_types=input_types,
                    output_types=output_types,
                    attributes=attributes,
                    count=1,
                )

        return ops

    def get_ops(self) -> list[OpInfo]:
        """
        Get all operations used in the model.

        Returns:
            List of OpInfo for all unique operations
        """
        if not self.model:
            raise RuntimeError("No model loaded. Call load() first.")

        if self._ops_cache:
            return list(self._ops_cache.values())

        # Get the forward graph
        graph = self.model.graph

        # Extract ops from main graph
        self._ops_cache = self._extract_ops_from_graph(graph)

        # Also check submodules
        for _name, module in self.model.named_modules():
            if hasattr(module, "graph"):
                subgraph_ops = self._extract_ops_from_graph(module.graph)
                for op_name, op_info in subgraph_ops.items():
                    if op_name in self._ops_cache:
                        self._ops_cache[op_name].count += op_info.count
                    else:
                        self._ops_cache[op_name] = op_info

        return list(self._ops_cache.values())

    def get_unsupported_ops(
        self, supported_ops: set[str] | None = None,
    ) -> list[OpInfo]:
        """
        Get operations that don't have direct MLX equivalents.

        Args:
            supported_ops: Set of supported op names (defaults to common MLX ops)

        Returns:
            List of OpInfo for unsupported operations
        """
        ops = self.get_ops()

        if supported_ops is None:
            # Default supported ops (direct MLX mappings)
            supported_ops = (
                self.LAYER_OPS
                | self.ACTIVATION_OPS
                | self.NORM_OPS
                | self.TENSOR_OPS
                | self.ARITHMETIC_OPS
                | self.REDUCTION_OPS
            )

        return [
            op
            for op in ops
            if op.name not in supported_ops
            and op.category in (OpCategory.CUSTOM, OpCategory.UNKNOWN)
        ]

    def get_weights(self) -> list[WeightInfo]:
        """
        Get information about all model weights.

        Returns:
            List of WeightInfo for all parameters
        """
        if not self.model:
            raise RuntimeError("No model loaded. Call load() first.")

        if self._weights_cache:
            return self._weights_cache

        for name, param in self.model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            self._weights_cache.append(
                WeightInfo(
                    name=name,
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                    requires_grad=param.requires_grad,
                    size_bytes=size_bytes,
                ),
            )

        # Also include buffers (non-trainable but saved state)
        for name, buffer in self.model.named_buffers():
            size_bytes = buffer.numel() * buffer.element_size()
            self._weights_cache.append(
                WeightInfo(
                    name=name,
                    shape=tuple(buffer.shape),
                    dtype=str(buffer.dtype),
                    requires_grad=False,
                    size_bytes=size_bytes,
                ),
            )

        return self._weights_cache

    def get_weight_mapping(self) -> dict[str, WeightInfo]:
        """
        Get a mapping of weight names to their info.

        Returns:
            Dictionary mapping parameter names to WeightInfo
        """
        weights = self.get_weights()
        return {w.name: w for w in weights}

    def get_architecture(
        self, sample_input: torch.Tensor | None = None,
    ) -> ModelArchitecture:
        """
        Extract complete model architecture.

        Args:
            sample_input: Optional sample input for shape inference

        Returns:
            ModelArchitecture containing all extracted info
        """
        if not self.model:
            raise RuntimeError("No model loaded. Call load() first.")

        ops = self.get_ops()
        weights = self.get_weights()

        # Calculate totals - only count trainable parameters (not buffers)
        total_params = sum(
            math.prod(w.shape) if w.shape else 1 for w in weights if w.requires_grad
        )
        total_size = sum(w.size_bytes for w in weights)

        # Try to infer input/output shapes
        input_shapes: list[tuple[int, ...]] = []
        output_shapes: list[tuple[int, ...]] = []

        if sample_input is not None:
            input_shapes = [tuple(sample_input.shape)]
            try:
                with torch.no_grad():
                    output = self.model(sample_input)
                    if isinstance(output, torch.Tensor):
                        output_shapes = [tuple(output.shape)]
                    elif isinstance(output, (tuple, list)):
                        output_shapes = [
                            tuple(o.shape)
                            for o in output
                            if isinstance(o, torch.Tensor)
                        ]
            except Exception:
                pass

        # Extract layer info from named modules
        layers = self._extract_layers()

        model_name = self.model_path.stem if self.model_path else "unknown"

        return ModelArchitecture(
            name=model_name,
            layers=layers,
            ops=ops,
            weights=weights,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            total_params=total_params,
            total_size_bytes=total_size,
        )

    def _extract_layers(self) -> list[LayerInfo]:
        """Extract layer information from model modules."""
        if self._layers_cache:
            return self._layers_cache

        if self.model is None:
            return []

        for name, module in self.model.named_modules():
            if name == "":  # Skip root module
                continue

            # For TorchScript models, use original_name if available
            if hasattr(module, "original_name"):
                layer_type = module.original_name
            else:
                layer_type = type(module).__name__

            # Get parameters for this layer
            params = {}
            for pname, param in module.named_parameters(recurse=False):
                params[pname] = tuple(param.shape)

            # Get layer-specific attributes
            attributes = {}
            for attr in [
                "in_features",
                "out_features",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "num_heads",
                "embed_dim",
                "hidden_size",
                "num_layers",
                "bias",
                "eps",
                "normalized_shape",
            ]:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if isinstance(val, (int, float, bool, tuple, list)):
                        attributes[attr] = val

            self._layers_cache.append(
                LayerInfo(
                    name=name,
                    layer_type=layer_type,
                    input_shapes=[],  # Would need tracing to determine
                    output_shapes=[],
                    params=params,
                    attributes=attributes,
                ),
            )

        return self._layers_cache

    def summarize(self) -> str:
        """
        Generate a human-readable summary of the model.

        Returns:
            Formatted string summary
        """
        arch = self.get_architecture()

        lines = [
            f"Model: {arch.name}",
            f"Total Parameters: {arch.total_params:,}",
            f"Total Size: {arch.total_size_bytes / (1024**2):.2f} MB",
            "",
            "Operations:",
        ]

        # Group ops by category
        by_category: dict[OpCategory, list[OpInfo]] = {}
        for op in arch.ops:
            by_category.setdefault(op.category, []).append(op)

        for category in OpCategory:
            if category in by_category:
                ops = by_category[category]
                lines.append(f"  {category.value}:")
                lines.extend(f"    {op.name}: {op.count}x" for op in ops)

        # List unsupported
        unsupported = self.get_unsupported_ops()
        if unsupported:
            lines.append("")
            lines.append("Unsupported Operations (need custom impl):")
            lines.extend(f"  {op.name}: {op.count}x" for op in unsupported)

        return "\n".join(lines)
