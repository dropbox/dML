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
MLX Code Generator

Generates MLX model code from analyzed PyTorch architectures.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..analyzer.op_mapper import OpMapper
from ..analyzer.torchscript_analyzer import LayerInfo, ModelArchitecture, OpInfo


@dataclass
class GeneratedModel:
    """Container for generated model code and metadata."""

    model_code: str  # Main model class code
    helper_code: str  # Helper functions (decompositions, custom ops)
    config_code: str  # Configuration class code
    weight_names: dict[str, str]  # Mapping from PyTorch to MLX weight names
    imports: set[str]  # Required imports


class MLXCodeGenerator:
    """
    Generates MLX Python code from PyTorch model architecture.

    Takes analyzed model structure and produces:
    1. MLX nn.Module class definition
    2. Forward pass implementation
    3. Weight loading code
    4. Helper functions for custom ops
    """

    # PyTorch layer type to MLX layer mapping
    LAYER_MAPPINGS = {
        "Linear": "nn.Linear",
        "Conv1d": "nn.Conv1d",
        "Conv2d": "nn.Conv2d",
        "Conv3d": "nn.Conv3d",
        "ConvTranspose1d": "nn.ConvTranspose1d",
        "ConvTranspose2d": "nn.ConvTranspose2d",
        "Embedding": "nn.Embedding",
        "LayerNorm": "nn.LayerNorm",
        "GroupNorm": "nn.GroupNorm",
        "RMSNorm": "nn.RMSNorm",
        "BatchNorm1d": "nn.BatchNorm",
        "BatchNorm2d": "nn.BatchNorm",
        "Dropout": "nn.Dropout",
        "LSTM": "nn.LSTM",
        "GRU": "nn.GRU",
        "MultiheadAttention": None,  # Custom implementation needed
    }

    # PyTorch dtype to MLX dtype
    DTYPE_MAPPINGS = {
        "torch.float32": "mx.float32",
        "torch.float16": "mx.float16",
        "torch.bfloat16": "mx.bfloat16",
        "torch.int32": "mx.int32",
        "torch.int64": "mx.int64",
        "torch.bool": "mx.bool_",
    }

    def __init__(self, op_mapper: OpMapper | None = None):
        """
        Initialize code generator.

        Args:
            op_mapper: OpMapper instance for operation mapping
        """
        self.op_mapper = op_mapper or OpMapper()
        self._imports: set[str] = set()

    def generate(
        self, architecture: ModelArchitecture, class_name: str | None = None,
    ) -> GeneratedModel:
        """
        Generate complete MLX model code from architecture.

        Args:
            architecture: Analyzed model architecture
            class_name: Name for generated class (default: derived from model name)

        Returns:
            GeneratedModel containing all generated code
        """
        if class_name is None:
            class_name = self._sanitize_name(architecture.name) + "MLX"

        self._imports = {"mlx.core as mx", "mlx.nn as nn"}

        # Generate components
        config_code = self._generate_config(architecture, class_name)
        helper_code = self._generate_helpers(architecture.ops)
        model_code = self._generate_model_class(architecture, class_name)
        weight_mapping = self._generate_weight_mapping(architecture)

        return GeneratedModel(
            model_code=model_code,
            helper_code=helper_code,
            config_code=config_code,
            weight_names=weight_mapping,
            imports=self._imports,
        )

    def _sanitize_name(self, name: str) -> str:
        """Convert name to valid Python identifier."""
        # Remove file extension
        name = re.sub(r"\.[^.]+$", "", name)
        # Replace invalid chars
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Convert to CamelCase: capitalize first letter of each part, preserve rest
        parts = name.split("_")
        result = "".join((p[0].upper() + p[1:] if p else "") for p in parts)
        # Ensure starts with letter (after CamelCase conversion)
        if result and result[0].isdigit():
            result = "_" + result
        return result

    def _generate_config(self, architecture: ModelArchitecture, class_name: str) -> str:
        """Generate configuration dataclass."""
        lines = [
            "from dataclasses import dataclass",
            "from typing import Optional, Tuple",
            "",
            "",
            "@dataclass",
            f"class {class_name}Config:",
            '    """Configuration for the model."""',
        ]

        # Extract configuration from layer attributes
        seen_attrs = set()
        for layer in architecture.layers:
            for attr, value in layer.attributes.items():
                if attr not in seen_attrs:
                    seen_attrs.add(attr)
                    dtype = self._infer_type(value)
                    lines.append(f"    {attr}: {dtype} = {repr(value)}")

        # Add some common defaults if not present
        if "hidden_size" not in seen_attrs:
            lines.append("    hidden_size: int = 768")
        if "num_layers" not in seen_attrs:
            lines.append("    num_layers: int = 12")
        if "num_heads" not in seen_attrs:
            lines.append("    num_heads: int = 12")

        return "\n".join(lines)

    def _infer_type(self, value: Any) -> str:
        """Infer Python type annotation from value."""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, tuple):
            return f"Tuple[{', '.join(self._infer_type(v) for v in value)}]"
        if isinstance(value, list):
            return "list"
        return "Any"

    def _generate_helpers(self, ops: list[OpInfo]) -> str:
        """Generate helper functions for decomposed/custom ops."""
        op_names = [op.name for op in ops]
        return self.op_mapper.generate_conversion_code(op_names)

    def _generate_model_class(
        self, architecture: ModelArchitecture, class_name: str,
    ) -> str:
        """Generate the main model class."""
        lines = [
            "",
            "",
            f"class {class_name}(nn.Module):",
            '    """',
            f"    MLX implementation of {architecture.name}.",
            "    ",
            "    Auto-generated from TorchScript model.",
            f"    Total parameters: {architecture.total_params:,}",
            '    """',
            "",
            f"    def __init__(self, config: {class_name}Config):",
            "        super().__init__()",
            "        self.config = config",
            "",
        ]

        # Generate layer initializations
        layer_inits = self._generate_layer_inits(architecture.layers)
        lines.extend(["        " + line for line in layer_inits])

        # Generate forward method
        lines.append("")
        forward_lines = self._generate_forward(architecture)
        lines.extend(forward_lines)

        return "\n".join(lines)

    def _generate_layer_inits(self, layers: list[LayerInfo]) -> list[str]:
        """Generate layer initialization code."""
        lines = []
        seen_layers = set()

        for layer in layers:
            # Skip duplicate layer types at same path
            layer_key = (layer.name, layer.layer_type)
            if layer_key in seen_layers:
                continue
            seen_layers.add(layer_key)

            # Convert layer name to attribute name
            attr_name = layer.name.replace(".", "_")

            # Get MLX layer type
            mlx_type = self.LAYER_MAPPINGS.get(layer.layer_type)

            if mlx_type is None:
                # Custom layer - generate placeholder
                lines.append(f"# TODO: Custom implementation for {layer.layer_type}")
                lines.append(f"# self.{attr_name} = ...")
                continue

            # Generate initialization based on layer type
            init_code = self._generate_layer_init(attr_name, mlx_type, layer)
            if init_code:
                lines.append(init_code)

        return lines

    def _generate_layer_init(
        self, attr_name: str, mlx_type: str, layer: LayerInfo,
    ) -> str | None:
        """Generate single layer initialization."""
        attrs = layer.attributes

        if "Linear" in mlx_type:
            in_features = attrs.get("in_features", "...")
            out_features = attrs.get("out_features", "...")
            bias = attrs.get("bias", True)
            return f"self.{attr_name} = {mlx_type}({in_features}, {out_features}, bias={bias})"

        if "Conv" in mlx_type and "Transpose" not in mlx_type:
            in_ch = attrs.get("in_channels", "...")
            out_ch = attrs.get("out_channels", "...")
            kernel = attrs.get("kernel_size", "...")
            stride = attrs.get("stride", 1)
            padding = attrs.get("padding", 0)
            bias = attrs.get("bias", True)
            return f"self.{attr_name} = {mlx_type}({in_ch}, {out_ch}, kernel_size={kernel}, stride={stride}, padding={padding}, bias={bias})"

        if "Embedding" in mlx_type:
            num_emb = attrs.get("num_embeddings", "...")
            emb_dim = attrs.get("embedding_dim", "...")
            return f"self.{attr_name} = {mlx_type}({num_emb}, {emb_dim})"

        if "LayerNorm" in mlx_type:
            norm_shape = attrs.get("normalized_shape", "...")
            eps = attrs.get("eps", 1e-5)
            return f"self.{attr_name} = {mlx_type}({norm_shape}, eps={eps})"

        if "GroupNorm" in mlx_type:
            num_groups = attrs.get("num_groups", 32)
            num_channels = attrs.get("num_channels", "...")
            eps = attrs.get("eps", 1e-5)
            return f"self.{attr_name} = {mlx_type}({num_groups}, {num_channels}, eps={eps})"

        if "Dropout" in mlx_type:
            p = attrs.get("p", 0.1)
            return f"self.{attr_name} = {mlx_type}(p={p})"

        if "LSTM" in mlx_type or "GRU" in mlx_type:
            input_size = attrs.get("input_size", "...")
            hidden_size = attrs.get("hidden_size", "...")
            num_layers = attrs.get("num_layers", 1)
            bias = attrs.get("bias", True)
            return f"self.{attr_name} = {mlx_type}({input_size}, {hidden_size}, num_layers={num_layers}, bias={bias})"

        return f"# self.{attr_name} = {mlx_type}(...)  # TODO: configure"

    def _generate_forward(self, architecture: ModelArchitecture) -> list[str]:
        """Generate forward method."""
        lines = [
            "    def __call__(self, x):",
            '        """',
            "        Forward pass.",
            "        ",
            "        Args:",
            "            x: Input tensor",
            "        ",
            "        Returns:",
            "            Model output",
            '        """',
            "        # TODO: Implement forward pass based on model architecture",
            "        # This is a placeholder - actual implementation depends on model",
            "        ",
        ]

        # Add placeholder forward implementation
        # Real implementation would trace through the graph
        if architecture.layers:
            lines.append("        # Layer sequence (actual order may differ):")
            for _i, layer in enumerate(architecture.layers[:10]):  # Show first 10 layers
                attr_name = layer.name.replace(".", "_")
                lines.append(f"        # x = self.{attr_name}(x)")
            if len(architecture.layers) > 10:
                lines.append(
                    f"        # ... and {len(architecture.layers) - 10} more layers",
                )
            lines.append("        ")

        lines.append("        return x")

        return lines

    def _generate_weight_mapping(
        self, architecture: ModelArchitecture,
    ) -> dict[str, str]:
        """Generate mapping from PyTorch to MLX weight names."""
        mapping = {}

        for weight in architecture.weights:
            pytorch_name = weight.name
            # MLX uses slightly different naming
            mlx_name = self._convert_weight_name(pytorch_name)
            mapping[pytorch_name] = mlx_name

        return mapping

    def _convert_weight_name(self, pytorch_name: str) -> str:
        """Convert PyTorch weight name to MLX convention."""
        # MLX generally uses the same naming, with minor differences
        mlx_name = pytorch_name

        # Common transformations
        mlx_name = mlx_name.replace("weight", "weight")
        mlx_name = mlx_name.replace("bias", "bias")

        # LayerNorm uses different names
        mlx_name = re.sub(r"\.gamma$", ".weight", mlx_name)
        return re.sub(r"\.beta$", ".bias", mlx_name)


    def generate_file(
        self,
        architecture: ModelArchitecture,
        output_path: str,
        class_name: str | None = None,
    ) -> None:
        """
        Generate and write model code to file.

        Args:
            architecture: Analyzed model architecture
            output_path: Path to write generated code
            class_name: Name for generated class
        """
        generated = self.generate(architecture, class_name)

        # Combine all code
        lines = [
            '"""',
            f"MLX Model: {architecture.name}",
            "",
            "Auto-generated by pytorch_to_mlx converter.",
            "Manual verification and adjustment may be required.",
            '"""',
            "",
        ]

        # Add imports
        lines.extend(f"import {imp}" for imp in sorted(generated.imports))
        lines.append("")

        # Add config
        lines.append(generated.config_code)

        # Add helpers
        if generated.helper_code.strip():
            lines.append("")
            lines.append("# Helper functions")
            lines.append(generated.helper_code)

        # Add model
        lines.append(generated.model_code)

        # Add weight loading function
        lines.extend(
            [
                "",
                "",
                f"def load_weights(model: {class_name or self._sanitize_name(architecture.name) + 'MLX'}, weights_path: str):",
                '    """Load converted weights into model."""',
                "    import mlx.core as mx",
                "    weights = mx.load(weights_path)",
                "    ",
                "    # Weight name mapping (PyTorch -> MLX)",
                "    name_map = {",
            ],
        )

        for pt_name, mlx_name in generated.weight_names.items():
            lines.append(f'        "{pt_name}": "{mlx_name}",')

        lines.extend(
            [
                "    }",
                "    ",
                "    # Apply mapping and load",
                "    mapped_weights = {}",
                "    for pt_name, tensor in weights.items():",
                "        mlx_name = name_map.get(pt_name, pt_name)",
                "        mapped_weights[mlx_name] = tensor",
                "    ",
                "    model.load_weights(list(mapped_weights.items()))",
                "    return model",
            ],
        )

        # Write file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))

    def generate_stub(self, architecture: ModelArchitecture) -> str:
        """
        Generate a minimal stub for manual implementation.

        Args:
            architecture: Analyzed model architecture

        Returns:
            Stub code string
        """
        class_name = self._sanitize_name(architecture.name) + "MLX"

        lines = [
            '"""',
            f"MLX Model Stub: {architecture.name}",
            "",
            "Generated stub for manual implementation.",
            f"Total parameters: {architecture.total_params:,}",
            f"Model size: {architecture.total_size_bytes / 1024 / 1024:.2f} MB",
            '"""',
            "",
            "import mlx.core as mx",
            "import mlx.nn as nn",
            "",
            "",
            f"class {class_name}(nn.Module):",
            '    """',
            f"    MLX implementation of {architecture.name}.",
            '    """',
            "",
            "    def __init__(self):",
            "        super().__init__()",
            "        # TODO: Define layers",
            "",
        ]

        # List operations that need implementation
        ops = architecture.ops
        report = self.op_mapper.get_coverage_report([op.name for op in ops])

        if report["unsupported_list"]:
            lines.append("        # Unsupported operations requiring custom impl:")
            lines.extend(f"        #   - {op}" for op in report["unsupported_list"])
            lines.append("")

        lines.extend(
            [
                "    def __call__(self, x):",
                '        """Forward pass."""',
                "        # TODO: Implement forward",
                "        raise NotImplementedError()",
                "",
                "",
                f"# Operation coverage: {report['coverage_percent']:.1f}%",
                f"# Direct mappings: {report['direct_mappings']}",
                f"# Decomposed ops: {report['decomposed_ops']}",
                f"# Custom ops: {report['custom_ops']}",
                f"# Unsupported: {report['unsupported_ops']}",
            ],
        )

        return "\n".join(lines)
