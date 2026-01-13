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
PyTorch to MLX Operation Mapper

Maps PyTorch/TorchScript operations to their MLX equivalents.
Handles direct mappings, decompositions, and custom implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MappingType(Enum):
    """Type of operation mapping."""

    DIRECT = "direct"  # 1:1 mapping (e.g., nn.Linear -> nn.Linear)
    RENAMED = "renamed"  # Same op, different name
    DECOMPOSED = "decomposed"  # Complex op broken into primitives
    CUSTOM = "custom"  # Requires custom implementation
    UNSUPPORTED = "unsupported"  # Cannot be mapped


@dataclass
class OpMapping:
    """Describes how a PyTorch op maps to MLX."""

    torch_op: str  # PyTorch operation name
    mlx_op: str  # MLX operation name (or None if custom)
    mapping_type: MappingType  # How the mapping works
    notes: str = ""  # Additional notes
    decomposition: str | None = None  # Code for decomposition
    custom_impl: str | None = None  # Code for custom implementation


class OpMapper:
    """
    Maps PyTorch operations to MLX equivalents.

    Handles three categories:
    1. Direct mappings - PyTorch op has exact MLX equivalent
    2. Decompositions - Complex op broken into MLX primitives
    3. Custom implementations - Ops needing hand-written MLX code
    """

    # Direct 1:1 mappings from PyTorch to MLX
    # Format: 'torch_op': 'mlx_op' (None means no-op in MLX)
    DIRECT_MAPPINGS: dict[str, str | None] = {
        # Neural network layers
        "aten::linear": "mx.nn.Linear",
        "aten::conv1d": "mx.nn.Conv1d",
        "aten::conv2d": "mx.nn.Conv2d",
        "aten::conv3d": "mx.nn.Conv3d",
        "aten::embedding": "mx.nn.Embedding",
        "aten::layer_norm": "mx.nn.LayerNorm",
        "aten::group_norm": "mx.nn.GroupNorm",
        "aten::lstm": "mx.nn.LSTM",
        "aten::gru": "mx.nn.GRU",
        "aten::dropout": "mx.nn.Dropout",
        # Activations
        "aten::relu": "mx.nn.relu",
        "aten::relu_": "mx.nn.relu",
        "aten::gelu": "mx.nn.gelu",
        "aten::silu": "mx.nn.silu",
        "aten::sigmoid": "mx.sigmoid",
        "aten::tanh": "mx.tanh",
        "aten::softmax": "mx.softmax",
        "aten::log_softmax": "mx.log_softmax",
        "aten::leaky_relu": "mx.nn.leaky_relu",
        "aten::elu": "mx.nn.elu",
        # Tensor operations
        "aten::reshape": "mx.reshape",
        "aten::view": "mx.reshape",
        "aten::transpose": "mx.transpose",
        "aten::permute": "mx.transpose",
        "aten::contiguous": None,  # No-op in MLX (lazy evaluation)
        "aten::squeeze": "mx.squeeze",
        "aten::unsqueeze": "mx.expand_dims",
        "aten::cat": "mx.concatenate",
        "aten::stack": "mx.stack",
        "aten::split": "mx.split",
        "aten::chunk": "mx.split",
        "aten::flatten": "mx.flatten",
        # Arithmetic
        "aten::add": "mx.add",
        "aten::sub": "mx.subtract",
        "aten::mul": "mx.multiply",
        "aten::div": "mx.divide",
        "aten::matmul": "mx.matmul",
        "aten::mm": "mx.matmul",
        "aten::bmm": "mx.matmul",
        "aten::einsum": "mx.einsum",
        "aten::pow": "mx.power",
        "aten::sqrt": "mx.sqrt",
        "aten::rsqrt": "mx.rsqrt",
        "aten::exp": "mx.exp",
        "aten::log": "mx.log",
        "aten::abs": "mx.abs",
        "aten::neg": "mx.negative",
        "aten::sin": "mx.sin",
        "aten::cos": "mx.cos",
        # Reductions
        "aten::sum": "mx.sum",
        "aten::mean": "mx.mean",
        "aten::max": "mx.max",
        "aten::min": "mx.min",
        "aten::prod": "mx.prod",
        "aten::argmax": "mx.argmax",
        "aten::argmin": "mx.argmin",
        "aten::all": "mx.all",
        "aten::any": "mx.any",
        # Comparison
        "aten::eq": "mx.equal",
        "aten::ne": "mx.not_equal",
        "aten::lt": "mx.less",
        "aten::le": "mx.less_equal",
        "aten::gt": "mx.greater",
        "aten::ge": "mx.greater_equal",
        # Indexing
        "aten::index_select": "mx.take",
        "aten::gather": "mx.take_along_axis",
        "aten::where": "mx.where",
        # Creation
        "aten::zeros": "mx.zeros",
        "aten::ones": "mx.ones",
        "aten::full": "mx.full",
        "aten::arange": "mx.arange",
        "aten::linspace": "mx.linspace",
        # Type conversion
        "aten::to": "mx.astype",
        "aten::float": "mx.astype",
        "aten::half": "mx.astype",
    }

    # Operations that need decomposition into primitives
    DECOMPOSITIONS: dict[str, str] = {
        "aten::addmm": '''
def addmm_mlx(bias, mat1, mat2, alpha=1.0, beta=1.0):
    """MLX equivalent of torch.addmm: beta * bias + alpha * (mat1 @ mat2)"""
    return beta * bias + alpha * mx.matmul(mat1, mat2)
''',
        "aten::baddbmm": '''
def baddbmm_mlx(batch1, batch2, bias, alpha=1.0, beta=1.0):
    """MLX equivalent of torch.baddbmm: beta * bias + alpha * (batch1 @ batch2)"""
    return beta * bias + alpha * mx.matmul(batch1, batch2)
''',
        "aten::scaled_dot_product_attention": '''
def scaled_dot_product_attention_mlx(query, key, value, attn_mask=None, dropout_p=0.0, scale=None):
    """MLX equivalent of F.scaled_dot_product_attention"""
    if scale is None:
        scale = 1.0 / mx.sqrt(mx.array(query.shape[-1], dtype=query.dtype))

    # Compute attention scores
    scores = mx.matmul(query, mx.transpose(key, axes=(-2, -1))) * scale

    # Apply mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Softmax
    attn_weights = mx.softmax(scores, axis=-1)

    # Apply attention to values
    return mx.matmul(attn_weights, value)
''',
        "aten::multi_head_attention_forward": '''
def multi_head_attention_mlx(query, key, value, embed_dim, num_heads,
                             in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                             attn_mask=None, key_padding_mask=None):
    """MLX equivalent of F.multi_head_attention_forward"""
    batch_size, seq_len, _ = query.shape
    head_dim = embed_dim // num_heads

    # Project Q, K, V
    qkv = mx.matmul(query, in_proj_weight.T)
    if in_proj_bias is not None:
        qkv = qkv + in_proj_bias

    # Split into Q, K, V
    q, k, v = mx.split(qkv, 3, axis=-1)

    # Reshape for multi-head attention
    q = mx.reshape(q, (batch_size, seq_len, num_heads, head_dim))
    k = mx.reshape(k, (batch_size, seq_len, num_heads, head_dim))
    v = mx.reshape(v, (batch_size, seq_len, num_heads, head_dim))

    # Transpose to (batch, heads, seq, dim)
    q = mx.transpose(q, axes=(0, 2, 1, 3))
    k = mx.transpose(k, axes=(0, 2, 1, 3))
    v = mx.transpose(v, axes=(0, 2, 1, 3))

    # Scaled dot-product attention
    scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=q.dtype))
    scores = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * scale

    if attn_mask is not None:
        scores = scores + attn_mask

    attn_weights = mx.softmax(scores, axis=-1)
    attn_output = mx.matmul(attn_weights, v)

    # Reshape back
    attn_output = mx.transpose(attn_output, axes=(0, 2, 1, 3))
    attn_output = mx.reshape(attn_output, (batch_size, seq_len, embed_dim))

    # Output projection
    output = mx.matmul(attn_output, out_proj_weight.T)
    if out_proj_bias is not None:
        output = output + out_proj_bias

    return output
''',
        "aten::batch_norm": '''
def batch_norm_mlx(input, running_mean, running_var, weight, bias, training, momentum, eps):
    """MLX equivalent of F.batch_norm"""
    if training:
        # Compute batch statistics
        mean = mx.mean(input, axis=(0, 2, 3), keepdims=True)
        var = mx.var(input, axis=(0, 2, 3), keepdims=True)
    else:
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)

    # Normalize
    x_norm = (input - mean) / mx.sqrt(var + eps)

    # Scale and shift
    if weight is not None:
        x_norm = x_norm * weight.reshape(1, -1, 1, 1)
    if bias is not None:
        x_norm = x_norm + bias.reshape(1, -1, 1, 1)

    return x_norm
''',
    }

    # Operations that require custom implementations
    CUSTOM_OPS: dict[str, str] = {
        "aten::stft": '''
def stft_mlx(input, n_fft, hop_length=None, win_length=None, window=None,
             center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True):
    """MLX equivalent of torch.stft using FFT"""
    import mlx.core as mx

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Pad input if center=True
    if center:
        pad_amount = n_fft // 2
        input = mx.pad(input, [(0, 0)] * (input.ndim - 1) + [(pad_amount, pad_amount)])

    # Apply window
    if window is None:
        window = mx.ones((win_length,))

    # Frame the signal
    batch_size = input.shape[0] if input.ndim > 1 else 1
    signal = input.reshape(-1) if input.ndim == 1 else input

    num_frames = (signal.shape[-1] - n_fft) // hop_length + 1
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[..., start:start + n_fft] * window
        frames.append(frame)

    frames = mx.stack(frames, axis=-2)

    # Apply FFT
    spectrum = mx.fft.rfft(frames) if onesided else mx.fft.fft(frames)

    if normalized:
        spectrum = spectrum / mx.sqrt(mx.array(n_fft, dtype=spectrum.dtype))

    return spectrum
''',
        "aten::istft": '''
def istft_mlx(input, n_fft, hop_length=None, win_length=None, window=None,
              center=True, normalized=False, onesided=True, length=None):
    """MLX equivalent of torch.istft using inverse FFT"""
    import mlx.core as mx

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if window is None:
        window = mx.ones((win_length,))

    # Inverse FFT
    if onesided:
        frames = mx.fft.irfft(input)
    else:
        frames = mx.fft.ifft(input).real

    if normalized:
        frames = frames * mx.sqrt(mx.array(n_fft, dtype=frames.dtype))

    # Overlap-add
    num_frames = frames.shape[-2]
    output_length = (num_frames - 1) * hop_length + n_fft

    output = mx.zeros((output_length,))
    window_sum = mx.zeros((output_length,))

    for i in range(num_frames):
        start = i * hop_length
        output = output.at[start:start + n_fft].add(frames[..., i, :] * window)
        window_sum = window_sum.at[start:start + n_fft].add(window ** 2)

    # Normalize by window overlap
    output = output / mx.maximum(window_sum, mx.array(1e-8))

    # Trim padding if center=True
    if center:
        pad_amount = n_fft // 2
        output = output[pad_amount:-pad_amount]

    if length is not None:
        output = output[:length]

    return output
''',
        "aten::grid_sample": '''
def grid_sample_mlx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    """MLX equivalent of F.grid_sample (2D bilinear interpolation)"""
    import mlx.core as mx

    N, C, H, W = input.shape
    _, H_out, W_out, _ = grid.shape

    # Denormalize grid coordinates
    if align_corners:
        grid_x = (grid[..., 0] + 1) * (W - 1) / 2
        grid_y = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        grid_x = ((grid[..., 0] + 1) * W - 1) / 2
        grid_y = ((grid[..., 1] + 1) * H - 1) / 2

    # Get corner coordinates
    x0 = mx.floor(grid_x).astype(mx.int32)
    x1 = x0 + 1
    y0 = mx.floor(grid_y).astype(mx.int32)
    y1 = y0 + 1

    # Compute interpolation weights
    wa = (x1 - grid_x) * (y1 - grid_y)
    wb = (x1 - grid_x) * (grid_y - y0)
    wc = (grid_x - x0) * (y1 - grid_y)
    wd = (grid_x - x0) * (grid_y - y0)

    # Clamp coordinates
    x0 = mx.clip(x0, 0, W - 1)
    x1 = mx.clip(x1, 0, W - 1)
    y0 = mx.clip(y0, 0, H - 1)
    y1 = mx.clip(y1, 0, H - 1)

    # Gather pixel values
    # This is simplified - full implementation would handle all modes
    output = (
        wa[..., None] * input[:, :, y0, x0].transpose(0, 2, 3, 1) +
        wb[..., None] * input[:, :, y1, x0].transpose(0, 2, 3, 1) +
        wc[..., None] * input[:, :, y0, x1].transpose(0, 2, 3, 1) +
        wd[..., None] * input[:, :, y1, x1].transpose(0, 2, 3, 1)
    )

    return output.transpose(0, 3, 1, 2)
''',
    }

    def __init__(self) -> None:
        """Initialize the op mapper."""
        self._mapping_cache: dict[str, OpMapping] = {}

    def map_op(self, torch_op: str) -> OpMapping:
        """
        Map a PyTorch operation to MLX.

        Args:
            torch_op: Name of the PyTorch operation (e.g., 'aten::linear')

        Returns:
            OpMapping describing how to convert the operation
        """
        if torch_op in self._mapping_cache:
            return self._mapping_cache[torch_op]

        # Check direct mappings
        if torch_op in self.DIRECT_MAPPINGS:
            mlx_op = self.DIRECT_MAPPINGS[torch_op]
            if mlx_op is None:
                mapping = OpMapping(
                    torch_op=torch_op,
                    mlx_op="",
                    mapping_type=MappingType.DIRECT,
                    notes="No-op in MLX (can be removed)",
                )
            else:
                mapping = OpMapping(
                    torch_op=torch_op,
                    mlx_op=mlx_op,
                    mapping_type=MappingType.DIRECT,
                    notes="Direct 1:1 mapping",
                )

        # Check decompositions
        elif torch_op in self.DECOMPOSITIONS:
            mapping = OpMapping(
                torch_op=torch_op,
                mlx_op="",
                mapping_type=MappingType.DECOMPOSED,
                notes="Decomposed into MLX primitives",
                decomposition=self.DECOMPOSITIONS[torch_op],
            )

        # Check custom implementations
        elif torch_op in self.CUSTOM_OPS:
            mapping = OpMapping(
                torch_op=torch_op,
                mlx_op="",
                mapping_type=MappingType.CUSTOM,
                notes="Requires custom implementation",
                custom_impl=self.CUSTOM_OPS[torch_op],
            )

        # Unknown operation
        else:
            mapping = OpMapping(
                torch_op=torch_op,
                mlx_op="",
                mapping_type=MappingType.UNSUPPORTED,
                notes=f"No known MLX equivalent for {torch_op}",
            )

        self._mapping_cache[torch_op] = mapping
        return mapping

    def get_all_mappings(self, ops: list[str]) -> dict[str, OpMapping]:
        """
        Get mappings for a list of operations.

        Args:
            ops: List of PyTorch operation names

        Returns:
            Dictionary mapping op names to OpMapping objects
        """
        return {op: self.map_op(op) for op in ops}

    def get_coverage_report(self, ops: list[str]) -> dict[str, Any]:
        """
        Generate a coverage report for a set of operations.

        Args:
            ops: List of PyTorch operation names

        Returns:
            Dictionary with coverage statistics
        """
        mappings = self.get_all_mappings(ops)

        direct = [m for m in mappings.values() if m.mapping_type == MappingType.DIRECT]
        decomposed = [
            m for m in mappings.values() if m.mapping_type == MappingType.DECOMPOSED
        ]
        custom = [m for m in mappings.values() if m.mapping_type == MappingType.CUSTOM]
        unsupported = [
            m for m in mappings.values() if m.mapping_type == MappingType.UNSUPPORTED
        ]

        total = len(mappings)
        supported = total - len(unsupported)

        return {
            "total_ops": total,
            "supported_ops": supported,
            "coverage_percent": (supported / total * 100) if total > 0 else 0,
            "direct_mappings": len(direct),
            "decomposed_ops": len(decomposed),
            "custom_ops": len(custom),
            "unsupported_ops": len(unsupported),
            "unsupported_list": [m.torch_op for m in unsupported],
            "mappings": mappings,
        }

    def generate_conversion_code(self, ops: list[str]) -> str:
        """
        Generate Python code for all necessary custom implementations.

        Args:
            ops: List of PyTorch operation names

        Returns:
            Python code string with all needed implementations
        """
        lines = [
            '"""Auto-generated MLX operation implementations."""',
            "",
            "import mlx.core as mx",
            "import mlx.nn as nn",
            "",
        ]

        mappings = self.get_all_mappings(ops)

        # Add decompositions
        decomposed = [
            m for m in mappings.values() if m.mapping_type == MappingType.DECOMPOSED
        ]
        if decomposed:
            lines.append("# Decomposed operations")
            for m in decomposed:
                if m.decomposition:
                    lines.append(m.decomposition.strip())
                    lines.append("")

        # Add custom implementations
        custom = [m for m in mappings.values() if m.mapping_type == MappingType.CUSTOM]
        if custom:
            lines.append("# Custom implementations")
            for m in custom:
                if m.custom_impl:
                    lines.append(m.custom_impl.strip())
                    lines.append("")

        return "\n".join(lines)

    @classmethod
    def list_supported_ops(cls) -> list[str]:
        """Get list of all supported PyTorch operations."""
        supported = set(cls.DIRECT_MAPPINGS.keys())
        supported.update(cls.DECOMPOSITIONS.keys())
        supported.update(cls.CUSTOM_OPS.keys())
        return sorted(supported)
