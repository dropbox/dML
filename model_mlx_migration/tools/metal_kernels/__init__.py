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
Custom Fused Metal Kernels for MLX TTS Models

This package provides optimized Metal kernels for:
- Kokoro TTS (AdaIN, Snake activation, Weight-norm Conv)
- CosyVoice TTS (Quantized linear, vocoder convolutions)

Usage:
    from tools.metal_kernels import fused_adain_conv1d, snake1d, int8_linear

    # Use as drop-in replacements for MLX operations
    out = fused_adain_conv1d(x, weight, gamma, beta, kernel_size=3)
    out = snake1d(x, alpha)
    out = int8_linear(x, weight_q, scales, zeros)
"""

from .ops import (
    # Utilities
    benchmark_kernel,
    # Phase 2.1: AdaIN + Conv fusion
    fused_adain_conv1d,
    # Phase 2.3: Instance normalization
    fused_instance_norm,
    # Phase 3.2: INT4 quantized operations
    int4_linear,
    # Phase 3.1: INT8 quantized operations
    int8_linear,
    int8_linear_bias,
    # Phase 2.2: Snake activation
    snake1d,
    snake1d_fused,
    verify_numerical_accuracy,
)

__version__ = "0.1.0"
__all__ = [
    "fused_adain_conv1d",
    "snake1d",
    "snake1d_fused",
    "fused_instance_norm",
    "int8_linear",
    "int8_linear_bias",
    "int4_linear",
    "benchmark_kernel",
    "verify_numerical_accuracy",
]
