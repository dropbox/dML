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
Custom Metal Kernels for MLX TTS Models

Individual kernel implementations.
"""

from .instance_norm_style import adain_conv1d, fused_instance_norm_style
from .int8_linear import int8_linear, quantize_weights_int8
from .snake1d import snake1d_baseline, snake1d_custom

__all__ = [
    "snake1d_custom",
    "snake1d_baseline",
    "fused_instance_norm_style",
    "adain_conv1d",
    "int8_linear",
    "quantize_weights_int8",
]
