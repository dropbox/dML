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
PyTorch to MLX Model Converter

A general-purpose tool for converting PyTorch/TorchScript models to
Apple's MLX framework.
"""

__version__ = "0.1.0"

from .analyzer.op_mapper import OpMapper
from .analyzer.torchscript_analyzer import TorchScriptAnalyzer
from .generator.mlx_code_generator import MLXCodeGenerator
from .generator.weight_converter import WeightConverter
from .validator.benchmark import Benchmark
from .validator.numerical_validator import NumericalValidator

__all__ = [
    "TorchScriptAnalyzer",
    "OpMapper",
    "MLXCodeGenerator",
    "WeightConverter",
    "NumericalValidator",
    "Benchmark",
]
