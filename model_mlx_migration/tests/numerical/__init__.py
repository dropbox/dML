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
DashVoice Numerical Equivalence Framework

Provides automated testing for PyTorch vs MLX numerical equivalence:
- Text models (MADLAD, LLaMA): max error <1e-5, mean error <1e-6
- Audio encoders (Whisper): max error <1e-4, mean error <1e-5
- TTS models (Kokoro, CosyVoice2): max error <1e-3, mean error <1e-4
- Speaker embeddings: max error <1e-4, mean error <1e-5
"""

from .numerical_framework import (
    NumericalResult,
    NumericalTarget,
    NumericalValidator,
    compare_tensors,
    run_numerical_tests,
)

__all__ = [
    "NumericalResult",
    "NumericalTarget",
    "NumericalValidator",
    "compare_tensors",
    "run_numerical_tests",
]
