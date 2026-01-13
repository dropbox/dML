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
Model-specific converters.

Each converter handles a specific model architecture, providing:
- Conversion from original format to MLX
- Validation of numerical equivalence
- Benchmarking against original implementation
"""

from .cosyvoice2_converter import CosyVoice2Converter
from .kokoro_converter import KokoroConverter
from .llama_converter import LLaMAConverter
from .madlad_converter import MADLADConverter
from .nllb_converter import NLLBConverter
from .wakeword_converter import WakeWordConverter
from .whisper_converter import WhisperConverter

__all__ = [
    "LLaMAConverter",
    "NLLBConverter",
    "MADLADConverter",
    "KokoroConverter",
    "CosyVoice2Converter",
    "WhisperConverter",
    "WakeWordConverter",
]
