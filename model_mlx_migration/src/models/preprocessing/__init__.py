# Copyright 2024-2026 Andrew Yates
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
Audio preprocessing pipeline for SOTA++ Voice Server.

Implements the preprocessing chain:
1. Resample (sinc interpolation) - ~1ms
2. DC removal (subtract mean) - <1ms
3. AGC (LUFS normalization) - <1ms
4. VAD (Silero) - ~10ms
5. Denoising - SKIP (critical per "Denoising Hurts ASR" paper)
6. Chunking - 320ms chunks

Total target latency: <15ms
"""

from .pipeline import (
    AudioChunk,
    PreprocessingConfig,
    PreprocessingPipeline,
    VADResult,
)

__all__ = [
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "AudioChunk",
    "VADResult",
]
