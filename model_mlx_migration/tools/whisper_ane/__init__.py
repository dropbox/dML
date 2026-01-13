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
WhisperANE - CoreML acceleration for WhisperMLX.

This module provides CoreML-based encoder acceleration. On M4 Max,
CPU_AND_GPU mode is fastest (~200ms for 30s audio). ANE may be
faster on M2/M3 or when GPU is busy.

Usage:
    from tools.whisper_ane import CoreMLEncoder, HybridWhisperMLX

    # Use CoreML encoder directly
    encoder = CoreMLEncoder.from_pretrained("large-v3")
    encoder_output = encoder(mel_spectrogram)

    # Or use the hybrid model (CoreML encoder + MLX decoder)
    model = HybridWhisperMLX.from_pretrained("large-v3")
    result = model.transcribe("audio.wav")
"""

from .encoder import CoreMLEncoder
from .hybrid import HybridWhisperMLX

__all__ = ["CoreMLEncoder", "HybridWhisperMLX"]
