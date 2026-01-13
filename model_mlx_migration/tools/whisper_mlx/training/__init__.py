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
Training infrastructure for Whisper MLX heads.

This module provides shared infrastructure for training various heads
on top of the frozen Whisper encoder:

- CTC head (51,865 vocab for ASR)
- Kokoro phoneme head (178 vocab for TTS)
- Punctuation head (6 classes for prosody)
- Paralinguistics head (18 classes for emotion/pitch/etc)
- Multi-head (combined training)

Key components:
- BaseTrainer: Common training loop, logging, checkpointing
- OptimizedDataLoader: Mel caching, prefetching, length-sorting
- ctc_loss_mlx: Native MLX CTC loss (validated against PyTorch)
"""

from .base_trainer import BaseTrainer, TrainingConfig
from .ctc_loss_mlx import ctc_loss, ctc_loss_batch
from .data_loader import MelCache, OptimizedDataLoader

__all__ = [
    "ctc_loss",
    "ctc_loss_batch",
    "OptimizedDataLoader",
    "MelCache",
    "BaseTrainer",
    "TrainingConfig",
]
