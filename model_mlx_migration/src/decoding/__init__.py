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
Decoding modules for ASR output combination and post-processing.

This package provides methods for combining multiple ASR hypotheses
and selecting the best output.

Modules:
- rover: ROVER voting algorithm for hypothesis combination
- rover_integration: Integration of multiple ASR sources (Whisper, CTC, Transducer)
"""

from .rover import (
    ROVER,
    AlignedHypothesis,
    Hypothesis,
    ROVERConfig,
    ROVERResult,
    align_hypotheses,
    vote_rover,
)
from .rover_integration import (
    CTCROVERSource,
    # High-accuracy decoder
    HighAccuracyDecoder,
    HighAccuracyResult,
    # Base classes
    ROVERSource,
    SourceResult,
    TransducerROVERSource,
    # Source implementations
    WhisperROVERSource,
    # Convenience functions
    combine_whisper_transducer,
)

__all__ = [
    # ROVER core
    "ROVER",
    "ROVERConfig",
    "ROVERResult",
    "Hypothesis",
    "AlignedHypothesis",
    "vote_rover",
    "align_hypotheses",
    # ROVER integration
    "ROVERSource",
    "SourceResult",
    "WhisperROVERSource",
    "CTCROVERSource",
    "TransducerROVERSource",
    "HighAccuracyDecoder",
    "HighAccuracyResult",
    "combine_whisper_transducer",
]
