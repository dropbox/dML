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
DashVoice Quality Validation Framework

Provides unified quality evaluation for all DashVoice models:
- Translation (MADLAD): BLEU, semantic similarity, hallucination detection
- TTS (Kokoro, CosyVoice2): WER, MOS estimation, speaker similarity
- STT (Whisper): WER, CER, language detection accuracy
- LLM-as-Judge: GPT-based quality evaluation for complex assessments
"""

from .quality_framework import (
    LLMJudge,
    QualityResult,
    QualityTarget,
    QualityValidator,
    STTValidator,
    TranslationValidator,
    TTSValidator,
    run_quality_tests,
)

__all__ = [
    "QualityResult",
    "QualityTarget",
    "QualityValidator",
    "TranslationValidator",
    "TTSValidator",
    "STTValidator",
    "LLMJudge",
    "run_quality_tests",
]
