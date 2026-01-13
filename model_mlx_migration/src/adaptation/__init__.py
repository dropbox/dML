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
Test-time adaptation modules for ASR.

This package provides adaptation methods for improving ASR performance
on domain-shifted or speaker-specific audio without fine-tuning.
"""

from .ebats import (
    EBATS,
    EBATSConfig,
    PromptBank,
    create_ebats,
)

__all__ = [
    "EBATS",
    "EBATSConfig",
    "PromptBank",
    "create_ebats",
]
