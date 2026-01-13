#!/usr/bin/env python3
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
Check timestamp handling in Python vs C++
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

# C++ tokens
CPP_TOKENS = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496, 34353, 82, 366, 382, 4048, 382, 257, 361, 18459, 13065, 33660, 2779, 74, 325, 17114, 311, 29822, 29822, 7563, 293, 275, 22943, 275, 22943, 275, 50257]

# Python tokens (from debug_no_vad output)
PY_TOKENS = [50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496]

def main():
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # Get timestamp range
    timestamp_begin = tokenizer.timestamp_begin
    eot = tokenizer.eot
    print(f"timestamp_begin: {timestamp_begin}")
    print(f"eot: {eot}")

    # Check C++ tokens
    print("\n=== C++ Tokens Analysis ===")
    print(f"Total: {len(CPP_TOKENS)} tokens")

    # Find special tokens
    timestamps = []
    for i, t in enumerate(CPP_TOKENS):
        if t >= timestamp_begin and t < eot:
            ts_value = (t - timestamp_begin) * 0.02
            timestamps.append((i, t, ts_value))
            print(f"  Position {i}: token {t} = timestamp {ts_value:.2f}s")
        elif t == eot:
            print(f"  Position {i}: token {t} = EOT")

    # Check for repetition patterns
    print("\n=== Repetition Analysis (C++) ===")
    for i in range(len(CPP_TOKENS) - 1):
        if CPP_TOKENS[i] == CPP_TOKENS[i+1] and CPP_TOKENS[i] < timestamp_begin:
            text = tokenizer.decode([CPP_TOKENS[i]])
            print(f"  Position {i}-{i+1}: token {CPP_TOKENS[i]} ({repr(text)}) repeated")

    # Count unique tokens
    unique = set(CPP_TOKENS)
    print(f"\n  Unique tokens: {len(unique)} / {len(CPP_TOKENS)}")

    # Check for 275+22943 pattern (mrs)
    for i in range(len(CPP_TOKENS) - 1):
        if CPP_TOKENS[i] == 275 and CPP_TOKENS[i+1] == 22943:
            print(f"  Position {i}: 'mrs' pattern (275+22943)")

if __name__ == "__main__":
    main()
