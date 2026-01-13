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
Sweep different speculative decoding configurations to find optimal settings.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark():
    from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
    ]

    print("Loading model...")
    converter = MADLADConverter(dtype="bfloat16", quantize=8)
    converter.load()

    # Load trained head
    head_path = Path("models/early_exit_heads/madlad400-3b-mt_exit4_head.safetensors")
    if head_path.exists():
        converter.load_early_exit_head(str(head_path))

    # Baseline
    print("\n--- Baseline ---")
    baseline_times = []
    for text in test_texts:
        start = time.time()
        result = converter.translate(text, tgt_lang="de")
        elapsed = (time.time() - start) * 1000
        baseline_times.append(elapsed)
    avg_baseline = sum(baseline_times) / len(baseline_times)
    print(f"Average: {avg_baseline:.1f}ms")

    # Sweep draft tokens
    print("\n--- Sweep num_draft_tokens ---")
    for num_draft in [2, 4, 6, 8, 10]:
        times = []
        acceptances = []
        for text in test_texts:
            result = converter.translate_speculative(text, tgt_lang="de", num_draft_tokens=num_draft)
            times.append(result.latency_ms)
            acceptances.append(result.acceptance_rate)
        avg_time = sum(times) / len(times)
        avg_acc = sum(acceptances) / len(acceptances)
        speedup = avg_baseline / avg_time
        print(f"  draft={num_draft}: {avg_time:.1f}ms (acc: {avg_acc:.1%}, speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    benchmark()
