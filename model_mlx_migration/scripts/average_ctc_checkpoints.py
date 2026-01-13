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
Checkpoint Averaging for CTC Head - Stochastic Weight Averaging (SWA)

Averages weights from multiple checkpoints to create a more robust model.
This technique can improve generalization and reduce noise in the weights.

Usage:
    # Average 3 checkpoints around step_14000
    python scripts/average_ctc_checkpoints.py \
        --checkpoints checkpoints/ctc_head_large_v3/step_13500.npz \
                      checkpoints/ctc_head_large_v3/step_14000.npz \
                      checkpoints/ctc_head_large_v3/step_14500.npz \
        --output checkpoints/ctc_head_large_v3/averaged_14000.npz

    # Average 5 checkpoints
    python scripts/average_ctc_checkpoints.py \
        --checkpoints checkpoints/ctc_head_large_v3/step_13000.npz \
                      checkpoints/ctc_head_large_v3/step_13500.npz \
                      checkpoints/ctc_head_large_v3/step_14000.npz \
                      checkpoints/ctc_head_large_v3/step_14500.npz \
                      checkpoints/ctc_head_large_v3/step_15000.npz \
        --output checkpoints/ctc_head_large_v3/averaged_14000_5pt.npz
"""

import argparse
from pathlib import Path
from typing import List, Dict

import mlx.core as mx
import numpy as np


def load_checkpoint(path: str) -> Dict[str, mx.array]:
    """Load a CTC checkpoint and return its parameters."""
    return dict(mx.load(path))


def average_checkpoints(checkpoint_paths: List[str]) -> Dict[str, mx.array]:
    """Average weights from multiple checkpoints."""
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    # Load first checkpoint as template
    averaged = load_checkpoint(checkpoint_paths[0])

    # Convert to float32 for accumulation
    averaged = {k: v.astype(mx.float32) for k, v in averaged.items()}

    # Accumulate weights from remaining checkpoints
    for path in checkpoint_paths[1:]:
        checkpoint = load_checkpoint(path)
        for key in averaged:
            if key in checkpoint:
                averaged[key] = averaged[key] + checkpoint[key].astype(mx.float32)
            else:
                print(f"Warning: Key {key} not found in {path}")

    # Average
    n = len(checkpoint_paths)
    averaged = {k: v / n for k, v in averaged.items()}

    return averaged


def save_checkpoint(params: Dict[str, mx.array], output_path: str):
    """Save averaged checkpoint."""
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as .npz
    mx.savez(output_path, **params)

    # Verify save
    loaded = dict(mx.load(output_path))
    assert set(loaded.keys()) == set(params.keys()), "Key mismatch after save"
    print(f"Saved: {output_path}")


def verify_checkpoint(path: str):
    """Verify a checkpoint can be loaded and has expected structure."""
    params = load_checkpoint(path)

    # Check expected keys for CTC head
    expected_prefixes = ["ln", "proj"]
    found_prefixes = set()

    for key in params.keys():
        prefix = key.split(".")[0]
        found_prefixes.add(prefix)

    for prefix in expected_prefixes:
        if prefix not in found_prefixes:
            print(f"Warning: Expected prefix '{prefix}' not found in {path}")

    # Print summary
    total_params = sum(np.prod(v.shape) for v in params.values())
    print(f"  Keys: {len(params)}")
    print(f"  Total params: {total_params:,}")
    print(f"  Keys: {list(params.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Average CTC head checkpoints")
    parser.add_argument("--checkpoints", "-c", type=str, nargs="+", required=True,
                        help="Checkpoint paths to average")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output path for averaged checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Verify checkpoints before averaging")

    args = parser.parse_args()

    print("=" * 70)
    print("CTC Checkpoint Averaging")
    print("=" * 70)
    print(f"Input checkpoints: {len(args.checkpoints)}")
    for i, path in enumerate(args.checkpoints):
        print(f"  {i+1}. {path}")
    print(f"Output: {args.output}")
    print()

    # Verify all checkpoints exist
    for path in args.checkpoints:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Optionally verify structure
    if args.verify:
        print("Verifying checkpoints...")
        for path in args.checkpoints:
            print(f"\n{path}:")
            verify_checkpoint(path)
        print()

    # Average
    print("Averaging weights...")
    averaged = average_checkpoints(args.checkpoints)

    # Save
    print(f"Saving to {args.output}...")
    save_checkpoint(averaged, args.output)

    # Verify output
    print("\nVerifying output checkpoint:")
    verify_checkpoint(args.output)

    # Show size
    output_size = Path(args.output).stat().st_size
    print(f"\nOutput size: {output_size:,} bytes ({output_size / 1024 / 1024:.1f} MB)")

    print("\n" + "=" * 70)
    print("Done! To evaluate, run:")
    print(f"  PYTHONPATH=. python scripts/test_ctc_streaming_eval.py --checkpoint {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
