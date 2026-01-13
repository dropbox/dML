#!/usr/bin/env python3
"""Probe/download a consolidated emotion dataset from HuggingFace.

This script is intended for quick discovery: it tries a list of likely repos,
prints split sizes, and optionally saves to disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_CANDIDATES = [
    "Walidzlh/combined_emotion_dataset",
    "speechbrain/emotion-recognition",
    "superb/superb",
    "anton-l/emotions",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_repo_root() / "data" / "emotion" / "hf_cache",
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Set HF_HUB_ENABLE_HF_TRANSFER=1 (requires hf_transfer installed)",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=DEFAULT_CANDIDATES,
        help="Dataset repos to try (first successful one stops the search)",
    )
    parser.add_argument(
        "--save-to-disk",
        type=Path,
        default=None,
        help="Optional output directory for dataset.save_to_disk()",
    )
    args = parser.parse_args()

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print("Searching for consolidated emotion datasets on HuggingFace...")
    for repo in args.candidates:
        print(f"\nTrying: {repo}")
        try:
            dataset = load_dataset(
                repo,
                trust_remote_code=True,
                cache_dir=str(args.cache_dir),
            )
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {str(e)[:160]}")
            continue

        print(f"SUCCESS: {repo}")
        print(f"Splits: {list(dataset.keys())}")
        for split in ("train", "validation", "test"):
            if split in dataset:
                print(f"{split}: {len(dataset[split])}")

        if args.save_to_disk is not None:
            args.save_to_disk.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(args.save_to_disk))
            print(f"Saved to: {args.save_to_disk}")
        return 0

    print("\nNo candidates succeeded.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
