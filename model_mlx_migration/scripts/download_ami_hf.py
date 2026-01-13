#!/usr/bin/env python3
"""Download AMI Meeting Corpus from HuggingFace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="edinburghcstr/ami",
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--config",
        default="headset-mix",
        help="Dataset config (AMI uses headset-mix)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_repo_root() / "data" / "ami" / "hf_cache",
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Set HF_HUB_ENABLE_HF_TRANSFER=1 (requires hf_transfer installed)",
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

    print("Downloading AMI corpus from HuggingFace...")
    print("This dataset can be very large depending on config.")
    dataset = load_dataset(
        args.repo,
        args.config,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
    )

    for split in ("train", "validation", "test"):
        if split in dataset:
            print(f"{split}: {len(dataset[split])}")

    if args.save_to_disk is not None:
        args.save_to_disk.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.save_to_disk))
        print(f"Saved to: {args.save_to_disk}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
