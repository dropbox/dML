#!/usr/bin/env python3
"""Download ICSI Meeting Corpus from HuggingFace."""

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
        default="edinburghcstr/icsi",
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_repo_root() / "data" / "icsi" / "hf_cache",
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Set HF_HUB_ENABLE_HF_TRANSFER=1 (requires hf_transfer installed)",
    )
    args = parser.parse_args()

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print("Downloading ICSI corpus from HuggingFace...")
    try:
        dataset = load_dataset(
            args.repo,
            trust_remote_code=True,
            cache_dir=str(args.cache_dir),
        )
    except Exception as e:
        print(f"ICSI not available on HuggingFace: {type(e).__name__}: {str(e)[:160]}")
        print("May need to download from official source")
        return 2

    print(f"Dataset splits: {list(dataset.keys())}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
