#!/usr/bin/env python3
"""Download LibriCSS from HuggingFace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_REPOS = ["fixie-ai/libricss", "speechcolab/libricss"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        nargs="*",
        default=DEFAULT_REPOS,
        help="HuggingFace dataset repos to try (first successful one wins)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_repo_root() / "data" / "libricss" / "hf_cache",
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

    print("Downloading LibriCSS from HuggingFace...")
    for repo in args.repos:
        try:
            dataset = load_dataset(
                repo,
                trust_remote_code=True,
                cache_dir=str(args.cache_dir),
            )
        except Exception as e:
            print(f"[FAIL] {repo}: {type(e).__name__}: {str(e)[:160]}")
            continue

        print(f"[OK] {repo} splits: {list(dataset.keys())}")
        return 0

    print("All repos failed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
