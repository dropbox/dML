#!/usr/bin/env python3
"""Download emotion datasets from HuggingFace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DATASETS: list[tuple[str, str, str | None]] = [
    ("consolidated_66k", "WadhahZulkilfie/combined_emotion_dataset", None),
    ("iemocap", "speechbrain/iemocap", None),
    ("resd", "Aniemore/resd", None),
    ("ravdess", "narad/ravdess", None),
    ("crema_d", "ylacombe/crema-d", None),
    ("emov_db", "HuggingFaceTB/emov_db", None),
    ("meld", "declare-lab/MELD", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "data" / "emotion",
        help="Directory to save datasets via save_to_disk()",
    )
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
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset()",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only download these dataset short names (e.g., ravdess iemocap)",
    )
    args = parser.parse_args()

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    for name, repo, config in DATASETS:
        if selected and name not in selected:
            continue

        output_path = args.output_dir / name
        if output_path.exists() and any(output_path.iterdir()):
            print(f"[SKIP] {name} already exists at {output_path}")
            continue

        print(f"\n[DOWN] {name}")
        print(f"  Repo: {repo}")

        try:
            load_kwargs = {
                "trust_remote_code": args.trust_remote_code,
                "cache_dir": str(args.cache_dir),
            }
            ds = (
                load_dataset(repo, config, **load_kwargs)
                if config
                else load_dataset(repo, **load_kwargs)
            )

            print(f"  Splits: {list(ds.keys())}")
            total = sum(len(ds[s]) for s in ds.keys())
            print(f"  Total samples: {total}")

            output_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(output_path))
            print(f"  Saved to {output_path}")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:160]}")

    print("\n=== Download Complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
