#!/usr/bin/env python3
"""
Model Asset Verifier - Hash/Size validation for reproducible performance tests

This script verifies that model files match expected hashes and sizes before
running performance benchmarks. This prevents false regressions from:
- Corrupted model downloads
- Wrong model versions
- Incomplete transfers

Usage:
    # Verify all models
    python scripts/verify_model_assets.py

    # Generate new manifest (after updating models)
    python scripts/verify_model_assets.py --generate-manifest

    # Verify specific category
    python scripts/verify_model_assets.py --category kokoro

    # Use in pytest (imports as module)
    from scripts.verify_model_assets import verify_all_models

Worker #222 - Audit Item #18: Model asset verifier (hash/size check)
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "model_manifest.json"


@dataclass
class ModelAsset:
    """Represents a model file with expected hash and size."""
    path: str           # Relative to models/
    size_bytes: int     # Expected file size
    sha256_prefix: str  # First 16 chars of SHA-256 (fast comparison)
    description: str    # Human-readable description


# Critical model files required for TTS/STT/Translation
# These contain expected hashes from model_manifest.json for verification
CRITICAL_MODELS = {
    "kokoro": [
        ModelAsset(
            path="kokoro/kokoro_mps.pt",
            size_bytes=328177931,
            sha256_prefix="1f387f1ad9cb6c3b",
            description="Kokoro TTS model (MPS/Metal, complex STFT)"
        ),
        ModelAsset(
            path="kokoro/voice_af_heart.pt",
            size_bytes=523866,
            sha256_prefix="97ce71c8f40b177c",
            description="Default English voice (af_heart) - full [510,1,256] voice pack"
        ),
        ModelAsset(
            path="kokoro/lexicon/us_gold.json",
            size_bytes=3000559,
            sha256_prefix="ca86bf361aedc8e5",
            description="English lexicon"
        ),
    ],
    "nllb": [
        ModelAsset(
            path="nllb/nllb-encoder-mps.pt",
            size_bytes=1658517676,
            sha256_prefix="15dd5ad6a21857a0",
            description="NLLB encoder (MPS)"
        ),
        ModelAsset(
            path="nllb/nllb-decoder-mps.pt",
            size_bytes=1860315440,
            sha256_prefix="7ceca1b9c3785662",
            description="NLLB decoder (MPS)"
        ),
        ModelAsset(
            path="nllb/nllb-decoder-first-mps.pt",
            size_bytes=1860285178,
            sha256_prefix="5285940c6e7e4f94",
            description="NLLB decoder first token (MPS)"
        ),
        ModelAsset(
            path="nllb/nllb-decoder-kvcache-mps.pt",
            size_bytes=1860296562,
            sha256_prefix="f2180d721549e22a",
            description="NLLB decoder with KV cache (MPS)"
        ),
        ModelAsset(
            path="nllb/sentencepiece.bpe.model",
            size_bytes=4852054,
            sha256_prefix="14bb8dfb35c0ffde",
            description="NLLB tokenizer"
        ),
    ],
    "whisper": [
        ModelAsset(
            path="whisper/ggml-large-v3.bin",
            size_bytes=3095033483,
            sha256_prefix="64d182b440b98d52",
            description="Whisper large-v3 (GGML)"
        ),
        ModelAsset(
            path="whisper/ggml-silero-v6.2.0.bin",
            size_bytes=885098,
            sha256_prefix="2aa269b785eeb53a",
            description="Silero VAD model"
        ),
    ],
}


def sha256_prefix(filepath: Path, prefix_len: int = 16) -> str:
    """Calculate SHA-256 hash prefix for a file (fast for large files)."""
    sha256 = hashlib.sha256()
    # Read in chunks for large files
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):  # 8MB chunks
            sha256.update(chunk)
    return sha256.hexdigest()[:prefix_len]


def verify_model(asset: ModelAsset, models_dir: Path) -> tuple[bool, str]:
    """
    Verify a single model file.

    Returns:
        (success, message) tuple
    """
    filepath = models_dir / asset.path

    # Check existence
    if not filepath.exists():
        return False, f"MISSING: {asset.path}"

    # Check size (if specified)
    if asset.size_bytes > 0:
        actual_size = filepath.stat().st_size
        if actual_size != asset.size_bytes:
            return False, f"SIZE MISMATCH: {asset.path} (expected {asset.size_bytes}, got {actual_size})"

    # Check hash (if specified)
    if asset.sha256_prefix:
        actual_hash = sha256_prefix(filepath)
        if not actual_hash.startswith(asset.sha256_prefix):
            return False, f"HASH MISMATCH: {asset.path} (expected {asset.sha256_prefix}..., got {actual_hash}...)"

    return True, f"OK: {asset.path}"


def verify_category(category: str, models_dir: Path = MODELS_DIR) -> tuple[bool, list[str]]:
    """
    Verify all models in a category.

    Returns:
        (all_passed, messages) tuple
    """
    if category not in CRITICAL_MODELS:
        return False, [f"Unknown category: {category}"]

    messages = []
    all_passed = True

    for asset in CRITICAL_MODELS[category]:
        passed, msg = verify_model(asset, models_dir)
        messages.append(msg)
        if not passed:
            all_passed = False

    return all_passed, messages


def verify_all_models(models_dir: Path = MODELS_DIR) -> tuple[bool, dict[str, list[str]]]:
    """
    Verify all model categories.

    Returns:
        (all_passed, {category: messages}) tuple
    """
    results = {}
    all_passed = True

    for category in CRITICAL_MODELS:
        passed, messages = verify_category(category, models_dir)
        results[category] = messages
        if not passed:
            all_passed = False

    return all_passed, results


def generate_manifest(models_dir: Path = MODELS_DIR) -> dict:
    """
    Generate a manifest of current model files with hashes.
    Use this after updating models to create a new baseline.
    """
    manifest = {}

    for category, assets in CRITICAL_MODELS.items():
        manifest[category] = []
        for asset in assets:
            filepath = models_dir / asset.path
            if filepath.exists():
                stat = filepath.stat()
                manifest[category].append({
                    "path": asset.path,
                    "size_bytes": stat.st_size,
                    "sha256_prefix": sha256_prefix(filepath),
                    "description": asset.description,
                })
            else:
                manifest[category].append({
                    "path": asset.path,
                    "status": "MISSING",
                    "description": asset.description,
                })

    return manifest


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> Optional[dict]:
    """Load existing manifest file."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict, manifest_path: Path = MANIFEST_PATH):
    """Save manifest to JSON file."""
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify model assets for reproducible performance tests"
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Generate new manifest from current model files"
    )
    parser.add_argument(
        "--category",
        choices=list(CRITICAL_MODELS.keys()),
        help="Verify only a specific category"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output on failure"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Models directory (default: {MODELS_DIR})"
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip hash verification (faster, only check size/existence)"
    )

    args = parser.parse_args()

    # Update verification behavior based on --skip-hash
    if args.skip_hash:
        # Clear all hash prefixes to skip hash checking
        for category in CRITICAL_MODELS:
            for asset in CRITICAL_MODELS[category]:
                # Use object attribute to modify in-place (dataclass is mutable)
                object.__setattr__(asset, 'sha256_prefix', "")

    if args.generate_manifest:
        print("Generating model manifest...")
        manifest = generate_manifest(args.models_dir)
        save_manifest(manifest)
        print("\nManifest contents:")
        print(json.dumps(manifest, indent=2))
        return 0

    # Verify models
    if args.category:
        passed, messages = verify_category(args.category, args.models_dir)
        categories_results = {args.category: messages}
    else:
        passed, categories_results = verify_all_models(args.models_dir)

    # Output results
    if not args.quiet or not passed:
        print("=" * 60)
        print("Model Asset Verification Report")
        print("=" * 60)

        for category, messages in categories_results.items():
            print(f"\n[{category.upper()}]")
            for msg in messages:
                status = "PASS" if msg.startswith("OK") else "FAIL"
                print(f"  [{status}] {msg}")

        print("\n" + "=" * 60)
        if passed:
            print("RESULT: ALL MODELS VERIFIED")
        else:
            print("RESULT: VERIFICATION FAILED")
            print("\nPerformance tests may produce invalid results!")
            print("Fix model files or regenerate manifest with --generate-manifest")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
