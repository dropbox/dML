#!/usr/bin/env python3
"""
Model Checksum Verification Script

Verifies critical model files have expected sizes and SHA256 checksums.
Run this after downloading/exporting models to ensure integrity.

Usage:
    python scripts/verify_model_checksums.py           # Size check only (fast)
    python scripts/verify_model_checksums.py --hash    # Full SHA256 verification (slow)
    python scripts/verify_model_checksums.py --update  # Update checksums file

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Optional

# Use relative paths from script location
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
CHECKSUMS_FILE = PROJECT_DIR / "scripts" / "model_checksums.json"

# Expected model files with minimum sizes (bytes)
# Format: {relative_path: {"min_size": bytes, "description": str}}
EXPECTED_MODELS = {
    # Kokoro TTS
    "kokoro/kokoro_mps.pt": {
        "min_size": 300_000_000,
        "description": "Kokoro MPS TorchScript model (~328MB)"
    },
    "kokoro/voice_af_heart.pt": {
        "min_size": 500_000,
        "description": "Kokoro voice embedding"
    },

    # NLLB Translation
    "nllb/nllb-200-distilled-600m.pt": {
        "min_size": 2_400_000_000,
        "description": "NLLB-600M full model (~2.4GB)"
    },
    "nllb/nllb-encoder-mps.pt": {
        "min_size": 1_600_000_000,
        "description": "NLLB encoder TorchScript (~1.6GB)"
    },
    "nllb/nllb-decoder-mps.pt": {
        "min_size": 1_800_000_000,
        "description": "NLLB decoder TorchScript (~1.8GB)"
    },
    "nllb/sentencepiece.bpe.model": {
        "min_size": 4_000_000,
        "description": "NLLB tokenizer (~4.8MB)"
    },

    # CosyVoice
    "cosyvoice/cosyvoice_qwen2_q8_0.gguf": {
        "min_size": 500_000_000,
        "description": "CosyVoice LLM GGUF (~550MB)"
    },

    # MMS-TTS (optional)
    "mms-tts/mms_tts_ar_cpu.pt": {
        "min_size": 140_000_000,
        "description": "MMS-TTS Arabic (~140MB)",
        "optional": True
    },
    "mms-tts/mms_tts_tr_cpu.pt": {
        "min_size": 140_000_000,
        "description": "MMS-TTS Turkish (~140MB)",
        "optional": True
    },
    "mms-tts/mms_tts_fa_cpu.pt": {
        "min_size": 140_000_000,
        "description": "MMS-TTS Persian (~140MB)",
        "optional": True
    },

    # Speaker Embedding
    "speaker/ecapa_tdnn.pt": {
        "min_size": 80_000_000,
        "description": "ECAPA-TDNN speaker embedding (~85MB)",
        "optional": True
    },

    # Whisper (GGML)
    "whisper/ggml-large-v3.bin": {
        "min_size": 3_000_000_000,
        "description": "Whisper large-v3 (~3GB)",
        "optional": True
    },
}


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size if path.exists() else 0


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file. Streams for memory efficiency."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):  # 8MB chunks
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checksums() -> dict:
    """Load checksums from JSON file if it exists."""
    if CHECKSUMS_FILE.exists():
        with open(CHECKSUMS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checksums(checksums: dict) -> None:
    """Save checksums to JSON file."""
    with open(CHECKSUMS_FILE, "w") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)
    print(f"Checksums saved to: {CHECKSUMS_FILE}")


def update_checksums() -> bool:
    """Update checksums file with current model hashes."""
    print("=" * 60)
    print("Updating Model Checksums")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    print()

    checksums = {}
    for rel_path, spec in EXPECTED_MODELS.items():
        full_path = MODELS_DIR / rel_path
        if not full_path.exists():
            print(f"  [SKIP] {rel_path} (not found)")
            continue

        actual_size = get_file_size(full_path)
        print(f"  Computing SHA256 for {rel_path}...", end=" ", flush=True)
        sha256_hash = compute_sha256(full_path)
        print(f"done")

        checksums[rel_path] = {
            "size": actual_size,
            "sha256": sha256_hash,
            "description": spec["description"]
        }

    print()
    save_checksums(checksums)
    print(f"Updated {len(checksums)} entries")
    return True


def verify_models(verify_hashes: bool = False) -> bool:
    """Verify all model files exist and have expected sizes/hashes."""
    print("=" * 60)
    print("Model Checksum Verification")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    if verify_hashes:
        print("Mode: Full SHA256 verification (slow)")
    else:
        print("Mode: Size check only (fast, use --hash for full verification)")
    print()

    # Load checksums for hash verification
    checksums = load_checksums() if verify_hashes else {}

    passed = 0
    failed = 0
    skipped = 0
    hash_mismatches = 0

    for rel_path, spec in EXPECTED_MODELS.items():
        full_path = MODELS_DIR / rel_path
        min_size = spec["min_size"]
        description = spec["description"]
        is_optional = spec.get("optional", False)

        if not full_path.exists():
            if is_optional:
                print(f"  [SKIP] {rel_path} (optional, not installed)")
                skipped += 1
            else:
                print(f"  [FAIL] {rel_path} - FILE MISSING")
                print(f"         {description}")
                failed += 1
            continue

        actual_size = get_file_size(full_path)

        # Size check first (fast)
        if actual_size < min_size:
            print(f"  [FAIL] {rel_path}")
            print(f"         Expected: >= {min_size:,} bytes")
            print(f"         Actual:   {actual_size:,} bytes")
            failed += 1
            continue

        # Hash verification if requested
        if verify_hashes:
            expected_checksum = checksums.get(rel_path, {})
            expected_hash = expected_checksum.get("sha256")
            expected_size = expected_checksum.get("size")

            if not expected_hash:
                print(f"  [WARN] {rel_path} - no checksum in checksums.json")
                size_mb = actual_size / 1_000_000
                print(f"         Size OK: {size_mb:.1f}MB (run --update to add checksum)")
                passed += 1
                continue

            # Verify exact size first (fast fail)
            if expected_size and actual_size != expected_size:
                print(f"  [FAIL] {rel_path} - SIZE MISMATCH")
                print(f"         Expected: {expected_size:,} bytes")
                print(f"         Actual:   {actual_size:,} bytes")
                failed += 1
                hash_mismatches += 1
                continue

            # Compute and verify hash
            print(f"  Verifying {rel_path}...", end=" ", flush=True)
            actual_hash = compute_sha256(full_path)
            if actual_hash == expected_hash:
                size_mb = actual_size / 1_000_000
                print(f"OK ({size_mb:.1f}MB)")
                passed += 1
            else:
                print(f"HASH MISMATCH!")
                print(f"  [FAIL] {rel_path}")
                print(f"         Expected: {expected_hash}")
                print(f"         Actual:   {actual_hash}")
                failed += 1
                hash_mismatches += 1
        else:
            # Size-only verification
            size_mb = actual_size / 1_000_000
            print(f"  [PASS] {rel_path} ({size_mb:.1f}MB)")
            passed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if verify_hashes and hash_mismatches > 0:
        print(f"WARNING: {hash_mismatches} hash/size mismatches detected!")
        print("         Models may be corrupted. Re-download or re-export recommended.")
    print("=" * 60)

    return failed == 0


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if "--update" in sys.argv:
        success = update_checksums()
        sys.exit(0 if success else 1)

    verify_hashes = "--hash" in sys.argv
    success = verify_models(verify_hashes=verify_hashes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
