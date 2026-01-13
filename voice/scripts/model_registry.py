#!/usr/bin/env python3
"""
Model Registry - Track model exports with signatures and metadata.

Prevents overwriting models during R&D by:
1. Recording SHA256 signatures of all exported models
2. Storing export metadata (dtype, device, script, parameters)
3. Versioning models instead of overwriting
4. Validating models before use

Usage:
    # Register a new model export
    python scripts/model_registry.py register models/kokoro/kokoro_mps.pt \
        --dtype float32 --device mps --script export_kokoro_torchscript.py

    # Verify a model matches registry
    python scripts/model_registry.py verify models/kokoro/kokoro_mps.pt

    # List all registered models
    python scripts/model_registry.py list

    # Export with automatic registration (wrapper)
    python scripts/model_registry.py export \
        --script scripts/export_kokoro_torchscript.py \
        --output models/kokoro/kokoro_mps.pt \
        --dtype float32 --device mps
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Registry file locations
PREFERRED_REGISTRY_FILE = Path(__file__).parent.parent / "models" / "model_registry.json"
LEGACY_REGISTRY_FILE = Path(__file__).parent / "model_registry_data.json"
# Default target for writes; load() may temporarily read from legacy
REGISTRY_FILE = PREFERRED_REGISTRY_FILE


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def resolve_registry_path_for_load() -> Path:
    """Prefer the primary registry path but allow legacy file for compatibility."""
    if PREFERRED_REGISTRY_FILE.exists():
        return PREFERRED_REGISTRY_FILE
    if LEGACY_REGISTRY_FILE.exists():
        return LEGACY_REGISTRY_FILE
    return PREFERRED_REGISTRY_FILE


def load_registry() -> dict:
    """Load the model registry from disk."""
    load_path = resolve_registry_path_for_load()
    if load_path.exists():
        with open(load_path, "r") as f:
            return json.load(f)
    return {"models": {}, "version": "1.0"}


def save_registry(registry: dict) -> None:
    """Save the model registry to disk."""
    target_path = PREFERRED_REGISTRY_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    if target_path != LEGACY_REGISTRY_FILE:
        try:
            LEGACY_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, LEGACY_REGISTRY_FILE)
        except Exception as copy_err:
            print(f"Warning: Failed to mirror registry to legacy path: {copy_err}")
    print(f"Registry saved to {target_path}")


def register_model(
    model_path: Path,
    dtype: str,
    device: str,
    script: str,
    script_args: Optional[str] = None,
    notes: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Register a model in the registry.

    If model already exists with different signature, creates a versioned backup.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    registry = load_registry()
    repo_root = Path(__file__).parent.parent.resolve()
    model_abs = model_path.resolve()
    if model_abs.is_relative_to(repo_root):
        model_key = str(model_abs.relative_to(repo_root))
    else:
        model_key = str(model_abs)

    # Compute signature
    sha256 = compute_sha256(model_path)
    size_bytes = get_file_size(model_path)
    size_mb = size_bytes / (1024 * 1024)

    # Check if model already registered
    if model_key in registry["models"]:
        existing = registry["models"][model_key]
        if existing["sha256"] == sha256:
            print(f"Model already registered with same signature: {model_key}")
            return existing

        if not force:
            # Create versioned backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_path.stem}_{timestamp}{model_path.suffix}"
            backup_path = model_path.parent / "archive" / backup_name
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"WARNING: Model signature changed!")
            print(f"  Old: {existing['sha256'][:16]}... ({existing['size_mb']:.1f}MB, {existing['dtype']})")
            print(f"  New: {sha256[:16]}... ({size_mb:.1f}MB, {dtype})")
            print(f"  Backing up old model to: {backup_path}")

            # Copy current to archive (don't move - keep the new one)
            # Actually, we should archive the OLD entry's info, not the file
            if "history" not in registry["models"][model_key]:
                registry["models"][model_key]["history"] = []
            registry["models"][model_key]["history"].append({
                "sha256": existing["sha256"],
                "size_bytes": existing["size_bytes"],
                "size_mb": existing["size_mb"],
                "dtype": existing["dtype"],
                "device": existing["device"],
                "archived_at": datetime.now().isoformat(),
            })

    # Create new entry
    entry = {
        "sha256": sha256,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "dtype": dtype,
        "device": device,
        "export_script": script,
        "export_args": script_args,
        "exported_at": datetime.now().isoformat(),
        "notes": notes,
    }

    # Preserve history if exists
    if model_key in registry["models"] and "history" in registry["models"][model_key]:
        entry["history"] = registry["models"][model_key]["history"]

    registry["models"][model_key] = entry
    save_registry(registry)

    print(f"Registered: {model_key}")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  Size: {size_mb:.1f}MB")
    print(f"  dtype: {dtype}, device: {device}")

    return entry


def verify_model(model_path: Path) -> bool:
    """
    Verify a model matches its registry entry.

    Returns True if valid, False if mismatch or not registered.
    """
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return False

    registry = load_registry()
    repo_root = Path(__file__).parent.parent.resolve()
    model_abs = model_path.resolve()
    if model_abs.is_relative_to(repo_root):
        model_key = str(model_abs.relative_to(repo_root))
    else:
        model_key = str(model_abs)

    if model_key not in registry["models"]:
        print(f"WARNING: Model not registered: {model_key}")
        print(f"  Run: python scripts/model_registry.py register {model_path} --dtype <dtype> --device <device> --script <script>")
        return False

    entry = registry["models"][model_key]

    # Compute current signature
    sha256 = compute_sha256(model_path)
    size_bytes = get_file_size(model_path)
    size_mb = size_bytes / (1024 * 1024)

    # Check signature
    if sha256 != entry["sha256"]:
        print(f"ERROR: Model signature mismatch: {model_key}")
        print(f"  Expected: {entry['sha256'][:16]}... ({entry['size_mb']:.1f}MB)")
        print(f"  Actual:   {sha256[:16]}... ({size_mb:.1f}MB)")
        print(f"  Model may be corrupted or was modified without updating registry!")
        return False

    # Check size (redundant but useful for quick checks)
    if size_bytes != entry["size_bytes"]:
        print(f"WARNING: Size mismatch but SHA256 matches (unusual)")
        print(f"  Expected: {entry['size_bytes']} bytes")
        print(f"  Actual:   {size_bytes} bytes")

    print(f"VERIFIED: {model_key}")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  Size: {size_mb:.1f}MB")
    print(f"  dtype: {entry['dtype']}, device: {entry['device']}")
    print(f"  Exported: {entry['exported_at']}")

    return True


def list_models() -> None:
    """List all registered models."""
    registry = load_registry()

    if not registry["models"]:
        print("No models registered.")
        return

    print(f"Registered Models ({len(registry['models'])} total):")
    print("-" * 80)

    for model_key, entry in sorted(registry["models"].items()):
        status = "?"
        model_path = Path(__file__).parent.parent / model_key
        if model_path.exists():
            current_sha = compute_sha256(model_path)
            if current_sha == entry["sha256"]:
                status = "OK"
            else:
                status = "MISMATCH"
        else:
            status = "MISSING"

        print(f"[{status:8}] {model_key}")
        print(f"           {entry['size_mb']:.1f}MB | {entry['dtype']} | {entry['device']} | {entry['exported_at'][:10]}")
        if entry.get("history"):
            print(f"           ({len(entry['history'])} previous versions)")

    print("-" * 80)


def export_with_registry(
    script: str,
    output: Path,
    dtype: str,
    device: str,
    extra_args: Optional[list] = None,
) -> bool:
    """
    Run an export script and automatically register the result.

    Prevents overwriting by checking if output already exists with different params.
    """
    registry = load_registry()
    repo_root = Path(__file__).parent.parent.resolve()
    output_abs = output.resolve()
    if output_abs.is_relative_to(repo_root):
        model_key = str(output_abs.relative_to(repo_root))
    else:
        model_key = str(output_abs)

    # Check if we'd be overwriting
    if output.exists() and model_key in registry["models"]:
        existing = registry["models"][model_key]
        if existing["dtype"] != dtype or existing["device"] != device:
            print(f"WARNING: Export would change model parameters!")
            print(f"  Current: dtype={existing['dtype']}, device={existing['device']}")
            print(f"  New:     dtype={dtype}, device={device}")

            # Create versioned output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_output = output.parent / f"{output.stem}_{dtype}_{device}_{timestamp}{output.suffix}"
            print(f"  Using versioned output: {versioned_output}")
            output = versioned_output

    # Build command
    cmd = [sys.executable, script, "--output", str(output), "--device", device]
    if dtype:
        cmd.extend(["--dtype", dtype])
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Export failed with code {result.returncode}")
        return False

    if not output.exists():
        print(f"ERROR: Export script did not create output file: {output}")
        return False

    # Register the new model
    register_model(
        output,
        dtype=dtype,
        device=device,
        script=script,
        script_args=" ".join(extra_args) if extra_args else None,
    )

    return True


def verify_all() -> bool:
    """Verify all registered models."""
    registry = load_registry()

    if not registry["models"]:
        print("No models registered.")
        return True

    all_valid = True
    for model_key in registry["models"]:
        model_path = Path(__file__).parent.parent / model_key
        if not verify_model(model_path):
            all_valid = False

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Model Registry - Track model exports with signatures"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a model")
    register_parser.add_argument("model_path", type=Path, help="Path to model file")
    register_parser.add_argument("--dtype", required=True, help="Model dtype (float16, float32)")
    register_parser.add_argument("--device", required=True, help="Target device (cpu, mps, cuda)")
    register_parser.add_argument("--script", required=True, help="Export script used")
    register_parser.add_argument("--args", help="Export script arguments")
    register_parser.add_argument("--notes", help="Additional notes")
    register_parser.add_argument("--force", action="store_true", help="Force overwrite without backup")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a model")
    verify_parser.add_argument("model_path", type=Path, nargs="?", help="Path to model file (or verify all)")

    # List command
    subparsers.add_parser("list", help="List all registered models")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export and register a model")
    export_parser.add_argument("--script", required=True, help="Export script to run")
    export_parser.add_argument("--output", type=Path, required=True, help="Output path")
    export_parser.add_argument("--dtype", required=True, help="Model dtype")
    export_parser.add_argument("--device", required=True, help="Target device")
    export_parser.add_argument("extra_args", nargs="*", help="Extra args for export script")

    args = parser.parse_args()

    if args.command == "register":
        register_model(
            args.model_path,
            dtype=args.dtype,
            device=args.device,
            script=args.script,
            script_args=args.args,
            notes=args.notes,
            force=args.force,
        )
    elif args.command == "verify":
        if args.model_path:
            success = verify_model(args.model_path)
        else:
            success = verify_all()
        sys.exit(0 if success else 1)
    elif args.command == "list":
        list_models()
    elif args.command == "export":
        success = export_with_registry(
            args.script,
            args.output,
            args.dtype,
            args.device,
            args.extra_args,
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
