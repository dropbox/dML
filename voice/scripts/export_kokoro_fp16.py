#!/usr/bin/env python3
"""
Export Kokoro TorchScript model to FP16 for faster Metal (MPS) inference.

This is a development helper only. Runtime remains 100% C++ (libtorch).

Usage:
    python scripts/export_kokoro_fp16.py \
        --input models/kokoro/kokoro_mps.pt \
        --output models/kokoro/kokoro_mps_fp16.pt

Notes:
    - Input model is expected to be a TorchScript module (kokoro_mps.pt)
    - Output is written atomically to avoid corrupting existing files
    - Verification step reloads the FP16 file and checks parameter dtype
"""

import argparse
import sys
from pathlib import Path

import torch


def convert_to_fp16(input_path: Path, output_path: Path, verify: bool, force: bool) -> None:
    if not input_path.exists():
        print(f"ERROR: Input model not found: {input_path}")
        sys.exit(1)

    if output_path.exists() and not force:
        print(f"ERROR: Output already exists: {output_path}")
        print("Use --force to overwrite.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fp16] Loading TorchScript model: {input_path}")
    model = torch.jit.load(str(input_path), map_location="cpu")
    model.eval()

    print("[fp16] Converting parameters and buffers to float16...")
    model = model.to(dtype=torch.float16)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"[fp16] Saving FP16 model to temporary path: {tmp_path}")
    torch.jit.save(model, str(tmp_path))
    tmp_size_mb = tmp_path.stat().st_size / (1024 * 1024)
    print(f"[fp16] Temporary file size: {tmp_size_mb:.1f} MB")

    if verify:
        print("[fp16] Verifying saved model dtype...")
        loaded = torch.jit.load(str(tmp_path), map_location="cpu")
        params = list(loaded.parameters())
        if not params:
            print("WARNING: No parameters found during verification.")
        else:
            dtype = params[0].dtype
            if dtype != torch.float16:
                print(f"ERROR: Expected float16 parameters, found {dtype}")
                sys.exit(1)
        print("[fp16] Verification successful (parameters are float16)")

    tmp_path.replace(output_path)
    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[fp16] FP16 model saved: {output_path} ({final_size_mb:.1f} MB)")


if __name__ == "__main__":
    # Use relative paths from script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    default_input = project_dir / "models/kokoro/kokoro_mps.pt"
    default_output = project_dir / "models/kokoro/kokoro_mps_fp16.pt"

    arg_parser = argparse.ArgumentParser(description="Convert Kokoro TorchScript model to FP16.")
    arg_parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to FP32 TorchScript model (default: models/kokoro/kokoro_mps.pt)",
    )
    arg_parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to write FP16 TorchScript model",
    )
    arg_parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Skip reload/parameter dtype verification",
    )
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )

    args = arg_parser.parse_args()
    convert_to_fp16(args.input, args.output, args.verify, args.force)
