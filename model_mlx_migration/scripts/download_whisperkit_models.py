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
WhisperKit CoreML Model Download Script

Downloads WhisperKit CoreML models from HuggingFace for ANE acceleration.
Models are organized as .mlmodelc directories (compiled CoreML format).

Repository: argmaxinc/whisperkit-coreml
Models available:
  - openai_whisper-large-v3 (3.09 GB)
  - openai_whisper-large-v3_turbo (1.5 GB approx)
  - openai_whisper-large-v3-v20240930 (latest version)

Each model contains:
  - AudioEncoder.mlmodelc - The encoder (runs on ANE)
  - MelSpectrogram.mlmodelc - Mel spectrogram computation
  - TextDecoder.mlmodelc - Text decoder
  - config.json, generation_config.json

Usage:
    python scripts/download_whisperkit_models.py
    python scripts/download_whisperkit_models.py --model large-v3-turbo
    python scripts/download_whisperkit_models.py --list
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default models directory
DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models" / "whisperkit"

# Available models from argmaxinc/whisperkit-coreml
AVAILABLE_MODELS = {
    "large-v3": "openai_whisper-large-v3",
    "large-v3-turbo": "openai_whisper-large-v3_turbo",
    "large-v3-v20240930": "openai_whisper-large-v3-v20240930",
    "large-v3-v20240930-turbo": "openai_whisper-large-v3-v20240930_turbo",
    "turbo": "openai_whisper-large-v3_turbo",  # Alias
}

# Components expected in each model
MODEL_COMPONENTS = [
    "AudioEncoder.mlmodelc",
    "MelSpectrogram.mlmodelc",
    "TextDecoder.mlmodelc",
    "config.json",
    "generation_config.json",
]

REPO_ID = "argmaxinc/whisperkit-coreml"


def download_model(
    model_name: str = "large-v3",
    output_dir: Path | None = None,
    encoder_only: bool = False,
    force: bool = False,
) -> Path:
    """
    Download a WhisperKit CoreML model from HuggingFace.

    Args:
        model_name: Short name like "large-v3" or "turbo"
        output_dir: Output directory (default: models/whisperkit)
        encoder_only: If True, only download AudioEncoder (faster)
        force: If True, re-download even if exists

    Returns:
        Path to downloaded model directory
    """
    from huggingface_hub import snapshot_download

    if output_dir is None:
        output_dir = DEFAULT_MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get full model name
    if model_name in AVAILABLE_MODELS:
        hf_model_name = AVAILABLE_MODELS[model_name]
    else:
        # Assume it's already the full HuggingFace path
        hf_model_name = model_name

    model_path = output_dir / hf_model_name

    # Check if already downloaded
    if model_path.exists() and not force:
        encoder_path = model_path / "AudioEncoder.mlmodelc"
        if encoder_path.exists():
            print(f"Model already exists at {model_path}")
            return model_path

    print(f"Downloading {hf_model_name} from {REPO_ID}...")

    # Build allow_patterns for the model
    # CoreML .mlmodelc directories contain many files, need to match all
    if encoder_only:
        # Download encoder, mel spectrogram (needed for input), and configs
        patterns = [
            f"{hf_model_name}/AudioEncoder.mlmodelc/**",
            f"{hf_model_name}/MelSpectrogram.mlmodelc/**",
            f"{hf_model_name}/config.json",
            f"{hf_model_name}/generation_config.json",
        ]
        print("  (encoder only mode - skipping TextDecoder)")
    else:
        # Download entire model directory
        patterns = [f"{hf_model_name}/**"]

    try:
        # Use snapshot_download to get the model
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=patterns,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print(f"Downloaded to: {downloaded_path}")

        # The model should be at output_dir/hf_model_name
        final_path = output_dir / hf_model_name

        # Verify download
        if verify_model(final_path, encoder_only=encoder_only):
            print(f"Model verified at {final_path}")
            return final_path
        else:
            print(f"Warning: Model verification failed at {final_path}")
            return final_path

    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def verify_model(model_path: Path, encoder_only: bool = False) -> bool:
    """
    Verify that a downloaded model has all required components.

    Args:
        model_path: Path to model directory
        encoder_only: If True, only verify encoder exists

    Returns:
        True if model is valid
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return False

    required = ["AudioEncoder.mlmodelc"]
    if not encoder_only:
        required.extend(["MelSpectrogram.mlmodelc", "TextDecoder.mlmodelc"])

    missing = []
    for component in required:
        component_path = model_path / component
        if not component_path.exists():
            missing.append(component)

    if missing:
        print(f"Missing components: {missing}")
        return False

    return True


def list_models(output_dir: Path | None = None):
    """List downloaded models."""
    if output_dir is None:
        output_dir = DEFAULT_MODELS_DIR
    output_dir = Path(output_dir)

    print("Available models to download:")
    for short_name, full_name in AVAILABLE_MODELS.items():
        print(f"  {short_name:25} -> {full_name}")

    print("\nDownloaded models:")
    if not output_dir.exists():
        print("  (none)")
        return

    found = False
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            encoder_exists = (item / "AudioEncoder.mlmodelc").exists()
            decoder_exists = (item / "TextDecoder.mlmodelc").exists()
            status = []
            if encoder_exists:
                status.append("encoder")
            if decoder_exists:
                status.append("decoder")
            if status:
                print(f"  {item.name}: [{', '.join(status)}]")
                found = True

    if not found:
        print("  (none)")


def check_coreml() -> bool:
    """Check if CoreML tools are available."""
    try:
        import coremltools
        print(f"coremltools version: {coremltools.__version__}")
        return True
    except ImportError:
        print("coremltools not installed. Run: pip install coremltools")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download WhisperKit CoreML models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Model to download (default: large-v3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/whisperkit)",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Only download encoder model (faster)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available and downloaded models",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check CoreML tools availability",
    )

    args = parser.parse_args()

    if args.check:
        return 0 if check_coreml() else 1

    if args.list:
        list_models(Path(args.output_dir) if args.output_dir else None)
        return 0

    # Check CoreML first
    if not check_coreml():
        return 1

    # Download model
    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        model_path = download_model(
            model_name=args.model,
            output_dir=output_dir,
            encoder_only=args.encoder_only,
            force=args.force,
        )
        print(f"\nModel ready at: {model_path}")
        return 0
    except Exception as e:
        print(f"Failed to download model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
