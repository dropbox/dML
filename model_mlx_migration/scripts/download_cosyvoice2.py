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
CosyVoice2 Model Download Script

Downloads CosyVoice2-0.5B model files from ModelScope or HuggingFace.
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "cosyvoice2-0.5b": {
        "modelscope": "iic/CosyVoice2-0.5B",
        "huggingface": "FunAudioLLM/CosyVoice2-0.5B",
        "description": "CosyVoice2 0.5B - Flow-matching TTS",
    },
    "cosyvoice-300m": {
        "modelscope": "iic/CosyVoice-300M",
        "huggingface": "FunAudioLLM/CosyVoice-300M",
        "description": "CosyVoice 300M - Smaller flow-matching TTS",
    },
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cosyvoice2"


def download_from_modelscope(model_id: str, local_dir: Path) -> bool:
    """Download model from ModelScope."""
    try:
        from modelscope import snapshot_download

        print(f"Downloading {model_id} from ModelScope...")
        snapshot_download(model_id, local_dir=str(local_dir))
        print(f"Downloaded to {local_dir}")
        return True
    except ImportError:
        print("modelscope not installed. Install with: pip install modelscope")
        return False
    except Exception as e:
        print(f"ModelScope download failed: {e}")
        return False


def download_from_huggingface(model_id: str, local_dir: Path) -> bool:
    """Download model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_id} from HuggingFace...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded to {local_dir}")
        return True
    except ImportError:
        print(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return False
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return False


def download_from_git(model_url: str, local_dir: Path) -> bool:
    """Download model using git clone with LFS."""
    try:
        print(f"Cloning {model_url}...")
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "clone", model_url, str(local_dir)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr}")
            return False

        print(f"Cloned to {local_dir}")
        return True
    except Exception as e:
        print(f"Git clone failed: {e}")
        return False


def list_model_files(model_dir: Path) -> None:
    """List files in downloaded model directory."""
    if not model_dir.exists():
        print(f"Directory not found: {model_dir}")
        return

    print(f"\nModel files in {model_dir}:")
    print("-" * 60)

    total_size = 0
    for f in sorted(model_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            size_str = format_size(size)
            rel_path = f.relative_to(model_dir)
            print(f"  {rel_path}: {size_str}")

    print("-" * 60)
    print(f"Total: {format_size(total_size)}")


def format_size(size_bytes: int | float) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def inspect_pytorch_files(model_dir: Path) -> None:
    """Inspect PyTorch model files in the directory."""
    pt_files = list(model_dir.glob("*.pt"))

    if not pt_files:
        print("No .pt files found")
        return

    print("\nPyTorch model files:")
    print("-" * 60)

    try:
        import torch

        for pt_file in pt_files:
            print(f"\n{pt_file.name}:")
            try:
                # Try loading as state dict first
                data = torch.load(pt_file, map_location="cpu", weights_only=False)

                if isinstance(data, dict):
                    print(f"  Type: dict with {len(data)} keys")
                    for i, (k, v) in enumerate(data.items()):
                        if i >= 10:
                            print(f"  ... and {len(data) - 10} more keys")
                            break
                        if hasattr(v, "shape"):
                            print(f"    {k}: {v.shape}")
                        else:
                            print(f"    {k}: {type(v).__name__}")
                elif hasattr(data, "state_dict"):
                    sd = data.state_dict()
                    print(f"  Type: model with state_dict ({len(sd)} params)")
                else:
                    print(f"  Type: {type(data).__name__}")
            except Exception as e:
                print(f"  Error loading: {e}")

    except ImportError:
        print("torch not installed. Install with: pip install torch")


def main():
    parser = argparse.ArgumentParser(description="Download CosyVoice2 models")
    parser.add_argument(
        "model",
        nargs="?",
        default="cosyvoice2-0.5b",
        choices=list(MODELS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Output directory for model files",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface", "git"],
        default="huggingface",
        help="Download source",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect downloaded model files",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['description']}")
        return 0

    model_info = MODELS[args.model]
    output_dir = args.output_dir / args.model

    if args.inspect:
        list_model_files(output_dir)
        inspect_pytorch_files(output_dir)
        return 0

    print(f"Model: {model_info['description']}")
    print(f"Output: {output_dir}")
    print()

    success = False

    if args.source == "modelscope":
        success = download_from_modelscope(model_info["modelscope"], output_dir)
    elif args.source == "huggingface":
        success = download_from_huggingface(model_info["huggingface"], output_dir)
    elif args.source == "git":
        git_url = f"https://www.modelscope.cn/{model_info['modelscope']}.git"
        success = download_from_git(git_url, output_dir)

    if success:
        list_model_files(output_dir)
        inspect_pytorch_files(output_dir)
        return 0
    else:
        print("\nDownload failed. Try a different source:")
        print("  --source modelscope")
        print("  --source huggingface")
        print("  --source git")
        return 1


if __name__ == "__main__":
    sys.exit(main())
