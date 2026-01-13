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
Download all SOTA models for DashVoice multi-speaker pipeline.

Models are downloaded from HuggingFace Hub and other sources.
All models are commercially licensed (MIT, Apache 2.0, BSD).
"""

import subprocess
import sys


def run_pip(package: str):
    """Install a pip package."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def download_hf_model(repo_id: str, local_dir: str = None):
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id}...")
    path = snapshot_download(repo_id, local_dir=local_dir)
    print(f"  Downloaded to: {path}")
    return path


def main():
    print("=" * 70)
    print("DashVoice Model Downloader")
    print("=" * 70)

    # Ensure dependencies
    print("\n1. Checking dependencies...")
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        run_pip("huggingface_hub")

    # Models to download
    models = {
        # Speaker Diarization
        "pyannote/speaker-diarization-3.1": {
            "purpose": "Speaker diarization (who spoke when)",
            "license": "MIT",
            "size": "~150MB",
        },
        "pyannote/segmentation-3.0": {
            "purpose": "Speaker segmentation",
            "license": "MIT",
            "size": "~20MB",
        },
        # Speaker Embedding / Verification
        "pyannote/wespeaker-voxceleb-resnet34-LM": {
            "purpose": "Speaker embedding (WeSpeaker)",
            "license": "Apache 2.0",
            "size": "~25MB",
        },
        "speechbrain/spkrec-ecapa-voxceleb": {
            "purpose": "Speaker embedding (ECAPA-TDNN)",
            "license": "Apache 2.0",
            "size": "~20MB",
        },
        "nvidia/speakerverification_en_titanet_large": {
            "purpose": "Speaker verification (TitaNet-Large)",
            "license": "Apache 2.0",
            "size": "~100MB",
        },
        # Source Separation
        "speechbrain/sepformer-whamr": {
            "purpose": "Source separation (SepFormer)",
            "license": "Apache 2.0",
            "size": "~150MB",
        },
        # Voice Activity Detection
        "snakers4/silero-vad": {
            "purpose": "Voice activity detection",
            "license": "MIT",
            "size": "~2MB",
        },
        # Audio Quality Metrics
        "sarulab-speech/UTMOS22": {
            "purpose": "MOS estimation (UTMOS)",
            "license": "MIT",
            "size": "~50MB",
        },
    }

    print(f"\n2. Downloading {len(models)} models...")

    downloaded = []
    failed = []

    for repo_id, info in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {repo_id}")
        print(f"Purpose: {info['purpose']}")
        print(f"License: {info['license']}")
        print(f"Size: {info['size']}")
        print("-" * 60)

        try:
            path = download_hf_model(repo_id)
            downloaded.append((repo_id, path))
            print(f"SUCCESS: {repo_id}")
        except Exception as e:
            print(f"FAILED: {repo_id} - {e}")
            failed.append((repo_id, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(f"\nSuccessfully downloaded: {len(downloaded)}/{len(models)}")
    for repo_id, path in downloaded:
        print(f"  [OK] {repo_id}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for repo_id, error in failed:
            print(f"  [FAIL] {repo_id}: {error}")

    # Additional models that need special handling
    print("\n" + "=" * 70)
    print("ADDITIONAL MODELS (manual download/install needed)")
    print("=" * 70)

    additional = [
        ("DeepFilterNet3", "pip install deepfilternet", "Noise reduction"),
        ("Silero VAD (torch)", "pip install silero", "Alternative VAD"),
        ("DTLN", "git clone https://github.com/breizhn/DTLN", "Echo cancellation"),
        ("FullSubNet+", "pip install fullsubnet-plus", "Speech enhancement"),
    ]

    for name, install, purpose in additional:
        print(f"  {name}: {purpose}")
        print(f"    Install: {install}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Run: pip install deepfilternet silero pyannote.audio speechbrain")
    print("2. Accept pyannote terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("3. Create HF token with access to gated models")
    print("=" * 70)


if __name__ == "__main__":
    main()
