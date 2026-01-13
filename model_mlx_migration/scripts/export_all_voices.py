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

"""Export all Kokoro voice packs to safetensors format.

Exports each voice as a safetensors file containing:
- embedding: The full [510, 1, 256] voice pack tensor

Usage:
    python scripts/export_all_voices.py [output_dir]

Output:
    output_dir/voices/{voice_name}.safetensors
"""

import sys
from pathlib import Path
from typing import List

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.numpy import save_file as save_safetensors


def get_all_voices(model_id: str = "hexgrad/Kokoro-82M") -> List[str]:
    """Get list of all available voices from HuggingFace."""
    files = list_repo_files(model_id)
    voices = [
        f.replace("voices/", "").replace(".pt", "")
        for f in files
        if f.startswith("voices/") and f.endswith(".pt")
    ]
    return sorted(voices)


def export_voice(
    voice_name: str,
    output_dir: Path,
    model_id: str = "hexgrad/Kokoro-82M",
) -> bool:
    """Export a single voice to safetensors format."""
    try:
        # Download voice file
        voice_file = hf_hub_download(
            model_id,
            f"voices/{voice_name}.pt",
        )

        # Load using torch
        voice_data = torch.load(voice_file, map_location="cpu", weights_only=True)

        if not isinstance(voice_data, torch.Tensor):
            print(f"  ERROR: Unknown voice format: {type(voice_data)}")
            return False

        # Convert to numpy
        voice_np = voice_data.numpy()

        # Verify shape [510, 1, 256] or similar
        print(f"  Shape: {voice_np.shape}, dtype: {voice_np.dtype}")

        # Save as safetensors
        output_path = output_dir / f"{voice_name}.safetensors"
        save_safetensors({"embedding": voice_np}, str(output_path))

        # Verify file size
        size_kb = output_path.stat().st_size / 1024
        print(f"  Saved: {output_path.name} ({size_kb:.1f} KB)")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    # Default output directory
    output_dir = Path("kokoro_cpp_export/voices")

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1]) / "voices"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Get all voices
    print("Fetching voice list from HuggingFace...")
    voices = get_all_voices()
    print(f"Found {len(voices)} voices")
    print()

    # Export each voice
    success_count = 0
    fail_count = 0

    for i, voice in enumerate(voices, 1):
        print(f"[{i}/{len(voices)}] Exporting {voice}...")

        # Check if already exists
        output_path = output_dir / f"{voice}.safetensors"
        if output_path.exists():
            print("  SKIP: Already exists")
            success_count += 1
            continue

        if export_voice(voice, output_dir):
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Export complete: {success_count} succeeded, {fail_count} failed")
    print(f"Voice files saved to: {output_dir}")

    # List by language
    print()
    print("Voices by language:")
    prefixes = {
        "af_": "American Female",
        "am_": "American Male",
        "bf_": "British Female",
        "bm_": "British Male",
        "ef_": "Spanish Female",
        "em_": "Spanish Male",
        "ff_": "French Female",
        "hf_": "Hindi Female",
        "hm_": "Hindi Male",
        "if_": "Italian Female",
        "im_": "Italian Male",
        "jf_": "Japanese Female",
        "jm_": "Japanese Male",
        "pf_": "Portuguese Female",
        "pm_": "Portuguese Male",
        "zf_": "Chinese Female",
        "zm_": "Chinese Male",
    }

    counts = {}
    for voice in voices:
        prefix = voice[:3]
        if prefix in prefixes:
            lang = prefixes[prefix]
            counts[lang] = counts.get(lang, 0) + 1

    for lang, count in sorted(counts.items()):
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
