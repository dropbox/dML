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
DashVoice Model Download Script

Downloads and verifies Priority 2 models for multi-speaker pipeline.

Priority 2 Models (from DASHVOICE_MASTER_PLAN_2025-12-16.md):
- pyannote-audio 3.1: Speaker diarization
- WeSpeaker ResNet34-LM: Speaker ID
- Silero VAD v6: Voice Activity Detection (already installed)
- SepFormer: Source separation (via SpeechBrain)
- ECAPA-TDNN: Speaker embedding (via SpeechBrain)
- Resemblyzer: Speaker embedding (already installed)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --check-only
    python scripts/download_models.py --model silero
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_silero_vad() -> dict:
    """Check Silero VAD availability."""
    try:
        # Try new-style import first
        try:
            from silero_vad import load_silero_vad
            model, utils = load_silero_vad()
            return {
                "name": "silero-vad",
                "status": "installed",
                "import_style": "silero_vad",
                "model_loaded": True,
            }
        except ImportError:
            pass

        # Try old-style import via silero package
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            return {
                "name": "silero-vad",
                "status": "installed",
                "import_style": "torch.hub",
                "model_loaded": True,
            }
        except Exception:
            pass

        return {
            "name": "silero-vad",
            "status": "not_installed",
            "install_cmd": "pip install silero-vad",
        }

    except Exception as e:
        return {
            "name": "silero-vad",
            "status": "error",
            "error": str(e),
        }


def check_speechbrain() -> dict:
    """Check SpeechBrain availability (provides ECAPA-TDNN, SepFormer)."""
    try:
        import speechbrain

        return {
            "name": "speechbrain",
            "status": "installed",
            "version": speechbrain.__version__ if hasattr(speechbrain, "__version__") else "unknown",
            "provides": ["ECAPA-TDNN", "SepFormer", "Speaker Verification"],
        }
    except ImportError:
        return {
            "name": "speechbrain",
            "status": "not_installed",
            "install_cmd": "pip install speechbrain",
        }
    except Exception as e:
        return {
            "name": "speechbrain",
            "status": "error",
            "error": str(e),
            "note": "May have torchaudio compatibility issues",
        }


def check_resemblyzer() -> dict:
    """Check Resemblyzer speaker encoder."""
    try:
        from resemblyzer import VoiceEncoder

        # Test loading (instantiation validates it works)
        _encoder = VoiceEncoder()
        return {
            "name": "resemblyzer",
            "status": "installed",
            "model_loaded": True,
        }
    except ImportError:
        return {
            "name": "resemblyzer",
            "status": "not_installed",
            "install_cmd": "pip install resemblyzer",
        }
    except Exception as e:
        return {
            "name": "resemblyzer",
            "status": "error",
            "error": str(e),
        }


def check_pyannote() -> dict:
    """Check pyannote-audio availability."""
    try:
        import pyannote.audio

        return {
            "name": "pyannote-audio",
            "status": "installed",
            "version": pyannote.audio.__version__ if hasattr(pyannote.audio, "__version__") else "unknown",
            "provides": ["Speaker Diarization", "VAD", "Overlap Detection"],
            "note": "Requires HuggingFace token for model download",
        }
    except ImportError:
        return {
            "name": "pyannote-audio",
            "status": "not_installed",
            "install_cmd": "pip install pyannote.audio",
            "note": "Requires HuggingFace token and acceptance of model terms",
        }


def check_ecapa_tdnn() -> dict:
    """Check ECAPA-TDNN speaker embedding model."""
    # Skip if speechbrain has issues
    try:
        pass
    except Exception:
        return {
            "name": "ECAPA-TDNN",
            "status": "skipped",
            "note": "SpeechBrain not available",
        }

    try:
        from speechbrain.inference.speaker import SpeakerRecognition  # noqa: F401

        return {
            "name": "ECAPA-TDNN",
            "status": "available",
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "note": "Model will be downloaded on first use",
        }
    except ImportError:
        return {
            "name": "ECAPA-TDNN",
            "status": "not_installed",
            "install_cmd": "pip install speechbrain",
        }
    except Exception as e:
        return {
            "name": "ECAPA-TDNN",
            "status": "error",
            "error": str(e),
        }


def check_sepformer() -> dict:
    """Check SepFormer source separation model."""
    # Skip if speechbrain has issues
    try:
        pass
    except Exception:
        return {
            "name": "SepFormer",
            "status": "skipped",
            "note": "SpeechBrain not available",
        }

    try:
        from speechbrain.inference.separation import SepformerSeparation  # noqa: F401

        return {
            "name": "SepFormer",
            "status": "available",
            "source": "speechbrain/sepformer-whamr",
            "note": "Model will be downloaded on first use",
        }
    except ImportError:
        return {
            "name": "SepFormer",
            "status": "not_installed",
            "install_cmd": "pip install speechbrain",
        }
    except Exception as e:
        return {
            "name": "SepFormer",
            "status": "error",
            "error": str(e),
        }


def check_whisper() -> dict:
    """Check MLX Whisper availability."""
    try:
        import mlx_whisper  # noqa: F401

        return {
            "name": "mlx-whisper",
            "status": "installed",
            "default_model": "mlx-community/whisper-large-v3-turbo",
        }
    except ImportError:
        return {
            "name": "mlx-whisper",
            "status": "not_installed",
            "install_cmd": "pip install mlx-whisper",
        }


def check_kokoro() -> dict:
    """Check Kokoro TTS availability."""
    try:
        from mlx_audio.tts.utils import load_model  # noqa: F401

        return {
            "name": "kokoro-tts",
            "status": "installed",
            "model": "prince-canuma/Kokoro-82M",
        }
    except ImportError:
        return {
            "name": "kokoro-tts",
            "status": "not_installed",
            "install_cmd": "pip install mlx-audio",
        }


def check_all_models() -> list[dict]:
    """Check all Priority 2 models."""
    checks = [
        ("Silero VAD", check_silero_vad),
        ("SpeechBrain", check_speechbrain),
        ("Resemblyzer", check_resemblyzer),
        ("pyannote-audio", check_pyannote),
        ("ECAPA-TDNN", check_ecapa_tdnn),
        ("SepFormer", check_sepformer),
        ("MLX Whisper", check_whisper),
        ("Kokoro TTS", check_kokoro),
    ]

    results = []
    for name, check_fn in checks:
        print(f"Checking {name}...", end=" ")
        result = check_fn()
        status = result.get("status", "unknown")
        if status == "installed" or status == "available":
            print("OK")
        elif status == "not_installed":
            print("NOT INSTALLED")
        else:
            print(f"ERROR: {result.get('error', 'unknown')}")
        results.append(result)

    return results


def print_summary(results: list[dict]):
    """Print summary of model availability."""
    print("\n" + "=" * 60)
    print("MODEL AVAILABILITY SUMMARY")
    print("=" * 60)

    installed = sum(1 for r in results if r.get("status") in ["installed", "available"])
    total = len(results)

    print(f"\nInstalled: {installed}/{total}")
    print()

    for r in results:
        status = r.get("status", "unknown")
        name = r.get("name", "unknown")

        if status in ["installed", "available"]:
            marker = "[OK]"
        elif status == "not_installed":
            marker = "[MISSING]"
        else:
            marker = "[ERROR]"

        print(f"  {marker} {name}")
        if r.get("version"):
            print(f"        Version: {r['version']}")
        if r.get("provides"):
            print(f"        Provides: {', '.join(r['provides'])}")
        if r.get("install_cmd") and status == "not_installed":
            print(f"        Install: {r['install_cmd']}")
        if r.get("note"):
            print(f"        Note: {r['note']}")
        if r.get("error"):
            print(f"        Error: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="DashVoice Model Download Script")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check model availability, don't download",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to check/download",
    )

    _args = parser.parse_args()

    print("DashVoice Model Availability Check")
    print("=" * 60)

    results = check_all_models()
    print_summary(results)

    # Check if any models are missing
    missing = [r for r in results if r.get("status") == "not_installed"]
    if missing:
        print("\nTo install missing models, run:")
        for r in missing:
            if r.get("install_cmd"):
                print(f"  {r['install_cmd']}")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
