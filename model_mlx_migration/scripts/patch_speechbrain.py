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
Patch SpeechBrain for torchaudio 2.9+ compatibility.

SpeechBrain checks for torchaudio backends using torchaudio.list_audio_backends(),
which was removed in torchaudio 2.9+. This script patches the backend check to
handle the new torchaudio API.

Usage:
    python scripts/patch_speechbrain.py

Requirements:
    - SpeechBrain installed: pip install speechbrain
    - torchaudio 2.9+ installed
    - torchcodec installed: pip install torchcodec (required for audio loading)
"""

import site
import sys
from pathlib import Path

PATCHED_BACKEND_PY = '''"""Library for checking the torchaudio backend.

Authors
-------
 * Mirco Ravanelli 2021
 * Adel Moumen 2025
 * Patched for torchaudio 2.x compatibility
"""

import platform
from typing import Optional

import torchaudio

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def try_parse_torchaudio_major_version() -> Optional[int]:
    """Tries parsing the torchaudio major version.

    Returns
    -------
    The parsed major version, otherwise ``None``.
    """
    if not hasattr(torchaudio, "__version__"):
        return None

    version_split = torchaudio.__version__.split(".")

    # expect in format x.y.z whatever; we care only about x

    if len(version_split) <= 2:
        # not sure how to parse this
        return None

    try:
        major_version = int(version_split[0])
        minor_version = int(version_split[1])
    except Exception:
        return None

    return major_version, minor_version


def check_torchaudio_backend():
    """Checks the torchaudio backend and sets it to soundfile if
    windows is detected.

    PATCHED: torchaudio 2.x removed list_audio_backends(), so we skip the check
    for versions >= 2.9. In torchaudio 2.x, FFmpeg is used by default.
    """
    result = try_parse_torchaudio_major_version()
    if result is None:
        logger.warning(
            "Failed to detect torchaudio major version; unsure how to check your setup. We recommend that you keep torchaudio up-to-date."
        )
        return

    torchaudio_major, torchaudio_minor = result

    # torchaudio 2.9+ removed list_audio_backends() - uses FFmpeg by default
    if torchaudio_major >= 2 and torchaudio_minor >= 9:
        # Skip backend check - torchaudio 2.9+ uses FFmpeg natively
        logger.debug(
            f"torchaudio {torchaudio_major}.{torchaudio_minor} detected. Using native FFmpeg backend."
        )
        return

    # For older torchaudio 2.x versions (2.1-2.8), check if list_audio_backends exists
    if torchaudio_major >= 2 and torchaudio_minor >= 1:
        if hasattr(torchaudio, 'list_audio_backends'):
            available_backends = torchaudio.list_audio_backends()
            if len(available_backends) == 0:
                logger.warning(
                    "SpeechBrain could not find any working torchaudio backend. Audio files may fail to load. Follow this link for instructions and troubleshooting: https://speechbrain.readthedocs.io/en/latest/audioloading.html"
                )
        else:
            # list_audio_backends doesn't exist, assume FFmpeg is available
            logger.debug(
                f"torchaudio {torchaudio_major}.{torchaudio_minor} without list_audio_backends. Assuming FFmpeg backend."
            )
    else:
        logger.warning(
            "This version of torchaudio is old. SpeechBrain no longer tries using the torchaudio global backend mechanism in recipes, so if you encounter issues, update torchaudio to >=2.1.0."
        )
        current_system = platform.system()
        if current_system == "Windows":
            logger.warning(
                'Switched audio backend to "soundfile" because you are running Windows and you are running an old torchaudio version.'
            )
            torchaudio.set_audio_backend("soundfile")


def validate_backend(backend):
    """
    Validates the specified audio backend.

    Parameters
    ----------
    backend : str or None
        The name of the backend to validate. Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Raises
    ------
    ValueError
        If the `backend` is not one of the allowed values.
    """
    allowed_backends = [None, "ffmpeg", "sox", "soundfile"]
    if backend not in allowed_backends:
        # For torchaudio 2.9+, list_audio_backends doesn't exist
        if hasattr(torchaudio, 'list_audio_backends'):
            available = torchaudio.list_audio_backends()
        else:
            available = ["ffmpeg (default in torchaudio 2.9+)"]
        raise ValueError(
            f"backend must be one of {allowed_backends}. "
            f"Available backends on your system: {available}"
        )
'''


def find_speechbrain_package() -> Path:
    """Find the speechbrain package installation path."""
    # Check common site-packages locations
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        sb_path = Path(site_dir) / "speechbrain"
        if sb_path.exists():
            return sb_path

    # Also check the current venv
    venv_path = Path(sys.prefix) / "lib"
    for py_dir in venv_path.glob("python*"):
        sb_path = py_dir / "site-packages" / "speechbrain"
        if sb_path.exists():
            return sb_path

    return None


def check_needs_patch(sb_path: Path) -> bool:
    """Check if the speechbrain package needs patching."""
    backend_path = sb_path / "utils" / "torch_audio_backend.py"
    if not backend_path.exists():
        return False

    content = backend_path.read_text()
    return "list_audio_backends()" in content and "PATCHED" not in content


def apply_patch(sb_path: Path) -> bool:
    """Apply the torchaudio 2.x compatibility patch."""
    backend_path = sb_path / "utils" / "torch_audio_backend.py"

    # Backup original
    backup_path = sb_path / "utils" / "torch_audio_backend.py.bak"
    if not backup_path.exists():
        backup_path.write_text(backend_path.read_text())
        print(f"Created backup: {backup_path}")

    # Write patched version
    backend_path.write_text(PATCHED_BACKEND_PY)
    print(f"Patched: {backend_path}")
    return True


def main():
    print("SpeechBrain torchaudio 2.9+ Compatibility Patch")
    print("=" * 60)

    # Find speechbrain package
    sb_path = find_speechbrain_package()
    if sb_path is None:
        print("Error: Could not find SpeechBrain package.")
        print("Install with: pip install speechbrain")
        sys.exit(1)

    print(f"Found speechbrain package: {sb_path}")

    # Check if patch is needed
    if not check_needs_patch(sb_path):
        print("Package already patched or does not need patching.")
        sys.exit(0)

    # Apply patch
    print("\nApplying torchaudio 2.9+ compatibility patch...")
    if apply_patch(sb_path):
        print("\nPatch applied successfully!")
        print("\nIMPORTANT: You also need to install torchcodec:")
        print("  pip install torchcodec")
        print("\nVerify by running:")
        print("  python -c 'from speechbrain.inference.separation import SepformerSeparation; print(\"Success!\")'")
    else:
        print("Failed to apply patch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
