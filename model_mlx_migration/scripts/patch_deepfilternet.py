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
Patch DeepFilterNet for torchaudio 2.x compatibility.

DeepFilterNet was built for torchaudio 1.x which had torchaudio.backend.common.AudioMetaData.
In torchaudio 2.x, this class was removed. This script patches df/io.py to work with
torchaudio 2.x by providing a compatibility shim.

Usage:
    python scripts/patch_deepfilternet.py

Requirements:
    - DeepFilterNet installed: pip install deepfilternet
    - torchaudio 2.x installed
"""

import site
import sys
from pathlib import Path

PATCHED_IO_PY = '''import os
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from torch import Tensor

from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root


# Compatibility shim for torchaudio 2.x which removed AudioMetaData
class AudioMetaData(NamedTuple):
    """Compatibility replacement for torchaudio.backend.common.AudioMetaData."""
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _get_audio_info(file: str, **kwargs) -> AudioMetaData:
    """Get audio metadata by loading the file (torchaudio 2.x compatible)."""
    audio, sr = ta.load(file, **kwargs)
    return AudioMetaData(
        sample_rate=sr,
        num_frames=audio.shape[-1],
        num_channels=audio.shape[0] if audio.ndim > 1 else 1,
    )


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using torchaudio.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Passed to torchaudio.load(). Depends on the backend. The resample method
            may be set via `method` which is passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")

    # Load audio
    audio, orig_sr = ta.load(file, **kwargs)

    # Create info after loading
    info = AudioMetaData(
        sample_rate=orig_sr,
        num_frames=audio.shape[-1],
        num_channels=audio.shape[0] if audio.ndim > 1 else 1,
    )

    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if dtype == torch.int16 and audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)
    ta.save(outpath, audio, sr)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
'''


def find_df_package() -> Path:
    """Find the df package installation path."""
    # Check common site-packages locations
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        df_path = Path(site_dir) / "df"
        if df_path.exists():
            return df_path

    # Also check the current venv
    venv_path = Path(sys.prefix) / "lib"
    for py_dir in venv_path.glob("python*"):
        df_path = py_dir / "site-packages" / "df"
        if df_path.exists():
            return df_path

    return None


def check_needs_patch(df_path: Path) -> bool:
    """Check if the df package needs patching."""
    io_path = df_path / "io.py"
    if not io_path.exists():
        return False

    content = io_path.read_text()
    return "torchaudio.backend.common" in content


def apply_patch(df_path: Path) -> bool:
    """Apply the torchaudio 2.x compatibility patch."""
    io_path = df_path / "io.py"

    # Backup original
    backup_path = df_path / "io.py.bak"
    if not backup_path.exists():
        backup_path.write_text(io_path.read_text())
        print(f"Created backup: {backup_path}")

    # Write patched version
    io_path.write_text(PATCHED_IO_PY)
    print(f"Patched: {io_path}")
    return True


def main():
    print("DeepFilterNet torchaudio 2.x Compatibility Patch")
    print("=" * 60)

    # Find df package
    df_path = find_df_package()
    if df_path is None:
        print("Error: Could not find DeepFilterNet (df) package.")
        print("Install with: pip install deepfilternet")
        sys.exit(1)

    print(f"Found df package: {df_path}")

    # Check if patch is needed
    if not check_needs_patch(df_path):
        print("Package already patched or does not need patching.")
        sys.exit(0)

    # Apply patch
    print("\nApplying torchaudio 2.x compatibility patch...")
    if apply_patch(df_path):
        print("\nPatch applied successfully!")
        print("\nVerify by running:")
        print("  python -c 'from df import enhance, init_df; print(\"Success!\")'")
    else:
        print("Failed to apply patch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
