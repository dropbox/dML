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
Audio feature extraction for Zipformer ASR.

Computes Kaldi-compatible 80-dimensional filterbank (fbank) features.
Uses torchaudio for compatibility with icefall checkpoints.
"""

from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import numpy as np

try:
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class FbankConfig:
    """Configuration for filterbank feature extraction.

    Default values match icefall/k2-fsa training for zipformer checkpoints.
    """
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length: float = 25.0  # ms
    frame_shift: float = 10.0  # ms
    dither: float = 0.0
    snip_edges: bool = True  # Must match icefall training (True is Kaldi default)
    energy_floor: float = 0.0  # Match icefall training
    # Kaldi defaults
    preemphasis_coefficient: float = 0.97
    window_type: str = "povey"
    use_energy: bool = False
    use_log_fbank: bool = True
    htk_compat: bool = True  # Required for icefall compatibility


class FbankExtractor:
    """
    Extract Kaldi-compatible filterbank features.

    Uses torchaudio for exact compatibility with icefall training.
    Output is converted to MLX arrays for inference.
    """

    def __init__(self, config: FbankConfig | None = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature configuration. Defaults to icefall standard.
        """
        if not HAS_TORCHAUDIO:
            raise ImportError(
                "torchaudio is required for feature extraction. "
                "Install with: pip install torchaudio",
            )
        self.config = config or FbankConfig()

    def extract(
        self,
        waveform: Union[np.ndarray, mx.array, "torch.Tensor"],
        sample_rate: int | None = None,
    ) -> mx.array:
        """
        Extract filterbank features from audio waveform.

        Args:
            waveform: Audio samples, shape (num_samples,) or (1, num_samples).
            sample_rate: Sample rate. If None, uses config.sample_rate.

        Returns:
            Filterbank features of shape (num_frames, num_mel_bins).
        """
        # Convert to torch tensor
        if isinstance(waveform, mx.array):
            waveform = np.array(waveform)
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        # Ensure float32
        waveform = waveform.float()

        # Ensure 2D: (1, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed
        sr = sample_rate or self.config.sample_rate
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate,
            )
            waveform = resampler(waveform)

        # Extract filterbank features using Kaldi-compatible function
        features = kaldi.fbank(
            waveform,
            sample_frequency=self.config.sample_rate,
            num_mel_bins=self.config.num_mel_bins,
            frame_length=self.config.frame_length,
            frame_shift=self.config.frame_shift,
            dither=self.config.dither,
            snip_edges=self.config.snip_edges,
            energy_floor=self.config.energy_floor,
            preemphasis_coefficient=self.config.preemphasis_coefficient,
            window_type=self.config.window_type,
            use_energy=self.config.use_energy,
            use_log_fbank=self.config.use_log_fbank,
            htk_compat=self.config.htk_compat,
        )

        # Convert to MLX
        return mx.array(features.numpy())

    def extract_batch(
        self,
        waveforms: list,
        sample_rates: list | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Extract features from a batch of waveforms.

        Args:
            waveforms: List of waveform arrays.
            sample_rates: Optional list of sample rates.

        Returns:
            Tuple of (features, lengths) where:
                features: Padded features (batch, max_frames, num_mel_bins)
                lengths: Frame lengths (batch,)
        """
        if sample_rates is None:
            sample_rates = [None] * len(waveforms)

        # Extract features for each waveform
        feature_list = []
        for wav, sr in zip(waveforms, sample_rates, strict=False):
            feat = self.extract(wav, sr)
            feature_list.append(feat)

        # Get lengths
        lengths = mx.array([f.shape[0] for f in feature_list], dtype=mx.int32)

        # Pad to same length
        max_len = int(max(lengths).item())
        num_mel = self.config.num_mel_bins

        padded = mx.zeros((len(feature_list), max_len, num_mel))
        for i, feat in enumerate(feature_list):
            T = feat.shape[0]
            # MLX doesn't support in-place assignment, use mx.where or concatenate
            # Build padded array properly
            padded = padded.at[i, :T, :].add(feat)

        return padded, lengths


def load_audio(
    path: str,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    # Try soundfile first (more reliable)
    if HAS_SOUNDFILE:
        waveform, sr = sf.read(path)

        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            if not HAS_TORCHAUDIO:
                raise ImportError(
                    f"Audio is {sr}Hz but target is {target_sr}Hz. "
                    "torchaudio required for resampling.",
                )
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform).squeeze(0).numpy()
            sr = target_sr

        return waveform.astype(np.float32), sr

    # Fall back to torchaudio
    if not HAS_TORCHAUDIO:
        raise ImportError("soundfile or torchaudio required for audio loading")

    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform.squeeze(0).numpy(), sr
