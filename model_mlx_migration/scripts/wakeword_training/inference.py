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
Wake Word Inference Module

Provides a simple interface for running wake word detection using our custom-trained
CNN model. Supports both MLX and ONNX backends.

Features:
- MLX backend (Apple Silicon optimized)
- ONNX backend (cross-platform)
- Optional Silero VAD preprocessing to filter noise/silence (fixes false positives)

Usage:
    from scripts.wakeword_training.inference import WakeWordDetector

    # Basic usage (without VAD - may have false positives on noise/silence)
    detector = WakeWordDetector.from_mlx("models/wakeword/hey_agent/hey_agent_mlx.safetensors")

    # Recommended: With VAD preprocessing (filters noise/silence)
    detector = WakeWordDetector.from_mlx(
        "models/wakeword/hey_agent/hey_agent_mlx.safetensors",
        use_vad=True
    )

    # Using ONNX model with VAD
    detector = WakeWordDetector.from_onnx(
        "models/wakeword/hey_agent/hey_agent_cnn.onnx",
        use_vad=True
    )

    # Detect wake word
    audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    probability = detector.detect(audio)
    if probability > 0.5:
        print("Wake word detected!")
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD (MIT licensed).

    Filters noise and silence before wake word detection to prevent false positives.
    The wake word model was trained only on speech samples, so it may incorrectly
    classify noise/silence as wake words. VAD preprocessing solves this issue.

    Uses torch.hub to load the model (works on Python 3.14+ where silero-vad pip
    package is not available due to onnxruntime incompatibility).
    """

    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.5):
        """
        Initialize Silero VAD.

        Args:
            sampling_rate: Audio sample rate (8000 or 16000)
            threshold: Speech detection threshold (0.0 to 1.0)
        """
        if sampling_rate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz sample rates")

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self._model = None
        self._get_speech_timestamps = None

    def _load_model(self):
        """Load the VAD model using torch.hub."""
        if self._model is None:
            try:
                import torch
                model, utils = torch.hub.load(
                    'snakers4/silero-vad',
                    'silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                self._model = model
                # utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
                self._get_speech_timestamps = utils[0]
            except Exception as e:
                raise ImportError(
                    f"Failed to load Silero VAD via torch.hub: {e}. "
                    "Ensure PyTorch is installed."
                )

    @property
    def model(self):
        """Lazy-load the VAD model."""
        if self._model is None:
            self._load_model()
        return self._model

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio contains speech.

        Args:
            audio: Audio array at self.sampling_rate Hz

        Returns:
            True if speech is detected, False otherwise
        """
        if self._model is None:
            self._load_model()

        audio = audio.astype(np.float32)
        timestamps = self._get_speech_timestamps(
            audio,
            self._model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
        )
        return len(timestamps) > 0

    def get_speech_segments(
        self, audio: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Get speech segments from audio.

        Args:
            audio: Audio array at self.sampling_rate Hz

        Returns:
            List of (start_sample, end_sample) tuples
        """
        if self._model is None:
            self._load_model()

        audio = audio.astype(np.float32)
        timestamps = self._get_speech_timestamps(
            audio,
            self._model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
        )
        return [(t["start"], t["end"]) for t in timestamps]

    def extract_speech(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speech from audio, concatenating all speech segments.

        Args:
            audio: Audio array at self.sampling_rate Hz

        Returns:
            Concatenated speech audio, or None if no speech detected
        """
        segments = self.get_speech_segments(audio)
        if not segments:
            return None

        speech_parts = [audio[start:end] for start, end in segments]
        return np.concatenate(speech_parts)


class WakeWordDetector:
    """
    Wake word detector using custom-trained CNN model.

    Supports both MLX (Apple Silicon optimized) and ONNX (cross-platform) backends.

    Optionally uses Silero VAD for preprocessing to filter noise/silence and
    prevent false positives. This is RECOMMENDED for production use.
    """

    def __init__(
        self,
        model=None,
        sample_rate: int = 16000,
        n_mels: int = 32,
        n_fft: int = 512,
        hop_length: int = 160,
        max_frames: int = 76,
        backend: str = "mlx",
        use_vad: bool = False,
        vad_threshold: float = 0.5,
    ):
        """
        Initialize the wake word detector.

        Args:
            model: The underlying model (MLX nn.Module or ONNX session)
            sample_rate: Expected audio sample rate (default: 16000)
            n_mels: Number of mel bands (default: 32)
            n_fft: FFT size (default: 512)
            hop_length: Hop length for STFT (default: 160)
            max_frames: Number of time frames to use (default: 76)
            backend: Either "mlx" or "onnx"
            use_vad: If True, use Silero VAD to filter noise/silence (recommended)
            vad_threshold: VAD speech detection threshold (0.0 to 1.0)
        """
        self.model = model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.backend = backend
        self.use_vad = use_vad
        self.vad = SileroVAD(sampling_rate=sample_rate, threshold=vad_threshold) if use_vad else None

    @classmethod
    def from_mlx(cls, model_path: Union[str, Path], **kwargs) -> "WakeWordDetector":
        """
        Load detector from MLX safetensors file.

        Args:
            model_path: Path to .safetensors file

        Returns:
            WakeWordDetector instance
        """
        import mlx.core as mx

        from scripts.wakeword_training.train_hey_agent import WakeWordCNN

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create and load model
        model = WakeWordCNN()
        flat_weights = dict(mx.load(str(model_path)))

        def unflatten_to_nested(flat):
            result = {}
            for key, value in flat.items():
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        model.update(unflatten_to_nested(flat_weights))
        model.eval()  # Set to eval mode for inference
        mx.eval(model.parameters())

        return cls(model=model, backend="mlx", **kwargs)

    @classmethod
    def from_onnx(cls, model_path: Union[str, Path], **kwargs) -> "WakeWordDetector":
        """
        Load detector from ONNX file.

        Args:
            model_path: Path to .onnx file

        Returns:
            WakeWordDetector instance
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required for ONNX backend. Install with: pip install onnxruntime")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        session = ort.InferenceSession(str(model_path))
        return cls(model=session, backend="onnx", **kwargs)

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: Audio array at self.sample_rate Hz

        Returns:
            Mel spectrogram [frames, n_mels]
        """
        try:
            import librosa
            # Use librosa for high-quality mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=20,
                fmax=8000,
            )
            # Convert to log scale (dB)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize to [-1, 1] range
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-10) * 2 - 1
            return mel_spec.T  # Shape: [time, n_mels]
        except ImportError:
            # Fallback to scipy
            from scipy import signal
            f, t, stft_result = signal.stft(audio, self.sample_rate, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
            power = np.abs(stft_result) ** 2

            # Create simple mel filterbank
            mel_freqs = 700 * (10 ** (np.linspace(0, 1, self.n_mels + 2) * 2.595) - 1)
            mel_freqs = np.clip(mel_freqs, 0, self.sample_rate / 2)

            mel_filterbank = np.zeros((self.n_mels, len(f)))
            for i in range(self.n_mels):
                low = mel_freqs[i]
                center = mel_freqs[i + 1]
                high = mel_freqs[i + 2]

                for j, freq in enumerate(f):
                    if low <= freq < center:
                        mel_filterbank[i, j] = (freq - low) / (center - low + 1e-10)
                    elif center <= freq < high:
                        mel_filterbank[i, j] = (high - freq) / (high - center + 1e-10)

            mel_spec = np.dot(mel_filterbank, power)
            mel_spec = np.log(mel_spec + 1e-10)
            return mel_spec.T

    def detect(self, audio: np.ndarray) -> float:
        """
        Detect wake word in audio.

        If VAD is enabled, first checks if audio contains speech. If no speech
        is detected, returns 0.0 immediately (prevents false positives on noise/silence).

        Args:
            audio: Audio array at self.sample_rate Hz

        Returns:
            Wake word probability (0.0 to 1.0)
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # VAD preprocessing: filter out noise/silence
        if self.use_vad and self.vad is not None:
            if not self.vad.has_speech(audio):
                return 0.0  # No speech detected = no wake word

        # Compute mel spectrogram
        mel = self.compute_mel_spectrogram(audio)

        # Pad or truncate to max_frames
        if mel.shape[0] < self.max_frames:
            mel = np.pad(mel, ((0, self.max_frames - mel.shape[0]), (0, 0)))
        else:
            mel = mel[:self.max_frames]

        # Add batch dimension
        mel = mel[np.newaxis, :, :].astype(np.float32)  # [1, frames, mels]

        # Run inference
        if self.backend == "mlx":
            import mlx.core as mx
            mel_mx = mx.array(mel)
            output = self.model(mel_mx)
            mx.eval(output)
            return float(output[0, 0])
        else:  # ONNX
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: mel})
            return float(output[0][0, 0])

    def detect_stream(
        self, audio_chunks: list, threshold: float = 0.5, return_all: bool = False
    ) -> list:
        """
        Detect wake word in a stream of audio chunks.

        If VAD is enabled, chunks without speech are automatically skipped.

        Args:
            audio_chunks: List of audio arrays
            threshold: Detection threshold (default: 0.5)
            return_all: If True, return probabilities for all chunks (including 0.0)

        Returns:
            List of (chunk_index, probability) tuples for detections
            If return_all=True, returns all chunks; otherwise only above threshold
        """
        results = []
        for i, chunk in enumerate(audio_chunks):
            prob = self.detect(chunk)
            if return_all or prob > threshold:
                results.append((i, prob))
        return results


def main():
    """Demo of wake word detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Wake word detection demo")
    parser.add_argument("--model", type=str, default="models/wakeword/hey_agent/hey_agent_mlx.safetensors",
                       help="Path to model file (.safetensors or .onnx)")
    parser.add_argument("--audio", type=str, help="Path to audio file (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--vad", action="store_true",
                       help="Enable VAD preprocessing (recommended to prevent false positives)")
    parser.add_argument("--compare-vad", action="store_true",
                       help="Compare detection with and without VAD")
    args = parser.parse_args()

    model_path = Path(args.model)

    # Load detector based on file extension
    if model_path.suffix == ".onnx":
        print(f"Loading ONNX model from {model_path}")
        detector = WakeWordDetector.from_onnx(model_path, use_vad=args.vad)
    else:
        print(f"Loading MLX model from {model_path}")
        detector = WakeWordDetector.from_mlx(model_path, use_vad=args.vad)

    if args.vad:
        print("VAD preprocessing: ENABLED")
    else:
        print("VAD preprocessing: DISABLED (use --vad to enable)")

    if args.audio:
        # Load and process audio file
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if sr != detector.sample_rate:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * detector.sample_rate / sr))
            audio = audio.astype(np.float32)

        prob = detector.detect(audio)
        print(f"Detection probability: {prob:.4f}")
        if prob > args.threshold:
            print("Wake word DETECTED!")
        else:
            print("No wake word detected.")
    elif args.compare_vad:
        # Compare VAD vs no VAD
        print("\n=== VAD Comparison Demo ===")
        print("This demonstrates how VAD prevents false positives on noise/silence.\n")

        # Create detector without VAD for comparison
        if model_path.suffix == ".onnx":
            detector_no_vad = WakeWordDetector.from_onnx(model_path, use_vad=False)
        else:
            detector_no_vad = WakeWordDetector.from_mlx(model_path, use_vad=False)

        detector_with_vad = WakeWordDetector.from_mlx(model_path, use_vad=True) \
            if model_path.suffix != ".onnx" else WakeWordDetector.from_onnx(model_path, use_vad=True)

        test_cases = [
            ("Silence", np.zeros(16000, dtype=np.float32)),
            ("Random noise", (np.random.randn(16000) * 0.1).astype(np.float32)),
            ("White noise", (np.random.randn(16000) * 0.5).astype(np.float32)),
        ]

        print(f"{'Test Case':<20} {'No VAD':<15} {'With VAD':<15}")
        print("-" * 50)

        for name, audio in test_cases:
            prob_no_vad = detector_no_vad.detect(audio)
            prob_with_vad = detector_with_vad.detect(audio)
            print(f"{name:<20} {prob_no_vad:.4f}         {prob_with_vad:.4f}")

        print("\n(With VAD, noise/silence correctly returns 0.0)")
    else:
        # Demo with random audio
        print("\nNo audio file provided. Testing with random audio...")
        audio = np.random.randn(16000).astype(np.float32)
        prob = detector.detect(audio)
        print(f"Random audio probability: {prob:.4f}")

        if not args.vad:
            print("\nNote: Without VAD, the model may return high probabilities for noise.")
            print("Run with --compare-vad to see the difference, or use --vad for production.")


if __name__ == "__main__":
    main()
