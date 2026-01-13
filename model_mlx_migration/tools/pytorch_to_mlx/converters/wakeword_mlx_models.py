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
MLX Wake Word Models

Native MLX implementations of OpenWakeWord models converted from ONNX.

Architecture:
1. MelSpectrogram: Audio -> Mel features using Conv-based STFT
2. Embedding: CNN extracting 96-dim embeddings from mel features
3. Classifier: Dense network classifying 16 embeddings into wake word probability

Usage:
    from wakeword_mlx_models import WakeWordPipeline

    pipeline = WakeWordPipeline.from_onnx("~/voice/models/wakeword")
    probability = pipeline(audio_array)
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:
    import onnx
    import onnx.numpy_helper

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class MelSpectrogram(nn.Module):
    """
    MLX Mel Spectrogram extractor.

    Performs STFT via 1D convolution then applies mel filterbank.

    Architecture (from ONNX):
    - Conv STFT: real and imaginary parts (257 freq bins from n_fft=512)
    - Power spectrum: real^2 + imag^2
    - Mel filterbank: 257 -> 32 mel bins
    - Log compression with normalization
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_freq = n_fft // 2 + 1  # 257 for n_fft=512

        # STFT conv weights (initialized, loaded from ONNX)
        # Shape: [n_freq, 1, n_fft] -> MLX Conv1d: [out_channels, in_channels, kernel]
        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=self.n_freq,
            kernel_size=n_fft,
            stride=hop_length,
            padding=0,
            bias=False,
        )
        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=self.n_freq,
            kernel_size=n_fft,
            stride=hop_length,
            padding=0,
            bias=False,
        )

        # Mel filterbank: [n_freq, n_mels]
        self.mel_filterbank = mx.zeros((self.n_freq, n_mels))

        # Normalization constants (from ONNX Clip ops)
        self.clip_min = 1e-10
        self.clip_max = 1e10

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Convert audio to mel spectrogram.

        Args:
            audio: Input audio [samples] or [batch, samples]

        Returns:
            Mel spectrogram [batch, 1, time, n_mels]
        """
        # Ensure 2D input [batch, samples]
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        batch_size = audio.shape[0]

        # MLX Conv1d expects NLC format: [batch, length, channels]
        # Our input is [batch, samples], add channel dim at end
        audio_nlc = audio.reshape(batch_size, -1, 1)  # [batch, samples, 1]

        # STFT via convolution
        real = self.conv_real(audio_nlc)  # [batch, time, n_freq]
        imag = self.conv_imag(audio_nlc)  # [batch, time, n_freq]

        # Power spectrum
        power = real**2 + imag**2  # [batch, time, n_freq]

        # Apply mel filterbank: [batch, time, n_mels]
        mel = mx.matmul(power, self.mel_filterbank)

        # Log compression with clipping
        mel = mx.clip(mel, self.clip_min, self.clip_max)
        log_mel = mx.log(mel)

        # Convert to dB scale: 10 * log10(x) = 10 * ln(x) / ln(10)
        # From ONNX: mul 10.0, div ln(10)
        log_mel = log_mel * 10.0
        log_mel = log_mel / 2.3025851249694824  # ln(10)

        # Dynamic range compression
        # From ONNX: ReduceMax, then Sub 80 (gives lower bound = max - 80)
        max_val = mx.max(log_mel, axis=(1, 2), keepdims=True)
        lower_bound = max_val - 80.0
        log_mel = mx.clip(log_mel, lower_bound, None)

        # Reshape to [batch, 1, time, n_mels] to match ONNX output
        return log_mel.reshape(batch_size, 1, log_mel.shape[1], log_mel.shape[2])



class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        leaky_alpha: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",  # type: ignore[arg-type]  # MLX supports string padding
            bias=True,
        )
        self.leaky_alpha = leaky_alpha

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        # LeakyReLU with max(x, alpha*x) pattern from ONNX
        return mx.maximum(x, x * self.leaky_alpha)


class EmbeddingModel(nn.Module):
    """
    MLX Wake Word Embedding model.

    CNN that extracts 96-dim embeddings from mel spectrograms.

    Architecture (from ONNX analysis):
    - Input: [batch, 76, 32, 1] (76 time frames, 32 mel bins, 1 channel)
    - 20 Conv2d layers with LeakyReLU (conv0-conv19)
    - 5 MaxPool2d layers with varying kernels
    - Output: [batch, 1, 1, 96]

    Pool configurations from ONNX:
    - pool1: (2,2) stride (2,2)  -> H/2, W/2
    - pool2: (1,2) stride (1,2)  -> H, W/2  (only width reduced)
    - pool3: (2,2) stride (2,2)  -> H/2, W/2
    - pool4: (1,2) stride (1,2)  -> H, W/2  (only width reduced)
    - pool5: (2,2) stride (2,2)  -> H/2, W/2
    """

    def __init__(self):
        super().__init__()

        # Build the CNN architecture matching ONNX exactly
        # Conv layers: 20 total (conv0-conv19)
        # Channels: 1 -> 24 -> 48 -> 72 -> 96
        # For kernel (H, W), same padding = ((H-1)//2, (W-1)//2)

        # Block 1: 3 conv layers, 1 pool (channels: 1 -> 24)
        # ONNX padding: (3,3) uses (0,1,0,1)=width-only, (3,1) uses no padding
        self.conv0 = nn.Conv2d(1, 24, (3, 3), padding=(0, 1), bias=True)
        self.conv1 = nn.Conv2d(24, 24, (1, 3), padding=(0, 1), bias=True)
        self.conv2 = nn.Conv2d(24, 24, (3, 1), padding=(0, 0), bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))  # H/2, W/2

        # Block 2: 4 conv layers, 1 pool (channels: 24 -> 48)
        self.conv3 = nn.Conv2d(24, 48, (1, 3), padding=(0, 1), bias=True)
        self.conv4 = nn.Conv2d(48, 48, (3, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(48, 48, (1, 3), padding=(0, 1), bias=True)
        self.conv6 = nn.Conv2d(48, 48, (3, 1), padding=(0, 0), bias=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=(1, 2))  # H, W/2

        # Block 3: 4 conv layers, 1 pool (channels: 48 -> 72)
        self.conv7 = nn.Conv2d(48, 72, (1, 3), padding=(0, 1), bias=True)
        self.conv8 = nn.Conv2d(72, 72, (3, 1), padding=(0, 0), bias=True)
        self.conv9 = nn.Conv2d(72, 72, (1, 3), padding=(0, 1), bias=True)
        self.conv10 = nn.Conv2d(72, 72, (3, 1), padding=(0, 0), bias=True)
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))  # H/2, W/2

        # Block 4: 4 conv layers, 1 pool (channels: 72 -> 96)
        self.conv11 = nn.Conv2d(72, 96, (1, 3), padding=(0, 1), bias=True)
        self.conv12 = nn.Conv2d(96, 96, (3, 1), padding=(0, 0), bias=True)
        self.conv13 = nn.Conv2d(96, 96, (1, 3), padding=(0, 1), bias=True)
        self.conv14 = nn.Conv2d(96, 96, (3, 1), padding=(0, 0), bias=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2))  # H, W/2

        # Block 5: 5 conv layers, 1 pool (channels: 96 -> 96)
        self.conv15 = nn.Conv2d(96, 96, (1, 3), padding=(0, 1), bias=True)
        self.conv16 = nn.Conv2d(96, 96, (3, 1), padding=(0, 0), bias=True)
        self.conv17 = nn.Conv2d(96, 96, (1, 3), padding=(0, 1), bias=True)
        self.conv18 = nn.Conv2d(96, 96, (3, 1), padding=(0, 0), bias=True)
        self.pool5 = nn.MaxPool2d((2, 2), stride=(2, 2))  # H/2, W/2

        # Final conv
        self.conv19 = nn.Conv2d(96, 96, (3, 1), padding=(0, 0), bias=True)

        self.leaky_alpha = 0.2  # From ONNX LeakyRelu alpha

    def _leaky_relu(self, x: mx.array) -> mx.array:
        """LeakyReLU with max pattern from ONNX."""
        return mx.maximum(x, x * self.leaky_alpha)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Extract embedding from mel spectrogram.

        Args:
            mel: Input [batch, 76, 32, 1] in NHWC format

        Returns:
            Embedding [batch, 1, 1, 96]
        """
        # Input is NHWC, MLX Conv2d expects NHWC by default
        x = mel

        # Block 1: 76x32 -> 38x16
        x = self._leaky_relu(self.conv0(x))
        x = self._leaky_relu(self.conv1(x))
        x = self._leaky_relu(self.conv2(x))
        x = self.pool1(x)

        # Block 2: 38x16 -> 38x8 (pool2 only reduces W)
        x = self._leaky_relu(self.conv3(x))
        x = self._leaky_relu(self.conv4(x))
        x = self._leaky_relu(self.conv5(x))
        x = self._leaky_relu(self.conv6(x))
        x = self.pool2(x)

        # Block 3: 38x8 -> 19x4
        x = self._leaky_relu(self.conv7(x))
        x = self._leaky_relu(self.conv8(x))
        x = self._leaky_relu(self.conv9(x))
        x = self._leaky_relu(self.conv10(x))
        x = self.pool3(x)

        # Block 4: 19x4 -> 19x2 (pool4 only reduces W)
        x = self._leaky_relu(self.conv11(x))
        x = self._leaky_relu(self.conv12(x))
        x = self._leaky_relu(self.conv13(x))
        x = self._leaky_relu(self.conv14(x))
        x = self.pool4(x)

        # Block 5: 19x2 -> 9x1
        x = self._leaky_relu(self.conv15(x))
        x = self._leaky_relu(self.conv16(x))
        x = self._leaky_relu(self.conv17(x))
        x = self._leaky_relu(self.conv18(x))
        x = self.pool5(x)

        # Final conv
        x = self.conv19(x)

        # Global average pool to match ONNX output shape [batch, 1, 1, 96]
        # ONNX uses NCHW internally with different padding, resulting in [batch, 96, 1, 1]
        # after final conv. We use global average pooling to get compatible shape.
        # Shape: [batch, H, W, C] -> [batch, 1, 1, C]
        return mx.mean(x, axis=(1, 2), keepdims=True)



class Classifier(nn.Module):
    """
    MLX Wake Word Classifier.

    Dense network that classifies 16 embeddings into wake word probability.

    Architecture:
    - Input: [batch, 16, 96] (16 embeddings of 96 dims)
    - Flatten to [batch, 1536]
    - Dense(1536, 128) + LayerNorm + ReLU
    - Dense(128, 128) + LayerNorm + ReLU
    - Dense(128, 1) + Sigmoid
    """

    def __init__(self):
        super().__init__()

        # Dense layers
        self.dense1 = nn.Linear(1536, 128)
        self.ln1 = nn.LayerNorm(128)
        self.dense2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dense3 = nn.Linear(128, 1)

    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Classify embeddings.

        Args:
            embeddings: Input [batch, 16, 96]

        Returns:
            Probability [batch, 1]
        """
        # Flatten
        x = embeddings.reshape(embeddings.shape[0], -1)  # [batch, 1536]

        # Block 1
        x = self.dense1(x)
        x = self.ln1(x)
        x = nn.relu(x)

        # Block 2
        x = self.dense2(x)
        x = self.ln2(x)
        x = nn.relu(x)

        # Output
        x = self.dense3(x)
        return mx.sigmoid(x)



class WakeWordPipeline(nn.Module):
    """
    Complete MLX wake word detection pipeline.

    Combines MelSpectrogram, Embedding, and Classifier into a single module.
    """

    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogram()
        self.embedding = EmbeddingModel()
        self.classifier = Classifier()

        # Pipeline constants
        self.mel_frames_required = 76
        self.embeddings_required = 16
        self.sample_rate = 16000
        self.min_samples = 12800  # ~800ms for 76 mel frames

    @staticmethod
    def from_onnx(model_dir: str) -> "WakeWordPipeline":
        """
        Load weights from ONNX models.

        Args:
            model_dir: Directory containing ONNX models

        Returns:
            WakeWordPipeline with loaded weights
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx is required for loading ONNX models")

        model_path = Path(model_dir).expanduser()
        pipeline = WakeWordPipeline()

        # Load mel spectrogram weights
        mel_path = model_path / "melspectrogram.onnx"
        if mel_path.exists():
            pipeline._load_mel_weights(mel_path)

        # Load embedding weights
        emb_path = model_path / "embedding_model.onnx"
        if emb_path.exists():
            pipeline._load_embedding_weights(emb_path)

        # Load classifier weights
        cls_path = model_path / "hey_agent.onnx"
        if cls_path.exists():
            pipeline._load_classifier_weights(cls_path)

        return pipeline

    def _load_mel_weights(self, model_path: Path) -> None:
        """Load mel spectrogram weights from ONNX."""
        model = onnx.load(str(model_path))
        weights = {
            init.name: onnx.numpy_helper.to_array(init)
            for init in model.graph.initializer
        }

        # STFT conv weights: ONNX [out, in, kernel] -> MLX [out, kernel, in]
        if "0.stft.conv_real.weight" in weights:
            w = weights["0.stft.conv_real.weight"]  # [257, 1, 512]
            w = np.transpose(w, (0, 2, 1))  # -> [257, 512, 1]
            self.mel.conv_real.weight = mx.array(w)
        if "0.stft.conv_imag.weight" in weights:
            w = weights["0.stft.conv_imag.weight"]  # [257, 1, 512]
            w = np.transpose(w, (0, 2, 1))  # -> [257, 512, 1]
            self.mel.conv_imag.weight = mx.array(w)

        # Mel filterbank
        if "1.melW" in weights:
            self.mel.mel_filterbank = mx.array(weights["1.melW"])

    def _load_embedding_weights(self, model_path: Path) -> None:
        """Load embedding model weights from ONNX.

        Weight mapping:
        - ONNX weights are in [out, in, H, W] format
        - MLX Conv2d expects [out, H, W, in] format
        - Transpose: (0, 2, 3, 1)

        ONNX layer naming:
        - conv0: model/conv2d/Conv2D_weights_fused_bn
        - conv1-18: model/conv2d_{N}/Conv2D_weights_fused_bn
        - conv19: model/conv2d_19/Conv2D/ReadVariableOp:0 (no bias)
        """
        model = onnx.load(str(model_path))
        weights = {
            init.name: onnx.numpy_helper.to_array(init)
            for init in model.graph.initializer
        }

        def load_conv(layer, weight_name, bias_name=None):
            """Load weights for a single conv layer."""
            if weight_name in weights:
                # ONNX: [out, in, H, W] -> MLX: [out, H, W, in]
                w = weights[weight_name]
                w = np.transpose(w, (0, 2, 3, 1))
                layer.weight = mx.array(w)
            if bias_name and bias_name in weights:
                layer.bias = mx.array(weights[bias_name])

        # conv0: special naming (no number)
        load_conv(
            self.embedding.conv0,
            "model/conv2d/Conv2D_weights_fused_bn",
            "model/conv2d/Conv2D_bias_fused_bn",
        )

        # conv1-18: standard naming pattern
        for i in range(1, 19):
            layer = getattr(self.embedding, f"conv{i}")
            load_conv(
                layer,
                f"model/conv2d_{i}/Conv2D_weights_fused_bn",
                f"model/conv2d_{i}/Conv2D_bias_fused_bn",
            )

        # conv19: different naming, no bias
        load_conv(
            self.embedding.conv19,
            "model/conv2d_19/Conv2D/ReadVariableOp:0",
            None,
        )

    def _load_classifier_weights(self, model_path: Path) -> None:
        """Load classifier weights from ONNX."""
        model = onnx.load(str(model_path))
        weights = {
            init.name: onnx.numpy_helper.to_array(init)
            for init in model.graph.initializer
        }

        # Dense layer weights: ONNX Gemm uses transposed weights
        if "model.1.weight" in weights:
            # ONNX: [out, in], MLX Linear: [out, in]
            self.classifier.dense1.weight = mx.array(weights["model.1.weight"])
        if "model.1.bias" in weights:
            self.classifier.dense1.bias = mx.array(weights["model.1.bias"])

        # LayerNorm weights
        if "model.2.weight" in weights:
            self.classifier.ln1.weight = mx.array(weights["model.2.weight"])
        if "model.2.bias" in weights:
            self.classifier.ln1.bias = mx.array(weights["model.2.bias"])

        if "model.4.weight" in weights:
            self.classifier.dense2.weight = mx.array(weights["model.4.weight"])
        if "model.4.bias" in weights:
            self.classifier.dense2.bias = mx.array(weights["model.4.bias"])

        if "model.5.weight" in weights:
            self.classifier.ln2.weight = mx.array(weights["model.5.weight"])
        if "model.5.bias" in weights:
            self.classifier.ln2.bias = mx.array(weights["model.5.bias"])

        if "model.7.weight" in weights:
            self.classifier.dense3.weight = mx.array(weights["model.7.weight"])
        if "model.7.bias" in weights:
            self.classifier.dense3.bias = mx.array(weights["model.7.bias"])

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Detect wake word in audio.

        Args:
            audio: Audio samples [samples] or [batch, samples]

        Returns:
            Wake word probability [batch, 1]
        """
        # Ensure batch dimension
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        # Pad if too short
        if audio.shape[1] < self.min_samples:
            padding = mx.zeros((audio.shape[0], self.min_samples - audio.shape[1]))
            audio = mx.concatenate([audio, padding], axis=1)

        # Stage 1: Mel spectrogram
        mel = self.mel(audio)  # [batch, 1, time, 32]

        # Get number of mel frames
        total_frames = mel.shape[2]
        if total_frames < self.mel_frames_required:
            raise ValueError(
                f"Not enough mel frames: {total_frames} < {self.mel_frames_required}",
            )

        # Stage 2: Generate embeddings
        num_possible = total_frames - self.mel_frames_required + 1
        if num_possible >= self.embeddings_required:
            step = max(1, (num_possible - 1) // (self.embeddings_required - 1))
            starts = [i * step for i in range(self.embeddings_required)]
        else:
            starts = list(range(num_possible))
            while len(starts) < self.embeddings_required:
                starts.append(starts[-1])

        # Build embeddings list (MLX doesn't have .at[].set() pattern)
        embedding_list = []
        for start in starts:
            # Extract window: [batch, 1, 76, 32]
            mel_window = mel[:, :, start : start + self.mel_frames_required, :]
            # Reshape for embedding: [batch, 76, 32, 1]
            mel_emb = mx.transpose(mel_window, (0, 2, 3, 1))

            # Run embedding
            emb = self.embedding(mel_emb)  # [batch, 1, 1, 96]
            embedding_list.append(emb.reshape(audio.shape[0], 96))

        # Stack embeddings: [batch, 16, 96]
        embeddings = mx.stack(embedding_list, axis=1)

        # Stage 3: Classify
        prob: mx.array = self.classifier(embeddings)

        return prob


def validate_against_onnx(model_dir: str, audio: np.ndarray | None = None) -> dict:
    """
    Validate MLX pipeline against ONNX runtime.

    Args:
        model_dir: Directory containing ONNX models
        audio: Optional test audio (generates random if None)

    Returns:
        Dict with validation results
    """
    import onnxruntime as ort

    model_path = Path(model_dir).expanduser()

    # Generate test audio if not provided
    if audio is None:
        rng = np.random.default_rng()
        audio = rng.standard_normal(16000).astype(np.float32) * 0.01

    # Load ONNX session for mel spectrogram
    providers = ["CPUExecutionProvider"]
    mel_session = ort.InferenceSession(
        str(model_path / "melspectrogram.onnx"), providers=providers,
    )

    # Load MLX pipeline
    mlx_pipeline = WakeWordPipeline.from_onnx(str(model_path))

    # Run ONNX inference
    mel_onnx = mel_session.run(None, {"input": audio.reshape(1, -1)})[0]

    # Run MLX mel
    mel_mlx = mlx_pipeline.mel(mx.array(audio))
    mx.eval(mel_mlx)
    mel_mlx_np = np.array(mel_mlx)

    return {
        "mel_onnx_shape": mel_onnx.shape,
        "mel_mlx_shape": mel_mlx_np.shape,
        "mel_max_diff": float(np.max(np.abs(mel_onnx - mel_mlx_np)))
        if mel_onnx.shape == mel_mlx_np.shape
        else "shape_mismatch",
        "note": "Mel spectrogram validation only. Full pipeline validation in test suite.",
    }

