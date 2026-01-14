# dML - Machine Learning Models

![Status](https://img.shields.io/badge/status-preview-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

ML and NN training and inference optimized for Apple Silicon via MLX. Currently focused on voice.

> All d* projects are entirely AI generated.

## Structure

| Directory | Description | Status |
|-----------|-------------|--------|
| **model_mlx_migration** | MLX models for Apple Silicon | Usable |
| **voice** | Streaming voice I/O, 14 languages | Preview |
| **metal_mps_parallel** | Metal/MPS GPU threading | Preview |

> **Note:** `model_mlx_migration` will be split into separate TTS, STT, and tooling repos.

## Models

### Speech-to-Text (STT)

| Model | Description |
|-------|-------------|
| **ZipFormer** | Streaming ASR encoder (k2/icefall), 2.85% WER |
| **Whisper large-v3-turbo** | Non-streaming fallback, 1.8% WER |
| **Silero VAD** | Voice activity detection |

### Text-to-Speech (TTS)

| Model | Description |
|-------|-------------|
| **CosyVoice3** | Alibaba TTS with voice cloning, DiT flow-matching |
| **Kokoro** | Lightweight TTS, 38x realtime |

### Rich Audio

9 classification heads: emotion (92% acc), paralinguistics (97% acc), language ID (99% acc), pitch, singing detection, punctuation.

## License

Apache 2.0

