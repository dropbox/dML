# dML - Machine Learning Models

![Status](https://img.shields.io/badge/status-preview-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Machine learning inference engines optimized for Apple Silicon via MLX.

## Thesis

**Your AI should live on your machine.** Cloud inference means latency, cost, and your data leaving your control. These models run locally on Apple Silicon at speeds that make real-time voice interaction feel instant—Whisper transcribing as you speak, Kokoro responding before you finish listening, translation happening in the pause between sentences. Privacy and performance, not a tradeoff.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| **model_mlx_migration** | MLX model collection for Apple Silicon | Planned |
| **voice** | Streaming voice I/O. 14 languages, P50 48-107ms latency. | Planned |
| **metal_mps_parallel** | Metal/MPS GPU threading. | Planned |

### Models in model_mlx_migration

| Model | Type | Description |
|-------|------|-------------|
| **whisper_mlx** | STT | Whisper speech-to-text (C++ MLX) |
| **kokoro** | TTS | Kokoro text-to-speech (38x realtime) |
| **llama** | LLM | LLaMA language model |
| **nllb** | Translation | NLLB-200 neural translation (200 languages) |
| **wake_word** | Audio | Wake word detection |

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Release History

See [RELEASES.md](RELEASES.md) for version history.
