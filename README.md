# dML - Machine Learning Models

Machine learning inference engines optimized for Apple Silicon via MLX.

## Thesis

On-device AI should be fast, private, and free from cloud dependencies. These models run locally on Apple Silicon, enabling real-time speech recognition, text-to-speech, and translation without network latency or data leaving your machine.

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
