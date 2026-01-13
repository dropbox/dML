# Voice - Streaming TTS for Claude Code

| Director | Status |
|:--------:|:------:|
| ML | ACTIVE |

**100% Local | 100% C++ | Metal GPU Accelerated**

Real-time text-to-speech with translation for narrating AI coding assistant progress.

## Features

- **14 Languages**: en, ja, zh, es, fr, hi, it, pt, ko, yi, zh-sichuan, ar, tr, fa
- **Low Latency**: P50 48-107ms on M4 Max with MPS acceleration
- **Multi-Stream**: FairTTSQueue with priority interrupts and bounded queue depth
- **Translation**: NLLB-200 (600M/3.3B) for EN→target language translation
- **TTS Engines**: Kokoro (primary, MOS 4.3+), CosyVoice2 (dialects), MMS-TTS (fallback)
- **Daemon Mode**: Persistent server keeps models loaded for instant response

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/dropbox/dML/voice.git
cd voice

# Build (Release, MPS)
cmake -S stream-tts-cpp -B stream-tts-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build stream-tts-cpp/build -j

# Speak English
./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en.yaml \
  --speak "Hello world"

# Save to WAV file
./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en.yaml \
  --save-audio /tmp/hello.wav --speak "Hello world"
```

## Daemon Mode

Keep models loaded for instant response:

```bash
# Start daemon
./stream-tts-cpp/build/stream-tts-cpp --daemon stream-tts-cpp/config/default.yaml &

# Send speech requests
./stream-tts-cpp/build/stream-tts-cpp --speak "Hello from daemon"
./stream-tts-cpp/build/stream-tts-cpp --speak "Urgent!" --priority 10  # Interrupts lower priority

# Control daemon
./stream-tts-cpp/build/stream-tts-cpp --status      # Get JSON status
./stream-tts-cpp/build/stream-tts-cpp --interrupt   # Stop current speech
./stream-tts-cpp/build/stream-tts-cpp --stop        # Shutdown daemon
```

## Translation

Translate English to target language before TTS:

```bash
# English to Japanese
./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en2ja.yaml \
  --speak "Build succeeded"

# English to Chinese
./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en2zh.yaml \
  --speak "All tests passed"

# English to Sichuanese dialect
./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en2zh-sichuan.yaml \
  --speak "Hello grandma"
```

## Language Support

| Language | Code | TTS Engine | Translation |
|----------|------|------------|-------------|
| English | en | Kokoro `af_heart` | - |
| Japanese | ja | Kokoro `jf_alpha` | EN→JA |
| Chinese | zh | Kokoro `zm_yunjian` | EN→ZH |
| Spanish | es | Kokoro `ef_dora` | EN→ES |
| French | fr | Kokoro `ff_siwis` | EN→FR |
| Hindi | hi | Kokoro `hf_alpha` | EN→HI |
| Italian | it | Kokoro `if_sara` | EN→IT |
| Portuguese | pt | Kokoro `pf_dora` | EN→PT |
| Korean | ko | Kokoro/CosyVoice2 | EN→KO |
| Yiddish | yi | Kokoro `af_heart` | EN→YI |
| Sichuanese | zh-sichuan | Kokoro/CosyVoice2 | EN→ZH-SICHUAN |
| Arabic | ar | MMS-TTS | EN→AR |
| Turkish | tr | MMS-TTS | EN→TR |
| Persian | fa | MMS-TTS | EN→FA |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Voice Daemon                           │
├─────────────────────────────────────────────────────────────┤
│  Unix Socket ──> Command Handler ──> Priority Queue         │
│                                            │                │
│                                            v                │
│  ┌────────────────────────────────────────────────────┐    │
│  │                  Worker Thread                      │    │
│  │  1. Translate (NLLB-200, cached)                   │    │
│  │  2. Phonemize (espeak-ng / lexicon)                │    │
│  │  3. Synthesize (Kokoro TorchScript/MPS)            │    │
│  │  4. Play (miniaudio, interruptible)                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Models (kept loaded):                                      │
│  • Kokoro TTS ~1.3GB (MPS)                                 │
│  • NLLB-200 ~2GB (MPS, optional for translation)           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Configs are in `stream-tts-cpp/config/`:

| Config | Use Case |
|--------|----------|
| `kokoro-mps-en.yaml` | English TTS only |
| `kokoro-mps-ja.yaml` | Japanese TTS only |
| `kokoro-mps-en2ja.yaml` | English→Japanese translation + TTS |
| `kokoro-mps-en2zh.yaml` | English→Chinese translation + TTS |
| `default.yaml` | Full pipeline with translation |
| `cosyvoice2-zh-sichuan.yaml` | Sichuanese dialect via CosyVoice2 |

## Technology Stack

- **libtorch**: PyTorch C++ API for Kokoro + NLLB (MPS acceleration)
- **SentencePiece + espeak-ng**: Tokenization and G2P
- **miniaudio**: Cross-platform audio playback (24kHz)
- **CosyVoice2**: Optional dialect support (Python server)

## Testing

```bash
# Quick smoke tests (<10s)
make test-smoke

# Unit tests (<60s)
make test-unit

# Integration tests
make test-integration
```

**Test Coverage**: 1276 tests (smoke, unit, integration, quality, stress).

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- CMake 3.20+
- Python 3.11+ (for test suite)
- ~5GB disk space for models

## Model Setup

Models are stored in `models/` directory:

```
models/
├── kokoro/           # Kokoro TTS (~1.3GB)
│   ├── kokoro_mps.pt
│   └── voice_*.pt
├── nllb/             # NLLB-200 translation (~2GB)
│   ├── nllb-encoder-mps.pt
│   └── nllb-decoder-mps.pt
└── whisper/          # Whisper STT (optional, for verification)
```

## Claude Code Integration

Voice can narrate Claude Code worker progress:

```bash
# Run worker with TTS enabled
./run_worker.sh --tts --language ja "continue"

# Listen to specific log file
./stream-tts-cpp/build/stream-tts-cpp --daemon stream-tts-cpp/config/default.yaml &
tail -f worker_logs/worker_iter_*.jsonl | \
  ./stream-tts-cpp/build/stream-tts-cpp --daemon-pipe
```

## License

Copyright 2025 Andrew Yates. All rights reserved.
