# Stream TTS Rust - Optimal M4 Max Architecture

**Status**: Phase 1 - Rust Parser Complete ✅

## Architecture Overview

```
Claude Code (--output-format stream-json)
    ↓
stream-tts (Rust coordinator) ← YOU ARE HERE
    ├─→ stdin parser (Rust) ✅ COMPLETE
    ├─→ Translation Worker (C++/Python + Metal)
    ├─→ TTS Worker (C++/Python + Metal)
    └─→ Audio Playback (Rust rodio)
```

## What's Working Now

### ✅ Rust Stdin Parser (COMPLETE)
- **Performance**: < 1ms per message
- **Features**:
  - Parse Claude's stream-json output
  - Extract text from assistant messages
  - Clean markdown, code blocks, URLs, file paths
  - Segment into sentences for streaming
  - Filter system noise and short messages
  - Async channel-based pipeline

### Test Results
```bash
cargo test
# All tests passing:
# - test_clean_text_for_speech ✅
# - test_segment_sentences ✅
# - test_should_speak ✅

# Live test:
echo '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Test message."}]}}' | cargo run
```

## Next Steps

### Phase 1: Python Prototype (Week 1)
Target: 100-150ms latency

1. **Translation Worker** (Python + Metal GPU)
   - Use NLLB-200 via Transformers
   - Enable Metal acceleration: `device_map="mps"`
   - Interface: stdin → text, stdout → translated text
   - Run as subprocess from Rust

2. **TTS Worker** (Python + Metal GPU)
   - Use Coqui XTTS v2 or similar
   - Enable Metal acceleration
   - Interface: stdin → text, stdout → audio file path
   - Run as subprocess from Rust

3. **Audio Playback** (Rust)
   - Use rodio crate (already in Cargo.toml)
   - Stream audio files as they're generated
   - Queue management for continuous playback

### Phase 2: C++ Production (Week 2-3)
Target: 50-70ms latency

- Replace Python workers with C++ implementations
- Direct Metal API via Objective-C++
- Use ONNX Runtime C++ API for inference
- Zero-copy data transfer between components

### Phase 3: GPU Kernel Optimization (Week 4)
Target: 30-50ms latency

- Custom Metal compute shaders
- Fused operations for translation + TTS
- M4 Max specific optimizations (40-core GPU)

## Design Rationale

**Why Rust wrapper + C++ ML cores?**

1. **Rust**: Best for I/O, parsing, coordination, async
   - Fast stdin processing
   - Safe concurrency with channels
   - Easy subprocess management
   - Excellent audio libraries (rodio)

2. **C++**: Best for ML inference
   - Direct access to Metal APIs
   - Mature ML libraries (ONNX Runtime, Core ML)
   - Zero-overhead abstractions
   - Apple Silicon optimizations

3. **Python**: Prototyping only
   - Quick validation of approach
   - Easy model loading with Transformers
   - Replaced in production (Phase 2)

## Current Implementation

### Rust Parser Features

**Input**: Claude stream-json on stdin
```json
{
  "type": "message",
  "message": {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "Your message here."}
    ]
  }
}
```

**Output**: Clean text segments via async channel
```rust
TextSegment {
    text: "Your message here",
    is_speakable: true
}
```

**Text Cleaning**:
- Removes `**bold**`, `_italic_`, `` `code` ``
- Removes code blocks (``` ... ```)
- Replaces URLs with "URL"
- Removes file paths
- Segments on sentence boundaries (. ! ? 。)
- Filters messages < 10 characters

**System Noise Filtering**:
- Skips "Co-Authored-By:" lines
- Skips "🤖 Generated with" lines
- Skips `<system-reminder>` blocks
- Skips malware check reminders

## Building

```bash
# Build
cargo build --release

# Test
cargo test

# Run
echo '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Hello world."}]}}' | cargo run --release
```

## Performance Targets

| Component | Phase 1 (Python) | Phase 2 (C++) | Phase 3 (Optimized) |
|-----------|------------------|---------------|---------------------|
| Parser    | < 1ms ✅         | < 1ms         | < 1ms               |
| Translation | 50-80ms        | 15-25ms       | 8-15ms              |
| TTS       | 80-100ms        | 35-50ms       | 30-40ms             |
| **Total** | **100-150ms**  | **50-70ms**   | **30-50ms** 🎯      |

## Directory Structure

```
stream-tts-rust/
├── Cargo.toml           # Rust dependencies
├── src/
│   └── main.rs          # Rust parser + coordinator ✅
├── cpp/                 # C++ ML inference (Phase 2)
│   ├── translation.cpp
│   └── tts.cpp
├── python/              # Python prototypes (Phase 1)
│   ├── translation_worker.py
│   └── tts_worker.py
└── models/              # ML model files
    ├── nllb-200/
    └── xtts/
```

## Hardware Target

**M4 Max (Expected)**:
- 40-core GPU
- 128GB unified memory
- Metal 3 API
- Neural Engine (16-core)

**Key Advantages**:
- All data in unified memory (zero-copy possible)
- Massive GPU parallelism
- Hardware-accelerated ML ops
- Low-latency audio I/O

## License

Copyright 2025 Andrew Yates. All rights reserved.
