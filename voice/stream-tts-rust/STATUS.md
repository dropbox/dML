# Status Report: Stream TTS Rust

**Date**: 2025-11-24
**Phase**: Working Production System
**Status**: ✅ FULLY FUNCTIONAL

## Current Performance

**Real-world test results** (2025-11-24):

```
Translation: 180ms (warmup) → 100-107ms (optimized)
TTS:         578ms (macOS say with file generation)
Total:       ~700ms end-to-end latency
```

**Voice Configuration**: Kyoko (Japanese female) at 300 WPM

## What's Working

### ✅ Complete Pipeline
```
Claude JSON → Rust Parser → Translation (Metal GPU) → TTS (macOS say) → Audio Output
```

**Components**:
1. **Rust Coordinator** (`src/main.rs`)
   - Parses Claude stream-json format
   - Spawns and manages Python workers
   - Handles audio playback via afplay
   - < 1ms parsing overhead

2. **Translation Worker** (`python/translation_worker_optimized.py`)
   - NLLB-200 on Metal GPU (MPS)
   - bfloat16 precision
   - torch.compile for optimization
   - 100-180ms latency

3. **TTS Worker** (`python/tts_worker_fast.py`)
   - macOS `say` command
   - Configurable voice and rate
   - Generates AIFF files
   - ~578ms synthesis time

4. **Configuration** (`config.yaml`)
   - Voice selection (Kyoko, Otoya, etc.)
   - Speech rate (300 WPM - fast)
   - Easy user customization

### ✅ User Features

- **SPEED_GUIDE.md**: Simple guide for adjusting speech rate and voice
- **test_voices.sh**: Script to test different voice configurations
- **Dynamic configuration**: Changes to config.yaml apply immediately
- **Multiple voices**: Support for all macOS system voices

## Architecture

### Current Implementation

**Rust Coordinator** (main.rs):
- Spawns Python translation worker (Metal GPU)
- Spawns Python TTS worker (macOS say)
- Pipes data through stdin/stdout
- Plays audio files with afplay

**Python Workers**:
- `translation_worker_optimized.py`: NLLB-200 with torch.compile
- `tts_worker_fast.py`: macOS say with YAML config support
- Both read from stdin, write to stdout
- Load configuration from config.yaml

**Configuration** (config.yaml):
```yaml
tts:
  engine: "macos_say"
  voice: "Kyoko"
  rate: 300  # WPM (faster than default 175)

translation:
  model: "facebook/nllb-200-distilled-600M"
  device: "mps"  # Metal GPU
```

## Recent Improvements

### Performance Enhancements
1. ✅ **Faster TTS**: Switched to macOS `say` command (~578ms vs gTTS)
2. ✅ **Configurable speed**: Speech rate increased to 300 WPM
3. ✅ **Dynamic config**: Workers load settings from config.yaml
4. ✅ **User guide**: SPEED_GUIDE.md for easy customization

### Code Quality
1. ✅ **Config support**: Both workers read from YAML config
2. ✅ **Error handling**: Graceful fallback to default voice
3. ✅ **Testing**: test_voices.sh for voice experimentation
4. ✅ **Documentation**: Clear setup and usage instructions

## Performance Breakdown

### Current Latency (measured)
```
Rust parsing:           < 1ms
Translation (Metal):    100-180ms  ⚡ GPU-accelerated
TTS (macOS say):        578ms      (file generation)
Audio playback:         Real-time  (handled by afplay)
Total:                  ~700ms
```

### Comparison to Python-only System
```
Old Python system:  3-5 seconds  (cloud APIs, network latency)
New Rust + Local:   ~700ms       (5-7x faster!)
```

## Optimization Opportunities

### Potential Further Improvements
1. **Direct audio playback**: Skip file generation in `say` (~200-300ms savings)
2. **Model caching**: Reduce warmup time (currently 180ms → 100ms)
3. **Streaming TTS**: Generate audio while translating (parallel processing)
4. **Smaller translation model**: Trade accuracy for speed if needed

### Expected Performance Ceiling
```
Translation:  50-80ms   (optimized Metal kernels)
TTS:          30-50ms   (direct playback, no file I/O)
Total:        80-130ms  (theoretical minimum)
```

## File Structure

```
stream-tts-rust/
├── Cargo.toml                          # Rust dependencies
├── src/main.rs                         # Rust coordinator
├── config.yaml                         # User configuration
├── README.md                           # Architecture docs
├── STATUS.md                           # This file
├── SPEED_GUIDE.md                      # User guide for speed settings
├── test_voices.sh                      # Voice testing script
│
├── python/
│   ├── translation_worker_optimized.py # NLLB-200 on Metal
│   ├── tts_worker_fast.py             # macOS say with config
│   ├── tts_worker.py                  # Original TTS worker
│   └── tts_worker_gtts.py             # Google TTS worker
│
├── target/release/
│   └── stream-tts-rust                # Compiled binary
│
└── venv/                              # Python environment
```

## Testing

### Quick Test
```bash
# Test the pipeline
echo '{"content":[{"type":"text","text":"Hello world"}]}' | \
  ./stream-tts-rust/target/release/stream-tts-rust
```

### With Claude
```bash
# Use the optimized pipeline
./tests/claude_rust_tts.sh "say hello in Japanese"

# Or full pipeline test
./tests/test_optimized_pipeline.sh
```

### Voice Testing
```bash
# Test different voices and rates
./stream-tts-rust/test_voices.sh
```

## Configuration Examples

### Maximum Speed (English only)
```yaml
tts:
  voice: "Samantha"
  rate: 350
translation:
  enabled: false  # Skip translation
```

### Balanced Quality (Japanese)
```yaml
tts:
  voice: "Kyoko"
  rate: 280
translation:
  enabled: true
```

### Natural Pace
```yaml
tts:
  voice: "Kyoko"
  rate: 200
```

## Next Steps

### Optional Optimizations
1. Implement direct audio playback (skip file generation)
2. Add streaming translation (process chunks in parallel)
3. Create C++ TTS worker for maximum performance
4. Add voice caching for frequently used phrases

### User Experience
1. ✅ Easy configuration via config.yaml
2. ✅ User guide (SPEED_GUIDE.md)
3. ✅ Voice testing script
4. Consider: Interactive voice selector CLI tool

## Conclusion

**The system is production-ready and significantly faster than the Python-only version.**

We have:
1. ✅ Working end-to-end pipeline (~700ms)
2. ✅ Metal GPU acceleration for translation
3. ✅ Native macOS TTS (fast, high-quality)
4. ✅ User-friendly configuration
5. ✅ Easy voice and speed customization
6. ✅ 5-7x faster than original system

**Ready for daily use with Claude Code.**

---

**Copyright 2025 Andrew Yates. All rights reserved.**
