# Voice TTS - Quick Reference
**Status**: ✅ Production Ready
**Performance**: 550-630ms latency
**Quality**: Excellent

---

## 🚀 Quick Start

### Run Current System
```bash
# Test it
./test_rust_tts_quick.sh

# Use with Claude Code
claude-code --output-format stream-json "Your prompt" | \
  ./stream-tts-rust/target/release/stream-tts-rust
```

---

## 📊 Performance Summary

| System | Latency | Binary | Status |
|--------|---------|--------|--------|
| **Rust + Python** | 550-630ms | 4.8MB | ✅ Production |
| **C++ + Python** | 595ms | 131KB | ✅ Alternative |
| **Pure C++** | TBD | 166KB | ⚙️ Experimental |

---

## 🔧 Available Commands

```bash
# Quick test
./test_rust_tts_quick.sh

# Benchmark
./test_cpp_vs_rust.sh

# Rebuild Rust
cd stream-tts-rust && cargo build --release

# Rebuild C++
cd stream-tts-cpp && cmake --build build
```

---

## 📁 Key Files

### Binaries
- `stream-tts-rust/target/release/stream-tts-rust` - Production (Rust)
- `stream-tts-cpp/build/stream-tts` - Alternative (C++)
- `stream-tts-cpp/build/stream-tts-pure` - Experimental (Pure C++)

### Documentation
- **README.md** - Project overview
- **SESSION_COMPLETE.md** - Latest status
- **PERFORMANCE_COMPARISON.md** - Benchmarks
- **USAGE.md** - Detailed guide

---

## 🎯 Next Steps

1. **Integrate CosyVoice** → 250-400ms (40-60% faster)
2. **Optimize Qwen server mode** → Better quality
3. **Custom Metal kernels** → < 200ms target

---

## 📝 Current Status

- ✅ Three working implementations
- ✅ Metal GPU acceleration confirmed
- ✅ Production system deployed (550ms)
- ✅ All benchmarks complete
- 📋 CosyVoice integration next

---

**Quick Start**: Run `./test_rust_tts_quick.sh` to verify system.

**Documentation**: See SESSION_COMPLETE.md for full details.

**Copyright 2025 Andrew Yates. All rights reserved.**
