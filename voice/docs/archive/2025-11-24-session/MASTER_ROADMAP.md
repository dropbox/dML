# MASTER ROADMAP: Ultimate TTS System
## From Current (258ms) → Ultimate (< 50ms) with Best Quality

**Date**: 2025-11-24
**Objective**: Build the ABSOLUTE BEST streaming translation + TTS system
**Target**: < 50ms latency, SOTA quality, C++ + Metal GPU
**Status**: Phase 1 in progress

---

## OVERVIEW

### Current State ✅
- **System**: Rust + NLLB-600M + gTTS
- **Performance**: 258ms latency
- **Quality**: BLEU 28-30, MOS 4.0 (good)
- **Status**: WORKING & TESTED

### Ultimate Goals 🎯
1. **Quality**: BLEU 35+, MOS 4.5+ (near GPT-4o + best TTS)
2. **Speed**: < 50ms total latency (10x faster than current)
3. **Features**: Voice cloning, emotion control
4. **Architecture**: Pure C++ with direct Metal API calls

---

## THREE-PHASE EXECUTION PLAN

### **PHASE 1: Best Quality (Python/Metal)** ⏱️ 2-4 hours
**Goal**: Get SOTA quality immediately with Python + Metal GPU

#### 1.1: Install Qwen2.5-7B ⚡ IN PROGRESS
- Model: Q3_K_M (3.4GB, single file)
- Quality: BLEU 33+ (excellent)
- Speed: 50-80ms on Metal
- **Status**: Downloading now

#### 1.2: Install CosyVoice 3.0 ⏱️ 30 min
- Model: CosyVoice-300M-SFT (~3GB)
- Quality: MOS 4.5+ (best available)
- Speed: ~150ms on Metal
- **Status**: Next after Qwen

#### 1.3: Create Ultimate Pipeline ⏱️ 1 hour
- Integrate Qwen + CosyVoice
- Python workers with Metal GPU
- Test end-to-end
- **Target**: 200-230ms total latency

#### 1.4: Compare Systems ⏱️ 15 min
- A/B test: Current vs Ultimate
- Measure quality improvement
- Document results

**Phase 1 Output**:
- ✅ Current system (258ms, BLEU 28-30)
- ✅ Ultimate system (200-230ms, BLEU 33+, MOS 4.5+)

---

### **PHASE 2: C++ Foundation** ⏱️ 2-3 days
**Goal**: Build production C++ system with modular architecture

#### 2.1: Project Setup ⏱️ 2 hours
```cpp
stream-tts-cpp/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── json_parser.cpp
│   ├── text_cleaner.cpp
│   ├── worker_manager.cpp
│   ├── metal_translation.mm
│   ├── metal_tts.mm
│   └── audio_player.cpp
├── include/
│   └── *.h
├── metal/
│   └── kernels.metal
└── tests/
    └── *_test.cpp
```

- CMake build system
- Project structure
- Dependencies (RapidJSON, Metal frameworks)
- **Target**: Compiles successfully

#### 2.2: JSON Parser + Text Cleaner ⏱️ 4 hours
- RapidJSON for fast parsing
- SIMD text cleaning (AVX2/NEON)
- Markdown/code removal
- Sentence segmentation
- **Target**: < 1ms per message

#### 2.3: Worker Manager ⏱️ 6 hours
- Launch Python workers
- Unix socket IPC
- Request/response protocol
- Error handling
- **Target**: Reliable communication

#### 2.4: Integration Test ⏱️ 2 hours
- End-to-end with Python workers
- Measure latency
- **Target**: Same quality, < 300ms

**Phase 2 Output**:
- ✅ C++ coordinator working
- ✅ Python workers integrated
- ✅ < 300ms latency

---

### **PHASE 3: Direct Metal Implementation** ⏱️ 1-2 weeks
**Goal**: Replace Python with pure C++/Metal for maximum speed

#### 3.1: Metal Translation ⏱️ 3-4 days

**Step 1**: Convert NLLB to Core ML
```bash
# Python conversion script
python3 convert_nllb_to_coreml.py \
    --model facebook/nllb-200-distilled-600M \
    --output nllb.mlpackage \
    --quantize int8
```

**Step 2**: C++/Objective-C++ wrapper
```objc
// metal_translation.mm
#import <CoreML/CoreML.h>

class MetalTranslator {
    MLModel* model;

public:
    MetalTranslator(const char* model_path) {
        NSURL* url = [NSURL fileURLWithPath:@(model_path)];
        model = [MLModel modelWithContentsOfURL:url error:nil];
    }

    std::string translate(const std::string& english) {
        // Tokenize
        // Run inference
        // Decode
        // Return Japanese
    }
};
```

**Target**: 15-25ms translation

#### 3.2: Metal TTS ⏱️ 4-5 days

**Option A**: XTTS v2 via Core ML
- Convert XTTS to Core ML
- Metal vocoder
- **Target**: 40-60ms

**Option B**: Native Metal TTS
- Implement Tacotron2 in Metal
- Fast vocoder (HiFiGAN)
- **Target**: 30-50ms

**Target**: 30-60ms TTS

#### 3.3: CoreAudio Integration ⏱️ 2 days
```cpp
// audio_player.cpp
class AudioPlayer {
    AudioUnit audio_unit;
    RingBuffer<float> buffer;  // Lock-free

public:
    void play(const float* samples, size_t count) {
        // Push to ring buffer
        // CoreAudio callback pulls from buffer
    }
};
```

**Target**: < 5ms audio latency

#### 3.4: End-to-End Optimization ⏱️ 2-3 days
- Profile with Instruments
- Optimize hot paths
- Custom Metal kernels
- Zero-copy pipelines
- **Target**: < 70ms total

**Phase 3 Output**:
- ✅ Pure C++ + Metal system
- ✅ 50-70ms latency
- ✅ BLEU 33+, MOS 4.5+

---

### **PHASE 4: Extreme Optimization** ⏱️ 1 week
**Goal**: Push to < 50ms with custom kernels

#### 4.1: Custom Metal Kernels ⏱️ 3 days
```metal
// attention_kernel.metal
kernel void fused_attention(
    device const float4* query [[buffer(0)]],
    device const float4* key [[buffer(1)]],
    device const float4* value [[buffer(2)]],
    device float4* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Optimized attention for M4 Max
    // Use SIMD groups, threadgroup memory
}
```

**Optimizations**:
- Fused attention + FFN
- INT8 quantization
- Tensor Core utilization (M4 AMX)
- **Target**: 8-15ms translation

#### 4.2: Vocoder Optimization ⏱️ 2 days
- Custom vocoder kernels
- Streaming generation
- **Target**: 25-35ms TTS

#### 4.3: Pipeline Parallelization ⏱️ 2 days
- Overlap translation + TTS
- Multiple command queues
- Async compute
- **Target**: 30-40ms effective latency

**Phase 4 Output**:
- ✅ **30-50ms total latency**
- ✅ **95%+ GPU utilization**
- ✅ **SOTA quality maintained**

---

## PERFORMANCE TARGETS

| Phase | Translation | TTS | Total | Quality | Status |
|-------|-------------|-----|-------|---------|--------|
| **Current** | 110ms | 464ms | 574ms | BLEU 28-30, MOS 4.0 | ✅ Working |
| **Phase 1 (Ultimate)** | 60ms | 150ms | 210ms | BLEU 33+, MOS 4.5+ | 🔄 Installing |
| **Phase 2 (C++ Base)** | 60ms | 150ms | 210ms | BLEU 33+, MOS 4.5+ | 📝 Planned |
| **Phase 3 (Metal)** | 20ms | 50ms | 70ms | BLEU 33+, MOS 4.5+ | 📝 Planned |
| **Phase 4 (Optimized)** | 12ms | 30ms | **42ms** | BLEU 33+, MOS 4.5+ | 🎯 Goal |

---

## TECHNICAL STACK

### Languages
- **Phase 1**: Python 3.11+
- **Phase 2-4**: C++20, Objective-C++, Metal Shading Language

### Frameworks
- **Metal Performance Shaders (MPS)** - Phase 1
- **Core ML** - Phase 3
- **Metal Compute** - Phase 3-4
- **CoreAudio** - Phase 2-4

### Libraries
- **RapidJSON** - Fast JSON parsing
- **PyTorch** - Phase 1 ML
- **llama.cpp** - Qwen inference
- **CosyVoice** - Phase 1 TTS

### Tools
- **CMake** - Build system
- **Xcode Instruments** - Profiling
- **Metal Debugger** - GPU analysis

---

## MODELS

### Translation
**Phase 1**: Qwen2.5-7B-Instruct Q3_K_M
- Size: 3.4GB
- Quality: BLEU 33+
- Speed: 60-80ms (llama.cpp + Metal)

**Phase 3+**: NLLB-200-distilled-600M
- Size: 600MB (INT8: 150MB)
- Quality: BLEU 28-30
- Speed: 12-20ms (Core ML + Metal)

### TTS
**Phase 1**: CosyVoice-300M-SFT
- Size: 3GB
- Quality: MOS 4.5+
- Speed: ~150ms (PyTorch + Metal)

**Phase 3+**: XTTS v2 or Custom
- Size: 2GB (optimized)
- Quality: MOS 4.3+
- Speed: 30-50ms (Core ML + Metal)

---

## DELIVERABLES

### Phase 1
- ✅ Qwen2.5-7B installed
- ✅ CosyVoice installed
- ✅ Ultimate pipeline working
- ✅ Performance report

### Phase 2
- ✅ C++ project structure
- ✅ JSON parser + text cleaner
- ✅ Worker manager
- ✅ Integration tests

### Phase 3
- ✅ Metal translation
- ✅ Metal TTS
- ✅ CoreAudio playback
- ✅ < 70ms latency

### Phase 4
- ✅ Custom Metal kernels
- ✅ < 50ms latency
- ✅ Production binary
- ✅ Complete documentation

---

## RISK MITIGATION

### Risk: Model conversion issues
**Mitigation**: Keep Python bridge, test incrementally

### Risk: Quality degradation with optimization
**Mitigation**: A/B testing at each phase, user validation

### Risk: Audio glitches
**Mitigation**: Lock-free buffers, real-time priority

### Risk: M4-specific code
**Mitigation**: Runtime feature detection, fallback paths

---

## SUCCESS METRICS

### Phase 1
- ✅ BLEU > 33
- ✅ MOS > 4.5
- ✅ Latency < 230ms

### Phase 2
- ✅ C++ compiles and runs
- ✅ Same quality as Phase 1
- ✅ Latency < 300ms

### Phase 3
- ✅ Pure C++ + Metal
- ✅ Latency < 70ms
- ✅ Quality maintained

### Phase 4
- ✅ **Latency < 50ms**
- ✅ **GPU utilization > 95%**
- ✅ **Production-ready**

---

## TIMELINE

- **Today (Nov 24)**: Phase 1.1-1.3 (Qwen + CosyVoice + pipeline)
- **Tomorrow (Nov 25)**: Phase 1.4 + Phase 2.1-2.2
- **Week 1**: Phase 2 complete
- **Week 2**: Phase 3 start
- **Week 3**: Phase 3 complete
- **Week 4**: Phase 4 optimization
- **Result**: **< 50ms production system in 4 weeks**

---

## CURRENT STATUS

**Right Now** (Nov 24, 22:40):
- ✅ Current system tested (258ms, working great!)
- 🔄 Qwen2.5-7B Q3_K_M downloading (~3.4GB)
- ⏳ CosyVoice next (after Qwen completes)
- ⏳ C++ implementation starts tomorrow

**Next 2 Hours**:
1. Finish Qwen download
2. Install CosyVoice
3. Create Ultimate pipeline
4. Test and compare

**Tomorrow**:
1. Start C++ project structure
2. Implement JSON parser
3. Begin worker manager

---

**LET'S BUILD THE BEST TTS SYSTEM IN THE WORLD** 🚀

**Copyright 2025 Andrew Yates. All rights reserved.**
