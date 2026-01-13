# Production Plan: Maximum Performance TTS System
## Path to < 50ms Latency on M4 Max

**Copyright 2025 Andrew Yates. All rights reserved.**

---

## OBJECTIVE

Build the **fastest possible** streaming TTS system for M4 Max:
- **Target Latency**: < 50ms total (translation + TTS + audio)
- **GPU Utilization**: > 95%
- **Quality**: Best available (NLLB-3.3B, XTTS v2)
- **No Compromises**: Maximum performance at all costs

---

## THREE-PHASE APPROACH

### Phase 1: Python Prototype (Week 1)
**Goal**: Validate models, measure baseline performance, prove architecture
**Duration**: 3-5 days
**Language**: Pure Python with PyTorch MPS

**Why Start Here**:
- Fast model integration (hours not days)
- Measure real Metal GPU performance
- Identify bottlenecks
- Validate quality
- Establish baseline for optimization

**Expected Performance**: 100-150ms (baseline)

### Phase 2: C++ Production (Week 2-3)
**Goal**: Maximum I/O performance, zero-cost abstractions
**Duration**: 7-10 days
**Language**: C++20 with Objective-C++ for Metal

**What Changes**:
- Replace Python I/O with C++
- Direct Metal API calls (no PyTorch overhead)
- Lock-free ring buffers
- Zero-copy IPC
- Inline GPU kernel dispatch

**Expected Performance**: 50-70ms (3x faster)

### Phase 3: GPU Optimization (Week 4)
**Goal**: Squeeze every microsecond
**Duration**: 5-7 days
**Language**: C++ + Metal Shading Language + CUDA-style kernels

**Optimizations**:
- Custom Metal kernels for attention
- Fused operations (translation + TTS in one pass)
- Tensor Core utilization (M4's matrix engines)
- Async compute with multiple queues
- Quantization (INT8/INT4) for speed

**Expected Performance**: 30-50ms (2x faster again)

**Final Target: 30-50ms total latency** ⚡

---

## DETAILED PHASE BREAKDOWN

## PHASE 1: PYTHON PROTOTYPE (Days 1-5)

### Day 1: Setup & Model Validation

**Tasks**:
1. Install PyTorch with MPS support
2. Download NLLB-200-3.3B model
3. Download Coqui XTTS v2 model
4. Verify models load on Metal GPU
5. Measure single-inference latency

**Deliverables**:
```python
# translation_bench.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B",
    torch_dtype=torch.bfloat16
).to("mps")

# Measure latency for 10 samples
# Target: 15-25ms per sentence
```

**Success Criteria**:
- Models load successfully
- Translation: < 30ms
- TTS: < 80ms

### Day 2: Basic Pipeline

**Tasks**:
1. Implement stdin JSON parser (Python)
2. Implement text cleaner
3. Connect parser → translation → TTS → audio
4. Measure end-to-end latency

**Deliverables**:
```python
# prototype_pipeline.py
import sys
import json
from translation import translate
from tts import synthesize
import sounddevice as sd

for line in sys.stdin:
    msg = json.loads(line)
    if is_assistant_text(msg):
        text = clean_text(msg['content'])
        translated = translate(text)  # Metal GPU
        audio = synthesize(translated)  # Metal GPU
        sd.play(audio, 22050)
```

**Success Criteria**:
- End-to-end pipeline works
- Total latency measured
- GPU utilization > 70%

### Day 3: Optimize Python Path

**Tasks**:
1. Profile with `py-spy` to find bottlenecks
2. Implement batch processing where possible
3. Async I/O with asyncio
4. Pre-allocate buffers
5. Minimize Python overhead

**Optimizations**:
```python
# Use torch.compile for M4
translation_model = torch.compile(
    translation_model,
    mode="max-autotune"
)

# Pre-allocate tensors
input_buffer = torch.zeros((1, 512), dtype=torch.long, device="mps")

# Async audio playback
async def audio_worker():
    while True:
        audio = await audio_queue.get()
        sd.play(audio, 22050, blocking=False)
```

**Success Criteria**:
- Latency improved by 20-30%
- Identified bottlenecks documented

### Day 4: Metal Performance Analysis

**Tasks**:
1. Use Xcode Instruments to profile Metal GPU
2. Identify GPU idle time
3. Measure memory bandwidth utilization
4. Find kernel inefficiencies
5. Document opportunities for C++ optimization

**Tools**:
```bash
# Profile with Instruments
instruments -t "GPU" -D trace.trace python prototype_pipeline.py

# Analyze Metal timeline
# Look for:
# - CPU-GPU synchronization overhead
# - Memory transfer latency
# - Kernel launch overhead
```

**Success Criteria**:
- Detailed performance profile
- List of optimization opportunities
- Estimated C++ performance gains

### Day 5: Baseline Documentation

**Tasks**:
1. Document final Python performance
2. Create performance report with graphs
3. Identify C++ optimization targets
4. Write Phase 2 implementation plan
5. Prepare test suite for C++ validation

**Deliverables**:
- `PHASE1_RESULTS.md` with:
  - Latency breakdown (JSON parse, translate, TTS, audio)
  - GPU utilization metrics
  - Memory usage analysis
  - Bottleneck identification
  - C++ optimization roadmap

**Success Criteria**:
- Python prototype: 100-150ms total latency
- Clear path to 50ms with C++

---

## PHASE 2: C++ PRODUCTION (Days 6-15)

### Day 6-7: C++ Project Foundation

**Tasks**:
1. Set up CMake project
2. Implement fast JSON parser (simdjson or RapidJSON)
3. Implement text cleaner with SIMD
4. Create Metal framework integration
5. Benchmark C++ I/O vs Python

**Implementation**:
```cpp
// Use simdjson for maximum speed
#include <simdjson.h>

simdjson::dom::parser parser;
auto doc = parser.parse(line);

// SIMD text cleaning
#include <immintrin.h>
// Use AVX2 for fast string operations
```

**Deliverables**:
- C++ project compiles
- JSON parsing: < 0.5ms
- Text cleaning: < 0.2ms

### Day 8-9: Metal Translation Integration

**Tasks**:
1. Convert NLLB model to Core ML or use Metal directly
2. Implement C++ → Metal inference pipeline
3. Optimize tensor allocation
4. Eliminate Python overhead
5. Benchmark vs Python

**Options**:

**Option A: Core ML (Easier)**
```cpp
#include <CoreML/CoreML.h>

// Load Core ML model
NSURL *modelURL = [NSURL fileURLWithPath:@"nllb.mlpackage"];
MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:&error];

// Inference
MLDictionaryFeatureProvider *input = ...;
id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
```

**Option B: Direct Metal (Faster)**
```cpp
#include <Metal/Metal.h>

id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];

// Load model weights to Metal buffers
id<MTLBuffer> weights = [device newBufferWithBytes:data
                                            length:size
                                           options:MTLResourceStorageModeShared];

// Dispatch compute kernels
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:input offset:0 atIndex:0];
[encoder setBuffer:output offset:0 atIndex:1];
[encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
```

**Target**: Translation latency < 15ms (vs 20ms Python)

### Day 10-11: Metal TTS Integration

**Tasks**:
1. Implement XTTS v2 in Metal/C++
2. Optimize vocoder (most expensive part)
3. Implement streaming audio generation
4. Zero-copy audio buffer
5. Benchmark vs Python

**Implementation**:
```cpp
// Streaming TTS - generate audio in chunks
class StreamingTTS {
    id<MTLBuffer> mel_buffer;
    id<MTLBuffer> audio_buffer;

    void synthesize_chunk(const char* text, size_t offset) {
        // Generate mel spectrogram
        encode_text_to_mel(text, mel_buffer, offset);

        // Stream through vocoder
        vocoder_generate(mel_buffer, audio_buffer, offset);

        // Push to audio ring buffer (non-blocking)
        audio_player->push_chunk(audio_buffer, chunk_size);
    }
};
```

**Target**: TTS latency < 40ms (vs 60ms Python)

### Day 12-13: Zero-Copy Audio Pipeline

**Tasks**:
1. Implement lock-free ring buffer
2. Direct CoreAudio integration
3. Shared memory for GPU → Audio
4. Eliminate all buffer copies
5. Test for audio glitches

**Implementation**:
```cpp
#include <AudioUnit/AudioUnit.h>

// Lock-free ring buffer using std::atomic
template<typename T, size_t Size>
class LockFreeRingBuffer {
    std::array<T, Size> buffer;
    std::atomic<size_t> write_pos{0};
    std::atomic<size_t> read_pos{0};

public:
    bool push(T value) {
        size_t current_write = write_pos.load(std::memory_order_relaxed);
        size_t next = (current_write + 1) % Size;

        if (next == read_pos.load(std::memory_order_acquire))
            return false; // Full

        buffer[current_write] = value;
        write_pos.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& value) {
        size_t current_read = read_pos.load(std::memory_order_relaxed);

        if (current_read == write_pos.load(std::memory_order_acquire))
            return false; // Empty

        value = buffer[current_read];
        read_pos.store((current_read + 1) % Size, std::memory_order_release);
        return true;
    }
};

// CoreAudio callback - called on real-time thread
OSStatus audio_callback(void* inRefCon, ...) {
    auto* ring_buffer = static_cast<LockFreeRingBuffer<float, 88200>*>(inRefCon);
    float* output = ...;

    for (size_t i = 0; i < num_frames; i++) {
        ring_buffer->pop(output[i]);
    }

    return noErr;
}
```

**Target**: Audio latency < 5ms, zero glitches

### Day 14-15: Integration & Testing

**Tasks**:
1. Wire all C++ components together
2. End-to-end testing with Claude
3. Stress testing (long sessions)
4. Memory leak detection
5. Performance validation

**Testing**:
```bash
# Stress test
for i in {1..1000}; do
    echo '{"content":[{"type":"text","text":"Test sentence"}]}'
done | ./stream-tts

# Profile
instruments -t "Time Profiler" ./stream-tts

# Memory leaks
leaks --atExit -- ./stream-tts
```

**Success Criteria**:
- End-to-end latency: 50-70ms
- No memory leaks
- No audio glitches
- Stable for hours

---

## PHASE 3: GPU OPTIMIZATION (Days 16-22)

### Day 16-17: Custom Metal Kernels

**Tasks**:
1. Profile NLLB to identify slow kernels
2. Rewrite bottleneck kernels in Metal Shading Language
3. Fuse operations (attention + FFN)
4. Optimize for M4's matrix engines
5. Benchmark improvements

**Metal Kernel Example**:
```metal
// optimized_attention.metal
kernel void fused_attention_kernel(
    device const float4* query [[buffer(0)]],
    device const float4* key [[buffer(1)]],
    device const float4* value [[buffer(2)]],
    device float4* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Use M4's SIMD width (32 for M4 Max)
    float4 q = query[gid.x];

    // Compute attention in shared memory (threadgroup memory)
    threadgroup float4 shared_mem[1024];

    // Fused attention + FFN in single kernel
    // Use simd_sum for fast reduction
    float score = simd_sum(q * k);

    // ... optimized attention implementation
}
```

**Target**: Translation latency 10-15ms (vs 15-20ms baseline)

### Day 18-19: Quantization & Compression

**Tasks**:
1. Quantize NLLB to INT8 (from BF16)
2. Implement fast INT8 kernels
3. Quantize-aware training if needed
4. Measure quality degradation
5. Balance speed vs quality

**Implementation**:
```cpp
// Dynamic quantization at runtime
void quantize_weights_int8() {
    // Convert FP32/BF16 weights to INT8
    for (auto& layer : model.layers) {
        // Find scale factor
        float max_val = find_max_abs(layer.weights);
        float scale = 127.0f / max_val;

        // Quantize
        for (auto& weight : layer.weights) {
            weight = static_cast<int8_t>(weight * scale);
        }

        layer.scale = scale;
    }
}

// INT8 matrix multiplication using M4's AMX
// (Advanced Matrix Extensions)
void int8_matmul_amx(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K) {
    // Use AMX instructions for 2-4x speedup on M4
    #if defined(__ARM_FEATURE_AMX)
    // AMX tile-based matrix multiplication
    // M4 can do 1024 INT8 ops per cycle per core
    #endif
}
```

**Target**: Translation latency 8-12ms (INT8), quality acceptable

### Day 20-21: Parallel Execution

**Tasks**:
1. Analyze dependency graph
2. Overlap translation + TTS where possible
3. Use multiple Metal command queues
4. Pipeline stages for continuous throughput
5. Async compute for maximum GPU utilization

**Implementation**:
```cpp
class PipelinedInference {
    id<MTLCommandQueue> translation_queue;
    id<MTLCommandQueue> tts_queue;
    id<MTLCommandQueue> copy_queue;

    void process_sentence(const std::string& text) {
        // Stage 1: Translate (queue 1)
        auto translation_cmd = [translation_queue commandBuffer];
        dispatch_translation(translation_cmd, text);
        [translation_cmd commit];

        // Stage 2: TTS (queue 2) - starts before translation finishes
        auto tts_cmd = [tts_queue commandBuffer];

        // Wait for translation to have first chunk
        [tts_cmd encodeWaitForEvent:translation_event value:1];
        dispatch_tts(tts_cmd, partial_translation);
        [tts_cmd commit];

        // Stage 3: Audio (queue 3) - copy while TTS runs
        auto copy_cmd = [copy_queue commandBuffer];
        [copy_cmd encodeWaitForEvent:tts_event value:1];
        copy_to_audio_buffer(copy_cmd);
        [copy_cmd commit];
    }
};
```

**Benefit**: Overlap = effective 30% latency reduction

### Day 22: Final Optimization Pass

**Tasks**:
1. Profile with Instruments one more time
2. Eliminate remaining bottlenecks
3. Tune buffer sizes
4. Optimize memory layout (cache-friendly)
5. Final performance validation

**Optimizations**:
```cpp
// Cache-friendly memory layout
struct alignas(64) CacheAlignedTensor {
    float data[16];  // Fit in cache line
};

// Prefetch next tensor
__builtin_prefetch(&tensors[i + 1]);

// Use huge pages for model weights
void* weights = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
madvise(weights, size, MADV_HUGEPAGE);
```

**Final Target**: 30-50ms total latency

---

## PERFORMANCE TARGETS

| Phase | Translation | TTS | Total | vs Baseline |
|-------|-------------|-----|-------|-------------|
| **Python Baseline** | 20-25ms | 60-80ms | 100-150ms | 1.0x |
| **C++ w/ Metal** | 15-18ms | 40-50ms | 60-75ms | 2.0x |
| **Custom Kernels** | 10-15ms | 35-45ms | 50-65ms | 2.5x |
| **Quantized + Optimized** | 8-12ms | 30-40ms | **40-55ms** | **3.0x** |
| **Pipelined + Fused** | - | - | **30-50ms** | **4.0x** |

**Final Goal: 30-50ms end-to-end latency**

---

## TECHNICAL STACK

### Languages
- **Phase 1**: Python 3.11+
- **Phase 2-3**: C++20, Objective-C++, Metal Shading Language

### Frameworks
- **Metal Performance Shaders (MPS)**
- **Core ML** (optional for easier integration)
- **Metal Compute** (direct kernel dispatch)
- **AudioUnit** / **CoreAudio**

### Libraries
- **simdjson**: Fast JSON parsing
- **Eigen** or **xtensor**: Tensor operations in C++
- **{fmt}**: Fast string formatting
- **spdlog**: Fast logging

### Tools
- **Xcode Instruments**: GPU profiling
- **lldb**: Debugging
- **CMake**: Build system
- **Metal Debugger**: Kernel analysis

---

## RISK MITIGATION

### Risk 1: Model Conversion to Metal
**Mitigation**: Keep Python bridge as fallback, convert incrementally

### Risk 2: Quality Degradation with Quantization
**Mitigation**: A/B test, user-configurable quality modes

### Risk 3: Audio Glitches Under Load
**Mitigation**: Lock-free buffers, real-time priority thread

### Risk 4: M4-Specific Code Portability
**Mitigation**: Runtime feature detection, fallback paths

---

## SUCCESS METRICS

### Phase 1 Success
- ✅ Python prototype < 150ms
- ✅ Models validated on Metal
- ✅ Bottlenecks identified
- ✅ C++ optimization roadmap

### Phase 2 Success
- ✅ C++ implementation < 70ms
- ✅ No audio glitches
- ✅ Stable for hours
- ✅ GPU utilization > 90%

### Phase 3 Success
- ✅ **Final latency < 50ms**
- ✅ **GPU utilization > 95%**
- ✅ **Quality maintained**
- ✅ **Production-ready**

---

## DELIVERABLES

### Phase 1
- `prototype_pipeline.py` - Working Python prototype
- `PHASE1_RESULTS.md` - Performance analysis
- Test suite for validation

### Phase 2
- `stream-tts/` - C++ codebase
- `CMakeLists.txt` - Build system
- `benchmarks/` - Performance tests
- `PHASE2_RESULTS.md` - C++ performance

### Phase 3
- `metal_kernels/` - Custom Metal shaders
- `optimized_models/` - Quantized models
- `FINAL_BENCHMARK.md` - Complete analysis
- Production binary

---

## TIMELINE

- **Week 1** (Days 1-5): Python Prototype
- **Week 2** (Days 6-10): C++ Foundation + Metal Integration
- **Week 3** (Days 11-15): Audio + Testing
- **Week 4** (Days 16-22): GPU Optimization

**Total: 3-4 weeks to production**

---

This plan is aggressive but achievable. The key is measuring at every step and optimizing based on data, not assumptions.

**Copyright 2025 Andrew Yates. All rights reserved.**
