# MLX Inference Engine

Unified C++ runtime for all MLX models, providing thread-safe parallel inference.

## Status (Updated 2025-12-23)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| Kokoro TTS | **VERIFIED** | ~95KB | Full integration, Whisper transcribes C++ output correctly (#1545) |
| Whisper STT | **FUNCTIONAL** | 4066 | Gate 0 passes (100% text match, 91% timestamp match). Caveats: VAD disabled, <10s files only |
| Translation | **VERIFIED** | 842 | DE/FR translations correct (#1545) |
| LLM | **VERIFIED** | 1065 | Text generation, chat, sampling tests pass (#1545) |
| CosyVoice TTS | NOT IMPLEMENTED | N/A | No C++ implementation |

**Note**: LLM, Translation, and Kokoro verified in commit #1545 (2025-12-23).

## Building

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- MLX C++ library (`brew install mlx`)
- espeak-ng library (`brew install espeak-ng`)
- clang++ with C++17 support

### Build Commands

```bash
cd src/mlx_inference_engine

# Build test executable
make

# Run basic tests (error handling)
make test

# Run with Kokoro model
./test_engine /path/to/kokoro_cpp_export
```

## Usage

```cpp
#include "mlx_inference_engine.hpp"

int main() {
    // Create engine
    mlx_inference::MLXInferenceEngine engine;

    // Load Kokoro TTS model
    engine.load_kokoro("/path/to/kokoro_cpp_export");

    // Configure TTS
    mlx_inference::TTSConfig config;
    config.voice = "af_heart";
    config.speed = 1.0f;

    // Synthesize speech
    mlx_inference::AudioOutput audio = engine.synthesize("Hello world", config);

    // Use audio.samples (std::vector<float>) and audio.sample_rate (24000 Hz)
    std::cout << "Generated " << audio.duration_seconds << " seconds of audio\n";

    return 0;
}
```

## API Reference

### MLXInferenceEngine

Main class providing unified access to all models.

#### Model Loading

```cpp
void load_kokoro(const std::string& model_path);      // Load Kokoro TTS
void load_cosyvoice(const std::string& model_path);   // Load CosyVoice TTS
void load_translation(const std::string& model_path, const std::string& model_type = "madlad");
void load_whisper(const std::string& model_name);     // Load Whisper STT
void load_llm(const std::string& model_path);         // Load LLM
```

#### TTS Inference

```cpp
AudioOutput synthesize(const std::string& text, const TTSConfig& config = TTSConfig());
AudioOutput synthesize_cosyvoice(const std::string& text, const TTSConfig& config = TTSConfig());
```

#### Translation Inference

```cpp
std::string translate(const std::string& text, const TranslationConfig& config = TranslationConfig());
```

#### STT Inference

```cpp
TranscriptionResult transcribe(const std::vector<float>& audio, int sample_rate,
                               const TranscriptionConfig& config = TranscriptionConfig());
TranscriptionResult transcribe_file(const std::string& audio_path,
                                    const TranscriptionConfig& config = TranscriptionConfig());
```

#### LLM Inference

```cpp
GenerationResult generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig());
```

### Configuration Structs

#### TTSConfig

```cpp
struct TTSConfig {
    std::string voice = "af_heart";     // Voice ID
    float speed = 1.0f;                 // Speaking rate (0.5 - 2.0)
    std::string emotion = "neutral";    // Emotion style
    bool enable_prosody = true;         // Enable prosody model
};
```

#### TranslationConfig

```cpp
struct TranslationConfig {
    std::string source_lang = "en";     // Source language code
    std::string target_lang = "de";     // Target language code
    int max_length = 512;               // Max output length
    bool use_quantized = true;          // Use 8-bit quantized model
};
```

#### TranscriptionConfig

```cpp
struct TranscriptionConfig {
    std::string model = "large-v3-turbo";  // Whisper model variant
    std::string language = "";             // Language (empty = auto-detect)
    bool enable_timestamps = false;        // Include word timestamps
    float temperature = 0.0f;              // 0 = greedy decoding
};
```

#### GenerationConfig

```cpp
struct GenerationConfig {
    int max_tokens = 512;           // Max tokens to generate
    float temperature = 0.7f;       // Sampling temperature
    float top_p = 0.9f;             // Nucleus sampling threshold
    int top_k = 40;                 // Top-k sampling limit
};
```

## Thread Safety

The engine is designed for thread-safe parallel inference:

- Multiple threads can call inference methods concurrently
- Model loading requires exclusive access (internally synchronized)
- Each inference call is independent

Example:

```cpp
mlx_inference::MLXInferenceEngine engine;
engine.load_kokoro("/path/to/model");

// Parallel synthesis from multiple threads
std::vector<std::thread> threads;
for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&engine, i]() {
        auto audio = engine.synthesize("Thread " + std::to_string(i));
        // Process audio...
    });
}
for (auto& t : threads) {
    t.join();
}
```

## Future Work

1. **CosyVoice TTS** - Flow-matching TTS integration
2. **Translation** - NLLB, MADLAD, OPUS-MT C++ wrappers
3. **Whisper STT** - WhisperMLX C++ integration
4. **LLM** - mlx-lm patterns for LLaMA inference
5. **Streaming** - Real-time streaming interfaces

## Files

- `mlx_inference_engine.hpp` - Public API header
- `mlx_inference_engine.cpp` - Implementation
- `test_engine.cpp` - Test program
- `CMakeLists.txt` - CMake build configuration
- `Makefile` - Simple makefile for quick builds
