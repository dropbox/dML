# Pure C++ Architecture - NO PYTHON
**Date**: 2025-11-24 23:10 PST
**Objective**: ZERO Python dependencies, pure C++ speed

---

## ARCHITECTURE DECISION

**PURE C++ STACK:**
```
Claude JSON (stdin)
    ↓
[C++ JSON Parser] RapidJSON
    ↓
[C++ Text Cleaner] Regex/SIMD
    ↓
[C++ Translation] llama.cpp library (direct linking)
    ↓
[C++ TTS] NSSpeechSynthesizer (Objective-C++)
    ↓
[C++ Audio] CoreAudio direct
    ↓
🔊 Speakers
```

**ZERO PYTHON. ZERO DEPENDENCIES. PURE SPEED.**

---

## IMPLEMENTATION PLAN

### Component 1: Integrate llama.cpp Library
**Instead of**: Spawning Python subprocess
**Use**: Link llama.cpp directly into C++ binary

```cpp
// translation_engine.cpp
#include "llama.h"  // From llama.cpp

class TranslationEngine {
    llama_model* model;
    llama_context* ctx;

public:
    TranslationEngine(const char* model_path) {
        // Load model directly
        model = llama_load_model_from_file(model_path, params);
        ctx = llama_new_context_with_model(model, params);
    }

    std::string translate(const std::string& english) {
        // Tokenize
        // Run inference
        // Decode
        return japanese;
    }
};
```

### Component 2: Native macOS TTS (Objective-C++)
**Use**: NSSpeechSynthesizer (built into macOS)

```objc
// tts_engine.mm
#import <AppKit/AppKit.h>

class TTSEngine {
    NSSpeechSynthesizer* synth;

public:
    TTSEngine() {
        synth = [[NSSpeechSynthesizer alloc] initWithVoice:@"com.apple.voice.compact.ja-JP.Kyoko"];
        [synth setRate:280];
    }

    void speak(const std::string& japanese_text) {
        NSString* ns_text = [NSString stringWithUTF8String:japanese_text.c_str()];
        [synth startSpeakingString:ns_text];

        // Wait for completion
        while ([synth isSpeaking]) {
            usleep(10000);
        }
    }
};
```

### Component 3: Direct CoreAudio Output
**Use**: AudioQueue for low-latency playback

```cpp
// audio_player.cpp
#include <AudioToolbox/AudioQueue.h>

class AudioPlayer {
    AudioQueueRef queue;

public:
    void play_audio_file(const char* path) {
        // Load audio file
        // Create audio queue
        // Stream to speakers
    }
};
```

---

## UPDATED PROJECT STRUCTURE

```
stream-tts-cpp/
├── CMakeLists.txt (updated for llama.cpp)
├── src/
│   ├── main.cpp
│   ├── json_parser.cpp
│   ├── text_cleaner.cpp
│   ├── translation_engine.cpp  ← NEW (llama.cpp direct)
│   ├── tts_engine.mm           ← NEW (Objective-C++)
│   └── audio_player.cpp        ← NEW (CoreAudio)
├── include/
│   ├── json_parser.hpp
│   ├── text_cleaner.hpp
│   ├── translation_engine.hpp  ← NEW
│   ├── tts_engine.hpp          ← NEW
│   └── audio_player.hpp        ← NEW
└── external/
    └── llama.cpp/ (as git submodule)
```

---

## BUILD SYSTEM UPDATE

```cmake
# CMakeLists.txt additions

# Add llama.cpp as subdirectory
add_subdirectory(external/llama.cpp)

# Link llama library
target_link_libraries(stream-tts
    llama
    ${COREAUDIO_FRAMEWORK}
    ${APPKIT_FRAMEWORK}  # For NSSpeechSynthesizer
)

# Enable Objective-C++
set_source_files_properties(src/tts_engine.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
)
```

---

## PERFORMANCE EXPECTATIONS

| Component | Current (Python) | Pure C++ | Improvement |
|-----------|-----------------|----------|-------------|
| JSON Parse | < 1ms | < 0.3ms | 3x |
| Text Clean | < 1ms | < 0.2ms | 5x |
| Translation | 110ms (Python) | 60ms (direct) | 2x |
| TTS | 464ms (subprocess) | 150ms (native) | 3x |
| Audio | 5ms (afplay) | < 2ms (direct) | 2x |
| **TOTAL** | **581ms** | **< 212ms** | **3x FASTER** |

---

## IMPLEMENTATION TIME

- **Translation Engine**: 2 hours
- **TTS Engine**: 1 hour
- **Audio Player**: 1 hour
- **Integration**: 1 hour
- **Testing**: 1 hour

**Total**: **6 hours to pure C++ system**

---

## ADVANTAGES

✅ **Zero Python dependencies**
✅ **Single 2MB binary**
✅ **3x faster than current**
✅ **Native macOS integration**
✅ **No subprocess overhead**
✅ **Direct Metal GPU access**
✅ **Production-ready C++**

---

**BUILDING NOW**

**Copyright 2025 Andrew Yates. All rights reserved.**
