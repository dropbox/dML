# Kokoro C++ Runtime

C++ implementation of the Kokoro TTS pipeline for maximum performance.

## Status

| Component | Status | File |
|-----------|--------|------|
| Tokenizer | ✓ Complete | tokenizer.{h,cpp} |
| G2P (Misaki) | ✓ Complete | g2p.{h,cpp} |
| API Header | ✓ Complete | kokoro.h |
| Model Loading | ✓ Complete | model.{h,cpp} |
| MLX Inference | ✓ Complete | model.cpp |

## Current Capabilities

Full TTS pipeline is implemented and produces intelligible speech (Whisper validated).

```cpp
// Full TTS pipeline
kokoro::Model model = kokoro::Model::load("path/to/model");
auto audio = model.synthesize("Hello world", "af_bella");
// Returns: AudioOutput with sample_rate=24000, audio samples

// Or use individual components:
kokoro::G2P g2p;
g2p.initialize("en-us");
std::string phonemes = g2p.phonemize("Hello world");
// Result: "həlˈoʊ wˈɜːld"

kokoro::Tokenizer tokenizer;
tokenizer.load_vocab("vocab/phonemes.json");
auto tokens = tokenizer.tokenize(phonemes);
// Result: [0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 0]
```

## Build Requirements

### Dependencies

1. **Misaki Lexicons** (for G2P - **REQUIRED**)

   The Kokoro model was trained on Misaki phonemes. You must export the Misaki
   lexicons before using the C++ runtime:

   ```bash
   # Export Misaki lexicons (run once)
   python3 scripts/export_misaki_lexicons.py
   # Creates: misaki_export/en/us_golds.json, us_silvers.json, etc.
   ```

2. **MLX C++** (installed via homebrew)
   ```bash
   brew install mlx
   ```

3. **espeak-ng** (optional fallback only)
   ```bash
   brew install espeak-ng
   ```
   Note: espeak-ng is NOT recommended for production use. Misaki is the correct
   G2P for Kokoro models.

### Building

The recommended way to build is using CMake from the mlx_inference_engine directory:

```bash
cd src/mlx_inference_engine
cmake -B build
cmake --build build -j8

# Run Misaki G2P test
./build/test_misaki_g2p  # Run from project root where misaki_export/ exists
```

For standalone builds:

```bash
cd src/kokoro

# Build full pipeline test (requires misaki_export/ in working directory)
clang++ -std=c++17 -O3 -DNDEBUG \
    -I/opt/homebrew/include \
    -I../mlx_inference_engine \
    -L/opt/homebrew/lib \
    kokoro.cpp model.cpp g2p.cpp tokenizer.cpp \
    ../mlx_inference_engine/misaki_g2p.cpp \
    test_pipeline.cpp \
    -lmlx -o test_pipeline

# Run pipeline test (make sure misaki_export/ exists!)
./test_pipeline /path/to/kokoro_cpp_export "Hello world"
```

### Running Tests

```bash
./test_tokenizer          # Tokenizer unit tests
./test_g2p               # G2P unit tests
./test_integration       # G2P + Tokenizer integration
./test_mlx_load          # MLX safetensors loading
./test_model             # Model loading
./test_pipeline          # Full pipeline test
```

## Architecture

```
Text → G2P → Phonemes → Tokenizer → Token IDs → Model → Audio
              ↑                         ↑
        Misaki lexicons            MLX C++
```

### G2P Priority Order

The G2P system uses Misaki lexicons with this priority order:
1. Manual overrides (add_symbols.json)
2. Symbols (symbols.json) - `%` → "percent", `&` → "and"
3. Gold lexicon (us_golds.json) - 176K high-confidence entries
4. Silver lexicon (us_silvers.json) - 186K derived entries
5. (Optional) espeak-ng fallback for OOV words - NOT recommended

## Files

- `kokoro.h` - Public API header (Model interface)
- `kokoro.cpp` - Public API implementation (integrates all components)
- `model.h/cpp` - Model weight loading and inference
- `tokenizer.h/cpp` - UTF-8 IPA tokenization
- `g2p.h/cpp` - Misaki lexicon-based grapheme-to-phoneme
- `test_*.cpp` - Unit and integration tests
- `CMakeLists.txt` - CMake build configuration

## Implementation Status

All components complete:
1. ~~Install MLX C++ library~~ ✓
2. ~~Implement model weight loading from safetensors~~ ✓
3. ~~Integrate with G2P and Tokenizer~~ ✓
4. ~~Forward pass (BERT, Predictor, Decoder)~~ ✓
5. Shape-bucketed compilation (LOW priority optimization)

## Performance Target

| Metric | Target |
|--------|--------|
| Latency | ~10-15ms for 3.58s audio |
| Real-time factor | 250-350x |
| Overhead | Zero per-call I/O/compilation/sync |

## Debug/Profiling Toggles

All debug outputs are disabled by default. Enable them only when actively debugging.

| Variable | Description |
|----------|-------------|
| `DEBUG_GENERATOR=1` | Print tensor stats during generator |
| `DEBUG_BERT=1` | Print BERT encoder tensor stats |
| `DEBUG_BERT_STEPS=1` | Print BERT intermediate step stats |
| `DEBUG_TOKENS=1` | Print phonemes and token IDs |
| `SAVE_DEBUG_TENSORS=1` | Save intermediate tensors to `/tmp/*.npy` |
| `SAVE_RB_TENSORS=1` | Save resblock inputs to `/tmp/*.npy` |
| `SAVE_ISTFT_INPUT=1` | Save ISTFT input to `/tmp/*.npy` |

## Notes

- **Misaki is the correct G2P for Kokoro** - the model was trained on Misaki phonemes
- espeak-ng produces different phoneme representations (oʊ vs O) and should not be used
- Misaki lexicons contain ~362K English entries (176K gold + 186K silver)
- Voice embeddings loaded from safetensors (shape [1, 256])
- WeightNorm layers are pre-folded during Python export
- Model loading verified: 454 weights, config, and voice packs
