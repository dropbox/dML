# Phase 5 Checklist: CosyVoice2 TTS Conversion

**Status**: EFFECTIVELY COMPLETE (10/11 tasks)
**Target**: 50-80 commits
**Actual**: 18 commits (#34-51)
**Remaining**: ONNX speaker encoder (blocked on Python 3.14 incompatibility)

---

## Model Information

| Aspect | Details |
|--------|---------|
| Model | CosyVoice2-0.5B (FunAudioLLM/CosyVoice2-0.5B) |
| Architecture | Flow-matching TTS with LLM backbone |
| Parameters | ~500M estimated |
| Format | PyTorch checkpoint (.pt) |
| Source | https://github.com/FunAudioLLM/CosyVoice |
| ModelScope | https://modelscope.cn/models/iic/CosyVoice2-0.5B |
| Target Accuracy | <1e-3 error |
| Target Performance | >=2.0x PyTorch |

---

## Architecture Overview

```
CosyVoice2 Architecture:

                    ┌─────────────────────────────┐
                    │      Text Input             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Speech Recognition       │
                    │   Tokenizer (ONNX)         │
                    │   - Text to supervised     │
                    │     semantic tokens        │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                     │
         ▼                                                     ▼
┌─────────────────────┐                        ┌─────────────────────┐
│   Qwen2 LLM         │                        │   Speaker Embedding │
│   (llm.pt)          │                        │   (voice cloning)   │
│   - Text encoder    │                        │   192-dim x-vector  │
│   - Token generator │                        └──────────┬──────────┘
│   - Autoregressive  │                                   │
└──────────┬──────────┘                                   │
           │                                              │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Flow Matching Model       │
                    │   (flow.pt)                 │
                    │   - MaskedDiffWithXvec      │
                    │   - Conditional generation  │
                    │   - Tokens → Mel spectrogram│
                    │   - 80-dim mel output       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   HiFi-GAN Vocoder          │
                    │   (hift.pt)                 │
                    │   - Mel → Waveform          │
                    │   - 22.05kHz audio          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Audio Waveform            │
                    │   (22.05kHz, mono)          │
                    └─────────────────────────────┘
```

---

## Component Breakdown

### 1. Speech Tokenizer (ONNX)
- Converts text to supervised semantic tokens
- Based on multilingual speech recognition model
- Vector quantization for discrete tokens
- Files: `speech_tokenizer_v2.onnx`, `campplus.onnx`

### 2. LLM Component (Qwen2-based)
- **Model**: Qwen2ForCausalLM pretrained
- **Classes**: Qwen2Encoder, Qwen2LM, TransformerLM
- **Function**: Text embedding + autoregressive token generation
- **Key methods**:
  - `inference()`: Main generation loop
  - `sampling_ids()`: Token sampling with top-k
  - Streaming support (unistream/bistream modes)
- **File**: `llm.pt`

### 3. Flow Matching Model
- **Classes**: MaskedDiffWithXvec, CausalMaskedDiffWithXvec
- **Architecture**:
  - Token vocab: 4096
  - Input embedding: 512
  - Speaker embedding (x-vector): 192
  - Output: 80-dim mel spectrogram
  - Decoder: 4 main blocks, 12 mid-blocks
  - Attention heads: 8
  - Channels: [256, 256]
- **Function**: Converts speech tokens to mel spectrograms
- **File**: `flow.pt`

### 4. HiFi-GAN Vocoder
- **Architecture**: Standard HiFi-GAN
- **Function**: Mel spectrogram to waveform
- **Output**: 22.05kHz audio
- **File**: `hift.pt`

---

## MLX Implementation Tasks

### Phase 5.1: Infrastructure & Model Download (3-5 commits)

- [x] Create `cosyvoice2_converter.py` scaffolding (commit #34)
- [x] Model download script (scripts/download_cosyvoice2.py) (commit #34)
- [x] CLI commands (cosyvoice2 convert/inspect/validate/list) (commit #34)
- [x] Unit tests (9 tests in test_cosyvoice2_converter.py) (commit #34)
- [x] Load and inspect model structure (commit #35)
- [x] Document weight mapping (commit #35) - reports/main/cosyvoice2_weight_mapping.md

### Phase 5.2: HiFi-GAN Vocoder (8-12 commits)

HiFi-GAN is well-documented and simpler - start here.

- [x] WeightNormConv1d (commit #36)
- [x] WeightNormConvTranspose1d (commit #36)
- [x] ResBlock1d with dilations (commit #36)
- [x] F0Predictor (commit #36)
- [x] SourceModule (commit #36)
- [x] HiFiGANVocoder generator class (commit #36)
- [x] Unit tests (10 tests in test_cosyvoice2_vocoder.py) (commit #36)
- [x] Weight loading from hift.pt (commit #37) - loads and runs
- [x] Validation vs PyTorch (commit #38) - 6/6 weight checks, forward pass validated

### Phase 5.3: Flow Matching Model (15-25 commits)

Core innovation of CosyVoice2.

- [x] Token embedding layer (vocab 6561 → 512) (commit #39)
- [x] Speaker embedding projection (192 → 80) (commit #39)
- [x] Pre-lookahead convolution layers (commit #39)
- [x] FlowEncoder (6 transformer layers) (commit #39)
- [x] Multi-head attention mechanism (8 heads) (commit #39)
- [x] Time embedding (sinusoidal + MLP) (commit #39)
- [x] UNet decoder blocks (down/mid/up) (commit #39)
- [x] MaskedDiffWithXvec class (commit #39)
- [x] Unit tests for flow model (25 tests) (commit #39)
- [x] DiT-style decoder blocks: DiTConvBlock, DiTAttentionBlock, DiTBlock (commit #40)
- [x] DiTDecoder: 1 down, 12 mid, 1 up blocks + final (commit #40)
- [x] Unit tests for DiT decoder (6 tests) (commit #40)
- [x] Weight loading from flow.pt (commit #41) - DiTDecoder loads 910 decoder keys
- [x] Validation vs PyTorch (commit #42) - DiTDecoder matches PyTorch (max error < 2e-6)
- [x] CausalMaskedDiffWithXvec (commit #43) - streaming variant with causal encoder, context preservation

### Phase 5.4: LLM Component (15-20 commits)

Qwen2-based text encoder and token generator.

- [x] Qwen2 config parsing (commit #44)
- [x] Qwen2Attention (GQA: 7 heads, 1 KV head) (commit #44)
- [x] Qwen2MLP (SwiGLU feedforward) (commit #44)
- [x] Qwen2DecoderLayer (commit #44)
- [x] Qwen2Model (24-layer transformer stack) (commit #44)
- [x] RoPE position embeddings (commit #44)
- [x] Qwen2RMSNorm (commit #44)
- [x] CosyVoice2LLM (llm_embedding, speech_embedding, decoder) (commit #44)
- [x] Unit tests for LLM (23 tests) (commit #44)
- [x] from_pretrained weight loading (commit #44)
- [x] Weight loading validation - 8/8 shape checks pass (commit #45)
- [x] PyTorch forward pass comparison - max error 1.8e-4 (float32 accumulation) (commit #46)
- [x] Sampling methods (top-k, top-p, temperature, repetition_penalty) (commit #46)
- [x] Streaming generation support - generate_speech_tokens_stream() (commit #46)

### Phase 5.5: Full Model & Integration (10-15 commits)

- [x] CosyVoice2Model main class (commit #47)
- [x] `from_pretrained()` weight loading - validated with all .pt files (commit #47)
- [x] Text tokenizer integration - Qwen2 BPE tokenizer (commit #48)
- [x] `synthesize_text()` method - direct text-to-speech (commit #48)
- [x] `synthesize_text_stream()` method - streaming text-to-speech (commit #48)
- [x] Speaker embedding placeholders - random_speaker_embedding(), zero_speaker_embedding() (commit #48)
- [ ] ONNX speaker encoder (requires onnxruntime - Python 3.14 incompatible)
- [x] CLI commands - convert, inspect, validate, synthesize, list (commit #50)
- [x] End-to-end validation - validate_cosyvoice2_e2e.py, RTF 0.04x (commit #49)
- [x] Performance benchmarks - benchmark_cosyvoice2.py, 26x real-time (commit #51)

---

## Key Implementation Challenges

### 1. Flow Matching in MLX
CosyVoice2 uses conditional flow matching - need to implement:
- ODE solver (Euler method or RK45)
- Noise scheduling (cosine scheduler)
- Masked generation for variable-length sequences

### 2. Qwen2 Model Porting
Large pretrained LLM component. Options:
- Port from scratch (most work, full control)
- Check if mlx-lm has Qwen2 support
- Use HuggingFace transformers as reference

### 3. ONNX Tokenizer
Speech tokenizer is in ONNX format. Options:
- Use ONNX runtime for tokenization
- Port ONNX model to MLX
- Extract and port underlying logic

### 4. Speaker Embeddings (X-vectors)
192-dimensional speaker embeddings for voice cloning.
- Need CAM++ model or equivalent
- May need to port speaker encoder too

### 5. Streaming Support
CosyVoice2 has streaming inference (~150ms latency).
- Requires causal attention variants
- Chunk-based processing
- State management for streaming

---

## Weight Mapping Reference

| PyTorch Key Pattern | Shape | MLX Target |
|---------------------|-------|------------|
| `llm.llm.model.embed_tokens.weight` | [vocab, dim] | `llm.embed_tokens` |
| `llm.llm.model.layers.*.self_attn.q_proj.weight` | [dim, dim] | `llm.layers.*.attention.q_proj` |
| `flow.encoder.*` | various | `flow.encoder.*` |
| `flow.decoder.*` | various | `flow.decoder.*` |
| `hift.generator.*` | various | `vocoder.generator.*` |

---

## Validation Plan

### Component Validation
1. Vocoder (HiFi-GAN): mel → audio, max error < 1e-3
2. Flow model: tokens → mel, max error < 1e-3
3. LLM: text → tokens, exact token match
4. Full pipeline: audible quality match

### Audio Quality Validation
1. Generate same text with same speaker
2. Compare spectrograms (mel distance)
3. Perceptual metrics if available (PESQ/STOI)

---

## Benchmark Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Audio generation speed | >= 2.0x real-time | 1s audio in < 0.5s |
| First-token latency | < 200ms | Streaming mode |
| PyTorch comparison | >= 2.0x speedup | vs PyTorch MPS |
| Memory usage | <= 2GB | Model weights |

---

## Files to Create

```
tools/pytorch_to_mlx/converters/
├── models/
│   ├── cosyvoice2.py          # Main MLX model implementation
│   ├── cosyvoice2_flow.py     # Flow matching components
│   ├── cosyvoice2_llm.py      # Qwen2 LLM components
│   └── cosyvoice2_vocoder.py  # HiFi-GAN vocoder
├── cosyvoice2_converter.py    # Main converter class
tests/
├── test_cosyvoice2_model.py   # Unit tests
├── test_cosyvoice2_converter.py # Integration tests
scripts/
├── download_cosyvoice2.py     # Model download helper
├── validate_cosyvoice2_e2e.py # E2E validation script
```

---

## Dependencies

- MLX >= 0.20.0
- torch >= 2.0.0 (for reference/validation)
- transformers (for Qwen2 reference)
- onnxruntime (for speech tokenizer)
- huggingface_hub (for downloading)
- soundfile (for audio I/O)
- numpy (for audio processing)

---

## References

- CosyVoice GitHub: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice Paper: https://arxiv.org/abs/2407.05407
- ModelScope: https://modelscope.cn/models/iic/CosyVoice2-0.5B
- Flow Matching Paper: https://arxiv.org/abs/2210.02747
- HiFi-GAN: https://arxiv.org/abs/2010.05646
- Qwen2: https://github.com/QwenLM/Qwen2
- MLX Documentation: https://ml-explore.github.io/mlx/

---

## Progress Log

| Commit | Task | Status |
|--------|------|--------|
| #34 | Phase 5 init: checklist, converter scaffolding, CLI, 9 tests | Complete |
| #35 | Model inspection: weight mapping, 775M params documented | Complete |
| #36 | HiFi-GAN vocoder: WeightNormConv1d, ResBlock1d, Generator, 10 tests | Complete |
| #37 | HiFi-GAN weight loading: corrected architecture, loads hift.pt | Complete |
| #38 | Phase 5.2 COMPLETE: vocoder validation, 6/6 weight checks pass | Complete |
| #39 | Phase 5.3 BEGIN: Flow model scaffolding, encoder, decoder, 25 tests | Complete |
| #40 | DiT decoder: DiTConvBlock, DiTAttentionBlock, DiTDecoder, 6 tests | Complete |
| #41 | DiT decoder weight loading: from_pretrained, _load_weights, validation | Complete |
| #42 | DiT decoder PyTorch validation: sinusoidal, Mish, GroupNorm fixes | Complete |
| #43 | Phase 5.3 COMPLETE: CausalMaskedDiffWithXvec streaming variant | Complete |
| #44 | Phase 5.4 BEGIN: Qwen2 LLM implementation, 10/13 tasks, 23 tests | Complete |
| #45 | LLM weight loading validated - 8/8 shape checks, forward pass works | Complete |
| #46 | Phase 5.4 COMPLETE - PyTorch comparison, sampling, streaming | Complete |
| #47 | Phase 5.5 BEGIN - CosyVoice2Model main class, from_pretrained, 10 tests | Complete |
| #48 | Phase 5.5 tokenizer - Qwen2 tokenizer, synthesize_text, 19+5 tests | Complete |
| #49 | E2E validation - script, Linear weight fix, pipeline working | Complete |
| #50 | CLI synthesize command - text-to-speech via CLI | Complete |
| #51 | Performance benchmarks - 26x real-time throughput | Complete |
