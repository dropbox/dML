# STT SOTA Literature Review - V2 Additions
**Date:** 2026-01-02
**Purpose:** Second-pass additions to ensure absolute SOTA coverage
**New Papers:** 50+ additional papers across 8 new categories

---

## Executive Summary - NEW CRITICAL FINDINGS

This second pass uncovered several breakthrough papers that significantly impact the BEYOND GOD TIER architecture:

### NEW Top-Priority Discoveries

| Paper | arXiv | Key Finding | Impact |
|-------|-------|-------------|--------|
| **Samba-ASR** | 2501.02832 | First SOTA Mamba-based ASR | Alternative to Transformer |
| **Transducer-Llama** | 2412.16464 | 17% WER reduction with LLM | Streaming + LLM |
| **Whale** | 2506.01439 | 2.4% WER LibriSpeech test-clean | Best reported WER |
| **XEUS** | - | New SUPERB record | Beats WavLM Large |
| **Canary-1B-v2** | 2509.14128 | 1.7M training hours, 10x faster than Whisper | NVIDIA SOTA |
| **DistilWhisper** | 2405.00966 | 12x smaller, 8x faster | Compression |
| **FLASepformer** | 2508.19528 | 2.29x speed, 15.8% GPU memory | Efficient separation |
| **Continuous-Token Diffusion** | 2510.12995 | 1.95% WER | Novel approach |

---

## NEW Category 13: Neural Transducers (RNN-T)

| Paper | arXiv | Key Contribution | Metrics |
|-------|-------|-----------------|---------|
| **Transducer-Llama** | 2412.16464 | LLM integration with streaming transducer | 17% WER reduction |
| WST | 2511.04035 | Weakly supervised transducer | Works with 70% transcript errors |
| HAINAN | 2410.02597 | Hybrid autoregressive transducer | AR + NAR support |
| CUSIDE-T | 2407.10255 | Streaming transducer improvements | Lower latency |
| Fast Context-Biasing | 2406.07096 | CTC + Transducer biasing | Better hotwords |

**Impact on GOD TIER:** Transducer-Llama shows streaming ASR can benefit from LLM integration without sacrificing latency. Consider adding LLM fusion to decoder path.

---

## NEW Category 14: Mamba Architecture

| Paper | arXiv | Key Contribution | vs Transformer |
|-------|-------|-----------------|----------------|
| **Samba-ASR** | 2501.02832 | First SOTA Mamba ASR | Superior accuracy |
| MLMA | 2510.18684 | Multilingual Mamba ASR | Competitive |
| Speech-Mamba | 2409.18654 | Long-context Mamba | Near-linear scaling |
| Mamba Streaming | 2410.00070 | Streaming Mamba ASR | Lookahead mechanism |

**Impact on GOD TIER:** Mamba offers O(n) complexity vs O(n^2) for transformers. Could replace attention in encoder for latency reduction.

---

## NEW Category 15: Efficient Transformers

| Paper | arXiv | Key Contribution | Speedup |
|-------|-------|-----------------|---------|
| **FLASepformer** | 2508.19528 | Focused linear attention separator | 2.29x, 15.8% memory |
| IMSE | 2511.14515 | Inception linear attention | 16.8% param reduction |
| Multi-head TLA | 2505.13544 | Temporal latent attention | 5.3x, 8.3x memory |
| NAR Decoders | 2511.09084 | Non-autoregressive for Conformer | 2.31x decoding |
| Magnitude Rectified | 2507.00698 | Better linear attention | Improved efficiency |

**Impact on GOD TIER:** FLASepformer could replace MossFormer2 with significant efficiency gains. Multi-head TLA could accelerate encoder.

---

## NEW Category 16: Audio Tokenization & Codecs

| Paper | arXiv | Key Contribution | Bitrate |
|-------|-------|-----------------|---------|
| Semantic Codebooks | 2512.21653 | HuBERT semantic tokens | Lower bitrate |
| SACodec | 2512.20944 | Asymmetric quantization | 1.5 kbps |
| XY-Tokenizer | 2506.23325 | Semantic-acoustic balance | Multi-task |
| JEPA Tokenizer | 2507.08530 | FSQ tokenization | Density-adaptive |
| Q2D2 | 2512.01537 | 2D quantization | Better compression |

**Impact on GOD TIER:** Audio tokenization could enable discrete speech processing. SACodec's 1.5 kbps is remarkable for potential edge deployment.

---

## NEW Category 17: LLM-Based ASR

| Paper | arXiv | Key Contribution | Application |
|-------|-------|-----------------|-------------|
| Contextual Biasing LLM | 2512.21828 | Hotword retrieval + RL | Better entity recognition |
| Speech Foundation + LLM | 2510.22961 | Unified modalities | End-to-end |
| **Omnilingual ASR** | 2511.09690 | 1600+ languages | Zero-shot |
| **Qwen2-Audio** | 2407.10759 | Voice chat + audio analysis | Multimodal |
| **Qwen3-Omni** | 2509.17765 | 19 languages, unified | Full multimodal |
| SLM-TTA | 2512.24739 | TTA for speech LLMs | Inference adaptation |

**Impact on GOD TIER:** Qwen2-Audio and Omnilingual show LLM integration can dramatically improve multilingual and zero-shot capabilities.

---

## NEW Category 18: Architecture Innovations

### E-Branchformer & FastConformer

| Paper | arXiv | Key Contribution | WER |
|-------|-------|-----------------|-----|
| **Whale** | 2506.01439 | E-Branchformer, large-scale | 2.4% LibriSpeech clean |
| **Canary-1B-v2** | 2509.14128 | FastConformer, 1.7M hours | 10x faster than Whisper |
| NEST | 2408.13106 | Self-supervised FastConformer | SSL framework |
| Multi-Convformer | 2407.03718 | Multi-conv alternative | 8% WER improvement |
| Splitformer | 2506.18035 | Parallel layer processing | Early-exit |

### Zipformer

| Paper | arXiv | Key Contribution | Metrics |
|-------|-------|-----------------|---------|
| k2SSL | 2411.17100 | SSL with Zipformer | 34.8% WER reduction vs HuBERT |
| XLSR-Transducer | 2407.04439 | Cross-lingual Zipformer | 4% abs WER improvement |

**Impact on GOD TIER:** Whale's 2.4% WER is current SOTA. E-Branchformer should replace standard transformer blocks in encoder.

---

## NEW Category 19: Distillation & Compression

| Paper | arXiv | Key Contribution | Compression |
|-------|-------|-----------------|-------------|
| **DistilWhisper** | 2405.00966 | Efficient distillation | 12x smaller, 8x faster |
| Multilingual DistilWhisper | 2311.01070 | Language-specific experts | Multi-language |
| Formatted KD | 2512.18967 | Punctuation-aware | Better formatting |
| Listen Like Teacher | 2511.14219 | Anti-hallucination KD | Reliability |

**Impact on GOD TIER:** DistilWhisper approach could create efficient inference variant for edge deployment.

---

## NEW Category 20: Emotion & Paralinguistics

| Paper | arXiv | Key Contribution | Accuracy |
|-------|-------|-----------------|----------|
| Emotion Graphs | 2509.25458 | Zero-shot SER with graphs | Improved zero-shot |
| HYFuse | 2506.03403 | Hyperbolic space fusion | Top performance |
| PARROT | 2506.01138 | Mamba + attention fusion | SOTA SER |
| Metadata-Enhanced | 2412.20707 | Two-stage fine-tuning | IEMOCAP improvement |
| Channel-Attention | 2412.10011 | CNN-BiLSTM lightweight | Up to 99.65% |

**Impact on GOD TIER:** PARROT and HYFuse could improve emotion head accuracy. Current architecture uses 8-class, could expand.

---

## NEW Category 21: Word Timestamps & Alignment

| Paper | arXiv | Key Contribution | Precision |
|-------|-------|-----------------|-----------|
| NeMo Timestamps | 2505.15646 | <timestamp> token training | 80-90%, 20-120ms |
| Label Priors CTC | 2406.02560 | Less peaky CTC | 12-40% boundary improvement |
| WhisperX | 2303.00747 | VAD + forced alignment | Long-form support |

**Impact on GOD TIER:** NeMo timestamp approach could provide better word-level timing than current CTC alignment.

---

## NEW Category 22: Noise Robustness

| Paper | arXiv | Key Contribution | Improvement |
|-------|-------|-----------------|-------------|
| **Denoising Hurts** | 2512.17562 | Enhancement degrades ASR | CRITICAL WARNING |
| Error Noise Embedding | 2512.17247 | Noise-aware embedding | 31.1% → 24.8% WER |
| NoisyD-CT | 2509.01087 | Tri-stage training | 25.7% WER reduction |
| Lightweight FE | 2509.21833 | 66% overhead reduction | Efficient |
| MAGE | 2509.19881 | Coarse-to-fine masking | Improved SE |

**Impact on GOD TIER:** "Denoising Hurts" paper is CRITICAL. Our adaptive cleaning must be validated against modern ASR models.

---

## Updated Breakthrough Rankings

### Tier 0: CRITICAL (Must Implement)

1. **E-BATS** (2506.07078) - Replace SUTA immediately
2. **CR-CTC** (2410.05101) - Add to CTC training
3. **DELULU** (2510.17662) - Replace ECAPA-TDNN
4. **FLASepformer** (2508.19528) - Consider replacing MossFormer2
5. **E-Branchformer** (Whale, 2506.01439) - Replace transformer blocks

### Tier 1: HIGH PRIORITY (Near-term)

6. **Transducer-Llama** (2412.16464) - LLM fusion for decoder
7. **MAS-LoRA** (2505.20006) - Replace single LoRA
8. **Samba-ASR** (2501.02832) - Mamba encoder alternative
9. **DistilWhisper** (2405.00966) - Compression approach
10. **JEDIS-LLM** (2511.16046) - Joint ASR+diarization

### Tier 2: IMPORTANT (Medium-term)

11. SPEAR (2510.25955) - SSL features
12. Listen Like Teacher (2511.14219) - Hallucination mitigation
13. Canary-1B-v2 (2509.14128) - FastConformer reference
14. LibriVAD (2512.17281) - ViT-based VAD
15. NeMo Timestamps (2505.15646) - Word timing

---

## Updated Performance Targets

Based on second-pass findings, revised targets for BEYOND GOD TIER:

| Metric | GOD TIER | BEYOND GOD TIER V2 |
|--------|----------|-------------------|
| LibriSpeech clean WER | Not specified | <2.5% (Whale achieves 2.4%) |
| LibriSpeech other WER | Not specified | <5% |
| CTC latency (clean) | <80ms | <50ms (with Mamba/linear attn) |
| CTC latency (noisy) | <140ms | <80ms |
| Speaker EER | <1.0% | <0.4% |
| Separation SI-SDRi | 18-21dB | >20dB (FLASepformer) |
| Model compression | None | 8-12x (DistilWhisper approach) |
| Streaming latency | <300ms | <200ms (Transducer-Llama) |

---

## Architecture Upgrade Recommendations V2

### Encoder Path
```
Current: Whisper Encoder (frozen) + Speaker Query Attention
Upgrade: E-Branchformer OR Samba-ASR encoder + Speaker Query Attention
Benefit: 25% faster (E-Branchformer) OR O(n) complexity (Mamba)
```

### Source Separation
```
Current: MossFormer2
Upgrade: FLASepformer
Benefit: 2.29x speedup, 15.8% memory, similar SI-SDRi
```

### Test-Time Adaptation
```
Current: SUTA (backprop, ~10ms)
Upgrade: E-BATS (backprop-free)
Benefit: Faster, multi-scale loss, EMA stability
```

### Speaker Embeddings
```
Current: ECAPA-TDNN (0.64% EER)
Upgrade: DELULU-style SSL
Benefit: 62% relative EER improvement
```

### LoRA System
```
Current: Single LoRA per speaker (Phase 10A), MoE-LoRA (Phase 10B)
Upgrade: MAS-LoRA with accent-specific experts immediately
Benefit: Better handling of speaker variation
```

### Decoder Path
```
Current: Whisper decoder + vocabulary boost + LoRA
Upgrade: Transducer-Llama style LLM fusion
Benefit: 17% WER reduction, maintains streaming
```

### CTC Training
```
Current: Standard CTC
Upgrade: CR-CTC + LLM-CTC intermediate loss
Benefit: Consistency regularization + better linguistic modeling
```

---

## Additional Papers to Download

```bash
# New papers from second pass
wget -nc -P pdfs https://arxiv.org/pdf/2501.02832.pdf  # Samba-ASR
wget -nc -P pdfs https://arxiv.org/pdf/2412.16464.pdf  # Transducer-Llama
wget -nc -P pdfs https://arxiv.org/pdf/2506.01439.pdf  # Whale
wget -nc -P pdfs https://arxiv.org/pdf/2509.14128.pdf  # Canary-1B-v2
wget -nc -P pdfs https://arxiv.org/pdf/2405.00966.pdf  # DistilWhisper
wget -nc -P pdfs https://arxiv.org/pdf/2508.19528.pdf  # FLASepformer
wget -nc -P pdfs https://arxiv.org/pdf/2510.12995.pdf  # Continuous-Token Diffusion
wget -nc -P pdfs https://arxiv.org/pdf/2511.04035.pdf  # WST
wget -nc -P pdfs https://arxiv.org/pdf/2410.02597.pdf  # HAINAN
wget -nc -P pdfs https://arxiv.org/pdf/2407.10255.pdf  # CUSIDE-T
wget -nc -P pdfs https://arxiv.org/pdf/2510.18684.pdf  # MLMA
wget -nc -P pdfs https://arxiv.org/pdf/2409.18654.pdf  # Speech-Mamba
wget -nc -P pdfs https://arxiv.org/pdf/2410.00070.pdf  # Mamba Streaming
wget -nc -P pdfs https://arxiv.org/pdf/2511.14515.pdf  # IMSE
wget -nc -P pdfs https://arxiv.org/pdf/2505.13544.pdf  # Multi-head TLA
wget -nc -P pdfs https://arxiv.org/pdf/2511.09084.pdf  # NAR Decoders
wget -nc -P pdfs https://arxiv.org/pdf/2512.21653.pdf  # Semantic Codebooks
wget -nc -P pdfs https://arxiv.org/pdf/2512.20944.pdf  # SACodec
wget -nc -P pdfs https://arxiv.org/pdf/2506.23325.pdf  # XY-Tokenizer
wget -nc -P pdfs https://arxiv.org/pdf/2512.21828.pdf  # Contextual Biasing LLM
wget -nc -P pdfs https://arxiv.org/pdf/2511.09690.pdf  # Omnilingual ASR
wget -nc -P pdfs https://arxiv.org/pdf/2407.10759.pdf  # Qwen2-Audio
wget -nc -P pdfs https://arxiv.org/pdf/2408.13106.pdf  # NEST
wget -nc -P pdfs https://arxiv.org/pdf/2407.03718.pdf  # Multi-Convformer
wget -nc -P pdfs https://arxiv.org/pdf/2506.18035.pdf  # Splitformer
wget -nc -P pdfs https://arxiv.org/pdf/2411.17100.pdf  # k2SSL
wget -nc -P pdfs https://arxiv.org/pdf/2407.04439.pdf  # XLSR-Transducer
wget -nc -P pdfs https://arxiv.org/pdf/2311.01070.pdf  # Multilingual DistilWhisper
wget -nc -P pdfs https://arxiv.org/pdf/2509.25458.pdf  # Emotion Graphs
wget -nc -P pdfs https://arxiv.org/pdf/2506.03403.pdf  # HYFuse
wget -nc -P pdfs https://arxiv.org/pdf/2506.01138.pdf  # PARROT
wget -nc -P pdfs https://arxiv.org/pdf/2505.15646.pdf  # NeMo Timestamps
wget -nc -P pdfs https://arxiv.org/pdf/2512.17247.pdf  # Error Noise Embedding
```

---

*V2 additions generated: 2026-01-02*
*New papers: 50+ across 10 new categories*
*Total papers indexed: 130+*
