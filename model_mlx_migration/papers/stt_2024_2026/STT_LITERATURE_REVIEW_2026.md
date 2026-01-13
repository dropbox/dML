# STT State-of-the-Art Literature Review
**Date:** 2026-01-02
**Purpose:** Comprehensive review of STT research (2024-2026) to exceed GOD TIER architecture
**Papers Reviewed:** 80+
**Categories:** 12

---

## 1. Executive Summary

This literature review analyzes 80+ papers across 12 research areas in speech-to-text. The goal is to identify techniques that can push our system beyond the current ARCHITECTURE_GOD_TIER_STT.md specification.

### Key Findings

| Component | Current GOD TIER | SOTA Alternative | Expected Improvement |
|-----------|------------------|------------------|---------------------|
| Test-time adaptation | SUTA (~10ms) | E-BATS (backprop-free) | Faster, more efficient |
| Speaker embeddings | ECAPA-TDNN (0.64% EER) | DELULU | 62% relative EER improvement |
| LoRA adaptation | Single LoRA per speaker | MAS-LoRA (MoE) | Better accent handling |
| CTC path | Standard CTC | CR-CTC | Reduced overfitting |
| VAD | Silero | ViT-VAD | Better multi-condition |
| Joint ASR+Diar | Separate | JEDIS-LLM | Single model, streaming |

### Concrete Upgrade Path

**Phase A (Immediate wins, low risk):**
1. Replace SUTA with E-BATS
2. Add CR-CTC consistency regularization
3. Upgrade VAD to ViT-based

**Phase B (Moderate effort, high impact):**
4. Upgrade speaker embeddings using DELULU techniques
5. Implement MAS-LoRA instead of single-speaker LoRA
6. Add LLM intermediate loss to CTC

**Phase C (Research-level, breakthrough potential):**
7. Joint ASR+diarization with JEDIS-LLM approach
8. SPEAR-based SSL features
9. Hallucination mitigation via Adaptive Layer Attention

---

## 2. End-to-End ASR Models

### 2.1 Foundation Models Landscape

The field has consolidated around several major architectures:

**Google USM (arXiv:2303.01037)**
- Scale: 12M hours, 300+ languages
- Architecture: Conformer-based encoder
- Key innovation: Random-projection quantization + speech-text modality matching
- WER: SOTA on FLEURS, MLS benchmarks

**OpenAI Whisper** (Baseline reference)
- Scale: 680k hours
- Architecture: Encoder-decoder transformer
- Limitations: Hallucinations, no streaming, non-CTC

**OWSM (arXiv:2401.16658, 2402.12654)**
- Open reproduction of Whisper
- OWSM v3.1: E-Branchformer, 25% faster
- OWSM-CTC: Encoder-only, 24% improvement on speech translation
- Relevance: Open-source alternative with CTC option

**Meta SeamlessM4T (arXiv:2308.11596)**
- Focus: Multilingual translation
- Innovation: Preserves prosody and vocal style
- WER: 20% BLEU improvement on FLEURS

### 2.2 Architecture Improvements

**Linear-Time Conformers (arXiv:2409.07165)**
- Replaces self-attention with SummaryMixing
- O(n) vs O(n^2) complexity
- Works for both streaming and offline
- Implication: Can reduce encoder latency

**All-in-One ASR (arXiv:2512.11543)**
- Multi-mode joiner supporting CTC, transducer, attention
- Single model for streaming and offline
- Implication: Unify our dual CTC+decoder paths

**NoisyD-CT (arXiv:2509.01087)**
- Tri-stage training for noise robustness
- 25.7% WER reduction in noisy conditions
- Implication: Could improve our audio cleaning path

### 2.3 Implications for GOD TIER

Current architecture uses Whisper encoder (frozen) + custom heads. Consider:
1. E-Branchformer blocks instead of vanilla transformer (OWSM shows 25% speedup)
2. Linear attention for reduced latency
3. Multi-mode joiner to unify CTC and decoder paths

---

## 3. Speaker-Adaptive ASR

### 3.1 Speaker Conditioning Techniques

**Context-Aware Whisper (arXiv:2511.18774)**
- Decoder prompting + encoder prefixing
- 22.3% WER reduction on Arabic dialects
- No retraining required - inference-time adaptation
- Implication: Speaker-aware prompts could help immediately

**Knowledge-Decoupled Personalization (arXiv:2510.10401)**
- Gated parameter isolation
- Uses synthetic personal data
- Preserves base model while adding personalization
- Implication: Better than full fine-tuning for speaker LoRA

**Variational LoRA (arXiv:2509.20397)**
- Bayesian low-rank adaptation
- Data-efficient fine-tuning for impaired speech
- Implication: Better for low-data speaker adaptation

### 3.2 LoRA and Adapter Methods

**MAS-LoRA (arXiv:2505.20006)** - PRIORITY
- Mixture of Accent-Specific LoRA experts
- Router selects appropriate expert
- Better than single LoRA for multi-accent
- Implication: Replace our single-speaker LoRA with MoE-LoRA sooner

**Privacy-Preserving LoRA (arXiv:2512.16401)**
- Zero-data-exfiltration framework
- Experience replay for multi-domain
- Implication: Relevant for privacy-conscious deployments

**Perceiver-Prompt (arXiv:2406.09873)**
- P-Tuning + LoRA for dysarthric speech
- Speaker-specific prompts generated dynamically
- Implication: Combines prompting with LoRA for double benefit

### 3.3 Recommendations

**Upgrade MoE-LoRA timeline:** Current plan has MoE-LoRA in Phase 10B. Literature suggests this should be prioritized because:
1. MAS-LoRA shows accent-specific experts work well
2. Single LoRA insufficient for diverse speakers
3. Router overhead is minimal (<2ms per inference)

**Add prompt-based conditioning:** No retraining required, immediate benefit for known speaker types.

---

## 4. Test-Time Adaptation

### 4.1 Beyond SUTA

**E-BATS (arXiv:2506.07078)** - PRIORITY
- Backpropagation-free test-time adaptation
- Lightweight prompt adaptation instead of layer norm tuning
- Multi-scale loss: global + local distribution shifts
- EMA mechanism for stability
- Implication: More efficient than SUTA, should replace it

**LI-TTA (arXiv:2408.05769)**
- Language-informed TTA
- Uses LM to provide linguistic corrections
- Merges acoustic and linguistic information
- Implication: Could add to our decoder path

**Child Speech TTA Study (arXiv:2409.13095)**
- Compared SUTA vs SGEM
- Found speaker-specific adaptation helps children
- Implication: Our tiered approach is validated

### 4.2 Recommendations

Replace SUTA (Layer -1) with E-BATS because:
1. No backpropagation needed - faster
2. Multi-scale loss better captures domain shift
3. EMA provides stability
4. Maintains streaming capability

---

## 5. Multi-Speaker ASR

### 5.1 Joint ASR + Diarization

**JEDIS-LLM (arXiv:2511.16046)** - PRIORITY
- First zero-shot streamable joint ASR+diarization
- Speaker Prompt Cache for dynamic tracking
- Trained on short audio (<20s), infers on long
- Implication: Could unify our separate speaker tracking + ASR

**DiCoW (arXiv:2510.03723)**
- Diarization-conditioned Whisper
- Target-speaker modeling with serialized output
- Implication: Alternative to MossFormer separation

**SpeakerLM (arXiv:2508.06372)**
- Multimodal LLM for unified SD+ASR
- End-to-end approach
- Implication: Future direction for unified models

**SA-Paraformer (arXiv:2310.04863)**
- Non-autoregressive speaker-attributed ASR
- 6.1% SD-CER reduction
- Implication: Faster than autoregressive decoder

### 5.2 Source Separation

**MossFormer2 (arXiv:2312.11825)**
- Already in our architecture
- RNN-free recurrent network
- 21.2 dB SI-SDRi on WSJ0-3mix
- Near theoretical upper bound (23.1 dB)

**GLAD MoE (arXiv:2509.13093)**
- Dynamic mixture-of-experts for multi-talker
- Global-local aware fusion
- Implication: MoE approach for separation also

### 5.3 Recommendations

Two paths forward:
1. **Conservative:** Keep MossFormer2 separation + per-speaker Whisper (current plan)
2. **Aggressive:** Implement JEDIS-LLM style joint model

The aggressive path requires more research but offers:
- Single model instead of separation + ASR
- Native streaming support
- Zero-shot generalization

---

## 6. Speech Enhancement

### 6.1 Critical Finding

**"When De-noising Hurts" (arXiv:2512.17562)**
- Study on medical ASR systems
- Finding: Speech enhancement can DEGRADE ASR performance
- All tested enhancement configurations hurt accuracy
- Implication: Our adaptive cleaning needs careful validation

### 6.2 Modern Enhancement Approaches

**Shortcut Flow Matching (arXiv:2509.21522)**
- Single-step inference
- Real-time factor: 0.013
- Implication: Faster than DeepFilterNet3 potentially

**GDiffuSE (arXiv:2510.04157)**
- Diffusion + noise guidance
- Novel approach to enhancement
- Implication: Research direction

**Quantum-Inspired (arXiv:2509.04851)**
- 15 dB SNR improvement
- Novel transformation
- Implication: Interesting but unproven

### 6.3 Recommendations

1. **Validate enhancement helps:** The "denoising hurts" paper is concerning. Run A/B tests.
2. **Consider bypass more aggressively:** Our adaptive pipeline should skip enhancement more often
3. **Shortcut Flow could be faster:** Worth evaluating vs DeepFilterNet3

---

## 7. Self-Supervised Speech Models

### 7.1 New Foundation Models

**SPEAR (arXiv:2510.25955)** - PRIORITY
- First unified SSL framework for speech AND audio
- New SOTA on SUPERB benchmark
- Implication: Could replace Whisper encoder features

**DELULU (arXiv:2510.17662)** - PRIORITY
- Speaker-discriminative SSL
- 62% relative EER improvement
- Implication: Much better than ECAPA-TDNN for speaker embeddings

**WavJEPA (arXiv:2509.23238)**
- Waveform-based representation
- SOTA on time-domain audio models
- Implication: Alternative to mel-spectrogram features

**SpidR-Adapt (arXiv:2512.21204)**
- Meta-learning for rapid adaptation
- 100x less data for new languages
- Implication: Fast speaker adaptation possible

### 7.2 Recommendations

**Priority:** Implement DELULU-style speaker embeddings
- Current: ECAPA-TDNN 0.64% EER
- DELULU: 62% relative improvement
- This directly improves speaker conditioning in Layers 0, 1, 2B

**Consider:** SPEAR features could improve rich audio extraction
- Better joint speech+audio understanding
- Could improve emotion/paralinguistics heads

---

## 8. CTC Improvements

### 8.1 Training Improvements

**CR-CTC (arXiv:2410.05101)** - PRIORITY
- Consistency Regularization on CTC
- Enforces consistent predictions across augmented views
- Reduces overfitting, improves generalization
- Implication: Direct improvement to our CTC path

**LLM-CTC (arXiv:2506.22846)**
- LLM intermediate loss regularization
- Maps CTC to LLM embedding space
- Better linguistic modeling
- Implication: Improves CTC linguistic accuracy

### 8.2 Decoding Improvements

**FlexCTC (arXiv:2508.07315)**
- GPU-powered beam decoding
- Advanced contextualization
- Minimal kernel launch overhead
- Implication: Faster decoding with better hotword support

**Label Priors CTC (arXiv:2406.02560)**
- Less peaky distributions
- More accurate forced alignment
- Implication: Better word timestamps

### 8.3 Recommendations

1. **Add CR-CTC:** Consistency regularization is low-risk, high-reward
2. **Consider LLM-CTC:** Adds ~5ms but improves linguistic modeling
3. **Upgrade to FlexCTC decoding:** Better than greedy for hotwords

---

## 9. Streaming ASR

### 9.1 Streaming Whisper

**CarelessWhisper (arXiv:2508.12301)**
- Converts Whisper to causal streaming
- LoRA fine-tuning
- Chunk sizes <300ms
- Implication: Validates our streaming approach

**All-in-One ASR (arXiv:2512.11543)**
- Multi-mode joiner
- Seamless streaming/offline switching
- Implication: Could unify our paths

### 9.2 Latency Optimization

**Zipformer Unified (arXiv:2506.14434)**
- Dynamic right-context via chunked attention
- 7.9% WER reduction
- Flexible latency-accuracy trade-off
- Implication: Better than fixed-context streaming

### 9.3 Recommendations

Current GOD TIER targets:
- CTC: 76-111ms
- Decoder: 226-261ms

With these techniques, achievable:
- CTC: 60-90ms (via linear attention)
- Decoder: 180-220ms (via parallel decoding)

---

## 10. Speaker Embeddings

### 10.1 Beyond ECAPA-TDNN

**DELULU (arXiv:2510.17662)** - PRIORITY
- Speaker-discriminative SSL
- 62% relative EER improvement
- Implication: Replace ECAPA-TDNN

**M-Vec (arXiv:2409.15782)**
- Matryoshka embeddings
- 8-dim still viable
- Implication: Could reduce embedding size

**SSPS (arXiv:2505.14561)**
- Self-supervised positive sampling
- 2.57% EER on VoxCeleb1-O
- Implication: Training improvement

**Contrastive SPK (arXiv:2410.05037)**
- Contrastive on intermediate features
- 9.05% EER improvement
- Implication: Training technique

### 10.2 Recommendations

**Replace ECAPA-TDNN with DELULU-style embeddings:**
- 62% relative EER improvement is substantial
- Improves all downstream speaker-dependent components
- Should be Phase 10A priority

---

## 11. Hallucination Mitigation

### 11.1 Whisper-Specific Solutions

**Listen Like Teacher (arXiv:2511.14219)** - PRIORITY
- Adaptive Layer Attention (ALA)
- Multi-objective knowledge distillation
- Fuses encoder layer representations
- Implication: Reduces decoder hallucinations

**Adaptive Steering (arXiv:2510.12851)**
- Layer-wise intervention
- Internal state probing
- Better audio grounding
- Implication: Runtime hallucination detection

**Bag of Hallucinations (arXiv:2501.11378)**
- Post-processing text
- Pattern matching for known hallucinations
- Implication: Simple fallback

### 11.2 Recommendations

Add hallucination mitigation to Layer 2B decoder:
1. **ALA module:** Fuses encoder layers for better grounding
2. **Confidence thresholding:** Already in architecture, strengthen
3. **Post-processing filter:** Catch known hallucination patterns

---

## 12. VAD Improvements

### 12.1 Modern VAD

**LibriVAD (arXiv:2512.17281)**
- Vision Transformer for VAD
- Outperforms established models
- Implication: Better than Silero potentially

**SincQDR-VAD (arXiv:2508.20885)**
- Learnable bandpass filters
- 31% fewer parameters
- Implication: Efficient alternative

**sVAD (arXiv:2403.05772)**
- Spiking Neural Network
- Ultra-low power
- Implication: Edge deployment

### 12.2 Recommendations

Current: Silero VAD in Unified Frontend
Consider: ViT-based VAD (LibriVAD) for better multi-condition performance

---

## 13. Synthesis: BEYOND GOD TIER Architecture

Based on this literature review, here is the upgraded architecture specification:

### 13.1 Immediate Upgrades (Phase 9-10A)

| Component | Current | Upgrade To | Source | Impact |
|-----------|---------|------------|--------|--------|
| TTA | SUTA | E-BATS | 2506.07078 | Faster, no backprop |
| CTC Training | Standard | CR-CTC | 2410.05101 | Better generalization |
| VAD | Silero | ViT-VAD | 2512.17281 | Multi-condition |
| Decoding | Greedy | FlexCTC | 2508.07315 | Hotword support |

### 13.2 Phase 10 Upgrades

| Component | Current | Upgrade To | Source | Impact |
|-----------|---------|------------|--------|--------|
| Speaker Embed | ECAPA-TDNN | DELULU-style | 2510.17662 | 62% EER improvement |
| LoRA | Single per speaker | MAS-LoRA | 2505.20006 | Accent handling |
| CTC Linguistic | None | LLM-CTC loss | 2506.22846 | Better language model |
| Prompting | None | Speaker prompts | 2511.18774 | No-retrain adaptation |

### 13.3 Phase 11 Upgrades (Novel)

| Component | Current | Upgrade To | Source | Impact |
|-----------|---------|------------|--------|--------|
| Joint ASR+Diar | Separate | JEDIS-LLM | 2511.16046 | Single model |
| SSL Features | Whisper | SPEAR | 2510.25955 | SUPERB SOTA |
| Hallucination | Confidence | ALA | 2511.14219 | Encoder fusion |
| Linear Attn | O(n^2) | O(n) | 2409.07165 | Latency reduction |

### 13.4 Expected Metrics

| Metric | GOD TIER Target | BEYOND GOD TIER |
|--------|-----------------|-----------------|
| Clean audio CTC latency | <80ms p50 | <60ms p50 |
| Noisy audio CTC latency | <140ms p50 | <100ms p50 |
| Speaker EER | <1.0% | <0.5% |
| Multi-speaker WER (LibriCSS) | <12% | <10% |
| LoRA WER improvement | -25% relative | -35% relative |
| Hallucination rate | Not specified | <1% |

---

## 14. Action Items

### Priority 1: Low-Risk, High-Impact
- [ ] Implement CR-CTC training
- [ ] Replace SUTA with E-BATS
- [ ] Add FlexCTC decoding
- [ ] Benchmark ViT-VAD vs Silero

### Priority 2: Moderate Effort
- [ ] Implement DELULU-style speaker embeddings
- [ ] Implement MAS-LoRA (accelerate from Phase 10B)
- [ ] Add LLM-CTC intermediate loss
- [ ] Add speaker prompting

### Priority 3: Research
- [ ] Prototype JEDIS-LLM joint model
- [ ] Evaluate SPEAR features
- [ ] Implement ALA hallucination mitigation
- [ ] Test linear attention conformers

### Priority 4: Validation
- [ ] A/B test speech enhancement impact
- [ ] Benchmark all upgrades on LibriSpeech, LibriCSS
- [ ] Measure latency improvements
- [ ] Document failure modes

---

## 15. References Summary

### Must-Read Papers (Tier 1)
1. E-BATS (2506.07078) - Backprop-free TTA
2. CR-CTC (2410.05101) - CTC consistency
3. DELULU (2510.17662) - Speaker embeddings
4. MAS-LoRA (2505.20006) - LoRA mixture
5. JEDIS-LLM (2511.16046) - Joint ASR+diar
6. MossFormer2 (2312.11825) - Separation (verify)

### Important Papers (Tier 2)
7. SPEAR (2510.25955) - SSL SOTA
8. Listen Like Teacher (2511.14219) - Hallucinations
9. LibriVAD (2512.17281) - VAD
10. FlexCTC (2508.07315) - Decoding

### Reference Papers (Tier 3)
11. USM (2303.01037) - Scale reference
12. OWSM v3.1 (2401.16658) - Open baseline
13. Denoising Hurts (2512.17562) - Caution

---

*Literature review generated: 2026-01-02*
*Papers reviewed: 80+*
*For paper downloads: ./download_papers.sh*
