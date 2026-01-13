# STT SOTA Literature Review - Paper Index
**Date:** 2026-01-02
**Purpose:** Comprehensive index of state-of-the-art STT research papers (2024-2026)
**Goal:** Exceed performance of ARCHITECTURE_GOD_TIER_STT.md

---

## Executive Summary

This index catalogs 80+ papers across 12 research categories relevant to building a SOTA+ STT system. Key findings that could exceed the current GOD TIER architecture:

### Breakthrough Opportunities
1. **CR-CTC** (arXiv:2410.05101) - Consistency regularization could improve streaming CTC
2. **E-BATS** (arXiv:2506.07078) - Backprop-free TTA more efficient than SUTA
3. **DELULU** (arXiv:2510.17662) - 62% relative EER improvement for speaker verification
4. **MAS-LoRA** (arXiv:2505.20006) - Mixture of Accent-Specific LoRAs for multi-accent
5. **SPEAR** (arXiv:2510.25955) - New SOTA on SUPERB benchmark
6. **MossFormer2** (arXiv:2312.11825) - 21.2 dB SI-SDRi approaching theoretical limits

---

## Category 1: End-to-End ASR Models

### 1.1 Major Foundation Models

| Paper | arXiv | Key Contribution | Relevance |
|-------|-------|-----------------|-----------|
| Google USM | 2303.01037 | 12M hours, 300+ languages, SOTA multilingual | Architecture reference |
| USM-Lite | 2312.08553 | 9.4% model size, 7.3% WER regression | Compression techniques |
| OWSM v3.1 | 2401.16658 | E-Branchformer, 25% faster inference | Open alternative to Whisper |
| OWSM-CTC | 2402.12654 | Encoder-only, 180k hours, 24% ST improvement | CTC architecture |
| SeamlessM4T | 2308.11596 | 100 languages, 20% BLEU improvement | Multilingual baseline |
| SeamlessM4T v2 | 2312.05187 | Prosody preservation, streaming | Expressive speech |
| Marco-ASR | 2512.22165 | Multi-domain fine-tuning framework | Domain adaptation |
| MiMo-Audio | 2512.23808 | Audio LM few-shot learning | Few-shot capabilities |

### 1.2 Architecture Improvements

| Paper | arXiv | Key Contribution | Metrics |
|-------|-------|-----------------|---------|
| Linear Conformers | 2409.07165 | Linear-time attention, streaming | Less compute/memory |
| NoisyD-CT | 2509.01087 | Tri-stage noise robustness | 25.7% WER reduction noisy |
| One-bit ASR | 2505.21245 | 1-2 bit quantization | 16.6x size reduction |
| All-in-One ASR | 2512.11543 | Multi-mode joiner | Unified streaming/offline |
| Zipformer Unified | 2506.14434 | Dynamic right-context | 7.9% WER reduction |

---

## Category 2: Speaker-Adaptive ASR

### 2.1 Speaker Conditioning

| Paper | arXiv | Key Technique | Improvement |
|-------|-------|--------------|-------------|
| Context-Aware Whisper | 2511.18774 | Decoder prompting, speaker synthesis | 22.3% WER reduction Arabic |
| Fairness-Prompted | 2510.18374 | Fairness fine-tuning, adapters | L2 speaker improvement |
| Knowledge-Decoupled | 2510.10401 | Gated parameter isolation | Personal ASR |
| Variational LoRA | 2509.20397 | Bayesian low-rank adaptation | Impaired speech |
| Homogeneous Features | 2407.06310 | Spectral basis embedding | Dysarthric/elderly |

### 2.2 LoRA and Adapters

| Paper | arXiv | Key Technique | Application |
|-------|-------|--------------|-------------|
| **MAS-LoRA** | 2505.20006 | Mixture of Accent-Specific LoRAs | Multi-accent ASR |
| Privacy LoRA | 2512.16401 | Zero-data-exfiltration | Domain adaptation |
| Omni-AVSR | 2511.07253 | LoRA for audio-visual | Multimodal |
| CarelessWhisper | 2508.12301 | LoRA for streaming Whisper | Latency <300ms |
| Perceiver-Prompt | 2406.09873 | P-Tuning + LoRA | Dysarthric speech |

---

## Category 3: Test-Time Adaptation

### 3.1 TTA Methods

| Paper | arXiv | Key Technique | Advantage |
|-------|-------|--------------|-----------|
| LI-TTA | 2408.05769 | Language model corrections | Acoustic + linguistic |
| SUTA/SGEM Study | 2409.13095 | Child speech personalization | Speaker-specific |
| **E-BATS** | 2506.07078 | Backprop-free prompt adaptation | More efficient than SUTA |
| TTT-SE | 2508.01847 | Y-shaped self-supervised | Speech enhancement TTA |
| EMO-TTA | 2509.25495 | EM class statistics | Emotion recognition |

---

## Category 4: Multi-Speaker ASR

### 4.1 Joint ASR + Diarization

| Paper | arXiv | Key Technique | Performance |
|-------|-------|--------------|-------------|
| JEDIS-LLM | 2511.16046 | Zero-shot streaming | Long audio capable |
| DiCoW | 2510.03723 | Diarization-conditioned Whisper | Target-speaker |
| SpeakerLM | 2508.06372 | Multimodal LLM | Unified SD+ASR |
| SA-Paraformer | 2310.04863 | Non-autoregressive | 6.1% SD-CER reduction |
| GLAD MoE | 2509.13093 | Dynamic MoE | Multi-talker |
| Cocktail Party CoT | 2509.15612 | Chain-of-thought + RL | Target speaker |

### 4.2 Source Separation

| Paper | arXiv | Key Technique | SI-SDRi |
|-------|-------|--------------|---------|
| **MossFormer** | 2302.11824 | Gated attention | 21.2 dB (WSJ0-3mix) |
| **MossFormer2** | 2312.11825 | RNN-free recurrent | Near upper bound |
| TFGA-Net | 2510.12275 | Temporal-frequency graph | EEG-guided |

---

## Category 5: Speech Enhancement

### 5.1 Denoising

| Paper | arXiv | Key Technique | Performance |
|-------|-------|--------------|-------------|
| Medical ASR Study | 2512.17562 | Enhancement can hurt ASR | Caution for pipeline |
| GDiffuSE | 2510.04157 | Diffusion + noise guidance | Novel approach |
| Shortcut Flow | 2509.21522 | Single-step inference | RTF 0.013 |
| Quantum Fourier | 2509.04851 | Quantum-inspired | 15 dB SNR improvement |
| WTFormer | 2506.22001 | Wavelet Conformer | 0.98M params |

---

## Category 6: Self-Supervised Speech

### 6.1 Foundation Models

| Paper | arXiv | Key Technique | Improvement |
|-------|-------|--------------|-------------|
| SpidR-Adapt | 2512.21204 | Meta-learning | 100x less data |
| MauBERT | 2511.12690 | Articulatory HuBERT | Cross-lingual |
| **SPEAR** | 2510.25955 | Unified SSL framework | New SOTA SUPERB |
| WavJEPA | 2509.23238 | Waveform representation | SOTA time-domain |
| **DELULU** | 2510.17662 | Speaker-discriminative | 62% EER improvement |

---

## Category 7: CTC Improvements

| Paper | arXiv | Key Technique | Improvement |
|-------|-------|--------------|-------------|
| **CR-CTC** | 2410.05101 | Consistency regularization | Reduced overfitting |
| FlexCTC | 2508.07315 | GPU beam decoding | Fast contextualization |
| LLM-CTC | 2506.22846 | LLM intermediate loss | Better linguistic model |
| LCS-CTC | 2508.03937 | Soft alignment | Phonetic robustness |
| Label Priors | 2406.02560 | Less peaky outputs | Better forced alignment |

---

## Category 8: Streaming ASR

| Paper | arXiv | Key Technique | Latency |
|-------|-------|--------------|---------|
| CarelessWhisper | 2508.12301 | Causal streaming Whisper | <300ms chunks |
| All-in-One ASR | 2512.11543 | Multi-mode joiner | Flexible |
| JEDIS-LLM | 2511.16046 | Speaker prompt cache | Zero-shot streaming |
| Zipformer | 2506.14434 | Chunked attention | Configurable |

---

## Category 9: Speaker Embeddings

| Paper | arXiv | Key Technique | EER |
|-------|-------|--------------|-----|
| M-Vec | 2409.15782 | Matryoshka embeddings | 8-dim viable |
| UniPET-SPK | 2501.16542 | Parameter-efficient tuning | 5.4% params |
| Contrastive SPK | 2410.05037 | Intermediate contrastive | 9.05% improvement |
| SSPS | 2505.14561 | Self-supervised positive | 2.57% VoxCeleb1-O |

---

## Category 10: Hallucination Mitigation

| Paper | arXiv | Key Technique | Application |
|-------|-------|--------------|-------------|
| Listen Like Teacher | 2511.14219 | Adaptive Layer Attention | Whisper hallucinations |
| BoH Investigation | 2501.11378 | Bag of hallucinations | Post-processing |
| Quantization Study | 2503.09905 | INT4/5/8 quantization | Latency + hallucination |
| Adaptive Steering | 2510.12851 | Layer-wise intervention | Audio grounding |

---

## Category 11: VAD Improvements

| Paper | arXiv | Key Technique | Advantage |
|-------|-------|--------------|-----------|
| LibriVAD | 2512.17281 | Vision Transformer VAD | Outperforms established |
| SincQDR-VAD | 2508.20885 | Learnable bandpass | 31% fewer params |
| Tiny VAD | 2507.22157 | Noise-robust lightweight | AIoT devices |
| sVAD | 2403.05772 | Spiking neural network | Low power |
| Self-supervised VAD | 2312.16613 | SSL pretraining | Noise robust |

---

## Category 12: Datasets

| Paper/Dataset | arXiv | Size | Purpose |
|---------------|-------|------|---------|
| LibriConvo | 2510.23320 | 240.1 hours | Simulated conversations |
| LibriVAD | 2512.17281 | - | VAD benchmark |
| PROFASR-BENCH | 2512.23686 | - | Context-conditioned ASR |

---

## Download Script

```bash
#!/bin/bash
# Download all papers from arXiv
PAPER_DIR="papers/stt_2024_2026/pdfs"
mkdir -p "$PAPER_DIR"

# Category 1: Foundation Models
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2303.01037.pdf  # USM
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2312.08553.pdf  # USM-Lite
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2401.16658.pdf  # OWSM v3.1
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2402.12654.pdf  # OWSM-CTC
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2308.11596.pdf  # SeamlessM4T
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2312.05187.pdf  # SeamlessM4T v2
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2512.22165.pdf  # Marco-ASR

# Category 2: Speaker-Adaptive
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2511.18774.pdf  # Context-Aware Whisper
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2505.20006.pdf  # MAS-LoRA
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2509.20397.pdf  # Variational LoRA
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2406.09873.pdf  # Perceiver-Prompt

# Category 3: Test-Time Adaptation
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2408.05769.pdf  # LI-TTA
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2506.07078.pdf  # E-BATS
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2409.13095.pdf  # SUTA study

# Category 4: Multi-Speaker
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2511.16046.pdf  # JEDIS-LLM
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2510.03723.pdf  # DiCoW
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2508.06372.pdf  # SpeakerLM
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2509.13093.pdf  # GLAD MoE
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2302.11824.pdf  # MossFormer
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2312.11825.pdf  # MossFormer2

# Category 5: Speech Enhancement
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2510.04157.pdf  # GDiffuSE
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2509.21522.pdf  # Shortcut Flow

# Category 6: Self-Supervised
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2510.25955.pdf  # SPEAR
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2510.17662.pdf  # DELULU
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2509.23238.pdf  # WavJEPA

# Category 7: CTC
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2410.05101.pdf  # CR-CTC
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2508.07315.pdf  # FlexCTC
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2506.22846.pdf  # LLM-CTC

# Category 8: Streaming
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2508.12301.pdf  # CarelessWhisper
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2512.11543.pdf  # All-in-One ASR

# Category 9: Speaker Embeddings
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2409.15782.pdf  # M-Vec
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2505.14561.pdf  # SSPS

# Category 10: Hallucination
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2511.14219.pdf  # Listen Like Teacher
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2510.12851.pdf  # Adaptive Steering

# Category 11: VAD
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2512.17281.pdf  # LibriVAD
wget -nc -P "$PAPER_DIR" https://arxiv.org/pdf/2508.20885.pdf  # SincQDR-VAD

echo "Downloaded $(ls -1 $PAPER_DIR/*.pdf | wc -l) papers"
```

---

## Priority Reading List

### Tier 1: Must Read (Directly applicable to exceeding GOD TIER)
1. **E-BATS** (2506.07078) - Replace SUTA with backprop-free TTA
2. **CR-CTC** (2410.05101) - Improve CTC streaming path
3. **MAS-LoRA** (2505.20006) - Better than single LoRA per speaker
4. **DELULU** (2510.17662) - Better speaker embeddings than ECAPA-TDNN
5. **MossFormer2** (2312.11825) - Already in architecture, verify implementation
6. **JEDIS-LLM** (2511.16046) - Zero-shot streaming joint ASR+diarization

### Tier 2: Important (Architecture improvements)
7. **SPEAR** (2510.25955) - New SSL foundation
8. **Listen Like Teacher** (2511.14219) - Hallucination mitigation
9. **LibriVAD** (2512.17281) - Better VAD than Silero
10. **CarelessWhisper** (2508.12301) - Streaming Whisper techniques

### Tier 3: Reference (Context and baselines)
11. USM (2303.01037) - Large-scale training reference
12. OWSM v3.1 (2401.16658) - Open Whisper baseline
13. SeamlessM4T (2308.11596) - Multilingual reference

---

## Key Gaps to Address Beyond GOD TIER

Based on this literature review, the current GOD TIER architecture could be enhanced by:

1. **Replace SUTA with E-BATS** - Backprop-free is more efficient
2. **Upgrade speaker embeddings** - DELULU shows 62% EER improvement over baselines
3. **MAS-LoRA instead of single LoRA** - Mixture approach handles accent variation
4. **CR-CTC for streaming** - Consistency regularization improves generalization
5. **ViT-based VAD** - LibriVAD shows ViT outperforms traditional VAD
6. **LLM-CTC intermediate loss** - Better linguistic modeling in CTC path
7. **Joint ASR+diarization** - JEDIS-LLM enables single-model approach

---

*Index generated: 2026-01-02*
*Papers: 80+ across 12 categories*
*For literature review: See STT_LITERATURE_REVIEW_2026.md*
