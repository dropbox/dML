# Model Index - SOTA++ Voice Server

**Purpose:** Comprehensive index of all models for SOTA++ streaming STT with rich audio features.
**Last Updated:** 2026-01-08
**Vision:** `reports/main/FINAL_ROADMAP_SOTA_PLUS_PLUS.md`
**Data Index:** `DATA_INDEX.md`

### Bug Fixes Applied (2026-01-08)

**Training Infrastructure (13 scripts fixed):**
- **max_safe_lr reduced**: 4e-5 -> 2.5e-5 to prevent 20% NaN rate in emotion training
- **eval() mode added**: ALL training scripts now use head.eval() during validation, head.train() after
  - Fixed: train_rich_audio_heads.py, train_language_head.py, train_multi_head.py
  - Fixed: train_punct_classifier.py, train_speaker_conditioned.py, train_paralinguistics.py
  - Fixed: train_prosody_ctc.py, train_punctuation.py, train_rich_decoder.py, train_rich_decoder_v3.py, train_ctc.py
- **Class weights enabled**: use_class_weights=True by default for imbalanced datasets
- **Logging spam fixed**: Rate-limited dataloader skip warnings (was 24.5% of log output)
- **Adaptive LR faster**: Triggers on first bad epoch instead of requiring 2 consecutive

**Head Models (7 heads fixed - dropout was configured but never implemented):**
- **emotion.py**: Added nn.Dropout to classifier and pooled features
- **paralinguistics.py**: Added nn.Dropout to classifier and pooled features
- **language.py**: Added nn.Dropout to classifier and pooled features
- **pitch.py**: Added nn.Dropout before final projection
- **singing.py**: Added nn.Dropout to shared layers and after pooling
- **timestamp.py**: Added nn.Dropout before final projection
- **phoneme.py**: Added nn.Dropout before classifier

### Remaining Issues (2026-01-08)
- **Missing Emotion Data**: `data/emotion/consolidated_66k` doesn't exist. Need to download 66K samples from HuggingFace.
- **Data Underutilization**: Only 7.6K emotion samples used. Run `scripts/create_unified_emotion_dataset.py` after downloading.
- **Empty Dataset**: `data/emotion/dusha_golos/` is empty (0B). Russian emotion data not downloaded.
- **Language ID training incomplete**: Training stopped at 500 steps (1 epoch), need to re-run for 5 epochs.
- **Chinese accuracy low**: 95.59% vs 99%+ for other languages. May need more Chinese training data.

---

## Executive Summary

### Current vs Target State

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| Primary ASR | Whisper (COMPLETE) + Zipformer (training heads) | Zipformer + Pruned RNN-T | Finish Zipformer heads |
| Streaming WER | ~5% (est.) | <2.5% | Train on more data |
| High-Accuracy WER | N/A | <1.5% | Add ROVER voting |
| Speaker Embeddings | Whisper encoder (256-dim) | DELULU (<0.8% EER) | Port DELULU, download VoxCeleb |
| Source Separation | None | FLASepformer | Port FLASepformer, download LibriMix |
| Rich Heads | 9 trained | 11 needed | Add timestamps, hallucination |

---

## Model Port Status (2026-01-06 Verified)

| Model | What It Is | Python MLX | C++ | Status |
|-------|------------|------------|-----|--------|
| **Whisper STT** | OpenAI's speech-to-text model. Encoder-decoder transformer. Batch processing (not streaming). | ✅ DONE | ✅ DONE | **COMPLETE** |
| **Zipformer Encoder** | Streaming speech encoder from k2/icefall. Processes audio chunk-by-chunk for real-time ASR. Replaces Whisper for streaming use cases. | ✅ DONE | ✅ Validated | **COMPLETE** |
| **Zipformer Heads** | Classification layers on top of Zipformer encoder for emotion, paralinguistics, pitch detection during streaming. | Training | Not started | IN PROGRESS |
| **Kokoro TTS** | Lightweight text-to-speech model. Converts text to speech audio. | PARTIAL | ~95KB | Needs verification |
| **CosyVoice3** | Alibaba's TTS model with voice cloning. DiT flow-matching + CausalHiFT vocoder. High-quality speech synthesis. | ✅ VALIDATED | ✅ ~2000 LOC | **COMPLETE** - See commit #2572 |
| **NLLB-200** | Meta's "No Language Left Behind" - translates between 200 languages. M2M-100 architecture. | ✅ 926 LOC VERIFIED | Not compatible | **Python MLX COMPLETE** - C++ for T5 only |
| **OPUS-MT** | Helsinki-NLP translation models. Fast, lightweight translation between language pairs. Marian architecture. | ✅ 408 LOC | T5 only | Python MLX implementation exists |
| **MADLAD-400** | Google's multilingual translation model covering 400+ languages. T5 architecture. | ✅ 747 LOC | ✅ 856 LOC | **COMPLETE** - No model weights downloaded |
| **Silero VAD** | Voice Activity Detection. Detects when someone is speaking vs silence/noise. Used to segment audio. | ✅ DONE | ✅ DONE | **COMPLETE** |

### SOTA Audio Models (7/7 Converted)

| Model | What It Is | Python MLX | Validation | Status |
|-------|------------|------------|------------|--------|
| **ECAPA-TDNN** | Language identification. TDNN with attentive statistics pooling. 14M params. | ✅ ~12K LOC | max_diff 2.6e-7 | **COMPLETE** |
| **AST** | Audio Spectrogram Transformer. ViT for audio classification. 87M params. | ✅ ~14K LOC | max_diff 1.2e-5 | **COMPLETE** |
| **emotion2vec** | Emotion recognition. wav2vec2-style encoder. 90M params. | ✅ ~16K LOC | max_diff 4e-4 | **COMPLETE** |
| **BEATs** | Audio pre-training with acoustic tokenizers. 90M params. | ✅ ~15K LOC | max_diff 5.6e-6 | **COMPLETE** |
| **wav2vec2-xlsr** | Multilingual phoneme recognition. 315M params. | ✅ ~16K LOC | Validated | **COMPLETE** |
| **wav2vec2-xlsr-SER** | Speech emotion recognition. 315M params. | ✅ ~16K LOC | max_diff 1.8e-5 | **COMPLETE** |
| **WavLM-large** | General speech representation. 316M params. | ✅ ~18K LOC | Validated | **COMPLETE** |

---

## Part 1: Model Inventory

### Core ASR Models

| Model | Status | Location | Accuracy | Data Used | Data Needed |
|-------|--------|----------|----------|-----------|-------------|
| **Zipformer Encoder** | ✅ COMPLETE | `checkpoints/zipformer/` | 2.85% WER (ref) | Pretrained (icefall) | N/A |
| **Pruned RNN-T** | NOT PORTED | - | - | - | Port from k2-fsa/icefall |
| **CTC English** | TRAINING | `checkpoints/ctc_english_full/` | loss=2.36 | LibriSpeech 100h | +MLS English OPUS (44k hrs) |
| **CTC Chinese** | PARTIAL | `checkpoints/ctc_chinese_v3/` | loss=4.49 | AISHELL-1/3 | More data |
| **CTC Japanese** | PARTIAL | `checkpoints/ctc_japanese_v3/` | loss=6.11 | ReazonSpeech | More training |
| **CTC Korean** | PARTIAL | `checkpoints/ctc_korean_v3/` | loss=4.05 | Zeroth Korean | More data |
| **CTC Hindi** | PARTIAL | `checkpoints/ctc_hindi_v3/` | loss=2.89 | Gramvaani | More training |
| **CTC German** | PARTIAL | `checkpoints/ctc_german_v3/` | loss=5.09 | MLS German | More training |
| **CTC French** | PARTIAL | `checkpoints/ctc_french_v3/` | loss=5.43 | MLS French | More training |
| **CTC Spanish** | PARTIAL | `checkpoints/ctc_spanish_v3/` | loss=5.60 | MLS Spanish | More training |
| **Whisper large-v3** | HAVE | `models/whisper-large-v3-turbo-mlx/` | 1.8% WER | N/A | Fallback only |

### Rich Audio Heads

| Head | Status | Location | Accuracy | Data Used | Data Gap |
|------|--------|----------|----------|-----------|----------|
| **Emotion (8 cls) - RichDecoder v3** | COMPLETE | `checkpoints/rich_decoder_v3_cached/` | **92.07%** | CREMA-D, RAVDESS, MELD (19K) | None (multi-task) |
| **Emotion (8 cls) - Simple Head v14** | UNDERTRAINED | `checkpoints/emotion_v14/` | 51.25% | CREMA-D, RAVDESS (7.6K) | **Fix LR/use 66K data** |
| **Paralinguistics (50 cls)** | COMPLETE | `checkpoints/paralinguistics_v3/` | **96.96%** | VocalSound, Fillers | None |
| **Language ID (9 lang)** | COMPLETE | `checkpoints/language_head_v1/` | **98.61%** | CommonVoice, OpenSLR, MLS | None |
| **Phoneme (178 IPA)** | UNDERTRAINED | `checkpoints/kokoro_phoneme_head_v3/` | 19.7% PER | LibriSpeech 450 | +TIMIT, more LS |
| **Pitch (F0 Hz)** | COMPLETE | `checkpoints/pitch_combined_v4/` | - | Prosody, VocalSet | None |
| **Singing Detection** | COMPLETE | `checkpoints/singing_v2/` | - | OpenSinger, VocalSet, M4Singer | None |
| **Punctuation** | COMPLETE | `checkpoints/punct_meld_full/` | 0.614 F1 | MELD, TEDLIUM | None |
| **Timestamps** | NOT TRAINED | - | - | - | Forced alignment data |
| **Hallucination** | NOT TRAINED | - | - | - | Phoneme mismatch logic |
| **CTC (ROVER)** | NOT TRAINED | - | - | - | Joint training needed |
| **Transducer** | NOT PORTED | - | - | - | Port Zipformer |

### Speaker & Separation Models

| Model | Status | Location | Metric | Data Used | Data Gap |
|-------|--------|----------|--------|-----------|----------|
| **DELULU Speaker** | NOT PORTED | - | Target: <0.8% EER | - | **VoxCeleb 1/2 (150G)** |
| **ECAPA-TDNN** | HAVE (SOTA) | `models/sota/ecapa-tdnn/` | 93.3% VoxLingua | N/A | Fallback option |
| **FLASepformer** | NOT PORTED | - | Target: >19dB SI-SDRi | - | **LibriMix (100G)** |
| **Conv-TasNet** | NOT PORTED | - | - | - | WHAM/WHAMR |

### Preprocessing Models

| Model | Status | Location | Notes |
|-------|--------|----------|-------|
| **Silero VAD** | COMPLETE | `models/silero_vad/` | 2.4MB, works |
| **LibriVAD** | NOT PORTED | - | Target upgrade |
| **E-BATS Adaptation** | NOT IMPLEMENTED | - | Prompt bank needed |

### External/TTS Models

| Model | Status | Location | Size |
|-------|--------|----------|------|
| **Whisper large-v3-turbo** | HAVE | `models/whisper-large-v3-turbo-mlx/` | 1.5G |
| **NLLB 600M** | HAVE | `models/mlx-nllb-600m/` | 4.3G |
| **CosyVoice3** | HAVE | `models/cosyvoice3_mlx/` | 2.8G |
| **Kokoro TTS** | HAVE | `models/kokoro/` | - |

---

## Part 2: Data Inventory

### Data We Have (Ready to Use)

| Category | Size | Samples/Hours | Quality | Commercial OK |
|----------|------|---------------|---------|---------------|
| **ASR - English** | 26G | ~1,060h | High | Yes |
| LibriSpeech | 19G | 960h | High | CC-BY-4.0 |
| MLS English (FLAC) | 7G | ~500h | High | CC-BY-4.0 |
| **ASR - Multilingual** | ~550G | ~15,000h | Mixed | Mostly |
| Japanese (ReazonSpeech) | 324G | Large | High | Apache 2.0 |
| Chinese (OpenSLR) | 70G | ~400h | High | Mixed |
| Hindi (Gramvaani) | 30G | 1,000h | Medium | CC-BY |
| MLS European (7 langs) | 93G | ~6,500h | High | CC-BY-4.0 |
| Korean (Zeroth) | 10G | 52h | High | CC-BY |
| Russian (OpenSLR) | 8.5G | - | High | CC-BY |
| **Emotion** | 33G | ~66K samples | High | Mixed |
| CREMA-D | 592M | 7,442 | High | ODbL (OK) |
| RAVDESS | 2G | 1,440 | High | CC-BY-NC |
| MELD | 100G | - | Medium | CC-BY-NC |
| **Paralinguistics** | 29G | 49,526 | High | Mostly |
| VocalSound | 5.8G | 21,024 | High | CC-BY-SA |
| Fillers/Podcast | 7G | ~10K | Medium | Research |
| **Singing** | 112G | - | High | Mixed |
| OpenSinger | 16G | - | High | Academic |
| VocalSet | 4.5G | - | High | CC-BY-4.0 |
| M4Singer | 2.6G | - | High | CC-BY-NC-SA |
| **Prosody/Pitch** | 23G | 53,304 | High | Mixed |
| **Phoneme (TIMIT)** | 678M | 6,300 | High | LDC |

### Data We Still Need

| Dataset | Size | Purpose | Priority | Blocker |
|---------|------|---------|----------|---------|
| **VoxCeleb 1/2** | **150G** | DELULU speaker embeddings | **CRITICAL** | Gated - requires registration/token |
| **AMI Meeting Corpus** | 100G | Multi-speaker meetings | MEDIUM | Not downloaded |
| **People's Speech** | ~350G | Scale English ASR | MEDIUM | Not downloaded |
| **GigaSpeech** | ~250G | Scale English ASR | LOW | ⚠️ License may be non-commercial; prefer MLS/People's Speech |

### Data Quality Issues (Need Processing Pass)

| Dataset | Issue | Action Needed |
|---------|-------|---------------|
| TEDLIUM-3 | Empty directory | Re-download from OpenSLR |
| WenetSpeech | Empty - unavailable | Find alternative source |
| SPGISpeech | Failed download | Retry or skip |
| Switchboard | Requires LDC | Purchase license or skip |
| VoxCeleb | Gated | Obtain download credentials; use `scripts/download_voxceleb.sh` |

---

## Part 3: Model-Data Gap Analysis

### SOTA++ Requirements vs Current State

| Component | Required Data | Have | Gap | Action |
|-----------|--------------|------|-----|--------|
| **Zipformer ASR** | LibriSpeech + MLS English OPUS | ✅ | - | Generate manifests + start training |
| **DELULU Speaker** | VoxCeleb 1/2 | 0G | **150G** | **Download VoxCeleb** |
| **FLASepformer** | LibriMix | ✅ (Libri2Mix generated) | - | Port FLASepformer |
| **Emotion Head** | CREMA-D, RAVDESS | 33G | 0 | Done |
| **Paralinguistics** | VocalSound, Fillers | 29G | 0 | Done |
| **Language ID** | CommonVoice, MLS | 230G | 0 | Done |
| **Phoneme Head** | LibriSpeech, TIMIT | 7G | 0 | More training |
| **Pitch Head** | Prosody, VocalSet | 26G | 0 | Done |
| **Singing Head** | OpenSinger, VocalSet | 112G | 0 | Done |
| **Timestamps** | Forced alignment | Have | 0 | Train head |
| **Hallucination** | Phoneme mismatch | Have | 0 | Implement logic |

### Download Priority Queue

```
CRITICAL (blocking SOTA++):
1. VoxCeleb 1/2     150G   voxceleb.com (needs form)
2. LibriMix         108G   data/librimix_generation/Libri2Mix (DONE)

HIGH (improves accuracy):
3. People's Speech  350G   huggingface.co/MLCommons/peoples_speech

MEDIUM (nice to have):
4. AMI Meeting      100G   groups.inf.ed.ac.uk/ami
```

---

## Part 4: Training Readiness

### Can Train NOW (Data Ready)

| Model | Data Ready | Script | Est. Time |
|-------|------------|--------|-----------|
| Phoneme Head (improve) | LibriSpeech + TIMIT | `train_kokoro_phoneme_head.py` | 4h |
| Timestamp Head | LibriSpeech (forced align) | Need to create | 8h |
| CTC multilingual | All ASR data | `train_ctc.py` | 24h+ |

### Blocked (Need Download)

| Model | Blocked By | Size to Download |
|-------|------------|------------------|
| DELULU Speaker | VoxCeleb 1/2 | 150G |
| Scaled ASR | People's Speech | 350G |

### Blocked (Need Porting)

| Model | Port From | Est. Commits |
|-------|-----------|--------------|
| Zipformer | k2-fsa/icefall | 20 |
| FLASepformer | arXiv paper | 10 |
| DELULU | arXiv paper | 8 |
| LibriVAD | arXiv paper | 5 |

---

## Part 5: Data Quality & Processing Plan

### Immediate Actions

1. **Re-download failed datasets**
   ```bash
   # TEDLIUM-3
   wget https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz

   # Verify VoxCeleb directories
   ls -la data/voxceleb1/ data/voxceleb2/
   ```

2. **Audit all "Ready" datasets**
   - Verify file counts match expectations
   - Check audio file integrity (sample rate, duration)
   - Validate label files exist and parse correctly

3. **Run test framework on each dataset**
   ```bash
   pytest tests/test_data_loaders.py -v
   ```

### Data Processing Checklist

| Dataset | Files OK | Labels OK | Loader OK | Test OK |
|---------|----------|-----------|-----------|---------|
| LibriSpeech | ? | ? | ? | ? |
| CommonVoice | ? | ? | ? | ? |
| CREMA-D | ? | ? | ? | ? |
| VocalSound | ? | ? | ? | ? |
| OpenSinger | ? | ? | ? | ? |
| TIMIT | ? | ? | ? | ? |
| MLS | ? | ? | ? | ? |
| ReazonSpeech | ? | ? | ? | ? |

### Test Infrastructure Needed

```bash
# Create data validation script
python scripts/validate_all_datasets.py --report data_quality_report.json

# Check each dataset:
# 1. Directory exists and not empty
# 2. Audio files readable (soundfile.read)
# 3. Transcripts/labels present
# 4. Sample rate = 16kHz (or resample)
# 5. Duration distribution reasonable
# 6. No corrupted files
```

---

## Part 6: File Locations Reference

### Checkpoints
```
checkpoints/
├── ctc_english_full/           24G   English CTC (training)
├── ctc_*_v3/                   ~15G  Language-specific CTC
├── paralinguistics_v3/         56M   96.96% acc
├── language_head_v1/           2.6M  98.61% acc
├── kokoro_phoneme_head_v3/     52M   19.7% PER
├── pitch_combined_v4/          5.3G  Pitch extraction
├── singing_v2/                 6.8G  Singing detection
├── rich_decoder_v3_cached/     2.7G  92.07% emotion
└── punct_meld_full/            -     Punctuation
```

### Production Models
```
models/
├── whisper-large-v3-turbo-mlx/ 1.5G  ASR backbone
├── silero_vad/                 2.4M  VAD
├── kokoro_phoneme_head/        5.8M  Phoneme (undertrained)
├── mlx-nllb-600m/              4.3G  Translation
├── cosyvoice3_mlx/             2.8G  TTS
└── sota/                       8.4G  External SOTA models
```

### Training Scripts
```
tools/whisper_mlx/
├── train_ctc.py                CTC training
├── train_kokoro_phoneme_head.py Phoneme
├── train_language_head.py      Language ID
├── train_paralinguistics.py    Non-speech sounds
├── train_multi_head.py         Pitch, singing
├── train_rich_decoder_v3.py    Multi-task emotion
└── train_punctuation.py        Punctuation
```

---

## References

- SOTA++ Roadmap: `reports/main/FINAL_ROADMAP_SOTA_PLUS_PLUS.md`
- Data Index: `DATA_INDEX.md`
- Dataset Training Mapping: `DATASET_TRAINING_INDEX.md`
- Architecture: `reports/main/ARCHITECTURE_S_TIER_STREAMING.md`
- Archived Model Index: `reports/main/archive/MODEL_INDEX_archived_2026-01-03.md`
