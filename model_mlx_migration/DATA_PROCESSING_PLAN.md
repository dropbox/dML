# Data Processing Plan

**Created:** 2026-01-03
**Status:** Phase 1 - Extraction/cleanup in progress

---

## Current State Assessment

### Archives Still Needing Extraction

| Archive | Size | Priority | Notes |
|---------|------|----------|-------|
| mls_english_opus.tar.gz | 698G | **CRITICAL** | 44,659 hours (MLS English OPUS) |
| train_M.tar.gz (AISHELL-4) | 12G | HIGH | Far-field Mandarin |
| test.tar.gz (AISHELL-4) | 4.9G | HIGH | Evaluation set |
| nsynth-train.jsonwav.tar.gz | 12G | MEDIUM | Musical instruments |

### Archives Extracted But Not Deleted (cleanup candidates)

| Archive | Size | Status |
|---------|------|--------|
| musan.tar.gz | 10G | Extracted - DELETE |
| small.tar, medium.tar (Libri-Light) | 28G | Extracted - DELETE |
| mls_*.tar.gz (7 languages) | ~80G | Extracted to *_opus/ - DELETE |
| train-clean-360.tar.gz (LibriSpeech) | 11G | Extracted - DELETE |
| train-other-500.tar.gz (LibriSpeech) | 5.2G | Extracted - DELETE |
| train_clean_360.tar.gz (LibriTTS-R) | 13G | Extracted - VERIFY then DELETE |
| train_other_500.tar.gz (LibriTTS-R) | 44G | Extracted - VERIFY then DELETE |
| train-clean-360.tar.gz (LibriTTS) | 13G | Extracted - VERIFY then DELETE |
| train-other-500.tar.gz (LibriTTS) | 14G | Extracted - VERIFY then DELETE |
| mls_english.tar.gz (MLS English FLAC subset) | 13G | Extracted - VERIFY then DELETE |
| mls_english_opus.tar.gz (MLS English OPUS) | 698G | Extracted - VERIFY then DELETE |
| UrbanSound8K.tar.gz | 5.6G | CHECK if extracted |
| musdb18.zip | 4.4G | CHECK if extracted |
| voxconverse_test_wav.zip | 4G | CHECK if extracted |
| FSD50K.dev_audio.zip | 2.1G | CHECK if extracted |

**Potential cleanup: ~230GB**

### Datasets Downloaded via HuggingFace (already extracted)

- VoxPopuli (60G) - hf_cache format
- FLEURS (11G) - partially downloaded
- AudioSet - download in progress

---

## Phase 1: Complete Extraction

### Step 1.1: Extract remaining critical archives

```bash
# MLS English OPUS (CRITICAL - 44k hours)
cd data/mls_english_opus
tar -xzf mls_english_opus.tar.gz

# AISHELL-4 remaining
cd data/aishell4
tar -xzf train_M.tar.gz
tar -xzf test.tar.gz

# NSynth (if not extracted)
cd data/nsynth_train
tar -xzf nsynth-train.jsonwav.tar.gz
```

### Step 1.2: Verify extractions before cleanup

```bash
# Verify MLS English OPUS
find data/mls_english_opus/mls_english_opus -name "*.opus" | wc -l
# Expect: very large (millions of files)

# Verify AISHELL-4
ls data/aishell4/
# Should have: train_L/, train_M/, train_S/, test/
```

### Step 1.3: Delete verified archives

Only delete after confirming extraction success.

---

## Phase 2: Data Standardization

All audio must be converted to a standard format for training.

### Target Format
- **Sample rate:** 16kHz (Whisper encoder input)
- **Channels:** Mono
- **Format:** WAV or FLAC (lossless)
- **Bit depth:** 16-bit

### Datasets Requiring Conversion

| Dataset | Current Format | Action |
|---------|---------------|--------|
| MLS | OPUS | Convert to WAV 16kHz |
| LibriTTS | WAV 24kHz | Resample to 16kHz |
| LibriTTS-R | WAV 24kHz | Resample to 16kHz |
| AISHELL-4 | WAV (various) | Verify/resample to 16kHz |
| NSynth | WAV 16kHz | No conversion needed |
| VoxPopuli | OGG | Convert to WAV 16kHz |
| FLEURS | WAV 16kHz | No conversion needed |
| CommonVoice | MP3 | Convert to WAV 16kHz |
| Hi-Fi TTS | FLAC 44.1kHz | Resample to 16kHz |

### Conversion Script Requirements

```python
# scripts/convert_audio.py
# - Use librosa or torchaudio for resampling
# - Parallel processing with multiprocessing
# - Progress tracking
# - Checkpointing (resume capability)
```

---

## Phase 3: Manifest Generation

Each dataset needs a manifest file mapping audio paths to labels.

### Standard Manifest Format (JSONL)

```json
{"audio_path": "data/mls/english/train/1234.wav", "text": "hello world", "duration": 3.2, "speaker_id": "spk_001", "language": "en"}
```

### Manifests to Generate

| Dataset | Label Types | Output |
|---------|------------|--------|
| MLS English | text, speaker, duration | mls_english_train.jsonl |
| MLS (7 langs) | text, speaker, language | mls_multilingual.jsonl |
| LibriTTS | text, speaker, chapter | libritts_train.jsonl |
| LibriTTS-R | text, speaker | libritts_r_train.jsonl |
| LibriSpeech | text, speaker | librispeech_train.jsonl |
| AISHELL-4 | text, speaker (diarization) | aishell4_train.jsonl |
| CommonVoice | text, speaker, accent | commonvoice_train.jsonl |
| VoxPopuli | text, speaker, language | voxpopuli_train.jsonl |
| FLEURS | text, language | fleurs_train.jsonl |
| MUSAN | category (music/speech/noise) | musan.jsonl |
| NSynth | instrument, pitch, velocity | nsynth.jsonl |
| VoxConverse | speaker segments (RTTM) | voxconverse.jsonl |
| MUSDB18 | stems (vocals/drums/bass/other) | musdb18.jsonl |
| UrbanSound8K | sound class | urbansound8k.jsonl |
| FSD50K | sound events (multi-label) | fsd50k.jsonl |

---

## Phase 4: Training Job Integration

### DATASET_TRAINING_INDEX.md Mapping

| Training Job | Datasets | Labels Used |
|-------------|----------|-------------|
| **CTC ASR** | MLS, LibriSpeech, LibriTTS, AISHELL, CommonVoice, VoxPopuli | text |
| **Phoneme Recognition** | TIMIT, LibriSpeech (forced aligned) | phonemes |
| **Emotion Recognition** | RAVDESS, CREMA-D, MELD, emotion/ | emotion class |
| **Pitch/Prosody** | LibriTTS, prosody/, VocalSet | F0 contours |
| **Paralinguistics** | VocalSound, SEP-28k, paralinguistics/ | event class |
| **Speaker Embedding** | VoxCeleb, LibriSpeech, AISHELL | speaker_id |
| **Language ID** | FLEURS, MLS, CommonVoice | language |
| **Singing Detection** | VocalSet, MUSDB18, singing/ | singing/speech |
| **Source Separation** | MUSDB18, LibriMix (to download) | stems |
| **Diarization** | VoxConverse, AISHELL-4, AMI | speaker segments |
| **Sound Events** | FSD50K, ESC-50, UrbanSound8K, AudioSet | event class |
| **Noise Augmentation** | MUSAN, WHAM, DNS Challenge | - |

---

## Phase 5: Encoder Feature Pre-extraction

For efficient training, pre-extract Whisper encoder features.

### Process
1. Load Whisper encoder (MLX)
2. Process audio in batches
3. Save features as .npy or .safetensors
4. Create feature manifest

### Storage Estimate
- ~100 bytes per frame (80-dim mel + position)
- ~50 frames/second
- 1 hour audio ≈ 18MB features
- 50,000 hours ≈ 900GB features

### Priority Order
1. MLS English (44k hrs) → ~800GB features
2. LibriSpeech (960 hrs) → ~17GB features
3. LibriTTS (585 hrs) → ~10GB features
4. CommonVoice (varies) → ~varies
5. Other datasets as needed

---

## Immediate Action Items

### TODAY (Priority Order)

1. **Extract MLS English** (CRITICAL)
   ```bash
   cd data/mls_english_opus
   tar -xzf mls_english_opus.tar.gz
   ```

2. **Extract AISHELL-4 remaining**
   ```bash
   cd /Users/ayates/model_mlx_migration/data/aishell4
   tar -xzf train_M.tar.gz && tar -xzf test.tar.gz
   ```

3. **Verify and cleanup** ~230GB of duplicate archives

4. **Create manifest generation script**
   - Start with LibriSpeech (well-documented format)
   - Extend to MLS, LibriTTS

### THIS WEEK

5. **Audio format standardization script**
   - OPUS → WAV conversion for MLS
   - Resampling for 24kHz datasets

6. **Encoder feature pre-extraction**
   - Start with LibriSpeech-100 (smallest, fastest iteration)
   - Scale to full datasets

---

## Disk Space Budget

| Category | Current | After Cleanup | After Features |
|----------|---------|---------------|----------------|
| Raw audio | 2.0TB | 1.8TB | 1.8TB |
| Archives (deletable) | 230GB | 0 | 0 |
| Encoder features | 366GB | 366GB | ~1.2TB |
| **Total** | ~2.6TB | ~2.2TB | ~3.0TB |

**Recommendation:** Delete archives after verification to make room for encoder features.
