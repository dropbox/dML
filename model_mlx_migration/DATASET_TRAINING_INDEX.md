# Dataset → Training Job Index

**Purpose:** Maps each dataset to its training job(s), label types, and trained models.
**Last Updated:** 2026-01-01
**Reference:** See `reports/main/UNIFIED_RICH_AUDIO_ARCHITECTURE.md` for full architecture details.
**Model Index:** See `MODEL_INDEX.md` for detailed model information.
**SOTA Comparison:** See `reports/main/SOTA_COMPARISON_2026-01-01-19-17.md` for benchmark analysis.

---

## Quick Reference

| Training Job | Datasets | Total Size | Label Type | Trained Model | Status |
|--------------|----------|------------|------------|---------------|--------|
| [CTC Text](#1-asr-text-ctc--decoder) | LibriSpeech, MLS, CommonVoice, OpenSLR | ~550G | Text transcripts | `checkpoints/ctc_english_full/` | ✅ Trained |
| [Phoneme (Kokoro)](#2-phoneme-head-kokoro) | LibriSpeech, VocalSet, OpenSinger | ~25G | IPA phonemes (Misaki) | `checkpoints/kokoro_phoneme_head_v3/` | ✅ 19.7% PER |
| [Emotion](#3-emotion-head-8-classes) | CREMA-D, RAVDESS, MELD, Dusha-Golos | ~110G | Categorical emotion | `checkpoints/rich_decoder_v3_cached/` | ✅ 92.07% acc |
| [Pitch (F0)](#4-pitch-head-f0) | Prosody, VocalSet, Singing | ~140G | F0 Hz continuous | `checkpoints/pitch_combined_v4/` | ✅ Trained |
| [Paralinguistics](#5-paralinguistics-head-50-classes) | VocalSound, Fillers, Singing | ~30G | Categorical (50 classes) | `checkpoints/paralinguistics_v3/` | ✅ 96.96% acc |
| [Singing Detection](#6-singing-detection--svs-classes-40-49) | OpenSinger, M4Singer, VocalSet | ~112G | Binary singing/speech | `checkpoints/singing_v2/` | Ready |
| [Language ID](#7-language-head-9-languages) | CommonVoice, OpenSLR, MLS | ~230G | Language code | `checkpoints/language_head_v1/` | ✅ 98.61% acc |
| [Punctuation](#8-punctuation-via-prosody-beam-search) | TEDLIUM-3, MELD, Prosody | ~170G | Punctuation marks | `checkpoints/punct_meld_full/` | ✅ 0.614 F1 |
| [Speaker Embedding](#9-speaker-embedding-256-dim) | LibriSpeech, RAVDESS, CREMA-D | ~10G | Speaker ID | Whisper encoder | ✅ Implemented |

---

## Detailed Mapping

### 1. ASR Text (CTC + Decoder)

Primary training for text transcription. CTC provides streaming output.

| Dataset | Location | Size | Languages | Label Type | License | Models Using |
|---------|----------|------|-----------|------------|---------|--------------|
| **LibriSpeech** | `data/LibriSpeech/` | 6.6G | EN | Text transcripts | CC-BY-4.0 | `ctc_english_full`, `ctc_head_large_v3` |
| **MLS** | `data/mls/` | 93G | DE, NL, FR, ES, IT, PT, PL | Text transcripts | CC-BY-4.0 | `ctc_german`, `ctc_french`, `ctc_spanish` |
| **CommonVoice** | `data/commonvoice/` | 43G | EN, HI, JA, ZH, TR | Text transcripts | CC0 | `ctc_*` language-specific |
| **OpenSLR AISHELL-1/3** | `data/openslr/zh/` | 33G | ZH | Chinese characters | Apache 2.0 | `ctc_chinese`, `ctc_chinese_v3` |
| **OpenSLR ST-CMDS** | `data/openslr/zh/` | 17G | ZH | Chinese characters | CC-BY | `ctc_chinese` |
| **OpenSLR Zeroth Korean** | `data/openslr/ko/` | 9.6G | KO | Korean text | CC-BY | `ctc_korean`, `ctc_korean_v3` |
| **OpenSLR Russian** | `data/openslr/ru/` | 8.5G | RU | Cyrillic text | CC-BY | N/A |
| **ReazonSpeech** | `data/multilingual/japanese/` | 324G | JA | Japanese text | Apache 2.0 | `ctc_japanese`, `ctc_japanese_v3` |
| **Gramvaani Hindi** | `data/multilingual/hindi/` | 30G | HI | Devanagari text | CC-BY | `ctc_hindi`, `ctc_hindi_v3` |

**Label Format:** UTF-8 text, tokenized via Whisper tokenizer (51,865 vocab)
**Output:** Text logits at 50Hz frame rate
**Training Script:** `tools/whisper_mlx/train_ctc.py`

---

### 2. Phoneme Head (Kokoro)

For phoneme recognition and hallucination detection via phoneme-text mismatch.

| Dataset | Location | Size | Label Type | Labels | Models Using |
|---------|----------|------|------------|--------|--------------|
| **LibriSpeech** | `data/LibriSpeech/` | 6.6G | IPA phonemes | Generated via Misaki phonemizer | `kokoro_phoneme_head_v1/v2/v3` |
| **OpenSLR Thorsten German** | `data/openslr/de/` | 2.8G | IPA phonemes | Generated via Misaki | N/A (planned) |
| **VocalSet** | `data/singing/vocalset/` | 2.6G | Vowel phonemes | Manual labels (a,e,i,o,u) | N/A (planned) |
| **OpenSinger** | `data/singing/OpenSinger/` | 16G | Pinyin phonemes | From lyrics | N/A (planned) |
| **M4Singer** | `data/singing/FULL/` | 2.6G | Pinyin phonemes | From lyrics | N/A (planned) |

**Label Format:** 178 Misaki IPA phoneme tokens (CTC output)
**Output:** Phoneme sequence at 50Hz
**Training Script:** `tools/whisper_mlx/train_kokoro_phoneme_head.py`
**Production Model:** `models/kokoro_phoneme_head/` (33.4% PER, undertrained)

---

### 3. Emotion Head (8 classes)

Classes: `happy`, `sad`, `neutral`, `angry`, `fear`, `disgust`, `surprise`, `contempt`

| Dataset | Location | Size | Label Type | Emotions | Models Using |
|---------|----------|------|------------|----------|--------------|
| **CREMA-D** | `data/emotion/crema-d/` | 592M | Categorical (6) | ANG, DIS, FEA, HAP, NEU, SAD | `emotion_consolidated`, `rich_decoder_v1/v3` |
| **RAVDESS** | `data/prosody/` | ~2G | Categorical (8) | +surprise, contempt | `emotion_unified_v2` |
| **MELD** | `data/emotion_punctuation/` | 100G | Categorical (7) | Friends TV emotions | `rich_decoder_v3_cached` |
| **Dusha-Golos** | `data/emotion/` | 5G | Categorical | Russian emotions | N/A (planned) |
| **JVNV** | `data/emotion/` | 2G | Categorical | Japanese emotions | N/A (planned) |
| **ESD** | `data/prosody/` | 5G | Categorical (5) | 5 basic emotions | N/A |

**Label Format:** Integer class ID (0-7) per utterance or segment
**Output:** 8 emotion probabilities per frame
**Training Script:** `tools/whisper_mlx/train_rich_decoder_v3.py`
**Best Model:** `checkpoints/rich_decoder_v3_cached/` (92.07% accuracy)

---

### 4. Pitch Head (F0)

Fundamental frequency prediction in Hz.

| Dataset | Location | Size | Label Type | Range | Models Using |
|---------|----------|------|------------|-------|--------------|
| **Prosody datasets** | `data/prosody/` | 23G | F0 Hz (float) | 50-500Hz speech | `pitch_combined_v4` |
| **VocalSet** | `data/singing/vocalset/` | 2.6G | F0 Hz (float) | C3-C6 (130-1046Hz) | `pitch_mir1k_v2` |
| **RAVDESS** | `data/prosody/` | ~2G | F0 Hz (float) | Varied | `pitch_combined_v4` |
| **OpenSinger** | `data/singing/OpenSinger/` | 16G | F0 Hz (float) | Singing range | N/A |
| **M4Singer** | `data/singing/FULL/` | 2.6G | F0 Hz (float) | Multi-singer | N/A |
| **ACE-OpenCPOP** | `data/singing/ace_opencpop/` | 83G | F0 Hz (float) | Chinese pop | N/A |

**Label Format:** Float F0 in Hz per frame (0 = unvoiced)
**Output:** F0 prediction per frame
**Training Script:** `tools/whisper_mlx/train_multi_head.py` (pitch head)

---

### 5. Paralinguistics Head (50 classes)

Non-speech sounds, fillers, singing vocalizations.

#### Universal Non-Verbal (classes 0-10)

| Dataset | Location | Size | Label Type | Classes | Models Using |
|---------|----------|------|------------|---------|--------------|
| **VocalSound** | `data/paralinguistics/vocalsound/` | 5.8G | Categorical | laugh, cough, sigh, sneeze, gasp, yawn, cry | `paralinguistics_v3` |
| **LaughterScape** | `data/paralinguistics/laughterscape/` | 514M | Binary | laughter | `paralinguistics_v3` |
| **CoughVID** | `data/paralinguistics/coughvid/` | 767M | Binary | cough | `paralinguistics_v3` |
| **ESC-50** | `data/paralinguistics/esc50/` | 846M | Categorical | Environmental | `paralinguistics_v3` |
| **Silence** | `data/paralinguistics/silence/` | 44M | Binary | silence (class 0) | `paralinguistics_v3` |

#### English Fillers (classes 11-15)

| Dataset | Location | Size | Label Type | Classes | Models Using |
|---------|----------|------|------------|---------|--------------|
| **Podcast Fillers** | `data/paralinguistics/podcast_fillers/` | 6.6G | Categorical | um, uh, hmm, er, ah | `paralinguistics_v3` |
| **Fillers** | `data/paralinguistics/fillers/` | 461M | Categorical | Filler words | `paralinguistics_v3` |

**Label Format:** Integer class ID (0-49) per segment
**Output:** 50 paralinguistic class probabilities per frame
**Training Script:** `tools/whisper_mlx/train_paralinguistics.py`
**Best Model:** `checkpoints/paralinguistics_v3/` (96.96% accuracy)

---

### 6. Singing Detection & SVS (classes 40-49)

Detect singing vs speech.

| Dataset | Location | Size | Label Type | Content | Models Using |
|---------|----------|------|------------|---------|--------------|
| **ACE-OpenCPOP** | `data/singing/ace_opencpop/` | 83G | Binary + phonemes | Chinese pop | `singing_v1`, `singing_v2` |
| **OpenSinger** | `data/singing/OpenSinger/` | 16G | Binary + lyrics | Chinese singing | `singing_v1`, `singing_v2` |
| **OpenSinger HF** | `data/singing/opensinger_hf/` | 3.6G | Codec features | Pre-extracted | N/A |
| **M4Singer** | `data/singing/FULL/` | 2.6G | Binary + lyrics | 20 singers | `singing_v2` |
| **VocalSet** | `data/singing/vocalset/` | 2.6G | Technique labels | 17 techniques | `singing_v1`, `singing_v2` |
| **DynamicSuperb** | `data/singing/dynamicsuperb/` | 2.0G | Evaluation | Benchmarks | N/A |
| **RAVDESS Song** | `data/singing/ravdess_song/` | 686M | Emotion + singing | 8 emotions | `singing_v1` |
| **K-Pop Voice** | `data/singing/kpop_voice/` | 407M | Binary | Korean pop | N/A |

**Label Format:** Binary (0=speech, 1=singing) or technique class
**Training Script:** `tools/whisper_mlx/train_multi_head.py` (singing head)

---

### 7. Language Head (9+ languages)

Per-token language identification.

| Dataset | Location | Size | Label Type | Languages | Models Using |
|---------|----------|------|------------|-----------|--------------|
| **CommonVoice** | `data/commonvoice/` | 43G | Language code | EN, HI, JA, ZH, TR | `language_head_v1` |
| **OpenSLR Chinese** | `data/openslr/zh/` | 70G | Language code | ZH | `language_head_v1` |
| **OpenSLR Korean** | `data/openslr/ko/` | 10G | Language code | KO | `language_head_v1` |
| **OpenSLR Russian** | `data/openslr/ru/` | 8.5G | Language code | RU | `language_head_v1` |
| **OpenSLR German** | `data/openslr/de/` | 2.8G | Language code | DE | `language_head_v1` |
| **OpenSLR Spanish** | `data/openslr/es/` | 2.0G | Language code | ES | `language_head_v1` |
| **OpenSLR French** | `data/openslr/fr/` | 1.7G | Language code | FR | `language_head_v1` |
| **MLS** | `data/mls/` | 93G | Language code | DE, NL, FR, ES, IT, PT, PL | `language_head_v1` |
| **ReazonSpeech** | `data/multilingual/japanese/` | 324G | Language code | JA | `language_head_v1` |
| **Gramvaani** | `data/multilingual/hindi/` | 30G | Language code | HI | `language_head_v1` |

**Label Format:** ISO 639-1 language code (2 letters)
**Output:** Language probabilities per token
**Training Script:** `tools/whisper_mlx/train_language_head.py`
**Best Model:** `checkpoints/language_head_v1/` (98.61% accuracy)

---

### 8. Punctuation (via Prosody Beam Search)

Question marks and exclamation points from prosody signals.

| Dataset | Location | Size | Label Type | Notes | Models Using |
|---------|----------|------|------------|-------|--------------|
| **TEDLIUM-3** | `data/emotion_punctuation/` | 50G | Punctuation marks | TED transcripts | `punct_meld_full` |
| **MELD** | `data/emotion_punctuation/` | 100G | Punctuation marks | Friends TV | `punct_meld_full` |
| **Prosody datasets** | `data/prosody/` | 23G | F0/energy contours | For rule-based | N/A |

**Label Format:** Punctuation characters (., ?, !, ,) in transcripts
**Training Script:** `tools/whisper_mlx/train_punctuation.py`

---

### 9. Speaker Embedding (256-dim)

For downstream clustering/verification.

| Dataset | Location | Size | Label Type | Speakers | Models Using |
|---------|----------|------|------------|----------|--------------|
| **LibriSpeech** | `data/LibriSpeech/` | 6.6G | Speaker ID | ~250 | Whisper encoder |
| **RAVDESS** | `data/prosody/` | ~2G | Speaker ID | 24 actors | Whisper encoder |
| **CREMA-D** | `data/emotion/crema-d/` | 592M | Speaker ID | 91 actors | Whisper encoder |
| **VocalSet** | `data/singing/vocalset/` | 2.6G | Singer ID | 20 vocalists | Whisper encoder |

**Label Format:** Unique speaker/singer ID string
**Output:** 256-dim L2-normalized embedding per utterance

---

## Label Type Summary

| Label Type | Format | Example | Training Jobs |
|------------|--------|---------|---------------|
| **Text transcript** | UTF-8 string | "Hello world" | CTC, Decoder |
| **IPA phonemes** | Misaki tokens | /həˈloʊ wɜːld/ | Phoneme Head |
| **Emotion class** | Integer 0-7 | 3 (angry) | Emotion Head |
| **F0 pitch** | Float Hz | 220.5 | Pitch Head |
| **Paralinguistic** | Integer 0-49 | 1 (laughter) | Paralinguistics |
| **Language code** | ISO 639-1 | "en", "zh" | Language Head |
| **Punctuation** | Characters | ".", "?", "!" | Punctuation |
| **Speaker ID** | String | "speaker_001" | Speaker Embedding |
| **Binary** | 0/1 | 1 (singing) | Singing Detection |

---

## Commercial Use Summary

### Safe for Commercial Models

| Dataset | Size | Label Types | Jobs |
|---------|------|-------------|------|
| LibriSpeech | 6.6G | Text, Phonemes, Speaker | CTC, Phoneme, Speaker |
| MLS (all) | 93G | Text, Language | CTC, Language |
| CommonVoice | 43G | Text, Language | CTC, Language |
| OpenSLR (most) | ~60G | Text, Language | CTC, Language |
| ReazonSpeech | 324G | Text, Language | CTC, Language |
| Gramvaani | 30G | Text, Language | CTC, Language |
| VocalSet | 2.6G | Technique, F0, Phoneme | Singing, Phoneme, Pitch |
| VocalSound | 5.8G | Paralinguistic class | Paralinguistics |
| CREMA-D | 592M | Emotion, Speaker | Emotion, Speaker |
| Dusha-Golos | 5G | Emotion | Emotion |

### Non-Commercial Only (Research/Eval)

| Dataset | Size | Label Types | Restriction |
|---------|------|-------------|-------------|
| MELD | 100G | Emotion, Punctuation | CC-BY-NC |
| TEDLIUM-3 | 50G | Punctuation | CC-BY-NC-ND |
| RAVDESS | ~2G | Emotion, Speaker | CC-BY-NC |
| ACE-OpenCPOP | 83G | F0, Phoneme | CC-BY-NC |
| OpenSinger | 16G | Lyrics, F0 | Academic |
| M4Singer | 2.6G | Lyrics, F0 | Academic |

---

## File Locations

```
data/
├── LibriSpeech/           # 6.6G - English ASR baseline
├── mls/                   # 93G - European languages
├── commonvoice/           # 43G - Community multilingual
├── openslr/               # 93G - ASR training corpora
│   ├── zh/                #   70G Chinese
│   ├── ko/                #   10G Korean
│   ├── ru/                #   8.5G Russian
│   ├── de/                #   2.8G German
│   ├── es/                #   2.0G Spanish
│   └── fr/                #   1.7G French
├── multilingual/          # 372G - Extended multilingual
│   ├── japanese/          #   324G ReazonSpeech
│   ├── hindi/             #   30G Gramvaani
│   └── chinese/           #   8G THCHS-30
├── singing/               # 112G - Singing voice
│   ├── OpenSinger/        #   16G raw audio
│   ├── ace_opencpop/      #   83G processed
│   ├── FULL/              #   2.6G M4Singer
│   ├── vocalset/          #   2.6G techniques
│   └── ...
├── emotion/               # Emotion datasets
│   └── crema-d/           #   592M (7,442 samples)
├── paralinguistics/       # 15G - Non-speech sounds
│   ├── vocalsound/        #   5.8G (3,856 files)
│   ├── podcast_fillers/   #   6.6G
│   └── ...
├── prosody/               # 23G - Pitch/intonation
└── emotion_punctuation/   # 146G - MELD, TEDLIUM
```

---

## 10. Benchmark Datasets (External Evaluation)

Standard academic benchmarks for fair SOTA comparison. These are **test sets only** - not for training.

### Emotion Benchmarks

| Dataset | Location | Size | Samples | Classes | Split | SOTA | Status |
|---------|----------|------|---------|---------|-------|------|--------|
| **IEMOCAP** | Requires USC request | ~12h | 10K | 4-6 emotions | 5-fold CV | 78-80% | ❌ Not downloaded |
| **RAVDESS** | `data/benchmarks/ravdess/` | 1.4GB | 1,440 | 8 emotions | Actor-independent | 82.23% | ⬇️ Download needed |
| **CREMA-D (test)** | `data/emotion/crema-d/` | 592M | 7,442 | 6 emotions | Actor-independent | ~75% | ✅ Have (need split) |

**Standard evaluation:** Actor-independent test split (no actor overlap between train/test)

### Paralinguistics Benchmarks

| Dataset | Location | Size | Samples | Classes | Split | SOTA | Status |
|---------|----------|------|---------|---------|-------|------|--------|
| **ESC-50** | `data/benchmarks/ESC-50/` | 600MB | 2,000 | 50 env sounds | 5-fold CV | 96.4% (BEATs) | ⬇️ Download needed |
| **AudioSet (eval)** | Requires Google download | 2M clips | 632 | 632 classes | Standard | 48.5% mAP | ❌ Not downloaded |
| **Speech Commands V2** | `data/benchmarks/speech_commands/` | 2.3GB | 105K | 35 commands | Standard | 98.1% | ⬇️ Download needed |

**Standard evaluation:** 5-fold cross-validation for ESC-50, standard test split for others

### Language ID Benchmarks

| Dataset | Location | Size | Samples | Languages | Split | SOTA | Status |
|---------|----------|------|---------|-----------|-------|------|--------|
| **VoxLingua107** | `data/benchmarks/voxlingua107/` | 100GB full / 5GB subset | 6,628h | 107 | Standard test | 93.3% | ⬇️ Download subset |
| **FLEURS** | `data/benchmarks/fleurs/` | ~10GB | - | 102 | Standard | - | ❌ Not downloaded |

**Standard evaluation:** VoxLingua107 test set (subset for our 9 languages: EN, ZH, JA, KO, HI, RU, FR, ES, DE)

### Phoneme Recognition Benchmarks

| Dataset | Location | Size | Samples | Phones | Split | SOTA | Status |
|---------|----------|------|---------|--------|-------|------|--------|
| **TIMIT** | Requires LDC ($250) | ~400MB | 6,300 | 61 | Standard | 5-11% PER | ❌ Not downloaded |
| **LibriSpeech test-clean** | `data/LibriSpeech/test-clean/` | ~350MB | 2,620 | - | Standard | 1.8% WER | ✅ Have |

**Note:** Our phoneme model uses 178 Misaki IPA tokens (different from TIMIT's 61 phones)

### ASR Benchmarks

| Dataset | Location | Size | Hours | Languages | Split | SOTA | Status |
|---------|----------|------|-------|-----------|-------|------|--------|
| **LibriSpeech test-clean** | `data/LibriSpeech/test-clean/` | 350MB | 5.4h | EN | Standard | 1.8% WER | ✅ Have |
| **LibriSpeech test-other** | `data/LibriSpeech/test-other/` | 350MB | 5.1h | EN | Standard | 3.3% WER | ✅ Have |
| **CommonVoice test** | `data/commonvoice/*/test/` | Varies | Varies | Multi | Standard | Varies | ✅ Have |

### Download Commands

```bash
# ESC-50 (paralinguistics benchmark)
git clone https://github.com/karolpiczak/ESC-50.git data/benchmarks/ESC-50

# VoxLingua107 test subset (language ID benchmark)
# Manual download from: https://bark.phon.ioc.ee/voxlingua107/

# RAVDESS (emotion benchmark)
# Download from Kaggle: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

# Speech Commands V2
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```

### Benchmark Script

```bash
# Run all benchmarks
python scripts/benchmark_sota_comparison.py --all --download-datasets

# Run specific benchmark
python scripts/benchmark_sota_comparison.py --task emotion
python scripts/benchmark_sota_comparison.py --task paralinguistics
```

---

## 11. SOTA Model Soft Labels (Knowledge Distillation)

Pre-computed probability distributions from SOTA models for training our heads.

| Source Model | Location | Samples | Classes | Target Head | Status |
|--------------|----------|---------|---------|-------------|--------|
| **emotion2vec** | `data/soft_labels/emotion2vec_soft_labels.npz` | 20K | 8 emotions | Emotion head | ❌ Not generated |
| **BEATs** | `data/soft_labels/beats_soft_labels.npz` | 21K | 50 para | Paralinguistics head | ❌ Not generated |
| **ECAPA-TDNN** | `data/soft_labels/ecapa_soft_labels.npz` | 16K | 9 languages | Language head | ❌ Not generated |
| **wav2vec2-xlsr** | `data/soft_labels/wav2vec2_phoneme_soft_labels.npz` | 450 | 178 IPA | Phoneme head | ❌ Not generated |

**Generation script:** `scripts/generate_sota_soft_labels.py` (to be created)
**Usage:** Knowledge distillation training to transfer SOTA knowledge to our lightweight heads

---

## References

- Architecture: `reports/main/UNIFIED_RICH_AUDIO_ARCHITECTURE.md`
- Data Index: `DATA_INDEX.md`
- Model Index: `MODEL_INDEX.md`
- Training Scripts: `tools/whisper_mlx/train_*.py`
- Model Heads: `tools/whisper_mlx/rich_ctc_head.py`
- SOTA Comparison: `reports/main/SOTA_COMPARISON_2026-01-01-19-17.md`
- SOTA Integration Plan: `reports/main/SOTA_INTEGRATION_STREAMING_PLAN.md`
