# Training Data Index

**Total Size:** ~2.4TB (after deduplication)
**Languages:** 25+
**Last Updated:** 2026-01-08

---

## Quick Reference

| Category | Size | Languages | Primary Use |
|----------|------|-----------|-------------|
| [VoxCeleb](#voxceleb) | 503G | EN | Speaker embeddings, verification |
| [People's Speech](#peoples-speech) | 370G | EN | Large-scale English ASR (30k hrs) |
| [Multilingual Speech](#multilingual-speech) | 372G | JA, ZH, HI, KS | ASR, Language ID |
| [MLS (Multilingual LibriSpeech)](#mls-multilingual-librispeech) | ~1.4TB | EN (OPUS 44k hrs) + NL/FR/DE/IT/PL/PT/ES | European ASR (50k+ hrs) |
| [Emotion & Punctuation](#emotion--punctuation) | 146G | EN | Emotion, Punctuation |
| [Benchmarks](#benchmarks) | 117G | Multi | Evaluation |
| [Singing](#singing-datasets) | 112G | ZH, KO, EN | Singing detection, SVS |
| [OpenSLR](#openslr-datasets) | 93G | ZH, KO, RU, DE, ES, FR | ASR training |
| [VoxPopuli](#voxpopuli) | 60G | Multi (EU) | European Parliament speech |
| [CommonVoice](#mozilla-commonvoice) | 43G | EN, HI, JA, ZH, TR | Community ASR |
| [Hi-Fi TTS](#hi-fi-tts) | 42G | EN | High-quality TTS |
| [AISHELL-4](#aishell-4) | 37G | ZH | Far-field Mandarin meetings |
| [WHAM/WHAMR Noise](#wham-noise) | 35G | - | Noise augmentation |
| [NSynth](#nsynth) | 35G | - | Musical instrument notes |
| [Emotion](#emotion-datasets) | 33G | EN, RU, JA, PL | Emotion recognition |
| [LibriTTS Full](#libritts) | 32G | EN | English TTS (585 hrs) |
| [Paralinguistics](#paralinguistics-datasets) | 29G | EN | Non-speech sounds (49,526 samples) |
| [LibriTTS-R](#libritts-r) | 28G | EN | Restored LibriTTS |
| [Libri-Light](#libri-light) | 28G | EN | Unlabeled English (6.6k hrs) |
| [Prosody](#prosody-datasets) | 23G | EN, Multi | Pitch, intonation |
| [Disfluency/Stuttering](#disfluency-datasets) | 23G | EN | Fillers, stuttering (20,906 clips) |
| [LibriSpeech Full](#librispeech) | 19G | EN | English ASR (960 hrs) |
| [MUSAN](#musan) | 12G | Multi | Music, Speech, Noise augmentation |
| [FLEURS](#fleurs) | 11G | 102 langs | Multilingual benchmark |
| [UrbanSound8K](#urbansound8k) | 5.6G | - | Urban sound events |
| [MUSDB18](#musdb18) | 4.4G | - | Music source separation |
| [VoxConverse](#voxconverse) | 4G | EN | Speaker diarization |
| [VocalSet](#vocalset) | 4.5G | EN | Singing techniques |
| [FSD50K](#fsd50k) | 2.2G | - | Sound event detection |
| [TIMIT](#timit) | 678M | EN | Phoneme recognition benchmark |

---

## Detailed Dataset Descriptions

### VoxCeleb

**Location:** `data/voxceleb_hf/` (489G) and `data/voxceleb1/` (14G)
**Total Size:** 503G
**License:** CC-BY-4.0 (VoxCeleb1), Research-only (VoxCeleb2 - requires signed agreement)
**Source:** https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

| Dataset | Location | Size | Speakers | Utterances |
|---------|----------|------|----------|------------|
| VoxCeleb1 (dev) | voxceleb_hf/vox1/ | ~37G | 1,211 | ~148k |
| VoxCeleb1 (test) | voxceleb1/ | 14G | 40 | ~4.8k |
| VoxCeleb2 (dev) | voxceleb_hf/vox2/ | ~452G | 5,994 | ~1.09M |

**Audio Format:** AAC (extracted from video, 16kHz)
**Note:** Video files (MP4) deleted to save ~257GB - only audio needed for speaker embeddings.

**Applications:**
- Speaker verification and identification
- Speaker diarization
- Voice cloning
- Speaker embedding extraction

---

### People's Speech

**Location:** `data/peoples_speech/`
**Total Size:** 370G
**License:** CC-BY-SA-4.0 (Commercial OK with attribution + share-alike)
**Source:** https://huggingface.co/datasets/MLCommons/peoples_speech

| Split | Files | Size | Hours |
|-------|-------|------|-------|
| clean/train | 804 parquet | ~350G | ~30,000 |
| Other | 9 parquet | ~20G | Various |
| **Total** | **813 parquet** | **370G** | **~30,000** |

**Audio Format:** 16kHz audio embedded in parquet files
**Quality:** Clean speech (filtered subset of full 86k hour corpus)

**Applications:**
- Large-scale English ASR training
- Punctuation prediction (replaces NC-licensed TEDLIUM-3)
- Language modeling
- Self-supervised pretraining

---

### Multilingual Speech

**Location:** `data/multilingual/`
**Total Size:** 372G (extracted, deduplicated)

| Dataset | Size | Language | Description | License |
|---------|------|----------|-------------|---------|
| **japanese/** | 324G | JA | ReazonSpeech large Japanese corpus | Apache 2.0 |
| **hindi/** | 30G | HI | Gramvaani 1000hr Hindi speech | CC-BY |
| **chinese/** | 8G | ZH | THCHS-30 Mandarin corpus | Apache 2.0 |
| **hindi_mucs/** | 10G | HI | MUCS Hindi corpus | Research |
| **kashmiri/** | 847M | KS | Low-resource Kashmiri | Research |
| **sichuanhua/** | 66M | ZH-SC | Sichuan dialect | Research |

**Note:** German, French, Spanish, Korean data moved to `data/mls/` and `data/openslr/` to avoid duplication.

**Applications:** Multilingual ASR, Language ID, Code-switching detection

---

### OpenSLR Datasets

**Location:** `data/openslr/`
**Total Size:** 93G (extracted)
**License:** Various (Apache 2.0, CC-BY, Research)

| Dataset | Size | Language | Description | License |
|---------|------|----------|-------------|---------|
| **Chinese (zh/)** | 70G | ZH | Combined Chinese corpora | |
| ├─ AISHELL-1 | 15G | ZH | 170h Mandarin read speech | Apache 2.0 |
| ├─ AISHELL-3 | 18G | ZH | 85h multi-speaker TTS | Apache 2.0 |
| ├─ Primewords | 20G | ZH | 100h Mandarin | CC-BY-NC-SA |
| └─ ST-CMDS | 17G | ZH | 100h commands | CC-BY |
| **Russian (ru/)** | 8.5G | RU | Russian LibriSpeech | CC-BY |
| **Korean (ko/)** | 10G | KO | Korean speech | |
| ├─ Zeroth Korean | 9.6G | KO | 52h clean Korean speech | CC-BY |
| └─ Deeply Korean | 268M | KO | Korean read speech corpus | CC-BY-NC-ND |
| **German (de/)** | 2.8G | DE | Thorsten German TTS | CC0 |
| **Spanish (es/)** | 2.0G | ES | Heroico Mexican Spanish | Research |
| **French (fr/)** | 1.7G | FR | African-accented French | Apache 2.0 |

**Applications:** ASR training, TTS training, Speaker verification

---

### Emotion & Punctuation

**Location:** `data/emotion_punctuation/`
**Total Size:** 146G

| Dataset | Size | Description | Labels | License |
|---------|------|-------------|--------|---------|
| **MELD** | ~100G | Friends TV multimodal | Emotion + Sentiment | CC-BY-NC |
| **TEDLIUM-3** | 50G | TED talks | Punctuation | CC-BY-NC-ND |
| ├─ train_1.tar.gz | 33G | Training split 1 | | |
| ├─ train_2.tar.gz | 17G | Training split 2 | | |
| ├─ dev.tar.gz | 167M | Development set | | |
| └─ test.tar.gz | 292M | Test set | | |
| LibriSpeech features | ~6G | Pre-extracted features | Various | CC-BY |

> **⚠️ WARNING: TEDLIUM-3 is CC-BY-NC-ND (Non-Commercial, No Derivatives). DO NOT USE for training production models. Use People's Speech instead for punctuation/ASR training.**

**Applications:**
- Emotion recognition from speech
- Punctuation prediction
- Sentiment analysis

---

### Benchmarks

**Location:** `data/benchmarks/`
**Total Size:** 117G

Standard evaluation sets for ASR, TTS, and audio understanding. Used for model evaluation only, not training.

---

### MLS (Multilingual LibriSpeech)

**Location:** `data/mls/` (7 languages OPUS) and `data/mls_english_opus/` (English OPUS)
**Total Size:** ~100G (7 languages OPUS) + ~1.3TB (English OPUS) + 16G (English FLAC subset)
**License:** CC-BY-4.0

| Language | Directory | Size | Hours | Status |
|----------|-----------|------|-------|--------|
| **English (FLAC subset)** | `data/mls/english/mls_english/` | 16G | ~500h | ✅ |
| **English (OPUS)** | `data/mls_english_opus/mls_english_opus/` | ~1.3TB | 44,659h | ✅ |
| German | mls_german_opus/ | 29G | 1,966h | ✅ |
| Dutch | mls_dutch_opus/ | 22G | 1,554h | ✅ |
| French | mls_french_opus/ | 16G | 1,076h | ✅ |
| Spanish | mls_spanish_opus/ | 14G | 918h | ✅ |
| Italian | mls_italian_opus/ | 4G | 247h | ✅ |
| Portuguese | mls_portuguese_opus/ | 9.3G | 161h | ✅ |
| Polish | mls_polish_opus/ | 6.2G | 103h | ✅ |

**Note:** English OPUS archive is large; avoid `du` on the directory (millions of files).

**Applications:** European language ASR, cross-lingual transfer

---

### Singing Datasets

**Location:** `data/singing/`
**Total Size:** 112G

| Dataset | Size | Language | Description | License |
|---------|------|----------|-------------|---------|
| **ACE-OpenCPOP** | 83G | ZH | Chinese pop singing (processed) | CC-BY-NC |
| **OpenSinger** | 16G | ZH | Chinese singing (raw audio) | Academic |
| **OpenSinger HF** | 3.6G | ZH | OpenSinger codec features | Academic |
| **M4Singer** | 2.6G | ZH | 20 multi-singer Chinese | Academic |
| **VocalSet** | 2.6G | EN | Singing techniques | CC-BY |
| **DynamicSuperb** | 2.0G | Multi | Dynamic singing eval | Academic |
| **RAVDESS Song** | 686M | EN | Emotional singing | CC-BY-NC |
| **K-Pop Voice** | 407M | KO | Korean pop vocals | Research |

**Applications:**
- Singing voice detection
- Singing voice synthesis (SVS)
- Emotion in singing
- Vocal technique classification

---

### Mozilla CommonVoice

**Location:** `data/commonvoice/`
**Total Size:** 43G (extracted)
**License:** CC0 (Public Domain)

| Language | Directory | Size |
|----------|-----------|------|
| Chinese (zh-CN) | cv-corpus-24.0-2025-12-05/zh-CN/ | 21G |
| Japanese (ja) | cv-corpus-24.0-2025-12-05/ja/ | 14G |
| English (en) | cv-corpus-24.0-2025-12-05/en/ | 1.6G |
| Hindi (hi) | cv-corpus-24.0-2025-12-05/hi/ | 524M |
| Turkish (tr) | sps-corpus-2.0-2025-12-05-tr/ | 4.2M |

**Version:** v24.0 (2025-12-05), Turkish spontaneous v2.0
**Applications:** Community-sourced ASR, accent diversity

---

### Emotion Datasets

**Location:** `data/emotion/`
**Total Size:** 33G

| Dataset | Size | Language | Emotions | License |
|---------|------|----------|----------|---------|
| **Consolidated 66K** | ~15G | EN | 8 classes | Various |
| **CREMA-D** | 592M | EN | 6 emotions (7,442 samples) | ODbL |
| **IEMOCAP** | ~3G | EN | 9 emotions | Research |
| **Dusha-Golos** | ~5G | RU | Russian emotions | CC-BY |
| **JVNV** | ~2G | JA | Japanese emotions | Research |
| **RESD** | ~1G | EN | Emotional speech | Research |
| **NeMo Polish** | ~2G | PL | Polish emotions | Apache 2.0 |

**Emotion Classes:** Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise, Contempt

**Applications:** Speech emotion recognition (SER), affective computing

---

### Prosody Datasets

**Location:** `data/prosody/`
**Total Size:** 23G
**Files:** 53,304

| Dataset | Size | Description | Labels |
|---------|------|-------------|--------|
| **Consolidated English** | ~10G | English prosody | Pitch contours |
| **CREMA-D Prosody** | ~3G | Emotional prosody | Emotion + pitch |
| **ESD (Emotional Speech)** | ~5G | Prosodic variation | Emotion + F0 |
| **RAVDESS** | ~2G | Acted emotions | Intensity + pitch |
| **Multilingual** | ~3G | Cross-lingual prosody | Language-specific |

**Labels provided:**
- Pitch (F0) contours
- Energy contours
- Duration
- Emotion annotations

**Applications:** Prosody modeling, intonation prediction, TTS

---

### Paralinguistics Datasets

**Location:** `data/paralinguistics/`
**Total Size:** 29G (includes original + converted files)
**Total Training Samples:** 49,526 (all at 16kHz mono)
**Audit Report:** `data/paralinguistics/AUDIT_REPORT.md`

---

#### VocalSound (Primary Dataset)

**Files:** 21,024 samples | **Size:** 4.4GB | **Location:** `vocalsound_labeled/audio_16k/`
**Source:** https://github.com/YuanGongND/vocalsound
**License:** CC BY-SA 4.0
**Citation:** Gong et al., "VocalSound: A Dataset for Improving Human Vocal Sound Recognition" (ICASSP 2022)

| Class | Count | Description |
|-------|-------|-------------|
| Laughter | 3,504 | Various laugh types |
| Sigh | 3,504 | Exhale expressions |
| Cough | 3,504 | Coughing sounds |
| Throat Clearing | 3,504 | Throat clearing |
| Sneeze | 3,504 | Sneezing |
| Sniff | 3,504 | Sniffing/inhaling |

**Audio Stats:** 16kHz mono WAV, ~2.6s avg per clip, **24.4 hours total**
**Quality:** Clean studio recordings, professionally labeled
**Use Cases:** Vocal event detection, health monitoring, speaker diarization

---

#### LaughterScape

**Files:** 8,170 samples | **Size:** 514MB | **Location:** `laughterscape/`
**Source:** Extracted from various sources
**License:** Research use

**Audio Stats:** 16kHz mono WAV, variable duration (0.5-5s)
**Quality:** Mixed (some background noise)
**Use Cases:** Laughter detection, emotion recognition, social interaction analysis

---

#### SEP-28k Interjections

**Files:** 7,535 samples | **Size:** Included in sep28k clips
**Source:** Apple ML Research - https://github.com/apple/ml-stuttering-events-dataset
**License:** Apple Research License
**Citation:** Lea et al., "SEP-28k: A Dataset for Stuttering Event Detection" (ICASSP 2021)

**Audio Stats:** 16kHz mono WAV, variable duration (0.5-10s)
**Quality:** Podcast audio, human-annotated, natural speech
**Use Cases:** Filler word detection ("um", "uh", "like", "you know"), disfluency detection
**Note:** Human-annotated (NOT ASR-extracted) - high quality ground truth

---

#### Fillers Dataset

**Files:** 2,051 samples | **Size:** 461MB | **Location:** `fillers/`
**Source:** Extracted from conversational speech
**License:** Research use

**Audio Stats:** 16kHz mono WAV, ~0.5-2s per clip
**Classes:** um, uh, hmm, er, ah, like, you know, etc.
**Quality:** Natural speech, some background noise
**Use Cases:** Filler word detection, ASR preprocessing, speech fluency analysis

---

#### ESC-50 (Environmental Sound Classification)

**Files:** 2,000 samples | **Size:** 313MB (16k) | **Location:** `esc50_16k/`
**Source:** https://github.com/karolpiczak/ESC-50
**License:** CC BY-NC 3.0 (Non-commercial)
**Citation:** Piczak, "ESC: Dataset for Environmental Sound Classification" (ACM MM 2015)

**Audio Stats:** 16kHz mono WAV (converted from 44.1kHz), fixed 5s clips
**Classes:** 50 categories, 40 samples each
**Quality:** Freesound.org clips, professionally curated

| Category Group | Classes |
|---------------|---------|
| Animals | dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow |
| Natural | rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm |
| Human (non-speech) | crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking/sipping |
| Domestic | door knock, mouse click, keyboard, door wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking |
| Urban | helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw |

**Use Cases:** Audio scene classification, environmental sound detection, acoustic event detection

---

#### MUSAN (Music, Speech, and Noise)

**Files:** 2,016 samples | **Size:** 12GB | **Location:** `musan/`
**Source:** https://www.openslr.org/17/
**License:** Various (see source)
**Citation:** Snyder et al., "MUSAN: A Music, Speech, and Noise Corpus" (2015)

| Category | Files | Description |
|----------|-------|-------------|
| Noise | 930 | Technical/ambient noise |
| Music | 660 | Musical excerpts |
| Speech | 426 | Speech samples |

**Audio Stats:** 16kHz mono WAV, variable duration
**Quality:** High quality, professionally curated
**Use Cases:** Data augmentation, noise robustness training, VAD training

---

#### Silence Dataset

**Files:** 2,000 samples | **Size:** 44MB | **Location:** `silence/`
**Source:** Generated/extracted from various sources
**License:** N/A

**Audio Stats:** 16kHz mono WAV, ~1s per clip
**Quality:** Near-zero amplitude audio
**Use Cases:** VAD training, silence vs speech classification, negative samples

---

#### VocalSound Extra

**Files:** 1,500 samples | **Size:** 196MB (16k) | **Location:** `vocalsound_extra_16k/`
**Source:** Additional VocalSound samples (high sample rate version)
**License:** CC BY-SA 4.0

| Class | Count |
|-------|-------|
| Cough | 300 |
| Laughter | 300 |
| Sigh | 300 |
| Sneeze | 300 |
| Throat Clearing | 300 |

**Audio Stats:** 16kHz mono WAV (converted from 192kHz)
**Quality:** High fidelity recordings
**Use Cases:** Supplement to main VocalSound dataset

---

#### FSD50K Paralinguistics Subset

**Files:** 1,522 samples | **Size:** 239MB (16k) | **Location:** `fsd50k_16k/`
**Source:** https://zenodo.org/record/4060432
**License:** CC BY 4.0
**Citation:** Fonseca et al., "FSD50K: An Open Dataset of Human-Labeled Sound Events" (2020)

| Class | Count | AudioSet Label |
|-------|-------|----------------|
| Breathing | 407 | /m/0lyf6 |
| Laughter | 500 | /m/01j3sz |
| Cough | 234 | /m/01b_21 |
| Crying | 91 | /m/0463cq4 |
| Sneeze | 52 | /m/0k65p |
| Sneeze Extra | 200 | Additional |
| Sigh | 38 | /m/07plz5l |

**Audio Stats:** 16kHz mono WAV (converted from various MP3)
**Quality:** Freesound clips with AudioSet labels, variable quality
**Use Cases:** Multi-label audio classification, sound event detection

---

#### CoughVID

**Files:** 972 samples | **Size:** 257MB (16k) | **Location:** `coughvid_16k/`
**Source:** https://coughvid.epfl.ch/
**License:** CC BY 4.0
**Citation:** Orlandic et al., "The COUGHVID crowdsourcing dataset" (Scientific Data 2021)

**Audio Stats:** 16kHz mono WAV (converted from 48kHz), ~2-10s per clip
**Quality:** Crowdsourced smartphone recordings, some noise
**Metadata:** Includes symptomatic/asymptomatic labels
**Use Cases:** COVID cough detection, respiratory health monitoring, cough classification

---

#### AudioSet Strong (Paralinguistics Subset)

**Files:** 456 samples | **Size:** 138MB (16k) | **Location:** `audioset_strong_16k/`
**Source:** AudioSet Strong subset
**License:** CC BY 4.0

| Class | Count |
|-------|-------|
| Laughter | 207 |
| Snoring | 75 |
| Crying | 70 |
| Sneeze | 50 |
| Cough | 26 |
| Breathing | 19 |
| Sigh | 5 |
| Gasp | 4 |

**Audio Stats:** 16kHz mono WAV (converted from 48kHz)
**Quality:** YouTube extracted, variable quality
**Use Cases:** Weakly-supervised audio classification

---

#### ESC-50 Paralinguistics Subset

**Files:** 280 samples | **Size:** 44MB (16k) | **Location:** `esc50_para_16k/`
**Source:** Subset of ESC-50 (human non-speech sounds only)
**License:** CC BY-NC 3.0

**Classes:** crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping (40 each, 7 selected)
**Audio Stats:** 16kHz mono WAV, fixed 5s clips
**Use Cases:** Human non-speech sound classification

---

#### Summary Table

| Dataset | Samples | License | Quality | Primary Use |
|---------|---------|---------|---------|-------------|
| vocalsound_labeled | 21,024 | CC BY-SA 4.0 | High | Vocal event detection |
| laughterscape | 8,170 | Research | Medium | Laughter detection |
| SEP-28k interjections | 7,535 | Apple Research | High | Filler detection |
| fillers | 2,051 | Research | Medium | Filler detection |
| esc50_16k | 2,000 | CC BY-NC 3.0 | High | Environmental sounds |
| musan | 2,016 | Various | High | Noise augmentation |
| silence | 2,000 | N/A | N/A | Negative samples |
| vocalsound_extra_16k | 1,500 | CC BY-SA 4.0 | High | Vocal events |
| fsd50k_16k | 1,522 | CC BY 4.0 | Medium | Multi-label audio |
| coughvid_16k | 972 | CC BY 4.0 | Medium | Cough detection |
| audioset_strong_16k | 456 | CC BY 4.0 | Medium | Audio classification |
| esc50_para_16k | 280 | CC BY-NC 3.0 | High | Human sounds |

**Total: 49,526 samples**

---

#### Target Classes for Training

```
Universal sounds: laugh, cough, sigh, breath, cry, yawn, throat_clear, sneeze, gasp, groan, snore
English fillers: um, uh, hmm, er, ah, like, you_know, so, well, actually
Multilingual fillers: nage_zh, eto_ja, eum_ko, matlab_hi, euh_fr, aeh_de
Environmental: footsteps, clapping, breathing, drinking
```

**Applications:**
- Non-speech audio event detection
- Filler word detection for ASR post-processing
- Voice activity detection (VAD)
- Health monitoring (cough, breathing, snoring)
- Emotion recognition (laughter, crying, sighing)
- Data augmentation with MUSAN noise

---

### Disfluency Datasets

**Location:** `data/sep28k/`
**Total Size:** 23G (21G episodes + 1.9G clips)
**License:** Apple Research License (SEP-28k), Research (FluencyBank)

#### SEP-28k (Stuttering Events in Podcasts)

**Source:** Apple ML Research - https://github.com/apple/ml-stuttering-events-dataset
**Status:** COMPLETE (258/385 episodes - 3 podcasts unavailable)

| Metric | Value |
|--------|-------|
| **Total Clips (dataset)** | 28,177 |
| **Downloaded Episodes** | 258/385 (67%) |
| **Extracted Clips** | 20,906 (74%) |
| **Interjection Clips** | 7,535 |
| **Audio Format** | 16kHz mono WAV |

**Download Results by Show:**

| Show | Episodes | Clips | Status |
|------|----------|-------|--------|
| WomenWhoStutter | 110 | 9,163 | Downloaded |
| StutterTalk | 82 | 5,064 | Downloaded |
| MyStutteringLife | 38 | 2,339 | Downloaded |
| HeStutters | 24 | 3,684 | Downloaded |
| HVSA | 4 | 736 | Downloaded |
| StutteringIsCool | 0 | 4,013 | Unavailable (404) |
| StrongVoices | 0 | 2,308 | Unavailable (404) |
| IStutterSoWhat | 0 | 870 | Unavailable (404) |

**Label Distribution (in extracted clips):**

| Label | Clips | Description |
|-------|-------|-------------|
| **Interjection** | 7,535 | Fillers (um, uh, like, you know) |
| **Block** | ~9,000 | Speech blocks/stoppages |
| **Prolongation** | ~6,400 | Sound prolongations |
| **SoundRep** | ~4,200 | Sound repetitions |
| **WordRep** | ~3,500 | Word repetitions |

**Files:**
- Episodes: `data/sep28k/wavs/` (21GB, 258 full podcast episodes)
- Clips: `data/sep28k/clips/` (1.9GB, 20,906 extracted clips)
- Interjection list: `data/sep28k/interjection_clips.txt` (7,535 paths)

#### FluencyBank

**Source:** TalkBank - https://fluency.talkbank.org/
**Status:** BLOCKED - Requires TalkBank authentication
**Total Clips:** 4,144

TalkBank media files require authentication. Direct wget downloads return an auth redirect page.
To access: Register at https://talkbank.org/ for research access.

**Applications:**
- Filler word detection (Interjection labels)
- Stuttering detection and classification
- Disfluency-aware ASR training
- Speech therapy research

---

### LibriSpeech

**Location:** `data/LibriSpeech/`
**Total Size:** 6.6G
**License:** CC-BY-4.0

| Split | Size | Hours | Description |
|-------|------|-------|-------------|
| train-clean-100 | ~6G | 100h | Clean read speech |
| dev-clean | ~300M | 5h | Validation set |

**Note:** Full LibriSpeech (train-clean-360, train-other-500) available at `data/benchmarks/librispeech/` via symlinks in `data/LibriSpeech_full/`

**Applications:** English ASR baseline, phoneme recognition

---

### TIMIT

**Location:** `data/timit/`
**Total Size:** 678MB
**License:** LDC User Agreement (Research only)
**Source:** HuggingFace (confit/timit)

| Split | Speakers | Utterances | Size |
|-------|----------|------------|------|
| TRAIN | 462 | 4,620 | 497MB |
| TEST | 168 | 1,680 | 181MB |
| **Total** | **630** | **6,300** | **678MB** |

**Audio Format:** 16kHz, 16-bit mono WAV

**Files per utterance:**
- `.wav` - Audio file
- `.TXT` - Orthographic transcript with sample-level timestamps
- `.PHN` - Phoneme-level alignments (61 phones)
- `.WRD` - Word-level alignments

**61 TIMIT Phonemes:**
```
Stops: b, d, g, p, t, k, dx, q
Affricates: jh, ch
Fricatives: s, sh, z, zh, f, th, v, dh
Nasals: m, n, ng, em, en, eng, nx
Semivowels/Glides: l, r, w, y, hh, hv, el
Vowels: iy, ih, eh, ey, ae, aa, aw, ay, ah, ao, oy, ow, uh, uw, ux, er, ax, ix, axr, ax-h
Others: pau, epi, h#, 1, 2
Closures: bcl, dcl, gcl, pcl, tck, kcl, tcl
```

**Dialect Regions (DR1-DR8):**
1. New England
2. Northern
3. North Midland
4. South Midland
5. Southern
6. New York City
7. Western
8. Army Brat (moved around)

**Applications:** Phoneme recognition (PER), forced alignment, acoustic-phonetic research

**SOTA Performance:** ~5% PER (wav2vec 2.0, HuBERT)

---

### VocalSet

**Location:** `data/vocalset/` and `data/singing/vocalset/`
**Total Size:** 4.5G
**License:** CC-BY-4.0

| Content | Description |
|---------|-------------|
| 20 singers | Professional vocalists |
| 17 techniques | Belt, breathy, vibrato, trill, etc. |
| 5 vowels | a, e, i, o, u |
| Multiple pitches | C3-C6 range |

**Singing Techniques:**
- Belt, Breathy, Inhaled, Lip trill, Spoken
- Straight, Vibrato, Vocal fry, Trillo
- Messa di voce, Crescendo, Decrescendo

**Applications:** Singing technique classification, vocal quality analysis

---

### Additional Small Datasets

| Dataset | Location | Size | Description |
|---------|----------|------|-------------|
| SEP-28k | `data/sep28k/` | 3.1M | Stuttering events |
| Disfluency | `data/disfluency/` | 1.5G | Speech disfluencies |
| Pitch (F0) | `data/pitch/` | 2G | Pitch annotations |

---

## License Summary

| License | Datasets |
|---------|----------|
| **CC0** | Mozilla CommonVoice, OpenSLR Thorsten German |
| **CC-BY-4.0** | LibriSpeech, MLS, VocalSet, OpenSLR Russian, OpenSLR Zeroth Korean |
| **CC-BY** | OpenSLR ST-CMDS, Dusha-Golos |
| **CC-BY-SA-4.0** | VocalSound |
| **CC-BY-NC** | MELD, ACE-OpenCPOP, RAVDESS |
| **CC-BY-NC-ND** | TEDLIUM-3, OpenSLR Deeply Korean |
| **CC-BY-NC-SA** | OpenSLR Primewords |
| **Apache 2.0** | OpenSLR AISHELL-1/3, OpenSLR African French, NeMo datasets, ReazonSpeech Japanese |
| **Academic/Research** | IEMOCAP, M4Singer, OpenSinger, OpenSLR Heroico, K-Pop Voice |
| **LDC User Agreement** | TIMIT |
| **ODbL** | CREMA-D |

---

## Commercial Use Classification

**CRITICAL: Only use datasets marked COMMERCIAL OK for production models.**

### COMMERCIAL OK (~1.6TB)

These datasets allow commercial use of trained models:

| Dataset | Size | License | Notes |
|---------|------|---------|-------|
| **VoxCeleb1** | ~51G | CC-BY-4.0 | Speaker embeddings |
| **People's Speech** | 370G | CC-BY-SA-4.0 | 30k hrs English (attribution + share-alike) |
| **LibriSpeech** | 6.6G | CC-BY-4.0 | English ASR baseline |
| **LibriSpeech Features** | 64G | CC-BY-4.0 | Pre-extracted encoder features |
| **MLS (all 7 languages)** | 93G | CC-BY-4.0 | European languages |
| **Mozilla CommonVoice** | 43G | CC0 | Public domain |
| **OpenSLR AISHELL-1/3** | 33G | Apache 2.0 | Mandarin Chinese |
| **OpenSLR ST-CMDS** | 17G | CC-BY | Chinese commands |
| **OpenSLR Zeroth Korean** | 9.6G | CC-BY | Korean speech |
| **OpenSLR Russian** | 8.5G | CC-BY | Russian LibriSpeech |
| **OpenSLR Thorsten German** | 2.8G | CC0 | German TTS |
| **OpenSLR African French** | 1.7G | Apache 2.0 | Accented French |
| **ReazonSpeech Japanese** | 324G | Apache 2.0 | Large Japanese corpus |
| **Gramvaani Hindi** | 30G | CC-BY | Hindi speech |
| **VocalSet** | 4.5G | CC-BY-4.0 | Singing techniques |
| **VocalSound** | 3G | CC-BY-SA-4.0 | Paralinguistics |
| **Dusha-Golos** | 5G | CC-BY | Russian emotions |
| **NeMo Polish** | 2G | Apache 2.0 | Polish emotions |
| **CREMA-D** | 592M | ODbL | Emotional speech (7,442 samples) |

### NON-COMMERCIAL ONLY (~700G)

**DO NOT USE for production models.** Research/evaluation only:

| Dataset | Size | License | Restriction |
|---------|------|---------|-------------|
| **VoxCeleb2** | 452G | Research Agreement | Requires signed agreement |
| **MELD** | 100G | CC-BY-NC | Friends TV - NC |
| **TEDLIUM-3** | 70G | CC-BY-NC-ND | TED talks - NC, ND |
| **OpenSLR Primewords** | 20G | CC-BY-NC-SA | Chinese - NC, SA |
| **OpenSLR Deeply Korean** | 268M | CC-BY-NC-ND | Korean - NC, ND |
| **ACE-OpenCPOP** | 15G | CC-BY-NC | Chinese singing - NC |
| **RAVDESS Song** | 5G | CC-BY-NC | Emotional singing - NC |
| **IEMOCAP** | 3G | Research | Academic only |
| **M4Singer** | 20G | Academic | Academic only |
| **OpenSinger** | 30G | Academic | Academic only |
| **K-Pop Voice** | 10G | Research | Research only |
| **OpenSLR Heroico** | 2G | Research | Mexican Spanish - Research |
| **JVNV** | 2G | Research | Japanese emotions |
| **RESD** | 1G | Research | Emotional speech |
| **DynamicSuperb** | 5G | Academic | Evaluation only |
| **TIMIT** | 678M | LDC User Agreement | Research only |

### UNKNOWN/VERIFY

| Dataset | Size | Notes |
|---------|------|-------|
| **MUCS Hindi** | 10G | Verify license before use |
| **Kashmiri** | 847M | Verify license before use |
| **Sichuanhua** | 66M | Verify license before use |
| **THCHS-30** | 8G | Verify license before use |
| **Consolidated 66K** | 15G | Mixed sources - verify each |
| **LaughterScape** | 5G | Verify license |
| **CoughVID** | 2G | Verify license |
| **ESC-50** | 1G | Verify license |

---

## Data Preparation Status

**Last Updated:** 2026-01-08

| Dataset | Size | Status |
|---------|------|--------|
| VoxCeleb 1/2 | 503G | ✅ Ready (audio only, video deleted) |
| People's Speech | 370G | ✅ Ready (813 parquet files) |
| Multilingual | 372G | ✅ Ready (deduplicated) |
| MLS (7 langs OPUS) | ~100G | ✅ Extracted |
| MLS English OPUS (44k hrs) | ~1.3TB | ✅ Extracted |
| Emotion & Punctuation | 146G | ✅ Ready |
| Benchmarks | 117G | ✅ Ready |
| Singing | 112G | ✅ Ready (OpenSinger, M4Singer, VocalSet extracted) |
| OpenSLR (6 languages) | 93G | ✅ Extracted |
| VoxPopuli (EU Parliament) | 60G | ✅ Downloaded |
| CommonVoice (5 languages) | 43G | ✅ Extracted |
| Hi-Fi TTS | 42G | ✅ Downloaded |
| AISHELL-4 (far-field Mandarin) | 37G | ✅ Extracting |
| WHAM/WHAMR Noise | 35G | ✅ Downloaded |
| NSynth (musical instruments) | 35G | ✅ Downloaded |
| Emotion | 33G | ✅ Ready |
| LibriTTS Full | 32G | ✅ Extracting |
| Paralinguistics | 29G | ✅ Ready |
| LibriTTS-R (restored) | 28G | ✅ Extracting |
| Libri-Light Small/Medium | 28G | ✅ Extracting |
| Prosody | 23G | ✅ Ready |
| LibriSpeech Full (960 hrs) | 19G | ✅ Ready |
| MUSAN (music/speech/noise) | 12G | ✅ Extracted |
| FLEURS (102 languages) | 11G | ✅ Downloaded |
| UrbanSound8K | 5.6G | ✅ Extracted |
| MUSDB18 (source separation) | 4.4G | ✅ Extracted |
| VoxConverse (diarization) | 4G | ✅ Extracted |
| VocalSet | 4.5G | ✅ Ready |
| FSD50K (sound events) | 2.2G | ✅ Extracted |
| TIMIT | 678M | ✅ Ready (HuggingFace download) |
| **TOTAL** | **~2.4TB** | ✅ |

**Notes:**
- Archives still present (~230GB) - delete after verification
- Deduplicated multilingual/ directory (~230GB saved - removed MLS/OpenSLR duplicates)
- MLS English OPUS: extracted in `data/mls_english_opus/` (44k hrs); delete `mls_english_opus.tar.gz` after verification to recover space
- Libri-Light Large (310GB) intentionally skipped - no transcripts, only for self-supervised pretraining

---

## Remaining Downloads & Preprocessing Requirements

**Last Updated:** 2026-01-08

This section documents datasets that are planned but not yet downloaded, along with their use cases and preprocessing requirements.

---

### Priority 1: Multi-Speaker & Meeting Data (CRITICAL for Diarization)

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **AMI Meeting Corpus** | ~100G | CC BY 4.0 | Far-field meeting ASR, diarization | Extract audio from video, segment by speaker, generate RTTM | groups.inf.ed.ac.uk/ami |
| **ICSI Meeting Corpus** | ~70G | BSD | Multi-speaker meetings | Similar to AMI - audio extraction, speaker segmentation | groups.inf.ed.ac.uk/ami/icsi |
| **LibriCSS** | ~13G | CC BY 4.0 | Continuous speech separation, overlap | Aligned with LibriSpeech transcripts, ready for separation training | openslr.org/131 |
| **Libri-Mixed-Speakers** | ~50G | CC BY-SA 3.0 | Overlapping speech training | Pre-mixed audio pairs with ground truth separation | openslr.org/135 |
| **SparseLibriMix** | ~20G | CC BY 4.0 | Sparse overlap (more realistic) | Pre-generated mixtures with timing annotations | github.com/popcornell/SparseLibriMix |

**Current Status:**
- AMI: 78M downloaded (incomplete - need full corpus)
- LibriCSS: Empty
- LibriMix: ✅ 458G complete (already have this for separation)

---

### Priority 2: Noise & Augmentation (CRITICAL for Robustness)

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **RIRS_NOISES** | ~1.2G | Apache 2.0 | Room impulse responses for reverb augmentation | Convolve with clean speech during training | openslr.org/28 |
| **DEMAND** | ~6.9G | CC BY-SA 3.0 | Multi-channel environmental noise | Resample to 16kHz, segment into clips | zenodo.org/record/1227121 |
| **WHAMR!** | ~100G | CC BY-NC 4.0 | Separation + noise + reverb combined | Pre-processed mixtures ready for training | wham.whisper.ai |

**Current Status:**
- DEMAND: 4.1G downloaded (partial)
- RIRS_NOISES: Not started
- WHAMR!: Not started (NC license - research only)

---

### Priority 3: Punctuation & Capitalization

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **LibriSpeech-PC** | ~20G | CC BY 4.0 | Punctuation/capitalization prediction | Aligned audio + text with punct/caps labels | openslr.org/145 |

**Note:** People's Speech (370G, now complete) can also be used for punctuation training as alternative.

**Current Status:** Empty - not started

---

### Priority 4: Dialect & Accent Robustness

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **OpenSLR Nigerian English (SLR70)** | ~1.3G | CC BY-SA 4.0 | African English accents | Standard audio + transcript format | openslr.org/70 |
| **OpenSLR UK/Ireland Dialects (SLR83)** | ~7.2G | CC BY-SA 4.0 | British Isles dialects | Standard audio + transcript format | openslr.org/83 |
| **VCTK** | ~44G | CC BY 4.0 | 110 English speakers, various accents | Resample to 16kHz, align transcripts | datashare.ed.ac.uk |

**Current Status:**
- VCTK: 1.6G downloaded (incomplete - need full 44G)
- SLR70/83: Not started

---

### Priority 5: Wakeword & Keyword Spotting

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **Speech Commands V2** | ~2.3G | CC BY 4.0 | Keyword spotting (35 commands) | Ready to use - 1 second clips | ai.google/research |
| **HI-MIA** | ~46G | Apache 2.0 | "Hi Mia" wakeword detection | Extract positive/negative examples | openslr.org/85 |
| **HI-MIA-CW** | ~0.55G | CC BY-SA 4.0 | Confusable words for wakeword | Negative mining for false positive reduction | openslr.org/120 |

**Current Status:**
- Speech Commands V2: ✅ 3.2G complete
- HI-MIA: Not started

---

### Priority 6: TTS & Prosody

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **LJSpeech** | ~2.6G | Public Domain | Single-speaker English TTS baseline | Ready to use - aligned transcripts | keithito.com/LJ-Speech-Dataset |
| **Thorsten Emotional TTS** | ~0.4G | CC0 | German emotional TTS | Emotion labels + aligned audio | openslr.org/110 |
| **PTDB-TUG** | TBD | ODbL 1.0 | Pitch tracking ground truth | F0 annotations for prosody heads | spsc.tugraz.at |
| **BibleTTS** | ~90G | CC BY-SA 4.0 | Multilingual TTS (10 languages) | Per-language extraction, alignment | openslr.org/129 |
| **CML-TTS** | ~380G | CC BY 4.0 | Large multilingual TTS | Per-language extraction | openslr.org/146 |

**Current Status:**
- LJSpeech: ✅ 3.6G complete
- Others: Not started

---

### Priority 7: Large-Scale Mandarin

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **WenetSpeech** | 22400+ hrs | CC BY 4.0 | Multi-domain Mandarin ASR | Segment extraction, transcript alignment | openslr.org/121 |
| **AISHELL-5** | ~100G | CC BY-SA 4.0 | In-car Mandarin (noisy) | Far-field audio preprocessing | openslr.org/159 |

**Current Status:**
- WenetSpeech: Empty (alternative source needed - HuggingFace may work)
- AISHELL-5: Not started
- AISHELL-1/3/4: ✅ Complete (115G total)

---

### Priority 8: Diarization Benchmarks (Evaluation Only)

| Dataset | Size | License | Use Case | Preprocessing | Source |
|---------|------|---------|----------|---------------|--------|
| **DIHARD III** | ~50G | LDC (research) | Diarization challenge benchmark | RTTM ground truth provided | dihardchallenge.github.io |
| **CHiME-5/6** | ~80G | See license | Dinner party conversations | Multi-channel audio, speaker labels | chimechallenge.github.io |

**Note:** These are primarily for evaluation, not training.

---

### Blocked/Unavailable

| Dataset | Size | Issue | Alternative |
|---------|------|-------|-------------|
| **GigaSpeech** | ~250G | Needs Dropbox legal approval (NC license) | Use People's Speech instead |
| **SPGISpeech** | ~5000 hrs | Download failed | Skip - covered by MLS English |
| **Switchboard** | ~300 hrs | Requires LDC license | Skip for now |

---

### Download Priority Summary

**Estimated Total Remaining:** ~1.1TB

| Priority | Datasets | Size | Impact |
|----------|----------|------|--------|
| 1 | AMI, ICSI, LibriCSS, Libri-Mixed | ~230G | Multi-speaker/diarization |
| 2 | RIRS, DEMAND (complete) | ~8G | Noise robustness |
| 3 | LibriSpeech-PC | ~20G | Punctuation |
| 4 | VCTK (complete), SLR70, SLR83 | ~50G | Accent robustness |
| 5 | HI-MIA | ~47G | Wakeword |
| 6 | BibleTTS, CML-TTS | ~470G | Multilingual TTS |
| 7 | WenetSpeech, AISHELL-5 | ~100G+ | Mandarin scale |

**Recommended Next Downloads:**
1. Download AMI Meeting Corpus (~100G) - critical for diarization
2. Complete HI-MIA Wakeword (~46G) - wakeword training
3. Complete UK/Ireland Dialects (SLR83) (~7G) - accent diversity
4. Complete Libri-Mixed-Speakers (~50G) - overlap speech evaluation

---

## Completed Downloads Summary

| Dataset | Directory | Size | Status |
|---------|-----------|------|--------|
| VoxCeleb1 | data/voxceleb_hf/vox1/, data/voxceleb1/ | 503GB | ✅ COMPLETE |
| VoxCeleb2 | data/voxceleb_hf/vox2/ | 452GB | ✅ COMPLETE |
| People's Speech | data/peoples_speech/ | 370GB | ✅ COMPLETE |
| CN-Celeb | data/cn-celeb/ | 21GB | ✅ COMPLETE |
| TEDLIUM-3 | data/tedlium3/ | 70GB | ✅ COMPLETE (NC) |
| CommonVoice | data/commonvoice/ | 43GB | ✅ COMPLETE |
| MLS English OPUS | data/mls_english_opus/ | 1.3TB | ✅ COMPLETE |
| LibriMix | data/librimix/ | 458GB | ✅ COMPLETE |
| Speech Commands V2 | data/speech_commands/ | 3.2GB | ✅ COMPLETE |
| LJSpeech | data/ljspeech/ | 3.6GB | ✅ COMPLETE |
| RIRS_NOISES | data/rirs_noises/ | 1.2GB | ✅ COMPLETE |
| Nigerian English (SLR70) | data/slr70_nigerian/ | 731MB | ✅ COMPLETE |
| LibriSpeech-PC | data/librispeech_pc/ | 25MB | ✅ COMPLETE (manifests) |
| DEMAND Noise | data/demand_noise/ | 4.1GB | ✅ COMPLETE |
| VCTK Multi-Speaker | data/vctk/ | 11GB | ✅ COMPLETE |
| Unified Emotion | data/emotion/unified_emotion/ | 6.2GB | ✅ COMPLETE (16,265 samples) |

## Downloads In Progress

| Dataset | Directory | Current | Target | Status |
|---------|-----------|---------|--------|--------|
| UK/Ireland Dialects (SLR83) | data/slr83_uk_ireland/ | ~255MB | ~7GB | 🔄 Downloading |
| HI-MIA Wakeword | data/hi_mia/ | ~34GB | ~46GB | 🔄 74% complete |
| Libri-Mixed-Speakers | data/libri_mixed_speakers/ | ~10GB | ~50GB | 🔄 20% complete |
| AMI Meeting Corpus | data/ami/audio/ | ~767MB | ~100GB | 🔄 Downloading |

## Blocked Downloads

| Dataset | Issue | Alternative |
|---------|-------|-------------|
| LibriCSS | Not available on HuggingFace/Zenodo (403) | Use LibriMix for separation |
| ICSI Meeting Corpus | Requires official access | Use AMI instead |

---

## Derived Data (Pre-computed Artifacts)

**Last Updated:** 2026-01-02

These artifacts are generated by training scripts and can be regenerated if deleted.

| Location | Size | Description | Generator Script |
|----------|------|-------------|------------------|
| `data/soft_labels/` | 40M | Teacher model soft labels for distillation | `scripts/generate_soft_labels.py` |
| `data/encoder_cache_train100/` | 22G | Encoder features for 100-sample training subset | `scripts/preextract_encoder_features.py` |
| `data/v4_expanded/encoder_cache/` | 266G | Full encoder cache for v4 training (~84K samples) | `scripts/preextract_encoder_features.py` |
| `data/v4_expanded/encoder_cache_val/` | 6.2G | Validation encoder cache (~2K samples) | `scripts/preextract_encoder_features.py` |
| `data/v3_multitask/encoder_cache/` | 72G | Legacy v3 multitask encoder cache | `scripts/preextract_encoder_features.py` |
| `data/pseudo_labels/` | ~1G | Pseudo-labeled data from weak supervision | Various |
| `data/meld_features/` | ~1G | Pre-extracted MELD features | Custom |
| **Total Derived** | **~366G** | | |

**Regeneration Notes:**
- Encoder caches take 4-8 hours to regenerate on M3 Max
- Soft labels require teacher model checkpoint (`models/distillation/`)
- Delete derived data to reclaim ~366GB disk space (regenerate before training)

---

## Candidate Datasets (Planned Adds; Verify Against Inventory)

**Policy:**
- **Commercial track:** Only add datasets with commercial-use rights (e.g., CC0, CC BY, Apache 2.0, MIT/BSD).
- **Research-only track:** CC BY-NC / NC-ND / academic datasets are allowed but must be segregated and used only in NC training.
- **Action required:** Verify each license on the official dataset card/site before download; re-check if a dataset appears elsewhere with different license text.
- **Inventory check:** Confirm the dataset is not already included in totals before downloading.

| Dataset | Est. Size | Domain | License (verified) | License Source | Status |
|---------|-----------|--------|--------------------|----------------|--------|
| **Speech Commands V2** | ~2.3G | Keyword ASR | CC BY 4.0 | https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html | ⬇️ Planned |
| **LibriTTS** | 32G | English TTS/ASR | CC BY 4.0 | https://www.openslr.org/60/ | ✅ Downloaded |
| **LibriTTS-R** | 28G | English TTS/ASR (cleaned) | CC BY 4.0 | https://www.openslr.org/141/ | ✅ Downloaded |
| **LibriSpeech-PC** | ~20G | English ASR (punct/case) | CC BY 4.0 | https://www.openslr.org/145/ | ⬇️ Planned |
| **Libri-Mixed-Speakers** | ~50G | Overlap speech | CC BY-SA 3.0 | https://www.openslr.org/135/ | ⬇️ Planned |
| **LibriCSS** | ~13G | English overlap/meeting | CC BY 4.0 | https://www.openslr.org/131/ | ⬇️ Planned |
| **RIRS_NOISES** | ~1.2G | Reverb + noise | Apache 2.0 | https://www.openslr.org/28/ | ⬇️ Planned |
| **AMI Meeting Corpus** | ~100G | English meetings (far-field) | CC BY 4.0 | https://groups.inf.ed.ac.uk/ami/corpus/ | ⬇️ Planned |
| **AISHELL-4** | 37G | Far-field Mandarin | CC BY-SA 4.0 | https://www.openslr.org/111/ | ✅ Downloaded |
| **AISHELL-5** | ~100G | In-car Mandarin | CC BY-SA 4.0 | https://www.openslr.org/159/ | ⬇️ Planned |
| **VCTK** | ~44G | English multi-speaker | CC BY 4.0 | https://datashare.ed.ac.uk/handle/10283/3443 | ⬇️ Planned |
| **FLEURS** | 11G | Multilingual (102 langs) | CC BY 4.0 | https://huggingface.co/datasets/google/fleurs/raw/main/README.md | ✅ Downloaded |
| **VoxPopuli** | 60G | Multilingual long-form | CC0 | https://raw.githubusercontent.com/facebookresearch/voxpopuli/main/README.md | ✅ Downloaded |
| **People's Speech** | 30k+ hours | English long-form | CC BY-SA / CC BY 4.0 | https://huggingface.co/MLCommons/peoples_speech | 🔜 Priority |
| **LJSpeech** | ~2.6G | English single-speaker | Public Domain | https://keithito.com/LJ-Speech-Dataset/ | ⬇️ Planned |
| **NSynth** | 35G | Musical notes | CC BY 4.0 | https://magenta.tensorflow.org/datasets/nsynth | ✅ Downloaded |
| **Thorsten Emotional TTS (OpenSLR 110)** | ~0.4G | Emotional TTS | CC0 | https://www.openslr.org/110/ | ⬇️ Planned |
| **BibleTTS (OpenSLR 129)** | ~90G | Multilingual TTS | CC BY-SA 4.0 | https://www.openslr.org/129/ | ⬇️ Planned |
| **CML-TTS (OpenSLR 146)** | ~380G | Multilingual TTS | CC BY 4.0 | https://www.openslr.org/146/ | ⬇️ Planned |
| **PTDB-TUG (Pitch Tracking)** | TBD | Pitch/prosody | ODbL 1.0 | https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html | ⬇️ Planned |
| | | | | | |
| **--- SOURCE SEPARATION ---** | | | | | |
| **LibriMix** | ~100G | 2-3 speaker separation | CC BY 4.0 | https://github.com/JorisCos/LibriMix | ⭐ PRIORITY |
| **Libri-Light Small** | 14G | Unlabeled English (600h) | MIT | https://github.com/facebookresearch/libri-light | ✅ Downloaded |
| **Libri-Light Medium** | 14G | Unlabeled English (6kh) | MIT | https://github.com/facebookresearch/libri-light | ✅ Downloaded |
| ~~**Libri-Light Large**~~ | ~~310G~~ | ~~Unlabeled English~~ | ~~MIT~~ | | ❌ SKIPPED |

> **Libri-Light Large SKIPPED:** 310GB of unlabeled audio (52k hours) provides NO transcripts - only useful for self-supervised pretraining (wav2vec2, HuBERT). Since we use Whisper encoder (already pretrained on 680k hours), self-supervised pretraining is unnecessary. Keeping Small (600h) + Medium (6kh) for any ablation experiments. Saves ~310GB and ~2 days download time.

| **WHAMR!** | ~100G | Separation + noise + reverb | CC BY-NC 4.0 | http://wham.whisper.ai/ | ⬇️ Planned |
| **SparseLibriMix** | ~20G | Sparse overlap separation | CC BY 4.0 | https://github.com/popcornell/SparseLibriMix | ⬇️ Planned |
| | | | | | |
| **--- DIARIZATION & MEETINGS ---** | | | | | |
| **VoxConverse** | 4G | Diarization benchmark | CC BY 4.0 | https://github.com/joonson/voxconverse | ✅ Downloaded |
| **ICSI Meeting Corpus** | ~70G | Multi-speaker meetings | BSD | https://groups.inf.ed.ac.uk/ami/icsi/ | ⬇️ Planned |
| **DIHARD III** | ~50G | Diarization challenge | LDC (research) | https://dihardchallenge.github.io/dihard3/ | ⬇️ Planned |
| **CHiME-5/6** | ~80G | Dinner party conversations | See license | https://chimechallenge.github.io/chime6/ | ⬇️ Planned |
| | | | | | |
| **--- NOISE AUGMENTATION ---** | | | | | |
| **MUSAN** | 12G | Music, speech, noise | CC BY 4.0 | https://www.openslr.org/17/ | ✅ Downloaded |
| **DEMAND (Diverse Environments Noise)** | ~6.9G | Noise (multi-channel) | CC BY-SA 3.0 | https://zenodo.org/record/1227121 | ⬇️ Planned |
| **OpenSLR Nigerian English (SLR70)** | ~1.3G | English dialect | CC BY-SA 4.0 | https://www.openslr.org/70/ | ⬇️ Planned |
| **OpenSLR UK/Ireland English Dialects (SLR83)** | ~7.2G | English dialect | CC BY-SA 4.0 | https://www.openslr.org/83/ | ⬇️ Planned |
| **Hi-Fi TTS (OpenSLR 109)** | 42G | English TTS | CC BY 4.0 | https://www.openslr.org/109/ | ✅ Downloaded |
| **HI-MIA (OpenSLR 85)** | ~46G | Wakeword (Hi, Mia) | Apache 2.0 | https://www.openslr.org/85/ | ⬇️ Planned |
| **HI-MIA-CW (OpenSLR 120)** | ~0.55G | Wakeword confusions | CC BY-SA 4.0 | https://www.openslr.org/120/ | ⬇️ Planned |
| **WenetSpeech (OpenSLR 121)** | 22400+ hours | Mandarin multi-domain | CC BY 4.0 | https://www.openslr.org/121/ | ⬇️ Planned |

### Research-only / NC/ND Candidates (Do Not Use for Commercial Training)

| Dataset | Est. Size | Domain | License (verified) | License Source | Status |
|---------|-----------|--------|--------------------|----------------|--------|
| **RAVDESS (full AV + song)** | ~24.8G | Emotional speech + song | CC BY-NC-SA 4.0 | https://zenodo.org/record/1188976 | Research-only |
| **EmoV-DB** | TBD | Emotional speech | Non-commercial | https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md | Research-only |
| **IEMOCAP** | ~3G | Emotional speech | Research-only (license form) | https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf | Research-only |
| **TEDLIUM-3 (OpenSLR 51)** | ~50G | English TED talks | CC BY-NC-ND 3.0 | https://www.openslr.org/51/ | Research-only |
| **OpenSLR Deeply Korean (SLR97)** | ~0.28G | Korean read speech | CC BY-NC-ND 4.0 | https://www.openslr.org/97/ | Research-only |
| **OpenSLR Primewords (SLR47)** | ~9G | Mandarin read speech | CC BY-NC-ND 4.0 | https://www.openslr.org/47/ | Research-only |
| **OpenSLR MagicData Mandarin (SLR68)** | ~55G | Mandarin read speech | CC BY-NC-ND 4.0 | https://www.openslr.org/68/ | Research-only |
| **OpenSLR Free ST Chinese (SLR38)** | ~8.2G | Mandarin read speech | CC BY-NC-ND 4.0 | https://www.openslr.org/38/ | Research-only |
| **OpenSLR Free ST American English (SLR45)** | ~0.35G | English read speech | CC BY-NC-ND 4.0 | https://www.openslr.org/45/ | Research-only |
| **Seoul Corpus (OpenSLR 113)** | ~2.5G | Korean spontaneous | CC BY-NC 2.0 | https://www.openslr.org/113/ | Research-only |
| **OpenCPOP (source)** | ~5.2h | Mandarin singing | CC BY-NC-ND 4.0 | https://wenet-e2e.github.io/opencpop/liscense/ | Research-only |
| **ACE-OpenCPOP (derived)** | ~83G | Mandarin pop singing | CC BY-NC-ND 4.0 (inherits OpenCPOP) | https://wenet-e2e.github.io/opencpop/liscense/ | Research-only |
| **M4Singer** | ~2.6G | Mandarin singing | CC BY-NC-SA 4.0 | https://github.com/M4Singer/M4Singer/blob/master/dataset_license.md | Research-only |
| **OpenSinger** | ~16G | Mandarin singing | License TBD (academic) | https://opensinger.github.io/ | Hold (license TBD) |
| **DynamicSuperb** | ~2.0G | Singing eval | License TBD (academic) | https://eskartur.github.io/dynamicsuperb/ | Hold (license TBD) |
| **K-Pop Voice** | ~0.4G | Korean singing | License TBD (research) | TBD | Hold (license TBD) |

**Notes:**
- These candidates are not included in the total size above unless already present in inventory; verify before download.
- Licenses verified where sources are provided; entries marked License TBD require verification before download.
- No public commercial-licensed datasets located yet for Sichuanhua or Busan-accent Korean; likely requires a separate commercial license or new collection.
- Research-only candidates must never be mixed into commercial training runs or released weights.

**Impact Notes (Why These Help):**
- **Streaming + wakeword stability:** Speech Commands, HI-MIA, HI-MIA-CW add short-utterance and confusion-negative coverage.
- **Noisy and far-field robustness:** DEMAND, RIRS_NOISES, LibriCSS, Libri-Mixed-Speakers improve overlap, reverb, and background noise handling.
- **Accent robustness (English):** SLR70 and SLR83 reduce dialectal WER spikes.
- **Mandarin scale + domain diversity:** WenetSpeech + AISHELL-4/5 improve Mandarin long-form and in-car/far-field performance.
- **Prosody/emotion modeling:** Thorsten Emotional TTS + PTDB-TUG add pitch/prosody supervision for expressive heads.
- **TTS alignment quality:** Hi-Fi TTS, LibriTTS/LibriTTS-R, VCTK improve text-audio alignment and stability.

---

## Usage Examples

### Load LibriSpeech
```python
from tools.whisper_mlx.data_loaders import LibriSpeechLoader
loader = LibriSpeechLoader("data/LibriSpeech/train-clean-100")
for audio, transcript in loader:
    ...
```

### Load Emotion Data
```python
from tools.whisper_mlx.data_loaders import EmotionLoader
loader = EmotionLoader("data/emotion/consolidated_66k")
for audio, emotion_label in loader:
    ...
```

---

## Adding New Data

1. Download to appropriate `data/` subdirectory
2. Update this index with size, license, description
3. Create data loader in `tools/whisper_mlx/data_loaders/`
4. Add preprocessing script if needed
5. Update `DATA_INDEX.md`

---

## References

- LibriSpeech: https://www.openslr.org/12/
- OpenSLR: https://www.openslr.org/
- MLS: https://www.openslr.org/94/
- CommonVoice: https://commonvoice.mozilla.org/
- MELD: https://affective-meld.github.io/
- VocalSet: https://zenodo.org/record/1442513
- TIMIT: https://catalog.ldc.upenn.edu/LDC93S1 (official), https://huggingface.co/datasets/confit/timit (HuggingFace)
