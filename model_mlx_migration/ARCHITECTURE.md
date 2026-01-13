# Complete System Architecture

**Last Updated:** 2026-01-02
**Status:** Production + Active Development
**Role:** CANONICAL SOURCE OF TRUTH for system architecture

---

## Related Documents

| Document | Role | Status |
|----------|------|--------|
| **ARCHITECTURE.md** (this file) | System overview, component status | **CANONICAL** |
| **[ARCHITECTURE_GOD_TIER_STT_V3.md](reports/main/ARCHITECTURE_GOD_TIER_STT_V3.md)** | Hardened spec: overlap, streaming, gates | **ACTIVE** |
| [REVIEW_GOD_TIER_STT_STREAMING_RISKS_PATCH](reports/main/REVIEW_GOD_TIER_STT_STREAMING_RISKS_PATCH_2026-01-02-21-33.md) | Risk review informing V3 | Reference |
| [SPEECH_ENHANCEMENT_SOTA_RESEARCH](reports/main/SPEECH_ENHANCEMENT_SOTA_RESEARCH_2026-01-02.md) | Denoising, AEC, dereverberation research | Reference |
| [ARCHITECTURE_GOD_TIER_STT.md](reports/main/ARCHITECTURE_GOD_TIER_STT.md) | V2.0 | **SUPERSEDED** |
| [ARCHITECTURE_SPEAKER_ADAPTIVE_SOTA_PLUS_v1](reports/main/archive/architecture_2026-01-02/ARCHITECTURE_SPEAKER_ADAPTIVE_SOTA_PLUS_v1.md) | V1.0 | **ARCHIVED** |

---

## Critical Contract Decisions (Binding)

These decisions resolve prior document conflicts and are BINDING:

| ID | Decision | Rationale |
|----|----------|-----------|
| **C1** | We do speaker-local tracking, NOT diarization | Session-local IDs; downstream does global clustering |
| **C2** | Paralinguistics: 50-class schema, 11 trained (96.96%) | Report metrics with class count |
| **C3** | Para tokens are OUT-OF-BAND, not in vocabulary | Checkpoint compatibility preserved |
| **C4** | MossFormer2 3-speaker uses 8kHz→16kHz resampling | Expect -3dB SI-SDRi vs 2-speaker |
| **C5** | Alignment IDs namespaced: `spk_{idx}_a{seq}` | No cross-speaker backtracks |
| **C6** | Latency: 1s chunk, batch=1, M2 Max baseline | CTC p50: 77ms, Decoder p50: 227ms |

See [ARCHITECTURE_GOD_TIER_STT_V3.md](reports/main/ARCHITECTURE_GOD_TIER_STT_V3.md) for full rationale, detailed design, and falsifiable benchmark gates.

---

## System Overview

A streaming, multi-speaker, rich audio understanding system built on Whisper MLX with multi-head outputs for emotion, pitch, phonemes, paralinguistics, speaker embeddings, and language identification.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE SYSTEM DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         RAW AUDIO INPUT (16kHz)                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                            │
│                                        ▼                                            │
│   ╔═════════════════════════════════════════════════════════════════════════════╗   │
│   ║                    LAYER 0: UPSTREAM PREPROCESSING                           ║   │
│   ╚═════════════════════════════════════════════════════════════════════════════╝   │
│                                        │                                            │
│            ┌───────────────────────────┼───────────────────────────┐               │
│            ▼                           ▼                           ▼               │
│   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐          │
│   │   SILERO VAD    │       │    OVERLAP      │       │   MOSSFORMER2   │          │
│   │                 │       │    DETECTOR     │       │      MLX        │          │
│   │   ✅ Production │       │                 │       │                 │          │
│   │   Latency: 2ms  │       │   ❌ Need Train │       │   ✅ Production │          │
│   │                 │       │   Latency: 5ms  │       │   Latency: 30ms │          │
│   │   Output:       │       │                 │       │   SI-SDRi: 21dB │          │
│   │   - speech_prob │       │   Output:       │       │   Speed: 9.5×RT │          │
│   │   - is_speech   │       │   - num_spkrs   │       │                 │          │
│   │                 │       │     (0,1,2,3)   │       │   Output:       │          │
│   └────────┬────────┘       └────────┬────────┘       │   - N separate  │          │
│            │                         │                │     waveforms   │          │
│            │                         │                └────────┬────────┘          │
│            │                         │                         │                   │
│            └─────────────────────────┼─────────────────────────┘                   │
│                                      │                                              │
│                                      ▼                                              │
│                      ┌───────────────────────────────┐                             │
│                      │     CONDITIONAL ROUTING       │                             │
│                      │                               │                             │
│                      │  if num_speakers <= 1:       │                             │
│                      │    → FAST PATH (no sep)      │                             │
│                      │                               │                             │
│                      │  if num_speakers >= 2:       │                             │
│                      │    → SEPARATION PATH         │                             │
│                      │      (MossFormer2)           │                             │
│                      └───────────────┬───────────────┘                             │
│                                      │                                              │
│                                      ▼                                              │
│   ╔═════════════════════════════════════════════════════════════════════════════╗   │
│   ║                    LAYER 1: WHISPER ENCODER                                  ║   │
│   ╚═════════════════════════════════════════════════════════════════════════════╝   │
│                                      │                                              │
│                                      ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                     WHISPER ENCODER (Frozen)                                 │   │
│   │                                                                              │   │
│   │   ✅ Production                                                              │   │
│   │   Model: large-v3                                                            │   │
│   │   Params: 1.5B (frozen)                                                      │   │
│   │   Output: 1280-dim embeddings @ 50Hz                                         │   │
│   │   Latency: ~45ms                                                             │   │
│   │                                                                              │   │
│   │   Audio → Mel Spectrogram → Conv Stem → 32× Transformer Blocks → Features   │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│                    ┌─────────────────┴─────────────────┐                           │
│                    │                                   │                           │
│                    ▼                                   ▼                           │
│   ╔═════════════════════════════╗   ╔═════════════════════════════════════════╗   │
│   ║  LAYER 2A: CTC STREAM       ║   ║  LAYER 2B: DECODER STREAM               ║   │
│   ║  (Low Latency ~60ms)        ║   ║  (High Quality ~200-300ms)              ║   │
│   ╚═════════════════════════════╝   ╚═════════════════════════════════════════╝   │
│                    │                                   │                           │
│                    ▼                                   ▼                           │
│   ┌─────────────────────────────┐   ┌─────────────────────────────────────────┐   │
│   │      RICH CTC HEAD          │   │          RICH DECODER                   │   │
│   │                             │   │                                         │   │
│   │   ✅ Production             │   │   🔄 Training (82.34%)                  │   │
│   │   Latency: ~10ms            │   │   Latency: ~150-200ms                   │   │
│   │                             │   │                                         │   │
│   │   ┌───────────────────────┐ │   │   Architecture:                         │   │
│   │   │ Text CTC (51,865)     │ │   │   - Whisper Decoder (frozen)            │   │
│   │   │ Emotion (8 classes)   │ │   │   - LoRA Adapters (trainable)           │   │
│   │   │ Pitch (F0 Hz)         │ │   │   - Prosody Cross-Attention             │   │
│   │   │ Para (50 classes)     │ │   │     (sees CTC emotion + pitch)          │   │
│   │   │ Phonemes (178 Misaki) │ │   │                                         │   │
│   │   │ Speaker Embed (256)   │ │   │   Outputs:                              │   │
│   │   │ Language (100)        │ │   │   - Text tokens + timestamps            │   │
│   │   └───────────────────────┘ │   │   - Punctuation (!?.,)                  │   │
│   │                             │   │   - Emotion (refined)                   │   │
│   │   All outputs @ 50Hz        │   │   - Phoneme deviation score             │   │
│   │   Frame-aligned             │   │   - Confidence scores                   │   │
│   └──────────────┬──────────────┘   └──────────────────┬──────────────────────┘   │
│                  │                                      │                          │
│                  │         Prosody Conditioning         │                          │
│                  │    (emotion_seq, pitch_seq) ────────►│                          │
│                  │                                      │                          │
│                  └──────────────────┬───────────────────┘                          │
│                                     │                                               │
│                                     ▼                                               │
│   ╔═════════════════════════════════════════════════════════════════════════════╗   │
│   ║                    LAYER 3: DUAL-STREAM FUSION                               ║   │
│   ╚═════════════════════════════════════════════════════════════════════════════╝   │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                    DUAL-STREAM CONSUMER                                      │   │
│   │                                                                              │   │
│   │   ✅ Production (RichStreamConsumer)                                         │   │
│   │                                                                              │   │
│   │   Event Types:                                                               │   │
│   │   ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐   │   │
│   │   │   TOKEN     │   CONFIRM   │    DIFF     │  BACKTRACK  │    FINAL    │   │   │
│   │   │  (CTC new)  │(decoder ok) │(decoder fix)│ (CTC revise)│ (committed) │   │   │
│   │   └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘   │   │
│   │                                                                              │   │
│   │   Timeline:                                                                  │   │
│   │   t=0ms    CTC: token "Hello" (provisional)                                 │   │
│   │   t=20ms   CTC: token "<|LAUGH|>" (provisional)                             │   │
│   │   t=200ms  Decoder: confirm "Hello" (final)                                 │   │
│   │   t=220ms  Decoder: confirm "<|LAUGH|>" (final)                             │   │
│   │                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                               │
│                                     ▼                                               │
│   ╔═════════════════════════════════════════════════════════════════════════════╗   │
│   ║                    LAYER 4: POST-PROCESSING                                  ║   │
│   ╚═════════════════════════════════════════════════════════════════════════════╝   │
│                                     │                                               │
│   ┌─────────────┬───────────────────┼───────────────────┬─────────────┐            │
│   ▼             ▼                   ▼                   ▼             ▼            │
│ ┌───────────┐ ┌───────────┐ ┌─────────────────┐ ┌───────────┐ ┌───────────────┐   │
│ │ SPEAKER   │ │ CUSTOM    │ │  HALLUCINATION  │ │UNRECOG    │ │  ADAPTATION   │   │
│ │ BUFFER    │ │ VOCAB     │ │   DETECTION     │ │WORD MEM   │ │  DATA         │   │
│ │           │ │           │ │                 │ │           │ │  COLLECTION   │   │
│ │ ✅ Design │ │ ✅ Design │ │  ✅ Production  │ │ ✅ Design │ │  ✅ Design    │   │
│ │           │ │           │ │                 │ │           │ │               │   │
│ │ Track IDs │ │ Hotword   │ │ Phoneme verify  │ │ Cluster   │ │ Per-speaker   │   │
│ │ EMA embed │ │ boosting  │ │ 55.6% recall    │ │ unknowns  │ │ fine-tune     │   │
│ └───────────┘ └───────────┘ └─────────────────┘ └───────────┘ └───────────────┘   │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         RICH TOKEN OUTPUT                                    │   │
│   │                                                                              │   │
│   │   {                                                                          │   │
│   │     "alignment_id": "a1",                                                    │   │
│   │     "stream": "decoder",                                                     │   │
│   │     "token": "Hello",                                                        │   │
│   │     "start_time_ms": 0.0,                                                    │   │
│   │     "end_time_ms": 320.0,                                                    │   │
│   │     "confidence": 0.95,                                                      │   │
│   │     "language": "en",                                                        │   │
│   │     "emotion": "happy",                                                      │   │
│   │     "pitch_hz": 185.5,                                                       │   │
│   │     "phonemes": ["h", "ə", "l", "oʊ"],                                       │   │
│   │     "phoneme_deviation": 0.05,                                               │   │
│   │     "para_class": null,                                                      │   │
│   │     "speaker_id": 0,                                                         │   │
│   │     "speaker_embedding": [0.12, -0.34, ...]                                  │   │
│   │   }                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Layer 0: Upstream Preprocessing

#### Silero VAD
```
Status:     ✅ Production
Location:   tools/whisper_mlx/upstream/vad.py (or external)
Latency:    ~2ms
Purpose:    Voice activity detection, speech/silence/music classification
Output:     speech_prob (float), is_speech (bool)
```

#### Overlap Detector
```
Status:     ❌ Need to Train
Location:   tools/whisper_mlx/upstream/overlap_detector.py (planned)
Latency:    ~5ms (target)
Purpose:    Count simultaneous speakers per frame
Output:     num_speakers (0, 1, 2, or 3)
Architecture: Small CNN on mel features → softmax over [0,1,2,3]
Training:   LibriMix, VoxConverse, LibriCSS
```

#### MossFormer2 MLX (Source Separation)
```
Status:     ✅ Production (third-party, tested)
Location:   tools/third_party/mossformer_ss_mlx/
Latency:    ~30ms per 100ms chunk
Speed:      9.5× real-time (warm)
Quality:    ~21 dB SI-SDRi (SOTA for available models)
Models:     2spk (16kHz), 3spk (8kHz), WHAMR (8kHz)
License:    Apache 2.0
Source:     github.com/starkdmi/mossformer_ss_mlx
```

---

### Layer 1: Whisper Encoder

```
Status:     ✅ Production (frozen)
Location:   tools/whisper_mlx/model.py
Model:      Whisper large-v3
Parameters: 1.5B (frozen, not trainable)
Output:     1280-dimensional embeddings @ 50Hz (20ms per frame)
Latency:    ~45ms for encoder pass

Architecture:
┌─────────────────────────────────────────────────────────────┐
│  Audio (16kHz) → Mel Spectrogram (80-dim, 25ms window)     │
│       ↓                                                     │
│  Conv Stem (2 conv layers)                                 │
│       ↓                                                     │
│  32× Transformer Blocks                                     │
│  - Self-attention                                           │
│  - Feed-forward (4× hidden)                                 │
│  - LayerNorm                                                │
│       ↓                                                     │
│  Encoder Output: (batch, T, 1280) where T = audio_len/320  │
└─────────────────────────────────────────────────────────────┘
```

---

### Layer 2A: RichCTC Head (Streaming Path)

```
Status:     ✅ Production
Location:   tools/whisper_mlx/rich_ctc_head.py
Latency:    ~10ms
Output:     All heads @ 50Hz, frame-aligned
```

#### Sub-Heads:

| Head | Output Dim | Status | Accuracy | Training Data |
|------|------------|--------|----------|---------------|
| **Text CTC** | 51,865 | ✅ | 43.45% WER (greedy) | LibriSpeech |
| **Emotion** | 8 classes | ✅ | ~85% | RAVDESS, CREMA-D, MELD |
| **Pitch** | 1 (F0 Hz) | ✅ | - | MIR-1K, PTDB-TUG |
| **Paralinguistics** | 50 classes | ✅ | 96.96% | VocalSound, SEP-28k |
| **Phonemes (Kokoro)** | 178 Misaki | ✅ | 19.5% PER | LibriSpeech (MFA aligned) |
| **Speaker Embedding** | 256-dim | ✅ | - | VoxCeleb-style |
| **Language** | 100 classes | ✅ | 98.61% | CommonVoice, OpenSLR |

#### Paralinguistics Classes (50):
```python
# Universal Non-Verbal (0-10)
speech, laughter, cough, sigh, breath, cry, yawn, throat_clear, sneeze, gasp, groan

# English Fillers (11-15)
um_en, uh_en, hmm_en, er_en, ah_en

# Chinese Fillers (16-19)
nage_zh, zhege_zh, jiushi_zh, en_zh

# Japanese Fillers (20-24)
eto_ja, ano_ja, ee_ja, maa_ja, un_ja

# Korean Fillers (25-28)
eo_ko, eum_ko, geuge_ko, mwo_ko

# Hindi Fillers (29-32)
matlab_hi, wo_hi, yeh_hi, haan_hi

# Other Languages (33-39)
este_es, pues_es, euh_fr, ben_fr, aeh_de, also_de, yani_ar

# Singing Vocalizations (40-49)
sing_a, sing_e, sing_i, sing_o, sing_u, vibrato, trill, vocal_fry, falsetto, belt
```

#### Phoneme Inventory (178 Misaki):
```
Based on Misaki G2P (hexgrad/misaki)
- IPA-based phoneme set
- Covers English, Japanese, Chinese
- Used for hallucination detection
- Used for lip sync / TTS alignment
```

---

### Layer 2B: RichDecoder (Refinement Path)

```
Status:     🔄 Training (82.34% accuracy)
Location:   tools/whisper_mlx/rich_decoder.py
Latency:    ~150-200ms
Purpose:    High-quality text + refinement of CTC outputs
```

#### Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                     RICH DECODER                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Whisper Decoder (frozen)                                   │
│       │                                                      │
│       ├── LoRA Adapters (trainable, rank=8)                 │
│       │   - q_proj, v_proj, fc1, fc2                        │
│       │   - Last 12 layers only                             │
│       │   - ~4M trainable params                            │
│       │                                                      │
│       └── Prosody Cross-Attention (NEW)                     │
│           - Attends to CTC emotion_seq (8-dim)              │
│           - Attends to CTC pitch_seq (1-dim)                │
│           - Enables prosody-aware decoding                  │
│                                                              │
│  Output Heads:                                               │
│  ├── Text Logits (51,865 + 50 para tokens = 51,915)        │
│  ├── Emotion (8 classes, refined)                           │
│  ├── Phoneme Deviation (hallucination score)                │
│  └── Confidence Scores                                       │
└─────────────────────────────────────────────────────────────┘
```

---

### Layer 3: Dual-Stream Fusion

```
Status:     ✅ Production
Location:   tools/whisper_mlx/dual_stream.py
Purpose:    Merge CTC (fast) and Decoder (accurate) streams
```

#### Stream Events:
```python
class EventType(Enum):
    TOKEN = "token"         # CTC emits new provisional token
    CONFIRM = "confirm"     # Decoder confirms CTC was correct
    DIFF = "diff"           # Decoder corrects CTC
    BACKTRACK = "backtrack" # CTC revises previous output
    FINAL = "final"         # Token committed, won't change

class StreamEvent:
    event_type: EventType
    alignment_id: str       # Links CTC ↔ Decoder
    stream: str             # "ctc" or "decoder"
    timestamp_ms: float
    token: Optional[RichToken]
    diff: Optional[Dict]    # {"field": "token", "ctc": "their", "decoder": "there"}
```

#### Consumer Logic:
```python
class RichStreamConsumer:
    def on_ctc_token(self, event):
        # Show immediately (provisional)
        self.display(event.token, provisional=True)

    def on_decoder_confirm(self, event):
        # Mark as final
        self.mark_final(event.alignment_id)

    def on_decoder_diff(self, event):
        # Apply correction
        self.apply_diff(event.alignment_id, event.diff)

    def on_backtrack(self, event):
        # Remove tokens after backtrack point
        self.remove_after(event.backtrack_to_id)
```

---

### Layer 4: Post-Processing

#### Speaker Buffer
```
Status:     ✅ Designed
Purpose:    Track and assign consistent speaker IDs across time
Method:     Cosine similarity of speaker embeddings + EMA update
Threshold:  0.7 for same-speaker match
```

#### Custom Vocabulary
```
Status:     ✅ Designed
Purpose:    Hotword boosting for names, jargon, domain terms
Method:     Trie-based prefix matching → logit biasing during decode
Boost:      +5.0 logits for matching tokens
```

#### Hallucination Detection
```
Status:     ✅ Production
Location:   tools/whisper_mlx/kokoro_phoneme_head.py
Method:     Compare CTC phonemes vs expected phonemes from decoder text
Metrics:    55.6% recall, 15% FPR
Threshold:  similarity < 0.7 → don't commit token
```

#### Unrecognized Word Memory
```
Status:     ✅ Designed
Purpose:    Track recurring unknown words for vocabulary updates
Method:     Cluster by phoneme hash, collect context, export candidates
```

#### Adaptation Data Collection
```
Status:     ✅ Designed
Purpose:    Collect per-speaker data for fine-tuning
Output:     SpeakerAdaptationData (audio, transcript, embedding, quality signals)
```

---

## Performance Metrics

### Latency Budget

#### Single Speaker (Fast Path):
| Stage | Latency | Cumulative |
|-------|---------|------------|
| VAD | 2ms | 2ms |
| Overlap Detection | 5ms | 7ms |
| Whisper Encoder | 45ms | 52ms |
| CTC Head | 10ms | **62ms** |
| Decoder (async) | +150ms | **212ms** |

#### Multi-Speaker (Separation Path):
| Stage | Latency | Cumulative |
|-------|---------|------------|
| VAD | 2ms | 2ms |
| Overlap Detection | 5ms | 7ms |
| MossFormer2 | 30ms | 37ms |
| Whisper Encoder | 45ms | 82ms |
| CTC Head | 10ms | **92ms** |
| Decoder (async) | +150ms | **242ms** |

### Quality Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| CTC Streaming Latency | <100ms | 62ms | ✅ |
| CTC RTF | <0.2 | 0.092 | ✅ (12.5× RT) |
| CTC Greedy WER | - | 43.45% | Expected (no LM) |
| Decoder WER | <20% | ~17% (training) | 🔄 |
| Emotion Accuracy | >80% | ~85% | ✅ |
| Paralinguistics Accuracy | >75% | 96.96% | ✅ |
| Language ID Accuracy | >90% | 98.61% | ✅ |
| Phoneme PER | <15% | 19.5% | ⚠️ |
| Hallucination Detection | >50% | 55.6% | ✅ |
| Source Separation SI-SDRi | >12dB | ~21dB | ✅ |

---

## File Structure

```
tools/whisper_mlx/
├── model.py                    # Whisper encoder/decoder
├── rich_ctc_head.py            # All CTC heads combined
├── rich_decoder.py             # LoRA decoder with prosody
├── kokoro_phoneme_head.py      # Phoneme head (hallucination)
├── dual_stream.py              # StreamEvent, RichStreamConsumer
├── confidence_calibration.py   # Temperature/Platt scaling
├── prosody_beam_search.py      # Punctuation from prosody
├── demo_rich_audio.py          # Terminal visualization
│
├── heads/
│   ├── emotion.py
│   ├── pitch.py
│   ├── paralinguistics.py
│   ├── language.py
│   └── speaker.py
│
├── upstream/
│   ├── vad.py                  # Silero VAD wrapper
│   ├── separator.py            # MossFormer2 wrapper (planned)
│   └── overlap_detector.py     # (planned)
│
├── train_*.py                  # Training scripts
└── benchmark_*.py              # Benchmark scripts

tools/third_party/
└── mossformer_ss_mlx/          # Source separation (Apache 2.0)

models/
├── kokoro_phoneme_head/        # Production phoneme head
├── sota/                       # Downloaded SOTA models for distillation
└── checkpoints/                # Training checkpoints

data/
├── LibriSpeech/                # ASR training
├── emotion/                    # RAVDESS, CREMA-D, etc.
├── paralinguistics/            # VocalSound, SEP-28k
├── singing/                    # VocalSet, OpenCPOP
├── augmentation/               # MUSAN, RIRS_NOISES
├── separation/                 # LibriMix (generating)
├── diarization/                # VoxConverse (downloading)
└── multilingual/               # OpenSLR, CommonVoice
```

---

## Training Pipeline

### Current Training Status

| Model | Status | Checkpoint | Next Steps |
|-------|--------|------------|------------|
| RichCTC (all heads) | ✅ Production | checkpoints/rich_ctc/ | - |
| Kokoro Phoneme | ✅ Production | models/kokoro_phoneme_head/ | - |
| Paralinguistics | ✅ Production | checkpoints/paralinguistics_v3/ | - |
| Language Head | ✅ Production | checkpoints/language_head_v1/ | - |
| Emotion (distilled) | ✅ Done | checkpoints/emotion_distilled_v2/ | - |
| RichDecoder | 🔄 82.34% | checkpoints/rich_decoder/ | Complete training |
| Overlap Detector | ❌ Not started | - | Download data, train |

### Data Requirements

| Task | Current | Needed | Gap |
|------|---------|--------|-----|
| Source Separation | 0 | LibriMix (~100GB) | Download + generate |
| Overlap Detection | 0 | VoxConverse, LibriCSS | Download |
| Emotion | 87K | 200K+ | Pseudo-label |
| Paralinguistics | 31K | 50K+ | - |
| Phoneme | TIMIT | 100K+ | MFA alignment |

---

## Integration: Kokoro TTS Fusion

```
┌─────────────────────────────────────────────────────────────┐
│                 STT ↔ TTS INTEGRATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SHARED COMPONENTS:                                          │
│  ├── Phoneme representation (178 Misaki)                    │
│  ├── Speaker embeddings (256-dim)                           │
│  └── Prosody features (pitch, emotion)                      │
│                                                              │
│  USE CASES:                                                  │
│  1. Voice Cloning: STT speaker_embed → TTS voice selection  │
│  2. Pronunciation: STT phonemes → TTS demo correct form     │
│  3. Emotion Transfer: STT emotion → TTS expressiveness      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## References

- Whisper: https://github.com/openai/whisper
- MLX: https://ml-explore.github.io/mlx/
- MossFormer2: https://github.com/alibabasglab/MossFormer2
- MossFormer2 MLX: https://github.com/starkdmi/mossformer_ss_mlx
- Misaki G2P: https://github.com/hexgrad/misaki
- LoRA: https://arxiv.org/abs/2106.09685

---

## Document History

| Date | Change |
|------|--------|
| 2026-01-02 | Created comprehensive architecture doc |
| 2026-01-02 | Added MossFormer2 MLX (replaces Conv-TasNet) |
| 2026-01-02 | Consolidated from multiple design docs |
