# TTS Backend Integration Guide

**Mission**: Systematically install, test, and evaluate all TTS solutions (SaaS + Self-hosted) for EN/JA/ZH.

---

## Progress Tracker

### SOTA Models (Highest Priority)

| Backend | Status | Japanese | English | Chinese | Latency | Notes |
|---------|--------|----------|---------|---------|---------|-------|
| **StyleTTS2** | TODO | No | SOTA | No | ~100ms | MOS 4.3+, Python wrapper needed |
| **Fish-Speech** | SETUP | Yes | SOTA | Yes | ~200ms | #1 TTS-Arena2, gated repo |
| **VoiceCraft-X** | TODO | Yes | Yes | Yes | ~500ms | 11 langs, speech editing, CC-BY-NC |
| **Tortoise-TTS** | TODO | No | Excellent | No | ~30s | Slow but very high quality |

### Production Ready

| Backend | Status | Japanese | English | Chinese | Latency | Notes |
|---------|--------|----------|---------|---------|---------|-------|
| Kokoro | DONE | 100% | 100% | Yes | 1.6s | Best working solution |
| MeloTTS | DONE | 77-100% | 100% | Yes | 1s | Long vowel issues |
| VOICEVOX | DONE | 100% | No | No | 700-900ms | Japanese-native, C++ integrated |
| GPT-SoVITS | TODO | Yes | Yes | Yes | ~500ms | Voice cloning, 0.5s M4 |
| Parler-TTS | TODO | No | Good | No | ~2s | Controllable style |
| Bark | TODO | Yes | Good | Yes | ~10s | Expressive, multilingual |

### Fast/Lightweight

| Backend | Status | Japanese | English | Chinese | Latency | Notes |
|---------|--------|----------|---------|---------|---------|-------|
| WhisperSpeech | TODO | Planned | Yes | No | ~80ms | 12x realtime, EN only now |
| Dia | TODO | No | Good | No | ~100ms | Fast English |
| F5-TTS | TODO | Check | Yes | Yes | ~150ms | Diffusion transformer |

### SaaS Providers

| Backend | Status | Japanese | English | Chinese | Latency | Notes |
|---------|--------|----------|---------|---------|---------|-------|
| ElevenLabs | TODO | Yes | Excellent | Yes | ~200ms | Industry leader |
| OpenAI TTS | VERIFIED | 87% | Excellent | Yes | ~2-3s | Easy API, tested working |
| Azure TTS | TODO | Yes | Excellent | Yes | ~200ms | Enterprise, many voices |
| Google TTS | TODO | Yes | WaveNet | Yes | ~200ms | Good coverage |
| iFLYTEK | TODO | No | No | Native | ~150ms | Best Mandarin |

### Fallback

| Backend | Status | Japanese | English | Chinese | Latency | Notes |
|---------|--------|----------|---------|---------|---------|-------|
| espeak-ng | FALLBACK | Low | Low | Low | 10ms | Emergency only |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TTS Evaluation System                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │   SaaS Providers    │    │  Self-Hosted Engines │                    │
│  ├─────────────────────┤    ├─────────────────────┤                    │
│  │ • ElevenLabs        │    │ • Kokoro ✓          │                    │
│  │ • OpenAI TTS        │    │ • MeloTTS ✓         │                    │
│  │ • Azure Neural TTS  │    │ • VOICEVOX ✓        │                    │
│  │ • Google Cloud TTS  │    │ • Fish-Speech       │                    │
│  │ • iFLYTEK (ZH)      │    │ • GPT-SoVITS        │                    │
│  └─────────────────────┘    │ • Dia / Dia-Multi   │                    │
│           │                  │ • StyleTTS2         │                    │
│           │                  └─────────────────────┘                    │
│           │                           │                                 │
│           └───────────┬───────────────┘                                 │
│                       ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Unified TTS Provider Interface                      │   │
│  │  synthesize(text, lang, voice_id) → audio_bytes                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                       │                                                 │
│                       ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Evaluation Harness                                  │   │
│  │  • STT Verification (Whisper)                                   │   │
│  │  • Latency Benchmarks                                           │   │
│  │  • Listening Tests (MOS)                                        │   │
│  │  • Decision Matrix                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
voice/
├── corpus/                    # Test sentences
│   ├── en.txt                # English test corpus
│   ├── ja.txt                # Japanese test corpus
│   └── zh.txt                # Chinese test corpus
├── providers/                 # Provider interface & clients
│   ├── base.py               # TtsProvider protocol
│   ├── elevenlabs_client.py
│   ├── openai_tts_client.py
│   ├── azure_tts_client.py
│   ├── google_tts_client.py
│   ├── iflytek_tts_client.py
│   └── voicevox_client.py
├── scripts/                   # TTS wrappers & utilities
│   ├── kokoro_tts.py         # ✓ DONE
│   ├── melotts_tts.py        # ✓ DONE
│   ├── voicevox_tts.py
│   ├── fishspeech_tts.py
│   ├── gptsovits_tts.py
│   ├── dia_tts.py
│   ├── batch_synthesize.py   # Batch synthesis for all providers
│   └── tts_router.py         # Unified routing service
├── self_host/                 # Self-hosted engine installations
│   ├── voicevox/
│   ├── fish_speech/
│   ├── gpt_sovits/
│   └── dia/
├── results/                   # Evaluation outputs
│   ├── saas/                 # SaaS provider outputs
│   ├── self_host/            # Self-hosted outputs
│   ├── manifest.csv          # All audio files index
│   ├── ratings.csv           # Human ratings
│   └── decision_*.md         # Final recommendations
└── tests/
    ├── test_japanese_stt.py  # ✓ DONE
    ├── test_latency.py       # ✓ DONE
    └── test_tts_backends.py  # Unified backend tests
```

---

## PHASE 1: Self-Hosted Backends (Priority)

### Task 1: Kokoro - DONE
**Status**: 100% STT accuracy for both Japanese and English
**File**: `scripts/kokoro_tts.py`

### Task 2: VOICEVOX Backend - DONE
**Status**: 100% STT accuracy for Japanese, integrated with C++ pipeline
**File**: `scripts/voicevox_tts.py`
**C++ Integration**: `include/voicevox_subprocess_tts.hpp`, `src/voicevox_subprocess_tts.cpp`

**Test Results (Worker #42, 2025-12-04):**
- Japanese STT: 100% accuracy (both test phrases verified)
- Latency: ~700-900ms warm, ~2.5s cold (model load)
- Voice: ずんだもん ノーマル (style_id=3)

**Deliverables:**
- [x] Download VOICEVOX core for macOS ARM64
- [x] Download Japanese voice model (zundamon + shikoku metan)
- [x] Create `scripts/voicevox_tts.py`
- [x] Verify with Japanese STT (100% - exceeds 80% requirement)
- [x] Benchmark latency (700-900ms warm)
- [x] C++ subprocess integration via `VoicevoxSubprocessTTS` class
- [x] Config file: `config/voicevox_japanese.yaml`

### Task 3: Fish-Speech Backend - SETUP COMPLETE
**Why**: SOTA quality, #1 on TTS-Arena2, good Japanese
**Status**: Script created, awaiting model download (gated repo)

```bash
# 1. Install (done)
pip install fish-speech

# 2. Request access & login (manual step required)
# Go to: https://huggingface.co/fishaudio/openaudio-s1-mini
huggingface-cli login

# 3. Download model (~2GB)
huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir self_host/fish_speech/checkpoints/openaudio-s1-mini

# 4. Test
python scripts/fishspeech_tts.py "Hello world" -o /tmp/test.wav
```

**Deliverables:**
- [x] Install fish-speech
- [x] Create `scripts/fishspeech_tts.py`
- [ ] Download model (needs HuggingFace auth)
- [ ] Test with S1-mini model first
- [ ] Verify Japanese/English STT
- [ ] Benchmark latency

### Task 4: GPT-SoVITS Backend
**Why**: Voice cloning, JA/EN/ZH/KO support, 0.5s on M4

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

**Deliverables:**
- [ ] Clone and setup GPT-SoVITS
- [ ] Create `scripts/gptsovits_tts.py`
- [ ] Test voice cloning capability
- [ ] Verify JA/EN STT
- [ ] Benchmark on M4

### Task 5: Dia / Dia-Multilingual
**Why**: Fast English TTS, good quality

```bash
# Check: https://github.com/nari-labs/dia
pip install dia-tts  # or similar
```

**Deliverables:**
- [ ] Install Dia
- [ ] Create `scripts/dia_tts.py`
- [ ] Verify English STT
- [ ] Test multilingual variant if available

### Task 6: StyleTTS2 (SOTA English)
**Why**: MOS 4.3+ (state-of-the-art English quality), Python wrapper only

**Note**: StyleTTS2 doesn't support Japanese. Use for English only.

```bash
# Clone StyleTTS2
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
pip install -r requirements.txt

# Download pretrained model (LJSpeech)
# https://huggingface.co/yl4579/StyleTTS2-LJSpeech
```

**Deliverables:**
- [x] Create `scripts/export_styletts2.py` - TorchScript export script
- [ ] Clone and setup StyleTTS2
- [ ] Create `scripts/styletts2_tts.py` - Python wrapper
- [ ] Export modules to TorchScript for C++ loading
- [ ] Verify English STT (must be >= 95%)
- [ ] Benchmark latency

### Task 7: VoiceCraft-X (11 Languages)
**Why**: 11 languages including Japanese, speech editing capability

**License**: CC-BY-NC (non-commercial use only)

```bash
# Clone VoiceCraft-X
git clone https://github.com/voicecraft-x/voicecraft-x.git
cd voicecraft-x
pip install -r requirements.txt
```

**Deliverables:**
- [ ] Clone and setup VoiceCraft-X
- [ ] Create `scripts/voicecraft_tts.py`
- [ ] Verify Japanese/English STT
- [ ] Test speech editing capability
- [ ] Benchmark latency

### Task 8: Tortoise-TTS (Highest Quality)
**Why**: Exceptional quality for English, good for batch/offline

**Note**: Very slow (~30s per sentence), not for real-time use

```bash
pip install tortoise-tts
```

**Deliverables:**
- [ ] Install Tortoise-TTS
- [ ] Create `scripts/tortoise_tts.py`
- [ ] Verify English STT (should be >98%)
- [ ] Benchmark latency (for reference)
- [ ] Document as "batch quality reference"

---

## PHASE 2: SaaS Providers

### Provider Interface

Create `providers/base.py`:

```python
from typing import Protocol

class TtsProvider(Protocol):
    name: str

    async def synthesize(
        self,
        text: str,
        lang: str,
        voice_id: str = "default"
    ) -> bytes:
        """Return audio bytes (WAV format)."""
        ...

    def get_supported_languages(self) -> list[str]:
        ...

    def get_voices(self, lang: str) -> list[str]:
        ...
```

### Task 6: ElevenLabs
**Best for**: High-quality English, voice cloning

```python
# providers/elevenlabs_client.py
import httpx

class ElevenLabsProvider:
    name = "elevenlabs"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"

    async def synthesize(self, text: str, lang: str, voice_id: str) -> bytes:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={"xi-api-key": self.api_key},
                json={"text": text, "model_id": "eleven_multilingual_v2"}
            )
            return resp.content
```

**Deliverables:**
- [ ] Create `providers/elevenlabs_client.py`
- [ ] Test with API key
- [ ] Benchmark quality and latency

### Task 7: OpenAI TTS
**Best for**: Easy integration, good quality

```python
# providers/openai_tts_client.py
from openai import OpenAI

class OpenAITTSProvider:
    name = "openai"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def synthesize(self, text: str, lang: str, voice_id: str) -> bytes:
        response = self.client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd"
            voice=voice_id or "alloy",
            input=text
        )
        return response.content
```

**Deliverables:**
- [ ] Create `providers/openai_tts_client.py`
- [ ] Test voices: alloy, echo, fable, onyx, nova, shimmer
- [ ] Compare tts-1 vs tts-1-hd quality

### Task 8: Azure Neural TTS
**Best for**: Enterprise, many languages/voices

```python
# providers/azure_tts_client.py
import azure.cognitiveservices.speech as speechsdk

class AzureTTSProvider:
    name = "azure"
    # ... implementation
```

**Deliverables:**
- [ ] Create `providers/azure_tts_client.py`
- [ ] Test Japanese neural voices
- [ ] Benchmark latency

### Task 9: Google Cloud TTS
**Best for**: WaveNet voices, language coverage

```python
# providers/google_tts_client.py
from google.cloud import texttospeech

class GoogleTTSProvider:
    name = "google"
    # ... implementation
```

**Deliverables:**
- [ ] Create `providers/google_tts_client.py`
- [ ] Test WaveNet vs Standard voices
- [ ] Benchmark

### Task 10: iFLYTEK
**Best for**: Mandarin Chinese

```python
# providers/iflytek_tts_client.py
class IFLYTEKProvider:
    name = "iflytek"
    # Chinese-focused implementation
```

**Deliverables:**
- [ ] Create `providers/iflytek_tts_client.py`
- [ ] Test Mandarin synthesis
- [ ] Benchmark

---

## PHASE 3: Test Corpus

### Create Test Corpora

Create `corpus/en.txt`, `corpus/ja.txt`, `corpus/zh.txt` with:
- ~50 lines each
- Mix of: UI prompts, conversations, long sentences, numbers/dates
- Format: `<ID>\t<TEXT>`

Example `corpus/ja.txt`:
```
ja_001	こんにちは
ja_002	私は日本語を話します
ja_003	サーバーが起動しています
ja_004	エラーが発生しました。もう一度お試しください。
ja_005	2024年12月4日、午後3時30分
...
```

**Deliverables:**
- [ ] Create `corpus/en.txt` (50 lines)
- [ ] Create `corpus/ja.txt` (50 lines)
- [ ] Create `corpus/zh.txt` (50 lines)
- [ ] Create `scripts/load_corpus.py` utility

---

## PHASE 4: Batch Synthesis & Evaluation

### Batch Synthesis Script

Create `scripts/batch_synthesize.py`:

```python
#!/usr/bin/env python3
"""
Batch synthesize test corpus across all providers.

Usage:
    python scripts/batch_synthesize.py --providers kokoro,melotts --langs en,ja
"""

import asyncio
from pathlib import Path

async def batch_synthesize(providers: list, langs: list):
    for provider in providers:
        for lang in langs:
            corpus = load_corpus(lang)
            for id, text in corpus:
                audio = await provider.synthesize(text, lang)
                output = Path(f"results/{provider.name}/{lang}/{id}.wav")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(audio)
```

### Evaluation Harness

Create `scripts/evaluate_backends.py`:

```python
#!/usr/bin/env python3
"""
Evaluate all synthesized audio with STT and metrics.
"""

import whisper
import pandas as pd
from pathlib import Path

def evaluate_all():
    results = []

    for wav_file in Path("results").rglob("*.wav"):
        provider = wav_file.parts[-3]
        lang = wav_file.parts[-2]
        sentence_id = wav_file.stem

        # STT verification
        transcribed = whisper_transcribe(wav_file, lang)
        expected = get_expected_text(lang, sentence_id)
        accuracy = compute_accuracy(expected, transcribed, lang)

        # Latency (from metadata)
        latency = get_synthesis_latency(wav_file)

        results.append({
            "provider": provider,
            "lang": lang,
            "sentence_id": sentence_id,
            "accuracy": accuracy,
            "latency_ms": latency
        })

    df = pd.DataFrame(results)
    df.to_csv("results/evaluation.csv", index=False)

    # Generate summary
    summary = df.groupby(["provider", "lang"]).agg({
        "accuracy": ["mean", "min"],
        "latency_ms": ["mean", "p50", "p95"]
    })
    print(summary)
```

---

## PHASE 5: Unified TTS Router

### Router Service

Create `scripts/tts_router.py`:

```python
#!/usr/bin/env python3
"""
Unified TTS Router - routes to best provider per language.

Usage:
    uvicorn scripts.tts_router:app --port 8080
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
from pathlib import Path

app = FastAPI()

# Routing config (update based on evaluation results)
ROUTING = {
    "en": {"realtime": "kokoro", "batch": "elevenlabs"},
    "ja": {"realtime": "kokoro", "batch": "voicevox"},
    "zh": {"realtime": "iflytek", "batch": "fish_speech"},
}

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    persona: str = "default"
    priority: str = "realtime"  # or "batch"

class TTSResponse(BaseModel):
    audio_url: str
    provider: str
    cached: bool

@app.post("/tts", response_model=TTSResponse)
async def synthesize(req: TTSRequest):
    # Check cache
    cache_key = hashlib.md5(f"{req.lang}:{req.persona}:{req.text}".encode()).hexdigest()
    cache_path = Path(f"results/cache/{cache_key}.wav")

    if cache_path.exists():
        return TTSResponse(audio_url=str(cache_path), provider="cache", cached=True)

    # Route to provider
    provider_name = ROUTING.get(req.lang, {}).get(req.priority, "kokoro")
    provider = get_provider(provider_name)

    audio = await provider.synthesize(req.text, req.lang)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(audio)

    return TTSResponse(audio_url=str(cache_path), provider=provider_name, cached=False)
```

---

## Quality Gate

Every backend MUST pass before integration:

```bash
# Generate audio
python scripts/{backend}_tts.py "私は日本語を話します" -o /tmp/test.wav -l ja

# Verify with STT (must be >= 80%)
python tests/test_japanese_stt.py /tmp/test.wav "私は日本語を話します" medium

# Benchmark latency
python tests/test_latency.py --command "python scripts/{backend}_tts.py {text} -o {output} -l ja"
```

---

## Commit Protocol

Each backend = ONE commit:

```
[W]#N: Add {Backend} TTS backend

**Deliverables:**
- scripts/{backend}_tts.py (or providers/{backend}_client.py for SaaS)
- STT verification results
- Latency benchmarks

**Test Results:**
- Japanese STT: X% accuracy
- English STT: X% accuracy
- Chinese STT: X% accuracy (if supported)
- Latency: Xms (warm), Xms (cold)

**Next AI:** Add next backend from priority list
```

---

## Priority Order

### Immediate (Self-Hosted)
1. **VOICEVOX** - Japanese-native, potential C++ integration
2. **Fish-Speech** - SOTA quality, multi-language
3. **GPT-SoVITS** - Voice cloning

### Next (SaaS)
4. **ElevenLabs** - High quality, voice cloning
5. **OpenAI TTS** - Easy integration
6. **Azure TTS** - Enterprise, many voices
7. **Google TTS** - WaveNet quality
8. **iFLYTEK** - Best for Mandarin

### Later
9. **Dia/Dia-Multilingual** - English focused
10. **F5-TTS** - Alternative high-quality

---

## Config Template

Create `config.example.yaml`:

```yaml
# API Keys (SaaS providers)
elevenlabs:
  api_key: "your-key-here"

openai:
  api_key: "your-key-here"

azure:
  subscription_key: "your-key-here"
  region: "eastus"

google:
  credentials_file: "path/to/credentials.json"

iflytek:
  app_id: "your-app-id"
  api_key: "your-key-here"
  api_secret: "your-secret-here"

# Self-hosted endpoints
voicevox:
  endpoint: "http://localhost:50021"

fish_speech:
  endpoint: "http://localhost:8000"

# Evaluation settings
evaluation:
  whisper_model: "medium"
  stt_threshold: 0.80
```

---

## Current Status

**Working:**
- Kokoro: 100% JA, 100% EN, 1.6s latency (BEST self-hosted)
- MeloTTS: 77-100% JA, 100% EN, 1s latency
- VOICEVOX: 100% JA, 700-900ms latency (Japanese-native, C++ integrated)
- OpenAI TTS: 87% JA, Excellent EN, 2-3s latency (VERIFIED with API key)
- Quality gates infrastructure
- STT verification tests
- Golden audio fixtures system
- TorchScript export script for StyleTTS2

**Infrastructure Added:**
- `providers/base.py` - TtsProvider protocol
- `providers/openai_tts_client.py` - OpenAI TTS client
- `scripts/export_styletts2.py` - TorchScript export for StyleTTS2
- `scripts/generate_golden_fixtures.py` - Generate reference audio
- `tests/golden_fixtures/README.md` - Fixture documentation
- `scripts/voicevox_tts.py` - VOICEVOX Python wrapper (Worker #42)
- `include/voicevox_subprocess_tts.hpp` - VOICEVOX C++ header (Worker #42)
- `src/voicevox_subprocess_tts.cpp` - VOICEVOX C++ impl (Worker #42)
- `scripts/styletts2_tts.py` - StyleTTS2 Python wrapper (Worker #44) - BROKEN
- `scripts/fishspeech_tts.py` - Fish-Speech Python wrapper (Worker #44) - needs model
- `config/voicevox_japanese.yaml` - VOICEVOX config (Worker #42)

## BLOCKERS

### 1. StyleTTS2 Inference Quality (CRITICAL) - UNRELIABLE
**Status**: UNRELIABLE - Output quality is inconsistent (Worker #45 investigation)

**Symptom**: Audio output is unpredictable. Some texts work, others produce completely wrong audio.

| Text | Expected | Got | Result |
|------|----------|-----|--------|
| "Hello" | Hello | "what's up?" | FAIL |
| "Hello world" | Hello world | "tavao" | FAIL |
| "Good morning everyone" | Good morning everyone | (correct) | PASS |
| Official demo text (long) | StyleTTS 2 is a... | (correct) | PASS |

**Root cause analysis (Worker #45)**:
- Model architecture is correct (dimensions match)
- Checkpoint loads without errors
- F0/N prediction dimensions handled correctly by decoder
- Issue appears to be checkpoint corruption or version mismatch
- Checkpoint has `style_predictor` key not in model (suspicious)

**Detailed report**: `reports/main/styletts2_investigation_2025-12-04.md`

**Workaround**: Use verified working backends:
- **Kokoro**: 100% English accuracy, 100% Japanese, 1.6s latency
- **VOICEVOX**: 100% Japanese accuracy, 700-900ms latency
- **macOS TTS**: High quality, 666ms latency (default for EN->JA)

**Next steps** (LOW PRIORITY):
1. Download fresh checkpoint from HuggingFace
2. Verify checkpoint hash against official release
3. Test with PyTorch 2.0 (current is 2.6+)

---

## WORKER TASK LIST (Priority Order)

### Task 1: Fix StyleTTS2 Symlink
```bash
cd /Users/ayates/voice/models/styletts2
rm modules
cp -r Models/LJSpeech/modules .
ls -la modules/  # Verify bert.pt exists
```

### Task 2: Test StyleTTS2 C++ Integration
```bash
cd /Users/ayates/voice/stream-tts-cpp
./build/stream-tts-cpp --daemon config/default.yaml &
sleep 10
./build/stream-tts-cpp --speak "The quick brown fox jumps over the lazy dog"
./build/stream-tts-cpp --stop
```

**Verify with STT:**
```bash
cd /Users/ayates/voice
source .venv/bin/activate
# Generate test audio
./stream-tts-cpp/build/stream-tts-cpp --tts "Hello world" -o /tmp/styletts2_test.wav
python tests/test_japanese_stt.py /tmp/styletts2_test.wav "Hello world" medium
# Should show >95% accuracy for English
```

### Task 3: Add Fish-Speech Backend (SOTA Quality)
```bash
source .venv/bin/activate
pip install fish-speech

# Create wrapper
cat > scripts/fishspeech_tts.py << 'SCRIPT'
#!/usr/bin/env python3
# Fish-Speech TTS wrapper - implement similar to kokoro_tts.py
SCRIPT

# Test with STT
python scripts/fishspeech_tts.py "こんにちは" -o /tmp/fish_ja.wav -l ja
python tests/test_japanese_stt.py /tmp/fish_ja.wav "こんにちは" medium
```

### Task 4: Generate Golden Audio Fixtures
```bash
source .venv/bin/activate
python scripts/generate_golden_fixtures.py --backend kokoro --langs en,ja
python scripts/generate_golden_fixtures.py --backend openai --langs en,ja
# Saves reference audio to tests/golden_fixtures/
```

---

## Quality Gates (ALL Must Pass)

| Language | STT Threshold | Test Command |
|----------|---------------|--------------|
| Japanese | ≥80% char | `python tests/test_japanese_stt.py <wav> <text> medium` |
| English | ≥95% word | Same command with English text |

---

## After Models Working: Performance Phase

Once all models pass quality gates:
1. Benchmark latency for each backend
2. Optimize hot paths in C++
3. Implement audio caching
4. Profile Metal GPU utilization

---

**WORKER: Start with Task 1 (StyleTTS2 symlink fix)**
