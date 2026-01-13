# Rust-Based GPU-Accelerated Streaming TTS System
## Technical Design & Implementation Roadmap

**Copyright 2025 Andrew Yates. All rights reserved.**

---

## 1. EXECUTIVE SUMMARY

### Objective
Build an extremely efficient, low-latency streaming system that:
1. Parses Claude Code's stream-json output in real-time
2. Filters and extracts Claude's text responses (not tool outputs)
3. Translates English → Japanese using **local GPU-accelerated models**
4. Synthesizes speech using **local high-quality TTS**
5. Streams audio output with < 500ms latency

### Why Rust
- **Zero-cost abstractions**: No runtime overhead
- **Memory safety**: No segfaults or data races
- **Concurrency**: Fearless parallelism with tokio async runtime
- **Performance**: 10-100x faster than Python for streaming workloads
- **GPU integration**: Excellent bindings for CUDA, ROCm, Metal
- **Small binaries**: 5-10MB vs 100s of MB for Python

### Performance Targets
- **Latency**: < 500ms from Claude text to audio output
- **Throughput**: Process 1000+ words/min without audio gaps
- **GPU utilization**: > 80% for translation + TTS
- **Memory**: < 2GB RAM (models loaded once)
- **CPU**: < 20% single core for I/O and coordination

---

## 2. ARCHITECTURE DESIGN

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Claude Code (stream-json)                  │
└───────────────────────────┬──────────────────────────────────┘
                            │ STDIN
┌───────────────────────────▼──────────────────────────────────┐
│              Rust Stream Parser (tokio::io)                   │
│  - Read JSON lines from stdin                                 │
│  - Filter for assistant text messages                         │
│  - Extract clean text (remove markdown, code blocks)          │
└───────────────────────────┬──────────────────────────────────┘
                            │ Channel (mpsc)
┌───────────────────────────▼──────────────────────────────────┐
│          Translation Pipeline (GPU Accelerated)               │
│  Model: NLLB-200 or Opus-MT (ONNX Runtime)                   │
│  - Sentence segmentation                                      │
│  - Batch translations (GPU)                                   │
│  - Stream translated segments                                 │
└───────────────────────────┬──────────────────────────────────┘
                            │ Channel (mpsc)
┌───────────────────────────▼──────────────────────────────────┐
│               TTS Pipeline (GPU/CPU Optimized)                │
│  Model: Piper (fast) or Coqui VITS (high quality)            │
│  - Convert text → mel spectrograms (GPU)                      │
│  - Vocoder → audio (GPU)                                      │
│  - Stream audio chunks                                        │
└───────────────────────────┬──────────────────────────────────┘
                            │ Audio Buffer
┌───────────────────────────▼──────────────────────────────────┐
│            Audio Output (rodio + cpal)                        │
│  - Buffer audio chunks                                        │
│  - Play seamlessly without gaps                               │
│  - Handle underruns gracefully                                │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Threading Model

```rust
// Main thread: Coordinate everything
// Thread 1 (tokio runtime): Read stdin, parse JSON
// Thread 2 (GPU pool): Translation inference
// Thread 3 (GPU pool): TTS inference
// Thread 4: Audio playback

use tokio::sync::mpsc;
use rayon::ThreadPool;

async fn main() {
    let (text_tx, text_rx) = mpsc::unbounded_channel();
    let (translated_tx, translated_rx) = mpsc::unbounded_channel();
    let (audio_tx, audio_rx) = mpsc::unbounded_channel();

    // Spawn tasks
    tokio::spawn(stdin_reader(text_tx));
    tokio::spawn(translator(text_rx, translated_tx));
    tokio::spawn(tts_synthesizer(translated_rx, audio_tx));
    tokio::spawn(audio_player(audio_rx));

    // Wait for ctrl-c
    signal::ctrl_c().await.unwrap();
}
```

### 2.3 Data Flow

1. **Input**: Claude stream-json → Rust binary via stdin
2. **Parse**: Extract text from JSON, filter noise
3. **Segment**: Split into sentences for translation
4. **Translate**: GPU-accelerated batch translation
5. **Synthesize**: GPU-accelerated TTS (parallel with translation)
6. **Play**: Stream audio to system output

---

## 3. TECHNOLOGY STACK

### 3.1 Translation Layer

#### Option A: NLLB-200 (Recommended)
**Model**: `facebook/nllb-200-distilled-600M`
**Format**: ONNX (optimized for inference)
**Accuracy**: State-of-the-art (BLEU 25+ for en→ja)
**Speed**: ~50ms/sentence on GPU

**Pros**:
- Best quality for Japanese translation
- Supports 200 languages (future-proof)
- Active development by Meta
- Good ONNX export

**Cons**:
- 600MB model size (distilled version)
- Requires ONNX Runtime with GPU support

**Rust Integration**:
```rust
use ort::{Session, GraphOptimizationLevel, ExecutionProvider};

let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_execution_providers([ExecutionProvider::CUDA(0)])?
    .commit_from_file("models/nllb-200-distilled-600m.onnx")?;
```

#### Option B: Opus-MT (Lightweight Alternative)
**Model**: `Helsinki-NLP/opus-mt-en-ja`
**Format**: ONNX
**Accuracy**: Good (BLEU 20+)
**Speed**: ~20ms/sentence on GPU

**Pros**:
- Smaller model (300MB)
- Faster inference
- Easier to set up

**Cons**:
- Lower quality than NLLB
- Single language pair only

#### Option C: Local LLM Translation (Future)
**Model**: Qwen2.5 3B or Llama 3.2 3B with translation fine-tune
**Quality**: Best possible
**Speed**: ~100-200ms/sentence
**Note**: Overkill for now, but option for future

**Recommendation**: Start with NLLB-200, benchmark, fall back to Opus-MT if needed.

### 3.2 TTS Layer

#### Option A: Piper TTS (Recommended for Speed)
**Models**: `en_US-lessac-medium`, custom Japanese models
**Format**: ONNX
**Quality**: Good (not best)
**Speed**: Real-time factor < 0.1 (10x faster than real-time)

**Pros**:
- Extremely fast
- Small models (50-100MB)
- Good quality for the speed
- Easy integration

**Cons**:
- Not the absolute best quality
- Limited voice selection

**Rust Integration**:
```rust
// Piper has a Rust library
use piper::PiperTTS;

let tts = PiperTTS::new("models/ja_JP-nanami-medium.onnx")?;
let audio = tts.synthesize("こんにちは")?;
```

#### Option B: Coqui VITS (Recommended for Quality)
**Models**: Custom VITS models for Japanese
**Format**: PyTorch → ONNX export
**Quality**: Excellent (near human)
**Speed**: Real-time factor 0.3-0.5

**Pros**:
- State-of-the-art quality
- Very natural sounding
- Customizable voice cloning

**Cons**:
- Slower than Piper
- Larger models (200-500MB)
- More complex setup

**Rust Integration**:
```rust
// Use ONNX Runtime
let session = Session::builder()?
    .with_execution_providers([ExecutionProvider::CUDA(0)])?
    .commit_from_file("models/vits_japanese.onnx")?;
```

#### Option C: StyleTTS2 (Future, Best Quality)
**Quality**: Best available
**Speed**: Slower (real-time factor ~1.0)
**Note**: Requires complex setup, overkill for now

**Recommendation**:
- Primary: Piper for low latency
- Secondary: Coqui VITS for high quality mode (configurable)

### 3.3 Audio Playback

**Library**: `rodio` + `cpal`
```rust
use rodio::{OutputStream, Sink};

let (_stream, stream_handle) = OutputStream::try_default()?;
let sink = Sink::try_new(&stream_handle)?;

// Stream audio chunks
sink.append(audio_source);
```

**Alternative**: `cpal` directly for lower-level control

### 3.4 Rust Crates

```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
ort = "1.16"  # ONNX Runtime
rodio = "0.17"
cpal = "0.15"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
clap = { version = "4.4", features = ["derive"] }
crossbeam-channel = "0.5"
parking_lot = "0.12"
once_cell = "1.19"

# Optional: GPU acceleration
cuda = "0.3"
```

---

## 4. DETAILED COMPONENT DESIGN

### 4.1 JSON Stream Parser

**File**: `src/parser.rs`

```rust
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

#[derive(Debug, Deserialize)]
struct StreamMessage {
    #[serde(rename = "type")]
    msg_type: String,
    content: Option<Vec<ContentBlock>>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

pub async fn parse_stdin(tx: mpsc::UnboundedSender<String>) -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        // Parse JSON
        if let Ok(msg) = serde_json::from_str::<StreamMessage>(&line) {
            // Filter for assistant text
            if let Some(content) = msg.content {
                for block in content {
                    if block.block_type == "text" {
                        if let Some(text) = block.text {
                            // Clean text (remove markdown, code blocks)
                            let clean = clean_text(&text);
                            if !clean.is_empty() {
                                tx.send(clean)?;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn clean_text(text: &str) -> String {
    // Remove markdown formatting
    let text = text.replace("**", "").replace("*", "");
    let text = text.replace("`", "");

    // Remove code blocks
    let mut result = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        if line.trim().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if !in_code_block {
            result.push_str(line);
            result.push('\n');
        }
    }

    // Remove URLs
    let re = regex::Regex::new(r"https?://\S+").unwrap();
    let result = re.replace_all(&result, "URL").to_string();

    result.trim().to_string()
}
```

### 4.2 Translation Module

**File**: `src/translation.rs`

```rust
use ort::{Session, Value, GraphOptimizationLevel, ExecutionProvider};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

pub struct Translator {
    session: Session,
    tokenizer: Tokenizer,
    src_lang: String,
    tgt_lang: String,
}

impl Translator {
    pub fn new(model_path: &str, src: &str, tgt: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ExecutionProvider::CUDA(0),
                ExecutionProvider::CPU(Default::default()),
            ])?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file("models/tokenizer.json")?;

        Ok(Self {
            session,
            tokenizer,
            src_lang: src.to_string(),
            tgt_lang: tgt.to_string(),
        })
    }

    pub fn translate(&self, text: &str) -> anyhow::Result<String> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids: Vec<i64> = encoding.get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        // Create input tensor
        let input_ids = ndarray::Array2::from_shape_vec(
            (1, input_ids.len()),
            input_ids,
        )?;

        // Run inference
        let outputs = self.session.run(ort::inputs![input_ids]?)?;

        // Decode output
        let output_ids: Vec<u32> = outputs[0]
            .try_extract_tensor::<i64>()?
            .iter()
            .map(|&id| id as u32)
            .collect();

        let translated = self.tokenizer.decode(&output_ids, true)?;

        Ok(translated)
    }
}

pub async fn translation_worker(
    mut rx: mpsc::UnboundedReceiver<String>,
    tx: mpsc::UnboundedSender<String>,
    translator: Translator,
) -> anyhow::Result<()> {
    while let Some(text) = rx.recv().await {
        // Segment into sentences
        let sentences = segment_sentences(&text);

        // Translate each sentence
        for sentence in sentences {
            match translator.translate(&sentence) {
                Ok(translated) => {
                    tx.send(translated)?;
                }
                Err(e) => {
                    tracing::error!("Translation error: {}", e);
                    // Fallback: send original
                    tx.send(sentence)?;
                }
            }
        }
    }

    Ok(())
}

fn segment_sentences(text: &str) -> Vec<String> {
    // Simple sentence segmentation
    text.split(&['.', '!', '?'][..])
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}
```

### 4.3 TTS Module

**File**: `src/tts.rs`

```rust
use ort::{Session, Value, ExecutionProvider};
use tokio::sync::mpsc;

pub struct TTSEngine {
    session: Session,
    sample_rate: u32,
}

impl TTSEngine {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                ExecutionProvider::CUDA(0),
                ExecutionProvider::CPU(Default::default()),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            sample_rate: 22050,
        })
    }

    pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Convert text to phonemes (implementation depends on model)
        let phonemes = text_to_phonemes(text);

        // Create input tensor
        let input_ids = ndarray::Array2::from_shape_vec(
            (1, phonemes.len()),
            phonemes,
        )?;

        // Run TTS inference
        let outputs = self.session.run(ort::inputs![input_ids]?)?;

        // Extract audio
        let audio: Vec<f32> = outputs[0]
            .try_extract_tensor::<f32>()?
            .iter()
            .copied()
            .collect();

        Ok(audio)
    }
}

pub async fn tts_worker(
    mut rx: mpsc::UnboundedReceiver<String>,
    tx: mpsc::UnboundedSender<Vec<f32>>,
    tts: TTSEngine,
) -> anyhow::Result<()> {
    while let Some(text) = rx.recv().await {
        match tts.synthesize(&text) {
            Ok(audio) => {
                tx.send(audio)?;
            }
            Err(e) => {
                tracing::error!("TTS error: {}", e);
            }
        }
    }

    Ok(())
}

fn text_to_phonemes(text: &str) -> Vec<i64> {
    // Simplified: convert text to phoneme IDs
    // Real implementation depends on TTS model
    text.chars()
        .map(|c| c as i64)
        .collect()
}
```

### 4.4 Audio Playback Module

**File**: `src/audio.rs`

```rust
use rodio::{OutputStream, Sink, Source};
use tokio::sync::mpsc;

pub struct AudioPlayer {
    sink: Sink,
    _stream: OutputStream,
    sample_rate: u32,
}

impl AudioPlayer {
    pub fn new(sample_rate: u32) -> anyhow::Result<Self> {
        let (stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;

        Ok(Self {
            sink,
            _stream: stream,
            sample_rate,
        })
    }

    pub fn play(&self, samples: Vec<f32>) {
        let source = SamplesBuffer::new(1, self.sample_rate, samples);
        self.sink.append(source);
    }
}

pub async fn audio_worker(
    mut rx: mpsc::UnboundedReceiver<Vec<f32>>,
    player: AudioPlayer,
) -> anyhow::Result<()> {
    while let Some(audio) = rx.recv().await {
        player.play(audio);
    }

    Ok(())
}
```

### 4.5 Configuration System

**File**: `src/config.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub translation: TranslationConfig,
    pub tts: TTSConfig,
    pub audio: AudioConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TranslationConfig {
    pub model: String,  // "nllb-200" or "opus-mt"
    pub model_path: String,
    pub source_lang: String,
    pub target_lang: String,
    pub use_gpu: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TTSConfig {
    pub engine: String,  // "piper" or "coqui"
    pub model_path: String,
    pub voice: String,
    pub speed: f32,
    pub use_gpu: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            translation: TranslationConfig {
                model: "nllb-200".to_string(),
                model_path: "models/nllb-200-distilled-600m.onnx".to_string(),
                source_lang: "eng_Latn".to_string(),
                target_lang: "jpn_Jpan".to_string(),
                use_gpu: true,
            },
            tts: TTSConfig {
                engine: "piper".to_string(),
                model_path: "models/ja_JP-nanami-medium.onnx".to_string(),
                voice: "nanami".to_string(),
                speed: 1.0,
                use_gpu: true,
            },
            audio: AudioConfig {
                sample_rate: 22050,
                buffer_size: 4096,
            },
        }
    }
}
```

---

## 5. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Day 1-2)

**Tasks**:
1. Create Rust project structure
   ```bash
   cargo new --bin voice-stream
   cd voice-stream
   ```

2. Add dependencies to `Cargo.toml`

3. Implement JSON stream parser
   - Read from stdin
   - Parse stream-json
   - Filter for text content
   - Unit tests

4. Implement text cleaning
   - Remove markdown
   - Remove code blocks
   - Remove URLs
   - Unit tests

**Deliverables**:
- Working stdin reader
- JSON parser with tests
- Text cleaner with tests

**Test**:
```bash
echo '{"type":"text","content":[{"type":"text","text":"Hello world"}]}' | cargo run
```

### Phase 2: Translation Integration (Day 3-4)

**Tasks**:
1. Download NLLB-200 model
   ```bash
   # Python script to export to ONNX
   python scripts/export_nllb.py
   ```

2. Integrate ONNX Runtime
   - Load model
   - Run inference
   - Benchmark latency

3. Implement sentence segmentation

4. Add translation worker

5. Test end-to-end translation

**Deliverables**:
- Working translation module
- ONNX model files
- Benchmark results

**Test**:
```bash
echo "Hello, how are you?" | cargo run | grep "こんにちは"
```

### Phase 3: TTS Integration (Day 5-6)

**Tasks**:
1. Download Piper models
   ```bash
   # Download Japanese voice
   wget https://...
   ```

2. Integrate Piper or ONNX TTS

3. Implement audio buffer

4. Add TTS worker

5. Test speech output

**Deliverables**:
- Working TTS module
- Audio playback
- End-to-end test

**Test**:
```bash
echo "Hello" | cargo run
# Should hear Japanese speech
```

### Phase 4: Optimization (Day 7-8)

**Tasks**:
1. Profile performance
   ```bash
   cargo flamegraph
   ```

2. Optimize GPU usage
   - Batch translations
   - Parallel TTS

3. Reduce latency
   - Stream audio chunks
   - Pipeline stages

4. Memory optimization
   - Reuse buffers
   - Efficient data structures

**Deliverables**:
- < 500ms latency
- > 80% GPU utilization
- Flame graphs

### Phase 5: Configuration & Polish (Day 9-10)

**Tasks**:
1. Implement config file
   - TOML format
   - CLI overrides

2. Add logging
   - tracing framework
   - Performance metrics

3. Error handling
   - Graceful fallbacks
   - Clear error messages

4. Documentation
   - README with setup
   - Model download scripts
   - Usage examples

**Deliverables**:
- Config system
- Comprehensive docs
- Release binary

---

## 6. MODEL ACQUISITION

### 6.1 Translation Model (NLLB-200)

**Script**: `scripts/export_nllb.py`
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Export to ONNX
ort_model = ORTModelForSeq2SeqLM.from_pretrained(
    model_id,
    export=True,
    provider="CUDAExecutionProvider"
)

# Save
ort_model.save_pretrained("models/nllb-200")
tokenizer.save_pretrained("models/nllb-200")
```

### 6.2 TTS Model (Piper)

**Download**:
```bash
mkdir -p models/piper
cd models/piper

# Japanese voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ja/ja_JP/nanami/medium/ja_JP-nanami-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ja/ja_JP/nanami/medium/ja_JP-nanami-medium.onnx.json
```

---

## 7. TESTING STRATEGY

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let input = "**Hello** world `code` here";
        let output = clean_text(input);
        assert_eq!(output, "Hello world code here");
    }

    #[test]
    fn test_sentence_segmentation() {
        let input = "First sentence. Second sentence! Third?";
        let sentences = segment_sentences(input);
        assert_eq!(sentences.len(), 3);
    }
}
```

### 7.2 Integration Tests

```bash
# Test full pipeline
echo '{"content":[{"type":"text","text":"Hello world"}]}' | \
  cargo run --release | \
  # Should hear Japanese audio
```

### 7.3 Performance Tests

```rust
#[bench]
fn bench_translation(b: &mut Bencher) {
    let translator = Translator::new("models/nllb-200.onnx", "en", "ja").unwrap();
    b.iter(|| {
        translator.translate("Hello world").unwrap()
    });
}
```

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 GPU Optimization

```rust
// Use CUDA streams for parallel inference
use cuda::stream::Stream;

let stream1 = Stream::new()?;
let stream2 = Stream::new()?;

// Translation on stream1
stream1.launch_translation(input1);

// TTS on stream2 (parallel)
stream2.launch_tts(input2);

// Synchronize
stream1.synchronize()?;
stream2.synchronize()?;
```

### 8.2 Memory Optimization

```rust
// Reuse buffers
use parking_lot::Mutex;
use once_cell::sync::Lazy;

static BUFFER_POOL: Lazy<Mutex<Vec<Vec<f32>>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

fn get_buffer(size: usize) -> Vec<f32> {
    BUFFER_POOL.lock()
        .pop()
        .unwrap_or_else(|| Vec::with_capacity(size))
}

fn return_buffer(mut buf: Vec<f32>) {
    buf.clear();
    BUFFER_POOL.lock().push(buf);
}
```

### 8.3 Latency Optimization

```rust
// Stream audio in chunks
const CHUNK_SIZE: usize = 1024;

fn stream_audio(audio: Vec<f32>, sink: &Sink) {
    for chunk in audio.chunks(CHUNK_SIZE) {
        let source = SamplesBuffer::new(1, 22050, chunk.to_vec());
        sink.append(source);
    }
}
```

---

## 9. CONFIGURATION EXAMPLES

### 9.1 High Quality Mode

**File**: `config/high_quality.toml`
```toml
[translation]
model = "nllb-200"
model_path = "models/nllb-200-distilled-600m.onnx"
source_lang = "eng_Latn"
target_lang = "jpn_Jpan"
use_gpu = true

[tts]
engine = "coqui"
model_path = "models/vits_japanese.onnx"
voice = "nanami-vits"
speed = 1.0
use_gpu = true

[audio]
sample_rate = 48000
buffer_size = 4096
```

### 9.2 Low Latency Mode

**File**: `config/low_latency.toml`
```toml
[translation]
model = "opus-mt"
model_path = "models/opus-mt-en-ja.onnx"
source_lang = "en"
target_lang = "ja"
use_gpu = true

[tts]
engine = "piper"
model_path = "models/ja_JP-nanami-medium.onnx"
voice = "nanami"
speed = 1.2
use_gpu = true

[audio]
sample_rate = 22050
buffer_size = 2048
```

---

## 10. DEPLOYMENT

### 10.1 Build Binary

```bash
# Release build with optimizations
cargo build --release

# The binary will be in target/release/voice-stream
# Size: ~5-10MB
```

### 10.2 Usage

```bash
# Basic usage
claude --output-format stream-json | ./voice-stream

# With config
./voice-stream --config config/high_quality.toml

# With worker script
# Modify run_worker.sh:
claude ... | tee "$LOG_FILE" | ./voice-stream
```

### 10.3 Model Setup

```bash
# Download models
./scripts/setup_models.sh

# This will:
# 1. Download NLLB-200 (600MB)
# 2. Download Piper Japanese voice (100MB)
# 3. Convert to ONNX if needed
# 4. Place in models/ directory
```

---

## 11. FUTURE ENHANCEMENTS

### 11.1 Voice Cloning
- Use Coqui TTS voice cloning
- Clone custom voices from audio samples
- Store voice profiles

### 11.2 Multi-Language
- Support more language pairs
- Auto-detect source language
- Mix languages in single stream

### 11.3 Advanced TTS Features
- Emotion control
- Speaking rate variation
- Prosody tuning

### 11.4 Web Interface
- WebSocket server
- Browser-based configuration
- Real-time visualization

---

## 12. ALTERNATIVE: HYBRID APPROACH

If pure Rust proves difficult for model loading, use hybrid:

**Architecture**:
- Rust: Parser, coordinator, audio playback
- Python subprocess: Translation + TTS (batched calls)
- Communication: Unix domain sockets or shared memory

**Benefits**:
- Easier model integration
- Still very fast
- Can optimize critical paths in Rust

**Tradeoff**:
- Slightly higher latency (~100ms)
- Larger memory footprint

---

## 13. SUCCESS CRITERIA

### Must Have
- ✅ < 500ms latency (text to audio)
- ✅ GPU acceleration working
- ✅ High quality Japanese speech
- ✅ No audio gaps/stuttering
- ✅ < 2GB memory usage

### Nice to Have
- ✅ < 300ms latency
- ✅ Multiple voice options
- ✅ Config file support
- ✅ Detailed logging
- ✅ Error recovery

### Excellence
- ✅ < 200ms latency
- ✅ > 90% GPU utilization
- ✅ Voice cloning support
- ✅ Multi-language support

---

## END OF DESIGN DOCUMENT

This document provides complete specifications for building a high-performance, GPU-accelerated streaming TTS system in Rust. An AI worker can follow this roadmap to implement the entire system in 10 days of focused development.

**Copyright 2025 Andrew Yates. All rights reserved.**
