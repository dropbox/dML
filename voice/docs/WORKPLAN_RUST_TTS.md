# WORKPLAN: Rust GPU-Accelerated Streaming TTS
## Autonomous Worker Implementation Guide

**Copyright 2025 Andrew Yates. All rights reserved.**

---

## WORKER INSTRUCTIONS

You are an AI worker implementing a high-performance streaming TTS system. Follow this workplan sequentially. Each phase builds on the previous one. Test thoroughly before moving to the next phase.

**Critical Rules**:
1. **Test everything**: Every component must have tests
2. **Benchmark continuously**: Measure latency at each stage
3. **GPU first**: Always try GPU before CPU fallback
4. **Handle errors gracefully**: Never crash, always log
5. **Document decisions**: Explain why you chose specific approaches

---

## PHASE 0: ENVIRONMENT SETUP

### Task 0.1: Verify System Requirements

**Action**: Check system has required tools
```bash
# Check Rust installation
rustc --version  # Should be 1.70+
cargo --version

# Check GPU availability
nvidia-smi  # For NVIDIA
# OR
rocm-smi  # For AMD
# OR
system_profiler SPDisplaysDataType  # For Apple Silicon

# Check CUDA toolkit (if NVIDIA)
nvcc --version

# Install ONNX Runtime with GPU support
# This is critical - document which GPU backend you're using
```

**Deliverable**: Document in `SYSTEM_INFO.md`:
- Rust version
- GPU type and memory
- CUDA/ROCm/Metal version
- OS and architecture

### Task 0.2: Create Project Structure

**Action**: Initialize Rust project
```bash
cd /Users/ayates/voice
cargo new --bin voice-stream-rust
cd voice-stream-rust

# Create directory structure
mkdir -p models scripts config tests benches
```

**Project structure**:
```
voice-stream-rust/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── parser.rs
│   ├── translation.rs
│   ├── tts.rs
│   ├── audio.rs
│   └── config.rs
├── models/
│   ├── translation/
│   └── tts/
├── scripts/
│   ├── export_nllb.py
│   ├── download_models.sh
│   └── setup.sh
├── config/
│   ├── default.toml
│   ├── high_quality.toml
│   └── low_latency.toml
├── tests/
│   └── integration_tests.rs
└── benches/
    └── performance.rs
```

**Deliverable**: Empty project with structure in place

---

## PHASE 1: STDIN PARSER (Priority 1)

### Objective
Read Claude's stream-json from stdin, parse it, extract text messages only.

### Task 1.1: Add Dependencies

**Action**: Edit `Cargo.toml`
```toml
[package]
name = "voice-stream-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
regex = "1.10"

[dev-dependencies]
tokio-test = "0.4"
```

**Test**: `cargo build` should succeed

### Task 1.2: Implement JSON Parser

**Action**: Create `src/parser.rs`

**Requirements**:
1. Read lines from stdin asynchronously
2. Parse each line as JSON
3. Extract text from `content` array where `type == "text"`
4. Filter out tool outputs (keep only assistant messages)
5. Pass through tool names (but don't speak them)

**Code skeleton**:
```rust
// src/parser.rs
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct StreamMessage {
    #[serde(rename = "type")]
    pub msg_type: Option<String>,
    pub content: Option<Vec<ContentBlock>>,
    pub role: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub name: Option<String>,  // For tool_use blocks
}

pub async fn parse_stdin_stream(
    tx: mpsc::UnboundedSender<String>
) -> Result<()> {
    // TODO: Implement
    // 1. Create BufReader from stdin
    // 2. Read lines in loop
    // 3. Parse JSON
    // 4. Filter for assistant text messages
    // 5. Send clean text to channel
    Ok(())
}

pub fn clean_text(text: &str) -> String {
    // TODO: Implement
    // 1. Remove markdown: **, *, `
    // 2. Remove code blocks: ```...```
    // 3. Remove URLs: https://...
    // 4. Remove file paths: /path/to/file
    // 5. Trim whitespace
    String::new()
}
```

**Tests**: Create `src/parser.rs` tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_markdown() {
        let input = "**bold** and *italic*";
        assert_eq!(clean_text(input), "bold and italic");
    }

    #[test]
    fn test_remove_code_blocks() {
        let input = "Text\n```rust\ncode\n```\nMore text";
        let clean = clean_text(input);
        assert!(!clean.contains("code"));
    }

    #[test]
    fn test_remove_urls() {
        let input = "Check https://example.com for info";
        let clean = clean_text(input);
        assert!(!clean.contains("https://"));
    }
}
```

**Run tests**: `cargo test parser`

### Task 1.3: Implement Main Loop

**Action**: Create `src/main.rs`
```rust
use tokio::sync::mpsc;
use tracing::{info, error};
use anyhow::Result;

mod parser;
mod config;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("voice_stream_rust=debug")
        .init();

    info!("Starting voice-stream-rust");

    // Create channels
    let (text_tx, mut text_rx) = mpsc::unbounded_channel();

    // Spawn parser
    let parser_handle = tokio::spawn(async move {
        parser::parse_stdin_stream(text_tx).await
    });

    // For now, just print received text
    while let Some(text) = text_rx.recv().await {
        println!("Received: {}", text);
    }

    parser_handle.await??;

    Ok(())
}
```

**Test manually**:
```bash
# Test with sample input
echo '{"type":"message","role":"assistant","content":[{"type":"text","text":"Hello world"}]}' | cargo run

# Should output: "Received: Hello world"
```

**Checkpoint**: Parser working, extracting text from JSON ✅

---

## PHASE 2: TRANSLATION MODULE (Priority 1)

### Objective
Translate English text to Japanese using GPU-accelerated NLLB-200 model.

### Task 2.1: Choose Translation Approach

**Decision Point**: You must choose one of these approaches:

#### Option A: ONNX Runtime (Recommended)
**Pros**: Native Rust, good performance, direct GPU access
**Cons**: Complex model export, requires ONNX expertise

#### Option B: Python subprocess (Faster to implement)
**Pros**: Easy model loading, proven libraries
**Cons**: Higher latency, more memory

#### Option C: Pure Rust with candle (Cutting edge)
**Pros**: Pure Rust, no dependencies
**Cons**: Newer, less stable, limited model support

**Recommendation**: Start with Option B (Python), migrate to Option A later if needed.

### Task 2.2: Implement Python Translation Bridge (Option B)

**Action**: Create `scripts/translation_server.py`
```python
#!/usr/bin/env python3
"""
Translation server using NLLB-200 with GPU acceleration.
Reads sentences from stdin, outputs translations to stdout.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TranslationServer:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Set language codes
        self.src_lang = "eng_Latn"
        self.tgt_lang = "jpn_Jpan"

        print("Model loaded. Ready.", file=sys.stderr)

    def translate(self, text):
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Set target language
        forced_bos_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )

        # Decode
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                translation = self.translate(line)
                print(translation, flush=True)
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                print(line, flush=True)  # Fallback to original

if __name__ == "__main__":
    server = TranslationServer()
    server.run()
```

**Test**:
```bash
chmod +x scripts/translation_server.py
echo "Hello, how are you?" | python3 scripts/translation_server.py
# Should output: こんにちは、元気ですか？
```

### Task 2.3: Integrate Translation in Rust

**Action**: Create `src/translation.rs`
```rust
use tokio::sync::mpsc;
use tokio::process::{Command, ChildStdin, ChildStdout};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use anyhow::Result;

pub struct TranslationEngine {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl TranslationEngine {
    pub async fn new() -> Result<Self> {
        let mut child = Command::new("python3")
            .arg("scripts/translation_server.py")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        // Wait for "Ready" message
        // TODO: Parse stderr to wait for model loading

        Ok(Self { stdin, stdout })
    }

    pub async fn translate(&mut self, text: &str) -> Result<String> {
        // Send text
        self.stdin.write_all(text.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        // Read translation
        let mut line = String::new();
        self.stdout.read_line(&mut line).await?;

        Ok(line.trim().to_string())
    }
}

pub async fn translation_worker(
    mut rx: mpsc::UnboundedReceiver<String>,
    tx: mpsc::UnboundedSender<String>,
) -> Result<()> {
    let mut engine = TranslationEngine::new().await?;

    while let Some(text) = rx.recv().await {
        // Segment into sentences
        let sentences = segment_sentences(&text);

        for sentence in sentences {
            if let Ok(translated) = engine.translate(&sentence).await {
                tx.send(translated)?;
            }
        }
    }

    Ok(())
}

fn segment_sentences(text: &str) -> Vec<String> {
    text.split(&['.', '!', '?'][..])
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}
```

**Update main.rs**:
```rust
mod translation;

// In main():
let (translated_tx, mut translated_rx) = mpsc::unbounded_channel();

let translation_handle = tokio::spawn(async move {
    translation::translation_worker(text_rx, translated_tx).await
});

while let Some(translated) = translated_rx.recv().await {
    println!("Translated: {}", translated);
}
```

**Test**:
```bash
echo '{"content":[{"type":"text","text":"Hello world"}]}' | cargo run
# Should output: "Translated: こんにちは、世界"
```

**Checkpoint**: Translation working with GPU ✅

---

## PHASE 3: TTS MODULE (Priority 1)

### Objective
Convert Japanese text to speech using high-quality TTS model.

### Task 3.1: Choose TTS Engine

**Decision Point**:

#### Option A: Piper (Fast, Good Quality)
- Pre-built ONNX models
- Easy integration
- Speed: 10x real-time

#### Option B: Coqui TTS (Best Quality)
- State-of-the-art voices
- More complex setup
- Speed: 2-3x real-time

#### Option C: Cloud API (Fallback)
- Google Cloud TTS
- Easy integration
- Requires API key and internet

**Recommendation**: Piper for Phase 3, evaluate Coqui in Phase 4.

### Task 3.2: Download Piper Models

**Action**: Create `scripts/download_models.sh`
```bash
#!/bin/bash
# Download Piper TTS models

mkdir -p models/tts/piper
cd models/tts/piper

# Japanese voice (Nanami)
echo "Downloading Japanese voice..."
wget -q --show-progress \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/ja/ja_JP/nanami/medium/ja_JP-nanami-medium.onnx

wget -q --show-progress \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/ja/ja_JP/nanami/medium/ja_JP-nanami-medium.onnx.json

echo "Models downloaded successfully"
```

**Run**: `chmod +x scripts/download_models.sh && ./scripts/download_models.sh`

### Task 3.3: Implement TTS Server

**Action**: Create `scripts/tts_server.py`
```python
#!/usr/bin/env python3
"""
TTS server using Piper.
Reads Japanese text from stdin, outputs WAV audio to stdout.
"""
import sys
import subprocess
import base64

class TTSServer:
    def __init__(self, model_path="models/tts/piper/ja_JP-nanami-medium.onnx"):
        self.model_path = model_path
        print(f"Using model: {model_path}", file=sys.stderr)
        print("TTS server ready", file=sys.stderr)

    def synthesize(self, text):
        # Call piper CLI
        result = subprocess.run(
            ["piper", "--model", self.model_path, "--output-raw"],
            input=text,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Piper failed: {result.stderr}")

        # Return raw PCM data as base64
        audio_data = base64.b64encode(result.stdout.encode()).decode()
        return audio_data

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                audio = self.synthesize(line)
                print(audio, flush=True)
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    server = TTSServer()
    server.run()
```

**Alternative: Use Rust directly with piper_rs crate**:

Check if `piper_rs` or similar crate exists:
```bash
cargo search piper
```

If available, use Rust implementation instead of Python subprocess.

### Task 3.4: Integrate TTS in Rust

**Action**: Create `src/tts.rs`
```rust
use tokio::sync::mpsc;
use tokio::process::{Command, ChildStdin, ChildStdout};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use anyhow::Result;

pub struct TTSEngine {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl TTSEngine {
    pub async fn new() -> Result<Self> {
        let mut child = Command::new("python3")
            .arg("scripts/tts_server.py")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        Ok(Self { stdin, stdout })
    }

    pub async fn synthesize(&mut self, text: &str) -> Result<Vec<u8>> {
        // Send text
        self.stdin.write_all(text.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        // Read audio (base64 encoded)
        let mut line = String::new();
        self.stdout.read_line(&mut line).await?;

        // Decode base64
        let audio = base64::decode(line.trim())?;
        Ok(audio)
    }
}

pub async fn tts_worker(
    mut rx: mpsc::UnboundedReceiver<String>,
    tx: mpsc::UnboundedSender<Vec<u8>>,
) -> Result<()> {
    let mut engine = TTSEngine::new().await?;

    while let Some(text) = rx.recv().await {
        if let Ok(audio) = engine.synthesize(&text).await {
            tx.send(audio)?;
        }
    }

    Ok(())
}
```

**Add to Cargo.toml**:
```toml
base64 = "0.21"
```

**Update main.rs**:
```rust
mod tts;

let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

let tts_handle = tokio::spawn(async move {
    tts::tts_worker(translated_rx, audio_tx).await
});

while let Some(audio) = audio_rx.recv().await {
    println!("Received audio: {} bytes", audio.len());
}
```

**Checkpoint**: TTS generating audio ✅

---

## PHASE 4: AUDIO PLAYBACK (Priority 1)

### Objective
Play audio in real-time without gaps or stuttering.

### Task 4.1: Add Audio Dependencies

**Action**: Update `Cargo.toml`
```toml
rodio = "0.17"
cpal = "0.15"
```

### Task 4.2: Implement Audio Player

**Action**: Create `src/audio.rs`
```rust
use rodio::{OutputStream, Sink, Source};
use std::io::Cursor;
use anyhow::Result;

pub struct AudioPlayer {
    _stream: OutputStream,
    sink: Sink,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let (stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;

        Ok(Self {
            _stream: stream,
            sink,
        })
    }

    pub fn play(&self, audio_data: Vec<u8>) -> Result<()> {
        // Decode audio (assuming WAV format)
        let cursor = Cursor::new(audio_data);
        let source = rodio::Decoder::new(cursor)?;

        self.sink.append(source);

        Ok(())
    }

    pub fn is_playing(&self) -> bool {
        !self.sink.empty()
    }
}
```

**Update main.rs**:
```rust
mod audio;

let player = audio::AudioPlayer::new()?;

while let Some(audio) = audio_rx.recv().await {
    player.play(audio)?;
}
```

**Test**: Should hear audio output

**Checkpoint**: Audio playback working ✅

---

## PHASE 5: INTEGRATION & TESTING (Priority 1)

### Task 5.1: Wire Everything Together

**Action**: Update `src/main.rs` with complete pipeline
```rust
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("voice_stream_rust=info")
        .init();

    info!("Starting voice-stream-rust");

    // Create channels
    let (text_tx, text_rx) = mpsc::unbounded_channel();
    let (translated_tx, translated_rx) = mpsc::unbounded_channel();
    let (audio_tx, audio_rx) = mpsc::unbounded_channel();

    // Spawn all workers
    let parser_handle = tokio::spawn(parser::parse_stdin_stream(text_tx));
    let translation_handle = tokio::spawn(translation::translation_worker(text_rx, translated_tx));
    let tts_handle = tokio::spawn(tts::tts_worker(translated_rx, audio_tx));

    // Audio player (blocking)
    let player = audio::AudioPlayer::new()?;
    let audio_handle = tokio::spawn(async move {
        while let Some(audio) = audio_rx.recv().await {
            if let Err(e) = player.play(audio) {
                error!("Audio playback error: {}", e);
            }
        }
    });

    // Wait for ctrl-c
    tokio::select! {
        _ = parser_handle => info!("Parser finished"),
        _ = translation_handle => info!("Translation finished"),
        _ = tts_handle => info!("TTS finished"),
        _ = audio_handle => info!("Audio finished"),
        _ = tokio::signal::ctrl_c() => info!("Received Ctrl-C"),
    }

    Ok(())
}
```

### Task 5.2: End-to-End Test

**Action**: Test with Claude Code
```bash
# Build release binary
cargo build --release

# Test with sample input
echo '{"content":[{"type":"text","text":"Hello, how are you today?"}]}' | \
  ./target/release/voice-stream-rust

# Should:
# 1. Parse JSON
# 2. Translate to Japanese
# 3. Synthesize speech
# 4. Play audio
```

### Task 5.3: Integration with run_worker.sh

**Action**: Create `run_worker_rust.sh`
```bash
#!/bin/bash
# Worker using Rust TTS system

LOG_DIR="worker_logs"
mkdir -p "$LOG_DIR"

iteration=1
while true; do
    echo "=== Worker Iteration $iteration ==="

    PROMPT="continue"
    LOG_FILE="$LOG_DIR/worker_iter_${iteration}_$(date +%Y%m%d_%H%M%S).jsonl"

    # Use Rust streaming processor
    claude --dangerously-skip-permissions -p "$PROMPT" \
        --permission-mode acceptEdits \
        --output-format stream-json \
        --verbose 2>&1 | tee "$LOG_FILE" | ./voice-stream-rust/target/release/voice-stream-rust

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Worker exited with error code $exit_code"
        break
    fi

    iteration=$((iteration + 1))
    sleep 2
done
```

**Test**: Run worker and listen for speech

**Checkpoint**: Full pipeline working ✅

---

## PHASE 6: OPTIMIZATION (Priority 2)

### Task 6.1: Benchmark Current Performance

**Action**: Create `benches/performance.rs`
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_parse_json(c: &mut Criterion) {
    let json = r#"{"content":[{"type":"text","text":"Hello world"}]}"#;
    c.bench_function("parse_json", |b| {
        b.iter(|| parser::parse_json(black_box(json)))
    });
}

fn bench_translation(c: &mut Criterion) {
    // TODO: Benchmark translation latency
}

fn bench_tts(c: &mut Criterion) {
    // TODO: Benchmark TTS latency
}

criterion_group!(benches, bench_parse_json, bench_translation, bench_tts);
criterion_main!(benches);
```

**Run**: `cargo bench`

**Document results** in `PERFORMANCE.md`

### Task 6.2: Profile with flamegraph

**Action**:
```bash
cargo install flamegraph
sudo cargo flamegraph

# Analyze hotspots
# Optimize bottlenecks
```

### Task 6.3: Optimize Critical Paths

**Targets**:
- Translation latency < 100ms
- TTS latency < 200ms
- Audio buffer < 50ms
- Total latency < 500ms

**Techniques**:
1. Batch translations (if multiple sentences)
2. Pipeline translation + TTS
3. Stream audio in chunks
4. Reuse buffers

---

## PHASE 7: CONFIGURATION & POLISH (Priority 3)

### Task 7.1: Add Configuration File

**Action**: Create `src/config.rs`
```rust
use serde::{Deserialize, Serialize};
use std::path::Path;
use anyhow::Result;

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub translation: TranslationConfig,
    pub tts: TTSConfig,
    pub audio: AudioConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TranslationConfig {
    pub model: String,
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TTSConfig {
    pub engine: String,
    pub voice: String,
    pub speed: f32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_size: usize,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
}
```

**Create** `config/default.toml`:
```toml
[translation]
model = "nllb-200"
source_lang = "en"
target_lang = "ja"

[tts]
engine = "piper"
voice = "nanami"
speed = 1.0

[audio]
sample_rate = 22050
buffer_size = 4096
```

### Task 7.2: Add CLI Arguments

**Action**: Use `clap` for CLI
```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Config file path
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    /// Translation model
    #[arg(long)]
    translation_model: Option<String>,

    /// TTS voice
    #[arg(long)]
    voice: Option<String>,
}
```

### Task 7.3: Add Logging

**Action**: Use `tracing` for structured logging
```rust
use tracing::{info, warn, error, debug};

info!("Starting translation");
debug!("Translated: {} -> {}", input, output);
error!("Translation failed: {}", err);
```

---

## PHASE 8: DOCUMENTATION (Priority 3)

### Task 8.1: Write README

**Action**: Create `voice-stream-rust/README.md`
```markdown
# Voice Stream Rust

GPU-accelerated streaming TTS system for Claude Code.

## Features
- Real-time English → Japanese translation
- High-quality TTS with Piper
- GPU acceleration for translation
- < 500ms latency

## Setup
\`\`\`bash
# Install dependencies
./scripts/download_models.sh

# Build
cargo build --release

# Run
echo "Hello" | ./target/release/voice-stream-rust
\`\`\`

## Performance
- Translation: ~50ms/sentence (GPU)
- TTS: ~100ms/sentence (CPU)
- Total latency: ~200ms
```

### Task 8.2: Document Architecture

**Action**: Create diagrams in `ARCHITECTURE.md`

### Task 8.3: Create Setup Script

**Action**: Create `scripts/setup.sh`
```bash
#!/bin/bash
set -e

echo "Setting up voice-stream-rust..."

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust not installed"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
elif command -v rocm-smi &> /dev/null; then
    echo "✓ AMD GPU detected"
else
    echo "⚠ No GPU detected, will use CPU"
fi

# Download models
echo "Downloading models..."
./scripts/download_models.sh

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch transformers optimum

# Build
echo "Building Rust binary..."
cargo build --release

echo "✓ Setup complete!"
echo "Run: ./target/release/voice-stream-rust"
```

---

## SUCCESS CRITERIA

### Phase 1-5 (Must Complete)
- ✅ Parse JSON from stdin
- ✅ Translate English → Japanese
- ✅ Synthesize speech
- ✅ Play audio without gaps
- ✅ < 1 second latency

### Phase 6-7 (Should Complete)
- ✅ < 500ms latency
- ✅ GPU acceleration working
- ✅ Configuration file support
- ✅ Error handling

### Phase 8 (Nice to Have)
- ✅ Documentation complete
- ✅ Automated setup
- ✅ Performance benchmarks

---

## TROUBLESHOOTING

### Issue: GPU not detected
**Solution**: Install CUDA toolkit or ROCm
```bash
# NVIDIA
sudo apt install nvidia-cuda-toolkit

# AMD
sudo apt install rocm-dev
```

### Issue: Translation slow
**Solution**: Verify GPU is being used
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue: Audio gaps
**Solution**: Increase buffer size in config
```toml
[audio]
buffer_size = 8192  # Larger buffer
```

### Issue: High latency
**Solution**: Profile and optimize
```bash
cargo flamegraph
# Look for bottlenecks
```

---

## REPORTING

### Daily Progress Report Format

```markdown
## Day N Progress Report

### Completed
- Task X.Y: Description
- Task X.Z: Description

### Blocked
- Issue with [component]: Description

### Next Steps
- Task X: Plan

### Measurements
- Translation latency: XXms
- TTS latency: XXms
- Total latency: XXms
- GPU utilization: XX%
```

---

## END OF WORKPLAN

Follow this plan sequentially. Report progress daily. Ask questions if blocked. Test continuously.

**Copyright 2025 Andrew Yates. All rights reserved.**
