# dashflow-voice

Voice TTS/STT integration for DashFlow agent orchestration.

## Features

- **Self-Speech Filtering**: Filters agent voice from microphone input for full-duplex conversation
- **Speaker Diarization**: Identifies multiple speakers (agent, user, others)
- **Non-blocking TTS**: `speak()` returns immediately (audio queued asynchronously)
- **Streaming STT**: Real-time transcription with partial results
- **gRPC Client**: Connects to the voice C++ daemon

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
dashflow-voice = { path = "../crates/dashflow-voice" }
```

## Quick Start

```rust
use dashflow_voice::{VoiceClient, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to voice service
    let client = VoiceClient::connect("http://localhost:50051").await?;

    // Speak (non-blocking - returns immediately)
    client.speak("Hello world").await?;

    // Listen with self-speech filtering (blocks until speech ends)
    let result = client.listen_filtered().await?;
    println!("User said: {}", result.text);

    Ok(())
}
```

## Voice Service Setup

First, build and start the voice C++ daemon:

```bash
cd stream-tts-cpp/build
cmake .. && make -j
./stream-tts-cpp --grpc
```

The daemon runs on `localhost:50051` by default.

## API Overview

### VoiceClient

Main entry point for voice operations:

```rust
let client = VoiceClient::connect("http://localhost:50051").await?;

// TTS (non-blocking)
client.speak("Hello").await?;
client.speak_lang("Bonjour", "fr").await?;
client.interrupt("Urgent!").await?;  // High priority

// STT (blocking)
let text = client.listen().await?;  // Basic STT
let filtered = client.listen_filtered().await?;  // With self-speech filtering
```

### FilteredSTT (Self-Speech Filtering)

For full-duplex conversation where agent and user can speak simultaneously:

```rust
use dashflow_voice::{FilteredSTT, FilterConfig};

let filtered = client.filtered_stt();

// Configure filtering
let config = FilterConfig {
    enable_text_matching: true,
    enable_aec: true,
    enable_speaker_diarization: true,
    agent_confidence_threshold: 0.7,
    ..Default::default()
};

// Listen with filtering
let result = filtered.listen_with_config("session-1", config).await?;

if result.confidence > 0.7 {
    println!("User: {}", result.text);
    println!("Speaker: {}", result.speaker_id);
}
```

### Graph Node Helpers

For dashflow graph integration:

```rust
use dashflow_voice::{voice_input_node, voice_output_node, filtered_input_node};

// Create node functions
let listen = filtered_input_node("http://localhost:50051");
let speak = voice_output_node("http://localhost:50051");

// Use in graph
let input = listen().await?;
speak(&response).await?;
```

### VoiceConversation

Convenience wrapper for conversation loops:

```rust
use dashflow_voice::VoiceConversation;

let conv = VoiceConversation::new("http://localhost:50051").await?;

loop {
    // Listen (blocks until speech ends)
    let user_text = conv.listen().await?;

    // Process with LLM
    let response = process_with_llm(&user_text).await?;

    // Speak (non-blocking)
    conv.speak(&response).await?;
}
```

## Self-Speech Filter Architecture

The filtering system uses four layers to separate agent voice from user voice:

```
Layer 1: Text Matching
  - Compares STT output with known TTS text queue
  - Filters exact/fuzzy matches

Layer 2: Acoustic Echo Cancellation (AEC)
  - Uses TTS audio as reference signal
  - Subtracts echo from microphone input

Layer 3: Speaker Diarization
  - Extracts voice embeddings (ECAPA-TDNN)
  - Identifies speakers by voice identity

Layer 4: Temporal Gating
  - Tracks TTS playback state
  - Adjusts confidence based on timing
```

## Examples

Run the examples:

```bash
# Basic voice agent
cargo run --example voice_agent

# Filtered listen demo
cargo run --example filtered_listen
```

## Configuration

```rust
use dashflow_voice::VoiceConfig;

let config = VoiceConfig::new("http://localhost:50051")
    .with_language("en")
    .with_filtering(true)
    .with_connect_timeout_ms(5000)
    .with_request_timeout_ms(30000);

let client = VoiceClient::connect_with_config(config).await?;
```

## License

MIT License - Copyright 2025 Andrew Yates
