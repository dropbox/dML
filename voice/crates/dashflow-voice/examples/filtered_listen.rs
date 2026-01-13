//! Example: Filtered listening with StartFilteredListen gRPC
//!
//! This example demonstrates the dashflow-optimized StartFilteredListen RPC:
//! - Bidirectional streaming
//! - Session-based filtering
//! - Real-time transcription events
//!
//! # Architecture
//!
//! ```text
//! Client                    Voice Service
//!   |                            |
//!   |-- FilteredListenRequest -->|  (config)
//!   |-- FilteredListenRequest -->|  (audio + TTS metadata)
//!   |                            |
//!   |<-- TranscriptionEvent -----|  (user-only text)
//!   |<-- TranscriptionEvent -----|  (speaker_id, confidence)
//!   |                            |
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example filtered_listen
//! ```

use dashflow_voice::{VoiceClient, FilterConfig, Result};
use tokio_stream::StreamExt;

const VOICE_ENDPOINT: &str = "http://localhost:50051";

#[tokio::main]
async fn main() -> Result<()> {
    println!("Filtered Listen Example (StartFilteredListen gRPC)");
    println!("===================================================\n");

    // Connect to voice service
    let client = match VoiceClient::connect(VOICE_ENDPOINT).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Connection failed: {}", e);
            eprintln!("\nStart the voice daemon first:");
            eprintln!("  cd stream-tts-cpp/build && ./stream-tts-cpp --grpc");
            return Err(e);
        }
    };

    println!("Connected to voice service");
    println!("Session ID: {}\n", client.session_id().await);

    // Configure filter
    let config = FilterConfig {
        enable_text_matching: true,
        enable_aec: true,
        enable_speaker_diarization: true,
        agent_confidence_threshold: 0.7,
        debug_logging: true,
        ..Default::default()
    };

    println!("Filter configuration:");
    println!("  Text matching: {}", config.enable_text_matching);
    println!("  AEC: {}", config.enable_aec);
    println!("  Speaker diarization: {}", config.enable_speaker_diarization);
    println!("  Agent threshold: {}", config.agent_confidence_threshold);
    println!();

    // Get filtered STT client
    let filtered_stt = client.filtered_stt();

    // Example 1: Single listen (blocking until speech ends)
    println!("=== Example 1: Single filtered listen ===");
    println!("Speak something...\n");

    let result = filtered_stt
        .listen_with_config(&client.session_id().await, config.clone())
        .await?;

    println!("Result:");
    println!("  Text: \"{}\"", result.text);
    println!("  Speaker: {}", result.speaker_id);
    println!("  Confidence: {:.2}", result.confidence);
    println!("  Speaker confidence: {:.2}", result.speaker_confidence);
    if let Some(details) = &result.filtering_details {
        println!("  Filtering details:");
        println!("    Filtered agent speech: \"{}\"", details.filtered_agent_speech);
        println!("    Text match active: {}", details.text_match_active);
        println!("    AEC active: {}", details.aec_active);
        println!("    Speaker ID active: {}", details.speaker_id_active);
    }
    println!();

    // Example 2: Streaming listen (real-time events)
    println!("=== Example 2: Streaming filtered listen ===");
    println!("Speak continuously (5 second timeout)...\n");

    let mut stream = filtered_stt
        .listen_stream(&client.session_id().await, config)
        .await?;

    let timeout = tokio::time::sleep(std::time::Duration::from_secs(5));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            event = stream.next() => {
                match event {
                    Some(Ok(transcription)) => {
                        let marker = if transcription.is_final { "[FINAL]" } else { "[partial]" };
                        println!("{} {} (speaker: {}, conf: {:.2})",
                            marker,
                            transcription.text,
                            transcription.speaker_id,
                            transcription.confidence
                        );
                        if transcription.is_final {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        eprintln!("Stream error: {}", e);
                        break;
                    }
                    None => {
                        println!("Stream ended");
                        break;
                    }
                }
            }
            _ = &mut timeout => {
                println!("\nTimeout reached");
                break;
            }
        }
    }

    println!("\nDone!");
    Ok(())
}
