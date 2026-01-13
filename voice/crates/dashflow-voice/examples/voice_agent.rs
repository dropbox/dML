//! Example: Voice-enabled agent with dashflow-voice
//!
//! This example shows a simple voice conversation loop:
//! 1. Listen for user speech (with self-speech filtering)
//! 2. Process the text (echo in this case)
//! 3. Speak the response (non-blocking)
//!
//! # Running
//!
//! First, start the voice daemon:
//! ```bash
//! cd stream-tts-cpp/build && ./stream-tts-cpp --grpc
//! ```
//!
//! Then run this example:
//! ```bash
//! cargo run --example voice_agent
//! ```

use dashflow_voice::{VoiceClient, VoiceConfig, FilterConfig, Result};

const VOICE_ENDPOINT: &str = "http://localhost:50051";

#[tokio::main]
async fn main() -> Result<()> {
    println!("Voice Agent Example");
    println!("==================");
    println!("Connecting to voice service at {}...", VOICE_ENDPOINT);

    // Create voice client with custom config
    let config = VoiceConfig::new(VOICE_ENDPOINT)
        .with_language("en")
        .with_filtering(true)
        .with_connect_timeout_ms(10000);

    let client = match VoiceClient::connect_with_config(config).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!("\nMake sure the voice daemon is running:");
            eprintln!("  cd stream-tts-cpp/build && ./stream-tts-cpp --grpc");
            return Err(e);
        }
    };

    // Check if service is ready
    match client.is_ready().await {
        Ok(true) => println!("Voice service ready!"),
        Ok(false) => {
            eprintln!("Voice service not ready (models not loaded)");
            return Ok(());
        }
        Err(e) => {
            eprintln!("Could not check status: {}", e);
        }
    }

    // Greeting
    println!("\nStarting voice conversation...");
    client.speak("Hello! I am your voice assistant. How can I help you?").await?;

    // Conversation loop
    let filter_config = FilterConfig::all_enabled();

    println!("\nListening for user speech (Ctrl+C to exit)...\n");

    loop {
        // Listen for user speech (blocks until speech ends)
        println!("[Listening...]");

        let result = match client.listen_filtered_with_config(filter_config.clone()).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Listen error: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        // Check confidence
        if result.confidence < 0.5 {
            println!("[Low confidence ({:.2}), ignoring]", result.confidence);
            continue;
        }

        println!("User ({}): {}", result.speaker_id, result.text);

        // Check for exit commands
        let text_lower = result.text.to_lowercase();
        if text_lower.contains("goodbye") || text_lower.contains("exit") || text_lower.contains("quit") {
            client.speak("Goodbye! Have a great day.").await?;
            println!("\n[Exiting]");
            break;
        }

        // Echo response (in a real agent, this would call an LLM)
        let response = format!("You said: {}", result.text);
        println!("Agent: {}", response);

        // Speak response (non-blocking)
        client.speak(&response).await?;

        // Small delay before next listen
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}
