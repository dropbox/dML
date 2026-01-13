//! Example: Wake Word Activated Voice Agent
//!
//! This example demonstrates hands-free voice activation using wake word detection:
//! 1. Wait for wake word activation ("hey_voice")
//! 2. Play acknowledgment chime
//! 3. Listen for user command
//! 4. Process and respond
//! 5. Return to idle (waiting for wake word)
//!
//! # Running
//!
//! First, start the voice daemon with wake word support:
//! ```bash
//! cd stream-tts-cpp/build && ./stream-tts-cpp --daemon --wake-word hey_voice
//! ```
//!
//! Then run this example:
//! ```bash
//! cargo run --example wake_word_agent
//! ```

use dashflow_voice::{WakeWordAgent, WakeWordConfig, FilterConfig, Result};

const VOICE_ENDPOINT: &str = "http://localhost:50051";

#[tokio::main]
async fn main() -> Result<()> {
    println!("Wake Word Agent Example");
    println!("=======================");
    println!("Connecting to voice service at {}...", VOICE_ENDPOINT);

    // Configure wake word detection
    let wake_word_config = WakeWordConfig::for_wake_words(vec!["hey_voice".to_string()])
        .with_threshold(0.5);

    // Configure self-speech filtering
    let filter_config = FilterConfig::all_enabled();

    // Create wake word agent
    let agent = match WakeWordAgent::with_config(
        VOICE_ENDPOINT,
        wake_word_config,
        filter_config,
    ).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!("\nMake sure the voice daemon is running with wake word support:");
            eprintln!("  cd stream-tts-cpp/build && ./stream-tts-cpp --daemon --wake-word hey_voice");
            return Err(e);
        }
    };

    // Check if service is ready
    match agent.is_ready().await {
        Ok(true) => println!("Voice service ready!"),
        Ok(false) => {
            eprintln!("Voice service not ready (models not loaded)");
            return Ok(());
        }
        Err(e) => {
            eprintln!("Could not check status: {}", e);
        }
    }

    // Show available wake word models
    match agent.list_wake_word_models().await {
        Ok(models) => {
            println!("\nAvailable wake word models:");
            for model in models {
                let status = if model.is_loaded { "loaded" } else { "available" };
                let custom = if model.is_custom { " (custom)" } else { "" };
                println!("  - {}{}: {}", model.name, custom, status);
            }
        }
        Err(e) => {
            eprintln!("Could not list models: {}", e);
        }
    }

    // Instructions
    println!("\nWake word agent ready!");
    println!("Say \"Hey Voice\" to activate, then speak your command.");
    println!("Say \"goodbye\" or \"exit\" to quit.\n");

    // Main loop
    loop {
        // Wait for wake word (blocks)
        println!("[Waiting for wake word...]");

        let event = match agent.wait_for_activation().await {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Wake word error: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        println!("Wake word '{}' detected! (confidence: {:.2}, latency: {}ms)",
                 event.wake_word, event.confidence, event.latency_ms);

        // Acknowledge activation
        agent.speak("I'm listening").await?;

        // Small delay for acknowledgment to start
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Listen for command (blocks until speech ends)
        println!("[Listening for command...]");

        let command = match agent.listen().await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Listen error: {}", e);
                agent.speak("Sorry, I didn't catch that").await?;
                continue;
            }
        };

        println!("User: {}", command);

        // Check for empty command
        if command.trim().is_empty() {
            agent.speak("I didn't hear anything. Please try again.").await?;
            continue;
        }

        // Check for exit commands
        let text_lower = command.to_lowercase();
        if text_lower.contains("goodbye") || text_lower.contains("exit") || text_lower.contains("quit") {
            agent.speak("Goodbye! Have a great day.").await?;
            println!("\n[Exiting]");
            break;
        }

        // Process command (in a real agent, this would call an LLM)
        let response = process_command(&command);
        println!("Agent: {}", response);

        // Speak response (non-blocking)
        agent.speak(&response).await?;

        // Wait a moment before returning to idle
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}

/// Simple command processor (replace with LLM in real agent)
fn process_command(command: &str) -> String {
    let lower = command.to_lowercase();

    if lower.contains("time") {
        let now = chrono::Local::now();
        format!("The current time is {}", now.format("%H:%M"))
    } else if lower.contains("hello") || lower.contains("hi") {
        "Hello! How can I help you?".to_string()
    } else if lower.contains("weather") {
        "I don't have access to weather data, but it looks like a nice day!".to_string()
    } else if lower.contains("help") {
        "I can respond to your voice commands. Try asking about the time, saying hello, or just speak naturally.".to_string()
    } else {
        format!("You said: {}", command)
    }
}
