//! Example: Meeting summarizer with dashflow-voice
//!
//! This example demonstrates live meeting summarization:
//! 1. Wake word activation ("Hey Voice")
//! 2. Listen and transcribe speech
//! 3. Generate summaries periodically or on-demand
//! 4. Speak summaries aloud
//!
//! # Running
//!
//! First, start the voice daemon with summarization enabled:
//! ```bash
//! cd stream-tts-cpp/build && ./stream-tts-cpp --grpc --auto-summarize
//! ```
//!
//! Then run this example:
//! ```bash
//! cargo run --example meeting_summarizer
//! ```
//!
//! # Commands
//!
//! Say "Hey Voice" to activate, then:
//! - "Summarize" - Generate a brief summary of the meeting so far
//! - "Action items" - Extract action items from the discussion
//! - "Detailed summary" - Generate a detailed summary with bullet points
//! - "Stop" or "Goodbye" - End the meeting and exit

use dashflow_voice::{
    VoiceClient, VoiceConfig, FilterConfig,
    SummarizationClient, SummarizationConfig,
    WakeWordClient, WakeWordConfig,
    Result,
};
use std::time::{Duration, Instant};

const VOICE_ENDPOINT: &str = "http://localhost:50051";

/// Accumulated meeting transcript
struct MeetingTranscript {
    segments: Vec<TranscriptSegment>,
    start_time: Instant,
}

struct TranscriptSegment {
    speaker: String,
    text: String,
    timestamp_secs: u64,
}

impl MeetingTranscript {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            start_time: Instant::now(),
        }
    }

    fn add(&mut self, speaker: &str, text: &str) {
        let timestamp_secs = self.start_time.elapsed().as_secs();
        self.segments.push(TranscriptSegment {
            speaker: speaker.to_string(),
            text: text.to_string(),
            timestamp_secs,
        });
    }

    fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| format!("[{}] {}: {}", format_time(s.timestamp_secs), s.speaker, s.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn word_count(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.text.split_whitespace().count())
            .sum()
    }

    fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

fn format_time(secs: u64) -> String {
    let mins = secs / 60;
    let secs = secs % 60;
    format!("{:02}:{:02}", mins, secs)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Meeting Summarizer Example");
    println!("==========================");
    println!("Connecting to voice service at {}...", VOICE_ENDPOINT);

    // Create clients
    let voice_config = VoiceConfig::new(VOICE_ENDPOINT)
        .with_language("en")
        .with_filtering(true)
        .with_connect_timeout_ms(10000);

    let voice_client = match VoiceClient::connect_with_config(voice_config).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!("\nMake sure the voice daemon is running:");
            eprintln!("  cd stream-tts-cpp/build && ./stream-tts-cpp --grpc --auto-summarize");
            return Err(e);
        }
    };

    let summarization_client = match SummarizationClient::connect(VOICE_ENDPOINT).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect to summarization service: {}", e);
            return Err(e);
        }
    };

    let wake_word_client = match WakeWordClient::connect(VOICE_ENDPOINT).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect to wake word service: {}", e);
            return Err(e);
        }
    };

    // Check if services are ready
    match voice_client.is_ready().await {
        Ok(true) => println!("Voice service ready!"),
        Ok(false) => {
            eprintln!("Voice service not ready (models not loaded)");
            return Ok(());
        }
        Err(e) => eprintln!("Could not check voice status: {}", e),
    }

    match summarization_client.is_ready().await {
        Ok(true) => println!("Summarization service ready!"),
        Ok(false) => {
            eprintln!("Summarization model not loaded (Qwen3-8B)");
            eprintln!("Start daemon with: --auto-summarize to enable");
        }
        Err(e) => eprintln!("Could not check summarization status: {}", e),
    }

    // Greeting
    println!("\nMeeting summarizer active!");
    println!("Say 'Hey Voice' to activate, then speak.");
    println!("Commands: 'summarize', 'action items', 'detailed summary', 'stop'\n");

    voice_client.speak("Meeting summarizer ready. Say Hey Voice to activate.").await?;

    // Meeting state
    let mut transcript = MeetingTranscript::new();
    let filter_config = FilterConfig::all_enabled();
    let wake_config = WakeWordConfig::for_wake_words(vec!["hey_voice".to_string()])
        .with_threshold(0.5);

    // Auto-summarize every N words
    const AUTO_SUMMARIZE_THRESHOLD: usize = 200;
    let mut last_summary_word_count: usize = 0;

    loop {
        // Wait for wake word
        println!("[Waiting for 'Hey Voice'...]");

        let mut wake_stream = match wake_word_client.start_listening(wake_config.clone()).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Wake word error: {}", e);
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        // Wait for detection
        match wake_stream.next().await {
            Some(Ok(detection)) => {
                println!("\n[Wake word '{}' detected (confidence: {:.2})]",
                         detection.wake_word, detection.confidence);
            }
            Some(Err(e)) => {
                eprintln!("Wake word detection error: {}", e);
                continue;
            }
            None => continue,
        }

        // Stop wake word listening
        let _ = wake_word_client.stop_listening().await;

        // Acknowledgment
        voice_client.speak("I'm listening").await?;

        // Listen for speech
        println!("[Listening...]");

        let result = match voice_client.listen_filtered_with_config(filter_config.clone()).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Listen error: {}", e);
                continue;
            }
        };

        if result.confidence < 0.3 {
            println!("[Low confidence ({:.2}), ignoring]", result.confidence);
            continue;
        }

        let text_lower = result.text.to_lowercase();
        println!("{}: {}", result.speaker_id, result.text);

        // Check for commands
        if text_lower.contains("stop") || text_lower.contains("goodbye") || text_lower.contains("end meeting") {
            // Final summary
            if !transcript.is_empty() {
                println!("\n[Generating final summary...]");
                voice_client.speak("Generating final meeting summary.").await?;

                match summarization_client.summarize(&transcript.full_text(),
                    SummarizationConfig::detailed()).await {
                    Ok(summary) => {
                        println!("\nFinal Summary:\n{}\n", summary.text);
                        voice_client.speak(&format!("Final summary: {}", summary.text)).await?;
                    }
                    Err(e) => eprintln!("Summarization error: {}", e),
                }
            }

            voice_client.speak("Meeting ended. Goodbye!").await?;
            println!("\n[Meeting ended]");
            break;
        } else if text_lower.contains("summarize") && !text_lower.contains("action") && !text_lower.contains("detailed") {
            // Brief summary
            if transcript.is_empty() {
                voice_client.speak("No meeting content to summarize yet.").await?;
            } else {
                println!("[Generating brief summary...]");
                match summarization_client.summarize_brief(&transcript.full_text()).await {
                    Ok(summary) => {
                        println!("Summary: {}", summary.text);
                        voice_client.speak(&summary.text).await?;
                    }
                    Err(e) => eprintln!("Summarization error: {}", e),
                }
            }
        } else if text_lower.contains("action items") || text_lower.contains("action item") {
            // Action items
            if transcript.is_empty() {
                voice_client.speak("No meeting content to extract action items from.").await?;
            } else {
                println!("[Extracting action items...]");
                match summarization_client.summarize_action_items(&transcript.full_text()).await {
                    Ok(summary) => {
                        println!("Action Items: {}", summary.text);
                        voice_client.speak(&format!("Action items: {}", summary.text)).await?;
                    }
                    Err(e) => eprintln!("Summarization error: {}", e),
                }
            }
        } else if text_lower.contains("detailed summary") || text_lower.contains("detailed") {
            // Detailed summary
            if transcript.is_empty() {
                voice_client.speak("No meeting content to summarize yet.").await?;
            } else {
                println!("[Generating detailed summary...]");
                match summarization_client.summarize(&transcript.full_text(),
                    SummarizationConfig::detailed()).await {
                    Ok(summary) => {
                        println!("Detailed Summary:\n{}", summary.text);
                        voice_client.speak(&summary.text).await?;
                    }
                    Err(e) => eprintln!("Summarization error: {}", e),
                }
            }
        } else {
            // Regular speech - add to transcript
            transcript.add(&result.speaker_id, &result.text);
            println!("[Added to transcript ({} words total)]", transcript.word_count());

            // Auto-summarize check
            let word_count = transcript.word_count();
            if word_count - last_summary_word_count >= AUTO_SUMMARIZE_THRESHOLD {
                println!("\n[Auto-generating summary ({}+ new words)...]", AUTO_SUMMARIZE_THRESHOLD);
                voice_client.speak("Let me summarize the discussion so far.").await?;

                match summarization_client.summarize_brief(&transcript.full_text()).await {
                    Ok(summary) => {
                        println!("Auto-Summary: {}", summary.text);
                        voice_client.speak(&summary.text).await?;
                        last_summary_word_count = word_count;
                    }
                    Err(e) => eprintln!("Auto-summarization error: {}", e),
                }
            }
        }

        // Small delay before next wake word detection
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    Ok(())
}
