//! `dashflow-voice` - Voice TTS/STT integration for DashFlow agent orchestration
//!
//! This crate provides non-blocking voice I/O for agent workflows via gRPC.
//! TTS operations are async and do NOT block agent execution.
//!
//! # Key Features
//!
//! - **Self-Speech Filtering**: Filters agent voice from microphone input
//! - **Speaker Diarization**: Identifies multiple speakers
//! - **Non-blocking TTS**: speak() returns immediately (audio queued)
//! - **Streaming STT**: Real-time transcription with partial results
//! - **Wake Word Detection**: Hands-free voice activation with custom wake words
//! - **Live Summarization**: Real-time transcript summarization with Qwen3-8B
//!
//! # Basic Example
//!
//! ```rust,no_run
//! use dashflow_voice::{VoiceClient, VoiceConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = VoiceClient::connect("http://localhost:50051").await?;
//!
//!     // Non-blocking TTS
//!     client.speak("Hello world").await?;
//!
//!     // Filtered STT (blocks until speech ends)
//!     let text = client.listen_filtered().await?;
//!     println!("User said: {}", text);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Wake Word Agent Example
//!
//! ```rust,no_run
//! use dashflow_voice::WakeWordAgent;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let agent = WakeWordAgent::new("http://localhost:50051").await?;
//!
//!     loop {
//!         // Wait for "Hey Voice" activation
//!         let event = agent.wait_for_activation().await?;
//!         println!("Activated by: {}", event.wake_word);
//!
//!         // Acknowledge
//!         agent.speak("I'm listening").await?;
//!
//!         // Listen for command
//!         let command = agent.listen().await?;
//!         println!("User said: {}", command);
//!
//!         // Respond
//!         agent.speak(&format!("You said: {}", command)).await?;
//!     }
//! }
//! ```

pub mod client;
pub mod config;
pub mod error;
pub mod tts;
pub mod stt;
pub mod filtered;
pub mod nodes;
pub mod wake_word;
pub mod summarization;

// Re-exports
pub use client::VoiceClient;
pub use config::VoiceConfig;
pub use error::{VoiceError, Result};
pub use tts::{VoiceTTS, SpeakHandle};
pub use stt::{VoiceSTT, Transcription, TranscriptionStream};
pub use filtered::{FilteredSTT, FilteredTranscription, FilterConfig};
pub use nodes::{
    voice_input_node, voice_output_node, filtered_input_node,
    wake_word_node, wake_word_node_for, WakeWordEvent, WakeWordAgent, VoiceConversation,
    summarize_text_node, live_summarize_node, summarize_brief_node, summarize_detailed_node,
    summarize_action_items_node, SummaryOutput, SummarizationAgent,
};
pub use wake_word::{WakeWordClient, WakeWordConfig, WakeWordDetection, WakeWordStatus, WakeWordModel};
pub use summarization::{
    SummarizationClient, SummarizationConfig, SummarizationMode,
    Summary, LiveSummary, LiveSummaryStream, SummarizationStatus, SummarizationMetrics,
    AudioChunk,
};

// Generated proto types
#[allow(clippy::all)]
pub mod proto {
    include!("generated/voice.rs");
}
