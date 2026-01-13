//! Main VoiceClient implementation
//!
//! Provides unified access to TTS, STT, and FilteredSTT services via gRPC.

use crate::config::VoiceConfig;
use crate::error::{Result, VoiceError};
use crate::tts::{VoiceTTS, SpeakHandle};
use crate::stt::{VoiceSTT, Transcription};
use crate::filtered::{FilteredSTT, FilteredTranscription, FilterConfig};

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};

/// Unified voice client for TTS, STT, and filtered listening
///
/// This client manages connections to the voice gRPC service and provides
/// convenient methods for speech synthesis and recognition.
///
/// # Example
///
/// ```rust,no_run
/// use dashflow_voice::VoiceClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = VoiceClient::connect("http://localhost:50051").await?;
///
///     // Speak (non-blocking)
///     client.speak("Hello!").await?;
///
///     // Listen with self-speech filtering
///     let text = client.listen_filtered().await?;
///     println!("User: {}", text);
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct VoiceClient {
    config: Arc<VoiceConfig>,
    channel: Channel,
    session_id: Arc<RwLock<String>>,
}

impl VoiceClient {
    /// Connect to voice service with default config
    pub async fn connect(endpoint: &str) -> Result<Self> {
        let config = VoiceConfig::new(endpoint);
        Self::connect_with_config(config).await
    }

    /// Connect to voice service with custom config
    pub async fn connect_with_config(config: VoiceConfig) -> Result<Self> {
        let endpoint = Endpoint::from_shared(config.endpoint.clone())
            .map_err(|e| VoiceError::InvalidConfig(e.to_string()))?
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            .timeout(Duration::from_millis(config.request_timeout_ms));

        let channel = endpoint.connect().await?;

        let session_id = format!("dashflow-{}", uuid_v4());

        Ok(Self {
            config: Arc::new(config),
            channel,
            session_id: Arc::new(RwLock::new(session_id)),
        })
    }

    /// Get a TTS client for speech synthesis
    pub fn tts(&self) -> VoiceTTS {
        VoiceTTS::new(
            self.channel.clone(),
            self.config.default_language.clone(),
            self.config.default_priority,
        )
    }

    /// Get an STT client for speech recognition
    pub fn stt(&self) -> VoiceSTT {
        VoiceSTT::new(
            self.channel.clone(),
            self.config.default_language.clone(),
        )
    }

    /// Get a filtered STT client with self-speech filtering
    pub fn filtered_stt(&self) -> FilteredSTT {
        FilteredSTT::new(
            self.channel.clone(),
            self.config.default_language.clone(),
        )
    }

    // Convenience methods

    /// Speak text (non-blocking, returns immediately)
    ///
    /// The audio is queued to the voice daemon and plays asynchronously.
    /// Agent execution is NOT blocked.
    pub async fn speak(&self, text: &str) -> Result<SpeakHandle> {
        self.tts().speak(text).await
    }

    /// Speak text in a specific language
    pub async fn speak_lang(&self, text: &str, language: &str) -> Result<SpeakHandle> {
        self.tts().speak_lang(text, language).await
    }

    /// High-priority interrupt (stops current speech)
    pub async fn interrupt(&self, text: &str) -> Result<SpeakHandle> {
        self.tts().interrupt(text).await
    }

    /// Listen for user speech (basic STT, no filtering)
    ///
    /// This blocks until the user finishes speaking.
    pub async fn listen(&self) -> Result<Transcription> {
        self.stt().listen().await
    }

    /// Listen with self-speech filtering
    ///
    /// Filters out agent voice and returns only user speech.
    /// This is the preferred method for full-duplex conversation.
    pub async fn listen_filtered(&self) -> Result<FilteredTranscription> {
        let session_id = self.session_id.read().await.clone();
        self.filtered_stt()
            .listen_with_session(&session_id)
            .await
    }

    /// Listen with self-speech filtering and custom config
    pub async fn listen_filtered_with_config(
        &self,
        filter_config: FilterConfig,
    ) -> Result<FilteredTranscription> {
        let session_id = self.session_id.read().await.clone();
        self.filtered_stt()
            .listen_with_config(&session_id, filter_config)
            .await
    }

    /// Get current TTS state (speaking/silent)
    pub async fn get_tts_state(&self) -> Result<TtsState> {
        self.filtered_stt().get_tts_state().await
    }

    /// Stop all current speech
    pub async fn stop_speech(&self) -> Result<()> {
        // Send interrupt command (empty text)
        self.tts().stop().await
    }

    /// Check if voice service is ready
    pub async fn is_ready(&self) -> Result<bool> {
        self.tts().check_status().await
    }

    /// Register a speaker for diarization
    pub async fn register_speaker(
        &self,
        speaker_id: &str,
        audio_samples: Vec<u8>,
        sample_rate: i32,
    ) -> Result<String> {
        self.filtered_stt()
            .register_speaker(speaker_id, audio_samples, sample_rate)
            .await
    }

    /// Get the current session ID
    pub async fn session_id(&self) -> String {
        self.session_id.read().await.clone()
    }

    /// Set a new session ID
    pub async fn set_session_id(&self, session_id: &str) {
        let mut id = self.session_id.write().await;
        *id = session_id.to_string();
    }
}

/// TTS playback state
#[derive(Debug, Clone)]
pub struct TtsState {
    /// Whether TTS is currently speaking
    pub is_speaking: bool,
    /// Text being spoken (if any)
    pub current_text: String,
    /// Elapsed time since speech started (ms)
    pub elapsed_ms: f32,
    /// Estimated total duration (ms)
    pub estimated_duration_ms: f32,
}

impl Default for TtsState {
    fn default() -> Self {
        Self {
            is_speaking: false,
            current_text: String::new(),
            elapsed_ms: 0.0,
            estimated_duration_ms: 0.0,
        }
    }
}

/// Generate a simple UUID v4 (no external dependency)
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let random_part = (nanos % 0xFFFFFFFFFFFF) as u64;
    format!("{:016x}", random_part)
}
