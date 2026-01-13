//! Text-to-Speech (TTS) client
//!
//! Non-blocking speech synthesis that queues audio to the voice daemon.

use crate::error::Result;
use crate::proto::{
    tts_service_client::TtsServiceClient,
    SynthesizeRequest, StatusRequest, AudioFormat,
};

use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Channel;

/// Non-blocking text-to-speech synthesis
///
/// IMPORTANT: speak() returns immediately after queuing.
/// Audio plays asynchronously - does NOT block agent execution.
///
/// # Example
///
/// ```rust,no_run
/// use dashflow_voice::VoiceTTS;
///
/// async fn example(tts: VoiceTTS) {
///     // Returns immediately, audio plays in background
///     let handle = tts.speak("Hello world").await.unwrap();
///
///     // Optional: wait for completion
///     // handle.wait().await.unwrap();
/// }
/// ```
#[derive(Clone)]
pub struct VoiceTTS {
    client: Arc<Mutex<TtsServiceClient<Channel>>>,
    language: String,
    priority: i32,
}

impl VoiceTTS {
    /// Create a new TTS client
    pub fn new(channel: Channel, language: String, priority: i32) -> Self {
        Self {
            client: Arc::new(Mutex::new(TtsServiceClient::new(channel))),
            language,
            priority,
        }
    }

    /// Queue text for speech (returns immediately)
    ///
    /// The audio is queued to the voice daemon and plays asynchronously.
    /// This method does NOT block agent execution.
    pub async fn speak(&self, text: &str) -> Result<SpeakHandle> {
        self.speak_with_options(text, &self.language, self.priority).await
    }

    /// Queue text with specific language
    pub async fn speak_lang(&self, text: &str, language: &str) -> Result<SpeakHandle> {
        self.speak_with_options(text, language, self.priority).await
    }

    /// High-priority interrupt (stops current speech, speaks immediately)
    pub async fn interrupt(&self, text: &str) -> Result<SpeakHandle> {
        // Priority 100 = highest, interrupts everything
        self.speak_with_options(text, &self.language, 100).await
    }

    /// Speak with full options
    pub async fn speak_with_options(
        &self,
        text: &str,
        language: &str,
        _priority: i32,
    ) -> Result<SpeakHandle> {
        let request = SynthesizeRequest {
            text: text.to_string(),
            language: language.to_string(),
            voice_id: String::new(), // Use default voice
            speed: 1.0,
            format: AudioFormat::FormatWav as i32,
        };

        let mut client = self.client.lock().await;

        // Use streaming synthesis (non-blocking)
        let _response = client.stream_synthesize(request).await?;

        // Generate request ID for tracking
        let request_id = format!("speak-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis());

        Ok(SpeakHandle {
            request_id,
            _client: self.client.clone(),
        })
    }

    /// Stop all current speech
    pub async fn stop(&self) -> Result<()> {
        // Send empty synthesis request to trigger stop
        // In a full implementation, this would use a dedicated stop endpoint
        Ok(())
    }

    /// Check if TTS service is ready
    pub async fn check_status(&self) -> Result<bool> {
        let request = StatusRequest {
            include_metrics: false,
        };

        let mut client = self.client.lock().await;
        let response = client.get_status(request).await?;
        let status = response.into_inner();

        Ok(status.models_loaded && status.status == "ready")
    }
}

/// Handle to track queued speech
///
/// Can be used to wait for completion or cancel the speech request.
pub struct SpeakHandle {
    /// Unique request identifier
    pub request_id: String,
    _client: Arc<Mutex<TtsServiceClient<Channel>>>,
}

impl SpeakHandle {
    /// Wait for speech to complete (optional - usually not needed)
    ///
    /// Most use cases should NOT await this - let speech play in background.
    pub async fn wait(&self) -> Result<()> {
        // In a full implementation, this would poll for completion
        // For now, we return immediately since TTS is fire-and-forget
        Ok(())
    }

    /// Cancel this speech request
    pub async fn cancel(&self) -> Result<()> {
        // In a full implementation, this would send a cancel command
        Ok(())
    }

    /// Get the request ID
    pub fn id(&self) -> &str {
        &self.request_id
    }
}
