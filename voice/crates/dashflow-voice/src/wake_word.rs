//! Wake Word Detection Client
//!
//! Provides wake word detection for hands-free voice activation.
//! Integrates with self-speech filtering to prevent false activations during TTS playback.

use crate::error::{Result, VoiceError};
use crate::proto::{
    wake_word_service_client::WakeWordServiceClient,
    Empty, WakeWordListenConfig, WakeWordDetectionEvent,
    WakeWordEnabledRequest, WakeWordStatusResponse,
    WakeWordTestRequest, WakeWordTestResponse,
};

use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tonic::transport::Channel;

/// Wake word detection client
///
/// Enables hands-free voice activation by detecting custom wake words
/// like "Hey Voice" or "Hey Agent".
///
/// # Features
///
/// - Local ONNX-based detection (no cloud required)
/// - Custom wake word support
/// - Integration with self-speech filter (won't trigger on TTS playback)
/// - Configurable detection threshold
///
/// # Example
///
/// ```rust,ignore
/// use dashflow_voice::{WakeWordClient, WakeWordConfig};
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let client = WakeWordClient::connect("http://localhost:50051").await?;
///
///     // Start listening for wake words
///     let mut stream = client.start_listening(WakeWordConfig::default()).await?;
///
///     while let Some(event) = stream.next().await {
///         match event {
///             Ok(detection) => {
///                 println!("Wake word '{}' detected with confidence {}",
///                          detection.wake_word, detection.confidence);
///                 // Trigger STT listening here
///             }
///             Err(e) => eprintln!("Detection error: {}", e),
///         }
///     }
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct WakeWordClient {
    client: Arc<Mutex<WakeWordServiceClient<Channel>>>,
}

impl WakeWordClient {
    /// Connect to wake word service
    pub async fn connect(endpoint: &str) -> Result<Self> {
        let channel = Channel::from_shared(endpoint.to_string())
            .map_err(|e| VoiceError::InvalidConfig(e.to_string()))?
            .connect()
            .await?;

        Ok(Self {
            client: Arc::new(Mutex::new(WakeWordServiceClient::new(channel))),
        })
    }

    /// Create from existing channel
    pub fn new(channel: Channel) -> Self {
        Self {
            client: Arc::new(Mutex::new(WakeWordServiceClient::new(channel))),
        }
    }

    /// Start listening for wake words
    ///
    /// Returns a stream of detection events. Each event contains:
    /// - The detected wake word name
    /// - Detection confidence (0.0-1.0)
    /// - Timestamp and latency information
    pub async fn start_listening(
        &self,
        config: WakeWordConfig,
    ) -> Result<WakeWordStream> {
        let proto_config = WakeWordListenConfig {
            wake_words: config.wake_words,
            threshold: config.threshold,
            refractory_ms: config.refractory_ms,
            enable_self_speech_filter: config.enable_self_speech_filter,
            play_activation_sound: config.play_activation_sound,
            activation_sound_path: config.activation_sound_path,
        };

        let mut client = self.client.lock().await;
        let response = client.start_listening(proto_config).await?;
        let stream = response.into_inner();

        Ok(WakeWordStream { inner: stream })
    }

    /// Stop listening for wake words
    pub async fn stop_listening(&self) -> Result<()> {
        let mut client = self.client.lock().await;
        client.stop_listening(Empty {}).await?;
        Ok(())
    }

    /// Enable or disable wake word detection
    pub async fn set_enabled(&self, enabled: bool) -> Result<bool> {
        let request = WakeWordEnabledRequest { enabled };
        let mut client = self.client.lock().await;
        let response = client.set_enabled(request).await?;
        let inner = response.into_inner();

        if !inner.success {
            return Err(VoiceError::Internal(inner.error_message));
        }

        Ok(inner.enabled)
    }

    /// Get current wake word status and statistics
    pub async fn get_status(&self) -> Result<WakeWordStatus> {
        let mut client = self.client.lock().await;
        let response = client.get_status(Empty {}).await?;
        let status = response.into_inner();

        Ok(WakeWordStatus::from(status))
    }

    /// List available wake word models
    pub async fn list_models(&self) -> Result<Vec<WakeWordModel>> {
        let mut client = self.client.lock().await;
        let response = client.list_models(Empty {}).await?;
        let list = response.into_inner();

        Ok(list.models.into_iter().map(WakeWordModel::from).collect())
    }

    /// Test wake word detection with audio sample
    ///
    /// Useful for verifying wake word detection is working correctly.
    pub async fn test_detection(
        &self,
        audio: Vec<u8>,
        sample_rate: i32,
        wake_word: Option<&str>,
    ) -> Result<WakeWordTestResult> {
        let request = WakeWordTestRequest {
            audio,
            sample_rate,
            wake_word: wake_word.unwrap_or("").to_string(),
        };

        let mut client = self.client.lock().await;
        let response = client.test_detection(request).await?;
        let result = response.into_inner();

        Ok(WakeWordTestResult::from(result))
    }
}

/// Wake word detection configuration
#[derive(Clone, Debug)]
pub struct WakeWordConfig {
    /// Wake words to detect (empty = detect all loaded models)
    pub wake_words: Vec<String>,
    /// Detection threshold (0.0-1.0, default 0.5)
    pub threshold: f32,
    /// Minimum time between activations in ms (default 2000)
    pub refractory_ms: i32,
    /// Enable self-speech filter (default true)
    pub enable_self_speech_filter: bool,
    /// Play activation sound on detection (default true)
    pub play_activation_sound: bool,
    /// Custom activation sound path (empty = default chime)
    pub activation_sound_path: String,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            wake_words: Vec::new(), // Detect all loaded models
            threshold: 0.5,
            refractory_ms: 2000,
            enable_self_speech_filter: true,
            play_activation_sound: true,
            activation_sound_path: String::new(),
        }
    }
}

impl WakeWordConfig {
    /// Create config for specific wake words
    pub fn for_wake_words(wake_words: Vec<String>) -> Self {
        Self {
            wake_words,
            ..Default::default()
        }
    }

    /// Set detection threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Disable self-speech filter
    pub fn without_self_speech_filter(mut self) -> Self {
        self.enable_self_speech_filter = false;
        self
    }

    /// Disable activation sound
    pub fn without_activation_sound(mut self) -> Self {
        self.play_activation_sound = false;
        self
    }
}

/// Stream of wake word detection events
pub struct WakeWordStream {
    inner: tonic::Streaming<WakeWordDetectionEvent>,
}

impl WakeWordStream {
    /// Get the next detection event
    ///
    /// Returns None when the stream ends.
    pub async fn next(&mut self) -> Option<Result<WakeWordDetection>> {
        match self.inner.next().await {
            Some(Ok(event)) => Some(Ok(WakeWordDetection::from(event))),
            Some(Err(e)) => Some(Err(VoiceError::GrpcStatus(e))),
            None => None,
        }
    }
}

/// Wake word detection event
#[derive(Clone, Debug)]
pub struct WakeWordDetection {
    /// Name of detected wake word (e.g., "hey_voice", "alexa")
    pub wake_word: String,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
    /// Timestamp in ms since listening started
    pub timestamp_ms: i64,
    /// Latency from audio end to callback in ms
    pub latency_ms: i64,
    /// Whether activation sound was played
    pub activation_sound_played: bool,
}

impl From<WakeWordDetectionEvent> for WakeWordDetection {
    fn from(event: WakeWordDetectionEvent) -> Self {
        Self {
            wake_word: event.wake_word,
            confidence: event.confidence,
            timestamp_ms: event.timestamp_ms,
            latency_ms: event.latency_ms,
            activation_sound_played: event.activation_sound_played,
        }
    }
}

/// Wake word service status
#[derive(Clone, Debug)]
pub struct WakeWordStatus {
    /// Whether actively listening
    pub is_listening: bool,
    /// Whether detection is enabled
    pub is_enabled: bool,
    /// Whether TTS is active (detection paused)
    pub is_speaking: bool,
    /// Names of loaded wake word models
    pub loaded_models: Vec<String>,
    /// Current detection threshold
    pub threshold: f32,
    /// Detection statistics
    pub stats: WakeWordStats,
}

impl From<WakeWordStatusResponse> for WakeWordStatus {
    fn from(response: WakeWordStatusResponse) -> Self {
        Self {
            is_listening: response.is_listening,
            is_enabled: response.is_enabled,
            is_speaking: response.is_speaking,
            loaded_models: response.loaded_models,
            threshold: response.threshold,
            stats: response.stats.map(WakeWordStats::from).unwrap_or_default(),
        }
    }
}

/// Wake word detection statistics
#[derive(Clone, Debug, Default)]
pub struct WakeWordStats {
    /// Total audio chunks processed
    pub chunks_processed: u64,
    /// Total wake word detections
    pub total_detections: u64,
    /// Detections blocked by self-speech filter
    pub false_accepts_filtered: u64,
    /// Average inference time in ms
    pub avg_inference_ms: f32,
    /// Uptime in seconds
    pub uptime_seconds: f32,
}

impl From<crate::proto::WakeWordStats> for WakeWordStats {
    fn from(stats: crate::proto::WakeWordStats) -> Self {
        Self {
            chunks_processed: stats.chunks_processed,
            total_detections: stats.total_detections,
            false_accepts_filtered: stats.false_accepts_filtered,
            avg_inference_ms: stats.avg_inference_ms,
            uptime_seconds: stats.uptime_seconds,
        }
    }
}

/// Wake word model information
#[derive(Clone, Debug)]
pub struct WakeWordModel {
    /// Model name (e.g., "hey_voice", "alexa")
    pub name: String,
    /// Path to model file
    pub path: String,
    /// Whether model is loaded
    pub is_loaded: bool,
    /// Whether this is a custom-trained model
    pub is_custom: bool,
    /// Model description
    pub description: String,
}

impl From<crate::proto::WakeWordModelInfo> for WakeWordModel {
    fn from(info: crate::proto::WakeWordModelInfo) -> Self {
        Self {
            name: info.name,
            path: info.path,
            is_loaded: info.is_loaded,
            is_custom: info.is_custom,
            description: info.description,
        }
    }
}

/// Wake word test result
#[derive(Clone, Debug)]
pub struct WakeWordTestResult {
    /// Whether wake word was detected
    pub detected: bool,
    /// Detection confidence
    pub confidence: f32,
    /// Name of detected wake word (if any)
    pub wake_word: String,
    /// Error message if any
    pub error_message: String,
    /// Scores for all wake word models
    pub scores: Vec<(String, f32)>,
}

impl From<WakeWordTestResponse> for WakeWordTestResult {
    fn from(response: WakeWordTestResponse) -> Self {
        Self {
            detected: response.detected,
            confidence: response.confidence,
            wake_word: response.wake_word,
            error_message: response.error_message,
            scores: response.scores.into_iter()
                .map(|s| (s.wake_word, s.score))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WakeWordConfig::default();
        assert!(config.wake_words.is_empty());
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.refractory_ms, 2000);
        assert!(config.enable_self_speech_filter);
        assert!(config.play_activation_sound);
    }

    #[test]
    fn test_config_builder() {
        let config = WakeWordConfig::for_wake_words(vec!["hey_voice".to_string()])
            .with_threshold(0.7)
            .without_activation_sound();

        assert_eq!(config.wake_words, vec!["hey_voice"]);
        assert_eq!(config.threshold, 0.7);
        assert!(!config.play_activation_sound);
        assert!(config.enable_self_speech_filter);
    }
}
