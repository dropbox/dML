//! Summarization Service Client
//!
//! Provides text and audio summarization using Qwen3-8B via llama.cpp.
//! Supports multiple summarization modes and live streaming summaries.

use crate::error::{Result, VoiceError};
use crate::proto::{
    summarization_service_client::SummarizationServiceClient,
    Empty, SummarizeTextRequest, SummarizeResponse, SummaryEvent,
    SummarizationStatusResponse, SummarizationMode as ProtoSummarizationMode,
    AudioInput,
};

use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tonic::transport::Channel;

/// Summarization service client
///
/// Enables text and audio summarization using local LLM inference.
/// Supports multiple modes (brief, standard, detailed, action items).
///
/// # Features
///
/// - Local Qwen3-8B inference (no cloud required)
/// - Multiple summarization modes
/// - Audio-to-summary pipeline (STT → LLM)
/// - Live streaming summaries with triggers
///
/// # Example
///
/// ```rust,ignore
/// use dashflow_voice::{SummarizationClient, SummarizationMode, SummarizationConfig};
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let client = SummarizationClient::connect("http://localhost:50051").await?;
///
///     // Check if model is loaded
///     let status = client.get_status().await?;
///     println!("Model loaded: {}", status.model_loaded);
///
///     // Summarize text
///     let config = SummarizationConfig::default();
///     let summary = client.summarize("Long text to summarize...", config).await?;
///     println!("Summary: {}", summary.text);
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct SummarizationClient {
    client: Arc<Mutex<SummarizationServiceClient<Channel>>>,
}

impl SummarizationClient {
    /// Connect to summarization service
    pub async fn connect(endpoint: &str) -> Result<Self> {
        let channel = Channel::from_shared(endpoint.to_string())
            .map_err(|e| VoiceError::InvalidConfig(e.to_string()))?
            .connect()
            .await?;

        Ok(Self {
            client: Arc::new(Mutex::new(SummarizationServiceClient::new(channel))),
        })
    }

    /// Create from existing channel
    pub fn new(channel: Channel) -> Self {
        Self {
            client: Arc::new(Mutex::new(SummarizationServiceClient::new(channel))),
        }
    }

    /// Summarize text
    ///
    /// # Arguments
    /// * `text` - Text to summarize
    /// * `config` - Summarization configuration
    ///
    /// # Returns
    /// Summary result with text and metrics
    pub async fn summarize(
        &self,
        text: &str,
        config: SummarizationConfig,
    ) -> Result<Summary> {
        let request = SummarizeTextRequest {
            text: text.to_string(),
            mode: config.mode.to_proto() as i32,
            source_language: config.source_language,
            target_language: config.target_language,
            max_tokens: config.max_tokens,
        };

        let mut client = self.client.lock().await;
        let response = client.summarize(request).await?;
        let inner = response.into_inner();

        Ok(Summary::from(inner))
    }

    /// Summarize text with default configuration
    pub async fn summarize_text(&self, text: &str) -> Result<Summary> {
        self.summarize(text, SummarizationConfig::default()).await
    }

    /// Summarize text in brief mode (1 sentence)
    pub async fn summarize_brief(&self, text: &str) -> Result<Summary> {
        self.summarize(text, SummarizationConfig::brief()).await
    }

    /// Summarize text extracting action items only
    pub async fn summarize_action_items(&self, text: &str) -> Result<Summary> {
        self.summarize(text, SummarizationConfig::action_items()).await
    }

    /// Start live summarization with streaming audio
    ///
    /// Returns a stream of summary events triggered by time, length, or silence.
    ///
    /// # Arguments
    /// * `audio_stream` - Stream of audio chunks
    ///
    /// # Returns
    /// Stream of summary events
    pub async fn live_summarize<S>(
        &self,
        audio_stream: S,
    ) -> Result<LiveSummaryStream>
    where
        S: futures::Stream<Item = AudioChunk> + Send + 'static,
    {
        let mapped_stream = audio_stream.map(|chunk| AudioInput {
            audio: chunk.audio,
            sample_rate: chunk.sample_rate,
            end_of_stream: chunk.end_of_stream,
            language: chunk.language,
            translate_to_english: false,
            config: None,
        });

        let mut client = self.client.lock().await;
        let response = client
            .live_summarize(mapped_stream)
            .await?;
        let stream = response.into_inner();

        Ok(LiveSummaryStream { inner: stream })
    }

    /// Get summarization service status
    ///
    /// Returns model loading state, memory usage, and performance metrics.
    pub async fn get_status(&self) -> Result<SummarizationStatus> {
        let mut client = self.client.lock().await;
        let response = client.get_summarization_status(Empty {}).await?;
        let status = response.into_inner();

        Ok(SummarizationStatus::from(status))
    }

    /// Check if the summarization model is loaded and ready
    pub async fn is_ready(&self) -> Result<bool> {
        let status = self.get_status().await?;
        Ok(status.model_loaded)
    }
}

/// Summarization mode
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SummarizationMode {
    /// 1 sentence, <30 words
    Brief,
    /// 2-3 sentences, key points (default)
    #[default]
    Standard,
    /// Bullet points, all topics
    Detailed,
    /// Only actionable items
    ActionItems,
    /// Tool call descriptions
    ToolCall,
}

impl SummarizationMode {
    /// Convert to proto enum value
    fn to_proto(self) -> ProtoSummarizationMode {
        match self {
            SummarizationMode::Brief => ProtoSummarizationMode::ModeBrief,
            SummarizationMode::Standard => ProtoSummarizationMode::ModeStandard,
            SummarizationMode::Detailed => ProtoSummarizationMode::ModeDetailed,
            SummarizationMode::ActionItems => ProtoSummarizationMode::ModeActionItems,
            SummarizationMode::ToolCall => ProtoSummarizationMode::ModeToolCall,
        }
    }

    /// Create from proto enum value
    #[allow(dead_code)]
    fn from_proto(value: i32) -> Self {
        match value {
            0 => SummarizationMode::Brief,
            1 => SummarizationMode::Standard,
            2 => SummarizationMode::Detailed,
            3 => SummarizationMode::ActionItems,
            4 => SummarizationMode::ToolCall,
            _ => SummarizationMode::Standard,
        }
    }
}

/// Summarization configuration
#[derive(Clone, Debug)]
pub struct SummarizationConfig {
    /// Summarization mode
    pub mode: SummarizationMode,
    /// Source language (for cross-lingual summarization)
    pub source_language: String,
    /// Target language for output (empty = same as source)
    pub target_language: String,
    /// Maximum tokens to generate (default: 150)
    pub max_tokens: i32,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            mode: SummarizationMode::Standard,
            source_language: String::new(),
            target_language: String::new(),
            max_tokens: 150,
        }
    }
}

impl SummarizationConfig {
    /// Create brief summarization config
    pub fn brief() -> Self {
        Self {
            mode: SummarizationMode::Brief,
            max_tokens: 50,
            ..Default::default()
        }
    }

    /// Create detailed summarization config
    pub fn detailed() -> Self {
        Self {
            mode: SummarizationMode::Detailed,
            max_tokens: 300,
            ..Default::default()
        }
    }

    /// Create action items summarization config
    pub fn action_items() -> Self {
        Self {
            mode: SummarizationMode::ActionItems,
            max_tokens: 200,
            ..Default::default()
        }
    }

    /// Create cross-lingual summarization config
    pub fn cross_lingual(source: &str, target: &str) -> Self {
        Self {
            source_language: source.to_string(),
            target_language: target.to_string(),
            ..Default::default()
        }
    }

    /// Set summarization mode
    pub fn with_mode(mut self, mode: SummarizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set source language
    pub fn with_source_language(mut self, language: &str) -> Self {
        self.source_language = language.to_string();
        self
    }

    /// Set target language
    pub fn with_target_language(mut self, language: &str) -> Self {
        self.target_language = language.to_string();
        self
    }
}

/// Summary result
#[derive(Clone, Debug)]
pub struct Summary {
    /// Summary text
    pub text: String,
    /// Processing latency in milliseconds
    pub latency_ms: i64,
    /// Detected/specified source language
    pub source_language: String,
    /// Number of input tokens processed
    pub input_tokens: i32,
    /// Number of output tokens generated
    pub output_tokens: i32,
}

impl From<SummarizeResponse> for Summary {
    fn from(response: SummarizeResponse) -> Self {
        Self {
            text: response.summary,
            latency_ms: response.latency_ms,
            source_language: response.source_language,
            input_tokens: response.input_token_count,
            output_tokens: response.output_token_count,
        }
    }
}

/// Audio chunk for streaming summarization
#[derive(Clone, Debug)]
pub struct AudioChunk {
    /// Raw PCM16 audio data
    pub audio: Vec<u8>,
    /// Sample rate in Hz (default: 16000)
    pub sample_rate: i32,
    /// Whether this is the last chunk
    pub end_of_stream: bool,
    /// Source language (empty = auto-detect)
    pub language: String,
}

impl AudioChunk {
    /// Create new audio chunk
    pub fn new(audio: Vec<u8>) -> Self {
        Self {
            audio,
            sample_rate: 16000,
            end_of_stream: false,
            language: String::new(),
        }
    }

    /// Create end-of-stream marker
    pub fn end() -> Self {
        Self {
            audio: Vec::new(),
            sample_rate: 16000,
            end_of_stream: true,
            language: String::new(),
        }
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: i32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set language
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = language.to_string();
        self
    }
}

/// Stream of live summary events
pub struct LiveSummaryStream {
    inner: tonic::Streaming<SummaryEvent>,
}

impl LiveSummaryStream {
    /// Get the next summary event
    ///
    /// Returns None when the stream ends.
    pub async fn next(&mut self) -> Option<Result<LiveSummary>> {
        match self.inner.next().await {
            Some(Ok(event)) => Some(Ok(LiveSummary::from(event))),
            Some(Err(e)) => Some(Err(VoiceError::GrpcStatus(e))),
            None => None,
        }
    }
}

/// Live summary event
#[derive(Clone, Debug)]
pub struct LiveSummary {
    /// Summary text
    pub text: String,
    /// When summary was generated (ms since stream start)
    pub timestamp_ms: i64,
    /// What triggered this summary ("time", "length", "silence", "manual")
    pub trigger_reason: String,
    /// Number of words in source transcript
    pub transcript_words: i32,
    /// Processing latency in milliseconds
    pub latency_ms: i64,
}

impl From<SummaryEvent> for LiveSummary {
    fn from(event: SummaryEvent) -> Self {
        Self {
            text: event.summary,
            timestamp_ms: event.timestamp_ms,
            trigger_reason: event.trigger_reason,
            transcript_words: event.transcript_words,
            latency_ms: event.latency_ms,
        }
    }
}

/// Summarization service status
#[derive(Clone, Debug)]
pub struct SummarizationStatus {
    /// Whether Qwen3 model is loaded
    pub model_loaded: bool,
    /// Path to loaded model
    pub model_path: String,
    /// Model memory usage in bytes
    pub memory_usage_bytes: i64,
    /// Whether GPU (Metal) is enabled
    pub gpu_enabled: bool,
    /// Model context window size
    pub context_size: i32,
    /// Performance metrics
    pub metrics: SummarizationMetrics,
}

impl From<SummarizationStatusResponse> for SummarizationStatus {
    fn from(response: SummarizationStatusResponse) -> Self {
        Self {
            model_loaded: response.model_loaded,
            model_path: response.model_path,
            memory_usage_bytes: response.memory_usage_bytes,
            gpu_enabled: response.gpu_enabled,
            context_size: response.context_size,
            metrics: response.metrics.map(SummarizationMetrics::from).unwrap_or_default(),
        }
    }
}

/// Summarization performance metrics
#[derive(Clone, Debug, Default)]
pub struct SummarizationMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Tokens per second throughput
    pub tokens_per_second: f32,
}

impl From<crate::proto::SummarizationMetrics> for SummarizationMetrics {
    fn from(metrics: crate::proto::SummarizationMetrics) -> Self {
        Self {
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            avg_latency_ms: metrics.avg_latency_ms,
            total_tokens_generated: metrics.total_tokens_generated,
            tokens_per_second: metrics.tokens_per_second,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SummarizationConfig::default();
        assert_eq!(config.mode, SummarizationMode::Standard);
        assert_eq!(config.max_tokens, 150);
        assert!(config.source_language.is_empty());
        assert!(config.target_language.is_empty());
    }

    #[test]
    fn test_brief_config() {
        let config = SummarizationConfig::brief();
        assert_eq!(config.mode, SummarizationMode::Brief);
        assert_eq!(config.max_tokens, 50);
    }

    #[test]
    fn test_config_builder() {
        let config = SummarizationConfig::default()
            .with_mode(SummarizationMode::Detailed)
            .with_max_tokens(500)
            .with_source_language("ja")
            .with_target_language("en");

        assert_eq!(config.mode, SummarizationMode::Detailed);
        assert_eq!(config.max_tokens, 500);
        assert_eq!(config.source_language, "ja");
        assert_eq!(config.target_language, "en");
    }

    #[test]
    fn test_cross_lingual_config() {
        let config = SummarizationConfig::cross_lingual("ja", "en");
        assert_eq!(config.source_language, "ja");
        assert_eq!(config.target_language, "en");
        assert_eq!(config.mode, SummarizationMode::Standard);
    }

    #[test]
    fn test_audio_chunk() {
        let chunk = AudioChunk::new(vec![1, 2, 3, 4])
            .with_sample_rate(48000)
            .with_language("en");

        assert_eq!(chunk.audio, vec![1, 2, 3, 4]);
        assert_eq!(chunk.sample_rate, 48000);
        assert_eq!(chunk.language, "en");
        assert!(!chunk.end_of_stream);
    }

    #[test]
    fn test_audio_chunk_end() {
        let chunk = AudioChunk::end();
        assert!(chunk.audio.is_empty());
        assert!(chunk.end_of_stream);
    }

    #[test]
    fn test_mode_conversion() {
        assert_eq!(
            SummarizationMode::Brief.to_proto(),
            ProtoSummarizationMode::ModeBrief
        );
        assert_eq!(
            SummarizationMode::Standard.to_proto(),
            ProtoSummarizationMode::ModeStandard
        );
        assert_eq!(
            SummarizationMode::from_proto(0),
            SummarizationMode::Brief
        );
        assert_eq!(
            SummarizationMode::from_proto(1),
            SummarizationMode::Standard
        );
        // Unknown value should default to Standard
        assert_eq!(
            SummarizationMode::from_proto(99),
            SummarizationMode::Standard
        );
    }
}
