//! Filtered STT with self-speech filtering and speaker diarization
//!
//! This is the recommended API for full-duplex voice conversation.
//! It filters out agent speech from microphone input.

use crate::client::TtsState;
use crate::error::{Result, VoiceError};
use crate::proto::{
    filtered_stt_service_client::FilteredSttServiceClient,
    FilteredListenRequest, FilteredAudioInput, FilterConfig as ProtoFilterConfig,
    Empty, RegisterSpeakerRequest, SttStreamingConfig,
    filtered_listen_request::Payload,
};

use std::pin::Pin;
use std::sync::Arc;
use futures::Stream;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tonic::transport::Channel;

/// Self-speech filtered STT client
///
/// Uses the multi-layer filtering system to separate agent voice from user voice:
/// - Layer 1: Text matching (TTS queue provides known text)
/// - Layer 2: Acoustic Echo Cancellation (AEC)
/// - Layer 3: Speaker embedding (voice identity)
/// - Layer 4: Temporal gating (TTS state)
///
/// # Example
///
/// ```rust,no_run
/// use dashflow_voice::FilteredSTT;
///
/// async fn example(stt: FilteredSTT) {
///     // Listen with self-speech filtering
///     let result = stt.listen_with_session("session-1").await.unwrap();
///
///     if result.confidence > 0.7 {
///         println!("User said: {}", result.text);
///     }
/// }
/// ```
#[derive(Clone)]
pub struct FilteredSTT {
    client: Arc<Mutex<FilteredSttServiceClient<Channel>>>,
    language: String,
}

impl FilteredSTT {
    /// Create a new filtered STT client
    pub fn new(channel: Channel, language: String) -> Self {
        Self {
            client: Arc::new(Mutex::new(FilteredSttServiceClient::new(channel))),
            language,
        }
    }

    /// Listen with self-speech filtering
    ///
    /// This is the primary method for full-duplex conversation.
    /// Agent speech is automatically filtered from the microphone input.
    pub async fn listen_with_session(&self, session_id: &str) -> Result<FilteredTranscription> {
        self.listen_with_config(session_id, FilterConfig::default()).await
    }

    /// Listen with custom filter configuration
    pub async fn listen_with_config(
        &self,
        session_id: &str,
        config: FilterConfig,
    ) -> Result<FilteredTranscription> {
        let mut client = self.client.lock().await;

        // Create request stream
        let request_stream = create_filtered_listen_stream(
            session_id,
            &self.language,
            config,
        );

        let response = client.start_filtered_listen(request_stream).await?;
        let mut stream = response.into_inner();

        // Wait for final transcription
        let mut result = FilteredTranscription::default();

        while let Some(event) = stream.next().await {
            let event = event?;

            if event.is_final {
                result = FilteredTranscription {
                    text: event.text,
                    speaker_id: event.speaker_id,
                    confidence: event.confidence,
                    speaker_confidence: event.speaker_confidence,
                    timestamp_ms: event.timestamp_ms as u64,
                    is_final: true,
                    filtering_details: event.filtering.map(|f| FilteringDetails {
                        filtered_agent_speech: f.filtered_agent_speech,
                        agent_confidence: f.agent_confidence,
                        text_match_score: f.text_match_score,
                        aec_suppression_ratio: f.aec_suppression_ratio,
                        text_match_active: f.text_match_active,
                        aec_active: f.aec_active,
                        speaker_id_active: f.speaker_id_active,
                    }),
                };
                break;
            }
        }

        Ok(result)
    }

    /// Start a streaming filtered listen session
    ///
    /// Returns a stream of transcription events as speech is detected.
    pub async fn listen_stream(
        &self,
        session_id: &str,
        config: FilterConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<FilteredTranscription>> + Send>>> {
        let mut client = self.client.lock().await;

        let request_stream = create_filtered_listen_stream(
            session_id,
            &self.language,
            config,
        );

        let response = client.start_filtered_listen(request_stream).await?;
        let stream = response.into_inner();

        let mapped = stream.map(|result| {
            result
                .map(|event| FilteredTranscription {
                    text: event.text,
                    speaker_id: event.speaker_id,
                    confidence: event.confidence,
                    speaker_confidence: event.speaker_confidence,
                    timestamp_ms: event.timestamp_ms as u64,
                    is_final: event.is_final,
                    filtering_details: event.filtering.map(|f| FilteringDetails {
                        filtered_agent_speech: f.filtered_agent_speech,
                        agent_confidence: f.agent_confidence,
                        text_match_score: f.text_match_score,
                        aec_suppression_ratio: f.aec_suppression_ratio,
                        text_match_active: f.text_match_active,
                        aec_active: f.aec_active,
                        speaker_id_active: f.speaker_id_active,
                    }),
                })
                .map_err(VoiceError::from)
        });

        Ok(Box::pin(mapped))
    }

    /// Get current TTS state
    pub async fn get_tts_state(&self) -> Result<TtsState> {
        let mut client = self.client.lock().await;
        let response = client.get_tts_state(Empty {}).await?;
        let state = response.into_inner();

        Ok(TtsState {
            is_speaking: state.is_speaking,
            current_text: state.current_text,
            elapsed_ms: state.elapsed_ms,
            estimated_duration_ms: state.estimated_duration_ms,
        })
    }

    /// Register a speaker for diarization
    pub async fn register_speaker(
        &self,
        speaker_id: &str,
        audio_samples: Vec<u8>,
        sample_rate: i32,
    ) -> Result<String> {
        let mut client = self.client.lock().await;

        let request = RegisterSpeakerRequest {
            speaker_id: speaker_id.to_string(),
            audio_samples,
            sample_rate,
        };

        let response = client.register_speaker(request).await?;
        let result = response.into_inner();

        if result.success {
            Ok(result.speaker_id)
        } else {
            Err(VoiceError::Internal(result.error_message))
        }
    }

    /// Configure filter settings
    pub async fn configure(&self, config: FilterConfig) -> Result<FilterConfig> {
        let mut client = self.client.lock().await;

        let proto_config = config.to_proto();
        let response = client.configure_filter(proto_config).await?;
        let result = response.into_inner();

        if result.success {
            result.current_config
                .map(FilterConfig::from_proto)
                .ok_or_else(|| VoiceError::Internal("No config returned".to_string()))
        } else {
            Err(VoiceError::Internal(result.error_message))
        }
    }
}

/// Filtered transcription result
#[derive(Debug, Clone, Default)]
pub struct FilteredTranscription {
    /// User speech only (agent filtered out)
    pub text: String,
    /// Identified speaker ID
    pub speaker_id: String,
    /// Confidence this is user speech (0-1)
    pub confidence: f32,
    /// Speaker identification confidence (0-1)
    pub speaker_confidence: f32,
    /// Timestamp from stream start
    pub timestamp_ms: u64,
    /// Whether this is a final result
    pub is_final: bool,
    /// Details about what was filtered
    pub filtering_details: Option<FilteringDetails>,
}

impl std::fmt::Display for FilteredTranscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Details about the filtering applied
#[derive(Debug, Clone)]
pub struct FilteringDetails {
    /// Text filtered out as agent speech
    pub filtered_agent_speech: String,
    /// Confidence the filtered text was agent
    pub agent_confidence: f32,
    /// Text matching similarity score
    pub text_match_score: f32,
    /// AEC echo suppression ratio
    pub aec_suppression_ratio: f32,
    /// Whether text matching was active
    pub text_match_active: bool,
    /// Whether AEC was active
    pub aec_active: bool,
    /// Whether speaker ID was active
    pub speaker_id_active: bool,
}

/// Filter configuration options
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Enable text-based filtering
    pub enable_text_matching: bool,
    /// Enable acoustic echo cancellation
    pub enable_aec: bool,
    /// Enable speaker diarization
    pub enable_speaker_diarization: bool,
    /// Threshold for agent detection (0-1)
    pub agent_confidence_threshold: f32,
    /// Weight for text matching (default: 0.3)
    pub text_match_weight: f32,
    /// Weight for AEC (default: 0.2)
    pub aec_weight: f32,
    /// Weight for speaker ID (default: 0.5)
    pub speaker_id_weight: f32,
    /// Cosine similarity threshold for speaker matching
    pub speaker_threshold: f32,
    /// Auto-learn new speakers
    pub enable_speaker_learning: bool,
    /// Enable debug logging
    pub debug_logging: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            enable_text_matching: true,
            enable_aec: true,
            enable_speaker_diarization: true,
            agent_confidence_threshold: 0.7,
            text_match_weight: 0.3,
            aec_weight: 0.2,
            speaker_id_weight: 0.5,
            speaker_threshold: 0.6,
            enable_speaker_learning: true,
            debug_logging: false,
        }
    }
}

impl FilterConfig {
    /// Create config with all filtering enabled (default)
    pub fn all_enabled() -> Self {
        Self::default()
    }

    /// Create config with only speaker diarization
    pub fn speaker_only() -> Self {
        Self {
            enable_text_matching: false,
            enable_aec: false,
            enable_speaker_diarization: true,
            ..Default::default()
        }
    }

    /// Create config with no filtering (pass-through)
    pub fn disabled() -> Self {
        Self {
            enable_text_matching: false,
            enable_aec: false,
            enable_speaker_diarization: false,
            ..Default::default()
        }
    }

    /// Convert to protobuf message
    pub fn to_proto(&self) -> ProtoFilterConfig {
        ProtoFilterConfig {
            enable_text_matching: self.enable_text_matching,
            enable_aec: self.enable_aec,
            enable_speaker_diarization: self.enable_speaker_diarization,
            agent_confidence_threshold: self.agent_confidence_threshold,
            text_match_weight: self.text_match_weight,
            aec_weight: self.aec_weight,
            speaker_id_weight: self.speaker_id_weight,
            speaker_threshold: self.speaker_threshold,
            enable_speaker_learning: self.enable_speaker_learning,
            debug_logging: self.debug_logging,
        }
    }

    /// Create from protobuf message
    pub fn from_proto(proto: ProtoFilterConfig) -> Self {
        Self {
            enable_text_matching: proto.enable_text_matching,
            enable_aec: proto.enable_aec,
            enable_speaker_diarization: proto.enable_speaker_diarization,
            agent_confidence_threshold: proto.agent_confidence_threshold,
            text_match_weight: proto.text_match_weight,
            aec_weight: proto.aec_weight,
            speaker_id_weight: proto.speaker_id_weight,
            speaker_threshold: proto.speaker_threshold,
            enable_speaker_learning: proto.enable_speaker_learning,
            debug_logging: proto.debug_logging,
        }
    }
}

/// Create a filtered listen request stream
fn create_filtered_listen_stream(
    session_id: &str,
    language: &str,
    config: FilterConfig,
) -> impl Stream<Item = FilteredListenRequest> {
    let session_id = session_id.to_string();
    let _language = language.to_string();

    // Initial config message
    let config_msg = FilteredListenRequest {
        session_id: session_id.clone(),
        payload: Some(Payload::Config(config.to_proto())),
        reset_state: false,
    };

    // Initial audio message with STT config
    let audio_init = FilteredListenRequest {
        session_id: session_id.clone(),
        payload: Some(Payload::Audio(FilteredAudioInput {
            audio: vec![],
            sample_rate: 16000,
            end_of_stream: false,
            stt_config: Some(SttStreamingConfig {
                step_ms: 3000,
                length_ms: 10000,
                keep_ms: 200,
                vad_threshold: 0.6,
                freq_threshold: 100.0,
                silence_threshold_ms: 1500,
                use_vad_segmentation: Some(true),
                max_segment_duration_ms: 30000,
            }),
            metadata: None,
        })),
        reset_state: false,
    };

    // End of stream message
    let end_msg = FilteredListenRequest {
        session_id,
        payload: Some(Payload::Audio(FilteredAudioInput {
            audio: vec![],
            sample_rate: 16000,
            end_of_stream: true,
            stt_config: None,
            metadata: None,
        })),
        reset_state: false,
    };

    // In a real implementation, this would be an async stream from audio capture
    futures::stream::iter(vec![config_msg, audio_init, end_msg])
}
