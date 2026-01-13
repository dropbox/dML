//! Speech-to-Text (STT) client
//!
//! Provides speech recognition (blocking - waits for user speech).

use crate::error::{Result, VoiceError};
use crate::proto::{
    stt_service_client::SttServiceClient,
    AudioInput, SttStreamingConfig,
};

use std::pin::Pin;
use std::sync::Arc;
use futures::Stream;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tonic::transport::Channel;

/// Speech-to-text transcription
///
/// NOTE: listen() IS blocking - agent waits for user speech.
/// This is inherent to voice INPUT (must wait for user to speak).
///
/// For self-speech filtering, use `FilteredSTT` instead.
///
/// # Example
///
/// ```rust,no_run
/// use dashflow_voice::VoiceSTT;
///
/// async fn example(stt: VoiceSTT) {
///     // Blocks until user finishes speaking
///     let result = stt.listen().await.unwrap();
///     println!("User said: {}", result.text);
/// }
/// ```
#[derive(Clone)]
pub struct VoiceSTT {
    client: Arc<Mutex<SttServiceClient<Channel>>>,
    language: String,
}

impl VoiceSTT {
    /// Create a new STT client
    pub fn new(channel: Channel, language: String) -> Self {
        Self {
            client: Arc::new(Mutex::new(SttServiceClient::new(channel))),
            language,
        }
    }

    /// Listen for speech and transcribe (BLOCKING - waits for user)
    ///
    /// This method blocks until the user finishes speaking.
    /// The transcription is returned when speech ends (detected by VAD).
    pub async fn listen(&self) -> Result<Transcription> {
        self.listen_with_config(default_streaming_config()).await
    }

    /// Listen with custom streaming configuration
    pub async fn listen_with_config(&self, config: SttStreamingConfig) -> Result<Transcription> {
        let mut client = self.client.lock().await;

        // Create audio input stream (from microphone)
        // In a real implementation, this would capture from audio device
        let audio_stream = create_audio_stream(&self.language, config);

        let response = client.stream_transcribe(audio_stream).await?;
        let mut stream = response.into_inner();

        let mut final_text = String::new();
        let mut total_confidence = 0.0f32;
        let mut num_chunks = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if chunk.is_final {
                final_text = chunk.text;
                total_confidence += chunk.confidence;
                num_chunks += 1;
                break;
            }
            // Accumulate partial results
            if !chunk.text.is_empty() {
                total_confidence += chunk.confidence;
                num_chunks += 1;
            }
        }

        let avg_confidence = if num_chunks > 0 {
            total_confidence / num_chunks as f32
        } else {
            0.0
        };

        Ok(Transcription {
            text: final_text,
            language: self.language.clone(),
            confidence: avg_confidence,
            duration_ms: 0, // Would be calculated from audio duration
        })
    }

    /// Stream transcription results (real-time partial results)
    ///
    /// Returns a stream of transcription chunks as speech is recognized.
    pub fn listen_stream(&self) -> TranscriptionStream {
        TranscriptionStream {
            client: self.client.clone(),
            language: self.language.clone(),
            started: false,
        }
    }
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Transcribed text
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Average confidence score (0-1)
    pub confidence: f32,
    /// Audio duration in milliseconds
    pub duration_ms: u64,
}

/// Stream of transcription chunks for real-time results
pub struct TranscriptionStream {
    client: Arc<Mutex<SttServiceClient<Channel>>>,
    language: String,
    started: bool,
}

impl TranscriptionStream {
    /// Start the transcription stream
    pub async fn start(&mut self) -> Result<Pin<Box<dyn Stream<Item = Result<TranscriptionChunk>> + Send>>> {
        if self.started {
            return Err(VoiceError::Internal("Stream already started".to_string()));
        }
        self.started = true;

        let mut client = self.client.lock().await;
        let audio_stream = create_audio_stream(&self.language, default_streaming_config());

        let response = client.stream_transcribe(audio_stream).await?;
        let stream = response.into_inner();

        let mapped = stream.map(|result| {
            result
                .map(|chunk| TranscriptionChunk {
                    text: chunk.text,
                    is_final: chunk.is_final,
                    speech_detected: chunk.speech_detected,
                    timestamp_ms: chunk.timestamp_ms as u64,
                    confidence: chunk.confidence,
                })
                .map_err(VoiceError::from)
        });

        Ok(Box::pin(mapped))
    }
}

/// Partial transcription chunk
#[derive(Debug, Clone)]
pub struct TranscriptionChunk {
    /// Text (partial or final)
    pub text: String,
    /// Whether this is a final result
    pub is_final: bool,
    /// Whether speech was detected
    pub speech_detected: bool,
    /// Timestamp from stream start
    pub timestamp_ms: u64,
    /// Confidence score
    pub confidence: f32,
}

/// Default streaming STT configuration
fn default_streaming_config() -> SttStreamingConfig {
    SttStreamingConfig {
        step_ms: 3000,              // Process every 3 seconds
        length_ms: 10000,           // 10 second sliding window
        keep_ms: 200,               // Keep 200ms context
        vad_threshold: 0.6,         // VAD threshold
        freq_threshold: 100.0,      // High-pass cutoff
        silence_threshold_ms: 1500, // Finalize after 1.5s silence
        use_vad_segmentation: Some(true),
        max_segment_duration_ms: 30000,
    }
}

/// Create audio input stream (placeholder)
///
/// In a real implementation, this would capture from the microphone.
fn create_audio_stream(
    language: &str,
    config: SttStreamingConfig,
) -> impl Stream<Item = AudioInput> {
    let language = language.to_string();

    // Create initial config message
    let init_msg = AudioInput {
        audio: vec![],
        sample_rate: 16000,
        end_of_stream: false,
        language: language.clone(),
        translate_to_english: false,
        config: Some(config),
    };

    // End message
    let end_msg = AudioInput {
        audio: vec![],
        sample_rate: 16000,
        end_of_stream: true,
        language,
        translate_to_english: false,
        config: None,
    };

    // In a real implementation, this would be an async stream from audio capture
    futures::stream::iter(vec![init_msg, end_msg])
}
