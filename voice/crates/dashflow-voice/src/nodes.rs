//! Graph node helpers for dashflow integration
//!
//! These helpers create nodes that can be added to dashflow StateGraphs.
//! They abstract the voice client setup and provide simple interfaces.

use crate::client::VoiceClient;
use crate::filtered::FilterConfig;
use crate::wake_word::{WakeWordClient, WakeWordConfig, WakeWordDetection};
use crate::error::Result;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Shared voice client instance (lazy initialized)
static VOICE_CLIENT: OnceCell<Arc<VoiceClient>> = OnceCell::const_new();

/// Get or create a shared voice client
async fn get_voice_client(endpoint: &str) -> Result<Arc<VoiceClient>> {
    // Try to get existing client
    if let Some(client) = VOICE_CLIENT.get() {
        return Ok(client.clone());
    }

    // Create new client
    let client = Arc::new(VoiceClient::connect(endpoint).await?);

    // Try to set it (race condition safe)
    let _ = VOICE_CLIENT.set(client.clone());

    Ok(VOICE_CLIENT.get().unwrap().clone())
}

/// Voice input node result
#[derive(Debug, Clone)]
pub struct VoiceInput {
    /// Transcribed user text
    pub text: String,
    /// Speaker ID (if diarization enabled)
    pub speaker_id: String,
    /// Confidence score
    pub confidence: f32,
}

/// Voice output node result
#[derive(Debug, Clone)]
pub struct VoiceOutput {
    /// Request ID for tracking
    pub request_id: String,
    /// Whether speech was queued successfully
    pub success: bool,
}

/// Create a voice input node that listens with filtering
///
/// This node:
/// 1. Connects to voice service (if not already connected)
/// 2. Listens for user speech with self-speech filtering
/// 3. Returns transcribed text
///
/// # Usage (pseudo-dashflow API)
///
/// ```rust,ignore
/// use dashflow_voice::filtered_input_node;
///
/// let input_fn = filtered_input_node("http://localhost:50051");
/// graph.add_node("listen", |state| async move {
///     let input = input_fn().await?;
///     state.set("user_input", input.text);
///     Ok(state)
/// });
/// ```
pub fn filtered_input_node(
    endpoint: &str,
) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<VoiceInput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();

    move || {
        let endpoint = endpoint.clone();
        Box::pin(async move {
            let client = get_voice_client(&endpoint).await?;
            let result = client.listen_filtered().await?;

            Ok(VoiceInput {
                text: result.text,
                speaker_id: result.speaker_id,
                confidence: result.confidence,
            })
        })
    }
}

/// Create a voice input node (basic STT, no filtering)
///
/// Use this when you don't need self-speech filtering.
///
/// # Usage
///
/// ```rust,ignore
/// let input_fn = voice_input_node("http://localhost:50051");
/// let result = input_fn().await?;
/// println!("User said: {}", result.text);
/// ```
pub fn voice_input_node(
    endpoint: &str,
) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<VoiceInput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();

    move || {
        let endpoint = endpoint.clone();
        Box::pin(async move {
            let client = get_voice_client(&endpoint).await?;
            let result = client.listen().await?;

            Ok(VoiceInput {
                text: result.text,
                speaker_id: String::new(),
                confidence: result.confidence,
            })
        })
    }
}

/// Create a voice output node that speaks text
///
/// This node:
/// 1. Connects to voice service (if not already connected)
/// 2. Queues text for speech (non-blocking)
/// 3. Returns immediately (does NOT wait for speech to complete)
///
/// # Usage
///
/// ```rust,ignore
/// use dashflow_voice::voice_output_node;
///
/// let speak_fn = voice_output_node("http://localhost:50051");
/// graph.add_node("speak", |state| async move {
///     let response = state.get::<String>("response").unwrap();
///     speak_fn(&response).await?;
///     Ok(state)
/// });
/// ```
pub fn voice_output_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<VoiceOutput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();

    move |text: &str| {
        let endpoint = endpoint.clone();
        let text = text.to_string();
        Box::pin(async move {
            let client = get_voice_client(&endpoint).await?;
            let handle = client.speak(&text).await?;

            Ok(VoiceOutput {
                request_id: handle.request_id.clone(),
                success: true,
            })
        })
    }
}

/// Create a voice output node with language selection
pub fn voice_output_node_lang(
    endpoint: &str,
) -> impl Fn(&str, &str) -> Pin<Box<dyn Future<Output = Result<VoiceOutput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();

    move |text: &str, language: &str| {
        let endpoint = endpoint.clone();
        let text = text.to_string();
        let language = language.to_string();
        Box::pin(async move {
            let client = get_voice_client(&endpoint).await?;
            let handle = client.speak_lang(&text, &language).await?;

            Ok(VoiceOutput {
                request_id: handle.request_id.clone(),
                success: true,
            })
        })
    }
}

/// Create a voice interrupt node (high-priority, stops current speech)
pub fn voice_interrupt_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<VoiceOutput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();

    move |text: &str| {
        let endpoint = endpoint.clone();
        let text = text.to_string();
        Box::pin(async move {
            let client = get_voice_client(&endpoint).await?;
            let handle = client.interrupt(&text).await?;

            Ok(VoiceOutput {
                request_id: handle.request_id.clone(),
                success: true,
            })
        })
    }
}

/// Combined voice conversation node
///
/// Provides a complete conversation turn:
/// 1. Listen for user speech (with filtering)
/// 2. Return user text for processing
/// 3. Caller processes and calls speak
///
/// # Usage
///
/// ```rust,ignore
/// let conv = VoiceConversation::new("http://localhost:50051").await?;
///
/// loop {
///     // Listen (blocking)
///     let user_text = conv.listen().await?;
///
///     // Process with LLM
///     let response = llm.chat(&user_text).await?;
///
///     // Speak (non-blocking)
///     conv.speak(&response).await?;
/// }
/// ```
#[derive(Clone)]
pub struct VoiceConversation {
    client: Arc<VoiceClient>,
    filter_config: FilterConfig,
}

impl VoiceConversation {
    /// Create a new conversation handler
    pub async fn new(endpoint: &str) -> Result<Self> {
        let client = get_voice_client(endpoint).await?;
        Ok(Self {
            client,
            filter_config: FilterConfig::default(),
        })
    }

    /// Create with custom filter configuration
    pub async fn with_config(endpoint: &str, filter_config: FilterConfig) -> Result<Self> {
        let client = get_voice_client(endpoint).await?;
        Ok(Self {
            client,
            filter_config,
        })
    }

    /// Listen for user speech (blocks until speech ends)
    pub async fn listen(&self) -> Result<String> {
        let result = self.client
            .listen_filtered_with_config(self.filter_config.clone())
            .await?;
        Ok(result.text)
    }

    /// Speak text (non-blocking, returns immediately)
    pub async fn speak(&self, text: &str) -> Result<()> {
        self.client.speak(text).await?;
        Ok(())
    }

    /// Interrupt current speech and speak new text
    pub async fn interrupt(&self, text: &str) -> Result<()> {
        self.client.interrupt(text).await?;
        Ok(())
    }

    /// Stop all speech
    pub async fn stop(&self) -> Result<()> {
        self.client.stop_speech().await
    }

    /// Check if ready
    pub async fn is_ready(&self) -> Result<bool> {
        self.client.is_ready().await
    }
}

// =============================================================================
// Wake Word Nodes
// =============================================================================

/// Shared wake word client instance (lazy initialized)
static WAKE_WORD_CLIENT: OnceCell<Arc<WakeWordClient>> = OnceCell::const_new();

/// Get or create a shared wake word client
async fn get_wake_word_client(endpoint: &str) -> Result<Arc<WakeWordClient>> {
    if let Some(client) = WAKE_WORD_CLIENT.get() {
        return Ok(client.clone());
    }

    let client = Arc::new(WakeWordClient::connect(endpoint).await?);
    let _ = WAKE_WORD_CLIENT.set(client.clone());
    Ok(WAKE_WORD_CLIENT.get().unwrap().clone())
}

/// Wake word event output
#[derive(Debug, Clone)]
pub struct WakeWordEvent {
    /// Name of detected wake word
    pub wake_word: String,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
    /// Timestamp in ms since listening started
    pub timestamp_ms: i64,
    /// Latency from audio end to callback
    pub latency_ms: i64,
}

impl From<WakeWordDetection> for WakeWordEvent {
    fn from(detection: WakeWordDetection) -> Self {
        Self {
            wake_word: detection.wake_word,
            confidence: detection.confidence,
            timestamp_ms: detection.timestamp_ms,
            latency_ms: detection.latency_ms,
        }
    }
}

/// Create a wake word detection node
///
/// This node:
/// 1. Connects to wake word service (if not already connected)
/// 2. Waits for a single wake word detection
/// 3. Returns the detection event
///
/// # Usage (pseudo-dashflow API)
///
/// ```rust,ignore
/// use dashflow_voice::wake_word_node;
///
/// let wait_for_wake_word = wake_word_node("http://localhost:50051", None);
/// graph.add_node("wait_for_activation", |state| async move {
///     let event = wait_for_wake_word().await?;
///     println!("Activated by: {}", event.wake_word);
///     // Transition to listening state
///     Ok(state)
/// });
/// ```
pub fn wake_word_node(
    endpoint: &str,
    config: Option<WakeWordConfig>,
) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<WakeWordEvent>> + Send>> + Clone {
    let endpoint = endpoint.to_string();
    let config = config.unwrap_or_default();

    move || {
        let endpoint = endpoint.clone();
        let config = config.clone();
        Box::pin(async move {
            let client = get_wake_word_client(&endpoint).await?;
            let mut stream = client.start_listening(config).await?;

            // Wait for a single detection
            match stream.next().await {
                Some(Ok(detection)) => Ok(WakeWordEvent::from(detection)),
                Some(Err(e)) => Err(e),
                None => Err(crate::error::VoiceError::StreamEnded),
            }
        })
    }
}

/// Create a wake word node for specific wake words
///
/// # Example
///
/// ```rust,ignore
/// let wait_for_hey_voice = wake_word_node_for(
///     "http://localhost:50051",
///     vec!["hey_voice".to_string()]
/// );
/// ```
pub fn wake_word_node_for(
    endpoint: &str,
    wake_words: Vec<String>,
) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<WakeWordEvent>> + Send>> + Clone {
    wake_word_node(endpoint, Some(WakeWordConfig::for_wake_words(wake_words)))
}

/// Voice agent that activates on wake word
///
/// Combines wake word detection with voice conversation in a single abstraction.
///
/// # Usage
///
/// ```rust,ignore
/// use dashflow_voice::WakeWordAgent;
///
/// let agent = WakeWordAgent::new("http://localhost:50051").await?;
///
/// loop {
///     // Wait for wake word (blocks)
///     let event = agent.wait_for_activation().await?;
///     println!("Wake word '{}' detected!", event.wake_word);
///
///     // Play acknowledgment (non-blocking)
///     agent.speak("I'm listening").await?;
///
///     // Listen for user command (blocks until speech ends)
///     let command = agent.listen().await?;
///     println!("User said: {}", command);
///
///     // Process and respond
///     let response = process_command(&command);
///     agent.speak(&response).await?;
/// }
/// ```
#[derive(Clone)]
pub struct WakeWordAgent {
    voice_client: Arc<VoiceClient>,
    wake_word_client: Arc<WakeWordClient>,
    wake_word_config: WakeWordConfig,
    filter_config: FilterConfig,
}

impl WakeWordAgent {
    /// Create a new wake word agent
    pub async fn new(endpoint: &str) -> Result<Self> {
        let voice_client = Arc::new(VoiceClient::connect(endpoint).await?);
        let wake_word_client = Arc::new(WakeWordClient::connect(endpoint).await?);

        Ok(Self {
            voice_client,
            wake_word_client,
            wake_word_config: WakeWordConfig::default(),
            filter_config: FilterConfig::default(),
        })
    }

    /// Create with custom configurations
    pub async fn with_config(
        endpoint: &str,
        wake_word_config: WakeWordConfig,
        filter_config: FilterConfig,
    ) -> Result<Self> {
        let voice_client = Arc::new(VoiceClient::connect(endpoint).await?);
        let wake_word_client = Arc::new(WakeWordClient::connect(endpoint).await?);

        Ok(Self {
            voice_client,
            wake_word_client,
            wake_word_config,
            filter_config,
        })
    }

    /// Wait for wake word activation (blocks)
    pub async fn wait_for_activation(&self) -> Result<WakeWordEvent> {
        let mut stream = self.wake_word_client
            .start_listening(self.wake_word_config.clone())
            .await?;

        match stream.next().await {
            Some(Ok(detection)) => {
                // Stop listening after detection
                let _ = self.wake_word_client.stop_listening().await;
                Ok(WakeWordEvent::from(detection))
            }
            Some(Err(e)) => Err(e),
            None => Err(crate::error::VoiceError::StreamEnded),
        }
    }

    /// Listen for user speech with filtering (blocks until speech ends)
    pub async fn listen(&self) -> Result<String> {
        let result = self.voice_client
            .listen_filtered_with_config(self.filter_config.clone())
            .await?;
        Ok(result.text)
    }

    /// Speak text (non-blocking, returns immediately)
    pub async fn speak(&self, text: &str) -> Result<()> {
        self.voice_client.speak(text).await?;
        Ok(())
    }

    /// Interrupt current speech and speak new text
    pub async fn interrupt(&self, text: &str) -> Result<()> {
        self.voice_client.interrupt(text).await?;
        Ok(())
    }

    /// Stop all speech
    pub async fn stop(&self) -> Result<()> {
        self.voice_client.stop_speech().await
    }

    /// Check if service is ready
    pub async fn is_ready(&self) -> Result<bool> {
        self.voice_client.is_ready().await
    }

    /// Get wake word status
    pub async fn wake_word_status(&self) -> Result<crate::wake_word::WakeWordStatus> {
        self.wake_word_client.get_status().await
    }

    /// List available wake word models
    pub async fn list_wake_word_models(&self) -> Result<Vec<crate::wake_word::WakeWordModel>> {
        self.wake_word_client.list_models().await
    }
}

// =============================================================================
// Summarization Nodes
// =============================================================================

use crate::summarization::{SummarizationClient, SummarizationConfig, Summary};

/// Shared summarization client instance (lazy initialized)
static SUMMARIZATION_CLIENT: OnceCell<Arc<SummarizationClient>> = OnceCell::const_new();

/// Get or create a shared summarization client
async fn get_summarization_client(endpoint: &str) -> Result<Arc<SummarizationClient>> {
    if let Some(client) = SUMMARIZATION_CLIENT.get() {
        return Ok(client.clone());
    }

    let client = Arc::new(SummarizationClient::connect(endpoint).await?);
    let _ = SUMMARIZATION_CLIENT.set(client.clone());
    Ok(SUMMARIZATION_CLIENT.get().unwrap().clone())
}

/// Summary output from summarization node
#[derive(Debug, Clone)]
pub struct SummaryOutput {
    /// Summary text
    pub text: String,
    /// Processing latency in milliseconds
    pub latency_ms: i64,
    /// Number of input tokens processed
    pub input_tokens: i32,
    /// Number of output tokens generated
    pub output_tokens: i32,
}

impl From<Summary> for SummaryOutput {
    fn from(summary: Summary) -> Self {
        Self {
            text: summary.text,
            latency_ms: summary.latency_ms,
            input_tokens: summary.input_tokens,
            output_tokens: summary.output_tokens,
        }
    }
}

/// Create a text summarization node
///
/// This node:
/// 1. Connects to summarization service (if not already connected)
/// 2. Summarizes the provided text using Qwen3-8B
/// 3. Returns the summary
///
/// # Usage (pseudo-dashflow API)
///
/// ```rust,ignore
/// use dashflow_voice::summarize_text_node;
///
/// let summarize_fn = summarize_text_node("http://localhost:50051", None);
/// graph.add_node("summarize", |state| async move {
///     let transcript = state.get::<String>("transcript").unwrap();
///     let summary = summarize_fn(&transcript).await?;
///     state.set("summary", summary.text);
///     Ok(state)
/// });
/// ```
pub fn summarize_text_node(
    endpoint: &str,
    config: Option<SummarizationConfig>,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<SummaryOutput>> + Send>> + Clone {
    let endpoint = endpoint.to_string();
    let config = config.unwrap_or_default();

    move |text: &str| {
        let endpoint = endpoint.clone();
        let config = config.clone();
        let text = text.to_string();
        Box::pin(async move {
            let client = get_summarization_client(&endpoint).await?;
            let summary = client.summarize(&text, config).await?;
            Ok(SummaryOutput::from(summary))
        })
    }
}

/// Create a brief summarization node (1 sentence)
///
/// Convenience wrapper around summarize_text_node with brief mode.
pub fn summarize_brief_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<SummaryOutput>> + Send>> + Clone {
    summarize_text_node(endpoint, Some(SummarizationConfig::brief()))
}

/// Create a detailed summarization node (bullet points)
///
/// Convenience wrapper around summarize_text_node with detailed mode.
pub fn summarize_detailed_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<SummaryOutput>> + Send>> + Clone {
    summarize_text_node(endpoint, Some(SummarizationConfig::detailed()))
}

/// Create an action items extraction node
///
/// Convenience wrapper around summarize_text_node with action items mode.
pub fn summarize_action_items_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<SummaryOutput>> + Send>> + Clone {
    summarize_text_node(endpoint, Some(SummarizationConfig::action_items()))
}

/// Holder for live summarization stream functionality
///
/// Use `SummarizationClient::live_summarize` directly for streaming summarization.
/// This node-based interface is provided for simpler text summarization use cases.
pub fn live_summarize_node(
    endpoint: &str,
) -> impl Fn(&str) -> Pin<Box<dyn Future<Output = Result<SummaryOutput>> + Send>> + Clone {
    // For now, live_summarize_node just does text summarization
    // Full streaming support requires more complex async handling
    summarize_text_node(endpoint, None)
}

/// Summarization agent that combines STT with summarization
///
/// Listens for speech and provides periodic summaries.
///
/// # Usage
///
/// ```rust,ignore
/// use dashflow_voice::SummarizationAgent;
///
/// let agent = SummarizationAgent::new("http://localhost:50051").await?;
///
/// // Summarize live speech
/// agent.start_listening().await?;
///
/// // Get summaries as they're generated
/// while let Some(summary) = agent.next_summary().await? {
///     println!("Summary: {}", summary.text);
///     agent.speak(&format!("Summary: {}", summary.text)).await?;
/// }
/// ```
#[derive(Clone)]
pub struct SummarizationAgent {
    voice_client: Arc<VoiceClient>,
    summarization_client: Arc<SummarizationClient>,
    config: SummarizationConfig,
}

impl SummarizationAgent {
    /// Create a new summarization agent
    pub async fn new(endpoint: &str) -> Result<Self> {
        let voice_client = Arc::new(VoiceClient::connect(endpoint).await?);
        let summarization_client = Arc::new(SummarizationClient::connect(endpoint).await?);

        Ok(Self {
            voice_client,
            summarization_client,
            config: SummarizationConfig::default(),
        })
    }

    /// Create with custom summarization configuration
    pub async fn with_config(endpoint: &str, config: SummarizationConfig) -> Result<Self> {
        let voice_client = Arc::new(VoiceClient::connect(endpoint).await?);
        let summarization_client = Arc::new(SummarizationClient::connect(endpoint).await?);

        Ok(Self {
            voice_client,
            summarization_client,
            config,
        })
    }

    /// Summarize text
    pub async fn summarize(&self, text: &str) -> Result<SummaryOutput> {
        let summary = self.summarization_client.summarize(text, self.config.clone()).await?;
        Ok(SummaryOutput::from(summary))
    }

    /// Summarize text briefly (1 sentence)
    pub async fn summarize_brief(&self, text: &str) -> Result<SummaryOutput> {
        let summary = self.summarization_client.summarize_brief(text).await?;
        Ok(SummaryOutput::from(summary))
    }

    /// Extract action items from text
    pub async fn extract_action_items(&self, text: &str) -> Result<SummaryOutput> {
        let summary = self.summarization_client.summarize_action_items(text).await?;
        Ok(SummaryOutput::from(summary))
    }

    /// Speak text (non-blocking)
    pub async fn speak(&self, text: &str) -> Result<()> {
        self.voice_client.speak(text).await?;
        Ok(())
    }

    /// Check if summarization service is ready
    pub async fn is_ready(&self) -> Result<bool> {
        self.summarization_client.is_ready().await
    }

    /// Get summarization service status
    pub async fn get_status(&self) -> Result<crate::summarization::SummarizationStatus> {
        self.summarization_client.get_status().await
    }
}
