//! Configuration for dashflow-voice client

use serde::{Deserialize, Serialize};

/// Voice client configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// gRPC endpoint (e.g., "http://localhost:50051")
    pub endpoint: String,

    /// Default language for TTS/STT
    pub default_language: String,

    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,

    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,

    /// Enable self-speech filtering by default
    pub enable_filtering: bool,

    /// Default TTS priority (0-100, higher = more urgent)
    pub default_priority: i32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:50051".to_string(),
            default_language: "en".to_string(),
            connect_timeout_ms: 5000,
            request_timeout_ms: 30000,
            enable_filtering: true,
            default_priority: 10,
        }
    }
}

impl VoiceConfig {
    /// Create a new config with the given endpoint
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            ..Default::default()
        }
    }

    /// Builder: set endpoint
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = endpoint.to_string();
        self
    }

    /// Builder: set default language
    pub fn with_language(mut self, language: &str) -> Self {
        self.default_language = language.to_string();
        self
    }

    /// Builder: enable/disable filtering
    pub fn with_filtering(mut self, enable: bool) -> Self {
        self.enable_filtering = enable;
        self
    }

    /// Builder: set connection timeout
    pub fn with_connect_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.connect_timeout_ms = timeout_ms;
        self
    }

    /// Builder: set request timeout
    pub fn with_request_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.request_timeout_ms = timeout_ms;
        self
    }
}
