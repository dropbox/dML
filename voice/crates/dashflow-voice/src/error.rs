//! Error types for dashflow-voice

use thiserror::Error;

/// Voice operation errors
#[derive(Error, Debug)]
pub enum VoiceError {
    /// Connection to voice service failed
    #[error("Failed to connect to voice service: {0}")]
    ConnectionFailed(String),

    /// gRPC transport error
    #[error("gRPC transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// gRPC status error
    #[error("gRPC error: {0}")]
    GrpcStatus(#[from] tonic::Status),

    /// Voice service not ready
    #[error("Voice service not ready")]
    NotReady,

    /// Timeout waiting for response
    #[error("Operation timed out after {0}ms")]
    Timeout(u64),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Audio processing error
    #[error("Audio processing error: {0}")]
    AudioError(String),

    /// Speaker not found
    #[error("Speaker not found: {0}")]
    SpeakerNotFound(String),

    /// Stream ended unexpectedly
    #[error("Stream ended unexpectedly")]
    StreamEnded,

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for voice operations
pub type Result<T> = std::result::Result<T, VoiceError>;
