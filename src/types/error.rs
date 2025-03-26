//! Error types for the SDK

use thiserror::Error;
use std::result::Result as StdResult;

/// A specialized Result type for the SDK
pub type Result<T> = StdResult<T, Error>;

/// Errors that can occur in the SDK
#[derive(Error, Debug)]
pub enum Error {
    /// An error occurred with the agent
    #[error("Agent error: {0}")]
    Agent(String),

    /// An error occurred with the runner
    #[error("Runner error: {0}")]
    Runner(String),

    /// An error occurred with a tool
    #[error("Tool error: {0}")]
    Tool(String),

    /// An error occurred with a model
    #[error("Model error: {0}")]
    Model(String),

    /// An error occurred with a handoff
    #[error("Handoff error: {0}")]
    Handoff(String),

    /// An error occurred with a guardrail
    #[error("Guardrail error: {0}")]
    Guardrail(String),

    /// An error occurred with tracing
    #[error("Tracing error: {0}")]
    Tracing(String),

    /// An error with async-openai
    #[error("OpenAI API error: {0}")]
    OpenAI(#[from] async_openai::error::OpenAIError),

    /// An error occurred with serialization or deserialization
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// An I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Maximum number of turns exceeded
    #[error("Maximum number of turns exceeded: {0}")]
    MaxTurnsExceeded(usize),

    /// Input guardrail triggered
    #[error("Input guardrail triggered: {0}")]
    InputGuardrailTriggered(String),

    /// Output guardrail triggered
    #[error("Output guardrail triggered: {0}")]
    OutputGuardrailTriggered(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Handoff not found
    #[error("Handoff not found: {0}")]
    HandoffNotFound(String),

    /// Invalid model response
    #[error("Invalid model response: {0}")]
    InvalidModelResponse(String),

    /// Model behavior error
    #[error("Model behavior error: {0}")]
    ModelBehavior(String),

    /// User error
    #[error("User error: {0}")]
    User(String),

    /// Other error
    #[error("Other error: {0}")]
    Other(String),
}