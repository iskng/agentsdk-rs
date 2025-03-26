//! Model provider trait for LLM interactions

use async_trait::async_trait;
use crate::types::message::Message;
use crate::types::result::ModelResponse;
use crate::types::error::Result;
use crate::model::settings::ModelSettings;

/// Trait for model providers that can be used to execute LLM requests
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;
    
    /// Generate a response from the model
    async fn generate(
        &self,
        model: &str,
        messages: Vec<Message>,
        settings: &ModelSettings,
    ) -> Result<ModelResponse>;
    
    /// Generate a streaming response from the model
    async fn generate_stream(
        &self,
        _model: &str,
        _messages: Vec<Message>,
        _settings: &ModelSettings,
    ) -> Result<Box<dyn futures::Stream<Item = Result<ModelResponse>> + Send + 'static>> {
        // Default implementation returns an error
        Err(crate::types::error::Error::Model(format!(
            "The {} provider does not support streaming",
            self.name()
        )))
    }
    
    /// Clone this provider
    fn clone_box(&self) -> Box<dyn ModelProvider>;
}