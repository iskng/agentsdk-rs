//! Mock model provider for testing

use async_trait::async_trait;
use serde_json::Value;

use crate::model::provider::ModelProvider;
use crate::model::settings::ModelSettings;
use crate::types::message::{ Message, Role };
use crate::types::result::{ ModelResponse, ModelChoice, Usage };
use crate::types::error::{ Result, Error };

/// A mock model provider for testing
pub struct MockModelProvider;

impl MockModelProvider {
    /// Create a new mock model provider
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockModelProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for MockModelProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        _model: &str,
        messages: Vec<Message>,
        settings: &ModelSettings
    ) -> Result<ModelResponse> {
        // Check if there's a mock response in the settings
        let mock_response = settings.additional_settings
            .get("mock_llm_response")
            .and_then(|v| v.as_str())
            .or_else(|| settings.additional_settings.get("mock_response").and_then(|v| v.as_str()))
            .ok_or_else(|| Error::Model("No mock response provided".to_string()))?;
        
        // Parse the mock response
        let mock_data: Value = serde_json::from_str(mock_response)
            .map_err(|e| Error::Model(format!("Failed to parse mock response: {}", e)))?;
        
        // Create a message from the mock data
        let content = mock_data.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("This is a mock response");
        
        let mut message = Message::assistant(content);
        
        // Add tool calls if present
        if let Some(tool_calls) = mock_data.get("tool_calls") {
            if let Some(tool_calls_array) = tool_calls.as_array() {
                let mut our_tool_calls = Vec::new();
                
                for tc in tool_calls_array {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_id").to_string();
                    let type_ = tc.get("type").and_then(|v| v.as_str()).unwrap_or("function").to_string();
                    let function = tc.get("function").unwrap_or(&Value::Null);
                    
                    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    let arguments = function.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}").to_string();
                    
                    our_tool_calls.push(crate::types::message::ToolCall {
                        id,
                        type_,
                        function: crate::types::message::FunctionCall {
                            name,
                            arguments,
                        },
                    });
                }
                
                message.tool_calls = Some(our_tool_calls);
            }
        }
        
        // Add function call if present (legacy format)
        if let Some(function_call) = mock_data.get("function_call") {
            let name = function_call.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let arguments = function_call.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}").to_string();
            
            message.function_call = Some(crate::types::message::FunctionCall {
                name,
                arguments,
            });
        }
        
        // Create a model choice
        let choice = ModelChoice {
            index: 0,
            message,
            finish_reason: Some("stop".to_string()),
        };
        
        // Create a model response
        let response = ModelResponse {
            id: "mock-response-id".to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: _model.to_string(),
            choices: vec![choice],
            usage: Usage {
                prompt_tokens: messages.len() * 10,
                completion_tokens: 20,
                total_tokens: messages.len() * 10 + 20,
            },
        };
        
        Ok(response)
    }

    async fn generate_stream(
        &self,
        _model: &str,
        _messages: Vec<Message>,
        _settings: &ModelSettings
    ) -> Result<Box<dyn futures::Stream<Item = Result<ModelResponse>> + Send + 'static>> {
        Err(Error::Model("Streaming is not implemented for mock provider".to_string()))
    }
    
    fn clone_box(&self) -> Box<dyn ModelProvider> {
        Box::new(self.clone())
    }
}

impl Clone for MockModelProvider {
    fn clone(&self) -> Self {
        Self
    }
}