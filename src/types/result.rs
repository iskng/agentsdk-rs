//! Result types for agent runs

use serde::{Serialize, Deserialize};
use std::pin::Pin;
use futures::{Stream, Future};
use crate::types::message::Message;

/// Usage statistics for a model call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of prompt tokens used
    #[serde(default)]
    pub prompt_tokens: usize,
    
    /// Number of completion tokens used
    #[serde(default)]
    pub completion_tokens: usize,
    
    /// Total number of tokens used
    #[serde(default)]
    pub total_tokens: usize,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

impl Usage {
    /// Combine with another usage
    pub fn combine(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// A raw response from a model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// ID of the response
    pub id: String,
    
    /// Type of the object
    pub object: String,
    
    /// Created timestamp
    pub created: u64,
    
    /// Model used
    pub model: String,
    
    /// Choices in the response
    pub choices: Vec<ModelChoice>,
    
    /// Usage statistics
    #[serde(default)]
    pub usage: Usage,
}

/// A choice in a model response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    /// Index of the choice
    pub index: usize,
    
    /// Message in the choice
    pub message: Message,
    
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// An item in a run
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Item {
    /// A message
    #[serde(rename = "message")]
    Message {
        /// ID of the message
        id: String,
        
        /// Role of the sender
        role: String,
        
        /// Content of the message
        content: String,
        
        /// Tool calls in the message
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<crate::types::message::ToolCall>>,
        
        /// Function call in the message (legacy format)
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<crate::types::message::FunctionCall>,
        
        /// Status of the message
        status: String,
    },
    
    /// A tool call
    #[serde(rename = "tool_call")]
    ToolCall {
        /// ID of the tool call
        id: String,
        
        /// Name of the tool
        tool_name: String,
        
        /// Arguments to the tool
        args: String,
    },
    
    /// A tool result
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the tool call
        call_id: String,
        
        /// Name of the tool
        tool_name: String,
        
        /// Result of the tool call
        result: String,
    },
    
    /// A handoff
    #[serde(rename = "handoff")]
    Handoff {
        /// ID of the handoff
        id: String,
        
        /// Name of the handoff
        handoff_name: String,
        
        /// Arguments to the handoff
        args: Option<String>,
    },
}

/// Helper for item operations
pub struct ItemHelpers;

impl ItemHelpers {
    /// Get the text content of a message item
    pub fn text_message_output(item: &Item) -> Option<String> {
        match item {
            Item::Message { content, .. } => Some(content.clone()),
            _ => None,
        }
    }
}

/// Result of running an agent
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Final output text
    pub final_output: String,
    
    /// Optional structured output
    pub structured_output: Option<serde_json::Value>,
    
    /// New items created during the run
    pub new_items: Vec<Item>,
    
    /// Last agent that produced output
    pub last_agent: String,
    
    /// Raw responses from the model
    pub raw_responses: Vec<ModelResponse>,
    
    /// Usage statistics
    pub usage: Usage,
}

impl RunResult {
    /// Create a new run result
    pub fn new(
        final_output: String,
        structured_output: Option<serde_json::Value>,
        new_items: Vec<Item>,
        last_agent: String,
        raw_responses: Vec<ModelResponse>,
        usage: Usage,
    ) -> Self {
        Self {
            final_output,
            structured_output,
            new_items,
            last_agent,
            raw_responses,
            usage,
        }
    }
    
    /// Convert the result to an input list for the next run
    pub fn to_input_list(&self) -> Vec<Message> {
        // Convert items to messages
        self.new_items
            .iter()
            .filter_map(|item| match item {
                Item::Message { role, content, tool_calls, function_call, .. } => {
                    let role_enum = match role.as_str() {
                        "system" => crate::types::message::Role::System,
                        "user" => crate::types::message::Role::User,
                        "assistant" => crate::types::message::Role::Assistant,
                        "tool" | "function" => crate::types::message::Role::Tool,
                        _ => return None,
                    };
                    
                    let message = Message {
                        role: role_enum,
                        content_container: crate::types::message::MessageContentContainer::String {
                            content: content.clone(),
                        },
                        tool_calls: tool_calls.clone(),
                        function_call: function_call.clone(),
                        name: None,
                        additional_properties: Default::default(),
                    };
                    
                    Some(message)
                },
                _ => None,
            })
            .collect()
    }
    
    /// Get the structured output as a specific type
    pub fn final_output_as<T: for<'de> Deserialize<'de>>(&self) -> Result<T, crate::types::error::Error> {
        match &self.structured_output {
            Some(value) => {
                serde_json::from_value(value.clone())
                    .map_err(|e| crate::types::error::Error::Serialization(e))
            },
            None => {
                Err(crate::types::error::Error::InvalidModelResponse(
                    "No structured output available".to_string()
                ))
            }
        }
    }
}

/// A streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// Type of the event
    pub event_type: String,
    
    /// Data of the event
    pub data: serde_json::Value,
    
    /// Timestamp of the event
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Result of a streamed run
pub struct StreamedRunResult {
    /// Stream of events
    pub events: Pin<Box<dyn Stream<Item = StreamEvent> + Send>>,
    
    /// Future that resolves to the final result
    pub result: Pin<Box<dyn Future<Output = Result<RunResult, crate::types::error::Error>> + Send>>,
}