//! Message types for communication with models

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Role of a message sender
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message
    System,
    
    /// User message
    User,
    
    /// Assistant message
    Assistant,
    
    /// Tool message (equivalent to function in some APIs)
    #[serde(alias = "function")]
    Tool,
}

/// A function call within a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function
    pub name: String,
    
    /// Arguments to the function as a JSON string
    pub arguments: String,
}

/// A tool call within a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// ID of the tool call
    pub id: String,
    
    /// Type of the tool call (always "function" for now)
    #[serde(rename = "type")]
    pub type_: String,
    
    /// Function call details
    pub function: FunctionCall,
}

/// Content of a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Text content
    Text(String),
    
    /// Content parts (for multimodal messages)
    Parts(Vec<ContentPart>),
}

/// A part of a multimodal message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text part
    #[serde(rename = "text")]
    Text { text: String },
    
    /// Image part
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// An image URL with optional detail
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL of the image
    pub url: String,
    
    /// Detail level of the image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    
    /// Content of the message
    #[serde(flatten)]
    pub content_container: MessageContentContainer,
    
    /// Tool calls in the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    
    /// Function call in the message (legacy format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    
    /// Name of the function or tool (for role = "function" or "tool")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    
    /// Additional properties
    #[serde(flatten)]
    pub additional_properties: HashMap<String, serde_json::Value>,
}

/// Container for message content to handle different formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContentContainer {
    /// Simple string content
    String {
        #[serde(rename = "content")]
        content: String,
    },
    
    /// Complex content (parts)
    Content {
        #[serde(rename = "content")]
        content: Vec<ContentPart>,
    },
    
    /// No content (for function/tool calls)
    Empty {
        #[serde(rename = "content")]
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
    },
}

impl Message {
    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content_container: MessageContentContainer::String {
                content: content.into(),
            },
            tool_calls: None,
            function_call: None,
            name: None,
            additional_properties: HashMap::new(),
        }
    }
    
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content_container: MessageContentContainer::String {
                content: content.into(),
            },
            tool_calls: None,
            function_call: None,
            name: None,
            additional_properties: HashMap::new(),
        }
    }
    
    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content_container: MessageContentContainer::String {
                content: content.into(),
            },
            tool_calls: None,
            function_call: None,
            name: None,
            additional_properties: HashMap::new(),
        }
    }
    
    /// Create a new tool message
    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        let mut additional_props = HashMap::new();
        let tool_call_id = format!("call_{}", uuid::Uuid::new_v4().to_string().replace("-", ""));
        additional_props.insert("tool_call_id".to_string(), serde_json::json!(tool_call_id));
        
        Self {
            role: Role::Tool,
            content_container: MessageContentContainer::String {
                content: content.into(),
            },
            tool_calls: None,
            function_call: None,
            name: Some(name.into()),
            additional_properties: additional_props,
        }
    }
    
    /// Get the content as a string if possible
    pub fn content_as_string(&self) -> Option<String> {
        match &self.content_container {
            MessageContentContainer::String { content } => Some(content.clone()),
            MessageContentContainer::Empty { content } => content.clone(),
            MessageContentContainer::Content { content } => {
                // Try to extract text from content parts
                let mut text_parts = Vec::new();
                for part in content {
                    if let ContentPart::Text { text } = part {
                        text_parts.push(text.clone());
                    }
                }
                if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join("\n"))
                }
            }
        }
    }
}

/// Convert a string to a Message
impl From<String> for Message {
    fn from(s: String) -> Self {
        Message::user(s)
    }
}

/// Convert a &str to a Message
impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Message::user(s)
    }
}