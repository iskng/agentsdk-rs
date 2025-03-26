//! Model settings for controlling LLM behavior

use serde::{Serialize, Deserialize};

/// Tool choice settings for the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Model decides whether to use tools
    Auto,
    
    /// Model must use a tool
    Required,
    
    /// Model must not use a tool
    None,
    
    /// Model must use a specific tool
    #[serde(untagged)]
    Specific(String),
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Auto
    }
}

/// Response format for the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Type of the response format
    #[serde(rename = "type")]
    pub type_: String,
}

/// Settings for truncation behavior
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    /// Automatic truncation
    Auto,
    
    /// Never truncate
    Never,
}

/// Settings for a model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    
    /// Tool choice
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    
    /// Response format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    
    /// Seed for deterministic sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    
    /// Truncation settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    
    /// Additional settings specific to a model provider
    #[serde(flatten)]
    pub additional_settings: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            stream: None,
            max_tokens: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tool_choice: None,
            response_format: None,
            seed: None,
            truncation: None,
            additional_settings: std::collections::HashMap::new(),
        }
    }
}

impl ModelSettings {
    /// Create new model settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    
    /// Set the top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set whether to stream the response
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
    
    /// Set the maximum number of tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    
    /// Set the stop sequences
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }
    
    /// Set the presence penalty
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }
    
    /// Set the frequency penalty
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }
    
    /// Set the tool choice
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }
    
    /// Convenience method to set the function calling mode
    pub fn with_function_calling(mut self, mode: impl Into<String>) -> Self {
        let mode_str = mode.into();
        self.tool_choice = match mode_str.as_str() {
            "auto" | "Auto" => Some(ToolChoice::Auto),
            "required" | "Required" => Some(ToolChoice::Required),
            "none" | "None" => Some(ToolChoice::None),
            name => Some(ToolChoice::Specific(name.to_string())),
        };
        self
    }
    
    /// Set the response format
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }
    
    /// Set the seed for deterministic sampling
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Set the truncation settings
    pub fn with_truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }
    
    /// Add an additional setting
    pub fn with_additional_setting(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.additional_settings.insert(key.into(), value.into());
        self
    }
}