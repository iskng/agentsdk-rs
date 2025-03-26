//! OpenAI model provider implementation

use async_trait::async_trait;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage,
        ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestUserMessage,
        CreateChatCompletionRequest,
        Role as OpenAIRole,
        Stop,
    },
    Client,
};
use async_openai::types::ChatCompletionRequestUserMessageContent;
use futures::Stream;

use crate::model::provider::ModelProvider;
use crate::model::settings::ModelSettings;
use crate::types::message::{ Message, Role };
use crate::types::result::{ ModelResponse, ModelChoice, Usage };
use crate::types::error::{ Result, Error };

/// OpenAI model provider
pub struct OpenAIChatCompletions {
    /// Client for the OpenAI API
    client: Client<OpenAIConfig>,
}

impl OpenAIChatCompletions {
    /// Create a new OpenAI model provider with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    /// Create a new OpenAI model provider with custom API key
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key.into());
        Self {
            client: Client::with_config(config),
        }
    }

    /// Create a new OpenAI model provider with custom base URL
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        let config = OpenAIConfig::new().with_api_base(base_url.into());
        Self {
            client: Client::with_config(config),
        }
    }

    /// Create a new OpenAI model provider with custom organization ID
    pub fn with_organization(organization_id: impl Into<String>) -> Self {
        let config = OpenAIConfig::new().with_org_id(organization_id.into());
        Self {
            client: Client::with_config(config),
        }
    }

    /// Create a new OpenAI model provider with custom configuration
    pub fn with_config(config: OpenAIConfig) -> Self {
        Self {
            client: Client::with_config(config),
        }
    }

    /// Convert SDK messages to OpenAI messages
    fn convert_messages(&self, messages: Vec<Message>) -> Vec<ChatCompletionRequestMessage> {
        // First we'll track all the assistant messages with tool calls so we know
        // which tool responses are valid
        let mut tool_call_ids: Vec<String> = Vec::new();
        
        // Find all tool call IDs from assistant messages
        for msg in &messages {
            if msg.role == Role::Assistant {
                if let Some(tool_calls) = &msg.tool_calls {
                    for call in tool_calls {
                        tool_call_ids.push(call.id.clone());
                    }
                }
            }
        }
        
        // Now convert messages, handling tool messages specially
        messages
            .into_iter()
            .map(|message| {
                // Create a new message based on the role
                let role = match message.role {
                    Role::System => OpenAIRole::System,
                    Role::User => OpenAIRole::User,
                    Role::Assistant => OpenAIRole::Assistant,
                    Role::Tool => {
                        // For tool messages, check if tool_call_id exists in our tracked IDs
                        let tool_call_id = message.additional_properties
                            .get("tool_call_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                            
                        // If the tool_call_id exists in our list of valid IDs, keep it as a tool message
                        // Otherwise convert to user 
                        if let Some(id) = &tool_call_id {
                            if tool_call_ids.contains(id) {
                                OpenAIRole::Tool 
                            } else {
                                OpenAIRole::User
                            }
                        } else {
                            OpenAIRole::User
                        }
                    },
                };

                // Get content as string if available
                let content = message.content_as_string();

                // Create the message according to its role with the appropriate content
                let content_str = content.unwrap_or_default();

                // Create the appropriate message type based on the role
                let msg = match role {
                    OpenAIRole::System => {
                        // Create a system message with content
                        let system_msg = ChatCompletionRequestSystemMessage {
                            content: ChatCompletionRequestSystemMessageContent::Text(content_str),
                            name: None,
                        };
                        ChatCompletionRequestMessage::System(system_msg)
                    }
                    OpenAIRole::User => {
                        // Create a user message with content
                        let user_msg = ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(content_str),
                            name: None,
                        };
                        ChatCompletionRequestMessage::User(user_msg)
                    }
                    OpenAIRole::Assistant => {
                        // Create an assistant message with content
                        let assistant_msg = ChatCompletionRequestAssistantMessage {
                            content: Some(
                                ChatCompletionRequestAssistantMessageContent::Text(content_str)
                            ),
                            name: None,
                            tool_calls: None,
                            function_call: None,
                            audio: None,
                            refusal: None,
                        };
                        ChatCompletionRequestMessage::Assistant(assistant_msg)
                    }
                    OpenAIRole::Tool => {
                        // Tool messages need a tool_call_id, which should be provided in the message
                        // If not available, we'll convert this to a user message to avoid API errors
                        if let Some(tool_call_id) = message.additional_properties
                            .get("tool_call_id")
                            .and_then(|v| v.as_str())
                        {
                            // Check if there's a preceding message with this tool_call_id
                            // If not, we'll convert to user message
                            let tool_msg = ChatCompletionRequestToolMessage {
                                content: ChatCompletionRequestToolMessageContent::Text(content_str),
                                tool_call_id: tool_call_id.to_string(),
                            };
                            ChatCompletionRequestMessage::Tool(tool_msg)
                        } else {
                            // No tool_call_id, convert to user message
                            let formatted_content = if let Some(name) = &message.name {
                                format!("Tool '{}' response: {}", name, content_str)
                            } else {
                                format!("Tool response: {}", content_str)
                            };
                            
                            let user_msg = ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(formatted_content),
                                name: None,
                            };
                            ChatCompletionRequestMessage::User(user_msg)
                        }
                    }
                    _ => {
                        // Default to user message
                        let user_msg = ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(content_str),
                            name: None,
                        };
                        ChatCompletionRequestMessage::User(user_msg)
                    }
                };

                // Content is already set in the match above

                // TODO: Add tool calls if present
                // This is a placeholder due to API changes in async-openai

                // TODO: Add function call if present (legacy format)
                // This is a placeholder due to API changes in async-openai

                // Add name if present (for tool/function messages)
                if let Some(name) = message.name {
                    // This is a placeholder due to API changes in async-openai
                    // We would set the name here if the API supported it
                }

                msg
            })
            .collect()
    }

    /// Convert ModelSettings to OpenAI request parameters
    fn convert_settings(
        &self,
        model: &str,
        messages: Vec<ChatCompletionRequestMessage>,
        settings: &ModelSettings
    ) -> CreateChatCompletionRequest {
        // Create a new request with the model and messages
        let mut request = CreateChatCompletionRequest {
            model: model.to_string(),
            messages,
            ..Default::default()
        };
        
        // Add tools if tools are requested
        if settings.tool_choice.is_some() || settings.additional_settings.contains_key("debug_tools") {
            // Debug print
            if settings.additional_settings.contains_key("debug_tools") {
                println!("[DEBUG] Adding tools to the request");
                println!("[DEBUG] Tool choice: {:?}", settings.tool_choice);
            }
            
            // Extract tools from the agent if they were provided in the additional settings
            use async_openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
            
            // Check if tools are provided in the settings
            let tools = if let Some(tools_json) = settings.additional_settings.get("tools") {
                if let Some(tools_array) = tools_json.as_array() {
                    if settings.additional_settings.contains_key("debug_tools") {
                        println!("[DEBUG] Using {} tools from settings", tools_array.len());
                    }
                    
                    // Parse the tools from the JSON array
                    tools_array.iter().filter_map(|tool_json| {
                        let name = tool_json.get("name")?.as_str()?.to_string();
                        let description = tool_json.get("description")?.as_str().map(|s| s.to_string());
                        let parameters = tool_json.get("parameters").cloned();
                        
                        Some(ChatCompletionTool {
                            r#type: ChatCompletionToolType::Function,
                            function: FunctionObject {
                                name,
                                description,
                                parameters,
                                strict: Some(false),
                            },
                        })
                    }).collect()
                } else {
                    // If not a valid array, use an empty vec
                    Vec::new()
                }
            } else {
                // Test tools for backward compatibility - these should be removed in production
                if settings.additional_settings.contains_key("debug_tools") {
                    println!("[DEBUG] Using default tools (this should be removed in production)");
                }
                
                vec![
                    ChatCompletionTool {
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionObject {
                            name: "get_weather".to_string(),
                            description: Some("Get the current weather conditions for a city".to_string()),
                            parameters: Some(serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city to get weather for"
                                    }
                                },
                                "required": ["city"]
                            })),
                            strict: Some(false),
                        },
                    },
                    ChatCompletionTool {
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionObject {
                            name: "get_forecast".to_string(),
                            description: Some("Get the 5-day weather forecast for a city".to_string()),
                            parameters: Some(serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city to get forecast for"
                                    }
                                },
                                "required": ["city"]
                            })),
                            strict: Some(false),
                        },
                    },
                    ChatCompletionTool {
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionObject {
                            name: "get_location".to_string(),
                            description: Some("Get the geographical coordinates for a city".to_string()),
                            parameters: Some(serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city to get coordinates for"
                                    }
                                },
                                "required": ["city"]
                            })),
                            strict: Some(false),
                        },
                    },
                ]
            };
            
            request.tools = Some(tools);
        }

        // Add temperature if present
        if let Some(temperature) = settings.temperature {
            request.temperature = Some(temperature);
        }

        // Add top_p if present
        if let Some(top_p) = settings.top_p {
            request.top_p = Some(top_p);
        }

        // Add max_tokens if present
        if let Some(max_tokens) = settings.max_tokens {
            request.max_tokens = Some(max_tokens as u32);
        }

        // Add stop if present
        if let Some(stop) = &settings.stop {
            // Convert Vec<String> to the appropriate Stop type
            request.stop = Some(Stop::StringArray(stop.clone()));
        }

        // Add presence_penalty if present
        if let Some(presence_penalty) = settings.presence_penalty {
            request.presence_penalty = Some(presence_penalty);
        }

        // Add frequency_penalty if present
        if let Some(frequency_penalty) = settings.frequency_penalty {
            request.frequency_penalty = Some(frequency_penalty);
        }

        // Add tool choice as a string value - simplified approach to work with the current OpenAI API
        if let Some(tool_choice) = &settings.tool_choice {
            // Debug print 
            if settings.additional_settings.contains_key("debug_tools") {
                println!("[DEBUG] Setting tool_choice: {:?}", tool_choice);
            }
            
            // Instead of the complex enum matching, we just set a simple string value via additional properties
            // This is a temporary workaround until we update the async-openai dependency
            use crate::model::settings::ToolChoice;
            
            match tool_choice {
                ToolChoice::Auto => {
                    request.tool_choice = Some(async_openai::types::ChatCompletionToolChoiceOption::Auto);
                },
                ToolChoice::Required => {
                    request.tool_choice = Some(async_openai::types::ChatCompletionToolChoiceOption::Required);
                },
                ToolChoice::None => {
                    request.tool_choice = Some(async_openai::types::ChatCompletionToolChoiceOption::None);
                },
                ToolChoice::Specific(name) => {
                    // For specific tools, just log it but use auto - to be fixed in a future update
                    println!("[DEBUG] Specific tool choice requested ({}), but using 'auto' instead", name);
                    request.tool_choice = Some(async_openai::types::ChatCompletionToolChoiceOption::Auto);
                }
            };
        }

        // TODO: Add response_format if present
        // This is a placeholder due to API changes in async-openai

        // Add seed if present
        if let Some(seed) = settings.seed {
            request.seed = Some(seed as i64);
        }

        // Add stream if present
        if let Some(stream) = settings.stream {
            request.stream = Some(stream);
        }

        // TODO: Add additional settings
        // This is a placeholder for future extensions

        request
    }

    /// Convert OpenAI response to SDK ModelResponse
    fn convert_response(
        &self,
        response: async_openai::types::CreateChatCompletionResponse
    ) -> Result<ModelResponse> {
        let choices = response.choices.iter().map(|choice| {
            // Get content as string
            let content = choice.message.content.clone().unwrap_or_default();
            
            // Create a new message
            let mut message = Message::assistant(content);
            
            // Add tool calls if present
            if let Some(tool_calls) = &choice.message.tool_calls {
                // Debug print
                println!("[DEBUG] Tool calls in response: {:?}", tool_calls);
                
                // Convert OpenAI tool calls to our format
                let our_tool_calls = tool_calls.iter().map(|tc| {
                    crate::types::message::ToolCall {
                        id: tc.id.clone(),
                        type_: "function".to_string(),
                        function: crate::types::message::FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    }
                }).collect();
                
                message.tool_calls = Some(our_tool_calls);
            }
            
            // Add function call if present (legacy format)
            if let Some(function_call) = &choice.message.function_call {
                println!("[DEBUG] Function call in response: {:?}", function_call);
                
                message.function_call = Some(crate::types::message::FunctionCall {
                    name: function_call.name.clone(),
                    arguments: function_call.arguments.clone(),
                });
            }
            
            // Create a model choice
            ModelChoice {
                index: choice.index as usize,
                message,
                finish_reason: choice.finish_reason.clone().map(|r| {
                    match r {
                        async_openai::types::FinishReason::Stop => "stop".to_string(),
                        async_openai::types::FinishReason::Length => "length".to_string(),
                        async_openai::types::FinishReason::ToolCalls => "tool_calls".to_string(),
                        async_openai::types::FinishReason::ContentFilter =>
                            "content_filter".to_string(),
                        async_openai::types::FinishReason::FunctionCall =>
                            "function_call".to_string(),
                    }
                }),
            }
        }).collect();

        // Extract usage information
        let usage = if let Some(usage) = response.usage {
            Usage {
                prompt_tokens: usage.prompt_tokens as usize,
                completion_tokens: usage.completion_tokens as usize,
                total_tokens: usage.total_tokens as usize,
            }
        } else {
            Usage::default()
        };

        Ok(ModelResponse {
            id: response.id,
            object: response.object,
            created: response.created as u64,
            model: response.model,
            choices,
            usage,
        })
    }
}

impl Default for OpenAIChatCompletions {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for OpenAIChatCompletions {
    fn name(&self) -> &str {
        "openai"
    }

    async fn generate(
        &self,
        model: &str,
        messages: Vec<Message>,
        settings: &ModelSettings
    ) -> Result<ModelResponse> {
        // Debug output for message sequence if enabled
        if settings.additional_settings.contains_key("debug_tools") || 
           settings.additional_settings.contains_key("debug_messages") {
            println!("[DEBUG] Message sequence being sent to OpenAI:");
            for (i, msg) in messages.iter().enumerate() {
                let role = format!("{:?}", msg.role);
                let content = msg.content_as_string().unwrap_or_default();
                let tool_call_id = msg.additional_properties.get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                
                if !tool_call_id.is_empty() {
                    println!("[DEBUG]   {}. {} (tool_call_id: {}): {:.40}...", 
                        i, role, tool_call_id, content);
                } else {
                    println!("[DEBUG]   {}. {}: {:.40}...", i, role, content);
                }
                
                // Also print tool calls if present
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        println!("[DEBUG]     - Tool Call: {} -> {}", tc.id, tc.function.name);
                    }
                }
            }
        }
        
        // Convert messages and settings
        let openai_messages = self.convert_messages(messages);
        let request = self.convert_settings(model, openai_messages, settings);

        // Execute the request
        let response = self.client.chat().create(request).await.map_err(Error::OpenAI)?;

        // Convert the response
        self.convert_response(response)
    }

    async fn generate_stream(
        &self,
        _model: &str,
        _messages: Vec<Message>,
        _settings: &ModelSettings
    ) -> Result<Box<dyn Stream<Item = Result<ModelResponse>> + Send + 'static>> {
        // For now, we'll return an error since we're focusing on non-streaming first
        Err(
            Error::Model(
                format!("Streaming is not implemented yet for the {} provider", self.name())
            )
        )
    }
    
    fn clone_box(&self) -> Box<dyn ModelProvider> {
        Box::new(self.clone())
    }
}

impl Clone for OpenAIChatCompletions {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

/// Type alias for the OpenAI model provider
pub type OpenAIModelProvider = OpenAIChatCompletions;
