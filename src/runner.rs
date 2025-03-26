//! Runner implementation
//!
//! This module provides the Runner struct for executing agents.

use std::sync::Arc;
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::collections::HashMap;
use std::env;

// Auto-load dotenv in library code
use dotenv::dotenv;

use futures::{ Stream, channel::mpsc };
use chrono::Utc;
use uuid::Uuid;

use crate::agent::Agent;
use crate::model::settings::ModelSettings;
use crate::model::provider::ModelProvider;
use crate::model::openai::OpenAIChatCompletions;
use crate::types::context::{ RunContext, RunContextWrapper };
use crate::types::message::{ Message, Role };
use crate::types::result::{
    RunResult,
    StreamedRunResult,
    ModelResponse,
    Usage,
    Item,
    StreamEvent,
    ModelChoice,
};
use crate::types::error::{ Result, Error };
use crate::tracing::trace::{ Trace, trace };
use crate::utils;

/// Configuration for a run
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// Trace ID
    pub trace_id: Option<String>,

    /// Group ID for related traces
    pub group_id: Option<String>,

    /// Maximum number of turns
    pub max_turns: Option<usize>,

    /// Whether to include sensitive data in traces
    pub trace_include_sensitive_data: bool,

    /// Whether to disable tracing
    pub tracing_disabled: bool,

    /// Name of the workflow
    pub workflow_name: Option<String>,

    /// Metadata for the trace
    pub metadata: HashMap<String, serde_json::Value>,

    /// Whether to try to continue after a tool error
    pub continue_on_tool_error: bool,

    /// Return the raw response from the model
    pub return_raw_response: bool,

    /// Return prompt tokens
    pub return_prompt_tokens: bool,

    /// Return completion tokens
    pub return_completion_tokens: bool,

    /// Return tool calls in items
    pub return_tool_calls: bool,

    /// Return intermediate messages in items
    pub return_intermediate_messages: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            trace_id: None,
            group_id: None,
            max_turns: Some(10),
            trace_include_sensitive_data: false,
            tracing_disabled: utils::is_tracing_disabled(),
            workflow_name: None,
            metadata: HashMap::new(),
            continue_on_tool_error: false,
            return_raw_response: true,
            return_prompt_tokens: true,
            return_completion_tokens: true,
            return_tool_calls: true,
            return_intermediate_messages: true,
        }
    }
}

impl RunConfig {
    /// Create a new run configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the trace ID
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set the group ID
    pub fn with_group_id(mut self, group_id: impl Into<String>) -> Self {
        self.group_id = Some(group_id.into());
        self
    }

    /// Set the maximum number of turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Set whether to include sensitive data in traces
    pub fn with_trace_include_sensitive_data(mut self, include: bool) -> Self {
        self.trace_include_sensitive_data = include;
        self
    }

    /// Set whether to disable tracing
    pub fn with_tracing_disabled(mut self, disabled: bool) -> Self {
        self.tracing_disabled = disabled;
        self
    }

    /// Set the workflow name
    pub fn with_workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = Some(name.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set whether to try to continue after a tool error
    pub fn with_continue_on_tool_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_tool_error = continue_on_error;
        self
    }

    /// Set whether to return the raw response from the model
    pub fn with_return_raw_response(mut self, return_raw: bool) -> Self {
        self.return_raw_response = return_raw;
        self
    }

    /// Set whether to return prompt tokens
    pub fn with_return_prompt_tokens(mut self, return_tokens: bool) -> Self {
        self.return_prompt_tokens = return_tokens;
        self
    }

    /// Set whether to return completion tokens
    pub fn with_return_completion_tokens(mut self, return_tokens: bool) -> Self {
        self.return_completion_tokens = return_tokens;
        self
    }

    /// Set whether to return tool calls in items
    pub fn with_return_tool_calls(mut self, return_calls: bool) -> Self {
        self.return_tool_calls = return_calls;
        self
    }

    /// Set whether to return intermediate messages in items
    pub fn with_return_intermediate_messages(mut self, return_messages: bool) -> Self {
        self.return_intermediate_messages = return_messages;
        self
    }

    /// Set minimal return options (just the final output)
    pub fn minimal(mut self) -> Self {
        self.return_raw_response = false;
        self.return_prompt_tokens = false;
        self.return_completion_tokens = false;
        self.return_tool_calls = false;
        self.return_intermediate_messages = false;
        self
    }
}

/// Input to a run
pub enum RunInput {
    /// A single text message
    Text(String),

    /// A list of messages
    Messages(Vec<Message>),
}

impl From<String> for RunInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for RunInput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Vec<Message>> for RunInput {
    fn from(messages: Vec<Message>) -> Self {
        Self::Messages(messages)
    }
}

/// Runs agents and orchestrates the execution flow
pub struct Runner;

impl Runner {
    /// Run a handoff between agents
    async fn run_handoff<C: Clone + Send + Sync + Default + 'static>(
        from_agent: &Agent<C>,
        to_agent: &Agent<C>,
        handoff: &crate::handoff::Handoff<C>,
        input: Vec<Message>,
        context: &RunContext<C>,
        config: &RunConfig,
        trace: &mut Trace
    ) -> Result<RunResult> {
        // Call the handoff callback if present
        if let Some(callback) = &handoff.on_handoff {
            let context_wrapper = crate::types::context::RunContextWrapper::new(context);
            callback(&context_wrapper).await;
        }

        // Filter the input if a filter is specified
        let filtered_input = if let Some(filter) = &handoff.input_filter {
            // Create a handoff input data object
            let handoff_input = crate::handoff::HandoffInputData {
                input_history: input.clone(),
                pre_handoff_items: Vec::new(), // In a real implementation, these would be populated
                new_items: Vec::new(), // In a real implementation, these would be populated
            };

            // Apply the filter
            let filtered = filter(handoff_input).await;

            // Return the filtered history
            filtered.input_history
        } else {
            // No filter, use the original input
            input.clone()
        };

        // Create a span for the handoff
        let span = crate::tracing::span::handoff_span(
            &trace.id,
            None, // Parent span ID would be set in a real implementation
            &to_agent.name,
            Some(serde_json::to_string(&filtered_input).unwrap_or_default()),
            None // Output will be set later
        );

        // Add the span to the trace
        trace.add_span(span);

        // Get the model and provider
        let model = to_agent.model.as_deref().unwrap_or("gpt-3.5-turbo");
        
        // Only use the default provider if one is not provided on the agent
        let provider = match &to_agent.model_provider {
            Some(provider) => provider.as_ref(),
            None => {
                // Try to load dotenv and check if we have an OpenAI API key
                dotenv().ok(); // Load environment variables from .env file
                if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                    // Return cached provider
                    static DEFAULT_PROVIDER: std::sync::OnceLock<Box<dyn crate::model::provider::ModelProvider>> = 
                        std::sync::OnceLock::new();
                    
                    let provider = DEFAULT_PROVIDER.get_or_init(|| {
                        Box::new(crate::model::openai::OpenAIModelProvider::with_api_key(api_key))
                    });
                    provider.as_ref()
                } else {
                    // Try to use default without API key (will likely fail)
                    static DEFAULT_PROVIDER: std::sync::OnceLock<Box<dyn crate::model::provider::ModelProvider>> = 
                        std::sync::OnceLock::new();
                    
                    let provider = DEFAULT_PROVIDER.get_or_init(|| {
                        Box::new(crate::model::openai::OpenAIModelProvider::default())
                    });
                    provider.as_ref()
                }
            }
        };
        let default_settings = ModelSettings::default();
        let settings = to_agent.model_settings.as_ref().unwrap_or(&default_settings);
        
        // Generate a response
        let response_result = provider.generate(
            model,
            filtered_input.clone(),
            settings
        ).await;
        
        // Handle the response
        let response = match response_result {
            Ok(response) => response,
            Err(e) => {
                return Err(e);
            }
        };
        
        // Extract the assistant message from the response
        let last_choice = response.choices.last().cloned().ok_or_else(|| {
            Error::Model("No choices in response".to_string())
        })?;
        
        // Get the final output text
        let final_output = last_choice.message.content_as_string().unwrap_or_default();
        
        // Create items tracking
        let mut all_items = Vec::new();
        
        // Add the handoff item
        let handoff_item = Item::Handoff {
            id: Uuid::new_v4().to_string(),
            handoff_name: to_agent.name.clone(),
            args: None,
        };
        all_items.push(handoff_item);
        
        // Add message items
        for message in &filtered_input {
            let item = Item::Message {
                id: Uuid::new_v4().to_string(),
                role: format!("{:?}", message.role).to_lowercase(),
                content: message.content_as_string().unwrap_or_default(),
                tool_calls: message.tool_calls.clone(),
                function_call: message.function_call.clone(),
                status: "completed".to_string(),
            };
            all_items.push(item);
        }
        
        // Add the assistant response
        let assistant_item = Item::Message {
            id: Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content: final_output.clone(),
            tool_calls: last_choice.message.tool_calls.clone(),
            function_call: last_choice.message.function_call.clone(),
            status: "completed".to_string(),
        };
        all_items.push(assistant_item);
        
        // Create the result
        let result = RunResult::new(
            final_output,
            None, // No structured output
            all_items,
            to_agent.name.clone(),
            vec![response.clone()],
            response.usage.clone()
        );

        // Return the result
        Ok(result)
    }

    /// Run an agent asynchronously
    pub async fn run<C: Send + Sync + Clone + Default + 'static>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>
    ) -> Result<RunResult> {
        // Create a context with a default value if none is provided
        let context = RunContext::new(context.unwrap_or_default());
        
        // Call run_with_context with the created context
        let input = input.into();
        let config = config.unwrap_or_default();
        
        Self::run_with_context(agent, input, context, Some(config)).await
    }

    /// Run an agent with an existing context
    async fn run_with_context<C: Send + Sync + Clone + Default + 'static>(
        agent: &Agent<C>,
        input: RunInput,
        context: RunContext<C>,
        config: Option<RunConfig>
    ) -> Result<RunResult> {
        let config = config.unwrap_or_default();

        // Create a trace if enabled
        let trace_id = config.trace_id
            .as_ref()
            .map(String::clone)
            .unwrap_or_else(utils::generate_trace_id);
        let mut trace = Trace::new(
            config.workflow_name.clone().unwrap_or_else(|| "Agent trace".to_string())
        );
        trace.id = trace_id.clone();
        if let Some(group_id) = &config.group_id {
            trace.group_id = Some(group_id.clone());
        }
        for (key, value) in &config.metadata {
            trace.metadata.insert(key.clone(), value.clone());
        }
        if config.tracing_disabled {
            trace.disabled = true;
        }

        // Start the trace
        trace.start();

        // Convert input to messages
        let messages = match input {
            RunInput::Text(text) => { vec![Message::user(text)] }
            RunInput::Messages(messages) => { messages }
        };

        // Initialize the run state
        let mut current_agent = agent;
        let mut current_messages = messages;
        let max_turns = config.max_turns.unwrap_or(10);
        let mut turn = 0;
        let mut usage = Usage::default();
        let mut all_responses: Vec<ModelResponse> = Vec::new();
        let mut all_items: Vec<Item> = Vec::new();

        // Add initial messages to items
        for message in &current_messages {
            // Convert to Item
            let item = Item::Message {
                id: Uuid::new_v4().to_string(),
                role: format!("{:?}", message.role).to_lowercase(),
                content: message.content_as_string().unwrap_or_default(),
                tool_calls: message.tool_calls.clone(),
                function_call: message.function_call.clone(),
                status: "completed".to_string(),
            };

            all_items.push(item);
        }

        // Start the run loop
        while turn < max_turns {
            turn += 1;

            // Apply input guardrails
            for guardrail in &current_agent.input_guardrails {
                let ctx_wrapper = RunContextWrapper::new(&context);
                let last_message = current_messages
                    .last()
                    .and_then(|m| m.content_as_string())
                    .unwrap_or_default();
                let result = guardrail.run(&context, current_agent, &last_message).await;

                if result.tripwire_triggered {
                    return Err(
                        Error::InputGuardrailTriggered(
                            serde_json::to_string(&result.output_info).unwrap_or_default()
                        )
                    );
                }
            }

            // Get the model and provider
            let model = current_agent.model.clone().unwrap_or_else(|| "gpt-4o".to_string());
            
            // Create a copy of model settings and add tools if they exist
            let mut model_settings = current_agent.model_settings.clone().unwrap_or_default();
            
            // Add agent tools to model settings if there are any
            if !current_agent.tools.is_empty() {
                // Convert tools to JSON representation
                let tools_json: Vec<serde_json::Value> = current_agent.tools.iter()
                    .map(|tool| {
                        serde_json::json!({
                            "name": tool.name(),
                            "description": tool.description(),
                            "parameters": tool.parameters_schema()
                        })
                    })
                    .collect();
                
                // Add tools to settings
                model_settings.additional_settings.insert("tools".to_string(), serde_json::json!(tools_json));
                
                // Debug logging if enabled
                if model_settings.additional_settings.contains_key("debug_tools") {
                    println!("[DEBUG] Added {} tools to model settings", tools_json.len());
                }
            }
            
            let provider: Box<dyn ModelProvider> = if
                let Some(provider) = &current_agent.model_provider
            {
                provider.clone()
            } else {
                // Use default OpenAI provider
                if let Some(api_key) = utils::get_openai_api_key() {
                    Box::new(OpenAIChatCompletions::with_api_key(api_key))
                } else {
                    Box::new(OpenAIChatCompletions::new())
                }
            };

            // Generate a response
            // Hook: on_generate_start
            current_agent.hooks.on_generate_start(&RunContextWrapper::new(&context), current_agent);

            // Add a system message if not present
            let mut messages_with_system = Vec::new();
            // Make a clone of current_messages for checking if it contains a system message
            let contains_system = current_messages.iter().any(|m| m.role == Role::System);
            if !contains_system {
                // Get instructions
                let instructions = match &current_agent.instructions {
                    crate::agent::Instructions::Static(s) => s.clone(),
                    crate::agent::Instructions::Dynamic(f) => {
                        f(&RunContextWrapper::new(&context), current_agent)
                    }
                    crate::agent::Instructions::AsyncDynamic(f) => {
                        f(&RunContextWrapper::new(&context), current_agent).await
                    }
                };

                messages_with_system.push(Message::system(instructions));
            }
            // Clone current_messages here to avoid moving it
            let current_messages_clone = current_messages.clone();
            messages_with_system.extend(current_messages_clone);

            // Generate a response
            let response = provider.generate(&model, messages_with_system, &model_settings).await?;

            // Add to all responses
            usage.combine(&response.usage);
            all_responses.push(response.clone());

            // Get the message from the response
            let message = if let Some(choice) = response.choices.first() {
                choice.message.clone()
            } else {
                return Err(Error::InvalidModelResponse("No choices in response".to_string()));
            };

            // Add the message to all items
            let item = Item::Message {
                id: Uuid::new_v4().to_string(),
                role: format!("{:?}", message.role).to_lowercase(),
                content: message.content_as_string().unwrap_or_default(),
                tool_calls: message.tool_calls.clone(),
                function_call: message.function_call.clone(),
                status: "completed".to_string(),
            };

            all_items.push(item);

            // Hook: on_generate_end
            current_agent.hooks.on_generate_end(
                &RunContextWrapper::new(&context),
                current_agent,
                &message.content_as_string().unwrap_or_default()
            );

            // Check for tool calls
            if let Some(tool_calls) = &message.tool_calls {
                if !tool_calls.is_empty() {
                    // Create a span for the tool group (all tool calls in this message)
                    let tool_group_span = crate::tracing::span::custom_span(
                        &trace.id,
                        None,
                        "tool_group",
                        HashMap::new()
                    );
                    trace.add_span(tool_group_span);

                    // Process tool calls
                    for tool_call in tool_calls {
                        let tool_name = &tool_call.function.name;

                        // Check if this is a handoff first
                        let handoff = current_agent.handoffs
                            .iter()
                            .find(|h| h.tool_name() == *tool_name);

                        if let Some(handoff) = handoff {
                            // This is a handoff disguised as a tool call

                            // Hook: on_handoff_start
                            current_agent.hooks.on_handoff_start(
                                &RunContextWrapper::new(&context),
                                current_agent,
                                handoff
                            );

                            // Add handoff to items
                            let handoff_item = Item::Handoff {
                                id: tool_call.id.clone(),
                                handoff_name: handoff.agent.name.clone(),
                                args: Some(tool_call.function.arguments.clone()),
                            };

                            all_items.push(handoff_item);

                            // Execute the handoff
                            let handoff_result = Self::run_handoff(
                                current_agent,
                                &handoff.agent,
                                handoff,
                                current_messages.clone(),
                                &context,
                                &config,
                                &mut trace
                            ).await?;

                            // Hook: on_handoff_end
                            current_agent.hooks.on_handoff_end(
                                &RunContextWrapper::new(&context),
                                current_agent,
                                handoff,
                                &handoff_result.final_output
                            );

                            // Return the handoff result
                            return Ok(handoff_result);
                        } else {
                            // This is a regular tool call
                            // Find the tool
                            let tool = current_agent.tools.iter().find(|t| t.name() == tool_name);

                            if let Some(tool) = tool {
                                // Debug print if enabled
                                if current_agent.model_settings.as_ref()
                                    .and_then(|s| s.additional_settings.get("debug_tools"))
                                    .is_some() 
                                {
                                    println!("[DEBUG] Found tool '{}' for execution", tool_name);
                                }
                                // Create a span for the tool call
                                let tool_span = crate::tracing::span::tool_span(
                                    &trace.id,
                                    None,
                                    tool_name,
                                    Some(tool_call.function.arguments.clone()),
                                    None
                                );
                                trace.add_span(tool_span);

                                // Hook: on_tool_start
                                current_agent.hooks.on_tool_start(
                                    &RunContextWrapper::new(&context),
                                    current_agent,
                                    tool.as_ref()
                                );

                                // Parse the arguments
                                let args_result = serde_json::from_str(
                                    &tool_call.function.arguments
                                );

                                // Add tool call to items if configured to do so
                                if config.return_tool_calls {
                                    let tool_call_item = Item::ToolCall {
                                        id: tool_call.id.clone(),
                                        tool_name: tool_name.clone(),
                                        args: tool_call.function.arguments.clone(),
                                    };

                                    all_items.push(tool_call_item);
                                }

                                // Handle argument parsing errors
                                let args = match args_result {
                                    Ok(args) => args,
                                    Err(e) => {
                                        let error_msg = format!("Failed to parse arguments: {}", e);

                                        // Add error as tool result
                                        let tool_result_item = Item::ToolResult {
                                            call_id: tool_call.id.clone(),
                                            tool_name: tool_name.clone(),
                                            result: error_msg.clone(),
                                        };

                                        all_items.push(tool_result_item);

                                        // Add error message
                                        current_messages.push(
                                            Message::tool(tool_name.clone(), error_msg)
                                        );

                                        // If we're not continuing on errors, return an error
                                        if !config.continue_on_tool_error {
                                            return Err(
                                                Error::Tool(
                                                    format!("Failed to parse arguments: {}", e)
                                                )
                                            );
                                        }

                                        // Skip to the next tool call
                                        continue;
                                    }
                                };

                                // Debug print the arguments
                                if current_agent.model_settings.as_ref()
                                    .and_then(|s| s.additional_settings.get("debug_tools"))
                                    .is_some() 
                                {
                                    println!("[DEBUG] Executing tool {} with args: {:?}", tool_name, args);
                                }
                                
                                // Execute the tool
                                let result = match tool.execute_raw(&context, args).await {
                                    Ok(result) => result,
                                    Err(e) => {
                                        let error_msg = format!("Tool execution failed: {}", e);

                                        // Add error as tool result
                                        let tool_result_item = Item::ToolResult {
                                            call_id: tool_call.id.clone(),
                                            tool_name: tool_name.clone(),
                                            result: error_msg.clone(),
                                        };

                                        all_items.push(tool_result_item);

                                        // Add error message
                                        current_messages.push(
                                            Message::tool(tool_name.clone(), error_msg)
                                        );

                                        // If we're not continuing on errors, return an error
                                        if !config.continue_on_tool_error {
                                            return Err(
                                                Error::Tool(format!("Tool execution failed: {}", e))
                                            );
                                        }

                                        // Skip to the next tool call
                                        continue;
                                    }
                                };

                                // Update the tool span with the result
                                let tool_span = crate::tracing::span::tool_span(
                                    &trace.id,
                                    None,
                                    tool_name,
                                    Some(tool_call.function.arguments.clone()),
                                    Some(result.clone())
                                );
                                trace.add_span(tool_span);

                                // Add tool result to items if configured to do so
                                if config.return_tool_calls {
                                    let tool_result_item = Item::ToolResult {
                                        call_id: tool_call.id.clone(),
                                        tool_name: tool_name.clone(),
                                        result: result.clone(),
                                    };

                                    all_items.push(tool_result_item);
                                }

                                // Create a tool response message linked to this tool call
                                // Format: first create a normal tool message
                                let mut tool_message = Message::tool(tool_name.clone(), result.clone());
                                
                                // Add a crucial field: the tool_call_id to link this response to the 
                                // specific tool call from the assistant. This is required for the 
                                // OpenAI API to recognize this as a valid tool response.
                                tool_message.additional_properties.insert("tool_call_id".to_string(), 
                                    serde_json::json!(tool_call.id.clone()));
                                    
                                // Debug logging if enabled
                                if current_agent.model_settings.as_ref()
                                    .and_then(|s| s.additional_settings.get("debug_tools"))
                                    .is_some() 
                                {
                                    println!("[DEBUG] Adding tool response for call ID: {}", tool_call.id);
                                }
                                
                                // Add the tool response message
                                current_messages.push(tool_message);

                                // Hook: on_tool_end
                                current_agent.hooks.on_tool_end(
                                    &RunContextWrapper::new(&context),
                                    current_agent,
                                    tool.as_ref(),
                                    &result
                                );

                                // If the agent is configured to stop after the first tool, return the result
                                if
                                    current_agent.tool_use_behavior ==
                                    crate::agent::ToolUseBehavior::StopOnFirstTool
                                {
                                    // Finish the trace
                                    trace.finish();

                                    return Ok(
                                        RunResult::new(
                                            result,
                                            None,
                                            all_items,
                                            current_agent.name.clone(),
                                            all_responses,
                                            usage
                                        )
                                    );
                                }
                            } else {
                                // Tool not found
                                let error_msg = format!("Tool not found: {}", tool_name);

                                // Add error as tool result
                                let tool_result_item = Item::ToolResult {
                                    call_id: tool_call.id.clone(),
                                    tool_name: tool_name.clone(),
                                    result: error_msg.clone(),
                                };

                                all_items.push(tool_result_item);

                                // Add error message
                                current_messages.push(Message::tool(tool_name.clone(), error_msg));

                                // If we're not continuing on errors, return an error
                                if !config.continue_on_tool_error {
                                    return Err(Error::ToolNotFound(tool_name.clone()));
                                }
                            }
                        }
                    }
                } else {
                    // No tool calls, just add the message if configured to do so
                    if config.return_intermediate_messages {
                        current_messages.push(message.clone());
                    }
                }
            } else if let Some(function_call) = &message.function_call {
                // Legacy function call format
                let tool_name = &function_call.name;

                // Check if this is a handoff
                let handoff = current_agent.handoffs.iter().find(|h| h.tool_name() == *tool_name);

                if let Some(handoff) = handoff {
                    // This is a handoff

                    // Hook: on_handoff_start
                    current_agent.hooks.on_handoff_start(
                        &RunContextWrapper::new(&context),
                        current_agent,
                        handoff
                    );

                    // Add handoff to items
                    let handoff_item = Item::Handoff {
                        id: Uuid::new_v4().to_string(),
                        handoff_name: handoff.agent.name.clone(),
                        args: Some(function_call.arguments.clone()),
                    };

                    all_items.push(handoff_item);

                    // Execute the handoff
                    let handoff_result = Self::run_handoff(
                        current_agent,
                        &handoff.agent,
                        handoff,
                        current_messages.clone(),
                        &context,
                        &config,
                        &mut trace
                    ).await?;

                    // Hook: on_handoff_end
                    current_agent.hooks.on_handoff_end(
                        &RunContextWrapper::new(&context),
                        current_agent,
                        handoff,
                        &handoff_result.final_output
                    );

                    // Return the handoff result
                    return Ok(handoff_result);
                } else {
                    // This is a regular tool call
                    // Find the tool
                    let tool = current_agent.tools.iter().find(|t| t.name() == tool_name);

                    if let Some(tool) = tool {
                        // Create a span for the tool call
                        let tool_span = crate::tracing::span::tool_span(
                            &trace.id,
                            None,
                            tool_name,
                            Some(function_call.arguments.clone()),
                            None
                        );
                        trace.add_span(tool_span);

                        // Hook: on_tool_start
                        current_agent.hooks.on_tool_start(
                            &RunContextWrapper::new(&context),
                            current_agent,
                            tool.as_ref()
                        );

                        // Parse the arguments
                        let args_result = serde_json::from_str(&function_call.arguments);

                        // Add tool call to items if configured to do so
                        if config.return_tool_calls {
                            let tool_call_item = Item::ToolCall {
                                id: Uuid::new_v4().to_string(),
                                tool_name: tool_name.clone(),
                                args: function_call.arguments.clone(),
                            };

                            all_items.push(tool_call_item);
                        }

                        // Handle argument parsing errors
                        let args = match args_result {
                            Ok(args) => args,
                            Err(e) => {
                                let error_msg = format!("Failed to parse arguments: {}", e);

                                // Add error as tool result
                                let tool_result_item = Item::ToolResult {
                                    call_id: Uuid::new_v4().to_string(),
                                    tool_name: tool_name.clone(),
                                    result: error_msg.clone(),
                                };

                                all_items.push(tool_result_item);

                                // Add error message
                                current_messages.push(Message::tool(tool_name.clone(), error_msg));

                                // If we're not continuing on errors, return an error
                                if !config.continue_on_tool_error {
                                    return Err(
                                        Error::Tool(format!("Failed to parse arguments: {}", e))
                                    );
                                }

                                // Continue with the next message
                                continue;
                            }
                        };

                        // Execute the tool
                        let result = match tool.execute_raw(&context, args).await {
                            Ok(result) => result,
                            Err(e) => {
                                let error_msg = format!("Tool execution failed: {}", e);

                                // Add error as tool result
                                let tool_result_item = Item::ToolResult {
                                    call_id: Uuid::new_v4().to_string(),
                                    tool_name: tool_name.clone(),
                                    result: error_msg.clone(),
                                };

                                all_items.push(tool_result_item);

                                // Add error message
                                current_messages.push(Message::tool(tool_name.clone(), error_msg));

                                // If we're not continuing on errors, return an error
                                if !config.continue_on_tool_error {
                                    return Err(
                                        Error::Tool(format!("Tool execution failed: {}", e))
                                    );
                                }

                                // Continue with the next message
                                continue;
                            }
                        };

                        // Update the tool span with the result
                        let tool_span = crate::tracing::span::tool_span(
                            &trace.id,
                            None,
                            tool_name,
                            Some(function_call.arguments.clone()),
                            Some(result.clone())
                        );
                        trace.add_span(tool_span);

                        // Add tool result to items if configured to do so
                        if config.return_tool_calls {
                            let tool_result_item = Item::ToolResult {
                                call_id: Uuid::new_v4().to_string(),
                                tool_name: tool_name.clone(),
                                result: result.clone(),
                            };

                            all_items.push(tool_result_item);
                        }

                        // Add the tool result as a message
                        current_messages.push(Message::tool(tool_name.clone(), result.clone()));

                        // Hook: on_tool_end
                        current_agent.hooks.on_tool_end(
                            &RunContextWrapper::new(&context),
                            current_agent,
                            tool.as_ref(),
                            &result
                        );

                        // If the agent is configured to stop after the first tool, return the result
                        if
                            current_agent.tool_use_behavior ==
                            crate::agent::ToolUseBehavior::StopOnFirstTool
                        {
                            // Finish the trace
                            trace.finish();

                            return Ok(
                                RunResult::new(
                                    result,
                                    None,
                                    all_items,
                                    current_agent.name.clone(),
                                    all_responses,
                                    usage
                                )
                            );
                        }
                    } else {
                        // Tool not found
                        let error_msg = format!("Tool not found: {}", tool_name);

                        // Add error as tool result
                        let tool_result_item = Item::ToolResult {
                            call_id: Uuid::new_v4().to_string(),
                            tool_name: tool_name.clone(),
                            result: error_msg.clone(),
                        };

                        all_items.push(tool_result_item);

                        // Add error message
                        current_messages.push(Message::tool(tool_name.clone(), error_msg));

                        // If we're not continuing on errors, return an error
                        if !config.continue_on_tool_error {
                            return Err(Error::ToolNotFound(tool_name.clone()));
                        }
                    }
                }
            } else {
                // No tool calls or function calls, just add the message
                current_messages.push(message.clone());

                // Check if this is a final answer
                let final_output = message.content_as_string().unwrap_or_default();

                // Apply output guardrails
                for guardrail in &current_agent.output_guardrails {
                    let ctx_wrapper = RunContextWrapper::new(&context);
                    let result = guardrail.run(&context, current_agent, &final_output).await;

                    if result.tripwire_triggered {
                        return Err(
                            Error::OutputGuardrailTriggered(
                                serde_json::to_string(&result.output_info).unwrap_or_default()
                            )
                        );
                    }
                }

                // Parse structured output if needed
                let structured_output = if let Some(output_type) = &current_agent.output_type {
                    // In a real implementation, we'd parse the output based on the schema
                    // This is a placeholder
                    None
                } else {
                    None
                };

                // Finish the trace
                trace.finish();

                // Return the final result
                return Ok(
                    RunResult::new(
                        final_output,
                        structured_output,
                        all_items,
                        current_agent.name.clone(),
                        all_responses,
                        usage
                    )
                );
            }
        }

        // If we get here, we exceeded the maximum number of turns
        Err(Error::MaxTurnsExceeded(max_turns))
    }

    /// Run an agent synchronously (blocking)
    pub fn run_sync<C: Send + Sync + Clone + Default + 'static>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>
    ) -> Result<RunResult> {
        // Create a new runtime for blocking calls
        let runtime = tokio::runtime::Builder
            ::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Other(format!("Failed to create runtime: {}", e)))?;

        // Run the async function
        runtime.block_on(Self::run(agent, input, context, config))
    }

    /// Run an agent with streaming output
    pub async fn run_streamed<C: Send + Sync + Clone + Default + 'static>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>
    ) -> Result<StreamedRunResult> {
        // Create channels for streaming events
        let (mut tx, rx) = mpsc::channel(100);

        // Clone necessary values for the task
        let agent_clone = agent.clone();
        let input = input.into();
        let config = config.unwrap_or_default();
        let context_value = context.unwrap_or_else(|| unsafe { std::mem::zeroed() });

        // Start the run in a separate task
        let handle = tokio::spawn(async move {
            let result = Self::run(&agent_clone, input, Some(context_value), Some(config)).await;

            // Send any events that weren't sent during the run
            if let Ok(result) = &result {
                // In a real implementation, we'd send events for each message, tool call, etc.
                let final_event = StreamEvent {
                    event_type: "final_output".to_string(),
                    data: serde_json::json!({
                        "output": result.final_output,
                    }),
                    timestamp: Utc::now(),
                };

                let _ = tx.try_send(final_event);
            }

            result
        });

        Ok(StreamedRunResult {
            events: Box::pin(rx),
            result: Box::pin(async move {
                handle.await.map_err(|e| Error::Other(format!("Task failed: {}", e)))?
            }),
        })
    }
}
