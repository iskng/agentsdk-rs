//! Handoff implementation
//!
//! This module provides the Handoff struct and related functions.

use std::sync::Arc;
use std::future::Future;
use std::pin::Pin;
use std::any::TypeId;

use async_trait::async_trait;
use crate::agent::Agent;
use crate::types::context::{RunContext, RunContextWrapper};
use crate::types::message::{Message, Role};
use crate::types::error::Error; 
use crate::tool::Tool;

/// Type for handoff input filter function
pub type HandoffInputFilterFn = Arc<dyn Fn(HandoffInputData) -> Pin<Box<dyn Future<Output = HandoffInputData> + Send>> + Send + Sync>;

/// Type for handoff callback function
pub type HandoffCallbackFn<C> = Arc<dyn Fn(&crate::types::context::RunContextWrapper<C>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Input data for a handoff
#[derive(Clone)]
pub struct HandoffInputData {
    /// History of inputs
    pub input_history: Vec<Message>,
    
    /// Items before the handoff
    pub pre_handoff_items: Vec<crate::types::result::Item>,
    
    /// New items
    pub new_items: Vec<crate::types::result::Item>,
}

/// Optional type for structured handoff input
pub enum HandoffInputType {
    /// JSON schema string
    Schema(String),
    
    /// Pydantic-compatible schema object
    PydanticSchema(serde_json::Value),
}

impl Clone for HandoffInputType {
    fn clone(&self) -> Self {
        match self {
            Self::Schema(s) => Self::Schema(s.clone()),
            Self::PydanticSchema(v) => Self::PydanticSchema(v.clone()),
        }
    }
}

/// A handoff to another agent
pub struct Handoff<C: Clone + Send + Sync + Default + 'static> {
    /// The agent to hand off to
    pub agent: Agent<C>,
    
    /// Optional name override for the handoff tool
    pub tool_name_override: Option<String>,
    
    /// Optional description override for the handoff tool
    pub tool_description_override: Option<String>,
    
    /// Optional callback to run when a handoff is initiated
    pub on_handoff: Option<HandoffCallbackFn<C>>,
    
    /// Optional function to filter the input before handing off
    pub input_filter: Option<HandoffInputFilterFn>,
    
    /// Optional type for structured input to the handoff
    pub input_type: Option<HandoffInputType>,
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for Handoff<C> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            tool_name_override: self.tool_name_override.clone(),
            tool_description_override: self.tool_description_override.clone(),
            on_handoff: self.on_handoff.clone(),
            input_filter: self.input_filter.clone(),
            input_type: self.input_type.clone(),
        }
    }
}

impl<C: Clone + Send + Sync + Default + 'static> Handoff<C> {
    /// Create a new handoff
    pub fn new(agent: Agent<C>) -> Self {
        Self {
            agent,
            tool_name_override: None,
            tool_description_override: None,
            on_handoff: None,
            input_filter: None,
            input_type: None,
        }
    }
    
    /// Get the name of the handoff tool
    pub fn tool_name(&self) -> String {
        self.tool_name_override.clone().unwrap_or_else(|| {
            format!("transfer_to_{}", self.agent.name.to_lowercase().replace(' ', "_"))
        })
    }
    
    /// Get the description of the handoff tool
    pub fn tool_description(&self) -> String {
        self.tool_description_override.clone().unwrap_or_else(|| {
            if let Some(desc) = &self.agent.handoff_description {
                format!("Transfer to {} agent: {}", self.agent.name, desc)
            } else {
                format!("Transfer to {} agent", self.agent.name)
            }
        })
    }
    
    /// Set a custom name for the handoff tool
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name_override = Some(name.into());
        self
    }
    
    /// Set a custom description for the handoff tool
    pub fn with_tool_description(mut self, description: impl Into<String>) -> Self {
        self.tool_description_override = Some(description.into());
        self
    }
    
    /// Set a callback to run when the handoff is initiated
    pub fn with_on_handoff<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(&crate::types::context::RunContextWrapper<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_handoff = Some(Arc::new(move |ctx| {
            let fut = callback(ctx);
            Box::pin(fut) as Pin<Box<dyn Future<Output = ()> + Send>>
        }));
        self
    }
    
    /// Set a filter for the input before handing off
    pub fn with_input_filter<F, Fut>(mut self, filter: F) -> Self
    where
        F: Fn(HandoffInputData) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandoffInputData> + Send + 'static,
    {
        self.input_filter = Some(Arc::new(move |data| {
            let fut = filter(data);
            Box::pin(fut) as Pin<Box<dyn Future<Output = HandoffInputData> + Send>>
        }));
        self
    }
    
    /// Set the input type
    pub fn with_input_type(mut self, input_type: HandoffInputType) -> Self {
        self.input_type = Some(input_type);
        self
    }
    
    /// Convert the handoff to a tool
    pub fn as_tool(&self) -> Box<dyn crate::tool::Tool> {
        Box::new(HandoffTool {
            name: self.tool_name(),
            description: self.tool_description(),
            handoff: self.clone(),
        })
    }
}

/// Create a handoff to another agent
pub fn handoff<C: Clone + Send + Sync + Default + 'static>(agent: Agent<C>) -> Handoff<C> {
    Handoff::new(agent)
}

/// Filter to remove all tools from handoff input
pub async fn remove_all_tools(data: HandoffInputData) -> HandoffInputData {
    // Create a new copy without tool calls
    let filtered_input_history = data.input_history
        .into_iter()
        .map(|mut message| {
            // Create a new message with the same content but no tool calls
            match message.role {
                Role::User => {
                    if let Some(content) = message.content_as_string() {
                        Message::user(content)
                    } else {
                        Message::user("")
                    }
                },
                Role::Assistant => {
                    if let Some(content) = message.content_as_string() {
                        Message::assistant(content)
                    } else {
                        Message::assistant("")
                    }
                },
                Role::System => {
                    if let Some(content) = message.content_as_string() {
                        Message::system(content)
                    } else {
                        Message::system("")
                    }
                },
                Role::Tool => {
                    if let Some(content) = message.content_as_string() {
                        if let Some(name) = message.name.clone() {
                            Message::tool(name, content)
                        } else {
                            Message::tool("unknown", content)
                        }
                    } else {
                        Message::tool("unknown", "")
                    }
                },
            }
        })
        .collect();
    
    HandoffInputData {
        input_history: filtered_input_history,
        pre_handoff_items: data.pre_handoff_items,
        new_items: data.new_items,
    }
}

/// A tool that wraps a handoff
pub struct HandoffTool<C: Clone + Send + Sync + Default + 'static> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// The handoff to execute
    pub handoff: Handoff<C>,
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for HandoffTool<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            handoff: self.handoff.clone(),
        }
    }
}

#[async_trait::async_trait]
impl<C: Clone + Send + Sync + Default + 'static> crate::tool::Tool for HandoffTool<C> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        match &self.handoff.input_type {
            Some(HandoffInputType::Schema(schema)) => {
                serde_json::from_str(schema).unwrap_or_else(|_| self.default_schema())
            },
            Some(HandoffInputType::PydanticSchema(schema)) => {
                schema.clone()
            },
            None => self.default_schema(),
        }
    }
    
    async fn execute(
        &self,
        context: &(dyn std::any::Any + Send + Sync),
        parameters: serde_json::Value
    ) -> crate::types::error::Result<crate::tool::ToolResult> {
        // Try to downcast the context to RunContext<C>
        let run_context = match context.downcast_ref::<RunContext<C>>() {
            Some(ctx) => ctx,
            None => {
                return Err(crate::types::error::Error::Tool(format!(
                    "Context type mismatch for handoff tool {}. Expected RunContext<{}>, got {:?}.",
                    self.name,
                    std::any::type_name::<C>(),
                    context.type_id()
                )));
            }
        };
        
        // Create a context wrapper for callbacks
        let context_wrapper = crate::types::context::RunContextWrapper::new(run_context);
        
        // Extract input from parameters
        let input = if let Some(input) = parameters.get("input").and_then(|v| v.as_str()) {
            input.to_string()
        } else {
            // Try to use the raw parameters as input if not found
            serde_json::to_string(&parameters)
                .map_err(|e| crate::types::error::Error::Tool(format!("Failed to parse input: {}", e)))?
        };
        
        // Create input data for filtering
        let mut input_data = HandoffInputData {
            input_history: vec![Message::user(input.clone())],
            pre_handoff_items: vec![],
            new_items: vec![],
        };
        
        // Apply input filter if provided
        if let Some(filter) = &self.handoff.input_filter {
            input_data = filter(input_data).await;
        }
        
        // Run the on_handoff callback if provided
        if let Some(callback) = &self.handoff.on_handoff {
            callback(&context_wrapper).await;
        }
        
        // Run the agent
        let agent_input = if let Some(content) = input_data.input_history.last().and_then(|m| m.content_as_string()) {
            content
        } else {
            input
        };
        
        // Run the target agent
        let result = crate::runner::Runner::run(
            &self.handoff.agent,
            agent_input.clone(), // Clone the string to own it
            Some(run_context.context.clone()),
            None,
        ).await?;
        
        Ok(crate::tool::ToolResult {
            result: result.final_output,
        })
    }
    
    fn clone_box(&self) -> Box<dyn crate::tool::Tool> {
        Box::new(self.clone())
    }
}

impl<C: Clone + Send + Sync + Default + 'static> HandoffTool<C> {
    /// Create the default schema for handoff input
    fn default_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to send to the target agent"
                }
            },
            "required": ["input"]
        })
    }
}