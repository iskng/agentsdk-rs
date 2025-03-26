//! Agent implementation
//!
//! This module provides the Agent struct, which is the core component of the SDK.
//! An agent is a combination of an LLM with instructions, tools, handoffs, and guardrails.

use std::marker::PhantomData;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

use crate::handoff::Handoff;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::model::settings::ModelSettings;
use crate::model::provider::ModelProvider;
use crate::tool::Tool;

/// Dynamic instructions function type
pub type InstructionsFn<C> = Arc<dyn Fn(&crate::types::context::RunContextWrapper<C>, &Agent<C>) -> String + Send + Sync>;

/// Async dynamic instructions function type
pub type AsyncInstructionsFn<C> = Arc<dyn Fn(&crate::types::context::RunContextWrapper<C>, &Agent<C>) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send>> + Send + Sync>;

/// Instructions for an agent
pub enum Instructions<C: Clone + Send + Sync + Default + 'static> {
    /// Static string instructions
    Static(String),
    
    /// Dynamic function instructions
    Dynamic(InstructionsFn<C>),
    
    /// Async dynamic function instructions
    AsyncDynamic(AsyncInstructionsFn<C>),
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for Instructions<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(s) => Self::Static(s.clone()),
            Self::Dynamic(f) => Self::Dynamic(f.clone()),
            Self::AsyncDynamic(f) => Self::AsyncDynamic(f.clone()),
        }
    }
}

/// Represents what happens when a tool is used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolUseBehavior {
    /// Continue processing after tool use
    Continue,
    
    /// Stop after first tool use
    StopOnFirstTool,
}

impl Default for ToolUseBehavior {
    fn default() -> Self {
        Self::Continue
    }
}

/// Definition of what type to use for structured output
pub enum OutputType {
    /// Use a JSON schema string
    Schema(String),
    
    /// Use a Pydantic-compatible JSON schema object
    PydanticSchema(serde_json::Value),
}

/// Agent hooks trait for lifecycle events
pub trait AgentHooks<C: Clone + Send + Sync + Default + 'static>: Send + Sync {
    /// Called when the agent is about to generate
    fn on_generate_start(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>) {}
    
    /// Called when the agent has generated
    fn on_generate_end(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>, _response: &str) {}
    
    /// Called when the agent is about to use a tool
    fn on_tool_start(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>, _tool: &dyn Tool) {}
    
    /// Called when the agent has used a tool
    fn on_tool_end(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>, _tool: &dyn Tool, _result: &str) {}
    
    /// Called when the agent is about to hand off
    fn on_handoff_start(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>, _handoff: &Handoff<C>) {}
    
    /// Called when the agent has handed off
    fn on_handoff_end(&self, _context: &crate::types::context::RunContextWrapper<C>, _agent: &Agent<C>, _handoff: &Handoff<C>, _result: &str) {}
}

/// A no-op implementation of agent hooks
pub struct NoopAgentHooks;

impl<C: Clone + Send + Sync + Default + 'static> AgentHooks<C> for NoopAgentHooks {}

/// Properties for creating a new agent with overrides
pub struct AgentOverrides<C: Clone + Send + Sync + Default + 'static> {
    /// Name of the agent
    pub name: Option<String>,
    
    /// Instructions for the agent
    pub instructions: Option<Instructions<C>>,
    
    /// Model to use
    pub model: Option<String>,
    
    /// Model settings
    pub model_settings: Option<ModelSettings>,
    
    /// Model provider
    pub model_provider: Option<Box<dyn ModelProvider>>,
    
    /// Tools available to the agent
    pub tools: Option<Vec<Box<dyn Tool>>>,
    
    /// Agents that this agent can hand off to
    pub handoffs: Option<Vec<Handoff<C>>>,
    
    /// Input guardrails
    pub input_guardrails: Option<Vec<InputGuardrail<C>>>,
    
    /// Output guardrails
    pub output_guardrails: Option<Vec<OutputGuardrail<C>>>,
    
    /// Output type
    pub output_type: Option<OutputType>,
    
    /// Lifecycle hooks
    pub hooks: Option<Box<dyn AgentHooks<C>>>,
    
    /// Handoff description
    pub handoff_description: Option<String>,
    
    /// Tool use behavior
    pub tool_use_behavior: Option<ToolUseBehavior>,
}

impl<C: Clone + Send + Sync + Default + 'static> Default for AgentOverrides<C> {
    fn default() -> Self {
        Self {
            name: None,
            instructions: None,
            model: None,
            model_settings: None,
            model_provider: None,
            tools: None,
            handoffs: None,
            input_guardrails: None,
            output_guardrails: None,
            output_type: None,
            hooks: None,
            handoff_description: None,
            tool_use_behavior: None,
        }
    }
}

/// An agent that can respond to queries, use tools, and hand off to other agents
pub struct Agent<C = ()> where C: Clone + Send + Sync + Default + 'static {
    /// Name of the agent
    pub name: String,

    /// Instructions for the agent (system prompt)
    pub instructions: Instructions<C>,

    /// Model to use for this agent
    pub model: Option<String>,

    /// Model settings for this agent
    pub model_settings: Option<ModelSettings>,

    /// Optional model provider to use instead of the default
    pub model_provider: Option<Box<dyn ModelProvider>>,

    /// Tools available to the agent
    pub tools: Vec<Box<dyn Tool>>,

    /// Agents that this agent can hand off to
    pub handoffs: Vec<Handoff<C>>,

    /// Input guardrails to validate input before processing
    pub input_guardrails: Vec<InputGuardrail<C>>,

    /// Output guardrails to validate output before returning
    pub output_guardrails: Vec<OutputGuardrail<C>>,

    /// Type for structured output
    pub output_type: Option<OutputType>,

    /// Lifecycle hooks for the agent
    pub hooks: Box<dyn AgentHooks<C>>,
    
    /// Optional description used when this agent is a handoff target
    pub handoff_description: Option<String>,
    
    /// What to do when a tool is used
    pub tool_use_behavior: ToolUseBehavior,
    
    /// Phantom data to track context type
    _context: PhantomData<C>,
}

impl<C: Clone + Send + Sync + Default + 'static> Agent<C> {
    /// Create a new agent with the specified name and instructions
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: Instructions::Static(instructions.into()),
            model: None,
            model_settings: None,
            model_provider: None,
            tools: Vec::new(),
            handoffs: Vec::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_type: None,
            hooks: Box::new(NoopAgentHooks),
            handoff_description: None,
            tool_use_behavior: ToolUseBehavior::default(),
            _context: PhantomData,
        }
    }
    
    /// Create a new agent with dynamic instructions
    pub fn with_dynamic_instructions<F>(name: impl Into<String>, instructions_fn: F) -> Self
    where
        F: Fn(&crate::types::context::RunContextWrapper<C>, &Agent<C>) -> String + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            instructions: Instructions::Dynamic(Arc::new(instructions_fn)),
            model: None,
            model_settings: None,
            model_provider: None,
            tools: Vec::new(),
            handoffs: Vec::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_type: None,
            hooks: Box::new(NoopAgentHooks),
            handoff_description: None,
            tool_use_behavior: ToolUseBehavior::default(),
            _context: PhantomData,
        }
    }
    
    /// Create a new agent with async dynamic instructions
    pub fn with_async_dynamic_instructions<F, Fut>(name: impl Into<String>, instructions_fn: F) -> Self
    where
        F: Fn(&crate::types::context::RunContextWrapper<C>, &Agent<C>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = String> + Send + 'static,
    {
        Self {
            name: name.into(),
            instructions: Instructions::AsyncDynamic(Arc::new(move |ctx, agent| {
                let fut = instructions_fn(ctx, agent);
                Box::pin(fut) as std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send>>
            })),
            model: None,
            model_settings: None,
            model_provider: None,
            tools: Vec::new(),
            handoffs: Vec::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_type: None,
            hooks: Box::new(NoopAgentHooks),
            handoff_description: None,
            tool_use_behavior: ToolUseBehavior::default(),
            _context: PhantomData,
        }
    }
    
    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    
    /// Set the model settings
    pub fn with_model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }
    
    /// Set the model provider
    pub fn with_model_provider(mut self, provider: Box<dyn ModelProvider>) -> Self {
        self.model_provider = Some(provider);
        self
    }
    
    /// Add tools to the agent
    pub fn with_tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
    
    /// Add a single tool to the agent
    pub fn with_tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }
    
    /// Add handoffs to the agent
    pub fn with_handoffs(mut self, handoffs: Vec<Handoff<C>>) -> Self {
        self.handoffs = handoffs;
        self
    }
    
    /// Add a single handoff to the agent
    pub fn with_handoff(mut self, handoff: Handoff<C>) -> Self {
        self.handoffs.push(handoff);
        self
    }
    
    /// Add input guardrails to the agent
    pub fn with_input_guardrails(mut self, guardrails: Vec<InputGuardrail<C>>) -> Self {
        self.input_guardrails = guardrails;
        self
    }
    
    /// Add a single input guardrail to the agent
    pub fn with_input_guardrail(mut self, guardrail: InputGuardrail<C>) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }
    
    /// Add output guardrails to the agent
    pub fn with_output_guardrails(mut self, guardrails: Vec<OutputGuardrail<C>>) -> Self {
        self.output_guardrails = guardrails;
        self
    }
    
    /// Add a single output guardrail to the agent
    pub fn with_output_guardrail(mut self, guardrail: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }
    
    /// Set the output type
    pub fn with_output_type(mut self, output_type: OutputType) -> Self {
        self.output_type = Some(output_type);
        self
    }
    
    /// Set the lifecycle hooks
    pub fn with_hooks(mut self, hooks: Box<dyn AgentHooks<C>>) -> Self {
        self.hooks = hooks;
        self
    }
    
    /// Set the handoff description
    pub fn with_handoff_description(mut self, description: impl Into<String>) -> Self {
        self.handoff_description = Some(description.into());
        self
    }
    
    /// Set the tool use behavior
    pub fn with_tool_use_behavior(mut self, behavior: ToolUseBehavior) -> Self {
        self.tool_use_behavior = behavior;
        self
    }
    
    /// Clone this agent with optional overrides
    pub fn clone_with(&self, overrides: AgentOverrides<C>) -> Self
    where
        C: Clone,
    {
        Self {
            name: overrides.name.unwrap_or_else(|| self.name.clone()),
            instructions: overrides.instructions.unwrap_or_else(|| self.instructions.clone()),
            model: overrides.model.or_else(|| self.model.clone()),
            model_settings: overrides.model_settings.or_else(|| self.model_settings.clone()),
            model_provider: overrides.model_provider.or_else(|| self.model_provider.clone()),
            tools: overrides.tools.unwrap_or_else(|| self.tools.clone()),
            handoffs: overrides.handoffs.unwrap_or_else(|| self.handoffs.clone()),
            input_guardrails: overrides.input_guardrails.unwrap_or_else(|| self.input_guardrails.clone()),
            output_guardrails: overrides.output_guardrails.unwrap_or_else(|| self.output_guardrails.clone()),
            output_type: overrides.output_type.or_else(|| self.output_type.clone()),
            hooks: overrides.hooks.unwrap_or_else(|| Box::new(NoopAgentHooks)),
            handoff_description: overrides.handoff_description.or_else(|| self.handoff_description.clone()),
            tool_use_behavior: overrides.tool_use_behavior.unwrap_or(self.tool_use_behavior),
            _context: PhantomData,
        }
    }
    
    /// Convert this agent into a tool that can be used by other agents
    pub fn as_tool(&self, tool_name: impl Into<String>, tool_description: impl Into<String>) -> Box<dyn Tool>
    where
        C: Clone + Send + Sync + Default + 'static,
    {
        Box::new(crate::tool::AgentTool {
            name: tool_name.into(),
            description: tool_description.into(),
            agent: self.clone_with(AgentOverrides::default()),
        })
    }
}

// Implement Clone for Agent
impl<C: Clone + Send + Sync + Default + 'static> Clone for Agent<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            instructions: self.instructions.clone(),
            model: self.model.clone(),
            model_settings: self.model_settings.clone(),
            model_provider: self.model_provider.clone(),
            tools: self.tools.clone(),
            handoffs: self.handoffs.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            output_type: self.output_type.clone(),
            hooks: Box::new(NoopAgentHooks), // Always use NoopAgentHooks when cloning
            handoff_description: self.handoff_description.clone(),
            tool_use_behavior: self.tool_use_behavior,
            _context: PhantomData,
        }
    }
}

// Implement cloning for specific types
impl Clone for Box<dyn AgentHooks<()>> {
    fn clone(&self) -> Self {
        Box::new(NoopAgentHooks)
    }
}

impl Clone for Box<dyn ModelProvider> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Clone for Box<dyn Tool> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Clone for OutputType {
    fn clone(&self) -> Self {
        match self {
            Self::Schema(s) => Self::Schema(s.clone()),
            Self::PydanticSchema(v) => Self::PydanticSchema(v.clone()),
        }
    }
}