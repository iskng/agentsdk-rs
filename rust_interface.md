# Rust Interface for Agents SDK

## Overview

The Rust implementation of the Agents SDK will provide similar functionality to the Python SDK while leveraging Rust's strengths: type safety, memory safety, and concurrency. We'll use the `async-openai` crate to interface with the OpenAI API.

## Core Components

### Agent

```rust
/// An agent that can respond to queries, use tools, and hand off to other agents
pub struct Agent<C = ()> {
    /// Name of the agent
    pub name: String,

    /// Instructions for the agent (system prompt)
    pub instructions: String,

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

    /// Type for structured output (using serde)
    pub output_type: Option<OutputType>,

    /// Lifecycle hooks for the agent
    pub hooks: Option<Box<dyn AgentHooks<C>>>,
    
    /// Optional description used when this agent is a handoff target
    pub handoff_description: Option<String>,
    
    /// Phantom data to track context type
    _context: PhantomData<C>,
}

impl<C> Agent<C> {
    /// Create a new agent with the specified parameters
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self { ... }
    
    /// Clone this agent with optional overrides
    pub fn clone_with(&self, overrides: AgentOverrides<C>) -> Self { ... }
    
    /// Convert this agent into a tool that can be used by other agents
    pub fn as_tool(&self, tool_name: impl Into<String>, tool_description: impl Into<String>) -> Box<dyn Tool> { ... }
}
```

### Runner

```rust
/// Runs agents and orchestrates the execution flow
pub struct Runner;

impl Runner {
    /// Run an agent asynchronously
    pub async fn run<C>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>,
    ) -> Result<RunResult, RunnerError> { ... }

    /// Run an agent synchronously (blocking)
    pub fn run_sync<C>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>,
    ) -> Result<RunResult, RunnerError> { ... }
    
    /// Run an agent with streaming output
    pub async fn run_streamed<C>(
        agent: &Agent<C>,
        input: impl Into<RunInput>,
        context: Option<C>,
        config: Option<RunConfig>,
    ) -> Result<StreamedRunResult, RunnerError> { ... }
}
```

### Tools

```rust
/// Trait for tools that agents can use
pub trait Tool: Send + Sync {
    /// Name of the tool
    fn name(&self) -> &str;
    
    /// Description of the tool
    fn description(&self) -> &str;
    
    /// JSON schema for the tool parameters
    fn parameters_schema(&self) -> serde_json::Value;
    
    /// Execute the tool with the given parameters
    async fn execute(
        &self, 
        context: &RunContext<impl Any>, 
        parameters: serde_json::Value
    ) -> Result<ToolResult, ToolError>;
}

/// A tool created from a function
pub struct FunctionTool<F, Args, Ret> {
    name: String,
    description: String,
    schema: serde_json::Value,
    function: F,
    _marker: PhantomData<(Args, Ret)>,
}

/// Create a function tool from a function
pub fn function_tool<F, Args, Ret>(function: F) -> FunctionTool<F, Args, Ret>
where
    F: Fn(Args) -> Ret + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + 'static,
    Ret: Serialize + Send + 'static,
{ ... }

// Create an async function tool from an async function
pub fn async_function_tool<F, Args, Ret, Fut>(function: F) -> AsyncFunctionTool<F, Args, Ret, Fut>
where
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + 'static,
    Ret: Serialize + Send + 'static,
    Fut: Future<Output = Ret> + Send + 'static,
{ ... }
```

### Handoffs

```rust
/// A handoff to another agent
pub struct Handoff<C> {
    /// The agent to hand off to
    pub agent: Agent<C>,
    
    /// Optional name override for the handoff tool
    pub tool_name_override: Option<String>,
    
    /// Optional description override for the handoff tool
    pub tool_description_override: Option<String>,
    
    /// Optional callback to run when a handoff is initiated
    pub on_handoff: Option<Box<dyn Fn(&RunContext<C>) -> BoxFuture<'static, ()> + Send + Sync>>,
    
    /// Optional function to filter the input before handing off
    pub input_filter: Option<Box<dyn Fn(HandoffInputData) -> BoxFuture<'static, HandoffInputData> + Send + Sync>>,
    
    /// Optional type for structured input to the handoff
    pub input_type: Option<HandoffInputType>,
}

/// Create a handoff to another agent
pub fn handoff<C>(agent: Agent<C>) -> Handoff<C> { ... }
```

### Guardrails

```rust
/// A guardrail for validating input
pub struct InputGuardrail<C> {
    /// Function to validate input
    pub guardrail_function: Box<dyn Fn(&RunContext<C>, &Agent<C>, &str) -> BoxFuture<'static, GuardrailResult> + Send + Sync>,
}

/// A guardrail for validating output
pub struct OutputGuardrail<C> {
    /// Function to validate output
    pub guardrail_function: Box<dyn Fn(&RunContext<C>, &Agent<C>, &str) -> BoxFuture<'static, GuardrailResult> + Send + Sync>,
}

/// Decorator to create an input guardrail
pub fn input_guardrail<C, F, Fut>(function: F) -> InputGuardrail<C>
where
    F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailResult> + Send + 'static,
{ ... }

/// Decorator to create an output guardrail
pub fn output_guardrail<C, F, Fut>(function: F) -> OutputGuardrail<C>
where
    F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailResult> + Send + 'static,
{ ... }
```

### Model Providers

```rust
/// Trait for model providers that can be used to execute LLM requests
pub trait ModelProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;
    
    /// Generate a response from the model
    async fn generate(
        &self,
        model: &str,
        messages: Vec<Message>,
        settings: &ModelSettings,
    ) -> Result<ModelResponse, ModelError>;
}

/// Built-in OpenAI model provider
pub struct OpenAIChatCompletions {
    /// Optional API key override
    pub api_key: Option<String>,
    
    /// Optional API base URL override
    pub base_url: Option<String>,
    
    /// Optional organization ID override
    pub organization_id: Option<String>,
}
```

### Tracing

```rust
/// A trace of agent execution
pub struct Trace {
    /// Trace ID
    pub id: String,
    
    /// Name of the workflow
    pub workflow_name: String,
    
    /// Optional group ID for related traces
    pub group_id: Option<String>,
    
    /// Metadata for the trace
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Spans in the trace
    pub spans: Vec<Span>,
}

/// A span within a trace
pub struct Span {
    /// Span ID
    pub id: String,
    
    /// Parent span ID
    pub parent_id: Option<String>,
    
    /// Trace ID
    pub trace_id: String,
    
    /// Start time
    pub started_at: DateTime<Utc>,
    
    /// End time
    pub ended_at: Option<DateTime<Utc>>,
    
    /// Span data
    pub span_data: SpanData,
}

/// Context manager for creating a trace
pub struct TraceContext {
    pub trace: Trace,
}

impl TraceContext {
    /// Create a new trace context
    pub fn new(workflow_name: impl Into<String>) -> Self { ... }
}
```

## Types for JSON Data and Interaction

```rust
/// Input to a run
pub enum RunInput {
    /// A single text message
    Text(String),
    
    /// A list of messages
    Messages(Vec<Message>),
}

/// A message in a conversation
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    
    /// Content of the message
    pub content: String,
    
    /// Optional tool calls in the message
    pub tool_calls: Option<Vec<ToolCall>>,
    
    /// Optional function call in the message (legacy format)
    pub function_call: Option<FunctionCall>,
}

/// Role of a message sender
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Function,
}

/// A tool call in a message
pub struct ToolCall {
    /// ID of the tool call
    pub id: String,
    
    /// Type of the tool call
    pub r#type: String,
    
    /// Function call details
    pub function: FunctionCall,
}

/// A function call
pub struct FunctionCall {
    /// Name of the function
    pub name: String,
    
    /// Arguments to the function
    pub arguments: String,
}

/// Result of running an agent
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

/// Streamed result of running an agent
pub struct StreamedRunResult {
    /// Stream of events
    pub events: Pin<Box<dyn Stream<Item = StreamEvent> + Send>>,
    
    /// Future that resolves to the final result
    pub result: Pin<Box<dyn Future<Output = Result<RunResult, RunnerError>> + Send>>,
}
```

## Using the SDK

### Basic Usage

```rust
use agents_sdk::{Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an agent
    let agent = Agent::new("Assistant", "You only respond in haikus.");

    // Run the agent
    let result = Runner::run(&agent, "Tell me about recursion in programming.", None, None).await?;
    
    // Print the result
    println!("{}", result.final_output);
    
    Ok(())
}
```

### Using Tools

```rust
use agents_sdk::{Agent, Runner, function_tool};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Weather {
    city: String,
    temperature_range: String,
    conditions: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a tool function
    let get_weather = function_tool(|city: String| {
        println!("[debug] get_weather called for {}", city);
        Weather {
            city,
            temperature_range: "14-20C".to_string(),
            conditions: "Sunny with wind.".to_string(),
        }
    });
    
    // Create an agent with the tool
    let agent = Agent::new("Weather Assistant", "You are a helpful weather assistant.")
        .with_tools(vec![get_weather]);
    
    // Run the agent
    let result = Runner::run(&agent, "What's the weather in Tokyo?", None, None).await?;
    
    // Print the result
    println!("{}", result.final_output);
    
    Ok(())
}
```

### Handoffs and Multi-Agent System

```rust
use agents_sdk::{Agent, Runner, handoff};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create specialized agents
    let spanish_agent = Agent::new(
        "Spanish Agent",
        "You only speak Spanish and are extremely concise."
    );
    
    let french_agent = Agent::new(
        "French Agent",
        "You only speak French and are extremely concise."
    );
    
    // Create a triage agent with handoffs
    let triage_agent = Agent::new(
        "Triage Agent",
        "Help the user with their questions. If they speak Spanish, hand off to the Spanish agent. \
         If they speak French, hand off to the French agent."
    ).with_handoffs(vec![
        handoff(spanish_agent),
        handoff(french_agent),
    ]);
    
    // Run the agent
    let result = Runner::run(
        &triage_agent, 
        "Hola, ¿cómo estás?", 
        None, 
        None
    ).await?;
    
    // Print the result
    println!("{}", result.final_output);
    
    Ok(())
}
```

## Implementation Strategy

The Rust implementation will focus on:

1. Strong typing with Rust's type system
2. Async-first design using tokio
3. Ergonomic API that feels natural in Rust
4. Serialization/deserialization with serde
5. Efficient memory management
6. Thread safety for concurrent operation

We'll implement the core components in this order:

1. Basic Agent and Runner with text input/output
2. Function tools
3. OpenAI model provider using async-openai
4. Handoffs
5. Guardrails
6. Tracing
7. Advanced features (streaming, structured outputs)