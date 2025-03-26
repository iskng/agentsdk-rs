//! # Agents SDK
//!
//! A Rust implementation of the OpenAI Agents SDK for building agentic AI applications.
//!
//! The Agents SDK enables you to build agentic AI applications in a lightweight, easy-to-use
//! package with very few abstractions. It provides a structured way to create, configure,
//! and orchestrate LLM-powered agents with tools, guardrails, handoffs, and tracing capabilities.
//!
//! ## Core Components
//!
//! - **Agents**: LLMs equipped with instructions and tools
//! - **Tools**: Actions that agents can perform
//! - **Handoffs**: Allow agents to delegate to other agents
//! - **Guardrails**: Enable input and output validation
//! - **Tracing**: Visualize and debug agent execution
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use agents_sdk::{Agent, Runner};
//! use std::error::Error;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn Error>> {
//!     // Create an agent
//!     let agent = Agent::new("Assistant", "You only respond in haikus.");
//!
//!     // Run the agent
//!     let result = Runner::run(&agent, "Tell me about recursion in programming.", None, None).await?;
//!     
//!     // Print the result
//!     println!("{}", result.final_output);
//!     
//!     Ok(())
//! }
//! ```

// Main modules
pub mod agent;
pub mod runner;
pub mod tool;
pub mod handoff;
pub mod guardrail;
pub mod model;
pub mod tracing;
pub mod types;
pub mod utils;

// Re-exports for convenience
pub use agent::Agent;
pub use runner::{Runner, RunConfig, RunInput};
pub use tool::{Tool, FunctionTool, function_tool, async_function_tool, named_function_tool, named_async_function_tool, context_function_tool, context_async_function_tool};
pub use handoff::{Handoff, handoff, remove_all_tools};
pub use guardrail::{InputGuardrail, OutputGuardrail, input_guardrail, output_guardrail};
pub use model::settings::{ModelSettings, ToolChoice, ResponseFormat};
pub use model::mock::MockModelProvider;
pub use types::{
    message::{Message, Role},
    result::{RunResult, StreamedRunResult, Item},
    context::{RunContext, RunContextWrapper},
    error::{Error, Result},
};

// Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");