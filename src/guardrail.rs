//! Guardrail implementation
//!
//! This module provides the guardrail types and functions for validating input and output.

use std::sync::Arc;
use std::future::Future;
use std::pin::Pin;

use serde::{Serialize, Deserialize};
use crate::agent::Agent;
use crate::types::context::RunContext;

/// Result of a guardrail function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailFunctionOutput {
    /// Additional output information
    pub output_info: serde_json::Value,
    
    /// Whether the guardrail was triggered
    pub tripwire_triggered: bool,
}

/// Type for guardrail function
pub type GuardrailFn<C: Clone + Send + Sync + Default + 'static> = Arc<dyn Fn(&RunContext<C>, &Agent<C>, &str) -> Pin<Box<dyn Future<Output = GuardrailFunctionOutput> + Send>> + Send + Sync>;

/// A guardrail for validating input
pub struct InputGuardrail<C: Clone + Send + Sync + Default + 'static> {
    /// Function to validate input
    pub guardrail_function: GuardrailFn<C>,
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for InputGuardrail<C> {
    fn clone(&self) -> Self {
        Self {
            guardrail_function: self.guardrail_function.clone(),
        }
    }
}

impl<C: Clone + Send + Sync + Default + 'static> InputGuardrail<C> {
    /// Create a new input guardrail
    pub fn new<F, Fut>(function: F) -> Self
    where
        F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
    {
        Self {
            guardrail_function: Arc::new(move |ctx, agent, input| {
                let fut = function(ctx, agent, input);
                Box::pin(fut) as Pin<Box<dyn Future<Output = GuardrailFunctionOutput> + Send>>
            }),
        }
    }
    
    /// Run the guardrail
    pub async fn run(
        &self,
        ctx: &RunContext<C>,
        agent: &Agent<C>,
        input: &str,
    ) -> GuardrailFunctionOutput {
        (self.guardrail_function)(ctx, agent, input).await
    }
}

/// A guardrail for validating output
pub struct OutputGuardrail<C: Clone + Send + Sync + Default + 'static> {
    /// Function to validate output
    pub guardrail_function: GuardrailFn<C>,
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for OutputGuardrail<C> {
    fn clone(&self) -> Self {
        Self {
            guardrail_function: self.guardrail_function.clone(),
        }
    }
}

impl<C: Clone + Send + Sync + Default + 'static> OutputGuardrail<C> {
    /// Create a new output guardrail
    pub fn new<F, Fut>(function: F) -> Self
    where
        F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
    {
        Self {
            guardrail_function: Arc::new(move |ctx, agent, output| {
                let fut = function(ctx, agent, output);
                Box::pin(fut) as Pin<Box<dyn Future<Output = GuardrailFunctionOutput> + Send>>
            }),
        }
    }
    
    /// Run the guardrail
    pub async fn run(
        &self,
        ctx: &RunContext<C>,
        agent: &Agent<C>,
        output: &str,
    ) -> GuardrailFunctionOutput {
        (self.guardrail_function)(ctx, agent, output).await
    }
}

/// Decorator to create an input guardrail
pub fn input_guardrail<C: Clone + Send + Sync + Default + 'static, F, Fut>(function: F) -> InputGuardrail<C>
where
    F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
{
    InputGuardrail::new(function)
}

/// Decorator to create an output guardrail
pub fn output_guardrail<C: Clone + Send + Sync + Default + 'static, F, Fut>(function: F) -> OutputGuardrail<C>
where
    F: Fn(&RunContext<C>, &Agent<C>, &str) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
{
    OutputGuardrail::new(function)
}