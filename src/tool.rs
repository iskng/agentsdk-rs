//! Tool implementations for agents
//!
//! This module provides the Tool trait and various implementations of it.

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::future::Future;
// No Pin import needed
use std::any::TypeId;

use crate::types::context::RunContext;
use crate::types::error::{Result, Error};
use crate::agent::Agent;

/// Result of executing a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The result of the tool execution
    pub result: String,
}

/// Trait for tools that agents can use
#[async_trait]
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
        context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult>;
    
    /// Execute the tool and return the raw result without wrapping in ToolResult
    /// This is used internally by the runner for better performance
    #[doc(hidden)]
    async fn execute_raw(
        &self,
        context: &(dyn std::any::Any + Send + Sync),
        parameters: serde_json::Value
    ) -> Result<String> {
        // Default implementation just calls execute and extracts the result
        let result = self.execute(context, parameters).await?;
        Ok(result.result)
    }
    
    /// Optional method to check if the tool can handle a given input
    fn can_handle(&self, _input: &str) -> bool {
        true
    }
    
    /// Clone the tool (most tools should implement Clone)
    fn clone_box(&self) -> Box<dyn Tool> {
        unimplemented!("Tool cloning is not implemented for this tool")
    }
}

/// A tool created from a function
pub struct FunctionTool<F, Args, Ret> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// JSON schema for the tool parameters
    pub schema: serde_json::Value,
    
    /// The function to execute
    pub function: Arc<F>,
    
    /// Whether this tool takes a context
    pub takes_ctx: bool,
    
    /// Marker for the argument and return types
    _marker: PhantomData<(Args, Ret)>,
}

impl<F, Args, Ret> Clone for FunctionTool<F, Args, Ret> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: self.function.clone(),
            takes_ctx: self.takes_ctx,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Args, Ret> Tool for FunctionTool<F, Args, Ret>
where
    F: Fn(Args) -> Ret + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    
    async fn execute(
        &self, 
        _context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult> {
        // Deserialize the parameters
        let args: Args = serde_json::from_value(parameters)
            .map_err(|e| Error::Tool(format!("Failed to deserialize parameters: {}", e)))?;
        
        // Execute the function
        let result = (self.function)(args);
        
        // Serialize the result
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Error::Tool(format!("Failed to serialize result: {}", e)))?;
        
        Ok(ToolResult {
            result: result_json,
        })
    }
    
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

/// A tool created from an async function
pub struct AsyncFunctionTool<F, Args, Ret, Fut> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// JSON schema for the tool parameters
    pub schema: serde_json::Value,
    
    /// The function to execute
    pub function: Arc<F>,
    
    /// Whether this tool takes a context
    pub takes_ctx: bool,
    
    /// Marker for the argument, return, and future types
    _marker: PhantomData<(Args, Ret, Fut)>,
}

impl<F, Args, Ret, Fut> Clone for AsyncFunctionTool<F, Args, Ret, Fut> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: self.function.clone(),
            takes_ctx: self.takes_ctx,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Args, Ret, Fut> Tool for AsyncFunctionTool<F, Args, Ret, Fut>
where
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
    Fut: Future<Output = Ret> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    
    async fn execute(
        &self, 
        _context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult> {
        // Deserialize the parameters
        let args: Args = serde_json::from_value(parameters)
            .map_err(|e| Error::Tool(format!("Failed to deserialize parameters: {}", e)))?;
        
        // Execute the function
        let result = (self.function)(args).await;
        
        // Serialize the result
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Error::Tool(format!("Failed to serialize result: {}", e)))?;
        
        Ok(ToolResult {
            result: result_json,
        })
    }
    
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

/// A context-aware function tool
pub struct ContextFunctionTool<F, C, Args, Ret> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// JSON schema for the tool parameters
    pub schema: serde_json::Value,
    
    /// The function to execute
    pub function: Arc<F>,
    
    /// Marker for the context, argument and return types
    _marker: PhantomData<(C, Args, Ret)>,
}

impl<F, C, Args, Ret> Clone for ContextFunctionTool<F, C, Args, Ret> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: self.function.clone(),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<F, C, Args, Ret> Tool for ContextFunctionTool<F, C, Args, Ret>
where
    F: Fn(&crate::types::context::RunContextWrapper<C>, Args) -> Ret + Send + Sync + 'static,
    C: Send + Sync + Clone + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    
    async fn execute(
        &self, 
        context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult> {
        // Try to downcast the context to RunContext<C>
        let run_context = match context.downcast_ref::<RunContext<C>>() {
            Some(ctx) => ctx,
            None => {
                return Err(Error::Tool(format!(
                    "Context type mismatch for tool {}. Expected RunContext<{}>, got {:?}.",
                    self.name,
                    std::any::type_name::<C>(),
                    context.type_id()
                )));
            }
        };
        
        // Deserialize the parameters
        let args: Args = serde_json::from_value(parameters)
            .map_err(|e| Error::Tool(format!("Failed to deserialize parameters: {}", e)))?;
        
        // Create a context wrapper
        let context_wrapper = crate::types::context::RunContextWrapper::new(run_context);
        
        // Execute the function
        let result = (self.function)(&context_wrapper, args);
        
        // Serialize the result
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Error::Tool(format!("Failed to serialize result: {}", e)))?;
        
        Ok(ToolResult {
            result: result_json,
        })
    }
    
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

/// A context-aware async function tool
pub struct ContextAsyncFunctionTool<F, C, Args, Ret, Fut> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// JSON schema for the tool parameters
    pub schema: serde_json::Value,
    
    /// The function to execute
    pub function: Arc<F>,
    
    /// Marker for the context, argument, return, and future types
    _marker: PhantomData<(C, Args, Ret, Fut)>,
}

impl<F, C, Args, Ret, Fut> Clone for ContextAsyncFunctionTool<F, C, Args, Ret, Fut> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: self.function.clone(),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<F, C, Args, Ret, Fut> Tool for ContextAsyncFunctionTool<F, C, Args, Ret, Fut>
where
    F: Fn(&crate::types::context::RunContextWrapper<C>, Args) -> Fut + Send + Sync + 'static,
    C: Send + Sync + Clone + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
    Fut: Future<Output = Ret> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    
    async fn execute(
        &self, 
        context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult> {
        // Try to downcast the context to RunContext<C>
        let run_context = match context.downcast_ref::<RunContext<C>>() {
            Some(ctx) => ctx,
            None => {
                return Err(Error::Tool(format!(
                    "Context type mismatch for tool {}. Expected RunContext<{}>, got {:?}.",
                    self.name,
                    std::any::type_name::<C>(),
                    context.type_id()
                )));
            }
        };
        
        // Deserialize the parameters
        let args: Args = serde_json::from_value(parameters)
            .map_err(|e| Error::Tool(format!("Failed to deserialize parameters: {}", e)))?;
        
        // Create a context wrapper
        let context_wrapper = crate::types::context::RunContextWrapper::new(run_context);
        
        // Execute the function
        let result = (self.function)(&context_wrapper, args).await;
        
        // Serialize the result
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Error::Tool(format!("Failed to serialize result: {}", e)))?;
        
        Ok(ToolResult {
            result: result_json,
        })
    }
    
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

/// Extract the schema for a type
pub fn schema_for_type<T>() -> serde_json::Value 
where
    T: for<'de> Deserialize<'de> + 'static
{
    use serde_json::{Map, Value};
    
    // Try to get the JSON schema for the type
    // This is a simplified implementation
    
    // For primitive types, we can provide simple schemas
    match std::any::TypeId::of::<T>() {
        id if id == TypeId::of::<String>() => {
            return serde_json::json!({
                "type": "string"
            });
        },
        id if id == TypeId::of::<bool>() => {
            return serde_json::json!({
                "type": "boolean"
            });
        },
        id if id == TypeId::of::<i32>() => {
            return serde_json::json!({
                "type": "integer"
            });
        },
        id if id == TypeId::of::<f64>() => {
            return serde_json::json!({
                "type": "number"
            });
        },
        // Add more primitive types if needed
        _ => {}
    }
    
    // For struct types, we create a generic schema
    // In a real implementation, we'd use something like serde_reflection
    let mut properties = Map::new();
    
    // Try to extract field information from the type name (very brittle)
    let type_name = std::any::type_name::<T>();
    
    // If the type name contains fields (e.g., in a struct name), try to parse them
    if type_name.contains('{') {
        let fields_part = type_name.split('{').nth(1).unwrap_or("").split('}').next().unwrap_or("");
        for field in fields_part.split(',') {
            let parts: Vec<&str> = field.trim().split(':').collect();
            if parts.len() == 2 {
                let field_name = parts[0].trim();
                let field_type = parts[1].trim();
                
                let type_value = match field_type {
                    "String" => Value::String("string".to_string()),
                    "bool" => Value::String("boolean".to_string()),
                    "i32" | "i64" | "u32" | "u64" => Value::String("integer".to_string()),
                    "f32" | "f64" => Value::String("number".to_string()),
                    _ => Value::String("object".to_string()),
                };
                
                let mut field_schema = Map::new();
                field_schema.insert("type".to_string(), type_value);
                properties.insert(field_name.to_string(), Value::Object(field_schema));
            }
        }
    }
    
    // Return a schema with the extracted properties
    // This is a fallback for when we can't introspect the type
    if properties.is_empty() {
        serde_json::json!({
            "type": "object",
            "description": format!("Parameters for type {}", type_name),
            "properties": {}
        })
    } else {
        serde_json::json!({
            "type": "object",
            "description": format!("Parameters for type {}", type_name),
            "properties": properties
        })
    }
}

/// Create a function tool from a function
pub fn function_tool<F, Args, Ret>(function: F) -> Box<dyn Tool>
where
    F: Fn(Args) -> Ret + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
{
    // Extract function name using function pointer address as a fallback
    let name = std::any::type_name::<F>()
        .split("::")
        .last()
        .unwrap_or("unknown_function");
    
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(FunctionTool {
        name: name.to_string(),
        description: format!("Tool function {}", name),
        schema,
        function: Arc::new(function),
        takes_ctx: false,
        _marker: PhantomData,
    })
}

/// Create an async function tool from an async function
pub fn async_function_tool<F, Args, Ret, Fut>(function: F) -> Box<dyn Tool>
where
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
    Fut: Future<Output = Ret> + Send + Sync + 'static,
{
    // Extract function name using function pointer address as a fallback
    let name = std::any::type_name::<F>()
        .split("::")
        .last()
        .unwrap_or("unknown_function");
    
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(AsyncFunctionTool {
        name: name.to_string(),
        description: format!("Async tool function {}", name),
        schema,
        function: Arc::new(function),
        takes_ctx: false,
        _marker: PhantomData,
    })
}

/// Create a context-aware function tool
pub fn context_function_tool<F, C, Args, Ret>(function: F) -> Box<dyn Tool>
where
    F: Fn(&crate::types::context::RunContextWrapper<C>, Args) -> Ret + Send + Sync + 'static,
    C: Send + Sync + Clone + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
{
    // Extract function name using function pointer address as a fallback
    let name = std::any::type_name::<F>()
        .split("::")
        .last()
        .unwrap_or("unknown_function");
    
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(ContextFunctionTool {
        name: name.to_string(),
        description: format!("Context-aware tool function {}", name),
        schema,
        function: Arc::new(function),
        _marker: PhantomData,
    })
}

/// Create a context-aware async function tool
pub fn context_async_function_tool<F, C, Args, Ret, Fut>(function: F) -> Box<dyn Tool>
where
    F: Fn(&crate::types::context::RunContextWrapper<C>, Args) -> Fut + Send + Sync + 'static,
    C: Send + Sync + Clone + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
    Fut: Future<Output = Ret> + Send + Sync + 'static,
{
    // Extract function name using function pointer address as a fallback
    let name = std::any::type_name::<F>()
        .split("::")
        .last()
        .unwrap_or("unknown_function");
    
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(ContextAsyncFunctionTool {
        name: name.to_string(),
        description: format!("Context-aware async tool function {}", name),
        schema,
        function: Arc::new(function),
        _marker: PhantomData,
    })
}

/// A tool that uses another agent
pub struct AgentTool<C: Clone + Send + Sync + Default + 'static> {
    /// Name of the tool
    pub name: String,
    
    /// Description of the tool
    pub description: String,
    
    /// Agent to use
    pub agent: Agent<C>,
}

#[async_trait]
impl<C: Send + Sync + Clone + Default + 'static> Tool for AgentTool<C> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to send to the agent"
                }
            },
            "required": ["input"]
        })
    }
    
    async fn execute(
        &self, 
        context: &(dyn std::any::Any + Send + Sync), 
        parameters: serde_json::Value
    ) -> Result<ToolResult> {
        // Extract the input parameter
        let input = parameters.get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing 'input' parameter".to_string()))?;
        
        // Try to downcast the context to RunContext<C>
        let run_context = match context.downcast_ref::<RunContext<C>>() {
            Some(ctx) => ctx,
            None => {
                return Err(Error::Tool(format!(
                    "Context type mismatch for agent tool {}. Expected RunContext<{}>, got {:?}.",
                    self.name,
                    std::any::type_name::<C>(),
                    context.type_id()
                )));
            }
        };
        
        // Run the agent
        // We'll use the Runner directly
        let result = crate::runner::Runner::run(
            &self.agent,
            input,
            Some(run_context.context.clone()),
            None,
        ).await?;
        
        Ok(ToolResult {
            result: result.final_output,
        })
    }
    
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(Self {
            name: self.name.clone(),
            description: self.description.clone(),
            agent: self.agent.clone(),
        })
    }
}

impl<C: Clone + Send + Sync + Default + 'static> Clone for AgentTool<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            agent: self.agent.clone(),
        }
    }
}

/// Create a tool with a custom name and description
pub fn named_function_tool<F, Args, Ret>(
    name: impl Into<String>,
    description: impl Into<String>,
    function: F
) -> Box<dyn Tool>
where
    F: Fn(Args) -> Ret + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
{
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(FunctionTool {
        name: name.into(),
        description: description.into(),
        schema,
        function: Arc::new(function),
        takes_ctx: false,
        _marker: PhantomData,
    })
}

/// Create an async tool with a custom name and description
pub fn named_async_function_tool<F, Args, Ret, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    function: F
) -> Box<dyn Tool>
where
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + Send + Sync + 'static,
    Ret: Serialize + Send + Sync + 'static,
    Fut: Future<Output = Ret> + Send + Sync + 'static,
{
    // Generate schema from type
    let schema = schema_for_type::<Args>();
    
    Box::new(AsyncFunctionTool {
        name: name.into(),
        description: description.into(),
        schema,
        function: Arc::new(function),
        takes_ctx: false,
        _marker: PhantomData,
    })
}