//! Context types for agent runs

use std::any::Any;
use std::sync::Arc;
use uuid::Uuid;

/// Context for a run
#[derive(Clone)]
pub struct RunContext<C = ()> {
    /// ID of the run
    pub id: Uuid,
    
    /// User-provided context
    pub context: C,
    
    /// Trace ID
    pub trace_id: Option<String>,
    
    /// Parent context
    pub parent: Option<Arc<dyn Any + Send + Sync>>,
}

impl<C: Send + Sync + 'static> RunContext<C> {
    /// Create a new run context
    pub fn new(context: C) -> Self {
        Self {
            id: Uuid::new_v4(),
            context,
            trace_id: None,
            parent: None,
        }
    }
    
    /// Create a new run context with a trace ID
    pub fn with_trace(context: C, trace_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            context,
            trace_id: Some(trace_id.into()),
            parent: None,
        }
    }
    
    /// Create a child context with a new user context
    pub fn child<T: Send + Sync + 'static>(&self, context: T) -> RunContext<T> 
    where C: Clone + 'static
    {
        RunContext {
            id: Uuid::new_v4(),
            context,
            trace_id: self.trace_id.clone(),
            parent: Some(Arc::new(self.clone()) as Arc<dyn Any + Send + Sync>),
        }
    }
}

// Manual implementation replaced by #[derive(Clone)]

/// A wrapper for run context that provides access to the context
#[derive(Clone)]
pub struct RunContextWrapper<C = ()> {
    /// The run context
    pub context: C,
    
    /// The run ID
    pub run_id: Uuid,
    
    /// The trace ID
    pub trace_id: Option<String>,
}

impl<C> RunContextWrapper<C> where C: Clone {
    /// Create a new run context wrapper
    pub fn new(context: &RunContext<C>) -> Self {
        Self {
            context: context.context.clone(),
            run_id: context.id,
            trace_id: context.trace_id.clone(),
        }
    }
    
    /// Create a new run context wrapper from components
    pub fn from_components(context: C, run_id: Uuid, trace_id: Option<String>) -> Self {
        Self {
            context,
            run_id,
            trace_id,
        }
    }
}