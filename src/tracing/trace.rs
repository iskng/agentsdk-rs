//! Trace implementation
//!
//! This module provides the Trace struct and related functions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::tracing::span::Span;

/// A trace of agent execution
#[derive(Clone)]
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
    pub spans: Arc<Mutex<Vec<Span>>>,
    
    /// Start time of the trace
    pub started_at: DateTime<Utc>,
    
    /// End time of the trace
    pub ended_at: Option<DateTime<Utc>>,
    
    /// Whether the trace is disabled
    pub disabled: bool,
}

impl Trace {
    /// Create a new trace
    pub fn new(workflow_name: impl Into<String>) -> Self {
        Self {
            id: format!("trace_{}", Uuid::new_v4().to_string().replace("-", "")),
            workflow_name: workflow_name.into(),
            group_id: None,
            metadata: HashMap::new(),
            spans: Arc::new(Mutex::new(Vec::new())),
            started_at: Utc::now(),
            ended_at: None,
            disabled: false,
        }
    }
    
    /// Create a new trace with a specific ID
    pub fn with_id(id: impl Into<String>, workflow_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            workflow_name: workflow_name.into(),
            group_id: None,
            metadata: HashMap::new(),
            spans: Arc::new(Mutex::new(Vec::new())),
            started_at: Utc::now(),
            ended_at: None,
            disabled: false,
        }
    }
    
    /// Set the group ID
    pub fn with_group_id(mut self, group_id: impl Into<String>) -> Self {
        self.group_id = Some(group_id.into());
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Disable the trace
    pub fn disable(mut self) -> Self {
        self.disabled = true;
        self
    }
    
    /// Start the trace
    pub fn start(&mut self) {
        self.started_at = Utc::now();
    }
    
    /// Finish the trace
    pub fn finish(&mut self) {
        self.ended_at = Some(Utc::now());
    }
    
    /// Add a span to the trace
    pub fn add_span(&self, span: Span) {
        if self.disabled {
            return;
        }
        
        let mut spans = self.spans.lock().unwrap();
        spans.push(span);
    }
}

/// Context manager for creating a trace
pub struct TraceContext {
    /// The trace
    pub trace: Trace,
}

impl TraceContext {
    /// Create a new trace context
    pub fn new(workflow_name: impl Into<String>) -> Self {
        Self {
            trace: Trace::new(workflow_name),
        }
    }
}

impl Drop for TraceContext {
    fn drop(&mut self) {
        self.trace.finish();
        // In a real implementation, we'd process the trace here
    }
}

/// Create a new trace
pub fn trace(workflow_name: impl Into<String>) -> TraceContext {
    TraceContext::new(workflow_name)
}