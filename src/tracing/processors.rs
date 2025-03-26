//! Trace processors
//!
//! This module provides trace processors for sending traces to various backends.

use async_trait::async_trait;
use std::sync::{Arc, Mutex};

use crate::tracing::trace::Trace;
use crate::tracing::span::Span;

/// Trait for trace processors
#[async_trait]
pub trait TraceProcessor: Send + Sync {
    /// Process a trace
    async fn process_trace(&self, trace: &Trace);
    
    /// Process a span
    async fn process_span(&self, span: &Span);
}

/// A processor that sends traces to the OpenAI backend
pub struct OpenAITraceProcessor {
    /// API key for OpenAI
    pub api_key: String,
    
    /// API base URL
    pub base_url: String,
}

impl OpenAITraceProcessor {
    /// Create a new OpenAI trace processor
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
    
    /// Create a new OpenAI trace processor with a custom base URL
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl TraceProcessor for OpenAITraceProcessor {
    async fn process_trace(&self, _trace: &Trace) {
        // Implementation would send the trace to the OpenAI API
        // This is a placeholder
    }
    
    async fn process_span(&self, _span: &Span) {
        // Implementation would send the span to the OpenAI API
        // This is a placeholder
    }
}

/// A processor that batches traces and spans
pub struct BatchTraceProcessor {
    /// The processor to send to
    processor: Arc<dyn TraceProcessor>,
    
    /// The batch size
    batch_size: usize,
    
    /// The queue of traces
    traces: Arc<Mutex<Vec<Trace>>>,
    
    /// The queue of spans
    spans: Arc<Mutex<Vec<Span>>>,
}

impl BatchTraceProcessor {
    /// Create a new batch trace processor
    pub fn new(processor: Arc<dyn TraceProcessor>, batch_size: usize) -> Self {
        Self {
            processor,
            batch_size,
            traces: Arc::new(Mutex::new(Vec::new())),
            spans: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl TraceProcessor for BatchTraceProcessor {
    async fn process_trace(&self, trace: &Trace) {
        // Create a local copy of the batch to avoid holding the lock during async operations
        let batch = {
            let mut traces = self.traces.lock().unwrap();
            traces.push(trace.clone());
            
            if traces.len() >= self.batch_size {
                std::mem::replace(&mut *traces, Vec::new())
            } else {
                Vec::new()
            }
        };
        
        // Process the batch outside the lock
        for trace in batch {
            self.processor.process_trace(&trace).await;
        }
    }
    
    async fn process_span(&self, span: &Span) {
        // Create a local copy of the batch to avoid holding the lock during async operations
        let batch = {
            let mut spans = self.spans.lock().unwrap();
            spans.push(span.clone());
            
            if spans.len() >= self.batch_size {
                std::mem::replace(&mut *spans, Vec::new())
            } else {
                Vec::new()
            }
        };
        
        // Process the batch outside the lock
        for span in batch {
            self.processor.process_span(&span).await;
        }
    }
}

/// A processor that logs traces and spans to the console
pub struct ConsoleTraceProcessor;

impl ConsoleTraceProcessor {
    /// Create a new console trace processor
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TraceProcessor for ConsoleTraceProcessor {
    async fn process_trace(&self, trace: &Trace) {
        println!("Trace: {} ({})", trace.workflow_name, trace.id);
        println!("  Started at: {}", trace.started_at);
        if let Some(ended_at) = trace.ended_at {
            println!("  Ended at: {}", ended_at);
        }
        if let Some(group_id) = &trace.group_id {
            println!("  Group ID: {}", group_id);
        }
        println!("  Metadata: {:?}", trace.metadata);
    }
    
    async fn process_span(&self, span: &Span) {
        println!("Span: {} ({})", span.id, span.trace_id);
        println!("  Started at: {}", span.started_at);
        if let Some(ended_at) = span.ended_at {
            println!("  Ended at: {}", ended_at);
        }
        if let Some(parent_id) = &span.parent_id {
            println!("  Parent ID: {}", parent_id);
        }
        println!("  Data: {:?}", span.span_data);
    }
}