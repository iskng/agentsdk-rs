//! Span implementation
//!
//! This module provides the Span struct and related types.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Data associated with a span
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SpanData {
    /// Agent span
    #[serde(rename = "agent")]
    Agent {
        /// Name of the agent
        agent_name: String,
        
        /// Instructions for the agent
        instructions: String,
        
        /// Model used
        model: Option<String>,
    },
    
    /// Generation span
    #[serde(rename = "generation")]
    Generation {
        /// Prompt for the generation
        prompt: Option<String>,
        
        /// Response from the model
        response: Option<String>,
        
        /// Model used
        model: String,
        
        /// Usage statistics
        usage: HashMap<String, u32>,
    },
    
    /// Tool span
    #[serde(rename = "tool")]
    Tool {
        /// Name of the tool
        tool_name: String,
        
        /// Input to the tool
        input: Option<String>,
        
        /// Output from the tool
        output: Option<String>,
    },
    
    /// Handoff span
    #[serde(rename = "handoff")]
    Handoff {
        /// Name of the agent to hand off to
        target_agent: String,
        
        /// Input to the handoff
        input: Option<String>,
        
        /// Output from the handoff
        output: Option<String>,
    },
    
    /// Guardrail span
    #[serde(rename = "guardrail")]
    Guardrail {
        /// Type of guardrail
        guardrail_type: String,
        
        /// Input to the guardrail
        input: Option<String>,
        
        /// Output from the guardrail
        output: Option<String>,
        
        /// Whether the guardrail was triggered
        triggered: bool,
    },
    
    /// Custom span
    #[serde(rename = "custom")]
    Custom {
        /// Name of the custom span
        name: String,
        
        /// Additional data
        data: HashMap<String, serde_json::Value>,
    },
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

impl Span {
    /// Create a new span
    pub fn new(trace_id: impl Into<String>, parent_id: Option<String>, span_data: SpanData) -> Self {
        Self {
            id: format!("span_{}", Uuid::new_v4().to_string().replace("-", "")),
            parent_id,
            trace_id: trace_id.into(),
            started_at: Utc::now(),
            ended_at: None,
            span_data,
        }
    }
    
    /// Start the span
    pub fn start(&mut self) {
        self.started_at = Utc::now();
    }
    
    /// Finish the span
    pub fn finish(&mut self) {
        self.ended_at = Some(Utc::now());
    }
}

impl Clone for Span {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            parent_id: self.parent_id.clone(),
            trace_id: self.trace_id.clone(),
            started_at: self.started_at,
            ended_at: self.ended_at,
            span_data: self.span_data.clone(),
        }
    }
}

/// Create an agent span
pub fn agent_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    agent_name: impl Into<String>,
    instructions: impl Into<String>,
    model: Option<String>,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Agent {
            agent_name: agent_name.into(),
            instructions: instructions.into(),
            model,
        },
    )
}

/// Create a generation span
pub fn generation_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    prompt: Option<String>,
    response: Option<String>,
    model: impl Into<String>,
    usage: HashMap<String, u32>,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Generation {
            prompt,
            response,
            model: model.into(),
            usage,
        },
    )
}

/// Create a tool span
pub fn tool_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    tool_name: impl Into<String>,
    input: Option<String>,
    output: Option<String>,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Tool {
            tool_name: tool_name.into(),
            input,
            output,
        },
    )
}

/// Create a handoff span
pub fn handoff_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    target_agent: impl Into<String>,
    input: Option<String>,
    output: Option<String>,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Handoff {
            target_agent: target_agent.into(),
            input,
            output,
        },
    )
}

/// Create a guardrail span
pub fn guardrail_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    guardrail_type: impl Into<String>,
    input: Option<String>,
    output: Option<String>,
    triggered: bool,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Guardrail {
            guardrail_type: guardrail_type.into(),
            input,
            output,
            triggered,
        },
    )
}

/// Create a custom span
pub fn custom_span(
    trace_id: impl Into<String>,
    parent_id: Option<String>,
    name: impl Into<String>,
    data: HashMap<String, serde_json::Value>,
) -> Span {
    Span::new(
        trace_id,
        parent_id,
        SpanData::Custom {
            name: name.into(),
            data,
        },
    )
}