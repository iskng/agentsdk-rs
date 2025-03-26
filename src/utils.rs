//! Utility functions
//!
//! This module provides utility functions used throughout the SDK.

use uuid::Uuid;
use once_cell::sync::Lazy;
use std::env;

/// Get the OpenAI API key from the environment
pub fn get_openai_api_key() -> Option<String> {
    env::var("OPENAI_API_KEY").ok()
}

/// Get the OpenAI organization ID from the environment
pub fn get_openai_organization_id() -> Option<String> {
    env::var("OPENAI_ORG_ID").ok()
}

/// Get the OpenAI API base URL from the environment
pub fn get_openai_api_base() -> Option<String> {
    env::var("OPENAI_API_BASE").ok()
}

/// Check if tracing is disabled
pub fn is_tracing_disabled() -> bool {
    static TRACING_DISABLED: Lazy<bool> = Lazy::new(|| {
        env::var("OPENAI_AGENTS_DISABLE_TRACING")
            .map(|val| val == "1" || val.to_lowercase() == "true")
            .unwrap_or(false)
    });
    
    *TRACING_DISABLED
}

/// Generate a new trace ID
pub fn generate_trace_id() -> String {
    format!("trace_{}", Uuid::new_v4().to_string().replace("-", ""))
}

/// Generate a new span ID
pub fn generate_span_id() -> String {
    format!("span_{}", Uuid::new_v4().to_string().replace("-", ""))
}

/// Generate a new run ID
pub fn generate_run_id() -> String {
    format!("run_{}", Uuid::new_v4().to_string().replace("-", ""))
}

/// Truncate a string to a maximum length
pub fn truncate_string(s: &str, max_length: usize) -> String {
    if s.len() <= max_length {
        s.to_string()
    } else {
        let mut result = s[..max_length - 3].to_string();
        result.push_str("...");
        result
    }
}

/// Sanitize a string for JSON
pub fn sanitize_for_json(s: &str) -> String {
    s.replace('\u{0000}', "")
}

/// Convert snake_case to CamelCase
pub fn snake_to_camel(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    
    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    
    result
}

/// Convert CamelCase to snake_case
pub fn camel_to_snake(s: &str) -> String {
    let mut result = String::new();
    
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    
    result
}