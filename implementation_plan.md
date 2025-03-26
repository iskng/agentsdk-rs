# Rust Agent SDK Implementation Plan

## Project Structure

```
agent_rust/
├── Cargo.toml
├── src/
│   ├── lib.rs             # Main library entry point
│   ├── agent.rs           # Agent implementation
│   ├── runner.rs          # Runner implementation
│   ├── tool.rs            # Tool trait and implementations
│   ├── handoff.rs         # Handoff implementation
│   ├── guardrail.rs       # Guardrail implementation
│   ├── model/
│   │   ├── mod.rs         # Model module entry point
│   │   ├── provider.rs    # ModelProvider trait
│   │   ├── settings.rs    # ModelSettings struct
│   │   └── openai.rs      # OpenAI implementation
│   ├── tracing/
│   │   ├── mod.rs         # Tracing module entry point
│   │   ├── trace.rs       # Trace implementation
│   │   ├── span.rs        # Span implementation
│   │   └── processors.rs  # Trace processors
│   ├── types/
│   │   ├── mod.rs         # Common types module entry point
│   │   ├── message.rs     # Message types
│   │   ├── result.rs      # Result types
│   │   ├── error.rs       # Error types
│   │   └── context.rs     # Context types
│   └── utils.rs           # Utility functions
└── examples/
    ├── basic/
    │   ├── hello_world.rs # Basic agent example
    │   └── tools.rs       # Example with tools
    ├── handoffs/
    │   └── message_filter.rs # Example with handoffs
    └── agent_patterns/
        └── agents_as_tools.rs # Example with agents as tools
```

## Phase 1: Core Components

### 1.1 Project Setup

- Set up Cargo.toml with dependencies
  - async-openai
  - tokio
  - serde/serde_json
  - chrono
  - futures
  - thiserror
  - tracing
  - uuid
  - reqwest

- Create basic library structure
  - lib.rs with module declarations
  - Empty files for each module

### 1.2 Basic Types

- Create message types (message.rs)
  - Message struct
  - Role enum
  - FunctionCall struct
  - ToolCall struct

- Create result types (result.rs)
  - RunResult struct
  - StreamedRunResult struct
  - Usage struct

- Create error types (error.rs)
  - AgentError enum
  - RunnerError enum
  - ToolError enum
  - ModelError enum

- Create context types (context.rs)
  - RunContext struct
  - RunContextWrapper struct

### 1.3 Agent Implementation

- Implement the Agent struct (agent.rs)
  - Constructor
  - Clone method
  - With methods for fluent API (with_tools, with_handoffs, etc.)

- Implement basic model interaction
  - Simple API to send messages to the model
  - Parse responses

### 1.4 Runner Implementation

- Implement the Runner struct (runner.rs)
  - run method
  - run_sync method

- Implement basic run loop
  - Turn user input into model messages
  - Get model response
  - Return final output

## Phase 2: Tool Support

### 2.1 Tool Trait

- Define Tool trait (tool.rs)
  - name method
  - description method
  - parameters_schema method
  - execute method

### 2.2 FunctionTool Implementation

- Implement FunctionTool struct
  - Constructor from a function
  - Parameter schema generation using serde reflection
  - Tool trait implementation

- Create function_tool macro/function
  - Parse function signature
  - Generate schema
  - Create FunctionTool instance

### 2.3 Tool Execution in Runner

- Extend Runner to detect tool calls
  - Parse model response for tool calls
  - Execute tools
  - Return results to model

- Implement the tool loop
  - Continue until final answer
  - Handle errors

## Phase 3: Model Providers

### 3.1 Model Provider Interface

- Define ModelProvider trait (model/provider.rs)
  - generate method
  - name method

- Create ModelSettings struct (model/settings.rs)
  - temperature
  - top_p
  - max_tokens
  - tool_choice
  - etc.

### 3.2 OpenAI Implementation

- Create OpenAIChatCompletions implementation (model/openai.rs)
  - Constructor with API key
  - generate method using async-openai
  - Parse OpenAI responses

- Add support for different models
  - GPT-3.5, GPT-4, etc.
  - Model-specific settings

## Phase 4: Handoffs

### 4.1 Handoff Implementation

- Create Handoff struct (handoff.rs)
  - Agent reference
  - Tool name/description overrides
  - Input filters
  - Callbacks

- Implement handoff function
  - Create Handoff from Agent
  - Apply overrides

### 4.2 Handoff Execution in Runner

- Extend Runner to detect handoffs
  - Parse model response for handoff calls
  - Execute handoff
  - Return control to original agent or end with handoff agent's response

- Implement HandoffInputData
  - Input history
  - Message filtering

## Phase 5: Guardrails

### 5.1 Guardrail Implementation

- Create InputGuardrail struct (guardrail.rs)
  - Constructor
  - Execute method

- Create OutputGuardrail struct
  - Constructor
  - Execute method

### 5.2 Guardrail Integration in Runner

- Extend Runner to use guardrails
  - Execute input guardrails before model call
  - Execute output guardrails after model response
  - Handle guardrail failures

## Phase 6: Tracing

### 6.1 Trace Implementation

- Create Trace struct (tracing/trace.rs)
  - Constructor
  - Start/end methods
  - Add span method

- Create Span struct (tracing/span.rs)
  - Constructor
  - Start/end methods
  - Different span types (agent, tool, generation, etc.)

### 6.2 Trace Processors

- Create TraceProcessor trait (tracing/processors.rs)
  - process_trace method
  - process_span method

- Implement BatchTraceProcessor
  - Queue spans and traces
  - Process in batches

### 6.3 Trace Integration

- Extend Runner to use tracing
  - Create trace for each run
  - Create spans for each step
  - Send to processors

## Phase 7: Advanced Features

### 7.1 Streaming Support

- Implement run_streamed method in Runner
  - Create StreamedRunResult
  - Stream events during execution

- Create event types
  - StreamEvent enum
  - Different event types (text, tool call, etc.)

### 7.2 Structured Outputs

- Add output_type to Agent
  - Type representation
  - JSON schema generation

- Extend Runner to support structured outputs
  - Format prompt for structured output
  - Parse structured response
  - Validate against schema

### 7.3 Dynamic Instructions

- Support function-based instructions in Agent
  - Callback to generate instructions
  - Context access

### 7.4 Lifecycle Hooks

- Create AgentHooks trait
  - Various hook methods
  - Default implementations

- Integrate hooks in Runner
  - Call hooks at appropriate times
  - Pass relevant data

## Phase 8: Testing and Documentation

### 8.1 Unit Tests

- Write tests for each component
  - Agent tests
  - Runner tests
  - Tool tests
  - Handoff tests
  - Guardrail tests
  - Tracing tests

### 8.2 Integration Tests

- Write end-to-end tests
  - Simple conversation
  - Tool usage
  - Handoffs
  - Guardrails
  - Tracing

### 8.3 Documentation

- Write API documentation
  - Doc comments for public items
  - Examples
  - Guides

- Create examples
  - Port Python examples to Rust
  - Add Rust-specific examples

## Phase 9: Performance Optimization

### 9.1 Benchmarking

- Create benchmarks
  - Agent creation
  - Runner execution
  - Tool execution
  - Trace processing

### 9.2 Optimization

- Identify bottlenecks
- Optimize memory usage
- Reduce allocations
- Improve concurrency

## Phase 10: Release Preparation

### 10.1 API Stabilization

- Review public API
- Ensure consistency
- Finalize types and interfaces

### 10.2 Documentation Finalization

- Complete documentation
- Add usage examples
- Write guide for Python SDK users

### 10.3 Packaging

- Prepare for crates.io
- Set up GitHub repository
- Configure CI/CD