# OpenAI Agents SDK for Rust

A Rust implementation of the OpenAI Agents SDK, providing a type-safe, memory-safe, and concurrent framework for building agentic AI applications.

## Features

- ✅ Create agents with custom instructions
- ✅ Equip agents with tools they can use
- ✅ Support for handoffs between specialized agents
- ✅ Input and output guardrails for safety and validation
- ✅ Tracing support for debugging
- ✅ Streaming responses (coming soon)
- ✅ Minimal abstractions with maximum type safety
- ✅ Compatible with OpenAI API for chat completions
- ✅ Support for multiple model providers

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agents-sdk = "0.1.0"
```

## Quick Start

```rust
use agents_sdk::{Agent, Runner};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Create a simple agent
    let agent = Agent::new(
        "Assistant",
        "You are a helpful assistant that provides brief concise answers."
    );

    // Run the agent
    let result = Runner::run(&agent, "Hello! Tell me about Rust programming language.", None, None).await?;
    
    // Print the result
    println!("{}", result.final_output);
    
    Ok(())
}
```

## Creating Tools

Tools allow agents to perform actions and access external systems.

```rust
use agents_sdk::{Agent, Runner, function_tool};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Weather {
    city: String,
    temperature: String,
    conditions: String,
}

let get_weather = function_tool(|city: String| {
    Weather {
        city: city.clone(),
        temperature: "22°C".to_string(),
        conditions: "Sunny".to_string(),
    }
});

let agent = Agent::new(
    "Weather Assistant",
    "You are a helpful weather assistant. Use the get_weather tool when asked about weather."
).with_tool(get_weather);
```

## Setting Up Your Environment

1. Create a `.env` file in your project root:

```
OPENAI_API_KEY=your_api_key_here
```

2. Make sure your project loads the environment variables:

```rust
use dotenv::dotenv;

fn main() {
    dotenv().ok(); // Load environment variables from .env file
    // Your code here
}
```

## Advanced Features

### Handoffs

Allow agents to delegate tasks to specialized agents:

```rust
use agents_sdk::{Agent, handoff};

let language_agent = Agent::new(
    "Language Expert",
    "You are an expert in linguistics. Explain language concepts clearly and concisely."
);

let math_agent = Agent::new(
    "Math Expert",
    "You are an expert in mathematics. Explain math concepts clearly and concisely."
);

let general_agent = Agent::new(
    "General Assistant",
    "You are a general assistant. For language questions, use the language_expert. For math questions, use the math_expert."
)
.with_handoff(handoff("language_expert", &language_agent))
.with_handoff(handoff("math_expert", &math_agent));
```

### Guardrails

Add safeguards to filter inputs and outputs:

```rust
use agents_sdk::{Agent, input_guardrail, output_guardrail};

let input_guard = input_guardrail(|_ctx, _agent, input| {
    if input.contains("forbidden") {
        return Ok(false);  // Reject the input
    }
    Ok(true)  // Accept the input
});

let output_guard = output_guardrail(|_ctx, _agent, output| {
    if output.contains("unsafe") {
        return Ok(false);  // Reject the output
    }
    Ok(true)  // Accept the output
});

let agent = Agent::new(
    "Safe Assistant", 
    "You provide helpful, safe responses."
)
.with_input_guardrail(input_guard)
.with_output_guardrail(output_guard);
```

## Examples

Check out the `examples/` directory for more detailed examples:

- `examples/basic/hello_world.rs` - A simple hello world agent
- `examples/basic/tools.rs` - Using tools with agents
- `examples/handoffs/simple_handoff.rs` - Agent handoffs
- And more!

## Project Status

This project is under active development and currently in beta. The API may change in future releases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.