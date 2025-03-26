# Porting Guide: Python to Rust Agent SDK

This guide covers the main patterns and techniques for porting agents from the Python OpenAI Agents SDK to this Rust implementation.

## Core Concepts

| Python | Rust |
|--------|------|
| `Agent` | `Agent<C>` |
| `Run` | `Runner` |
| `AgentOutputParser` | Built-in to model providers |
| `tool` decorator | `function_tool()` function |
| `handoff` | `handoff()` function |

## Setting Up an Agent

### Python
```python
from openai.agents import Agent

agent = Agent(
    name="My Assistant",
    instructions="You are a helpful AI assistant.",
    model="gpt-4"
)
```

### Rust
```rust
use agents_sdk::{Agent, ModelSettings};

let agent: Agent = Agent::new(
    "My Assistant",
    "You are a helpful AI assistant."
)
.with_model("gpt-4");
```

## Adding Tools

### Python
```python
from openai.agents import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

agent = Agent(
    name="Weather Assistant",
    instructions="You are a weather assistant.",
    tools=[get_weather]
)
```

### Rust
```rust
use agents_sdk::{Agent, function_tool};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Weather {
    location: String,
    condition: String,
}

let get_weather = function_tool(|location: String| {
    Weather {
        location,
        condition: "sunny".to_string(),
    }
});

let agent = Agent::new(
    "Weather Assistant",
    "You are a weather assistant."
)
.with_tool(get_weather);
```

## Running Agents

### Python
```python
from openai.agents import Run

run = Run.create(
    agent=agent,
    instructions="What's the weather in London?",
)
result = run.get_result()
print(result.output.content)
```

### Rust
```rust
use agents_sdk::{Agent, Runner};

let result = Runner::run(
    &agent,
    "What's the weather in London?",
    None,  // Optional context
    None,  // Optional run config
).await?;

println!("{}", result.final_output);
```

## Creating Handoffs

### Python
```python
from openai.agents import Agent, handoff

language_agent = Agent(
    name="Language Expert",
    instructions="You are a language expert."
)

general_agent = Agent(
    name="General Assistant",
    instructions="You are a general assistant.",
    handoffs=[handoff("language", language_agent)]
)
```

### Rust
```rust
use agents_sdk::{Agent, handoff};

let language_agent = Agent::new(
    "Language Expert",
    "You are a language expert."
);

let general_agent = Agent::new(
    "General Assistant",
    "You are a general assistant."
)
.with_handoff(handoff("language", &language_agent));
```

## Adding Guardrails

### Python
```python
from openai.agents import Agent, input_guardrail, output_guardrail

def check_input(input_text):
    if "forbidden" in input_text:
        return False
    return True

def check_output(output_text):
    if "unsafe" in output_text:
        return False
    return True

agent = Agent(
    name="Safe Assistant",
    instructions="You are a safe assistant.",
    input_guardrails=[input_guardrail(check_input)],
    output_guardrails=[output_guardrail(check_output)]
)
```

### Rust
```rust
use agents_sdk::{Agent, input_guardrail, output_guardrail};

let input_guard = input_guardrail(|_ctx, _agent, input| {
    if input.contains("forbidden") {
        return Ok(false);
    }
    Ok(true)
});

let output_guard = output_guardrail(|_ctx, _agent, output| {
    if output.contains("unsafe") {
        return Ok(false);
    }
    Ok(true)
});

let agent = Agent::new(
    "Safe Assistant", 
    "You are a safe assistant."
)
.with_input_guardrail(input_guard)
.with_output_guardrail(output_guard);
```

## Using Context

Rust has a more explicit context pattern using generics:

### Python
```python
# Python context is implicit or passed as kwargs
def get_data(name):
    return f"Data for {name}"

@tool
def fetch_data(name: str) -> str:
    return get_data(name)
```

### Rust
```rust
use agents_sdk::{Agent, context_function_tool};

// Define a context type
#[derive(Clone, Default)]
struct MyContext {
    data_source: String,
}

let fetch_data = context_function_tool(|ctx, name: String| {
    format!("Data for {} from {}", name, ctx.context.data_source)
});

let agent: Agent<MyContext> = Agent::new(
    "Context-aware Assistant",
    "You are a context-aware assistant."
)
.with_tool(fetch_data);

// When running the agent:
let context = MyContext {
    data_source: "main database".to_string(),
};

let result = Runner::run(&agent, "Fetch data for customer123", Some(context), None).await?;
```

## Async Functions

### Python
```python
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    # Async implementation
    return "Data from URL"
```

### Rust
```rust
use agents_sdk::{Agent, async_function_tool};

let fetch_data = async_function_tool(|url: String| async move {
    // Async implementation
    format!("Data from {}", url)
});
```

## Key Differences

1. **Type Safety**: Rust requires explicit types for all parameters and return values.

2. **Context Handling**: Rust uses a generic context parameter `<C>` to pass state to tools and across agent boundaries.

3. **Error Handling**: Rust requires explicit error handling via `Result<T, E>` types.

4. **Concurrency**: Rust uses `async/await` with the Tokio runtime for asynchronous operations.

5. **Memory Management**: Rust's ownership system requires careful handling of references and clones.

6. **Trait System**: Rust uses traits like `Clone`, `Send`, and `Sync` to enable safe concurrency.

## Common Patterns

### Function Tools in Rust
```rust
// Simple function tool
let simple_tool = function_tool(|input: String| {
    format!("Processed: {}", input)
});

// Named function tool with custom name and description
let named_tool = named_function_tool(
    "custom_name",
    "Does something useful",
    |input: String| {
        format!("Processed: {}", input)
    }
);

// Async function tool
let async_tool = async_function_tool(|input: String| async move {
    // async operations
    format!("Async processed: {}", input)
});

// Context-aware function tool
let context_tool = context_function_tool(|ctx, input: String| {
    format!("Context: {}, Input: {}", ctx.context.some_field, input)
});
```

### Working with Complex Return Types
```rust
#[derive(Serialize, Deserialize)]
struct ComplexOutput {
    id: String,
    value: i32,
    details: Vec<String>,
}

let complex_tool = function_tool(|id: String| {
    ComplexOutput {
        id,
        value: 42,
        details: vec!["detail1".to_string(), "detail2".to_string()],
    }
});
```

## Best Practices

1. **Use Strong Types**: Define proper structs for tool inputs and outputs rather than using raw strings.

2. **Handle Errors Gracefully**: Use proper error handling with `Result` types.

3. **Leverage Context**: Use the context parameter to share state between tools and agents.

4. **Implement Clone**: Make sure your context types implement `Clone`, `Send`, `Sync`, and `Default`.

5. **Use Async When Appropriate**: Use async functions for I/O-bound operations.

6. **Keep Tools Simple**: Each tool should do one thing well.

7. **Provide Clear Instructions**: Write clear instructions for your agents to guide them in tool usage.

## Troubleshooting

### Common Issues

1. **"Error deserializing tool arguments"**: Ensure your tool parameter types match what the LLM is sending.

2. **"Tool not found"**: Verify the tool name matches what the agent is trying to call.

3. **Trait bound errors**: Make sure your context type implements all required traits: `Clone + Send + Sync + Default + 'static`.

4. **"Error: MaxTurnsExceeded"**: The agent hit the maximum number of turns. Increase the limit or improve the prompt to require fewer turns.

5. **Ownership issues**: Use references or clones to avoid ownership problems when passing data around.

## Example Conversions

### Example 1: Simple Agent

#### Python
```python
from openai.agents import Agent, Run

agent = Agent(
    name="Math Tutor",
    instructions="You are a helpful math tutor.",
    model="gpt-4"
)

run = Run.create(agent=agent, instructions="What is 2+2?")
result = run.get_result()
print(result.output.content)
```

#### Rust
```rust
use agents_sdk::{Agent, Runner};

let agent = Agent::new(
    "Math Tutor",
    "You are a helpful math tutor."
)
.with_model("gpt-4");

let result = Runner::run(&agent, "What is 2+2?", None, None).await?;
println!("{}", result.final_output);
```

### Example 2: Agent with Tools

#### Python
```python
from openai.agents import Agent, Run, tool

@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

agent = Agent(
    name="Calculator",
    instructions="You are a calculator assistant. Use the calculate tool.",
    tools=[calculate],
    model="gpt-4"
)

run = Run.create(agent=agent, instructions="What is 135 * 29?")
result = run.get_result()
print(result.output.content)
```

#### Rust
```rust
use agents_sdk::{Agent, Runner, function_tool};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expression {
    expression: String,
}

let calculate = function_tool(|input: Expression| {
    // In a real application, use a safe expression evaluator
    // This is just for demonstration
    format!("Result: {}", input.expression)
});

let agent = Agent::new(
    "Calculator",
    "You are a calculator assistant. Use the calculate tool."
)
.with_model("gpt-4")
.with_tool(calculate);

let result = Runner::run(&agent, "What is 135 * 29?", None, None).await?;
println!("{}", result.final_output);
```

## Conclusion

Porting from Python to Rust requires a shift in thinking from dynamic typing to static typing and from implicit reference handling to explicit ownership management. However, the benefits in terms of safety, performance, and maintainability make it worthwhile for production applications.

The Rust Agents SDK closely mirrors the Python API while adding Rust-specific features like strong typing and explicit context handling.