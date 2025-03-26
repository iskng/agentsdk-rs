# Agent Rust Examples

This directory contains examples demonstrating the usage of the Agents SDK for Rust.

## Setup

Before running any examples, make sure to:

1. Set up your OpenAI API key in `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

2. The examples use the Tokio runtime for async execution. Make sure you have Tokio as a dependency.

## Basic Examples

### Hello World

```bash
cargo run --example hello_world
```

A simple example that creates a basic agent and sends a query to it.

### Tool Usage

```bash
cargo run --example tools
```

Demonstrates how to create and use tools with agents. This example creates a weather assistant with multiple tools:
- `get_weather`: Get current weather conditions for a city
- `get_forecast`: Get a 5-day forecast for a city
- `get_location`: Get geographic coordinates for a city

The agent uses these tools to answer a query about the weather in Tokyo, providing a comprehensive response by calling multiple tools in sequence.

### Agent Lifecycle

```bash
cargo run --example agent_lifecycle
```

Shows the lifecycle of an agent, including hooks for various events (before/after generation, tool usage, handoffs).

## Handoff Examples

### Simple Handoff

```bash
cargo run --example simple_handoff
```

Demonstrates how one agent can hand off to another specialized agent.

### Message Filter

```bash
cargo run --example message_filter
```

Shows how to filter messages before they're handed off to another agent.

## Agent Patterns

Various patterns for building agent applications:

- **Agents as Tools**: Using agents as tools for other agents
- **Deterministic Behavior**: Creating agents with predictable behavior
- **Forcing Tool Use**: Ensuring agents use specific tools for tasks
- **Input/Output Guardrails**: Validating inputs and outputs
- **LLM as a Judge**: Using an LLM to evaluate outputs
- **Parallelization**: Running multiple agents in parallel
- **Routing**: Routing requests to appropriate specialized agents

## Advanced Examples

Check out the `customer_service`, `financial_research_agent`, and `research_bot` directories for more complex examples of multi-agent systems.

## Troubleshooting

If you encounter issues:

1. Make sure your OpenAI API key is correctly set in the `.env` file
2. Check that you're using a compatible model (GPT-3.5-Turbo or GPT-4 are recommended)
3. Verify that your Rust toolchain is up-to-date (use `rustup update`)
4. If you're having dependency issues, try `cargo clean` followed by `cargo build`