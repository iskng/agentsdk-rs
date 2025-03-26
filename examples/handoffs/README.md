# Handoff Examples

This directory contains examples of how to use handoffs in the Agents SDK.

## Running the Examples

Before running the examples, set your OpenAI API key by either:

1. Creating a `.env` file in the project root with the following content:

```
OPENAI_API_KEY=your_api_key_here
```

OR

2. Setting the environment variable directly:

```bash
export OPENAI_API_KEY=your_api_key
```

Then run the examples using Cargo:

```bash
cargo run --example simple_handoff
cargo run --example message_filter
```

## Available Examples

- **simple_handoff.rs**: Demonstrates a language assistant that routes queries to specialized language agents (Spanish, French, Italian).
- **message_filter.rs**: Shows how to filter message content before handing off to another agent, including removing tool calls and adding disclaimers.

## Key Concepts

### Handoffs

Handoffs allow agents to transfer control to other agents when specialized knowledge or capabilities are needed. This enables building complex, multi-agent systems that can handle a wide variety of tasks.

### Message Filtering

When performing handoffs, you can apply filters to the input history before it's passed to the target agent. This enables:
- Removing sensitive information
- Adding context or disclaimers
- Formatting the input in a specific way for the target agent
- Removing tool calls from previous interactions