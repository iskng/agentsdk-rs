Great — I’ll put together a complete Python-based guide to using the OpenAI Agents SDK, written for an LLM that hasn't seen the code before. This will include setup, architecture, core concepts, and examples from the GitHub resources you shared. I’ll let you know as soon as it’s ready.

# OpenAI Agents SDK Developer Guide

## Overview of the OpenAI Agents SDK

The **OpenAI Agents SDK** is a Python framework for building *agentic* AI applications – programs where a large language model (LLM) can make decisions and take actions autonomously ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=OpenAI%20Agents%20SDK)). It provides a lightweight architecture with minimal abstractions, focusing on letting developers orchestrate complex workflows of one or multiple agents. Typical use cases include conversational assistants that use tools (e.g. web search or APIs), multi-step research or data analysis agents, customer support bots, code assistants, and more ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=The%20Agents%20SDK%20is%20suitable,architecture%20simplified%20the%20process%20of)). 

At a high level, an *agent* in this SDK is an LLM configured with specific behavior and access to tools. The SDK handles the loop of the agent reasoning, using tools, and producing a final answer. Key features of the Agents SDK include: 

- **Agents** – easy-to-configure LLMs with clear instructions and optional built-in tools ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=Improvements%20include%3A)). An agent can be given a role or goal via a system prompt and a set of tools it can use.
- **Tool Use** – a mechanism for the agent to invoke external functions or APIs (called *tools*) during its reasoning process. The agent’s tool calls (actions) are automatically executed, and the results are fed back to the agent for further reasoning.
- **Handoffs** – the ability for an agent to delegate to other sub-agents. This enables multi-agent workflows where one agent can pass control to a more specialized agent (for example, a triage agent handing off to a language-specific agent) ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=1,debug%20and%20optimize%20your%20workflows)).
- **Guardrails** – configurable input/output validators for safety and validity checks. You can define guardrail rules to intercept or modify the agent’s inputs or outputs to enforce policies.
- **Tracing & Observability** – built-in tracing of agent runs to help debug and optimize. Every step (LLM call, tool usage, handoff) can be logged, and the SDK can integrate with monitoring tools to visualize the agent’s decision steps ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=,to%20debug%20and%20optimize%20performance)) ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=,to%20debug%20and%20optimize%20performance)).

 ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python)) *Example: The Agents SDK includes tracing tools to visualize an agent's execution steps (here showing a multi-agent workflow with tool calls and handoffs) for debugging and analysis.*

In summary, the Agents SDK provides the “brain” (LLM agent) and the “hands” (tools/actions) for building AI systems that go beyond simple question-answering. It orchestrates the flow of **messages → model reasoning → actions → observations** in a loop until the task is complete.

## Installation Instructions

Setting up the OpenAI Agents SDK is straightforward. First, ensure you have a compatible Python environment (Python 3.8+). It’s recommended to use a virtual environment for isolation:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate
```

Next, install the SDK via pip:

```bash
pip install openai-agents
```

This will download the `openai-agents` package and its dependencies. After installation, you should be able to import the SDK in Python. If you plan to use **voice features** (for speech recognition or text-to-speech with agents), install the extra dependencies with:

```bash
pip install "openai-agents[voice]"
``` 

Before running any agent, you need to set your OpenAI API key so the agent can access the LLM. The SDK uses the OpenAI API under the hood (by default) for model calls. Set the `OPENAI_API_KEY` environment variable with your API key:

```bash
export OPENAI_API_KEY="sk-YourApiKeyHere"
```

*Note:* If you’re in a Jupyter or other environment, you can alternatively set the API key in code (e.g. via `openai.api_key = "..."`) or configure a `.env` file. With installation complete and your API credentials set, you’re ready to develop agents using the SDK. 

## Core Concepts and Components

The Agents SDK introduces a few core concepts that you will use to build agent applications. Understanding these components – **Agent**, **Tool**, **Action**, **Environment/Context** – and how they interact is essential.

### Agent

An **Agent** is the central object representing your AI assistant or worker. In this SDK, an agent is essentially an LLM with a specific configuration. You supply: 

- **Instructions**: a description of the agent’s role or behavior (similar to a system prompt). This guides the LLM on how to respond (e.g. “You are a helpful travel assistant.”).
- **Tools**: a list of functions or actions the agent can use to help fulfill its task (e.g. a web search tool, calculator function, database query tool).
- **Model and Settings**: which underlying LLM to use (e.g. `gpt-4` or `gpt-3.5-turbo`), along with any model parameters like temperature or top_p (these can be set via a `model_settings` object).
- **Handoffs (optional)**: other agents that this agent can defer to. For example, a high-level agent might hand off to a specialized agent for a sub-problem.
- **Guardrails (optional)**: validation rules for inputs or outputs.

In code, you create an agent by instantiating the `Agent` class. For example, a simple agent with just a name and instructions can be created as: 

```python
from agents import Agent
agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
``` 

This agent will use the default model (if not specified) and has no tools – it behaves like a plain chatbot following the given persona. Agents are the **primary actors** in the SDK: they receive user inputs, decide on actions (like calling a tool or giving an answer), and produce outputs. Under the hood, an agent’s instructions are used as the system message for the LLM, and the SDK keeps track of the agent’s state (context, conversation history, etc.) ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=,tools%2C%20guardrails%2C%20handoffs%20and%20more)). Typically you will configure an agent with at least one tool to make it more capable.

**Context:** Each agent can be parameterized by a context object (sometimes thought of as the agent’s environment or state). The context can be any Python object that holds data or services the agent might need (e.g. a database connection, user profile info, etc.). You can define a dataclass for your context and pass an instance when running the agent. The context object will be available to all tools and agent logic during the run ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=Agents%20are%20generic%20on%20their,Python%20object%20as%20the%20context)). Using context is an advanced feature for dependency injection – it allows your tools or guardrails to access shared state (for example, caching data across tool calls, or storing results).

### Tool

A **Tool** is an action that the agent can take to assist in completing its task. In practice, a tool is usually a Python function (or an API call, etc.) that’s been registered with the agent. For example, tools might include things like searching the web, looking up a file, doing math, or calling an external API. The SDK makes it easy to define tools – you can turn any Python function into a tool using a decorator or helper. 

When an agent has tools, the LLM is made aware of them (the SDK supplies the function names and descriptions to the model). The model can then decide to “call” a tool by name with certain arguments instead of directly answering. The SDK will catch that request, execute the corresponding Python function, and return the result back to the model. This mechanism uses OpenAI’s function calling under the hood, so the LLM’s decision to use a tool appears as a function-call response.

There are built-in tool classes for common functionalities (e.g. `WebSearchTool` for internet search, `FileSearchTool` for vector database queries, `ComputerTool` for executing safe shell commands) ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=class%20FileSearchTool%3A)) ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=class%20WebSearchTool%3A)). You can also create **custom tools** easily:

- **Function tools**: The simplest way to create a custom tool is by writing a Python function and marking it with the `@function_tool` decorator. The SDK will automatically generate a JSON schema for the function’s arguments and make it available to the agent ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=,helpers%20to)). For example, a function `def add(a: int, b: int) -> int` decorated with `@function_tool` would allow the agent to call `add` with two numbers.
- **Tool classes**: For advanced use cases, you can create a tool by instantiating or subclassing tool classes (like `FunctionTool`, or writing a custom `BaseTool`). However, in most cases the decorator approach is sufficient and easier.

Each tool has a **name** and **description** that the LLM sees. The name is how the agent refers to the tool in its thoughts, and the description helps the model decide when the tool is relevant. The output of a tool call will be fed back to the agent as part of the next input (the observation). The SDK handles converting the tool’s Python return value into a string or JSON that the model can read. Essentially, tools let the agent extend its capabilities beyond what the base LLM can do on its own ([Building AI Agents with OpenAI's SDK: A Beginner's Guide - Medium](https://medium.com/@Micheal-Lanham/building-ai-agents-with-openais-sdk-a-beginner-s-guide-1debab18e0eb#:~:text=Medium%20medium,function%20into%20a%20tool)).

### Action and Observation

When an agent uses a tool, it’s performing an **action**. In the typical agent loop, the agent will alternate between reasoning (LLM thinking) and taking actions until it arrives at an answer. In this thought–action–observation cycle:

1. **Thought (LLM reasoning)**: The agent considers the user input and its instructions. It may produce either an answer *or* decide it needs to use a tool.
2. **Action (Tool call)**: If the agent decides to use a tool, the LLM’s output will indicate a function call (e.g. calling a tool by name with certain parameters). This is an *action* – the agent is effectively saying "I will use Tool X with these inputs."
3. **Observation (Tool result)**: The SDK executes the tool action in the environment (e.g. calls the Python function for that tool) and obtains a result. This result is then returned to the agent. The agent receives this as new information – an observation of what happened after the action.

The agent’s loop then continues: the observation is added to the agent’s context or conversation, and the LLM gets to respond again (now with the benefit of that new information). This loop repeats until the agent produces a final answer instead of another action ([Running agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/running_agents/#:~:text=1,run%20the%20loop)) ([Running agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/running_agents/#:~:text=Note)). 

To illustrate, suppose a user asks: “What’s the weather in Tokyo tomorrow?” The agent might think internally: *I have a tool for weather, I should use it.* The agent (LLM) outputs an **action**: a function call `get_weather(city="Tokyo")`. The SDK sees this and calls the actual `get_weather` Python function. The function returns, say, “Tokyo will be 15°C and sunny.” This is the **observation**. The observation is provided back to the LLM (e.g. as if the assistant said: `The weather tool says: "Tokyo will be 15°C and sunny."`). Now the agent has that info and can respond to the user with the final answer: “It’s going to be sunny and around 15°C in Tokyo.” In the conversation, the tool usage is hidden from the user – they only see the final answer, while the tool call and result happened behind the scenes.

In the Agents SDK, you usually don’t have to manually manage these steps; the `Runner` (discussed below) orchestrates this loop automatically. But it’s important to understand that **actions** correspond to tool invocations or handoffs, and **observations** are the outcomes of those actions that inform subsequent reasoning ([OpenAI Agents SDK -II - Medium](https://medium.com/@danushidk507/openai-agents-sdk-ii-15a11d48e718#:~:text=Tools%3A%20These%20are%20functions%20the,the%20loop%20for%20further%20processing)). 

### Environment (and Context)

In an AI agent scenario, the **environment** typically means everything the agent can interact with or the external world it operates in. In this SDK, the environment is represented in two ways: 

- The set of **tools and other agents** available (this defines what “actions” the agent can take in the world).
- The **context object** passed to the agent run (this provides any external state or services).

When you run an agent, you can supply a context (`Runner.run(agent, input, context=my_context)`). This context can hold environment-specific data (for example, the user’s location if that should influence a weather tool, or a database connection if the agent needs to fetch data). The context is passed into every tool call, guardrail, and handoff, so it effectively allows your agent to **carry an environment with it** through the run ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=Agents%20are%20generic%20on%20their,Python%20object%20as%20the%20context)). For instance, if you have a tool that needs access to a database or a cache, you can store that connection in the context instead of using global variables.

The environment also includes any external systems the agent interacts with. For example, if the agent uses a web search tool, the “world” it interacts with is the internet (via that tool). If it uses a file search tool, the environment includes the vector database that tool searches. As a developer, you set up the environment by enabling the appropriate tools or agents and providing necessary configuration (API keys for external services, file indexes, etc.). 

In multi-agent setups, one agent’s environment may include *other agents* (through handoffs). The SDK treats handoffs as a special kind of tool where the “tool” is actually another agent to delegate to ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=1,debug%20and%20optimize%20your%20workflows)). In such cases, the environment is a network of agents – the orchestration ensures control is passed along and returned appropriately.

To summarize: **Agent = LLM brain; Tools = means to act on environment; Context = additional environment state.** The interplay of these components is managed by the Agents SDK so that you can focus on defining the agent’s abilities and instructions, rather than writing the loop logic yourself.

## Code Structure Walkthrough

The OpenAI Agents SDK is organized into a Python package with clear separation of concerns. If you look at the repository structure, you’ll find the main implementation under the `src/agents` directory. Some notable modules and their purposes include:

- **`agent.py`** – Defines the `Agent` class and related logic. This is where the agent’s configuration (instructions, tools, model, guardrails, handoffs, etc.) is implemented. The `Agent` class is a dataclass that holds these properties ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=name%3A%20str)) ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=tools%3A%20list)) and includes methods to assist in running or copying agents.
- **`tool.py`** – Contains classes and utilities for tools. It defines the `FunctionTool` dataclass (which wraps a Python function as a tool) and built-in tool classes like `WebSearchTool`, `FileSearchTool`, `ComputerTool` ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=class%20FileSearchTool%3A)) ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=class%20WebSearchTool%3A)). It also provides the `function_tool` decorator that you use to easily create tools from functions. If you need to implement custom tool behaviors, this module is where to look.
- **`run.py`** and **`_run_impl.py`** – Implement the logic for running the agent loop. The `Runner` class (often used via `Runner.run()` or `Runner.run_sync()`) is responsible for taking an Agent and an input and executing the reasoning loop until completion. It handles calling the LLM, detecting tool calls or handoffs in the LLM’s output, executing them, and feeding results back in, as well as stopping conditions (like reaching a final answer or hitting a max turn limit).
- **`agent_output.py`** – Defines how the agent’s final output is represented (especially if you use structured outputs). For example, if you specify an `output_type` for the agent (a Pydantic model for the final answer), this module helps validate and construct that output ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=Output%20types)) ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=agent%20%3D%20Agent%28%20name%3D,output_type%3DCalendarEvent%2C)).
- **`handoffs.py`** – Contains the `Handoff` class and logic for agent-to-agent handoffs. This includes the mechanism by which an agent’s output can indicate switching to another agent, and how the context or input is transferred.
- **`guardrail.py`** – Defines guardrail classes for input and output. Guardrails are essentially checks or transformations that run before an agent processes input or before it returns output, to enforce constraints (like “the answer should not contain offensive content” or “the query must be about finance”).
- **`tracing/`** – A sub-package with tracing integration. It contains classes for Trace and Span which capture events during agent runs, and processors to send these traces to various logging or monitoring backends. By default, the SDK can send traces to the OpenAI dashboard (as seen in the tracing UI screenshot above) or you can hook it to other systems.
- **`models/`** – This holds model provider interfaces (for example, an `OpenAIChatCompletions` model implementation). The SDK is designed to be model-agnostic to some extent – it can integrate with any model that speaks the OpenAI Chat Completions protocol ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=Notably%2C%20our%20SDK%20is%20compatible,OpenAI%20Chat%20Completions%20API%20format)). The `models` module lets advanced users plug in different LLM backends if needed (like Azure OpenAI, or even open-source models that mimic the chat API).

In addition to the core library code, the repository includes an **`examples/`** directory ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=Explore%20the%20examples%20directory%20to,our%20documentation%20for%20more%20details)). These example scripts are extremely useful to see how the SDK is used in practice. Some highlights:

- **Basic examples** (`examples/basic/`): Simple scripts to demonstrate fundamentals. For instance, `hello_world.py` shows a minimal agent answering a prompt ([openai-agents-python/examples/basic/hello_world.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/examples/basic/hello_world.py#:~:text=async%20def%20main)) ([openai-agents-python/examples/basic/hello_world.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/examples/basic/hello_world.py#:~:text=result%20%3D%20await%20Runner.run%28agent%2C%20,me%20about%20recursion%20in%20programming)), and `tools.py` shows how to define and use a custom tool (similar to the example we’ll walk through below). There are also examples for streaming outputs (`stream_text.py`, `stream_items.py`) and using lifecycle event hooks (`agent_lifecycle_example.py`).
- **Agent patterns** (`examples/agent_patterns/`): This contains examples of different prompt/agent strategies (like React-style thinking, iterative refinement, etc.), showing how you can structure the agent’s reasoning patterns.
- **Handoffs** (`examples/handoffs/`): Demonstrations of multiple agents working together. For instance, an example where a triage agent decides whether to hand the request to an English-only agent or a Spanish-only agent based on input language.
- **Customer service bot** (`examples/customer_service/`): A more realistic multi-agent workflow, perhaps simulating a customer support scenario with specialized sub-agents (refund processing, product lookup, etc.).
- **Financial research agent** (`examples/financial_research_agent.py`): A complex agent that likely uses the web search tool to gather information and then produces a synthesized report on a financial topic.
- **Research assistant (open-ended)** (`examples/research_bot/`): Another illustrative agent possibly using iterative search and summarize techniques.
- **Voice agent** (`examples/voice/`): How to use the voice features of the SDK, such as speech-to-text and text-to-speech, to enable a spoken conversation agent.

Each example is a standalone Python script that you can run after installing the SDK (after setting your API key). Reading through these will give you a sense of how to structure prompts, when to use handoffs vs. tools, and how to integrate various capabilities. The SDK’s documentation site also references these examples for deeper insights into certain patterns.

In summary, the codebase is organized to separate the concerns of defining agent behavior (Agent class), executing agent loops (Runner and run implementations), and extending capabilities (Tool implementations, guardrails, etc.). As a developer, you primarily interact with the high-level API (Agent, Runner, and the decorator functions to create tools). The lower-level modules (like `lifecycle.py`, `items.py`, `usage.py`) handle internal details such as event hooks, message item representations, and usage tracking (tokens). Knowing they exist can help if you need to debug or extend the SDK, but you can build powerful agents without diving into those internals.

## Example: Creating a Custom Agent

Now that we have an understanding of the components, let’s walk through a concrete example of building a custom agent. In this example, we’ll create a simple **Weather Assistant** agent that can fetch weather information for a given city using a custom tool. This will demonstrate how to define a tool, register it with an agent, and run the agent to get a result.

**Scenario:** The user wants to ask the agent about the weather in different cities. We’ll provide the agent with a tool that returns weather data (we’ll simulate this with a dummy function, but imagine it could call a real API).

Below is the complete code for our custom agent, with inline comments explaining each part:

```python
from pydantic import BaseModel
from agents import Agent, Runner, function_tool

# 1. Define a Pydantic model for structured tool output (optional but helpful for clarity)
class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str

# 2. Define a function that fetches or computes weather info.
# Use the @function_tool decorator to turn it into an agent-usable tool.
@function_tool
def get_weather(city: str) -> Weather:
    """
    A tool function that returns dummy weather info for the given city.
    In a real application, this could call a weather API or database.
    """
    print(f"[Tool] Fetching weather for {city}...")  # debug log to show tool usage
    # Return a Weather object (the SDK will convert this to JSON for the LLM)
    return Weather(
        city=city,
        temperature_range="14-20°C",
        conditions="Sunny with light winds"
    )

# 3. Create an Agent and configure its abilities
agent = Agent(
    name="WeatherAssistant",
    instructions="You are a weather assistant. Answer questions about the weather by using the available tools.",
    tools=[get_weather]  # Register our custom tool with the agent
    # (We could specify a model here; by default it will use the default OpenAI model)
)

# 4. Use the Runner to execute the agent on a user query.
# We'll use run_sync for simplicity (there is also an async Runner.run for async contexts).
result = Runner.run_sync(agent, "What's the weather in Tokyo?")

# 5. Print out the final output from the agent
print(result.final_output)
# Expected output (the agent's answer):
# "The weather in Tokyo is sunny with light winds, around 14°C to 20°C."
```

Let’s break down what happens when this code runs:

- We defined a `Weather` data model using Pydantic. This is used to structure the output of our `get_weather` tool. Using structured outputs is optional, but it can be useful to provide the LLM with a clear schema of what the tool returns (city, temperature_range, conditions). The SDK will automatically convert this dataclass into a JSON schema for the LLM and parse the function result into a JSON object for the model.
- The `@function_tool` decorator on `get_weather` does a lot behind the scenes. When Python imports this module, the decorator registers `get_weather` as a tool. It infers the tool’s name (in this case `"get_weather"`) and its parameters (a string `city`) and return type (`Weather`). The SDK will know that this tool can be called with an argument `city` and will advertise it to the LLM as: *Tool name:* `get_weather`, *description:* (it may use the docstring or you can explicitly provide one), *parameters:* `{"city": "string"}`. Our implementation of `get_weather` simply prints a debug line (so we can see that it was actually called) and returns a `Weather` object with some dummy data.
- We create the `agent` with a name and instructions. The instructions clearly tell the agent it should answer weather questions using its tools. We include the `tools=[get_weather]` so the agent knows it has access to the `get_weather` function. We could add more tools to this list if needed (for example, another tool for forecasting or for a different data source).
- We then run the agent using `Runner.run_sync(agent, input_text)`. The input is a user’s question: *"What's the weather in Tokyo?"*. The Runner will start the agent loop:
  1. It will prompt the LLM (according to the agent’s instructions) with the user question.
  2. The LLM (given it knows about the `get_weather` tool from the agent’s configuration) is likely to output a function call: `get_weather` with the argument `"Tokyo"`.
  3. The SDK sees this function call in the LLM’s response and invokes our `get_weather` Python function with `"Tokyo"`. The debug print in our tool function will appear in the console, confirming the tool was called.
  4. The tool returns the `Weather` object for Tokyo. The SDK takes this result and formats it for the LLM. Essentially, it adds a message like: *Tool result:* `{"city": "Tokyo", "temperature_range": "14-20°C", "conditions": "Sunny with light winds"}` to the conversation for the LLM.
  5. The LLM receives that and now can use it to formulate a final answer. Based on the instructions, it will likely respond with something like: “The weather in Tokyo is sunny with light winds, and temperatures between 14°C and 20°C.”
  6. The Runner captures that final answer (since the LLM didn’t request another tool or handoff, this is a completion) and returns it as the `final_output` in the `result`.
- Finally, we print `result.final_output` which contains the answer string. The comment in the code shows an expected output. In practice, the wording might vary slightly depending on the LLM chosen (but it should convey the same information from the tool result).

This example showcases the simplicity of adding a tool: we didn’t have to write any parsing or special prompt for the tool use – the SDK handled integrating the function into the model’s decision loop. As a developer, you focus on implementing the function (what you want the tool to do) and describing the agent’s overall behavior. The model’s innate ability to decide when to call the function (thanks to the function calling feature) takes care of the rest.

You can run this agent locally. Make sure to set your `OPENAI_API_KEY` and then run the script. You should see the debug print from the tool, and the final answer printed out. Try modifying the question or adding another tool (for example, a tool that gives a fun fact about the city, and instruct the agent to sometimes provide a fun fact). The Agents SDK will scale to manage multiple tools – the agent will choose which one to call based on the query and its instructions.

## Tool Use and Observations in Detail

In the custom agent example above, we touched on how the agent calls a tool and gets the result. Let’s generalize that process and highlight how you can define and handle tools and observations.

**Defining and Registering Tools:** The most convenient way to add tools to an agent is using the `@function_tool` decorator. By decorating a function, you let the SDK generate a **FunctionTool** object behind the scenes ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=,helpers%20to)). This object includes the function’s name, description, parameter schema, and a reference to the function itself. You can also provide an explicit description if needed (for example, via the docstring or parameters descriptions). After decorating, simply include the function in the agent’s `tools` list. Alternatively, you could manually construct a `FunctionTool` dataclass instance, but the decorator does this for you. 

If your tool function needs to access some external resource (like a database or a web service), you can utilize the context. For example, suppose you have a database connection in your context, your tool function signature could be `def lookup_user(user_id: int, ctx: MyContext) -> UserData:` where `MyContext` holds a DB connection. The SDK allows tools to optionally accept the context as a parameter (depending on how you define the function or by setting `takes_ctx=True` if using lower-level APIs). This way, when the tool is invoked, the SDK will pass the context in. (When using the decorator, you might need to use a special signature or the `context_tool` variant if you want the context passed automatically.)

**How the LLM decides to use tools:** Once tools are registered with the agent, the SDK will format a system message for the model that includes each tool’s name, description, and parameters. This is similar to how you’d call the OpenAI API with function definitions. The LLM, when responding, can choose to output a function call (like `{"function_call": {"name": "tool_name", "arguments": "{...}"}}`). The Agents SDK monitors the LLM’s output. If a function call is present, that triggers the SDK to execute the corresponding tool (action).

**Actions as JSON and Observations as Messages:** The agent’s action is captured as a `FunctionToolResult` internally, which pairs the tool that was called with its output ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=class%20FunctionToolResult%3A)). After execution, the SDK prepares an observation message. Under the hood, it sends a special assistant message of role “function” (as per OpenAI’s protocol) containing the tool’s name and a JSON result. For example, after calling `get_weather`, the assistant message might look like: `{"role": "function", "name": "get_weather", "content": "{ \"city\": \"Tokyo\", \"temperature_range\": \"14-20°C\", \"conditions\": \"Sunny with light winds\"}"}`. The model then gets to see this content in the next turn.

From a developer perspective, you typically don’t need to format or intercept these messages yourself – it’s managed by the Runner. However, understanding this helps in **debugging**. If the agent isn’t using a tool when it should, check that the tool’s name and description make it obvious to the model. If the model output is not parsing to a valid function call (causing errors), you might need to adjust the description or ensure the function signature is simple (the SDK’s automatic schema generation is usually robust with Pydantic). 

**Multiple tools and tool selection:** You can provide multiple tools to an agent. The LLM will choose among them based on what it thinks is needed. For example, if you had both a `get_weather` and `get_time` tool, the agent will call the appropriate one in context (and you could see this in the trace or by printing debug info in each tool function). There is a configuration (`ModelSettings.tool_choice`) that can influence this. By default it’s `auto` (the model decides) ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=match%20at%20L248%20Supplying%20a,Valid%20values%20are)). If you *always* want the model to use a tool before answering (for instance, you have a calculation agent that should use a calculator tool every time), you can force tool usage by setting `tool_choice="force"` in the model settings. Conversely, you could turn off tool usage by not providing any tools or instructing the agent not to use them, but generally if you supply tools you expect the agent to decide when to use them.

**Observations and loop control:** Each tool invocation adds an observation. The agent will incorporate that observation into its next reasoning step. The loop continues until no further actions are taken. It’s worth noting that the SDK has a safeguard `max_turns` parameter you can set when running an agent ([Running agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/running_agents/#:~:text=3,we%20raise%20a%20MaxTurnsExceeded%20exception)). This prevents infinite loops in case the agent gets stuck calling tools repeatedly or handing off back and forth without producing an answer. By default, the loop will exit once the agent produces a final answer (a message that is not a tool call or handoff). The final answer can be plain text or a structured output (if `output_type` is specified, the loop will stop when the model returns an output conforming to that type) ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=Final%20output%20is%20the%20last,agent%20produces%20in%20the%20loop)).

**Inspecting results:** The `Runner.run` (or `run_sync`) returns a `RunResult` object. We used `result.final_output` in the example to get the answer. But `RunResult` contains more information. For instance, `result.new_items` is a list of message items produced during the run (it will include the user message, the assistant’s tool call, the function result message, and the final assistant answer, among others). You can inspect these items to see step-by-step what happened (this is essentially how the tracing works). There are also fields like `result.last_agent` (in case of handoffs, which agent gave the final answer) ([Results - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/results/#:~:text=,Streaming)), and `result.raw_responses` which can include raw model outputs if needed ([Results - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/results/#:~:text=,14)). For most uses, you won’t need to dig into these, but they are there for debugging or advanced workflows.

In summary, to *build and use tools effectively*: define your function with clear purpose and signature, decorate it with `@function_tool`, give your agent good instructions that hint *when* to use the tool, and then let the SDK handle the call/response. Always test your agent with a few sample inputs to ensure the LLM uses the tool as expected. If it doesn’t, you may need to adjust the tool description or the prompt instructions (sometimes explicitly telling the agent “If the user asks for X, use the Y tool to find the answer” in the instructions can help). The Agents SDK greatly simplifies the mechanics of tool use – you get to focus on what the tool does and how it helps answer the user’s query.

## Integration and Deployment

Once your agent is developed and working in a local test, you’ll likely want to integrate it into a larger application or deploy it as a service. Here are some considerations for running Agents SDK in different environments:

**Running Locally:** The simplest scenario is running the agent in a local script or application (as we did in our example). This just requires an internet connection (to reach the OpenAI API) and your API key configured. You can incorporate the agent’s run in any Python workflow. For example, you might build a command-line chatbot around it or have a scheduled job that uses an agent to gather data. Local execution will use the CPU for running your Python tool functions and rely on OpenAI’s cloud for the LLM calls (unless you configured a different model provider).

**Asynchronous usage:** The SDK supports asynchronous operation. In our example, we used `Runner.run_sync` for convenience. In an async web server (like FastAPI or Flask with async), you would instead `await Runner.run(agent, input)` inside an async function. Under the hood, the OpenAI API calls (via `openai` library) can be asynchronous, and your tool functions can also be async if needed. The SDK will await them appropriately. This means you can serve multiple agent queries concurrently in an async I/O framework (each `Runner.run` is an coroutine you can schedule). Ensure thread safety if your tools share resources, or stick to async to avoid threading issues.

**Deploying to Production:** If you want to expose the agent via an API or a bot interface, you can wrap the agent call in a web service. For example, you could create a Flask route that accepts a question and returns the agent’s answer. Inside the route, you’d do something like:

```python
user_question = request.json["question"]
result = Runner.run_sync(agent, user_question)
return jsonify({"answer": result.final_output})
```

Ensure to handle errors – the SDK might raise exceptions (for instance, if the OpenAI API returns an error or if a tool function raises an exception). The `agents.exceptions` module defines specific exception types like `ToolError`, `HandoffError`, `MaxTurnsExceeded` etc. ([Running agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/running_agents/#:~:text=3,we%20raise%20a%20MaxTurnsExceeded%20exception)) that you might catch for graceful handling (e.g., returning a message to the user that something went wrong, or a fallback answer).

**Remote environments:** Deployment could be on a cloud VM, a container, or a serverless function. In all cases, you need to include the `openai-agents` package in your environment (add to requirements.txt or the equivalent). Also set up the environment variable for the API key in that environment. No other special configuration is needed for the SDK itself. If your agent uses additional resources (like files, databases, or external APIs), ensure those are accessible in the deployed environment and handle any credentials for those as well (just like any Python application). 

**Model providers:** By default, the agent will use OpenAI’s API with the model name you specify (or a default). The Agents SDK is compatible with any provider that supports the OpenAI Chat Completions format ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=Notably%2C%20our%20SDK%20is%20compatible,OpenAI%20Chat%20Completions%20API%20format)). This means you could, for example, use Azure OpenAI (by configuring the openai library to use Azure endpoints), or even open-source models through an adapter that mimics the API. The `model_provider` parameter in `Runner.run` or the `model` field in Agent can be used to specify a different backend. In the repository’s `examples/model_providers`, there may be demonstrations of using Anthropic or others. When deploying, you might configure the model via environment variables or a config file, so that you can switch from a dev (maybe `gpt-3.5-turbo`) model to a production model (maybe `gpt-4`) without changing code. 

**Scaling and performance:** Each agent run will involve one or multiple API calls to the LLM. If you plan to serve many requests or very frequent calls, keep in mind the rate limits and throughput of the OpenAI model you use. You may need to run multiple instances of your service to handle load, or queue requests if an agent’s response time is high (especially if it uses many tools or a slow model). Caching can be a useful technique – for instance, if the same question is asked often, you could cache the result to avoid running the agent repeatedly for identical inputs.

**Maintaining state between runs:** The Agents SDK treats each `Runner.run` as one isolated interaction (which could internally involve multiple agents and steps, but ultimately ends with one final output). If you are building a multi-turn chat with a user (iterative conversation), you will need to manage the conversation state across runs. The SDK allows you to pass the conversation history as the input instead of a simple string. For example, you can supply a list of messages (using the `Item` classes or raw dicts) to continue a conversation. Another approach is to have the agent’s final output include a prompt for the next user turn and maintain a memory in the context. This is a more complex use case; initially, you can treat each question independently.

**Security considerations:** If deploying an agent that can execute tools like `ComputerTool` (which can run shell commands) or making external API calls, be mindful of what input you allow and what the agent can do. Guardrails are important in such cases – e.g., to prevent the agent from executing destructive commands. The `ComputerTool` in the SDK is somewhat sandboxed, but you should still scope what it can access. On a server, prefer to disable any tools that you don’t absolutely need to expose. If the agent is only answering questions using safe tools (like read-only APIs), the risk is lower. Always sanitize or validate the agent’s output if it’s used in a sensitive context (for example, if an agent writes to a database or executes code, have checks in place).

**Monitoring and logging:** In production, you’ll want to monitor your agent’s performance. The built-in tracing can be very useful here. By default, traces might be sent to OpenAI’s servers (where you can view them in the developer dashboard under “Logs/Trace”). You can also set up custom trace processors to send data to your own logging system (the SDK can integrate with third-party observability tools like Langfuse, etc.). At minimum, logging the input questions and outputs, and perhaps any exceptions, will help in debugging issues over time.

In short, deploying an agent is like deploying any Python service that calls external APIs: make sure to handle keys, errors, and scaling. The Agents SDK is designed to be **production-ready** and has been used to deploy complex agent workflows ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=In%20addition%20to%20building%20the,successfully%20deployed%20by%20multiple%20customers)) ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=The%20Agents%20SDK%20is%20suitable,developers%20focus%20more%20on%20meaningful)). It doesn’t require any special infrastructure beyond what’s needed to call the OpenAI API. So you can containerize your agent app or run it on a cloud function, and it should work as long as network access to the API is available. 

## Limitations and Best Practices

When building with the OpenAI Agents SDK, keep in mind some limitations of both the SDK and the underlying AI systems, and follow best practices to get the best results.

**1. LLM behavior and reliability:** The agent’s “brain” is an LLM, which means it may not always behave 100% deterministically. Sometimes the agent might not use a tool even if it seems obvious to you, or it might call a tool with slightly wrong arguments. The SDK tries to mitigate this with things like strict JSON schemas for function calls (by default `strict_json_schema=True` for tools to enforce correct arg formats ([openai-agents-python/src/agents/tool.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py#:~:text=strict_json_schema%3A%20bool%20%3D%20True))). **Best Practice:** Provide very clear instructions. If tool use is critical, explicitly tell the agent in the instructions *when* to use which tool. You can also adjust the model’s temperature down to make it more deterministic if needed. In critical applications, consider using `output_guardrails` to validate the final answer and maybe retry or correct course if the answer is not adequate.

**2. Tool design:** The quality of your tools can greatly affect the agent’s performance. If a tool returns a huge amount of data (e.g., a long text), the LLM will have to digest that which could be hard or could hit token limits. If a tool’s purpose overlaps with what the LLM can do itself, the agent might ignore the tool. **Best Practice:** Make tools focused and use them for things the LLM inherently can’t do (e.g., retrieve fresh information, compute something precisely). Test each tool individually by prompting the agent to use it. Ensure the tool’s description is accurate. If the agent tends to hallucinate or ignore a tool, you might implement a guardrail or use `ModelSettings.tool_choice="force"` to require a tool be used at least once ([Agents - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/#:~:text=match%20at%20L248%20Supplying%20a,Valid%20values%20are)).

**3. Handoffs complexity:** If you use multiple agents with handoffs, design their interaction carefully to avoid ping-pong handoffs or confusion about who should handle what. Handoffs are essentially learned by the model – the triage agent in our earlier example had instructions on when to pass to Spanish vs English agent. **Best Practice:** Clearly delineate the responsibilities of each agent in their `instructions` and perhaps use the `handoff_description` field for each sub-agent to guide the parent agent ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=handoff_description%3A%20str%20,None)) ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=handoffs%3A%20list%5BAgent%5BAny%5D%20,default_factory%3Dlist)). Use the provided examples as templates for structuring these interactions.

**4. Performance and token usage:** Each iteration of the agent loop incurs a call to the LLM. Complex agent tasks (especially those that call many tools or chain through multiple agents) can be slow or token-intensive. **Best Practice:** Where possible, try to get the model to solve problems with fewer high-quality steps rather than many incremental steps. Set a reasonable `max_turns` to prevent runaway loops. Monitor the `result.usage` (token usage) or use OpenAI’s billing tools to understand the cost. If you find the agent is using too many tokens reasoning verbosely, you might shorten the instructions or use a more concise tool result format. Also, consider using `output_type` to constrain the final answer format; this can make the model stop when it has produced the required output structure rather than continuing to ramble ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=Final%20output)).

**5. Error handling:** Sometimes tools can fail (maybe an API is down, or the tool function raises an exception for bad input). The SDK will catch exceptions from tools and can either surface them as an error message to the LLM or stop the run. By default, an exception in a tool will typically abort the run with a `ToolException`. **Best Practice:** Anticipate errors. For example, you can wrap the logic in your tool and return a message like `"Error: location not found"` as a normal string result instead of raising, to let the agent handle it gracefully. Alternatively, use guardrails to catch certain issues (like if the user input is invalid, you can pre-empt with an input guardrail).

**6. Do not overprompt or underprompt:** If your agent is not performing well, review the system instructions (and tool descriptions). Too little detail and the model might not know what to do; too much detail and you risk hitting token limits or conflicting instructions. For instance, if you have many tools, listing all of them with long descriptions uses up context. Try to keep descriptions concise. You might also leverage the `context` to dynamically generate instructions if needed (the Agent supports `instructions` as a callable that can use context ([openai-agents-python/src/agents/agent.py at main · openai/openai-agents-python · GitHub](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py#:~:text=instructions%3A%20))). This way you can inject situational instructions without hardcoding them.

**7. Extensibility:** The Agents SDK is open-source and designed to be extensible. If you have advanced needs, you can subclass the Agent class or create custom tool classes. For example, if you wanted a tool that streams its result back token by token to the LLM (maybe for very large outputs), you might create a custom implementation. The SDK’s design was inspired by frameworks like Pydantic AI and Swarm, and it tries to remain flexible ([GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python#:~:text=We%27d%20like%20to%20acknowledge%20the,source%20community%2C%20especially)). **Best Practice:** Before extending, check the documentation – many features (like streaming, voice, custom models) might already be supported via configuration. Use extension points (like lifecycle events hooks in `lifecycle.py`) to inject logic at steps of the agent run (for example, logging each tool call or modifying the prompt before each LLM call).

**8. Testing and iteration:** Building an agent often requires a bit of experimentation. **Best Practice:** Use the tracing tools during development. Run your agent with `trace_id` set so you can examine the full trace of actions. This will help you spot where the agent took a wrong turn. Write unit tests for your tool functions (since they are plain Python) to ensure they handle edge cases. You can even write tests for the agent by mocking the OpenAI responses (though that’s advanced). At minimum, try a variety of inputs and make sure the agent responds sensibly and within expected limits (time, token usage, correctness). 

**9. Know the limits of the model:** If your agent uses a base model that has knowledge cut off (e.g., GPT-4’s training data might not include very recent events), ensure you compensate with tools (like web search) for queries on latest information. If the model has size limits (context length), be mindful if your agent carries over a long conversation or if tool outputs are lengthy.

**10. Security and misuse:** Always consider how your agent could be prompted or manipulated. A user might try to get the agent to reveal info or perform actions it shouldn’t. The SDK’s guardrails feature can help filter inputs or outputs (for example, block certain content), but they are not foolproof. Keep sensitive operations (like file writes, sending emails, financial transactions) behind additional confirmation or logic rather than fully trusting the AI. Use `output_guardrails` to validate anything critical the agent produces (e.g., if the agent drafts an email or code, have checks or human review steps as needed).

By following these best practices, you can harness the power of the OpenAI Agents SDK effectively while minimizing unexpected behavior. As a developer, always maintain a bit of oversight: logs and traces are your friend to understand your agent’s decisions. The OpenAI Agents SDK is a powerful tool that, when used wisely, can enable AI agents that are both robust and safe, performing complex tasks that integrate reasoning and action ([New tools for building agents | OpenAI](https://openai.com/index/new-tools-for-building-agents/#:~:text=In%20addition%20to%20building%20the,successfully%20deployed%20by%20multiple%20customers)). Happy building!

