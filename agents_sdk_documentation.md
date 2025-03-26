# OpenAI Agents SDK Documentation

## Overview

The OpenAI Agents SDK is a lightweight yet powerful framework for building multi-agent workflows. It provides a structured way to create, configure, and orchestrate LLM-powered agents with tools, guardrails, handoffs, and tracing capabilities.

## Core Concepts

### Agents

Agents are the primary building blocks of the SDK. An agent combines an LLM with instructions, tools, guardrails, and handoff capabilities.

```python
from agents import Agent

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4o",  # Optional, defaults to gpt-4o
    tools=[],        # Optional list of tools
    handoffs=[],     # Optional list of agents to hand off to
    output_type=None # Optional structured output type
)
```

### Running Agents

The `Runner` class provides methods to execute agents:

```python
from agents import Runner

# Synchronous execution
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Asynchronous execution
result = await Runner.run(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Streaming execution
result = Runner.run_streamed(agent, "Write a haiku about recursion in programming.")
async for event in result.stream_events():
    # Process streaming events
    pass
```

### Tools

Tools allow agents to perform actions and access external functionality. The SDK provides several ways to define tools:

#### Function Tools

The simplest way to create tools is by decorating Python functions:

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."
```

#### Tools using other agents

Agents can use other agents as tools:

```python
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish"
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="You are a translation agent. Use tools to translate.",
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish"
        )
    ]
)
```

#### Built-in Tools

The SDK includes built-in tools like `WebSearchTool` and `ComputerTool`:

```python
from agents import WebSearchTool, ComputerTool

# Web search tool
agent = Agent(
    name="Web searcher",
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})]
)

# Computer tool (for controlling a computer)
computer = AsyncComputer() # Custom implementation
agent = Agent(
    name="Computer user",
    tools=[ComputerTool(computer)],
    model="computer-use-preview",
    model_settings=ModelSettings(truncation="auto")
)
```

### Handoffs

Handoffs allow agents to transfer control to other agents:

```python
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish."
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English"
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent]
)
```

### Guardrails

Guardrails are safety mechanisms that validate inputs and outputs:

#### Input Guardrails

Input guardrails validate user inputs before they reach the agent:

```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def profanity_filter(context, agent, input_text):
    contains_profanity = check_profanity(input_text)
    
    return GuardrailFunctionOutput(
        output_info={"contains_profanity": contains_profanity},
        tripwire_triggered=contains_profanity
    )

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    input_guardrails=[profanity_filter]
)
```

#### Output Guardrails

Output guardrails validate agent outputs before they are returned to the user:

```python
from agents import output_guardrail, GuardrailFunctionOutput

@output_guardrail
async def sensitive_data_check(context, agent, output):
    contains_sensitive_data = check_for_sensitive_data(output)
    
    return GuardrailFunctionOutput(
        output_info={"contains_sensitive_data": contains_sensitive_data},
        tripwire_triggered=contains_sensitive_data
    )

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_guardrails=[sensitive_data_check]
)
```

### Structured Outputs

Agents can return structured data using Pydantic models:

```python
from pydantic import BaseModel, Field

class WeatherReport(BaseModel):
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Current weather conditions")
    location: str = Field(description="Location for this report")

agent = Agent(
    name="Weather agent",
    instructions="You provide weather information.",
    output_type=WeatherReport
)
```

### Tracing

The SDK includes built-in tracing for monitoring and debugging agent behavior:

```python
from agents import trace

# Trace a specific block of code
with trace("My workflow"):
    result = await Runner.run(agent, "What's the weather like?")
```

Tracing can be customized or extended with various processors:

```python
from agents.tracing import setup_tracing, BatchTraceProcessor, ConsoleSpanExporter

# Custom tracing setup
exporter = ConsoleSpanExporter()
processor = BatchTraceProcessor(exporter)
setup_tracing(processor=processor)
```

## Multi-Agent Patterns

The SDK supports various patterns for building complex agent workflows:

### Sequential Agents with Handoffs

Multiple agents can be chained together using handoffs:

```python
research_agent = Agent(name="Researcher", instructions="Research the topic thoroughly.")
writer_agent = Agent(name="Writer", instructions="Write a summary based on the research.")
editor_agent = Agent(name="Editor", instructions="Proofread and improve the text.")

research_agent.handoffs = [writer_agent]
writer_agent.handoffs = [editor_agent]

result = await Runner.run(research_agent, "Tell me about quantum computing.")
```

### Agents as Tools

One agent can use other agents as tools:

```python
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An english to italian translator",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
)
```

### Parallel Execution

Multiple agents can run in parallel:

```python
import asyncio

async def run_agents_parallel(input_text):
    tasks = [
        Runner.run(agent1, input_text),
        Runner.run(agent2, input_text),
        Runner.run(agent3, input_text)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### LLM as a Judge

Using an LLM to evaluate and select the best response from multiple agents:

```python
from pydantic import BaseModel, Field

class JudgementResult(BaseModel):
    winning_response: str = Field(description="The best response chosen")
    reasoning: str = Field(description="Explanation of why this response was chosen")

judge_agent = Agent(
    name="Judge",
    instructions=(
        "You are a judge who evaluates multiple responses and selects the best one. "
        "Consider accuracy, completeness, and helpfulness in your evaluation."
    ),
    output_type=JudgementResult
)

async def evaluate_responses(prompt, agents_to_evaluate):
    # Collect responses from all agents
    results = await asyncio.gather(*[Runner.run(agent, prompt) for agent in agents_to_evaluate])
    responses = [f"Response from {agent.name}:\n{result.final_output}" for agent, result in zip(agents_to_evaluate, results)]
    
    # Judge chooses the best response
    judge_input = f"Question: {prompt}\n\n" + "\n\n".join(responses)
    judge_result = await Runner.run(judge_agent, judge_input)
    
    return judge_result.final_output
```

### Deterministic Flows

Creating a predictable sequence of agent actions with clear transition points:

```python
from pydantic import BaseModel, Field

class ResearchPlan(BaseModel):
    key_questions: list[str] = Field(description="List of key questions to research")
    approach: str = Field(description="The approach to answering these questions")

class ResearchFindings(BaseModel):
    findings: dict[str, str] = Field(description="Key findings for each research question")
    sources: list[str] = Field(description="Sources of information")

class FinalReport(BaseModel):
    summary: str = Field(description="Executive summary of findings")
    details: str = Field(description="Detailed explanation of research results")
    recommendations: list[str] = Field(description="List of recommendations based on findings")

# Step 1: Planner agent creates a structured research plan
planner_agent = Agent(
    name="Planner",
    instructions="Create a structured research plan for the given topic.",
    output_type=ResearchPlan
)

# Step 2: Researcher agent follows the plan and produces findings
researcher_agent = Agent(
    name="Researcher",
    instructions="Follow the research plan and produce detailed findings.",
    output_type=ResearchFindings
)

# Step 3: Report writer creates the final report
report_agent = Agent(
    name="ReportWriter",
    instructions="Create a comprehensive final report based on the research findings.",
    output_type=FinalReport
)

async def run_research_workflow(topic):
    # Execute each step in sequence with structured inputs/outputs
    plan_result = await Runner.run(planner_agent, topic)
    plan = plan_result.final_output  # ResearchPlan object
    
    research_input = f"Topic: {topic}\nResearch plan: {plan.model_dump_json()}"
    research_result = await Runner.run(researcher_agent, research_input)
    findings = research_result.final_output  # ResearchFindings object
    
    report_input = f"Topic: {topic}\nFindings: {findings.model_dump_json()}"
    report_result = await Runner.run(report_agent, report_input)
    
    return report_result.final_output  # FinalReport object
```

### Forcing Tool Use

Ensuring agents use specific tools rather than their own knowledge:

```python
@function_tool
def fetch_company_information(company_name: str) -> str:
    # In a real system, this would query a database or API
    companies = {
        "acme corp": "Founded in 1942, revenue $1.2B, 5000 employees",
        "globex": "Founded in 1988, revenue $3.5B, 12000 employees",
        "initech": "Founded in 1996, revenue $500M, 2000 employees"
    }
    return companies.get(company_name.lower(), "No information available")

@function_tool
def get_stock_price(ticker: str) -> str:
    # In a real system, this would query a financial API
    stocks = {
        "ACME": "$420.69 (+1.2%)",
        "GLO": "$123.45 (-0.8%)",
        "INT": "$45.67 (+0.3%)"
    }
    return stocks.get(ticker.upper(), "No stock information available")

# Agent configured to always use tools for factual information
agent = Agent(
    name="Financial Advisor",
    instructions=(
        "You provide financial advice. NEVER provide company information or stock prices from your own knowledge. "
        "ALWAYS use the appropriate tools to get this information. If you don't have a tool for the requested "
        "information, inform the user that you cannot provide that information."
    ),
    tools=[fetch_company_information, get_stock_price],
    model_settings=ModelSettings(tool_choice="required")  # Force tool use
)
```

### Routing Agents

Building a dispatching system to route queries to specialized agents:

```python
from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    destination: str = Field(description="The name of the agent to route to")
    reason: str = Field(description="Explanation of why this routing decision was made")

router_agent = Agent(
    name="Router",
    instructions=(
        "You are a router that directs user queries to the appropriate specialized agent. "
        "Analyze the query carefully and select the most appropriate destination."
    ),
    output_type=RoutingDecision
)

# Specialized agents
tech_agent = Agent(name="TechSupport", instructions="You provide technical support for our products.")
billing_agent = Agent(name="Billing", instructions="You handle billing and payment inquiries.")
sales_agent = Agent(name="Sales", instructions="You help customers with purchasing decisions.")

async def route_and_respond(user_query):
    # First, determine where to route the query
    route_result = await Runner.run(router_agent, user_query)
    routing = route_result.final_output
    
    # Map the routing decision to the actual agent
    agent_map = {
        "TechSupport": tech_agent,
        "Billing": billing_agent,
        "Sales": sales_agent
    }
    
    target_agent = agent_map.get(routing.destination)
    if not target_agent:
        return "Sorry, I couldn't route your query appropriately."
    
    # Route to the appropriate agent and get response
    response = await Runner.run(target_agent, user_query)
    return response.final_output
```

## Model Providers

The SDK is compatible with any model provider supporting the OpenAI Chat Completions API format:

```python
from agents import Agent, ModelProvider, ModelSettings

# Define a custom model provider
class MyCustomProvider(ModelProvider):
    # Implement required methods
    pass

# Use a custom model provider
agent = Agent(
    name="Custom agent",
    model_provider=MyCustomProvider(),
    model="my-custom-model",
    model_settings=ModelSettings(temperature=0.7)
)
```

## Agent Lifecycle Hooks

The SDK provides hooks to customize agent behavior at various points:

```python
from agents import Agent, AgentHooks, RunContextWrapper

class MyAgentHooks(AgentHooks):
    async def on_tool_start(self, context, agent, tool):
        print(f"Agent {agent.name} is about to use tool: {tool}")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"Agent {agent.name} used tool: {tool} with result: {result}")

agent = Agent(
    name="Agent with hooks",
    instructions="You are a helpful assistant.",
    hooks=MyAgentHooks()
)
```

## Best Practices

### Managing Context and Dependencies

Use the context parameter to pass data to tools and hooks:

```python
class MyContext:
    def __init__(self, database):
        self.database = database

context = MyContext(database=my_db_connection)
result = await Runner.run(agent, input="Query the database", context=context)
```

### Error Handling

Handle exceptions appropriately:

```python
from agents.exceptions import OutputGuardrailTripwireTriggered, InputGuardrailTripwireTriggered

try:
    result = await Runner.run(agent, "User input")
    print(result.final_output)
except InputGuardrailTripwireTriggered as e:
    print(f"Input guardrail triggered: {e.guardrail_result.output.output_info}")
except OutputGuardrailTripwireTriggered as e:
    print(f"Output guardrail triggered: {e.guardrail_result.output.output_info}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Streaming Responses

Process streaming responses for better user experience:

```python
result = Runner.run_streamed(agent, "Generate a long response")

async for event in result.stream_events():
    if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
        print(event.data.delta, end="", flush=True)
```

## Example Workflows

### Customer Service Bot

```python
from agents import Agent, function_tool, Runner

@function_tool
def get_order_status(order_id: str) -> str:
    # In a real implementation, this would query a database
    return f"Order {order_id} is being shipped and will arrive on Tuesday."

@function_tool
def get_return_policy() -> str:
    return "Returns are accepted within 30 days with receipt."

agent = Agent(
    name="Customer Service Bot",
    instructions="You are a helpful customer service representative.",
    tools=[get_order_status, get_return_policy]
)

result = await Runner.run(agent, "What's the status of my order #12345?")
print(result.final_output)
```

### Research Assistant

A more complex example with multiple specialized agents:

```python
planner_agent = Agent(
    name="PlannerAgent",
    instructions="Create a research plan for the topic provided by the user."
)

search_agent = Agent(
    name="SearchAgent",
    instructions="Search for accurate information on the topic."
)

writer_agent = Agent(
    name="WriterAgent",
    instructions="Write a comprehensive report based on the research."
)

# Connect agents with handoffs
planner_agent.handoffs = [search_agent]
search_agent.handoffs = [writer_agent]

# Run the workflow
result = await Runner.run(planner_agent, "Research the impact of AI on healthcare.")
print(result.final_output)
```

## Conclusion

The OpenAI Agents SDK provides a flexible, extensible framework for building sophisticated AI agent applications. By combining agents, tools, guardrails, and other components, developers can create powerful workflows that leverage the capabilities of large language models in a structured, controlled manner. 

## Web Search Tool

The WebSearchTool is a built-in tool that allows agents to search the web for real-time information, enabling them to answer questions about current events, recent developments, and factual information that may not be available in the model's training data.

### How WebSearchTool Works

The WebSearchTool connects to a web search API to retrieve relevant information from the internet. When an agent uses this tool, it:

1. Sends the search query to the search API
2. Receives search results including snippets and URLs from web pages
3. Returns these results to the agent for interpretation and synthesis

### Basic Usage

Here's how to use the WebSearchTool with an agent:

```python
import asyncio
from agents import Agent, Runner, WebSearchTool, trace

async def main():
    # Create an agent with WebSearchTool
    agent = Agent(
        name="Web searcher",
        instructions="You are a helpful agent that can search for real-time information.",
        tools=[WebSearchTool()]
    )

    # Run the agent with a query that requires web search
    with trace("Web search example"):
        result = await Runner.run(
            agent, 
            "What are the latest developments in quantum computing?"
        )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Options

The WebSearchTool can be configured with several parameters:

```python
from agents import WebSearchTool

# Basic configuration with user location
web_search = WebSearchTool(
    user_location={"type": "approximate", "city": "New York"}
)

# Advanced configuration
web_search = WebSearchTool(
    user_location={"type": "approximate", "city": "San Francisco"},
    result_count=5,          # Number of search results to return (default: 3)
    time_range="day",        # Time range for results: 'day', 'week', 'month', 'year' (optional)
    search_region="us",      # Region code for search results (optional)
    include_domains=["nytimes.com", "theguardian.com"],  # Only include specific domains (optional)
    exclude_domains=["pinterest.com", "instagram.com"],  # Exclude specific domains (optional)
)
```

### Example: Targeted News Search

```python
import asyncio
from agents import Agent, Runner, WebSearchTool

async def main():
    # Create a news search tool that focuses on specific domains
    news_search = WebSearchTool(
        include_domains=["bbc.com", "reuters.com", "apnews.com"],
        result_count=4,
        time_range="day"
    )
    
    agent = Agent(
        name="News researcher",
        instructions=(
            "You are a news research assistant. When searching for news, carefully analyze "
            "the search results and provide a concise summary of the key information, "
            "including relevant sources."
        ),
        tools=[news_search]
    )
    
    result = await Runner.run(
        agent,
        "What are the latest developments in the ongoing climate conference?"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Local Information Search

The user_location parameter helps provide more relevant local information:

```python
import asyncio
from agents import Agent, Runner, WebSearchTool

async def main():
    # Create a local information search tool
    local_search = WebSearchTool(
        user_location={"type": "approximate", "city": "Boston"}
    )
    
    agent = Agent(
        name="Local guide",
        instructions=(
            "You are a local guide assistant. When asked about local information, "
            "use the search tool to find relevant details about places, events, "
            "or services in the user's area."
        ),
        tools=[local_search]
    )
    
    result = await Runner.run(
        agent,
        "What are some popular restaurants in the area for dinner tonight?"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Best Practices

When using WebSearchTool, consider these best practices:

1. **Be specific in search queries**: Instruct your agent to create targeted search queries rather than broad ones for better results.

2. **Handle search failures gracefully**: Sometimes searches may not return useful results. Ensure your agent can handle these cases.

3. **Provide context in agent instructions**: Include specific guidance in your agent's instructions about how to interpret and synthesize search results.

4. **Use time ranges appropriately**: For time-sensitive topics, configure the appropriate time_range parameter.

5. **Consider rate limits**: Web search APIs may have rate limits. In high-traffic applications, implement appropriate handling for rate limit errors.

The WebSearchTool enables agents to access the most current information available on the web, allowing them to provide up-to-date answers and insights rather than relying solely on their training data. 