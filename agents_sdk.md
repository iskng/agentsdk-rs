<directory_structure>
.tmp1JnanN
├── .git
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── model_provider.md
│   │   └── question.md
│   ├── PULL_REQUEST_TEMPLATE
│   │   └── pull_request_template.md
│   └── workflows
│       ├── docs.yml
│       ├── issues.yml
│       ├── publish.yml
│       └── tests.yml
├── docs
│   ├── assets
│   │   ├── images
│   │   │   ├── favicon-platform.svg
│   │   │   └── orchestration.png
│   │   └── logo.svg
│   ├── ref
│   │   ├── extensions
│   │   │   ├── handoff_filters.md
│   │   │   └── handoff_prompt.md
│   │   ├── models
│   │   │   ├── interface.md
│   │   │   ├── openai_chatcompletions.md
│   │   │   └── openai_responses.md
│   │   ├── tracing
│   │   │   ├── create.md
│   │   │   ├── index.md
│   │   │   ├── processor_interface.md
│   │   │   ├── processors.md
│   │   │   ├── scope.md
│   │   │   ├── setup.md
│   │   │   ├── span_data.md
│   │   │   ├── spans.md
│   │   │   ├── traces.md
│   │   │   └── util.md
│   │   ├── voice
│   │   │   ├── models
│   │   │   │   ├── openai_provider.md
│   │   │   │   ├── openai_stt.md
│   │   │   │   └── openai_tts.md
│   │   │   ├── events.md
│   │   │   ├── exceptions.md
│   │   │   ├── input.md
│   │   │   ├── model.md
│   │   │   ├── pipeline.md
│   │   │   ├── pipeline_config.md
│   │   │   ├── result.md
│   │   │   ├── utils.md
│   │   │   └── workflow.md
│   │   ├── agent.md
│   │   ├── agent_output.md
│   │   ├── exceptions.md
│   │   ├── function_schema.md
│   │   ├── guardrail.md
│   │   ├── handoffs.md
│   │   ├── index.md
│   │   ├── items.md
│   │   ├── lifecycle.md
│   │   ├── model_settings.md
│   │   ├── result.md
│   │   ├── run.md
│   │   ├── run_context.md
│   │   ├── stream_events.md
│   │   ├── tool.md
│   │   └── usage.md
│   ├── stylesheets
│   │   └── extra.css
│   ├── voice
│   │   ├── pipeline.md
│   │   ├── quickstart.md
│   │   └── tracing.md
│   ├── agents.md
│   ├── config.md
│   ├── context.md
│   ├── examples.md
│   ├── guardrails.md
│   ├── handoffs.md
│   ├── index.md
│   ├── models.md
│   ├── multi_agent.md
│   ├── quickstart.md
│   ├── results.md
│   ├── running_agents.md
│   ├── streaming.md
│   ├── tools.md
│   └── tracing.md
├── examples
│   ├── agent_patterns
│   │   ├── README.md
│   │   ├── agents_as_tools.py
│   │   ├── deterministic.py
│   │   ├── forcing_tool_use.py
│   │   ├── input_guardrails.py
│   │   ├── llm_as_a_judge.py
│   │   ├── output_guardrails.py
│   │   ├── parallelization.py
│   │   └── routing.py
│   ├── basic
│   │   ├── agent_lifecycle_example.py
│   │   ├── dynamic_system_prompt.py
│   │   ├── hello_world.py
│   │   ├── hello_world_jupyter.py
│   │   ├── lifecycle_example.py
│   │   ├── stream_items.py
│   │   ├── stream_text.py
│   │   └── tools.py
│   ├── customer_service
│   │   └── main.py
│   ├── financial_research_agent
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── financials_agent.py
│   │   │   ├── planner_agent.py
│   │   │   ├── risk_agent.py
│   │   │   ├── search_agent.py
│   │   │   ├── verifier_agent.py
│   │   │   └── writer_agent.py
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── manager.py
│   │   └── printer.py
│   ├── handoffs
│   │   ├── message_filter.py
│   │   └── message_filter_streaming.py
│   ├── model_providers
│   │   ├── README.md
│   │   ├── custom_example_agent.py
│   │   ├── custom_example_global.py
│   │   └── custom_example_provider.py
│   ├── research_bot
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── planner_agent.py
│   │   │   ├── search_agent.py
│   │   │   └── writer_agent.py
│   │   ├── sample_outputs
│   │   │   ├── product_recs.md
│   │   │   ├── product_recs.txt
│   │   │   ├── vacation.md
│   │   │   └── vacation.txt
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── manager.py
│   │   └── printer.py
│   ├── tools
│   │   ├── computer_use.py
│   │   ├── file_search.py
│   │   └── web_search.py
│   ├── voice
│   │   ├── static
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   └── util.py
│   │   ├── streamed
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   └── my_workflow.py
│   │   └── __init__.py
│   └── __init__.py
├── src
│   └── agents
│       ├── extensions
│       │   ├── __init__.py
│       │   ├── handoff_filters.py
│       │   └── handoff_prompt.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── _openai_shared.py
│       │   ├── fake_id.py
│       │   ├── interface.py
│       │   ├── openai_chatcompletions.py
│       │   ├── openai_provider.py
│       │   └── openai_responses.py
│       ├── tracing
│       │   ├── __init__.py
│       │   ├── create.py
│       │   ├── logger.py
│       │   ├── processor_interface.py
│       │   ├── processors.py
│       │   ├── scope.py
│       │   ├── setup.py
│       │   ├── span_data.py
│       │   ├── spans.py
│       │   ├── traces.py
│       │   └── util.py
│       ├── util
│       │   ├── __init__.py
│       │   ├── _coro.py
│       │   ├── _error_tracing.py
│       │   ├── _json.py
│       │   ├── _pretty_print.py
│       │   ├── _transforms.py
│       │   └── _types.py
│       ├── voice
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   ├── openai_model_provider.py
│       │   │   ├── openai_stt.py
│       │   │   └── openai_tts.py
│       │   ├── __init__.py
│       │   ├── events.py
│       │   ├── exceptions.py
│       │   ├── imports.py
│       │   ├── input.py
│       │   ├── model.py
│       │   ├── pipeline.py
│       │   ├── pipeline_config.py
│       │   ├── result.py
│       │   ├── utils.py
│       │   └── workflow.py
│       ├── __init__.py
│       ├── _config.py
│       ├── _debug.py
│       ├── _run_impl.py
│       ├── agent.py
│       ├── agent_output.py
│       ├── computer.py
│       ├── exceptions.py
│       ├── function_schema.py
│       ├── guardrail.py
│       ├── handoffs.py
│       ├── items.py
│       ├── lifecycle.py
│       ├── logger.py
│       ├── model_settings.py
│       ├── py.typed
│       ├── result.py
│       ├── run.py
│       ├── run_context.py
│       ├── stream_events.py
│       ├── strict_schema.py
│       ├── tool.py
│       ├── usage.py
│       └── version.py
├── tests
│   ├── tracing
│   │   └── test_processor_api_key.py
│   ├── voice
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── fake_models.py
│   │   ├── helpers.py
│   │   ├── test_input.py
│   │   ├── test_openai_stt.py
│   │   ├── test_openai_tts.py
│   │   ├── test_pipeline.py
│   │   └── test_workflow.py
│   ├── README.md
│   ├── __init__.py
│   ├── conftest.py
│   ├── fake_model.py
│   ├── test_agent_config.py
│   ├── test_agent_hooks.py
│   ├── test_agent_runner.py
│   ├── test_agent_runner_streamed.py
│   ├── test_agent_tracing.py
│   ├── test_computer_action.py
│   ├── test_config.py
│   ├── test_doc_parsing.py
│   ├── test_extension_filters.py
│   ├── test_function_schema.py
│   ├── test_function_tool.py
│   ├── test_function_tool_decorator.py
│   ├── test_global_hooks.py
│   ├── test_guardrails.py
│   ├── test_handoff_tool.py
│   ├── test_items_helpers.py
│   ├── test_max_turns.py
│   ├── test_openai_chatcompletions.py
│   ├── test_openai_chatcompletions_converter.py
│   ├── test_openai_chatcompletions_stream.py
│   ├── test_openai_responses_converter.py
│   ├── test_output_tool.py
│   ├── test_pretty_print.py
│   ├── test_responses.py
│   ├── test_responses_tracing.py
│   ├── test_result_cast.py
│   ├── test_run_config.py
│   ├── test_run_step_execution.py
│   ├── test_run_step_processing.py
│   ├── test_strict_schema.py
│   ├── test_tool_choice_reset.py
│   ├── test_tool_converter.py
│   ├── test_tool_use_behavior.py
│   ├── test_trace_processor.py
│   ├── test_tracing.py
│   ├── test_tracing_errors.py
│   ├── test_tracing_errors_streamed.py
│   └── testing_processor.py
├── .gitignore
├── .prettierrc
├── LICENSE
├── Makefile
├── README.md
├── mkdocs.yml
├── pyproject.toml
└── uv.lock

</directory_structure>

<file_info>
path: README.md
name: README.md
</file_info>
# OpenAI Agents SDK

The OpenAI Agents SDK is a lightweight yet powerful framework for building multi-agent workflows.

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

### Core concepts:

1. [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs configured with instructions, tools, guardrails, and handoffs
2. [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): A specialized tool call used by the Agents SDK for transferring control between agents
3. [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation
4. [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

Explore the [examples](examples) directory to see the SDK in action, and read our [documentation](https://openai.github.io/openai-agents-python/) for more details.

Notably, our SDK [is compatible](https://openai.github.io/openai-agents-python/models/) with any model providers that support the OpenAI Chat Completions API format.

## Get started

1. Set up your Python environment

```
python -m venv env
source env/bin/activate
```

2. Install Agents SDK

```
pip install openai-agents
```

For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Hello world example

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

(_For Jupyter notebook users, see [hello_world_jupyter.py](examples/basic/hello_world_jupyter.py)_)

## Handoffs example

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())
```

## Functions example

```python
import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
```

## The agent loop

When you call `Runner.run()`, we run a loop until we get a final output.

1. We call the LLM, using the model and settings on the agent, and the message history.
2. The LLM returns a response, which may include tool calls.
3. If the response has a final output (see below for more on this), we return it and end the loop.
4. If the response has a handoff, we set the agent to the new agent and go back to step 1.
5. We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `max_turns` parameter that you can use to limit the number of times the loop executes.

### Final output

Final output is the last thing the agent produces in the loop.

1.  If you set an `output_type` on the agent, the final output is when the LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `output_type` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

As a result, the mental model for the agent loop is:

1. If the current agent has an `output_type`, the loop runs until the agent produces structured output matching that type.
2. If the current agent does not have an `output_type`, the loop runs until the current agent produces a message without any tool calls/handoffs.

## Common agent patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide range of LLM workflows including deterministic flows, iterative loops, and more. See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

The Agents SDK automatically traces your agent runs, making it easy to track and debug the behavior of your agents. Tracing is extensible by design, supporting custom spans and a wide variety of external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). For more details about how to customize or disable tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Development (only needed if you need to edit the SDK/examples)

0. Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv --version
```

1. Install dependencies

```bash
make sync
```

2. (After making changes) lint/test

```
make tests  # run tests
make mypy   # run typechecker
make lint   # run linter
```

## Acknowledgements

We'd like to acknowledge the excellent work of the open-source community, especially:

-   [Pydantic](https://docs.pydantic.dev/latest/) (data validation) and [PydanticAI](https://ai.pydantic.dev/) (advanced agent framework)
-   [MkDocs](https://github.com/squidfunk/mkdocs-material)
-   [Griffe](https://github.com/mkdocstrings/griffe)
-   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We're committed to continuing to build the Agents SDK as an open source framework so others in the community can expand on our approach.


<file_info>
path: tests/testing_processor.py
name: testing_processor.py
</file_info>
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Literal

from agents.tracing import Span, Trace, TracingProcessor

TestSpanProcessorEvent = Literal["trace_start", "trace_end", "span_start", "span_end"]


class SpanProcessorForTests(TracingProcessor):
    """
    A simple processor that stores finished spans in memory.
    This is thread-safe and suitable for tests or basic usage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Dictionary of trace_id -> list of spans
        self._spans: list[Span[Any]] = []
        self._traces: list[Trace] = []
        self._events: list[TestSpanProcessorEvent] = []

    def on_trace_start(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._events.append("trace_start")

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            # We don't append the trace here, we want to do that in on_trace_start
            self._events.append("trace_end")

    def on_span_start(self, span: Span[Any]) -> None:
        with self._lock:
            # Purposely not appending the span here, we want to do that in on_span_end
            self._events.append("span_start")

    def on_span_end(self, span: Span[Any]) -> None:
        with self._lock:
            self._events.append("span_end")
            self._spans.append(span)

    def get_ordered_spans(self, including_empty: bool = False) -> list[Span[Any]]:
        with self._lock:
            spans = [x for x in self._spans if including_empty or x.export()]
            return sorted(spans, key=lambda x: x.started_at or 0)

    def get_traces(self, including_empty: bool = False) -> list[Trace]:
        with self._lock:
            traces = [x for x in self._traces if including_empty or x.export()]
            return traces

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._traces.clear()
            self._events.clear()

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


SPAN_PROCESSOR_TESTING = SpanProcessorForTests()


def fetch_ordered_spans() -> list[Span[Any]]:
    return SPAN_PROCESSOR_TESTING.get_ordered_spans()


def fetch_traces() -> list[Trace]:
    return SPAN_PROCESSOR_TESTING.get_traces()


def fetch_events() -> list[TestSpanProcessorEvent]:
    return SPAN_PROCESSOR_TESTING._events


def assert_no_spans():
    spans = fetch_ordered_spans()
    if spans:
        raise AssertionError(f"Expected 0 spans, got {len(spans)}")


def assert_no_traces():
    traces = fetch_traces()
    if traces:
        raise AssertionError(f"Expected 0 traces, got {len(traces)}")
    assert_no_spans()


def fetch_normalized_spans(
    keep_span_id: bool = False, keep_trace_id: bool = False
) -> list[dict[str, Any]]:
    nodes: dict[tuple[str, str | None], dict[str, Any]] = {}
    traces = []
    for trace_obj in fetch_traces():
        trace = trace_obj.export()
        assert trace
        assert trace.pop("object") == "trace"
        assert trace["id"].startswith("trace_")
        if not keep_trace_id:
            del trace["id"]
        trace = {k: v for k, v in trace.items() if v is not None}
        nodes[(trace_obj.trace_id, None)] = trace
        traces.append(trace)

    assert traces, "Use assert_no_traces() to check for empty traces"

    for span_obj in fetch_ordered_spans():
        span = span_obj.export()
        assert span
        assert span.pop("object") == "trace.span"
        assert span["id"].startswith("span_")
        if not keep_span_id:
            del span["id"]
        assert datetime.fromisoformat(span.pop("started_at"))
        assert datetime.fromisoformat(span.pop("ended_at"))
        parent_id = span.pop("parent_id")
        assert "type" not in span
        span_data = span.pop("span_data")
        span = {"type": span_data.pop("type")} | {k: v for k, v in span.items() if v is not None}
        span_data = {k: v for k, v in span_data.items() if v is not None}
        if span_data:
            span["data"] = span_data
        nodes[(span_obj.trace_id, span_obj.span_id)] = span
        nodes[(span.pop("trace_id"), parent_id)].setdefault("children", []).append(span)
    return traces


<file_info>
path: tests/test_pretty_print.py
name: test_pretty_print.py
</file_info>
import json

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from agents import Agent, Runner
from agents.agent_output import _WRAPPER_DICT_KEY
from agents.util._pretty_print import pretty_print_result, pretty_print_run_result_streaming
from tests.fake_model import FakeModel

from .test_responses import get_final_output_message, get_text_message


@pytest.mark.asyncio
async def test_pretty_result():
    model = FakeModel()
    model.set_next_output([get_text_message("Hi there")])

    agent = Agent(name="test_agent", model=model)
    result = await Runner.run(agent, input="Hello")

    assert pretty_print_result(result) == snapshot("""\
RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (str):
    Hi there
- 1 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResult` for more details)\
""")


@pytest.mark.asyncio
async def test_pretty_run_result_streaming():
    model = FakeModel()
    model.set_next_output([get_text_message("Hi there")])

    agent = Agent(name="test_agent", model=model)
    result = Runner.run_streamed(agent, input="Hello")
    async for _ in result.stream_events():
        pass

    assert pretty_print_run_result_streaming(result) == snapshot("""\
RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: True
- Final output (str):
    Hi there
- 1 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResultStreaming` for more details)\
""")


class Foo(BaseModel):
    bar: str


@pytest.mark.asyncio
async def test_pretty_run_result_structured_output():
    model = FakeModel()
    model.set_next_output(
        [
            get_text_message("Test"),
            get_final_output_message(Foo(bar="Hi there").model_dump_json()),
        ]
    )

    agent = Agent(name="test_agent", model=model, output_type=Foo)
    result = await Runner.run(agent, input="Hello")

    assert pretty_print_result(result) == snapshot("""\
RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (Foo):
    {
      "bar": "Hi there"
    }
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResult` for more details)\
""")


@pytest.mark.asyncio
async def test_pretty_run_result_streaming_structured_output():
    model = FakeModel()
    model.set_next_output(
        [
            get_text_message("Test"),
            get_final_output_message(Foo(bar="Hi there").model_dump_json()),
        ]
    )

    agent = Agent(name="test_agent", model=model, output_type=Foo)
    result = Runner.run_streamed(agent, input="Hello")

    async for _ in result.stream_events():
        pass

    assert pretty_print_run_result_streaming(result) == snapshot("""\
RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: True
- Final output (Foo):
    {
      "bar": "Hi there"
    }
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResultStreaming` for more details)\
""")


@pytest.mark.asyncio
async def test_pretty_run_result_list_structured_output():
    model = FakeModel()
    model.set_next_output(
        [
            get_text_message("Test"),
            get_final_output_message(
                json.dumps(
                    {
                        _WRAPPER_DICT_KEY: [
                            Foo(bar="Hi there").model_dump(),
                            Foo(bar="Hi there 2").model_dump(),
                        ]
                    }
                )
            ),
        ]
    )

    agent = Agent(name="test_agent", model=model, output_type=list[Foo])
    result = await Runner.run(agent, input="Hello")

    assert pretty_print_result(result) == snapshot("""\
RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (list):
    [Foo(bar='Hi there'), Foo(bar='Hi there 2')]
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResult` for more details)\
""")


@pytest.mark.asyncio
async def test_pretty_run_result_streaming_list_structured_output():
    model = FakeModel()
    model.set_next_output(
        [
            get_text_message("Test"),
            get_final_output_message(
                json.dumps(
                    {
                        _WRAPPER_DICT_KEY: [
                            Foo(bar="Test").model_dump(),
                            Foo(bar="Test 2").model_dump(),
                        ]
                    }
                )
            ),
        ]
    )

    agent = Agent(name="test_agent", model=model, output_type=list[Foo])
    result = Runner.run_streamed(agent, input="Hello")

    async for _ in result.stream_events():
        pass

    assert pretty_print_run_result_streaming(result) == snapshot("""\
RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: True
- Final output (list):
    [Foo(bar='Test'), Foo(bar='Test 2')]
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `RunResultStreaming` for more details)\
""")


<file_info>
path: tests/voice/test_openai_tts.py
name: test_openai_tts.py
</file_info>
# Tests for the OpenAI text-to-speech model (OpenAITTSModel).

from types import SimpleNamespace
from typing import Any

import pytest

try:
    from agents.voice import OpenAITTSModel, TTSModelSettings
except ImportError:
    pass


class _FakeStreamResponse:
    """A minimal async context manager to simulate streaming audio bytes."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def iter_bytes(self, chunk_size: int = 1024):
        for chunk in self._chunks:
            yield chunk


def _make_fake_openai_client(fake_create) -> SimpleNamespace:
    """Construct an object with nested audio.speech.with_streaming_response.create."""
    return SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(with_streaming_response=SimpleNamespace(create=fake_create))
        )
    )


@pytest.mark.asyncio
async def test_openai_tts_default_voice_and_instructions() -> None:
    """If no voice is specified, OpenAITTSModel uses its default voice and passes instructions."""
    chunks = [b"abc", b"def"]
    captured: dict[str, object] = {}

    def fake_create(
        *, model: str, voice: str, input: str, response_format: str, extra_body: dict[str, Any]
    ) -> _FakeStreamResponse:
        captured["model"] = model
        captured["voice"] = voice
        captured["input"] = input
        captured["response_format"] = response_format
        captured["extra_body"] = extra_body
        return _FakeStreamResponse(chunks)

    client = _make_fake_openai_client(fake_create)
    tts_model = OpenAITTSModel(model="test-model", openai_client=client)  # type: ignore[arg-type]
    settings = TTSModelSettings()
    out: list[bytes] = []
    async for b in tts_model.run("hello world", settings):
        out.append(b)
    assert out == chunks
    assert captured["model"] == "test-model"
    assert captured["voice"] == "ash"
    assert captured["input"] == "hello world"
    assert captured["response_format"] == "pcm"
    assert captured["extra_body"] == {"instructions": settings.instructions}


@pytest.mark.asyncio
async def test_openai_tts_custom_voice_and_instructions() -> None:
    """Specifying voice and instructions are forwarded to the API."""
    chunks = [b"x"]
    captured: dict[str, object] = {}

    def fake_create(
        *, model: str, voice: str, input: str, response_format: str, extra_body: dict[str, Any]
    ) -> _FakeStreamResponse:
        captured["model"] = model
        captured["voice"] = voice
        captured["input"] = input
        captured["response_format"] = response_format
        captured["extra_body"] = extra_body
        return _FakeStreamResponse(chunks)

    client = _make_fake_openai_client(fake_create)
    tts_model = OpenAITTSModel(model="my-model", openai_client=client)  # type: ignore[arg-type]
    settings = TTSModelSettings(voice="fable", instructions="Custom instructions")
    out: list[bytes] = []
    async for b in tts_model.run("hi", settings):
        out.append(b)
    assert out == chunks
    assert captured["voice"] == "fable"
    assert captured["extra_body"] == {"instructions": "Custom instructions"}


<file_info>
path: tests/test_result_cast.py
name: test_result_cast.py
</file_info>
from typing import Any

import pytest
from pydantic import BaseModel

from agents import Agent, RunResult


def create_run_result(final_output: Any) -> RunResult:
    return RunResult(
        input="test",
        new_items=[],
        raw_responses=[],
        final_output=final_output,
        input_guardrail_results=[],
        output_guardrail_results=[],
        _last_agent=Agent(name="test"),
    )


class Foo(BaseModel):
    bar: int


def test_result_cast_typechecks():
    """Correct casts should work fine."""
    result = create_run_result(1)
    assert result.final_output_as(int) == 1

    result = create_run_result("test")
    assert result.final_output_as(str) == "test"

    result = create_run_result(Foo(bar=1))
    assert result.final_output_as(Foo) == Foo(bar=1)


def test_bad_cast_doesnt_raise():
    """Bad casts shouldn't error unless we ask for it."""
    result = create_run_result(1)
    result.final_output_as(str)

    result = create_run_result("test")
    result.final_output_as(Foo)


def test_bad_cast_with_param_raises():
    """Bad casts should raise a TypeError when we ask for it."""
    result = create_run_result(1)
    with pytest.raises(TypeError):
        result.final_output_as(str, raise_if_incorrect_type=True)

    result = create_run_result("test")
    with pytest.raises(TypeError):
        result.final_output_as(Foo, raise_if_incorrect_type=True)

    result = create_run_result(Foo(bar=1))
    with pytest.raises(TypeError):
        result.final_output_as(int, raise_if_incorrect_type=True)


<file_info>
path: tests/test_trace_processor.py
name: test_trace_processor.py
</file_info>
import os
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor
from agents.tracing.span_data import AgentSpanData
from agents.tracing.spans import SpanImpl
from agents.tracing.traces import TraceImpl


def get_span(processor: TracingProcessor) -> SpanImpl[AgentSpanData]:
    """Create a minimal agent span for testing processors."""
    return SpanImpl(
        trace_id="test_trace_id",
        span_id="test_span_id",
        parent_id=None,
        processor=processor,
        span_data=AgentSpanData(name="test_agent"),
    )


def get_trace(processor: TracingProcessor) -> TraceImpl:
    """Create a minimal trace."""
    return TraceImpl(
        name="test_trace",
        trace_id="test_trace_id",
        group_id="test_session_id",
        metadata={},
        processor=processor,
    )


@pytest.fixture
def mocked_exporter():
    exporter = MagicMock()
    exporter.export = MagicMock()
    return exporter


def test_batch_trace_processor_on_trace_start(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter, schedule_delay=0.1)
    test_trace = get_trace(processor)

    processor.on_trace_start(test_trace)
    assert processor._queue.qsize() == 1, "Trace should be added to the queue"

    # Shutdown to clean up the worker thread
    processor.shutdown()


def test_batch_trace_processor_on_span_end(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter, schedule_delay=0.1)
    test_span = get_span(processor)

    processor.on_span_end(test_span)
    assert processor._queue.qsize() == 1, "Span should be added to the queue"

    # Shutdown to clean up the worker thread
    processor.shutdown()


def test_batch_trace_processor_queue_full(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter, max_queue_size=2, schedule_delay=0.1)
    # Fill the queue
    processor.on_trace_start(get_trace(processor))
    processor.on_trace_start(get_trace(processor))
    assert processor._queue.full() is True

    # Next item should not be queued
    processor.on_trace_start(get_trace(processor))
    assert processor._queue.qsize() == 2, "Queue should not exceed max_queue_size"

    processor.on_span_end(get_span(processor))
    assert processor._queue.qsize() == 2, "Queue should not exceed max_queue_size"

    processor.shutdown()


def test_batch_processor_doesnt_enqueue_on_trace_end_or_span_start(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter)

    processor.on_trace_start(get_trace(processor))
    assert processor._queue.qsize() == 1, "Trace should be queued"

    processor.on_span_start(get_span(processor))
    assert processor._queue.qsize() == 1, "Span should not be queued"

    processor.on_span_end(get_span(processor))
    assert processor._queue.qsize() == 2, "Span should be queued"

    processor.on_trace_end(get_trace(processor))
    assert processor._queue.qsize() == 2, "Nothing new should be queued"

    processor.shutdown()


def test_batch_trace_processor_force_flush(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter, max_batch_size=2, schedule_delay=5.0)

    processor.on_trace_start(get_trace(processor))
    processor.on_span_end(get_span(processor))
    processor.on_span_end(get_span(processor))

    processor.force_flush()

    # Ensure exporter.export was called with all items
    # Because max_batch_size=2, it may have been called multiple times
    total_exported = 0
    for call_args in mocked_exporter.export.call_args_list:
        batch = call_args[0][0]  # first positional arg to export() is the items list
        total_exported += len(batch)

    # We pushed 3 items; ensure they all got exported
    assert total_exported == 3

    processor.shutdown()


def test_batch_trace_processor_shutdown_flushes(mocked_exporter):
    processor = BatchTraceProcessor(exporter=mocked_exporter, schedule_delay=5.0)
    processor.on_trace_start(get_trace(processor))
    processor.on_span_end(get_span(processor))
    qsize_before = processor._queue.qsize()
    assert qsize_before == 2

    processor.shutdown()

    # Ensure everything was exported after shutdown
    total_exported = 0
    for call_args in mocked_exporter.export.call_args_list:
        batch = call_args[0][0]
        total_exported += len(batch)

    assert total_exported == 2, "All items in the queue should be exported upon shutdown"


def test_batch_trace_processor_scheduled_export(mocked_exporter):
    """
    Tests that items are automatically exported when the schedule_delay expires.
    We mock time.time() so we can trigger the condition without waiting in real time.
    """
    with patch("time.time") as mock_time:
        base_time = 1000.0
        mock_time.return_value = base_time

        processor = BatchTraceProcessor(exporter=mocked_exporter, schedule_delay=1.0)

        processor.on_span_end(get_span(processor))  # queue size = 1

        # Now artificially advance time beyond the next export time
        mock_time.return_value = base_time + 2.0  # > base_time + schedule_delay
        # Let the background thread run a bit
        time.sleep(0.3)

        # Check that exporter.export was eventually called
        # Because the background thread runs, we might need a small sleep
        processor.shutdown()

    total_exported = 0
    for call_args in mocked_exporter.export.call_args_list:
        batch = call_args[0][0]
        total_exported += len(batch)

    assert total_exported == 1, "Item should be exported after scheduled delay"


@pytest.fixture
def patched_time_sleep():
    """
    Fixture to replace time.sleep with a no-op to speed up tests
    that rely on retry/backoff logic.
    """
    with patch("time.sleep") as mock_sleep:
        yield mock_sleep


def mock_processor():
    processor = MagicMock()
    processor.on_trace_start = MagicMock()
    processor.on_span_end = MagicMock()
    return processor


@patch("httpx.Client")
def test_backend_span_exporter_no_items(mock_client):
    exporter = BackendSpanExporter(api_key="test_key")
    exporter.export([])
    # No calls should be made if there are no items
    mock_client.return_value.post.assert_not_called()
    exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_no_api_key(mock_client):
    # Ensure that os.environ is empty (sometimes devs have the openai api key set in their env)

    with patch.dict(os.environ, {}, clear=True):
        exporter = BackendSpanExporter(api_key=None)
        exporter.export([get_span(mock_processor())])

        # Should log an error and return without calling post
        mock_client.return_value.post.assert_not_called()
        exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_2xx_success(mock_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.return_value.post.return_value = mock_response

    exporter = BackendSpanExporter(api_key="test_key")
    exporter.export([get_span(mock_processor()), get_trace(mock_processor())])

    # Should have called post exactly once
    mock_client.return_value.post.assert_called_once()
    exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_4xx_client_error(mock_client):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_client.return_value.post.return_value = mock_response

    exporter = BackendSpanExporter(api_key="test_key")
    exporter.export([get_span(mock_processor())])

    # 4xx should not be retried
    mock_client.return_value.post.assert_called_once()
    exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_5xx_retry(mock_client, patched_time_sleep):
    mock_response = MagicMock()
    mock_response.status_code = 500

    # Make post() return 500 every time
    mock_client.return_value.post.return_value = mock_response

    exporter = BackendSpanExporter(api_key="test_key", max_retries=3, base_delay=0.1, max_delay=0.2)
    exporter.export([get_span(mock_processor())])

    # Should retry up to max_retries times
    assert mock_client.return_value.post.call_count == 3

    exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_request_error(mock_client, patched_time_sleep):
    # Make post() raise a RequestError each time
    mock_client.return_value.post.side_effect = httpx.RequestError("Network error")

    exporter = BackendSpanExporter(api_key="test_key", max_retries=2, base_delay=0.1, max_delay=0.2)
    exporter.export([get_span(mock_processor())])

    # Should retry up to max_retries times
    assert mock_client.return_value.post.call_count == 2

    exporter.close()


@patch("httpx.Client")
def test_backend_span_exporter_close(mock_client):
    exporter = BackendSpanExporter(api_key="test_key")
    exporter.close()

    # Ensure underlying http client is closed
    mock_client.return_value.close.assert_called_once()


<file_info>
path: examples/tools/web_search.py
name: web_search.py
</file_info>
import asyncio

from agents import Agent, Runner, WebSearchTool, trace


async def main():
    agent = Agent(
        name="Web searcher",
        instructions="You are a helpful agent.",
        tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})],
    )

    with trace("Web search example"):
        result = await Runner.run(
            agent,
            "search the web for 'local sports news' and give me 1 interesting update in a sentence.",
        )
        print(result.final_output)
        # The New York Giants are reportedly pursuing quarterback Aaron Rodgers after his ...


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/financial_research_agent/agents/verifier_agent.py
name: verifier_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

# Agent to sanity‑check a synthesized report for consistency and recall.
# This can be used to flag potential gaps or obvious mistakes.
VERIFIER_PROMPT = (
    "You are a meticulous auditor. You have been handed a financial analysis report. "
    "Your job is to verify the report is internally consistent, clearly sourced, and makes "
    "no unsupported claims. Point out any issues or uncertainties."
)


class VerificationResult(BaseModel):
    verified: bool
    """Whether the report seems coherent and plausible."""

    issues: str
    """If not verified, describe the main issues or concerns."""


verifier_agent = Agent(
    name="VerificationAgent",
    instructions=VERIFIER_PROMPT,
    model="gpt-4o",
    output_type=VerificationResult,
)


<file_info>
path: examples/basic/hello_world_jupyter.py
name: hello_world_jupyter.py
</file_info>
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# Intended for Jupyter notebooks where there's an existing event loop
result = await Runner.run(agent, "Write a haiku about recursion in programming.")  # type: ignore[top-level-await]  # noqa: F704
print(result.final_output)

# Code within code loops,
# Infinite mirrors reflect—
# Logic folds on self.


<file_info>
path: examples/agent_patterns/agents_as_tools.py
name: agents_as_tools.py
</file_info>
import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
"""

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

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
)


async def main():
    msg = input("Hi! What would you like translated, and to which languages? ")

    # Run the entire orchestration in a single trace
    with trace("Orchestrator evaluator"):
        orchestrator_result = await Runner.run(orchestrator_agent, msg)

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")

        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/traces.py
name: traces.py
</file_info>
from __future__ import annotations

import abc
import contextvars
from typing import Any

from ..logger import logger
from . import util
from .processor_interface import TracingProcessor
from .scope import Scope


class Trace:
    """
    A trace is the root level object that tracing creates. It represents a logical "workflow".
    """

    @abc.abstractmethod
    def __enter__(self) -> Trace:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """
        Start the trace.

        Args:
            mark_as_current: If true, the trace will be marked as the current trace.
        """
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False):
        """
        Finish the trace.

        Args:
            reset_current: If true, the trace will be reset as the current trace.
        """
        pass

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """
        The trace ID.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the workflow being traced.
        """
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """
        Export the trace as a dictionary.
        """
        pass


class NoOpTrace(Trace):
    """
    A no-op trace that will not be recorded.
    """

    def __init__(self):
        self._started = False
        self._prev_context_token: contextvars.Token[Trace | None] | None = None

    def __enter__(self) -> Trace:
        if self._started:
            if not self._prev_context_token:
                logger.error("Trace already started but no context token set")
            return self

        self._started = True
        self.start(mark_as_current=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(reset_current=True)

    def start(self, mark_as_current: bool = False):
        if mark_as_current:
            self._prev_context_token = Scope.set_current_trace(self)

    def finish(self, reset_current: bool = False):
        if reset_current and self._prev_context_token is not None:
            Scope.reset_current_trace(self._prev_context_token)
            self._prev_context_token = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def name(self) -> str:
        return "no-op"

    def export(self) -> dict[str, Any] | None:
        return None


NO_OP_TRACE = NoOpTrace()


class TraceImpl(Trace):
    """
    A trace that will be recorded by the tracing library.
    """

    __slots__ = (
        "_name",
        "_trace_id",
        "group_id",
        "metadata",
        "_prev_context_token",
        "_processor",
        "_started",
    )

    def __init__(
        self,
        name: str,
        trace_id: str | None,
        group_id: str | None,
        metadata: dict[str, Any] | None,
        processor: TracingProcessor,
    ):
        self._name = name
        self._trace_id = trace_id or util.gen_trace_id()
        self.group_id = group_id
        self.metadata = metadata
        self._prev_context_token: contextvars.Token[Trace | None] | None = None
        self._processor = processor
        self._started = False

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    def start(self, mark_as_current: bool = False):
        if self._started:
            return

        self._started = True
        self._processor.on_trace_start(self)

        if mark_as_current:
            self._prev_context_token = Scope.set_current_trace(self)

    def finish(self, reset_current: bool = False):
        if not self._started:
            return

        self._processor.on_trace_end(self)

        if reset_current and self._prev_context_token is not None:
            Scope.reset_current_trace(self._prev_context_token)
            self._prev_context_token = None

    def __enter__(self) -> Trace:
        if self._started:
            if not self._prev_context_token:
                logger.error("Trace already started but no context token set")
            return self

        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(reset_current=exc_type is not GeneratorExit)

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace",
            "id": self.trace_id,
            "workflow_name": self.name,
            "group_id": self.group_id,
            "metadata": self.metadata,
        }


<file_info>
path: src/agents/run_context.py
name: run_context.py
</file_info>
from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

from .usage import Usage

TContext = TypeVar("TContext", default=Any)


@dataclass
class RunContextWrapper(Generic[TContext]):
    """This wraps the context object that you passed to `Runner.run()`. It also contains
    information about the usage of the agent run so far.

    NOTE: Contexts are not passed to the LLM. They're a way to pass dependencies and data to code
    you implement, like tool functions, callbacks, hooks, etc.
    """

    context: TContext
    """The context object (or None), passed by you to `Runner.run()`"""

    usage: Usage = field(default_factory=Usage)
    """The usage of the agent run so far. For streamed responses, the usage will be stale until the
    last chunk of the stream is processed.
    """


<file_info>
path: src/agents/agent_output.py
name: agent_output.py
</file_info>
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict, get_args, get_origin

from .exceptions import ModelBehaviorError, UserError
from .strict_schema import ensure_strict_json_schema
from .tracing import SpanError
from .util import _error_tracing, _json

_WRAPPER_DICT_KEY = "response"


@dataclass(init=False)
class AgentOutputSchema:
    """An object that captures the JSON schema of the output, as well as validating/parsing JSON
    produced by the LLM into the output type.
    """

    output_type: type[Any]
    """The type of the output."""

    _type_adapter: TypeAdapter[Any]
    """A type adapter that wraps the output type, so that we can validate JSON."""

    _is_wrapped: bool
    """Whether the output type is wrapped in a dictionary. This is generally done if the base
    output type cannot be represented as a JSON Schema object.
    """

    _output_schema: dict[str, Any]
    """The JSON schema of the output."""

    strict_json_schema: bool
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input.
    """

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """
        Args:
            output_type: The type of the output.
            strict_json_schema: Whether the JSON schema is in strict mode. We **strongly** recommend
                setting this to True, as it increases the likelihood of correct JSON input.
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema

        if output_type is None or output_type is str:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()
            return

        # We should wrap for things that are not plain text, and for things that would definitely
        # not be a JSON Schema object.
        self._is_wrapped = not _is_subclass_of_base_model_or_dict(output_type)

        if self._is_wrapped:
            OutputType = TypedDict(
                "OutputType",
                {
                    _WRAPPER_DICT_KEY: output_type,  # type: ignore
                },
            )
            self._type_adapter = TypeAdapter(OutputType)
            self._output_schema = self._type_adapter.json_schema()
        else:
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()

        if self.strict_json_schema:
            self._output_schema = ensure_strict_json_schema(self._output_schema)

    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        return self.output_type is None or self.output_type is str

    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of the output type."""
        if self.is_plain_text():
            raise UserError("Output type is plain text, so no JSON schema is available")
        return self._output_schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        """Validate a JSON string against the output type. Returns the validated object, or raises
        a `ModelBehaviorError` if the JSON is invalid.
        """
        validated = _json.validate_json(json_str, self._type_adapter, partial)
        if self._is_wrapped:
            if not isinstance(validated, dict):
                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Invalid JSON",
                        data={"details": f"Expected a dict, got {type(validated)}"},
                    )
                )
                raise ModelBehaviorError(
                    f"Expected a dict, got {type(validated)} for JSON: {json_str}"
                )

            if _WRAPPER_DICT_KEY not in validated:
                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Invalid JSON",
                        data={"details": f"Could not find key {_WRAPPER_DICT_KEY} in JSON"},
                    )
                )
                raise ModelBehaviorError(
                    f"Could not find key {_WRAPPER_DICT_KEY} in JSON: {json_str}"
                )
            return validated[_WRAPPER_DICT_KEY]
        return validated

    def output_type_name(self) -> str:
        """The name of the output type."""
        return _type_to_str(self.output_type)


def _is_subclass_of_base_model_or_dict(t: Any) -> bool:
    if not isinstance(t, type):
        return False

    # If it's a generic alias, 'origin' will be the actual type, e.g. 'list'
    origin = get_origin(t)

    allowed_types = (BaseModel, dict)
    # If it's a generic alias e.g. list[str], then we should check the origin type i.e. list
    return issubclass(origin or t, allowed_types)


def _type_to_str(t: type[Any]) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        # It's a simple type like `str`, `int`, etc.
        return t.__name__
    elif args:
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin.__name__}[{args_str}]"
    else:
        return str(t)


<file_info>
path: tests/test_tracing.py
name: test_tracing.py
</file_info>
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inline_snapshot import snapshot

from agents.tracing import (
    Span,
    Trace,
    agent_span,
    custom_span,
    function_span,
    generation_span,
    handoff_span,
    trace,
)
from agents.tracing.spans import SpanError

from .testing_processor import (
    SPAN_PROCESSOR_TESTING,
    assert_no_traces,
    fetch_events,
    fetch_normalized_spans,
)

### HELPERS


def standard_span_checks(
    span: Span[Any], trace_id: str, parent_id: str | None, span_type: str
) -> None:
    assert span.span_id is not None
    assert span.trace_id == trace_id
    assert span.parent_id == parent_id
    assert span.started_at is not None
    assert span.ended_at is not None
    assert span.span_data.type == span_type


def standard_trace_checks(trace: Trace, name_check: str | None = None) -> None:
    assert trace.trace_id is not None

    if name_check:
        assert trace.name == name_check


### TESTS


def simple_tracing():
    x = trace("test")
    x.start()

    span_1 = agent_span(name="agent_1", span_id="span_1", parent=x)
    span_1.start()
    span_1.finish()

    span_2 = custom_span(name="custom_1", span_id="span_2", parent=x)
    span_2.start()

    span_3 = custom_span(name="custom_2", span_id="span_3", parent=span_2)
    span_3.start()
    span_3.finish()

    span_2.finish()

    x.finish()


def test_simple_tracing() -> None:
    simple_tracing()

    assert fetch_normalized_spans(keep_span_id=True) == snapshot(
        [
            {
                "workflow_name": "test",
                "children": [
                    {
                        "type": "agent",
                        "id": "span_1",
                        "data": {"name": "agent_1"},
                    },
                    {
                        "type": "custom",
                        "id": "span_2",
                        "data": {"name": "custom_1", "data": {}},
                        "children": [
                            {
                                "type": "custom",
                                "id": "span_3",
                                "data": {"name": "custom_2", "data": {}},
                            }
                        ],
                    },
                ],
            }
        ]
    )


def ctxmanager_spans():
    with trace(workflow_name="test", trace_id="trace_123", group_id="456"):
        with custom_span(name="custom_1", span_id="span_1"):
            with custom_span(name="custom_2", span_id="span_1_inner"):
                pass

        with custom_span(name="custom_2", span_id="span_2"):
            pass


def test_ctxmanager_spans() -> None:
    ctxmanager_spans()

    assert fetch_normalized_spans(keep_span_id=True) == snapshot(
        [
            {
                "workflow_name": "test",
                "group_id": "456",
                "children": [
                    {
                        "type": "custom",
                        "id": "span_1",
                        "data": {"name": "custom_1", "data": {}},
                        "children": [
                            {
                                "type": "custom",
                                "id": "span_1_inner",
                                "data": {"name": "custom_2", "data": {}},
                            }
                        ],
                    },
                    {"type": "custom", "id": "span_2", "data": {"name": "custom_2", "data": {}}},
                ],
            }
        ]
    )


async def run_subtask(span_id: str | None = None) -> None:
    with generation_span(span_id=span_id):
        await asyncio.sleep(0.0001)


async def simple_async_tracing():
    with trace(workflow_name="test", trace_id="trace_123", group_id="group_456"):
        await run_subtask(span_id="span_1")
        await run_subtask(span_id="span_2")


@pytest.mark.asyncio
async def test_async_tracing() -> None:
    await simple_async_tracing()

    assert fetch_normalized_spans(keep_span_id=True) == snapshot(
        [
            {
                "workflow_name": "test",
                "group_id": "group_456",
                "children": [
                    {"type": "generation", "id": "span_1"},
                    {"type": "generation", "id": "span_2"},
                ],
            }
        ]
    )


async def run_tasks_parallel(span_ids: list[str]) -> None:
    await asyncio.gather(
        *[run_subtask(span_id=span_id) for span_id in span_ids],
    )


async def run_tasks_as_children(first_span_id: str, second_span_id: str) -> None:
    with generation_span(span_id=first_span_id):
        await run_subtask(span_id=second_span_id)


async def complex_async_tracing():
    with trace(workflow_name="test", trace_id="trace_123", group_id="456"):
        await asyncio.gather(
            run_tasks_parallel(["span_1", "span_2"]),
            run_tasks_parallel(["span_3", "span_4"]),
        )
        await asyncio.gather(
            run_tasks_as_children("span_5", "span_6"),
            run_tasks_as_children("span_7", "span_8"),
        )


@pytest.mark.asyncio
async def test_complex_async_tracing() -> None:
    for _ in range(300):
        SPAN_PROCESSOR_TESTING.clear()
        await complex_async_tracing()

        assert fetch_normalized_spans(keep_span_id=True) == (
            [
                {
                    "workflow_name": "test",
                    "group_id": "456",
                    "children": [
                        {"type": "generation", "id": "span_1"},
                        {"type": "generation", "id": "span_2"},
                        {"type": "generation", "id": "span_3"},
                        {"type": "generation", "id": "span_4"},
                        {
                            "type": "generation",
                            "id": "span_5",
                            "children": [{"type": "generation", "id": "span_6"}],
                        },
                        {
                            "type": "generation",
                            "id": "span_7",
                            "children": [{"type": "generation", "id": "span_8"}],
                        },
                    ],
                }
            ]
        )


def spans_with_setters():
    with trace(workflow_name="test", trace_id="trace_123", group_id="456"):
        with agent_span(name="agent_1") as span_a:
            span_a.span_data.name = "agent_2"

            with function_span(name="function_1") as span_b:
                span_b.span_data.input = "i"
                span_b.span_data.output = "o"

            with generation_span() as span_c:
                span_c.span_data.input = [{"foo": "bar"}]

            with handoff_span(from_agent="agent_1", to_agent="agent_2"):
                pass


def test_spans_with_setters() -> None:
    spans_with_setters()

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test",
                "group_id": "456",
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "agent_2"},
                        "children": [
                            {
                                "type": "function",
                                "data": {"name": "function_1", "input": "i", "output": "o"},
                            },
                            {
                                "type": "generation",
                                "data": {"input": [{"foo": "bar"}]},
                            },
                            {
                                "type": "handoff",
                                "data": {"from_agent": "agent_1", "to_agent": "agent_2"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


def disabled_tracing():
    with trace(workflow_name="test", trace_id="123", group_id="456", disabled=True):
        with agent_span(name="agent_1"):
            with function_span(name="function_1"):
                pass


def test_disabled_tracing():
    disabled_tracing()
    assert_no_traces()


def enabled_trace_disabled_span():
    with trace(workflow_name="test", trace_id="trace_123"):
        with agent_span(name="agent_1"):
            with function_span(name="function_1", disabled=True):
                with generation_span():
                    pass


def test_enabled_trace_disabled_span():
    enabled_trace_disabled_span()

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test",
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "agent_1"},
                    }
                ],
            }
        ]
    )


def test_start_and_end_called_manual():
    simple_tracing()

    events = fetch_events()

    assert events == [
        "trace_start",
        "span_start",  # span_1
        "span_end",  # span_1
        "span_start",  # span_2
        "span_start",  # span_3
        "span_end",  # span_3
        "span_end",  # span_2
        "trace_end",
    ]


def test_start_and_end_called_ctxmanager():
    with trace(workflow_name="test", trace_id="123", group_id="456"):
        with custom_span(name="custom_1", span_id="span_1"):
            with custom_span(name="custom_2", span_id="span_1_inner"):
                pass

        with custom_span(name="custom_2", span_id="span_2"):
            pass

    events = fetch_events()

    assert events == [
        "trace_start",
        "span_start",  # span_1
        "span_start",  # span_1_inner
        "span_end",  # span_1_inner
        "span_end",  # span_1
        "span_start",  # span_2
        "span_end",  # span_2
        "trace_end",
    ]


@pytest.mark.asyncio
async def test_start_and_end_called_async_ctxmanager():
    await simple_async_tracing()

    events = fetch_events()

    assert events == [
        "trace_start",
        "span_start",  # span_1
        "span_end",  # span_1
        "span_start",  # span_2
        "span_end",  # span_2
        "trace_end",
    ]


async def test_noop_span_doesnt_record():
    with trace(workflow_name="test", disabled=True) as t:
        with custom_span(name="span_1") as span:
            span.set_error(SpanError(message="test", data={}))

    assert_no_traces()

    assert t.export() is None
    assert span.export() is None
    assert span.started_at is None
    assert span.ended_at is None
    assert span.error is None


async def test_multiple_span_start_finish_doesnt_crash():
    with trace(workflow_name="test", trace_id="123", group_id="456"):
        with custom_span(name="span_1") as span:
            span.start()

        span.finish()


async def test_noop_parent_is_noop_child():
    tr = trace(workflow_name="test", disabled=True)

    span = custom_span(name="span_1", parent=tr)
    span.start()
    span.finish()

    assert span.export() is None

    span_2 = custom_span(name="span_2", parent=span)
    span_2.start()
    span_2.finish()

    assert span_2.export() is None


<file_info>
path: tests/test_computer_action.py
name: test_computer_action.py
</file_info>
"""Unit tests for the ComputerAction methods in `agents._run_impl`.

These confirm that the correct computer action method is invoked for each action type and
that screenshots are taken and wrapped appropriately, and that the execute function invokes
hooks and returns the expected ToolCallOutputItem."""

from typing import Any

import pytest
from openai.types.responses.response_computer_tool_call import (
    ActionClick,
    ActionDoubleClick,
    ActionDrag,
    ActionDragPath,
    ActionKeypress,
    ActionMove,
    ActionScreenshot,
    ActionScroll,
    ActionType,
    ActionWait,
    ResponseComputerToolCall,
)

from agents import (
    Agent,
    AgentHooks,
    AsyncComputer,
    Computer,
    ComputerTool,
    RunConfig,
    RunContextWrapper,
    RunHooks,
)
from agents._run_impl import ComputerAction, ToolRunComputerAction
from agents.items import ToolCallOutputItem


class LoggingComputer(Computer):
    """A `Computer` implementation that logs calls to its methods for verification in tests."""

    def __init__(self, screenshot_return: str = "screenshot"):
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._screenshot_return = screenshot_return

    @property
    def environment(self):
        return "mac"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (800, 600)

    def screenshot(self) -> str:
        self.calls.append(("screenshot", ()))
        return self._screenshot_return

    def click(self, x: int, y: int, button: str) -> None:
        self.calls.append(("click", (x, y, button)))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.calls.append(("scroll", (x, y, scroll_x, scroll_y)))

    def type(self, text: str) -> None:
        self.calls.append(("type", (text,)))

    def wait(self) -> None:
        self.calls.append(("wait", ()))

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", (x, y)))

    def keypress(self, keys: list[str]) -> None:
        self.calls.append(("keypress", (keys,)))

    def drag(self, path: list[tuple[int, int]]) -> None:
        self.calls.append(("drag", (tuple(path),)))


class LoggingAsyncComputer(AsyncComputer):
    """An `AsyncComputer` implementation that logs calls to its methods for verification."""

    def __init__(self, screenshot_return: str = "async_screenshot"):
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._screenshot_return = screenshot_return

    @property
    def environment(self):
        return "mac"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (800, 600)

    async def screenshot(self) -> str:
        self.calls.append(("screenshot", ()))
        return self._screenshot_return

    async def click(self, x: int, y: int, button: str) -> None:
        self.calls.append(("click", (x, y, button)))

    async def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.calls.append(("scroll", (x, y, scroll_x, scroll_y)))

    async def type(self, text: str) -> None:
        self.calls.append(("type", (text,)))

    async def wait(self) -> None:
        self.calls.append(("wait", ()))

    async def move(self, x: int, y: int) -> None:
        self.calls.append(("move", (x, y)))

    async def keypress(self, keys: list[str]) -> None:
        self.calls.append(("keypress", (keys,)))

    async def drag(self, path: list[tuple[int, int]]) -> None:
        self.calls.append(("drag", (tuple(path),)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,expected_call",
    [
        (ActionClick(type="click", x=10, y=21, button="left"), ("click", (10, 21, "left"))),
        (ActionDoubleClick(type="double_click", x=42, y=47), ("double_click", (42, 47))),
        (
            ActionDrag(type="drag", path=[ActionDragPath(x=1, y=2), ActionDragPath(x=3, y=4)]),
            ("drag", (((1, 2), (3, 4)),)),
        ),
        (ActionKeypress(type="keypress", keys=["a", "b"]), ("keypress", (["a", "b"],))),
        (ActionMove(type="move", x=100, y=200), ("move", (100, 200))),
        (ActionScreenshot(type="screenshot"), ("screenshot", ())),
        (
            ActionScroll(type="scroll", x=1, y=2, scroll_x=3, scroll_y=4),
            ("scroll", (1, 2, 3, 4)),
        ),
        (ActionType(type="type", text="hello"), ("type", ("hello",))),
        (ActionWait(type="wait"), ("wait", ())),
    ],
)
async def test_get_screenshot_sync_executes_action_and_takes_screenshot(
    action: Any, expected_call: tuple[str, tuple[Any, ...]]
) -> None:
    """For each action type, assert that the corresponding computer method is invoked
    and that a screenshot is taken and returned."""
    computer = LoggingComputer(screenshot_return="synthetic")
    tool_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=action,
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    screenshot_output = await ComputerAction._get_screenshot_sync(computer, tool_call)
    # The last call is always to screenshot()
    if isinstance(action, ActionScreenshot):
        # Screenshot is taken twice: initial explicit call plus final capture.
        assert computer.calls == [("screenshot", ()), ("screenshot", ())]
    else:
        assert computer.calls == [expected_call, ("screenshot", ())]
    assert screenshot_output == "synthetic"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,expected_call",
    [
        (ActionClick(type="click", x=2, y=3, button="right"), ("click", (2, 3, "right"))),
        (ActionDoubleClick(type="double_click", x=12, y=13), ("double_click", (12, 13))),
        (
            ActionDrag(type="drag", path=[ActionDragPath(x=5, y=6), ActionDragPath(x=6, y=7)]),
            ("drag", (((5, 6), (6, 7)),)),
        ),
        (ActionKeypress(type="keypress", keys=["ctrl", "c"]), ("keypress", (["ctrl", "c"],))),
        (ActionMove(type="move", x=8, y=9), ("move", (8, 9))),
        (ActionScreenshot(type="screenshot"), ("screenshot", ())),
        (
            ActionScroll(type="scroll", x=9, y=8, scroll_x=7, scroll_y=6),
            ("scroll", (9, 8, 7, 6)),
        ),
        (ActionType(type="type", text="world"), ("type", ("world",))),
        (ActionWait(type="wait"), ("wait", ())),
    ],
)
async def test_get_screenshot_async_executes_action_and_takes_screenshot(
    action: Any, expected_call: tuple[str, tuple[Any, ...]]
) -> None:
    """For each action type on an `AsyncComputer`, the corresponding coroutine should be awaited
    and a screenshot taken."""
    computer = LoggingAsyncComputer(screenshot_return="async_return")
    assert computer.environment == "mac"
    assert computer.dimensions == (800, 600)
    tool_call = ResponseComputerToolCall(
        id="c2",
        type="computer_call",
        action=action,
        call_id="c2",
        pending_safety_checks=[],
        status="completed",
    )
    screenshot_output = await ComputerAction._get_screenshot_async(computer, tool_call)
    if isinstance(action, ActionScreenshot):
        assert computer.calls == [("screenshot", ()), ("screenshot", ())]
    else:
        assert computer.calls == [expected_call, ("screenshot", ())]
    assert screenshot_output == "async_return"


class LoggingRunHooks(RunHooks[Any]):
    """Capture on_tool_start and on_tool_end invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.started: list[tuple[Agent[Any], Any]] = []
        self.ended: list[tuple[Agent[Any], Any, str]] = []

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Any
    ) -> None:
        self.started.append((agent, tool))

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Any, result: str
    ) -> None:
        self.ended.append((agent, tool, result))


class LoggingAgentHooks(AgentHooks[Any]):
    """Minimal override to capture agent's tool hook invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.started: list[tuple[Agent[Any], Any]] = []
        self.ended: list[tuple[Agent[Any], Any, str]] = []

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Any
    ) -> None:
        self.started.append((agent, tool))

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Any, result: str
    ) -> None:
        self.ended.append((agent, tool, result))


@pytest.mark.asyncio
async def test_execute_invokes_hooks_and_returns_tool_call_output() -> None:
    # ComputerAction.execute should invoke lifecycle hooks and return a proper ToolCallOutputItem.
    computer = LoggingComputer(screenshot_return="xyz")
    comptool = ComputerTool(computer=computer)
    # Create a dummy click action to trigger a click and screenshot.
    action = ActionClick(type="click", x=1, y=2, button="left")
    tool_call = ResponseComputerToolCall(
        id="tool123",
        type="computer_call",
        action=action,
        call_id="tool123",
        pending_safety_checks=[],
        status="completed",
    )
    tool_call.call_id = "tool123"

    # Wrap tool call in ToolRunComputerAction
    tool_run = ToolRunComputerAction(tool_call=tool_call, computer_tool=comptool)
    # Setup agent and hooks.
    agent = Agent(name="test_agent", tools=[comptool])
    # Attach per-agent hooks as well as global run hooks.
    agent_hooks = LoggingAgentHooks()
    agent.hooks = agent_hooks
    run_hooks = LoggingRunHooks()
    context_wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)
    # Execute the computer action.
    output_item = await ComputerAction.execute(
        agent=agent,
        action=tool_run,
        hooks=run_hooks,
        context_wrapper=context_wrapper,
        config=RunConfig(),
    )
    # Both global and per-agent hooks should have been called once.
    assert len(run_hooks.started) == 1 and len(agent_hooks.started) == 1
    assert len(run_hooks.ended) == 1 and len(agent_hooks.ended) == 1
    # The hook invocations should refer to our agent and tool.
    assert run_hooks.started[0][0] is agent
    assert run_hooks.ended[0][0] is agent
    assert run_hooks.started[0][1] is comptool
    assert run_hooks.ended[0][1] is comptool
    # The result passed to on_tool_end should be the raw screenshot string.
    assert run_hooks.ended[0][2] == "xyz"
    assert agent_hooks.ended[0][2] == "xyz"
    # The computer should have performed a click then a screenshot.
    assert computer.calls == [("click", (1, 2, "left")), ("screenshot", ())]
    # The returned item should include the agent, output string, and a ComputerCallOutput.
    assert output_item.agent is agent
    assert isinstance(output_item, ToolCallOutputItem)
    assert output_item.output == "data:image/png;base64,xyz"
    raw = output_item.raw_item
    # Raw item is a dict-like mapping with expected output fields.
    assert isinstance(raw, dict)
    assert raw["type"] == "computer_call_output"
    assert raw["output"]["type"] == "computer_screenshot"
    assert "image_url" in raw["output"]
    assert raw["output"]["image_url"].endswith("xyz")


<file_info>
path: examples/tools/computer_use.py
name: computer_use.py
</file_info>
import asyncio
import base64
from typing import Literal, Union

from playwright.async_api import Browser, Page, Playwright, async_playwright

from agents import (
    Agent,
    AsyncComputer,
    Button,
    ComputerTool,
    Environment,
    ModelSettings,
    Runner,
    trace,
)

# Uncomment to see very verbose logs
# import logging
# logging.getLogger("openai.agents").setLevel(logging.DEBUG)
# logging.getLogger("openai.agents").addHandler(logging.StreamHandler())


async def main():
    async with LocalPlaywrightComputer() as computer:
        with trace("Computer use example"):
            agent = Agent(
                name="Browser user",
                instructions="You are a helpful agent.",
                tools=[ComputerTool(computer)],
                # Use the computer using model, and set truncation to auto because its required
                model="computer-use-preview",
                model_settings=ModelSettings(truncation="auto"),
            )
            result = await Runner.run(agent, "Search for SF sports news and summarize.")
            print(result.final_output)


CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}


class LocalPlaywrightComputer(AsyncComputer):
    """A computer, implemented using a local Playwright browser."""

    def __init__(self):
        self._playwright: Union[Playwright, None] = None
        self._browser: Union[Browser, None] = None
        self._page: Union[Page, None] = None

    async def _get_browser_and_page(self) -> tuple[Browser, Page]:
        width, height = self.dimensions
        launch_args = [f"--window-size={width},{height}"]
        browser = await self.playwright.chromium.launch(headless=False, args=launch_args)
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        await page.goto("https://www.bing.com")
        return browser, page

    async def __aenter__(self):
        # Start Playwright and call the subclass hook for getting browser/page
        self._playwright = await async_playwright().start()
        self._browser, self._page = await self._get_browser_and_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @property
    def playwright(self) -> Playwright:
        assert self._playwright is not None
        return self._playwright

    @property
    def browser(self) -> Browser:
        assert self._browser is not None
        return self._browser

    @property
    def page(self) -> Page:
        assert self._page is not None
        return self._page

    @property
    def environment(self) -> Environment:
        return "browser"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (1024, 768)

    async def screenshot(self) -> str:
        """Capture only the viewport (not full_page)."""
        png_bytes = await self.page.screenshot(full_page=False)
        return base64.b64encode(png_bytes).decode("utf-8")

    async def click(self, x: int, y: int, button: Button = "left") -> None:
        playwright_button: Literal["left", "middle", "right"] = "left"

        # Playwright only supports left, middle, right buttons
        if button in ("left", "right", "middle"):
            playwright_button = button  # type: ignore

        await self.page.mouse.click(x, y, button=playwright_button)

    async def double_click(self, x: int, y: int) -> None:
        await self.page.mouse.dblclick(x, y)

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self.page.mouse.move(x, y)
        await self.page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    async def type(self, text: str) -> None:
        await self.page.keyboard.type(text)

    async def wait(self) -> None:
        await asyncio.sleep(1)

    async def move(self, x: int, y: int) -> None:
        await self.page.mouse.move(x, y)

    async def keypress(self, keys: list[str]) -> None:
        for key in keys:
            mapped_key = CUA_KEY_TO_PLAYWRIGHT_KEY.get(key.lower(), key)
            await self.page.keyboard.press(mapped_key)

    async def drag(self, path: list[tuple[int, int]]) -> None:
        if not path:
            return
        await self.page.mouse.move(path[0][0], path[0][1])
        await self.page.mouse.down()
        for px, py in path[1:]:
            await self.page.mouse.move(px, py)
        await self.page.mouse.up()


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/research_bot/agents/__init__.py
name: __init__.py
</file_info>


<file_info>
path: examples/agent_patterns/output_guardrails.py
name: output_guardrails.py
</file_info>
from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)

"""
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the agent's response contains
a phone number.
"""


# The agent's output type
class MessageOutput(BaseModel):
    reasoning: str = Field(description="Thoughts on how to respond to the user's message")
    response: str = Field(description="The response to the user's message")
    user_name: str | None = Field(description="The name of the user who sent the message, if known")


@output_guardrail
async def sensitive_data_check(
    context: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    phone_number_in_response = "650" in output.response
    phone_number_in_reasoning = "650" in output.reasoning

    return GuardrailFunctionOutput(
        output_info={
            "phone_number_in_response": phone_number_in_response,
            "phone_number_in_reasoning": phone_number_in_reasoning,
        },
        tripwire_triggered=phone_number_in_response or phone_number_in_reasoning,
    )


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[sensitive_data_check],
)


async def main():
    # This should be ok
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed")

    # This should trip the guardrail
    try:
        result = await Runner.run(
            agent, "My phone number is 650-123-4567. Where do you think I live?"
        )
        print(
            f"Guardrail didn't trip - this is unexpected. Output: {json.dumps(result.final_output.model_dump(), indent=2)}"
        )

    except OutputGuardrailTripwireTriggered as e:
        print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/processors.py
name: processors.py
</file_info>
from __future__ import annotations

import os
import queue
import random
import threading
import time
from functools import cached_property
from typing import Any

import httpx

from ..logger import logger
from .processor_interface import TracingExporter, TracingProcessor
from .spans import Span
from .traces import Trace


class ConsoleSpanExporter(TracingExporter):
    """Prints the traces and spans to the console."""

    def export(self, items: list[Trace | Span[Any]]) -> None:
        for item in items:
            if isinstance(item, Trace):
                print(f"[Exporter] Export trace_id={item.trace_id}, name={item.name}, ")
            else:
                print(f"[Exporter] Export span: {item.export()}")


class BackendSpanExporter(TracingExporter):
    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        endpoint: str = "https://api.openai.com/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """
        Args:
            api_key: The API key for the "Authorization" header. Defaults to
                `os.environ["OPENAI_API_KEY"]` if not provided.
            organization: The OpenAI organization to use. Defaults to
                `os.environ["OPENAI_ORG_ID"]` if not provided.
            project: The OpenAI project to use. Defaults to
                `os.environ["OPENAI_PROJECT_ID"]` if not provided.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
        """
        self._api_key = api_key
        self._organization = organization
        self._project = project
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Keep a client open for connection pooling across multiple export calls
        self._client = httpx.Client(timeout=httpx.Timeout(timeout=60, connect=5.0))

    def set_api_key(self, api_key: str):
        """Set the OpenAI API key for the exporter.

        Args:
            api_key: The OpenAI API key to use. This is the same key used by the OpenAI Python
                client.
        """
        # We're specifically setting the underlying cached property as well
        self._api_key = api_key
        self.api_key = api_key

    @cached_property
    def api_key(self):
        return self._api_key or os.environ.get("OPENAI_API_KEY")

    @cached_property
    def organization(self):
        return self._organization or os.environ.get("OPENAI_ORG_ID")

    @cached_property
    def project(self):
        return self._project or os.environ.get("OPENAI_PROJECT_ID")

    def export(self, items: list[Trace | Span[Any]]) -> None:
        if not items:
            return

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set, skipping trace export")
            return

        data = [item.export() for item in items if item.export()]
        payload = {"data": data}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "traces=v1",
        }

        # Exponential backoff loop
        attempt = 0
        delay = self.base_delay
        while True:
            attempt += 1
            try:
                response = self._client.post(url=self.endpoint, headers=headers, json=payload)

                # If the response is successful, break out of the loop
                if response.status_code < 300:
                    logger.debug(f"Exported {len(items)} items")
                    return

                # If the response is a client error (4xx), we wont retry
                if 400 <= response.status_code < 500:
                    logger.error(
                        f"[non-fatal] Tracing client error {response.status_code}: {response.text}"
                    )
                    return

                # For 5xx or other unexpected codes, treat it as transient and retry
                logger.warning(
                    f"[non-fatal] Tracing: server error {response.status_code}, retrying."
                )
            except httpx.RequestError as exc:
                # Network or other I/O error, we'll retry
                logger.warning(f"[non-fatal] Tracing: request failed: {exc}")

            # If we reach here, we need to retry or give up
            if attempt >= self.max_retries:
                logger.error("[non-fatal] Tracing: max retries reached, giving up on this batch.")
                return

            # Exponential backoff + jitter
            sleep_time = delay + random.uniform(0, 0.1 * delay)  # 10% jitter
            time.sleep(sleep_time)
            delay = min(delay * 2, self.max_delay)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()


class BatchTraceProcessor(TracingProcessor):
    """Some implementation notes:
    1. Using Queue, which is thread-safe.
    2. Using a background thread to export spans, to minimize any performance issues.
    3. Spans are stored in memory until they are exported.
    """

    def __init__(
        self,
        exporter: TracingExporter,
        max_queue_size: int = 8192,
        max_batch_size: int = 128,
        schedule_delay: float = 5.0,
        export_trigger_ratio: float = 0.7,
    ):
        """
        Args:
            exporter: The exporter to use.
            max_queue_size: The maximum number of spans to store in the queue. After this, we will
                start dropping spans.
            max_batch_size: The maximum number of spans to export in a single batch.
            schedule_delay: The delay between checks for new spans to export.
            export_trigger_ratio: The ratio of the queue size at which we will trigger an export.
        """
        self._exporter = exporter
        self._queue: queue.Queue[Trace | Span[Any]] = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._max_batch_size = max_batch_size
        self._schedule_delay = schedule_delay
        self._shutdown_event = threading.Event()

        # The queue size threshold at which we export immediately.
        self._export_trigger_size = int(max_queue_size * export_trigger_ratio)

        # Track when we next *must* perform a scheduled export
        self._next_export_time = time.time() + self._schedule_delay

        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def on_trace_start(self, trace: Trace) -> None:
        try:
            self._queue.put_nowait(trace)
        except queue.Full:
            logger.warning("Queue is full, dropping trace.")

    def on_trace_end(self, trace: Trace) -> None:
        # We send traces via on_trace_start, so we don't need to do anything here.
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        # We send spans via on_span_end, so we don't need to do anything here.
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            logger.warning("Queue is full, dropping span.")

    def shutdown(self, timeout: float | None = None):
        """
        Called when the application stops. We signal our thread to stop, then join it.
        """
        self._shutdown_event.set()
        self._worker_thread.join(timeout=timeout)

    def force_flush(self):
        """
        Forces an immediate flush of all queued spans.
        """
        self._export_batches(force=True)

    def _run(self):
        while not self._shutdown_event.is_set():
            current_time = time.time()
            queue_size = self._queue.qsize()

            # If it's time for a scheduled flush or queue is above the trigger threshold
            if current_time >= self._next_export_time or queue_size >= self._export_trigger_size:
                self._export_batches(force=False)
                # Reset the next scheduled flush time
                self._next_export_time = time.time() + self._schedule_delay
            else:
                # Sleep a short interval so we don't busy-wait.
                time.sleep(0.2)

        # Final drain after shutdown
        self._export_batches(force=True)

    def _export_batches(self, force: bool = False):
        """Drains the queue and exports in batches. If force=True, export everything.
        Otherwise, export up to `max_batch_size` repeatedly until the queue is empty or below a
        certain threshold.
        """
        while True:
            items_to_export: list[Span[Any] | Trace] = []

            # Gather a batch of spans up to max_batch_size
            while not self._queue.empty() and (
                force or len(items_to_export) < self._max_batch_size
            ):
                try:
                    items_to_export.append(self._queue.get_nowait())
                except queue.Empty:
                    # Another thread might have emptied the queue between checks
                    break

            # If we collected nothing, we're done
            if not items_to_export:
                break

            # Export the batch
            self._exporter.export(items_to_export)


# Create a shared global instance:
_global_exporter = BackendSpanExporter()
_global_processor = BatchTraceProcessor(_global_exporter)


def default_exporter() -> BackendSpanExporter:
    """The default exporter, which exports traces and spans to the backend in batches."""
    return _global_exporter


def default_processor() -> BatchTraceProcessor:
    """The default processor, which exports traces and spans to the backend in batches."""
    return _global_processor


<file_info>
path: src/agents/voice/exceptions.py
name: exceptions.py
</file_info>
from ..exceptions import AgentsException


class STTWebsocketConnectionError(AgentsException):
    """Exception raised when the STT websocket connection fails."""

    def __init__(self, message: str):
        self.message = message


<file_info>
path: src/agents/voice/workflow.py
name: workflow.py
</file_info>
from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from ..agent import Agent
from ..items import TResponseInputItem
from ..result import RunResultStreaming
from ..run import Runner


class VoiceWorkflowBase(abc.ABC):
    """
    A base class for a voice workflow. You must implement the `run` method. A "workflow" is any
    code you want, that receives a transcription and yields text that will be turned into speech
    by a text-to-speech model.
    In most cases, you'll create `Agent`s and use `Runner.run_streamed()` to run them, returning
    some or all of the text events from the stream. You can use the `VoiceWorkflowHelper` class to
    help with extracting text events from the stream.
    If you have a simple workflow that has a single starting agent and no custom logic, you can
    use `SingleAgentVoiceWorkflow` directly.
    """

    @abc.abstractmethod
    def run(self, transcription: str) -> AsyncIterator[str]:
        """
        Run the voice workflow. You will receive an input transcription, and must yield text that
        will be spoken to the user. You can run whatever logic you want here. In most cases, the
        final logic will involve calling `Runner.run_streamed()` and yielding any text events from
        the stream.
        """
        pass


class VoiceWorkflowHelper:
    @classmethod
    async def stream_text_from(cls, result: RunResultStreaming) -> AsyncIterator[str]:
        """Wraps a `RunResultStreaming` object and yields text events from the stream."""
        async for event in result.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta


class SingleAgentWorkflowCallbacks:
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        """Called when the workflow is run."""
        pass


class SingleAgentVoiceWorkflow(VoiceWorkflowBase):
    """A simple voice workflow that runs a single agent. Each transcription and result is added to
    the input history.
    For more complex workflows (e.g. multiple Runner calls, custom message history, custom logic,
    custom configs), subclass `VoiceWorkflowBase` and implement your own logic.
    """

    def __init__(self, agent: Agent[Any], callbacks: SingleAgentWorkflowCallbacks | None = None):
        """Create a new single agent voice workflow.

        Args:
            agent: The agent to run.
            callbacks: Optional callbacks to call during the workflow.
        """
        self._input_history: list[TResponseInputItem] = []
        self._current_agent = agent
        self._callbacks = callbacks

    async def run(self, transcription: str) -> AsyncIterator[str]:
        if self._callbacks:
            self._callbacks.on_run(self, transcription)

        # Add the transcription to the input history
        self._input_history.append(
            {
                "role": "user",
                "content": transcription,
            }
        )

        # Run the agent
        result = Runner.run_streamed(self._current_agent, self._input_history)

        # Stream the text from the result
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Update the input history and current agent
        self._input_history = result.to_input_list()
        self._current_agent = result.last_agent


<file_info>
path: src/agents/_debug.py
name: _debug.py
</file_info>
import os


def _debug_flag_enabled(flag: str) -> bool:
    flag_value = os.getenv(flag)
    return flag_value is not None and (flag_value == "1" or flag_value.lower() == "true")


DONT_LOG_MODEL_DATA = _debug_flag_enabled("OPENAI_AGENTS_DONT_LOG_MODEL_DATA")
"""By default we don't log LLM inputs/outputs, to prevent exposing sensitive information. Set this
flag to enable logging them.
"""

DONT_LOG_TOOL_DATA = _debug_flag_enabled("OPENAI_AGENTS_DONT_LOG_TOOL_DATA")
"""By default we don't log tool call inputs/outputs, to prevent exposing sensitive information. Set
this flag to enable logging them.
"""


<file_info>
path: src/agents/tool.py
name: tool.py
</file_info>
from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, overload

from openai.types.responses.file_search_tool_param import Filters, RankingOptions
from openai.types.responses.web_search_tool_param import UserLocation
from pydantic import ValidationError
from typing_extensions import Concatenate, ParamSpec

from . import _debug
from .computer import AsyncComputer, Computer
from .exceptions import ModelBehaviorError
from .function_schema import DocstringStyle, function_schema
from .items import RunItem
from .logger import logger
from .run_context import RunContextWrapper
from .tracing import SpanError
from .util import _error_tracing
from .util._types import MaybeAwaitable

ToolParams = ParamSpec("ToolParams")

ToolFunctionWithoutContext = Callable[ToolParams, Any]
ToolFunctionWithContext = Callable[Concatenate[RunContextWrapper[Any], ToolParams], Any]

ToolFunction = Union[ToolFunctionWithoutContext[ToolParams], ToolFunctionWithContext[ToolParams]]


@dataclass
class FunctionToolResult:
    tool: FunctionTool
    """The tool that was run."""

    output: Any
    """The output of the tool."""

    run_item: RunItem
    """The run item that was produced as a result of the tool call."""


@dataclass
class FunctionTool:
    """A tool that wraps a function. In most cases, you should use  the `function_tool` helpers to
    create a FunctionTool, as they let you easily wrap a Python function.
    """

    name: str
    """The name of the tool, as shown to the LLM. Generally the name of the function."""

    description: str
    """A description of the tool, as shown to the LLM."""

    params_json_schema: dict[str, Any]
    """The JSON schema for the tool's parameters."""

    on_invoke_tool: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    """A function that invokes the tool with the given context and parameters. The params passed
    are:
    1. The tool run context.
    2. The arguments from the LLM, as a JSON string.

    You must return a string representation of the tool output, or something we can call `str()` on.
    In case of errors, you can either raise an Exception (which will cause the run to fail) or
    return a string error message (which will be sent back to the LLM).
    """

    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input."""


@dataclass
class FileSearchTool:
    """A hosted tool that lets the LLM search through a vector store. Currently only supported with
    OpenAI models, using the Responses API.
    """

    vector_store_ids: list[str]
    """The IDs of the vector stores to search."""

    max_num_results: int | None = None
    """The maximum number of results to return."""

    include_search_results: bool = False
    """Whether to include the search results in the output produced by the LLM."""

    ranking_options: RankingOptions | None = None
    """Ranking options for search."""

    filters: Filters | None = None
    """A filter to apply based on file attributes."""

    @property
    def name(self):
        return "file_search"


@dataclass
class WebSearchTool:
    """A hosted tool that lets the LLM search the web. Currently only supported with OpenAI models,
    using the Responses API.
    """

    user_location: UserLocation | None = None
    """Optional location for the search. Lets you customize results to be relevant to a location."""

    search_context_size: Literal["low", "medium", "high"] = "medium"
    """The amount of context to use for the search."""

    @property
    def name(self):
        return "web_search_preview"


@dataclass
class ComputerTool:
    """A hosted tool that lets the LLM control a computer."""

    computer: Computer | AsyncComputer
    """The computer implementation, which describes the environment and dimensions of the computer,
    as well as implements the computer actions like click, screenshot, etc.
    """

    @property
    def name(self):
        return "computer_use_preview"


Tool = Union[FunctionTool, FileSearchTool, WebSearchTool, ComputerTool]
"""A tool that can be used in an agent."""


def default_tool_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """The default tool error function, which just returns a generic error message."""
    return f"An error occurred while running the tool. Please try again. Error: {str(error)}"


ToolErrorFunction = Callable[[RunContextWrapper[Any], Exception], MaybeAwaitable[str]]


@overload
def function_tool(
    func: ToolFunction[...],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    strict_mode: bool = True,
) -> FunctionTool:
    """Overload for usage as @function_tool (no parentheses)."""
    ...


@overload
def function_tool(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    strict_mode: bool = True,
) -> Callable[[ToolFunction[...]], FunctionTool]:
    """Overload for usage as @function_tool(...)."""
    ...


def function_tool(
    func: ToolFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = default_tool_error_function,
    strict_mode: bool = True,
) -> FunctionTool | Callable[[ToolFunction[...]], FunctionTool]:
    """
    Decorator to create a FunctionTool from a function. By default, we will:
    1. Parse the function signature to create a JSON schema for the tool's parameters.
    2. Use the function's docstring to populate the tool's description.
    3. Use the function's docstring to populate argument descriptions.
    The docstring style is detected automatically, but you can override it.

    If the function takes a `RunContextWrapper` as the first argument, it *must* match the
    context type of the agent that uses the tool.

    Args:
        func: The function to wrap.
        name_override: If provided, use this name for the tool instead of the function's name.
        description_override: If provided, use this description for the tool instead of the
            function's docstring.
        docstring_style: If provided, use this style for the tool's docstring. If not provided,
            we will attempt to auto-detect the style.
        use_docstring_info: If True, use the function's docstring to populate the tool's
            description and argument descriptions.
        failure_error_function: If provided, use this function to generate an error message when
            the tool call fails. The error message is sent to the LLM. If you pass None, then no
            error message will be sent and instead an Exception will be raised.
        strict_mode: Whether to enable strict mode for the tool's JSON schema. We *strongly*
            recommend setting this to True, as it increases the likelihood of correct JSON input.
            If False, it allows non-strict JSON schemas. For example, if a parameter has a default
            value, it will be optional, additional properties are allowed, etc. See here for more:
            https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#supported-schemas
    """

    def _create_function_tool(the_func: ToolFunction[...]) -> FunctionTool:
        schema = function_schema(
            func=the_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_mode,
        )

        async def _on_invoke_tool_impl(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                json_data: dict[str, Any] = json.loads(input) if input else {}
            except Exception as e:
                if _debug.DONT_LOG_TOOL_DATA:
                    logger.debug(f"Invalid JSON input for tool {schema.name}")
                else:
                    logger.debug(f"Invalid JSON input for tool {schema.name}: {input}")
                raise ModelBehaviorError(
                    f"Invalid JSON input for tool {schema.name}: {input}"
                ) from e

            if _debug.DONT_LOG_TOOL_DATA:
                logger.debug(f"Invoking tool {schema.name}")
            else:
                logger.debug(f"Invoking tool {schema.name} with input {input}")

            try:
                parsed = (
                    schema.params_pydantic_model(**json_data)
                    if json_data
                    else schema.params_pydantic_model()
                )
            except ValidationError as e:
                raise ModelBehaviorError(f"Invalid JSON input for tool {schema.name}: {e}") from e

            args, kwargs_dict = schema.to_call_args(parsed)

            if not _debug.DONT_LOG_TOOL_DATA:
                logger.debug(f"Tool call args: {args}, kwargs: {kwargs_dict}")

            if inspect.iscoroutinefunction(the_func):
                if schema.takes_context:
                    result = await the_func(ctx, *args, **kwargs_dict)
                else:
                    result = await the_func(*args, **kwargs_dict)
            else:
                if schema.takes_context:
                    result = the_func(ctx, *args, **kwargs_dict)
                else:
                    result = the_func(*args, **kwargs_dict)

            if _debug.DONT_LOG_TOOL_DATA:
                logger.debug(f"Tool {schema.name} completed.")
            else:
                logger.debug(f"Tool {schema.name} returned {result}")

            return result

        async def _on_invoke_tool(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                return await _on_invoke_tool_impl(ctx, input)
            except Exception as e:
                if failure_error_function is None:
                    raise

                result = failure_error_function(ctx, e)
                if inspect.isawaitable(result):
                    return await result

                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Error running tool (non-fatal)",
                        data={
                            "tool_name": schema.name,
                            "error": str(e),
                        },
                    )
                )
                return result

        return FunctionTool(
            name=schema.name,
            description=schema.description or "",
            params_json_schema=schema.params_json_schema,
            on_invoke_tool=_on_invoke_tool,
            strict_json_schema=strict_mode,
        )

    # If func is actually a callable, we were used as @function_tool with no parentheses
    if callable(func):
        return _create_function_tool(func)

    # Otherwise, we were used as @function_tool(...), so return a decorator
    def decorator(real_func: ToolFunction[...]) -> FunctionTool:
        return _create_function_tool(real_func)

    return decorator


<file_info>
path: tests/test_tracing_errors_streamed.py
name: test_tracing_errors_streamed.py
</file_info>
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)
from .testing_processor import fetch_normalized_spans


@pytest.mark.asyncio
async def test_single_turn_model_error():
    model = FakeModel(tracing_enabled=True)
    model.set_next_output(ValueError("test error"))

    agent = Agent(
        name="test_agent",
        model=model,
    )
    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in agent run", "data": {"error": "test error"}},
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                        "children": [
                            {
                                "type": "generation",
                                "error": {
                                    "message": "Error",
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multi_turn_no_handoffs():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: error
            ValueError("test error"),
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in agent run", "data": {"error": "test error"}},
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "foo",
                                    "input": '{"a": "b"}',
                                    "output": "tool_result",
                                },
                            },
                            {
                                "type": "generation",
                                "error": {
                                    "message": "Error",
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_tool_call_error():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result", hide_errors=True)],
    )

    model.set_next_output(
        [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")],
    )

    with pytest.raises(ModelBehaviorError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Error in agent run",
                            "data": {"error": "Invalid JSON input for tool foo: bad_json"},
                        },
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "error": {
                                    "message": "Error running tool",
                                    "data": {
                                        "tool_name": "foo",
                                        "error": "Invalid JSON input for tool foo: bad_json",
                                    },
                                },
                                "data": {"name": "foo", "input": "bad_json"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_1, "should have picked first handoff"

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test",
                            "handoffs": ["test", "test"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {"type": "handoff", "data": {"from_agent": "test", "to_agent": "test"}},
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [{"type": "generation"}],
                    },
                ],
            }
        ]
    )


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_no_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

    model.set_next_output(
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ]
    )

    result = Runner.run_streamed(agent_1, input="user_message")
    async for _ in result.stream_events():
        pass

    assert isinstance(result.final_output, dict)
    assert result.final_output["bar"] == "abc"

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "Foo"},
                        "children": [{"type": "generation"}],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
    )
    agent_2 = Agent(
        name="test_agent_2",
        model=model,
        handoffs=[agent_1],
        tools=[get_function_tool("some_function", "result")],
    )
    agent_3 = Agent(
        name="test_agent_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Fourth turn: handoff
            [get_handoff_tool_call(agent_3)],
            # Fifth turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_3, (
        f"should have ended on the third agent, got {result.last_agent.name}"
    )

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": ["test_agent_3"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
                                "data": {"from_agent": "test_agent_1", "to_agent": "test_agent_3"},
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [{"type": "generation"}],
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_max_turns_exceeded():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test",
        model=model,
        output_type=Foo,
        tools=[get_function_tool("foo", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
        ]
    )

    with pytest.raises(MaxTurnsExceeded):
        result = Runner.run_streamed(agent, input="user_message", max_turns=2)
        async for _ in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Max turns exceeded", "data": {"max_turns": 2}},
                        "data": {
                            "name": "test",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "Foo",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


def input_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_input_guardrail_error():
    model = FakeModel()

    agent = Agent(
        name="test",
        model=model,
        input_guardrails=[InputGuardrail(guardrail_function=input_guardrail_function)],
    )
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Guardrail tripwire triggered",
                            "data": {
                                "guardrail": "input_guardrail_function",
                                "type": "input_guardrail",
                            },
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
                                "data": {"name": "input_guardrail_function", "triggered": True},
                            }
                        ],
                    }
                ],
            }
        ]
    )


def output_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_output_guardrail_error():
    model = FakeModel()

    agent = Agent(
        name="test",
        model=model,
        output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail_function)],
    )
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Guardrail tripwire triggered",
                            "data": {"guardrail": "output_guardrail_function"},
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
                                "data": {"name": "output_guardrail_function", "triggered": True},
                            }
                        ],
                    }
                ],
            }
        ]
    )


<file_info>
path: examples/model_providers/custom_example_provider.py
name: custom_example_provider.py
</file_info>
from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )


"""This example uses a custom provider for some calls to Runner.run(), and direct calls to OpenAI for
others. Steps:
1. Create a custom OpenAI client.
2. Create a ModelProvider that uses the custom client.
3. Use the ModelProvider in calls to Runner.run(), only when we want to use the custom LLM provider.

Note that in this example, we disable tracing under the assumption that you don't have an API key
from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
or call set_tracing_export_api_key() to set a tracing specific key.
"""
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    agent = Agent(name="Assistant", instructions="You only respond in haikus.", tools=[get_weather])

    # This will use the custom model provider
    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
    )
    print(result.final_output)

    # If you uncomment this, it will use OpenAI directly, not the custom provider
    # result = await Runner.run(
    #     agent,
    #     "What's the weather in Tokyo?",
    # )
    # print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/basic/lifecycle_example.py
name: lifecycle_example.py
</file_info>
import asyncio
import random
from typing import Any

from pydantic import BaseModel

from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool, Usage, function_tool


class ExampleHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0

    def _usage_to_str(self, usage: Usage) -> str:
        return f"{usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} ended with output {output}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} ended with result {result}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Handoff from {from_agent.name} to {to_agent.name}. Usage: {self._usage_to_str(context.usage)}"
        )


hooks = ExampleHooks()

###


@function_tool
def random_number(max: int) -> int:
    """Generate a random number up to the provided max."""
    return random.randint(0, max)


@function_tool
def multiply_by_two(x: int) -> int:
    """Return x times two."""
    return x * 2


class FinalResult(BaseModel):
    number: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="Multiply the number by 2 and then return the final result.",
    tools=[multiply_by_two],
    output_type=FinalResult,
)

start_agent = Agent(
    name="Start Agent",
    instructions="Generate a random number. If it's even, stop. If it's odd, hand off to the multiplier agent.",
    tools=[random_number],
    output_type=FinalResult,
    handoffs=[multiply_agent],
)


async def main() -> None:
    user_input = input("Enter a max number: ")
    await Runner.run(
        start_agent,
        hooks=hooks,
        input=f"Generate a random number between 0 and {user_input}.",
    )

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
"""
$ python examples/basic/lifecycle_example.py

Enter a max number: 250
### 1: Agent Start Agent started. Usage: 0 requests, 0 input tokens, 0 output tokens, 0 total tokens
### 2: Tool random_number started. Usage: 1 requests, 148 input tokens, 15 output tokens, 163 total tokens
### 3: Tool random_number ended with result 101. Usage: 1 requests, 148 input tokens, 15 output tokens, 163 total tokens
### 4: Agent Start Agent started. Usage: 1 requests, 148 input tokens, 15 output tokens, 163 total tokens
### 5: Handoff from Start Agent to Multiply Agent. Usage: 2 requests, 323 input tokens, 30 output tokens, 353 total tokens
### 6: Agent Multiply Agent started. Usage: 2 requests, 323 input tokens, 30 output tokens, 353 total tokens
### 7: Tool multiply_by_two started. Usage: 3 requests, 504 input tokens, 46 output tokens, 550 total tokens
### 8: Tool multiply_by_two ended with result 202. Usage: 3 requests, 504 input tokens, 46 output tokens, 550 total tokens
### 9: Agent Multiply Agent started. Usage: 3 requests, 504 input tokens, 46 output tokens, 550 total tokens
### 10: Agent Multiply Agent ended with output number=202. Usage: 4 requests, 714 input tokens, 63 output tokens, 777 total tokens
Done!

"""


<file_info>
path: examples/voice/streamed/main.py
name: main.py
</file_info>
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static
from typing_extensions import override

from agents.voice import StreamedAudioInput, VoicePipeline

# Import MyWorkflow class - handle both module and package use cases
if TYPE_CHECKING:
    # For type checking, use the relative import
    from .my_workflow import MyWorkflow
else:
    # At runtime, try both import styles
    try:
        # Try relative import first (when used as a package)
        from .my_workflow import MyWorkflow
    except ImportError:
        # Fall back to direct import (when run as a script)
        from my_workflow import MyWorkflow

CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1


class Header(Static):
    """A header widget."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return "Speak to the agent. When you stop speaking, it will respond."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    should_send_audio: asyncio.Event
    audio_player: sd.OutputStream
    last_audio_item_id: str | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.pipeline = VoicePipeline(
            workflow=MyWorkflow(secret_word="dog", on_start=self._on_transcription)
        )
        self._audio_input = StreamedAudioInput()
        self.audio_player = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT,
        )

    def _on_transcription(self, transcription: str) -> None:
        try:
            self.query_one("#bottom-pane", RichLog).write(f"Transcription: {transcription}")
        except Exception:
            pass

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield Header(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.start_voice_pipeline())
        self.run_worker(self.send_mic_audio())

    async def start_voice_pipeline(self) -> None:
        try:
            self.audio_player.start()
            self.result = await self.pipeline.run(self._audio_input)

            async for event in self.result.stream():
                bottom_pane = self.query_one("#bottom-pane", RichLog)
                if event.type == "voice_stream_event_audio":
                    self.audio_player.write(event.data)
                    bottom_pane.write(
                        f"Received audio: {len(event.data) if event.data is not None else '0'} bytes"
                    )
                elif event.type == "voice_stream_event_lifecycle":
                    bottom_pane.write(f"Lifecycle event: {event.event}")
        except Exception as e:
            bottom_pane = self.query_one("#bottom-pane", RichLog)
            bottom_pane.write(f"Error: {e}")
        finally:
            self.audio_player.close()

    async def send_mic_audio(self) -> None:
        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)

                await self._audio_input.add_audio(data)
                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()


<file_info>
path: src/agents/util/_types.py
name: _types.py
</file_info>
from collections.abc import Awaitable
from typing import Union

from typing_extensions import TypeVar

T = TypeVar("T")
MaybeAwaitable = Union[Awaitable[T], T]


<file_info>
path: src/agents/strict_schema.py
name: strict_schema.py
</file_info>
from __future__ import annotations

from typing import Any

from openai import NOT_GIVEN
from typing_extensions import TypeGuard

from .exceptions import UserError

_EMPTY_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}


def ensure_strict_json_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the OpenAI API expects.
    """
    if schema == {}:
        return _EMPTY_SCHEMA
    return _ensure_strict_json_schema(schema, path=(), root=schema)


# Adapted from https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema, path=(*path, "definitions", definition_name), root=root
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif (
        typ == "object"
        and "additionalProperties" in json_schema
        and json_schema["additionalProperties"] is True
    ):
        raise UserError(
            "additionalProperties should not be set for object types. This could be because "
            "you're using an older version of Pydantic, or because you configured additional "
            "properties to be allowed. If you really need this, update the function or output tool "
            "to not use a strict schema."
        )

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root)
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", NOT_GIVEN) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(value), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


def is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False


<file_info>
path: src/agents/voice/input.py
name: input.py
</file_info>
from __future__ import annotations

import asyncio
import base64
import io
import wave
from dataclasses import dataclass

from ..exceptions import UserError
from .imports import np, npt

DEFAULT_SAMPLE_RATE = 24000


def _buffer_to_audio_file(
    buffer: npt.NDArray[np.int16 | np.float32],
    frame_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = 2,
    channels: int = 1,
) -> tuple[str, io.BytesIO, str]:
    if buffer.dtype == np.float32:
        # convert to int16
        buffer = np.clip(buffer, -1.0, 1.0)
        buffer = (buffer * 32767).astype(np.int16)
    elif buffer.dtype != np.int16:
        raise UserError("Buffer must be a numpy array of int16 or float32")

    audio_file = io.BytesIO()
    with wave.open(audio_file, "w") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(buffer.tobytes())
        audio_file.seek(0)

    # (filename, bytes, content_type)
    return ("audio.wav", audio_file, "audio/wav")


@dataclass
class AudioInput:
    """Static audio to be used as input for the VoicePipeline."""

    buffer: npt.NDArray[np.int16 | np.float32]
    """
    A buffer containing the audio data for the agent. Must be a numpy array of int16 or float32.
    """

    frame_rate: int = DEFAULT_SAMPLE_RATE
    """The sample rate of the audio data. Defaults to 24000."""

    sample_width: int = 2
    """The sample width of the audio data. Defaults to 2."""

    channels: int = 1
    """The number of channels in the audio data. Defaults to 1."""

    def to_audio_file(self) -> tuple[str, io.BytesIO, str]:
        """Returns a tuple of (filename, bytes, content_type)"""
        return _buffer_to_audio_file(self.buffer, self.frame_rate, self.sample_width, self.channels)

    def to_base64(self) -> str:
        """Returns the audio data as a base64 encoded string."""
        if self.buffer.dtype == np.float32:
            # convert to int16
            self.buffer = np.clip(self.buffer, -1.0, 1.0)
            self.buffer = (self.buffer * 32767).astype(np.int16)
        elif self.buffer.dtype != np.int16:
            raise UserError("Buffer must be a numpy array of int16 or float32")

        return base64.b64encode(self.buffer.tobytes()).decode("utf-8")


class StreamedAudioInput:
    """Audio input represented as a stream of audio data. You can pass this to the `VoicePipeline`
    and then push audio data into the queue using the `add_audio` method.
    """

    def __init__(self):
        self.queue: asyncio.Queue[npt.NDArray[np.int16 | np.float32]] = asyncio.Queue()

    async def add_audio(self, audio: npt.NDArray[np.int16 | np.float32]):
        """Adds more audio data to the stream.

        Args:
            audio: The audio data to add. Must be a numpy array of int16 or float32.
        """
        await self.queue.put(audio)


<file_info>
path: src/agents/exceptions.py
name: exceptions.py
</file_info>
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .guardrail import InputGuardrailResult, OutputGuardrailResult


class AgentsException(Exception):
    """Base class for all exceptions in the Agents SDK."""


class MaxTurnsExceeded(AgentsException):
    """Exception raised when the maximum number of turns is exceeded."""

    message: str

    def __init__(self, message: str):
        self.message = message


class ModelBehaviorError(AgentsException):
    """Exception raised when the model does something unexpected, e.g. calling a tool that doesn't
    exist, or providing malformed JSON.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


class UserError(AgentsException):
    """Exception raised when the user makes an error using the SDK."""

    message: str

    def __init__(self, message: str):
        self.message = message


class InputGuardrailTripwireTriggered(AgentsException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "InputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "InputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


class OutputGuardrailTripwireTriggered(AgentsException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "OutputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "OutputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


<file_info>
path: src/agents/_config.py
name: _config.py
</file_info>
from openai import AsyncOpenAI
from typing_extensions import Literal

from .models import _openai_shared
from .tracing import set_tracing_export_api_key


def set_default_openai_key(key: str, use_for_tracing: bool) -> None:
    _openai_shared.set_default_openai_key(key)

    if use_for_tracing:
        set_tracing_export_api_key(key)


def set_default_openai_client(client: AsyncOpenAI, use_for_tracing: bool) -> None:
    _openai_shared.set_default_openai_client(client)

    if use_for_tracing:
        set_tracing_export_api_key(client.api_key)


def set_default_openai_api(api: Literal["chat_completions", "responses"]) -> None:
    if api == "chat_completions":
        _openai_shared.set_use_responses_by_default(False)
    else:
        _openai_shared.set_use_responses_by_default(True)


<file_info>
path: tests/test_guardrails.py
name: test_guardrails.py
</file_info>
from __future__ import annotations

from typing import Any

import pytest

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    TResponseInputItem,
    UserError,
)
from agents.guardrail import input_guardrail, output_guardrail


def get_sync_guardrail(triggers: bool, output_info: Any | None = None):
    def sync_guardrail(
        context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
    ):
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=triggers,
        )

    return sync_guardrail


@pytest.mark.asyncio
async def test_sync_input_guardrail():
    guardrail = InputGuardrail(guardrail_function=get_sync_guardrail(triggers=False))
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = InputGuardrail(guardrail_function=get_sync_guardrail(triggers=True))
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = InputGuardrail(
        guardrail_function=get_sync_guardrail(triggers=True, output_info="test")
    )
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info == "test"


def get_async_input_guardrail(triggers: bool, output_info: Any | None = None):
    async def async_guardrail(
        context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
    ):
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=triggers,
        )

    return async_guardrail


@pytest.mark.asyncio
async def test_async_input_guardrail():
    guardrail = InputGuardrail(guardrail_function=get_async_input_guardrail(triggers=False))
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = InputGuardrail(guardrail_function=get_async_input_guardrail(triggers=True))
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = InputGuardrail(
        guardrail_function=get_async_input_guardrail(triggers=True, output_info="test")
    )
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info == "test"


@pytest.mark.asyncio
async def test_invalid_input_guardrail_raises_user_error():
    with pytest.raises(UserError):
        # Purposely ignoring type error
        guardrail = InputGuardrail(guardrail_function="foo")  # type: ignore
        await guardrail.run(
            agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
        )


def get_sync_output_guardrail(triggers: bool, output_info: Any | None = None):
    def sync_guardrail(context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any):
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=triggers,
        )

    return sync_guardrail


@pytest.mark.asyncio
async def test_sync_output_guardrail():
    guardrail = OutputGuardrail(guardrail_function=get_sync_output_guardrail(triggers=False))
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = OutputGuardrail(guardrail_function=get_sync_output_guardrail(triggers=True))
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = OutputGuardrail(
        guardrail_function=get_sync_output_guardrail(triggers=True, output_info="test")
    )
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info == "test"


def get_async_output_guardrail(triggers: bool, output_info: Any | None = None):
    async def async_guardrail(
        context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
    ):
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=triggers,
        )

    return async_guardrail


@pytest.mark.asyncio
async def test_async_output_guardrail():
    guardrail = OutputGuardrail(guardrail_function=get_async_output_guardrail(triggers=False))
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = OutputGuardrail(guardrail_function=get_async_output_guardrail(triggers=True))
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info is None

    guardrail = OutputGuardrail(
        guardrail_function=get_async_output_guardrail(triggers=True, output_info="test")
    )
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert result.output.tripwire_triggered
    assert result.output.output_info == "test"


@pytest.mark.asyncio
async def test_invalid_output_guardrail_raises_user_error():
    with pytest.raises(UserError):
        # Purposely ignoring type error
        guardrail = OutputGuardrail(guardrail_function="foo")  # type: ignore
        await guardrail.run(
            agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
        )


@input_guardrail
def decorated_input_guardrail(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info="test_1",
        tripwire_triggered=False,
    )


@input_guardrail(name="Custom name")
def decorated_named_input_guardrail(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info="test_2",
        tripwire_triggered=False,
    )


@pytest.mark.asyncio
async def test_input_guardrail_decorators():
    guardrail = decorated_input_guardrail
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info == "test_1"

    guardrail = decorated_named_input_guardrail
    result = await guardrail.run(
        agent=Agent(name="test"), input="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info == "test_2"
    assert guardrail.get_name() == "Custom name"


@output_guardrail
def decorated_output_guardrail(
    context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info="test_3",
        tripwire_triggered=False,
    )


@output_guardrail(name="Custom name")
def decorated_named_output_guardrail(
    context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info="test_4",
        tripwire_triggered=False,
    )


@pytest.mark.asyncio
async def test_output_guardrail_decorators():
    guardrail = decorated_output_guardrail
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info == "test_3"

    guardrail = decorated_named_output_guardrail
    result = await guardrail.run(
        agent=Agent(name="test"), agent_output="test", context=RunContextWrapper(context=None)
    )
    assert not result.output.tripwire_triggered
    assert result.output.output_info == "test_4"
    assert guardrail.get_name() == "Custom name"


<file_info>
path: tests/voice/fake_models.py
name: fake_models.py
</file_info>
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import numpy as np
import numpy.typing as npt

try:
    from agents.voice import (
        AudioInput,
        StreamedAudioInput,
        StreamedTranscriptionSession,
        STTModel,
        STTModelSettings,
        TTSModel,
        TTSModelSettings,
        VoiceWorkflowBase,
    )
except ImportError:
    pass


class FakeTTS(TTSModel):
    """Fakes TTS by just returning string bytes."""

    def __init__(self, strategy: Literal["default", "split_words"] = "default"):
        self.strategy = strategy

    @property
    def model_name(self) -> str:
        return "fake_tts"

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        if self.strategy == "default":
            yield np.zeros(2, dtype=np.int16).tobytes()
        elif self.strategy == "split_words":
            for _ in text.split():
                yield np.zeros(2, dtype=np.int16).tobytes()

    async def verify_audio(self, text: str, audio: bytes, dtype: npt.DTypeLike = np.int16) -> None:
        assert audio == np.zeros(2, dtype=dtype).tobytes()

    async def verify_audio_chunks(
        self, text: str, audio_chunks: list[bytes], dtype: npt.DTypeLike = np.int16
    ) -> None:
        assert audio_chunks == [np.zeros(2, dtype=dtype).tobytes() for _word in text.split()]


class FakeSession(StreamedTranscriptionSession):
    """A fake streamed transcription session that yields preconfigured transcripts."""

    def __init__(self):
        self.outputs: list[str] = []

    async def transcribe_turns(self) -> AsyncIterator[str]:
        for t in self.outputs:
            yield t

    async def close(self) -> None:
        return None


class FakeSTT(STTModel):
    """A fake STT model that either returns a single transcript or yields multiple."""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = outputs or []

    @property
    def model_name(self) -> str:
        return "fake_stt"

    async def transcribe(self, _: AudioInput, __: STTModelSettings, ___: bool, ____: bool) -> str:
        return self.outputs.pop(0)

    async def create_session(
        self,
        _: StreamedAudioInput,
        __: STTModelSettings,
        ___: bool,
        ____: bool,
    ) -> StreamedTranscriptionSession:
        session = FakeSession()
        session.outputs = self.outputs
        return session


class FakeWorkflow(VoiceWorkflowBase):
    """A fake workflow that yields preconfigured outputs."""

    def __init__(self, outputs: list[list[str]] | None = None):
        self.outputs = outputs or []

    def add_output(self, output: list[str]) -> None:
        self.outputs.append(output)

    def add_multiple_outputs(self, outputs: list[list[str]]) -> None:
        self.outputs.extend(outputs)

    async def run(self, _: str) -> AsyncIterator[str]:
        if not self.outputs:
            raise ValueError("No output configured")
        output = self.outputs.pop(0)
        for t in output:
            yield t


class FakeStreamedAudioInput:
    @classmethod
    async def get(cls, count: int) -> StreamedAudioInput:
        input = StreamedAudioInput()
        for _ in range(count):
            await input.add_audio(np.zeros(2, dtype=np.int16))
        return input


<file_info>
path: tests/test_extension_filters.py
name: test_extension_filters.py
</file_info>
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from agents import Agent, HandoffInputData
from agents.extensions.handoff_filters import remove_all_tools
from agents.items import (
    HandoffOutputItem,
    MessageOutputItem,
    ToolCallOutputItem,
    TResponseInputItem,
)


def fake_agent():
    return Agent(
        name="fake_agent",
    )


def _get_message_input_item(content: str) -> TResponseInputItem:
    return {
        "role": "assistant",
        "content": content,
    }


def _get_function_result_input_item(content: str) -> TResponseInputItem:
    return {
        "call_id": "1",
        "output": content,
        "type": "function_call_output",
    }


def _get_message_output_run_item(content: str) -> MessageOutputItem:
    return MessageOutputItem(
        agent=fake_agent(),
        raw_item=ResponseOutputMessage(
            id="1",
            content=[ResponseOutputText(text=content, annotations=[], type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        ),
    )


def _get_tool_output_run_item(content: str) -> ToolCallOutputItem:
    return ToolCallOutputItem(
        agent=fake_agent(),
        raw_item={
            "call_id": "1",
            "output": content,
            "type": "function_call_output",
        },
        output=content,
    )


def _get_handoff_input_item(content: str) -> TResponseInputItem:
    return {
        "call_id": "1",
        "output": content,
        "type": "function_call_output",
    }


def _get_handoff_output_run_item(content: str) -> HandoffOutputItem:
    return HandoffOutputItem(
        agent=fake_agent(),
        raw_item={
            "call_id": "1",
            "output": content,
            "type": "function_call_output",
        },
        source_agent=fake_agent(),
        target_agent=fake_agent(),
    )


def test_empty_data():
    handoff_input_data = HandoffInputData(input_history=(), pre_handoff_items=(), new_items=())
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data


def test_str_historyonly():
    handoff_input_data = HandoffInputData(input_history="Hello", pre_handoff_items=(), new_items=())
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data


def test_str_history_and_list():
    handoff_input_data = HandoffInputData(
        input_history="Hello",
        pre_handoff_items=(),
        new_items=(_get_message_output_run_item("Hello"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data


def test_list_history_and_list():
    handoff_input_data = HandoffInputData(
        input_history=(_get_message_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("123"),),
        new_items=(_get_message_output_run_item("World"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data


def test_removes_tools_from_history():
    handoff_input_data = HandoffInputData(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_function_result_input_item("World"),
            _get_message_input_item("Hello2"),
        ),
        pre_handoff_items=(
            _get_tool_output_run_item("abc"),
            _get_message_output_run_item("123"),
        ),
        new_items=(_get_message_output_run_item("World"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1


def test_removes_tools_from_new_items():
    handoff_input_data = HandoffInputData(
        input_history=(),
        pre_handoff_items=(),
        new_items=(
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 0
    assert len(filtered_data.pre_handoff_items) == 0
    assert len(filtered_data.new_items) == 1


def test_removes_tools_from_new_items_and_history():
    handoff_input_data = HandoffInputData(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_function_result_input_item("World"),
            _get_message_input_item("Hello2"),
        ),
        pre_handoff_items=(
            _get_message_output_run_item("123"),
            _get_tool_output_run_item("456"),
        ),
        new_items=(
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1


def test_removes_handoffs_from_history():
    handoff_input_data = HandoffInputData(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_handoff_input_item("World"),
        ),
        pre_handoff_items=(
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
            _get_handoff_output_run_item("World"),
        ),
        new_items=(
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
            _get_handoff_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 1
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1


<file_info>
path: tests/test_function_schema.py
name: test_function_schema.py
</file_info>
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from agents import RunContextWrapper
from agents.exceptions import UserError
from agents.function_schema import function_schema


def no_args_function():
    """This function has no args."""

    return "ok"


def test_no_args_function():
    func_schema = function_schema(no_args_function)
    assert func_schema.params_json_schema.get("title") == "no_args_function_args"
    assert func_schema.description == "This function has no args."
    assert not func_schema.takes_context

    parsed = func_schema.params_pydantic_model()
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = no_args_function(*args, **kwargs_dict)
    assert result == "ok"


def no_args_function_with_context(ctx: RunContextWrapper[str]):
    return "ok"


def test_no_args_function_with_context() -> None:
    func_schema = function_schema(no_args_function_with_context)
    assert func_schema.takes_context

    context = RunContextWrapper(context="test")
    parsed = func_schema.params_pydantic_model()
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = no_args_function_with_context(context, *args, **kwargs_dict)
    assert result == "ok"


def simple_function(a: int, b: int = 5):
    """
    Args:
        a: The first argument
        b: The second argument

    Returns:
        The sum of a and b
    """
    return a + b


def test_simple_function():
    """Test a function that has simple typed parameters and defaults."""

    func_schema = function_schema(simple_function)
    # Check that the JSON schema is a dictionary with title, type, etc.
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "simple_function_args"
    assert (
        func_schema.params_json_schema.get("properties", {}).get("a").get("description")
        == "The first argument"
    )
    assert (
        func_schema.params_json_schema.get("properties", {}).get("b").get("description")
        == "The second argument"
    )
    assert not func_schema.takes_context

    # Valid input
    valid_input = {"a": 3}
    parsed = func_schema.params_pydantic_model(**valid_input)
    args_tuple, kwargs_dict = func_schema.to_call_args(parsed)
    result = simple_function(*args_tuple, **kwargs_dict)
    assert result == 8  # 3 + 5

    # Another valid input
    valid_input2 = {"a": 3, "b": 10}
    parsed2 = func_schema.params_pydantic_model(**valid_input2)
    args_tuple2, kwargs_dict2 = func_schema.to_call_args(parsed2)
    result2 = simple_function(*args_tuple2, **kwargs_dict2)
    assert result2 == 13  # 3 + 10

    # Invalid input: 'a' must be int
    with pytest.raises(ValidationError):
        func_schema.params_pydantic_model(**{"a": "not an integer"})


def varargs_function(x: int, *numbers: float, flag: bool = False, **kwargs: Any):
    return x, numbers, flag, kwargs


def test_varargs_function():
    """Test a function that uses *args and **kwargs."""

    func_schema = function_schema(varargs_function)
    # Check JSON schema structure
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "varargs_function_args"

    # Valid input including *args in 'numbers' and **kwargs in 'kwargs'
    valid_input = {
        "x": 10,
        "numbers": [1.1, 2.2, 3.3],
        "flag": True,
        "kwargs": {"extra1": "hello", "extra2": 42},
    }
    parsed = func_schema.params_pydantic_model(**valid_input)
    args, kwargs_dict = func_schema.to_call_args(parsed)

    result = varargs_function(*args, **kwargs_dict)
    # result should be (10, (1.1, 2.2, 3.3), True, {"extra1": "hello", "extra2": 42})
    assert result[0] == 10
    assert result[1] == (1.1, 2.2, 3.3)
    assert result[2] is True
    assert result[3] == {"extra1": "hello", "extra2": 42}

    # Missing 'x' should raise error
    with pytest.raises(ValidationError):
        func_schema.params_pydantic_model(**{"numbers": [1.1, 2.2]})

    # 'flag' can be omitted because it has a default
    valid_input_no_flag = {"x": 7, "numbers": [9.9], "kwargs": {"some_key": "some_value"}}
    parsed2 = func_schema.params_pydantic_model(**valid_input_no_flag)
    args2, kwargs_dict2 = func_schema.to_call_args(parsed2)
    result2 = varargs_function(*args2, **kwargs_dict2)
    # result2 should be (7, (9.9,), False, {'some_key': 'some_value'})
    assert result2 == (7, (9.9,), False, {"some_key": "some_value"})


class Foo(TypedDict):
    a: int
    b: str


class InnerModel(BaseModel):
    a: int
    b: str


class OuterModel(BaseModel):
    inner: InnerModel
    foo: Foo


def complex_args_function(model: OuterModel) -> str:
    return f"{model.inner.a}, {model.inner.b}, {model.foo['a']}, {model.foo['b']}"


def test_nested_data_function():
    func_schema = function_schema(complex_args_function)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "complex_args_function_args"

    # Valid input
    model = OuterModel(inner=InnerModel(a=1, b="hello"), foo=Foo(a=2, b="world"))
    valid_input = {
        "model": model.model_dump(),
    }

    parsed = func_schema.params_pydantic_model(**valid_input)
    args, kwargs_dict = func_schema.to_call_args(parsed)

    result = complex_args_function(*args, **kwargs_dict)
    assert result == "1, hello, 2, world"


def complex_args_and_docs_function(model: OuterModel, some_flag: int = 0) -> str:
    """
    This function takes a model and a flag, and returns a string.

    Args:
        model: A model with an inner and foo field
        some_flag: An optional flag with a default of 0

    Returns:
        A string with the values of the model and flag
    """
    return f"{model.inner.a}, {model.inner.b}, {model.foo['a']}, {model.foo['b']}, {some_flag or 0}"


def test_complex_args_and_docs_function():
    func_schema = function_schema(complex_args_and_docs_function)

    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "complex_args_and_docs_function_args"

    # Check docstring is parsed correctly
    properties = func_schema.params_json_schema.get("properties", {})
    assert properties.get("model").get("description") == "A model with an inner and foo field"
    assert properties.get("some_flag").get("description") == "An optional flag with a default of 0"

    # Valid input
    model = OuterModel(inner=InnerModel(a=1, b="hello"), foo=Foo(a=2, b="world"))
    valid_input = {
        "model": model.model_dump(),
    }

    parsed = func_schema.params_pydantic_model(**valid_input)
    args, kwargs_dict = func_schema.to_call_args(parsed)

    result = complex_args_and_docs_function(*args, **kwargs_dict)
    assert result == "1, hello, 2, world, 0"

    # Invalid input: 'some_flag' must be int
    with pytest.raises(ValidationError):
        func_schema.params_pydantic_model(
            **{"model": model.model_dump(), "some_flag": "not an int"}
        )

    # Valid input: 'some_flag' can be omitted because it has a default
    valid_input_no_flag = {"model": model.model_dump()}
    parsed2 = func_schema.params_pydantic_model(**valid_input_no_flag)
    args2, kwargs_dict2 = func_schema.to_call_args(parsed2)
    result2 = complex_args_and_docs_function(*args2, **kwargs_dict2)
    assert result2 == "1, hello, 2, world, 0"


def function_with_context(ctx: RunContextWrapper[str], a: int, b: int = 5):
    return a + b


def test_function_with_context():
    func_schema = function_schema(function_with_context)
    assert func_schema.takes_context

    context = RunContextWrapper(context="test")

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)

    result = function_with_context(context, *args, **kwargs_dict)
    assert result == 3


class MyClass:
    def foo(self, a: int, b: int = 5):
        return a + b

    def foo_ctx(self, ctx: RunContextWrapper[str], a: int, b: int = 5):
        return a + b

    @classmethod
    def bar(cls, a: int, b: int = 5):
        return a + b

    @classmethod
    def bar_ctx(cls, ctx: RunContextWrapper[str], a: int, b: int = 5):
        return a + b

    @staticmethod
    def baz(a: int, b: int = 5):
        return a + b

    @staticmethod
    def baz_ctx(ctx: RunContextWrapper[str], a: int, b: int = 5):
        return a + b


def test_class_based_functions():
    context = RunContextWrapper(context="test")

    # Instance method
    instance = MyClass()
    func_schema = function_schema(instance.foo)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "foo_args"

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = instance.foo(*args, **kwargs_dict)
    assert result == 3

    # Instance method with context
    func_schema = function_schema(instance.foo_ctx)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "foo_ctx_args"
    assert func_schema.takes_context

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = instance.foo_ctx(context, *args, **kwargs_dict)
    assert result == 3

    # Class method
    func_schema = function_schema(MyClass.bar)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "bar_args"

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = MyClass.bar(*args, **kwargs_dict)
    assert result == 3

    # Class method with context
    func_schema = function_schema(MyClass.bar_ctx)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "bar_ctx_args"
    assert func_schema.takes_context

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = MyClass.bar_ctx(context, *args, **kwargs_dict)
    assert result == 3

    # Static method
    func_schema = function_schema(MyClass.baz)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "baz_args"

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = MyClass.baz(*args, **kwargs_dict)
    assert result == 3

    # Static method with context
    func_schema = function_schema(MyClass.baz_ctx)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "baz_ctx_args"
    assert func_schema.takes_context

    input = {"a": 1, "b": 2}
    parsed = func_schema.params_pydantic_model(**input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = MyClass.baz_ctx(context, *args, **kwargs_dict)
    assert result == 3


class MyEnum(str, Enum):
    FOO = "foo"
    BAR = "bar"
    BAZ = "baz"


def enum_and_literal_function(a: MyEnum, b: Literal["a", "b", "c"]) -> str:
    return f"{a.value} {b}"


def test_enum_and_literal_function():
    func_schema = function_schema(enum_and_literal_function)
    assert isinstance(func_schema.params_json_schema, dict)
    assert func_schema.params_json_schema.get("title") == "enum_and_literal_function_args"

    # Check that the enum values are included in the JSON schema
    assert func_schema.params_json_schema.get("$defs", {}).get("MyEnum", {}).get("enum") == [
        "foo",
        "bar",
        "baz",
    ]

    # Check that the enum is expressed as a def
    assert (
        func_schema.params_json_schema.get("properties", {}).get("a", {}).get("$ref")
        == "#/$defs/MyEnum"
    )

    # Check that the literal values are included in the JSON schema
    assert func_schema.params_json_schema.get("properties", {}).get("b", {}).get("enum") == [
        "a",
        "b",
        "c",
    ]

    # Valid input
    valid_input = {"a": "foo", "b": "a"}
    parsed = func_schema.params_pydantic_model(**valid_input)
    args, kwargs_dict = func_schema.to_call_args(parsed)
    result = enum_and_literal_function(*args, **kwargs_dict)
    assert result == "foo a"

    # Invalid input: 'a' must be a valid enum value
    with pytest.raises(ValidationError):
        func_schema.params_pydantic_model(**{"a": "not an enum value", "b": "a"})

    # Invalid input: 'b' must be a valid literal value
    with pytest.raises(ValidationError):
        func_schema.params_pydantic_model(**{"a": "foo", "b": "not a literal value"})


def test_run_context_in_non_first_position_raises_value_error():
    # When a parameter (after the first) is annotated as RunContextWrapper,
    # function_schema() should raise a UserError.
    def func(a: int, context: RunContextWrapper) -> None:
        pass

    with pytest.raises(UserError):
        function_schema(func, use_docstring_info=False)


def test_var_positional_tuple_annotation():
    # When a function has a var-positional parameter annotated with a tuple type,
    # function_schema() should convert it into a field with type List[<tuple-element>].
    def func(*args: tuple[int, ...]) -> int:
        total = 0
        for arg in args:
            total += sum(arg)
        return total

    fs = function_schema(func, use_docstring_info=False)

    properties = fs.params_json_schema.get("properties", {})
    assert properties.get("args").get("type") == "array"
    assert properties.get("args").get("items").get("type") == "integer"


def test_var_keyword_dict_annotation():
    # Case 3:
    # When a function has a var-keyword parameter annotated with a dict type,
    # function_schema() should convert it into a field with type Dict[<key>, <value>].
    def func(**kwargs: dict[str, int]):
        return kwargs

    fs = function_schema(func, use_docstring_info=False)

    properties = fs.params_json_schema.get("properties", {})
    # The name of the field is "kwargs", and it's a JSON object i.e. a dict.
    assert properties.get("kwargs").get("type") == "object"
    # The values in the dict are integers.
    assert properties.get("kwargs").get("additionalProperties").get("type") == "integer"


<file_info>
path: examples/financial_research_agent/agents/__init__.py
name: __init__.py
</file_info>


<file_info>
path: examples/financial_research_agent/main.py
name: main.py
</file_info>
import asyncio

from .manager import FinancialResearchManager


# Entrypoint for the financial bot example.
# Run this as `python -m examples.financial_bot.main` and enter a
# financial research query, for example:
# "Write up an analysis of Apple Inc.'s most recent quarter."
async def main() -> None:
    query = input("Enter a financial research query: ")
    mgr = FinancialResearchManager()
    await mgr.run(query)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/basic/stream_text.py
name: stream_text.py
</file_info>
import asyncio

from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner


async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/basic/stream_items.py
name: stream_items.py
</file_info>
import asyncio
import random

from agents import Agent, ItemHelpers, Runner, function_tool


@function_tool
def how_many_jokes() -> int:
    return random.randint(1, 10)


async def main():
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
    )
    print("=== Run starting ===")
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")


if __name__ == "__main__":
    asyncio.run(main())

    # === Run starting ===
    # Agent updated: Joker
    # -- Tool was called
    # -- Tool output: 4
    # -- Message output:
    #  Sure, here are four jokes for you:

    # 1. **Why don't skeletons fight each other?**
    #    They don't have the guts!

    # 2. **What do you call fake spaghetti?**
    #    An impasta!

    # 3. **Why did the scarecrow win an award?**
    #    Because he was outstanding in his field!

    # 4. **Why did the bicycle fall over?**
    #    Because it was two-tired!
    # === Run complete ===


<file_info>
path: examples/agent_patterns/forcing_tool_use.py
name: forcing_tool_use.py
</file_info>
from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputFunction,
    ToolsToFinalOutputResult,
    function_tool,
)

"""
This example shows how to force the agent to use a tool. It uses `ModelSettings(tool_choice="required")`
to force the agent to use any tool.

You can run it with 3 options:
1. `default`: The default behavior, which is to send the tool output to the LLM. In this case,
    `tool_choice` is not set, because otherwise it would result in an infinite loop - the LLM would
    call the tool, the tool would run and send the results to the LLM, and that would repeat
    (because the model is forced to use a tool every time.)
2. `first_tool_result`: The first tool result is used as the final output.
3. `custom`: A custom tool use behavior function is used. The custom function receives all the tool
    results, and chooses to use the first tool result to generate the final output.

Usage:
python examples/agent_patterns/forcing_tool_use.py -t default
python examples/agent_patterns/forcing_tool_use.py -t first_tool
python examples/agent_patterns/forcing_tool_use.py -t custom
"""


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind")


async def custom_tool_use_behavior(
    context: RunContextWrapper[Any], results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    weather: Weather = results[0].output
    return ToolsToFinalOutputResult(
        is_final_output=True, final_output=f"{weather.city} is {weather.conditions}."
    )


async def main(tool_use_behavior: Literal["default", "first_tool", "custom"] = "default"):
    if tool_use_behavior == "default":
        behavior: Literal["run_llm_again", "stop_on_first_tool"] | ToolsToFinalOutputFunction = (
            "run_llm_again"
        )
    elif tool_use_behavior == "first_tool":
        behavior = "stop_on_first_tool"
    elif tool_use_behavior == "custom":
        behavior = custom_tool_use_behavior

    agent = Agent(
        name="Weather agent",
        instructions="You are a helpful agent.",
        tools=[get_weather],
        tool_use_behavior=behavior,
        model_settings=ModelSettings(
            tool_choice="required" if tool_use_behavior != "default" else None
        ),
    )

    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tool-use-behavior",
        type=str,
        required=True,
        choices=["default", "first_tool", "custom"],
        help="The behavior to use for tool use. Default will cause tool outputs to be sent to the model. "
        "first_tool_result will cause the first tool result to be used as the final output. "
        "custom will use a custom tool use behavior function.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tool_use_behavior))


<file_info>
path: src/agents/run.py
name: run.py
</file_info>
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, cast

from openai.types.responses import ResponseCompletedEvent

from ._run_impl import (
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    QueueCompleteSentinel,
    RunImpl,
    SingleStepResult,
    TraceCtxManager,
    get_model_tracing_impl,
)
from .agent import Agent
from .agent_output import AgentOutputSchema
from .exceptions import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
)
from .guardrail import InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult
from .handoffs import Handoff, HandoffInputFilter, handoff
from .items import ItemHelpers, ModelResponse, RunItem, TResponseInputItem
from .lifecycle import RunHooks
from .logger import logger
from .model_settings import ModelSettings
from .models.interface import Model, ModelProvider
from .models.openai_provider import OpenAIProvider
from .result import RunResult, RunResultStreaming
from .run_context import RunContextWrapper, TContext
from .stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent
from .tracing import Span, SpanError, agent_span, get_current_trace, trace
from .tracing.span_data import AgentSpanData
from .usage import Usage
from .util import _coro, _error_tracing

DEFAULT_MAX_TURNS = 10


@dataclass
class RunConfig:
    """Configures settings for the entire agent run."""

    model: str | Model | None = None
    """The model to use for the entire agent run. If set, will override the model set on every
    agent. The model_provider passed in below must be able to resolve this model name.
    """

    model_provider: ModelProvider = field(default_factory=OpenAIProvider)
    """The model provider to use when looking up string model names. Defaults to OpenAI."""

    model_settings: ModelSettings | None = None
    """Configure global model settings. Any non-null values will override the agent-specific model
    settings.
    """

    handoff_input_filter: HandoffInputFilter | None = None
    """A global input filter to apply to all handoffs. If `Handoff.input_filter` is set, then that
    will take precedence. The input filter allows you to edit the inputs that are sent to the new
    agent. See the documentation in `Handoff.input_filter` for more details.
    """

    input_guardrails: list[InputGuardrail[Any]] | None = None
    """A list of input guardrails to run on the initial run input."""

    output_guardrails: list[OutputGuardrail[Any]] | None = None
    """A list of output guardrails to run on the final output of the run."""

    tracing_disabled: bool = False
    """Whether tracing is disabled for the agent run. If disabled, we will not trace the agent run.
    """

    trace_include_sensitive_data: bool = True
    """Whether we include potentially sensitive data (for example: inputs/outputs of tool calls or
    LLM generations) in traces. If False, we'll still create spans for these events, but the
    sensitive data will not be included.
    """

    workflow_name: str = "Agent workflow"
    """The name of the run, used for tracing. Should be a logical name for the run, like
    "Code generation workflow" or "Customer support agent".
    """

    trace_id: str | None = None
    """A custom trace ID to use for tracing. If not provided, we will generate a new trace ID."""

    group_id: str | None = None
    """
    A grouping identifier to use for tracing, to link multiple traces from the same conversation
    or process. For example, you might use a chat thread ID.
    """

    trace_metadata: dict[str, Any] | None = None
    """
    An optional dictionary of additional metadata to include with the trace.
    """


class Runner:
    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Run a workflow starting at the given agent. The agent will run in a loop until a final
        output is generated. The loop runs like so:
        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
            `agent.output_type`, the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:
        1. If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note that only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user message,
                or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined as one
                AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.

        Returns:
            A run result containing all the inputs, guardrail results and the output of the last
            agent. Agents may perform handoffs, so we don't know the specific type of the output.
        """
        if hooks is None:
            hooks = RunHooks[Any]()
        if run_config is None:
            run_config = RunConfig()

        with TraceCtxManager(
            workflow_name=run_config.workflow_name,
            trace_id=run_config.trace_id,
            group_id=run_config.group_id,
            metadata=run_config.trace_metadata,
            disabled=run_config.tracing_disabled,
        ):
            current_turn = 0
            original_input: str | list[TResponseInputItem] = copy.deepcopy(input)
            generated_items: list[RunItem] = []
            model_responses: list[ModelResponse] = []

            context_wrapper: RunContextWrapper[TContext] = RunContextWrapper(
                context=context,  # type: ignore
            )

            input_guardrail_results: list[InputGuardrailResult] = []

            current_span: Span[AgentSpanData] | None = None
            current_agent = starting_agent
            should_run_agent_start_hooks = True

            try:
                while True:
                    # Start an agent span if we don't have one. This span is ended if the current
                    # agent changes, or if the agent loop ends.
                    if current_span is None:
                        handoff_names = [h.agent_name for h in cls._get_handoffs(current_agent)]
                        tool_names = [t.name for t in current_agent.tools]
                        if output_schema := cls._get_output_schema(current_agent):
                            output_type_name = output_schema.output_type_name()
                        else:
                            output_type_name = "str"

                        current_span = agent_span(
                            name=current_agent.name,
                            handoffs=handoff_names,
                            tools=tool_names,
                            output_type=output_type_name,
                        )
                        current_span.start(mark_as_current=True)

                    current_turn += 1
                    if current_turn > max_turns:
                        _error_tracing.attach_error_to_span(
                            current_span,
                            SpanError(
                                message="Max turns exceeded",
                                data={"max_turns": max_turns},
                            ),
                        )
                        raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")

                    logger.debug(
                        f"Running agent {current_agent.name} (turn {current_turn})",
                    )

                    if current_turn == 1:
                        input_guardrail_results, turn_result = await asyncio.gather(
                            cls._run_input_guardrails(
                                starting_agent,
                                starting_agent.input_guardrails
                                + (run_config.input_guardrails or []),
                                copy.deepcopy(input),
                                context_wrapper,
                            ),
                            cls._run_single_turn(
                                agent=current_agent,
                                original_input=original_input,
                                generated_items=generated_items,
                                hooks=hooks,
                                context_wrapper=context_wrapper,
                                run_config=run_config,
                                should_run_agent_start_hooks=should_run_agent_start_hooks,
                            ),
                        )
                    else:
                        turn_result = await cls._run_single_turn(
                            agent=current_agent,
                            original_input=original_input,
                            generated_items=generated_items,
                            hooks=hooks,
                            context_wrapper=context_wrapper,
                            run_config=run_config,
                            should_run_agent_start_hooks=should_run_agent_start_hooks,
                        )
                    should_run_agent_start_hooks = False

                    model_responses.append(turn_result.model_response)
                    original_input = turn_result.original_input
                    generated_items = turn_result.generated_items

                    if isinstance(turn_result.next_step, NextStepFinalOutput):
                        output_guardrail_results = await cls._run_output_guardrails(
                            current_agent.output_guardrails + (run_config.output_guardrails or []),
                            current_agent,
                            turn_result.next_step.output,
                            context_wrapper,
                        )
                        return RunResult(
                            input=original_input,
                            new_items=generated_items,
                            raw_responses=model_responses,
                            final_output=turn_result.next_step.output,
                            _last_agent=current_agent,
                            input_guardrail_results=input_guardrail_results,
                            output_guardrail_results=output_guardrail_results,
                        )
                    elif isinstance(turn_result.next_step, NextStepHandoff):
                        current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                        current_span.finish(reset_current=True)
                        current_span = None
                        should_run_agent_start_hooks = True
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        pass
                    else:
                        raise AgentsException(
                            f"Unknown next step type: {type(turn_result.next_step)}"
                        )
            finally:
                if current_span:
                    current_span.finish(reset_current=True)

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Run a workflow synchronously, starting at the given agent. Note that this just wraps the
        `run` method, so it will not work if there's already an event loop (e.g. inside an async
        function, or in a Jupyter notebook or async context like FastAPI). For those cases, use
        the `run` method instead.

        The agent will run in a loop until a final output is generated. The loop runs like so:
        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
            `agent.output_type`, the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:
        1. If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note that only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user message,
                or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined as one
                AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.

        Returns:
            A run result containing all the inputs, guardrail results and the output of the last
            agent. Agents may perform handoffs, so we don't know the specific type of the output.
        """
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
            )
        )

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResultStreaming:
        """Run a workflow starting at the given agent in streaming mode. The returned result object
        contains a method you can use to stream semantic events as they are generated.

        The agent will run in a loop until a final output is generated. The loop runs like so:
        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
            `agent.output_type`, the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:
        1. If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note that only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user message,
                or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined as one
                AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.

        Returns:
            A result object that contains data about the run, as well as a method to stream events.
        """
        if hooks is None:
            hooks = RunHooks[Any]()
        if run_config is None:
            run_config = RunConfig()

        # If there's already a trace, we don't create a new one. In addition, we can't end the
        # trace here, because the actual work is done in `stream_events` and this method ends
        # before that.
        new_trace = (
            None
            if get_current_trace()
            else trace(
                workflow_name=run_config.workflow_name,
                trace_id=run_config.trace_id,
                group_id=run_config.group_id,
                metadata=run_config.trace_metadata,
                disabled=run_config.tracing_disabled,
            )
        )
        # Need to start the trace here, because the current trace contextvar is captured at
        # asyncio.create_task time
        if new_trace:
            new_trace.start(mark_as_current=True)

        output_schema = cls._get_output_schema(starting_agent)
        context_wrapper: RunContextWrapper[TContext] = RunContextWrapper(
            context=context  # type: ignore
        )

        streamed_result = RunResultStreaming(
            input=copy.deepcopy(input),
            new_items=[],
            current_agent=starting_agent,
            raw_responses=[],
            final_output=None,
            is_complete=False,
            current_turn=0,
            max_turns=max_turns,
            input_guardrail_results=[],
            output_guardrail_results=[],
            _current_agent_output_schema=output_schema,
            _trace=new_trace,
        )

        # Kick off the actual agent loop in the background and return the streamed result object.
        streamed_result._run_impl_task = asyncio.create_task(
            cls._run_streamed_impl(
                starting_input=input,
                streamed_result=streamed_result,
                starting_agent=starting_agent,
                max_turns=max_turns,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )
        )
        return streamed_result

    @classmethod
    async def _run_input_guardrails_with_queue(
        cls,
        agent: Agent[Any],
        guardrails: list[InputGuardrail[TContext]],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
        streamed_result: RunResultStreaming,
        parent_span: Span[Any],
    ):
        queue = streamed_result._input_guardrail_queue

        # We'll run the guardrails and push them onto the queue as they complete
        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_input_guardrail(agent, guardrail, input, context)
            )
            for guardrail in guardrails
        ]
        guardrail_results = []
        try:
            for done in asyncio.as_completed(guardrail_tasks):
                result = await done
                if result.output.tripwire_triggered:
                    _error_tracing.attach_error_to_span(
                        parent_span,
                        SpanError(
                            message="Guardrail tripwire triggered",
                            data={
                                "guardrail": result.guardrail.get_name(),
                                "type": "input_guardrail",
                            },
                        ),
                    )
                queue.put_nowait(result)
                guardrail_results.append(result)
        except Exception:
            for t in guardrail_tasks:
                t.cancel()
            raise

        streamed_result.input_guardrail_results = guardrail_results

    @classmethod
    async def _run_streamed_impl(
        cls,
        starting_input: str | list[TResponseInputItem],
        streamed_result: RunResultStreaming,
        starting_agent: Agent[TContext],
        max_turns: int,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ):
        current_span: Span[AgentSpanData] | None = None
        current_agent = starting_agent
        current_turn = 0
        should_run_agent_start_hooks = True

        streamed_result._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=current_agent))

        try:
            while True:
                if streamed_result.is_complete:
                    break

                # Start an agent span if we don't have one. This span is ended if the current
                # agent changes, or if the agent loop ends.
                if current_span is None:
                    handoff_names = [h.agent_name for h in cls._get_handoffs(current_agent)]
                    tool_names = [t.name for t in current_agent.tools]
                    if output_schema := cls._get_output_schema(current_agent):
                        output_type_name = output_schema.output_type_name()
                    else:
                        output_type_name = "str"

                    current_span = agent_span(
                        name=current_agent.name,
                        handoffs=handoff_names,
                        tools=tool_names,
                        output_type=output_type_name,
                    )
                    current_span.start(mark_as_current=True)

                current_turn += 1
                streamed_result.current_turn = current_turn

                if current_turn > max_turns:
                    _error_tracing.attach_error_to_span(
                        current_span,
                        SpanError(
                            message="Max turns exceeded",
                            data={"max_turns": max_turns},
                        ),
                    )
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    break

                if current_turn == 1:
                    # Run the input guardrails in the background and put the results on the queue
                    streamed_result._input_guardrails_task = asyncio.create_task(
                        cls._run_input_guardrails_with_queue(
                            starting_agent,
                            starting_agent.input_guardrails + (run_config.input_guardrails or []),
                            copy.deepcopy(ItemHelpers.input_to_new_input_list(starting_input)),
                            context_wrapper,
                            streamed_result,
                            current_span,
                        )
                    )
                try:
                    turn_result = await cls._run_single_turn_streamed(
                        streamed_result,
                        current_agent,
                        hooks,
                        context_wrapper,
                        run_config,
                        should_run_agent_start_hooks,
                    )
                    should_run_agent_start_hooks = False

                    streamed_result.raw_responses = streamed_result.raw_responses + [
                        turn_result.model_response
                    ]
                    streamed_result.input = turn_result.original_input
                    streamed_result.new_items = turn_result.generated_items

                    if isinstance(turn_result.next_step, NextStepHandoff):
                        current_agent = turn_result.next_step.new_agent
                        current_span.finish(reset_current=True)
                        current_span = None
                        should_run_agent_start_hooks = True
                        streamed_result._event_queue.put_nowait(
                            AgentUpdatedStreamEvent(new_agent=current_agent)
                        )
                    elif isinstance(turn_result.next_step, NextStepFinalOutput):
                        streamed_result._output_guardrails_task = asyncio.create_task(
                            cls._run_output_guardrails(
                                current_agent.output_guardrails
                                + (run_config.output_guardrails or []),
                                current_agent,
                                turn_result.next_step.output,
                                context_wrapper,
                            )
                        )

                        try:
                            output_guardrail_results = await streamed_result._output_guardrails_task
                        except Exception:
                            # Exceptions will be checked in the stream_events loop
                            output_guardrail_results = []

                        streamed_result.output_guardrail_results = output_guardrail_results
                        streamed_result.final_output = turn_result.next_step.output
                        streamed_result.is_complete = True
                        streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        pass
                except Exception as e:
                    if current_span:
                        _error_tracing.attach_error_to_span(
                            current_span,
                            SpanError(
                                message="Error in agent run",
                                data={"error": str(e)},
                            ),
                        )
                    streamed_result.is_complete = True
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    raise

            streamed_result.is_complete = True
        finally:
            if current_span:
                current_span.finish(reset_current=True)

    @classmethod
    async def _run_single_turn_streamed(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
    ) -> SingleStepResult:
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
            )

        output_schema = cls._get_output_schema(agent)

        streamed_result.current_agent = agent
        streamed_result._current_agent_output_schema = output_schema

        system_prompt = await agent.get_system_prompt(context_wrapper)

        handoffs = cls._get_handoffs(agent)

        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        final_response: ModelResponse | None = None

        input = ItemHelpers.input_to_new_input_list(streamed_result.input)
        input.extend([item.to_input_item() for item in streamed_result.new_items])

        # 1. Stream the output events
        async for event in model.stream_response(
            system_prompt,
            input,
            model_settings,
            agent.tools,
            output_schema,
            handoffs,
            get_model_tracing_impl(
                run_config.tracing_disabled, run_config.trace_include_sensitive_data
            ),
        ):
            if isinstance(event, ResponseCompletedEvent):
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                    )
                    if event.response.usage
                    else Usage()
                )
                final_response = ModelResponse(
                    output=event.response.output,
                    usage=usage,
                    referenceable_id=event.response.id,
                )

            streamed_result._event_queue.put_nowait(RawResponsesStreamEvent(data=event))

        # 2. At this point, the streaming is complete for this turn of the agent loop.
        if not final_response:
            raise ModelBehaviorError("Model did not produce a final response!")

        # 3. Now, we can process the turn as we do in the non-streaming case
        single_step_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=streamed_result.input,
            pre_step_items=streamed_result.new_items,
            new_response=final_response,
            output_schema=output_schema,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

        RunImpl.stream_step_result_to_queue(single_step_result, streamed_result._event_queue)
        return single_step_result

    @classmethod
    async def _run_single_turn(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
    ) -> SingleStepResult:
        # Ensure we run the hooks before anything else
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
            )

        system_prompt = await agent.get_system_prompt(context_wrapper)

        output_schema = cls._get_output_schema(agent)
        handoffs = cls._get_handoffs(agent)
        input = ItemHelpers.input_to_new_input_list(original_input)
        input.extend([generated_item.to_input_item() for generated_item in generated_items])

        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            handoffs,
            context_wrapper,
            run_config,
        )

        return await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=original_input,
            pre_step_items=generated_items,
            new_response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _get_single_step_result_from_response(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        processed_response = RunImpl.process_model_response(
            agent=agent,
            response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
        )
        return await RunImpl.execute_tools_and_side_effects(
            agent=agent,
            original_input=original_input,
            pre_step_items=pre_step_items,
            new_response=new_response,
            processed_response=processed_response,
            output_schema=output_schema,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _run_input_guardrails(
        cls,
        agent: Agent[Any],
        guardrails: list[InputGuardrail[TContext]],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
    ) -> list[InputGuardrailResult]:
        if not guardrails:
            return []

        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_input_guardrail(agent, guardrail, input, context)
            )
            for guardrail in guardrails
        ]

        guardrail_results = []

        for done in asyncio.as_completed(guardrail_tasks):
            result = await done
            if result.output.tripwire_triggered:
                # Cancel all guardrail tasks if a tripwire is triggered.
                for t in guardrail_tasks:
                    t.cancel()
                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Guardrail tripwire triggered",
                        data={"guardrail": result.guardrail.get_name()},
                    )
                )
                raise InputGuardrailTripwireTriggered(result)
            else:
                guardrail_results.append(result)

        return guardrail_results

    @classmethod
    async def _run_output_guardrails(
        cls,
        guardrails: list[OutputGuardrail[TContext]],
        agent: Agent[TContext],
        agent_output: Any,
        context: RunContextWrapper[TContext],
    ) -> list[OutputGuardrailResult]:
        if not guardrails:
            return []

        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_output_guardrail(guardrail, agent, agent_output, context)
            )
            for guardrail in guardrails
        ]

        guardrail_results = []

        for done in asyncio.as_completed(guardrail_tasks):
            result = await done
            if result.output.tripwire_triggered:
                # Cancel all guardrail tasks if a tripwire is triggered.
                for t in guardrail_tasks:
                    t.cancel()
                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Guardrail tripwire triggered",
                        data={"guardrail": result.guardrail.get_name()},
                    )
                )
                raise OutputGuardrailTripwireTriggered(result)
            else:
                guardrail_results.append(result)

        return guardrail_results

    @classmethod
    async def _get_new_response(
        cls,
        agent: Agent[TContext],
        system_prompt: str | None,
        input: list[TResponseInputItem],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> ModelResponse:
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        new_response = await model.get_response(
            system_instructions=system_prompt,
            input=input,
            model_settings=model_settings,
            tools=agent.tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=get_model_tracing_impl(
                run_config.tracing_disabled, run_config.trace_include_sensitive_data
            ),
        )

        context_wrapper.usage.add(new_response.usage)

        return new_response

    @classmethod
    def _get_output_schema(cls, agent: Agent[Any]) -> AgentOutputSchema | None:
        if agent.output_type is None or agent.output_type is str:
            return None

        return AgentOutputSchema(agent.output_type)

    @classmethod
    def _get_handoffs(cls, agent: Agent[Any]) -> list[Handoff]:
        handoffs = []
        for handoff_item in agent.handoffs:
            if isinstance(handoff_item, Handoff):
                handoffs.append(handoff_item)
            elif isinstance(handoff_item, Agent):
                handoffs.append(handoff(handoff_item))
        return handoffs

    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model:
        if isinstance(run_config.model, Model):
            return run_config.model
        elif isinstance(run_config.model, str):
            return run_config.model_provider.get_model(run_config.model)
        elif isinstance(agent.model, Model):
            return agent.model

        return run_config.model_provider.get_model(agent.model)


<file_info>
path: tests/test_agent_tracing.py
name: test_agent_tracing.py
</file_info>
from __future__ import annotations

import asyncio

import pytest
from inline_snapshot import snapshot

from agents import Agent, RunConfig, Runner, trace

from .fake_model import FakeModel
from .test_responses import get_text_message
from .testing_processor import assert_no_traces, fetch_normalized_spans


@pytest.mark.asyncio
async def test_single_run_is_single_trace():
    agent = Agent(
        name="test_agent",
        model=FakeModel(
            initial_output=[get_text_message("first_test")],
        ),
    )

    await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multiple_runs_are_multiple_traces():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
        ]
    )
    agent = Agent(
        name="test_agent_1",
        model=model,
    )

    await Runner.run(agent, input="first_test")
    await Runner.run(agent, input="second_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            },
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_wrapped_trace_is_single_trace():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
            [get_text_message("third_test")],
        ]
    )
    with trace(workflow_name="test_workflow"):
        agent = Agent(
            name="test_agent_1",
            model=model,
        )

        await Runner.run(agent, input="first_test")
        await Runner.run(agent, input="second_test")
        await Runner.run(agent, input="third_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test_workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_parent_disabled_trace_disabled_agent_trace():
    with trace(workflow_name="test_workflow", disabled=True):
        agent = Agent(
            name="test_agent",
            model=FakeModel(
                initial_output=[get_text_message("first_test")],
            ),
        )

        await Runner.run(agent, input="first_test")

    assert_no_traces()


@pytest.mark.asyncio
async def test_manual_disabling_works():
    agent = Agent(
        name="test_agent",
        model=FakeModel(
            initial_output=[get_text_message("first_test")],
        ),
    )

    await Runner.run(agent, input="first_test", run_config=RunConfig(tracing_disabled=True))

    assert_no_traces()


@pytest.mark.asyncio
async def test_trace_config_works():
    agent = Agent(
        name="test_agent",
        model=FakeModel(
            initial_output=[get_text_message("first_test")],
        ),
    )

    await Runner.run(
        agent,
        input="first_test",
        run_config=RunConfig(workflow_name="Foo bar", group_id="123", trace_id="trace_456"),
    )

    assert fetch_normalized_spans(keep_trace_id=True) == snapshot(
        [
            {
                "id": "trace_456",
                "workflow_name": "Foo bar",
                "group_id": "123",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_not_starting_streaming_creates_trace():
    agent = Agent(
        name="test_agent",
        model=FakeModel(
            initial_output=[get_text_message("first_test")],
        ),
    )

    result = Runner.run_streamed(agent, input="first_test")

    # Purposely don't await the stream
    while True:
        if result.is_complete:
            break
        await asyncio.sleep(0.1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            }
        ]
    )

    # Await the stream to avoid warnings about it not being awaited
    async for _ in result.stream_events():
        pass


@pytest.mark.asyncio
async def test_streaming_single_run_is_single_trace():
    agent = Agent(
        name="test_agent",
        model=FakeModel(
            initial_output=[get_text_message("first_test")],
        ),
    )

    x = Runner.run_streamed(agent, input="first_test")
    async for _ in x.stream_events():
        pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multiple_streamed_runs_are_multiple_traces():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
        ]
    )
    agent = Agent(
        name="test_agent_1",
        model=model,
    )

    x = Runner.run_streamed(agent, input="first_test")
    async for _ in x.stream_events():
        pass

    x = Runner.run_streamed(agent, input="second_test")
    async for _ in x.stream_events():
        pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            },
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    }
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_wrapped_streaming_trace_is_single_trace():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
            [get_text_message("third_test")],
        ]
    )
    with trace(workflow_name="test_workflow"):
        agent = Agent(
            name="test_agent_1",
            model=model,
        )

        x = Runner.run_streamed(agent, input="first_test")
        async for _ in x.stream_events():
            pass

        x = Runner.run_streamed(agent, input="second_test")
        async for _ in x.stream_events():
            pass

        x = Runner.run_streamed(agent, input="third_test")
        async for _ in x.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test_workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_wrapped_mixed_trace_is_single_trace():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
            [get_text_message("third_test")],
        ]
    )
    with trace(workflow_name="test_workflow"):
        agent = Agent(
            name="test_agent_1",
            model=model,
        )

        x = Runner.run_streamed(agent, input="first_test")
        async for _ in x.stream_events():
            pass

        await Runner.run(agent, input="second_test")

        x = Runner.run_streamed(agent, input="third_test")
        async for _ in x.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test_workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_parent_disabled_trace_disables_streaming_agent_trace():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
        ]
    )
    with trace(workflow_name="test_workflow", disabled=True):
        agent = Agent(
            name="test_agent",
            model=model,
        )

        x = Runner.run_streamed(agent, input="first_test")
        async for _ in x.stream_events():
            pass

    assert_no_traces()


@pytest.mark.asyncio
async def test_manual_streaming_disabling_works():
    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [get_text_message("first_test")],
            [get_text_message("second_test")],
        ]
    )
    agent = Agent(
        name="test_agent",
        model=model,
    )

    x = Runner.run_streamed(agent, input="first_test", run_config=RunConfig(tracing_disabled=True))
    async for _ in x.stream_events():
        pass

    assert_no_traces()


<file_info>
path: examples/financial_research_agent/agents/financials_agent.py
name: financials_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

# A sub‑agent focused on analyzing a company's fundamentals.
FINANCIALS_PROMPT = (
    "You are a financial analyst focused on company fundamentals such as revenue, "
    "profit, margins and growth trajectory. Given a collection of web (and optional file) "
    "search results about a company, write a concise analysis of its recent financial "
    "performance. Pull out key metrics or quotes. Keep it under 2 paragraphs."
)


class AnalysisSummary(BaseModel):
    summary: str
    """Short text summary for this aspect of the analysis."""


financials_agent = Agent(
    name="FundamentalsAnalystAgent",
    instructions=FINANCIALS_PROMPT,
    output_type=AnalysisSummary,
)


<file_info>
path: examples/financial_research_agent/manager.py
name: manager.py
</file_info>
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from rich.console import Console

from agents import Runner, RunResult, custom_span, gen_trace_id, trace

from .agents.financials_agent import financials_agent
from .agents.planner_agent import FinancialSearchItem, FinancialSearchPlan, planner_agent
from .agents.risk_agent import risk_agent
from .agents.search_agent import search_agent
from .agents.verifier_agent import VerificationResult, verifier_agent
from .agents.writer_agent import FinancialReportData, writer_agent
from .printer import Printer


async def _summary_extractor(run_result: RunResult) -> str:
    """Custom output extractor for sub‑agents that return an AnalysisSummary."""
    # The financial/risk analyst agents emit an AnalysisSummary with a `summary` field.
    # We want the tool call to return just that summary text so the writer can drop it inline.
    return str(run_result.final_output.summary)


class FinancialResearchManager:
    """
    Orchestrates the full flow: planning, searching, sub‑analysis, writing, and verification.
    """

    def __init__(self) -> None:
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Financial research trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"View trace: https://platform.openai.com/traces/{trace_id}",
                is_done=True,
                hide_checkmark=True,
            )
            self.printer.update_item("start", "Starting financial research...", is_done=True)
            search_plan = await self._plan_searches(query)
            search_results = await self._perform_searches(search_plan)
            report = await self._write_report(query, search_results)
            verification = await self._verify_report(report)

            final_report = f"Report summary\n\n{report.short_summary}"
            self.printer.update_item("final_report", final_report, is_done=True)

            self.printer.end()

        # Print to stdout
        print("\n\n=====REPORT=====\n\n")
        print(f"Report:\n{report.markdown_report}")
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        print("\n".join(report.follow_up_questions))
        print("\n\n=====VERIFICATION=====\n\n")
        print(verification)

    async def _plan_searches(self, query: str) -> FinancialSearchPlan:
        self.printer.update_item("planning", "Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        self.printer.update_item(
            "planning",
            f"Will perform {len(result.final_output.searches)} searches",
            is_done=True,
        )
        return result.final_output_as(FinancialSearchPlan)

    async def _perform_searches(self, search_plan: FinancialSearchPlan) -> Sequence[str]:
        with custom_span("Search the web"):
            self.printer.update_item("searching", "Searching...")
            tasks = [asyncio.create_task(self._search(item)) for item in search_plan.searches]
            results: list[str] = []
            num_completed = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                self.printer.update_item(
                    "searching", f"Searching... {num_completed}/{len(tasks)} completed"
                )
            self.printer.mark_item_done("searching")
            return results

    async def _search(self, item: FinancialSearchItem) -> str | None:
        input_data = f"Search term: {item.query}\nReason: {item.reason}"
        try:
            result = await Runner.run(search_agent, input_data)
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(self, query: str, search_results: Sequence[str]) -> FinancialReportData:
        # Expose the specialist analysts as tools so the writer can invoke them inline
        # and still produce the final FinancialReportData output.
        fundamentals_tool = financials_agent.as_tool(
            tool_name="fundamentals_analysis",
            tool_description="Use to get a short write‑up of key financial metrics",
            custom_output_extractor=_summary_extractor,
        )
        risk_tool = risk_agent.as_tool(
            tool_name="risk_analysis",
            tool_description="Use to get a short write‑up of potential red flags",
            custom_output_extractor=_summary_extractor,
        )
        writer_with_tools = writer_agent.clone(tools=[fundamentals_tool, risk_tool])
        self.printer.update_item("writing", "Thinking about report...")
        input_data = f"Original query: {query}\nSummarized search results: {search_results}"
        result = Runner.run_streamed(writer_with_tools, input_data)
        update_messages = [
            "Planning report structure...",
            "Writing sections...",
            "Finalizing report...",
        ]
        last_update = time.time()
        next_message = 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_message < len(update_messages):
                self.printer.update_item("writing", update_messages[next_message])
                next_message += 1
                last_update = time.time()
        self.printer.mark_item_done("writing")
        return result.final_output_as(FinancialReportData)

    async def _verify_report(self, report: FinancialReportData) -> VerificationResult:
        self.printer.update_item("verifying", "Verifying report...")
        result = await Runner.run(verifier_agent, report.markdown_report)
        self.printer.mark_item_done("verifying")
        return result.final_output_as(VerificationResult)


<file_info>
path: examples/handoffs/message_filter_streaming.py
name: message_filter_streaming.py
</file_info>
from __future__ import annotations

import json
import random

from agents import Agent, HandoffInputData, Runner, function_tool, handoff, trace
from agents.extensions import handoff_filters


@function_tool
def random_number_tool(max: int) -> int:
    """Return a random integer between 0 and the given maximum."""
    return random.randint(0, max)


def spanish_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    # First, we'll remove any tool-related messages from the message history
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # Second, we'll also remove the first two items from the history, just for demonstration
    history = (
        tuple(handoff_message_data.input_history[2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )


first_agent = Agent(
    name="Assistant",
    instructions="Be extremely concise.",
    tools=[random_number_tool],
)

spanish_agent = Agent(
    name="Spanish Assistant",
    instructions="You only speak Spanish and are extremely concise.",
    handoff_description="A Spanish-speaking assistant.",
)

second_agent = Agent(
    name="Assistant",
    instructions=(
        "Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant."
    ),
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
)


async def main():
    # Trace the entire run as a single workflow
    with trace(workflow_name="Streaming message filter"):
        # 1. Send a regular message to the first agent
        result = await Runner.run(first_agent, input="Hi, my name is Sora.")

        print("Step 1 done")

        # 2. Ask it to generate a number
        result = await Runner.run(
            first_agent,
            input=result.to_input_list()
            + [{"content": "Can you generate a random number between 0 and 100?", "role": "user"}],
        )

        print("Step 2 done")

        # 3. Call the second agent
        result = await Runner.run(
            second_agent,
            input=result.to_input_list()
            + [
                {
                    "content": "I live in New York City. Whats the population of the city?",
                    "role": "user",
                }
            ],
        )

        print("Step 3 done")

        # 4. Cause a handoff to occur
        stream_result = Runner.run_streamed(
            second_agent,
            input=result.to_input_list()
            + [
                {
                    "content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?",
                    "role": "user",
                }
            ],
        )
        async for _ in stream_result.stream_events():
            pass

        print("Step 4 done")

    print("\n===Final messages===\n")

    # 5. That should have caused spanish_handoff_message_filter to be called, which means the
    # output should be missing the first two messages, and have no tool calls.
    # Let's print the messages to see what happened
    for item in stream_result.to_input_list():
        print(json.dumps(item, indent=2))
        """
        $python examples/handoffs/message_filter_streaming.py
        Step 1 done
        Step 2 done
        Step 3 done
        Tu nombre y lugar de residencia no los tengo disponibles. Solo sé que mencionaste vivir en la ciudad de Nueva York.
        Step 4 done

        ===Final messages===

        {
            "content": "Can you generate a random number between 0 and 100?",
            "role": "user"
            }
            {
            "id": "...",
            "content": [
                {
                "annotations": [],
                "text": "Sure! Here's a random number between 0 and 100: **37**.",
                "type": "output_text"
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message"
            }
            {
            "content": "I live in New York City. Whats the population of the city?",
            "role": "user"
            }
            {
            "id": "...",
            "content": [
                {
                "annotations": [],
                "text": "As of the latest estimates, New York City's population is approximately 8.5 million people. Would you like more information about the city?",
                "type": "output_text"
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message"
            }
            {
            "content": "Por favor habla en espa\u00f1ol. \u00bfCu\u00e1l es mi nombre y d\u00f3nde vivo?",
            "role": "user"
            }
            {
            "id": "...",
            "content": [
                {
                "annotations": [],
                "text": "No s\u00e9 tu nombre, pero me dijiste que vives en Nueva York.",
                "type": "output_text"
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message"
            }
        """


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


<file_info>
path: src/agents/version.py
name: version.py
</file_info>
import importlib.metadata

try:
    __version__ = importlib.metadata.version("agents")
except importlib.metadata.PackageNotFoundError:
    # Fallback if running from source without being installed
    __version__ = "0.0.0"


<file_info>
path: src/agents/util/_transforms.py
name: _transforms.py
</file_info>
import re


def transform_string_function_style(name: str) -> str:
    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)

    return name.lower()


<file_info>
path: src/agents/_run_impl.py
name: _run_impl.py
</file_info>
from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
)
from openai.types.responses.response_computer_tool_call import (
    ActionClick,
    ActionDoubleClick,
    ActionDrag,
    ActionKeypress,
    ActionMove,
    ActionScreenshot,
    ActionScroll,
    ActionType,
    ActionWait,
)
from openai.types.responses.response_input_param import ComputerCallOutput
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from .agent import Agent, ToolsToFinalOutputResult
from .agent_output import AgentOutputSchema
from .computer import AsyncComputer, Computer
from .exceptions import AgentsException, ModelBehaviorError, UserError
from .guardrail import InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult
from .handoffs import Handoff, HandoffInputData
from .items import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    ReasoningItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from .lifecycle import RunHooks
from .logger import logger
from .model_settings import ModelSettings
from .models.interface import ModelTracing
from .run_context import RunContextWrapper, TContext
from .stream_events import RunItemStreamEvent, StreamEvent
from .tool import ComputerTool, FunctionTool, FunctionToolResult, Tool
from .tracing import (
    SpanError,
    Trace,
    function_span,
    get_current_trace,
    guardrail_span,
    handoff_span,
    trace,
)
from .util import _coro, _error_tracing

if TYPE_CHECKING:
    from .run import RunConfig


class QueueCompleteSentinel:
    pass


QUEUE_COMPLETE_SENTINEL = QueueCompleteSentinel()

_NOT_FINAL_OUTPUT = ToolsToFinalOutputResult(is_final_output=False, final_output=None)


@dataclass
class ToolRunHandoff:
    handoff: Handoff
    tool_call: ResponseFunctionToolCall


@dataclass
class ToolRunFunction:
    tool_call: ResponseFunctionToolCall
    function_tool: FunctionTool


@dataclass
class ToolRunComputerAction:
    tool_call: ResponseComputerToolCall
    computer_tool: ComputerTool


@dataclass
class ProcessedResponse:
    new_items: list[RunItem]
    handoffs: list[ToolRunHandoff]
    functions: list[ToolRunFunction]
    computer_actions: list[ToolRunComputerAction]

    def has_tools_to_run(self) -> bool:
        # Handoffs, functions and computer actions need local processing
        # Hosted tools have already run, so there's nothing to do.
        return any(
            [
                self.handoffs,
                self.functions,
                self.computer_actions,
            ]
        )


@dataclass
class NextStepHandoff:
    new_agent: Agent[Any]


@dataclass
class NextStepFinalOutput:
    output: Any


@dataclass
class NextStepRunAgain:
    pass


@dataclass
class SingleStepResult:
    original_input: str | list[TResponseInputItem]
    """The input items i.e. the items before run() was called. May be mutated by handoff input
    filters."""

    model_response: ModelResponse
    """The model response for the current step."""

    pre_step_items: list[RunItem]
    """Items generated before the current step."""

    new_step_items: list[RunItem]
    """Items generated during this current step."""

    next_step: NextStepHandoff | NextStepFinalOutput | NextStepRunAgain
    """The next step to take."""

    @property
    def generated_items(self) -> list[RunItem]:
        """Items generated during the agent run (i.e. everything generated after
        `original_input`)."""
        return self.pre_step_items + self.new_step_items


def get_model_tracing_impl(
    tracing_disabled: bool, trace_include_sensitive_data: bool
) -> ModelTracing:
    if tracing_disabled:
        return ModelTracing.DISABLED
    elif trace_include_sensitive_data:
        return ModelTracing.ENABLED
    else:
        return ModelTracing.ENABLED_WITHOUT_DATA


class RunImpl:
    @classmethod
    async def execute_tools_and_side_effects(
        cls,
        *,
        agent: Agent[TContext],
        # The original input to the Runner
        original_input: str | list[TResponseInputItem],
        # Everything generated by Runner since the original input, but before the current step
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        processed_response: ProcessedResponse,
        output_schema: AgentOutputSchema | None,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        # Make a copy of the generated items
        pre_step_items = list(pre_step_items)

        new_step_items: list[RunItem] = []
        new_step_items.extend(processed_response.new_items)

        # First, lets run the tool calls - function tools and computer actions
        function_results, computer_results = await asyncio.gather(
            cls.execute_function_tool_calls(
                agent=agent,
                tool_runs=processed_response.functions,
                hooks=hooks,
                context_wrapper=context_wrapper,
                config=run_config,
            ),
            cls.execute_computer_actions(
                agent=agent,
                actions=processed_response.computer_actions,
                hooks=hooks,
                context_wrapper=context_wrapper,
                config=run_config,
            ),
        )
        new_step_items.extend([result.run_item for result in function_results])
        new_step_items.extend(computer_results)

        # Reset tool_choice to "auto" after tool execution to prevent infinite loops
        if processed_response.functions or processed_response.computer_actions:
            tools = agent.tools

            if (
                run_config.model_settings and
                cls._should_reset_tool_choice(run_config.model_settings, tools)
            ):
                # update the run_config model settings with a copy
                new_run_config_settings = dataclasses.replace(
                    run_config.model_settings,
                    tool_choice="auto"
                )
                run_config = dataclasses.replace(run_config, model_settings=new_run_config_settings)

            if cls._should_reset_tool_choice(agent.model_settings, tools):
                # Create a modified copy instead of modifying the original agent
                new_model_settings = dataclasses.replace(
                    agent.model_settings,
                    tool_choice="auto"
                )
                agent = dataclasses.replace(agent, model_settings=new_model_settings)

        # Second, check if there are any handoffs
        if run_handoffs := processed_response.handoffs:
            return await cls.execute_handoffs(
                agent=agent,
                original_input=original_input,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                new_response=new_response,
                run_handoffs=run_handoffs,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )

        # Third, we'll check if the tool use should result in a final output
        check_tool_use = await cls._check_for_final_output_from_tools(
            agent=agent,
            tool_results=function_results,
            context_wrapper=context_wrapper,
            config=run_config,
        )

        if check_tool_use.is_final_output:
            # If the output type is str, then let's just stringify it
            if not agent.output_type or agent.output_type is str:
                check_tool_use.final_output = str(check_tool_use.final_output)

            if check_tool_use.final_output is None:
                logger.error(
                    "Model returned a final output of None. Not raising an error because we assume"
                    "you know what you're doing."
                )

            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=check_tool_use.final_output,
                hooks=hooks,
                context_wrapper=context_wrapper,
            )

        # Now we can check if the model also produced a final output
        message_items = [item for item in new_step_items if isinstance(item, MessageOutputItem)]

        # We'll use the last content output as the final output
        potential_final_output_text = (
            ItemHelpers.extract_last_text(message_items[-1].raw_item) if message_items else None
        )

        # There are two possibilities that lead to a final output:
        # 1. Structured output schema => always leads to a final output
        # 2. Plain text output schema => only leads to a final output if there are no tool calls
        if output_schema and not output_schema.is_plain_text() and potential_final_output_text:
            final_output = output_schema.validate_json(potential_final_output_text)
            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=final_output,
                hooks=hooks,
                context_wrapper=context_wrapper,
            )
        elif (
            not output_schema or output_schema.is_plain_text()
        ) and not processed_response.has_tools_to_run():
            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=potential_final_output_text or "",
                hooks=hooks,
                context_wrapper=context_wrapper,
            )
        else:
            # If there's no final output, we can just run again
            return SingleStepResult(
                original_input=original_input,
                model_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                next_step=NextStepRunAgain(),
            )

    @classmethod
    def _should_reset_tool_choice(cls, model_settings: ModelSettings, tools: list[Tool]) -> bool:
        if model_settings is None or model_settings.tool_choice is None:
            return False

        # for specific tool choices
        if (
            isinstance(model_settings.tool_choice, str) and
            model_settings.tool_choice not in ["auto", "required", "none"]
        ):
            return True

        # for one tool and required tool choice
        if model_settings.tool_choice == "required":
            return len(tools) == 1

        return False

    @classmethod
    def process_model_response(
        cls,
        *,
        agent: Agent[Any],
        response: ModelResponse,
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> ProcessedResponse:
        items: list[RunItem] = []

        run_handoffs = []
        functions = []
        computer_actions = []

        handoff_map = {handoff.tool_name: handoff for handoff in handoffs}
        function_map = {tool.name: tool for tool in agent.tools if isinstance(tool, FunctionTool)}
        computer_tool = next((tool for tool in agent.tools if isinstance(tool, ComputerTool)), None)

        for output in response.output:
            if isinstance(output, ResponseOutputMessage):
                items.append(MessageOutputItem(raw_item=output, agent=agent))
            elif isinstance(output, ResponseFileSearchToolCall):
                items.append(ToolCallItem(raw_item=output, agent=agent))
            elif isinstance(output, ResponseFunctionWebSearch):
                items.append(ToolCallItem(raw_item=output, agent=agent))
            elif isinstance(output, ResponseReasoningItem):
                items.append(ReasoningItem(raw_item=output, agent=agent))
            elif isinstance(output, ResponseComputerToolCall):
                items.append(ToolCallItem(raw_item=output, agent=agent))
                if not computer_tool:
                    _error_tracing.attach_error_to_current_span(
                        SpanError(
                            message="Computer tool not found",
                            data={},
                        )
                    )
                    raise ModelBehaviorError(
                        "Model produced computer action without a computer tool."
                    )
                computer_actions.append(
                    ToolRunComputerAction(tool_call=output, computer_tool=computer_tool)
                )
            elif not isinstance(output, ResponseFunctionToolCall):
                logger.warning(f"Unexpected output type, ignoring: {type(output)}")
                continue

            # At this point we know it's a function tool call
            if not isinstance(output, ResponseFunctionToolCall):
                continue

            # Handoffs
            if output.name in handoff_map:
                items.append(HandoffCallItem(raw_item=output, agent=agent))
                handoff = ToolRunHandoff(
                    tool_call=output,
                    handoff=handoff_map[output.name],
                )
                run_handoffs.append(handoff)
            # Regular function tool call
            else:
                if output.name not in function_map:
                    _error_tracing.attach_error_to_current_span(
                        SpanError(
                            message="Tool not found",
                            data={"tool_name": output.name},
                        )
                    )
                    raise ModelBehaviorError(f"Tool {output.name} not found in agent {agent.name}")
                items.append(ToolCallItem(raw_item=output, agent=agent))
                functions.append(
                    ToolRunFunction(
                        tool_call=output,
                        function_tool=function_map[output.name],
                    )
                )

        return ProcessedResponse(
            new_items=items,
            handoffs=run_handoffs,
            functions=functions,
            computer_actions=computer_actions,
        )

    @classmethod
    async def execute_function_tool_calls(
        cls,
        *,
        agent: Agent[TContext],
        tool_runs: list[ToolRunFunction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        config: RunConfig,
    ) -> list[FunctionToolResult]:
        async def run_single_tool(
            func_tool: FunctionTool, tool_call: ResponseFunctionToolCall
        ) -> Any:
            with function_span(func_tool.name) as span_fn:
                if config.trace_include_sensitive_data:
                    span_fn.span_data.input = tool_call.arguments
                try:
                    _, _, result = await asyncio.gather(
                        hooks.on_tool_start(context_wrapper, agent, func_tool),
                        (
                            agent.hooks.on_tool_start(context_wrapper, agent, func_tool)
                            if agent.hooks
                            else _coro.noop_coroutine()
                        ),
                        func_tool.on_invoke_tool(context_wrapper, tool_call.arguments),
                    )

                    await asyncio.gather(
                        hooks.on_tool_end(context_wrapper, agent, func_tool, result),
                        (
                            agent.hooks.on_tool_end(context_wrapper, agent, func_tool, result)
                            if agent.hooks
                            else _coro.noop_coroutine()
                        ),
                    )
                except Exception as e:
                    _error_tracing.attach_error_to_current_span(
                        SpanError(
                            message="Error running tool",
                            data={"tool_name": func_tool.name, "error": str(e)},
                        )
                    )
                    if isinstance(e, AgentsException):
                        raise e
                    raise UserError(f"Error running tool {func_tool.name}: {e}") from e

                if config.trace_include_sensitive_data:
                    span_fn.span_data.output = result
            return result

        tasks = []
        for tool_run in tool_runs:
            function_tool = tool_run.function_tool
            tasks.append(run_single_tool(function_tool, tool_run.tool_call))

        results = await asyncio.gather(*tasks)

        return [
            FunctionToolResult(
                tool=tool_run.function_tool,
                output=result,
                run_item=ToolCallOutputItem(
                    output=result,
                    raw_item=ItemHelpers.tool_call_output_item(tool_run.tool_call, str(result)),
                    agent=agent,
                ),
            )
            for tool_run, result in zip(tool_runs, results)
        ]

    @classmethod
    async def execute_computer_actions(
        cls,
        *,
        agent: Agent[TContext],
        actions: list[ToolRunComputerAction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        config: RunConfig,
    ) -> list[RunItem]:
        results: list[RunItem] = []
        # Need to run these serially, because each action can affect the computer state
        for action in actions:
            results.append(
                await ComputerAction.execute(
                    agent=agent,
                    action=action,
                    hooks=hooks,
                    context_wrapper=context_wrapper,
                    config=config,
                )
            )

        return results

    @classmethod
    async def execute_handoffs(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        new_response: ModelResponse,
        run_handoffs: list[ToolRunHandoff],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        # If there is more than one handoff, add tool responses that reject those handoffs
        if len(run_handoffs) > 1:
            output_message = "Multiple handoffs detected, ignoring this one."
            new_step_items.extend(
                [
                    ToolCallOutputItem(
                        output=output_message,
                        raw_item=ItemHelpers.tool_call_output_item(
                            handoff.tool_call, output_message
                        ),
                        agent=agent,
                    )
                    for handoff in run_handoffs[1:]
                ]
            )

        actual_handoff = run_handoffs[0]
        with handoff_span(from_agent=agent.name) as span_handoff:
            handoff = actual_handoff.handoff
            new_agent: Agent[Any] = await handoff.on_invoke_handoff(
                context_wrapper, actual_handoff.tool_call.arguments
            )
            span_handoff.span_data.to_agent = new_agent.name

            # Append a tool output item for the handoff
            new_step_items.append(
                HandoffOutputItem(
                    agent=agent,
                    raw_item=ItemHelpers.tool_call_output_item(
                        actual_handoff.tool_call,
                        handoff.get_transfer_message(new_agent),
                    ),
                    source_agent=agent,
                    target_agent=new_agent,
                )
            )

            # Execute handoff hooks
            await asyncio.gather(
                hooks.on_handoff(
                    context=context_wrapper,
                    from_agent=agent,
                    to_agent=new_agent,
                ),
                (
                    agent.hooks.on_handoff(
                        context_wrapper,
                        agent=new_agent,
                        source=agent,
                    )
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
            )

            # If there's an input filter, filter the input for the next agent
            input_filter = handoff.input_filter or (
                run_config.handoff_input_filter if run_config else None
            )
            if input_filter:
                logger.debug("Filtering inputs for handoff")
                handoff_input_data = HandoffInputData(
                    input_history=tuple(original_input)
                    if isinstance(original_input, list)
                    else original_input,
                    pre_handoff_items=tuple(pre_step_items),
                    new_items=tuple(new_step_items),
                )
                if not callable(input_filter):
                    _error_tracing.attach_error_to_span(
                        span_handoff,
                        SpanError(
                            message="Invalid input filter",
                            data={"details": "not callable()"},
                        ),
                    )
                    raise UserError(f"Invalid input filter: {input_filter}")
                filtered = input_filter(handoff_input_data)
                if not isinstance(filtered, HandoffInputData):
                    _error_tracing.attach_error_to_span(
                        span_handoff,
                        SpanError(
                            message="Invalid input filter result",
                            data={"details": "not a HandoffInputData"},
                        ),
                    )
                    raise UserError(f"Invalid input filter result: {filtered}")

                original_input = (
                    filtered.input_history
                    if isinstance(filtered.input_history, str)
                    else list(filtered.input_history)
                )
                pre_step_items = list(filtered.pre_handoff_items)
                new_step_items = list(filtered.new_items)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepHandoff(new_agent),
        )

    @classmethod
    async def execute_final_output(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        new_response: ModelResponse,
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        final_output: Any,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> SingleStepResult:
        # Run the on_end hooks
        await cls.run_final_output_hooks(agent, hooks, context_wrapper, final_output)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepFinalOutput(final_output),
        )

    @classmethod
    async def run_final_output_hooks(
        cls,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        final_output: Any,
    ):
        await asyncio.gather(
            hooks.on_agent_end(context_wrapper, agent, final_output),
            agent.hooks.on_end(context_wrapper, agent, final_output)
            if agent.hooks
            else _coro.noop_coroutine(),
        )

    @classmethod
    async def run_single_input_guardrail(
        cls,
        agent: Agent[Any],
        guardrail: InputGuardrail[TContext],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
    ) -> InputGuardrailResult:
        with guardrail_span(guardrail.get_name()) as span_guardrail:
            result = await guardrail.run(agent, input, context)
            span_guardrail.span_data.triggered = result.output.tripwire_triggered
            return result

    @classmethod
    async def run_single_output_guardrail(
        cls,
        guardrail: OutputGuardrail[TContext],
        agent: Agent[Any],
        agent_output: Any,
        context: RunContextWrapper[TContext],
    ) -> OutputGuardrailResult:
        with guardrail_span(guardrail.get_name()) as span_guardrail:
            result = await guardrail.run(agent=agent, agent_output=agent_output, context=context)
            span_guardrail.span_data.triggered = result.output.tripwire_triggered
            return result

    @classmethod
    def stream_step_result_to_queue(
        cls,
        step_result: SingleStepResult,
        queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel],
    ):
        for item in step_result.new_step_items:
            if isinstance(item, MessageOutputItem):
                event = RunItemStreamEvent(item=item, name="message_output_created")
            elif isinstance(item, HandoffCallItem):
                event = RunItemStreamEvent(item=item, name="handoff_requested")
            elif isinstance(item, HandoffOutputItem):
                event = RunItemStreamEvent(item=item, name="handoff_occured")
            elif isinstance(item, ToolCallItem):
                event = RunItemStreamEvent(item=item, name="tool_called")
            elif isinstance(item, ToolCallOutputItem):
                event = RunItemStreamEvent(item=item, name="tool_output")
            elif isinstance(item, ReasoningItem):
                event = RunItemStreamEvent(item=item, name="reasoning_item_created")
            else:
                logger.warning(f"Unexpected item type: {type(item)}")
                event = None

            if event:
                queue.put_nowait(event)

    @classmethod
    async def _check_for_final_output_from_tools(
        cls,
        *,
        agent: Agent[TContext],
        tool_results: list[FunctionToolResult],
        context_wrapper: RunContextWrapper[TContext],
        config: RunConfig,
    ) -> ToolsToFinalOutputResult:
        """Returns (i, final_output)."""
        if not tool_results:
            return _NOT_FINAL_OUTPUT

        if agent.tool_use_behavior == "run_llm_again":
            return _NOT_FINAL_OUTPUT
        elif agent.tool_use_behavior == "stop_on_first_tool":
            return ToolsToFinalOutputResult(
                is_final_output=True, final_output=tool_results[0].output
            )
        elif isinstance(agent.tool_use_behavior, dict):
            names = agent.tool_use_behavior.get("stop_at_tool_names", [])
            for tool_result in tool_results:
                if tool_result.tool.name in names:
                    return ToolsToFinalOutputResult(
                        is_final_output=True, final_output=tool_result.output
                    )
            return ToolsToFinalOutputResult(is_final_output=False, final_output=None)
        elif callable(agent.tool_use_behavior):
            if inspect.iscoroutinefunction(agent.tool_use_behavior):
                return await cast(
                    Awaitable[ToolsToFinalOutputResult],
                    agent.tool_use_behavior(context_wrapper, tool_results),
                )
            else:
                return cast(
                    ToolsToFinalOutputResult, agent.tool_use_behavior(context_wrapper, tool_results)
                )

        logger.error(f"Invalid tool_use_behavior: {agent.tool_use_behavior}")
        raise UserError(f"Invalid tool_use_behavior: {agent.tool_use_behavior}")


class TraceCtxManager:
    """Creates a trace only if there is no current trace, and manages the trace lifecycle."""

    def __init__(
        self,
        workflow_name: str,
        trace_id: str | None,
        group_id: str | None,
        metadata: dict[str, Any] | None,
        disabled: bool,
    ):
        self.trace: Trace | None = None
        self.workflow_name = workflow_name
        self.trace_id = trace_id
        self.group_id = group_id
        self.metadata = metadata
        self.disabled = disabled

    def __enter__(self) -> TraceCtxManager:
        current_trace = get_current_trace()
        if not current_trace:
            self.trace = trace(
                workflow_name=self.workflow_name,
                trace_id=self.trace_id,
                group_id=self.group_id,
                metadata=self.metadata,
                disabled=self.disabled,
            )
            self.trace.start(mark_as_current=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            self.trace.finish(reset_current=True)


class ComputerAction:
    @classmethod
    async def execute(
        cls,
        *,
        agent: Agent[TContext],
        action: ToolRunComputerAction,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        config: RunConfig,
    ) -> RunItem:
        output_func = (
            cls._get_screenshot_async(action.computer_tool.computer, action.tool_call)
            if isinstance(action.computer_tool.computer, AsyncComputer)
            else cls._get_screenshot_sync(action.computer_tool.computer, action.tool_call)
        )

        _, _, output = await asyncio.gather(
            hooks.on_tool_start(context_wrapper, agent, action.computer_tool),
            (
                agent.hooks.on_tool_start(context_wrapper, agent, action.computer_tool)
                if agent.hooks
                else _coro.noop_coroutine()
            ),
            output_func,
        )

        await asyncio.gather(
            hooks.on_tool_end(context_wrapper, agent, action.computer_tool, output),
            (
                agent.hooks.on_tool_end(context_wrapper, agent, action.computer_tool, output)
                if agent.hooks
                else _coro.noop_coroutine()
            ),
        )

        # TODO: don't send a screenshot every single time, use references
        image_url = f"data:image/png;base64,{output}"
        return ToolCallOutputItem(
            agent=agent,
            output=image_url,
            raw_item=ComputerCallOutput(
                call_id=action.tool_call.call_id,
                output={
                    "type": "computer_screenshot",
                    "image_url": image_url,
                },
                type="computer_call_output",
            ),
        )

    @classmethod
    async def _get_screenshot_sync(
        cls,
        computer: Computer,
        tool_call: ResponseComputerToolCall,
    ) -> str:
        action = tool_call.action
        if isinstance(action, ActionClick):
            computer.click(action.x, action.y, action.button)
        elif isinstance(action, ActionDoubleClick):
            computer.double_click(action.x, action.y)
        elif isinstance(action, ActionDrag):
            computer.drag([(p.x, p.y) for p in action.path])
        elif isinstance(action, ActionKeypress):
            computer.keypress(action.keys)
        elif isinstance(action, ActionMove):
            computer.move(action.x, action.y)
        elif isinstance(action, ActionScreenshot):
            computer.screenshot()
        elif isinstance(action, ActionScroll):
            computer.scroll(action.x, action.y, action.scroll_x, action.scroll_y)
        elif isinstance(action, ActionType):
            computer.type(action.text)
        elif isinstance(action, ActionWait):
            computer.wait()

        return computer.screenshot()

    @classmethod
    async def _get_screenshot_async(
        cls,
        computer: AsyncComputer,
        tool_call: ResponseComputerToolCall,
    ) -> str:
        action = tool_call.action
        if isinstance(action, ActionClick):
            await computer.click(action.x, action.y, action.button)
        elif isinstance(action, ActionDoubleClick):
            await computer.double_click(action.x, action.y)
        elif isinstance(action, ActionDrag):
            await computer.drag([(p.x, p.y) for p in action.path])
        elif isinstance(action, ActionKeypress):
            await computer.keypress(action.keys)
        elif isinstance(action, ActionMove):
            await computer.move(action.x, action.y)
        elif isinstance(action, ActionScreenshot):
            await computer.screenshot()
        elif isinstance(action, ActionScroll):
            await computer.scroll(action.x, action.y, action.scroll_x, action.scroll_y)
        elif isinstance(action, ActionType):
            await computer.type(action.text)
        elif isinstance(action, ActionWait):
            await computer.wait()

        return await computer.screenshot()


<file_info>
path: tests/test_tracing_errors.py
name: test_tracing_errors.py
</file_info>
from __future__ import annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)
from .testing_processor import fetch_normalized_spans


@pytest.mark.asyncio
async def test_single_turn_model_error():
    model = FakeModel(tracing_enabled=True)
    model.set_next_output(ValueError("test error"))

    agent = Agent(
        name="test_agent",
        model=model,
    )
    with pytest.raises(ValueError):
        await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                        "children": [
                            {
                                "type": "generation",
                                "error": {
                                    "message": "Error",
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multi_turn_no_handoffs():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: error
            ValueError("test error"),
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    with pytest.raises(ValueError):
        await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "foo",
                                    "input": '{"a": "b"}',
                                    "output": "tool_result",
                                },
                            },
                            {
                                "type": "generation",
                                "error": {
                                    "message": "Error",
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_tool_call_error():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result", hide_errors=True)],
    )

    model.set_next_output(
        [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")],
    )

    with pytest.raises(ModelBehaviorError):
        await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "error": {
                                    "message": "Error running tool",
                                    "data": {
                                        "tool_name": "foo",
                                        "error": "Invalid JSON input for tool foo: bad_json",
                                    },
                                },
                                "data": {"name": "foo", "input": "bad_json"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    result = await Runner.run(agent_3, input="user_message")
    assert result.last_agent == agent_1, "should have picked first handoff"

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test",
                            "handoffs": ["test", "test"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {"type": "handoff", "data": {"from_agent": "test", "to_agent": "test"}},
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [{"type": "generation"}],
                    },
                ],
            }
        ]
    )


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_doesnt_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

    model.set_next_output(
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ]
    )

    result = await Runner.run(agent_1, input="user_message")
    assert result.final_output == Foo(bar="abc")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "Foo"},
                        "children": [{"type": "generation"}],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
    )
    agent_2 = Agent(
        name="test_agent_2",
        model=model,
        handoffs=[agent_1],
        tools=[get_function_tool("some_function", "result")],
    )
    agent_3 = Agent(
        name="test_agent_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Fourth turn: handoff
            [get_handoff_tool_call(agent_3)],
            # Fifth turn: text message
            [get_text_message("done")],
        ]
    )
    result = await Runner.run(agent_3, input="user_message")

    assert result.last_agent == agent_3, (
        f"should have ended on the third agent, got {result.last_agent.name}"
    )

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": ["test_agent_3"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
                                "data": {"from_agent": "test_agent_1", "to_agent": "test_agent_3"},
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [{"type": "generation"}],
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_max_turns_exceeded():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test",
        model=model,
        output_type=Foo,
        tools=[get_function_tool("foo", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
        ]
    )

    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, input="user_message", max_turns=2)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Max turns exceeded", "data": {"max_turns": 2}},
                        "data": {
                            "name": "test",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "Foo",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                        ],
                    }
                ],
            }
        ]
    )


def guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_guardrail_error():
    agent = Agent(
        name="test", input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)]
    )
    model = FakeModel()
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, input="user_message")

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Guardrail tripwire triggered",
                            "data": {"guardrail": "guardrail_function"},
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
                                "data": {"name": "guardrail_function", "triggered": True},
                            }
                        ],
                    }
                ],
            }
        ]
    )


<file_info>
path: tests/test_agent_config.py
name: test_agent_config.py
</file_info>
import pytest
from pydantic import BaseModel

from agents import Agent, Handoff, RunContextWrapper, Runner, handoff


@pytest.mark.asyncio
async def test_system_instructions():
    agent = Agent[None](
        name="test",
        instructions="abc123",
    )
    context = RunContextWrapper(None)

    assert await agent.get_system_prompt(context) == "abc123"

    def sync_instructions(agent: Agent[None], context: RunContextWrapper[None]) -> str:
        return "sync_123"

    agent = agent.clone(instructions=sync_instructions)
    assert await agent.get_system_prompt(context) == "sync_123"

    async def async_instructions(agent: Agent[None], context: RunContextWrapper[None]) -> str:
        return "async_123"

    agent = agent.clone(instructions=async_instructions)
    assert await agent.get_system_prompt(context) == "async_123"


@pytest.mark.asyncio
async def test_handoff_with_agents():
    agent_1 = Agent(
        name="agent_1",
    )

    agent_2 = Agent(
        name="agent_2",
    )

    agent_3 = Agent(
        name="agent_3",
        handoffs=[agent_1, agent_2],
    )

    handoffs = Runner._get_handoffs(agent_3)
    assert len(handoffs) == 2

    assert handoffs[0].agent_name == "agent_1"
    assert handoffs[1].agent_name == "agent_2"

    first_return = await handoffs[0].on_invoke_handoff(RunContextWrapper(None), "")
    assert first_return == agent_1

    second_return = await handoffs[1].on_invoke_handoff(RunContextWrapper(None), "")
    assert second_return == agent_2


@pytest.mark.asyncio
async def test_handoff_with_handoff_obj():
    agent_1 = Agent(
        name="agent_1",
    )

    agent_2 = Agent(
        name="agent_2",
    )

    agent_3 = Agent(
        name="agent_3",
        handoffs=[
            handoff(agent_1),
            handoff(
                agent_2,
                tool_name_override="transfer_to_2",
                tool_description_override="description_2",
            ),
        ],
    )

    handoffs = Runner._get_handoffs(agent_3)
    assert len(handoffs) == 2

    assert handoffs[0].agent_name == "agent_1"
    assert handoffs[1].agent_name == "agent_2"

    assert handoffs[0].tool_name == Handoff.default_tool_name(agent_1)
    assert handoffs[1].tool_name == "transfer_to_2"

    assert handoffs[0].tool_description == Handoff.default_tool_description(agent_1)
    assert handoffs[1].tool_description == "description_2"

    first_return = await handoffs[0].on_invoke_handoff(RunContextWrapper(None), "")
    assert first_return == agent_1

    second_return = await handoffs[1].on_invoke_handoff(RunContextWrapper(None), "")
    assert second_return == agent_2


@pytest.mark.asyncio
async def test_handoff_with_handoff_obj_and_agent():
    agent_1 = Agent(
        name="agent_1",
    )

    agent_2 = Agent(
        name="agent_2",
    )

    agent_3 = Agent(
        name="agent_3",
        handoffs=[handoff(agent_1), agent_2],
    )

    handoffs = Runner._get_handoffs(agent_3)
    assert len(handoffs) == 2

    assert handoffs[0].agent_name == "agent_1"
    assert handoffs[1].agent_name == "agent_2"

    assert handoffs[0].tool_name == Handoff.default_tool_name(agent_1)
    assert handoffs[1].tool_name == Handoff.default_tool_name(agent_2)

    assert handoffs[0].tool_description == Handoff.default_tool_description(agent_1)
    assert handoffs[1].tool_description == Handoff.default_tool_description(agent_2)

    first_return = await handoffs[0].on_invoke_handoff(RunContextWrapper(None), "")
    assert first_return == agent_1

    second_return = await handoffs[1].on_invoke_handoff(RunContextWrapper(None), "")
    assert second_return == agent_2


@pytest.mark.asyncio
async def test_agent_cloning():
    agent = Agent(
        name="test",
        handoff_description="test_description",
        model="o3-mini",
    )

    cloned = agent.clone(
        handoff_description="new_description",
        model="o1",
    )

    assert cloned.name == "test"
    assert cloned.handoff_description == "new_description"
    assert cloned.model == "o1"


class Foo(BaseModel):
    bar: str


@pytest.mark.asyncio
async def test_agent_final_output():
    agent = Agent(
        name="test",
        output_type=Foo,
    )

    schema = Runner._get_output_schema(agent)
    assert schema is not None
    assert schema.output_type == Foo
    assert schema.strict_json_schema is True
    assert schema.json_schema() is not None
    assert not schema.is_plain_text()


<file_info>
path: tests/test_doc_parsing.py
name: test_doc_parsing.py
</file_info>
from agents.function_schema import generate_func_documentation


def func_foo_google(a: int, b: float) -> str:
    """
    This is func_foo.

    Args:
        a: The first argument.
        b: The second argument.

    Returns:
        A result
    """

    return "ok"


def func_foo_numpy(a: int, b: float) -> str:
    """
    This is func_foo.

    Parameters
    ----------
    a: int
        The first argument.
    b: float
        The second argument.

    Returns
    -------
    str
        A result
    """
    return "ok"


def func_foo_sphinx(a: int, b: float) -> str:
    """
    This is func_foo.

    :param a: The first argument.
    :param b: The second argument.
    :return: A result
    """
    return "ok"


class Bar:
    def func_bar(self, a: int, b: float) -> str:
        """
        This is func_bar.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            A result
        """
        return "ok"

    @classmethod
    def func_baz(cls, a: int, b: float) -> str:
        """
        This is func_baz.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            A result
        """
        return "ok"


def test_functions_are_ok():
    func_foo_google(1, 2.0)
    func_foo_numpy(1, 2.0)
    func_foo_sphinx(1, 2.0)
    Bar().func_bar(1, 2.0)
    Bar.func_baz(1, 2.0)


def test_auto_detection() -> None:
    doc = generate_func_documentation(func_foo_google)
    assert doc.name == "func_foo_google"
    assert doc.description == "This is func_foo."
    assert doc.param_descriptions == {"a": "The first argument.", "b": "The second argument."}

    doc = generate_func_documentation(func_foo_numpy)
    assert doc.name == "func_foo_numpy"
    assert doc.description == "This is func_foo."
    assert doc.param_descriptions == {"a": "The first argument.", "b": "The second argument."}

    doc = generate_func_documentation(func_foo_sphinx)
    assert doc.name == "func_foo_sphinx"
    assert doc.description == "This is func_foo."
    assert doc.param_descriptions == {"a": "The first argument.", "b": "The second argument."}


def test_instance_method() -> None:
    bar = Bar()
    doc = generate_func_documentation(bar.func_bar)
    assert doc.name == "func_bar"
    assert doc.description == "This is func_bar."
    assert doc.param_descriptions == {"a": "The first argument.", "b": "The second argument."}


def test_classmethod() -> None:
    doc = generate_func_documentation(Bar.func_baz)
    assert doc.name == "func_baz"
    assert doc.description == "This is func_baz."
    assert doc.param_descriptions == {"a": "The first argument.", "b": "The second argument."}


<file_info>
path: tests/test_agent_hooks.py
name: test_agent_hooks.py
</file_info>
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pytest
from typing_extensions import TypedDict

from agents.agent import Agent
from agents.lifecycle import AgentHooks
from agents.run import Runner
from agents.run_context import RunContextWrapper, TContext
from agents.tool import Tool

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)


class AgentHooksForTests(AgentHooks):
    def __init__(self):
        self.events: dict[str, int] = defaultdict(int)

    def reset(self):
        self.events.clear()

    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        self.events["on_start"] += 1

    async def on_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        self.events["on_end"] += 1

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        source: Agent[TContext],
    ) -> None:
        self.events["on_handoff"] += 1

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
    ) -> None:
        self.events["on_tool_start"] += 1

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: str,
    ) -> None:
        self.events["on_tool_end"] += 1


@pytest.mark.asyncio
async def test_non_streamed_agent_hooks():
    hooks = AgentHooksForTests()
    model = FakeModel()
    agent_1 = Agent(
        name="test_1",
        model=model,
    )
    agent_2 = Agent(
        name="test_2",
        model=model,
    )
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        hooks=hooks,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_text_message("user_message")])
    output = await Runner.run(agent_3, input="user_message")
    assert hooks.events == {"on_start": 1, "on_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message")

    # Shouldn't have on_end because it's not the last agent
    assert hooks.events == {
        "on_start": 1,  # Agent runs once
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: text message
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message")

    assert hooks.events == {
        "on_start": 2,  # Agent runs twice
        "on_tool_start": 2,  # Only one tool call
        "on_tool_end": 2,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_end": 1,  # Agent 3 is the last agent
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


@pytest.mark.asyncio
async def test_streamed_agent_hooks():
    hooks = AgentHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        hooks=hooks,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_text_message("user_message")])
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass
    assert hooks.events == {"on_start": 1, "on_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass

    # Shouldn't have on_end because it's not the last agent
    assert hooks.events == {
        "on_start": 1,  # Agent runs twice
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: text message
            [get_text_message("done")],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass

    assert hooks.events == {
        "on_start": 2,  # Agent runs twice
        "on_tool_start": 2,  # Only one tool call
        "on_tool_end": 2,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_end": 1,  # Agent 3 is the last agent
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


class Foo(TypedDict):
    a: str


@pytest.mark.asyncio
async def test_structed_output_non_streamed_agent_hooks():
    hooks = AgentHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        hooks=hooks,
        output_type=Foo,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_final_output_message(json.dumps({"a": "b"}))])
    output = await Runner.run(agent_3, input="user_message")
    assert hooks.events == {"on_start": 1, "on_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: end message (for agent 1)
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message")

    # Shouldn't have on_end because it's not the last agent
    assert hooks.events == {
        "on_start": 1,  # Agent runs twice
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: end message (for agent 3)
            [get_final_output_message(json.dumps({"a": "b"}))],
        ]
    )
    await Runner.run(agent_3, input="user_message")

    assert hooks.events == {
        "on_start": 2,  # Agent runs twice
        "on_tool_start": 2,  # Only one tool call
        "on_tool_end": 2,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_end": 1,  # Agent 3 is the last agent
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


@pytest.mark.asyncio
async def test_structed_output_streamed_agent_hooks():
    hooks = AgentHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        hooks=hooks,
        output_type=Foo,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_final_output_message(json.dumps({"a": "b"}))])
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass
    assert hooks.events == {"on_start": 1, "on_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: end message (for agent 1)
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message")
    # Shouldn't have on_end because it's not the last agent
    assert hooks.events == {
        "on_start": 1,  # Agent runs twice
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: end message (for agent 3)
            [get_final_output_message(json.dumps({"a": "b"}))],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass

    assert hooks.events == {
        "on_start": 2,  # Agent runs twice
        "on_tool_start": 2,  # 2 tool calls
        "on_tool_end": 2,  # 2 tool calls
        "on_handoff": 1,  # 1 handoff
        "on_end": 1,  # Agent 3 is the last agent
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


class EmptyAgentHooks(AgentHooks):
    pass


@pytest.mark.asyncio
async def test_base_agent_hooks_dont_crash():
    hooks = EmptyAgentHooks()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        hooks=hooks,
        output_type=Foo,
    )
    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_final_output_message(json.dumps({"a": "b"}))])
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: end message (for agent 1)
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message")

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: end message (for agent 3)
            [get_final_output_message(json.dumps({"a": "b"}))],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message")
    async for _ in output.stream_events():
        pass


<file_info>
path: examples/research_bot/manager.py
name: manager.py
</file_info>
from __future__ import annotations

import asyncio
import time

from rich.console import Console

from agents import Runner, custom_span, gen_trace_id, trace

from .agents.planner_agent import WebSearchItem, WebSearchPlan, planner_agent
from .agents.search_agent import search_agent
from .agents.writer_agent import ReportData, writer_agent
from .printer import Printer


class ResearchManager:
    def __init__(self):
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"View trace: https://platform.openai.com/traces/{trace_id}",
                is_done=True,
                hide_checkmark=True,
            )

            self.printer.update_item(
                "starting",
                "Starting research...",
                is_done=True,
                hide_checkmark=True,
            )
            search_plan = await self._plan_searches(query)
            search_results = await self._perform_searches(search_plan)
            report = await self._write_report(query, search_results)

            final_report = f"Report summary\n\n{report.short_summary}"
            self.printer.update_item("final_report", final_report, is_done=True)

            self.printer.end()

        print("\n\n=====REPORT=====\n\n")
        print(f"Report: {report.markdown_report}")
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        follow_up_questions = "\n".join(report.follow_up_questions)
        print(f"Follow up questions: {follow_up_questions}")

    async def _plan_searches(self, query: str) -> WebSearchPlan:
        self.printer.update_item("planning", "Planning searches...")
        result = await Runner.run(
            planner_agent,
            f"Query: {query}",
        )
        self.printer.update_item(
            "planning",
            f"Will perform {len(result.final_output.searches)} searches",
            is_done=True,
        )
        return result.final_output_as(WebSearchPlan)

    async def _perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        with custom_span("Search the web"):
            self.printer.update_item("searching", "Searching...")
            num_completed = 0
            tasks = [asyncio.create_task(self._search(item)) for item in search_plan.searches]
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                self.printer.update_item(
                    "searching", f"Searching... {num_completed}/{len(tasks)} completed"
                )
            self.printer.mark_item_done("searching")
            return results

    async def _search(self, item: WebSearchItem) -> str | None:
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(self, query: str, search_results: list[str]) -> ReportData:
        self.printer.update_item("writing", "Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = Runner.run_streamed(
            writer_agent,
            input,
        )
        update_messages = [
            "Thinking about report...",
            "Planning report structure...",
            "Writing outline...",
            "Creating sections...",
            "Cleaning up formatting...",
            "Finalizing report...",
            "Finishing report...",
        ]

        last_update = time.time()
        next_message = 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_message < len(update_messages):
                self.printer.update_item("writing", update_messages[next_message])
                next_message += 1
                last_update = time.time()

        self.printer.mark_item_done("writing")
        return result.final_output_as(ReportData)


<file_info>
path: src/agents/tracing/span_data.py
name: span_data.py
</file_info>
from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.responses import Response, ResponseInputItemParam


class SpanData(abc.ABC):
    @abc.abstractmethod
    def export(self) -> dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass


class AgentSpanData(SpanData):
    __slots__ = ("name", "handoffs", "tools", "output_type")

    def __init__(
        self,
        name: str,
        handoffs: list[str] | None = None,
        tools: list[str] | None = None,
        output_type: str | None = None,
    ):
        self.name = name
        self.handoffs: list[str] | None = handoffs
        self.tools: list[str] | None = tools
        self.output_type: str | None = output_type

    @property
    def type(self) -> str:
        return "agent"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "handoffs": self.handoffs,
            "tools": self.tools,
            "output_type": self.output_type,
        }


class FunctionSpanData(SpanData):
    __slots__ = ("name", "input", "output")

    def __init__(self, name: str, input: str | None, output: Any | None):
        self.name = name
        self.input = input
        self.output = output

    @property
    def type(self) -> str:
        return "function"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": str(self.output) if self.output else None,
        }


class GenerationSpanData(SpanData):
    __slots__ = (
        "input",
        "output",
        "model",
        "model_config",
        "usage",
    )

    def __init__(
        self,
        input: Sequence[Mapping[str, Any]] | None = None,
        output: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        model_config: Mapping[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ):
        self.input = input
        self.output = output
        self.model = model
        self.model_config = model_config
        self.usage = usage

    @property
    def type(self) -> str:
        return "generation"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": self.output,
            "model": self.model,
            "model_config": self.model_config,
            "usage": self.usage,
        }


class ResponseSpanData(SpanData):
    __slots__ = ("response", "input")

    def __init__(
        self,
        response: Response | None = None,
        input: str | list[ResponseInputItemParam] | None = None,
    ) -> None:
        self.response = response
        # This is not used by the OpenAI trace processors, but is useful for other tracing
        # processor implementations
        self.input = input

    @property
    def type(self) -> str:
        return "response"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "response_id": self.response.id if self.response else None,
        }


class HandoffSpanData(SpanData):
    __slots__ = ("from_agent", "to_agent")

    def __init__(self, from_agent: str | None, to_agent: str | None):
        self.from_agent = from_agent
        self.to_agent = to_agent

    @property
    def type(self) -> str:
        return "handoff"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
        }


class CustomSpanData(SpanData):
    __slots__ = ("name", "data")

    def __init__(self, name: str, data: dict[str, Any]):
        self.name = name
        self.data = data

    @property
    def type(self) -> str:
        return "custom"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
        }


class GuardrailSpanData(SpanData):
    __slots__ = ("name", "triggered")

    def __init__(self, name: str, triggered: bool = False):
        self.name = name
        self.triggered = triggered

    @property
    def type(self) -> str:
        return "guardrail"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "triggered": self.triggered,
        }


class TranscriptionSpanData(SpanData):
    __slots__ = (
        "input",
        "output",
        "model",
        "model_config",
    )

    def __init__(
        self,
        input: str | None = None,
        input_format: str | None = "pcm",
        output: str | None = None,
        model: str | None = None,
        model_config: Mapping[str, Any] | None = None,
    ):
        self.input = input
        self.input_format = input_format
        self.output = output
        self.model = model
        self.model_config = model_config

    @property
    def type(self) -> str:
        return "transcription"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": {
                "data": self.input or "",
                "format": self.input_format,
            },
            "output": self.output,
            "model": self.model,
            "model_config": self.model_config,
        }


class SpeechSpanData(SpanData):
    __slots__ = ("input", "output", "model", "model_config", "first_byte_at")

    def __init__(
        self,
        input: str | None = None,
        output: str | None = None,
        output_format: str | None = "pcm",
        model: str | None = None,
        model_config: Mapping[str, Any] | None = None,
        first_content_at: str | None = None,
    ):
        self.input = input
        self.output = output
        self.output_format = output_format
        self.model = model
        self.model_config = model_config
        self.first_content_at = first_content_at

    @property
    def type(self) -> str:
        return "speech"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": {
                "data": self.output or "",
                "format": self.output_format,
            },
            "model": self.model,
            "model_config": self.model_config,
            "first_content_at": self.first_content_at,
        }


class SpeechGroupSpanData(SpanData):
    __slots__ = "input"

    def __init__(
        self,
        input: str | None = None,
    ):
        self.input = input

    @property
    def type(self) -> str:
        return "speech-group"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
        }


<file_info>
path: src/agents/logger.py
name: logger.py
</file_info>
import logging

logger = logging.getLogger("openai.agents")


<file_info>
path: src/agents/extensions/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/extensions/handoff_prompt.py
name: handoff_prompt.py
</file_info>
# A recommended prompt prefix for agents that use handoffs. We recommend including this or
# similar instructions in any agents that use handoffs.
RECOMMENDED_PROMPT_PREFIX = (
    "# System context\n"
    "You are part of a multi-agent system called the Agents SDK, designed to make agent "
    "coordination and execution easy. Agents uses two primary abstraction: **Agents** and "
    "**Handoffs**. An agent encompasses instructions and tools and can hand off a "
    "conversation to another agent when appropriate. "
    "Handoffs are achieved by calling a handoff function, generally named "
    "`transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background;"
    " do not mention or draw attention to these transfers in your conversation with the user.\n"
)


def prompt_with_handoff_instructions(prompt: str) -> str:
    """
    Add recommended instructions to the prompt for agents that use handoffs.
    """
    return f"{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}"


<file_info>
path: src/agents/voice/events.py
name: events.py
</file_info>
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from typing_extensions import TypeAlias

from .imports import np, npt


@dataclass
class VoiceStreamEventAudio:
    """Streaming event from the VoicePipeline"""

    data: npt.NDArray[np.int16 | np.float32] | None
    """The audio data."""

    type: Literal["voice_stream_event_audio"] = "voice_stream_event_audio"
    """The type of event."""


@dataclass
class VoiceStreamEventLifecycle:
    """Streaming event from the VoicePipeline"""

    event: Literal["turn_started", "turn_ended", "session_ended"]
    """The event that occurred."""

    type: Literal["voice_stream_event_lifecycle"] = "voice_stream_event_lifecycle"
    """The type of event."""


@dataclass
class VoiceStreamEventError:
    """Streaming event from the VoicePipeline"""

    error: Exception
    """The error that occurred."""

    type: Literal["voice_stream_event_error"] = "voice_stream_event_error"
    """The type of event."""


VoiceStreamEvent: TypeAlias = Union[
    VoiceStreamEventAudio, VoiceStreamEventLifecycle, VoiceStreamEventError
]
"""An event from the `VoicePipeline`, streamed via `StreamedAudioResult.stream()`."""


<file_info>
path: src/agents/voice/imports.py
name: imports.py
</file_info>
try:
    import numpy as np
    import numpy.typing as npt
    import websockets
except ImportError as _e:
    raise ImportError(
        "`numpy` + `websockets` are required to use voice. You can install them via the optional "
        "dependency group: `pip install 'openai-agents[voice]'`."
    ) from _e

__all__ = ["np", "npt", "websockets"]


<file_info>
path: src/agents/voice/utils.py
name: utils.py
</file_info>
import re
from typing import Callable


def get_sentence_based_splitter(
    min_sentence_length: int = 20,
) -> Callable[[str], tuple[str, str]]:
    """Returns a function that splits text into chunks based on sentence boundaries.

    Args:
        min_sentence_length: The minimum length of a sentence to be included in a chunk.

    Returns:
        A function that splits text into chunks based on sentence boundaries.
    """

    def sentence_based_text_splitter(text_buffer: str) -> tuple[str, str]:
        """
        A function to split the text into chunks. This is useful if you want to split the text into
        chunks before sending it to the TTS model rather than waiting for the whole text to be
        processed.

        Args:
            text_buffer: The text to split.

        Returns:
            A tuple of the text to process and the remaining text buffer.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text_buffer.strip())
        if len(sentences) >= 1:
            combined_sentences = " ".join(sentences[:-1])
            if len(combined_sentences) >= min_sentence_length:
                remaining_text_buffer = sentences[-1]
                return combined_sentences, remaining_text_buffer
        return "", text_buffer

    return sentence_based_text_splitter


<file_info>
path: src/agents/voice/pipeline_config.py
name: pipeline_config.py
</file_info>
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tracing.util import gen_group_id
from .model import STTModelSettings, TTSModelSettings, VoiceModelProvider
from .models.openai_model_provider import OpenAIVoiceModelProvider


@dataclass
class VoicePipelineConfig:
    """Configuration for a `VoicePipeline`."""

    model_provider: VoiceModelProvider = field(default_factory=OpenAIVoiceModelProvider)
    """The voice model provider to use for the pipeline. Defaults to OpenAI."""

    tracing_disabled: bool = False
    """Whether to disable tracing of the pipeline. Defaults to `False`."""

    trace_include_sensitive_data: bool = True
    """Whether to include sensitive data in traces. Defaults to `True`. This is specifically for the
      voice pipeline, and not for anything that goes on inside your Workflow."""

    trace_include_sensitive_audio_data: bool = True
    """Whether to include audio data in traces. Defaults to `True`."""

    workflow_name: str = "Voice Agent"
    """The name of the workflow to use for tracing. Defaults to `Voice Agent`."""

    group_id: str = field(default_factory=gen_group_id)
    """
    A grouping identifier to use for tracing, to link multiple traces from the same conversation
    or process. If not provided, we will create a random group ID.
    """

    trace_metadata: dict[str, Any] | None = None
    """
    An optional dictionary of additional metadata to include with the trace.
    """

    stt_settings: STTModelSettings = field(default_factory=STTModelSettings)
    """The settings to use for the STT model."""

    tts_settings: TTSModelSettings = field(default_factory=TTSModelSettings)
    """The settings to use for the TTS model."""


<file_info>
path: src/agents/agent.py
name: agent.py
</file_info>
from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, cast

from typing_extensions import TypeAlias, TypedDict

from .guardrail import InputGuardrail, OutputGuardrail
from .handoffs import Handoff
from .items import ItemHelpers
from .logger import logger
from .model_settings import ModelSettings
from .models.interface import Model
from .run_context import RunContextWrapper, TContext
from .tool import FunctionToolResult, Tool, function_tool
from .util import _transforms
from .util._types import MaybeAwaitable

if TYPE_CHECKING:
    from .lifecycle import AgentHooks
    from .result import RunResult


@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    """Whether this is the final output. If False, the LLM will run again and receive the tool call
    output.
    """

    final_output: Any | None = None
    """The final output. Can be None if `is_final_output` is False, otherwise must match the
    `output_type` of the agent.
    """


ToolsToFinalOutputFunction: TypeAlias = Callable[
    [RunContextWrapper[TContext], list[FunctionToolResult]],
    MaybeAwaitable[ToolsToFinalOutputResult],
]
"""A function that takes a run context and a list of tool results, and returns a
`ToolToFinalOutputResult`.
"""


class StopAtTools(TypedDict):
    stop_at_tool_names: list[str]
    """A list of tool names, any of which will stop the agent from running further."""


@dataclass
class Agent(Generic[TContext]):
    """An agent is an AI model configured with instructions, tools, guardrails, handoffs and more.

    We strongly recommend passing `instructions`, which is the "system prompt" for the agent. In
    addition, you can pass `handoff_description`, which is a human-readable description of the
    agent, used when the agent is used inside tools/handoffs.

    Agents are generic on the context type. The context is a (mutable) object you create. It is
    passed to tool functions, handoffs, guardrails, etc.
    """

    name: str
    """The name of the agent."""

    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None
    """The instructions for the agent. Will be used as the "system prompt" when this agent is
    invoked. Describes what the agent should do, and how it responds.

    Can either be a string, or a function that dynamically generates instructions for the agent. If
    you provide a function, it will be called with the context and the agent instance. It must
    return a string.
    """

    handoff_description: str | None = None
    """A description of the agent. This is used when the agent is used as a handoff, so that an
    LLM knows what it does and when to invoke it.
    """

    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)
    """Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs,
    and the agent can choose to delegate to them if relevant. Allows for separation of concerns and
    modularity.
    """

    model: str | Model | None = None
    """The model implementation to use when invoking the LLM.

    By default, if not set, the agent will use the default model configured in
    `model_settings.DEFAULT_MODEL`.
    """

    model_settings: ModelSettings = field(default_factory=ModelSettings)
    """Configures model-specific tuning parameters (e.g. temperature, top_p).
    """

    tools: list[Tool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    """A list of checks that run in parallel to the agent's execution, before generating a
    response. Runs only if the agent is the first agent in the chain.
    """

    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    """A list of checks that run on the final output of the agent, after generating a response.
    Runs only if the agent produces a final output.
    """

    output_type: type[Any] | None = None
    """The type of the output object. If not provided, the output will be `str`."""

    hooks: AgentHooks[TContext] | None = None
    """A class that receives callbacks on various lifecycle events for this agent.
    """

    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction
    ) = "run_llm_again"
    """This lets you configure how tool use is handled.
    - "run_llm_again": The default behavior. Tools are run, and then the LLM receives the results
        and gets to respond.
    - "stop_on_first_tool": The output of the first tool call is used as the final output. This
        means that the LLM does not process the result of the tool call.
    - A list of tool names: The agent will stop running if any of the tools in the list are called.
        The final output will be the output of the first matching tool call. The LLM does not
        process the result of the tool call.
    - A function: If you pass a function, it will be called with the run context and the list of
      tool results. It must return a `ToolToFinalOutputResult`, which determines whether the tool
      calls result in a final output.

      NOTE: This configuration is specific to FunctionTools. Hosted tools, such as file search,
      web search, etc are always processed by the LLM.
    """

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Make a copy of the agent, with the given arguments changed. For example, you could do:
        ```
        new_agent = agent.clone(instructions="New instructions")
        ```
        """
        return dataclasses.replace(self, **kwargs)

    def as_tool(
        self,
        tool_name: str | None,
        tool_description: str | None,
        custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
    ) -> Tool:
        """Transform this agent into a tool, callable by other agents.

        This is different from handoffs in two ways:
        1. In handoffs, the new agent receives the conversation history. In this tool, the new agent
           receives generated input.
        2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is
           called as a tool, and the conversation is continued by the original agent.

        Args:
            tool_name: The name of the tool. If not provided, the agent's name will be used.
            tool_description: The description of the tool, which should indicate what it does and
                when to use it.
            custom_output_extractor: A function that extracts the output from the agent. If not
                provided, the last message from the agent will be used.
        """

        @function_tool(
            name_override=tool_name or _transforms.transform_string_function_style(self.name),
            description_override=tool_description or "",
        )
        async def run_agent(context: RunContextWrapper, input: str) -> str:
            from .run import Runner

            output = await Runner.run(
                starting_agent=self,
                input=input,
                context=context.context,
            )
            if custom_output_extractor:
                return await custom_output_extractor(output)

            return ItemHelpers.text_message_outputs(output.new_items)

        return run_agent

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            else:
                return cast(str, self.instructions(run_context, self))
        elif self.instructions is not None:
            logger.error(f"Instructions must be a string or a function, got {self.instructions}")

        return None


<file_info>
path: tests/test_tool_converter.py
name: test_tool_converter.py
</file_info>
import pytest
from pydantic import BaseModel

from agents import Agent, Handoff, function_tool, handoff
from agents.exceptions import UserError
from agents.models.openai_chatcompletions import ToolConverter
from agents.tool import FileSearchTool, WebSearchTool


def some_function(a: str, b: list[int]) -> str:
    return "hello"


def test_to_openai_with_function_tool():
    some_function(a="foo", b=[1, 2, 3])

    tool = function_tool(some_function)
    result = ToolConverter.to_openai(tool)

    assert result["type"] == "function"
    assert result["function"]["name"] == "some_function"
    params = result.get("function", {}).get("parameters")
    assert params is not None
    properties = params.get("properties", {})
    assert isinstance(properties, dict)
    assert properties.keys() == {"a", "b"}


class Foo(BaseModel):
    a: str
    b: list[int]


def test_convert_handoff_tool():
    agent = Agent(name="test_1", handoff_description="test_2")
    handoff_obj = handoff(agent=agent)
    result = ToolConverter.convert_handoff_tool(handoff_obj)

    assert result["type"] == "function"
    assert result["function"]["name"] == Handoff.default_tool_name(agent)
    assert result["function"].get("description") == Handoff.default_tool_description(agent)
    params = result.get("function", {}).get("parameters")
    assert params is not None

    for key, value in handoff_obj.input_json_schema.items():
        assert params[key] == value


def test_tool_converter_hosted_tools_errors():
    with pytest.raises(UserError):
        ToolConverter.to_openai(WebSearchTool())

    with pytest.raises(UserError):
        ToolConverter.to_openai(FileSearchTool(vector_store_ids=["abc"], max_num_results=1))


<file_info>
path: tests/test_strict_schema.py
name: test_strict_schema.py
</file_info>
import pytest

from agents.exceptions import UserError
from agents.strict_schema import ensure_strict_json_schema


def test_empty_schema_has_additional_properties_false():
    strict_schema = ensure_strict_json_schema({})
    assert strict_schema["additionalProperties"] is False


def test_non_dict_schema_errors():
    with pytest.raises(TypeError):
        ensure_strict_json_schema([])  # type: ignore


def test_object_without_additional_properties():
    # When an object type schema has properties but no additionalProperties,
    # it should be added and the "required" list set from the property keys.
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    result = ensure_strict_json_schema(schema)
    assert result["type"] == "object"
    assert result["additionalProperties"] is False
    assert result["required"] == ["a"]
    # The inner property remains unchanged (no additionalProperties is added for non-object types)
    assert result["properties"]["a"] == {"type": "string"}


def test_object_with_true_additional_properties():
    # If additionalProperties is explicitly set to True for an object, a UserError should be raised.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "additionalProperties": True,
    }
    with pytest.raises(UserError):
        ensure_strict_json_schema(schema)


def test_array_items_processing_and_default_removal():
    # When processing an array, the items schema is processed recursively.
    # Also, any "default": None should be removed.
    schema = {
        "type": "array",
        "items": {"type": "number", "default": None},
    }
    result = ensure_strict_json_schema(schema)
    # "default" should be stripped from the items schema.
    assert "default" not in result["items"]
    assert result["items"]["type"] == "number"


def test_anyOf_processing():
    # Test that anyOf schemas are processed.
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "number", "default": None},
        ]
    }
    result = ensure_strict_json_schema(schema)
    # For the first variant: object type should get additionalProperties and required keys set.
    variant0 = result["anyOf"][0]
    assert variant0["type"] == "object"
    assert variant0["additionalProperties"] is False
    assert variant0["required"] == ["a"]

    # For the second variant: the "default": None should be removed.
    variant1 = result["anyOf"][1]
    assert variant1["type"] == "number"
    assert "default" not in variant1


def test_allOf_single_entry_merging():
    # When an allOf list has a single entry, its content should be merged into the parent.
    schema = {
        "type": "object",
        "allOf": [{"properties": {"a": {"type": "boolean"}}}],
    }
    result = ensure_strict_json_schema(schema)
    # allOf should be removed and merged.
    assert "allOf" not in result
    # The object should now have additionalProperties set and required set.
    assert result["additionalProperties"] is False
    assert result["required"] == ["a"]
    assert "a" in result["properties"]
    assert result["properties"]["a"]["type"] == "boolean"


def test_default_removal_on_non_object():
    # Test that "default": None is stripped from schemas that are not objects.
    schema = {"type": "string", "default": None}
    result = ensure_strict_json_schema(schema)
    assert result["type"] == "string"
    assert "default" not in result


def test_ref_expansion():
    # Construct a schema with a definitions section and a property with a $ref.
    schema = {
        "definitions": {"refObj": {"type": "string", "default": None}},
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/refObj", "description": "desc"}},
    }
    result = ensure_strict_json_schema(schema)
    a_schema = result["properties"]["a"]
    # The $ref should be expanded so that the type is from the referenced definition,
    # the description from the original takes precedence, and default is removed.
    assert a_schema["type"] == "string"
    assert a_schema["description"] == "desc"
    assert "default" not in a_schema


def test_ref_no_expansion_when_alone():
    # If the schema only contains a $ref key, it should not be expanded.
    schema = {"$ref": "#/definitions/refObj"}
    result = ensure_strict_json_schema(schema)
    # Because there is only one key, the $ref remains unchanged.
    assert result == {"$ref": "#/definitions/refObj"}


def test_invalid_ref_format():
    # A $ref that does not start with "#/" should trigger a ValueError when resolved.
    schema = {"type": "object", "properties": {"a": {"$ref": "invalid", "description": "desc"}}}
    with pytest.raises(ValueError):
        ensure_strict_json_schema(schema)


<file_info>
path: tests/voice/conftest.py
name: conftest.py
</file_info>
import os
import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.version_info[:2] == (3, 9):
        this_dir = os.path.dirname(__file__)
        skip_marker = pytest.mark.skip(reason="Skipped on Python 3.9")

        for item in items:
            if item.fspath.dirname.startswith(this_dir):
                item.add_marker(skip_marker)


<file_info>
path: tests/voice/test_openai_stt.py
name: test_openai_stt.py
</file_info>
# test_openai_stt_transcription_session.py

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

try:
    from agents.voice import OpenAISTTTranscriptionSession, StreamedAudioInput, STTModelSettings
    from agents.voice.exceptions import STTWebsocketConnectionError
    from agents.voice.models.openai_stt import EVENT_INACTIVITY_TIMEOUT

    from .fake_models import FakeStreamedAudioInput
except ImportError:
    pass


# ===== Helpers =====


def create_mock_websocket(messages: list[str]) -> AsyncMock:
    """
    Creates a mock websocket (AsyncMock) that will return the provided incoming_messages
    from __aiter__() as if they came from the server.
    """

    mock_ws = AsyncMock()
    mock_ws.__aenter__.return_value = mock_ws
    # The incoming_messages are strings that we pretend come from the server
    mock_ws.__aiter__.return_value = iter(messages)
    return mock_ws


def fake_time(increment: int):
    current = 1000
    while True:
        yield current
        current += increment


# ===== Tests =====
@pytest.mark.asyncio
async def test_non_json_messages_should_crash():
    """This tests that non-JSON messages will raise an exception"""
    # Setup: mock websockets.connect
    mock_ws = create_mock_websocket(["not a json message"])
    with patch("websockets.connect", return_value=mock_ws):
        # Instantiate the session
        input_audio = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=input_audio,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        with pytest.raises(STTWebsocketConnectionError):
            # Start reading from transcribe_turns, which triggers _process_websocket_connection
            turns = session.transcribe_turns()

            async for _ in turns:
                pass

        await session.close()


@pytest.mark.asyncio
async def test_session_connects_and_configures_successfully():
    """
    Test that the session:
    1) Connects to the correct URL with correct headers.
    2) Receives a 'session.created' event.
    3) Sends an update message for session config.
    4) Receives a 'session.updated' event.
    """
    # Setup: mock websockets.connect
    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "transcription_session.created"}),
            json.dumps({"type": "transcription_session.updated"}),
        ]
    )
    with patch("websockets.connect", return_value=mock_ws) as mock_connect:
        # Instantiate the session
        input_audio = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=input_audio,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        # Start reading from transcribe_turns, which triggers _process_websocket_connection
        turns = session.transcribe_turns()

        async for _ in turns:
            pass

        # Check connect call
        args, kwargs = mock_connect.call_args
        assert "wss://api.openai.com/v1/realtime?intent=transcription" in args[0]
        headers = kwargs.get("additional_headers", {})
        assert headers.get("Authorization") == "Bearer FAKE_KEY"
        assert headers.get("OpenAI-Beta") == "realtime=v1"
        assert headers.get("OpenAI-Log-Session") == "1"

        # Check that we sent a 'transcription_session.update' message
        sent_messages = [call.args[0] for call in mock_ws.send.call_args_list]
        assert any('"type": "transcription_session.update"' in msg for msg in sent_messages), (
            f"Expected 'transcription_session.update' in {sent_messages}"
        )

        await session.close()


@pytest.mark.asyncio
async def test_stream_audio_sends_correct_json():
    """
    Test that when audio is placed on the input queue, the session:
    1) Base64-encodes the data.
    2) Sends the correct JSON message over the websocket.
    """
    # Simulate a single "transcription_session.created" and "transcription_session.updated" event,
    # before we test streaming.
    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "transcription_session.created"}),
            json.dumps({"type": "transcription_session.updated"}),
        ]
    )

    with patch("websockets.connect", return_value=mock_ws):
        # Prepare
        audio_input = StreamedAudioInput()
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=audio_input,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        # Kick off the transcribe_turns generator
        turn_iter = session.transcribe_turns()
        async for _ in turn_iter:
            pass

        # Now push some audio data

        buffer1 = np.array([1, 2, 3, 4], dtype=np.int16)
        await audio_input.add_audio(buffer1)
        await asyncio.sleep(0.1)  # give time for _stream_audio to consume
        await asyncio.sleep(4)

        # Check that the websocket sent an "input_audio_buffer.append" message
        found_audio_append = False
        for call_arg in mock_ws.send.call_args_list:
            print("call_arg", call_arg)
            print("test", session._turn_audio_buffer)
            sent_str = call_arg.args[0]
            print("sent_str", sent_str)
            if '"type": "input_audio_buffer.append"' in sent_str:
                msg_dict = json.loads(sent_str)
                assert msg_dict["type"] == "input_audio_buffer.append"
                assert "audio" in msg_dict
                found_audio_append = True
        assert found_audio_append, "No 'input_audio_buffer.append' message was sent."

        await session.close()


@pytest.mark.asyncio
async def test_transcription_event_puts_output_in_queue():
    """
    Test that a 'conversation.item.input_audio_transcription.completed' event
    yields a transcript from transcribe_turns().
    """
    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "transcription_session.created"}),
            json.dumps({"type": "transcription_session.updated"}),
            # Once configured, we mock a completed transcription event:
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Hello world!",
                }
            ),
        ]
    )

    with patch("websockets.connect", return_value=mock_ws):
        # Prepare
        audio_input = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=audio_input,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )
        turns = session.transcribe_turns()

        # We'll collect transcribed turns in a list
        collected_turns = []
        async for turn in turns:
            collected_turns.append(turn)
        await session.close()

        # Check we got "Hello world!"
        assert "Hello world!" in collected_turns
        # Cleanup


@pytest.mark.asyncio
async def test_timeout_waiting_for_created_event(monkeypatch):
    """
    If the 'session.created' event does not arrive before SESSION_CREATION_TIMEOUT,
    the session should raise a TimeoutError.
    """
    time_gen = fake_time(increment=30)  # increment by 30 seconds each time

    # Define a replacement function that returns the next time
    def fake_time_func():
        return next(time_gen)

    # Monkey-patch time.time with our fake_time_func
    monkeypatch.setattr(time, "time", fake_time_func)

    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "unknown"}),
        ]
    )  # add a fake event to the mock websocket to make sure it doesn't raise a different exception

    with patch("websockets.connect", return_value=mock_ws):
        audio_input = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=audio_input,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )
        turns = session.transcribe_turns()

        # We expect an exception once the generator tries to connect + wait for event
        with pytest.raises(STTWebsocketConnectionError) as exc_info:
            async for _ in turns:
                pass

            assert "Timeout waiting for transcription_session.created event" in str(exc_info.value)

        await session.close()


@pytest.mark.asyncio
async def test_session_error_event():
    """
    If the session receives an event with "type": "error", it should propagate an exception
    and put an ErrorSentinel in the output queue.
    """
    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "transcription_session.created"}),
            json.dumps({"type": "transcription_session.updated"}),
            # Then an error from the server
            json.dumps({"type": "error", "error": "Simulated server error!"}),
        ]
    )

    with patch("websockets.connect", return_value=mock_ws):
        audio_input = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=audio_input,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        with pytest.raises(STTWebsocketConnectionError) as exc_info:
            turns = session.transcribe_turns()
            async for _ in turns:
                pass

            assert "Simulated server error!" in str(exc_info.value)

        await session.close()


@pytest.mark.asyncio
async def test_inactivity_timeout():
    """
    Test that if no events arrive in EVENT_INACTIVITY_TIMEOUT ms,
    _handle_events breaks out and a SessionCompleteSentinel is placed in the output queue.
    """
    # We'll feed only the creation + updated events. Then do nothing.
    # The handle_events loop should eventually time out.
    mock_ws = create_mock_websocket(
        [
            json.dumps({"type": "unknown"}),
            json.dumps({"type": "unknown"}),
            json.dumps({"type": "transcription_session.created"}),
            json.dumps({"type": "transcription_session.updated"}),
        ]
    )

    # We'll artificially manipulate the "time" to simulate inactivity quickly.
    # The code checks time.time() for inactivity over EVENT_INACTIVITY_TIMEOUT.
    # We'll increment the return_value manually.
    with (
        patch("websockets.connect", return_value=mock_ws),
        patch(
            "time.time",
            side_effect=[
                1000.0,
                1000.0 + EVENT_INACTIVITY_TIMEOUT + 1,
                2000.0 + EVENT_INACTIVITY_TIMEOUT + 1,
                3000.0 + EVENT_INACTIVITY_TIMEOUT + 1,
                9999,
            ],
        ),
    ):
        audio_input = await FakeStreamedAudioInput.get(count=2)
        stt_settings = STTModelSettings()

        session = OpenAISTTTranscriptionSession(
            input=audio_input,
            client=AsyncMock(api_key="FAKE_KEY"),
            model="whisper-1",
            settings=stt_settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        collected_turns: list[str] = []
        with pytest.raises(STTWebsocketConnectionError) as exc_info:
            async for turn in session.transcribe_turns():
                collected_turns.append(turn)

            assert "Timeout waiting for transcription_session" in str(exc_info.value)

            assert len(collected_turns) == 0, "No transcripts expected, but we got something?"

        await session.close()


<file_info>
path: tests/fake_model.py
name: fake_model.py
</file_info>
from __future__ import annotations

from collections.abc import AsyncIterator

from openai.types.responses import Response, ResponseCompletedEvent

from agents.agent_output import AgentOutputSchema
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.tracing import SpanError, generation_span
from agents.usage import Usage


class FakeModel(Model):
    def __init__(
        self,
        tracing_enabled: bool = False,
        initial_output: list[TResponseOutputItem] | Exception | None = None,
    ):
        if initial_output is None:
            initial_output = []
        self.turn_outputs: list[list[TResponseOutputItem] | Exception] = (
            [initial_output] if initial_output else []
        )
        self.tracing_enabled = tracing_enabled

    def set_next_output(self, output: list[TResponseOutputItem] | Exception):
        self.turn_outputs.append(output)

    def add_multiple_turn_outputs(self, outputs: list[list[TResponseOutputItem] | Exception]):
        self.turn_outputs.extend(outputs)

    def get_next_output(self) -> list[TResponseOutputItem] | Exception:
        if not self.turn_outputs:
            return []
        return self.turn_outputs.pop(0)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        with generation_span(disabled=not self.tracing_enabled) as span:
            output = self.get_next_output()

            if isinstance(output, Exception):
                span.set_error(
                    SpanError(
                        message="Error",
                        data={
                            "name": output.__class__.__name__,
                            "message": str(output),
                        },
                    )
                )
                raise output

            return ModelResponse(
                output=output,
                usage=Usage(),
                referenceable_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        with generation_span(disabled=not self.tracing_enabled) as span:
            output = self.get_next_output()
            if isinstance(output, Exception):
                span.set_error(
                    SpanError(
                        message="Error",
                        data={
                            "name": output.__class__.__name__,
                            "message": str(output),
                        },
                    )
                )
                raise output

            yield ResponseCompletedEvent(
                type="response.completed",
                response=get_response_obj(output),
            )


def get_response_obj(output: list[TResponseOutputItem], response_id: str | None = None) -> Response:
    return Response(
        id=response_id or "123",
        created_at=123,
        model="test_model",
        object="response",
        output=output,
        tool_choice="none",
        tools=[],
        top_p=None,
        parallel_tool_calls=False,
    )


<file_info>
path: examples/model_providers/custom_example_agent.py
name: custom_example_agent.py
</file_info>
import asyncio
import os

from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

"""This example uses a custom provider for a specific agent. Steps:
1. Create a custom OpenAI client.
2. Create a `Model` that uses the custom client.
3. Set the `model` on the Agent.

Note that in this example, we disable tracing under the assumption that you don't have an API key
from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
or call set_tracing_export_api_key() to set a tracing specific key.
"""
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# An alternate approach that would also work:
# PROVIDER = OpenAIProvider(openai_client=client)
# agent = Agent(..., model="some-custom-model")
# Runner.run(agent, ..., run_config=RunConfig(model_provider=PROVIDER))


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/financial_research_agent/printer.py
name: printer.py
</file_info>
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner


class Printer:
    """
    Simple wrapper to stream status updates. Used by the financial bot
    manager as it orchestrates planning, search and writing.
    """

    def __init__(self, console: Console) -> None:
        self.live = Live(console=console)
        self.items: dict[str, tuple[str, bool]] = {}
        self.hide_done_ids: set[str] = set()
        self.live.start()

    def end(self) -> None:
        self.live.stop()

    def hide_done_checkmark(self, item_id: str) -> None:
        self.hide_done_ids.add(item_id)

    def update_item(
        self, item_id: str, content: str, is_done: bool = False, hide_checkmark: bool = False
    ) -> None:
        self.items[item_id] = (content, is_done)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self.flush()

    def mark_item_done(self, item_id: str) -> None:
        self.items[item_id] = (self.items[item_id][0], True)
        self.flush()

    def flush(self) -> None:
        renderables: list[Any] = []
        for item_id, (content, is_done) in self.items.items():
            if is_done:
                prefix = "✅ " if item_id not in self.hide_done_ids else ""
                renderables.append(prefix + content)
            else:
                renderables.append(Spinner("dots", text=content))
        self.live.update(Group(*renderables))


<file_info>
path: examples/research_bot/agents/planner_agent.py
name: planner_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

PROMPT = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for."
)


class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""


planner_agent = Agent(
    name="PlannerAgent",
    instructions=PROMPT,
    model="gpt-4o",
    output_type=WebSearchPlan,
)


<file_info>
path: examples/agent_patterns/input_guardrails.py
name: input_guardrails.py
</file_info>
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

"""
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that output messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.
"""


### 1. An agent-based guardrail that is triggered if the user is asking to do math homework
class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input
    is a math homework question.
    """
    result = await Runner.run(guardrail_agent, input, context=context.context)
    final_output = result.final_output_as(MathHomeworkOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_math_homework,
    )


### 2. The run loop


async def main():
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        input_guardrails=[math_guardrail],
    )

    input_data: list[TResponseInputItem] = []

    while True:
        user_input = input("Enter a message: ")
        input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = await Runner.run(agent, input_data)
            print(result.final_output)
            # If the guardrail didn't trigger, we use the result as the input for the next run
            input_data = result.to_input_list()
        except InputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = "Sorry, I can't help you with your math homework."
            print(message)
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    # Sample run:
    # Enter a message: What's the capital of California?
    # The capital of California is Sacramento.
    # Enter a message: Can you help me solve for x: 2x + 5 = 11
    # Sorry, I can't help you with your math homework.


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/scope.py
name: scope.py
</file_info>
# Holds the current active span
import contextvars
from typing import TYPE_CHECKING, Any

from ..logger import logger

if TYPE_CHECKING:
    from .spans import Span
    from .traces import Trace

_current_span: contextvars.ContextVar["Span[Any] | None"] = contextvars.ContextVar(
    "current_span", default=None
)

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)


class Scope:
    @classmethod
    def get_current_span(cls) -> "Span[Any] | None":
        return _current_span.get()

    @classmethod
    def set_current_span(cls, span: "Span[Any] | None") -> "contextvars.Token[Span[Any] | None]":
        return _current_span.set(span)

    @classmethod
    def reset_current_span(cls, token: "contextvars.Token[Span[Any] | None]") -> None:
        _current_span.reset(token)

    @classmethod
    def get_current_trace(cls) -> "Trace | None":
        return _current_trace.get()

    @classmethod
    def set_current_trace(cls, trace: "Trace | None") -> "contextvars.Token[Trace | None]":
        logger.debug(f"Setting current trace: {trace.trace_id if trace else None}")
        return _current_trace.set(trace)

    @classmethod
    def reset_current_trace(cls, token: "contextvars.Token[Trace | None]") -> None:
        logger.debug("Resetting current trace")
        _current_trace.reset(token)


<file_info>
path: src/agents/models/_openai_shared.py
name: _openai_shared.py
</file_info>
from __future__ import annotations

from openai import AsyncOpenAI

_default_openai_key: str | None = None
_default_openai_client: AsyncOpenAI | None = None
_use_responses_by_default: bool = True


def set_default_openai_key(key: str) -> None:
    global _default_openai_key
    _default_openai_key = key


def get_default_openai_key() -> str | None:
    return _default_openai_key


def set_default_openai_client(client: AsyncOpenAI) -> None:
    global _default_openai_client
    _default_openai_client = client


def get_default_openai_client() -> AsyncOpenAI | None:
    return _default_openai_client


def set_use_responses_by_default(use_responses: bool) -> None:
    global _use_responses_by_default
    _use_responses_by_default = use_responses


def get_use_responses_by_default() -> bool:
    return _use_responses_by_default


<file_info>
path: src/agents/result.py
name: result.py
</file_info>
from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypeVar

from ._run_impl import QueueCompleteSentinel
from .agent import Agent
from .agent_output import AgentOutputSchema
from .exceptions import InputGuardrailTripwireTriggered, MaxTurnsExceeded
from .guardrail import InputGuardrailResult, OutputGuardrailResult
from .items import ItemHelpers, ModelResponse, RunItem, TResponseInputItem
from .logger import logger
from .stream_events import StreamEvent
from .tracing import Trace
from .util._pretty_print import pretty_print_result, pretty_print_run_result_streaming

if TYPE_CHECKING:
    from ._run_impl import QueueCompleteSentinel
    from .agent import Agent

T = TypeVar("T")


@dataclass
class RunResultBase(abc.ABC):
    input: str | list[TResponseInputItem]
    """The original input items i.e. the items before run() was called. This may be a mutated
    version of the input, if there are handoff input filters that mutate the input.
    """

    new_items: list[RunItem]
    """The new items generated during the agent run. These include things like new messages, tool
    calls and their outputs, etc.
    """

    raw_responses: list[ModelResponse]
    """The raw LLM responses generated by the model during the agent run."""

    final_output: Any
    """The output of the last agent."""

    input_guardrail_results: list[InputGuardrailResult]
    """Guardrail results for the input messages."""

    output_guardrail_results: list[OutputGuardrailResult]
    """Guardrail results for the final output of the agent."""

    @property
    @abc.abstractmethod
    def last_agent(self) -> Agent[Any]:
        """The last agent that was run."""

    def final_output_as(self, cls: type[T], raise_if_incorrect_type: bool = False) -> T:
        """A convenience method to cast the final output to a specific type. By default, the cast
        is only for the typechecker. If you set `raise_if_incorrect_type` to True, we'll raise a
        TypeError if the final output is not of the given type.

        Args:
            cls: The type to cast the final output to.
            raise_if_incorrect_type: If True, we'll raise a TypeError if the final output is not of
                the given type.

        Returns:
            The final output casted to the given type.
        """
        if raise_if_incorrect_type and not isinstance(self.final_output, cls):
            raise TypeError(f"Final output is not of type {cls.__name__}")

        return cast(T, self.final_output)

    def to_input_list(self) -> list[TResponseInputItem]:
        """Creates a new input list, merging the original input with all the new items generated."""
        original_items: list[TResponseInputItem] = ItemHelpers.input_to_new_input_list(self.input)
        new_items = [item.to_input_item() for item in self.new_items]

        return original_items + new_items


@dataclass
class RunResult(RunResultBase):
    _last_agent: Agent[Any]

    @property
    def last_agent(self) -> Agent[Any]:
        """The last agent that was run."""
        return self._last_agent

    def __str__(self) -> str:
        return pretty_print_result(self)


@dataclass
class RunResultStreaming(RunResultBase):
    """The result of an agent run in streaming mode. You can use the `stream_events` method to
    receive semantic events as they are generated.

    The streaming method will raise:
    - A MaxTurnsExceeded exception if the agent exceeds the max_turns limit.
    - A GuardrailTripwireTriggered exception if a guardrail is tripped.
    """

    current_agent: Agent[Any]
    """The current agent that is running."""

    current_turn: int
    """The current turn number."""

    max_turns: int
    """The maximum number of turns the agent can run for."""

    final_output: Any
    """The final output of the agent. This is None until the agent has finished running."""

    _current_agent_output_schema: AgentOutputSchema | None = field(repr=False)

    _trace: Trace | None = field(repr=False)

    is_complete: bool = False
    """Whether the agent has finished running."""

    # Queues that the background run_loop writes to
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=asyncio.Queue, repr=False
    )
    _input_guardrail_queue: asyncio.Queue[InputGuardrailResult] = field(
        default_factory=asyncio.Queue, repr=False
    )

    # Store the asyncio tasks that we're waiting on
    _run_impl_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _input_guardrails_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _output_guardrails_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _stored_exception: Exception | None = field(default=None, repr=False)

    @property
    def last_agent(self) -> Agent[Any]:
        """The last agent that was run. Updates as the agent run progresses, so the true last agent
        is only available after the agent run is complete.
        """
        return self.current_agent

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream deltas for new items as they are generated. We're using the types from the
        OpenAI Responses API, so these are semantic events: each event has a `type` field that
        describes the type of the event, along with the data for that event.

        This will raise:
        - A MaxTurnsExceeded exception if the agent exceeds the max_turns limit.
        - A GuardrailTripwireTriggered exception if a guardrail is tripped.
        """
        while True:
            self._check_errors()
            if self._stored_exception:
                logger.debug("Breaking due to stored exception")
                self.is_complete = True
                break

            if self.is_complete and self._event_queue.empty():
                break

            try:
                item = await self._event_queue.get()
            except asyncio.CancelledError:
                break

            if isinstance(item, QueueCompleteSentinel):
                self._event_queue.task_done()
                # Check for errors, in case the queue was completed due to an exception
                self._check_errors()
                break

            yield item
            self._event_queue.task_done()

        if self._trace:
            self._trace.finish(reset_current=True)

        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception

    def _check_errors(self):
        if self.current_turn > self.max_turns:
            self._stored_exception = MaxTurnsExceeded(f"Max turns ({self.max_turns}) exceeded")

        # Fetch all the completed guardrail results from the queue and raise if needed
        while not self._input_guardrail_queue.empty():
            guardrail_result = self._input_guardrail_queue.get_nowait()
            if guardrail_result.output.tripwire_triggered:
                self._stored_exception = InputGuardrailTripwireTriggered(guardrail_result)

        # Check the tasks for any exceptions
        if self._run_impl_task and self._run_impl_task.done():
            exc = self._run_impl_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._input_guardrails_task and self._input_guardrails_task.done():
            exc = self._input_guardrails_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._output_guardrails_task and self._output_guardrails_task.done():
            exc = self._output_guardrails_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

    def _cleanup_tasks(self):
        if self._run_impl_task and not self._run_impl_task.done():
            self._run_impl_task.cancel()

        if self._input_guardrails_task and not self._input_guardrails_task.done():
            self._input_guardrails_task.cancel()

        if self._output_guardrails_task and not self._output_guardrails_task.done():
            self._output_guardrails_task.cancel()

    def __str__(self) -> str:
        return pretty_print_run_result_streaming(self)


<file_info>
path: tests/test_output_tool.py
name: test_output_tool.py
</file_info>
import json

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents import Agent, AgentOutputSchema, ModelBehaviorError, Runner, UserError
from agents.agent_output import _WRAPPER_DICT_KEY
from agents.util import _json


def test_plain_text_output():
    agent = Agent(name="test")
    output_schema = Runner._get_output_schema(agent)
    assert not output_schema, "Shouldn't have an output tool config without an output type"

    agent = Agent(name="test", output_type=str)
    assert not output_schema, "Shouldn't have an output tool config with str output type"


class Foo(BaseModel):
    bar: str


def test_structured_output_pydantic():
    agent = Agent(name="test", output_type=Foo)
    output_schema = Runner._get_output_schema(agent)
    assert output_schema, "Should have an output tool config with a structured output type"

    assert output_schema.output_type == Foo, "Should have the correct output type"
    assert not output_schema._is_wrapped, "Pydantic objects should not be wrapped"
    for key, value in Foo.model_json_schema().items():
        assert output_schema.json_schema()[key] == value

    json_str = Foo(bar="baz").model_dump_json()
    validated = output_schema.validate_json(json_str)
    assert validated == Foo(bar="baz")


class Bar(TypedDict):
    bar: str


def test_structured_output_typed_dict():
    agent = Agent(name="test", output_type=Bar)
    output_schema = Runner._get_output_schema(agent)
    assert output_schema, "Should have an output tool config with a structured output type"
    assert output_schema.output_type == Bar, "Should have the correct output type"
    assert not output_schema._is_wrapped, "TypedDicts should not be wrapped"

    json_str = json.dumps(Bar(bar="baz"))
    validated = output_schema.validate_json(json_str)
    assert validated == Bar(bar="baz")


def test_structured_output_list():
    agent = Agent(name="test", output_type=list[str])
    output_schema = Runner._get_output_schema(agent)
    assert output_schema, "Should have an output tool config with a structured output type"
    assert output_schema.output_type == list[str], "Should have the correct output type"
    assert output_schema._is_wrapped, "Lists should be wrapped"

    # This is testing implementation details, but it's useful  to make sure this doesn't break
    json_str = json.dumps({_WRAPPER_DICT_KEY: ["foo", "bar"]})
    validated = output_schema.validate_json(json_str)
    assert validated == ["foo", "bar"]


def test_bad_json_raises_error(mocker):
    agent = Agent(name="test", output_type=Foo)
    output_schema = Runner._get_output_schema(agent)
    assert output_schema, "Should have an output tool config with a structured output type"

    with pytest.raises(ModelBehaviorError):
        output_schema.validate_json("not valid json")

    agent = Agent(name="test", output_type=list[str])
    output_schema = Runner._get_output_schema(agent)
    assert output_schema, "Should have an output tool config with a structured output type"

    mock_validate_json = mocker.patch.object(_json, "validate_json")
    mock_validate_json.return_value = ["foo"]

    with pytest.raises(ModelBehaviorError):
        output_schema.validate_json(json.dumps(["foo"]))

    mock_validate_json.return_value = {"value": "foo"}

    with pytest.raises(ModelBehaviorError):
        output_schema.validate_json(json.dumps(["foo"]))


def test_plain_text_obj_doesnt_produce_schema():
    output_wrapper = AgentOutputSchema(output_type=str)
    with pytest.raises(UserError):
        output_wrapper.json_schema()


def test_structured_output_is_strict():
    output_wrapper = AgentOutputSchema(output_type=Foo)
    assert output_wrapper.strict_json_schema
    for key, value in Foo.model_json_schema().items():
        assert output_wrapper.json_schema()[key] == value

    assert (
        "additionalProperties" in output_wrapper.json_schema()
        and not output_wrapper.json_schema()["additionalProperties"]
    )


def test_setting_strict_false_works():
    output_wrapper = AgentOutputSchema(output_type=Foo, strict_json_schema=False)
    assert not output_wrapper.strict_json_schema
    assert output_wrapper.json_schema() == Foo.model_json_schema()
    assert output_wrapper.json_schema() == Foo.model_json_schema()


<file_info>
path: tests/test_function_tool.py
name: test_function_tool.py
</file_info>
import json
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents import FunctionTool, ModelBehaviorError, RunContextWrapper, function_tool
from agents.tool import default_tool_error_function


def argless_function() -> str:
    return "ok"


@pytest.mark.asyncio
async def test_argless_function():
    tool = function_tool(argless_function)
    assert tool.name == "argless_function"

    result = await tool.on_invoke_tool(RunContextWrapper(None), "")
    assert result == "ok"


def argless_with_context(ctx: RunContextWrapper[str]) -> str:
    return "ok"


@pytest.mark.asyncio
async def test_argless_with_context():
    tool = function_tool(argless_with_context)
    assert tool.name == "argless_with_context"

    result = await tool.on_invoke_tool(RunContextWrapper(None), "")
    assert result == "ok"

    # Extra JSON should not raise an error
    result = await tool.on_invoke_tool(RunContextWrapper(None), '{"a": 1}')
    assert result == "ok"


def simple_function(a: int, b: int = 5):
    return a + b


@pytest.mark.asyncio
async def test_simple_function():
    tool = function_tool(simple_function, failure_error_function=None)
    assert tool.name == "simple_function"

    result = await tool.on_invoke_tool(RunContextWrapper(None), '{"a": 1}')
    assert result == 6

    result = await tool.on_invoke_tool(RunContextWrapper(None), '{"a": 1, "b": 2}')
    assert result == 3

    # Missing required argument should raise an error
    with pytest.raises(ModelBehaviorError):
        await tool.on_invoke_tool(RunContextWrapper(None), "")


class Foo(BaseModel):
    a: int
    b: int = 5


class Bar(TypedDict):
    x: str
    y: int


def complex_args_function(foo: Foo, bar: Bar, baz: str = "hello"):
    return f"{foo.a + foo.b} {bar['x']}{bar['y']} {baz}"


@pytest.mark.asyncio
async def test_complex_args_function():
    tool = function_tool(complex_args_function, failure_error_function=None)
    assert tool.name == "complex_args_function"

    valid_json = json.dumps(
        {
            "foo": Foo(a=1).model_dump(),
            "bar": Bar(x="hello", y=10),
        }
    )
    result = await tool.on_invoke_tool(RunContextWrapper(None), valid_json)
    assert result == "6 hello10 hello"

    valid_json = json.dumps(
        {
            "foo": Foo(a=1, b=2).model_dump(),
            "bar": Bar(x="hello", y=10),
        }
    )
    result = await tool.on_invoke_tool(RunContextWrapper(None), valid_json)
    assert result == "3 hello10 hello"

    valid_json = json.dumps(
        {
            "foo": Foo(a=1, b=2).model_dump(),
            "bar": Bar(x="hello", y=10),
            "baz": "world",
        }
    )
    result = await tool.on_invoke_tool(RunContextWrapper(None), valid_json)
    assert result == "3 hello10 world"

    # Missing required argument should raise an error
    with pytest.raises(ModelBehaviorError):
        await tool.on_invoke_tool(RunContextWrapper(None), '{"foo": {"a": 1}}')


def test_function_config_overrides():
    tool = function_tool(simple_function, name_override="custom_name")
    assert tool.name == "custom_name"

    tool = function_tool(simple_function, description_override="custom description")
    assert tool.description == "custom description"

    tool = function_tool(
        simple_function,
        name_override="custom_name",
        description_override="custom description",
    )
    assert tool.name == "custom_name"
    assert tool.description == "custom description"


def test_func_schema_is_strict():
    tool = function_tool(simple_function)
    assert tool.strict_json_schema, "Should be strict by default"
    assert (
        "additionalProperties" in tool.params_json_schema
        and not tool.params_json_schema["additionalProperties"]
    )

    tool = function_tool(complex_args_function)
    assert tool.strict_json_schema, "Should be strict by default"
    assert (
        "additionalProperties" in tool.params_json_schema
        and not tool.params_json_schema["additionalProperties"]
    )


@pytest.mark.asyncio
async def test_manual_function_tool_creation_works():
    def do_some_work(data: str) -> str:
        return f"{data}_done"

    class FunctionArgs(BaseModel):
        data: str

    async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
        parsed = FunctionArgs.model_validate_json(args)
        return do_some_work(data=parsed.data)

    tool = FunctionTool(
        name="test",
        description="Processes extracted user data",
        params_json_schema=FunctionArgs.model_json_schema(),
        on_invoke_tool=run_function,
    )

    assert tool.name == "test"
    assert tool.description == "Processes extracted user data"
    for key, value in FunctionArgs.model_json_schema().items():
        assert tool.params_json_schema[key] == value
    assert tool.strict_json_schema

    result = await tool.on_invoke_tool(RunContextWrapper(None), '{"data": "hello"}')
    assert result == "hello_done"

    tool_not_strict = FunctionTool(
        name="test",
        description="Processes extracted user data",
        params_json_schema=FunctionArgs.model_json_schema(),
        on_invoke_tool=run_function,
        strict_json_schema=False,
    )

    assert not tool_not_strict.strict_json_schema
    assert "additionalProperties" not in tool_not_strict.params_json_schema

    result = await tool_not_strict.on_invoke_tool(
        RunContextWrapper(None), '{"data": "hello", "bar": "baz"}'
    )
    assert result == "hello_done"


@pytest.mark.asyncio
async def test_function_tool_default_error_works():
    def my_func(a: int, b: int = 5):
        raise ValueError("test")

    tool = function_tool(my_func)
    ctx = RunContextWrapper(None)

    result = await tool.on_invoke_tool(ctx, "")
    assert "Invalid JSON" in str(result)

    result = await tool.on_invoke_tool(ctx, "{}")
    assert "Invalid JSON" in str(result)

    result = await tool.on_invoke_tool(ctx, '{"a": 1}')
    assert result == default_tool_error_function(ctx, ValueError("test"))

    result = await tool.on_invoke_tool(ctx, '{"a": 1, "b": 2}')
    assert result == default_tool_error_function(ctx, ValueError("test"))


@pytest.mark.asyncio
async def test_sync_custom_error_function_works():
    def my_func(a: int, b: int = 5):
        raise ValueError("test")

    def custom_sync_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
        return f"error_{error.__class__.__name__}"

    tool = function_tool(my_func, failure_error_function=custom_sync_error_function)
    ctx = RunContextWrapper(None)

    result = await tool.on_invoke_tool(ctx, "")
    assert result == "error_ModelBehaviorError"

    result = await tool.on_invoke_tool(ctx, "{}")
    assert result == "error_ModelBehaviorError"

    result = await tool.on_invoke_tool(ctx, '{"a": 1}')
    assert result == "error_ValueError"

    result = await tool.on_invoke_tool(ctx, '{"a": 1, "b": 2}')
    assert result == "error_ValueError"


@pytest.mark.asyncio
async def test_async_custom_error_function_works():
    async def my_func(a: int, b: int = 5):
        raise ValueError("test")

    def custom_sync_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
        return f"error_{error.__class__.__name__}"

    tool = function_tool(my_func, failure_error_function=custom_sync_error_function)
    ctx = RunContextWrapper(None)

    result = await tool.on_invoke_tool(ctx, "")
    assert result == "error_ModelBehaviorError"

    result = await tool.on_invoke_tool(ctx, "{}")
    assert result == "error_ModelBehaviorError"

    result = await tool.on_invoke_tool(ctx, '{"a": 1}')
    assert result == "error_ValueError"

    result = await tool.on_invoke_tool(ctx, '{"a": 1, "b": 2}')
    assert result == "error_ValueError"


<file_info>
path: tests/test_openai_chatcompletions_stream.py
name: test_openai_chatcompletions_stream.py
</file_info>
from collections.abc import AsyncIterator

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_response_yields_events_for_text_content(monkeypatch) -> None:
    """
    Validate that `stream_response` emits the correct sequence of events when
    streaming a simple assistant message consisting of plain text content.
    We simulate two chunks of text returned from the chat completion stream.
    """
    # Create two chunks that will be emitted by the fake stream.
    chunk1 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(content="He"))],
    )
    # Mark last chunk with usage so stream_response knows this is final.
    chunk2 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(content="llo"))],
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=7, total_tokens=12),
    )

    async def fake_stream() -> AsyncIterator[ChatCompletionChunk]:
        for c in (chunk1, chunk2):
            yield c

    # Patch _fetch_response to inject our fake stream
    async def patched_fetch_response(self, *args, **kwargs):
        # `_fetch_response` is expected to return a Response skeleton and the async stream
        resp = Response(
            id="resp-id",
            created_at=0,
            model="fake-model",
            object="response",
            output=[],
            tool_choice="none",
            tools=[],
            parallel_tool_calls=False,
        )
        return resp, fake_stream()

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    output_events = []
    async for event in model.stream_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    ):
        output_events.append(event)
    # We expect a response.created, then a response.output_item.added, content part added,
    # two content delta events (for "He" and "llo"), a content part done, the assistant message
    # output_item.done, and finally response.completed.
    # There should be 8 events in total.
    assert len(output_events) == 8
    # First event indicates creation.
    assert output_events[0].type == "response.created"
    # The output item added and content part added events should mark the assistant message.
    assert output_events[1].type == "response.output_item.added"
    assert output_events[2].type == "response.content_part.added"
    # Two text delta events.
    assert output_events[3].type == "response.output_text.delta"
    assert output_events[3].delta == "He"
    assert output_events[4].type == "response.output_text.delta"
    assert output_events[4].delta == "llo"
    # After streaming, the content part and item should be marked done.
    assert output_events[5].type == "response.content_part.done"
    assert output_events[6].type == "response.output_item.done"
    # Last event indicates completion of the stream.
    assert output_events[7].type == "response.completed"
    # The completed response should have one output message with full text.
    completed_resp = output_events[7].response
    assert isinstance(completed_resp.output[0], ResponseOutputMessage)
    assert isinstance(completed_resp.output[0].content[0], ResponseOutputText)
    assert completed_resp.output[0].content[0].text == "Hello"

    assert completed_resp.usage, "usage should not be None"
    assert completed_resp.usage.input_tokens == 7
    assert completed_resp.usage.output_tokens == 5
    assert completed_resp.usage.total_tokens == 12


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_response_yields_events_for_refusal_content(monkeypatch) -> None:
    """
    Validate that when the model streams a refusal string instead of normal content,
    `stream_response` emits the appropriate sequence of events including
    `response.refusal.delta` events for each chunk of the refusal message and
    constructs a completed assistant message with a `ResponseOutputRefusal` part.
    """
    # Simulate refusal text coming in two pieces, like content but using the `refusal`
    # field on the delta rather than `content`.
    chunk1 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(refusal="No"))],
    )
    chunk2 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(refusal="Thanks"))],
        usage=CompletionUsage(completion_tokens=2, prompt_tokens=2, total_tokens=4),
    )

    async def fake_stream() -> AsyncIterator[ChatCompletionChunk]:
        for c in (chunk1, chunk2):
            yield c

    async def patched_fetch_response(self, *args, **kwargs):
        resp = Response(
            id="resp-id",
            created_at=0,
            model="fake-model",
            object="response",
            output=[],
            tool_choice="none",
            tools=[],
            parallel_tool_calls=False,
        )
        return resp, fake_stream()

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    output_events = []
    async for event in model.stream_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    ):
        output_events.append(event)
    # Expect sequence similar to text: created, output_item.added, content part added,
    # two refusal delta events, content part done, output_item.done, completed.
    assert len(output_events) == 8
    assert output_events[0].type == "response.created"
    assert output_events[1].type == "response.output_item.added"
    assert output_events[2].type == "response.content_part.added"
    assert output_events[3].type == "response.refusal.delta"
    assert output_events[3].delta == "No"
    assert output_events[4].type == "response.refusal.delta"
    assert output_events[4].delta == "Thanks"
    assert output_events[5].type == "response.content_part.done"
    assert output_events[6].type == "response.output_item.done"
    assert output_events[7].type == "response.completed"
    completed_resp = output_events[7].response
    assert isinstance(completed_resp.output[0], ResponseOutputMessage)
    refusal_part = completed_resp.output[0].content[0]
    assert isinstance(refusal_part, ResponseOutputRefusal)
    assert refusal_part.refusal == "NoThanks"


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_response_yields_events_for_tool_call(monkeypatch) -> None:
    """
    Validate that `stream_response` emits the correct sequence of events when
    the model is streaming a function/tool call instead of plain text.
    The function call will be split across two chunks.
    """
    # Simulate a single tool call whose ID stays constant and function name/args built over chunks.
    tool_call_delta1 = ChoiceDeltaToolCall(
        index=0,
        id="tool-id",
        function=ChoiceDeltaToolCallFunction(name="my_", arguments="arg1"),
        type="function",
    )
    tool_call_delta2 = ChoiceDeltaToolCall(
        index=0,
        id="tool-id",
        function=ChoiceDeltaToolCallFunction(name="func", arguments="arg2"),
        type="function",
    )
    chunk1 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(tool_calls=[tool_call_delta1]))],
    )
    chunk2 = ChatCompletionChunk(
        id="chunk-id",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(tool_calls=[tool_call_delta2]))],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )

    async def fake_stream() -> AsyncIterator[ChatCompletionChunk]:
        for c in (chunk1, chunk2):
            yield c

    async def patched_fetch_response(self, *args, **kwargs):
        resp = Response(
            id="resp-id",
            created_at=0,
            model="fake-model",
            object="response",
            output=[],
            tool_choice="none",
            tools=[],
            parallel_tool_calls=False,
        )
        return resp, fake_stream()

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    output_events = []
    async for event in model.stream_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    ):
        output_events.append(event)
    # Sequence should be: response.created, then after loop we expect function call-related events:
    # one response.output_item.added for function call, a response.function_call_arguments.delta,
    # a response.output_item.done, and finally response.completed.
    assert output_events[0].type == "response.created"
    # The next three events are about the tool call.
    assert output_events[1].type == "response.output_item.added"
    # The added item should be a ResponseFunctionToolCall.
    added_fn = output_events[1].item
    assert isinstance(added_fn, ResponseFunctionToolCall)
    assert added_fn.name == "my_func"  # Name should be concatenation of both chunks.
    assert added_fn.arguments == "arg1arg2"
    assert output_events[2].type == "response.function_call_arguments.delta"
    assert output_events[2].delta == "arg1arg2"
    assert output_events[3].type == "response.output_item.done"
    assert output_events[4].type == "response.completed"
    assert output_events[2].delta == "arg1arg2"
    assert output_events[3].type == "response.output_item.done"
    assert output_events[4].type == "response.completed"
    assert added_fn.name == "my_func"  # Name should be concatenation of both chunks.
    assert added_fn.arguments == "arg1arg2"
    assert output_events[2].type == "response.function_call_arguments.delta"
    assert output_events[2].delta == "arg1arg2"
    assert output_events[3].type == "response.output_item.done"
    assert output_events[4].type == "response.completed"


<file_info>
path: examples/customer_service/main.py
name: main.py
</file_info>
from __future__ import annotations as _annotations

import asyncio
import random
import uuid

from pydantic import BaseModel

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

### CONTEXT


class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### TOOLS


@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    if "bag" in question or "baggage" in question:
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif "seats" in question or "plane" in question:
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif "wifi" in question:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


### HOOKS


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


### AGENTS

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


### RUN


async def main():
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    # Normally, each input from the user would be an API request to your app, and you can wrap the request in a trace()
    # Here, we'll just use a random UUID for the conversation ID
    conversation_id = uuid.uuid4().hex[:16]

    while True:
        user_input = input("Enter your message: ")
        with trace("Customer service", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()
            current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/agent_patterns/routing.py
name: routing.py
</file_info>
import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace

"""
This example shows the handoffs/routing pattern. The triage agent receives the first message, and
then hands off to the appropriate agent based on the language of the request. Responses are
streamed to the user.
"""

french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish",
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    msg = input("Hi! We speak French, Spanish and English. How can I help? ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/spans.py
name: spans.py
</file_info>
from __future__ import annotations

import abc
import contextvars
from typing import Any, Generic, TypeVar

from typing_extensions import TypedDict

from ..logger import logger
from . import util
from .processor_interface import TracingProcessor
from .scope import Scope
from .span_data import SpanData

TSpanData = TypeVar("TSpanData", bound=SpanData)


class SpanError(TypedDict):
    message: str
    data: dict[str, Any] | None


class Span(abc.ABC, Generic[TSpanData]):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def span_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData:
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """
        Start the span.

        Args:
            mark_as_current: If true, the span will be marked as the current span.
        """
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        """
        Finish the span.

        Args:
            reset_current: If true, the span will be reset as the current span.
        """
        pass

    @abc.abstractmethod
    def __enter__(self) -> Span[TSpanData]:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abc.abstractmethod
    def parent_id(self) -> str | None:
        pass

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None:
        pass

    @property
    @abc.abstractmethod
    def error(self) -> SpanError | None:
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        pass

    @property
    @abc.abstractmethod
    def started_at(self) -> str | None:
        pass

    @property
    @abc.abstractmethod
    def ended_at(self) -> str | None:
        pass


class NoOpSpan(Span[TSpanData]):
    __slots__ = ("_span_data", "_prev_span_token")

    def __init__(self, span_data: TSpanData):
        self._span_data = span_data
        self._prev_span_token: contextvars.Token[Span[TSpanData] | None] | None = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def span_id(self) -> str:
        return "no-op"

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> str | None:
        return None

    def start(self, mark_as_current: bool = False):
        if mark_as_current:
            self._prev_span_token = Scope.set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_span_token is not None:
            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Span[TSpanData]:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_current = True
        if exc_type is GeneratorExit:
            logger.debug("GeneratorExit, skipping span reset")
            reset_current = False

        self.finish(reset_current=reset_current)

    def set_error(self, error: SpanError) -> None:
        pass

    @property
    def error(self) -> SpanError | None:
        return None

    def export(self) -> dict[str, Any] | None:
        return None

    @property
    def started_at(self) -> str | None:
        return None

    @property
    def ended_at(self) -> str | None:
        return None


class SpanImpl(Span[TSpanData]):
    __slots__ = (
        "_trace_id",
        "_span_id",
        "_parent_id",
        "_started_at",
        "_ended_at",
        "_error",
        "_prev_span_token",
        "_processor",
        "_span_data",
    )

    def __init__(
        self,
        trace_id: str,
        span_id: str | None,
        parent_id: str | None,
        processor: TracingProcessor,
        span_data: TSpanData,
    ):
        self._trace_id = trace_id
        self._span_id = span_id or util.gen_span_id()
        self._parent_id = parent_id
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._processor = processor
        self._error: SpanError | None = None
        self._prev_span_token: contextvars.Token[Span[TSpanData] | None] | None = None
        self._span_data = span_data

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    def start(self, mark_as_current: bool = False):
        if self.started_at is not None:
            logger.warning("Span already started")
            return

        self._started_at = util.time_iso()
        self._processor.on_span_start(self)
        if mark_as_current:
            self._prev_span_token = Scope.set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        if self.ended_at is not None:
            logger.warning("Span already finished")
            return

        self._ended_at = util.time_iso()
        self._processor.on_span_end(self)
        if reset_current and self._prev_span_token is not None:
            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Span[TSpanData]:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_current = True
        if exc_type is GeneratorExit:
            logger.debug("GeneratorExit, skipping span reset")
            reset_current = False

        self.finish(reset_current=reset_current)

    def set_error(self, error: SpanError) -> None:
        self._error = error

    @property
    def error(self) -> SpanError | None:
        return self._error

    @property
    def started_at(self) -> str | None:
        return self._started_at

    @property
    def ended_at(self) -> str | None:
        return self._ended_at

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace.span",
            "id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self._parent_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "span_data": self.span_data.export(),
            "error": self._error,
        }


<file_info>
path: src/agents/extensions/handoff_filters.py
name: handoff_filters.py
</file_info>
from __future__ import annotations

from ..handoffs import HandoffInputData
from ..items import (
    HandoffCallItem,
    HandoffOutputItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)

"""Contains common handoff input filters, for convenience. """


def remove_all_tools(handoff_input_data: HandoffInputData) -> HandoffInputData:
    """Filters out all tool items: file search, web search and function calls+output."""

    history = handoff_input_data.input_history
    new_items = handoff_input_data.new_items

    filtered_history = (
        _remove_tool_types_from_input(history) if isinstance(history, tuple) else history
    )
    filtered_pre_handoff_items = _remove_tools_from_items(handoff_input_data.pre_handoff_items)
    filtered_new_items = _remove_tools_from_items(new_items)

    return HandoffInputData(
        input_history=filtered_history,
        pre_handoff_items=filtered_pre_handoff_items,
        new_items=filtered_new_items,
    )


def _remove_tools_from_items(items: tuple[RunItem, ...]) -> tuple[RunItem, ...]:
    filtered_items = []
    for item in items:
        if (
            isinstance(item, HandoffCallItem)
            or isinstance(item, HandoffOutputItem)
            or isinstance(item, ToolCallItem)
            or isinstance(item, ToolCallOutputItem)
        ):
            continue
        filtered_items.append(item)
    return tuple(filtered_items)


def _remove_tool_types_from_input(
    items: tuple[TResponseInputItem, ...],
) -> tuple[TResponseInputItem, ...]:
    tool_types = [
        "function_call",
        "function_call_output",
        "computer_call",
        "computer_call_output",
        "file_search_call",
        "web_search_call",
    ]

    filtered_items: list[TResponseInputItem] = []
    for item in items:
        itype = item.get("type")
        if itype in tool_types:
            continue
        filtered_items.append(item)
    return tuple(filtered_items)


<file_info>
path: src/agents/voice/pipeline.py
name: pipeline.py
</file_info>
from __future__ import annotations

import asyncio

from .._run_impl import TraceCtxManager
from ..exceptions import UserError
from ..logger import logger
from .input import AudioInput, StreamedAudioInput
from .model import STTModel, TTSModel
from .pipeline_config import VoicePipelineConfig
from .result import StreamedAudioResult
from .workflow import VoiceWorkflowBase


class VoicePipeline:
    """An opinionated voice agent pipeline. It works in three steps:
    1. Transcribe audio input into text.
    2. Run the provided `workflow`, which produces a sequence of text responses.
    3. Convert the text responses into streaming audio output.
    """

    def __init__(
        self,
        *,
        workflow: VoiceWorkflowBase,
        stt_model: STTModel | str | None = None,
        tts_model: TTSModel | str | None = None,
        config: VoicePipelineConfig | None = None,
    ):
        """Create a new voice pipeline.

        Args:
            workflow: The workflow to run. See `VoiceWorkflowBase`.
            stt_model: The speech-to-text model to use. If not provided, a default OpenAI
                model will be used.
            tts_model: The text-to-speech model to use. If not provided, a default OpenAI
                model will be used.
            config: The pipeline configuration. If not provided, a default configuration will be
                used.
        """
        self.workflow = workflow
        self.stt_model = stt_model if isinstance(stt_model, STTModel) else None
        self.tts_model = tts_model if isinstance(tts_model, TTSModel) else None
        self._stt_model_name = stt_model if isinstance(stt_model, str) else None
        self._tts_model_name = tts_model if isinstance(tts_model, str) else None
        self.config = config or VoicePipelineConfig()

    async def run(self, audio_input: AudioInput | StreamedAudioInput) -> StreamedAudioResult:
        """Run the voice pipeline.

        Args:
            audio_input: The audio input to process. This can either be an `AudioInput` instance,
                which is a single static buffer, or a `StreamedAudioInput` instance, which is a
                stream of audio data that you can append to.

        Returns:
            A `StreamedAudioResult` instance. You can use this object to stream audio events and
            play them out.
        """
        if isinstance(audio_input, AudioInput):
            return await self._run_single_turn(audio_input)
        elif isinstance(audio_input, StreamedAudioInput):
            return await self._run_multi_turn(audio_input)
        else:
            raise UserError(f"Unsupported audio input type: {type(audio_input)}")

    def _get_tts_model(self) -> TTSModel:
        if not self.tts_model:
            self.tts_model = self.config.model_provider.get_tts_model(self._tts_model_name)
        return self.tts_model

    def _get_stt_model(self) -> STTModel:
        if not self.stt_model:
            self.stt_model = self.config.model_provider.get_stt_model(self._stt_model_name)
        return self.stt_model

    async def _process_audio_input(self, audio_input: AudioInput) -> str:
        model = self._get_stt_model()
        return await model.transcribe(
            audio_input,
            self.config.stt_settings,
            self.config.trace_include_sensitive_data,
            self.config.trace_include_sensitive_audio_data,
        )

    async def _run_single_turn(self, audio_input: AudioInput) -> StreamedAudioResult:
        # Since this is single turn, we can use the TraceCtxManager to manage starting/ending the
        # trace
        with TraceCtxManager(
            workflow_name=self.config.workflow_name or "Voice Agent",
            trace_id=None,  # Automatically generated
            group_id=self.config.group_id,
            metadata=self.config.trace_metadata,
            disabled=self.config.tracing_disabled,
        ):
            input_text = await self._process_audio_input(audio_input)

            output = StreamedAudioResult(
                self._get_tts_model(), self.config.tts_settings, self.config
            )

            async def stream_events():
                try:
                    async for text_event in self.workflow.run(input_text):
                        await output._add_text(text_event)
                    await output._turn_done()
                    await output._done()
                except Exception as e:
                    logger.error(f"Error processing single turn: {e}")
                    await output._add_error(e)
                    raise e

            output._set_task(asyncio.create_task(stream_events()))
            return output

    async def _run_multi_turn(self, audio_input: StreamedAudioInput) -> StreamedAudioResult:
        with TraceCtxManager(
            workflow_name=self.config.workflow_name or "Voice Agent",
            trace_id=None,
            group_id=self.config.group_id,
            metadata=self.config.trace_metadata,
            disabled=self.config.tracing_disabled,
        ):
            output = StreamedAudioResult(
                self._get_tts_model(), self.config.tts_settings, self.config
            )

            transcription_session = await self._get_stt_model().create_session(
                audio_input,
                self.config.stt_settings,
                self.config.trace_include_sensitive_data,
                self.config.trace_include_sensitive_audio_data,
            )

            async def process_turns():
                try:
                    async for input_text in transcription_session.transcribe_turns():
                        result = self.workflow.run(input_text)
                        async for text_event in result:
                            await output._add_text(text_event)
                        await output._turn_done()
                except Exception as e:
                    logger.error(f"Error processing turns: {e}")
                    await output._add_error(e)
                    raise e
                finally:
                    await transcription_session.close()
                    await output._done()

            output._set_task(asyncio.create_task(process_turns()))
            return output


<file_info>
path: tests/tracing/test_processor_api_key.py
name: test_processor_api_key.py
</file_info>
import pytest

from agents.tracing.processors import BackendSpanExporter


@pytest.mark.asyncio
async def test_processor_api_key(monkeypatch):
    # If the API key is not set, it should be None
    monkeypatch.delenv("OPENAI_API_KEY", None)
    processor = BackendSpanExporter()
    assert processor.api_key is None

    # If we set it afterwards, it should be the new value
    processor.set_api_key("test_api_key")
    assert processor.api_key == "test_api_key"


@pytest.mark.asyncio
async def test_processor_api_key_from_env(monkeypatch):
    # If the API key is not set at creation time but set before access time, it should be the new
    # value
    monkeypatch.delenv("OPENAI_API_KEY", None)
    processor = BackendSpanExporter()

    # If we set it afterwards, it should be the new value
    monkeypatch.setenv("OPENAI_API_KEY", "foo_bar_123")
    assert processor.api_key == "foo_bar_123"


<file_info>
path: tests/__init__.py
name: __init__.py
</file_info>


<file_info>
path: tests/test_run_step_processing.py
name: test_run_step_processing.py
</file_info>
from __future__ import annotations

import pytest
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_computer_tool_call import ActionClick
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from pydantic import BaseModel

from agents import (
    Agent,
    Computer,
    ComputerTool,
    Handoff,
    ModelBehaviorError,
    ModelResponse,
    ReasoningItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    Usage,
)
from agents._run_impl import RunImpl

from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)


def test_empty_response():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[],
        usage=Usage(),
        referenceable_id=None,
    )

    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert not result.handoffs
    assert not result.functions


def test_no_tool_calls():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[get_text_message("Hello, world!")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert not result.handoffs
    assert not result.functions


def test_single_tool_call():
    agent = Agent(name="test", tools=[get_function_tool(name="test")])
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("test", ""),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert not result.handoffs
    assert result.functions and len(result.functions) == 1

    func = result.functions[0]
    assert func.tool_call.name == "test"
    assert func.tool_call.arguments == ""


def test_missing_tool_call_raises_error():
    agent = Agent(name="test", tools=[get_function_tool(name="test")])
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("missing", ""),
        ],
        usage=Usage(),
        referenceable_id=None,
    )

    with pytest.raises(ModelBehaviorError):
        RunImpl.process_model_response(
            agent=agent, response=response, output_schema=None, handoffs=[]
        )


def test_multiple_tool_calls():
    agent = Agent(
        name="test",
        tools=[
            get_function_tool(name="test_1"),
            get_function_tool(name="test_2"),
            get_function_tool(name="test_3"),
        ],
    )
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("test_1", "abc"),
            get_function_tool_call("test_2", "xyz"),
        ],
        usage=Usage(),
        referenceable_id=None,
    )

    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert not result.handoffs
    assert result.functions and len(result.functions) == 2

    func_1 = result.functions[0]
    assert func_1.tool_call.name == "test_1"
    assert func_1.tool_call.arguments == "abc"

    func_2 = result.functions[1]
    assert func_2.tool_call.name == "test_2"
    assert func_2.tool_call.arguments == "xyz"


@pytest.mark.asyncio
async def test_handoffs_parsed_correctly():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])
    response = ModelResponse(
        output=[get_text_message("Hello, world!")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent_3, response=response, output_schema=None, handoffs=[]
    )
    assert not result.handoffs, "Shouldn't have a handoff here"

    response = ModelResponse(
        output=[get_text_message("Hello, world!"), get_handoff_tool_call(agent_1)],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent_3,
        response=response,
        output_schema=None,
        handoffs=Runner._get_handoffs(agent_3),
    )
    assert len(result.handoffs) == 1, "Should have a handoff here"
    handoff = result.handoffs[0]
    assert handoff.handoff.tool_name == Handoff.default_tool_name(agent_1)
    assert handoff.handoff.tool_description == Handoff.default_tool_description(agent_1)
    assert handoff.handoff.agent_name == agent_1.name

    handoff_agent = await handoff.handoff.on_invoke_handoff(
        RunContextWrapper(None), handoff.tool_call.arguments
    )
    assert handoff_agent == agent_1


@pytest.mark.asyncio
async def test_missing_handoff_fails():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1])
    response = ModelResponse(
        output=[get_text_message("Hello, world!"), get_handoff_tool_call(agent_2)],
        usage=Usage(),
        referenceable_id=None,
    )
    with pytest.raises(ModelBehaviorError):
        RunImpl.process_model_response(
            agent=agent_3,
            response=response,
            output_schema=None,
            handoffs=Runner._get_handoffs(agent_3),
        )


def test_multiple_handoffs_doesnt_error():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_handoff_tool_call(agent_1),
            get_handoff_tool_call(agent_2),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent_3,
        response=response,
        output_schema=None,
        handoffs=Runner._get_handoffs(agent_3),
    )
    assert len(result.handoffs) == 2, "Should have multiple handoffs here"


class Foo(BaseModel):
    bar: str


def test_final_output_parsed_correctly():
    agent = Agent(name="test", output_type=Foo)
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_final_output_message(Foo(bar="123").model_dump_json()),
        ],
        usage=Usage(),
        referenceable_id=None,
    )

    RunImpl.process_model_response(
        agent=agent,
        response=response,
        output_schema=Runner._get_output_schema(agent),
        handoffs=[],
    )


def test_file_search_tool_call_parsed_correctly():
    # Ensure that a ResponseFileSearchToolCall output is parsed into a ToolCallItem and that no tool
    # runs are scheduled.

    agent = Agent(name="test")
    file_search_call = ResponseFileSearchToolCall(
        id="fs1",
        queries=["query"],
        status="completed",
        type="file_search_call",
    )
    response = ModelResponse(
        output=[get_text_message("hello"), file_search_call],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    # The final item should be a ToolCallItem for the file search call
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is file_search_call
        for item in result.new_items
    )
    assert not result.functions
    assert not result.handoffs


def test_function_web_search_tool_call_parsed_correctly():
    agent = Agent(name="test")
    web_search_call = ResponseFunctionWebSearch(id="w1", status="completed", type="web_search_call")
    response = ModelResponse(
        output=[get_text_message("hello"), web_search_call],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is web_search_call
        for item in result.new_items
    )
    assert not result.functions
    assert not result.handoffs


def test_reasoning_item_parsed_correctly():
    # Verify that a Reasoning output item is converted into a ReasoningItem.

    reasoning = ResponseReasoningItem(
        id="r1", type="reasoning", summary=[Summary(text="why", type="summary_text")]
    )
    response = ModelResponse(
        output=[reasoning],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=Agent(name="test"), response=response, output_schema=None, handoffs=[]
    )
    assert any(
        isinstance(item, ReasoningItem) and item.raw_item is reasoning for item in result.new_items
    )


class DummyComputer(Computer):
    """Minimal computer implementation for testing."""

    @property
    def environment(self):
        return "mac"  # pragma: no cover

    @property
    def dimensions(self):
        return (0, 0)  # pragma: no cover

    def screenshot(self) -> str:
        return ""  # pragma: no cover

    def click(self, x: int, y: int, button: str) -> None:
        return None  # pragma: no cover

    def double_click(self, x: int, y: int) -> None:
        return None  # pragma: no cover

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        return None  # pragma: no cover

    def type(self, text: str) -> None:
        return None  # pragma: no cover

    def wait(self) -> None:
        return None  # pragma: no cover

    def move(self, x: int, y: int) -> None:
        return None  # pragma: no cover

    def keypress(self, keys: list[str]) -> None:
        return None  # pragma: no cover

    def drag(self, path: list[tuple[int, int]]) -> None:
        return None  # pragma: no cover


def test_computer_tool_call_without_computer_tool_raises_error():
    # If the agent has no ComputerTool in its tools, process_model_response should raise a
    # ModelBehaviorError when encountering a ResponseComputerToolCall.
    computer_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=ActionClick(type="click", x=1, y=2, button="left"),
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    response = ModelResponse(
        output=[computer_call],
        usage=Usage(),
        referenceable_id=None,
    )
    with pytest.raises(ModelBehaviorError):
        RunImpl.process_model_response(
            agent=Agent(name="test"), response=response, output_schema=None, handoffs=[]
        )


def test_computer_tool_call_with_computer_tool_parsed_correctly():
    # If the agent contains a ComputerTool, ensure that a ResponseComputerToolCall is parsed into a
    # ToolCallItem and scheduled to run in computer_actions.
    dummy_computer = DummyComputer()
    agent = Agent(name="test", tools=[ComputerTool(computer=dummy_computer)])
    computer_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=ActionClick(type="click", x=1, y=2, button="left"),
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    response = ModelResponse(
        output=[computer_call],
        usage=Usage(),
        referenceable_id=None,
    )
    result = RunImpl.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[]
    )
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is computer_call
        for item in result.new_items
    )
    assert result.computer_actions and result.computer_actions[0].tool_call == computer_call


def test_tool_and_handoff_parsed_correctly():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(
        name="test_3", tools=[get_function_tool(name="test")], handoffs=[agent_1, agent_2]
    )
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("test", "abc"),
            get_handoff_tool_call(agent_1),
        ],
        usage=Usage(),
        referenceable_id=None,
    )

    result = RunImpl.process_model_response(
        agent=agent_3,
        response=response,
        output_schema=None,
        handoffs=Runner._get_handoffs(agent_3),
    )
    assert result.functions and len(result.functions) == 1
    assert len(result.handoffs) == 1, "Should have a handoff here"
    handoff = result.handoffs[0]
    assert handoff.handoff.tool_name == Handoff.default_tool_name(agent_1)
    assert handoff.handoff.tool_description == Handoff.default_tool_description(agent_1)
    assert handoff.handoff.agent_name == agent_1.name


<file_info>
path: tests/test_config.py
name: test_config.py
</file_info>
import os

import openai
import pytest

from agents import set_default_openai_api, set_default_openai_client, set_default_openai_key
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from agents.models.openai_responses import OpenAIResponsesModel


def test_cc_no_default_key_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(openai.OpenAIError):
        OpenAIProvider(use_responses=False).get_model("gpt-4")


def test_cc_set_default_openai_key():
    set_default_openai_key("test_key")
    chat_model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    assert chat_model._client.api_key == "test_key"  # type: ignore


def test_cc_set_default_openai_client():
    client = openai.AsyncOpenAI(api_key="test_key")
    set_default_openai_client(client)
    chat_model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    assert chat_model._client.api_key == "test_key"  # type: ignore


def test_resp_no_default_key_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert os.getenv("OPENAI_API_KEY") is None
    with pytest.raises(openai.OpenAIError):
        OpenAIProvider(use_responses=True).get_model("gpt-4")


def test_resp_set_default_openai_key():
    set_default_openai_key("test_key")
    resp_model = OpenAIProvider(use_responses=True).get_model("gpt-4")
    assert resp_model._client.api_key == "test_key"  # type: ignore


def test_resp_set_default_openai_client():
    client = openai.AsyncOpenAI(api_key="test_key")
    set_default_openai_client(client)
    resp_model = OpenAIProvider(use_responses=True).get_model("gpt-4")
    assert resp_model._client.api_key == "test_key"  # type: ignore


def test_set_default_openai_api():
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIResponsesModel), (
        "Default should be responses"
    )

    set_default_openai_api("chat_completions")
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIChatCompletionsModel), (
        "Should be chat completions model"
    )

    set_default_openai_api("responses")
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIResponsesModel), (
        "Should be responses model"
    )


<file_info>
path: tests/test_global_hooks.py
name: test_global_hooks.py
</file_info>
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pytest
from typing_extensions import TypedDict

from agents import Agent, RunContextWrapper, RunHooks, Runner, TContext, Tool

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)


class RunHooksForTests(RunHooks):
    def __init__(self):
        self.events: dict[str, int] = defaultdict(int)

    def reset(self):
        self.events.clear()

    async def on_agent_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        self.events["on_agent_start"] += 1

    async def on_agent_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        self.events["on_agent_end"] += 1

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        self.events["on_handoff"] += 1

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
    ) -> None:
        self.events["on_tool_start"] += 1

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: str,
    ) -> None:
        self.events["on_tool_end"] += 1


@pytest.mark.asyncio
async def test_non_streamed_agent_hooks():
    hooks = RunHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_text_message("user_message")])
    output = await Runner.run(agent_3, input="user_message", hooks=hooks)
    assert hooks.events == {"on_agent_start": 1, "on_agent_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message", hooks=hooks)
    assert hooks.events == {
        # We only invoke on_agent_start when we begin executing a new agent.
        # Although agent_3 runs two turns internally before handing off,
        # that's one logical agent segment, so on_agent_start fires once.
        # Then we hand off to agent_1, so on_agent_start fires for that agent.
        "on_agent_start": 2,
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: text message
            [get_text_message("done")],
        ]
    )
    await Runner.run(agent_3, input="user_message", hooks=hooks)

    assert hooks.events == {
        # agent_3 starts (fires on_agent_start), runs two turns and hands off.
        # agent_1 starts (fires on_agent_start), then hands back to agent_3.
        # agent_3 starts again (fires on_agent_start) to complete execution.
        "on_agent_start": 3,
        "on_tool_start": 2,  # 2 tool calls
        "on_tool_end": 2,  # 2 tool calls
        "on_handoff": 2,  # 2 handoffs
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


@pytest.mark.asyncio
async def test_streamed_agent_hooks():
    hooks = RunHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_text_message("user_message")])
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass
    assert hooks.events == {"on_agent_start": 1, "on_agent_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass
    assert hooks.events == {
        # As in the non-streamed case above, two logical agent segments:
        # starting agent_3, then handoff to agent_1.
        "on_agent_start": 2,
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: text message
            [get_text_message("done")],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass

    assert hooks.events == {
        # Same three logical agent segments as in the non-streamed case,
        # so on_agent_start fires three times.
        "on_agent_start": 3,
        "on_tool_start": 2,  # 2 tool calls
        "on_tool_end": 2,  # 2 tool calls
        "on_handoff": 2,  # 2 handoffs
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


class Foo(TypedDict):
    a: str


@pytest.mark.asyncio
async def test_structured_output_non_streamed_agent_hooks():
    hooks = RunHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        output_type=Foo,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_final_output_message(json.dumps({"a": "b"}))])
    output = await Runner.run(agent_3, input="user_message", hooks=hooks)
    assert hooks.events == {"on_agent_start": 1, "on_agent_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: end message (for agent 1)
            [get_text_message("done")],
        ]
    )
    output = await Runner.run(agent_3, input="user_message", hooks=hooks)

    assert hooks.events == {
        # As with unstructured output, we expect on_agent_start once for
        # agent_3 and once for agent_1.
        "on_agent_start": 2,
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: end message (for agent 3)
            [get_final_output_message(json.dumps({"a": "b"}))],
        ]
    )
    await Runner.run(agent_3, input="user_message", hooks=hooks)

    assert hooks.events == {
        # We still expect three logical agent segments, as before.
        "on_agent_start": 3,
        "on_tool_start": 2,  # 2 tool calls
        "on_tool_end": 2,  # 2 tool calls
        "on_handoff": 2,  # 2 handoffs
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


@pytest.mark.asyncio
async def test_structured_output_streamed_agent_hooks():
    hooks = RunHooksForTests()
    model = FakeModel()
    agent_1 = Agent(name="test_1", model=model)
    agent_2 = Agent(name="test_2", model=model)
    agent_3 = Agent(
        name="test_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
        output_type=Foo,
    )

    agent_1.handoffs.append(agent_3)

    model.set_next_output([get_final_output_message(json.dumps({"a": "b"}))])
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass
    assert hooks.events == {"on_agent_start": 1, "on_agent_end": 1}, f"{output}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: end message (for agent 1)
            [get_text_message("done")],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass

    assert hooks.events == {
        # Two agent segments: agent_3 and then agent_1.
        "on_agent_start": 2,
        "on_tool_start": 1,  # Only one tool call
        "on_tool_end": 1,  # Only one tool call
        "on_handoff": 1,  # Only one handoff
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message, another tool call, and a handoff
            [
                get_text_message("a_message"),
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_handoff_tool_call(agent_1),
            ],
            # Third turn: a message and a handoff back to the orig agent
            [get_text_message("a_message"), get_handoff_tool_call(agent_3)],
            # Fourth turn: end message (for agent 3)
            [get_final_output_message(json.dumps({"a": "b"}))],
        ]
    )
    output = Runner.run_streamed(agent_3, input="user_message", hooks=hooks)
    async for _ in output.stream_events():
        pass

    assert hooks.events == {
        # Three agent segments: agent_3, agent_1, agent_3 again.
        "on_agent_start": 3,
        "on_tool_start": 2,  # 2 tool calls
        "on_tool_end": 2,  # 2 tool calls
        "on_handoff": 2,  # 2 handoffs
        "on_agent_end": 1,  # Should always have one end
    }, f"got unexpected event count: {hooks.events}"
    hooks.reset()


<file_info>
path: examples/financial_research_agent/__init__.py
name: __init__.py
</file_info>


<file_info>
path: examples/financial_research_agent/agents/risk_agent.py
name: risk_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

# A sub‑agent specializing in identifying risk factors or concerns.
RISK_PROMPT = (
    "You are a risk analyst looking for potential red flags in a company's outlook. "
    "Given background research, produce a short analysis of risks such as competitive threats, "
    "regulatory issues, supply chain problems, or slowing growth. Keep it under 2 paragraphs."
)


class AnalysisSummary(BaseModel):
    summary: str
    """Short text summary for this aspect of the analysis."""


risk_agent = Agent(
    name="RiskAnalystAgent",
    instructions=RISK_PROMPT,
    output_type=AnalysisSummary,
)


<file_info>
path: examples/__init__.py
name: __init__.py
</file_info>
# Make the examples directory into a package to avoid top-level module name collisions.
# This is needed so that mypy treats files like examples/customer_service/main.py and
# examples/researcher_app/main.py as distinct modules rather than both named "main".


<file_info>
path: examples/research_bot/printer.py
name: printer.py
</file_info>
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner


class Printer:
    def __init__(self, console: Console):
        self.live = Live(console=console)
        self.items: dict[str, tuple[str, bool]] = {}
        self.hide_done_ids: set[str] = set()
        self.live.start()

    def end(self) -> None:
        self.live.stop()

    def hide_done_checkmark(self, item_id: str) -> None:
        self.hide_done_ids.add(item_id)

    def update_item(
        self, item_id: str, content: str, is_done: bool = False, hide_checkmark: bool = False
    ) -> None:
        self.items[item_id] = (content, is_done)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self.flush()

    def mark_item_done(self, item_id: str) -> None:
        self.items[item_id] = (self.items[item_id][0], True)
        self.flush()

    def flush(self) -> None:
        renderables: list[Any] = []
        for item_id, (content, is_done) in self.items.items():
            if is_done:
                prefix = "✅ " if item_id not in self.hide_done_ids else ""
                renderables.append(prefix + content)
            else:
                renderables.append(Spinner("dots", text=content))
        self.live.update(Group(*renderables))


<file_info>
path: examples/voice/static/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/tracing/logger.py
name: logger.py
</file_info>
import logging

logger = logging.getLogger("openai.agents.tracing")


<file_info>
path: src/agents/util/_error_tracing.py
name: _error_tracing.py
</file_info>
from typing import Any

from ..logger import logger
from ..tracing import Span, SpanError, get_current_span


def attach_error_to_span(span: Span[Any], error: SpanError) -> None:
    span.set_error(error)


def attach_error_to_current_span(error: SpanError) -> None:
    span = get_current_span()
    if span:
        attach_error_to_span(span, error)
    else:
        logger.warning(f"No span to add error {error} to")


<file_info>
path: src/agents/util/_pretty_print.py
name: _pretty_print.py
</file_info>
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..result import RunResult, RunResultBase, RunResultStreaming


def _indent(text: str, indent_level: int) -> str:
    indent_string = "  " * indent_level
    return "\n".join(f"{indent_string}{line}" for line in text.splitlines())


def _final_output_str(result: "RunResultBase") -> str:
    if result.final_output is None:
        return "None"
    elif isinstance(result.final_output, str):
        return result.final_output
    elif isinstance(result.final_output, BaseModel):
        return result.final_output.model_dump_json(indent=2)
    else:
        return str(result.final_output)


def pretty_print_result(result: "RunResult") -> str:
    output = "RunResult:"
    output += f'\n- Last agent: Agent(name="{result.last_agent.name}", ...)'
    output += (
        f"\n- Final output ({type(result.final_output).__name__}):\n"
        f"{_indent(_final_output_str(result), 2)}"
    )
    output += f"\n- {len(result.new_items)} new item(s)"
    output += f"\n- {len(result.raw_responses)} raw response(s)"
    output += f"\n- {len(result.input_guardrail_results)} input guardrail result(s)"
    output += f"\n- {len(result.output_guardrail_results)} output guardrail result(s)"
    output += "\n(See `RunResult` for more details)"

    return output


def pretty_print_run_result_streaming(result: "RunResultStreaming") -> str:
    output = "RunResultStreaming:"
    output += f'\n- Current agent: Agent(name="{result.current_agent.name}", ...)'
    output += f"\n- Current turn: {result.current_turn}"
    output += f"\n- Max turns: {result.max_turns}"
    output += f"\n- Is complete: {result.is_complete}"
    output += (
        f"\n- Final output ({type(result.final_output).__name__}):\n"
        f"{_indent(_final_output_str(result), 2)}"
    )
    output += f"\n- {len(result.new_items)} new item(s)"
    output += f"\n- {len(result.raw_responses)} raw response(s)"
    output += f"\n- {len(result.input_guardrail_results)} input guardrail result(s)"
    output += f"\n- {len(result.output_guardrail_results)} output guardrail result(s)"
    output += "\n(See `RunResultStreaming` for more details)"
    return output


<file_info>
path: src/agents/models/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/models/openai_provider.py
name: openai_provider.py
</file_info>
from __future__ import annotations

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from . import _openai_shared
from .interface import Model, ModelProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel
from .openai_responses import OpenAIResponsesModel

DEFAULT_MODEL: str = "gpt-4o"


_http_client: httpx.AsyncClient | None = None


# If we create a new httpx client for each request, that would mean no sharing of connection pools,
# which would mean worse latency and resource usage. So, we share the client across requests.
def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = DefaultAsyncHttpxClient()
    return _http_client


class OpenAIProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
        use_responses: bool | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            api_key: The API key to use for the OpenAI client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
                default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            organization: The organization to use for the OpenAI client.
            project: The project to use for the OpenAI client.
            use_responses: Whether to use the OpenAI responses API.
        """
        if openai_client is not None:
            assert api_key is None and base_url is None, (
                "Don't provide api_key or base_url if you provide openai_client"
            )
            self._client: AsyncOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project

        if use_responses is not None:
            self._use_responses = use_responses
        else:
            self._use_responses = _openai_shared.get_use_responses_by_default()

    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
    # AsyncOpenAI() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=shared_http_client(),
            )

        return self._client

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = DEFAULT_MODEL

        client = self._get_client()

        return (
            OpenAIResponsesModel(model=model_name, openai_client=client)
            if self._use_responses
            else OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        )


<file_info>
path: src/agents/voice/__init__.py
name: __init__.py
</file_info>
from .events import VoiceStreamEvent, VoiceStreamEventAudio, VoiceStreamEventLifecycle
from .exceptions import STTWebsocketConnectionError
from .input import AudioInput, StreamedAudioInput
from .model import (
    StreamedTranscriptionSession,
    STTModel,
    STTModelSettings,
    TTSModel,
    TTSModelSettings,
    VoiceModelProvider,
)
from .models.openai_model_provider import OpenAIVoiceModelProvider
from .models.openai_stt import OpenAISTTModel, OpenAISTTTranscriptionSession
from .models.openai_tts import OpenAITTSModel
from .pipeline import VoicePipeline
from .pipeline_config import VoicePipelineConfig
from .result import StreamedAudioResult
from .utils import get_sentence_based_splitter
from .workflow import (
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoiceWorkflowBase,
    VoiceWorkflowHelper,
)

__all__ = [
    "AudioInput",
    "StreamedAudioInput",
    "STTModel",
    "STTModelSettings",
    "TTSModel",
    "TTSModelSettings",
    "VoiceModelProvider",
    "StreamedAudioResult",
    "SingleAgentVoiceWorkflow",
    "OpenAIVoiceModelProvider",
    "OpenAISTTModel",
    "OpenAITTSModel",
    "VoiceStreamEventAudio",
    "VoiceStreamEventLifecycle",
    "VoiceStreamEvent",
    "VoicePipeline",
    "VoicePipelineConfig",
    "get_sentence_based_splitter",
    "VoiceWorkflowHelper",
    "VoiceWorkflowBase",
    "SingleAgentWorkflowCallbacks",
    "StreamedTranscriptionSession",
    "OpenAISTTTranscriptionSession",
    "STTWebsocketConnectionError",
]


<file_info>
path: src/agents/voice/result.py
name: result.py
</file_info>
from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import UserError
from ..logger import logger
from ..tracing import Span, SpeechGroupSpanData, speech_group_span, speech_span
from ..tracing.util import time_iso
from .events import (
    VoiceStreamEvent,
    VoiceStreamEventAudio,
    VoiceStreamEventError,
    VoiceStreamEventLifecycle,
)
from .imports import np, npt
from .model import TTSModel, TTSModelSettings
from .pipeline_config import VoicePipelineConfig


def _audio_to_base64(audio_data: list[bytes]) -> str:
    joined_audio_data = b"".join(audio_data)
    return base64.b64encode(joined_audio_data).decode("utf-8")


class StreamedAudioResult:
    """The output of a `VoicePipeline`. Streams events and audio data as they're generated."""

    def __init__(
        self,
        tts_model: TTSModel,
        tts_settings: TTSModelSettings,
        voice_pipeline_config: VoicePipelineConfig,
    ):
        """Create a new `StreamedAudioResult` instance.

        Args:
            tts_model: The TTS model to use.
            tts_settings: The TTS settings to use.
            voice_pipeline_config: The voice pipeline config to use.
        """
        self.tts_model = tts_model
        self.tts_settings = tts_settings
        self.total_output_text = ""
        self.instructions = tts_settings.instructions
        self.text_generation_task: asyncio.Task[Any] | None = None

        self._voice_pipeline_config = voice_pipeline_config
        self._text_buffer = ""
        self._turn_text_buffer = ""
        self._queue: asyncio.Queue[VoiceStreamEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._ordered_tasks: list[
            asyncio.Queue[VoiceStreamEvent | None]
        ] = []  # New: list to hold local queues for each text segment
        self._dispatcher_task: asyncio.Task[Any] | None = (
            None  # Task to dispatch audio chunks in order
        )

        self._done_processing = False
        self._buffer_size = tts_settings.buffer_size
        self._started_processing_turn = False
        self._first_byte_received = False
        self._generation_start_time: str | None = None
        self._completed_session = False
        self._stored_exception: BaseException | None = None
        self._tracing_span: Span[SpeechGroupSpanData] | None = None

    async def _start_turn(self):
        if self._started_processing_turn:
            return

        self._tracing_span = speech_group_span()
        self._tracing_span.start()
        self._started_processing_turn = True
        self._first_byte_received = False
        self._generation_start_time = time_iso()
        await self._queue.put(VoiceStreamEventLifecycle(event="turn_started"))

    def _set_task(self, task: asyncio.Task[Any]):
        self.text_generation_task = task

    async def _add_error(self, error: Exception):
        await self._queue.put(VoiceStreamEventError(error))

    def _transform_audio_buffer(
        self, buffer: list[bytes], output_dtype: npt.DTypeLike
    ) -> npt.NDArray[np.int16 | np.float32]:
        np_array = np.frombuffer(b"".join(buffer), dtype=np.int16)

        if output_dtype == np.int16:
            return np_array
        elif output_dtype == np.float32:
            return (np_array.astype(np.float32) / 32767.0).reshape(-1, 1)
        else:
            raise UserError("Invalid output dtype")

    async def _stream_audio(
        self,
        text: str,
        local_queue: asyncio.Queue[VoiceStreamEvent | None],
        finish_turn: bool = False,
    ):
        with speech_span(
            model=self.tts_model.model_name,
            input=text if self._voice_pipeline_config.trace_include_sensitive_data else "",
            model_config={
                "voice": self.tts_settings.voice,
                "instructions": self.instructions,
                "speed": self.tts_settings.speed,
            },
            output_format="pcm",
            parent=self._tracing_span,
        ) as tts_span:
            try:
                first_byte_received = False
                buffer: list[bytes] = []
                full_audio_data: list[bytes] = []

                async for chunk in self.tts_model.run(text, self.tts_settings):
                    if not first_byte_received:
                        first_byte_received = True
                        tts_span.span_data.first_content_at = time_iso()

                    if chunk:
                        buffer.append(chunk)
                        full_audio_data.append(chunk)
                        if len(buffer) >= self._buffer_size:
                            audio_np = self._transform_audio_buffer(buffer, self.tts_settings.dtype)
                            if self.tts_settings.transform_data:
                                audio_np = self.tts_settings.transform_data(audio_np)
                            await local_queue.put(
                                VoiceStreamEventAudio(data=audio_np)
                            )  # Use local queue
                            buffer = []
                if buffer:
                    audio_np = self._transform_audio_buffer(buffer, self.tts_settings.dtype)
                    if self.tts_settings.transform_data:
                        audio_np = self.tts_settings.transform_data(audio_np)
                    await local_queue.put(VoiceStreamEventAudio(data=audio_np))  # Use local queue

                if self._voice_pipeline_config.trace_include_sensitive_audio_data:
                    tts_span.span_data.output = _audio_to_base64(full_audio_data)
                else:
                    tts_span.span_data.output = ""

                if finish_turn:
                    await local_queue.put(VoiceStreamEventLifecycle(event="turn_ended"))
                else:
                    await local_queue.put(None)  # Signal completion for this segment
            except Exception as e:
                tts_span.set_error(
                    {
                        "message": str(e),
                        "data": {
                            "text": text
                            if self._voice_pipeline_config.trace_include_sensitive_data
                            else "",
                        },
                    }
                )
                logger.error(f"Error streaming audio: {e}")

                # Signal completion for whole session because of error
                await local_queue.put(VoiceStreamEventLifecycle(event="session_ended"))
                raise e

    async def _add_text(self, text: str):
        await self._start_turn()

        self._text_buffer += text
        self.total_output_text += text
        self._turn_text_buffer += text

        combined_sentences, self._text_buffer = self.tts_settings.text_splitter(self._text_buffer)

        if len(combined_sentences) >= 20:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)
            self._tasks.append(
                asyncio.create_task(self._stream_audio(combined_sentences, local_queue))
            )
            if self._dispatcher_task is None:
                self._dispatcher_task = asyncio.create_task(self._dispatch_audio())

    async def _turn_done(self):
        if self._text_buffer:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)  # Append the local queue for the final segment
            self._tasks.append(
                asyncio.create_task(
                    self._stream_audio(self._text_buffer, local_queue, finish_turn=True)
                )
            )
            self._text_buffer = ""
        self._done_processing = True
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_audio())
        await asyncio.gather(*self._tasks)

    def _finish_turn(self):
        if self._tracing_span:
            if self._voice_pipeline_config.trace_include_sensitive_data:
                self._tracing_span.span_data.input = self._turn_text_buffer
            else:
                self._tracing_span.span_data.input = ""

            self._tracing_span.finish()
            self._tracing_span = None
        self._turn_text_buffer = ""
        self._started_processing_turn = False

    async def _done(self):
        self._completed_session = True
        await self._wait_for_completion()

    async def _dispatch_audio(self):
        # Dispatch audio chunks from each segment in the order they were added
        while True:
            if len(self._ordered_tasks) == 0:
                if self._completed_session:
                    break
                await asyncio.sleep(0)
                continue
            local_queue = self._ordered_tasks.pop(0)
            while True:
                chunk = await local_queue.get()
                if chunk is None:
                    break
                await self._queue.put(chunk)
                if isinstance(chunk, VoiceStreamEventLifecycle):
                    local_queue.task_done()
                    if chunk.event == "turn_ended":
                        self._finish_turn()
                        break
        await self._queue.put(VoiceStreamEventLifecycle(event="session_ended"))

    async def _wait_for_completion(self):
        tasks: list[asyncio.Task[Any]] = self._tasks
        if self._dispatcher_task is not None:
            tasks.append(self._dispatcher_task)
        await asyncio.gather(*tasks)

    def _cleanup_tasks(self):
        self._finish_turn()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()

        if self.text_generation_task and not self.text_generation_task.done():
            self.text_generation_task.cancel()

    def _check_errors(self):
        for task in self._tasks:
            if task.done():
                if task.exception():
                    self._stored_exception = task.exception()
                    break

    async def stream(self) -> AsyncIterator[VoiceStreamEvent]:
        """Stream the events and audio data as they're generated."""
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break
            if isinstance(event, VoiceStreamEventError):
                self._stored_exception = event.error
                logger.error(f"Error processing output: {event.error}")
                break
            if event is None:
                break
            yield event
            if event.type == "voice_stream_event_lifecycle" and event.event == "session_ended":
                break

        self._check_errors()
        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception


<file_info>
path: tests/conftest.py
name: conftest.py
</file_info>
from __future__ import annotations

import pytest

from agents.models import _openai_shared
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tracing import set_trace_processors
from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

from .testing_processor import SPAN_PROCESSOR_TESTING


# This fixture will run once before any tests are executed
@pytest.fixture(scope="session", autouse=True)
def setup_span_processor():
    set_trace_processors([SPAN_PROCESSOR_TESTING])


# This fixture will run before each test
@pytest.fixture(autouse=True)
def clear_span_processor():
    SPAN_PROCESSOR_TESTING.force_flush()
    SPAN_PROCESSOR_TESTING.shutdown()
    SPAN_PROCESSOR_TESTING.clear()


# This fixture will run before each test
@pytest.fixture(autouse=True)
def clear_openai_settings():
    _openai_shared._default_openai_key = None
    _openai_shared._default_openai_client = None
    _openai_shared._use_responses_by_default = True


# This fixture will run after all tests end
@pytest.fixture(autouse=True, scope="session")
def shutdown_trace_provider():
    yield
    GLOBAL_TRACE_PROVIDER.shutdown()


@pytest.fixture(autouse=True)
def disable_real_model_clients(monkeypatch, request):
    # If the test is marked to allow the method call, don't override it.
    if request.node.get_closest_marker("allow_call_model_methods"):
        return

    def failing_version(*args, **kwargs):
        pytest.fail("Real models should not be used in tests!")

    monkeypatch.setattr(OpenAIResponsesModel, "get_response", failing_version)
    monkeypatch.setattr(OpenAIResponsesModel, "stream_response", failing_version)
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", failing_version)
    monkeypatch.setattr(OpenAIChatCompletionsModel, "stream_response", failing_version)


<file_info>
path: tests/test_agent_runner.py
name: test_agent_runner.py
</file_info>
from __future__ import annotations

import json
from typing import Any

import pytest
from typing_extensions import TypedDict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    Handoff,
    HandoffInputData,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    ModelBehaviorError,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    UserError,
    handoff,
)
from agents.agent import ToolsToFinalOutputResult
from agents.tool import FunctionToolResult, function_tool

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_input_item,
    get_text_message,
)


@pytest.mark.asyncio
async def test_simple_first_run():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )
    model.set_next_output([get_text_message("first")])

    result = await Runner.run(agent, input="test")
    assert result.input == "test"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "first"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert result.raw_responses[0].output == [get_text_message("first")]
    assert result.last_agent == agent

    assert len(result.to_input_list()) == 2, "should have original input and generated item"

    model.set_next_output([get_text_message("second")])

    result = await Runner.run(
        agent, input=[get_text_input_item("message"), get_text_input_item("another_message")]
    )
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "second"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert len(result.to_input_list()) == 3, "should have original input and generated item"


@pytest.mark.asyncio
async def test_subsequent_runs():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )
    model.set_next_output([get_text_message("third")])

    result = await Runner.run(agent, input="test")
    assert result.input == "test"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert len(result.to_input_list()) == 2, "should have original input and generated item"

    model.set_next_output([get_text_message("fourth")])

    result = await Runner.run(agent, input=result.to_input_list())
    assert len(result.input) == 2, f"should have previous input but got {result.input}"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "fourth"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert result.raw_responses[0].output == [get_text_message("fourth")]
    assert result.last_agent == agent
    assert len(result.to_input_list()) == 3, "should have original input and generated items"


@pytest.mark.asyncio
async def test_tool_call_runs():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: text message
            [get_text_message("done")],
        ]
    )

    result = await Runner.run(agent, input="user_message")

    assert result.final_output == "done"
    assert len(result.raw_responses) == 2, (
        "should have two responses: the first which produces a tool call, and the second which"
        "handles the tool result"
    )

    assert len(result.to_input_list()) == 5, (
        "should have five inputs: the original input, the message, the tool call, the tool result "
        "and the done message"
    )


@pytest.mark.asyncio
async def test_handoffs():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    result = await Runner.run(agent_3, input="user_message")

    assert result.final_output == "done"
    assert len(result.raw_responses) == 3, "should have three model responses"
    assert len(result.to_input_list()) == 7, (
        "should have 7 inputs: orig input, tool call, tool result, message, handoff, handoff"
        "result, and done message"
    )
    assert result.last_agent == agent_1, "should have handed off to agent_1"


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_structured_output():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("bar", "bar_result")],
        output_type=Foo,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "foo_result")],
        handoffs=[agent_1],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("foo", json.dumps({"bar": "baz"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: tool call and structured output
            [
                get_function_tool_call("bar", json.dumps({"bar": "baz"})),
                get_final_output_message(json.dumps(Foo(bar="baz"))),
            ],
        ]
    )

    result = await Runner.run(
        agent_2,
        input=[
            get_text_input_item("user_message"),
            get_text_input_item("another_message"),
        ],
    )

    assert result.final_output == Foo(bar="baz")
    assert len(result.raw_responses) == 3, "should have three model responses"
    assert len(result.to_input_list()) == 10, (
        "should have input: 2 orig inputs, function call, function call result, message, handoff, "
        "handoff output, tool call, tool call result, final output message"
    )

    assert result.last_agent == agent_1, "should have handed off to agent_1"
    assert result.final_output == Foo(bar="baz"), "should have structured output"


def remove_new_items(handoff_input_data: HandoffInputData) -> HandoffInputData:
    return HandoffInputData(
        input_history=handoff_input_data.input_history,
        pre_handoff_items=(),
        new_items=(),
    )


@pytest.mark.asyncio
async def test_handoff_filters():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                input_filter=remove_new_items,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    result = await Runner.run(agent_2, input="user_message")

    assert result.final_output == "last"
    assert len(result.raw_responses) == 2, "should have two model responses"
    assert len(result.to_input_list()) == 2, (
        "should only have 2 inputs: orig input and last message"
    )


@pytest.mark.asyncio
async def test_async_input_filter_fails():
    # DO NOT rename this without updating pyproject.toml

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    async def invalid_input_filter(data: HandoffInputData) -> HandoffInputData:
        return data  # pragma: no cover

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                # Purposely ignoring the type error here to simulate invalid input
                input_filter=invalid_input_filter,  # type: ignore
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        await Runner.run(agent_2, input="user_message")


@pytest.mark.asyncio
async def test_invalid_input_filter_fails():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    def invalid_input_filter(data: HandoffInputData) -> HandoffInputData:
        # Purposely returning a string to simulate invalid output
        return "foo"  # type: ignore

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                input_filter=invalid_input_filter,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        await Runner.run(agent_2, input="user_message")


@pytest.mark.asyncio
async def test_non_callable_input_filter_causes_error():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                # Purposely ignoring the type error here to simulate invalid input
                input_filter="foo",  # type: ignore
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        await Runner.run(agent_2, input="user_message")


@pytest.mark.asyncio
async def test_handoff_on_input():
    call_output: str | None = None

    def on_input(_ctx: RunContextWrapper[Any], data: Foo) -> None:
        nonlocal call_output
        call_output = data["bar"]

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                on_handoff=on_input,
                input_type=Foo,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("1"),
                get_text_message("2"),
                get_handoff_tool_call(agent_1, args=json.dumps(Foo(bar="test_input"))),
            ],
            [get_text_message("last")],
        ]
    )

    result = await Runner.run(agent_2, input="user_message")

    assert result.final_output == "last"

    assert call_output == "test_input", "should have called the handoff with the correct input"


@pytest.mark.asyncio
async def test_async_handoff_on_input():
    call_output: str | None = None

    async def on_input(_ctx: RunContextWrapper[Any], data: Foo) -> None:
        nonlocal call_output
        call_output = data["bar"]

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                on_handoff=on_input,
                input_type=Foo,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("1"),
                get_text_message("2"),
                get_handoff_tool_call(agent_1, args=json.dumps(Foo(bar="test_input"))),
            ],
            [get_text_message("last")],
        ]
    )

    result = await Runner.run(agent_2, input="user_message")

    assert result.final_output == "last"

    assert call_output == "test_input", "should have called the handoff with the correct input"


@pytest.mark.asyncio
async def test_wrong_params_on_input_causes_error():
    agent_1 = Agent(
        name="test",
    )

    def _on_handoff_too_many_params(ctx: RunContextWrapper[Any], foo: Foo, bar: str) -> None:
        pass

    with pytest.raises(UserError):
        handoff(
            agent_1,
            input_type=Foo,
            # Purposely ignoring the type error here to simulate invalid input
            on_handoff=_on_handoff_too_many_params,  # type: ignore
        )

    def on_handoff_too_few_params(ctx: RunContextWrapper[Any]) -> None:
        pass

    with pytest.raises(UserError):
        handoff(
            agent_1,
            input_type=Foo,
            # Purposely ignoring the type error here to simulate invalid input
            on_handoff=on_handoff_too_few_params,  # type: ignore
        )


@pytest.mark.asyncio
async def test_invalid_handoff_input_json_causes_error():
    agent = Agent(name="test")
    h = handoff(agent, input_type=Foo, on_handoff=lambda _ctx, _input: None)

    with pytest.raises(ModelBehaviorError):
        await h.on_invoke_handoff(
            RunContextWrapper(None),
            # Purposely ignoring the type error here to simulate invalid input
            None,  # type: ignore
        )

    with pytest.raises(ModelBehaviorError):
        await h.on_invoke_handoff(RunContextWrapper(None), "invalid")


@pytest.mark.asyncio
async def test_input_guardrail_tripwire_triggered_causes_exception():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], input: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    agent = Agent(
        name="test", input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)]
    )
    model = FakeModel()
    model.set_next_output([get_text_message("user_message")])

    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, input="user_message")


@pytest.mark.asyncio
async def test_output_guardrail_tripwire_triggered_causes_exception():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    model = FakeModel()
    agent = Agent(
        name="test",
        output_guardrails=[OutputGuardrail(guardrail_function=guardrail_function)],
        model=model,
    )
    model.set_next_output([get_text_message("user_message")])

    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, input="user_message")


@function_tool
def test_tool_one():
    return Foo(bar="tool_one_result")


@function_tool
def test_tool_two():
    return "tool_two_result"


@pytest.mark.asyncio
async def test_tool_use_behavior_first_output():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "tool_result"), test_tool_one, test_tool_two],
        tool_use_behavior="stop_on_first_tool",
        output_type=Foo,
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [
                get_text_message("a_message"),
                get_function_tool_call("test_tool_one", None),
                get_function_tool_call("test_tool_two", None),
            ],
        ]
    )

    result = await Runner.run(agent, input="user_message")

    assert result.final_output == Foo(bar="tool_one_result"), (
        "should have used the first tool result"
    )


def custom_tool_use_behavior(
    context: RunContextWrapper[Any], results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    if "test_tool_one" in [result.tool.name for result in results]:
        return ToolsToFinalOutputResult(is_final_output=True, final_output="the_final_output")
    else:
        return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


@pytest.mark.asyncio
async def test_tool_use_behavior_custom_function():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "tool_result"), test_tool_one, test_tool_two],
        tool_use_behavior=custom_tool_use_behavior,
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [
                get_text_message("a_message"),
                get_function_tool_call("test_tool_two", None),
            ],
            # Second turn: a message and tool call
            [
                get_text_message("a_message"),
                get_function_tool_call("test_tool_one", None),
                get_function_tool_call("test_tool_two", None),
            ],
        ]
    )

    result = await Runner.run(agent, input="user_message")

    assert len(result.raw_responses) == 2, "should have two model responses"
    assert result.final_output == "the_final_output", "should have used the custom function"


<file_info>
path: examples/financial_research_agent/agents/planner_agent.py
name: planner_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

# Generate a plan of searches to ground the financial analysis.
# For a given financial question or company, we want to search for
# recent news, official filings, analyst commentary, and other
# relevant background.
PROMPT = (
    "You are a financial research planner. Given a request for financial analysis, "
    "produce a set of web searches to gather the context needed. Aim for recent "
    "headlines, earnings calls or 10‑K snippets, analyst commentary, and industry background. "
    "Output between 5 and 15 search terms to query for."
)


class FinancialSearchItem(BaseModel):
    reason: str
    """Your reasoning for why this search is relevant."""

    query: str
    """The search term to feed into a web (or file) search."""


class FinancialSearchPlan(BaseModel):
    searches: list[FinancialSearchItem]
    """A list of searches to perform."""


planner_agent = Agent(
    name="FinancialPlannerAgent",
    instructions=PROMPT,
    model="o3-mini",
    output_type=FinancialSearchPlan,
)


<file_info>
path: examples/basic/tools.py
name: tools.py
</file_info>
import asyncio

from pydantic import BaseModel

from agents import Agent, Runner, function_tool


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/research_bot/__init__.py
name: __init__.py
</file_info>



<file_info>
path: examples/research_bot/main.py
name: main.py
</file_info>
import asyncio

from .manager import ResearchManager


async def main() -> None:
    query = input("What would you like to research? ")
    await ResearchManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/voice/streamed/my_workflow.py
name: my_workflow.py
</file_info>
import random
from collections.abc import AsyncIterator
from typing import Callable

from agents import Agent, Runner, TResponseInputItem, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)


class MyWorkflow(VoiceWorkflowBase):
    def __init__(self, secret_word: str, on_start: Callable[[str], None]):
        """
        Args:
            secret_word: The secret word to guess.
            on_start: A callback that is called when the workflow starts. The transcription
                is passed in as an argument.
        """
        self._input_history: list[TResponseInputItem] = []
        self._current_agent = agent
        self._secret_word = secret_word.lower()
        self._on_start = on_start

    async def run(self, transcription: str) -> AsyncIterator[str]:
        self._on_start(transcription)

        # Add the transcription to the input history
        self._input_history.append(
            {
                "role": "user",
                "content": transcription,
            }
        )

        # If the user guessed the secret word, do alternate logic
        if self._secret_word in transcription.lower():
            yield "You guessed the secret word!"
            self._input_history.append(
                {
                    "role": "assistant",
                    "content": "You guessed the secret word!",
                }
            )
            return

        # Otherwise, run the agent
        result = Runner.run_streamed(self._current_agent, self._input_history)

        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Update the input history and current agent
        self._input_history = result.to_input_list()
        self._current_agent = result.last_agent


<file_info>
path: src/agents/tracing/create.py
name: create.py
</file_info>
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..logger import logger
from .setup import GLOBAL_TRACE_PROVIDER
from .span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpeechGroupSpanData,
    SpeechSpanData,
    TranscriptionSpanData,
)
from .spans import Span
from .traces import Trace

if TYPE_CHECKING:
    from openai.types.responses import Response


def trace(
    workflow_name: str,
    trace_id: str | None = None,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Trace:
    """
    Create a new trace. The trace will not be started automatically; you should either use
    it as a context manager (`with trace(...):`) or call `trace.start()` + `trace.finish()`
    manually.

    In addition to the workflow name and optional grouping identifier, you can provide
    an arbitrary metadata dictionary to attach additional user-defined information to
    the trace.

    Args:
        workflow_name: The name of the logical app or workflow. For example, you might provide
            "code_bot" for a coding agent, or "customer_support_agent" for a customer support agent.
        trace_id: The ID of the trace. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_trace_id()` to generate a trace ID, to guarantee that IDs are
            correctly formatted.
        group_id: Optional grouping identifier to link multiple traces from the same conversation
            or process. For instance, you might use a chat thread ID.
        metadata: Optional dictionary of additional metadata to attach to the trace.
        disabled: If True, we will return a Trace but the Trace will not be recorded. This will
            not be checked if there's an existing trace and `even_if_trace_running` is True.

    Returns:
        The newly created trace object.
    """
    current_trace = GLOBAL_TRACE_PROVIDER.get_current_trace()
    if current_trace:
        logger.warning(
            "Trace already exists. Creating a new trace, but this is probably a mistake."
        )

    return GLOBAL_TRACE_PROVIDER.create_trace(
        name=workflow_name,
        trace_id=trace_id,
        group_id=group_id,
        metadata=metadata,
        disabled=disabled,
    )


def get_current_trace() -> Trace | None:
    """Returns the currently active trace, if present."""
    return GLOBAL_TRACE_PROVIDER.get_current_trace()


def get_current_span() -> Span[Any] | None:
    """Returns the currently active span, if present."""
    return GLOBAL_TRACE_PROVIDER.get_current_span()


def agent_span(
    name: str,
    handoffs: list[str] | None = None,
    tools: list[str] | None = None,
    output_type: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[AgentSpanData]:
    """Create a new agent span. The span will not be started automatically, you should either do
    `with agent_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the agent.
        handoffs: Optional list of agent names to which this agent could hand off control.
        tools: Optional list of tool names available to this agent.
        output_type: Optional name of the output type produced by the agent.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created agent span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AgentSpanData(name=name, handoffs=handoffs, tools=tools, output_type=output_type),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def function_span(
    name: str,
    input: str | None = None,
    output: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[FunctionSpanData]:
    """Create a new function span. The span will not be started automatically, you should either do
    `with function_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the function.
        input: The input to the function.
        output: The output of the function.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created function span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=FunctionSpanData(name=name, input=input, output=output),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def generation_span(
    input: Sequence[Mapping[str, Any]] | None = None,
    output: Sequence[Mapping[str, Any]] | None = None,
    model: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[GenerationSpanData]:
    """Create a new generation span. The span will not be started automatically, you should either
    do `with generation_span() ...` or call `span.start()` + `span.finish()` manually.

    This span captures the details of a model generation, including the
    input message sequence, any generated outputs, the model name and
    configuration, and usage data. If you only need to capture a model
    response identifier, use `response_span()` instead.

    Args:
        input: The sequence of input messages sent to the model.
        output: The sequence of output messages received from the model.
        model: The model identifier used for the generation.
        model_config: The model configuration (hyperparameters) used.
        usage: A dictionary of usage information (input tokens, output tokens, etc.).
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created generation span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=GenerationSpanData(
            input=input,
            output=output,
            model=model,
            model_config=model_config,
            usage=usage,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def response_span(
    response: Response | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[ResponseSpanData]:
    """Create a new response span. The span will not be started automatically, you should either do
    `with response_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        response: The OpenAI Response object.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=ResponseSpanData(response=response),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def handoff_span(
    from_agent: str | None = None,
    to_agent: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[HandoffSpanData]:
    """Create a new handoff span. The span will not be started automatically, you should either do
    `with handoff_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        from_agent: The name of the agent that is handing off.
        to_agent: The name of the agent that is receiving the handoff.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created handoff span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=HandoffSpanData(from_agent=from_agent, to_agent=to_agent),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def custom_span(
    name: str,
    data: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    """Create a new custom span, to which you can add your own metadata. The span will not be
    started automatically, you should either do `with custom_span() ...` or call
    `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the custom span.
        data: Arbitrary structured data to associate with the span.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created custom span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=CustomSpanData(name=name, data=data or {}),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def guardrail_span(
    name: str,
    triggered: bool = False,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[GuardrailSpanData]:
    """Create a new guardrail span. The span will not be started automatically, you should either
    do `with guardrail_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the guardrail.
        triggered: Whether the guardrail was triggered.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=GuardrailSpanData(name=name, triggered=triggered),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def transcription_span(
    model: str | None = None,
    input: str | None = None,
    input_format: str | None = "pcm",
    output: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[TranscriptionSpanData]:
    """Create a new transcription span. The span will not be started automatically, you should
    either do `with transcription_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        model: The name of the model used for the speech-to-text.
        input: The audio input of the speech-to-text transcription, as a base64 encoded string of
            audio bytes.
        input_format: The format of the audio input (defaults to "pcm").
        output: The output of the speech-to-text transcription.
        model_config: The model configuration (hyperparameters) used.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created speech-to-text span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=TranscriptionSpanData(
            input=input,
            input_format=input_format,
            output=output,
            model=model,
            model_config=model_config,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def speech_span(
    model: str | None = None,
    input: str | None = None,
    output: str | None = None,
    output_format: str | None = "pcm",
    model_config: Mapping[str, Any] | None = None,
    first_content_at: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[SpeechSpanData]:
    """Create a new speech span. The span will not be started automatically, you should either do
    `with speech_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        model: The name of the model used for the text-to-speech.
        input: The text input of the text-to-speech.
        output: The audio output of the text-to-speech as base64 encoded string of PCM audio bytes.
        output_format: The format of the audio output (defaults to "pcm").
        model_config: The model configuration (hyperparameters) used.
        first_content_at: The time of the first byte of the audio output.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=SpeechSpanData(
            model=model,
            input=input,
            output=output,
            output_format=output_format,
            model_config=model_config,
            first_content_at=first_content_at,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def speech_group_span(
    input: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[SpeechGroupSpanData]:
    """Create a new speech group span. The span will not be started automatically, you should
    either do `with speech_group_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        input: The input text used for the speech request.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=SpeechGroupSpanData(input=input),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


<file_info>
path: src/agents/usage.py
name: usage.py
</file_info>
from dataclasses import dataclass


@dataclass
class Usage:
    requests: int = 0
    """Total requests made to the LLM API."""

    input_tokens: int = 0
    """Total input tokens sent, across all requests."""

    output_tokens: int = 0
    """Total output tokens received, across all requests."""

    total_tokens: int = 0
    """Total tokens sent and received, across all requests."""

    def add(self, other: "Usage") -> None:
        self.requests += other.requests if other.requests else 0
        self.input_tokens += other.input_tokens if other.input_tokens else 0
        self.output_tokens += other.output_tokens if other.output_tokens else 0
        self.total_tokens += other.total_tokens if other.total_tokens else 0


<file_info>
path: tests/test_agent_runner_streamed.py
name: test_agent_runner_streamed.py
</file_info>
from __future__ import annotations

import json
from typing import Any

import pytest
from typing_extensions import TypedDict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    Handoff,
    HandoffInputData,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    UserError,
    handoff,
)
from agents.items import RunItem
from agents.run import RunConfig
from agents.stream_events import AgentUpdatedStreamEvent

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_input_item,
    get_text_message,
)


@pytest.mark.asyncio
async def test_simple_first_run():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )
    model.set_next_output([get_text_message("first")])

    result = Runner.run_streamed(agent, input="test")
    async for _ in result.stream_events():
        pass

    assert result.input == "test"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "first"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert result.raw_responses[0].output == [get_text_message("first")]
    assert result.last_agent == agent

    assert len(result.to_input_list()) == 2, "should have original input and generated item"

    model.set_next_output([get_text_message("second")])

    result = Runner.run_streamed(
        agent, input=[get_text_input_item("message"), get_text_input_item("another_message")]
    )
    async for _ in result.stream_events():
        pass

    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "second"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert len(result.to_input_list()) == 3, "should have original input and generated item"


@pytest.mark.asyncio
async def test_subsequent_runs():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )
    model.set_next_output([get_text_message("third")])

    result = Runner.run_streamed(agent, input="test")
    async for _ in result.stream_events():
        pass

    assert result.input == "test"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert len(result.to_input_list()) == 2, "should have original input and generated item"

    model.set_next_output([get_text_message("fourth")])

    result = Runner.run_streamed(agent, input=result.to_input_list())
    async for _ in result.stream_events():
        pass

    assert len(result.input) == 2, f"should have previous input but got {result.input}"
    assert len(result.new_items) == 1, "exactly one item should be generated"
    assert result.final_output == "fourth"
    assert len(result.raw_responses) == 1, "exactly one model response should be generated"
    assert result.raw_responses[0].output == [get_text_message("fourth")]
    assert result.last_agent == agent
    assert len(result.to_input_list()) == 3, "should have original input and generated items"


@pytest.mark.asyncio
async def test_tool_call_runs():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: text message
            [get_text_message("done")],
        ]
    )

    result = Runner.run_streamed(agent, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "done"
    assert len(result.raw_responses) == 2, (
        "should have two responses: the first which produces a tool call, and the second which"
        "handles the tool result"
    )

    assert len(result.to_input_list()) == 5, (
        "should have five inputs: the original input, the message, the tool call, the tool result "
        "and the done message"
    )


@pytest.mark.asyncio
async def test_handoffs():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "done"
    assert len(result.raw_responses) == 3, "should have three model responses"
    assert len(result.to_input_list()) == 7, (
        "should have 7 inputs: orig input, tool call, tool result, message, handoff, handoff"
        "result, and done message"
    )
    assert result.last_agent == agent_1, "should have handed off to agent_1"


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_structured_output():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("bar", "bar_result")],
        output_type=Foo,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "foo_result")],
        handoffs=[agent_1],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("foo", json.dumps({"bar": "baz"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: tool call and structured output
            [
                get_function_tool_call("bar", json.dumps({"bar": "baz"})),
                get_final_output_message(json.dumps(Foo(bar="baz"))),
            ],
        ]
    )

    result = Runner.run_streamed(
        agent_2,
        input=[
            get_text_input_item("user_message"),
            get_text_input_item("another_message"),
        ],
    )
    async for _ in result.stream_events():
        pass

    assert result.final_output == Foo(bar="baz")
    assert len(result.raw_responses) == 3, "should have three model responses"
    assert len(result.to_input_list()) == 10, (
        "should have input: 2 orig inputs, function call, function call result, message, handoff, "
        "handoff output, tool call, tool call result, final output"
    )

    assert result.last_agent == agent_1, "should have handed off to agent_1"
    assert result.final_output == Foo(bar="baz"), "should have structured output"


def remove_new_items(handoff_input_data: HandoffInputData) -> HandoffInputData:
    return HandoffInputData(
        input_history=handoff_input_data.input_history,
        pre_handoff_items=(),
        new_items=(),
    )


@pytest.mark.asyncio
async def test_handoff_filters():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                input_filter=remove_new_items,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    result = Runner.run_streamed(agent_2, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "last"
    assert len(result.raw_responses) == 2, "should have two model responses"
    assert len(result.to_input_list()) == 2, (
        "should only have 2 inputs: orig input and last message"
    )


@pytest.mark.asyncio
async def test_async_input_filter_fails():
    # DO NOT rename this without updating pyproject.toml

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    async def invalid_input_filter(data: HandoffInputData) -> HandoffInputData:
        return data  # pragma: no cover

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                # Purposely ignoring the type error here to simulate invalid input
                input_filter=invalid_input_filter,  # type: ignore
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        result = Runner.run_streamed(agent_2, input="user_message")
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_invalid_input_filter_fails():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    def invalid_input_filter(data: HandoffInputData) -> HandoffInputData:
        # Purposely returning a string to simulate invalid output
        return "foo"  # type: ignore

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                input_filter=invalid_input_filter,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        result = Runner.run_streamed(agent_2, input="user_message")
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_non_callable_input_filter_causes_error():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    async def on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
        return agent_1

    agent_2 = Agent[None](
        name="test",
        model=model,
        handoffs=[
            Handoff(
                tool_name=Handoff.default_tool_name(agent_1),
                tool_description=Handoff.default_tool_description(agent_1),
                input_json_schema={},
                on_invoke_handoff=on_invoke_handoff,
                agent_name=agent_1.name,
                # Purposely ignoring the type error here to simulate invalid input
                input_filter="foo",  # type: ignore
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_text_message("2"), get_handoff_tool_call(agent_1)],
            [get_text_message("last")],
        ]
    )

    with pytest.raises(UserError):
        result = Runner.run_streamed(agent_2, input="user_message")
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_handoff_on_input():
    call_output: str | None = None

    def on_input(_ctx: RunContextWrapper[Any], data: Foo) -> None:
        nonlocal call_output
        call_output = data["bar"]

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                on_handoff=on_input,
                input_type=Foo,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("1"),
                get_text_message("2"),
                get_handoff_tool_call(agent_1, args=json.dumps(Foo(bar="test_input"))),
            ],
            [get_text_message("last")],
        ]
    )

    result = Runner.run_streamed(agent_2, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "last"

    assert call_output == "test_input", "should have called the handoff with the correct input"


@pytest.mark.asyncio
async def test_async_handoff_on_input():
    call_output: str | None = None

    async def on_input(_ctx: RunContextWrapper[Any], data: Foo) -> None:
        nonlocal call_output
        call_output = data["bar"]

    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        handoffs=[
            handoff(
                agent=agent_1,
                on_handoff=on_input,
                input_type=Foo,
            )
        ],
    )

    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("1"),
                get_text_message("2"),
                get_handoff_tool_call(agent_1, args=json.dumps(Foo(bar="test_input"))),
            ],
            [get_text_message("last")],
        ]
    )

    result = Runner.run_streamed(agent_2, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "last"

    assert call_output == "test_input", "should have called the handoff with the correct input"


@pytest.mark.asyncio
async def test_input_guardrail_tripwire_triggered_causes_exception_streamed():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], input: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    agent = Agent(
        name="test",
        input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)],
        model=FakeModel(),
    )

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_output_guardrail_tripwire_triggered_causes_exception_streamed():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    model = FakeModel(initial_output=[get_text_message("first_test")])

    agent = Agent(
        name="test",
        output_guardrails=[OutputGuardrail(guardrail_function=guardrail_function)],
        model=model,
    )

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_run_input_guardrail_tripwire_triggered_causes_exception_streamed():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], input: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    agent = Agent(
        name="test",
        model=FakeModel(),
    )

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(
            agent,
            input="user_message",
            run_config=RunConfig(
                input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)]
            ),
        )
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_run_output_guardrail_tripwire_triggered_causes_exception_streamed():
    def guardrail_function(
        context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=True,
        )

    model = FakeModel(initial_output=[get_text_message("first_test")])

    agent = Agent(
        name="test",
        model=model,
    )

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(
            agent,
            input="user_message",
            run_config=RunConfig(
                output_guardrails=[OutputGuardrail(guardrail_function=guardrail_function)]
            ),
        )
        async for _ in result.stream_events():
            pass


@pytest.mark.asyncio
async def test_streaming_events():
    model = FakeModel()
    agent_1 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("bar", "bar_result")],
        output_type=Foo,
    )

    agent_2 = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "foo_result")],
        handoffs=[agent_1],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("foo", json.dumps({"bar": "baz"}))],
            # Second turn: a message and a handoff
            [get_text_message("a_message"), get_handoff_tool_call(agent_1)],
            # Third turn: tool call and structured output
            [
                get_function_tool_call("bar", json.dumps({"bar": "baz"})),
                get_final_output_message(json.dumps(Foo(bar="baz"))),
            ],
        ]
    )

    # event_type: (count, event)
    event_counts: dict[str, int] = {}
    item_data: list[RunItem] = []
    agent_data: list[AgentUpdatedStreamEvent] = []

    result = Runner.run_streamed(
        agent_2,
        input=[
            get_text_input_item("user_message"),
            get_text_input_item("another_message"),
        ],
    )
    async for event in result.stream_events():
        event_counts[event.type] = event_counts.get(event.type, 0) + 1
        if event.type == "run_item_stream_event":
            item_data.append(event.item)
        elif event.type == "agent_updated_stream_event":
            agent_data.append(event)

    assert result.final_output == Foo(bar="baz")
    assert len(result.raw_responses) == 3, "should have three model responses"
    assert len(result.to_input_list()) == 10, (
        "should have input: 2 orig inputs, function call, function call result, message, handoff, "
        "handoff output, tool call, tool call result, final output"
    )

    assert result.last_agent == agent_1, "should have handed off to agent_1"
    assert result.final_output == Foo(bar="baz"), "should have structured output"

    # Now lets check the events

    expected_item_type_map = {
        "tool_call": 2,
        "tool_call_output": 2,
        "message": 2,
        "handoff": 1,
        "handoff_output": 1,
    }

    total_expected_item_count = sum(expected_item_type_map.values())

    assert event_counts["run_item_stream_event"] == total_expected_item_count, (
        f"Expected {total_expected_item_count} events, got {event_counts['run_item_stream_event']}"
        f"Expected events were: {expected_item_type_map}, got {event_counts}"
    )

    assert len(item_data) == total_expected_item_count, (
        f"should have {total_expected_item_count} run items"
    )
    assert len(agent_data) == 2, "should have 2 agent updated events"
    assert agent_data[0].new_agent == agent_2, "should have started with agent_2"
    assert agent_data[1].new_agent == agent_1, "should have handed off to agent_1"


<file_info>
path: examples/agent_patterns/deterministic.py
name: deterministic.py
</file_info>
import asyncio

from pydantic import BaseModel

from agents import Agent, Runner, trace

"""
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
"""

story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
)


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool


outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions="Read the given story outline, and judge the quality. Also, determine if it is a scifi story.",
    output_type=OutlineCheckerOutput,
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,
)


async def main():
    input_prompt = input("What kind of story do you want? ")

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):
        # 1. Generate an outline
        outline_result = await Runner.run(
            story_outline_agent,
            input_prompt,
        )
        print("Outline generated")

        # 2. Check the outline
        outline_checker_result = await Runner.run(
            outline_checker_agent,
            outline_result.final_output,
        )

        # 3. Add a gate to stop if the outline is not good quality or not a scifi story
        assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
        if not outline_checker_result.final_output.good_quality:
            print("Outline is not good quality, so we stop here.")
            exit(0)

        if not outline_checker_result.final_output.is_scifi:
            print("Outline is not a scifi story, so we stop here.")
            exit(0)

        print("Outline is good quality and a scifi story, so we continue to write the story.")

        # 4. Write the story
        story_result = await Runner.run(
            story_agent,
            outline_result.final_output,
        )
        print(f"Story: {story_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/setup.py
name: setup.py
</file_info>
from __future__ import annotations

import os
import threading
from typing import Any

from ..logger import logger
from . import util
from .processor_interface import TracingProcessor
from .scope import Scope
from .spans import NoOpSpan, Span, SpanImpl, TSpanData
from .traces import NoOpTrace, Trace, TraceImpl


class SynchronousMultiTracingProcessor(TracingProcessor):
    """
    Forwards all calls to a list of TracingProcessors, in order of registration.
    """

    def __init__(self):
        # Using a tuple to avoid race conditions when iterating over processors
        self._processors: tuple[TracingProcessor, ...] = ()
        self._lock = threading.Lock()

    def add_tracing_processor(self, tracing_processor: TracingProcessor):
        """
        Add a processor to the list of processors. Each processor will receive all traces/spans.
        """
        with self._lock:
            self._processors += (tracing_processor,)

    def set_processors(self, processors: list[TracingProcessor]):
        """
        Set the list of processors. This will replace the current list of processors.
        """
        with self._lock:
            self._processors = tuple(processors)

    def on_trace_start(self, trace: Trace) -> None:
        """
        Called when a trace is started.
        """
        for processor in self._processors:
            processor.on_trace_start(trace)

    def on_trace_end(self, trace: Trace) -> None:
        """
        Called when a trace is finished.
        """
        for processor in self._processors:
            processor.on_trace_end(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        """
        Called when a span is started.
        """
        for processor in self._processors:
            processor.on_span_start(span)

    def on_span_end(self, span: Span[Any]) -> None:
        """
        Called when a span is finished.
        """
        for processor in self._processors:
            processor.on_span_end(span)

    def shutdown(self) -> None:
        """
        Called when the application stops.
        """
        for processor in self._processors:
            logger.debug(f"Shutting down trace processor {processor}")
            processor.shutdown()

    def force_flush(self):
        """
        Force the processors to flush their buffers.
        """
        for processor in self._processors:
            processor.force_flush()


class TraceProvider:
    def __init__(self):
        self._multi_processor = SynchronousMultiTracingProcessor()
        self._disabled = os.environ.get("OPENAI_AGENTS_DISABLE_TRACING", "false").lower() in (
            "true",
            "1",
        )

    def register_processor(self, processor: TracingProcessor):
        """
        Add a processor to the list of processors. Each processor will receive all traces/spans.
        """
        self._multi_processor.add_tracing_processor(processor)

    def set_processors(self, processors: list[TracingProcessor]):
        """
        Set the list of processors. This will replace the current list of processors.
        """
        self._multi_processor.set_processors(processors)

    def get_current_trace(self) -> Trace | None:
        """
        Returns the currently active trace, if any.
        """
        return Scope.get_current_trace()

    def get_current_span(self) -> Span[Any] | None:
        """
        Returns the currently active span, if any.
        """
        return Scope.get_current_span()

    def set_disabled(self, disabled: bool) -> None:
        """
        Set whether tracing is disabled.
        """
        self._disabled = disabled

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace:
        """
        Create a new trace.
        """
        if self._disabled or disabled:
            logger.debug(f"Tracing is disabled. Not creating trace {name}")
            return NoOpTrace()

        trace_id = trace_id or util.gen_trace_id()

        logger.debug(f"Creating trace {name} with id {trace_id}")

        return TraceImpl(
            name=name,
            trace_id=trace_id,
            group_id=group_id,
            metadata=metadata,
            processor=self._multi_processor,
        )

    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Trace | Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        """
        Create a new span.
        """
        if self._disabled or disabled:
            logger.debug(f"Tracing is disabled. Not creating span {span_data}")
            return NoOpSpan(span_data)

        if not parent:
            current_span = Scope.get_current_span()
            current_trace = Scope.get_current_trace()
            if current_trace is None:
                logger.error(
                    "No active trace. Make sure to start a trace with `trace()` first"
                    "Returning NoOpSpan."
                )
                return NoOpSpan(span_data)
            elif isinstance(current_trace, NoOpTrace) or isinstance(current_span, NoOpSpan):
                logger.debug(
                    f"Parent {current_span} or {current_trace} is no-op, returning NoOpSpan"
                )
                return NoOpSpan(span_data)

            parent_id = current_span.span_id if current_span else None
            trace_id = current_trace.trace_id

        elif isinstance(parent, Trace):
            if isinstance(parent, NoOpTrace):
                logger.debug(f"Parent {parent} is no-op, returning NoOpSpan")
                return NoOpSpan(span_data)
            trace_id = parent.trace_id
            parent_id = None
        elif isinstance(parent, Span):
            if isinstance(parent, NoOpSpan):
                logger.debug(f"Parent {parent} is no-op, returning NoOpSpan")
                return NoOpSpan(span_data)
            parent_id = parent.span_id
            trace_id = parent.trace_id

        logger.debug(f"Creating span {span_data} with id {span_id}")

        return SpanImpl(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            processor=self._multi_processor,
            span_data=span_data,
        )

    def shutdown(self) -> None:
        try:
            logger.debug("Shutting down trace provider")
            self._multi_processor.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down trace provider: {e}")


GLOBAL_TRACE_PROVIDER = TraceProvider()


<file_info>
path: src/agents/computer.py
name: computer.py
</file_info>
import abc
from typing import Literal

Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]


class Computer(abc.ABC):
    """A computer implemented with sync operations. The Computer interface abstracts the
    operations needed to control a computer or browser."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def screenshot(self) -> str:
        pass

    @abc.abstractmethod
    def click(self, x: int, y: int, button: Button) -> None:
        pass

    @abc.abstractmethod
    def double_click(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        pass

    @abc.abstractmethod
    def type(self, text: str) -> None:
        pass

    @abc.abstractmethod
    def wait(self) -> None:
        pass

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    def keypress(self, keys: list[str]) -> None:
        pass

    @abc.abstractmethod
    def drag(self, path: list[tuple[int, int]]) -> None:
        pass


class AsyncComputer(abc.ABC):
    """A computer implemented with async operations. The Computer interface abstracts the
    operations needed to control a computer or browser."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    async def screenshot(self) -> str:
        pass

    @abc.abstractmethod
    async def click(self, x: int, y: int, button: Button) -> None:
        pass

    @abc.abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        pass

    @abc.abstractmethod
    async def type(self, text: str) -> None:
        pass

    @abc.abstractmethod
    async def wait(self) -> None:
        pass

    @abc.abstractmethod
    async def move(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    async def keypress(self, keys: list[str]) -> None:
        pass

    @abc.abstractmethod
    async def drag(self, path: list[tuple[int, int]]) -> None:
        pass


<file_info>
path: src/agents/voice/models/openai_tts.py
name: openai_tts.py
</file_info>
from collections.abc import AsyncIterator
from typing import Literal

from openai import AsyncOpenAI

from ..model import TTSModel, TTSModelSettings

DEFAULT_VOICE: Literal["ash"] = "ash"


class OpenAITTSModel(TTSModel):
    """A text-to-speech model for OpenAI."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
    ):
        """Create a new OpenAI text-to-speech model.

        Args:
            model: The name of the model to use.
            openai_client: The OpenAI client to use.
        """
        self.model = model
        self._client = openai_client

    @property
    def model_name(self) -> str:
        return self.model

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        """Run the text-to-speech model.

        Args:
            text: The text to convert to speech.
            settings: The settings to use for the text-to-speech model.

        Returns:
            An iterator of audio chunks.
        """
        response = self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=settings.voice or DEFAULT_VOICE,
            input=text,
            response_format="pcm",
            extra_body={
                "instructions": settings.instructions,
            },
        )

        async with response as stream:
            async for chunk in stream.iter_bytes(chunk_size=1024):
                yield chunk


<file_info>
path: src/agents/voice/model.py
name: model.py
</file_info>
from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .imports import np, npt
from .input import AudioInput, StreamedAudioInput
from .utils import get_sentence_based_splitter

DEFAULT_TTS_INSTRUCTIONS = (
    "You will receive partial sentences. Do not complete the sentence, just read out the text."
)
DEFAULT_TTS_BUFFER_SIZE = 120


@dataclass
class TTSModelSettings:
    """Settings for a TTS model."""

    voice: (
        Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"] | None
    ) = None
    """
    The voice to use for the TTS model. If not provided, the default voice for the respective model
    will be used.
    """

    buffer_size: int = 120
    """The minimal size of the chunks of audio data that are being streamed out."""

    dtype: npt.DTypeLike = np.int16
    """The data type for the audio data to be returned in."""

    transform_data: (
        Callable[[npt.NDArray[np.int16 | np.float32]], npt.NDArray[np.int16 | np.float32]] | None
    ) = None
    """
    A function to transform the data from the TTS model. This is useful if you want the resulting
    audio stream to have the data in a specific shape already.
    """

    instructions: str = (
        "You will receive partial sentences. Do not complete the sentence just read out the text."
    )
    """
    The instructions to use for the TTS model. This is useful if you want to control the tone of the
    audio output.
    """

    text_splitter: Callable[[str], tuple[str, str]] = get_sentence_based_splitter()
    """
    A function to split the text into chunks. This is useful if you want to split the text into
    chunks before sending it to the TTS model rather than waiting for the whole text to be
    processed.
    """

    speed: float | None = None
    """The speed with which the TTS model will read the text. Between 0.25 and 4.0."""


class TTSModel(abc.ABC):
    """A text-to-speech model that can convert text into audio output."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """The name of the TTS model."""
        pass

    @abc.abstractmethod
    def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        """Given a text string, produces a stream of audio bytes, in PCM format.

        Args:
            text: The text to convert to audio.

        Returns:
            An async iterator of audio bytes, in PCM format.
        """
        pass


class StreamedTranscriptionSession(abc.ABC):
    """A streamed transcription of audio input."""

    @abc.abstractmethod
    def transcribe_turns(self) -> AsyncIterator[str]:
        """Yields a stream of text transcriptions. Each transcription is a turn in the conversation.

        This method is expected to return only after `close()` is called.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Closes the session."""
        pass


@dataclass
class STTModelSettings:
    """Settings for a speech-to-text model."""

    prompt: str | None = None
    """Instructions for the model to follow."""

    language: str | None = None
    """The language of the audio input."""

    temperature: float | None = None
    """The temperature of the model."""

    turn_detection: dict[str, Any] | None = None
    """The turn detection settings for the model when using streamed audio input."""


class STTModel(abc.ABC):
    """A speech-to-text model that can convert audio input into text."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """The name of the STT model."""
        pass

    @abc.abstractmethod
    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        """Given an audio input, produces a text transcription.

        Args:
            input: The audio input to transcribe.
            settings: The settings to use for the transcription.
            trace_include_sensitive_data: Whether to include sensitive data in traces.
            trace_include_sensitive_audio_data: Whether to include sensitive audio data in traces.

        Returns:
            The text transcription of the audio input.
        """
        pass

    @abc.abstractmethod
    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        """Creates a new transcription session, which you can push audio to, and receive a stream
        of text transcriptions.

        Args:
            input: The audio input to transcribe.
            settings: The settings to use for the transcription.
            trace_include_sensitive_data: Whether to include sensitive data in traces.
            trace_include_sensitive_audio_data: Whether to include sensitive audio data in traces.

        Returns:
            A new transcription session.
        """
        pass


class VoiceModelProvider(abc.ABC):
    """The base interface for a voice model provider.

    A model provider is responsible for creating speech-to-text and text-to-speech models, given a
    name.
    """

    @abc.abstractmethod
    def get_stt_model(self, model_name: str | None) -> STTModel:
        """Get a speech-to-text model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The speech-to-text model.
        """
        pass

    @abc.abstractmethod
    def get_tts_model(self, model_name: str | None) -> TTSModel:
        """Get a text-to-speech model by name."""


<file_info>
path: src/agents/stream_events.py
name: stream_events.py
</file_info>
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from typing_extensions import TypeAlias

from .agent import Agent
from .items import RunItem, TResponseStreamEvent


@dataclass
class RawResponsesStreamEvent:
    """Streaming event from the LLM. These are 'raw' events, i.e. they are directly passed through
    from the LLM.
    """

    data: TResponseStreamEvent
    """The raw responses streaming event from the LLM."""

    type: Literal["raw_response_event"] = "raw_response_event"
    """The type of the event."""


@dataclass
class RunItemStreamEvent:
    """Streaming events that wrap a `RunItem`. As the agent processes the LLM response, it will
    generate these events for new messages, tool calls, tool outputs, handoffs, etc.
    """

    name: Literal[
        "message_output_created",
        "handoff_requested",
        "handoff_occured",
        "tool_called",
        "tool_output",
        "reasoning_item_created",
    ]
    """The name of the event."""

    item: RunItem
    """The item that was created."""

    type: Literal["run_item_stream_event"] = "run_item_stream_event"


@dataclass
class AgentUpdatedStreamEvent:
    """Event that notifies that there is a new agent running."""

    new_agent: Agent[Any]
    """The new agent."""

    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


StreamEvent: TypeAlias = Union[RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent]
"""A streaming event from an agent."""


<file_info>
path: src/agents/items.py
name: items.py
</file_info>
from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

from openai.types.responses import (
    Response,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseStreamEvent,
)
from openai.types.responses.response_input_item_param import ComputerCallOutput, FunctionCallOutput
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel
from typing_extensions import TypeAlias

from .exceptions import AgentsException, ModelBehaviorError
from .usage import Usage

if TYPE_CHECKING:
    from .agent import Agent

TResponse = Response
"""A type alias for the Response type from the OpenAI SDK."""

TResponseInputItem = ResponseInputItemParam
"""A type alias for the ResponseInputItemParam type from the OpenAI SDK."""

TResponseOutputItem = ResponseOutputItem
"""A type alias for the ResponseOutputItem type from the OpenAI SDK."""

TResponseStreamEvent = ResponseStreamEvent
"""A type alias for the ResponseStreamEvent type from the OpenAI SDK."""

T = TypeVar("T", bound=Union[TResponseOutputItem, TResponseInputItem])


@dataclass
class RunItemBase(Generic[T], abc.ABC):
    agent: Agent[Any]
    """The agent whose run caused this item to be generated."""

    raw_item: T
    """The raw Responses item from the run. This will always be a either an output item (i.e.
    `openai.types.responses.ResponseOutputItem` or an input item
    (i.e. `openai.types.responses.ResponseInputItemParam`).
    """

    def to_input_item(self) -> TResponseInputItem:
        """Converts this item into an input item suitable for passing to the model."""
        if isinstance(self.raw_item, dict):
            # We know that input items are dicts, so we can ignore the type error
            return self.raw_item  # type: ignore
        elif isinstance(self.raw_item, BaseModel):
            # All output items are Pydantic models that can be converted to input items.
            return self.raw_item.model_dump(exclude_unset=True)  # type: ignore
        else:
            raise AgentsException(f"Unexpected raw item type: {type(self.raw_item)}")


@dataclass
class MessageOutputItem(RunItemBase[ResponseOutputMessage]):
    """Represents a message from the LLM."""

    raw_item: ResponseOutputMessage
    """The raw response output message."""

    type: Literal["message_output_item"] = "message_output_item"


@dataclass
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    """Represents a tool call for a handoff from one agent to another."""

    raw_item: ResponseFunctionToolCall
    """The raw response function tool call that represents the handoff."""

    type: Literal["handoff_call_item"] = "handoff_call_item"


@dataclass
class HandoffOutputItem(RunItemBase[TResponseInputItem]):
    """Represents the output of a handoff."""

    raw_item: TResponseInputItem
    """The raw input item that represents the handoff taking place."""

    source_agent: Agent[Any]
    """The agent that made the handoff."""

    target_agent: Agent[Any]
    """The agent that is being handed off to."""

    type: Literal["handoff_output_item"] = "handoff_output_item"


ToolCallItemTypes: TypeAlias = Union[
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
]
"""A type that represents a tool call item."""


@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Represents a tool call e.g. a function call or computer action call."""

    raw_item: ToolCallItemTypes
    """The raw tool call item."""

    type: Literal["tool_call_item"] = "tool_call_item"


@dataclass
class ToolCallOutputItem(RunItemBase[Union[FunctionCallOutput, ComputerCallOutput]]):
    """Represents the output of a tool call."""

    raw_item: FunctionCallOutput | ComputerCallOutput
    """The raw item from the model."""

    output: Any
    """The output of the tool call. This is whatever the tool call returned; the `raw_item`
    contains a string representation of the output.
    """

    type: Literal["tool_call_output_item"] = "tool_call_output_item"


@dataclass
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    """Represents a reasoning item."""

    raw_item: ResponseReasoningItem
    """The raw reasoning item."""

    type: Literal["reasoning_item"] = "reasoning_item"


RunItem: TypeAlias = Union[
    MessageOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
]
"""An item generated by an agent."""


@dataclass
class ModelResponse:
    output: list[TResponseOutputItem]
    """A list of outputs (messages, tool calls, etc) generated by the model"""

    usage: Usage
    """The usage information for the response."""

    referenceable_id: str | None
    """An ID for the response which can be used to refer to the response in subsequent calls to the
    model. Not supported by all model providers.
    """

    def to_input_items(self) -> list[TResponseInputItem]:
        """Convert the output into a list of input items suitable for passing to the model."""
        # We happen to know that the shape of the Pydantic output items are the same as the
        # equivalent TypedDict input items, so we can just convert each one.
        # This is also tested via unit tests.
        return [it.model_dump(exclude_unset=True) for it in self.output]  # type: ignore


class ItemHelpers:
    @classmethod
    def extract_last_content(cls, message: TResponseOutputItem) -> str:
        """Extracts the last text content or refusal from a message."""
        if not isinstance(message, ResponseOutputMessage):
            return ""

        last_content = message.content[-1]
        if isinstance(last_content, ResponseOutputText):
            return last_content.text
        elif isinstance(last_content, ResponseOutputRefusal):
            return last_content.refusal
        else:
            raise ModelBehaviorError(f"Unexpected content type: {type(last_content)}")

    @classmethod
    def extract_last_text(cls, message: TResponseOutputItem) -> str | None:
        """Extracts the last text content from a message, if any. Ignores refusals."""
        if isinstance(message, ResponseOutputMessage):
            last_content = message.content[-1]
            if isinstance(last_content, ResponseOutputText):
                return last_content.text

        return None

    @classmethod
    def input_to_new_input_list(
        cls, input: str | list[TResponseInputItem]
    ) -> list[TResponseInputItem]:
        """Converts a string or list of input items into a list of input items."""
        if isinstance(input, str):
            return [
                {
                    "content": input,
                    "role": "user",
                }
            ]
        return copy.deepcopy(input)

    @classmethod
    def text_message_outputs(cls, items: list[RunItem]) -> str:
        """Concatenates all the text content from a list of message output items."""
        text = ""
        for item in items:
            if isinstance(item, MessageOutputItem):
                text += cls.text_message_output(item)
        return text

    @classmethod
    def text_message_output(cls, message: MessageOutputItem) -> str:
        """Extracts all the text content from a single message output item."""
        text = ""
        for item in message.raw_item.content:
            if isinstance(item, ResponseOutputText):
                text += item.text
        return text

    @classmethod
    def tool_call_output_item(
        cls, tool_call: ResponseFunctionToolCall, output: str
    ) -> FunctionCallOutput:
        """Creates a tool call output item from a tool call and its output."""
        return {
            "call_id": tool_call.call_id,
            "output": output,
            "type": "function_call_output",
        }


<file_info>
path: tests/test_tool_use_behavior.py
name: test_tool_use_behavior.py
</file_info>
# Copyright

from __future__ import annotations

from typing import cast

import pytest
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agents import (
    Agent,
    FunctionToolResult,
    RunConfig,
    RunContextWrapper,
    ToolCallOutputItem,
    ToolsToFinalOutputResult,
    UserError,
)
from agents._run_impl import RunImpl

from .test_responses import get_function_tool


def _make_function_tool_result(
    agent: Agent, output: str, tool_name: str | None = None
) -> FunctionToolResult:
    # Construct a FunctionToolResult with the given output using a simple function tool.
    tool = get_function_tool(tool_name or "dummy", return_value=output)
    raw_item: FunctionCallOutput = cast(
        FunctionCallOutput,
        {
            "call_id": "1",
            "output": output,
            "type": "function_call_output",
        },
    )
    # For this test we don't care about the specific RunItem subclass, only the output field
    run_item = ToolCallOutputItem(agent=agent, raw_item=raw_item, output=output)
    return FunctionToolResult(tool=tool, output=output, run_item=run_item)


@pytest.mark.asyncio
async def test_no_tool_results_returns_not_final_output() -> None:
    # If there are no tool results at all, tool_use_behavior should not produce a final output.
    agent = Agent(name="test")
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=[],
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None


@pytest.mark.asyncio
async def test_run_llm_again_behavior() -> None:
    # With the default run_llm_again behavior, even with tools we still expect to keep running.
    agent = Agent(name="test", tool_use_behavior="run_llm_again")
    tool_results = [_make_function_tool_result(agent, "ignored")]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None


@pytest.mark.asyncio
async def test_stop_on_first_tool_behavior() -> None:
    # When tool_use_behavior is stop_on_first_tool, we should surface first tool output as final.
    agent = Agent(name="test", tool_use_behavior="stop_on_first_tool")
    tool_results = [
        _make_function_tool_result(agent, "first_tool_output"),
        _make_function_tool_result(agent, "ignored"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "first_tool_output"


@pytest.mark.asyncio
async def test_custom_tool_use_behavior_sync() -> None:
    """If tool_use_behavior is a sync function, we should call it and propagate its return."""

    def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "custom"


@pytest.mark.asyncio
async def test_custom_tool_use_behavior_async() -> None:
    """If tool_use_behavior is an async function, we should await it and propagate its return."""

    async def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="async_custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "async_custom"


@pytest.mark.asyncio
async def test_invalid_tool_use_behavior_raises() -> None:
    """If tool_use_behavior is invalid, we should raise a UserError."""
    agent = Agent(name="test")
    # Force an invalid value; mypy will complain, so ignore the type here.
    agent.tool_use_behavior = "bad_value"  # type: ignore[assignment]
    tool_results = [_make_function_tool_result(agent, "ignored")]
    with pytest.raises(UserError):
        await RunImpl._check_for_final_output_from_tools(
            agent=agent,
            tool_results=tool_results,
            context_wrapper=RunContextWrapper(context=None),
            config=RunConfig(),
        )


@pytest.mark.asyncio
async def test_tool_names_to_stop_at_behavior() -> None:
    agent = Agent(
        name="test",
        tools=[
            get_function_tool("tool1", return_value="tool1_output"),
            get_function_tool("tool2", return_value="tool2_output"),
            get_function_tool("tool3", return_value="tool3_output"),
        ],
        tool_use_behavior={"stop_at_tool_names": ["tool1"]},
    )

    tool_results = [
        _make_function_tool_result(agent, "ignored1", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is False, "We should not have stopped at tool1"

    # Now test with a tool that matches the list
    tool_results = [
        _make_function_tool_result(agent, "output1", "tool1"),
        _make_function_tool_result(agent, "ignored2", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True, "We should have stopped at tool1"
    assert result.final_output == "output1"


<file_info>
path: tests/voice/__init__.py
name: __init__.py
</file_info>


<file_info>
path: tests/voice/test_workflow.py
name: test_workflow.py
</file_info>
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from inline_snapshot import snapshot
from openai.types.responses import ResponseCompletedEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from agents import Agent, Model, ModelSettings, ModelTracing, Tool
from agents.agent_output import AgentOutputSchema
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)

try:
    from agents.voice import SingleAgentVoiceWorkflow

    from ..fake_model import get_response_obj
    from ..test_responses import get_function_tool, get_function_tool_call, get_text_message
except ImportError:
    pass


class FakeStreamingModel(Model):
    def __init__(self):
        self.turn_outputs: list[list[TResponseOutputItem]] = []

    def set_next_output(self, output: list[TResponseOutputItem]):
        self.turn_outputs.append(output)

    def add_multiple_turn_outputs(self, outputs: list[list[TResponseOutputItem]]):
        self.turn_outputs.extend(outputs)

    def get_next_output(self) -> list[TResponseOutputItem]:
        if not self.turn_outputs:
            return []
        return self.turn_outputs.pop(0)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        raise NotImplementedError("Not implemented")

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        output = self.get_next_output()
        for item in output:
            if (
                item.type == "message"
                and len(item.content) == 1
                and item.content[0].type == "output_text"
            ):
                yield ResponseTextDeltaEvent(
                    content_index=0,
                    delta=item.content[0].text,
                    type="response.output_text.delta",
                    output_index=0,
                    item_id=item.id,
                )

        yield ResponseCompletedEvent(
            type="response.completed",
            response=get_response_obj(output),
        )


@pytest.mark.asyncio
async def test_single_agent_workflow(monkeypatch) -> None:
    model = FakeStreamingModel()
    model.add_multiple_turn_outputs(
        [
            # First turn: a message and a tool call
            [
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_text_message("a_message"),
            ],
            # Second turn: text message
            [get_text_message("done")],
        ]
    )

    agent = Agent(
        "initial_agent",
        model=model,
        tools=[get_function_tool("some_function", "tool_result")],
    )

    workflow = SingleAgentVoiceWorkflow(agent)
    output = []
    async for chunk in workflow.run("transcription_1"):
        output.append(chunk)

    # Validate that the text yielded matches our fake events
    assert output == ["a_message", "done"]
    # Validate that internal state was updated
    assert workflow._input_history == snapshot(
        [
            {"content": "transcription_1", "role": "user"},
            {
                "arguments": '{"a": "b"}',
                "call_id": "2",
                "name": "some_function",
                "type": "function_call",
                "id": "1",
            },
            {
                "id": "1",
                "content": [{"annotations": [], "text": "a_message", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"call_id": "2", "output": "tool_result", "type": "function_call_output"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
    )
    assert workflow._current_agent == agent

    model.set_next_output([get_text_message("done_2")])

    # Run it again with a new transcription to make sure the input history is updated
    output = []
    async for chunk in workflow.run("transcription_2"):
        output.append(chunk)

    assert workflow._input_history == snapshot(
        [
            {"role": "user", "content": "transcription_1"},
            {
                "arguments": '{"a": "b"}',
                "call_id": "2",
                "name": "some_function",
                "type": "function_call",
                "id": "1",
            },
            {
                "id": "1",
                "content": [{"annotations": [], "text": "a_message", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"call_id": "2", "output": "tool_result", "type": "function_call_output"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"role": "user", "content": "transcription_2"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done_2", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
    )
    assert workflow._current_agent == agent


<file_info>
path: tests/test_openai_chatcompletions.py
name: test_openai_chatcompletions.py
</file_info>
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import NOT_GIVEN
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from agents import (
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    OpenAIProvider,
    generation_span,
)
from agents.models.fake_id import FAKE_RESPONSES_ID


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_text_message(monkeypatch) -> None:
    """
    When the model returns a ChatCompletionMessage with plain text content,
    `get_response` should produce a single `ResponseOutputMessage` containing
    a `ResponseOutputText` with that content, and a `Usage` populated from
    the completion's usage.
    """
    msg = ChatCompletionMessage(role="assistant", content="Hello")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=7, total_tokens=12),
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )
    # Should have produced exactly one output message with one text part
    assert isinstance(resp, ModelResponse)
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], ResponseOutputMessage)
    msg_item = resp.output[0]
    assert len(msg_item.content) == 1
    assert isinstance(msg_item.content[0], ResponseOutputText)
    assert msg_item.content[0].text == "Hello"
    # Usage should be preserved from underlying ChatCompletion.usage
    assert resp.usage.input_tokens == 7
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 12
    assert resp.referenceable_id is None


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_refusal(monkeypatch) -> None:
    """
    When the model returns a ChatCompletionMessage with a `refusal` instead
    of normal `content`, `get_response` should produce a single
    `ResponseOutputMessage` containing a `ResponseOutputRefusal` part.
    """
    msg = ChatCompletionMessage(role="assistant", refusal="No thanks")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=None,
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], ResponseOutputMessage)
    refusal_part = resp.output[0].content[0]
    assert isinstance(refusal_part, ResponseOutputRefusal)
    assert refusal_part.refusal == "No thanks"
    # With no usage from the completion, usage defaults to zeros.
    assert resp.usage.requests == 0
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_tool_call(monkeypatch) -> None:
    """
    If the ChatCompletionMessage includes one or more tool_calls, `get_response`
    should append corresponding `ResponseFunctionToolCall` items after the
    assistant message item with matching name/arguments.
    """
    tool_call = ChatCompletionMessageToolCall(
        id="call-id",
        type="function",
        function=Function(name="do_thing", arguments="{'x':1}"),
    )
    msg = ChatCompletionMessage(role="assistant", content="Hi", tool_calls=[tool_call])
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=None,
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )
    # Expect a message item followed by a function tool call item.
    assert len(resp.output) == 2
    assert isinstance(resp.output[0], ResponseOutputMessage)
    fn_call_item = resp.output[1]
    assert isinstance(fn_call_item, ResponseFunctionToolCall)
    assert fn_call_item.call_id == "call-id"
    assert fn_call_item.name == "do_thing"
    assert fn_call_item.arguments == "{'x':1}"


@pytest.mark.asyncio
async def test_fetch_response_non_stream(monkeypatch) -> None:
    """
    Verify that `_fetch_response` builds the correct OpenAI API call when not
    streaming and returns the ChatCompletion object directly. We supply a
    dummy ChatCompletion through a stubbed OpenAI client and inspect the
    captured kwargs.
    """

    # Dummy completions to record kwargs
    class DummyCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return chat

    class DummyClient:
        def __init__(self, completions: DummyCompletions) -> None:
            self.chat = type("_Chat", (), {"completions": completions})()
            self.base_url = httpx.URL("http://fake")

    msg = ChatCompletionMessage(role="assistant", content="ignored")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
    )
    completions = DummyCompletions()
    dummy_client = DummyClient(completions)
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=dummy_client)  # type: ignore
    # Execute the private fetch with a system instruction and simple string input.
    with generation_span(disabled=True) as span:
        result = await model._fetch_response(
            system_instructions="sys",
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            span=span,
            tracing=ModelTracing.DISABLED,
            stream=False,
        )
    assert result is chat
    # Ensure expected args were passed through to OpenAI client.
    kwargs = completions.kwargs
    assert kwargs["stream"] is False
    assert kwargs["model"] == "gpt-4"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][0]["content"] == "sys"
    assert kwargs["messages"][1]["role"] == "user"
    # Defaults for optional fields become the NOT_GIVEN sentinel
    assert kwargs["tools"] is NOT_GIVEN
    assert kwargs["tool_choice"] is NOT_GIVEN
    assert kwargs["response_format"] is NOT_GIVEN
    assert kwargs["stream_options"] is NOT_GIVEN


@pytest.mark.asyncio
async def test_fetch_response_stream(monkeypatch) -> None:
    """
    When `stream=True`, `_fetch_response` should return a bare `Response`
    object along with the underlying async stream. The OpenAI client call
    should include `stream_options` to request usage-delimited chunks.
    """

    async def event_stream() -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield  # pragma: no cover

    class DummyCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return event_stream()

    class DummyClient:
        def __init__(self, completions: DummyCompletions) -> None:
            self.chat = type("_Chat", (), {"completions": completions})()
            self.base_url = httpx.URL("http://fake")

    completions = DummyCompletions()
    dummy_client = DummyClient(completions)
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=dummy_client)  # type: ignore
    with generation_span(disabled=True) as span:
        response, stream = await model._fetch_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            span=span,
            tracing=ModelTracing.DISABLED,
            stream=True,
        )
    # Check OpenAI client was called for streaming
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["stream_options"] == {"include_usage": True}
    # Response is a proper openai Response
    assert isinstance(response, Response)
    assert response.id == FAKE_RESPONSES_ID
    assert response.model == "gpt-4"
    assert response.object == "response"
    assert response.output == []
    # We returned the async iterator produced by our dummy.
    assert hasattr(stream, "__aiter__")


<file_info>
path: examples/basic/hello_world.py
name: hello_world.py
</file_info>
import asyncio

from agents import Agent, Runner


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/basic/agent_lifecycle_example.py
name: agent_lifecycle_example.py
</file_info>
import asyncio
import random
from typing import Any

from pydantic import BaseModel

from agents import Agent, AgentHooks, RunContextWrapper, Runner, Tool, function_tool


class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


###


@function_tool
def random_number(max: int) -> int:
    """
    Generate a random number up to the provided maximum.
    """
    return random.randint(0, max)


@function_tool
def multiply_by_two(x: int) -> int:
    """Simple multiplication by two."""
    return x * 2


class FinalResult(BaseModel):
    number: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="Multiply the number by 2 and then return the final result.",
    tools=[multiply_by_two],
    output_type=FinalResult,
    hooks=CustomAgentHooks(display_name="Multiply Agent"),
)

start_agent = Agent(
    name="Start Agent",
    instructions="Generate a random number. If it's even, stop. If it's odd, hand off to the multiply agent.",
    tools=[random_number],
    output_type=FinalResult,
    handoffs=[multiply_agent],
    hooks=CustomAgentHooks(display_name="Start Agent"),
)


async def main() -> None:
    user_input = input("Enter a max number: ")
    await Runner.run(
        start_agent,
        input=f"Generate a random number between 0 and {user_input}.",
    )

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
"""
$ python examples/basic/agent_lifecycle_example.py

Enter a max number: 250
### (Start Agent) 1: Agent Start Agent started
### (Start Agent) 2: Agent Start Agent started tool random_number
### (Start Agent) 3: Agent Start Agent ended tool random_number with result 37
### (Start Agent) 4: Agent Start Agent started
### (Start Agent) 5: Agent Start Agent handed off to Multiply Agent
### (Multiply Agent) 1: Agent Multiply Agent started
### (Multiply Agent) 2: Agent Multiply Agent started tool multiply_by_two
### (Multiply Agent) 3: Agent Multiply Agent ended tool multiply_by_two with result 74
### (Multiply Agent) 4: Agent Multiply Agent started
### (Multiply Agent) 5: Agent Multiply Agent ended with output number=74
Done!
"""


<file_info>
path: examples/voice/static/util.py
name: util.py
</file_info>
import curses
import time

import numpy as np
import numpy.typing as npt
import sounddevice as sd


def _record_audio(screen: curses.window) -> npt.NDArray[np.float32]:
    screen.nodelay(True)  # Non-blocking input
    screen.clear()
    screen.addstr(
        "Press <spacebar> to start recording. Press <spacebar> again to stop recording.\n"
    )
    screen.refresh()

    recording = False
    audio_buffer: list[npt.NDArray[np.float32]] = []

    def _audio_callback(indata, frames, time_info, status):
        if status:
            screen.addstr(f"Status: {status}\n")
            screen.refresh()
        if recording:
            audio_buffer.append(indata.copy())

    # Open the audio stream with the callback.
    with sd.InputStream(samplerate=24000, channels=1, dtype=np.float32, callback=_audio_callback):
        while True:
            key = screen.getch()
            if key == ord(" "):
                recording = not recording
                if recording:
                    screen.addstr("Recording started...\n")
                else:
                    screen.addstr("Recording stopped.\n")
                    break
                screen.refresh()
            time.sleep(0.01)

    # Combine recorded audio chunks.
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
    else:
        audio_data = np.empty((0,), dtype=np.float32)

    return audio_data


def record_audio():
    # Using curses to record audio in a way that:
    # - doesn't require accessibility permissions on macos
    # - doesn't block the terminal
    audio_data = curses.wrapper(_record_audio)
    return audio_data


class AudioPlayer:
    def __enter__(self):
        self.stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.close()

    def add_audio(self, audio_data: npt.NDArray[np.int16]):
        self.stream.write(audio_data)


<file_info>
path: src/agents/tracing/__init__.py
name: __init__.py
</file_info>
import atexit

from .create import (
    agent_span,
    custom_span,
    function_span,
    generation_span,
    get_current_span,
    get_current_trace,
    guardrail_span,
    handoff_span,
    response_span,
    speech_group_span,
    speech_span,
    trace,
    transcription_span,
)
from .processor_interface import TracingProcessor
from .processors import default_exporter, default_processor
from .setup import GLOBAL_TRACE_PROVIDER
from .span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
    SpeechGroupSpanData,
    SpeechSpanData,
    TranscriptionSpanData,
)
from .spans import Span, SpanError
from .traces import Trace
from .util import gen_span_id, gen_trace_id

__all__ = [
    "add_trace_processor",
    "agent_span",
    "custom_span",
    "function_span",
    "generation_span",
    "get_current_span",
    "get_current_trace",
    "guardrail_span",
    "handoff_span",
    "response_span",
    "set_trace_processors",
    "set_tracing_disabled",
    "trace",
    "Trace",
    "SpanError",
    "Span",
    "SpanData",
    "AgentSpanData",
    "CustomSpanData",
    "FunctionSpanData",
    "GenerationSpanData",
    "GuardrailSpanData",
    "HandoffSpanData",
    "ResponseSpanData",
    "SpeechGroupSpanData",
    "SpeechSpanData",
    "TranscriptionSpanData",
    "TracingProcessor",
    "gen_trace_id",
    "gen_span_id",
    "speech_group_span",
    "speech_span",
    "transcription_span",
]


def add_trace_processor(span_processor: TracingProcessor) -> None:
    """
    Adds a new trace processor. This processor will receive all traces/spans.
    """
    GLOBAL_TRACE_PROVIDER.register_processor(span_processor)


def set_trace_processors(processors: list[TracingProcessor]) -> None:
    """
    Set the list of trace processors. This will replace the current list of processors.
    """
    GLOBAL_TRACE_PROVIDER.set_processors(processors)


def set_tracing_disabled(disabled: bool) -> None:
    """
    Set whether tracing is globally disabled.
    """
    GLOBAL_TRACE_PROVIDER.set_disabled(disabled)


def set_tracing_export_api_key(api_key: str) -> None:
    """
    Set the OpenAI API key for the backend exporter.
    """
    default_exporter().set_api_key(api_key)


# Add the default processor, which exports traces and spans to the backend in batches. You can
# change the default behavior by either:
# 1. calling add_trace_processor(), which adds additional processors, or
# 2. calling set_trace_processors(), which replaces the default processor.
add_trace_processor(default_processor())

atexit.register(GLOBAL_TRACE_PROVIDER.shutdown)


<file_info>
path: src/agents/models/openai_chatcompletions.py
name: openai_chatcompletions.py
</file_info>
from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, NotGiven
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFileSearchToolCallParam,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_input_param import FunctionCallOutput, ItemReference, Message
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from .. import _debug
from ..agent_output import AgentOutputSchema
from ..exceptions import AgentsException, UserError
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from ..logger import logger
from ..tool import FunctionTool, Tool
from ..tracing import generation_span
from ..tracing.span_data import GenerationSpanData
from ..tracing.spans import Span
from ..usage import Usage
from ..version import __version__
from .fake_id import FAKE_RESPONSES_ID
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}


@dataclass
class _StreamingState:
    started: bool = False
    text_content_index_and_output: tuple[int, ResponseOutputText] | None = None
    refusal_content_index_and_output: tuple[int, ResponseOutputRefusal] | None = None
    function_calls: dict[int, ResponseFunctionToolCall] = field(default_factory=dict)


class OpenAIChatCompletionsModel(Model):
    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else NOT_GIVEN

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=dataclasses.asdict(model_settings)
            | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
            )

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                logger.debug(
                    f"LLM resp:\n{json.dumps(response.choices[0].message.model_dump(), indent=2)}\n"
                )

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else Usage()
            )
            if tracing.include_data():
                span_generation.span_data.output = [response.choices[0].message.model_dump()]
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = _Converter.message_to_output_items(response.choices[0].message)

            return ModelResponse(
                output=items,
                usage=usage,
                referenceable_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        Yields a partial message as it is generated, as well as the usage information.
        """
        with generation_span(
            model=str(self.model),
            model_config=dataclasses.asdict(model_settings)
            | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response, stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
            )

            usage: CompletionUsage | None = None
            state = _StreamingState()

            async for chunk in stream:
                if not state.started:
                    state.started = True
                    yield ResponseCreatedEvent(
                        response=response,
                        type="response.created",
                    )

                # The usage is only available in the last chunk
                usage = chunk.usage

                if not chunk.choices or not chunk.choices[0].delta:
                    continue

                delta = chunk.choices[0].delta

                # Handle text
                if delta.content:
                    if not state.text_content_index_and_output:
                        # Initialize a content tracker for streaming text
                        state.text_content_index_and_output = (
                            0 if not state.refusal_content_index_and_output else 1,
                            ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                        )
                        # Start a new assistant message stream
                        assistant_item = ResponseOutputMessage(
                            id=FAKE_RESPONSES_ID,
                            content=[],
                            role="assistant",
                            type="message",
                            status="in_progress",
                        )
                        # Notify consumers of the start of a new output message + first content part
                        yield ResponseOutputItemAddedEvent(
                            item=assistant_item,
                            output_index=0,
                            type="response.output_item.added",
                        )
                        yield ResponseContentPartAddedEvent(
                            content_index=state.text_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            part=ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                            type="response.content_part.added",
                        )
                    # Emit the delta for this segment of content
                    yield ResponseTextDeltaEvent(
                        content_index=state.text_content_index_and_output[0],
                        delta=delta.content,
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        type="response.output_text.delta",
                    )
                    # Accumulate the text into the response part
                    state.text_content_index_and_output[1].text += delta.content

                # Handle refusals (model declines to answer)
                if delta.refusal:
                    if not state.refusal_content_index_and_output:
                        # Initialize a content tracker for streaming refusal text
                        state.refusal_content_index_and_output = (
                            0 if not state.text_content_index_and_output else 1,
                            ResponseOutputRefusal(refusal="", type="refusal"),
                        )
                        # Start a new assistant message if one doesn't exist yet (in-progress)
                        assistant_item = ResponseOutputMessage(
                            id=FAKE_RESPONSES_ID,
                            content=[],
                            role="assistant",
                            type="message",
                            status="in_progress",
                        )
                        # Notify downstream that assistant message + first content part are starting
                        yield ResponseOutputItemAddedEvent(
                            item=assistant_item,
                            output_index=0,
                            type="response.output_item.added",
                        )
                        yield ResponseContentPartAddedEvent(
                            content_index=state.refusal_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            part=ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                            type="response.content_part.added",
                        )
                    # Emit the delta for this segment of refusal
                    yield ResponseRefusalDeltaEvent(
                        content_index=state.refusal_content_index_and_output[0],
                        delta=delta.refusal,
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        type="response.refusal.delta",
                    )
                    # Accumulate the refusal string in the output part
                    state.refusal_content_index_and_output[1].refusal += delta.refusal

                # Handle tool calls
                # Because we don't know the name of the function until the end of the stream, we'll
                # save everything and yield events at the end
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index not in state.function_calls:
                            state.function_calls[tc_delta.index] = ResponseFunctionToolCall(
                                id=FAKE_RESPONSES_ID,
                                arguments="",
                                name="",
                                type="function_call",
                                call_id="",
                            )
                        tc_function = tc_delta.function

                        state.function_calls[tc_delta.index].arguments += (
                            tc_function.arguments if tc_function else ""
                        ) or ""
                        state.function_calls[tc_delta.index].name += (
                            tc_function.name if tc_function else ""
                        ) or ""
                        state.function_calls[tc_delta.index].call_id += tc_delta.id or ""

            function_call_starting_index = 0
            if state.text_content_index_and_output:
                function_call_starting_index += 1
                # Send end event for this content part
                yield ResponseContentPartDoneEvent(
                    content_index=state.text_content_index_and_output[0],
                    item_id=FAKE_RESPONSES_ID,
                    output_index=0,
                    part=state.text_content_index_and_output[1],
                    type="response.content_part.done",
                )

            if state.refusal_content_index_and_output:
                function_call_starting_index += 1
                # Send end event for this content part
                yield ResponseContentPartDoneEvent(
                    content_index=state.refusal_content_index_and_output[0],
                    item_id=FAKE_RESPONSES_ID,
                    output_index=0,
                    part=state.refusal_content_index_and_output[1],
                    type="response.content_part.done",
                )

            # Actually send events for the function calls
            for function_call in state.function_calls.values():
                # First, a ResponseOutputItemAdded for the function call
                yield ResponseOutputItemAddedEvent(
                    item=ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=function_call.call_id,
                        arguments=function_call.arguments,
                        name=function_call.name,
                        type="function_call",
                    ),
                    output_index=function_call_starting_index,
                    type="response.output_item.added",
                )
                # Then, yield the args
                yield ResponseFunctionCallArgumentsDeltaEvent(
                    delta=function_call.arguments,
                    item_id=FAKE_RESPONSES_ID,
                    output_index=function_call_starting_index,
                    type="response.function_call_arguments.delta",
                )
                # Finally, the ResponseOutputItemDone
                yield ResponseOutputItemDoneEvent(
                    item=ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=function_call.call_id,
                        arguments=function_call.arguments,
                        name=function_call.name,
                        type="function_call",
                    ),
                    output_index=function_call_starting_index,
                    type="response.output_item.done",
                )

            # Finally, send the Response completed event
            outputs: list[ResponseOutputItem] = []
            if state.text_content_index_and_output or state.refusal_content_index_and_output:
                assistant_msg = ResponseOutputMessage(
                    id=FAKE_RESPONSES_ID,
                    content=[],
                    role="assistant",
                    type="message",
                    status="completed",
                )
                if state.text_content_index_and_output:
                    assistant_msg.content.append(state.text_content_index_and_output[1])
                if state.refusal_content_index_and_output:
                    assistant_msg.content.append(state.refusal_content_index_and_output[1])
                outputs.append(assistant_msg)

                # send a ResponseOutputItemDone for the assistant message
                yield ResponseOutputItemDoneEvent(
                    item=assistant_msg,
                    output_index=0,
                    type="response.output_item.done",
                )

            for function_call in state.function_calls.values():
                outputs.append(function_call)

            final_response = response.model_copy()
            final_response.output = outputs
            final_response.usage = (
                ResponseUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                        if usage.completion_tokens_details
                        and usage.completion_tokens_details.reasoning_tokens
                        else 0
                    ),
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=usage.prompt_tokens_details.cached_tokens
                        if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens
                        else 0
                    ),
                )
                if usage
                else None
            )

            yield ResponseCompletedEvent(
                response=final_response,
                type="response.completed",
            )
            if tracing.include_data():
                span_generation.span_data.output = [final_response.model_dump()]

            if usage:
                span_generation.span_data.usage = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
    ) -> ChatCompletion: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        converted_messages = _Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True if model_settings.parallel_tool_calls and tools and len(tools) > 0 else NOT_GIVEN
        )
        tool_choice = _Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = _Converter.convert_response_format(output_schema)

        converted_tools = [ToolConverter.to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(ToolConverter.convert_handoff_tool(handoff))

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"{json.dumps(converted_messages, indent=2)}\n"
                f"Tools:\n{json.dumps(converted_tools, indent=2)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        ret = await self._get_client().chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools or NOT_GIVEN,
            temperature=self._non_null_or_not_given(model_settings.temperature),
            top_p=self._non_null_or_not_given(model_settings.top_p),
            frequency_penalty=self._non_null_or_not_given(model_settings.frequency_penalty),
            presence_penalty=self._non_null_or_not_given(model_settings.presence_penalty),
            max_tokens=self._non_null_or_not_given(model_settings.max_tokens),
            tool_choice=tool_choice,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            stream_options={"include_usage": True} if stream else NOT_GIVEN,
            extra_headers=_HEADERS,
        )

        if isinstance(ret, ChatCompletion):
            return ret

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
            if tool_choice != NOT_GIVEN
            else "auto",
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
        )
        return response, ret

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client


class _Converter:
    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> ChatCompletionToolChoiceOptionParam | NotGiven:
        if tool_choice is None:
            return NOT_GIVEN
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            return {
                "type": "function",
                "function": {
                    "name": tool_choice,
                },
            }

    @classmethod
    def convert_response_format(
        cls, final_output_schema: AgentOutputSchema | None
    ) -> ResponseFormat | NotGiven:
        if not final_output_schema or final_output_schema.is_plain_text():
            return NOT_GIVEN

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "strict": final_output_schema.strict_json_schema,
                "schema": final_output_schema.json_schema(),
            },
        }

    @classmethod
    def message_to_output_items(cls, message: ChatCompletionMessage) -> list[TResponseOutputItem]:
        items: list[TResponseOutputItem] = []

        message_item = ResponseOutputMessage(
            id=FAKE_RESPONSES_ID,
            content=[],
            role="assistant",
            type="message",
            status="completed",
        )
        if message.content:
            message_item.content.append(
                ResponseOutputText(text=message.content, type="output_text", annotations=[])
            )
        if message.refusal:
            message_item.content.append(
                ResponseOutputRefusal(refusal=message.refusal, type="refusal")
            )
        if message.audio:
            raise AgentsException("Audio is not currently supported")

        if message_item.content:
            items.append(message_item)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                items.append(
                    ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=tool_call.id,
                        arguments=tool_call.function.arguments,
                        name=tool_call.function.name,
                        type="function_call",
                    )
                )

        return items

    @classmethod
    def maybe_easy_input_message(cls, item: Any) -> EasyInputMessageParam | None:
        if not isinstance(item, dict):
            return None

        keys = item.keys()
        # EasyInputMessageParam only has these two keys
        if keys != {"content", "role"}:
            return None

        role = item.get("role", None)
        if role not in ("user", "assistant", "system", "developer"):
            return None

        if "content" not in item:
            return None

        return cast(EasyInputMessageParam, item)

    @classmethod
    def maybe_input_message(cls, item: Any) -> Message | None:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role")
            in (
                "user",
                "system",
                "developer",
            )
        ):
            return cast(Message, item)

        return None

    @classmethod
    def maybe_file_search_call(cls, item: Any) -> ResponseFileSearchToolCallParam | None:
        if isinstance(item, dict) and item.get("type") == "file_search_call":
            return cast(ResponseFileSearchToolCallParam, item)
        return None

    @classmethod
    def maybe_function_tool_call(cls, item: Any) -> ResponseFunctionToolCallParam | None:
        if isinstance(item, dict) and item.get("type") == "function_call":
            return cast(ResponseFunctionToolCallParam, item)
        return None

    @classmethod
    def maybe_function_tool_call_output(
        cls,
        item: Any,
    ) -> FunctionCallOutput | None:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return cast(FunctionCallOutput, item)
        return None

    @classmethod
    def maybe_item_reference(cls, item: Any) -> ItemReference | None:
        if isinstance(item, dict) and item.get("type") == "item_reference":
            return cast(ItemReference, item)
        return None

    @classmethod
    def maybe_response_output_message(cls, item: Any) -> ResponseOutputMessageParam | None:
        # ResponseOutputMessage is only used for messages with role assistant
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        ):
            return cast(ResponseOutputMessageParam, item)
        return None

    @classmethod
    def extract_text_content(
        cls, content: str | Iterable[ResponseInputContentParam]
    ) -> str | list[ChatCompletionContentPartTextParam]:
        all_content = cls.extract_all_content(content)
        if isinstance(all_content, str):
            return all_content
        out: list[ChatCompletionContentPartTextParam] = []
        for c in all_content:
            if c.get("type") == "text":
                out.append(cast(ChatCompletionContentPartTextParam, c))
        return out

    @classmethod
    def extract_all_content(
        cls, content: str | Iterable[ResponseInputContentParam]
    ) -> str | list[ChatCompletionContentPartParam]:
        if isinstance(content, str):
            return content
        out: list[ChatCompletionContentPartParam] = []

        for c in content:
            if isinstance(c, dict) and c.get("type") == "input_text":
                casted_text_param = cast(ResponseInputTextParam, c)
                out.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=casted_text_param["text"],
                    )
                )
            elif isinstance(c, dict) and c.get("type") == "input_image":
                casted_image_param = cast(ResponseInputImageParam, c)
                if "image_url" not in casted_image_param or not casted_image_param["image_url"]:
                    raise UserError(
                        f"Only image URLs are supported for input_image {casted_image_param}"
                    )
                out.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={
                            "url": casted_image_param["image_url"],
                            "detail": casted_image_param["detail"],
                        },
                    )
                )
            elif isinstance(c, dict) and c.get("type") == "input_file":
                raise UserError(f"File uploads are not supported for chat completions {c}")
            else:
                raise UserError(f"Unknown content: {c}")
        return out

    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[TResponseInputItem],
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert a sequence of 'Item' objects into a list of ChatCompletionMessageParam.

        Rules:
        - EasyInputMessage or InputMessage (role=user) => ChatCompletionUserMessageParam
        - EasyInputMessage or InputMessage (role=system) => ChatCompletionSystemMessageParam
        - EasyInputMessage or InputMessage (role=developer) => ChatCompletionDeveloperMessageParam
        - InputMessage (role=assistant) => Start or flush a ChatCompletionAssistantMessageParam
        - response_output_message => Also produces/flushes a ChatCompletionAssistantMessageParam
        - tool calls get attached to the *current* assistant message, or create one if none.
        - tool outputs => ChatCompletionToolMessageParam
        """

        if isinstance(items, str):
            return [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=items,
                )
            ]

        result: list[ChatCompletionMessageParam] = []
        current_assistant_msg: ChatCompletionAssistantMessageParam | None = None

        def flush_assistant_message() -> None:
            nonlocal current_assistant_msg
            if current_assistant_msg is not None:
                # The API doesn't support empty arrays for tool_calls
                if not current_assistant_msg.get("tool_calls"):
                    del current_assistant_msg["tool_calls"]
                result.append(current_assistant_msg)
                current_assistant_msg = None

        def ensure_assistant_message() -> ChatCompletionAssistantMessageParam:
            nonlocal current_assistant_msg
            if current_assistant_msg is None:
                current_assistant_msg = ChatCompletionAssistantMessageParam(role="assistant")
                current_assistant_msg["tool_calls"] = []
            return current_assistant_msg

        for item in items:
            # 1) Check easy input message
            if easy_msg := cls.maybe_easy_input_message(item):
                role = easy_msg["role"]
                content = easy_msg["content"]

                if role == "user":
                    flush_assistant_message()
                    msg_user: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    flush_assistant_message()
                    msg_system: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    flush_assistant_message()
                    msg_developer: ChatCompletionDeveloperMessageParam = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                elif role == "assistant":
                    flush_assistant_message()
                    msg_assistant: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_assistant)
                else:
                    raise UserError(f"Unexpected role in easy_input_message: {role}")

            # 2) Check input message
            elif in_msg := cls.maybe_input_message(item):
                role = in_msg["role"]
                content = in_msg["content"]
                flush_assistant_message()

                if role == "user":
                    msg_user = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    msg_system = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    msg_developer = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                else:
                    raise UserError(f"Unexpected role in input_message: {role}")

            # 3) response output message => assistant
            elif resp_msg := cls.maybe_response_output_message(item):
                flush_assistant_message()
                new_asst = ChatCompletionAssistantMessageParam(role="assistant")
                contents = resp_msg["content"]

                text_segments = []
                for c in contents:
                    if c["type"] == "output_text":
                        text_segments.append(c["text"])
                    elif c["type"] == "refusal":
                        new_asst["refusal"] = c["refusal"]
                    elif c["type"] == "output_audio":
                        # Can't handle this, b/c chat completions expects an ID which we dont have
                        raise UserError(
                            f"Only audio IDs are supported for chat completions, but got: {c}"
                        )
                    else:
                        raise UserError(f"Unknown content type in ResponseOutputMessage: {c}")

                if text_segments:
                    combined = "\n".join(text_segments)
                    new_asst["content"] = combined

                new_asst["tool_calls"] = []
                current_assistant_msg = new_asst

            # 4) function/file-search calls => attach to assistant
            elif file_search := cls.maybe_file_search_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = ChatCompletionMessageToolCallParam(
                    id=file_search["id"],
                    type="function",
                    function={
                        "name": "file_search_call",
                        "arguments": json.dumps(
                            {
                                "queries": file_search.get("queries", []),
                                "status": file_search.get("status"),
                            }
                        ),
                    },
                )
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls

            elif func_call := cls.maybe_function_tool_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = ChatCompletionMessageToolCallParam(
                    id=func_call["call_id"],
                    type="function",
                    function={
                        "name": func_call["name"],
                        "arguments": func_call["arguments"],
                    },
                )
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls
            # 5) function call output => tool message
            elif func_output := cls.maybe_function_tool_call_output(item):
                flush_assistant_message()
                msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": func_output["call_id"],
                    "content": func_output["output"],
                }
                result.append(msg)

            # 6) item reference => handle or raise
            elif item_ref := cls.maybe_item_reference(item):
                raise UserError(
                    f"Encountered an item_reference, which is not supported: {item_ref}"
                )

            # 7) If we haven't recognized it => fail or ignore
            else:
                raise UserError(f"Unhandled item type or structure: {item}")

        flush_assistant_message()
        return result


class ToolConverter:
    @classmethod
    def to_openai(cls, tool: Tool) -> ChatCompletionToolParam:
        if isinstance(tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.params_json_schema,
                },
            }

        raise UserError(
            f"Hosted tools are not supported with the ChatCompletions API. FGot tool type: "
            f"{type(tool)}, tool: {tool}"
        )

    @classmethod
    def convert_handoff_tool(cls, handoff: Handoff[Any]) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": handoff.input_json_schema,
            },
        }


<file_info>
path: tests/test_run_config.py
name: test_run_config.py
</file_info>
from __future__ import annotations

import pytest

from agents import Agent, RunConfig, Runner
from agents.models.interface import Model, ModelProvider

from .fake_model import FakeModel
from .test_responses import get_text_message


class DummyProvider(ModelProvider):
    """A simple model provider that always returns the same model, and
    records the model name it was asked to provide."""

    def __init__(self, model_to_return: Model | None = None) -> None:
        self.last_requested: str | None = None
        self.model_to_return: Model = model_to_return or FakeModel()

    def get_model(self, model_name: str | None) -> Model:
        # record the requested model name and return our test model
        self.last_requested = model_name
        return self.model_to_return


@pytest.mark.asyncio
async def test_model_provider_on_run_config_is_used_for_agent_model_name() -> None:
    """
    When the agent's ``model`` attribute is a string and no explicit model override is
    provided in the ``RunConfig``, the ``Runner`` should resolve the model using the
    ``model_provider`` on the ``RunConfig``.
    """
    fake_model = FakeModel(initial_output=[get_text_message("from-provider")])
    provider = DummyProvider(model_to_return=fake_model)
    agent = Agent(name="test", model="test-model")
    run_config = RunConfig(model_provider=provider)
    result = await Runner.run(agent, input="any", run_config=run_config)
    # We picked up the model from our dummy provider
    assert provider.last_requested == "test-model"
    assert result.final_output == "from-provider"


@pytest.mark.asyncio
async def test_run_config_model_name_override_takes_precedence() -> None:
    """
    When a model name string is set on the RunConfig, then that name should be looked up
    using the RunConfig's model_provider, and should override any model on the agent.
    """
    fake_model = FakeModel(initial_output=[get_text_message("override-name")])
    provider = DummyProvider(model_to_return=fake_model)
    agent = Agent(name="test", model="agent-model")
    run_config = RunConfig(model="override-name", model_provider=provider)
    result = await Runner.run(agent, input="any", run_config=run_config)
    # We should have requested the override name, not the agent.model
    assert provider.last_requested == "override-name"
    assert result.final_output == "override-name"


@pytest.mark.asyncio
async def test_run_config_model_override_object_takes_precedence() -> None:
    """
    When a concrete Model instance is set on the RunConfig, then that instance should be
    returned by Runner._get_model regardless of the agent's model.
    """
    fake_model = FakeModel(initial_output=[get_text_message("override-object")])
    agent = Agent(name="test", model="agent-model")
    run_config = RunConfig(model=fake_model)
    result = await Runner.run(agent, input="any", run_config=run_config)
    # Our FakeModel on the RunConfig should have been used.
    assert result.final_output == "override-object"


@pytest.mark.asyncio
async def test_agent_model_object_is_used_when_present() -> None:
    """
    If the agent has a concrete Model object set as its model, and the RunConfig does
    not specify a model override, then that object should be used directly without
    consulting the RunConfig's model_provider.
    """
    fake_model = FakeModel(initial_output=[get_text_message("from-agent-object")])
    provider = DummyProvider()
    agent = Agent(name="test", model=fake_model)
    run_config = RunConfig(model_provider=provider)
    result = await Runner.run(agent, input="any", run_config=run_config)
    # The dummy provider should never have been called, and the output should come from
    # the FakeModel on the agent.
    assert provider.last_requested is None
    assert result.final_output == "from-agent-object"


<file_info>
path: tests/test_openai_responses_converter.py
name: test_openai_responses_converter.py
</file_info>
# Copyright (c) OpenAI
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Unit tests for the `Converter` class defined in
`agents.models.openai_responses`. The converter is responsible for
translating various agent tool types and output schemas into the parameter
structures expected by the OpenAI Responses API.

We test the following aspects:

- `convert_tool_choice` correctly maps high-level tool choice strings into
  the tool choice values accepted by the Responses API, including special types
  like `file_search` and `web_search`, and falling back to function names
  for arbitrary string values.
- `get_response_format` returns `openai.NOT_GIVEN` for plain-text response
  formats and an appropriate format dict when a JSON-structured output schema
  is provided.
- `convert_tools` maps our internal `Tool` dataclasses into the appropriate
  request payloads and includes list, and enforces constraints like at most
  one `ComputerTool`.
"""

import pytest
from openai import NOT_GIVEN
from pydantic import BaseModel

from agents import (
    Agent,
    AgentOutputSchema,
    Computer,
    ComputerTool,
    FileSearchTool,
    Handoff,
    Tool,
    UserError,
    WebSearchTool,
    function_tool,
    handoff,
)
from agents.models.openai_responses import Converter


def test_convert_tool_choice_standard_values():
    """
    Make sure that the standard tool_choice values map to themselves or
    to "auto"/"required"/"none" as appropriate, and that special string
    values map to the appropriate dicts.
    """
    assert Converter.convert_tool_choice(None) is NOT_GIVEN
    assert Converter.convert_tool_choice("auto") == "auto"
    assert Converter.convert_tool_choice("required") == "required"
    assert Converter.convert_tool_choice("none") == "none"
    # Special tool types are represented as dicts of type only.
    assert Converter.convert_tool_choice("file_search") == {"type": "file_search"}
    assert Converter.convert_tool_choice("web_search_preview") == {"type": "web_search_preview"}
    assert Converter.convert_tool_choice("computer_use_preview") == {"type": "computer_use_preview"}
    # Arbitrary string should be interpreted as a function name.
    assert Converter.convert_tool_choice("my_function") == {
        "type": "function",
        "name": "my_function",
    }


def test_get_response_format_plain_text_and_json_schema():
    """
    For plain text output (default, or output type of `str`), the converter
    should return NOT_GIVEN, indicating no special response format constraint.
    If an output schema is provided for a structured type, the converter
    should return a `format` dict with the schema and strictness. The exact
    JSON schema depends on the output type; we just assert that required
    keys are present and that we get back the original schema.
    """
    # Default output (None) should be considered plain text.
    assert Converter.get_response_format(None) is NOT_GIVEN
    # An explicit plain-text schema (str) should also yield NOT_GIVEN.
    assert Converter.get_response_format(AgentOutputSchema(str)) is NOT_GIVEN

    # A model-based schema should produce a format dict.
    class OutModel(BaseModel):
        foo: int
        bar: str

    out_schema = AgentOutputSchema(OutModel)
    fmt = Converter.get_response_format(out_schema)
    assert isinstance(fmt, dict)
    assert "format" in fmt
    inner = fmt["format"]
    assert inner.get("type") == "json_schema"
    assert inner.get("name") == "final_output"
    assert isinstance(inner.get("schema"), dict)
    # Should include a strict flag matching the schema's strictness setting.
    assert inner.get("strict") == out_schema.strict_json_schema


def test_convert_tools_basic_types_and_includes():
    """
    Construct a variety of tool types and make sure `convert_tools` returns
    a matching list of tool param dicts and the expected includes. Also
    check that only a single computer tool is allowed.
    """
    # Simple function tool
    tool_fn = function_tool(lambda a: "x", name_override="fn")
    # File search tool with include_search_results set
    file_tool = FileSearchTool(
        max_num_results=3, vector_store_ids=["vs1"], include_search_results=True
    )
    # Web search tool with custom params
    web_tool = WebSearchTool(user_location=None, search_context_size="high")

    # Dummy computer tool subclassing the Computer ABC with minimal methods.
    class DummyComputer(Computer):
        @property
        def environment(self):
            return "mac"

        @property
        def dimensions(self):
            return (800, 600)

        def screenshot(self) -> str:
            raise NotImplementedError

        def click(self, x: int, y: int, button: str) -> None:
            raise NotImplementedError

        def double_click(self, x: int, y: int) -> None:
            raise NotImplementedError

        def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
            raise NotImplementedError

        def type(self, text: str) -> None:
            raise NotImplementedError

        def wait(self) -> None:
            raise NotImplementedError

        def move(self, x: int, y: int) -> None:
            raise NotImplementedError

        def keypress(self, keys: list[str]) -> None:
            raise NotImplementedError

        def drag(self, path: list[tuple[int, int]]) -> None:
            raise NotImplementedError

    # Wrap our concrete computer in a ComputerTool for conversion.
    comp_tool = ComputerTool(computer=DummyComputer())
    tools: list[Tool] = [tool_fn, file_tool, web_tool, comp_tool]
    converted = Converter.convert_tools(tools, handoffs=[])
    assert isinstance(converted.tools, list)
    assert isinstance(converted.includes, list)
    # The includes list should have exactly the include for file search when include_search_results
    # is True.
    assert converted.includes == ["file_search_call.results"]
    # There should be exactly four converted tool dicts.
    assert len(converted.tools) == 4
    # Extract types and verify.
    types = [ct["type"] for ct in converted.tools]
    assert "function" in types
    assert "file_search" in types
    assert "web_search_preview" in types
    assert "computer_use_preview" in types
    # Verify file search tool contains max_num_results and vector_store_ids
    file_params = next(ct for ct in converted.tools if ct["type"] == "file_search")
    assert file_params.get("max_num_results") == file_tool.max_num_results
    assert file_params.get("vector_store_ids") == file_tool.vector_store_ids
    # Verify web search tool contains user_location and search_context_size
    web_params = next(ct for ct in converted.tools if ct["type"] == "web_search_preview")
    assert web_params.get("user_location") == web_tool.user_location
    assert web_params.get("search_context_size") == web_tool.search_context_size
    # Verify computer tool contains environment and computed dimensions
    comp_params = next(ct for ct in converted.tools if ct["type"] == "computer_use_preview")
    assert comp_params.get("environment") == "mac"
    assert comp_params.get("display_width") == 800
    assert comp_params.get("display_height") == 600
    # The function tool dict should have name and description fields.
    fn_params = next(ct for ct in converted.tools if ct["type"] == "function")
    assert fn_params.get("name") == tool_fn.name
    assert fn_params.get("description") == tool_fn.description

    # Only one computer tool should be allowed.
    with pytest.raises(UserError):
        Converter.convert_tools(tools=[comp_tool, comp_tool], handoffs=[])


def test_convert_tools_includes_handoffs():
    """
    When handoff objects are included, `convert_tools` should append their
    tool param dicts after tools and include appropriate descriptions.
    """
    agent = Agent(name="support", handoff_description="Handles support")
    handoff_obj = handoff(agent)
    converted = Converter.convert_tools(tools=[], handoffs=[handoff_obj])
    assert isinstance(converted.tools, list)
    assert len(converted.tools) == 1
    handoff_tool = converted.tools[0]
    assert handoff_tool.get("type") == "function"
    assert handoff_tool.get("name") == Handoff.default_tool_name(agent)
    assert handoff_tool.get("description") == Handoff.default_tool_description(agent)
    # No includes for handoffs by default.
    assert converted.includes == []


<file_info>
path: tests/voice/helpers.py
name: helpers.py
</file_info>
try:
    from agents.voice import StreamedAudioResult
except ImportError:
    pass


async def extract_events(result: StreamedAudioResult) -> tuple[list[str], list[bytes]]:
    """Collapse pipeline stream events to simple labels for ordering assertions."""
    flattened: list[str] = []
    audio_chunks: list[bytes] = []

    async for ev in result.stream():
        if ev.type == "voice_stream_event_audio":
            if ev.data is not None:
                audio_chunks.append(ev.data.tobytes())
            flattened.append("audio")
        elif ev.type == "voice_stream_event_lifecycle":
            flattened.append(ev.event)
        elif ev.type == "voice_stream_event_error":
            flattened.append("error")
    return flattened, audio_chunks


<file_info>
path: tests/voice/test_pipeline.py
name: test_pipeline.py
</file_info>
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

try:
    from agents.voice import AudioInput, TTSModelSettings, VoicePipeline, VoicePipelineConfig

    from .fake_models import FakeStreamedAudioInput, FakeSTT, FakeTTS, FakeWorkflow
    from .helpers import extract_events
except ImportError:
    pass


@pytest.mark.asyncio
async def test_voicepipeline_run_single_turn() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["out_1"]])
    fake_tts = FakeTTS()
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0])


@pytest.mark.asyncio
async def test_voicepipeline_streamed_audio_input() -> None:
    # Multi turn. Should produce 2 audio outputs, which are the TTS outputs of "out_1" and "out_2"

    fake_stt = FakeSTT(["first", "second"])
    workflow = FakeWorkflow([["out_1"], ["out_2"]])
    fake_tts = FakeTTS()
    pipeline = VoicePipeline(workflow=workflow, stt_model=fake_stt, tts_model=fake_tts)

    streamed_audio_input = await FakeStreamedAudioInput.get(count=2)

    result = await pipeline.run(streamed_audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # out_1
        "turn_ended",
        "turn_started",
        "audio",  # out_2
        "turn_ended",
        "session_ended",
    ]
    assert len(audio_chunks) == 2
    await fake_tts.verify_audio("out_1", audio_chunks[0])
    await fake_tts.verify_audio("out_2", audio_chunks[1])


@pytest.mark.asyncio
async def test_voicepipeline_run_single_turn_split_words() -> None:
    # Single turn. Should produce multiple audio outputs, which are the TTS outputs of "foo bar baz"
    # split into words and then "foo2 bar2 baz2" split into words.

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["foo bar baz"]])
    fake_tts = FakeTTS(strategy="split_words")
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # foo
        "audio",  # bar
        "audio",  # baz
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio_chunks("foo bar baz", audio_chunks)


@pytest.mark.asyncio
async def test_voicepipeline_run_multi_turn_split_words() -> None:
    # Multi turn. Should produce multiple audio outputs, which are the TTS outputs of "foo bar baz"
    # split into words.

    fake_stt = FakeSTT(["first", "second"])
    workflow = FakeWorkflow([["foo bar baz"], ["foo2 bar2 baz2"]])
    fake_tts = FakeTTS(strategy="split_words")
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    streamed_audio_input = await FakeStreamedAudioInput.get(count=6)
    result = await pipeline.run(streamed_audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # foo
        "audio",  # bar
        "audio",  # baz
        "turn_ended",
        "turn_started",
        "audio",  # foo2
        "audio",  # bar2
        "audio",  # baz2
        "turn_ended",
        "session_ended",
    ]
    assert len(audio_chunks) == 6
    await fake_tts.verify_audio_chunks("foo bar baz", audio_chunks[:3])
    await fake_tts.verify_audio_chunks("foo2 bar2 baz2", audio_chunks[3:])


@pytest.mark.asyncio
async def test_voicepipeline_float32() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["out_1"]])
    fake_tts = FakeTTS()
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1, dtype=np.float32))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0], dtype=np.float32)


@pytest.mark.asyncio
async def test_voicepipeline_transform_data() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    def _transform_data(
        data_chunk: npt.NDArray[np.int16 | np.float32],
    ) -> npt.NDArray[np.int16]:
        return data_chunk.astype(np.int16)

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["out_1"]])
    fake_tts = FakeTTS()
    config = VoicePipelineConfig(
        tts_settings=TTSModelSettings(
            buffer_size=1,
            dtype=np.float32,
            transform_data=_transform_data,
        )
    )
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0], dtype=np.int16)


<file_info>
path: tests/test_tool_choice_reset.py
name: test_tool_choice_reset.py
</file_info>
import pytest

from agents import Agent, ModelSettings, Runner, Tool
from agents._run_impl import RunImpl

from .fake_model import FakeModel
from .test_responses import (
    get_function_tool,
    get_function_tool_call,
    get_text_message,
)


class TestToolChoiceReset:

    def test_should_reset_tool_choice_direct(self):
        """
        Test the _should_reset_tool_choice method directly with various inputs
        to ensure it correctly identifies cases where reset is needed.
        """
        # Case 1: tool_choice = None should not reset
        model_settings = ModelSettings(tool_choice=None)
        tools1: list[Tool] = [get_function_tool("tool1")]
        # Cast to list[Tool] to fix type checking issues
        assert not RunImpl._should_reset_tool_choice(model_settings, tools1)

        # Case 2: tool_choice = "auto" should not reset
        model_settings = ModelSettings(tool_choice="auto")
        assert not RunImpl._should_reset_tool_choice(model_settings, tools1)

        # Case 3: tool_choice = "none" should not reset
        model_settings = ModelSettings(tool_choice="none")
        assert not RunImpl._should_reset_tool_choice(model_settings, tools1)

        # Case 4: tool_choice = "required" with one tool should reset
        model_settings = ModelSettings(tool_choice="required")
        assert RunImpl._should_reset_tool_choice(model_settings, tools1)

        # Case 5: tool_choice = "required" with multiple tools should not reset
        model_settings = ModelSettings(tool_choice="required")
        tools2: list[Tool] = [get_function_tool("tool1"), get_function_tool("tool2")]
        assert not RunImpl._should_reset_tool_choice(model_settings, tools2)

        # Case 6: Specific tool choice should reset
        model_settings = ModelSettings(tool_choice="specific_tool")
        assert RunImpl._should_reset_tool_choice(model_settings, tools1)

    @pytest.mark.asyncio
    async def test_required_tool_choice_with_multiple_runs(self):
        """
        Test scenario 1: When multiple runs are executed with tool_choice="required"
        Ensure each run works correctly and doesn't get stuck in infinite loop
        Also verify that tool_choice remains "required" between runs
        """
        # Set up our fake model with responses for two runs
        fake_model = FakeModel()
        fake_model.add_multiple_turn_outputs([
            [get_text_message("First run response")],
            [get_text_message("Second run response")]
        ])

        # Create agent with a custom tool and tool_choice="required"
        custom_tool = get_function_tool("custom_tool")
        agent = Agent(
            name="test_agent",
            model=fake_model,
            tools=[custom_tool],
            model_settings=ModelSettings(tool_choice="required"),
        )

        # First run should work correctly and preserve tool_choice
        result1 = await Runner.run(agent, "first run")
        assert result1.final_output == "First run response"
        assert agent.model_settings.tool_choice == "required", "tool_choice should stay required"

        # Second run should also work correctly with tool_choice still required
        result2 = await Runner.run(agent, "second run")
        assert result2.final_output == "Second run response"
        assert agent.model_settings.tool_choice == "required", "tool_choice should stay required"

    @pytest.mark.asyncio
    async def test_required_with_stop_at_tool_name(self):
        """
        Test scenario 2: When using required tool_choice with stop_at_tool_names behavior
        Ensure it correctly stops at the specified tool
        """
        # Set up fake model to return a tool call for second_tool
        fake_model = FakeModel()
        fake_model.set_next_output([
            get_function_tool_call("second_tool", "{}")
        ])

        # Create agent with two tools and tool_choice="required" and stop_at_tool behavior
        first_tool = get_function_tool("first_tool", return_value="first tool result")
        second_tool = get_function_tool("second_tool", return_value="second tool result")

        agent = Agent(
            name="test_agent",
            model=fake_model,
            tools=[first_tool, second_tool],
            model_settings=ModelSettings(tool_choice="required"),
            tool_use_behavior={"stop_at_tool_names": ["second_tool"]},
        )

        # Run should stop after using second_tool
        result = await Runner.run(agent, "run test")
        assert result.final_output == "second tool result"

    @pytest.mark.asyncio
    async def test_specific_tool_choice(self):
        """
        Test scenario 3: When using a specific tool choice name
        Ensure it doesn't cause infinite loops
        """
        # Set up fake model to return a text message
        fake_model = FakeModel()
        fake_model.set_next_output([get_text_message("Test message")])

        # Create agent with specific tool_choice
        tool1 = get_function_tool("tool1")
        tool2 = get_function_tool("tool2")
        tool3 = get_function_tool("tool3")

        agent = Agent(
            name="test_agent",
            model=fake_model,
            tools=[tool1, tool2, tool3],
            model_settings=ModelSettings(tool_choice="tool1"),  # Specific tool
        )

        # Run should complete without infinite loops
        result = await Runner.run(agent, "first run")
        assert result.final_output == "Test message"

    @pytest.mark.asyncio
    async def test_required_with_single_tool(self):
        """
        Test scenario 4: When using required tool_choice with only one tool
        Ensure it doesn't cause infinite loops
        """
        # Set up fake model to return a tool call followed by a text message
        fake_model = FakeModel()
        fake_model.add_multiple_turn_outputs([
            # First call returns a tool call
            [get_function_tool_call("custom_tool", "{}")],
            # Second call returns a text message
            [get_text_message("Final response")]
        ])

        # Create agent with a single tool and tool_choice="required"
        custom_tool = get_function_tool("custom_tool", return_value="tool result")
        agent = Agent(
            name="test_agent",
            model=fake_model,
            tools=[custom_tool],
            model_settings=ModelSettings(tool_choice="required"),
        )

        # Run should complete without infinite loops
        result = await Runner.run(agent, "first run")
        assert result.final_output == "Final response"


<file_info>
path: examples/model_providers/custom_example_global.py
name: custom_example_global.py
</file_info>
import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )


"""This example uses a custom provider for all requests by default. We do three things:
1. Create a custom client.
2. Set it as the default OpenAI client, and don't use it for tracing.
3. Set the default API as Chat Completions, as most LLM providers don't yet support Responses API.

Note that in this example, we disable tracing under the assumption that you don't have an API key
from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
or call set_tracing_export_api_key() to set a tracing specific key.
"""

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=MODEL_NAME,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/basic/dynamic_system_prompt.py
name: dynamic_system_prompt.py
</file_info>
import asyncio
import random
from typing import Literal

from agents import Agent, RunContextWrapper, Runner


class CustomContext:
    def __init__(self, style: Literal["haiku", "pirate", "robot"]):
        self.style = style


def custom_instructions(
    run_context: RunContextWrapper[CustomContext], agent: Agent[CustomContext]
) -> str:
    context = run_context.context
    if context.style == "haiku":
        return "Only respond in haikus."
    elif context.style == "pirate":
        return "Respond as a pirate."
    else:
        return "Respond as a robot and say 'beep boop' a lot."


agent = Agent(
    name="Chat agent",
    instructions=custom_instructions,
)


async def main():
    choice: Literal["haiku", "pirate", "robot"] = random.choice(["haiku", "pirate", "robot"])
    context = CustomContext(style=choice)
    print(f"Using style: {choice}\n")

    user_message = "Tell me a joke."
    print(f"User: {user_message}")
    result = await Runner.run(agent, user_message, context=context)

    print(f"Assistant: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

"""
$ python examples/basic/dynamic_system_prompt.py

Using style: haiku

User: Tell me a joke.
Assistant: Why don't eggs tell jokes?
They might crack each other's shells,
leaving yolk on face.

$ python examples/basic/dynamic_system_prompt.py
Using style: robot

User: Tell me a joke.
Assistant: Beep boop! Why was the robot so bad at soccer? Beep boop... because it kept kicking up a debug! Beep boop!

$ python examples/basic/dynamic_system_prompt.py
Using style: pirate

User: Tell me a joke.
Assistant: Why did the pirate go to school?

To improve his arrr-ticulation! Har har har! 🏴‍☠️
"""


<file_info>
path: examples/agent_patterns/llm_as_a_judge.py
name: llm_as_a_judge.py
</file_info>
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

"""
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied
with the outline.
"""

story_outline_generator = Agent(
    name="story_outline_generator",
    instructions=(
        "You generate a very short story outline based on the user's input."
        "If there is any feedback provided, use it to improve the outline."
    ),
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a story outline and decide if it's good enough."
        "If it's not good enough, you provide feedback on what needs to be improved."
        "Never give it a pass on the first try."
    ),
    output_type=EvaluationFeedback,
)


async def main() -> None:
    msg = input("What kind of story would you like to hear? ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline: str | None = None

    # We'll run the entire workflow in a single trace
    with trace("LLM as a judge"):
        while True:
            story_outline_result = await Runner.run(
                story_outline_generator,
                input_items,
            )

            input_items = story_outline_result.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(story_outline_result.new_items)
            print("Story outline generated")

            evaluator_result = await Runner.run(evaluator, input_items)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")

            if result.score == "pass":
                print("Story outline is good enough, exiting.")
                break

            print("Re-running with feedback")

            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final story outline: {latest_outline}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/processor_interface.py
name: processor_interface.py
</file_info>
import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spans import Span
    from .traces import Trace


class TracingProcessor(abc.ABC):
    """Interface for processing spans."""

    @abc.abstractmethod
    def on_trace_start(self, trace: "Trace") -> None:
        """Called when a trace is started.

        Args:
            trace: The trace that started.
        """
        pass

    @abc.abstractmethod
    def on_trace_end(self, trace: "Trace") -> None:
        """Called when a trace is finished.

        Args:
            trace: The trace that started.
        """
        pass

    @abc.abstractmethod
    def on_span_start(self, span: "Span[Any]") -> None:
        """Called when a span is started.

        Args:
            span: The span that started.
        """
        pass

    @abc.abstractmethod
    def on_span_end(self, span: "Span[Any]") -> None:
        """Called when a span is finished. Should not block or raise exceptions.

        Args:
            span: The span that finished.
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Called when the application stops."""
        pass

    @abc.abstractmethod
    def force_flush(self) -> None:
        """Forces an immediate flush of all queued spans/traces."""
        pass


class TracingExporter(abc.ABC):
    """Exports traces and spans. For example, could log them or send them to a backend."""

    @abc.abstractmethod
    def export(self, items: list["Trace | Span[Any]"]) -> None:
        """Exports a list of traces and spans.

        Args:
            items: The items to export.
        """
        pass


<file_info>
path: src/agents/models/fake_id.py
name: fake_id.py
</file_info>
FAKE_RESPONSES_ID = "__fake_id__"
"""This is a placeholder ID used to fill in the `id` field in Responses API related objects. It's
useful when you're creating Responses objects from non-Responses APIs, e.g. the OpenAI Chat
Completions API or other LLM providers.
"""


<file_info>
path: src/agents/models/interface.py
name: interface.py
</file_info>
from __future__ import annotations

import abc
import enum
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..agent_output import AgentOutputSchema
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..tool import Tool

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


class ModelTracing(enum.Enum):
    DISABLED = 0
    """Tracing is disabled entirely."""

    ENABLED = 1
    """Tracing is enabled, and all data is included."""

    ENABLED_WITHOUT_DATA = 2
    """Tracing is enabled, but inputs/outputs are not included."""

    def is_disabled(self) -> bool:
        return self == ModelTracing.DISABLED

    def include_data(self) -> bool:
        return self == ModelTracing.ENABLED


class Model(abc.ABC):
    """The base interface for calling an LLM."""

    @abc.abstractmethod
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        """Get a response from the model.

        Args:
            system_instructions: The system instructions to use.
            input: The input items to the model, in OpenAI Responses format.
            model_settings: The model settings to use.
            tools: The tools available to the model.
            output_schema: The output schema to use.
            handoffs: The handoffs available to the model.
            tracing: Tracing configuration.

        Returns:
            The full model response.
        """
        pass

    @abc.abstractmethod
    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream a response from the model.

        Args:
            system_instructions: The system instructions to use.
            input: The input items to the model, in OpenAI Responses format.
            model_settings: The model settings to use.
            tools: The tools available to the model.
            output_schema: The output schema to use.
            handoffs: The handoffs available to the model.
            tracing: Tracing configuration.

        Returns:
            An iterator of response stream events, in OpenAI Responses format.
        """
        pass


class ModelProvider(abc.ABC):
    """The base interface for a model provider.

    Model provider is responsible for looking up Models by name.
    """

    @abc.abstractmethod
    def get_model(self, model_name: str | None) -> Model:
        """Get a model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The model.
        """


<file_info>
path: src/agents/voice/models/openai_model_provider.py
name: openai_model_provider.py
</file_info>
from __future__ import annotations

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from ...models import _openai_shared
from ..model import STTModel, TTSModel, VoiceModelProvider
from .openai_stt import OpenAISTTModel
from .openai_tts import OpenAITTSModel

_http_client: httpx.AsyncClient | None = None


# If we create a new httpx client for each request, that would mean no sharing of connection pools,
# which would mean worse latency and resource usage. So, we share the client across requests.
def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = DefaultAsyncHttpxClient()
    return _http_client


DEFAULT_STT_MODEL = "gpt-4o-transcribe"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"


class OpenAIVoiceModelProvider(VoiceModelProvider):
    """A voice model provider that uses OpenAI models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        """Create a new OpenAI voice model provider.

        Args:
            api_key: The API key to use for the OpenAI client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
                default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            organization: The organization to use for the OpenAI client.
            project: The project to use for the OpenAI client.
        """
        if openai_client is not None:
            assert api_key is None and base_url is None, (
                "Don't provide api_key or base_url if you provide openai_client"
            )
            self._client: AsyncOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project

    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
    # AsyncOpenAI() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=shared_http_client(),
            )

        return self._client

    def get_stt_model(self, model_name: str | None) -> STTModel:
        """Get a speech-to-text model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The speech-to-text model.
        """
        return OpenAISTTModel(model_name or DEFAULT_STT_MODEL, self._get_client())

    def get_tts_model(self, model_name: str | None) -> TTSModel:
        """Get a text-to-speech model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The text-to-speech model.
        """
        return OpenAITTSModel(model_name or DEFAULT_TTS_MODEL, self._get_client())


<file_info>
path: src/agents/lifecycle.py
name: lifecycle.py
</file_info>
from typing import Any, Generic

from .agent import Agent
from .run_context import RunContextWrapper, TContext
from .tool import Tool


class RunHooks(Generic[TContext]):
    """A class that receives callbacks on various lifecycle events in an agent run. Subclass and
    override the methods you need.
    """

    async def on_agent_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before the agent is invoked. Called each time the current agent changes."""
        pass

    async def on_agent_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        pass

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called when a handoff occurs."""
        pass

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
    ) -> None:
        """Called before a tool is invoked."""
        pass

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: str,
    ) -> None:
        """Called after a tool is invoked."""
        pass


class AgentHooks(Generic[TContext]):
    """A class that receives callbacks on various lifecycle events for a specific agent. You can
    set this on `agent.hooks` to receive events for that specific agent.

    Subclass and override the methods you need.
    """

    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        """Called before the agent is invoked. Called each time the running agent is changed to this
        agent."""
        pass

    async def on_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        pass

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        source: Agent[TContext],
    ) -> None:
        """Called when the agent is being handed off to. The `source` is the agent that is handing
        off to this agent."""
        pass

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
    ) -> None:
        """Called before a tool is invoked."""
        pass

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: str,
    ) -> None:
        """Called after a tool is invoked."""
        pass


<file_info>
path: src/agents/handoffs.py
name: handoffs.py
</file_info>
from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, cast, overload

from pydantic import TypeAdapter
from typing_extensions import TypeAlias, TypeVar

from .exceptions import ModelBehaviorError, UserError
from .items import RunItem, TResponseInputItem
from .run_context import RunContextWrapper, TContext
from .strict_schema import ensure_strict_json_schema
from .tracing.spans import SpanError
from .util import _error_tracing, _json, _transforms

if TYPE_CHECKING:
    from .agent import Agent


# The handoff input type is the type of data passed when the agent is called via a handoff.
THandoffInput = TypeVar("THandoffInput", default=Any)

OnHandoffWithInput = Callable[[RunContextWrapper[Any], THandoffInput], Any]
OnHandoffWithoutInput = Callable[[RunContextWrapper[Any]], Any]


@dataclass(frozen=True)
class HandoffInputData:
    input_history: str | tuple[TResponseInputItem, ...]
    """
    The input history before `Runner.run()` was called.
    """

    pre_handoff_items: tuple[RunItem, ...]
    """
    The items generated before the agent turn where the handoff was invoked.
    """

    new_items: tuple[RunItem, ...]
    """
    The new items generated during the current agent turn, including the item that triggered the
    handoff and the tool output message representing the response from the handoff output.
    """


HandoffInputFilter: TypeAlias = Callable[[HandoffInputData], HandoffInputData]
"""A function that filters the input data passed to the next agent."""


@dataclass
class Handoff(Generic[TContext]):
    """A handoff is when an agent delegates a task to another agent.
    For example, in a customer support scenario you might have a "triage agent" that determines
    which agent should handle the user's request, and sub-agents that specialize in different
    areas like billing, account management, etc.
    """

    tool_name: str
    """The name of the tool that represents the handoff."""

    tool_description: str
    """The description of the tool that represents the handoff."""

    input_json_schema: dict[str, Any]
    """The JSON schema for the handoff input. Can be empty if the handoff does not take an input.
    """

    on_invoke_handoff: Callable[[RunContextWrapper[Any], str], Awaitable[Agent[TContext]]]
    """The function that invokes the handoff. The parameters passed are:
    1. The handoff run context
    2. The arguments from the LLM, as a JSON string. Empty string if input_json_schema is empty.

    Must return an agent.
    """

    agent_name: str
    """The name of the agent that is being handed off to."""

    input_filter: HandoffInputFilter | None = None
    """A function that filters the inputs that are passed to the next agent. By default, the new
    agent sees the entire conversation history. In some cases, you may want to filter inputs e.g.
    to remove older inputs, or remove tools from existing inputs.

    The function will receive the entire conversation history so far, including the input item
    that triggered the handoff and a tool call output item representing the handoff tool's output.

    You are free to modify the input history or new items as you see fit. The next agent that
    runs will receive `handoff_input_data.all_items`.

    IMPORTANT: in streaming mode, we will not stream anything as a result of this function. The
    items generated before will already have been streamed.
    """

    strict_json_schema: bool = True
    """Whether the input JSON schema is in strict mode. We **strongly** recommend setting this to
    True, as it increases the likelihood of correct JSON input.
    """

    def get_transfer_message(self, agent: Agent[Any]) -> str:
        base = f"{{'assistant': '{agent.name}'}}"
        return base

    @classmethod
    def default_tool_name(cls, agent: Agent[Any]) -> str:
        return _transforms.transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_tool_description(cls, agent: Agent[Any]) -> str:
        return (
            f"Handoff to the {agent.name} agent to handle the request. "
            f"{agent.handoff_description or ''}"
        )


@overload
def handoff(
    agent: Agent[TContext],
    *,
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


@overload
def handoff(
    agent: Agent[TContext],
    *,
    on_handoff: OnHandoffWithInput[THandoffInput],
    input_type: type[THandoffInput],
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


@overload
def handoff(
    agent: Agent[TContext],
    *,
    on_handoff: OnHandoffWithoutInput,
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


def handoff(
    agent: Agent[TContext],
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    on_handoff: OnHandoffWithInput[THandoffInput] | OnHandoffWithoutInput | None = None,
    input_type: type[THandoffInput] | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]:
    """Create a handoff from an agent.

    Args:
        agent: The agent to handoff to, or a function that returns an agent.
        tool_name_override: Optional override for the name of the tool that represents the handoff.
        tool_description_override: Optional override for the description of the tool that
            represents the handoff.
        on_handoff: A function that runs when the handoff is invoked.
        input_type: the type of the input to the handoff. If provided, the input will be validated
            against this type. Only relevant if you pass a function that takes an input.
        input_filter: a function that filters the inputs that are passed to the next agent.
    """
    assert (on_handoff and input_type) or not (on_handoff and input_type), (
        "You must provide either both on_input and input_type, or neither"
    )
    type_adapter: TypeAdapter[Any] | None
    if input_type is not None:
        assert callable(on_handoff), "on_handoff must be callable"
        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 2:
            raise UserError("on_handoff must take two arguments: context and input")

        type_adapter = TypeAdapter(input_type)
        input_json_schema = type_adapter.json_schema()
    else:
        type_adapter = None
        input_json_schema = {}
        if on_handoff is not None:
            sig = inspect.signature(on_handoff)
            if len(sig.parameters) != 1:
                raise UserError("on_handoff must take one argument: context")

    async def _invoke_handoff(
        ctx: RunContextWrapper[Any], input_json: str | None = None
    ) -> Agent[Any]:
        if input_type is not None and type_adapter is not None:
            if input_json is None:
                _error_tracing.attach_error_to_current_span(
                    SpanError(
                        message="Handoff function expected non-null input, but got None",
                        data={"details": "input_json is None"},
                    )
                )
                raise ModelBehaviorError("Handoff function expected non-null input, but got None")

            validated_input = _json.validate_json(
                json_str=input_json,
                type_adapter=type_adapter,
                partial=False,
            )
            input_func = cast(OnHandoffWithInput[THandoffInput], on_handoff)
            if inspect.iscoroutinefunction(input_func):
                await input_func(ctx, validated_input)
            else:
                input_func(ctx, validated_input)
        elif on_handoff is not None:
            no_input_func = cast(OnHandoffWithoutInput, on_handoff)
            if inspect.iscoroutinefunction(no_input_func):
                await no_input_func(ctx)
            else:
                no_input_func(ctx)

        return agent

    tool_name = tool_name_override or Handoff.default_tool_name(agent)
    tool_description = tool_description_override or Handoff.default_tool_description(agent)

    # Always ensure the input JSON schema is in strict mode
    # If there is a need, we can make this configurable in the future
    input_json_schema = ensure_strict_json_schema(input_json_schema)

    return Handoff(
        tool_name=tool_name,
        tool_description=tool_description,
        input_json_schema=input_json_schema,
        on_invoke_handoff=_invoke_handoff,
        input_filter=input_filter,
        agent_name=agent.name,
    )


<file_info>
path: tests/test_items_helpers.py
name: test_items_helpers.py
</file_info>
from __future__ import annotations

from openai.types.responses.response_computer_tool_call import (
    ActionScreenshot,
    ResponseComputerToolCall,
)
from openai.types.responses.response_computer_tool_call_param import ResponseComputerToolCallParam
from openai.types.responses.response_file_search_tool_call import ResponseFileSearchToolCall
from openai.types.responses.response_file_search_tool_call_param import (
    ResponseFileSearchToolCallParam,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam
from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search_param import ResponseFunctionWebSearchParam
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam

from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    ReasoningItem,
    RunItem,
    TResponseInputItem,
    Usage,
)


def make_message(
    content_items: list[ResponseOutputText | ResponseOutputRefusal],
) -> ResponseOutputMessage:
    """
    Helper to construct a ResponseOutputMessage with a single batch of content
    items, using a fixed id/status.
    """
    return ResponseOutputMessage(
        id="msg123",
        content=content_items,
        role="assistant",
        status="completed",
        type="message",
    )


def test_extract_last_content_of_text_message() -> None:
    # Build a message containing two text segments.
    content1 = ResponseOutputText(annotations=[], text="Hello ", type="output_text")
    content2 = ResponseOutputText(annotations=[], text="world!", type="output_text")
    message = make_message([content1, content2])
    # Helpers should yield the last segment's text.
    assert ItemHelpers.extract_last_content(message) == "world!"


def test_extract_last_content_of_refusal_message() -> None:
    # Build a message whose last content entry is a refusal.
    content1 = ResponseOutputText(annotations=[], text="Before refusal", type="output_text")
    refusal = ResponseOutputRefusal(refusal="I cannot do that", type="refusal")
    message = make_message([content1, refusal])
    # Helpers should extract the refusal string when last content is a refusal.
    assert ItemHelpers.extract_last_content(message) == "I cannot do that"


def test_extract_last_content_non_message_returns_empty() -> None:
    # Construct some other type of output item, e.g. a tool call, to verify non-message returns "".
    tool_call = ResponseFunctionToolCall(
        id="tool123",
        arguments="{}",
        call_id="call123",
        name="func",
        type="function_call",
    )
    assert ItemHelpers.extract_last_content(tool_call) == ""


def test_extract_last_text_returns_text_only() -> None:
    # A message whose last segment is text yields the text.
    first_text = ResponseOutputText(annotations=[], text="part1", type="output_text")
    second_text = ResponseOutputText(annotations=[], text="part2", type="output_text")
    message = make_message([first_text, second_text])
    assert ItemHelpers.extract_last_text(message) == "part2"
    # Whereas when last content is a refusal, extract_last_text returns None.
    message2 = make_message([first_text, ResponseOutputRefusal(refusal="no", type="refusal")])
    assert ItemHelpers.extract_last_text(message2) is None


def test_input_to_new_input_list_from_string() -> None:
    result = ItemHelpers.input_to_new_input_list("hi")
    # Should wrap the string into a list with a single dict containing content and user role.
    assert isinstance(result, list)
    assert result == [{"content": "hi", "role": "user"}]


def test_input_to_new_input_list_deep_copies_lists() -> None:
    # Given a list of message dictionaries, ensure the returned list is a deep copy.
    original: list[TResponseInputItem] = [{"content": "abc", "role": "developer"}]
    new_list = ItemHelpers.input_to_new_input_list(original)
    assert new_list == original
    # Mutating the returned list should not mutate the original.
    new_list.pop()
    assert "content" in original[0] and original[0].get("content") == "abc"


def test_text_message_output_concatenates_text_segments() -> None:
    # Build a message with both text and refusal segments, only text segments are concatenated.
    pieces: list[ResponseOutputText | ResponseOutputRefusal] = []
    pieces.append(ResponseOutputText(annotations=[], text="a", type="output_text"))
    pieces.append(ResponseOutputRefusal(refusal="denied", type="refusal"))
    pieces.append(ResponseOutputText(annotations=[], text="b", type="output_text"))
    message = make_message(pieces)
    # Wrap into MessageOutputItem to feed into text_message_output.
    item = MessageOutputItem(agent=Agent(name="test"), raw_item=message)
    assert ItemHelpers.text_message_output(item) == "ab"


def test_text_message_outputs_across_list_of_runitems() -> None:
    """
    Compose several RunItem instances, including a non-message run item, and ensure
    that only MessageOutputItem instances contribute any text. The non-message
    (ReasoningItem) should be ignored by Helpers.text_message_outputs.
    """
    message1 = make_message([ResponseOutputText(annotations=[], text="foo", type="output_text")])
    message2 = make_message([ResponseOutputText(annotations=[], text="bar", type="output_text")])
    item1: RunItem = MessageOutputItem(agent=Agent(name="test"), raw_item=message1)
    item2: RunItem = MessageOutputItem(agent=Agent(name="test"), raw_item=message2)
    # Create a non-message run item of a different type, e.g., a reasoning trace.
    reasoning = ResponseReasoningItem(id="rid", summary=[], type="reasoning")
    non_message_item: RunItem = ReasoningItem(agent=Agent(name="test"), raw_item=reasoning)
    # Confirm only the message outputs are concatenated.
    assert ItemHelpers.text_message_outputs([item1, non_message_item, item2]) == "foobar"


def test_tool_call_output_item_constructs_function_call_output_dict():
    # Build a simple ResponseFunctionToolCall.
    call = ResponseFunctionToolCall(
        id="call-abc",
        arguments='{"x": 1}',
        call_id="call-abc",
        name="do_something",
        type="function_call",
    )
    payload = ItemHelpers.tool_call_output_item(call, "result-string")

    assert isinstance(payload, dict)
    assert payload["type"] == "function_call_output"
    assert payload["call_id"] == call.id
    assert payload["output"] == "result-string"


# The following tests ensure that every possible output item type defined by
# OpenAI's API can be converted back into an input item dict via
# ModelResponse.to_input_items. The output and input schema for each item are
# intended to be symmetric, so given any ResponseOutputItem, its model_dump
# should produce a dict that can satisfy the corresponding TypedDict input
# type. These tests construct minimal valid instances of each output type,
# invoke to_input_items, and then verify that the resulting dict can be used
# to round-trip back into a Pydantic output model without errors.


def test_to_input_items_for_message() -> None:
    """An output message should convert into an input dict matching the message's own structure."""
    content = ResponseOutputText(annotations=[], text="hello world", type="output_text")
    message = ResponseOutputMessage(
        id="m1", content=[content], role="assistant", status="completed", type="message"
    )
    resp = ModelResponse(output=[message], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    # The dict should contain exactly the primitive values of the message
    expected: ResponseOutputMessageParam = {
        "id": "m1",
        "content": [
            {
                "annotations": [],
                "text": "hello world",
                "type": "output_text",
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    assert input_items[0] == expected


def test_to_input_items_for_function_call() -> None:
    """A function tool call output should produce the same dict as a function tool call input."""
    tool_call = ResponseFunctionToolCall(
        id="f1", arguments="{}", call_id="c1", name="func", type="function_call"
    )
    resp = ModelResponse(output=[tool_call], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    expected: ResponseFunctionToolCallParam = {
        "id": "f1",
        "arguments": "{}",
        "call_id": "c1",
        "name": "func",
        "type": "function_call",
    }
    assert input_items[0] == expected


def test_to_input_items_for_file_search_call() -> None:
    """A file search tool call output should produce the same dict as a file search input."""
    fs_call = ResponseFileSearchToolCall(
        id="fs1", queries=["query"], status="completed", type="file_search_call"
    )
    resp = ModelResponse(output=[fs_call], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    expected: ResponseFileSearchToolCallParam = {
        "id": "fs1",
        "queries": ["query"],
        "status": "completed",
        "type": "file_search_call",
    }
    assert input_items[0] == expected


def test_to_input_items_for_web_search_call() -> None:
    """A web search tool call output should produce the same dict as a web search input."""
    ws_call = ResponseFunctionWebSearch(id="w1", status="completed", type="web_search_call")
    resp = ModelResponse(output=[ws_call], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    expected: ResponseFunctionWebSearchParam = {
        "id": "w1",
        "status": "completed",
        "type": "web_search_call",
    }
    assert input_items[0] == expected


def test_to_input_items_for_computer_call_click() -> None:
    """A computer call output should yield a dict whose shape matches the computer call input."""
    action = ActionScreenshot(type="screenshot")
    comp_call = ResponseComputerToolCall(
        id="comp1",
        action=action,
        type="computer_call",
        call_id="comp1",
        pending_safety_checks=[],
        status="completed",
    )
    resp = ModelResponse(output=[comp_call], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    converted_dict = input_items[0]
    # Top-level keys should match what we expect for a computer call input
    expected: ResponseComputerToolCallParam = {
        "id": "comp1",
        "type": "computer_call",
        "action": {"type": "screenshot"},
        "call_id": "comp1",
        "pending_safety_checks": [],
        "status": "completed",
    }
    assert converted_dict == expected


def test_to_input_items_for_reasoning() -> None:
    """A reasoning output should produce the same dict as a reasoning input item."""
    rc = Summary(text="why", type="summary_text")
    reasoning = ResponseReasoningItem(id="rid1", summary=[rc], type="reasoning")
    resp = ModelResponse(output=[reasoning], usage=Usage(), referenceable_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    converted_dict = input_items[0]

    expected: ResponseReasoningItemParam = {
        "id": "rid1",
        "summary": [{"text": "why", "type": "summary_text"}],
        "type": "reasoning",
    }
    print(converted_dict)
    print(expected)
    assert converted_dict == expected


<file_info>
path: examples/research_bot/agents/search_agent.py
name: search_agent.py
</file_info>
from agents import Agent, WebSearchTool
from agents.model_settings import ModelSettings

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and"
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300"
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good"
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the"
    "essence and ignore any fluff. Do not include any additional commentary other than the summary"
    "itself."
)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)


<file_info>
path: examples/agent_patterns/parallelization.py
name: parallelization.py
</file_info>
import asyncio

from agents import Agent, ItemHelpers, Runner, trace

"""
This example shows the parallelization pattern. We run the agent three times in parallel, and pick
the best result.
"""

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Spanish translation from the given options.",
)


async def main():
    msg = input("Hi! Enter a message, and we'll translate it to Spanish.\n\n")

    # Ensure the entire workflow is a single trace
    with trace("Parallel translation"):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(
                spanish_agent,
                msg,
            ),
            Runner.run(
                spanish_agent,
                msg,
            ),
            Runner.run(
                spanish_agent,
                msg,
            ),
        )

        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
            ItemHelpers.text_message_outputs(res_3.new_items),
        ]

        translations = "\n\n".join(outputs)
        print(f"\n\nTranslations:\n\n{translations}")

        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
        )

    print("\n\n-----")

    print(f"Best translation: {best_translation.final_output}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/handoffs/message_filter.py
name: message_filter.py
</file_info>
from __future__ import annotations

import json
import random

from agents import Agent, HandoffInputData, Runner, function_tool, handoff, trace
from agents.extensions import handoff_filters


@function_tool
def random_number_tool(max: int) -> int:
    """Return a random integer between 0 and the given maximum."""
    return random.randint(0, max)


def spanish_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    # First, we'll remove any tool-related messages from the message history
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # Second, we'll also remove the first two items from the history, just for demonstration
    history = (
        tuple(handoff_message_data.input_history[2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )


first_agent = Agent(
    name="Assistant",
    instructions="Be extremely concise.",
    tools=[random_number_tool],
)

spanish_agent = Agent(
    name="Spanish Assistant",
    instructions="You only speak Spanish and are extremely concise.",
    handoff_description="A Spanish-speaking assistant.",
)

second_agent = Agent(
    name="Assistant",
    instructions=(
        "Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant."
    ),
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
)


async def main():
    # Trace the entire run as a single workflow
    with trace(workflow_name="Message filtering"):
        # 1. Send a regular message to the first agent
        result = await Runner.run(first_agent, input="Hi, my name is Sora.")

        print("Step 1 done")

        # 2. Ask it to generate a number
        result = await Runner.run(
            first_agent,
            input=result.to_input_list()
            + [{"content": "Can you generate a random number between 0 and 100?", "role": "user"}],
        )

        print("Step 2 done")

        # 3. Call the second agent
        result = await Runner.run(
            second_agent,
            input=result.to_input_list()
            + [
                {
                    "content": "I live in New York City. Whats the population of the city?",
                    "role": "user",
                }
            ],
        )

        print("Step 3 done")

        # 4. Cause a handoff to occur
        result = await Runner.run(
            second_agent,
            input=result.to_input_list()
            + [
                {
                    "content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?",
                    "role": "user",
                }
            ],
        )

        print("Step 4 done")

    print("\n===Final messages===\n")

    # 5. That should have caused spanish_handoff_message_filter to be called, which means the
    # output should be missing the first two messages, and have no tool calls.
    # Let's print the messages to see what happened
    for message in result.to_input_list():
        print(json.dumps(message, indent=2))
        # tool_calls = message.tool_calls if isinstance(message, AssistantMessage) else None

        # print(f"{message.role}: {message.content}\n  - Tool calls: {tool_calls or 'None'}")
        """
        $python examples/handoffs/message_filter.py
        Step 1 done
        Step 2 done
        Step 3 done
        Step 4 done

        ===Final messages===

        {
            "content": "Can you generate a random number between 0 and 100?",
            "role": "user"
        }
        {
        "id": "...",
        "content": [
            {
            "annotations": [],
            "text": "Sure! Here's a random number between 0 and 100: **42**.",
            "type": "output_text"
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
        }
        {
        "content": "I live in New York City. Whats the population of the city?",
        "role": "user"
        }
        {
        "id": "...",
        "content": [
            {
            "annotations": [],
            "text": "As of the most recent estimates, the population of New York City is approximately 8.6 million people. However, this number is constantly changing due to various factors such as migration and birth rates. For the latest and most accurate information, it's always a good idea to check the official data from sources like the U.S. Census Bureau.",
            "type": "output_text"
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
        }
        {
        "content": "Por favor habla en espa\u00f1ol. \u00bfCu\u00e1l es mi nombre y d\u00f3nde vivo?",
        "role": "user"
        }
        {
        "id": "...",
        "content": [
            {
            "annotations": [],
            "text": "No tengo acceso a esa informaci\u00f3n personal, solo s\u00e9 lo que me has contado: vives en Nueva York.",
            "type": "output_text"
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
        }
        """


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


<file_info>
path: src/agents/util/_json.py
name: _json.py
</file_info>
from __future__ import annotations

from typing import Literal

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeVar

from ..exceptions import ModelBehaviorError
from ..tracing import SpanError
from ._error_tracing import attach_error_to_current_span

T = TypeVar("T")


def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool) -> T:
    partial_setting: bool | Literal["off", "on", "trailing-strings"] = (
        "trailing-strings" if partial else False
    )
    try:
        validated = type_adapter.validate_json(json_str, experimental_allow_partial=partial_setting)
        return validated
    except ValidationError as e:
        attach_error_to_current_span(
            SpanError(
                message="Invalid JSON provided",
                data={},
            )
        )
        raise ModelBehaviorError(
            f"Invalid JSON when parsing {json_str} for {type_adapter}; {e}"
        ) from e


<file_info>
path: src/agents/function_schema.py
name: function_schema.py
</file_info>
from __future__ import annotations

import contextlib
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, Field, create_model

from .exceptions import UserError
from .run_context import RunContextWrapper
from .strict_schema import ensure_strict_json_schema


@dataclass
class FuncSchema:
    """
    Captures the schema for a python function, in preparation for sending it to an LLM as a tool.
    """

    name: str
    """The name of the function."""
    description: str | None
    """The description of the function."""
    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""
    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""
    signature: inspect.Signature
    """The signature of the function."""
    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""
    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input."""

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
        the original function.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False

        # Use enumerate() so we can skip the first parameter if it's context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            # If the function takes a RunContextWrapper and this is the first parameter, skip it.
            if self.takes_context and idx == 0:
                continue

            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
        return positional_args, keyword_args


@dataclass
class FuncDocumentation:
    """Contains metadata about a python function, extracted from its docstring."""

    name: str
    """The name of the function, via `__name__`."""
    description: str | None
    """The description of the function, derived from the docstring."""
    param_descriptions: dict[str, str] | None
    """The parameter descriptions of the function, derived from the docstring."""


DocstringStyle = Literal["google", "numpy", "sphinx"]


# As of Feb 2025, the automatic style detection in griffe is an Insiders feature. This
# code approximates it.
def _detect_docstring_style(doc: str) -> DocstringStyle:
    scores: dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters', 'Returns', or 'Yields' followed by
    # a dashed underline
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    # Priority order: sphinx > numpy > google in case of tie
    styles: list[DocstringStyle] = ["sphinx", "numpy", "google"]

    for style in styles:
        if scores[style] == max_score:
            return style

    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    # Supresses warnings about missing annotations for params
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, we will attempt to
            auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=name, description=None, param_descriptions=None)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )

    param_descriptions: dict[str, str] = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FuncDocumentation(
        name=func.__name__,
        description=description,
        param_descriptions=param_descriptions or None,
    )


def function_schema(
    func: Callable[..., Any],
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """
    Given a python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided, we will
            attempt to auto-detect the style.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one derived from the
            docstring.
        use_docstring_info: If True, uses the docstring to generate the description and parameter
            descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll ensure that
            the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**
            recommend setting this to True, as it increases the likelihood of the LLM providing
            correct JSON input.

    Returns:
        A `FuncSchema` object containing the function's name, description, parameter descriptions,
        and other metadata.
    """

    # 1. Grab docstring info
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = doc_info.param_descriptions or {}
    else:
        doc_info = None
        param_descs = {}

    func_name = name_override or doc_info.name if doc_info else func.__name__

    # 2. Inspect function signature and get type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    if params:
        first_name, first_param = params[0]
        # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper:
                takes_context = True  # Mark that the function takes context
            else:
                filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))

    # For parameters other than the first, raise error if any use RunContextWrapper.
    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper:
                raise UserError(
                    f"RunContextWrapper param found at non-first position in function"
                    f" {func.__name__}"
                )
        filtered_params.append((name, param))

    # We will collect field definitions for create_model as a dict:
    #   field_name -> (type_annotation, default_value_or_Field(...))
    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default

        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any

        # If a docstring param description exists, use it
        field_description = param_descs.get(name, None)

        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = list[ann]  # type: ignore

            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),  # type: ignore
            )

        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = dict[str, ann]  # type: ignore

            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),  # type: ignore
            )

        else:
            # Normal parameter
            if default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )

    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)

    # 4. Build JSON schema from that model
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # 5. Return as a FuncSchema dataclass
    return FuncSchema(
        name=func_name,
        description=description_override or doc_info.description if doc_info else None,
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )


<file_info>
path: tests/test_openai_chatcompletions_converter.py
name: test_openai_chatcompletions_converter.py
</file_info>
# Copyright (c) OpenAI
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Unit tests for the internal `_Converter` class defined in
`agents.models.openai_chatcompletions`. The converter is responsible for
translating between internal "item" structures (e.g., `ResponseOutputMessage`
and related types from `openai.types.responses`) and the ChatCompletion message
structures defined by the OpenAI client library.

These tests exercise both conversion directions:

- `_Converter.message_to_output_items` turns a `ChatCompletionMessage` (as
  returned by the OpenAI API) into a list of `ResponseOutputItem` instances.

- `_Converter.items_to_messages` takes in either a simple string prompt, or a
  list of input/output items such as `ResponseOutputMessage` and
  `ResponseFunctionToolCallParam` dicts, and constructs a list of
  `ChatCompletionMessageParam` dicts suitable for sending back to the API.
"""

from __future__ import annotations

from typing import Literal, cast

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agents.agent_output import AgentOutputSchema
from agents.exceptions import UserError
from agents.items import TResponseInputItem
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.openai_chatcompletions import _Converter


def test_message_to_output_items_with_text_only():
    """
    Make sure a simple ChatCompletionMessage with string content is converted
    into a single ResponseOutputMessage containing one ResponseOutputText.
    """
    msg = ChatCompletionMessage(role="assistant", content="Hello")
    items = _Converter.message_to_output_items(msg)
    # Expect exactly one output item (the message)
    assert len(items) == 1
    message_item = cast(ResponseOutputMessage, items[0])
    assert message_item.id == FAKE_RESPONSES_ID
    assert message_item.role == "assistant"
    assert message_item.type == "message"
    assert message_item.status == "completed"
    # Message content should have exactly one text part with the same text.
    assert len(message_item.content) == 1
    text_part = cast(ResponseOutputText, message_item.content[0])
    assert text_part.type == "output_text"
    assert text_part.text == "Hello"


def test_message_to_output_items_with_refusal():
    """
    Make sure a message with a refusal string produces a ResponseOutputMessage
    with a ResponseOutputRefusal content part.
    """
    msg = ChatCompletionMessage(role="assistant", refusal="I'm sorry")
    items = _Converter.message_to_output_items(msg)
    assert len(items) == 1
    message_item = cast(ResponseOutputMessage, items[0])
    assert len(message_item.content) == 1
    refusal_part = cast(ResponseOutputRefusal, message_item.content[0])
    assert refusal_part.type == "refusal"
    assert refusal_part.refusal == "I'm sorry"


def test_message_to_output_items_with_tool_call():
    """
    If the ChatCompletionMessage contains one or more tool_calls, they should
    be reflected as separate `ResponseFunctionToolCall` items appended after
    the message item.
    """
    tool_call = ChatCompletionMessageToolCall(
        id="tool1",
        type="function",
        function=Function(name="myfn", arguments='{"x":1}'),
    )
    msg = ChatCompletionMessage(role="assistant", content="Hi", tool_calls=[tool_call])
    items = _Converter.message_to_output_items(msg)
    # Should produce a message item followed by one function tool call item
    assert len(items) == 2
    message_item = cast(ResponseOutputMessage, items[0])
    assert isinstance(message_item, ResponseOutputMessage)
    fn_call_item = cast(ResponseFunctionToolCall, items[1])
    assert fn_call_item.id == FAKE_RESPONSES_ID
    assert fn_call_item.call_id == tool_call.id
    assert fn_call_item.name == tool_call.function.name
    assert fn_call_item.arguments == tool_call.function.arguments
    assert fn_call_item.type == "function_call"


def test_items_to_messages_with_string_user_content():
    """
    A simple string as the items argument should be converted into a user
    message param dict with the same content.
    """
    result = _Converter.items_to_messages("Ask me anything")
    assert isinstance(result, list)
    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "user"
    assert msg["content"] == "Ask me anything"


def test_items_to_messages_with_easy_input_message():
    """
    Given an easy input message dict (just role/content), the converter should
    produce the appropriate ChatCompletionMessageParam with the same content.
    """
    items: list[TResponseInputItem] = [
        {
            "role": "user",
            "content": "How are you?",
        }
    ]
    messages = _Converter.items_to_messages(items)
    assert len(messages) == 1
    out = messages[0]
    assert out["role"] == "user"
    # For simple string inputs, the converter returns the content as a bare string
    assert out["content"] == "How are you?"


def test_items_to_messages_with_output_message_and_function_call():
    """
    Given a sequence of one ResponseOutputMessageParam followed by a
    ResponseFunctionToolCallParam, the converter should produce a single
    ChatCompletionAssistantMessageParam that includes both the assistant's
    textual content and a populated `tool_calls` reflecting the function call.
    """
    # Construct output message param dict with two content parts.
    output_text: ResponseOutputText = ResponseOutputText(
        text="Part 1",
        type="output_text",
        annotations=[],
    )
    refusal: ResponseOutputRefusal = ResponseOutputRefusal(
        refusal="won't do that",
        type="refusal",
    )
    resp_msg: ResponseOutputMessage = ResponseOutputMessage(
        id="42",
        type="message",
        role="assistant",
        status="completed",
        content=[output_text, refusal],
    )
    # Construct a function call item dict (as if returned from model)
    func_item: ResponseFunctionToolCallParam = {
        "id": "99",
        "call_id": "abc",
        "name": "math",
        "arguments": "{}",
        "type": "function_call",
    }
    items: list[TResponseInputItem] = [
        resp_msg.model_dump(),  # type:ignore
        func_item,
    ]
    messages = _Converter.items_to_messages(items)
    # Should return a single assistant message
    assert len(messages) == 1
    assistant = messages[0]
    assert assistant["role"] == "assistant"
    # Content combines text portions of the output message
    assert "content" in assistant
    assert assistant["content"] == "Part 1"
    # Refusal in output message should be represented in assistant message
    assert "refusal" in assistant
    assert assistant["refusal"] == refusal.refusal
    # Tool calls list should contain one ChatCompletionMessageToolCall dict
    tool_calls = assistant.get("tool_calls")
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "math"
    assert tool_call["function"]["arguments"] == "{}"


def test_convert_tool_choice_handles_standard_and_named_options() -> None:
    """
    The `_Converter.convert_tool_choice` method should return NOT_GIVEN
    if no choice is provided, pass through values like "auto", "required",
    or "none" unchanged, and translate any other string into a function
    selection dict.
    """
    assert _Converter.convert_tool_choice(None).__class__.__name__ == "NotGiven"
    assert _Converter.convert_tool_choice("auto") == "auto"
    assert _Converter.convert_tool_choice("required") == "required"
    assert _Converter.convert_tool_choice("none") == "none"
    tool_choice_dict = _Converter.convert_tool_choice("mytool")
    assert isinstance(tool_choice_dict, dict)
    assert tool_choice_dict["type"] == "function"
    assert tool_choice_dict["function"]["name"] == "mytool"


def test_convert_response_format_returns_not_given_for_plain_text_and_dict_for_schemas() -> None:
    """
    The `_Converter.convert_response_format` method should return NOT_GIVEN
    when no output schema is provided or if the output schema indicates
    plain text. For structured output schemas, it should return a dict
    with type `json_schema` and include the generated JSON schema and
    strict flag from the provided `AgentOutputSchema`.
    """
    # when output is plain text (schema None or output_type str), do not include response_format
    assert _Converter.convert_response_format(None).__class__.__name__ == "NotGiven"
    assert (
        _Converter.convert_response_format(AgentOutputSchema(str)).__class__.__name__ == "NotGiven"
    )
    # For e.g. integer output, we expect a response_format dict
    schema = AgentOutputSchema(int)
    resp_format = _Converter.convert_response_format(schema)
    assert isinstance(resp_format, dict)
    assert resp_format["type"] == "json_schema"
    assert resp_format["json_schema"]["name"] == "final_output"
    assert "strict" in resp_format["json_schema"]
    assert resp_format["json_schema"]["strict"] == schema.strict_json_schema
    assert "schema" in resp_format["json_schema"]
    assert resp_format["json_schema"]["schema"] == schema.json_schema()


def test_items_to_messages_with_function_output_item():
    """
    A function call output item should be converted into a tool role message
    dict with the appropriate tool_call_id and content.
    """
    func_output_item: FunctionCallOutput = {
        "type": "function_call_output",
        "call_id": "somecall",
        "output": '{"foo": "bar"}',
    }
    messages = _Converter.items_to_messages([func_output_item])
    assert len(messages) == 1
    tool_msg = messages[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == func_output_item["call_id"]
    assert tool_msg["content"] == func_output_item["output"]


def test_extract_all_and_text_content_for_strings_and_lists():
    """
    The converter provides helpers for extracting user-supplied message content
    either as a simple string or as a list of `input_text` dictionaries.
    When passed a bare string, both `extract_all_content` and
    `extract_text_content` should return the string unchanged.
    When passed a list of input dictionaries, `extract_all_content` should
    produce a list of `ChatCompletionContentPart` dicts, and `extract_text_content`
    should filter to only the textual parts.
    """
    prompt = "just text"
    assert _Converter.extract_all_content(prompt) == prompt
    assert _Converter.extract_text_content(prompt) == prompt
    text1: ResponseInputTextParam = {"type": "input_text", "text": "one"}
    text2: ResponseInputTextParam = {"type": "input_text", "text": "two"}
    all_parts = _Converter.extract_all_content([text1, text2])
    assert isinstance(all_parts, list)
    assert len(all_parts) == 2
    assert all_parts[0]["type"] == "text" and all_parts[0]["text"] == "one"
    assert all_parts[1]["type"] == "text" and all_parts[1]["text"] == "two"
    text_parts = _Converter.extract_text_content([text1, text2])
    assert isinstance(text_parts, list)
    assert all(p["type"] == "text" for p in text_parts)
    assert [p["text"] for p in text_parts] == ["one", "two"]


def test_items_to_messages_handles_system_and_developer_roles():
    """
    Roles other than `user` (e.g. `system` and `developer`) need to be
    converted appropriately whether provided as simple dicts or as full
    `message` typed dicts.
    """
    sys_items: list[TResponseInputItem] = [{"role": "system", "content": "setup"}]
    sys_msgs = _Converter.items_to_messages(sys_items)
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["role"] == "system"
    assert sys_msgs[0]["content"] == "setup"
    dev_items: list[TResponseInputItem] = [{"role": "developer", "content": "debug"}]
    dev_msgs = _Converter.items_to_messages(dev_items)
    assert len(dev_msgs) == 1
    assert dev_msgs[0]["role"] == "developer"
    assert dev_msgs[0]["content"] == "debug"


def test_maybe_input_message_allows_message_typed_dict():
    """
    The `_Converter.maybe_input_message` should recognize a dict with
    "type": "message" and a supported role as an input message. Ensure
    that such dicts are passed through by `items_to_messages`.
    """
    # Construct a dict with the proper required keys for a ResponseInputParam.Message
    message_dict: TResponseInputItem = {
        "type": "message",
        "role": "user",
        "content": "hi",
    }
    assert _Converter.maybe_input_message(message_dict) is not None
    # items_to_messages should process this correctly
    msgs = _Converter.items_to_messages([message_dict])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"


def test_tool_call_conversion():
    """
    Test that tool calls are converted correctly.
    """
    function_call = ResponseFunctionToolCallParam(
        id="tool1",
        call_id="abc",
        name="math",
        arguments="{}",
        type="function_call",
    )

    messages = _Converter.items_to_messages([function_call])
    assert len(messages) == 1
    tool_msg = messages[0]
    assert tool_msg["role"] == "assistant"
    assert tool_msg.get("content") is None
    tool_calls = list(tool_msg.get("tool_calls", []))
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["id"] == function_call["call_id"]
    assert tool_call["function"]["name"] == function_call["name"]
    assert tool_call["function"]["arguments"] == function_call["arguments"]


@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_input_message_with_all_roles(role: str):
    """
    The `_Converter.maybe_input_message` should recognize a dict with
    "type": "message" and a supported role as an input message. Ensure
    that such dicts are passed through by `items_to_messages`.
    """
    # Construct a dict with the proper required keys for a ResponseInputParam.Message
    casted_role = cast(Literal["user", "system", "developer"], role)
    message_dict: TResponseInputItem = {
        "type": "message",
        "role": casted_role,
        "content": "hi",
    }
    assert _Converter.maybe_input_message(message_dict) is not None
    # items_to_messages should process this correctly
    msgs = _Converter.items_to_messages([message_dict])
    assert len(msgs) == 1
    assert msgs[0]["role"] == casted_role
    assert msgs[0]["content"] == "hi"


def test_item_reference_errors():
    """
    Test that item references are converted correctly.
    """
    with pytest.raises(UserError):
        _Converter.items_to_messages(
            [
                {
                    "type": "item_reference",
                    "id": "item1",
                }
            ]
        )


class TestObject:
    pass


def test_unknown_object_errors():
    """
    Test that unknown objects are converted correctly.
    """
    with pytest.raises(UserError, match="Unhandled item type or structure"):
        # Purposely ignore the type error
        _Converter.items_to_messages([TestObject()])  # type: ignore


def test_assistant_messages_in_history():
    """
    Test that assistant messages are added to the history.
    """
    messages = _Converter.items_to_messages(
        [
            {
                "role": "user",
                "content": "Hello",
            },
            {
                "role": "assistant",
                "content": "Hello?",
            },
            {
                "role": "user",
                "content": "What was my Name?",
            },
        ]
    )

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello?"},
        {"role": "user", "content": "What was my Name?"},
    ]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello?"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "What was my Name?"


<file_info>
path: tests/test_function_tool_decorator.py
name: test_function_tool_decorator.py
</file_info>
import asyncio
import json
from typing import Any, Optional

import pytest

from agents import function_tool
from agents.run_context import RunContextWrapper


class DummyContext:
    def __init__(self):
        self.data = "something"


def ctx_wrapper() -> RunContextWrapper[DummyContext]:
    return RunContextWrapper(DummyContext())


@function_tool
def sync_no_context_no_args() -> str:
    return "test_1"


@pytest.mark.asyncio
async def test_sync_no_context_no_args_invocation():
    tool = sync_no_context_no_args
    output = await tool.on_invoke_tool(ctx_wrapper(), "")
    assert output == "test_1"


@function_tool
def sync_no_context_with_args(a: int, b: int) -> int:
    return a + b


@pytest.mark.asyncio
async def test_sync_no_context_with_args_invocation():
    tool = sync_no_context_with_args
    input_data = {"a": 5, "b": 7}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert int(output) == 12


@function_tool
def sync_with_context(ctx: RunContextWrapper[DummyContext], name: str) -> str:
    return f"{name}_{ctx.context.data}"


@pytest.mark.asyncio
async def test_sync_with_context_invocation():
    tool = sync_with_context
    input_data = {"name": "Alice"}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "Alice_something"


@function_tool
async def async_no_context(a: int, b: int) -> int:
    await asyncio.sleep(0)  # Just to illustrate async
    return a * b


@pytest.mark.asyncio
async def test_async_no_context_invocation():
    tool = async_no_context
    input_data = {"a": 3, "b": 4}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert int(output) == 12


@function_tool
async def async_with_context(ctx: RunContextWrapper[DummyContext], prefix: str, num: int) -> str:
    await asyncio.sleep(0)
    return f"{prefix}-{num}-{ctx.context.data}"


@pytest.mark.asyncio
async def test_async_with_context_invocation():
    tool = async_with_context
    input_data = {"prefix": "Value", "num": 42}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "Value-42-something"


@function_tool(name_override="my_custom_tool", description_override="custom desc")
def sync_no_context_override() -> str:
    return "override_result"


@pytest.mark.asyncio
async def test_sync_no_context_override_invocation():
    tool = sync_no_context_override
    assert tool.name == "my_custom_tool"
    assert tool.description == "custom desc"
    output = await tool.on_invoke_tool(ctx_wrapper(), "")
    assert output == "override_result"


@function_tool(failure_error_function=None)
def will_fail_on_bad_json(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_error_on_invalid_json():
    tool = will_fail_on_bad_json
    # Passing an invalid JSON string
    with pytest.raises(Exception) as exc_info:
        await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert "Invalid JSON input for tool" in str(exc_info.value)


def sync_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    return f"error_{error.__class__.__name__}"


@function_tool(failure_error_function=sync_error_handler)
def will_not_fail_on_bad_json(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_no_error_on_invalid_json():
    tool = will_not_fail_on_bad_json
    # Passing an invalid JSON string
    result = await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert result == "error_ModelBehaviorError"


def async_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    return f"error_{error.__class__.__name__}"


@function_tool(failure_error_function=sync_error_handler)
def will_not_fail_on_bad_json_async(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_no_error_on_invalid_json_async():
    tool = will_not_fail_on_bad_json_async
    result = await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert result == "error_ModelBehaviorError"


@function_tool(strict_mode=False)
def optional_param_function(a: int, b: Optional[int] = None) -> str:
    if b is None:
        return f"{a}_no_b"
    return f"{a}_{b}"


@pytest.mark.asyncio
async def test_non_strict_mode_function():
    tool = optional_param_function

    assert tool.strict_json_schema is False, "strict_json_schema should be False"

    assert tool.params_json_schema.get("required") == ["a"], "required should only be a"

    input_data = {"a": 5}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "5_no_b"

    input_data = {"a": 5, "b": 10}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "5_10"


@function_tool(strict_mode=False)
def all_optional_params_function(
    x: int = 42,
    y: str = "hello",
    z: Optional[int] = None,
) -> str:
    if z is None:
        return f"{x}_{y}_no_z"
    return f"{x}_{y}_{z}"


@pytest.mark.asyncio
async def test_all_optional_params_function():
    tool = all_optional_params_function

    assert tool.strict_json_schema is False, "strict_json_schema should be False"

    assert tool.params_json_schema.get("required") is None, "required should be empty"

    input_data: dict[str, Any] = {}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "42_hello_no_z"

    input_data = {"x": 10, "y": "world"}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "10_world_no_z"

    input_data = {"x": 10, "y": "world", "z": 99}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "10_world_99"


<file_info>
path: tests/test_handoff_tool.py
name: test_handoff_tool.py
</file_info>
from typing import Any

import pytest
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel

from agents import (
    Agent,
    Handoff,
    HandoffInputData,
    MessageOutputItem,
    ModelBehaviorError,
    RunContextWrapper,
    Runner,
    UserError,
    handoff,
)


def message_item(content: str, agent: Agent[Any]) -> MessageOutputItem:
    return MessageOutputItem(
        agent=agent,
        raw_item=ResponseOutputMessage(
            id="123",
            status="completed",
            role="assistant",
            type="message",
            content=[ResponseOutputText(text=content, type="output_text", annotations=[])],
        ),
    )


def get_len(data: HandoffInputData) -> int:
    input_len = len(data.input_history) if isinstance(data.input_history, tuple) else 1
    pre_handoff_len = len(data.pre_handoff_items)
    new_items_len = len(data.new_items)
    return input_len + pre_handoff_len + new_items_len


def test_single_handoff_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2", handoffs=[agent_1])

    assert not agent_1.handoffs
    assert agent_2.handoffs == [agent_1]

    assert not Runner._get_handoffs(agent_1)

    handoff_objects = Runner._get_handoffs(agent_2)
    assert len(handoff_objects) == 1
    obj = handoff_objects[0]
    assert obj.tool_name == Handoff.default_tool_name(agent_1)
    assert obj.tool_description == Handoff.default_tool_description(agent_1)
    assert obj.agent_name == agent_1.name


def test_multiple_handoffs_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])

    assert agent_3.handoffs == [agent_1, agent_2]
    assert not agent_1.handoffs
    assert not agent_2.handoffs

    handoff_objects = Runner._get_handoffs(agent_3)
    assert len(handoff_objects) == 2
    assert handoff_objects[0].tool_name == Handoff.default_tool_name(agent_1)
    assert handoff_objects[1].tool_name == Handoff.default_tool_name(agent_2)

    assert handoff_objects[0].tool_description == Handoff.default_tool_description(agent_1)
    assert handoff_objects[1].tool_description == Handoff.default_tool_description(agent_2)

    assert handoff_objects[0].agent_name == agent_1.name
    assert handoff_objects[1].agent_name == agent_2.name


def test_custom_handoff_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(
        name="test_3",
        handoffs=[
            agent_1,
            handoff(
                agent_2,
                tool_name_override="custom_tool_name",
                tool_description_override="custom tool description",
            ),
        ],
    )

    assert len(agent_3.handoffs) == 2
    assert not agent_1.handoffs
    assert not agent_2.handoffs

    handoff_objects = Runner._get_handoffs(agent_3)
    assert len(handoff_objects) == 2

    first_handoff = handoff_objects[0]
    assert isinstance(first_handoff, Handoff)
    assert first_handoff.tool_name == Handoff.default_tool_name(agent_1)
    assert first_handoff.tool_description == Handoff.default_tool_description(agent_1)
    assert first_handoff.agent_name == agent_1.name

    second_handoff = handoff_objects[1]
    assert isinstance(second_handoff, Handoff)
    assert second_handoff.tool_name == "custom_tool_name"
    assert second_handoff.tool_description == "custom tool description"
    assert second_handoff.agent_name == agent_2.name


class Foo(BaseModel):
    bar: str


@pytest.mark.asyncio
async def test_handoff_input_type():
    async def _on_handoff(ctx: RunContextWrapper[Any], input: Foo):
        pass

    agent = Agent(name="test")
    obj = handoff(agent, input_type=Foo, on_handoff=_on_handoff)
    for key, value in Foo.model_json_schema().items():
        assert obj.input_json_schema[key] == value

    # Invalid JSON should raise an error
    with pytest.raises(ModelBehaviorError):
        await obj.on_invoke_handoff(RunContextWrapper(agent), "not json")

    # Empty JSON should raise an error
    with pytest.raises(ModelBehaviorError):
        await obj.on_invoke_handoff(RunContextWrapper(agent), "")

    # Valid JSON should call the on_handoff function
    invoked = await obj.on_invoke_handoff(
        RunContextWrapper(agent), Foo(bar="baz").model_dump_json()
    )
    assert invoked == agent


@pytest.mark.asyncio
async def test_on_handoff_called():
    was_called = False

    async def _on_handoff(ctx: RunContextWrapper[Any], input: Foo):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, input_type=Foo, on_handoff=_on_handoff)
    for key, value in Foo.model_json_schema().items():
        assert obj.input_json_schema[key] == value

    invoked = await obj.on_invoke_handoff(
        RunContextWrapper(agent), Foo(bar="baz").model_dump_json()
    )
    assert invoked == agent

    assert was_called, "on_handoff should have been called"


@pytest.mark.asyncio
async def test_on_handoff_without_input_called():
    was_called = False

    def _on_handoff(ctx: RunContextWrapper[Any]):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, on_handoff=_on_handoff)

    invoked = await obj.on_invoke_handoff(RunContextWrapper(agent), "")
    assert invoked == agent

    assert was_called, "on_handoff should have been called"


@pytest.mark.asyncio
async def test_async_on_handoff_without_input_called():
    was_called = False

    async def _on_handoff(ctx: RunContextWrapper[Any]):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, on_handoff=_on_handoff)

    invoked = await obj.on_invoke_handoff(RunContextWrapper(agent), "")
    assert invoked == agent

    assert was_called, "on_handoff should have been called"


@pytest.mark.asyncio
async def test_invalid_on_handoff_raises_error():
    was_called = False

    async def _on_handoff(ctx: RunContextWrapper[Any], blah: str):
        nonlocal was_called
        was_called = True  # pragma: no cover

    agent = Agent(name="test")

    with pytest.raises(UserError):
        # Purposely ignoring the type error here to simulate invalid input
        handoff(agent, on_handoff=_on_handoff)  # type: ignore


def test_handoff_input_data():
    agent = Agent(name="test")

    data = HandoffInputData(
        input_history="",
        pre_handoff_items=(),
        new_items=(),
    )
    assert get_len(data) == 1

    data = HandoffInputData(
        input_history=({"role": "user", "content": "foo"},),
        pre_handoff_items=(),
        new_items=(),
    )
    assert get_len(data) == 1

    data = HandoffInputData(
        input_history=(
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ),
        pre_handoff_items=(),
        new_items=(),
    )
    assert get_len(data) == 2

    data = HandoffInputData(
        input_history=({"role": "user", "content": "foo"},),
        pre_handoff_items=(
            message_item("foo", agent),
            message_item("foo2", agent),
        ),
        new_items=(
            message_item("bar", agent),
            message_item("baz", agent),
        ),
    )
    assert get_len(data) == 5

    data = HandoffInputData(
        input_history=(
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ),
        pre_handoff_items=(message_item("baz", agent),),
        new_items=(
            message_item("baz", agent),
            message_item("qux", agent),
        ),
    )

    assert get_len(data) == 5


def test_handoff_input_schema_is_strict():
    agent = Agent(name="test")
    obj = handoff(agent, input_type=Foo, on_handoff=lambda ctx, input: None)
    for key, value in Foo.model_json_schema().items():
        assert obj.input_json_schema[key] == value

    assert obj.strict_json_schema, "Input schema should be strict"

    assert (
        "additionalProperties" in obj.input_json_schema
        and not obj.input_json_schema["additionalProperties"]
    ), "Input schema should be strict and have additionalProperties=False"


<file_info>
path: examples/financial_research_agent/agents/search_agent.py
name: search_agent.py
</file_info>
from agents import Agent, WebSearchTool
from agents.model_settings import ModelSettings

# Given a search term, use web search to pull back a brief summary.
# Summaries should be concise but capture the main financial points.
INSTRUCTIONS = (
    "You are a research assistant specializing in financial topics. "
    "Given a search term, use web search to retrieve up‑to‑date context and "
    "produce a short summary of at most 300 words. Focus on key numbers, events, "
    "or quotes that will be useful to a financial analyst."
)

search_agent = Agent(
    name="FinancialSearchAgent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)


<file_info>
path: examples/research_bot/agents/writer_agent.py
name: writer_agent.py
</file_info>
# Agent used to synthesize a final report from the individual summaries.
from pydantic import BaseModel

from agents import Agent

PROMPT = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""

    markdown_report: str
    """The final report"""

    follow_up_questions: list[str]
    """Suggested topics to research further"""


writer_agent = Agent(
    name="WriterAgent",
    instructions=PROMPT,
    model="o3-mini",
    output_type=ReportData,
)


<file_info>
path: examples/voice/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/model_settings.py
name: model_settings.py
</file_info>
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelSettings:
    """Settings to use when calling an LLM.

    This class holds optional model configuration parameters (e.g. temperature,
    top_p, penalties, truncation, etc.).

    Not all models/providers support all of these parameters, so please check the API documentation
    for the specific model and provider you are using.
    """

    temperature: float | None = None
    """The temperature to use when calling the model."""

    top_p: float | None = None
    """The top_p to use when calling the model."""

    frequency_penalty: float | None = None
    """The frequency penalty to use when calling the model."""

    presence_penalty: float | None = None
    """The presence penalty to use when calling the model."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """The tool choice to use when calling the model."""

    parallel_tool_calls: bool | None = False
    """Whether to use parallel tool calls when calling the model."""

    truncation: Literal["auto", "disabled"] | None = None
    """The truncation strategy to use when calling the model."""

    max_tokens: int | None = None
    """The maximum number of output tokens to generate."""

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        """Produce a new ModelSettings by overlaying any non-None values from the
        override on top of this instance."""
        if override is None:
            return self
        return ModelSettings(
            temperature=override.temperature or self.temperature,
            top_p=override.top_p or self.top_p,
            frequency_penalty=override.frequency_penalty or self.frequency_penalty,
            presence_penalty=override.presence_penalty or self.presence_penalty,
            tool_choice=override.tool_choice or self.tool_choice,
            parallel_tool_calls=override.parallel_tool_calls or self.parallel_tool_calls,
            truncation=override.truncation or self.truncation,
            max_tokens=override.max_tokens or self.max_tokens,
        )


<file_info>
path: src/agents/__init__.py
name: __init__.py
</file_info>
import logging
import sys
from typing import Literal

from openai import AsyncOpenAI

from . import _config
from .agent import Agent, ToolsToFinalOutputFunction, ToolsToFinalOutputResult
from .agent_output import AgentOutputSchema
from .computer import AsyncComputer, Button, Computer, Environment
from .exceptions import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    UserError,
)
from .guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
    input_guardrail,
    output_guardrail,
)
from .handoffs import Handoff, HandoffInputData, HandoffInputFilter, handoff
from .items import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    ReasoningItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from .lifecycle import AgentHooks, RunHooks
from .model_settings import ModelSettings
from .models.interface import Model, ModelProvider, ModelTracing
from .models.openai_chatcompletions import OpenAIChatCompletionsModel
from .models.openai_provider import OpenAIProvider
from .models.openai_responses import OpenAIResponsesModel
from .result import RunResult, RunResultStreaming
from .run import RunConfig, Runner
from .run_context import RunContextWrapper, TContext
from .stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
)
from .tool import (
    ComputerTool,
    FileSearchTool,
    FunctionTool,
    FunctionToolResult,
    Tool,
    WebSearchTool,
    default_tool_error_function,
    function_tool,
)
from .tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    Span,
    SpanData,
    SpanError,
    SpeechGroupSpanData,
    SpeechSpanData,
    Trace,
    TracingProcessor,
    TranscriptionSpanData,
    add_trace_processor,
    agent_span,
    custom_span,
    function_span,
    gen_span_id,
    gen_trace_id,
    generation_span,
    get_current_span,
    get_current_trace,
    guardrail_span,
    handoff_span,
    set_trace_processors,
    set_tracing_disabled,
    set_tracing_export_api_key,
    speech_group_span,
    speech_span,
    trace,
    transcription_span,
)
from .usage import Usage


def set_default_openai_key(key: str, use_for_tracing: bool = True) -> None:
    """Set the default OpenAI API key to use for LLM requests (and optionally tracing(). This is
    only necessary if the OPENAI_API_KEY environment variable is not already set.

    If provided, this key will be used instead of the OPENAI_API_KEY environment variable.

    Args:
        key: The OpenAI key to use.
        use_for_tracing: Whether to also use this key to send traces to OpenAI. Defaults to True
            If False, you'll either need to set the OPENAI_API_KEY environment variable or call
            set_tracing_export_api_key() with the API key you want to use for tracing.
    """
    _config.set_default_openai_key(key, use_for_tracing)


def set_default_openai_client(client: AsyncOpenAI, use_for_tracing: bool = True) -> None:
    """Set the default OpenAI client to use for LLM requests and/or tracing. If provided, this
    client will be used instead of the default OpenAI client.

    Args:
        client: The OpenAI client to use.
        use_for_tracing: Whether to use the API key from this client for uploading traces. If False,
            you'll either need to set the OPENAI_API_KEY environment variable or call
            set_tracing_export_api_key() with the API key you want to use for tracing.
    """
    _config.set_default_openai_client(client, use_for_tracing)


def set_default_openai_api(api: Literal["chat_completions", "responses"]) -> None:
    """Set the default API to use for OpenAI LLM requests. By default, we will use the responses API
    but you can set this to use the chat completions API instead.
    """
    _config.set_default_openai_api(api)


def enable_verbose_stdout_logging():
    """Enables verbose logging to stdout. This is useful for debugging."""
    logger = logging.getLogger("openai.agents")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))


__all__ = [
    "Agent",
    "ToolsToFinalOutputFunction",
    "ToolsToFinalOutputResult",
    "Runner",
    "Model",
    "ModelProvider",
    "ModelTracing",
    "ModelSettings",
    "OpenAIChatCompletionsModel",
    "OpenAIProvider",
    "OpenAIResponsesModel",
    "AgentOutputSchema",
    "Computer",
    "AsyncComputer",
    "Environment",
    "Button",
    "AgentsException",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "UserError",
    "InputGuardrail",
    "InputGuardrailResult",
    "OutputGuardrail",
    "OutputGuardrailResult",
    "GuardrailFunctionOutput",
    "input_guardrail",
    "output_guardrail",
    "handoff",
    "Handoff",
    "HandoffInputData",
    "HandoffInputFilter",
    "TResponseInputItem",
    "MessageOutputItem",
    "ModelResponse",
    "RunItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "ReasoningItem",
    "ModelResponse",
    "ItemHelpers",
    "RunHooks",
    "AgentHooks",
    "RunContextWrapper",
    "TContext",
    "RunResult",
    "RunResultStreaming",
    "RunConfig",
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "AgentUpdatedStreamEvent",
    "StreamEvent",
    "FunctionTool",
    "FunctionToolResult",
    "ComputerTool",
    "FileSearchTool",
    "Tool",
    "WebSearchTool",
    "function_tool",
    "Usage",
    "add_trace_processor",
    "agent_span",
    "custom_span",
    "function_span",
    "generation_span",
    "get_current_span",
    "get_current_trace",
    "guardrail_span",
    "handoff_span",
    "set_trace_processors",
    "set_tracing_disabled",
    "speech_group_span",
    "transcription_span",
    "speech_span",
    "trace",
    "Trace",
    "TracingProcessor",
    "SpanError",
    "Span",
    "SpanData",
    "AgentSpanData",
    "CustomSpanData",
    "FunctionSpanData",
    "GenerationSpanData",
    "GuardrailSpanData",
    "HandoffSpanData",
    "SpeechGroupSpanData",
    "SpeechSpanData",
    "TranscriptionSpanData",
    "set_default_openai_key",
    "set_default_openai_client",
    "set_default_openai_api",
    "set_tracing_export_api_key",
    "enable_verbose_stdout_logging",
    "gen_trace_id",
    "gen_span_id",
    "default_tool_error_function",
]


<file_info>
path: src/agents/voice/models/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/voice/models/openai_stt.py
name: openai_stt.py
</file_info>
from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

from agents.exceptions import AgentsException

from ... import _debug
from ...logger import logger
from ...tracing import Span, SpanError, TranscriptionSpanData, transcription_span
from ..exceptions import STTWebsocketConnectionError
from ..imports import np, npt, websockets
from ..input import AudioInput, StreamedAudioInput
from ..model import StreamedTranscriptionSession, STTModel, STTModelSettings

EVENT_INACTIVITY_TIMEOUT = 1000  # Timeout for inactivity in event processing
SESSION_CREATION_TIMEOUT = 10  # Timeout waiting for session.created event
SESSION_UPDATE_TIMEOUT = 10  # Timeout waiting for session.updated event

DEFAULT_TURN_DETECTION = {"type": "semantic_vad"}


@dataclass
class ErrorSentinel:
    error: Exception


class SessionCompleteSentinel:
    pass


class WebsocketDoneSentinel:
    pass


def _audio_to_base64(audio_data: list[npt.NDArray[np.int16 | np.float32]]) -> str:
    concatenated_audio = np.concatenate(audio_data)
    if concatenated_audio.dtype == np.float32:
        # convert to int16
        concatenated_audio = np.clip(concatenated_audio, -1.0, 1.0)
        concatenated_audio = (concatenated_audio * 32767).astype(np.int16)
    audio_bytes = concatenated_audio.tobytes()
    return base64.b64encode(audio_bytes).decode("utf-8")


async def _wait_for_event(
    event_queue: asyncio.Queue[dict[str, Any]], expected_types: list[str], timeout: float
):
    """
    Wait for an event from event_queue whose type is in expected_types within the specified timeout.
    """
    start_time = time.time()
    while True:
        remaining = timeout - (time.time() - start_time)
        if remaining <= 0:
            raise TimeoutError(f"Timeout waiting for event(s): {expected_types}")
        evt = await asyncio.wait_for(event_queue.get(), timeout=remaining)
        evt_type = evt.get("type", "")
        if evt_type in expected_types:
            return evt
        elif evt_type == "error":
            raise Exception(f"Error event: {evt.get('error')}")


class OpenAISTTTranscriptionSession(StreamedTranscriptionSession):
    """A transcription session for OpenAI's STT model."""

    def __init__(
        self,
        input: StreamedAudioInput,
        client: AsyncOpenAI,
        model: str,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ):
        self.connected: bool = False
        self._client = client
        self._model = model
        self._settings = settings
        self._turn_detection = settings.turn_detection or DEFAULT_TURN_DETECTION
        self._trace_include_sensitive_data = trace_include_sensitive_data
        self._trace_include_sensitive_audio_data = trace_include_sensitive_audio_data

        self._input_queue: asyncio.Queue[npt.NDArray[np.int16 | np.float32]] = input.queue
        self._output_queue: asyncio.Queue[str | ErrorSentinel | SessionCompleteSentinel] = (
            asyncio.Queue()
        )
        self._websocket: websockets.ClientConnection | None = None
        self._event_queue: asyncio.Queue[dict[str, Any] | WebsocketDoneSentinel] = asyncio.Queue()
        self._state_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._turn_audio_buffer: list[npt.NDArray[np.int16 | np.float32]] = []
        self._tracing_span: Span[TranscriptionSpanData] | None = None

        # tasks
        self._listener_task: asyncio.Task[Any] | None = None
        self._process_events_task: asyncio.Task[Any] | None = None
        self._stream_audio_task: asyncio.Task[Any] | None = None
        self._connection_task: asyncio.Task[Any] | None = None
        self._stored_exception: Exception | None = None

    def _start_turn(self) -> None:
        self._tracing_span = transcription_span(
            model=self._model,
            model_config={
                "temperature": self._settings.temperature,
                "language": self._settings.language,
                "prompt": self._settings.prompt,
                "turn_detection": self._turn_detection,
            },
        )
        self._tracing_span.start()

    def _end_turn(self, _transcript: str) -> None:
        if len(_transcript) < 1:
            return

        if self._tracing_span:
            if self._trace_include_sensitive_audio_data:
                self._tracing_span.span_data.input = _audio_to_base64(self._turn_audio_buffer)

            self._tracing_span.span_data.input_format = "pcm"

            if self._trace_include_sensitive_data:
                self._tracing_span.span_data.output = _transcript

            self._tracing_span.finish()
            self._turn_audio_buffer = []
            self._tracing_span = None

    async def _event_listener(self) -> None:
        assert self._websocket is not None, "Websocket not initialized"

        async for message in self._websocket:
            try:
                event = json.loads(message)

                if event.get("type") == "error":
                    raise STTWebsocketConnectionError(f"Error event: {event.get('error')}")

                if event.get("type") in [
                    "session.updated",
                    "transcription_session.updated",
                    "session.created",
                    "transcription_session.created",
                ]:
                    await self._state_queue.put(event)

                await self._event_queue.put(event)
            except Exception as e:
                await self._output_queue.put(ErrorSentinel(e))
                raise STTWebsocketConnectionError("Error parsing events") from e
        await self._event_queue.put(WebsocketDoneSentinel())

    async def _configure_session(self) -> None:
        assert self._websocket is not None, "Websocket not initialized"
        await self._websocket.send(
            json.dumps(
                {
                    "type": "transcription_session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {"model": self._model},
                        "turn_detection": self._turn_detection,
                    },
                }
            )
        )

    async def _setup_connection(self, ws: websockets.ClientConnection) -> None:
        self._websocket = ws
        self._listener_task = asyncio.create_task(self._event_listener())

        try:
            event = await _wait_for_event(
                self._state_queue,
                ["session.created", "transcription_session.created"],
                SESSION_CREATION_TIMEOUT,
            )
        except TimeoutError as e:
            wrapped_err = STTWebsocketConnectionError(
                "Timeout waiting for transcription_session.created event"
            )
            await self._output_queue.put(ErrorSentinel(wrapped_err))
            raise wrapped_err from e
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise e

        await self._configure_session()

        try:
            event = await _wait_for_event(
                self._state_queue,
                ["session.updated", "transcription_session.updated"],
                SESSION_UPDATE_TIMEOUT,
            )
            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Session updated")
            else:
                logger.debug(f"Session updated: {event}")
        except TimeoutError as e:
            wrapped_err = STTWebsocketConnectionError(
                "Timeout waiting for transcription_session.updated event"
            )
            await self._output_queue.put(ErrorSentinel(wrapped_err))
            raise wrapped_err from e
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise

    async def _handle_events(self) -> None:
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=EVENT_INACTIVITY_TIMEOUT
                )
                if isinstance(event, WebsocketDoneSentinel):
                    # processed all events and websocket is done
                    break

                event_type = event.get("type", "unknown")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = cast(str, event.get("transcript", ""))
                    if len(transcript) > 0:
                        self._end_turn(transcript)
                        self._start_turn()
                        await self._output_queue.put(transcript)
                await asyncio.sleep(0)  # yield control
            except asyncio.TimeoutError:
                # No new events for a while. Assume the session is done.
                break
            except Exception as e:
                await self._output_queue.put(ErrorSentinel(e))
                raise e
        await self._output_queue.put(SessionCompleteSentinel())

    async def _stream_audio(
        self, audio_queue: asyncio.Queue[npt.NDArray[np.int16 | np.float32]]
    ) -> None:
        assert self._websocket is not None, "Websocket not initialized"
        self._start_turn()
        while True:
            buffer = await audio_queue.get()
            if buffer is None:
                break

            self._turn_audio_buffer.append(buffer)
            try:
                await self._websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                        }
                    )
                )
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                await self._output_queue.put(ErrorSentinel(e))
                raise e

            await asyncio.sleep(0)  # yield control

    async def _process_websocket_connection(self) -> None:
        try:
            async with websockets.connect(
                "wss://api.openai.com/v1/realtime?intent=transcription",
                additional_headers={
                    "Authorization": f"Bearer {self._client.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                    "OpenAI-Log-Session": "1",
                },
            ) as ws:
                await self._setup_connection(ws)
                self._process_events_task = asyncio.create_task(self._handle_events())
                self._stream_audio_task = asyncio.create_task(self._stream_audio(self._input_queue))
                self.connected = True
                if self._listener_task:
                    await self._listener_task
                else:
                    logger.error("Listener task not initialized")
                    raise AgentsException("Listener task not initialized")
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise e

    def _check_errors(self) -> None:
        if self._connection_task and self._connection_task.done():
            exc = self._connection_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._process_events_task and self._process_events_task.done():
            exc = self._process_events_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._stream_audio_task and self._stream_audio_task.done():
            exc = self._stream_audio_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._listener_task and self._listener_task.done():
            exc = self._listener_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

    def _cleanup_tasks(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()

        if self._process_events_task and not self._process_events_task.done():
            self._process_events_task.cancel()

        if self._stream_audio_task and not self._stream_audio_task.done():
            self._stream_audio_task.cancel()

        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()

    async def transcribe_turns(self) -> AsyncIterator[str]:
        self._connection_task = asyncio.create_task(self._process_websocket_connection())

        while True:
            try:
                turn = await self._output_queue.get()
            except asyncio.CancelledError:
                break

            if (
                turn is None
                or isinstance(turn, ErrorSentinel)
                or isinstance(turn, SessionCompleteSentinel)
            ):
                self._output_queue.task_done()
                break
            yield turn
            self._output_queue.task_done()

        if self._tracing_span:
            self._end_turn("")

        if self._websocket:
            await self._websocket.close()

        self._check_errors()
        if self._stored_exception:
            raise self._stored_exception

    async def close(self) -> None:
        if self._websocket:
            await self._websocket.close()

        self._cleanup_tasks()


class OpenAISTTModel(STTModel):
    """A speech-to-text model for OpenAI."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
    ):
        """Create a new OpenAI speech-to-text model.

        Args:
            model: The name of the model to use.
            openai_client: The OpenAI client to use.
        """
        self.model = model
        self._client = openai_client

    @property
    def model_name(self) -> str:
        return self.model

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else None  # NOT_GIVEN

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        """Transcribe an audio input.

        Args:
            input: The audio input to transcribe.
            settings: The settings to use for the transcription.

        Returns:
            The transcribed text.
        """
        with transcription_span(
            model=self.model,
            input=input.to_base64() if trace_include_sensitive_audio_data else "",
            input_format="pcm",
            model_config={
                "temperature": self._non_null_or_not_given(settings.temperature),
                "language": self._non_null_or_not_given(settings.language),
                "prompt": self._non_null_or_not_given(settings.prompt),
            },
        ) as span:
            try:
                response = await self._client.audio.transcriptions.create(
                    model=self.model,
                    file=input.to_audio_file(),
                    prompt=self._non_null_or_not_given(settings.prompt),
                    language=self._non_null_or_not_given(settings.language),
                    temperature=self._non_null_or_not_given(settings.temperature),
                )
                if trace_include_sensitive_data:
                    span.span_data.output = response.text
                return response.text
            except Exception as e:
                span.span_data.output = ""
                span.set_error(SpanError(message=str(e), data={}))
                raise e

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        """Create a new transcription session.

        Args:
            input: The audio input to transcribe.
            settings: The settings to use for the transcription.
            trace_include_sensitive_data: Whether to include sensitive data in traces.
            trace_include_sensitive_audio_data: Whether to include sensitive audio data in traces.

        Returns:
            A new transcription session.
        """
        return OpenAISTTTranscriptionSession(
            input,
            self._client,
            self.model,
            settings,
            trace_include_sensitive_data,
            trace_include_sensitive_audio_data,
        )


<file_info>
path: tests/test_responses_tracing.py
name: test_responses_tracing.py
</file_info>
import pytest
from inline_snapshot import snapshot
from openai import AsyncOpenAI
from openai.types.responses import ResponseCompletedEvent

from agents import ModelSettings, ModelTracing, OpenAIResponsesModel, trace
from agents.tracing.span_data import ResponseSpanData
from tests import fake_model

from .testing_processor import assert_no_spans, fetch_normalized_spans, fetch_ordered_spans


class DummyTracing:
    def is_disabled(self):
        return False


class DummyUsage:
    def __init__(self, input_tokens=1, output_tokens=1, total_tokens=2):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class DummyResponse:
    def __init__(self):
        self.id = "dummy-id"
        self.output = []
        self.usage = DummyUsage()

    def __aiter__(self):
        yield ResponseCompletedEvent(
            type="response.completed",
            response=fake_model.get_response_obj(self.output),
        )


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_creates_trace(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Mock _fetch_response to return a dummy response with a known id
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            return DummyResponse()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Call get_response
        await model.get_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.ENABLED
        )

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test",
                "children": [{"type": "response", "data": {"response_id": "dummy-id"}}],
            }
        ]
    )


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_non_data_tracing_doesnt_set_response_id(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Mock _fetch_response to return a dummy response with a known id
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            return DummyResponse()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Call get_response
        await model.get_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.ENABLED_WITHOUT_DATA
        )

    assert fetch_normalized_spans() == snapshot(
        [{"workflow_name": "test", "children": [{"type": "response"}]}]
    )

    [span] = fetch_ordered_spans()
    assert span.span_data.response is None


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_disable_tracing_does_not_create_span(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Mock _fetch_response to return a dummy response with a known id
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            return DummyResponse()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Call get_response
        await model.get_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.DISABLED
        )

    assert fetch_normalized_spans() == snapshot([{"workflow_name": "test"}])

    assert_no_spans()


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_response_creates_trace(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Define a dummy fetch function that returns an async stream with a dummy response
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            class DummyStream:
                async def __aiter__(self):
                    yield ResponseCompletedEvent(
                        type="response.completed",
                        response=fake_model.get_response_obj([], "dummy-id-123"),
                    )

            return DummyStream()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Consume the stream to trigger processing of the final response
        async for _ in model.stream_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.ENABLED
        ):
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "test",
                "children": [{"type": "response", "data": {"response_id": "dummy-id-123"}}],
            }
        ]
    )


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_non_data_tracing_doesnt_set_response_id(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Define a dummy fetch function that returns an async stream with a dummy response
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            class DummyStream:
                async def __aiter__(self):
                    yield ResponseCompletedEvent(
                        type="response.completed",
                        response=fake_model.get_response_obj([], "dummy-id-123"),
                    )

            return DummyStream()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Consume the stream to trigger processing of the final response
        async for _ in model.stream_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.ENABLED_WITHOUT_DATA
        ):
            pass

    assert fetch_normalized_spans() == snapshot(
        [{"workflow_name": "test", "children": [{"type": "response"}]}]
    )

    [span] = fetch_ordered_spans()
    assert isinstance(span.span_data, ResponseSpanData)
    assert span.span_data.response is None


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_stream_disabled_tracing_doesnt_create_span(monkeypatch):
    with trace(workflow_name="test"):
        # Create an instance of the model
        model = OpenAIResponsesModel(model="test-model", openai_client=AsyncOpenAI(api_key="test"))

        # Define a dummy fetch function that returns an async stream with a dummy response
        async def dummy_fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, stream
        ):
            class DummyStream:
                async def __aiter__(self):
                    yield ResponseCompletedEvent(
                        type="response.completed",
                        response=fake_model.get_response_obj([], "dummy-id-123"),
                    )

            return DummyStream()

        monkeypatch.setattr(model, "_fetch_response", dummy_fetch_response)

        # Consume the stream to trigger processing of the final response
        async for _ in model.stream_response(
            "instr", "input", ModelSettings(), [], None, [], ModelTracing.DISABLED
        ):
            pass

    assert fetch_normalized_spans() == snapshot([{"workflow_name": "test"}])

    assert_no_spans()


<file_info>
path: tests/voice/test_input.py
name: test_input.py
</file_info>
import io
import wave

import numpy as np
import pytest

try:
    from agents import UserError
    from agents.voice import AudioInput, StreamedAudioInput
    from agents.voice.input import DEFAULT_SAMPLE_RATE, _buffer_to_audio_file
except ImportError:
    pass


def test_buffer_to_audio_file_int16():
    # Create a simple sine wave in int16 format
    t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE)
    buffer = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    filename, audio_file, content_type = _buffer_to_audio_file(buffer)

    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    assert isinstance(audio_file, io.BytesIO)

    # Verify the WAV file contents
    with wave.open(audio_file, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == DEFAULT_SAMPLE_RATE
        assert wav_file.getnframes() == len(buffer)


def test_buffer_to_audio_file_float32():
    # Create a simple sine wave in float32 format
    t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE)
    buffer = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    filename, audio_file, content_type = _buffer_to_audio_file(buffer)

    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    assert isinstance(audio_file, io.BytesIO)

    # Verify the WAV file contents
    with wave.open(audio_file, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == DEFAULT_SAMPLE_RATE
        assert wav_file.getnframes() == len(buffer)


def test_buffer_to_audio_file_invalid_dtype():
    # Create a buffer with invalid dtype (float64)
    buffer = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(UserError, match="Buffer must be a numpy array of int16 or float32"):
        # Purposely ignore the type error
        _buffer_to_audio_file(buffer)  # type: ignore


class TestAudioInput:
    def test_audio_input_default_params(self):
        # Create a simple sine wave
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE)
        buffer = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_input = AudioInput(buffer=buffer)

        assert audio_input.frame_rate == DEFAULT_SAMPLE_RATE
        assert audio_input.sample_width == 2
        assert audio_input.channels == 1
        assert np.array_equal(audio_input.buffer, buffer)

    def test_audio_input_custom_params(self):
        # Create a simple sine wave
        t = np.linspace(0, 1, 48000)
        buffer = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_input = AudioInput(buffer=buffer, frame_rate=48000, sample_width=4, channels=2)

        assert audio_input.frame_rate == 48000
        assert audio_input.sample_width == 4
        assert audio_input.channels == 2
        assert np.array_equal(audio_input.buffer, buffer)

    def test_audio_input_to_audio_file(self):
        # Create a simple sine wave
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE)
        buffer = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_input = AudioInput(buffer=buffer)
        filename, audio_file, content_type = audio_input.to_audio_file()

        assert filename == "audio.wav"
        assert content_type == "audio/wav"
        assert isinstance(audio_file, io.BytesIO)

        # Verify the WAV file contents
        with wave.open(audio_file, "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == DEFAULT_SAMPLE_RATE
            assert wav_file.getnframes() == len(buffer)


class TestStreamedAudioInput:
    @pytest.mark.asyncio
    async def test_streamed_audio_input(self):
        streamed_input = StreamedAudioInput()

        # Create some test audio data
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE)
        audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)

        # Add audio to the queue
        await streamed_input.add_audio(audio1)
        await streamed_input.add_audio(audio2)

        # Verify the queue contents
        assert streamed_input.queue.qsize() == 2
        # Test non-blocking get
        assert np.array_equal(streamed_input.queue.get_nowait(), audio1)
        # Test blocking get
        assert np.array_equal(await streamed_input.queue.get(), audio2)
        assert streamed_input.queue.empty()


<file_info>
path: tests/test_responses.py
name: test_responses.py
</file_info>
from __future__ import annotations

from typing import Any

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agents import (
    Agent,
    FunctionTool,
    Handoff,
    TResponseInputItem,
    default_tool_error_function,
    function_tool,
)


def get_text_input_item(content: str) -> TResponseInputItem:
    return {
        "content": content,
        "role": "user",
    }


def get_text_message(content: str) -> ResponseOutputItem:
    return ResponseOutputMessage(
        id="1",
        type="message",
        role="assistant",
        content=[ResponseOutputText(text=content, type="output_text", annotations=[])],
        status="completed",
    )


def get_function_tool(
    name: str | None = None, return_value: str | None = None, hide_errors: bool = False
) -> FunctionTool:
    def _foo() -> str:
        return return_value or "result_ok"

    return function_tool(
        _foo,
        name_override=name,
        failure_error_function=None if hide_errors else default_tool_error_function,
    )


def get_function_tool_call(name: str, arguments: str | None = None) -> ResponseOutputItem:
    return ResponseFunctionToolCall(
        id="1",
        call_id="2",
        type="function_call",
        name=name,
        arguments=arguments or "",
    )


def get_handoff_tool_call(
    to_agent: Agent[Any], override_name: str | None = None, args: str | None = None
) -> ResponseOutputItem:
    name = override_name or Handoff.default_tool_name(to_agent)
    return get_function_tool_call(name, args)


def get_final_output_message(args: str) -> ResponseOutputItem:
    return ResponseOutputMessage(
        id="1",
        type="message",
        role="assistant",
        content=[ResponseOutputText(text=args, type="output_text", annotations=[])],
        status="completed",
    )


<file_info>
path: tests/test_max_turns.py
name: test_max_turns.py
</file_info>
from __future__ import annotations

import json

import pytest
from typing_extensions import TypedDict

from agents import Agent, MaxTurnsExceeded, Runner

from .fake_model import FakeModel
from .test_responses import get_function_tool, get_function_tool_call, get_text_message


@pytest.mark.asyncio
async def test_non_streamed_max_turns():
    model = FakeModel()
    agent = Agent(
        name="test_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
    )

    func_output = json.dumps({"a": "b"})

    model.add_multiple_turn_outputs(
        [
            [get_text_message("1"), get_function_tool_call("some_function", func_output)],
            [get_text_message("2"), get_function_tool_call("some_function", func_output)],
            [get_text_message("3"), get_function_tool_call("some_function", func_output)],
            [get_text_message("4"), get_function_tool_call("some_function", func_output)],
            [get_text_message("5"), get_function_tool_call("some_function", func_output)],
        ]
    )
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, input="user_message", max_turns=3)


@pytest.mark.asyncio
async def test_streamed_max_turns():
    model = FakeModel()
    agent = Agent(
        name="test_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
    )
    func_output = json.dumps({"a": "b"})

    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("1"),
                get_function_tool_call("some_function", func_output),
            ],
            [
                get_text_message("2"),
                get_function_tool_call("some_function", func_output),
            ],
            [
                get_text_message("3"),
                get_function_tool_call("some_function", func_output),
            ],
            [
                get_text_message("4"),
                get_function_tool_call("some_function", func_output),
            ],
            [
                get_text_message("5"),
                get_function_tool_call("some_function", func_output),
            ],
        ]
    )
    with pytest.raises(MaxTurnsExceeded):
        output = Runner.run_streamed(agent, input="user_message", max_turns=3)
        async for _ in output.stream_events():
            pass


class Foo(TypedDict):
    a: str


@pytest.mark.asyncio
async def test_structured_output_non_streamed_max_turns():
    model = FakeModel()
    agent = Agent(
        name="test_1",
        model=model,
        output_type=Foo,
        tools=[get_function_tool("tool_1", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
        ]
    )
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, input="user_message", max_turns=3)


@pytest.mark.asyncio
async def test_structured_output_streamed_max_turns():
    model = FakeModel()
    agent = Agent(
        name="test_1",
        model=model,
        output_type=Foo,
        tools=[get_function_tool("tool_1", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
            [get_function_tool_call("tool_1")],
        ]
    )
    with pytest.raises(MaxTurnsExceeded):
        output = Runner.run_streamed(agent, input="user_message", max_turns=3)
        async for _ in output.stream_events():
            pass


<file_info>
path: tests/test_run_step_execution.py
name: test_run_step_execution.py
</file_info>
from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agents import (
    Agent,
    MessageOutputItem,
    ModelResponse,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    RunItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    Usage,
)
from agents._run_impl import (
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    RunImpl,
    SingleStepResult,
)

from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_input_item,
    get_text_message,
)


@pytest.mark.asyncio
async def test_empty_response_is_final_output():
    agent = Agent[None](name="test")
    response = ModelResponse(
        output=[],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"
    assert result.generated_items == []
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == ""


@pytest.mark.asyncio
async def test_plaintext_agent_no_tool_calls_is_final_output():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[get_text_message("hello_world")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"
    assert len(result.generated_items) == 1
    assert_item_is_message(result.generated_items[0], "hello_world")
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == "hello_world"


@pytest.mark.asyncio
async def test_plaintext_agent_no_tool_calls_multiple_messages_is_final_output():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[
            get_text_message("hello_world"),
            get_text_message("bye"),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(
        agent,
        response,
        original_input=[
            get_text_input_item("test"),
            get_text_input_item("test2"),
        ],
    )

    assert len(result.original_input) == 2
    assert len(result.generated_items) == 2
    assert_item_is_message(result.generated_items[0], "hello_world")
    assert_item_is_message(result.generated_items[1], "bye")

    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == "bye"


@pytest.mark.asyncio
async def test_plaintext_agent_with_tool_call_is_run_again():
    agent = Agent(name="test", tools=[get_function_tool(name="test", return_value="123")])
    response = ModelResponse(
        output=[get_text_message("hello_world"), get_function_tool_call("test", "")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"

    # 3 items: new message, tool call, tool result
    assert len(result.generated_items) == 3
    assert isinstance(result.next_step, NextStepRunAgain)

    items = result.generated_items
    assert_item_is_message(items[0], "hello_world")
    assert_item_is_function_tool_call(items[1], "test", None)
    assert_item_is_function_tool_call_output(items[2], "123")

    assert isinstance(result.next_step, NextStepRunAgain)


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    agent = Agent(
        name="test",
        tools=[
            get_function_tool(name="test_1", return_value="123"),
            get_function_tool(name="test_2", return_value="456"),
            get_function_tool(name="test_3", return_value="789"),
        ],
    )
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("test_1"),
            get_function_tool_call("test_2"),
        ],
        usage=Usage(),
        referenceable_id=None,
    )

    result = await get_execute_result(agent, response)
    assert result.original_input == "hello"

    # 5 items: new message, 2 tool calls, 2 tool call outputs
    assert len(result.generated_items) == 5
    assert isinstance(result.next_step, NextStepRunAgain)

    items = result.generated_items
    assert_item_is_message(items[0], "Hello, world!")
    assert_item_is_function_tool_call(items[1], "test_1", None)
    assert_item_is_function_tool_call(items[2], "test_2", None)

    assert isinstance(result.next_step, NextStepRunAgain)


@pytest.mark.asyncio
async def test_handoff_output_leads_to_handoff_next_step():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])
    response = ModelResponse(
        output=[get_text_message("Hello, world!"), get_handoff_tool_call(agent_1)],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent_3, response)

    assert isinstance(result.next_step, NextStepHandoff)
    assert result.next_step.new_agent == agent_1

    assert len(result.generated_items) == 3


class Foo(BaseModel):
    bar: str


@pytest.mark.asyncio
async def test_final_output_without_tool_runs_again():
    agent = Agent(name="test", output_type=Foo, tools=[get_function_tool("tool_1", "result")])
    response = ModelResponse(
        output=[get_function_tool_call("tool_1")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent, response)

    assert isinstance(result.next_step, NextStepRunAgain)
    assert len(result.generated_items) == 2, "expected 2 items: tool call, tool call output"


@pytest.mark.asyncio
async def test_final_output_leads_to_final_output_next_step():
    agent = Agent(name="test", output_type=Foo)
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_final_output_message(Foo(bar="123").model_dump_json()),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent, response)

    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == Foo(bar="123")


@pytest.mark.asyncio
async def test_handoff_and_final_output_leads_to_handoff_next_step():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2], output_type=Foo)
    response = ModelResponse(
        output=[
            get_final_output_message(Foo(bar="123").model_dump_json()),
            get_handoff_tool_call(agent_1),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent_3, response)

    assert isinstance(result.next_step, NextStepHandoff)
    assert result.next_step.new_agent == agent_1


@pytest.mark.asyncio
async def test_multiple_final_output_leads_to_final_output_next_step():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2], output_type=Foo)
    response = ModelResponse(
        output=[
            get_final_output_message(Foo(bar="123").model_dump_json()),
            get_final_output_message(Foo(bar="456").model_dump_json()),
        ],
        usage=Usage(),
        referenceable_id=None,
    )
    result = await get_execute_result(agent_3, response)

    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == Foo(bar="456")


# === Helpers ===


def assert_item_is_message(item: RunItem, text: str) -> None:
    assert isinstance(item, MessageOutputItem)
    assert item.raw_item.type == "message"
    assert item.raw_item.role == "assistant"
    assert item.raw_item.content[0].type == "output_text"
    assert item.raw_item.content[0].text == text


def assert_item_is_function_tool_call(
    item: RunItem, name: str, arguments: str | None = None
) -> None:
    assert isinstance(item, ToolCallItem)
    assert item.raw_item.type == "function_call"
    assert item.raw_item.name == name
    assert not arguments or item.raw_item.arguments == arguments


def assert_item_is_function_tool_call_output(item: RunItem, output: str) -> None:
    assert isinstance(item, ToolCallOutputItem)
    assert item.raw_item["type"] == "function_call_output"
    assert item.raw_item["output"] == output


async def get_execute_result(
    agent: Agent[Any],
    response: ModelResponse,
    *,
    original_input: str | list[TResponseInputItem] | None = None,
    generated_items: list[RunItem] | None = None,
    hooks: RunHooks[Any] | None = None,
    context_wrapper: RunContextWrapper[Any] | None = None,
    run_config: RunConfig | None = None,
) -> SingleStepResult:
    output_schema = Runner._get_output_schema(agent)
    handoffs = Runner._get_handoffs(agent)

    processed_response = RunImpl.process_model_response(
        agent=agent,
        response=response,
        output_schema=output_schema,
        handoffs=handoffs,
    )
    return await RunImpl.execute_tools_and_side_effects(
        agent=agent,
        original_input=original_input or "hello",
        new_response=response,
        pre_step_items=generated_items or [],
        processed_response=processed_response,
        output_schema=output_schema,
        hooks=hooks or RunHooks(),
        context_wrapper=context_wrapper or RunContextWrapper(None),
        run_config=run_config or RunConfig(),
    )


<file_info>
path: examples/tools/file_search.py
name: file_search.py
</file_info>
import asyncio

from agents import Agent, FileSearchTool, Runner, trace


async def main():
    agent = Agent(
        name="File searcher",
        instructions="You are a helpful agent.",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=["vs_67bf88953f748191be42b462090e53e7"],
                include_search_results=True,
            )
        ],
    )

    with trace("File search example"):
        result = await Runner.run(
            agent, "Be concise, and tell me 1 sentence about Arrakis I might not know."
        )
        print(result.final_output)
        """
        Arrakis, the desert planet in Frank Herbert's "Dune," was inspired by the scarcity of water
        as a metaphor for oil and other finite resources.
        """

        print("\n".join([str(out) for out in result.new_items]))
        """
        {"id":"...", "queries":["Arrakis"], "results":[...]}
        """


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: examples/financial_research_agent/agents/writer_agent.py
name: writer_agent.py
</file_info>
from pydantic import BaseModel

from agents import Agent

# Writer agent brings together the raw search results and optionally calls out
# to sub‑analyst tools for specialized commentary, then returns a cohesive markdown report.
WRITER_PROMPT = (
    "You are a senior financial analyst. You will be provided with the original query and "
    "a set of raw search summaries. Your task is to synthesize these into a long‑form markdown "
    "report (at least several paragraphs) including a short executive summary and follow‑up "
    "questions. If needed, you can call the available analysis tools (e.g. fundamentals_analysis, "
    "risk_analysis) to get short specialist write‑ups to incorporate."
)


class FinancialReportData(BaseModel):
    short_summary: str
    """A short 2‑3 sentence executive summary."""

    markdown_report: str
    """The full markdown report."""

    follow_up_questions: list[str]
    """Suggested follow‑up questions for further research."""


# Note: We will attach handoffs to specialist analyst agents at runtime in the manager.
# This shows how an agent can use handoffs to delegate to specialized subagents.
writer_agent = Agent(
    name="FinancialWriterAgent",
    instructions=WRITER_PROMPT,
    model="gpt-4.5-preview-2025-02-27",
    output_type=FinancialReportData,
)


<file_info>
path: examples/voice/streamed/__init__.py
name: __init__.py
</file_info>


<file_info>
path: examples/voice/static/main.py
name: main.py
</file_info>
import asyncio
import random

from agents import Agent, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)

from .util import AudioPlayer, record_audio

"""
This is a simple example that uses a recorded audio buffer. Run it via:
`python -m examples.voice.static.main`

1. You can record an audio clip in the terminal.
2. The pipeline automatically transcribes the audio.
3. The agent workflow is a simple one that starts at the Assistant agent.
4. The output of the agent is streamed to the audio player.

Try examples like:
- Tell me a joke (will respond with a joke)
- What's the weather in Tokyo? (will call the `get_weather` tool and then speak)
- Hola, como estas? (will handoff to the spanish agent)
"""


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)


class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        print(f"[debug] on_run called with transcription: {transcription}")


async def main():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent, callbacks=WorkflowCallbacks())
    )

    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)
                print("Received audio")
            elif event.type == "voice_stream_event_lifecycle":
                print(f"Received lifecycle event: {event.event}")


if __name__ == "__main__":
    asyncio.run(main())


<file_info>
path: src/agents/tracing/util.py
name: util.py
</file_info>
import uuid
from datetime import datetime, timezone


def time_iso() -> str:
    """Returns the current time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def gen_trace_id() -> str:
    """Generates a new trace ID."""
    return f"trace_{uuid.uuid4().hex}"


def gen_span_id() -> str:
    """Generates a new span ID."""
    return f"span_{uuid.uuid4().hex[:24]}"


def gen_group_id() -> str:
    """Generates a new group ID."""
    return f"group_{uuid.uuid4().hex[:24]}"


<file_info>
path: src/agents/util/__init__.py
name: __init__.py
</file_info>


<file_info>
path: src/agents/util/_coro.py
name: _coro.py
</file_info>
async def noop_coroutine() -> None:
    pass


<file_info>
path: src/agents/models/openai_responses.py
name: openai_responses.py
</file_info>
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream, NotGiven
from openai.types import ChatModel
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
    WebSearchToolParam,
    response_create_params,
)

from .. import _debug
from ..agent_output import AgentOutputSchema
from ..exceptions import UserError
from ..handoffs import Handoff
from ..items import ItemHelpers, ModelResponse, TResponseInputItem
from ..logger import logger
from ..tool import ComputerTool, FileSearchTool, FunctionTool, Tool, WebSearchTool
from ..tracing import SpanError, response_span
from ..usage import Usage
from ..version import __version__
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}

# From the Responses API
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]


class OpenAIResponsesModel(Model):
    """
    Implementation of `Model` that uses the OpenAI Responses API.
    """

    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else NOT_GIVEN

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                response = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    stream=False,
                )

                if _debug.DONT_LOG_MODEL_DATA:
                    logger.debug("LLM responded")
                else:
                    logger.debug(
                        "LLM resp:\n"
                        f"{json.dumps([x.model_dump() for x in response.output], indent=2)}\n"
                    )

                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                    if response.usage
                    else Usage()
                )

                if tracing.include_data():
                    span_response.span_data.response = response
                    span_response.span_data.input = input
            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error getting response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                request_id = e.request_id if isinstance(e, APIStatusError) else None
                logger.error(f"Error getting response: {e}. (request_id: {request_id})")
                raise

        return ModelResponse(
            output=response.output,
            usage=usage,
            referenceable_id=response.id,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """
        Yields a partial message as it is generated, as well as the usage information.
        """
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                stream = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    stream=True,
                )

                final_response: Response | None = None

                async for chunk in stream:
                    if isinstance(chunk, ResponseCompletedEvent):
                        final_response = chunk.response
                    yield chunk

                if final_response and tracing.include_data():
                    span_response.span_data.response = final_response
                    span_response.span_data.input = input

            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error streaming response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                logger.error(f"Error streaming response: {e}")
                raise

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True],
    ) -> AsyncStream[ResponseStreamEvent]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True] | Literal[False] = False,
    ) -> Response | AsyncStream[ResponseStreamEvent]:
        list_input = ItemHelpers.input_to_new_input_list(input)

        parallel_tool_calls = (
            True if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )

        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = Converter.convert_tools(tools, handoffs)
        response_format = Converter.get_response_format(output_schema)

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"Calling LLM {self.model} with input:\n"
                f"{json.dumps(list_input, indent=2)}\n"
                f"Tools:\n{json.dumps(converted_tools.tools, indent=2)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        return await self._client.responses.create(
            instructions=self._non_null_or_not_given(system_instructions),
            model=self.model,
            input=list_input,
            include=converted_tools.includes,
            tools=converted_tools.tools,
            temperature=self._non_null_or_not_given(model_settings.temperature),
            top_p=self._non_null_or_not_given(model_settings.top_p),
            truncation=self._non_null_or_not_given(model_settings.truncation),
            max_output_tokens=self._non_null_or_not_given(model_settings.max_tokens),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            extra_headers=_HEADERS,
            text=response_format,
        )

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client


@dataclass
class ConvertedTools:
    tools: list[ToolParam]
    includes: list[IncludeLiteral]


class Converter:
    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> response_create_params.ToolChoice | NotGiven:
        if tool_choice is None:
            return NOT_GIVEN
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "file_search":
            return {
                "type": "file_search",
            }
        elif tool_choice == "web_search_preview":
            return {
                "type": "web_search_preview",
            }
        elif tool_choice == "computer_use_preview":
            return {
                "type": "computer_use_preview",
            }
        else:
            return {
                "type": "function",
                "name": tool_choice,
            }

    @classmethod
    def get_response_format(
        cls, output_schema: AgentOutputSchema | None
    ) -> ResponseTextConfigParam | NotGiven:
        if output_schema is None or output_schema.is_plain_text():
            return NOT_GIVEN
        else:
            return {
                "format": {
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": output_schema.json_schema(),
                    "strict": output_schema.strict_json_schema,
                }
            }

    @classmethod
    def convert_tools(
        cls,
        tools: list[Tool],
        handoffs: list[Handoff[Any]],
    ) -> ConvertedTools:
        converted_tools: list[ToolParam] = []
        includes: list[IncludeLiteral] = []

        computer_tools = [tool for tool in tools if isinstance(tool, ComputerTool)]
        if len(computer_tools) > 1:
            raise UserError(f"You can only provide one computer tool. Got {len(computer_tools)}")

        for tool in tools:
            converted_tool, include = cls._convert_tool(tool)
            converted_tools.append(converted_tool)
            if include:
                includes.append(include)

        for handoff in handoffs:
            converted_tools.append(cls._convert_handoff_tool(handoff))

        return ConvertedTools(tools=converted_tools, includes=includes)

    @classmethod
    def _convert_tool(cls, tool: Tool) -> tuple[ToolParam, IncludeLiteral | None]:
        """Returns converted tool and includes"""

        if isinstance(tool, FunctionTool):
            converted_tool: ToolParam = {
                "name": tool.name,
                "parameters": tool.params_json_schema,
                "strict": tool.strict_json_schema,
                "type": "function",
                "description": tool.description,
            }
            includes: IncludeLiteral | None = None
        elif isinstance(tool, WebSearchTool):
            ws: WebSearchToolParam = {
                "type": "web_search_preview",
                "user_location": tool.user_location,
                "search_context_size": tool.search_context_size,
            }
            converted_tool = ws
            includes = None
        elif isinstance(tool, FileSearchTool):
            converted_tool = {
                "type": "file_search",
                "vector_store_ids": tool.vector_store_ids,
            }
            if tool.max_num_results:
                converted_tool["max_num_results"] = tool.max_num_results
            if tool.ranking_options:
                converted_tool["ranking_options"] = tool.ranking_options
            if tool.filters:
                converted_tool["filters"] = tool.filters

            includes = "file_search_call.results" if tool.include_search_results else None
        elif isinstance(tool, ComputerTool):
            converted_tool = {
                "type": "computer_use_preview",
                "environment": tool.computer.environment,
                "display_width": tool.computer.dimensions[0],
                "display_height": tool.computer.dimensions[1],
            }
            includes = None

        else:
            raise UserError(f"Unknown tool type: {type(tool)}, tool")

        return converted_tool, includes

    @classmethod
    def _convert_handoff_tool(cls, handoff: Handoff) -> ToolParam:
        return {
            "name": handoff.tool_name,
            "parameters": handoff.input_json_schema,
            "strict": handoff.strict_json_schema,
            "type": "function",
            "description": handoff.tool_description,
        }


<file_info>
path: src/agents/guardrail.py
name: guardrail.py
</file_info>
from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Union, overload

from typing_extensions import TypeVar

from .exceptions import UserError
from .items import TResponseInputItem
from .run_context import RunContextWrapper, TContext
from .util._types import MaybeAwaitable

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function."""

    output_info: Any
    """
    Optional information about the guardrail's output. For example, the guardrail could include
    information about the checks it performed and granular results.
    """

    tripwire_triggered: bool
    """
    Whether the tripwire was triggered. If triggered, the agent's execution will be halted.
    """


@dataclass
class InputGuardrailResult:
    """The result of a guardrail run."""

    guardrail: InputGuardrail[Any]
    """
    The guardrail that was run.
    """

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class OutputGuardrailResult:
    """The result of a guardrail run."""

    guardrail: OutputGuardrail[Any]
    """
    The guardrail that was run.
    """

    agent_output: Any
    """
    The output of the agent that was checked by the guardrail.
    """

    agent: Agent[Any]
    """
    The agent that was checked by the guardrail.
    """

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class InputGuardrail(Generic[TContext]):
    """Input guardrails are checks that run in parallel to the agent's execution.
    They can be used to do things like:
    - Check if input messages are off-topic
    - Take over control of the agent's execution if an unexpected input is detected

    You can use the `@input_guardrail()` decorator to turn a function into an `InputGuardrail`, or
    create an `InputGuardrail` manually.

    Guardrails return a `GuardrailResult`. If `result.tripwire_triggered` is `True`, the agent
    execution will immediately stop and a `InputGuardrailTripwireTriggered` exception will be raised
    """

    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], str | list[TResponseInputItem]],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    """A function that receives the agent input and the context, and returns a
     `GuardrailResult`. The result marks whether the tripwire was triggered, and can optionally
     include information about the guardrail's output.
    """

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, we'll use the guardrail
    function's name.
    """

    def get_name(self) -> str:
        if self.name:
            return self.name

        return self.guardrail_function.__name__

    async def run(
        self,
        agent: Agent[Any],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
    ) -> InputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        output = self.guardrail_function(context, agent, input)
        if inspect.isawaitable(output):
            return InputGuardrailResult(
                guardrail=self,
                output=await output,
            )

        return InputGuardrailResult(
            guardrail=self,
            output=output,
        )


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Output guardrails are checks that run on the final output of an agent.
    They can be used to do check if the output passes certain validation criteria

    You can use the `@output_guardrail()` decorator to turn a function into an `OutputGuardrail`,
    or create an `OutputGuardrail` manually.

    Guardrails return a `GuardrailResult`. If `result.tripwire_triggered` is `True`, a
    `OutputGuardrailTripwireTriggered` exception will be raised.
    """

    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], Any],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    """A function that receives the final agent, its output, and the context, and returns a
     `GuardrailResult`. The result marks whether the tripwire was triggered, and can optionally
     include information about the guardrail's output.
    """

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, we'll use the guardrail
    function's name.
    """

    def get_name(self) -> str:
        if self.name:
            return self.name

        return self.guardrail_function.__name__

    async def run(
        self, context: RunContextWrapper[TContext], agent: Agent[Any], agent_output: Any
    ) -> OutputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        output = self.guardrail_function(context, agent, agent_output)
        if inspect.isawaitable(output):
            return OutputGuardrailResult(
                guardrail=self,
                agent=agent,
                agent_output=agent_output,
                output=await output,
            )

        return OutputGuardrailResult(
            guardrail=self,
            agent=agent,
            agent_output=agent_output,
            output=output,
        )


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)

# For InputGuardrail
_InputGuardrailFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Union[str, list[TResponseInputItem]]],
    GuardrailFunctionOutput,
]
_InputGuardrailFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Union[str, list[TResponseInputItem]]],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    func: _InputGuardrailFuncAsync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
    InputGuardrail[TContext_co],
]: ...


def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co]
    | _InputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    InputGuardrail[TContext_co]
    | Callable[
        [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
        InputGuardrail[TContext_co],
    ]
):
    """
    Decorator that transforms a sync or async function into an `InputGuardrail`.
    It can be used directly (no parentheses) or with keyword args, e.g.:

        @input_guardrail
        def my_sync_guardrail(...): ...

        @input_guardrail(name="guardrail_name")
        async def my_async_guardrail(...): ...
    """

    def decorator(
        f: _InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co],
    ) -> InputGuardrail[TContext_co]:
        return InputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        # Decorator was used without parentheses
        return decorator(func)

    # Decorator used with keyword arguments
    return decorator


_OutputGuardrailFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    GuardrailFunctionOutput,
]
_OutputGuardrailFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    func: _OutputGuardrailFuncAsync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
    OutputGuardrail[TContext_co],
]: ...


def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co]
    | _OutputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    OutputGuardrail[TContext_co]
    | Callable[
        [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
        OutputGuardrail[TContext_co],
    ]
):
    """
    Decorator that transforms a sync or async function into an `OutputGuardrail`.
    It can be used directly (no parentheses) or with keyword args, e.g.:

        @output_guardrail
        def my_sync_guardrail(...): ...

        @output_guardrail(name="guardrail_name")
        async def my_async_guardrail(...): ...
    """

    def decorator(
        f: _OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co],
    ) -> OutputGuardrail[TContext_co]:
        return OutputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        # Decorator was used without parentheses
        return decorator(func)

    # Decorator used with keyword arguments
    return decorator


