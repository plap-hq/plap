# Messages and Requests

Provider SDKs use different Python objects for the same chat concepts. `plap.llms` defines one set of messages, tools,
requests, results, and stream deltas that every completion client accepts.

## Build a message history

A request contains messages in the order the model should read them:

```python
from plap.llms.completions import ChatCompletionRequest, ChatMessage

request = ChatCompletionRequest(
    model="openai/gpt-oss-20b",
    messages=[
        ChatMessage(role="developer", content="Answer in one sentence."),
        ChatMessage(role="user", content="What is a mutex?"),
    ],
)
```

Roles are `system`, `developer`, `user`, `assistant`, and `tool`. Message content can be a string, `None`, or a list of
structured content parts for text, images, files, and input audio. The content-part classes live in
`plap.llms.completions.chat`.

`ChatMessage.memory` stores application metadata beside a message. It is preserved by plap serialization but omitted from
provider requests.

## Add a function tool

A function definition is wrapped in `ChatTool` before it is added to a request:

```python
from plap.llms.completions import ChatFunctionTool, ChatTool

lookup = ChatTool(
    function=ChatFunctionTool(
        name="lookup",
        description="Look up a record by ID.",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        strict=True,
    )
)
```

Add the tool and a tool-selection rule with `dataclasses.replace`:

```python
from dataclasses import replace

request = replace(
    request,
    tools=[lookup],
    tool_choice="auto",
)
```

When the model calls it, `result.message.tool_calls` contains `ChatToolCall` values. `arguments` remains JSON text. After
executing the function, append the assistant message and a matching tool result before the next completion:

```python
from plap.llms.completions import ChatMessage

call = result.message.tool_calls[0]
next_request = replace(
    request,
    messages=[
        *request.messages,
        result.message,
        ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content="record contents",
        ),
    ],
)
```

The tool message must use the call's ID so the model can match the result to its request.

## Request options

`ChatCompletionRequest` groups options that providers commonly expose:

| Group | Fields |
| --- | --- |
| Output size | `max_completion_tokens` |
| Sampling | `temperature`, `top_p`, `min_p`, `top_k`, penalties, `seed`, `stop` |
| Tools | `tools`, `tool_choice`, `parallel_tool_calls` |
| Structured output | `response_format` |
| Reasoning | `reasoning_effort` |
| Streaming | `stream_options` |
| Provider metadata | `user`, `prompt_cache_key`, `metadata`, `service_tier` |
| Budgeting | `output_equivalence` |

Providers reject or adapt unsupported fields through their configured quirks. `output_equivalence` is required only by a
[budgeted client](budgeting.md).

## Read a result

`ChatCompletionResult` contains the accepted assistant message and final metadata:

```python
text = result.message.content
tool_calls = result.message.tool_calls
finish_reason = result.finish_reason
usage = result.usage
```

`finish_reason` explains why generation stopped, such as `stop`, `length`, `tool_calls`, or `content_filter`. `usage` may
include input, output, cached, and reasoning token counts when the provider reports them.

For streaming, the same information arrives incrementally as `ChatCompletionDelta` values. [Streaming](streaming.md)
explains how to consume those deltas and assemble a result.
