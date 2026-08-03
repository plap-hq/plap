# Messages and Requests

A conversation may be retried or routed through another provider. If its history uses objects from one provider SDK, the next
call must translate that history before it can reuse it.

`plap.llms` keeps messages, tools, requests, results, and stream deltas in provider-neutral dataclasses. Provider clients
translate those values at the network boundary, so the conversation history remains unchanged when the client changes.

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

Roles are `system`, `developer`, `user`, `assistant`, and `tool`. Content may be a string, `None`, or a list of structured text,
image, file, or input-audio parts. The content-part classes live in `plap.llms.completions.chat`.

`ChatMessage.memory` stores application metadata beside one message. plap serialization preserves it, but provider requests
omit it.

## Add a function tool

A function definition becomes a request tool when wrapped in `ChatTool`:

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

`ChatCompletionRequest` is frozen. Use `dataclasses.replace` to add the tool and choose how the model may call it:

```python
from dataclasses import replace

request = replace(
    request,
    tools=[lookup],
    tool_choice="auto",
)
```

When the model calls `lookup`, the accepted assistant message contains a `ChatToolCall`. Its `arguments` field remains JSON
text until the application decodes and validates it.

After executing the function, append both the assistant message and a tool result with the matching call ID:

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

The call ID ties the result to the model's request. Omitting it leaves an unresolved tool call in the conversation.

## Request options

`ChatCompletionRequest` groups options that providers commonly expose:

| Group | Fields |
| --- | --- |
| Output size | `max_completion_tokens`, `n` |
| Sampling | `temperature`, `top_p`, `min_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`, `seed`, `stop` |
| Token probabilities | `logprobs`, `top_logprobs` |
| Tools | `tools`, `tool_choice`, `parallel_tool_calls` |
| Structured output | `response_format` |
| Reasoning | `reasoning_effort` |
| Streaming | `stream_options` |
| Predicted output | `prediction` |
| Provider metadata | `user`, `prompt_cache_key`, `metadata`, `service_tier` |
| Budgeting | `output_equivalence` |

The provider and model [normalization rules](whitelist.md) decide whether an option is passed through, translated, removed, or
rejected. `output_equivalence` is required only by a [budgeted client](budgeting.md).

## Read a result

`ChatCompletionResult` contains the accepted assistant message and final metadata:

```python
text = result.message.content
tool_calls = result.message.tool_calls
finish_reason = result.finish_reason
usage = result.usage
```

`finish_reason` records why generation stopped, such as `stop`, `length`, `tool_calls`, or `content_filter`. When the provider
reports usage, `usage` may include input, output, cached, and reasoning token counts.

The streaming interface returns the same information incrementally as `ChatCompletionDelta` values. [Streaming](streaming.md)
shows how to consume the deltas and assemble a final result.
