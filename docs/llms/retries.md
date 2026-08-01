# Completion Retries

A provider call can succeed while the model result is unusable. The model might call an unknown tool, return arguments that
do not match the tool schema, or ignore a required tool choice. `plap.llms.retry` can add a correction message and ask the
model to try the same task again.

Routing fallback handles provider and transport failures. Completion retries handle model output after a call has produced
a result. The retry stream also restarts an attempt that fails after partial output but before a final result.

## Use the built-in validators

`retry_on_tool_choice_mismatch` checks `tool_choice` and `parallel_tool_calls`.
`retry_on_unusable_tool_calls` checks tool names, JSON-object arguments, and declared JSON Schemas.

```python
from plap.llms import (
    complete as complete_with_retries,
    retry_on_tool_choice_mismatch,
    retry_on_unusable_tool_calls,
)

snapshot = await complete_with_retries(
    client,
    request,
    validators=(
        retry_on_tool_choice_mismatch,
        retry_on_unusable_tool_calls,
    ),
    max_attempts=3,
)

result = snapshot.result
if result is None:
    raise RuntimeError("completion ended without an accepted result")
```

`complete_with_retries` uses the client's streaming method and returns the last `Snapshot`, not a bare
`ChatCompletionResult`.

## Write a validator

A validator returns `None` to accept a result or a correction message to reject it:

```python
from plap.llms.completions import ChatCompletionRequest, ChatCompletionResult


async def require_text(
    result: ChatCompletionResult,
    request: ChatCompletionRequest,
) -> str | None:
    _ = request
    content = result.message.content
    if isinstance(content, str) and content.strip():
        return None
    return "Reply with a non-empty text answer."
```

Validators run in tuple order. The first correction message rejects the attempt; later validators do not run for that
result.

## What enters the next attempt

When a validator rejects a result, the next request contains:

1. The original request messages.
2. Messages from earlier rejected attempts.
3. Placeholder tool results for rejected tool calls.
4. The validator's correction as a user message.

The placeholders prevent an unresolved tool call from corrupting the retry history. They state that the rejected call was
not executed.

## Observe retry streams

`plap.llms.stream` yields `Snapshot` values throughout every attempt. A snapshot can contain partial messages, completed
attempt results, or the correction history inserted between attempts:

```python
from plap.llms import stream as stream_with_retries

async for snapshot in stream_with_retries(
    client,
    request,
    validators=(require_text,),
):
    if snapshot.messages:
        print(snapshot.messages[-1].content)
```

If all attempts are rejected, the iterator raises `RetryLimitExceededError`. Its `last_retry_message` contains the final
validator correction. An invalid tool schema raises `RetryToolSchemaError` instead of asking the model to satisfy a schema
that could not be compiled.
