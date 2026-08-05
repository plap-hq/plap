# Completion Retries

A provider can complete a request successfully and still return an answer the application cannot use. A tool name may be
unknown, its arguments may fail the schema, or the model may ignore a required tool choice.

Transport retries repeat a call that failed to complete. Completion retries handle the different case: they preserve the
rejected answer, add a correction message, and ask the model to try the task again.

A retry validator inspects a completed attempt. It returns nothing when the result can be used, or a correction when the
model should try again.

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

`complete_with_retries` uses the client's streaming method and returns the last `Snapshot`. The accepted
`ChatCompletionResult`, when present, is available as `snapshot.result`.

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

Validators run in tuple order. The first correction rejects the attempt, so later validators do not run for that result.

## What enters the next attempt

When a validator rejects a result, the next model request contains:

1. The original request messages.
2. Messages from earlier rejected attempts.
3. Placeholder tool results for rejected tool calls.
4. The validator's correction as a user message.

The placeholders close rejected tool calls without pretending that the application executed them. The model can then read the
failed attempt and the correction as one valid conversation history.

## Observe retry streams

`plap.llms.stream` yields `Snapshot` values from every attempt, including partial messages and the correction history inserted
between attempts:

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

The retry stream also starts another attempt when a provider fails after partial output but before producing a final result.
If every completed attempt is rejected, it raises `RetryLimitExceededError`; `last_retry_message` contains the final correction.
An invalid tool schema raises `RetryToolSchemaError` immediately because another model attempt cannot repair the schema itself.
