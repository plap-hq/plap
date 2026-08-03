# Streaming

A provider stream contains partial updates rather than complete messages. Text may be split across many events, tool arguments
may arrive one fragment at a time, and usage may arrive after the finish reason.

`ChatCompletionClient` converts provider events into `ChatCompletionDelta`. Code that only needs live updates can read those
deltas directly. `Accumulator` joins text, refusal, reasoning, tool-call, metadata, and usage fragments into snapshots and a
final result.

## Read deltas directly

Each delta contains only the fields changed by one stream event:

```python
from plap.llms.completions import ChatCompletionRequest, IChatCompletionClient


async def print_stream(
    client: IChatCompletionClient,
    request: ChatCompletionRequest,
) -> None:
    async for delta in client.stream(request):
        if delta.content_delta is not None:
            print(delta.content_delta, end="", flush=True)
```

The main incremental fields are:

| Field | Carries |
| --- | --- |
| `content_delta` | Assistant text |
| `reasoning_delta` | Hidden reasoning text reported by the provider |
| `refusal_delta` | Refusal text |
| `tool_call_delta` | Tool-call ID, name, or argument fragment |
| `finish_reason` | The terminal reason for the completion |
| `usage` | Token usage when the provider reports it |

A tool call may span several deltas. Decode its arguments only after accumulation has joined the fragments.

## Assemble a result

Create one `Accumulator` for each completion stream. Pass the request tools so completed tool arguments can be recovered and
normalized against their schemas:

```python
from plap.llms import Accumulator
from plap.llms.completions import (
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)


async def collect_stream(
    client: IChatCompletionClient,
    request: ChatCompletionRequest,
) -> ChatCompletionResult:
    accumulator = Accumulator(tools=tuple(request.tools))
    final: ChatCompletionResult | None = None

    async for delta in client.stream(request):
        snapshot = accumulator.apply(delta)
        if snapshot.result is not None:
            final = snapshot.result

    if final is None:
        raise RuntimeError("completion stream ended without a result")
    return final
```

Before the terminal delta, `snapshot.messages` contains the current partial assistant message and `snapshot.result` is `None`.
A delta with `finish_reason` produces a `ChatCompletionResult` in `snapshot.result`.

`Accumulator` repairs JSON syntax and performs conservative schema-guided normalization for tool arguments. It does not prove
that the final arguments satisfy the tool schema. Use a [retry validator](retries.md) when invalid arguments should cause
another model attempt.

## Provider stream normalization

Providers disagree about terminal events and trailing usage. `ChatCompletionClient` normalizes those differences before it
yields deltas. It ensures tool-call streams finish with `tool_calls` and raises a provider error when a stream ends without a
usable terminal state.

Provider transports close their active stream after completion, failure, or cancellation. A caller that abandons an iterator
early should close that iterator. Code that owns the completion client must also call `client.aclose()` when the client is no
longer needed.
