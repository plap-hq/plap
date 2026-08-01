# Streaming

Streaming exposes text, reasoning, and tool calls while the model is generating them. A live UI can render those updates
immediately. Application logic can use `Accumulator` to build a complete `ChatMessage` or `ChatCompletionResult` from the
deltas.

## Read deltas directly

Each delta contains the fields changed by one stream event:

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

A tool call may arrive across several deltas. Do not decode its argument fragments separately.

## Assemble a result

Create one `Accumulator` per completion stream. Pass the request tools so completed tool arguments can be recovered and
normalized with their schemas:

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

Before the terminal delta, `snapshot.messages` contains the current partial assistant message and `snapshot.result` is
`None`. When a delta contains `finish_reason`, the snapshot contains a `ChatCompletionResult` and exposes it through
`snapshot.result`.

`Accumulator` repairs JSON syntax and performs conservative schema-guided normalization for tool arguments. It does not
guarantee that the final arguments satisfy the tool schema. Use a [retry validator](retries.md) when invalid arguments should
cause another model attempt.

## Stream normalization

`ChatCompletionClient` normalizes provider streams before yielding deltas. It combines provider-specific terminal signals,
ensures tool-call streams finish as `tool_calls`, and raises a provider error when a stream ends without a usable terminal
state.

Provider transports close their active stream after normal completion, failure, or cancellation. A caller that abandons an
iterator early should close it explicitly. Code that owns the completion client must also call `client.aclose()` when the
client itself is no longer needed.
