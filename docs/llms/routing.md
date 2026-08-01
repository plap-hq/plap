# Routing

A direct `ChatCompletionClient` sends every request to one provider. `RoutingChatCompletionClient` selects a provider from
the model name and can try another model after a provider failure.

## Route by model prefix

Create one `ModelRoute` per prefix:

```python
from plap.llms.completions import ModelRoute, RoutingChatCompletionClient

client = RoutingChatCompletionClient(
    [
        ModelRoute(prefix="groq/", client=groq_client),
        ModelRoute(prefix="openrouter/", client=openrouter_client),
    ]
)
```

The router removes the matching prefix before calling the child. If several routes match, the longest prefix wins.

A route prefix is part of the application's model name, not the provider's model name. The result and streamed deltas use
the prefixed name again when they return through the router.

## Add a fallback chain

Separate model attempts with commas, in priority order:

```python
from dataclasses import replace

fallback_request = replace(
    request,
    model=(
        "groq/openai/gpt-oss-20b,"
        "openrouter/openai/gpt-oss-20b"
    ),
)
```

An attempt can fall back when its route is missing, its child does not support the model, or the child raises a completion
provider error. Missing routes are skipped; every provider in the model string does not need to be configured in the
current environment.

## Transport retries

Before moving to the next model, the router retries a timeout or plain `ChatCompletionProviderError` twice with exponential
jitter. Authentication, rate-limit, invalid-request, context-length, and unsupported-request errors move to the next model
immediately.

Router retries repeat a failed provider call. [Completion retries](retries.md) reject a completed call whose model answer
cannot be used.

## Streaming fallback

The router can change providers only before the caller receives model output:

If a provider fails before yielding content, reasoning, refusal, or tool-call output, the router can retry or fall back. If
the provider fails after yielding any of that output, the router raises the error instead of mixing output from another
model into the stream.

Empty metadata deltas do not lock the route. Content, reasoning, refusal, or tool-call deltas do. The router also applies a
first-output timeout and an idle timeout between later deltas.

Set `stream_first_delta_timeout_seconds=None` to disable the first-output timeout for a routing client. The idle timeout is a
library constant rather than a constructor option.

## Close routed clients

Calling `client.aclose()` closes all distinct child clients. If two prefixes refer to the same child object, it is closed
once.
