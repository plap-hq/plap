# Make Your First Completion

Use this tutorial to call OpenRouter through `plap.llms` without starting the Responses server. It shows the provider client,
common request type, and streaming interface in isolation.

## 1. Set the API key

Add an OpenRouter key to root `.env`:

```dotenv
OPENROUTER_API_KEY=your-key
```

Load it into the shell:

```sh
set -a
source .env
set +a
```

## 2. Create a client and request

Run this from the repository root:

```sh
pixi run python - <<'PY'
import os

import anyio

from plap.llms.completions import ChatCompletionClient, ChatCompletionRequest, ChatMessage
from plap.llms.completions.providers import build_openrouter_provider


async def main() -> None:
    provider = build_openrouter_provider(
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    client = ChatCompletionClient(provider)
    request = ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[
            ChatMessage(role="user", content="Say hello in one sentence."),
        ],
    )

    try:
        result = await client.complete(request)
        print(result.message.content)
    finally:
        await client.aclose()


anyio.run(main)
PY
```

`ChatCompletionRequest` uses the provider's model name because this client talks directly to OpenRouter. A
`RoutingChatCompletionClient` adds provider prefixes such as `openrouter/`; [Routing](routing.md) explains that layer.

The `finally` block closes the HTTP client even when the completion fails.

## 3. Stream the response

`complete()` waits for a final `ChatCompletionResult`. `stream()` yields `ChatCompletionDelta` values as the provider sends
them. Replace the `try` block above with:

```python
try:
    async for delta in client.stream(request):
        if delta.content_delta is not None:
            print(delta.content_delta, end="", flush=True)
    print()
finally:
    await client.aclose()
```

A delta contains only the fields updated by that stream event. Use an [Accumulator](streaming.md#assemble-a-result) when code
needs the current assembled message or a final result while streaming.

## Continue learning

You can now send a provider-neutral completion and read it all at once or as a stream.

Continue with [Messages and requests](messages-and-requests.md) to build richer conversations with developer instructions,
tools, and structured content.
