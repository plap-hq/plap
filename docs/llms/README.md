# The LLM Library

`plap.llms` makes chat-completion code independent of a provider SDK. The same request can be sent through OpenRouter,
Fireworks, or another configured provider, then processed with the same streaming, retry, JSON, routing, and budgeting
tools.

Use this library directly when an application needs chat completions without the Responses server, or when a plugin needs
an additional model call. A plugin call should use the `BudgetedChatCompletionClient` from `state.svcs` so it shares the
response budget.

## First completion

[Make your first completion](getting-started.md) constructs an OpenRouter client, sends one request, and streams another.

## Guides

| Goal | Guide |
| --- | --- |
| Build messages, tools, and requests | [Messages and requests](messages-and-requests.md) |
| Connect and close provider clients | [Providers](providers.md) |
| Select providers and configure fallback | [Routing](routing.md) |
| Consume deltas and assemble results | [Streaming](streaming.md) |
| Ask the model to correct an unusable result | [Retries](retries.md) |
| Decode, recover, normalize, and validate model JSON | [JSON](json.md) |
| Share one limit across several model calls | [Budgeting](budgeting.md) |
| Estimate prompt size before a call | [Token measurement](tokens.md) |

The common client contract is `IChatCompletionClient`:

```python
class IChatCompletionClient(Protocol):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult: ...
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]: ...
    async def aclose(self) -> None: ...
```

Provider clients, routers, and budgeted clients implement this contract, so callers can combine them without changing the
request or result types.
