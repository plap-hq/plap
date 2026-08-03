# The LLM Library

A provider SDK works well when every completion goes to one provider. plap may route the same completion through another
provider, so SDK request types, response objects, stream events, and exceptions cannot become part of the calling code.

`plap.llms` converts those provider differences into three common types: `ChatCompletionRequest`, `ChatCompletionResult`, and
`ChatCompletionDelta`. Provider clients perform the wire translation. Routers, retries, and completion budgets use the same
types, so they can be composed without introducing provider-specific branches.

Use the library directly for chat completions outside the Responses server. Inside a response plugin, get
`BudgetedChatCompletionClient` from `state.svcs`; that client charges the additional call to the response budget.

## Client contract

Every provider client, router, and budgeted client implements `IChatCompletionClient`:

```python
class IChatCompletionClient(Protocol):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult: ...
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]: ...
    async def aclose(self) -> None: ...
```

Callers choose a client without changing how they build requests or consume results.

## First completion

[Make your first completion](getting-started.md) constructs an OpenRouter client, sends one request, and streams another.

## Guides

| Goal | Guide |
| --- | --- |
| Build messages, tools, and requests | [Messages and requests](messages-and-requests.md) |
| Connect and close provider clients | [Providers](providers.md) |
| Understand which provider and model combinations may run | [Model whitelist](whitelist.md) |
| Select providers and configure fallback | [Routing](routing.md) |
| Consume deltas and assemble results | [Streaming](streaming.md) |
| Ask the model to correct an unusable result | [Retries](retries.md) |
| Decode, recover, normalize, and validate model JSON | [JSON](json.md) |
| Share one limit across several model calls | [Budgeting](budgeting.md) |
| Estimate prompt size before a call | [Token measurement](tokens.md) |
