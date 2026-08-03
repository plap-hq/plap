# Providers

Provider SDKs disagree on request fields, response objects, stream events, and exceptions. If those SDK types reach the
calling code, changing providers requires changes throughout the application.

A `Provider` owns the transport for one model API. `ChatCompletionClient` applies the provider and model quirks before the
call, then converts the raw response into `ChatCompletionResult` or `ChatCompletionDelta`. Callers only work with the common
completion types.

## Use a built-in provider

Provider builders are available from `plap.llms.completions.providers`:

| Builder | Service |
| --- | --- |
| `build_openrouter_provider` | OpenRouter |
| `build_vercel_provider` | Vercel AI Gateway |
| `build_fireworks_provider` | Fireworks |
| `build_groq_provider` | Groq |
| `build_cerebras_provider` | Cerebras |
| `build_lightning_provider` | Lightning AI |
| `build_gmicloud_provider` | GMI Cloud |
| `build_novita_provider` | Novita AI |
| `build_crof_provider` | CROF |
| `build_qubrid_provider` | Qubrid AI |
| `build_wandb_provider` | Weights & Biases Inference |

Each builder accepts an API key and returns a `Provider`. Wrap it in `ChatCompletionClient` before sending requests, as shown
in [Make your first completion](getting-started.md).

Built-in providers use a [model whitelist](whitelist.md). A model absent from the provider's table raises
`ChatCompletionUnsupportedRequestError` before a network call.

## Use another OpenAI-compatible endpoint

`OpenAIProvider` supplies the transport for an endpoint that implements OpenAI Chat Completions. The `models` table is
required because sharing the OpenAI wire format does not guarantee that every model handles the same request fields.

```python
import os

from plap.llms.completions import ChatCompletionClient, OpenAIProvider

provider = OpenAIProvider(
    name="internal",
    api_key=os.environ["INTERNAL_LLM_API_KEY"],
    base_url="https://llm.example.com/v1",
    models={
        "my-model": (),
    },
)
client = ChatCompletionClient(provider)
```

The empty tuple admits `my-model` with no model-specific quirks. Provider-wide and model-specific quirks can move, remove,
set, or reject fields when the endpoint differs from the common completion shape. [Model whitelist](whitelist.md) explains
how those rules are assigned.

For a non-OpenAI transport, subclass `Provider` and implement `complete`, `stream`, and `aclose`. The completion methods return
raw OpenAI-shaped dictionaries; `ChatCompletionClient` converts them into the common result and delta types.

## Handle errors

Providers convert SDK exceptions into the completion error hierarchy:

| Error | Meaning |
| --- | --- |
| `ChatCompletionAuthenticationError` | The provider rejected the credentials |
| `ChatCompletionRateLimitError` | The provider applied a rate limit |
| `ChatCompletionContextLengthExceededError` | The prompt exceeded the model context window |
| `ChatCompletionInvalidRequestError` | The provider rejected another request field |
| `ChatCompletionTimeoutError` | A provider or transport timeout occurred |
| `ChatCompletionProviderError` | Another provider or transport failure occurred |
| `ChatCompletionUnsupportedRequestError` | The provider does not accept the model or one of its request options |

Catch these errors instead of SDK exceptions. A [routing client](routing.md) uses the error type to decide whether to retry the
same provider or move to the next model.

## Close the client

`ChatCompletionClient.aclose()` closes its provider transport:

```python
try:
    result = await client.complete(request)
finally:
    await client.aclose()
```

The code that creates the client owns its lifetime. A routing client closes each distinct child once, including a child shared
by several route prefixes.
