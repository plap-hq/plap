# Providers

A completion provider translates the common `ChatCompletionRequest` into one model API's wire format. It also converts the
provider response and exceptions back into common plap types. Application code can therefore change providers without
adopting another SDK interface.

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

Built-in providers have explicit model tables. Requesting a model absent from that provider's table raises
`ChatCompletionUnsupportedRequestError` before the network call.

## Use another OpenAI-compatible endpoint

`OpenAIProvider` can connect to an endpoint that implements OpenAI Chat Completions. Its model table declares the names the
client may send:

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

The empty tuple means that `my-model` needs no model-specific quirks. Provider-wide and model-specific quirks can reject,
remove, rename, or add wire fields when an endpoint differs from the common request format.

For a non-OpenAI transport, subclass `Provider` and implement `complete`, `stream`, and `aclose`. The completion methods
return raw OpenAI-shaped dictionaries; `ChatCompletionClient` converts them into the common result and delta types.

## Handle errors

Provider implementations convert SDK exceptions into the completion error hierarchy:

| Error | Meaning |
| --- | --- |
| `ChatCompletionAuthenticationError` | The provider rejected the credentials |
| `ChatCompletionRateLimitError` | The provider applied a rate limit |
| `ChatCompletionContextLengthExceededError` | The prompt exceeded the model context window |
| `ChatCompletionInvalidRequestError` | The provider rejected another request field |
| `ChatCompletionTimeoutError` | A provider or transport timeout occurred |
| `ChatCompletionProviderError` | Another provider or transport failure occurred |
| `ChatCompletionUnsupportedRequestError` | No configured model or feature can handle the request |

Catch these errors instead of provider SDK exceptions. The router uses their types to decide whether to retry the same
provider or move to a fallback model.

## Close the client

`ChatCompletionClient.aclose()` closes the provider transport. Call it once when the application no longer needs the client:

```python
try:
    result = await client.complete(request)
finally:
    await client.aclose()
```

A [routing client](routing.md) closes each distinct child client once, including a child shared by several prefixes.
