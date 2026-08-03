# Model Whitelist

Most LLM proxies begin with a provider and model name, then use a provider adapter to send the request. Model-specific
differences are handled when the call is made: unsupported parameters may be translated, rejected, or dropped. General-purpose
gateways such as LiteLLM use this approach to expose many provider models through one API.

plap requires the provider layer to remove those differences. Everything above `plap.llms` uses the same request, result, and
delta types. Fields such as `reasoning_effort`, hidden reasoning, and tool choice must keep the same meaning and shape regardless
of which provider handles the call.

Adding a model to plap therefore includes the rules needed to produce that standard behavior. Provider and model quirks
translate requests and outputs or reject behavior that cannot be represented correctly. The whitelist names the provider/model
pairs with those rules. Allowing arbitrary models through would force provider-specific handling into every caller above
`plap.llms`.

## How a request is admitted

[Routing](routing.md) chooses a provider client from a prefix such as `groq/`. The provider whitelist then checks the remaining
model name.

For `groq/openai/gpt-oss-20b`, plap:

1. Selects the `groq/` route and removes the prefix.
2. Looks up `openai/gpt-oss-20b` in the Groq provider's model table.
3. Combines Groq's provider quirks with the quirks for that model.
4. Builds the provider request and starts the network call.

`Provider.lookup()` uses an exact dictionary lookup by default. A provider with no `models` entries rejects every model.

OpenRouter and Vercel parse routing suffixes before lookup, then check the base model against their tables. A suffix cannot
make an unknown base model valid.

## What a model entry contains

The key in a provider's `models` dictionary is an accepted model name. The value is a tuple of model-specific `Quirk` objects.

The table is application code rather than a runtime copy of the provider catalog. A newly advertised model remains unavailable
until its normalization rules have been added.

An empty tuple admits the model using only the provider-wide quirks. A non-empty tuple adds rules for differences such as
structured output, reasoning, tool choice, or local rate limits.

Provider-wide quirks run first. Built-in providers use them for behavior shared by the endpoint, including `Only(...)`, which
removes top-level request fields absent from the provider's field list. Model quirks then translate or reject behavior that
differs for one model.

## Configure a custom whitelist

This provider accepts two models with different structured-output behavior:

```python
import os

from plap.llms.completions import OpenAIProvider
from plap.llms.completions.quirks import Only, RejectResponseFormat

provider = OpenAIProvider(
    name="internal",
    api_key=os.environ["INTERNAL_LLM_API_KEY"],
    base_url="https://llm.example.com/v1",
    quirks=(
        Only(
            "model",
            "messages",
            "stream",
            "stream_options",
            "response_format",
        ),
    ),
    models={
        "chat-v1": (),
        "chat-v2": (RejectResponseFormat("json_schema"),),
    },
)
```

`chat-v1` uses the provider-wide field filter. `chat-v2` also rejects JSON Schema output. Any other model name raises
`ChatCompletionUnsupportedRequestError` before the endpoint receives a request.

## Rejection and fallback

An unknown model and a known model with an unsupported request option both raise `ChatCompletionUnsupportedRequestError`. The
error message identifies whether lookup failed or a quirk rejected the option.

`RoutingChatCompletionClient` moves to the next fallback after this error. Each fallback still needs a matching route and a
matching entry in the child provider's model table.

## Extend a built-in whitelist

Built-in provider modules define tables such as `GROQ_MODELS`, `OPENROUTER_MODELS`, and `VERCEL_MODELS`. To add a model:

1. Add its exact provider-facing name to the model table.
2. Use an empty tuple when the provider-wide quirks cover the model.
3. Add model quirks for behavior that requires translation or rejection.
4. Test lookup and the resulting request body or local error.

Provider field lists are changed separately. Adding a field to a provider's `Only(...)` list allows it through for every model
on that provider; model quirks can then reject or translate it as needed.
