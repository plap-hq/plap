# Response State

The `server_time` tool ignored its `state` argument because its result depended only on the clock. Other tools and hooks need
information from the current response. They may inspect the caller's request, read the selected model configuration, carry
data between requests, change model context, or make another budgeted model call.

`State` is the shared context for that work during one response.

## Inspect the caller's request

`state.request` is the `ResponseCreateRequest` accepted for the current response:

```python
model = state.request.model
cache_key = state.request.prompt_cache_key
requested_tools = state.request.tools
```

These values come from the Responses API request. They are separate from the provider-facing `ChatCompletionRequest` built
by `response.request`.

## Read the selected configuration

plap resolves CUE overlays for the requested public model, reasoning effort, and service tier. `state.config` contains the
resolved values:

```python
main_config = state.config.main
provider_model = main_config.model
max_tokens = main_config.max_completion_tokens
```

A plugin can add fields through its own CUE schema. Register the schema directory with
[`bootstrap.config`](easy/bootstrap.md#load-cue-configuration).

## Carry data between requests

`state.memory` carries plugin-owned JSON data from one request to the next when the caller continues a response. It stores
data that does not belong in the model's message history:

```python
plugin_memory = state.memory.setdefault("my_plugin", {})
plugin_memory["review_count"] = plugin_memory.get("review_count", 0) + 1
```

Store each plugin's data under its own key. Values must contain JSON data, not clients, locks, or other live Python objects.

## Attach data to one message

`ChatMessage.memory` stores JSON metadata with one message instead of the whole response:

```python
from plap.llms.completions import ChatMessage

message = ChatMessage(
    role="assistant",
    content="Reviewed answer",
    memory={"my_plugin": {"approved": True}},
)
```

Message memory is saved and replayed with that message. Providers do not receive it.

## Read or change model context

`state.threads` contains the message histories maintained during the response. Core builds the main model request from
`state.threads["main"]`:

```python
main_history = state.threads["main"]
pending_main_calls = state.open_calls("main")
```

Additional [threads](threads.md) keep auxiliary model histories separate from the main model's context.

## Use a registered service

`state.svcs` contains services available during the current response. An additional model call uses the budgeted completion
client so it consumes the same completion budget as the main model:

```python
from plap.llms.completions import BudgetedChatCompletionClient

client = await state.svcs.aget(BudgetedChatCompletionClient)
```

Plugins register their own services with [`bootstrap.services`](easy/bootstrap.md#register-services).

## Persist state changes

Hooks and server tools mutate the live `State`. Three methods write those changes to response output:

| Method | Effect |
| --- | --- |
| `save_progress()` | Start or replace an in-progress reasoning item with the current memory and threads |
| `ensure_progress()` | Attempt `save_progress()` when no reasoning item is open |
| `commit()` | Finalize current state and publish eligible main messages and client-tool calls |

The default response flow calls `commit()` after the response loop stops. A plugin that emits intermediate output can call
`ensure_progress()` before the first event and `save_progress()` after a complete chunk so the output remains attached to
the state that produced it.
The [reasoning-summary pathway](summary.md) uses this pattern to stream progress before the main answer is committed.
