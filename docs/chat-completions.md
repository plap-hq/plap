# Use Chat Completions

If an application already uses the OpenAI Chat Completions API, it can call plap without first changing to the Responses API.

After [starting plap](../README.md#start-plap), point the OpenAI client at the development server:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["PLAP_DEV_BASE_URL"],
    api_key=os.environ["PLAP_DEV_API_KEY"],
)

completion = client.chat.completions.create(
    model=os.environ["PLAP_DEV_MODEL"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(completion.choices[0].message.content)
```

## Continue a conversation

Send the earlier messages again when the user continues the conversation. Assistant messages returned by plap may include
OpenRouter-style `reasoning_details`; include that list unchanged with the assistant message:

```python
assistant = completion.choices[0].message

continued = client.chat.completions.create(
    model=os.environ["PLAP_DEV_MODEL"],
    messages=[
        {"role": "user", "content": "Say hello in one sentence."},
        {
            "role": "assistant",
            "content": assistant.content,
            "reasoning_details": assistant.reasoning_details,
        },
        {"role": "user", "content": "Now say it in French."},
    ],
)
```

Keeping the complete list preserves private reasoning and plugin state across turns. There are two distinct fallback and error
cases:

- If encrypted details are omitted, the request remains valid. plap continues from the visible assistant content, refusal, and
  tool calls, but private state from that turn is lost. Any remaining summary or text details are not trusted as private state.
- If an encrypted detail is present but altered or invalid, plap rejects the request with `invalid_reasoning_replay` before
  calling a model.

This means clients may deliberately strip encrypted state, but they must not modify encrypted values they send back.

## Stream a response

Reasoning details can arrive on separate streaming chunks. Collect them in order if the streamed assistant message will be
used in a later request:

```python
stream = client.chat.completions.create(
    model=os.environ["PLAP_DEV_MODEL"],
    messages=[{"role": "user", "content": "Explain the result."}],
    stream=True,
)

text = []
reasoning_details = []

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        text.append(delta.content)
    reasoning_details.extend(getattr(delta, "reasoning_details", None) or [])
```

The OpenAI Python client exposes `reasoning_details` as an extra field, so `getattr` works across client versions whose static
types do not yet declare the OpenRouter extension.

## Return tool results

Use the returned `tool_calls[].id` unchanged in both the assistant tool call and the corresponding tool message. Include the
assistant's complete `reasoning_details` when private state should continue through the tool round trip.

A transcript imported from another service is also valid. Its visible messages and tool calls are treated as conversation
history even though it has no plap reasoning state.

## Compatibility limits

The endpoint supports one completion choice, function tools, text/image/file input, structured text output, reasoning effort
and summaries, streaming usage, and text log probabilities. Unsupported or unknown fields are rejected rather than silently
discarded.

Chat requests default to `store=False`. Set `store=True` only when the response also needs to be retrievable later by its ID.
