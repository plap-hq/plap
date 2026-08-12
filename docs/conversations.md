# Continue a Conversation

The examples in the [README](../README.md#send-a-response) send one prompt and receive one answer. A follow-up needs the
earlier turn too, including private model and plugin state that is not present in the answer text.

Each API has its own way to return that turn to plap.

## Responses

plap stores Responses API output by default. Pass the first response ID with the user's follow-up:

```python
first = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    input="Say hello in one sentence.",
)

continued = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    previous_response_id=first.id,
    input="Now say it in French.",
)

print(continued.output_text)
```

plap loads the stored output and private state before handling the new input. The client does not rebuild the earlier turn.

### Keep the state in the client

A client that does not want plap to store responses can carry the turn itself. Ask for encrypted reasoning in the first
response, then return that output before the follow-up:

```python
first = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    input="Say hello in one sentence.",
    include=["reasoning.encrypted_content"],
    store=False,
)

continued = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    input=[
        {"role": "user", "content": "Say hello in one sentence."},
        *first.output,
        {"role": "user", "content": "Now say it in French."},
    ],
    include=["reasoning.encrypted_content"],
    store=False,
)

print(continued.output_text)
```

The reasoning items in `first.output` contain opaque `encrypted_content`. Return those items unchanged; plap authenticates
them before restoring the private state they contain.

## Chat Completions

Chat Completions has no previous response ID, so the client sends the transcript with each request. Assistant messages
returned by plap may contain OpenRouter-style `reasoning_details`; preserve that list when adding the assistant message to
the transcript:

```python
first = client.chat.completions.create(
    model=os.environ["PLAP_DEV_MODEL"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
assistant = first.choices[0].message

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

print(continued.choices[0].message.content)
```

Encrypted entries in `reasoning_details` carry private model and plugin state. Omitting them continues from the visible
transcript without that state. Altering one causes plap to reject the request before calling a model.

## Streaming

plap uses the OpenAI SDK's normal streaming interface. The SDK's [Streaming responses](https://github.com/openai/openai-python#streaming-responses)
example works with the same client configuration used throughout this guide.
