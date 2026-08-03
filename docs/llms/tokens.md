# Token Measurement

Provider usage arrives after a model call, too late to decide whether the prompt should be compacted or rejected before
generation. Token measurement estimates the prompt size before the request is sent.

The estimate covers the model-visible request. It is suitable for preflight decisions, but provider billing remains the
authoritative usage record.

## Configure the tokenizer

Measurement accepts any object implementing `ITokenizerConfig`:

```python
from dataclasses import dataclass

from plap.llms.completions import ITokenizerConfig


@dataclass(frozen=True)
class TokenizerConfig(ITokenizerConfig):
    tokenizer_hf_repo: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
```

Set `tokenizer_hf_repo` when the model has a compatible Hugging Face tokenizer and chat template. Pin
`tokenizer_revision` for reproducible counts. Enable `tokenizer_trust_remote_code` only for a repository whose code you trust.

Without a Hugging Face repository, measurement uses the fallback tokenizer and a deterministic rendering of the request.

## Measure a request

`measure_request_tokens` includes messages, tool definitions, and response format. Tokenizer-specific encodings may also use
the request's reasoning mode:

```python
from plap.llms.completions import (
    ChatCompletionRequest,
    ChatMessage,
    measure_request_tokens,
)

request = ChatCompletionRequest(
    model="openai/gpt-oss-20b",
    messages=[
        ChatMessage(role="user", content="Explain tokenization."),
    ],
)

tokens = measure_request_tokens(
    request,
    tokenizer_config=TokenizerConfig(),
)
```

Message memory is excluded because providers do not receive it. Tool-call arguments, reasoning content, structured content,
tool schemas, and structured-output schemas are included.

## Measure prompt parts

Use `measure_prompt_tokens` before a full request has been built:

```python
from plap.llms.completions import measure_prompt_tokens

tokens = measure_prompt_tokens(
    request.messages,
    tokenizer_config=TokenizerConfig(),
    tools=request.tools,
    response_format=request.response_format,
    reasoning_effort=request.reasoning_effort,
)
```

Use `estimate_text_tokens(text)` for one standalone string. It uses the fallback text encoding and returns at least one token,
including for empty or `None` input.

| Function | Measures |
| --- | --- |
| `estimate_text_tokens` | One text value with the fallback encoding |
| `measure_prompt_tokens` | Messages plus optional tools and response format |
| `measure_request_tokens` | The complete model-visible prompt in a request |
