# Token Measurement

Token measurement estimates whether a prompt fits a model before sending it. It is useful for choosing when to compact
history, reserving output space, or rejecting an oversized request without paying for a provider call.

The result is an estimate of the model-visible prompt surface. Provider billing remains authoritative.

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

Set `tokenizer_hf_repo` to a Hugging Face tokenizer repository when the model has a compatible chat template. Pin
`tokenizer_revision` when reproducible counts matter. Enable `tokenizer_trust_remote_code` only for a repository whose code
you trust.

With no Hugging Face repository, measurement uses the library's fallback tokenizer and a deterministic rendering of the
request.

## Measure a request

`measure_request_tokens` includes messages, tool definitions, and response format. Tokenizer-specific encodings can also use
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
tool schemas, and structured-output schemas are included in the measured surface.

## Measure prompt parts

Use `measure_prompt_tokens` when the code has prompt components but has not built a full request:

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

Use `estimate_text_tokens(text)` for a standalone string. It uses the fallback text encoding and returns at least one token,
including for empty or `None` input.

| Function | Measures |
| --- | --- |
| `estimate_text_tokens` | One text value with the fallback encoding |
| `measure_prompt_tokens` | Messages plus optional tools and response format |
| `measure_request_tokens` | The complete model-visible prompt in a request |
