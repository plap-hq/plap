# Multimodal Conformance

This directory owns the llms-layer chat-completions lowering policy.

The client-facing Responses contract may be wider than generic Chat Completions.
The runtime stays chat-oriented, and this layer is where the current wire-shape
conformance rules live.

## Source Of Truth

- Internal runtime content model: `src/plap/llms/completions/chat.py`
- Generic request-body lowering: `src/plap/llms/completions/common.py`
- Token-surface rendering: `src/plap/llms/completions/tokens.py`
- Provider-local request quirks: `src/plap/llms/completions/providers/`

## Generic Wire Coercions

These are backend-agnostic chat-wire coercions applied before provider-specific
quirks.

- `image_url.detail == "original"` is lowered to `"high"`.
- `file.detail` is dropped.
- `system` / `developer` messages with mixed text and non-text content are
  lowered into multiple wire messages:
  - contiguous text runs stay same-role multipart text messages
  - contiguous non-text runs become synthetic `user` messages carrying only the
    media parts
- Direct string content stays direct string content. It is not wrapped into a
  one-element content array.

## Provider-Specific Request Rewrites

### OpenRouter

File parts get one provider-local rewrite:

- `file.file_url -> file.file_data`

OpenRouter currently remains the only backend where `file.file_url` and
`file.file_id` are permitted by plap.

### OpenAI-Style Providers

The following providers reject file reference fields on request shaping:

- Lightning
- Cerebras
- Groq
- GMICloud
- Novita
- Crof
- Qubrid
- Fireworks
- Vercel AI Gateway

Rejected file fields:

- `file.file_id`
- `file.file_url`

These rejections happen in llms-layer provider quirks, not in request prep,
store, or ingest.

### Vercel AI Gateway

In addition to rejecting `file.file_id` and `file.file_url`, Vercel rewrites:

- `file.file_data -> file.data`

## Footguns

`file_id` and `file_url` are not generic portable chat semantics just because a
particular provider accepts them.

- `file_id` is effectively a provider/account-local opaque handle.
- `file_url` is a provider-local file reference convention unless the backend
  explicitly defines a lowering for it.

OpenRouter support does not make either field safe across provider fallback
chains. Treat them as provider-native references.

## Token Counting

Token counting follows the same message lowering as real request serialization.

- `tokens.py` reuses `wire_messages(...)` from `common.py`.
- DSV4 counting keeps the DSV4 path alive for structured content by projecting
  non-text parts to deterministic text fragments locally in `tokens.py`.
- Non-DSV4 template counting first tries the native structured template surface,
  then a projected text surface, and only then falls back to JSON.

These token projections are counting aids only. They are not additional runtime
or request-shaping semantics.

## Remaining Work

### Multimodal Tool Output

This is the main unresolved multimodal bucket.

Current state:

- Responses ingest preserves structured `function_call_output` content in the
  internal chat runtime.
- Generic chat backends still have a much narrower `role="tool"` wire shape.

What still needs to be decided and implemented:

- whether OpenRouter can carry structured multimodal tool output directly in
  tool messages
- what the generic OpenAI-style / Vercel lowering should be for multimodal tool
  outputs while preserving the required contiguous tool-response protocol after
  an assistant `tool_calls` message
- how attribution should be surfaced to the model when a single assistant turn
  opens multiple tool calls and later media needs to be associated back to the
  right tool result

Until that work is done, multimodal tool output should be treated as unresolved
wire-shape behavior rather than assumed portable support.
