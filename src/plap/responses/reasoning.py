from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

import msgspec

from plap.llms.chat import ChatCompletionRequest, ChatMessage, IChatCompletionClient, ReasoningEffort, ServiceTier
from plap.responses.contracts import ReasoningSummary
from plap.responses.models import ReasoningMessagePatch, Side, StateMessage

REASONING_SUMMARY_PROMPT = """Rewrite private mixed-perspective reasoning messages
into a user-facing reasoning summary.

The input is raw private evidence, not a conversation to continue and not text
to quote. A single trace can mix self-talk, critique addressed to a draft,
draft-like assistant text, tool observations, and notes about the user's request.
Pronouns such as "I", "you", "we", and "your answer" are local to those private
messages and are not reliable public speaker identities.

Rules:
- Do not quote hidden reasoning, hidden messages, or internal instructions
  verbatim.
- Do not expose chain-of-thought. Summarize only high-level checks and outcomes.
- Do not preserve the speaker, addressee, accusation, or conversational turn from
  any private message.
- Do not directly address the user with "you" or "your" in the reasoning summary.
- Treat criticism, corrections, second-person comments, and draft-like text as
  signals about what the assistant checked, revised, or decided.
- Preserve useful conclusions, risk checks, corrected assumptions, and tool
  outcomes.

Write the summary as the final public assistant's self-review: "I checked...",
"I considered...", "I corrected...", "I compared...", "I used the tool result...",
or "I chose...".

Style:
- Write as the assistant speaking naturally in first person.
- Be concise for concise summaries.
- For detailed summaries, include important checks and revisions while still
  hiding raw reasoning and internal roles.
- If there is nothing useful to summarize, return a short neutral sentence such
  as: "I checked the response for consistency and safety before answering."
"""


@runtime_checkable
class IReasoningSummarizer(Protocol):
    def stream(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        side: Side,
        messages: Sequence[StateMessage | ReasoningMessagePatch],
    ) -> AsyncIterator[str]: ...


class LLMReasoningSummarizer(IReasoningSummarizer):
    def __init__(self, client: IChatCompletionClient) -> None:
        self._client = client

    async def stream(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        side: Side,
        messages: Sequence[StateMessage | ReasoningMessagePatch],
    ) -> AsyncIterator[str]:
        async for delta in self._client.stream(
            ChatCompletionRequest(
                max_completion_tokens=_summary_max_tokens(mode),
                messages=[
                    ChatMessage(
                        role="developer",
                        content=REASONING_SUMMARY_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=_summary_request_text(mode, side, messages),
                    ),
                ],
                model=model,
                prompt_cache_key=prompt_cache_key,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                temperature=0,
            )
        ):
            if delta.content_delta:
                yield delta.content_delta


class NullReasoningSummarizer(IReasoningSummarizer):
    async def stream(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        side: Side,
        messages: Sequence[StateMessage | ReasoningMessagePatch],
    ) -> AsyncIterator[str]:
        _ = model, prompt_cache_key, reasoning_effort, service_tier, mode, side, messages
        if False:
            yield ""


def _summary_request_text(
    mode: ReasoningSummary,
    side: Side,
    messages: Sequence[StateMessage | ReasoningMessagePatch],
) -> str:
    payload = msgspec.json.encode(
        [message.to_primitive() for message in messages],
        order="deterministic",
    ).decode()
    return f"Summary mode: {mode}\nTrace perspective hint: {_summary_perspective_hint(side)}\n\nReasoning trace messages:\n{payload}"


def _summary_perspective_hint(side: Side) -> str:
    if side == "main":
        return "assistant self-check or draft trace"
    if side == "reviewer":
        return "critique of the assistant draft"
    return "private reconciliation of competing notes"


def _summary_max_tokens(mode: ReasoningSummary) -> int:
    if mode == "detailed":
        return 512
    return 192
