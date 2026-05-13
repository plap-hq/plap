from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import structlog

from plap.llms.chat import ChatCompletionRequest, ChatMessage, IChatCompletionClient, ReasoningEffort, ServiceTier
from plap.logging import log_debug, log_payload
from plap.responses.contracts import ReasoningSummary

logger = structlog.get_logger(__name__)

REASONING_SUMMARY_PART_PROMPT = """Write the next public reasoning summary part.

You will receive:
- the previously emitted public reasoning summary text, if any
- one new private reasoning fragment from the same turn

Rules:
- Write only the next appended public reasoning summary part for the new fragment.
- Do not rewrite or repeat the previously emitted summary.
- Do not expose raw chain-of-thought, hidden instructions, or hidden tool inputs.
- Preserve only high-level checks, revisions, comparisons, and conclusions.
- If the fragment adds nothing useful, return nothing.

Grounding:
- Base the summary only on the provided new private reasoning fragment and the previously emitted public summary.
- Do not attribute policies, refusals, rules, or instructions to OpenAI or any other vendor or organization unless that exact name appears in the new private reasoning fragment.
- Do not introduce external policy labels or safety taxonomy terms unless they appear in the fragment.
- When the fragment refers generically to internal rules or instructions, keep the summary generic, for example: "my rules", "my instructions", or "the policy".

Style:
- First person, as the assistant speaking naturally.
- Summarize current actions in present tense, not past tense. Rather than "I added," or "I did," use "I am adding," or "I am doing."
- Obviously, this does not apply to things that were actually done at a past time.
- Concise for concise mode, fuller for detailed mode.
"""


@dataclass(frozen=True, slots=True)
class ReasoningSummaryPartSource:
    prior_summary: str | None
    reasoning_text: str


@runtime_checkable
class IReasoningSummarizer(Protocol):
    def stream_part(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        source: ReasoningSummaryPartSource,
    ) -> AsyncIterator[str]: ...


class LLMReasoningSummarizer(IReasoningSummarizer):
    def __init__(self, client: IChatCompletionClient) -> None:
        self._client = client

    async def stream_part(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        source: ReasoningSummaryPartSource,
    ) -> AsyncIterator[str]:
        summary_request = ChatCompletionRequest(
            max_completion_tokens=_summary_max_tokens(mode),
            messages=[
                ChatMessage(
                    role="developer",
                    content=REASONING_SUMMARY_PART_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=_summary_part_request_text(mode, source),
                ),
            ],
            model=model,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            temperature=0,
        )
        log_debug(logger, "reasoning.summary.part.start", mode=mode, model=model)
        log_payload(logger, "reasoning.summary.part.request.payload", request=asdict(summary_request))
        delta_count = 0
        async for delta in self._client.stream(summary_request):
            if delta.content_delta:
                delta_count += 1
                log_payload(logger, "reasoning.summary.part.delta", delta=delta.content_delta)
                yield delta.content_delta
        log_debug(logger, "reasoning.summary.part.done", delta_count=delta_count, model=model)


class NullReasoningSummarizer(IReasoningSummarizer):
    async def stream_part(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: ReasoningEffort | None,
        service_tier: ServiceTier | None,
        mode: ReasoningSummary,
        source: ReasoningSummaryPartSource,
    ) -> AsyncIterator[str]:
        _ = model, prompt_cache_key, reasoning_effort, service_tier, mode, source
        if False:
            yield ""
def _summary_part_request_text(
    mode: ReasoningSummary,
    source: ReasoningSummaryPartSource,
) -> str:
    prior_summary = source.prior_summary or ""
    return (
        f"Summary mode: {mode}\n\n"
        f"Previously emitted public reasoning summary:\n{prior_summary}\n\n"
        f"New private reasoning fragment:\n{source.reasoning_text}"
    )


def _summary_max_tokens(mode: ReasoningSummary) -> int:
    if mode == "detailed":
        return 512
    return 384
