from __future__ import annotations

from collections.abc import AsyncIterator

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    IChatCompletionClient,
)
from plap.responses.reasoning import (
    ASSISTANT_PRIVATE_REASONING_FRAGMENT_HEADER,
    PREVIOUS_PUBLIC_SUMMARY_HEADER,
    REASONING_SUMMARY_PART_PROMPT,
    REASONING_SUMMARY_SOURCE_INSTRUCTION,
    SUMMARY_MODE_HEADER,
    LLMReasoningSummarizer,
    ReasoningSummaryPartSource,
)


async def test_reasoning_summarizer_stream_part_uses_append_prompt() -> None:
    client = _StreamingChatClient(("part",))
    summarizer = LLMReasoningSummarizer(client)

    deltas = [
        delta
        async for delta in summarizer.stream_part(
            model="model-a",
            prompt_cache_key="cache-a|reasoning_summarizer",
            reasoning_effort=None,
            service_tier=None,
            mode="concise",
            source=ReasoningSummaryPartSource(
                prior_summary="I checked the goal.",
                reasoning_text="I verified the tool choice.",
            ),
        )
    ]

    assert deltas == ["part"]
    request = client.requests[0]
    assert request.model == "model-a"
    assert request.max_completion_tokens == 384
    assert request.prompt_cache_key == "cache-a|reasoning_summarizer"
    assert request.temperature == 0
    assert request.messages[0].role == "developer"
    assert request.messages[0].content == REASONING_SUMMARY_PART_PROMPT
    assert request.messages[1].role == "user"
    assert "I checked the goal." in (request.messages[1].content or "")
    assert "I verified the tool choice." in (request.messages[1].content or "")
    assert SUMMARY_MODE_HEADER in (request.messages[1].content or "")
    assert REASONING_SUMMARY_SOURCE_INSTRUCTION in (request.messages[1].content or "")
    assert PREVIOUS_PUBLIC_SUMMARY_HEADER in (request.messages[1].content or "")
    assert ASSISTANT_PRIVATE_REASONING_FRAGMENT_HEADER in (request.messages[1].content or "")


class _StreamingChatClient(IChatCompletionClient):
    def __init__(self, deltas: tuple[str, ...]) -> None:
        self.deltas = deltas
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest):
        _ = request
        raise NotImplementedError

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        self.requests.append(request)
        for delta in self.deltas:
            yield ChatCompletionDelta(
                choice_index=0,
                content_delta=delta,
                created_at=None,
                id="chatcmpl_test",
                model=request.model,
            )
