from __future__ import annotations

from collections.abc import AsyncIterator

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    IChatCompletionClient,
)
from plap.responses.reasoning import (
    REASONING_SUMMARY_PROMPT,
    LLMReasoningSummarizer,
)


async def test_reasoning_summarizer_sends_strict_prompt_and_trace_payload() -> None:
    client = _StreamingChatClient(("summary",))
    summarizer = LLMReasoningSummarizer(client)

    deltas = [
        delta
        async for delta in summarizer.stream(
            model="model-a",
            mode="concise",
            side="reviewer",
            messages=[
                {
                    "content_hash": "abc123",
                    "reasoning_content": "critic says side B is better",
                }
            ],
        )
    ]

    assert deltas == ["summary"]
    request = client.requests[0]
    assert request.model == "model-a"
    assert request.temperature == 0
    assert request.messages[0].role == "developer"
    assert request.messages[0].content == REASONING_SUMMARY_PROMPT
    assert request.messages[1].role == "user"
    assert "Summary mode: concise" in (request.messages[1].content or "")
    assert "Trace perspective hint: critique of the assistant draft" in (request.messages[1].content or "")
    assert "critic says side B is better" in (request.messages[1].content or "")


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
