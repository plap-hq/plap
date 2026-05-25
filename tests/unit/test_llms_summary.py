from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import anyio

from plap.llms.accumulator import Snapshot
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatToolCallDelta,
    IChatCompletionClient,
)
from plap.llms.summary import (
    ChatReasoningSummarizer,
    IReasoningSummarizer,
    SummaryDelta,
    SummaryDone,
    SummaryResult,
    collect_summary,
    with_summary,
)


def _delta_snapshot(
    *,
    reasoning_delta: str | None = None,
    tool_boundary: bool = False,
    finish_reason: str | None = None,
) -> Snapshot:
    return Snapshot(
        messages=(),
        results=(),
        delta=ChatCompletionDelta(
            id="chatcmpl_test",
            model="model-a",
            created_at=None,
            choice_index=0,
            reasoning_delta=reasoning_delta,
            tool_call_delta=ChatToolCallDelta(index=0) if tool_boundary else None,
            finish_reason=finish_reason,
        ),
    )


def _retry_boundary_snapshot() -> Snapshot:
    return Snapshot(messages=(), results=(), delta=None)


async def _source(*snapshots: Snapshot) -> AsyncIterator[Snapshot]:
    for snapshot in snapshots:
        yield snapshot


class _SequenceSummarizer(IReasoningSummarizer):
    def __init__(self, responses: Sequence[Sequence[object]]) -> None:
        self._responses = [tuple(response) for response in responses]
        self.calls: list[tuple[str, str | None, str]] = []

    async def stream(
        self,
        *,
        mode: str,
        prior_summary: str | None,
        fragment: str,
    ) -> AsyncIterator[str]:
        self.calls.append((mode, prior_summary, fragment))
        response = self._responses[len(self.calls) - 1]
        for item in response:
            if isinstance(item, Exception):
                raise item
            yield item


class _BlockingSummarizer(IReasoningSummarizer):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def stream(
        self,
        *,
        mode: str,
        prior_summary: str | None,
        fragment: str,
    ) -> AsyncIterator[str]:
        _ = mode, prior_summary, fragment
        self.started.set()
        await self.release.wait()
        yield "summary"


class _StreamingChatClient(IChatCompletionClient):
    def __init__(self, deltas: tuple[str, ...]) -> None:
        self._deltas = deltas
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest):
        _ = request
        raise NotImplementedError

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        self.requests.append(request)
        for delta in self._deltas:
            yield ChatCompletionDelta(
                id="chatcmpl_test",
                model=request.model,
                created_at=None,
                choice_index=0,
                content_delta=delta,
            )


async def test_with_summary_forwards_snapshot_before_summary_finishes() -> None:
    summarizer = _BlockingSummarizer()
    stream = with_summary(
        _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
        mode="concise",
        summarizer=summarizer,
    )

    first = await anext(stream)

    assert isinstance(first, Snapshot)
    with anyio.fail_after(1):
        await summarizer.started.wait()
    summarizer.release.set()

    rest = [item async for item in stream]

    assert rest == [SummaryDelta(index=0, text="summary"), SummaryDone(index=0)]


async def test_with_summary_streams_summary_delta_then_done() -> None:
    summarizer = _SequenceSummarizer([("checked ", "answer")])

    items = [
        item
        async for item in with_summary(
            _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    assert isinstance(items[0], Snapshot)
    assert items[1:] == [
        SummaryDelta(index=0, text="checked "),
        SummaryDelta(index=0, text="answer"),
        SummaryDone(index=0),
    ]


async def test_with_summary_flushes_on_tool_boundary() -> None:
    summarizer = _SequenceSummarizer([("checked",)])

    _ = [
        item
        async for item in with_summary(
            _source(
                _delta_snapshot(reasoning_delta="thinking"),
                _delta_snapshot(tool_boundary=True),
            ),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    assert summarizer.calls == [("concise", None, "thinking")]


async def test_with_summary_flushes_on_retry_boundary() -> None:
    summarizer = _SequenceSummarizer([("checked",)])

    _ = [
        item
        async for item in with_summary(
            _source(
                _delta_snapshot(reasoning_delta="thinking"),
                _retry_boundary_snapshot(),
            ),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    assert summarizer.calls == [("concise", None, "thinking")]


async def test_with_summary_carries_prior_summary_between_fragments() -> None:
    summarizer = _SequenceSummarizer([("first",), ("second",)])

    _ = [
        item
        async for item in with_summary(
            _source(
                _delta_snapshot(reasoning_delta="one"),
                _delta_snapshot(tool_boundary=True),
                _delta_snapshot(reasoning_delta="two", finish_reason="stop"),
            ),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    assert summarizer.calls == [
        ("concise", None, "one"),
        ("concise", "first", "two"),
    ]


async def test_with_summary_skips_empty_summary_output() -> None:
    summarizer = _SequenceSummarizer([()])

    items = [
        item
        async for item in with_summary(
            _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    assert len(items) == 1
    assert isinstance(items[0], Snapshot)


async def test_with_summary_closes_partial_summary_after_error_and_carries_it_forward() -> None:
    summarizer = _SequenceSummarizer(
        [
            ("part", RuntimeError("boom")),
            ("next",),
        ]
    )

    items = [
        item
        async for item in with_summary(
            _source(
                _delta_snapshot(reasoning_delta="one"),
                _delta_snapshot(tool_boundary=True),
                _delta_snapshot(reasoning_delta="two", finish_reason="stop"),
            ),
            mode="concise",
            summarizer=summarizer,
        )
    ]

    summary_items = [item for item in items if isinstance(item, (SummaryDelta, SummaryDone))]

    assert summary_items == [
        SummaryDelta(index=0, text="part"),
        SummaryDone(index=0),
        SummaryDelta(index=1, text="next"),
        SummaryDone(index=1),
    ]
    assert summarizer.calls[1] == ("concise", "part", "two")


async def test_chat_reasoning_summarizer_binds_request_config_once() -> None:
    client = _StreamingChatClient(("part",))
    summarizer = ChatReasoningSummarizer(
        client=client,
        model="model-a",
        prompt_cache_key="cache-a|reasoning_summarizer",
        reasoning_effort=None,
        service_tier=None,
    )

    deltas = [
        delta
        async for delta in summarizer.stream(
            mode="concise",
            prior_summary="I checked the goal.",
            fragment="I verified the tool choice.",
        )
    ]

    assert deltas == ["part"]
    request = client.requests[0]
    assert request.model == "model-a"
    assert request.max_completion_tokens == 512
    assert request.prompt_cache_key == "cache-a|reasoning_summarizer"
    assert request.temperature == 0
    assert request.messages[0].role == "developer"
    assert request.messages[0].content is not None
    assert request.messages[1].role == "user"
    assert "Summary mode: concise" in (request.messages[1].content or "")
    assert "I checked the goal." in (request.messages[1].content or "")
    assert "I verified the tool choice." in (request.messages[1].content or "")


async def test_collect_summary_returns_final_snapshot_and_joined_parts() -> None:
    final = _delta_snapshot(reasoning_delta="thinking", finish_reason="stop")

    result = await collect_summary(
        with_summary(
            _source(final),
            mode="concise",
            summarizer=_SequenceSummarizer([("checked ", "answer")]),
        )
    )

    assert result == SummaryResult(snapshot=final, summaries=("checked answer",))
    assert result.text == "checked answer"


async def test_collect_summary_waits_for_summary_after_final_snapshot() -> None:
    final = _delta_snapshot(reasoning_delta="thinking", finish_reason="stop")
    summarizer = _BlockingSummarizer()

    async def run() -> SummaryResult:
        return await collect_summary(
            with_summary(
                _source(final),
                mode="concise",
                summarizer=summarizer,
            )
        )

    async with anyio.create_task_group() as task_group:
        holder: dict[str, SummaryResult] = {}

        async def collect() -> None:
            holder["result"] = await run()

        task_group.start_soon(collect)
        with anyio.fail_after(1):
            await summarizer.started.wait()
        assert "result" not in holder
        summarizer.release.set()

    assert holder["result"].snapshot == final
    assert holder["result"].summaries == ("summary",)
