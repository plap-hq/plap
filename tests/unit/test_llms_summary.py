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
    _SummaryChunker,
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


class _ClosingBlockingSummarizer(IReasoningSummarizer):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.closed = anyio.Event()

    async def stream(
        self,
        *,
        mode: str,
        prior_summary: str | None,
        fragment: str,
    ) -> AsyncIterator[str]:
        _ = mode, prior_summary, fragment
        self.started.set()
        try:
            await anyio.sleep_forever()
        finally:
            self.closed.set()
        if False:
            yield ""


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


def test_summary_chunker_waits_on_rule_lists_without_paragraph_breaks() -> None:
    buffer = "\n".join(
        (
            "Key points from the rules:",
            "- Output ONLY a thread title. Nothing else.",
            "- Single line, 50 characters or fewer.",
            "- No explanations.",
            "- Use the same language as the user message.",
            "- Never include tool names in the title.",
            "- Focus on the main topic or question the user needs to retrieve.",
            "- Keep exact technical terms, numbers, filenames, and HTTP codes.",
            "- Never use tools.",
            "- Always output something meaningful.",
        )
    )

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == ()
    assert chunker.buffer == buffer


def test_summary_chunker_prefers_paragraph_boundary_near_end() -> None:
    paragraph_1 = "I am checking the request constraints and comparing them with the current plan. " * 4
    paragraph_2 = "I am reviewing the main failure modes and narrowing the likely cause before changing code. " * 4
    paragraph_3 = "z" * 320
    buffer = f"{paragraph_1}\n\n{paragraph_2}\n\n{paragraph_3}"

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == (f"{paragraph_1}\n\n{paragraph_2}".strip(),)
    assert chunker.buffer == paragraph_3


def test_summary_chunker_hard_flushes_large_boundary_free_text() -> None:
    buffer = "x" * 900

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == (buffer,)
    assert chunker.buffer == ""


async def test_with_summary_forwards_snapshot_before_summary_finishes() -> None:
    summarizer = _BlockingSummarizer()

    async with with_summary(
        _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        first = await anext(stream)

        assert isinstance(first, Snapshot)
        with anyio.fail_after(1):
            await summarizer.started.wait()
        summarizer.release.set()

        rest = [item async for item in stream]

    assert rest == [SummaryDelta(index=0, text="summary"), SummaryDone(index=0)]


async def test_with_summary_streams_summary_delta_then_done() -> None:
    summarizer = _SequenceSummarizer([("checked ", "answer")])

    async with with_summary(
        _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        items = [item async for item in stream]

    assert isinstance(items[0], Snapshot)
    assert items[1:] == [
        SummaryDelta(index=0, text="checked "),
        SummaryDelta(index=0, text="answer"),
        SummaryDone(index=0),
    ]


async def test_with_summary_context_exit_cancels_background_tasks_cleanly() -> None:
    summarizer = _BlockingSummarizer()

    async with with_summary(
        _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        first = await anext(stream)

        assert isinstance(first, Snapshot)
        with anyio.fail_after(1):
            await summarizer.started.wait()


async def test_with_summary_consumer_task_cancellation_cleans_background_tasks() -> None:
    source_closed = anyio.Event()
    summarizer = _ClosingBlockingSummarizer()

    async def source() -> AsyncIterator[Snapshot]:
        try:
            yield _delta_snapshot(reasoning_delta="thinking", finish_reason="stop")
            await anyio.sleep_forever()
        finally:
            source_closed.set()

    async def consume() -> None:
        async with with_summary(
            source(),
            mode="concise",
            summarizer=summarizer,
        ) as stream:
            first = await anext(stream)
            assert isinstance(first, Snapshot)
            await anext(stream)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(consume)
        with anyio.fail_after(1):
            await summarizer.started.wait()
        task_group.cancel_scope.cancel()

    assert source_closed.is_set() is True
    assert summarizer.closed.is_set() is True


async def test_with_summary_flushes_on_tool_boundary() -> None:
    summarizer = _SequenceSummarizer([("checked",)])

    async with with_summary(
        _source(
            _delta_snapshot(reasoning_delta="thinking"),
            _delta_snapshot(tool_boundary=True),
        ),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        _ = [item async for item in stream]

    assert summarizer.calls == [("concise", None, "thinking")]


async def test_with_summary_flushes_on_retry_boundary() -> None:
    summarizer = _SequenceSummarizer([("checked",)])

    async with with_summary(
        _source(
            _delta_snapshot(reasoning_delta="thinking"),
            _retry_boundary_snapshot(),
        ),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        _ = [item async for item in stream]

    assert summarizer.calls == [("concise", None, "thinking")]


async def test_with_summary_carries_prior_summary_between_fragments() -> None:
    summarizer = _SequenceSummarizer([("first",), ("second",)])

    async with with_summary(
        _source(
            _delta_snapshot(reasoning_delta="one"),
            _delta_snapshot(tool_boundary=True),
            _delta_snapshot(reasoning_delta="two", finish_reason="stop"),
        ),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        _ = [item async for item in stream]

    assert summarizer.calls == [
        ("concise", None, "one"),
        ("concise", "first", "two"),
    ]


async def test_with_summary_skips_empty_summary_output() -> None:
    summarizer = _SequenceSummarizer([()])

    async with with_summary(
        _source(_delta_snapshot(reasoning_delta="thinking", finish_reason="stop")),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        items = [item async for item in stream]

    assert len(items) == 1
    assert isinstance(items[0], Snapshot)


async def test_with_summary_closes_partial_summary_after_error_and_carries_it_forward() -> None:
    summarizer = _SequenceSummarizer(
        [
            ("part", RuntimeError("boom")),
            ("next",),
        ]
    )

    async with with_summary(
        _source(
            _delta_snapshot(reasoning_delta="one"),
            _delta_snapshot(tool_boundary=True),
            _delta_snapshot(reasoning_delta="two", finish_reason="stop"),
        ),
        mode="concise",
        summarizer=summarizer,
    ) as stream:
        items = [item async for item in stream]

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

    async with with_summary(
        _source(final),
        mode="concise",
        summarizer=_SequenceSummarizer([("checked ", "answer")]),
    ) as stream:
        result = await collect_summary(stream)

    assert result == SummaryResult(snapshot=final, summaries=("checked answer",))
    assert result.text == "checked answer"


async def test_collect_summary_waits_for_summary_after_final_snapshot() -> None:
    final = _delta_snapshot(reasoning_delta="thinking", finish_reason="stop")
    summarizer = _BlockingSummarizer()

    async def run() -> SummaryResult:
        async with with_summary(
            _source(final),
            mode="concise",
            summarizer=summarizer,
        ) as stream:
            return await collect_summary(stream)

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
