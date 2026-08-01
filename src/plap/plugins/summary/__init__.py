from __future__ import annotations

from collections.abc import AsyncIterator

import anyio

from plap.bus import bus
from plap.llms.completions.budget import BudgetedChatCompletionClient
from plap.plugins.core.request import build_output_equivalence
from plap.plugins.easy import bootstrap
from plap.plugins.summary.summarizer import (
    ChatReasoningSummarizer,
    _append_summary,
    _summary_mode,
    _SummaryChunker,
)
from plap.responses.state import State
from plap.responses.summary import SummaryDelta, SummaryDone

bootstrap.config(__file__)


async def _emit_summary_fragment(
    *,
    mode: str,
    summarizer: ChatReasoningSummarizer,
    prior_summary: str | None,
    fragment: str,
    send: anyio.abc.ObjectSendStream[SummaryDelta | SummaryDone],
) -> str | None:
    full = ""
    async for text in summarizer.stream(mode=mode, prior_summary=prior_summary, fragment=fragment):
        if not text:
            continue
        full += text
        await send.send(SummaryDelta(text=text))

    if not full:
        return prior_summary

    await send.send(SummaryDone())
    return _append_summary(prior_summary, full)


async def _run_summarizer(
    mode: str,
    summarizer: ChatReasoningSummarizer,
    source: AsyncIterator[SummaryDelta | SummaryDone],
    send: anyio.abc.ObjectSendStream[SummaryDelta | SummaryDone],
) -> None:
    chunker = _SummaryChunker()
    prior_summary: str | None = None

    async with send:
        async for item in source:
            if isinstance(item, SummaryDelta):
                for fragment in chunker.push(item.text):
                    prior_summary = await _emit_summary_fragment(
                        mode=mode,
                        summarizer=summarizer,
                        prior_summary=prior_summary,
                        fragment=fragment,
                        send=send,
                    )
                continue

            if isinstance(item, SummaryDone):
                for fragment in chunker.flush():
                    prior_summary = await _emit_summary_fragment(
                        mode=mode,
                        summarizer=summarizer,
                        prior_summary=prior_summary,
                        fragment=fragment,
                        send=send,
                    )

        for fragment in chunker.finish():
            prior_summary = await _emit_summary_fragment(
                mode=mode,
                summarizer=summarizer,
                prior_summary=prior_summary,
                fragment=fragment,
                send=send,
            )


@bus.listen("response.summary")
async def wrap_summary(
    state: State,
    source: AsyncIterator[SummaryDelta | SummaryDone],
    *,
    next,
) -> None:
    mode = _summary_mode(state)
    if mode is None:
        return await next(state=state, source=source)

    summarizer = ChatReasoningSummarizer(
        client=await state.svcs.aget(BudgetedChatCompletionClient),
        model=state.config.summary.model,
        max_completion_tokens=state.config.summary.max_completion_tokens,
        prompt_cache_key=state.request.prompt_cache_key,
        reasoning_effort=state.config.summary.reasoning_effort,
        service_tier=state.config.summary.service_tier,
        output_equivalence=build_output_equivalence(state.config.summary),
    )

    downstream_send, downstream_receive = anyio.create_memory_object_stream[SummaryDelta | SummaryDone](32)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_run_summarizer, mode, summarizer, source, downstream_send)
        try:
            return await next(state=state, source=downstream_receive)
        finally:
            task_group.cancel_scope.cancel()
