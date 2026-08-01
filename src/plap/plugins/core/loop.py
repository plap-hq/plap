"""Execute a response as a loop of main-agent turns.

One response runs one loop. Each loop iteration builds one request and runs one
turn. A turn wraps one retried main-model completion. Commit runs once after the
loop has stopped.
"""

from __future__ import annotations

from typing import NoReturn

import anyio
import structlog
from opentelemetry import trace

from plap.bus import bus
from plap.llms.accumulator import Snapshot
from plap.llms.completions.budget import BudgetedChatCompletionClient, CompletionBudget, CompletionBudgetExhaustedError
from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, ChatFinishReason, ChatUsage
from plap.llms.retry import RetryValidator, retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.llms.retry import stream as retry_stream
from plap.plugins.core.request import build_response_request
from plap.responses.contracts import ResponseUsage, ResponseUsageInputTokensDetails, ResponseUsageOutputTokensDetails
from plap.responses.state import State
from plap.responses.streaming import ResponseFinalizationError, StreamCoordinator
from plap.responses.summary import SummaryDelta, SummaryDone

logger = structlog.stdlib.get_logger(__name__)
tracer = trace.get_tracer(__name__)

SUMMARY_HARD_FLUSH_CHARS = 800


def _raise_task_error(exc: BaseExceptionGroup) -> NoReturn:
    if len(exc.exceptions) != 1:
        raise exc
    inner = exc.exceptions[0]
    if isinstance(inner, BaseExceptionGroup):
        _raise_task_error(inner)
    raise inner


@bus.emit("response.request")
async def response_request(state: State) -> ChatCompletionRequest:
    return build_response_request(state)


@bus.emit("response.snapshot")
async def response_snapshot(
    state: State,
    request: ChatCompletionRequest,
    snapshot: Snapshot,
) -> Snapshot:
    return snapshot


@bus.emit("response.completion")
async def response_completion(
    state: State,
    request: ChatCompletionRequest,
    validators: tuple[RetryValidator, ...],
) -> ChatCompletionResult:
    client = await state.svcs.aget(BudgetedChatCompletionClient)
    budget = state.svcs.get(CompletionBudget)
    main = state.threads["main"]
    suffix = len(main)
    result: ChatCompletionResult | None = None
    first_main_attempt_usage: ChatUsage | None = None

    logger.info("response.turn.started", model=request.model, tool_count=len(request.tools))

    summary_send, summary_receive = anyio.create_memory_object_stream[SummaryDelta | SummaryDone](32)

    async def run_summary() -> None:
        await bus.emit("response.summary", state=state, source=summary_receive)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_summary)
            try:
                async for raw_snapshot in retry_stream(client, request, validators=validators):
                    if first_main_attempt_usage is None:
                        first_main_attempt_usage = next(
                            (attempt.usage for attempt in raw_snapshot.results if attempt.usage is not None),
                            None,
                        )
                    snapshot = await response_snapshot(
                        state=state,
                        request=request,
                        snapshot=raw_snapshot,
                    )
                    main[suffix:] = snapshot.messages
                    if snapshot.result is not None:
                        result = snapshot.result
                    delta = snapshot.delta
                    if delta is not None and delta.reasoning_delta is not None:
                        await summary_send.send(SummaryDelta(text=delta.reasoning_delta))
                    if delta is None or delta.tool_call_delta is not None or delta.finish_reason is not None:
                        await summary_send.send(SummaryDone())
            finally:
                await summary_send.aclose()
    except BaseExceptionGroup as exc:
        _raise_task_error(exc)
    finally:
        budget.anchor(input_usage=first_main_attempt_usage)

    if result is None:
        raise RuntimeError("response stream ended without an accepted final result")

    return result


@bus.emit("response.turn")
async def response_turn(
    state: State,
    request: ChatCompletionRequest,
) -> ChatCompletionResult:
    return await response_completion(
        state=state,
        request=request,
        validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
    )


@bus.emit("response.loop")
async def response_loop(state: State) -> ChatCompletionResult | None:
    if "main" not in state.threads.active or state.open_calls("main"):
        return None

    while True:
        request = await response_request(state=state)
        result = await response_turn(state=state, request=request)
        main = state.threads["main"]
        if "main" not in state.threads.active or state.open_calls("main") or not main or main[-1].is_assistant():
            return result


@bus.emit("response.commit")
async def response_commit(state: State) -> None:
    await state.commit()


def _response_usage(usage: ChatUsage | None) -> ResponseUsage | None:
    if usage is None:
        return None
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=usage.cached_tokens or 0),
        output_tokens=usage.output_tokens,
        output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=usage.reasoning_tokens or 0),
        total_tokens=usage.total_tokens,
    )


@bus.emit("response.start")
async def run_response(state: State) -> None:
    coordinator = state.svcs.get(StreamCoordinator)
    budget = state.svcs.get(CompletionBudget)
    with tracer.start_as_current_span("response.execute") as span:
        span.set_attribute("plap.response.id", coordinator.response_id)
        span.set_attribute("plap.response.model", state.request.model)
        if state.request.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", state.request.conversation_id)
        result: ChatCompletionResult | None = None
        budget_exhausted = False
        try:
            result = await response_loop(state=state)
        except CompletionBudgetExhaustedError:
            budget_exhausted = True
            result = None
        try:
            with anyio.CancelScope(shield=True):
                usage = _response_usage(
                    budget.finish(
                        output_usage=None if result is None or budget_exhausted else result.usage,
                    )
                )
                await response_commit(state=state)
                if budget_exhausted:
                    await coordinator.incomplete(usage=usage)
                elif result is None:
                    await coordinator.completed(usage=usage)
                elif result.finish_reason == ChatFinishReason.LENGTH:
                    await coordinator.incomplete(usage=usage)
                else:
                    await coordinator.completed(usage=usage)
        except Exception as exc:
            raise ResponseFinalizationError("response finalization failed") from exc


@bus.emit("response.summary")
async def default_summary(
    state: State,
    source: anyio.abc.ObjectReceiveStream[SummaryDelta | SummaryDone],
) -> None:
    coordinator = state.svcs.get(StreamCoordinator)
    open_part = False
    accumulated = 0
    async for item in source:
        if isinstance(item, SummaryDelta):
            if not open_part:
                await state.ensure_progress()
                open_part = True
                await coordinator.summary_delta(SummaryDelta(text=item.text))
            accumulated += len(item.text)
            if accumulated >= SUMMARY_HARD_FLUSH_CHARS:
                await coordinator.summary_done(SummaryDone())
                await state.save_progress()
                open_part = False
                accumulated = 0
        elif isinstance(item, SummaryDone):
            if open_part:
                await coordinator.summary_done(SummaryDone())
                await state.save_progress()
                open_part = False
                accumulated = 0
    if open_part:
        await coordinator.summary_done(SummaryDone())
        await state.save_progress()
