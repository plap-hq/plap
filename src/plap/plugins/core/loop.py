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
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms import RetryLimitExceededError
from plap.llms.accumulator import Snapshot
from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, ChatFinishReason, IChatCompletionClient
from plap.llms.retry import RetryValidator, retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.llms.retry import stream as retry_stream
from plap.plugins.core.budget import ResponseBudget, ResponseBudgetExhaustedError, budgeted
from plap.plugins.core.request import build_config_request, build_response_request
from plap.responses.state import State
from plap.responses.summary import SummaryDelta, SummaryDone

logger = structlog.stdlib.get_logger(__name__)
tracer = trace.get_tracer(__name__)

SUMMARY_HARD_FLUSH_CHARS = 800


def _retry_limit_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=500,
            type="server_error",
            code="internal_error",
            message="An unexpected error occurred.",
        ),
        private=PrivateError(
            event="response.retry_limit_exceeded",
            reason="retry_limit_exceeded",
            message="responses execution exhausted retry attempts after unusable tool calls",
            level=ErrorLevel.ERROR,
        ),
    )


def _raise_retry_limit_error() -> NoReturn:
    raise _retry_limit_error()


def _raise_task_error(exc: BaseExceptionGroup) -> NoReturn:
    if len(exc.exceptions) != 1:
        raise exc
    inner = exc.exceptions[0]
    if isinstance(inner, BaseExceptionGroup):
        _raise_task_error(inner)
    raise inner


def _unexpected_public_error() -> PublicError:
    return PublicError(
        status_code=500,
        type="server_error",
        code="internal_error",
        message="An unexpected error occurred.",
    )


async def resolve_config(state: State, request: dict[str, object]) -> CueBox:
    loaded = state.svcs.get(CueBox)
    return loaded.plap.config.resolve(request)


@bus.emit("response.request")
async def response_request(state: State, config: CueBox) -> ChatCompletionRequest:
    return build_response_request(state, config)


@bus.emit("response.snapshot")
async def response_snapshot(
    state: State,
    config: CueBox,
    request: ChatCompletionRequest,
    snapshot: Snapshot,
) -> Snapshot:
    return snapshot


@bus.emit("response.completion")
async def response_completion(
    state: State,
    config: CueBox,
    budget: ResponseBudget,
    request: ChatCompletionRequest,
    validators: tuple[RetryValidator, ...],
) -> ChatCompletionResult:
    raw_client = await state.svcs.aget(IChatCompletionClient)
    client = budgeted(raw_client, budget, config.main)
    main = state.threads["main"]
    suffix = len(main)
    result: ChatCompletionResult | None = None

    logger.info("response.turn.started", model=request.model, tool_count=len(request.tools))

    summary_send, summary_receive = anyio.create_memory_object_stream[SummaryDelta | SummaryDone](32)

    async def run_summary() -> None:
        await bus.emit("response.summary", state=state, config=config, budget=budget, source=summary_receive)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_summary)
            try:
                async for raw_snapshot in retry_stream(client, request, validators=validators):
                    snapshot = await response_snapshot(
                        state=state,
                        config=config,
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

    if result is None:
        raise RuntimeError("response stream ended without an accepted final result")

    return result


@bus.emit("response.turn")
async def response_turn(
    state: State,
    config: CueBox,
    budget: ResponseBudget,
    request: ChatCompletionRequest,
) -> ChatCompletionResult:
    return await response_completion(
        state=state,
        config=config,
        budget=budget,
        request=request,
        validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
    )


@bus.emit("response.loop")
async def response_loop(state: State, config: CueBox, budget: ResponseBudget) -> ChatCompletionResult | None:
    if "main" not in state.threads.active or state.open_calls("main"):
        return None

    while True:
        request = await response_request(state=state, config=config)
        result = await response_turn(state=state, config=config, budget=budget, request=request)
        main = state.threads["main"]
        if "main" not in state.threads.active or state.open_calls("main") or not main or main[-1].is_assistant():
            return result


@bus.emit("response.commit")
async def response_commit(state: State) -> None:
    await state.commit()


@bus.emit("response.start")
async def run_response(state: State) -> None:
    with tracer.start_as_current_span("response.execute") as span:
        span.set_attribute("plap.response.id", state.coordinator.response_id)
        span.set_attribute("plap.response.model", state.prepared.response_request.model)
        if state.prepared.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", state.prepared.conversation_id)
        created = False
        try:
            try:
                config = await resolve_config(state=state, request=build_config_request(state))
                budget = ResponseBudget(config, state.prepared.execution_request.max_output_tokens)
                await anyio.sleep(0)
                with anyio.CancelScope(shield=True):
                    await state.coordinator.created()
                created = True
                await state.coordinator.in_progress()
                result: ChatCompletionResult | None = None
                budget_exhausted: ResponseBudgetExhaustedError | None = None
                retry_exhausted = False
                try:
                    result = await response_loop(state=state, config=config, budget=budget)
                except ResponseBudgetExhaustedError as exc:
                    budget_exhausted = exc
                    result = None
                except RetryLimitExceededError:
                    retry_exhausted = True
                    result = None
            except anyio.get_cancelled_exc_class():
                if created:
                    with anyio.CancelScope(shield=True):
                        await state.save_progress()
                    with anyio.CancelScope(shield=True):
                        await state.coordinator.cancelled()
                raise
            with anyio.CancelScope(shield=True):
                if retry_exhausted:
                    await response_commit(state=state)
                    _raise_retry_limit_error()
                usage = budget.finish(None if result is None or budget_exhausted is not None else result.usage)
                await response_commit(state=state)
                if budget_exhausted is not None:
                    await state.coordinator.incomplete(
                        service_tier=budget_exhausted.last_service_tier,
                        usage=usage,
                    )
                elif result is None:
                    await state.coordinator.completed(service_tier=None, usage=usage)
                else:
                    if result.finish_reason == ChatFinishReason.LENGTH:
                        await state.coordinator.incomplete(service_tier=result.service_tier, usage=usage)
                    else:
                        await state.coordinator.completed(service_tier=result.service_tier, usage=usage)
        except PlapError as exc:
            public = exc.public or _unexpected_public_error()
            exc.log(
                logger,
                failure_code=public.code,
                failure_type=public.type,
                status_code=public.status_code,
            )
            with anyio.CancelScope(shield=True):
                await state.coordinator.fail(public)
            raise
        except Exception:
            public = _unexpected_public_error()
            logger.exception(
                "response.execute.failed",
                failure_code=public.code,
                failure_type=public.type,
                status_code=public.status_code,
            )
            with anyio.CancelScope(shield=True):
                await state.coordinator.fail(public)
            raise


@bus.emit("response.summary")
async def default_summary(
    state: State,
    config: CueBox,
    budget: ResponseBudget,
    source: anyio.abc.ObjectReceiveStream[SummaryDelta | SummaryDone],
) -> None:
    _ = config, budget
    open_part = False
    accumulated = 0
    async for item in source:
        if isinstance(item, SummaryDelta):
            if not open_part:
                await state.ensure_progress()
                open_part = True
            await state.coordinator.summary_delta(SummaryDelta(text=item.text))
            accumulated += len(item.text)
            if accumulated >= SUMMARY_HARD_FLUSH_CHARS:
                await state.coordinator.summary_done(SummaryDone())
                await state.save_progress()
                open_part = False
                accumulated = 0
        elif isinstance(item, SummaryDone):
            if open_part:
                await state.coordinator.summary_done(SummaryDone())
                await state.save_progress()
                open_part = False
                accumulated = 0
    if open_part:
        await state.coordinator.summary_done(SummaryDone())
        await state.save_progress()
