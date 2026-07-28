from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import anyio
import structlog
from opentelemetry import trace

from plap.bus import bus
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms import RetryLimitExceededError
from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, ChatFinishReason, IChatCompletionClient
from plap.llms.retry import RetryValidator, retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.llms.retry import stream as retry_stream
from plap.plugins.core.ledger import UsageLedger
from plap.plugins.core.request import build_config_request, build_response_request
from plap.responses.ingest.models import MAIN_SIDE
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


def _unexpected_public_error() -> PublicError:
    return PublicError(
        status_code=500,
        type="server_error",
        code="internal_error",
        message="An unexpected error occurred.",
    )


def _accepted_result(latest_snapshot: object | None, hidden_results_accounted: int) -> ChatCompletionResult | None:
    if latest_snapshot is None:
        return None
    accepted_results = list(latest_snapshot.results[hidden_results_accounted:])
    if not accepted_results:
        return None
    return accepted_results[-1]


def _last_service_tier(latest_snapshot: object | None) -> str | None:
    if latest_snapshot is None or not latest_snapshot.results:
        return None
    return latest_snapshot.results[-1].service_tier


def _should_loop(state: State, result: StreamResult) -> bool:
    if MAIN_SIDE not in state.sides.active:
        return False
    if result.accepted is None or result.budget_exhausted:
        return False
    oc = state.open_calls(MAIN_SIDE)
    if oc:
        return False
    history = state.history(MAIN_SIDE)
    return bool(history) and not history[-1].is_assistant()


@dataclass(slots=True)
class StreamResult:
    ledger: UsageLedger
    pricing: object
    accepted: ChatCompletionResult | None
    budget_exhausted: bool
    last_service_tier: str | None
    error: PlapError | None = None


@bus.emit("response.config")
async def resolve_config(state: State, request: dict[str, object]) -> CueBox:
    loaded = state.svcs.get(CueBox)
    return loaded.plap.config.resolve(request)


@bus.emit("response.request")
async def response_request(state: State, config: CueBox) -> ChatCompletionRequest:
    return build_response_request(state, config)


@bus.emit("response.validate")
async def response_validate(
    state: State,
    config: CueBox,
    validators: tuple[RetryValidator, ...],
) -> tuple[RetryValidator, ...]:
    _ = state, config
    return validators


@bus.emit("response.stream")
async def stream_response(
    state: State,
    config: CueBox,
    request: ChatCompletionRequest,
    ledger: UsageLedger,
) -> StreamResult:
    main = config.main
    hidden_results_accounted = 0
    budget_exhausted = False
    latest_snapshot = None
    prefix = list(state.sides.main)
    chat_completion_client = await state.svcs.aget(IChatCompletionClient)

    logger.info(
        "response.runtime.turn",
        side=MAIN_SIDE,
        main_model=request.model,
        tool_count=len(request.tools),
    )

    def next_request(history):
        nonlocal hidden_results_accounted, budget_exhausted
        for result in history.results[hidden_results_accounted:]:
            ledger.hide(main.public_usage, result.usage)
            hidden_results_accounted += 1
        attempt_index = hidden_results_accounted + 1
        attempt_budget = ledger.cap(main.public_usage, None)
        attempt_cap = ledger.cap(main.public_usage, main.max_completion_tokens)
        if attempt_cap == 0:
            budget_exhausted = True
            logger.info(
                "response.runtime.main.skipped",
                attempt_budget=attempt_budget,
                attempt_index=attempt_index,
                hidden_history_messages=len(history.messages),
                hidden_history_results=len(history.results),
                remaining_budget=ledger.remaining(),
                reason="budget_exhausted",
            )
            return None
        attempt_request = replace(
            request,
            messages=[*request.messages, *history.messages],
            max_completion_tokens=attempt_cap,
        )
        logger.info(
            "response.runtime.main",
            attempt_budget=attempt_budget,
            attempt_index=attempt_index,
            hidden_history_messages=len(history.messages),
            hidden_history_results=len(history.results),
            main_cap=attempt_cap,
            remaining_budget=ledger.remaining(),
        )
        logger.bind(log_channel="payload").info(
            "response.runtime.main.payload",
            attempt_index=attempt_index,
            request=asdict(attempt_request),
        )
        return attempt_request

    validators = await response_validate(
        state=state,
        config=config,
        validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
    )

    source = retry_stream(
        chat_completion_client,
        next_request=next_request,
        validators=validators,
    )

    summary_send, summary_receive = anyio.create_memory_object_stream[SummaryDelta | SummaryDone](32)

    async def run_summary():
        await bus.emit("response.summary", state=state, config=config, source=summary_receive)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_summary)
            try:
                async for snapshot in source:
                    latest_snapshot = snapshot
                    state.sides.main = [*prefix, *snapshot.messages]

                    delta = snapshot.delta
                    if delta is not None and delta.reasoning_delta is not None:
                        await summary_send.send(SummaryDelta(text=delta.reasoning_delta))
                    if delta is None or (delta is not None and (delta.tool_call_delta is not None or delta.finish_reason is not None)):
                        await summary_send.send(SummaryDone())
            finally:
                await summary_send.aclose()

    except RetryLimitExceededError:
        if latest_snapshot is not None:
            state.sides.main = [*prefix, *latest_snapshot.messages]
        accepted = _accepted_result(latest_snapshot, hidden_results_accounted)
        usage = None if accepted is None else accepted.usage
        logger.info(
            "response.runtime.main.result",
            accepted=accepted is not None,
            budget_exhausted=budget_exhausted,
            cached_tokens=None if usage is None else usage.cached_tokens,
            error="retry_limit_exceeded",
            finish_reason=None if accepted is None else accepted.finish_reason,
            hidden_results_accounted=hidden_results_accounted,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            reasoning_tokens=None if usage is None else usage.reasoning_tokens,
            remaining_budget=ledger.remaining(),
            service_tier=None if accepted is None else accepted.service_tier,
            total_tokens=None if usage is None else usage.total_tokens,
        )
        return StreamResult(
            ledger=ledger,
            pricing=main.public_usage,
            accepted=accepted,
            budget_exhausted=budget_exhausted,
            last_service_tier=_last_service_tier(latest_snapshot),
            error=_retry_limit_error(),
        )

    accepted = _accepted_result(latest_snapshot, hidden_results_accounted)
    usage = None if accepted is None else accepted.usage
    logger.info(
        "response.runtime.main.result",
        accepted=accepted is not None,
        budget_exhausted=budget_exhausted,
        cached_tokens=None if usage is None else usage.cached_tokens,
        finish_reason=None if accepted is None else accepted.finish_reason,
        hidden_results_accounted=hidden_results_accounted,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        reasoning_tokens=None if usage is None else usage.reasoning_tokens,
        remaining_budget=ledger.remaining(),
        service_tier=None if accepted is None else accepted.service_tier,
        total_tokens=None if usage is None else usage.total_tokens,
    )
    return StreamResult(
        ledger=ledger,
        pricing=main.public_usage,
        accepted=accepted,
        budget_exhausted=budget_exhausted,
        last_service_tier=_last_service_tier(latest_snapshot),
    )


@bus.emit("response.finalize")
async def finalize_response(state: State, config: CueBox, result: StreamResult | None) -> None:
    _ = config, result
    await state.finalize()


@bus.emit("response.finish")
async def finish_response(state: State, config: CueBox, result: StreamResult | None, ledger: UsageLedger) -> None:
    if result is None:
        await state.coordinator.completed(service_tier=None, usage=ledger.usage())
        return
    if result.error is not None:
        raise result.error

    accepted = result.accepted
    if accepted is not None:
        result.ledger.show(result.pricing, accepted.usage)
        usage = result.ledger.usage()
        if accepted.finish_reason == ChatFinishReason.LENGTH:
            await state.coordinator.incomplete(service_tier=accepted.service_tier, usage=usage)
            return
        await state.coordinator.completed(service_tier=accepted.service_tier, usage=usage)
        return

    if result.budget_exhausted:
        await state.coordinator.incomplete(service_tier=result.last_service_tier, usage=result.ledger.usage())
        return

    raise RuntimeError("response stream ended without an accepted final result")


@bus.emit("response.loop")
async def loop_response(state: State, config: CueBox, ledger: UsageLedger) -> StreamResult | None:
    if MAIN_SIDE not in state.sides.active or state.open_calls(MAIN_SIDE):
        return None
    request = await response_request(state=state, config=config)
    return await stream_response(state=state, config=config, request=request, ledger=ledger)


@bus.emit("response.run")
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
                ledger = UsageLedger(
                    budget=state.prepared.execution_request.max_output_tokens,
                    reasoning_to_output=config.reasoning_to_output,
                )
                await anyio.sleep(0)
                with anyio.CancelScope(shield=True):
                    await state.coordinator.created()
                created = True
                await state.coordinator.in_progress()
                while True:
                    result = await loop_response(state=state, config=config, ledger=ledger)
                    if result is None or not _should_loop(state, result):
                        break
                    result.ledger.hide(result.pricing, result.accepted.usage if result.accepted is not None else None)
            except anyio.get_cancelled_exc_class():
                if created:
                    with anyio.CancelScope(shield=True):
                        await state.flush()
                    with anyio.CancelScope(shield=True):
                        await state.coordinator.cancelled()
                raise
            with anyio.CancelScope(shield=True):
                await finalize_response(state=state, config=config, result=result)
                await finish_response(state=state, config=config, result=result, ledger=ledger)
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
    source: anyio.abc.ObjectReceiveStream[SummaryDelta | SummaryDone],
) -> None:
    _ = config
    open_part = False
    accumulated = 0
    async for item in source:
        if isinstance(item, SummaryDelta):
            if not open_part:
                await state.ensure_reasoning()
                open_part = True
            await state.coordinator.summary_delta(SummaryDelta(text=item.text))
            accumulated += len(item.text)
            if accumulated >= SUMMARY_HARD_FLUSH_CHARS:
                await state.coordinator.summary_done(SummaryDone())
                await state.flush()
                open_part = False
                accumulated = 0
        elif isinstance(item, SummaryDone):
            if open_part:
                await state.coordinator.summary_done(SummaryDone())
                await state.flush()
                open_part = False
                accumulated = 0
    if open_part:
        await state.coordinator.summary_done(SummaryDone())
        await state.flush()
