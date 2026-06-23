from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import anyio
import structlog
from opentelemetry import trace

from plap.bus import bus
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms import RetryLimitExceededError
from plap.llms.completions.chat import ChatCompletionRequest, ChatFinishReason, IChatCompletionClient
from plap.llms.retry import retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
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


def _unsupported_continuation_error(state: State) -> PlapError:
    if state.last_side is None:
        return PlapError(
            public=_unexpected_public_error(),
            private=PrivateError(
                event="response.continuation_side_missing",
                reason="continuation_side_missing",
                message="responses execution cannot continue because replay did not select a continuation side",
                level=ErrorLevel.WARNING,
            ),
        )
    return PlapError(
        public=_unexpected_public_error(),
        private=PrivateError(
            event="response.continuation_side_unsupported",
            reason="continuation_side_unsupported",
            message=f"responses execution has no handler for continuation side {state.last_side!r}",
            level=ErrorLevel.WARNING,
            context={"side": state.last_side},
        ),
    )


def _require_continued(
    state: State,
    continued: StreamResult | None,
) -> StreamResult:
    if continued is None:
        raise _unsupported_continuation_error(state)
    return continued


@dataclass(slots=True)
class StreamResult:
    ledger: UsageLedger
    latest_snapshot: object | None
    hidden_results_accounted: int
    input_anchor_seen: bool
    budget_exhausted: bool
    error: PlapError | None = None


@bus.emit("response.config")
async def resolve_config(state: State, request: dict[str, object]) -> CueBox:
    loaded = state.svcs.get(CueBox)
    return loaded.plap.config.resolve(request)


@bus.emit("response.request")
async def response_request(state: State, config: CueBox) -> ChatCompletionRequest:
    return build_response_request(state, config)


@bus.emit("response.stream")
async def stream_response(state: State, config: CueBox, request: ChatCompletionRequest) -> StreamResult:
    main = config.main
    ledger = UsageLedger(
        budget=state.prepared.execution_request.max_output_tokens,
        reasoning_to_output=config.reasoning_to_output,
    )
    hidden_results_accounted = 0
    input_anchor_seen = False
    budget_exhausted = False
    latest_snapshot = None
    chat_completion_client = await state.svcs.aget(IChatCompletionClient)

    logger.info(
        "response.runtime.turn",
        continuation_side="main",
        main_model=request.model,
        tool_count=len(request.tools),
    )

    def next_request(history):
        nonlocal hidden_results_accounted, input_anchor_seen, budget_exhausted
        for result in history.results[hidden_results_accounted:]:
            if not input_anchor_seen:
                ledger.set_input_anchor(result.usage)
                input_anchor_seen = True
            ledger.record_hidden(main.public_usage, result.usage)
            hidden_results_accounted += 1
        attempt_index = hidden_results_accounted + 1
        attempt_budget = ledger.budget_cap_for(main.public_usage)
        attempt_cap = ledger.completion_cap_for(main.public_usage, main)
        if attempt_cap == 0:
            budget_exhausted = True
            logger.info(
                "response.runtime.main_request.skipped",
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
            "response.runtime.main_request",
            attempt_budget=attempt_budget,
            attempt_index=attempt_index,
            hidden_history_messages=len(history.messages),
            hidden_history_results=len(history.results),
            main_cap=attempt_cap,
            remaining_budget=ledger.remaining(),
        )
        logger.bind(log_channel="payload").info(
            "response.runtime.main_request.payload",
            attempt_index=attempt_index,
            request=asdict(attempt_request),
        )
        return attempt_request

    source = retry_stream(
        chat_completion_client,
        next_request=next_request,
        validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
    )

    summary_send, summary_receive = anyio.create_memory_object_stream[SummaryDelta | SummaryDone](32)

    async def run_summary():
        await bus.emit("response.summary", state=state, source=summary_receive)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_summary)
            try:
                async for snapshot in source:
                    latest_snapshot = snapshot
                    state.main = list(snapshot.messages)

                    delta = snapshot.delta
                    if delta is not None and delta.reasoning_delta is not None:
                        await summary_send.send(SummaryDelta(text=delta.reasoning_delta, index=0))
                    if delta is None or (delta is not None and (delta.tool_call_delta is not None or delta.finish_reason is not None)):
                        await summary_send.send(SummaryDone(index=0))
            finally:
                await summary_send.aclose()

    except RetryLimitExceededError:
        if latest_snapshot is not None:
            state.main = list(latest_snapshot.messages)
        return StreamResult(
            ledger=ledger,
            latest_snapshot=latest_snapshot,
            hidden_results_accounted=hidden_results_accounted,
            input_anchor_seen=input_anchor_seen,
            budget_exhausted=budget_exhausted,
            error=_retry_limit_error(),
        )

    return StreamResult(
        ledger=ledger,
        latest_snapshot=latest_snapshot,
        hidden_results_accounted=hidden_results_accounted,
        input_anchor_seen=input_anchor_seen,
        budget_exhausted=budget_exhausted,
    )


@bus.emit("response.finalize")
async def finalize_response(state: State, config: CueBox, result: StreamResult) -> None:
    _ = config, result
    await state.finalize()


@bus.emit("response.finish")
async def finish_response(state: State, config: CueBox, result: StreamResult) -> None:
    if result.error is not None:
        raise result.error

    if result.latest_snapshot is None:
        if result.budget_exhausted:
            await state.coordinator.incomplete(usage=result.ledger.to_response_usage())
            return
        raise RuntimeError("response stream produced no snapshots")

    accepted_results = list(result.latest_snapshot.results[result.hidden_results_accounted :])
    if accepted_results:
        final_result = accepted_results[-1]
        if not result.input_anchor_seen:
            result.ledger.set_input_anchor(final_result.usage)
            result.input_anchor_seen = True
        result.ledger.record_output(config.main.public_usage, final_result.usage)
        usage = result.ledger.to_response_usage()
        if final_result.finish_reason == ChatFinishReason.LENGTH:
            await state.coordinator.incomplete(service_tier=final_result.service_tier, usage=usage)
            return
        await state.coordinator.completed(service_tier=final_result.service_tier, usage=usage)
        return

    if result.budget_exhausted:
        service_tier = result.latest_snapshot.results[-1].service_tier if result.latest_snapshot.results else None
        await state.coordinator.incomplete(service_tier=service_tier, usage=result.ledger.to_response_usage())
        return

    raise RuntimeError("response stream ended without an accepted final result")


@bus.emit("response.continue")
async def continue_response(state: State, config: CueBox) -> StreamResult | None:
    if state.last_side != MAIN_SIDE:
        return None
    request = await response_request(state=state, config=config)
    return await stream_response(state=state, config=config, request=request)


@bus.emit("response.run")
async def run_response(state: State) -> None:
    with tracer.start_as_current_span("response.execute") as span:
        span.set_attribute("plap.response.id", state.coordinator.response_id)
        span.set_attribute("plap.response.model", state.prepared.response_request.model)
        if state.prepared.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", state.prepared.conversation_id)
        created = False
        try:
            config = await resolve_config(state=state, request=build_config_request(state))
            await anyio.sleep(0)
            with anyio.CancelScope(shield=True):
                await state.coordinator.created()
            created = True
            await state.coordinator.in_progress()
            result = _require_continued(state, await continue_response(state=state, config=config))
            await finalize_response(state=state, config=config, result=result)
            await finish_response(state=state, config=config, result=result)
        except anyio.get_cancelled_exc_class():
            if created:
                with anyio.CancelScope(shield=True):
                    await state.flush()
                with anyio.CancelScope(shield=True):
                    await state.coordinator.cancelled()
            raise
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
    source: anyio.abc.ObjectReceiveStream[SummaryDelta | SummaryDone],
) -> None:
    index = 0
    open_part = False
    accumulated = 0
    async for item in source:
        if isinstance(item, SummaryDelta):
            if not open_part:
                await state.ensure_reasoning()
                open_part = True
            await state.coordinator.summary_delta(SummaryDelta(text=item.text, index=index))
            accumulated += len(item.text)
            if accumulated >= SUMMARY_HARD_FLUSH_CHARS:
                await state.coordinator.summary_done(SummaryDone(index=index))
                await state.flush()
                index += 1
                open_part = False
                accumulated = 0
        elif isinstance(item, SummaryDone):
            if open_part:
                await state.coordinator.summary_done(SummaryDone(index=index))
                await state.flush()
                index += 1
                open_part = False
                accumulated = 0
    if open_part:
        await state.coordinator.summary_done(SummaryDone(index=index))
        await state.flush()
