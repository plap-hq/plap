from __future__ import annotations

import anyio
import structlog
from opentelemetry import trace

from plap.errors import PlapError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import IChatCompletionClient
from plap.logging import bound_context
from plap.responses.ingest.models import Ingested
from plap.responses.runner import State, execute
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.settings import RuntimeSelector, Settings
from plap.tools import IMCPToolProvider, IToolCallPolicyResolver, IToolPolicyResolver

logger = structlog.stdlib.get_logger(__name__)
tracer = trace.get_tracer(__name__)


def _selector_from_request(request) -> RuntimeSelector:
    reasoning = request.reasoning
    return RuntimeSelector(
        service_tier=request.service_tier,
        reasoning_effort=reasoning.effort if reasoning is not None else None,
    )


def _unexpected_public_error() -> PublicError:
    return PublicError(
        status_code=500,
        type="server_error",
        code="internal_error",
        message="An unexpected error occurred.",
    )


async def run_response(
    *,
    prepared: PreparedRequest,
    ingested: Ingested,
    coordinator: StreamCoordinator,
    sealing_keyring: SealingKeyring,
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
    with (
        bound_context(conversation_id=prepared.conversation_id, response_id=coordinator.response_id),
        tracer.start_as_current_span("response.execute") as span,
    ):
        span.set_attribute("plap.response.id", coordinator.response_id)
        span.set_attribute("plap.response.model", prepared.response_request.model)
        if prepared.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", prepared.conversation_id)
        created = False
        state: State | None = None
        try:
            await anyio.sleep(0)
            profile = settings.resolve_runtime_model_profile(
                prepared.response_request.model,
                selector=_selector_from_request(prepared.response_request),
            )
            await anyio.sleep(0)
            with anyio.CancelScope(shield=True):
                await coordinator.created()
            created = True
            await coordinator.in_progress()
            state = State.from_ingested(coordinator, sealing_keyring, ingested)
            await execute(
                state=state,
                prepared=prepared,
                profile=profile,
                chat_completion_client=chat_completion_client,
                tool_policy_resolver=tool_policy_resolver,
                tool_call_policy_resolver=tool_call_policy_resolver,
                mcp_tool_providers=mcp_tool_providers,
            )
        except anyio.get_cancelled_exc_class():
            if created:
                if state is not None:
                    with anyio.CancelScope(shield=True):
                        await state.flush()
                with anyio.CancelScope(shield=True):
                    await coordinator.cancelled()
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
                await coordinator.fail(public)
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
                await coordinator.fail(public)
            raise
