from __future__ import annotations

import logging as stdlib_logging
import os
import time
from dataclasses import dataclass

import structlog
from litestar.contrib.opentelemetry import OpenTelemetryConfig, OpenTelemetryPlugin
from litestar.types import ASGIApp, Message, Receive, Scope, Send
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

from plap.logging import bound_context

_ACCESS_LOGGER = structlog.stdlib.get_logger("plap.access")
_REQUEST_LOG_KEY = "plap_request_log"
_SERVICE_NAME = "plap"
_CLIENTS_INSTRUMENTED = False
_LOGGER_PROVIDER: LoggerProvider | None = None
_TRACER_PROVIDER: TracerProvider | None = None


@dataclass(frozen=True, slots=True)
class Telemetry:
    log_handler: stdlib_logging.Handler
    plugin: OpenTelemetryPlugin


@dataclass(slots=True)
class _RequestLog:
    started_at: float
    accepted: bool = False
    conversation_id: str | None = None
    logged: bool = False
    response_id: str | None = None
    status_code: int | None = None


def _configured_exporters(env_var: str, *fallback_env_vars: str) -> tuple[str, ...]:
    raw = os.environ.get(env_var, "")
    if raw:
        exporters = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
        if "none" in exporters:
            return ()
        return exporters
    if any(os.environ.get(name) for name in fallback_env_vars):
        return ("otlp",)
    return ()


def _configured_logs_exporters() -> tuple[str, ...]:
    return _configured_exporters(
        "OTEL_LOGS_EXPORTER",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    )


def _configured_trace_exporters() -> tuple[str, ...]:
    return _configured_exporters(
        "OTEL_TRACES_EXPORTER",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    )


def _resource() -> Resource:
    return Resource.create({SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", _SERVICE_NAME)})


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _instrument_clients() -> None:
    global _CLIENTS_INSTRUMENTED  # noqa: PLW0603

    if _CLIENTS_INSTRUMENTED:
        return
    HTTPXClientInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()
    _CLIENTS_INSTRUMENTED = True


def _trace_provider() -> TracerProvider:
    global _TRACER_PROVIDER  # noqa: PLW0603

    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        _TRACER_PROVIDER = provider
        return provider

    provider = TracerProvider(resource=_resource())
    for exporter in _configured_trace_exporters():
        if exporter == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        if exporter == "otlp":
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    return provider


def _log_provider() -> LoggerProvider:
    global _LOGGER_PROVIDER  # noqa: PLW0603

    if _LOGGER_PROVIDER is not None:
        return _LOGGER_PROVIDER

    provider = get_logger_provider()
    if isinstance(provider, LoggerProvider):
        _LOGGER_PROVIDER = provider
        return provider

    provider = LoggerProvider(resource=_resource())
    for exporter in _configured_logs_exporters():
        if exporter == "console":
            provider.add_log_record_processor(SimpleLogRecordProcessor(ConsoleLogExporter()))
        if exporter == "otlp":
            provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(provider)
    _LOGGER_PROVIDER = provider
    return provider


def build_telemetry() -> Telemetry:
    tracer_provider = _trace_provider()
    logger_provider = _log_provider()
    _instrument_clients()
    return Telemetry(
        log_handler=LoggingHandler(level=stdlib_logging.NOTSET, logger_provider=logger_provider, log_code_attributes=True),
        plugin=OpenTelemetryPlugin(OpenTelemetryConfig(exclude_spans=["receive", "send"], tracer_provider=tracer_provider)),
    )


def shutdown_telemetry() -> None:
    if _LOGGER_PROVIDER is not None:
        _LOGGER_PROVIDER.force_flush()
    if _TRACER_PROVIDER is not None:
        _TRACER_PROVIDER.force_flush()


def _scope_context(scope: Scope) -> dict[str, object]:
    scope_type = str(scope.get("type"))
    return {
        "method": scope.get("method"),
        "path": scope.get("path"),
        "transport": scope_type,
    }


def _should_observe_scope(scope_type: object) -> bool:
    return scope_type in {"http", "websocket"}


def _request_state(scope: Scope) -> _RequestLog | None:
    state = scope.setdefault("state", {})
    if not isinstance(state, dict):
        return None
    request_state = state.get(_REQUEST_LOG_KEY)
    if isinstance(request_state, _RequestLog):
        return request_state
    request_state = _RequestLog(started_at=time.perf_counter())
    state[_REQUEST_LOG_KEY] = request_state
    return request_state


def record_scope_context(scope: Scope, **context: object) -> None:
    request_state = _request_state(scope)
    if request_state is None:
        return
    response_id = context.get("response_id")
    if isinstance(response_id, str):
        request_state.response_id = response_id
    conversation_id = context.get("conversation_id")
    if isinstance(conversation_id, str):
        request_state.conversation_id = conversation_id


def _log_http_completion(request_state: _RequestLog) -> None:
    _ACCESS_LOGGER.info(
        "http.request.completed",
        conversation_id=request_state.conversation_id,
        duration_ms=_duration_ms(request_state.started_at),
        response_id=request_state.response_id,
        status_code=request_state.status_code,
    )


def _log_websocket_completion(request_state: _RequestLog) -> None:
    _ACCESS_LOGGER.info(
        "websocket.connection.completed",
        accepted=request_state.accepted,
        duration_ms=_duration_ms(request_state.started_at),
    )


def emit_access_log(message: Message, scope: Scope) -> None:
    if not _should_observe_scope(scope.get("type")):
        return

    request_state = _request_state(scope)
    if request_state is None or request_state.logged:
        return

    scope_type = scope.get("type")
    if scope_type == "http":
        message_type = message.get("type")
        if message_type == "http.response.start":
            status_code = message.get("status")
            if isinstance(status_code, int):
                request_state.status_code = status_code
            return
        if message_type == "http.response.body" and not bool(message.get("more_body", False)):
            _log_http_completion(request_state)
            request_state.logged = True
        return

    message_type = message.get("type")
    if message_type == "websocket.accept":
        request_state.accepted = True
        return
    if message_type == "websocket.close":
        _log_websocket_completion(request_state)
        request_state.logged = True


def request_context_middleware(app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if not _should_observe_scope(scope.get("type")):
            await app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        request_state = _request_state(scope)
        if request_state is None:
            await app(scope, receive, send)
            return
        request_state.accepted = False
        request_state.conversation_id = None
        request_state.logged = False
        request_state.response_id = None
        request_state.started_at = time.perf_counter()
        request_state.status_code = None
        with bound_context(**_scope_context(scope)):
            await app(scope, receive, send)

    return middleware
