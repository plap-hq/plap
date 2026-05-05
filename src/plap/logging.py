from __future__ import annotations

import logging as stdlib_logging
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, TextIO

import structlog

_DEBUG_ENABLED = False
_PAYLOAD_DEBUG_ENABLED = False
_LOG_STREAM: TextIO | None = None


def configure_logging(settings: Any) -> None:
    global _DEBUG_ENABLED, _PAYLOAD_DEBUG_ENABLED, _LOG_STREAM  # noqa: PLW0603

    _DEBUG_ENABLED = bool(getattr(settings, "debug", False))
    _PAYLOAD_DEBUG_ENABLED = bool(getattr(settings, "debug_payloads", False))
    log_json = bool(getattr(settings, "log_json", False))
    level = stdlib_logging.DEBUG if _DEBUG_ENABLED else stdlib_logging.INFO

    if _LOG_STREAM is not None and _LOG_STREAM is not sys.stderr:
        _LOG_STREAM.close()
    _LOG_STREAM = _resolve_log_stream(getattr(settings, "log_file", None))

    renderer = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if log_json
        else structlog.dev.ConsoleRenderer(colors=_LOG_STREAM.isatty())
    )
    structlog.configure(
        cache_logger_on_first_use=True,
        logger_factory=structlog.PrintLoggerFactory(file=_LOG_STREAM),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def bound_context(**context: object) -> AbstractContextManager[None]:
    filtered = {key: value for key, value in context.items() if value is not None}
    if not filtered:
        return nullcontext()
    return structlog.contextvars.bound_contextvars(**filtered)


def _resolve_log_stream(log_file: str | None) -> TextIO:
    if not log_file:
        return sys.stderr
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8", buffering=1)


def log_debug(logger: structlog.typing.FilteringBoundLogger, event: str, /, **context: object) -> None:
    if _DEBUG_ENABLED:
        logger.debug(event, **context)


def log_payload(logger: structlog.typing.FilteringBoundLogger, event: str, /, **context: object) -> None:
    if _PAYLOAD_DEBUG_ENABLED:
        logger.debug(event, **context)
