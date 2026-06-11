from __future__ import annotations

import logging as stdlib_logging
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import structlog

_LOG_KWARG_NAMES = ("exc_info", "stack_info", "stacklevel")


class _SplitLevelFilter(stdlib_logging.Filter):
    def __init__(self, *, app_level: int, foreign_level: int) -> None:
        super().__init__()
        self._app_level = app_level
        self._foreign_level = foreign_level

    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        threshold = self._app_level if record.name == "plap" or record.name.startswith("plap.") else self._foreign_level
        return record.levelno >= threshold


class _DefaultRecordAttributesFilter(stdlib_logging.Filter):
    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = record.getMessage()
        if not hasattr(record, "log_channel"):
            record.log_channel = "app"
        if not hasattr(record, "logger"):
            record.logger = record.name
        if not hasattr(record, "level"):
            record.level = record.levelname.lower()
        return True


def _add_default_log_channel(_logger: Any, _method_name: str, event_dict: structlog.typing.EventDict) -> structlog.typing.EventDict:
    event_dict.setdefault("log_channel", "app")
    return event_dict


def _render_to_log_kwargs(_logger: Any, _method_name: str, event_dict: structlog.typing.EventDict) -> dict[str, object]:
    extra = dict(event_dict)
    kwargs = {key: extra.pop(key) for key in _LOG_KWARG_NAMES if key in extra}
    event = extra.get("event")
    return {
        "msg": event,
        "extra": extra,
        **kwargs,
    }


def _parse_log_level(value: object, *, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a logging level name or number")
    normalized = value.strip().upper()
    if normalized.isdigit():
        return int(normalized)
    level = stdlib_logging.getLevelNamesMapping().get(normalized)
    if not isinstance(level, int):
        raise TypeError(f"{field_name} must be a valid logging level, got {value!r}")
    return level


def _processors() -> list[structlog.types.Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        _add_default_log_channel,
        structlog.processors.TimeStamper(fmt="iso"),
        _render_to_log_kwargs,
    ]


def configure_logging(settings: Any, *, handlers: tuple[stdlib_logging.Handler, ...]) -> None:
    app_level = _parse_log_level(getattr(settings, "log_level", "INFO"), field_name="log_level")
    foreign_level = _parse_log_level(getattr(settings, "foreign_log_level", "WARNING"), field_name="foreign_log_level")
    root_level = min(app_level, foreign_level)
    structlog.configure(
        cache_logger_on_first_use=False,
        logger_factory=structlog.stdlib.LoggerFactory(),
        processors=_processors(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

    root_logger = stdlib_logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(root_level)

    for handler in handlers:
        handler.setLevel(root_level)
        handler.addFilter(_SplitLevelFilter(app_level=app_level, foreign_level=foreign_level))
        handler.addFilter(_DefaultRecordAttributesFilter())
        root_logger.addHandler(handler)

    stdlib_logging.getLogger("uvicorn.access").setLevel(stdlib_logging.WARNING)


def bound_context(**context: object) -> AbstractContextManager[None]:
    filtered = {key: value for key, value in context.items() if value is not None}
    if not filtered:
        return nullcontext()
    return structlog.contextvars.bound_contextvars(**filtered)
