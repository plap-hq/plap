from __future__ import annotations

from typing import Any

import httpx
import msgspec

from plap.llms.completions.client import Call


def exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        cause = current.__cause__
        if cause is not None:
            current = cause
            continue
        context = current.__context__
        if context is not None and not current.__suppress_context__:
            current = context
            continue
        current = None
    return tuple(chain)


def request_body_bytes(body: object) -> int | None:
    try:
        return len(msgspec.json.encode(body))
    except Exception:
        return None


def timeout_phase(exc: BaseException) -> str | None:
    for current in exception_chain(exc):
        if isinstance(current, httpx.ConnectTimeout):
            return "connect"
        if isinstance(current, httpx.ReadTimeout):
            return "read"
        if isinstance(current, httpx.WriteTimeout):
            return "write"
        if isinstance(current, httpx.PoolTimeout):
            return "pool"
    return None


def timeout_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def timeout_config(timeout: object) -> dict[str, float | None] | None:
    if timeout is None:
        return None
    if isinstance(timeout, (int, float)):
        seconds = float(timeout)
        return {
            "connect": seconds,
            "read": seconds,
            "write": seconds,
            "pool": seconds,
        }
    return {
        "connect": timeout_value(getattr(timeout, "connect", None)),
        "read": timeout_value(getattr(timeout, "read", None)),
        "write": timeout_value(getattr(timeout, "write", None)),
        "pool": timeout_value(getattr(timeout, "pool", None)),
    }


def timeout_error_message(exc: BaseException) -> str:
    phase = timeout_phase(exc)
    if phase is None:
        return str(exc)
    return f"{exc} (phase: {phase})"


def log_transport_error(
    *,
    logger: Any,
    provider: str,
    base_url: str | None,
    client_max_retries: int | None,
    call: Call,
    exc: Exception,
    streaming: bool,
    should_log: bool,
    extra_context: dict[str, object] | None = None,
) -> None:
    if not should_log:
        return
    request = getattr(exc, "request", None)
    chain = exception_chain(exc)
    root_cause = chain[-1] if len(chain) > 1 else None
    messages = call.body.get("messages")
    tools = call.body.get("tools")
    context: dict[str, object] = {
        "provider": provider,
        "base_url": base_url,
        "stream": streaming,
        "request_model": call.request.model,
        "wire_model": call.body.get("model"),
        "sdk_error_type": type(exc).__name__,
        "sdk_error_message": str(exc),
        "timeout_phase": timeout_phase(exc),
        "cause_chain_types": [type(current).__name__ for current in chain],
        "root_cause_type": type(root_cause).__name__ if root_cause is not None else None,
        "root_cause_message": str(root_cause) if root_cause is not None else None,
        "request_method": getattr(request, "method", None),
        "request_url": str(request.url) if request is not None else None,
        "client_max_retries": client_max_retries,
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tool_count": len(tools) if isinstance(tools, list) else None,
        "request_body_bytes": request_body_bytes(call.body),
    }
    if extra_context is not None:
        context.update(extra_context)
    logger.warning("llm.provider.request_error", **context)


__all__ = [
    "exception_chain",
    "log_transport_error",
    "timeout_config",
    "timeout_error_message",
    "timeout_phase",
    "timeout_value",
]
