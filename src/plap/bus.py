from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable
from typing import Any


def _check_next(fn: Callable) -> None:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    ok = any(p.name == "next" and p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)
    if not ok:
        raise TypeError(f"{fn.__name__} must accept keyword-only 'next'")


def _check_no_next(fn: Callable) -> tuple[str, ...]:
    sig = inspect.signature(fn)
    params = tuple(sig.parameters.keys())
    if "next" in params:
        raise TypeError(f"{fn.__name__} must not accept 'next'")
    return params


class _Handler:
    __slots__ = ("fn", "params")

    def __init__(self, fn: Callable, params: tuple[str, ...]) -> None:
        self.fn = fn
        self.params = params


def _run(handlers: list[_Handler], kwargs: dict[str, Any]):
    """Build and run the handler chain, returning a coroutine.

    The last registered handler runs first (outermost).  The first registered
    handler runs last (innermost).  This gives natural onion dispatch when
    bootstrap registers the terminal body first and plugins register wrappers
    later.
    """

    async def dispatch(index: int, current: dict[str, Any]):
        if index < 0:
            return None

        handler = handlers[index]
        resolved = {key: current[key] for key in handler.params if key in current}

        async def next(**overrides: Any):
            downstream = current if not overrides else {**current, **overrides}
            return await dispatch(index - 1, downstream)

        if "next" in handler.params:
            return await handler.fn(**resolved, next=next)

        return await handler.fn(**resolved)

    return dispatch(len(handlers) - 1, kwargs)


class _Emit:
    __slots__ = ("_handlers", "_name")

    def __init__(self, handlers: dict[str, list[_Handler]], name: str) -> None:
        self._handlers = handlers
        self._name = name

    def __call__(self, fn: object = None, **kwargs: Any) -> Any:
        if fn is not None and callable(fn):
            params = _check_no_next(fn)
            self._handlers[self._name].append(_Handler(fn, params))

            async def dispatch(**d: Any):
                return await _run(list(self._handlers[self._name]), d)

            return dispatch
        return _run(list(self._handlers.get(self._name, [])), kwargs)

    def __await__(self) -> Any:
        return _run(list(self._handlers.get(self._name, [])), {}).__await__()


class EventBus:
    """Lightweight signal bus with onion-style dispatch.

    Usage::

        bus = EventBus()

        @bus.listen("start")
        async def handler(state, *, next):
            ...

        @bus.emit("start")
        async def body(state):
            ...

        await bus.emit("start", state=...)
        await bus.emit("tick")
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[_Handler]] = defaultdict(list)

    def emit(self, name: str, **kwargs: Any) -> Any:
        emitter = _Emit(self._handlers, name)
        if kwargs:
            return emitter(**kwargs)
        return emitter

    def listen(self, name: str) -> Callable[[Callable], Callable]:
        def decorate(fn: Callable) -> Callable:
            _check_next(fn)
            params = tuple(inspect.signature(fn).parameters.keys())
            self._handlers[name].append(_Handler(fn, params))
            return fn

        return decorate

    def reset(self) -> None:
        self._handlers.clear()


bus = EventBus()
